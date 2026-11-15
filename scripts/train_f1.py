#!/usr/bin/env python3
"""Train F1, the fine-tuned dual encoder (docs/f1-dual-encoder.md), on the data built by
scripts/f1_data.py. Standalone: needs only torch and transformers, so it runs unchanged
on Colab (docs/f1-colab-run.md) or anywhere else.

    python train_f1.py --data f1_data.tar.gz --out runs --plan        # all five runs
    python train_f1.py --data f1_data --out runs --run F1@3k          # one run
    python train_f1.py --data f1_data_smoke --out runs --run smoke --smoke   # Mac CPU test

Every run writes runs/<name with @ as ->/: train_log.jsonl (loss every 20 steps, each
evaluation), best/ (the HF checkpoint with the best held-out nDCG@5, plus f1_meta.json),
last/state.pt while running (resume point, removed at the end) and done.json. A rerun
skips finished runs and resumes an interrupted one from its last evaluation.

Loss (§4): masked, graded multi-positive InfoNCE over the batch's icons (each query's
target spread over every in-batch icon the judge graded for it, ∝ grade; the judge's
relevant-but-ungraded icons and free-signal look-alikes masked), plus an icon→query term
with each icon's own near-misses as hard negative queries.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE_MODEL = "intfloat/multilingual-e5-small"
# name -> (pool tasks (None = all), hold out 5% of icons). The method itself first, so a
# short Colab session still produces it.
PLAN = {
    "F1": (None, False),
    "F1@all": (None, True),
    "F1@0": (0, True),
    "F1@1k": (1000, True),
    "F1@3k": (3000, True),
}
NEG = -1e4      # masked logit (fp32)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --- data ----------------------------------------------------------------------------------

def open_data(path: Path) -> Path:
    """A data dir, or a .tar.gz that is unpacked next to itself (once)."""
    if path.is_dir():
        return path
    name = path.name.replace(".tar.gz", "")
    out = path.parent / name
    if not (out / "manifest.json").exists():
        with tarfile.open(path) as t:
            t.extractall(path.parent)
    return out


def read_jsonl(p: Path) -> list:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


class Data:
    def __init__(self, root: Path, pool_tasks, holdout: bool, smoke: bool):
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.icons = read_jsonl(root / "icons.jsonl")
        self.docs = [self.manifest["passage_prefix"] + r["doc"] for r in self.icons]
        self.holdout_all = {i for i, r in enumerate(self.icons) if r["holdout"]}
        self.held = self.holdout_all if holdout else set()
        qs = read_jsonl(root / "queries.jsonl")
        for q in qs:
            q["grades"] = {int(k): v for k, v in q["grades"].items()}
            q["ambiguous"] = set(q["ambiguous"])
            q["pos"] = [i for i in q["grades"] if i not in self.held]
            q["negs"] = [i for i in q["hard_negs"] if i not in self.held]
        self.val = [q for q in qs if q["split"] == "val"]
        pool = [q for q in qs if q["split"] == "train" and q["source"].startswith("pool")]
        keep = self._select_tasks(pool, pool_tasks)
        self.pool = [q for q in pool if q["task_id"] in keep and q["pos"]]
        self.free = [q for q in qs if q["source"].startswith("free") and q["pos"]]
        self.pool_task_count = len(keep)
        if smoke:
            self.val = self.val[:200]

    @staticmethod
    def _select_tasks(pool: list, n) -> set:
        """n train tasks, proportional per source, in a fixed hash order (all if None)."""
        by_src: dict = {}
        for q in pool:
            by_src.setdefault(q["source"], set()).add(q["task_id"])
        if n is None:
            return set().union(*by_src.values()) if by_src else set()
        total = sum(len(v) for v in by_src.values())
        keep = set()
        for src, ids in sorted(by_src.items()):
            k = round(n * len(ids) / total) if total else 0
            keep.update(sorted(ids, key=lambda t: _h("f1-curve:" + t))[:k])
        return keep

    def epoch_rows(self, epoch: int, seed: int, free_per_epoch: int) -> list:
        rng = random.Random(seed * 1000 + epoch)
        rows = list(self.pool) + rng.sample(self.free, min(free_per_epoch, len(self.free)))
        rng.shuffle(rows)
        return rows


# --- model ---------------------------------------------------------------------------------

class Encoder:
    """Mean-pooled, L2-normalized e5 (as tcr/finetune.py's matcher encodes)."""

    def __init__(self, path: str, device: str, train_embeddings: bool, checkpointing: bool):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device)
        if not train_embeddings:
            self.model.embeddings.word_embeddings.weight.requires_grad_(False)
        if checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.amp = device == "cuda"

    def __call__(self, texts, max_len: int) -> torch.Tensor:
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=max_len,
                       return_tensors="pt").to(self.device)
        ctx = torch.autocast("cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext()
        with ctx:
            h = self.model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(v.float(), dim=-1)

    @torch.no_grad()
    def encode(self, texts, max_len: int, batch: int = 128) -> torch.Tensor:
        was = self.model.training
        self.model.eval()
        out = [self(texts[s:s + batch], max_len) for s in range(0, len(texts), batch)]
        self.model.train(was)
        return torch.cat(out)


# --- loss ----------------------------------------------------------------------------------

def batch_loss(enc: Encoder, data: Data, rows: list, rng: random.Random, a) -> tuple:
    man = data.manifest
    pos = [rng.choices(q["pos"], weights=[q["grades"][i] ** 2 for i in q["pos"]])[0] for q in rows]
    neg = [rng.choice(q["negs"]) if q["negs"] else None for q in rows]
    icons = sorted(set(pos) | {n for n in neg if n is not None})
    col = {ic: j for j, ic in enumerate(icons)}
    texts = [q["text_parent"] if q.get("text_parent") and rng.random() < a.parent_p else q["text"]
             for q in rows]
    Q = enc([man["query_prefix"] + t for t in texts], man["max_query_len"])
    P = enc([data.docs[i] for i in icons], a.max_passage_len)
    B, M = len(rows), len(icons)
    G = torch.zeros(B, M)
    A = torch.zeros(B, M, dtype=torch.bool)
    for r, q in enumerate(rows):
        for ic, j in col.items():
            if ic in q["grades"] and ic not in data.held:
                G[r, j] = q["grades"][ic]
            elif ic in q["ambiguous"]:
                A[r, j] = True
    G, A = G.to(Q.device), A.to(Q.device)
    S = a.scale * Q @ P.T
    T = G / G.sum(1, keepdim=True)
    loss_q = -(T * F.log_softmax(S.masked_fill(A, NEG), dim=1)).sum(1).mean()

    # icon -> query: rows = icons some query is graded for; own near-miss as an extra column
    rows_i = (G.sum(0) > 0).nonzero().flatten().tolist()
    loss_i = torch.zeros((), device=Q.device)
    if a.i2q_weight > 0 and rows_i:
        nm_text, has_nm = [], []
        for j in rows_i:
            nms = data.icons[icons[j]]["near_misses"]
            has_nm.append(bool(nms))
            if nms:
                nm_text.append(man["query_prefix"] + rng.choice(nms))
        S2 = a.scale * P[rows_i] @ Q.T                                  # (R, B)
        S2 = S2.masked_fill(A[:, rows_i].T, NEG)
        extra = torch.full((len(rows_i), 1), NEG, device=Q.device)
        if nm_text:
            N = enc(nm_text, man["max_query_len"])
            k = torch.tensor([i for i, h in enumerate(has_nm) if h], device=Q.device)
            extra[k, 0] = a.scale * (P[rows_i][k] * N).sum(1)
        S2 = torch.cat([S2, extra], dim=1)
        T2 = torch.cat([G[:, rows_i].T, torch.zeros(len(rows_i), 1, device=Q.device)], dim=1)
        T2 = T2 / T2.sum(1, keepdim=True)
        loss_i = -(T2 * F.log_softmax(S2, dim=1)).sum(1).mean()
    return loss_q, loss_i, M


# --- evaluation ----------------------------------------------------------------------------

def evaluate(enc: Encoder, data: Data, max_passage_len: int) -> dict:
    man = data.manifest
    I = enc.encode(data.docs, max_passage_len)
    Qv = enc.encode([man["query_prefix"] + q["text"] for q in data.val], man["max_query_len"])
    top = (Qv @ I.T).topk(10, dim=1).indices.cpu().numpy()
    disc = 1 / np.log2(np.arange(2, 7))
    per = {"all": [], "en": [], "es+ru": []}
    by_src: dict = {}
    unseen = []
    for q, t in zip(data.val, top):
        g = q["grades"]
        dcg = sum(g.get(int(i), 0) * d for i, d in zip(t[:5], disc))
        ideal = sorted(g.values(), reverse=True)[:5]
        idcg = sum(x * d for x, d in zip(ideal, disc))
        m = (dcg / idcg if idcg else 0.0, int(g.get(int(t[0]), 0) >= 2))
        per["all"].append(m)
        per["en" if q["lang"] == "en" else "es+ru"].append(m)
        by_src.setdefault(q["source"], []).append(m)
        held3 = {i for i, v in g.items() if v == 3 and i in data.holdout_all}
        if held3:
            unseen.append(int(bool(held3 & {int(i) for i in t})))
    out = {}
    for k, v in list(per.items()) + list(by_src.items()):
        if v:
            out[k] = {"n": len(v), "ndcg5": round(float(np.mean([x[0] for x in v])), 4),
                      "hit1": round(float(np.mean([x[1] for x in v])), 4)}
    out["unseen_icon_recall10"] = {"n": len(unseen), "value": round(float(np.mean(unseen)), 4) if unseen else None}
    return out


# --- training ------------------------------------------------------------------------------

def pick_device(want: str) -> str:
    if want != "auto":
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"   # MPS only when asked: see §9


def train(name: str, pool_tasks, holdout: bool, a, data_root: Path) -> None:
    out = Path(a.out) / name.replace("@", "-")
    if (out / "done.json").exists():
        log(f"{name}: done already, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    data = Data(data_root, pool_tasks, holdout, a.smoke)
    log(f"{name}: device {dev}; {data.pool_task_count} pool tasks ({len(data.pool)} rows), "
        f"{len(data.free)} free-signal rows ({a.free_per_epoch} per epoch), {len(data.val)} val, "
        f"{len(data.held)} icons held out")
    torch.manual_seed(a.seed)
    enc = Encoder(BASE_MODEL, dev, a.train_embeddings, not a.no_checkpointing)
    params = [p for p in enc.model.parameters() if p.requires_grad]
    decay = [p for n, p in enc.model.named_parameters() if p.requires_grad and p.ndim > 1]
    nodecay = [p for n, p in enc.model.named_parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr)
    rows0 = data.epoch_rows(0, a.seed, a.free_per_epoch)
    steps_per_epoch = max(1, len(rows0) // a.batch)
    total = a.max_steps or steps_per_epoch * a.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    seg = max(1, steps_per_epoch // a.evals_per_epoch)

    st = {"step": 0, "best": -1.0, "bad": 0, "history": [], "t_train": 0.0}
    last = out / "last" / "state.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        enc.model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        st = ck["st"]
        log(f"{name}: resumed at step {st['step']}")
    logf = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def write(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    def do_eval(tag):
        t0 = time.time()
        m = evaluate(enc, data, a.max_passage_len)
        rec = {"eval": tag, "step": st["step"], "epoch": round(st["step"] / steps_per_epoch, 2),
               "sec": round(time.time() - t0, 1), **m}
        write(rec)
        log(f"{name} eval {tag} step {st['step']}: nDCG@5 {m['all']['ndcg5']:.4f} hit@1 "
            f"{m['all']['hit1']:.4f} (en {m.get('en', {}).get('ndcg5')}, es+ru "
            f"{m.get('es+ru', {}).get('ndcg5')}); unseen-icon R@10 {m['unseen_icon_recall10']['value']}")
        return m

    if st["step"] == 0 and not st["history"]:
        base = do_eval("base")
        st["history"].append({"step": 0, **base})
        st["base"] = base

    t_start = time.time()
    stop = False
    while st["step"] < total and not stop:
        epoch = st["step"] // steps_per_epoch
        rows = data.epoch_rows(epoch, a.seed, a.free_per_epoch)
        in_epoch = st["step"] % steps_per_epoch
        enc.model.train()
        t_log, lq_acc, li_acc, m_acc, n_acc = time.time(), 0.0, 0.0, 0, 0
        for b in range(in_epoch, steps_per_epoch):
            batch = rows[b * a.batch:(b + 1) * a.batch]
            rng = random.Random(f"{a.seed}:{st['step']}")
            loss_q, loss_i, M = batch_loss(enc, data, batch, rng, a)
            loss = loss_q + a.i2q_weight * loss_i
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            st["step"] += 1
            lq_acc += loss_q.item(); li_acc += loss_i.item(); m_acc += M; n_acc += 1
            if st["step"] % 20 == 0 or a.smoke:
                dt = (time.time() - t_log) / n_acc
                write({"step": st["step"], "loss_q": round(lq_acc / n_acc, 4),
                       "loss_i": round(li_acc / n_acc, 4), "icons_per_batch": m_acc // n_acc,
                       "lr": sched.get_last_lr()[0], "sec_per_step": round(dt, 3)})
                eta = (total - st["step"]) * dt / 60
                log(f"{name} step {st['step']}/{total} loss {lq_acc / n_acc:.3f} + {li_acc / n_acc:.3f} "
                    f"({dt:.2f} s/step, ≤ {eta:.0f} min left)")
                t_log, lq_acc, li_acc, m_acc, n_acc = time.time(), 0.0, 0.0, 0, 0
            seg_end = (st["step"] % steps_per_epoch) % seg == 0 or st["step"] % steps_per_epoch == 0
            minutes = (time.time() - t_start) / 60
            if a.max_minutes and minutes > a.max_minutes:
                log(f"{name}: --max-minutes reached")
                stop = True
            if seg_end or st["step"] >= total or stop:
                m = do_eval("val")
                st["history"].append({"step": st["step"], **m})
                score = m["all"]["ndcg5"]
                if score > st["best"]:
                    st["best"], st["bad"] = score, 0
                    save_best(enc, out / "best", name, a, data, st, m)
                else:
                    st["bad"] += 1
                (out / "last").mkdir(exist_ok=True)
                torch.save({"model": enc.model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "st": st}, last)
                if st["bad"] >= a.patience:
                    log(f"{name}: no improvement in {a.patience} evaluations, stopping")
                    stop = True
            if stop or st["step"] >= total:
                break
    if stop and a.max_minutes and st["bad"] < a.patience and st["step"] < total:
        log(f"{name}: paused at step {st['step']}; rerun the same command to resume")
        return
    best = max(st["history"][1:], key=lambda h: h["all"]["ndcg5"]) if len(st["history"]) > 1 else None
    (out / "done.json").write_text(json.dumps({
        "run": name, "pool_tasks": data.pool_task_count, "holdout_icons": len(data.held),
        "steps": st["step"], "steps_per_epoch": steps_per_epoch, "base": st.get("base"), "best": best,
        "args": vars(a)}, indent=1))
    shutil.rmtree(out / "last", ignore_errors=True)
    log(f"{name}: done; best nDCG@5 {st['best']:.4f} (base {st['base']['all']['ndcg5']:.4f}) -> {out / 'best'}")


def save_best(enc: Encoder, path: Path, name: str, a, data: Data, st: dict, m: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    enc.model.save_pretrained(tmp, safe_serialization=True)
    enc.tok.save_pretrained(tmp)
    man = data.manifest
    (tmp / "f1_meta.json").write_text(json.dumps({
        "run": name, "base_model": BASE_MODEL, "pooling": "mean", "normalize": True,
        "query_prefix": man["query_prefix"], "passage_prefix": man["passage_prefix"],
        "max_query_len": man["max_query_len"], "max_passage_len": a.max_passage_len,
        "doc_format": man["format"], "step": st["step"], "val": m,
        "pool_tasks": data.pool_task_count, "holdout_icons": sorted(data.icons[i]["name"] for i in data.held),
        "args": vars(a)}, indent=1))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="f1_data dir or .tar.gz")
    ap.add_argument("--out", required=True, help="runs directory")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help=f"the planned runs in order: {', '.join(PLAN)}")
    g.add_argument("--run", help="one run: a PLAN name or any other name (with --pool-tasks)")
    ap.add_argument("--pool-tasks", type=int, default=None, help="for a non-PLAN --run; default all")
    ap.add_argument("--no-holdout", action="store_true", help="for a non-PLAN --run")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--i2q-weight", type=float, default=0.3)
    ap.add_argument("--free-per-epoch", type=int, default=25000)
    ap.add_argument("--parent-p", type=float, default=0.5)
    ap.add_argument("--max-passage-len", type=int, default=256)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--train-embeddings", action="store_true")
    ap.add_argument("--no-checkpointing", action="store_true", help="faster, needs ~2x GPU memory")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (keeps a laptop usable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="pause (resumable) after this long")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU test: batch 8, 10 steps, 200 val")
    a = ap.parse_args()
    if a.smoke:
        a.batch, a.max_steps, a.free_per_epoch, a.max_passage_len = 8, 10, 200, 128
        a.evals_per_epoch, a.patience = 1, 99
    root = open_data(Path(a.data))
    runs = PLAN.items() if a.plan else [(a.run, PLAN.get(a.run, (a.pool_tasks, not a.no_holdout)))]
    for name, (n, held) in runs:
        train(name, n, held, a, root)


if __name__ == "__main__":
    main()
