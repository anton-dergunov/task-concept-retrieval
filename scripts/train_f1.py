#!/usr/bin/env python3
"""Train F1, the fine-tuned dual encoder (docs/f1-dual-encoder.md), on the data built by
scripts/f1_data.py. Standalone: needs only torch and transformers, so it runs unchanged
on Colab (docs/f1-colab-run.md) or anywhere else.

    python train_f1.py --data f1_data.tar.gz --out runs --plan        # round 1: five runs
    python train_f1.py --data f1_data --out runs --plan round2        # round 2 (PLANS)
    python train_f1.py --data f1_data --out runs --run F1@3k          # one run
    python train_f1.py --data f1_data --out runs --run F1@all-lr,F1@all-bank   # a queue
    python train_f1.py --data f1_data --out runs --probe              # speed / memory check
    python train_f1.py --data f1_data_smoke --out runs --run smoke --smoke   # Mac CPU test

Two GPUs (Kaggle): one process per GPU, each with its own queue, e.g.
CUDA_VISIBLE_DEVICES=1 python train_f1.py ... --run F1@all-base (docs/f1-kaggle-run.md).

Every run writes runs/<name with @ as ->/: train_log.jsonl (loss every 20 steps, each
evaluation), best/ (the HF checkpoint with the best held-out nDCG@5, plus f1_meta.json),
last/state.pt while running (resume point, removed at the end) and done.json. A rerun
skips finished runs and resumes an interrupted one from its last evaluation. best/ also
holds icon_matrix.npz (every icon's vector, with a hash of its document), so scoring a
checkpoint elsewhere encodes only the queries.

Loss (§4): masked, graded multi-positive InfoNCE over the batch's icons (each query's
target spread over every in-batch icon the judge graded for it, ∝ grade; the judge's
relevant-but-ungraded icons and free-signal look-alikes masked), plus an icon→query term
with each icon's own near-misses as hard negative queries. --bank-negs adds every other
icon of the catalogue as a negative, from stale vectors refreshed every --bank-refresh
steps (no gradient through them), so each query is ranked against all icons, as in
evaluation.

Distillation (round 3, docs/next-data-and-distillation.md): a run with a teacher (preset
"teacher" or --teacher) first scores every training text with the best finished teacher
run (--score does only that: <out>/teacher/<run>.npz, top-64 icons per text), then adds
--kd-weight × KL(teacher ‖ student) over the whole catalogue (fresh vectors for the
batch's icons, the stale bank for the rest). Unlabelled rows (--distill-sources) count
only there; rewrites of judged tasks (--para) either keep their original's grades where
the teacher agrees ("graded") or learn from the teacher alone ("kd").

    python train_f1.py --data f1_data --out runs --run F1@all-base-lr,F1@all-kd,F1@all-kd-ms
    python train_f1.py --data f1_data --out runs --score F1@all-base-lr,F1@all-lr
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np

# Before torch touches CUDA: lets the allocator grow segments instead of failing on a
# fragmented cache (round 3's first e5-large try died with 3.3 GB reserved but unusable).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

BASE_MODEL = "intfloat/multilingual-e5-small"
# Teacher candidates for the students, in --out: the finished one with the best held-out
# nDCG@5 is used (override with --teacher).
TEACHERS = ["F1@all-large", "F1@all-base-lr", "F1@all-base"]
TOPK = 64            # teacher scores kept per text
BASE_LANGS = ("en", "es", "ru")
# plan -> run name -> settings: pool_tasks (None = all), holdout (5% of icons), and any
# argument overrides. Round 1: the method itself first, so a short session still produces
# it, then the learning curve. Round 2: the recipe (all comparable with F1@all).
PLANS = {
    "round1": {
        "F1": {"pool_tasks": None, "holdout": False},
        "F1@all": {"pool_tasks": None, "holdout": True},
        "F1@0": {"pool_tasks": 0, "holdout": True},
        "F1@1k": {"pool_tasks": 1000, "holdout": True},
        "F1@3k": {"pool_tasks": 3000, "holdout": True},
    },
    "round2": {
        "F1@all-lr": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10},
        "F1@all-bank": {"pool_tasks": None, "holdout": True, "bank_negs": True},
        "F1@all-lrbank": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                          "bank_negs": True},
        "F1@all-base": {"pool_tasks": None, "holdout": True,
                        "model": "intfloat/multilingual-e5-base"},
    },
    # Round 3 (docs/next-data-and-distillation.md): teachers, then e5-small students that
    # learn the best teacher's scores over the whole catalogue (--teacher picks it).
    "round3": {
        "F1@all-base-lr": {"pool_tasks": None, "holdout": True, "lr": 3e-5, "epochs": 10,
                           "model": "intfloat/multilingual-e5-base"},
        "F1@all-large": {"pool_tasks": None, "holdout": True, "epochs": 5,
                         "model": "intfloat/multilingual-e5-large", "chunk_tokens": 2048},
        "F1@all-kd": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                      "teacher": TEACHERS, "kd_weight": 1.0},
        "F1@all-kd-ms": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                         "teacher": TEACHERS, "kd_weight": 1.0, "distill_sources": ["distill-mslatte"]},
    },
    # Round 4: the cheap text from Vertex V1 (rewrites, generated tasks); `para` rows are
    # rewrites of judged pool tasks, used with the original's grades ("graded", kept only
    # where the teacher agrees) or with teacher targets only ("kd"). The students below learn
    # from F1@all-large (the best held-out teacher after round 3), pinned so that a teacher
    # finishing mid-session cannot change it; round 3's F1@all-kd(-ms) learned from
    # F1@all-base-lr, so -L repeats them with e5-large: the controls for -para and -gen.
    # F1@all-large-e8: e5-large was still improving at its last step; 8 epochs, ~10 h.
    "round4": {
        "F1@all-large-e8": {"pool_tasks": None, "holdout": True, "epochs": 8,
                            "model": "intfloat/multilingual-e5-large", "chunk_tokens": 2048},
        "F1@all-kd-L": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                        "teacher": ["F1@all-large"], "kd_weight": 1.0},
        "F1@all-kd-ms-L": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                           "teacher": ["F1@all-large"], "kd_weight": 1.0,
                           "distill_sources": ["distill-mslatte"]},
        "F1@all-lr-para": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                           "teacher": ["F1@all-large"], "kd_weight": 0.0, "para": "graded"},
        "F1@all-kd-para": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                           "teacher": ["F1@all-large"], "kd_weight": 1.0, "para": "kd"},
        "F1@all-kd-gen": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                          "teacher": ["F1@all-large"], "kd_weight": 1.0,
                          "distill_sources": ["distill-mslatte", "distill-cand"]},
    },
    # Round 5: + the Argon labels of round 3's selected tasks (2k active + 1k random). Equal
    # sizes answer "did active selection beat random?"; all3k the learning-curve point; then
    # the retrained teachers (e5-large, the better teacher in round 3; e5-base, the mid-size
    # bundle option) and the final student.
    # After round 4 (silver 2026-11-20): F1@all-large-e8 beat F1@all-large (+0.006 [+0.002,
    # +0.009]), so the retrained teacher has the 8-epoch schedule. Rewrites with the original's
    # grades were the best cheap text (F1@all-lr-para +0.019 over F1@all-lr), distillation added
    # only +0.003-0.008, so the final student takes graded rewrites plus distillation, pinned
    # to F1@all-large-e8: it runs on GPU 0 in the same session as the teacher's retraining.
    # F1@all-lr-para-all3k is its control (does distillation add anything on top?).
    "round5": {
        "F1@all-lr-rand1k": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                             "sel": {"random": 1000}},
        "F1@all-lr-act1k": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                            "sel": {"active": 1000}},
        # The same question where 1k tasks still matter (F1@0 -> F1@1k was +0.04): the
        # selected tasks alone, in F1@1k's recipe, so F1@1k (1k random tasks of the earlier
        # pools) is a third point. On top of 8.9k, 1k tasks add only ~+0.004.
        "F1@1k-rand": {"pool_tasks": None, "holdout": True, "sel": {"random": 1000}, "sel_only": True},
        "F1@1k-act": {"pool_tasks": None, "holdout": True, "sel": {"active": 1000}, "sel_only": True},
        "F1@all-lr-all3k": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                            "sel": {"active": None, "random": None}},
        "F1@all-large-all3k": {"pool_tasks": None, "holdout": True, "epochs": 8,
                               "model": "intfloat/multilingual-e5-large", "chunk_tokens": 2048,
                               "sel": {"active": None, "random": None}},
        "F1@all-lr-para-all3k": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                                 "teacher": ["F1@all-large-e8"], "kd_weight": 0.0, "para": "graded",
                                 "sel": {"active": None, "random": None}},
        "F1@all-base-lr-all3k": {"pool_tasks": None, "holdout": True, "lr": 3e-5, "epochs": 10,
                                 "model": "intfloat/multilingual-e5-base",
                                 "sel": {"active": None, "random": None}},
        "F1@all-kd-final": {"pool_tasks": None, "holdout": True, "lr": 5e-5, "epochs": 10,
                            "teacher": ["F1@all-large-e8"], "kd_weight": 1.0,
                            "sel": {"active": None, "random": None}, "para": "graded",
                            "distill_sources": ["distill-mslatte", "distill-cand"]},
    },
}
ALL_RUNS = {n: c for plan in PLANS.values() for n, c in plan.items()}
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
    def __init__(self, root: Path, pool_tasks, holdout: bool, smoke: bool, distill_sources=(),
                 para=None, teacher=None, sel=None, pool_langs=None, sel_only=False):
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.icons = read_jsonl(root / "icons.jsonl")
        self.docs = [self.manifest["passage_prefix"] + r["doc"] for r in self.icons]
        self.holdout_all = {i for i, r in enumerate(self.icons) if r["holdout"]}
        self.held = self.holdout_all if holdout else set()
        self.teacher = teacher
        qs = read_jsonl(root / "queries.jsonl")
        self._sel_keep = {}
        for v, n in (sel or {}).items():
            if n is not None:      # n judged tasks: pool rows only (rewrites follow their original)
                ids = sorted({q["task_id"] for q in qs if q.get("sel") == v and q["source"].startswith("pool")},
                             key=lambda t: _h("f1-sel:" + t))
                self._sel_keep[v] = set(ids[:n])
        for i, q in enumerate(qs):
            q["_i"] = i                       # row of the teacher's score arrays
            q["grades"] = {int(k): v for k, v in q["grades"].items()}
            q["ambiguous"] = set(q["ambiguous"])
            q["pos"] = [i for i in q["grades"] if i not in self.held]
            q["negs"] = [i for i in q["hard_negs"] if i not in self.held]
        # Tasks of the selected pools (round 3's Argon judging) carry `sel`; they never join
        # validation (the data puts all of them in the train split), and train only in runs
        # that ask for them: sel = {value: max tasks or None}.
        # Pool translations beyond EN/ES/RU train only where --pool-langs lists them; the
        # validation set stays EN/ES/RU, so held-out numbers stay comparable.
        # sel_only: the selected tasks alone, without the earlier pools (F1@1k-act/-rand).
        langs = set(BASE_LANGS) | set(pool_langs or ())
        self.val = [q for q in qs if q["split"] == "val" and not q.get("sel") and q["lang"] in BASE_LANGS]
        pool = [q for q in qs if q["split"] == "train" and q["source"].startswith("pool")
                and q["lang"] in langs and self._sel_ok(q, sel or {}) and (q.get("sel") or not sel_only)]
        keep = self._select_tasks(pool, pool_tasks)
        self.pool = [q for q in pool if q["task_id"] in keep and q["pos"]]
        self.free = [q for q in qs if q["source"].startswith("free") and q["pos"]]
        self.pool_task_count = len(keep)
        # Unlabelled text: learns from the teacher's scores only.
        self.distill = [q for q in qs if q["split"] == "train"
                        and any(q["source"] == s for s in distill_sources)]
        # Rewrites of judged pool tasks (only those whose original is in this run's pool).
        paras = [q for q in qs if q["split"] == "train" and q["source"].startswith("para")
                 and q.get("orig_task_id") in keep] if para else []
        self.para_dropped = 0
        if para == "graded":
            ok = [q for q in paras if q["pos"] and self.teacher_agrees(q)]
            self.para_dropped = len(paras) - len(ok)
            self.pool += ok
        elif para == "kd":
            for q in paras:
                q["grades"], q["pos"], q["negs"], q["ambiguous"] = {}, [], [], set()
            self.distill += paras
        if self.distill and teacher is None:
            raise SystemExit("distillation rows need a teacher (--teacher or the run's preset)")
        if smoke:
            self.val = self.val[:200]

    _sel_keep: dict = {}

    def _sel_ok(self, q: dict, sel: dict) -> bool:
        v = q.get("sel")
        if not v:
            return True
        if v not in sel:
            return False
        if sel[v] is None:
            return True
        if v not in self._sel_keep:     # the first n tasks of that kind in a fixed hash order
            raise RuntimeError("call _prepare_sel first")
        return q["task_id"] in self._sel_keep[v]

    def teacher_agrees(self, q: dict, k: int = 10) -> bool:
        """A rewrite keeps its original's grades only if the teacher still ranks one of the
        original's grade-3 icons in its top k."""
        if self.teacher is None:
            raise SystemExit("para=graded filters rewrites with a teacher; none was found")
        top = set(int(x) for x in self.teacher.top(q["_i"], 0)[:k])
        return bool(top & {i for i, g in q["grades"].items() if g == 3})

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

    def epoch_rows(self, epoch: int, seed: int, free_per_epoch: int, distill_per_epoch: int = 0) -> list:
        rng = random.Random(seed * 1000 + epoch)
        rows = list(self.pool) + rng.sample(self.free, min(free_per_epoch, len(self.free)))
        if self.distill:
            n = len(self.distill) if not distill_per_epoch else min(distill_per_epoch, len(self.distill))
            rows += rng.sample(self.distill, n)
        rng.shuffle(rows)
        return rows


class Teacher:
    """A teacher's top-K icons and scaled scores per query row and form (0 = title,
    1 = parent + title), written by score_texts(); -1 where a row or form is missing."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=False)
        self.path, self.run = path, str(d["run"])
        self.idx, self.val = d["idx"], d["val"]
        self.names = [str(n) for n in d["names"]]

    def top(self, row: int, form: int) -> np.ndarray:
        ix = self.idx[row, form]
        if ix[0] < 0 and form == 1:
            ix = self.idx[row, 0]
        return ix[ix >= 0]

    def target(self, row: int, form: int, n_icons: int, held: set, tau: float):
        """Dense teacher distribution over the catalogue (held-out icons get no mass)."""
        if self.idx[row, form, 0] < 0:
            form = 0
        ix, v = self.idx[row, form], self.val[row, form].astype(np.float32)
        ok = ix >= 0
        if held:
            ok &= ~np.isin(ix, list(held))
        if not ok.any():
            return None
        ix, v = ix[ok], v[ok] / tau
        p = np.exp(v - v.max())
        t = np.zeros(n_icons, np.float32)
        t[ix] = p / p.sum()
        return t


# --- model ---------------------------------------------------------------------------------

class Encoder:
    """Mean-pooled, L2-normalized e5 (as tcr/finetune.py's matcher encodes)."""

    def __init__(self, path: str, device: str, train_embeddings: bool, checkpointing: bool,
                 chunk_tokens: int = 0):
        self.checkpointing = checkpointing
        self.chunk_tokens = chunk_tokens
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device)
        if not train_embeddings:
            self.model.embeddings.word_embeddings.weight.requires_grad_(False)
        if checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.amp = device == "cuda"
        self.pad_full = False          # worst-case padding, for the memory check

    def set_checkpointing(self, on: bool) -> None:
        if on == self.checkpointing:
            return
        if on:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.model.gradient_checkpointing_disable()
        self.checkpointing = on

    def __call__(self, texts, max_len: int) -> torch.Tensor:
        enc = self.tok(list(texts), padding="max_length" if self.pad_full else True,
                       truncation=True, max_length=max_len, pad_to_multiple_of=8,
                       return_tensors="pt").to(self.device)
        keys = list(enc.keys())
        n, L = enc["input_ids"].shape
        step = max(1, self.chunk_tokens // L) if self.chunk_tokens else n
        if step >= n or not torch.is_grad_enabled():
            return self._embed(keys, *enc.values())
        # --chunk-tokens: encode the texts in chunks of about this many tokens, each under
        # an outer checkpoint that keeps only its output vectors and reruns the chunk in
        # the backward pass (same dropout). The loss still sees the whole batch, so the
        # gradients are those of the unchunked batch (as GradCache), at one chunk's
        # activation memory; peak memory no longer grows with the batch's icon count.
        return torch.cat([checkpoint(self._embed, keys, *(enc[k][s:s + step] for k in keys),
                                     use_reentrant=False)
                          for s in range(0, n, step)])

    def _embed(self, keys, *tensors) -> torch.Tensor:
        enc = dict(zip(keys, tensors))
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

def batch_loss(enc: Encoder, data: Data, rows: list, rng: random.Random, a, bank=None,
               teacher: "Teacher | None" = None) -> tuple:
    """(graded loss, icon->query loss, distillation loss, icons in the batch). Rows without
    grades (distillation text) bring the teacher's top icon into the batch instead of a
    drawn positive and count only in the distillation loss."""
    man = data.manifest
    pos = [rng.choices(q["pos"], weights=[q["grades"][i] ** 2 for i in q["pos"]])[0] if q["pos"]
           else _teacher_top1(teacher, q, data.held) for q in rows]
    neg = [rng.choice(q["negs"]) if q["negs"] else None for q in rows]
    icons = sorted({p for p in pos if p is not None} | {n for n in neg if n is not None})
    col = {ic: j for j, ic in enumerate(icons)}
    forms = [1 if q.get("text_parent") and rng.random() < a.parent_p else 0 for q in rows]
    texts = [q["text_parent"] if f else q["text"] for q, f in zip(rows, forms)]
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
    has = G.sum(1) > 0                           # rows with grades (all of them before round 3)
    T = G / G.sum(1, keepdim=True).clamp(min=1e-9)
    logits = S.masked_fill(A, NEG)
    if bank is not None and a.bank_negs:
        # every other icon as a negative (stale vectors, no gradient): masked where it is
        # in the batch already, held out, graded or set aside for this query
        Mb = torch.zeros(B, bank.shape[0], dtype=torch.bool)
        Mb[:, icons] = True
        if data.held:
            Mb[:, sorted(data.held)] = True
        for r, q in enumerate(rows):
            Mb[r, list(q["grades"])] = True
            if q["ambiguous"]:
                Mb[r, list(q["ambiguous"])] = True
        Sb = (a.scale * Q @ bank.T).masked_fill(Mb.to(Q.device), NEG)
        logits = torch.cat([logits, Sb], dim=1)
        T = torch.cat([T, torch.zeros_like(Sb)], dim=1)
    per_row = -(T * F.log_softmax(logits, dim=1)).sum(1)
    loss_q = per_row[has].mean() if bool(has.any()) else per_row.sum() * 0

    # distillation: KL(teacher || student) over the whole catalogue, fresh vectors for the
    # batch's icons, the stale bank for the rest; held-out icons excluded on both sides
    loss_kd = torch.zeros((), device=Q.device)
    if teacher is not None and bank is not None:
        tgt = [teacher.target(q["_i"], f, len(data.docs), data.held, a.kd_temp)
               for q, f in zip(rows, forms)]
        keep = [r for r, t in enumerate(tgt) if t is not None]
        if keep:
            Tk = torch.from_numpy(np.stack([tgt[r] for r in keep])).to(Q.device)
            Sf = a.scale * Q[keep] @ bank.T
            Sf = Sf.index_copy(1, torch.tensor(icons, device=Q.device), S[keep])
            if data.held:
                Sf[:, sorted(data.held)] = NEG
            logp = F.log_softmax(Sf / a.kd_temp, dim=1)
            ent = (Tk * torch.log(Tk.clamp(min=1e-12))).sum(1)
            loss_kd = ((ent - (Tk * logp).sum(1)) * a.kd_temp ** 2).mean()

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
    return loss_q, loss_i, loss_kd, M


def _teacher_top1(teacher, q: dict, held: set):
    if teacher is None:
        raise SystemExit(f"row {q['task_id']} has no grades and there is no teacher")
    for ic in teacher.top(q["_i"], 0):
        if int(ic) not in held:
            return int(ic)
    return None


# --- evaluation ----------------------------------------------------------------------------

def evaluate(enc: Encoder, data: Data, max_passage_len: int) -> dict:
    man = data.manifest
    I = enc.encode(data.docs, max_passage_len)
    enc.last_icons = I
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
    dev = pick_device(a.device)
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    teacher = None
    if a.teacher:
        cands = a.teacher if isinstance(a.teacher, list) else a.teacher.split(",")
        t_run = pick_teacher(a, cands)
        if t_run is None:
            log(f"{name}: no finished teacher among {cands} in {a.out}; skipped")
            return
        teacher = Teacher(score_texts(a, data_root, t_run))
        a.teacher_used = teacher.run
        log(f"{name}: teacher {teacher.run}")
    out.mkdir(parents=True, exist_ok=True)
    data = Data(data_root, pool_tasks, holdout, a.smoke, a.distill_sources or (), a.para, teacher,
                a.sel, a.pool_langs, a.sel_only)
    t_loss = teacher if a.kd_weight > 0 else None
    log(f"{name}: device {dev}; {data.pool_task_count} pool tasks ({len(data.pool)} rows), "
        f"{len(data.free)} free-signal rows ({a.free_per_epoch} per epoch), {len(data.distill)} "
        f"distillation rows, {len(data.val)} val, {len(data.held)} icons held out"
        + (f"; {data.para_dropped} rewrites dropped (teacher disagrees)" if a.para == "graded" else ""))
    torch.manual_seed(a.seed)
    enc = Encoder(a.model, dev, a.train_embeddings, a.checkpointing != "off", a.chunk_tokens)
    params = [p for p in enc.model.parameters() if p.requires_grad]
    decay = [p for n, p in enc.model.named_parameters() if p.requires_grad and p.ndim > 1]
    nodecay = [p for n, p in enc.model.named_parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr)
    rows0 = data.epoch_rows(0, a.seed, a.free_per_epoch, a.distill_per_epoch)
    steps_per_epoch = max(1, len(rows0) // a.batch)
    total = a.max_steps or steps_per_epoch * a.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    # In-epoch positions of the evaluations, the epoch end being 0. Before round 5 the test was
    # `pos % (spe // E) == 0`, which with an odd epoch (401 steps) also fired one step before
    # its end: two evaluations a step apart, the second spending a patience slot.
    eval_at = {k * steps_per_epoch // a.evals_per_epoch % steps_per_epoch
               for k in range(1, max(1, a.evals_per_epoch) + 1)}
    if a.checkpointing == "auto" and dev == "cuda":
        try:
            choose_checkpointing(enc, data, rows0, a, name, t_loss)
        except Exception as e:      # the check must never cost a run: fall back to the safe side
            enc.pad_full = False
            enc.model.zero_grad(set_to_none=True)
            enc.set_checkpointing(True)
            log(f"{name}: memory check failed ({type(e).__name__}: {e}); checkpointing on")

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
    bank = None

    need_bank = a.bank_negs or t_loss is not None

    def refresh_bank():
        nonlocal bank
        if need_bank:
            bank = enc.encode(data.docs, a.max_passage_len).detach()

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
    refresh_bank()
    while st["step"] < total and not stop:
        epoch = st["step"] // steps_per_epoch
        rows = data.epoch_rows(epoch, a.seed, a.free_per_epoch, a.distill_per_epoch)
        in_epoch = st["step"] % steps_per_epoch
        enc.model.train()
        t_log, lq_acc, li_acc, lk_acc, m_acc, n_acc = time.time(), 0.0, 0.0, 0.0, 0, 0
        for b in range(in_epoch, steps_per_epoch):
            batch = rows[b * a.batch:(b + 1) * a.batch]
            rng = random.Random(f"{a.seed}:{st['step']}")
            if need_bank and st["step"] % a.bank_refresh == 0 and st["step"] > 0:
                refresh_bank()
            loss_q, loss_i, loss_kd, M = batch_loss(enc, data, batch, rng, a, bank, t_loss)
            loss = loss_q + a.i2q_weight * loss_i + a.kd_weight * loss_kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            st["step"] += 1
            lq_acc += loss_q.item(); li_acc += loss_i.item(); lk_acc += loss_kd.item()
            m_acc += M; n_acc += 1
            if st["step"] % 20 == 0 or a.smoke:
                dt = (time.time() - t_log) / n_acc
                mem = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None
                rec = {"step": st["step"], "loss_q": round(lq_acc / n_acc, 4),
                       "loss_i": round(li_acc / n_acc, 4), "icons_per_batch": m_acc // n_acc,
                       "lr": sched.get_last_lr()[0], "sec_per_step": round(dt, 3), "max_mem_gb": mem}
                if t_loss is not None:
                    rec["loss_kd"] = round(lk_acc / n_acc, 4)
                write(rec)
                eta = (total - st["step"]) * dt / 60
                kd = f" + kd {lk_acc / n_acc:.3f}" if t_loss is not None else ""
                log(f"{name} step {st['step']}/{total} loss {lq_acc / n_acc:.3f} + {li_acc / n_acc:.3f}{kd} "
                    f"({dt:.2f} s/step, ≤ {eta:.0f} min left)")
                t_log, lq_acc, li_acc, lk_acc, m_acc, n_acc = time.time(), 0.0, 0.0, 0.0, 0, 0
            seg_end = st["step"] % steps_per_epoch in eval_at
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
        "distill_rows": len(data.distill), "teacher": teacher.run if teacher else None,
        "para_dropped": data.para_dropped,
        "steps": st["step"], "steps_per_epoch": steps_per_epoch, "base": st.get("base"), "best": best,
        "checkpointing": enc.checkpointing, "train_minutes": round((time.time() - t_start) / 60, 1),
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
        "run": name, "base_model": a.model, "pooling": "mean", "normalize": True,
        "query_prefix": man["query_prefix"], "passage_prefix": man["passage_prefix"],
        "max_query_len": man["max_query_len"], "max_passage_len": a.max_passage_len,
        "doc_format": man["format"], "step": st["step"], "val": m,
        "pool_tasks": data.pool_task_count, "holdout_icons": sorted(data.icons[i]["name"] for i in data.held),
        "args": vars(a)}, indent=1))
    I = getattr(enc, "last_icons", None)          # from the evaluation that triggered this save
    if I is not None:
        np.savez_compressed(tmp / "icon_matrix.npz", names=np.array([r["name"] for r in data.icons]),
                            doc_sha1=np.array([hashlib.sha1(d.encode()).hexdigest() for d in data.docs]),
                            max_len=a.max_passage_len, matrix=I.cpu().numpy().astype(np.float32))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def choose_checkpointing(enc: Encoder, data: Data, rows: list, a, name: str, teacher=None) -> None:
    """--checkpointing auto: one worst-case step (every passage padded to the maximum length)
    without checkpointing; keep it off if peak memory stays under 85% of the GPU."""
    total = torch.cuda.get_device_properties(0).total_memory
    enc.set_checkpointing(False)
    enc.pad_full = True
    torch.cuda.reset_peak_memory_stats()
    try:
        bank = (torch.zeros(len(data.docs), enc.model.config.hidden_size, device=enc.device)
                if a.bank_negs or teacher is not None else None)
        lq, li, lk, _ = batch_loss(enc, data, rows[:a.batch], random.Random(0), a, bank, teacher)
        (lq + a.i2q_weight * li + a.kd_weight * lk).backward()
        peak = torch.cuda.max_memory_allocated()
        ok = peak < 0.85 * total
    except torch.cuda.OutOfMemoryError:
        peak, ok = total, False
    enc.model.zero_grad(set_to_none=True)
    enc.pad_full = False
    torch.cuda.empty_cache()
    enc.set_checkpointing(not ok)
    chunk = f" (chunks of {a.chunk_tokens} tokens)" if a.chunk_tokens else ""
    log(f"{name}: worst-case step without checkpointing{chunk} peaks at {peak / 2**30:.1f} of "
        f"{total / 2**30:.1f} GB -> checkpointing {'off' if ok else 'on'}")


def run_dir(a, run: str) -> Path:
    return Path(a.out) / run.replace("@", "-")


def pick_teacher(a, cands) -> "str | None":
    """The finished candidate (a run in --out, or a checkpoint dir) with the best held-out
    nDCG@5; a checkpoint dir given directly is taken as is."""
    best = None
    for r in cands:
        if Path(r).is_dir() and (Path(r) / "config.json").exists():
            return r
        d = run_dir(a, r)
        if (d / "done.json").exists() and (d / "best").is_dir():
            s = (json.loads((d / "done.json").read_text()).get("best") or {}).get("all", {}).get("ndcg5", -1)
            log(f"teacher candidate {r}: held-out nDCG@5 {s}")
            if best is None or s > best[1]:
                best = (r, s)
    return best[0] if best else None


def score_texts(a, data_root: Path, run: str) -> Path:
    """Encode every training text (title and parent forms) with a trained checkpoint and
    keep its top-K icons and scaled scores: <out>/teacher/<run>.npz. Reused while
    queries.jsonl is unchanged. Also the input of scripts/select_active.py."""
    src = Path(run) if Path(run).is_dir() else run_dir(a, run) / "best"
    tag = (src.parent.name if src.name == "best" else src.name).replace("@", "-")
    qp = data_root / "queries.jsonl"
    sha = hashlib.sha1(qp.read_bytes()).hexdigest()
    out = Path(a.out) / "teacher" / f"{tag}.npz"
    if out.exists() and str(np.load(out)["queries_sha1"]) == sha:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    man = json.loads((data_root / "manifest.json").read_text())
    icons = read_jsonl(data_root / "icons.jsonl")
    docs = [man["passage_prefix"] + r["doc"] for r in icons]
    qs = read_jsonl(qp)
    jobs = [(r, f, t) for r, q in enumerate(qs) if q["split"] == "train"
            for f, t in ((0, q["text"]), (1, q.get("text_parent"))) if t]
    t0 = time.time()
    enc = Encoder(str(src), pick_device(a.device), False, False)
    I = enc.encode(docs, a.max_passage_len)
    idx = np.full((len(qs), 2, TOPK), -1, np.int16)
    val = np.zeros((len(qs), 2, TOPK), np.float16)
    for s in range(0, len(jobs), 4096):
        chunk = jobs[s:s + 4096]
        Qv = enc.encode([man["query_prefix"] + t for _, _, t in chunk], man["max_query_len"], batch=512)
        v, ix = (a.scale * Qv @ I.T).topk(TOPK, dim=1)
        v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for k, (r, f, _) in enumerate(chunk):
            idx[r, f], val[r, f] = ix[k], v[k]
    np.savez_compressed(out, run=tag, names=np.array([r["name"] for r in icons]), idx=idx, val=val,
                        queries_sha1=sha, scale=a.scale)
    log(f"teacher scores of {tag}: {len(jobs)} texts in {time.time() - t0:.0f} s -> {out}")
    del enc, I
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def probe(a, data_root: Path) -> None:
    """Time 30 steps with and without gradient checkpointing (no checkpoint written)."""
    dev = pick_device(a.device)
    data = Data(data_root, None, True, a.smoke)
    rows = data.epoch_rows(0, a.seed, a.free_per_epoch)
    enc = Encoder(a.model, dev, a.train_embeddings, True, a.chunk_tokens)
    opt = torch.optim.AdamW([p for p in enc.model.parameters() if p.requires_grad], lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    bank = (enc.encode(data.docs, a.max_passage_len) if a.bank_negs else None)
    for ck in (True, False):
        enc.set_checkpointing(ck)
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            for b in range(30):
                if b == 5:
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                lq, li, _, _ = batch_loss(enc, data, rows[b * a.batch:(b + 1) * a.batch],
                                          random.Random(b), a, bank)
                opt.zero_grad(set_to_none=True)
                scaler.scale(lq + a.i2q_weight * li).backward()
                scaler.step(opt)
                scaler.update()
            if dev == "cuda":
                torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
            log(f"probe {a.model} checkpointing {'on ' if ck else 'off'}: "
                f"{(time.time() - t0) / 25:.2f} s/step, peak {mem:.1f} GB")
        except torch.cuda.OutOfMemoryError:
            log(f"probe {a.model} checkpointing {'on ' if ck else 'off'}: out of GPU memory")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="f1_data dir or .tar.gz")
    ap.add_argument("--out", required=True, help="runs directory")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", nargs="?", const="round1", choices=list(PLANS),
                   help="the planned runs in order (default round1); " +
                   "; ".join(f"{k}: {', '.join(v)}" for k, v in PLANS.items()))
    g.add_argument("--run", help="runs in order, comma-separated: PLANS names or new names "
                                 "(new ones take --pool-tasks / --no-holdout and the flags)")
    g.add_argument("--probe", action="store_true", help="time 30 steps with and without checkpointing")
    g.add_argument("--score", help="only write teacher scores (<out>/teacher/<run>.npz) for these "
                                   "runs or checkpoint dirs, comma-separated (scripts/select_active.py)")
    ap.add_argument("--pool-tasks", type=int, default=None, help="for a new --run name; default all")
    ap.add_argument("--no-holdout", action="store_true", help="for a new --run name")
    ap.add_argument("--model", default=BASE_MODEL, help="base encoder (HF id)")
    ap.add_argument("--bank-negs", action="store_true", help="all other icons as stale negatives")
    ap.add_argument("--bank-refresh", type=int, default=100, help="steps between bank refreshes")
    ap.add_argument("--teacher", default=None,
                    help="distil from the best finished of these runs (comma-separated) or a checkpoint dir")
    ap.add_argument("--kd-weight", type=float, default=0.0, help="weight of the distillation loss")
    ap.add_argument("--kd-temp", type=float, default=1.0, help="softmax temperature of the distillation loss")
    ap.add_argument("--distill-sources", type=lambda s: s.split(","), default=[],
                    help="unlabelled row sources to train on (distill-mslatte, distill-cand)")
    ap.add_argument("--distill-per-epoch", type=int, default=0, help="sample of distillation rows per epoch (0 = all)")
    ap.add_argument("--para", choices=["graded", "kd"], default=None, help="use the rewrite rows")
    ap.add_argument("--pool-langs", type=lambda s: s.split(","), default=[],
                    help="pool translations beyond EN/ES/RU to train on (augment_pools.py translate)")
    ap.add_argument("--sel", type=json.loads, default=None,
                    help='selected-pool tasks to train on, JSON {"active": n|null, "random": n|null}')
    ap.add_argument("--sel-only", action="store_true",
                    help="train on the --sel tasks alone, without the earlier pools")
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
    ap.add_argument("--checkpointing", choices=["auto", "on", "off"], default="auto",
                    help="gradient checkpointing; auto: off when a worst-case step fits the GPU")
    ap.add_argument("--chunk-tokens", type=int, default=0,
                    help="encode a training batch in checkpointed chunks of about this many tokens "
                         "(same gradients, far less activation memory; 0 = off; e5-large's preset: 2048)")
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
    if a.probe:
        probe(a, root)
        return
    if a.score:
        for r in a.score.split(","):
            score_texts(a, root, r)
        return
    if a.plan:
        runs = list(PLANS[a.plan].items())
    else:
        runs = [(n, ALL_RUNS.get(n, {"pool_tasks": a.pool_tasks, "holdout": not a.no_holdout}))
                for n in a.run.split(",")]
    for name, cfg in runs:
        ra = argparse.Namespace(**{**vars(a), **{k: v for k, v in cfg.items()
                                                 if k not in ("pool_tasks", "holdout")}})
        train(name, cfg["pool_tasks"], cfg["holdout"], ra, root)


if __name__ == "__main__":
    main()
