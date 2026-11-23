"""F1, the fine-tuned dual encoder (docs/f1-dual-encoder.md): the icon document, the
training rows built from the judge's pool labels and v2's free signal, the splits, and
the matcher that scores with a trained checkpoint.

Training itself is `scripts/train_f1.py`, standalone so it runs on Colab; the data it
reads is built here by `scripts/f1_data.py`. The encoder below (mean pooling, prefixes,
lengths) mirrors the one in train_f1.py; keep them in step.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import config, silver
from .data import IconDoc, load_icons
from .datasets import read_jsonl
from .methods.base import Matcher
from .org_tasks import query_text

MODELS_DIR = config.ROOT / ".cache" / "models"
OUT_DIR = config.PRIVATE_DIR / "f1"
AUG_FILES = {
    "personal_synth_pool": config.PRIVATE_DIR / "train_pools" / "personal_synth_pool.aug.jsonl",
    "train_mslatte": config.DATA_DIR / "train_pools" / "train_mslatte.aug.jsonl",
}
SOURCE = {"personal_synth_pool": "pool-synth", "train_mslatte": "pool-mslatte"}
# Register rewrites of the judged pool tasks (augment_pools.py rewrite; Vertex V1 of
# docs/next-data-and-distillation.md): one line per task, {"task_id", "variants": [{"lang",
# "style", "title"}], "contaminated": [variant indices]}.
REWRITE_FILES = {
    "personal_synth_pool": config.PRIVATE_DIR / "train_pools" / "personal_synth_pool.rw.jsonl",
    "train_mslatte": config.DATA_DIR / "train_pools" / "train_mslatte.rw.jsonl",
}
PARA_SOURCE = {"personal_synth_pool": "para-synth", "train_mslatte": "para-mslatte"}
# Round 6's minimal pairs (augment_pools.py minpair, argon-experiments.md #3): a judged
# training task edited in 1-3 words so the right icon should change, the edit judged by
# Argon (judge run, edits as judge input). Pairs of validation tasks are the contrast set
# (scripts/contrast_eval.py) and never become rows. `mp-*` rows carry `orig_task_id` and
# are used only by runs that ask for them.
MP_RUNS = {
    "personal_synth_pool": ("mp-synth-j4", config.PRIVATE_DIR / "train_pools" / "personal_synth_pool.mp_pool.jsonl"),
    "train_mslatte": ("mp-mslatte-j4", config.DATA_DIR / "train_pools" / "train_mslatte.mp_pool.jsonl"),
}
MP_SOURCE = {"personal_synth_pool": "mp-synth", "train_mslatte": "mp-mslatte"}
# Round 6's adversarial tasks (augment_pools.py adversarial, argon-experiments.md #2): new
# tasks written by Flash from the big model's errors on validation tasks, judged by Argon,
# with ES/RU translations (AUG_FILES) and rewrites (REWRITE_FILES) under their own ids. All
# train rows (`adv-*`; rewrites `para-adv-*` with orig_task_id), used only by runs that ask
# for them (train_f1.py --adv). Their seeds are validation tasks, so a run that trains on
# them reads its held-out numbers with a little optimism; silver is unaffected.
ADV_RUNS = {
    "personal_synth_pool": ("adv-synth-j4", config.PRIVATE_DIR / "train_pools" / "personal_synth_pool.adv_pool.jsonl"),
    "train_mslatte": ("adv-mslatte-j4", config.DATA_DIR / "train_pools" / "train_mslatte.adv_pool.jsonl"),
}
ADV_SOURCE = {"personal_synth_pool": "adv-synth", "train_mslatte": "adv-mslatte"}
# Round 6's VLM labels (scripts/vlm_rerank.py, argon-experiments.md #5): Argon's ranking and
# 0-3 grades of the big model's top 20 for 1,000 judged training tasks ("train") and 3,000
# unlabelled candidates ("cand", which are distill-* rows). f1_data.py writes them as
# vlm.jsonl beside queries.jsonl, keyed by task_id, so queries.jsonl keeps its rows.
VLM_LABELS = {s: config.PRIVATE_DIR / "vlm" / "labels" / f"{s}.jsonl" for s in ("train", "cand")}
# Translations of the pools into more languages (augment_pools.py translate), used only by
# runs that list the language in train_f1.py --pool-langs.
XL_FILES = {k: v.with_name(v.name.replace(".aug.", ".xl.")) for k, v in AUG_FILES.items()}
# Unlabelled text for distillation (rows without grades), shared dataset schema.
CANDIDATES_FILE = config.PRIVATE_DIR / "train_pools" / "personal_synth_cand.jsonl"
QUERY_PREFIX, PASSAGE_PREFIX = "query: ", "passage: "
MAX_QUERY_LEN, MAX_PASSAGE_LEN = 64, 256
VAL_EVERY = 10          # 1 in 10 pool tasks held out for early stopping
HOLDOUT_EVERY = 20      # 1 in 20 icons held out entirely (F1@… runs)
DOC_FORMAT = "f1-doc-1"


def _bucket(salt: str, key: str, n: int) -> bool:
    return int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:8], 16) % n == 0


def is_val_task(task_id: str) -> bool:
    return _bucket("f1-val", task_id, VAL_EVERY)


def is_holdout_icon(name: str) -> bool:
    return _bucket("f1-icon", name, HOLDOUT_EVERY)


def icon_doc(ic: IconDoc) -> str:
    """The icon side (without the passage prefix): depiction, readings, intents, the
    contrast with look-alikes, reasoning last so truncation cuts it first. Never the
    example / context tasks or near-misses (query-side text), never names or tags."""
    if ic.version != "v2":
        raise ValueError("F1's icon document is built from v2 descriptions")
    parts = [f"Shows: {'; '.join(ic.visual_concepts)}.",
             f"Means: {'; '.join(ic.readings)}.",
             f"For: {'; '.join(ic.task_intents)}."]
    if ic.distinguishing:
        parts.append(ic.distinguishing.strip())
    if ic.reasoning:
        parts.append(ic.reasoning.strip())
    return " ".join(p for p in parts if p)


def _v2_raw(name: str) -> dict:
    with open(config.ICON_DESC_V2_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _norm(t: str) -> str:
    from .describe_v2 import norm
    return norm(t)


# --- building the training data ---------------------------------------------------------

def build(smoke: bool = False, distill: Optional[dict] = None) -> dict:
    """Everything train_f1.py needs: {"manifest", "icons", "queries"} (see f1_data.py).

    `distill`: {source: [task in the shared schema]} of unlabelled text, written as rows
    without grades (e.g. "distill-mslatte", "distill-cand"); train_f1.py uses them only in
    runs that name the source, with a teacher. Rewrites in REWRITE_FILES, when present,
    become `para-*` rows carrying their original's grades and split."""
    icons = list(load_icons(version="v2"))
    index = {ic.name: i for i, ic in enumerate(icons)}
    stats: Counter = Counter()

    def idx(names) -> List[int]:
        return sorted(index[n] for n in names if n in index)

    icon_rows = []
    for ic in icons:
        raw = _v2_raw(ic.name)
        looks = {silver.canonical(n) for n in (raw.get("_meta") or {}).get("lookalikes") or []}
        icon_rows.append({"name": ic.name, "doc": icon_doc(ic), "holdout": is_holdout_icon(ic.name),
                          "lookalikes": idx(looks - {ic.name}),
                          "near_misses": [query_text(n["task"]) for n in raw.get("near_misses", [])
                                          if str(n.get("task", "")).strip()]})

    queries: List[dict] = []
    pool_ids = set()          # judged pool tasks with rows (smoke: minimal pairs of these only)
    eval_ids = _eval_ids()
    for ds, runs in silver.POOL_RUNS.items():
        aug = {r["task_id"]: r for r in read_jsonl(AUG_FILES[ds])}
        rw = ({r["task_id"]: r for r in read_jsonl(REWRITE_FILES[ds])}
              if REWRITE_FILES[ds].exists() else {})
        xl = {r["task_id"]: r for r in read_jsonl(XL_FILES[ds])} if XL_FILES[ds].exists() else {}
        for run in runs:
            gold = silver.load_gold([run])
            tasks = silver.run_tasks(run)
            if smoke:
                tasks = tasks[:100]
            for t in tasks:
                assert t["id"] not in eval_ids, f"eval task {t['id']} in pool run {run}"
                g = gold.get(t["id"])
                if g is None:
                    stats["pool task without label (failed)"] += 1
                    continue
                grades = {index[n]: v for n, v in g.grades.items() if n in index}
                stats["graded icons outside v2 (dropped)"] += len(g.grades) - len(grades)
                if not grades:
                    stats["pool task with no v2 icon graded (skipped)"] += 1
                    continue
                ambiguous = idx(g.relevant)
                hard = idx(g.seen - g.relevant - set(g.grades))
                a = aug.get(t["id"]) or {}
                # Round 3's selected pools carry `sel` and never validate (train_f1.py), so
                # all of them train: a val bucket would only drop paid labels.
                sel = (t.get("meta") or {}).get("selection")
                base = {"task_id": t["id"], "source": SOURCE[ds], "run": run,
                        "split": "val" if is_val_task(t["id"]) and not sel else "train",
                        "grades": {str(k): v for k, v in grades.items()},
                        "ambiguous": ambiguous, "hard_negs": hard}
                if sel:
                    base["sel"] = sel
                parents = list(t.get("parents") or []) or list(a.get("parents") or [])
                queries.append({**base, "lang": "en", **_texts(t["title"], parents)})
                pool_ids.add(t["id"])
                for lang in ("es", "ru"):
                    tr = a.get(lang)
                    if not tr:
                        stats[f"no {lang} translation"] += 1
                        continue
                    if lang in (a.get("contaminated") or []):
                        stats[f"{lang} skipped: contaminated"] += 1
                        continue
                    if _norm(tr["title"]) == _norm(t["title"]):
                        stats[f"{lang} skipped: equal to EN"] += 1
                        continue
                    queries.append({**base, "lang": lang, **_texts(tr["title"], tr.get("parents") or [])})
                x = xl.get(t["id"]) or {}
                for lang in x.get("langs") or []:
                    if lang in (x.get("contaminated") or []) or lang not in x:
                        continue
                    queries.append({**base, "lang": lang,
                                    **_texts(x[lang]["title"], x[lang].get("parents") or [])})
                r = rw.get(t["id"])
                if r and base["split"] == "train":       # never a rewrite of a validation task
                    bad = set(r.get("contaminated") or [])
                    for k, v in enumerate(r.get("variants") or []):
                        title = str(v.get("title", "")).strip()
                        if k in bad or not title or _norm(title) == _norm(t["title"]):
                            stats["rewrite skipped (contaminated, empty or unchanged)"] += 1
                            continue
                        lang = v.get("lang", "en")
                        par = parents if lang == "en" else list(((a.get(lang) or {}).get("parents")) or [])
                        queries.append({**base, "source": PARA_SOURCE[ds], "orig_task_id": t["id"],
                                        "task_id": f"{t['id']}~rw{k}", "lang": lang, "style": v.get("style"),
                                        **_texts(title, par)})

    for src, tasks in (distill or {}).items():
        for t in tasks[:200] if smoke else tasks:
            assert t["id"] not in eval_ids, f"eval task {t['id']} in distillation source {src}"
            queries.append({"task_id": t["id"], "source": src, "split": "train",
                            "lang": t.get("lang", "en"), "grades": {}, "ambiguous": [], "hard_negs": [],
                            **_texts(t["title"], list(t.get("parents") or []))})

    free = 0
    for i, ic in enumerate(icons):
        if smoke and i % 10:
            continue
        raw = _v2_raw(ic.name)
        rows = [(e["task"], [e["parent"]] if str(e.get("parent", "")).strip() else [], "en", 3, "free-example")
                for e in raw.get("example_tasks", [])]
        rows += [(x, [], lang, 3, "free-example") for lang in ("es", "ru")
                 for x in raw.get(f"example_tasks_{lang}", []) if str(x).strip()]
        rows += [(c["task"], [], "en", 2, "free-context") for c in raw.get("context_tasks", [])]
        for title, parents, lang, grade, src in rows:
            queries.append({"task_id": f"{ic.name}#{free}", "source": src, "split": "train",
                            "lang": lang, "grades": {str(i): grade},
                            "ambiguous": icon_rows[i]["lookalikes"], "hard_negs": [],
                            **_texts(title, parents)})
            free += 1

    # Rows added since round 5 come last, so an earlier build's rows keep their positions
    # (f1_data.py extends that build's teacher scores to the new rows).
    for ds, (run, pool) in MP_RUNS.items():
        if not pool.exists() or not silver.label_path(run).exists():
            continue
        gold = silver.load_gold([run])
        for t in read_jsonl(pool):
            m = t.get("meta") or {}
            if smoke and m.get("orig_task_id") not in pool_ids:
                continue
            if m.get("val"):
                stats["minimal pair of a validation task (contrast set, not a row)"] += 1
                continue
            assert t["id"] not in eval_ids and not is_val_task(m["orig_task_id"]), t["id"]
            g = gold.get(t["id"])
            if g is None or g.no_good_icon:
                stats["minimal pair without a label or a good icon"] += 1
                continue
            grades = {index[n]: v for n, v in g.grades.items() if n in index}
            if not grades:
                continue
            queries.append({"task_id": t["id"], "source": MP_SOURCE[ds], "run": run, "split": "train",
                            "orig_task_id": m["orig_task_id"], "lang": t.get("lang") or "en",
                            "grades": {str(k): v for k, v in grades.items()},
                            "ambiguous": idx(g.relevant), "hard_negs": idx(g.seen - g.relevant - set(g.grades)),
                            **_texts(t["title"], list(t.get("parents") or []))})

    for ds, (run, pool) in ADV_RUNS.items():
        if not pool.exists() or not silver.label_path(run).exists():
            continue
        gold = silver.load_gold([run])
        aug = {r["task_id"]: r for r in read_jsonl(AUG_FILES[ds])}
        rw = ({r["task_id"]: r for r in read_jsonl(REWRITE_FILES[ds])}
              if REWRITE_FILES[ds].exists() else {})
        tasks = silver.run_tasks(run)
        for t in tasks[:40] if smoke else tasks:
            m = t.get("meta") or {}
            assert t["id"] not in eval_ids, t["id"]
            g = gold.get(t["id"])
            if g is None or g.no_good_icon:
                stats["adversarial task without a label or a good icon"] += 1
                continue
            grades = {index[n]: v for n, v in g.grades.items() if n in index}
            if not grades:
                continue
            base = {"task_id": t["id"], "source": ADV_SOURCE[ds], "run": run, "split": "train",
                    "grades": {str(k): v for k, v in grades.items()},
                    "ambiguous": idx(g.relevant), "hard_negs": idx(g.seen - g.relevant - set(g.grades)),
                    "seed_task_id": m.get("seed_task_id")}
            parents = list(t.get("parents") or [])
            queries.append({**base, "lang": "en", **_texts(t["title"], parents)})
            stats["adv rows"] += 1
            a = aug.get(t["id"]) or {}
            for lang in ("es", "ru"):
                tr = a.get(lang)
                if not tr or lang in (a.get("contaminated") or []) or _norm(tr["title"]) == _norm(t["title"]):
                    continue
                queries.append({**base, "lang": lang, **_texts(tr["title"], tr.get("parents") or [])})
                stats["adv rows"] += 1
            r = rw.get(t["id"])
            bad = set((r or {}).get("contaminated") or [])
            for k, v in enumerate((r or {}).get("variants") or []):
                title = str(v.get("title", "")).strip()
                if k in bad or not title or _norm(title) == _norm(t["title"]):
                    continue
                lang = v.get("lang", "en")
                par = parents if lang == "en" else list(((a.get(lang) or {}).get("parents")) or [])
                queries.append({**base, "source": "para-" + ADV_SOURCE[ds], "orig_task_id": t["id"],
                                "task_id": f"{t['id']}~rw{k}", "lang": lang, "style": v.get("style"),
                                **_texts(title, par)})
                stats["adv rewrite rows"] += 1

    # Exact contamination check over every training text (the near check ran upstream).
    from .describe_v2 import contamination_hits
    texts = [q["text"] for q in queries]
    exact, _ = contamination_hits(texts, embed=False)
    if exact:
        stats["dropped: equal to an eval title"] += len(exact)
        queries = [q for k, q in enumerate(queries) if k not in exact]

    by = Counter((q["source"], q["split"], q["lang"]) for q in queries)
    manifest = {
        "format": DOC_FORMAT, "base_model": config.MULTILINGUAL_MODEL, "smoke": smoke,
        "query_prefix": QUERY_PREFIX, "passage_prefix": PASSAGE_PREFIX,
        "max_query_len": MAX_QUERY_LEN, "max_passage_len": MAX_PASSAGE_LEN,
        "icons": len(icon_rows), "holdout_icons": sum(r["holdout"] for r in icon_rows),
        "queries": len(queries), "pool_tasks": len({q["task_id"] for q in queries if q["source"].startswith("pool")}),
        "rows_by_source_split_lang": {"/".join(k): v for k, v in sorted(by.items())},
        "notes": dict(stats),
    }
    return {"manifest": manifest, "icons": icon_rows, "queries": queries}


def _texts(title: str, parents: Sequence[str]) -> dict:
    """The bare-title query and, when there is a parent, the "Parent: title" form."""
    out = {"text": query_text(title)}
    if parents:
        out["text_parent"] = query_text(title, parents=parents)
    return out


def _eval_ids() -> frozenset:
    from .datasets import DATASETS, load
    ids = set()
    for key in silver.EVAL_RUNS:
        if DATASETS[key].path.exists():
            ids.update(t["id"] for t in load(key))
    return frozenset(ids)


# --- the matcher -------------------------------------------------------------------------

def model_dir(tag: Optional[str] = None) -> Path:
    """`.cache/models/F1/` for the method F1, `.cache/models/F1-<tag>/` for run F1@<tag>."""
    return MODELS_DIR / ("F1" if not tag else f"F1-{tag}")


class _Encoder:
    """Mean-pooled, L2-normalized e5 encoder; the same as train_f1.py's."""

    def __init__(self, path: Path, device: Optional[str] = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or os.environ.get("TCR_F1_DEVICE", "cpu")
        if self.device == "cpu":
            torch.set_num_threads(int(os.environ.get("TCR_F1_THREADS", "4")))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()

    def encode(self, texts: Sequence[str], max_len: int, batch: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), batch):
                enc = self.tok(list(texts[s:s + batch]), padding=True, truncation=True,
                               max_length=max_len, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


class FineTunedMatcher(Matcher):
    """Cosine of the fine-tuned query vector against precomputed icon vectors."""

    def __init__(self, icons: Sequence[IconDoc], path: Path, name: str = "F1", **kw):
        super().__init__(name=name, **kw)
        if not path.exists():
            raise FileNotFoundError(f"no F1 checkpoint at {path} (docs/f1-colab-run.md)")
        meta_p = path / "f1_meta.json"
        self.meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        self.icons = list(icons)
        self.names = [ic.name for ic in self.icons]
        self.enc = _Encoder(path)
        docs = [PASSAGE_PREFIX + icon_doc(ic) for ic in self.icons]
        self.matrix = self._icon_matrix(path, docs)

    def _icon_matrix(self, path: Path, docs: List[str]) -> np.ndarray:
        h = hashlib.sha1()
        for f in sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin")):
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
        h.update("\n".join(docs).encode())
        cache = config.CACHE_DIR / f"f1__{path.name}__{h.hexdigest()[:16]}.npz"
        if cache.exists():
            data = np.load(cache)
            if list(data["names"]) == self.names:
                return data["matrix"]
        mat = self._precomputed(path, docs)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, names=np.array(self.names), matrix=mat)
        return mat

    def _precomputed(self, path: Path, docs: List[str]) -> np.ndarray:
        """Icon vectors saved by train_f1.py next to the checkpoint (best/icon_matrix.npz),
        for every icon whose document is unchanged; the rest are encoded here."""
        saved = {}
        p = path / "icon_matrix.npz"
        if p.exists():
            d = np.load(p)
            if int(d["max_len"]) == MAX_PASSAGE_LEN:
                mat = d["matrix"]            # read once: every d[key] decompresses again
                saved = {str(n): (str(h), mat[i]) for i, (n, h) in enumerate(zip(d["names"], d["doc_sha1"]))}
        rows, todo = [None] * len(docs), []
        for i, (n, doc) in enumerate(zip(self.names, docs)):
            hit = saved.get(n)
            if hit and hit[0] == hashlib.sha1(doc.encode()).hexdigest():
                rows[i] = hit[1]
            else:
                todo.append(i)
        if todo:
            enc = self.enc.encode([docs[i] for i in todo], MAX_PASSAGE_LEN)
            for i, v in zip(todo, enc):
                rows[i] = v
        return np.stack(rows).astype(np.float32)

    def full_scores(self, query_text_: str):
        q = self.enc.encode([QUERY_PREFIX + query_text_], MAX_QUERY_LEN)[0]
        return self.names, self.matrix @ q
