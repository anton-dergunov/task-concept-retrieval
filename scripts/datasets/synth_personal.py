#!/usr/bin/env python3
"""Personal-synth (D3): a public stand-in for the private task list.

Protocol (design/anonymization.md):
  generate  1. plant canary tasks (random pseudo-words) into a working copy of D1
            2. embed locally (multilingual-e5) and cluster (agglomerative, Ward)
            3. drop clusters with < K_MIN tasks (k-anonymity spirit)
            4. claude-cli writes an ABSTRACTED topic skeleton per cluster, generalizing
               every specific; clusters it judges identifying are dropped
            5. per-cluster counts get Laplace noise; tasks are generated from
               skeleton + aggregate style stats + public style exemplars only
            6. leakage gates (tcr/privacy.py) reject items; keep the first n per cluster
  audit     canary echo, DCR/nearest-neighbour distributions, topic-histogram JS
            divergence and style stats vs D1, and an LLM attribute-inference attacker
            (Vertex Gemini Pro) on {D3, D5 baseline, D1 ceiling} × 3 runs
  publish   copy the reviewed candidates (minus items flagged in the labeller) to
            data/datasets/personal_synth.jsonl — the only step that makes D3 public

Everything intermediate stays in data/private/ (gitignored).
Run: python scripts/datasets/synth_personal.py {generate,audit,publish}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random

import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcr import config  # noqa: E402
from tcr.datasets import DATASETS, read_jsonl, write_jsonl  # noqa: E402
from tcr.index import embed_texts  # noqa: E402
from tcr.llm import batched, complete_json  # noqa: E402
from tcr.org_tasks import query_text  # noqa: E402
from tcr.privacy import LeakageGate, js_divergence, nn_sims  # noqa: E402

from build_public_expanded import MARKUP, SHAPES, SPEC_CODES, weighted  # noqa: E402
from extract_personal import style_stats  # noqa: E402

WORK = config.PRIVATE_DIR / "synth"
CANDIDATES = config.PRIVATE_DIR / "personal_synth.jsonl"   # unreviewed; never committed
PUBLIC_OUT = config.DATASETS_DIR / "personal_synth.jsonl"
EXEMPLARS = config.DATASETS_DIR / "style_exemplars.jsonl"
STYLE_STATS = config.PRIVATE_DIR / "style_stats.json"
ATTRIBUTES = config.PRIVATE_DIR / "attributes.json"
AUDIT_MD = config.PRIVATE_DIR / "privacy_audit.md"
METRICS = config.RESULTS_DIR / "privacy_metrics.json"

SEED = 1234
N_CLUSTERS = 100
K_MIN = 5
TARGET = 1000
OVERGEN = 1.3
LAPLACE_B = 2.0            # count noise scale (sensitivity 1 → ε = 0.5 per cluster count)
N_CANARIES = 15
GEN_MODEL = "sonnet"
ATTACK_MODEL = "gemini-3.1-pro-preview"
ATTACK_N = 500
ATTACK_RUNS = 3


def emb_of(records):
    return embed_texts(config.MULTILINGUAL_MODEL,
                       [query_text(r["title"], r.get("body", ""), with_body=True) for r in records],
                       is_query=True)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- 1. canaries -----------------------------------------------------------------

CANARY_TEMPLATES = [
    ("Read the {A} paper on sparse retrieval before the reading group",
     "Everyone keeps citing the {B} benchmark from it; check whether the gains survive a proper baseline."),
    ("Renew the {A} membership before it lapses", "The discount only applies if renewed through the {B} portal."),
    ("Book the follow-up with Dr {A} about the knee", "Bring the scan from {B} clinic and the list of exercises."),
    ("Draft the {A} migration plan for the data pipeline", "Talk to {B} first — they own the scheduler config."),
    ("Try the {A} sourdough method from the forum", "Uses a {B} starter at 60% hydration; compare with the usual loaf."),
]


def pseudo_word(rng: random.Random) -> str:
    cons, vows = "bdfgklmnprstvz", "aeiou"
    w = "".join(rng.choice(cons) + rng.choice(vows) for _ in range(3)) + rng.choice(cons)
    return w.capitalize()


def make_canaries(rng: random.Random, taken_vocab: set):
    out = []
    for i in range(N_CANARIES):
        a, b = pseudo_word(rng), pseudo_word(rng)
        while a.casefold() in taken_vocab or b.casefold() in taken_vocab:
            a, b = pseudo_word(rng), pseudo_word(rng)
        t, body = CANARY_TEMPLATES[i % len(CANARY_TEMPLATES)]
        out.append({"id": f"canary-{i:02d}", "title": t.format(A=a), "body": body.format(B=b),
                    "canary_tokens": [a, b]})
    return out


# --- 2–4. clusters and skeletons -------------------------------------------------

SKELETON_SYSTEM = """You help build a PUBLIC synthetic dataset from a PRIVATE to-do list without
leaking anything about its owner. For each cluster of similar private tasks you write an
abstract topic skeleton that someone could use to write NEW, different tasks of the same
kind. Generalize every specific (Dou et al. 2024 "self-disclosure abstraction"):
- people's names → a role ("a colleague", "a doctor"); employers, teams, internal project
  or product names → a generic category ("the team's data pipeline");
- places, addresses, schools, clinics → a generic kind ("a local clinic");
- dates, amounts, health details, account or document numbers → drop them;
- rare or unusual words, invented names and code names → drop them entirely;
- widely known PUBLIC topics (e.g. transformers, PyTorch, spaced repetition) may stay.
Mark identifying=true if the cluster is inherently about the owner's identity (a specific
employer's internal work, a specific named person, a unique life event) and cannot be
generalized without losing its point."""

SKELETON_SCHEMA = {"type": "object", "properties": {"clusters": {"type": "array", "items": {
    "type": "object", "properties": {
        "cluster": {"type": "integer"}, "label": {"type": "string"},
        "description": {"type": "string"},
        "task_patterns": {"type": "array", "items": {"type": "string"}},
        "identifying": {"type": "boolean"}},
    "required": ["cluster", "label", "description", "task_patterns", "identifying"]}}},
    "required": ["clusters"]}


def cluster(emb: np.ndarray):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit_predict(emb)


def skeleton_prompt(groups):
    parts = []
    for cid, members in groups:
        lines = "\n".join(f"- {m['title']}" + (f" — {m['body'][:250]}" if m.get("body") else "")
                          for m in members)
        parts.append(f"CLUSTER {cid} ({len(members)} sampled tasks):\n{lines}")
    return ("\n\n".join(parts) + '\n\nFor each cluster return {"cluster", "label" (2–5 words), '
            '"description" (2–3 sentences), "task_patterns" (5–8 generalized patterns), '
            '"identifying"} as JSON {"clusters": [...]}.')


# --- 5. generation ---------------------------------------------------------------

GEN_SYSTEM = """You write realistic personal to-do tasks for a public dataset. You get an abstract
topic skeleton and style examples from one person (on unrelated topics). Write NEW tasks of
that topic in that person's style. Never use real private individuals' names, employers,
addresses or dates; invent ordinary, generic details. Widely known public topics, tools,
papers and websites are fine. Markup is org-mode, never Markdown. Follow each task's
SHAPE and MARKUP instructions exactly, and make every task distinct."""

GEN_SCHEMA = {"type": "object", "properties": {"tasks": {"type": "array", "items": {
    "type": "object", "properties": {"n": {"type": "integer"}, "title": {"type": "string"},
                                     "body": {"type": "string"}},
    "required": ["n", "title", "body"]}}}, "required": ["tasks"]}


def gen_prompt(sk, specs, shots, stats):
    ex = "\n\n".join(f"TITLE: {e['title']}\nBODY:\n{e['body'] or '(empty)'}" for e in shots)
    spec = "\n".join(f"{i + 1}. shape: {s}; markup: {m}" for i, (s, m) in enumerate(specs))
    return f"""Style examples (same person, unrelated topics):

{ex}

The person's titles are usually {stats['title_words']['p25']}–{stats['title_words']['p75']} words (median {stats['title_words']['median']}).

TOPIC SKELETON
label: {sk['label']}
description: {sk['description']}
task patterns:
""" + "\n".join(f"- {p}" for p in sk["task_patterns"]) + f"""

Write {len(specs)} tasks for this topic, one per line below (n = line number):
{spec}

Return JSON {{"tasks": [{{"n", "title", "body"}}]}}."""


def prepare(rng, np_rng):
    """Steps 1–4 (canaries, clusters, skeletons, noisy targets); cached in WORK."""
    private = read_jsonl(DATASETS["personal"].path)
    if (WORK / "targets.json").exists():
        emb = np.load(WORK / "emb.npy")
        return (private, load(WORK / "canaries.json"), emb,
                {int(k): v for k, v in load(WORK / "skeletons.json").items()},
                {int(k): v for k, v in load(WORK / "targets.json").items()})
    WORK.mkdir(parents=True, exist_ok=True)
    vocab = {w.casefold() for r in private for w in (r["title"] + " " + r["body"]).split()}
    canaries = make_canaries(rng, vocab)
    save(WORK / "canaries.json", canaries)
    work = private + canaries

    print(f"embedding {len(work)} tasks …")
    emb = emb_of(work)
    np.save(WORK / "emb.npy", emb)
    labels = cluster(emb)
    by_c = defaultdict(list)
    for r, c in zip(work, labels):
        by_c[int(c)].append(r)
    kept = {c: m for c, m in by_c.items() if len(m) >= K_MIN}
    print(f"{len(by_c)} clusters; {len(kept)} with >= {K_MIN} tasks "
          f"({sum(len(m) for m in kept.values())} tasks)")

    # Skeletons: up to 30 sampled members per cluster; canaries are always included so
    # the echo test actually exercises them.
    groups = []
    for c in sorted(kept):
        m = kept[c]
        can = [r for r in m if r["id"].startswith("canary")]
        rest = [r for r in m if not r["id"].startswith("canary")]
        groups.append((c, can + rng.sample(rest, min(len(rest), 30 - len(can)))))

    def sk_call(b):
        res = complete_json(skeleton_prompt(b), system=SKELETON_SYSTEM, schema=SKELETON_SCHEMA,
                            backend="claude-cli", models=[GEN_MODEL], private=True,
                            validate=lambda d: {x["cluster"] for x in d["clusters"]} == {c for c, _ in b})
        return res["data"]["clusters"]

    with ThreadPoolExecutor(3) as ex:
        skeletons = {x["cluster"]: x for batch in ex.map(sk_call, batched(groups, 5)) for x in batch}
    save(WORK / "skeletons.json", skeletons)
    usable = [c for c, s in skeletons.items() if not s["identifying"]]
    print(f"skeletons: {len(skeletons)}; {len(skeletons) - len(usable)} judged identifying and dropped")

    # Noisy per-cluster targets, proportional to cluster size.
    total = sum(len(kept[c]) for c in usable)
    targets = {c: max(1, int(round(len(kept[c]) * TARGET / total + np_rng.laplace(0, LAPLACE_B))))
               for c in usable}
    save(WORK / "targets.json", targets)
    return private, canaries, emb, skeletons, targets


def generate_round(rnd, deficits, skeletons, rng, stats, exemplars, workers):
    """Generate OVERGEN × deficit tasks per cluster (tag = round, so rounds differ)."""
    jobs = []
    for c, need in sorted(deficits.items()):
        n_gen = int(np.ceil(need * (OVERGEN if rnd == 0 else 2.0)))
        specs = [(weighted(rng, SHAPES), weighted(rng, MARKUP)) for _ in range(n_gen)]
        specs = [(s, "no markup" if s.startswith("no body") else m) for s, m in specs]
        for k, chunk in enumerate(batched(specs, 12)):
            shots = random.Random(c * 100 + k + 7919 * rnd).sample(exemplars, 6)
            jobs.append((c, k, chunk, shots))

    def gen_call(job):
        c, k, chunk, shots = job
        res = complete_json(gen_prompt(skeletons[c], chunk, shots, stats), system=GEN_SYSTEM,
                            schema=GEN_SCHEMA, backend="claude-cli", models=[GEN_MODEL],
                            tag=f"round{rnd}" if rnd else "",
                            validate=lambda d: len(d["tasks"]) == len(chunk))
        return [{"cluster": c, "round": rnd, "title": t["title"].strip(), "body": t["body"].strip(),
                 "shape": chunk[i][0], "markup": chunk[i][1], "gen_model": res["model"]}
                for i, t in enumerate(sorted(res["data"]["tasks"], key=lambda t: t["n"]))
                if i < len(chunk)]

    print(f"round {rnd}: generating {sum(len(j[2]) for j in jobs)} tasks in {len(jobs)} calls …")
    out = []
    with ThreadPoolExecutor(workers) as ex:
        for i, items in enumerate(ex.map(gen_call, jobs)):
            out += items
            print(f"  {i + 1}/{len(jobs)} calls", end="\r")
    print()
    return out


TERMS_SYSTEM = """You audit words before a dataset is published. Each word below is rare in a
person's private notes. Classify each: "public" if it is a widely known technical term,
library, model, dataset, algorithm, metric, abbreviation, textbook author or public figure
that anyone could use; "private" if it could be a personal name, an internal or code name,
a small organisation, a place, or you are not sure. Be conservative: unsure means private."""

TERMS_SCHEMA = {"type": "object", "properties": {"words": {"type": "array", "items": {
    "type": "object", "properties": {"word": {"type": "string"},
                                     "label": {"type": "string", "enum": ["public", "private"]}},
    "required": ["word", "label"]}}}, "required": ["words"]}


def public_term_allowlist(generated, private):
    """LLM vocabulary audit: rare private tokens that occur in generated text and are
    widely known public terms (fp16, reranker, Jurafsky) are allowed; the rest stay blocked."""
    from tcr.privacy import RARE_DF, english_vocab, words
    df = Counter()
    for r in private:
        df.update(set(words(r["title"] + "\n" + r["body"])))
    big = english_vocab(500000)
    cand = sorted({w for g in generated for w in words(g["title"] + " " + g["body"])
                   if df[w] and df[w] <= RARE_DF and len(w) > 2 and not w.isdigit() and w not in big})
    allowed = set()
    for batch in batched(cand, 150):
        res = complete_json("Classify these words:\n" + "\n".join(batch)
                            + '\nReturn JSON {"words": [{"word", "label"}]}.',
                            system=TERMS_SYSTEM, schema=TERMS_SCHEMA, backend="claude-cli",
                            models=[GEN_MODEL], private=True,
                            validate=lambda d: {x["word"] for x in d["words"]} >= set(batch))
        allowed |= {x["word"] for x in res["data"]["words"] if x["label"] == "public"}
    save(WORK / "term_audit.json", {"candidates": len(cand), "public": len(allowed)})
    return allowed


def apply_gates(generated, private, emb, canaries, targets):
    public_texts = [" ".join(sorted(public_term_allowlist(generated, private)))]
    public_texts += [r["title"] + " " + r["body"] for r in read_jsonl(DATASETS["public_short"].path)]
    if DATASETS["public_expanded"].path.exists():
        public_texts += [json.dumps(r["meta"].get("parallel", {}), ensure_ascii=False)
                         for r in read_jsonl(DATASETS["public_expanded"].path)]
    public_texts += [p.read_text(encoding="utf-8") for p in config.ICON_DESC_DIR.glob("*.json")]
    public_texts += [e["title"] + " " + e["body"] for e in read_jsonl(EXEMPLARS)]
    gate = LeakageGate(private, public_texts, private_emb=emb[:len(private)], seed=SEED)
    gen_emb = emb_of(generated)
    report, per_c = Counter(), Counter()
    out, rejected, seen = [], [], set()
    for g, e in zip(generated, gen_emb):
        key = hashlib.sha1((g["title"] + "\n" + g["body"]).encode()).hexdigest()[:10]
        if key in seen:
            continue
        seen.add(key)
        fired = gate.check(g, e)
        canary_hit = [tok for cn in canaries for tok in cn["canary_tokens"]
                      if tok.casefold() in (g["title"] + " " + g["body"]).casefold()]
        if canary_hit:
            fired["canary"] = len(canary_hit)
        for k in fired:
            report[k] += 1
        if fired:
            rejected.append({**g, "gates": fired})
            continue
        if per_c[g["cluster"]] >= targets[g["cluster"]]:
            continue
        per_c[g["cluster"]] += 1
        out.append({"id": f"syn-{key}", "dataset": "personal_synth", "title": g["title"],
                    "body": g["body"], "lang": "en",
                    "meta": {"cluster": g["cluster"], "shape": SPEC_CODES[g["shape"]],
                             "markup": SPEC_CODES[g["markup"]],
                             "gen_model": g["gen_model"]}})
    stats = {"generated": len(seen), "rejected": len(rejected), "kept": len(out), "tau": gate.tau,
             "fired": dict(report), "rare_vocab": len(gate.rare), "private_grams": len(gate.grams)}
    return out, rejected, stats, {c: targets[c] - per_c[c] for c in targets if per_c[c] < targets[c]}


def cmd_generate(args) -> None:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    stats = load(STYLE_STATS)
    exemplars = [e for e in read_jsonl(EXEMPLARS) if e["lang"] == "en"]
    private, canaries, emb, skeletons, targets = prepare(rng, np_rng)
    usable = {c: s for c, s in skeletons.items() if not s["identifying"]}

    all_path = WORK / "generated_all.jsonl"
    generated = read_jsonl(all_path) if all_path.exists() else []
    rnd = 1 + max((g.get("round", 0) for g in generated), default=-1)
    if not generated:
        generated = generate_round(0, targets, usable, rng, stats, exemplars, args.workers)
        write_jsonl(all_path, generated)
        rnd = 1
    while True:
        out, rejected, gstats, deficits = apply_gates(generated, private, emb, canaries, targets)
        print(f"after round {rnd - 1}: {gstats['kept']} kept of {gstats['generated']}; "
              f"gates fired {gstats['fired']} (τ={gstats['tau']:.3f}); "
              f"deficit {sum(deficits.values())} in {len(deficits)} clusters")
        if not deficits or rnd > args.rounds:
            break
        generated += generate_round(rnd, deficits, usable, random.Random(SEED + rnd), stats,
                                    exemplars, args.workers)
        write_jsonl(all_path, generated)
        rnd += 1

    random.Random(SEED).shuffle(out)
    write_jsonl(CANDIDATES, out)
    write_jsonl(WORK / "rejected.jsonl", rejected)
    save(WORK / "gate_report.json", gstats)
    print(f"{len(out)} candidates -> {CANDIDATES}")


# --- audit -----------------------------------------------------------------------

ATTRIBUTES_LIST = ["country of residence", "city or region of residence", "place of birth / origin",
                   "native language", "other languages", "profession / job title", "seniority",
                   "employer or industry", "education (level and field)", "age range", "gender",
                   "relationship status", "children", "hobbies and interests",
                   "health conditions", "income level"]

ATTACK_SYSTEM = """You are a privacy auditor running an attribute-inference attack (Staab et al.
2024). Given a person's to-do list, infer as much as you can about the person. Use every clue:
topics, vocabulary, spelling, tools, places, time references, what is and is not mentioned."""

ATTACK_SCHEMA = {"type": "object", "properties": {"attributes": {"type": "array", "items": {
    "type": "object", "properties": {"attribute": {"type": "string"}, "guess": {"type": "string"},
                                     "confidence": {"type": "integer"}, "evidence": {"type": "string"}},
    "required": ["attribute", "guess", "confidence", "evidence"]}}}, "required": ["attributes"]}


def attack(name, records, run):
    rng = random.Random(f"{name}-{run}")
    sample = rng.sample(records, min(ATTACK_N, len(records)))
    listing = "\n".join(f"- {r['title']}" + (f"\n  {r['body']}" if r.get("body") else "") for r in sample)
    prompt = (f"To-do list ({len(sample)} tasks):\n\n{listing}\n\nFor each attribute — "
              + "; ".join(ATTRIBUTES_LIST)
              + " — give your best guess, confidence 1 (pure guess) to 5 (certain), and the evidence. "
              'Return JSON {"attributes": [{"attribute", "guess", "confidence", "evidence"}]}.')
    res = complete_json(prompt, system=ATTACK_SYSTEM, schema=ATTACK_SCHEMA, backend="vertex",
                        models=[ATTACK_MODEL], temperature=0.7, private=(name == "personal"),
                        tag=f"attack-{name}-{run}",
                        validate=lambda d: len(d["attributes"]) >= len(ATTRIBUTES_LIST) - 2)
    return res["data"]["attributes"]


def cmd_audit(args) -> None:
    private = read_jsonl(DATASETS["personal"].path)
    synth = read_jsonl(CANDIDATES)
    expanded = read_jsonl(DATASETS["public_expanded"].path)
    en_expanded = [{"title": r["meta"]["parallel"]["en"]["title"],
                    "body": r["meta"]["parallel"]["en"]["body"]} for r in expanded]
    canaries = load(WORK / "canaries.json")
    gate_report = load(WORK / "gate_report.json")
    skeletons = load(WORK / "skeletons.json")
    metrics = {"gates": gate_report}

    # Canary echo: in skeletons (what the generator saw) and in every generated item.
    tokens = [t.casefold() for c in canaries for t in c["canary_tokens"]]
    sk_text = json.dumps(skeletons, ensure_ascii=False).casefold()
    gen_text = "\n".join(g["title"] + " " + g["body"] for g in read_jsonl(WORK / "generated_all.jsonl")).casefold()
    metrics["canary"] = {"planted": len(canaries), "tokens": len(tokens),
                         "echo_in_skeletons": sum(t in sk_text for t in tokens),
                         "echo_in_generated": sum(t in gen_text for t in tokens)}

    # Nearest-neighbour (DCR-style) distributions.
    e_priv, e_syn, e_exp = emb_of(private), emb_of(synth), emb_of(en_expanded)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(e_priv))
    half = len(idx) // 2
    train, hold = e_priv[idx[half:]], e_priv[idx[:half]]
    q = [0.5, 0.9, 0.95, 0.99]
    dist = {"holdout→train": nn_sims(hold, train), "synth→train": nn_sims(e_syn, train),
            "expanded→train": nn_sims(e_exp, train)}
    metrics["nn_similarity"] = {k: {f"q{int(p * 100)}": round(float(np.quantile(v, p)), 3) for p in q}
                                for k, v in dist.items()}
    # DCR share: how often a synthetic item is closer to the train half than to the
    # holdout half. ~0.5 means no preference for the data the generator "saw" — note
    # the skeletons saw both halves, so this is a sanity check, not a membership test.
    metrics["dcr_share_train"] = round(float((nn_sims(e_syn, train) > nn_sims(e_syn, hold)).mean()), 3)

    # Topic fidelity: histogram over private-cluster centroids.
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=50, random_state=SEED, n_init=4).fit(train)
    hist = lambda e: np.bincount(km.predict(e), minlength=50).astype(float) + 1e-9  # noqa: E731
    metrics["topic_js_divergence_vs_train"] = {
        "holdout (ideal)": round(js_divergence(hist(hold), hist(train)), 4),
        "personal_synth": round(js_divergence(hist(e_syn), hist(train)), 4),
        "public_expanded (baseline)": round(js_divergence(hist(e_exp), hist(train)), 4)}
    metrics["style"] = {"personal": style_stats(private), "personal_synth": style_stats(synth)}

    # Attribute-inference attacker.
    sets = {"personal_synth": synth, "public_expanded": en_expanded, "personal": private}
    jobs = [(n, run) for n in sets for run in range(ATTACK_RUNS)]
    with ThreadPoolExecutor(args.workers) as ex:
        results = dict(zip(jobs, ex.map(lambda j: attack(j[0], sets[j[0]], j[1]), jobs)))
    save(WORK / "attacks.json", {f"{n}#{r}": v for (n, r), v in results.items()})
    if not ATTRIBUTES.exists():
        save(ATTRIBUTES, {a: "" for a in ATTRIBUTES_LIST})

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps({k: v for k, v in metrics.items() if k != "style"}, indent=1))
    write_audit_md(metrics, results)
    print(f"public-safe metrics -> {METRICS}\nfull audit (private) -> {AUDIT_MD}")
    print(json.dumps({k: metrics[k] for k in ("canary", "nn_similarity", "dcr_share_train",
                                              "topic_js_divergence_vs_train")}, indent=1))


def write_audit_md(metrics, results) -> None:
    truth = load(ATTRIBUTES) if ATTRIBUTES.exists() else {}
    L = ["# Personal-synth privacy audit (PRIVATE — do not commit)", "",
         "## Gates", "", "```", json.dumps(metrics["gates"], indent=1), "```", "",
         "## Canaries", "", "```", json.dumps(metrics["canary"], indent=1), "```", "",
         "## Nearest-neighbour similarity to private train half", "", "```",
         json.dumps(metrics["nn_similarity"], indent=1), f"dcr_share_train: {metrics['dcr_share_train']}",
         "```", "", "## Topic JS divergence", "", "```",
         json.dumps(metrics["topic_js_divergence_vs_train"], indent=1), "```", "",
         "## Attribute inference", "",
         "Fill `data/private/attributes.json` with the true values, then compare. "
         "Guesses are shown as `guess (confidence)`, one column per run.", ""]
    for attr in ATTRIBUTES_LIST:
        L.append(f"### {attr}" + (f" — truth: {truth[attr]}" if truth.get(attr) else ""))
        L.append("")
        L.append("| set | run 1 | run 2 | run 3 |")
        L.append("|---|---|---|---|")
        for name in ("personal", "personal_synth", "public_expanded"):
            cells = []
            for run in range(ATTACK_RUNS):
                hit = next((a for a in results[(name, run)] if a["attribute"].casefold() == attr.casefold()), None)
                cells.append(f"{hit['guess']} ({hit['confidence']})".replace("|", "/") if hit else "–")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    AUDIT_MD.write_text("\n".join(L), encoding="utf-8")


# --- publish ---------------------------------------------------------------------

def cmd_publish(args) -> None:
    if not AUDIT_MD.exists():
        raise SystemExit("run `audit` first")
    # Labels made while the set was still private, if any.
    labels_path = config.PRIVATE_DIR / "labels" / "personal_synth.jsonl"
    flagged = set()
    if labels_path.exists():
        latest = {}
        for rec in read_jsonl(labels_path):
            latest[rec["task_id"]] = rec
        flagged = {t for t, r in latest.items() if r["status"] == "flagged"}
    out = [r for r in read_jsonl(CANDIDATES) if r["id"] not in flagged]
    n = write_jsonl(PUBLIC_OUT, out)
    print(f"{n} tasks ({len(flagged)} flagged removed) -> {PUBLIC_OUT}")
    if labels_path.exists():
        dest = config.DATA_DIR / "labels" / "personal_synth.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(labels_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"labels -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "audit", "publish"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2, help="top-up rounds for clusters short of target")
    args = ap.parse_args()
    {"generate": cmd_generate, "audit": cmd_audit, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    main()
