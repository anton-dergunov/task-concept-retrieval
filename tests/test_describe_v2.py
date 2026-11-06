"""The v2 description request: validation, references, no icon names, cache keys."""

import copy

import pytest

from tcr import describe_v2 as dv2
from tcr import llm

GOOD = {
    "depiction": {"primary_object": "bathroom scale with a display", "modifiers": ["three dots"],
                  "composition": "single object, front view"},
    "readings": [{"meaning": "body weight", "strength": "primary"}],
    "ambiguity": "",
    "modifier_semantics": [],
    "domains": ["health_fitness", "home"],
    "task_intents": [f"track weight {i}" for i in range(6)],
    "example_tasks": (
        [{"task": "Weigh in", "parent": "", "domain": "health_fitness"}] * 5
        + [{"task": "Log my weight every Monday in the sheet", "parent": "", "domain": "home"}] * 4
        + [{"task": "Weigh the suitcase", "parent": "Trip to Lisbon", "domain": "travel"}] * 4
        + [{"task": "Buy a scale " + "x " * 9, "parent": "", "domain": "shopping"}] * 3),
    "example_tasks_es": ["Pesarme el lunes", "Comprar báscula", "Pesar la maleta"],
    "example_tasks_ru": ["Взвеситься утром", "Купить весы", "Взвесить чемодан"],
    "context_tasks": [{"task": "Log calories", "central": "food"}] * 3,
    "near_misses": [{"task": "Buy a heavy jacket", "better": "jacket", "trap": "property"},
                    {"task": "Check the sync", "better": "sync", "trap": "generic_verb"},
                    {"task": "Practice piano scales", "better": "piano", "trap": "shared_vocabulary"},
                    {"task": "Reply to the chat", "better": "chat bubble", "trap": "lookalike_glyph"},
                    {"task": "Track the parcel", "better": "parcel", "trap": "shared_vocabulary"}],
    "scores": {"recognizability": 4, "task_fit": 4, "distinctiveness": 3},
    "discard": False,
    "reasoning": "A scale.",
}


def test_good_response_has_no_problems():
    assert dv2.hard_problems(GOOD) == []
    assert dv2.soft_problems(GOOD) == []


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: d.pop("scores"), "missing scores"),
    (lambda d: d["scores"].update(task_fit=7), "scores.task_fit invalid"),
    (lambda d: d["scores"].update(task_fit=True), "scores.task_fit invalid"),
    (lambda d: d.update(discard="no"), "discard not boolean"),
    (lambda d: d["depiction"].update(primary_object=" "), "depiction.primary_object empty"),
    (lambda d: d.update(example_tasks=d["example_tasks"][:3]), "fewer than 8 example_tasks"),
    (lambda d: d["readings"][0].update(meaning="card, " * 200), "runaway string of 1200 characters"),
])
def test_hard_problems(mutate, expect):
    d = copy.deepcopy(GOOD)
    mutate(d)
    assert expect in dv2.hard_problems(d)


def test_soft_problems():
    d = copy.deepcopy(GOOD)
    d["example_tasks_ru"] = ["Weigh myself", "Купить весы", "Взвесить чемодан"]
    d["near_misses"] = [dict(n, trap="property") for n in d["near_misses"]]
    probs = " ".join(dv2.soft_problems(d))
    assert "example_tasks_ru not all Cyrillic" in probs
    assert "traps (< 3)" in probs


def test_references_are_deterministic_and_never_eval():
    a, b = dv2.references_for("glyph-1"), dv2.references_for("glyph-1")
    assert a == b and len(a) == 8
    assert dv2.references_for("glyph-2") != a
    banned = {dv2.norm(t) for t in dv2.eval_titles()}
    short, long_ = dv2.reference_pool()
    assert not any(dv2.norm(t) in banned for t in short + long_)


def test_request_never_names_the_icon():
    for name in ("monitor_weight", "electric_meter", "add_shopping_cart"):
        system, prompt, images = dv2.build_request(name)
        text = (system + prompt).lower()
        assert name not in text and name.replace("_", " ") not in text
        assert images and images[0][1] == "image/png" and images[0][0][:4] == b"\x89PNG"


def test_cache_key_unchanged_without_images():
    """Text-only calls keep the keys their responses were cached under before
    images / generation options existed."""
    import hashlib
    import json
    old = json.dumps(["gemini", "sys", "prompt", None], ensure_ascii=False, sort_keys=True)
    expected = llm.CACHE_DIR / (hashlib.sha1(old.encode("utf-8")).hexdigest() + ".json")
    assert llm._cache_path("gemini", "sys", "prompt", None) == expected
    assert llm._cache_path("gemini", "sys", "prompt", None, {"images": ["x"], "gen": {}}) != expected
