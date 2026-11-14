"""Silver gold and metrics (tcr/silver.py) on hand-made rankings."""

import math

from tcr import silver
from tcr.data import glyph_hashes, rejected_names


def _gold(grades, seen=()):
    return silver.Gold("t", dict(grades), frozenset(seen) | frozenset(grades), list(grades))


def test_graded_ndcg():
    g = {"a": 3, "b": 2, "c": 1}
    assert silver.ndcg(g, ["a", "b", "c"]) == 1.0
    want = (2 / math.log2(2) + 3 / math.log2(3)) / (3 + 2 / math.log2(3) + 1 / math.log2(4))
    assert math.isclose(silver.ndcg(g, ["b", "a", "x"]), want)
    assert silver.ndcg({}, ["a"]) == 0.0


def test_aliases_share_a_name():
    hashes = glyph_hashes()
    # alarm / access_alarm draw the same glyph (tcr/data.py)
    assert hashes["alarm"] == hashes["access_alarm"]
    assert silver.canonical("alarm") == silver.canonical("access_alarm")


def test_rejected_glyph_scored_and_served():
    rej = sorted(rejected_names())
    assert "speed_1_75" in rej and "1x_mobiledata" in rej
    bad = silver.canonical("speed_1_75")
    g = _gold({"calculate": 3, "functions": 2}, seen={bad})
    m = silver.task_metrics(g, [bad, "calculate", "functions"])
    assert m["rej1"] == 1 and m["hit1"] == 0 and m["judged1"] == 1
    assert m["hit1_served"] == 1 and m["ndcg5_served"] == 1.0
    m = silver.task_metrics(g, ["zzz_unseen", "calculate"])
    assert m["judged1"] == 0 and m["hit1_judged"] is None


def test_rejected_icons_left_out_of_gold():
    gold = silver.load_gold(["realistic-j4"])
    rej = silver.rejected_canonical()
    assert gold and not any(set(g.grades) & rej for g in gold.values())


def test_mcnemar_exact():
    assert silver.mcnemar_exact(0, 0) == 1.0
    assert math.isclose(silver.mcnemar_exact(0, 5), 2 / 32)
    assert silver.mcnemar_exact(3, 3) == 1.0
