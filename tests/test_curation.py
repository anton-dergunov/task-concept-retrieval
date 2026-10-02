"""The icon review's event log (labeller/server.py Curation, tcr.data.removed_by_hand)."""

import importlib.util
import json
from pathlib import Path

import pytest

from tcr.data import removed_by_hand

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("labeller_server", ROOT / "labeller" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@pytest.fixture
def dirs(tmp_path):
    bundle, labels = tmp_path / "bundle", tmp_path / "labels"
    bundle.mkdir()
    (bundle / "curation.json").write_text(json.dumps(
        {"build": "b", "cols": 2, "rows": 2, "per_sheet": 4,
         "icons": ["a", "b", "c"], "groups": [[0, 2], [2, 1]]}))
    return bundle, labels


def test_latest_event_per_icon_wins(dirs):
    cur = server.Curation(*dirs)
    assert cur.state()["removed"] == []
    cur.save({"events": [{"icon": "a", "removed": True}, {"icon": "c", "removed": True}]})
    assert cur.save({"events": [{"icon": "a", "removed": False}]}) == {"removed": 1}
    assert cur.state()["removed"] == ["c"]
    assert cur.state()["groups"] == [[0, 2], [2, 1]]


def test_log_survives_restart_and_matches_the_repo_reader(dirs):
    cur = server.Curation(*dirs)
    cur.save({"events": [{"icon": "b", "removed": True, "bulk": True},
                         {"icon": "a", "removed": True}], "client": "iPad"})
    cur.save({"events": [{"icon": "b", "removed": False}]})
    log = dirs[1] / server.CURATION_LOG
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["icon"], r["removed"], r["bulk"], r["client"]) for r in recs] == [
        ("b", True, True, "iPad"), ("a", True, False, "iPad"), ("b", False, False, "")]
    assert server.Curation(*dirs).state()["removed"] == ["a"]
    assert removed_by_hand(log) == {"a"}


def test_unknown_icon_rejects_the_whole_batch(dirs):
    cur = server.Curation(*dirs)
    with pytest.raises(KeyError):
        cur.save({"events": [{"icon": "a", "removed": True}, {"icon": "nope", "removed": True}]})
    assert cur.state()["removed"] == []
    assert not (dirs[1] / server.CURATION_LOG).exists()
