"""The icon review's event log (labeller/server.py Curation, tcr.data.icon_labels)."""

import importlib.util
import json
from pathlib import Path

import pytest

from tcr.data import CURATION_LABELS, icon_labels

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


def test_levels_are_the_same_on_both_sides():
    assert server.CURATION_LABELS == CURATION_LABELS


def test_latest_event_per_icon_wins(dirs):
    cur = server.Curation(*dirs)
    assert cur.state()["labels"] == {}
    cur.save({"events": [{"icon": "a", "label": "reject"}, {"icon": "c", "label": "useful"},
                         {"icon": "b", "label": "somewhat"}]})
    counts = cur.save({"events": [{"icon": "a", "label": "useful"}, {"icon": "b", "label": None}]})
    assert counts == {"useful": 2}
    assert cur.state()["labels"] == {"a": "useful", "c": "useful"}
    assert cur.state()["groups"] == [[0, 2], [2, 1]]


def test_log_survives_restart_and_matches_the_repo_reader(dirs):
    cur = server.Curation(*dirs)
    cur.save({"events": [{"icon": "b", "label": "useful", "bulk": True},
                         {"icon": "a", "label": "somewhat"}], "client": "iPad"})
    cur.save({"events": [{"icon": "b", "label": None}]})
    log = dirs[1] / server.CURATION_LOG
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["icon"], r["label"], r["bulk"], r["client"]) for r in recs] == [
        ("b", "useful", True, "iPad"), ("a", "somewhat", False, "iPad"), ("b", None, False, "")]
    assert server.Curation(*dirs).state()["labels"] == {"a": "somewhat"}
    assert icon_labels(log) == {"a": "somewhat"}


def test_two_state_events_still_read_as_reject(dirs):
    """Logs and queued changes from before the three levels only say `removed`."""
    dirs[1].mkdir()
    log = dirs[1] / server.CURATION_LOG
    log.write_text('{"icon": "a", "removed": true}\n{"icon": "b", "removed": true}\n'
                   '{"icon": "b", "removed": false}\n')
    cur = server.Curation(*dirs)
    assert cur.state()["labels"] == {"a": "reject"} == icon_labels(log)
    cur.save({"events": [{"icon": "c", "removed": True}]})
    assert json.loads(log.read_text().splitlines()[-1])["label"] == "reject"


@pytest.mark.parametrize("event, error", [
    ({"icon": "nope", "label": "reject"}, KeyError),
    ({"icon": "b", "label": "great"}, ValueError),
])
def test_a_bad_event_rejects_the_whole_batch(dirs, event, error):
    cur = server.Curation(*dirs)
    with pytest.raises(error):
        cur.save({"events": [{"icon": "a", "label": "reject"}, event]})
    assert cur.state()["labels"] == {}
    assert not (dirs[1] / server.CURATION_LOG).exists()
