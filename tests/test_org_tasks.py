from pathlib import Path

from tcr.org_tasks import (guess_lang, iter_org_files, parse_org_tasks, task_id,
                           to_records)

FIXTURE = """\
#+TITLE: Fixture
Preamble prose that belongs to no task.
* Section heading
** TODO [#A] Read the *attention* paper [1/3]                 :online:easy:
SCHEDULED: <2026-05-21 Thu +1d> DEADLINE: <2026-05-30 Sat>
:PROPERTIES:
:ID: 1234
:END:
:LOGBOOK:
CLOCK: [2026-05-20 Wed 10:00]--[2026-05-20 Wed 11:00] =>  1:00
:END:
   Reproduce the results; see [[obsidian:Transformers]].
   https://arxiv.org/abs/1706.03762

*** NEXT Nested task with =code=
    Started: [2026-05-01 Fri]
    Nested body.
** Plain subheading, not a task
Text under a non-task heading.
** DONE Finished thing
** TODO Planning with prose
DEADLINE: <2026-06-01 Mon> Then write it up.
** TODO Прочитать статью
** TODO
"""


def test_parse_titles_keywords_levels():
    tasks = parse_org_tasks(FIXTURE)
    assert [(t.keyword, t.level, t.title) for t in tasks] == [
        ("TODO", 2, "Read the *attention* paper"),
        ("NEXT", 3, "Nested task with =code="),
        ("DONE", 2, "Finished thing"),
        ("TODO", 2, "Planning with prose"),
        ("TODO", 2, "Прочитать статью"),
    ]


def test_tags_split_off_and_body_cleaned():
    t = parse_org_tasks(FIXTURE)[0]
    assert t.tags == ["online", "easy"]
    assert t.body == ("Reproduce the results; see [[obsidian:Transformers]].\n"
                      "https://arxiv.org/abs/1706.03762")


def test_nested_task_body_is_separate():
    tasks = parse_org_tasks(FIXTURE)
    assert "Nested" not in tasks[0].body
    assert tasks[1].body == "Nested body."
    assert tasks[2].body == ""  # non-task heading closes the previous task
    assert tasks[3].body == "Then write it up."  # planning prefix stripped, prose kept


def test_query_text_normalizes_markup_and_links():
    t = parse_org_tasks(FIXTURE)[0]
    assert t.query_text() == "Read the attention paper"
    q = t.query_text(with_body=True)
    assert q.startswith("Read the attention paper. Reproduce the results; see Transformers.")
    assert "online" not in q  # tags never reach the query


def test_ids_stable_and_dedup():
    tasks = parse_org_tasks(FIXTURE + "** TODO Finished thing\n")
    recs = to_records(tasks, "personal", "per")
    assert len(recs) == 5  # duplicate "Finished thing" (same body) dropped
    assert recs[0]["id"] == task_id("per", tasks[0].title, tasks[0].body)
    assert recs[0]["id"] == to_records(tasks, "personal", "per")[0]["id"]
    assert recs[4]["lang"] == "ru" and recs[0]["lang"] == "en"
    assert "file" not in recs[0]


def test_parents_are_ancestor_headings():
    tasks = parse_org_tasks(FIXTURE)
    assert tasks[0].parents == ["Section heading"]
    assert tasks[1].parents == ["Section heading", "Read the *attention* paper"]  # nested task
    assert tasks[2].parents == ["Section heading"]  # sibling of a non-task heading
    recs = to_records(tasks, "personal", "per")
    assert recs[1]["parents"] == tasks[1].parents


def test_guess_lang():
    assert guess_lang("Buy milk") == "en"
    assert guess_lang("Купить молоко") == "ru"


def test_file_filter(tmp_path: Path):
    for rel in ["Inbox.org", "init.org", "workspace.org", "Work/Plan.org",
                "Journal/2026.org", "Archive/Old.org", ".ps/x.org",
                "Work/Plan (conflicted copy 2026-08-19).org"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("* TODO x\n")
    got = [str(p.relative_to(tmp_path)) for p in iter_org_files(tmp_path)]
    assert got == ["Inbox.org", "Work/Plan.org"]
