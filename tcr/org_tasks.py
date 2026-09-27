"""Extract tasks (title + body) from org files into dataset records.

Unlike `org_query.parse_heading` (heading-only, for ad-hoc queries), this scans
whole files and keeps each task's body. Title and body keep their org markup —
the labelling UI renders it, and matchers normalize it via `query_text`.

What is dropped, because it describes the task's *state/schedule*, not its idea:
the TODO keyword, priority cookie, statistics cookies, tags (kept aside in
`tags`), planning lines (SCHEDULED/DEADLINE/CLOSED, custom `Started:`), CLOCK
lines and drawers.

A task's body runs until the next heading of any level, so nested tasks become
separate records and never leak into their parent's body.

`parents` holds the titles of the task's ancestor headings (outermost first): the
project or section a task sits under often carries its topic ("Music catalog support"
› "Evaluate bids against my catalog rules"). The file name is not included.
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Sequence

from .org_query import TODO_KEYWORDS, _PRIORITY_RE, _TAGS_RE, normalize_text

_HEADING_RE = re.compile(r"^(\*+)\s+(.*?)\s*$")
_STATS_COOKIE_RE = re.compile(r"\s*\[\d*(?:/\d*|%)\]")
# Planning stamps at the start of a line; any prose after them is kept.
_PLANNING_PREFIX_RE = re.compile(
    r"^\s*(?:(?:SCHEDULED|DEADLINE|CLOSED):\s*[<\[][^>\]]*[>\]]\s*)+")
_STARTED_LINE_RE = re.compile(r"^\s*Started:\s*[<\[][^>\]]*[>\]]\s*$")
_CLOCK_LINE_RE = re.compile(r"^\s*CLOCK:")
_DRAWER_START_RE = re.compile(r"^\s*:[A-Za-z][\w-]*:\s*$")
_DRAWER_END_RE = re.compile(r"^\s*:END:\s*$", re.IGNORECASE)

# Mirrors productivity-system's ps-org-files.el: files the agenda never scans.
_EXCLUDED_FILE_RES = [re.compile(r"^init\.org$"), re.compile(r"^workspace\.org$"),
                      re.compile(r"conflicted copy")]
_EXCLUDED_DIR_RES = [re.compile(r"^\."), re.compile(r"^Journal$"), re.compile(r"^Archive$")]


@dataclass
class OrgTask:
    title: str                      # org markup kept
    body: str                       # org markup kept, planning/drawers removed
    tags: List[str] = field(default_factory=list)
    keyword: str = "TODO"
    level: int = 1
    parents: List[str] = field(default_factory=list)   # ancestor heading titles, outermost first

    def query_text(self, with_body: bool = False, body_chars: int = 400,
                   with_parent: bool = False) -> str:
        return query_text(self.title, self.body, with_body, body_chars,
                          self.parents if with_parent else ())


def query_text(title: str, body: str = "", with_body: bool = False, body_chars: int = 400,
               parents: Sequence[str] = ()) -> str:
    """Plain text for matchers: normalized title, optionally prefixed by the nearest
    parent heading ("Music catalog support: Evaluate bids …") and/or + a body prefix."""
    title = normalize_text(title)
    if parents:
        parent = normalize_text(parents[-1])
        title = f"{parent}: {title}" if parent else title
    if not with_body or not body:
        return title
    body = normalize_text(body.replace("\n", " "))[:body_chars]
    return f"{title}. {body}" if title else body


def _split_heading(rest: str):
    """`rest` = heading text after the stars. Returns (keyword, title, tags) or None."""
    parts = rest.split(None, 1)
    if not parts or parts[0] not in TODO_KEYWORDS:
        return None
    keyword = parts[0]
    title = parts[1] if len(parts) > 1 else ""
    tags: List[str] = []
    mtags = _TAGS_RE.search(title)
    if mtags:
        tags = [t for t in mtags.group(1).strip(":").split(":") if t]
        title = title[: mtags.start()]
    title = _PRIORITY_RE.sub("", title.strip())
    title = _STATS_COOKIE_RE.sub("", title)
    return keyword, re.sub(r"\s+", " ", title).strip(), tags


def _clean_body(lines: List[str]) -> str:
    kept: List[str] = []
    in_drawer = False
    for line in lines:
        if in_drawer:
            if _DRAWER_END_RE.match(line):
                in_drawer = False
            continue
        if _DRAWER_START_RE.match(line) and not _DRAWER_END_RE.match(line):
            in_drawer = True
            continue
        if _STARTED_LINE_RE.match(line) or _CLOCK_LINE_RE.match(line):
            continue
        m = _PLANNING_PREFIX_RE.match(line)
        if m:
            line = line[:len(line) - len(line.lstrip())] + line[m.end():]
            if not line.strip():
                continue
        kept.append(line.rstrip())
    text = textwrap.dedent("\n".join(kept)).strip("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _heading_title(rest: str) -> str:
    """Clean title of ANY heading (task or not): no keyword, priority, cookies or tags."""
    split = _split_heading(rest)
    if split:
        return split[1]
    mtags = _TAGS_RE.search(rest)
    if mtags:
        rest = rest[: mtags.start()]
    rest = _STATS_COOKIE_RE.sub("", _PRIORITY_RE.sub("", rest.strip()))
    return re.sub(r"\s+", " ", rest).strip()


def parse_org_tasks(text: str) -> List[OrgTask]:
    tasks: List[OrgTask] = []
    current = None          # (level, keyword, title, tags, parents) of the open task
    body: List[str] = []
    stack: List[tuple] = []  # (level, title) of the enclosing headings

    def flush():
        if current is not None:
            level, keyword, title, tags, parents = current
            tasks.append(OrgTask(title=title, body=_clean_body(body), tags=tags,
                                 keyword=keyword, level=level, parents=parents))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            body = []
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            parents = [t for _, t in stack if t]
            split = _split_heading(m.group(2))
            current = (level,) + split + (parents,) if split else None
            stack.append((level, _heading_title(m.group(2))))
            continue
        if current is not None:
            body.append(line)
    flush()
    return [t for t in tasks if t.title or t.body]


def is_task_file(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(rx.search(d) for d in rel.parts[:-1] for rx in _EXCLUDED_DIR_RES):
        return False
    return not any(rx.search(rel.name) for rx in _EXCLUDED_FILE_RES)


def iter_org_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.org")):
        if path.is_file() and is_task_file(path, root):
            yield path


def task_id(prefix: str, title: str, body: str) -> str:
    """Stable id from content, so labels survive re-extraction and reordering."""
    key = normalize_text(title).casefold() + "\n" + re.sub(r"\s+", " ", body).strip()
    return f"{prefix}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]}"


def guess_lang(text: str) -> str:
    """Coarse language tag: 'ru' if mostly Cyrillic letters, else 'en'."""
    letters = [c for c in text if c.isalpha()]
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return "ru" if letters and cyr / len(letters) > 0.3 else "en"


def to_records(tasks: List[OrgTask], dataset: str, prefix: str,
               with_parents: bool = True) -> List[dict]:
    """Dataset records in the shared schema, deduplicated by id (first wins)."""
    seen = set()
    records = []
    for t in tasks:
        tid = task_id(prefix, t.title, t.body)
        if tid in seen:
            continue
        seen.add(tid)
        rec = {"id": tid, "dataset": dataset, "title": t.title, "body": t.body}
        if with_parents:
            rec["parents"] = t.parents
        records.append({**rec,
                        "lang": guess_lang(t.title + " " + t.body), "meta": {"tags": t.tags}})
    return records
