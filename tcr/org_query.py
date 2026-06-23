"""Parse org-mode tasks and normalize a heading into a clean query.

A heading is a *task* only if it carries a TODO-type keyword. Bare headings are
projects/sections and are skipped (they never appear in the agenda).

Normalization keeps the representation system-agnostic plain text:
title + tags, markup and links cleaned up. See design/experimentation-strategy.md §2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Optional

# TODO-type keywords that mark a heading as an actionable task.
TODO_KEYWORDS = {"TODO", "NEXT", "INPR", "WAIT", "MAYB", "DONE"}

_HEADING_RE = re.compile(r"^(\*+)\s+(.*)$")
_PRIORITY_RE = re.compile(r"^\[#[A-Za-z]\]\s*")
_TAGS_RE = re.compile(r"\s+(:(?:[A-Za-z0-9_@#%]+:)+)\s*$")
_TIMESTAMP_RE = re.compile(r"[<\[]\d{4}-\d{2}-\d{2}[^>\]]*[>\]]")
_SCHEDULING_RE = re.compile(r"\b(SCHEDULED|DEADLINE|CLOSED|Started):", re.IGNORECASE)

# Links --------------------------------------------------------------------
# org link with description: [[target][desc]] -> desc
_ORG_LINK_DESC_RE = re.compile(r"\[\[[^\]]*?\]\[([^\]]+)\]\]")
# bare org link: [[target]]
_ORG_LINK_BARE_RE = re.compile(r"\[\[([^\]]+)\]\]")
# inline url
_URL_RE = re.compile(r"\bhttps?://\S+")
_MAILTO_RE = re.compile(r"\bmailto:(\S+)")

# Emphasis markers to strip (keep the inner text).
_EMPHASIS_RE = re.compile(r"(?<![\w])([*/_=~+])(\S(?:.*?\S)?)\1(?![\w])")


@dataclass
class Query:
    title: str          # cleaned heading text (no tags)
    tags: List[str]     # org tags
    text: str           # title + tags joined — what the matcher consumes

    @classmethod
    def from_text(cls, raw: str, tags: Optional[List[str]] = None) -> "Query":
        tags = tags or []
        title = normalize_text(raw)
        joined = title
        if tags:
            joined = f"{title} ({', '.join(tags)})" if title else ", ".join(tags)
        return cls(title=title, tags=tags, text=joined)


def _readable_from_target(target: str) -> str:
    """Turn a link target into readable words, or '' if it looks opaque."""
    t = target.strip()
    # strip a known protocol prefix: obsidian:, id:, file:, roam:, mailto:, http(s):
    m = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:(.*)$", t)
    if m:
        t = m.group(1)
    # drop leading // and url host noise, keep last path segment
    t = t.lstrip("/")
    if "/" in t:
        t = t.rstrip("/").split("/")[-1]
    t = re.sub(r"[#?].*$", "", t)          # drop fragments/queries
    t = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", t)  # drop a file extension
    t = t.replace("_", " ").replace("-", " ").strip()
    # opaque if it's a uuid-ish / hex / mostly non-alpha token
    if re.fullmatch(r"[0-9a-fA-F]{8,}", t) or not re.search(r"[A-Za-z]{2,}", t):
        return ""
    return t


def _clean_links(text: str) -> str:
    text = _ORG_LINK_DESC_RE.sub(lambda m: m.group(1), text)
    text = _ORG_LINK_BARE_RE.sub(lambda m: _readable_from_target(m.group(1)), text)
    text = _MAILTO_RE.sub(lambda m: _readable_from_target(m.group(1)), text)
    text = _URL_RE.sub(lambda m: _readable_from_target(m.group(0)), text)
    return text


def _strip_emphasis(text: str) -> str:
    prev = None
    while prev != text:  # handle nested/adjacent markers
        prev = text
        text = _EMPHASIS_RE.sub(lambda m: m.group(2), text)
    return text


def normalize_text(raw: str) -> str:
    """Normalize a single heading body (keyword/priority/tags already removed
    upstream by `parse_heading`, but this is also safe to call on raw text)."""
    text = raw.strip()
    # remove a leading TODO keyword if present
    parts = text.split(None, 1)
    if parts and parts[0] in TODO_KEYWORDS:
        text = parts[1] if len(parts) > 1 else ""
    text = _PRIORITY_RE.sub("", text)
    # drop tags if still attached
    mtags = _TAGS_RE.search(text)
    if mtags:
        text = text[: mtags.start()]
    text = _SCHEDULING_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)
    text = _clean_links(text)
    text = _strip_emphasis(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_heading(line: str) -> Optional[Query]:
    """If `line` is a TODO-type task heading, return its Query; else None."""
    m = _HEADING_RE.match(line.rstrip("\n"))
    if not m:
        return None
    body = m.group(2).strip()
    parts = body.split(None, 1)
    if not parts or parts[0] not in TODO_KEYWORDS:
        return None
    rest = parts[1] if len(parts) > 1 else ""
    # extract tags
    tags: List[str] = []
    mtags = _TAGS_RE.search(rest)
    if mtags:
        tags = [t for t in mtags.group(1).strip(":").split(":") if t]
        rest = rest[: mtags.start()]
    return Query.from_text(rest, tags)


def iter_tasks(org_text: str) -> Iterator[Query]:
    for line in org_text.splitlines():
        q = parse_heading(line)
        if q and q.text:
            yield q
