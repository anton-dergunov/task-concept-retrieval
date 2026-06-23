"""Command-line entry point.

  # batch match a JSON array of task strings (stdin) -> {task: icon|null}
  echo '["Walk at least 8k steps daily","Caminar 8000 pasos al día"]' | python -m tcr.cli

  # single query, ranked shortlist with scores
  python -m tcr.cli search "save money for vacation"

  # web UI with rendered icons
  python -m tcr.cli serve

Options: --method {B0,B1,M1,M2,M3} (default M3), -k N.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import List

from . import search


def _cmd_batch(texts: List[str], method: str, k: int) -> None:
    matcher = search.get_matcher(method)
    out = {}
    latencies = []
    for t in texts:
        t0 = time.perf_counter()
        decision = matcher.match_one(t, top_k=k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        out[t] = decision.icon if decision.shown else None
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if latencies:
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
        print(f"# latency p50={p50:.1f}ms p95={p95:.1f}ms (n={len(latencies)})", file=sys.stderr)


def _cmd_search(query: str, method: str, k: int) -> None:
    result = search.search(query, method=method, k=k)
    print(f"query : {result.query}")
    print(f"method: {result.method}   latency: {result.latency_ms} ms   "
          f"confidence: {result.confidence}")
    if result.shown:
        print(f"decision: SHOW '{result.decision_icon}'")
    else:
        print(f"decision: ABSTAIN ({result.reason})")
    print("rank  score   icon")
    for i, h in enumerate(result.hits, 1):
        mark = " *" if h.shown else "  "
        glyph = f" {h.char}" if h.char else ""
        print(f"{i:>3}{mark} {h.score:>6.3f}  {h.name}{glyph}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="tcr.cli", description="task -> icon matcher")
    parser.add_argument("command", nargs="?", default=None,
                        help="'search <query>' or 'serve'; omit to read JSON tasks from stdin")
    parser.add_argument("rest", nargs="*", help="query words for 'search'")
    parser.add_argument("--method", default="M3", choices=search.available_methods())
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args(argv)

    if args.command == "serve":
        from .server import serve
        serve(method=args.method)
        return

    if args.command == "search":
        query = " ".join(args.rest).strip()
        if not query:
            print("usage: python -m tcr.cli search \"<query>\"", file=sys.stderr)
            sys.exit(2)
        _cmd_search(query, args.method, args.k)
        return

    # No subcommand: read a JSON array of task strings from stdin; if a TTY, serve.
    if args.command is None and sys.stdin.isatty():
        from .server import serve
        serve(method=args.method)
        return

    payload = sys.stdin.read() if args.command is None else args.command
    texts = json.loads(payload)
    if not isinstance(texts, list):
        raise SystemExit("input must be a JSON list of task strings")
    _cmd_batch(texts, args.method, args.k)


if __name__ == "__main__":
    main()
