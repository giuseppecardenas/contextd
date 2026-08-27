"""Score a ``contextd bench`` spec against contextd *and* against grep.

Both systems answer the same labelled queries and are scored with the same
``contextd.bench.metrics`` functions, so the numbers are directly comparable.

grep baselines (``--grep-mode``):

* ``term``    — split the query into content words (stopwords dropped,
  identifiers such as ``FR-ACT-002`` / ``register_material`` kept whole), run
  one ``grep -rniwF`` over the corpus with every term, rank files by the
  number of distinct terms they match, then by match count. This is what a
  person does with grep when they do not know the exact wording.
* ``literal`` — ``grep -rniF`` for the whole query string. Only meaningful
  for exact-identifier queries; included so the spread is visible.

Beyond recall/precision/MRR/IoU the script reports two things ``contextd
bench`` does not:

* **reading budget** — characters a reader consumes, walking the results in
  rank order, before reaching the first result that satisfies an expectation
  (grep: matched lines ± ``--context`` lines; contextd: the ``evidence`` text
  of each hit). Reported as the median over queries that were answered.
* **found rate** — fraction of queries with at least one satisfying hit in
  the top-k.

Optional ``--graph-expand N`` adds, after contextd's direct hits, the File
nodes reachable within two hops of the top-N section hits (via entities or
structural edges), for relational queries whose answer is a set of files.

Usage::

    python scripts/bench_vs_grep.py --spec .contextd/bench/g2-semantic.toml \
        --root C:/Users/giuse/src/runeledger --corpus runeledger --k 5 \
        --return-unit auto --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from contextd.bench.metrics import (
    QueryScore,
    Target,
    line_iou,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    satisfies,
    summarise,
)
from contextd.bench.run import row_to_target
from contextd.bench.spec import BenchQuery, load_spec
from contextd.config import Config
from contextd.mcp import tools
from contextd.providers.factory import build_embedding_provider
from contextd.search.collapse import ReturnUnit
from contextd.storage.base import GraphStore
from contextd.storage.factory import build_graph_store

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "without",
        "into",
        "onto",
        "over",
        "under",
        "between",
        "among",
        "about",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "it",
        "its",
        "they",
        "them",
        "their",
        "his",
        "her",
        "he",
        "she",
        "we",
        "you",
        "your",
        "our",
        "i",
        "me",
        "my",
        "mine",
        "one",
        "ones",
        "any",
        "some",
        "all",
        "each",
        "every",
        "both",
        "either",
        "neither",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "own",
        "same",
        "such",
        "more",
        "most",
        "less",
        "least",
        "much",
        "many",
        "few",
        "other",
        "another",
        "again",
        "further",
        "once",
        "out",
        "up",
        "down",
        "off",
        "before",
        "after",
        "above",
        "below",
        "through",
        "during",
        "while",
        "until",
        "because",
        "get",
        "gets",
        "got",
        "make",
        "makes",
        "made",
        "take",
        "takes",
        "took",
        "use",
        "uses",
        "used",
        "using",
        "does",
        "something",
        "anything",
        "everything",
        "nothing",
        "way",
        "ways",
        "thing",
        "things",
        "kind",
        "kinds",
        "part",
        "parts",
        "across",
        "per",
        "still",
        "yet",
        "even",
        "ever",
        "never",
        "always",
        "often",
    ]
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


def tokenize(query: str) -> list[str]:
    """Content terms for term-grep: identifiers kept whole, stopwords dropped."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(query):
        low = tok.lower()
        is_identifier = any(ch.isdigit() or ch in "_-" for ch in tok)
        if not is_identifier and (low in _STOPWORDS or len(low) < 3):
            continue
        if low not in out:
            out.append(low)
    return out


@dataclass
class Hit:
    target: Target
    text: str
    """What a reader would consume for this hit (evidence or matched lines)."""


@dataclass
class QueryResult:
    query: str
    k: int
    terms: list[str]
    hits: list[dict[str, Any]]
    recall: float
    precision: float
    mrr: float
    iou: float | None
    found: bool
    chars_to_answer: int | None
    latency_ms: float


@dataclass
class SystemReport:
    system: str
    config: dict[str, Any]
    results: list[QueryResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        scores = [
            QueryScore(query=r.query, recall=r.recall, precision=r.precision, mrr=r.mrr, iou=r.iou)
            for r in self.results
        ]
        out: dict[str, Any] = dict(summarise(scores))
        answered = [r.chars_to_answer for r in self.results if r.chars_to_answer is not None]
        lat = [r.latency_ms for r in self.results]
        out["found_rate"] = (
            sum(1 for r in self.results if r.found) / len(self.results) if self.results else 0.0
        )
        out["median_chars_to_answer"] = statistics.median(answered) if answered else None
        out["latency_p50_ms"] = statistics.median(lat) if lat else 0.0
        out["latency_p95_ms"] = _percentile(lat, 0.95) if lat else 0.0
        out["queries"] = len(self.results)
        return out


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def _score(
    query: BenchQuery, k: int, hits: list[Hit], latency_ms: float, terms: list[str]
) -> QueryResult:
    targets = [h.target for h in hits]
    chars: int | None = None
    consumed = 0
    for h in hits[:k]:
        consumed += len(h.text)
        if any(satisfies(h.target, e) for e in query.expect):
            chars = consumed
            break
    return QueryResult(
        query=query.q,
        k=k,
        terms=terms,
        hits=[
            {
                "path": h.target.path,
                "anchor": h.target.anchor,
                "lines": list(h.target.lines) if h.target.lines else None,
                "chars": len(h.text),
            }
            for h in hits[:k]
        ],
        recall=recall_at_k(targets, query.expect, k),
        precision=precision_at_k(targets, query.expect, k),
        mrr=reciprocal_rank(targets, query.expect),
        iou=line_iou(targets, query.expect, k),
        found=chars is not None,
        chars_to_answer=chars,
        latency_ms=latency_ms,
    )


# --------------------------------------------------------------------------- grep


class GrepCorpus:
    def __init__(self, root: Path, includes: list[str], excludes: list[str], context: int):
        self.root = root
        self.includes = includes
        self.excludes = excludes
        self.context = context
        self._lines: dict[Path, list[str]] = {}

    def _file_lines(self, path: Path) -> list[str]:
        if path not in self._lines:
            try:
                self._lines[path] = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                self._lines[path] = []
        return self._lines[path]

    def run(self, terms: list[str], *, literal: bool) -> tuple[list[tuple[Path, int, str]], float]:
        """Run one grep; return (path, 0-based line, text) rows and wall ms."""
        # Case-insensitivity is folded into the pattern by hand: the grep 3.0
        # shipped with Git for Windows aborts (exit 1536, no output) on
        # ``-i`` combined with ``-F`` or with several ``-e`` terms when it is
        # launched from a non-MSYS parent process.
        cmd = ["grep", "-rnI", "--color=never", "-E"]
        if not literal:
            cmd.append("-w")
        for inc in self.includes:
            cmd.append(f"--include={inc}")
        for exc in self.excludes:
            cmd.append(f"--exclude={exc}")
        cmd += ["-e", "(" + "|".join(_fold_case(t) for t in terms) + ")", "."]
        started = time.perf_counter()
        proc = subprocess.run(cmd, cwd=self.root, capture_output=True, check=False)
        elapsed = (time.perf_counter() - started) * 1000.0
        rows: list[tuple[Path, int, str]] = []
        for raw in proc.stdout.decode("utf-8", errors="replace").splitlines():
            parts = raw.split(":", 2)
            if len(parts) < 3:
                continue
            rel, lineno, text = parts
            rel = rel.lstrip("./").replace("\\", "/")
            try:
                rows.append((self.root / rel, int(lineno) - 1, text))
            except ValueError:
                continue
        return rows, elapsed

    def hits(self, query: str, *, mode: str, k: int) -> tuple[list[Hit], float, list[str]]:
        terms = [query] if mode == "literal" else tokenize(query)
        if not terms:
            return [], 0.0, terms
        rows, elapsed = self.run(terms, literal=(mode == "literal"))
        per_file: dict[Path, list[tuple[int, set[str]]]] = defaultdict(list)
        for path, line, text in rows:
            low = text.lower()
            matched = {t for t in terms if _term_in(t, low, word=(mode != "literal"))}
            per_file[path].append((line, matched))

        ranked: list[tuple[int, int, str, Path]] = []
        for path, entries in per_file.items():
            distinct = set().union(*(m for _, m in entries)) if entries else set()
            ranked.append((-len(distinct), -len(entries), str(path), path))
        ranked.sort()

        hits: list[Hit] = []
        for _, _, _, path in ranked[:k]:
            entries = sorted(per_file[path], key=lambda e: e[0])
            start, end = self._best_cluster(entries)
            lines = self._file_lines(path)
            lo, hi = max(0, start - self.context), min(len(lines), end + self.context)
            text = "\n".join(lines[lo:hi])
            rel = path.relative_to(self.root).as_posix()
            hits.append(Hit(target=Target(path=rel, lines=(lo, max(hi, lo + 1))), text=text))
        return hits, elapsed, terms

    @staticmethod
    def _best_cluster(entries: list[tuple[int, set[str]]], gap: int = 30) -> tuple[int, int]:
        """Group matched lines closer than ``gap``; return the span of the
        cluster with the most distinct terms (ties: most lines)."""
        clusters: list[list[tuple[int, set[str]]]] = []
        for entry in entries:
            if clusters and entry[0] - clusters[-1][-1][0] <= gap:
                clusters[-1].append(entry)
            else:
                clusters.append([entry])

        def key(c: list[tuple[int, set[str]]]) -> tuple[int, int]:
            return (len(set().union(*(m for _, m in c))), len(c))

        best = max(clusters, key=key)
        return best[0][0], best[-1][0] + 1


def _fold_case(term: str) -> str:
    """ERE for ``term`` matching either case, all other characters escaped."""
    out: list[str] = []
    for ch in term:
        if ch.isalpha() and ch.lower() != ch.upper():
            out.append(f"[{ch.lower()}{ch.upper()}]")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _term_in(term: str, text: str, *, word: bool) -> bool:
    if not word:
        return term.lower() in text
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", text) is not None


# ----------------------------------------------------------------------- contextd


class ContextdRunner:
    def __init__(
        self,
        cfg: Config,
        *,
        corpus: str,
        profiles: list[str] | None,
        return_unit: ReturnUnit,
        graph_expand: int,
        mode: str | None = None,
    ):
        self.cfg = cfg
        self.corpus = corpus
        self.profiles = profiles
        self.return_unit = return_unit
        self.graph_expand = graph_expand
        self.mode: Any = mode or cfg.search.mode
        self.store: GraphStore = build_graph_store(cfg)
        self.store.connect()
        self.embedder = build_embedding_provider(cfg)

    def close(self) -> None:
        self.store.close()

    def hits(self, query: str, *, k: int) -> tuple[list[Hit], float]:
        started = time.perf_counter()
        rows = self._search(query, k=k, return_unit=self.return_unit)
        hits: list[Hit] = []
        for row in rows:
            ev = row.get("evidence") or {}
            text = str(ev.get("text") or row.get("summary") or "")
            hits.append(Hit(target=row_to_target(row), text=text))
        if self.graph_expand:
            hits = self._expand(query, hits, k)
        elapsed = (time.perf_counter() - started) * 1000.0
        return hits, elapsed

    def _search(self, query: str, *, k: int, return_unit: ReturnUnit) -> list[dict[str, Any]]:
        sc = self.cfg.search
        return tools.search(
            self.store,
            query,
            limit=k,
            embedder=self.embedder,
            mode=self.mode,
            rrf_k=sc.rrf_k,
            fetch_k=sc.fetch_k,
            vector_weight=sc.vector_weight,
            fulltext_weight=sc.fulltext_weight,
            corpus=self.corpus,
            profiles=self.profiles,
            return_unit=return_unit,
            auto_merge_threshold=sc.auto_merge_threshold,
            window=0,
            max_evidence_chars=sc.max_evidence_chars,
        )

    def _expand(self, query: str, direct: list[Hit], k: int) -> list[Hit]:
        """Graph-first ranking for relational queries.

        Seeds = the top-N *section* hits for the query (a PRD passage that
        describes the requirement). Files within two hops of a seed — via
        the entities the section and the file both reference, or structural
        edges — are ranked by how many such paths connect them, and placed
        ahead of the direct file hits, which fill the remainder of ``k``.
        """
        seed_rows = self._search(query, k=self.graph_expand, return_unit="section")
        seeds: list[str] = []
        for row in seed_rows:
            node_id = row.get("parent_id") if row.get("unit") == "chunk" else row.get("id")
            if isinstance(node_id, str) and node_id not in seeds:
                seeds.append(node_id)
        weight: dict[str, float] = defaultdict(float)
        summary: dict[str, str] = {}
        for seed in seeds:
            for r in self.store.exec_read(
                "MATCH (s {id: $id})-[*1..2]-(f:File {corpus: $corpus}) "
                "WHERE f.path <> s.path "
                "RETURN f.path AS path, f.summary AS summary, count(*) AS w",
                {"id": seed, "corpus": self.corpus},
            ):
                path = str(r["path"]).replace("\\", "/")
                weight[path] += float(r["w"])
                summary[path] = str(r.get("summary") or "")
        ranked = sorted(weight, key=lambda p: (-weight[p], p))
        out: list[Hit] = [Hit(target=Target(path=p), text=summary[p]) for p in ranked]
        seen = set(ranked)
        for h in direct:
            p = h.target.path.replace("\\", "/")
            if p not in seen:
                seen.add(p)
                out.append(h)
        return out[:k]


# ---------------------------------------------------------------------------- main


def _render(reports: list[SystemReport]) -> str:
    cols = [
        ("recall", "recall@k"),
        ("precision", "prec@k"),
        ("mrr", "MRR"),
        ("iou", "line IoU"),
        ("found_rate", "found"),
        ("median_chars_to_answer", "chars→ans"),
        ("latency_p50_ms", "p50 ms"),
        ("latency_p95_ms", "p95 ms"),
    ]
    head = "| system | " + " | ".join(label for _, label in cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|"
    lines = [head, sep]
    for rep in reports:
        s = rep.summary()
        cells = []
        for key, _ in cols:
            v = s.get(key)
            if v is None:
                cells.append("-")
            elif key in ("median_chars_to_answer",):
                cells.append(f"{int(v):,}")
            elif key.startswith("latency"):
                cells.append(f"{v:,.0f}")
            else:
                cells.append(f"{v:.3f}")
        lines.append(f"| {rep.system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="corpus root for grep")
    ap.add_argument("--corpus", required=True, help="contextd corpus name")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--return-unit", default="auto", choices=["chunk", "section", "file", "auto"])
    ap.add_argument("--profiles", default=None, help="comma list of chunk profiles")
    ap.add_argument("--graph-expand", type=int, default=0, help="expand top-N hits via graph")
    ap.add_argument(
        "--mode",
        default=None,
        choices=["hybrid", "fulltext", "vector"],
        help="ranker mode; default: [search] mode from config.toml",
    )
    ap.add_argument("--grep-mode", action="append", choices=["term", "literal"], default=None)
    ap.add_argument("--include", action="append", default=None, help="grep --include glob")
    ap.add_argument("--exclude", action="append", default=None, help="grep --exclude glob")
    ap.add_argument("--context", type=int, default=3, help="grep context lines for reading budget")
    ap.add_argument("--skip-contextd", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--label", default=None, help="label for the contextd row")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    includes = args.include or ["*.md", "*.lua"]
    excludes = args.exclude or ["_audit-methodology.md"]
    grep_modes = args.grep_mode or ["term"]
    profiles = [p.strip() for p in args.profiles.split(",")] if args.profiles else None
    reports: list[SystemReport] = []

    if not args.skip_contextd:
        runner = ContextdRunner(
            Config.load_default(),
            corpus=args.corpus,
            profiles=profiles,
            return_unit=cast(ReturnUnit, args.return_unit),
            graph_expand=args.graph_expand,
            mode=args.mode,
        )
        label = args.label or (
            f"contextd[{args.return_unit}"
            + (f",{args.mode}" if args.mode else "")
            + (f",{','.join(profiles)}" if profiles else "")
            + (f",+graph{args.graph_expand}" if args.graph_expand else "")
            + "]"
        )
        rep = SystemReport(
            system=label,
            config={
                "return_unit": args.return_unit,
                "profiles": profiles,
                "graph_expand": args.graph_expand,
                "k": args.k,
            },
        )
        try:
            for q in spec.queries:
                k = q.k or args.k
                hits, ms = runner.hits(q.q, k=k)
                rep.results.append(_score(q, k, hits, ms, []))
        finally:
            runner.close()
        reports.append(rep)

    corpus = GrepCorpus(args.root, includes, excludes, args.context)
    for mode in grep_modes:
        rep = SystemReport(system=f"grep[{mode}]", config={"mode": mode, "k": args.k})
        for q in spec.queries:
            k = q.k or args.k
            hits, ms, terms = corpus.hits(q.q, mode=mode, k=k)
            rep.results.append(_score(q, k, hits, ms, terms))
        reports.append(rep)

    sys.stdout.write(_render(reports) + "\n")
    if args.json:
        payload = {
            "spec": str(args.spec),
            "reports": [
                {
                    "system": r.system,
                    "config": r.config,
                    "summary": r.summary(),
                    "results": [asdict(x) for x in r.results],
                }
                for r in reports
            ],
        }
        args.json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
