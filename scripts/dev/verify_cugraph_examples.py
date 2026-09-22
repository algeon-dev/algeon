#!/usr/bin/env python3
"""Run every SQL statement of the cuGraph citation examples against a live server.

The helper reads the hand-owned pages under website/docs/cugraph/_examples that
link the citation demo dataset, splits each ```sql block into statements, and
executes them in page order through ``algeon_tools flight-sql-query``. With
``--data``, the direct-Parquet registration block of demo-dataset.mdx runs
first, with ``<data>`` replaced by that directory.

A statement fails when the client exits non-zero or times out, or when a query
(anything other than CREATE/DROP) returns no rows. Result values are not
compared, because the pages quote single-run captures. The Flight SQL server
must already run with cuGraph enabled and hold the GPU lock itself; this helper
only acts as a client.

    python3 scripts/dev/verify_cugraph_examples.py \\
      --data "$PWD/fixture/graph/citation_network/v1" --report /tmp/report.md
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "website" / "docs" / "cugraph"
SQL_FENCE = re.compile(r"^```sql\n(.*?)^```", re.S | re.M)
DDL = re.compile(r"^(CREATE|DROP)\b", re.I)
PREVIEW_LINES = 12


@dataclass
class Outcome:
    page: str
    index: int
    sql: str
    status: str
    rows: int
    seconds: float
    preview: list[str]
    stderr: str


def split_statements(sql: str) -> list[str]:
    """Split on `;` outside single-quoted strings and `--` line comments."""
    statements: list[str] = []
    current: list[str] = []
    in_string = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if in_string:
            # A doubled quote closes and immediately reopens the literal.
            in_string = ch != "'"
        elif sql.startswith("--", i):
            end = sql.find("\n", i)
            end = len(sql) if end == -1 else end
            current.append(sql[i:end])
            i = end
            continue
        elif ch == "'":
            in_string = True
        elif ch == ";":
            statements.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    statements.append("".join(current))
    return [statement.strip() for statement in statements if without_comments(statement)]


def without_comments(statement: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in statement.splitlines()).strip()


def page_statements(path: Path) -> list[str]:
    return [statement for block in SQL_FENCE.findall(path.read_text()) for statement in split_statements(block)]


def registration_statements(data_dir: str) -> list[str]:
    blocks = [block for block in SQL_FENCE.findall((DOCS / "demo-dataset.mdx").read_text()) if "<data>" in block]
    if len(blocks) != 1:
        raise SystemExit(f"expected one <data> registration block in demo-dataset.mdx, found {len(blocks)}")
    return split_statements(blocks[0].replace("<data>", data_dir))


def run_statement(tools: str, url: str, sql: str, timeout: float) -> tuple[str, int, float, list[str], str]:
    with tempfile.TemporaryFile("w+") as stdout:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [tools, "flight-sql-query", url, sql],
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "FAIL (timeout)", 0, time.monotonic() - started, [], f"no result within {timeout:.0f} s"
        seconds = time.monotonic() - started
        stdout.seek(0)
        preview: list[str] = []
        lines = 0
        for line in stdout:
            if lines < PREVIEW_LINES:
                preview.append(line.rstrip("\n"))
            lines += 1
    if completed.returncode != 0:
        return "FAIL (error)", 0, seconds, preview, completed.stderr.strip()
    rows = 0 if preview[:1] == ["ok"] else max(lines - 1, 0)
    if rows == 0 and not DDL.match(without_comments(sql)):
        return "FAIL (0 rows)", 0, seconds, preview, completed.stderr.strip()
    return "PASS", rows, seconds, preview, ""


def gpu_name() -> str:
    if shutil.which("nvidia-smi") is None:
        return "unknown"
    query = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    return subprocess.run(query, capture_output=True, text=True, check=False).stdout.strip() or "unknown"


def render_report(outcomes: list[Outcome], url: str, skipped: list[str]) -> str:
    failures = [outcome for outcome in outcomes if outcome.status != "PASS"]
    lines = [
        "# cuGraph citation example verification",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Server: {url}",
        f"- GPU: {gpu_name()}",
        f"- Statements: {len(outcomes)} run, {len(outcomes) - len(failures)} passed, {len(failures)} failed",
        f"- Skipped pages (no demo-dataset link): {', '.join(skipped) or 'none'}",
        "",
        "| Page | # | Statement | Status | Rows | Seconds |",
        "|---|---|---|---|---|---|",
    ]
    for outcome in outcomes:
        head = without_comments(outcome.sql).splitlines()[0].replace("|", "\\|")
        lines.append(
            f"| {outcome.page} | {outcome.index} | `{head}` | {outcome.status} | {outcome.rows} | {outcome.seconds:.1f} |"
        )
    for outcome in outcomes:
        lines += ["", f"## {outcome.page} #{outcome.index}: {outcome.status}", "", "```sql", outcome.sql, "```"]
        if outcome.preview:
            lines += ["", "```text", *outcome.preview, "```"]
        if outcome.stderr:
            lines += ["", "```text", outcome.stderr, "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:50051")
    parser.add_argument("--tools", default=str(ROOT / "target" / "release" / "algeon_tools"))
    parser.add_argument("--data", help="register the demo tables from this citation_network/v1 directory first")
    parser.add_argument("--pages", nargs="*", default=[], help="run only pages whose file name contains one of these")
    parser.add_argument("--passes", type=int, default=1, help="run the selected pages this many times in a row")
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-statement timeout in seconds")
    parser.add_argument("--report", type=Path, help="write the Markdown report here instead of stdout")
    args = parser.parse_args()

    pages = sorted((DOCS / "_examples").glob("*.mdx"))
    skipped = [page.name for page in pages if "demo-dataset.mdx" not in page.read_text()]
    pages = [page for page in pages if page.name not in skipped]
    if args.pages:
        pages = [page for page in pages if any(selector in page.name for selector in args.pages)]

    work: list[tuple[str, list[str]]] = []
    if args.data:
        work.append(("demo-dataset.mdx", registration_statements(args.data)))
    for _ in range(args.passes):
        work += [(page.name, page_statements(page)) for page in pages]

    outcomes: list[Outcome] = []
    for page, statements in work:
        for index, sql in enumerate(statements, start=1):
            status, rows, seconds, preview, stderr = run_statement(args.tools, args.url, sql, args.timeout)
            outcomes.append(Outcome(page, index, sql, status, rows, seconds, preview, stderr))
            print(f"{status:14} {seconds:8.1f}s {rows:>9} rows  {page} #{index}", file=sys.stderr, flush=True)

    report = render_report(outcomes, args.url, skipped)
    if args.report:
        args.report.write_text(report)
    else:
        sys.stdout.write(report)
    return 1 if any(outcome.status != "PASS" for outcome in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
