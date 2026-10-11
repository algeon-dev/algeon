#!/usr/bin/env python3
"""Summarize a native_tpch_benchmark TSV into a CPU-vs-GPU review table.

Reads the TSV emitted by `native_tpch_benchmark`, classifies every row into
exactly one outcome class (planning_rejected, cpu_failed,
runtime_failed:<code>, matched, mismatched, or not_run), and prints:
  - per-query CPU/GPU wall, speedup, parity, final disposition, dominant
    runtime class and the GPU time breakdown that explains where the wall
    went;
  - a coverage block: how many queries fall into each outcome bucket, and
    which query IDs;
  - totals computed ONLY over the "paired" set -- rows classified `matched`
    that report both a CPU and a GPU wall time. A query that failed, was
    rejected before execution, or never ran cannot contribute a CPU/GPU
    ratio, so it is excluded from cpu/gpu/speedup totals rather than
    silently coerced into a zero;
  - a flagged list of queries that failed, were rejected, mismatched, or
    ran slower than --slow-threshold.

Unknown memory is never reported as zero: allocator high-water mark,
max task observed peak, and resident dataframe peak are three distinct
fields (the older single "peak live reservation" field no longer exists in
the TSV). A row that does not report a given field shows `n/a` for it; the
summary states how many of the N rows actually reported each field before
taking its max.

Rendering is terminal-friendly by default: when the `rich` library is
importable and stdout is a terminal, tables print with borders and speedup
highlighting; otherwise the plain-text layout is used (full wide table
when it fits the terminal, compact one-line-per-query otherwise). Override
with --pretty and --width.

Usage: tpch_bench_review.py <tsv> [--slow-threshold 2.0] [--pretty auto] [--width 0]
"""
from __future__ import annotations
import sys, argparse, shutil

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False

# The TSV header is frozen for downstream compatibility. Its values are
# FinalPlanDisposition labels: datafusion, native, mixed, or rejected.
FINAL_DISPOSITION_TSV_COLUMN = "post_native_execution_path"

# Wide-table columns and widths; the table needs the sum plus two-space gaps.
WIDE_COLUMNS = ["query", "disposition", "parity", "cpu_ms", "gpu_ms", "speedup", "rt_class",
        "src_read", "operator", "fmerge", "fsort", "matzn", "alloc_hw_GiB", "task_peak_GiB",
        "retry", "retry_rx"]
WIDE_WIDTHS = [5, 14, 9, 9, 9, 7, 16, 9, 9, 8, 8, 8, 12, 12, 5, 8]
WIDE_TABLE_WIDTH = sum(WIDE_WIDTHS) + 2 * (len(WIDE_WIDTHS) - 1)


def load(tsv):
    meta = {}
    hdr = None
    rows = []
    for line in open(tsv):
        line = line.rstrip("\n")
        if line.startswith("#"):
            body = line[1:].strip()
            if "=" in body:
                k, _, v = body.partition("=")
                meta[k.strip()] = v.strip()
            continue
        cells = line.split("\t")
        if hdr is None:
            hdr = cells
            continue
        rows.append(dict(zip(hdr, cells)))
    return meta, hdr, rows


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt_ms(x):
    v = f(x)
    return f"{v:8.1f}" if v is not None else "     n/a"


def fmt_gib(x):
    return f"{x / (1024 ** 3):8.2f}" if x is not None else "     n/a"


def c_ms(x):
    return f"{x:.1f}ms" if x is not None else "n/a"


def c_sp(x):
    return f"x{x:.2f}" if x is not None else "n/a"


def c_gib(x):
    return f"{x / (1024 ** 3):.2f}GiB" if x is not None else "n/a"


def term_width(override):
    if override and override > 0:
        return override
    return shutil.get_terminal_size().columns


def resolve_pretty(mode):
    if mode == "never":
        return False
    if mode == "always":
        if not HAVE_RICH:
            print("warning: --pretty always but 'rich' is not importable; using plain text",
                  file=sys.stderr)
            return False
        return True
    return HAVE_RICH and sys.stdout.isatty()


def speedup_cell(sp, slow_threshold):
    txt = c_sp(sp)
    if sp is None or sp < slow_threshold:
        return Text(txt, style="red")
    if sp > 1.0:
        return Text(txt, style="green")
    return txt


def flags_cell(flags):
    return Text(flags, style="yellow") if flags else ""


def _facts_dict(facts):
    d = {}
    if not facts or facts == "none":
        return d
    for part in facts.split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            d[k] = v
    return d


def rt_label(row):
    """The dominant runtime operator class, or whether native execution ever started.

    `runtime_operator_class` is `not_run` for any row that never reached
    native execution; in that case the operator class is meaningless and
    `native_runtime_started` is the only informative signal.
    """
    op_class = row.get("runtime_operator_class", "")
    if op_class and op_class not in ("not_run", "none"):
        return op_class
    return "started" if row.get("native_runtime_started", "") == "true" else "not_started"


def classify(row):
    """Classify one TSV row into exactly one outcome bucket.

    Order matters: a row rejected before execution is never also counted as
    a runtime failure, and a row that failed on both CPU and native is
    counted once, as cpu_failed.
    """
    if row.get("rejected_before_execution", "") == "true" or row.get("planning_outcome", "") != "selected":
        return "planning_rejected"
    parity = row.get("parity_status", "")
    if parity == "cpu_failed":
        return "cpu_failed"
    cpu_failure_code = row.get("cpu_failure_code", "")
    runtime_outcome = row.get("runtime_outcome", "")
    if cpu_failure_code not in ("none", "") and runtime_outcome == "failed":
        return "cpu_failed"
    if parity == "native_failed" or runtime_outcome == "failed":
        code = row.get("runtime_failure_code", "") or "none"
        return f"runtime_failed:{code}"
    if parity == "matched":
        return "matched"
    if parity == "mismatched":
        return "mismatched"
    return "not_run"


def failure_detail(row):
    """A short human-readable explanation for a failed or rejected row; "" otherwise."""
    cls = classify(row)
    facts = _facts_dict(row.get("runtime_failure_facts", ""))
    if cls.startswith("runtime_failed:"):
        parts = []
        for key in ("cudf_code", "budget_boundary", "failure_source"):
            if facts.get(key):
                parts.append(f"{key}={facts[key]}")
        msg = facts.get("cudf_message")
        if msg:
            snippet = msg if len(msg) <= 60 else msg[-60:]
            parts.append(f"cudf_message={snippet}")
        return " ".join(parts)
    if cls == "planning_rejected":
        parts = []
        reason_code = row.get("reason_code", "")
        if reason_code and reason_code != "none":
            parts.append(f"reason_code={reason_code}")
        unsupported_category = row.get("unsupported_category", "")
        if unsupported_category and unsupported_category != "none":
            parts.append(f"unsupported_category={unsupported_category}")
        runtime_failure_code = row.get("runtime_failure_code", "")
        if runtime_failure_code and runtime_failure_code != "none":
            parts.append(f"runtime_failure_code={runtime_failure_code}")
        if facts.get("capability_reason"):
            parts.append(f"capability_reason={facts['capability_reason']}")
        return " ".join(parts)
    return ""


def flags_for(disposition, cls, sp, slow_threshold):
    if cls == "matched":
        flags = []
        if sp is None or sp < slow_threshold:
            flags.append("SLOW")
        if disposition != "native":
            flags.append("DISP")
        return ",".join(flags)
    if cls == "mismatched":
        flags = ["PAR"]
        if disposition != "native":
            flags.append("DISP")
        return ",".join(flags)
    return f"FAIL:{cls}"


def header_lines(meta):
    scale = meta.get("tpch_dataset_scale_factor", meta.get("tpch_benchmark_dataset_mode", "?"))
    line1 = (f"dataset: scale={scale} "
             f"fingerprint={meta.get('tpch_dataset_metadata_fingerprint', '?')} "
             f"warmups={meta.get('tpch_benchmark_warmups', '?')} "
             f"iterations={meta.get('tpch_benchmark_iterations', '?')}")

    cap_bytes = f(meta.get("tpch_benchmark_native_device_budget_bytes"))
    cap = f"{cap_bytes / (1024 ** 3):.2f}GiB" if cap_bytes is not None else "?"
    chunk_bytes = f(meta.get("tpch_benchmark_native_max_source_chunk_bytes"))
    if chunk_bytes is None:
        chunk = "?"
    else:
        gib = chunk_bytes / (1024 ** 3)
        chunk = f"{gib:.2f}GiB" if gib >= 1 else f"{chunk_bytes / (1024 ** 2):.0f}MiB"
    line2 = (f"native: profile={meta.get('tpch_benchmark_native_profile_name', '?')}"
             f"/{meta.get('tpch_benchmark_native_profile_source', '?')} "
             f"cap={cap} chunk={chunk} "
             f"row_groups_per_chunk={meta.get('tpch_benchmark_native_max_row_groups_per_chunk', '?')} "
             f"chunk_planning={meta.get('tpch_benchmark_native_source_chunk_planning_mode', '?')} "
             f"reads_in_flight={meta.get('tpch_benchmark_native_max_source_reads_in_flight', '?')} "
             f"relational={meta.get('tpch_benchmark_relational_execution', '?')} "
             f"memory_policy={meta.get('tpch_benchmark_native_memory_policy', '?')}")
    return [line1, line2]


def _memory_field_summary(records, key, label):
    reporting = [rec[key] for rec in records if rec[key] is not None]
    val = f"{max(reporting) / (1024 ** 3):.2f}GiB" if reporting else "n/a"
    return f"{label}={val} ({len(reporting)}/{len(records)} rows report it)"


_COVERAGE_ORDER = ["native_selected", "native_runtime_started", "parity_matched", "parity_mismatched",
                    "cpu_failed"]


def coverage_lines(records):
    lines = [f"total: {len(records)}  " + ",".join(rec["q"] for rec in records)]

    def bucket(name, qs):
        if qs:
            lines.append(f"{name}: {len(qs)}  " + ",".join(qs))

    bucket("native_selected",
           [rec["q"] for rec in records if rec["planning_outcome"] == "selected" and rec["rejected"] != "true"])
    bucket("native_runtime_started", [rec["q"] for rec in records if rec["native_runtime_started"] == "true"])
    bucket("parity_matched", [rec["q"] for rec in records if rec["cls"] == "matched"])
    bucket("parity_mismatched", [rec["q"] for rec in records if rec["cls"] == "mismatched"])
    bucket("cpu_failed", [rec["q"] for rec in records if rec["cls"] == "cpu_failed"])
    for code_cls in sorted({rec["cls"] for rec in records if rec["cls"].startswith("runtime_failed:")}):
        bucket(code_cls, [rec["q"] for rec in records if rec["cls"] == code_cls])
    bucket("planning_rejected", [rec["q"] for rec in records if rec["cls"] == "planning_rejected"])
    bucket("not_run", [rec["q"] for rec in records if rec["cls"] == "not_run"])

    path_counts = {}
    for rec in records:
        path_counts.setdefault(rec["disposition"], []).append(rec["q"])
    lines.append("runtime_path: " + " ".join(f"{path}={len(qs)}" for path, qs in sorted(path_counts.items())))
    return lines


def flagged_list(records):
    order = ["cpu_failed"]
    order += sorted({rec["cls"] for rec in records if rec["cls"].startswith("runtime_failed:")})
    order += ["planning_rejected", "not_run"]
    flagged = []
    for cls in order:
        flagged.extend(rec for rec in records if rec["cls"] == cls)
    flagged.extend(rec for rec in records if rec["cls"] == "mismatched")
    slow_matched = [rec for rec in records if rec["cls"] == "matched" and rec["flags"]]
    slow_matched.sort(key=lambda rec: (rec["sp"] if rec["sp"] is not None else -1))
    flagged.extend(slow_matched)
    return flagged


def summarize(meta, rows, slow_threshold):
    records = []
    for r in rows:
        cpu = f(r.get("cpu_wall_ms"))
        gpu = f(r.get("native_wall_ms"))
        sp = f(r.get("speedup_vs_cpu"))
        disposition = r.get(FINAL_DISPOSITION_TSV_COLUMN, "?")
        cls = classify(r)
        rec = {
            "q": r.get("query_id", "?"),
            "cpu": cpu, "gpu": gpu, "sp": sp,
            "disposition": disposition,
            "parity": r.get("parity_status", "?"),
            "rt": rt_label(r),
            "src": f(r.get("source_read_time_ms")),
            "op": f(r.get("operator_time_ms")),
            "fmerge": f(r.get("final_merge_time_ms")),
            "fsort": f(r.get("final_sort_time_ms")),
            "matzn": f(r.get("materialization_time_ms")),
            "alloc_hw": f(r.get("allocator_high_water_bytes")),
            "task_peak": f(r.get("max_task_observed_peak_bytes")),
            "resident_peak": f(r.get("resident_dataframe_peak_bytes")),
            "retry": r.get("retry_count", "0"),
            "retry_rx": r.get("retry_resource_exhausted_count", "0"),
            "planning_outcome": r.get("planning_outcome", ""),
            "rejected": r.get("rejected_before_execution", ""),
            "native_runtime_started": r.get("native_runtime_started", ""),
            "cls": cls,
            "detail": failure_detail(r),
        }
        rec["paired"] = cls == "matched" and cpu is not None and gpu is not None
        rec["flags"] = flags_for(disposition, cls, sp, slow_threshold)
        records.append(rec)

    paired = [rec for rec in records if rec["paired"]]
    cpu_total = sum(rec["cpu"] for rec in paired)
    gpu_total = sum(rec["gpu"] for rec in paired)
    overall = (cpu_total / gpu_total) if gpu_total else None
    faster = sum(1 for rec in paired if rec["sp"] is not None and rec["sp"] > 1.0)

    k, n = len(paired), len(records)
    totals = [
        f"TOTAL (paired matched {k}/{n})  cpu={cpu_total / 1000:.2f}s gpu={gpu_total / 1000:.2f}s  "
        f"speedup={c_sp(overall)}",
        f"       gpu faster: {faster}/{k}",
    ]
    if k < n:
        not_paired = [rec["q"] for rec in records if not rec["paired"]]
        totals.append(f"full-suite speedup: n/a ({n - k} queries not in paired set: " + ",".join(not_paired) + ")")

    summary = {
        "cpu": cpu_total, "gpu": gpu_total, "overall": overall, "faster": faster,
        "paired": k, "total": n,
        "header": header_lines(meta),
        "coverage": coverage_lines(records),
        "memory": [
            _memory_field_summary(records, "alloc_hw", "alloc_hw"),
            _memory_field_summary(records, "task_peak", "task_peak"),
            _memory_field_summary(records, "resident_peak", "resident_peak"),
        ],
        "totals_lines": totals,
        "flagged": flagged_list(records),
    }
    return records, summary


def timing_breakdown_text(rec):
    return (f"src={c_ms(rec['src'])} op={c_ms(rec['op'])} fmerge={c_ms(rec['fmerge'])} "
            f"fsort={c_ms(rec['fsort'])} mat={c_ms(rec['matzn'])}")


def flagged_detail_text(rec):
    text = (f"class={rec['cls']} speedup={c_sp(rec['sp'])} disposition={rec['disposition']} "
            f"parity={rec['parity']} rt={rec['rt']} cpu={c_ms(rec['cpu'])} gpu={c_ms(rec['gpu'])} "
            f"{timing_breakdown_text(rec)} retry={rec['retry']} retry_rx={rec['retry_rx']}")
    if rec["detail"]:
        text += f" detail={rec['detail']}"
    return text


def print_common_blocks(summary):
    for line in summary["coverage"]:
        print(line)
    print()
    for line in summary["memory"]:
        print(line)
    for line in summary["totals_lines"]:
        print(line)


def render_rich(records, summary, slow_threshold, full, width):
    console = Console(width=width if width > 0 else None)
    for line in summary["header"]:
        console.print(line)
    console.print()
    if full:
        table = Table()
        for col in ["q", "disposition", "parity", "cpu", "gpu", "speedup", "rt",
                    "src", "op", "fmerge", "fsort", "mat", "alloc_hw GiB", "task_peak GiB", "retry", "retry_rx"]:
            table.add_column(col)
        for rec in records:
            table.add_row(
                rec["q"], rec["disposition"], rec["parity"],
                c_ms(rec["cpu"]), c_ms(rec["gpu"]),
                speedup_cell(rec["sp"], slow_threshold), rec["rt"],
                c_ms(rec["src"]), c_ms(rec["op"]),
                c_ms(rec["fmerge"]), c_ms(rec["fsort"]), c_ms(rec["matzn"]),
                c_gib(rec["alloc_hw"]), c_gib(rec["task_peak"]),
                str(rec["retry"]), str(rec["retry_rx"]),
            )
    else:
        table = Table()
        for col in ["q", "cpu", "gpu", "spd", "flag", "parity", "rt"]:
            table.add_column(col)
        for rec in records:
            table.add_row(
                rec["q"], c_ms(rec["cpu"]), c_ms(rec["gpu"]),
                speedup_cell(rec["sp"], slow_threshold),
                flags_cell(rec["flags"]), rec["parity"], rec["rt"],
            )
    console.print(table)
    print_common_blocks(summary)
    if summary["flagged"]:
        console.print("FLAGGED:")
        detail = Table(show_header=False)
        detail.add_column("query")
        detail.add_column("detail")
        for rec in summary["flagged"]:
            detail.add_row(rec["q"], flagged_detail_text(rec))
        console.print(detail)
    else:
        console.print("FLAGGED: none; all queries matched, native, and fast.")


def render_wide(records, summary, slow_threshold):
    for line in summary["header"]:
        print(line)
    print()
    print("  ".join(c.ljust(w) for c, w in zip(WIDE_COLUMNS, WIDE_WIDTHS)))
    for rec in records:
        print("  ".join([
            rec["q"].ljust(5), rec["disposition"].ljust(14), rec["parity"].ljust(9),
            fmt_ms(rec["cpu"]), fmt_ms(rec["gpu"]),
            (f"{rec['sp']:6.2f}" if rec["sp"] is not None else "   n/a").ljust(7),
            rec["rt"][:16].ljust(16),
            fmt_ms(rec["src"]), fmt_ms(rec["op"]),
            fmt_ms(rec["fmerge"]), fmt_ms(rec["fsort"]),
            fmt_ms(rec["matzn"]),
            fmt_gib(rec["alloc_hw"]), fmt_gib(rec["task_peak"]),
            str(rec["retry"]).rjust(5), str(rec["retry_rx"]).rjust(8),
        ]))
    print()
    print_common_blocks(summary)
    print()
    if summary["flagged"]:
        print("FLAGGED:")
        for rec in summary["flagged"]:
            print(f"  {rec['q']}: {flagged_detail_text(rec)}")
    else:
        print("FLAGGED: none; all queries matched, native, and fast.")


def render_compact(records, summary, slow_threshold):
    for line in summary["header"]:
        print(line)
    print()
    print(f"{'q':<4} {'cpu':>9} {'gpu':>9} {'spd':>7} {'flag':<24} {'parity':<7} rt")
    for rec in records:
        print(f"{rec['q']:<4} {c_ms(rec['cpu']):>9} {c_ms(rec['gpu']):>9} "
              f"{c_sp(rec['sp']):>7} {rec['flags']:<24} {rec['parity']:<7} {rec['rt']}")
    print()
    print_common_blocks(summary)
    print()
    if summary["flagged"]:
        print("FLAGGED:")
        for rec in summary["flagged"]:
            print(f"  {rec['q']} {flagged_detail_text(rec)}")
    else:
        print("FLAGGED: none; all queries matched, native, and fast.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--slow-threshold", type=float, default=2.0,
                    help="flag queries with speedup below this")
    ap.add_argument("--width", type=int, default=0,
                    help="assume this terminal width; 0 auto-detects "
                         "(wide table prints only when it fits)")
    ap.add_argument("--pretty", choices=["auto", "always", "never"], default="auto",
                    help="rich tables when available (auto: only on a terminal; "
                         "always: even when piped; never: plain text)")
    args = ap.parse_args()
    meta, hdr, rows = load(args.tsv)
    records, summary = summarize(meta, rows, args.slow_threshold)

    wide = term_width(args.width) >= WIDE_TABLE_WIDTH
    if resolve_pretty(args.pretty):
        render_rich(records, summary, args.slow_threshold, full=wide, width=args.width)
    elif wide:
        render_wide(records, summary, args.slow_threshold)
    else:
        render_compact(records, summary, args.slow_threshold)


if __name__ == "__main__":
    main()
