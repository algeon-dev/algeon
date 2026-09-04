#!/usr/bin/env python3
"""Fail closed on unclassified `#[doc(hidden)] pub` surface in datafusion-nexus.

`cargo public-api` omits `#[doc(hidden)]` items, so the snapshot gate in
``check_public_api_surface.sh`` cannot see them. This check closes that gap for
the root adapter crate: every `#[doc(hidden)]`-attributed `pub` item in
``src/`` must resolve to exactly one entry in
``.config/policies/public-api/datafusion-nexus-hidden.tsv`` naming its
feature contract, owner, consumer, and removal condition. Module entries cover
unattributed children inside that module; anything else fails.

Usage:
  python3 scripts/ci/check_public_api_hidden.py [LEDGER]

Exits nonzero on any ledger/source mismatch, with the offending symbols
printed to stderr.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
DEFAULT_LEDGER = ROOT / ".config" / "policies" / "public-api" / "datafusion-nexus-hidden.tsv"

HEADER = ["symbol", "feature", "owner", "consumer", "removal_condition"]
ALLOWED_FEATURES = {"server-internal", "bench-support", "qualification"}

ATTR_RE = re.compile(r"^#\[(?P<body>[^\]]*)\]\s*$")
CFG_FEATURE_RE = re.compile(r'feature\s*=\s*"(?P<feature>[^"]+)"')
ITEM_RE = re.compile(r"^pub\s+(?P<kind>mod|struct|enum|fn|const|static|type|trait|use)\b\s*(?P<rest>.*)$")
MOD_DECL_RE = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?mod\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<tail>.*)$")


def fail(message: str) -> None:
    print(f"ERROR: hidden public API policy: {message}", file=sys.stderr)


def module_path_for_file(path: Path) -> list[str]:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "mod":
        parts = parts[:-1]
    if parts == ["lib"]:
        return ["datafusion_nexus"]
    return ["datafusion_nexus", *parts]


def features_in_cfg(body: str) -> set[str]:
    return set(CFG_FEATURE_RE.findall(body))


def is_test_only_cfg(body: str) -> bool:
    return "test" in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body)


def parent_of(mod_stack: list[tuple[str, str | None]], module_path: list[str]) -> str:
    return "::".join([*module_path, *[entry for entry, _ in mod_stack]])


def record_use_leaves(rest: str, parent: str, path: Path, line_number: int, record) -> None:
    for alias in re.findall(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)", rest):
        record(f"{parent}::{alias}", line_number)
    if "{" not in rest:
        leaf = rest.split("::")[-1].strip().rstrip(",}; ")
        if leaf and leaf not in {"*", "self", "super", "crate"} and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", leaf):
            record(f"{parent}::{leaf}", line_number)


def strip_line_comment(line: str) -> str:
    in_string = False
    escaped = False
    for index, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "/" and index + 1 < len(line) and line[index + 1] == "/":
            return line[:index]
    return line


def scan_file(path: Path) -> tuple[dict[str, str | None], dict[tuple[str, str], str]]:
    """Return ({symbol: feature-or-None} hidden sites, {(parent, mod): feature} gates)."""
    if "test" in path.name:
        return {}, {}
    module_path = module_path_for_file(path)
    lines = path.read_text().splitlines()
    sites: dict[str, str | None] = {}
    mod_gates: dict[tuple[str, str], str] = {}
    mod_stack: list[tuple[str, str | None]] = []
    pending_cfg: set[str] = set()
    pending_cfg_test_only = False
    pending_hidden = False
    pending_use: str | None = None
    suppress_depth: int | None = None
    brace_depth = 0

    def reset_pending() -> None:
        nonlocal pending_cfg, pending_cfg_test_only, pending_hidden, pending_use
        pending_cfg, pending_cfg_test_only, pending_hidden, pending_use = set(), False, False, None

    def current_feature() -> str | None:
        if len(pending_cfg) == 1:
            return next(iter(pending_cfg))
        if len(pending_cfg) > 1:
            return "|".join(sorted(pending_cfg))
        for _, feature in reversed(mod_stack):
            if feature is not None:
                return feature
        return None

    def record(symbol: str, line_number: int) -> None:
        if symbol in sites:
            fail(f"{path}:{line_number}: duplicate hidden site for {symbol}")
            raise SystemExit(1)
        sites[symbol] = current_feature()

    for line_number, raw in enumerate(lines, 1):
        line = strip_line_comment(raw).strip()
        if suppress_depth is not None:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= suppress_depth:
                suppress_depth = None
            continue
        if not line or line.startswith("///") or line.startswith("//!"):
            continue
        attr = ATTR_RE.match(line)
        if attr:
            body = attr.group("body")
            if body.startswith("doc(hidden"):
                pending_hidden = True
            elif body.startswith("cfg("):
                inner = body[len("cfg("):]
                if is_test_only_cfg(inner):
                    pending_cfg_test_only = True
                else:
                    pending_cfg |= features_in_cfg(inner)
            continue
        mod_decl = MOD_DECL_RE.match(line)
        if mod_decl:
            name = mod_decl.group("name")
            tail = mod_decl.group("tail").strip()
            test_only = pending_cfg_test_only or "test" in name
            hidden = pending_hidden and not test_only
            feature = next(iter(pending_cfg)) if len(pending_cfg) == 1 else None
            parent = "::".join([*module_path, *[entry for entry, _ in mod_stack]])
            if hidden:
                record(f"{parent}::{name}", line_number)
                feature = current_feature()
                if tail.startswith(";"):
                    if feature is not None:
                        mod_gates[(parent, name)] = feature
                elif tail.count("{") > tail.count("}"):
                    suppress_depth = brace_depth
            elif test_only:
                pass
            elif tail.startswith(";"):
                if feature is not None:
                    mod_gates[(parent, name)] = feature
            elif "{" in tail:
                mod_stack.append((name, feature))
                brace_depth += tail.count("{") - tail.count("}")
            reset_pending()
            continue
        item = ITEM_RE.match(line)
        if item and not pending_cfg_test_only:
            kind = item.group("kind")
            rest = item.group("rest").strip()
            parent = "::".join([*module_path, *[entry for entry, _ in mod_stack]])
            if kind == "use":
                if "{" in rest and "}" not in rest:
                    pending_use = rest
                    continue
                if pending_hidden:
                    record_use_leaves(rest, parent, path, line_number, record)
                reset_pending()
                continue
            token = re.split(r"[\s<(;:,{]", rest, 1)[0]
            named = bool(token) and (token[0].isupper() or kind in {"fn", "const", "static", "type", "trait"})
            if pending_hidden and named:
                record(f"{parent}::{token}", line_number)
                if line.count("{") > line.count("}"):
                    suppress_depth = brace_depth
            reset_pending()
            brace_depth += line.count("{") - line.count("}")
            continue
        if pending_use is not None:
            pending_use += " " + line
            if "}" in line:
                rest = pending_use
                hidden, test_only = pending_hidden, pending_cfg_test_only
                if hidden and not test_only:
                    record_use_leaves(rest, parent_of(mod_stack, module_path), path, line_number, record)
                reset_pending()
            continue
        if line == "}":
            if mod_stack:
                mod_stack.pop()
            reset_pending()
            continue
        brace_depth += line.count("{") - line.count("}")
        if re.match(r"^(pub|fn|struct|enum|impl|trait|const|static|type|mod|use)\b", line):
            if pending_hidden and line.startswith("pub "):
                fail(f"{path}:{line_number}: unrecognized hidden item shape: {line}")
                raise SystemExit(1)
            reset_pending()

    return sites, mod_gates


def resolve_feature(
    symbol: str, direct: str | None, mod_gates: dict[tuple[str, str], str]
) -> str | None:
    if direct is not None:
        return direct
    parts = symbol.split("::")
    for index in range(len(parts) - 1, 0, -1):
        key = ("::".join(parts[:index]), parts[index])
        if key in mod_gates:
            return mod_gates[key]
    return None


def main() -> int:
    ledger_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LEDGER
    if not ledger_path.is_file():
        fail(f"missing ledger {ledger_path}")
        return 1
    with ledger_path.open(newline="") as ledger_file:
        reader = csv.DictReader((row for row in ledger_file if not row.startswith("#")), delimiter="\t")
        if reader.fieldnames != HEADER:
            fail(f"{ledger_path} header must be {' '.join(HEADER)}")
            return 1
        rows = list(reader)

    ledger: dict[tuple[str, str], dict[str, str]] = {}
    status = 0
    for line_number, row in enumerate(rows, 2):
        symbol, feature = row["symbol"], row["feature"]
        if not all(row[column] for column in HEADER):
            fail(f"{ledger_path}:{line_number} has an empty field")
            status = 1
        if feature not in ALLOWED_FEATURES:
            fail(f"{ledger_path}:{line_number} has unapproved feature {feature!r}")
            status = 1
        if (symbol, feature) in ledger:
            fail(f"{ledger_path}:{line_number} duplicates {(symbol, feature)}")
            status = 1
        ledger[(symbol, feature)] = row
    if status:
        return status

    all_sites: dict[str, str | None] = {}
    all_gates: dict[tuple[str, str], str] = {}
    for path in sorted(SRC.rglob("*.rs")):
        sites, gates = scan_file(path)
        all_sites.update(sites)
        all_gates.update(gates)

    actual: dict[tuple[str, str], None] = {}
    for symbol in sorted(all_sites):
        feature = resolve_feature(symbol, all_sites[symbol], all_gates)
        if feature is None:
            fail(f"{symbol} has no approved feature contract")
            status = 1
            continue
        if feature not in ALLOWED_FEATURES:
            fail(f"{symbol} resolves to unapproved feature contract {feature!r}")
            status = 1
            continue
        actual[(symbol, feature)] = None

    for symbol, feature in sorted(actual):
        if (symbol, feature) not in ledger:
            fail(f"unlisted hidden symbol {symbol} [{feature}] has no ledger entry")
            status = 1
    for symbol, feature in sorted(ledger):
        if (symbol, feature) not in actual:
            fail(f"stale ledger entry {symbol} [{feature}] matches no hidden site")
            status = 1
    if status == 0:
        print(f"PASS: {len(actual)} hidden symbols license-checked against {ledger_path.name}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
