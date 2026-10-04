#!/usr/bin/env python3

import csv
import re
import sys
from pathlib import Path


HEADER = ["public_symbol", "owner_type", "consumer", "decision", "facade_path", "reason"]
DECISIONS = {"RE-EXPORT", "ADAPT", "HIDE", "INTERNAL CONSUMER"}
# Types owned by the planning, runtime, and engine crates, as cargo public-api
# prints them: by defining path, never through a re-exporting crate.
OWNER_TYPE = re.compile(r"query_(?:engine|planning|runtime)(?:::[A-Za-z_][A-Za-z0-9_]*)+")


def fail(message: str) -> None:
    print(f"ERROR: public API facade policy: {message}", file=sys.stderr)


def main() -> int:
    if len(sys.argv) != 3:
        fail("usage: check_public_api_facade.py SNAPSHOT LEDGER")
        return 2

    snapshot_path = Path(sys.argv[1])
    ledger_path = Path(sys.argv[2])
    if not snapshot_path.is_file():
        fail(f"missing snapshot {snapshot_path}")
        return 1
    if not ledger_path.is_file():
        fail(f"missing disposition ledger {ledger_path}")
        return 1

    snapshot = snapshot_path.read_text()
    with ledger_path.open(newline="") as ledger_file:
        reader = csv.DictReader(ledger_file, delimiter="\t")
        if reader.fieldnames != HEADER:
            fail(f"{ledger_path} header must be {' '.join(HEADER)}")
            return 1
        rows = list(reader)

    status = 0
    by_type: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, 2):
        owner_type = row["owner_type"]
        decision = row["decision"]
        if not all(row[column] for column in HEADER):
            fail(f"{ledger_path}:{line_number} has an empty field")
            status = 1
        if decision not in DECISIONS:
            fail(f"{ledger_path}:{line_number} has unknown decision {decision!r}")
            status = 1
        if owner_type in by_type:
            fail(f"{ledger_path}:{line_number} duplicates disposition for {owner_type}")
            status = 1
        by_type[owner_type] = row

    live_types = set(OWNER_TYPE.findall(snapshot))

    def row_is_live(row: dict[str, str]) -> bool:
        return row["owner_type"] in live_types or f"pub use {row['public_symbol']}" in snapshot

    for owner_type in sorted(live_types):
        row = by_type.get(owner_type)
        if row is None:
            fail(f"live owner type has no disposition: {owner_type}")
            status = 1
            continue
        if row["decision"] != "RE-EXPORT":
            fail(f"{owner_type} is live but disposition is {row['decision']}")
            status = 1
            continue
        facade_path = row["facade_path"]
        if f"pub use {facade_path}" not in snapshot:
            fail(f"approved facade path is not reachable: {facade_path} for {owner_type}")
            status = 1

    for owner_type, row in sorted(by_type.items()):
        live = row_is_live(row)
        if row["decision"] == "RE-EXPORT" and not live:
            fail(f"stale RE-EXPORT disposition is not used by the live surface: {owner_type}")
            status = 1
        if row["decision"] in {"ADAPT", "HIDE", "INTERNAL CONSUMER"} and live:
            fail(f"forbidden {row['decision']} type reappeared in the live surface: {owner_type}")
            status = 1

    if status == 0:
        live_ledger_identities = {owner_type for owner_type, row in by_type.items() if row_is_live(row)}
        print(f"PASS: {len(live_ledger_identities)} live owner types have unique approved facade reachability")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
