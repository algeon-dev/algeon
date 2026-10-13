#!/usr/bin/env python3
"""Keep CC BY-NC derived MerRec data confined to the documentation site.

The repository is Apache-2.0, but the saved MerRec query results carry the
dataset's CC BY-NC 4.0 terms. They may live only under website/static/data,
where NOTICE and LICENSE-MERREC.md declare them. Every such file must name
its terms in a top-level ``license`` field so a copy taken out of the tree
still shows them, LICENSE-MERREC.md must list exactly the tracked files, and
the Docker packaging scripts must not reach into the website or fixture trees,
so no shipped image or crate bundles the data.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path


DATA_DIR = Path("website/static/data")
DATA_GLOB = "merrec-*.json"
LICENSE_FILE = DATA_DIR / "LICENSE-MERREC.md"
LICENSE_PREFIX = "CC-BY-NC-4.0"
NOTICE_GLOB = "website/static/data/merrec-*.json"
PACKAGING_DIR = Path("docker")
FORBIDDEN_PACKAGING_PATHS = ("website/static", "fixture/demo")


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True, text=True
    ).stdout
    return [Path(entry) for entry in output.split("\0") if entry]


def check(root: Path) -> list[str]:
    errors: list[str] = []
    tracked = tracked_files(root)
    derived = [path for path in tracked if fnmatch.fnmatch(path.name, DATA_GLOB)]

    for path in derived:
        if path.parent != DATA_DIR:
            errors.append(f"{path}: derived MerRec data must live under {DATA_DIR}")
            continue
        try:
            data = json.loads((root / path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            errors.append(f"{path}: unreadable JSON ({error})")
            continue
        license_field = data.get("license") if isinstance(data, dict) else None
        if not isinstance(license_field, str) or not license_field.startswith(LICENSE_PREFIX):
            errors.append(f"{path}: top-level \"license\" must start with {LICENSE_PREFIX}")

    expected = {path.name for path in derived if path.parent == DATA_DIR}
    license_path = root / LICENSE_FILE
    if license_path.is_file():
        listed = set(re.findall(r"`(merrec-[\w.-]+\.json)`", license_path.read_text(encoding="utf-8")))
        for name in sorted(expected - listed):
            errors.append(f"{LICENSE_FILE}: missing {name}")
        for name in sorted(listed - expected):
            errors.append(f"{LICENSE_FILE}: lists untracked {name}")
    elif expected:
        errors.append(f"{LICENSE_FILE}: missing")

    notice = root / "NOTICE"
    if expected and (not notice.is_file() or NOTICE_GLOB not in notice.read_text(encoding="utf-8")):
        errors.append(f"NOTICE: must declare {NOTICE_GLOB}")

    for path in tracked:
        if path.parent != PACKAGING_DIR or not (root / path).is_file():
            continue
        text = (root / path).read_text(encoding="utf-8", errors="replace")
        for forbidden in FORBIDDEN_PACKAGING_PATHS:
            if forbidden in text:
                errors.append(f"{path}: packaging must not reference {forbidden}")
    return errors


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    errors = check(root)
    for error in errors:
        print(f"FAIL: third-party data policy: {error}", file=sys.stderr)
    if errors:
        raise SystemExit(1)
    count = len([path for path in tracked_files(root) if fnmatch.fnmatch(path.name, DATA_GLOB)])
    print(f"third-party data policy: {count} MerRec captures confined to {DATA_DIR} with declared terms")


if __name__ == "__main__":
    main()
