#!/usr/bin/env bash
# Guard public migration terminology. Rust implementation details and ignored
# planning/report artifacts are intentionally outside this public surface.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

exec python3 - "${ROOT}" "$@" <<'PY'
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mdx",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    suffixes: frozenset[str] | None = None


RULES = (
    Rule("GpuFallbackPolicy", re.compile(r"\bGpuFallbackPolicy\b")),
    Rule("GpuPreferred policy wording", re.compile(r"\bGpuPreferred\b")),
    Rule("GpuOnly policy wording", re.compile(r"\bGpuOnly\b")),
    Rule("PartialNative", re.compile(r"\bPartialNative\b")),
    Rule(
        "NEXUS_SERVER_CPU_FALLBACK_* environment variables",
        re.compile(r"\bNEXUS_SERVER_CPU_FALLBACK_[A-Z0-9_]*\b"),
    ),
    Rule(
        "legacy fallback TOML settings",
        re.compile(
            r"(?:\bfallback\.(?:max_concurrent_statements|wait_timeout_ms)\b|"
            r"\[\s*fallback\s*\]|\bfallback\s*=\s*\{)",
            re.IGNORECASE,
        ),
        # A JSX `fallback={...}` prop is unrelated to the removed TOML
        # surface. Rendered Docusaurus documents are inspected as HTML.
        frozenset({".html", ".md", ".mdx", ".toml", ".txt"}),
    ),
    Rule("iceberg-cpu", re.compile(r"\biceberg-cpu\b", re.IGNORECASE)),
    Rule("IcebergProviderKind", re.compile(r"\bIcebergProviderKind\b")),
    Rule(
        "legacy CPU fallback contract tool or environment",
        re.compile(
            r"\b(?:cpu[-_]fallback[-_]contract[-_]validate|CPU_FALLBACK_WORK_DIR|"
            r"NEXUS_CPU_FALLBACK_[A-Z0-9_]*)\b",
            re.IGNORECASE,
        ),
    ),
    Rule(
        "cpu_fallback or cpu-fallback runtime scenario alias",
        # The trailing word-character guard preserves the stable machine field
        # cpu_fallback_after_native_start=false.
        re.compile(
            r"(?<![A-Za-z0-9_])cpu_fallback(?![A-Za-z0-9_])|"
            r"(?<![A-Za-z0-9])cpu-fallback(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    ),
)


def parse_args(default_root: Path, argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if removed fallback-policy terminology reaches maintained public "
            "documentation, examples, tool help/selectors, or the rendered website."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="repository root to check (default: the repository containing this script)",
    )
    return parser.parse_args(argv)


def require_path(path: Path, kind: str) -> None:
    if kind == "file" and path.is_file():
        return
    if kind == "directory" and path.is_dir():
        return
    raise RuntimeError(f"required {kind} is missing: {path}")


def text_files(path: Path, excluded_directories: frozenset[Path] = frozenset()) -> Iterable[Path]:
    if path.is_file():
        if path.suffix in TEXT_SUFFIXES:
            yield path
        return
    for directory, child_directories, filenames in os.walk(path):
        directory_path = Path(directory)
        child_directories[:] = sorted(
            child
            for child in child_directories
            if directory_path / child not in excluded_directories
        )
        for filename in sorted(filenames):
            candidate = directory_path / filename
            if candidate.suffix in TEXT_SUFFIXES:
                yield candidate


def nonpublic_rust_test(path: Path, source_root: Path) -> bool:
    if path.suffix != ".rs":
        return False
    relative = path.relative_to(source_root)
    return (
        "tests" in relative.parts
        or path.name == "tests.rs"
        or path.name.endswith("_tests.rs")
    )


def rust_without_test_items(text: str) -> str:
    """Blank test-only Rust items while preserving source line numbers."""
    rendered: list[str] = []
    skipping = False
    pending_test_item = False
    brace_depth = 0
    test_attribute = re.compile(r"^#\[[^\]]*\btest\b[^\]]*\]")
    test_module = re.compile(r"^(?:pub\s*(?:\([^)]*\))?\s+)?mod\s+tests\b")

    for line in text.splitlines():
        stripped = line.strip()
        if not skipping and not pending_test_item:
            if test_attribute.match(stripped) or test_module.match(stripped):
                pending_test_item = True
                rendered.append("")
                if "{" in line:
                    pending_test_item = False
                    brace_depth = line.count("{") - line.count("}")
                    skipping = brace_depth > 0
                continue
            rendered.append(line)
            continue

        rendered.append("")
        if pending_test_item:
            if "{" not in line:
                if stripped.endswith(";"):
                    pending_test_item = False
                continue
            pending_test_item = False
            brace_depth = line.count("{") - line.count("}")
            skipping = brace_depth > 0
            continue

        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            skipping = False

    return "\n".join(rendered)


def public_source_files(root: Path) -> list[Path]:
    required_files = [
        root / "README.md",
        root / "crates/datafusion-nexus-tools/Cargo.toml",
        root / "website/package.json",
    ]
    required_directories = [
        root / "crates/datafusion-nexus-examples",
        root / "crates/datafusion-nexus-tools/src/bin",
        root / "website",
    ]
    for path in required_files:
        require_path(path, "file")
    for path in required_directories:
        require_path(path, "directory")

    files: set[Path] = set(required_files)

    # Maintained prose at the repository boundary. /docs is ignored private
    # planning/report output, so it is intentionally not a source root.
    for relative in (".config/README.md", "docker/README.md", "fixture/README.md", "scripts/README.md"):
        candidate = root / relative
        if candidate.is_file():
            files.add(candidate)
    for candidate in (root / "crates").glob("*/README.md"):
        if candidate.is_file():
            files.add(candidate)

    # Public runnable examples, not library tests.
    files.update(text_files(root / "crates/datafusion-nexus-examples"))

    # Tool feature selectors and sources that back user-visible commands/help.
    # Dedicated Rust test files are not public surfaces.
    tool_source = root / "crates/datafusion-nexus-tools/src"
    for candidate in text_files(tool_source):
        if not nonpublic_rust_test(candidate, tool_source):
            files.add(candidate)

    # Source inputs for the published Docusaurus site. Generated dependencies
    # and output are excluded here and checked separately after a fresh build.
    website = root / "website"
    excluded = frozenset({website / "node_modules", website / "build", website / ".docusaurus"})
    files.update(text_files(website, excluded))

    return sorted(files)


def rendered_website_files(website: Path) -> list[Path]:
    build = website / "build"
    require_path(build, "directory")
    files = list(text_files(build))
    if not files:
        raise RuntimeError(f"rendered website has no text files to inspect: {build}")
    return files


def build_website(website: Path) -> None:
    print(f"==> Building Docusaurus website: {website}", flush=True)
    subprocess.run(["npm", "--prefix", str(website), "run", "build"], check=True)


def scan(paths: Iterable[Path], root: Path) -> list[str]:
    violations: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            violations.append(f"ERROR: cannot read {path}: {error}")
            continue
        except UnicodeDecodeError:
            # All selected source types are maintained text. Do not silently
            # skip a malformed public file that could hide a stale term.
            violations.append(f"ERROR: public text file is not UTF-8: {path}")
            continue
        if path.suffix == ".rs":
            text = rust_without_test_items(text)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule in RULES:
                if rule.suffixes is not None and path.suffix not in rule.suffixes:
                    continue
                if rule.pattern.search(line):
                    try:
                        display_path = path.relative_to(root)
                    except ValueError:
                        display_path = path
                    violations.append(f"{display_path}:{line_number}: {rule.name}")
    return violations


def main() -> int:
    default_root = Path(sys.argv[1]).resolve()
    args = parse_args(default_root, sys.argv[2:])
    root = args.root.resolve()
    try:
        require_path(root, "directory")
        website = root / "website"
        require_path(website, "directory")
        build_website(website)
        source_violations = scan(public_source_files(root), root)
        rendered_violations = scan(rendered_website_files(website), root)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: public terminology check could not complete: {error}", file=sys.stderr)
        return 2

    violations = source_violations + rendered_violations
    if violations:
        print("FAIL: removed public fallback terminology remains:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1

    print("OK: maintained public docs, examples, tool help/selectors, and rendered website use current terminology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
