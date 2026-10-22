#!/usr/bin/env bash
# Fast static check for AI writing slop (em-dashes and banned phrases)
# and front matter YAML syntax.
# Scans markdown / prose documentation for prohibited characters and tropes.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"

python3 - "$@" <<'PY'
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

USAGE = """Usage: check_prose_slop.sh [OPTIONS] [PATHS...]

Checks markdown documentation for AI writing slop (em-dashes and banned phrases)
and validates YAML front matter syntax.
When no paths or git flags are given, defaults to scanning website/docs/.
The gitignored docs/ and note/ trees are outside git and CI and are never
scanned by default.

Options:
  --staged            Check files currently staged in git
  --changed           Check files modified in working tree against HEAD
  --exclude <regex>   Regex pattern of paths to exclude (can be repeated)
  -h, --help          Show this help message
"""

DEFAULT_TARGETS = ["website/docs"]
DOC_EXTENSIONS = {".md", ".mdx", ".markdown", ".txt"}

BANNED_PATTERNS = [
    ("em-dash ('—')", re.compile(r"\u2014")),
    ("banned phrase 'delve'", re.compile(r"\bdelve(?:s|d|ing)?\b", re.IGNORECASE)),
    ("banned phrase 'seamlessly'", re.compile(r"\bseamlessly\b", re.IGNORECASE)),
    ("banned phrase 'pivotal'", re.compile(r"\bpivotal\b", re.IGNORECASE)),
    ("banned phrase 'testament to'", re.compile(r"\btestament to\b", re.IGNORECASE)),
    ("banned phrase 'robust and scalable'", re.compile(r"\brobust and scalable\b", re.IGNORECASE)),
    ("banned phrase 'in today\'s data landscape'", re.compile(r"\bin today['\u2019]s data landscape\b", re.IGNORECASE)),
    ("banned phrase 'revolutionize'", re.compile(r"\brevolutionize(?:s|d|ing)?\b", re.IGNORECASE)),
    ("banned phrase 'harness the power of'", re.compile(r"\bharness the power of\b", re.IGNORECASE)),
]

def parse_args(argv):
    targets = []
    excludes = []
    mode = "paths"
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in ("-h", "--help"):
            print(USAGE)
            sys.exit(0)
        elif arg == "--staged":
            mode = "staged"
        elif arg == "--changed":
            mode = "changed"
        elif arg == "--exclude":
            idx += 1
            if idx >= len(argv):
                sys.exit("error: --exclude requires an argument")
            excludes.append(re.compile(argv[idx]))
        elif arg.startswith("--exclude="):
            excludes.append(re.compile(arg.split("=", 1)[1]))
        elif arg.startswith("-"):
            sys.exit(f"error: unknown option {arg}\n{USAGE}")
        else:
            targets.append(arg)
        idx += 1
    return mode, targets, excludes

def get_git_files(diff_args):
    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=d"] + diff_args,
            text=True
        )
        return [f.strip() for f in output.splitlines() if f.strip()]
    except Exception as e:
        sys.exit(f"error running git diff: {e}")

def collect_files(mode, targets, excludes):
    files = []
    if mode == "staged":
        raw = get_git_files(["--cached"])
        for f in raw:
            p = Path(f)
            if p.suffix.lower() in DOC_EXTENSIONS and p.is_file():
                files.append(p)
    elif mode == "changed":
        raw = get_git_files(["HEAD"])
        for f in raw:
            p = Path(f)
            if p.suffix.lower() in DOC_EXTENSIONS and p.is_file():
                files.append(p)
    else:
        for target in targets or DEFAULT_TARGETS:
            p = Path(target)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for root, _, filenames in os.walk(p):
                    for fn in filenames:
                        fp = Path(root) / fn
                        if fp.suffix.lower() in DOC_EXTENSIONS:
                            files.append(fp)
            else:
                sys.exit(f"error: target path does not exist: {target}")

    filtered = []
    for f in sorted(set(files)):
        rel = f.as_posix()
        if any(exc.search(rel) for exc in excludes):
            continue
        filtered.append(f)
    return filtered

def scan_file(path):
    violations = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as err:
        return [f"cannot read file: {err}"]

    # Validate YAML front matter on markdown documents starting with '---'
    if path.suffix.lower() in {".md", ".mdx", ".markdown"} and lines and lines[0].strip() == "---":
        closing_idx = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() in ("---", "..."):
                closing_idx = idx
                break
        if closing_idx is None:
            violations.append(f"{path}:1: unclosed front matter delimiter '---'")
        elif yaml is None:
            violations.append(
                f"{path}:1: PyYAML (python3-yaml) is required to validate front matter"
            )
        else:
            fm_text = "".join(lines[1:closing_idx])
            try:
                parsed = yaml.safe_load(fm_text)
                if parsed is not None and not isinstance(parsed, dict):
                    violations.append(
                        f"{path}:1: invalid front matter: expected YAML mapping (key-value pairs)"
                    )
            except yaml.YAMLError as err:
                mark = getattr(err, "problem_mark", None)
                line_no = (mark.line + 2) if mark else 1
                col_no = (mark.column + 1) if mark else 1
                problem = getattr(err, "problem", str(err))
                violations.append(
                    f"{path}:{line_no}:{col_no}: invalid front matter YAML: {problem}"
                )
            except Exception as err:
                violations.append(
                    f"{path}:1: invalid front matter YAML: {err}"
                )

    in_fence = False
    fence_tag = ""

    for line_idx, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()

        # Track fenced code blocks (``` or ~~~)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            tag = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_tag = tag
                continue
            elif stripped.startswith(fence_tag):
                in_fence = False
                fence_tag = ""
                continue

        if in_fence:
            continue

        # Strip inline code spans (e.g. `—`, `delve`) to avoid flagging definitions
        prose_line = re.sub(r"`[^`\n]*`", " ", line)

        for label, pattern in BANNED_PATTERNS:
            match = pattern.search(prose_line)
            if match:
                snippet = line.strip()
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                violations.append(
                    f"{path}:{line_idx}: {label}: \"{snippet}\""
                )

    return violations

def main():
    mode, targets, excludes = parse_args(sys.argv[1:])
    files = collect_files(mode, targets, excludes)

    total_violations = []
    for f in files:
        violations = scan_file(f)
        if violations:
            total_violations.extend(violations)

    if total_violations:
        print("\n".join(total_violations), file=sys.stderr)
        print(
            f"\nFAIL: check_prose_slop found {len(total_violations)} violation(s) "
            f"across {len(files)} file(s).",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"prose anti-slop checks passed: {len(files)} file(s) checked.")
    sys.exit(0)

if __name__ == "__main__":
    main()
PY
