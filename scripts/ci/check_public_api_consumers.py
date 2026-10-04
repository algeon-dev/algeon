#!/usr/bin/env python3
"""Check each hidden-API ledger's consumer column against measured uses.

The hidden-API ledgers (``<LEDGER_DIR>/<package>-hidden.tsv``) name each
`#[doc(hidden)] pub` symbol's consumers, and ``check_public_api_hidden.py``
matches their rows to source sites. This check measures the consumers: it
copies WORKSPACE's tracked and unignored files to ``OUT_DIR/tree``, puts
``#[deprecated(note = "algeon-public-api-probe <symbol>")]`` on every hidden
definition, and runs ``cargo check`` with ``--force-warn=deprecated``, so rustc
reports each use together with the compilation unit that contains it. rustc
ignores deprecation on a re-export, so a hidden ``pub use`` marks the
definition it names. Deprecation on a module reaches its unattributed
children, and the probe marks the definitions of unattributed items a hidden
module re-exports with the module's symbol too. rustc reports a path's last
segment only, so a module row's consumers are the packages that name anything
under the module.

A use counts for the package that compiles it:

* ``prod``: the use compiles into a library or binary target of a build with
  every feature except ``test-hooks``;
* ``tools``: the same, in a package listed in ``TOOL_PACKAGES``;
* ``tests``: any other use: test, bench, and example targets, ``cfg(test)``
  code, and code that compiles only with ``test-hooks``.

Uses inside the owning package's library, including its unit tests, are not
consumers; its integration tests are. A row's ``prod``, ``tools``, and
``tests`` clauses must equal the measured packages. A row no package uses
must name, in ``exposed-by``, the public symbols whose signatures reach it;
otherwise it can be ``pub(crate)``.

Usage:
  python3 scripts/ci/check_public_api_consumers.py WORKSPACE LEDGER_DIR OUT_DIR

Writes ``OUT_DIR/consumers.tsv`` (package, symbol, feature, measured
consumers) and exits nonzero on any mismatch.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_public_api_hidden import DefinitionIndex, ScanError, parse_consumers, read_ledger  # noqa: E402

TOOL_PACKAGES = frozenset({"bench", "examples", "tools", "workloads"})
TEST_ONLY_FEATURE = "test-hooks"
MEASURED_KINDS = ("prod", "tools", "tests")
MARKER = "algeon-public-api-probe "
SEPARATOR = " | "
LIBRARY_KINDS = frozenset({"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro", "bin"})


def copy_tree(workspace: Path, tree: Path) -> None:
    listed = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "-s"],
        check=True,
        capture_output=True,
    ).stdout
    # `-s` prints `mode object stage\tpath` for tracked files; untracked files
    # have no stage line. Gitlinks (mode 160000) are the native submodules:
    # they are linked rather than copied, because CPU contract tests
    # `include_str!` their sources.
    files, submodules = [], []
    for entry in listed.split(b"\0"):
        if not entry:
            continue
        if b"\t" in entry:
            stage, path = entry.split(b"\t", 1)
            if stage.startswith(b"160000"):
                submodules.append(path.decode())
                continue
        else:
            path = entry
        files.append(path.decode())
    if tree.exists():
        shutil.rmtree(tree)
    for relative in submodules:
        (tree / relative).parent.mkdir(parents=True, exist_ok=True)
        (tree / relative).symlink_to(workspace / relative, target_is_directory=True)
    for relative in files:
        source = workspace / relative
        if source.is_file():
            (tree / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, tree / relative)


def ledger_crates(metadata: dict, ledger_dir: Path) -> list[tuple[str, Path, str, Path]]:
    """(package, source directory, library root module, ledger) for each ledger."""
    libraries = {
        package["name"]: target
        for package in metadata["packages"]
        for target in package["targets"]
        if "lib" in target["kind"]
    }
    crates = []
    for ledger in sorted(ledger_dir.glob("*-hidden.tsv")):
        package = ledger.name.removesuffix("-hidden.tsv")
        if package not in libraries:
            raise SystemExit(f"ERROR: {ledger} names no workspace library package {package!r}")
        library = libraries[package]
        crates.append((package, Path(library["src_path"]).parent, library["name"], ledger))
    return crates


def plan_marks(crates: list[tuple[str, Path, str, Path]]):
    """Return {file: {offset: [symbols]}}, each symbol's (package, feature), module symbols, and unprobed symbols."""
    marks: dict[Path, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    owners: dict[str, tuple[str, str]] = {}
    modules: set[str] = set()
    unprobed: dict[str, str] = {}
    for package, source_dir, root, _ in crates:
        index = DefinitionIndex(source_dir, root)
        try:
            sites = index.scan()
        except ScanError as error:
            raise SystemExit(f"ERROR: {error}")
        for (symbol, feature), where in sites.items():
            owners[symbol] = (package, feature)
            if tuple(symbol.split("::")) in index.modules:
                modules.add(symbol)
            site = where[0]
            if site.source is None:
                if " as " in symbol:
                    unprobed[symbol] = "a trait impl member; rustc attributes its uses to the trait item"
                    continue
                targets = [(site.path, site.pos)]
            elif symbol.rsplit("::", 1)[-1].startswith("*("):
                unprobed[symbol] = "a glob re-export names no single definition"
                continue
            else:
                targets = index.resolve(site.module, site.source)
                if not targets:
                    unprobed[symbol] = f"re-export of `{'::'.join(site.source)}` resolves outside the crate"
                    continue
            for path, pos in targets:
                marks[path][pos].append(symbol)
        # A hidden module's row also covers the unattributed items it
        # re-exports, whose definitions its own deprecation does not reach.
        for symbol in [symbol for symbol in modules if owners[symbol][0] == package]:
            module = tuple(symbol.split("::"))
            for name, hidden, _, _, source in index.public_children[module]:
                if source is None or hidden or name == "*":
                    continue
                for path, pos in index.resolve(module, source) or []:
                    marks[path][pos].append(symbol)
    return marks, owners, modules, unprobed


def apply_marks(marks: dict[Path, dict[int, list[str]]]) -> None:
    for path, offsets in marks.items():
        text = path.read_text()
        for pos in sorted(offsets, reverse=True):
            note = MARKER + SEPARATOR.join(sorted(set(offsets[pos])))
            text = f'{text[:pos]}#[deprecated(note = "{note}")] {text[pos:]}'
        path.write_text(text)


def cargo_check(tree: Path, target_dir: Path, selection: list[str], log: Path) -> list[dict]:
    env = dict(os.environ)
    env.update(
        {
            "CARGO_TARGET_DIR": str(target_dir),
            # Skips native linking; `cargo check` needs no RAPIDS libraries.
            "DOCS_RS": "1",
            "RUSTFLAGS": "--cap-lints=warn --force-warn=deprecated",
        }
    )
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    command = ["cargo", "check", "--workspace", "--locked", *selection, "--message-format=json"]
    with log.open("w") as stderr:
        result = subprocess.run(command, cwd=tree, env=env, stdout=subprocess.PIPE, stderr=stderr, text=True)
    messages = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode != 0:
        for message in messages:
            if message.get("reason") == "compiler-message" and message["message"]["level"] == "error":
                print(message["message"]["rendered"], file=sys.stderr)
        raise SystemExit(f"ERROR: `{' '.join(command)}` failed in {tree}; see {log}")
    return messages


def probe_uses(messages: list[dict], packages: dict[str, str]):
    """Yield (symbol, consumer package, target kinds, span) for each probe warning."""
    for message in messages:
        if message.get("reason") != "compiler-message":
            continue
        diagnostic = message["message"]
        code = (diagnostic.get("code") or {}).get("code")
        if code != "deprecated" or MARKER not in diagnostic["message"]:
            continue
        symbols = diagnostic["message"].split(MARKER, 1)[1].split(SEPARATOR)
        span = next(span for span in diagnostic["spans"] if span["is_primary"])
        key = (span["file_name"], span["line_start"], span["column_start"])
        package = packages[message["manifest_path"]]
        for symbol in symbols:
            yield symbol.strip(), package, frozenset(message["target"]["kind"]), key


def measure(tree: Path, out: Path, metadata: dict, owners: dict[str, tuple[str, str]]):
    packages = {package["manifest_path"]: package["name"] for package in metadata["packages"]}
    production_features = [
        f"{package['name']}/{feature}"
        for package in metadata["packages"]
        for feature in package["features"]
        if feature != TEST_ONLY_FEATURE
    ]
    selection = ["--lib", "--bins"]
    if production_features:
        selection += ["--features", ",".join(production_features)]
    target_dir = out / "target"
    library = cargo_check(tree, target_dir, selection, out / "check-production.log")
    every = cargo_check(tree, target_dir, ["--all-features", "--all-targets"], out / "check-all-targets.log")

    production_spans = {(symbol, package, key) for symbol, package, _, key in probe_uses(library, packages)}
    consumers: dict[str, dict[str, set[str]]] = defaultdict(lambda: {kind: set() for kind in MEASURED_KINDS})
    for symbol, package, kinds, key in probe_uses(every, packages):
        in_library = bool(kinds & LIBRARY_KINDS)
        if package == owners[symbol][0] and in_library:
            continue
        if in_library and (symbol, package, key) in production_spans:
            consumers[symbol]["tools" if package in TOOL_PACKAGES else "prod"].add(package)
        else:
            consumers[symbol]["tests"].add(package)
    return consumers


def format_consumers(groups: dict[str, set[str]]) -> str:
    clauses = [f"{kind}: {', '.join(sorted(groups[kind]))}" for kind in MEASURED_KINDS if groups[kind]]
    return "; ".join(clauses) if clauses else "none"


def compare(ledger_path: Path, row: dict[str, str], measured: dict[str, set[str]]) -> list[str]:
    symbol = row["symbol"]
    recorded = parse_consumers(row["consumer"])
    errors = []
    for kind in MEASURED_KINDS:
        listed, used = set(recorded.get(kind, ())), measured[kind]
        if listed != used:
            errors.append(
                f"{ledger_path.name}: {symbol}: {kind} consumers are {', '.join(sorted(used)) or 'none'}, "
                f"ledger lists {', '.join(sorted(listed)) or 'none'}"
            )
    named = any(measured[kind] for kind in MEASURED_KINDS)
    if named and "exposed-by" in recorded:
        errors.append(f"{ledger_path.name}: {symbol}: exposed-by applies only to a symbol no package names")
    if not named and "exposed-by" not in recorded:
        errors.append(f"{ledger_path.name}: {symbol}: no package names it; make it pub(crate) or record exposed-by")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    workspace, ledger_dir, out = (Path(arg).resolve() for arg in argv)
    tree = out / "tree"
    out.mkdir(parents=True, exist_ok=True)
    copy_tree(workspace, tree)
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
            cwd=tree,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    crates = ledger_crates(metadata, ledger_dir)
    marks, owners, modules, unprobed = plan_marks(crates)
    apply_marks(marks)
    consumers = measure(tree, out, metadata, owners)
    for module in modules:
        for symbol in [symbol for symbol in consumers if symbol.startswith(f"{module}::")]:
            for kind in MEASURED_KINDS:
                consumers[module][kind] |= consumers[symbol][kind]

    with (out / "consumers.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["package", "symbol", "feature", "measured"])
        for symbol, (package, feature) in sorted(owners.items(), key=lambda entry: (entry[1][0], entry[0])):
            writer.writerow([package, symbol, feature, format_consumers(consumers[symbol])])

    errors = [f"cannot probe {symbol}: {reason}" for symbol, reason in sorted(unprobed.items())]
    compared = 0
    for _, _, _, ledger_path in crates:
        ledger = read_ledger(ledger_path)
        if ledger is None:
            errors.append(f"{ledger_path.name} is malformed; see the errors above")
            continue
        for (symbol, _), row in sorted(ledger.items()):
            if symbol not in owners:
                errors.append(f"{ledger_path.name}: {symbol} matches no hidden site")
            elif symbol not in unprobed:
                compared += 1
                errors.extend(compare(ledger_path, row, consumers[symbol]))
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"Measured consumers: {out / 'consumers.tsv'}", file=sys.stderr)
        return 1
    print(f"PASS: {compared} hidden symbols' consumers match their ledgers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
