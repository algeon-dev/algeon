#!/usr/bin/env python3
"""Stage the license texts of every component bundled in the Docker payload.

Writes <dist>/licenses/ with the Algeon LICENSE and NOTICE, the LICENSE of each
RAPIDS submodule whose libraries are staged, the license files of the native
packages that CPM or FetchContent fetched into each staged layer's build tree,
and the license files of every Rust package that `cargo metadata` shows linked
into algeon_server. INVENTORY.tsv maps each component to the staged payload
files that bundle it, and FORKS.tsv names the repository and commit of each
staged RAPIDS submodule. A staged file without an owning component, a
component without a license source, or a modified submodule fails the staging.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Layer:
    submodule: str
    libraries: tuple[str, ...]


# Library patterns are matched against staged file names under lib/.
LAYERS = {
    "cudf": Layer(
        "components/cudf/cudf",
        ("libcudf.so*", "librmm.so*", "librapids_logger.so*", "libkvikio.so*", "libnvcomp*.so*"),
    ),
    "cuvs": Layer("components/cuvs/cuvs", ("libcuvs.so*", "libcuvs_c.so*")),
    "cugraph": Layer("components/cugraph/cugraph", ("libcugraph*.so*",)),
    "cuml": Layer("components/cuml/cuml", ("libcuml.so*", "libtreelite.so*", "libnvforest.so*")),
}
# Algeon's own artifacts: the server and the C++ shims built from this repository.
ALGEON_FILES = ("bin/algeon_server", "lib/libcudf_rust.so", "lib/libcudf_rust_rmm_interop.so", "lib/libcuml_rust.so*")

# GoogleTest builds only native test binaries and rapids-cmake holds CMake
# modules run at configure time; neither reaches a staged library.
NOT_SHIPPED_FETCHES = {"gtest", "googletest", "rapids-cmake"}

LICENSE_NAME = re.compile(r"(?i)^(copying|notice|copyright|unlicense)|licen[cs]e")
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inl", ".rs", ".py", ".go", ".js",
    ".cmake", ".in", ".sh", ".toml", ".json", ".yml", ".yaml", ".html", ".css",
}
# Bindings, documentation, tests, benchmarks, examples and tools never reach
# the linked library, and their license files (GPL command-line tools, web
# assets) would misstate what the image ships.
UNSHIPPED_DIRS = {
    ".git", ".github", "assets", "bench", "benches", "benchmarks", "bin", "ci", "cli", "contrib",
    "dart", "doc", "docs", "example", "examples", "googlemock", "googletest", "gtest", "python",
    "r", "swift", "test", "testing", "tests", "tools",
}
# Standard texts for crates that declare an SPDX license but ship no license
# file. Apache-2.0 is the repository's own LICENSE. MIT and BSD-3-Clause
# (NOTICE_TEXTS) require the copyright notice, so their texts carry
# COPYRIGHT_MARKER, which the staging replaces with the crate's cargo metadata
# authors, or with its name and repository when it lists none.
COPYRIGHT_MARKER = "{copyright_holders}"
STANDARD_TEXTS = {
    "Apache-2.0": "LICENSE",
    "BSD-3-Clause": "docker/license-texts/BSD-3-Clause.txt",
    "MIT": "docker/license-texts/MIT.txt",
}
NOTICE_TEXTS = {"BSD-3-Clause", "MIT"}
INVENTORY_HEADER = ("kind", "name", "version", "declared_license", "source", "covers", "license_files")
FORKS_HEADER = ("layer", "repository", "commit", "changes")
FORK_CHANGES_URL = "https://algeon.dev/reference/forks"


class LicenseError(RuntimeError):
    """A bundled component has no license source, or a staged file has no owner."""


def is_license_file(path: Path) -> bool:
    return path.is_file() and bool(LICENSE_NAME.search(path.name)) and path.suffix.lower() not in SOURCE_SUFFIXES


def license_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (top-level license files, license files of vendored code below the top level)."""
    top = sorted(entry for entry in root.iterdir() if is_license_file(entry))
    nested: list[Path] = []
    pending = sorted(entry for entry in root.iterdir() if entry.is_dir() and entry.name.lower() not in UNSHIPPED_DIRS)
    while pending:
        directory = pending.pop(0)
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                if entry.name.lower() not in UNSHIPPED_DIRS and not entry.is_symlink():
                    pending.append(entry)
            elif is_license_file(entry) or (directory.name.lower() == "licenses" and entry.is_file()):
                nested.append(entry)
    return top, nested


def copy_tree_files(files: list[Path], source_root: Path, destination: Path, licenses_dir: Path) -> list[str]:
    copied = []
    for path in files:
        # A license-file outside the package directory keeps only its name.
        relative = path.relative_to(source_root) if path.is_relative_to(source_root) else Path(path.name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(str(target.relative_to(licenses_dir)))
    return copied


def parse_cmake_cache(build_dir: Path) -> dict[str, str]:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        raise LicenseError(f"{cache_path} not found; the license sources of a staged native layer come from its build tree")
    entries = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.fullmatch(r"([^#/][^:=]*):[A-Z]+=(.*)", line)
        if match:
            entries[match.group(1)] = match.group(2)
    return entries


def fetched_packages(build_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (name, version, source dir) for each package CPM or FetchContent fetched into the build tree."""
    cache = parse_cmake_cache(build_dir)
    packages: dict[Path, tuple[str, str]] = {}
    for key, value in cache.items():
        match = re.fullmatch(r"CPM_PACKAGE_(.+)_SOURCE_DIR", key)
        if match and value:
            name = match.group(1)
            packages[Path(value).resolve()] = (name, cache.get(f"CPM_PACKAGE_{name}_VERSION", ""))
    # nvCOMP's binary and some transitive sources come through FetchContent
    # without a CPM entry; their checkouts sit beside the CPM ones.
    base = cache.get("FETCHCONTENT_BASE_DIR")
    if base and Path(base).is_dir():
        for source in sorted(Path(base).glob("*-src")):
            if source.is_dir():
                packages.setdefault(source.resolve(), (source.name.removesuffix("-src"), ""))
    result = []
    for source, (name, version) in sorted(packages.items(), key=lambda item: item[1][0].lower()):
        if name.lower() in NOT_SHIPPED_FETCHES or source.name.removesuffix("-src").lower() in NOT_SHIPPED_FETCHES:
            continue
        if not source.is_dir():
            raise LicenseError(f"{build_dir} fetched {name} from {source}, which no longer exists")
        result.append((name, version, source))
    return result


def submodule_commit(repo_root: Path, submodule: str) -> str:
    """Return the checked-out commit of a submodule that has no modified tracked files."""
    path = repo_root / submodule

    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(path), *arguments], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise LicenseError(f"cannot read the git state of submodule {submodule}: {error}") from error

    if git("status", "--porcelain", "--untracked-files=no"):
        raise LicenseError(f"submodule {submodule} has modified files; the image must name the exact commit it ships")
    return git("rev-parse", "HEAD")


def copyright_holders(package: dict[str, Any]) -> str:
    authors = [author for author in package.get("authors") or [] if author.strip()]
    if authors:
        return ", ".join(authors)
    holders = f"the {package['name']} authors"
    return f"{holders} ({package['repository']})" if package.get("repository") else holders


def standard_text(repo_root: Path, identifier: str, package: dict[str, Any]) -> str:
    path = repo_root / STANDARD_TEXTS[identifier]
    if not path.is_file():
        raise LicenseError(f"the standard license text {path} is missing")
    text = path.read_text(encoding="utf-8")
    if identifier in NOTICE_TEXTS and COPYRIGHT_MARKER not in text:
        raise LicenseError(f"the standard {identifier} text {path} lacks the {COPYRIGHT_MARKER} copyright line")
    text = text.replace(COPYRIGHT_MARKER, copyright_holders(package))
    if re.search(r"<(year|copyright holders|owner)>|\{copyright_holders\}", text):
        raise LicenseError(f"the standard {identifier} text for {package['name']} still holds a placeholder")
    return text


def parse_spdx(expression: str) -> Any:
    tokens = re.findall(r"\(|\)|/|[A-Za-z0-9.+-]+", expression)
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> str:
        nonlocal position
        token = tokens[position]
        position += 1
        return token

    def parse_or() -> Any:
        terms = [parse_and()]
        while peek() in ("OR", "/"):
            take()
            terms.append(parse_and())
        return ("or", terms) if len(terms) > 1 else terms[0]

    def parse_and() -> Any:
        factors = [parse_with()]
        while peek() == "AND":
            take()
            factors.append(parse_with())
        return ("and", factors) if len(factors) > 1 else factors[0]

    def parse_with() -> Any:
        if peek() == "(":
            take()
            inner = parse_or()
            if peek() != ")":
                raise LicenseError(f"unbalanced SPDX expression {expression!r}")
            take()
            return inner
        identifier = take()
        if peek() == "WITH":
            take()
            return ("with", identifier, take())
        return ("id", identifier)

    try:
        tree = parse_or()
    except IndexError as error:
        raise LicenseError(f"cannot parse SPDX expression {expression!r}") from error
    if position != len(tokens):
        raise LicenseError(f"cannot parse SPDX expression {expression!r}")
    return tree


def standard_text_choice(tree: Any) -> list[str] | None:
    """Pick the license identifiers to ship: every AND operand, the first OR alternative with known texts."""
    kind = tree[0]
    if kind == "id":
        return [tree[1]] if tree[1] in STANDARD_TEXTS else None
    if kind == "with":
        return None
    if kind == "and":
        chosen: list[str] = []
        for factor in tree[1]:
            part = standard_text_choice(factor)
            if part is None:
                return None
            chosen.extend(part)
        return chosen
    for term in tree[1]:
        part = standard_text_choice(term)
        if part is not None:
            return part
    return None


def linked_rust_packages(metadata: dict[str, Any], root_name: str) -> list[dict[str, Any]]:
    """Packages reachable from the root through normal dependency edges, excluding proc macros and their deps."""
    packages = {package["id"]: package for package in metadata["packages"]}
    roots = [package_id for package_id, package in packages.items() if package["name"] == root_name and package.get("source") is None]
    if len(roots) != 1:
        raise LicenseError(f"cargo metadata has {len(roots)} workspace packages named {root_name}")
    resolve = metadata.get("resolve")
    if not resolve:
        raise LicenseError("cargo metadata has no dependency resolution; it must be captured without --no-deps")
    nodes = {node["id"]: node for node in resolve["nodes"]}
    seen = {roots[0]}
    pending = [roots[0]]
    while pending:
        node = nodes[pending.pop()]
        for dependency in node.get("deps", []):
            if not any(kind.get("kind") is None for kind in dependency.get("dep_kinds", [])):
                continue
            package_id = dependency["pkg"]
            if package_id in seen:
                continue
            seen.add(package_id)
            if any("proc-macro" in target.get("kind", []) for target in packages[package_id].get("targets", [])):
                continue
            pending.append(package_id)
    return [
        packages[package_id]
        for package_id in sorted(seen - {roots[0]})
        if not any("proc-macro" in target.get("kind", []) for target in packages[package_id].get("targets", []))
    ]


def stage(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    dist_dir = args.dist_dir.resolve()
    licenses_dir = dist_dir / "licenses"
    if licenses_dir.exists():
        raise LicenseError(f"{licenses_dir} already exists; stage into a fresh dist directory")

    layers: dict[str, Path] = {}
    for value in args.layer:
        name, separator, build_dir = value.partition("=")
        if not separator or name not in LAYERS or name in layers:
            raise LicenseError(f"--layer expects one of {sorted(LAYERS)} as <name>=<build dir>, got {value!r}")
        layers[name] = Path(build_dir)
    if "cudf" not in layers:
        raise LicenseError("every payload stages the cudf layer")

    staged = sorted(
        str(path.relative_to(dist_dir))
        for directory in ("bin", "lib")
        for path in (dist_dir / directory).iterdir()
        if path.is_file()
    )

    def covered(patterns: tuple[str, ...]) -> list[str]:
        return [path for path in staged if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)]

    algeon_covers = covered(ALGEON_FILES)
    layer_covers = {name: covered(tuple(f"lib/{pattern}" for pattern in LAYERS[name].libraries)) for name in layers}
    owned = set(algeon_covers).union(*layer_covers.values())
    orphans = [path for path in staged if path not in owned]
    if orphans:
        raise LicenseError(f"staged files belong to no licensed component: {orphans}")
    for name, covers in layer_covers.items():
        if not covers:
            raise LicenseError(f"the {name} layer was selected but stages no library")

    rows: list[tuple[str, ...]] = []
    licenses_dir.mkdir(parents=True)

    def require(path: Path, what: str) -> Path:
        if not path.is_file():
            raise LicenseError(f"{what} {path} is missing")
        return path

    algeon_files = []
    for name in ("LICENSE", "NOTICE"):
        source = require(repo_root / name, "the Algeon license file")
        (licenses_dir / "algeon").mkdir(exist_ok=True)
        shutil.copyfile(source, licenses_dir / "algeon" / name)
        algeon_files.append(f"algeon/{name}")
    workspace = json.loads(args.cargo_metadata.read_text(encoding="utf-8"))
    root_package = next(
        (package for package in workspace["packages"] if package["name"] == args.root_package and package.get("source") is None),
        {},
    )
    rows.append(
        (
            "algeon",
            "algeon",
            root_package.get("version", ""),
            root_package.get("license") or "",
            root_package.get("repository") or "",
            " ".join(algeon_covers),
            " ".join(algeon_files),
        )
    )

    gitmodules = require(repo_root / ".gitmodules", "the submodule list").read_text(encoding="utf-8")
    forks: list[tuple[str, ...]] = []
    for name, build_dir in layers.items():
        layer = LAYERS[name]
        cache = parse_cmake_cache(build_dir)
        source = require(repo_root / layer.submodule / "LICENSE", f"the {name} submodule license")
        destination = licenses_dir / "rapids" / name / "LICENSE"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(source, destination)
        url = re.search(rf'path = {re.escape(layer.submodule)}\n\s*url = (\S+)', gitmodules)
        if url is None:
            raise LicenseError(f".gitmodules has no url for {layer.submodule}")
        forks.append((name, url.group(1), submodule_commit(repo_root, layer.submodule), FORK_CHANGES_URL))
        covers = " ".join(layer_covers[name])
        rows.append(
            (
                "rapids",
                name,
                cache.get("CMAKE_PROJECT_VERSION", ""),
                "",
                url.group(1),
                covers,
                str(destination.relative_to(licenses_dir)),
            )
        )
        for package, version, source_dir in fetched_packages(build_dir):
            top, nested = license_files(source_dir)
            if not top:
                raise LicenseError(f"{package} fetched into {build_dir} has no top-level license file in {source_dir}")
            copied = copy_tree_files(top + nested, source_dir, licenses_dir / "native" / name / package, licenses_dir)
            rows.append(("native", package, version, "", "", covers, " ".join(copied)))

    for package in linked_rust_packages(workspace, args.root_package):
        package_dir = Path(package["manifest_path"]).parent
        label = f"{package['name']}-{package['version']}"
        if package.get("source") is None:
            if not package_dir.resolve().is_relative_to(repo_root):
                raise LicenseError(f"path dependency {label} lies outside {repo_root}")
            continue
        top, nested = license_files(package_dir)
        if package.get("license_file"):
            top.append(require(package_dir / package["license_file"], f"the license-file of {label}"))
            top = sorted(set(top))
        destination = licenses_dir / "rust" / label
        copied = copy_tree_files(top + nested, package_dir, destination, licenses_dir)
        if not top:
            expression = package.get("license") or ""
            chosen = standard_text_choice(parse_spdx(expression)) if expression else None
            if chosen is None:
                raise LicenseError(
                    f"{label} ships no license file and declares {expression or 'no license'!r}; "
                    f"no standard text is available ({sorted(STANDARD_TEXTS)})"
                )
            destination.mkdir(parents=True, exist_ok=True)
            for identifier in chosen:
                target = destination / f"{identifier}.txt"
                target.write_text(standard_text(repo_root, identifier, package), encoding="utf-8")
                copied.append(str(target.relative_to(licenses_dir)))
        rows.append(
            (
                "crate",
                package["name"],
                package["version"],
                package.get("license") or "",
                package.get("source") or "",
                "bin/algeon_server",
                " ".join(copied),
            )
        )

    lines = ["\t".join(INVENTORY_HEADER)] + ["\t".join(row) for row in rows]
    (licenses_dir / "INVENTORY.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fork_lines = ["\t".join(FORKS_HEADER)] + ["\t".join(row) for row in forks]
    (licenses_dir / "FORKS.tsv").write_text("\n".join(fork_lines) + "\n", encoding="utf-8")
    counts = {kind: sum(1 for row in rows if row[0] == kind) for kind in ("algeon", "rapids", "native", "crate")}
    files = sum(1 for path in licenses_dir.rglob("*") if path.is_file())
    print(f"==> Staged {files} license files: {counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--cargo-metadata", type=Path, required=True)
    parser.add_argument("--root-package", required=True)
    parser.add_argument("--layer", action="append", default=[], help="<name>=<native build dir>, one per staged layer")
    args = parser.parse_args()
    try:
        stage(args)
    except (LicenseError, OSError, json.JSONDecodeError, KeyError) as error:
        print(f"ERROR: license staging failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
