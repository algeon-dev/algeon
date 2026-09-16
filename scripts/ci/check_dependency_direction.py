#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
PLANNING_PACKAGE = "nexus-query-planning"
RUNTIME_PACKAGE = "nexus-query-runtime"
EXECUTION_PACKAGE = "nexus-query-engine"
COMPONENT_PACKAGES = (
    "cudf-nexus",
    "cudf-nexus-sys",
    "cugraph-nexus",
    "cugraph-nexus-sys",
    "cuvs-nexus",
    "cuvs-nexus-sys",
)
DATAFUSION_PACKAGE_PREFIX = "datafusion"

# Packages whose presence anywhere in the planning closure means planning
# cannot be built, tested, or packaged without CUDA/RAPIDS or DataFusion. A
# declared-dependency check is not enough: any of these can arrive through a
# helper crate several edges away, which is exactly how the coupling grew
# before. `--all-features` resolution makes the union of every feature the
# subject, so an optional dependency cannot hide behind a feature flag either.
PLANNING_FORBIDDEN_PACKAGES = frozenset(COMPONENT_PACKAGES) | {RUNTIME_PACKAGE, EXECUTION_PACKAGE}
NATIVE_SYS_SUFFIX = "-sys"
NATIVE_PACKAGE_MARKERS = ("cudf", "cugraph", "cuvs", "cuda", "rmm", "nvml", "rapids")

class ContractError(ValueError):
    pass


def load_metadata(root: Path) -> dict[str, object]:
    cargo = os.environ.get("CARGO", "cargo")
    command = [cargo, "metadata", "--locked", "--all-features", "--format-version", "1"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or "no stderr output"
        raise ContractError(f"cargo metadata failed with status {result.returncode}: {detail}")
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ContractError(f"cargo metadata returned invalid JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise ContractError("cargo metadata root must be an object")
    return metadata


def workspace_packages(metadata: dict[str, object]) -> tuple[dict[str, dict[str, object]], set[str]]:
    packages_value = metadata.get("packages")
    members_value = metadata.get("workspace_members")
    if not isinstance(packages_value, list) or not isinstance(members_value, list):
        raise ContractError("cargo metadata must contain packages and workspace_members arrays")

    packages: dict[str, dict[str, object]] = {}
    for package in packages_value:
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise ContractError("cargo metadata contains a package without a string id")
        packages[package["id"]] = package

    member_ids = {member for member in members_value if isinstance(member, str)}
    if len(member_ids) != len(members_value):
        raise ContractError("cargo metadata workspace_members must contain only strings")
    missing = sorted(member_ids - packages.keys())
    if missing:
        raise ContractError(f"workspace member ids are missing from packages: {missing}")

    by_name: dict[str, dict[str, object]] = {}
    for package_id in member_ids:
        package = packages[package_id]
        name = package.get("name")
        if not isinstance(name, str):
            raise ContractError(f"workspace package {package_id} has no string name")
        if name in by_name:
            raise ContractError(f"workspace contains duplicate package name {name!r}")
        by_name[name] = package
    return by_name, member_ids


def resolved_workspace_edges(
    metadata: dict[str, object], by_name: dict[str, dict[str, object]], member_ids: set[str]
) -> set[tuple[str, str]]:
    resolve = metadata.get("resolve")
    if not isinstance(resolve, dict) or not isinstance(resolve.get("nodes"), list):
        raise ContractError("cargo metadata must contain resolve.nodes")

    id_to_name = {package["id"]: name for name, package in by_name.items()}
    edges: set[tuple[str, str]] = set()
    for node in resolve["nodes"]:
        if not isinstance(node, dict):
            raise ContractError("cargo metadata resolve.nodes contains a non-object")
        source_id = node.get("id")
        if source_id not in member_ids:
            continue
        source_name = id_to_name[source_id]
        dependencies = node.get("deps", [])
        if not isinstance(dependencies, list):
            raise ContractError(f"resolved dependencies for {source_name} must be an array")
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                raise ContractError(f"resolved dependency of {source_name} is not an object")
            target_id = dependency.get("pkg")
            target_name = id_to_name.get(target_id)
            if target_name is not None:
                edges.add((source_name, target_name))
    return edges


def resolved_closure(
    metadata: dict[str, object], by_name: dict[str, dict[str, object]], root_package: str
) -> set[str]:
    """Every package reachable from `root_package` under the resolved graph."""
    resolve = metadata.get("resolve")
    if not isinstance(resolve, dict) or not isinstance(resolve.get("nodes"), list):
        raise ContractError("cargo metadata must contain resolve.nodes")

    packages_value = metadata.get("packages")
    if not isinstance(packages_value, list):
        raise ContractError("cargo metadata must contain a packages array")
    id_to_name: dict[str, str] = {}
    for package in packages_value:
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise ContractError("cargo metadata contains a package without a string id")
        name = package.get("name")
        if not isinstance(name, str):
            raise ContractError(f"cargo metadata package {package['id']} has no string name")
        id_to_name[package["id"]] = name

    edges: dict[str, list[str]] = {}
    for node in resolve["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise ContractError("cargo metadata resolve.nodes contains a node without a string id")
        dependencies = node.get("deps", [])
        if not isinstance(dependencies, list):
            raise ContractError(f"resolved dependencies for {node['id']} must be an array")
        targets = []
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(dependency.get("pkg"), str):
                raise ContractError(f"resolved dependency of {node['id']} is not an object with a pkg id")
            targets.append(dependency["pkg"])
        edges[node["id"]] = targets

    root = by_name.get(root_package)
    if root is None or not isinstance(root.get("id"), str):
        raise ContractError(f"workspace package {root_package} is missing from cargo metadata")

    seen: set[str] = set()
    stack = [root["id"]]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, []))
    return {id_to_name[package_id] for package_id in seen if package_id in id_to_name}


def looks_native(package_name: str) -> bool:
    lowered = package_name.lower()
    return lowered.endswith(NATIVE_SYS_SUFFIX) and any(marker in lowered for marker in NATIVE_PACKAGE_MARKERS)


def check_planning_closure(metadata: dict[str, object]) -> list[str]:
    """Planning must build without DataFusion, CUDA/RAPIDS, runtime, or engine.

    This is the contract that keeps the crate publishable and testable on a
    host with no GPU toolchain. It is checked against the resolved closure, not
    the manifest, so a helper crate cannot reintroduce the coupling quietly.
    """
    by_name, _member_ids = workspace_packages(metadata)
    closure = resolved_closure(metadata, by_name, PLANNING_PACKAGE) - {PLANNING_PACKAGE}

    violations = []
    for package_name in sorted(closure):
        if package_name in PLANNING_FORBIDDEN_PACKAGES:
            reason = "native component" if package_name in COMPONENT_PACKAGES else "query layer"
            violations.append(
                f"planning closure -> {reason}: {PLANNING_PACKAGE} resolves {package_name} under --all-features"
            )
        elif package_name == DATAFUSION_PACKAGE_PREFIX or package_name.startswith(f"{DATAFUSION_PACKAGE_PREFIX}-"):
            violations.append(
                f"planning closure -> DataFusion: {PLANNING_PACKAGE} resolves {package_name} under --all-features"
            )
        elif looks_native(package_name):
            violations.append(
                f"planning closure -> native sys crate: {PLANNING_PACKAGE} resolves {package_name} "
                "under --all-features"
            )
    return violations


def check_package_edges(metadata: dict[str, object]) -> list[str]:
    by_name, member_ids = workspace_packages(metadata)
    edges = resolved_workspace_edges(metadata, by_name, member_ids)
    present = set(by_name)
    required = {PLANNING_PACKAGE, RUNTIME_PACKAGE, EXECUTION_PACKAGE, *COMPONENT_PACKAGES}
    missing = sorted(required - present)
    if missing:
        raise ContractError(f"required workspace packages are missing: {', '.join(missing)}")

    rules = [
        (PLANNING_PACKAGE, RUNTIME_PACKAGE, "planning -> runtime"),
        (PLANNING_PACKAGE, EXECUTION_PACKAGE, "planning -> engine-exec"),
        (RUNTIME_PACKAGE, EXECUTION_PACKAGE, "runtime -> engine-exec"),
    ]
    for component in COMPONENT_PACKAGES:
        rules.append((component, EXECUTION_PACKAGE, "component -> engine"))

    violations = [
        f"{label}: resolved Cargo edge {source} -> {target}"
        for source, target, label in rules
        if (source, target) in edges
    ]
    for layer_package in (PLANNING_PACKAGE, RUNTIME_PACKAGE):
        package = by_name.get(layer_package)
        if package is None:
            continue
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ContractError(f"workspace package {layer_package} dependencies must be an array")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                raise ContractError(f"workspace package {layer_package} has an invalid dependency")
            dependency_name = dependency["name"]
            if dependency_name == DATAFUSION_PACKAGE_PREFIX or dependency_name.startswith(
                f"{DATAFUSION_PACKAGE_PREFIX}-"
            ):
                violations.append(
                    f"{layer_package.removeprefix('nexus-query-')} -> DataFusion: "
                    f"declared Cargo dependency {dependency_name}"
                )
    return violations


def mask_rust_non_code(text: str) -> str:
    """Blank comments and literals while preserving offsets and line numbers."""
    chars = list(text)
    length = len(text)

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "

    index = 0
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index)
            if end < 0:
                end = length
            blank(index, end)
            index = end
            continue
        if text.startswith("/*", index):
            start = index
            index += 2
            depth = 1
            while index < length and depth:
                if text.startswith("/*", index):
                    depth += 1
                    index += 2
                elif text.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            blank(start, index)
            continue
        if text[index] == "r":
            delimiter = index + 1
            while delimiter < length and text[delimiter] == "#":
                delimiter += 1
            if delimiter < length and text[delimiter] == '"':
                hashes = text[index + 1 : delimiter]
                end_marker = '"' + hashes
                end = text.find(end_marker, delimiter + 1)
                end = length if end < 0 else end + len(end_marker)
                blank(index, end)
                index = end
                continue
        if text[index] == '"':
            start = index
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            blank(start, index)
            continue
        index += 1
    return "".join(chars)


DIRECT_LAYER_PATH = re.compile(
    r"\b(?:crate|self|super(?:\s*::\s*super)*|nexus_query_engine|"
    r"nexus_query_planning|nexus_query_runtime)"
    r"\s*::(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)*(?P<target>runtime|exec)\s*(?:::|[,;)}])"
)
USE_STATEMENT = re.compile(r"\buse\s+.*?;", re.DOTALL)
USE_ROOT = re.compile(
    r"\b(?:crate|self|super|nexus_query_engine|nexus_query_planning|nexus_query_runtime)\s*::"
)
USE_LAYER_SEGMENT = re.compile(r"\b(?P<target>runtime|exec)\s*(?:::|[,;)}])")
USE_RUNTIME_MODULE_SEGMENT = re.compile(r"\b(?P<target>attempt|metrics|observability)\s*(?:::|[,;)}])")
PACKAGE_PATH = re.compile(r"\b(?P<package>nexus_query_engine|nexus_query_runtime)\s*::")
PACKAGE_ROOT_IMPORT = re.compile(
    r"\buse\s+(?P<package>nexus_query_engine|nexus_query_runtime)"
    r"\s*(?:as\s+[A-Za-z_][A-Za-z0-9_]*)?\s*;"
)
ENGINE_EXEC_PATH = re.compile(r"\bnexus_query_engine\s*::\s*exec\s*::")
EXTERN_PACKAGE = re.compile(
    r"\bextern\s+crate\s+(?P<package>nexus_query_engine|nexus_query_runtime)\b"
)
RUNTIME_MODULE_PATH = re.compile(
    r"\b(?:crate|self|super(?:\s*::\s*super)*|nexus_query_engine)"
    r"\s*::(?P<target>attempt|metrics|observability)\s*(?:::|[,;)}])"
)
DATAFUSION_PATH = re.compile(r"\bdatafusion(?:_[A-Za-z_][A-Za-z0-9_]*)?\s*::")
DATAFUSION_ROOT_IMPORT = re.compile(
    r"\b(?:use|extern\s+crate)\s+datafusion(?:_[A-Za-z_][A-Za-z0-9_]*)?\b"
)
OBSERVABILITY_MODULE = "observability"
RUNTIME_MODULE = "runtime"
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ROOTED_PATH = re.compile(r"(?<![A-Za-z0-9_:])(?:crate|self|super)(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)+")
USE_DECLARATION = re.compile(r"\buse\s+(?P<tree>[^;]*);")
USE_TREE_TOKEN = re.compile(r"::|[{},*]|[A-Za-z_][A-Za-z0-9_]*")
INLINE_MODULE = re.compile(r"\bmod\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\{")
CHAR_BRACE_LITERAL = re.compile(r"'\\?[{}]'")
PATH_ATTRIBUTE_MODULE = re.compile(
    r'#\s*\[\s*path\s*=\s*"(?P<path>[^"]+)"\s*\]\s*(?:#\s*\[[^\]]*\]\s*)*'
    r"(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;"
)


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def source_import_violations(path: Path, layer: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    masked = mask_rust_non_code(text)
    lines = text.splitlines()
    findings: set[tuple[str, int, str]] = set()

    def add(rule: str, offset: int, token: str) -> None:
        findings.add((rule, line_number(masked, offset), token))

    for match in DIRECT_LAYER_PATH.finditer(masked):
        target = match.group("target")
        if target == "runtime" and layer == "planning":
            add("planning -> runtime", match.start(), match.group(0).strip())
        elif target == "exec":
            add(f"{layer} -> engine-exec", match.start(), match.group(0).strip())

    for match in RUNTIME_MODULE_PATH.finditer(masked):
        if layer == "planning":
            add("planning -> runtime", match.start(), match.group(0).strip())

    for statement in USE_STATEMENT.finditer(masked):
        if USE_ROOT.search(statement.group(0)) is None:
            continue
        module = USE_RUNTIME_MODULE_SEGMENT.search(statement.group(0))
        if module is not None and layer == "planning":
            add("planning -> runtime", statement.start() + module.start(), module.group(0).strip())
        segment = USE_LAYER_SEGMENT.search(statement.group(0))
        if segment is None:
            continue
        target = segment.group("target")
        offset = statement.start() + segment.start()
        if target == "runtime" and layer == "planning":
            add("planning -> runtime", offset, segment.group(0).strip())
        elif target == "exec":
            add(f"{layer} -> engine-exec", offset, segment.group(0).strip())

    for match in PACKAGE_PATH.finditer(masked):
        package = match.group("package")
        if package == "nexus_query_runtime" and layer == "planning":
            add("planning -> runtime", match.start(), package)
        elif ENGINE_EXEC_PATH.match(masked, match.start()) is not None:
            add(f"{layer} -> engine-exec", match.start(), package)

    for match in PACKAGE_ROOT_IMPORT.finditer(masked):
        package = match.group("package")
        if package == "nexus_query_runtime" and layer == "planning":
            add("planning -> runtime", match.start(), package)
        elif package == "nexus_query_engine":
            add(f"{layer} -> engine-exec", match.start(), package)

    for match in EXTERN_PACKAGE.finditer(masked):
        package = match.group("package")
        if package == "nexus_query_runtime" and layer == "planning":
            add("planning -> runtime", match.start(), package)
        elif package == "nexus_query_engine":
            add(f"{layer} -> engine-exec", match.start(), package)

    for pattern in (DATAFUSION_PATH, DATAFUSION_ROOT_IMPORT):
        for match in pattern.finditer(masked):
            add(f"{layer} -> DataFusion", match.start(), match.group(0).strip())

    result = []
    for rule, number, token in sorted(findings):
        snippet = lines[number - 1].strip() if 0 < number <= len(lines) else ""
        result.append(f"{rule}: {path}:{number}: {snippet or token}")
    return result


def rust_module_path(source_root: Path, path: Path) -> list[str]:
    parts = list(path.relative_to(source_root).with_suffix("").parts)
    if parts[-1] == "mod":
        parts.pop()
    return parts


def observability_module_paths(source_root: Path) -> dict[Path, list[str]]:
    files = source_files([source_root / f"{OBSERVABILITY_MODULE}.rs", source_root / OBSERVABILITY_MODULE])
    modules = {path.resolve(): rust_module_path(source_root, path) for path in files}
    for path in files:
        text = path.read_text(encoding="utf-8")
        masked = mask_rust_non_code(text)
        for match in PATH_ATTRIBUTE_MODULE.finditer(text):
            target = (path.parent / match.group("path")).resolve()
            if masked[match.start()] == "#" and target in modules:
                modules[target] = [*modules[path.resolve()], match.group("name")]
    return modules


def inline_module_spans(masked: str) -> list[tuple[int, int, str]]:
    code = CHAR_BRACE_LITERAL.sub(lambda match: " " * len(match.group(0)), masked)
    spans = []
    for match in INLINE_MODULE.finditer(code):
        depth = 0
        for index in range(match.end() - 1, len(code)):
            if code[index] == "{":
                depth += 1
            elif code[index] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((match.end(), index, match.group("name")))
                    break
    return spans


def use_tree_paths(tokens: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
    paths: list[list[tuple[str, int]]] = []

    def parse(index: int, prefix: list[tuple[str, int]]) -> int:
        segments = list(prefix)
        while index < len(tokens):
            token = tokens[index][0]
            if token == "{":
                index += 1
                while index < len(tokens) and tokens[index][0] != "}":
                    index = index + 1 if tokens[index][0] == "," else parse(index, segments)
                return index + 1
            if token in (",", "}"):
                break
            if token == "as":
                index += 2
                continue
            if token not in ("::", "*"):
                segments.append(tokens[index])
            index += 1
        paths.append(segments)
        return index

    index = 0
    while index < len(tokens):
        index = index + 1 if tokens[index][0] in (",", "}") else parse(index, [])
    return paths


def crate_root_segment(module: list[str], segments: list[tuple[str, int]]) -> tuple[str, int] | None:
    """Returns the segment naming a crate-root item when a rooted path reaches the crate root."""
    names = [name for name, _ in segments]
    if not names or names[0] not in ("crate", "self", "super"):
        return None
    index = 1
    if names[0] != "crate":
        resolved = list(module)
        index = 1 if names[0] == "self" else 0
        while index < len(names) and names[index] == "super":
            if not resolved:
                return None
            resolved.pop()
            index += 1
        if resolved:
            return None
    return segments[index] if index < len(segments) else None


def observability_runtime_violations(source_root: Path) -> list[str]:
    """Source-level `observability -> runtime` check for the runtime crate.

    `crate`, `self`, and `super` paths in expressions, types, and nested use
    trees resolve against each file's module path, including `#[path]` module
    declarations and inline modules. Macro expansion and re-exports are not
    resolved.
    """
    violations = []
    for path, module in sorted(observability_module_paths(source_root).items()):
        text = path.read_text(encoding="utf-8")
        masked = mask_rust_non_code(text)
        lines = text.splitlines()
        spans = inline_module_spans(masked)
        candidates = [
            [(name.group(0), match.start() + name.start()) for name in IDENTIFIER.finditer(match.group(0))]
            for match in ROOTED_PATH.finditer(masked)
        ]
        for statement in USE_DECLARATION.finditer(masked):
            tokens = [
                (token.group(0), statement.start("tree") + token.start())
                for token in USE_TREE_TOKEN.finditer(statement.group("tree"))
            ]
            candidates.extend(use_tree_paths(tokens))
        numbers = set()
        for segments in candidates:
            if not segments:
                continue
            scope = [*module, *(name for start, end, name in spans if start <= segments[0][1] < end)]
            target = crate_root_segment(scope, segments)
            if target is not None and target[0] == RUNTIME_MODULE:
                numbers.add(line_number(masked, target[1]))
        for number in sorted(numbers):
            snippet = lines[number - 1].strip() if 0 < number <= len(lines) else RUNTIME_MODULE
            violations.append(f"observability -> runtime: {path}:{number}: {snippet}")
    return violations


def source_files(roots: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".rs":
            files.add(root)
        elif root.is_dir():
            files.update(path for path in root.rglob("*.rs") if path.is_file())
    return sorted(files)


def package_directory(root: Path, by_name: dict[str, dict[str, object]], package_name: str) -> Path:
    package = by_name.get(package_name)
    if package is None:
        raise ContractError(f"required workspace package is missing: {package_name}")
    manifest_path = package.get("manifest_path")
    if not isinstance(manifest_path, str):
        raise ContractError(f"workspace package {package_name} has no manifest_path")
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    return manifest.resolve().parent


def package_roots(
    root: Path,
    by_name: dict[str, dict[str, object]],
    package_name: str,
) -> list[Path]:
    package_root = package_directory(root, by_name, package_name)
    package = by_name[package_name]
    roots = [package_root / "src", package_root / "tests"]
    targets = package.get("targets", [])
    if not isinstance(targets, list):
        raise ContractError(f"workspace package {package_name} targets must be an array")
    for target in targets:
        if not isinstance(target, dict):
            raise ContractError(f"workspace package {package_name} has a non-object target")
        source_path = target.get("src_path")
        if not isinstance(source_path, str):
            continue
        path = Path(source_path)
        roots.append(path if path.is_absolute() else root / path)
    return roots


def check_sources(root: Path, metadata: dict[str, object]) -> list[str]:
    by_name, _ = workspace_packages(metadata)
    violations: list[str] = []

    planning_roots = package_roots(root, by_name, PLANNING_PACKAGE)
    runtime_roots = package_roots(root, by_name, RUNTIME_PACKAGE)

    for path in source_files(planning_roots):
        violations.extend(source_import_violations(path, "planning"))
    for path in source_files(runtime_roots):
        violations.extend(source_import_violations(path, "runtime"))
    violations.extend(observability_runtime_violations(package_directory(root, by_name, RUNTIME_PACKAGE) / "src"))

    for component in COMPONENT_PACKAGES:
        roots = package_roots(root, by_name, component)
        for path in source_files(roots):
            text = mask_rust_non_code(path.read_text(encoding="utf-8"))
            for match in PACKAGE_PATH.finditer(text):
                if match.group("package") == "nexus_query_engine":
                    number = line_number(text, match.start())
                    original_lines = path.read_text(encoding="utf-8").splitlines()
                    snippet = (
                        original_lines[number - 1].strip()
                        if 0 < number <= len(original_lines)
                        else match.group(0)
                    )
                    violations.append(f"component -> engine: {path}:{number}: {snippet}")
            for match in PACKAGE_ROOT_IMPORT.finditer(text):
                if match.group("package") == "nexus_query_engine":
                    number = line_number(text, match.start())
                    original_lines = path.read_text(encoding="utf-8").splitlines()
                    snippet = (
                        original_lines[number - 1].strip()
                        if 0 < number <= len(original_lines)
                        else match.group(0)
                    )
                    violations.append(f"component -> engine: {path}:{number}: {snippet}")
            for match in EXTERN_PACKAGE.finditer(text):
                if match.group("package") == "nexus_query_engine":
                    number = line_number(text, match.start())
                    violations.append(f"component -> engine: {path}:{number}: extern crate {match.group('package')}")
    return violations


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Cargo and source dependency direction contracts.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="workspace root (for policy tests)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: workspace root does not exist: {root}", file=sys.stderr)
        return 2
    try:
        metadata = load_metadata(root)
        graph_violations = check_package_edges(metadata)
        closure_violations = check_planning_closure(metadata)
        source_violations = check_sources(root, metadata)
        violations = graph_violations + closure_violations + source_violations
        if violations:
            print("ERROR: dependency direction contracts violated:", file=sys.stderr)
            for violation in sorted(set(violations)):
                print(f"  {violation}", file=sys.stderr)
            return 1
        print("dependency direction contracts OK")
        return 0
    except (OSError, ContractError) as error:
        print(f"ERROR: dependency direction check: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
