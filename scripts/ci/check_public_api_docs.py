#!/usr/bin/env python3
"""Check documentation on the application API and reachable workspace types."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path


class PolicyError(Exception):
    pass


def workspace_libraries(metadata: dict) -> dict[str, str]:
    members = set(metadata["workspace_members"])
    return {
        target["name"].replace("-", "_"): package["name"]
        for package in metadata["packages"]
        if package["id"] in members
        for target in package["targets"]
        if "lib" in target["kind"]
    }


def option_value(args: list[str], option: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == option:
            return args[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return None


def build_arguments(args: list[str]) -> list[str]:
    result = []
    index = 0
    values = {"--features", "-F", "--target", "--target-dir"}
    while index < len(args):
        arg = args[index]
        if arg in {"--all-features", "--no-default-features"}:
            result.append(arg)
        elif arg in values:
            result.extend([arg, args[index + 1]])
            index += 1
        elif any(arg.startswith(flag + "=") for flag in values) or arg.startswith("-F"):
            result.append(arg)
        index += 1
    return result


def build(metadata: dict, destination: Path, args: list[str]) -> None:
    libraries = workspace_libraries(metadata)
    command = ["cargo", "+nightly", "doc", "--workspace", "--lib", "--no-deps", "--locked"]
    command.extend(build_arguments(args))
    environment = os.environ.copy()
    # Private items supply signatures for reachable types. Their link rendering
    # is outside this check; the ordinary public rustdoc phase keeps strict links.
    environment["RUSTDOCFLAGS"] = " ".join([
        environment.get("RUSTDOCFLAGS", ""), "-Z unstable-options --output-format json",
        "--document-private-items --document-hidden-items",
        "-A rustdoc::broken_intra_doc_links -A rustdoc::private_intra_doc_links -A rustdoc::redundant_explicit_links",
    ]).strip()
    result = subprocess.run(command, env=environment)
    if result.returncode:
        raise PolicyError("complete workspace rustdoc JSON build failed")
    target_dir = Path(option_value(args, "--target-dir") or metadata["target_directory"])
    target = option_value(args, "--target")
    if target:
        target_dir /= target
    destination.mkdir(parents=True, exist_ok=True)
    for crate, package in libraries.items():
        source = target_dir / "doc" / f"{crate}.json"
        if not source.is_file():
            raise PolicyError(f"workspace rustdoc produced no JSON for {package}: {source}")
        shutil.copyfile(source, destination / source.name)


def path_references(value):
    if isinstance(value, dict):
        if "id" in value and "path" in value:
            yield value["id"]
        for child in value.values():
            yield from path_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from path_references(child)


class DocumentationPolicy:
    def __init__(self, libraries: dict[str, str], json_dir: Path,
                 generated_files: set[Path] | None = None, workspace_root: Path | None = None):
        self.libraries = libraries
        self.json_dir = json_dir
        self.crates = {}
        self.identities = {}
        self.hidden_ancestors = {}
        self.named_items = {}
        self.queue = deque()
        self.visited = set()
        self.missing = {}
        self.generated_files = generated_files or set()
        self.workspace_root = workspace_root or Path.cwd()

    def load(self, crate: str) -> dict:
        if crate not in self.crates:
            path = self.json_dir / f"{crate}.json"
            if not path.is_file():
                raise PolicyError(
                    f"reachable workspace owner {self.libraries[crate]} has no current-run rustdoc JSON ({path})"
                )
            document = json.loads(path.read_text())
            if not document["includes_private"]:
                raise PolicyError(f"{path} omits private items needed for reachable method signatures")
            if document["index"][str(document["root"])]["name"] != crate:
                raise PolicyError(f"{path} has the wrong crate root")
            self.crates[crate] = document
            self.identities[crate] = {
                (tuple(summary["path"]), summary["kind"]): item_id
                for item_id, summary in document["paths"].items()
                if summary["crate_id"] == 0
            }
            hidden_ancestors = set()
            named_items = set()

            def modules(item_id, inherited_hidden=False, visible=True):
                item_id = str(item_id)
                item = document["index"][item_id]
                hidden = inherited_hidden or self.hidden(item)
                if inherited_hidden:
                    hidden_ancestors.add(item_id)
                visible = visible and not hidden and item["visibility"] == "public"
                if visible:
                    named_items.add(item_id)
                    used = item["inner"].get("use")
                    if used and used["id"] is not None:
                        named_items.add(str(used["id"]))
                module = item["inner"].get("module")
                if module:
                    for child_id in module["items"]:
                        modules(child_id, hidden, visible)

            modules(document["root"])
            self.hidden_ancestors[crate] = hidden_ancestors
            self.named_items[crate] = named_items
            for item_id in list(named_items):
                self.mark_named(crate, item_id)
        return self.crates[crate]

    def resolve(self, crate: str, item_id) -> tuple[str, str] | None:
        document = self.load(crate)
        item_id = str(item_id)
        summary = document["paths"].get(item_id)
        if summary and summary["path"][0] != crate:
            owner = summary["path"][0]
            if owner not in self.libraries:
                return None
            self.load(owner)
            identity = (tuple(summary["path"]), summary["kind"])
            owner_id = self.identities[owner].get(identity)
            if owner_id is None:
                raise PolicyError(f"cannot resolve reachable owner item {'::'.join(summary['path'])}")
            return owner, owner_id
        if item_id in document["index"]:
            return crate, item_id
        if summary and summary["path"][0] not in self.libraries:
            return None
        raise PolicyError(f"{crate} rustdoc JSON omits reachable item {item_id}")

    def enqueue(self, crate: str, item_id, path: str, require_docs: bool = True) -> None:
        resolved = self.resolve(crate, item_id)
        if resolved:
            self.queue.append((*resolved, path, require_docs))

    def hidden(self, item: dict) -> bool:
        return any(isinstance(attr, dict) and re.search(r"doc\s*\([^)]*\bhidden\b", attr.get("other", ""))
                   for attr in item["attrs"])

    def excluded(self, crate: str, item_id: str) -> bool:
        item = self.load(crate)["index"][item_id]
        return self.hidden(item) or (item_id in self.hidden_ancestors[crate]
                                    and item_id not in self.named_items[crate])

    def mark_named(self, crate: str, item_id: str) -> None:
        document = self.load(crate)
        pending = [item_id]
        visited = set()
        while pending:
            current = str(pending.pop())
            item = document["index"].get(current)
            if item is None or current in visited or self.hidden(item):
                continue
            visited.add(current)
            self.named_items[crate].add(current)
            module = item["inner"].get("module")
            if module:
                pending.extend(child_id for child_id in module["items"]
                               if document["index"][str(child_id)]["visibility"] == "public")
            used = item["inner"].get("use")
            if used and used["id"] is not None:
                pending.append(used["id"])

    def generated(self, item: dict) -> bool:
        span = item["span"]
        if not span:
            return False
        path = Path(span["filename"])
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve() in self.generated_files

    def references(self, crate: str, value, path: str) -> None:
        source = self.load(crate)
        for item_id in path_references(value):
            # Anonymous-const macro types, including bitflags' InternalBitFlags,
            # can have a reference ID without a rustdoc item or owner path.
            # Membership IDs and identified workspace owners remain mandatory.
            if str(item_id) not in source["index"] and str(item_id) not in source["paths"]:
                continue
            resolved = self.resolve(crate, item_id)
            if resolved:
                owner, owner_id = resolved
                document = self.load(owner)
                summary = document["paths"].get(owner_id)
                name = "::".join(summary["path"]) if summary else document["index"][owner_id]["name"]
                self.queue.append((owner, owner_id, f"{path} -> {name}", True))

    def children(self, crate: str, ids, path: str, require_docs: bool = True,
                 public_only: bool = False) -> None:
        document = self.load(crate)
        for item_id in ids:
            if item_id is None:
                continue
            item = document["index"].get(str(item_id))
            if item is None:
                raise PolicyError(f"{crate} rustdoc JSON omits child item {item_id} of {path}")
            if public_only and item["visibility"] != "public":
                continue
            name = item["name"] or item["inner"].get("use", {}).get("name", "impl")
            self.enqueue(crate, item_id, f"{path}::{name}", require_docs)

    def implementations(self, crate: str, ids, path: str) -> None:
        document = self.load(crate)
        for item_id in ids:
            item = document["index"][str(item_id)]
            implementation = item["inner"]["impl"]
            if (implementation["is_synthetic"] or implementation["blanket_impl"] is not None
                    or "automatically_derived" in item["attrs"] or self.hidden(item)):
                continue
            trait = implementation["trait"]
            if trait:
                resolved = self.resolve(crate, trait["id"])
                if resolved:
                    owner, trait_id = resolved
                    definition = self.load(owner)["index"][trait_id]
                    if definition["visibility"] != "public" or self.excluded(owner, trait_id):
                        continue
                self.references(crate, trait, path)
            self.references(crate, implementation["generics"], path)
            # Trait members inherit their documentation, but concrete associated
            # types and method signatures still extend the application surface.
            inherent = implementation["trait"] is None
            self.children(crate, implementation["items"], path, inherent, inherent)

    def check(self, root: str = "algeon_datafusion") -> list[str]:
        document = self.load(root)
        self.enqueue(root, document["root"], root)
        while self.queue:
            crate, item_id, path, require_docs = self.queue.popleft()
            item = self.load(crate)["index"][item_id]
            if self.excluded(crate, item_id):
                continue
            kind, payload = next(iter(item["inner"].items()))
            if kind == "trait" and item["visibility"] != "public":
                continue
            if require_docs and kind not in {"use", "impl", "extern_crate"} and not self.generated(item):
                if not (item["docs"] or "").strip():
                    self.missing[(crate, item_id)] = (path, item)
            identity = (crate, item_id)
            if identity in self.visited:
                continue
            self.visited.add(identity)
            if kind == "module":
                self.children(crate, payload["items"], path, public_only=True)
            elif kind == "use":
                if payload["id"] is not None:
                    resolved = self.resolve(crate, payload["id"])
                    if resolved:
                        owner, target_id = resolved
                        self.mark_named(owner, target_id)
                    self.enqueue(crate, payload["id"], path)
            elif kind in {"struct", "union", "enum", "trait"}:
                self.references(crate, payload.get("generics"), path)
                if kind == "struct":
                    form = payload["kind"]
                    fields = form.get("plain", {}).get("fields", []) if isinstance(form, dict) else []
                    tuple_fields = isinstance(form, dict) and "tuple" in form
                    if tuple_fields:
                        fields = form["tuple"]
                    self.children(crate, fields, path, not tuple_fields, public_only=True)
                elif kind == "union":
                    self.children(crate, payload["fields"], path, public_only=True)
                elif kind == "enum":
                    self.children(crate, payload["variants"], path)
                else:
                    self.children(crate, payload["items"], path)
                    self.references(crate, payload["bounds"], path)
                self.implementations(crate, payload.get("impls", []), path)
            elif kind == "variant":
                form = payload["kind"]
                if isinstance(form, dict):
                    fields = form.get("tuple", form.get("struct", {}).get("fields", []))
                    self.children(crate, fields, path, "tuple" not in form)
            else:
                self.references(crate, payload, path)
        failures = []
        for (crate, _), (path, item) in sorted(self.missing.items(), key=lambda row: row[1][0]):
            span = item["span"]
            location = f"{span['filename']}:{span['begin'][0]}" if span else crate
            failures.append(f"{location}: missing application API documentation for {path} (owner {crate})")
        return failures

    def findings(self) -> list[dict]:
        findings = []
        for (crate, item_id), (path, item) in self.missing.items():
            summary = self.load(crate)["paths"].get(item_id)
            span = item["span"]
            findings.append({
                "owner": crate,
                "item": "::".join(summary["path"]) if summary else item["name"],
                "application_path": path,
                "kind": next(iter(item["inner"])),
                "file": span["filename"] if span else None,
                "line": span["begin"][0] if span else None,
            })
        return sorted(findings, key=lambda finding: (finding["owner"], finding["file"] or "", finding["line"] or 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "check"])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="write missing-documentation findings as JSON")
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    cargo_args = arguments[separator + 1:]
    try:
        metadata = json.loads(args.metadata.read_text())
        if args.mode == "build":
            build(metadata, args.json_dir, cargo_args)
            return 0
        root = Path(metadata["workspace_root"])
        generated_files = {
            (Path(package["manifest_path"]).parent / "src" / "bindings.rs").resolve()
            for package in metadata["packages"]
            if package["name"] in {"algeon-cudf-sys", "algeon-cugraph-sys", "algeon-cuvs-sys", "algeon-cuml-sys"}
        }
        policy = DocumentationPolicy(workspace_libraries(metadata), args.json_dir, generated_files, root)
        failures = policy.check()
        if args.report:
            args.report.write_text(json.dumps(policy.findings(), indent=2) + "\n")
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        if failures:
            return 1
        print(f"PASS: application API documentation checked across {len(policy.visited)} reachable items")
        return 0
    except (PolicyError, OSError, ValueError, KeyError, IndexError) as error:
        print(f"ERROR: application API documentation policy: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
