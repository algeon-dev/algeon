#!/usr/bin/env python3
"""Print the workspace packages whose test builds link no native RAPIDS library.

A workspace member with a `links` key owns a native library. A package is
native-free when neither it nor anything its tests compile reaches such a
member: its own normal, build and dev dependencies count, and below that only
normal and build dependencies, because Cargo never builds a dependency's
dev-dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def native_free_packages(metadata: dict) -> list[str]:
    packages = {package["id"]: package for package in metadata["packages"]}
    members = metadata["workspace_members"]
    native = {member for member in members if packages[member].get("links")}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}

    def reaches_native(root: str) -> bool:
        seen: set[str] = set()
        pending = [root]
        while pending:
            current = pending.pop()
            for dependency in nodes[current]["deps"]:
                kinds = {kind["kind"] for kind in dependency["dep_kinds"]}
                if current != root and kinds == {"dev"}:
                    continue
                if dependency["pkg"] in native:
                    return True
                if dependency["pkg"] not in seen:
                    seen.add(dependency["pkg"])
                    pending.append(dependency["pkg"])
        return False

    return sorted(packages[member]["name"] for member in members
                  if member not in native and not reaches_native(member))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: native_free_packages.py <cargo-metadata.json>", file=sys.stderr)
        return 2
    names = native_free_packages(json.loads(Path(sys.argv[1]).read_text()))
    if not names:
        print("error: no workspace package builds its tests without native libraries", file=sys.stderr)
        return 1
    print("\n".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
