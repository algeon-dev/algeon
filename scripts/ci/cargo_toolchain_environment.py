#!/usr/bin/env python3
"""Print Cargo's effective toolchain environment from its resolved config."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tomllib


TOOLCHAIN_KEYS = (
    "CC", "CXX", "CUDACXX", "CUDAHOSTCXX", "CUDA_HOME", "CUDA_PATH",
    "CUDAToolkit_ROOT", "CMAKE_CUDA_ARCHITECTURES",
)


def effective_environment(config_text: str, environ: dict[str, str]) -> dict[str, str | None]:
    configured = tomllib.loads(config_text).get("env", {})
    origins = {}
    for line in config_text.splitlines():
        match = re.match(r"env\.(\w+)(?:\.value)?\s*=.* # (.+)$", line)
        if match:
            origins[match[1]] = match[2]
    values = {}
    for key in TOOLCHAIN_KEYS:
        setting = configured.get(key)
        if isinstance(setting, str):
            setting = {"value": setting}
        if not setting or (not setting.get("force", False) and key in environ):
            values[key] = environ.get(key)
            continue
        value = setting["value"]
        if setting.get("relative", False):
            # Cargo resolves relative env values against the config's parent
            # directory above .cargo, including user-level configuration.
            value = str(Path(origins[key]).parent.parent / value)
        values[key] = value
    return values


def main() -> int:
    for key, value in effective_environment(sys.stdin.read(), dict(os.environ)).items():
        print(f"check_all: Cargo {key}={value if value is not None else '<unset>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
