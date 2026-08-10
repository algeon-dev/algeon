#!/usr/bin/env python3
"""CPU packaging policy, canonical build, and provenance helper."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "config" / "cpu-packaging.toml"
SERVER_BINARY = "datafusion_nexus_server"


class PolicyError(RuntimeError):
    """A packaging input or artifact violates the declared policy."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as policy_file:
            policy = tomllib.load(policy_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PolicyError(f"cannot load CPU packaging policy {path}: {error}") from error

    if set(policy) != {"clean_environment", "cpu"}:
        raise PolicyError("CPU packaging policy top-level keys must be exactly clean_environment and cpu")
    clean = policy.get("clean_environment")
    cpu = policy.get("cpu")
    if not isinstance(clean, dict) or not isinstance(cpu, dict):
        raise PolicyError("policy requires clean_environment and cpu")
    for list_name in (
        "pass_through",
        "explicit_inputs",
        "forbidden_exact",
        "forbidden_patterns",
    ):
        value = clean.get(list_name)
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise PolicyError(f"clean_environment.{list_name} must be a non-empty-string list")
    fixed_environment = clean.get("fixed_build_environment")
    tool_commands = clean.get("tool_version_commands")
    if not isinstance(fixed_environment, dict) or not fixed_environment:
        raise PolicyError("clean_environment.fixed_build_environment must be a non-empty table")
    if not all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in fixed_environment.items()):
        raise PolicyError("fixed build environment names and values must be non-empty strings")
    if set(clean["pass_through"]) & set(fixed_environment):
        raise PolicyError("fixed build environment variables must not also be ambient pass-throughs")
    if not isinstance(tool_commands, dict) or not tool_commands:
        raise PolicyError("clean_environment.tool_version_commands must be a non-empty table")
    if not all(
        isinstance(name, str)
        and name
        and isinstance(command, list)
        and command
        and all(isinstance(argument, str) and argument for argument in command)
        for name, command in tool_commands.items()
    ):
        raise PolicyError("tool version commands must be non-empty string lists")

    required = {
        "baseline",
        "target_triple",
        "rustflags",
        "cargo_profile",
        "artifact_dir",
        "image_arch",
        "deployment_requirement",
    }
    missing = required - cpu.keys()
    if missing:
        raise PolicyError(f"cpu is missing: {', '.join(sorted(missing))}")
    for key in required - {"rustflags"}:
        if not isinstance(cpu[key], str) or not cpu[key]:
            raise PolicyError(f"cpu.{key} must be a non-empty string")
    expected_flags = ["-C", f"target-cpu={cpu['baseline']}"]
    if cpu["rustflags"] != expected_flags:
        raise PolicyError(f"cpu.rustflags must be exactly {expected_flags!r}")
    for key in ("target_triple", "cargo_profile", "artifact_dir", "image_arch"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", cpu[key]):
            raise PolicyError(f"cpu.{key} contains unsafe characters")
    return policy


def canonical_features(raw: str) -> list[str]:
    features = [feature.strip() for feature in raw.split(",") if feature.strip()]
    if not features:
        raise PolicyError("packaging features must not be empty")
    if "server" in features:
        raise PolicyError("the nexus-server package has no 'server' feature")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", feature) for feature in features):
        raise PolicyError(f"invalid Cargo feature list: {raw!r}")
    return sorted(set(features))


def cpu_snapshot(cpu: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": cpu["baseline"],
        "target_triple": cpu["target_triple"],
        "effective_rustflags": list(cpu["rustflags"]),
        "cargo_profile": cpu["cargo_profile"],
        "artifact_dir": cpu["artifact_dir"],
        "image_arch": cpu["image_arch"],
        "deployment_requirement": cpu["deployment_requirement"],
    }


def ambient_overrides(policy: Mapping[str, Any], environment: Mapping[str, str]) -> list[str]:
    clean = policy["clean_environment"]
    forbidden = set(clean["forbidden_exact"])
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in clean["forbidden_patterns"]]
    rejected = []
    for name in environment:
        upper = name.upper()
        if upper in forbidden or any(pattern.search(upper) for pattern in patterns):
            rejected.append(name)
    return sorted(rejected)


def reject_ambient_overrides(policy: Mapping[str, Any]) -> None:
    rejected = ambient_overrides(policy, os.environ)
    if rejected:
        raise PolicyError(
            "ambient codegen/config override variables are forbidden for CPU packaging: "
            + ", ".join(rejected)
        )


def parse_explicit_env(policy: Mapping[str, Any], values: Sequence[str]) -> dict[str, str]:
    allowed = set(policy["clean_environment"]["explicit_inputs"])
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise PolicyError(f"explicit environment input must be NAME=VALUE: {value!r}")
        name, content = value.split("=", 1)
        if name not in allowed:
            raise PolicyError(f"explicit environment input {name!r} is not allowlisted")
        if not content:
            raise PolicyError(f"explicit environment input {name!r} must not be empty")
        parsed[name] = content
    return parsed


def clean_build_environment(
    policy: Mapping[str, Any], cargo_home: Path, explicit: Mapping[str, str]
) -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in policy["clean_environment"]["pass_through"]
        if name in os.environ and os.environ[name]
    }
    environment.update(policy["clean_environment"]["fixed_build_environment"])
    environment.update(explicit)
    environment["CARGO_HOME"] = str(cargo_home)
    environment["CARGO_TERM_COLOR"] = "never"
    return environment


def prepare_clean_cargo_home(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    source_home = Path.home() / ".cargo"
    for cache_name in ("registry", "git"):
        source = source_home / cache_name
        if source.exists():
            (destination / cache_name).symlink_to(source, target_is_directory=True)


def ensure_clean_cwd(path: Path) -> None:
    for candidate in (path, *path.parents):
        cargo_dir = candidate / ".cargo"
        for config_name in ("config", "config.toml"):
            if (cargo_dir / config_name).exists():
                raise PolicyError(f"unexpected Cargo config can affect canonical build: {cargo_dir / config_name}")


def run_text(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PolicyError(f"tool command failed ({shlex.join(command)}): {error}") from error
    return result.stdout.rstrip("\n")


def collect_fixed_tool_versions(
    policy: Mapping[str, Any], *, cwd: Path, environment: Mapping[str, str]
) -> dict[str, str]:
    return {
        name: run_text(command, cwd=cwd, environment=environment)
        for name, command in sorted(policy["clean_environment"]["tool_version_commands"].items())
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_build_provenance(
    *,
    policy_path: Path,
    cpu: Mapping[str, Any],
    features: Sequence[str],
    binary: Path,
    output: Path,
    cargo_command: Sequence[str],
    cargo_path: str,
    rustc_path: str,
    cwd: Path,
    environment: Mapping[str, str],
    fixed_tool_versions: Mapping[str, str],
) -> dict[str, Any]:
    if not binary.is_file():
        raise PolicyError(f"canonical Cargo build did not produce {binary}")
    provenance = {
        "provenance_kind": "canonical-build-artifact",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy_sha256": sha256_file(policy_path),
        "helper_sha256": sha256_file(Path(__file__).resolve()),
        "fixed_build_environment": dict(
            sorted(load_policy(policy_path)["clean_environment"]["fixed_build_environment"].items())
        ),
        "fixed_tool_versions": dict(fixed_tool_versions),
        "cpu": cpu_snapshot(cpu),
        "features": list(features),
        "feature_string": ",".join(features),
        "binary_name": SERVER_BINARY,
        "binary_sha256": sha256_file(binary),
        "rustc_version": run_text([rustc_path, "--version"], cwd=cwd, environment=environment),
        "rustc_vv": run_text([rustc_path, "-Vv"], cwd=cwd, environment=environment),
        "cargo_version": run_text([cargo_path, "--version"], cwd=cwd, environment=environment),
        "cargo_command": list(cargo_command),
    }
    atomic_json(output, provenance)
    return provenance


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyError(f"cannot load provenance {path}: {error}") from error
    if not isinstance(value, dict):
        raise PolicyError(f"provenance {path} must contain a JSON object")
    return value


def verify_build_provenance(
    *,
    policy_path: Path,
    policy: Mapping[str, Any],
    cpu: Mapping[str, Any],
    features: Sequence[str],
    binary: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    provenance = load_json(provenance_path)
    expected = {
        "provenance_kind": "canonical-build-artifact",
        "policy_sha256": sha256_file(policy_path),
        "helper_sha256": sha256_file(Path(__file__).resolve()),
        "fixed_build_environment": dict(
            sorted(policy["clean_environment"]["fixed_build_environment"].items())
        ),
        "cpu": cpu_snapshot(cpu),
        "features": list(features),
        "feature_string": ",".join(features),
        "binary_name": SERVER_BINARY,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise PolicyError(f"build provenance mismatch for {key}: expected {value!r}")
    for key in ("rustc_version", "rustc_vv", "cargo_version", "cargo_command"):
        if not provenance.get(key):
            raise PolicyError(f"build provenance is missing {key}")
    tool_versions = provenance.get("fixed_tool_versions")
    expected_tools = set(policy["clean_environment"]["tool_version_commands"])
    if not isinstance(tool_versions, dict) or set(tool_versions) != expected_tools:
        raise PolicyError("build provenance fixed_tool_versions do not match policy tools")
    if not all(isinstance(version, str) and version for version in tool_versions.values()):
        raise PolicyError("build provenance contains an empty fixed tool version")
    if not binary.is_file():
        raise PolicyError(f"canonical artifact is missing: {binary}")
    actual_hash = sha256_file(binary)
    if provenance.get("binary_sha256") != actual_hash:
        raise PolicyError(
            f"canonical artifact hash mismatch: provenance={provenance.get('binary_sha256')!r}, actual={actual_hash}"
        )
    return provenance


def write_labels(path: Path, provenance: Mapping[str, Any]) -> None:
    cpu = provenance["cpu"]
    values = {
        "binary_sha256": provenance["binary_sha256"],
        "cpu_baseline": cpu["baseline"],
        "cpu_arch": cpu["image_arch"],
        "cpu_target": cpu["target_triple"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}={values[key]}\n" for key in sorted(values)), encoding="utf-8")


def parse_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PolicyError(f"cannot load provenance labels {path}: {error}") from error
    for line in lines:
        if not line or "=" not in line:
            raise PolicyError(f"malformed provenance label line: {line!r}")
        key, value = line.split("=", 1)
        if key in labels or not key or not value:
            raise PolicyError(f"malformed or duplicate provenance label: {key!r}")
        labels[key] = value
    return labels


def verify_staged_labels(provenance: Mapping[str, Any], labels: Mapping[str, str]) -> None:
    cpu = provenance.get("cpu", {})
    expected = {
        "binary_sha256": provenance.get("binary_sha256"),
        "cpu_baseline": cpu.get("baseline"),
        "cpu_arch": cpu.get("image_arch"),
        "cpu_target": cpu.get("target_triple"),
    }
    if provenance.get("provenance_kind") != "staged-image-artifact":
        raise PolicyError("image provenance must describe a staged-image-artifact")
    for key, value in expected.items():
        if labels.get(key) != value:
            raise PolicyError(f"provenance label mismatch for {key}: expected {value!r}")


def command_describe(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy)
    cpu = policy["cpu"]
    values = {
        "CPU_BASELINE": cpu["baseline"],
        "CPU_TARGET_TRIPLE": cpu["target_triple"],
        "CPU_CARGO_PROFILE": cpu["cargo_profile"],
        "CPU_ARTIFACT_DIR": cpu["artifact_dir"],
        "CPU_IMAGE_ARCH": cpu["image_arch"],
        "CPU_DEPLOYMENT_REQUIREMENT": cpu["deployment_requirement"],
        "CPU_RUSTFLAGS_JSON": json.dumps(cpu["rustflags"], separators=(",", ":")),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")


def command_check_env(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy)
    reject_ambient_overrides(policy)
    print("OK: ambient packaging environment contains no codegen/config overrides.")


def command_normalize_features(args: argparse.Namespace) -> None:
    print(",".join(canonical_features(args.features)))


def command_build(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    policy = load_policy(policy_path)
    cpu = policy["cpu"]
    reject_ambient_overrides(policy)
    features = canonical_features(args.features)
    explicit = parse_explicit_env(policy, args.explicit_env)

    cargo_path = shutil.which("cargo")
    rustc_path = shutil.which("rustc")
    if cargo_path is None or rustc_path is None:
        raise PolicyError("cargo and rustc must be available on PATH")

    target_root = args.target_root.resolve()
    cargo_target_dir = target_root / cpu["artifact_dir"]
    cargo_home = target_root / ".cargo-home"
    prepare_clean_cargo_home(cargo_home)
    environment = clean_build_environment(policy, cargo_home, explicit)
    target = cpu["target_triple"]
    rustflags_config = json.dumps(cpu["rustflags"], separators=(",", ":"))
    command = [
        cargo_path,
        "--config",
        f"build.target={json.dumps(target)}",
        "--config",
        f"target.{target}.rustflags={rustflags_config}",
        "build",
        "--manifest-path",
        str(args.manifest.resolve()),
        "--locked",
        "--profile",
        cpu["cargo_profile"],
        "--target",
        target,
        "--target-dir",
        str(cargo_target_dir),
        "--no-default-features",
        "--features",
        ",".join(features),
        "--package",
        "nexus-server",
        "--bin",
        SERVER_BINARY,
    ]
    with tempfile.TemporaryDirectory(prefix="datafusion-nexus-cpu-package-") as temporary:
        cwd = Path(temporary)
        ensure_clean_cwd(cwd)
        fixed_tool_versions = collect_fixed_tool_versions(policy, cwd=cwd, environment=environment)
        try:
            subprocess.run(command, cwd=cwd, env=environment, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise PolicyError(f"canonical Cargo build failed: {error}") from error

        binary = cargo_target_dir / target / cpu["cargo_profile"] / SERVER_BINARY
        provenance_path = binary.with_name(binary.name + ".cpu-provenance.json")
        create_build_provenance(
            policy_path=policy_path,
            cpu=cpu,
            features=features,
            binary=binary,
            output=provenance_path,
            cargo_command=command,
            cargo_path=cargo_path,
            rustc_path=rustc_path,
            cwd=cwd,
            environment=environment,
            fixed_tool_versions=fixed_tool_versions,
        )
    print(f"==> canonical artifact: {binary}")
    print(f"==> build provenance: {provenance_path}")


def command_verify_build(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    policy = load_policy(policy_path)
    cpu = policy["cpu"]
    features = canonical_features(args.features)
    verify_build_provenance(
        policy_path=policy_path,
        policy=policy,
        cpu=cpu,
        features=features,
        binary=args.binary.resolve(),
        provenance_path=args.provenance.resolve(),
    )
    print("OK: canonical build artifact hash and provenance match the selected policy.")


def command_stage(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    policy = load_policy(policy_path)
    cpu = policy["cpu"]
    features = canonical_features(args.features)
    build = verify_build_provenance(
        policy_path=policy_path,
        policy=policy,
        cpu=cpu,
        features=features,
        binary=args.source_binary.resolve(),
        provenance_path=args.build_provenance.resolve(),
    )
    staged_binary = args.staged_binary.resolve()
    if not staged_binary.is_file():
        raise PolicyError(f"staged binary is missing: {staged_binary}")
    provenance = dict(build)
    provenance.update(
        {
            "provenance_kind": "staged-image-artifact",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_binary_sha256": build["binary_sha256"],
            "binary_sha256": sha256_file(staged_binary),
        }
    )
    atomic_json(args.output.resolve(), provenance)
    write_labels(args.labels_output.resolve(), provenance)
    verify_staged_labels(provenance, parse_labels(args.labels_output.resolve()))
    print(f"==> staged provenance: {args.output.resolve()}")
    print(f"==> Docker label inputs: {args.labels_output.resolve()}")


def command_verify_image_inputs(args: argparse.Namespace) -> None:
    provenance_path = args.provenance.resolve()
    provenance = load_json(provenance_path)
    labels = parse_labels(args.labels.resolve())
    verify_staged_labels(provenance, labels)
    expected_cli = {
        "cpu_baseline": args.baseline,
        "cpu_arch": args.arch,
        "cpu_target": args.target,
        "binary_sha256": args.binary_sha256,
    }
    for key, value in expected_cli.items():
        if value is not None and labels.get(key) != value:
            raise PolicyError(f"Docker build input mismatch for {key}: expected {labels.get(key)!r}, got {value!r}")
    if args.provenance_sha256 is not None:
        actual = sha256_file(provenance_path)
        if args.provenance_sha256 != actual:
            raise PolicyError(
                f"Docker build input mismatch for provenance sha256: expected {actual}, got {args.provenance_sha256}"
            )
    print("OK: provenance, label inputs, binary hash, and Docker build inputs agree.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    subparsers = result.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe")
    describe.set_defaults(function=command_describe)

    check_env = subparsers.add_parser("check-env")
    check_env.set_defaults(function=command_check_env)

    features = subparsers.add_parser("normalize-features")
    features.add_argument("--features", required=True)
    features.set_defaults(function=command_normalize_features)

    build = subparsers.add_parser("build")
    build.add_argument("--features", required=True)
    build.add_argument("--target-root", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--explicit-env", action="append", default=[])
    build.set_defaults(function=command_build)

    verify = subparsers.add_parser("verify-build")
    verify.add_argument("--features", required=True)
    verify.add_argument("--binary", type=Path, required=True)
    verify.add_argument("--provenance", type=Path, required=True)
    verify.set_defaults(function=command_verify_build)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--features", required=True)
    stage.add_argument("--source-binary", type=Path, required=True)
    stage.add_argument("--staged-binary", type=Path, required=True)
    stage.add_argument("--build-provenance", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--labels-output", type=Path, required=True)
    stage.set_defaults(function=command_stage)

    image = subparsers.add_parser("verify-image-inputs")
    image.add_argument("--provenance", type=Path, required=True)
    image.add_argument("--labels", type=Path, required=True)
    image.add_argument("--baseline")
    image.add_argument("--arch")
    image.add_argument("--target")
    image.add_argument("--binary-sha256")
    image.add_argument("--provenance-sha256")
    image.set_defaults(function=command_verify_image_inputs)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except PolicyError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
