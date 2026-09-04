#!/usr/bin/env python3
"""Controlled clean-build and CPU artifact payload policy helper."""

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
DEFAULT_POLICY = ROOT / ".config" / "policies" / "cpu-packaging.toml"
SERVER_BINARY = "datafusion_nexus_server"
MANIFEST_SCHEMA_VERSION = 1


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
    for list_name in ("pass_through", "explicit_inputs"):
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
        missing_str = ", ".join(sorted(missing))
        raise PolicyError(f"cpu is missing: {missing_str}")
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


def artifact_affecting_fingerprint(policy: Mapping[str, Any]) -> str:
    cpu = policy["cpu"]
    projection = {
        "cpu": {
            "baseline": cpu["baseline"],
            "target_triple": cpu["target_triple"],
            "rustflags": list(cpu["rustflags"]),
            "cargo_profile": cpu["cargo_profile"],
            "artifact_dir": cpu["artifact_dir"],
            "image_arch": cpu["image_arch"],
        },
        "fixed_build_environment": dict(sorted(policy["clean_environment"]["fixed_build_environment"].items())),
        "tool_version_commands": dict(sorted(policy["clean_environment"]["tool_version_commands"].items())),
    }
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_features(raw: str) -> list[str]:
    features = [feature.strip() for feature in raw.split(",") if feature.strip()]
    if not features:
        raise PolicyError("packaging features must not be empty")
    if "server" in features:
        raise PolicyError("the nexus-server package has no server feature")
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


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PolicyError(f"JSON {path} must contain an object")
    return value


def create_build_record(
    *,
    policy: Mapping[str, Any],
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
    record = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "record_kind": "canonical-build-record",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy_fingerprint": artifact_affecting_fingerprint(policy),
        "fixed_build_environment": dict(
            sorted(policy["clean_environment"]["fixed_build_environment"].items())
        ),
        "fixed_tool_versions": dict(fixed_tool_versions),
        "cpu": cpu_snapshot(policy["cpu"]),
        "features": list(features),
        "feature_string": ",".join(features),
        "binary_name": SERVER_BINARY,
        "binary_sha256": sha256_file(binary),
        "rustc_version": run_text([rustc_path, "--version"], cwd=cwd, environment=environment),
        "rustc_vv": run_text([rustc_path, "-Vv"], cwd=cwd, environment=environment),
        "cargo_version": run_text([cargo_path, "--version"], cwd=cwd, environment=environment),
        "cargo_command": list(cargo_command),
    }
    atomic_json(output, record)
    return record


def verify_build_record(
    *,
    policy: Mapping[str, Any],
    features: Sequence[str],
    binary: Path,
    record_path: Path,
) -> dict[str, Any]:
    record = load_json(record_path)
    if record.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PolicyError(f"build record schema_version mismatch: {record.get('schema_version')}")
    expected_fingerprint = artifact_affecting_fingerprint(policy)
    if record.get("policy_fingerprint") != expected_fingerprint:
        raise PolicyError(
            f"policy_fingerprint mismatch: record={record.get('policy_fingerprint')!r}, current={expected_fingerprint}"
        )
    if record.get("features") != list(features):
        raise PolicyError(f"features mismatch: record={record.get('features')!r}, requested={list(features)!r}")
    if record.get("binary_name") != SERVER_BINARY:
        raise PolicyError(f"binary name mismatch: {record.get('binary_name')!r}")
    if not binary.is_file():
        raise PolicyError(f"canonical artifact is missing: {binary}")
    actual_hash = sha256_file(binary)
    if record.get("binary_sha256") != actual_hash:
        raise PolicyError(f"canonical artifact hash mismatch: record={record.get('binary_sha256')!r}, actual={actual_hash}")
    return record


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
        "CPU_POLICY_FINGERPRINT": artifact_affecting_fingerprint(policy),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")


def command_normalize_features(args: argparse.Namespace) -> None:
    print(",".join(canonical_features(args.features)))


def command_build(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    policy = load_policy(policy_path)
    cpu = policy["cpu"]
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
        record_path = binary.with_name(binary.name + ".build-record.json")
        create_build_record(
            policy=policy,
            features=features,
            binary=binary,
            output=record_path,
            cargo_command=command,
            cargo_path=cargo_path,
            rustc_path=rustc_path,
            cwd=cwd,
            environment=environment,
            fixed_tool_versions=fixed_tool_versions,
        )
    print(f"==> canonical artifact: {binary}")
    print(f"==> build record: {record_path}")


def command_verify_build(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    policy = load_policy(policy_path)
    features = canonical_features(args.features)
    verify_build_record(
        policy=policy,
        features=features,
        binary=args.binary.resolve(),
        record_path=args.record.resolve(),
    )
    print("OK: canonical build artifact hash and record match the selected policy.")


def command_create_manifest(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    policy = load_policy(policy_path)
    dist_dir = args.dist_dir.resolve()
    record_path = args.build_record.resolve()
    record = load_json(record_path)

    expected_fingerprint = artifact_affecting_fingerprint(policy)
    if record.get("policy_fingerprint") != expected_fingerprint:
        raise PolicyError("build record policy fingerprint does not match current policy")

    output_path = args.output.resolve()
    files: dict[str, str] = {}
    for entry in sorted(dist_dir.rglob("*")):
        if entry.is_file() and entry != output_path:
            rel_path = str(entry.relative_to(dist_dir))
            files[rel_path] = sha256_file(entry)

    staged_binary_rel = f"bin/{SERVER_BINARY}"
    if staged_binary_rel not in files:
        raise PolicyError(f"staged binary {staged_binary_rel} missing from dist")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_kind": "cpu-payload-manifest",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy_fingerprint": expected_fingerprint,
        "cpu": record["cpu"],
        "features": record["features"],
        "feature_string": record["feature_string"],
        "binary_name": SERVER_BINARY,
        "binary_sha256": files[staged_binary_rel],
        "source_binary_sha256": record["binary_sha256"],
        "files": files,
    }
    atomic_json(output_path, manifest)
    print(f"==> payload manifest: {output_path}")


def command_verify_manifest(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    policy = load_policy(policy_path)
    dist_dir = args.dist_dir.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PolicyError(f"manifest schema_version mismatch: {manifest.get('schema_version')}")

    expected_fingerprint = artifact_affecting_fingerprint(policy)
    if manifest.get("policy_fingerprint") != expected_fingerprint:
        raise PolicyError(
            f"policy_fingerprint mismatch: manifest={manifest.get('policy_fingerprint')!r}, current={expected_fingerprint}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise PolicyError("manifest contains no files")

    actual_files = {
        str(entry.relative_to(dist_dir))
        for entry in dist_dir.rglob("*")
        if entry.is_file() and entry.resolve() != manifest_path.resolve()
    }
    expected_files = set(files.keys())
    if actual_files != expected_files:
        extra = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise PolicyError(f"staged payload file inventory mismatch: extra={extra}, missing={missing}")

    for rel_path, expected_hash in files.items():
        file_path = dist_dir / rel_path
        if not file_path.is_file():
            raise PolicyError(f"staged payload file missing: {rel_path}")
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            raise PolicyError(f"hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")

    staged_binary_rel = f"bin/{SERVER_BINARY}"
    if files.get(staged_binary_rel) != manifest.get("binary_sha256"):
        raise PolicyError("binary_sha256 does not match staged binary file hash")

    cpu = manifest["cpu"]
    manifest_sha256 = sha256_file(manifest_path)
    values = {
        "CPU_BASELINE": cpu["baseline"],
        "CPU_IMAGE_ARCH": cpu["image_arch"],
        "CPU_TARGET_TRIPLE": cpu["target_triple"],
        "CPU_BINARY_SHA256": manifest["binary_sha256"],
        "CPU_MANIFEST_SHA256": manifest_sha256,
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")
    print("OK: payload manifest and staged file hashes match policy.", file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    subparsers = result.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe")
    describe.set_defaults(function=command_describe)

    features = subparsers.add_parser("normalize-features")
    features.add_argument("--features", required=True)
    features.set_defaults(function=command_normalize_features)

    build = subparsers.add_parser("build")
    build.add_argument("--features", required=True)
    build.add_argument("--target-root", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--explicit-env", action="append", default=[])
    build.set_defaults(function=command_build)

    verify_build = subparsers.add_parser("verify-build")
    verify_build.add_argument("--features", required=True)
    verify_build.add_argument("--binary", type=Path, required=True)
    verify_build.add_argument("--record", type=Path, required=True)
    verify_build.set_defaults(function=command_verify_build)

    create_manifest = subparsers.add_parser("create-manifest")
    create_manifest.add_argument("--dist-dir", type=Path, required=True)
    create_manifest.add_argument("--build-record", type=Path, required=True)
    create_manifest.add_argument("--output", type=Path, required=True)
    create_manifest.set_defaults(function=command_create_manifest)

    verify_manifest = subparsers.add_parser("verify-manifest")
    verify_manifest.add_argument("--manifest", type=Path, required=True)
    verify_manifest.add_argument("--dist-dir", type=Path, required=True)
    verify_manifest.set_defaults(function=command_verify_manifest)

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
