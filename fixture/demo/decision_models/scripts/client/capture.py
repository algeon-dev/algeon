"""Run capture: timings, token counts, cost, hit rate, recorded responses and result tables.

A capture is self-sufficient for replay: `responses` maps
`content_hash:question_set_hash` to the normalized answers of every decision
the run used, with the raw backend response for those fetched in that run. The writer refuses any content that looks like a token or names the
decision endpoint host, so a capture can be shared.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.parse

from client.decision_client import DecisionBatch, replay_key

ROOT = Path(__file__).resolve().parents[5]
MERREC_COMMON = ROOT / "scripts/dev/merrec_verify_common.py"
JWT_SHAPE = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}")


def merrec_license() -> str:
    spec = importlib.util.spec_from_file_location("merrec_verify_common", MERREC_COMMON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MERREC_LICENSE


def forbidden_fragments() -> list[str]:
    """Endpoint hosts and keys from the environment; present only when the shell exports them."""
    fragments = []
    for name in ("JEV_API_URL", "OPENAI_COMPAT_BASE_URL"):
        value = os.environ.get(name)
        if value:
            host = urllib.parse.urlsplit(value).netloc
            fragments.extend(fragment for fragment in (value, host) if fragment)
    for name in ("JEV_API_KEY", "OPENAI_COMPAT_API_KEY"):
        value = os.environ.get(name)
        if value:
            fragments.append(value)
    return fragments


def scan_secrets(text: str) -> str | None:
    if JWT_SHAPE.search(text):
        return "JWT-shaped string"
    if "Bearer " in text:
        return "Bearer literal"
    for fragment in forbidden_fragments():
        if fragment in text:
            return "decision endpoint host or key"
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(server_binary: Path | None) -> dict:
    record = {
        "git_head": git_head(),
        "python": platform.python_version(),
        "pyarrow": __import__("pyarrow").__version__,
    }
    if server_binary is not None and Path(server_binary).exists():
        record["server_binary"] = str(Path(server_binary).resolve().relative_to(ROOT)) \
            if Path(server_binary).resolve().is_relative_to(ROOT) else str(server_binary)
        record["server_binary_sha256"] = file_sha256(Path(server_binary))
    return record


class Capture:
    def __init__(self, case: str, mode: str, backend: str, *, label: str | None = None,
                 license: str | None = None, server_binary: Path | None = None):
        self.data = {
            "case": case,
            "label": label,
            "mode": mode,
            "backend": backend,
            "model_ids": [],
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": environment(server_binary),
            "stages": [],
            "decisions": [],
            "responses": {},
            "results": {},
            "notes": [],
        }
        if license:
            self.data["license"] = license

    def stage(self, stage) -> None:
        self.data["stages"].append(stage.record())

    def decision_batch(self, name: str, batch: DecisionBatch) -> None:
        self.data["decisions"].append({"name": name, **batch.summary})
        models = set(self.data["model_ids"]) | set(batch.summary["model_ids"])
        self.data["model_ids"] = sorted(models)
        # Registry-served decisions are recorded too (without a raw response), so a capture alone can replay
        # a run whose answers came from an earlier run's registry. A later registry hit on a state this
        # run already fetched keeps the fetched record, which carries the raw response and the cost.
        for decision in batch.decisions:
            key = replay_key(decision.content_hash, decision.question_set_hash)
            if decision.raw is None and key in self.data["responses"]:
                continue
            self.data["responses"][key] = {
                "entity_key": decision.entity_key,
                "source": decision.source,
                "model_id": decision.model_id,
                "answers": decision.answers,
                "input_tokens": decision.input_tokens,
                "output_tokens": decision.output_tokens,
                "cost_usd": decision.cost_usd,
                "latency_ms": decision.latency_ms,
                "response": decision.raw,
            }

    def result(self, name: str, value) -> None:
        self.data["results"][name] = value

    def note(self, text: str) -> None:
        self.data["notes"].append(text)

    def totals(self) -> dict:
        batches = self.data["decisions"]
        stages = self.data["stages"]
        costs = [batch["cost_usd"] for batch in batches]
        return {
            "gpu_elapsed_s": sum(stage["elapsed_s"] for stage in stages),
            "decide_wall_s": sum(batch["wall_s"] for batch in batches),
            "entities": sum(batch["entities"] for batch in batches),
            "registry_hits": sum(batch["registry_hits"] for batch in batches),
            "api_calls": sum(batch["api_calls"] for batch in batches),
            "input_tokens": sum(batch["input_tokens"] for batch in batches),
            "output_tokens": sum(batch["output_tokens"] for batch in batches),
            "cost_usd": None if any(cost is None for cost in costs) else sum(costs),
            "max_host_boundaries": max(
                (((stage.get("execution") or {}).get("host_datafusion_boundary_count") or 0) for stage in stages),
                default=0),
        }

    def write(self, path: Path) -> Path:
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.data["totals"] = self.totals()
        text = json.dumps(self.data, indent=2, default=str, ensure_ascii=False) + "\n"
        problem = scan_secrets(text)
        if problem:
            raise RuntimeError(f"refusing to write capture {path}: content contains a {problem}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        return path


def print_summary(capture: Capture, path: Path) -> None:
    totals = capture.data["totals"]
    print(json.dumps({"capture": str(path), **totals}, default=str), file=sys.stderr)
