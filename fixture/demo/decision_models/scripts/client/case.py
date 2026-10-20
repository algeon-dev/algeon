"""Shared case runner: argument parsing, backend selection, registries, capture output.

A case directory holds its SQL files, `questions.json` and `policy.json`. The
runner opens registries by name under `registry/`, builds the backend for the
requested mode, times every Algeon stage into the capture and writes
`captures/<case>[_<label>][.<backend>]_<mode>.json` at the end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from client.algeon import DEFAULT_URI, ROOT, Algeon, Stage
from client.capture import Capture, print_summary
from client.decision_client import (
    DecisionBatch, DecisionClient, Entity, JevBackend, OpenAICompatibleBackend, ReplayBackend)
from client.policy import Policy
from client.registry import Registry

DEMO_DIR = Path(__file__).resolve().parents[2]
REGISTRY_DIR = DEMO_DIR / "registry"
CAPTURES_DIR = DEMO_DIR / "captures"
DEFAULT_SERVER_BINARY = ROOT / "target/release/algeon_server"
BACKENDS = ("jev", "openai_compatible")


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=("live", "replay"), required=True,
                        help="live: registry first, backend on a miss; replay: registry first, captures on a miss, no API")
    parser.add_argument("--refresh", action="store_true", help="ignore registry hits and decide every entity again")
    parser.add_argument("--backend", choices=BACKENDS, default="jev")
    parser.add_argument("--model", default=None, help="backend model name (jev-1.13 or an OPENAI_COMPAT_MODEL)")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-entities", type=int, default=None, help="override the case's probe limit")
    parser.add_argument("--label", default=None, help="capture suffix, for example a time window")
    parser.add_argument("--replay-from", type=Path, action="append", default=[],
                        help="additional capture files that serve replay responses")
    parser.add_argument("--registry-dir", type=Path, default=REGISTRY_DIR)
    parser.add_argument("--captures-dir", type=Path, default=CAPTURES_DIR)
    parser.add_argument("--server-binary", type=Path, default=DEFAULT_SERVER_BINARY)
    return parser


class Case:
    def __init__(self, case_dir: Path, args: argparse.Namespace, *, license: str | None = None):
        self.dir = Path(case_dir)
        self.name = self.dir.name
        self.args = args
        self.questions = self.load_json("questions.json")
        policy_path = self.dir / "policy.json"
        self.policy = Policy.load(policy_path) if policy_path.exists() else None
        self.capture = Capture(self.name, args.mode, args.backend, label=args.label, license=license,
                               server_binary=args.server_binary)
        self.algeon = Algeon(args.uri)
        self.backend = self._build_backend()
        self.capture.data["model"] = getattr(self.backend, "model", None)

    def load_json(self, name: str) -> dict:
        with open(self.dir / name, encoding="utf-8") as handle:
            return json.load(handle)

    def sql(self, name: str) -> str:
        return (self.dir / name).read_text(encoding="utf-8")

    def _suffix(self) -> str:
        parts = [self.name]
        if self.args.label:
            parts.append(self.args.label)
        name = "_".join(parts)
        if self.args.backend != "jev":
            name += f".{self.args.backend}"
        return name

    def capture_path(self, mode: str | None = None) -> Path:
        return self.args.captures_dir / f"{self._suffix()}_{mode or self.args.mode}.json"

    def _build_backend(self):
        if self.args.mode == "replay":
            sources = [path for path in (self.capture_path("live"), *self.args.replay_from) if path.exists()]
            if not sources:
                raise SystemExit(f"replay needs {self.capture_path('live')} or --replay-from captures")
            self.capture.data["replay_sources"] = [str(path) for path in sources]
            return ReplayBackend.from_captures(sources)
        if self.args.backend == "jev":
            return JevBackend(self.args.model or "jev-1.13")
        return OpenAICompatibleBackend(self.args.model)

    def registry(self, name: str, questions: dict | None = None) -> Registry:
        """Open the named registry; `name` is shared across cases (brand_ontology, brand_canonical)."""
        file_name = name if self.args.backend == "jev" else f"{name}.{self.args.backend}"
        return Registry(self.args.registry_dir / f"{file_name}.parquet", questions or self.questions)

    def run_stage(self, name: str, sql: str, *, explain: bool = True) -> Stage:
        stage = self.algeon.run(name, sql, explain=explain)
        self.capture.stage(stage)
        print(json.dumps({"stage": name, "rows": stage.rows, "elapsed_s": round(stage.elapsed_s, 3),
                          "execution": stage.execution}), file=sys.stderr)
        return stage

    def decide(self, registry: Registry, entities: list[Entity], *, name: str = "decide",
               questions: dict | None = None) -> DecisionBatch:
        client = DecisionClient(self.backend, registry, max_workers=self.args.max_workers,
                                refresh=self.args.refresh and self.args.mode == "live")
        batch = client.decide(entities, questions or registry.questions)
        self.capture.decision_batch(name, batch)
        print(json.dumps({"decide": name, **{key: batch.summary[key] for key in (
            "entities", "registry_hits", "api_calls", "cost_usd", "wall_s")}}, default=str), file=sys.stderr)
        return batch

    def decision_table(self, table_name: str, arrow_table, file_name: str | None = None) -> Path:
        """Write a typed decision table under registry/ and register it in Algeon."""
        path = self.args.registry_dir / (file_name or f"{table_name}.engine.parquet")
        self.algeon.register_parquet(table_name, path, arrow_table)
        return path

    def finish(self) -> Path:
        path = self.capture.write(self.capture_path())
        print_summary(self.capture, path)
        return path
