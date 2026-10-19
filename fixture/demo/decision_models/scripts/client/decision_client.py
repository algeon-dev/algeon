"""Decision backends and the batched decide loop.

A decision backend answers a fixed question set for one entity state. Three
question types exist: `choice` (one option plus its probability), `noul`
(probability that a statement holds) and `score` (expected value on a numbered
rubric). Every backend returns the same normalized answer shape, so the
registry, policy and recompute SQL never depend on which backend answered.

Backends read their endpoint and credentials from environment variables only.
Nothing in this module stores, logs or raises a URL, header or token.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_QUESTIONS = 12
QUESTION_TYPES = ("choice", "noul", "score")
RETRY_STATUS = (408, 429, 500, 502, 503, 504)


class DecisionError(Exception):
    """Base class for decision-loop failures."""


class DecisionConfigError(DecisionError):
    """A backend is missing configuration or a question set is malformed."""


class ReplayMiss(DecisionError):
    """A replay run needed a response that no capture provides."""


def content_hash(state: str) -> str:
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def canonical_questions(questions: dict) -> str:
    return json.dumps(questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def question_set_hash(questions: dict) -> str:
    return hashlib.sha256(canonical_questions(questions).encode("utf-8")).hexdigest()


def validate_questions(questions: dict) -> None:
    if not isinstance(questions, dict) or not questions:
        raise DecisionConfigError("questions must be a non-empty object")
    if len(questions) > MAX_QUESTIONS:
        raise DecisionConfigError(f"question set has {len(questions)} questions; the limit is {MAX_QUESTIONS}")
    for name, spec in questions.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise DecisionConfigError(f"question name {name!r} must be a lowercase identifier")
        kind = spec.get("type")
        if kind not in QUESTION_TYPES:
            raise DecisionConfigError(f"question {name!r} has type {kind!r}; expected one of {QUESTION_TYPES}")
        if not spec.get("instructions"):
            raise DecisionConfigError(f"question {name!r} has no instructions")
        criteria = spec.get("criteria")
        if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2):
            raise DecisionConfigError(f"choice question {name!r} needs at least two criteria options")
        if kind == "noul" and (not isinstance(criteria, dict) or set(criteria) != {"true", "false"}):
            raise DecisionConfigError(f"noul question {name!r} needs criteria with keys 'true' and 'false'")
        if kind == "score" and score_range(spec) is None:
            raise DecisionConfigError(f"score question {name!r} needs criteria that start with their numeric level")


def score_range(spec: dict) -> tuple[float, float] | None:
    """Numeric bounds of a score rubric, read from the leading number of each criterion."""
    criteria = spec.get("criteria")
    labels = list(criteria.values()) if isinstance(criteria, dict) else criteria
    if not isinstance(labels, list) or not labels:
        return None
    levels = []
    for label in labels:
        match = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(label))
        if match is None:
            return None
        levels.append(float(match.group(1)))
    return min(levels), max(levels)


@dataclass(frozen=True)
class Entity:
    key: str
    state: str


@dataclass
class Decision:
    entity_key: str
    content_hash: str
    question_set_hash: str
    model_id: str
    decided_at: datetime
    answers: dict[str, dict]
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: int
    source: str
    raw: dict | None = None

    def value(self, question: str):
        answer = self.answers[question]
        return answer["choice"] if answer["type"] == "choice" else answer["value"]

    def probability(self, question: str) -> float | None:
        answer = self.answers[question]
        return answer.get("p") if answer["type"] == "choice" else answer.get("value")


@dataclass
class BackendResponse:
    answers: dict[str, dict]
    model_id: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: int
    raw: dict


@dataclass
class DecisionBatch:
    decisions: list[Decision]
    summary: dict

    def by_key(self) -> dict[str, Decision]:
        return {decision.entity_key: decision for decision in self.decisions}


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def normalize_jev_answers(payload: dict, questions: dict) -> dict[str, dict]:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise DecisionError("jev response has no answers object")
    normalized = {}
    for name, spec in questions.items():
        answer = answers.get(name)
        if not isinstance(answer, dict):
            raise DecisionError(f"jev response has no answer for question {name!r}")
        kind = spec["type"]
        if kind == "choice":
            choice = answer.get("choice")
            if choice not in spec["criteria"]:
                raise DecisionError(f"jev answered {choice!r} for {name!r}, which is not an option")
            probabilities = answer.get("probabilities") or {}
            normalized[name] = {"type": "choice", "choice": choice,
                                "p": float(probabilities.get(choice, answer.get("confidence", 0.0)))}
        elif kind == "noul":
            normalized[name] = {"type": "noul", "value": _clamp(answer["noul"], 0.0, 1.0)}
        else:
            low, high = score_range(spec)
            normalized[name] = {"type": "score", "value": _clamp(answer["score"], low, high)}
    return normalized


class Backend:
    name = "backend"
    model = ""

    def decide(self, entity: Entity, questions: dict) -> BackendResponse:
        raise NotImplementedError


def _scrub(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return re.sub(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_.-]+", "<token>", text)


def _post_json(url: str, key: str, payload: dict, *, timeout: float, retries: int, backend: str,
               secrets: tuple[str, ...]) -> tuple[dict, int]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "User-Agent": "algeon-decision-demo/1.0"}
    last_error = "no attempt"
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data, int(round((time.perf_counter() - started) * 1000))
        except urllib.error.HTTPError as error:
            detail = _scrub(error.read().decode("utf-8", errors="replace")[:300], secrets)
            last_error = f"decision backend {backend!r} returned HTTP {error.code}: {detail}"
            if error.code not in RETRY_STATUS or attempt + 1 == retries:
                raise DecisionError(last_error) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
            last_error = f"decision backend {backend!r} request failed: {_scrub(str(error), secrets)}"
            if attempt + 1 == retries:
                raise DecisionError(last_error) from None
        time.sleep(0.5 * (2 ** attempt))
    raise DecisionError(last_error)


def _require_env(*names: str) -> list[str]:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise DecisionConfigError(
            f"missing environment variable(s): {', '.join(missing)}; export them in the shell, never in files under the repository")
    return [os.environ[name] for name in names]


class JevBackend(Backend):
    """TypeSafe Jev `/systemone` request: one state, K independent questions."""

    name = "jev"

    def __init__(self, model: str = "jev-1.13", *, timeout: float = 60.0, retries: int = 3):
        self._url, self._key = _require_env("JEV_API_URL", "JEV_API_KEY")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        host = urllib.parse.urlsplit(self._url).netloc
        self._secrets = (self._key, self._url, host)

    def decide(self, entity: Entity, questions: dict) -> BackendResponse:
        payload = {"model": self.model, "state": entity.state, "questions": questions}
        data, latency_ms = _post_json(self._url, self._key, payload, timeout=self.timeout, retries=self.retries,
                                      backend=self.name, secrets=self._secrets)
        usage = data.get("usage") or {}
        return BackendResponse(
            answers=normalize_jev_answers(data, questions),
            model_id=str(data.get("model") or self.model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_usd=usage.get("cost"),
            latency_ms=latency_ms,
            raw=data,
        )


def openai_schema(questions: dict) -> dict:
    """JSON schema that makes a chat model return the normalized answer shape."""
    properties = {}
    for name, spec in questions.items():
        if spec["type"] == "choice":
            properties[name] = {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": list(spec["criteria"])},
                    "p": {"type": "number", "description": "probability that the chosen option is correct, 0 to 1"},
                },
                "required": ["choice", "p"],
                "additionalProperties": False,
            }
        else:
            properties[name] = {"type": "number"}
    return {"type": "object", "properties": properties, "required": list(questions), "additionalProperties": False}


def openai_prompt(questions: dict) -> str:
    lines = ["Evaluate each question independently against the state. Answer every question.",
             "Return JSON only, with one key per question name."]
    for name, spec in questions.items():
        kind = spec["type"]
        lines.append("")
        lines.append(f"Question `{name}` ({kind}): {spec['instructions']}")
        if kind == "choice":
            lines.append("Return {\"choice\": <option>, \"p\": <probability 0..1 that the option is correct>}. Options:")
            for option, text in spec["criteria"].items():
                lines.append(f"- {option}: {text}")
        elif kind == "noul":
            lines.append("Return the probability, 0 to 1, that the statement is true.")
            lines.append(f"- true: {spec['criteria']['true']}")
            lines.append(f"- false: {spec['criteria']['false']}")
        else:
            low, high = score_range(spec)
            lines.append(f"Return the expected value on this rubric as a number from {low:g} to {high:g}:")
            labels = spec["criteria"].values() if isinstance(spec["criteria"], dict) else spec["criteria"]
            for label in labels:
                lines.append(f"- {label}")
    return "\n".join(lines)


class OpenAICompatibleBackend(Backend):
    """Any chat-completions endpoint that honors `response_format` JSON schemas."""

    name = "openai_compatible"

    def __init__(self, model: str | None = None, *, timeout: float = 120.0, retries: int = 3):
        base_url, self._key = _require_env("OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY")
        self.model = model or os.environ.get("OPENAI_COMPAT_MODEL") or ""
        if not self.model:
            raise DecisionConfigError("set OPENAI_COMPAT_MODEL or pass --model for the openai_compatible backend")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.retries = retries
        host = urllib.parse.urlsplit(self._url).netloc
        self._secrets = (self._key, self._url, base_url, host)

    def decide(self, entity: Entity, questions: dict) -> BackendResponse:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": openai_prompt(questions)},
                {"role": "user", "content": entity.state},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "decision", "strict": True, "schema": openai_schema(questions)}},
        }
        last_error = None
        for attempt in range(self.retries):
            if attempt:
                # Resending an identical request at temperature 0 repeats the same malformed output.
                payload["temperature"] = 0.2
                payload["messages"][0]["content"] = openai_prompt(questions) + (
                    "\n\nThe previous answer did not match the schema. Return exactly one JSON object whose "
                    "top-level keys are the question names, with a plain number for every numeric question.")
            data, latency_ms = _post_json(self._url, self._key, payload, timeout=self.timeout, retries=self.retries,
                                          backend=self.name, secrets=self._secrets)
            try:
                content = data["choices"][0]["message"]["content"]
                answers = coerce_openai_answers(json.loads(content), questions)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                last_error = DecisionError(f"openai_compatible response had no JSON content: {type(error).__name__}")
                continue
            except DecisionError as error:
                # Some endpoints ignore the strict schema on a few requests; a fresh request usually conforms.
                last_error = error
                continue
            usage = data.get("usage") or {}
            return BackendResponse(
                answers=answers,
                model_id=str(data.get("model") or self.model),
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                cost_usd=None,
                latency_ms=latency_ms,
                raw={"model": data.get("model"), "usage": usage, "content": content, "schema_retries": attempt},
            )
        raise last_error


def coerce_openai_answers(parsed, questions: dict) -> dict[str, dict]:
    """Normalize a chat model's JSON into typed answers, unwrapping nested or stringified objects."""
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise DecisionError("openai_compatible response is not a JSON object")
    if not all(name in parsed for name in questions):
        for value in parsed.values():
            candidate = value
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
            if isinstance(candidate, dict) and all(name in candidate for name in questions):
                parsed = candidate
                break
    answers = {}
    for name, spec in questions.items():
        value = parsed.get(name)
        if isinstance(value, str) and spec["type"] == "choice":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {"choice": value, "p": 0.0}
        if spec["type"] == "choice":
            if not isinstance(value, dict) or value.get("choice") not in spec["criteria"]:
                raise DecisionError(
                    f"openai_compatible answered an unknown option for {name!r}: {repr(value)[:120]}")
            answers[name] = {"type": "choice", "choice": value["choice"], "p": _clamp(value.get("p", 0.0), 0.0, 1.0)}
        elif spec["type"] == "noul":
            answers[name] = {"type": "noul", "value": _clamp(_number(value, name), 0.0, 1.0)}
        else:
            low, high = score_range(spec)
            answers[name] = {"type": "score", "value": _clamp(_number(value, name), low, high)}
    return answers


def _number(value, name: str) -> float:
    if isinstance(value, dict):
        # A model that nests the number under a key such as "value" or "p" still answered the question.
        numeric = [item for item in value.values() if isinstance(item, (int, float)) and not isinstance(item, bool)]
        if len(numeric) == 1:
            return float(numeric[0])
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DecisionError(
            f"openai_compatible answered a non-numeric value for {name!r}: {type(value).__name__} {repr(value)[:80]}")
    try:
        return float(value)
    except ValueError:
        raise DecisionError(f"openai_compatible answered a non-numeric value for {name!r}: {repr(value)[:80]}") from None


def replay_key(content: str, question_set: str) -> str:
    return f"{content}:{question_set}"


class ReplayBackend(Backend):
    """Serves responses recorded by earlier live runs; never calls an API."""

    name = "replay"

    def __init__(self, responses: dict[str, dict]):
        self.responses = responses
        self.model = ""

    @classmethod
    def from_captures(cls, paths) -> "ReplayBackend":
        responses: dict[str, dict] = {}
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                capture = json.load(handle)
            responses.update(capture.get("responses") or {})
        return cls(responses)

    def decide(self, entity: Entity, questions: dict) -> BackendResponse:
        key = replay_key(content_hash(entity.state), question_set_hash(questions))
        record = self.responses.get(key)
        if record is None:
            raise ReplayMiss(f"no recorded response for entity {entity.key!r}; run with --mode live first")
        return BackendResponse(
            answers=record["answers"],
            model_id=record["model_id"],
            input_tokens=record.get("input_tokens"),
            output_tokens=record.get("output_tokens"),
            cost_usd=record.get("cost_usd"),
            latency_ms=int(record.get("latency_ms") or 0),
            raw=record.get("response"),
        )


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


class DecisionClient:
    """Registry-first decide loop with a bounded thread pool for backend calls."""

    def __init__(self, backend: Backend, registry, *, max_workers: int = 16, refresh: bool = False):
        """`refresh` ignores registry rows that existed before this process; rows it writes are reused."""
        self.backend = backend
        self.registry = registry
        self.max_workers = max_workers
        self.refresh = refresh

    def decide(self, entities: list[Entity], questions: dict) -> DecisionBatch:
        validate_questions(questions)
        set_hash = question_set_hash(questions)
        started = time.perf_counter()
        hashes = [content_hash(entity.state) for entity in entities]
        decisions: list[Decision | None] = [None] * len(entities)
        pending: dict[str, list[int]] = {}
        hits = 0
        for index, (entity, digest) in enumerate(zip(entities, hashes)):
            stale = self.refresh and (digest, set_hash) not in self.registry.fresh
            cached = None if stale else self.registry.lookup(digest, set_hash)
            if cached is not None:
                hits += 1
                decisions[index] = Decision(
                    entity_key=entity.key, content_hash=digest, question_set_hash=set_hash,
                    model_id=cached.model_id, decided_at=cached.decided_at, answers=cached.answers,
                    input_tokens=cached.input_tokens, output_tokens=None, cost_usd=None,
                    latency_ms=cached.latency_ms, source="registry")
            else:
                pending.setdefault(digest, []).append(index)

        latencies: list[int] = []
        totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        cost_known = True
        model_ids: set[str] = set()

        def call(digest: str) -> tuple[str, BackendResponse]:
            return digest, self.backend.decide(entities[pending[digest][0]], questions)

        new_decisions = []
        failures: list[DecisionError] = []
        if pending:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                outcomes = []
                for future in [pool.submit(call, digest) for digest in pending]:
                    try:
                        outcomes.append(future.result())
                    except DecisionError as error:
                        failures.append(error)
                for digest, response in outcomes:
                    decided_at = utc_now()
                    latencies.append(response.latency_ms)
                    model_ids.add(response.model_id)
                    totals["input_tokens"] += response.input_tokens or 0
                    totals["output_tokens"] += response.output_tokens or 0
                    if response.cost_usd is None:
                        cost_known = False
                    else:
                        totals["cost_usd"] += response.cost_usd
                    source = "replay" if isinstance(self.backend, ReplayBackend) else "backend"
                    for index in pending[digest]:
                        decision = Decision(
                            entity_key=entities[index].key, content_hash=digest, question_set_hash=set_hash,
                            model_id=response.model_id, decided_at=decided_at, answers=response.answers,
                            input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                            cost_usd=response.cost_usd, latency_ms=response.latency_ms, source=source,
                            raw=response.raw)
                        decisions[index] = decision
                        new_decisions.append(decision)
            # Decided entities reach the registry before any failure is raised, so a rerun only retries the rest.
            self.registry.upsert(new_decisions)
            self.registry.save()
            if failures:
                kind = ReplayMiss if all(isinstance(error, ReplayMiss) for error in failures) else DecisionError
                raise kind(f"{len(failures)} of {len(pending)} backend calls failed; first failure: {failures[0]}")
        for decision in decisions:
            if decision is not None and decision.source == "registry":
                model_ids.add(decision.model_id)
        wall_s = time.perf_counter() - started
        calls = len(pending)
        summary = {
            "backend": self.backend.name,
            "model_ids": sorted(model_ids),
            "entities": len(entities),
            "distinct_states": len(set(hashes)),
            "registry_hits": hits,
            "hit_rate": (hits / len(entities)) if entities else None,
            "backend_calls": calls,
            "api_calls": 0 if isinstance(self.backend, ReplayBackend) else calls,
            "max_workers": self.max_workers,
            "wall_s": wall_s,
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "cost_usd": totals["cost_usd"] if cost_known else None,
            "latency_ms_p50": _percentile(latencies, 0.5),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "latency_ms_max": max(latencies) if latencies else None,
        }
        return DecisionBatch([decision for decision in decisions if decision is not None], summary)
