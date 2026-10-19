"""Unit tests for the decision client: hashing, registry hits, replay consistency, policy bands.

Run from the repository root:

    python3 -m unittest discover -s fixture/demo/decision_models/scripts/tests -t fixture/demo/decision_models/scripts
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client.capture import Capture, scan_secrets  # noqa: E402
from client.decision_client import (  # noqa: E402
    Backend, BackendResponse, Decision, DecisionClient, DecisionConfigError, Entity, ReplayBackend, ReplayMiss,
    content_hash, normalize_jev_answers, openai_schema, question_set_hash, score_range, validate_questions)
from client.policy import Policy  # noqa: E402
from client.registry import Registry  # noqa: E402

QUESTIONS = {
    "relationship_type": {
        "type": "choice",
        "instructions": "Relationship between the two brand entries.",
        "criteria": {"same_entity_alias": "same brand", "distinct_co_liked_peers": "different brands"},
    },
    "should_merge_brand_id": {
        "type": "noul",
        "instructions": "Should the two ids be merged?",
        "criteria": {"true": "same entity", "false": "different entities"},
    },
    "alignment_score": {
        "type": "score",
        "instructions": "Rate the equivalence.",
        "criteria": ["0 - separate", "1 - related", "2 - sub-brand", "3 - identical"],
    },
}


class FakeBackend(Backend):
    """Deterministic backend that counts calls; the answer depends on the state text."""

    name = "fake"
    model = "fake-1"

    def __init__(self):
        self.calls = 0

    def decide(self, entity: Entity, questions: dict) -> BackendResponse:
        self.calls += 1
        alias = "TXT" in entity.state
        answers = {
            "relationship_type": {"type": "choice", "choice": "same_entity_alias" if alias else "distinct_co_liked_peers",
                                  "p": 0.9 if alias else 0.8},
            "should_merge_brand_id": {"type": "noul", "value": 0.85 if alias else 0.1},
            "alignment_score": {"type": "score", "value": 2.7 if alias else 0.3},
        }
        return BackendResponse(answers=answers, model_id="fake/1", input_tokens=10, output_tokens=2,
                               cost_usd=0.001, latency_ms=5, raw={"answers": answers, "usage": {"cost": 0.001}})


def entities() -> list[Entity]:
    return [Entity("1:2", "Brand A: TXT\nBrand B: Tomorrow X Together"),
            Entity("3:4", "Brand A: Panini\nBrand B: Topps")]


class HashTests(unittest.TestCase):
    def test_content_hash_is_byte_sensitive(self):
        self.assertNotEqual(content_hash("a"), content_hash("a "))

    def test_question_hash_ignores_key_order(self):
        reordered = {name: QUESTIONS[name] for name in reversed(list(QUESTIONS))}
        self.assertEqual(question_set_hash(QUESTIONS), question_set_hash(reordered))
        changed = json.loads(json.dumps(QUESTIONS))
        changed["alignment_score"]["instructions"] += "!"
        self.assertNotEqual(question_set_hash(QUESTIONS), question_set_hash(changed))

    def test_validation_rejects_bad_sets(self):
        with self.assertRaises(DecisionConfigError):
            validate_questions({"Bad-Name": QUESTIONS["alignment_score"]})
        with self.assertRaises(DecisionConfigError):
            validate_questions({f"q{i}": QUESTIONS["should_merge_brand_id"] for i in range(13)})
        self.assertEqual(score_range(QUESTIONS["alignment_score"]), (0.0, 3.0))


class NormalizationTests(unittest.TestCase):
    def test_jev_response_normalizes_to_typed_answers(self):
        payload = {"answers": {
            "relationship_type": {"type": "choice", "choice": "same_entity_alias",
                                  "probabilities": {"same_entity_alias": 0.97, "distinct_co_liked_peers": 0.03}},
            "should_merge_brand_id": {"type": "noul", "noul": 0.92},
            "alignment_score": {"type": "score", "score": 3.4},
        }}
        answers = normalize_jev_answers(payload, QUESTIONS)
        self.assertEqual(answers["relationship_type"], {"type": "choice", "choice": "same_entity_alias", "p": 0.97})
        self.assertEqual(answers["alignment_score"]["value"], 3.0)

    def test_openai_schema_mirrors_question_types(self):
        schema = openai_schema(QUESTIONS)
        self.assertEqual(schema["properties"]["relationship_type"]["properties"]["choice"]["enum"],
                         ["same_entity_alias", "distinct_co_liked_peers"])
        self.assertEqual(schema["properties"]["should_merge_brand_id"], {"type": "number"})
        self.assertEqual(schema["required"], list(QUESTIONS))


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry_path = Path(self.temp.name) / "brand_canonical.parquet"

    def tearDown(self):
        self.temp.cleanup()

    def test_second_run_hits_registry_and_replay_matches_live(self):
        backend = FakeBackend()
        live = DecisionClient(backend, Registry(self.registry_path, QUESTIONS)).decide(entities(), QUESTIONS)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(live.summary["registry_hits"], 0)
        self.assertEqual(live.summary["cost_usd"], 0.002)

        again = DecisionClient(backend, Registry(self.registry_path, QUESTIONS)).decide(entities(), QUESTIONS)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(again.summary["registry_hits"], 2)
        self.assertEqual(again.summary["api_calls"], 0)
        self.assertEqual([d.answers for d in again.decisions], [d.answers for d in live.decisions])
        self.assertEqual([d.decided_at for d in again.decisions], [d.decided_at for d in live.decisions])

        capture = Capture("unit", "live", "fake")
        capture.decision_batch("decide", live)
        capture_path = Path(self.temp.name) / "unit_live.json"
        capture.write(capture_path)

        empty_registry = Registry(Path(self.temp.name) / "fresh.parquet", QUESTIONS)
        replay = DecisionClient(ReplayBackend.from_captures([capture_path]), empty_registry).decide(entities(), QUESTIONS)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(replay.summary["api_calls"], 0)
        self.assertEqual(replay.summary["backend_calls"], 2)
        self.assertEqual([d.answers for d in replay.decisions], [d.answers for d in live.decisions])
        self.assertEqual([d.model_id for d in replay.decisions], [d.model_id for d in live.decisions])
        self.assertEqual(replay.summary["cost_usd"], live.summary["cost_usd"])
        self.assertEqual(len(empty_registry), 2)

        with self.assertRaises(ReplayMiss):
            DecisionClient(ReplayBackend({}), Registry(Path(self.temp.name) / "none.parquet", QUESTIONS)).decide(
                entities(), QUESTIONS)

        # A run served entirely from the registry still yields a capture that can replay on its own.
        hits_capture = Capture("unit", "live", "fake")
        hits_capture.decision_batch("decide", again)
        hits_path = Path(self.temp.name) / "unit_hits_live.json"
        hits_capture.write(hits_path)
        from_hits = DecisionClient(ReplayBackend.from_captures([hits_path]),
                                   Registry(Path(self.temp.name) / "fresh2.parquet", QUESTIONS)).decide(entities(), QUESTIONS)
        self.assertEqual([d.answers for d in from_hits.decisions], [d.answers for d in live.decisions])
        self.assertIsNone(from_hits.summary["cost_usd"])

    def test_registry_hit_after_fetch_keeps_fetched_record(self):
        backend = FakeBackend()
        registry = Registry(self.registry_path, QUESTIONS)
        fetched = DecisionClient(backend, registry).decide(entities()[:1], QUESTIONS)
        hit = DecisionClient(backend, registry).decide(entities(), QUESTIONS)
        capture = Capture("unit", "live", "fake")
        capture.decision_batch("first", fetched)
        capture.decision_batch("second", hit)
        records = list(capture.data["responses"].values())
        self.assertEqual(len(records), 2)
        kept = next(record for record in records if record["entity_key"] == "1:2")
        self.assertEqual(kept["source"], "backend")
        self.assertEqual(kept["cost_usd"], 0.001)
        self.assertIsNotNone(kept["response"])

    def test_refresh_bypasses_registry_and_content_change_misses(self):
        backend = FakeBackend()
        DecisionClient(backend, Registry(self.registry_path, QUESTIONS)).decide(entities(), QUESTIONS)
        refreshed = Registry(self.registry_path, QUESTIONS)
        DecisionClient(backend, refreshed, refresh=True).decide(entities(), QUESTIONS)
        self.assertEqual(backend.calls, 4)
        # Within one refresh run a state is decided once; a later batch reuses this run's answer.
        again = DecisionClient(backend, refreshed, refresh=True).decide(entities(), QUESTIONS)
        self.assertEqual(backend.calls, 4)
        self.assertEqual(again.summary["registry_hits"], 2)
        changed = [Entity("1:2", "Brand A: TXT\nBrand B: Tomorrow X Together (renamed)")]
        batch = DecisionClient(backend, Registry(self.registry_path, QUESTIONS)).decide(changed, QUESTIONS)
        self.assertEqual(batch.summary["registry_hits"], 0)
        self.assertEqual(backend.calls, 5)

    def test_engine_table_has_typed_answer_columns(self):
        registry = Registry(self.registry_path, QUESTIONS)
        batch = DecisionClient(FakeBackend(), registry).decide(entities(), QUESTIONS)
        table = registry.engine_table(batch.decisions, {"brand_id_a": [1, 3], "brand_id_b": [2, 4]})
        self.assertEqual(table.column_names, ["entity_key", "brand_id_a", "brand_id_b", "relationship_type",
                                              "relationship_type_p", "should_merge_brand_id", "alignment_score"])
        self.assertEqual(str(table.schema.field("should_merge_brand_id").type), "double")
        self.assertEqual(table.column("relationship_type").to_pylist(), ["same_entity_alias", "distinct_co_liked_peers"])
        with self.assertRaises(ValueError):
            registry.engine_table(batch.decisions, {"brand_id_a": [1]})

    def test_registry_rejects_other_question_set(self):
        registry = Registry(self.registry_path, QUESTIONS)
        DecisionClient(FakeBackend(), registry).decide(entities(), QUESTIONS)
        other = {"entity_kind": QUESTIONS["relationship_type"]}
        with self.assertRaises(ValueError):
            Registry(self.registry_path, other)


class PolicyTests(unittest.TestCase):
    def decision(self, choice: str, noul: float, key: str = "k") -> Decision:
        answers = {"relationship_type": {"type": "choice", "choice": choice, "p": 0.9},
                   "should_merge_brand_id": {"type": "noul", "value": noul},
                   "alignment_score": {"type": "score", "value": 1.0}}
        return Decision(key, "c", "q", "m", datetime(2026, 10, 19), answers, 1, 1, 0.0, 1, "backend")

    def test_bands_follow_action_noul_and_required_choice(self):
        policy = Policy({"action_question": "should_merge_brand_id", "accept_min": 0.70, "review_min": 0.30,
                         "require": {"relationship_type": ["same_entity_alias"]}})
        self.assertEqual(policy.band(self.decision("same_entity_alias", 0.78)), "accept")
        self.assertEqual(policy.band(self.decision("same_entity_alias", 0.44)), "review")
        self.assertEqual(policy.band(self.decision("same_entity_alias", 0.28)), "reject")
        self.assertEqual(policy.band(self.decision("distinct_co_liked_peers", 0.90)), "review")
        bands, counts = policy.apply([self.decision("same_entity_alias", 0.78, "a"), self.decision("x", 0.1, "b")])
        self.assertEqual(counts, {"accept": 1, "review": 0, "reject": 1})
        self.assertEqual(bands, {"a": "accept", "b": "reject"})


class CaptureTests(unittest.TestCase):
    def test_secret_scan_refuses_tokens(self):
        self.assertEqual(scan_secrets("Authorization: Bearer abc"), "Bearer literal")
        # Assembled at runtime so that no JWT-shaped literal is stored in the repository.
        jwt_shaped = "eyJ" + "a" * 12 + "." + "eyJ" + "b" * 12 + ".sig"
        self.assertEqual(scan_secrets(jwt_shaped), "JWT-shaped string")
        self.assertIsNone(scan_secrets('{"backend": "jev", "model_id": "typesafe/jev-1.13-20260917"}'))


if __name__ == "__main__":
    unittest.main()
