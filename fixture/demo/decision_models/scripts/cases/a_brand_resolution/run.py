#!/usr/bin/env python3
"""Case A (Resolve): MerRec brand entity resolution.

The probe projects the top 2,000 standalone brands (case B's all-window
`brand_ontology_all.engine.parquet` removes entries whose band is not `accept`) on the like bipartite graph and
keeps the pairs with the highest fan Jaccard. The decision model judges each
pair; accepted alias pairs feed weakly connected components, which writes
`registry/brand_canonical.engine.parquet` for case B, and the dedup statement
recomputes exact fan unions over every like. A trigram name-similarity control
runs over all brand names.

Commands, from `fixture/demo/decision_models/scripts`; every GPU run holds the
shared lock, and a live run needs `JEV_API_URL` and `JEV_API_KEY` in the
environment:

    flock /tmp/decision-demo-gpu.lock python3 cases/a_brand_resolution/run.py --mode live
    flock /tmp/decision-demo-gpu.lock python3 cases/a_brand_resolution/run.py --mode replay
    python3 cases/a_brand_resolution/compare_captures.py \\
        ../captures/a_brand_resolution_live.json ../captures/a_brand_resolution_replay.json

A replay with the shared registry serves every decision from the registry and
falls back to the live capture for anything the registry lacks; the comparator
checks the answers, not the hit counters.

Case flags: `--ontology` (case B's decision table, default
`<registry-dir>/brand_ontology_all.engine.parquet`).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from client.capture import merrec_license  # noqa: E402
from client.case import Case, build_parser  # noqa: E402
from client.decision_client import Entity  # noqa: E402
import control_trigram  # noqa: E402

DEFAULT_MAX_ENTITIES = 120
GPU_STAGES = ("probe", "canonical", "dedup")
CANONICAL_SCHEMA = pa.schema([("entity_key", pa.string()), ("brand_id", pa.int64()), ("canonical_id", pa.int64())])


def require_native(case: Case, stage) -> None:
    """Stop when a recompute-path stage crossed into DataFusion on the host."""
    boundaries = (stage.execution or {}).get("host_datafusion_boundary_count")
    if boundaries != 0:
        case.capture.note(f"stage {stage.name} reported host_datafusion_boundary_count={boundaries}; "
                          f"EXPLAIN summary {stage.explain}")
        case.finish()
        raise SystemExit(f"stage {stage.name} has host_datafusion_boundary_count={boundaries}; "
                         f"EXPLAIN summary {stage.explain}")


def brand_text(brand_id: int, name: str, fans: int, context: dict) -> str:
    titles = [context[f"sample_{i}"] for i in (1, 2, 3) if context.get(f"sample_{i}")]
    return (f"{name} (brand_id={brand_id}, fans={fans}, listings={context['listings']}, "
            f"avg_price=${context['avg_price']:.2f})\n"
            f"  Top Categories: {context['top_category']} | {context['second_category']}\n"
            f"  Sample Titles: {' // '.join(titles)}")


def pair_state(pair: dict, context: dict[int, dict]) -> str:
    return (f"Behavioral Fan Overlap: Jaccard Similarity = {pair['jaccard']:.4f}\n"
            f"Brand A: {brand_text(pair['brand_id_a'], pair['brand_a'], pair['fans_a'], context[pair['brand_id_a']])}\n"
            f"Brand B: {brand_text(pair['brand_id_b'], pair['brand_b'], pair['fans_b'], context[pair['brand_id_b']])}")


def ontology_summary(path: Path) -> dict:
    table = pq.read_table(path, columns=["brand_id", "band"])
    excluded = table.filter(pc.not_equal(table["band"], "accept"))
    return {"rows": table.num_rows, "excluded_brand_ids": len(set(excluded["brand_id"].to_pylist()))}


def main() -> None:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--ontology", type=Path, default=None,
                        help="case B decision table; default <registry-dir>/brand_ontology_all.engine.parquet")
    args = parser.parse_args()
    max_entities = args.max_entities or DEFAULT_MAX_ENTITIES
    ontology = args.ontology or args.registry_dir / "brand_ontology_all.engine.parquet"
    if not ontology.exists():
        raise SystemExit(f"{ontology} is missing; run case B (b_bridge_purification) first")

    case = Case(CASE_DIR, args, license=merrec_license())
    case.capture.data["max_entities"] = max_entities
    case.algeon.register_merrec()
    case.algeon.register_parquet("brand_ontology", ontology)
    case.capture.result("ontology", ontology_summary(ontology))

    probe = case.run_stage("probe", case.sql("01_probe.sql").replace("{max_entities}", str(max_entities)))
    require_native(case, probe)
    pairs = probe.table.to_pylist()
    brand_ids = sorted({pair["brand_id_a"] for pair in pairs} | {pair["brand_id_b"] for pair in pairs})
    context_stage = case.run_stage(
        "context", case.sql("02_context.sql").replace("{brand_ids}", ", ".join(str(b) for b in brand_ids)))
    context = {row["brand_id"]: row for row in context_stage.table.to_pylist()}

    entities = [Entity(f"{pair['brand_id_a']}:{pair['brand_id_b']}", pair_state(pair, context)) for pair in pairs]
    registry = case.registry("brand_pairs")
    batch = case.decide(registry, entities)
    by_key = batch.by_key()
    decisions = [by_key[entity.key] for entity in entities]
    bands, band_counts = case.policy.apply(decisions)

    engine = registry.engine_table(decisions, {
        "brand_id_a": pa.array([pair["brand_id_a"] for pair in pairs], pa.int64()),
        "brand_id_b": pa.array([pair["brand_id_b"] for pair in pairs], pa.int64()),
    })
    engine = engine.append_column("band", pa.array([bands[entity.key] for entity in entities], pa.string()))
    case.decision_table("brand_pairs", engine)

    records = []
    for rank, (pair, decision) in enumerate(zip(pairs, decisions), start=1):
        records.append({
            "rank": rank, **pair,
            "relationship_type": decision.value("relationship_type"),
            "relationship_type_p": decision.probability("relationship_type"),
            "should_merge_brand_id": decision.value("should_merge_brand_id"),
            "alignment_score": decision.value("alignment_score"),
            "band": bands[decision.entity_key],
        })

    accepted = [record for record in records if record["band"] == "accept"]
    clusters = []
    if accepted:
        canonical = case.run_stage("canonical", case.sql("03a_canonical.sql"))
        require_native(case, canonical)
        rows = canonical.table.to_pylist()
        canonical_table = pa.table({
            "entity_key": [str(row["brand_id"]) for row in rows],
            "brand_id": pa.array([row["brand_id"] for row in rows], pa.int64()),
            "canonical_id": pa.array([row["canonical_id"] for row in rows], pa.int64()),
        }, schema=CANONICAL_SCHEMA)
        case.decision_table("brand_canonical", canonical_table)

        dedup = case.run_stage("dedup", case.sql("03_recompute.sql"))
        require_native(case, dedup)
        names = {}
        for record in records:
            names[record["brand_id_a"]] = (record["brand_a"], record["fans_a"])
            names[record["brand_id_b"]] = (record["brand_b"], record["fans_b"])
        members: dict[int, list[int]] = {}
        for row in rows:
            members.setdefault(row["canonical_id"], []).append(row["brand_id"])
        for row in dedup.table.to_pylist():
            ordered = sorted(members[row["canonical_id"]], key=lambda brand_id: (-names[brand_id][1], brand_id))
            clusters.append({**row, "member_ids": ordered, "member_names": [names[brand_id][0] for brand_id in ordered]})
    else:
        case.decision_table("brand_canonical", CANONICAL_SCHEMA.empty_table())
        case.capture.note("no pair reached the accept band; brand_canonical is empty and the dedup stage did not run")

    names_stage = case.run_stage("brand_names", control_trigram.NAMES_SQL)
    control = control_trigram.run_control(names_stage.table.to_pylist(), records)

    summary = batch.summary
    brands = names_stage.rows
    all_pairs = brands * (brands - 1) // 2
    per_decision = summary["cost_usd"] / summary["backend_calls"] \
        if summary["backend_calls"] and summary["cost_usd"] is not None else None
    tokens_per_decision = summary["input_tokens"] / summary["backend_calls"] if summary["backend_calls"] else None
    case.capture.result("full_pairwise_cost_estimate", {
        "brands": brands,
        "pairs": all_pairs,
        "probe_decisions": len(records),
        "pairs_per_probe_decision": all_pairs / len(records) if records else None,
        "mean_cost_usd_per_decision": per_decision,
        "mean_input_tokens_per_decision": tokens_per_decision,
        "estimated_cost_usd": per_decision * all_pairs if per_decision is not None else None,
        "estimated_input_tokens": tokens_per_decision * all_pairs if tokens_per_decision is not None else None,
        "basis": "this run's backend calls; not executed",
    })

    case.capture.result("pairs", records)
    case.capture.result("band_counts", band_counts)
    case.capture.result("relationship_counts", dict(sorted(Counter(r["relationship_type"] for r in records).items())))
    case.capture.result("accepted_pairs", accepted)
    case.capture.result("review_queue", sorted((r for r in records if r["band"] == "review"),
                                               key=lambda r: (-r["should_merge_brand_id"], r["rank"])))
    case.capture.result("clusters", clusters)
    case.capture.result("trigram_control", control)
    case.capture.result("gpu_stages", {stage["name"]: {
        "elapsed_s": stage["elapsed_s"],
        "host_datafusion_boundary_count": (stage["execution"] or {}).get("host_datafusion_boundary_count"),
    } for stage in case.capture.data["stages"] if stage["name"] in GPU_STAGES})
    case.finish()


if __name__ == "__main__":
    main()
