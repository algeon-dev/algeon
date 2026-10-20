#!/usr/bin/env python3
"""Case F1 (Purify gate): five EMBER2024 import graph variants and the model-audited DLL exclusion.

The cohort is the 3,225 Win32 files of the EMBER2024 challenge split (2,325
with a family label, 152 families, 68 with at least 2 labeled members). The
probe lists every DLL that has an import node in V1 with its footprint and its
8 most widely imported functions; the decision model judges whether shared
imports from that DLL are compiler or GUI boilerplate (`dll_noise`). Five graph
variants are built in SQL from `variant_edges.sql`:

    V0  imports only, doc frequency 3..100 (the published graph: 4,988 symbols, 80,673 edges)
    V1  V0 plus CAPA, MITRE and MBC behavior nodes, doc frequency 3..300
    V2  V1 with IDF edge weights LN(3225.0 / doc_freq) (non-model control)
    V3  V1 minus the DLLs of control_dll_list.json (analyst control)
    V4  V1 minus the DLLs in the accept band of `is_compiler_or_gui_noise`

Each variant runs five native statements: coverage (02), the 300 most similar
labeled pairs (03), the neighbors of 50 fixed seeds (04), the labeled pairs at
Jaccard 1.0 (05) and the published review queue
`07_challenge_import_review.sql` over the variant's tables. The gate passes
when V4's precision@300 exceeds V2's and V3's and both bootstrap intervals of
the paired differences exclude 0. The verdict is recorded either way, together
with the same rule applied to `exact_match_precision`, which does not decide
the gate; F2 is not part of this script.

Commands, from fixture/demo/decision_models/scripts (prepare once, then one GPU
run at a time under the shared lock):

    python3 cases/f_import_gate/prepare_tables.py
    eval "$(grep -E '^export (JEV_API_URL|JEV_API_KEY)=' ~/.bashrc)"
    flock /tmp/decision-demo-gpu.lock python3 cases/f_import_gate/run.py --mode live
    flock /tmp/decision-demo-gpu.lock python3 cases/f_import_gate/run.py --mode replay
    python3 cases/f_import_gate/compare_captures.py \\
        ../captures/f_import_gate_live.json ../captures/f_import_gate_replay.json

Decisions taken while implementing the plan:

- The probe returns the 112 DLLs that own at least one V1 import node. Counting
  DLL-level frequency over every import gives 116; the 4 extra DLLs have no
  symbol imported by 3..100 samples, so no variant can change through them.
- Coverage is a fourth statement per variant: per-sample retained node counts
  with labels and family sizes, plus the distinct node count. The 50 seeds are
  drawn from V0's coverage rows, so every variant ranks the same seeds.
- A seed with no retained node in a variant is absent from that graph; its 15
  neighbor slots count as misses.
- The review queue keeps the text of 07 and swaps five relation names: edges,
  symbols, sample counts and metadata for the variant's CTEs and
  `ember_f_samples`, and the seed table for a CTE that casts the Int32 seed id
  to the Int64 vertex type of the variant edges. For V2 both edge subqueries
  also pass the IDF `weight`, so the weighted variant changes the PageRank
  candidates and the Jaccard order. In V1..V4 `retained_import_count` counts
  behavior nodes too, and `example_shared_import` can name a behavior.
- More than 300 labeled pairs reach Jaccard 1.0 in every variant (538 in V0,
  about 800 in V1..V4), so 03 orders pairs tied on similarity by the number of
  retained nodes the two samples share, then by id. `exact_match_precision`
  covers the whole block at 1.0; its bound allows 1e-9 because IDF-weighted
  sums of identical node sets can fall one ulp below 1.0 in V2.
- The published 07 runs once unchanged as `review_published`; the capture
  records whether V0's queue matches it.
- Bootstrap intervals use 2,000 resamples, a fresh `random.Random(20261019)`
  per interval and the percentile bounds `sorted[50]` and `sorted[1949]`. Paired
  differences resample both variants' outcomes independently per iteration.
  Besides V4 - V2 and V4 - V3, the capture records V1 - V0 (the behavior nodes'
  contribution); it is not part of the gate.

EMBER2024 is a research dataset; the capture records it as `dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.algeon import DEMO_ROOT, statement_body  # noqa: E402
from client.case import Case, build_parser  # noqa: E402
from client.decision_client import Entity  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_ROOT / "decision_models/data/ember"
REVIEW_SQL = DEMO_ROOT / "ember2024/scripts/scale/07_challenge_import_review.sql"
CASE_TABLES = ("ember_f_samples", "ember_f_nodes", "ember_f_sample_nodes")
REGISTRY = "dll_noise"
DEFAULT_MAX_ENTITIES = 120
WIN32_SAMPLES = 3225
SEED = 20261019
RESAMPLES = 2000
SEED_COUNT = 50
NEIGHBORS = 15
PRECISION_PAIRS = 300
MIN_ACTIVE_NODES = 5
ALL_KINDS = ("import", "capa", "mitre", "mbc")
EXCLUDE_NONE = "ember_f_exclude_none"
EXCLUDE_CONTROL = "ember_f_exclude_control"
EXCLUDE_MODEL = "ember_f_exclude_model"
VARIANTS = {
    "V0": {"graph": "imports, doc frequency 3..100", "kinds": ("import",), "exclusion": EXCLUDE_NONE,
           "weighted": False},
    "V1": {"graph": "V0 plus CAPA, MITRE and MBC behaviors, doc frequency 3..300", "kinds": ALL_KINDS,
           "exclusion": EXCLUDE_NONE, "weighted": False},
    "V2": {"graph": f"V1 with IDF edge weights LN({WIN32_SAMPLES}.0 / doc_freq)", "kinds": ALL_KINDS,
           "exclusion": EXCLUDE_NONE, "weighted": True},
    "V3": {"graph": "V1 minus the hand-written CRT, GUI and COM DLL list", "kinds": ALL_KINDS,
           "exclusion": EXCLUDE_CONTROL, "weighted": False},
    "V4": {"graph": "V1 minus DLLs in the accept band of is_compiler_or_gui_noise", "kinds": ALL_KINDS,
           "exclusion": EXCLUDE_MODEL, "weighted": False},
}
DIFFERENCES = (("V4", "V2"), ("V4", "V3"), ("V1", "V0"))
DIFFERENCE_METRICS = ("precision_at_300", "exact_match_precision", "seed_family_rate")
GATE_DIFFERENCES = (("V4", "V2"), ("V4", "V3"))
# 07 relation name -> variant relation; the seed CTE casts the Int32 id to the Int64 vertex type.
REVIEW_RELATIONS = {
    "ember_48_challenge_import_edges": "variant_import_edges",
    "ember_48_challenge_import_symbols": "variant_import_symbols",
    "ember_48_challenge_import_sample_counts": "variant_sample_counts",
    "ember_metadata_challenge": "ember_f_samples",
    "ember_49_challenge_graph_seed": "variant_seed",
}
REVIEW_EDGE_PROJECTION = "SELECT src_id AS src, dst_id AS dst\n"
EXCLUSION_SCHEMA = pa.schema([("dll_lower", pa.string())])


def native_stage(case: Case, name: str, sql: str):
    """Run one statement and stop the case when it crossed into DataFusion on the host."""
    stage = case.run_stage(name, sql)
    boundaries = (stage.execution or {}).get("host_datafusion_boundary_count")
    if boundaries != 0:
        case.capture.note(f"stage {name} reported host_datafusion_boundary_count={boundaries}; "
                          f"EXPLAIN summary {stage.explain}")
        case.finish()
        raise SystemExit(f"stage {name} has host_datafusion_boundary_count={boundaries}; "
                         f"EXPLAIN summary {stage.explain}")
    return stage


def register_tables(case: Case) -> None:
    case.algeon.register_ember()
    for name in CASE_TABLES:
        path = DATA_DIR / f"{name}.parquet"
        if not path.exists():
            raise SystemExit(f"{path} is missing; run cases/f_import_gate/prepare_tables.py first")
        case.algeon.register_parquet(name, path)


def variant_edges(case: Case, variant: str) -> str:
    spec = VARIANTS[variant]
    weight = f", LN({WIN32_SAMPLES}.0 / CAST(n.doc_freq AS DOUBLE)) AS weight" if spec["weighted"] else ""
    return (statement_body(case.sql("variant_edges.sql"))
            .replace("{kinds}", ", ".join(f"'{kind}'" for kind in spec["kinds"]))
            .replace("{exclusion_table}", spec["exclusion"])
            .replace("{weight_column}", weight))


def variant_sql(case: Case, name: str, variant: str, **values) -> str:
    sql = (case.sql(name)
           .replace("{variant_edges}", variant_edges(case, variant))
           .replace("{edge_columns}", "src, dst, weight" if VARIANTS[variant]["weighted"] else "src, dst"))
    for key, value in values.items():
        sql = sql.replace("{" + key + "}", value)
    return sql


def review_sql(case: Case, variant: str) -> str:
    """07_challenge_import_review.sql over the variant's edges, symbols and sample counts."""
    body = statement_body(REVIEW_SQL.read_text(encoding="utf-8")).rstrip().rstrip(";")
    if not body.startswith("WITH candidates AS ("):
        raise SystemExit(f"{REVIEW_SQL} no longer starts with the candidates CTE")
    weighted = VARIANTS[variant]["weighted"]
    if weighted:
        if body.count(REVIEW_EDGE_PROJECTION) != 2:
            raise SystemExit(f"{REVIEW_SQL} no longer has the two edge projections the weighted variant extends")
        body = body.replace(REVIEW_EDGE_PROJECTION, "SELECT src_id AS src, dst_id AS dst, weight\n")
    for published, local in REVIEW_RELATIONS.items():
        body, count = re.subn(rf"\b{published}\b", local, body)
        if count == 0:
            raise SystemExit(f"{REVIEW_SQL} no longer reads {published}")
    prefix = "\n".join([
        f"WITH {variant_edges(case, variant)}, variant_import_edges AS (",
        f"  SELECT src AS src_id, dst AS dst_id{', weight' if weighted else ''} FROM variant_edges",
        "), variant_import_symbols AS (",
        "  SELECT node_id AS symbol_id, name AS symbol FROM ember_f_nodes",
        "), variant_sample_counts AS (",
        "  SELECT src AS id, COUNT(*) AS retained_import_count FROM variant_edges GROUP BY src",
        "), variant_seed AS (",
        "  SELECT CAST(id AS BIGINT) AS id FROM ember_49_challenge_graph_seed",
        "), ",
    ])
    return prefix + body[len("WITH "):]


def state(row: dict) -> str:
    functions = [row[f"symbol_{index}"].split("!", 1)[1] for index in range(1, 9) if row[f"symbol_{index}"]]
    return "\n".join([
        f"DLL: {row['dll_lower']}",
        f"Footprint among the {WIN32_SAMPLES:,} Win32 malware samples of the EMBER2024 challenge set, counting only "
        f"functions imported by 3 to 100 samples: {row['samples']:,} samples import {row['symbols']:,} distinct "
        f"function{'' if row['symbols'] == 1 else 's'} from this DLL ({row['edges']:,} sample-function pairs).",
        f"Most widely imported of those functions: {', '.join(functions)}",
    ])


def control_list(graph_dlls: set[str], all_dlls: set[str]) -> tuple[list[str], dict]:
    spec = json.loads((CASE_DIR / "control_dll_list.json").read_text(encoding="utf-8"))
    listed = list(spec["dlls"])
    prefix_matches = sorted(dll for dll in all_dlls if dll.startswith(tuple(spec["prefixes"])))
    entries = sorted(set(listed) | set(prefix_matches))
    return entries, {
        "listed": listed,
        "prefixes": spec["prefixes"],
        "prefix_matches": prefix_matches,
        "excluded": len(entries),
        "present": [dll for dll in entries if dll in graph_dlls],
        "absent": [dll for dll in entries if dll not in graph_dlls],
    }


def register_exclusion(case: Case, name: str, dlls: list[str]) -> None:
    table = pa.Table.from_pylist([{"dll_lower": dll} for dll in sorted(set(dlls))], schema=EXCLUSION_SCHEMA)
    case.algeon.register_parquet(name, DATA_DIR / f"{name}.parquet", table)


def percentile_interval(stats: list[float]) -> list[float]:
    ordered = sorted(stats)
    return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]]


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def bootstrap_mean(values: list[float]) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(SEED)
    return percentile_interval([mean(rng.choices(values, k=len(values))) for _ in range(RESAMPLES)])


def bootstrap_difference(left: list[float], right: list[float]) -> list[float] | None:
    if not left or not right:
        return None
    rng = random.Random(SEED)
    return percentile_interval([mean(rng.choices(left, k=len(left))) - mean(rng.choices(right, k=len(right)))
                                for _ in range(RESAMPLES)])


def difference_rows(samples: dict[str, list[float]]) -> list[dict]:
    return [{"difference": f"{left} - {right}", "value": mean(samples[left]) - mean(samples[right]),
             "ci95": bootstrap_difference(samples[left], samples[right])} for left, right in DIFFERENCES]


def verdict(metric: str, rows: list[dict]) -> dict:
    by_name = {row["difference"]: row for row in rows}
    gate_rows = [by_name[f"{left} - {right}"] for left, right in GATE_DIFFERENCES]
    wins = all(row["value"] > 0 for row in gate_rows)
    excludes_zero = all(row["ci95"] is not None and (row["ci95"][0] > 0 or row["ci95"][1] < 0) for row in gate_rows)
    return {"rule": f"V4 {metric} exceeds V2 and V3, and both difference intervals exclude 0",
            "v4_exceeds_v2_and_v3": wins, "intervals_exclude_zero": excludes_zero, "passed": wins and excludes_zero}


def coverage(rows: list[dict]) -> dict:
    return {
        "nodes": rows[0]["variant_nodes"] if rows else 0,
        "edges": sum(row["retained_nodes"] for row in rows),
        "samples_with_1_node": len(rows),
        "samples_with_5_nodes": sum(row["retained_nodes"] >= MIN_ACTIVE_NODES for row in rows),
        "labeled_active": sum(row["labeled"] and row["retained_nodes"] >= MIN_ACTIVE_NODES for row in rows),
    }


def draw_seeds(v0_rows: list[dict]) -> tuple[list[int], int]:
    pool = sorted(row["id"] for row in v0_rows
                  if row["labeled"] and row["family_members"] >= 2 and row["retained_nodes"] >= MIN_ACTIVE_NODES)
    return sorted(random.Random(SEED).sample(pool, SEED_COUNT)), len(pool)


def seed_hits(rows: list[dict], seeds: list[int], family_of: dict[int, str]) -> list[int]:
    neighbors: dict[int, list[dict]] = {seed: [] for seed in seeds}
    for row in rows:
        neighbors[row["seed_id"]].append(row)
    hits = []
    for seed in seeds:
        top = sorted(neighbors[seed], key=lambda row: (-row["similarity"], row["neighbor_id"]))[:NEIGHBORS]
        hits.append(sum(row["neighbor_family"] == family_of[seed] for row in top))
    return hits


def main() -> None:
    parser = build_parser(__doc__.splitlines()[0])
    args = parser.parse_args()
    case = Case(CASE_DIR, args)
    case.capture.data["dataset"] = "EMBER2024 challenge split"
    max_entities = args.max_entities or DEFAULT_MAX_ENTITIES

    register_tables(case)
    probe = native_stage(case, "probe", case.sql("01_probe.sql").replace("{max_entities}", str(max_entities)))
    dll_rows = probe.table.to_pylist()
    case.capture.result("probe", {"dlls": len(dll_rows), "max_entities": max_entities,
                                  "rows": [{key: row[key] for key in ("dll_lower", "edges", "samples", "symbols")}
                                           for row in dll_rows]})

    registry = case.registry(REGISTRY)
    entities = [Entity(row["dll_lower"], state(row)) for row in dll_rows]
    batch = case.decide(registry, entities)
    decisions = batch.decisions
    bands, band_counts = case.policy.apply(decisions)
    engine = registry.engine_table(decisions, {"dll_lower": [decision.entity_key for decision in decisions]})
    engine = engine.append_column("band", pa.array([bands[decision.entity_key] for decision in decisions]))
    case.decision_table(REGISTRY, engine)

    nodes = pq.read_table(DATA_DIR / "ember_f_nodes.parquet", columns=["kind", "dll_lower", "doc_freq"]).to_pylist()
    all_dlls = {node["dll_lower"] for node in nodes if node["kind"] == "import"}
    graph_dlls = {node["dll_lower"] for node in nodes if node["kind"] == "import" and 3 <= node["doc_freq"] <= 100}
    control_entries, control_record = control_list(graph_dlls, all_dlls)
    accepted = sorted(key for key, band in bands.items() if band == "accept")
    register_exclusion(case, EXCLUDE_NONE, [])
    register_exclusion(case, EXCLUDE_CONTROL, control_entries)
    register_exclusion(case, EXCLUDE_MODEL, accepted)
    control_set = set(control_entries)
    case.capture.result("bands", band_counts)
    case.capture.result("decisions", [
        {"dll_lower": decision.entity_key, "dll_role": decision.value("dll_role"),
         "dll_role_p": decision.probability("dll_role"),
         "is_compiler_or_gui_noise": decision.value("is_compiler_or_gui_noise"),
         "forensic_signal_score": decision.value("forensic_signal_score"),
         "band": bands[decision.entity_key], "in_control_list": decision.entity_key in control_set}
        for decision in sorted(decisions, key=lambda decision: decision.entity_key)])
    case.capture.result("review_queue", [
        {"dll_lower": decision.entity_key, "dll_role": decision.value("dll_role"),
         "is_compiler_or_gui_noise": decision.value("is_compiler_or_gui_noise"),
         "forensic_signal_score": decision.value("forensic_signal_score")}
        for decision in sorted(decisions, key=lambda decision: -decision.value("is_compiler_or_gui_noise"))
        if bands[decision.entity_key] == "review"])
    case.capture.result("dll_list_control", control_record)
    case.capture.result("model_vs_control", {
        "accepted": accepted,
        "accepted_in_control_list": [dll for dll in accepted if dll in control_set],
        "accepted_outside_control_list": [dll for dll in accepted if dll not in control_set],
        "control_present_not_accepted": [dll for dll in control_record["present"] if dll not in set(accepted)],
    })

    published = native_stage(case, "review_published", REVIEW_SQL.read_text(encoding="utf-8"))
    published_rows = published.table.to_pylist()

    variants, queues = [], {}
    samples: dict[str, dict[str, list[float]]] = {metric: {} for metric in DIFFERENCE_METRICS}
    outcomes, exact_outcomes, seed_rates = (samples[metric] for metric in DIFFERENCE_METRICS)
    seeds = family_of = None
    precision_pairs, seed_hit_counts = {}, {}
    for variant, spec in VARIANTS.items():
        rows = native_stage(case, f"{variant}_coverage", variant_sql(case, "02_coverage.sql", variant)).table.to_pylist()
        if seeds is None:
            seeds, pool = draw_seeds(rows)
            family_of = {row["id"]: row["family"] for row in rows}
            case.capture.result("seeds", {"pool": pool, "ids": seeds})
        pairs = native_stage(case, f"{variant}_precision_pairs",
                             variant_sql(case, "03_precision_pairs.sql", variant)).table.to_pylist()
        outcomes[variant] = [float(row["family_first"] == row["family_second"]) for row in pairs]
        precision_pairs[variant] = [
            {"first": row["first"], "second": row["second"], "similarity": row["similarity"],
             "shared_nodes": row["shared_nodes"], "fam_a": row["family_first"], "fam_b": row["family_second"],
             "same_family": row["family_first"] == row["family_second"]} for row in pairs]
        exact = native_stage(case, f"{variant}_exact_pairs",
                             variant_sql(case, "05_exact_pairs.sql", variant)).table.to_pylist()[0]
        exact_outcomes[variant] = [1.0] * exact["same_family"] + [0.0] * (exact["pairs"] - exact["same_family"])
        neighbors = native_stage(case, f"{variant}_seed_neighbors", variant_sql(
            case, "04_seed_neighbors.sql", variant, seed_ids=", ".join(map(str, seeds)))).table.to_pylist()
        hits = seed_hits(neighbors, seeds, family_of)
        seed_hit_counts[variant] = hits
        seed_rates[variant] = [hit / NEIGHBORS for hit in hits]
        queue = native_stage(case, f"{variant}_review", review_sql(case, variant)).table.to_pylist()
        queues[variant] = queue
        variants.append({
            "variant": variant,
            "graph": spec["graph"],
            "excluded_dlls": {EXCLUDE_NONE: 0, EXCLUDE_CONTROL: len(control_entries),
                              EXCLUDE_MODEL: len(accepted)}[spec["exclusion"]],
            **coverage(rows),
            "precision_pairs": len(pairs),
            "same_family_pairs": int(sum(outcomes[variant])),
            "precision_at_300": mean(outcomes[variant]),
            "precision_at_300_ci95": bootstrap_mean(outcomes[variant]),
            "similarity_at_300": pairs[-1]["similarity"] if pairs else None,
            "pairs_at_similarity_1": sum(row["similarity"] >= 0.999999999 for row in pairs),
            "shared_nodes_range": [min(row["shared_nodes"] for row in pairs),
                                   max(row["shared_nodes"] for row in pairs)] if pairs else None,
            "exact_match_pairs": exact["pairs"],
            "exact_match_same_family": exact["same_family"],
            "exact_match_precision": mean(exact_outcomes[variant]),
            "exact_match_precision_ci95": bootstrap_mean(exact_outcomes[variant]),
            "seeds_in_graph": len({row["seed_id"] for row in neighbors}),
            "seed_family_rate": mean(seed_rates[variant]),
            "seed_family_rate_ci95": bootstrap_mean(seed_rates[variant]),
            "review_rows": len(queue),
            "review_labeled": sum(bool(row["family"]) for row in queue),
        })

    differences = {metric: difference_rows(samples[metric]) for metric in DIFFERENCE_METRICS}
    case.capture.result("variants", variants)
    case.capture.result("differences", differences)
    case.capture.result("gate", {**verdict("precision_at_300", differences["precision_at_300"]),
                                 "exact_match_precision": verdict("exact_match_precision",
                                                                  differences["exact_match_precision"])})
    v0_queue = [(row["candidate_id"], round(row["similarity"], 12)) for row in queues["V0"]]
    case.capture.result("review_published", {
        "rows": published_rows,
        "v0_matches_published": v0_queue == [(row["candidate_id"], round(row["similarity"], 12))
                                             for row in published_rows],
    })
    case.capture.result("variant_review_queues", queues)
    case.capture.result("precision_pairs", precision_pairs)
    case.capture.result("seed_hits", seed_hit_counts)
    case.capture.result("bootstrap", {"seed": SEED, "resamples": RESAMPLES, "interval": "percentile sorted[50], sorted[1949]"})
    print(json.dumps({"bands": band_counts, "gate": case.capture.data["results"]["gate"],
                      "variants": [{key: row[key] for key in ("variant", "nodes", "edges", "labeled_active",
                                                              "precision_at_300", "exact_match_precision",
                                                              "seed_family_rate")}
                                   for row in variants]}, default=str), file=sys.stderr)
    case.finish()


if __name__ == "__main__":
    main()
