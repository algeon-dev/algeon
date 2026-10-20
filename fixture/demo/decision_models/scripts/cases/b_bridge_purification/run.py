"""Case B (Purify): remove non-brand hubs from the MerRec brand graph before Jaccard, Leiden and betweenness.

The probe ranks the top 120 bridge brands of `31_world_bridges.sql` for one
time window, the decision model classifies each entry into `brand_ontology`,
and the recompute rebuilds the graph without the entries outside the `accept`
band. Two controls run beside it: the original ranking filtered after the fact
to accepted entries, and the same recompute driven by the hand-written
`control_denylist.json`.

Run from `fixture/demo/decision_models/scripts`, one GPU run at a time, with a
distinct `--label` per window (it names the capture and the engine tables):

    eval "$(grep -E '^export (JEV_API_URL|JEV_API_KEY)=' ~/.bashrc)"
    flock /tmp/decision-demo-gpu.lock python3 cases/b_bridge_purification/run.py --mode live --label all --passes 2
    flock /tmp/decision-demo-gpu.lock python3 cases/b_bridge_purification/run.py --mode live --label sep_oct --passes 2 \\
        --window-start 2023-09-01 --window-end 2023-11-01

Replay serves every decision from the registry or the live capture with zero
API calls; pass the same window flags as the live run:

    flock /tmp/decision-demo-gpu.lock python3 cases/b_bridge_purification/run.py --mode replay --label all --passes 2
    flock /tmp/decision-demo-gpu.lock python3 cases/b_bridge_purification/run.py --mode replay --label sep_oct --passes 2 \\
        --window-start 2023-09-01 --window-end 2023-11-01
    python3 cases/b_bridge_purification/compare_captures.py all sep_oct

`--passes N` repeats the loop: the recompute of pass k-1 is the probe of
pass k (the same `03_recompute.sql` statement over all decisions so far), its
top 120 entries not decided yet go to the backend, and the engine table holds
the bands of every pass. Pass the same `--passes` to the replay. The denylist
control uses pass 1; the review queue, band counts and kind counts cover
every pass. `--with-canonical` adds the
recompute through case A's `registry/brand_canonical.engine.parquet` when that
file exists.

The entity state holds only window-invariant catalog context (name, footprint,
top categories, sample titles). Graph metrics stay out of it so that the same
brand hashes to the same registry key in every window, and a new window calls
the backend only for brands no earlier window has decided.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.capture import merrec_license  # noqa: E402
from client.case import Case, build_parser  # noqa: E402
from client.decision_client import Entity  # noqa: E402
from client.policy import BANDS  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_TABLE = "brand_ontology"
DENYLIST_TABLE = "brand_denylist"
CANONICAL_TABLE = "brand_canonical"
NATIVE_STAGES = ("probe", "recompute", "recompute_denylist", "recompute_canonical")


def parse_args():
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--window-start", default="2023-05-01", help="inclusive event_time lower bound")
    parser.add_argument("--window-end", default="2023-11-01", help="exclusive event_time upper bound")
    parser.add_argument("--passes", type=int, default=1,
                        help="probe, decide and recompute rounds; pass k probes the graph purified by passes before it")
    parser.add_argument("--with-canonical", action="store_true",
                        help="also recompute through case A's brand_canonical table when it exists")
    args = parser.parse_args()
    try:
        start, end = datetime.fromisoformat(args.window_start), datetime.fromisoformat(args.window_end)
    except ValueError as error:
        parser.error(f"window bounds must be ISO dates or timestamps: {error}")
    if args.passes < 1:
        parser.error("--passes must be at least 1")
    if start >= end:
        parser.error("--window-start must be before --window-end")
    return args


def render(sql: str, **values: str) -> str:
    for name, value in values.items():
        sql = sql.replace("{" + name + "}", value)
    return sql


def require_native(stage) -> None:
    boundaries = (stage.execution or {}).get("host_datafusion_boundary_count")
    if boundaries != 0:
        print(json.dumps({"stage": stage.name, "host_datafusion_boundary_count": boundaries,
                          "explain": stage.explain, "execution": stage.execution}), file=sys.stderr)
        raise SystemExit(f"stage {stage.name} did not run fully native; see the EXPLAIN summary above")


def check_replay_run(case: Case, window: dict) -> None:
    live = case.capture_path("live")
    if case.args.mode != "replay" or not live.exists():
        return
    with open(live, encoding="utf-8") as handle:
        results = json.load(handle).get("results") or {}
    recorded = results.get("window")
    if recorded and recorded != window:
        raise SystemExit(f"live capture {live.name} used window {recorded}; pass the same --window-start/--window-end")
    passes = results.get("passes_requested", 1)
    if passes != case.args.passes:
        raise SystemExit(f"live capture {live.name} requested {passes} pass(es); pass --passes {passes}")


def state_text(row: dict, context: dict) -> str:
    categories = [f"{context[f'cat{i}']} ({context[f'cat{i}_listings']})" for i in range(1, 4) if context.get(f"cat{i}")]
    titles = [context[f"title{i}"] for i in range(1, 5) if context.get(f"title{i}")]
    avg_price = context.get("avg_price")
    return (
        f"Brand Entry Name: {row['brand_name']} (brand_id={row['brand_id']})\n"
        f"Catalog Footprint: {context.get('total_listings', 0)} listings across "
        f"{context.get('distinct_c0', 0)} top-level categories and {context.get('distinct_c1', 0)} subcategories "
        f"(avg price=${0.0 if avg_price is None else avg_price:.2f})\n"
        f"Top Categories: {'; '.join(categories)}\n"
        f"Sample Item Titles: {' | '.join(titles)}"
    )


def ranks(rows: list[dict]) -> dict[int, int]:
    return {row["brand_id"]: rank for rank, row in enumerate(rows, 1)}


def ranking_rows(original: list[dict], purified: list[dict], denied: list[dict], decisions: dict, bands: dict) -> list:
    original_rank = ranks(original)
    postfilter_rank = ranks([row for row in original if bands[str(row["brand_id"])] == "accept"])
    purified_rank = ranks(purified)
    denylist_rank = ranks(denied)
    original_by_id = {row["brand_id"]: row for row in original}
    purified_by_id = {row["brand_id"]: row for row in purified}
    names = {row["brand_id"]: row["brand_name"] for row in (*denied, *purified, *original)}
    table = []
    for brand_id in names:
        decision = decisions.get(str(brand_id))
        before = original_by_id.get(brand_id, {}).get("betweenness")
        after = purified_by_id.get(brand_id, {}).get("betweenness")
        table.append({
            "brand_id": brand_id,
            "name": names[brand_id],
            "band": bands.get(str(brand_id)),
            "entity_kind": decision.value("entity_kind") if decision else None,
            "is_standalone_brand": decision.value("is_standalone_brand") if decision else None,
            "wormhole_pollution": decision.value("wormhole_pollution") if decision else None,
            "original_rank": original_rank.get(brand_id),
            "original_betweenness": before,
            "postfilter_rank": postfilter_rank.get(brand_id),
            "purified_rank": purified_rank.get(brand_id),
            "purified_betweenness": after,
            "betweenness_multiplier": after / before if before and after is not None else None,
            "denylist_rank": denylist_rank.get(brand_id),
        })
    unranked = float("inf")
    table.sort(key=lambda row: tuple(unranked if row[key] is None else row[key]
                                     for key in ("purified_rank", "original_rank", "denylist_rank")))
    return table


def context_summary(context: dict) -> dict:
    return {key: context.get(key) for key in (
        "total_listings", "distinct_c0", "distinct_c1", "avg_price", "cat1", "cat1_listings")}


def band_counts(bands: dict, keys) -> dict[str, int]:
    counts = {band: 0 for band in BANDS}
    for key in keys:
        counts[bands[key]] += 1
    return counts


def main() -> None:
    args = parse_args()
    window = {"start": args.window_start, "end": args.window_end}
    case = Case(CASE_DIR, args, license=merrec_license())
    check_replay_run(case, window)
    case.capture.result("window", window)
    case.capture.result("passes_requested", args.passes)
    case.algeon.register_merrec()
    suffix = f"_{args.label}" if args.label else ""
    window_values = {"window_start": args.window_start, "window_end": args.window_end}
    recompute_sql = case.sql("03_recompute.sql")
    registry = case.registry(ONTOLOGY_TABLE)

    probe = case.run_stage("probe", render(case.sql("01_probe.sql"), **window_values))
    require_native(probe)
    original = probe.table.to_pylist()[:args.max_entities]
    candidates = original
    decisions = {}
    bands: dict[str, str] = {}
    ids_in_order: list[int] = []
    judged_in: dict[int, tuple[int, int]] = {}
    contexts_all: dict = {}
    names_all: dict[int, str] = {}
    passes = []
    first_bands: dict[str, str] = {}
    denied: list[dict] = []
    for number in range(1, args.passes + 1):
        tag = "" if number == 1 else f"_pass{number}"
        ids = [row["brand_id"] for row in candidates]
        new_ids = [brand_id for brand_id in ids if str(brand_id) not in decisions]
        if not new_ids:
            case.capture.note(f"pass {number}: every probe entry was decided in an earlier pass; stopped")
            break
        context_stage = case.run_stage(f"context{tag}", render(
            case.sql("02_context.sql"), brand_ids=", ".join(str(brand_id) for brand_id in ids)))
        contexts = {row["brand_id"]: row for row in context_stage.table.to_pylist()}
        contexts_all.update(contexts)
        names_all.update({row["brand_id"]: row["brand_name"] for row in candidates})
        probe_rank = ranks(candidates)
        entities = [Entity(str(row["brand_id"]), state_text(row, contexts.get(row["brand_id"], {})))
                    for row in candidates]
        batch = case.decide(registry, entities, name=f"decide{tag}")
        pass_bands, _ = case.policy.apply(batch.decisions)
        for decision in batch.decisions:
            if decision.entity_key not in decisions:
                ids_in_order.append(int(decision.entity_key))
                judged_in[int(decision.entity_key)] = (number, probe_rank[int(decision.entity_key)])
            decisions[decision.entity_key] = decision
        bands.update(pass_bands)

        union = [decisions[str(brand_id)] for brand_id in ids_in_order]
        ontology = registry.engine_table(union, {"brand_id": pa.array(ids_in_order, pa.int64())})
        ontology = ontology.append_column(
            "band", pa.array([bands[str(brand_id)] for brand_id in ids_in_order], pa.string()))
        case.decision_table(ONTOLOGY_TABLE, ontology, f"{ONTOLOGY_TABLE}{suffix}.engine.parquet")

        purified_stage = case.run_stage(f"recompute{tag}", render(
            recompute_sql, ontology_table=ONTOLOGY_TABLE, **window_values))
        require_native(purified_stage)
        purified = purified_stage.table.to_pylist()

        if number == 1:
            first_bands = pass_bands
            denylist_spec = case.load_json("control_denylist.json")
            denylist = {name for names in denylist_spec.values() for name in names}
            denylist_table = pa.table({
                "brand_id": pa.array(ids, pa.int64()),
                "band": pa.array(["reject" if row["brand_name"] in denylist else "accept" for row in original],
                                 pa.string()),
            })
            case.decision_table(DENYLIST_TABLE, denylist_table, f"{DENYLIST_TABLE}{suffix}.engine.parquet")
            denied_stage = case.run_stage("recompute_denylist",
                                          render(recompute_sql, ontology_table=DENYLIST_TABLE, **window_values))
            require_native(denied_stage)
            denied = denied_stage.table.to_pylist()

        passes.append({
            "pass": number,
            "probe": [{"rank": rank, "brand_id": row["brand_id"], "name": row["brand_name"],
                       "betweenness": row["betweenness"]} for rank, row in enumerate(candidates, 1)],
            "new_entries": len(new_ids),
            "new_band_counts": band_counts(bands, (str(brand_id) for brand_id in new_ids)),
            "registry_hits": batch.summary["registry_hits"],
            "api_calls": batch.summary["api_calls"],
            "ranking": ranking_rows(original, purified, denied, decisions, bands),
        })
        candidates = purified[:args.max_entities]

    canonical_path = args.registry_dir / f"{CANONICAL_TABLE}.engine.parquet"
    if args.with_canonical and canonical_path.exists():
        case.algeon.register_parquet(CANONICAL_TABLE, canonical_path)
        canonical_stage = case.run_stage("recompute_canonical", render(
            case.sql("04_recompute_canonical.sql"), ontology_table=ONTOLOGY_TABLE, **window_values))
        require_native(canonical_stage)
        case.capture.result("canonical_ranking", [
            {"canonical_rank": rank, "brand_id": row["brand_id"], "name": row["brand_name"],
             "betweenness": row["betweenness"], "neighbors": row["neighbors"], "worlds_reached": row["worlds_reached"]}
            for rank, row in enumerate(canonical_stage.table.to_pylist(), 1)])
    elif args.with_canonical:
        case.capture.note(f"canonical recompute skipped: {canonical_path.name} does not exist (case A has not run)")
    else:
        case.capture.note("canonical recompute not requested (--with-canonical)")

    names = {str(row["brand_id"]): row["brand_name"] for row in original}
    original_rank = ranks(original)
    model_excluded = {names[key] for key, band in first_bands.items() if band != "accept"}
    denylist_excluded = {row["brand_name"] for row in original if row["brand_name"] in denylist}
    case.capture.result("ranking", passes[-1]["ranking"])
    case.capture.result("passes", passes)
    case.capture.result("band_counts", band_counts(bands, bands))
    case.capture.result("kind_counts", dict(sorted(Counter(
        decision.value("entity_kind") for decision in decisions.values()).items())))
    case.capture.result("review_queue", [
        {"brand_id": brand_id, "name": names_all[brand_id], "pass": judged_in[brand_id][0],
         "probe_rank": judged_in[brand_id][1], "original_rank": original_rank.get(brand_id),
         "entity_kind": decisions[str(brand_id)].value("entity_kind"),
         "entity_kind_p": decisions[str(brand_id)].probability("entity_kind"),
         "is_standalone_brand": decisions[str(brand_id)].value("is_standalone_brand"),
         "wormhole_pollution": decisions[str(brand_id)].value("wormhole_pollution"),
         "context": context_summary(contexts_all.get(brand_id, {}))}
        for brand_id in ids_in_order if bands[str(brand_id)] == "review"])
    case.capture.result("denylist_vs_model", {
        "denylist_size": len(denylist),
        "denylist_matched": len(denylist_excluded),
        "denylist_unmatched": sorted(denylist - denylist_excluded),
        "model_excluded": len(model_excluded),
        "overlap": len(model_excluded & denylist_excluded),
        "only_denylist": sorted(denylist_excluded - model_excluded),
        "only_model": sorted(model_excluded - denylist_excluded),
    })
    case.capture.result("decisions", [
        {"brand_id": brand_id, "name": names_all[brand_id], "pass": judged_in[brand_id][0],
         "band": bands[str(brand_id)], "content_hash": decisions[str(brand_id)].content_hash,
         "model_id": decisions[str(brand_id)].model_id, "answers": decisions[str(brand_id)].answers}
        for brand_id in ids_in_order])
    case.finish()


if __name__ == "__main__":
    main()
