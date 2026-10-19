#!/usr/bin/env python3
"""Case D (Explain): which mechanisms carry the Barbie affinity shift of the top MerRec brands.

The probe is the published lift query `10_barbie_tie_ins.sql`. For the probe's
top 10 brands, `02_bridge_items.sql` returns the 25 items Barbie fans liked
most in July and August 2023; the decision model labels each item's mechanism,
whether it is an official Barbie license, and whether it carries a pink color
signal. `03_counterfactual.sql` recomputes the lift with the like events of an
exclusion table removed from the graph, four times: an empty table (baseline),
every non-Barbie item whose title contains 'barbie' (keyword control), the
bridge items in the accept band (model), and every bridge item whose
`mechanism` is `official_barbie_collab` regardless of band (model_mechanism).
The model_mechanism run exists to show the gap between the choice head and the
action noul on this question: the choice labels most audited collab items
official while `official_license` leaves most of them in the review band. The
policy is unchanged; `official_license_by_band` records the license
probabilities of the official-labelled items per band. `04_decomposition.sql`
splits each brand's audited Barbie-fan likes by mechanism, band and keyword.

Commands, from fixture/demo/decision_models/scripts:

    eval "$(grep -E '^export (JEV_API_URL|JEV_API_KEY)=' ~/.bashrc)"
    flock /tmp/decision-demo-gpu.lock python3 cases/d_barbie_mechanisms/run.py --mode live
    flock /tmp/decision-demo-gpu.lock python3 cases/d_barbie_mechanisms/run.py --mode replay
    python3 cases/d_barbie_mechanisms/compare_captures.py \\
        ../captures/d_barbie_mechanisms_live.json ../captures/d_barbie_mechanisms_replay.json

Decisions the specification left open:

- `01_probe.sql` is a copy of the published query; the run stops when its
  statement differs from `merrec/scripts/scale/10_barbie_tie_ins.sql`. The
  published output has no `brand_id`, so the brand ids come from the baseline
  counterfactual, whose first 25 rows must equal the probe's rows exactly
  (brand name, fan counts and the three lift values).
- `03_counterfactual.sql` adds `brand_id` and drops `LIMIT 25`, so every probe
  brand keeps a row when an exclusion reorders the ranking. The fan thresholds
  (1,000 before and after) stay, so a brand whose fans fall below them under an
  exclusion reports a null counterfactual lift.
- The keyword control excludes items whose title contains 'barbie' and whose
  brand is not Barbie (2595), as in the earlier counterfactual study. Removing
  the Barbie brand's own items would change the seed's neighborhood and measure
  a different graph. The two model exclusions touch only audited bridge items,
  none of which belong to brand 2595.
- `excess_retained_pct = 100 * (cf_lift_change - 1) / (base_lift_change - 1)`;
  it is null when the base lift change is 1 or the counterfactual row is missing.
- Likes count `item_like` events; Barbie-fan users count distinct fans. The
  second admission rule ("a title containing 'barbie' with at least 3 likes")
  counts all likes, so keyword items that Barbie fans ignored stay visible.
- `--max-entities` (default 250) caps the bridge rows taken in probe rank, then
  item rank.
- `keyword_barbie` in the decision table is `'barbie' in title.lower()` in
  Python. The bridge SQL computes the same flag with `LOWER(name) LIKE`; the
  capture records how many rows disagree (`keyword_flag_mismatches`).
- "The model calls an item official" means the accept band (official_license
  >= 0.70 with mechanism official_barbie_collab). A pink item is one with
  `pink_aesthetic >= 0.70`.
- The behavioral signal for the reject examples is the item's Barbie-fan share
  of its July to August likes. A reject-band item counts as a behavioral false
  collab when that share reaches the median share of the accept-band items.
- The bridge rows and the exclusion tables are written under `data/barbie/`
  (local only, CC BY-NC like the other MerRec-derived files).
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.algeon import DEMO_ROOT, statement_body  # noqa: E402
from client.capture import merrec_license  # noqa: E402
from client.case import DEMO_DIR, Case, build_parser  # noqa: E402
from client.decision_client import Entity  # noqa: E402
from client.policy import BANDS  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent
PUBLISHED_PROBE = DEMO_ROOT / "merrec/scripts/scale/10_barbie_tie_ins.sql"
DATA_DIR = DEMO_DIR / "data/barbie"
REGISTRY = "barbie_items"
BRIDGE_TABLE = "barbie_bridge_items"
BARBIE_BRAND_ID = 2595
AUDITED_BRANDS = 10
PINK_MIN = 0.70
EXAMPLES = 5
KEYWORD_ITEMS_SQL = (
    f"SELECT item_id FROM merrec_items WHERE LOWER(name) LIKE '%barbie%' AND brand_id <> {BARBIE_BRAND_ID}")
COUNTERFACTUALS = ("keyword", "model", "model_mechanism")
OFFICIAL = "official_barbie_collab"
PROBE_COLUMNS = ("brand_name", "fans_before", "fans_after", "lift_before", "lift_after", "lift_change")
# Figures of the earlier keyword counterfactual on the same query: expectations, never inputs.
PRIOR = {
    "NYX": {"base_lift_change": 2.133, "keyword_lift_change": 1.075, "keyword_retained_pct": 6.6},
    "Stanley": {"base_lift_change": 1.396, "keyword_lift_change": 1.417, "keyword_retained_pct": 105.3},
}


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


def render(sql: str, **values: str) -> str:
    for name, value in values.items():
        sql = sql.replace("{" + name + "}", value)
    return sql


def item_id_table(item_ids) -> pa.Table:
    return pa.table({"item_id": pa.array(sorted(set(item_ids)), pa.int64())})


def counterfactual(case: Case, label: str, item_ids) -> list[dict]:
    table_name = f"barbie_exclude_{label}"
    case.algeon.register_parquet(table_name, DATA_DIR / f"{table_name}.parquet", item_id_table(item_ids))
    stage = native_stage(case, f"counterfactual_{label}",
                         render(case.sql("03_counterfactual.sql"), excluded_items=table_name))
    return stage.table.to_pylist()


def state_text(row: dict) -> str:
    price = row["price"]
    return "\n".join([
        f"Brand: {row['brand_name']} (brand_id={row['brand_id']})",
        f"Item Title: {row['item_name'] or ''}",
        f"Category: {row['category_path'] or 'unknown'}",
        f"Color: {row['color'] or 'not given'}",
        f"Price: {'unknown' if price is None else f'${price:.2f}'}",
        f"Likes in July and August 2023: {row['likes_after']} in total, {row['fan_likes_after']} from Barbie fans "
        f"(users who liked a Barbie-brand item between May and August 2023)",
    ])


def keyword(row: dict) -> bool:
    return "barbie" in (row["item_name"] or "").lower()


def retained(base: float | None, cf: float | None) -> float | None:
    if base is None or cf is None or base == 1:
        return None
    return 100 * (cf - 1) / (base - 1)


def counterfactual_table(probe: list[dict], ids: list[int], runs: dict[str, list[dict]]) -> list[dict]:
    by_label = {label: {row["brand_id"]: row for row in rows} for label, rows in runs.items()}
    ranks = {label: {row["brand_id"]: rank for rank, row in enumerate(rows, 1)} for label, rows in runs.items()}
    table = []
    for rank, (row, brand_id) in enumerate(zip(probe, ids), 1):
        base = row["lift_change"]
        record = {"probe_rank": rank, "brand_id": brand_id, "brand_name": row["brand_name"],
                  "base_lift_change": base}
        for label in COUNTERFACTUALS:
            cf_row = by_label[label].get(brand_id)
            cf = cf_row["lift_change"] if cf_row else None
            record[f"{label}_lift_change"] = cf
            record[f"{label}_rank"] = ranks[label].get(brand_id)
            record[f"{label}_fans_after"] = cf_row["fans_after"] if cf_row else None
            record[f"{label}_excess_retained_pct"] = retained(base, cf)
        table.append(record)
    return table


def item_record(row: dict, decision, band: str) -> dict:
    return {
        "brand_id": row["brand_id"], "brand_name": row["brand_name"], "item_id": row["item_id"],
        "item_rank": row["item_rank"],
        "title": row["item_name"], "color": row["color"], "price": row["price"],
        "likes_after": row["likes_after"], "fan_likes_after": row["fan_likes_after"],
        "keyword_barbie": keyword(row), "band": band,
        "mechanism": decision.value("mechanism"), "mechanism_p": decision.probability("mechanism"),
        "official_license": decision.value("official_license"), "pink_aesthetic": decision.value("pink_aesthetic"),
    }


def examples(items: list[dict]) -> dict:
    ordered = sorted(items, key=lambda item: (-item["fan_likes_after"], item["item_rank"]))
    return {"items": len(items), "fan_likes": sum(item["fan_likes_after"] for item in items),
            "examples": ordered[:EXAMPLES]}


def disagreements(records: list[dict], brands: list[dict]) -> list[dict]:
    result = []
    for brand in brands:
        items = [item for item in records if item["brand_id"] == brand["brand_id"]]
        result.append({
            "brand_id": brand["brand_id"], "brand_name": brand["brand_name"],
            "keyword_not_accepted": examples([item for item in items
                                              if item["keyword_barbie"] and item["band"] != "accept"]),
            "accepted_without_keyword": examples([item for item in items
                                                  if item["band"] == "accept" and not item["keyword_barbie"]]),
            "pink_without_keyword": examples([item for item in items
                                              if item["pink_aesthetic"] >= PINK_MIN and not item["keyword_barbie"]]),
        })
    return result


def fan_share(item: dict) -> float:
    return item["fan_likes_after"] / item["likes_after"] if item["likes_after"] else 0.0


def behavioral_rejects(records: list[dict]) -> dict:
    accepted = [fan_share(item) for item in records if item["band"] == "accept"]
    if not accepted:
        return {"threshold_fan_share": None, "items": 0, "examples": []}
    threshold = statistics.median(accepted)
    rejects = [{**item, "fan_share": fan_share(item)} for item in records
               if item["band"] == "reject" and fan_share(item) >= threshold]
    rejects.sort(key=lambda item: (-item["fan_likes_after"], item["brand_name"], item["item_rank"]))
    return {"threshold_fan_share": threshold, "items": len(rejects), "examples": rejects[:20]}


def prior_comparison(table: list[dict]) -> list[dict]:
    by_name = {row["brand_name"]: row for row in table}
    rows = []
    for name, prior in PRIOR.items():
        row = by_name.get(name, {})
        rows.append({"brand_name": name, "prior": prior, "observed": {
            "base_lift_change": row.get("base_lift_change"),
            "keyword_lift_change": row.get("keyword_lift_change"),
            "keyword_retained_pct": row.get("keyword_excess_retained_pct"),
            "model_lift_change": row.get("model_lift_change"),
            "model_retained_pct": row.get("model_excess_retained_pct"),
            "model_mechanism_lift_change": row.get("model_mechanism_lift_change"),
            "model_mechanism_retained_pct": row.get("model_mechanism_excess_retained_pct"),
        }})
    return rows


def official_license_by_band(records: list[dict]) -> list[dict]:
    rows = []
    for band in BANDS:
        values = sorted(item["official_license"] for item in records
                        if item["mechanism"] == OFFICIAL and item["band"] == band)
        rows.append({"band": band, "items": len(values),
                     "official_license_min": values[0] if values else None,
                     "official_license_median": statistics.median(values) if values else None,
                     "official_license_max": values[-1] if values else None})
    return rows


def main() -> None:
    parser = build_parser(__doc__.splitlines()[0])
    parser.set_defaults(max_entities=250)
    args = parser.parse_args()
    case = Case(CASE_DIR, args, license=merrec_license())
    case.algeon.register_merrec()

    probe_sql = case.sql("01_probe.sql")
    if statement_body(probe_sql) != statement_body(PUBLISHED_PROBE.read_text(encoding="utf-8")):
        raise SystemExit(f"01_probe.sql no longer matches {PUBLISHED_PROBE}")
    probe = native_stage(case, "probe", probe_sql).table.to_pylist()
    case.capture.result("probe", probe)

    runs = {"baseline": counterfactual(case, "baseline", [])}
    baseline_top = [{key: row[key] for key in PROBE_COLUMNS} for row in runs["baseline"][:len(probe)]]
    if baseline_top != probe:
        raise SystemExit("the baseline counterfactual with an empty exclusion table did not reproduce the probe rows")
    ids = [row["brand_id"] for row in runs["baseline"][:len(probe)]]
    audited = [{"probe_rank": rank, "brand_id": brand_id, "brand_name": row["brand_name"],
                "lift_change": row["lift_change"]}
               for rank, (row, brand_id) in enumerate(zip(probe[:AUDITED_BRANDS], ids), 1)]
    case.capture.result("audited_brands", audited)

    bridge_stage = native_stage(case, "bridge_items", render(
        case.sql("02_bridge_items.sql"), brand_ids=", ".join(str(brand["brand_id"]) for brand in audited)))
    brand_rank = {brand["brand_id"]: brand["probe_rank"] for brand in audited}
    bridge = sorted(bridge_stage.table.to_pylist(),
                    key=lambda row: (brand_rank[row["brand_id"]], row["item_rank"]))[:args.max_entities]
    case.algeon.register_parquet(BRIDGE_TABLE, DATA_DIR / f"{BRIDGE_TABLE}.parquet",
                                 pa.Table.from_pylist(bridge, schema=bridge_stage.table.schema))
    case.capture.result("keyword_flag_mismatches", sum(keyword(row) != row["keyword_barbie"] for row in bridge))
    case.capture.result("bridge_items", [
        {"brand_id": brand["brand_id"], "brand_name": brand["brand_name"],
         "items": sum(row["brand_id"] == brand["brand_id"] for row in bridge),
         "keyword_items": sum(row["brand_id"] == brand["brand_id"] and keyword(row) for row in bridge),
         "fan_likes": sum(row["fan_likes_after"] for row in bridge if row["brand_id"] == brand["brand_id"])}
        for brand in audited])

    registry = case.registry(REGISTRY)
    batch = case.decide(registry, [Entity(str(row["item_id"]), state_text(row)) for row in bridge])
    decisions = batch.by_key()
    bands, counts = case.policy.apply(batch.decisions)
    case.capture.result("band_counts", counts)
    case.capture.result("mechanism_counts", dict(sorted(Counter(
        decision.value("mechanism") for decision in batch.decisions).items())))

    ordered = [decisions[str(row["item_id"])] for row in bridge]
    engine = registry.engine_table(ordered, {
        "item_id": pa.array([row["item_id"] for row in bridge], pa.int64()),
        "brand_id": pa.array([row["brand_id"] for row in bridge], pa.int64()),
    })
    engine = engine.append_column("band", pa.array([bands[str(row["item_id"])] for row in bridge], pa.string()))
    engine = engine.append_column("keyword_barbie", pa.array([keyword(row) for row in bridge], pa.bool_()))
    case.decision_table(REGISTRY, engine)

    keyword_ids = native_stage(case, "keyword_items", KEYWORD_ITEMS_SQL).table.column("item_id").to_pylist()
    model_ids = [row["item_id"] for row in bridge if bands[str(row["item_id"])] == "accept"]
    mechanism_ids = [row["item_id"] for row in bridge
                     if decisions[str(row["item_id"])].value("mechanism") == OFFICIAL]
    case.capture.result("exclusions", {"baseline": 0, "keyword": len(set(keyword_ids)), "model": len(set(model_ids)),
                                       "model_mechanism": len(set(mechanism_ids))})
    runs["keyword"] = counterfactual(case, "keyword", keyword_ids)
    runs["model"] = counterfactual(case, "model", model_ids)
    runs["model_mechanism"] = counterfactual(case, "model_mechanism", mechanism_ids)
    table = counterfactual_table(probe, ids, runs)
    case.capture.result("counterfactual", table)
    case.capture.result("prior_comparison", prior_comparison(table))

    decomposition = native_stage(case, "decomposition", render(
        case.sql("04_decomposition.sql"), bridge_items=BRIDGE_TABLE, decisions=REGISTRY)).table.to_pylist()
    decomposition.sort(key=lambda row: brand_rank[row["brand_id"]])
    case.capture.result("decomposition", decomposition)

    records = [item_record(row, decisions[str(row["item_id"])], bands[str(row["item_id"])]) for row in bridge]
    case.capture.result("disagreements", disagreements(records, audited))
    case.capture.result("official_license_by_band", official_license_by_band(records))
    case.capture.result("review_queue", [item for item in records if item["band"] == "review"])
    case.capture.result("behavioral_rejects", behavioral_rejects(records))
    case.capture.result("band_counts_by_brand", [
        {"brand_id": brand["brand_id"], "brand_name": brand["brand_name"],
         **{band: sum(item["brand_id"] == brand["brand_id"] and item["band"] == band for item in records)
            for band in BANDS}}
        for brand in audited])
    case.capture.result("decisions", [
        {"item_id": row["item_id"], "content_hash": decisions[str(row["item_id"])].content_hash,
         "model_id": decisions[str(row["item_id"])].model_id, "answers": decisions[str(row["item_id"])].answers}
        for row in bridge])
    case.finish()


if __name__ == "__main__":
    main()
