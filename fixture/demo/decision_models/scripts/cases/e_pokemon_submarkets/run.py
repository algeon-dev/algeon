#!/usr/bin/env python3
"""Case E (Project, forward validation): Pokemon sub-markets and the next months' purchase rate on MerRec.

The probe (`01_probe.sql`) returns the `--max-entities` (default 200) Pokemon
items (`brand_id = 17470`) with the most `item_view` events before 2023-09-01,
at least 50 views each, with their views, likes, `buy_comp` purchases and
distinct users (any event type) before September. The decision model sees only
the title, category path, condition, color and price of each item; the state
text never carries the engagement numbers, the item id or any later outcome, so
two listings with the same text share one registry decision. The answers become
`registry/pokemon_items.engine.parquet` (entity_key, item_id, answers, band),
registered as `pokemon_items`.

`02_forward_validation.sql` is a template over `{label_table}` and a label
expression `{label}`. A shopper is eligible when they viewed a labelled item
before 2023-09-01 and completed no Pokemon purchase (`buy_comp` on any
`brand_id = 17470` item) from 2023-07-01 to 2023-09-01. The outcome is a Pokemon
purchase of any Pokemon item on or after 2023-09-01, up to the end of the data.
A shopper who viewed items of several labels counts once in each label, so the
groups overlap and the two-proportion z treats them as independent samples; the
z values are therefore approximate. The statement also reports the catalog side
per label: items, mean price and the views, likes and purchases of the labelled
items before September.

Decisions the spec leaves open:

- Every probe item is grouped by its `submarket` choice whatever its band. The
  band follows the action noul `is_tcg_card` (policy: accept at 0.70, review at
  0.30, no `require`) and selects the review queue.
- The collector contrast splits items on the expected value of
  `habitual_collector_score` (at least 2 against below 2).
- The control is `03_control_rules.sql`: category path and title keywords in a
  fixed priority order (Trading Cards category with a bundle keyword, other
  Trading Cards, video game category or game keywords, plush or apparel
  category and keywords, otherwise unclassified). Short tokens (`lot`, `ds`,
  `3ds`, `gba`) match as space-delimited words so that `lot` skips `Lotad` and
  `ds` skips `cards`. The labels go to `registry/pokemon_item_rules.engine.parquet`,
  registered as `pokemon_item_rules`, and the same forward validation runs over
  them.
- `plan_prior` records the earlier pilot's sub-market figures beside the
  measured ones. The pilot used its own eligibility query, so the run reports
  both without a pass or fail check.
- The audience re-ranking lift of the earlier pilot (+17.2% on a 5,000-shopper
  target list) is out of scope and not implemented.

Commands, from fixture/demo/decision_models/scripts, one GPU run at a time:

    eval "$(grep -E '^export (JEV_API_URL|JEV_API_KEY)=' ~/.bashrc)"
    flock /tmp/decision-demo-gpu.lock python3 cases/e_pokemon_submarkets/run.py --mode live
    flock /tmp/decision-demo-gpu.lock python3 cases/e_pokemon_submarkets/run.py --mode replay
    python3 cases/e_pokemon_submarkets/compare_captures.py \\
        ../captures/e_pokemon_submarkets_live.json ../captures/e_pokemon_submarkets_replay.json

Replay serves each decision from the registry, or from the live capture when
the registry lacks it (pass `--registry-dir "$(mktemp -d)"` to force the
capture path), with zero API calls. Pass the same `--max-entities` as the live
run. A live run served entirely from the registry records no backend cost or
raw responses; keep the capture of the run that called the backend and pass it
to `--replay-from` for a replay from captures alone.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from pathlib import Path
import sys

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.capture import merrec_license  # noqa: E402
from client.case import Case, build_parser  # noqa: E402
from client.decision_client import Entity  # noqa: E402
from client.policy import BANDS  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent
REGISTRY = "pokemon_items"
DECISION_TABLE = "pokemon_items"
RULES_TABLE = "pokemon_item_rules"
DEFAULT_MAX_ENTITIES = 200
UNCLASSIFIED = "unclassified"
COLLECTOR_LABEL = ("CASE WHEN d.habitual_collector_score >= 2 THEN 'score_2_to_3' "
                   "ELSE 'score_0_to_1' END")
# Earlier pilot (buyers, eligible shoppers) per sub-market, reported beside the measured rates.
PLAN_PRIOR = {
    "tcg_bundle_lot_or_binder": (89, 7211),
    "tcg_sealed_or_single_card": (189, 18220),
    "video_game_or_hardware": (34, 4803),
    "plush_toy_or_apparel": (20, 4701),
}
PLAN_PRIOR_CONTRAST = ("tcg_bundle_lot_or_binder", "plush_toy_or_apparel", 4.5)
Z95 = 1.959963984540054


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


def render(sql: str, **values) -> str:
    for name, value in values.items():
        sql = sql.replace("{" + name + "}", str(value))
    return sql


def wilson(successes: int, total: int, z: float = Z95) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> float | None:
    """Pooled two-proportion z for rate 1 minus rate 2."""
    if n1 == 0 or n2 == 0:
        return None
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return (x1 / n1 - x2 / n2) / se if se else None


def state_text(row: dict) -> str:
    price = row["price"]
    return (
        f"Listing Title: {row['name'] or ''}\n"
        f"Category: {row['category_path'] or 'not specified'}\n"
        f"Condition: {row['item_condition_name'] or 'not specified'}\n"
        f"Color: {row['color'] or 'not specified'}\n"
        f"Price: {'not specified' if price is None else f'${price:.2f}'}"
    )


def rate_rows(stage_rows: list[dict]) -> list[dict]:
    rows = []
    for row in stage_rows:
        shoppers, buyers = row["shoppers"], row["buyers"]
        rows.append({
            "label": row["label"], "items": row["items"], "mean_price": row["mean_price"],
            "views": row["views"], "likes": row["likes"], "buys": row["buys"],
            "shoppers": shoppers, "buyers": buyers,
            "rate": buyers / shoppers if shoppers else None,
            "wilson95": wilson(buyers, shoppers),
        })
    return rows


def pairwise(rows: list[dict], exclude: tuple[str, ...] = ()) -> list[dict]:
    """z for every pair of labels, ordered by |z| descending; the higher rate comes first in each pair."""
    usable = [row for row in rows if row["label"] not in exclude and row["shoppers"]]
    pairs = []
    for left, right in combinations(usable, 2):
        high, low = (left, right) if left["rate"] >= right["rate"] else (right, left)
        z = two_proportion_z(high["buyers"], high["shoppers"], low["buyers"], low["shoppers"])
        pairs.append({
            "higher": high["label"], "lower": low["label"],
            "higher_rate": high["rate"], "lower_rate": low["rate"],
            "rate_ratio": high["rate"] / low["rate"] if low["rate"] else None,
            "difference_pp": 100 * (high["rate"] - low["rate"]),
            "z": z,
        })
    pairs.sort(key=lambda pair: -abs(pair["z"] or 0.0))
    return pairs


def plan_prior(rows: list[dict], pairs: list[dict]) -> dict:
    by_label = {row["label"]: row for row in rows}
    prior_rows = []
    for label, (buyers, shoppers) in PLAN_PRIOR.items():
        measured = by_label.get(label, {})
        prior_rows.append({
            "label": label, "prior_buyers": buyers, "prior_shoppers": shoppers, "prior_rate": buyers / shoppers,
            "measured_buyers": measured.get("buyers"), "measured_shoppers": measured.get("shoppers"),
            "measured_rate": measured.get("rate"), "measured_wilson95": measured.get("wilson95"),
            "prior_inside_measured_interval": (measured.get("wilson95") is not None
                                               and measured["wilson95"][0] <= buyers / shoppers
                                               <= measured["wilson95"][1]),
        })
    high, low, prior_z = PLAN_PRIOR_CONTRAST
    measured = next((pair for pair in pairs if {pair["higher"], pair["lower"]} == {high, low}), None)
    return {
        "rows": prior_rows,
        "contrast": {"higher": high, "lower": low, "prior_z": prior_z,
                     "prior_z_recomputed": two_proportion_z(*PLAN_PRIOR[high], *PLAN_PRIOR[low]),
                     "measured_higher": measured["higher"] if measured else None,
                     "measured_z": measured["z"] if measured else None},
        "rank_order_prior": sorted(PLAN_PRIOR, key=lambda label: -PLAN_PRIOR[label][0] / PLAN_PRIOR[label][1]),
        "rank_order_measured": [row["label"] for row in sorted(rows, key=lambda row: -(row["rate"] or 0.0))],
    }


def control_summary(items: list[dict], rates: list[dict]) -> dict:
    total = len(items)
    unclassified = [item for item in items if item["rule_label"] == UNCLASSIFIED]
    classified = [item for item in items if item["rule_label"] != UNCLASSIFIED]
    agree = sum(item["rule_label"] == item["submarket"] for item in items)
    classified_agree = sum(item["rule_label"] == item["submarket"] for item in classified)
    confusion = Counter((item["rule_label"], item["submarket"]) for item in items)
    return {
        "items": total,
        "agreement": agree / total if total else None,
        "agreement_wilson95": wilson(agree, total),
        "classified_items": len(classified),
        "classified_agreement": classified_agree / len(classified) if classified else None,
        "unclassified_items": len(unclassified),
        "unclassified_share": len(unclassified) / total if total else None,
        "unclassified_model_submarkets": dict(sorted(Counter(item["submarket"] for item in unclassified).items())),
        "confusion_rule_vs_model": [{"rule_label": rule, "submarket": model, "items": count}
                                    for (rule, model), count in sorted(confusion.items())],
        "disagreements": [{key: item[key] for key in ("item_id", "name", "category_path", "rule_label", "submarket")}
                          for item in classified if item["rule_label"] != item["submarket"]],
        "forward_validation": rates,
        "pairwise_z": pairwise(rates, exclude=(UNCLASSIFIED,)),
        "pairwise_z_with_unclassified": pairwise(rates),
    }


def collector_contrast(rows: list[dict]) -> dict:
    by_label = {row["label"]: row for row in rows}
    high, low = by_label.get("score_2_to_3"), by_label.get("score_0_to_1")
    contrast = {"threshold": 2.0, "groups": rows}
    if high and low:
        contrast.update({
            "z_high_minus_low": two_proportion_z(high["buyers"], high["shoppers"], low["buyers"], low["shoppers"]),
            "rate_ratio": high["rate"] / low["rate"] if low["rate"] else None,
            "difference_pp": 100 * (high["rate"] - low["rate"]),
        })
    return contrast


def stage_timings(case: Case) -> list[dict]:
    return [{"name": stage["name"], "rows": stage["rows"], "elapsed_s": stage["elapsed_s"],
             "gpu_path": (stage["explain"] or {}).get("gpu_path"),
             "cpu_boundaries": (stage["explain"] or {}).get("cpu_boundaries"),
             "host_datafusion_boundary_count": (stage["execution"] or {}).get("host_datafusion_boundary_count")}
            for stage in case.capture.data["stages"]]


def main() -> None:
    args = build_parser(__doc__.splitlines()[0]).parse_args()
    max_entities = args.max_entities or DEFAULT_MAX_ENTITIES
    case = Case(CASE_DIR, args, license=merrec_license())
    case.algeon.register_merrec()
    case.capture.result("max_entities", max_entities)

    probe = native_stage(case, "probe", render(case.sql("01_probe.sql"), max_entities=max_entities)).table.to_pylist()
    entities = [Entity(str(row["item_id"]), state_text(row)) for row in probe]
    case.capture.result("probe", {
        "items": len(probe),
        "distinct_states": len({entity.state for entity in entities}),
        "views_min": min(row["views"] for row in probe),
        "views_max": max(row["views"] for row in probe),
        "views_total": sum(row["views"] for row in probe),
        "likes_total": sum(row["likes"] for row in probe),
        "buys_total": sum(row["buys"] for row in probe),
        "users_total": sum(row["users"] for row in probe),
        "mean_price": sum(row["price"] or 0.0 for row in probe) / len(probe),
        "category_paths": dict(Counter(row["category_path"] for row in probe).most_common()),
    })

    registry = case.registry(REGISTRY)
    batch = case.decide(registry, entities)
    decisions = batch.by_key()
    bands, counts = case.policy.apply(batch.decisions)
    item_ids = [int(decision.entity_key) for decision in batch.decisions]
    engine = registry.engine_table(batch.decisions, {"item_id": pa.array(item_ids, pa.int64())})
    engine = engine.append_column("band", pa.array([bands[str(item_id)] for item_id in item_ids], pa.string()))
    case.decision_table(DECISION_TABLE, engine)
    case.capture.result("band_counts", counts)
    case.capture.result("submarket_counts", dict(sorted(Counter(
        decision.value("submarket") for decision in batch.decisions).items())))
    case.capture.result("submarket_band_counts", [
        {"submarket": submarket, **{band: sum(1 for decision in batch.decisions
                                              if decision.value("submarket") == submarket
                                              and bands[decision.entity_key] == band) for band in BANDS}}
        for submarket in case.questions["submarket"]["criteria"]])

    template = case.sql("02_forward_validation.sql")
    forward = rate_rows(native_stage(case, "forward_validation", render(
        template, label_table=DECISION_TABLE, label="d.submarket")).table.to_pylist())
    pairs = pairwise(forward)
    case.capture.result("forward_validation", forward)
    case.capture.result("pairwise_z", pairs)
    case.capture.result("largest_contrast", pairs[0] if pairs else None)
    case.capture.result("plan_prior", plan_prior(forward, pairs))

    collector = rate_rows(native_stage(case, "collector_contrast", render(
        template, label_table=DECISION_TABLE, label=COLLECTOR_LABEL)).table.to_pylist())
    case.capture.result("collector_contrast", collector_contrast(collector))

    rules = {row["item_id"]: row["rule_label"]
             for row in native_stage(case, "control_rules", case.sql("03_control_rules.sql")).table.to_pylist()}
    missing = set(item_ids) - set(rules)
    if missing:
        raise SystemExit(f"control rules returned no label for {len(missing)} items")
    case.decision_table(RULES_TABLE, pa.table({
        "item_id": pa.array(item_ids, pa.int64()),
        "rule_label": pa.array([rules[item_id] for item_id in item_ids], pa.string()),
    }))
    control_rates = rate_rows(native_stage(case, "control_forward_validation", render(
        template, label_table=RULES_TABLE, label="d.rule_label")).table.to_pylist())

    items = []
    for row in probe:
        decision = decisions[str(row["item_id"])]
        items.append({
            "item_id": row["item_id"], "name": row["name"], "category_path": row["category_path"],
            "price": row["price"], "views": row["views"], "likes": row["likes"], "buys": row["buys"],
            "users": row["users"], "submarket": decision.value("submarket"),
            "submarket_p": decision.probability("submarket"),
            "habitual_collector_score": decision.value("habitual_collector_score"),
            "is_tcg_card": decision.value("is_tcg_card"),
            "is_hardware_or_game": decision.value("is_hardware_or_game"),
            "band": bands[str(row["item_id"])], "rule_label": rules[row["item_id"]],
        })
    case.capture.result("control", control_summary(items, control_rates))
    case.capture.result("review_queue", [
        {key: item[key] for key in ("item_id", "name", "category_path", "price", "views", "submarket", "submarket_p",
                                    "is_tcg_card", "is_hardware_or_game", "habitual_collector_score", "rule_label")}
        for item in items if item["band"] == "review"])
    case.capture.result("items", items)
    case.capture.result("stage_timings", stage_timings(case))
    case.finish()


if __name__ == "__main__":
    main()
