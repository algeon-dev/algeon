#!/usr/bin/env python3
"""Case C (Project): complaint causes for the 99,539 low-rating headphone reviews.

The probe fits the 24 kmeans clusters of `03_headphone_emerging.sql`, runs a
64-neighbor brute-force kNN over the same vectors and ranks each cluster's
reviews by mean distance to their same-cluster neighbors, all in one statement.
The full assignment table is saved as `registry/review_clusters.engine.parquet`.
The 8 densest reviews per cluster (192) are decided on four questions; the
majority `complaint_cause` labels the cluster. A cluster whose majority share is
below `cluster_consistency_min` is mixed, and up to `escalation_cap` of its
densest reviews are decided one by one. Cluster labels, overridden by per-review
labels for escalated reviews, are joined back onto every review for the
recompute, the rankings and the keyword control. A holdout of 100 other reviews
is decided individually to measure agreement with the cluster labels.

Commands, from fixture/demo/decision_models/scripts:

    eval "$(grep -E '^export (JEV_API_URL|JEV_API_KEY)=' ~/.bashrc)"
    flock /tmp/decision-demo-gpu.lock python3 cases/c_complaint_causes/run.py --mode live

    # second backend on the same 192 representatives, compared with the jev registry
    flock /tmp/decision-demo-gpu.lock python3 cases/c_complaint_causes/run.py --mode live \\
        --backend openai_compatible --reuse-clusters --representatives-only

    # replay: an empty registry directory makes every decision come from the live capture
    flock /tmp/decision-demo-gpu.lock python3 cases/c_complaint_causes/run.py --mode replay \\
        --reuse-clusters --clusters ../registry/review_clusters.engine.parquet --registry-dir "$(mktemp -d)"
    python3 cases/c_complaint_causes/compare_captures.py \\
        ../captures/c_complaint_causes_live.json ../captures/c_complaint_causes_replay.json

Case flags:

    --reuse-clusters        Skip the kmeans and kNN probe and read the saved assignment table. kmeans
                            cluster sizes drift slightly between fits and cluster ids are query-local,
                            so a run that must reproduce earlier states (replay, second backend) has to
                            reuse the saved assignments.
    --clusters PATH         Assignment table for --reuse-clusters
                            (default: <registry-dir>/review_clusters.engine.parquet).
    --representatives-only  Decide the representatives, label the clusters and stop. Required for a
                            backend other than jev, so its engine tables never replace the jev ones;
                            that run reports agreement with the jev registry.
    --holdout-size N        Reviews in the individual holdout check (default 100).

Incremental design (not implemented): cuvs_kmeans returns no centroids, so a new
month of reviews would be assigned by running cuvs_brute_force_knn with the new
review vectors as queries against the saved clustered vectors and taking the
majority cluster id among the k neighbors. Reviews whose neighbor vote is weak,
or whose nearest distance exceeds the cluster's usual neighbor distance, are
outliers; only those, and clusters that the vote cannot fill, go to the decision
backend, and every other new review inherits its cluster label at no API cost.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.case import Case, build_parser  # noqa: E402
from client.decision_client import Entity  # noqa: E402
from client.registry import Registry  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent


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
REGISTRY = "review_causes"
DEFECT = "product_defect_or_failure"
REPRESENTATIVES_PER_CLUSTER = 8
HOLDOUT_SEED = 20261019
CONTEXT_CHUNK = 2000
NUMERIC_QUESTIONS = ("manufacturer_actionable", "safety_hazard", "severity")
# Coarse mapping used only to score the keyword control against model labels.
KEYWORD_TO_CAUSE = {
    "durability": DEFECT,
    "charging": DEFECT,
    "connection": DEFECT,
    "battery": "performance_below_spec",
    "sound": "performance_below_spec",
    "microphone": "performance_below_spec",
    "comfort": "expectation_or_usability_mismatch",
    "returns": "price_or_service",
}
CLUSTER_SCHEMA = pa.schema([
    ("id", pa.int64()), ("cluster_id", pa.int32()), ("same_cluster_neighbors", pa.int64()),
    ("mean_distance", pa.float64()), ("density_rank", pa.int64()),
])


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def load_clusters(case: Case) -> list[dict]:
    args = case.args
    if args.reuse_clusters:
        path = Path(args.clusters or args.registry_dir / "review_clusters.engine.parquet")
        table = pq.read_table(path).select(CLUSTER_SCHEMA.names).cast(CLUSTER_SCHEMA)
        case.algeon.register_parquet("review_clusters", path)
        case.capture.result("skipped_stages", ["probe"])
    else:
        stage = native_stage(case, "probe", case.sql("01_probe.sql"))
        table = stage.table.select(CLUSTER_SCHEMA.names).cast(CLUSTER_SCHEMA)
        case.decision_table("review_clusters", table)
        case.capture.result("skipped_stages", [])
    rows = table.to_pylist()
    if any(row["cluster_id"] is None for row in rows):
        raise SystemExit("probe returned reviews without a cluster; a review lost its self neighbor in the kNN")
    return rows


def fetch_context(case: Case, name: str, ids: list[int]) -> dict[int, dict]:
    template = case.sql("02_context.sql")
    ordered = sorted(ids)
    contexts = {}
    for start in range(0, len(ordered), CONTEXT_CHUNK):
        chunk = ordered[start:start + CONTEXT_CHUNK]
        stage_name = name if len(ordered) <= CONTEXT_CHUNK else f"{name}_{start // CONTEXT_CHUNK + 1}"
        stage = native_stage(case, stage_name, template.replace("{review_ids}", ", ".join(map(str, chunk))))
        contexts.update({row["id"]: row for row in stage.table.to_pylist()})
    missing = set(ordered) - set(contexts)
    if missing:
        raise SystemExit(f"context query returned no row for {len(missing)} reviews")
    return contexts


def state(row: dict) -> str:
    text = row["review_text"] or ""
    if (row["text_chars"] or 0) > len(text):
        text += " [truncated]"
    return "\n".join([
        f"Product: {row['product_title'] or 'unknown'}",
        f"Rating: {row['rating']} of 5 stars",
        f"Verified purchase: {'yes' if row['verified_purchase'] else 'no'}",
        f"Review title: {row['review_title'] or ''}",
        f"Review text: {text}",
        f"Similar complaints: this review belongs to a group of {row['cluster_size']:,} similar low-rating "
        f"headphone reviews covering {row['cluster_products']:,} products.",
    ])


def decide_reviews(case: Case, registry: Registry, name: str, ids: list[int]):
    contexts = fetch_context(case, f"context_{name}", ids)
    entities = [Entity(str(review_id), state(contexts[review_id])) for review_id in sorted(ids)]
    return case.decide(registry, entities, name=f"decide_{name}")


def label_clusters(case: Case, decisions_by_cluster: dict[int, list], sizes: Counter) -> dict[int, dict]:
    spec = case.policy.spec
    options = list(case.questions["complaint_cause"]["criteria"])
    labels = {}
    for cluster_id in sorted(sizes):
        decisions = decisions_by_cluster.get(cluster_id, [])
        counts = Counter(decision.value("complaint_cause") for decision in decisions)
        probability = defaultdict(float)
        for decision in decisions:
            probability[decision.value("complaint_cause")] += decision.probability("complaint_cause")
        label = max(options, key=lambda option: (counts[option], probability[option], -options.index(option)))
        consistency = counts[label] / len(decisions) if decisions else 0.0
        actionable = mean(decision.value("manufacturer_actionable") for decision in decisions)
        labels[cluster_id] = {
            "cluster_id": cluster_id,
            "label": label,
            "consistency": consistency,
            "size": sizes[cluster_id],
            "mixed": consistency < spec["cluster_consistency_min"],
            "representatives": len(decisions),
            "mean_manufacturer_actionable": actionable,
            "mean_safety_hazard": mean(decision.value("safety_hazard") for decision in decisions),
            "mean_severity": mean(decision.value("severity") for decision in decisions),
            "defect_actionable": label == DEFECT and actionable is not None and actionable >= case.policy.accept_min,
            "cause_counts": dict(sorted(counts.items())),
        }
    return labels


def cluster_label_table(labels: dict[int, dict]) -> pa.Table:
    rows = [{key: value for key, value in label.items() if key != "cause_counts"} for label in labels.values()]
    return pa.Table.from_pylist(rows, schema=pa.schema([
        ("cluster_id", pa.int32()), ("label", pa.string()), ("consistency", pa.float64()), ("size", pa.int64()),
        ("mixed", pa.bool_()), ("representatives", pa.int64()), ("mean_manufacturer_actionable", pa.float64()),
        ("mean_safety_hazard", pa.float64()), ("mean_severity", pa.float64()), ("defect_actionable", pa.bool_()),
    ]))


def backend_agreement(case: Case, batch, clusters_of: dict[int, int], labels: dict[int, dict]) -> dict:
    """Compare these representative decisions with the jev registry's decisions of the same states."""
    reference = Registry(case.args.registry_dir / f"{REGISTRY}.parquet", case.questions)
    pairs = []
    for decision in batch.decisions:
        other = reference.lookup(decision.content_hash, decision.question_set_hash)
        if other is not None:
            pairs.append((decision, other))
    same = sum(mine.value("complaint_cause") == theirs.value("complaint_cause") for mine, theirs in pairs)
    by_cluster = defaultdict(list)
    for _, theirs in pairs:
        by_cluster[clusters_of[int(theirs.entity_key)]].append(theirs)
    reference_labels = label_clusters(case, by_cluster, Counter({cluster: labels[cluster]["size"] for cluster in labels}))
    confusion = Counter((theirs.value("complaint_cause"), mine.value("complaint_cause")) for mine, theirs in pairs)
    return {
        "compared": len(pairs),
        "complaint_cause_agreement": same / len(pairs) if pairs else None,
        "complaint_cause_agreement_wilson95": wilson(same, len(pairs)),
        "mean_absolute_difference": {
            name: mean(abs(mine.value(name) - theirs.value(name)) for mine, theirs in pairs) for name in NUMERIC_QUESTIONS},
        "cluster_label_agreement": sum(
            reference_labels[cluster]["label"] == labels[cluster]["label"] for cluster in labels) / len(labels),
        "confusion_jev_vs_this_backend": [
            {"jev": key[0], "this_backend": key[1], "reviews": count} for key, count in sorted(confusion.items())],
    }


def rows(stage) -> list[dict]:
    return stage.table.to_pylist()


def keyword_comparison(crosstab: list[dict], per_product: list[dict], model_ranking: list[dict]) -> dict:
    total = sum(row["reviews"] for row in crosstab)
    unclassified = sum(row["reviews"] for row in crosstab if row["keyword_cause"] == "unclassified")
    classified = [row for row in crosstab if row["keyword_cause"] != "unclassified"]
    agree = sum(row["reviews"] for row in classified if KEYWORD_TO_CAUSE[row["keyword_cause"]] == row["model_cause"])
    classified_total = sum(row["reviews"] for row in classified)
    by_product = {row["product_id"]: row for row in per_product}
    top = []
    for rank, product in enumerate(model_ranking[:10], 1):
        control = by_product.get(product["product_id"], {})
        keyword = control.get("keyword_cause")
        top.append({
            "rank": rank, "product_id": product["product_id"], "title": product["title"],
            "model_cause_2022": control.get("model_cause"), "model_reviews": control.get("model_reviews"),
            "keyword_cause_2022": keyword, "keyword_reviews": control.get("keyword_reviews"),
            "keyword_unclassified": control.get("keyword_unclassified"), "reviews2022": control.get("reviews2022"),
            "mapped_agrees": keyword is not None and KEYWORD_TO_CAUSE[keyword] == control.get("model_cause"),
        })
    return {
        "reviews": total,
        "unclassified": unclassified,
        "unclassified_share": unclassified / total if total else None,
        "mapping": KEYWORD_TO_CAUSE,
        "classified_agreement_under_mapping": agree / classified_total if classified_total else None,
        "top10_products": top,
    }


def holdout(case: Case, registry: Registry, clusters: list[dict], excluded: set[int], labels: dict[int, dict]) -> dict:
    pool = sorted(row["id"] for row in clusters if row["id"] not in excluded)
    sample = random.Random(HOLDOUT_SEED).sample(pool, case.args.holdout_size)
    clusters_of = {row["id"]: row["cluster_id"] for row in clusters}
    batch = decide_reviews(case, registry, "holdout", sample)
    outcomes = []
    for decision in batch.decisions:
        cluster_id = clusters_of[int(decision.entity_key)]
        outcomes.append({
            "id": int(decision.entity_key), "cluster_id": cluster_id, "mixed": labels[cluster_id]["mixed"],
            "cluster_label": labels[cluster_id]["label"], "review_label": decision.value("complaint_cause"),
            "agrees": decision.value("complaint_cause") == labels[cluster_id]["label"],
        })

    def summary(subset):
        agree = sum(outcome["agrees"] for outcome in subset)
        return {"reviews": len(subset), "agree": agree, "agreement": agree / len(subset) if subset else None,
                "wilson95": wilson(agree, len(subset))}

    return {
        "seed": HOLDOUT_SEED,
        "pool": len(pool),
        "all": summary(outcomes),
        "consistent_clusters": summary([outcome for outcome in outcomes if not outcome["mixed"]]),
        "mixed_clusters": summary([outcome for outcome in outcomes if outcome["mixed"]]),
        "reviews": outcomes,
    }


def main() -> None:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--reuse-clusters", action="store_true",
                        help="skip the kmeans probe and read the saved assignment table")
    parser.add_argument("--clusters", type=Path, default=None,
                        help="assignment table for --reuse-clusters (default <registry-dir>/review_clusters.engine.parquet)")
    parser.add_argument("--representatives-only", action="store_true",
                        help="decide the representatives, label clusters and stop")
    parser.add_argument("--holdout-size", type=int, default=100)
    args = parser.parse_args()
    if args.backend != "jev" and not args.representatives_only:
        parser.error("a backend other than jev runs only with --representatives-only")

    case = Case(CASE_DIR, args)
    case.algeon.register_amazon()
    spec = case.policy.spec
    clusters = load_clusters(case)
    sizes = Counter(row["cluster_id"] for row in clusters)
    clusters_of = {row["id"]: row["cluster_id"] for row in clusters}
    per_cluster = (args.max_entities or REPRESENTATIVES_PER_CLUSTER * len(sizes)) // len(sizes)
    representative_ids = [row["id"] for row in clusters if row["density_rank"] <= per_cluster]
    case.capture.result("probe", {
        "reviews": len(clusters), "clusters": len(sizes), "representatives": len(representative_ids),
        "interior_reviews": sum(row["same_cluster_neighbors"] >= 32 for row in clusters),
    })

    registry = case.registry(REGISTRY)
    representatives = decide_reviews(case, registry, "representatives", representative_ids)
    by_cluster = defaultdict(list)
    for decision in representatives.decisions:
        by_cluster[clusters_of[int(decision.entity_key)]].append(decision)
    labels = label_clusters(case, by_cluster, sizes)
    case.capture.result("cluster_labels", list(labels.values()))
    case.capture.result("representatives", [
        {"id": int(decision.entity_key), "cluster_id": clusters_of[int(decision.entity_key)],
         **{name: decision.value(name) for name in case.questions}} for decision in representatives.decisions])

    if args.representatives_only:
        if args.backend != "jev":
            case.capture.result("backend_agreement", backend_agreement(case, representatives, clusters_of, labels))
        case.finish()
        return

    mixed = [cluster_id for cluster_id, label in labels.items() if label["mixed"]]
    escalated_ids = [row["id"] for row in clusters
                     if labels[row["cluster_id"]]["mixed"] and row["density_rank"] <= spec["escalation_cap"]]
    escalated = decide_reviews(case, registry, "escalation", escalated_ids) if escalated_ids else None
    escalation = {"mixed_clusters": mixed, "reviews": len(escalated_ids), "per_cluster": []}
    if escalated is not None:
        escalation["extra_cost_usd"] = escalated.summary["cost_usd"]
        escalation["extra_backend_calls"] = escalated.summary["backend_calls"]
        escalation["wall_s"] = escalated.summary["wall_s"]
        for cluster_id in mixed:
            counts = Counter(decision.value("complaint_cause") for decision in escalated.decisions
                             if clusters_of[int(decision.entity_key)] == cluster_id)
            decided = sum(counts.values())
            escalation["per_cluster"].append({
                "cluster_id": cluster_id, "size": sizes[cluster_id], "decided": decided,
                "representative_label": labels[cluster_id]["label"],
                "per_review_causes": dict(sorted(counts.items())),
                "per_review_majority_share": max(counts.values()) / decided if decided else None,
            })
    case.capture.result("escalation", escalation)

    engine_decisions = [decision for decision in representatives.decisions
                        if not labels[clusters_of[int(decision.entity_key)]]["mixed"]]
    sources = ["representative"] * len(engine_decisions)
    if escalated is not None:
        engine_decisions += escalated.decisions
        sources += ["escalated"] * len(escalated.decisions)
    bands, band_counts = case.policy.apply(engine_decisions)
    review_ids = [int(decision.entity_key) for decision in engine_decisions]
    engine = registry.engine_table(engine_decisions, {
        "id": pa.array(review_ids, pa.int64()),
        "cluster_id": pa.array([clusters_of[review_id] for review_id in review_ids], pa.int32()),
        "label_source": sources,
        "band": [bands[decision.entity_key] for decision in engine_decisions],
        "defect_actionable": pa.array([decision.value("complaint_cause") == DEFECT
                                       and bands[decision.entity_key] == "accept" for decision in engine_decisions]),
    })
    case.decision_table("review_causes", engine)
    case.decision_table("cluster_labels", cluster_label_table(labels))
    case.capture.result("bands", band_counts)

    ranking = rows(native_stage(case, "recompute", case.sql("03_recompute.sql")))
    case.capture.result("defect_actionable_ranking", ranking)
    case.capture.result("product_cause_shares", rows(native_stage(case, 
        "product_cause_shares", case.sql("04_product_cause_shares.sql"))))
    case.capture.result("cause_shares", rows(native_stage(case, "cause_shares", case.sql("05_cause_shares.sql"))))
    anonymous = rows(native_stage(case, "anonymous_ranking", case.sql("06_anonymous_ranking.sql")))
    case.capture.result("anonymous_ranking", anonymous)
    low_rating = rows(native_stage(case, "low_rating_ranking", case.sql("07_low_rating_ranking.sql")))
    case.capture.result("low_rating_ranking", low_rating)
    weighted = rows(native_stage(case, "helpful_weighted", case.sql("08_helpful_weighted.sql")))
    unweighted_totals = {row["product_id"]: (row["total2021"], row["total2022"]) for row in ranking}
    case.capture.result("helpful_weighted", {
        "join": "semantic_reviews_text.id = reviews.review_id",
        "products_with_complete_join": sum(
            unweighted_totals.get(row["product_id"]) == (row["total2021"], row["total2022"]) for row in weighted),
        "products": len(ranking),
        "ranking": weighted,
    })
    case.capture.result("safety_clusters", rows(native_stage(case, "safety_clusters", case.sql("09_safety_clusters.sql"))))
    case.capture.result("rankings_top10", [
        {"rank": index + 1,
         "defect_actionable": (ranking[index]["product_id"], ranking[index]["title"], ranking[index]["change_pp"])
         if index < len(ranking) else None,
         "anonymous_cluster": (anonymous[index]["product_id"], anonymous[index]["title"], anonymous[index]["cluster_id"],
                               anonymous[index]["change_pp"]) if index < len(anonymous) else None,
         "low_rating": (low_rating[index]["product_id"], low_rating[index]["title"], low_rating[index]["change_pp"])
         if index < len(low_rating) else None,
         "helpful_weighted": (weighted[index]["product_id"], weighted[index]["title"], weighted[index]["change_pp"])
         if index < len(weighted) else None}
        for index in range(10)])

    crosstab = rows(native_stage(case, "control_keywords", case.sql("control_keywords.sql")))
    per_product = rows(native_stage(case, "control_keywords_products", case.sql("control_keywords_products.sql")))
    case.capture.result("control_keywords", {"crosstab": crosstab, **keyword_comparison(crosstab, per_product, ranking)})
    case.capture.result("control_keywords_products", per_product)

    excluded = set(representative_ids) | set(escalated_ids)
    case.capture.result("holdout", holdout(case, registry, clusters, excluded, labels))
    case.finish()


if __name__ == "__main__":
    main()
