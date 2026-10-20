"""Trigram blocking control: brand-name similarity without behavior or a decision model.

Names are lowercased, punctuation is deleted, and each word contributes
pg_trgm-style trigrams (two leading spaces, one trailing space). Two brands
form a candidate pair when the Jaccard similarity of their trigram sets is at
least `THRESHOLD`, 0.5 here, stricter than the pg_trgm default of 0.3: at
0.5 "NCT" and "NCT 127" still pair (similarity 0.5) while "Dior" and
"Christian Dior" (0.33) do not.

Candidate generation uses prefix filtering over a global trigram order
(rarest first): if J(x, y) >= t then the first |x| - ceil(t |x|) + 1 trigrams
of x and the first |y| - ceil(t |y|) + 1 trigrams of y share a trigram, so the
inverted index over prefixes finds every qualifying pair and each candidate is
then verified exactly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import time
import unicodedata

THRESHOLD = 0.5
NAMES_SQL = """SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id ORDER BY brand_id"""


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKC", name or "").lower()
    kept = "".join(ch if ch.isalnum() or ch.isspace() else "" for ch in text)
    return " ".join(kept.split())


def trigrams(name: str) -> frozenset[str]:
    grams = set()
    for word in normalize(name).split():
        padded = f"  {word} "
        grams.update(padded[i:i + 3] for i in range(len(padded) - 2))
    return frozenset(grams)


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    shared = len(a & b)
    return shared / (len(a) + len(b) - shared)


def candidate_pairs(grams: dict[int, frozenset[str]], threshold: float = THRESHOLD) -> dict[tuple[int, int], float]:
    """All brand pairs with trigram Jaccard >= threshold, keyed by (smaller id, larger id)."""
    frequency = Counter(gram for value in grams.values() for gram in value)
    index: dict[str, list[int]] = defaultdict(list)
    for brand_id in sorted(grams):
        ordered = sorted(grams[brand_id], key=lambda gram: (frequency[gram], gram))
        prefix = len(ordered) - math.ceil(threshold * len(ordered) - 1e-9) + 1
        for gram in ordered[:prefix]:
            index[gram].append(brand_id)
    seen: set[tuple[int, int]] = set()
    pairs: dict[tuple[int, int], float] = {}
    for posting in index.values():
        for i, first in enumerate(posting):
            size = len(grams[first])
            for second in posting[i + 1:]:
                key = (first, second)
                if key in seen:
                    continue
                seen.add(key)
                other = len(grams[second])
                if other < threshold * size or size < threshold * other:
                    continue
                value = similarity(grams[first], grams[second])
                if value >= threshold:
                    pairs[key] = value
    return pairs


def pair_view(pair: dict, value: float) -> dict:
    return {"brand_id_a": pair["brand_id_a"], "brand_a": pair["brand_a"], "brand_id_b": pair["brand_id_b"],
            "brand_b": pair["brand_b"], "trigram_similarity": round(value, 4),
            "relationship_type": pair["relationship_type"], "should_merge_brand_id": pair["should_merge_brand_id"],
            "band": pair["band"]}


def run_control(names: list[dict], probe_pairs: list[dict], threshold: float = THRESHOLD) -> dict:
    """Run the control over every brand name and compare it with the model's bands on the probe pairs.

    `probe_pairs` rows carry `brand_id_a`, `brand_id_b`, their names, the model
    answers and the policy `band`; each row gains `trigram_similarity`.
    """
    started = time.perf_counter()
    grams = {row["brand_id"]: trigrams(row["brand_name"]) for row in names}
    empty = sum(1 for value in grams.values() if not value)
    grams = {key: value for key, value in grams.items() if value}
    pairs = candidate_pairs(grams, threshold)
    name_of = {row["brand_id"]: row["brand_name"] for row in names}

    found, missed, wrong = [], [], []
    for pair in probe_pairs:
        value = similarity(grams.get(pair["brand_id_a"], frozenset()), grams.get(pair["brand_id_b"], frozenset()))
        pair["trigram_similarity"] = round(value, 4)
        key = (min(pair["brand_id_a"], pair["brand_id_b"]), max(pair["brand_id_a"], pair["brand_id_b"]))
        proposed = key in pairs
        if proposed != (value >= threshold):
            raise AssertionError(f"prefix filter missed probe pair {key}")
        if pair["band"] == "accept":
            (found if proposed else missed).append(pair_view(pair, value))
        elif proposed:
            wrong.append(pair_view(pair, value))

    wrong_by_relationship = Counter(row["relationship_type"] for row in wrong)
    top = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:20]
    return {
        "threshold": threshold,
        "normalization": "NFKC, lowercase, punctuation deleted, pg_trgm word trigrams",
        "brand_names": len(names),
        "names_without_trigrams": empty,
        "pairs_proposed_all_brands": len(pairs),
        "examples_all_brands": [
            {"brand_id_a": a, "brand_a": name_of[a], "brand_id_b": b, "brand_b": name_of[b],
             "trigram_similarity": round(value, 4)} for (a, b), value in top],
        "probe_pairs": len(probe_pairs),
        "probe_pairs_proposed": sum(1 for pair in probe_pairs if pair["trigram_similarity"] >= threshold),
        "accepted_aliases": len(found) + len(missed),
        "accepted_found": len(found),
        "accepted_missed": len(missed),
        "wrongly_proposed": len(wrong),
        "wrongly_proposed_by_relationship": dict(sorted(wrong_by_relationship.items())),
        "found": found,
        "missed": missed,
        "wrongly_proposed_pairs": wrong,
        "elapsed_s": time.perf_counter() - started,
    }
