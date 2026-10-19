"""Decision-model loop client: probe, decide, recompute.

The package is the orchestration layer of the decision-model demo. Algeon
returns a bounded candidate set, a decision backend answers a fixed question
set for each entity, the answers are cached in a Parquet registry with their
provenance, and a typed decision table goes back into Algeon for the
recompute query.
"""
