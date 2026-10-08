//! Shared-subplan materialization: one native aggregate for a family of
//! repeated aggregates over the same local Parquet sources.
//!
//! The rule matches two or more aggregate nodes whose group keys, aggregate
//! expressions and source set agree except for the sources' reader predicates (a
//! grouped family), or scalar aggregates without `FILTER` clauses over sources
//! that differ only in reader predicate (a scalar filtered family). In the
//! rewrite, the family's representative subtree is cloned once with its sources
//! widened to the union of the branch predicates and columns; each original
//! aggregate becomes a filter over that one aggregate (grouped) or a projection
//! that renames its branch's `FILTER`ed aggregate outputs (scalar). As a
//! precondition, a grouped branch predicate must be evaluable after the
//! aggregate, over group columns the source emits unchanged; a scalar branch
//! predicate must reference source columns the widened subtree can carry to the
//! aggregate input. A family that fails either is reported with a
//! [`SharedSubplanMaterializationRejectedReason`] and left as it is. Facts: a
//! widened source gets a new output schema, the shared scalar aggregate's schema
//! is rebuilt from the branch outputs, and every other cloned node keeps its
//! facts.
//!
//! Two passes of the optimizer sequence live here. Scalar-aggregate
//! derivation (`scalar_derivation`) runs early; family materialization runs
//! late, after aggregate pushdown and pruning have settled the source shapes
//! the family signature fingerprints.

mod api;
mod diagnostics;
mod discovery;
mod estimate;
mod expr_helpers;
mod lineage;
mod rewrite;
mod scalar_derivation;
mod signature;
mod source_widening;

pub use api::{SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, SharedSubplanMaterializationOptions};
pub(crate) use api::{
    optimize_scalar_aggregate_from_grouped_aggregate_if_enabled,
    optimize_shared_subplan_materialization_without_scalar_derivation,
};
pub use diagnostics::{
    SharedSubplanMaterializationCandidateDiagnostics, SharedSubplanMaterializationDiagnostics,
    SharedSubplanMaterializationRejectedReason,
};
