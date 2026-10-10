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
