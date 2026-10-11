use super::{
    diagnostics::SharedSubplanMaterializationDiagnostics,
    discovery::discover_shared_subplan_materialization_candidates, rewrite::SharedSubplanMaterializationRewriter,
    scalar_derivation::optimize_scalar_aggregate_from_grouped_aggregate,
};
use crate::{
    error::QueryError,
    optimizer::NativeOptimizerChange,
    plan::{QueryPlan, RewrittenNodes},
};

/// Stable rule name in diagnostics rows; both passes of this module report
/// under it.
pub const SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME: &str = "shared_subplan_materialization";

/// Configuration for shared-subplan materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedSubplanMaterializationOptions {
    /// Whether the pass may rewrite repeated native subplans.
    pub enabled: bool,
}

impl SharedSubplanMaterializationOptions {
    /// Returns options with shared-subplan materialization disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for SharedSubplanMaterializationOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Outcome of one pass of this module: the plan to continue with, whether it
/// differs from the input, and the candidate rows the pass produced.
pub(crate) struct SharedSubplanMaterializationResult {
    pub(crate) plan: QueryPlan,
    pub(crate) change: NativeOptimizerChange,
    pub(crate) diagnostics: SharedSubplanMaterializationDiagnostics,
}

/// The early pass: derives a scalar `SUM` aggregate from a grouped aggregate
/// over the same row set. See `scalar_derivation`.
pub(crate) fn optimize_scalar_aggregate_from_grouped_aggregate_if_enabled(
    plan: QueryPlan,
    options: &SharedSubplanMaterializationOptions,
) -> Result<SharedSubplanMaterializationResult, QueryError> {
    if options.enabled {
        optimize_scalar_aggregate_from_grouped_aggregate(plan)
    } else {
        Ok(SharedSubplanMaterializationResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: SharedSubplanMaterializationDiagnostics::default(),
        })
    }
}

/// The late pass: discovers repeated aggregate families and rewrites the
/// applicable ones through one shared materialization. An `Err` means the
/// rewriter produced a node list that fails the plan invariants.
pub(crate) fn optimize_shared_subplan_materialization_without_scalar_derivation(
    plan: QueryPlan,
    options: &SharedSubplanMaterializationOptions,
) -> Result<SharedSubplanMaterializationResult, QueryError> {
    if !options.enabled {
        return Ok(SharedSubplanMaterializationResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: SharedSubplanMaterializationDiagnostics::default(),
        });
    }

    // Scalar derivation runs as its own earlier pass, so this result stands in
    // for it with no change and no rows; the composition below still merges
    // its fields as if it had run here.
    let scalar_derivation = SharedSubplanMaterializationResult {
        plan,
        change: NativeOptimizerChange::Unchanged,
        diagnostics: SharedSubplanMaterializationDiagnostics::default(),
    };
    let discovery = discover_shared_subplan_materialization_candidates(&scalar_derivation.plan);
    if discovery.families.is_empty() {
        if scalar_derivation.change.changed() || !scalar_derivation.diagnostics.is_empty() {
            return Ok(scalar_derivation);
        }
        return Ok(SharedSubplanMaterializationResult {
            plan: scalar_derivation.plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: discovery.diagnostics,
        });
    }

    let root_id = scalar_derivation.plan.root_id();
    let mut rewriter =
        SharedSubplanMaterializationRewriter::new(&scalar_derivation.plan, discovery.families, discovery.diagnostics);
    let new_root = rewriter.rewrite_subtree(root_id);
    let mut diagnostics = scalar_derivation.diagnostics;
    diagnostics.merge(rewriter.diagnostics);
    let changed = rewriter.changed;
    let rewritten_plan =
        scalar_derivation.plan.rewrite(RewrittenNodes::rebuilt(rewriter.nodes, new_root, rewriter.memo))?;
    let change = if changed || scalar_derivation.change.changed() {
        NativeOptimizerChange::Changed
    } else {
        NativeOptimizerChange::Unchanged
    };
    diagnostics.applied_count = diagnostics.candidates.iter().filter(|candidate| candidate.applied).count();
    Ok(SharedSubplanMaterializationResult { plan: rewritten_plan, change, diagnostics })
}
