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

/// Diagnostics for shared-subplan materialization.
pub(crate) struct SharedSubplanMaterializationResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: SharedSubplanMaterializationDiagnostics,
}

pub(crate) fn optimize_scalar_aggregate_from_grouped_aggregate_if_enabled(
    plan: QueryPlan,
    options: &SharedSubplanMaterializationOptions,
) -> SharedSubplanMaterializationResult {
    if options.enabled {
        optimize_scalar_aggregate_from_grouped_aggregate(plan)
    } else {
        SharedSubplanMaterializationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: SharedSubplanMaterializationDiagnostics::default(),
        }
    }
}

pub(crate) fn optimize_shared_subplan_materialization_without_scalar_derivation(
    plan: QueryPlan,
    options: &SharedSubplanMaterializationOptions,
) -> SharedSubplanMaterializationResult {
    if !options.enabled {
        return SharedSubplanMaterializationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: SharedSubplanMaterializationDiagnostics::default(),
        };
    }

    let scalar_derivation = SharedSubplanMaterializationResult {
        plan,
        change: super::NativeOptimizerChange::Unchanged,
        diagnostics: SharedSubplanMaterializationDiagnostics::default(),
    };
    let discovery = discover_shared_subplan_materialization_candidates(&scalar_derivation.plan);
    if discovery.families.is_empty() {
        if scalar_derivation.change.changed() || !scalar_derivation.diagnostics.is_empty() {
            return scalar_derivation;
        }
        return SharedSubplanMaterializationResult {
            plan: scalar_derivation.plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: discovery.diagnostics,
        };
    }

    let root_id = scalar_derivation.plan.root_id();
    let sink = scalar_derivation.plan.sink().clone();
    let mut rewriter =
        SharedSubplanMaterializationRewriter::new(&scalar_derivation.plan, discovery.families, discovery.diagnostics);
    let new_root = rewriter.rewrite_subtree(root_id);
    let mut diagnostics = scalar_derivation.diagnostics;
    diagnostics.merge(rewriter.diagnostics);
    let changed = rewriter.changed;
    let rewritten_plan = rewrite::finish_rewritten_plan(rewriter.nodes, new_root, &sink)
        .with_keyed_hash_join_strategy_options_from(&scalar_derivation.plan);
    let change = if changed || scalar_derivation.change.changed() {
        super::NativeOptimizerChange::Changed
    } else {
        super::NativeOptimizerChange::Unchanged
    };
    diagnostics.applied_count = diagnostics.candidates.iter().filter(|candidate| candidate.applied).count();
    SharedSubplanMaterializationResult { plan: rewritten_plan, change, diagnostics }
}
