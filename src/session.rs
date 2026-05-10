use std::sync::Arc;

use datafusion::{
    execution::SessionStateBuilder,
    physical_optimizer::{PhysicalOptimizerRule, optimizer::PhysicalOptimizer},
};

use crate::planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig};

pub const CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_cudf_native";
pub const CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_cudf_native_count_only";
const DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME: &str = "aggregate_statistics";
const DATAFUSION_FINAL_VALIDATION_RULE_NAME: &str = "SanityCheckPlan";

pub type PhysicalOptimizerRules = Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>;

/// Return DataFusion's default physical optimizer rules with the cudf native
/// rewrite inserted before DataFusion's final plan validation rule.
pub fn cudf_native_physical_optimizer_rules(config: CudfOptimizerConfig) -> PhysicalOptimizerRules {
    let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(config));
    cudf_native_physical_optimizer_rules_with_rule(rule)
}

/// Return DataFusion's default physical optimizer rules with a caller-retained
/// cudf native rule inserted before DataFusion's final plan validation rule.
pub fn cudf_native_physical_optimizer_rules_with_rule(
    rule: Arc<CudfNativePhysicalOptimizerRule>,
) -> PhysicalOptimizerRules {
    let mut rules = PhysicalOptimizer::new().rules;
    insert_cudf_native_rules_before_final_validation(&mut rules, rule);
    rules
}

pub trait CudfSessionStateBuilderExt {
    /// Register cudf native rewriting before DataFusion's final physical-plan
    /// validation rule.
    fn with_cudf_native(self, config: CudfOptimizerConfig) -> Self;

    /// Register a caller-retained cudf native rule before DataFusion's final
    /// physical-plan validation rule.
    fn with_cudf_native_rule(self, rule: Arc<CudfNativePhysicalOptimizerRule>) -> Self;
}

impl CudfSessionStateBuilderExt for SessionStateBuilder {
    fn with_cudf_native(self, config: CudfOptimizerConfig) -> Self {
        self.with_cudf_native_rule(Arc::new(CudfNativePhysicalOptimizerRule::new(config)))
    }

    fn with_cudf_native_rule(mut self, rule: Arc<CudfNativePhysicalOptimizerRule>) -> Self {
        let mut rules = take_physical_optimizer_rules(&mut self);
        insert_cudf_native_rules_before_final_validation(&mut rules, rule);
        self.with_physical_optimizer_rules(rules)
    }
}

fn take_physical_optimizer_rules(builder: &mut SessionStateBuilder) -> PhysicalOptimizerRules {
    let mut rules = builder
        .physical_optimizers()
        .take()
        .map(|optimizer| optimizer.rules)
        .unwrap_or_else(|| PhysicalOptimizer::new().rules);

    if let Some(pending_rules) = builder.physical_optimizer_rules().take() {
        rules.extend(pending_rules);
    }

    rules
}

fn insert_cudf_native_rules_before_final_validation(
    rules: &mut PhysicalOptimizerRules,
    native_rule: Arc<CudfNativePhysicalOptimizerRule>,
) {
    rules.retain(|existing| {
        existing.name() != CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME
    });
    let count_only_rule = Arc::new(native_rule.count_only_probe_rule());
    let insertion_position =
        rules.iter().position(|existing| existing.name() == DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME).unwrap_or(0);
    rules.insert(insertion_position, count_only_rule);
    let final_validation = rules
        .iter()
        .rposition(|existing| existing.name() == DATAFUSION_FINAL_VALIDATION_RULE_NAME)
        .map(|position| rules.remove(position));
    rules.push(native_rule);
    if let Some(final_validation) = final_validation {
        rules.push(final_validation);
    }
}
