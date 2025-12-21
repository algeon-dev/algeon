use std::sync::Arc;

use datafusion::{
    execution::SessionStateBuilder,
    physical_optimizer::{PhysicalOptimizerRule, optimizer::PhysicalOptimizer},
};

use crate::planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfPhysicalOptimizerRule};

pub const CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_cudf_native";
pub const CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_cudf_native_count_only";
pub const CUDF_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_cudf_gpu_island";
const DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME: &str = "aggregate_statistics";
const DATAFUSION_FINAL_VALIDATION_RULE_NAME: &str = "SanityCheckPlan";

pub type PhysicalOptimizerRules = Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>;

/// Return DataFusion's default physical optimizer rules with the cudf GPU island
/// rewrite inserted before DataFusion's final plan validation rule.
pub fn cudf_physical_optimizer_rules(config: CudfOptimizerConfig) -> PhysicalOptimizerRules {
    let rule = Arc::new(CudfPhysicalOptimizerRule::new(config));
    cudf_physical_optimizer_rules_with_rule(rule)
}

/// Return DataFusion's default physical optimizer rules with a caller-retained
/// cudf GPU island rule inserted before DataFusion's final plan validation rule.
pub fn cudf_physical_optimizer_rules_with_rule(rule: Arc<CudfPhysicalOptimizerRule>) -> PhysicalOptimizerRules {
    let mut rules = PhysicalOptimizer::new().rules;
    insert_cudf_rule_before_final_validation(&mut rules, rule);
    rules
}

/// Return DataFusion's default physical optimizer rules with caller-retained cudf
/// native and GPU-island rules inserted before final plan validation.
pub fn cudf_physical_optimizer_rules_with_retained_rules(
    native_rule: Option<Arc<CudfNativePhysicalOptimizerRule>>,
    gpu_island_rule: Arc<CudfPhysicalOptimizerRule>,
) -> PhysicalOptimizerRules {
    let mut rules = PhysicalOptimizer::new().rules;
    insert_cudf_rules_before_final_validation(&mut rules, native_rule, gpu_island_rule);
    rules
}

pub trait CudfSessionStateBuilderExt {
    /// Register cudf GPU island rewriting before DataFusion's final physical-plan
    /// validation rule.
    fn with_cudf_gpu_islands(self, config: CudfOptimizerConfig) -> Self;

    /// Register a caller-retained cudf GPU island rule before DataFusion's final
    /// physical-plan validation rule.
    fn with_cudf_gpu_island_rule(self, rule: Arc<CudfPhysicalOptimizerRule>) -> Self;

    /// Register caller-retained cudf native and GPU island rules before
    /// DataFusion's final physical-plan validation rule.
    fn with_cudf_native_and_gpu_island_rules(
        self,
        native_rule: Arc<CudfNativePhysicalOptimizerRule>,
        gpu_island_rule: Arc<CudfPhysicalOptimizerRule>,
    ) -> Self;
}

impl CudfSessionStateBuilderExt for SessionStateBuilder {
    fn with_cudf_gpu_islands(self, config: CudfOptimizerConfig) -> Self {
        self.with_cudf_gpu_island_rule(Arc::new(CudfPhysicalOptimizerRule::new(config)))
    }

    fn with_cudf_gpu_island_rule(mut self, rule: Arc<CudfPhysicalOptimizerRule>) -> Self {
        let mut rules = take_physical_optimizer_rules(&mut self);
        insert_cudf_rule_before_final_validation(&mut rules, rule);
        self.with_physical_optimizer_rules(rules)
    }

    fn with_cudf_native_and_gpu_island_rules(
        mut self,
        native_rule: Arc<CudfNativePhysicalOptimizerRule>,
        gpu_island_rule: Arc<CudfPhysicalOptimizerRule>,
    ) -> Self {
        let mut rules = take_physical_optimizer_rules(&mut self);
        insert_cudf_rules_before_final_validation(&mut rules, Some(native_rule), gpu_island_rule);
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

fn insert_cudf_rule_before_final_validation(rules: &mut PhysicalOptimizerRules, rule: Arc<CudfPhysicalOptimizerRule>) {
    let native_rule = rule
        .config()
        .native_execution_enabled()
        .then(|| Arc::new(CudfNativePhysicalOptimizerRule::new(rule.config().clone())));
    insert_cudf_rules_before_final_validation(rules, native_rule, rule);
}

fn insert_cudf_rules_before_final_validation(
    rules: &mut PhysicalOptimizerRules,
    native_rule: Option<Arc<CudfNativePhysicalOptimizerRule>>,
    gpu_island_rule: Arc<CudfPhysicalOptimizerRule>,
) {
    rules.retain(|existing| {
        existing.name() != CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != CUDF_PHYSICAL_OPTIMIZER_RULE_NAME
    });
    if let Some(native_rule) = native_rule.as_ref() {
        let count_only_rule = Arc::new(native_rule.count_only_probe_rule());
        let insertion_position =
            rules.iter().position(|existing| existing.name() == DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME).unwrap_or(0);
        rules.insert(insertion_position, count_only_rule);
    }
    let final_validation = rules
        .iter()
        .rposition(|existing| existing.name() == DATAFUSION_FINAL_VALIDATION_RULE_NAME)
        .map(|position| rules.remove(position));
    if let Some(native_rule) = native_rule {
        rules.push(native_rule);
    }
    rules.push(gpu_island_rule);
    if let Some(final_validation) = final_validation {
        rules.push(final_validation);
    }
}
