//! Session-state helpers for installing datafusion-nexus public integrations.
//!
//! The native physical optimizer uses two insertion points. A count-only probe
//! is placed before DataFusion aggregate statistics so count queries can be
//! detected early. The final native rewrite is placed immediately before
//! DataFusion's final sanity check so rewritten plans are still validated.

use std::sync::Arc;

#[cfg(any(feature = "server", feature = "cugraph"))]
use datafusion::execution::SessionState;
use datafusion::{
    common::Result as DataFusionResult,
    execution::SessionStateBuilder,
    physical_optimizer::{PhysicalOptimizerRule, optimizer::PhysicalOptimizer},
};

#[cfg(feature = "cugraph")]
use crate::NativeCudfExecutionConfig;
#[cfg(feature = "cugraph")]
use crate::graph_sql::{CugraphSqlConfig, cugraph_table_functions};
use crate::{
    planner::{
        CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig,
        CudfTableFormatGpuOnlyLeafGuardRule,
    },
    session_config::DataFusionNexusSessionConfig,
};

/// Public name and deduplication key for the final native rewrite rule.
pub const CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_nexus_native";
/// Public name and deduplication key for the count-only native probe rule.
pub const CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_nexus_native_count_only";
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

/// Extension methods for adding native GPU planning to a DataFusion session.
pub trait CudfSessionStateBuilderExt {
    /// Register cudf native rewriting before DataFusion's final physical-plan
    /// validation rule.
    fn with_cudf_native(self, config: CudfOptimizerConfig) -> Self;

    #[cfg(feature = "cugraph")]
    fn with_cugraph_sql(self, config: CugraphSqlConfig) -> Self;

    /// Register a caller-retained cudf native rule before DataFusion's final
    /// physical-plan validation rule.
    fn with_cudf_native_rule(self, rule: Arc<CudfNativePhysicalOptimizerRule>) -> Self;
}

impl CudfSessionStateBuilderExt for SessionStateBuilder {
    fn with_cudf_native(self, config: CudfOptimizerConfig) -> Self {
        self.with_cudf_native_rule(Arc::new(CudfNativePhysicalOptimizerRule::new(config)))
    }

    #[cfg(feature = "cugraph")]
    fn with_cugraph_sql(mut self, config: CugraphSqlConfig) -> Self {
        let config = cugraph_config_with_builder_native_admission(&mut self, config);
        let functions = self.table_functions().get_or_insert_with(Default::default);
        for function in cugraph_table_functions(config) {
            functions.insert(function.name().to_string(), function);
        }
        self
    }

    fn with_cudf_native_rule(mut self, rule: Arc<CudfNativePhysicalOptimizerRule>) -> Self {
        ensure_datafusion_nexus_session_config(&mut self, rule.config());
        let mut rules = take_physical_optimizer_rules(&mut self);
        insert_cudf_native_rules_before_final_validation(&mut rules, rule);
        self.with_physical_optimizer_rules(rules)
    }
}

pub(crate) fn install_cudf_native_rule_with_session_config(
    mut builder: SessionStateBuilder,
    rule: Arc<CudfNativePhysicalOptimizerRule>,
    session_extension: DataFusionNexusSessionConfig,
) -> DataFusionResult<SessionStateBuilder> {
    let session_config = builder.config().get_or_insert_with(Default::default);
    if session_extension.is_backend_admission_frozen()
        && let Some(existing) = session_config.options().extensions.get::<DataFusionNexusSessionConfig>()
    {
        if rule.config().native_execution_config().has_query_service_handle() {
            existing.validate_matches_query_service_policy(rule.config())?;
        } else {
            existing.validate_matches_optimizer_config(rule.config())?;
        }
    }
    session_config.options_mut().extensions.insert(session_extension);
    let mut rules = take_physical_optimizer_rules(&mut builder);
    insert_cudf_native_rules_before_final_validation(&mut rules, rule);
    Ok(builder.with_physical_optimizer_rules(rules))
}

#[cfg(feature = "cugraph")]
pub(crate) fn session_state_with_native_execution_config(
    state: &SessionState,
    native_execution_config: NativeCudfExecutionConfig,
) -> Option<SessionState> {
    let optimizer_config =
        installed_cudf_native_optimizer_config(state)?.with_native_execution_config(native_execution_config);
    let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let mut rules = state.physical_optimizers().to_vec();
    insert_cudf_native_rules_before_final_validation(&mut rules, rule);
    Some(SessionStateBuilder::new_from_existing(state.clone()).with_physical_optimizer_rules(rules).build())
}

#[cfg(any(feature = "server", feature = "cugraph"))]
pub(crate) fn installed_cudf_native_optimizer_config(state: &SessionState) -> Option<CudfOptimizerConfig> {
    installed_cudf_native_optimizer_config_from_rules(state.physical_optimizers())
}

#[cfg(feature = "cugraph")]
fn cugraph_config_with_builder_native_admission(
    builder: &mut SessionStateBuilder,
    config: CugraphSqlConfig,
) -> CugraphSqlConfig {
    if config.native_execution_config.has_query_admission_authority() {
        return config;
    }
    let Some(optimizer_config) = installed_cudf_native_optimizer_config_from_builder(builder) else {
        return config;
    };
    let native_execution_config = optimizer_config.native_execution_config();
    if native_execution_config.has_query_admission_authority() {
        config.with_native_execution_config(native_execution_config.clone())
    } else {
        config
    }
}

#[cfg(feature = "cugraph")]
fn installed_cudf_native_optimizer_config_from_builder(
    builder: &mut SessionStateBuilder,
) -> Option<CudfOptimizerConfig> {
    if let Some(config) = builder
        .physical_optimizers()
        .as_ref()
        .and_then(|optimizer| installed_cudf_native_optimizer_config_from_rules(&optimizer.rules))
    {
        return Some(config);
    }
    builder
        .physical_optimizer_rules()
        .as_ref()
        .and_then(|rules| installed_cudf_native_optimizer_config_from_rules(rules))
}

#[cfg(any(feature = "server", feature = "cugraph"))]
fn installed_cudf_native_optimizer_config_from_rules(
    rules: &[Arc<dyn PhysicalOptimizerRule + Send + Sync>],
) -> Option<CudfOptimizerConfig> {
    rules.iter().find_map(|rule| cudf_native_optimizer_config_from_rule(rule.as_ref()))
}

#[cfg(any(feature = "server", feature = "cugraph"))]
fn cudf_native_optimizer_config_from_rule(
    rule: &(dyn PhysicalOptimizerRule + Send + Sync),
) -> Option<CudfOptimizerConfig> {
    let rule = rule as &dyn std::any::Any;
    rule.downcast_ref::<CudfNativePhysicalOptimizerRule>().map(|native| native.config().clone())
}

fn ensure_datafusion_nexus_session_config(builder: &mut SessionStateBuilder, config: &CudfOptimizerConfig) {
    let session_config = builder.config().get_or_insert_with(Default::default);
    if session_config.options().extensions.get::<DataFusionNexusSessionConfig>().is_none() {
        session_config.options_mut().extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(config));
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
            && existing.name() != CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME
    });
    let count_only_rule = Arc::new(native_rule.count_only_probe_rule());
    let insertion_position =
        rules.iter().position(|existing| existing.name() == DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME).unwrap_or(0);
    rules.insert(insertion_position, count_only_rule);
    let final_validation = rules
        .iter()
        .rposition(|existing| existing.name() == DATAFUSION_FINAL_VALIDATION_RULE_NAME)
        .map(|position| rules.remove(position));
    let leaf_guard = Arc::new(CudfTableFormatGpuOnlyLeafGuardRule::with_native_rule(Arc::clone(&native_rule)));
    rules.push(native_rule);
    rules.push(leaf_guard);
    if let Some(final_validation) = final_validation {
        rules.push(final_validation);
    }
}

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use nexus_query_engine::{
        DeploymentIoCapability, GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryService,
        QueryServiceConfig, QueryServiceTenantConfig, TenantId,
    };

    use super::*;
    use crate::native_exec::QueryAdmissionSource;

    fn admission_only_source() -> QueryAdmissionSource {
        let tenant_id = TenantId::new("session-cugraph-test");
        let deployment = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let service = QueryService::new_admission_only(
            QueryServiceConfig::default(),
            vec![QueryServiceTenantConfig::new(tenant_id.clone(), 1024 * 1024)],
            deployment,
        )
        .expect("admission-only query service should build");
        QueryAdmissionSource::new(service, tenant_id)
    }

    #[test]
    fn cugraph_sql_config_inherits_installed_backend_admission() {
        let source = admission_only_source();
        let native_execution_config =
            NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let optimizer_config =
            CudfOptimizerConfig::default().with_native_execution_config(native_execution_config.clone());
        let mut builder = SessionStateBuilder::new().with_cudf_native(optimizer_config);

        let cugraph_config = cugraph_config_with_builder_native_admission(&mut builder, CugraphSqlConfig::default());

        assert!(cugraph_config.native_execution_config.has_query_admission_authority());
        assert_eq!(
            cugraph_config.native_execution_config.memory_control_policy(),
            native_execution_config.memory_control_policy()
        );
    }
}
