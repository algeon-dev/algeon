//! Session-state helpers for installing datafusion-nexus public integrations.
//!
//! The native physical optimizer uses two insertion points. A count-only probe
//! is placed before DataFusion aggregate statistics so count queries can be
//! detected early. The final native rewrite is placed immediately before
//! DataFusion's final sanity check so rewritten plans are still validated.

use std::sync::Arc;

use datafusion::{
    common::Result as DataFusionResult,
    execution::{SessionState, SessionStateBuilder},
    physical_optimizer::{PhysicalOptimizerRule, optimizer::PhysicalOptimizer},
};

#[cfg(feature = "cugraph")]
use crate::graph_sql::{CugraphGpuFunctionFamily, CugraphSqlConfig, cugraph_table_functions};
#[cfg(feature = "cugraph")]
use crate::native_exec::NexusNativeExecutionConfig;
use crate::{
    cuvs_sql::{CuvsGpuFunctionFamily, CuvsQueryPlanner, cuvs_gpu_function_availability, cuvs_relation_planner},
    gpu_coverage_sql::gpu_coverage_table_function,
    gpu_functions::{GpuFunctionCatalog, gpu_metadata_table_functions, install_gpu_function_family},
    planner::{
        HostExecutionQosConfig, NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME, NEXUS_HOST_EXECUTION_QOS_RULE_NAME,
        NexusFinalPlanRequirementRule, NexusHostExecutionQosRule, NexusNativeOptimizerConfig,
        NexusNativePhysicalOptimizerRule,
    },
    session_config::DataFusionNexusSessionConfig,
};

/// Public name and deduplication key for the final native rewrite rule.
pub const NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_nexus_native";
/// Public name and deduplication key for the count-only native probe rule.
pub const NEXUS_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_nexus_native_count_only";
const DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME: &str = "aggregate_statistics";
const DATAFUSION_FINAL_VALIDATION_RULE_NAME: &str = "SanityCheckPlan";

pub type PhysicalOptimizerRules = Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>;

/// Return DataFusion's default physical optimizer rules with the nexus native
/// rewrite inserted before DataFusion's final plan validation rule.
pub fn nexus_native_physical_optimizer_rules(config: NexusNativeOptimizerConfig) -> PhysicalOptimizerRules {
    let rule = Arc::new(NexusNativePhysicalOptimizerRule::new(config));
    nexus_native_physical_optimizer_rules_with_rule(rule)
}

/// Return DataFusion's default physical optimizer rules with host execution
/// QoS inserted as the outer statement-final rule.
pub fn nexus_host_execution_qos_physical_optimizer_rules(config: HostExecutionQosConfig) -> PhysicalOptimizerRules {
    let mut rules = PhysicalOptimizer::new().rules;
    insert_nexus_host_execution_qos_rule_before_final_validation(
        &mut rules,
        Arc::new(NexusHostExecutionQosRule::new(config)),
    );
    rules
}

/// Return DataFusion's default physical optimizer rules with a caller-retained
/// nexus native rule inserted before DataFusion's final plan validation rule.
pub fn nexus_native_physical_optimizer_rules_with_rule(
    rule: Arc<NexusNativePhysicalOptimizerRule>,
) -> PhysicalOptimizerRules {
    let mut rules = PhysicalOptimizer::new().rules;
    insert_nexus_native_rules_before_final_validation(&mut rules, rule);
    rules
}

/// Extension methods for adding native GPU planning to a DataFusion session.
pub trait NexusSessionStateBuilderExt {
    /// Register nexus native rewriting before DataFusion's final physical-plan
    /// validation rule.
    fn with_nexus_native(self, config: NexusNativeOptimizerConfig) -> Self;

    /// Register host-side statement QoS independently of relational native
    /// selection. The rule wraps final DataFusion and mixed plans once.
    fn with_host_execution_qos(self, config: HostExecutionQosConfig) -> Self;

    /// Install relation-aware cuVS SQL, metadata, and typed physical planning.
    /// Existing opaque query planners are rejected; callers that need other
    /// physical extension planners can compose them with the public
    /// [`CuvsQueryPlanner`].
    fn try_with_cuvs_sql(self) -> DataFusionResult<Self>
    where Self: Sized;

    #[cfg(feature = "cugraph")]
    fn with_cugraph_sql(self, config: CugraphSqlConfig) -> Self;

    /// Register a caller-retained nexus native rule before DataFusion's final
    /// physical-plan validation rule.
    fn with_nexus_native_rule(self, rule: Arc<NexusNativePhysicalOptimizerRule>) -> Self;
}

impl NexusSessionStateBuilderExt for SessionStateBuilder {
    fn with_nexus_native(self, config: NexusNativeOptimizerConfig) -> Self {
        self.with_nexus_native_rule(Arc::new(NexusNativePhysicalOptimizerRule::new(config)))
    }

    fn with_host_execution_qos(mut self, config: HostExecutionQosConfig) -> Self {
        let mut rules = take_physical_optimizer_rules(&mut self);
        insert_nexus_host_execution_qos_rule_before_final_validation(
            &mut rules,
            Arc::new(NexusHostExecutionQosRule::new(config)),
        );
        self.with_physical_optimizer_rules(rules)
    }

    fn try_with_cuvs_sql(mut self) -> DataFusionResult<Self> {
        // A cloned/session-rebuilt state can retain the cuVS planner and its
        // immutable catalog family. Treat that exact installation as a no-op
        // before checking the otherwise-exclusive opaque query planner.
        if self
            .config()
            .as_ref()
            .and_then(|config| config.get_extension::<GpuFunctionCatalog>())
            .is_some_and(|catalog| catalog.contains_family("cuvs"))
        {
            return Ok(self);
        }
        if self.query_planner().is_some() {
            return Err(crate::error::Error::new(
                crate::error::ErrorCode::InvalidRequest,
                "datafusion_nexus.cuvs_sql.install",
                "cannot install cuVS SQL over an existing query planner",
            )
            .with_log_safe_fact("reason", "query_planner_conflict")
            .into());
        }
        self.relation_planners().get_or_insert_with(Vec::new).insert(0, cuvs_relation_planner());
        let availability = cuvs_gpu_function_availability(self.config().as_ref());
        install_gpu_function_family(&mut self, Arc::new(CuvsGpuFunctionFamily::new(availability)))?;
        Ok(self.with_query_planner(Arc::new(CuvsQueryPlanner::default())))
    }

    #[cfg(feature = "cugraph")]
    fn with_cugraph_sql(mut self, config: CugraphSqlConfig) -> Self {
        let config = cugraph_config_with_builder_native_admission(&mut self, config);
        let functions = self.table_functions().get_or_insert_with(Default::default);
        for function in cugraph_table_functions(config.clone()) {
            functions.insert(function.name().to_string(), function);
        }
        install_gpu_function_family(&mut self, Arc::new(CugraphGpuFunctionFamily::new(config)))
            .expect("cuGraph GPU function catalog installation must be valid");
        self
    }

    fn with_nexus_native_rule(mut self, rule: Arc<NexusNativePhysicalOptimizerRule>) -> Self {
        ensure_datafusion_nexus_session_config(&mut self, rule.config());
        refresh_cuvs_gpu_function_family_availability(&mut self)
            .expect("cuVS GPU function catalog refresh must be valid");
        install_gpu_coverage_table_function(&mut self);
        let mut rules = take_physical_optimizer_rules(&mut self);
        insert_nexus_native_rules_before_final_validation(&mut rules, rule);
        self.with_physical_optimizer_rules(rules)
    }
}

pub(crate) fn install_nexus_native_rule_with_session_config(
    mut builder: SessionStateBuilder,
    rule: Arc<NexusNativePhysicalOptimizerRule>,
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
    refresh_cuvs_gpu_function_family_availability(&mut builder)?;
    install_gpu_coverage_table_function(&mut builder);
    let mut rules = take_physical_optimizer_rules(&mut builder);
    insert_nexus_native_rules_before_final_validation(&mut rules, rule);
    Ok(builder.with_physical_optimizer_rules(rules))
}

#[cfg(feature = "cugraph")]
pub(crate) fn session_state_with_native_execution_config(
    state: &SessionState,
    native_execution_config: NexusNativeExecutionConfig,
) -> Option<SessionState> {
    let optimizer_config =
        installed_nexus_native_optimizer_config(state)?.with_native_execution_config(native_execution_config);
    let rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config));
    let mut rules = state.physical_optimizers().to_vec();
    insert_nexus_native_rules_for_cugraph_subplan(&mut rules, rule);
    Some(SessionStateBuilder::new_from_existing(state.clone()).with_physical_optimizer_rules(rules).build())
}

pub(crate) fn installed_nexus_native_optimizer_config(state: &SessionState) -> Option<NexusNativeOptimizerConfig> {
    installed_nexus_native_optimizer_config_from_rules(state.physical_optimizers())
}

#[cfg(feature = "server")]
pub(crate) fn installed_host_execution_qos_config(state: &SessionState) -> Option<HostExecutionQosConfig> {
    state.physical_optimizers().iter().find_map(|rule| {
        let rule = rule.as_ref() as &dyn std::any::Any;
        rule.downcast_ref::<NexusHostExecutionQosRule>().map(|host_execution_qos| host_execution_qos.config().clone())
    })
}

#[cfg(feature = "cugraph")]
fn cugraph_config_with_builder_native_admission(
    builder: &mut SessionStateBuilder,
    config: CugraphSqlConfig,
) -> CugraphSqlConfig {
    if config.native_execution_config.has_query_admission_authority() {
        return config;
    }
    let Some(optimizer_config) = installed_nexus_native_optimizer_config_from_builder(builder) else {
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
fn installed_nexus_native_optimizer_config_from_builder(
    builder: &mut SessionStateBuilder,
) -> Option<NexusNativeOptimizerConfig> {
    if let Some(config) = builder
        .physical_optimizers()
        .as_ref()
        .and_then(|optimizer| installed_nexus_native_optimizer_config_from_rules(&optimizer.rules))
    {
        return Some(config);
    }
    builder
        .physical_optimizer_rules()
        .as_ref()
        .and_then(|rules| installed_nexus_native_optimizer_config_from_rules(rules))
}

fn installed_nexus_native_optimizer_config_from_rules(
    rules: &[Arc<dyn PhysicalOptimizerRule + Send + Sync>],
) -> Option<NexusNativeOptimizerConfig> {
    rules.iter().find_map(|rule| nexus_native_optimizer_config_from_rule(rule.as_ref()))
}

fn nexus_native_optimizer_config_from_rule(
    rule: &(dyn PhysicalOptimizerRule + Send + Sync),
) -> Option<NexusNativeOptimizerConfig> {
    let rule = rule as &dyn std::any::Any;
    rule.downcast_ref::<NexusNativePhysicalOptimizerRule>().map(|native| native.config().clone())
}

fn ensure_datafusion_nexus_session_config(builder: &mut SessionStateBuilder, config: &NexusNativeOptimizerConfig) {
    let session_config = builder.config().get_or_insert_with(Default::default);
    if session_config.options().extensions.get::<DataFusionNexusSessionConfig>().is_none() {
        session_config.options_mut().extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(config));
    }
}

/// Rebuild only the cuVS family after native session options become known.
///
/// `try_with_cuvs_sql` may run before a backend or native optimizer attaches
/// its session extension. The unified catalog snapshot must therefore be
/// refreshed once the final memory-control mode is available; otherwise a
/// bounded session could incorrectly advertise cuVS as plannable.
fn refresh_cuvs_gpu_function_family_availability(builder: &mut SessionStateBuilder) -> DataFusionResult<()> {
    let availability = cuvs_gpu_function_availability(builder.config().as_ref());
    let Some(prior) = builder.config().as_ref().and_then(|config| config.get_extension::<GpuFunctionCatalog>()) else {
        return Ok(());
    };
    let mut families = prior.families().to_vec();
    let Some(position) = families.iter().position(|family| family.family_id() == "cuvs") else {
        return Ok(());
    };
    families[position] = Arc::new(CuvsGpuFunctionFamily::new(availability));
    let catalog = Arc::new(GpuFunctionCatalog::try_new(families)?);
    let config = builder.config().get_or_insert_with(Default::default);
    config.set_extension(Arc::clone(&catalog));
    let functions = builder.table_functions().get_or_insert_with(Default::default);
    for function in gpu_metadata_table_functions(catalog) {
        functions.insert(function.name().to_string(), function);
    }
    Ok(())
}

fn install_gpu_coverage_table_function(builder: &mut SessionStateBuilder) {
    let function = gpu_coverage_table_function();
    builder.table_functions().get_or_insert_with(Default::default).insert(function.name().to_string(), function);
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

fn insert_nexus_native_rules_before_final_validation(
    rules: &mut PhysicalOptimizerRules,
    native_rule: Arc<NexusNativePhysicalOptimizerRule>,
) {
    let host_execution_qos_rule =
        rules.iter().find(|existing| existing.name() == NEXUS_HOST_EXECUTION_QOS_RULE_NAME).cloned();
    rules.retain(|existing| {
        existing.name() != NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != NEXUS_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != NEXUS_HOST_EXECUTION_QOS_RULE_NAME
            && existing.name() != NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME
    });
    let count_only_rule = Arc::new(native_rule.count_only_probe_rule());
    let insertion_position =
        rules.iter().position(|existing| existing.name() == DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME).unwrap_or(0);
    rules.insert(insertion_position, count_only_rule);
    let final_validation = rules
        .iter()
        .rposition(|existing| existing.name() == DATAFUSION_FINAL_VALIDATION_RULE_NAME)
        .map(|position| rules.remove(position));
    let final_plan_requirement = Arc::new(NexusFinalPlanRequirementRule::with_native_rule(
        native_rule.config().clone(),
        Arc::clone(&native_rule),
    ));
    rules.push(native_rule);
    if let Some(host_execution_qos_rule) = host_execution_qos_rule {
        rules.push(host_execution_qos_rule);
    }
    rules.push(final_plan_requirement);
    if let Some(final_validation) = final_validation {
        rules.push(final_validation);
    }
}

fn insert_nexus_host_execution_qos_rule_before_final_validation(
    rules: &mut PhysicalOptimizerRules,
    host_execution_qos_rule: Arc<NexusHostExecutionQosRule>,
) {
    let final_plan_requirement =
        rules.iter().find(|existing| existing.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME).cloned();
    rules.retain(|existing| {
        existing.name() != NEXUS_HOST_EXECUTION_QOS_RULE_NAME
            && existing.name() != NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME
    });
    let final_validation = rules
        .iter()
        .rposition(|existing| existing.name() == DATAFUSION_FINAL_VALIDATION_RULE_NAME)
        .map(|position| rules.remove(position));
    rules.push(host_execution_qos_rule);
    if let Some(final_plan_requirement) = final_plan_requirement {
        rules.push(final_plan_requirement);
    }
    if let Some(final_validation) = final_validation {
        rules.push(final_validation);
    }
}

#[cfg(feature = "cugraph")]
fn insert_nexus_native_rules_for_cugraph_subplan(
    rules: &mut PhysicalOptimizerRules,
    native_rule: Arc<NexusNativePhysicalOptimizerRule>,
) {
    rules.retain(|existing| {
        existing.name() != NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != NEXUS_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != NEXUS_HOST_EXECUTION_QOS_RULE_NAME
            && existing.name() != NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME
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

#[cfg(test)]
mod host_execution_qos_tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::planner::{HostExecutionQosConfig, NEXUS_HOST_EXECUTION_QOS_RULE_NAME};

    #[test]
    fn host_execution_qos_installs_without_relational_native_selection() {
        let state = SessionStateBuilder::new()
            .with_host_execution_qos(HostExecutionQosConfig::new(NonZeroUsize::new(1).unwrap()))
            .build();
        let rules = state.physical_optimizers();

        assert!(rules.iter().any(|rule| rule.name() == NEXUS_HOST_EXECUTION_QOS_RULE_NAME));
        assert!(!rules.iter().any(|rule| rule.name() == NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME));
    }

    #[test]
    fn host_execution_qos_is_between_native_selection_and_final_requirements() {
        let state = SessionStateBuilder::new()
            .with_host_execution_qos(HostExecutionQosConfig::new(NonZeroUsize::new(1).unwrap()))
            .with_nexus_native(NexusNativeOptimizerConfig::default().with_no_datafusion_cpu_requirement())
            .build();
        let rules = state.physical_optimizers();
        let native = rules
            .iter()
            .position(|rule| rule.name() == NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME)
            .expect("native selection must be installed");
        let host_qos = rules
            .iter()
            .position(|rule| rule.name() == NEXUS_HOST_EXECUTION_QOS_RULE_NAME)
            .expect("host QoS must be installed");
        let requirement = rules
            .iter()
            .position(|rule| rule.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME)
            .expect("final-plan requirement must be installed");

        assert!(native < host_qos && host_qos < requirement);
    }
}

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use nexus_query_engine::runtime::{
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
            NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let optimizer_config =
            NexusNativeOptimizerConfig::default().with_native_execution_config(native_execution_config.clone());
        let mut builder = SessionStateBuilder::new().with_nexus_native(optimizer_config);

        let cugraph_config = cugraph_config_with_builder_native_admission(&mut builder, CugraphSqlConfig::default());

        assert!(cugraph_config.native_execution_config.has_query_admission_authority());
        assert_eq!(
            cugraph_config.native_execution_config.memory_control_policy(),
            native_execution_config.memory_control_policy()
        );
    }

    #[test]
    fn cugraph_subplanning_omits_outer_statement_final_rules() {
        use std::num::NonZeroUsize;

        use crate::planner::{HostExecutionQosConfig, NEXUS_HOST_EXECUTION_QOS_RULE_NAME};

        let optimizer_config = NexusNativeOptimizerConfig::default().with_no_datafusion_cpu_requirement();
        let state = SessionStateBuilder::new()
            .with_nexus_native(optimizer_config.clone())
            .with_host_execution_qos(HostExecutionQosConfig::new(NonZeroUsize::new(1).unwrap()))
            .build();
        let nested =
            session_state_with_native_execution_config(&state, optimizer_config.native_execution_config().clone())
                .expect("nested cuGraph planning requires installed native selection");
        let rules = nested.physical_optimizers();

        assert!(rules.iter().any(|rule| rule.name() == NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME));
        assert!(!rules.iter().any(|rule| rule.name() == NEXUS_HOST_EXECUTION_QOS_RULE_NAME));
        assert!(!rules.iter().any(|rule| rule.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME));
    }
}
