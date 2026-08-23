//! Installation paths for caller-owned DataFusion session builders.
//!
//! | Entry point | Installs | Required input | Idempotency | Owns admission |
//! | --- | --- | --- | --- | --- |
//! | [`NexusGpuBackend::install_on`](crate::backend::NexusGpuBackend::install_on) | backend-bound optimizer, coverage | built backend | same backend | yes |
//! | [`NexusSessionStateBuilderExt::with_nexus_native`] | optimizer, coverage | [`NexusNativeOptimizerConfig`] | replaces existing Nexus rules | no |
//! | [`NexusSessionStateBuilderExt::try_with_cuvs_sql`] | cuVS planner, metadata, function family | none | identical installation is a no-op | no |
//! | [`NexusSessionStateBuilderExt::try_with_cugraph_sql`] | cuGraph function family | [`CugraphSqlConfig`] | equal config is a no-op; conflicts error | no |
//!
//! The native physical optimizer uses two insertion points. A count-only probe
//! precedes DataFusion aggregate statistics, and the final native rewrite
//! precedes DataFusion's final sanity check so rewritten plans remain
//! validated. Embedded applications should use a backend entry point; the raw
//! extension methods are for planning, coverage, and controlled composition.

use std::sync::Arc;

#[cfg(feature = "cugraph")]
use datafusion::execution::SessionState;
use datafusion::{
    catalog::Session,
    common::Result as DataFusionResult,
    execution::SessionStateBuilder,
    physical_optimizer::{PhysicalOptimizerRule, optimizer::PhysicalOptimizer},
};

#[cfg(feature = "cugraph")]
use crate::native_exec::NexusNativeExecutionConfig;
use crate::{
    cugraph_sql::{CugraphGpuFunctionFamily, CugraphSqlConfig, cugraph_table_functions},
    cuvs_sql::{CuvsGpuFunctionFamily, CuvsQueryPlanner, cuvs_gpu_function_availability, cuvs_relation_planner},
    gpu_coverage_sql::gpu_coverage_table_function,
    gpu_functions::{GpuFunctionCatalog, install_gpu_function_family},
    planner::{
        NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME, NexusFinalPlanRequirementRule, NexusNativeOptimizerConfig,
        NexusNativePhysicalOptimizerRule,
    },
};

/// Public name and deduplication key for the final native rewrite rule.
pub const NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_nexus_native";
/// Public name and deduplication key for the count-only native probe rule.
pub const NEXUS_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "datafusion_nexus_native_count_only";
const DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME: &str = "aggregate_statistics";
const DATAFUSION_FINAL_VALIDATION_RULE_NAME: &str = "SanityCheckPlan";

type PhysicalOptimizerRules = Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>;

/// Extension methods for adding native GPU planning to a DataFusion session.
pub trait NexusSessionStateBuilderExt {
    /// Planning and coverage path: register nexus native rewriting before
    /// DataFusion's final physical-plan validation rule. This method does not
    /// create or own a GPU admission source, although a config obtained from
    /// [`NexusGpuBackend::optimizer_config`](crate::backend::NexusGpuBackend::optimizer_config)
    /// can already be bound to its backend. For canonical production
    /// installation, use
    /// [`NexusGpuBackend::install_on`](crate::backend::NexusGpuBackend::install_on).
    fn with_nexus_native(self, config: NexusNativeOptimizerConfig) -> Self;

    /// Install relation-aware cuVS SQL, metadata, and typed physical planning.
    /// Existing opaque query planners are rejected; callers that need other
    /// physical extension planners can compose them with the public
    /// [`CuvsQueryPlanner`].
    fn try_with_cuvs_sql(self) -> DataFusionResult<Self>
    where Self: Sized;

    /// Install cuGraph SQL table functions and the family's immutable catalog
    /// snapshot.
    ///
    /// Reinstalling the same configuration is idempotent; reinstalling the
    /// `cugraph` family with a different descriptor projection, such as
    /// changed graph defaults, is rejected with a structured setup error.
    fn try_with_cugraph_sql(self, config: CugraphSqlConfig) -> DataFusionResult<Self>
    where Self: Sized;

    /// Advanced planning path: register a caller-retained nexus native rule
    /// before DataFusion's final physical-plan validation rule. Use it when
    /// the caller keeps the rule to read its structured planning reports
    /// (`last_report`, `last_planning_diagnostics`) after planning; ordinary
    /// installation uses [`Self::with_nexus_native`].
    fn with_nexus_native_rule(self, rule: Arc<NexusNativePhysicalOptimizerRule>) -> Self;
}

impl NexusSessionStateBuilderExt for SessionStateBuilder {
    fn with_nexus_native(self, config: NexusNativeOptimizerConfig) -> Self {
        self.with_nexus_native_rule(Arc::new(NexusNativePhysicalOptimizerRule::new(config)))
    }

    fn try_with_cuvs_sql(mut self) -> DataFusionResult<Self> {
        let catalog_has_cuvs = self
            .config()
            .as_ref()
            .and_then(|config| config.get_extension::<GpuFunctionCatalog>())
            .is_some_and(|catalog| catalog.contains_family("cuvs"));

        if catalog_has_cuvs {
            let is_cuvs_planner = self
                .query_planner()
                .as_ref()
                .is_some_and(|planner| (planner.as_ref() as &dyn std::any::Any).is::<CuvsQueryPlanner>());
            if !is_cuvs_planner {
                return Err(crate::error::Error::new(
                    crate::error::ErrorCode::InvalidRequest,
                    "datafusion_nexus.cuvs_sql.install",
                    "cannot install cuVS SQL over an existing query planner",
                )
                .with_log_safe_fact("reason", "query_planner_conflict")
                .into());
            }
            if let Some(relation_planners) = self.relation_planners().as_ref() {
                let has_cuvs_relation_planner =
                    relation_planners.iter().any(|p| format!("{p:?}").contains("CuvsRelationPlanner"));
                if !has_cuvs_relation_planner {
                    return Err(crate::error::Error::new(
                        crate::error::ErrorCode::InvalidRequest,
                        "datafusion_nexus.cuvs_sql.install",
                        "inconsistent cuVS SQL installation: catalog contains cuVS family but relation planner is missing",
                    )
                    .with_log_safe_fact("reason", "inconsistent_installation")
                    .into());
                }
            }
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

    fn try_with_cugraph_sql(mut self, config: CugraphSqlConfig) -> DataFusionResult<Self> {
        install_gpu_function_family(&mut self, Arc::new(CugraphGpuFunctionFamily::new(config.clone())))?;
        let functions = self.table_functions().get_or_insert_with(Default::default);
        for function in cugraph_table_functions(config) {
            functions.insert(function.name().to_string(), function);
        }
        Ok(self)
    }

    fn with_nexus_native_rule(mut self, rule: Arc<NexusNativePhysicalOptimizerRule>) -> Self {
        install_gpu_coverage_table_function(&mut self);
        let mut rules = take_physical_optimizer_rules(&mut self);
        insert_nexus_native_rules_before_final_validation(&mut rules, rule);
        self.with_physical_optimizer_rules(rules)
    }
}

#[cfg(feature = "cugraph")]
pub(crate) fn session_state_with_native_execution_config(
    state: &SessionState,
    native_execution_config: NexusNativeExecutionConfig,
) -> Option<SessionState> {
    let optimizer_config =
        installed_nexus_native_optimizer_config(state)?.with_native_execution_config(native_execution_config);
    let rule = Arc::new(NexusNativePhysicalOptimizerRule::new_cugraph_subplan(optimizer_config));
    let mut rules = state.physical_optimizers().to_vec();
    insert_nexus_native_rules_for_cugraph_subplan(&mut rules, rule);
    Some(build_cugraph_subplan_state(state, rules))
}

#[cfg(feature = "cugraph")]
fn build_cugraph_subplan_state(state: &SessionState, rules: PhysicalOptimizerRules) -> SessionState {
    // Graph edge and predicate sources feed graph construction or host-side
    // staging, so each nested scan must present one ordered run: a declared
    // fixture ordering (the source-sorted DBLP layout) only survives from a
    // single file group, and native graph ingress parallelizes reads internally.
    // `new_from_existing` disables default catalog creation because the nested
    // planner shares the parent catalog list. Preserve that property after
    // changing subplan parallelism: recreating the default catalog would
    // overwrite the parent's registered relation providers.
    let subplan_config = state.config().clone().with_target_partitions(1).with_create_default_catalog_and_schema(false);
    SessionStateBuilder::new_from_existing(state.clone())
        .with_config(subplan_config)
        .with_physical_optimizer_rules(rules)
        .build()
}

pub(crate) fn installed_nexus_native_optimizer_config(session: &dyn Session) -> Option<NexusNativeOptimizerConfig> {
    installed_nexus_native_optimizer_config_from_rules(session.physical_optimizers())
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
    remove_nexus_native_rules(rules);
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
    rules.push(final_plan_requirement);
    if let Some(final_validation) = final_validation {
        rules.push(final_validation);
    }
}

#[cfg(feature = "cugraph")]
fn insert_nexus_native_rules_for_cugraph_subplan(
    rules: &mut PhysicalOptimizerRules,
    native_rule: Arc<NexusNativePhysicalOptimizerRule>,
) {
    remove_nexus_native_rules(rules);
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

fn remove_nexus_native_rules(rules: &mut PhysicalOptimizerRules) {
    rules.retain(|existing| {
        existing.name() != NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != NEXUS_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME
    });
}

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::{datasource::MemTable, prelude::SessionContext};

    use super::*;
    use crate::{backend::InstalledTestGpuBackend, native_exec::QueryAdmissionSource};

    fn admission_source(backend: &InstalledTestGpuBackend) -> QueryAdmissionSource {
        backend
            .optimizer_config()
            .native_execution_config()
            .query_admission_source()
            .expect("installed backend must provide admission authority")
    }

    #[test]
    fn cugraph_installation_order_preserves_same_backend_admission_authority() {
        let backend = InstalledTestGpuBackend::new();

        let first_builder = SessionStateBuilder::new();
        let first_builder = backend.install_on(first_builder).expect("backend install");
        let first_builder = first_builder.try_with_cugraph_sql(CugraphSqlConfig::default()).expect("cugraph install");
        let first_state = first_builder.build();

        let second_builder = SessionStateBuilder::new();
        let second_builder = second_builder.try_with_cugraph_sql(CugraphSqlConfig::default()).expect("cugraph install");
        let second_builder = backend.install_on(second_builder).expect("backend install");
        let second_state = second_builder.build();

        assert!(backend.backend().is_session_state_bound(&first_state));
        assert!(backend.backend().is_session_state_bound(&second_state));

        let first_optimizer = installed_nexus_native_optimizer_config(&first_state).unwrap();
        let second_optimizer = installed_nexus_native_optimizer_config(&second_state).unwrap();
        let first_source = first_optimizer.native_execution_config().query_admission_source().unwrap();
        let second_source = second_optimizer.native_execution_config().query_admission_source().unwrap();
        let backend_source = admission_source(&backend);
        assert!(first_source.same_backend(&second_source));
        assert!(first_source.same_backend(&backend_source));
    }

    #[test]
    fn cugraph_subplanning_omits_final_plan_requirement_rule() {
        let optimizer_config = NexusNativeOptimizerConfig::default().with_no_datafusion_cpu_requirement();
        let state = SessionStateBuilder::new().with_nexus_native(optimizer_config.clone()).build();
        let nested =
            session_state_with_native_execution_config(&state, optimizer_config.native_execution_config().clone())
                .expect("nested cuGraph planning requires installed native selection");
        let rules = nested.physical_optimizers();

        assert!(rules.iter().any(|rule| rule.name() == NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME));
        assert!(!rules.iter().any(|rule| rule.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME));
    }

    #[test]
    fn cugraph_subplanning_plans_single_group_scans_for_declared_orderings() {
        let optimizer_config = NexusNativeOptimizerConfig::default().with_no_datafusion_cpu_requirement();
        let state = SessionStateBuilder::new().with_nexus_native(optimizer_config.clone()).build();
        let nested =
            session_state_with_native_execution_config(&state, optimizer_config.native_execution_config().clone())
                .expect("nested cuGraph planning requires installed native selection");
        assert_eq!(nested.config().target_partitions(), 1);
        // The caller's own session keeps whatever parallelism it configured.
        assert!(state.config().target_partitions() >= 1);
    }

    #[test]
    fn cugraph_subplanning_preserves_parent_catalog_relations() {
        let optimizer_config = NexusNativeOptimizerConfig::default().with_no_datafusion_cpu_requirement();
        let state =
            SessionStateBuilder::new_with_default_features().with_nexus_native(optimizer_config.clone()).build();
        let ctx = SessionContext::new_with_state(state);
        let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int32, false)]));
        let table = Arc::new(MemTable::try_new(schema, vec![Vec::new()]).expect("empty test table"));
        ctx.register_table("graph_edges", table).expect("test relation registers");

        let _nested = session_state_with_native_execution_config(
            &ctx.state(),
            optimizer_config.native_execution_config().clone(),
        )
        .expect("nested cuGraph planning requires installed native selection");

        assert!(
            futures::executor::block_on(ctx.table("graph_edges")).is_ok(),
            "cuGraph subplanning must not overwrite the parent catalog"
        );
    }
}
