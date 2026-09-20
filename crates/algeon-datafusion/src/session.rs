//! Installation paths for caller-owned DataFusion session builders.
//!
//! | Entry point | Installs | Required input | Idempotency | Owns admission |
//! | --- | --- | --- | --- | --- |
//! | [`AlgeonGpuBackend::install_on`](crate::backend::AlgeonGpuBackend::install_on) | backend-bound optimizer, coverage | built backend | same backend | yes |
//! | [`AlgeonSessionStateBuilderExt::with_algeon_native`] | optimizer, coverage | [`AlgeonNativeOptimizerConfig`] | replaces existing Algeon rules | no |
//! | [`AlgeonSessionStateBuilderExt::try_with_cuvs_sql`] | cuVS planner, metadata, function family | none | identical installation is a no-op | no |
//! | [`AlgeonSessionStateBuilderExt::try_with_cuml_sql`] | cuML planner, metadata, function family | none | identical installation is a no-op | no |
//! | [`AlgeonSessionStateBuilderExt::try_with_cuml_sql_config`] | cuML planner, metadata, function family, models root | [`CumlSqlConfig`] | equal config is a no-op; conflicts error | no |
//! | [`AlgeonSessionStateBuilderExt::try_with_cugraph_sql`] | cuGraph planner, metadata, function family | [`CugraphSqlConfig`] | equal config is a no-op; conflicts error | no |
//!
//! The native physical optimizer uses two insertion points. A count-only probe
//! precedes DataFusion aggregate statistics, and the final native rewrite runs
//! after every other rule already in the pipeline and immediately before
//! DataFusion's final sanity check, so the DataFusion CPU policy it enforces
//! and the sanity check both see the completed plan. Rules added to a builder
//! after installation run after that check. Embedded applications should use a
//! backend entry point; the raw extension methods are for planning, coverage,
//! and controlled composition.

use std::sync::Arc;

#[cfg(feature = "cugraph")]
use datafusion::execution::SessionState;
use datafusion::{
    catalog::Session,
    common::{DataFusionError, Result as DataFusionResult},
    execution::SessionStateBuilder,
    logical_expr::planner::RelationPlanner,
    physical_optimizer::{PhysicalOptimizerRule, optimizer::PhysicalOptimizer},
    physical_planner::ExtensionPlanner,
};

#[cfg(all(test, feature = "cugraph"))]
use crate::planner::AlgeonGpuExecutionMode;
#[cfg(feature = "cugraph")]
use crate::planner::RelationalNativeRewritePolicy;
use crate::{
    cugraph_sql::{CugraphExtensionPlanner, CugraphGpuFunctionFamily, CugraphSqlConfig, cugraph_relation_planner},
    cuml_sql::{
        CumlExtensionPlanner, CumlGpuFunctionFamily, CumlModelStore, CumlSqlConfig, cuml_gpu_function_availability,
        cuml_relation_planner_with_store,
    },
    cuvs_sql::{CuvsExtensionPlanner, CuvsGpuFunctionFamily, cuvs_gpu_function_availability, cuvs_relation_planner},
    error::{Error, ErrorCode},
    gpu_coverage::gpu_coverage_table_function,
    gpu_functions::{GpuFunctionCatalog, GpuFunctionFamily, install_gpu_function_family},
    native_exec::AlgeonNativeExecutionConfig,
    planner::{AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule, native::Invocation},
};

/// Public name and deduplication key for the final native rewrite rule.
pub const ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "algeon_datafusion_native";
/// Public name and deduplication key for the count-only native probe rule.
pub const ALGEON_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME: &str = "algeon_datafusion_native_count_only";
const DATAFUSION_AGGREGATE_STATISTICS_RULE_NAME: &str = "aggregate_statistics";
const DATAFUSION_FINAL_VALIDATION_RULE_NAME: &str = "SanityCheckPlan";
#[cfg(feature = "cugraph")]
const DATAFUSION_POST_OPTIMIZATION_FILTER_PUSHDOWN_RULE_NAME: &str = "FilterPushdown(Post)";

pub use crate::gpu_query_planner::AlgeonQueryPlanner;

type PhysicalOptimizerRules = Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>;

/// Extension methods for adding native GPU planning to a DataFusion session.
pub trait AlgeonSessionStateBuilderExt {
    /// Planning and coverage path: register algeon native rewriting before
    /// DataFusion's final physical-plan validation rule. This method does not
    /// create or own a GPU admission source, although a config obtained from
    /// [`AlgeonGpuBackend::optimizer_config`](crate::backend::AlgeonGpuBackend::optimizer_config)
    /// can already be bound to its backend. For canonical production
    /// installation, use
    /// [`AlgeonGpuBackend::install_on`](crate::backend::AlgeonGpuBackend::install_on).
    fn with_algeon_native(self, config: AlgeonNativeOptimizerConfig) -> Self;

    /// Install relation-aware cuVS SQL, metadata, and typed physical planning.
    /// Existing opaque query planners are rejected; callers that need other
    /// physical extension planners can compose them with the public
    /// [`AlgeonQueryPlanner`].
    fn try_with_cuvs_sql(self) -> DataFusionResult<Self>
    where Self: Sized;

    /// Install relation-aware cuML SQL, metadata, and typed physical planning
    /// without a models root, equivalent to
    /// [`Self::try_with_cuml_sql_config`] with [`CumlSqlConfig::default`];
    /// `cuml_forest_predict` then reports `model_root_unconfigured`.
    fn try_with_cuml_sql(self) -> DataFusionResult<Self>
    where Self: Sized;

    /// Install relation-aware cuML SQL, metadata, and typed physical planning.
    /// Existing opaque query planners are rejected.
    ///
    /// `config.model_store` names the models root for `cuml_forest_predict`;
    /// a configured root that does not resolve to a directory is rejected.
    /// Reinstalling an equal configuration is idempotent; reinstalling with a
    /// different model store (root or `max_model_bytes`) is rejected with a
    /// structured setup error.
    fn try_with_cuml_sql_config(self, config: CumlSqlConfig) -> DataFusionResult<Self>
    where Self: Sized;

    /// Install relation-aware cuGraph SQL, metadata, and typed physical planning.
    ///
    /// Reinstalling the same configuration is idempotent; reinstalling the
    /// `cugraph` family with a different descriptor projection, such as
    /// changed graph defaults, is rejected with a structured setup error.
    fn try_with_cugraph_sql(self, config: CugraphSqlConfig) -> DataFusionResult<Self>
    where Self: Sized;
}

impl AlgeonSessionStateBuilderExt for SessionStateBuilder {
    fn with_algeon_native(mut self, config: AlgeonNativeOptimizerConfig) -> Self {
        install_gpu_coverage_table_function(&mut self);
        let mut rules = take_physical_optimizer_rules(&mut self);
        install_algeon_native_rules(&mut rules, Arc::new(AlgeonNativePhysicalOptimizerRule::new(config)));
        self.with_physical_optimizer_rules(rules)
    }

    fn try_with_cuvs_sql(mut self) -> DataFusionResult<Self> {
        let availability = cuvs_gpu_function_availability(self.config().as_ref());
        install_gpu_sql_family(
            self,
            GpuSqlFamilyInstall {
                family_id: "cuvs",
                display_name: "cuVS",
                install_operation: "algeon_datafusion.cuvs_sql.install",
                extension_planner: Arc::new(CuvsExtensionPlanner),
                relation_planner: cuvs_relation_planner(),
                gpu_family: Arc::new(CuvsGpuFunctionFamily::new(availability)),
            },
        )
    }

    fn try_with_cuml_sql(self) -> DataFusionResult<Self> {
        self.try_with_cuml_sql_config(CumlSqlConfig::default())
    }

    fn try_with_cuml_sql_config(mut self, config: CumlSqlConfig) -> DataFusionResult<Self> {
        let availability = cuml_gpu_function_availability(self.config().as_ref());
        let model_store = config.model_store.map(CumlModelStore::try_new).transpose()?.map(Arc::new);
        install_gpu_sql_family(
            self,
            GpuSqlFamilyInstall {
                family_id: "cuml",
                display_name: "cuML",
                install_operation: "algeon_datafusion.cuml_sql.install",
                extension_planner: Arc::new(CumlExtensionPlanner),
                relation_planner: cuml_relation_planner_with_store(model_store.clone()),
                gpu_family: Arc::new(CumlGpuFunctionFamily::new(availability, model_store)),
            },
        )
    }

    fn try_with_cugraph_sql(self, config: CugraphSqlConfig) -> DataFusionResult<Self> {
        install_gpu_sql_family(
            self,
            GpuSqlFamilyInstall {
                family_id: "cugraph",
                display_name: "cuGraph",
                install_operation: "algeon_datafusion.cugraph_sql.install",
                extension_planner: Arc::new(CugraphExtensionPlanner),
                relation_planner: cugraph_relation_planner(config.clone()),
                gpu_family: Arc::new(CugraphGpuFunctionFamily::new(config)),
            },
        )
    }
}

struct GpuSqlFamilyInstall {
    family_id: &'static str,
    display_name: &'static str,
    install_operation: &'static str,
    extension_planner: Arc<dyn ExtensionPlanner + Send + Sync>,
    relation_planner: Arc<dyn RelationPlanner>,
    gpu_family: Arc<dyn GpuFunctionFamily>,
}

fn install_gpu_sql_family(
    mut builder: SessionStateBuilder,
    family: GpuSqlFamilyInstall,
) -> DataFusionResult<SessionStateBuilder> {
    let family_was_installed = builder
        .config()
        .as_ref()
        .and_then(|config| config.get_extension::<GpuFunctionCatalog>())
        .is_some_and(|catalog| catalog.contains_family(family.family_id));
    install_gpu_function_family(&mut builder, family.gpu_family)?;

    let planner_conflict = || {
        DataFusionError::from(
            Error::new(
                ErrorCode::InvalidRequest,
                family.install_operation,
                format!("cannot install {} SQL over an existing query planner", family.display_name),
            )
            .with_log_safe_fact("reason", "query_planner_conflict"),
        )
    };
    let query_planner = match builder.query_planner().as_ref() {
        None => AlgeonQueryPlanner::default(),
        Some(planner) => (planner.as_ref() as &dyn std::any::Any)
            .downcast_ref::<AlgeonQueryPlanner>()
            .cloned()
            .ok_or_else(planner_conflict)?,
    };
    if family_was_installed {
        let has_relation_planner = query_planner.family_relation_planner(family.family_id).is_some_and(|installed| {
            builder.relation_planners().as_ref().is_some_and(|relation_planners| {
                relation_planners.iter().any(|planner| Arc::ptr_eq(planner, installed))
            })
        });
        if !query_planner.contains_family(family.family_id) || !has_relation_planner {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                family.install_operation,
                format!(
                    "inconsistent {} SQL installation: catalog contains the {} family but its planners are missing",
                    family.display_name, family.display_name
                ),
            )
            .with_log_safe_fact("reason", "inconsistent_installation")
            .into());
        }
        return Ok(builder);
    }
    let query_planner =
        query_planner.with_family(family.family_id, family.extension_planner, Arc::clone(&family.relation_planner));
    builder.relation_planners().get_or_insert_with(Vec::new).insert(0, family.relation_planner);
    Ok(builder.with_query_planner(Arc::new(query_planner)))
}

#[cfg(feature = "cugraph")]
pub(crate) fn session_state_with_native_execution_config(
    state: &SessionState,
    native_execution_config: AlgeonNativeExecutionConfig,
) -> Option<SessionState> {
    let optimizer_config =
        installed_algeon_native_optimizer_config(state)?.with_native_execution_config(native_execution_config);
    let relational_rewrite_enabled =
        optimizer_config.relational_native_rewrite_policy() == RelationalNativeRewritePolicy::Enabled;
    let rule = Arc::new(AlgeonNativePhysicalOptimizerRule::new_cugraph_subplan(optimizer_config));
    let mut rules = state.physical_optimizers().to_vec();
    // The outer optimizer run revisits this subtree through `CugraphAlgoExec`'s
    // children. DataFusion forbids `JoinSelection` from swapping a hash join
    // after `FilterPushdown(Post)` attached its dynamic filter, so only the
    // outer run may construct dynamic filters.
    rules.retain(|existing| existing.name() != DATAFUSION_POST_OPTIMIZATION_FILTER_PUSHDOWN_RULE_NAME);
    install_algeon_native_rules(&mut rules, rule);
    Some(build_cugraph_subplan_state(state, rules, relational_rewrite_enabled))
}

#[cfg(feature = "cugraph")]
fn build_cugraph_subplan_state(
    state: &SessionState,
    rules: PhysicalOptimizerRules,
    relational_rewrite_enabled: bool,
) -> SessionState {
    // Graph edge and predicate sources feed graph construction or host-side
    // staging, so each nested scan must present one ordered run: a declared
    // fixture ordering (the source-sorted DBLP layout) only survives from a
    // single file group, and native graph ingress parallelizes reads internally.
    // `new_from_existing` disables default catalog creation because the nested
    // planner shares the parent catalog list. Preserve that property after
    // changing subplan parallelism: recreating the default catalog would
    // overwrite the parent's registered relation providers.
    let subplan_config = if relational_rewrite_enabled {
        state.config().clone().with_target_partitions(1)
    } else {
        state.config().clone()
    }
    .with_create_default_catalog_and_schema(false);
    SessionStateBuilder::new_from_existing(state.clone())
        .with_config(subplan_config)
        .with_physical_optimizer_rules(rules)
        .build()
}

/// The execution config that an extension planner copies into the GPU node it
/// builds for `session`: the installed native rule's config, or the bounded
/// planning defaults without one, with the session invocation's execution
/// inputs applied.
pub(crate) fn planning_native_execution_config(session: &dyn Session) -> AlgeonNativeExecutionConfig {
    let config = installed_algeon_native_optimizer_config(session)
        .map(|config| config.native_execution_config().clone())
        .unwrap_or_else(AlgeonNativeExecutionConfig::bounded_default);
    match Invocation::of_options(session.config_options()) {
        Some(invocation) => invocation.execution_config(config),
        None => config,
    }
}

pub(crate) fn installed_algeon_native_optimizer_config(session: &dyn Session) -> Option<AlgeonNativeOptimizerConfig> {
    installed_algeon_native_optimizer_config_from_rules(session.physical_optimizers())
}

fn installed_algeon_native_optimizer_config_from_rules(
    rules: &[Arc<dyn PhysicalOptimizerRule + Send + Sync>],
) -> Option<AlgeonNativeOptimizerConfig> {
    rules.iter().find_map(|rule| algeon_native_optimizer_config_from_rule(rule.as_ref()))
}

fn algeon_native_optimizer_config_from_rule(
    rule: &(dyn PhysicalOptimizerRule + Send + Sync),
) -> Option<AlgeonNativeOptimizerConfig> {
    let rule = rule as &dyn std::any::Any;
    rule.downcast_ref::<AlgeonNativePhysicalOptimizerRule>().map(|native| native.config().clone())
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

/// The one place that orders Algeon's rules in an optimizer pipeline. The
/// count-only probe runs before DataFusion's aggregate statistics; the final
/// rewrite, whose last step enforces the DataFusion CPU policy, runs after every
/// other rule in `rules` and immediately before DataFusion's sanity check.
fn install_algeon_native_rules(
    rules: &mut PhysicalOptimizerRules,
    native_rule: Arc<AlgeonNativePhysicalOptimizerRule>,
) {
    rules.retain(|existing| {
        existing.name() != ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME
            && existing.name() != ALGEON_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME
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

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
    use datafusion::{
        common::config::ConfigOptions,
        datasource::MemTable,
        physical_plan::{ExecutionPlan, empty::EmptyExec, union::UnionExec},
        prelude::{SessionConfig, SessionContext},
    };
    use futures::executor::block_on;

    use super::*;
    use crate::{
        backend::InstalledTestGpuBackend, error::DataFusionErrorExt, native_exec::QueryAdmissionSource,
        planner::DataFusionCpuExecutionPolicy,
    };

    /// Native lowering rejects struct columns, so this plan stays on DataFusion.
    fn cpu_only_plan(schema: SchemaRef) -> Arc<dyn ExecutionPlan> {
        UnionExec::try_new(vec![Arc::new(EmptyExec::new(Arc::clone(&schema))), Arc::new(EmptyExec::new(schema))])
            .expect("union over empty inputs")
    }

    fn struct_schema() -> SchemaRef {
        let fields = Fields::from(vec![Field::new("x", DataType::Int64, true)]);
        Arc::new(Schema::new(vec![Field::new("s", DataType::Struct(fields), true)]))
    }

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

        let first_optimizer = installed_algeon_native_optimizer_config(&first_state).unwrap();
        let second_optimizer = installed_algeon_native_optimizer_config(&second_state).unwrap();
        let first_source = first_optimizer.native_execution_config().query_admission_source().unwrap();
        let second_source = second_optimizer.native_execution_config().query_admission_source().unwrap();
        let backend_source = admission_source(&backend);
        assert!(first_source.same_backend(&second_source));
        assert!(first_source.same_backend(&backend_source));
    }

    /// A cuGraph input subplan is a child of the query root, so its final
    /// rewrite leaves the DataFusion CPU policy to the outer run.
    #[test]
    fn cugraph_subplanning_leaves_the_cpu_policy_to_the_outer_run() {
        let optimizer_config = AlgeonNativeOptimizerConfig::default()
            .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject);
        let state = SessionStateBuilder::new().with_algeon_native(optimizer_config.clone()).build();
        let nested =
            session_state_with_native_execution_config(&state, optimizer_config.native_execution_config().clone())
                .expect("nested cuGraph planning requires installed native selection");
        let final_rewrite = |state: &SessionState| {
            Arc::clone(
                state
                    .physical_optimizers()
                    .iter()
                    .find(|rule| rule.name() == ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME)
                    .expect("installed final native rewrite"),
            )
        };
        let cpu_plan = || cpu_only_plan(struct_schema());
        let options = ConfigOptions::default();

        assert!(final_rewrite(&state).optimize(cpu_plan(), &options).is_err());
        assert!(final_rewrite(&nested).optimize(cpu_plan(), &options).is_ok());
    }

    /// A rule the builder holds when the native rules are installed runs
    /// before the final rewrite, so the rewrite's DataFusion CPU check sees
    /// that rule's output.
    #[tokio::test]
    async fn final_rewrite_checks_the_output_of_rules_installed_before_it() {
        #[derive(Debug)]
        struct ReplaceWithCpuUnion;

        impl PhysicalOptimizerRule for ReplaceWithCpuUnion {
            fn optimize(
                &self,
                plan: Arc<dyn ExecutionPlan>,
                _config: &ConfigOptions,
            ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
                Ok(cpu_only_plan(plan.schema()))
            }

            fn name(&self) -> &str {
                "replace_with_cpu_union"
            }

            fn schema_check(&self) -> bool {
                true
            }
        }

        let state = SessionStateBuilder::new_with_default_features()
            .with_physical_optimizer_rule(Arc::new(ReplaceWithCpuUnion))
            .with_algeon_native(
                AlgeonNativeOptimizerConfig::default()
                    .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject),
            )
            .build();
        let rule_names = state.physical_optimizers().iter().map(|rule| rule.name()).collect::<Vec<_>>();
        assert_eq!(
            rule_names[rule_names.len() - 3..],
            [
                "replace_with_cpu_union",
                ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
                DATAFUSION_FINAL_VALIDATION_RULE_NAME
            ]
        );

        let failure = SessionContext::new_with_state(state)
            .sql("SELECT named_struct('x', 1) AS s")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .expect_err("the final rewrite rejects the CPU plan the earlier rule produced");
        let error = failure.find_algeon_error().expect("typed adapter error");
        assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        assert_eq!(error.fact_value("physical_plan_root"), Some("UnionExec"));
    }

    #[test]
    fn cugraph_subplanning_leaves_dynamic_filter_pushdown_to_the_outer_run() {
        let optimizer_config = AlgeonNativeOptimizerConfig::default()
            .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject);
        let state =
            SessionStateBuilder::new_with_default_features().with_algeon_native(optimizer_config.clone()).build();
        let nested =
            session_state_with_native_execution_config(&state, optimizer_config.native_execution_config().clone())
                .expect("nested cuGraph planning requires installed native selection");
        let is_post_pushdown = |rule: &Arc<dyn PhysicalOptimizerRule + Send + Sync>| {
            rule.name() == DATAFUSION_POST_OPTIMIZATION_FILTER_PUSHDOWN_RULE_NAME
        };

        assert!(state.physical_optimizers().iter().any(is_post_pushdown));
        assert!(!nested.physical_optimizers().iter().any(is_post_pushdown));
    }

    #[test]
    fn cugraph_subplanning_plans_single_group_scans_for_declared_orderings() {
        let optimizer_config = AlgeonNativeOptimizerConfig::default()
            .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject);
        let state = SessionStateBuilder::new().with_algeon_native(optimizer_config.clone()).build();
        let nested =
            session_state_with_native_execution_config(&state, optimizer_config.native_execution_config().clone())
                .expect("nested cuGraph planning requires installed native selection");
        assert_eq!(nested.config().target_partitions(), 1);
        // The caller's own session keeps whatever parallelism it configured.
        assert!(state.config().target_partitions() >= 1);
    }

    #[test]
    fn functions_only_cugraph_subplanning_preserves_parent_parallelism() {
        let optimizer_config =
            AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::FunctionsOnly);
        let state = SessionStateBuilder::new()
            .with_config(SessionConfig::new().with_target_partitions(4))
            .with_algeon_native(optimizer_config.clone())
            .build();
        let nested =
            session_state_with_native_execution_config(&state, optimizer_config.native_execution_config().clone())
                .expect("nested cuGraph planning requires installed native selection");

        assert_eq!(nested.config().target_partitions(), 4);
    }

    #[test]
    fn cugraph_subplanning_preserves_parent_catalog_relations() {
        let optimizer_config = AlgeonNativeOptimizerConfig::default()
            .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject);
        let state =
            SessionStateBuilder::new_with_default_features().with_algeon_native(optimizer_config.clone()).build();
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
            block_on(ctx.table("graph_edges")).is_ok(),
            "cuGraph subplanning must not overwrite the parent catalog"
        );
    }
}
