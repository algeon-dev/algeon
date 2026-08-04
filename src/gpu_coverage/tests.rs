use std::{any::Any, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::{
    common::{
        DataFusionError, Result as DataFusionResult,
        config::{ConfigEntry, ConfigExtension, ExtensionOptions},
    },
    execution::SessionStateBuilder,
    physical_expr::expressions::col,
    physical_plan::{
        ExecutionPlan,
        projection::{ProjectionExec, ProjectionExpr},
    },
    prelude::{SessionConfig, SessionContext},
};
#[cfg(feature = "cugraph")]
use nexus_query_engine::{expr::Expr, plan::ProjectionOutput};
use nexus_query_engine::{
    plan::{NativePlanCostReason, QueryPlanBuilder},
    source::{LocalParquetSource, SourceNode},
};

use super::{physical::candidate_outcomes_from_report, *};
#[cfg(feature = "cugraph")]
use crate::native_exec::NexusNativeComposedExec;
use crate::{
    capability::CapabilityReason,
    error::{Error, ErrorCode},
    final_plan_observation::observe_final_plan,
    native_exec::{NexusNativeExec, NexusNativeExecutionConfig},
    native_report::FinalPlanDisposition,
    planner::NexusNativeOptimizerConfig,
    report::{CandidateOutcome, GpuCandidate, NativeCandidateCostEvidence, PlanningReport},
    session::NexusSessionStateBuilderExt,
    session_config::DataFusionNexusSessionConfig,
};

fn test_execution_config() -> NexusNativeExecutionConfig {
    NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1)
}

fn bounded_optimizer_config() -> NexusNativeOptimizerConfig {
    NexusNativeOptimizerConfig::default().with_native_execution_config(NexusNativeExecutionConfig::bounded_default())
}

fn validation_context(config: NexusNativeOptimizerConfig) -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features().with_nexus_native(config).build();
    SessionContext::new_with_state(state)
}

/// Mimics an invocation-only adapter extension such as an attempt admission
/// binding. It deliberately exposes no declarative configuration entries.
#[derive(Debug, Clone)]
struct InvocationOnlyExtension;

impl ConfigExtension for InvocationOnlyExtension {
    const PREFIX: &'static str = "datafusion_nexus_test_invocation";
}

impl ExtensionOptions for InvocationOnlyExtension {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, _key: &str, _value: &str) -> DataFusionResult<()> {
        unreachable!("invocation-only extensions expose no settings")
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        Vec::new()
    }
}

fn one_column_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]))
}

fn native_exec_leaf() -> Arc<dyn ExecutionPlan> {
    let schema = one_column_schema();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/query-gpu-coverage-not-read.parquet",
        schema,
        vec!["key".to_string()],
    )));
    Arc::new(NexusNativeExec::try_new(builder.finish_with_host_sink(source), test_execution_config()).unwrap())
}

#[cfg(feature = "cugraph")]
fn native_edge_exec() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/query-gpu-coverage-edges-not-read.parquet",
        schema,
        vec!["src".to_string(), "dst".to_string()],
    )));
    Arc::new(NexusNativeExec::try_new(builder.finish_with_host_sink(source), test_execution_config()).unwrap())
}

#[cfg(feature = "cugraph")]
fn composed_exec_with_child(child: Arc<dyn ExecutionPlan>, input_name: impl Into<String>) -> Arc<dyn ExecutionPlan> {
    let schema = one_column_schema();
    let mut builder = QueryPlanBuilder::new();
    let source = builder
        .add_source(SourceNode::GpuResident(nexus_query_engine::source::GpuResidentSource::new(input_name, schema)));
    Arc::new(
        NexusNativeComposedExec::try_new(builder.finish_with_host_sink(source), vec![child], test_execution_config())
            .unwrap(),
    )
}

#[tokio::test]
async fn validate_query_rejects_multiple_statements_but_not_semicolon_literal() {
    let state = SessionStateBuilder::new()
        .with_nexus_native(NexusNativeOptimizerConfig::default().with_native_execution_config(test_execution_config()))
        .build();
    let ctx = SessionContext::new_with_state(state);

    let literal = validate_query(&ctx, "SELECT ';' AS marker").await.unwrap();
    assert!(!matches!(literal.overall, QueryGpuCoverageOverall::Rejected { .. }));

    let literal_function_name =
        validate_query(&ctx, "SELECT 'nexus_explain_coverage(''SELECT 1'')' AS marker").await.unwrap();
    assert!(!matches!(literal_function_name.overall, QueryGpuCoverageOverall::Rejected { .. }));

    let empty = validate_query(&ctx, "").await.unwrap();
    assert!(matches!(
        empty.overall,
        QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
            if reason_code == "empty_sql"
    ));
    let empty_json = empty.to_json();
    assert_eq!(empty_json["overall"]["retry_advice"]["availability"], "unavailable");
    assert_eq!(empty_json["overall"]["retry_advice"]["reason"], "no_sealed_attempt_report");
    assert!(empty_json["overall"].get("retryable").is_none());

    let rejected = validate_query(&ctx, "SELECT ';' AS marker; SELECT 1").await.unwrap();
    assert!(matches!(
        rejected.overall,
        QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
            if reason_code == "multi_statement_unsupported"
    ));

    let recursive_exists =
        validate_query(&ctx, "SELECT EXISTS (SELECT 1 FROM nexus_explain_coverage('SELECT 1')) AS recursive")
            .await
            .unwrap();
    assert!(matches!(
        recursive_exists.overall,
        QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
            if reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_array =
        validate_query(&ctx, "SELECT ARRAY[(SELECT count(*) FROM nexus_explain_coverage('SELECT 1'))] AS recursive")
            .await
            .unwrap();
    assert!(matches!(
        recursive_array.overall,
        QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
            if reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_pipe_where =
        validate_query(&ctx, "SELECT 1 |> WHERE EXISTS (SELECT 1 FROM nexus_explain_coverage('SELECT 1'))")
            .await
            .unwrap();
    assert!(matches!(
        recursive_pipe_where.overall,
        QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
            if reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_pipe_select = validate_query(
        &ctx,
        "SELECT 1 |> SELECT (SELECT count(*) FROM nexus_explain_coverage('SELECT 1')) AS recursive",
    )
    .await
    .unwrap();
    assert!(matches!(
        recursive_pipe_select.overall,
        QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
            if reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_unnest =
        validate_query(&ctx, "SELECT * FROM UNNEST(ARRAY[(SELECT count(*) FROM nexus_explain_coverage('SELECT 1'))])")
            .await
            .unwrap();
    assert!(matches!(
        recursive_unnest.overall,
        QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
            if reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_named_window = validate_query(
        &ctx,
        "SELECT row_number() OVER w AS rn \
             FROM (SELECT 1 AS x) t \
             WINDOW w AS (PARTITION BY (SELECT count(*) FROM nexus_explain_coverage('SELECT 1')))",
    )
    .await
    .unwrap();
    assert!(matches!(
        recursive_named_window.overall,
        QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
            if reason_code == "coverage_validator_recursion_unsupported"
    ));
}

#[tokio::test]
async fn validate_query_preflight_uses_session_sql_dialect() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().set_str("datafusion.sql_parser.dialect", "postgresql"))
        .with_nexus_native(NexusNativeOptimizerConfig::default().with_native_execution_config(test_execution_config()))
        .build();

    let rejection = preflight_single_query(&state, "SELECT $$nexus_explain_coverage('SELECT 1')$$ AS marker");
    assert!(rejection.is_none(), "postgresql dollar-quoted literal should not be rejected: {rejection:?}");
}

#[tokio::test]
async fn validate_query_reports_capability_misses_without_terminal_rejection() {
    let ctx = validation_context(bounded_optimizer_config());

    let cpu = validate_query(&ctx, "SELECT random() AS r").await.unwrap();
    assert_eq!(cpu.overall, QueryGpuCoverageOverall::CpuOnly);
    assert_eq!(cpu.final_disposition, FinalPlanDisposition::DataFusion);
    assert_eq!(cpu.domains.native_fragment_count, 0);
    assert!(cpu.candidate_outcomes.iter().any(|outcome| outcome.selection
        == QueryGpuCandidateSelectionOutcome::NotSupported
        && outcome.reason_code.as_deref() == Some(CapabilityReason::UnsupportedExpression.code())
        && outcome.reason_kind == Some(QueryGpuCandidateReasonKind::Expression)
        && outcome.remedy.as_ref().is_some_and(|remedy| remedy.code == "rewrite_expression")));

    let unsupported = validate_query(&ctx, "SELECT array[1, 2] AS values").await.unwrap();
    assert_eq!(unsupported.overall, QueryGpuCoverageOverall::CpuOnly);
    assert_eq!(unsupported.final_disposition, FinalPlanDisposition::DataFusion);
    assert_eq!(unsupported.domains.native_fragment_count, 0);
    assert_eq!(unsupported.domains.datafusion_boundary_count, 0);
    assert!(unsupported.candidate_outcomes.iter().any(|outcome| outcome.selection
        == QueryGpuCandidateSelectionOutcome::NotSupported
        && outcome.reason_code.as_deref() == Some(CapabilityReason::UnsupportedType.code())
        && outcome.reason_kind == Some(QueryGpuCandidateReasonKind::ArrowType)
        && outcome.remedy.as_ref().is_some_and(|remedy| remedy.code == "cast_or_exclude_type")));
}

#[test]
fn candidate_outcomes_retain_selected_candidates_without_selection_reasons() {
    let report = PlanningReport::from_outcomes([CandidateOutcome::Selected {
        candidate: GpuCandidate::new("ProjectionExec", 1),
    }]);

    let outcomes = candidate_outcomes_from_report(&report);

    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].selection, QueryGpuCandidateSelectionOutcome::Selected);
    assert_eq!(outcomes[0].detail, "ProjectionExec");
    assert_eq!(outcomes[0].candidate_family, "other_local_parquet_input");
    assert_eq!(outcomes[0].reason_code, None);
    assert_eq!(outcomes[0].reason_kind, None);
    assert_eq!(outcomes[0].category, None);
    assert_eq!(outcomes[0].remedy, None);
    assert_eq!(outcomes[0].cost_evidence, None);
}

#[test]
fn candidate_outcomes_classify_cost_decisions_as_cost() {
    let report = PlanningReport::from_outcomes([CandidateOutcome::NotSelectedByCost {
        candidate: GpuCandidate::new("ProjectionExec", 1),
        cost_reason: NativePlanCostReason::SmallSelectiveTopK,
        evidence: NativeCandidateCostEvidence {
            node_count: 1,
            source_count: 1,
            local_parquet_source_count: 1,
            selected_row_groups_known: true,
            selected_row_groups: Some(1),
            estimated_source_rows: Some(1),
            estimated_source_bytes: Some(1),
            selected_source_bytes_known: true,
            join_count: 0,
            residual_join_count: 0,
            aggregate_count: 0,
            grouped_aggregate_count: 0,
            sort_count: 0,
            limit_count: 0,
            min_topk_fetch: Some(1),
        },
    }]);

    let outcomes = candidate_outcomes_from_report(&report);

    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].selection, QueryGpuCandidateSelectionOutcome::NotSelectedByCost);
    assert_eq!(outcomes[0].reason_kind, Some(QueryGpuCandidateReasonKind::Cost));
    assert_eq!(outcomes[0].category.as_deref(), Some("cost"));
    assert_eq!(outcomes[0].reason_code.as_deref(), Some("native_small_selective_topk_not_selected_by_cost"));
    assert_eq!(outcomes[0].remedy.as_ref().map(|remedy| remedy.code), Some("keep_cpu_for_small_selective_work"));
}

#[test]
fn candidate_outcomes_retain_shape_families_for_unsupported_candidates() {
    let report = PlanningReport::from_outcomes([CandidateOutcome::NotSupported {
        candidate: GpuCandidate::new("UnsupportedNativePhysicalPlan", 1),
        capability_reason: CapabilityReason::UnsupportedExpression,
    }]);

    let outcomes = candidate_outcomes_from_report(&report);

    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].selection, QueryGpuCandidateSelectionOutcome::NotSupported);
    assert_eq!(outcomes[0].candidate_family, "unsupported_native_physical_plan");
    assert_eq!(outcomes[0].reason_kind, Some(QueryGpuCandidateReasonKind::Expression));
}

#[tokio::test]
async fn validate_query_preserves_the_installed_final_plan_requirement() {
    let ctx = validation_context(bounded_optimizer_config().with_no_datafusion_cpu_requirement());
    let coverage = validate_query(&ctx, "SELECT random() AS r").await.unwrap();

    assert!(
        matches!(
            coverage.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::PhysicalPlanning, .. }
        ),
        "coverage={coverage:?}"
    );
    assert_eq!(coverage.final_disposition, FinalPlanDisposition::Rejected);
    assert!(
        coverage
            .candidate_outcomes
            .iter()
            .any(|outcome| outcome.reason_code.as_deref() == Some("unsupported_expression"))
    );
    assert!(
        matches!(&coverage.overall, QueryGpuCoverageOverall::Rejected { detail, .. } if detail.contains("final physical plan")),
        "coverage={coverage:?}"
    );
    assert_eq!(coverage.validated_under.final_plan_requirements, ["no_datafusion_cpu"]);
}

#[tokio::test]
async fn validate_query_without_a_native_rule_keeps_the_default_session_cpu_baseline() {
    let ctx = SessionContext::new();

    let coverage = validate_query(&ctx, "SELECT random() AS r").await.unwrap();

    assert_eq!(coverage.overall, QueryGpuCoverageOverall::CpuOnly);
    assert!(!coverage.validated_under.native_optimizer_rule_installed);
    assert_eq!(coverage.validated_under.optimizer_config_source, "native_rule_absent");
    assert_eq!(coverage.validated_under.execution_mode, None);
    assert!(coverage.validated_under.final_plan_requirements.is_empty());
    assert!(coverage.candidate_outcomes.is_empty(), "coverage={coverage:?}");
}

#[tokio::test]
async fn validate_query_metadata_uses_effective_session_optimizer_options() {
    let optimizer_config = NexusNativeOptimizerConfig::default()
        .with_native_execution_config(test_execution_config())
        .with_no_datafusion_cpu_requirement();
    let mut session_config = SessionConfig::new();
    session_config
        .options_mut()
        .extensions
        .insert(DataFusionNexusSessionConfig::from_optimizer_config(&optimizer_config));
    // The deleted pipeline-mode session key is rejected; the single bounded
    // path projects the canonical execution-mode label.
    assert!(session_config.options_mut().set("datafusion_nexus.native.pipeline_execution_mode", "adaptive").is_err());
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_nexus_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);

    let coverage = validate_query(&ctx, "SELECT 1 AS value").await.unwrap();

    assert_eq!(coverage.validated_under.execution_mode.as_deref(), Some("bounded"));
    assert_eq!(coverage.validated_under.final_plan_requirements, ["no_datafusion_cpu"]);
}

#[tokio::test]
async fn validate_query_session_fingerprint_ignores_invocation_local_extensions() {
    let state = SessionStateBuilder::new_with_default_features().with_nexus_native(bounded_optimizer_config()).build();
    let baseline =
        validate_query(&SessionContext::new_with_state(state.clone()), "SELECT random() AS r").await.unwrap();

    let mut invocation_builder = SessionStateBuilder::new_from_existing(state);
    invocation_builder
        .config()
        .get_or_insert_with(Default::default)
        .options_mut()
        .extensions
        .insert(InvocationOnlyExtension);
    let invocation =
        validate_query(&SessionContext::new_with_state(invocation_builder.build()), "SELECT random() AS r")
            .await
            .unwrap();

    assert_eq!(
        baseline.validated_under.session_config_fingerprint, invocation.validated_under.session_config_fingerprint,
        "opaque invocation state must not change coverage evidence"
    );
}

#[tokio::test]
async fn validate_query_concurrent_calls_do_not_share_retained_rule_state() {
    let ctx = validation_context(bounded_optimizer_config());

    let (native, cpu) =
        tokio::join!(validate_query(&ctx, "SELECT 1 AS value"), validate_query(&ctx, "SELECT random() AS r"));
    let native = native.unwrap();
    let cpu = cpu.unwrap();

    assert_eq!(native.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
    assert_eq!(native.final_disposition, FinalPlanDisposition::Native);
    assert!(native.candidate_outcomes.iter().any(|outcome| {
        outcome.selection == QueryGpuCandidateSelectionOutcome::Selected
            && outcome.reason_code.is_none()
            && outcome.reason_kind.is_none()
            && !outcome.candidate_family.is_empty()
    }));
    assert!(native.input_physical_plan_root.is_some(), "coverage={native:?}");
    assert!(native.input_physical_plan_shape.is_some(), "coverage={native:?}");
    assert!(native.to_json()["candidate_outcomes"].as_array().is_some_and(|outcomes| {
        outcomes.iter().any(|outcome| {
            outcome["outcome"].as_str() == Some("selected")
                && outcome["reason_code"].is_null()
                && outcome["reason_kind"].is_null()
                && outcome["candidate_family"].as_str().is_some_and(|family| !family.is_empty())
                && outcome["category"].is_null()
                && outcome["remedy_code"].is_null()
                && outcome["remedy_kind"].is_null()
                && outcome["remedy"].is_null()
                && outcome["cost_evidence"].is_null()
        })
    }));
    assert_eq!(cpu.overall, QueryGpuCoverageOverall::CpuOnly);
    assert!(cpu.candidate_outcomes.iter().any(|outcome| {
        outcome.reason_code.as_deref() == Some("unsupported_expression") && !outcome.candidate_family.is_empty()
    }));
}

#[test]
fn planning_io_phase_marks_planning_did_io_without_retained_diagnostics() {
    let error = DataFusionError::External(Box::new(Error::new(
        ErrorCode::SourceTemporaryFailure,
        "datafusion_nexus.test",
        "temporary source failure",
    )));
    let phase = classify_physical_planning_error(&error);

    assert_eq!(phase, QueryGpuRejectionPhase::PlanningIo);
    assert!(planning_did_io_for_physical_error(phase, None));
    assert!(!planning_did_io_for_physical_error(QueryGpuRejectionPhase::PhysicalPlanning, None));
}

#[test]
fn coverage_preflight_ignores_retired_rejection_facts() {
    let retired_facts: [(&'static str, &'static str); 2] = [
        ("fallback_reason", "iceberg_table_format_unavailable"),
        ("leaf_guard_first_rejection", "unsupported_legacy_leaf"),
    ];

    for (retired_key, retired_value) in retired_facts {
        let error = DataFusionError::External(Box::new(
            Error::new(ErrorCode::UnsupportedOperator, "datafusion_nexus.test", "unsupported operator")
                .with_fact(retired_key, retired_value),
        ));

        let rejection = QueryGpuRejection::from_datafusion_error(QueryGpuRejectionPhase::PhysicalPlanning, &error);

        assert_eq!(rejection.reason_code, ErrorCode::UnsupportedOperator.as_str(), "retired fact `{retired_key}`");
        assert_eq!(
            classify_physical_planning_error(&error),
            QueryGpuRejectionPhase::PhysicalPlanning,
            "retired fact `{retired_key}` must not change coverage phase"
        );
    }
}

#[test]
fn generic_cpu_parent_over_native_fragment_is_partial() {
    let child = native_exec_leaf();
    let schema = child.schema();
    let plan = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("key", &schema).unwrap(), alias: "key".to_string() }],
            child,
        )
        .unwrap(),
    ) as Arc<dyn ExecutionPlan>;

    let observation = observe_final_plan(plan.as_ref());
    let coverage = QueryGpuCoverage::planned(
        validated_under_for_tests(),
        plan.as_ref(),
        observation,
        Vec::new(),
        Vec::new(),
        false,
        None,
    );

    assert!(matches!(coverage.overall, QueryGpuCoverageOverall::PartialGpuPlanned { .. }));
    assert_eq!(coverage.final_disposition, FinalPlanDisposition::Mixed);
    assert_eq!(coverage.domains.datafusion_boundary_count, 1);
}

#[cfg(feature = "cugraph")]
mod cugraph {
    use arrow::{
        array::{ArrayRef, Int64Array},
        record_batch::RecordBatch,
    };
    use datafusion::{
        common::config::ConfigOptions, datasource::memory::MemorySourceConfig,
        physical_optimizer::PhysicalOptimizerRule, physical_plan::coop::CooperativeExec,
    };

    use super::*;
    use crate::{
        cugraph_sql::{
            CugraphAlgoExec, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec,
            CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphTableReference, CugraphTriangleCountAllParams,
            result_schema_for_params,
        },
        error::DataFusionErrorExt,
        planner::{FinalPlanDisposition, NexusFinalPlanRequirementRule, classify_final_plan},
    };

    fn edge_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]))
    }

    fn memory_edge_source() -> Arc<dyn ExecutionPlan> {
        let schema = edge_schema();
        let src = Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef;
        let dst = Arc::new(Int64Array::from(vec![2_i64, 3])) as ArrayRef;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![src, dst]).unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    fn cugraph_spec() -> CugraphAlgorithmSpec {
        let params = CugraphAlgorithmParams::TriangleCountAll(CugraphTriangleCountAllParams);
        CugraphAlgorithmSpec {
            algorithm: CugraphAlgorithm::TriangleCountAll,
            table_name: CugraphTableReference::parse("edges").unwrap(),
            src_col: "src".to_string(),
            dst_col: "dst".to_string(),
            weight_col: None,
            edge_id_col: None,
            graph_options: CugraphGraphOptions {
                directed: false,
                renumber: true,
                construction_policy: CugraphGraphConstructionPolicy::PythonCugraph,
            },
            output_schema: result_schema_for_params(CugraphAlgorithm::TriangleCountAll, &params),
            params,
        }
    }

    fn cugraph_exec(edge_source: Arc<dyn ExecutionPlan>) -> Arc<CugraphAlgoExec> {
        Arc::new(
            CugraphAlgoExec::try_new(cugraph_spec(), edge_source, Vec::new(), None, None, test_execution_config())
                .unwrap(),
        )
    }

    fn composed_cugraph_edge_plan() -> Arc<dyn ExecutionPlan> {
        let child = cugraph_exec(memory_edge_source());
        let input_name = child.coverage_gpu_result_input_name().to_string();
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(nexus_query_engine::source::GpuResidentSource::new(
            input_name,
            child.schema(),
        )));
        let projected = builder.add_projection(
            source,
            vec![
                ProjectionOutput::new(Expr::column("vertex"), Field::new("src", DataType::Int64, false)),
                ProjectionOutput::new(Expr::column("triangle_count"), Field::new("dst", DataType::Int64, false)),
            ],
        );
        Arc::new(
            NexusNativeComposedExec::try_new(
                builder.finish_with_host_sink(projected),
                vec![child],
                test_execution_config(),
            )
            .unwrap(),
        )
    }

    fn composed_cugraph_plan(edge_source: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let cugraph = cugraph_exec(edge_source);
        let input_name = cugraph.coverage_gpu_result_input_name().to_string();
        composed_exec_with_child(cugraph, input_name)
    }

    fn assert_no_datafusion_cpu_requirement_rejects_mixed_plan(plan: Arc<dyn ExecutionPlan>) {
        assert_eq!(classify_final_plan(plan.as_ref()), FinalPlanDisposition::Mixed);
        let rule = NexusFinalPlanRequirementRule::new(
            NexusNativeOptimizerConfig::default().with_no_datafusion_cpu_requirement(),
        );

        let error =
            rule.optimize(plan, &ConfigOptions::default()).expect_err("the requirement must reject a mixed final plan");
        let error = error.find_nexus_error().expect("typed adapter error");

        assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("mixed"));
    }

    #[test]
    fn final_observation_matches_runtime_one_level_cooperative_unwrap() {
        let single_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_edge_exec()));
        assert_eq!(classify_final_plan(cugraph_exec(single_wrapped).as_ref()), FinalPlanDisposition::Native);

        let inner_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_edge_exec()));
        let nested_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(inner_wrapped));
        assert_eq!(classify_final_plan(cugraph_exec(nested_wrapped).as_ref()), FinalPlanDisposition::Mixed);
    }

    #[test]
    fn no_datafusion_cpu_requirement_rejects_composed_cugraph_cpu_edge_path() {
        assert_no_datafusion_cpu_requirement_rejects_mixed_plan(composed_cugraph_plan(memory_edge_source()));
    }

    #[test]
    fn no_datafusion_cpu_requirement_rejects_nested_cooperative_cugraph_input() {
        let inner: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_edge_exec()));
        let nested: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(inner));

        assert_no_datafusion_cpu_requirement_rejects_mixed_plan(composed_cugraph_plan(nested));
    }

    #[test]
    fn no_datafusion_cpu_requirement_rejects_cugraph_composed_child_with_cpu_producer() {
        assert_no_datafusion_cpu_requirement_rejects_mixed_plan(cugraph_exec(composed_cugraph_edge_plan()));
    }

    #[test]
    fn bare_cugraph_with_native_child_is_native() {
        let plan: Arc<dyn ExecutionPlan> = cugraph_exec(native_edge_exec());
        let observation = observe_final_plan(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            observation,
            Vec::new(),
            super::physical::cugraph_calls(plan.as_ref()),
            false,
            None,
        );

        assert_eq!(coverage.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
        assert_eq!(coverage.final_disposition, FinalPlanDisposition::Native);
        assert_eq!(coverage.domains.native_fragment_count, 1);
        assert_eq!(coverage.domains.explicit_gpu_island_count, 1);
        assert_eq!(coverage.domains.datafusion_boundary_count, 0);
    }

    #[test]
    fn wrapped_cugraph_with_cpu_collected_child_is_partial_not_false_native() {
        let cugraph = cugraph_exec(memory_edge_source());
        let input_name = cugraph.coverage_gpu_result_input_name().to_string();
        let plan = composed_exec_with_child(cugraph, input_name);
        let observation = observe_final_plan(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            observation,
            Vec::new(),
            super::physical::cugraph_calls(plan.as_ref()),
            false,
            None,
        );

        assert!(matches!(coverage.overall, QueryGpuCoverageOverall::PartialGpuPlanned { .. }));
        assert_eq!(coverage.final_disposition, FinalPlanDisposition::Mixed);
        assert_eq!(coverage.domains.native_fragment_count, 1);
        assert_eq!(coverage.domains.explicit_gpu_island_count, 1);
        assert!(coverage.domains.datafusion_boundary_count > 0);
    }

    #[test]
    fn wrapped_cugraph_with_native_child_stays_full() {
        let cugraph = cugraph_exec(native_edge_exec());
        let input_name = cugraph.coverage_gpu_result_input_name().to_string();
        let plan = composed_exec_with_child(cugraph, input_name);
        let observation = observe_final_plan(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            observation,
            Vec::new(),
            super::physical::cugraph_calls(plan.as_ref()),
            false,
            None,
        );

        assert_eq!(coverage.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
        assert_eq!(coverage.final_disposition, FinalPlanDisposition::Native);
        assert_eq!(coverage.domains.datafusion_boundary_count, 0);
        assert!(!coverage.runtime_caveats.is_empty());
    }
}

fn validated_under_for_tests() -> QueryGpuValidatedUnder {
    QueryGpuValidatedUnder {
        native_optimizer_rule_installed: true,
        optimizer_config_source: "test".to_string(),
        execution_mode: Some("strict_bounded".to_string()),
        final_plan_requirements: Vec::new(),
        session_config_fingerprint: "test".to_string(),
        optimizer_fingerprint: "test".to_string(),
        catalog_snapshot: "test".to_string(),
    }
}
