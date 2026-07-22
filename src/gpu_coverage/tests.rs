use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::{
    common::DataFusionError,
    execution::SessionStateBuilder,
    physical_expr::expressions::col,
    physical_plan::{
        ExecutionPlan,
        projection::{ProjectionExec, ProjectionExpr},
    },
    prelude::{SessionConfig, SessionContext},
};
use nexus_query_engine::{
    plan::QueryPlanBuilder,
    source::{LocalParquetSource, SourceNode},
};

use super::{
    physical::{cugraph_child_stays_gpu_resident, walk_gpu_coverage},
    *,
};
use crate::{
    error::{Error, ErrorCode},
    native_exec::{NexusNativeComposedExec, NexusNativeExec, NexusNativeExecutionConfig},
    native_report::NativeDataFusionPostNativeExecutionPath,
    planner::NexusNativeOptimizerConfig,
    policy::{FallbackReason, GpuFallbackPolicy},
    session::NexusSessionStateBuilderExt,
};

fn test_execution_config() -> NexusNativeExecutionConfig {
    NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1)
}

fn unbounded_optimizer_config(policy: GpuFallbackPolicy) -> NexusNativeOptimizerConfig {
    NexusNativeOptimizerConfig::new(policy).with_native_execution_config(NexusNativeExecutionConfig::unbounded())
}

fn validation_context(policy: GpuFallbackPolicy) -> SessionContext {
    let state =
        SessionStateBuilder::new_with_default_features().with_nexus_native(unbounded_optimizer_config(policy)).build();
    SessionContext::new_with_state(state)
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
        .with_nexus_native(
            NexusNativeOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
                .with_native_execution_config(test_execution_config()),
        )
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
        .with_nexus_native(
            NexusNativeOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
                .with_native_execution_config(test_execution_config()),
        )
        .build();

    let rejection = preflight_single_query(&state, "SELECT $$nexus_explain_coverage('SELECT 1')$$ AS marker");
    assert!(rejection.is_none(), "postgresql dollar-quoted literal should not be rejected: {rejection:?}");
}

#[tokio::test]
async fn validate_query_reports_cpu_only_and_partial_fallback_axes() {
    let ctx = validation_context(GpuFallbackPolicy::GpuPreferred);

    let cpu = validate_query(&ctx, "SELECT random() AS r").await.unwrap();
    assert_eq!(cpu.overall, QueryGpuCoverageOverall::CpuOnly);
    assert_eq!(cpu.domains.nexus_native_coverage, QueryGpuNexusNativeCoverage::None);
    assert!(
        cpu.fallback_reasons.iter().any(|reason| reason.reason_code == FallbackReason::UnsupportedExpression.code()
            && reason.kind == QueryGpuFallbackReasonKind::Expression)
    );

    let partial = validate_query(&ctx, "SELECT array[1, 2] AS values").await.unwrap();
    assert!(matches!(partial.overall, QueryGpuCoverageOverall::PartialGpuPlanned { .. }));
    assert_eq!(partial.domains.nexus_native_coverage, QueryGpuNexusNativeCoverage::Partial);
    assert_eq!(partial.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Present);
    assert!(partial.fallback_reasons.iter().any(|reason| reason.reason_code == FallbackReason::UnsupportedType.code()
        && reason.kind == QueryGpuFallbackReasonKind::ArrowType));
}

#[tokio::test]
async fn validate_query_stamps_actual_policy_but_uses_diagnostic_gpu_preferred() {
    let cpu_only_ctx = validation_context(GpuFallbackPolicy::CpuOnly);
    let cpu_only = validate_query(&cpu_only_ctx, "SELECT 1").await.unwrap();
    assert_eq!(cpu_only.validated_under.actual_policy, GpuFallbackPolicy::CpuOnly.to_string());
    assert_eq!(cpu_only.validated_under.diagnostic_policy, GpuFallbackPolicy::GpuPreferred.to_string());
    assert_eq!(cpu_only.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);

    let gpu_only_ctx = validation_context(GpuFallbackPolicy::GpuOnly);
    let gpu_only = validate_query(&gpu_only_ctx, "SELECT random() AS r").await.unwrap();
    assert_eq!(gpu_only.validated_under.actual_policy, GpuFallbackPolicy::GpuOnly.to_string());
    assert_eq!(gpu_only.validated_under.diagnostic_policy, GpuFallbackPolicy::GpuPreferred.to_string());
    assert_eq!(gpu_only.overall, QueryGpuCoverageOverall::CpuOnly);
    assert!(gpu_only.runtime_caveats.iter().any(|caveat| caveat.code == "gpu_only_policy_would_error"));
}

#[tokio::test]
async fn validate_query_concurrent_calls_do_not_share_retained_rule_state() {
    let ctx = validation_context(GpuFallbackPolicy::GpuPreferred);

    let (native, cpu) =
        tokio::join!(validate_query(&ctx, "SELECT 1 AS value"), validate_query(&ctx, "SELECT random() AS r"));
    let native = native.unwrap();
    let cpu = cpu.unwrap();

    assert_eq!(native.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
    assert!(native.fallback_reasons.is_empty());
    assert_eq!(cpu.overall, QueryGpuCoverageOverall::CpuOnly);
    assert!(cpu.fallback_reasons.iter().any(|reason| reason.reason_code == "unsupported_expression"));
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

    let walk = walk_gpu_coverage(plan.as_ref());
    let coverage = QueryGpuCoverage::planned(
        validated_under_for_tests(),
        plan.as_ref(),
        NativeDataFusionPostNativeExecutionPath::PartialNative,
        walk,
        Vec::new(),
        false,
    );

    assert!(matches!(coverage.overall, QueryGpuCoverageOverall::PartialGpuPlanned { .. }));
    assert_eq!(coverage.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Present);
}

#[cfg(feature = "cugraph")]
mod cugraph {
    use arrow::{
        array::{ArrayRef, Int64Array},
        record_batch::RecordBatch,
    };
    use datafusion::{datasource::memory::MemorySourceConfig, physical_plan::coop::CooperativeExec};

    use super::*;
    use crate::graph_sql::{
        CugraphAlgoExec, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec,
        CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphTableReference, CugraphTriangleCountAllParams,
        result_schema_for_params,
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

    #[test]
    fn cugraph_child_residency_matches_runtime_one_level_cooperative_unwrap() {
        let single_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_exec_leaf()));
        assert!(cugraph_child_stays_gpu_resident(single_wrapped.as_ref()));

        let inner_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_exec_leaf()));
        let nested_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(inner_wrapped));
        assert!(!cugraph_child_stays_gpu_resident(nested_wrapped.as_ref()));
    }

    #[test]
    fn bare_cugraph_with_native_child_is_no_cpu_even_when_cudf_axis_is_none() {
        let plan: Arc<dyn ExecutionPlan> = cugraph_exec(native_exec_leaf());
        let walk = walk_gpu_coverage(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            NativeDataFusionPostNativeExecutionPath::Cpu,
            walk,
            Vec::new(),
            false,
        );

        assert_eq!(coverage.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
        assert_eq!(coverage.domains.nexus_native_coverage, QueryGpuNexusNativeCoverage::None);
        assert_eq!(coverage.domains.cugraph_operator, QueryGpuCugraphOperatorDomain::Present);
        assert_eq!(coverage.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Absent);
    }

    #[test]
    fn wrapped_cugraph_with_cpu_collected_child_is_partial_not_false_native() {
        let cugraph = cugraph_exec(memory_edge_source());
        let input_name = cugraph.coverage_gpu_result_input_name().to_string();
        let plan = composed_exec_with_child(cugraph, input_name);
        let walk = walk_gpu_coverage(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            NativeDataFusionPostNativeExecutionPath::Native,
            walk,
            Vec::new(),
            false,
        );

        assert!(matches!(coverage.overall, QueryGpuCoverageOverall::PartialGpuPlanned { .. }));
        assert_eq!(coverage.domains.nexus_native_coverage, QueryGpuNexusNativeCoverage::Full);
        assert_eq!(coverage.domains.cugraph_operator, QueryGpuCugraphOperatorDomain::Present);
        assert_eq!(coverage.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Present);
    }

    #[test]
    fn wrapped_cugraph_with_native_child_stays_full() {
        let cugraph = cugraph_exec(native_exec_leaf());
        let input_name = cugraph.coverage_gpu_result_input_name().to_string();
        let plan = composed_exec_with_child(cugraph, input_name);
        let walk = walk_gpu_coverage(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            NativeDataFusionPostNativeExecutionPath::Native,
            walk,
            Vec::new(),
            false,
        );

        assert_eq!(coverage.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
        assert_eq!(coverage.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Absent);
        assert!(!coverage.runtime_caveats.is_empty());
    }
}

fn validated_under_for_tests() -> QueryGpuValidatedUnder {
    QueryGpuValidatedUnder {
        actual_policy: "gpu_preferred".to_string(),
        diagnostic_policy: "gpu_preferred".to_string(),
        optimizer_config_source: "test".to_string(),
        execution_mode: "strict_bounded".to_string(),
        session_config_fingerprint: "test".to_string(),
        optimizer_fingerprint: "test".to_string(),
        catalog_snapshot: "test".to_string(),
    }
}
