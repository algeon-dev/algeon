use std::{any::Any, sync::Arc};

use arrow::{
    array::AsArray,
    datatypes::{DataType, Field, Schema},
};
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
use query_planning::source::GpuResidentSource;
#[cfg(feature = "cugraph")]
use query_planning::{expr::Expr, plan::ProjectionOutput};
use query_planning::{
    plan::{NativePlanCostReason, QueryPlanBuilder},
    source::{LocalParquetSource, SourceNode},
};

use super::{physical::candidate_outcomes_from_report, sql::coverage_batch, *};
use crate::{
    capability::CapabilityReason,
    error::{Error, ErrorCode, client::MASKED_INTERNAL_ERROR_MESSAGE},
    final_plan_observation::observe_final_plan,
    native_exec::{AlgeonNativeExec, AlgeonNativeExecutionConfig},
    native_report::FinalPlanDisposition,
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
    report::{
        CandidateOutcome, GpuCandidate, NativeCandidateCostEvidence, NotSupportedDetail, PlanningReport,
        SinglePartitionOutcome,
    },
    session::AlgeonSessionStateBuilderExt,
};

fn test_execution_config() -> AlgeonNativeExecutionConfig {
    AlgeonNativeExecutionConfig::default()
}

fn bounded_optimizer_config() -> AlgeonNativeOptimizerConfig {
    AlgeonNativeOptimizerConfig::default().with_native_execution_config(AlgeonNativeExecutionConfig::default())
}

fn validation_context(config: AlgeonNativeOptimizerConfig) -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features().with_algeon_native(config).build();
    SessionContext::new_with_state(state)
}

/// Mimics an invocation-only adapter extension such as an attempt admission
/// binding. It deliberately exposes no declarative configuration entries.
#[derive(Debug, Clone)]
struct InvocationOnlyExtension;

impl ConfigExtension for InvocationOnlyExtension {
    const PREFIX: &'static str = "algeon_datafusion_test_invocation";
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
    Arc::new(
        AlgeonNativeExec::try_new_host_for_test(
            builder.finish_with_host_sink(source).unwrap(),
            test_execution_config(),
        )
        .unwrap(),
    )
}

#[cfg(feature = "cugraph")]
/// A native edge relation in the form that planning produces for a graph input: a
/// GPU-resident producer the graph reads on the device.
fn native_edge_producer() -> Arc<dyn ExecutionPlan> {
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
    let plan = builder.finish_with_host_sink(source).unwrap().clone_with_gpu_resident_sink();
    Arc::new(AlgeonNativeExec::try_new_producer_for_test("query_gpu_coverage_edges", plan).unwrap())
}

#[cfg(feature = "cugraph")]
fn composed_exec_with_child(child: Arc<dyn ExecutionPlan>, input_name: impl Into<String>) -> Arc<dyn ExecutionPlan> {
    let schema = one_column_schema();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    Arc::new(
        AlgeonNativeExec::try_new_composed_for_test(
            builder.finish_with_host_sink(source).unwrap(),
            vec![child],
            test_execution_config(),
        )
        .unwrap(),
    )
}

#[tokio::test]
async fn validate_query_rejects_multiple_statements_but_not_semicolon_literal() {
    let state = SessionStateBuilder::new()
        .with_algeon_native(
            AlgeonNativeOptimizerConfig::default().with_native_execution_config(test_execution_config()),
        )
        .build();
    let ctx = SessionContext::new_with_state(state);

    let literal = validate_query(&ctx, "SELECT ';' AS marker").await.unwrap();
    assert!(literal.rejection().is_none());

    let literal_function_name =
        validate_query(&ctx, "SELECT 'algeon_explain_coverage(''SELECT 1'')' AS marker").await.unwrap();
    assert!(literal_function_name.rejection().is_none());

    let empty = validate_query(&ctx, "").await.unwrap();
    assert!(matches!(
        empty.planning_outcome,
        QueryGpuPlanningOutcome::Rejected(ref rejection)
            if rejection.phase == QueryGpuRejectionPhase::Parsing && rejection.reason_code == "empty_sql"
    ));
    let empty_json = empty.to_json();
    assert_eq!(empty_json["planning_outcome"]["status"], "rejected");
    assert_eq!(empty_json["planning_outcome"]["rejection"]["reason_code"], "empty_sql");
    assert!(empty_json.get("overall").is_none());

    let rejected = validate_query(&ctx, "SELECT ';' AS marker; SELECT 1").await.unwrap();
    assert!(matches!(
        rejected.planning_outcome,
        QueryGpuPlanningOutcome::Rejected(ref rejection)
            if rejection.phase == QueryGpuRejectionPhase::Parsing && rejection.reason_code == "multi_statement_unsupported"
    ));

    let recursive_exists =
        validate_query(&ctx, "SELECT EXISTS (SELECT 1 FROM algeon_explain_coverage('SELECT 1')) AS recursive")
            .await
            .unwrap();
    assert!(matches!(
        recursive_exists.planning_outcome,
        QueryGpuPlanningOutcome::Rejected(ref rejection)
            if rejection.phase == QueryGpuRejectionPhase::Parsing && rejection.reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_array =
        validate_query(&ctx, "SELECT ARRAY[(SELECT count(*) FROM algeon_explain_coverage('SELECT 1'))] AS recursive")
            .await
            .unwrap();
    assert!(matches!(
        recursive_array.planning_outcome,
        QueryGpuPlanningOutcome::Rejected(ref rejection)
            if rejection.phase == QueryGpuRejectionPhase::Parsing && rejection.reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_pipe_where =
        validate_query(&ctx, "SELECT 1 |> WHERE EXISTS (SELECT 1 FROM algeon_explain_coverage('SELECT 1'))")
            .await
            .unwrap();
    assert!(matches!(
        recursive_pipe_where.planning_outcome,
        QueryGpuPlanningOutcome::Rejected(ref rejection)
            if rejection.phase == QueryGpuRejectionPhase::Parsing && rejection.reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_pipe_select = validate_query(
        &ctx,
        "SELECT 1 |> SELECT (SELECT count(*) FROM algeon_explain_coverage('SELECT 1')) AS recursive",
    )
    .await
    .unwrap();
    assert!(matches!(
        recursive_pipe_select.planning_outcome,
        QueryGpuPlanningOutcome::Rejected(ref rejection)
            if rejection.phase == QueryGpuRejectionPhase::Parsing && rejection.reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_unnest =
        validate_query(&ctx, "SELECT * FROM UNNEST(ARRAY[(SELECT count(*) FROM algeon_explain_coverage('SELECT 1'))])")
            .await
            .unwrap();
    assert!(matches!(
        recursive_unnest.planning_outcome,
        QueryGpuPlanningOutcome::Rejected(ref rejection)
            if rejection.phase == QueryGpuRejectionPhase::Parsing && rejection.reason_code == "coverage_validator_recursion_unsupported"
    ));

    let recursive_named_window = validate_query(
        &ctx,
        "SELECT row_number() OVER w AS rn \
             FROM (SELECT 1 AS x) t \
             WINDOW w AS (PARTITION BY (SELECT count(*) FROM algeon_explain_coverage('SELECT 1')))",
    )
    .await
    .unwrap();
    assert!(matches!(
        recursive_named_window.planning_outcome,
        QueryGpuPlanningOutcome::Rejected(ref rejection)
            if rejection.phase == QueryGpuRejectionPhase::Parsing && rejection.reason_code == "coverage_validator_recursion_unsupported"
    ));
}

#[tokio::test]
async fn validate_query_preflight_uses_session_sql_dialect() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().set_str("datafusion.sql_parser.dialect", "postgresql"))
        .with_algeon_native(
            AlgeonNativeOptimizerConfig::default().with_native_execution_config(test_execution_config()),
        )
        .build();

    let rejection = preflight_single_query(&state, "SELECT $$algeon_explain_coverage('SELECT 1')$$ AS marker");
    assert!(rejection.is_none(), "postgresql dollar-quoted literal should not be rejected: {rejection:?}");
}

#[tokio::test]
async fn validate_query_reports_capability_misses_without_terminal_rejection() {
    let ctx = validation_context(bounded_optimizer_config());

    let cpu = validate_query(&ctx, "SELECT random() AS r").await.unwrap();
    assert_eq!(cpu.gpu_path(), "cpu");
    assert!(cpu.validated_under.native_optimizer_rule_installed);
    assert_eq!(cpu.validated_under.optimizer_config_source, "installed_session_rule");
    let cpu_cov = cpu.final_plan_coverage().expect("cpu final plan coverage");
    assert_eq!(cpu_cov.disposition(), FinalPlanDisposition::DataFusion);
    assert_eq!(cpu_cov.native_fragment_count(), 0);
    assert!(cpu.candidate_outcomes.iter().any(|outcome| outcome.selection == CandidateOutcomeKind::NotSupported
        && outcome.reason_code.as_deref() == Some(CapabilityReason::UnsupportedExpression.code())
        && outcome.reason_kind == Some(QueryGpuCandidateReasonKind::Expression)
        && outcome.remedy.as_ref().is_some_and(|remedy| remedy.code == "rewrite_expression")));

    let unsupported = validate_query(&ctx, "SELECT CAST('1' AS DECIMAL(39, 0)) AS values").await.unwrap();
    assert_eq!(unsupported.gpu_path(), "cpu");
    let unsup_cov = unsupported.final_plan_coverage().expect("unsupported final plan coverage");
    assert_eq!(unsup_cov.disposition(), FinalPlanDisposition::DataFusion);
    assert_eq!(unsup_cov.native_fragment_count(), 0);
    assert_eq!(unsup_cov.datafusion_boundary_count(), 0);
    assert!(unsupported.candidate_outcomes.iter().any(|outcome| outcome.selection
        == CandidateOutcomeKind::NotSupported
        && outcome.reason_code.as_deref() == Some(CapabilityReason::UnsupportedType.code())
        && outcome.reason_kind == Some(QueryGpuCandidateReasonKind::ArrowType)
        && outcome.remedy.as_ref().is_some_and(|remedy| remedy.code == "cast_or_exclude_type")));
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
    assert_eq!(outcomes[0].selection, CandidateOutcomeKind::NotSelectedByCost);
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
        detail: NotSupportedDetail::default(),
    }]);

    let outcomes = candidate_outcomes_from_report(&report);

    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].selection, CandidateOutcomeKind::NotSupported);
    assert_eq!(outcomes[0].candidate_family, "unsupported_native_physical_plan");
    assert_eq!(outcomes[0].reason_kind, Some(QueryGpuCandidateReasonKind::Expression));
}

#[test]
fn candidate_outcomes_name_the_engine_gap_when_one_partition_keeps_the_rejection() {
    let rejected = |single_partition| CandidateOutcome::NotSupported {
        candidate: GpuCandidate::new("FinalAggregateExec", 1),
        capability_reason: CapabilityReason::NativeDistributedAggregateUnsupported,
        detail: NotSupportedDetail { single_partition, ..NotSupportedDetail::default() },
    };
    let report = PlanningReport::from_outcomes([
        rejected(SinglePartitionOutcome::Lowers),
        rejected(SinglePartitionOutcome::StillRejected),
    ]);

    let remedies = candidate_outcomes_from_report(&report)
        .into_iter()
        .map(|outcome| outcome.remedy.map(|remedy| (remedy.code, remedy.kind)))
        .collect::<Vec<_>>();

    assert_eq!(
        remedies,
        [
            Some(("set_target_partitions_one", QueryGpuRemedyKind::ConfigAction)),
            Some(("unsupported_operator_gap", QueryGpuRemedyKind::EngineGap)),
        ]
    );
}

/// The branches of a `UNION ALL` stay separate partitions at one partition, so
/// DataFusion still merges them to order the rows, and the validation report
/// carries the engine gap instead of the setting that cannot help.
#[tokio::test]
async fn validate_query_names_the_engine_gap_for_a_merge_of_union_branches_at_one_partition() {
    let ctx = validation_context(bounded_optimizer_config());
    ctx.sql("SET datafusion.execution.target_partitions = 1").await.unwrap().collect().await.unwrap();

    let coverage =
        validate_query(&ctx, "SELECT a, b FROM (SELECT 1 AS a, 2 AS b UNION ALL SELECT 3, 4) ORDER BY b + 1")
            .await
            .unwrap();

    let rejected = coverage
        .candidate_outcomes
        .iter()
        .find(|outcome| outcome.selection == CandidateOutcomeKind::NotSupported)
        .expect("the merge of union branches is rejected");
    assert_eq!(rejected.reason_code.as_deref(), Some("native_sort_preserving_merge_unsupported"));
    let remedy = rejected.remedy.as_ref().expect("a rejected candidate carries a remedy");
    assert_eq!((remedy.code, remedy.kind), ("unsupported_operator_gap", QueryGpuRemedyKind::EngineGap));
}

#[tokio::test]
async fn validate_query_preserves_the_installed_final_plan_policy() {
    let ctx =
        validation_context(bounded_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired));
    let coverage = validate_query(&ctx, "SELECT random() AS r").await.unwrap();

    let rejection = coverage.rejection().expect("rejection");
    assert_eq!(rejection.phase, QueryGpuRejectionPhase::PhysicalPlanning);
    assert_eq!(coverage.gpu_path(), "rejected");
    assert!(
        coverage
            .candidate_outcomes
            .iter()
            .any(|outcome| outcome.reason_code.as_deref() == Some("unsupported_expression"))
    );
    let blocked = coverage
        .candidate_outcomes
        .iter()
        .find(|outcome| outcome.selection == CandidateOutcomeKind::NotSupported)
        .and_then(|outcome| outcome.blocking.as_ref())
        .expect("the first not-supported candidate names its blocking subject");
    // The execution message ends with a pointer to this report, so the report
    // omits it.
    assert_eq!(rejection.detail, format!("GPU execution does not support {}", blocked.text()), "coverage={coverage:?}");
    assert_eq!(coverage.validated_under.final_plan_requirements, ["no_datafusion_cpu"]);
}

#[tokio::test]
async fn validate_query_without_a_native_rule_fails_closed_with_structured_error() {
    let ctx = SessionContext::new();

    let error = validate_query(&ctx, "SELECT random() AS r").await.unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.operation(), "algeon_datafusion.gpu_coverage.validation_config");
    assert_eq!(error.fact_value("reason"), Some("native_optimizer_rule_not_installed"));
}

#[tokio::test]
async fn validate_query_metadata_uses_effective_session_optimizer_options() {
    let optimizer_config = AlgeonNativeOptimizerConfig::default()
        .with_native_execution_config(test_execution_config())
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired);
    let mut session_config = SessionConfig::new();
    // The deleted pipeline-mode session key is rejected; the single bounded
    // path projects the canonical execution-mode label.
    assert!(session_config.options_mut().set("algeon_datafusion.native.pipeline_execution_mode", "adaptive").is_err());
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_algeon_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);

    let coverage = validate_query(&ctx, "SELECT 1 AS value").await.unwrap();

    assert_eq!(coverage.validated_under.execution_mode.as_deref(), Some("bounded"));
    assert_eq!(coverage.validated_under.final_plan_requirements, ["no_datafusion_cpu"]);
}

#[tokio::test]
async fn validate_query_session_fingerprint_ignores_invocation_local_extensions() {
    let state = SessionStateBuilder::new_with_default_features().with_algeon_native(bounded_optimizer_config()).build();
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
async fn validate_query_concurrent_calls_report_their_own_planning_diagnostics() {
    let ctx = validation_context(bounded_optimizer_config());

    let (native, cpu) = tokio::join!(
        validate_query(&ctx, "SELECT x AS value FROM (VALUES (1)) AS t(x)"),
        validate_query(&ctx, "SELECT random() AS r")
    );
    let native = native.unwrap();
    let cpu = cpu.unwrap();

    assert_eq!(native.gpu_path(), "native");
    assert_eq!(native.final_plan_coverage().unwrap().disposition(), FinalPlanDisposition::Native);
    assert!(native.candidate_outcomes.iter().any(|outcome| {
        outcome.selection == CandidateOutcomeKind::Selected
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
                && outcome["blocking"].is_null()
                && outcome["cost_evidence"].is_null()
        })
    }));
    assert_eq!(cpu.gpu_path(), "cpu");
    assert!(cpu.candidate_outcomes.iter().any(|outcome| {
        outcome.reason_code.as_deref() == Some("unsupported_expression") && !outcome.candidate_family.is_empty()
    }));
    let blocking = cpu
        .candidate_outcomes
        .iter()
        .find_map(|outcome| outcome.blocking.as_ref())
        .expect("the unsupported random() candidate names what blocked it");
    assert_eq!((blocking.kind(), blocking.text()), ("scalar_function", "scalar function random in `random()`"));
    assert!(cpu.to_json()["candidate_outcomes"].as_array().is_some_and(|outcomes| {
        outcomes.iter().any(|outcome| outcome["blocking"]["function"].as_str() == Some("random"))
    }));
}

#[test]
fn planning_io_phase_marks_planning_did_io_without_retained_diagnostics() {
    let error = DataFusionError::External(Box::new(Error::new(
        ErrorCode::SourceTemporaryFailure,
        "algeon_datafusion.test",
        "temporary source failure",
    )));
    let phase = classify_physical_planning_error(&error);

    assert_eq!(phase, QueryGpuRejectionPhase::PlanningIo);
    assert!(planning_did_io_for_physical_error(phase));
    assert!(!planning_did_io_for_physical_error(QueryGpuRejectionPhase::PhysicalPlanning));
}

#[test]
fn coverage_preflight_ignores_retired_rejection_facts() {
    let retired_facts: [(&'static str, &'static str); 2] = [
        ("fallback_reason", "iceberg_table_format_unavailable"),
        ("leaf_guard_first_rejection", "unsupported_legacy_leaf"),
    ];

    for (retired_key, retired_value) in retired_facts {
        let error = DataFusionError::External(Box::new(
            Error::new(ErrorCode::UnsupportedOperator, "algeon_datafusion.test", "unsupported operator")
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

/// A rejection shows what the Flight status shows for the same failure: the
/// code `classify` assigns, and the static message for failures whose raw text
/// the server withholds.
#[tokio::test]
async fn rejection_matches_the_code_and_masking_of_the_flight_status() {
    let foreign = DataFusionError::Execution("reader failed on /private/customer.parquet".to_string());
    let rejection = QueryGpuRejection::from_datafusion_error(QueryGpuRejectionPhase::PhysicalPlanning, &foreign);
    assert_eq!(rejection.reason_code, ErrorCode::Internal.as_str());
    assert_eq!(rejection.detail, MASKED_INTERNAL_ERROR_MESSAGE);

    let typed = DataFusionError::External(Box::new(Error::new(
        ErrorCode::Internal,
        "algeon_datafusion.test",
        "invariant broke at /private/customer.parquet",
    )));
    let rejection = QueryGpuRejection::from_datafusion_error(QueryGpuRejectionPhase::PhysicalPlanning, &typed);
    assert_eq!(rejection.reason_code, ErrorCode::Internal.as_str());
    assert_eq!(rejection.detail, MASKED_INTERNAL_ERROR_MESSAGE);

    let ctx = validation_context(bounded_optimizer_config());
    let coverage = validate_query(&ctx, "SELEC 1").await.unwrap();
    let rejection = coverage.rejection().expect("a syntax error is a rejection");
    assert_eq!(rejection.phase, QueryGpuRejectionPhase::Parsing);
    assert_eq!(rejection.reason_code, ErrorCode::SqlParse.as_str());
    assert!(rejection.detail.starts_with("invalid SQL statement: "), "{}", rejection.detail);
}

/// A failure whose raw text the server withholds keeps that text on the
/// rejection for in-process callers. No serialized projection carries it.
#[tokio::test]
async fn masked_rejection_keeps_its_raw_text_in_process_only() {
    const RAW_PATH: &str = "/private/customer.parquet";
    let foreign = DataFusionError::Execution(format!("reader failed on {RAW_PATH}"));
    let typed = DataFusionError::External(Box::new(Error::new(
        ErrorCode::Internal,
        "algeon_datafusion.test",
        format!("invariant broke at {RAW_PATH}"),
    )));

    for error in [foreign, typed] {
        let rejection = QueryGpuRejection::from_datafusion_error(QueryGpuRejectionPhase::PhysicalPlanning, &error);
        assert_eq!(rejection.detail, MASKED_INTERNAL_ERROR_MESSAGE);
        assert!(rejection.trusted_diagnostic.contains(RAW_PATH), "{}", rejection.trusted_diagnostic);

        let coverage = QueryGpuCoverage::rejected(validated_under_for_tests(), rejection);
        let rejection = coverage.rejection().expect("a rejected coverage carries its rejection");
        assert_eq!(rejection.detail, MASKED_INTERNAL_ERROR_MESSAGE);
        assert!(rejection.trusted_diagnostic.contains(RAW_PATH), "{}", rejection.trusted_diagnostic);

        assert!(!coverage.to_json().to_string().contains(RAW_PATH));
        let batch = coverage_batch(&coverage, ExplainGpuOutputMode::Rows, None).unwrap();
        for column in batch.columns().iter().filter_map(|column| column.as_string_opt::<i32>()) {
            assert!(column.iter().flatten().all(|value| !value.contains(RAW_PATH)), "{column:?}");
        }
    }

    let ctx = validation_context(bounded_optimizer_config());
    let empty = validate_query(&ctx, "").await.unwrap();
    let rejection = empty.rejection().expect("empty SQL is a rejection");
    assert_eq!(rejection.trusted_diagnostic, rejection.detail);
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
        None,
    );

    assert_eq!(coverage.gpu_path(), "partial_native");
    let plan_cov = coverage.final_plan_coverage().unwrap();
    assert_eq!(plan_cov.disposition(), FinalPlanDisposition::Mixed);
    assert_eq!(plan_cov.datafusion_boundary_count(), 1);
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
            CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphTriangleCountAllParams,
            result_schema_for_params,
        },
        error::DataFusionErrorExt,
        planner::{AlgeonNativePhysicalOptimizerRule, FinalPlanDisposition, classify_final_plan},
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
            has_weight: false,
            has_edge_id: false,
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
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, child.schema())));
        let projected = builder
            .add_projection(
                source,
                vec![
                    ProjectionOutput::new(Expr::column("vertex"), Field::new("src", DataType::Int64, false)),
                    ProjectionOutput::new(Expr::column("triangle_count"), Field::new("dst", DataType::Int64, false)),
                ],
            )
            .unwrap();
        Arc::new(
            AlgeonNativeExec::try_new_composed_for_test(
                builder.finish_with_host_sink(projected).unwrap(),
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

    fn assert_reject_cpu_policy_rejects_mixed_plan(plan: Arc<dyn ExecutionPlan>) {
        assert_eq!(classify_final_plan(plan.as_ref()), FinalPlanDisposition::Mixed);
        let rule = AlgeonNativePhysicalOptimizerRule::new(
            AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
        );

        let error = rule
            .optimize(plan, &ConfigOptions::default())
            .expect_err("the reject policy must reject a mixed final plan");
        let error = error.find_algeon_error().expect("typed adapter error");

        assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("mixed"));
    }

    #[test]
    fn final_observation_recursively_unwraps_cooperative_native_inputs() {
        let single_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_edge_producer()));
        assert_eq!(classify_final_plan(cugraph_exec(single_wrapped).as_ref()), FinalPlanDisposition::Native);

        let inner_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_edge_producer()));
        let nested_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(inner_wrapped));
        assert_eq!(classify_final_plan(cugraph_exec(nested_wrapped).as_ref()), FinalPlanDisposition::Native);
    }

    #[test]
    fn reject_cpu_policy_rejects_composed_cugraph_cpu_edge_path() {
        assert_reject_cpu_policy_rejects_mixed_plan(composed_cugraph_plan(memory_edge_source()));
    }

    #[test]
    fn reject_cpu_policy_accepts_nested_cooperative_cugraph_input() {
        let inner: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_edge_producer()));
        let nested: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(inner));
        let plan = composed_cugraph_plan(nested);
        assert_eq!(classify_final_plan(plan.as_ref()), FinalPlanDisposition::Native);
        let rule = AlgeonNativePhysicalOptimizerRule::new(
            AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
        );
        rule.optimize(plan, &ConfigOptions::default())
            .expect("the reject policy must accept a recursively native graph input");
    }

    #[test]
    fn reject_cpu_policy_rejects_cugraph_composed_child_with_cpu_producer() {
        assert_reject_cpu_policy_rejects_mixed_plan(cugraph_exec(composed_cugraph_edge_plan()));
    }

    #[test]
    fn bare_cugraph_with_native_child_is_native() {
        let plan: Arc<dyn ExecutionPlan> = cugraph_exec(native_edge_producer());
        let observation = observe_final_plan(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            observation,
            Vec::new(),
            super::physical::cugraph_calls(plan.as_ref()),
            None,
        );

        assert_eq!(coverage.gpu_path(), "native");
        let plan_cov = coverage.final_plan_coverage().unwrap();
        assert_eq!(plan_cov.disposition(), FinalPlanDisposition::Native);
        assert_eq!(plan_cov.native_fragment_count(), 1);
        assert_eq!(plan_cov.explicit_gpu_island_count(), 1);
        assert_eq!(plan_cov.datafusion_boundary_count(), 0);
        assert!(coverage.cugraph_calls[0].inputs_native);
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
            None,
        );

        assert_eq!(coverage.gpu_path(), "partial_native");
        let plan_cov = coverage.final_plan_coverage().unwrap();
        assert_eq!(plan_cov.disposition(), FinalPlanDisposition::Mixed);
        assert_eq!(plan_cov.native_fragment_count(), 1);
        assert_eq!(plan_cov.explicit_gpu_island_count(), 1);
        assert!(plan_cov.datafusion_boundary_count() > 0);
        let call = &coverage.cugraph_calls[0];
        assert!(!call.inputs_native, "{call:?}");
        assert!(call.notes.iter().any(|note| note.contains("host Arrow")), "{call:?}");
    }

    #[test]
    fn wrapped_cugraph_with_native_child_stays_full() {
        let cugraph = cugraph_exec(native_edge_producer());
        let input_name = cugraph.coverage_gpu_result_input_name().to_string();
        let plan = composed_exec_with_child(cugraph, input_name);
        let observation = observe_final_plan(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            observation,
            Vec::new(),
            super::physical::cugraph_calls(plan.as_ref()),
            None,
        );

        assert_eq!(coverage.gpu_path(), "native");
        let plan_cov = coverage.final_plan_coverage().unwrap();
        assert_eq!(plan_cov.disposition(), FinalPlanDisposition::Native);
        assert_eq!(plan_cov.datafusion_boundary_count(), 0);
        assert!(!coverage.runtime_caveats.is_empty());
    }
}

fn validated_under_for_tests() -> QueryGpuValidatedUnder {
    QueryGpuValidatedUnder {
        native_optimizer_rule_installed: true,
        optimizer_config_source: "test".to_string(),
        execution_mode: Some("bounded".to_string()),
        final_plan_requirements: Vec::new(),
        session_config_fingerprint: "test".to_string(),
        optimizer_fingerprint: "test".to_string(),
        catalog_snapshot: "test".to_string(),
    }
}
