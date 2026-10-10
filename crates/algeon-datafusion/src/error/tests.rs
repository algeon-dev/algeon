use std::{collections::HashSet, sync::Arc};

use arrow::error::ArrowError;
use cudf::{ErrorCode as CudfErrorCode, io::parquet::RemoteParquetSource, testing::error::new_with_for_test};
use datafusion::common::{DataFusionError, Diagnostic, Result as DataFusionResult};
use query_planning::{
    error::{QueryError, QueryErrorCode, QueryErrorStatus},
    triage::{TriageCode, TriageVerdict},
};
use query_runtime::{native_error::cudf_query_error, runtime::ExecutionCancellation};

use super::*;

#[derive(Debug)]
struct TestSourceError {
    message: &'static str,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl TestSourceError {
    fn leaf(message: &'static str) -> Self {
        Self { message, source: None }
    }

    fn with_source(message: &'static str, source: impl StdError + Send + Sync + 'static) -> Self {
        Self { message, source: Some(Box::new(source)) }
    }
}

impl fmt::Display for TestSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl StdError for TestSourceError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
    }
}

fn construction_error(operation: &'static str) -> Error {
    Error::new(ErrorCode::Internal, operation, "test Algeon error")
}

#[test]
fn construction_and_fact_collisions_do_not_unwind_and_preserve_first_values() {
    let error = std::panic::catch_unwind(|| {
        Error::new(ErrorCode::Internal, "", "")
            .with_fact("", "ignored-empty-key")
            .with_fact("unsafe_first", "first-unsafe")
            .with_log_safe_fact("unsafe_first", "second-safe")
            .with_log_safe_fact("safe_first", "first-safe")
            .with_fact("safe_first", "second-unsafe")
    })
    .expect("constructing an error must not unwind");

    assert_eq!(error.operation(), FALLBACK_OPERATION);
    assert_eq!(error.message(), FALLBACK_MESSAGE);
    assert_eq!(error.fact_value(""), None);
    assert_eq!(error.fact_value("unsafe_first"), Some("first-unsafe"));
    assert_eq!(error.fact_value("safe_first"), Some("first-safe"));
    assert_eq!(error.fact_value(FACT_COLLISION_KEYS), Some("<empty>,safe_first,unsafe_first"));

    let projection = error.facts().log_safe_projection();
    assert_eq!(
        projection.entries,
        vec![
            (FACT_COLLISION_KEYS, "<empty>,safe_first,unsafe_first".to_string()),
            ("safe_first", "first-safe".to_string()),
        ]
    );
}

#[test]
fn repeated_source_attachment_does_not_unwind_or_replace_the_first_source() {
    let first_typed = std::panic::catch_unwind(|| {
        Error::new(ErrorCode::Internal, "algeon_datafusion.test.source", "source test")
            .with_source(TestSourceError::leaf("first typed source"))
            .with_boxed_source(Box::new(TestSourceError::leaf("second boxed source")))
    })
    .expect("attaching a second boxed source must not unwind");
    assert_eq!(first_typed.source_error().map(ToString::to_string).as_deref(), Some("first typed source"));

    let first_boxed = std::panic::catch_unwind(|| {
        Error::new(ErrorCode::Internal, "algeon_datafusion.test.source", "source test")
            .with_boxed_source(Box::new(TestSourceError::leaf("first boxed source")))
            .with_source(TestSourceError::leaf("second typed source"))
    })
    .expect("attaching a second typed source must not unwind");
    assert_eq!(first_boxed.source_error().map(ToString::to_string).as_deref(), Some("first boxed source"));
}

#[test]
fn log_safe_fact_projection_sanitizes_and_sorts_without_changing_raw_facts() {
    let error = Error::new(ErrorCode::Internal, "algeon_datafusion.test.facts", "fact projection test")
        .with_log_safe_fact("z_newline", "first\nsecond\tthird")
        .with_log_safe_fact("a_uri", "read s3://user:secret@bucket/path?token=value now")
        .with_log_safe_fact("b_credential", "Authorization=Bearer-secret")
        .with_log_safe_fact("c_client_secret", "CLIENT-SECRET=client-secret-value")
        .with_log_safe_fact("d_refresh_token", "Refresh.Token:refresh-token-value")
        .with_log_safe_fact("e_aws_secret", "AWS_SECRET_ACCESS_KEY=aws-secret-value")
        .with_fact("literal", "customer-secret")
        .with_fact("path", "/private/customer.parquet")
        .with_fact("config_value", "super-secret-config")
        .with_fact("native_message", "native secret")
        .with_fact("cudf_message", "cudf secret");

    assert_eq!(error.fact_value("literal"), Some("customer-secret"));
    assert_eq!(error.fact_value("path"), Some("/private/customer.parquet"));
    let projection = error.facts().log_safe_projection();
    assert_eq!(
        projection.entries,
        vec![
            ("a_uri", "read <redacted-uri> now".to_string()),
            ("b_credential", "<redacted-credential>".to_string()),
            ("c_client_secret", "<redacted-credential>".to_string()),
            ("d_refresh_token", "<redacted-credential>".to_string()),
            ("e_aws_secret", "<redacted-credential>".to_string()),
            ("z_newline", "first second third".to_string()),
        ]
    );
    assert!(!projection.truncated());
    let rendered = projection.rendered().expect("safe facts must render");
    assert_eq!(
        rendered,
        "a_uri=read <redacted-uri> now, b_credential=<redacted-credential>, c_client_secret=<redacted-credential>, d_refresh_token=<redacted-credential>, e_aws_secret=<redacted-credential>, z_newline=first second third"
    );
    for hidden in
        ["customer-secret", "/private/customer.parquet", "super-secret-config", "native secret", "cudf secret"]
    {
        assert!(!rendered.contains(hidden), "safe projection leaked {hidden}");
    }
}

#[test]
fn log_safe_fact_projection_caps_values_count_and_total_size() {
    let long_value = "v".repeat(MAX_LOG_SAFE_FACT_VALUE_BYTES + 20);
    let value_capped = Error::new(ErrorCode::Internal, "algeon_datafusion.test.value_cap", "value cap")
        .with_log_safe_fact("long", &long_value);
    let projection = value_capped.facts().log_safe_projection();
    assert!(projection.truncated());
    assert_eq!(projection.entries.len(), 1);
    assert_eq!(projection.entries[0].1.len(), MAX_LOG_SAFE_FACT_VALUE_BYTES);
    assert!(projection.entries[0].1.ends_with("..."));

    let keys = [
        "fact_00", "fact_01", "fact_02", "fact_03", "fact_04", "fact_05", "fact_06", "fact_07", "fact_08", "fact_09",
        "fact_10", "fact_11", "fact_12", "fact_13", "fact_14", "fact_15", "fact_16",
    ];
    let mut count_capped = Error::new(ErrorCode::Internal, "algeon_datafusion.test.count_cap", "count cap");
    for key in keys {
        count_capped = count_capped.with_log_safe_fact(key, "safe");
    }
    let projection = count_capped.facts().log_safe_projection();
    assert_eq!(projection.entries.len(), MAX_LOG_SAFE_FACT_COUNT);
    assert!(projection.truncated());
    assert_eq!(projection.retained_entries()[0], (ERROR_FACTS_TRUNCATED_KEY.to_string(), "true".to_string()));

    let mut total_capped = Error::new(ErrorCode::Internal, "algeon_datafusion.test.total_cap", "total cap");
    for key in keys.into_iter().take(10) {
        total_capped = total_capped.with_log_safe_fact(key, "x".repeat(MAX_LOG_SAFE_FACT_VALUE_BYTES));
    }
    let projection = total_capped.facts().log_safe_projection();
    let rendered = projection.rendered().expect("partially projected facts must render");
    assert!(projection.truncated());
    assert!(projection.entries.len() < 10);
    assert!(rendered.len() <= MAX_LOG_SAFE_FACTS_RENDERED_BYTES);
}

#[test]
fn reserved_truncation_marker_cannot_spoof_retained_projection() {
    let not_truncated = Error::new(ErrorCode::Internal, "algeon_datafusion.test.marker", "marker")
        .with_log_safe_fact(ERROR_FACTS_TRUNCATED_KEY, "true")
        .with_log_safe_fact("safe", "classification");
    assert_eq!(not_truncated.fact_value(ERROR_FACTS_TRUNCATED_KEY), Some("true"));
    let projection = not_truncated.facts().log_safe_projection();
    assert_eq!(projection.entries, vec![("safe", "classification".to_string())]);
    assert!(!projection.truncated());
    assert!(projection.retained_entries().iter().all(|(key, _)| key != ERROR_FACTS_TRUNCATED_KEY));

    let actually_truncated = Error::new(ErrorCode::Internal, "algeon_datafusion.test.marker", "marker")
        .with_log_safe_fact(ERROR_FACTS_TRUNCATED_KEY, "false")
        .with_log_safe_fact("long", "x".repeat(MAX_LOG_SAFE_FACT_VALUE_BYTES + 1));
    assert_eq!(actually_truncated.fact_value(ERROR_FACTS_TRUNCATED_KEY), Some("false"));
    let projection = actually_truncated.facts().log_safe_projection();
    assert!(projection.truncated());
    assert!(projection.entries.iter().all(|(key, _)| *key != ERROR_FACTS_TRUNCATED_KEY));
    let retained = projection.retained_entries();
    assert_eq!(
        retained.iter().filter(|(key, _)| key == ERROR_FACTS_TRUNCATED_KEY).collect::<Vec<_>>(),
        vec![&(ERROR_FACTS_TRUNCATED_KEY.to_string(), "true".to_string())]
    );
}

#[test]
fn facts_debug_uses_only_the_sanitized_capped_projection() {
    let error = Error::new(ErrorCode::Internal, "algeon_datafusion.test.debug", "debug projection")
        .with_fact("literal", "never-log-this-literal")
        .with_log_safe_fact("safe", "line one\nhttps://user:password@example.test/private line two")
        .with_log_safe_fact("long", "z".repeat(MAX_LOG_SAFE_FACT_VALUE_BYTES + 1));

    let debug = format!("{:?}", error.facts());
    assert!(!debug.contains("never-log-this-literal"), "{debug}");
    assert!(!debug.contains("user:password"), "{debug}");
    assert!(!debug.contains('\n'), "{debug}");
    assert!(debug.contains("<redacted-uri>"), "{debug}");
    assert!(debug.contains("error_facts_truncated: true"), "{debug}");
}

#[test]
fn foreign_wrappers_only_expose_audited_classification_facts() {
    let native = QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.test.safe_projection",
        "native-message-must-stay-hidden",
    )
    .with_fact("literal", "native-user-literal")
    .with_fact("path", "/private/native.parquet");
    let native_adapter = Error::from_native("algeon_datafusion.test.safe_projection", "wrapped native error", native);
    let projection = native_adapter.facts().log_safe_projection();
    assert_eq!(
        projection.entries.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        vec!["native_code", "native_kind", "native_operation", "native_status"]
    );
    let rendered = projection.rendered().unwrap();
    for hidden in ["native-message-must-stay-hidden", "native-user-literal", "/private/native.parquet"] {
        assert!(!rendered.contains(hidden), "{rendered}");
    }

    let cudf =
        RemoteParquetSource::s3("gs://user:password@bucket/private.parquet").expect_err("non-s3 scheme must fail");
    let cudf_adapter = Error::from_cudf("algeon_datafusion.test.safe_projection", "wrapped cudf error", cudf);
    let projection = cudf_adapter.facts().log_safe_projection();
    assert_eq!(
        projection.entries.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        vec!["cudf_code", "cudf_kind", "cudf_operation", "cudf_status"]
    );
    assert!(!projection.rendered().unwrap().contains("password"));
}

#[test]
fn blocking_native_diagnostics_reach_display_without_unrelated_facts() {
    let native = QueryError::new(
        QueryErrorCode::UnsupportedPipelineContract,
        "query_engine.admission.composed_consumer",
        "a composed graph attempt lacks a complete ordered memory schedule",
    )
    .with_fact("schedule_reason", "consumer_local_source_rows_unavailable")
    .with_fact("stage", "prepared_composed_admission")
    .with_fact("node_id", 17)
    .with_fact("path", "/private/customer.parquet")
    .with_fact("literal", "secret");
    let adapter = Error::from_native_with_blocking_diagnostics(
        "algeon_datafusion.native_composed.admission_evidence",
        "failed to derive the complete engine-owned composed admission bound",
        native,
    );
    let display = adapter.to_string();

    assert!(display.contains("a composed graph attempt lacks a complete ordered memory schedule"));
    assert!(display.contains("schedule_reason=consumer_local_source_rows_unavailable"));
    assert!(display.contains("stage=prepared_composed_admission"));
    assert!(display.contains("node_id=17"));
    assert!(!display.contains("/private/customer.parquet"));
    assert!(!display.contains("secret"));
    assert_eq!(adapter.fact_value("path"), Some("/private/customer.parquet"));
    assert!(DataFusionError::from(adapter).to_string().contains("stage=prepared_composed_admission"));

    let native = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "query_engine.query_service.admission",
        "the complete requirement exceeds every compatible device capacity",
    )
    .with_fact("reason", "impossible_requirement")
    .with_fact("stage", "ticket_admission")
    .with_fact("minimum_compatible_required_floor_bytes", 200)
    .with_fact("max_compatible_managed_capacity_bytes", 100)
    .with_fact("minimum_compatible_dominant_owner", "consumer")
    .with_fact("minimum_compatible_dominant_owner_role", "consumer_output")
    .with_fact("minimum_compatible_dominant_owner_bytes", 150);
    let adapter = Error::from_native_with_blocking_diagnostics(
        "algeon_datafusion.native_exec.context",
        "failed to acquire backend query-service algeon native execution context",
        native,
    );
    assert_eq!(adapter.fact_value("native_code"), Some("service_overloaded"));
    let display = DataFusionError::from(adapter).to_string();
    for fact in [
        "reason=impossible_requirement",
        "minimum_compatible_required_floor_bytes=200",
        "max_compatible_managed_capacity_bytes=100",
        "minimum_compatible_dominant_owner=consumer",
        "minimum_compatible_dominant_owner_role=consumer_output",
        "minimum_compatible_dominant_owner_bytes=150",
    ] {
        assert!(display.contains(fact), "{display}");
    }

    let native = QueryError::new(
        QueryErrorCode::CudfInvalidInput,
        "query_engine.vector_algorithm.pack",
        "failed to pack the prepared vector feature table",
    )
    .with_fact("relation_role", "dataset")
    .with_fact("training_rows", 0)
    .with_fact("validation_reason", "non_finite_vector_value")
    .with_fact("non_finite_count", 5)
    .with_fact("remedy", "replace or reject null, NaN, and infinite values before GPU consumer execution")
    .with_fact("cuvs_message", "/private/customer.parquet");
    let adapter = Error::from_native_with_blocking_diagnostics(
        "algeon_datafusion.cuvs_sql.execute",
        "native cuVS vector query execution failed",
        native,
    );
    assert_eq!(adapter.kind(), ErrorKind::InvalidInput);
    let display = adapter.to_string();
    for fact in [
        "relation_role=dataset",
        "training_rows=0",
        "validation_reason=non_finite_vector_value",
        "non_finite_count=5",
        "remedy=replace or reject null, NaN, and infinite values before GPU consumer execution",
    ] {
        assert!(display.contains(fact), "{display}");
    }
    assert!(!display.contains("/private/customer.parquet"), "{display}");
}

#[test]
fn error_debug_omits_native_and_cudf_source_payloads_but_keeps_typed_access() {
    let native =
        QueryError::new(QueryErrorCode::InvalidPlan, "query_engine.test.debug_source", "native-source-message-secret")
            .with_fact("literal", "native-source-fact-secret");
    let native_adapter = Error::from_native("algeon_datafusion.test.debug_source", "native wrapper", native);
    let native_source = native_adapter
        .source_error()
        .and_then(|source| source.downcast_ref::<QueryError>())
        .expect("native source must remain typed");
    assert_eq!(native_source.fact_value("literal"), Some("native-source-fact-secret"));
    let native_debug = format!("{native_adapter:?}");
    assert!(native_debug.contains("<source omitted>"), "{native_debug}");
    assert!(!native_debug.contains("native-source-message-secret"), "{native_debug}");
    assert!(!native_debug.contains("native-source-fact-secret"), "{native_debug}");

    let cudf = new_with_for_test(
        CudfErrorCode::ForeignInvalidArgument,
        "cudf.test.debug_source",
        "cudf-source-detail-secret client_secret=cudf-secret-value",
    );
    let cudf_adapter = Error::from_cudf("algeon_datafusion.test.debug_source", "cudf wrapper", cudf);
    let cudf_source = cudf_adapter
        .source_error()
        .and_then(|source| source.downcast_ref::<cudf::Error>())
        .expect("cuDF source must remain typed");
    assert_eq!(cudf_source.message(), "cudf-source-detail-secret client_secret=cudf-secret-value");
    let cudf_debug = format!("{cudf_adapter:?}");
    assert!(cudf_debug.contains("<source omitted>"), "{cudf_debug}");
    assert!(!cudf_debug.contains("cudf-source-detail-secret"), "{cudf_debug}");
    assert!(!cudf_debug.contains("cudf-secret-value"), "{cudf_debug}");
}

#[test]
fn find_algeon_error_finds_direct_external_error() {
    let error = DataFusionError::External(Box::new(construction_error("algeon_datafusion.test.direct")));

    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.direct"));
}

#[test]
fn native_error_status_projection_agrees_with_the_engine_for_every_code() {
    for native_code in QueryErrorCode::ALL {
        let native =
            QueryError::new(*native_code, "query_engine.test.status_projection", "native status projection test");
        let adapter =
            Error::from_native("algeon_datafusion.test.status_projection", "adapter status projection test", native);

        assert_eq!(
            adapter.status().as_str(),
            native_code.status().as_str(),
            "{native_code} must retain its native retry/remediation status"
        );
    }
}

#[test]
fn find_algeon_error_traverses_context() {
    let error = DataFusionError::Context(
        "context".to_string(),
        Box::new(DataFusionError::External(Box::new(construction_error("algeon_datafusion.test.context")))),
    );

    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.context"));
}

#[test]
fn find_algeon_error_traverses_diagnostic() {
    let error = DataFusionError::Diagnostic(
        Box::new(Diagnostic::new_error("diagnostic", None)),
        Box::new(DataFusionError::External(Box::new(construction_error("algeon_datafusion.test.diagnostic")))),
    );

    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.diagnostic"));
}

#[test]
fn find_algeon_error_traverses_shared() {
    let error = DataFusionError::Shared(Arc::new(DataFusionError::External(Box::new(construction_error(
        "algeon_datafusion.test.shared",
    )))));

    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.shared"));
}

#[test]
fn find_algeon_error_traverses_nested_collection_in_deterministic_order() {
    let error = DataFusionError::Collection(vec![
        DataFusionError::Plan("no Algeon error".to_string()),
        DataFusionError::Collection(vec![
            DataFusionError::Execution("still no Algeon error".to_string()),
            DataFusionError::External(Box::new(construction_error("algeon_datafusion.test.first"))),
        ]),
        DataFusionError::External(Box::new(construction_error("algeon_datafusion.test.second"))),
    ]);

    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.first"));
}

#[test]
fn find_algeon_error_traverses_collection_inside_arrow_external_wrapper() {
    let collection = DataFusionError::Collection(vec![
        DataFusionError::Plan("no Algeon error".to_string()),
        DataFusionError::External(Box::new(construction_error("algeon_datafusion.test.arrow_collection"))),
    ]);
    let arrow = ArrowError::from_external_error(Box::new(collection));
    let error = DataFusionError::ArrowError(Box::new(arrow), None);

    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.arrow_collection"));
}

#[test]
fn find_algeon_error_returns_none_without_match() {
    let error = DataFusionError::Collection(vec![
        DataFusionError::Plan("planning failed".to_string()),
        DataFusionError::Context(
            "context".to_string(),
            Box::new(DataFusionError::Execution("execution failed".to_string())),
        ),
    ]);

    assert!(error.find_algeon_error().is_none());
}

#[test]
fn source_diagnostics_walk_sources_and_deduplicate_adjacent_messages() {
    let error = TestSourceError::with_source(
        "dependency wrapper",
        TestSourceError::with_source("dependency wrapper", TestSourceError::leaf("socket refused")),
    );

    let diagnostics = ErrorSourceDiagnostics::from_error(&error);

    assert_eq!(diagnostics.chain(), Some("dependency wrapper: socket refused"));
    assert_eq!(diagnostics.root_cause(), Some("socket refused"));
    assert_eq!(diagnostics.root_cause_when_distinct(), Some("socket refused"));
}

#[test]
fn source_diagnostics_bound_extreme_chain_bytes_without_rewriting_the_error() {
    let mut error = DataFusionError::Execution("y".repeat(4_096));
    for depth in 0..8 {
        error = DataFusionError::Context(format!("wrapper-{depth}"), Box::new(error));
    }
    let error_display = error.to_string();
    assert!(error_display.len() > 4_096);

    let diagnostics = ErrorSourceDiagnostics::from_error(&error);

    let chain = diagnostics.chain().expect("source chain should be present");
    assert!(chain.len() <= 2_048, "chain must be bounded: {}", chain.len());
    assert!(chain.contains("..."), "the cut must stay observable: {chain}");
    let root_cause = diagnostics.root_cause().expect("root cause should be present");
    assert!(root_cause.len() <= 512, "root cause must be bounded: {}", root_cause.len());
    assert_eq!(error.to_string(), error_display, "projection must not rewrite the in-process error");
}

#[test]
fn log_diagnostics_mark_truncation_from_any_layer_not_just_the_cause() {
    let error = Error::new(ErrorCode::Internal, "algeon_datafusion.test.log_layer_truncation", "outer wrapper")
        .with_source(DataFusionError::Context(
            "middle ".to_string() + &"m".repeat(4_000),
            Box::new(DataFusionError::Execution("leaf cause".to_string())),
        ));

    let diagnostics = ErrorLogDiagnostics::from_error(&error);

    assert_eq!(diagnostics.cause(), Some("leaf cause"));
    assert!(diagnostics.truncated(), "an intermediate cut must stay observable");
}

#[test]
fn retained_capture_keeps_caller_schema_detail_outside_log_projection() {
    let error = Error::new(
        ErrorCode::FinalPlanRequirementUnsatisfied,
        "algeon_datafusion.test.strict_capture",
        "strict rejection probe",
    )
    .with_client_detail(ClientDetail::UnsupportedColumn {
        name: "countDetail".to_string(),
        data_type: "List(Int64)".to_string(),
        role: "projected",
    })
    .with_log_safe_fact("reason", "unsupported_type");

    let logs = error.facts().log_safe_projection().rendered().unwrap_or_default();
    assert!(!logs.contains("countDetail"), "operational logs must not carry caller schema detail: {logs}");

    let retained = error.facts().retained_capture_entries(error.client_detail());
    assert!(
        retained.contains(&("unsupported_column".to_string(), "countDetail".to_string())),
        "query-caller capture must retain the strict-rejection detail: {retained:?}"
    );
    assert!(retained.contains(&("unsupported_column_type".to_string(), "List(Int64)".to_string())), "{retained:?}");
    assert!(retained.contains(&("reason".to_string(), "unsupported_type".to_string())), "{retained:?}");
}

#[test]
fn source_diagnostics_render_project_errors_by_message_not_structured_display() {
    let native = QueryError::new(QueryErrorCode::CudfDependencyFailure, "query_engine.test", "native root cause")
        .with_fact("native_fact", "value");
    let adapter = Error::from_native("algeon_datafusion.test", "adapter wrapper", native);
    let error = DataFusionError::External(Box::new(adapter));

    let diagnostics = ErrorSourceDiagnostics::from_error(&error);
    let chain = diagnostics.chain().expect("source chain should be present");

    assert_eq!(chain, "adapter wrapper: native root cause");
    assert_eq!(diagnostics.root_cause(), Some("native root cause"));
    assert!(!chain.contains("code="), "{chain}");
    assert!(!chain.contains("facts="), "{chain}");
}

#[test]
fn log_diagnostics_keep_one_bounded_redacted_cudf_cause() {
    let cudf = new_with_for_test(
        CudfErrorCode::ForeignInvalidArgument,
        "cudf.test.log_diagnostics",
        "CUDF failure at: /home/customer/cudf/reader.cu:518: Parquet header parsing failed with code(s) 0x400",
    );
    let native = cudf_query_error(
        "query_engine.test.log_diagnostics",
        "failed to read bounded native local parquet source chunk with cudf",
        cudf,
    )
    .with_fact("path", "/srv/algeon/dataset/part-00000.parquet");
    let adapter = Error::from_native(
        "algeon_datafusion.test.log_diagnostics",
        "native cuGraph edge-source execution failed",
        native,
    );

    let diagnostics = ErrorLogDiagnostics::from_error(&adapter);

    assert_eq!(
        diagnostics.cause(),
        Some("CUDF failure at: /home/customer/cudf/reader.cu:518: Parquet header parsing failed with code(s) 0x400")
    );
    assert_eq!(diagnostics.source_path(), Some("/srv/algeon/dataset/part-00000.parquet"));
    assert!(!diagnostics.truncated());
}

#[test]
fn log_diagnostics_redact_credentials_and_bound_foreign_messages() {
    let credentials = TestSourceError::leaf("request failed Authorization=Bearer-secret");
    assert_eq!(ErrorLogDiagnostics::from_error(&credentials).cause(), Some("<redacted-credential>"));

    let uri = Error::new(ErrorCode::Internal, "algeon_datafusion.test.log_uri", "read failed")
        .with_fact("path", "s3://user:secret@bucket/private.parquet");
    assert_eq!(ErrorLogDiagnostics::from_error(&uri).source_path(), Some("<redacted-uri>"));

    let long =
        TestSourceError { message: Box::leak("x".repeat(MAX_LOG_CAUSE_BYTES + 1).into_boxed_str()), source: None };
    let diagnostics = ErrorLogDiagnostics::from_error(&long);
    assert_eq!(diagnostics.cause().expect("bounded cause").len(), MAX_LOG_CAUSE_BYTES);
    assert!(diagnostics.cause().expect("bounded cause").ends_with("..."));
    assert!(diagnostics.truncated());
}

fn cudf_allocation_error_for_test(
    code: CudfErrorCode,
    failure_source: Option<cudf::CudfErrorFailureSource>,
    with_budget_scope: bool,
) -> cudf::Error {
    use cudf::testing::error;
    let mut error = error::new_with_for_test(code, "cudf.test.allocation_attribution", "synthetic cudf failure");
    if let Some(failure_source) = failure_source {
        error = error::with_ffi_detail_for_test(
            error,
            cudf::CudfErrorDetail {
                status: 0,
                failure_source,
                domain_id: 0,
                member_id: 0,
                generation: 0,
                requested_bytes: 4096,
                alignment: 256,
                cuda_error: -1,
                message: "synthetic allocation failure".to_string(),
            },
        );
    }
    if with_budget_scope {
        error = error::with_budget_exhaustion_for_test(
            error,
            error::budget_exhaustion_detail_for_test(
                1024,
                1024,
                1024,
                512,
                512,
                4096,
                1,
                "test_boundary",
                "cudf.test.allocation_attribution",
            ),
        );
    }
    error
}

#[test]
fn cudf_physical_failure_with_budget_scope_is_device_out_of_memory() {
    let error = Error::from_cudf(
        "algeon_datafusion.test.allocation_attribution",
        "wrapped cudf failure",
        cudf_allocation_error_for_test(CudfErrorCode::OutOfMemory, Some(cudf::CudfErrorFailureSource::Physical), true),
    );

    assert_eq!(error.code(), ErrorCode::DeviceOutOfMemory);
}

#[test]
fn cudf_reservation_limit_is_query_memory_limit_exceeded() {
    let error = Error::from_cudf(
        "algeon_datafusion.test.allocation_attribution",
        "wrapped cudf failure",
        cudf_allocation_error_for_test(
            CudfErrorCode::OutOfMemory,
            Some(cudf::CudfErrorFailureSource::ReservationLimit),
            true,
        ),
    );

    assert_eq!(error.code(), ErrorCode::QueryMemoryLimitExceeded);
}

#[test]
fn cudf_unknown_allocation_with_scope_usage_is_not_a_cap_rejection() {
    let error = Error::from_cudf(
        "algeon_datafusion.test.allocation_attribution",
        "wrapped cudf failure",
        cudf_allocation_error_for_test(CudfErrorCode::OutOfMemory, Some(cudf::CudfErrorFailureSource::Unknown), true),
    );

    assert_eq!(error.code(), ErrorCode::DeviceAllocationFailed);
}

#[test]
fn cudf_dependency_failure_through_a_budgeted_scope_is_not_a_cap_rejection() {
    let error = Error::from_cudf(
        "algeon_datafusion.test.allocation_attribution",
        "wrapped cudf failure",
        cudf_allocation_error_for_test(CudfErrorCode::IoFailure, None, true),
    );

    assert_eq!(error.code(), ErrorCode::CudfDependencyFailure);
}

#[test]
fn adapter_and_native_boundaries_agree_on_cudf_allocation_attribution() {
    let cases = [
        (
            CudfErrorCode::OutOfMemory,
            Some(cudf::CudfErrorFailureSource::Physical),
            true,
            ErrorCode::DeviceOutOfMemory,
            QueryErrorCode::DeviceOutOfMemory,
        ),
        (
            CudfErrorCode::OutOfMemory,
            Some(cudf::CudfErrorFailureSource::ReservationLimit),
            true,
            ErrorCode::QueryMemoryLimitExceeded,
            QueryErrorCode::QueryMemoryLimitExceeded,
        ),
        (
            CudfErrorCode::OutOfMemory,
            Some(cudf::CudfErrorFailureSource::Unknown),
            true,
            ErrorCode::DeviceAllocationFailed,
            QueryErrorCode::DeviceAllocationFailed,
        ),
        (CudfErrorCode::IoFailure, None, true, ErrorCode::CudfDependencyFailure, QueryErrorCode::CudfDependencyFailure),
    ];

    for (index, (code, failure_source, with_budget_scope, expected_adapter, expected_native)) in
        cases.into_iter().enumerate()
    {
        let adapter = Error::from_cudf(
            "algeon_datafusion.test.allocation_attribution",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(code, failure_source, with_budget_scope),
        );
        assert_eq!(adapter.code(), expected_adapter, "adapter mismatch for case {index}");
        let native = cudf_query_error(
            "query_engine.test.allocation_attribution",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(code, failure_source, with_budget_scope),
        );
        assert_eq!(native.code(), expected_native, "native mismatch for case {index}");
    }
}

const ADAPTER_STATUS_CASES: &[(ErrorCode, ErrorStatus)] = &[
    (ErrorCode::Cancelled, ErrorStatus::Permanent),
    (ErrorCode::SqlParse, ErrorStatus::Permanent),
    (ErrorCode::InvalidRequest, ErrorStatus::Permanent),
    (ErrorCode::InvalidConfiguration, ErrorStatus::Permanent),
    (ErrorCode::DataFusionPlanning, ErrorStatus::Permanent),
    (ErrorCode::PlanningTimeout, ErrorStatus::Permanent),
    (ErrorCode::ObjectNotFound, ErrorStatus::Permanent),
    (ErrorCode::SourceAccessDenied, ErrorStatus::Permanent),
    (ErrorCode::SourceUnauthenticated, ErrorStatus::Permanent),
    (ErrorCode::SourceObjectNotFound, ErrorStatus::Permanent),
    (ErrorCode::SourceTemporaryFailure, ErrorStatus::Temporary),
    (ErrorCode::SourceDependencyFailure, ErrorStatus::Permanent),
    (ErrorCode::FieldNotFound, ErrorStatus::Permanent),
    (ErrorCode::GraphInputInvalid, ErrorStatus::Permanent),
    (ErrorCode::RequiredGpuOperationUnsupported, ErrorStatus::Permanent),
    (ErrorCode::FinalPlanRequirementUnsatisfied, ErrorStatus::Permanent),
    (ErrorCode::UnsupportedOperator, ErrorStatus::Permanent),
    (ErrorCode::UnsupportedExpression, ErrorStatus::Permanent),
    (ErrorCode::UnsupportedType, ErrorStatus::Permanent),
    (ErrorCode::InvalidExecutionPlan, ErrorStatus::Permanent),
    (ErrorCode::InvalidChildCount, ErrorStatus::Permanent),
    (ErrorCode::CudfInvalidInput, ErrorStatus::Permanent),
    (ErrorCode::CudfUnsupported, ErrorStatus::Permanent),
    (ErrorCode::SizeOverflow, ErrorStatus::Permanent),
    (ErrorCode::HostMaterializationLimitExceeded, ErrorStatus::Permanent),
    (ErrorCode::HostStagingLimitExceeded, ErrorStatus::Permanent),
    (ErrorCode::HostMemoryExhausted, ErrorStatus::Temporary),
    (ErrorCode::CudfTemporaryFailure, ErrorStatus::Temporary),
    (ErrorCode::CudfPersistentFailure, ErrorStatus::Persistent),
    (ErrorCode::CudfDependencyFailure, ErrorStatus::Permanent),
    (ErrorCode::AlgorithmDidNotConverge, ErrorStatus::Permanent),
    (ErrorCode::CudfInvariantViolation, ErrorStatus::Permanent),
    (ErrorCode::MemoryReservationFailed, ErrorStatus::Temporary),
    (ErrorCode::ServiceOverloaded, ErrorStatus::Temporary),
    (ErrorCode::QueryBudgetUnavailable, ErrorStatus::Temporary),
    (ErrorCode::QueryMemoryLimitExceeded, ErrorStatus::Permanent),
    (ErrorCode::DeviceOutOfMemory, ErrorStatus::Temporary),
    (ErrorCode::DeviceAllocationFailed, ErrorStatus::Temporary),
    (ErrorCode::DeviceFault, ErrorStatus::Persistent),
    (ErrorCode::CleanupFailed, ErrorStatus::Persistent),
    (ErrorCode::InternalInvariantViolation, ErrorStatus::Permanent),
    (ErrorCode::FlightFrameTooLarge, ErrorStatus::Permanent),
    (ErrorCode::ServerTransportFailure, ErrorStatus::Permanent),
    (ErrorCode::Internal, ErrorStatus::Permanent),
];

const NATIVE_ADAPTER_STATUS_CASES: &[(QueryErrorCode, ErrorCode, QueryErrorStatus)] = &[
    (QueryErrorCode::Cancelled, ErrorCode::Cancelled, QueryErrorStatus::Permanent),
    (QueryErrorCode::UnsupportedPipelineContract, ErrorCode::UnsupportedOperator, QueryErrorStatus::Permanent),
    (QueryErrorCode::UnsupportedOperator, ErrorCode::UnsupportedOperator, QueryErrorStatus::Permanent),
    (QueryErrorCode::UnsupportedExpression, ErrorCode::UnsupportedExpression, QueryErrorStatus::Permanent),
    (QueryErrorCode::UnsupportedType, ErrorCode::UnsupportedType, QueryErrorStatus::Permanent),
    (QueryErrorCode::InvalidPlan, ErrorCode::InvalidExecutionPlan, QueryErrorStatus::Permanent),
    (QueryErrorCode::InvalidAttemptCorrelation, ErrorCode::InvalidRequest, QueryErrorStatus::Permanent),
    (QueryErrorCode::InvalidConfiguration, ErrorCode::InvalidConfiguration, QueryErrorStatus::Permanent),
    (QueryErrorCode::GraphInputInvalid, ErrorCode::GraphInputInvalid, QueryErrorStatus::Permanent),
    (QueryErrorCode::CudfInvalidInput, ErrorCode::CudfInvalidInput, QueryErrorStatus::Permanent),
    (QueryErrorCode::CudfUnsupported, ErrorCode::CudfUnsupported, QueryErrorStatus::Permanent),
    (QueryErrorCode::SizeOverflow, ErrorCode::SizeOverflow, QueryErrorStatus::Permanent),
    (
        QueryErrorCode::HostMaterializationLimitExceeded,
        ErrorCode::HostMaterializationLimitExceeded,
        QueryErrorStatus::Permanent,
    ),
    (QueryErrorCode::CudfTemporaryFailure, ErrorCode::CudfTemporaryFailure, QueryErrorStatus::Temporary),
    (QueryErrorCode::CudfPersistentFailure, ErrorCode::CudfPersistentFailure, QueryErrorStatus::Persistent),
    (QueryErrorCode::CudfDependencyFailure, ErrorCode::CudfDependencyFailure, QueryErrorStatus::Permanent),
    (QueryErrorCode::AlgorithmDidNotConverge, ErrorCode::AlgorithmDidNotConverge, QueryErrorStatus::Permanent),
    (QueryErrorCode::SourceAccessDenied, ErrorCode::SourceAccessDenied, QueryErrorStatus::Permanent),
    (QueryErrorCode::SourceUnauthenticated, ErrorCode::SourceUnauthenticated, QueryErrorStatus::Permanent),
    (QueryErrorCode::SourceObjectNotFound, ErrorCode::SourceObjectNotFound, QueryErrorStatus::Permanent),
    (QueryErrorCode::SourceTemporaryFailure, ErrorCode::SourceTemporaryFailure, QueryErrorStatus::Temporary),
    (QueryErrorCode::CudfInvariantViolation, ErrorCode::CudfInvariantViolation, QueryErrorStatus::Permanent),
    (QueryErrorCode::MemoryReservationFailed, ErrorCode::MemoryReservationFailed, QueryErrorStatus::Temporary),
    (QueryErrorCode::ServiceOverloaded, ErrorCode::ServiceOverloaded, QueryErrorStatus::Temporary),
    (QueryErrorCode::QueryBudgetUnavailable, ErrorCode::QueryBudgetUnavailable, QueryErrorStatus::Temporary),
    (QueryErrorCode::QueryMemoryLimitExceeded, ErrorCode::QueryMemoryLimitExceeded, QueryErrorStatus::Permanent),
    (QueryErrorCode::DeviceOutOfMemory, ErrorCode::DeviceOutOfMemory, QueryErrorStatus::Temporary),
    (QueryErrorCode::DeviceAllocationFailed, ErrorCode::DeviceAllocationFailed, QueryErrorStatus::Temporary),
    (QueryErrorCode::DeviceFault, ErrorCode::DeviceFault, QueryErrorStatus::Persistent),
    (QueryErrorCode::CleanupFailed, ErrorCode::CleanupFailed, QueryErrorStatus::Persistent),
    (QueryErrorCode::InternalInvariantViolation, ErrorCode::InternalInvariantViolation, QueryErrorStatus::Permanent),
    (QueryErrorCode::Internal, ErrorCode::Internal, QueryErrorStatus::Permanent),
];

#[test]
fn every_adapter_code_has_the_expected_retry_status() {
    let mapped_codes = ADAPTER_STATUS_CASES.iter().map(|(code, _)| *code).collect::<HashSet<_>>();
    let all_codes = ErrorCode::ALL.iter().copied().collect::<HashSet<_>>();
    assert_eq!(mapped_codes.len(), ADAPTER_STATUS_CASES.len(), "adapter status policy table contains duplicate codes");
    assert_eq!(
        mapped_codes, all_codes,
        "adapter status policy table must cover every ErrorCode::ALL variant exactly once"
    );

    for &(code, expected_status) in ADAPTER_STATUS_CASES {
        let error = Error::new(code, "algeon_datafusion.test.retry_policy", "retry policy test");

        assert_eq!(code.status(), expected_status, "unexpected status for {code}");
        assert_eq!(error.status(), expected_status, "Error disagrees with its code for {code}");
    }
}

#[test]
fn every_native_status_is_preserved_at_the_adapter_boundary() {
    let mapped_codes = NATIVE_ADAPTER_STATUS_CASES.iter().map(|(code, _, _)| *code).collect::<HashSet<_>>();
    let engine_codes = QueryErrorCode::ALL.iter().copied().collect::<HashSet<_>>();
    assert_eq!(mapped_codes.len(), NATIVE_ADAPTER_STATUS_CASES.len(), "native mapping table contains duplicate codes");
    assert_eq!(mapped_codes, engine_codes, "native mapping table must cover every engine error code");

    for &(native_code, expected_adapter_code, expected_native_status) in NATIVE_ADAPTER_STATUS_CASES {
        let native = QueryError::new(native_code, "query_engine.test.retry_policy", "retry policy test");
        assert_eq!(native_code.status(), expected_native_status, "unexpected native status for {native_code}");
        assert_eq!(native.status(), expected_native_status, "QueryError disagrees with its code for {native_code}");

        let adapter = Error::from_native("algeon_datafusion.test.retry_policy", "wrapped native failure", native);
        assert_eq!(adapter.code(), expected_adapter_code, "unexpected adapter mapping for {native_code}");
        assert_eq!(adapter.status().as_str(), expected_native_status.as_str(), "status changed for {native_code}");
        assert_eq!(adapter.fact_value("native_status"), Some(expected_native_status.as_str()));
    }
}

#[test]
fn error_code_projections_are_stable_policy_contracts() {
    assert_eq!(ErrorCode::Cancelled.as_str(), "cancelled");
    assert_eq!(ErrorCode::Cancelled.kind(), ErrorKind::Cancelled);
    assert_eq!(ErrorCode::Cancelled.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SqlParse.as_str(), "sql_parse");
    assert_eq!(ErrorCode::SqlParse.kind(), ErrorKind::InvalidInput);
    assert_eq!(ErrorCode::SqlParse.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::InvalidRequest.as_str(), "invalid_request");
    assert_eq!(ErrorCode::InvalidRequest.kind(), ErrorKind::InvalidInput);
    assert_eq!(ErrorCode::InvalidRequest.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::PlanningTimeout.as_str(), "planning_timeout");
    assert_eq!(ErrorCode::PlanningTimeout.kind(), ErrorKind::Planning);
    assert_eq!(ErrorCode::PlanningTimeout.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::ObjectNotFound.as_str(), "object_not_found");
    assert_eq!(ErrorCode::ObjectNotFound.kind(), ErrorKind::NotFound);
    assert_eq!(ErrorCode::ObjectNotFound.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SourceAccessDenied.as_str(), "source_access_denied");
    assert_eq!(ErrorCode::SourceAccessDenied.kind(), ErrorKind::PermissionDenied);
    assert_eq!(ErrorCode::SourceAccessDenied.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SourceUnauthenticated.as_str(), "source_unauthenticated");
    assert_eq!(ErrorCode::SourceUnauthenticated.kind(), ErrorKind::Unauthenticated);
    assert_eq!(ErrorCode::SourceUnauthenticated.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SourceObjectNotFound.as_str(), "source_object_not_found");
    assert_eq!(ErrorCode::SourceObjectNotFound.kind(), ErrorKind::NotFound);
    assert_eq!(ErrorCode::SourceObjectNotFound.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SourceTemporaryFailure.as_str(), "source_temporary_failure");
    assert_eq!(ErrorCode::SourceTemporaryFailure.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::SourceTemporaryFailure.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::FieldNotFound.as_str(), "field_not_found");
    assert_eq!(ErrorCode::FieldNotFound.kind(), ErrorKind::InvalidInput);
    assert_eq!(ErrorCode::FieldNotFound.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::RequiredGpuOperationUnsupported.as_str(), "required_gpu_operation_unsupported");
    assert_eq!(ErrorCode::RequiredGpuOperationUnsupported.kind(), ErrorKind::Unsupported);
    assert_eq!(ErrorCode::RequiredGpuOperationUnsupported.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::FinalPlanRequirementUnsatisfied.as_str(), "final_plan_requirement_unsatisfied");
    assert_eq!(ErrorCode::FinalPlanRequirementUnsatisfied.kind(), ErrorKind::Planning);
    assert_eq!(ErrorCode::FinalPlanRequirementUnsatisfied.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::UnsupportedExpression.as_str(), "unsupported_expression");
    assert_eq!(ErrorCode::UnsupportedExpression.kind(), ErrorKind::Unsupported);
    assert_eq!(ErrorCode::UnsupportedExpression.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::InvalidConfiguration.as_str(), "invalid_configuration");
    assert_eq!(ErrorCode::InvalidConfiguration.kind(), ErrorKind::InvalidInput);
    assert_eq!(ErrorCode::InvalidConfiguration.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::SizeOverflow.as_str(), "size_overflow");
    assert_eq!(ErrorCode::SizeOverflow.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::SizeOverflow.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::HostMaterializationLimitExceeded.as_str(), "host_materialization_limit_exceeded");
    assert_eq!(ErrorCode::HostMaterializationLimitExceeded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::HostMaterializationLimitExceeded.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::HostStagingLimitExceeded.as_str(), "host_staging_limit_exceeded");
    assert_eq!(ErrorCode::HostStagingLimitExceeded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::HostStagingLimitExceeded.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::HostMemoryExhausted.as_str(), "host_memory_exhausted");
    assert_eq!(ErrorCode::HostMemoryExhausted.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::HostMemoryExhausted.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::CudfPersistentFailure.as_str(), "cudf_persistent_failure");
    assert_eq!(ErrorCode::CudfPersistentFailure.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::CudfPersistentFailure.status(), ErrorStatus::Persistent);

    assert_eq!(ErrorCode::MemoryReservationFailed.as_str(), "memory_reservation_failed");
    assert_eq!(ErrorCode::MemoryReservationFailed.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::MemoryReservationFailed.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::ServiceOverloaded.as_str(), "service_overloaded");
    assert_eq!(ErrorCode::ServiceOverloaded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::ServiceOverloaded.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::QueryBudgetUnavailable.as_str(), "query_budget_unavailable");
    assert_eq!(ErrorCode::QueryBudgetUnavailable.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::QueryBudgetUnavailable.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::QueryMemoryLimitExceeded.as_str(), "query_memory_limit_exceeded");
    assert_eq!(ErrorCode::QueryMemoryLimitExceeded.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::QueryMemoryLimitExceeded.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::DeviceOutOfMemory.as_str(), "device_out_of_memory");
    assert_eq!(ErrorCode::DeviceOutOfMemory.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::DeviceOutOfMemory.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::DeviceAllocationFailed.as_str(), "device_allocation_failed");
    assert_eq!(ErrorCode::DeviceAllocationFailed.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::DeviceAllocationFailed.status(), ErrorStatus::Temporary);

    assert_eq!(ErrorCode::DeviceFault.as_str(), "device_fault");
    assert_eq!(ErrorCode::DeviceFault.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::DeviceFault.status(), ErrorStatus::Persistent);

    assert_eq!(ErrorCode::CleanupFailed.as_str(), "cleanup_failed");
    assert_eq!(ErrorCode::CleanupFailed.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::CleanupFailed.status(), ErrorStatus::Persistent);

    assert_eq!(ErrorCode::InternalInvariantViolation.as_str(), "internal_invariant_violation");
    assert_eq!(ErrorCode::InternalInvariantViolation.kind(), ErrorKind::InvariantViolation);
    assert_eq!(ErrorCode::InternalInvariantViolation.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::FlightFrameTooLarge.as_str(), "flight_frame_too_large");
    assert_eq!(ErrorCode::FlightFrameTooLarge.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(ErrorCode::FlightFrameTooLarge.status(), ErrorStatus::Permanent);

    assert_eq!(ErrorCode::ServerTransportFailure.as_str(), "server_transport_failure");
    assert_eq!(ErrorCode::ServerTransportFailure.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ErrorCode::ServerTransportFailure.status(), ErrorStatus::Permanent);
}

#[test]
fn native_source_storage_errors_preserve_adapter_status_mapping() {
    for (native_code, adapter_code, adapter_kind, adapter_status) in [
        (
            QueryErrorCode::SourceAccessDenied,
            ErrorCode::SourceAccessDenied,
            ErrorKind::PermissionDenied,
            ErrorStatus::Permanent,
        ),
        (
            QueryErrorCode::SourceUnauthenticated,
            ErrorCode::SourceUnauthenticated,
            ErrorKind::Unauthenticated,
            ErrorStatus::Permanent,
        ),
        (
            QueryErrorCode::SourceObjectNotFound,
            ErrorCode::SourceObjectNotFound,
            ErrorKind::NotFound,
            ErrorStatus::Permanent,
        ),
        (
            QueryErrorCode::SourceTemporaryFailure,
            ErrorCode::SourceTemporaryFailure,
            ErrorKind::DependencyFailure,
            ErrorStatus::Temporary,
        ),
    ] {
        let native_error = QueryError::new(
            native_code,
            "query_engine.source_metadata.object_store_footer",
            "source object storage failure",
        )
        .with_fact("object_store_error_kind", native_code.as_str());

        let adapter_error = Error::from_native(
            "algeon_datafusion.native.execute",
            "Algeon native query execution failed",
            native_error,
        );

        assert_eq!(adapter_error.code(), adapter_code);
        assert_eq!(adapter_error.kind(), adapter_kind);
        assert_eq!(adapter_error.status(), adapter_status);
        assert_eq!(adapter_error.fact_value("native_code"), Some(native_code.as_str()));
    }
}

#[test]
fn native_service_overloaded_error_preserves_adapter_classification_and_facts() {
    let native_error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "query_engine.query_service.admit",
        "native query service is overloaded",
    )
    .with_fact("retry_action", "backoff");

    let adapter_error =
        Error::from_native("algeon_datafusion.native.execute", "Algeon native query execution failed", native_error);

    assert_eq!(adapter_error.code(), ErrorCode::ServiceOverloaded);
    assert_eq!(adapter_error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(adapter_error.status(), ErrorStatus::Temporary);
    assert_eq!(adapter_error.fact_value("native_code"), Some("service_overloaded"));
    assert_eq!(adapter_error.fact_value("retry_action"), Some("backoff"));
}

#[test]
fn native_query_error_json_includes_triage_summary() {
    let native_error = QueryError::new(
        QueryErrorCode::MemoryReservationFailed,
        "query_engine.test.memory",
        "memory reservation failed",
    )
    .with_query_id("query-err")
    .with_fact("node_id", "n3")
    .with_fact("compiled_task_id", "task-3");

    let summary = native_error.triage_summary();
    assert_eq!(summary.verdict, TriageVerdict::Failed);
    assert_eq!(summary.signals[0].code, TriageCode::MemoryReservationDenied);

    let json = native_error.to_json();
    assert_eq!(json["triage"]["verdict"], "failed");
    assert_eq!(json["triage"]["signals"][0]["code"], "memory_reservation_denied");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["query_id"], "query-err");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["plan_node_id"], "n3");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["compiled_task_id"], "task-3");
}

#[test]
fn native_query_error_json_uses_fact_only_correlation_ids() {
    let native_error = QueryError::new(
        QueryErrorCode::MemoryReservationFailed,
        "query_engine.test.memory",
        "memory reservation failed",
    )
    .with_fact("query_id", "fact-query")
    .with_fact("trace_id", "fact-trace")
    .with_fact("node_id", "n3");

    let json = native_error.to_json();
    assert_eq!(json["query_id"], "fact-query");
    assert_eq!(json["trace_id"], "fact-trace");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["query_id"], "fact-query");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["plan_node_id"], "n3");
}

#[test]
fn native_query_error_triage_codes_preserve_invalid_input_and_unsupported_capability() {
    assert_eq!(TriageCode::from(QueryErrorCode::CudfInvalidInput), TriageCode::InvalidInput);
    assert_eq!(TriageCode::from(QueryErrorCode::InvalidConfiguration), TriageCode::InvalidInput);
    assert_eq!(TriageCode::from(QueryErrorCode::CudfUnsupported), TriageCode::UnsupportedOperator);
}

#[test]
fn native_error_message_is_preserved_as_adapter_fact() {
    let native_error = QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.graph_algorithm.betweenness_seed_policy",
        "exact betweenness requires explicit seeds or k above the configured vertex threshold",
    );

    let adapter_error = Error::from_native(
        "algeon_datafusion.cugraph_sql.execute_gpu",
        "native cuGraph SQL GPU-resident query execution failed",
        native_error,
    );

    assert_eq!(
        adapter_error.fact_value("native_message"),
        Some("exact betweenness requires explicit seeds or k above the configured vertex threshold")
    );
}

#[test]
fn native_correlation_facts_follow_stamped_query_error() {
    let native_error = QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.query_service.context",
        "query service handle was released before context acquisition",
    )
    .with_fact("query_id", "service-local-query")
    .with_query_id("request-query");

    let adapter_error = Error::from_native(
        "algeon_datafusion.native_exec.context",
        "failed to acquire query service algeon native execution context",
        native_error,
    );

    assert_eq!(adapter_error.fact_value("query_id"), Some("request-query"));
}

#[test]
fn native_fact_collisions_preserve_adapter_facts_and_record_skipped_keys() {
    let native_error = QueryError::new(
        QueryErrorCode::UnsupportedPipelineContract,
        "query_engine.plan.lower",
        "native plan shape is not supported",
    )
    .with_fact("native_code", "lower_layer_override")
    .with_fact("native_operation", "lower_layer_operation")
    .with_fact("native_fact_collision_keys", "lower_layer_collision_value")
    .with_fact("fallback_reason", "unsupported_shape");

    let adapter_error =
        Error::from_native("algeon_datafusion.native.lower", "algeon native lowering failed", native_error);

    assert_eq!(adapter_error.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(adapter_error.fact_value("native_code"), Some("unsupported_pipeline_contract"));
    assert_eq!(adapter_error.fact_value("native_operation"), Some("query_engine.plan.lower"));
    assert_eq!(adapter_error.fact_value("fallback_reason"), Some("unsupported_shape"));
    assert_eq!(
        adapter_error.fact_value("native_fact_collision_keys"),
        Some("native_code,native_fact_collision_keys,native_operation")
    );
}

#[test]
fn error_display_is_human_only_while_typed_fields_remain_available() {
    let error = Error::new(
        ErrorCode::FinalPlanRequirementUnsatisfied,
        "algeon_datafusion.planner.final_plan_requirement",
        "final physical plan did not satisfy the configured requirement",
    )
    .with_fact("observed_final_plan_disposition", "datafusion")
    .with_fact("requirement", "no_datafusion_cpu")
    .with_source(std::io::Error::other("sensitive dependency source"));

    assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(error.kind(), ErrorKind::Planning);
    assert_eq!(error.status(), ErrorStatus::Permanent);
    assert_eq!(error.operation(), "algeon_datafusion.planner.final_plan_requirement");
    assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("datafusion"));

    assert_eq!(error.to_string(), "final physical plan did not satisfy the configured requirement");
    for hidden in [
        "final_plan_requirement_unsatisfied",
        "algeon_datafusion.planner.final_plan_requirement",
        "observed_final_plan_disposition",
        "no_datafusion_cpu",
        "sensitive dependency source",
    ] {
        assert!(!error.to_string().contains(hidden), "Display leaked structured or source data: {hidden}");
    }
    let debug = format!("{error:?}");
    assert!(debug.contains("FinalPlanRequirementUnsatisfied"), "{debug}");
    assert!(debug.contains("algeon_datafusion.planner.final_plan_requirement"), "{debug}");
    assert!(!debug.contains("observed_final_plan_disposition"), "{debug}");
    assert!(!debug.contains("no_datafusion_cpu"), "{debug}");
}

fn algeon_error(operation: &'static str, message: &'static str) -> Error {
    Error::new(ErrorCode::CudfDependencyFailure, operation, message)
        .with_fact("device_ordinal", 2)
        .with_source(std::io::Error::other("CUDA dependency detail"))
}

#[test]
fn algeon_error_uses_exact_datafusion_external_prose_and_preserves_structured_fields() {
    let error: DataFusionError = algeon_error("algeon_datafusion.native.execute", "native GPU execution failed").into();

    assert!(matches!(error, DataFusionError::External(_)));
    assert_eq!(error.to_string(), "External error: native GPU execution failed");
    for hidden in ["cudf_dependency_failure", "device_ordinal", "CUDA dependency detail"] {
        assert!(!error.to_string().contains(hidden), "DataFusion Display leaked structured or source data: {hidden}");
    }

    let algeon = error.find_algeon_error().expect("typed Algeon error must remain discoverable");
    assert_eq!(algeon.code(), ErrorCode::CudfDependencyFailure);
    assert_eq!(algeon.kind(), ErrorKind::DependencyFailure);
    assert_eq!(algeon.status(), ErrorStatus::Permanent);
    assert_eq!(algeon.operation(), "algeon_datafusion.native.execute");
    assert_eq!(algeon.fact_value("device_ordinal"), Some("2"));
    assert!(algeon.source_error().is_some());
}

#[test]
fn plain_pre_algeon_datafusion_failure_keeps_its_variant_and_prose() {
    let error = DataFusionError::Plan("unknown function plain_udf".to_string());

    assert!(matches!(error, DataFusionError::Plan(_)));
    assert_eq!(error.to_string(), "Error during planning: unknown function plain_udf");
    assert!(matches!(error.find_root(), DataFusionError::Plan(_)));
    assert!(error.find_algeon_error().is_none());
}

fn algeon_control_plane_failure() -> Result<()> {
    Err(algeon_error("algeon_datafusion.control_plane.open", "failed to open Algeon backend"))
}

fn datafusion_owned_surface() -> DataFusionResult<()> {
    algeon_control_plane_failure()?;
    Ok(())
}

#[test]
fn question_mark_converts_algeon_result_at_datafusion_boundary() {
    let error = datafusion_owned_surface().expect_err("Algeon failure must propagate");

    let DataFusionError::External(source) = &error else {
        panic!("Algeon Result must convert exactly at the DataFusion-owned boundary: {error:?}");
    };
    assert!(source.downcast_ref::<Error>().is_some(), "External must preserve the typed Algeon source");
    assert_eq!(error.to_string(), "External error: failed to open Algeon backend");
    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.control_plane.open"));
}

fn external_algeon_error(operation: &'static str) -> DataFusionError {
    algeon_error(operation, "wrapped native GPU execution failed").into()
}

#[test]
fn datafusion_wrappers_keep_find_root_behavior_and_algeon_discovery() {
    let wrappers = [
        DataFusionError::Context(
            "query context".to_string(),
            Box::new(external_algeon_error("algeon_datafusion.test.context")),
        ),
        DataFusionError::Diagnostic(
            Box::new(Diagnostic::new_error("query diagnostic", None)),
            Box::new(external_algeon_error("algeon_datafusion.test.diagnostic")),
        ),
        DataFusionError::Shared(Arc::new(external_algeon_error("algeon_datafusion.test.shared"))),
        DataFusionError::ArrowError(
            Box::new(ArrowError::from_external_error(Box::new(external_algeon_error("algeon_datafusion.test.arrow")))),
            None,
        ),
    ];

    for (error, expected_operation) in wrappers.into_iter().zip([
        "algeon_datafusion.test.context",
        "algeon_datafusion.test.diagnostic",
        "algeon_datafusion.test.shared",
        "algeon_datafusion.test.arrow",
    ]) {
        assert!(
            matches!(error.find_root(), DataFusionError::External(_)),
            "DataFusion find_root must still stop at its deepest DataFusion error: {error:?}"
        );
        assert_eq!(error.find_algeon_error().map(Error::operation), Some(expected_operation));
    }
}

#[test]
fn collection_preserves_datafusion_root_and_uses_documented_algeon_order() {
    let error = DataFusionError::Collection(vec![
        DataFusionError::Plan("plain first branch".to_string()),
        DataFusionError::Collection(vec![
            DataFusionError::Execution("plain nested branch".to_string()),
            external_algeon_error("algeon_datafusion.test.first"),
        ]),
        external_algeon_error("algeon_datafusion.test.second"),
    ]);

    assert!(matches!(error.find_root(), DataFusionError::Plan(message) if message == "plain first branch"));
    assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.first"));
}

#[derive(Debug, PartialEq, Eq)]
enum BackendDisposition {
    Algeon(ErrorCode),
    DataFusionPlanning,
    DataFusionResourceExhausted,
    DataFusionOther,
}

fn existing_datafusion_handler(error: &DataFusionError) -> BackendDisposition {
    match error.find_root() {
        DataFusionError::Plan(_) => BackendDisposition::DataFusionPlanning,
        DataFusionError::ResourcesExhausted(_) => BackendDisposition::DataFusionResourceExhausted,
        _ => BackendDisposition::DataFusionOther,
    }
}

fn algeon_aware_backend_handler(error: &DataFusionError) -> BackendDisposition {
    if let Some(algeon) = error.find_algeon_error() {
        return BackendDisposition::Algeon(algeon.code());
    }
    existing_datafusion_handler(error)
}

#[test]
fn backend_handler_adds_algeon_branch_then_reuses_existing_datafusion_fallback() {
    let algeon = DataFusionError::Context(
        "query".to_string(),
        Box::new(external_algeon_error("algeon_datafusion.test.backend")),
    );
    assert_eq!(algeon_aware_backend_handler(&algeon), BackendDisposition::Algeon(ErrorCode::CudfDependencyFailure));

    for plain in [
        DataFusionError::Plan("plain planning failure".to_string()),
        DataFusionError::ArrowError(
            Box::new(ArrowError::from_external_error(Box::new(DataFusionError::ResourcesExhausted(
                "plain memory pressure".to_string(),
            )))),
            None,
        ),
    ] {
        assert_eq!(algeon_aware_backend_handler(&plain), existing_datafusion_handler(&plain));
    }
}

#[test]
fn datafusion_conversion_preserves_typed_adapter_error() {
    let error = Error::new(
        ErrorCode::InvalidExecutionPlan,
        "algeon_datafusion.native.try_new",
        "cannot build native cuDF execution plan",
    )
    .with_fact("plan_node", "EmptyExec");

    let datafusion_error: DataFusionError = error.into();
    let DataFusionError::External(source) = datafusion_error else {
        panic!("adapter errors must cross the DataFusion boundary as typed external errors");
    };
    let adapter_error =
        source.downcast_ref::<Error>().expect("DataFusion external error should preserve crate::error::Error");

    assert_eq!(adapter_error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(adapter_error.kind(), ErrorKind::Planning);
    assert_eq!(adapter_error.fact_value("plan_node"), Some("EmptyExec"));
}

#[test]
fn native_cancelled_error_preserves_adapter_classification_and_facts() {
    let native_error = ExecutionCancellation::new();
    native_error.cancel();
    let adapter_error = Error::from_native(
        "algeon_datafusion.native_exec.execute",
        "Algeon native query execution failed",
        native_error.check("query_engine.exec.execute_query_plan").unwrap_err(),
    );

    assert_eq!(adapter_error.code(), ErrorCode::Cancelled);
    assert_eq!(adapter_error.kind(), ErrorKind::Cancelled);
    assert_eq!(adapter_error.status(), ErrorStatus::Permanent);
    assert_eq!(adapter_error.fact_value("native_code"), Some("cancelled"));
    assert_eq!(adapter_error.fact_value("cancellation_requested"), Some("true"));
}

#[test]
fn native_retry_error_preserves_resume_boundary_facts() {
    let native_error = QueryError::new(
        QueryErrorCode::MemoryReservationFailed,
        "query_engine.final_sort.topk_concat",
        "native final sort exhausted bounded memory",
    )
    .with_fact("retry_action", "reduce_final_sort_fan_in")
    .with_fact("resume_boundary", "final_sort")
    .with_fact("committed_chunks", 4)
    .with_fact("committed_partials", 4)
    .with_fact("merge_level", 2)
    .with_fact("merge_fan_in", 8);

    let adapter_error = Error::from_native(
        "algeon_datafusion.native_exec.execute",
        "Algeon native query execution failed",
        native_error,
    );

    assert_eq!(adapter_error.code(), ErrorCode::MemoryReservationFailed);
    assert_eq!(adapter_error.fact_value("native_operation"), Some("query_engine.final_sort.topk_concat"));
    assert_eq!(adapter_error.fact_value("retry_action"), Some("reduce_final_sort_fan_in"));
    assert_eq!(adapter_error.fact_value("resume_boundary"), Some("final_sort"));
    assert_eq!(adapter_error.fact_value("committed_chunks"), Some("4"));
    assert_eq!(adapter_error.fact_value("committed_partials"), Some("4"));
    assert_eq!(adapter_error.fact_value("merge_level"), Some("2"));
    assert_eq!(adapter_error.fact_value("merge_fan_in"), Some("8"));
}
