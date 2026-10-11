use std::{
    collections::BTreeMap,
    io::Write,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use arrow::{
    array::{Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::{decode::FlightRecordBatchStream, error::FlightError};
use datafusion::{
    catalog::{CatalogProvider, SchemaProvider},
    common::{Result as DataFusionResult, TableReference},
    datasource::{MemTable, TableProvider, provider_as_source},
    execution::SessionStateBuilder,
    logical_expr::LogicalPlanBuilder,
    prelude::SessionConfig,
};
use tracing::subscriber::{set_default, with_default};

use super::{execution::cancel_invalid_argument_status, *};
use crate::{
    error::client::MASKED_INTERNAL_ERROR_MESSAGE,
    gpu_functions::{GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION},
    server::{
        coordinator::{CoordinatorArtifact, QueryTerminalOutcome, RetainedUnavailableReason},
        error::{
            ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_OPERATION_METADATA_KEY, ERROR_STATUS_METADATA_KEY,
            REASON_METADATA_KEY,
        },
    },
    session::AlgeonSessionStateBuilderExt,
};

#[derive(Clone, Default)]
struct SharedLogBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for SharedLogBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log buffer lock").extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn cancel_invalid_argument_status_carries_adapter_metadata() {
    let invalid_query_id = vec![0xff];
    let status = cancel_invalid_argument_status(
        "Flight SQL cancel action contained a non-UTF-8 query id",
        "query_id_utf8",
        std::str::from_utf8(&invalid_query_id).unwrap_err(),
    );

    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(status.metadata().get(ERROR_CODE_METADATA_KEY).unwrap().to_str().unwrap(), "invalid_request");
    assert_eq!(status.metadata().get(ERROR_KIND_METADATA_KEY).unwrap().to_str().unwrap(), "invalid_input");
    assert_eq!(status.metadata().get(ERROR_STATUS_METADATA_KEY).unwrap().to_str().unwrap(), "permanent");
    assert_eq!(status.message(), "Flight SQL cancel action contained a non-UTF-8 query id");
}

#[test]
fn get_tables_metadata_log_uses_the_same_collection_error_for_fields_and_cause() {
    let target = Error::new(ErrorCode::Internal, "algeon_datafusion.test.collection_target", "target adapter failure")
        .with_fact("path", "/srv/algeon/target.parquet")
        .with_source(std::io::Error::other("target root cause"));
    let error = DataFusionError::Collection(vec![
        DataFusionError::Execution("unrelated first failure".to_string()),
        target.into(),
    ]);
    let buffer = SharedLogBuffer::default();
    let writer = buffer.clone();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_target(false)
        .with_writer(move || writer.clone())
        .finish();

    with_default(subscriber, || {
        DatafusionAlgeonFlightSqlService::warn_get_tables_metadata_error(
            "catalog", "schema", "table", "schema", &error,
        );
    });

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    assert!(output.contains("error_operation=\"algeon_datafusion.test.collection_target\""), "{output}");
    assert!(output.contains("error_cause=\"target root cause\""), "{output}");
    assert!(output.contains("error_source_path=\"/srv/algeon/target.parquet\""), "{output}");
    assert!(!output.contains("unrelated first failure"), "{output}");
}

#[test]
fn statement_status_log_flattens_resolved_sql_and_keeps_all_fields() {
    let buffer = SharedLogBuffer::default();
    let writer = buffer.clone();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_target(false)
        .with_writer(move || writer.clone())
        .finish();
    let mut status = Status::invalid_argument("terminal failure naming caller_column");
    status.metadata_mut().insert(ERROR_CODE_METADATA_KEY, MetadataValue::from_static("sql_planning"));
    status.metadata_mut().insert(REASON_METADATA_KEY, MetadataValue::from_static("unsupported_type"));

    with_default(subscriber, || {
        log_statement_status(
            "terminal-query",
            Some("SELECT 1\tFROM t\nWHERE x = 1"),
            "planning",
            Instant::now(),
            &status,
        );
    });

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    assert_eq!(output.lines().count(), 1, "resolved SQL must remain one log line: {output}");
    assert!(output.contains("query_id=\"terminal-query\""), "{output}");
    assert!(output.contains("sql=SELECT 1 FROM t WHERE x = 1"), "{output}");
    assert!(output.contains("outcome=\"failed\""), "{output}");
    assert!(output.contains("stage=\"planning\""), "{output}");
    assert!(output.contains("grpc_code=InvalidArgument"), "{output}");
    assert!(output.contains("elapsed_ms="), "{output}");
    assert!(output.contains("error_code=\"sql_planning\""), "{output}");
    assert!(output.contains("reason=\"unsupported_type\""), "{output}");
    assert!(!output.contains("caller_column"), "the status message must stay out of the log: {output}");
}

#[test]
fn statement_status_log_omits_sql_before_ticket_resolution() {
    let buffer = SharedLogBuffer::default();
    let writer = buffer.clone();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_target(false)
        .with_writer(move || writer.clone())
        .finish();
    let status = Status::invalid_argument("unknown ticket");

    with_default(subscriber, || {
        log_statement_status("unknown-query", None, "ticket", Instant::now(), &status);
    });

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    assert!(output.contains("query_id=\"unknown-query\""), "{output}");
    assert!(output.contains("outcome=\"failed\""), "{output}");
    assert!(output.contains("stage=\"ticket\""), "{output}");
    assert!(output.contains("grpc_code=InvalidArgument"), "{output}");
    assert!(output.contains("elapsed_ms="), "{output}");
    assert!(!output.contains("unknown ticket"), "the status message must stay out of the log: {output}");
    assert!(!output.contains("sql="), "unresolved tickets must not fabricate SQL context: {output}");
}

/// Failed statements log the bounded error code, never the status message:
/// planning and execution errors can name the caller's columns and values.
#[tokio::test]
async fn failed_statement_logs_omit_the_status_message_on_get_flight_info_and_do_get() {
    let buffer = SharedLogBuffer::default();
    let writer = buffer.clone();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_target(false)
        .with_writer(move || writer.clone())
        .finish();
    let _logs = set_default(subscriber);
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::new(SessionContext::new()),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vec!["caller_value"]))]).unwrap();
    service.ctx.register_table("caller_rows", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();

    let planning = service
        .get_flight_info_statement(
            CommandStatementQuery { query: "SELECT caller_column FROM caller_rows".to_string(), transaction_id: None },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect_err("an unknown column must fail planning");
    assert!(planning.message().contains("caller_column"), "the client still sees the planning error: {planning:?}");

    let response = service
        .get_flight_info_statement(
            CommandStatementQuery {
                query: "SELECT to_timestamp(s) FROM caller_rows".to_string(),
                transaction_id: None,
            },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect("a timestamp parse plans and fails only during DoGet");
    let query_id = query_id_from_flight_info(&response);
    let stream = service.execute_ticket(query_id, None).await.expect("DoGet returns a lazy stream").into_inner();
    let error = FlightRecordBatchStream::new_from_flight_data(stream.map_err(FlightError::from))
        .try_collect::<Vec<_>>()
        .await
        .expect_err("parsing the caller's value must fail the stream");
    let FlightError::Tonic(execution) = error else { panic!("DoGet must fail with a gRPC status: {error}") };

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    let failures = output.lines().filter(|line| line.contains("failed Flight SQL statement")).collect::<Vec<_>>();
    assert_eq!(failures.len(), 2, "one GetFlightInfo and one DoGet failure: {output}");
    for (line, status) in failures.iter().zip([&planning, &*execution]) {
        assert!(line.contains("error_code="), "{line}");
        assert!(!line.contains(status.message()), "the status message must stay out of the log: {line}");
    }
}

#[test]
fn execution_terminal_log_includes_resolved_sql() {
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::new(SessionContext::new()),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
    );
    let sql = "SELECT 1\nFROM user_rows";
    let planning = service
        .coordinator
        .register_with_id("terminal-execution-query", sql, None, service.runtime_observation_config)
        .expect("coordinator-owned fixture registers");
    let execution = planning.into_execution().expect("planning promotes to execution");
    let terminal = CoordinatorExecutionTerminal::new(
        execution,
        "terminal-execution-query".to_string(),
        Arc::from(sql),
        Instant::now(),
    );
    let buffer = SharedLogBuffer::default();
    let writer = buffer.clone();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_target(false)
        .with_writer(move || writer.clone())
        .finish();
    let status = Status::internal("stream failed");

    with_default(subscriber, || {
        terminal.finish_captured("stream", &status, CapturedError::from_status(&status));
    });

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    assert!(output.contains("query_id=\"terminal-execution-query\""), "{output}");
    assert!(output.contains("sql=SELECT 1 FROM user_rows"), "{output}");
    assert!(output.contains("stage=\"stream\""), "{output}");
    assert!(output.contains("grpc_code=Internal"), "{output}");
}

fn query_stats_payload(service: &DatafusionAlgeonFlightSqlService, query_id: &str) -> String {
    String::from_utf8(
        service.coordinator.query_stats(query_id).expect("completed Flight query should retain stats").to_vec(),
    )
    .expect("query stats are UTF-8")
}

fn query_id_from_flight_info(response: &Response<FlightInfo>) -> String {
    response
        .metadata()
        .get(QUERY_ID_METADATA_KEY)
        .and_then(|value| value.to_str().ok())
        .expect("FlightInfo response metadata carries a query id")
        .to_string()
}

async fn execute_ticket_batches(service: &DatafusionAlgeonFlightSqlService, query_id: String) -> Vec<RecordBatch> {
    let response =
        service.execute_ticket(query_id, None).await.expect("physical planning and execution should succeed");
    FlightRecordBatchStream::new_from_flight_data(response.into_inner().map_err(FlightError::from))
        .try_collect::<Vec<_>>()
        .await
        .expect("Flight result should decode into Arrow batches")
}

fn assert_single_sealed_attempt_report(payload: &str, expected_outcome: &str) {
    assert_eq!(payload.matches("# attempt_id=attempt-").count(), 1, "{payload}");
    assert_eq!(payload.matches("# attempt_outcome=").count(), 1, "{payload}");
    assert!(payload.contains(&format!("# attempt_outcome={expected_outcome}")), "{payload}");
    assert!(payload.contains(&format!("# outcome={expected_outcome}")), "{payload}");
    assert!(payload.contains("# sealed=true"), "{payload}");
}

#[test]
fn statement_execution_domain_classifier_is_fail_closed() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let validate_call = r#"{"relations":{"input":{"table":"input"}},"options":{}}"#;
    let metadata_queries = [
        "SHOW TABLES",
        "SHOW COLUMNS FROM t",
        "SHOW CREATE TABLE t",
        "DESCRIBE t",
        "DESC t",
        "SELECT table_name FROM information_schema.tables",
        "SELECT a.table_name FROM information_schema.tables a JOIN information_schema.columns b ON a.table_name = b.table_name",
        &format!(
            "SELECT function_name, provider, available, summary \
             FROM {GPU_LIST_FUNCTIONS_FUNCTION}() \
             WHERE provider IN ('cugraph', 'cuvs') \
             ORDER BY function_name"
        ),
        &format!(
            "SELECT function_name, provider \
             FROM {GPU_DESCRIBE_FUNCTION_FUNCTION}('cuvs_pca') \
             WHERE provider = 'cuvs' \
             ORDER BY function_name"
        ),
        &format!(
            "SELECT valid, error_code, message \
             FROM {GPU_VALIDATE_CALL_FUNCTION}('cuvs_pca', '{validate_call}') \
             WHERE valid \
             ORDER BY function_name"
        ),
        &format!(
            "SELECT i.table_name, g.function_name \
             FROM information_schema.tables i \
             JOIN {GPU_LIST_FUNCTIONS_FUNCTION}() g ON true \
             WHERE i.table_name IS NOT NULL \
             ORDER BY g.function_name"
        ),
    ];
    for sql in metadata_queries {
        let statement = service.parse_flight_statement(sql).unwrap();
        assert_eq!(service.classify_statement_candidate(&statement), StatementExecutionDomain::Metadata, "{sql}");
    }

    let data_queries = [
        "SELECT 1",
        "SHOW FUNCTIONS",
        "SHOW FULL TABLES",
        "SHOW CREATE VIEW t",
        "SELECT * FROM user_table",
        "SELECT i.table_name FROM information_schema.tables i JOIN user_table u ON true",
        "SELECT * FROM gpu_list_functions",
        "SELECT * FROM unknown_gpu_metadata_udtf()",
        "SELECT g.function_name, u.id FROM gpu_list_functions() g JOIN user_table u ON true",
    ];
    for sql in data_queries {
        let statement = service.parse_flight_statement(sql).unwrap();
        assert_eq!(service.classify_statement_candidate(&statement), StatementExecutionDomain::Data, "{sql}");
    }
}

#[test]
fn metadata_planning_removes_only_the_algeon_physical_rules() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let base_rules =
        service.ctx.state().physical_optimizers().iter().map(|rule| rule.name().to_string()).collect::<Vec<_>>();
    let metadata_rules = service
        .metadata_query_context()
        .state()
        .physical_optimizers()
        .iter()
        .map(|rule| rule.name().to_string())
        .collect::<Vec<_>>();
    let removed = [ALGEON_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME];

    assert!(removed.iter().all(|name| base_rules.iter().any(|rule| rule == name)), "{base_rules:?}");
    assert!(removed.iter().all(|name| metadata_rules.iter().all(|rule| rule != name)), "{metadata_rules:?}");
    assert_eq!(
        metadata_rules,
        base_rules.into_iter().filter(|rule| !removed.contains(&rule.as_str())).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn metadata_logical_proof_rejects_user_sources_and_source_free_queries() {
    let backend = super::service::default_test_backend();
    let state = backend
        .install_on(
            SessionStateBuilder::new_with_default_features()
                .with_config(SessionConfig::new().with_information_schema(true)),
        )
        .unwrap()
        .try_with_cuvs_sql()
        .unwrap()
        .build();
    let ctx = Arc::new(SessionContext::new_with_state(state));
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    ctx.register_table("user_table", Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap())).unwrap();
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::clone(&ctx),
        Arc::new(QueryAdmission::from_backend(&backend)),
        Duration::from_secs(30),
    );

    let validate_call = r#"{"relations":{"input":{"table":"input"}},"options":{}}"#;
    for sql in [
        format!(
            "SELECT function_name, provider \
             FROM {GPU_LIST_FUNCTIONS_FUNCTION}() \
             WHERE provider = 'cuvs' \
             ORDER BY function_name"
        ),
        format!(
            "SELECT function_name, provider \
             FROM {GPU_DESCRIBE_FUNCTION_FUNCTION}('cuvs_pca') \
             WHERE provider = 'cuvs' \
             ORDER BY function_name"
        ),
        format!(
            "SELECT valid, error_code, message \
             FROM {GPU_VALIDATE_CALL_FUNCTION}('cuvs_pca', '{validate_call}') \
             WHERE valid \
             ORDER BY function_name"
        ),
    ] {
        let plan = ctx.state().create_logical_plan(&sql).await.unwrap();
        service
            .verify_metadata_logical_plan(&plan, "algeon_datafusion.test.metadata_proof")
            .expect("real built-in GPU metadata providers must satisfy the metadata proof");
    }

    let information_schema =
        ctx.state().create_logical_plan("SELECT table_name FROM information_schema.tables").await.unwrap();
    service.verify_metadata_logical_plan(&information_schema, "algeon_datafusion.test.metadata_proof").unwrap();
    let describe = ctx.state().create_logical_plan("DESCRIBE user_table").await.unwrap();
    service.verify_metadata_logical_plan(&describe, "algeon_datafusion.test.metadata_proof").unwrap();
    let show_create = ctx.state().create_logical_plan("SHOW CREATE TABLE user_table").await.unwrap();
    service.verify_metadata_logical_plan(&show_create, "algeon_datafusion.test.metadata_proof").unwrap();

    let spoofed_provider = Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap());
    let spoofed_plan = LogicalPlanBuilder::scan(
        TableReference::Bare { table: format!("{GPU_LIST_FUNCTIONS_FUNCTION}()").into() },
        provider_as_source(spoofed_provider),
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    let error = service
        .verify_metadata_logical_plan(&spoofed_plan, "algeon_datafusion.test.metadata_proof")
        .expect_err("a same-name user provider must not satisfy the metadata proof");
    assert_eq!(error.code(), ErrorCode::InvalidRequest);

    for sql in ["SELECT 1", "SELECT i.table_name FROM information_schema.tables i JOIN user_table u ON true"] {
        let plan = ctx.state().create_logical_plan(sql).await.unwrap();
        let error = service
            .verify_metadata_logical_plan(&plan, "algeon_datafusion.test.metadata_proof")
            .expect_err("non-metadata logical plans must fail proof");
        assert_eq!(error.code(), ErrorCode::InvalidRequest, "{sql}: {error:?}");
    }

    let mixed_gpu_user = ctx
        .state()
        .create_logical_plan(&format!(
            "SELECT g.function_name, u.id \
             FROM {GPU_LIST_FUNCTIONS_FUNCTION}() g \
             JOIN user_table u ON true"
        ))
        .await
        .unwrap();
    let error = service
        .verify_metadata_logical_plan(&mixed_gpu_user, "algeon_datafusion.test.metadata_proof")
        .expect_err("GPU metadata mixed with a user table must fail proof");
    assert_eq!(error.code(), ErrorCode::InvalidRequest);
}

/// Admission, not the capture budget, decides overload: a statement the
/// active capture budget cannot hold runs without capture.
#[tokio::test]
async fn statement_beyond_the_capture_budget_runs_and_reports_the_budget_as_unavailable() {
    let service = super::service::bounded_test_service_with_capture_budget(Duration::from_secs(30), 0);
    let response = service
        .get_flight_info_statement(
            CommandStatementQuery {
                query: "SELECT x FROM (VALUES (1), (2)) AS t(x)".to_string(),
                transaction_id: None,
            },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect("registration does not depend on the capture budget");
    let query_id = query_id_from_flight_info(&response);

    let batches = execute_ticket_batches(&service, query_id.clone()).await;

    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let payload = query_stats_payload(&service, &query_id);
    assert!(payload.contains("# outcome=succeeded"), "{payload}");
    assert!(payload.contains("# unavailable_reason=active_capture_budget"), "{payload}");
}

#[tokio::test]
async fn explain_gpu_success_retains_one_sealed_attempt_report() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let response = service
        .get_flight_info_statement(
            CommandStatementQuery { query: "EXPLAIN GPU SELECT 1 AS one".to_string(), transaction_id: None },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect("EXPLAIN GPU FlightInfo should issue a ticket");
    let query_id = query_id_from_flight_info(&response);

    let batches = execute_ticket_batches(&service, query_id.clone()).await;

    assert_eq!(batches.len(), 1);
    let payload = query_stats_payload(&service, &query_id);
    assert!(payload.contains("# outcome=succeeded"), "{payload}");
    assert_single_sealed_attempt_report(&payload, "succeeded");
}

#[tokio::test]
async fn explain_gpu_planning_failure_retains_one_sealed_attempt_report() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let status = service
        .get_flight_info_statement(
            CommandStatementQuery { query: "EXPLAIN GPU FORMAT TEXT SELECT 1".to_string(), transaction_id: None },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect_err("unsupported EXPLAIN GPU format should fail planning");
    let query_id = status
        .metadata()
        .get(QUERY_ID_METADATA_KEY)
        .and_then(|value| value.to_str().ok())
        .expect("planning failure carries a query id")
        .to_string();

    assert_eq!(status.code(), Code::InvalidArgument);
    let payload = query_stats_payload(&service, &query_id);
    assert!(payload.contains("# outcome=failed"), "{payload}");
    assert_single_sealed_attempt_report(&payload, "failed");
}

#[tokio::test]
async fn explain_gpu_cancel_before_do_get_retains_one_sealed_attempt_report() {
    use prost::Message as _;

    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let response = service
        .get_flight_info_statement(
            CommandStatementQuery { query: "EXPLAIN GPU SELECT 1 AS one".to_string(), transaction_id: None },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect("EXPLAIN GPU FlightInfo should issue a ticket");
    let query_id = query_id_from_flight_info(&response);
    let info = response.into_inner();

    assert!(
        service
            .cancel_from_flight_info(&info.encode_to_vec(), &Owner::Anonymous)
            .expect("cancel request should decode")
    );

    let payload = query_stats_payload(&service, &query_id);
    assert!(payload.contains("# outcome=cancelled"), "{payload}");
    assert!(payload.contains("# delivery_outcome=failed"), "{payload}");
    assert_single_sealed_attempt_report(&payload, "cancelled");
}

#[tokio::test]
async fn explain_gpu_stream_drop_retains_one_sealed_attempt_report() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let response = service
        .get_flight_info_statement(
            CommandStatementQuery { query: "EXPLAIN GPU SELECT 1 AS one".to_string(), transaction_id: None },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect("EXPLAIN GPU FlightInfo should issue a ticket");
    let query_id = query_id_from_flight_info(&response);
    let attempt_cancellation = service
        .coordinator
        .ticket_cancellation(&query_id)
        .expect("EXPLAIN GPU ticket retains its attempt cancellation");
    let response =
        service.execute_ticket(query_id.clone(), None).await.expect("EXPLAIN GPU DoGet should return a lazy stream");

    drop(response);

    assert!(attempt_cancellation.is_cancelled(), "dropping EXPLAIN GPU must cancel its Flight lifecycle");
    assert_eq!(service.cancellations.len(), 0, "dropped EXPLAIN GPU stream must unregister cancellation");
    let payload = query_stats_payload(&service, &query_id);
    assert!(payload.contains("# outcome=cancelled"), "{payload}");
    assert!(payload.contains("# delivery_outcome=dropped"), "{payload}");
    assert!(payload.contains("# attempt_outcome=cancelled"), "{payload}");
    assert_single_sealed_attempt_report(&payload, "cancelled");
}

#[tokio::test]
async fn explain_gpu_post_batch_drop_records_failed_delivery_outcome() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let response = service
        .get_flight_info_statement(
            CommandStatementQuery { query: "EXPLAIN GPU SELECT 1 AS one".to_string(), transaction_id: None },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect("EXPLAIN GPU FlightInfo should issue a ticket");
    let query_id = query_id_from_flight_info(&response);
    let attempt_cancellation = service
        .coordinator
        .ticket_cancellation(&query_id)
        .expect("EXPLAIN GPU ticket retains its attempt cancellation");
    let mut stream = service
        .execute_ticket(query_id.clone(), None)
        .await
        .expect("EXPLAIN GPU DoGet should return a lazy stream")
        .into_inner();

    stream.next().await.expect("schema frame").expect("schema frame is valid");
    stream.next().await.expect("EXPLAIN GPU batch frame").expect("EXPLAIN GPU batch frame is valid");
    assert!(!attempt_cancellation.is_cancelled(), "delivering the EXPLAIN GPU batch keeps the query live until EOF");

    drop(stream);

    assert!(
        attempt_cancellation.is_cancelled(),
        "dropping a partially delivered EXPLAIN GPU stream cancels native work"
    );
    assert_eq!(service.cancellations.len(), 0, "post-batch drop must unregister cancellation");
    let payload = query_stats_payload(&service, &query_id);
    assert!(payload.contains("# outcome=cancelled"), "{payload}");
    assert!(payload.contains("# delivery_outcome=failed"), "{payload}");
    assert_single_sealed_attempt_report(&payload, "cancelled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_ddl_read_only_schema_failure_does_not_classify_upstream_error_prose() {
    let ctx = Arc::new(SessionContext::new_with_config(
        SessionConfig::new().with_default_catalog_and_schema("glue", "algeon_graph"),
    ));
    ctx.register_catalog(
        "glue",
        Arc::new(ReadOnlyCatalog::new("algeon_graph", [("edges_by_src", read_only_edge_table())])),
    );
    let service =
        DatafusionAlgeonFlightSqlService::new(ctx, Arc::new(QueryAdmission::new_for_tests()), Duration::from_secs(30));
    let query_id = "read-only-ddl-target".to_string();
    let sql = "CREATE VIEW target_edges AS SELECT src, dst FROM edges_by_src".to_string();
    let planning_lease = service.register_query(query_id.clone(), sql.clone(), None).unwrap();

    let status = service
        .execute_update_sql(planning_lease, sql, None, UpdateStatementKind::Statement, None)
        .await
        .expect_err("read-only source schema should reject DDL registration");

    assert_eq!(status.code(), Code::Internal);
    assert_status_metadata(&status, "internal", "internal", "permanent");
    assert_eq!(status.message(), MASKED_INTERNAL_ERROR_MESSAGE);
    assert!(!status.message().contains("glue.algeon_graph"), "status={status:?}");
    let payload = String::from_utf8(
        service.coordinator.query_stats(&query_id).expect("failed DDL update should retain query stats").to_vec(),
    )
    .unwrap();
    assert!(payload.contains("operation=algeon_datafusion.server.flight.do_put_statement_update.sql"), "{payload}");
    assert!(payload.contains("fact.datafusion_error=execution"), "{payload}");
    for raw_fact in ["fact.target_catalog=", "fact.target_schema=", "fact.target_table=", "fact.statement_kind="] {
        assert!(!payload.contains(raw_fact), "raw identifier fact must not be retained: {payload}");
    }
}

/// Each Flight query plans in a copy of the shared session, so a `SET` could
/// only change a discarded copy. Every endpoint rejects it with one message,
/// and the server's settings stay as they were.
#[tokio::test]
async fn set_and_reset_are_rejected_on_every_endpoint() {
    const MESSAGE: &str = "SET and RESET are not supported over Flight SQL; configure DataFusion session options with DATAFUSION_* environment variables before starting the server";
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_information_schema(true));
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::new(ctx),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
    );
    let time_zone_setting = || async {
        let response = service
            .get_flight_info_statement(
                CommandStatementQuery {
                    query:
                        "SELECT value FROM information_schema.df_settings WHERE name = 'datafusion.execution.time_zone'"
                            .to_string(),
                    transaction_id: None,
                },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
            .expect("df_settings is readable over the statement endpoint");
        let batches = execute_ticket_batches(&service, query_id_from_flight_info(&response)).await;
        let values = batches[0].column(0).as_any().downcast_ref::<StringArray>().expect("df_settings value is Utf8");
        values.is_valid(0).then(|| values.value(0).to_string())
    };
    let before = time_zone_setting().await;

    for sql in ["SET datafusion.execution.time_zone = 'UTC'", "RESET datafusion.execution.time_zone"] {
        let statement = service
            .get_flight_info_statement(
                CommandStatementQuery { query: sql.to_string(), transaction_id: None },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
            .expect_err("the statement endpoint rejects session statements");
        let prepared = service
            .do_action_create_prepared_statement(
                ActionCreatePreparedStatementRequest { query: sql.to_string(), transaction_id: None },
                Request::new(arrow_flight::Action { r#type: String::new(), body: Default::default() }),
            )
            .await
            .expect_err("the prepared endpoint rejects session statements");
        let query_id = format!("update-{sql}");
        let planning_lease = service.register_query(query_id, sql.to_string(), None).unwrap();
        let update = service
            .execute_update_sql(planning_lease, sql.to_string(), None, UpdateStatementKind::Statement, None)
            .await
            .expect_err("the update endpoint rejects session statements");
        for status in [statement, prepared, update] {
            assert_eq!(status.code(), Code::Unimplemented, "{sql}: {status:?}");
            assert_eq!(status.message(), MESSAGE, "{sql}");
            assert_status_metadata(&status, "unsupported_operator", "unsupported", "permanent");
        }
    }

    assert_eq!(time_zone_setting().await, before);
}

/// SQL from the update and prepared paths reaches INFO logs only after
/// `OPTIONS` values are masked, since those values carry storage credentials.
#[tokio::test]
async fn statement_logs_mask_options_values_on_the_update_and_prepared_paths() {
    const SECRET: &str = "super-secret-value";
    let buffer = SharedLogBuffer::default();
    let writer = buffer.clone();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_target(false)
        .with_writer(move || writer.clone())
        .finish();
    let _logs = set_default(subscriber);
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::new(SessionContext::new()),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
    );
    let sql = |table: &str| {
        format!(
            "CREATE EXTERNAL TABLE {table} STORED AS PARQUET LOCATION '/nonexistent/algeon/{table}/' OPTIONS ('aws.secret_access_key' '{SECRET}')"
        )
    };

    let update_sql = sql("update_secret_rows");
    let planning_lease = service.register_query("update-secret".to_string(), update_sql.clone(), None).unwrap();
    // The location does not exist, so the update fails after its statement log.
    let _ = service.execute_update_sql(planning_lease, update_sql, None, UpdateStatementKind::Statement, None).await;

    let prepared = service
        .do_action_create_prepared_statement(
            ActionCreatePreparedStatementRequest { query: sql("prepared_secret_rows"), transaction_id: None },
            Request::new(arrow_flight::Action { r#type: String::new(), body: Default::default() }),
        )
        .await
        .expect("CREATE EXTERNAL TABLE prepares without reading its location");
    service
        .get_flight_info_prepared_statement(
            CommandPreparedStatementQuery { prepared_statement_handle: prepared.prepared_statement_handle },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect("prepared DDL plans without executing");

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    for message in ["executing Flight SQL update statement", "planning Flight SQL prepared statement"] {
        let line = output.lines().find(|line| line.contains(message)).unwrap_or_else(|| panic!("{message}: {output}"));
        assert!(line.contains("'aws.secret_access_key' '***'"), "{line}");
    }
    assert!(!output.contains(SECRET), "{output}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deduplicated_prepared_update_retains_sealed_attempt_report() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let query_id = "deduplicated-prepared-update".to_string();
    let sql = "CREATE VIEW deduplicated_update_view AS SELECT 1".to_string();
    let planning_lease = service.register_query(query_id.clone(), sql.clone(), None).unwrap();

    let count = service
        .complete_deduplicated_update(planning_lease, sql, None)
        .await
        .expect("deduplicated prepared update should complete through its attempt owner");

    assert_eq!(count, NO_UPDATED_RECORDS_COUNT);
    let payload = String::from_utf8(
        service
            .coordinator
            .query_stats(&query_id)
            .expect("deduplicated update should retain the attempt-owned query report")
            .to_vec(),
    )
    .unwrap();
    assert!(payload.contains("# attempt_id=attempt-"), "{payload}");
    assert_eq!(payload.matches("# attempt_outcome=").count(), 1, "{payload}");
    assert!(payload.contains("# sealed=true"), "{payload}");
}

#[tokio::test]
async fn prepared_update_shutdown_promotion_preserves_operation_and_request_metadata() {
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::new(SessionContext::new()),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
    );
    let query_id = "prepared-update-shutdown-race".to_string();
    let correlation_id = "prepared-update-correlation".to_string();
    let sql = "CREATE VIEW prepared_shutdown_view AS SELECT 1".to_string();
    let planning_lease = service
        .register_query(query_id.clone(), sql.clone(), Some(correlation_id.clone()))
        .expect("prepared update planning lease");
    assert_eq!(service.coordinator.begin_shutdown(Instant::now()).live_count(), 1);

    let status = service
        .execute_update_sql(
            planning_lease,
            sql,
            Some(correlation_id.clone()),
            UpdateStatementKind::PreparedStatement,
            None,
        )
        .await
        .expect_err("shutdown must reject prepared update promotion");

    assert_eq!(status.code(), Code::Cancelled);
    assert_status_metadata(&status, "cancelled", "cancelled", "permanent");
    assert_eq!(
        status.metadata().get(ERROR_OPERATION_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some("algeon_datafusion.server.flight.do_put_prepared_statement_update.into_execution")
    );
    assert_eq!(
        status.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(query_id.as_str())
    );
    assert_eq!(
        status.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(correlation_id.as_str())
    );
    assert_eq!(service.coordinator.terminal(&query_id), Some(QueryTerminalOutcome::Cancelled));
}

fn assert_masked_status_points_at_query_id(status: &Status, query_id: &str) {
    assert_eq!(status.code(), Code::Internal, "{status:?}");
    assert_eq!(status.message(), MASKED_INTERNAL_ERROR_MESSAGE);
    assert_eq!(
        status.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(query_id),
        "the metadata value the message points at must be present: {status:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn masked_internal_failures_carry_the_query_id_on_every_statement_path() {
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::new(SessionContext::new()),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vec!["not a time"]))]).unwrap();
    service.ctx.register_table("caller_rows", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
    let sql = "SELECT to_timestamp(s) FROM caller_rows";

    async fn failed_do_get(service: &DatafusionAlgeonFlightSqlService, query_id: String) -> Status {
        let stream = service.execute_ticket(query_id, None).await.expect("DoGet returns a lazy stream").into_inner();
        let error = FlightRecordBatchStream::new_from_flight_data(stream.map_err(FlightError::from))
            .try_collect::<Vec<_>>()
            .await
            .expect_err("parsing the caller's value must fail the stream");
        let FlightError::Tonic(status) = error else { panic!("DoGet must fail with a gRPC status: {error}") };
        *status
    }

    let statement = issue_statement_ticket(&service, sql).await.expect("a timestamp parse plans");
    let statement_id = query_id_from_flight_info(&statement);
    assert_masked_status_points_at_query_id(&failed_do_get(&service, statement_id.clone()).await, &statement_id);

    let prepared = service
        .do_action_create_prepared_statement(
            ActionCreatePreparedStatementRequest { query: sql.to_string(), transaction_id: None },
            Request::new(arrow_flight::Action::new("CreatePreparedStatement", Vec::<u8>::new())),
        )
        .await
        .expect("a timestamp parse prepares");
    let prepared_info = service
        .get_flight_info_prepared_statement(
            CommandPreparedStatementQuery { prepared_statement_handle: prepared.prepared_statement_handle },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
        .expect("a prepared timestamp parse plans");
    let prepared_id = query_id_from_flight_info(&prepared_info);
    assert_masked_status_points_at_query_id(&failed_do_get(&service, prepared_id.clone()).await, &prepared_id);

    let ctx = Arc::new(SessionContext::new_with_config(
        SessionConfig::new().with_default_catalog_and_schema("glue", "algeon_graph"),
    ));
    ctx.register_catalog(
        "glue",
        Arc::new(ReadOnlyCatalog::new("algeon_graph", [("edges_by_src", read_only_edge_table())])),
    );
    let update_service =
        DatafusionAlgeonFlightSqlService::new(ctx, Arc::new(QueryAdmission::new_for_tests()), Duration::from_secs(30));
    for (kind, query_id) in [
        (UpdateStatementKind::Statement, "masked-update-statement"),
        (UpdateStatementKind::PreparedStatement, "masked-update-prepared"),
    ] {
        let sql = "CREATE VIEW target_edges AS SELECT src, dst FROM edges_by_src".to_string();
        let planning_lease = update_service.register_query(query_id.to_string(), sql.clone(), None).unwrap();
        let status = update_service
            .execute_update_sql(planning_lease, sql, None, kind, None)
            .await
            .expect_err("read-only source schema should reject DDL registration");
        assert_masked_status_points_at_query_id(&status, query_id);
    }
}

#[tokio::test]
async fn invalid_ticket_kind_status_carries_the_query_id_and_correlation_id() {
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::new(SessionContext::new()),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
    );
    let query_id = "invalid-ticket-kind".to_string();
    let planning_lease = service.register_query(query_id.clone(), "SELECT 1".to_string(), None).unwrap();
    let now = Instant::now();
    planning_lease
        .issue_ticket(CoordinatorTicketPayload {
            sql: "SELECT 1".to_string(),
            correlation_id: None,
            prepared_handle: None,
            kind: u8::MAX,
            execution_domain: StatementExecutionDomain::Data,
            attempt: None,
            started_at: now,
            deadline: now + Duration::from_secs(30),
            forest_model_digests: Vec::new(),
        })
        .expect("ticket issues");

    let status = service
        .execute_ticket(query_id.clone(), Some("request-correlation".to_string()))
        .await
        .err()
        .expect("a ticket kind outside the coordinator contract must fail DoGet");

    assert_eq!(status.code(), Code::Internal);
    assert_eq!(
        status.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(query_id.as_str())
    );
    assert_eq!(
        status.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some("request-correlation")
    );
}

async fn issue_statement_ticket(
    service: &DatafusionAlgeonFlightSqlService,
    sql: &str,
) -> Result<Response<FlightInfo>, Status> {
    service
        .get_flight_info_statement(
            CommandStatementQuery { query: sql.to_string(), transaction_id: None },
            Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
        )
        .await
}

#[tokio::test]
async fn issuing_a_new_ticket_sweeps_abandoned_tickets_without_do_get() {
    let service = DatafusionAlgeonFlightSqlService::new(
        Arc::new(SessionContext::new()),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::ZERO,
    );

    issue_statement_ticket(&service, "SELECT 1").await.expect("first ticket should issue");
    let abandoned = only_ticket_query_id(&service);
    issue_statement_ticket(&service, "SELECT 2").await.expect("second ticket should issue");

    let query_ids = service.coordinator.ticket_query_ids();
    assert_eq!(query_ids.len(), 1);
    assert!(!query_ids.contains(&abandoned));
    let status = service.execute_ticket(abandoned.clone(), None).await.err().expect("expired ticket rejects DoGet");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_status_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("expired"));
    assert_eq!(
        status.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(abandoned.as_str())
    );
    assert!(!status.message().contains(&abandoned));
    assert!(matches!(
        service.coordinator.artifact(&abandoned),
        Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::TicketExpired))
    ));
}

fn only_ticket_query_id(service: &DatafusionAlgeonFlightSqlService) -> String {
    let query_ids = service.coordinator.ticket_query_ids();
    assert_eq!(query_ids.len(), 1, "expected exactly one live statement ticket, got {query_ids:?}");
    query_ids[0].clone()
}

fn assert_status_metadata(status: &Status, code: &str, kind: &str, error_status: &str) {
    assert_eq!(status.metadata().get(ERROR_CODE_METADATA_KEY).and_then(|value| value.to_str().ok()), Some(code));
    assert_eq!(status.metadata().get(ERROR_KIND_METADATA_KEY).and_then(|value| value.to_str().ok()), Some(kind));
    assert_eq!(
        status.metadata().get(ERROR_STATUS_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(error_status)
    );
}

#[derive(Debug)]
struct ReadOnlyCatalog {
    schema_name: String,
    schema: Arc<ReadOnlySchema>,
}

impl ReadOnlyCatalog {
    fn new(
        schema_name: impl Into<String>,
        tables: impl IntoIterator<Item = (&'static str, Arc<dyn TableProvider>)>,
    ) -> Self {
        Self { schema_name: schema_name.into(), schema: Arc::new(ReadOnlySchema::new(tables)) }
    }
}

impl CatalogProvider for ReadOnlyCatalog {
    fn schema_names(&self) -> Vec<String> {
        vec![self.schema_name.clone()]
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        (name == self.schema_name).then(|| Arc::clone(&self.schema) as Arc<dyn SchemaProvider>)
    }
}

#[derive(Debug)]
struct ReadOnlySchema {
    tables: BTreeMap<String, Arc<dyn TableProvider>>,
}

impl ReadOnlySchema {
    fn new(tables: impl IntoIterator<Item = (&'static str, Arc<dyn TableProvider>)>) -> Self {
        Self { tables: tables.into_iter().map(|(name, table)| (name.to_string(), table)).collect() }
    }
}

#[async_trait::async_trait]
impl SchemaProvider for ReadOnlySchema {
    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        Ok(self.tables.get(name).cloned())
    }

    async fn table_type(&self, name: &str) -> DataFusionResult<Option<TableType>> {
        Ok(self.tables.get(name).map(|table| table.table_type()))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}

fn read_only_edge_table() -> Arc<dyn TableProvider> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![0, 1])), Arc::new(Int64Array::from(vec![1, 2]))],
    )
    .unwrap();
    Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
}
