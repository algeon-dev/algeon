use std::{
    collections::BTreeMap,
    io::Write,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use arrow::{
    array::{ArrayRef, Int32Builder, ListBuilder, MapBuilder, MapFieldNames, StringBuilder},
    record_batch::RecordBatch,
};
use arrow_flight::decode::FlightRecordBatchStream;
use datafusion::{
    common::TableReference,
    datasource::{MemTable, TableProvider, provider_as_source},
    execution::SessionStateBuilder,
    logical_expr::LogicalPlanBuilder,
    prelude::SessionConfig,
};
use datafusion_nexus::gpu_functions::{
    GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION,
};
use futures::future::BoxFuture;

use super::*;
use crate::{
    coordinator::QueryTerminalOutcome,
    error::{
        ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_OPERATION_METADATA_KEY, ERROR_STATUS_METADATA_KEY,
    },
    table_format::{
        iceberg::NexusIcebergTableProvider,
        scan::{
            TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile,
            TableFormatDataFileFormat, TableFormatPlanner, TableFormatReadCapability, TableFormatScanPlan,
            TableFormatScanRequest,
        },
    },
};

type PlannerFuture<'a> = BoxFuture<'a, crate::adapter_error::Result<TableFormatScanPlan>>;

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

#[derive(Debug)]
struct SleepingPlanner {
    plan: TableFormatScanPlan,
    delay: Duration,
}

impl SleepingPlanner {
    fn new(plan: TableFormatScanPlan, delay: Duration) -> Self {
        Self { plan, delay }
    }
}

impl TableFormatPlanner for SleepingPlanner {
    fn resolve_scan(&self, _request: TableFormatScanRequest) -> PlannerFuture<'_> {
        let plan = self.plan.clone();
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Ok(plan)
        })
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
    let target = Error::new(ErrorCode::Internal, "datafusion_nexus.test.collection_target", "target adapter failure")
        .with_fact("path", "/srv/nexus/target.parquet")
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

    tracing::subscriber::with_default(subscriber, || {
        DatafusionNexusFlightSqlService::warn_get_tables_metadata_error("catalog", "schema", "table", "schema", &error);
    });

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    assert!(output.contains("error_operation=\"datafusion_nexus.test.collection_target\""), "{output}");
    assert!(output.contains("error_cause=\"target root cause\""), "{output}");
    assert!(output.contains("error_source_path=\"/srv/nexus/target.parquet\""), "{output}");
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
    let status = Status::invalid_argument("terminal failure");

    tracing::subscriber::with_default(subscriber, || {
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
    assert!(output.contains("error=\"terminal failure\""), "{output}");
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

    tracing::subscriber::with_default(subscriber, || {
        log_statement_status("unknown-query", None, "ticket", Instant::now(), &status);
    });

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    assert!(output.contains("query_id=\"unknown-query\""), "{output}");
    assert!(output.contains("outcome=\"failed\""), "{output}");
    assert!(output.contains("stage=\"ticket\""), "{output}");
    assert!(output.contains("grpc_code=InvalidArgument"), "{output}");
    assert!(output.contains("elapsed_ms="), "{output}");
    assert!(output.contains("error=\"unknown ticket\""), "{output}");
    assert!(!output.contains("sql="), "unresolved tickets must not fabricate SQL context: {output}");
}

#[test]
fn execution_terminal_log_includes_resolved_sql() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let sql = "SELECT 1\nFROM user_rows";
    let planning = service
        .coordinator
        .register_with_id("terminal-execution-query", sql, None, None, service.runtime_observation_config)
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

    tracing::subscriber::with_default(subscriber, || {
        terminal.finish_captured("stream", &status, CapturedError::from_status(&status));
    });

    let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
    assert!(output.contains("query_id=\"terminal-execution-query\""), "{output}");
    assert!(output.contains("sql=SELECT 1 FROM user_rows"), "{output}");
    assert!(output.contains("stage=\"stream\""), "{output}");
    assert!(output.contains("grpc_code=Internal"), "{output}");
}

fn query_stats_payload(service: &DatafusionNexusFlightSqlService, query_id: &str) -> String {
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
    let validate_call = r#"{"schema_version":1,"relations":{"input":{"table":"input"}},"options":{}}"#;
    let metadata_queries = [
        "SHOW TABLES",
        "SHOW COLUMNS FROM t",
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
fn metadata_planning_removes_only_the_three_nexus_physical_rules() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let base_rules =
        service.ctx.state().physical_optimizers().iter().map(|rule| rule.name().to_string()).collect::<Vec<_>>();
    let metadata_rules = service
        .metadata_query_planning()
        .ctx
        .state()
        .physical_optimizers()
        .iter()
        .map(|rule| rule.name().to_string())
        .collect::<Vec<_>>();
    let removed = [
        NEXUS_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME,
        NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME,
    ];

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
        .build();
    let ctx = Arc::new(SessionContext::new_with_state(state));
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    ctx.register_table("user_table", Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap())).unwrap();
    let service = DatafusionNexusFlightSqlService::new(
        Arc::clone(&ctx),
        backend.optimizer_config().clone(),
        Arc::new(QueryAdmission::from_backend(&backend)),
        Duration::from_secs(30),
        #[cfg(feature = "cugraph")]
        None,
    );

    let validate_call = r#"{"schema_version":1,"relations":{"input":{"table":"input"}},"options":{}}"#;
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
            .verify_metadata_logical_plan(&plan, "datafusion_nexus.test.metadata_proof")
            .expect("real built-in GPU metadata providers must satisfy the metadata proof");
    }

    let information_schema =
        ctx.state().create_logical_plan("SELECT table_name FROM information_schema.tables").await.unwrap();
    service.verify_metadata_logical_plan(&information_schema, "datafusion_nexus.test.metadata_proof").unwrap();
    let describe = ctx.state().create_logical_plan("DESCRIBE user_table").await.unwrap();
    service.verify_metadata_logical_plan(&describe, "datafusion_nexus.test.metadata_proof").unwrap();

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
        .verify_metadata_logical_plan(&spoofed_plan, "datafusion_nexus.test.metadata_proof")
        .expect_err("a same-name user provider must not satisfy the metadata proof");
    assert_eq!(error.code(), ErrorCode::InvalidRequest);

    for sql in ["SELECT 1", "SELECT i.table_name FROM information_schema.tables i JOIN user_table u ON true"] {
        let plan = ctx.state().create_logical_plan(sql).await.unwrap();
        let error = service
            .verify_metadata_logical_plan(&plan, "datafusion_nexus.test.metadata_proof")
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
        .verify_metadata_logical_plan(&mixed_gpu_user, "datafusion_nexus.test.metadata_proof")
        .expect_err("GPU metadata mixed with a user table must fail proof");
    assert_eq!(error.code(), ErrorCode::InvalidRequest);
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

    assert!(service.cancel_from_flight_info(&info.encode_to_vec()).expect("cancel request should decode"));

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
    let native_cancellation =
        service.coordinator.ticket_cancellation(&query_id).expect("EXPLAIN GPU ticket retains native cancellation");
    let response =
        service.execute_ticket(query_id.clone(), None).await.expect("EXPLAIN GPU DoGet should return a lazy stream");

    drop(response);

    assert!(native_cancellation.is_cancelled(), "dropping EXPLAIN GPU must cancel its Flight lifecycle");
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
    let native_cancellation =
        service.coordinator.ticket_cancellation(&query_id).expect("EXPLAIN GPU ticket retains native cancellation");
    let mut stream = service
        .execute_ticket(query_id.clone(), None)
        .await
        .expect("EXPLAIN GPU DoGet should return a lazy stream")
        .into_inner();

    stream.next().await.expect("schema frame").expect("schema frame is valid");
    stream.next().await.expect("EXPLAIN GPU batch frame").expect("EXPLAIN GPU batch frame is valid");
    assert!(!native_cancellation.is_cancelled(), "delivering the EXPLAIN GPU batch keeps the query live until EOF");

    drop(stream);

    assert!(
        native_cancellation.is_cancelled(),
        "dropping a partially delivered EXPLAIN GPU stream cancels native work"
    );
    assert_eq!(service.cancellations.len(), 0, "post-batch drop must unregister cancellation");
    let payload = query_stats_payload(&service, &query_id);
    assert!(payload.contains("# outcome=cancelled"), "{payload}");
    assert!(payload.contains("# delivery_outcome=failed"), "{payload}");
    assert_single_sealed_attempt_report(&payload, "cancelled");
}

#[derive(Debug)]
struct CountingPlanner {
    plan: TableFormatScanPlan,
    requests: AtomicUsize,
}

impl CountingPlanner {
    fn new(plan: TableFormatScanPlan) -> Self {
        Self { plan, requests: AtomicUsize::new(0) }
    }

    fn request_count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

impl TableFormatPlanner for CountingPlanner {
    fn resolve_scan(&self, _request: TableFormatScanRequest) -> PlannerFuture<'_> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let plan = self.plan.clone();
        Box::pin(async move { Ok(plan) })
    }
}

fn test_iceberg_provider<P>(schema: Arc<Schema>, planner: Arc<P>) -> NexusIcebergTableProvider
where P: TableFormatPlanner + Send + Sync + 'static {
    test_iceberg_provider_with_batches(schema, vec![], planner)
}

fn test_iceberg_provider_with_batches<P>(
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    planner: Arc<P>,
) -> NexusIcebergTableProvider
where
    P: TableFormatPlanner + Send + Sync + 'static,
{
    let cpu_provider: Arc<dyn TableProvider> =
        Arc::new(MemTable::try_new(schema, vec![batches]).expect("test CPU provider must build"));
    NexusIcebergTableProvider::from_cpu_provider_and_native_planner(cpu_provider, None, planner)
}

#[tokio::test]
async fn table_format_distinct_list_issues_ticket_and_executes_datafusion_delegate() {
    let schema = list_string_schema();
    let planner = Arc::new(CountingPlanner::new(list_string_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(
        test_iceberg_provider_with_batches(
            Arc::clone(&schema),
            vec![list_string_batch(Arc::clone(&schema))],
            planner.clone(),
        )
        .with_catalog_type("local"),
    );
    let ctx = Arc::new(SessionContext::new());
    ctx.register_table("iceberg_t", provider).expect("test registers iceberg table");
    let service = DatafusionNexusFlightSqlService::new(
        ctx,
        NexusNativeOptimizerConfig::default(),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
        #[cfg(feature = "cugraph")]
        None,
    );

    issue_statement_ticket(&service, "SELECT DISTINCT \"pCodes\" FROM iceberg_t")
        .await
        .expect("logical planning should issue a ticket for DataFusion-executable List semantics");
    assert_eq!(planner.request_count(), 0, "scan planning belongs to the execution-ticket path");
    let query_id = only_ticket_query_id(&service);
    let batches = execute_ticket_batches(&service, query_id.clone()).await;

    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    assert_eq!(batches[0].schema().field(0).data_type(), &list_string_type());
    assert_eq!(planner.request_count(), 1, "physical planning must resolve the Iceberg scan once");
    assert_datafusion_baseline_evidence(&service, &query_id, None);
}

#[tokio::test]
async fn table_format_map_preserves_datafusion_baseline() {
    let iceberg_schema = map_schema();
    let planner = Arc::new(CountingPlanner::new(map_scan_plan(Arc::clone(&iceberg_schema))));
    let provider = Arc::new(
        test_iceberg_provider_with_batches(
            Arc::clone(&iceberg_schema),
            vec![map_batch(Arc::clone(&iceberg_schema))],
            planner.clone(),
        )
        .with_catalog_type("local"),
    );
    let ctx = Arc::new(SessionContext::new());
    ctx.register_table("iceberg_t", provider).expect("test registers table once");

    let memory_schema = Arc::new(Schema::new(vec![Field::new("mem_id", DataType::Int64, false)]));
    let memory_batch = RecordBatch::try_new(
        Arc::clone(&memory_schema),
        vec![Arc::new(arrow::array::Int64Array::from(vec![1_i64])) as ArrayRef],
    )
    .unwrap();
    let memory_table = Arc::new(MemTable::try_new(memory_schema, vec![vec![memory_batch]]).unwrap());
    ctx.register_table("mem_t", memory_table).expect("test registers memory table");

    let service = DatafusionNexusFlightSqlService::new(
        ctx,
        NexusNativeOptimizerConfig::default(),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
        #[cfg(feature = "cugraph")]
        None,
    );

    issue_statement_ticket(&service, "SELECT i.\"countDetail\" FROM iceberg_t AS i CROSS JOIN mem_t AS m")
        .await
        .expect("logical planning should issue a ticket for DataFusion-executable Map semantics");
    assert_eq!(planner.request_count(), 0, "scan planning belongs to the execution-ticket path");
    let query_id = only_ticket_query_id(&service);
    let batches = execute_ticket_batches(&service, query_id.clone()).await;

    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert_eq!(batches[0].schema().field(0).data_type(), &map_type());
    assert_eq!(planner.request_count(), 1, "physical planning must resolve the Iceberg scan once");
    assert_datafusion_baseline_evidence(&service, &query_id, Some("native_outcome=not_supported"));
}

async fn execute_ticket_batches(service: &DatafusionNexusFlightSqlService, query_id: String) -> Vec<RecordBatch> {
    let response = service
        .execute_ticket(query_id, None)
        .await
        .expect("physical planning and DataFusion delegate execution should succeed");
    FlightRecordBatchStream::new_from_flight_data(response.into_inner().map_err(arrow_flight::error::FlightError::from))
        .try_collect::<Vec<_>>()
        .await
        .expect("Flight result should decode into Arrow batches")
}

fn assert_datafusion_baseline_evidence(
    service: &DatafusionNexusFlightSqlService,
    query_id: &str,
    physical_plan_fragment: Option<&str>,
) {
    let payload = String::from_utf8(
        service.coordinator.query_stats(query_id).expect("query stats capture should be retained").to_vec(),
    )
    .expect("query stats are UTF-8");
    assert!(payload.contains("# outcome=succeeded"), "{payload}");
    assert!(
        payload.contains("NexusIcebergScanExec"),
        "the DataFusion baseline must retain the executable wrapper: {payload}"
    );
    if let Some(fragment) = physical_plan_fragment {
        assert!(payload.contains(fragment), "missing physical-plan evidence `{fragment}`: {payload}");
    }

    let planning = payload
        .split_once("\n[planning]\n")
        .map(|(_, planning)| planning)
        .expect("query stats should expose native planning evidence");
    let mut lines = planning.lines();
    let header = lines.next().expect("planning TSV header").split('\t').collect::<Vec<_>>();
    let outcome = header.iter().position(|column| *column == "outcome").expect("outcome column");
    let reason = header.iter().position(|column| *column == "reason_code").expect("reason_code column");
    let execution_path = header
        .iter()
        .position(|column| *column == "post_native_execution_path")
        .expect("post_native_execution_path column");
    let rejected = header
        .iter()
        .position(|column| *column == "rejected_before_execution")
        .expect("rejected_before_execution column");
    let has_datafusion_baseline_outcome = lines.take_while(|line| !line.is_empty()).any(|line| {
        let fields = line.split('\t').collect::<Vec<_>>();
        fields.get(outcome) == Some(&"not_supported")
            && fields.get(reason) == Some(&"unsupported_type")
            && fields.get(execution_path) == Some(&"datafusion")
            && fields.get(rejected) == Some(&"false")
    });
    assert!(
        has_datafusion_baseline_outcome,
        "missing structured unsupported-type DataFusion-baseline evidence: {payload}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn planning_timeout_consumes_statement_ticket_without_holding_admission() {
    let schema = iceberg_test_schema();
    let planner =
        Arc::new(SleepingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema)), Duration::from_millis(50)));
    let provider = Arc::new(
        test_iceberg_provider(Arc::clone(&schema), planner)
            .with_catalog_type("local")
            .with_scan_planning_timeout(Some(Duration::from_millis(1))),
    );
    let ctx = Arc::new(SessionContext::new());
    ctx.register_table("iceberg_t", provider).expect("test registers table once");

    let service = DatafusionNexusFlightSqlService::new(
        ctx,
        NexusNativeOptimizerConfig::default(),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
        #[cfg(feature = "cugraph")]
        None,
    );

    issue_statement_ticket(&service, "SELECT COUNT(*) AS c FROM iceberg_t")
        .await
        .expect("logical planning should issue a ticket before table scan planning");
    let query_id = only_ticket_query_id(&service);
    let status = match service.execute_ticket(query_id, None).await {
        Ok(_) => panic!("physical scan planning should time out"),
        Err(status) => status,
    };
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_status_metadata(&status, "planning_timeout", "planning", "permanent");
    assert!(service.coordinator.ticket_query_ids().is_empty(), "timed-out statement ticket should be consumed");

    tokio::time::timeout(Duration::from_secs(1), issue_statement_ticket(&service, "SELECT 1 AS one"))
        .await
        .expect("second statement should not wait on the timed-out query")
        .expect("second statement should issue a ticket");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_ddl_read_only_schema_failure_does_not_classify_upstream_error_prose() {
    let ctx = Arc::new(SessionContext::new_with_config(
        SessionConfig::new().with_default_catalog_and_schema("glue", "nexus_graph"),
    ));
    ctx.register_catalog(
        "glue",
        Arc::new(ReadOnlyCatalog::new("nexus_graph", [("edges_by_src", read_only_edge_table())])),
    );
    let service = DatafusionNexusFlightSqlService::new(
        ctx,
        NexusNativeOptimizerConfig::default(),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
        #[cfg(feature = "cugraph")]
        None,
    );
    let query_id = "read-only-ddl-target".to_string();
    let sql = "CREATE VIEW target_edges AS SELECT src, dst FROM edges_by_src".to_string();
    let planning_lease = service.register_query(query_id.clone(), sql.clone(), None).unwrap();

    let status = service
        .execute_update_sql(planning_lease, sql, None, UpdateStatementKind::Statement)
        .await
        .expect_err("read-only source schema should reject DDL registration");

    assert_eq!(status.code(), Code::Internal);
    assert_status_metadata(&status, "internal", "internal", "permanent");
    assert_eq!(status.message(), "internal server error while processing the request");
    assert!(!status.message().contains("glue.nexus_graph"), "status={status:?}");
    let payload = String::from_utf8(
        service.coordinator.query_stats(&query_id).expect("failed DDL update should retain query stats").to_vec(),
    )
    .unwrap();
    assert!(payload.contains("operation=datafusion_nexus.server.flight.do_put_statement_update.sql"), "{payload}");
    assert!(payload.contains("fact.datafusion_error=execution"), "{payload}");
    for raw_fact in [
        "fact.target_catalog=",
        "fact.target_schema=",
        "fact.target_table=",
        "fact.workspace_catalog=",
        "fact.workspace_schema=",
        "fact.statement_kind=",
        "fact.iceberg_enabled=",
    ] {
        assert!(!payload.contains(raw_fact), "raw identifier fact must not be retained: {payload}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_set_datafusion_nexus_key_is_rejected_loudly() {
    let ctx = Arc::new(SessionContext::new());
    let service = DatafusionNexusFlightSqlService::new(
        ctx,
        NexusNativeOptimizerConfig::default(),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
        #[cfg(feature = "cugraph")]
        None,
    );
    let query_id = "reject-nexus-set".to_string();
    let sql = "SET datafusion_nexus.request.requested_grant_bytes = '1048576'".to_string();
    let planning_lease = service.register_query(query_id.clone(), sql.clone(), None).unwrap();

    let status = service
        .execute_update_sql(planning_lease, sql, None, UpdateStatementKind::Statement)
        .await
        .expect_err("Flight update SET for datafusion-nexus keys should be rejected");

    assert_eq!(status.code(), Code::InvalidArgument);
    assert_status_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("server configuration or admission defaults"), "status={status:?}");
    let payload = String::from_utf8(
        service.coordinator.query_stats(&query_id).expect("failed SET update should retain query stats").to_vec(),
    )
    .unwrap();
    assert!(payload.contains("fact.reason=flight_session_scoped_datafusion_nexus_keys_unsupported"), "{payload}");
    assert!(payload.contains("fact.configuration_scope=server_or_admission_defaults"), "{payload}");
    assert!(!payload.contains("fact.config_key="), "raw config key must not be retained: {payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_set_plain_datafusion_key_still_succeeds() {
    let service = super::service::bounded_test_service(Duration::from_secs(30));
    let query_id = "plain-datafusion-set".to_string();
    let sql = "SET datafusion.execution.batch_size = '1024'".to_string();
    let planning_lease = service.register_query(query_id.clone(), sql.clone(), None).unwrap();

    let count = service
        .execute_update_sql(planning_lease, sql, None, UpdateStatementKind::Statement)
        .await
        .expect("plain DataFusion SET should keep existing update-lane behavior");

    assert_eq!(count, NO_UPDATED_RECORDS_COUNT);
    let payload = String::from_utf8(
        service
            .coordinator
            .query_stats(&query_id)
            .expect("bounded DoPut update should retain the attempt-owned query report")
            .to_vec(),
    )
    .unwrap();
    assert!(payload.contains("# attempt_id=attempt-"), "{payload}");
    assert_eq!(payload.matches("# attempt_outcome=").count(), 1, "{payload}");
    assert!(payload.contains("# sealed=true"), "{payload}");
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
    let service = DatafusionNexusFlightSqlService::new(
        Arc::new(SessionContext::new()),
        NexusNativeOptimizerConfig::default(),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::from_secs(30),
        #[cfg(feature = "cugraph")]
        None,
    );
    let query_id = "prepared-update-shutdown-race".to_string();
    let correlation_id = "prepared-update-correlation".to_string();
    let sql = "CREATE VIEW prepared_shutdown_view AS SELECT 1".to_string();
    let planning_lease = service
        .register_query(query_id.clone(), sql.clone(), Some(correlation_id.clone()))
        .expect("prepared update planning lease");
    assert_eq!(service.coordinator.begin_shutdown(Instant::now()).live_count(), 1);

    let status = service
        .execute_update_sql(planning_lease, sql, Some(correlation_id.clone()), UpdateStatementKind::PreparedStatement)
        .await
        .expect_err("shutdown must reject prepared update promotion");

    assert_eq!(status.code(), Code::Cancelled);
    assert_status_metadata(&status, "cancelled", "cancelled", "permanent");
    assert_eq!(
        status.metadata().get(ERROR_OPERATION_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some("datafusion_nexus.server.flight.do_put_prepared_statement_update.into_execution")
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

async fn issue_statement_ticket(
    service: &DatafusionNexusFlightSqlService,
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
    let service = DatafusionNexusFlightSqlService::new(
        Arc::new(SessionContext::new()),
        NexusNativeOptimizerConfig::default(),
        Arc::new(QueryAdmission::new_for_tests()),
        Duration::ZERO,
        #[cfg(feature = "cugraph")]
        None,
    );

    issue_statement_ticket(&service, "SELECT 1").await.expect("first ticket should issue");
    let abandoned = only_ticket_query_id(&service);
    issue_statement_ticket(&service, "SELECT 2").await.expect("second ticket should issue");

    let query_ids = service.coordinator.ticket_query_ids();
    assert_eq!(query_ids.len(), 1);
    assert!(!query_ids.contains(&abandoned));
    assert!(matches!(
        service.coordinator.artifact(&abandoned),
        Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::TicketExpired))
    ));
}

fn only_ticket_query_id(service: &DatafusionNexusFlightSqlService) -> String {
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
        tables: impl IntoIterator<Item = (&'static str, Arc<dyn datafusion::datasource::TableProvider>)>,
    ) -> Self {
        Self { schema_name: schema_name.into(), schema: Arc::new(ReadOnlySchema::new(tables)) }
    }
}

impl datafusion::catalog::CatalogProvider for ReadOnlyCatalog {
    fn schema_names(&self) -> Vec<String> {
        vec![self.schema_name.clone()]
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn datafusion::catalog::SchemaProvider>> {
        (name == self.schema_name).then(|| Arc::clone(&self.schema) as Arc<dyn datafusion::catalog::SchemaProvider>)
    }
}

#[derive(Debug)]
struct ReadOnlySchema {
    tables: BTreeMap<String, Arc<dyn datafusion::datasource::TableProvider>>,
}

impl ReadOnlySchema {
    fn new(tables: impl IntoIterator<Item = (&'static str, Arc<dyn datafusion::datasource::TableProvider>)>) -> Self {
        Self { tables: tables.into_iter().map(|(name, table)| (name.to_string(), table)).collect() }
    }
}

#[async_trait::async_trait]
impl datafusion::catalog::SchemaProvider for ReadOnlySchema {
    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    async fn table(
        &self,
        name: &str,
    ) -> datafusion::common::Result<Option<Arc<dyn datafusion::datasource::TableProvider>>> {
        Ok(self.tables.get(name).cloned())
    }

    async fn table_type(&self, name: &str) -> datafusion::common::Result<Option<TableType>> {
        Ok(self.tables.get(name).map(|table| table.table_type()))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}

fn read_only_edge_table() -> Arc<dyn datafusion::datasource::TableProvider> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = arrow::record_batch::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow::array::Int64Array::from(vec![0, 1])),
            Arc::new(arrow::array::Int64Array::from(vec![1, 2])),
        ],
    )
    .unwrap();
    Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
}

fn iceberg_test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
}

fn native_admissible_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(11),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
}

fn list_string_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("pCodes", list_string_type(), true)]))
}

fn list_string_type() -> DataType {
    DataType::List(Arc::new(Field::new("element", DataType::Utf8, true)))
}

fn list_string_batch(schema: Arc<Schema>) -> RecordBatch {
    let mut values =
        ListBuilder::new(StringBuilder::new()).with_field(Arc::new(Field::new("element", DataType::Utf8, true)));
    values.append_value([Some("north"), Some("east")]);
    values.append_value([Some("north"), Some("east")]);
    values.append_value([Some("south")]);
    RecordBatch::try_new(schema, vec![Arc::new(values.finish()) as ArrayRef]).unwrap()
}

fn list_string_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    let data_type = list_string_type();
    single_column_scan_plan(schema, "pCodes", data_type)
}

fn map_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("countDetail", map_type(), true)]))
}

fn map_type() -> DataType {
    DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(arrow::datatypes::Fields::from(vec![
                Arc::new(Field::new("key", DataType::Utf8, false)),
                Arc::new(Field::new("value", DataType::Int32, true)),
            ])),
            false,
        )),
        false,
    )
}

fn map_batch(schema: Arc<Schema>) -> RecordBatch {
    let mut values = MapBuilder::new(
        Some(MapFieldNames { entry: "entries".to_string(), key: "key".to_string(), value: "value".to_string() }),
        StringBuilder::new(),
        Int32Builder::new(),
    );
    for (key, value) in [("a", 1), ("a", 1), ("b", 2)] {
        values.keys().append_value(key);
        values.values().append_value(value);
        values.append(true).unwrap();
    }
    RecordBatch::try_new(schema, vec![Arc::new(values.finish()) as ArrayRef]).unwrap()
}

fn map_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    single_column_scan_plan(schema, "countDetail", map_type())
}

fn single_column_scan_plan(schema: Arc<Schema>, column_name: &'static str, data_type: DataType) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(11),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            column_name,
            Some(column_name),
            data_type,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
}
