use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use datafusion::{datasource::MemTable, prelude::SessionConfig};
use futures::future::BoxFuture;

use super::*;
use crate::{
    policy::GpuFallbackPolicy,
    server::{
        auth::{AuthMode, TokenStore},
        coordinator::QueryTerminalOutcome,
        error::{
            ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_OPERATION_METADATA_KEY, ERROR_STATUS_METADATA_KEY,
        },
    },
    table_format::{
        iceberg::CudfIcebergTableProvider,
        scan::{
            TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile,
            TableFormatDataFileFormat, TableFormatPlanner, TableFormatReadCapability, TableFormatScanPlan,
            TableFormatScanRequest,
        },
    },
};

type PlannerFuture<'a> = BoxFuture<'a, crate::error::Result<TableFormatScanPlan>>;

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
    fn plan_scan(&self, _request: TableFormatScanRequest) -> PlannerFuture<'_> {
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
    fn plan_scan(&self, _request: TableFormatScanRequest) -> PlannerFuture<'_> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let plan = self.plan.clone();
        Box::pin(async move { Ok(plan) })
    }
}

#[tokio::test]
async fn unsupported_table_format_distinct_list_fast_fails_before_scan_planning() {
    let schema = list_string_schema();
    let planner = Arc::new(CountingPlanner::new(list_string_scan_plan(Arc::clone(&schema))));
    assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
        Arc::clone(&schema),
        planner,
        "SELECT DISTINCT \"pCodes\" FROM iceberg_t",
        "distinct",
        "pCodes",
        "List(",
    )
    .await;
}

#[tokio::test]
async fn unsupported_table_format_group_by_list_fast_fails_before_scan_planning() {
    let schema = list_string_schema();
    let planner = Arc::new(CountingPlanner::new(list_string_scan_plan(Arc::clone(&schema))));
    assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
        Arc::clone(&schema),
        planner,
        "SELECT \"pCodes\", COUNT(*) AS n FROM iceberg_t GROUP BY \"pCodes\"",
        "group_by",
        "pCodes",
        "List(",
    )
    .await;
}

#[tokio::test]
async fn unsupported_table_format_count_distinct_list_fast_fails_before_scan_planning() {
    let schema = list_string_schema();
    let planner = Arc::new(CountingPlanner::new(list_string_scan_plan(Arc::clone(&schema))));
    assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
        Arc::clone(&schema),
        planner,
        "SELECT COUNT(DISTINCT \"pCodes\") AS n FROM iceberg_t",
        "aggregate_distinct",
        "pCodes",
        "List(",
    )
    .await;
}

#[tokio::test]
async fn unsupported_table_format_distinct_map_fast_fails_before_scan_planning() {
    let schema = map_schema();
    let planner = Arc::new(CountingPlanner::new(map_scan_plan(Arc::clone(&schema))));
    assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
        Arc::clone(&schema),
        planner,
        "SELECT DISTINCT \"countDetail\" FROM iceberg_t",
        "distinct",
        "countDetail",
        "Map(",
    )
    .await;
}

#[tokio::test]
async fn unsupported_non_iceberg_operator_column_in_mixed_plan_does_not_fast_fail() {
    let iceberg_schema = iceberg_test_schema();
    let planner = Arc::new(CountingPlanner::new(native_admissible_scan_plan(Arc::clone(&iceberg_schema))));
    let provider = Arc::new(
        CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&iceberg_schema), planner.clone())
            .with_catalog_type("local"),
    );
    let ctx = SessionContext::new();
    ctx.register_table("iceberg_t", provider).expect("test registers iceberg table");

    let memory_schema = Arc::new(Schema::new(vec![Field::new("memCodes", list_string_type(), true)]));
    let memory_table = Arc::new(MemTable::try_new(Arc::clone(&memory_schema), vec![vec![]]).unwrap());
    ctx.register_table("mem_t", memory_table).expect("test registers memory table");

    let df = ctx
        .sql("SELECT DISTINCT m.\"memCodes\" FROM mem_t AS m CROSS JOIN iceberg_t AS i")
        .await
        .expect("logical plan builds");

    assert!(fast_fail_unsupported_table_format_operator_plan(df.logical_plan()).is_none());
    assert_eq!(planner.request_count(), 0, "logical fast-fail proof must not trigger Iceberg scan planning");
}

#[tokio::test]
async fn unsupported_iceberg_operator_column_in_mixed_plan_fast_fails() {
    let iceberg_schema = map_schema();
    let planner = Arc::new(CountingPlanner::new(map_scan_plan(Arc::clone(&iceberg_schema))));
    let provider = Arc::new(
        CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&iceberg_schema), planner.clone())
            .with_catalog_type("local"),
    );
    let ctx = Arc::new(SessionContext::new());
    ctx.register_table("iceberg_t", provider).expect("test registers iceberg table");

    let memory_schema = Arc::new(Schema::new(vec![Field::new("mem_id", DataType::Int64, false)]));
    let memory_table = Arc::new(MemTable::try_new(Arc::clone(&memory_schema), vec![vec![]]).unwrap());
    ctx.register_table("mem_t", memory_table).expect("test registers memory table");

    let service = DatafusionCudfFlightSqlService::new(
        ctx,
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
        ServerSessionProfile::Native,
        Arc::new(QueryAdmission::new(1).expect("test admission limit")),
        Duration::from_secs(30),
        AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
        #[cfg(feature = "cugraph")]
        None,
    );

    let status =
        issue_statement_ticket(&service, "SELECT DISTINCT i.\"countDetail\" FROM iceberg_t AS i CROSS JOIN mem_t AS m")
            .await
            .expect_err("unsupported Iceberg operator column should fast-fail through mixed joins");

    assert_eq!(status.code(), Code::InvalidArgument);
    assert_status_metadata(&status, "native_candidate_rejected", "planning", "permanent");
    assert!(status.message().contains("operator=distinct"), "status={status:?}");
    assert!(status.message().contains("column=countDetail"), "status={status:?}");
    assert!(status.message().contains("type=Map("), "status={status:?}");
    assert_eq!(planner.request_count(), 0, "Iceberg scan planning must be skipped");
    assert!(
        service.coordinator.ticket_query_ids().is_empty(),
        "fast-failed statements should not issue execution tickets"
    );
}

async fn assert_unsupported_table_format_operator_fast_fails_before_scan_planning(
    schema: Arc<Schema>,
    planner: Arc<CountingPlanner>,
    sql: &str,
    operator: &str,
    column: &str,
    type_prefix: &str,
) {
    let provider = Arc::new(
        CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone())
            .with_catalog_type("local"),
    );
    let ctx = Arc::new(SessionContext::new());
    ctx.register_table("iceberg_t", provider).expect("test registers table once");

    let service = DatafusionCudfFlightSqlService::new(
        ctx,
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
        ServerSessionProfile::Native,
        Arc::new(QueryAdmission::new(1).expect("test admission limit")),
        Duration::from_secs(30),
        AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
        #[cfg(feature = "cugraph")]
        None,
    );

    let status = issue_statement_ticket(&service, sql)
        .await
        .expect_err("unsupported table-format operator should fail before table scan planning");

    assert_eq!(status.code(), Code::InvalidArgument);
    assert_status_metadata(&status, "native_candidate_rejected", "planning", "permanent");
    assert!(status.message().contains(&format!("operator={operator}")), "status={status:?}");
    assert!(status.message().contains(&format!("column={column}")), "status={status:?}");
    assert!(status.message().contains(&format!("type={type_prefix}")), "status={status:?}");
    assert_eq!(planner.request_count(), 0, "Iceberg scan planning must be skipped");
    assert!(
        service.coordinator.ticket_query_ids().is_empty(),
        "fast-failed statements should not issue execution tickets"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn planning_timeout_consumes_statement_ticket_without_holding_admission() {
    let schema = iceberg_test_schema();
    let planner =
        Arc::new(SleepingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema)), Duration::from_millis(50)));
    let provider = Arc::new(
        CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner)
            .with_catalog_type("local")
            .with_scan_planning_timeout(Some(Duration::from_millis(1))),
    );
    let ctx = Arc::new(SessionContext::new());
    ctx.register_table("iceberg_t", provider).expect("test registers table once");

    let service = DatafusionCudfFlightSqlService::new(
        ctx,
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
        ServerSessionProfile::Native,
        Arc::new(QueryAdmission::new(1).expect("test admission limit")),
        Duration::from_secs(30),
        AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
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
async fn update_ddl_read_only_schema_failure_captures_catalog_target_facts() {
    let ctx = Arc::new(SessionContext::new_with_config(
        SessionConfig::new().with_default_catalog_and_schema("glue", "nexus_graph"),
    ));
    ctx.register_catalog(
        "glue",
        Arc::new(ReadOnlyCatalog::new("nexus_graph", [("edges_by_src", read_only_edge_table())])),
    );
    let service = DatafusionCudfFlightSqlService::new(
        ctx,
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
        ServerSessionProfile::Native,
        Arc::new(QueryAdmission::new(1).expect("test admission limit")),
        Duration::from_secs(30),
        AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
        #[cfg(feature = "cugraph")]
        None,
    )
    .with_catalog_diagnostics(ServerCatalogDiagnostics {
        iceberg_enabled: true,
        workspace_catalog: Some("datafusion".to_string()),
        workspace_schema: Some("public".to_string()),
    });
    let query_id = "read-only-ddl-target".to_string();
    let sql = "CREATE VIEW target_edges AS SELECT src, dst FROM edges_by_src".to_string();
    let planning_lease = service.register_query(query_id.clone(), sql.clone(), None).unwrap();

    let status = service
        .execute_update_sql(planning_lease, sql, None, UpdateStatementKind::Statement)
        .await
        .expect_err("read-only source schema should reject DDL registration");

    assert_eq!(status.code(), Code::InvalidArgument);
    assert_status_metadata(&status, "datafusion_planning", "planning", "permanent");
    assert!(status.message().contains("glue.nexus_graph"), "status={status:?}");
    let payload = String::from_utf8(
        service.coordinator.query_stats(&query_id).expect("failed DDL update should retain query stats").to_vec(),
    )
    .unwrap();
    assert!(payload.contains("operation=datafusion_nexus.server.flight.do_put_statement_update.sql"), "{payload}");
    assert!(payload.contains("fact.statement_kind=CreateView"), "{payload}");
    assert!(payload.contains("fact.iceberg_enabled=true"), "{payload}");
    for raw_fact in [
        "fact.target_catalog=",
        "fact.target_schema=",
        "fact.target_table=",
        "fact.workspace_catalog=",
        "fact.workspace_schema=",
    ] {
        assert!(!payload.contains(raw_fact), "raw identifier fact must not be retained: {payload}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_set_datafusion_nexus_key_is_rejected_loudly() {
    let ctx = Arc::new(SessionContext::new());
    let service = DatafusionCudfFlightSqlService::new(
        ctx,
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
        ServerSessionProfile::Native,
        Arc::new(QueryAdmission::new(1).expect("test admission limit")),
        Duration::from_secs(30),
        AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
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
    assert!(status.message().contains("server or tenant defaults"), "status={status:?}");
    let payload = String::from_utf8(
        service.coordinator.query_stats(&query_id).expect("failed SET update should retain query stats").to_vec(),
    )
    .unwrap();
    assert!(payload.contains("fact.reason=flight_session_scoped_datafusion_nexus_keys_unsupported"), "{payload}");
    assert!(payload.contains("fact.configuration_scope=server_or_tenant_defaults"), "{payload}");
    assert!(!payload.contains("fact.config_key="), "raw config key must not be retained: {payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_set_plain_datafusion_key_still_succeeds() {
    let ctx = Arc::new(SessionContext::new());
    let service = DatafusionCudfFlightSqlService::new(
        ctx,
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
        ServerSessionProfile::Native,
        Arc::new(QueryAdmission::new(1).expect("test admission limit")),
        Duration::from_secs(30),
        AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
        #[cfg(feature = "cugraph")]
        None,
    );
    let query_id = "plain-datafusion-set".to_string();
    let sql = "SET datafusion.execution.batch_size = '1024'".to_string();
    let planning_lease = service.register_query(query_id.clone(), sql.clone(), None).unwrap();

    let count = service
        .execute_update_sql(planning_lease, sql, None, UpdateStatementKind::Statement)
        .await
        .expect("plain DataFusion SET should keep existing update-lane behavior");

    assert_eq!(count, NO_UPDATED_RECORDS_COUNT);
}

#[tokio::test]
async fn prepared_update_shutdown_promotion_preserves_operation_and_request_metadata() {
    let service = DatafusionCudfFlightSqlService::new(
        Arc::new(SessionContext::new()),
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
        ServerSessionProfile::Native,
        Arc::new(QueryAdmission::new(1).expect("test admission limit")),
        Duration::from_secs(30),
        AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
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
    service: &DatafusionCudfFlightSqlService,
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
    let service = DatafusionCudfFlightSqlService::new(
        Arc::new(SessionContext::new()),
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
        ServerSessionProfile::Native,
        Arc::new(QueryAdmission::new(1).expect("test admission limit")),
        Duration::ZERO,
        AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
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

fn only_ticket_query_id(service: &DatafusionCudfFlightSqlService) -> String {
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
