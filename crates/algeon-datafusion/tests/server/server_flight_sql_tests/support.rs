fn unwrap_flight_status(error: FlightError) -> tonic::Status {
    match error {
        FlightError::Tonic(status) => *status,
        other => panic!("expected tonic Status, got {other:?}"),
    }
}

fn query_id_from_status(status: &tonic::Status) -> String {
    status
        .metadata()
        .get(QUERY_ID_METADATA_KEY)
        .and_then(|value| value.to_str().ok())
        .expect("status must carry query id metadata")
        .to_string()
}

fn query_id_from_statement_flight_info(flight_info: &arrow_flight::FlightInfo) -> String {
    use prost::Message as _;

    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.as_ref()).unwrap();
    let any = Any::decode(&*ticket.ticket).unwrap();
    let query_id = if let Some(ticket) = any.unpack::<TicketStatementQuery>().unwrap() {
        ticket.statement_handle
    } else if let Some(ticket) = any.unpack::<CommandPreparedStatementQuery>().unwrap() {
        ticket.prepared_statement_handle
    } else {
        panic!("statement FlightInfo ticket must encode a statement or prepared-statement query")
    };
    String::from_utf8(query_id.to_vec()).unwrap()
}

async fn get_query_stats_raw(client: &mut FlightSqlServiceClient<tonic::transport::Channel>, query_id: &str) -> String {
    let action =
        arrow_flight::Action { r#type: "GetQueryStats".to_string(), body: query_id.as_bytes().to_vec().into() };
    let mut stream = client.do_action(action).await.unwrap();
    let result = stream.message().await.unwrap().expect("GetQueryStats action must return one result");
    assert!(stream.message().await.unwrap().is_none(), "GetQueryStats action returns exactly one result");
    String::from_utf8(result.body.to_vec()).unwrap()
}

/// A completed native Flight attempt publishes exactly one immutable report.
/// `GetQueryStats` is a retained projection of that report, so duplicate
/// headers would expose a second publication path rather than more telemetry.
fn assert_single_sealed_attempt_report(payload: &str) {
    let values = |key: &str| payload.lines().filter_map(|line| line.strip_prefix(key)).collect::<Vec<_>>();
    let attempt_ids = values("# attempt_id=");
    assert_eq!(attempt_ids.len(), 1, "expected exactly one published attempt report:\n{payload}");
    assert!(!attempt_ids[0].is_empty(), "attempt report identity must be non-empty:\n{payload}");
    let attempt_outcomes = values("# attempt_outcome=");
    assert_eq!(attempt_outcomes.len(), 1, "attempt report has one terminal outcome:\n{payload}");
    let outcomes = values("# outcome=");
    assert_eq!(outcomes.len(), 1, "attempt report has one canonical outcome:\n{payload}");
    assert_eq!(outcomes[0], attempt_outcomes[0], "attempt report owns the canonical outcome:\n{payload}");
    assert_eq!(
        values("# attempt_capacity_disposition=").len(),
        1,
        "attempt report has one capacity commit:\n{payload}"
    );
    assert!(
        payload.contains("# sealed=true"),
        "attempt report is available only after its observation seals:\n{payload}"
    );
    assert!(
        !payload.contains("# attempt_capacity_disposition=pending_release"),
        "a retained attempt report must include its release commit:\n{payload}"
    );
}

fn query_stats_tsv_single_row<'a>(payload: &'a str, section: &str) -> (Vec<&'a str>, Vec<&'a str>) {
    let marker = format!("[{section}]");
    let mut lines = payload
        .lines()
        .skip_while(|line| *line != marker)
        .skip(1)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));
    let header =
        lines.next().unwrap_or_else(|| panic!("missing [{section}] header in query_stats payload:\n{payload}"));
    let row = lines.next().unwrap_or_else(|| panic!("missing [{section}] row in query_stats payload:\n{payload}"));
    (header.split('\t').collect(), row.split('\t').collect())
}

fn tsv_value<'a>(header: &[&str], row: &'a [&str], column: &str) -> &'a str {
    let index = header
        .iter()
        .position(|name| *name == column)
        .unwrap_or_else(|| panic!("missing TSV column {column}; header={header:?}; row={row:?}"));
    row.get(index).copied().unwrap_or_else(|| panic!("missing TSV value for column {column}; row={row:?}"))
}

async fn cancel_query_raw(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    flight_info: &FlightInfo,
) -> ActionCancelQueryResult {
    let action_body = ActionCancelQueryRequest { info: flight_info.encode_to_vec().into() };
    let action =
        arrow_flight::Action { r#type: "CancelQuery".to_string(), body: action_body.as_any().encode_to_vec().into() };
    let mut stream = client.do_action(action).await.unwrap();
    let result = stream.message().await.unwrap().expect("CancelQuery action must return one result");
    assert!(stream.message().await.unwrap().is_none(), "CancelQuery action returns exactly one result");
    let any = Any::decode(&*result.body).unwrap();
    any.unpack::<ActionCancelQueryResult>().unwrap().unwrap()
}

async fn collect_metadata_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    flight_info: &arrow_flight::FlightInfo,
) -> Vec<arrow::record_batch::RecordBatch> {
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap()
}

/// Like `collect_metadata_batches`, but also returns the stream's schema
/// extracted from the IPC schema-header message. `try_collect::<Vec<_>>`
/// drops the schema and yields only RecordBatch payloads, so for empty-
/// result metadata RPCs (`get_primary_keys`, foreign-key RPCs) the batch
/// vector is empty and the schema would otherwise be unreachable.
async fn collect_metadata_schema_and_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    flight_info: &arrow_flight::FlightInfo,
) -> (arrow::datatypes::SchemaRef, Vec<arrow::record_batch::RecordBatch>) {
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let mut stream = client.do_get(ticket).await.unwrap();
    let mut batches = Vec::new();
    while let Some(batch) = stream.try_next().await.unwrap() {
        batches.push(batch);
    }
    let schema = stream.schema().cloned().expect("metadata stream must expose a schema");
    (schema, batches)
}

fn collect_name_and_type(batches: &[arrow::record_batch::RecordBatch]) -> Vec<(String, String)> {
    let names = collect_string_column(batches, "table_name");
    let types = collect_string_column(batches, "table_type");
    assert_eq!(names.len(), types.len(), "GetTables name/type columns desynced");
    names.into_iter().zip(types).collect()
}

fn collect_string_column(batches: &[arrow::record_batch::RecordBatch], column: &str) -> Vec<String> {
    let mut out = Vec::new();
    for batch in batches {
        let Some(index) = batch.schema().index_of(column).ok() else {
            continue;
        };
        let array = batch.column(index).as_any().downcast_ref::<StringArray>().expect("metadata column must be Utf8");
        for i in 0..array.len() {
            if array.is_valid(i) {
                out.push(array.value(i).to_string());
            }
        }
    }
    out
}

#[derive(Debug)]
struct ScanFailingTableProvider {
    schema: Arc<Schema>,
    scan_called: AtomicBool,
}

impl ScanFailingTableProvider {
    fn new(schema: Arc<Schema>) -> Self {
        Self { schema, scan_called: AtomicBool::new(false) }
    }

    fn scan_called(&self) -> bool {
        self.scan_called.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl TableProvider for ScanFailingTableProvider {
    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        self.scan_called.store(true, Ordering::SeqCst);
        Err(datafusion::error::DataFusionError::Execution("metadata regression provider row scan invoked".to_string()))
    }
}

#[derive(Debug)]
struct ControlledPlanReplacement {
    enabled: AtomicBool,
    replacement: LogicalPlan,
}

impl ControlledPlanReplacement {
    fn new(replacement: LogicalPlan) -> Self {
        Self { enabled: AtomicBool::new(false), replacement }
    }

    fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }
}

impl RelationPlanner for ControlledPlanReplacement {
    fn plan_relation(
        &self,
        relation: TableFactor,
        _context: &mut dyn RelationPlannerContext,
    ) -> datafusion::common::Result<RelationPlanning> {
        let alias = match &relation {
            TableFactor::Table { name, alias, .. }
                if self.enabled.load(Ordering::SeqCst)
                    && name.to_string().eq_ignore_ascii_case("information_schema.tables") =>
            {
                alias.clone()
            },
            _ => return Ok(RelationPlanning::Original(Box::new(relation))),
        };
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(self.replacement.clone(), alias))))
    }
}

/// Warm the process-wide CUDA context exactly once. The first CUDA call pays a
/// multi-second `cuInit()` cost; doing it here (every test reserves an address
/// before spawning a server) keeps that cost off each server's connect window,
/// and the `OnceLock` collapses concurrent callers onto a single init.
fn warm_cuda_once() {
    static CUDA_WARM: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        // Best-effort: serve_with_listener re-runs (and surfaces) any error.
        let _ = algeon_datafusion::server::init_cuda_device(0);
    });
}

/// Reserve a loopback port and hand back the live listener so the caller can
/// pass it straight to `serve_with_listener`. Keeping the socket bound closes
/// the bind/serve gap where a parallel test could steal the port.
fn free_local_addr() -> (TcpListener, std::net::SocketAddr) {
    warm_cuda_once();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

async fn connect_with_retry(addr: std::net::SocketAddr) -> FlightSqlServiceClient<tonic::transport::Channel> {
    let uri = format!("http://{addr}");
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut backoff = Duration::from_millis(10);
    loop {
        match Endpoint::from_shared(uri.clone()).unwrap().connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) => {
                if std::time::Instant::now() >= deadline {
                    panic!("Flight SQL server did not accept connections at {addr} within 15s: {error:?}");
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
            },
        }
    }
}

async fn wait_for_shared_ledger_queue(backend: &AlgeonGpuBackend, expected_queued_queries: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let snapshot = backend.admission_snapshot().unwrap();
        if snapshot.active_queries == 1 && snapshot.queued_queries == expected_queued_queries {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "queries did not queue on the shared backend: expected active=1 queued={expected_queued_queries}, snapshot={snapshot:?}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Wait until the backend query-service GPU admission ledger is empty.
///
/// Native CUDA cleanup is handed off to the service GPU execution pool, so a
/// just-finished handle may still briefly occupy `active_queries` / cleanup
/// accounting while stream pool work drains. DataFusion-baseline assertions use this
/// fence before reading ledger counters.
async fn wait_for_empty_gpu_ledger(backend: &AlgeonGpuBackend) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let snapshot = backend.admission_snapshot().unwrap();
        if snapshot.active_queries == 0 && snapshot.queued_queries == 0 {
            return;
        }
        assert!(Instant::now() < deadline, "backend GPU admission ledger did not become empty: snapshot={snapshot:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn execute_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> Vec<arrow::record_batch::RecordBatch> {
    let flight_info = client.execute(sql.to_string(), None).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap()
}

async fn execute_flight_info_with_stream_schema(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    flight_info: FlightInfo,
) -> (Schema, Vec<arrow::record_batch::RecordBatch>) {
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let mut stream = client.do_get(ticket).await.unwrap();
    let mut batches = Vec::new();
    while let Some(batch) = stream.try_next().await.unwrap() {
        batches.push(batch);
    }
    let schema = stream.schema().expect("Flight stream must include a schema").as_ref().clone();
    (schema, batches)
}

async fn create_prepared_raw(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: String,
) -> ActionCreatePreparedStatementResult {
    use prost::Message as _;

    let action_body = ActionCreatePreparedStatementRequest { query: sql, transaction_id: None };
    let action = arrow_flight::Action {
        r#type: "CreatePreparedStatement".to_string(),
        body: action_body.as_any().encode_to_vec().into(),
    };
    let mut stream = client.do_action(action).await.unwrap();
    let result = stream.message().await.unwrap().expect("CreatePreparedStatement action must return one result");
    let any = Any::decode(&*result.body).unwrap();
    any.unpack::<ActionCreatePreparedStatementResult>().unwrap().unwrap()
}

async fn close_prepared_raw(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    handle: prost::bytes::Bytes,
) {
    use arrow_flight::sql::ActionClosePreparedStatementRequest;
    use prost::Message as _;

    let action_body = ActionClosePreparedStatementRequest { prepared_statement_handle: handle };
    let action = arrow_flight::Action {
        r#type: "ClosePreparedStatement".to_string(),
        body: action_body.as_any().encode_to_vec().into(),
    };
    let mut stream = client.do_action(action).await.unwrap();
    while stream.message().await.unwrap().is_some() {}
}

fn write_update_edges_parquet(path: &Path) {
    write_int64_pairs_parquet(path, ["src", "dst"], [0, 1], [1, 2]);
}

fn write_unsupported_decimal_list_parquet(path: &Path) {
    let child = Arc::new(Field::new("element", DataType::Decimal128(10, 2), true));
    let values = ListArray::new(
        Arc::clone(&child),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2])),
        Arc::new(Decimal128Array::from(vec![100_i128, 200]).with_precision_and_scale(10, 2).unwrap()),
        None,
    );
    let schema = Arc::new(Schema::new(vec![Field::new("unsupported_values", DataType::List(child), true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values)]).unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_basic_list_parquet(path: &Path) {
    let string_child = Arc::new(Field::new("element", DataType::Utf8, true));
    let null_strings = || {
        Arc::new(ListArray::new(
            Arc::clone(&string_child),
            OffsetBuffer::new(ScalarBuffer::from(vec![0; 13])),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
            Some(NullBuffer::from(vec![false; 12])),
        )) as arrow::array::ArrayRef
    };
    let int_child = Arc::new(Field::new("element", DataType::Int64, true));
    let ports = ListArray::new(
        Arc::clone(&int_child),
        OffsetBuffer::new(ScalarBuffer::from((0..=12).collect::<Vec<i32>>())),
        Arc::new(Int64Array::from_iter_values(100..112)),
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("vertexId", DataType::Int64, false),
        Field::new("vertexAttributes__tag", DataType::List(Arc::clone(&string_child)), true),
        Field::new("vertexAttributes__label", DataType::List(Arc::clone(&string_child)), true),
        Field::new("vertexAttributes__rank", DataType::List(int_child), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from_iter_values(0..12)), null_strings(), null_strings(), Arc::new(ports)],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_int64_pairs_parquet(path: &Path, names: [&str; 2], left: [i64; 2], right: [i64; 2]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(names[0], DataType::Int64, false),
        Field::new(names[1], DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(left.to_vec())), Arc::new(Int64Array::from(right.to_vec()))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn int_value(array: &arrow::array::ArrayRef) -> Option<i64> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|array| array.value(0))
        .or_else(|| array.as_any().downcast_ref::<Int32Array>().map(|array| i64::from(array.value(0))))
}

fn sql_info_string(batch: &arrow::record_batch::RecordBatch, info: SqlInfo) -> String {
    let (values, row) = sql_info_values(batch, info);
    assert_eq!(values.type_id(row), 0, "SqlInfo {info:?} must be encoded as a string");
    values
        .child(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("SqlInfo string child must be StringArray")
        .value(values.value_offset(row))
        .to_string()
}

fn sql_info_bool(batch: &arrow::record_batch::RecordBatch, info: SqlInfo) -> bool {
    let (values, row) = sql_info_values(batch, info);
    assert_eq!(values.type_id(row), 1, "SqlInfo {info:?} must be encoded as a bool");
    values
        .child(1)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("SqlInfo bool child must be BooleanArray")
        .value(values.value_offset(row))
}

fn sql_info_i32(batch: &arrow::record_batch::RecordBatch, info: SqlInfo) -> i32 {
    let (values, row) = sql_info_values(batch, info);
    assert_eq!(values.type_id(row), 3, "SqlInfo {info:?} must be encoded as an i32");
    values
        .child(3)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("SqlInfo i32 child must be Int32Array")
        .value(values.value_offset(row))
}

fn sql_info_values(batch: &arrow::record_batch::RecordBatch, info: SqlInfo) -> (&UnionArray, usize) {
    let names =
        batch.column(0).as_any().downcast_ref::<UInt32Array>().expect("SqlInfo info_name column must be UInt32Array");
    let values =
        batch.column(1).as_any().downcast_ref::<UnionArray>().expect("SqlInfo value column must be UnionArray");
    let expected = info as u32;
    let row = (0..names.len())
        .find(|row| names.value(*row) == expected)
        .unwrap_or_else(|| panic!("missing SqlInfo {info:?}"));
    (values, row)
}

fn sql_info_bitmask(values: impl IntoIterator<Item = i32>) -> i32 {
    values.into_iter().fold(0, |mask, value| mask | (1_i32 << value))
}

fn assert_error_metadata(status: &tonic::Status, code: &str, kind: &str, error_status: &str) {
    let actual_code = status.metadata().get(ERROR_CODE_METADATA_KEY).expect("error code metadata must be present");
    assert_eq!(actual_code.to_str().expect("error code metadata must be ASCII"), code);
    let actual_kind = status.metadata().get(ERROR_KIND_METADATA_KEY).expect("error kind metadata must be present");
    assert_eq!(actual_kind.to_str().expect("error kind metadata must be ASCII"), kind);
    let actual_status = status
        .metadata()
        .get(ERROR_STATUS_METADATA_KEY)
        .expect("error retry/remediation status metadata must be present");
    assert_eq!(actual_status.to_str().expect("error status metadata must be ASCII"), error_status);
}

fn assert_query_id_metadata(status: &tonic::Status) {
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!query_id.is_empty());
}

fn assert_no_diagnostic_fields(message: &str) {
    for needle in ["code=", "kind=", "status=", "operation=", "facts=", "source="] {
        assert!(!message.contains(needle), "client message leaked diagnostic field {needle}: {message}");
    }
}
