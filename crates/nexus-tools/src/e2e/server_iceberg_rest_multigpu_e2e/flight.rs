use std::{collections::BTreeMap, sync::Arc, time::Duration};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use arrow_flight::{
    FlightInfo,
    sql::{
        ActionCancelQueryRequest, ActionCancelQueryResult, Any, ProstMessageExt, TicketStatementQuery,
        client::FlightSqlServiceClient,
    },
};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_nexus::{
    native_report::bench_support::{ParityFloatTolerance, ParityStatus, compare_record_batches},
    table_format::iceberg::catalog::{
        IcebergCatalogBackend, NexusIcebergCatalogSource, RestIcebergCatalogBackend,
        build_scoped_catalog_provider_from_backend,
    },
};
use futures::TryStreamExt;
use prost::Message as _;
use tokio::{sync::Barrier, task::JoinSet};
use tonic::transport::Channel;

use super::{
    E2eResult,
    server::Params,
    validation::{parse_u64_field, parse_usize_field, require_value, tsv_section_single_row},
};

pub(super) const Q06_WAVE_SIZE: usize = 64;
pub(super) const QUERY_TIMEOUT: Duration = Duration::from_secs(180);
pub(super) const CANCEL_RESULT_CANCELLING: i32 = 2;

#[derive(Clone)]
pub(super) struct Workload {
    pub(super) id: &'static str,
    pub(super) sql: String,
    pub(super) ordered: bool,
}

pub(super) fn tpch_workloads(params: &Params) -> BTreeMap<&'static str, Workload> {
    let lineitem = params.table("lineitem");
    let customer = params.table("customer");
    let orders = params.table("orders");
    let nation = params.table("nation");
    BTreeMap::from([
        (
            "q01",
            Workload {
                id: "q01",
                ordered: true,
                sql: format!(
                    "SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS sum_qty, \
                     SUM(l_extendedprice) AS sum_base_price, \
                     SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, \
                     SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, \
                     AVG(l_quantity) AS avg_qty, AVG(l_extendedprice) AS avg_price, \
                     AVG(l_discount) AS avg_disc, COUNT(*) AS count_order FROM {lineitem} \
                     WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY \
                     GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus"
                ),
            },
        ),
        (
            "q03",
            Workload {
                id: "q03",
                ordered: true,
                sql: format!(
                    "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, \
                     o_orderdate, o_shippriority FROM {customer} JOIN {orders} ON c_custkey = o_custkey \
                     JOIN {lineitem} ON l_orderkey = o_orderkey WHERE c_mktsegment = 'BUILDING' \
                     AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15' \
                     GROUP BY l_orderkey, o_orderdate, o_shippriority \
                     ORDER BY revenue DESC, o_orderdate LIMIT 10"
                ),
            },
        ),
        (
            "q06",
            Workload {
                id: "q06",
                ordered: true,
                sql: format!(
                    "SELECT SUM(l_extendedprice * l_discount) AS revenue FROM {lineitem} \
                     WHERE l_shipdate >= DATE '1994-01-01' \
                     AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR \
                     AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"
                ),
            },
        ),
        (
            "q10",
            Workload {
                id: "q10",
                ordered: true,
                sql: format!(
                    "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue, \
                     c_acctbal, n_name, c_address, c_phone, c_comment FROM {customer} \
                     JOIN {orders} ON c_custkey = o_custkey JOIN {lineitem} ON l_orderkey = o_orderkey \
                     JOIN {nation} ON c_nationkey = n_nationkey \
                     WHERE o_orderdate >= DATE '1993-10-01' \
                     AND o_orderdate < DATE '1993-10-01' + INTERVAL '3' MONTH AND l_returnflag = 'R' \
                     GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment \
                     ORDER BY revenue DESC LIMIT 20"
                ),
            },
        ),
    ])
}

#[derive(Clone)]
pub(super) struct CpuReference {
    pub(super) schema: SchemaRef,
    pub(super) batches: Vec<RecordBatch>,
}

pub(super) async fn build_cpu_references(
    params: &Params,
    workloads: &BTreeMap<&'static str, Workload>,
) -> E2eResult<BTreeMap<&'static str, CpuReference>> {
    let backend: Arc<dyn IcebergCatalogBackend> = Arc::new(NexusIcebergCatalogSource::new(
        RestIcebergCatalogBackend::connect(params.rest_config()).await?.into_backend(),
    ));
    let provider = build_scoped_catalog_provider_from_backend(backend, "rest", &params.namespace).await?;
    let context = SessionContext::new_with_config(SessionConfig::new());
    context.register_catalog("lake", Arc::new(provider));

    let mut references = BTreeMap::new();
    for workload in workloads.values() {
        let dataframe = context.sql(&workload.sql).await?;
        let batches = dataframe.collect().await?;
        let schema = batches
            .first()
            .map(RecordBatch::schema)
            .ok_or_else(|| format!("CPU reference {} returned no batches", workload.id))?;
        let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        if rows == 0 {
            return Err(format!("CPU reference {} returned no rows", workload.id).into());
        }
        eprintln!("CPU reference {}: {rows} row(s)", workload.id);
        references.insert(workload.id, CpuReference { schema, batches });
    }
    Ok(references)
}

#[derive(Clone)]
pub(super) struct QueryEvidence {
    pub(super) phase: String,
    pub(super) workload: &'static str,
    pub(super) query_id: String,
    pub(super) device_ordinal: usize,
    pub(super) output_rows: u64,
    pub(super) d2h_count: u64,
    pub(super) d2h_bytes: u64,
    pub(super) parity_mode: String,
    pub(super) cpu_fingerprint: String,
    pub(super) native_fingerprint: String,
    pub(super) stats_payload: String,
}

pub(super) async fn run_wave(
    channel: &Channel,
    phase: &'static str,
    workloads: Vec<Workload>,
    cpu_reference: CpuReference,
) -> E2eResult<Vec<QueryEvidence>> {
    let barrier = Arc::new(Barrier::new(workloads.len() + 1));
    let mut tasks = JoinSet::new();
    for workload in workloads {
        let channel = channel.clone();
        let barrier = Arc::clone(&barrier);
        let cpu_reference = cpu_reference.clone();
        tasks.spawn(async move {
            barrier.wait().await;
            tokio::time::timeout(QUERY_TIMEOUT, execute_success(&channel, phase, workload, cpu_reference))
                .await
                .map_err(|_| format!("{phase} statement exceeded {} seconds", QUERY_TIMEOUT.as_secs()))?
        });
    }
    barrier.wait().await;
    collect_query_tasks(tasks).await
}

pub(super) async fn run_mixed_wave(
    channel: &Channel,
    phase: &'static str,
    workloads: Vec<Workload>,
    cpu_references: &BTreeMap<&'static str, CpuReference>,
) -> E2eResult<Vec<QueryEvidence>> {
    let barrier = Arc::new(Barrier::new(workloads.len() + 1));
    let mut tasks = JoinSet::new();
    for workload in workloads {
        let channel = channel.clone();
        let barrier = Arc::clone(&barrier);
        let cpu_reference = cpu_references.get(workload.id).expect("mixed CPU reference").clone();
        tasks.spawn(async move {
            barrier.wait().await;
            tokio::time::timeout(QUERY_TIMEOUT, execute_success(&channel, phase, workload, cpu_reference))
                .await
                .map_err(|_| format!("{phase} statement exceeded {} seconds", QUERY_TIMEOUT.as_secs()))?
        });
    }
    barrier.wait().await;
    collect_query_tasks(tasks).await
}

async fn collect_query_tasks(mut tasks: JoinSet<E2eResult<QueryEvidence>>) -> E2eResult<Vec<QueryEvidence>> {
    let mut rows = Vec::new();
    while let Some(result) = tasks.join_next().await {
        rows.push(result.map_err(|error| format!("query task failed: {error}"))??);
    }
    Ok(rows)
}

async fn execute_success(
    channel: &Channel,
    phase: &str,
    workload: Workload,
    cpu_reference: CpuReference,
) -> E2eResult<QueryEvidence> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let flight_info = client.execute(workload.sql.clone(), None).await?;
    collect_success_from_info(channel, phase, workload, cpu_reference, flight_info).await
}

pub(super) async fn collect_success_from_info(
    channel: &Channel,
    phase: &str,
    workload: Workload,
    cpu_reference: CpuReference,
    flight_info: FlightInfo,
) -> E2eResult<QueryEvidence> {
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let (native_schema, native_batches) = collect_flight_info(channel, flight_info).await?;
    let comparison = compare_record_batches(
        cpu_reference.schema.as_ref(),
        &cpu_reference.batches,
        native_schema.as_ref(),
        &native_batches,
        workload.ordered,
        ParityFloatTolerance::exact(),
    );
    if comparison.status != ParityStatus::Matched {
        return Err(format!(
            "{} {} CPU/native parity failed: status={} mode={} detail={:?}",
            phase,
            workload.id,
            comparison.status.as_str(),
            comparison.mode.as_str(),
            comparison.detail
        )
        .into());
    }

    let stats_payload = get_query_stats(channel, &query_id).await?;
    let planning = tsv_section_single_row(&stats_payload, "planning")?;
    let execution = tsv_section_single_row(&stats_payload, "execution")?;
    require_value(&planning, "outcome", "selected", &query_id)?;
    require_value(&planning, "post_native_execution_path", "native", &query_id)?;
    require_value(&execution, "planning_outcome", "selected", &query_id)?;
    require_value(&execution, "post_native_execution_path", "native", &query_id)?;
    require_value(&execution, "native_runtime_started", "true", &query_id)?;
    require_value(&execution, "runtime_outcome", "succeeded", &query_id)?;
    if !stats_payload.contains("# outcome=succeeded\n") || !stats_payload.contains("# sealed=true\n") {
        return Err(
            format!("successful query {query_id} was not retained as sealed+succeeded:\n{stats_payload}").into()
        );
    }

    let device_ordinal = parse_usize_field(&execution, "device_ordinal", &query_id)?;
    let output_rows = parse_u64_field(&execution, "output_rows", &query_id)?;
    let d2h_count = parse_u64_field(&execution, "device_to_host_materializations", &query_id)?;
    let d2h_bytes = parse_u64_field(&execution, "device_to_host_materialized_bytes", &query_id)?;
    if d2h_count == 0 || d2h_bytes == 0 {
        return Err(format!(
            "successful consumed query {query_id} lacks positive D2H evidence: count={d2h_count} bytes={d2h_bytes}"
        )
        .into());
    }
    let native_rows = native_batches.iter().map(RecordBatch::num_rows).sum::<usize>() as u64;
    if output_rows != native_rows {
        return Err(format!(
            "query {query_id} execution output_rows={output_rows} but Flight stream yielded {native_rows} rows"
        )
        .into());
    }

    Ok(QueryEvidence {
        phase: phase.to_string(),
        workload: workload.id,
        query_id,
        device_ordinal,
        output_rows,
        d2h_count,
        d2h_bytes,
        parity_mode: comparison.mode.as_str().to_string(),
        cpu_fingerprint: comparison.cpu_fingerprint,
        native_fingerprint: comparison.native_fingerprint,
        stats_payload,
    })
}

async fn collect_flight_info(channel: &Channel, flight_info: FlightInfo) -> E2eResult<(SchemaRef, Vec<RecordBatch>)> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let mut schema = None;
    let mut batches = Vec::new();
    for endpoint in flight_info.endpoint {
        let ticket = endpoint.ticket.ok_or("Flight SQL endpoint missing ticket")?;
        let mut stream = client.do_get(ticket).await?;
        while let Some(batch) = stream.try_next().await? {
            batches.push(batch);
        }
        schema = stream.schema().cloned().or(schema);
    }
    let schema =
        schema.or_else(|| batches.first().map(RecordBatch::schema)).ok_or("Flight stream did not expose schema")?;
    Ok((schema, batches))
}

pub(super) fn query_id_from_statement_flight_info(flight_info: &FlightInfo) -> E2eResult<String> {
    let ticket = flight_info
        .endpoint
        .first()
        .and_then(|endpoint| endpoint.ticket.as_ref())
        .ok_or("FlightInfo missing statement ticket")?;
    let any = Any::decode(&*ticket.ticket)?;
    let statement = any.unpack::<TicketStatementQuery>()?.ok_or("FlightInfo ticket is not TicketStatementQuery")?;
    Ok(String::from_utf8(statement.statement_handle.to_vec())?)
}

pub(super) async fn get_query_stats(channel: &Channel, query_id: &str) -> E2eResult<String> {
    action_payload(channel, "GetQueryStats", query_id.as_bytes().to_vec()).await
}

pub(super) async fn get_admission_stats(channel: &Channel) -> E2eResult<String> {
    action_payload(channel, "GetAdmissionStats", Vec::new()).await
}

async fn action_payload(channel: &Channel, action_type: &str, body: Vec<u8>) -> E2eResult<String> {
    let mut frames = action_frames(channel, action_type, body).await?;
    if frames.len() != 1 {
        return Err(format!("{action_type} returned {} results, expected exactly one", frames.len()).into());
    }
    Ok(String::from_utf8(frames.remove(0))?)
}

pub(super) async fn action_frames(channel: &Channel, action_type: &str, body: Vec<u8>) -> E2eResult<Vec<Vec<u8>>> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let action = arrow_flight::Action { r#type: action_type.to_string(), body: body.into() };
    let mut stream = client.do_action(action).await?;
    let mut frames = Vec::new();
    while let Some(result) = stream.message().await? {
        frames.push(result.body.to_vec());
    }
    if frames.is_empty() {
        return Err(format!("{action_type} returned no result").into());
    }
    Ok(frames)
}

pub(super) async fn lightweight_get_flight_info(channel: &Channel) -> E2eResult<FlightInfo> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let flight_info = client.execute("SELECT 1 AS liveness_probe".to_string(), None).await?;
    if flight_info.endpoint.is_empty() {
        return Err("GetFlightInfo(SELECT 1) returned no endpoint".into());
    }
    Ok(flight_info)
}

pub(super) async fn cancel_query(channel: &Channel, flight_info: &FlightInfo) -> E2eResult<ActionCancelQueryResult> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let request = ActionCancelQueryRequest { info: flight_info.encode_to_vec().into() };
    let action =
        arrow_flight::Action { r#type: "CancelQuery".to_string(), body: request.as_any().encode_to_vec().into() };
    let mut stream = client.do_action(action).await?;
    let result = stream.message().await?.ok_or("CancelQuery returned no result")?;
    if stream.message().await?.is_some() {
        return Err("CancelQuery returned more than one result".into());
    }
    let any = Any::decode(&*result.body)?;
    any.unpack::<ActionCancelQueryResult>()?.ok_or_else(|| "CancelQuery returned wrong result type".into())
}
