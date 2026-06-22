#![cfg(feature = "server")]

use std::{fs::File, net::TcpListener, path::Path, sync::Arc, time::Duration};

use arrow::{
    array::{Array, BooleanArray, Int32Array, Int64Array, StringArray, UInt8Array, UInt32Array, UnionArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::{
    FlightInfo,
    error::FlightError,
    sql::{
        ActionCancelQueryRequest, ActionCancelQueryResult, ActionCreatePreparedStatementRequest,
        ActionCreatePreparedStatementResult, Any, CommandGetCrossReference, CommandGetDbSchemas,
        CommandGetExportedKeys, CommandGetImportedKeys, CommandGetPrimaryKeys, CommandGetTables,
        CommandPreparedStatementQuery, ProstMessageExt, SqlInfo, SqlNullOrdering, SqlOuterJoinsSupportLevel,
        SqlSupportedGroupBy, SqlSupportedResultSetConcurrency, SqlSupportedResultSetType, SqlSupportedTransaction,
        SqlSupportedUnions, TicketStatementQuery, client::FlightSqlServiceClient,
    },
};
use datafusion_nexus::server::{
    ServerConfig,
    error::{ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_STATUS_METADATA_KEY},
    serve_with_listener,
};
use futures::TryStreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use prost::Message as _;
use tonic::{Code, transport::Endpoint};

const QUERY_ID_METADATA_KEY: &str = "x-datafusion-nexus-query-id";
const CANCEL_RESULT_CANCELLING: i32 = 2;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_client_can_run_select_one() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select 1 as one".to_string(), None).await.unwrap();
    let schema = flight_info.clone().try_decode_schema().unwrap();
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "one");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_returns_completed_statement_observability() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select 1 as one".to_string(), None).await.unwrap();
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(int_value(batches[0].column(0)), Some(1));

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# datafusion_nexus_query_stats=1"), "{payload}");
    assert!(payload.contains(&format!("# query_id={query_id}")), "{payload}");
    assert!(payload.contains("# outcome=succeeded"), "{payload}");
    assert!(payload.contains("[physical_plan]"), "{payload}");

    let (header, row) = query_stats_tsv_single_row(&payload, "execution");
    assert_eq!(tsv_value(&header, &row, "runtime_outcome"), "succeeded", "{payload}");
    assert_eq!(tsv_value(&header, &row, "output_batches"), "1", "{payload}");
    assert_eq!(tsv_value(&header, &row, "output_rows"), "1", "{payload}");
    assert!(tsv_value(&header, &row, "output_bytes").parse::<usize>().unwrap() > 0, "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_returns_structured_error_after_planning_failure() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("select missing from (select 1 as one) as t".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    let query_id = query_id_from_status(&status);

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# outcome=failed"), "{payload}");
    assert!(payload.contains("[structured_error]"), "{payload}");
    assert!(payload.contains("code=field_not_found"), "{payload}");
    assert!(
        payload.contains("operation=datafusion_nexus.server.flight.get_flight_info_statement.create_logical_plan"),
        "{payload}"
    );
    assert!(payload.contains("missing"), "{payload}");
    assert!(payload.contains("one"), "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_surfaces_complete_cpu_fallback_rejection() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-stats-fallback-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let _ = execute_batches(
        &mut client,
        &format!(
            "CREATE EXTERNAL TABLE stats_fallback_edges STORED AS PARQUET LOCATION '{}'",
            parquet_path.to_string_lossy()
        ),
    )
    .await;
    let flight_info = client
        .execute("SELECT src FROM stats_fallback_edges WHERE TRY_CAST(src AS INT) >= 0 ORDER BY src".to_string(), None)
        .await
        .unwrap();
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let batches = collect_metadata_batches(&mut client, &flight_info).await;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# outcome=succeeded"), "{payload}");

    let (planning_header, planning_row) = query_stats_tsv_single_row(&payload, "planning");
    assert_eq!(tsv_value(&planning_header, &planning_row, "outcome"), "rejected_before_execution", "{payload}");
    assert_eq!(
        tsv_value(&planning_header, &planning_row, "reason_code"),
        "native_sort_preserving_merge_unsupported",
        "{payload}"
    );
    assert_eq!(tsv_value(&planning_header, &planning_row, "unsupported_category"), "operator_sort", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "post_native_execution_path"), "cpu", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "native_plan_fragment_count"), "0", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "host_datafusion_boundary_count"), "0", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "rejected_before_execution"), "true", "{payload}");

    let (execution_header, execution_row) = query_stats_tsv_single_row(&payload, "execution");
    assert_eq!(
        tsv_value(&execution_header, &execution_row, "planning_outcome"),
        "rejected_before_execution",
        "{payload}"
    );
    assert_eq!(tsv_value(&execution_header, &execution_row, "post_native_execution_path"), "cpu", "{payload}");
    assert_eq!(tsv_value(&execution_header, &execution_row, "native_runtime_started"), "false", "{payload}");
    assert_eq!(tsv_value(&execution_header, &execution_row, "runtime_outcome"), "not_started", "{payload}");
    assert_eq!(tsv_value(&execution_header, &execution_row, "output_rows"), "not_run", "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_unknown_query_returns_not_found() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let action = arrow_flight::Action {
        r#type: "GetQueryStats".to_string(),
        body: b"missing-query-id".as_slice().to_vec().into(),
    };
    let error = client.do_action(action).await.unwrap_err();
    let status = unwrap_flight_status(error);

    server.abort();

    assert_eq!(status.code(), Code::NotFound);
    assert!(status.message().contains("unknown or expired query_id"), "{status:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_returns_transaction_structured_error() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client
        .execute("select 1 as one".to_string(), Some(prost::bytes::Bytes::from_static(b"txn")))
        .await
        .unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::Unimplemented);
    let query_id = query_id_from_status(&status);

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# outcome=failed"), "{payload}");
    assert!(payload.contains("code=unsupported_operator"), "{payload}");
    assert!(payload.contains("operation=datafusion_nexus.server.flight.transaction"), "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_returns_cancelled_for_unconsumed_ticket() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select 1 as one".to_string(), None).await.unwrap();
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let result = cancel_query_raw(&mut client, &flight_info).await;
    assert_eq!(result.result, CANCEL_RESULT_CANCELLING);

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# outcome=failed"), "{payload}");
    assert!(payload.contains("# sealed=true"), "{payload}");
    assert!(payload.contains("[structured_error]"), "{payload}");
    assert!(payload.contains("code=cancelled"), "{payload}");
    assert!(payload.contains("operation=datafusion_nexus.server.flight.cancel_query"), "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_view_statement_executes_once_and_persists_across_statements() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client
        .execute(
            "CREATE VIEW target_edges AS \
         SELECT * FROM (VALUES (0::BIGINT, 1::BIGINT), (1::BIGINT, 2::BIGINT)) AS t(src, dst)"
                .to_string(),
            None,
        )
        .await
        .unwrap();
    let advertised_schema = flight_info.clone().try_decode_schema().unwrap();
    let (stream_schema, batches) = execute_flight_info_with_stream_schema(&mut client, flight_info).await;
    assert_eq!(advertised_schema.fields().len(), 0, "DDL FlightInfo must advertise no result-set columns");
    assert_eq!(stream_schema.fields().len(), 0, "DDL DoGet stream must match the advertised empty schema");
    assert_eq!(batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 0);

    let batches = execute_batches(&mut client, "SELECT COUNT(*) AS edge_count FROM target_edges").await;

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_create_view_query_advertises_empty_schema_and_persists() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let mut prepared = client
        .prepare(
            "CREATE VIEW prepared_target_edges AS \
             SELECT * FROM (VALUES (10::BIGINT, 11::BIGINT), (11::BIGINT, 12::BIGINT)) AS t(src, dst)"
                .to_string(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(prepared.dataset_schema().unwrap().fields().len(), 0, "DDL prepared statements do not return rows");

    let flight_info = prepared.execute().await.unwrap();
    let advertised_schema = flight_info.clone().try_decode_schema().unwrap();
    let (stream_schema, batches) = execute_flight_info_with_stream_schema(&mut client, flight_info).await;
    assert_eq!(advertised_schema.fields().len(), 0, "prepared DDL FlightInfo must advertise no result set");
    assert_eq!(stream_schema.fields().len(), 0, "prepared DDL DoGet stream must match the advertised empty schema");
    assert_eq!(batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 0);

    prepared.close().await.unwrap();
    let batches = execute_batches(&mut client, "SELECT src, dst FROM prepared_target_edges ORDER BY src").await;

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema().fields().len(), 2);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int_value(batches[0].column(0)), Some(10));
    assert_eq!(int_value(batches[0].column(1)), Some(11));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_table_on_view_reports_object_type_mismatch() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let _ = execute_batches(&mut client, "CREATE VIEW drop_view_probe AS SELECT 1 AS x").await;

    let error = client.execute("DROP TABLE drop_view_probe".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("is a view"), "message must explain actual object type: {}", status.message());
    assert!(status.message().contains("DROP VIEW"), "message must name the corrective DDL: {}", status.message());
    assert_no_diagnostic_fields(status.message());
    let payload = get_query_stats_raw(&mut client, &query_id_from_status(&status)).await;
    assert!(
        payload.contains("operation=datafusion_nexus.server.flight.get_flight_info_statement.create_logical_plan"),
        "{payload}"
    );
    assert!(payload.contains("fact.object_name=drop_view_probe"), "{payload}");
    assert!(payload.contains("fact.requested_table_type=TABLE"), "{payload}");
    assert!(payload.contains("fact.actual_table_type=VIEW"), "{payload}");

    let error = client.execute_update("DROP TABLE IF EXISTS drop_view_probe".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("DROP VIEW"));
    assert_no_diagnostic_fields(status.message());

    let _ = execute_batches(&mut client, "DROP VIEW drop_view_probe").await;
    let error = client.execute("SELECT * FROM drop_view_probe".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);

    server.abort();

    assert_eq!(status.code(), Code::NotFound);
    assert_error_metadata(&status, "object_not_found", "not_found", "permanent");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_view_on_table_reports_object_type_mismatch() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let _ = execute_batches(&mut client, "CREATE TABLE drop_table_probe AS SELECT 1 AS x").await;

    let error = client.execute("DROP VIEW drop_table_probe".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("base table"), "message must explain actual object type: {}", status.message());
    assert!(status.message().contains("DROP TABLE"), "message must name the corrective DDL: {}", status.message());
    assert_no_diagnostic_fields(status.message());

    let error = client.execute_update("DROP VIEW IF EXISTS drop_table_probe".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("DROP TABLE"));
    assert_no_diagnostic_fields(status.message());

    let _ = execute_batches(&mut client, "DROP TABLE drop_table_probe").await;
    let error = client.execute("SELECT * FROM drop_table_probe".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);

    server.abort();

    assert_eq!(status.code(), Code::NotFound);
    assert_error_metadata(&status, "object_not_found", "not_found", "permanent");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_update_create_external_table_executes_and_persists() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-update-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let record_count = client
        .execute_update(
            format!(
                "CREATE EXTERNAL TABLE update_edges_direct STORED AS PARQUET LOCATION '{}'",
                parquet_path.to_string_lossy()
            ),
            None,
        )
        .await
        .unwrap();
    let batches = execute_batches(&mut client, "SELECT COUNT(*) AS edge_count FROM update_edges_direct").await;

    server.abort();

    assert_eq!(record_count, 0, "DDL statements do not update table rows");
    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_update_create_external_table_executes_and_persists() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-prepared-update-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let mut prepared = client
        .prepare(
            format!(
                "CREATE EXTERNAL TABLE update_edges_prepared STORED AS PARQUET LOCATION '{}'",
                parquet_path.to_string_lossy()
            ),
            None,
        )
        .await
        .unwrap();
    assert_eq!(prepared.dataset_schema().unwrap().fields().len(), 0, "DDL prepared statements do not return rows");

    let record_count = prepared.execute_update().await.unwrap();
    prepared.close().await.unwrap();
    let batches = execute_batches(&mut client, "SELECT COUNT(*) AS edge_count FROM update_edges_prepared").await;

    server.abort();

    assert_eq!(record_count, 0, "DDL statements do not update table rows");
    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_ddl_create_action_omits_dataset_schema_bytes() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-prepared-schema-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let prepared = create_prepared_raw(
        &mut client,
        format!(
            "CREATE EXTERNAL TABLE schema_probe_edges STORED AS PARQUET LOCATION '{}'",
            parquet_path.to_string_lossy()
        ),
    )
    .await;
    close_prepared_raw(&mut client, prepared.prepared_statement_handle.clone()).await;

    server.abort();

    assert!(
        prepared.dataset_schema.is_empty(),
        "non-result-set prepared DDL must omit dataset_schema rather than advertise a 0-column result"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_create_action_handle_resolves_query_stats() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let prepared = create_prepared_raw(&mut client, "select 1 as one".to_string()).await;
    let query_id =
        String::from_utf8(prepared.prepared_statement_handle.to_vec()).expect("handle must be UTF-8 query id");
    let payload = get_query_stats_raw(&mut client, &query_id).await;
    close_prepared_raw(&mut client, prepared.prepared_statement_handle).await;

    server.abort();

    assert!(payload.contains(&format!("# query_id={query_id}")), "{payload}");
    assert!(payload.contains("# outcome=succeeded"), "{payload}");
    assert!(payload.contains("# sql=select 1 as one"), "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_query_create_external_table_executes_and_persists() {
    // JDBC clients such as DataGrip wrap every statement in a server-side
    // prepared statement and run it through `execute()` (the query channel),
    // never `execute_update()`. DDL must therefore execute on the prepared
    // query path the same way a non-prepared statement does.
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-prepared-query-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let mut prepared = client
        .prepare(
            format!(
                "CREATE EXTERNAL TABLE query_edges_prepared STORED AS PARQUET LOCATION '{}'",
                parquet_path.to_string_lossy()
            ),
            None,
        )
        .await
        .unwrap();
    assert_eq!(prepared.dataset_schema().unwrap().fields().len(), 0, "DDL prepared statements do not return rows");

    // Execute through the query channel; DDL yields an empty result set rather
    // than the `execute_update` rejection that used to block DataGrip-class
    // clients.
    let flight_info = prepared.execute().await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let ddl_batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    let ddl_rows: usize = ddl_batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(ddl_rows, 0, "DDL executed through the query channel returns no rows");

    prepared.close().await.unwrap();
    let batches = execute_batches(&mut client, "SELECT COUNT(*) AS edge_count FROM query_edges_prepared").await;

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

// Regression for the observed JDBC/DataGrip flow: the same prepared DDL handle
// is first executed as an update, then the client also asks for a query ticket.
// The query-channel pass must be a no-op with an empty result, not a duplicate
// DDL execution or a client-visible failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_ddl_update_then_query_double_send_is_deduplicated() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-dedup-mixed-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let create_sql = format!(
        "CREATE EXTERNAL TABLE dedup_mixed_edges STORED AS PARQUET LOCATION '{}'",
        parquet_path.to_string_lossy()
    );

    let mut prepared = client.prepare(create_sql, None).await.unwrap();
    assert_eq!(prepared.dataset_schema().unwrap().fields().len(), 0, "DDL prepared statements do not return rows");

    let update_count = prepared.execute_update().await.unwrap();
    assert_eq!(update_count, 0, "DDL statements do not update table rows");

    let flight_info = prepared.execute().await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let rows: usize =
        client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap().iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 0, "query-channel duplicate of an already-executed DDL handle must be an empty no-op");

    prepared.close().await.unwrap();
    let batches = execute_batches(&mut client, "SELECT COUNT(*) AS edge_count FROM dedup_mixed_edges").await;

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

// Regression: a prepared DDL handle re-executed twice — the double-send a
// JDBC client such as DataGrip triggers — must succeed both times. DataGrip /
// the Arrow Flight SQL JDBC driver classify DDL prepared statements as updates
// when no result-set schema is advertised. The second execution is deduplicated
// into a no-op instead of failing with "already exists".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_ddl_double_execute_is_deduplicated_on_update_channel() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-dedup-update-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let create_sql = format!(
        "CREATE EXTERNAL TABLE dedup_update_edges STORED AS PARQUET LOCATION '{}'",
        parquet_path.to_string_lossy()
    );

    // Prepare once; the client holds a single handle and re-uses it.
    let mut prepared = client.prepare(create_sql, None).await.unwrap();
    assert_eq!(prepared.dataset_schema().unwrap().fields().len(), 0, "DDL prepared statements do not return rows");

    let first = prepared.execute_update().await.unwrap();
    assert_eq!(first, 0, "DDL statements do not update table rows");

    // The double-send: same handle, executed again. Must be a no-op, not an
    // "already exists" rejection.
    let second = prepared.execute_update().await.unwrap();
    assert_eq!(second, 0, "re-executing the same prepared DDL handle must be a deduplicated no-op");

    prepared.close().await.unwrap();
    let batches = execute_batches(&mut client, "SELECT COUNT(*) AS edge_count FROM dedup_update_edges").await;

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

// Regression: the same double-send on the query channel (a client that runs a
// prepared DDL through `execute()` + `do_get` twice) is likewise deduplicated.
// The second `do_get` returns an empty result rather than re-running the DDL.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_ddl_double_execute_is_deduplicated_on_query_channel() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-dedup-query-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let create_sql = format!(
        "CREATE EXTERNAL TABLE dedup_query_edges STORED AS PARQUET LOCATION '{}'",
        parquet_path.to_string_lossy()
    );

    let mut prepared = client.prepare(create_sql, None).await.unwrap();
    assert_eq!(prepared.dataset_schema().unwrap().fields().len(), 0, "DDL prepared statements do not return rows");

    // First execution on the query channel: getFlightInfo + do_get.
    let info1 = prepared.execute().await.unwrap();
    let ticket1 = info1.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let rows1: usize =
        client.do_get(ticket1).await.unwrap().try_collect::<Vec<_>>().await.unwrap().iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows1, 0, "DDL executed through the query channel returns no rows");

    // The double-send: same handle, executed again on the query channel.
    let info2 = prepared.execute().await.unwrap();
    let ticket2 = info2.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let rows2: usize =
        client.do_get(ticket2).await.unwrap().try_collect::<Vec<_>>().await.unwrap().iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows2, 0, "re-executing the same prepared DDL handle must be a deduplicated no-op");

    prepared.close().await.unwrap();
    let batches = execute_batches(&mut client, "SELECT COUNT(*) AS edge_count FROM dedup_query_edges").await;

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_update_invalid_sql_uses_server_error_presentation() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute_update("selekt 1".to_string(), None).await.unwrap_err();

    server.abort();

    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "sql_parse", "invalid_input", "permanent");
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_update_rejects_transaction_id_with_structured_error() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let txn_id = prost::bytes::Bytes::from_static(b"fake-txn");
    let error = client
        .execute_update("CREATE VIEW tx_update_probe AS SELECT 1 AS x".to_string(), Some(txn_id))
        .await
        .unwrap_err();

    server.abort();

    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::Unimplemented);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "unsupported_operator", "unsupported", "permanent");
    assert!(status.message().contains("transactions are not supported"));
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_update_rejects_select_without_side_effect_path() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute_update("SELECT 1 AS one".to_string(), None).await.unwrap_err();

    server.abort();

    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("statement query endpoint"));
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_query_rejects_sql_prepare_namespace() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("PREPARE local_stmt AS SELECT 1".to_string(), None).await.unwrap_err();

    server.abort();

    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("SQL PREPARE"));
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_planning_errors_include_query_id_metadata() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("selekt 1".to_string(), None).await.unwrap_err();

    server.abort();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::InvalidArgument);
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!query_id.is_empty());
    assert_error_metadata(&status, "sql_parse", "invalid_input", "permanent");
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_table_errors_are_not_found_with_structured_code() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("select * from does_not_exist".to_string(), None).await.unwrap_err();

    server.abort();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::NotFound);
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!query_id.is_empty());
    assert_error_metadata(&status, "object_not_found", "not_found", "permanent");
    assert!(status.message().contains("does_not_exist"));
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_column_errors_are_invalid_input_with_structured_code() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("select missing from (select 1 as one) as t".to_string(), None).await.unwrap_err();

    server.abort();

    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!query_id.is_empty());
    assert_error_metadata(&status, "field_not_found", "invalid_input", "permanent");
    assert!(status.message().contains("missing"), "missing field absent from message: {}", status.message());
    assert!(status.message().contains("one"), "valid field absent from message: {}", status.message());
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unconsumed_statement_ticket_holds_admission_permit() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.max_in_flight = 1;
    config.statement_ticket_ttl = Duration::from_secs(30);

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut first_client = connect_with_retry(addr).await;
    let mut second_client = connect_with_retry(addr).await;

    let first_info = first_client.execute("select 1 as one".to_string(), None).await.unwrap();
    let first_ticket = first_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();

    let mut second = tokio::spawn(async move { second_client.execute("select 2 as two".to_string(), None).await });
    assert!(
        tokio::time::timeout(Duration::from_millis(150), &mut second).await.is_err(),
        "second GetFlightInfo must wait while the first ticket holds the only admission permit"
    );

    let batches = first_client.do_get(first_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(int_value(batches[0].column(0)), Some(1));

    let second_info = tokio::time::timeout(Duration::from_secs(5), second).await.unwrap().unwrap().unwrap();
    let second_ticket = second_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = first_client.do_get(second_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();

    server.abort();

    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abandoned_statement_ticket_expires_and_releases_admission_permit() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.max_in_flight = 1;
    config.statement_ticket_ttl = Duration::from_millis(50);

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut first_client = connect_with_retry(addr).await;
    let mut second_client = connect_with_retry(addr).await;

    let first_info = first_client.execute("select 1 as one".to_string(), None).await.unwrap();
    let first_ticket = first_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();

    let second_info =
        tokio::time::timeout(Duration::from_secs(5), second_client.execute("select 2 as two".to_string(), None))
            .await
            .expect("second GetFlightInfo should proceed after the first ticket expires")
            .unwrap();
    let second_ticket = second_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = second_client.do_get(second_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(int_value(batches[0].column(0)), Some(2));

    let error = first_client.do_get(first_ticket).await.unwrap_err();

    server.abort();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("expired"));
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!status.message().contains(query_id), "query id must stay in metadata, not message");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_sql_info_advertises_core_server_capabilities() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let requested = vec![
        SqlInfo::FlightSqlServerName,
        SqlInfo::FlightSqlServerSql,
        SqlInfo::FlightSqlServerReadOnly,
        SqlInfo::FlightSqlServerTransaction,
        SqlInfo::FlightSqlServerCancel,
        SqlInfo::SqlIdentifierQuoteChar,
        SqlInfo::SqlNullOrdering,
        SqlInfo::SqlSupportedGroupBy,
        SqlInfo::SqlOuterJoinsSupportLevel,
        SqlInfo::SqlSchemaTerm,
        SqlInfo::SqlCatalogTerm,
        SqlInfo::SqlCatalogAtStart,
        SqlInfo::SqlSupportedUnions,
        SqlInfo::SqlTransactionsSupported,
        SqlInfo::SqlSupportedResultSetTypes,
        SqlInfo::SqlSupportedConcurrenciesForResultSetForwardOnly,
    ];
    let flight_info = client.get_sql_info(requested).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();

    server.abort();

    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 16);
    assert_eq!(sql_info_string(batch, SqlInfo::FlightSqlServerName), "datafusion-nexus");
    assert!(sql_info_bool(batch, SqlInfo::FlightSqlServerSql));
    assert!(!sql_info_bool(batch, SqlInfo::FlightSqlServerReadOnly));
    assert_eq!(sql_info_i32(batch, SqlInfo::FlightSqlServerTransaction), SqlSupportedTransaction::None as i32);
    assert!(sql_info_bool(batch, SqlInfo::FlightSqlServerCancel));
    assert_eq!(sql_info_string(batch, SqlInfo::SqlIdentifierQuoteChar), r#"""#);
    assert_eq!(sql_info_i32(batch, SqlInfo::SqlNullOrdering), SqlNullOrdering::SqlNullsSortedHigh as i32);
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlSupportedGroupBy),
        sql_info_bitmask([
            SqlSupportedGroupBy::SqlGroupByUnrelated as i32,
            SqlSupportedGroupBy::SqlGroupByBeyondSelect as i32,
        ])
    );
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlOuterJoinsSupportLevel),
        SqlOuterJoinsSupportLevel::SqlFullOuterJoins as i32
    );
    assert_eq!(sql_info_string(batch, SqlInfo::SqlSchemaTerm), "schema");
    assert_eq!(sql_info_string(batch, SqlInfo::SqlCatalogTerm), "catalog");
    assert!(sql_info_bool(batch, SqlInfo::SqlCatalogAtStart));
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlSupportedUnions),
        sql_info_bitmask([SqlSupportedUnions::SqlUnion as i32, SqlSupportedUnions::SqlUnionAll as i32,])
    );
    assert!(!sql_info_bool(batch, SqlInfo::SqlTransactionsSupported));
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlSupportedResultSetTypes),
        sql_info_bitmask([SqlSupportedResultSetType::SqlResultSetTypeForwardOnly as i32])
    );
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlSupportedConcurrenciesForResultSetForwardOnly),
        sql_info_bitmask([SqlSupportedResultSetConcurrency::SqlResultSetConcurrencyReadOnly as i32])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_catalogs_lists_default_catalog() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.get_catalogs().await.unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let names = collect_string_column(&batches, "catalog_name");
    assert!(names.contains(&"datafusion".to_string()), "expected default catalog, got {names:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_db_schemas_lists_default_schemas() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client
        .get_db_schemas(CommandGetDbSchemas { catalog: Some("datafusion".into()), db_schema_filter_pattern: None })
        .await
        .unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    // `information_schema` is exposed by `SessionState`, not by the catalog
    // provider, so it does not surface here. The default `public` schema does.
    let schemas = collect_string_column(&batches, "db_schema_name");
    assert!(schemas.contains(&"public".to_string()), "missing public: {schemas:?}");
    let catalogs = collect_string_column(&batches, "catalog_name");
    assert!(catalogs.iter().all(|c| c == "datafusion"), "filter leaked: {catalogs:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_tables_distinguishes_views_from_base_tables() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    // Seed both a real base table and a view so we can assert that the
    // metadata layer reports each with the correct Flight SQL table_type.
    let _ = execute_batches(
        &mut client,
        "CREATE TABLE metadata_probe_base AS \
         SELECT * FROM (VALUES (1::INT, 'a'), (2::INT, 'b')) AS t(id, label)",
    )
    .await;
    let _ = execute_batches(&mut client, "CREATE VIEW metadata_probe_view AS SELECT id FROM metadata_probe_base").await;

    let flight_info = client
        .get_tables(CommandGetTables {
            catalog: Some("datafusion".into()),
            db_schema_filter_pattern: Some("public".into()),
            table_name_filter_pattern: Some("metadata_probe%".into()),
            table_types: vec![],
            include_schema: false,
        })
        .await
        .unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let pairs = collect_name_and_type(&batches);
    assert!(
        pairs.contains(&("metadata_probe_base".to_string(), "TABLE".to_string())),
        "missing seeded base table or wrong type: {pairs:?}"
    );
    assert!(
        pairs.contains(&("metadata_probe_view".to_string(), "VIEW".to_string())),
        "missing seeded view or wrong type: {pairs:?}"
    );
    let catalogs = collect_string_column(&batches, "catalog_name");
    assert!(catalogs.iter().all(|c| c == "datafusion"), "catalog filter leaked: {catalogs:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_tables_respects_table_types_filter() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let _ = execute_batches(&mut client, "CREATE TABLE table_types_filter_base AS SELECT 1 AS x").await;
    let _ =
        execute_batches(&mut client, "CREATE VIEW table_types_filter_view AS SELECT x FROM table_types_filter_base")
            .await;

    let flight_info = client
        .get_tables(CommandGetTables {
            catalog: Some("datafusion".into()),
            db_schema_filter_pattern: Some("public".into()),
            table_name_filter_pattern: Some("table_types_filter%".into()),
            table_types: vec!["VIEW".into()],
            include_schema: false,
        })
        .await
        .unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let pairs = collect_name_and_type(&batches);
    assert_eq!(pairs, vec![("table_types_filter_view".to_string(), "VIEW".to_string())]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_tables_with_include_schema_emits_arrow_schema() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let _ = execute_batches(
        &mut client,
        "CREATE TABLE include_schema_probe AS \
         SELECT 1::INT AS id, 'a' AS label",
    )
    .await;

    let flight_info = client
        .get_tables(CommandGetTables {
            catalog: Some("datafusion".into()),
            db_schema_filter_pattern: Some("public".into()),
            table_name_filter_pattern: Some("include_schema_probe".into()),
            table_types: vec![],
            include_schema: true,
        })
        .await
        .unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let pairs = collect_name_and_type(&batches);
    assert_eq!(pairs, vec![("include_schema_probe".to_string(), "TABLE".to_string())]);

    let schemas: Vec<arrow::datatypes::Schema> = batches
        .iter()
        .filter_map(|batch| {
            let index = batch.schema().index_of("table_schema").ok()?;
            let bytes = batch
                .column(index)
                .as_any()
                .downcast_ref::<arrow::array::BinaryArray>()
                .expect("table_schema column must be Binary");
            (0..bytes.len()).map(|i| arrow::ipc::convert::try_schema_from_ipc_buffer(bytes.value(i)).unwrap()).next()
        })
        .collect();
    assert_eq!(schemas.len(), 1, "expected exactly one table_schema row");
    let field_names: Vec<String> = schemas[0].fields().iter().map(|f| f.name().clone()).collect();
    assert_eq!(field_names, vec!["id".to_string(), "label".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_table_types_returns_table_and_view() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.get_table_types().await.unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let types = collect_string_column(&batches, "table_type");
    assert_eq!(types, vec!["TABLE".to_string(), "VIEW".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_round_trip_executes_select() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let mut prepared = client.prepare("select 1 as one".to_string(), None).await.unwrap();
    assert_eq!(prepared.dataset_schema().unwrap().fields().len(), 1, "dataset_schema must reflect projected columns");

    // Execute twice to confirm the prepared handle is not consumed.
    for expected_iteration in 0..2 {
        let flight_info = prepared.execute().await.unwrap();
        let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
        let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches.len(), 1, "iteration {expected_iteration} should return one batch");
        assert_eq!(int_value(batches[0].column(0)), Some(1));
    }
    prepared.close().await.unwrap();

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_update_rejects_select_and_preserves_handle() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let mut prepared = client.prepare("select 1 as one".to_string(), None).await.unwrap();
    let error = prepared.execute_update().await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("statement query endpoint"));
    assert_no_diagnostic_fields(status.message());

    let flight_info = prepared.execute().await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(1));
    prepared.close().await.unwrap();

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_update_rejects_sql_prepare_namespace() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute_update("PREPARE local_stmt AS SELECT 1".to_string(), None).await.unwrap_err();

    server.abort();

    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("SQL PREPARE"));
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_rejects_transaction_id() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let txn_id = prost::bytes::Bytes::from_static(b"fake-txn");
    let error = client.prepare("select 1".to_string(), Some(txn_id)).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::Unimplemented);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "unsupported_operator", "unsupported", "permanent");

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_close_is_idempotent_via_raw_action() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    // Send a Close action for a handle that was never created. The server
    // must treat this as a no-op (Ok) rather than UNIMPLEMENTED or an error.
    use arrow_flight::sql::{ActionClosePreparedStatementRequest, ProstMessageExt};
    use prost::Message;
    let action_body =
        ActionClosePreparedStatementRequest { prepared_statement_handle: prost::bytes::Bytes::from_static(b"ghost") };
    let action = arrow_flight::Action {
        r#type: "ClosePreparedStatement".to_string(),
        body: action_body.as_any().encode_to_vec().into(),
    };
    let mut stream = client.do_action(action).await.unwrap();
    while stream.message().await.unwrap().is_some() {}

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_invalid_sql_fails_at_create() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.prepare("selekt 1".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "sql_parse", "invalid_input", "permanent");
    assert_no_diagnostic_fields(status.message());
    let payload = get_query_stats_raw(&mut client, &query_id_from_status(&status)).await;
    assert!(
        payload.contains(
            "operation=datafusion_nexus.server.flight.do_action_create_prepared_statement.create_logical_plan"
        ),
        "{payload}"
    );

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_prepared_statement_query_stats_remains_resolvable() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let command =
        CommandPreparedStatementQuery { prepared_statement_handle: prost::bytes::Bytes::from_static(b"ghost") };
    let descriptor = arrow_flight::FlightDescriptor::new_cmd(command.as_any().encode_to_vec());
    let status = client.inner_mut().get_flight_info(descriptor).await.unwrap_err();
    assert_eq!(status.code(), Code::NotFound);
    let query_id = query_id_from_status(&status);

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# outcome=failed"), "{payload}");
    assert!(payload.contains("code=object_not_found"), "{payload}");
    assert!(
        payload.contains("operation=datafusion_nexus.server.flight.get_flight_info_prepared_statement.resolve"),
        "{payload}"
    );
    assert!(payload.contains("unknown prepared statement handle"), "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_missing_column_uses_server_error_presentation() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.prepare("select missing from (select 1 as one) as t".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "field_not_found", "invalid_input", "permanent");
    assert!(status.message().contains("missing"), "missing field absent from message: {}", status.message());
    assert!(status.message().contains("one"), "valid field absent from message: {}", status.message());
    assert_no_diagnostic_fields(status.message());

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_primary_keys_returns_empty_with_schema() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let cmd = CommandGetPrimaryKeys { catalog: None, db_schema: None, table: "any_table".to_string() };
    let flight_info = client.get_primary_keys(cmd).await.unwrap();
    let (schema, batches) = collect_metadata_schema_and_batches(&mut client, &flight_info).await;

    server.abort();

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "primary keys must be empty for DataFusion catalog");
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        field_names,
        vec!["catalog_name", "db_schema_name", "table_name", "column_name", "key_name", "key_sequence"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_foreign_keys_rpcs_return_empty_with_shared_schema() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let imported_info = client
        .get_imported_keys(CommandGetImportedKeys { catalog: None, db_schema: None, table: "t".to_string() })
        .await
        .unwrap();
    let exported_info = client
        .get_exported_keys(CommandGetExportedKeys { catalog: None, db_schema: None, table: "t".to_string() })
        .await
        .unwrap();
    let xref_info = client
        .get_cross_reference(CommandGetCrossReference {
            pk_catalog: None,
            pk_db_schema: None,
            pk_table: "p".to_string(),
            fk_catalog: None,
            fk_db_schema: None,
            fk_table: "f".to_string(),
        })
        .await
        .unwrap();

    let (imported_schema, imported) = collect_metadata_schema_and_batches(&mut client, &imported_info).await;
    let (exported_schema, exported) = collect_metadata_schema_and_batches(&mut client, &exported_info).await;
    let (xref_schema, xref) = collect_metadata_schema_and_batches(&mut client, &xref_info).await;

    server.abort();

    let expected_names = vec![
        "pk_catalog_name",
        "pk_db_schema_name",
        "pk_table_name",
        "pk_column_name",
        "fk_catalog_name",
        "fk_db_schema_name",
        "fk_table_name",
        "fk_column_name",
        "key_sequence",
        "fk_key_name",
        "pk_key_name",
        "update_rule",
        "delete_rule",
    ];
    for (label, schema, batches) in [
        ("imported", &imported_schema, &imported),
        ("exported", &exported_schema, &exported),
        ("cross_reference", &xref_schema, &xref),
    ] {
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 0, "{label} keys must be empty");
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, expected_names, "{label} keys schema mismatch");
        let update_rule = schema.field_with_name("update_rule").unwrap();
        assert_eq!(update_rule.data_type(), &arrow::datatypes::DataType::UInt8, "{label} update_rule must be UInt8");
    }
    // Reference the UInt8Array import to keep it warning-free even when batches are empty.
    let _ = UInt8Array::from(Vec::<u8>::new());
}

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
    let ticket = any.unpack::<TicketStatementQuery>().unwrap().expect("statement ticket");
    String::from_utf8(ticket.statement_handle.to_vec()).unwrap()
}

async fn get_query_stats_raw(client: &mut FlightSqlServiceClient<tonic::transport::Channel>, query_id: &str) -> String {
    let action =
        arrow_flight::Action { r#type: "GetQueryStats".to_string(), body: query_id.as_bytes().to_vec().into() };
    let mut stream = client.do_action(action).await.unwrap();
    let result = stream.message().await.unwrap().expect("GetQueryStats action must return one result");
    assert!(stream.message().await.unwrap().is_none(), "GetQueryStats action returns exactly one result");
    String::from_utf8(result.body.to_vec()).unwrap()
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

/// Warm the process-wide CUDA context exactly once. The first CUDA call pays a
/// multi-second `cuInit()` cost; doing it here (every test reserves an address
/// before spawning a server) keeps that cost off each server's connect window,
/// and the `OnceLock` collapses concurrent callers onto a single init.
fn warm_cuda_once() {
    static CUDA_WARM: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        // Best-effort: serve_with_listener re-runs (and surfaces) any error.
        let _ = datafusion_nexus::server::init_cuda_device(0);
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
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![0, 1])), Arc::new(Int64Array::from(vec![1, 2]))],
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
