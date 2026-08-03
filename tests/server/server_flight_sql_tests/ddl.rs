#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_view_statement_executes_once_and_persists_across_statements() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    assert!(payload.contains("fact.requested_table_type=TABLE"), "{payload}");
    assert!(payload.contains("fact.actual_table_type=VIEW"), "{payload}");
    assert!(!payload.contains("fact.object_name="), "raw object name must not be retained: {payload}");

    let error = client.execute_update("DROP TABLE IF EXISTS drop_view_probe".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("DROP VIEW"));
    assert_no_diagnostic_fields(status.message());
    let payload = get_query_stats_raw(&mut client, &query_id_from_status(&status)).await;
    assert!(payload.contains("operation=datafusion_nexus.server.flight.do_put_statement_update.sql"), "{payload}");
    assert!(payload.contains("fact.requested_table_type=TABLE"), "{payload}");
    assert!(payload.contains("fact.actual_table_type=VIEW"), "{payload}");
    assert!(!payload.contains("fact.object_name="), "raw object name must not be retained: {payload}");

    let _ = execute_batches(&mut client, "DROP VIEW drop_view_probe").await;
    let error = client.execute("SELECT * FROM drop_view_probe".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);

    server.abort();

    assert_eq!(status.code(), Code::InvalidArgument);
    assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_view_on_table_reports_object_type_mismatch() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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

    assert_eq!(status.code(), Code::InvalidArgument);
    assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_update_create_external_table_executes_and_persists() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let ddl_batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    let ddl_rows: usize = ddl_batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(ddl_rows, 0, "DDL executed through the query channel returns no rows");
    let payload = get_query_stats_raw(&mut client, &query_id).await;
    assert_single_sealed_attempt_report(&payload);

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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
