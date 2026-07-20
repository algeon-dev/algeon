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
        .execute("SELECT src FROM stats_fallback_edges WHERE src % dst >= 0 ORDER BY src".to_string(), None)
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
    assert_eq!(tsv_value(&planning_header, &planning_row, "reason_code"), "unsupported_expression", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "unsupported_category"), "expression", "{payload}");
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
