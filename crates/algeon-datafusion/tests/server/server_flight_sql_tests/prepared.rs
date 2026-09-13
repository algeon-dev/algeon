#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_round_trip_executes_select() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let txn_id = Bytes::from_static(b"fake-txn");
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    // Send a Close action for a handle that was never created. The server
    // must treat this as a no-op (Ok) rather than UNIMPLEMENTED or an error.
    use arrow_flight::sql::{ActionClosePreparedStatementRequest, ProstMessageExt};
    use prost::Message;
    let action_body =
        ActionClosePreparedStatementRequest { prepared_statement_handle: Bytes::from_static(b"ghost") };
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
            "operation=algeon_datafusion.server.flight.do_action_create_prepared_statement.create_logical_plan"
        ),
        "{payload}"
    );

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_prepared_statement_query_stats_remains_resolvable() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let command =
        CommandPreparedStatementQuery { prepared_statement_handle: Bytes::from_static(b"ghost") };
    let descriptor = arrow_flight::FlightDescriptor::new_cmd(command.as_any().encode_to_vec());
    let status = client.inner_mut().get_flight_info(descriptor).await.unwrap_err();
    assert_eq!(status.code(), Code::NotFound);
    let query_id = query_id_from_status(&status);

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# outcome=failed"), "{payload}");
    assert!(payload.contains("code=object_not_found"), "{payload}");
    assert!(
        payload.contains("operation=algeon_datafusion.server.flight.get_flight_info_prepared_statement.resolve"),
        "{payload}"
    );
    assert!(payload.contains("unknown prepared statement handle"), "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_missing_column_uses_server_error_presentation() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
