#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_update_invalid_sql_uses_server_error_presentation() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let txn_id = Bytes::from_static(b"fake-txn");
    let error = client
        .execute_update("CREATE VIEW tx_update_probe AS SELECT 1 AS x".to_string(), Some(txn_id))
        .await
        .unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::Unimplemented);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "unsupported_operator", "unsupported", "permanent");
    assert!(status.message().contains("transactions are not supported"));
    assert_no_diagnostic_fields(status.message());

    let query_id = query_id_from_status(&status);
    let payload = get_query_stats_raw(&mut client, &query_id).await;
    assert!(payload.contains("# outcome=failed"), "{payload}");
    assert!(payload.contains("# sealed=true"), "{payload}");
    assert!(
        !payload.contains("# attempt_id="),
        "a DoPut preflight rejection has no attempt owner and must not retain an attempt-owned coordinator record: {payload}"
    );

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_update_rejects_select_without_side_effect_path() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
async fn unknown_table_errors_remain_generic_planning_without_message_parsing() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("select * from does_not_exist".to_string(), None).await.unwrap_err();

    server.abort();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::InvalidArgument);
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!query_id.is_empty());
    assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
    assert!(status.message().contains("does_not_exist"));
    assert_no_diagnostic_fields(status.message());
}

#[cfg(not(feature = "cugraph"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn feature_off_cugraph_ticket_returns_structured_capability_error() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config_with_env(std::collections::BTreeMap::from([(
        algeon_datafusion::server::ALGEON_SERVER_CUGRAPH_ENABLED_ENV,
        "true".to_string(),
    )]));
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    execute_batches(
        &mut client,
        "CREATE VIEW feature_off_cugraph_edges AS \
         SELECT * FROM (VALUES (0::BIGINT, 1::BIGINT), (1::BIGINT, 2::BIGINT)) AS t(src, dst)",
    )
    .await;

    let flight_info = client
        .execute("SELECT * FROM cugraph_pagerank('feature_off_cugraph_edges')".to_string(), None)
        .await
        .expect("the enabled feature-off frontend must issue a statement ticket");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let error = client
        .do_get(ticket)
        .await
        .expect("ticket execution must begin before the disabled executor rejects it")
        .try_collect::<Vec<_>>()
        .await
        .expect_err("feature-off cuGraph execution must reject through the Flight result stream");

    server.abort();

    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::Unimplemented);
    assert_query_id_metadata(&status);
    assert_error_metadata(&status, "unsupported_operator", "unsupported", "permanent");
    assert_eq!(
        status.metadata().get(ERROR_OPERATION_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some("algeon_datafusion.cugraph_sql.execute")
    );
    assert!(status.message().contains("cuGraph SQL execution requires the cugraph Cargo feature"));
    assert_no_diagnostic_fields(status.message());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_column_errors_are_invalid_input_with_structured_code() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
