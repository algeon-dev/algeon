use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unconsumed_statement_ticket_does_not_block_get_flight_info() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    config.statement_ticket_ttl = Duration::from_secs(30);

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut first_client = connect_with_retry(addr).await;
    let mut second_client = connect_with_retry(addr).await;

    let first_info = first_client.execute("select 1 as one".to_string(), None).await.unwrap();
    let first_ticket = first_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();

    let mut second = tokio::spawn(async move { second_client.execute("select 2 as two".to_string(), None).await });
    let second_info = timeout(Duration::from_secs(5), &mut second)
        .await
        .expect("second GetFlightInfo must not wait on an unconsumed ticket")
        .unwrap()
        .unwrap();

    let batches = first_client.do_get(first_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(int_value(batches[0].column(0)), Some(1));

    let second_ticket = second_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = first_client.do_get(second_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();

    server.abort();

    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abandoned_statement_ticket_expires_and_invalidates_do_get() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    config.statement_ticket_ttl = Duration::from_millis(50);

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut first_client = connect_with_retry(addr).await;
    let mut second_client = connect_with_retry(addr).await;

    let first_info = first_client.execute("select 1 as one".to_string(), None).await.unwrap();
    let first_ticket = first_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();

    let second_info = timeout(Duration::from_secs(5), second_client.execute("select 2 as two".to_string(), None))
        .await
        .expect("second GetFlightInfo should proceed while the first ticket is unconsumed")
        .unwrap();
    let second_ticket = second_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = second_client.do_get(second_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(int_value(batches[0].column(0)), Some(2));

    sleep(Duration::from_millis(75)).await;
    let error = first_client.do_get(first_ticket).await.unwrap_err();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_error_metadata(&status, "invalid_request", "invalid_input", "permanent");
    assert!(status.message().contains("expired"));
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!status.message().contains(query_id), "query id must stay in metadata, not message");
    let payload = get_query_stats_raw(&mut first_client, query_id).await;
    assert_single_sealed_attempt_report(&payload);

    server.abort();
}
