#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_client_can_run_select_one() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    let replacement_config = config.clone();
    let backend = NexusGpuBackend::from_server_config(&config).unwrap().expect("default server config is bounded");
    let cleanup_backend = backend.clone();

    let server = FlightSqlServer::builder().config(config).backend(backend).listener(listener).build().await.unwrap();
    let server = tokio::spawn(async move { server.serve().await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select 1 as one".to_string(), None).await.unwrap();
    let schema = flight_info.clone().try_decode_schema().unwrap();
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "one");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    let delayed_cleanup = cleanup_backend
        .query_service()
        .admit_query("default")
        .expect("test cleanup owner must be admitted before shutdown");

    drop(client);
    server.abort();
    let error = server.await.expect_err("aborted server task must report cancellation");
    assert!(error.is_cancelled());
    let error = NexusGpuBackend::from_server_config(&replacement_config)
        .expect_err("hard abort does not synchronously complete active GPU query cleanup");
    let diagnostic = format!("{error:?}");
    assert!(diagnostic.contains("code: InvalidExecutionPlan"), "{diagnostic}");
    assert!(diagnostic.contains("operation: \"datafusion_nexus.backend.query_service\""), "{diagnostic}");
    assert!(
        diagnostic.contains("native_operation\", \"nexus_query_engine.query_service.tenant.validate"),
        "{diagnostic}"
    );
    drop(delayed_cleanup);
    tokio::task::spawn_blocking(move || {
        cleanup_backend.query_service().close_and_wait_for_quiescence(Duration::from_secs(10))
    })
    .await
    .expect("aborted server cleanup waiter must not panic")
    .expect("aborted server must reach query-service quiescence");
    let replacement = NexusGpuBackend::from_server_config(&replacement_config)
        .expect("quiescent aborted server must release its closed QueryService governor claim")
        .expect("default server config is bounded");
    drop(replacement);

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(1));
}

#[tokio::test(flavor = "current_thread")]
async fn aborting_server_future_never_blocks_current_thread_runtime_drop() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    let replacement_config = config.clone();
    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    tokio::task::yield_now().await;

    server.abort();
    let error = tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("server task Drop must not block the current-thread runtime")
        .expect_err("aborted server task must report cancellation");
    assert!(error.is_cancelled());

    let replacement = NexusGpuBackend::from_server_config(&replacement_config)
        .expect("nonblocking abort must release a quiescent server governor claim")
        .expect("default server config is bounded");
    drop(replacement);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_builder_can_run_select_one_and_shutdown_by_token() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.admission_policy = config.admission_policy.with_slot_guarantee(SlotGuarantee::Optimistic);
    let shutdown = CancellationToken::new();
    let server = FlightSqlServer::builder()
        .config(config)
        .listener(listener)
        .shutdown(ServerShutdown::Token(shutdown.clone()))
        .build()
        .await
        .unwrap();
    assert_eq!(server.local_addr(), Some(addr));

    let server = tokio::spawn(async move { server.serve().await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select 1 as one".to_string(), None).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    drop(client);
    shutdown.cancel();
    let server_result = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("token shutdown should drain promptly")
        .expect("server task should not panic");

    server_result.expect("token shutdown should stop the server cleanly");
    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unbounded_local_parquet_server_shutdown_allows_fresh_generation() {
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-unbounded-parquet-shutdown-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    for generation in 0..2 {
        let (listener, addr) = free_local_addr();
        let mut config = ServerConfig::from_env_get(|name| {
            (name == "NEXUS_SERVER_NATIVE_MEMORY_POLICY").then(|| "unbounded".to_string())
        })
        .unwrap();
        config.bind_addr = addr;
        let shutdown = CancellationToken::new();
        let server = FlightSqlServer::builder()
            .config(config)
            .listener(listener)
            .shutdown(ServerShutdown::Token(shutdown.clone()))
            .build()
            .await
            .unwrap();
        assert!(server.backend().is_none(), "generation {generation} must use semaphore/unbounded admission");
        server
            .session_context()
            .register_parquet("shutdown_edges", parquet_path.to_str().unwrap(), ParquetReadOptions::default())
            .await
            .unwrap();

        let server = tokio::spawn(async move { server.serve().await });
        let mut client = connect_with_retry(addr).await;
        let batches =
            execute_batches(&mut client, "SELECT SUM(src) + SUM(dst) AS edge_total FROM shutdown_edges").await;
        assert_eq!(batches.len(), 1);
        assert_eq!(int_value(batches[0].column(0)), Some(4));

        drop(client);
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .unwrap_or_else(|_| panic!("generation {generation} shutdown timed out"))
            .unwrap_or_else(|error| panic!("generation {generation} server task failed: {error}"))
            .unwrap_or_else(|error| panic!("generation {generation} shutdown failed: {error}"));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_cpu_fallback_statement_never_enters_gpu_ledger() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    let backend = NexusGpuBackend::from_server_config(&config).unwrap().expect("default server config is bounded");
    let shutdown = CancellationToken::new();
    let server = FlightSqlServer::builder()
        .config(config)
        .backend(backend.clone())
        .listener(listener)
        .shutdown(ServerShutdown::Token(shutdown.clone()))
        .build()
        .await
        .unwrap();

    let server = tokio::spawn(async move { server.serve().await });
    let mut client = connect_with_retry(addr).await;

    // Case-insensitive LIKE remains a structured native expression rejection.
    // Keep this independent of regexp_replace lowering coverage so the test
    // continues to exercise host-side CPU fallback admission, not the GPU ledger.
    let flight_info = client.execute("select 'ABC' ilike 'a%' as label_match".to_string(), None).await.unwrap();
    wait_for_empty_gpu_ledger(&backend).await;
    let snapshot = backend.query_service().admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 0, "GetFlightInfo must not occupy GPU admission: {snapshot:?}");
    assert_eq!(snapshot.queued_queries, 0, "GetFlightInfo must not queue GPU admission: {snapshot:?}");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    // Drain any async GPU pool cleanup path before reading counters so a pure
    // CPU-fallback statement cannot race with unrelated pool bookkeeping.
    wait_for_empty_gpu_ledger(&backend).await;
    let snapshot = backend.query_service().admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 0, "DoGet CPU fallback must not occupy GPU admission: {snapshot:?}");
    assert_eq!(snapshot.queued_queries, 0, "DoGet CPU fallback must not queue GPU admission: {snapshot:?}");

    drop(client);
    shutdown.cancel();
    let server_result = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("token shutdown should drain promptly")
        .expect("server task should not panic");

    server_result.expect("token shutdown should stop the server cleanly");
    assert_eq!(batches.len(), 1);
    let matched = batches[0].column(0).as_any().downcast_ref::<BooleanArray>().expect("ILIKE result must be BooleanArray");
    assert!(matched.value(0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_builder_shares_injected_backend_with_session_context() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.max_in_flight = 1;
    config.admission_policy =
        config.admission_policy.with_max_concurrent_queries(1).with_slot_guarantee(SlotGuarantee::Optimistic);
    let backend = NexusGpuBackend::from_server_config(&config).unwrap().expect("default server config is bounded");
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-shared-backend-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let shutdown = CancellationToken::new();
    let server = FlightSqlServer::builder()
        .config(config)
        .backend(backend.clone())
        .listener(listener)
        .shutdown(ServerShutdown::Token(shutdown.clone()))
        .build()
        .await
        .unwrap();
    assert!(server.backend().unwrap().query_service().same_service(backend.query_service()));
    let ctx = Arc::clone(server.session_context());
    ctx.register_parquet("shared_edges", parquet_path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let server = tokio::spawn(async move { server.serve().await });
    let mut client = connect_with_retry(addr).await;
    let flight_info =
        client.execute("select src from shared_edges order by src limit 1".to_string(), None).await.unwrap();
    assert_eq!(
        backend.query_service().admission_snapshot().unwrap().active_queries,
        0,
        "GetFlightInfo must not acquire GPU admission before DoGet"
    );

    let direct_handle = backend
        .query_service()
        .admit_async(backend.tenant_id().clone())
        .await
        .expect("direct backend admission should occupy the shared slot");
    assert_eq!(backend.query_service().admission_snapshot().unwrap().active_queries, 1);

    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let flight_query =
        tokio::spawn(async move { client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap() });

    let session_ctx = Arc::clone(&ctx);
    let session_query = tokio::spawn(async move {
        let df = session_ctx.sql("select src from shared_edges order by src limit 1").await.unwrap();
        df.collect().await
    });
    wait_for_shared_ledger_queue(&backend, 2).await;
    direct_handle.finish();
    drop(direct_handle);

    let flight_batches = tokio::time::timeout(Duration::from_secs(10), flight_query)
        .await
        .expect("Flight query should finish after direct backend admission releases")
        .expect("Flight query task should not panic");
    let session_batches = tokio::time::timeout(Duration::from_secs(10), session_query)
        .await
        .expect("session query should finish after direct backend admission releases")
        .expect("session query task should not panic")
        .unwrap();

    shutdown.cancel();
    let server_result = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("token shutdown should drain promptly")
        .expect("server task should not panic");

    server_result.expect("token shutdown should stop the server cleanly");
    assert_eq!(flight_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(session_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let snapshot = backend.query_service().admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 0);
    assert_eq!(snapshot.queued_queries, 0);
}
