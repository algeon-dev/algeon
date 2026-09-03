#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_client_can_run_select_one() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    let backend = build_injected_test_backend(&mut config);
    let replacement_config = config.clone();
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
    let delayed_cleanup = cleanup_backend.admit_query().expect("test cleanup owner must be admitted before shutdown");

    drop(client);
    server.abort();
    let error = server.await.expect_err("aborted server task must report cancellation");
    assert!(error.is_cancelled());
    let error = nexus_server::backend_from_server_config(&replacement_config)
        .expect_err("hard abort does not synchronously complete active GPU query cleanup");
    let diagnostic = format!("{error:?}");
    assert!(diagnostic.contains("code: ServiceOverloaded"), "{diagnostic}");
    assert!(diagnostic.contains("operation: \"datafusion_nexus.backend.gpu_core\""), "{diagnostic}");
    assert!(diagnostic.contains("native_operation\", \"nexus_query_engine.gpu.owner.claim"), "{diagnostic}");
    drop(delayed_cleanup);
    tokio::task::spawn_blocking(move || cleanup_backend.close_and_wait(Duration::from_secs(10)))
        .await
        .expect("aborted server cleanup waiter must not panic")
        .expect("aborted server must reach query-service quiescence");
    let replacement = nexus_server::backend_from_server_config(&replacement_config)
        .expect("quiescent aborted server must release its closed QueryService governor claim");
    drop(replacement);

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(1));
}

#[tokio::test(flavor = "current_thread")]
async fn aborting_server_future_never_blocks_current_thread_runtime_drop() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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

    // The aborted build may already hold the device-owner lease; its close
    // and rollback release it asynchronously. Wait for that release instead
    // of assuming it already happened: the race between the first yield and
    // the ownership claim is environment-dependent.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let replacement = loop {
        match nexus_server::backend_from_server_config(&replacement_config) {
            Ok(backend) => break backend,
            Err(error)
                if error.fact_value("native_operation") == Some("nexus_query_engine.gpu.owner.claim")
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(25));
            },
            Err(error) => {
                panic!("nonblocking abort must release a quiescent server governor claim: {error:?}")
            },
        }
    };
    drop(replacement);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_builder_can_run_select_one_and_shutdown_by_token() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
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
async fn removed_unbounded_server_setting_is_rejected_during_config_load() {
    let error = ServerConfig::from_env_get(|name| {
        (name == "NEXUS_SERVER_NATIVE_MEMORY_POLICY").then(|| "unbounded".to_string())
    })
    .unwrap_err();
    assert_eq!(error.code(), datafusion_nexus::error::ErrorCode::InvalidRequest);
    assert_eq!(error.operation(), "datafusion_nexus.server.config");
    assert_eq!(error.fact_value("config_field"), Some("NEXUS_SERVER_NATIVE_MEMORY_POLICY"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_datafusion_baseline_statement_never_enters_gpu_ledger() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    let backend = build_injected_test_backend(&mut config);
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
    // continues to exercise DataFusion-baseline execution, not the GPU ledger.
    let attempts_before = backend.attempt_counters_snapshot().attempts_begun;
    let flight_info = client.execute("select 'ABC' ilike 'a%' as label_match".to_string(), None).await.unwrap();
    assert_eq!(
        backend.attempt_counters_snapshot().attempts_begun,
        attempts_before + 1,
        "GetFlightInfo begins the sole invocation attempt before any GPU admission"
    );
    wait_for_empty_gpu_ledger(&backend).await;
    let snapshot = backend.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 0, "GetFlightInfo must not occupy GPU admission: {snapshot:?}");
    assert_eq!(snapshot.queued_queries, 0, "GetFlightInfo must not queue GPU admission: {snapshot:?}");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    // Drain any async GPU pool cleanup path before reading counters so a pure
    // DataFusion-baseline statement cannot race with unrelated pool bookkeeping.
    wait_for_empty_gpu_ledger(&backend).await;
    let snapshot = backend.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 0, "DoGet DataFusion baseline must not occupy GPU admission: {snapshot:?}");
    assert_eq!(snapshot.queued_queries, 0, "DoGet DataFusion baseline must not queue GPU admission: {snapshot:?}");

    drop(client);
    shutdown.cancel();
    let server_result = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("token shutdown should drain promptly")
        .expect("server task should not panic");

    server_result.expect("token shutdown should stop the server cleanly");
    assert_eq!(batches.len(), 1);
    let matched =
        batches[0].column(0).as_any().downcast_ref::<BooleanArray>().expect("ILIKE result must be BooleanArray");
    assert!(matched.value(0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_builder_shares_injected_backend_with_session_context() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    config.gpu_device_profiles[0] = config.gpu_device_profiles[0].clone().with_max_active_attempts(1);
    config.queue_policy = config.queue_policy.clone().with_max_queue_wait(Duration::from_secs(60));
    let backend = build_injected_test_backend(&mut config);
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
    assert!(server.backend().unwrap().same_backend(&backend));
    let ctx = Arc::clone(server.session_context());
    ctx.register_parquet("shared_edges", parquet_path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let server = tokio::spawn(async move { server.serve().await });
    let mut client = connect_with_retry(addr).await;
    let flight_info =
        client.execute("select src from shared_edges order by src limit 1".to_string(), None).await.unwrap();
    assert_eq!(
        backend.admission_snapshot().unwrap().active_queries,
        0,
        "GetFlightInfo must not acquire GPU admission before DoGet"
    );

    let direct_handle = backend.admit_async().await.expect("direct backend admission should occupy the shared slot");
    assert_eq!(backend.admission_snapshot().unwrap().active_queries, 1);

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

    // Either queued query may win the one shared slot. Await them together so
    // the assertion does not impose an ordering, and allow the configured
    // 60-second admission window when the shared-GPU lane is fully occupied.
    let (flight_batches, session_batches) =
        tokio::time::timeout(Duration::from_secs(60), async { tokio::join!(flight_query, session_query) })
            .await
            .expect("Flight and session queries should finish after direct backend admission releases");
    let flight_batches = flight_batches.expect("Flight query task should not panic");
    let session_batches = session_batches.expect("session query task should not panic").unwrap();

    shutdown.cancel();
    let server_result = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("token shutdown should drain promptly")
        .expect("server task should not panic");

    server_result.expect("token shutdown should stop the server cleanly");
    assert_eq!(flight_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(session_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let close =
        backend.close_and_wait(Duration::from_secs(1)).expect("server shutdown must quiesce the shared backend");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_clone_installed_sessions_share_identity_queue_and_worker_pool_records() {
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.gpu_device_profiles[0] = config.gpu_device_profiles[0].clone().with_max_active_attempts(1);
    config.queue_policy = config
        .queue_policy
        .clone()
        .with_max_queued_attempts(2)
        .with_max_queue_wait(Duration::from_secs(60));
    let backend = nexus_server::backend_from_server_config(&config).unwrap();
    let first_backend = backend.clone();
    let second_backend = backend.clone();
    let first_session = Arc::new(
        nexus_server::build_session_context_with_backend(&config, &first_backend)
            .await
            .expect("first cloned backend must install a session"),
    );
    let second_session = Arc::new(
        nexus_server::build_session_context_with_backend(&config, &second_backend)
            .await
            .expect("second cloned backend must install a session"),
    );

    let identity = backend.identity();
    assert_eq!(first_backend.identity(), identity);
    assert_eq!(second_backend.identity(), identity);
    assert!(first_backend.same_backend(&second_backend));
    assert!(identity.resource_core > 0);
    assert!(identity.observation_core > 0);
    assert!(identity.resource_generation > 0);
    let first_startup_records = first_backend.device_startup_records();
    let second_startup_records = second_backend.device_startup_records();
    assert_eq!(first_startup_records, second_startup_records);
    assert_eq!(first_startup_records.len(), 1);
    for (first, second) in first_startup_records.iter().zip(second_startup_records.iter()) {
        assert!(first.backend_health_generation > 0);
        assert_eq!(first.backend_health_generation, second.backend_health_generation);
        assert!(first.device_health_generation > 0);
        assert_eq!(first.device_health_generation, second.device_health_generation);
        assert!(first.memory_manager_installation.identity.is_some());
        assert_eq!(first.memory_manager_installation.identity, second.memory_manager_installation.identity);
        assert!(first.memory_manager_installation.installation_generation.is_some_and(|generation| generation > 0));
        assert_eq!(
            first.memory_manager_installation.installation_generation,
            second.memory_manager_installation.installation_generation
        );
    }
    assert_eq!(
        first_backend.device_admission_snapshots().unwrap(),
        second_backend.device_admission_snapshots().unwrap()
    );
    assert_eq!(first_backend.device_ledger_reports().unwrap(), second_backend.device_ledger_reports().unwrap());
    assert_eq!(
        first_backend.device_memory_failure_reports().unwrap(),
        second_backend.device_memory_failure_reports().unwrap()
    );
    let first_telemetry = first_backend.device_telemetry_snapshots().unwrap();
    let second_telemetry = second_backend.device_telemetry_snapshots().unwrap();
    assert_eq!(first_telemetry.len(), second_telemetry.len());
    for (first, second) in first_telemetry.iter().zip(&second_telemetry) {
        // Telemetry payloads include independently sampled free-memory and
        // wall-clock values; device identity and provider availability are the
        // stable public projections shared by cloned backend sessions.
        assert_eq!(first.device_ordinal, second.device_ordinal);
        assert_eq!(first.cuda_status, second.cuda_status);
        assert_eq!(first.nvml_status, second.nvml_status);
    }

    let first_worker_pools = first_backend.gpu_execution_pool_reports().unwrap();
    let second_worker_pools = second_backend.gpu_execution_pool_reports().unwrap();
    assert_eq!(first_worker_pools, second_worker_pools);
    assert_eq!(first_worker_pools.len(), 1);
    assert!(first_worker_pools[0].pool_id > 0);
    assert_eq!(first_worker_pools[0].pool_id, second_worker_pools[0].pool_id);
    assert_eq!(first_worker_pools[0].device_ordinal, backend.devices()[0].device_ordinal());
    assert_eq!(first_worker_pools[0].worker_count, 1);

    let dir = tempfile::Builder::new().prefix("datafusion-nexus-cloned-session-backend-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);
    first_session
        .register_parquet("first_edges", parquet_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    second_session
        .register_parquet("second_edges", parquet_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();

    let holder = backend.admit_async().await.expect("direct admission must occupy the shared backend slot");
    assert_eq!(first_backend.admission_snapshot().unwrap().active_queries, 1);
    assert_eq!(second_backend.admission_snapshot().unwrap().active_queries, 1);

    let first_query = tokio::spawn(async move {
        let frame = first_session.sql("select src from first_edges order by src limit 1").await.unwrap();
        frame.collect().await.unwrap()
    });
    let second_query = tokio::spawn(async move {
        let frame = second_session.sql("select src from second_edges order by src limit 1").await.unwrap();
        frame.collect().await.unwrap()
    });
    wait_for_shared_ledger_queue(&backend, 2).await;
    assert_eq!(
        first_backend.admission_snapshot().unwrap(),
        second_backend.admission_snapshot().unwrap(),
        "both installed sessions must observe one query-service queue and ledger"
    );

    holder.finish();
    drop(holder);
    let (first_batches, second_batches) =
        tokio::time::timeout(Duration::from_secs(60), async { tokio::join!(first_query, second_query) })
            .await
            .expect("both cloned-session queries should run after the shared slot is released");
    let first_batches = first_batches.expect("first cloned-session task must not panic");
    let second_batches = second_batches.expect("second cloned-session task must not panic");
    assert_eq!(first_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(second_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    wait_for_empty_gpu_ledger(&backend).await;

    assert_eq!(
        first_backend.gpu_execution_pool_reports().unwrap(),
        second_backend.gpu_execution_pool_reports().unwrap()
    );
    drop(first_backend);
    drop(second_backend);
    let close = backend.close_and_wait(Duration::from_secs(10)).expect("cloned-session backend must close cleanly");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}
