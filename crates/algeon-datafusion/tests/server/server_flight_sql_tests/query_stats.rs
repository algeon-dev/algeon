#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_returns_completed_statement_observability() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select x as one from (values (1)) as t(x)".to_string(), None).await.unwrap();
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(int_value(batches[0].column(0)), Some(1));

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# algeon_datafusion_query_stats=1"), "{payload}");
    assert!(payload.contains(&format!("# query_id={query_id}")), "{payload}");
    assert!(payload.contains("# outcome=succeeded"), "{payload}");
    assert!(payload.contains("[physical_plan]"), "{payload}");
    assert!(payload.contains("# attempt_id=attempt-"), "{payload}");
    assert!(payload.contains("# attempt_disposition=native"), "{payload}");
    assert!(payload.contains("# attempt_outcome=succeeded"), "{payload}");
    assert_single_sealed_attempt_report(&payload);

    let (header, row) = query_stats_tsv_single_row(&payload, "execution");
    assert_eq!(tsv_value(&header, &row, "runtime_outcome"), "succeeded", "{payload}");
    assert_eq!(tsv_value(&header, &row, "output_batches"), "1", "{payload}");
    assert_eq!(tsv_value(&header, &row, "output_rows"), "1", "{payload}");
    assert!(tsv_value(&header, &row, "output_bytes").parse::<usize>().unwrap() > 0, "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_returns_structured_error_after_planning_failure() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
        payload.contains("operation=algeon_datafusion.server.flight.get_flight_info_statement.create_logical_plan"),
        "{payload}"
    );
    assert!(payload.contains("missing"), "{payload}");
    assert!(payload.contains("one"), "{payload}");
    assert_single_sealed_attempt_report(&payload);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn strict_flight_reports_unsupported_list_column_in_status_metadata_and_stats() {
    let (listener, addr) = free_local_addr();
    let env = std::collections::BTreeMap::from([(
        ALGEON_SERVER_FINAL_PLAN_REQUIREMENT_ENV,
        "no_datafusion_cpu".to_string(),
    )]);
    let mut config = test_server_config_with_env(env);
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("algeon-datafusion-flight-unsupported-list-").tempdir().unwrap();
    let path = dir.path().join("unsupported.parquet");
    write_unsupported_decimal_list_parquet(&path);
    execute_batches(
        &mut client,
        &format!("CREATE EXTERNAL TABLE unsupported_lists STORED AS PARQUET LOCATION '{}'", path.to_string_lossy()),
    )
    .await;

    let flight_info = client.execute("SELECT * FROM unsupported_lists".to_string(), None).await.unwrap();
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let status = unwrap_flight_status(client.do_get(ticket).await.unwrap_err());
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(
        status.message(),
        "GPU execution does not support projected column `unsupported_values` with Arrow type `List(Decimal128(10, 2))`"
    );
    assert_eq!(
        status.metadata().get(UNSUPPORTED_COLUMN_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some("unsupported_values")
    );
    assert_eq!(
        status.metadata().get(UNSUPPORTED_COLUMN_TYPE_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some("List(Decimal128(10, 2))")
    );
    assert_eq!(
        status.metadata().get(UNSUPPORTED_COLUMN_ROLE_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some("projected")
    );
    assert_eq!(
        status.metadata().get(REASON_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some("unsupported_type")
    );
    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("[structured_error]"), "{payload}");
    assert!(payload.contains("code=final_plan_requirement_unsatisfied\n"), "{payload}");
    assert!(payload.contains("fact.reason=unsupported_type\n"), "{payload}");
    assert!(payload.contains("fact.unsupported_column=unsupported_values\n"), "{payload}");
    assert!(payload.contains("fact.unsupported_column_type=List(Decimal128(10, 2))\n"), "{payload}");
    assert!(payload.contains("fact.unsupported_column_role=projected\n"), "{payload}");
    assert!(!payload.contains(path.to_string_lossy().as_ref()), "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn strict_flight_executes_basic_list_limit_natively() {
    let (listener, addr) = free_local_addr();
    let env = std::collections::BTreeMap::from([(
        ALGEON_SERVER_FINAL_PLAN_REQUIREMENT_ENV,
        "no_datafusion_cpu".to_string(),
    )]);
    let mut config = test_server_config_with_env(env);
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("algeon-datafusion-flight-basic-list-").tempdir().unwrap();
    let path = dir.path().join("vertexinfo.parquet");
    write_basic_list_parquet(&path);
    execute_batches(
        &mut client,
        &format!("CREATE EXTERNAL TABLE vertex_info_lists STORED AS PARQUET LOCATION '{}'", path.to_string_lossy()),
    )
    .await;

    let explain =
        client.execute("EXPLAIN GPU SELECT * FROM vertex_info_lists LIMIT 10".to_string(), None).await.unwrap();
    let (explain_schema, explain_batches) = execute_flight_info_with_stream_schema(&mut client, explain).await;
    let explain = &explain_batches[0];
    let summary = (0..explain.num_rows())
        .find(|row| {
            explain
                .column(explain_schema.index_of("row_kind").unwrap())
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(*row)
                == "summary"
        })
        .unwrap();
    let gpu_path =
        explain.column(explain_schema.index_of("gpu_path").unwrap()).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(gpu_path.value(summary), "native");
    let selected = coverage_candidate_row(explain, &explain_schema, "selected");
    assert!(coverage_string(explain, &explain_schema, "candidate_shape", selected).contains("LocalParquetScan"));

    let flight_info = client.execute("SELECT * FROM vertex_info_lists LIMIT 10".to_string(), None).await.unwrap();
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let (schema, batches) = execute_flight_info_with_stream_schema(&mut client, flight_info).await;
    assert_eq!(
        schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        ["vertexId", "vertexAttributes__tag", "vertexAttributes__label", "vertexAttributes__rank",]
    );
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 10);
    let batch = batches.iter().find(|batch| batch.num_rows() > 0).unwrap();
    for index in [1, 2] {
        let lists = batch.column(index).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(lists.values().len(), 0);
        assert_eq!(lists.null_count(), batch.num_rows());
    }
    let ports = batch.column(3).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(ports.len(), batch.num_rows());
    assert_eq!(
        ports.values().as_any().downcast_ref::<Int64Array>().unwrap().values().as_ref(),
        &(100..110).collect::<Vec<_>>()
    );

    let payload = get_query_stats_raw(&mut client, &query_id).await;
    server.abort();

    assert!(payload.contains("# outcome=succeeded"), "{payload}");
    assert!(payload.contains("# attempt_disposition=native"), "{payload}");
    assert!(payload.contains("AlgeonNativeExec"), "{payload}");
    let (planning_header, planning_row) = query_stats_tsv_single_row(&payload, "planning");
    assert_eq!(tsv_value(&planning_header, &planning_row, "native_plan_fragment_count"), "1", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "host_datafusion_boundary_count"), "0", "{payload}");
    // Runtime columns project the metrics sealed with the attempt report, not
    // the wrapper metrics that existed when planning was captured.
    let (header, row) = query_stats_tsv_single_row(&payload, "execution");
    assert_eq!(tsv_value(&header, &row, "runtime_outcome"), "succeeded", "{payload}");
    assert_eq!(tsv_value(&header, &row, "device_ordinal"), "0", "{payload}");
    assert!(tsv_value(&header, &row, "query_grant_bytes").parse::<u64>().unwrap() > 0, "{payload}");
    assert_eq!(tsv_value(&header, &row, "device_to_host_materialized_rows"), "10", "{payload}");
    assert!(tsv_value(&header, &row, "device_to_host_materialized_bytes").parse::<u64>().unwrap() > 0, "{payload}");
    assert_eq!(tsv_value(&header, &row, "parquet_files"), "1", "{payload}");
    assert!(tsv_value(&header, &row, "max_task_observed_peak_bytes").parse::<u64>().is_ok(), "{payload}");
    assert!(payload.contains("# metric_projection_truncated=false\n"), "{payload}");
    assert_single_sealed_attempt_report(&payload);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_surfaces_complete_cpu_baseline_not_supported() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("algeon-datafusion-flight-sql-stats-fallback-").tempdir().unwrap();
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
    assert_eq!(tsv_value(&planning_header, &planning_row, "outcome"), "not_supported", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "reason_code"), "unsupported_expression", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "unsupported_category"), "expression", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "post_native_execution_path"), "datafusion", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "native_plan_fragment_count"), "0", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "host_datafusion_boundary_count"), "0", "{payload}");
    assert_eq!(tsv_value(&planning_header, &planning_row, "rejected_before_execution"), "false", "{payload}");

    let (execution_header, execution_row) = query_stats_tsv_single_row(&payload, "execution");
    assert_eq!(tsv_value(&execution_header, &execution_row, "planning_outcome"), "not_supported", "{payload}");
    assert_eq!(tsv_value(&execution_header, &execution_row, "post_native_execution_path"), "datafusion", "{payload}");
    assert_eq!(tsv_value(&execution_header, &execution_row, "native_runtime_started"), "false", "{payload}");
    assert_eq!(tsv_value(&execution_header, &execution_row, "runtime_outcome"), "not_started", "{payload}");
    assert_eq!(tsv_value(&execution_header, &execution_row, "output_rows"), "not_run", "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_unknown_query_returns_not_found() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let error = client
        .execute("select 1 as one".to_string(), Some(Bytes::from_static(b"txn")))
        .await
        .unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::Unimplemented);
    let query_id = query_id_from_status(&status);

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# outcome=failed"), "{payload}");
    assert!(payload.contains("code=unsupported_operator"), "{payload}");
    assert!(payload.contains("operation=algeon_datafusion.server.flight.transaction"), "{payload}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_query_stats_returns_cancelled_for_unconsumed_ticket() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select 1 as one".to_string(), None).await.unwrap();
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let result = cancel_query_raw(&mut client, &flight_info).await;
    assert_eq!(result.result, CANCEL_RESULT_CANCELLING);

    let payload = get_query_stats_raw(&mut client, &query_id).await;

    server.abort();

    assert!(payload.contains("# outcome=cancelled"), "{payload}");
    assert!(payload.contains("# delivery_outcome=failed"), "{payload}");
    assert!(payload.contains("# sealed=true"), "{payload}");
    assert!(payload.contains("[structured_error]"), "{payload}");
    assert!(payload.contains("code=cancelled"), "{payload}");
    assert!(payload.contains("operation=algeon_datafusion.server.flight.cancel_query"), "{payload}");
    assert_single_sealed_attempt_report(&payload);
}
