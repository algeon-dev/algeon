use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_explain_gpu_reuses_coverage_schema_and_rejects_other_endpoints() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let native_info = client.execute("EXPLAIN GPU SELECT x FROM (VALUES (1)) AS t(x)".to_string(), None).await.unwrap();
    let (native_schema, native_batches) = execute_flight_info_with_stream_schema(&mut client, native_info).await;
    let native_batch = &native_batches[0];
    let native_query_id = native_batch
        .column(native_schema.index_of("query_id").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let native_row_kind = native_batch
        .column(native_schema.index_of("row_kind").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let native_gpu_path = native_batch
        .column(native_schema.index_of("gpu_path").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(native_query_id.is_valid(0));
    assert!(!native_query_id.value(0).is_empty());
    assert_eq!(native_row_kind.value(0), "summary");
    assert_eq!(native_gpu_path.value(0), "native");
    let native_query_id = native_query_id.value(0).to_string();
    let native_stats = get_query_stats_raw(&mut client, &native_query_id).await;
    assert!(native_stats.contains("# outcome=succeeded"), "{native_stats}");
    assert!(native_stats.contains("# attempt_outcome=succeeded"), "{native_stats}");
    assert_single_sealed_attempt_report(&native_stats);

    // FlightInfo carries the fixed coverage schema without planning the inner
    // statement. A missing relation is represented as terminal coverage only
    // when DoGet performs its single validate_query invocation.
    let deferred_info =
        client.execute("EXPLAIN GPU SELECT * FROM explain_gpu_missing_relation".to_string(), None).await.unwrap();
    let (deferred_schema, deferred_batches) = execute_flight_info_with_stream_schema(&mut client, deferred_info).await;
    let deferred_batch = &deferred_batches[0];
    let deferred_gpu_path = deferred_batch
        .column(deferred_schema.index_of("gpu_path").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(deferred_batch.num_rows(), 1);
    assert_eq!(deferred_gpu_path.value(0), "rejected");
    // The rejected summary carries the code and the message that executing the
    // statement returns.
    let execution_status = unwrap_flight_status(
        client.execute("SELECT * FROM explain_gpu_missing_relation".to_string(), None).await.unwrap_err(),
    );
    assert_eq!(
        coverage_string(deferred_batch, &deferred_schema, "reason_code", 0),
        execution_status.metadata().get(ERROR_CODE_METADATA_KEY).unwrap().to_str().unwrap()
    );
    let deferred_detail: serde_json::Value =
        serde_json::from_str(coverage_string(deferred_batch, &deferred_schema, "detail_json", 0)).unwrap();
    assert_eq!(deferred_detail["detail"], execution_status.message());

    let rows_info = client.execute("EXPLAIN GPU VERBOSE SELECT random() AS r".to_string(), None).await.unwrap();
    let advertised_schema = rows_info.clone().try_decode_schema().unwrap();
    let expected_columns = [
        "query_id",
        "row_kind",
        "gpu_path",
        "gpu_fragments",
        "cpu_boundaries",
        "physical_root",
        "post_native_plan_root",
        "candidate_shape",
        "candidate_family",
        "outcome",
        "reason_code",
        "unsupported_category",
        "remedy_code",
        "remedy_kind",
        "remedy",
        "blocking",
        "detail_json",
        "coverage_json",
    ];
    assert_eq!(advertised_schema.fields().iter().map(|field| field.name()).collect::<Vec<_>>(), expected_columns);
    let (stream_schema, batches) = execute_flight_info_with_stream_schema(&mut client, rows_info).await;
    assert_eq!(stream_schema, advertised_schema);
    let batch = &batches[0];
    let row_kind_index = batch.schema().index_of("row_kind").unwrap();
    let reason_index = batch.schema().index_of("reason_code").unwrap();
    let remedy_code_index = batch.schema().index_of("remedy_code").unwrap();
    let remedy_kind_index = batch.schema().index_of("remedy_kind").unwrap();
    let row_kind = batch.column(row_kind_index).as_any().downcast_ref::<StringArray>().unwrap();
    let reason_code = batch.column(reason_index).as_any().downcast_ref::<StringArray>().unwrap();
    let remedy_code = batch.column(remedy_code_index).as_any().downcast_ref::<StringArray>().unwrap();
    let remedy_kind = batch.column(remedy_kind_index).as_any().downcast_ref::<StringArray>().unwrap();
    let candidate = (0..batch.num_rows())
        .find(|row| {
            row_kind.value(*row) == "candidate"
                && reason_code.is_valid(*row)
                && reason_code.value(*row) == "unsupported_expression"
        })
        .expect("random() must produce an unsupported-expression candidate row");
    assert_eq!(remedy_code.value(candidate), "rewrite_expression");
    assert_eq!(remedy_kind.value(candidate), "user_action");
    let blocking =
        batch.column(batch.schema().index_of("blocking").unwrap()).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(blocking.value(candidate), "scalar function random in `random()`");
    assert!(blocking.is_null(0), "the summary row names no blocking subject");

    // The two SQL surfaces intentionally share validation evidence and the
    // row projection. Their only per-surface value is Flight's query id.
    let udtf_info = client
        .execute("SELECT * FROM algeon_explain_coverage('SELECT random() AS r')".to_string(), None)
        .await
        .unwrap();
    let (udtf_schema, udtf_batches) = execute_flight_info_with_stream_schema(&mut client, udtf_info).await;
    assert_eq!(udtf_schema, advertised_schema);
    assert_eq!(udtf_batches.len(), 1);
    let udtf = &udtf_batches[0];
    let udtf_query_id =
        udtf.column(udtf_schema.index_of("query_id").unwrap()).as_any().downcast_ref::<StringArray>().unwrap();
    assert!((0..udtf.num_rows()).all(|row| udtf_query_id.is_null(row)));
    assert_coverage_rows_equal_except_query_id(udtf, batch);

    let json_info = client.execute("EXPLAIN GPU FORMAT JSON SELECT random() AS r".to_string(), None).await.unwrap();
    assert_eq!(json_info.clone().try_decode_schema().unwrap(), advertised_schema);
    let (json_stream_schema, json_batches) = execute_flight_info_with_stream_schema(&mut client, json_info).await;
    assert_eq!(json_stream_schema, advertised_schema);
    assert_eq!(json_batches.len(), 1);
    assert_eq!(json_batches[0].num_rows(), 1);
    let json_row_kind = json_batches[0]
        .column(json_batches[0].schema().index_of("row_kind").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let coverage_json = json_batches[0]
        .column(json_batches[0].schema().index_of("coverage_json").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(json_row_kind.value(0), "summary");
    let json_gpu_path = json_batches[0]
        .column(json_batches[0].schema().index_of("gpu_path").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(json_gpu_path.value(0), "cpu");
    assert!(coverage_json.is_valid(0));
    let json: serde_json::Value = serde_json::from_str(coverage_json.value(0)).unwrap();
    assert!(json.get("overall").is_none());

    let regular_explain = client.execute("EXPLAIN SELECT 1 AS one".to_string(), None).await.unwrap();
    let regular_schema = regular_explain.try_decode_schema().unwrap();
    assert!(regular_schema.field_with_name("row_kind").is_err(), "regular EXPLAIN must keep its DataFusion schema");

    let prepare_error = client.prepare("EXPLAIN GPU SELECT 1".to_string(), None).await.unwrap_err();
    let prepare_status = unwrap_flight_status(prepare_error);
    assert_eq!(prepare_status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&prepare_status);
    assert_error_metadata(&prepare_status, "invalid_request", "invalid_input", "permanent");
    assert!(prepare_status.message().contains("cannot be prepared"));

    let update_error = client.execute_update("EXPLAIN GPU SELECT 1".to_string(), None).await.unwrap_err();
    let update_status = unwrap_flight_status(update_error);
    assert_eq!(update_status.code(), Code::InvalidArgument);
    assert_query_id_metadata(&update_status);
    assert_error_metadata(&update_status, "invalid_request", "invalid_input", "permanent");
    assert!(update_status.message().contains("statement-query operation"));

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_explain_gpu_reports_native_and_partial_native_local_parquet_paths() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    let backend = build_injected_test_backend(&mut config);
    // Match the certified mixed shape: a CPU ILIKE filter directly over the
    // local-Parquet native leaf. The server's normal multi-partition defaults
    // insert RepartitionExec and correctly make this candidate unsupported.
    let single_partition_config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_joins(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    let state_builder = backend
        .install_on(SessionStateBuilder::new_with_default_features().with_config(single_partition_config))
        .unwrap();
    let state = state_builder.build();
    let shutdown = CancellationToken::new();
    let server = FlightSqlServer::builder()
        .config(config)
        .backend(backend)
        .session_context(Arc::new(SessionContext::new_with_state(state)))
        .listener(listener)
        .shutdown(ServerShutdown::Token(shutdown.clone()))
        .build()
        .await
        .unwrap();
    let server = tokio::spawn(async move { server.serve().await });
    let mut client = connect_with_retry(addr).await;
    let dir = tempfile::Builder::new().prefix("algeon-datafusion-flight-sql-explain-gpu-").tempdir().unwrap();
    let parquet_path = dir.path().join("labels.parquet");
    write_explain_gpu_labels_parquet(&parquet_path);
    client
        .execute_update(
            format!(
                "CREATE EXTERNAL TABLE explain_gpu_labels STORED AS PARQUET LOCATION '{}'",
                parquet_path.to_string_lossy()
            ),
            None,
        )
        .await
        .unwrap();

    let native_info =
        client.execute("EXPLAIN GPU SELECT label FROM explain_gpu_labels".to_string(), None).await.unwrap();
    let (native_schema, native_batches) = execute_flight_info_with_stream_schema(&mut client, native_info).await;
    let native = &native_batches[0];
    let native_summary = coverage_summary_row(native, &native_schema);
    assert_coverage_summary(native, &native_schema, native_summary, "native");
    assert_eq!(coverage_string(native, &native_schema, "outcome", native_summary), "accepted_replaced");
    assert!(coverage_string_array(native, &native_schema, "physical_root").is_valid(native_summary));
    assert!(coverage_string_array(native, &native_schema, "post_native_plan_root").is_valid(native_summary));
    let native_physical_root = coverage_string(native, &native_schema, "physical_root", native_summary);
    let native_post_native_root = coverage_string(native, &native_schema, "post_native_plan_root", native_summary);
    let native_coverage_json: serde_json::Value =
        serde_json::from_str(coverage_string(native, &native_schema, "coverage_json", native_summary)).unwrap();
    assert_eq!(native_coverage_json["input_physical_plan_root"], native_physical_root);
    assert_eq!(native_coverage_json["physical_plan_root"], native_post_native_root);
    let native_selected = coverage_candidate_row(native, &native_schema, "selected");
    assert!(coverage_string_array(native, &native_schema, "candidate_family").is_valid(native_selected));

    let partial_info = client
        .execute("EXPLAIN GPU SELECT label FROM explain_gpu_labels WHERE label ILIKE 'al%'".to_string(), None)
        .await
        .unwrap();
    let (partial_schema, partial_batches) = execute_flight_info_with_stream_schema(&mut client, partial_info).await;
    let partial = &partial_batches[0];
    let partial_summary = coverage_summary_row(partial, &partial_schema);
    assert_coverage_summary(partial, &partial_schema, partial_summary, "partial_native");
    assert_eq!(coverage_string(partial, &partial_schema, "outcome", partial_summary), "accepted_replaced");
    let cpu_boundaries = partial
        .column(partial_schema.index_of("cpu_boundaries").unwrap())
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert!(cpu_boundaries.is_valid(partial_summary));
    assert!(cpu_boundaries.value(partial_summary) > 0, "partial summary={partial:?}");

    // The mixed-boundary certificate clears the duplicated Parquet pruning hint only
    // after keeping the exact CPU ILIKE filter. Execute the query as a
    // semantic regression check, not merely a coverage-shape assertion.
    let result_batches =
        execute_batches(&mut client, "SELECT label FROM explain_gpu_labels WHERE label ILIKE 'al%'").await;
    let mut labels = result_batches.iter().flat_map(|batch| utf8_values(batch.column(0))).collect::<Vec<_>>();
    labels.sort_unstable();
    assert_eq!(labels, ["ALMANAC", "Alpha"]);

    drop(client);
    shutdown.cancel();
    server.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_explain_gpu_cancel_before_do_get_records_cancelled_stats() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("EXPLAIN GPU SELECT 1".to_string(), None).await.unwrap();
    let query_id = query_id_from_statement_flight_info(&flight_info);
    let result = cancel_query_raw(&mut client, &flight_info).await;
    assert_eq!(result.result, CANCEL_RESULT_CANCELLING);

    let payload = get_query_stats_raw(&mut client, &query_id).await;
    server.abort();

    assert!(payload.contains("# outcome=cancelled"), "{payload}");
    assert!(payload.contains("# delivery_outcome=failed"), "{payload}");
    assert!(payload.contains("# sealed=true"), "{payload}");
    assert!(payload.contains("# attempt_outcome=cancelled"), "{payload}");
    assert!(payload.contains("code=cancelled"), "{payload}");
    assert!(payload.contains("operation=algeon_datafusion.server.flight.cancel_query"), "{payload}");
    assert_single_sealed_attempt_report(&payload);
}

fn write_explain_gpu_labels_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("label", DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vec!["Alpha", "beta", "ALMANAC"]))])
            .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn coverage_summary_row(batch: &RecordBatch, schema: &Schema) -> usize {
    (0..batch.num_rows())
        .find(|row| coverage_string(batch, schema, "row_kind", *row) == "summary")
        .expect("coverage output must have one summary row")
}

pub(super) fn coverage_candidate_row(batch: &RecordBatch, schema: &Schema, expected_outcome: &str) -> usize {
    (0..batch.num_rows())
        .find(|row| {
            coverage_string(batch, schema, "row_kind", *row) == "candidate"
                && coverage_string(batch, schema, "outcome", *row) == expected_outcome
        })
        .unwrap_or_else(|| panic!("coverage output must contain a {expected_outcome} candidate row: {batch:?}"))
}

fn assert_coverage_summary(batch: &RecordBatch, schema: &Schema, summary: usize, expected_gpu_path: &str) {
    assert_eq!(coverage_string(batch, schema, "row_kind", summary), "summary");
    assert_eq!(
        coverage_string(batch, schema, "gpu_path", summary),
        expected_gpu_path,
        "coverage_json={}",
        coverage_string(batch, schema, "coverage_json", summary)
    );
}

fn assert_coverage_rows_equal_except_query_id(expected: &RecordBatch, actual: &RecordBatch) {
    assert_eq!(expected.schema(), actual.schema());
    assert_eq!(expected.num_rows(), actual.num_rows());
    for column in 1..expected.num_columns() {
        assert_eq!(
            expected.column(column).to_data(),
            actual.column(column).to_data(),
            "coverage column {} differs between UDTF and EXPLAIN GPU",
            expected.schema().field(column).name()
        );
    }
}

fn utf8_values(array: &ArrayRef) -> Vec<String> {
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return values.iter().map(|value| value.expect("fixture labels are non-null").to_string()).collect();
    }
    if let Some(values) = array.as_any().downcast_ref::<StringViewArray>() {
        return values.iter().map(|value| value.expect("fixture labels are non-null").to_string()).collect();
    }
    panic!("mixed-boundary result label must be a UTF-8 Arrow array, got {:?}", array.data_type());
}

pub(super) fn coverage_string<'a>(batch: &'a RecordBatch, schema: &Schema, column: &str, row: usize) -> &'a str {
    let values = coverage_string_array(batch, schema, column);
    assert!(values.is_valid(row), "{column} must be non-null at row {row}: {batch:?}");
    values.value(row)
}

fn coverage_string_array<'a>(batch: &'a RecordBatch, schema: &Schema, column: &str) -> &'a StringArray {
    batch
        .column(schema.index_of(column).unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| panic!("{column} must be Utf8"))
}
