use super::*;

#[test]
fn whole_source_cache_window_output_pins_alias_until_gpu_result_drop() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-window-pin");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("rn", DataType::UInt64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-window-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let window = builder.add_window(
        source,
        WindowSpec::new(
            vec![Expr::column("key")],
            vec![SortExpr::new(Expr::column("payload"), false, false)],
            vec![WindowFunction::row_number("rn")],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_gpu_resident(window);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_cache_misses, 1);
    assert_eq!(result.metrics().source_cache_inserts, 1);
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("gpu sink should retain the window output");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 3);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    assert!(
        governor
            .try_request(GrantRequest::new(
                AdmissionRequestId::new(301),
                governor.capacity_bytes(),
                governor.capacity_bytes()
            ))
            .unwrap()
            .is_none(),
        "pressure admission must not evict source-cache entry while window GPU output aliases it"
    );

    drop(result);
    let grant = governor
        .try_request(GrantRequest::new(
            AdmissionRequestId::new(302),
            governor.capacity_bytes(),
            governor.capacity_bytes(),
        ))
        .unwrap()
        .expect("unpinned window source-cache entry should be evicted under pressure");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}
