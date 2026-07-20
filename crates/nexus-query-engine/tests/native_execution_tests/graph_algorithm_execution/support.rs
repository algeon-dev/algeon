static GRAPH_ADAPTIVE_RETRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_query_service_exact_runtime_warm_pricing(mode: NativePipelineExecutionMode) {
    let _serial = GRAPH_ADAPTIVE_RETRY_TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let tenant_id = TenantId::new("graph-warm-pricing");
    let cold_grant = 32 * 1024 * 1024;
    let warm_grant = 16 * 1024 * 1024;
    let service = QueryService::new(
        QueryServiceConfig {
            device_capacity_bytes: Some(256 * 1024 * 1024),
            max_concurrent_queries: 2,
            max_queries_per_device: Some(2),
            slot_guarantee: nexus_query_engine::runtime::SlotGuarantee::Optimistic,
            max_source_chunk_bytes: cold_grant,
            resident_graph_cache_max_retained_bytes: Some(64 * 1024 * 1024),
            native_pipeline_execution_mode: mode,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), 8 * 1024 * 1024, 64 * 1024 * 1024)],
    )
    .unwrap();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 2])), Arc::new(Int32Array::from(vec![1, 2, 0, 3]))],
    )
    .unwrap();
    let local_source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/query-service-warm-pricing.parquet")
                .with_file_size_bytes(1)
                .with_stable_id("local:/tmp/query-service-warm-pricing.parquet:1:mtime=test"),
        ],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    );

    for (algorithm, params) in [
        (GraphAlgorithm::Bfs, GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None)))),
        (GraphAlgorithm::PageRank, GraphAlgorithmParams::PageRank(GraphPageRankParams::default())),
    ] {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            algorithm,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            params,
        );
        let mut descriptor_builder = QueryPlanBuilder::new();
        let descriptor_source = descriptor_builder.add_source(SourceNode::LocalParquet(local_source.clone()));
        let descriptor_plan = descriptor_builder.finish_with_host_sink(descriptor_source);
        let GraphResidencyPlanAnalysis::Keyable(descriptor) =
            graph_residency_descriptor_for_edge_plan(&descriptor_plan, &spec)
        else {
            panic!("test edge source must be keyable")
        };
        let resident_key = planned_resident_graph_key(&descriptor, &spec).expect("algorithm must be cacheable");

        let mut builder = QueryPlanBuilder::new();
        let edges =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&edge_schema))));
        let graph = builder.add_graph_algorithm(edges, spec);
        let plan = builder.finish_with_host_sink_schema(graph, stable_graph_result_schema(algorithm));
        let admission_request = || {
            QueryAdmissionRequest::new()
                .with_grant_request(QueryGrantRequest::with_requested_grant_bytes(cold_grant))
                .with_warm_graph_grant_request(QueryGrantRequest::with_requested_grant_bytes(warm_grant))
                .with_placement_hint(QueryPlacementHint::for_resident_graph_key(resident_key.clone()))
        };

        let cold_handle = service.admit_query_with_request(tenant_id.clone(), admission_request()).unwrap();
        assert_eq!(cold_handle.device_budget_bytes(), cold_grant);
        let cold_context = cold_handle.acquire_context().unwrap();
        let cold_frame = GpuDataFrame::new(
            DataFrame::from_record_batch(&cold_context.context().runtime().unwrap(), &edge_batch).unwrap(),
        )
        .with_graph_residency(descriptor.clone());
        let mut cold_inputs = GpuResidentInputMap::new();
        cold_inputs.insert(
            "edges".to_string(),
            GpuResultHandle::new(cold_context.context(), cold_frame, Arc::clone(&edge_schema)).unwrap(),
        );
        let cold = execute_query_plan_with_gpu_inputs(cold_context.context(), &plan, &cold_inputs).unwrap();
        assert!(
            cold.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata.contains("graph_cache=miss_inserted")
        );
        drop(cold_inputs);
        drop(cold_context);
        cold_handle.finish();
        drop(cold_handle);
        service.wait_for_stream_pools_idle(std::time::Duration::from_secs(5)).unwrap();

        // The cache now has a device-level hit, but its exact owning runtime
        // is deliberately leased by unrelated work. Warm pricing must remain
        // unavailable even though placement affinity sees the resident key.
        let blocker_handle = service
            .admit_query_with_request(
                tenant_id.clone(),
                QueryAdmissionRequest::new()
                    .with_grant_request(QueryGrantRequest::with_requested_grant_bytes(8 * 1024 * 1024)),
            )
            .unwrap();
        let blocker_context = blocker_handle.acquire_context().unwrap();
        let device_only_handle = service.admit_query_with_request(tenant_id.clone(), admission_request()).unwrap();
        assert_eq!(
            device_only_handle.device_budget_bytes(),
            cold_grant,
            "device-only cache affinity must not select the exact-runtime warm price"
        );
        device_only_handle.finish();
        drop(device_only_handle);
        drop(blocker_context);
        blocker_handle.finish();
        drop(blocker_handle);
        service.wait_for_stream_pools_idle(std::time::Duration::from_secs(5)).unwrap();

        let warm_handle = service.admit_query_with_request(tenant_id.clone(), admission_request()).unwrap();
        assert_eq!(warm_handle.device_budget_bytes(), warm_grant);
        let warm_context = warm_handle.acquire_context().unwrap();
        let warm_frame = GpuDataFrame::new(
            DataFrame::from_record_batch(&warm_context.context().runtime().unwrap(), &edge_batch).unwrap(),
        )
        .with_graph_residency(descriptor);
        let mut warm_inputs = GpuResidentInputMap::new();
        warm_inputs.insert(
            "edges".to_string(),
            GpuResultHandle::new(warm_context.context(), warm_frame, Arc::clone(&edge_schema)).unwrap(),
        );
        let warm = execute_query_plan_with_gpu_inputs(warm_context.context(), &plan, &warm_inputs).unwrap();
        assert!(warm.metrics().graph_algorithm_runtime_rows[0].algorithm_metadata.contains("graph_cache=hit;"));
        if algorithm == GraphAlgorithm::PageRank {
            let cold_reservation = &cold.metrics().graph_reservation_rows[0];
            let warm_reservation = &warm.metrics().graph_reservation_rows[0];
            assert!(
                warm_reservation.model_requested_bytes < cold_reservation.model_requested_bytes,
                "warm PageRank report must omit graph construction bytes in {mode:?} mode"
            );
            assert_eq!(cold_reservation.budget_bytes, Some(cold_grant));
            assert_eq!(warm_reservation.budget_bytes, Some(warm_grant));
        } else {
            assert!(cold.metrics().graph_reservation_rows.is_empty());
            assert!(warm.metrics().graph_reservation_rows.is_empty());
        }
        let (QueryOutput::HostArrow(cold_batch), QueryOutput::HostArrow(warm_batch)) = (cold.output(), warm.output())
        else {
            panic!("test graph plans must materialize host output")
        };
        assert_eq!(cold_batch, warm_batch);
        drop(warm_inputs);
        drop(warm_context);
        warm_handle.finish();
        drop(warm_handle);
    }

    let cache_rows = service.resident_graph_cache_snapshots().unwrap();
    assert_eq!(cache_rows.len(), 1);
    assert!(cache_rows[0].enabled);
    assert_eq!(cache_rows[0].entries, 2);
    assert_eq!(cache_rows[0].hits, 2);
    assert_eq!(cache_rows[0].misses, 2);
    assert_eq!(cache_rows[0].builds_in_flight, 0);
    assert_eq!(cache_rows[0].in_use_bytes, 0);
    let governor = service.memory_governor_reports().unwrap();
    assert_eq!(governor[0].evictable_bytes, cache_rows[0].resident_bytes);
}

fn assert_vertex_betweenness_batches_close(expected: &RecordBatch, actual: &RecordBatch) {
    let expected = vertex_betweenness_scores(expected);
    let actual = vertex_betweenness_scores(actual);
    assert_eq!(actual.len(), expected.len());
    for (vertex, expected_value) in expected {
        let actual_value = actual.get(&vertex).unwrap_or_else(|| panic!("missing vertex {vertex}"));
        assert!(
            (actual_value - expected_value).abs() <= 1.0e-6,
            "vertex {vertex}: expected {expected_value}, got {actual_value}"
        );
    }
}

fn vertex_betweenness_scores(batch: &RecordBatch) -> std::collections::BTreeMap<i64, f64> {
    let vertices = int64_array(batch, 0);
    let values = float64_array(batch, 1);
    (0..batch.num_rows()).map(|row| (vertices.value(row), values.value(row))).collect()
}

fn assert_edge_betweenness_batches_close(expected: &RecordBatch, actual: &RecordBatch) {
    let expected = edge_betweenness_scores(expected);
    let actual = edge_betweenness_scores(actual);
    assert_eq!(actual.len(), expected.len());
    for (edge, expected_value) in expected {
        let actual_value = actual.get(&edge).unwrap_or_else(|| panic!("missing edge {edge:?}"));
        assert!(
            (actual_value - expected_value).abs() <= 1.0e-6,
            "edge {edge:?}: expected {expected_value}, got {actual_value}"
        );
    }
}

fn edge_betweenness_scores(batch: &RecordBatch) -> std::collections::BTreeMap<(i64, i64), f64> {
    let sources = int64_array(batch, 0);
    let destinations = int64_array(batch, 1);
    let scores = float64_array(batch, 2);
    (0..batch.num_rows()).map(|row| ((sources.value(row), destinations.value(row)), scores.value(row))).collect()
}
