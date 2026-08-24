fn pagerank_plan(edge_schema: std::sync::Arc<arrow::datatypes::Schema>) -> crate::plan::QueryPlan {
    let params = crate::plan::GraphAlgorithmParams::PageRank(crate::plan::GraphPageRankParams::default());
    let vertex_type = edge_schema.field_with_name("src").unwrap().data_type().clone();
    let mut builder = crate::plan::QueryPlanBuilder::new();
    let edges = builder.add_source(crate::source::SourceNode::GpuResident(crate::source::GpuResidentSource::new(
        "edges",
        std::sync::Arc::clone(&edge_schema),
    )));
    let output_schema =
        crate::plan::graph_result_schema_for_vertex_type(crate::plan::GraphAlgorithm::PageRank, &params, &vertex_type);
    let spec = crate::plan::GraphAlgorithmSpec::new(
        crate::plan::GraphAlgorithm::PageRank,
        crate::plan::GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        crate::plan::GraphConstructionOptions {
            directed: true,
            renumber: true,
            policy: crate::plan::GraphConstructionPolicy::RawLibcugraph,
        },
        params.clone(),
        std::sync::Arc::clone(&output_schema),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    builder.finish_with_host_sink_schema(graph, output_schema)
}

fn bfs_plan(edge_schema: std::sync::Arc<arrow::datatypes::Schema>) -> crate::plan::QueryPlan {
    let params = crate::plan::GraphAlgorithmParams::Bfs(Box::new(crate::plan::GraphBfsParams::new(0, None)));
    let vertex_type = edge_schema.field_with_name("src").unwrap().data_type().clone();
    let mut builder = crate::plan::QueryPlanBuilder::new();
    let edges = builder.add_source(crate::source::SourceNode::GpuResident(crate::source::GpuResidentSource::new(
        "edges",
        std::sync::Arc::clone(&edge_schema),
    )));
    let output_schema =
        crate::plan::graph_result_schema_for_vertex_type(crate::plan::GraphAlgorithm::Bfs, &params, &vertex_type);
    let graph = builder.add_graph_algorithm(
        edges,
        crate::plan::GraphAlgorithmSpec::new(
            crate::plan::GraphAlgorithm::Bfs,
            crate::plan::GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            crate::plan::GraphConstructionOptions {
                directed: true,
                renumber: true,
                policy: crate::plan::GraphConstructionPolicy::RawLibcugraph,
            },
            params,
            std::sync::Arc::clone(&output_schema),
        ),
    );
    builder.finish_with_host_sink_schema(graph, output_schema)
}

#[test]
fn frozen_resident_edges_release_the_producer_cap_and_feed_both_graph_orientations() {
    const CAPACITY_BYTES: u64 = 128 * 1024 * 1024;
    let service = runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds::default(),
        vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: CAPACITY_BYTES })
                .with_default_query_limit_bytes(CAPACITY_BYTES)
                .with_source_work_per_attempt(2)
                .with_max_active_attempts(1)
                .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(3, 1, 1)),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )],
    )
    .expect("single-flight full-cap service");
    let edge_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("src", arrow::datatypes::DataType::Int32, false),
        arrow::datatypes::Field::new("dst", arrow::datatypes::DataType::Int32, false),
    ]));
    let pagerank = pagerank_plan(std::sync::Arc::clone(&edge_schema));
    let bfs = bfs_plan(std::sync::Arc::clone(&edge_schema));
    let mut producer_evidence = crate::admission::AttemptCapabilityEvidence::for_query_plan(&pagerank);
    producer_evidence.attest_cugraph_graph_execution_scope(&pagerank);
    let lifecycle = crate::runtime::BackendLifecycle::new();
    let counters = std::sync::Arc::new(crate::attempt::AttemptBackendCounters::default());
    let observed = crate::attempt::AttemptOwner::begin(
        crate::attempt::BeginAttemptRequest {
            observation: crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..crate::attempt::BeginAttemptRequest::default()
        },
        lifecycle.try_begin_observation().expect("observation permit"),
        counters,
        None,
    );
    let producer_owner = observed.owner;
    let producer_completion = observed.completion.completion();
    let producer_observation = producer_owner.handle();
    producer_observation.record_planning_started().unwrap();
    producer_observation.record_planning_finished().unwrap();
    producer_observation.set_disposition(crate::observability::AttemptDisposition::Native).unwrap();
    producer_observation.record_admission_requested().unwrap();
    let producer = service
        .admit_query_with_request(
            QueryAdmissionRequest::new()
                .with_attempt_binding(producer_owner.admission_binding())
                .with_query_plan_capability_evidence(producer_evidence),
        )
        .expect("producer admission");
    producer_observation
        .record_admitted(crate::observability::AttemptAdmissionFact {
            admission_sequence: producer.admission_sequence(),
            admission_wait_ns: producer.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: producer.device_ordinal(),
            grant_bytes: producer.device_budget_bytes(),
        })
        .unwrap();
    producer_observation.record_physical_execution_started().unwrap();
    producer_observation.record_gpu_execution_started().unwrap();
    let producer_context = producer.acquire_context().expect("producer context");
    let producer_runtime = producer_context.context().shared_runtime().expect("producer runtime");
    let batch = arrow::record_batch::RecordBatch::try_new(
        std::sync::Arc::clone(&edge_schema),
        vec![
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![0, 1, 2, 2])),
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![1, 2, 0, 3])),
        ],
    )
    .unwrap();
    let gpu_edges = crate::runtime::resident::GpuDataFrame::new(
        cudf::DataFrame::from_record_batch(producer_runtime.as_ref(), &batch).expect("edge upload"),
    );
    let resident_bytes = gpu_edges.device_allocated_size_bytes().expect("resident edge bytes");
    let resident_lease = producer_context
        .context()
        .memory_grant()
        .expect("producer grant access")
        .expect("producer grant")
        .retain_resident_owner(resident_bytes)
        .expect("resident edges fit the producer cap");
    let resident = crate::runtime::resident::GpuResultHandle::new_with_resident_owner(
        producer_context.context(),
        gpu_edges,
        std::sync::Arc::clone(&edge_schema),
        resident_lease,
    )
    .expect("resident edge owner");
    let resident_placement = resident.placement();
    let frozen_edge_bits = resident.to_host_record_batch().expect("baseline frozen edge values");
    let resident_view = resident.clone();
    drop(resident);
    let mut inputs = crate::runtime::resident::GpuResidentInputMap::new();
    inputs.insert("edges".to_string(), resident_view);
    drop(producer_runtime);
    drop(producer_context);
    producer_owner.commit_terminal(crate::observability::RuntimeTerminalOutcome::Succeeded).unwrap();
    producer.finish();
    drop(producer);

    let producer_report =
        producer_completion.wait_timeout(std::time::Duration::from_secs(10)).expect("producer frozen report");
    assert_eq!(producer_report.capacity_disposition, crate::observability::AttemptCapacityDisposition::FrozenResident);
    let freeze_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let frozen = loop {
        let snapshot = service.device_ledger_reports().unwrap().remove(0);
        if snapshot.active_attempts == 0 && snapshot.frozen_resident_bytes > 0 {
            break snapshot;
        }
        assert!(std::time::Instant::now() < freeze_deadline, "frozen charge publishes after the attempt report");
        std::thread::yield_now();
    };
    assert_eq!(frozen.active_attempts, 0);
    assert_eq!(frozen.live_grants_bytes, 0);
    assert!(frozen.frozen_resident_bytes > 0);
    assert!(frozen.frozen_resident_bytes < CAPACITY_BYTES);

    let mut tiny_evidence = crate::admission::AttemptCapabilityEvidence::for_query_plan(&pagerank);
    tiny_evidence.attest_cugraph_graph_execution_scope(&pagerank);
    let tiny = service
        .admit_query_with_request(
            QueryAdmissionRequest::new().with_query_cap_bytes(1).with_query_plan_capability_evidence(tiny_evidence),
        )
        .expect("tiny explicit cap is admissible");
    let tiny_context = tiny.acquire_context().expect("tiny context");
    let tiny_error = match crate::exec::execute_query_plan_with_gpu_inputs(tiny_context.context(), &pagerank, &inputs) {
        Ok(_) => panic!("actual allocation must exceed the one-byte query cap"),
        Err(error) => error,
    };
    assert_eq!(
        tiny_error.code(),
        nexus_query_engine::error::QueryErrorCode::QueryMemoryLimitExceeded,
        "unexpected tiny-cap failure: {tiny_error:?}"
    );
    drop(tiny_context);
    tiny.finish();
    drop(tiny);

    for plan in [&pagerank, &bfs] {
        let mut evidence = crate::admission::AttemptCapabilityEvidence::for_query_plan(plan);
        evidence.attest_cugraph_graph_execution_scope(plan);
        let consumer = service
            .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))
            .expect("remaining capacity admits graph consumer");
        let consumer_context = consumer.acquire_context().expect("consumer context");
        let result = crate::exec::execute_query_plan_with_gpu_inputs(consumer_context.context(), plan, &inputs)
            .expect("frozen edges execute without relocation");
        assert_eq!(result.metrics().boundary.device_to_device_copies, 0);
        assert_eq!(result.metrics().boundary.device_to_device_copied_bytes, 0);
        assert_eq!(result.metrics().graph_algorithm_runtime_rows.len(), 1);
        drop(result);
        drop(consumer_context);
        consumer.finish();
        drop(consumer);
        assert_eq!(inputs.get("edges").unwrap().placement(), resident_placement);
        assert_eq!(
            inputs.get("edges").unwrap().to_host_record_batch().expect("frozen edges remain readable"),
            frozen_edge_bits,
            "consumer-owned graph orientation must not mutate the frozen producer values"
        );
    }

    assert!(service.device_ledger_reports().unwrap().remove(0).frozen_resident_bytes > 0);
    drop(inputs);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while service.device_ledger_reports().unwrap().remove(0).frozen_resident_bytes != 0 {
        assert!(std::time::Instant::now() < deadline, "resident charge releases after final owner");
        std::thread::yield_now();
    }
    service.close_and_wait_for_quiescence(std::time::Duration::from_secs(10)).unwrap();
}

#[test]
fn graph_gpu_input_from_another_admitted_attempt_borrows_the_resident_domain_without_copy() {
    let service = try_guaranteed_runtime_test_service_with_concurrency(2)
        .expect("runtime-backed service with two same-device attempt lanes");
    let edge_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("src", arrow::datatypes::DataType::Int32, false),
        arrow::datatypes::Field::new("dst", arrow::datatypes::DataType::Int32, false),
    ]));
    let plan = pagerank_plan(std::sync::Arc::clone(&edge_schema));
    let mut evidence = crate::admission::AttemptCapabilityEvidence::for_query_plan(&plan);
    evidence.attest_cugraph_graph_execution_scope(&plan);
    let producer_attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence.clone()))
        .expect("producer attempt admission");
    let consumer_attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))
        .expect("consumer attempt admission");
    let producer_context = producer_attempt.acquire_context().expect("producer context");
    let consumer_context = consumer_attempt.acquire_context().expect("consumer context");

    let producer_domain =
        producer_context.context().attempt_allocation_domain_id().expect("producer immutable allocation domain");
    let consumer_domain =
        consumer_context.context().attempt_allocation_domain_id().expect("consumer immutable allocation domain");
    assert_ne!(producer_domain, consumer_domain, "simultaneous attempts must own distinct allocation domains");

    let producer_runtime = producer_context.context().shared_runtime().expect("producer runtime");
    let consumer_runtime = consumer_context.context().shared_runtime().expect("consumer runtime");
    assert_eq!(producer_runtime.device(), consumer_runtime.device(), "both attempts must use the same test device");
    assert_ne!(
        producer_runtime.binding_id(),
        consumer_runtime.binding_id(),
        "same-device attempts must use distinct exact execution bindings"
    );
    let edge_batch = arrow::record_batch::RecordBatch::try_new(
        std::sync::Arc::clone(&edge_schema),
        vec![
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![0, 1, 2, 2])),
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![1, 2, 0, 3])),
        ],
    )
    .expect("valid edge batch");
    let producer_input = crate::runtime::resident::GpuResultHandle::new(
        producer_context.context(),
        crate::runtime::resident::GpuDataFrame::new(
            cudf::DataFrame::from_record_batch(producer_runtime.as_ref(), &edge_batch).expect("producer edge upload"),
        ),
        std::sync::Arc::clone(&edge_schema),
    )
    .expect("producer GPU input handle");
    let producer_input_placement = producer_input.placement();
    let producer_usage_before = producer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("producer domain usage")
        .expect("producer domain usage is present");
    assert!(producer_usage_before.current_bytes > 0, "producer input must be retained in its own domain");

    let mut inputs = crate::runtime::resident::GpuResidentInputMap::new();
    inputs.insert("edges".to_string(), producer_input);
    let source = crate::source::GpuResidentSource::new("edges", std::sync::Arc::clone(&edge_schema));
    let mut same_attempt_metrics = crate::metrics::QueryMetrics::default();
    let same_attempt_frame =
        crate::exec::read_gpu_resident_source(producer_context.context(), &source, &inputs, &mut same_attempt_metrics)
            .expect("same-attempt GPU input remains a zero-copy handoff");
    assert_eq!(same_attempt_metrics.boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(same_attempt_metrics.boundary.device_to_device_copies, 0);
    assert_eq!(same_attempt_frame.placement(), producer_input_placement);
    drop(same_attempt_frame);

    let consumer_usage_before = consumer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("consumer domain usage")
        .expect("consumer domain usage is present");
    let mut handoff_metrics = crate::metrics::QueryMetrics::default();
    let borrowed =
        crate::exec::read_gpu_resident_source(consumer_context.context(), &source, &inputs, &mut handoff_metrics)
            .expect("foreign GPU input is borrowed before graph execution");

    assert_eq!(handoff_metrics.boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(handoff_metrics.boundary.device_to_device_copies, 0);
    assert_eq!(handoff_metrics.boundary.device_to_device_copied_bytes, 0);
    assert_eq!(borrowed.placement(), producer_input_placement);
    let consumer_usage_after = consumer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("consumer domain usage after relocation")
        .expect("consumer domain usage is present");
    assert!(
        consumer_usage_after.current_bytes == consumer_usage_before.current_bytes,
        "resident input bytes must not be charged to the consuming attempt domain"
    );
    let producer_usage_after = producer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("producer domain usage after relocation")
        .expect("producer domain usage is present");
    assert_eq!(
        producer_usage_after.current_bytes, producer_usage_before.current_bytes,
        "borrowing the resident input must not allocate into the producer attempt domain"
    );
    drop(borrowed);
    consumer_runtime.synchronize().expect("consumer handoff stream synchronization");

    let result = crate::exec::execute_query_plan_with_gpu_inputs(consumer_context.context(), &plan, &inputs)
        .expect("graph execution after the source-domain handoff");
    assert_eq!(result.metrics().boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(result.metrics().boundary.device_to_device_copies, 0);
    assert_eq!(result.metrics().graph_algorithm_runtime_rows.len(), 1, "the graph runs from the resident borrow");
    let crate::exec::output::QueryOutput::HostArrow(output) = result.output() else {
        panic!("graph plan must materialize its result at the declared host sink");
    };
    assert_eq!(output.num_rows(), 4, "PageRank must see the copied four-vertex edge list");

    drop(result);
    drop(inputs);
    drop(producer_runtime);
    drop(consumer_runtime);
    drop(producer_context);
    drop(consumer_context);
    producer_attempt.finish();
    consumer_attempt.finish();
    drop(producer_attempt);
    drop(consumer_attempt);
    service
        .close_and_wait_for_quiescence(std::time::Duration::from_secs(10))
        .expect("query service cleanup after cross-attempt graph handoff");
}

#[test]
fn cache_enabled_pagerank_promotes_one_graph_into_the_ledger_owned_domain() {
    const CACHE_CAP_BYTES: u64 = 16 * 1024 * 1024;

    let service = runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds::default(),
        vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 128 * 1024 * 1024 })
                .with_default_query_limit_bytes(8 * 1024 * 1024)
                .with_source_work_per_attempt(2)
                .with_max_active_attempts(8)
                .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(24, 8, 8))
                .with_cache_cap_bytes(CACHE_CAP_BYTES),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )],
    )
    .expect("cache-enabled runtime service");
    let edge_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("src", arrow::datatypes::DataType::Int32, false),
        arrow::datatypes::Field::new("dst", arrow::datatypes::DataType::Int32, false),
    ]));
    let plan = pagerank_plan(std::sync::Arc::clone(&edge_schema));
    let mut evidence = crate::admission::AttemptCapabilityEvidence::for_query_plan(&plan);
    evidence.attest_cugraph_graph_execution_scope(&plan);
    let backend_lifecycle = crate::runtime::BackendLifecycle::new();
    let counters = std::sync::Arc::new(crate::attempt::AttemptBackendCounters::default());
    let observed_attempt = crate::attempt::AttemptOwner::begin(
        crate::attempt::BeginAttemptRequest {
            observation: crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..crate::attempt::BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = observed_attempt.owner;
    let completion = observed_attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(crate::observability::AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let attempt = service
        .admit_query_with_request(
            QueryAdmissionRequest::new()
                .with_attempt_binding(attempt_owner.admission_binding())
                .with_query_plan_capability_evidence(evidence),
        )
        .expect("admission");
    attempt_handle
        .record_admitted(crate::observability::AttemptAdmissionFact {
            admission_sequence: attempt.admission_sequence(),
            admission_wait_ns: attempt.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: attempt.device_ordinal(),
            grant_bytes: attempt.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    let admitted = attempt.acquire_context().expect("admitted context");
    let context = admitted.context();
    let runtime = context.runtime().expect("attempt runtime");
    let edge_batch = arrow::record_batch::RecordBatch::try_new(
        std::sync::Arc::clone(&edge_schema),
        vec![
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![0, 1, 2, 2])),
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![1, 2, 0, 3])),
        ],
    )
    .expect("valid edge batch");
    let descriptor = crate::plan::GraphResidencyDescriptor::for_content_identity(
        "accounted-pagerank-graph-v1",
        arrow::datatypes::DataType::Int32,
        None,
        None,
    )
    .expect("stable graph source identity");
    let input = crate::runtime::resident::GpuResultHandle::new(
        context,
        crate::runtime::resident::GpuDataFrame::new(
            cudf::DataFrame::from_record_batch(&runtime, &edge_batch).expect("attempt edge upload"),
        )
        .with_graph_residency(descriptor),
        std::sync::Arc::clone(&edge_schema),
    )
    .expect("resident graph input");
    let mut inputs = crate::runtime::resident::GpuResidentInputMap::new();
    inputs.insert("edges".to_string(), input);
    let result = crate::exec::execute_query_plan_with_gpu_inputs(context, &plan, &inputs)
        .expect("cache-enabled PageRank execution");
    let ledger_report = service.device_ledger_reports().expect("ledger report").remove(0);
    assert!(ledger_report.cache_evictable_resident_bytes > 0, "the parked graph remains ledger-owned after execution");
    assert_eq!(ledger_report.cache_pinned_in_use_bytes, 0, "the graph checkout releases its pin after execution");
    assert!(ledger_report.cache_evictable_resident_bytes <= CACHE_CAP_BYTES);

    drop(result);
    drop(inputs);
    drop(runtime);
    drop(admitted);
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");
    attempt_owner
        .commit_terminal(crate::observability::RuntimeTerminalOutcome::Succeeded)
        .expect("primary attempt outcome commits before release");
    attempt.finish();
    drop(attempt);
    let sealed = completion.wait_timeout(std::time::Duration::from_secs(10)).expect("sealed attempt report");
    let evidence = sealed.resource_evidence.expect("admitted cache attempt retains sealed resource evidence");
    assert_eq!(
        evidence.cache_evictable_resident_bytes.value,
        Some(ledger_report.cache_evictable_resident_bytes),
        "sealed evidence projects the same ledger snapshot that owns the parked graph"
    );
    assert_eq!(
        evidence.cache_pinned_in_use_bytes.value,
        Some(ledger_report.cache_pinned_in_use_bytes),
        "sealed evidence projects the same ledger snapshot that owns the cache pin state"
    );
    assert_eq!(
        evidence.ledger_sequence.value,
        Some(ledger_report.sequence),
        "all cache evidence fields retain the ledger snapshot sequence"
    );
    service.close_and_wait_for_quiescence(std::time::Duration::from_secs(10)).expect("service drains cache entries");
    let drained_report = service.device_ledger_reports().expect("drained ledger report").remove(0);
    assert_eq!(drained_report.cache_evictable_resident_bytes, 0);
    assert_eq!(drained_report.cache_pinned_in_use_bytes, 0);
}

/// Writes a multi-row-group Parquet file whose columns include a nullable
/// string. Per-chunk string offsets and null-mask padding make the sum of the
/// chunk reservations larger than the concatenated output, so the executor's
/// retained charge and the allocator's measurement disagree by construction.
fn write_multi_row_group_string_parquet(path: &std::path::Path) -> std::sync::Arc<arrow::datatypes::Schema> {
    use arrow::datatypes::{DataType, Field, Schema};

    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
        Field::new("weight", DataType::Int64, true),
    ]));
    let rows = 4_096usize;
    let src = (0..rows).map(|row| (row % 128) as i32).collect::<Vec<_>>();
    let dst = (0..rows).map(|row| ((row + 1) % 128) as i32).collect::<Vec<_>>();
    let label =
        (0..rows).map(|row| (row % 7 != 0).then(|| format!("edge-label-{row:06}"))).collect::<Vec<Option<String>>>();
    let weight = (0..rows).map(|row| (row % 5 != 0).then_some(row as i64)).collect::<Vec<Option<i64>>>();
    let batch = arrow::record_batch::RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(arrow::array::Int32Array::from(src)),
            std::sync::Arc::new(arrow::array::Int32Array::from(dst)),
            std::sync::Arc::new(arrow::array::StringArray::from(label)),
            std::sync::Arc::new(arrow::array::Int64Array::from(weight)),
        ],
    )
    .expect("multi row-group fixture batch");
    let properties =
        parquet::file::properties::WriterProperties::builder().set_max_row_group_row_count(Some(512)).build();
    let file = std::fs::File::create(path).expect("fixture parquet file");
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, std::sync::Arc::clone(&schema), Some(properties))
        .expect("fixture parquet writer");
    writer.write(&batch).expect("fixture parquet rows");
    writer.close().expect("fixture parquet footer");
    schema
}

/// An executor-produced GPU-resident output must freeze into a resident
/// result. The producer here is a real bounded execution over a multi
/// row-group Parquet source with string and nullable columns, so its
/// executor-side retained charge does not equal the allocator's own
/// measurement of the concatenated output.
#[test]
fn executor_produced_resident_parquet_output_freezes_without_quarantine() {
    const CAPACITY_BYTES: u64 = 128 * 1024 * 1024;

    let service = runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds::default(),
        vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: CAPACITY_BYTES })
                // A cap well below the device capacity makes the source read
                // split into several chunks, so the concatenated output is
                // charged as the sum of its chunk reservations while the
                // allocator measures the smaller merged result.
                .with_default_query_limit_bytes(384 * 1024)
                .with_source_work_per_attempt(2)
                .with_max_active_attempts(1)
                .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(3, 1, 1)),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )],
    )
    .expect("runtime-backed service for executor-produced resident freeze");

    let dir = std::env::temp_dir().join(format!(
        "nexus-resident-freeze-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir).expect("fixture directory");
    let path = dir.join("edges.parquet");
    let schema = write_multi_row_group_string_parquet(&path);

    let file_size_bytes = std::fs::metadata(&path).expect("fixture parquet metadata").len();
    let source = crate::source::LocalParquetSource::from_files(
        vec![
            crate::source::LocalParquetFile::new(path.to_str().expect("utf-8 fixture path"))
                .with_stable_id(format!("local:{}:{file_size_bytes}", path.display()))
                .with_file_size_bytes(file_size_bytes)
                .with_row_group_rows(vec![512; 8]),
        ],
        std::sync::Arc::clone(&schema),
        vec!["src".to_string(), "dst".to_string(), "label".to_string(), "weight".to_string()],
    )
    // A scan limit keeps each planned row group as its own read, so the
    // resident output is a concatenation of several chunks rather than one
    // merged read.
    .with_scan_limit(Some(4_096));
    let mut builder = crate::plan::QueryPlanBuilder::new();
    let scan = builder.add_source(crate::source::SourceNode::LocalParquet(source));
    let plan = builder.finish_gpu_resident(scan);

    let lifecycle = crate::runtime::BackendLifecycle::new();
    let counters = std::sync::Arc::new(crate::attempt::AttemptBackendCounters::default());
    let observed = crate::attempt::AttemptOwner::begin(
        crate::attempt::BeginAttemptRequest {
            observation: crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..crate::attempt::BeginAttemptRequest::default()
        },
        lifecycle.try_begin_observation().expect("observation permit"),
        counters,
        None,
    );
    let attempt_owner = observed.owner;
    let completion = observed.completion.completion();
    let attempt_observation = attempt_owner.handle();
    attempt_observation.record_planning_started().unwrap();
    attempt_observation.record_planning_finished().unwrap();
    attempt_observation.set_disposition(crate::observability::AttemptDisposition::Native).unwrap();
    attempt_observation.record_admission_requested().unwrap();
    let evidence = crate::admission::AttemptCapabilityEvidence::for_query_plan(&plan);
    let producer = service
        .admit_query_with_request(
            QueryAdmissionRequest::new()
                .with_attempt_binding(attempt_owner.admission_binding())
                .with_query_plan_capability_evidence(evidence),
        )
        .expect("producer admission");
    attempt_observation
        .record_admitted(crate::observability::AttemptAdmissionFact {
            admission_sequence: producer.admission_sequence(),
            admission_wait_ns: producer.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: producer.device_ordinal(),
            grant_bytes: producer.device_budget_bytes(),
        })
        .unwrap();
    attempt_observation.record_physical_execution_started().unwrap();
    attempt_observation.record_gpu_execution_started().unwrap();
    let producer_context = producer.acquire_context().expect("producer context");
    let result = crate::exec::execute_query_plan(producer_context.context(), &plan)
        .expect("multi row-group resident scan executes");
    let (output, _) = result.into_parts();
    let resident = output.as_gpu_resident().expect("GPU-resident sink produces a resident output owner");
    let resident_rows = resident.to_host_record_batch().expect("resident output is readable").num_rows();
    assert_eq!(resident_rows, 4_096);

    drop(producer_context);
    attempt_owner.commit_terminal(crate::observability::RuntimeTerminalOutcome::Succeeded).unwrap();
    producer.finish();
    drop(producer);

    let report = completion.wait_timeout(std::time::Duration::from_secs(10)).expect("producer report");
    assert_eq!(
        report.capacity_disposition,
        crate::observability::AttemptCapacityDisposition::FrozenResident,
        "an executor-produced resident output must freeze, not quarantine the device"
    );
    assert_eq!(report.cleanup_failure_count, 0);

    // The B-to-R publish runs after the attempt lane is released, so the
    // frozen charge becomes visible strictly after the sealed report.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let snapshot = loop {
        let snapshot = service.device_ledger_reports().unwrap().remove(0);
        if snapshot.active_attempts == 0 && snapshot.frozen_resident_bytes > 0 {
            break snapshot;
        }
        assert!(std::time::Instant::now() < deadline, "the frozen output publishes a measured resident charge");
        std::thread::yield_now();
    };
    assert_eq!(snapshot.live_grants_bytes, 0);
    assert!(snapshot.frozen_resident_bytes < CAPACITY_BYTES);

    drop(output);
    let _ = std::fs::remove_dir_all(&dir);
}
