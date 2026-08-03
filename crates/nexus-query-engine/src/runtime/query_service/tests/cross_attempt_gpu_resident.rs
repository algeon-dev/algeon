#[test]
fn graph_gpu_input_from_another_admitted_attempt_is_relocated_into_the_consumer_domain() {
    let service = try_guaranteed_runtime_test_service_with_concurrency(2)
        .expect("runtime-backed service with two same-device attempt lanes");
    let producer_attempt = service.admit_query().expect("producer attempt admission");
    let consumer_attempt = service.admit_query().expect("consumer attempt admission");
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

    let edge_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("src", arrow::datatypes::DataType::Int32, false),
        arrow::datatypes::Field::new("dst", arrow::datatypes::DataType::Int32, false),
    ]));
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
    let relocated =
        crate::exec::read_gpu_resident_source(consumer_context.context(), &source, &inputs, &mut handoff_metrics)
            .expect("foreign GPU input is relocated before graph execution");

    assert_eq!(handoff_metrics.boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(handoff_metrics.boundary.device_to_device_copies, 1);
    assert!(handoff_metrics.boundary.device_to_device_copied_bytes > 0);
    assert_eq!(relocated.placement().stream_debug, consumer_runtime.placement().stream_debug);
    assert_ne!(
        relocated.placement().memory_resource_debug,
        producer_input_placement.memory_resource_debug,
        "the resolved frame must not retain the producer attempt memory resource"
    );
    let consumer_usage_after = consumer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("consumer domain usage after relocation")
        .expect("consumer domain usage is present");
    assert!(
        consumer_usage_after.current_bytes > consumer_usage_before.current_bytes,
        "the handoff copy must be retained by the consuming attempt domain"
    );
    let producer_usage_after = producer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("producer domain usage after relocation")
        .expect("producer domain usage is present");
    assert_eq!(
        producer_usage_after.current_bytes, producer_usage_before.current_bytes,
        "the consuming handoff must not allocate into the producer attempt domain"
    );
    drop(relocated);
    consumer_runtime.synchronize().expect("consumer handoff stream synchronization");

    let mut builder = crate::plan::QueryPlanBuilder::new();
    let edges = builder.add_source(crate::source::SourceNode::GpuResident(crate::source::GpuResidentSource::new(
        "edges",
        std::sync::Arc::clone(&edge_schema),
    )));
    let spec = crate::plan::GraphAlgorithmSpec::with_stable_schema(
        crate::plan::GraphAlgorithm::PageRank,
        crate::plan::GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        crate::plan::GraphConstructionOptions {
            directed: true,
            renumber: true,
            policy: crate::plan::GraphConstructionPolicy::RawLibcugraph,
        },
        crate::plan::GraphAlgorithmParams::PageRank(crate::plan::GraphPageRankParams::default()),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(
        graph,
        crate::plan::stable_graph_result_schema(crate::plan::GraphAlgorithm::PageRank),
    );
    let result = crate::exec::execute_query_plan_with_gpu_inputs(consumer_context.context(), &plan, &inputs)
        .expect("graph execution after the source-domain handoff");
    assert_eq!(result.metrics().boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(result.metrics().boundary.device_to_device_copies, 1);
    assert_eq!(result.metrics().graph_algorithm_runtime_rows.len(), 1, "the graph runs after source relocation");
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
        vec![runtime_test_device_profile_with_limits(
            0,
            128 * 1024 * 1024,
            8 * 1024 * 1024,
            8,
            24,
            8,
            8,
        )
        .with_cache_cap_bytes(CACHE_CAP_BYTES)],
    )
    .expect("cache-enabled runtime service");
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
    let completion = observed_attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(crate::observability::AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
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
    let edge_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("src", arrow::datatypes::DataType::Int32, false),
        arrow::datatypes::Field::new("dst", arrow::datatypes::DataType::Int32, false),
    ]));
    let edge_batch = arrow::record_batch::RecordBatch::try_new(
        std::sync::Arc::clone(&edge_schema),
        vec![
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![0, 1, 2, 2])),
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![1, 2, 0, 3])),
        ],
    )
    .expect("valid edge batch");
    let descriptor = crate::plan::GraphResidencyDescriptor::for_content_identity(
        "p5-accounted-pagerank-graph-v1",
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
    let mut builder = crate::plan::QueryPlanBuilder::new();
    let edges = builder.add_source(crate::source::SourceNode::GpuResident(crate::source::GpuResidentSource::new(
        "edges",
        std::sync::Arc::clone(&edge_schema),
    )));
    let spec = crate::plan::GraphAlgorithmSpec::with_stable_schema(
        crate::plan::GraphAlgorithm::PageRank,
        crate::plan::GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        crate::plan::GraphConstructionOptions {
            directed: true,
            renumber: true,
            policy: crate::plan::GraphConstructionPolicy::RawLibcugraph,
        },
        crate::plan::GraphAlgorithmParams::PageRank(crate::plan::GraphPageRankParams::default()),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink_schema(
        graph,
        crate::plan::stable_graph_result_schema(crate::plan::GraphAlgorithm::PageRank),
    );

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
    let evidence = sealed.resource_evidence.expect("admitted cache attempt retains P3 resource evidence");
    assert_eq!(
        evidence.cache_evictable_resident_bytes.value,
        Some(ledger_report.cache_evictable_resident_bytes),
        "sealed P3 evidence projects the same ledger snapshot that owns the parked graph"
    );
    assert_eq!(
        evidence.cache_pinned_in_use_bytes.value,
        Some(ledger_report.cache_pinned_in_use_bytes),
        "sealed P3 evidence projects the same ledger snapshot that owns the cache pin state"
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
