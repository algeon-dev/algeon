use std::{
    env,
    fs,
    path::Path,
    process,
    sync::{Arc, Barrier},
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Float64Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema},
    record_batch::RecordBatch,
};
use cudf::DataFrame as CudfDataFrame;
use parquet::{arrow as parquet_arrow, file::properties as parquet_properties};

fn pagerank_plan(edge_schema: Arc<ArrowSchema>) -> plan::QueryPlan {
    let params = plan::GraphAlgorithmParams::PageRank(plan::GraphPageRankParams::default());
    let vertex_type = edge_schema.field_with_name("src").unwrap().data_type().clone();
    let mut builder = plan::QueryPlanBuilder::new();
    let edges = builder.add_source(source::SourceNode::GpuResident(source::GpuResidentSource::new(
        "edges",
        Arc::clone(&edge_schema),
    )));
    let output_schema =
        plan::graph_result_schema_for_vertex_type(plan::GraphAlgorithm::PageRank, &params, &vertex_type);
    let spec = plan::GraphAlgorithmSpec::new(
        plan::GraphAlgorithm::PageRank,
        plan::GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        plan::GraphConstructionOptions {
            directed: true,
            renumber: true,
            policy: plan::GraphConstructionPolicy::RawLibcugraph,
        },
        params,
        Arc::clone(&output_schema),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    builder.finish_with_host_sink_schema(graph, output_schema)
}

fn bfs_plan(edge_schema: Arc<ArrowSchema>) -> plan::QueryPlan {
    let params = plan::GraphAlgorithmParams::Bfs(Box::new(plan::GraphBfsParams::new(0, None)));
    let vertex_type = edge_schema.field_with_name("src").unwrap().data_type().clone();
    let mut builder = plan::QueryPlanBuilder::new();
    let edges = builder.add_source(source::SourceNode::GpuResident(source::GpuResidentSource::new(
        "edges",
        Arc::clone(&edge_schema),
    )));
    let output_schema =
        plan::graph_result_schema_for_vertex_type(plan::GraphAlgorithm::Bfs, &params, &vertex_type);
    let graph = builder.add_graph_algorithm(
        edges,
        plan::GraphAlgorithmSpec::new(
            plan::GraphAlgorithm::Bfs,
            plan::GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            plan::GraphConstructionOptions {
                directed: true,
                renumber: true,
                policy: plan::GraphConstructionPolicy::RawLibcugraph,
            },
            params,
            Arc::clone(&output_schema),
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
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("src", ArrowDataType::Int32, false),
        ArrowField::new("dst", ArrowDataType::Int32, false),
    ]));
    let pagerank = pagerank_plan(Arc::clone(&edge_schema));
    let bfs = bfs_plan(Arc::clone(&edge_schema));
    let mut producer_evidence = admission::AttemptCapabilityEvidence::for_query_plan(&pagerank);
    producer_evidence.attest_cugraph_graph_execution_scope(&pagerank);
    let lifecycle = runtime::BackendLifecycle::new();
    let counters = Arc::new(attempt::AttemptBackendCounters::default());
    let observed = attempt::AttemptOwner::begin(
        attempt::BeginAttemptRequest {
            observation: observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..attempt::BeginAttemptRequest::default()
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
    producer_observation.set_disposition(observability::AttemptDisposition::Native).unwrap();
    producer_observation.record_admission_requested().unwrap();
    let producer = service
        .admit_query_with_request(
            QueryAdmissionRequest::new()
                .with_attempt_binding(producer_owner.admission_binding())
                .with_query_plan_capability_evidence(producer_evidence),
        )
        .expect("producer admission");
    producer_observation
        .record_admitted(observability::AttemptAdmissionFact {
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
    let batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 0, 3])),
        ],
    )
    .unwrap();
    let gpu_edges = runtime::resident::GpuDataFrame::new(
        CudfDataFrame::from_record_batch(producer_runtime.as_ref(), &batch).expect("edge upload"),
    );
    let resident_bytes = gpu_edges.device_allocated_size_bytes().expect("resident edge bytes");
    let resident_lease = producer_context
        .context()
        .memory_grant()
        .expect("producer grant access")
        .expect("producer grant")
        .retain_resident_owner(resident_bytes)
        .expect("resident edges fit the producer cap");
    let resident = runtime::resident::GpuResultHandle::new_with_resident_owner(
        producer_context.context(),
        gpu_edges,
        Arc::clone(&edge_schema),
        resident_lease,
    )
    .expect("resident edge owner");
    let resident_placement = resident.placement();
    let frozen_edge_bits = resident.to_host_record_batch().expect("baseline frozen edge values");
    let resident_view = resident.clone();
    drop(resident);
    let mut inputs = runtime::resident::GpuResidentInputMap::new();
    inputs.insert("edges".to_string(), resident_view);
    drop(producer_runtime);
    drop(producer_context);
    producer_owner.commit_terminal(observability::RuntimeTerminalOutcome::Succeeded).unwrap();
    producer.finish();
    drop(producer);

    let producer_report =
        producer_completion.wait_timeout(Duration::from_secs(10)).expect("producer frozen report");
    assert_eq!(producer_report.capacity_disposition, observability::AttemptCapacityDisposition::FrozenResident);
    let freeze_deadline = Instant::now() + Duration::from_secs(10);
    let frozen = loop {
        let snapshot = service.device_ledger_reports().unwrap().remove(0);
        if snapshot.active_attempts == 0 && snapshot.frozen_resident_bytes > 0 {
            break snapshot;
        }
        assert!(Instant::now() < freeze_deadline, "frozen charge publishes after the attempt report");
        thread::yield_now();
    };
    assert_eq!(frozen.active_attempts, 0);
    assert_eq!(frozen.live_grants_bytes, 0);
    assert!(frozen.frozen_resident_bytes > 0);
    assert!(frozen.frozen_resident_bytes < CAPACITY_BYTES);

    let mut tiny_evidence = admission::AttemptCapabilityEvidence::for_query_plan(&pagerank);
    tiny_evidence.attest_cugraph_graph_execution_scope(&pagerank);
    let tiny = service
        .admit_query_with_request(
            QueryAdmissionRequest::new().with_query_cap_bytes(1).with_query_plan_capability_evidence(tiny_evidence),
        )
        .expect("tiny explicit cap is admissible");
    let tiny_context = tiny.acquire_context().expect("tiny context");
    let tiny_error = match execute_plan(tiny_context.context(), &pagerank, Some(&inputs)) {
        Ok(_) => panic!("actual allocation must exceed the one-byte query cap"),
        Err(error) => error,
    };
    assert_eq!(
        tiny_error.code(),
        error::QueryErrorCode::QueryMemoryLimitExceeded,
        "unexpected tiny-cap failure: {tiny_error:?}"
    );
    drop(tiny_context);
    tiny.finish();
    drop(tiny);

    for plan in [&pagerank, &bfs] {
        let mut evidence = admission::AttemptCapabilityEvidence::for_query_plan(plan);
        evidence.attest_cugraph_graph_execution_scope(plan);
        let consumer = service
            .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))
            .expect("remaining capacity admits graph consumer");
        let consumer_context = consumer.acquire_context().expect("consumer context");
        let result = execute_plan(consumer_context.context(), plan, Some(&inputs))
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
    let deadline = Instant::now() + Duration::from_secs(10);
    while service.device_ledger_reports().unwrap().remove(0).frozen_resident_bytes != 0 {
        assert!(Instant::now() < deadline, "resident charge releases after final owner");
        thread::yield_now();
    }
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).unwrap();
}

#[test]
fn graph_gpu_input_from_another_admitted_attempt_borrows_the_resident_domain_without_copy() {
    let service = try_guaranteed_runtime_test_service_with_concurrency(2)
        .expect("runtime-backed service with two same-device attempt lanes");
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("src", ArrowDataType::Int32, false),
        ArrowField::new("dst", ArrowDataType::Int32, false),
    ]));
    let plan = pagerank_plan(Arc::clone(&edge_schema));
    let mut evidence = admission::AttemptCapabilityEvidence::for_query_plan(&plan);
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
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 0, 3])),
        ],
    )
    .expect("valid edge batch");
    let producer_input = runtime::resident::GpuResultHandle::new(
        producer_context.context(),
        runtime::resident::GpuDataFrame::new(
            CudfDataFrame::from_record_batch(producer_runtime.as_ref(), &edge_batch).expect("producer edge upload"),
        ),
        Arc::clone(&edge_schema),
    )
    .expect("producer GPU input handle");
    let producer_usage_before = producer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("producer domain usage")
        .expect("producer domain usage is present");
    assert!(producer_usage_before.current_bytes > 0, "producer input must be retained in its own domain");

    let mut inputs = runtime::resident::GpuResidentInputMap::new();
    inputs.insert("edges".to_string(), producer_input);
    let same_attempt =
        execute_plan(producer_context.context(), &plan, Some(&inputs)).expect("same-attempt zero-copy graph execution");
    assert_eq!(same_attempt.metrics().boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(same_attempt.metrics().boundary.device_to_device_copies, 0);
    assert_eq!(same_attempt.metrics().boundary.device_to_device_copied_bytes, 0);
    drop(same_attempt);

    let consumer_usage_before = consumer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("consumer domain usage")
        .expect("consumer domain usage is present");
    let result = execute_plan(consumer_context.context(), &plan, Some(&inputs))
        .expect("graph execution after the source-domain handoff");
    assert_eq!(result.metrics().boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(result.metrics().boundary.device_to_device_copies, 0);
    assert_eq!(result.metrics().graph_algorithm_runtime_rows.len(), 1, "the graph runs from the resident borrow");
    let exec::output::QueryOutput::HostArrow(output) = result.output() else {
        panic!("graph plan must materialize its result at the declared host sink");
    };
    assert_eq!(output.num_rows(), 4, "PageRank must see the copied four-vertex edge list");
    let consumer_usage_after = consumer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("consumer domain usage after execution")
        .expect("consumer domain usage is present");
    assert_eq!(
        consumer_usage_after.current_bytes, consumer_usage_before.current_bytes,
        "resident input bytes must not remain charged to the consuming attempt domain"
    );
    let producer_usage_after = producer_context
        .context()
        .attempt_allocation_domain_usage()
        .expect("producer domain usage after execution")
        .expect("producer domain usage is present");
    assert_eq!(
        producer_usage_after.current_bytes, producer_usage_before.current_bytes,
        "borrowing the resident input must not allocate into the producer attempt domain"
    );

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
        .close_and_wait_for_quiescence(Duration::from_secs(10))
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
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("src", ArrowDataType::Int32, false),
        ArrowField::new("dst", ArrowDataType::Int32, false),
    ]));
    let plan = pagerank_plan(Arc::clone(&edge_schema));
    let mut evidence = admission::AttemptCapabilityEvidence::for_query_plan(&plan);
    evidence.attest_cugraph_graph_execution_scope(&plan);
    let backend_lifecycle = runtime::BackendLifecycle::new();
    let counters = Arc::new(attempt::AttemptBackendCounters::default());
    let observed_attempt = attempt::AttemptOwner::begin(
        attempt::BeginAttemptRequest {
            observation: observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..attempt::BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::clone(&counters),
        None,
    );
    let attempt_owner = observed_attempt.owner;
    let completion = observed_attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(observability::AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let attempt = service
        .admit_query_with_request(
            QueryAdmissionRequest::new()
                .with_attempt_binding(attempt_owner.admission_binding())
                .with_query_plan_capability_evidence(evidence),
        )
        .expect("admission");
    attempt_handle
        .record_admitted(observability::AttemptAdmissionFact {
            admission_sequence: attempt.admission_sequence(),
            admission_wait_ns: attempt.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: attempt.device_ordinal(),
            grant_bytes: attempt.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    let admitted = attempt.acquire_context().expect("admitted context");
    let context = admitted.context();
    let runtime = context.runtime().expect("attempt runtime");
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 0, 3])),
        ],
    )
    .expect("valid edge batch");
    let descriptor = plan::GraphResidencyDescriptor::for_content_identity(
        "accounted-pagerank-graph-v1",
        ArrowDataType::Int32,
        None,
        None,
    )
    .expect("stable graph source identity");
    let input = runtime::resident::GpuResultHandle::new(
        context,
        runtime::resident::GpuDataFrame::new(
            CudfDataFrame::from_record_batch(&runtime, &edge_batch).expect("attempt edge upload"),
        )
        .with_graph_residency(descriptor),
        Arc::clone(&edge_schema),
    )
    .expect("resident graph input");
    let mut inputs = runtime::resident::GpuResidentInputMap::new();
    inputs.insert("edges".to_string(), input);
    let result = execute_plan(context, &plan, Some(&inputs))
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
        .commit_terminal(observability::RuntimeTerminalOutcome::Succeeded)
        .expect("primary attempt outcome commits before release");
    attempt.finish();
    drop(attempt);
    let sealed = completion.wait_timeout(Duration::from_secs(10)).expect("sealed attempt report");
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
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains cache entries");
    let drained_report = service.device_ledger_reports().expect("drained ledger report").remove(0);
    assert_eq!(drained_report.cache_evictable_resident_bytes, 0);
    assert_eq!(drained_report.cache_pinned_in_use_bytes, 0);
}

/// A warm resident graph executes inside the borrowing attempt domain: the
/// borrower stream waits on the park event while workspace and outputs
/// allocate through the attempt binding, so they charge to the borrower's
/// grant and never to the cache share.
///
/// The capped attempt proves this by grant arithmetic rather than peak
/// ordering. A cold attempt peak cannot serve as the cap: the cold path
/// stages cross-domain copies the warm hit never repeats, so the cold
/// high-water structurally exceeds the warm one and a cold-sized grant would
/// fund the workspace too. Instead the cap covers exactly the warm attempt's
/// non-workspace footprint; the edge upload plus one full output row-set;
/// with wide margins on both sides. The capped run must then clear the
/// upload and fail with a typed query-domain capacity error inside engine
/// execution; had the workspace charged the cache share instead, the attempt
/// would never exceed the cap and the run would succeed. The graph stays
/// reusable with identical results afterwards.
#[test]
fn warm_graph_executes_in_attempt_domain_and_survives_failed_cleanup() {
    const CACHE_CAP_BYTES: u64 = 16 * 1024 * 1024;
    const VERTEX_COUNT: i32 = 20_000;

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
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("src", ArrowDataType::Int32, false),
        ArrowField::new("dst", ArrowDataType::Int32, false),
    ]));
    let plan = pagerank_plan(Arc::clone(&edge_schema));
    // Deterministic 2-regular-ish digraph: a directed cycle plus long chords.
    // PageRank workspace holds several vertex-sized vectors live at once,
    // which dominates the edge upload (two int32 columns) and the output copy
    // (one float column) once the vertex count is large.
    let (src, dst): (Vec<i32>, Vec<i32>) = (0..2 * VERTEX_COUNT)
        .map(|edge| {
            let vertex = edge % VERTEX_COUNT;
            if edge < VERTEX_COUNT {
                (vertex, (vertex + 1) % VERTEX_COUNT)
            } else {
                (vertex, (vertex + VERTEX_COUNT / 2 + 1) % VERTEX_COUNT)
            }
        })
        .unzip();
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(src)),
            Arc::new(Int32Array::from(dst)),
        ],
    )
    .expect("valid edge batch");

    let cache_evictable_bytes = || {
        service.device_ledger_reports().expect("ledger report").remove(0).cache_evictable_resident_bytes
    };

    // Cold execution parks the graph; the attempt domain covers only the edge
    // upload and the output copy because the build runs in the cache domain.
    let (cold_batch, cold_peak, _) =
        execute_pagerank_attempt(&service, &plan, &edge_schema, &edge_batch, "attempt-domain-pagerank-graph-v1", None)
            .expect("cold pagerank execution");
    let parked_bytes = cache_evictable_bytes();
    assert!(parked_bytes > 0, "the cold run parks the graph in the cache share");
    assert!(cold_peak > 0, "the cold attempt measures its own upload and copy");

    // The cap sits strictly between the warm attempt's non-workspace
    // footprint and its workspace-inclusive footprint. It funds the edge
    // upload plus one and a half PageRank output row-sets (int32 vertex plus
    // float64 rank per vertex); headroom for the upload, the output, and
    // handle creation; but not the multi-vector algorithm workspace that
    // stays live beside the upload during execution. The capped hit must
    // therefore clear the upload and fail inside engine execution with a
    // typed query-domain capacity error, never by borrowing cache capacity,
    // leaving the parked bytes untouched. The strict separation also keeps
    // the test sensitive to routing regressions: were the workspace charged
    // anywhere but the attempt grant, the capped run would succeed and the
    // expectation below would fail.
    let capped_grant_bytes =
        edge_batch.get_array_memory_size() as u64 + VERTEX_COUNT as u64 * (4 + 8) * 3 / 2;
    let tiny_error = execute_pagerank_attempt(
        &service,
        &plan,
        &edge_schema,
        &edge_batch,
        "attempt-domain-pagerank-graph-v1",
        Some(capped_grant_bytes),
    )
    .expect_err("workspace-sized grant cannot fund warm-graph workspace");
    assert_eq!(
        tiny_error.code(),
        error::QueryErrorCode::QueryMemoryLimitExceeded,
        "unexpected capped-grant failure: {tiny_error:?}"
    );
    assert!(
        !tiny_error.operation().contains("upload_edges"),
        "the capped attempt must clear the edge upload and fail inside engine execution: {}",
        tiny_error.operation()
    );
    assert_eq!(cache_evictable_bytes(), parked_bytes, "failed cleanup adds no cache charge");

    // The warm hit runs workspace and outputs in the attempt domain while the
    // parked bytes stay exactly where the cold run left them.
    let (warm_batch, _, _) =
        execute_pagerank_attempt(&service, &plan, &edge_schema, &edge_batch, "attempt-domain-pagerank-graph-v1", None)
            .expect("warm pagerank execution after failed cleanup");
    assert_eq!(cache_evictable_bytes(), parked_bytes, "warm execution adds no cache-share charge");
    assert_eq!(warm_batch, cold_batch, "results are identical before and after the failed cleanup");

    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains cache entries");
    assert_eq!(cache_evictable_bytes(), 0);
}

/// Two different resident graphs execute in genuinely overlapping attempts
/// under default caps: a PageRank attempt and a BFS attempt over distinct
/// graphs are barrier-released together, and the test observes both grants
/// live at the same instant instead of inferring overlap from timing. Both
/// attempts use the service default cap; no per-query tuning; and must
/// succeed with results identical to their sequential baselines, in disjoint
/// allocation domains, without adding any cache-share charge.
#[test]
fn mixed_graph_attempts_overlap_with_independent_domains_and_parity() {
    const CACHE_CAP_BYTES: u64 = 16 * 1024 * 1024;
    const VERTEX_COUNT: i32 = 20_000;
    const PAGERANK_GRAPH_IDENTITY: &str = "overlap-mixed-pagerank-graph-v1";
    const BFS_GRAPH_IDENTITY: &str = "overlap-mixed-bfs-graph-v1";

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
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("src", ArrowDataType::Int32, false),
        ArrowField::new("dst", ArrowDataType::Int32, false),
    ]));
    let pagerank = pagerank_plan(Arc::clone(&edge_schema));
    let bfs = bfs_plan(Arc::clone(&edge_schema));
    // Same directed-cycle shape, different chords, so the two graphs have
    // distinct content while staying comparable in size.
    let pagerank_edges = cycle_chord_edge_batch(&edge_schema, VERTEX_COUNT, VERTEX_COUNT / 2 + 1);
    let bfs_edges = cycle_chord_edge_batch(&edge_schema, VERTEX_COUNT, VERTEX_COUNT / 3 + 1);

    let cache_evictable_bytes = || {
        service.device_ledger_reports().expect("ledger report").remove(0).cache_evictable_resident_bytes
    };
    let live_grants = || {
        service.device_ledger_reports().expect("ledger report").remove(0).grants.len()
    };

    // Sequential baselines park both graphs and fix the expected outputs.
    let (pagerank_baseline, _, _) =
        execute_pagerank_attempt(&service, &pagerank, &edge_schema, &pagerank_edges, PAGERANK_GRAPH_IDENTITY, None)
            .expect("sequential pagerank baseline");
    let (bfs_baseline, _, _) =
        execute_pagerank_attempt(&service, &bfs, &edge_schema, &bfs_edges, BFS_GRAPH_IDENTITY, None)
            .expect("sequential bfs baseline");
    let parked_bytes = cache_evictable_bytes();
    assert!(parked_bytes > 0, "both sequential runs park their graphs in the cache share");

    // Release both warm attempts at the same barrier instant, then observe
    // the ledger until both grants are live together or both attempts finish.
    // A grant spans admission to release with execution in between, and the
    // barrier holds pre-admission skew to microseconds against executions
    // that each take orders of magnitude longer, so a two-grant observation
    // lands inside genuine execution overlap rather than timer luck.
    let start = Barrier::new(3);
    let mut overlapped = false;
    let (pagerank_outcome, bfs_outcome) = thread::scope(|scope| {
        let pagerank_worker = scope.spawn(|| {
            start.wait();
            execute_pagerank_attempt(&service, &pagerank, &edge_schema, &pagerank_edges, PAGERANK_GRAPH_IDENTITY, None)
        });
        let bfs_worker = scope.spawn(|| {
            start.wait();
            execute_pagerank_attempt(&service, &bfs, &edge_schema, &bfs_edges, BFS_GRAPH_IDENTITY, None)
        });
        start.wait();
        let deadline = Instant::now() + Duration::from_secs(60);
        while !(pagerank_worker.is_finished() && bfs_worker.is_finished()) {
            if live_grants() >= 2 {
                overlapped = true;
                break;
            }
            assert!(Instant::now() < deadline, "overlapping attempts did not finish in time");
            thread::sleep(Duration::from_millis(1));
        }
        (
            pagerank_worker.join().expect("pagerank worker does not panic"),
            bfs_worker.join().expect("bfs worker does not panic"),
        )
    });
    assert!(overlapped, "both mixed-graph attempts must be live on the device at the same instant");

    let (pagerank_batch, _, pagerank_domain) = pagerank_outcome.expect("overlapped pagerank succeeds");
    let (bfs_batch, _, bfs_domain) = bfs_outcome.expect("overlapped bfs succeeds");
    assert_eq!(pagerank_batch, pagerank_baseline, "overlapped pagerank matches its sequential baseline");
    assert_eq!(bfs_batch, bfs_baseline, "overlapped bfs matches its sequential baseline");
    assert_ne!(pagerank_domain, bfs_domain, "overlapping attempts must not share an allocation domain");
    assert!(pagerank_domain.is_some() && bfs_domain.is_some(), "both attempts report their allocation domain");
    assert_eq!(cache_evictable_bytes(), parked_bytes, "overlapped warm execution adds no cache-share charge");

    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains cache entries");
    assert_eq!(cache_evictable_bytes(), 0);
}

/// Builds a deterministic two-edge-per-vertex digraph: a directed cycle plus
/// chords of the given offset. Shared by the mixed-overlap graphs so their
/// sizes stay comparable while their contents differ.
fn cycle_chord_edge_batch(
    edge_schema: &Arc<ArrowSchema>,
    vertex_count: i32,
    chord_offset: i32,
) -> RecordBatch {
    let (src, dst): (Vec<i32>, Vec<i32>) = (0..2 * vertex_count)
        .map(|edge| {
            let vertex = edge % vertex_count;
            if edge < vertex_count {
                (vertex, (vertex + 1) % vertex_count)
            } else {
                (vertex, (vertex + chord_offset) % vertex_count)
            }
        })
        .unzip();
    RecordBatch::try_new(
        Arc::clone(edge_schema),
        vec![
            Arc::new(Int32Array::from(src)),
            Arc::new(Int32Array::from(dst)),
        ],
    )
    .expect("valid edge batch")
}

/// Admits one graph attempt with an optional explicit grant, executes it, and
/// returns the host output batch with the attempt-domain peak and allocation
/// domain id. Only the execution outcome is returned; every ceremony step
/// uses expect like the surrounding tests. The attempt is fully finished and
/// released before returning.
fn execute_pagerank_attempt(
    service: &runtime::QueryService,
    plan: &plan::QueryPlan,
    edge_schema: &Arc<ArrowSchema>,
    edge_batch: &RecordBatch,
    graph_identity: &str,
    cap_bytes: Option<u64>,
) -> error::Result<(RecordBatch, u64, Option<u64>)> {
    let mut evidence = admission::AttemptCapabilityEvidence::for_query_plan(plan);
    evidence.attest_cugraph_graph_execution_scope(plan);
    let backend_lifecycle = runtime::BackendLifecycle::new();
    let counters = Arc::new(attempt::AttemptBackendCounters::default());
    let observed_attempt = attempt::AttemptOwner::begin(
        attempt::BeginAttemptRequest {
            observation: observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..attempt::BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::clone(&counters),
        None,
    );
    let attempt_owner = observed_attempt.owner;
    let completion = observed_attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(observability::AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let mut request = QueryAdmissionRequest::new()
        .with_attempt_binding(attempt_owner.admission_binding())
        .with_query_plan_capability_evidence(evidence);
    if let Some(cap_bytes) = cap_bytes {
        request = request.with_query_cap_bytes(cap_bytes);
    }
    let attempt = service.admit_query_with_request(request).expect("attempt admits");
    attempt_handle
        .record_admitted(observability::AttemptAdmissionFact {
            admission_sequence: attempt.admission_sequence(),
            admission_wait_ns: attempt.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: attempt.device_ordinal(),
            grant_bytes: attempt.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    let admitted = attempt.acquire_context().expect("admitted context");
    let context = admitted.context();
    let runtime = context.runtime().expect("attempt runtime");
    // Edge upload already charges the attempt domain: an undersized grant
    // fails here with the same typed capacity error as workspace exhaustion.
    let result = (|| {
        let input = runtime::resident::GpuResultHandle::new(
            context,
            runtime::resident::GpuDataFrame::new(
                CudfDataFrame::from_record_batch(&runtime, edge_batch).map_err(|error| {
                    native_error::cudf_query_error(
                        "query_engine.graph_algorithm.attempt_domain.upload_edges",
                        "failed to upload warm-graph test edges into the attempt domain",
                        error,
                    )
                })?,
            )
            .with_graph_residency(
                plan::GraphResidencyDescriptor::for_content_identity(
                    graph_identity,
                    ArrowDataType::Int32,
                    None,
                    None,
                )
                .expect("stable graph source identity"),
            ),
            Arc::clone(edge_schema),
        )?;
        let mut inputs = runtime::resident::GpuResidentInputMap::new();
        inputs.insert("edges".to_string(), input);
        execute_plan(context, plan, Some(&inputs))
    })();
    let peak_bytes =
        context.attempt_allocation_domain_usage().expect("attempt domain usage").map(|usage| usage.peak_bytes).unwrap_or(0);
    let domain_id = context.attempt_allocation_domain_id();
    drop(runtime);
    drop(admitted);
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");
    if result.is_ok() {
        attempt_owner
            .commit_terminal(observability::RuntimeTerminalOutcome::Succeeded)
            .expect("terminal outcome commits");
        attempt.finish();
        drop(attempt);
        completion.wait_timeout(Duration::from_secs(10)).expect("sealed attempt report");
    } else {
        attempt.finish();
        drop(attempt);
    }
    let result = result?;
    let exec::output::QueryOutput::HostArrow(batch) = result.output() else {
        panic!("PageRank host sink should materialize Arrow output");
    };
    Ok((batch.clone(), peak_bytes, domain_id))
}

/// Writes a multi-row-group Parquet file whose columns include a nullable
/// string. Per-chunk string offsets and null-mask padding make the sum of the
/// chunk reservations larger than the concatenated output, so the executor's
/// retained charge and the allocator's measurement disagree by construction.
fn write_multi_row_group_string_parquet(path: &Path) -> Arc<ArrowSchema> {
    use arrow::datatypes::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![
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
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(src)),
            Arc::new(Int32Array::from(dst)),
            Arc::new(StringArray::from(label)),
            Arc::new(Int64Array::from(weight)),
        ],
    )
    .expect("multi row-group fixture batch");
    let properties =
        parquet_properties::WriterProperties::builder().set_max_row_group_row_count(Some(512)).build();
    let file = fs::File::create(path).expect("fixture parquet file");
    let mut writer = parquet_arrow::ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties))
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

    let dir = env::temp_dir().join(format!(
        "algeon-resident-freeze-{}-{}",
        process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default()
    ));
    fs::create_dir_all(&dir).expect("fixture directory");
    let path = dir.join("edges.parquet");
    let schema = write_multi_row_group_string_parquet(&path);

    let file_size_bytes = fs::metadata(&path).expect("fixture parquet metadata").len();
    let source = source::LocalParquetSource::from_files(
        vec![
            source::LocalParquetFile::new(path.to_str().expect("utf-8 fixture path"))
                .with_stable_id(format!("local:{}:{file_size_bytes}", path.display()))
                .with_file_size_bytes(file_size_bytes)
                .with_row_group_rows(vec![512; 8]),
        ],
        Arc::clone(&schema),
        vec!["src".to_string(), "dst".to_string(), "label".to_string(), "weight".to_string()],
    )
    // A scan limit keeps each planned row group as its own read, so the
    // resident output is a concatenation of several chunks rather than one
    // merged read.
    .with_scan_limit(Some(4_096));
    let mut builder = plan::QueryPlanBuilder::new();
    let scan = builder.add_source(source::SourceNode::LocalParquet(source));
    let plan = builder.finish_gpu_resident(scan);

    let lifecycle = runtime::BackendLifecycle::new();
    let counters = Arc::new(attempt::AttemptBackendCounters::default());
    let observed = attempt::AttemptOwner::begin(
        attempt::BeginAttemptRequest {
            observation: observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..attempt::BeginAttemptRequest::default()
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
    attempt_observation.set_disposition(observability::AttemptDisposition::Native).unwrap();
    attempt_observation.record_admission_requested().unwrap();
    let evidence = admission::AttemptCapabilityEvidence::for_query_plan(&plan);
    let producer = service
        .admit_query_with_request(
            QueryAdmissionRequest::new()
                .with_attempt_binding(attempt_owner.admission_binding())
                .with_query_plan_capability_evidence(evidence),
        )
        .expect("producer admission");
    attempt_observation
        .record_admitted(observability::AttemptAdmissionFact {
            admission_sequence: producer.admission_sequence(),
            admission_wait_ns: producer.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: producer.device_ordinal(),
            grant_bytes: producer.device_budget_bytes(),
        })
        .unwrap();
    attempt_observation.record_physical_execution_started().unwrap();
    attempt_observation.record_gpu_execution_started().unwrap();
    let producer_context = producer.acquire_context().expect("producer context");
    let result = execute_plan(producer_context.context(), &plan, None)
        .expect("multi row-group resident scan executes");
    let (output, _) = result.into_parts();
    let resident = output.as_gpu_resident().expect("GPU-resident sink produces a resident output owner");
    let resident_rows = resident.to_host_record_batch().expect("resident output is readable").num_rows();
    assert_eq!(resident_rows, 4_096);

    drop(producer_context);
    attempt_owner.commit_terminal(observability::RuntimeTerminalOutcome::Succeeded).unwrap();
    producer.finish();
    drop(producer);

    let report = completion.wait_timeout(Duration::from_secs(10)).expect("producer report");
    assert_eq!(
        report.capacity_disposition,
        observability::AttemptCapacityDisposition::FrozenResident,
        "an executor-produced resident output must freeze, not quarantine the device"
    );
    assert_eq!(report.cleanup_failure_count, 0);

    // The B-to-R publish runs after the attempt lane is released, so the
    // frozen charge becomes visible strictly after the sealed report.
    let deadline = Instant::now() + Duration::from_secs(10);
    let snapshot = loop {
        let snapshot = service.device_ledger_reports().unwrap().remove(0);
        if snapshot.active_attempts == 0 && snapshot.frozen_resident_bytes > 0 {
            break snapshot;
        }
        assert!(Instant::now() < deadline, "the frozen output publishes a measured resident charge");
        thread::yield_now();
    };
    assert_eq!(snapshot.live_grants_bytes, 0);
    assert!(snapshot.frozen_resident_bytes < CAPACITY_BYTES);

    drop(output);
    let _ = fs::remove_dir_all(&dir);
}
