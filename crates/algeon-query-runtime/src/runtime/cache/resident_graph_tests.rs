use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
use cugraph::cudf_interop::{
    GraphConstructionPolicy as CugraphConstructionPolicy, TableGraphOptions, graph_from_prepared_table_export,
};
use query_planning::{
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
        GraphConstructionPolicy, GraphEdgeColumnBindings, GraphResidencyDescriptor,
    },
    source::LocalParquetSource,
};

use super::*;
use crate::runtime::{
    ExecutionContext,
    cache::{
        DeviceCacheAllocationFactory,
        local_parquet::{GpuLocalParquetSourceCacheKey, GpuLocalParquetSourceCachePromotion},
    },
    gpu::{DeviceAdmissionHealth, DeviceCapabilities, DeviceResourceEnvelope, DeviceResourceProfile},
    resident::{GpuDataFrame, GpuResultHandle},
};

const CACHE_CAP_BYTES: u64 = 64 * 1024 * 1024;

fn cache_test_parts() -> (GpuResidentGraphCache, DeviceLedger) {
    let device = cudf::DeviceId::current().expect("resident graph cache test requires a CUDA device");
    let profile = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
        device_ordinal: device.ordinal(),
        configured_limit_bytes: 512 * 1024 * 1024,
        headroom_bytes: 0,
        backend_reserve_bytes: 0,
        default_query_limit_bytes: 1024 * 1024,
        max_active_attempts: 1,
        source_work_per_attempt: 1,
        stream_pool_size: 2,
        worker_job_capacity: 2,
        device_capabilities: DeviceCapabilities::default(),
        cache_cap_bytes: CACHE_CAP_BYTES,
    })
    .expect("cache test profile must resolve");
    let ledger = DeviceLedger::new(profile);
    let domain = cudf::CacheAllocationDomain::new(device, CACHE_CAP_BYTES)
        .expect("cache test allocation domain must be created");
    let cache = GpuResidentGraphCache::new(ledger.clone(), domain).expect("graph cache must use shared domain");
    (cache, ledger)
}

fn resident_key() -> ResidentGraphKey {
    resident_key_for_identity("resident-graph-cache-test-v1")
}

fn resident_key_for_identity(identity: &str) -> ResidentGraphKey {
    let descriptor = GraphResidencyDescriptor::for_content_identity(identity, DataType::Int32, None, None)
        .expect("test identity is stable");
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    descriptor.resident_key(&spec, GraphResidencyOrientation::Forward)
}

fn insert_forward_graph(cache: &GpuResidentGraphCache, key: ResidentGraphKey) -> u64 {
    let reservation_bytes = cache.available_reservation_bytes().expect("available graph cache reservation");
    assert!(reservation_bytes > 0, "resident graph insertion must have cache headroom");
    let transaction = cache
        .begin_insert(reservation_bytes)
        .expect("cache pre-reservation succeeds")
        .expect("cache cap accepts the graph charge");
    let parked = build_forward_graph(&transaction);
    let mut transaction = transaction;
    assert!(transaction.shrink_to_scope_usage().expect("measure retained graph bytes"));
    let retained_graph_bytes = transaction.bytes();
    assert!(retained_graph_bytes > 0, "parked graph must retain measured cache-domain bytes");
    assert_eq!(transaction.insert_if_absent(key, parked).expect("retain graph"), GpuResidentGraphCacheInsert::Inserted);
    retained_graph_bytes
}

fn build_forward_graph(transaction: &GpuResidentGraphInsertTransaction) -> cugraph::ParkedGraph<cugraph::Forward> {
    let runtime = transaction.runtime();
    let handle = cugraph::ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("cache runtime gets a handle");
    let entered = handle.enter("query_engine.resident_graph_cache_test.build").expect("enter cache runtime");
    let src = runtime.series(&[0_i32, 1, 2]).expect("source series");
    let dst = runtime.series(&[1_i32, 2, 0]).expect("destination series");
    let edges = DataFrame::from_series([("src", src), ("dst", dst)]).expect("edge frame");
    let export = edges.try_prepare_device_table_export().expect("prepared edge export");
    let options = TableGraphOptions::new(0, 1).renumber().construction_policy(CugraphConstructionPolicy::RawLibcugraph);
    let graph = graph_from_prepared_table_export::<cugraph::Forward>(&entered, &export, &options).expect("build graph");
    let parked = graph.park(&entered).expect("park graph");
    drop(entered);
    drop(handle);
    drop(edges);
    parked
}

fn source_fixture() -> (Arc<ExecutionContext>, GpuResultHandle, GpuLocalParquetSourceCacheKey) {
    let runtime = Arc::new(cudf::Runtime::new().expect("test CUDA runtime"));
    let context = Arc::new(
        ExecutionContext::builder().with_shared_runtime(Arc::clone(&runtime)).build().expect("test execution context"),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from_iter_values(0..4096))])
        .expect("test Arrow batch");
    let frame = DataFrame::from_record_batch(runtime.as_ref(), &batch).expect("test GPU frame");
    let handle = GpuResultHandle::new(context.as_ref(), GpuDataFrame::new(frame), Arc::clone(&schema))
        .expect("test GPU result handle");
    let source = LocalParquetSource::new("/tmp/algeon-shared-cache-test.parquet", schema, vec!["value".into()]);
    (context, handle, GpuLocalParquetSourceCacheKey::from_source(&source))
}

#[test]
fn separate_resident_graph_scopes_keep_distinct_current_resource_routes() {
    let (cache, ledger) = cache_test_parts();
    let first_bytes = insert_forward_graph(&cache, resident_key_for_identity("resident-graph-cache-route-first"));
    let second_bytes = insert_forward_graph(&cache, resident_key_for_identity("resident-graph-cache-route-second"));

    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.cache_evictable_resident_bytes, first_bytes + second_bytes);
    assert_eq!(cache.snapshot().unwrap().available_entries, 2);

    cache.close_and_drain_until(Instant::now() + Duration::from_secs(10)).expect("drain independent graph scopes");
    let drained = ledger.snapshot().unwrap();
    assert_eq!(drained.cache_evictable_resident_bytes, 0);
    assert_eq!(drained.cache_pinned_in_use_bytes, 0);
}

#[test]
fn graph_scope_charge_survives_source_cache_eviction_between_build_and_commit() {
    let device = cudf::DeviceId::current().expect("shared cache test requires a CUDA device");
    let profile = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
        device_ordinal: device.ordinal(),
        configured_limit_bytes: 512 * 1024 * 1024,
        headroom_bytes: 0,
        backend_reserve_bytes: 0,
        default_query_limit_bytes: 1024 * 1024,
        max_active_attempts: 1,
        source_work_per_attempt: 1,
        stream_pool_size: 2,
        worker_job_capacity: 2,
        device_capabilities: DeviceCapabilities::default(),
        cache_cap_bytes: CACHE_CAP_BYTES,
    })
    .expect("shared cache test profile must resolve");
    let ledger = DeviceLedger::new(profile);
    let factory = DeviceCacheAllocationFactory::new(ledger.clone())
        .expect("cache factory must be created")
        .expect("non-zero cache cap creates a factory");
    let source_cache = factory.source_cache().expect("source cache lease");
    let graph_cache = factory.graph_cache().expect("graph cache lease");
    let (context, source, source_key) = source_fixture();
    assert_eq!(
        source_cache.promote(source_key, context.as_ref(), &source).expect("populate source cache").outcome,
        GpuLocalParquetSourceCachePromotion::Inserted
    );

    let reservation_bytes = graph_cache.available_reservation_bytes().expect("remaining graph reservation");
    assert!(reservation_bytes > 0, "source cache must leave graph insertion headroom");
    let transaction = graph_cache
        .begin_insert(reservation_bytes)
        .expect("begin graph insertion")
        .expect("graph insertion precharge fits");
    let parked = build_forward_graph(&transaction);

    // This is the former aggregate-delta hole: source residency disappears
    // after graph construction but before graph charge reconciliation. The
    // graph scope is allocator-owned and excludes that source mutation.
    source_cache.close_and_drain().expect("evict source cache between graph build and commit");

    let mut transaction = transaction;
    assert!(transaction.shrink_to_scope_usage().expect("measure exact graph scope bytes"));
    let graph_bytes = transaction.bytes();
    assert!(graph_bytes > 0);
    assert_eq!(
        transaction.insert_if_absent(resident_key(), parked).expect("commit graph cache entry"),
        GpuResidentGraphCacheInsert::Inserted
    );

    let report = factory.report().expect("coherent paired cache report");
    assert_eq!(report.ledger_evictable_resident_bytes, graph_bytes);
    assert_eq!(report.ledger_pinned_in_use_bytes, 0);
    assert_eq!(report.physical_current_bytes, graph_bytes);

    graph_cache.close_and_drain_until(Instant::now() + Duration::from_secs(10)).expect("drain graph cache");
    let drained = factory.report().expect("coherent drained cache report");
    assert_eq!(drained.ledger_evictable_resident_bytes, 0);
    assert_eq!(drained.ledger_pinned_in_use_bytes, 0);
    assert_eq!(drained.physical_current_bytes, 0);
}

#[test]
fn empty_graph_candidate_abort_releases_precharge_only_after_scope_proof() {
    let (cache, ledger) = cache_test_parts();
    let transaction = cache
        .begin_insert(CACHE_CAP_BYTES)
        .expect("cache pre-reservation succeeds")
        .expect("cache cap accepts the graph charge");
    let reserved_bytes = transaction.bytes();
    assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, reserved_bytes);

    transaction.abort_after_candidate_destroyed().expect("empty scope proves candidate teardown");

    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.cache_evictable_resident_bytes, 0);
    assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
    assert_eq!(snapshot.health, DeviceAdmissionHealth::Open);
}

#[test]
fn live_scope_candidate_drop_retains_precharge_and_closes_admission() {
    let (cache, ledger) = cache_test_parts();
    let transaction = cache
        .begin_insert(CACHE_CAP_BYTES)
        .expect("cache pre-reservation succeeds")
        .expect("cache cap accepts the graph charge");
    let reserved_bytes = transaction.bytes();

    // Keep a scope-routed allocation alive across the transaction drop. The
    // destructor must see that its candidate teardown is unproven and leave
    // the prospective ledger charge in place rather than releasing it.
    let runtime = transaction.runtime();
    let live_candidate_allocation = runtime.device_buffer(1024).expect("allocate a live candidate object");
    drop(transaction);

    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.cache_evictable_resident_bytes, reserved_bytes);
    assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
    assert!(matches!(snapshot.health, DeviceAdmissionHealth::Closed { .. }));

    drop(live_candidate_allocation);
    runtime.synchronize().expect("retire the test candidate allocation");
}

#[test]
fn resident_graph_refuses_final_preparation_pin_after_close_and_drains_charge() {
    let (cache, ledger) = cache_test_parts();
    let key = resident_key();
    let retained_graph_bytes = insert_forward_graph(&cache, key.clone());
    assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, retained_graph_bytes);

    let pin = cache
        .try_pin_available_key(&key)
        .expect("first preparation pin succeeds")
        .expect("resident graph is available");
    assert_eq!(ledger.snapshot().unwrap().cache_pinned_in_use_bytes, retained_graph_bytes);
    cache.close().expect("close cache");
    assert!(
        cache.try_pin_available_key(&key).expect("closed cache reports no final pin").is_none(),
        "a final preparation pin must be refused once close begins"
    );

    drop(pin);
    cache.close_and_drain_until(Instant::now() + Duration::from_secs(10)).expect("drain graph after pin release");
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.cache_evictable_resident_bytes, 0);
    assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
}

#[test]
fn resident_graph_closed_before_an_untrusted_return_never_reopens_the_key() {
    let (cache, ledger) = cache_test_parts();
    let key = resident_key();
    let retained_graph_bytes = insert_forward_graph(&cache, key.clone());
    let mut checkout =
        cache.checkout::<cugraph::Forward>(&key, None).expect("checkout succeeds").expect("cached graph is present");
    assert_eq!(ledger.snapshot().unwrap().cache_pinned_in_use_bytes, retained_graph_bytes);
    let parked = checkout.take_parked();

    cache.close().expect("close cache");
    checkout.return_parked(parked).expect("closed return destroys graph safely");
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.cache_evictable_resident_bytes, 0);
    assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
    let cache_snapshot = cache.snapshot().unwrap();
    assert!(cache_snapshot.closed);
    assert_eq!(cache_snapshot.available_entries, 0);
    assert_eq!(cache_snapshot.checked_out_entries, 0);
    assert!(
        cache.checkout::<cugraph::Forward>(&key, None).is_err(),
        "a graph returned after cache closure must not become reusable"
    );
}
