use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::datatypes::DataType;
use cugraph::cudf_interop::{SeriesGraphOptions, graph_from_series};

use super::*;
use crate::{
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
        GraphConstructionPolicy, GraphEdgeColumnBindings, GraphResidencyDescriptor,
    },
    runtime::gpu::{DeviceCapabilities, DeviceResourceProfile},
};

const CACHE_CAP_BYTES: u64 = 64 * 1024 * 1024;

fn cache_test_parts() -> (GpuResidentGraphCache, DeviceLedger) {
    let device = cudf::DeviceId::current().expect("resident graph cache test requires a CUDA device");
    let profile = DeviceResourceProfile::resolve_with_capabilities_and_cache_cap(
        device.ordinal(),
        512 * 1024 * 1024,
        0,
        0,
        1024 * 1024,
        1,
        1,
        2,
        2,
        DeviceCapabilities::default(),
        CACHE_CAP_BYTES,
    )
    .expect("cache test profile must resolve");
    let ledger = DeviceLedger::new(profile);
    let domain = cudf::CacheAllocationDomain::new(device, CACHE_CAP_BYTES)
        .expect("cache test allocation domain must be created");
    let cache = GpuResidentGraphCache::new(ledger.clone(), domain).expect("graph cache must use shared domain");
    (cache, ledger)
}

fn resident_key() -> ResidentGraphKey {
    let descriptor =
        GraphResidencyDescriptor::for_content_identity("resident-graph-cache-test-v1", DataType::Int32, None, None)
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
    let transaction = cache
        .begin_insert(CACHE_CAP_BYTES)
        .expect("cache pre-reservation succeeds")
        .expect("cache cap accepts the graph charge");
    let before = cache.allocation_usage().expect("read cache-domain usage before graph build");
    let runtime = Arc::clone(transaction.runtime());
    let handle = cugraph::ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("cache runtime gets a handle");
    let entered = handle.enter("nexus_query_engine.resident_graph_cache_test.build").expect("enter cache runtime");
    let src = runtime.series(&[0_i32, 1, 2]).expect("source series");
    let dst = runtime.series(&[1_i32, 2, 0]).expect("destination series");
    let graph =
        graph_from_series::<cugraph::Forward>(&entered, &src, &dst, None, &SeriesGraphOptions::default().renumber())
            .expect("build graph");
    let parked = graph.park(&entered).expect("park graph");
    drop(entered);
    drop(handle);
    drop(src);
    drop(dst);
    let mut transaction = transaction;
    assert!(transaction.shrink_to_domain_delta(before).expect("measure retained graph bytes"));
    let retained_graph_bytes = transaction.bytes();
    assert!(retained_graph_bytes > 0, "parked graph must retain measured cache-domain bytes");
    assert_eq!(transaction.insert_if_absent(key, parked).expect("retain graph"), GpuResidentGraphCacheInsert::Inserted);
    retained_graph_bytes
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
fn resident_graph_return_after_close_destroys_before_releasing_ledger_charge() {
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
    assert_eq!(cache.snapshot().unwrap().available_entries, 0);
    assert_eq!(cache.snapshot().unwrap().checked_out_entries, 0);
}
