use cudf::{CacheAllocationDomain, CacheAllocationLease, DeviceId, ErrorCode, ErrorKind, testing::prelude::*};
use static_assertions::assert_impl_all;

assert_impl_all!(CacheAllocationDomain: Clone, Send, Sync);
assert_impl_all!(CacheAllocationLease: Send, Sync);

#[test]
fn zero_cache_capacity_cannot_create_an_allocation_domain() {
    let error = CacheAllocationDomain::new(DeviceId::DEFAULT, 0).unwrap_err();

    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
}

#[test]
fn source_and_graph_leases_share_one_domain_and_one_live_byte_counter() {
    let cache_cap_bytes = 16 * 1024 * 1024;
    let source_bytes = 4 * 1024 * 1024;
    let graph_bytes = 2 * 1024 * 1024;
    let domain = CacheAllocationDomain::new(DeviceId::current().unwrap(), cache_cap_bytes).unwrap();

    let source_cache = domain.lease_runtime().unwrap();
    let graph_cache = domain.lease_runtime().unwrap();

    assert_eq!(source_cache.domain_id(), graph_cache.domain_id());
    assert_eq!(source_cache.domain_id(), domain.id());
    assert_eq!(source_cache.domain().device(), graph_cache.domain().device());
    assert_eq!(
        source_cache.runtime().memory_resource().testing_raw_handle_value(),
        graph_cache.runtime().memory_resource().testing_raw_handle_value()
    );

    let source_allocation = source_cache.runtime().device_buffer(source_bytes).unwrap();
    let graph_allocation = graph_cache.runtime().device_buffer(graph_bytes).unwrap();
    source_cache.runtime().synchronize().unwrap();
    graph_cache.runtime().synchronize().unwrap();

    let measured_live_bytes =
        source_allocation.size_bytes().unwrap() as u64 + graph_allocation.size_bytes().unwrap() as u64;
    let usage = domain.usage().unwrap();
    assert_eq!(usage.allocation_limit_bytes, cache_cap_bytes);
    assert_eq!(usage.current_bytes, measured_live_bytes);
    assert_eq!(usage.current_allocations, 2);

    drop(source_allocation);
    let after_source_release = domain.usage().unwrap();
    assert_eq!(after_source_release.current_bytes, graph_allocation.size_bytes().unwrap() as u64);

    drop(graph_allocation);
    assert_eq!(domain.usage().unwrap().current_bytes, 0);
}

#[test]
fn cache_cap_is_enforced_across_independent_leases() {
    let cache_cap_bytes = 8 * 1024 * 1024;
    let domain = CacheAllocationDomain::new(DeviceId::current().unwrap(), cache_cap_bytes).unwrap();
    let source_cache = domain.lease_runtime().unwrap();
    let graph_cache = domain.lease_runtime().unwrap();

    let source_allocation = source_cache.runtime().device_buffer(5 * 1024 * 1024).unwrap();
    let error = match graph_cache.runtime().device_buffer(5 * 1024 * 1024) {
        Ok(_) => panic!("the shared cache cap must reject the second allocation"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    let usage = domain.usage().unwrap();
    assert_eq!(usage.current_bytes, source_allocation.size_bytes().unwrap() as u64);
    assert_eq!(usage.failed_allocation_count, 1);
}
