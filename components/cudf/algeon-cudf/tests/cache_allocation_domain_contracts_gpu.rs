use cudf::{
    CacheAllocationDomain, CacheAllocationLease, CacheAllocationScope, DeviceId, DeviceMemoryManager, ErrorCode,
    ErrorKind,
    testing::{prelude::*, run_named_cases},
};
use static_assertions::assert_impl_all;

assert_impl_all!(CacheAllocationDomain: Clone, Send, Sync);
assert_impl_all!(CacheAllocationLease: Send, Sync);
assert_impl_all!(CacheAllocationScope: Send, Sync);

#[test]
fn zero_cache_capacity_cannot_create_an_allocation_domain() {
    let error = CacheAllocationDomain::new(DeviceId::DEFAULT, 0).unwrap_err();

    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
}

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

fn allocation_scopes_report_exact_operation_usage_with_shared_provenance() {
    let cache_cap_bytes = 16 * 1024 * 1024;
    let source_bytes = 4 * 1024 * 1024;
    let graph_bytes = 2 * 1024 * 1024;
    let domain = CacheAllocationDomain::new(DeviceId::current().unwrap(), cache_cap_bytes).unwrap();

    let source_scope = domain.allocation_scope().unwrap();
    let graph_scope = domain.allocation_scope().unwrap();

    assert_eq!(source_scope.domain_id(), domain.id());
    assert_eq!(graph_scope.domain_id(), domain.id());
    assert_eq!(source_scope.device(), domain.device());
    assert_eq!(graph_scope.device(), domain.device());
    assert_ne!(
        source_scope.runtime().stream().testing_raw_handle_value(),
        graph_scope.runtime().stream().testing_raw_handle_value(),
        "per-operation scopes must have distinct stream identities for independent current-resource routes"
    );
    assert_ne!(
        source_scope.runtime().binding_id(),
        graph_scope.runtime().binding_id(),
        "per-operation scopes must have distinct runtime bindings for independent current-resource routes"
    );
    assert_ne!(
        source_scope.memory_resource().testing_raw_handle_value(),
        graph_scope.memory_resource().testing_raw_handle_value(),
        "each operation must receive its own tracking resource"
    );
    assert_ne!(
        source_scope.memory_resource().testing_raw_handle_value(),
        domain.memory_resource().testing_raw_handle_value(),
        "the scope must track through a child of the shared hard-cap resource"
    );
    let memory_manager = DeviceMemoryManager::for_device(domain.device()).unwrap();
    let _source_current_resource_route =
        memory_manager.route_current_resource(source_scope.runtime().stream(), source_scope.memory_resource()).unwrap();
    let _graph_current_resource_route =
        memory_manager.route_current_resource(graph_scope.runtime().stream(), graph_scope.memory_resource()).unwrap();

    let source_allocation = source_scope.runtime().device_buffer(source_bytes).unwrap();
    let graph_allocation = graph_scope.runtime().device_buffer(graph_bytes).unwrap();
    source_scope.runtime().synchronize().unwrap();
    graph_scope.runtime().synchronize().unwrap();

    let source_usage = source_scope.usage().unwrap();
    assert_eq!(source_usage.allocation_limit_bytes, cache_cap_bytes);
    assert_eq!(source_usage.current_bytes, source_allocation.size_bytes().unwrap() as u64);
    assert_eq!(source_usage.current_allocations, 1);

    let graph_usage = graph_scope.usage().unwrap();
    assert_eq!(graph_usage.allocation_limit_bytes, cache_cap_bytes);
    assert_eq!(graph_usage.current_bytes, graph_allocation.size_bytes().unwrap() as u64);
    assert_eq!(graph_usage.current_allocations, 1);

    let domain_usage = domain.usage().unwrap();
    assert_eq!(
        domain_usage.current_bytes,
        source_allocation.size_bytes().unwrap() as u64 + graph_allocation.size_bytes().unwrap() as u64
    );
    assert_eq!(domain_usage.current_allocations, 2);

    drop(source_allocation);
    assert_eq!(source_scope.usage().unwrap().current_bytes, 0);
    assert_eq!(graph_scope.usage().unwrap().current_bytes, graph_allocation.size_bytes().unwrap() as u64);
}

fn graph_scope_usage_is_exact_when_parent_snapshots_span_another_scope_release() {
    let cache_cap_bytes = 16 * 1024 * 1024;
    let source_bytes = 4 * 1024 * 1024;
    let graph_bytes = 2 * 1024 * 1024;
    let domain = CacheAllocationDomain::new(DeviceId::current().unwrap(), cache_cap_bytes).unwrap();
    let source_scope = domain.allocation_scope().unwrap();
    let graph_scope = domain.allocation_scope().unwrap();

    let source_allocation = source_scope.runtime().device_buffer(source_bytes).unwrap();
    source_scope.runtime().synchronize().unwrap();
    let parent_before_graph_insert = domain.usage().unwrap();

    let graph_allocation = graph_scope.runtime().device_buffer(graph_bytes).unwrap();
    graph_scope.runtime().synchronize().unwrap();
    drop(source_allocation);

    let parent_after_source_release = domain.usage().unwrap();
    let parent_delta =
        i128::from(parent_after_source_release.current_bytes) - i128::from(parent_before_graph_insert.current_bytes);
    assert!(
        parent_delta < 0,
        "the parent delta includes source release as well as graph insertion: before={parent_before_graph_insert:?}, after={parent_after_source_release:?}"
    );
    assert_ne!(parent_delta, i128::from(graph_allocation.size_bytes().unwrap() as u64));

    assert_eq!(source_scope.usage().unwrap().current_bytes, 0);
    assert_eq!(graph_scope.usage().unwrap().current_bytes, graph_allocation.size_bytes().unwrap() as u64);

    drop(graph_allocation);
    assert_eq!(source_scope.usage().unwrap().current_bytes, 0);
    assert_eq!(graph_scope.usage().unwrap().current_bytes, 0);
    assert_eq!(domain.usage().unwrap().current_bytes, 0);
}

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

fn cache_cap_is_enforced_across_independently_tracked_scopes() {
    let cache_cap_bytes = 8 * 1024 * 1024;
    let domain = CacheAllocationDomain::new(DeviceId::current().unwrap(), cache_cap_bytes).unwrap();
    let source_scope = domain.allocation_scope().unwrap();
    let graph_scope = domain.allocation_scope().unwrap();

    let source_allocation = source_scope.runtime().device_buffer(5 * 1024 * 1024).unwrap();
    let error = match graph_scope.runtime().device_buffer(5 * 1024 * 1024) {
        Ok(_) => panic!("the shared cache cap must reject the second scope allocation"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    let source_usage = source_scope.usage().unwrap();
    assert_eq!(source_usage.current_bytes, source_allocation.size_bytes().unwrap() as u64);
    assert_eq!(source_usage.current_allocations, 1);
    let graph_usage = graph_scope.usage().unwrap();
    assert_eq!(graph_usage.current_bytes, 0);
    assert_eq!(graph_usage.current_allocations, 0);
    assert_eq!(graph_usage.failed_allocation_count, 1);

    let domain_usage = domain.usage().unwrap();
    assert_eq!(domain_usage.current_bytes, source_allocation.size_bytes().unwrap() as u64);
    assert_eq!(domain_usage.failed_allocation_count, 1);
}

#[test]
fn cache_allocation_domain_cases() {
    run_named_cases(&[
        (
            "cache_cap_is_enforced_across_independently_tracked_scopes",
            cache_cap_is_enforced_across_independently_tracked_scopes,
        ),
        (
            "source_and_graph_leases_share_one_domain_and_one_live_byte_counter",
            source_and_graph_leases_share_one_domain_and_one_live_byte_counter,
        ),
        (
            "allocation_scopes_report_exact_operation_usage_with_shared_provenance",
            allocation_scopes_report_exact_operation_usage_with_shared_provenance,
        ),
        (
            "graph_scope_usage_is_exact_when_parent_snapshots_span_another_scope_release",
            graph_scope_usage_is_exact_when_parent_snapshots_span_another_scope_release,
        ),
        ("cache_cap_is_enforced_across_independent_leases", cache_cap_is_enforced_across_independent_leases),
    ]);
}
