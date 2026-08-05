//! Integration tests for runtime allocator-routing behavior.
mod common;

use std::{ptr, sync::Arc};

use common::cudf::{
    assert_score_maps_close_f32, explicit_runtime, pagerank_score_map, python_weighted_reciprocal_expected_pagerank_map,
};
use cudf::testing::prelude::*;
use cudf_sys::test_support;
use cugraph::{
    DeviceArray, GraphBuilder,
    cudf_interop::{
        GraphConstructionPolicy, SeriesGraphOptions, graph_from_borrowed_columns, graph_from_prepared_columns,
    },
    graph::Transposed,
    handle::ResourceHandle,
    types::GraphProperties,
};

fn current_device_resource_raw() -> *mut std::ffi::c_void {
    let mut out = ptr::null_mut();
    let status = unsafe { test_support::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read current device resource");
    out
}

fn reset_exec_observation() {
    unsafe { test_support::cudf_test_exec_reset_observation() };
}

fn last_exec_stream_value() -> usize {
    let mut out = 0usize;
    let status = unsafe { test_support::cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last execution stream");
    out
}

fn last_exec_mr_value() -> usize {
    let mut out = ptr::null_mut();
    let status = unsafe { test_support::cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last execution memory resource");
    out as usize
}

fn python_weighted_options() -> SeriesGraphOptions {
    SeriesGraphOptions::default()
        .properties(GraphProperties::undirected())
        .renumber()
        .construction_policy(GraphConstructionPolicy::PythonCugraph)
}

fn runtime_with_explicit_pool_memory_resource() -> Arc<cudf::Runtime> {
    Arc::new(cudf::Runtime::builder().with_pool_memory_resource(8 * 1024 * 1024, None).build().expect("runtime"))
}

#[test]
fn runtime_bound_borrowed_python_imports_use_runtime_execution_context() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = runtime_with_explicit_pool_memory_resource();
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("runtime-bound handle");
    let handle = handle.enter("runtime_bound_borrowed_python_imports_use_runtime_execution_context").expect("entered");
    let expected = python_weighted_reciprocal_expected_pagerank_map(&handle);

    let src_runtime = explicit_runtime();
    let dst_runtime = explicit_runtime();
    let wgt_runtime = explicit_runtime();
    let src = src_runtime.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = dst_runtime.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = wgt_runtime.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("weight");
    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let wgt_borrowed = wgt.try_borrow_column_view().expect("weight borrowed");
    reset_exec_observation();

    let mut graph = graph_from_borrowed_columns::<Transposed>(
        &handle,
        &src_borrowed,
        &dst_borrowed,
        Some(&wgt_borrowed),
        &python_weighted_options(),
    )
    .expect("graph_from_borrowed_columns");
    let actual = pagerank_score_map(&handle, &mut graph);
    runtime.synchronize().expect("runtime synchronize");

    assert_eq!(
        last_exec_stream_value(),
        runtime.stream().testing_raw_handle_value(),
        "borrowed import path should run cudf work on the runtime stream"
    );
    assert_eq!(
        last_exec_mr_value(),
        runtime.memory_resource().testing_raw_handle_value(),
        "borrowed import path should run cudf work on the runtime memory resource"
    );
    assert_score_maps_close_f32(&actual, &expected, 1.0e-6, "borrowed allocator routing");
}

#[test]
fn runtime_bound_prepared_python_imports_use_runtime_execution_context() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = runtime_with_explicit_pool_memory_resource();
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("runtime-bound handle");
    let handle = handle.enter("runtime_bound_prepared_python_imports_use_runtime_execution_context").expect("entered");
    let expected = python_weighted_reciprocal_expected_pagerank_map(&handle);

    let src_runtime = explicit_runtime();
    let dst_runtime = explicit_runtime();
    let wgt_runtime = explicit_runtime();
    let src = src_runtime.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = dst_runtime.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = wgt_runtime.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("weight");
    let src_export = src.try_prepare_device_column_export().expect("src export");
    let dst_export = dst.try_prepare_device_column_export().expect("dst export");
    let wgt_export = wgt.try_prepare_device_column_export().expect("weight export");
    reset_exec_observation();

    let mut graph = graph_from_prepared_columns::<Transposed>(
        &handle,
        &src_export,
        &dst_export,
        Some(&wgt_export),
        &python_weighted_options(),
    )
    .expect("graph_from_prepared_columns");
    let actual = pagerank_score_map(&handle, &mut graph);
    runtime.synchronize().expect("runtime synchronize");

    assert_eq!(
        last_exec_stream_value(),
        runtime.stream().testing_raw_handle_value(),
        "prepared import path should run cudf work on the runtime stream"
    );
    assert_eq!(
        last_exec_mr_value(),
        runtime.memory_resource().testing_raw_handle_value(),
        "prepared import path should run cudf work on the runtime memory resource"
    );
    assert_score_maps_close_f32(&actual, &expected, 1.0e-6, "prepared allocator routing");
}

#[test]
fn runtime_bound_parked_graph_retains_bytes_in_its_cache_allocation_domain() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let device = cudf::DeviceId::current().expect("current CUDA device");
    let manager = cudf::DeviceMemoryManager::for_device(device).expect("permanent device memory manager");
    let current_before = current_device_resource_raw();
    assert_eq!(
        current_before as usize,
        manager.resource().testing_raw_handle_value(),
        "the production allocator route requires cuDF's permanent manager resource to remain current"
    );
    let cache_domain = cudf::CacheAllocationDomain::new(device, 64 * 1024 * 1024).expect("cache allocation domain");
    let cache_lease = cache_domain.lease_runtime().expect("cache runtime lease");
    let cache_runtime = Arc::new(cache_lease.runtime().clone());
    let cache_handle = ResourceHandle::for_runtime(Arc::clone(&cache_runtime)).expect("cache-bound resource handle");
    let cache_entered = cache_handle
        .enter("runtime_bound_parked_graph_retains_bytes_in_its_cache_allocation_domain")
        .expect("entered cache context");

    let input_runtime = explicit_runtime();
    let input_runtime = Arc::new(input_runtime);
    let input_handle = ResourceHandle::for_runtime(Arc::clone(&input_runtime)).expect("input resource handle");
    let input_entered = input_handle.enter("runtime_bound_parked_graph_input").expect("entered input context");
    let src = DeviceArray::from_slice(&input_entered, &[0i32, 1, 2, 2]).expect("source vertices");
    let dst = DeviceArray::from_slice(&input_entered, &[1i32, 2, 0, 1]).expect("destination vertices");
    let src_view = src.view().expect("source view");
    let dst_view = dst.view().expect("destination view");

    let domain_before = cache_domain.usage().expect("cache domain usage before graph construction");

    let graph = GraphBuilder::new(&src_view, &dst_view)
        .renumber()
        .build::<Transposed>(&cache_entered)
        .expect("cache-bound graph construction");
    let parked = graph.park(&cache_entered).expect("park cache-bound graph");
    cache_runtime.synchronize().expect("cache runtime synchronize");

    let domain_with_parked_graph = cache_domain.usage().expect("cache domain usage with parked graph");
    assert!(
        domain_with_parked_graph.current_bytes > domain_before.current_bytes,
        "a parked graph must retain live bytes in its runtime-bound cache domain: before={domain_before:?}, after={domain_with_parked_graph:?}"
    );
    assert_eq!(
        current_device_resource_raw(),
        current_before,
        "runtime-bound graph construction must not replace cuDF's permanent current RMM resource"
    );

    drop(parked);
    cache_runtime.synchronize().expect("cache runtime synchronize after parked graph drop");
    let domain_after_drop = cache_domain.usage().expect("cache domain usage after parked graph drop");
    assert_eq!(
        domain_after_drop.current_bytes, domain_before.current_bytes,
        "dropping the parked graph must release exactly the cache-domain bytes it retained: before={domain_before:?}, after={domain_after_drop:?}"
    );
    assert_eq!(
        current_device_resource_raw(),
        current_before,
        "cache graph teardown must leave cuDF's permanent current RMM resource unchanged"
    );
}
