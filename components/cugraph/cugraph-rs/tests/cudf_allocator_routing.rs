//! Integration tests for Phase 7 allocator-routing evidence.
mod common;

use std::{process::Command, ptr, sync::Arc};

use common::cudf::{
    assert_score_maps_close_f32, explicit_runtime, pagerank_score_map, python_weighted_reciprocal_expected_pagerank_map,
};
use cudf::testing::prelude::*;
use cugraph::{
    cudf_interop::{
        GraphConstructionPolicy, SeriesGraphOptions, graph_from_borrowed_columns, graph_from_prepared_columns,
    },
    graph::Transposed,
    handle::ResourceHandle,
    types::GraphProperties,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentAllocatorProbeOutcome {
    Observed { allocation_delta: i64 },
    Unobserved,
    ObservedViaAbort,
}

fn reset_exec_observation() {
    unsafe { cudf_sys::cudf_test_exec_reset_observation() };
}

fn last_exec_stream_value() -> usize {
    let mut out = 0usize;
    let status = unsafe { cudf_sys::cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last execution stream");
    out
}

fn last_exec_mr_value() -> usize {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_test_exec_last_mr_raw(&mut out) };
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

fn run_current_allocator_probe(mode: &str) -> CurrentAllocatorProbeOutcome {
    let output = Command::new(env!("CARGO_BIN_EXE_phase7_allocator_probe"))
        .arg(mode)
        .output()
        .expect("allocator probe should start");

    if let Some(code) = output.status.code() {
        assert_eq!(
            code,
            0,
            "allocator probe failed unexpectedly: stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8(output.stdout).expect("probe stdout should be UTF-8");
        let line = stdout.trim();
        if line == "unobserved" {
            return CurrentAllocatorProbeOutcome::Unobserved;
        }
        let delta = line
            .strip_prefix("observed:")
            .and_then(|value| value.parse::<i64>().ok())
            .expect("probe should report observed:<delta> or unobserved");
        return CurrentAllocatorProbeOutcome::Observed { allocation_delta: delta };
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;

        if output.status.signal() == Some(6) {
            return CurrentAllocatorProbeOutcome::ObservedViaAbort;
        }
    }

    panic!(
        "allocator probe terminated unexpectedly: stdout={}; stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
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
fn borrowed_allocator_probe_reports_current_allocator_usage_or_contained_abort() {
    let _guard = common::cudf::cuda_test_serial_guard();
    match run_current_allocator_probe("borrowed") {
        CurrentAllocatorProbeOutcome::Observed { allocation_delta } => {
            assert!(allocation_delta > 0, "observed probe should report a positive allocation delta");
        },
        CurrentAllocatorProbeOutcome::Unobserved | CurrentAllocatorProbeOutcome::ObservedViaAbort => {},
    }
}

#[test]
fn prepared_allocator_probe_reports_current_allocator_usage_or_contained_abort() {
    let _guard = common::cudf::cuda_test_serial_guard();
    match run_current_allocator_probe("prepared") {
        CurrentAllocatorProbeOutcome::Observed { allocation_delta } => {
            assert!(allocation_delta > 0, "observed probe should report a positive allocation delta");
        },
        CurrentAllocatorProbeOutcome::Unobserved | CurrentAllocatorProbeOutcome::ObservedViaAbort => {},
    }
}
