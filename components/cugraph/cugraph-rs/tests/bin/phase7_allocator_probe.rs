use std::{ffi::c_void, ptr, sync::Arc};

use cudf::testing::prelude::*;
use cugraph::{
    centrality::PageRankParams,
    cudf_interop::{
        GraphConstructionPolicy, SeriesGraphOptions, graph_from_borrowed_columns, graph_from_prepared_columns,
    },
    graph::Transposed,
    handle::ResourceHandle,
    types::GraphProperties,
};

struct CurrentDeviceResourceRestore {
    previous_raw: *mut c_void,
}

impl Drop for CurrentDeviceResourceRestore {
    fn drop(&mut self) {
        unsafe { cudf_sys::test_set_current_device_resource_raw(self.previous_raw) }
            .expect("failed to restore previous current device resource");
    }
}

fn current_device_resource_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read current device resource");
    out
}

fn install_current_device_resource(mr: &cudf::core::memory::DeviceMemoryResource) -> CurrentDeviceResourceRestore {
    let previous_raw = current_device_resource_raw();
    let borrowed = mr.as_borrowed_sys();
    cudf_sys::test_set_current_device_resource(&borrowed).expect("failed to install current device resource");
    CurrentDeviceResourceRestore { previous_raw }
}

fn explicit_runtime() -> cudf::Runtime {
    cudf::Runtime::builder().build().expect("runtime")
}

fn runtime_with_explicit_pool_memory_resource() -> Arc<cudf::Runtime> {
    Arc::new(cudf::Runtime::builder().with_pool_memory_resource(8 * 1024 * 1024, None).build().expect("runtime"))
}

fn python_weighted_options() -> SeriesGraphOptions {
    SeriesGraphOptions::default()
        .properties(GraphProperties::undirected())
        .renumber()
        .construction_policy(GraphConstructionPolicy::PythonCugraph)
}

fn run_borrowed_probe(handle: &ResourceHandle) {
    let src_runtime = explicit_runtime();
    let dst_runtime = explicit_runtime();
    let wgt_runtime = explicit_runtime();
    let src = src_runtime.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = dst_runtime.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = wgt_runtime.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("weight");
    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let wgt_borrowed = wgt.try_borrow_column_view().expect("weight borrowed");

    let mut graph = graph_from_borrowed_columns::<Transposed>(
        handle,
        &src_borrowed,
        &dst_borrowed,
        Some(&wgt_borrowed),
        &python_weighted_options(),
    )
    .expect("graph_from_borrowed_columns");

    let result = graph.pagerank(handle, &PageRankParams::default()).expect("pagerank");
    let _scores = result.scores().expect("scores").to_vec_f32(handle).expect("scores copy");
}

fn run_prepared_probe(handle: &ResourceHandle) {
    let src_runtime = explicit_runtime();
    let dst_runtime = explicit_runtime();
    let wgt_runtime = explicit_runtime();
    let src = src_runtime.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = dst_runtime.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = wgt_runtime.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("weight");
    let src_export = src.try_prepare_device_column_export().expect("src export");
    let dst_export = dst.try_prepare_device_column_export().expect("dst export");
    let wgt_export = wgt.try_prepare_device_column_export().expect("weight export");

    let mut graph = graph_from_prepared_columns::<Transposed>(
        handle,
        &src_export,
        &dst_export,
        Some(&wgt_export),
        &python_weighted_options(),
    )
    .expect("graph_from_prepared_columns");

    let result = graph.pagerank(handle, &PageRankParams::default()).expect("pagerank");
    let _scores = result.scores().expect("scores").to_vec_f32(handle).expect("scores copy");
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: phase7_allocator_probe <borrowed|prepared>");
        std::process::exit(2);
    });

    let runtime = runtime_with_explicit_pool_memory_resource();
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("runtime-bound handle");
    let current_stats_mr = cudf::testing::core::stats_memory_resource().expect("current stats memory resource");
    assert_ne!(
        current_stats_mr.testing_raw_handle_value(),
        runtime.memory_resource().testing_raw_handle_value(),
        "probe requires a distinct current allocator and runtime allocator"
    );

    let _restore = install_current_device_resource(&current_stats_mr);
    let baseline = current_stats_mr.testing_allocation_count().expect("current allocation count");

    match mode.as_str() {
        "borrowed" => run_borrowed_probe(&handle),
        "prepared" => run_prepared_probe(&handle),
        _ => {
            eprintln!("unknown probe mode: {mode}");
            std::process::exit(2);
        },
    }

    runtime.synchronize().expect("runtime synchronize");
    let delta = current_stats_mr.testing_allocation_count().expect("current allocation count after") - baseline;
    if delta == 0 {
        println!("unobserved");
    } else {
        println!("observed:{delta}");
    }
    use std::io::Write;
    std::io::stdout().flush().expect("flush probe output");
    std::process::exit(0);
}
