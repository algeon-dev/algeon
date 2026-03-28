//! Integration tests for execution-context plumbing.
//!
//! These tests verify that non-default CUDA streams and memory resources are
//! correctly propagated through the FFI shim layer and (eventually) through
//! the cudf-rs safe API.
//!
//! # Test categories
//!
//! 1. **Type properties** — `ExecutionContext` is Copy, default has null handles.
//! 2. **Existing `_ex` FFI** — `cudf_column_copy_ex`, `cudf_column_view_to_host_ex`,
//!    `cudf_column_create_from_device_data_ex` use explicit streams (pass today).
//! 3. **Future `_with_context` high-level API** — `sort_values_with_context`, `merge_with_context`,
//!    `groupby().agg_with_context()` etc. (will fail until EC-SHIM + EC-HIGHLEVEL land).
//! 4. **Default-context fallback** — existing APIs still use the default stream.
//! 5. **Allocator context** — explicit memory resource is passed through.
//! 6. **cudf-rs type contracts** — CudaStream, DeviceMemoryResource are Send+Sync.

use std::{ffi::c_void, ptr};

use cudf_sys::{
    cudf_test_current_device_resource_raw, cudf_test_default_stream_value, cudf_test_exec_last_mr_raw,
    cudf_test_exec_last_stream_value, cudf_test_exec_last_sync_stream_value, cudf_test_exec_reset_observation,
    cudf_test_stream_handle_value,
};

// ---------------------------------------------------------------------------
// Safe wrappers around observation hooks
// ---------------------------------------------------------------------------

/// Reset observation state so the next `_ex` call's context is cleanly visible.
fn obs_reset() {
    unsafe { cudf_test_exec_reset_observation() };
}

/// Last stream value seen by an `_ex` shim call.
fn obs_last_stream() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_stream_value failed");
    out
}

/// Last synchronized stream value (for host-materializing ops).
#[allow(dead_code)]
fn obs_last_sync_stream() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_sync_stream_value failed");
    out
}

/// Last memory-resource raw pointer seen by an `_ex` shim call.
fn obs_last_mr() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_mr_raw failed");
    out
}

/// Process-global default stream value.
fn obs_default_stream() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_default_stream_value(&mut out) };
    assert_eq!(status, 0, "cudf_test_default_stream_value failed");
    out
}

/// Convert a stream handle to its raw uintptr value (for comparison).
fn obs_stream_handle_value(handle: *mut c_void) -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_stream_handle_value(handle, &mut out) };
    assert_eq!(status, 0, "cudf_test_stream_handle_value failed");
    out
}

/// Process-global current device memory resource raw pointer.
fn obs_current_device_resource() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, 0, "cudf_test_current_device_resource_raw failed");
    out
}

// ---------------------------------------------------------------------------
// Helper: create a cudf_sys column from i32 data (via cudf safe API, then
// extract the raw handle for _ex FFI calls).
// ---------------------------------------------------------------------------

/// Create an i32 column and return its raw cudf_column_handle.
///
/// The caller is responsible for destroying the handle.
fn create_i32_column_raw(data: &[i32]) -> cudf_sys::cudf_column_handle {
    let mut handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(), // no validity mask
            0,           // null_count = 0
            &mut handle,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create i32 column");
    assert!(!handle.is_null());
    handle
}

/// Create an i64 column and return its raw cudf_column_handle.
fn create_i64_column_raw(data: &[i64]) -> cudf_sys::cudf_column_handle {
    let mut handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT64,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &mut handle,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create i64 column");
    assert!(!handle.is_null());
    handle
}

/// Get a view handle from a column handle. Caller must destroy the view.
fn column_view_raw(col: cudf_sys::cudf_column_handle) -> cudf_sys::cudf_column_view_handle {
    let mut view: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_view(col, &mut view) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create column view");
    assert!(!view.is_null());
    view
}

/// Read device data pointer from a column view.
fn column_view_device_ptr(view: cudf_sys::cudf_column_view_handle) -> *const c_void {
    let mut ptr: *const c_void = ptr::null();
    let status = unsafe { cudf_sys::cudf_column_view_data_ptr(view, &mut ptr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to get device data ptr");
    ptr
}

// ===========================================================================
// Category 1: ExecutionContext type properties (cudf-sys layer)
// ===========================================================================

/// Default ExecutionContext has null stream and null mr.
#[test]
fn execution_context_default_has_null_handles() {
    let ctx = cudf_sys::ExecutionContext::default();
    let raw = ctx.raw();
    assert!(raw.stream.is_null(), "default stream should be null");
    assert!(raw.mr.is_null(), "default mr should be null");
    assert_eq!(raw.flags, 0, "default flags should be zero");
}

/// ExecutionContext stores a non-null stream handle.
#[test]
fn execution_context_with_stream_stores_handle() {
    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&stream);
    let raw = ctx.raw();
    assert!(!raw.stream.is_null(), "stream handle should be non-null");
    assert_eq!(raw.stream, stream.as_raw(), "stream handle mismatch");
}

/// ExecutionContext stores a non-null memory-resource handle.
#[test]
fn execution_context_with_mr_stores_handle() {
    let mr = cudf_sys::MemoryResource::new_cuda().expect("failed to create mr");
    let ctx = cudf_sys::ExecutionContext::new().with_memory_resource(&mr);
    let raw = ctx.raw();
    assert!(!raw.mr.is_null(), "mr handle should be non-null");
    assert_eq!(raw.mr, mr.as_raw(), "mr handle mismatch");
}

/// ExecutionContext is Copy — copying produces an identical struct.
#[test]
fn execution_context_is_copy() {
    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&stream);
    let copy = ctx; // Copy
    assert_eq!(ctx.raw().stream, copy.raw().stream);
    assert_eq!(ctx.raw().mr, copy.raw().mr);
    assert_eq!(ctx.raw().flags, copy.raw().flags);
}

/// ExecutionContext should eventually be Send + Sync (required for
/// spawn_blocking). Currently it is NOT because cudf_exec_options_t
/// contains raw pointers. This test documents the contract that
/// The EC-HIGHLEVEL milestone must satisfy this by either adding unsafe Send+Sync impls
/// or by wrapping the pointers.
///
/// Once EC-HIGHLEVEL adds Send+Sync, replace this test body with:
///   fn assert_send_sync<T: Send + Sync>() {}
///   assert_send_sync::<cudf_sys::ExecutionContext<'_>>();
#[test]
fn execution_context_send_sync_status_documented() {
    // cudf_exec_options_t contains *mut c_void, so ExecutionContext is
    // currently !Send + !Sync. This test documents the status quo.
    // When EC-HIGHLEVEL adds `unsafe impl Send/Sync`, update this test
    // to assert the positive trait bounds.
    //
    // For now, we just verify the type is Copy (which it is).
    let ctx = cudf_sys::ExecutionContext::default();
    let _copy = ctx;
    let _also_ctx = ctx; // proves Copy
}

/// struct_size is correctly initialized.
#[test]
fn execution_context_struct_size_is_correct() {
    let ctx = cudf_sys::ExecutionContext::default();
    assert_eq!(
        ctx.raw().struct_size,
        std::mem::size_of::<cudf_sys::cudf_exec_options_t>(),
        "struct_size must equal size_of::<cudf_exec_options_t>()"
    );
}

// ===========================================================================
// Category 2: Existing _ex operations (pass today)
//
// These tests use the 3 existing _ex FFI variants to verify that explicit
// stream handles reach the C++ shim's observation hooks.
//
// All column creation and manipulation uses cudf_sys FFI directly (not the
// cudf safe API) because Column::raw_handle() is pub(crate).
// ===========================================================================

/// `cudf_column_copy_ex` with an explicit stream records that stream in the
/// observation hook, not the process-global default.
#[test]
fn column_copy_ex_uses_explicit_stream() {
    let col = create_i32_column_raw(&[10, 20, 30]);

    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&stream);

    let default_stream = obs_default_stream();
    let explicit_stream = obs_stream_handle_value(stream.as_raw());
    assert_ne!(
        explicit_stream, default_stream,
        "explicit stream must differ from default for this test to be meaningful"
    );

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_copy_ex(col, ctx.as_raw(), &mut out_handle) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "cudf_column_copy_ex failed");
    assert!(!out_handle.is_null());

    let observed = obs_last_stream();
    assert_eq!(observed, explicit_stream, "observation hook should record the explicit stream, not the default");

    stream.synchronize().expect("stream sync failed");
    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_destroy(col);
    };
}

/// `cudf_column_view_to_host_ex` with an explicit stream records that stream
/// in the observation hook.
#[test]
fn column_to_host_ex_uses_explicit_stream() {
    let data: [i32; 3] = [100, 200, 300];
    let col = create_i32_column_raw(&data);
    let view = column_view_raw(col);

    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&stream);

    let explicit_stream = obs_stream_handle_value(stream.as_raw());

    obs_reset();
    let mut host_buf = [0i32; 3];
    let buf_size = std::mem::size_of_val(&host_buf);
    let status = unsafe {
        cudf_sys::cudf_column_view_to_host_ex(view, host_buf.as_mut_ptr() as *mut c_void, buf_size, ctx.as_raw())
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "cudf_column_view_to_host_ex failed");

    let observed = obs_last_stream();
    assert_eq!(observed, explicit_stream, "observation hook should record the explicit stream for to_host_ex");

    assert_eq!(host_buf, data, "host data should match original");

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    };
}

/// `cudf_column_create_from_device_data_ex` with an explicit stream records
/// that stream in the observation hook.
#[test]
fn column_from_device_data_ex_uses_explicit_stream() {
    let data: [i64; 4] = [1, 2, 3, 4];
    let src_col = create_i64_column_raw(&data);
    let src_view = column_view_raw(src_col);
    let device_ptr = column_view_device_ptr(src_view);

    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&stream);

    let explicit_stream = obs_stream_handle_value(stream.as_raw());

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_from_device_data_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT64,
            device_ptr,
            4,
            ctx.as_raw(),
            &mut out_handle,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "cudf_column_create_from_device_data_ex failed");
    assert!(!out_handle.is_null());

    let observed = obs_last_stream();
    assert_eq!(observed, explicit_stream, "observation hook should record the explicit stream for from_device_data_ex");

    // The _ex variant does NOT synchronize — the caller must keep src alive
    // until the stream completes.
    stream.synchronize().expect("stream sync failed");

    // Verify round-trip: read back the copied column.
    let out_view = column_view_raw(out_handle);
    let mut host_buf = [0i64; 4];
    let s3 = unsafe {
        cudf_sys::cudf_column_view_to_host(
            out_view,
            host_buf.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(&host_buf),
        )
    };
    assert_eq!(s3, cudf_sys::CUDF_SUCCESS);
    assert_eq!(host_buf, data, "round-trip data should match");

    unsafe {
        cudf_sys::cudf_column_view_destroy(out_view);
        cudf_sys::cudf_column_view_destroy(src_view);
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_destroy(src_col);
    };
}

/// When `_ex` ops are called with null options (default context), the
/// observation hook records the process-global default stream.
#[test]
fn column_copy_ex_null_options_uses_default_stream() {
    let col = create_i32_column_raw(&[10, 20, 30]);

    let default_stream = obs_default_stream();

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_copy_ex(col, ptr::null(), &mut out_handle) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out_handle.is_null());

    let observed = obs_last_stream();
    assert_eq!(observed, default_stream, "null options should resolve to the default stream");

    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_destroy(col);
    };
}

/// When `_ex` ops are called with default ExecutionContext (null handles),
/// the observation hook records the default stream and default mr.
#[test]
fn column_copy_ex_default_context_uses_defaults() {
    let col = create_i32_column_raw(&[1, 2, 3]);

    let default_stream = obs_default_stream();
    let default_mr = obs_current_device_resource();

    let ctx = cudf_sys::ExecutionContext::default();

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_copy_ex(col, ctx.as_raw(), &mut out_handle) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    assert_eq!(obs_last_stream(), default_stream);
    assert_eq!(obs_last_mr(), default_mr);

    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_destroy(col);
    };
}

/// Explicit memory resource is recorded by the observation hook.
#[test]
fn column_copy_ex_explicit_mr_is_observed() {
    let col = create_i32_column_raw(&[5, 10, 15]);

    let mr = cudf_sys::MemoryResource::new_cuda().expect("failed to create mr");
    let ctx = cudf_sys::ExecutionContext::new().with_memory_resource(&mr);

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_copy_ex(col, ctx.as_raw(), &mut out_handle) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let observed_mr = obs_last_mr();
    assert!(!observed_mr.is_null(), "observed mr should be non-null");

    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_destroy(col);
    };
}

// ===========================================================================
// Category 3: Future _with_context high-level API operations
//
// These tests document the EXPECTED behavior for cudf-rs safe API methods
// that will accept ExecutionContext once EC-SHIM and EC-HIGHLEVEL are complete.
//
// They are #[ignore] because the _with_context methods do not exist yet.
// Each test describes the contract that the implementation must satisfy.
// ===========================================================================

/// Contract: `DataFrame::sort_values_with_context` routes work through an
/// explicit non-default stream. The observation hook must record the
/// explicit stream, not the default.
///
/// Will fail until EC-SHIM adds `cudf_sort_ex` and EC-HIGHLEVEL adds
/// `DataFrame::sort_values_with_context`.
#[test]
#[ignore = "requires sort_values_with_context (EC-HIGHLEVEL)"]
fn sort_with_context_uses_explicit_stream() {
    // Setup
    let ids = cudf::Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let df = cudf::DataFrame::from_series([("id", ids)]).unwrap();

    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let explicit_stream = obs_stream_handle_value(stream.as_raw());

    obs_reset();

    // TODO: uncomment when sort_values_with_context exists
    // let ctx = cudf::ExecutionContext::with_stream(&stream_wrapper);
    // let sorted = df.sort_values_with_context(["id"], cudf::SortOptions::default(), &ctx).unwrap();
    // stream.synchronize().unwrap();

    // let observed = obs_last_stream();
    // assert_eq!(observed, explicit_stream,
    //     "sort_values_with_context should route through explicit stream");

    // Verify correctness
    // let col = sorted.series("id").unwrap();
    // let vals: Vec<i32> = col.column_view().unwrap().to_vec().unwrap();
    // assert_eq!(vals, vec![1, 2, 3]);

    let _ = (df, explicit_stream);
    panic!("sort_values_with_context not yet implemented");
}

/// Contract: `DataFrame::merge_with_context` routes join work through
/// an explicit non-default stream.
///
/// Will fail until EC-SHIM adds `cudf_inner_join_ex` etc. and EC-HIGHLEVEL
/// adds `DataFrame::merge_with_context`.
#[test]
#[ignore = "requires merge_with_context (EC-HIGHLEVEL)"]
fn join_with_context_uses_explicit_stream() {
    let left_ids = cudf::Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let left_vals = cudf::Series::new::<f64>("val", &[10.0, 20.0, 30.0]).unwrap();
    let left = cudf::DataFrame::from_series([("id", left_ids), ("val", left_vals)]).unwrap();

    let right_ids = cudf::Series::new::<i32>("id", &[2, 3, 4]).unwrap();
    let right_scores = cudf::Series::new::<f64>("score", &[200.0, 300.0, 400.0]).unwrap();
    let right = cudf::DataFrame::from_series([("id", right_ids), ("score", right_scores)]).unwrap();

    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let explicit_stream = obs_stream_handle_value(stream.as_raw());

    obs_reset();

    // TODO: uncomment when merge_with_context exists
    // let ctx = cudf::ExecutionContext::with_stream(&stream_wrapper);
    // let joined = left.merge_with_context(
    //     &right,
    //     cudf::MergeOptions::new().how_inner().on(["id"]),
    //     &ctx,
    // ).unwrap();
    // stream.synchronize().unwrap();

    // let observed = obs_last_stream();
    // assert_eq!(observed, explicit_stream,
    //     "merge_with_context should route through explicit stream");

    // assert_eq!(joined.num_rows(), 2); // ids 2, 3

    let _ = (left, right, explicit_stream);
    panic!("merge_with_context not yet implemented");
}

/// Contract: `GroupBy::agg_with_context` routes groupby aggregation through
/// an explicit non-default stream.
///
/// Will fail until EC-SHIM adds `cudf_groupby_aggregate_ex` and EC-HIGHLEVEL
/// adds `GroupBy::agg_with_context`.
#[test]
#[ignore = "requires agg_with_context (EC-HIGHLEVEL)"]
fn groupby_with_context_uses_explicit_stream() {
    let keys = cudf::Series::new::<i32>("key", &[1, 1, 2, 2, 3]).unwrap();
    let vals = cudf::Series::new::<f64>("val", &[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let df = cudf::DataFrame::from_series([("key", keys), ("val", vals)]).unwrap();

    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let explicit_stream = obs_stream_handle_value(stream.as_raw());

    obs_reset();

    // TODO: uncomment when agg_with_context exists
    // let ctx = cudf::ExecutionContext::with_stream(&stream_wrapper);
    // let result = df.groupby(["key"])
    //     .agg_with_context(
    //         &[cudf::AggSpec::new("val", cudf::Agg::Sum)],
    //         &ctx,
    //     ).unwrap();
    // stream.synchronize().unwrap();

    // let observed = obs_last_stream();
    // assert_eq!(observed, explicit_stream,
    //     "agg_with_context should route through explicit stream");

    // assert_eq!(result.num_rows(), 3); // 3 unique keys

    let _ = (df, explicit_stream);
    panic!("agg_with_context not yet implemented");
}

/// Contract: `DataFrame::read_parquet` with builder-carried context routes
/// I/O through an explicit non-default stream.
///
/// Will fail until EC-SHIM adds `cudf_io_read_parquet_ex` and EC-HIGHLEVEL
/// adds `ParquetReaderOptions::with_context`.
#[test]
#[ignore = "requires ParquetReaderOptions::with_context (EC-HIGHLEVEL)"]
fn read_parquet_with_context_uses_explicit_stream() {
    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let explicit_stream = obs_stream_handle_value(stream.as_raw());

    obs_reset();

    // TODO: uncomment when ParquetReaderOptions::with_context exists
    // let ctx = cudf::ExecutionContext::with_stream(&stream_wrapper);
    // let opts = cudf::io::parquet::ParquetReaderOptions::new("test.parquet")
    //     .with_context(&ctx);
    // let df = cudf::DataFrame::read_parquet_with_options(opts).unwrap();
    // stream.synchronize().unwrap();

    // let observed = obs_last_stream();
    // assert_eq!(observed, explicit_stream,
    //     "read_parquet with context should route through explicit stream");

    let _ = explicit_stream;
    panic!("ParquetReaderOptions::with_context not yet implemented");
}

/// Contract: `DataFrame::write_parquet` with builder-carried context routes
/// I/O through an explicit non-default stream.
#[test]
#[ignore = "requires ParquetWriterOptions::with_context (EC-HIGHLEVEL)"]
fn write_parquet_with_context_uses_explicit_stream() {
    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let explicit_stream = obs_stream_handle_value(stream.as_raw());

    obs_reset();

    // TODO: uncomment when ParquetWriterOptions::with_context exists
    // let ids = cudf::Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    // let df = cudf::DataFrame::from_series([("id", ids)]).unwrap();
    // let ctx = cudf::ExecutionContext::with_stream(&stream_wrapper);
    // let opts = cudf::io::parquet::ParquetWriterOptions::new("/tmp/test_ec.parquet")
    //     .with_context(&ctx);
    // df.write_parquet_with_options(opts).unwrap();
    // stream.synchronize().unwrap();

    // let observed = obs_last_stream();
    // assert_eq!(observed, explicit_stream);

    let _ = explicit_stream;
    panic!("ParquetWriterOptions::with_context not yet implemented");
}

// ===========================================================================
// Category 4: Default-context fallback
//
// Verify that existing (non-_with_context) APIs still use the default stream
// by checking the observation hook after a regular operation.
// ===========================================================================

/// Calling `Column::copy()` (the safe API, which uses `cudf_column_copy`)
/// results in the default stream being observed.
#[test]
fn column_copy_default_uses_default_stream() {
    let col = cudf::Column::from_slice(&[1i32, 2, 3]).unwrap();

    let default_stream = obs_default_stream();

    obs_reset();
    let _copy = col.copy().unwrap();

    // Column::copy() uses cudf_column_copy which internally calls
    // cudf_column_copy_ex with null options, resolving to default stream.
    let observed = obs_last_stream();
    assert_eq!(observed, default_stream, "Column::copy() should use the default stream");
}

/// Existing `DataFrame::sort_values` (no context) should produce correct
/// results. This is the backward-compatibility baseline for
/// sort_values_with_context.
#[test]
fn sort_values_default_produces_correct_results() {
    let ids = cudf::Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let df = cudf::DataFrame::from_series([("id", ids)]).unwrap();

    let sorted = df.sort_values(["id"], cudf::SortOptions::default()).unwrap();

    let col = sorted.series("id").unwrap();
    let host: Vec<i32> = col.column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![1, 2, 3], "sort_values result should be correct");
}

/// Existing `DataFrame::merge` (no context) should produce correct results.
/// This is the backward-compatibility baseline for merge_with_context.
#[test]
fn merge_default_produces_correct_results() {
    let left_ids = cudf::Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let left_vals = cudf::Series::new::<f64>("val", &[10.0, 20.0, 30.0]).unwrap();
    let left = cudf::DataFrame::from_series([("id", left_ids), ("val", left_vals)]).unwrap();

    let right_ids = cudf::Series::new::<i32>("id", &[2, 3, 4]).unwrap();
    let right_scores = cudf::Series::new::<f64>("score", &[200.0, 300.0, 400.0]).unwrap();
    let right = cudf::DataFrame::from_series([("id", right_ids), ("score", right_scores)]).unwrap();

    let joined = left.merge(&right, cudf::MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(joined.num_rows(), 2, "inner join on [2,3] should yield 2 rows");
}

/// Existing `GroupBy::agg` (no context) should produce correct results.
/// This is the backward-compatibility baseline for agg_with_context.
#[test]
fn groupby_agg_default_produces_correct_results() {
    let keys = cudf::Series::new::<i32>("key", &[1, 1, 2, 2]).unwrap();
    let vals = cudf::Series::new::<i64>("val", &[10, 20, 30, 40]).unwrap();
    let df = cudf::DataFrame::from_series([("key", keys), ("val", vals)]).unwrap();

    let result = df.groupby(["key"]).agg(&[cudf::AggSpec::new("val", cudf::Agg::Sum)]).unwrap();

    assert_eq!(result.num_rows(), 2, "groupby on 2 unique keys should yield 2 rows");
}

// ===========================================================================
// Category 5: Allocator context
//
// Verify that explicit memory resources are recorded by observation hooks
// when passed through _ex operations.
// ===========================================================================

/// Column copy with an explicit pool memory resource records that resource.
#[test]
fn column_copy_ex_pool_mr_is_observed() {
    let col = create_i32_column_raw(&[42, 43, 44]);

    let pool_mr = cudf_sys::MemoryResource::new_pool(
        16 * 1024 * 1024, // 16 MB initial
        0,                // unbounded max
    )
    .expect("failed to create pool MR");

    let ctx = cudf_sys::ExecutionContext::new().with_memory_resource(&pool_mr);

    let default_mr = obs_current_device_resource();

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_copy_ex(col, ctx.as_raw(), &mut out_handle) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let observed_mr = obs_last_mr();
    assert!(!observed_mr.is_null(), "observed mr should be non-null after _ex call");

    // The pool MR's raw pointer should differ from the process-global default.
    assert_ne!(observed_mr, default_mr, "pool MR should differ from the process-global default MR");

    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_destroy(col);
    };
}

/// Column copy with both explicit stream AND memory resource.
#[test]
fn column_copy_ex_stream_and_mr_combined() {
    let col = create_i32_column_raw(&[7, 8, 9]);

    let stream = cudf_sys::Stream::new().expect("failed to create stream");
    let pool_mr = cudf_sys::MemoryResource::new_pool(16 * 1024 * 1024, 0).expect("failed to create pool MR");

    let ctx = cudf_sys::ExecutionContext::new().with_stream(&stream).with_memory_resource(&pool_mr);

    let explicit_stream = obs_stream_handle_value(stream.as_raw());
    let default_mr = obs_current_device_resource();

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_copy_ex(col, ctx.as_raw(), &mut out_handle) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    assert_eq!(obs_last_stream(), explicit_stream, "stream mismatch");

    let observed_mr = obs_last_mr();
    assert_ne!(observed_mr, default_mr, "pool MR should differ from default");

    stream.synchronize().expect("stream sync failed");
    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_destroy(col);
    };
}

/// Future contract: `sort_values_with_context` with explicit memory resource
/// should produce output allocated from that resource.
#[test]
#[ignore = "requires sort_values_with_context + allocator propagation (EC-HIGHLEVEL)"]
fn sort_with_context_uses_explicit_mr() {
    // TODO: uncomment when ExecutionContext with_memory_resource is available
    // at the cudf-rs layer.
    //
    // let mr = Arc::new(cudf::memory::DeviceMemoryResource::pool(256 * 1024 * 1024, 0).unwrap());
    // let ctx = cudf::ExecutionContext::with_memory_resource(&mr);
    //
    // let ids = cudf::Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    // let df = cudf::DataFrame::from_series([("id", ids)]).unwrap();
    //
    // obs_reset();
    // let sorted = df.sort_values_with_context(["id"], SortOptions::default(), &ctx).unwrap();
    //
    // let observed_mr = obs_last_mr();
    // let default_mr = obs_current_device_resource();
    // assert_ne!(observed_mr, default_mr, "should use explicit MR, not default");
    //
    // let col = sorted.series("id").unwrap();
    // let vals: Vec<i32> = col.column_view().unwrap().to_vec().unwrap();
    // assert_eq!(vals, vec![1, 2, 3]);
    panic!("sort_values_with_context with MR not yet implemented");
}

// ===========================================================================
// Category 6: cudf-rs layer CudaStream / DeviceMemoryResource type contracts
//
// These verify that the cudf-rs wrappers (CudaStream, DeviceMemoryResource)
// have the properties needed for ExecutionContext construction.
// ===========================================================================

/// CudaStream is Send + Sync (required to move into spawn_blocking closures).
#[test]
fn cuda_stream_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<cudf::stream::CudaStream>();
}

/// DeviceMemoryResource is Send + Sync (required for Arc sharing).
#[test]
fn device_memory_resource_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<cudf::memory::DeviceMemoryResource>();
}

/// CudaStream can be created and synchronized.
#[test]
fn cuda_stream_create_and_sync() {
    let stream = cudf::stream::CudaStream::new().unwrap();
    stream.synchronize().unwrap();
    assert!(stream.is_ready().unwrap());
}

/// CudaStream is accessible via the `cudf::stream` module path (not re-exported at crate root).
#[test]
fn cuda_stream_accessible_via_module_path() {
    let stream = cudf::stream::CudaStream::new().unwrap();
    stream.synchronize().unwrap();
}

/// DeviceMemoryResource can be created (cuda variant).
#[test]
fn device_memory_resource_create_cuda() {
    let _mr = cudf::memory::DeviceMemoryResource::cuda().unwrap();
}

/// DeviceMemoryResource can be created (pool variant).
#[test]
fn device_memory_resource_create_pool() {
    let _mr = cudf::memory::DeviceMemoryResource::pool(16 * 1024 * 1024, 0).unwrap();
}
