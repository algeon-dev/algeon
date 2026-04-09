//! Integration tests for execution-context plumbing.
//!
//! These tests verify that non-default CUDA streams and memory resources are
//! correctly propagated through the FFI shim layer and through the current
//! cudf-rs safe-API slices.
//!
//! # Test categories
//!
//! 1. **Type properties** — `ExecutionContext` is Copy, default has null handles.
//! 2. **Existing `_ex` FFI** — `cudf_column_copy_ex`, `cudf_column_view_to_host_ex`,
//!    `cudf_column_create_from_device_data_ex` use explicit streams (pass today).
//! 3. **High-level execution-context contract** — hidden safe-layer hooks
//!    route explicit-lineage operations through `_ex` entry points and reject
//!    ambient-default promotion from borrowed overrides.
//! 4. **Default-context fallback** — existing APIs still use the default stream.
//! 5. **Allocator context** — explicit memory resource is passed through.
//! 6. **cudf-rs type contracts** — CudaStream, DeviceMemoryResource are Send+Sync.

use std::{ffi::c_void, ptr};

use cudf::{
    DeviceId,
    core::memory::DeviceMemoryResource,
    runtime::{CudaEvent, CudaStream},
};
use cudf_sys::{
    cudf_test_current_device_resource_raw, cudf_test_default_stream_value, cudf_test_exec_last_device,
    cudf_test_exec_last_mr_raw, cudf_test_exec_last_stream_value, cudf_test_exec_last_sync_stream_value,
    cudf_test_exec_reset_observation, cudf_test_stream_handle_value,
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
fn obs_last_sync_stream() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_sync_stream_value failed");
    out
}

/// Last device observed by an `_ex` shim call.
fn obs_last_device() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_exec_last_device(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_device failed");
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

/// Create a bool column and return its raw cudf_column_handle.
fn create_bool_column_raw(data: &[bool]) -> cudf_sys::cudf_column_handle {
    let mut handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_BOOL8,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &mut handle,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create bool column");
    assert!(!handle.is_null());
    handle
}

/// Create a nullable i32 column and return its raw cudf_column_handle.
fn create_i32_nullable_column_raw(data: &[i32], validity_bits: u8) -> cudf_sys::cudf_column_handle {
    let mut handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let mask = [validity_bits];
    let null_count = data.len() as i32 - validity_bits.count_ones() as i32;
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            mask.as_ptr(),
            null_count,
            &mut handle,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create nullable i32 column");
    assert!(!handle.is_null());
    handle
}

/// Create an i32 scalar and return its raw cudf_scalar_handle.
fn create_i32_scalar_raw(value: i32) -> cudf_sys::cudf_scalar_handle {
    let mut handle: cudf_sys::cudf_scalar_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_scalar_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            &value as *const i32 as *const c_void,
            1,
            &mut handle,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create i32 scalar");
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

/// Get a table view handle from a table handle. Caller must destroy the view.
fn table_view_raw(table: cudf_sys::cudf_table_handle) -> cudf_sys::cudf_table_view_handle {
    let mut view: cudf_sys::cudf_table_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view(table, &mut view) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create table view");
    assert!(!view.is_null());
    view
}

/// Get a column view handle from a table view by index. Caller must destroy the view.
fn table_view_column_raw(table: cudf_sys::cudf_table_view_handle, index: i32) -> cudf_sys::cudf_column_view_handle {
    let mut view: cudf_sys::cudf_column_view_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_column(table, index, &mut view) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create table column view");
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

fn read_i32_column_raw(col: cudf_sys::cudf_column_handle, len: usize) -> Vec<i32> {
    let view = column_view_raw(col);
    let mut host = vec![0i32; len];
    let status = unsafe {
        cudf_sys::cudf_column_view_to_host(
            view,
            host.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(host.as_slice()),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read i32 column to host");
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    host
}

fn read_first_i32_table_column_raw(table: cudf_sys::cudf_table_handle, len: usize) -> Vec<i32> {
    let table_view = table_view_raw(table);
    let column_view = table_view_column_raw(table_view, 0);
    let mut host = vec![0i32; len];
    let status = unsafe {
        cudf_sys::cudf_column_view_to_host(
            column_view,
            host.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(host.as_slice()),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read first table column to host");
    unsafe {
        cudf_sys::cudf_column_view_destroy(column_view);
        cudf_sys::cudf_table_view_destroy(table_view);
    }
    host
}

fn create_single_column_table_raw(data: &[i32]) -> cudf_sys::cudf_table_handle {
    let col = create_i32_column_raw(data);
    let mut columns = [col];
    let mut table: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), columns.len() as i32, &mut table) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to create table");
    assert!(!table.is_null());
    table
}

fn explicit_i32_series(name: &str, data: &[i32], identity: u64) -> cudf::Series {
    cudf::Series::from_column(
        cudf::core::Column::from_slice(data).unwrap().__debug_rebind_execution_domain_explicit(identity).unwrap(),
    )
    .unwrap()
    .rename(name)
}

fn explicit_i64_series(name: &str, data: &[i64], identity: u64) -> cudf::Series {
    cudf::Series::from_column(
        cudf::core::Column::from_slice(data).unwrap().__debug_rebind_execution_domain_explicit(identity).unwrap(),
    )
    .unwrap()
    .rename(name)
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
    let stream = CudaStream::new().expect("failed to create stream");
    let borrowed = stream.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed);
    let raw = ctx.raw();
    assert!(!raw.stream.is_null(), "stream handle should be non-null");
    assert_eq!(raw.stream, borrowed.as_raw(), "stream handle mismatch");
}

/// ExecutionContext stores a non-null memory-resource handle.
#[test]
fn execution_context_with_mr_stores_handle() {
    let mr = DeviceMemoryResource::cuda().expect("failed to create mr");
    let borrowed = mr.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_memory_resource(&borrowed);
    let raw = ctx.raw();
    assert!(!raw.mr.is_null(), "mr handle should be non-null");
    assert_eq!(raw.mr, borrowed.as_raw(), "mr handle mismatch");
}

/// ExecutionContext is Copy — copying produces an identical struct.
#[test]
fn execution_context_is_copy() {
    let stream = CudaStream::new().expect("failed to create stream");
    let borrowed = stream.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed);
    let copy = ctx; // Copy
    assert_eq!(ctx.raw().stream, copy.raw().stream);
    assert_eq!(ctx.raw().mr, copy.raw().mr);
    assert_eq!(ctx.raw().flags, copy.raw().flags);
}

/// The low-level `cudf_sys::ExecutionContext` remains a transient FFI builder,
/// not the owned safe-layer override type.
///
/// It is therefore still documented here as `!Send + !Sync` because
/// `cudf_exec_options_t` contains raw pointers. The public safe-layer
/// `cudf::ExecutionContext` has its own `Send + Sync` contract coverage in the
/// `runtime_v1_public_api_tests` suite.
#[test]
fn execution_context_send_sync_status_documented() {
    // cudf_exec_options_t contains *mut c_void, so the low-level FFI builder
    // remains !Send + !Sync. This test documents that status quo.
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

    let stream = CudaStream::new().expect("failed to create stream");
    let borrowed = stream.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed);

    let default_stream = obs_default_stream();
    let explicit_stream = stream.__debug_raw_handle_value();
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

    let stream = CudaStream::new().expect("failed to create stream");
    let borrowed = stream.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed);

    let explicit_stream = stream.__debug_raw_handle_value();

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

    let stream = CudaStream::new().expect("failed to create stream");
    let borrowed = stream.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed);

    let explicit_stream = stream.__debug_raw_handle_value();

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

#[test]
fn replace_nulls_scalar_ex_uses_explicit_stream() {
    let input = create_i32_nullable_column_raw(&[1, 0, 3], 0b0000_0101);
    let input_view = column_view_raw(input);
    let replacement = create_i32_scalar_raw(99);

    let stream = CudaStream::new().expect("failed to create stream");
    let borrowed = stream.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed);
    let explicit_stream = stream.__debug_raw_handle_value();

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_replace_nulls_scalar_ex(input_view, replacement, &mut out_handle, ctx.as_raw()) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "cudf_replace_nulls_scalar_ex failed");
    assert!(!out_handle.is_null());

    assert_eq!(obs_last_stream(), explicit_stream, "observation hook should record the explicit stream");

    stream.synchronize().expect("stream sync failed");
    assert_eq!(read_i32_column_raw(out_handle, 3), vec![1, 99, 3]);

    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_scalar_destroy(replacement);
        cudf_sys::cudf_column_view_destroy(input_view);
        cudf_sys::cudf_column_destroy(input);
    };
}

#[test]
fn replace_nulls_column_ex_uses_explicit_stream() {
    let input = create_i32_nullable_column_raw(&[1, 0, 3], 0b0000_0101);
    let input_view = column_view_raw(input);
    let replacement = create_i32_column_raw(&[10, 20, 30]);
    let replacement_view = column_view_raw(replacement);

    let stream = CudaStream::new().expect("failed to create stream");
    let borrowed = stream.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed);
    let explicit_stream = stream.__debug_raw_handle_value();

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_replace_nulls_column_ex(input_view, replacement_view, &mut out_handle, ctx.as_raw()) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "cudf_replace_nulls_column_ex failed");
    assert!(!out_handle.is_null());

    assert_eq!(obs_last_stream(), explicit_stream, "observation hook should record the explicit stream");

    stream.synchronize().expect("stream sync failed");
    assert_eq!(read_i32_column_raw(out_handle, 3), vec![1, 20, 3]);

    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_view_destroy(replacement_view);
        cudf_sys::cudf_column_destroy(replacement);
        cudf_sys::cudf_column_view_destroy(input_view);
        cudf_sys::cudf_column_destroy(input);
    };
}

#[test]
fn gather_ex_uses_explicit_stream_and_mr() {
    let table = create_single_column_table_raw(&[10, 20, 30]);
    let table_view = table_view_raw(table);
    let gather_map = create_i32_column_raw(&[2, 0]);
    let gather_map_view = column_view_raw(gather_map);

    let stream = CudaStream::new().expect("failed to create stream");
    let mr = DeviceMemoryResource::pool(16 * 1024 * 1024, 0).expect("failed to create pool MR");
    let borrowed_stream = stream.as_borrowed_sys();
    let borrowed_mr = mr.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);
    let explicit_stream = stream.__debug_raw_handle_value();
    let default_mr = obs_current_device_resource();

    obs_reset();
    let mut out_table: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_gather_ex(
            table_view,
            gather_map_view,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            &mut out_table,
            ctx.as_raw(),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "cudf_gather_ex failed");
    assert!(!out_table.is_null());

    assert_eq!(obs_last_stream(), explicit_stream, "observation hook should record the explicit stream");
    assert_ne!(obs_last_mr(), default_mr, "observation hook should record the explicit memory resource");

    stream.synchronize().expect("stream sync failed");
    assert_eq!(read_first_i32_table_column_raw(out_table, 2), vec![30, 10]);

    unsafe {
        cudf_sys::cudf_table_destroy(out_table);
        cudf_sys::cudf_column_view_destroy(gather_map_view);
        cudf_sys::cudf_column_destroy(gather_map);
        cudf_sys::cudf_table_view_destroy(table_view);
        cudf_sys::cudf_table_destroy(table);
    };
}

#[test]
fn apply_boolean_mask_ex_uses_explicit_stream_and_mr() {
    let table = create_single_column_table_raw(&[10, 20, 30]);
    let table_view = table_view_raw(table);
    let mask = create_bool_column_raw(&[true, false, true]);
    let mask_view = column_view_raw(mask);

    let stream = CudaStream::new().expect("failed to create stream");
    let mr = DeviceMemoryResource::pool(16 * 1024 * 1024, 0).expect("failed to create pool MR");
    let borrowed_stream = stream.as_borrowed_sys();
    let borrowed_mr = mr.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);
    let explicit_stream = stream.__debug_raw_handle_value();
    let default_mr = obs_current_device_resource();

    obs_reset();
    let mut out_table: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_apply_boolean_mask_ex(table_view, mask_view, &mut out_table, ctx.as_raw()) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "cudf_apply_boolean_mask_ex failed");
    assert!(!out_table.is_null());

    assert_eq!(obs_last_stream(), explicit_stream, "observation hook should record the explicit stream");
    assert_ne!(obs_last_mr(), default_mr, "observation hook should record the explicit memory resource");

    stream.synchronize().expect("stream sync failed");
    assert_eq!(read_first_i32_table_column_raw(out_table, 2), vec![10, 30]);

    unsafe {
        cudf_sys::cudf_table_destroy(out_table);
        cudf_sys::cudf_column_view_destroy(mask_view);
        cudf_sys::cudf_column_destroy(mask);
        cudf_sys::cudf_table_view_destroy(table_view);
        cudf_sys::cudf_table_destroy(table);
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

    let mr = DeviceMemoryResource::cuda().expect("failed to create mr");
    let borrowed = mr.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_memory_resource(&borrowed);

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

#[test]
fn column_copy_from_view_ex_uses_explicit_stream_and_mr() {
    let col = create_i32_column_raw(&[5, 10, 15]);
    let view = column_view_raw(col);
    let stream = CudaStream::new().expect("failed to create stream");
    let mr = DeviceMemoryResource::cuda().expect("failed to create mr");
    let default_mr = obs_current_device_resource();
    let explicit_stream = stream.__debug_raw_handle_value();
    let borrowed_stream = stream.as_borrowed_sys();
    let borrowed_mr = mr.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);

    obs_reset();
    let mut out_handle: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_copy_from_view_ex(view, ctx.as_raw(), &mut out_handle) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out_handle.is_null());

    assert_eq!(obs_last_stream(), explicit_stream);
    assert_ne!(obs_last_mr(), std::ptr::null_mut());
    assert_ne!(obs_last_mr(), default_mr);

    unsafe {
        cudf_sys::cudf_column_destroy(out_handle);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    };
}

#[test]
fn table_copy_ex_uses_explicit_stream_and_mr() {
    let table = create_single_column_table_raw(&[5, 10, 15]);
    let stream = CudaStream::new().expect("failed to create stream");
    let mr = DeviceMemoryResource::cuda().expect("failed to create mr");
    let default_mr = obs_current_device_resource();
    let explicit_stream = stream.__debug_raw_handle_value();
    let borrowed_stream = stream.as_borrowed_sys();
    let borrowed_mr = mr.as_borrowed_sys();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);

    obs_reset();
    let mut out_handle: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_copy_ex(table, ctx.as_raw(), &mut out_handle) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out_handle.is_null());

    assert_eq!(obs_last_stream(), explicit_stream);
    assert_ne!(obs_last_mr(), std::ptr::null_mut());
    assert_ne!(obs_last_mr(), default_mr);

    unsafe {
        cudf_sys::cudf_table_destroy(out_handle);
        cudf_sys::cudf_table_destroy(table);
    };
}

// ===========================================================================
// Category 3: High-level execution-context contract
//
// These tests exercise hidden safe-layer hooks that validate execution-context
// routing below the current public `Runtime v1` surface.
// ===========================================================================

#[test]
fn sort_with_execution_context_on_explicit_domain_uses_explicit_stream_and_retains_override() {
    let df = cudf::DataFrame::from_series([explicit_i32_series("id", &[3, 1, 2], 501)]).unwrap();

    let stream = CudaStream::new().unwrap();
    let explicit_stream = stream.__debug_raw_handle_value();
    let ctx = cudf::__DebugExecutionContext::new().with_stream(&stream);

    obs_reset();
    let sorted = df.__debug_sort_values_in_context(["id"], cudf::SortOptions::default(), &ctx).unwrap();

    assert_eq!(obs_last_stream(), explicit_stream);
    assert_eq!(sorted.__debug_execution_domain_id(), 501);
    assert_eq!(sorted.__debug_execution_domain_stream_value(), explicit_stream);

    obs_reset();
    let vals: Vec<i32> = sorted.series("id").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
    assert_eq!(obs_last_sync_stream(), explicit_stream);
}

#[test]
fn sort_with_execution_context_rejects_ambient_default_promotion() {
    let ids = cudf::Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let df = cudf::DataFrame::from_series([ids]).unwrap();

    let stream = CudaStream::new().unwrap();
    let ctx = cudf::__DebugExecutionContext::new().with_stream(&stream);

    let err = df.__debug_sort_values_in_context(["id"], cudf::SortOptions::default(), &ctx).unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "execution_context::resolve_output_domain");
    assert!(
        err.context().iter().any(|(key, value)| *key == "operation_domain" && value.starts_with("ambient_default"))
    );
    assert!(err.context().iter().any(|(key, value)| *key == "borrowed_override_stream" && value == "true"));
}

#[test]
fn sort_with_execution_context_on_explicit_domain_uses_explicit_memory_resource() {
    let df = cudf::DataFrame::from_series([explicit_i32_series("id", &[3, 1, 2], 502)]).unwrap();

    let mr = DeviceMemoryResource::pool(16 * 1024 * 1024, 0).unwrap();
    let explicit_mr = mr.__debug_raw_handle_value();
    let ctx = cudf::__DebugExecutionContext::new().with_memory_resource(&mr);
    let default_mr = obs_current_device_resource() as usize;

    obs_reset();
    let sorted = df.__debug_sort_values_in_context(["id"], cudf::SortOptions::default(), &ctx).unwrap();

    assert_ne!(obs_last_mr() as usize, default_mr);
    assert_eq!(sorted.__debug_execution_domain_id(), 502);
    assert_eq!(sorted.__debug_execution_domain_mr_value(), explicit_mr);
}

#[test]
fn merge_with_execution_context_on_explicit_domains_uses_explicit_stream() {
    let left = cudf::DataFrame::from_series([
        explicit_i32_series("id", &[1, 2, 3], 503),
        explicit_i64_series("val", &[10, 20, 30], 503),
    ])
    .unwrap();
    let right = cudf::DataFrame::from_series([
        explicit_i32_series("id", &[2, 3, 4], 503),
        explicit_i64_series("score", &[200, 300, 400], 503),
    ])
    .unwrap();

    let stream = CudaStream::new().unwrap();
    let explicit_stream = stream.__debug_raw_handle_value();
    let ctx = cudf::__DebugExecutionContext::new().with_stream(&stream);

    obs_reset();
    let joined = left.__debug_merge_in_context(&right, cudf::MergeOptions::new().how_inner().on(["id"]), &ctx).unwrap();

    assert_eq!(obs_last_stream(), explicit_stream);
    assert_eq!(joined.num_rows(), 2);
    assert_eq!(joined.__debug_execution_domain_id(), 503);
    assert_eq!(joined.__debug_execution_domain_stream_value(), explicit_stream);
}

#[test]
fn groupby_with_execution_context_on_explicit_domain_uses_explicit_stream() {
    let df = cudf::DataFrame::from_series([
        explicit_i32_series("key", &[1, 1, 2, 2], 504),
        explicit_i64_series("val", &[10, 20, 30, 40], 504),
    ])
    .unwrap();

    let stream = CudaStream::new().unwrap();
    let explicit_stream = stream.__debug_raw_handle_value();
    let ctx = cudf::__DebugExecutionContext::new().with_stream(&stream);

    obs_reset();
    let result =
        df.groupby(["key"]).__debug_agg_in_context(&[cudf::AggSpec::new("val", cudf::GroupByAgg::Sum)], &ctx).unwrap();

    assert_eq!(obs_last_stream(), explicit_stream);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.__debug_execution_domain_id(), 504);
    assert_eq!(result.__debug_execution_domain_stream_value(), explicit_stream);
}

#[test]
fn gather_with_execution_context_on_explicit_domain_uses_explicit_stream_and_retains_override() {
    let df = cudf::DataFrame::from_series([explicit_i32_series("id", &[3, 1, 2], 505)]).unwrap();
    let indices = explicit_i32_series("gather", &[2, 0], 505);

    let stream = CudaStream::new().unwrap();
    let explicit_stream = stream.__debug_raw_handle_value();
    let ctx = cudf::__DebugExecutionContext::new().with_stream(&stream);

    obs_reset();
    let gathered = df.__debug_gather_in_context(&indices, &ctx).unwrap();

    assert_eq!(obs_last_stream(), explicit_stream);
    assert_eq!(gathered.__debug_execution_domain_id(), 505);
    assert_eq!(gathered.__debug_execution_domain_stream_value(), explicit_stream);

    obs_reset();
    let vals: Vec<i32> = gathered.series("id").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(vals, vec![2, 3]);
    assert_eq!(obs_last_sync_stream(), explicit_stream);
}

#[test]
fn gather_with_execution_context_rejects_ambient_default_promotion() {
    let ids = cudf::Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let df = cudf::DataFrame::from_series([ids]).unwrap();
    let indices = cudf::Series::new::<i32>("gather", &[2, 0]).unwrap();

    let stream = CudaStream::new().unwrap();
    let ctx = cudf::__DebugExecutionContext::new().with_stream(&stream);

    let err = df.__debug_gather_in_context(&indices, &ctx).unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "execution_context::resolve_output_domain");
    assert!(
        err.context().iter().any(|(key, value)| *key == "operation_domain" && value.starts_with("ambient_default"))
    );
    assert!(err.context().iter().any(|(key, value)| *key == "borrowed_override_stream" && value == "true"));
}

#[test]
fn filter_with_execution_context_on_explicit_domain_uses_explicit_stream_and_retains_override() {
    let df = cudf::DataFrame::from_series([explicit_i32_series("id", &[3, 1, 2], 506)]).unwrap();
    let mask = cudf::Series::from_column(
        cudf::core::Column::from_slice(&[true, false, true])
            .unwrap()
            .__debug_rebind_execution_domain_explicit(506)
            .unwrap(),
    )
    .unwrap()
    .rename("mask");

    let stream = CudaStream::new().unwrap();
    let explicit_stream = stream.__debug_raw_handle_value();
    let ctx = cudf::__DebugExecutionContext::new().with_stream(&stream);

    obs_reset();
    let filtered = df.__debug_filter_in_context(&mask, &ctx).unwrap();

    assert_eq!(obs_last_stream(), explicit_stream);
    assert_eq!(filtered.__debug_execution_domain_id(), 506);
    assert_eq!(filtered.__debug_execution_domain_stream_value(), explicit_stream);

    obs_reset();
    let vals: Vec<i32> = filtered.series("id").unwrap().as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(vals, vec![3, 2]);
    assert_eq!(obs_last_sync_stream(), explicit_stream);
}

#[test]
fn filter_with_execution_context_rejects_ambient_default_promotion() {
    let ids = cudf::Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let df = cudf::DataFrame::from_series([ids]).unwrap();
    let mask = cudf::Series::new::<bool>("mask", &[true, false, true]).unwrap();

    let stream = CudaStream::new().unwrap();
    let ctx = cudf::__DebugExecutionContext::new().with_stream(&stream);

    let err = df.__debug_filter_in_context(&mask, &ctx).unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
    assert!(err.is_permanent());
    assert_eq!(err.operation(), "execution_context::resolve_output_domain");
    assert!(
        err.context().iter().any(|(key, value)| *key == "operation_domain" && value.starts_with("ambient_default"))
    );
    assert!(err.context().iter().any(|(key, value)| *key == "borrowed_override_stream" && value == "true"));
}

// ===========================================================================
// Category 4: Default-context fallback
//
// Verify that existing high-level APIs outside the scoped override path still use the default stream
// by checking the observation hook after a regular operation.
// ===========================================================================

/// Calling `Column::deep_copy()` (the safe API, which uses `cudf_column_copy`)
/// results in the default stream being observed.
#[test]
fn column_deep_copy_default_uses_default_stream() {
    let col = cudf::core::Column::from_slice(&[1i32, 2, 3]).unwrap();

    let default_stream = obs_default_stream();

    obs_reset();
    let _copy = col.deep_copy().unwrap();

    // Column::deep_copy() uses cudf_column_copy which internally calls
    // cudf_column_copy_ex with null options, resolving to default stream.
    let observed = obs_last_stream();
    assert_eq!(observed, default_stream, "Column::deep_copy() should use the default stream");
}

/// Existing `DataFrame::sort_values` (no context) should produce correct
/// results. This is the default-path baseline for
/// `DataFrame::in_context(&ctx).sort_values(...)`.
#[test]
fn sort_values_default_produces_correct_results() {
    let ids = cudf::Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let df = cudf::DataFrame::from_series([ids]).unwrap();

    let sorted = df.sort_values(["id"], cudf::SortOptions::default()).unwrap();

    let col = sorted.series("id").unwrap();
    let host: Vec<i32> = col.as_column_view().unwrap().to_vec().unwrap();
    assert_eq!(host, vec![1, 2, 3], "sort_values result should be correct");
}

/// Existing `DataFrame::merge` (no context) should produce correct results.
/// This is the default-path baseline for `DataFrame::in_context(&ctx).merge(...)`.
#[test]
fn merge_default_produces_correct_results() {
    let left_ids = cudf::Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let left_vals = cudf::Series::new::<f64>("val", &[10.0, 20.0, 30.0]).unwrap();
    let left = cudf::DataFrame::from_series([left_ids, left_vals]).unwrap();

    let right_ids = cudf::Series::new::<i32>("id", &[2, 3, 4]).unwrap();
    let right_scores = cudf::Series::new::<f64>("score", &[200.0, 300.0, 400.0]).unwrap();
    let right = cudf::DataFrame::from_series([right_ids, right_scores]).unwrap();

    let joined = left.merge(&right, cudf::MergeOptions::new().how_inner().on(["id"])).unwrap();

    assert_eq!(joined.num_rows(), 2, "inner join on [2,3] should yield 2 rows");
}

/// Existing `GroupBy::agg` (no context) should produce correct results.
/// This is the default-path baseline for `GroupBy::in_context(&ctx).agg(...)`.
#[test]
fn groupby_agg_default_produces_correct_results() {
    let keys = cudf::Series::new::<i32>("key", &[1, 1, 2, 2]).unwrap();
    let vals = cudf::Series::new::<i64>("val", &[10, 20, 30, 40]).unwrap();
    let df = cudf::DataFrame::from_series([keys, vals]).unwrap();

    let result = df.groupby(["key"]).agg(&[cudf::AggSpec::new("val", cudf::GroupByAgg::Sum)]).unwrap();

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

    let pool_mr = DeviceMemoryResource::pool(
        16 * 1024 * 1024, // 16 MB initial
        0,                // unbounded max
    )
    .expect("failed to create pool MR");
    let borrowed_mr = pool_mr.as_borrowed_sys();

    let ctx = cudf_sys::ExecutionContext::new().with_memory_resource(&borrowed_mr);

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

    let stream = CudaStream::new().expect("failed to create stream");
    let pool_mr = DeviceMemoryResource::pool(16 * 1024 * 1024, 0).expect("failed to create pool MR");
    let borrowed_stream = stream.as_borrowed_sys();
    let borrowed_mr = pool_mr.as_borrowed_sys();

    let ctx = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);

    let explicit_stream = stream.__debug_raw_handle_value();
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
    assert_send_sync::<CudaStream>();
}

/// DeviceMemoryResource is Send + Sync (required for Arc sharing).
#[test]
fn device_memory_resource_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DeviceMemoryResource>();
}

/// CudaEvent is Send + Sync (required for retained owner sharing).
#[test]
fn cuda_event_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaEvent>();
}

/// CudaStream can be created and synchronized.
#[test]
fn cuda_stream_create_and_sync() {
    let stream = CudaStream::new().unwrap();
    stream.synchronize().unwrap();
    assert!(stream.is_ready().unwrap());
}

/// CudaStream is accessible via the `cudf::runtime` module path (not re-exported at crate root).
#[test]
fn cuda_stream_accessible_via_module_path() {
    let stream = CudaStream::new().unwrap();
    stream.synchronize().unwrap();
}

/// DeviceMemoryResource can be created (cuda variant).
#[test]
fn device_memory_resource_create_cuda() {
    let _mr = DeviceMemoryResource::cuda().unwrap();
}

/// DeviceMemoryResource can be created (pool variant).
#[test]
fn device_memory_resource_create_pool() {
    let _mr = DeviceMemoryResource::pool(16 * 1024 * 1024, 0).unwrap();
}

/// `CudaStream::as_borrowed_sys` preserves the sys-handle identity.
#[test]
fn cuda_stream_as_borrowed_sys_preserves_handle() {
    let stream = CudaStream::new().unwrap();
    let borrowed = stream.as_borrowed_sys();
    assert_eq!(stream.__debug_raw_handle_value(), obs_stream_handle_value(borrowed.as_raw()));
}

/// `CudaEvent::as_borrowed_sys` preserves the sys-handle identity.
#[test]
fn cuda_event_as_borrowed_sys_preserves_handle() {
    let stream = CudaStream::new().unwrap();
    let event = CudaEvent::new().unwrap();
    event.record(&stream).unwrap();

    let borrowed = event.as_borrowed_sys();
    assert_eq!(event.__debug_raw_handle_value(), borrowed.as_raw() as usize);
}

/// `DeviceMemoryResource::as_borrowed_sys` preserves the sys-handle identity.
#[test]
fn device_memory_resource_as_borrowed_sys_preserves_handle() {
    let mr = DeviceMemoryResource::cuda().unwrap();
    let borrowed = mr.as_borrowed_sys();
    let mut underlying = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_test_memory_resource_raw(borrowed.as_raw(), &mut underlying) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(mr.__debug_raw_handle_value(), underlying as usize);
}

// ===========================================================================
// Category 7: Exec options validation (negative tests)
//
// These tests construct raw `cudf_exec_options_t` structs with deliberately
// invalid fields and verify that the C++ shim rejects them before performing
// any side-effects (such as device switching).
// ===========================================================================

/// A `cudf_exec_options_t` with `struct_size` smaller than the actual struct
/// is rejected with `CUDF_ERROR_INVALID_OPERATION`.
#[test]
fn exec_options_struct_size_too_small_returns_error() {
    let col = create_i32_column_raw(&[1, 2, 3]);

    let mut opts = cudf_sys::cudf_exec_options_t::defaults();
    opts.struct_size = 8; // too small

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_copy_ex(col, &opts as *const cudf_sys::cudf_exec_options_t as *const _, &mut out)
    };

    assert_eq!(
        status,
        cudf_sys::CUDF_ERROR_INVALID_OPERATION,
        "struct_size too small should return CUDF_ERROR_INVALID_OPERATION"
    );
    assert!(out.is_null(), "output handle should remain null on validation error");

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

/// A `cudf_exec_options_t` with unknown flag bits is rejected with
/// `CUDF_ERROR_INVALID_OPERATION`.
#[test]
fn exec_options_unknown_flags_returns_error() {
    let col = create_i32_column_raw(&[1, 2, 3]);

    let mut opts = cudf_sys::cudf_exec_options_t::defaults();
    opts.flags = 0x8000_0000; // unknown bit

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_copy_ex(col, &opts as *const cudf_sys::cudf_exec_options_t as *const _, &mut out)
    };

    assert_eq!(
        status,
        cudf_sys::CUDF_ERROR_INVALID_OPERATION,
        "unknown flag bits should return CUDF_ERROR_INVALID_OPERATION"
    );
    assert!(out.is_null(), "output handle should remain null on validation error");

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

/// A `cudf_exec_options_t` with a non-zero reserved field is rejected with
/// `CUDF_ERROR_INVALID_OPERATION`.
#[test]
fn exec_options_nonzero_reserved_returns_error() {
    let col = create_i32_column_raw(&[1, 2, 3]);

    let mut opts = cudf_sys::cudf_exec_options_t::defaults();
    opts.reserved[2] = 0xDEAD_BEEF;

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_copy_ex(col, &opts as *const cudf_sys::cudf_exec_options_t as *const _, &mut out)
    };

    assert_eq!(
        status,
        cudf_sys::CUDF_ERROR_INVALID_OPERATION,
        "non-zero reserved field should return CUDF_ERROR_INVALID_OPERATION"
    );
    assert!(out.is_null(), "output handle should remain null on validation error");

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

/// A `cudf_exec_options_t` with an explicit device ordinal that does not exist
/// is rejected with `CUDF_ERROR_CUDA` before any execution observation occurs.
#[test]
fn exec_options_invalid_explicit_device_returns_error_without_side_effects() {
    let col = create_i32_column_raw(&[1, 2, 3]);
    let before = DeviceId::current().unwrap();

    let mut opts = cudf_sys::cudf_exec_options_t::defaults();
    opts.flags = cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT;
    opts.device = DeviceId::count().unwrap() as i32 + 1000;

    obs_reset();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_copy_ex(col, &opts as *const cudf_sys::cudf_exec_options_t as *const _, &mut out)
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_CUDA, "invalid explicit device should return CUDF_ERROR_CUDA");
    assert!(out.is_null(), "output handle should remain null on device-selection error");
    assert_eq!(obs_last_device(), -1, "failed device selection should not record execution observation");
    assert_eq!(DeviceId::current().unwrap(), before, "failed device selection should not change the ambient device");

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

/// A `cudf_exec_options_t` constructed via `defaults()` (correct struct_size,
/// zero flags, zero reserved) is accepted.
#[test]
fn exec_options_valid_defaults_succeeds() {
    let col = create_i32_column_raw(&[1, 2, 3]);

    let opts = cudf_sys::cudf_exec_options_t::defaults();

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_copy_ex(col, &opts as *const cudf_sys::cudf_exec_options_t as *const _, &mut out)
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "valid defaults should succeed");
    assert!(!out.is_null(), "output handle should be non-null on success");

    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_destroy(col);
    };
}
