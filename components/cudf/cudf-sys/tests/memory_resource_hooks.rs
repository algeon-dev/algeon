mod support;

use std::{
    ffi::c_void,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

use support::{
    CurrentDeviceResourceRestore, OwnedMemoryResource, OwnedStream, current_device, current_device_resource_raw,
    lock_shim_state, required_exec_options,
};

fn make_i32_column_with_memory_resource(data: &[i32], mr: &cudf_sys::BorrowedMemoryResource<'_>) -> *mut c_void {
    let (status, out) = try_make_i32_column_with_memory_resource(data, mr);
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn try_make_i32_column_with_memory_resource(
    data: &[i32],
    mr: &cudf_sys::BorrowedMemoryResource<'_>,
) -> (cudf_sys::cudf_status_t, *mut c_void) {
    let stream = OwnedStream::new();
    let stream = stream.borrowed();
    let exec = required_exec_options(&stream, mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &exec,
            &mut out,
        )
    };
    (st, out)
}

fn create_cuda_memory_resource() -> *mut c_void {
    let mut mr: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());
    mr
}

fn create_pool_memory_resource() -> *mut c_void {
    let mut mr: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_pool(1 << 20, 1 << 24, &mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());
    mr
}

struct RawMemoryResource {
    raw: *mut c_void,
    device: i32,
}

impl RawMemoryResource {
    fn from_raw(raw: *mut c_void) -> Self {
        assert!(!raw.is_null());
        Self { raw, device: current_device() }
    }

    fn raw(&self) -> *mut c_void {
        self.raw
    }

    fn borrowed(&self) -> cudf_sys::BorrowedMemoryResource<'_> {
        unsafe { cudf_sys::BorrowedMemoryResource::from_raw(self.raw) }
    }
}

impl Drop for RawMemoryResource {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let status = unsafe { cudf_sys::cudf_memory_resource_destroy(self.raw, self.device) };
            assert_eq!(status, cudf_sys::CUDF_SUCCESS);
            self.raw = ptr::null_mut();
        }
    }
}

fn create_limiting_memory_resource(upstream: *mut c_void, allocation_limit: usize) -> RawMemoryResource {
    let mut raw = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_limiting(upstream, allocation_limit, &mut raw) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    RawMemoryResource::from_raw(raw)
}

fn create_budget_tracking_memory_resource(upstream: *mut c_void, allocation_limit: usize) -> RawMemoryResource {
    let mut raw = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_budget_tracking(upstream, allocation_limit, &mut raw) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    RawMemoryResource::from_raw(raw)
}

fn create_failure_callback_memory_resource(
    upstream: *mut c_void,
    callback: cudf_sys::cudf_memory_failure_callback_t,
    callback_arg: *mut c_void,
) -> RawMemoryResource {
    let mut raw = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_memory_resource_create_failure_callback(upstream, callback, callback_arg, &mut raw) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    RawMemoryResource::from_raw(raw)
}

unsafe extern "C" fn stateless_decline_callback(_bytes: usize, _callback_arg: *mut c_void) -> i32 {
    0
}

unsafe extern "C" fn retry_once_then_decline_callback(bytes: usize, callback_arg: *mut c_void) -> i32 {
    if bytes == 0 || callback_arg.is_null() {
        return 0;
    }
    let calls = unsafe { &*(callback_arg as *const AtomicUsize) };
    i32::from(calls.fetch_add(1, Ordering::SeqCst) == 0)
}

#[test]
fn stats_memory_resource_can_be_installed_as_current_resource_and_used_explicitly() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let stats = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let borrowed = stats.borrowed();

    cudf_sys::test_set_current_device_resource(&borrowed).expect("failed to install current test allocator");

    let installed_raw = current_device_resource_raw();
    assert_ne!(installed_raw, previous_raw, "test allocator should replace the previous current allocator");

    let before = stats.allocation_count();
    let col = make_i32_column_with_memory_resource(&[1, 2, 3, 4], &borrowed);
    let after = stats.allocation_count();

    assert!(
        after > before,
        "allocating a column through the installed stats resource should increase its allocation count"
    );

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

#[test]
fn destroying_current_memory_resource_returns_invalid_operation_and_preserves_current_resource() {
    let _shim_state = lock_shim_state();
    let device = current_device();
    let previous_raw = current_device_resource_raw();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);

    for (name, create) in [
        ("cuda", create_cuda_memory_resource as fn() -> *mut c_void),
        ("pool", create_pool_memory_resource as fn() -> *mut c_void),
    ] {
        let mr = create();

        let mut installed_raw: *mut c_void = ptr::null_mut();
        let status = unsafe { cudf_sys::cudf_memory_resource_raw(mr, &mut installed_raw) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        assert!(!installed_raw.is_null());

        let status = unsafe { cudf_sys::cudf_test_memory_resource_set_current(mr) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        assert_eq!(
            current_device_resource_raw(),
            installed_raw,
            "{name} test allocator should be installed as the current device resource before destroy"
        );

        let destroy_status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
        let current_after_destroy = current_device_resource_raw();

        unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource before cleanup");

        if destroy_status == cudf_sys::CUDF_ERROR_INVALID_OPERATION {
            let cleanup_status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
            assert_eq!(cleanup_status, cudf_sys::CUDF_SUCCESS);
        }

        assert_eq!(destroy_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
        assert_eq!(
            current_after_destroy, installed_raw,
            "failed destroy of the current {name} device resource must not rewrite global current resource state"
        );
    }
}

#[test]
fn destroying_non_current_memory_resource_still_succeeds_after_current_destroy_rejection() {
    let _shim_state = lock_shim_state();
    let device = current_device();
    let previous_raw = current_device_resource_raw();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);

    let mr = create_cuda_memory_resource();

    let status = unsafe { cudf_sys::cudf_test_memory_resource_set_current(mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let destroy_status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
    assert_eq!(destroy_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource before cleanup");

    let cleanup_status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, device) };
    assert_eq!(cleanup_status, cudf_sys::CUDF_SUCCESS);

    let fresh_mr = create_cuda_memory_resource();
    let fresh_destroy_status = unsafe { cudf_sys::cudf_memory_resource_destroy(fresh_mr, device) };
    assert_eq!(fresh_destroy_status, cudf_sys::CUDF_SUCCESS);
}

#[test]
fn raw_memory_resource_accessor_returns_underlying_device_resource_pointer() {
    let mr = OwnedMemoryResource::new_cuda();
    let raw = mr.raw_device_resource();
    assert!(!raw.is_null(), "underlying device memory resource pointer must be non-null");
}

#[test]
fn raw_stream_accessor_returns_underlying_cuda_stream_pointer() {
    let stream = OwnedStream::new();
    let raw = stream.raw_cuda_stream();
    assert!(!raw.is_null(), "underlying CUDA stream pointer must be non-null");
}

#[test]
fn failure_callback_memory_resource_rejects_null_callback() {
    let upstream = RawMemoryResource::from_raw(create_cuda_memory_resource());
    let mut out = ptr::null_mut();

    let status = unsafe {
        cudf_sys::cudf_memory_resource_create_failure_callback(upstream.raw(), None, ptr::null_mut(), &mut out)
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn failure_callback_memory_resource_allows_null_callback_arg() {
    let upstream = RawMemoryResource::from_raw(create_cuda_memory_resource());
    let _callback_mr =
        create_failure_callback_memory_resource(upstream.raw(), Some(stateless_decline_callback), ptr::null_mut());
}

#[test]
fn failure_callback_memory_resource_retries_until_callback_declines() {
    let upstream = RawMemoryResource::from_raw(create_cuda_memory_resource());
    let limiting = create_limiting_memory_resource(upstream.raw(), 1);
    let calls = AtomicUsize::new(0);
    let callback_mr = create_failure_callback_memory_resource(
        limiting.raw(),
        Some(retry_once_then_decline_callback),
        &calls as *const AtomicUsize as *mut c_void,
    );
    let borrowed = callback_mr.borrowed();

    let (status, out) = try_make_i32_column_with_memory_resource(&[1, 2, 3], &borrowed);

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out.is_null());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn failure_callback_memory_resource_retries_budget_tracking_limit_failures() {
    let upstream = RawMemoryResource::from_raw(create_cuda_memory_resource());
    let budget_tracking = create_budget_tracking_memory_resource(upstream.raw(), 1);
    let calls = AtomicUsize::new(0);
    let callback_mr = create_failure_callback_memory_resource(
        budget_tracking.raw(),
        Some(retry_once_then_decline_callback),
        &calls as *const AtomicUsize as *mut c_void,
    );
    let borrowed = callback_mr.borrowed();

    let (status, out) = try_make_i32_column_with_memory_resource(&[1, 2, 3], &borrowed);

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out.is_null());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
