// Shared by multiple integration-test crates, each of which only uses a subset
// of these helpers.
#![allow(dead_code)]

use std::{
    env,
    ffi::{CStr, c_char, c_void},
    panic, ptr,
};

use cudf_sys::{
    CUDF_SUCCESS,
    test_support::{
        cudf_test_current_device_resource_raw, cudf_test_default_stream_value, cudf_test_exec_last_mr_raw,
        cudf_test_exec_last_stream_value, cudf_test_exec_last_sync_stream_value, cudf_test_exec_reset_observation,
        cudf_test_memory_resource_active_allocation_count, cudf_test_memory_resource_allocation_count,
        cudf_test_memory_resource_create_stats, cudf_test_memory_resource_set_current,
        cudf_test_memory_resource_set_current_raw, cudf_test_stream_handle_value,
    },
};

pub const NO_SYNC_RECORDED: usize = usize::MAX;

pub struct CurrentDeviceResourceRestore {
    previous_raw: *mut c_void,
}

impl CurrentDeviceResourceRestore {
    pub fn new(previous_raw: *mut c_void) -> Self {
        Self { previous_raw }
    }
}

impl Drop for CurrentDeviceResourceRestore {
    fn drop(&mut self) {
        unsafe { set_current_device_resource_raw(self.previous_raw) }
            .expect("failed to restore previous current device resource");
    }
}

/// Install an owned test resource as the process-global current RMM resource.
///
/// This private test helper deliberately wraps only the generated test-support
/// declaration; it is not part of the `cudf_sys` production or test API.
pub fn set_current_device_resource(mr: &cudf_sys::BorrowedMemoryResource<'_>) -> Result<(), cudf_sys::cudf_status_t> {
    let status = unsafe { cudf_test_memory_resource_set_current(mr.as_raw()) };
    if status == CUDF_SUCCESS { Ok(()) } else { Err(status) }
}

/// Restore a raw RMM resource pointer saved from the test process.
///
/// # Safety
/// `raw_mr` must remain a valid RMM resource for this test scope.
pub unsafe fn set_current_device_resource_raw(raw_mr: *mut c_void) -> Result<(), cudf_sys::cudf_status_t> {
    let status = unsafe { cudf_test_memory_resource_set_current_raw(raw_mr) };
    if status == CUDF_SUCCESS { Ok(()) } else { Err(status) }
}

pub fn last_error_message() -> String {
    let ptr = unsafe { cudf_sys::cudf_last_error_message() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

pub fn reset_exec_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

pub fn last_exec_stream_value() -> usize {
    let mut out = usize::MAX;
    let st = unsafe { cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(st, CUDF_SUCCESS);
    out
}

pub fn last_exec_sync_stream_value() -> usize {
    let mut out = usize::MAX;
    let st = unsafe { cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(st, CUDF_SUCCESS);
    out
}

pub fn last_exec_mr_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let st = unsafe { cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(st, CUDF_SUCCESS);
    out
}

pub fn default_stream_value() -> usize {
    let mut out = usize::MAX;
    let st = unsafe { cudf_test_default_stream_value(&mut out) };
    assert_eq!(st, CUDF_SUCCESS);
    out
}

pub fn stream_value(stream: cudf_sys::cudf_stream_handle) -> usize {
    let mut out = usize::MAX;
    let st = unsafe { cudf_test_stream_handle_value(stream, &mut out) };
    assert_eq!(st, CUDF_SUCCESS);
    out
}

pub fn current_device_resource_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let st = unsafe { cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(st, CUDF_SUCCESS);
    out
}

pub fn current_device() -> i32 {
    let mut out = -1;
    let st = unsafe { cudf_sys::cudf_cuda_get_device(&mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    out
}

pub fn set_current_device(device: i32) {
    let st = unsafe { cudf_sys::cudf_cuda_set_device(device) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
}

pub fn visible_device_count() -> i32 {
    let mut count = 0;
    let st = unsafe { cudf_sys::cudf_cuda_get_device_count(&mut count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    count
}

pub fn required_exec_options(
    stream: &cudf_sys::BorrowedStream<'_>,
    mr: &cudf_sys::BorrowedMemoryResource<'_>,
) -> cudf_sys::cudf_exec_options_t {
    let mut options = cudf_sys::cudf_exec_options_t::defaults();
    options.device = current_device();
    options.flags = cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT;
    options.stream = stream.as_raw();
    options.mr = mr.as_raw();
    options
}

pub struct CurrentDeviceRestore {
    previous: i32,
}

impl CurrentDeviceRestore {
    pub fn new() -> Self {
        Self { previous: current_device() }
    }
}

impl Drop for CurrentDeviceRestore {
    fn drop(&mut self) {
        set_current_device(self.previous);
    }
}

pub struct OwnedStream {
    raw: cudf_sys::cudf_stream_handle,
}

impl OwnedStream {
    pub fn new() -> Self {
        let mut raw = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_stream_create(&mut raw) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!raw.is_null());
        Self { raw }
    }

    pub fn as_raw(&self) -> cudf_sys::cudf_stream_handle {
        self.raw
    }

    pub fn borrowed(&self) -> cudf_sys::BorrowedStream<'_> {
        unsafe { cudf_sys::BorrowedStream::from_raw(self.raw) }
    }

    pub fn synchronize(&self) {
        let st = unsafe { cudf_sys::cudf_stream_synchronize(self.raw) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    }

    pub fn raw_cuda_stream(&self) -> *mut c_void {
        let mut out = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_stream_raw(self.raw, &mut out) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!out.is_null());
        out
    }
}

impl Drop for OwnedStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cudf_sys::cudf_stream_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

pub struct OwnedEvent {
    raw: cudf_sys::cudf_event_handle,
}

impl OwnedEvent {
    pub fn new() -> Self {
        let mut raw = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_event_create(&mut raw) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!raw.is_null());
        Self { raw }
    }

    pub fn as_raw(&self) -> cudf_sys::cudf_event_handle {
        self.raw
    }

    pub fn borrowed(&self) -> cudf_sys::BorrowedEvent<'_> {
        unsafe { cudf_sys::BorrowedEvent::from_raw(self.raw) }
    }

    pub fn record(&self, stream: &OwnedStream) {
        let st = unsafe { cudf_sys::cudf_event_record(self.raw, stream.as_raw()) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    }

    pub fn raw_cuda_event(&self) -> *mut c_void {
        let mut out = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_event_raw(self.raw, &mut out) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!out.is_null());
        out
    }
}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cudf_sys::cudf_event_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

pub struct OwnedMemoryResource {
    raw: cudf_sys::cudf_memory_resource_handle,
    device: i32,
}

impl OwnedMemoryResource {
    pub fn new_cuda() -> Self {
        let device = current_device();
        let mut raw = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut raw) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!raw.is_null());
        Self { raw, device }
    }

    pub fn new_test_stats() -> Self {
        let device = current_device();
        let mut raw = ptr::null_mut();
        let st = unsafe { cudf_test_memory_resource_create_stats(&mut raw) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!raw.is_null());
        Self { raw, device }
    }

    pub fn borrowed(&self) -> cudf_sys::BorrowedMemoryResource<'_> {
        unsafe { cudf_sys::BorrowedMemoryResource::from_raw(self.raw) }
    }

    pub fn allocation_count(&self) -> i64 {
        let mut out = 0_i64;
        let st = unsafe { cudf_test_memory_resource_allocation_count(self.raw, &mut out) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        out
    }

    pub fn active_allocation_count(&self) -> i64 {
        let mut out = 0_i64;
        let st = unsafe { cudf_test_memory_resource_active_allocation_count(self.raw, &mut out) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        out
    }

    pub fn raw_device_resource(&self) -> *mut c_void {
        let mut out = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_memory_resource_raw(self.raw, &mut out) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!out.is_null());
        out
    }
}

impl Drop for OwnedMemoryResource {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let st = unsafe { cudf_sys::cudf_memory_resource_destroy(self.raw, self.device) };
            assert_eq!(st, cudf_sys::CUDF_SUCCESS);
            self.raw = ptr::null_mut();
        }
    }
}

pub fn make_stats_memory_resource() -> OwnedMemoryResource {
    OwnedMemoryResource::new_test_stats()
}

pub fn owned_string_value(ptr: *mut c_char, len: i32) -> String {
    assert!(!ptr.is_null());
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    let value = String::from_utf8(bytes.to_vec()).unwrap();
    unsafe { cudf_sys::cudf_string_free(ptr) };
    value
}

/// Runs cases in order on the current thread and reports all failing names.
///
/// `ALGEON_TEST_CASES` selects names containing any comma-separated substring.
/// An unset or empty selector runs every case. A selector matching no case
/// fails the table. Cases must restore any state needed by subsequent cases
/// during unwinding; a panic does not stop the remaining cases.
pub fn run_named_cases(cases: &[(&str, fn())]) {
    let selector = env::var_os("ALGEON_TEST_CASES")
        .map(|value| value.into_string().expect("ALGEON_TEST_CASES must contain Unicode"))
        .unwrap_or_default();
    run_selected_cases(cases, &selector);
}

fn run_selected_cases(cases: &[(&str, fn())], selector: &str) {
    let filters = selector.split(',').map(str::trim).filter(|filter| !filter.is_empty()).collect::<Vec<_>>();
    let selected = cases
        .iter()
        .filter(|(name, _)| selector.trim().is_empty() || filters.iter().any(|filter| name.contains(filter)))
        .collect::<Vec<_>>();
    assert!(!selected.is_empty(), "ALGEON_TEST_CASES={selector:?} matched no cases");
    let failed = selected
        .iter()
        .filter(|(name, case)| {
            eprintln!("case `{name}` running");
            unsafe { cudf_sys::cudf_clear_error() };
            reset_exec_observation();
            let panicked = panic::catch_unwind(*case).is_err();
            if panicked {
                eprintln!("case `{name}` failed; its panic message is above");
            }
            panicked
        })
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    assert!(failed.is_empty(), "{} of {} cases failed: {}", failed.len(), selected.len(), failed.join(", "));
}
