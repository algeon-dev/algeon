#![allow(dead_code)]

use std::{
    ffi::{CStr, c_char, c_void},
    ptr,
    sync::{Mutex, MutexGuard},
};

static SHIM_STATE_LOCK: Mutex<()> = Mutex::new(());

pub const NO_SYNC_RECORDED: usize = usize::MAX;

pub fn lock_shim_state() -> MutexGuard<'static, ()> {
    SHIM_STATE_LOCK.lock().unwrap()
}

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
        unsafe { cudf_sys::test_set_current_device_resource_raw(self.previous_raw) }
            .expect("failed to restore previous current device resource");
    }
}

pub fn last_error_message() -> String {
    let ptr = unsafe { cudf_sys::cudf_last_error_message() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

pub fn reset_exec_observation() {
    unsafe { cudf_sys::cudf_test_exec_reset_observation() };
}

pub fn last_exec_stream_value() -> usize {
    let mut out = usize::MAX;
    let st = unsafe { cudf_sys::cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    out
}

pub fn last_exec_sync_stream_value() -> usize {
    let mut out = usize::MAX;
    let st = unsafe { cudf_sys::cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    out
}

pub fn last_exec_mr_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    out
}

pub fn default_stream_value() -> usize {
    let mut out = usize::MAX;
    let st = unsafe { cudf_sys::cudf_test_default_stream_value(&mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    out
}

pub fn stream_value(stream: cudf_sys::cudf_stream_handle) -> usize {
    let mut out = usize::MAX;
    let st = unsafe { cudf_sys::cudf_test_stream_handle_value(stream, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    out
}

pub fn current_device_resource_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
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

    pub fn is_ready(&self) -> bool {
        let mut out = 0;
        let st = unsafe { cudf_sys::cudf_stream_is_ready(self.raw, &mut out) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        out != 0
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

    pub fn synchronize(&self) {
        let st = unsafe { cudf_sys::cudf_event_synchronize(self.raw) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    }

    pub fn is_ready(&self) -> bool {
        let mut out = 0;
        let st = unsafe { cudf_sys::cudf_event_is_ready(self.raw, &mut out) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        out != 0
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

    pub fn new_pool(initial_pool_size: usize, maximum_pool_size: usize) -> Self {
        let device = current_device();
        let mut raw = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_memory_resource_create_pool(initial_pool_size, maximum_pool_size, &mut raw) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!raw.is_null());
        Self { raw, device }
    }

    pub fn new_test_stats() -> Self {
        let device = current_device();
        let mut raw = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_test_memory_resource_create_stats(&mut raw) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert!(!raw.is_null());
        Self { raw, device }
    }

    pub fn as_raw(&self) -> cudf_sys::cudf_memory_resource_handle {
        self.raw
    }

    pub fn borrowed(&self) -> cudf_sys::BorrowedMemoryResource<'_> {
        unsafe { cudf_sys::BorrowedMemoryResource::from_raw(self.raw) }
    }

    pub fn allocation_count(&self) -> i64 {
        let mut out = 0_i64;
        let st = unsafe { cudf_sys::cudf_test_memory_resource_allocation_count(self.raw, &mut out) };
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
