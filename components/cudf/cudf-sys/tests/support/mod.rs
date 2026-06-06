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

pub fn make_stats_memory_resource() -> cudf_sys::MemoryResource {
    cudf_sys::MemoryResource::new_test_stats().unwrap()
}

pub fn owned_string_value(ptr: *mut c_char, len: i32) -> String {
    assert!(!ptr.is_null());
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    let value = String::from_utf8(bytes.to_vec()).unwrap();
    unsafe { cudf_sys::cudf_string_free(ptr) };
    value
}
