use std::{ffi::c_void, ptr};

use cudf::{Column, DataFrame, Runtime, Series, core::memory::DeviceMemoryResource, runtime::CudaStream};
use cudf_sys::{
    cudf_test_exec_last_mr_raw, cudf_test_exec_last_stream_value, cudf_test_exec_last_sync_stream_value,
    cudf_test_exec_reset_observation,
};

#[allow(dead_code)]
pub fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[3i32, 1, 4, 1, 5]).unwrap();
    let scores = Column::from_slice(&[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

#[allow(dead_code)]
pub fn reset_exec_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

#[allow(dead_code)]
pub fn reset_sync_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

#[allow(dead_code)]
pub fn last_exec_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last exec stream observation");
    out
}

#[allow(dead_code)]
pub fn last_exec_mr_value() -> usize {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last exec mr observation");
    out as usize
}

#[allow(dead_code)]
pub fn last_sync_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last synchronized stream observation");
    out
}

#[allow(dead_code)]
pub fn explicit_i32_column(data: &[i32], identity: u64) -> Column {
    Column::from_slice(data).unwrap().__debug_rebind_execution_domain_explicit(identity).unwrap()
}

#[allow(dead_code)]
pub fn explicit_i32_series(name: &str, data: &[i32], identity: u64) -> Series {
    Series::from_column(explicit_i32_column(data, identity)).unwrap().rename(name)
}

#[allow(dead_code)]
pub fn explicit_bool_column(data: &[bool], identity: u64) -> Column {
    Column::from_slice(data).unwrap().__debug_rebind_execution_domain_explicit(identity).unwrap()
}

#[allow(dead_code)]
pub fn explicit_bool_series(name: &str, data: &[bool], identity: u64) -> Series {
    Series::from_column(explicit_bool_column(data, identity)).unwrap().rename(name)
}

#[allow(dead_code)]
pub fn sample_runtime() -> Runtime {
    let stream = CudaStream::new().unwrap();
    let mr = DeviceMemoryResource::pool(16 * 1024 * 1024, 0).unwrap();
    Runtime::builder().with_stream(stream).with_memory_resource(mr).build().unwrap()
}
