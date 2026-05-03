use std::{
    ffi::c_void,
    ptr,
    sync::{Mutex, MutexGuard, OnceLock},
};

use cudf::{
    DataFrame, DataType, DeviceId, ErrorCode, HostColumnData, Runtime, Series, TimestampResolution, TypeId,
    core::{
        Column, CudfType, DeviceBuffer, Scalar,
        error::{FactKeyName, FactValue},
    },
    runtime::CudaStream,
    testing::{self, prelude::*},
};
use cudf_sys::{
    CUDF_SUCCESS, cudf_cuda_get_device, cudf_cuda_set_device, cudf_test_event_last_wait_stream_raw,
    cudf_test_event_reset_observation, cudf_test_event_stream_wait_call_count,
    cudf_test_event_stream_wait_call_count_for_stream_raw, cudf_test_exec_last_device, cudf_test_exec_last_mr_raw,
    cudf_test_exec_last_stream_value, cudf_test_exec_last_sync_stream_value, cudf_test_exec_reset_observation,
};
use rapids_interop::InteropError;

#[allow(dead_code)]
pub fn sample_df() -> DataFrame {
    let ids = column_from_slice(&[3i32, 1, 4, 1, 5]).unwrap();
    let scores = column_from_slice(&[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap();
    dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

thread_local! {
    static TEST_RUNTIME: Runtime = sample_runtime();
}

#[allow(dead_code)]
pub fn with_test_runtime<T>(f: impl FnOnce(&Runtime) -> T) -> T {
    TEST_RUNTIME.with(f)
}

#[allow(dead_code)]
pub fn column_from_slice<T: CudfType>(data: &[T]) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_slice(data))
}

#[allow(dead_code)]
pub fn column_from_slice_nullable<T: CudfType>(data: &[T], validity: &[bool]) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_slice_nullable(data, validity))
}

#[allow(dead_code)]
pub fn column_from_strs(data: &[&str]) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_strs(data))
}

#[allow(dead_code)]
pub fn column_from_strs_nullable(data: &[&str], validity: &[bool]) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_strs_nullable(data, validity))
}

#[allow(dead_code)]
pub fn null_column(dtype: impl Into<DataType>, len: usize) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.null_column(dtype, len))
}

#[allow(dead_code)]
pub fn empty_column(dtype: impl Into<DataType>) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.empty_column(dtype))
}

#[allow(dead_code)]
pub unsafe fn column_from_device_ptr(device_ptr: *const c_void, len: usize, type_id: TypeId) -> cudf::Result<Column> {
    with_test_runtime(|rt| unsafe { rt.column_from_device_ptr(device_ptr, len, type_id) })
}

#[allow(dead_code)]
pub fn column_from_device_buffer(buffer: DeviceBuffer, row_count: usize, type_id: TypeId) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_device_buffer(buffer, row_count, type_id))
}

#[allow(dead_code)]
pub fn column_from_device_buffer_nullable(
    data_buffer: DeviceBuffer,
    null_mask_buffer: DeviceBuffer,
    null_count: usize,
    row_count: usize,
    type_id: TypeId,
) -> cudf::Result<Column> {
    with_test_runtime(|rt| {
        rt.column_from_device_buffer_nullable(data_buffer, null_mask_buffer, null_count, row_count, type_id)
    })
}

#[allow(dead_code)]
pub fn scalar<T: CudfType>(value: T) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.scalar(value))
}

#[allow(dead_code)]
pub fn string_scalar(value: &str) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.string_scalar(value))
}

#[allow(dead_code)]
pub fn null_scalar<T: CudfType>() -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.null_scalar::<T>())
}

#[allow(dead_code)]
pub fn null_string_scalar() -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.null_string_scalar())
}

#[allow(dead_code)]
pub fn decimal32_raw_scalar(raw: i32, scale: i32) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.decimal32_raw_scalar(raw, scale))
}

#[allow(dead_code)]
pub fn decimal64_raw_scalar(raw: i64, scale: i32) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.decimal64_raw_scalar(raw, scale))
}

#[allow(dead_code)]
pub fn decimal128_raw_scalar(raw: i128, scale: i32) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.decimal128_raw_scalar(raw, scale))
}

#[allow(dead_code)]
pub fn timestamp_raw_scalar(raw: i64, resolution: TimestampResolution) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.timestamp_raw_scalar(raw, resolution))
}

#[allow(dead_code)]
pub fn duration_raw_scalar(raw: i64, resolution: TimestampResolution) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.duration_raw_scalar(raw, resolution))
}

#[allow(dead_code)]
pub fn series<T: HostColumnData>(name: impl Into<String>, data: &[T]) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.series(data))
}

#[allow(dead_code)]
pub fn unnamed_series<T: HostColumnData>(data: &[T]) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.series(data))
}

#[allow(dead_code)]
pub fn nullable_series<T: HostColumnData>(
    name: impl Into<String>,
    data: &[T],
    validity: &[bool],
) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.nullable_series(data, validity))
}

#[allow(dead_code)]
pub fn unnamed_nullable_series<T: HostColumnData>(data: &[T], validity: &[bool]) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.nullable_series(data, validity))
}

#[allow(dead_code)]
pub fn series_opt<T: CudfType>(name: impl Into<String>, data: &[Option<T>]) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.series_opt(data))
}

#[allow(dead_code)]
pub fn unnamed_series_opt<T: CudfType>(data: &[Option<T>]) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.series_opt(data))
}

#[allow(dead_code)]
pub fn null_series(name: impl Into<String>, dtype: impl Into<DataType>, len: usize) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.nulls(dtype, len))
}

#[allow(dead_code)]
pub fn unnamed_null_series(dtype: impl Into<DataType>, len: usize) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.nulls(dtype, len))
}

#[allow(dead_code)]
pub fn empty_series(name: impl Into<String>, dtype: impl Into<DataType>) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.empty_series(dtype))
}

#[allow(dead_code)]
pub fn unnamed_empty_series(dtype: impl Into<DataType>) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.empty_series(dtype))
}

#[allow(dead_code)]
pub fn dataframe_from_columns(names: Vec<String>, columns: Vec<Column>) -> cudf::Result<DataFrame> {
    let operation = "dataframe::new";

    if names.len() != columns.len() {
        return Err(testing::dataframe::new_column_length_mismatch_error_for_test(names.len(), columns.len()));
    }

    if names.is_empty() {
        return Ok(sample_runtime().empty_dataframe());
    }

    let pairs = names
        .into_iter()
        .zip(columns)
        .map(|(name, column)| {
            Series::from_column(column)
                .map(|series| (name, series))
                .map_err(|e| testing::error::reanchor_for_test(e, operation))
        })
        .collect::<cudf::Result<Vec<_>>>()?;

    DataFrame::from_columns(pairs).map_err(|e| testing::error::reanchor_for_test(e, operation))
}

#[allow(dead_code)]
pub fn dataframe_device_ptr(df: &DataFrame, name: &str) -> *const c_void {
    df.series(name).unwrap().as_column().device_data_ptr().unwrap()
}

#[allow(dead_code)]
pub fn series_device_ptr(series: &Series) -> *const c_void {
    series.as_column().device_data_ptr().unwrap()
}

#[allow(dead_code)]
pub fn reset_exec_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

#[allow(dead_code)]
pub fn fail_next_column_view_metadata_read() {
    unsafe { cudf_sys::cudf_test_fail_next_column_view_metadata_read() };
}

#[allow(dead_code)]
pub fn arm_next_io_metadata_fault(fault: i32) {
    unsafe {
        cudf_sys::cudf_test_io_reset_metadata_fault();
        cudf_sys::cudf_test_io_set_next_metadata_fault(fault);
    }
}

#[allow(dead_code)]
pub fn reset_io_metadata_fault() {
    unsafe { cudf_sys::cudf_test_io_reset_metadata_fault() };
}

#[allow(dead_code)]
pub fn reset_sync_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

#[allow(dead_code)]
pub fn reset_dependency_observation() {
    unsafe { cudf_test_event_reset_observation() };
}

#[allow(dead_code)]
pub fn reset_row_predicate_observation() {
    unsafe { cudf_sys::cudf_test_row_predicate_reset_observation() };
}

#[allow(dead_code)]
pub fn arm_next_groupby_negative_total_columns_fault() {
    unsafe { cudf_sys::cudf_test_groupby_fail_next_negative_total_columns() };
}

#[allow(dead_code)]
pub fn arm_next_table_view_negative_num_columns_fault() {
    unsafe { cudf_sys::cudf_test_table_view_fail_next_negative_num_columns() };
}

#[allow(dead_code)]
pub fn arm_next_table_view_negative_num_rows_fault() {
    unsafe { cudf_sys::cudf_test_table_view_fail_next_negative_num_rows() };
}

#[allow(dead_code)]
pub fn lock_event_observation() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
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
pub fn last_exec_device() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_exec_last_device(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last exec device observation");
    out
}

#[allow(dead_code)]
pub fn set_current_device(device: DeviceId) {
    let status = unsafe { cudf_cuda_set_device(device.ordinal() as i32) };
    assert_eq!(status, CUDF_SUCCESS, "failed to set current device to {}", device.ordinal());
}

/// RAII guard that saves the current CUDA device on construction and restores
/// it on drop. Use at the top of any test that calls `set_current_device` to
/// prevent device-state leakage between tests in the same binary.
#[allow(dead_code)]
pub struct DeviceRestoreGuard {
    previous: i32,
}

#[allow(dead_code)]
impl DeviceRestoreGuard {
    pub fn new() -> Self {
        let mut device: i32 = 0;
        let status = unsafe { cudf_cuda_get_device(&mut device) };
        assert_eq!(status, CUDF_SUCCESS, "failed to query current CUDA device");
        Self { previous: device }
    }
}

impl Drop for DeviceRestoreGuard {
    fn drop(&mut self) {
        unsafe { cudf_cuda_set_device(self.previous) };
    }
}

#[allow(dead_code)]
pub fn last_sync_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last synchronized stream observation");
    out
}

#[allow(dead_code)]
pub fn dependency_wait_call_count() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_event_stream_wait_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read dependency wait call count");
    out
}

#[allow(dead_code)]
pub fn dependency_wait_call_count_for_stream(stream_raw: usize) -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read dependency wait call count for stream");
    out
}

#[allow(dead_code)]
pub fn last_dependency_wait_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_event_last_wait_stream_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last dependency wait stream observation");
    out
}

#[allow(dead_code)]
pub fn last_row_predicate_filter_calls() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_sys::cudf_test_row_predicate_filter_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read row predicate filter call count");
    out
}

#[allow(dead_code)]
pub fn last_row_predicate_compute_mask_calls() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_sys::cudf_test_row_predicate_compute_mask_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read row predicate compute-mask call count");
    out
}

#[allow(dead_code)]
pub fn last_row_predicate_final_filter_calls() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_sys::cudf_test_row_predicate_final_filter_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read row predicate final-filter call count");
    out
}

#[allow(dead_code)]
pub fn last_row_predicate_mask_api_calls() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_sys::cudf_test_apply_boolean_mask_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read apply_boolean_mask call count");
    out
}

#[allow(dead_code)]
pub fn explicit_i32_column(data: &[i32], identity: u64) -> Column {
    column_from_slice(data).unwrap().testing_rebind_execution_domain_explicit(identity).unwrap()
}

#[allow(dead_code)]
pub fn explicit_i32_series(name: &str, data: &[i32], identity: u64) -> Series {
    let _ = name;
    Series::from_column(explicit_i32_column(data, identity)).unwrap()
}

#[allow(dead_code)]
pub fn explicit_bool_column(data: &[bool], identity: u64) -> Column {
    column_from_slice(data).unwrap().testing_rebind_execution_domain_explicit(identity).unwrap()
}

#[allow(dead_code)]
pub fn explicit_bool_series(name: &str, data: &[bool], identity: u64) -> Series {
    let _ = name;
    Series::from_column(explicit_bool_column(data, identity)).unwrap()
}

#[allow(dead_code)]
pub fn sample_runtime() -> Runtime {
    let stream = CudaStream::new().unwrap();
    Runtime::builder().with_stream(stream).with_pool_memory_resource(16 * 1024 * 1024, None).build().unwrap()
}

#[allow(dead_code)]
pub fn error_context_value(err: &cudf::Error, key: &str) -> Option<String> {
    // Canonical facts take precedence over legacy diagnostic details. This
    // mirrors the §5 rule that a canonical key is machine-read from `facts()`
    // only — producers no longer duplicate it into `details()`.
    if let Some(fact_name) = FactKeyName::from_snake_case(key) {
        if let Some(value) = err.fact_value(fact_name) {
            return Some(fact_value_to_string(value));
        }
    }
    err.details().iter().find_map(|(ctx_key, ctx_value)| (*ctx_key == key).then(|| ctx_value.to_string()))
}

/// Stringify a `FactValue` using the same canonical representation that
/// production errors expose through `Display`.
#[allow(dead_code)]
fn fact_value_to_string(value: &FactValue) -> String {
    match value {
        FactValue::Str(s) => s.to_string(),
        FactValue::Usize(n) => n.to_string(),
        FactValue::I64(n) => n.to_string(),
        FactValue::TypeId(t) => t.to_string(),
        FactValue::DeviceId(d) => d.to_string(),
        // `FactValue` is non_exhaustive; new variants default to their Display form.
        other => other.to_string(),
    }
}

#[allow(dead_code)]
pub fn interop_error_context_value<'a>(err: &'a InteropError, key: &str) -> Option<&'a str> {
    err.context().iter().find_map(|(ctx_key, ctx_value)| (*ctx_key == key).then_some(ctx_value.as_str()))
}

#[allow(dead_code)]
pub fn assert_missing_column_error(err: &cudf::Error, expected_column: &str) {
    assert_eq!(err.kind(), cudf::ErrorKind::NotFound);
    assert_eq!(err.code(), ErrorCode::MissingColumn);
    assert_eq!(err.fact_value(FactKeyName::ColumnName), Some(&FactValue::Str(expected_column.to_string().into())));
}

#[allow(dead_code)]
pub fn assert_duplicate_name_error(err: &cudf::Error, expected_name: &str) {
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::DuplicateColumnName);
    assert_eq!(error_context_value(err, "duplicate_name").as_deref(), Some(expected_name));
    assert_eq!(error_context_value(err, "resource").as_deref(), Some("column_name"));
    assert!(error_context_value(err, "reason").is_some(), "duplicate-name contract should expose a reason");
    assert!(error_context_value(err, "remedy").is_some(), "duplicate-name contract should expose a remedy");
}
