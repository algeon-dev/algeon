#![allow(dead_code, reason = "shared test helpers used across independent integration test targets")]

use std::{
    ffi::c_void,
    fs::File,
    ptr,
    sync::{Arc, Mutex, MutexGuard, OnceLock},
};

use arrow::{
    array::{ArrayRef, Int64Array, Int64Builder, ListBuilder, StringArray, StructArray},
    datatypes::Field,
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, DataType, DeviceId, ErrorCode, Runtime, Series, TimestampResolution, TypeId,
    core::{
        Column, CudfType, DeviceBuffer, Scalar,
        error::{FactKeyName, FactValue},
    },
    frame::HostColumnData,
    runtime::CudaStream,
    testing::{self, prelude::*},
};
use cudf_sys::{
    CUDF_SUCCESS, cudf_cuda_get_device, cudf_cuda_set_device,
    test_support::{
        cudf_test_apply_boolean_mask_call_count, cudf_test_event_last_wait_stream_raw,
        cudf_test_event_reset_observation, cudf_test_event_stream_wait_call_count,
        cudf_test_event_stream_wait_call_count_for_stream_raw, cudf_test_exec_last_device,
        cudf_test_exec_last_memory_resource_destroy_raw, cudf_test_exec_last_memory_resource_destroy_sequence,
        cudf_test_exec_last_mr_raw, cudf_test_exec_last_stream_value, cudf_test_exec_last_sync_sequence,
        cudf_test_exec_last_sync_stream_value, cudf_test_exec_observed_mr_count, cudf_test_exec_observed_mr_overflowed,
        cudf_test_exec_observed_mr_raw_at, cudf_test_exec_reset_observation,
        cudf_test_fail_next_column_view_metadata_read, cudf_test_groupby_fail_next_negative_total_columns,
        cudf_test_io_metadata_fault_t, cudf_test_io_reset_metadata_fault, cudf_test_io_set_next_metadata_fault,
        cudf_test_row_predicate_compute_mask_call_count, cudf_test_row_predicate_filter_call_count,
        cudf_test_row_predicate_final_filter_call_count, cudf_test_row_predicate_reset_observation,
        cudf_test_table_view_fail_next_negative_num_columns, cudf_test_table_view_fail_next_negative_num_rows,
    },
};
use parquet::arrow::ArrowWriter;
use rapids_interop::{ColumnLayout, InteropError, Placement};
use tempfile::TempDir;

pub fn sample_df() -> DataFrame {
    let ids = column_from_slice(&[3i32, 1, 4, 1, 5]).unwrap();
    let scores = column_from_slice(&[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap();
    dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

thread_local! {
    static TEST_RUNTIME: Runtime = sample_runtime();
}

pub fn with_test_runtime<T>(f: impl FnOnce(&Runtime) -> T) -> T {
    TEST_RUNTIME.with(f)
}

pub fn column_from_slice<T: CudfType>(data: &[T]) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_slice(data))
}

pub fn column_from_slice_nullable<T: CudfType>(data: &[T], validity: &[bool]) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_slice_nullable(data, validity))
}

pub fn column_from_strs(data: &[&str]) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_strs(data))
}

pub fn column_from_strs_nullable(data: &[&str], validity: &[bool]) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_strs_nullable(data, validity))
}

pub fn null_column(dtype: impl Into<DataType>, len: usize) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.null_column(dtype, len))
}

pub fn empty_column(dtype: impl Into<DataType>) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.empty_column(dtype))
}

pub unsafe fn column_from_device_ptr(device_ptr: *const c_void, len: usize, type_id: TypeId) -> cudf::Result<Column> {
    with_test_runtime(|rt| unsafe { rt.column_from_device_ptr(device_ptr, len, type_id) })
}

pub fn column_from_device_buffer(buffer: DeviceBuffer, row_count: usize, type_id: TypeId) -> cudf::Result<Column> {
    with_test_runtime(|rt| rt.column_from_device_buffer(buffer, row_count, type_id))
}

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

pub fn scalar<T: CudfType>(value: T) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.scalar(value))
}

pub fn string_scalar(value: &str) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.string_scalar(value))
}

pub fn null_scalar<T: CudfType>() -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.null_scalar::<T>())
}

pub fn null_string_scalar() -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.null_string_scalar())
}

pub fn decimal32_raw_scalar(raw: i32, scale: i32) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.decimal32_raw_scalar(raw, scale))
}

pub fn decimal64_raw_scalar(raw: i64, scale: i32) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.decimal64_raw_scalar(raw, scale))
}

pub fn decimal128_raw_scalar(raw: i128, scale: i32) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.decimal128_raw_scalar(raw, scale))
}

pub fn timestamp_raw_scalar(raw: i64, resolution: TimestampResolution) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.timestamp_raw_scalar(raw, resolution))
}

pub fn duration_raw_scalar(raw: i64, resolution: TimestampResolution) -> cudf::Result<Scalar> {
    with_test_runtime(|rt| rt.duration_raw_scalar(raw, resolution))
}

pub fn series<T: HostColumnData>(name: impl Into<String>, data: &[T]) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.series(data))
}

pub fn unnamed_series<T: HostColumnData>(data: &[T]) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.series(data))
}

pub fn nullable_series<T: HostColumnData>(
    name: impl Into<String>,
    data: &[T],
    validity: &[bool],
) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.nullable_series(data, validity))
}

pub fn unnamed_nullable_series<T: HostColumnData>(data: &[T], validity: &[bool]) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.nullable_series(data, validity))
}

pub fn host_export_fixed_width_bytes(series: &Series) -> Vec<u8> {
    let export = series.try_prepare_host_column_export().unwrap();
    let values_id = match export.column().layout() {
        ColumnLayout::FixedWidth { values } => *values,
        other => panic!("expected fixed-width host export, got {other:?}"),
    };
    let binding =
        export.bindings().iter().find(|binding| binding.id() == values_id).expect("missing fixed-width values binding");

    assert_eq!(*binding.placement(), Placement::Host);

    // SAFETY: prepared host exports retain their backing host buffers, and the
    // binding lifetime is tied to the export value held in this function.
    unsafe { std::slice::from_raw_parts(binding.bytes().base_ptr() as *const u8, binding.bytes().byte_len()).to_vec() }
}

pub fn host_export_i32_values(series: &Series) -> Vec<i32> {
    host_export_fixed_width_bytes(series)
        .as_chunks::<{ std::mem::size_of::<i32>() }>()
        .0
        .iter()
        .map(|chunk| i32::from_ne_bytes(*chunk))
        .collect()
}

pub fn host_export_i64_values(series: &Series) -> Vec<i64> {
    host_export_fixed_width_bytes(series)
        .as_chunks::<{ std::mem::size_of::<i64>() }>()
        .0
        .iter()
        .map(|chunk| i64::from_ne_bytes(*chunk))
        .collect()
}

pub fn series_opt<T: CudfType>(name: impl Into<String>, data: &[Option<T>]) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.series_opt(data))
}

pub fn unnamed_series_opt<T: CudfType>(data: &[Option<T>]) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.series_opt(data))
}

pub fn null_series(name: impl Into<String>, dtype: impl Into<DataType>, len: usize) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.nulls(dtype, len))
}

pub fn unnamed_null_series(dtype: impl Into<DataType>, len: usize) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.nulls(dtype, len))
}

pub fn empty_series(name: impl Into<String>, dtype: impl Into<DataType>) -> cudf::Result<Series> {
    let _ = name.into();
    with_test_runtime(|rt| rt.empty_series(dtype))
}

pub fn unnamed_empty_series(dtype: impl Into<DataType>) -> cudf::Result<Series> {
    with_test_runtime(|rt| rt.empty_series(dtype))
}

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

    DataFrame::from_series(pairs).map_err(|e| testing::error::reanchor_for_test(e, operation))
}

pub fn dataframe_device_ptr(df: &DataFrame, name: &str) -> *const c_void {
    df.series(name).unwrap().as_column().device_data_ptr().unwrap()
}

pub fn series_device_ptr(series: &Series) -> *const c_void {
    series.as_column().device_data_ptr().unwrap()
}

pub fn reset_exec_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

pub fn fail_next_column_view_metadata_read() {
    unsafe { cudf_test_fail_next_column_view_metadata_read() };
}

pub fn arm_next_io_metadata_fault(fault: cudf_test_io_metadata_fault_t) {
    unsafe {
        cudf_test_io_reset_metadata_fault();
        cudf_test_io_set_next_metadata_fault(fault);
    }
}

pub fn reset_io_metadata_fault() {
    unsafe { cudf_test_io_reset_metadata_fault() };
}

pub fn reset_sync_observation() {
    unsafe { cudf_test_exec_reset_observation() };
}

pub fn reset_dependency_observation() {
    unsafe { cudf_test_event_reset_observation() };
}

pub fn reset_row_predicate_observation() {
    unsafe { cudf_test_row_predicate_reset_observation() };
}

pub fn arm_next_groupby_negative_total_columns_fault() {
    unsafe { cudf_test_groupby_fail_next_negative_total_columns() };
}

pub fn arm_next_table_view_negative_num_columns_fault() {
    unsafe { cudf_test_table_view_fail_next_negative_num_columns() };
}

pub fn arm_next_table_view_negative_num_rows_fault() {
    unsafe { cudf_test_table_view_fail_next_negative_num_rows() };
}

pub fn lock_event_observation() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

pub fn last_exec_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last exec stream observation");
    out
}

pub fn last_exec_mr_value() -> usize {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last exec mr observation");
    out as usize
}

pub fn observed_exec_mr_values() -> Vec<usize> {
    let mut count: usize = 0;
    let status = unsafe { cudf_test_exec_observed_mr_count(&mut count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read exec mr observation count");

    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let mut out: *mut c_void = ptr::null_mut();
        let status = unsafe { cudf_test_exec_observed_mr_raw_at(index, &mut out) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read exec mr observation at index {index}");
        values.push(out as usize);
    }
    values
}

pub fn observed_exec_mr_overflowed() -> bool {
    let mut out: i32 = 0;
    let status = unsafe { cudf_test_exec_observed_mr_overflowed(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read exec mr observation overflow status");
    out != 0
}

pub fn last_exec_device() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_exec_last_device(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last exec device observation");
    out
}

pub fn set_current_device(device: DeviceId) {
    let status = unsafe { cudf_cuda_set_device(device.ordinal() as i32) };
    assert_eq!(status, CUDF_SUCCESS, "failed to set current device to {}", device.ordinal());
}

/// RAII guard that saves the current CUDA device on construction and restores
/// it on drop. Use at the top of any test that calls `set_current_device` to
/// prevent device-state leakage between tests in the same binary.
pub struct DeviceRestoreGuard {
    previous: i32,
}

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

pub fn last_sync_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_sync_stream_value(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last synchronized stream observation");
    out
}

pub fn last_sync_sequence() -> u64 {
    let mut out = 0;
    let status = unsafe { cudf_test_exec_last_sync_sequence(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last synchronization sequence");
    out
}

pub fn last_destroyed_memory_resource_value() -> usize {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_test_exec_last_memory_resource_destroy_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last destroyed memory resource observation");
    out as usize
}

pub fn last_memory_resource_destroy_sequence() -> u64 {
    let mut out = 0;
    let status = unsafe { cudf_test_exec_last_memory_resource_destroy_sequence(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read memory resource destruction sequence");
    out
}

pub fn dependency_wait_call_count() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_event_stream_wait_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read dependency wait call count");
    out
}

pub fn dependency_wait_call_count_for_stream(stream_raw: usize) -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read dependency wait call count for stream");
    out
}

pub fn last_dependency_wait_stream_value() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_event_last_wait_stream_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last dependency wait stream observation");
    out
}

pub fn last_row_predicate_filter_calls() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_row_predicate_filter_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read row predicate filter call count");
    out
}

pub fn last_row_predicate_compute_mask_calls() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_row_predicate_compute_mask_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read row predicate compute-mask call count");
    out
}

pub fn last_row_predicate_final_filter_calls() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_row_predicate_final_filter_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read row predicate final-filter call count");
    out
}

pub fn last_row_predicate_mask_api_calls() -> i32 {
    let mut out: i32 = -1;
    let status = unsafe { cudf_test_apply_boolean_mask_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read apply_boolean_mask call count");
    out
}

pub fn explicit_i32_column(data: &[i32], identity: u64) -> Column {
    column_from_slice(data).unwrap().testing_rebind_execution_domain_explicit(identity).unwrap()
}

pub fn explicit_i32_series(name: &str, data: &[i32], identity: u64) -> Series {
    let _ = name;
    Series::from_column(explicit_i32_column(data, identity)).unwrap()
}

pub fn explicit_bool_column(data: &[bool], identity: u64) -> Column {
    column_from_slice(data).unwrap().testing_rebind_execution_domain_explicit(identity).unwrap()
}

pub fn explicit_bool_series(name: &str, data: &[bool], identity: u64) -> Series {
    let _ = name;
    Series::from_column(explicit_bool_column(data, identity)).unwrap()
}

pub fn sample_runtime() -> Runtime {
    let stream = CudaStream::new().unwrap();
    Runtime::builder().with_stream(stream).with_pool_memory_resource(16 * 1024 * 1024, None).build().unwrap()
}

pub fn error_context_value(err: &cudf::Error, key: &str) -> Option<String> {
    // Canonical facts take precedence over legacy diagnostic details because
    // producers do not duplicate canonical keys into `details()`.
    if let Some(fact_name) = FactKeyName::from_snake_case(key)
        && let Some(value) = err.fact_value(fact_name)
    {
        return Some(fact_value_to_string(value));
    }
    err.details().iter().find_map(|(ctx_key, ctx_value)| (*ctx_key == key).then(|| ctx_value.to_string()))
}

/// Stringify a `FactValue` using the same canonical representation that
/// production errors expose through `Display`.
fn fact_value_to_string(value: &FactValue) -> String {
    match value {
        FactValue::Str(s) => s.to_string(),
        FactValue::Usize(n) => n.to_string(),
        FactValue::I64(n) => n.to_string(),
        FactValue::TypeId(t) => t.to_string(),
        FactValue::DeviceId(d) => d.to_string(),
        FactValue::U64(n) => n.to_string(),
        // `FactValue` is non_exhaustive; new variants default to their Display form.
        other => other.to_string(),
    }
}

pub fn interop_error_context_value<'a>(err: &'a InteropError, key: &str) -> Option<&'a str> {
    err.context().iter().find_map(|(ctx_key, ctx_value)| (*ctx_key == key).then_some(ctx_value.as_str()))
}

pub fn assert_missing_column_error(err: &cudf::Error, expected_column: &str) {
    assert_eq!(err.kind(), cudf::ErrorKind::NotFound);
    assert_eq!(err.code(), ErrorCode::MissingColumn);
    assert_eq!(err.fact_value(FactKeyName::ColumnName), Some(&FactValue::Str(expected_column.to_string().into())));
}

pub fn assert_duplicate_name_error(err: &cudf::Error, expected_name: &str) {
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::DuplicateColumnName);
    assert_eq!(error_context_value(err, "duplicate_name").as_deref(), Some(expected_name));
    assert_eq!(error_context_value(err, "resource").as_deref(), Some("column_name"));
    assert!(error_context_value(err, "reason").is_some(), "duplicate-name contract should expose a reason");
    assert!(error_context_value(err, "remedy").is_some(), "duplicate-name contract should expose a remedy");
}

/// Runs named cases in order on the current thread, then fails once listing
/// every case that panicked. Cases share the thread's test runtime, so a table
/// pays for one CUDA context instead of one per test process.
pub fn run_named_cases(cases: &[(&str, fn())]) {
    let failed = cases
        .iter()
        .filter(|(name, case)| {
            let panicked = std::panic::catch_unwind(*case).is_err();
            if panicked {
                eprintln!("case `{name}` failed; its panic message is above");
            }
            panicked
        })
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    assert!(failed.is_empty(), "{} of {} cases failed: {}", failed.len(), cases.len(), failed.join(", "));
}

pub fn int64_array(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

pub fn utf8_array(values: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

pub fn int64_list_array(rows: &[Vec<i64>]) -> ArrayRef {
    let mut builder = ListBuilder::new(Int64Builder::new());
    for row in rows {
        builder.values().append_slice(row);
        builder.append(true);
    }
    Arc::new(builder.finish())
}

pub fn nested_int64_list_array(rows: &[Vec<Vec<i64>>]) -> ArrayRef {
    let mut builder = ListBuilder::new(ListBuilder::new(Int64Builder::new()));
    for row in rows {
        for inner in row {
            builder.values().values().append_slice(inner);
            builder.values().append(true);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

pub fn struct_array(children: Vec<(&str, ArrayRef)>) -> ArrayRef {
    let fields = children
        .into_iter()
        .map(|(name, array)| (Arc::new(Field::new(name, array.data_type().clone(), false)), array))
        .collect::<Vec<_>>();
    Arc::new(StructArray::from(fields))
}

/// Read Arrow columns into a frame through Parquet. The host Arrow import has
/// no Struct or List-of-List support, so Parquet is the route to nested cudf
/// columns.
pub fn nested_dataframe(runtime: &Runtime, columns: Vec<(&str, ArrayRef)>) -> DataFrame {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("nested.parquet");
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    DataFrame::read_parquet(runtime, &path).unwrap()
}
