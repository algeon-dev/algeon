use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cudf::{
    DataFrame, DataType, DeviceId, ErrorKind, Runtime, Series, TypeId,
    core::{Column, Table},
    interop::{
        arrow_device::{DeviceArrowDeviceType, DeviceArrowExportMaterialization, DeviceArrowImportMaterialization},
        borrowed::{BufferBinding, ColumnLayout, PhysicalValidity, SchemaNullability, SemanticColumn},
        prepared::{ExportProfile, PreparedColumnExport},
    },
    ops::dictionary,
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::{
    last_exec_device, last_exec_mr_value, last_exec_stream_value, lock_event_observation, reset_exec_observation,
};

fn should_run_multi_gpu_test() -> bool {
    let count = DeviceId::count().expect("failed to query visible CUDA devices");
    if count < 2 {
        eprintln!("skipping multi-GPU test because only {count} CUDA device is visible");
        return false;
    }
    true
}

fn assert_export_exec_observation_matches_runtime(runtime: &Runtime) {
    assert_eq!(last_exec_device(), runtime.device().ordinal() as i32);
    assert_eq!(last_exec_stream_value(), runtime.stream().testing_raw_handle_value());
    assert_eq!(last_exec_mr_value(), runtime.memory_resource().testing_raw_handle_value());
}

fn reset_event_observation() {
    unsafe { cudf_sys::cudf_test_event_reset_observation() };
}

fn observed_event_synchronize_call_count() -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_synchronize_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read event synchronize observation count");
    out
}

fn observed_stream_wait_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read stream-wait observation count");
    out
}

fn observed_raw_record_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_raw_record_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read raw event-record observation count");
    out
}

fn observed_last_raw_record_stream_raw() -> usize {
    let mut out = 0;
    let status = unsafe { cudf_sys::cudf_test_event_last_raw_record_stream_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last raw event-record stream");
    out
}

fn observed_last_raw_record_event_raw() -> usize {
    let mut out = 0;
    let status = unsafe { cudf_sys::cudf_test_event_last_raw_record_event_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read last raw event-record event");
    out
}

fn raw_cuda_event_from_handle(handle: *mut c_void) -> usize {
    let mut out = std::ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_event_raw(handle, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read raw CUDA event from borrowed handle");
    out as usize
}

fn assert_fixed_width_pointer_rejected<T: std::fmt::Debug>(result: std::result::Result<T, cudf::Error>, label: &str) {
    let err = result.expect_err(label);
    assert_eq!(err.kind(), ErrorKind::InvalidArgument, "{label}: unexpected error kind");
    assert!(err.message().contains("fixed-width"), "{label}: unexpected error message: {}", err.message());
}

fn read_json_dataframe(filename: &str, rows: &str) -> DataFrame {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(filename);
    std::fs::write(&path, rows).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    DataFrame::read_json(&runtime, &path).unwrap()
}

fn nested_dictionary_dataframe(filename: &str, rows: &str) -> DataFrame {
    let frame = read_json_dataframe(filename, rows);
    let name = frame.series("name").unwrap();
    let lists = frame.series("lists").unwrap();
    let point = frame.series("point").unwrap();
    let category_source = frame.series("category").unwrap().as_column().deep_copy().unwrap();
    let category =
        Series::from_column(dictionary::encode(&category_source.testing_view().unwrap(), TypeId::Int32).unwrap())
            .unwrap()
            .rename("category");
    DataFrame::from_columns(::std::iter::IntoIterator::into_iter([name, lists, point, category]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap()
}

fn binding_bytes(binding: &BufferBinding<'_>) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(binding.bytes().base_ptr() as *const u8, binding.bytes().byte_len()) }.to_vec()
}

fn binding_map<'a>(bindings: &'a [BufferBinding<'a>]) -> HashMap<u32, &'a BufferBinding<'a>> {
    bindings.iter().map(|binding| (binding.id().get(), binding)).collect()
}

fn semantic_column_map(columns: &[SemanticColumn]) -> HashMap<u32, &SemanticColumn> {
    columns.iter().map(|column| (column.id().get(), column)).collect()
}

fn validity_mask_is_elidable(validity: &PhysicalValidity, _bindings: &HashMap<u32, &BufferBinding<'_>>) -> bool {
    matches!(validity, PhysicalValidity::ValidityMask { null_count: Some(0), .. })
}

fn assert_binding_equivalent(source: &BufferBinding<'_>, imported: &BufferBinding<'_>) {
    assert_eq!(source.placement(), imported.placement());
    assert_eq!(binding_bytes(source), binding_bytes(imported));
}

fn assert_validity_equivalent(
    source: &PhysicalValidity,
    imported: &PhysicalValidity,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    imported_bindings: &HashMap<u32, &BufferBinding<'_>>,
) {
    match (source, imported) {
        (PhysicalValidity::NoValidityBuffer, PhysicalValidity::NoValidityBuffer) => {},
        (
            PhysicalValidity::ValidityMask { mask: source_mask, null_count: source_null_count },
            PhysicalValidity::ValidityMask { mask: imported_mask, null_count: imported_null_count },
        ) => {
            assert_eq!(source_null_count, imported_null_count);
            assert_binding_equivalent(
                source_bindings.get(&source_mask.get()).expect("missing source validity binding"),
                imported_bindings.get(&imported_mask.get()).expect("missing imported validity binding"),
            );
        },
        (PhysicalValidity::ValidityMask { .. }, PhysicalValidity::NoValidityBuffer) => {
            assert!(validity_mask_is_elidable(source, source_bindings));
        },
        (PhysicalValidity::NoValidityBuffer, PhysicalValidity::ValidityMask { .. }) => {
            assert!(validity_mask_is_elidable(imported, imported_bindings));
        },
    }
}

fn schema_nullability_is_elidable(
    source: &SemanticColumn,
    imported: &SemanticColumn,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    imported_bindings: &HashMap<u32, &BufferBinding<'_>>,
) -> bool {
    match (source.schema_nullability(), imported.schema_nullability()) {
        (SchemaNullability::Nullable, SchemaNullability::NonNullable) => {
            validity_mask_is_elidable(source.validity(), source_bindings)
                && matches!(imported.validity(), PhysicalValidity::NoValidityBuffer)
        },
        (SchemaNullability::NonNullable, SchemaNullability::Nullable) => {
            matches!(source.validity(), PhysicalValidity::NoValidityBuffer)
                && validity_mask_is_elidable(imported.validity(), imported_bindings)
        },
        _ => false,
    }
}

fn assert_semantic_column_equivalent(
    source: &SemanticColumn,
    imported: &SemanticColumn,
    source_columns: &HashMap<u32, &SemanticColumn>,
    imported_columns: &HashMap<u32, &SemanticColumn>,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    imported_bindings: &HashMap<u32, &BufferBinding<'_>>,
) {
    assert_eq!(source.data_type(), imported.data_type());
    assert_eq!(source.len(), imported.len());
    assert_eq!(source.offset(), imported.offset());
    if source.schema_nullability() != imported.schema_nullability() {
        assert!(
            schema_nullability_is_elidable(source, imported, source_bindings, imported_bindings),
            "schema nullability mismatch: source={source:?} imported={imported:?}"
        );
    }
    assert_validity_equivalent(source.validity(), imported.validity(), source_bindings, imported_bindings);

    match (source.layout(), imported.layout()) {
        (ColumnLayout::FixedWidth { values: source_values }, ColumnLayout::FixedWidth { values: imported_values }) => {
            assert_binding_equivalent(
                source_bindings.get(&source_values.get()).expect("missing source fixed-width binding"),
                imported_bindings.get(&imported_values.get()).expect("missing imported fixed-width binding"),
            );
        },
        (
            ColumnLayout::BinaryLike { offsets: source_offsets, data: source_data },
            ColumnLayout::BinaryLike { offsets: imported_offsets, data: imported_data },
        ) => {
            assert_binding_equivalent(
                source_bindings.get(&source_offsets.get()).expect("missing source binary offsets binding"),
                imported_bindings.get(&imported_offsets.get()).expect("missing imported binary offsets binding"),
            );
            assert_binding_equivalent(
                source_bindings.get(&source_data.get()).expect("missing source binary data binding"),
                imported_bindings.get(&imported_data.get()).expect("missing imported binary data binding"),
            );
        },
        (
            ColumnLayout::List { offsets: source_offsets, child: source_child },
            ColumnLayout::List { offsets: imported_offsets, child: imported_child },
        ) => {
            assert_binding_equivalent(
                source_bindings.get(&source_offsets.get()).expect("missing source list offsets binding"),
                imported_bindings.get(&imported_offsets.get()).expect("missing imported list offsets binding"),
            );
            assert_semantic_column_equivalent(
                source_columns.get(&source_child.get()).expect("missing source list child"),
                imported_columns.get(&imported_child.get()).expect("missing imported list child"),
                source_columns,
                imported_columns,
                source_bindings,
                imported_bindings,
            );
        },
        (ColumnLayout::Struct { children: source_children }, ColumnLayout::Struct { children: imported_children }) => {
            assert_eq!(source_children.len(), imported_children.len());
            for (source_child, imported_child) in source_children.iter().zip(imported_children.iter()) {
                assert_semantic_column_equivalent(
                    source_columns.get(&source_child.get()).expect("missing source struct child"),
                    imported_columns.get(&imported_child.get()).expect("missing imported struct child"),
                    source_columns,
                    imported_columns,
                    source_bindings,
                    imported_bindings,
                );
            }
        },
        (
            ColumnLayout::Dictionary { indices: source_indices, keys: source_keys },
            ColumnLayout::Dictionary { indices: imported_indices, keys: imported_keys },
        ) => {
            assert_binding_equivalent(
                source_bindings.get(&source_indices.get()).expect("missing source dictionary indices binding"),
                imported_bindings.get(&imported_indices.get()).expect("missing imported dictionary indices binding"),
            );
            assert_semantic_column_equivalent(
                source_columns.get(&source_keys.get()).expect("missing source dictionary keys"),
                imported_columns.get(&imported_keys.get()).expect("missing imported dictionary keys"),
                source_columns,
                imported_columns,
                source_bindings,
                imported_bindings,
            );
        },
        _ => panic!("layout mismatch: source={:?} imported={:?}", source.layout(), imported.layout()),
    }
}

fn assert_host_column_exports_structurally_equivalent(
    source_export: &PreparedColumnExport<'_>,
    imported_export: &PreparedColumnExport<'_>,
) {
    assert_eq!(source_export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(imported_export.diagnostics().profile(), ExportProfile::PreparedHostExport);

    let source_columns = semantic_column_map(source_export.semantic_columns());
    let imported_columns = semantic_column_map(imported_export.semantic_columns());
    let source_bindings = binding_map(source_export.bindings());
    let imported_bindings = binding_map(imported_export.bindings());

    assert_semantic_column_equivalent(
        source_export.column(),
        imported_export.column(),
        &source_columns,
        &imported_columns,
        &source_bindings,
        &imported_bindings,
    );
}

fn assert_host_column_exports_equivalent(source: &Column, imported: &Column) {
    let source_export = source.try_prepare_host_column_export().unwrap();
    let imported_export = imported.try_prepare_host_column_export().unwrap();
    assert_host_column_exports_structurally_equivalent(&source_export, &imported_export);
}

fn assert_host_table_exports_equivalent(source: &Table, imported: &Table) {
    let source_export = source.try_prepare_host_table_export().unwrap();
    let imported_export = imported.try_prepare_host_table_export().unwrap();

    assert_eq!(source_export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(imported_export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(source_export.table().num_rows(), imported_export.table().num_rows());
    assert_eq!(source_export.table().columns().len(), imported_export.table().columns().len());

    let source_columns = semantic_column_map(source_export.semantic_columns());
    let imported_columns = semantic_column_map(imported_export.semantic_columns());
    let source_bindings = binding_map(source_export.bindings());
    let imported_bindings = binding_map(imported_export.bindings());

    for (source_root, imported_root) in
        source_export.table().columns().iter().zip(imported_export.table().columns().iter())
    {
        assert_semantic_column_equivalent(
            source_columns.get(&source_root.get()).expect("missing source table root"),
            imported_columns.get(&imported_root.get()).expect("missing imported table root"),
            &source_columns,
            &imported_columns,
            &source_bindings,
            &imported_bindings,
        );
    }
}

fn decimal64_series(runtime: &Runtime, name: &str, values: &[i64], scale: i32) -> Series {
    runtime.series(name, values).unwrap().astype(DataType::fixed_point(TypeId::Decimal64, scale)).unwrap()
}

#[repr(C)]
struct ArrowSchema {
    format: *const i8,
    name: *const i8,
    metadata: *const i8,
    flags: i64,
    n_children: i64,
    children: *mut *mut ArrowSchema,
    dictionary: *mut ArrowSchema,
    release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    private_data: *mut c_void,
}

#[repr(C)]
struct ArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut ArrowArray,
    dictionary: *mut ArrowArray,
    release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    private_data: *mut c_void,
}

#[repr(C)]
struct ArrowDeviceArray {
    array: ArrowArray,
    device_id: i64,
    device_type: i32,
    sync_event: *mut c_void,
    reserved: [i64; 3],
}

struct ArrowReleaseObserverState {
    schema_releases: Arc<AtomicUsize>,
    array_releases: Arc<AtomicUsize>,
    schema_release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    schema_private_data: *mut c_void,
    array_release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    array_private_data: *mut c_void,
}

unsafe extern "C" fn observe_arrow_schema_release(schema: *mut ArrowSchema) {
    if schema.is_null() {
        return;
    }

    let state = (*schema).private_data.cast::<ArrowReleaseObserverState>();
    assert!(!state.is_null(), "schema release observer state should be present");
    (*state).schema_releases.fetch_add(1, Ordering::SeqCst);
    (*schema).release = (*state).schema_release;
    (*schema).private_data = (*state).schema_private_data;
    if let Some(release) = (*state).schema_release {
        release(schema);
    }
}

unsafe extern "C" fn observe_arrow_array_release(array: *mut ArrowArray) {
    if array.is_null() {
        return;
    }

    let state = (*array).private_data.cast::<ArrowReleaseObserverState>();
    assert!(!state.is_null(), "array release observer state should be present");
    (*state).array_releases.fetch_add(1, Ordering::SeqCst);
    (*array).release = (*state).array_release;
    (*array).private_data = (*state).array_private_data;
    if let Some(release) = (*state).array_release {
        release(array);
    }
    drop(Box::from_raw(state));
}

unsafe fn install_release_observers(schema: *mut c_void, array: *mut c_void) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let schema = schema.cast::<ArrowSchema>();
    let array = array.cast::<ArrowDeviceArray>();
    let schema_releases = Arc::new(AtomicUsize::new(0));
    let array_releases = Arc::new(AtomicUsize::new(0));
    let state = Box::into_raw(Box::new(ArrowReleaseObserverState {
        schema_releases: Arc::clone(&schema_releases),
        array_releases: Arc::clone(&array_releases),
        schema_release: (*schema).release,
        schema_private_data: (*schema).private_data,
        array_release: (*array).array.release,
        array_private_data: (*array).array.private_data,
    }));

    (*schema).release = Some(observe_arrow_schema_release);
    (*schema).private_data = state.cast();
    (*array).array.release = Some(observe_arrow_array_release);
    (*array).array.private_data = state.cast();

    (schema_releases, array_releases)
}

#[test]
fn numeric_column_device_arrow_export_reports_producer_retained_view() {
    let _observation_lock = lock_event_observation();
    reset_exec_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3, 4]).unwrap();
    let producer_ptr = column.device_data_ptr().unwrap();

    let export = column.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
    assert_eq!(export.fixed_width_values_ptr().unwrap(), producer_ptr);
    assert_export_exec_observation_matches_runtime(&runtime);
}

#[test]
fn numeric_column_device_arrow_export_with_deep_copy_reports_bridge_owned_device_copy() {
    let _observation_lock = lock_event_observation();
    reset_exec_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3, 4]).unwrap();
    let source_ptr = column.device_data_ptr().unwrap();

    let export = column.try_prepare_device_arrow_export_with_deep_copy().unwrap();
    let export_ptr = export.fixed_width_values_ptr().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
    assert_ne!(export_ptr, source_ptr);
    assert_eq!(column.device_data_ptr().unwrap(), source_ptr);
    assert_eq!(column.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_export_exec_observation_matches_runtime(&runtime);

    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);

    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.device_data_ptr().unwrap(), export_ptr);
    assert_eq!(Series::from_column(imported).unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn numeric_column_device_arrow_raw_parts_round_trip_preserves_pointer() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[11i32, 22, 33]).unwrap();
    let producer_ptr = column.device_data_ptr().unwrap();

    let raw_parts = column.try_prepare_device_arrow_export().unwrap().into_raw_parts();

    assert_eq!(raw_parts.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);
    assert_eq!(raw_parts.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(raw_parts.device_id(), runtime.device().ordinal() as i32);
    assert!(raw_parts.sync_event_present());
    assert!(!raw_parts.schema_ptr().is_null());
    assert!(!raw_parts.device_array_ptr().is_null());
    assert_eq!(raw_parts.fixed_width_values_ptr().unwrap(), producer_ptr);

    let foreign = raw_parts.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);

    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.device_data_ptr().unwrap(), producer_ptr);
    assert_eq!(Series::from_column(imported).unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn bool_column_device_arrow_export_and_import_report_bridge_owned_device_copy() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[true, false, true]).unwrap();

    let export = column.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());

    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);

    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();
    assert_eq!(Series::from_column(imported).unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn bool_column_device_arrow_fixed_width_values_ptr_is_rejected() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[true, false, true]).unwrap();

    let export = column.try_prepare_device_arrow_export().unwrap();
    let err = export
        .fixed_width_values_ptr()
        .expect_err("BOOL8 device-arrow export must not expose fixed-width values pointer access");

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert!(err.message().contains("fixed-width"));
}

#[test]
fn numeric_series_device_arrow_export_reports_producer_retained_view() {
    let _observation_lock = lock_event_observation();
    reset_exec_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[1i32, 2, 3, 4]).unwrap();
    let producer_ptr = series.as_column().device_data_ptr().unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
    assert_eq!(export.fixed_width_values_ptr().unwrap(), producer_ptr);
    assert_export_exec_observation_matches_runtime(&runtime);
}

#[test]
fn numeric_series_device_arrow_export_records_sync_event_on_runtime_stream() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[1i32, 2, 3, 4]).unwrap();
    let runtime_stream_raw = runtime.stream().testing_raw_handle_value();

    reset_event_observation();
    let baseline_record = observed_raw_record_call_count_for_stream_raw(runtime_stream_raw);

    let export = series.try_prepare_device_arrow_export().unwrap();

    assert!(export.sync_event_present());
    assert!(
        observed_raw_record_call_count_for_stream_raw(runtime_stream_raw) > baseline_record,
        "device-arrow export should record a sync event on the runtime stream",
    );
    assert_eq!(observed_last_raw_record_stream_raw(), runtime_stream_raw);
    assert_eq!(observed_last_raw_record_event_raw(), raw_cuda_event_from_handle(export.sync_event_raw_ptr()));
}

#[test]
fn bool_series_device_arrow_export_reports_bridge_owned_device_copy() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("flag", &[true, false, true]).unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
}

#[test]
fn series_device_arrow_export_exposes_stable_sync_event_metadata_across_handoffs() {
    let _observation_lock = lock_event_observation();

    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[7i32, 8, 9]).unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();
    let export_sync_event = export.sync_event_raw_ptr();
    let raw_parts = export.into_raw_parts();
    let raw_sync_event = raw_parts.sync_event_raw_ptr();
    let foreign = raw_parts.into_foreign();

    assert!(!export_sync_event.is_null());
    assert_eq!(raw_sync_event, export_sync_event);
    assert_eq!(foreign.sync_event_raw_ptr(), export_sync_event);
}

#[test]
fn numeric_series_device_arrow_raw_parts_round_trip_preserves_pointer() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[11i32, 22, 33]).unwrap();
    let producer_ptr = series.as_column().device_data_ptr().unwrap();

    let raw_parts = series.try_prepare_device_arrow_export().unwrap().into_raw_parts();

    assert_eq!(raw_parts.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);
    assert_eq!(raw_parts.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(raw_parts.device_id(), runtime.device().ordinal() as i32);
    assert!(raw_parts.sync_event_present());
    assert!(!raw_parts.schema_ptr().is_null());
    assert!(!raw_parts.device_array_ptr().is_null());
    assert_eq!(raw_parts.fixed_width_values_ptr().unwrap(), producer_ptr);

    let foreign = raw_parts.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);

    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.device_data_ptr().unwrap(), producer_ptr);
    assert_eq!(Series::from_column(imported).unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn numeric_series_device_arrow_owned_abi_handoff_survives_export_drop() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let (foreign, producer_ptr) = {
        let series = runtime.series("values", &[41i32, 42, 43]).unwrap();
        let producer_ptr = series.as_column().device_data_ptr().unwrap();
        let abi = series.try_prepare_device_arrow_export().unwrap().into_owned_abi_parts();

        assert_eq!(abi.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);
        assert_eq!(abi.device_type(), DeviceArrowDeviceType::Cuda);
        assert_eq!(abi.device_id(), runtime.device().ordinal() as i32);
        assert!(abi.sync_event_present());
        assert_eq!(abi.fixed_width_values_ptr().unwrap(), producer_ptr);

        let (schema, array, materialization) = abi.into_raw_abi();
        let foreign = unsafe {
            cudf::interop::arrow_device::ForeignDeviceArrowColumn::from_raw_abi_parts(schema, array, materialization)
        }
        .expect("owned ABI raw parts should re-wrap as a foreign device-arrow column");

        (foreign, producer_ptr)
    };

    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);
    assert_eq!(imported.device_data_ptr().unwrap(), producer_ptr);
    assert_eq!(Series::from_column(imported).unwrap().to_vec_strict::<i32>().unwrap(), vec![41, 42, 43]);
}

#[test]
fn series_device_arrow_import_retains_owned_abi_release_until_imported_column_drop() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[41i32, 42, 43]).unwrap();
    let abi = series.try_prepare_device_arrow_export().unwrap().into_owned_abi_parts();
    let (schema, array, materialization) = abi.into_raw_abi();
    let (schema_releases, array_releases) = unsafe { install_release_observers(schema, array) };
    let foreign = unsafe {
        cudf::interop::arrow_device::ForeignDeviceArrowColumn::from_raw_abi_parts(schema, array, materialization)
    }
    .expect("owned ABI raw parts should re-wrap as a foreign device-arrow column");

    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    drop(foreign);
    assert_eq!(schema_releases.load(Ordering::SeqCst), 0);
    assert_eq!(array_releases.load(Ordering::SeqCst), 0);

    drop(imported);
    assert_eq!(schema_releases.load(Ordering::SeqCst), 1);
    assert_eq!(array_releases.load(Ordering::SeqCst), 1);
}

#[test]
fn foreign_device_arrow_column_from_raw_abi_parts_rejects_null_schema_and_array() {
    let err = match unsafe {
        cudf::interop::arrow_device::ForeignDeviceArrowColumn::from_raw_abi_parts(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            DeviceArrowImportMaterialization::ForeignRetainedView,
        )
    } {
        Ok(_) => panic!("null Arrow device ABI roots must be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert_eq!(err.operation(), "interop::arrow_device::ForeignDeviceArrowColumn::from_raw_abi_parts");
    assert!(err.message().contains("schema pointer"));
}

#[test]
fn numeric_device_arrow_import_reports_foreign_retained_view_and_preserves_pointer() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[10i32, 20, 30]).unwrap();
    let producer_ptr = series.as_column().device_data_ptr().unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(foreign.device_id(), runtime.device().ordinal() as i32);

    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.device_data_ptr().unwrap(), producer_ptr);
    let imported_series = Series::from_column(imported).unwrap();
    let values = imported_series.to_vec_strict::<i32>().unwrap();
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn series_device_arrow_import_preserves_sync_event_metadata_without_host_fencing_in_this_layer() {
    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().build().unwrap();
    let consumer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("values", &[10i32, 20, 30]).unwrap();
    let export = series.try_prepare_device_arrow_export().unwrap();
    let exported_sync_event = export.sync_event_raw_ptr();
    let consumer_stream_raw = consumer_runtime.stream().testing_raw_handle_value();

    reset_event_observation();
    let baseline_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);

    let foreign = export.into_foreign();
    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &consumer_runtime).unwrap();

    assert_eq!(imported.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(foreign.sync_event_raw_ptr(), exported_sync_event);
    assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_wait);
    assert_eq!(observed_event_synchronize_call_count(), 0);
}

#[test]
fn series_device_arrow_import_round_trip_returns_unnamed_series() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[10i32, 20, 30]).unwrap();
    let producer_ptr = series.as_column().device_data_ptr().unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();
    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.name(), None);
    assert_eq!(imported.as_column().device_data_ptr().unwrap(), producer_ptr);
    assert_eq!(imported.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn series_device_arrow_round_trip_does_not_call_host_arrow_path() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[10i32, 20, 30]).unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();
    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn bool_device_arrow_import_reports_bridge_owned_device_copy() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("flag", &[true, false, true, true]).unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);

    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();
    let imported_series = Series::from_column(imported).unwrap();
    let values = imported_series.to_vec_strict::<bool>().unwrap();
    assert_eq!(values, vec![true, false, true, true]);
}

#[test]
fn decimal64_series_device_arrow_export_reports_producer_retained_view() {
    let _observation_lock = lock_event_observation();
    reset_exec_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = decimal64_series(&runtime, "amount", &[12345i64, -67890, 5], -2);
    let producer_ptr = series.as_column().device_data_ptr().unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
    assert_eq!(export.fixed_width_values_ptr().unwrap(), producer_ptr);
    assert_export_exec_observation_matches_runtime(&runtime);
}

#[test]
fn decimal64_series_device_arrow_import_reports_foreign_retained_view_and_preserves_scale_and_values() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = decimal64_series(&runtime, "amount", &[12345i64, -67890, 5], -2);
    let producer_ptr = series.as_column().device_data_ptr().unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(foreign.device_id(), runtime.device().ordinal() as i32);

    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.as_column().device_data_ptr().unwrap(), producer_ptr);
    assert_eq!(imported.as_column().data_type().unwrap(), DataType::fixed_point(TypeId::Decimal64, -2));
    assert_eq!(imported.eq(&series).unwrap().to_vec_strict::<bool>().unwrap(), vec![true, true, true]);
}

#[test]
fn decimal64_series_device_arrow_round_trip_does_not_call_host_arrow_path() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let series = decimal64_series(&runtime, "amount", &[12345i64, -67890, 5], -2);

    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();
    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.as_column().data_type().unwrap(), DataType::fixed_point(TypeId::Decimal64, -2));
    assert_eq!(imported.eq(&series).unwrap().to_vec_strict::<bool>().unwrap(), vec![true, true, true]);
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn string_series_device_arrow_export_reports_producer_retained_view() {
    let _observation_lock = lock_event_observation();
    reset_exec_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series =
        runtime.nullable_series("words", &["alpha", "", "placeholder", "omega"], &[true, true, false, true]).unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
    assert_export_exec_observation_matches_runtime(&runtime);
}

#[test]
fn string_series_device_arrow_import_reports_foreign_retained_view_and_preserves_values() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series =
        runtime.nullable_series("words", &["alpha", "", "placeholder", "omega"], &[true, true, false, true]).unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(foreign.device_id(), runtime.device().ordinal() as i32);

    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(
        imported.to_optional_strings().unwrap(),
        vec![Some("alpha".to_string()), Some(String::new()), None, Some("omega".to_string()),]
    );
}

#[test]
fn list_series_device_arrow_round_trip_preserves_recursive_children_and_reports_retained_views() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let frame = read_json_dataframe(
        "arrow_device_list.jsonl",
        "{\"nested\":[[1,2],[3]]}\n{\"nested\":[[],[4,5]]}\n{\"nested\":[[6]]}\n",
    );
    let runtime = Runtime::builder().build().unwrap();
    let source = frame.series("nested").unwrap();

    let export = source.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);

    let foreign = export.into_foreign();
    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);

    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.as_column().data_type().unwrap().id(), TypeId::List);
    assert_host_column_exports_equivalent(source.as_column(), imported.as_column());
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn struct_series_device_arrow_round_trip_preserves_recursive_children_and_reports_retained_views() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let frame = read_json_dataframe(
        "arrow_device_struct.jsonl",
        "{\"point\":{\"coords\":[1,2],\"label\":\"one\"}}\n{\"point\":{\"coords\":[],\"label\":\"two\"}}\n{\"point\":{\"coords\":[3],\"label\":\"three\"}}\n",
    );
    let runtime = Runtime::builder().build().unwrap();
    let source = frame.series("point").unwrap();

    let export = source.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);

    let foreign = export.into_foreign();
    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);

    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.as_column().data_type().unwrap().id(), TypeId::Struct);
    assert_host_column_exports_equivalent(source.as_column(), imported.as_column());
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn dictionary_series_device_arrow_round_trip_preserves_keys_and_reports_retained_views() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let plain = runtime.column_from_strs(&["beta", "alpha", "beta", "gamma"]).unwrap();
    let source =
        Series::from_column(dictionary::encode(&plain.testing_view().unwrap(), TypeId::Int32).unwrap()).unwrap();

    let export = source.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);

    let foreign = export.into_foreign();
    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);

    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.as_column().data_type().unwrap().id(), TypeId::Dictionary32);
    assert_host_column_exports_equivalent(source.as_column(), imported.as_column());
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn list_of_bool_series_device_arrow_round_trip_reports_bridge_owned_device_copy() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let frame = read_json_dataframe(
        "arrow_device_list_bool.jsonl",
        "{\"flags\":[true,false,true]}\n{\"flags\":[]}\n{\"flags\":[false,true]}\n",
    );
    let runtime = Runtime::builder().build().unwrap();
    let source = frame.series("flags").unwrap();

    let export = source.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);

    let foreign = export.into_foreign();
    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy);

    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.as_column().data_type().unwrap().id(), TypeId::List);
    assert_host_column_exports_equivalent(source.as_column(), imported.as_column());
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn empty_string_series_device_arrow_export_uses_bridge_owned_offsets_but_import_is_foreign_retained() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.empty_series("words", TypeId::String).unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());

    let foreign = export.into_foreign();
    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);

    let imported = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime).unwrap();
    assert_eq!(imported.to_optional_strings().unwrap(), Vec::<Option<String>>::new());
}

#[test]
fn string_series_device_arrow_fixed_width_values_ptr_is_rejected() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("words", &["alpha", "beta"]).unwrap();

    let export = series.try_prepare_device_arrow_export().unwrap();
    assert_fixed_width_pointer_rejected(
        export.fixed_width_values_ptr(),
        "string device-arrow export must not expose fixed-width values pointer access",
    );
}

#[test]
fn nested_series_device_arrow_fixed_width_values_ptr_is_rejected() {
    let _observation_lock = lock_event_observation();

    let list_frame = read_json_dataframe(
        "arrow_device_list_fixed_width_ptr.jsonl",
        "{\"values\":[1,2,3]}\n{\"values\":[]}\n{\"values\":[4]}\n",
    );
    let list_export = list_frame.series("values").unwrap().try_prepare_device_arrow_export().unwrap();
    assert_fixed_width_pointer_rejected(
        list_export.fixed_width_values_ptr(),
        "list device-arrow export must not expose fixed-width values pointer access",
    );

    let struct_frame = read_json_dataframe(
        "arrow_device_struct_fixed_width_ptr.jsonl",
        "{\"point\":{\"x\":1,\"y\":2}}\n{\"point\":{\"x\":3,\"y\":4}}\n",
    );
    let struct_export = struct_frame.series("point").unwrap().try_prepare_device_arrow_export().unwrap();
    assert_fixed_width_pointer_rejected(
        struct_export.fixed_width_values_ptr(),
        "struct device-arrow export must not expose fixed-width values pointer access",
    );

    let dictionary_frame = read_json_dataframe(
        "arrow_device_dictionary_fixed_width_ptr.jsonl",
        "{\"category\":\"alpha\"}\n{\"category\":\"beta\"}\n{\"category\":\"alpha\"}\n",
    );
    let dictionary_source = dictionary_frame.series("category").unwrap().as_column().deep_copy().unwrap();
    let dictionary_series =
        Series::from_column(dictionary::encode(&dictionary_source.testing_view().unwrap(), TypeId::Int32).unwrap())
            .unwrap();
    let dictionary_export = dictionary_series.try_prepare_device_arrow_export().unwrap();
    assert_fixed_width_pointer_rejected(
        dictionary_export.fixed_width_values_ptr(),
        "dictionary device-arrow export must not expose fixed-width values pointer access",
    );
}

#[test]
fn numeric_table_device_arrow_export_reports_producer_retained_view() {
    let _observation_lock = lock_event_observation();
    reset_exec_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = runtime.column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let ids_ptr = ids.device_data_ptr().unwrap();
    let scores_ptr = scores.device_data_ptr().unwrap();
    let table = Table::from_columns(vec![ids, scores]).unwrap();

    let export = table.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
    assert_eq!(export.fixed_width_values_ptr(0).unwrap(), ids_ptr);
    assert_eq!(export.fixed_width_values_ptr(1).unwrap(), scores_ptr);
    assert_export_exec_observation_matches_runtime(&runtime);
}

#[test]
fn numeric_table_device_arrow_export_with_deep_copy_reports_bridge_owned_device_copy() {
    let _observation_lock = lock_event_observation();
    reset_exec_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = runtime.column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let ids_ptr = ids.device_data_ptr().unwrap();
    let scores_ptr = scores.device_data_ptr().unwrap();
    let table = Table::from_columns(vec![ids, scores]).unwrap();

    let export = table.try_prepare_device_arrow_export_with_deep_copy().unwrap();
    let export_ids_ptr = export.fixed_width_values_ptr(0).unwrap();
    let export_scores_ptr = export.fixed_width_values_ptr(1).unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
    assert_ne!(export_ids_ptr, ids_ptr);
    assert_ne!(export_scores_ptr, scores_ptr);
    assert_export_exec_observation_matches_runtime(&runtime);

    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);

    let imported = Table::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();
    let imported_columns = imported.into_columns().unwrap();

    assert_eq!(imported_columns[0].device_data_ptr().unwrap(), export_ids_ptr);
    assert_eq!(imported_columns[1].device_data_ptr().unwrap(), export_scores_ptr);
    assert_eq!(imported_columns[0].to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(imported_columns[1].to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn mixed_table_device_arrow_fixed_width_values_ptr_rejects_non_fixed_width_column() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let words = runtime.series("words", &["alpha", "beta"]).unwrap().as_column().deep_copy().unwrap();
    let ids = runtime.column_from_slice(&[1i32, 2]).unwrap();
    let ids_ptr = ids.device_data_ptr().unwrap();
    let table = Table::from_columns(vec![words, ids]).unwrap();

    let export = table.try_prepare_device_arrow_export().unwrap();
    let err = export
        .fixed_width_values_ptr(0)
        .expect_err("string table column must not expose fixed-width values pointer access");

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert!(err.message().contains("fixed-width"));
    assert_eq!(export.fixed_width_values_ptr(1).unwrap(), ids_ptr);
}

#[test]
fn mixed_table_device_arrow_fixed_width_values_ptr_rejects_bool_list_struct_and_dictionary_columns() {
    let _observation_lock = lock_event_observation();

    let runtime = Runtime::builder().build().unwrap();
    let flags = runtime.series("flags", &[true, false, true]).unwrap().as_column().deep_copy().unwrap();
    let flag_ids = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let flag_ids_ptr = flag_ids.device_data_ptr().unwrap();
    let bool_table = Table::from_columns(vec![flags, flag_ids]).unwrap();
    let bool_export = bool_table.try_prepare_device_arrow_export().unwrap();
    assert_fixed_width_pointer_rejected(
        bool_export.fixed_width_values_ptr(0),
        "bool table column must not expose fixed-width values pointer access",
    );
    assert_eq!(bool_export.fixed_width_values_ptr(1).unwrap(), flag_ids_ptr);

    let list_frame = read_json_dataframe(
        "arrow_device_list_table_fixed_width_ptr.jsonl",
        "{\"values\":[1,2],\"id\":10}\n{\"values\":[],\"id\":20}\n{\"values\":[3],\"id\":30}\n",
    );
    let list_values = list_frame.series("values").unwrap().as_column().deep_copy().unwrap();
    let list_ids = list_frame.series("id").unwrap().as_column().deep_copy().unwrap();
    let list_ids_ptr = list_ids.device_data_ptr().unwrap();
    let list_table = Table::from_columns(vec![list_values, list_ids]).unwrap();
    let list_export = list_table.try_prepare_device_arrow_export().unwrap();
    assert_fixed_width_pointer_rejected(
        list_export.fixed_width_values_ptr(0),
        "list table column must not expose fixed-width values pointer access",
    );
    assert_eq!(list_export.fixed_width_values_ptr(1).unwrap(), list_ids_ptr);

    let struct_frame = read_json_dataframe(
        "arrow_device_struct_table_fixed_width_ptr.jsonl",
        "{\"point\":{\"x\":1,\"y\":2},\"id\":7}\n{\"point\":{\"x\":3,\"y\":4},\"id\":8}\n",
    );
    let struct_values = struct_frame.series("point").unwrap().as_column().deep_copy().unwrap();
    let struct_ids = struct_frame.series("id").unwrap().as_column().deep_copy().unwrap();
    let struct_ids_ptr = struct_ids.device_data_ptr().unwrap();
    let struct_table = Table::from_columns(vec![struct_values, struct_ids]).unwrap();
    let struct_export = struct_table.try_prepare_device_arrow_export().unwrap();
    assert_fixed_width_pointer_rejected(
        struct_export.fixed_width_values_ptr(0),
        "struct table column must not expose fixed-width values pointer access",
    );
    assert_eq!(struct_export.fixed_width_values_ptr(1).unwrap(), struct_ids_ptr);

    let dictionary_frame = read_json_dataframe(
        "arrow_device_dictionary_table_fixed_width_ptr.jsonl",
        "{\"category\":\"alpha\",\"id\":1}\n{\"category\":\"beta\",\"id\":2}\n{\"category\":\"alpha\",\"id\":3}\n",
    );
    let dictionary_source = dictionary_frame.series("category").unwrap().as_column().deep_copy().unwrap();
    let dictionary_values = dictionary::encode(&dictionary_source.testing_view().unwrap(), TypeId::Int32).unwrap();
    let dictionary_ids = dictionary_frame.series("id").unwrap().as_column().deep_copy().unwrap();
    let dictionary_ids_ptr = dictionary_ids.device_data_ptr().unwrap();
    let dictionary_table = Table::from_columns(vec![dictionary_values, dictionary_ids]).unwrap();
    let dictionary_export = dictionary_table.try_prepare_device_arrow_export().unwrap();
    assert_fixed_width_pointer_rejected(
        dictionary_export.fixed_width_values_ptr(0),
        "dictionary table column must not expose fixed-width values pointer access",
    );
    assert_eq!(dictionary_export.fixed_width_values_ptr(1).unwrap(), dictionary_ids_ptr);
}

#[test]
fn numeric_table_device_arrow_export_records_sync_event_on_runtime_stream() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = runtime.column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![ids, scores]).unwrap();
    let runtime_stream_raw = runtime.stream().testing_raw_handle_value();

    reset_event_observation();
    let baseline_record = observed_raw_record_call_count_for_stream_raw(runtime_stream_raw);

    let export = table.try_prepare_device_arrow_export().unwrap();

    assert!(export.sync_event_present());
    assert!(
        observed_raw_record_call_count_for_stream_raw(runtime_stream_raw) > baseline_record,
        "table device-arrow export should record a sync event on the runtime stream",
    );
    assert_eq!(observed_last_raw_record_stream_raw(), runtime_stream_raw);
    assert_eq!(observed_last_raw_record_event_raw(), raw_cuda_event_from_handle(export.sync_event_raw_ptr()));
}

#[test]
fn dataframe_device_arrow_table_raw_parts_round_trip_preserves_names_and_values() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[5i32, 6, 7]).unwrap();
    let flags = runtime.series("source_flag", &[true, false, true]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("flag", flags)]).unwrap();

    let raw_parts = df.try_prepare_device_arrow_export().unwrap().into_raw_parts();

    assert_eq!(raw_parts.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(raw_parts.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(raw_parts.device_id(), runtime.device().ordinal() as i32);
    assert!(raw_parts.sync_event_present());
    assert!(!raw_parts.schema_ptr().is_null());
    assert!(!raw_parts.device_array_ptr().is_null());
    assert_eq!(raw_parts.column_names().unwrap(), vec!["id".to_string(), "flag".to_string()]);

    let foreign = raw_parts.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);

    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.column_names(), &["id", "flag"]);
    assert_eq!(imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![5, 6, 7]);
    assert_eq!(imported.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn dataframe_device_arrow_owned_abi_handoff_survives_export_drop() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let foreign = {
        let ids = runtime.series("source_id", &[8i32, 9, 10]).unwrap();
        let flags = runtime.series("source_flag", &[true, false, true]).unwrap();
        let df = DataFrame::from_columns([("id", ids), ("flag", flags)]).unwrap();
        let abi = df.try_prepare_device_arrow_export().unwrap().into_owned_abi_parts();

        assert_eq!(abi.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);
        assert_eq!(abi.device_type(), DeviceArrowDeviceType::Cuda);
        assert_eq!(abi.device_id(), runtime.device().ordinal() as i32);
        assert!(abi.sync_event_present());
        assert_eq!(abi.column_names().unwrap(), vec!["id".to_string(), "flag".to_string()]);

        let (schema, array, materialization) = abi.into_raw_abi();
        unsafe {
            cudf::interop::arrow_device::ForeignDeviceArrowTable::from_raw_abi_parts(schema, array, materialization)
        }
        .expect("owned ABI raw parts should re-wrap as a foreign device-arrow table")
    };

    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(imported.column_names(), &["id", "flag"]);
    assert_eq!(imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![8, 9, 10]);
    assert_eq!(imported.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn dataframe_device_arrow_import_retains_owned_abi_release_until_imported_dataframe_drop() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[8i32, 9, 10]).unwrap();
    let flags = runtime.series("source_flag", &[true, false, true]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("flag", flags)]).unwrap();
    let abi = df.try_prepare_device_arrow_export().unwrap().into_owned_abi_parts();
    let (schema, array, materialization) = abi.into_raw_abi();
    let (schema_releases, array_releases) = unsafe { install_release_observers(schema, array) };
    let foreign = unsafe {
        cudf::interop::arrow_device::ForeignDeviceArrowTable::from_raw_abi_parts(schema, array, materialization)
    }
    .expect("owned ABI raw parts should re-wrap as a foreign device-arrow table");

    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();

    drop(foreign);
    assert_eq!(schema_releases.load(Ordering::SeqCst), 0);
    assert_eq!(array_releases.load(Ordering::SeqCst), 0);

    drop(imported);
    assert_eq!(schema_releases.load(Ordering::SeqCst), 1);
    assert_eq!(array_releases.load(Ordering::SeqCst), 1);
}

#[test]
fn foreign_device_arrow_table_from_raw_abi_parts_rejects_null_schema_and_array() {
    let err = match unsafe {
        cudf::interop::arrow_device::ForeignDeviceArrowTable::from_raw_abi_parts(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            DeviceArrowImportMaterialization::ForeignRetainedView,
        )
    } {
        Ok(_) => panic!("null Arrow device ABI roots must be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert_eq!(err.operation(), "interop::arrow_device::ForeignDeviceArrowTable::from_raw_abi_parts");
    assert!(err.message().contains("schema pointer"));
}

#[test]
fn bool_table_device_arrow_export_reports_bridge_owned_device_copy() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let flags = runtime.column_from_slice(&[true, false, true]).unwrap();
    let table = Table::from_columns(vec![ids, flags]).unwrap();

    let export = table.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(export.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
}

#[test]
fn numeric_table_device_arrow_import_reports_foreign_retained_view_and_preserves_pointers() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.column_from_slice(&[7i32, 8, 9]).unwrap();
    let scores = runtime.column_from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let ids_ptr = ids.device_data_ptr().unwrap();
    let scores_ptr = scores.device_data_ptr().unwrap();
    let table = Table::from_columns(vec![ids, scores]).unwrap();

    let export = table.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);
    assert_eq!(foreign.device_id(), runtime.device().ordinal() as i32);

    let imported = Table::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();
    let imported_columns = imported.into_columns().unwrap();
    assert_eq!(imported_columns.len(), 2);
    assert_eq!(imported_columns[0].device_data_ptr().unwrap(), ids_ptr);
    assert_eq!(imported_columns[1].device_data_ptr().unwrap(), scores_ptr);
    assert_eq!(
        Series::from_column(imported_columns[0].deep_copy().unwrap()).unwrap().to_vec_strict::<i32>().unwrap(),
        vec![7, 8, 9]
    );
    assert_eq!(
        Series::from_column(imported_columns[1].deep_copy().unwrap()).unwrap().to_vec_strict::<f64>().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
}

#[test]
fn bool_table_device_arrow_import_reports_bridge_owned_device_copy() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.column_from_slice(&[1i32, 2, 3, 4]).unwrap();
    let flags = runtime.column_from_slice(&[true, false, true, true]).unwrap();
    let table = Table::from_columns(vec![ids, flags]).unwrap();

    let export = table.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy);
    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::Cuda);

    let imported = Table::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();
    let imported_columns = imported.into_columns().unwrap();
    assert_eq!(imported_columns.len(), 2);
    assert_eq!(
        Series::from_column(imported_columns[0].deep_copy().unwrap()).unwrap().to_vec_strict::<i32>().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        Series::from_column(imported_columns[1].deep_copy().unwrap()).unwrap().to_vec_strict::<bool>().unwrap(),
        vec![true, false, true, true]
    );
}

#[test]
fn decimal64_dataframe_device_arrow_round_trip_preserves_names_scale_and_values() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[3i32, 1, 4]).unwrap();
    let amounts = decimal64_series(&runtime, "source_amount", &[12345i64, -67890, 5], -2);
    let df = DataFrame::from_columns([("id", ids), ("amount", amounts.clone())]).unwrap();

    let export = df.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);

    let foreign = export.into_foreign();
    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.column_names(), &["id", "amount"]);
    assert_eq!(imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![3, 1, 4]);
    assert_eq!(
        imported.series("amount").unwrap().as_column().data_type().unwrap(),
        DataType::fixed_point(TypeId::Decimal64, -2)
    );
    assert_eq!(
        imported.series("amount").unwrap().eq(&amounts).unwrap().to_vec_strict::<bool>().unwrap(),
        vec![true, true, true]
    );
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn string_dataframe_device_arrow_round_trip_preserves_names_and_values() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[3i32, 1, 4, 1]).unwrap();
    let words = runtime
        .nullable_series("source_words", &["alpha", "", "placeholder", "omega"], &[true, true, false, true])
        .unwrap();
    let df = DataFrame::from_columns([("id", ids), ("word", words)]).unwrap();

    let export = df.try_prepare_device_arrow_export().unwrap();
    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);

    let foreign = export.into_foreign();
    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);

    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.column_names(), &["id", "word"]);
    assert_eq!(imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![3, 1, 4, 1]);
    assert_eq!(
        imported.series("word").unwrap().to_optional_strings().unwrap(),
        vec![Some("alpha".to_string()), Some(String::new()), None, Some("omega".to_string()),]
    );
}

#[test]
fn string_dataframe_device_arrow_round_trip_does_not_call_host_arrow_path() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[3i32, 1, 4, 1]).unwrap();
    let words = runtime
        .nullable_series("source_words", &["alpha", "", "placeholder", "omega"], &[true, true, false, true])
        .unwrap();
    let df = DataFrame::from_columns([("id", ids), ("word", words)]).unwrap();

    let export = df.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();
    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.column_names(), &["id", "word"]);
    assert_eq!(
        imported.series("word").unwrap().to_optional_strings().unwrap(),
        vec![Some("alpha".to_string()), Some(String::new()), None, Some("omega".to_string()),]
    );
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn mixed_nested_dataframe_device_arrow_round_trip_preserves_names_structure_and_retained_views() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let source = nested_dictionary_dataframe(
        "arrow_device_nested_frame.jsonl",
        "{\"name\":\"alice\",\"lists\":[1,2],\"point\":{\"coords\":[1,2],\"label\":\"one\"},\"category\":\"beta\"}\n{\"name\":\"bob\",\"lists\":[3],\"point\":{\"coords\":[],\"label\":\"two\"},\"category\":\"alpha\"}\n{\"name\":\"cara\",\"lists\":[],\"point\":{\"coords\":[3],\"label\":\"three\"},\"category\":\"beta\"}\n",
    );

    let export = source.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.materialization(), DeviceArrowExportMaterialization::ProducerRetainedView);

    let foreign = export.into_foreign();
    assert_eq!(foreign.materialization(), DeviceArrowImportMaterialization::ForeignRetainedView);

    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();
    let (source_table, source_names) = source.clone().try_into_table().unwrap();
    let (imported_table, imported_names) = imported.try_into_table().unwrap();

    assert_eq!(source_names, imported_names);
    assert_eq!(imported_names, vec!["name", "lists", "point", "category"]);
    assert_host_table_exports_equivalent(&source_table, &imported_table);
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn string_dataframe_device_arrow_round_trip_matches_host_arrow_round_trip_for_values() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[3i32, 1, 4, 1]).unwrap();
    let words = runtime
        .nullable_series("source_words", &["alpha", "", "placeholder", "omega"], &[true, true, false, true])
        .unwrap();
    let df = DataFrame::from_columns([("id", ids), ("word", words)]).unwrap();

    let device_foreign = df.try_prepare_device_arrow_export().unwrap().into_foreign();
    let device_imported =
        DataFrame::try_import_from_device_arrow_table_into_runtime(&device_foreign, &runtime).unwrap();
    let host_batch = df.to_record_batch().unwrap();
    let host_imported = DataFrame::from_record_batch(&runtime, &host_batch).unwrap();

    assert_eq!(device_imported.column_names(), host_imported.column_names());
    assert_eq!(
        device_imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
        host_imported.series("id").unwrap().to_vec_strict::<i32>().unwrap()
    );
    assert_eq!(
        device_imported.series("word").unwrap().to_optional_strings().unwrap(),
        host_imported.series("word").unwrap().to_optional_strings().unwrap()
    );
}

#[test]
fn dataframe_device_arrow_round_trip_preserves_names_and_values() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[3i32, 1, 4]).unwrap();
    let flags = runtime.series("source_flag", &[true, false, true]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("flag", flags)]).unwrap();

    let export = df.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();
    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.column_names(), &["id", "flag"]);
    assert_eq!(imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![3, 1, 4]);
    assert_eq!(imported.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn dataframe_device_arrow_export_uses_producer_execution_domain() {
    let _observation_lock = lock_event_observation();
    reset_exec_observation();

    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[3i32, 1, 4]).unwrap();
    let flags = runtime.series("source_flag", &[true, false, true]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("flag", flags)]).unwrap();

    let export = df.try_prepare_device_arrow_export().unwrap();

    assert_eq!(export.device_id(), runtime.device().ordinal() as i32);
    assert!(export.sync_event_present());
    assert_export_exec_observation_matches_runtime(&runtime);
}

#[test]
fn dataframe_device_arrow_round_trip_does_not_call_host_arrow_path() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[3i32, 1, 4]).unwrap();
    let flags = runtime.series("source_flag", &[true, false, true]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("flag", flags)]).unwrap();

    let export = df.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();
    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime).unwrap();

    assert_eq!(imported.column_names(), &["id", "flag"]);
    assert_eq!(imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![3, 1, 4]);
    assert_eq!(imported.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn dataframe_device_arrow_import_waits_on_sync_event_via_consumer_stream_without_host_fencing() {
    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().build().unwrap();
    let consumer_runtime = Runtime::builder().build().unwrap();
    let ids = producer_runtime.series("source_id", &[3i32, 1, 4]).unwrap();
    let flags = producer_runtime.series("source_flag", &[true, false, true]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("flag", flags)]).unwrap();
    let export = df.try_prepare_device_arrow_export().unwrap();
    let exported_sync_event = export.sync_event_raw_ptr();
    let consumer_stream_raw = consumer_runtime.stream().testing_raw_handle_value();

    reset_event_observation();
    let baseline_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);

    let foreign = export.into_foreign();
    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &consumer_runtime).unwrap();

    assert_eq!(imported.column_names(), &["id", "flag"]);
    assert_eq!(imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![3, 1, 4]);
    assert_eq!(imported.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(foreign.sync_event_raw_ptr(), exported_sync_event);
    assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_wait);
    assert_eq!(observed_event_synchronize_call_count(), 0);
}

#[test]
fn dataframe_device_arrow_round_trip_matches_host_arrow_round_trip_for_first_slice_values() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("source_id", &[3i32, 1, 4]).unwrap();
    let flags = runtime.series("source_flag", &[true, false, true]).unwrap();
    let scores = runtime.series("source_score", &[1.25f64, 2.5, 3.75]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("flag", flags), ("score", scores)]).unwrap();

    let device_foreign = df.try_prepare_device_arrow_export().unwrap().into_foreign();
    let device_imported =
        DataFrame::try_import_from_device_arrow_table_into_runtime(&device_foreign, &runtime).unwrap();
    let host_batch = df.to_record_batch().unwrap();
    let host_imported = DataFrame::from_record_batch(&runtime, &host_batch).unwrap();

    assert_eq!(device_imported.column_names(), host_imported.column_names());
    assert_eq!(
        device_imported.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
        host_imported.series("id").unwrap().to_vec_strict::<i32>().unwrap()
    );
    assert_eq!(
        device_imported.series("flag").unwrap().to_vec_strict::<bool>().unwrap(),
        host_imported.series("flag").unwrap().to_vec_strict::<bool>().unwrap()
    );
    assert_eq!(
        device_imported.series("score").unwrap().to_vec_strict::<f64>().unwrap(),
        host_imported.series("score").unwrap().to_vec_strict::<f64>().unwrap()
    );
}

#[test]
fn table_device_arrow_import_rejects_different_runtime_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    let consumer_runtime = Runtime::builder().with_device(DeviceId::new(1).unwrap()).build().unwrap();
    let ids = producer_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = producer_runtime.column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![ids, scores]).unwrap();
    let export = table.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    let err = Table::try_import_from_device_arrow_table_into_runtime(&foreign, &consumer_runtime)
        .expect_err("cross-device device-arrow table import should reject mismatched runtime placement");

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert!(err.message().contains("same CUDA device"));
}

#[test]
fn device_arrow_import_rejects_different_runtime_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    let consumer_runtime = Runtime::builder().with_device(DeviceId::new(1).unwrap()).build().unwrap();
    let series = producer_runtime.series("values", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    let err = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &consumer_runtime)
        .expect_err("cross-device device-arrow import should reject mismatched runtime placement");

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert!(err.message().contains("same CUDA device"));
}

#[test]
fn device_arrow_import_rejects_non_cuda_device_type() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[1i32, 2, 3]).unwrap();
    let raw_parts = series.try_prepare_device_arrow_export().unwrap().into_raw_parts();

    unsafe {
        cudf::testing::force_device_arrow_array_device_type_for_test(
            raw_parts.device_array_ptr(),
            cudf::testing::arrow_device_cpu_type_for_test(),
        );
    }

    assert_eq!(raw_parts.device_type(), DeviceArrowDeviceType::Other(cudf::testing::arrow_device_cpu_type_for_test()));

    let foreign = raw_parts.into_foreign();
    let err = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime)
        .expect_err("non-CUDA Arrow device roots should be rejected at the safe Rust import boundary");

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert!(err.message().contains("ARROW_DEVICE_CUDA"));
}

#[test]
fn device_arrow_import_accepts_cuda_host_device_type() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("values", &[11i32, 22, 33]).unwrap();
    let producer_ptr = series.as_column().device_data_ptr().unwrap();
    let raw_parts = series.try_prepare_device_arrow_export().unwrap().into_raw_parts();

    unsafe {
        cudf::testing::force_device_arrow_array_device_type_for_test(
            raw_parts.device_array_ptr(),
            cudf::testing::arrow_device_cuda_host_type_for_test(),
        );
    }

    assert_eq!(raw_parts.device_type(), DeviceArrowDeviceType::CudaHost);

    let foreign = raw_parts.into_foreign();
    let imported = Column::try_import_from_device_arrow_column_into_runtime(&foreign, &runtime)
        .expect("CUDA_HOST Arrow device roots should be accepted at the safe Rust import boundary");

    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::CudaHost);
    assert_eq!(imported.device_data_ptr().unwrap(), producer_ptr);
    assert_eq!(Series::from_column(imported).unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn table_device_arrow_import_accepts_cuda_managed_device_type() {
    let _observation_lock = lock_event_observation();
    cudf::testing::reset_arrow_host_interop_observation_for_test();

    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.column_from_slice(&[5i32, 6, 7]).unwrap();
    let scores = runtime.column_from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let ids_ptr = ids.device_data_ptr().unwrap();
    let scores_ptr = scores.device_data_ptr().unwrap();
    let table = Table::from_columns(vec![ids, scores]).unwrap();
    let raw_parts = table.try_prepare_device_arrow_export().unwrap().into_raw_parts();

    unsafe {
        cudf::testing::force_device_arrow_array_device_type_for_test(
            raw_parts.device_array_ptr(),
            cudf::testing::arrow_device_cuda_managed_type_for_test(),
        );
    }

    assert_eq!(raw_parts.device_type(), DeviceArrowDeviceType::CudaManaged);

    let foreign = raw_parts.into_foreign();
    let imported = Table::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime)
        .expect("CUDA_MANAGED Arrow device roots should be accepted at the safe Rust import boundary");

    assert_eq!(foreign.device_type(), DeviceArrowDeviceType::CudaManaged);
    let imported_columns = imported.into_columns().unwrap();
    assert_eq!(imported_columns.len(), 2);
    assert_eq!(imported_columns[0].device_data_ptr().unwrap(), ids_ptr);
    assert_eq!(imported_columns[1].device_data_ptr().unwrap(), scores_ptr);
    assert_eq!(
        Series::from_column(imported_columns[0].deep_copy().unwrap()).unwrap().to_vec_strict::<i32>().unwrap(),
        vec![5, 6, 7]
    );
    assert_eq!(
        Series::from_column(imported_columns[1].deep_copy().unwrap()).unwrap().to_vec_strict::<f64>().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(cudf::testing::arrow_host_export_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow_host_import_call_count_for_test(), 0);
}

#[test]
fn series_device_arrow_import_rejects_different_runtime_device() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    let consumer_runtime = Runtime::builder().with_device(DeviceId::new(1).unwrap()).build().unwrap();
    let series = producer_runtime.series("values", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_arrow_export().unwrap();
    let foreign = export.into_foreign();

    let err = Series::try_import_from_device_arrow_column_into_runtime(&foreign, &consumer_runtime)
        .expect_err("cross-device series device-arrow import should reject mismatched runtime placement");

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert!(err.message().contains("same CUDA device"));
}

#[test]
fn table_device_arrow_import_rejects_non_cuda_device_type() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = runtime.column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![ids, scores]).unwrap();
    let raw_parts = table.try_prepare_device_arrow_export().unwrap().into_raw_parts();

    unsafe {
        cudf::testing::force_device_arrow_array_device_type_for_test(
            raw_parts.device_array_ptr(),
            cudf::testing::arrow_device_cpu_type_for_test(),
        );
    }

    assert_eq!(raw_parts.device_type(), DeviceArrowDeviceType::Other(cudf::testing::arrow_device_cpu_type_for_test()));

    let foreign = raw_parts.into_foreign();
    let err = Table::try_import_from_device_arrow_table_into_runtime(&foreign, &runtime)
        .expect_err("non-CUDA Arrow device table roots should be rejected at the safe Rust import boundary");

    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert!(err.message().contains("ARROW_DEVICE_CUDA"));
}
