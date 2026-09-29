//! Interop namespace smoke tests.
//!
//! This file intentionally keeps only a few representative compile-time and
//! construction-time checks so the interop surface remains usable without
//! locking down every exact re-export and function spelling.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use cudf::{
    DataFrame, Runtime, Series,
    core::{Column, Table},
    interop::{
        arrow_device::{DeviceArrowColumnExport, DeviceArrowTableExport},
        arrow_host, cuda_stream_handle,
    },
    runtime::CudaStream,
};
use rapids_interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, CompletionAuthority, CompletionWitness, InteropError, LeaseActivation,
    LeaseBackend, PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReadStartAuthority,
};

use crate::common::run_named_cases;

struct TestActivation;

impl ActiveLeaseActivation for TestActivation {}

struct TestBackend;

impl ActiveLeaseBackend for TestBackend {
    fn activate(
        &self,
        _readiness: ProducerReadiness<'_>,
        _read_start: ReadStartAuthority<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> std::result::Result<LeaseActivation, InteropError> {
        Ok(LeaseActivation::with_state(Arc::new(TestActivation)))
    }

    fn complete(
        &self,
        _readiness: ProducerReadiness<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> std::result::Result<Option<CompletionWitness>, InteropError> {
        Ok(None)
    }

    fn drop_incomplete(
        &self,
        _readiness: ProducerReadiness<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> std::result::Result<(), InteropError> {
        Ok(())
    }
}

fn interop_lease_backend_surface_smoke() {
    let stream = CudaStream::new().unwrap();
    let readiness = ProducerReadiness::DeviceStreamOrdered { stream: cuda_stream_handle(&stream).unwrap() };
    let _wrapped = LeaseBackend::new(TestBackend);
    let backend = TestBackend;

    let _activation = backend
        .activate(readiness, ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert!(backend.complete(readiness, CompletionAuthority::SynchronousByReturn).unwrap().is_none());
}

fn interop_shared_authority_surface_smoke() {
    let stream = CudaStream::new().unwrap();

    let _read_start = ReadStartAuthority::ConsumerStream { stream: cuda_stream_handle(&stream).unwrap() };
    let _completion = CompletionAuthority::SameDeviceStream { stream: cuda_stream_handle(&stream).unwrap() };
}

fn interop_high_level_adapter_surface_smoke() {
    type DataFramePreparedImport = for<'a, 'b> fn(
        &'a PreparedTableExport<'b>,
        Option<Vec<String>>,
        &'a Runtime,
    ) -> std::result::Result<DataFrame, InteropError>;

    fn assert_arrow_host_record_batch_import(
        _f: fn(&Runtime, &RecordBatch) -> std::result::Result<DataFrame, cudf::Error>,
    ) {
    }
    fn assert_series_prepared_import(
        _f: for<'a, 'b> fn(&'a PreparedColumnExport<'b>, &'a Runtime) -> std::result::Result<Series, InteropError>,
    ) {
    }
    fn assert_dataframe_prepared_import(_f: DataFramePreparedImport) {}

    assert_arrow_host_record_batch_import(arrow_host::dataframe_from_record_batch);
    assert_series_prepared_import(Series::try_import_from_prepared_column_export_into_runtime);
    assert_dataframe_prepared_import(DataFrame::try_import_from_prepared_table_export_into_runtime);
}

fn interop_low_level_adapter_surface_smoke() {
    fn assert_column_prepared_root_export(
        _f: fn(Column) -> std::result::Result<PreparedColumnExport<'static>, InteropError>,
    ) {
    }
    fn assert_column_device_arrow_export(_f: fn(Column) -> std::result::Result<DeviceArrowColumnExport, cudf::Error>) {}
    fn assert_column_device_arrow_export_with_deep_copy(
        _f: for<'a> fn(&'a Column) -> std::result::Result<DeviceArrowColumnExport, cudf::Error>,
    ) {
    }
    fn assert_table_prepared_export(
        _f: for<'a> fn(&'a Table) -> std::result::Result<PreparedTableExport<'a>, InteropError>,
    ) {
    }
    fn assert_table_prepared_root_export(
        _f: fn(Table) -> std::result::Result<PreparedTableExport<'static>, InteropError>,
    ) {
    }
    fn assert_table_device_arrow_export_with_deep_copy(
        _f: for<'a> fn(&'a Table) -> std::result::Result<DeviceArrowTableExport, cudf::Error>,
    ) {
    }

    assert_column_prepared_root_export(Column::try_into_prepared_device_column_export);
    assert_column_device_arrow_export(Column::try_prepare_device_arrow_export);
    assert_column_device_arrow_export_with_deep_copy(Column::try_prepare_device_arrow_export_with_deep_copy);
    assert_table_prepared_export(Table::try_prepare_device_table_export);
    assert_table_prepared_root_export(Table::try_into_prepared_device_table_export);
    assert_table_device_arrow_export_with_deep_copy(Table::try_prepare_device_arrow_export_with_deep_copy);
}

#[test]
fn surface_smoke_tests_cases() {
    run_named_cases(&[
        ("interop_lease_backend_surface_smoke", interop_lease_backend_surface_smoke),
        ("interop_shared_authority_surface_smoke", interop_shared_authority_surface_smoke),
        ("interop_high_level_adapter_surface_smoke", interop_high_level_adapter_surface_smoke),
        ("interop_low_level_adapter_surface_smoke", interop_low_level_adapter_surface_smoke),
    ]);
}
