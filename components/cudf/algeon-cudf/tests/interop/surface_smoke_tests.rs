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
        PartitionedLocalTableResults,
        arrow_device::{DeviceArrowColumnExport, DeviceArrowTableExport},
        arrow_host, cuda_event_handle, cuda_explicit_completion_required_lease_backend, cuda_stream_handle,
        frame::PartitionedDataFrameExport,
        runtime_release::{OwnedTableTransferReleaser, PartitionedOwnedTableTransferReleaser},
    },
    runtime::{CudaEvent, CudaStream},
};
use rapids_interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, BorrowedColumnView, BorrowedTableView, CompletionAuthority,
    CompletionBinding, IncompleteDropBehavior, InteropError, LeaseActivation, LeaseBackend, LeaseCompletionResult,
    LeaseExportContext, LeaseReadiness, OwnedColumnTransfer, OwnedTableTransfer, PreparedColumnExport,
    PreparedTableExport, ReadStartAuthority, ReadStartBinding, ReleaseSynchronization,
};

use crate::common::run_named_cases;

struct TestActivation;

impl ActiveLeaseActivation for TestActivation {}

struct TestBackend;

impl ActiveLeaseBackend for TestBackend {
    fn activate(
        &self,
        _export: &LeaseExportContext,
        _read_start: ReadStartBinding,
        _completion: CompletionBinding,
    ) -> std::result::Result<LeaseActivation, InteropError> {
        Ok(LeaseActivation::with_state(Arc::new(TestActivation)))
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        IncompleteDropBehavior::GuaranteedFallback
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> std::result::Result<LeaseCompletionResult, InteropError> {
        Ok(LeaseCompletionResult::FallbackSynchronized)
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> std::result::Result<(), InteropError> {
        Ok(())
    }
}

fn interop_lease_backend_surface_smoke() {
    let export = LeaseExportContext::new(None, LeaseReadiness::HostVisible);
    let wrapped = LeaseBackend::new(TestBackend);
    let backend = TestBackend;

    let _activation = backend
        .activate(&export, ReadStartBinding::SynchronousByReturn, CompletionBinding::SynchronousByReturn)
        .unwrap();
    assert_eq!(wrapped.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);
    assert_eq!(
        backend.complete(&export, CompletionBinding::SynchronousByReturn).unwrap(),
        LeaseCompletionResult::FallbackSynchronized
    );
    assert_eq!(
        LeaseBackend::semantic_explicit_completion_required().incomplete_drop_behavior(),
        IncompleteDropBehavior::ExplicitCompletionRequired
    );
    assert_eq!(
        cuda_explicit_completion_required_lease_backend().incomplete_drop_behavior(),
        IncompleteDropBehavior::ExplicitCompletionRequired
    );
}

fn interop_shared_authority_surface_smoke() {
    let stream = CudaStream::new().unwrap();
    let event = CudaEvent::new().unwrap();

    let _read_start = ReadStartAuthority::ConsumerStream { stream: cuda_stream_handle(&stream).unwrap() };
    let _completion = CompletionAuthority::CrossDeviceEvent { event: cuda_event_handle(&event).unwrap() };
    let _result = LeaseCompletionResult::FallbackSynchronized;

    assert_eq!(
        cuda_explicit_completion_required_lease_backend().incomplete_drop_behavior(),
        IncompleteDropBehavior::ExplicitCompletionRequired
    );
}

fn interop_high_level_adapter_surface_smoke() {
    type DataFrameBorrowedImport = for<'a, 'b> fn(
        &'a BorrowedTableView<'b>,
        Option<Vec<String>>,
        &'a Runtime,
    ) -> std::result::Result<DataFrame, InteropError>;
    type DataFramePreparedImport = for<'a, 'b> fn(
        &'a PreparedTableExport<'b>,
        Option<Vec<String>>,
        &'a Runtime,
    ) -> std::result::Result<DataFrame, InteropError>;
    type DataFrameOwnedImport =
        fn(OwnedTableTransfer, Option<Vec<String>>, &Runtime) -> std::result::Result<DataFrame, InteropError>;

    fn assert_arrow_host_record_batch_import(
        _f: fn(&Runtime, &RecordBatch) -> std::result::Result<DataFrame, cudf::Error>,
    ) {
    }
    fn assert_series_prepared_import(
        _f: for<'a, 'b> fn(&'a PreparedColumnExport<'b>, &'a Runtime) -> std::result::Result<Series, InteropError>,
    ) {
    }
    fn assert_dataframe_borrowed_import(_f: DataFrameBorrowedImport) {}
    fn assert_dataframe_prepared_import(_f: DataFramePreparedImport) {}
    fn assert_dataframe_owned_import(_f: DataFrameOwnedImport) {}
    fn assert_dataframe_owned_producer(
        _f: fn(
            DataFrame,
            ReleaseSynchronization,
        ) -> std::result::Result<(OwnedTableTransfer, OwnedTableTransferReleaser), InteropError>,
    ) {
    }
    fn assert_partitioned_owned_producer(
        _f: fn(
            PartitionedDataFrameExport,
            ReleaseSynchronization,
        ) -> std::result::Result<
            (rapids_interop::PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferReleaser),
            InteropError,
        >,
    ) {
    }
    fn assert_partitioned_table_prepared_import(
        _f: for<'a, 'b> fn(
            &'a rapids_interop::PartitionedPreparedTableExport<'b>,
            &[&Runtime],
        ) -> std::result::Result<PartitionedLocalTableResults, InteropError>,
    ) {
    }

    assert_arrow_host_record_batch_import(arrow_host::dataframe_from_record_batch);
    assert_series_prepared_import(Series::try_import_from_prepared_column_export_into_runtime);
    assert_dataframe_borrowed_import(DataFrame::try_import_from_borrowed_table_view_into_runtime);
    assert_dataframe_prepared_import(DataFrame::try_import_from_prepared_table_export_into_runtime);
    assert_dataframe_owned_import(DataFrame::try_import_from_owned_device_table_transfer_into_runtime);
    assert_dataframe_owned_producer(DataFrame::try_into_owned_transfer_with_release_synchronization);
    assert_partitioned_owned_producer(PartitionedDataFrameExport::try_into_owned_transfer_with_release_synchronization);
    assert_partitioned_table_prepared_import(
        Table::try_import_partitioned_local_tables_from_prepared_table_export_into_runtimes,
    );
}

fn interop_low_level_adapter_surface_smoke() {
    fn assert_column_borrowed_export(
        _f: for<'a> fn(&'a Column) -> std::result::Result<BorrowedColumnView<'a>, InteropError>,
    ) {
    }
    fn assert_column_prepared_root_export(
        _f: fn(Column) -> std::result::Result<PreparedColumnExport<'static>, InteropError>,
    ) {
    }
    fn assert_column_device_arrow_export(_f: fn(Column) -> std::result::Result<DeviceArrowColumnExport, cudf::Error>) {}
    fn assert_column_device_arrow_export_with_deep_copy(
        _f: for<'a> fn(&'a Column) -> std::result::Result<DeviceArrowColumnExport, cudf::Error>,
    ) {
    }
    fn assert_column_owned_import(_f: fn(OwnedColumnTransfer, &Runtime) -> std::result::Result<Column, InteropError>) {}
    fn assert_table_borrowed_import(
        _f: for<'a, 'b> fn(&'a BorrowedTableView<'b>, &'a Runtime) -> std::result::Result<Table, InteropError>,
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

    assert_column_borrowed_export(Column::try_borrow_column_view);
    assert_column_prepared_root_export(Column::try_into_prepared_device_column_export);
    assert_column_device_arrow_export(Column::try_prepare_device_arrow_export);
    assert_column_device_arrow_export_with_deep_copy(Column::try_prepare_device_arrow_export_with_deep_copy);
    assert_column_owned_import(Column::try_import_from_owned_device_column_transfer_into_runtime);
    assert_table_borrowed_import(Table::try_import_from_borrowed_table_view_into_runtime);
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
