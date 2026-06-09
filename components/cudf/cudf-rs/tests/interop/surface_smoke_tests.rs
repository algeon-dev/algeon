//! Interop namespace smoke tests.
//!
//! This file intentionally keeps only a few representative compile-time and
//! construction-time checks so the interop surface remains usable without
//! locking down every exact re-export and function spelling.

use std::sync::Arc;

use cudf::{
    DataFrame, Runtime, Series,
    core::{Column, Table},
    interop::{
        PartitionedLocalTableResults,
        borrowed::{BorrowedColumnView, BorrowedTableView},
        frame::PartitionedDataFrameExport,
        prepared::{
            ActiveLeaseActivation, ActiveLeaseBackend, CompletionBinding, IncompleteDropBehavior, InteropError,
            LeaseActivation, LeaseBackend, LeaseCompletionResult, LeaseExportContext, LeaseReadiness,
            PreparedColumnExport, PreparedTableExport, ReadStartBinding,
        },
        runtime::cuda_explicit_completion_required_lease_backend,
        transfer::{
            OwnedColumnTransfer, OwnedTableTransfer, PartitionedRuntimeOwnedTableTransferReleaser,
            ReleaseSynchronization, RuntimeOwnedTableTransferReleaser,
        },
    },
};

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

#[test]
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

#[test]
fn interop_runtime_authority_surface_smoke() {
    use cudf::{
        interop::runtime::{
            RuntimeCompletionAuthority, RuntimeLeaseCompletionResult, RuntimeReadStartAuthority,
            cuda_explicit_completion_required_lease_backend as namespaced_cuda_explicit_completion_required_lease_backend,
        },
        runtime::{CudaEvent, CudaStream},
    };

    let stream = CudaStream::new().unwrap();
    let event = CudaEvent::new().unwrap();

    let _read_start = RuntimeReadStartAuthority::consumer_stream(&stream);
    let _completion = RuntimeCompletionAuthority::cross_device_event(&event);
    let _result = RuntimeLeaseCompletionResult::FallbackSynchronized;

    assert_eq!(
        cuda_explicit_completion_required_lease_backend().incomplete_drop_behavior(),
        namespaced_cuda_explicit_completion_required_lease_backend().incomplete_drop_behavior()
    );
}

#[test]
fn interop_high_level_adapter_surface_smoke() {
    fn assert_series_prepared_import(
        _f: for<'a, 'b> fn(&'a PreparedColumnExport<'b>, &'a Runtime) -> std::result::Result<Series, InteropError>,
    ) {
    }
    fn assert_dataframe_prepared_import_with_names(
        _f: for<'a, 'b> fn(
            &'a PreparedTableExport<'b>,
            &'a Runtime,
            Vec<String>,
        ) -> std::result::Result<DataFrame, InteropError>,
    ) {
    }
    fn assert_dataframe_owned_producer(
        _f: fn(
            DataFrame,
            ReleaseSynchronization,
        ) -> std::result::Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError>,
    ) {
    }
    fn assert_partitioned_owned_producer(
        _f: fn(
            PartitionedDataFrameExport,
            ReleaseSynchronization,
        ) -> std::result::Result<
            (rapids_interop::PartitionedOwnedTableTransfer, PartitionedRuntimeOwnedTableTransferReleaser),
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

    assert_series_prepared_import(Series::try_import_from_prepared_column_export_into_runtime);
    assert_dataframe_prepared_import_with_names(
        DataFrame::try_import_from_prepared_table_export_into_runtime_and_names,
    );
    assert_dataframe_owned_producer(DataFrame::try_into_owned_transfer_with_release_synchronization);
    assert_partitioned_owned_producer(PartitionedDataFrameExport::try_into_owned_transfer_with_release_synchronization);
    assert_partitioned_table_prepared_import(
        Table::try_import_partitioned_local_tables_from_prepared_table_export_into_runtimes,
    );
}

#[test]
fn interop_low_level_adapter_surface_smoke() {
    fn assert_column_borrowed_export(
        _f: for<'a> fn(&'a Column) -> std::result::Result<BorrowedColumnView<'a>, InteropError>,
    ) {
    }
    fn assert_column_owned_import(_f: fn(OwnedColumnTransfer) -> std::result::Result<Column, InteropError>) {}
    fn assert_table_borrowed_import(
        _f: for<'a, 'b> fn(&'a BorrowedTableView<'b>, &'a Runtime) -> std::result::Result<Table, InteropError>,
    ) {
    }
    fn assert_table_prepared_export(
        _f: for<'a> fn(&'a Table) -> std::result::Result<PreparedTableExport<'a>, InteropError>,
    ) {
    }

    assert_column_borrowed_export(Column::try_borrow_column_view);
    assert_column_owned_import(Column::try_from_owned_device_column_transfer);
    assert_table_borrowed_import(Table::try_import_from_borrowed_table_view_into_runtime);
    assert_table_prepared_export(Table::try_prepare_device_table_export);
}
