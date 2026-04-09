use std::sync::Arc;

use cudf::{
    DataFrame, Runtime, Series,
    interop::{
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
            RuntimeOwnedColumnTransferReleaser, RuntimeOwnedTableTransferReleaser,
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
fn cudf_interop_reexports_shared_activation_backend_surface() {
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
}

#[test]
fn cudf_interop_exposes_builtin_explicit_completion_backend() {
    let wrapped = LeaseBackend::semantic_explicit_completion_required();

    assert_eq!(wrapped.incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);
}

#[test]
fn cudf_interop_exposes_cuda_explicit_completion_backend() {
    let wrapped = cuda_explicit_completion_required_lease_backend();

    assert_eq!(wrapped.incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);
}

#[test]
fn cudf_interop_exposes_high_level_consumer_import_adapters() {
    fn assert_series_borrowed_import(
        _f: for<'a, 'b> fn(&'a BorrowedColumnView<'b>, &'a Runtime) -> std::result::Result<Series, InteropError>,
    ) {
    }
    fn assert_series_prepared_import(
        _f: for<'a, 'b> fn(&'a PreparedColumnExport<'b>, &'a Runtime) -> std::result::Result<Series, InteropError>,
    ) {
    }
    fn assert_series_owned_import(_f: fn(OwnedColumnTransfer) -> std::result::Result<Series, InteropError>) {}
    fn assert_dataframe_borrowed_import(
        _f: for<'a, 'b> fn(
            &'a BorrowedTableView<'b>,
            &'a Runtime,
            Vec<String>,
        ) -> std::result::Result<DataFrame, InteropError>,
    ) {
    }
    fn assert_dataframe_prepared_import(
        _f: for<'a, 'b> fn(&'a PreparedTableExport<'b>, &'a Runtime) -> std::result::Result<DataFrame, InteropError>,
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
    fn assert_dataframe_owned_import(_f: fn(OwnedTableTransfer) -> std::result::Result<DataFrame, InteropError>) {}
    fn assert_dataframe_owned_import_with_names(
        _f: fn(OwnedTableTransfer, Vec<String>) -> std::result::Result<DataFrame, InteropError>,
    ) {
    }

    assert_series_borrowed_import(Series::try_import_from_borrowed_column_view_into_runtime);
    assert_series_prepared_import(Series::try_import_from_prepared_column_export_into_runtime);
    assert_series_owned_import(Series::try_from_owned_device_column_transfer);
    assert_dataframe_borrowed_import(DataFrame::try_import_from_borrowed_table_view_into_runtime_and_names);
    assert_dataframe_prepared_import(DataFrame::try_import_from_prepared_table_export_into_runtime);
    assert_dataframe_prepared_import_with_names(
        DataFrame::try_import_from_prepared_table_export_into_runtime_and_names,
    );
    assert_dataframe_owned_import(DataFrame::try_from_owned_device_table_transfer);
    assert_dataframe_owned_import_with_names(DataFrame::try_from_owned_device_table_transfer_and_names);
}

#[test]
fn cudf_interop_exposes_high_level_owned_transfer_producers() {
    fn assert_series_owned_producer(_f: fn(Series) -> std::result::Result<OwnedColumnTransfer, InteropError>) {}
    fn assert_dataframe_owned_producer(_f: fn(DataFrame) -> std::result::Result<OwnedTableTransfer, InteropError>) {}
    fn assert_series_runtime_release_owned_producer(
        _f: fn(Series) -> std::result::Result<(OwnedColumnTransfer, RuntimeOwnedColumnTransferReleaser), InteropError>,
    ) {
    }
    fn assert_dataframe_runtime_release_owned_producer(
        _f: fn(DataFrame) -> std::result::Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError>,
    ) {
    }
    fn assert_partitioned_dataframe_runtime_release_owned_producer(
        _f: fn(
            PartitionedDataFrameExport,
        ) -> std::result::Result<
            (rapids_interop::PartitionedOwnedTableTransfer, PartitionedRuntimeOwnedTableTransferReleaser),
            InteropError,
        >,
    ) {
    }

    assert_series_owned_producer(Series::try_into_owned_device_column_transfer);
    assert_dataframe_owned_producer(DataFrame::try_into_owned_device_table_transfer);
    assert_series_runtime_release_owned_producer(Series::try_into_owned_device_column_transfer_with_runtime_release);
    assert_dataframe_runtime_release_owned_producer(
        DataFrame::try_into_owned_device_table_transfer_with_runtime_release,
    );
    assert_partitioned_dataframe_runtime_release_owned_producer(
        PartitionedDataFrameExport::try_into_owned_device_partitioned_table_transfer_with_runtime_release,
    );
}

#[test]
fn cudf_interop_exposes_grouped_frame_namespace() {
    let _: Option<PartitionedDataFrameExport> = None;
    let _: Option<PartitionedRuntimeOwnedTableTransferReleaser> = None;
}
