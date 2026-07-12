use cudf::{
    Runtime, Series,
    interop::{
        arrow_device::DeviceArrowColumnExport,
        runtime_release::{OwnedColumnTransferReleaser, TransferCompletionAuthority},
    },
};
use rapids_interop::{
    CompletionAuthority, InteropError, OwnedColumnTransfer, PreparedColumnExport, ReadStartAuthority,
    ReleaseSynchronization,
};

fn assert_series_prepared_export(
    _f: for<'a> fn(&'a Series) -> std::result::Result<PreparedColumnExport<'a>, InteropError>,
) {
}

fn assert_series_runtime_release_export(
    _f: fn(
        Series,
        ReleaseSynchronization,
    ) -> std::result::Result<(OwnedColumnTransfer, OwnedColumnTransferReleaser), InteropError>,
) {
}

fn assert_runtime_releaser_release(
    _f: fn(OwnedColumnTransferReleaser, TransferCompletionAuthority<'_>) -> std::result::Result<(), InteropError>,
) {
}

#[test]
fn interop_canonical_contract_imports_are_publicly_importable() {
    let _ = std::any::TypeId::of::<Runtime>();
    let _ = std::any::TypeId::of::<DeviceArrowColumnExport>();
    let _ = CompletionAuthority::SynchronousByReturn;
    let _ = ReadStartAuthority::SynchronousByReturn;
    let _ = TransferCompletionAuthority::synchronous_by_return();

    assert_series_prepared_export(Series::try_prepare_device_column_export);
    assert_series_runtime_release_export(Series::try_into_owned_transfer_with_release_synchronization);
    assert_runtime_releaser_release(OwnedColumnTransferReleaser::release);
}
