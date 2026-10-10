use cudf::{
    Runtime, Series,
    interop::contract::{CompletionAuthority, InteropError, PreparedColumnExport, ReadStartAuthority},
};

fn assert_series_prepared_export(
    _f: for<'a> fn(&'a Series) -> std::result::Result<PreparedColumnExport<'a>, InteropError>,
) {
}

#[test]
fn interop_canonical_contract_imports_are_publicly_importable() {
    let _ = std::any::TypeId::of::<Runtime>();
    let _ = CompletionAuthority::SynchronousByReturn;
    let _ = ReadStartAuthority::SynchronousByReturn;

    assert_series_prepared_export(Series::try_prepare_device_column_export);
}
