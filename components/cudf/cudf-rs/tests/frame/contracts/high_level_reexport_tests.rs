//! Public API smoke tests.
//!
//! These intentionally keep only a small amount of import-surface coverage:
//! one crate-root workflow smoke test, one runtime-first factory smoke test,
//! one scoped control-plane smoke test, and one advanced owned/core access
//! smoke test.

use crate::common::column_from_slice;

#[test]
fn crate_root_doc_workflow_smoke() {
    use cudf::{DataFrame, Runtime};

    let rt = Runtime::builder().build().unwrap();
    let ids = rt.series(&[1i32, 2, 3]).unwrap();
    let scores = rt.series(&[10.0f64, 20.0, 30.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);

    let score_series = df.series("score").unwrap();
    assert_eq!(score_series.sum_value::<f64>().unwrap(), 60.0);
    assert_eq!(score_series.mean_value().unwrap(), Some(20.0));
}

#[test]
fn host_column_data_is_publicly_reexported() {
    use cudf::{HostColumnData, Series};

    fn build_series<T: HostColumnData>(name: &str, data: &[T]) -> cudf::Result<Series> {
        crate::common::series(name, data)
    }

    let strings = build_series("names", &["alice", "bob"]).unwrap();
    assert_eq!(strings.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);
}

#[test]
fn runtime_first_factories_smoke() {
    use cudf::{Runtime, TypeId};

    let rt = Runtime::builder().build().unwrap();
    let ints = rt.named_series("ints", &[1i32, 2, 3]).unwrap();
    let nullable = rt.named_nullable_series("nullable", &[10i32, 0, 30], &[true, false, true]).unwrap();
    let strings = rt.named_series("names", &["alice", "bob"]).unwrap();
    let nulls = rt.named_nulls("missing", TypeId::Int32, 2).unwrap();
    let repeated = rt.named_series_from_scalar("seed", &rt.scalar(7i32).unwrap(), 3).unwrap();

    assert_eq!(ints.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(nullable.to_optional_vec::<i32>().unwrap(), vec![Some(10), None, Some(30)]);
    assert_eq!(strings.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);
    assert_eq!(nulls.null_count(), 2);
    assert_eq!(repeated.to_vec_strict::<i32>().unwrap(), vec![7, 7, 7]);
}

#[test]
fn factorization_surface_smoke() {
    use cudf::frame::factorization::{
        Codebook, DictionaryEncodingOptions, DictionaryKeyOrder, EncodedSeries, Factorized,
    };

    fn assert_factorized_types(_: &Factorized, _: &Codebook, _: &EncodedSeries) {}

    let series = crate::common::series("kind", &["a", "b", "a"]).unwrap();
    let factorized = series.factorize().unwrap();
    assert_factorized_types(&factorized, factorized.codebook(), factorized.row_codes());

    let decoded = factorized.decode().unwrap();
    assert_eq!(decoded.name(), Some("kind"));
    assert_eq!(decoded.to_strings_strict().unwrap(), vec!["a", "b", "a"]);

    let first_seen =
        series.factorize_with(DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen)).unwrap();
    assert_eq!(first_seen.codebook().keys().to_strings_strict().unwrap(), vec!["a", "b"]);
}

#[test]
fn lookup_table_is_publicly_reexported() {
    use cudf::LookupTable;

    let keys = crate::common::series("id", &["u1", "u2"]).unwrap();
    let values = crate::common::series("label", &["alice", "bob"]).unwrap();
    let table = LookupTable::new(keys, values).unwrap();
    let input = crate::common::series("vertex", &["u2", "u9"]).unwrap();

    let probed = table.probe(&input).unwrap();

    assert_eq!(probed.name(), Some("label"));
    assert_eq!(probed.to_optional_strings().unwrap(), vec![Some("bob".to_string()), None]);
}

#[test]
fn scoped_control_plane_smoke() {
    use cudf::{AggSpec, DataFrame, ExecutionOverrides, GroupByAgg, Runtime, SortOptions, runtime::CudaStream};

    let rt = Runtime::builder().build().unwrap();
    let df = DataFrame::from_columns([
        ("id", rt.series(&[2i32, 1, 2]).unwrap()),
        ("value", rt.series(&[10i64, 20, 30]).unwrap()),
    ])
    .unwrap();

    let override_stream = CudaStream::new().unwrap();
    let execution = ExecutionOverrides::new().with_stream(&override_stream);

    let sorted = df.with_execution(&execution).sort_values(["id"], SortOptions::new()).unwrap();
    let grouped =
        sorted.groupby(["id"]).with_execution(&execution).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    assert_eq!(grouped.column_names(), vec!["id", "value_sum"]);
    assert_eq!(grouped.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(grouped.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap(), vec![20, 40]);
}

#[test]
fn advanced_owned_and_high_level_ops_smoke() {
    use cudf::DataFrame;

    let ids = crate::common::unnamed_series::<i32>(&[10, 20, 30]).unwrap();
    let scores = crate::common::unnamed_series::<f64>(&[1.0, 2.0, 3.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();
    let gather_map = cudf::Series::from_column(column_from_slice(&[2i32, 0]).unwrap()).unwrap();

    let gathered = df.gather(&gather_map).unwrap();
    let id_series = gathered.series("id").unwrap();
    let score_series = gathered.series("score").unwrap();
    let id_column = id_series.as_column();
    let score_column = score_series.as_column();

    assert_eq!(id_column.len().unwrap(), 2);
    assert_eq!(score_column.len().unwrap(), 2);
    assert_eq!(id_column.to_optional_vec::<i32>().unwrap(), vec![Some(30), Some(10)]);
    assert_eq!(score_column.to_optional_vec::<f64>().unwrap(), vec![Some(3.0), Some(1.0)]);
}

#[test]
fn empty_dataframe_table_copy_smoke() {
    use cudf::Runtime;

    let rt = Runtime::builder().build().unwrap();
    let df = rt.empty_dataframe();
    let table = df.to_table_copy().unwrap();

    assert_eq!(table.num_columns().unwrap(), 0);
    assert_eq!(table.num_rows().unwrap(), 0);
}

#[test]
fn frame_submodules_expose_specialized_high_level_types() {
    use cudf::{
        Runtime,
        frame::{
            dataframe::{JsonReadOptions, MeltOptions, ParquetReadOptions},
            factorization::{DictionaryEncodingOptions, DictionaryKeyOrder},
        },
    };

    let rt = Runtime::builder().build().unwrap();
    let _json = JsonReadOptions::builder().filepath("data.jsonl").with_runtime(&rt).build();
    let _parquet = ParquetReadOptions::builder().filepath("data.parquet").with_runtime(&rt).build();
    let _melt = MeltOptions::builder().id_vars(&["id"]).value_vars(&["value"]).build();
    let _factorization = DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen);
}

#[test]
fn core_and_runtime_modules_keep_explicit_compatibility_surface() {
    use cudf::{
        core::{
            Column, CudfType, DataType, DeviceBuffer, DeviceMemoryResource, DeviceSlice, Error, ErrorKind, ErrorStatus,
            NullEquality, NullPolicy, Result, Scalar, Table, TimestampResolution, TypeId,
        },
        runtime::{CudaEvent, CudaStream, DeviceId, ExecutionOverrides, Runtime, RuntimeBuilder},
    };

    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_result_alias(_: Result<()>) {}
    fn assert_cudf_type<T: CudfType>() {}
    fn assert_column_len(_f: fn(&Column) -> Result<usize>) {}
    fn assert_table_num_rows(_f: fn(&Table) -> Result<usize>) {}
    fn assert_device_slice<'a>(_slice: Option<DeviceSlice<'a>>) {}

    assert_send_sync::<DeviceId>();
    assert_send_sync::<CudaStream>();
    assert_send_sync::<CudaEvent>();
    assert_send_sync::<Runtime>();
    assert_send_sync::<RuntimeBuilder>();
    assert_send_sync::<ExecutionOverrides<'static>>();
    assert_send_sync::<DeviceBuffer>();
    assert_send_sync::<DeviceMemoryResource>();
    assert_send_sync::<Scalar>();

    assert_result_alias(Ok(()));
    assert_cudf_type::<i32>();
    assert_column_len(Column::len);
    assert_table_num_rows(Table::num_rows);
    assert_device_slice(None);

    let _ = DataType::new(TypeId::Int32);
    let _ = Error::new(ErrorKind::InvalidOperation, "surface smoke");
    let _ = ErrorStatus::Permanent;
    let _ = NullEquality::Equal;
    let _ = NullPolicy::Include;
    let _ = TimestampResolution::Milliseconds;
}

#[test]
fn hidden_internal_reexports_are_not_publicly_importable() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/public_api/ui/no_debug_execution_context.rs");
    t.compile_fail("tests/public_api/ui/no_into_with_column_expr.rs");
    t.compile_fail("tests/public_api/ui/no_root_debug_validity_helper.rs");
    t.compile_fail("tests/public_api/ui/no_arrow_debug_triggers.rs");
    t.compile_fail("tests/public_api/ui/no_core_view_types.rs");
    t.compile_fail("tests/public_api/ui/no_column_view_method.rs");
    t.compile_fail("tests/public_api/ui/no_table_view_method.rs");
    t.compile_fail("tests/public_api/ui/no_dataframe_view_method.rs");
    t.compile_fail("tests/public_api/ui/no_series_view_method.rs");
    t.compile_fail("tests/public_api/ui/no_series_as_column_view.rs");
    t.compile_fail("tests/public_api/ui/no_testing_column_view_factorize_parts.rs");
    t.compile_fail("tests/public_api/ui/no_testing_column_view_validity.rs");
    t.compile_fail("tests/public_api/ui/no_series_string_scalar_aliases.rs");
    t.compile_fail("tests/public_api/ui/no_series_bitwise_operator_overloads.rs");
    t.compile_fail("tests/public_api/ui/no_legacy_result_type_names.rs");
    t.compile_fail("tests/public_api/ui/no_execution_context_type.rs");
    t.compile_fail("tests/public_api/ui/no_root_dataframe_specialized_types.rs");
    t.compile_fail("tests/public_api/ui/no_root_factorization_types.rs");
    t.compile_fail("tests/public_api/ui/no_dataframe_in_context.rs");
    t.compile_fail("tests/public_api/ui/no_named_first_runtime_series_constructors.rs");
    t.compile_fail("tests/public_api/ui/no_interop_runtime_wrappers.rs");
    t.compile_fail("tests/public_api/ui/no_interop_shared_contract_nouns.rs");
    t.compile_fail("tests/public_api/ui/no_interop_borrowed_namespace.rs");
    t.compile_fail("tests/public_api/ui/no_interop_prepared_namespace.rs");
    t.compile_fail("tests/public_api/ui/no_interop_transfer_namespace.rs");
    t.pass("tests/public_api/ui/interop_canonical_contract_imports.rs");
}
