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
    let ints = rt.series(&[1i32, 2, 3]).unwrap();
    let nullable = rt.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap();
    let strings = rt.series(&["alice", "bob"]).unwrap();
    let nulls = rt.nulls(TypeId::Int32, 2).unwrap();
    let repeated = rt.series_from_scalar(&rt.scalar(7i32).unwrap(), 3).unwrap();

    assert_eq!(ints.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(nullable.to_optional_vec::<i32>().unwrap(), vec![Some(10), None, Some(30)]);
    assert_eq!(strings.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);
    assert_eq!(nulls.null_count(), 2);
    assert_eq!(repeated.to_vec_strict::<i32>().unwrap(), vec![7, 7, 7]);
}

#[test]
fn runtime_raw_scalar_factories_smoke() {
    use cudf::{DataType, Runtime, TimestampResolution, TypeId};

    let rt = Runtime::builder().build().unwrap();
    let decimal32 = rt.decimal32_raw_scalar(12_345, -2).unwrap();
    let decimal64 = rt.decimal64_raw_scalar(-98_765_432, -4).unwrap();
    let decimal128 = rt.decimal128_raw_scalar(123_456_789_012_345_678_901i128, -6).unwrap();
    let decimal = rt.decimal_raw_scalar(-77, DataType::fixed_point(TypeId::Decimal64, -3)).unwrap();
    let timestamp = rt.timestamp_raw_scalar(1_700_000_123_456, TimestampResolution::Microseconds).unwrap();
    let duration = rt.duration_raw_scalar(-42_000_000, TimestampResolution::Nanoseconds).unwrap();
    let null_decimal = rt.null_decimal32_raw_scalar(-2).unwrap();
    let null_timestamp = rt.null_timestamp_raw_scalar(TimestampResolution::Milliseconds).unwrap();
    let null_duration = rt.null_duration_raw_scalar(TimestampResolution::Seconds).unwrap();

    assert_eq!(decimal32.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal32, -2));
    assert_eq!(decimal32.decimal32_raw_value().unwrap(), 12_345);
    assert_eq!(decimal64.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4));
    assert_eq!(decimal64.decimal64_raw_value().unwrap(), -98_765_432);
    assert_eq!(decimal128.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal128, -6));
    assert_eq!(decimal128.decimal128_raw_value().unwrap(), 123_456_789_012_345_678_901i128);
    assert_eq!(decimal.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal64, -3));
    assert_eq!(decimal.decimal64_raw_value().unwrap(), -77);
    assert_eq!(timestamp.data_type().unwrap(), DataType::new(TypeId::TimestampMicroseconds));
    assert_eq!(timestamp.timestamp_raw_value().unwrap(), 1_700_000_123_456);
    assert_eq!(duration.data_type().unwrap(), DataType::new(TypeId::DurationNanoseconds));
    assert_eq!(duration.duration_raw_value().unwrap(), -42_000_000);
    assert_eq!(null_decimal.data_type().unwrap(), DataType::fixed_point(TypeId::Decimal32, -2));
    assert_eq!(null_timestamp.data_type().unwrap(), DataType::new(TypeId::TimestampMilliseconds));
    assert_eq!(null_duration.data_type().unwrap(), DataType::new(TypeId::DurationSeconds));
    assert!(!null_decimal.is_valid().unwrap());
    assert!(!null_timestamp.is_valid().unwrap());
    assert!(!null_duration.is_valid().unwrap());
}

#[test]
fn ops_expr_raw_literal_functions_are_publicly_importable() {
    use cudf::{
        DataFrame, DataType, Expr, Runtime, TimestampResolution, TypeId,
        ops::expr::{lit_decimal32_raw, lit_decimal64_raw, lit_decimal128_raw, lit_duration_raw, lit_timestamp_raw},
    };

    fn assert_expr(_: Expr) {}

    assert_expr(lit_decimal32_raw(12_345, -2));
    assert_expr(lit_decimal128_raw(123_456_789_012_345_678_901i128, -6));
    assert_expr(lit_timestamp_raw(1_700_000_123_456, TimestampResolution::Microseconds));
    assert_expr(lit_duration_raw(-42_000_000, TimestampResolution::Nanoseconds));

    let rt = Runtime::builder().build().unwrap();
    let df = DataFrame::from_columns([("x", rt.series(&[1i32, 2, 3]).unwrap())]).unwrap();
    let decimal = lit_decimal64_raw(-98_765_432, -4).alias("amount").eval(&df).unwrap();

    assert_eq!(decimal.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4));
    assert_eq!(decimal.len(), 3);
}

#[test]
fn datetime_extract_surface_is_publicly_importable() {
    use cudf::{
        TypeId,
        ops::datetime::{DatetimeComponent, extract_component},
        testing::prelude::*,
    };

    fn assert_component(_: DatetimeComponent) {}

    let ts = crate::common::series("ts", &[0i32, 10_561])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    let view = ts.as_column().testing_view().unwrap();
    let year = extract_component(&view, DatetimeComponent::Year).unwrap();
    let month = ts.dt().month().unwrap();
    let day = ts.dt().day().unwrap();

    assert_component(DatetimeComponent::Month);
    assert_eq!(year.testing_view().unwrap().to_vec::<i16>().unwrap(), vec![1970, 1998]);
    assert_eq!(month.to_vec_strict::<i16>().unwrap(), vec![1, 12]);
    assert_eq!(day.to_vec_strict::<i16>().unwrap(), vec![1, 1]);
}

#[test]
fn string_like_and_slice_surface_is_publicly_importable() {
    use cudf::{StringSliceOptions, TypeId, ops::strings, testing::prelude::*};

    fn assert_options(_: StringSliceOptions) {}

    let options = StringSliceOptions::new(Some(1), Some(3)).with_step(1);
    assert_options(options);

    let s = crate::common::series("s", &["abc", "axc", "abbc"]).unwrap();
    let view = s.as_column().testing_view().unwrap();
    let low_level = strings::like(&view, "a_c", None).unwrap();
    let high_level = s.str().slice(options).unwrap();

    assert_eq!(low_level.testing_view().unwrap().data_type().unwrap().id(), TypeId::Bool8);
    assert_eq!(high_level.dtype().unwrap().id(), TypeId::String);
}

#[test]
fn execution_placement_summary_is_publicly_reexported() {
    use cudf::{DataFrame, ExecutionPlacement, Runtime};

    fn assert_summary(_: ExecutionPlacement) {}

    let rt = Runtime::builder().build().unwrap();
    let series = rt.series(&[1i32, 2, 3]).unwrap();
    let df = DataFrame::from_columns([("id", series.clone())]).unwrap();

    let runtime_placement = rt.placement();
    let series_placement = series.placement();
    let frame_placement = df.placement();

    assert_summary(runtime_placement);
    assert_eq!(series_placement, runtime_placement);
    assert_eq!(frame_placement, runtime_placement);
    assert_eq!(frame_placement.device, rt.device());
    assert_ne!(frame_placement.workflow_id, 0);
    assert_ne!(frame_placement.bundle_id, 0);
    assert_ne!(frame_placement.stream_debug, 0);
    assert_ne!(frame_placement.memory_resource_debug, 0);
}

#[test]
fn boundary_observation_shape_is_publicly_reexported() {
    use cudf::{BoundaryKind, BoundaryObservation, BoundaryObserver, DataFrame, Runtime};

    fn assert_observation(_: BoundaryObservation) {}
    fn assert_observer(_: &dyn BoundaryObserver) {}

    let rt = Runtime::builder().build().unwrap();
    let observer = |_observation: BoundaryObservation| {};
    assert_observer(&observer);
    let df = DataFrame::from_columns([("id", rt.series(&[1i32, 2, 3]).unwrap())]).unwrap();
    let observation = BoundaryObservation {
        kind: BoundaryKind::DeviceToHost,
        placement: df.placement(),
        rows: df.num_rows(),
        columns: df.num_columns(),
        synchronized_for_host: true,
    };

    assert_observation(observation);
    assert_eq!(observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(observation.placement, df.placement());
    assert_eq!(observation.rows, 3);
    assert_eq!(observation.columns, 1);
    assert!(observation.synchronized_for_host);
}

#[test]
fn query_execution_context_is_publicly_reexported() {
    use cudf::{DataFrame, QueryExecutionContext, Runtime, SortOptions};

    fn assert_query_context(_: &QueryExecutionContext<'_>) {}

    let rt = Runtime::builder().build().unwrap();
    let query = rt.query_execution();
    assert_query_context(&query);
    assert_eq!(query.runtime().placement(), rt.placement());
    assert_eq!(query.placement(), rt.placement());
    assert_eq!(query.device(), rt.device());

    let df = DataFrame::from_columns([("id", rt.series(&[2i32, 1, 3]).unwrap())]).unwrap();
    let sorted = df.with_execution(query.overrides()).sort_values(["id"], SortOptions::new()).unwrap();

    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn dataframe_join_helpers_are_publicly_reexported() {
    use cudf::{DataFrame, DataFrameJoinIndices, DataFrameJoinOptions, Runtime};

    fn assert_indices(_: &DataFrameJoinIndices) {}
    fn assert_options(_: DataFrameJoinOptions) {}

    let rt = Runtime::builder().build().unwrap();
    let left = DataFrame::from_columns([
        ("id", rt.series(&[1i32, 2, 3]).unwrap()),
        ("value", rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_columns([("id", rt.series(&[2i32]).unwrap())]).unwrap();
    let options = DataFrameJoinOptions::new().on(["id"]);
    assert_options(options.clone());

    let indices = left.left_semi_join_indices(&right, options).unwrap();
    assert_indices(&indices);
    let materialized = left.materialize_join_indices(&indices, ["value"]).unwrap();
    let semi = left.left_semi_join(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let anti = left.left_anti_join(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    assert_eq!(indices.len(), 1);
    assert_eq!(materialized.series("value").unwrap().to_vec_strict::<i32>().unwrap(), vec![20]);
    assert_eq!(semi.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![2]);
    assert_eq!(anti.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3]);
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
    assert_eq!(decoded.to_strings_strict().unwrap(), vec!["a", "b", "a"]);

    let first_seen =
        series.factorize_with(DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen)).unwrap();
    assert_eq!(first_seen.codebook().keys().to_strings_strict().unwrap(), vec!["a", "b"]);
}

#[test]
fn lookup_table_is_publicly_reexported() {
    use cudf::{ExecutionOverrides, LookupTable, Runtime};

    let runtime = Runtime::builder().build().unwrap();
    let keys = crate::common::series("id", &["u1", "u2"]).unwrap();
    let values = crate::common::series("label", &["alice", "bob"]).unwrap();
    let table = LookupTable::new_in_runtime(&runtime, keys, values).unwrap();
    let copied = table.copy_into_runtime(&runtime).unwrap();
    let input = runtime.series(&["u2", "u9"]).unwrap();
    let ctx = ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());

    assert_eq!(table.placement(), runtime.placement());
    assert_eq!(copied.placement(), runtime.placement());

    let probed = copied.probe_in(&input, &ctx).unwrap();
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
        DataFrame, Runtime,
        frame::{
            dataframe::{GpuRecordBatch, JsonReadOptions, MeltOptions, MultiParquetReadOptions, ParquetReadOptions},
            factorization::{DictionaryEncodingOptions, DictionaryKeyOrder},
        },
    };

    let rt = Runtime::builder().build().unwrap();
    let _json = JsonReadOptions::builder().filepath("data.jsonl").with_runtime(&rt).build();
    let _parquet = ParquetReadOptions::builder().filepath("data.parquet").with_runtime(&rt).build();
    let _multi_parquet =
        MultiParquetReadOptions::builder().filepaths(&["part-0.parquet", "part-1.parquet"]).with_runtime(&rt).build();
    let _melt = MeltOptions::builder().id_vars(&["id"]).value_vars(&["value"]).build();
    let _factorization = DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen);

    let frame = DataFrame::from_columns([("id", rt.series(&[1i32, 2, 3]).unwrap())]).unwrap();
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "id",
        arrow::datatypes::DataType::Int32,
        false,
    )]));
    let batch = GpuRecordBatch::new(frame, schema).unwrap();
    assert_eq!(batch.num_rows(), 3);
}

#[test]
fn hidden_internal_reexports_are_not_publicly_importable() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/public_api/ui/no_debug_execution_context.rs");
    t.compile_fail("tests/public_api/ui/no_into_with_column_expr.rs");
    t.compile_fail("tests/public_api/ui/no_root_debug_validity_helper.rs");
    t.compile_fail("tests/public_api/ui/no_arrow_debug_triggers.rs");
    t.compile_fail("tests/public_api/ui/no_internal_testing_hooks_modules.rs");
    t.compile_fail("tests/public_api/ui/no_testing_flat_primitives.rs");
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
    t.compile_fail("tests/public_api/ui/no_legacy_error_kind_names.rs");
    t.compile_fail("tests/public_api/ui/no_legacy_error_code_names.rs");
    t.compile_fail("tests/public_api/ui/no_execution_context_type.rs");
    t.compile_fail("tests/public_api/ui/no_error_new_with_constructor.rs");
    t.compile_fail("tests/public_api/ui/no_facts_write_surface.rs");
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

#[test]
fn testing_submodules_expose_domain_scoped_hooks() {
    let _ = cudf::testing::arrow::trigger_numeric_to_arrow_error_for_test;
    let _ = cudf::testing::column::trigger_column_view_fixed_width_host_byte_range_overflow_for_test;
    let _ = cudf::testing::concatenate::trigger_concatenate_column_count_limit_error_for_test;
    let _ = cudf::testing::core::stats_memory_resource;
    let _ = cudf::testing::core::trigger_truthful_error_contract_for_test;
    let _ = cudf::testing::core::validity_to_host_null_mask_bytes;
    let _ = cudf::testing::dataframe::trigger_dataframe_row_range_overflow_error_for_test;
    let _ = cudf::testing::dictionary::trigger_match_dictionaries_result_count_mismatch_for_test;
    let _ = cudf::testing::factorization::trigger_factorize_empty_row_code_dtype_error_for_test;
    let _ = cudf::testing::groupby::trigger_groupby_request_count_limit_error_for_test;
    let _ = cudf::testing::interop::trigger_interop_projection_negative_offset_for_test;
    let _ = cudf::testing::join::trigger_merge_reassembly_key_count_error_for_test;
    let _ = cudf::testing::series::trigger_series_row_range_overflow_error_for_test;
    let _ = cudf::testing::sorting::trigger_sort_key_count_limit_error_for_test;
}
