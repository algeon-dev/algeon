//! Tests for the public query request API.

use cudf::{
    AggSpec, DataFrame, DataType, ErrorCode, ErrorKind, ErrorStatus, GroupByAgg, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::expr::col,
    query::{
        NamedExpr, QueryAggregateExpr, QueryAggregateFunction, QueryAggregateSpec, QueryBinaryOp, QueryLiteral,
        QuerySortExpr, QuerySortSpec,
    },
};

fn grouped_df() -> DataFrame {
    let key = crate::common::series::<i32>("k", &[1, 1, 2]).unwrap();
    let value = crate::common::series::<i32>("v", &[10, 20, 30]).unwrap();
    DataFrame::from_series([("k", key), ("v", value)]).unwrap()
}

#[test]
fn query_literals_and_binary_ops_build_public_exprs() {
    let df = grouped_df();
    let expr = QueryBinaryOp::Add.expr(col("v"), QueryLiteral::Int32(5).into_expr());
    let named = NamedExpr::new(expr, "v_plus_5");

    assert_eq!(named.name(), "v_plus_5");

    let projected = df.project_query([named]).unwrap();

    assert_eq!(projected.column_names(), vec!["v_plus_5"]);
    assert_eq!(projected.series("v_plus_5").unwrap().to_vec_strict::<i32>().unwrap(), vec![15, 25, 35]);
}

#[test]
fn query_request_accessors_expose_lowering_contract_without_public_fields() {
    let named = NamedExpr::new(col("k"), "key");
    assert_eq!(named.name(), "key");
    let (_expr, name) = named.clone().into_parts();
    assert_eq!(name, "key");

    let aggregate = QueryAggregateExpr::new(QueryAggregateFunction::Sum, col("v"), "total_v");
    assert!(matches!(aggregate.function(), QueryAggregateFunction::Sum));
    assert_eq!(aggregate.output_name(), "total_v");
    let (_function, _input, output_name) = aggregate.clone().into_parts();
    assert_eq!(output_name, "total_v");

    let spec = QueryAggregateSpec::new([named], [aggregate]);
    assert_eq!(spec.group_exprs().len(), 1);
    assert_eq!(spec.aggregate_exprs().len(), 1);
    let (groups, aggregates) = spec.into_parts();
    assert_eq!(groups[0].name(), "key");
    assert_eq!(aggregates[0].output_name(), "total_v");

    let sort_key = QuerySortExpr::ascending(col("v")).with_nulls_first(true);
    assert!(!sort_key.is_descending());
    assert!(sort_key.nulls_first());
    let (_expr, descending, nulls_first) = sort_key.clone().into_parts();
    assert!(!descending);
    assert!(nulls_first);

    let sort = QuerySortSpec::new([sort_key]).with_stable(true);
    assert_eq!(sort.expressions().len(), 1);
    assert!(sort.stable());
    assert_eq!(sort.into_expressions().len(), 1);
}

#[test]
fn project_query_uses_required_output_names() {
    let df = grouped_df();

    let projected = df.project_query([NamedExpr::new(col("v") * 2i32, "double_v")]).unwrap();

    assert_eq!(projected.column_names(), vec!["double_v"]);
    assert_eq!(projected.series("double_v").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 40, 60]);
}

#[test]
fn aggregate_specs_expose_stable_naming_contract() {
    let default = AggSpec::new("v", GroupByAgg::Sum);
    assert_eq!(default.input_column(), "v");
    assert!(matches!(default.agg(), GroupByAgg::Sum));
    assert_eq!(default.alias(), None);
    assert_eq!(default.resolved_output_name(), "v_sum");

    let aliased = AggSpec::new("v", GroupByAgg::Max).with_alias("max_value");
    assert_eq!(aliased.input_column(), "v");
    assert!(matches!(aliased.agg(), GroupByAgg::Max));
    assert_eq!(aliased.alias(), Some("max_value"));
    assert_eq!(aliased.resolved_output_name(), "max_value");
}

#[test]
fn aggregate_functions_expose_planner_visible_output_dtypes() {
    assert_eq!(
        QueryAggregateFunction::Sum.output_dtype(&DataType::new(TypeId::Int32)).unwrap(),
        DataType::new(TypeId::Int64)
    );
    assert_eq!(
        QueryAggregateFunction::Mean.output_dtype(&DataType::new(TypeId::Int32)).unwrap(),
        DataType::new(TypeId::Float64)
    );
    assert_eq!(
        QueryAggregateFunction::Count { include_nulls: true }.output_dtype(&DataType::new(TypeId::String)).unwrap(),
        DataType::new(TypeId::Int32)
    );
    assert_eq!(
        QueryAggregateFunction::NUnique.output_dtype(&DataType::new(TypeId::String)).unwrap(),
        DataType::new(TypeId::Int32)
    );
}

#[test]
fn aggregate_functions_lock_unsupported_and_string_minmax_dtype_contracts() {
    let i32_dtype = DataType::new(TypeId::Int32);
    let string_dtype = DataType::new(TypeId::String);

    assert_eq!(QueryAggregateFunction::Min.output_dtype(&string_dtype).unwrap(), string_dtype);
    assert_eq!(QueryAggregateFunction::Max.output_dtype(&string_dtype).unwrap(), string_dtype);
    assert_eq!(QueryAggregateFunction::Median.output_dtype(&i32_dtype).unwrap(), DataType::new(TypeId::Float64));
    assert_eq!(
        QueryAggregateFunction::Std { ddof: 1 }.output_dtype(&i32_dtype).unwrap(),
        DataType::new(TypeId::Float64)
    );
    assert_eq!(
        QueryAggregateFunction::Variance { ddof: 1 }.output_dtype(&i32_dtype).unwrap(),
        DataType::new(TypeId::Float64)
    );

    let err = QueryAggregateFunction::Sum.output_dtype(&string_dtype).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "query::aggregate_output_dtype");
}

#[test]
fn aggregate_query_accepts_group_and_value_expressions_with_stable_names() {
    let df = grouped_df();
    let spec = QueryAggregateSpec::new(
        [NamedExpr::new(col("k"), "key")],
        [QueryAggregateExpr::new(QueryAggregateFunction::Sum, col("v") + 1i32, "sum_plus_one")],
    );

    let aggregated = df.aggregate_query(spec).unwrap().sort_values(["key"], cudf::SortOptions::new()).unwrap();

    assert_eq!(aggregated.column_names(), vec!["key", "sum_plus_one"]);
    assert_eq!(aggregated.series("key").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(aggregated.series("sum_plus_one").unwrap().to_vec_strict::<i64>().unwrap(), vec![32, 31]);
}

#[test]
fn aggregate_query_preserves_multiple_aggregate_aliases_and_order() {
    let df = grouped_df();
    let spec = QueryAggregateSpec::new(
        [NamedExpr::new(col("k"), "key")],
        [
            QueryAggregateExpr::new(QueryAggregateFunction::Sum, col("v") + 1i32, "sum_plus_one"),
            QueryAggregateExpr::new(QueryAggregateFunction::Max, col("v") * 2i32, "max_double"),
            QueryAggregateExpr::new(QueryAggregateFunction::Count { include_nulls: true }, col("v"), "row_count"),
        ],
    );

    let aggregated = df.aggregate_query(spec).unwrap().sort_values(["key"], cudf::SortOptions::new()).unwrap();

    assert_eq!(aggregated.column_names(), vec!["key", "sum_plus_one", "max_double", "row_count"]);
    assert_eq!(aggregated.series("key").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(aggregated.series("sum_plus_one").unwrap().to_vec_strict::<i64>().unwrap(), vec![32, 31]);
    assert_eq!(aggregated.series("max_double").unwrap().to_vec_strict::<i32>().unwrap(), vec![40, 60]);
    assert_eq!(aggregated.series("row_count").unwrap().to_vec_strict::<i32>().unwrap(), vec![2, 1]);
}

#[test]
fn aggregate_query_without_groups_returns_one_row_global_aggregate() {
    let df = grouped_df();
    let spec = QueryAggregateSpec::new(
        [],
        [
            QueryAggregateExpr::new(QueryAggregateFunction::Sum, col("v") + 1i32, "sum_plus_one"),
            QueryAggregateExpr::new(QueryAggregateFunction::Count { include_nulls: true }, col("v"), "row_count"),
        ],
    );

    let aggregated = df.aggregate_query(spec).unwrap();

    assert_eq!(aggregated.num_rows(), 1);
    assert_eq!(aggregated.column_names(), vec!["sum_plus_one", "row_count"]);
    assert_eq!(aggregated.series("sum_plus_one").unwrap().to_vec_strict::<i64>().unwrap(), vec![63]);
    assert_eq!(aggregated.series("row_count").unwrap().to_vec_strict::<i32>().unwrap(), vec![3]);
}

#[test]
fn aggregate_query_rejects_duplicate_output_names_before_execution() {
    let df = grouped_df();
    let spec = QueryAggregateSpec::new(
        [NamedExpr::new(col("k"), "dup")],
        [QueryAggregateExpr::new(QueryAggregateFunction::Max, col("v"), "dup")],
    );

    let err = df.aggregate_query(spec).unwrap_err();

    assert_eq!(err.code(), ErrorCode::DuplicateColumnName);
    assert_eq!(err.detail_value("reason"), Some("duplicate_query_output_name"));
}

#[test]
fn aggregate_query_rejects_empty_aggregate_requests_before_execution() {
    let df = grouped_df();
    let spec = QueryAggregateSpec::new([NamedExpr::new(col("k"), "key")], []);

    let err = df.aggregate_query(spec).unwrap_err();

    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.detail_value("resource"), Some("query_aggregate"));
    assert_eq!(err.detail_value("reason"), Some("empty_aggregate_expressions"));
}

#[test]
fn sort_query_accepts_expression_sort_keys_without_returning_temp_columns() {
    let df = grouped_df();
    let spec = QuerySortSpec::new([QuerySortExpr::ascending(-col("v"))]);

    let sorted = df.sort_query(spec).unwrap();

    assert_eq!(sorted.column_names(), vec!["k", "v"]);
    assert_eq!(sorted.series("v").unwrap().to_vec_strict::<i32>().unwrap(), vec![30, 20, 10]);
}

#[test]
fn sort_query_forwards_nulls_first_descending_and_stable_per_key() {
    let row = crate::common::series::<i32>("row", &[0, 1, 2, 3, 4, 5]).unwrap();
    let a = crate::common::nullable_series::<i32>("a", &[2, 0, 1, 0, 1, 1], &[true, false, true, false, true, true])
        .unwrap();
    let b = crate::common::series::<i32>("b", &[5, 1, 2, 9, 7, 7]).unwrap();
    let df = DataFrame::from_series([("row", row), ("a", a), ("b", b)]).unwrap();

    let spec = QuerySortSpec::new([
        QuerySortExpr::ascending(col("a")).with_nulls_first(true),
        QuerySortExpr::descending(col("b")),
    ])
    .with_stable(true);

    let sorted = df.sort_query(spec).unwrap();

    assert_eq!(sorted.column_names(), vec!["row", "a", "b"]);
    assert_eq!(sorted.series("row").unwrap().to_vec_strict::<i32>().unwrap(), vec![3, 1, 4, 5, 2, 0]);
    assert_eq!(
        sorted.series("a").unwrap().to_optional_vec::<i32>().unwrap(),
        vec![None, None, Some(1), Some(1), Some(1), Some(2)]
    );
    assert_eq!(sorted.series("b").unwrap().to_vec_strict::<i32>().unwrap(), vec![9, 1, 7, 7, 2, 5]);
}

#[test]
fn count_scalar_overflow_uses_shared_range_limit_contract() {
    let err = cudf::testing::query::trigger_count_scalar_overflow_for_test().unwrap_err();

    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "query::count_scalar");
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(i32::MAX as usize + 1)));
    assert_eq!(err.detail_value("actual_count"), None);
    assert_eq!(err.detail_value("resource"), Some("actual_count"));
    assert_eq!(err.detail_value("reason"), Some("exceeds_i32_max"));
    assert_eq!(err.detail_value("remedy"), Some("reduce actual_count to fit within i32::MAX"));
}
