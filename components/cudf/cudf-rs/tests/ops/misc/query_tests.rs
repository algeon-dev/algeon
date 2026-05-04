//! Tests for the public query request API.

use cudf::{
    AggSpec, DataFrame, DataType, ErrorCode, GroupByAgg, TypeId,
    ops::expr::col,
    query::{
        NamedExpr, QueryAggregateExpr, QueryAggregateFunction, QueryAggregateSpec, QueryBinaryOp, QueryLiteral,
        QuerySortExpr, QuerySortSpec,
    },
};

fn grouped_df() -> DataFrame {
    let key = crate::common::series::<i32>("k", &[1, 1, 2]).unwrap();
    let value = crate::common::series::<i32>("v", &[10, 20, 30]).unwrap();
    DataFrame::from_columns([("k", key), ("v", value)]).unwrap()
}

#[test]
fn query_literals_and_binary_ops_build_public_exprs() {
    let df = grouped_df();
    let expr = QueryBinaryOp::Add.expr(col("v"), QueryLiteral::Int32(5).into_expr());

    let projected = df.project_query([NamedExpr::new(expr, "v_plus_5")]).unwrap();

    assert_eq!(projected.column_names(), vec!["v_plus_5"]);
    assert_eq!(projected.series("v_plus_5").unwrap().to_vec_strict::<i32>().unwrap(), vec![15, 25, 35]);
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
fn sort_query_accepts_expression_sort_keys_without_returning_temp_columns() {
    let df = grouped_df();
    let spec = QuerySortSpec::new([QuerySortExpr::ascending(-col("v"))]);

    let sorted = df.sort_query(spec).unwrap();

    assert_eq!(sorted.column_names(), vec!["k", "v"]);
    assert_eq!(sorted.series("v").unwrap().to_vec_strict::<i32>().unwrap(), vec![30, 20, 10]);
}
