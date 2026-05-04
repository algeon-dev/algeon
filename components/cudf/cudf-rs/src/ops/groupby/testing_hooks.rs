use std::sync::Arc;

use super::{
    GroupByAggregationOutput,
    errors::{
        groupby_request_agg_count_limit_error, groupby_request_count_limit_error,
        groupby_total_agg_handle_count_limit_error,
    },
    output::{take_groupby_result_columns_for_output, value_counts_result_columns},
};
use crate::{
    DataFrame, Runtime, Series,
    core::{
        Column, Table,
        error::Result,
        types::{DataType, TypeId},
    },
};

fn empty_i32_column_in_domain(
    output_domain: &Arc<crate::runtime::execution_domain::ExecutionDomain>,
    operation: &'static str,
) -> Result<Column> {
    Column::empty_with_dtype_in_domain(DataType::new(TypeId::Int32), Arc::clone(output_domain), operation)
        .map_err(|e| e.reanchor(operation))
}

pub(crate) fn trigger_groupby_request_count_limit_error_for_test() -> Result<()> {
    Err(groupby_request_count_limit_error("groupby::groupby_aggregate", usize::MAX))
}

pub(crate) fn trigger_groupby_request_agg_count_limit_error_for_test() -> Result<()> {
    Err(groupby_request_agg_count_limit_error("groupby::groupby_aggregate", 0, usize::MAX, 1))
}

pub(crate) fn trigger_groupby_total_agg_handle_count_limit_error_for_test() -> Result<()> {
    Err(groupby_total_agg_handle_count_limit_error("groupby::groupby_aggregate", usize::MAX, 1))
}

pub(crate) fn trigger_groupby_result_group_index_out_of_bounds_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let col = Column::from_slice_in_domain(&[1i32], runtime.execution_domain())?;
    let grouped_results = vec![vec![Some(col)]];
    take_groupby_result_columns_for_output(grouped_results, &[(1, 0)], "groupby::GroupBy::agg").map(|_| ())
}

pub(crate) fn trigger_groupby_result_column_index_out_of_bounds_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let col = Column::from_slice_in_domain(&[1i32], runtime.execution_domain())?;
    let grouped_results = vec![vec![Some(col)]];
    take_groupby_result_columns_for_output(grouped_results, &[(0, 1)], "groupby::GroupBy::agg").map(|_| ())
}

pub(crate) fn trigger_groupby_duplicate_result_column_consumption_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let col = Column::from_slice_in_domain(&[1i32], runtime.execution_domain())?;
    let grouped_results = vec![vec![Some(col)]];
    take_groupby_result_columns_for_output(grouped_results, &[(0, 0), (0, 0)], "groupby::GroupBy::agg").map(|_| ())
}

pub(crate) fn trigger_value_counts_missing_key_columns_for_test() -> Result<DataFrame> {
    let operation = "series::value_counts";
    let runtime = Runtime::new()?;
    let output_domain = Arc::clone(runtime.execution_domain());
    let keys = Table::empty_in_domain(Arc::clone(&output_domain), "testing::value_counts_missing_key_columns")?;

    let (key_col, count_col) =
        value_counts_result_columns(GroupByAggregationOutput { keys, results: vec![Vec::new()] }, operation)?;
    DataFrame::from_series([
        ("value".to_string(), Series::from_column(key_col)?),
        ("count".to_string(), Series::from_column(count_col)?),
    ])
    .map_err(|e| e.reanchor(operation))
}

pub(crate) fn trigger_value_counts_missing_count_results_for_test() -> Result<DataFrame> {
    let operation = "series::value_counts";
    let runtime = Runtime::new()?;
    let output_domain = Arc::clone(runtime.execution_domain());
    let key_col = empty_i32_column_in_domain(&output_domain, "testing::value_counts_missing_count_results")?;
    let keys = Table::from_columns(vec![key_col])?;

    let (key_col, count_col) =
        value_counts_result_columns(GroupByAggregationOutput { keys, results: Vec::new() }, operation)?;
    DataFrame::from_series([
        ("value".to_string(), Series::from_column(key_col)?),
        ("count".to_string(), Series::from_column(count_col)?),
    ])
    .map_err(|e| e.reanchor(operation))
}

pub(crate) fn trigger_value_counts_empty_count_result_for_test() -> Result<DataFrame> {
    let operation = "series::value_counts";
    let runtime = Runtime::new()?;
    let output_domain = Arc::clone(runtime.execution_domain());
    let key_col = empty_i32_column_in_domain(&output_domain, "testing::value_counts_empty_count_result")?;
    let keys = Table::from_columns(vec![key_col])?;

    let (key_col, count_col) =
        value_counts_result_columns(GroupByAggregationOutput { keys, results: vec![Vec::new()] }, operation)?;
    DataFrame::from_series([
        ("value".to_string(), Series::from_column(key_col)?),
        ("count".to_string(), Series::from_column(count_col)?),
    ])
    .map_err(|e| e.reanchor(operation))
}
