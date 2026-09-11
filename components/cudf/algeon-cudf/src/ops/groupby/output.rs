use crate::{
    core::{
        column::Column,
        error::{FACT_ACTUAL_COUNT, FACT_TABLE_COLUMN_COUNT, Result},
    },
    ops::groupby::{GroupByAggregationOutput, errors::foreign_groupby_contract_error},
};

const HIGH_LEVEL_GROUPBY_RESULT_REMEDY: &str =
    "ensure the foreign groupby backend returns one result column per requested aggregation in stable request order";
const VALUE_COUNTS_GROUPBY_RESULT_REMEDY: &str =
    "ensure the foreign groupby backend returns one key column and one non-empty count result for value_counts";

pub(super) fn take_groupby_result_columns_for_output(
    mut grouped_results: Vec<Vec<Option<Column>>>,
    output_order: &[(usize, usize)],
    operation: &'static str,
) -> Result<Vec<Column>> {
    let mut all_result_cols: Vec<Column> = Vec::with_capacity(output_order.len());
    for (request_index, (group_idx, agg_idx)) in output_order.iter().enumerate() {
        let group_count = grouped_results.len();
        let group = grouped_results.get_mut(*group_idx).ok_or_else(|| {
            foreign_groupby_contract_error(
                operation,
                "groupby result group index out of bounds",
                "groupby_result_shape",
                "result_group_index_out_of_bounds",
                HIGH_LEVEL_GROUPBY_RESULT_REMEDY,
            )
            .with_detail("request_index", request_index.to_string())
            .with_detail("group_index", group_idx.to_string())
            .with_detail("group_count", group_count.to_string())
        })?;

        let result_column_count = group.len();
        let col_slot = group.get_mut(*agg_idx).ok_or_else(|| {
            foreign_groupby_contract_error(
                operation,
                "groupby result index out of bounds",
                "groupby_result_shape",
                "result_column_index_out_of_bounds",
                HIGH_LEVEL_GROUPBY_RESULT_REMEDY,
            )
            .with_detail("request_index", request_index.to_string())
            .with_detail("group_index", group_idx.to_string())
            .with_detail("agg_index", agg_idx.to_string())
            .with_fact(FACT_ACTUAL_COUNT, result_column_count)
            .with_detail("result_column_count", result_column_count.to_string())
        })?;

        let col = col_slot.take().ok_or_else(|| {
            foreign_groupby_contract_error(
                operation,
                "groupby result column consumed more than once",
                "groupby_result_shape",
                "duplicate_result_column_consumption",
                HIGH_LEVEL_GROUPBY_RESULT_REMEDY,
            )
            .with_detail("request_index", request_index.to_string())
            .with_detail("group_index", group_idx.to_string())
            .with_detail("agg_index", agg_idx.to_string())
        })?;

        all_result_cols.push(col);
    }

    Ok(all_result_cols)
}

pub(crate) fn value_counts_result_columns(
    gb: GroupByAggregationOutput,
    operation: &'static str,
) -> Result<(Column, Column)> {
    let mut key_columns = gb.keys.into_columns()?;
    let key_column_count = key_columns.len();
    let key_col = key_columns.pop().ok_or_else(|| {
        foreign_groupby_contract_error(
            operation,
            "groupby aggregate returned no key columns",
            "groupby_result_shape",
            "missing_key_columns",
            VALUE_COUNTS_GROUPBY_RESULT_REMEDY,
        )
        .with_fact(FACT_TABLE_COLUMN_COUNT, key_column_count)
    })?;

    let mut agg_results = gb.results;
    let result_group_count = agg_results.len();
    let mut count_cols = agg_results.pop().ok_or_else(|| {
        foreign_groupby_contract_error(
            operation,
            "groupby aggregate returned no count results",
            "groupby_result_shape",
            "missing_count_results",
            VALUE_COUNTS_GROUPBY_RESULT_REMEDY,
        )
        .with_detail("result_group_count", result_group_count.to_string())
    })?;

    let result_column_count = count_cols.len();
    let count_col = count_cols.pop().ok_or_else(|| {
        foreign_groupby_contract_error(
            operation,
            "groupby aggregate returned empty count result",
            "groupby_result_shape",
            "empty_count_result",
            VALUE_COUNTS_GROUPBY_RESULT_REMEDY,
        )
        .with_fact(FACT_ACTUAL_COUNT, result_column_count)
        .with_detail("result_column_count", result_column_count.to_string())
    })?;

    Ok((key_col, count_col))
}
