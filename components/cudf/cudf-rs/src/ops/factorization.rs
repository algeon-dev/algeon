//! Internal factorization operators shared by high-level Series/DataFrame APIs.

mod errors;
pub(crate) mod testing_hooks;

use std::sync::Arc;

use self::errors::foreign_factorization_row_code_dtype_error;
use crate::{
    core::{
        column::{Column, ColumnView, copy_view_to_owned, position_sequence_in_domain},
        error::Result,
        table::TableView,
        types::TypeId,
    },
    ops::{
        copying, dictionary, lookup,
        sorting::{self, NullOrder, Order},
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
    },
    runtime::execution_domain::ExecutionDomain,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyOrder {
    Sorted,
    FirstSeen,
}

pub(crate) struct FactorizationParts {
    pub(crate) row_codes: Column,
    pub(crate) keys: Column,
}

struct FirstSeenKeyOrder {
    ordered_sorted_codes: Column,
    ordered_keys: Column,
}

struct SortedKeyOrder {
    sorted_original_codes: Column,
    sorted_keys: Column,
}

fn empty_code_column(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<Column> {
    let input_dtype = input.data_type().map_err(|e| e.reanchor(operation))?;
    if input_dtype.id() != TypeId::Int32 {
        return Err(foreign_factorization_row_code_dtype_error(operation, input_dtype.id()));
    }

    Column::empty_in_domain(TypeId::Int32, Arc::clone(output_domain), operation).map_err(|e| e.reanchor(operation))
}

fn ordered_first_seen_keys(
    row_codes: &ColumnView<'_>,
    sorted_keys: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<FirstSeenKeyOrder> {
    if row_codes.len().map_err(|e| e.reanchor(operation))? == 0 {
        return Ok(FirstSeenKeyOrder {
            ordered_sorted_codes: empty_code_column(row_codes, output_domain, operation)?,
            ordered_keys: copy_view_to_owned(sorted_keys).map_err(|e| e.reanchor(operation))?,
        });
    }

    let row_idx =
        position_sequence_in_domain(row_codes.len().map_err(|e| e.reanchor(operation))?, output_domain, operation)
            .map_err(|e| e.reanchor(operation))?;
    let row_idx_view = row_idx.view().map_err(|e| e.reanchor(operation))?;
    let code_and_row = TableView::from_column_views(&[row_codes, &row_idx_view]).map_err(|e| e.reanchor(operation))?;

    let non_null_code_rows =
        stream_compaction::drop_nulls_impl(&code_and_row, &[0], 1, output_domain, exec_options, operation)
            .map_err(|e| e.reanchor(operation))?;
    let non_null_code_rows_view = non_null_code_rows.view().map_err(|e| e.reanchor(operation))?;
    if non_null_code_rows_view.num_rows().map_err(|e| e.reanchor(operation))? == 0 {
        return Ok(FirstSeenKeyOrder {
            ordered_sorted_codes: empty_code_column(row_codes, output_domain, operation)?,
            ordered_keys: copy_view_to_owned(sorted_keys).map_err(|e| e.reanchor(operation))?,
        });
    }

    let first_code_rows = stream_compaction::distinct_impl(
        &non_null_code_rows_view,
        stream_compaction::DistinctConfig::new(
            &[0],
            DuplicateKeepOption::KeepFirst,
            NullEquality::Equal,
            NanEquality::AllEqual,
        ),
        stream_compaction::StreamCompactionExecution::new(output_domain, exec_options, operation),
    )
    .map_err(|e| e.reanchor(operation))?;
    let first_code_rows_view = first_code_rows.view().map_err(|e| e.reanchor(operation))?;
    let first_row_idx = first_code_rows_view.select_columns(&[1]).map_err(|e| e.reanchor(operation))?;
    let ordered_code_rows = sorting::stable_sort_by_key(
        &first_code_rows_view,
        &first_row_idx,
        Some(&[Order::Ascending]),
        Some(&[NullOrder::After]),
    )
    .map_err(|e| e.reanchor(operation))?;

    let mut ordered_columns = ordered_code_rows.into_columns().map_err(|e| e.reanchor(operation))?;
    let ordered_sorted_codes = ordered_columns.remove(0);
    let ordered_keys = {
        let ordered_sorted_codes_view = ordered_sorted_codes.view().map_err(|e| e.reanchor(operation))?;
        let sorted_keys_table = TableView::from_column_views(&[sorted_keys]).map_err(|e| e.reanchor(operation))?;
        let ordered_keys_table =
            copying::gather(&sorted_keys_table, &ordered_sorted_codes_view).map_err(|e| e.reanchor(operation))?;
        let mut ordered_key_columns = ordered_keys_table.into_columns().map_err(|e| e.reanchor(operation))?;
        ordered_key_columns.remove(0)
    };

    Ok(FirstSeenKeyOrder { ordered_sorted_codes, ordered_keys })
}

fn sorted_key_order(
    encoded_keys: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<SortedKeyOrder> {
    let key_count = encoded_keys.len().map_err(|e| e.reanchor(operation))?;
    if key_count == 0 {
        return Ok(SortedKeyOrder {
            sorted_original_codes: Column::empty_in_domain(TypeId::Int32, Arc::clone(output_domain), operation)
                .map_err(|e| e.reanchor(operation))?,
            sorted_keys: copy_view_to_owned(encoded_keys).map_err(|e| e.reanchor(operation))?,
        });
    }

    let encoded_keys_table = TableView::from_column_views(&[encoded_keys]).map_err(|e| e.reanchor(operation))?;
    let sorted_original_codes = sorting::sorted_order_impl(
        &encoded_keys_table,
        sorting::SortKeyConfig::new(Some(&[Order::Ascending]), Some(&[NullOrder::After])),
        sorting::SortExecution::new(output_domain, exec_options, operation),
    )
    .map_err(|e| e.reanchor(operation))?;
    let sorted_keys_table = {
        let sorted_original_codes_view = sorted_original_codes.view().map_err(|e| e.reanchor(operation))?;
        copying::gather_impl(
            &encoded_keys_table,
            &sorted_original_codes_view,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
            "dont_check",
            output_domain,
            exec_options,
            operation,
        )
        .map_err(|e| e.reanchor(operation))?
    };
    let mut sorted_key_columns = sorted_keys_table.into_columns().map_err(|e| e.reanchor(operation))?;
    let sorted_keys = sorted_key_columns.remove(0);

    Ok(SortedKeyOrder { sorted_original_codes, sorted_keys })
}

pub(crate) fn keys_impl(
    input: &ColumnView<'_>,
    indices_type: TypeId,
    key_order: KeyOrder,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let dict = dictionary::encode_impl(input, indices_type, output_domain, exec_options, operation)
        .map_err(|e| e.reanchor(operation))?;
    let dict_view = dict.view().map_err(|e| e.reanchor(operation))?;
    let row_codes = dictionary::indices(&dict_view).map_err(|e| e.reanchor(operation))?;
    let sorted_keys = dictionary::keys(&dict_view).map_err(|e| e.reanchor(operation))?;

    match key_order {
        KeyOrder::Sorted => {
            sorted_key_order(&sorted_keys, output_domain, exec_options, operation).map(|ordered| ordered.sorted_keys)
        },
        KeyOrder::FirstSeen => {
            ordered_first_seen_keys(&row_codes, &sorted_keys, output_domain, exec_options, operation)
                .map(|ordered| ordered.ordered_keys)
        },
    }
}

pub(crate) fn factorize_impl(
    input: &ColumnView<'_>,
    indices_type: TypeId,
    key_order: KeyOrder,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<FactorizationParts> {
    let dict = dictionary::encode_impl(input, indices_type, output_domain, exec_options, operation)
        .map_err(|e| e.reanchor(operation))?;
    let dict_view = dict.view().map_err(|e| e.reanchor(operation))?;
    let row_codes = dictionary::indices(&dict_view).map_err(|e| e.reanchor(operation))?;
    let sorted_keys = dictionary::keys(&dict_view).map_err(|e| e.reanchor(operation))?;

    match key_order {
        KeyOrder::Sorted => {
            let sorted = sorted_key_order(&sorted_keys, output_domain, exec_options, operation)?;
            let sorted_key_count =
                sorted.sorted_original_codes.view().and_then(|view| view.len()).map_err(|e| e.reanchor(operation))?;

            if sorted_key_count == 0 {
                return Ok(FactorizationParts {
                    row_codes: copy_view_to_owned(&row_codes).map_err(|e| e.reanchor(operation))?,
                    keys: sorted.sorted_keys,
                });
            }

            let sorted_positions = position_sequence_in_domain(sorted_key_count, output_domain, operation)
                .map_err(|e| e.reanchor(operation))?;
            let sorted_original_codes_view = sorted.sorted_original_codes.view().map_err(|e| e.reanchor(operation))?;
            let sorted_positions_view = sorted_positions.view().map_err(|e| e.reanchor(operation))?;
            let remapped_row_codes =
                lookup::probe_prevalidated(&row_codes, &sorted_original_codes_view, &sorted_positions_view, operation)
                    .map_err(|e| e.reanchor(operation))?;

            Ok(FactorizationParts { row_codes: remapped_row_codes, keys: sorted.sorted_keys })
        },
        KeyOrder::FirstSeen => {
            let ordered = ordered_first_seen_keys(&row_codes, &sorted_keys, output_domain, exec_options, operation)?;
            let ordered_code_count =
                ordered.ordered_sorted_codes.view().and_then(|view| view.len()).map_err(|e| e.reanchor(operation))?;

            if ordered_code_count == 0 {
                return Ok(FactorizationParts {
                    row_codes: copy_view_to_owned(&row_codes).map_err(|e| e.reanchor(operation))?,
                    keys: ordered.ordered_keys,
                });
            }

            let remapped_values = position_sequence_in_domain(ordered_code_count, output_domain, operation)
                .map_err(|e| e.reanchor(operation))?;
            let ordered_sorted_codes_view = ordered.ordered_sorted_codes.view().map_err(|e| e.reanchor(operation))?;
            let remapped_values_view = remapped_values.view().map_err(|e| e.reanchor(operation))?;
            let remapped_row_codes =
                lookup::probe_prevalidated(&row_codes, &ordered_sorted_codes_view, &remapped_values_view, operation)
                    .map_err(|e| e.reanchor(operation))?;

            Ok(FactorizationParts { row_codes: remapped_row_codes, keys: ordered.ordered_keys })
        },
    }
}
