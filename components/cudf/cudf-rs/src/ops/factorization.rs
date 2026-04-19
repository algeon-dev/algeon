//! Internal factorization operators shared by high-level Series/DataFrame APIs.

use std::sync::Arc;

use crate::{
    core::{
        column::{Column, ColumnView, copy_view_to_owned},
        error::{Error, ErrorKind, Result},
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

fn empty_code_column(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<Column> {
    let input_dtype = input.data_type().map_err(|e| e.with_operation(operation))?;
    if input_dtype.id() != TypeId::Int32 {
        return Err(Error::new(ErrorKind::InvalidOperation, "factorization row-code dtype must be Int32")
            .with_operation(operation)
            .with_context("row_code_type_id", format!("{:?}", input_dtype.id()).to_lowercase()));
    }

    Column::empty_in_domain(TypeId::Int32, Arc::clone(output_domain), operation)
        .map_err(|e| e.with_operation(operation))
}

fn ordered_first_seen_keys(
    row_codes: &ColumnView<'_>,
    sorted_keys: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<FirstSeenKeyOrder> {
    if row_codes.len().map_err(|e| e.with_operation(operation))? == 0 {
        return Ok(FirstSeenKeyOrder {
            ordered_sorted_codes: empty_code_column(row_codes, output_domain, operation)?,
            ordered_keys: copy_view_to_owned(sorted_keys).map_err(|e| e.with_operation(operation))?,
        });
    }

    let row_idx = Column::from_sequence_in_domain(
        0i32,
        1i32,
        row_codes.len().map_err(|e| e.with_operation(operation))?,
        Arc::clone(output_domain),
        operation,
    )
    .map_err(|e| e.with_operation(operation))?;
    let row_idx_view = row_idx.view().map_err(|e| e.with_operation(operation))?;
    let code_and_row =
        TableView::from_column_views(&[row_codes, &row_idx_view]).map_err(|e| e.with_operation(operation))?;

    let non_null_code_rows =
        stream_compaction::drop_nulls_impl(&code_and_row, &[0], 1, output_domain, exec_options, operation)
            .map_err(|e| e.with_operation(operation))?;
    let non_null_code_rows_view = non_null_code_rows.view().map_err(|e| e.with_operation(operation))?;
    if non_null_code_rows_view.num_rows().map_err(|e| e.with_operation(operation))? == 0 {
        return Ok(FirstSeenKeyOrder {
            ordered_sorted_codes: empty_code_column(row_codes, output_domain, operation)?,
            ordered_keys: copy_view_to_owned(sorted_keys).map_err(|e| e.with_operation(operation))?,
        });
    }

    let first_code_rows = stream_compaction::distinct_impl(
        &non_null_code_rows_view,
        &[0],
        DuplicateKeepOption::KeepFirst,
        NullEquality::Equal,
        NanEquality::AllEqual,
        output_domain,
        exec_options,
        operation,
    )
    .map_err(|e| e.with_operation(operation))?;
    let first_code_rows_view = first_code_rows.view().map_err(|e| e.with_operation(operation))?;
    let first_row_idx = first_code_rows_view.select_columns(&[1]).map_err(|e| e.with_operation(operation))?;
    let ordered_code_rows = sorting::stable_sort_by_key(
        &first_code_rows_view,
        &first_row_idx,
        Some(&[Order::Ascending]),
        Some(&[NullOrder::After]),
    )
    .map_err(|e| e.with_operation(operation))?;

    let mut ordered_columns = ordered_code_rows.into_columns().map_err(|e| e.with_operation(operation))?;
    let ordered_sorted_codes = ordered_columns.remove(0);
    let ordered_keys = {
        let ordered_sorted_codes_view = ordered_sorted_codes.view().map_err(|e| e.with_operation(operation))?;
        let sorted_keys_table =
            TableView::from_column_views(&[sorted_keys]).map_err(|e| e.with_operation(operation))?;
        let ordered_keys_table =
            copying::gather(&sorted_keys_table, &ordered_sorted_codes_view).map_err(|e| e.with_operation(operation))?;
        let mut ordered_key_columns = ordered_keys_table.into_columns().map_err(|e| e.with_operation(operation))?;
        ordered_key_columns.remove(0)
    };

    Ok(FirstSeenKeyOrder { ordered_sorted_codes, ordered_keys })
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
        .map_err(|e| e.with_operation(operation))?;
    let dict_view = dict.view().map_err(|e| e.with_operation(operation))?;
    let row_codes = dictionary::indices(&dict_view).map_err(|e| e.with_operation(operation))?;
    let sorted_keys = dictionary::keys(&dict_view).map_err(|e| e.with_operation(operation))?;

    match key_order {
        KeyOrder::Sorted => copy_view_to_owned(&sorted_keys).map_err(|e| e.with_operation(operation)),
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
        .map_err(|e| e.with_operation(operation))?;
    let dict_view = dict.view().map_err(|e| e.with_operation(operation))?;
    let row_codes = dictionary::indices(&dict_view).map_err(|e| e.with_operation(operation))?;
    let sorted_keys = dictionary::keys(&dict_view).map_err(|e| e.with_operation(operation))?;

    match key_order {
        KeyOrder::Sorted => Ok(FactorizationParts {
            row_codes: copy_view_to_owned(&row_codes).map_err(|e| e.with_operation(operation))?,
            keys: copy_view_to_owned(&sorted_keys).map_err(|e| e.with_operation(operation))?,
        }),
        KeyOrder::FirstSeen => {
            let ordered = ordered_first_seen_keys(&row_codes, &sorted_keys, output_domain, exec_options, operation)?;
            let ordered_code_count = ordered
                .ordered_sorted_codes
                .view()
                .and_then(|view| view.len())
                .map_err(|e| e.with_operation(operation))?;

            if ordered_code_count == 0 {
                return Ok(FactorizationParts {
                    row_codes: copy_view_to_owned(&row_codes).map_err(|e| e.with_operation(operation))?,
                    keys: ordered.ordered_keys,
                });
            }

            let remapped_values =
                Column::from_sequence_in_domain(0i32, 1i32, ordered_code_count, Arc::clone(output_domain), operation)
                    .map_err(|e| e.with_operation(operation))?;
            let ordered_sorted_codes_view =
                ordered.ordered_sorted_codes.view().map_err(|e| e.with_operation(operation))?;
            let remapped_values_view = remapped_values.view().map_err(|e| e.with_operation(operation))?;
            let remapped_row_codes =
                lookup::probe_prevalidated(&row_codes, &ordered_sorted_codes_view, &remapped_values_view, operation)
                    .map_err(|e| e.with_operation(operation))?;

            Ok(FactorizationParts { row_codes: remapped_row_codes, keys: ordered.ordered_keys })
        },
    }
}
