use crate::core::{
    column::ColumnView,
    error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_TYPE_ID, Result},
    types::TypeId,
};

pub(super) fn unsupported_dictionary_validation_error(
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new_with(ErrorCode::UnsupportedOperation, operation, message)
        .with_detail("reason", reason)
        .with_detail("remedy", remedy)
}

pub(super) fn foreign_dictionary_match_result_shape_error(
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
) -> Error {
    error::foreign_invariant_violation(
        operation,
        message,
        "dictionary_match_result_shape",
        reason,
        "ensure the foreign dictionary match backend returns one output column per input dictionary column",
    )
}

pub(super) fn dictionary_column_count_limit_error(operation: &'static str, table_column_count: usize) -> Error {
    error::limit_validation_error(
        operation,
        "dictionary column count exceeds i32::MAX",
        "table_column_count",
        "table_column_count",
        "i32::MAX",
        table_column_count,
        "reduce table_column_count to fit within i32::MAX",
    )
}

fn dictionary_type_mismatch_error(
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    type_id: TypeId,
) -> Error {
    Error::new_with(ErrorCode::TypeMismatch, operation, message)
        .with_detail("resource", resource)
        .with_fact(FACT_TYPE_ID, type_id)
}

pub(super) fn require_dictionary_column(
    column: &ColumnView<'_>,
    operation: &'static str,
    resource: &'static str,
    column_index: Option<usize>,
) -> Result<()> {
    let type_id = column.data_type().map_err(|e| e.reanchor(operation))?.id();
    if type_id == TypeId::Dictionary32 {
        return Ok(());
    }

    let mut err = dictionary_type_mismatch_error(
        operation,
        "dictionary operation requires dictionary-encoded input columns",
        resource,
        type_id,
    );
    if let Some(column_index) = column_index {
        err = err.with_fact(FACT_COLUMN_INDEX, column_index);
    }
    Err(err)
}

pub(super) fn validate_set_keys_inputs(
    dict_column: &ColumnView<'_>,
    keys_column: &ColumnView<'_>,
    operation: &'static str,
) -> Result<()> {
    require_dictionary_column(dict_column, operation, "dict_column", None)?;

    let dict_keys = super::keys(dict_column).map_err(|e| e.reanchor(operation))?;
    let left_type = dict_keys.data_type().map_err(|e| e.reanchor(operation))?.id();
    let right_type = keys_column.data_type().map_err(|e| e.reanchor(operation))?.id();
    if left_type != right_type {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "dictionary keys column type must match the dictionary key type",
        )
        .with_fact(FACT_LEFT_TYPE_ID, left_type)
        .with_fact(FACT_RIGHT_TYPE_ID, right_type));
    }

    let null_count = keys_column.null_count().map_err(|e| e.reanchor(operation))?;
    if null_count > 0 {
        return Err(Error::new_with(ErrorCode::NullUnexpected, operation, "dictionary keys must not contain nulls")
            .with_detail("resource", "keys"));
    }

    Ok(())
}
