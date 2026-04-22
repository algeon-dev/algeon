use crate::core::{
    column::ColumnView,
    error::{self, Error, ErrorCode, ErrorKind, Result},
    types::TypeId,
};

fn type_label(type_id: TypeId) -> String {
    format!("{type_id:?}").to_lowercase()
}

pub(super) fn unsupported_dictionary_validation_error(
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context("reason", reason)
        .with_context("remedy", remedy)
}

pub(super) fn foreign_dictionary_match_result_shape_error(
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
) -> Error {
    Error::new(ErrorKind::InvariantViolation, message)
        .with_code(ErrorCode::InvariantViolation)
        .with_operation(operation)
        .with_context("resource", "dictionary_match_result_shape")
        .with_context("reason", reason)
        .with_context(
            "remedy",
            "ensure the foreign dictionary match backend returns one output column per input dictionary column",
        )
}

pub(super) fn dictionary_column_count_limit_error(operation: &'static str, table_column_count: usize) -> Error {
    error::limit_validation_error(
        ErrorKind::InvalidInput,
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
    Error::new(ErrorKind::InvalidInput, message)
        .with_code(ErrorCode::TypeMismatch)
        .with_operation(operation)
        .with_context("resource", resource)
        .with_context("type_id", type_label(type_id))
}

pub(super) fn require_dictionary_column(
    column: &ColumnView<'_>,
    operation: &'static str,
    resource: &'static str,
    column_index: Option<usize>,
) -> Result<()> {
    let type_id = column.data_type().map_err(|e| e.with_operation(operation))?.id();
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
        err = err.with_context("column_index", column_index.to_string());
    }
    Err(err)
}

pub(super) fn validate_set_keys_inputs(
    dict_column: &ColumnView<'_>,
    keys_column: &ColumnView<'_>,
    operation: &'static str,
) -> Result<()> {
    require_dictionary_column(dict_column, operation, "dict_column", None)?;

    let dict_keys = super::keys(dict_column).map_err(|e| e.with_operation(operation))?;
    let left_type = dict_keys.data_type().map_err(|e| e.with_operation(operation))?.id();
    let right_type = keys_column.data_type().map_err(|e| e.with_operation(operation))?.id();
    if left_type != right_type {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "dictionary keys column type must match the dictionary key type",
        )
        .with_code(ErrorCode::TypeMismatch)
        .with_operation(operation)
        .with_context("left_type_id", type_label(left_type))
        .with_context("right_type_id", type_label(right_type)));
    }

    let null_count = keys_column.null_count().map_err(|e| e.with_operation(operation))?;
    if null_count > 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "dictionary keys must not contain nulls")
            .with_code(ErrorCode::NullUnexpected)
            .with_operation(operation)
            .with_context("resource", "keys"));
    }

    Ok(())
}
