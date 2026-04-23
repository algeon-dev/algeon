use crate::core::error::{self, Error};

pub(super) fn foreign_column_view_metadata_contract_error(
    operation: &'static str,
    message: impl Into<String>,
) -> Error {
    error::foreign_invariant_violation(
        operation,
        message,
        "column_view_metadata",
        "invalid_foreign_column_view_metadata",
        "ensure the foreign column view exposes valid dtype, size, null-count, child-count, and offset metadata",
    )
}

pub(super) fn classify_column_view_metadata_error(err: Error) -> Error {
    error::annotate_foreign_invariant_contract_if_absent(
        err,
        "column_view_metadata",
        "invalid_foreign_column_view_metadata",
        "ensure the foreign column view exposes valid dtype, size, null-count, child-count, and offset metadata",
    )
}

pub(super) fn foreign_string_view_contract_error(operation: &'static str, message: impl Into<String>) -> Error {
    error::foreign_invariant_violation(
        operation,
        message,
        "string_view_layout",
        "invalid_string_view_metadata",
        "ensure the STRING column view exposes valid offsets and chars metadata",
    )
}

pub(super) fn classify_string_view_contract_error(err: Error) -> Error {
    error::annotate_foreign_invariant_contract_if_absent(
        err,
        "string_view_layout",
        "invalid_string_view_metadata",
        "ensure the STRING column view exposes valid offsets and chars metadata",
    )
}
