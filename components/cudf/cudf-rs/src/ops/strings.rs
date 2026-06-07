//! GPU-accelerated string operations.
//!
//! These functions operate on `STRING`-typed columns and return new columns.
//! All operations are null-aware: null input elements produce null output.

use std::{ptr, sync::Arc};

use crate::core::{
    column::{Column, ColumnView},
    error::{self, Result},
    scalar::Scalar,
};

// ---------------------------------------------------------------------------
// Strip side type
// ---------------------------------------------------------------------------

/// Which side(s) of each string to strip characters from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum StripSide {
    /// Strip from the beginning (left) only.
    Left = 0,
    /// Strip from the end (right) only.
    Right = 1,
    /// Strip from both sides.
    Both = 2,
}

// ---------------------------------------------------------------------------
// Case conversion
// ---------------------------------------------------------------------------

/// Convert each string element to lowercase. Returns a new STRING column.
pub fn to_lower(input: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    to_lower_impl(input, exec_options.as_ref(), "strings::to_lower")
        .map(|column| column.with_execution_domain(output_domain))
}

fn to_lower_impl(
    input: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_strings_to_lower_ex(input.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_strings_to_lower(input.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Convert each string element to uppercase. Returns a new STRING column.
pub fn to_upper(input: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    to_upper_impl(input, exec_options.as_ref(), "strings::to_upper")
        .map(|column| column.with_execution_domain(output_domain))
}

fn to_upper_impl(
    input: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_strings_to_upper_ex(input.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_strings_to_upper(input.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    Ok(unsafe { Column::from_raw(out) })
}

// ---------------------------------------------------------------------------
// Find / contains / starts_with / ends_with
// ---------------------------------------------------------------------------

/// Element-wise literal substring containment test. Returns a BOOL8 column.
///
/// `target` must be a valid (non-null) string scalar.
pub fn str_contains(input: &ColumnView<'_>, target: &Scalar) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    str_contains_impl(input, target, exec_options.as_ref(), "strings::contains")
        .map(|column| column.with_execution_domain(output_domain))
}

fn str_contains_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_strings_contains_ex(input.raw_handle(), target.raw_handle(), &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_strings_contains(input.raw_handle(), target.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Element-wise starts-with test against a scalar string. Returns a BOOL8 column.
pub fn starts_with(input: &ColumnView<'_>, target: &Scalar) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    starts_with_impl(input, target, exec_options.as_ref(), "strings::starts_with")
        .map(|column| column.with_execution_domain(output_domain))
}

fn starts_with_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_strings_starts_with_ex(input.raw_handle(), target.raw_handle(), &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_strings_starts_with(input.raw_handle(), target.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Element-wise ends-with test against a scalar string. Returns a BOOL8 column.
pub fn ends_with(input: &ColumnView<'_>, target: &Scalar) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    ends_with_impl(input, target, exec_options.as_ref(), "strings::ends_with")
        .map(|column| column.with_execution_domain(output_domain))
}

fn ends_with_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_strings_ends_with_ex(input.raw_handle(), target.raw_handle(), &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_strings_ends_with(input.raw_handle(), target.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    Ok(unsafe { Column::from_raw(out) })
}

// ---------------------------------------------------------------------------
// Strip / trim
// ---------------------------------------------------------------------------

/// Strip characters from one or both sides of each string.
///
/// `to_strip` specifies the set of characters to remove. Pass a valid empty
/// string scalar to strip whitespace.
pub fn strip(input: &ColumnView<'_>, side: StripSide, to_strip: &Scalar) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    strip_impl(input, side, to_strip, exec_options.as_ref(), "strings::strip")
        .map(|column| column.with_execution_domain(output_domain))
}

fn strip_impl(
    input: &ColumnView<'_>,
    side: StripSide,
    to_strip: &Scalar,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    // Safety: StripSide and cudf_sys::cudf_strip_side_t are both #[repr(i32)]
    // with identical discriminant values.
    let side_ffi: cudf_sys::cudf_strip_side_t = unsafe { std::mem::transmute(side) };
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_strings_strip_ex(input.raw_handle(), side_ffi, to_strip.raw_handle(), &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_strings_strip(input.raw_handle(), side_ffi, to_strip.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    Ok(unsafe { Column::from_raw(out) })
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

/// Character count (not byte count) per string. Returns an INT32 column.
///
/// Equivalent to cuDF Python's `Series.str.len()`.
pub fn count_characters(input: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    count_characters_impl(input, exec_options.as_ref(), "strings::count_characters")
        .map(|column| column.with_execution_domain(output_domain))
}

fn count_characters_impl(
    input: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_strings_count_characters_ex(input.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_strings_count_characters(input.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Byte count per string. Returns an INT32 column.
pub fn count_bytes(input: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    count_bytes_impl(input, exec_options.as_ref(), "strings::count_bytes")
        .map(|column| column.with_execution_domain(output_domain))
}

fn count_bytes_impl(
    input: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_strings_count_bytes_ex(input.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_strings_count_bytes(input.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    Ok(unsafe { Column::from_raw(out) })
}
