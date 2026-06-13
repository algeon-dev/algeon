//! GPU-accelerated string operations.
//!
//! These functions operate on `STRING`-typed columns and return new columns.
//! All operations are null-aware: null input elements produce null output.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        scalar::Scalar,
    },
    runtime::execution_domain::ExecutionDomain,
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

impl StripSide {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_strip_side_t {
        match self {
            StripSide::Left => cudf_sys::cudf_strip_side_t::CUDF_STRIP_LEFT,
            StripSide::Right => cudf_sys::cudf_strip_side_t::CUDF_STRIP_RIGHT,
            StripSide::Both => cudf_sys::cudf_strip_side_t::CUDF_STRIP_BOTH,
        }
    }
}

// ---------------------------------------------------------------------------
// Case conversion
// ---------------------------------------------------------------------------

/// Convert each string element to lowercase. Returns a new STRING column.
pub fn to_lower(input: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    to_lower_impl(input, &output_domain, exec_options.as_ref(), "strings::to_lower")
}

pub(crate) fn to_lower_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_strings_to_lower_ex(input.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_strings_to_lower(input.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Convert each string element to uppercase. Returns a new STRING column.
pub fn to_upper(input: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    to_upper_impl(input, &output_domain, exec_options.as_ref(), "strings::to_upper")
}

pub(crate) fn to_upper_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_strings_to_upper_ex(input.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_strings_to_upper(input.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

// ---------------------------------------------------------------------------
// Find / contains / starts_with / ends_with
// ---------------------------------------------------------------------------

/// Element-wise literal substring containment test. Returns a BOOL8 column.
///
/// `target` must be a valid (non-null) string scalar.
pub fn str_contains(input: &ColumnView<'_>, target: &Scalar) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    ExecutionDomain::insert_stream_dependency(target.execution_domain(), &output_domain, "strings::contains")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    str_contains_impl(input, target, &output_domain, exec_options.as_ref(), "strings::contains")
}

pub(crate) fn str_contains_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
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
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Element-wise starts-with test against a scalar string. Returns a BOOL8 column.
pub fn starts_with(input: &ColumnView<'_>, target: &Scalar) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    ExecutionDomain::insert_stream_dependency(target.execution_domain(), &output_domain, "strings::starts_with")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    starts_with_impl(input, target, &output_domain, exec_options.as_ref(), "strings::starts_with")
}

pub(crate) fn starts_with_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
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
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Element-wise ends-with test against a scalar string. Returns a BOOL8 column.
pub fn ends_with(input: &ColumnView<'_>, target: &Scalar) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    ExecutionDomain::insert_stream_dependency(target.execution_domain(), &output_domain, "strings::ends_with")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    ends_with_impl(input, target, &output_domain, exec_options.as_ref(), "strings::ends_with")
}

pub(crate) fn ends_with_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
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
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
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
    ExecutionDomain::insert_stream_dependency(to_strip.execution_domain(), &output_domain, "strings::strip")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    strip_impl(input, side, to_strip, &output_domain, exec_options.as_ref(), "strings::strip")
}

pub(crate) fn strip_impl(
    input: &ColumnView<'_>,
    side: StripSide,
    to_strip: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let side_ffi = side.to_raw();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_strings_strip_ex(input.raw_handle(), side_ffi, to_strip.raw_handle(), &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_strings_strip(input.raw_handle(), side_ffi, to_strip.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
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
    count_characters_impl(input, &output_domain, exec_options.as_ref(), "strings::count_characters")
}

pub(crate) fn count_characters_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_strings_count_characters_ex(input.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_strings_count_characters(input.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Byte count per string. Returns an INT32 column.
pub fn count_bytes(input: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    count_bytes_impl(input, &output_domain, exec_options.as_ref(), "strings::count_bytes")
}

pub(crate) fn count_bytes_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_strings_count_bytes_ex(input.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_strings_count_bytes(input.raw_handle(), &mut out) },
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
