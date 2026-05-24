//! GPU-accelerated string operations.
//!
//! These functions operate on `STRING`-typed columns and return new columns.
//! All operations are null-aware: null input elements produce null output.

use std::{os::raw::c_char, ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        scalar::Scalar,
        types::DataType,
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
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
// String slice options
// ---------------------------------------------------------------------------

/// Fixed-position string slicing options.
///
/// `start` and `stop` use libcudf's zero-based character positions and the
/// half-open range `[start, stop)`. `None` leaves the corresponding bound open.
/// `step` is the character stride and must be non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringSliceOptions {
    /// First character position to include, or open start when `None`.
    pub start: Option<i32>,
    /// First character position to exclude, or open end when `None`.
    pub stop: Option<i32>,
    /// Character stride. Must not be zero.
    pub step: i32,
}

impl StringSliceOptions {
    /// Create string slice options with a default `step` of `1`.
    pub fn new(start: Option<i32>, stop: Option<i32>) -> Self {
        Self { start, stop, step: 1 }
    }

    /// Set the character stride.
    pub fn with_step(mut self, step: i32) -> Self {
        self.step = step;
        self
    }
}

impl Default for StringSliceOptions {
    fn default() -> Self {
        Self::new(None, None)
    }
}

// ---------------------------------------------------------------------------
// Case conversion
// ---------------------------------------------------------------------------

/// Convert each string element to lowercase. Returns a new STRING column.
pub fn to_lower(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "strings::to_lower")?;
    to_lower_impl(input, execution.output_domain(), execution.exec_options(), "strings::to_lower")
}

pub(crate) fn to_lower_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_to_lower_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Convert each string element to uppercase. Returns a new STRING column.
pub fn to_upper(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "strings::to_upper")?;
    to_upper_impl(input, execution.output_domain(), execution.exec_options(), "strings::to_upper")
}

pub(crate) fn to_upper_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_to_upper_ex(input.raw_handle(), &mut out, exec_options) };
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
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(input.execution_domain(), target.execution_domain(), &ctx, "strings::contains")?;
    str_contains_impl(input, target, execution.output_domain(), execution.exec_options(), "strings::contains")
}

pub(crate) fn str_contains_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_strings_contains_ex(input.raw_handle(), target.raw_handle(), &mut out, exec_options) };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// SQL `LIKE` pattern match against a literal pattern. Returns a BOOL8 column.
///
/// The pattern uses SQL wildcard semantics: `%` matches zero or more
/// characters and `_` matches one character. `escape` may specify zero or one
/// ASCII escape character for literal wildcard matching. This is not a regex
/// API.
///
/// # Synchronization
/// The C shim synchronizes the selected stream before returning so libcudf no
/// longer borrows the host pattern or escape bytes.
pub fn like(input: &ColumnView<'_>, pattern: &str, escape: Option<&str>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "strings::like")?;
    like_impl(input, pattern, escape, execution.output_domain(), execution.exec_options(), "strings::like")
}

pub(crate) fn like_impl(
    input: &ColumnView<'_>,
    pattern: &str,
    escape: Option<&str>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let pattern_len = error::usize_to_i32(pattern.len(), operation, "pattern_len")?;
    let escape = escape.unwrap_or("");
    let escape_len = error::usize_to_i32(escape.len(), operation, "escape_len")?;
    let escape_ptr = if escape.is_empty() { ptr::null() } else { escape.as_ptr() as *const c_char };
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_like_ex(
            input.raw_handle(),
            pattern.as_ptr() as *const c_char,
            pattern_len,
            escape_ptr,
            escape_len,
            &mut out,
            exec_options,
        )
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Regex replacement using a back-reference replacement template.
///
/// The pattern uses libcudf regex syntax. The replacement template may include
/// back-reference markers such as `\1`.
///
/// # Synchronization
/// The C shim synchronizes the selected stream before returning so libcudf no
/// longer borrows the host pattern or replacement bytes.
pub fn replace_with_backrefs(input: &ColumnView<'_>, pattern: &str, replacement: &str) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "strings::replace_with_backrefs")?;
    replace_with_backrefs_impl(
        input,
        pattern,
        replacement,
        execution.output_domain(),
        execution.exec_options(),
        "strings::replace_with_backrefs",
    )
}

pub(crate) fn replace_with_backrefs_impl(
    input: &ColumnView<'_>,
    pattern: &str,
    replacement: &str,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let pattern_len = error::usize_to_i32(pattern.len(), operation, "pattern_len")?;
    let replacement_len = error::usize_to_i32(replacement.len(), operation, "replacement_len")?;
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_replace_with_backrefs_ex(
            input.raw_handle(),
            pattern.as_ptr() as *const c_char,
            pattern_len,
            replacement.as_ptr() as *const c_char,
            replacement_len,
            &mut out,
            exec_options,
        )
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Element-wise starts-with test against a scalar string. Returns a BOOL8 column.
pub fn starts_with(input: &ColumnView<'_>, target: &Scalar) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(input.execution_domain(), target.execution_domain(), &ctx, "strings::starts_with")?;
    starts_with_impl(input, target, execution.output_domain(), execution.exec_options(), "strings::starts_with")
}

pub(crate) fn starts_with_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_starts_with_ex(input.raw_handle(), target.raw_handle(), &mut out, exec_options)
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Element-wise ends-with test against a scalar string. Returns a BOOL8 column.
pub fn ends_with(input: &ColumnView<'_>, target: &Scalar) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(input.execution_domain(), target.execution_domain(), &ctx, "strings::ends_with")?;
    ends_with_impl(input, target, execution.output_domain(), execution.exec_options(), "strings::ends_with")
}

pub(crate) fn ends_with_impl(
    input: &ColumnView<'_>,
    target: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_strings_ends_with_ex(input.raw_handle(), target.raw_handle(), &mut out, exec_options) };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Slice each string using zero-based character positions. Returns a STRING column.
///
/// Bounds use libcudf's `[start, stop)` character-position semantics. This
/// allocates a new GPU string column and preserves nulls.
pub fn slice(input: &ColumnView<'_>, options: StringSliceOptions) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "strings::slice")?;
    slice_impl(input, options, execution.output_domain(), execution.exec_options(), "strings::slice")
}

pub(crate) fn slice_impl(
    input: &ColumnView<'_>,
    options: StringSliceOptions,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let (start, start_valid) = match options.start {
        Some(start) => (start, 1),
        None => (0, 0),
    };
    let (stop, stop_valid) = match options.stop {
        Some(stop) => (stop, 1),
        None => (0, 0),
    };
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_slice_ex(
            input.raw_handle(),
            start,
            start_valid,
            stop,
            stop_valid,
            options.step,
            &mut out,
            exec_options,
        )
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

// ---------------------------------------------------------------------------
// Concatenate
// ---------------------------------------------------------------------------

/// Row-wise concatenate string columns with an empty separator.
///
/// Null input values are treated as empty strings, matching DataFusion SQL
/// `concat` semantics. At least two input columns are required.
pub fn concatenate(inputs: &[&ColumnView<'_>]) -> Result<Column> {
    concatenate_impl(inputs, "strings::concatenate")
}

pub(crate) fn concatenate_impl(inputs: &[&ColumnView<'_>], operation: &'static str) -> Result<Column> {
    if inputs.len() < 2 {
        return Err(error::Error::new_with(
            error::ErrorCode::EmptyInput,
            operation,
            "row-wise string concatenation requires at least two input columns",
        ));
    }
    let num_columns = error::usize_to_i32(inputs.len(), operation, "num_columns")?;
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::many(
        inputs[0].execution_domain(),
        inputs.iter().skip(1).map(|input| input.execution_domain()),
        &ctx,
        operation,
    )?;
    let handles = inputs.iter().map(|input| input.raw_handle()).collect::<Vec<_>>();
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_concatenate_ex(handles.as_ptr(), num_columns, &mut out, execution.exec_options())
    };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }
}

// ---------------------------------------------------------------------------
// Strip / trim
// ---------------------------------------------------------------------------

/// Strip characters from one or both sides of each string.
///
/// `to_strip` specifies the set of characters to remove. Pass a valid empty
/// string scalar to strip whitespace.
pub fn strip(input: &ColumnView<'_>, side: StripSide, to_strip: &Scalar) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(input.execution_domain(), to_strip.execution_domain(), &ctx, "strings::strip")?;
    strip_impl(input, side, to_strip, execution.output_domain(), execution.exec_options(), "strings::strip")
}

pub(crate) fn strip_impl(
    input: &ColumnView<'_>,
    side: StripSide,
    to_strip: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let side_ffi = side.to_raw();
    let status = unsafe {
        cudf_sys::cudf_strings_strip_ex(input.raw_handle(), side_ffi, to_strip.raw_handle(), &mut out, exec_options)
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
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "strings::count_characters")?;
    count_characters_impl(input, execution.output_domain(), execution.exec_options(), "strings::count_characters")
}

pub(crate) fn count_characters_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_count_characters_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Byte count per string. Returns an INT32 column.
pub fn count_bytes(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "strings::count_bytes")?;
    count_bytes_impl(input, execution.output_domain(), execution.exec_options(), "strings::count_bytes")
}

pub(crate) fn count_bytes_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_count_bytes_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

// ---------------------------------------------------------------------------
// Numeric conversion (null-on-failure / TRY_CAST contract)
// ---------------------------------------------------------------------------

/// Parse each string element as an integer of `output_type`, producing null
/// rows for any element that does not parse cleanly or whose value would
/// overflow the target type. Null inputs stay null. Mirrors SQL `TRY_CAST`
/// semantics for `STRING → INTEGER`.
///
/// `output_type` must be an integral type. Boolean is not accepted.
pub fn try_to_integers(input: &ColumnView<'_>, output_type: &DataType) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "strings::try_to_integers")?;
    try_to_integers_impl(
        input,
        output_type,
        execution.output_domain(),
        execution.exec_options(),
        "strings::try_to_integers",
    )
}

pub(crate) fn try_to_integers_impl(
    input: &ColumnView<'_>,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle_for(operation)?;
    let mut out = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_strings_try_to_integers_ex(input.raw_handle(), dtype_handle, &mut out, exec_options) };
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
