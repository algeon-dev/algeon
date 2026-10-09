//! GPU-accelerated string operations.
//!
//! These functions operate on `STRING`-typed columns and return new columns.
//! All operations are null-aware: null input elements produce null output.

use std::{os::raw::c_char, ptr, ptr::NonNull, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, Result},
        native_teardown::drop_device_owned_handle,
        scalar::Scalar,
        types::DataType,
    },
    frame::series::Series,
    runtime::{
        BudgetedExecutionOutcome, ExecutionBudgetToken, execution_context::Exec, execution_domain::ExecutionDomain,
        execution_resolver::ResolvedExecution,
    },
};

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
            StripSide::Left => cudf_sys::CUDF_STRIP_LEFT,
            StripSide::Right => cudf_sys::CUDF_STRIP_RIGHT,
            StripSide::Both => cudf_sys::CUDF_STRIP_BOTH,
        }
    }
}

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

/// Stable contract label for the libcudf-owned replace-with-backrefs size preflight.
pub const REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT: &str = "regexp_replace_output_size_preflight";

/// Exact output bytes and conservative workspace preflight for string
/// replacement with back-references.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplaceWithBackrefsOutputSizePreflight {
    retained_output_bytes: usize,
    temporary_workspace_bytes: usize,
    reservation_required_bytes: usize,
}

impl ReplaceWithBackrefsOutputSizePreflight {
    /// Exact retained strings-column allocation for the output.
    pub fn retained_output_bytes(self) -> usize {
        self.retained_output_bytes
    }

    /// Conservative requested-byte workspace upper bound for size and execute.
    ///
    /// This is a requested/CUDA-utilized byte fact. It does not include the aligned
    /// reservation charge. Use [`required_budget_bytes`](Self::required_budget_bytes) for
    /// a bounded token.
    pub fn temporary_workspace_bytes(self) -> usize {
        self.temporary_workspace_bytes
    }

    /// Conservative requested/CUDA-utilized byte upper bound.
    pub fn requested_bytes_upper_bound(self) -> usize {
        self.retained_output_bytes.saturating_add(self.temporary_workspace_bytes)
    }

    /// Hard budget required before launching execute under a bounded token.
    ///
    /// This is calculated by applying the reservation allocator's actual
    /// per-allocation alignment charge and is safe to use directly; it is not
    /// the sum of aggregate requested-byte facts.
    pub fn required_budget_bytes(self) -> usize {
        self.reservation_required_bytes
    }
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

pub(crate) fn replace_with_backrefs_output_size_impl(
    input: &ColumnView<'_>,
    pattern: &str,
    replacement: &str,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<ReplaceWithBackrefsOutputSizePreflight> {
    let pattern_len = error::usize_to_i32(pattern.len(), operation, "pattern_len")?;
    let replacement_len = error::usize_to_i32(replacement.len(), operation, "replacement_len")?;
    let mut retained_output_bytes = 0usize;
    let mut temporary_workspace_bytes = 0usize;
    let mut reservation_required_bytes = 0usize;
    let status = unsafe {
        cudf_sys::cudf_strings_replace_with_backrefs_output_size_ex(
            input.raw_handle(),
            pattern.as_ptr() as *const c_char,
            pattern_len,
            replacement.as_ptr() as *const c_char,
            replacement_len,
            &mut retained_output_bytes,
            &mut temporary_workspace_bytes,
            &mut reservation_required_bytes,
            exec_options,
        )
    };
    error::check_status_in(status, operation)?;
    Ok(ReplaceWithBackrefsOutputSizePreflight {
        retained_output_bytes,
        temporary_workspace_bytes,
        reservation_required_bytes,
    })
}

/// Exact retained-size facts and a conservative execute-workspace fact from a
/// prepared replace-with-backrefs handle.
///
/// Requested-byte facts are kept distinct from the reservation-safe execute
/// bound. `required_budget_bytes` applies the allocator's actual per-allocation
/// alignment and can be used directly for an execute token after prepare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplaceWithBackrefsPreparedOutputSizeFacts {
    retained_output_bytes: usize,
    retained_state_bytes: usize,
    peak_execute_workspace_bytes: usize,
    execute_reservation_required_bytes: usize,
}

impl ReplaceWithBackrefsPreparedOutputSizeFacts {
    /// Exact retained strings-column allocation for the final output.
    pub fn retained_output_bytes(self) -> usize {
        self.retained_output_bytes
    }

    /// Exact prepared-handle allocation after the size pass.
    pub fn retained_state_bytes(self) -> usize {
        self.retained_state_bytes
    }

    /// Conservative requested-byte non-output workspace bound for execute.
    pub fn peak_execute_workspace_bytes(self) -> usize {
        self.peak_execute_workspace_bytes
    }

    /// Conservative requested/CUDA-utilized byte upper bound for execute.
    pub fn requested_execute_bytes_upper_bound(self) -> usize {
        self.retained_output_bytes.saturating_add(self.peak_execute_workspace_bytes)
    }

    /// Hard budget required before launching execute under a bounded token.
    ///
    /// Uses the exact RMM alignment supplied by each execute allocation. The
    /// prepared allocations already exist and therefore are not charged again
    /// when a token is attached only for execute.
    pub fn required_budget_bytes(self) -> usize {
        self.execute_reservation_required_bytes
    }
}

/// Owned prepared replace-with-backrefs state bound to one input series.
///
/// Execute consumes the handle and performs only the write pass. Drop releases
/// every retained allocation exactly once when the handle is unused. Execute
/// must resolve to the same CUDA stream used by prepare.
pub struct SeriesPreparedReplaceWithBackrefs {
    input: Series,
    handle: Option<NonNull<cudf_sys::cudf_prepared_replace_with_backrefs_handle_state>>,
    domain: Arc<ExecutionDomain>,
    facts: ReplaceWithBackrefsPreparedOutputSizeFacts,
}

impl SeriesPreparedReplaceWithBackrefs {
    /// Borrow the retained input series.
    pub fn input(&self) -> &Series {
        &self.input
    }

    /// Borrow prepared output-size / retained-state facts.
    pub fn output_size_facts(&self) -> ReplaceWithBackrefsPreparedOutputSizeFacts {
        self.facts
    }

    /// Consume the prepared handle and materialize the replaced strings column.
    pub fn execute(self) -> Result<Series> {
        let operation = "series::prepared_replace_with_backrefs::execute";
        let ctx = Exec::RETAINED;
        self.execute_in_context(ctx, operation, |error| error)
    }

    /// Consume the prepared handle under a hard execution-budget token.
    ///
    /// Execute allocations are routed through the token's stream-bound
    /// reservation. The token must use the same stream as prepare. The report
    /// records retained prepared-state bytes and exact retained output bytes.
    /// Execute must not re-run the size pass.
    pub fn execute_with_budget(self, token: &ExecutionBudgetToken) -> Result<BudgetedExecutionOutcome<Series>> {
        let operation = "series::prepared_replace_with_backrefs::execute_with_budget";
        let facts = self.facts;
        let ctx = token.exec();
        let output = self.execute_in_context(ctx, operation, |error| token.annotate_error(error, operation))?;
        let output_bytes =
            output.device_allocated_size_bytes().map_err(|error| token.annotate_error(error, operation))?;
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(facts.retained_state_bytes())
            .with_output_bytes(output_bytes)
            .with_output_rows(output.len())
            .with_context_facts([
                ("replace_with_backrefs_retained_output_bytes", facts.retained_output_bytes().to_string()),
                (
                    "replace_with_backrefs_peak_execute_workspace_bytes",
                    facts.peak_execute_workspace_bytes().to_string(),
                ),
                (
                    "replace_with_backrefs_requested_execute_bytes_upper_bound",
                    facts.requested_execute_bytes_upper_bound().to_string(),
                ),
                ("replace_with_backrefs_required_budget_bytes", facts.required_budget_bytes().to_string()),
            ]);
        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    fn execute_in_context(
        mut self,
        ctx: Exec<'_>,
        operation: &'static str,
        map_execution_error: impl Fn(Error) -> Error,
    ) -> Result<Series> {
        let execution = ResolvedExecution::unary(&self.domain, ctx, operation)
            .map_err(|error| map_execution_error(error.reanchor_if_needed(operation)))?;
        let cv =
            self.input.as_column().view().map_err(|error| map_execution_error(error.reanchor_if_needed(operation)))?;
        let handle = match self.handle.take() {
            Some(handle) => handle.as_ptr(),
            None => {
                return Err(map_execution_error(Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    operation,
                    "prepared replace-with-backrefs handle was already consumed",
                )));
            },
        };

        let mut out = ptr::null_mut();
        let status = unsafe {
            cudf_sys::cudf_strings_execute_replace_with_backrefs_ex(
                cv.raw_handle(),
                handle,
                &mut out,
                execution.exec_options(),
            )
        };
        // On failure the FFI already consumed the handle; do not destroy again.
        error::check_status_in(status, operation).map_err(&map_execution_error)?;
        // The write pass allocated the output on the resolved stream and
        // memory resource, which may be a budget token's own adaptor; the
        // column retains that domain so the allocation outlives the token.
        let column = unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }
            .map_err(&map_execution_error)?;
        Series::from_column(column).map_err(map_execution_error)
    }
}

impl Drop for SeriesPreparedReplaceWithBackrefs {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        drop_device_owned_handle(
            handle.as_ptr(),
            self.domain.device_id(),
            "SeriesPreparedReplaceWithBackrefs::drop",
            |owned| unsafe { cudf_sys::cudf_strings_prepared_replace_with_backrefs_destroy(owned) },
        );
    }
}

/// Prepare reusable replace-with-backrefs state after one size pass.
///
/// Retains the input [`Series`] and borrows its column for the prepared
/// handle's lifetime. Device allocations use the explicit preflight token's
/// execution memory resource when a token is supplied.
pub(crate) fn prepare_replace_with_backrefs_with_budget(
    input: &Series,
    pattern: &str,
    replacement: &str,
    token: &ExecutionBudgetToken,
) -> Result<BudgetedExecutionOutcome<SeriesPreparedReplaceWithBackrefs>> {
    let operation = "strings::prepare_replace_with_backrefs_with_budget";
    let ctx = token.exec();
    let prepared = prepare_replace_with_backrefs_in_context(input, pattern, replacement, ctx, operation)
        .map_err(|error| token.annotate_error(error, operation))?;
    let facts = prepared.output_size_facts();
    let report = token
        .reservation_enforced_report()
        .with_retained_bytes(facts.retained_state_bytes())
        .with_output_bytes(facts.retained_output_bytes())
        .with_output_rows(input.len())
        .with_context_facts([
            ("replace_with_backrefs_peak_execute_workspace_bytes", facts.peak_execute_workspace_bytes().to_string()),
            (
                "replace_with_backrefs_requested_execute_bytes_upper_bound",
                facts.requested_execute_bytes_upper_bound().to_string(),
            ),
            ("replace_with_backrefs_required_budget_bytes", facts.required_budget_bytes().to_string()),
        ]);
    Ok(BudgetedExecutionOutcome::new(prepared, report))
}

/// Prepare reusable replace-with-backrefs state using the default execution context.
pub(crate) fn prepare_replace_with_backrefs(
    input: &Series,
    pattern: &str,
    replacement: &str,
) -> Result<SeriesPreparedReplaceWithBackrefs> {
    prepare_replace_with_backrefs_in_context(
        input,
        pattern,
        replacement,
        Exec::RETAINED,
        "strings::prepare_replace_with_backrefs",
    )
}

pub(crate) fn prepare_replace_with_backrefs_in_context(
    input: &Series,
    pattern: &str,
    replacement: &str,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<SeriesPreparedReplaceWithBackrefs> {
    let execution = ResolvedExecution::unary(input.as_column().execution_domain(), ctx, operation)?;
    let cv = input.as_column().view().map_err(|error| error.reanchor_if_needed(operation))?;
    let pattern_len = error::usize_to_i32(pattern.len(), operation, "pattern_len")?;
    let replacement_len = error::usize_to_i32(replacement.len(), operation, "replacement_len")?;
    let mut handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_strings_prepare_replace_with_backrefs_ex(
            cv.raw_handle(),
            pattern.as_ptr() as *const c_char,
            pattern_len,
            replacement.as_ptr() as *const c_char,
            replacement_len,
            &mut handle,
            execution.exec_options(),
        )
    };
    error::check_status_in(status, operation)?;
    let handle = error::require_non_null(handle, operation, "prepared replace with backrefs")?;

    let mut retained_output_bytes = 0usize;
    let mut retained_state_bytes = 0usize;
    let mut peak_execute_workspace_bytes = 0usize;
    let mut execute_reservation_required_bytes = 0usize;
    let facts_status = unsafe {
        cudf_sys::cudf_strings_prepared_replace_with_backrefs_output_size_facts(
            handle.as_ptr(),
            &mut retained_output_bytes,
            &mut retained_state_bytes,
            &mut peak_execute_workspace_bytes,
            &mut execute_reservation_required_bytes,
        )
    };
    if let Err(error) = error::check_status_in(facts_status, operation) {
        drop_device_owned_handle(
            handle.as_ptr(),
            execution.output_domain().device_id(),
            "strings::prepare_replace_with_backrefs::facts_cleanup",
            |owned| unsafe { cudf_sys::cudf_strings_prepared_replace_with_backrefs_destroy(owned) },
        );
        return Err(error);
    }

    Ok(SeriesPreparedReplaceWithBackrefs {
        input: input.clone(),
        handle: Some(handle),
        domain: Arc::clone(execution.output_domain()),
        facts: ReplaceWithBackrefsPreparedOutputSizeFacts {
            retained_output_bytes,
            retained_state_bytes,
            peak_execute_workspace_bytes,
            execute_reservation_required_bytes,
        },
    })
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

pub(crate) fn concatenate_impl(inputs: &[&ColumnView<'_>], operation: &'static str) -> Result<Column> {
    if inputs.len() < 2 {
        return Err(error::Error::new_with(
            error::ErrorCode::EmptyInput,
            operation,
            "row-wise string concatenation requires at least two input columns",
        ));
    }
    let num_columns = error::usize_to_i32(inputs.len(), operation, "num_columns")?;
    let ctx = Exec::RETAINED;
    let execution = ResolvedExecution::many(
        inputs[0].execution_domain(),
        inputs.iter().skip(1).map(|input| input.execution_domain()),
        ctx,
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

pub(crate) fn from_integers_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_strings_from_integers_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in(status, operation)?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        QueryMemoryDomainOptions,
        runtime::{ExecutionBudgetRequest, ExecutionBudgetToken, Runtime},
    };

    #[test]
    fn budgeted_replace_with_backrefs_output_carries_the_execution_domain() {
        let runtime = Runtime::new().expect("test runtime");
        let input = runtime.series(&["abc-123", "def-456"]).expect("input series");
        let (pattern, replacement) = (r"([a-z]+)-(\d+)", r"\2/\1");
        let facts =
            input.on(Exec::RETAINED).str().replace_with_backrefs_output_size(pattern, replacement).expect("size pass");
        let domain = runtime
            .device_memory_manager()
            .expect("memory manager")
            .query_memory_domain(1, QueryMemoryDomainOptions::new(facts.required_budget_bytes()))
            .expect("query domain");
        let lease = domain.reserve_stream(runtime.stream(), 1, facts.required_budget_bytes()).expect("member");
        let token = ExecutionBudgetToken::from_shared_lease(
            &runtime,
            ExecutionBudgetRequest::new(facts.required_budget_bytes()).with_boundary("replace_backrefs_domain"),
            lease,
        );
        let execution_domain =
            token.exec().resolve_output_domain(input.as_column().execution_domain()).expect("token output domain");
        assert!(
            !ExecutionDomain::same_provenance(&execution_domain, input.as_column().execution_domain()),
            "the token must allocate through its own memory resource"
        );

        let (prepared, _) = input
            .prepare_replace_with_backrefs_with_budget(pattern, replacement, &token)
            .expect("prepare")
            .into_parts();
        let (output, _) = prepared.execute_with_budget(&token).expect("execute").into_parts();

        assert!(
            ExecutionDomain::same_provenance(output.as_column().execution_domain(), &execution_domain),
            "the output must carry the domain that allocated it"
        );
        assert_eq!(output.to_strings_strict().expect("output values"), vec!["123/abc", "456/def"]);
    }
}
