//! GPU-accelerated row gathering (take / index-select), scatter, and conditional copy operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_ACTUAL_ROWS, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID,
            FACT_RIGHT_TYPE_ID, FACT_TABLE_ROW_COUNT, Result,
        },
        scalar::Scalar,
        table::{Table, TableView},
        types::TypeId,
    },
    internal::hash::{FastHashSet, fast_hash_set_with_capacity},
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

pub(crate) fn gather_impl(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    bounds_policy: cudf_sys::cudf_out_of_bounds_policy_t,
    bounds_policy_label: &'static str,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Table> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_gather_ex(input.raw_handle(), gather_map.raw_handle(), bounds_policy, &mut out, exec_options)
    };
    error::check_status_in_with_details(
        status,
        operation,
        [("bounds_policy", bounds_policy_label.to_owned()), ("execution_authority", "retained_domain".to_string())],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn gather_device_allocated_size_impl(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    bounds_policy: cudf_sys::cudf_out_of_bounds_policy_t,
    bounds_policy_label: &'static str,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
    reservation_contract: &'static str,
) -> Result<usize> {
    let mut out = 0usize;
    let status = unsafe {
        cudf_sys::cudf_gather_device_allocated_size_ex(
            input.raw_handle(),
            gather_map.raw_handle(),
            bounds_policy,
            &mut out,
            exec_options,
        )
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("bounds_policy", bounds_policy_label.to_owned()),
            ("execution_authority", "retained_domain".to_string()),
            ("reservation_contract", reservation_contract.to_owned()),
        ],
    )?;
    Ok(out)
}

/// Gather (select) rows from a table by index.
///
/// Produces a new table whose row `i` is `source_table[gather_map[i]]`.
/// The `gather_map` must be a non-nullable column of `INT32` indices.
///
/// A negative value `i` in the gather map is interpreted as `i + n`,
/// where `n` is the number of rows in the source table.
///
/// Out-of-bounds indices produce null rows in the output.
///
/// # Example
/// ```no_run
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "value" => rt.series(&[10i32, 20, 30]).unwrap(),
/// }
/// .unwrap();
/// let map = rt.series(&[2i32, 0]).unwrap();
/// let result = df.gather(&map).unwrap();
/// assert_eq!(
///     result.series("value").unwrap().to_vec_strict::<i32>().unwrap(),
///     vec![30, 10]
/// );
/// ```
pub fn gather(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(input.execution_domain(), gather_map.execution_domain(), &ctx, "copying::gather")?;
    gather_impl(
        input,
        gather_map,
        cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
        "nullify",
        execution.output_domain(),
        execution.exec_options(),
        "copying::gather",
    )
}

/// Gather (select) rows from a table by index, without bounds checking.
///
/// This is identical to [`gather`] but skips bounds validation for
/// better performance. Out-of-bounds indices cause **undefined behavior**
/// (invalid GPU memory access).
///
/// # Safety
///
/// Caller must ensure that every value in `gather_map` is a valid index
/// into `input` (i.e., in the range `[-n, n)` where `n` is the number
/// of rows in `input`). Violating this causes undefined behavior.
///
/// # Example
/// ```ignore
/// // `ColumnView`/`TableView` have no public constructor, so this advanced
/// // entry point cannot be demonstrated from outside the crate. See
/// // `tests/ops/copying/copying_tests.rs::test_gather_unchecked_basic` for an
/// // executable example.
/// let result = unsafe { cudf::ops::copying::gather_unchecked(&input, &gather_map)? };
/// ```
///
/// Prefer [`gather`] unless profiling shows the bounds check is a bottleneck.
pub unsafe fn gather_unchecked(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::binary(
        input.execution_domain(),
        gather_map.execution_domain(),
        &ctx,
        "copying::gather_unchecked",
    )?;
    gather_impl(
        input,
        gather_map,
        cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
        "dont_check",
        execution.output_domain(),
        execution.exec_options(),
        "copying::gather_unchecked",
    )
}

pub(crate) fn scatter_impl(
    source: &TableView<'_>,
    scatter_map: &ColumnView<'_>,
    target: &TableView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Table> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_scatter_ex(
            source.raw_handle(),
            scatter_map.raw_handle(),
            target.raw_handle(),
            &mut out,
            exec_options,
        )
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("bounds_policy", "dont_check".to_owned()),
            ("duplicate_policy", "undefined_if_duplicate".to_owned()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

fn validate_scatter_map(
    source: &TableView<'_>,
    scatter_map: &ColumnView<'_>,
    target: &TableView<'_>,
    operation: &'static str,
) -> Result<()> {
    let map_dtype = scatter_map.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
    if map_dtype.id() != TypeId::Int32 {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            format!("scatter map must be Int32, got {:?}", map_dtype.id()),
        )
        .with_fact(FACT_LEFT_TYPE_ID, map_dtype.id())
        .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Int32)
        .with_detail("bounds_policy", "validated_on_host")
        .with_detail("duplicate_policy", "reject"));
    }

    let map_len = scatter_map.len().map_err(|e| e.reanchor_if_needed(operation))?;
    let source_rows = source.num_rows().map_err(|e| e.reanchor_if_needed(operation))?;
    if map_len > source_rows {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            format!("scatter map length ({map_len}) exceeds source row count ({source_rows})"),
        )
        .with_fact(FACT_ACTUAL_ROWS, map_len)
        .with_fact(FACT_EXPECTED_ROWS, source_rows)
        .with_detail("resource", "scatter_map")
        .with_detail("reason", "scatter_map_too_long")
        .with_detail("remedy", "provide at most one scatter index per source row"));
    }

    let null_count = scatter_map.null_count().map_err(|e| e.reanchor_if_needed(operation))?;
    if null_count != 0 {
        return Err(Error::new_with(ErrorCode::NullUnexpected, operation, "scatter map must not contain nulls")
            .with_fact(FACT_ACTUAL_COUNT, null_count)
            .with_detail("resource", "scatter_map")
            .with_detail("reason", "null_indices")
            .with_detail("remedy", "drop or fill null scatter indices before scatter"));
    }

    let target_rows = target.num_rows().map_err(|e| e.reanchor_if_needed(operation))?;
    let map = scatter_map.to_vec::<i32>().map_err(|e| e.reanchor_if_needed(operation))?;
    let mut seen: FastHashSet<i32> = fast_hash_set_with_capacity(map.len());
    for (row, index) in map.into_iter().enumerate() {
        let normalized = if index < 0 { index + target_rows as i32 } else { index };
        if normalized < 0 || normalized as usize >= target_rows {
            return Err(Error::new_with(
                ErrorCode::IndexOutOfBounds,
                operation,
                format!("scatter index {index} at source row {row} is out of bounds for {target_rows} target rows"),
            )
            .with_fact(FACT_TABLE_ROW_COUNT, target_rows)
            .with_detail("resource", "scatter_map")
            .with_detail("reason", "index_out_of_bounds")
            .with_detail("bounds_policy", "validated_on_host")
            .with_detail("source_row_index", row.to_string())
            .with_detail("scatter_index", index.to_string())
            .with_detail("remedy", "use scatter indices in the range [-target_rows, target_rows)"));
        }
        if !seen.insert(normalized) {
            return Err(Error::new_with(
                ErrorCode::DuplicateKey,
                operation,
                format!("scatter map contains duplicate target row index {normalized}"),
            )
            .with_fact(FACT_TABLE_ROW_COUNT, target_rows)
            .with_detail("resource", "scatter_map")
            .with_detail("reason", "duplicate_target_index")
            .with_detail("duplicate_policy", "reject")
            .with_detail("source_row_index", row.to_string())
            .with_detail("normalized_index", normalized.to_string())
            .with_detail("remedy", "deduplicate scatter indices before scatter"));
        }
    }

    Ok(())
}

/// Scatter rows from `source` into a copy of `target`.
///
/// For each row `i` covered by `scatter_map`, row `i` of `source` replaces row
/// `scatter_map[i]` in `target`; other target rows are copied through unchanged.
/// Negative scatter indices are interpreted relative to the target row count,
/// so `-1` addresses the final target row.
///
/// The scatter map must be a non-null `INT32` device column whose length is no
/// greater than the number of source rows. Safe Rust validates the scatter map
/// before calling libcudf: every index must be in
/// `[-target.num_rows(), target.num_rows())` after negative index handling, and
/// duplicate target indices are rejected because libcudf documents duplicate
/// scatter destinations as undefined behavior.
///
/// # Synchronization
///
/// Validation materializes the `INT32` scatter map to host memory and may
/// synchronize the map's execution domain before launching the scatter. Use
/// [`scatter_unchecked`] only when the caller can prove the map is valid and
/// wants to avoid this host validation.
pub fn scatter(source: &TableView<'_>, scatter_map: &ColumnView<'_>, target: &TableView<'_>) -> Result<Table> {
    let operation = "copying::scatter";
    validate_scatter_map(source, scatter_map, target, operation)?;
    unsafe { scatter_unchecked(source, scatter_map, target).map_err(|e| e.reanchor_if_needed(operation)) }
}

/// Scatter rows from `source` into a copy of `target` using a borrowed
/// execution override.
///
/// Same result semantics as [`scatter`]. Safe Rust still validates the scatter
/// map before calling libcudf; validation materializes the `INT32` scatter map
/// to host memory and may synchronize the map's execution domain before
/// launching the scatter. The borrowed override selects the stream and/or
/// memory resource used for the newly allocated output table when compatible
/// with the source, map, and target devices.
pub fn scatter_in(
    source: &TableView<'_>,
    scatter_map: &ColumnView<'_>,
    target: &TableView<'_>,
    ctx: &ExecutionOverrides<'_>,
) -> Result<Table> {
    let operation = "copying::scatter_in";
    validate_scatter_map(source, scatter_map, target, operation)?;
    unsafe { scatter_unchecked_in(source, scatter_map, target, ctx).map_err(|e| e.reanchor_if_needed(operation)) }
}

/// Scatter rows from `source` into a copy of `target` without validating the map.
///
/// This calls libcudf directly. The result is a newly allocated table in the
/// target execution domain; source and target column data may be copied into
/// that output as needed by libcudf.
///
/// # Safety
///
/// Caller must ensure that `scatter_map` is a non-null `INT32` column whose
/// length is no greater than the number of source rows, every index is within
/// `[-target.num_rows(), target.num_rows())`, and no two entries address the
/// same normalized target row. Duplicate or out-of-bounds scatter indices cause
/// undefined behavior in libcudf.
pub unsafe fn scatter_unchecked(
    source: &TableView<'_>,
    scatter_map: &ColumnView<'_>,
    target: &TableView<'_>,
) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    unsafe { scatter_unchecked_with_context(source, scatter_map, target, &ctx, "copying::scatter_unchecked") }
}

/// Scatter rows from `source` into a copy of `target` without validating the
/// map, using a borrowed execution override.
///
/// Same result and safety requirements as [`scatter_unchecked`]. The borrowed
/// override selects the stream and/or memory resource used for the newly
/// allocated output table when compatible with the source, map, and target
/// devices.
///
/// # Safety
///
/// Caller must ensure that `scatter_map` is a non-null `INT32` column whose
/// length is no greater than the number of source rows, every index is within
/// `[-target.num_rows(), target.num_rows())`, and no two entries address the
/// same normalized target row. Duplicate or out-of-bounds scatter indices cause
/// undefined behavior in libcudf.
pub unsafe fn scatter_unchecked_in(
    source: &TableView<'_>,
    scatter_map: &ColumnView<'_>,
    target: &TableView<'_>,
    ctx: &ExecutionOverrides<'_>,
) -> Result<Table> {
    unsafe { scatter_unchecked_with_context(source, scatter_map, target, ctx, "copying::scatter_unchecked_in") }
}

unsafe fn scatter_unchecked_with_context(
    source: &TableView<'_>,
    scatter_map: &ColumnView<'_>,
    target: &TableView<'_>,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<Table> {
    let execution = ResolvedExecution::many(
        target.execution_domain(),
        [source.execution_domain(), scatter_map.execution_domain()],
        ctx,
        operation,
    )?;
    scatter_impl(source, scatter_map, target, execution.output_domain(), execution.exec_options(), operation)
}

// ---------------------------------------------------------------------------
// copy_if_else (column-column)
// ---------------------------------------------------------------------------

pub(crate) fn copy_if_else_impl(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    boolean_mask: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_copy_if_else_vv_ex(
            lhs.raw_handle(),
            rhs.raw_handle(),
            boolean_mask.raw_handle(),
            &mut out,
            exec_options,
        )
    };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Conditionally select elements from two columns based on a boolean mask.
///
/// Produces a new column where:
/// - `output[i] = lhs[i]` when `boolean_mask[i]` is valid and `true`
/// - `output[i] = rhs[i]` when `boolean_mask[i]` is null or `false`
///
/// `lhs` and `rhs` must have the same type and length as `boolean_mask`.
/// `boolean_mask` must be of type `BOOL8`.
///
/// # Example
/// ```no_run
/// use cudf::ops::expr::{col, when};
///
/// let rt = cudf::Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "lhs" => rt.series(&[10i32, 20, 30]).unwrap(),
///     "rhs" => rt.series(&[1i32, 2, 3]).unwrap(),
///     "keep_lhs" => rt.series(&[true, false, true]).unwrap(),
/// }
/// .unwrap();
/// let out = df
///     .with_columns([when(col("keep_lhs"))
///         .then(col("lhs"))
///         .otherwise(col("rhs"))
///         .alias("out")])
///     .unwrap();
/// assert_eq!(
///     out.series("out").unwrap().to_vec_strict::<i32>().unwrap(),
///     vec![10, 2, 30]
/// );
/// ```
pub fn copy_if_else(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>, boolean_mask: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::many(
        lhs.execution_domain(),
        [rhs.execution_domain(), boolean_mask.execution_domain()],
        &ctx,
        "copying::copy_if_else",
    )?;
    copy_if_else_impl(
        lhs,
        rhs,
        boolean_mask,
        execution.output_domain(),
        execution.exec_options(),
        "copying::copy_if_else",
    )
}

/// Conditionally select elements from two columns using an explicit execution override.
pub fn copy_if_else_in(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    boolean_mask: &ColumnView<'_>,
    ctx: &ExecutionOverrides<'_>,
) -> Result<Column> {
    let execution = ResolvedExecution::many(
        lhs.execution_domain(),
        [rhs.execution_domain(), boolean_mask.execution_domain()],
        ctx,
        "copying::copy_if_else_in",
    )?;
    copy_if_else_impl(
        lhs,
        rhs,
        boolean_mask,
        execution.output_domain(),
        execution.exec_options(),
        "copying::copy_if_else_in",
    )
}

// ---------------------------------------------------------------------------
// shift
// ---------------------------------------------------------------------------

/// Shift the rows of a column by `offset` positions and fill indeterminate
/// outputs with `fill_value`.
///
/// `output[i] = input[i - offset]` when `i - offset` is in `[0, input.len())`,
/// otherwise `output[i] = fill_value`. A positive `offset` moves rows toward
/// larger indices (a "lag"); a negative `offset` moves rows toward smaller
/// indices (a "lead").
///
/// libcudf restricts `input` to fixed-width or string types. `fill_value`
/// must have the same data type as `input`; the check runs in Rust before
/// libcudf is invoked, with structured facts attached on failure. If `input`
/// is nullable or `fill_value` is null-valued, the output column is also
/// nullable.
///
/// # Example
/// ```ignore
/// // `ColumnView` has no public constructor, so this low-level entry point
/// // cannot be demonstrated from outside the crate. See
/// // `tests/ops/copying/shift_tests.rs::shift_forward_with_null_fill` for an
/// // executable example.
/// let shifted = cudf::ops::copying::shift(&input, 1, &fill)?;
/// // Output: [NULL, 10, 20, 30]
/// ```
pub fn shift(input: &ColumnView<'_>, offset: i32, fill_value: &Scalar) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    shift_with_context(input, offset, fill_value, &ctx, "copying::shift")
}

/// Shift with a borrowed execution override.
pub fn shift_in(
    input: &ColumnView<'_>,
    offset: i32,
    fill_value: &Scalar,
    ctx: &ExecutionOverrides<'_>,
) -> Result<Column> {
    shift_with_context(input, offset, fill_value, ctx, "copying::shift_in")
}

fn shift_with_context(
    input: &ColumnView<'_>,
    offset: i32,
    fill_value: &Scalar,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<Column> {
    validate_shift_fill_dtype(input, fill_value, operation)?;
    let execution = ResolvedExecution::binary(input.execution_domain(), fill_value.execution_domain(), ctx, operation)?;
    shift_impl(input, offset, fill_value, execution.output_domain(), execution.exec_options(), operation)
}

fn validate_shift_fill_dtype(input: &ColumnView<'_>, fill_value: &Scalar, operation: &'static str) -> Result<()> {
    let input_dtype = input.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
    let fill_dtype = fill_value.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
    if input_dtype != fill_dtype {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            format!("shift fill_value dtype ({:?}) must match input dtype ({:?})", fill_dtype.id(), input_dtype.id()),
        )
        .with_fact(FACT_LEFT_TYPE_ID, input_dtype.id())
        .with_fact(FACT_RIGHT_TYPE_ID, fill_dtype.id())
        .with_detail("resource", "shift_fill_value")
        .with_detail("reason", "dtype_mismatch")
        .with_detail("remedy", "construct fill_value with the same DataType as input"));
    }
    Ok(())
}

pub(crate) fn shift_impl(
    input: &ColumnView<'_>,
    offset: i32,
    fill_value: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_shift_ex(input.raw_handle(), offset, fill_value.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [("offset", offset.to_string()), ("execution_authority", "retained_domain".to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
