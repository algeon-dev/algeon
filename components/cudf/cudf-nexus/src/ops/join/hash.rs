use std::sync::Arc;

use super::*;

pub(crate) fn join_impl(
    tables: SplitJoinTables<'_, '_>,
    null_equality: NullEquality,
    join_type: JoinType,
    context: JoinCallContext<'_>,
) -> Result<JoinSides> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match join_type {
        JoinType::Inner => unsafe {
            cudf_sys::cudf_inner_join_ex(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                context.exec_options,
            )
        },
        JoinType::Left => unsafe {
            cudf_sys::cudf_left_join_ex(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                context.exec_options,
            )
        },
        JoinType::Full => unsafe {
            cudf_sys::cudf_full_join_ex(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                context.exec_options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        context.operation,
        [
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;

    Ok(JoinSides {
        left: unsafe { Table::from_raw_with_domain(out_left, Arc::clone(context.output_domain)) }?,
        right: unsafe { Table::from_raw_with_domain(out_right, Arc::clone(context.output_domain)) }?,
    })
}
pub(crate) fn join_indices_impl(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    join_type: JoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<JoinIndices> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match join_type {
        JoinType::Inner => unsafe {
            cudf_sys::cudf_inner_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                exec_options,
            )
        },
        JoinType::Left => unsafe {
            cudf_sys::cudf_left_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                exec_options,
            )
        },
        JoinType::Full => unsafe {
            cudf_sys::cudf_full_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                exec_options,
            )
        },
    };

    check_join_indices_status(status, null_equality, operation)?;

    Ok(JoinIndices {
        left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
        right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
    })
}

pub(crate) fn single_side_join_indices_impl(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    semi_join: bool,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let mut out = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = if semi_join {
        unsafe {
            cudf_sys::cudf_left_semi_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out,
                exec_options,
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_left_anti_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out,
                exec_options,
            )
        }
    };

    check_join_indices_status(status, null_equality, operation)?;

    Ok(SingleSideJoinIndices { indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }? })
}

pub fn inner_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinSides> {
    let operation = "join::inner_join";
    let tables = SplitJoinTables { left_keys, right_keys, left_payload, right_payload };
    let execution = compatible_join_domain(tables, operation)?;
    join_impl(
        tables,
        null_equality,
        JoinType::Inner,
        JoinCallContext::new(execution.output_domain(), execution.exec_options(), operation),
    )
}

fn join_indices_with_context(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    join_type: JoinType,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let execution = compatible_join_indices_domain(left_keys, right_keys, ctx, operation)?;
    join_indices_impl(
        left_keys,
        right_keys,
        null_equality,
        join_type,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

/// Inner join indices for matching rows in both key tables.
///
/// Returns `INT32` device columns that can be used as gather maps for the
/// left and right tables. This allocates only the index columns; payload
/// columns are not copied or gathered by this operation.
pub fn inner_join_indices(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    join_indices_with_context(left_keys, right_keys, null_equality, JoinType::Inner, &ctx, "join::inner_join_indices")
}

/// Inner join indices using a borrowed execution override.
///
/// Same result semantics as [`inner_join_indices`]. This allocates only
/// `INT32` device gather maps and does not gather or copy payload columns. See
/// the [join module](crate::ops::join)'s shared `_in` execution-context
/// contract.
pub fn inner_join_indices_in(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    join_indices_with_context(left_keys, right_keys, null_equality, JoinType::Inner, ctx, "join::inner_join_indices_in")
}

/// Left join: all left rows are preserved; right side is null-padded
/// for non-matching rows.
pub fn left_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinSides> {
    let operation = "join::left_join";
    let tables = SplitJoinTables { left_keys, right_keys, left_payload, right_payload };
    let execution = compatible_join_domain(tables, operation)?;
    join_impl(
        tables,
        null_equality,
        JoinType::Left,
        JoinCallContext::new(execution.output_domain(), execution.exec_options(), operation),
    )
}

/// Left join indices preserving all rows from the left key table.
///
/// Misses in `right` contain libcudf's no-match sentinel and should be used
/// with nullifying gather behavior when materializing the right side.
pub fn left_join_indices(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    join_indices_with_context(left_keys, right_keys, null_equality, JoinType::Left, &ctx, "join::left_join_indices")
}

/// Left join indices using a borrowed execution override.
///
/// Same result semantics as [`left_join_indices`]. This allocates only
/// `INT32` device gather maps and does not gather or copy payload columns. See
/// the [join module](crate::ops::join)'s shared `_in` execution-context
/// contract.
pub fn left_join_indices_in(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    join_indices_with_context(left_keys, right_keys, null_equality, JoinType::Left, ctx, "join::left_join_indices_in")
}

fn single_side_join_indices_with_context(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    semi_join: bool,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let execution = compatible_join_indices_domain(left_keys, right_keys, ctx, operation)?;
    single_side_join_indices_impl(
        left_keys,
        right_keys,
        null_equality,
        semi_join,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

/// Left semi join indices for rows from the left key table that have a match.
///
/// Returns a single `INT32` device gather map indexing rows in `left_keys`.
/// Payload columns are not copied or gathered by this operation. Duplicate
/// matches in `right_keys` do not duplicate left rows; each matching left row
/// appears at most once in the returned gather map.
pub fn left_semi_join_indices(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<SingleSideJoinIndices> {
    let ctx = ExecutionOverrides::new();
    single_side_join_indices_with_context(
        left_keys,
        right_keys,
        null_equality,
        true,
        &ctx,
        "join::left_semi_join_indices",
    )
}

/// Left semi join indices using a borrowed execution override.
///
/// Same result semantics as [`left_semi_join_indices`]. This allocates only an
/// `INT32` device gather map and does not gather or copy payload columns. See
/// the [join module](crate::ops::join)'s shared `_in` execution-context
/// contract.
pub fn left_semi_join_indices_in(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
) -> Result<SingleSideJoinIndices> {
    single_side_join_indices_with_context(
        left_keys,
        right_keys,
        null_equality,
        true,
        ctx,
        "join::left_semi_join_indices_in",
    )
}

/// Left anti join indices for rows from the left key table that have no match.
///
/// Returns a single `INT32` device gather map indexing rows in `left_keys`.
/// Payload columns are not copied or gathered by this operation. Duplicate
/// matches in `right_keys` do not duplicate left rows.
pub fn left_anti_join_indices(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<SingleSideJoinIndices> {
    let ctx = ExecutionOverrides::new();
    single_side_join_indices_with_context(
        left_keys,
        right_keys,
        null_equality,
        false,
        &ctx,
        "join::left_anti_join_indices",
    )
}

/// Left anti join indices using a borrowed execution override.
///
/// Same result semantics as [`left_anti_join_indices`]. This allocates only an
/// `INT32` device gather map and does not gather or copy payload columns. See
/// the [join module](crate::ops::join)'s shared `_in` execution-context
/// contract.
pub fn left_anti_join_indices_in(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
) -> Result<SingleSideJoinIndices> {
    single_side_join_indices_with_context(
        left_keys,
        right_keys,
        null_equality,
        false,
        ctx,
        "join::left_anti_join_indices_in",
    )
}

pub fn full_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinSides> {
    let operation = "join::full_join";
    let tables = SplitJoinTables { left_keys, right_keys, left_payload, right_payload };
    let execution = compatible_join_domain(tables, operation)?;
    join_impl(
        tables,
        null_equality,
        JoinType::Full,
        JoinCallContext::new(execution.output_domain(), execution.exec_options(), operation),
    )
}

/// Full join indices preserving rows from both key tables.
///
/// Misses on either side contain libcudf's no-match sentinel and should be
/// used with nullifying gather behavior when materializing either side.
pub fn full_join_indices(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    join_indices_with_context(left_keys, right_keys, null_equality, JoinType::Full, &ctx, "join::full_join_indices")
}

/// Full join indices using a borrowed execution override.
///
/// Same result semantics as [`full_join_indices`]. This allocates only
/// `INT32` device gather maps and does not gather or copy payload columns. See
/// the [join module](crate::ops::join)'s shared `_in` execution-context
/// contract.
pub fn full_join_indices_in(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    join_indices_with_context(left_keys, right_keys, null_equality, JoinType::Full, ctx, "join::full_join_indices_in")
}

fn checked_cross_join_output_rows(
    left: &TableView<'_>,
    right: &TableView<'_>,
    operation: &'static str,
) -> Result<usize> {
    let left_rows = left.num_rows().map_err(|e| e.reanchor_if_needed(operation))?;
    let right_rows = right.num_rows().map_err(|e| e.reanchor_if_needed(operation))?;
    checked_cross_join_output_rows_for_counts(left_rows, right_rows, operation)
}

pub(crate) fn checked_cross_join_output_rows_for_counts(
    left_rows: usize,
    right_rows: usize,
    operation: &'static str,
) -> Result<usize> {
    checked_cross_join_side_rows(left_rows, "left_rows", operation)?;
    checked_cross_join_side_rows(right_rows, "right_rows", operation)?;
    let output_rows = left_rows.checked_mul(right_rows).ok_or_else(|| {
        error::limit_validation_error(
            operation,
            "cross join output row count overflowed usize",
            "output_rows",
            "table_row_count",
            "usize::MAX",
            usize::MAX,
            "reduce left or right row count before cross joining",
        )
    })?;
    if output_rows > i32::MAX as usize {
        return Err(error::limit_validation_error(
            operation,
            "cross join output row count exceeds i32::MAX",
            "output_rows",
            "table_row_count",
            "i32::MAX",
            output_rows,
            "reduce left or right row count before cross joining",
        ));
    }
    Ok(output_rows)
}

fn checked_cross_join_side_rows(rows: usize, value_key: &'static str, operation: &'static str) -> Result<()> {
    if rows > i32::MAX as usize {
        return Err(error::limit_validation_error(
            operation,
            "cross join input row count exceeds i32::MAX",
            value_key,
            value_key,
            "i32::MAX",
            rows,
            "reduce left or right row count before cross joining",
        ));
    }
    Ok(())
}

pub(crate) fn projected_cross_join_impl(
    left_selected: Option<&TableView<'_>>,
    right_selected: Option<&TableView<'_>>,
    left_rows: usize,
    right_rows: usize,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Table> {
    let output_rows = checked_cross_join_output_rows_for_counts(left_rows, right_rows, operation)?;
    if left_selected.is_none() && right_selected.is_none() {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "projected cross join requires at least one selected output column",
        )
        .with_detail("reason", "empty_cross_join_projection")
        .with_detail("remedy", "select at least one left or right output column"));
    }

    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_projected_cross_join_ex(
            left_selected.map_or(ptr::null_mut(), TableView::raw_handle),
            right_selected.map_or(ptr::null_mut(), TableView::raw_handle),
            left_rows,
            right_rows,
            &mut out,
            exec_options,
        )
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("output_rows", output_rows.to_string()),
            ("left_projected_columns", left_selected.is_some().to_string()),
            ("right_projected_columns", right_selected.is_some().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn projected_cross_join_device_allocated_size_impl(
    left_selected: Option<&TableView<'_>>,
    right_selected: Option<&TableView<'_>>,
    left_rows: usize,
    right_rows: usize,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
    reservation_contract: &'static str,
) -> Result<usize> {
    let output_rows = checked_cross_join_output_rows_for_counts(left_rows, right_rows, operation)?;
    if left_selected.is_none() && right_selected.is_none() {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "projected cross join requires at least one selected output column",
        )
        .with_detail("reason", "empty_cross_join_projection")
        .with_detail("remedy", "select at least one left or right output column"));
    }

    let mut out = 0usize;
    let status = unsafe {
        cudf_sys::cudf_projected_cross_join_device_allocated_size_ex(
            left_selected.map_or(ptr::null_mut(), TableView::raw_handle),
            right_selected.map_or(ptr::null_mut(), TableView::raw_handle),
            left_rows,
            right_rows,
            &mut out,
            exec_options,
        )
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("output_rows", output_rows.to_string()),
            ("left_projected_columns", left_selected.is_some().to_string()),
            ("right_projected_columns", right_selected.is_some().to_string()),
            ("execution_authority", "retained_domain".to_string()),
            ("reservation_contract", reservation_contract.to_string()),
        ],
    )?;
    Ok(out)
}

fn cross_join_with_context(
    left: &TableView<'_>,
    right: &TableView<'_>,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<Table> {
    let output_rows = checked_cross_join_output_rows(left, right, operation)?;

    let execution = ResolvedExecution::binary(left.execution_domain(), right.execution_domain(), ctx, operation)?;

    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_cross_join_ex(left.raw_handle(), right.raw_handle(), &mut out, execution.exec_options())
    };
    error::check_status_in_with_details(
        status,
        operation,
        [("output_rows", output_rows.to_string()), ("execution_authority", "retained_domain".to_string())],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }
}

/// Cross join two tables.
///
/// The output cardinality is `left.num_rows() * right.num_rows()`. This
/// operation materializes the cartesian product into a new table and copies
/// column data into that output; it is not a zero-copy row selection. The
/// computed row count is checked before calling libcudf and must fit in
/// libcudf's `INT32` row-count range.
pub fn cross_join(left: &TableView<'_>, right: &TableView<'_>) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    cross_join_with_context(left, right, &ctx, "join::cross_join")
}

/// Cross join two tables using a borrowed execution override.
///
/// Same result semantics as [`cross_join`]. The borrowed override selects the
/// stream and/or memory resource used for the newly materialized output table
/// when compatible with the left and right input devices.
pub fn cross_join_in(left: &TableView<'_>, right: &TableView<'_>, ctx: &ExecutionOverrides<'_>) -> Result<Table> {
    cross_join_with_context(left, right, ctx, "join::cross_join_in")
}

impl HashJoin {
    /// Return exact device bytes that must be reserved before building a
    /// reusable hash-join artifact from this build key table.
    ///
    /// This is a planning contract for bounded engines. It returns the exact
    /// retained bytes for the reusable hash table and build-key device views,
    /// without building or allocating retained payload. Flat nullable string
    /// keys include their retained null-payload sanitization allocation; nested
    /// build keys are rejected by libcudf.
    pub fn pre_build_reservation_size_bytes(build: &Table, options: HashJoinOptions) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        Self::pre_build_reservation_size_bytes_in(build, options, &ctx)
    }

    /// Return exact device bytes that must be reserved before building a
    /// reusable hash-join artifact using a borrowed execution override.
    ///
    /// The execution override is part of the contract so sizing validates and
    /// reports failures under the same stream and memory-resource policy as
    /// [`HashJoin::new_in`].
    pub fn pre_build_reservation_size_bytes_in(
        build: &Table,
        options: HashJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize> {
        Self::pre_build_reservation_size_bytes_with_context(
            build,
            options,
            ctx,
            "hash_join::pre_build_reservation_size_bytes_in",
        )
    }

    /// Build a reusable hash-join artifact from an owned build key table.
    ///
    /// The build table is retained by the returned handle because libcudf's
    /// hash-join state borrows build-side device memory. Construction may
    /// allocate GPU memory for the hash table but does not copy payload
    /// columns.
    pub fn new(build: Table, options: HashJoinOptions) -> Result<Self> {
        let ctx = ExecutionOverrides::new();
        Self::new_with_context(build, options, &ctx, "hash_join::new")
    }

    /// Build a reusable hash-join artifact using a borrowed execution override.
    ///
    /// Same lifetime and result semantics as [`HashJoin::new`]. The borrowed
    /// override selects the stream and/or memory resource used while building
    /// the libcudf hash table when compatible with the build table's device.
    pub fn new_in(build: Table, options: HashJoinOptions, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        Self::new_with_context(build, options, ctx, "hash_join::new_in")
    }

    fn pre_build_reservation_size_bytes_with_context(
        build: &Table,
        options: HashJoinOptions,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        let build_view = build.view().map_err(|e| e.reanchor_if_needed(operation))?;
        match Self::pre_build_reservation_size_bytes_for_view(&build_view, options, ctx, operation) {
            Ok(bytes) => Ok(bytes),
            Err(error)
                if is_hash_join_retained_preprocessing_reservation_gap(&error)
                    && hash_join_retained_preprocessing_fallback_supported(&build_view)? =>
            {
                let build_copy = build.deep_copy().map_err(|e| e.reanchor_if_needed(operation))?;
                let join = Self::new_in(build_copy, options, ctx).map_err(|e| e.reanchor_if_needed(operation))?;
                join.device_allocated_size_bytes().map_err(|e| e.reanchor_if_needed(operation))
            },
            Err(error) => Err(error),
        }
    }

    pub(crate) fn pre_build_reservation_size_bytes_for_view(
        build_view: &TableView<'_>,
        options: HashJoinOptions,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        let execution = ResolvedExecution::unary(build_view.execution_domain(), ctx, operation)?;
        let mut out = 0usize;
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_hash_join_pre_build_reservation_size_ex(
                    build_view.raw_handle(),
                    options.load_factor_value(),
                    &mut out,
                    execution.exec_options(),
                )
            },
            operation,
            [
                ("load_factor", options.load_factor_value().to_string()),
                ("execution_authority", "retained_domain".to_string()),
                ("reservation_contract", "hash_join_pre_build_reservation_size_bytes".to_string()),
            ],
        )?;
        Ok(out)
    }

    fn new_with_context(
        build: Table,
        options: HashJoinOptions,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let handle = {
            let build_view = build.view().map_err(|e| e.reanchor_if_needed(operation))?;
            let execution = ResolvedExecution::unary(build.execution_domain(), ctx, operation)?;
            let mut handle = ptr::null_mut();
            error::check_status_in_with_details(
                unsafe {
                    cudf_sys::cudf_hash_join_create_ex(
                        build_view.raw_handle(),
                        options.nullable_join_setting().to_raw(),
                        options.null_eq().to_join_raw(),
                        options.load_factor_value(),
                        &mut handle,
                        execution.exec_options(),
                    )
                },
                operation,
                [
                    ("nullable_join", format!("{:?}", options.nullable_join_setting()).to_lowercase()),
                    ("null_equality", format!("{:?}", options.null_eq()).to_lowercase()),
                    ("load_factor", options.load_factor_value().to_string()),
                    ("execution_authority", "retained_domain".to_string()),
                ],
            )?;
            HashJoinHandle {
                handle: error::require_non_null(handle, operation, "hash join")?,
                _domain: Arc::clone(execution.output_domain()),
            }
        };

        Ok(Self { handle: Arc::new(handle), build: Arc::new(build), options })
    }

    /// Returns the retained build key table.
    pub fn build_keys(&self) -> &Table {
        &self.build
    }

    /// Returns the options used when this hash table was built.
    pub fn options(&self) -> HashJoinOptions {
        self.options
    }

    /// Returns bytes retained by this reusable hash-join build artifact.
    ///
    /// This is observed from the device memory resource used while constructing
    /// the libcudf hash join. It excludes the retained build key table and any
    /// per-probe output/index allocations.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_hash_join_device_allocated_size(self.handle.handle.as_ptr(), &mut out) },
            "hash_join::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    /// Probe with inner-join semantics.
    ///
    /// The returned `left` column indexes the probe table, and `right` indexes
    /// the retained build table.
    pub fn inner_join_indices(&self, probe_keys: &TableView<'_>) -> Result<JoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.probe_indices_with_context(probe_keys, JoinType::Inner, &ctx, "hash_join::inner_join_indices")
    }

    /// Probe with inner-join semantics using an exact caller-supplied output row count.
    pub fn inner_join_indices_with_output_size(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
    ) -> Result<JoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.probe_indices_with_output_size_in(
            probe_keys,
            JoinType::Inner,
            output_size,
            &ctx,
            "hash_join::inner_join_indices_with_output_size",
        )
    }

    /// Probe with inner-join semantics using a borrowed execution override.
    ///
    /// Same result semantics as [`HashJoin::inner_join_indices`]. This
    /// allocates only `INT32` device gather maps and does not gather or copy
    /// payload columns. See the [join module](crate::ops::join)'s shared `_in`
    /// execution-context contract.
    pub fn inner_join_indices_in(
        &self,
        probe_keys: &TableView<'_>,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<JoinIndices> {
        self.probe_indices_with_context(probe_keys, JoinType::Inner, ctx, "hash_join::inner_join_indices_in")
    }

    /// Probe with inner-join semantics using a borrowed execution override and
    /// an exact caller-supplied output row count.
    pub fn inner_join_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<JoinIndices> {
        self.probe_indices_with_output_size_in(
            probe_keys,
            JoinType::Inner,
            output_size,
            ctx,
            "hash_join::inner_join_indices_with_output_size_in",
        )
    }

    /// Return the exact output row count for an inner probe.
    ///
    /// This calls libcudf's reusable hash-join size API and does not allocate
    /// or return join gather maps.
    pub fn inner_join_size(&self, probe_keys: &TableView<'_>) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.probe_size_with_context(probe_keys, JoinType::Inner, &ctx, "hash_join::inner_join_size")
    }

    /// Return the exact output row count for an inner probe using a borrowed execution override.
    pub fn inner_join_size_in(&self, probe_keys: &TableView<'_>, ctx: &ExecutionOverrides<'_>) -> Result<usize> {
        self.probe_size_with_context(probe_keys, JoinType::Inner, ctx, "hash_join::inner_join_size_in")
    }

    /// Probe with left-join semantics.
    ///
    /// Misses in the returned `right` column contain libcudf's no-match
    /// sentinel for nullifying gather.
    pub fn left_join_indices(&self, probe_keys: &TableView<'_>) -> Result<JoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.probe_indices_with_context(probe_keys, JoinType::Left, &ctx, "hash_join::left_join_indices")
    }

    /// Probe with left-join semantics using an exact caller-supplied output row count.
    pub fn left_join_indices_with_output_size(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
    ) -> Result<JoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.probe_indices_with_output_size_in(
            probe_keys,
            JoinType::Left,
            output_size,
            &ctx,
            "hash_join::left_join_indices_with_output_size",
        )
    }

    /// Probe with left-join semantics using a borrowed execution override.
    ///
    /// Same result semantics as [`HashJoin::left_join_indices`]. This
    /// allocates only `INT32` device gather maps and does not gather or copy
    /// payload columns. See the [join module](crate::ops::join)'s shared `_in`
    /// execution-context contract.
    pub fn left_join_indices_in(
        &self,
        probe_keys: &TableView<'_>,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<JoinIndices> {
        self.probe_indices_with_context(probe_keys, JoinType::Left, ctx, "hash_join::left_join_indices_in")
    }

    /// Probe with left-join semantics using a borrowed execution override and
    /// an exact caller-supplied output row count.
    pub fn left_join_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<JoinIndices> {
        self.probe_indices_with_output_size_in(
            probe_keys,
            JoinType::Left,
            output_size,
            ctx,
            "hash_join::left_join_indices_with_output_size_in",
        )
    }

    /// Return the exact output row count for a left probe.
    ///
    /// This calls libcudf's reusable hash-join size API and does not allocate
    /// or return join gather maps.
    pub fn left_join_size(&self, probe_keys: &TableView<'_>) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.probe_size_with_context(probe_keys, JoinType::Left, &ctx, "hash_join::left_join_size")
    }

    /// Return the exact output row count for a left probe using a borrowed execution override.
    pub fn left_join_size_in(&self, probe_keys: &TableView<'_>, ctx: &ExecutionOverrides<'_>) -> Result<usize> {
        self.probe_size_with_context(probe_keys, JoinType::Left, ctx, "hash_join::left_join_size_in")
    }

    /// Probe with full-join semantics.
    ///
    /// Misses on either side contain libcudf's no-match sentinel for
    /// nullifying gather.
    pub fn full_join_indices(&self, probe_keys: &TableView<'_>) -> Result<JoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.probe_indices_with_context(probe_keys, JoinType::Full, &ctx, "hash_join::full_join_indices")
    }

    /// Probe with full-join semantics using an exact caller-supplied output row count.
    pub fn full_join_indices_with_output_size(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
    ) -> Result<JoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.probe_indices_with_output_size_in(
            probe_keys,
            JoinType::Full,
            output_size,
            &ctx,
            "hash_join::full_join_indices_with_output_size",
        )
    }

    /// Probe with full-join semantics using a borrowed execution override.
    ///
    /// Same result semantics as [`HashJoin::full_join_indices`]. This
    /// allocates only `INT32` device gather maps and does not gather or copy
    /// payload columns. See the [join module](crate::ops::join)'s shared `_in`
    /// execution-context contract.
    pub fn full_join_indices_in(
        &self,
        probe_keys: &TableView<'_>,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<JoinIndices> {
        self.probe_indices_with_context(probe_keys, JoinType::Full, ctx, "hash_join::full_join_indices_in")
    }

    /// Probe with full-join semantics using a borrowed execution override and
    /// an exact caller-supplied output row count.
    pub fn full_join_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<JoinIndices> {
        self.probe_indices_with_output_size_in(
            probe_keys,
            JoinType::Full,
            output_size,
            ctx,
            "hash_join::full_join_indices_with_output_size_in",
        )
    }

    /// Return the exact output row count for a full probe.
    ///
    /// This calls libcudf's reusable hash-join size API and does not allocate
    /// or return join gather maps.
    pub fn full_join_size(&self, probe_keys: &TableView<'_>) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.probe_size_with_context(probe_keys, JoinType::Full, &ctx, "hash_join::full_join_size")
    }

    /// Return the exact output row count for a full probe using a borrowed execution override.
    pub fn full_join_size_in(&self, probe_keys: &TableView<'_>, ctx: &ExecutionOverrides<'_>) -> Result<usize> {
        self.probe_size_with_context(probe_keys, JoinType::Full, ctx, "hash_join::full_join_size_in")
    }

    fn probe_indices_with_context(
        &self,
        probe_keys: &TableView<'_>,
        join_type: JoinType,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<JoinIndices> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        self.probe_indices_impl(
            probe_keys,
            join_type,
            execution.output_domain(),
            None,
            execution.exec_options(),
            operation,
        )
    }

    fn probe_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        join_type: JoinType,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<JoinIndices> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        self.probe_indices_impl(
            probe_keys,
            join_type,
            execution.output_domain(),
            Some(output_size),
            execution.exec_options(),
            operation,
        )
    }

    fn probe_size_with_context(
        &self,
        probe_keys: &TableView<'_>,
        join_type: JoinType,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        self.probe_size_impl(probe_keys, join_type, execution.exec_options(), operation)
    }

    fn probe_size_impl(
        &self,
        probe_keys: &TableView<'_>,
        join_type: JoinType,
        exec_options: &cudf_sys::cudf_exec_options_t,
        operation: &'static str,
    ) -> Result<usize> {
        let mut out = 0usize;
        let status = match join_type {
            JoinType::Inner => unsafe {
                cudf_sys::cudf_hash_join_inner_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    exec_options,
                )
            },
            JoinType::Left => unsafe {
                cudf_sys::cudf_hash_join_left_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    exec_options,
                )
            },
            JoinType::Full => unsafe {
                cudf_sys::cudf_hash_join_full_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    exec_options,
                )
            },
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("nullable_join", format!("{:?}", self.options.nullable_join_setting()).to_lowercase()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_authority", "retained_domain".to_string()),
            ],
        )?;

        Ok(out)
    }

    fn probe_indices_impl(
        &self,
        probe_keys: &TableView<'_>,
        join_type: JoinType,
        output_domain: &Arc<ExecutionDomain>,
        output_size: Option<usize>,
        exec_options: &cudf_sys::cudf_exec_options_t,
        operation: &'static str,
    ) -> Result<JoinIndices> {
        let mut out_left = ptr::null_mut();
        let mut out_right = ptr::null_mut();
        let status = match (join_type, output_size) {
            (JoinType::Inner, None) => unsafe {
                cudf_sys::cudf_hash_join_inner_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    exec_options,
                )
            },
            (JoinType::Left, None) => unsafe {
                cudf_sys::cudf_hash_join_left_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    exec_options,
                )
            },
            (JoinType::Full, None) => unsafe {
                cudf_sys::cudf_hash_join_full_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    exec_options,
                )
            },
            (JoinType::Inner, Some(output_size)) => unsafe {
                cudf_sys::cudf_hash_join_inner_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                    exec_options,
                )
            },
            (JoinType::Left, Some(output_size)) => unsafe {
                cudf_sys::cudf_hash_join_left_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                    exec_options,
                )
            },
            (JoinType::Full, Some(output_size)) => unsafe {
                cudf_sys::cudf_hash_join_full_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                    exec_options,
                )
            },
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("nullable_join", format!("{:?}", self.options.nullable_join_setting()).to_lowercase()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_authority", "retained_domain".to_string()),
                ("output_size_provided", output_size.is_some().to_string()),
            ],
        )?;

        Ok(JoinIndices {
            left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
            right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
        })
    }
}

/// Width of a variable-width key column's offsets buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashJoinStringOffsetWidth {
    I32,
    I64,
}

/// Retained representation of one hash-join key column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashJoinKeyStorage {
    FixedWidth { type_id: TypeId },
    String { payload_upper_bound_bytes: usize, offset_width: HashJoinStringOffsetWidth },
}

/// Typed metadata for one hash-join key column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HashJoinKeyMetadata {
    storage: HashJoinKeyStorage,
    nullable: bool,
}

impl HashJoinKeyMetadata {
    pub fn fixed_width(type_id: TypeId, nullable: bool) -> Self {
        Self { storage: HashJoinKeyStorage::FixedWidth { type_id }, nullable }
    }

    pub fn string(payload_upper_bound_bytes: usize, offset_width: HashJoinStringOffsetWidth, nullable: bool) -> Self {
        Self { storage: HashJoinKeyStorage::String { payload_upper_bound_bytes, offset_width }, nullable }
    }

    pub fn storage(self) -> HashJoinKeyStorage {
        self.storage
    }

    pub fn nullable(self) -> bool {
        self.nullable
    }
}

/// Retained preprocessing selected by the runtime hash-join implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashJoinPreprocessingMode {
    FlatKeys,
    SanitizeNullableStringPayload,
}

impl HashJoinPreprocessingMode {
    pub fn for_keys(keys: &[HashJoinKeyMetadata]) -> Self {
        if keys.iter().any(|key| key.nullable && matches!(key.storage, HashJoinKeyStorage::String { .. })) {
            Self::SanitizeNullableStringPayload
        } else {
            Self::FlatKeys
        }
    }
}

pub(crate) fn is_hash_join_retained_preprocessing_reservation_gap(error: &Error) -> bool {
    error.code() == ErrorCode::ForeignInvalidArgument
        && error.detail_value("reservation_contract") == Some("hash_join_pre_build_reservation_size_bytes")
        && error.message().contains("preprocessing buffers")
}

pub(crate) fn hash_join_retained_preprocessing_fallback_supported(build: &TableView<'_>) -> Result<bool> {
    for index in 0..build.num_columns()? {
        match build.column(index)?.data_type()?.id() {
            TypeId::String | TypeId::List | TypeId::Struct => return Ok(false),
            _ => {},
        }
    }
    Ok(true)
}
