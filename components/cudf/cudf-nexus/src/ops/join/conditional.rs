use std::sync::Arc;

use super::*;

fn conditional_join_indices_impl(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    join_type: JoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<JoinIndices> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let status = match join_type {
        JoinType::Inner => unsafe {
            cudf_sys::cudf_conditional_inner_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out_left,
                &mut out_right,
                exec_options,
            )
        },
        JoinType::Left => unsafe {
            cudf_sys::cudf_conditional_left_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out_left,
                &mut out_right,
                exec_options,
            )
        },
        JoinType::Full => unsafe {
            cudf_sys::cudf_conditional_full_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
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
            ("join_type", format!("{:?}", join_type).to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
            ("output_size_provided", "false".to_string()),
        ],
    )?;

    Ok(JoinIndices {
        left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
        right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
    })
}

fn conditional_join_indices_with_output_size_impl(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    join_type: SizedConditionalPairJoinType,
    output_size: usize,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<JoinIndices> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let status = match join_type {
        SizedConditionalPairJoinType::Inner => unsafe {
            cudf_sys::cudf_conditional_inner_join_indices_with_output_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out_left,
                &mut out_right,
                exec_options,
            )
        },
        SizedConditionalPairJoinType::Left => unsafe {
            cudf_sys::cudf_conditional_left_join_indices_with_output_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
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
            ("join_type", join_type.as_str().to_string()),
            ("execution_authority", "retained_domain".to_string()),
            ("output_size_provided", "true".to_string()),
        ],
    )?;

    Ok(JoinIndices {
        left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
        right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
    })
}

fn conditional_single_side_join_indices_impl(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    semi_join: bool,
    output_size: Option<usize>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let mut out = ptr::null_mut();
    let status = match (semi_join, output_size) {
        (true, None) => unsafe {
            cudf_sys::cudf_conditional_left_semi_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                exec_options,
            )
        },
        (false, None) => unsafe {
            cudf_sys::cudf_conditional_left_anti_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                exec_options,
            )
        },
        (true, Some(output_size)) => unsafe {
            cudf_sys::cudf_conditional_left_semi_join_indices_with_output_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out,
                exec_options,
            )
        },
        (false, Some(output_size)) => unsafe {
            cudf_sys::cudf_conditional_left_anti_join_indices_with_output_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out,
                exec_options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
            ("execution_authority", "retained_domain".to_string()),
            ("output_size_provided", output_size.is_some().to_string()),
        ],
    )?;

    Ok(SingleSideJoinIndices { indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }? })
}

fn conditional_join_size_impl(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    join_type: SizedConditionalPairJoinType,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<usize> {
    let mut out = 0usize;
    let status = match join_type {
        SizedConditionalPairJoinType::Inner => unsafe {
            cudf_sys::cudf_conditional_inner_join_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                exec_options,
            )
        },
        SizedConditionalPairJoinType::Left => unsafe {
            cudf_sys::cudf_conditional_left_join_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                exec_options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [("join_type", join_type.as_str().to_string()), ("execution_authority", "retained_domain".to_string())],
    )?;

    Ok(out)
}

fn conditional_full_join_begin_size_data_impl<'a>(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &'a AstExpression,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<ConditionalFullJoinSizeData<'a>> {
    let mut handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_conditional_full_join_begin_size_data_ex(
            left.raw_handle(),
            right.raw_handle(),
            predicate.raw_handle(),
            &mut handle,
            exec_options,
        )
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", "full".to_string()),
            ("execution_authority", "retained_domain".to_string()),
            ("reservation_contract", "conditional_full_join_retained_size_data".to_string()),
        ],
    )?;

    let handle = ConditionalFullJoinSizeDataHandle {
        handle: error::require_non_null(handle, operation, "conditional full join size data")?,
        _domain: Arc::clone(output_domain),
    };
    let mut output_size = 0usize;
    error::check_status_in(
        unsafe { cudf_sys::cudf_conditional_full_join_size_data_output_size(handle.handle.as_ptr(), &mut output_size) },
        operation,
    )?;
    Ok(ConditionalFullJoinSizeData {
        handle,
        output_domain: Arc::clone(output_domain),
        output_size,
        _predicate: predicate,
    })
}

fn conditional_single_side_join_size_impl(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    semi_join: bool,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<usize> {
    let mut out = 0usize;
    let status = if semi_join {
        unsafe {
            cudf_sys::cudf_conditional_left_semi_join_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                exec_options,
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_conditional_left_anti_join_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                exec_options,
            )
        }
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;

    Ok(out)
}

fn conditional_join_indices_with_context(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    join_type: JoinType,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let execution = compatible_conditional_join_domain(left, right, predicate, ctx, operation)?;
    conditional_join_indices_impl(
        left,
        right,
        predicate,
        join_type,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

fn conditional_join_indices_with_output_size_with_context(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    join_type: SizedConditionalPairJoinType,
    output_size: usize,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let execution = compatible_conditional_join_domain(left, right, predicate, ctx, operation)?;
    conditional_join_indices_with_output_size_impl(
        left,
        right,
        predicate,
        join_type,
        output_size,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

fn conditional_single_side_join_indices_with_context(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    semi_join: bool,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let execution = compatible_conditional_join_domain(left, right, predicate, ctx, operation)?;
    conditional_single_side_join_indices_impl(
        left,
        right,
        predicate,
        semi_join,
        None,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

fn conditional_single_side_join_indices_with_output_size_with_context(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    semi_join: bool,
    output_size: usize,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let execution = compatible_conditional_join_domain(left, right, predicate, ctx, operation)?;
    conditional_single_side_join_indices_impl(
        left,
        right,
        predicate,
        semi_join,
        Some(output_size),
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

fn conditional_join_size_with_context(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    join_type: SizedConditionalPairJoinType,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<usize> {
    let execution = compatible_conditional_join_domain(left, right, predicate, ctx, operation)?;
    conditional_join_size_impl(left, right, predicate, join_type, execution.exec_options(), operation)
}

fn conditional_full_join_begin_size_data_with_context<'a>(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &'a AstExpression,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<ConditionalFullJoinSizeData<'a>> {
    let execution = compatible_conditional_join_domain(left, right, predicate, ctx, operation)?;
    conditional_full_join_begin_size_data_impl(
        left,
        right,
        predicate,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

fn conditional_single_side_join_size_with_context(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    semi_join: bool,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<usize> {
    let execution = compatible_conditional_join_domain(left, right, predicate, ctx, operation)?;
    conditional_single_side_join_size_impl(left, right, predicate, semi_join, execution.exec_options(), operation)
}

/// Conditional inner join indices for row pairs where `predicate` is true.
///
/// `predicate` is evaluated against rows from `left` and `right` using
/// [`AstExpression`] column references. The
/// returned `INT32` device columns are gather maps only: this operation
/// allocates GPU index columns but does not gather or copy payload columns. The
/// output keeps provenance from `left`; same-device inputs are ordered against
/// the resolved output stream, and cross-device inputs are rejected before
/// calling libcudf. This call enqueues work on the resolved stream and does not
/// materialize host-visible data before returning. A predicate result of null
/// is treated as non-matching by libcudf.
pub fn conditional_inner_join_indices(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_join_indices_with_context(
        left,
        right,
        predicate,
        JoinType::Inner,
        &ctx,
        "join::conditional_inner_join_indices",
    )
}

/// Conditional inner join indices using a borrowed execution override.
///
/// Same result semantics as [`conditional_inner_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn conditional_inner_join_indices_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    let operation = "join::conditional_inner_join_indices_in";
    conditional_join_indices_with_context(left, right, predicate, JoinType::Inner, ctx, operation)
}

/// Return the exact output row count for a conditional inner join.
pub fn conditional_inner_join_size(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<usize> {
    let ctx = ExecutionOverrides::new();
    conditional_join_size_with_context(
        left,
        right,
        predicate,
        SizedConditionalPairJoinType::Inner,
        &ctx,
        "join::conditional_inner_join_size",
    )
}

/// Return the exact output row count for a conditional inner join using a
/// borrowed execution override.
pub fn conditional_inner_join_size_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<usize> {
    let operation = "join::conditional_inner_join_size_in";
    conditional_join_size_with_context(left, right, predicate, SizedConditionalPairJoinType::Inner, ctx, operation)
}

/// Conditional inner join indices using an exact caller-supplied output row
/// count.
pub fn conditional_inner_join_indices_with_output_size(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    output_size: usize,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_join_indices_with_output_size_with_context(
        left,
        right,
        predicate,
        SizedConditionalPairJoinType::Inner,
        output_size,
        &ctx,
        "join::conditional_inner_join_indices_with_output_size",
    )
}

/// Conditional inner join indices using a borrowed execution override and an
/// exact caller-supplied output row count.
pub fn conditional_inner_join_indices_with_output_size_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    output_size: usize,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    let operation = "join::conditional_inner_join_indices_with_output_size_in";
    conditional_join_indices_with_output_size_with_context(
        left,
        right,
        predicate,
        SizedConditionalPairJoinType::Inner,
        output_size,
        ctx,
        operation,
    )
}

/// Conditional left join indices preserving all left rows.
///
/// Returned right-side misses contain libcudf's no-match sentinel for
/// nullifying gather. This operation allocates GPU `INT32` gather maps but does
/// not gather or copy payload columns. The output keeps provenance from `left`;
/// same-device inputs are ordered against the resolved output stream, and
/// cross-device inputs are rejected before calling libcudf. This call enqueues
/// work on the resolved stream and does not materialize host-visible data
/// before returning.
pub fn conditional_left_join_indices(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_join_indices_with_context(
        left,
        right,
        predicate,
        JoinType::Left,
        &ctx,
        "join::conditional_left_join_indices",
    )
}

/// Conditional left join indices using a borrowed execution override.
///
/// Same result semantics as [`conditional_left_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn conditional_left_join_indices_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    let operation = "join::conditional_left_join_indices_in";
    conditional_join_indices_with_context(left, right, predicate, JoinType::Left, ctx, operation)
}

/// Return the exact output row count for a conditional left join.
pub fn conditional_left_join_size(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<usize> {
    let ctx = ExecutionOverrides::new();
    conditional_join_size_with_context(
        left,
        right,
        predicate,
        SizedConditionalPairJoinType::Left,
        &ctx,
        "join::conditional_left_join_size",
    )
}

/// Return the exact output row count for a conditional left join using a
/// borrowed execution override.
pub fn conditional_left_join_size_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<usize> {
    let operation = "join::conditional_left_join_size_in";
    conditional_join_size_with_context(left, right, predicate, SizedConditionalPairJoinType::Left, ctx, operation)
}

/// Conditional left join indices using an exact caller-supplied output row
/// count.
pub fn conditional_left_join_indices_with_output_size(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    output_size: usize,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_join_indices_with_output_size_with_context(
        left,
        right,
        predicate,
        SizedConditionalPairJoinType::Left,
        output_size,
        &ctx,
        "join::conditional_left_join_indices_with_output_size",
    )
}

/// Conditional left join indices using a borrowed execution override and an
/// exact caller-supplied output row count.
pub fn conditional_left_join_indices_with_output_size_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    output_size: usize,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    let operation = "join::conditional_left_join_indices_with_output_size_in";
    conditional_join_indices_with_output_size_with_context(
        left,
        right,
        predicate,
        SizedConditionalPairJoinType::Left,
        output_size,
        ctx,
        operation,
    )
}

/// Conditional full join indices preserving unmatched rows from both sides.
///
/// Misses on either side contain libcudf's no-match sentinel for nullifying
/// gather. This operation allocates GPU `INT32` gather maps but does not gather
/// or copy payload columns. The output keeps provenance from `left`;
/// same-device inputs are ordered against the resolved output stream, and
/// cross-device inputs are rejected before calling libcudf. This call enqueues
/// work on the resolved stream and does not materialize host-visible data
/// before returning.
pub fn conditional_full_join_indices(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_join_indices_with_context(
        left,
        right,
        predicate,
        JoinType::Full,
        &ctx,
        "join::conditional_full_join_indices",
    )
}

/// Conditional full join indices using a borrowed execution override.
///
/// Same result semantics as [`conditional_full_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn conditional_full_join_indices_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    let operation = "join::conditional_full_join_indices_in";
    conditional_join_indices_with_context(left, right, predicate, JoinType::Full, ctx, operation)
}

/// Begin retained size-data computation for a conditional full join using a
/// borrowed execution override.
pub(crate) fn conditional_full_join_begin_size_data_in<'a>(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &'a AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<ConditionalFullJoinSizeData<'a>> {
    let operation = "join::conditional_full_join_begin_size_data_in";
    conditional_full_join_begin_size_data_with_context(left, right, predicate, ctx, operation)
}

/// Conditional left semi join indices for rows from `left` with at least one
/// matching `right` row.
///
/// Returns a single `INT32` device gather map. This operation allocates that
/// GPU index column but does not gather or copy payload columns. The output
/// keeps provenance from `left`; same-device inputs are ordered against the
/// resolved output stream, and cross-device inputs are rejected before calling
/// libcudf. This call enqueues work on the resolved stream and does not
/// materialize host-visible data before returning.
pub fn conditional_left_semi_join_indices(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<SingleSideJoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_single_side_join_indices_with_context(
        left,
        right,
        predicate,
        true,
        &ctx,
        "join::conditional_left_semi_join_indices",
    )
}

/// Conditional left semi join indices using a borrowed execution override.
///
/// Same result semantics as [`conditional_left_semi_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn conditional_left_semi_join_indices_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<SingleSideJoinIndices> {
    let operation = "join::conditional_left_semi_join_indices_in";
    conditional_single_side_join_indices_with_context(left, right, predicate, true, ctx, operation)
}

/// Return the exact output row count for a conditional left semi join.
pub fn conditional_left_semi_join_size(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<usize> {
    let ctx = ExecutionOverrides::new();
    conditional_single_side_join_size_with_context(
        left,
        right,
        predicate,
        true,
        &ctx,
        "join::conditional_left_semi_join_size",
    )
}

/// Return the exact output row count for a conditional left semi join using a
/// borrowed execution override.
pub fn conditional_left_semi_join_size_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<usize> {
    let operation = "join::conditional_left_semi_join_size_in";
    conditional_single_side_join_size_with_context(left, right, predicate, true, ctx, operation)
}

/// Conditional left semi join indices using an exact caller-supplied output row
/// count.
pub fn conditional_left_semi_join_indices_with_output_size(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    output_size: usize,
) -> Result<SingleSideJoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_single_side_join_indices_with_output_size_with_context(
        left,
        right,
        predicate,
        true,
        output_size,
        &ctx,
        "join::conditional_left_semi_join_indices_with_output_size",
    )
}

/// Conditional left semi join indices using a borrowed execution override and
/// an exact caller-supplied output row count.
pub fn conditional_left_semi_join_indices_with_output_size_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    output_size: usize,
    ctx: &ExecutionOverrides<'_>,
) -> Result<SingleSideJoinIndices> {
    let operation = "join::conditional_left_semi_join_indices_with_output_size_in";
    conditional_single_side_join_indices_with_output_size_with_context(
        left,
        right,
        predicate,
        true,
        output_size,
        ctx,
        operation,
    )
}

/// Conditional left anti join indices for rows from `left` with no matching
/// `right` row.
///
/// Returns a single `INT32` device gather map. This operation allocates that
/// GPU index column but does not gather or copy payload columns. The output
/// keeps provenance from `left`; same-device inputs are ordered against the
/// resolved output stream, and cross-device inputs are rejected before calling
/// libcudf. This call enqueues work on the resolved stream and does not
/// materialize host-visible data before returning.
pub fn conditional_left_anti_join_indices(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<SingleSideJoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_single_side_join_indices_with_context(
        left,
        right,
        predicate,
        false,
        &ctx,
        "join::conditional_left_anti_join_indices",
    )
}

/// Conditional left anti join indices using a borrowed execution override.
///
/// Same result semantics as [`conditional_left_anti_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn conditional_left_anti_join_indices_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<SingleSideJoinIndices> {
    let operation = "join::conditional_left_anti_join_indices_in";
    conditional_single_side_join_indices_with_context(left, right, predicate, false, ctx, operation)
}

/// Return the exact output row count for a conditional left anti join.
pub fn conditional_left_anti_join_size(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
) -> Result<usize> {
    let ctx = ExecutionOverrides::new();
    conditional_single_side_join_size_with_context(
        left,
        right,
        predicate,
        false,
        &ctx,
        "join::conditional_left_anti_join_size",
    )
}

/// Return the exact output row count for a conditional left anti join using a
/// borrowed execution override.
pub fn conditional_left_anti_join_size_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
) -> Result<usize> {
    let operation = "join::conditional_left_anti_join_size_in";
    conditional_single_side_join_size_with_context(left, right, predicate, false, ctx, operation)
}

/// Conditional left anti join indices using an exact caller-supplied output row
/// count.
pub fn conditional_left_anti_join_indices_with_output_size(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    output_size: usize,
) -> Result<SingleSideJoinIndices> {
    let ctx = ExecutionOverrides::new();
    conditional_single_side_join_indices_with_output_size_with_context(
        left,
        right,
        predicate,
        false,
        output_size,
        &ctx,
        "join::conditional_left_anti_join_indices_with_output_size",
    )
}

/// Conditional left anti join indices using a borrowed execution override and
/// an exact caller-supplied output row count.
pub fn conditional_left_anti_join_indices_with_output_size_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    output_size: usize,
    ctx: &ExecutionOverrides<'_>,
) -> Result<SingleSideJoinIndices> {
    let operation = "join::conditional_left_anti_join_indices_with_output_size_in";
    conditional_single_side_join_indices_with_output_size_with_context(
        left,
        right,
        predicate,
        false,
        output_size,
        ctx,
        operation,
    )
}

impl ConditionalFullJoinSizeData<'_> {
    /// Returns the exact output row count for this conditional full join.
    pub(crate) fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns bytes retained by libcudf's conditional full join size-data state.
    pub(crate) fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_conditional_full_join_size_data_device_allocated_size(
                    self.handle.handle.as_ptr(),
                    &mut out,
                )
            },
            "conditional_full_join_size_data::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    /// Return exact bytes that must be reserved for this conditional full
    /// join's output gather-map pair.
    pub(crate) fn gather_map_reservation_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_conditional_full_join_size_data_gather_map_size(self.handle.handle.as_ptr(), &mut out)
            },
            "conditional_full_join_size_data::gather_map_reservation_size_bytes",
        )?;
        Ok(out)
    }

    /// Materialize conditional full join gather-map pairs from retained size data.
    pub(crate) fn materialize_indices(self) -> Result<JoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.materialize_indices_in(&ctx)
    }

    /// Materialize conditional full join gather-map pairs from retained size
    /// data using a borrowed execution override.
    pub(crate) fn materialize_indices_in(self, ctx: &ExecutionOverrides<'_>) -> Result<JoinIndices> {
        let operation = "conditional_full_join_size_data::materialize_indices_in";
        let execution = ResolvedExecution::unary(&self.output_domain, ctx, operation)?;
        let mut out_left = ptr::null_mut();
        let mut out_right = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_conditional_full_join_size_data_materialize_indices_ex(
                    self.handle.handle.as_ptr(),
                    &mut out_left,
                    &mut out_right,
                    execution.exec_options(),
                )
            },
            operation,
            [
                ("join_type", "full".to_string()),
                ("output_size", self.output_size.to_string()),
                ("execution_authority", "resolved_domain".to_string()),
                ("reservation_contract", "conditional_full_join_retained_size_data_materialize_indices".to_string()),
            ],
        )?;
        Ok(JoinIndices {
            left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(execution.output_domain())) }?,
            right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(execution.output_domain())) }?,
        })
    }
}
