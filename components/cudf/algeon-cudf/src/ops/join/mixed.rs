use std::sync::Arc;

use super::{
    AstExpression, Column, Exec, ExecutionDomain, JoinIndices, JoinType, MixedJoinSizeData, MixedJoinSizeDataHandle,
    MixedJoinTables, MixedSingleSideJoinSizeData, NullEquality, ResolvedExecution, Result, SingleSideJoinIndices,
    SizedMixedJoinType, SizedMixedSingleSideJoinType, TableView, check_join_indices_status,
    compatible_mixed_join_domain, error, ptr,
};

fn mixed_join_indices_impl(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
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
            cudf_sys::cudf_mixed_inner_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                exec_options,
            )
        },
        JoinType::Left => unsafe {
            cudf_sys::cudf_mixed_left_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                exec_options,
            )
        },
        JoinType::Full => unsafe {
            cudf_sys::cudf_mixed_full_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
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

fn mixed_join_begin_size_data_impl<'a>(
    tables: MixedJoinTables<'_, '_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    join_type: SizedMixedJoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<MixedJoinSizeData<'a>> {
    let mut handle = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match join_type {
        SizedMixedJoinType::Inner => unsafe {
            cudf_sys::cudf_mixed_inner_join_begin_size_data_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut handle,
                exec_options,
            )
        },
        SizedMixedJoinType::Left => unsafe {
            cudf_sys::cudf_mixed_left_join_begin_size_data_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut handle,
                exec_options,
            )
        },
        SizedMixedJoinType::Full => unsafe {
            cudf_sys::cudf_mixed_full_join_begin_size_data_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut handle,
                exec_options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", join_type.as_str().to_string()),
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
            ("reservation_contract", "mixed_join_retained_size_data".to_string()),
        ],
    )?;

    let handle = MixedJoinSizeDataHandle {
        handle: error::require_non_null(handle, operation, "mixed join size data")?,
        _domain: Arc::clone(output_domain),
    };
    let mut output_size = 0usize;
    error::check_status_in(
        unsafe { cudf_sys::cudf_mixed_join_size_data_output_size(handle.handle.as_ptr(), &mut output_size) },
        operation,
    )?;
    Ok(MixedJoinSizeData {
        handle,
        output_domain: Arc::clone(output_domain),
        output_size,
        join_type,
        _predicate: predicate,
    })
}

fn mixed_single_side_join_begin_size_data_impl<'a>(
    tables: MixedJoinTables<'_, '_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    join_type: SizedMixedSingleSideJoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<MixedSingleSideJoinSizeData<'a>> {
    let mut handle = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match join_type {
        SizedMixedSingleSideJoinType::LeftSemi => unsafe {
            cudf_sys::cudf_mixed_left_semi_join_begin_size_data_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut handle,
                exec_options,
            )
        },
        SizedMixedSingleSideJoinType::LeftAnti => unsafe {
            cudf_sys::cudf_mixed_left_anti_join_begin_size_data_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut handle,
                exec_options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", join_type.as_str().to_string()),
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
            ("reservation_contract", "mixed_join_retained_size_data".to_string()),
        ],
    )?;

    let handle = MixedJoinSizeDataHandle {
        handle: error::require_non_null(handle, operation, "mixed join size data")?,
        _domain: Arc::clone(output_domain),
    };
    let mut output_size = 0usize;
    error::check_status_in(
        unsafe { cudf_sys::cudf_mixed_join_size_data_output_size(handle.handle.as_ptr(), &mut output_size) },
        operation,
    )?;
    Ok(MixedSingleSideJoinSizeData {
        handle,
        output_domain: Arc::clone(output_domain),
        output_size,
        join_type,
        _predicate: predicate,
    })
}

fn mixed_single_side_join_indices_impl(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
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
            cudf_sys::cudf_mixed_left_semi_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out,
                exec_options,
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_mixed_left_anti_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out,
                exec_options,
            )
        }
    };

    check_join_indices_status(status, null_equality, operation)?;

    Ok(SingleSideJoinIndices { indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }? })
}

fn mixed_join_indices_with_context(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    join_type: JoinType,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let execution = compatible_mixed_join_domain(tables, predicate, ctx, operation)?;
    mixed_join_indices_impl(
        tables,
        predicate,
        null_equality,
        join_type,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

fn mixed_join_begin_size_data_with_context<'a>(
    tables: MixedJoinTables<'_, '_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    join_type: SizedMixedJoinType,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<MixedJoinSizeData<'a>> {
    let execution = compatible_mixed_join_domain(tables, predicate, ctx, operation)?;
    mixed_join_begin_size_data_impl(
        tables,
        predicate,
        null_equality,
        join_type,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

fn mixed_single_side_join_begin_size_data_with_context<'a>(
    tables: MixedJoinTables<'_, '_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    join_type: SizedMixedSingleSideJoinType,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<MixedSingleSideJoinSizeData<'a>> {
    let execution = compatible_mixed_join_domain(tables, predicate, ctx, operation)?;
    mixed_single_side_join_begin_size_data_impl(
        tables,
        predicate,
        null_equality,
        join_type,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

fn mixed_single_side_join_indices_with_context(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    semi_join: bool,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let execution = compatible_mixed_join_domain(tables, predicate, ctx, operation)?;
    mixed_single_side_join_indices_impl(
        tables,
        predicate,
        null_equality,
        semi_join,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

/// Mixed inner join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_inner_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub(crate) fn mixed_inner_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<JoinIndices> {
    let operation = "join::mixed_inner_join_indices_in";
    mixed_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        JoinType::Inner,
        ctx,
        operation,
    )
}

pub(crate) fn mixed_inner_join_begin_size_data_in<'a>(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<MixedJoinSizeData<'a>> {
    let operation = "join::mixed_inner_join_begin_size_data_in";
    mixed_join_begin_size_data_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        SizedMixedJoinType::Inner,
        ctx,
        operation,
    )
}

/// Mixed left join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_left_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub(crate) fn mixed_left_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<JoinIndices> {
    let operation = "join::mixed_left_join_indices_in";
    mixed_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        JoinType::Left,
        ctx,
        operation,
    )
}

pub(crate) fn mixed_left_join_begin_size_data_in<'a>(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<MixedJoinSizeData<'a>> {
    let operation = "join::mixed_left_join_begin_size_data_in";
    mixed_join_begin_size_data_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        SizedMixedJoinType::Left,
        ctx,
        operation,
    )
}

pub(crate) fn mixed_full_join_begin_size_data_in<'a>(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<MixedJoinSizeData<'a>> {
    let operation = "join::mixed_full_join_begin_size_data_in";
    mixed_join_begin_size_data_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        SizedMixedJoinType::Full,
        ctx,
        operation,
    )
}

/// Mixed full join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_full_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub(crate) fn mixed_full_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<JoinIndices> {
    let operation = "join::mixed_full_join_indices_in";
    mixed_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        JoinType::Full,
        ctx,
        operation,
    )
}

/// Mixed left semi join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_left_semi_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub(crate) fn mixed_left_semi_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<SingleSideJoinIndices> {
    let operation = "join::mixed_left_semi_join_indices_in";
    mixed_single_side_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        true,
        ctx,
        operation,
    )
}

pub(crate) fn mixed_left_semi_join_begin_size_data_in<'a>(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<MixedSingleSideJoinSizeData<'a>> {
    let operation = "join::mixed_left_semi_join_begin_size_data_in";
    mixed_single_side_join_begin_size_data_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        SizedMixedSingleSideJoinType::LeftSemi,
        ctx,
        operation,
    )
}

/// Mixed left anti join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_left_anti_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub(crate) fn mixed_left_anti_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<SingleSideJoinIndices> {
    let operation = "join::mixed_left_anti_join_indices_in";
    mixed_single_side_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        false,
        ctx,
        operation,
    )
}

pub(crate) fn mixed_left_anti_join_begin_size_data_in<'a>(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &'a AstExpression,
    null_equality: NullEquality,
    ctx: Exec<'_>,
) -> Result<MixedSingleSideJoinSizeData<'a>> {
    let operation = "join::mixed_left_anti_join_begin_size_data_in";
    mixed_single_side_join_begin_size_data_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        SizedMixedSingleSideJoinType::LeftAnti,
        ctx,
        operation,
    )
}

fn mixed_size_data_gather_map_reservation_size_bytes(
    handle: &MixedJoinSizeDataHandle,
    operation: &'static str,
) -> Result<usize> {
    let mut out = 0usize;
    error::check_status_in(
        unsafe { cudf_sys::cudf_mixed_join_size_data_gather_map_size(handle.handle.as_ptr(), &mut out) },
        operation,
    )?;
    Ok(out)
}

impl MixedJoinSizeData<'_> {
    /// Returns the exact output row count for this mixed join.
    pub(crate) fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns bytes retained by libcudf's mixed join size-data buffer.
    pub(crate) fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_mixed_join_size_data_device_allocated_size(self.handle.handle.as_ptr(), &mut out) },
            "mixed_join_size_data::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    /// Return exact bytes that must be reserved for this mixed join's output
    /// gather-map pair.
    pub(crate) fn gather_map_reservation_size_bytes(&self) -> Result<usize> {
        mixed_size_data_gather_map_reservation_size_bytes(
            &self.handle,
            "mixed_join_size_data::gather_map_reservation_size_bytes",
        )
    }

    /// Materialize mixed join gather-map pairs from retained size data using a
    /// borrowed execution override.
    pub(crate) fn materialize_indices_in(self, ctx: Exec<'_>) -> Result<JoinIndices> {
        let operation = "mixed_join_size_data::materialize_indices_in";
        let execution = ResolvedExecution::unary(&self.output_domain, ctx, operation)?;
        let mut out_left = ptr::null_mut();
        let mut out_right = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_mixed_join_size_data_materialize_indices_ex(
                    self.handle.handle.as_ptr(),
                    &mut out_left,
                    &mut out_right,
                    execution.exec_options(),
                )
            },
            operation,
            [
                ("join_type", self.join_type.as_str().to_string()),
                ("output_size", self.output_size.to_string()),
                ("execution_authority", "resolved_domain".to_string()),
                ("reservation_contract", "mixed_join_retained_size_data_materialize_indices".to_string()),
            ],
        )?;
        Ok(JoinIndices {
            left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(execution.output_domain())) }?,
            right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(execution.output_domain())) }?,
        })
    }
}

impl MixedSingleSideJoinSizeData<'_> {
    /// Returns the exact output row count for this mixed left semi/anti join.
    pub(crate) fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns bytes retained by libcudf's mixed join size-data buffer.
    pub(crate) fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_mixed_join_size_data_device_allocated_size(self.handle.handle.as_ptr(), &mut out) },
            "mixed_single_side_join_size_data::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    /// Return exact bytes that must be reserved for this mixed join's single
    /// output gather map.
    pub(crate) fn gather_map_reservation_size_bytes(&self) -> Result<usize> {
        mixed_size_data_gather_map_reservation_size_bytes(
            &self.handle,
            "mixed_single_side_join_size_data::gather_map_reservation_size_bytes",
        )
    }

    /// Materialize mixed left semi/anti join gather map from retained size data
    /// using a borrowed execution override.
    pub(crate) fn materialize_indices_in(self, ctx: Exec<'_>) -> Result<SingleSideJoinIndices> {
        let operation = "mixed_single_side_join_size_data::materialize_indices_in";
        let execution = ResolvedExecution::unary(&self.output_domain, ctx, operation)?;
        let mut out = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_mixed_join_size_data_materialize_single_indices_ex(
                    self.handle.handle.as_ptr(),
                    &mut out,
                    execution.exec_options(),
                )
            },
            operation,
            [
                ("join_type", self.join_type.as_str().to_string()),
                ("output_size", self.output_size.to_string()),
                ("execution_authority", "resolved_domain".to_string()),
                ("reservation_contract", "mixed_join_retained_size_data_materialize_single_indices".to_string()),
            ],
        )?;
        Ok(SingleSideJoinIndices {
            indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }?,
        })
    }
}
