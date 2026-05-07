//! GPU-accelerated join operations.
//!
//! **Advanced/Core.** Most users should prefer
//! [`DataFrame::merge`](crate::DataFrame::merge) with
//! [`MergeOptions`].
//!
//! # Example
//!
//! ```no_run
//! use cudf::{MergeOptions, Runtime};
//!
//! let rt = Runtime::new().unwrap();
//! let left = cudf::dataframe! {
//!     "id" => rt.series(&[1i32, 2, 3]).unwrap(),
//!     "left_val" => rt.series(&[10i32, 20, 30]).unwrap(),
//! }
//! .unwrap();
//! let right = cudf::dataframe! {
//!     "id" => rt.series(&[2i32, 3, 4]).unwrap(),
//!     "right_val" => rt.series(&[200i32, 300, 400]).unwrap(),
//! }
//! .unwrap();
//!
//! let result =
//!     left.merge(&right, MergeOptions::new().on(["id"]).how_inner()).unwrap();
//! assert_eq!(
//!     result.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
//!     vec![2, 3]
//! );
//! ```
//!
//! # Join-Index `_in` Execution Context
//!
//! Join-index APIs ending in `_in` share the same execution-context contract.
//! The borrowed [`crate::ExecutionOverrides`] selects the stream and/or memory
//! resource for GPU index allocation when compatible with the primary left
//! input's device (`left_keys` for equality joins, `probe_keys` for
//! `HashJoin` probes, `left` for conditional joins, and `left_equality` for
//! mixed joins). Output provenance is resolved from that primary input plus the
//! override. Same-device table inputs and predicate literal domains are
//! ordered against the resolved output stream; cross-device inputs are rejected
//! before calling libcudf. These calls enqueue GPU `INT32` index allocation and
//! do not gather payload columns or materialize host-visible data before
//! returning.
//!
//! `filter_join_indices_in` follows the same split, but uses the left index
//! column as the primary input because it transforms already-computed join
//! gather maps rather than building equality matches itself.

use std::{
    fmt,
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

mod dataframe;
mod options;
pub(crate) mod testing_hooks;

pub use options::{FilteredJoinOptions, HashJoinOptions, JoinType, MergeOptions, NullEquality, NullableJoin};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_ROWS, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result,
        },
        table::{Table, TableView},
        types::TypeId,
    },
    ops::ast::AstExpression,
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

/// Device-side gather maps produced by an equality join.
///
/// Both columns are `INT32` device columns. They identify rows in the left and
/// right key tables respectively; unmatched rows contain libcudf's no-match
/// sentinel and are intended for gather paths that use nullifying
/// out-of-bounds handling. Payload columns are not copied until callers gather
/// with these indices.
#[derive(Debug)]
pub struct JoinIndices {
    /// Row indices for the left/probe table.
    pub left: Column,
    /// Row indices for the right/build table.
    pub right: Column,
}

/// Device-side gather map produced by a single-side join.
///
/// The column is an `INT32` device column indexing rows in the left key table.
/// Payload columns are not copied by these index-producing operations. For
/// left semi joins, duplicate matches on the right side do not duplicate left
/// rows in the output index map.
#[derive(Debug)]
pub struct SingleSideJoinIndices {
    /// Row indices for the left table.
    pub indices: Column,
}

/// Split output of a low-level join operation.
#[derive(Debug)]
pub struct JoinSides {
    /// The left-side result table (keys + payload for matching rows).
    pub left: Table,
    /// The right-side result table (keys + payload for matching rows).
    pub right: Table,
}

#[derive(Clone, Copy)]
pub(crate) struct SplitJoinTables<'a, 'view> {
    pub(crate) left_keys: &'a TableView<'view>,
    pub(crate) right_keys: &'a TableView<'view>,
    pub(crate) left_payload: &'a TableView<'view>,
    pub(crate) right_payload: &'a TableView<'view>,
}

#[derive(Clone, Copy)]
struct FilterJoinInputs<'a, 'view> {
    left: &'a TableView<'view>,
    right: &'a TableView<'view>,
    left_indices: &'a ColumnView<'view>,
    right_indices: &'a ColumnView<'view>,
    predicate: &'a AstExpression,
}

#[derive(Clone, Copy)]
struct MixedJoinTables<'a, 'view> {
    left_equality: &'a TableView<'view>,
    right_equality: &'a TableView<'view>,
    left_conditional: &'a TableView<'view>,
    right_conditional: &'a TableView<'view>,
}

#[derive(Clone, Copy)]
pub(crate) struct JoinCallContext<'a> {
    output_domain: &'a Arc<ExecutionDomain>,
    exec_options: Option<&'a cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
}

impl<'a> JoinCallContext<'a> {
    pub(crate) fn new(
        output_domain: &'a Arc<ExecutionDomain>,
        exec_options: Option<&'a cudf_sys::cudf_exec_options_t>,
        operation: &'static str,
    ) -> Self {
        Self { output_domain, exec_options, operation }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SizedConditionalPairJoinType {
    Inner,
    Left,
}

impl SizedConditionalPairJoinType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Inner => "inner",
            Self::Left => "left",
        }
    }
}

pub(crate) fn join_impl(
    tables: SplitJoinTables<'_, '_>,
    null_equality: NullEquality,
    join_type: JoinType,
    context: JoinCallContext<'_>,
) -> Result<JoinSides> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match (join_type, context.exec_options) {
        (JoinType::Inner, None) => unsafe {
            cudf_sys::cudf_inner_join(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Left, None) => unsafe {
            cudf_sys::cudf_left_join(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Full, None) => unsafe {
            cudf_sys::cudf_full_join(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Inner, Some(options)) => unsafe {
            cudf_sys::cudf_inner_join_ex(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Left, Some(options)) => unsafe {
            cudf_sys::cudf_left_join_ex(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Full, Some(options)) => unsafe {
            cudf_sys::cudf_full_join_ex(
                tables.left_keys.raw_handle(),
                tables.right_keys.raw_handle(),
                tables.left_payload.raw_handle(),
                tables.right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        context.operation,
        [
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_context_provided", context.exec_options.is_some().to_string()),
        ],
    )?;

    Ok(JoinSides {
        left: unsafe { Table::from_raw_with_domain(out_left, Arc::clone(context.output_domain)) }?,
        right: unsafe { Table::from_raw_with_domain(out_right, Arc::clone(context.output_domain)) }?,
    })
}

fn check_join_indices_status(
    status: cudf_sys::cudf_status_t,
    null_equality: NullEquality,
    exec_options_present: bool,
    operation: &'static str,
) -> Result<()> {
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_context_provided", exec_options_present.to_string()),
        ],
    )
}

pub(crate) fn join_indices_impl(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    join_type: JoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match (join_type, exec_options) {
        (JoinType::Inner, None) => unsafe {
            cudf_sys::cudf_inner_join_indices(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Left, None) => unsafe {
            cudf_sys::cudf_left_join_indices(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Full, None) => unsafe {
            cudf_sys::cudf_full_join_indices(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Inner, Some(options)) => unsafe {
            cudf_sys::cudf_inner_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Left, Some(options)) => unsafe {
            cudf_sys::cudf_left_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Full, Some(options)) => unsafe {
            cudf_sys::cudf_full_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
    };

    check_join_indices_status(status, null_equality, exec_options.is_some(), operation)?;

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
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let mut out = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match (semi_join, exec_options) {
        (true, None) => unsafe {
            cudf_sys::cudf_left_semi_join_indices(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out,
            )
        },
        (false, None) => unsafe {
            cudf_sys::cudf_left_anti_join_indices(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out,
            )
        },
        (true, Some(options)) => unsafe {
            cudf_sys::cudf_left_semi_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out,
                options,
            )
        },
        (false, Some(options)) => unsafe {
            cudf_sys::cudf_left_anti_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out,
                options,
            )
        },
    };

    check_join_indices_status(status, null_equality, exec_options.is_some(), operation)?;

    Ok(SingleSideJoinIndices { indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }? })
}

fn compatible_join_domain(tables: SplitJoinTables<'_, '_>, operation: &'static str) -> Result<ResolvedExecution> {
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::many(
        tables.left_keys.execution_domain(),
        [
            tables.right_keys.execution_domain(),
            tables.left_payload.execution_domain(),
            tables.right_payload.execution_domain(),
        ],
        &ctx,
        operation,
    )
}

fn compatible_join_indices_domain(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    ResolvedExecution::binary(left_keys.execution_domain(), right_keys.execution_domain(), ctx, operation)
}

fn compatible_conditional_join_domain(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let mut secondaries = Vec::new();
    secondaries.push(right.execution_domain());
    predicate.literal_domains(&mut secondaries);
    ResolvedExecution::many(left.execution_domain(), secondaries, ctx, operation)
}

fn compatible_mixed_join_domain(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let mut secondaries = Vec::new();
    secondaries.push(tables.right_equality.execution_domain());
    secondaries.push(tables.left_conditional.execution_domain());
    secondaries.push(tables.right_conditional.execution_domain());
    predicate.literal_domains(&mut secondaries);
    ResolvedExecution::many(tables.left_equality.execution_domain(), secondaries, ctx, operation)
}

fn compatible_filter_join_indices_domain(
    inputs: FilterJoinInputs<'_, '_>,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let mut secondaries = Vec::new();
    secondaries.push(inputs.right_indices.execution_domain());
    secondaries.push(inputs.left.execution_domain());
    secondaries.push(inputs.right.execution_domain());
    inputs.predicate.literal_domains(&mut secondaries);
    ResolvedExecution::many(inputs.left_indices.execution_domain(), secondaries, ctx, operation)
}

fn conditional_join_indices_impl(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    join_type: JoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let status = match (join_type, exec_options) {
        (JoinType::Inner, None) => unsafe {
            cudf_sys::cudf_conditional_inner_join_indices(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Left, None) => unsafe {
            cudf_sys::cudf_conditional_left_join_indices(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Full, None) => unsafe {
            cudf_sys::cudf_conditional_full_join_indices(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Inner, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_inner_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Left, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Full, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_full_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out_left,
                &mut out_right,
                options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", format!("{:?}", join_type).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
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
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let status = match (join_type, exec_options) {
        (SizedConditionalPairJoinType::Inner, None) => unsafe {
            cudf_sys::cudf_conditional_inner_join_indices_with_output_size(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out_left,
                &mut out_right,
            )
        },
        (SizedConditionalPairJoinType::Left, None) => unsafe {
            cudf_sys::cudf_conditional_left_join_indices_with_output_size(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out_left,
                &mut out_right,
            )
        },
        (SizedConditionalPairJoinType::Inner, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_inner_join_indices_with_output_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (SizedConditionalPairJoinType::Left, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_join_indices_with_output_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", join_type.as_str().to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
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
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let mut out = ptr::null_mut();
    let status = match (semi_join, output_size, exec_options) {
        (true, None, None) => unsafe {
            cudf_sys::cudf_conditional_left_semi_join_indices(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
            )
        },
        (false, None, None) => unsafe {
            cudf_sys::cudf_conditional_left_anti_join_indices(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
            )
        },
        (true, None, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_semi_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                options,
            )
        },
        (false, None, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_anti_join_indices_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                options,
            )
        },
        (true, Some(output_size), None) => unsafe {
            cudf_sys::cudf_conditional_left_semi_join_indices_with_output_size(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out,
            )
        },
        (false, Some(output_size), None) => unsafe {
            cudf_sys::cudf_conditional_left_anti_join_indices_with_output_size(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out,
            )
        },
        (true, Some(output_size), Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_semi_join_indices_with_output_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out,
                options,
            )
        },
        (false, Some(output_size), Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_anti_join_indices_with_output_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                output_size,
                &mut out,
                options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
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
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<usize> {
    let mut out = 0usize;
    let status = match (join_type, exec_options) {
        (SizedConditionalPairJoinType::Inner, None) => unsafe {
            cudf_sys::cudf_conditional_inner_join_size(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
            )
        },
        (SizedConditionalPairJoinType::Left, None) => unsafe {
            cudf_sys::cudf_conditional_left_join_size(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
            )
        },
        (SizedConditionalPairJoinType::Inner, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_inner_join_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                options,
            )
        },
        (SizedConditionalPairJoinType::Left, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_join_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", join_type.as_str().to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;

    Ok(out)
}

fn conditional_single_side_join_size_impl(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    semi_join: bool,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<usize> {
    let mut out = 0usize;
    let status = match (semi_join, exec_options) {
        (true, None) => unsafe {
            cudf_sys::cudf_conditional_left_semi_join_size(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
            )
        },
        (false, None) => unsafe {
            cudf_sys::cudf_conditional_left_anti_join_size(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
            )
        },
        (true, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_semi_join_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                options,
            )
        },
        (false, Some(options)) => unsafe {
            cudf_sys::cudf_conditional_left_anti_join_size_ex(
                left.raw_handle(),
                right.raw_handle(),
                predicate.raw_handle(),
                &mut out,
                options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;

    Ok(out)
}

fn filter_join_kind_raw(join_type: JoinType) -> cudf_sys::cudf_join_kind_t {
    match join_type {
        JoinType::Inner => cudf_sys::cudf_join_kind_t::CUDF_JOIN_KIND_INNER_JOIN,
        JoinType::Left => cudf_sys::cudf_join_kind_t::CUDF_JOIN_KIND_LEFT_JOIN,
        JoinType::Full => cudf_sys::cudf_join_kind_t::CUDF_JOIN_KIND_FULL_JOIN,
    }
}

fn validate_filter_join_index_column(
    indices: &ColumnView<'_>,
    resource: &'static str,
    operation: &'static str,
) -> Result<usize> {
    let dtype = indices.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
    if dtype.id() != TypeId::Int32 {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            format!("{resource} must be Int32, got {:?}", dtype.id()),
        )
        .with_fact(FACT_LEFT_TYPE_ID, dtype.id())
        .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Int32)
        .with_detail("resource", resource)
        .with_detail("reason", "join_index_type_mismatch")
        .with_detail("remedy", "provide INT32 join-index columns produced by cudf-rs join-index APIs"));
    }

    let nullable = indices.nullable().map_err(|e| e.reanchor_if_needed(operation))?;
    if nullable {
        return Err(Error::new_with(ErrorCode::NullUnexpected, operation, format!("{resource} must not be nullable"))
            .with_detail("resource", resource)
            .with_detail("reason", "nullable_join_indices")
            .with_detail("remedy", "use non-nullable INT32 join-index columns produced by cudf-rs join-index APIs"));
    }

    indices.len().map_err(|e| e.reanchor_if_needed(operation))
}

fn validate_filter_join_indices(
    left_indices: &ColumnView<'_>,
    right_indices: &ColumnView<'_>,
    operation: &'static str,
) -> Result<()> {
    let left_len = validate_filter_join_index_column(left_indices, "left_indices", operation)?;
    let right_len = validate_filter_join_index_column(right_indices, "right_indices", operation)?;
    if left_len != right_len {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            format!("left_indices length ({left_len}) does not match right_indices length ({right_len})"),
        )
        .with_fact(FACT_ACTUAL_ROWS, right_len)
        .with_fact(FACT_EXPECTED_ROWS, left_len)
        .with_detail("resource", "join_indices")
        .with_detail("reason", "left_right_index_length_mismatch")
        .with_detail("remedy", "filter index pairs produced by the same equality join result"));
    }
    Ok(())
}

fn filter_join_indices_impl(
    inputs: FilterJoinInputs<'_, '_>,
    join_type: JoinType,
    context: JoinCallContext<'_>,
) -> Result<JoinIndices> {
    validate_filter_join_indices(inputs.left_indices, inputs.right_indices, context.operation)?;

    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let status = match context.exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_filter_join_indices_ex(
                inputs.left.raw_handle(),
                inputs.right.raw_handle(),
                inputs.left_indices.raw_handle(),
                inputs.right_indices.raw_handle(),
                inputs.predicate.raw_handle(),
                filter_join_kind_raw(join_type),
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_filter_join_indices(
                inputs.left.raw_handle(),
                inputs.right.raw_handle(),
                inputs.left_indices.raw_handle(),
                inputs.right_indices.raw_handle(),
                inputs.predicate.raw_handle(),
                filter_join_kind_raw(join_type),
                &mut out_left,
                &mut out_right,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        context.operation,
        [
            ("join_type", format!("{:?}", join_type).to_lowercase()),
            ("execution_context_provided", context.exec_options.is_some().to_string()),
        ],
    )?;

    Ok(JoinIndices {
        left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(context.output_domain)) }?,
        right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(context.output_domain)) }?,
    })
}

fn mixed_join_indices_impl(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    join_type: JoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match (join_type, exec_options) {
        (JoinType::Inner, None) => unsafe {
            cudf_sys::cudf_mixed_inner_join_indices(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Left, None) => unsafe {
            cudf_sys::cudf_mixed_left_join_indices(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Full, None) => unsafe {
            cudf_sys::cudf_mixed_full_join_indices(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Inner, Some(options)) => unsafe {
            cudf_sys::cudf_mixed_inner_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Left, Some(options)) => unsafe {
            cudf_sys::cudf_mixed_left_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Full, Some(options)) => unsafe {
            cudf_sys::cudf_mixed_full_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
    };

    check_join_indices_status(status, null_equality, exec_options.is_some(), operation)?;

    Ok(JoinIndices {
        left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
        right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
    })
}

fn mixed_single_side_join_indices_impl(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    semi_join: bool,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let mut out = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match (semi_join, exec_options) {
        (true, None) => unsafe {
            cudf_sys::cudf_mixed_left_semi_join_indices(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out,
            )
        },
        (false, None) => unsafe {
            cudf_sys::cudf_mixed_left_anti_join_indices(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out,
            )
        },
        (true, Some(options)) => unsafe {
            cudf_sys::cudf_mixed_left_semi_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out,
                options,
            )
        },
        (false, Some(options)) => unsafe {
            cudf_sys::cudf_mixed_left_anti_join_indices_ex(
                tables.left_equality.raw_handle(),
                tables.right_equality.raw_handle(),
                tables.left_conditional.raw_handle(),
                tables.right_conditional.raw_handle(),
                predicate.raw_handle(),
                null_eq_ffi,
                &mut out,
                options,
            )
        },
    };

    check_join_indices_status(status, null_equality, exec_options.is_some(), operation)?;

    Ok(SingleSideJoinIndices { indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }? })
}

/// Inner join: returns only rows with matching keys in both tables.
///
/// `left_keys` and `right_keys` define the join condition.
/// `left_payload` and `right_payload` are additional columns carried through.
/// The result tables contain keys + payload columns concatenated.
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
        JoinCallContext::new(execution.output_domain(), Some(execution.exec_options()), operation),
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
        Some(execution.exec_options()),
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
        JoinCallContext::new(execution.output_domain(), Some(execution.exec_options()), operation),
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
        Some(execution.exec_options()),
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
        Some(execution.exec_options()),
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
        Some(execution.exec_options()),
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
        Some(execution.exec_options()),
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
        Some(execution.exec_options()),
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
    conditional_join_size_impl(left, right, predicate, join_type, Some(execution.exec_options()), operation)
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
    conditional_single_side_join_size_impl(left, right, predicate, semi_join, Some(execution.exec_options()), operation)
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

fn filter_join_indices_with_context(
    inputs: FilterJoinInputs<'_, '_>,
    join_type: JoinType,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let execution = compatible_filter_join_indices_domain(inputs, ctx, operation)?;
    filter_join_indices_impl(
        inputs,
        join_type,
        JoinCallContext::new(execution.output_domain(), Some(execution.exec_options()), operation),
    )
}

/// Filter existing join-index pairs with an AST predicate.
///
/// This is a two-step mixed-join primitive: callers first produce equality
/// join maps, then pass those non-nullable `INT32` index columns here with the
/// conditional tables used by `predicate`. The result is new `INT32` device
/// gather maps only; payload columns are not gathered or copied. `Inner`
/// keeps passing valid pairs, `Left` preserves left rows with right-side
/// no-match sentinels where no predicate-passing match remains, and `Full`
/// preserves raw input-pair semantics by passing through existing no-match
/// pairs and splitting predicate-failing valid pairs into separate no-match
/// sentinel pairs. `Full` filtering does not deduplicate rows already covered
/// by other passing pairs; call [`mixed_full_join_indices`] for libcudf's
/// complete mixed full-join row coverage.
///
/// The output keeps provenance from `left_indices`; same-device table, index,
/// and predicate literal inputs are ordered against the resolved output stream,
/// and cross-device inputs are rejected before calling libcudf. This call
/// enqueues work on the resolved stream and does not materialize host-visible
/// data before returning.
pub fn filter_join_indices(
    left: &TableView<'_>,
    right: &TableView<'_>,
    left_indices: &ColumnView<'_>,
    right_indices: &ColumnView<'_>,
    predicate: &AstExpression,
    join_type: JoinType,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    filter_join_indices_with_context(
        FilterJoinInputs { left, right, left_indices, right_indices, predicate },
        join_type,
        &ctx,
        "join::filter_join_indices",
    )
}

/// Filter existing join-index pairs using a borrowed execution override.
///
/// Same result semantics as [`filter_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn filter_join_indices_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    left_indices: &ColumnView<'_>,
    right_indices: &ColumnView<'_>,
    predicate: &AstExpression,
    join_type: JoinType,
    ctx: &ExecutionOverrides<'_>,
) -> Result<JoinIndices> {
    filter_join_indices_with_context(
        FilterJoinInputs { left, right, left_indices, right_indices, predicate },
        join_type,
        ctx,
        "join::filter_join_indices_in",
    )
}

fn mixed_join_indices_with_context(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    join_type: JoinType,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let execution = compatible_mixed_join_domain(tables, predicate, ctx, operation)?;
    mixed_join_indices_impl(
        tables,
        predicate,
        null_equality,
        join_type,
        execution.output_domain(),
        Some(execution.exec_options()),
        operation,
    )
}

fn mixed_single_side_join_indices_with_context(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    semi_join: bool,
    ctx: &ExecutionOverrides<'_>,
    operation: &'static str,
) -> Result<SingleSideJoinIndices> {
    let execution = compatible_mixed_join_domain(tables, predicate, ctx, operation)?;
    mixed_single_side_join_indices_impl(
        tables,
        predicate,
        null_equality,
        semi_join,
        execution.output_domain(),
        Some(execution.exec_options()),
        operation,
    )
}

/// Mixed inner join indices for equality keys plus an AST predicate.
///
/// Equality tables are joined with `null_equality`; the AST predicate is
/// evaluated against the conditional tables. The returned `INT32` device
/// columns are gather maps only: this operation allocates GPU index columns
/// but does not gather or copy payload columns. The output keeps provenance
/// from `left_equality`; same-device equality, conditional, and predicate
/// literal inputs are ordered against the resolved output stream, and
/// cross-device inputs are rejected before calling libcudf. This call enqueues
/// work on the resolved stream and does not materialize host-visible data
/// before returning.
pub fn mixed_inner_join_indices(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    mixed_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        JoinType::Inner,
        &ctx,
        "join::mixed_inner_join_indices",
    )
}

/// Mixed inner join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_inner_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn mixed_inner_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
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

/// Mixed left join indices preserving all left equality/conditional rows.
///
/// Returned right-side misses contain libcudf's no-match sentinel for
/// nullifying gather. This operation allocates GPU `INT32` gather maps but does
/// not gather or copy payload columns. The output keeps provenance from
/// `left_equality`; same-device equality, conditional, and predicate literal
/// inputs are ordered against the resolved output stream, and cross-device
/// inputs are rejected before calling libcudf. This call enqueues work on the
/// resolved stream and does not materialize host-visible data before returning.
pub fn mixed_left_join_indices(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    mixed_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        JoinType::Left,
        &ctx,
        "join::mixed_left_join_indices",
    )
}

/// Mixed left join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_left_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn mixed_left_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
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

/// Mixed full join indices preserving unmatched rows from both sides.
///
/// Misses on either side contain libcudf's no-match sentinel for nullifying
/// gather. This operation allocates GPU `INT32` gather maps but does not gather
/// or copy payload columns. The output keeps provenance from `left_equality`;
/// same-device equality, conditional, and predicate literal inputs are ordered
/// against the resolved output stream, and cross-device inputs are rejected
/// before calling libcudf. This call enqueues work on the resolved stream and
/// does not materialize host-visible data before returning.
pub fn mixed_full_join_indices(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let ctx = ExecutionOverrides::new();
    mixed_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        JoinType::Full,
        &ctx,
        "join::mixed_full_join_indices",
    )
}

/// Mixed full join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_full_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn mixed_full_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
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

/// Mixed left semi join indices for rows from the left equality/conditional
/// tables with at least one equality-and-predicate match.
///
/// Returns a single `INT32` device gather map. This operation allocates that
/// GPU index column but does not gather or copy payload columns. The output
/// keeps provenance from `left_equality`; same-device equality, conditional,
/// and predicate literal inputs are ordered against the resolved output stream,
/// and cross-device inputs are rejected before calling libcudf. This call
/// enqueues work on the resolved stream and does not materialize host-visible
/// data before returning.
pub fn mixed_left_semi_join_indices(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
) -> Result<SingleSideJoinIndices> {
    let ctx = ExecutionOverrides::new();
    mixed_single_side_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        true,
        &ctx,
        "join::mixed_left_semi_join_indices",
    )
}

/// Mixed left semi join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_left_semi_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn mixed_left_semi_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
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

/// Mixed left anti join indices for rows from the left equality/conditional
/// tables with no equality-and-predicate match.
///
/// Returns a single `INT32` device gather map. This operation allocates that
/// GPU index column but does not gather or copy payload columns. The output
/// keeps provenance from `left_equality`; same-device equality, conditional,
/// and predicate literal inputs are ordered against the resolved output stream,
/// and cross-device inputs are rejected before calling libcudf. This call
/// enqueues work on the resolved stream and does not materialize host-visible
/// data before returning.
pub fn mixed_left_anti_join_indices(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
) -> Result<SingleSideJoinIndices> {
    let ctx = ExecutionOverrides::new();
    mixed_single_side_join_indices_with_context(
        MixedJoinTables { left_equality, right_equality, left_conditional, right_conditional },
        predicate,
        null_equality,
        false,
        &ctx,
        "join::mixed_left_anti_join_indices",
    )
}

/// Mixed left anti join indices using a borrowed execution override.
///
/// Same result semantics as [`mixed_left_anti_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub fn mixed_left_anti_join_indices_in(
    left_equality: &TableView<'_>,
    right_equality: &TableView<'_>,
    left_conditional: &TableView<'_>,
    right_conditional: &TableView<'_>,
    predicate: &AstExpression,
    null_equality: NullEquality,
    ctx: &ExecutionOverrides<'_>,
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

// ---------------------------------------------------------------------------
// Low-level join functions
// ---------------------------------------------------------------------------

/// Full (outer) join: all rows from both sides are preserved, with
/// null-padding where there is no match.
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
        JoinCallContext::new(execution.output_domain(), Some(execution.exec_options()), operation),
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
        [("output_rows", output_rows.to_string()), ("execution_context_provided", true.to_string())],
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

struct HashJoinHandle {
    handle: NonNull<c_void>,
    _domain: Arc<ExecutionDomain>,
}

struct FilteredJoinHandle {
    handle: NonNull<c_void>,
    _domain: Arc<ExecutionDomain>,
}

struct FilteredJoinProbeHandle {
    handle: NonNull<c_void>,
    _domain: Arc<ExecutionDomain>,
}

impl fmt::Debug for HashJoinHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoinHandle").field("handle", &self.handle).finish()
    }
}

impl fmt::Debug for FilteredJoinHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredJoinHandle").field("handle", &self.handle).finish()
    }
}

impl fmt::Debug for FilteredJoinProbeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredJoinProbeHandle").field("handle", &self.handle).finish()
    }
}

// SAFETY: libcudf `hash_join` is documented as reusable for repeated and
// possibly parallel probes. The safe wrapper retains the build table so the
// borrowed build-side device memory outlives the handle.
unsafe impl Send for HashJoinHandle {}
unsafe impl Sync for HashJoinHandle {}

// SAFETY: libcudf `filtered_join` is documented as reusable for repeated
// probes. The safe wrapper retains the build table so the borrowed build-side
// device memory outlives the handle.
unsafe impl Send for FilteredJoinHandle {}
unsafe impl Sync for FilteredJoinHandle {}

// SAFETY: The retained probe-state handle owns its device membership buffer and
// no longer borrows the probe table. The wrapper retains the filtered join so
// build-side device memory remains live until the state is dropped.
unsafe impl Send for FilteredJoinProbeHandle {}
unsafe impl Sync for FilteredJoinProbeHandle {}

impl Drop for HashJoinHandle {
    fn drop(&mut self) {
        unsafe { cudf_sys::cudf_hash_join_destroy(self.handle.as_ptr()) };
    }
}

impl Drop for FilteredJoinHandle {
    fn drop(&mut self) {
        unsafe { cudf_sys::cudf_filtered_join_destroy(self.handle.as_ptr()) };
    }
}

impl Drop for FilteredJoinProbeHandle {
    fn drop(&mut self) {
        unsafe { cudf_sys::cudf_filtered_join_probe_destroy(self.handle.as_ptr()) };
    }
}

/// Reusable duplicate-key hash-join build artifact.
///
/// `HashJoin` owns the build key table and a libcudf hash table that borrows
/// that build table's device memory. Probe methods return `INT32` device
/// gather maps; they do not gather payload columns.
///
/// The handle is `Send + Sync`: cloned handles may be probed from multiple
/// host threads while the retained build table stays alive. Each probe still
/// resolves execution provenance from its probe table and returns device
/// columns in that resolved execution domain.
#[derive(Clone)]
pub struct HashJoin {
    handle: Arc<HashJoinHandle>,
    build: Arc<Table>,
    options: HashJoinOptions,
}

/// Reusable filtered-join build artifact for equality left semi/anti probes.
///
/// `FilteredJoin` owns the right/build key table and a libcudf filtered-join
/// hash table that borrows that build table's device memory. Probe methods
/// return `INT32` device gather maps indexing rows in the left/probe table;
/// they do not gather payload columns.
#[derive(Clone)]
pub struct FilteredJoin {
    handle: Arc<FilteredJoinHandle>,
    build: Arc<Table>,
    options: FilteredJoinOptions,
}

/// Retained filtered-join probe state for equality left semi/anti probes.
///
/// `FilteredJoinProbeState` owns the device-side membership map produced by a
/// single filtered-join hash-table probe. It exposes the exact selected row
/// count before the gather map is allocated, then materializes the gather map
/// from retained state without probing the hash table again.
pub struct FilteredJoinProbeState {
    handle: FilteredJoinProbeHandle,
    join: FilteredJoin,
    output_domain: Arc<ExecutionDomain>,
    output_size: usize,
    semi_join: bool,
}

impl fmt::Debug for HashJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoin")
            .field("options", &self.options)
            .field("placement", &self.build.execution_domain().placement())
            .finish()
    }
}

impl fmt::Debug for FilteredJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredJoin")
            .field("options", &self.options)
            .field("placement", &self.build.execution_domain().placement())
            .finish()
    }
}

impl fmt::Debug for FilteredJoinProbeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredJoinProbeState")
            .field("join_type", &if self.semi_join { "left_semi" } else { "left_anti" })
            .field("output_size", &self.output_size)
            .field("placement", &self.output_domain.placement())
            .finish()
    }
}

impl HashJoin {
    /// Return exact device bytes that must be reserved before building a
    /// reusable hash-join artifact from this build key table.
    ///
    /// This is a planning contract for bounded engines. It returns the exact
    /// retained bytes for the reusable hash table and build-key device views,
    /// without building or retaining the hash table. Build keys that may retain
    /// equality-preprocessing buffers, such as nested columns or nullable
    /// string columns with non-empty null payloads, are rejected by libcudf.
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
        Self::pre_build_reservation_size_bytes_for_view(&build_view, options, ctx, operation)
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
                ("execution_context_provided", true.to_string()),
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
                    ("execution_context_provided", true.to_string()),
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
            Some(execution.exec_options()),
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
            Some(execution.exec_options()),
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
        self.probe_size_impl(probe_keys, join_type, Some(execution.exec_options()), operation)
    }

    fn probe_size_impl(
        &self,
        probe_keys: &TableView<'_>,
        join_type: JoinType,
        exec_options: Option<&cudf_sys::cudf_exec_options_t>,
        operation: &'static str,
    ) -> Result<usize> {
        let mut out = 0usize;
        let status = match (join_type, exec_options) {
            (JoinType::Inner, None) => unsafe {
                cudf_sys::cudf_hash_join_inner_join_size(self.handle.handle.as_ptr(), probe_keys.raw_handle(), &mut out)
            },
            (JoinType::Left, None) => unsafe {
                cudf_sys::cudf_hash_join_left_join_size(self.handle.handle.as_ptr(), probe_keys.raw_handle(), &mut out)
            },
            (JoinType::Full, None) => unsafe {
                cudf_sys::cudf_hash_join_full_join_size(self.handle.handle.as_ptr(), probe_keys.raw_handle(), &mut out)
            },
            (JoinType::Inner, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_inner_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    options,
                )
            },
            (JoinType::Left, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_left_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    options,
                )
            },
            (JoinType::Full, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_full_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    options,
                )
            },
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("nullable_join", format!("{:?}", self.options.nullable_join_setting()).to_lowercase()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_context_provided", exec_options.is_some().to_string()),
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
        exec_options: Option<&cudf_sys::cudf_exec_options_t>,
        operation: &'static str,
    ) -> Result<JoinIndices> {
        let mut out_left = ptr::null_mut();
        let mut out_right = ptr::null_mut();
        let status = match (join_type, output_size, exec_options) {
            (JoinType::Inner, None, None) => unsafe {
                cudf_sys::cudf_hash_join_inner_join(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Left, None, None) => unsafe {
                cudf_sys::cudf_hash_join_left_join(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Full, None, None) => unsafe {
                cudf_sys::cudf_hash_join_full_join(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Inner, None, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_inner_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
            (JoinType::Left, None, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_left_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
            (JoinType::Full, None, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_full_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
            (JoinType::Inner, Some(output_size), None) => unsafe {
                cudf_sys::cudf_hash_join_inner_join_with_output_size(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Left, Some(output_size), None) => unsafe {
                cudf_sys::cudf_hash_join_left_join_with_output_size(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Full, Some(output_size), None) => unsafe {
                cudf_sys::cudf_hash_join_full_join_with_output_size(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Inner, Some(output_size), Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_inner_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
            (JoinType::Left, Some(output_size), Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_left_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
            (JoinType::Full, Some(output_size), Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_full_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("nullable_join", format!("{:?}", self.options.nullable_join_setting()).to_lowercase()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_context_provided", exec_options.is_some().to_string()),
                ("output_size_provided", output_size.is_some().to_string()),
            ],
        )?;

        Ok(JoinIndices {
            left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
            right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
        })
    }
}

impl FilteredJoin {
    /// Return exact device bytes that must be reserved before building a
    /// reusable filtered-join artifact from this right-side build key table.
    ///
    /// This is a planning contract for bounded engines. libcudf does not
    /// expose a formula-only filtered-join sizing entry point, so the FFI shim
    /// builds the artifact against an observing memory resource, reports the
    /// retained bytes, and immediately releases it.
    pub fn pre_build_reservation_size_bytes(build: &Table, options: FilteredJoinOptions) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        Self::pre_build_reservation_size_bytes_in(build, options, &ctx)
    }

    /// Return exact device bytes that must be reserved before building a
    /// reusable filtered-join artifact using a borrowed execution override.
    pub fn pre_build_reservation_size_bytes_in(
        build: &Table,
        options: FilteredJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize> {
        Self::pre_build_reservation_size_bytes_with_context(
            build,
            options,
            ctx,
            "filtered_join::pre_build_reservation_size_bytes_in",
        )
    }

    /// Build a reusable filtered-join artifact from an owned right-side build key table.
    pub fn new(build: Table, options: FilteredJoinOptions) -> Result<Self> {
        let ctx = ExecutionOverrides::new();
        Self::new_with_context(build, options, &ctx, "filtered_join::new")
    }

    /// Build a reusable filtered-join artifact using a borrowed execution override.
    pub fn new_in(build: Table, options: FilteredJoinOptions, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        Self::new_with_context(build, options, ctx, "filtered_join::new_in")
    }

    fn pre_build_reservation_size_bytes_with_context(
        build: &Table,
        options: FilteredJoinOptions,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        let build_view = build.view().map_err(|e| e.reanchor_if_needed(operation))?;
        Self::pre_build_reservation_size_bytes_for_view(&build_view, options, ctx, operation)
    }

    pub(crate) fn pre_build_reservation_size_bytes_for_view(
        build_view: &TableView<'_>,
        options: FilteredJoinOptions,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        let execution = ResolvedExecution::unary(build_view.execution_domain(), ctx, operation)?;
        let mut out = 0usize;
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_filtered_join_pre_build_reservation_size_ex(
                    build_view.raw_handle(),
                    options.null_eq().to_join_raw(),
                    options.load_factor_value(),
                    &mut out,
                    execution.exec_options(),
                )
            },
            operation,
            [
                ("null_equality", format!("{:?}", options.null_eq()).to_lowercase()),
                ("load_factor", options.load_factor_value().to_string()),
                ("execution_context_provided", true.to_string()),
                ("reservation_contract", "filtered_join_pre_build_reservation_size_bytes".to_string()),
            ],
        )?;
        Ok(out)
    }

    fn new_with_context(
        build: Table,
        options: FilteredJoinOptions,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let handle = {
            let build_view = build.view().map_err(|e| e.reanchor_if_needed(operation))?;
            let execution = ResolvedExecution::unary(build.execution_domain(), ctx, operation)?;
            let mut handle = ptr::null_mut();
            error::check_status_in_with_details(
                unsafe {
                    cudf_sys::cudf_filtered_join_create_ex(
                        build_view.raw_handle(),
                        options.null_eq().to_join_raw(),
                        options.load_factor_value(),
                        &mut handle,
                        execution.exec_options(),
                    )
                },
                operation,
                [
                    ("null_equality", format!("{:?}", options.null_eq()).to_lowercase()),
                    ("load_factor", options.load_factor_value().to_string()),
                    ("execution_context_provided", true.to_string()),
                ],
            )?;
            FilteredJoinHandle {
                handle: error::require_non_null(handle, operation, "filtered join")?,
                _domain: Arc::clone(execution.output_domain()),
            }
        };

        Ok(Self { handle: Arc::new(handle), build: Arc::new(build), options })
    }

    /// Returns the retained right-side build key table.
    pub fn build_keys(&self) -> &Table {
        &self.build
    }

    /// Returns the options used when this filtered join was built.
    pub fn options(&self) -> FilteredJoinOptions {
        self.options
    }

    /// Returns the configured null equality setting.
    pub fn null_eq(&self) -> NullEquality {
        self.options.null_eq()
    }

    /// Returns the configured hash-table load factor.
    pub fn load_factor_value(&self) -> f64 {
        self.options.load_factor_value()
    }

    /// Returns bytes retained by this reusable filtered-join build artifact.
    ///
    /// This excludes the retained build key table and any per-probe output
    /// index allocations.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_filtered_join_device_allocated_size(self.handle.handle.as_ptr(), &mut out) },
            "filtered_join::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    /// Begin a retained left-semi probe.
    ///
    /// This probes the filtered hash table once and returns a state object
    /// that exposes the exact output size before the gather map is allocated.
    pub fn begin_left_semi_probe(&self, probe_keys: &TableView<'_>) -> Result<FilteredJoinProbeState> {
        let ctx = ExecutionOverrides::new();
        self.begin_probe_with_context(probe_keys, true, &ctx, "filtered_join::begin_left_semi_probe")
    }

    /// Begin a retained left-semi probe using a borrowed execution override.
    pub fn begin_left_semi_probe_in(
        &self,
        probe_keys: &TableView<'_>,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<FilteredJoinProbeState> {
        self.begin_probe_with_context(probe_keys, true, ctx, "filtered_join::begin_left_semi_probe_in")
    }

    /// Begin a retained left-anti probe.
    pub fn begin_left_anti_probe(&self, probe_keys: &TableView<'_>) -> Result<FilteredJoinProbeState> {
        let ctx = ExecutionOverrides::new();
        self.begin_probe_with_context(probe_keys, false, &ctx, "filtered_join::begin_left_anti_probe")
    }

    /// Begin a retained left-anti probe using a borrowed execution override.
    pub fn begin_left_anti_probe_in(
        &self,
        probe_keys: &TableView<'_>,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<FilteredJoinProbeState> {
        self.begin_probe_with_context(probe_keys, false, ctx, "filtered_join::begin_left_anti_probe_in")
    }

    /// Return the exact output row count for a left semi probe.
    ///
    /// This legacy sizing API does not retain probe membership. Bounded
    /// callers that need to reserve state and then materialize indices from
    /// the same probe must use [`Self::begin_left_semi_probe`] instead.
    pub fn left_semi_join_size(&self, probe_keys: &TableView<'_>) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.probe_size_with_context(probe_keys, true, &ctx, "filtered_join::left_semi_join_size")
    }

    /// Return the exact output row count for a left semi probe using a borrowed execution override.
    ///
    /// This legacy sizing API does not retain probe membership. Bounded
    /// callers that need to reserve state and then materialize indices from
    /// the same probe must use [`Self::begin_left_semi_probe_in`] instead.
    pub fn left_semi_join_size_in(&self, probe_keys: &TableView<'_>, ctx: &ExecutionOverrides<'_>) -> Result<usize> {
        self.probe_size_with_context(probe_keys, true, ctx, "filtered_join::left_semi_join_size_in")
    }

    /// Return the exact output row count for a left anti probe.
    ///
    /// This legacy sizing API does not retain probe membership. Bounded
    /// callers that need to reserve state and then materialize indices from
    /// the same probe must use [`Self::begin_left_anti_probe`] instead.
    pub fn left_anti_join_size(&self, probe_keys: &TableView<'_>) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.probe_size_with_context(probe_keys, false, &ctx, "filtered_join::left_anti_join_size")
    }

    /// Return the exact output row count for a left anti probe using a borrowed execution override.
    ///
    /// This legacy sizing API does not retain probe membership. Bounded
    /// callers that need to reserve state and then materialize indices from
    /// the same probe must use [`Self::begin_left_anti_probe_in`] instead.
    pub fn left_anti_join_size_in(&self, probe_keys: &TableView<'_>, ctx: &ExecutionOverrides<'_>) -> Result<usize> {
        self.probe_size_with_context(probe_keys, false, ctx, "filtered_join::left_anti_join_size_in")
    }

    /// Probe with left semi semantics using an exact caller-supplied output row count.
    ///
    /// The supplied size is a validation hint, not a retained-probe bounded
    /// memory contract. Bounded callers must use [`Self::begin_left_semi_probe`]
    /// and materialize indices from the returned probe state.
    pub fn left_semi_join_indices_with_output_size(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
    ) -> Result<SingleSideJoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.probe_indices_with_output_size_in(
            probe_keys,
            true,
            output_size,
            &ctx,
            "filtered_join::left_semi_join_indices_with_output_size",
        )
    }

    /// Probe with left semi semantics using a borrowed execution override and
    /// an exact caller-supplied output row count.
    ///
    /// The supplied size is a validation hint, not a retained-probe bounded
    /// memory contract. Bounded callers must use
    /// [`Self::begin_left_semi_probe_in`] and materialize indices from the
    /// returned probe state.
    pub fn left_semi_join_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<SingleSideJoinIndices> {
        self.probe_indices_with_output_size_in(
            probe_keys,
            true,
            output_size,
            ctx,
            "filtered_join::left_semi_join_indices_with_output_size_in",
        )
    }

    /// Probe with left anti semantics using an exact caller-supplied output row count.
    ///
    /// The supplied size is a validation hint, not a retained-probe bounded
    /// memory contract. Bounded callers must use [`Self::begin_left_anti_probe`]
    /// and materialize indices from the returned probe state.
    pub fn left_anti_join_indices_with_output_size(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
    ) -> Result<SingleSideJoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.probe_indices_with_output_size_in(
            probe_keys,
            false,
            output_size,
            &ctx,
            "filtered_join::left_anti_join_indices_with_output_size",
        )
    }

    /// Probe with left anti semantics using a borrowed execution override and
    /// an exact caller-supplied output row count.
    ///
    /// The supplied size is a validation hint, not a retained-probe bounded
    /// memory contract. Bounded callers must use
    /// [`Self::begin_left_anti_probe_in`] and materialize indices from the
    /// returned probe state.
    pub fn left_anti_join_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<SingleSideJoinIndices> {
        self.probe_indices_with_output_size_in(
            probe_keys,
            false,
            output_size,
            ctx,
            "filtered_join::left_anti_join_indices_with_output_size_in",
        )
    }

    fn probe_size_with_context(
        &self,
        probe_keys: &TableView<'_>,
        semi_join: bool,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        let mut out = 0usize;
        let status = if semi_join {
            unsafe {
                cudf_sys::cudf_filtered_join_left_semi_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    execution.exec_options(),
                )
            }
        } else {
            unsafe {
                cudf_sys::cudf_filtered_join_left_anti_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    execution.exec_options(),
                )
            }
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_context_provided", true.to_string()),
                ("reservation_contract", "filtered_join_exact_output_size".to_string()),
            ],
        )?;
        Ok(out)
    }

    fn probe_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        semi_join: bool,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<SingleSideJoinIndices> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        let mut out = ptr::null_mut();
        let status = if semi_join {
            unsafe {
                cudf_sys::cudf_filtered_join_left_semi_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out,
                    execution.exec_options(),
                )
            }
        } else {
            unsafe {
                cudf_sys::cudf_filtered_join_left_anti_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out,
                    execution.exec_options(),
                )
            }
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_context_provided", true.to_string()),
                ("output_size_provided", true.to_string()),
                ("output_size", output_size.to_string()),
            ],
        )?;

        Ok(SingleSideJoinIndices {
            indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }?,
        })
    }

    fn begin_probe_with_context(
        &self,
        probe_keys: &TableView<'_>,
        semi_join: bool,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<FilteredJoinProbeState> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        let mut handle = ptr::null_mut();
        let status = if semi_join {
            unsafe {
                cudf_sys::cudf_filtered_join_begin_left_semi_probe_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut handle,
                    execution.exec_options(),
                )
            }
        } else {
            unsafe {
                cudf_sys::cudf_filtered_join_begin_left_anti_probe_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut handle,
                    execution.exec_options(),
                )
            }
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_context_provided", true.to_string()),
                ("reservation_contract", "filtered_join_retained_probe_state".to_string()),
            ],
        )?;
        let handle = FilteredJoinProbeHandle {
            handle: error::require_non_null(handle, operation, "filtered join probe state")?,
            _domain: Arc::clone(execution.output_domain()),
        };
        let mut output_size = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_filtered_join_probe_output_size(handle.handle.as_ptr(), &mut output_size) },
            operation,
        )?;
        Ok(FilteredJoinProbeState {
            handle,
            join: self.clone(),
            output_domain: Arc::clone(execution.output_domain()),
            output_size,
            semi_join,
        })
    }
}

impl FilteredJoinProbeState {
    /// Returns the reusable filtered join that produced this probe state.
    pub fn join(&self) -> &FilteredJoin {
        &self.join
    }

    /// Returns the exact selected row count for this probe.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns bytes retained by this probe state.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_filtered_join_probe_device_allocated_size(self.handle.handle.as_ptr(), &mut out) },
            "filtered_join_probe_state::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    /// Return exact bytes that must be reserved for this probe state's output
    /// gather map.
    pub fn gather_map_reservation_size_bytes(&self) -> Result<usize> {
        self.output_size.checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                "filtered_join_probe_state::gather_map_reservation_size_bytes",
                "filtered join gather-map reservation size overflowed usize",
            )
            .with_detail("reason", "size_overflowed_usize")
            .with_detail("output_rows", self.output_size.to_string())
            .with_detail("index_size_bytes", std::mem::size_of::<i32>().to_string())
        })
    }

    /// Materialize selected probe indices from retained membership state.
    pub fn materialize_indices(self) -> Result<SingleSideJoinIndices> {
        let mut out = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_filtered_join_probe_materialize_indices(self.handle.handle.as_ptr(), &mut out) },
            "filtered_join_probe_state::materialize_indices",
            [
                ("join_type", if self.semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("output_size", self.output_size.to_string()),
                ("reservation_contract", "filtered_join_retained_probe_state_materialize_indices".to_string()),
            ],
        )?;
        Ok(SingleSideJoinIndices {
            indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(&self.output_domain)) }?,
        })
    }

    /// Materialize selected probe indices from retained membership state using
    /// a borrowed execution override.
    pub fn materialize_indices_in(self, ctx: &ExecutionOverrides<'_>) -> Result<SingleSideJoinIndices> {
        let operation = "filtered_join_probe_state::materialize_indices_in";
        let execution =
            ResolvedExecution::many(&self.output_domain, [self.join.build.execution_domain()], ctx, operation)?;
        let mut out = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_filtered_join_probe_materialize_indices_ex(
                    self.handle.handle.as_ptr(),
                    &mut out,
                    execution.exec_options(),
                )
            },
            operation,
            [
                ("join_type", if self.semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("output_size", self.output_size.to_string()),
                ("execution_context_provided", true.to_string()),
                ("reservation_contract", "filtered_join_retained_probe_state_materialize_indices".to_string()),
            ],
        )?;
        Ok(SingleSideJoinIndices {
            indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }?,
        })
    }
}
