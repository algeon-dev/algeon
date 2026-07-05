//! GPU-accelerated unary operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        types::DataType,
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

/// Unary operators matching `cudf::unary_operator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum UnaryOperator {
    /// Trigonometric sine.
    Sin = 0,
    /// Trigonometric cosine.
    Cos,
    /// Trigonometric tangent.
    Tan,
    /// Inverse sine (arcsin).
    ArcSin,
    /// Inverse cosine (arccos).
    ArcCos,
    /// Inverse tangent (arctan).
    ArcTan,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Inverse hyperbolic sine.
    ArcSinh,
    /// Inverse hyperbolic cosine.
    ArcCosh,
    /// Inverse hyperbolic tangent.
    ArcTanh,
    /// Natural exponential (`e^x`).
    Exp,
    /// Natural logarithm (`ln(x)`).
    Log,
    /// Square root.
    Sqrt,
    /// Cube root.
    Cbrt,
    /// Ceiling — smallest integer >= x.
    Ceil,
    /// Floor — largest integer <= x.
    Floor,
    /// Absolute value.
    Abs,
    /// Round to nearest integer (half-to-even / banker's rounding).
    Rint,
    /// Population count — number of `1`-bits in the binary representation.
    BitCount,
    /// Bitwise inversion (flips every bit). For integer types.
    BitInvert,
    /// Logical NOT. For `Bool8` columns: `true` becomes `false` and vice versa.
    Not,
    /// Arithmetic negation (`-x`). For numeric types.
    Negate,
}

impl UnaryOperator {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_unary_operator_t {
        match self {
            UnaryOperator::Sin => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_SIN,
            UnaryOperator::Cos => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_COS,
            UnaryOperator::Tan => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_TAN,
            UnaryOperator::ArcSin => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_ARCSIN,
            UnaryOperator::ArcCos => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_ARCCOS,
            UnaryOperator::ArcTan => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_ARCTAN,
            UnaryOperator::Sinh => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_SINH,
            UnaryOperator::Cosh => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_COSH,
            UnaryOperator::Tanh => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_TANH,
            UnaryOperator::ArcSinh => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_ARCSINH,
            UnaryOperator::ArcCosh => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_ARCCOSH,
            UnaryOperator::ArcTanh => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_ARCTANH,
            UnaryOperator::Exp => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_EXP,
            UnaryOperator::Log => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_LOG,
            UnaryOperator::Sqrt => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_SQRT,
            UnaryOperator::Cbrt => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_CBRT,
            UnaryOperator::Ceil => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_CEIL,
            UnaryOperator::Floor => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_FLOOR,
            UnaryOperator::Abs => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_ABS,
            UnaryOperator::Rint => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_RINT,
            UnaryOperator::BitCount => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_BIT_COUNT,
            UnaryOperator::BitInvert => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_BIT_INVERT,
            UnaryOperator::Not => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_NOT,
            UnaryOperator::Negate => cudf_sys::cudf_unary_operator_t::CUDF_UNARY_NEGATE,
        }
    }
}

/// Apply a unary operation element-wise.
pub fn unary_operation(input: &ColumnView<'_>, op: UnaryOperator) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "unary::unary_operation")?;
    unary_operation_impl(input, op, execution.output_domain(), execution.exec_options(), "unary::unary_operation")
}

pub(crate) fn unary_operation_impl(
    input: &ColumnView<'_>,
    op: UnaryOperator,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let op_ffi = op.to_raw();
    let status = unsafe { cudf_sys::cudf_unary_operation_ex(input.raw_handle(), op_ffi, &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [("operator", format!("{:?}", op).to_lowercase()), ("execution_authority", "retained_domain".to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Round values away from zero at half-way ties.
pub fn round_half_up(input: &ColumnView<'_>, decimal_places: i32) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "unary::round_half_up")?;
    round_half_up_impl(
        input,
        decimal_places,
        execution.output_domain(),
        execution.exec_options(),
        "unary::round_half_up",
    )
}

pub(crate) fn round_half_up_impl(
    input: &ColumnView<'_>,
    decimal_places: i32,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_round_half_up_ex(input.raw_handle(), decimal_places, &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("decimal_places", decimal_places.to_string()),
            ("rounding_method", "half_up".to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Return a boolean column with each element logically negated.
///
/// Input must be Bool8. Equivalent to `unary_operation(input, UnaryOperator::Not)`.
pub fn logical_not(input: &ColumnView<'_>) -> Result<Column> {
    unary_operation(input, UnaryOperator::Not)
}

/// Return a boolean column: `true` where elements are null.
pub fn is_null(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "unary::is_null")?;
    is_null_impl(input, execution.output_domain(), execution.exec_options(), "unary::is_null")
}

pub(crate) fn is_null_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_is_null_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Return a boolean column: `true` where elements are valid (non-null).
pub fn is_valid(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "unary::is_valid")?;
    is_valid_impl(input, execution.output_domain(), execution.exec_options(), "unary::is_valid")
}

pub(crate) fn is_valid_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_is_valid_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Cast a column to the specified data type.
pub fn cast(input: &ColumnView<'_>, output_type: &DataType) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "unary::cast")?;
    cast_impl(input, output_type, execution.output_domain(), execution.exec_options(), "unary::cast")
}

pub(crate) fn cast_impl(
    input: &ColumnView<'_>,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let input_type = input.data_type()?;
    let dtype_handle = output_type.to_raw_handle_for("unary::cast")?;
    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull). dtype_handle was returned by
    // a successful to_raw_handle_for call.
    let status = unsafe { cudf_sys::cudf_cast_ex(input.raw_handle(), dtype_handle, &mut out, exec_options) };
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("input_type_id", format!("{:?}", input_type.id()).to_lowercase()),
            ("input_type_scale", input_type.scale().to_string()),
            ("output_type_id", format!("{:?}", output_type.id()).to_lowercase()),
            ("output_type_scale", output_type.scale().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Check whether a cast between two data types is supported.
pub fn is_supported_cast(from: &DataType, to: &DataType) -> Result<bool> {
    let from_handle = from.to_raw_handle_for("unary::is_supported_cast")?;
    let to_handle = to.to_raw_handle_for("unary::is_supported_cast")?;
    let mut out: i32 = 0;
    // SAFETY: from_handle and to_handle were returned by successful
    // to_raw_handle_for calls.
    let status = unsafe { cudf_sys::cudf_is_supported_cast(from_handle, to_handle, &mut out) };
    // SAFETY: Both handles are valid and must be destroyed.
    unsafe {
        cudf_sys::cudf_data_type_destroy(from_handle);
        cudf_sys::cudf_data_type_destroy(to_handle);
    }
    error::check_status_in_with_details(
        status,
        "unary::is_supported_cast",
        [
            ("from_type_id", format!("{:?}", from.id()).to_lowercase()),
            ("to_type_id", format!("{:?}", to.id()).to_lowercase()),
        ],
    )?;
    Ok(out != 0)
}

/// Return a boolean column: `true` where elements are NaN.
pub fn is_nan(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "unary::is_nan")?;
    is_nan_impl(input, execution.output_domain(), execution.exec_options(), "unary::is_nan")
}

pub(crate) fn is_nan_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_is_nan_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Return a boolean column: `true` where elements are not NaN.
pub fn is_not_nan(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "unary::is_not_nan")?;
    is_not_nan_impl(input, execution.output_domain(), execution.exec_options(), "unary::is_not_nan")
}

pub(crate) fn is_not_nan_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_is_not_nan_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
