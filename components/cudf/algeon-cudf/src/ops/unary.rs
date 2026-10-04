//! GPU-accelerated unary operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView, copy_view_in},
        error::{self, Error, ErrorCode, Result},
        types::{DataType, TypeId},
    },
    frame::series::{Series, reduction::scalar_to_optional_value},
    ops::reduction,
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
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
    /// Ceiling; smallest integer >= x.
    Ceil,
    /// Floor; largest integer <= x.
    Floor,
    /// Absolute value.
    Abs,
    /// Round to nearest integer (half-to-even / banker's rounding).
    Rint,
    /// Population count; number of `1`-bits in the binary representation.
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
            UnaryOperator::Sin => cudf_sys::CUDF_UNARY_SIN,
            UnaryOperator::Cos => cudf_sys::CUDF_UNARY_COS,
            UnaryOperator::Tan => cudf_sys::CUDF_UNARY_TAN,
            UnaryOperator::ArcSin => cudf_sys::CUDF_UNARY_ARCSIN,
            UnaryOperator::ArcCos => cudf_sys::CUDF_UNARY_ARCCOS,
            UnaryOperator::ArcTan => cudf_sys::CUDF_UNARY_ARCTAN,
            UnaryOperator::Sinh => cudf_sys::CUDF_UNARY_SINH,
            UnaryOperator::Cosh => cudf_sys::CUDF_UNARY_COSH,
            UnaryOperator::Tanh => cudf_sys::CUDF_UNARY_TANH,
            UnaryOperator::ArcSinh => cudf_sys::CUDF_UNARY_ARCSINH,
            UnaryOperator::ArcCosh => cudf_sys::CUDF_UNARY_ARCCOSH,
            UnaryOperator::ArcTanh => cudf_sys::CUDF_UNARY_ARCTANH,
            UnaryOperator::Exp => cudf_sys::CUDF_UNARY_EXP,
            UnaryOperator::Log => cudf_sys::CUDF_UNARY_LOG,
            UnaryOperator::Sqrt => cudf_sys::CUDF_UNARY_SQRT,
            UnaryOperator::Cbrt => cudf_sys::CUDF_UNARY_CBRT,
            UnaryOperator::Ceil => cudf_sys::CUDF_UNARY_CEIL,
            UnaryOperator::Floor => cudf_sys::CUDF_UNARY_FLOOR,
            UnaryOperator::Abs => cudf_sys::CUDF_UNARY_ABS,
            UnaryOperator::Rint => cudf_sys::CUDF_UNARY_RINT,
            UnaryOperator::BitCount => cudf_sys::CUDF_UNARY_BIT_COUNT,
            UnaryOperator::BitInvert => cudf_sys::CUDF_UNARY_BIT_INVERT,
            UnaryOperator::Not => cudf_sys::CUDF_UNARY_NOT,
            UnaryOperator::Negate => cudf_sys::CUDF_UNARY_NEGATE,
        }
    }
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
pub(crate) fn is_valid(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = Exec::RETAINED;
    let execution = ResolvedExecution::unary(input.execution_domain(), ctx, "unary::is_valid")?;
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
pub(crate) fn cast(input: &ColumnView<'_>, output_type: &DataType) -> Result<Column> {
    let ctx = Exec::RETAINED;
    let execution = ResolvedExecution::unary(input.execution_domain(), ctx, "unary::cast")?;
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
    if input_type == *output_type {
        return copy_view_in(input, output_domain, exec_options).map_err(|error| error.reanchor(operation));
    }
    validate_float64_to_int64_cast_input(input, &input_type, output_type, output_domain, exec_options, operation)?;
    let dtype_handle = output_type.to_raw_handle_for("unary::cast")?;
    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull). dtype_handle was returned by
    // a successful to_raw_handle_for call.
    let status = unsafe { cudf_sys::cudf_cast_ex(input.raw_handle(), dtype_handle, &mut out, exec_options) };
    let status_result = error::check_status_in_with_details(
        status,
        operation,
        [
            ("input_type_id", format!("{:?}", input_type.id()).to_lowercase()),
            ("input_type_scale", input_type.scale().to_string()),
            ("output_type_id", format!("{:?}", output_type.id()).to_lowercase()),
            ("output_type_scale", output_type.scale().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    );
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    status_result?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

// libcudf's fixed-point scalar uses RMM's default device-allocation alignment.
const DECIMAL_RESCALE_SCALAR_RESERVATION_BYTES: usize = 256;

/// Returns the exact scalar workspace reservation used by libcudf's same-width decimal rescale.
///
/// Returns `None` outside that dispatch and `Some(0)` when the decimal scale is unchanged.
pub fn decimal_rescale_temporary_workspace_reservation_bytes(
    input_type: &DataType,
    output_type: &DataType,
) -> Option<usize> {
    if input_type.id() != output_type.id() || !input_type.id().is_decimal() {
        return None;
    }
    Some(if input_type.scale() == output_type.scale() { 0 } else { DECIMAL_RESCALE_SCALAR_RESERVATION_BYTES })
}

const INT64_EXCLUSIVE_UPPER_BOUND_AS_F64: f64 = 9_223_372_036_854_775_808.0;

fn validate_float64_to_int64_cast_input(
    input: &ColumnView<'_>,
    input_type: &DataType,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<()> {
    if input_type.id() != TypeId::Float64 || output_type.id() != TypeId::Int64 {
        return Ok(());
    }

    let nan_mask = is_nan_impl(input, output_domain, exec_options, operation)?;
    if Series::from_column(nan_mask)?.on(Exec::RETAINED).any_value()? {
        return Err(float64_to_int64_cast_error(operation, "nan"));
    }

    let (min_scalar, max_scalar) = reduction::minmax_impl(input, output_domain, exec_options, operation)?;
    let min_value = scalar_to_optional_value::<f64>(min_scalar)?;
    let max_value = scalar_to_optional_value::<f64>(max_scalar)?;
    if min_value.is_some_and(|value| !value.is_finite() || value < i64::MIN as f64) {
        return Err(float64_to_int64_cast_error(operation, "below_int64_range"));
    }
    if max_value.is_some_and(|value| !value.is_finite() || value >= INT64_EXCLUSIVE_UPPER_BOUND_AS_F64) {
        return Err(float64_to_int64_cast_error(operation, "above_int64_range"));
    }

    Ok(())
}

fn float64_to_int64_cast_error(operation: &'static str, reason: &'static str) -> Error {
    Error::new_with(
        ErrorCode::ValueOutOfRange,
        operation,
        "Float64 to Int64 cast input cannot be represented as a DataFusion-compatible Int64",
    )
    .with_detail("reason", reason)
    .with_detail("source_type_id", "float64")
    .with_detail("target_type_id", "int64")
}

/// Check whether a cast between two data types is supported.
pub fn is_supported_cast(from: &DataType, to: &DataType) -> Result<bool> {
    let from_handle = from.to_raw_handle_for("unary::is_supported_cast")?;
    let to_handle = to.to_raw_handle_for("unary::is_supported_cast")?;
    let mut out: i32 = 0;
    // SAFETY: from_handle and to_handle were returned by successful
    // to_raw_handle_for calls.
    let status = unsafe { cudf_sys::cudf_is_supported_cast(from_handle, to_handle, &mut out) };
    let status_result = error::check_status_in_with_details(
        status,
        "unary::is_supported_cast",
        [
            ("from_type_id", format!("{:?}", from.id()).to_lowercase()),
            ("to_type_id", format!("{:?}", to.id()).to_lowercase()),
        ],
    );
    // SAFETY: Both handles are valid and must be destroyed.
    unsafe {
        cudf_sys::cudf_data_type_destroy(from_handle);
        cudf_sys::cudf_data_type_destroy(to_handle);
    }
    status_result?;
    Ok(out != 0)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::memory::DeviceMemoryResource,
        runtime::{CudaStream, DeviceId, Runtime},
    };

    /// A same-dtype cast with an explicit output domain copies on that
    /// domain. The input budget fits the input once but never twice, so a
    /// copy charged to the input MR is rejected while the output-domain copy
    /// succeeds.
    #[test]
    fn same_dtype_cast_copies_on_the_output_domain() {
        let device = DeviceId::current().expect("test device");
        let input_budget = DeviceMemoryResource::cuda_on(device)
            .expect("input upstream MR")
            .budget_tracking_limit(12 * 1024 * 1024)
            .expect("input budget");
        let input_runtime = Runtime::builder()
            .with_stream(CudaStream::new().expect("input stream"))
            .with_memory_resource(input_budget)
            .build()
            .expect("input runtime");
        let output_runtime = Runtime::builder()
            .with_stream(CudaStream::new().expect("output stream"))
            .with_pool_memory_resource(64 * 1024 * 1024, None)
            .build()
            .expect("output runtime");

        let values: Vec<i32> = (0..2_000_000).collect();
        let series = input_runtime.series(&values).expect("input series");
        let view = series.as_column().view().expect("input view");
        let dtype = view.data_type().expect("input dtype");

        let output_domain = Arc::clone(output_runtime.execution_domain());
        let exec_options = output_domain.enter_execution().expect("output execution");
        let result = cast_impl(&view, &dtype, &output_domain, exec_options.options(), "test::same_dtype_cast")
            .expect("same-dtype cast must copy on the output domain, not the capped input budget");
        assert!(
            ExecutionDomain::same_provenance(result.execution_domain(), &output_domain),
            "cast output must retain the output domain"
        );
        let result_series = Series::from_column(result).expect("result series");
        assert_eq!(result_series.to_vec_strict::<i32>().expect("result values"), values);
    }
}
