//! GPU-accelerated binary operations.
//!
//! # Supported types
//!
//! Binary operations work on both numeric and string columns:
//!
//! - **Numeric columns**: All arithmetic (`Add`, `Sub`, `Mul`, `Div`, …),
//!   comparison (`Equal`, `Less`, …), bitwise, and logical operators.
//! - **String columns**: Comparison operators only (`Equal`, `NotEqual`,
//!   `Less`, `Greater`, `LessEqual`, `GreaterEqual`). String comparisons
//!   are lexicographic and case-sensitive. Arithmetic and bitwise operators
//!   on strings will return an error.

use std::ptr;

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorKind, Result},
        scalar::Scalar,
        types::{DataType, TypeId},
    },
    runtime::execution_domain::ExecutionDomain,
};

/// Binary operators matching `cudf::binary_operator` (same integer values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum BinaryOperator {
    /// Addition (`lhs + rhs`).
    Add = 0,
    /// Subtraction (`lhs - rhs`).
    Sub = 1,
    /// Multiplication (`lhs * rhs`).
    Mul = 2,
    /// Integer division (`lhs / rhs`), truncating toward zero.
    Div = 3,
    /// True (floating-point) division — always produces a float result.
    TrueDiv = 4,
    /// Floor division — rounds the quotient toward negative infinity.
    FloorDiv = 5,
    /// C/C++ modulo (`lhs % rhs`). The result has the sign of `lhs`.
    Mod = 6,
    /// Positive modulo — the result is always non-negative (`[0, |rhs|)`).
    Pmod = 7,
    /// Python-style modulo — the result has the sign of `rhs`.
    Pymod = 8,
    /// Exponentiation (`pow(lhs, rhs)`), producing a float result.
    Pow = 9,
    /// Integer exponentiation (`pow(lhs, rhs)`), keeping the integer type.
    IntPow = 10,
    /// Logarithm of `lhs` with base `rhs` (`log_rhs(lhs)`).
    LogBase = 11,
    /// Two-argument arctangent (`atan2(lhs, rhs)`), in radians.
    Atan2 = 12,
    /// Bitwise left shift (`lhs << rhs`).
    ShiftLeft = 13,
    /// Arithmetic (sign-extending) right shift (`lhs >> rhs`).
    ShiftRight = 14,
    /// Logical (zero-filling) right shift (`lhs >>> rhs`).
    ShiftRightUnsigned = 15,
    /// Bitwise AND (`lhs & rhs`).
    BitwiseAnd = 16,
    /// Bitwise OR (`lhs | rhs`).
    BitwiseOr = 17,
    /// Bitwise XOR (`lhs ^ rhs`).
    BitwiseXor = 18,
    /// Logical AND (`lhs && rhs`). Operates on `Bool8` columns.
    LogicalAnd = 19,
    /// Logical OR (`lhs || rhs`). Operates on `Bool8` columns.
    LogicalOr = 20,
    /// Equality comparison (`lhs == rhs`). Null compared to anything is null.
    Equal = 21,
    /// Inequality comparison (`lhs != rhs`). Null compared to anything is null.
    NotEqual = 22,
    /// Less-than comparison (`lhs < rhs`).
    Less = 23,
    /// Greater-than comparison (`lhs > rhs`).
    Greater = 24,
    /// Less-than-or-equal comparison (`lhs <= rhs`).
    LessEqual = 25,
    /// Greater-than-or-equal comparison (`lhs >= rhs`).
    GreaterEqual = 26,
    /// Null-aware equality: `null == null` evaluates to `true` (not null).
    NullEquals = 27,
    /// Null-aware inequality: `null != null` evaluates to `false` (not null).
    NullNotEquals = 28,
    /// Element-wise max, treating nulls as missing (returns the non-null operand).
    NullMax = 29,
    /// Element-wise min, treating nulls as missing (returns the non-null operand).
    NullMin = 30,
}

impl BinaryOperator {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_binary_operator_t {
        match self {
            BinaryOperator::Add => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_ADD,
            BinaryOperator::Sub => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_SUB,
            BinaryOperator::Mul => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_MUL,
            BinaryOperator::Div => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_DIV,
            BinaryOperator::TrueDiv => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_TRUE_DIV,
            BinaryOperator::FloorDiv => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_FLOOR_DIV,
            BinaryOperator::Mod => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_MOD,
            BinaryOperator::Pmod => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_PMOD,
            BinaryOperator::Pymod => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_PYMOD,
            BinaryOperator::Pow => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_POW,
            BinaryOperator::IntPow => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_INT_POW,
            BinaryOperator::LogBase => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_LOG_BASE,
            BinaryOperator::Atan2 => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_ATAN2,
            BinaryOperator::ShiftLeft => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_SHIFT_LEFT,
            BinaryOperator::ShiftRight => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_SHIFT_RIGHT,
            BinaryOperator::ShiftRightUnsigned => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_SHIFT_RIGHT_UNSIGNED,
            BinaryOperator::BitwiseAnd => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_BITWISE_AND,
            BinaryOperator::BitwiseOr => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_BITWISE_OR,
            BinaryOperator::BitwiseXor => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_BITWISE_XOR,
            BinaryOperator::LogicalAnd => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_LOGICAL_AND,
            BinaryOperator::LogicalOr => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_LOGICAL_OR,
            BinaryOperator::Equal => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_EQUAL,
            BinaryOperator::NotEqual => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_NOT_EQUAL,
            BinaryOperator::Less => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_LESS,
            BinaryOperator::Greater => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_GREATER,
            BinaryOperator::LessEqual => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_LESS_EQUAL,
            BinaryOperator::GreaterEqual => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_GREATER_EQUAL,
            BinaryOperator::NullEquals => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_NULL_EQUALS,
            BinaryOperator::NullNotEquals => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_NULL_NOT_EQUALS,
            BinaryOperator::NullMax => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_NULL_MAX,
            BinaryOperator::NullMin => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_NULL_MIN,
        }
    }

    /// Returns `true` if this operator is a comparison operator.
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::Greater
                | BinaryOperator::LessEqual
                | BinaryOperator::GreaterEqual
                | BinaryOperator::NullEquals
                | BinaryOperator::NullNotEquals
        )
    }
}

/// Infer the output `DataType` for a binary operation given the operand types.
///
/// Follows cudf/pandas promotion rules:
/// - Comparison operators always → `Bool8`
/// - Same type, arithmetic → same type
/// - Int + Float → wider float
/// - Float + Float → wider float
/// - Signed int + unsigned int of same width → next wider signed int
/// - Unsupported combinations → `ErrorKind::NotSupported`
pub fn infer_binary_output_type(lhs: TypeId, rhs: TypeId, op: BinaryOperator) -> Result<DataType> {
    let make_err = |msg: &str| {
        Err(Error::new(ErrorKind::NotSupported, msg.to_string())
            .with_operation("binary::infer_binary_output_type")
            .with_context("lhs_type_id", format!("{:?}", lhs).to_lowercase())
            .with_context("rhs_type_id", format!("{:?}", rhs).to_lowercase())
            .with_context("operator", format!("{:?}", op).to_lowercase()))
    };

    // Comparison operators always infer Bool8.
    if op.is_comparison() {
        return Ok(DataType::new(TypeId::Bool8));
    }

    // Arithmetic: string types not allowed
    if lhs == TypeId::String || rhs == TypeId::String {
        return make_err("cannot perform arithmetic on string types");
    }

    // Bool8 + Bool8 → Bool8 (logical/bitwise context)
    if lhs == TypeId::Bool8 && rhs == TypeId::Bool8 {
        return Ok(DataType::new(TypeId::Bool8));
    }

    // Both must be numeric from here
    let lhs_rank = match lhs.numeric_rank() {
        Some(r) => r,
        None => return make_err("unsupported type for binary operation"),
    };
    let rhs_rank = match rhs.numeric_rank() {
        Some(r) => r,
        None => return make_err("unsupported type for binary operation"),
    };

    // Same type → same type
    if lhs == rhs {
        return Ok(DataType::new(lhs));
    }

    // Both float → wider float
    if lhs.is_float() && rhs.is_float() {
        return Ok(DataType::new(if lhs_rank >= rhs_rank { lhs } else { rhs }));
    }

    // One float, one int → the float (float always wins over int)
    if lhs.is_float() && !rhs.is_float() {
        return Ok(DataType::new(lhs));
    }
    if rhs.is_float() && !lhs.is_float() {
        return Ok(DataType::new(rhs));
    }

    // Both signed or both unsigned → wider type
    if (lhs.is_signed_int() && rhs.is_signed_int()) || (lhs.is_unsigned_int() && rhs.is_unsigned_int()) {
        return Ok(DataType::new(if lhs_rank >= rhs_rank { lhs } else { rhs }));
    }

    // Mixed signed/unsigned integer promotion.
    // If the signed type is strictly wider than the unsigned type, keep signed type.
    // Otherwise, promote based on unsigned width to a signed type that can hold it.
    let (signed_id, unsigned_id) = if lhs.is_signed_int() { (lhs, rhs) } else { (rhs, lhs) };

    let signed_width_rank = match signed_id {
        TypeId::Int8 => 0,
        TypeId::Int16 => 1,
        TypeId::Int32 => 2,
        TypeId::Int64 => 3,
        _ => return make_err("unsupported signed integer type"),
    };

    let unsigned_width_rank = match unsigned_id {
        TypeId::UInt8 => 0,
        TypeId::UInt16 => 1,
        TypeId::UInt32 => 2,
        TypeId::UInt64 => 3,
        _ => return make_err("unsupported unsigned integer type"),
    };

    if signed_width_rank > unsigned_width_rank {
        return Ok(DataType::new(signed_id));
    }

    let promoted = match unsigned_width_rank {
        0 => TypeId::Int16, // UInt8  -> Int16
        1 => TypeId::Int32, // UInt16 -> Int32
        2 => TypeId::Int64, // UInt32 -> Int64
        3 => {
            // UInt64 cannot be safely represented by available signed integer types.
            return make_err("cannot promote signed/unsigned integer pair: no wider signed type available");
        },
        _ => return make_err("unsupported type for binary operation"),
    };

    Ok(DataType::new(promoted))
}

/// Binary operation: column op column.
///
/// Works with both numeric and string columns. For strings, only comparison
/// operators (`Equal`, `NotEqual`, `Less`, `Greater`, `LessEqual`,
/// `GreaterEqual`) are supported — arithmetic operators return an error.
pub fn binary_operation_vv(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(lhs.execution_domain());
    ExecutionDomain::require_same_device_for_mixed_inputs(
        lhs.execution_domain(),
        rhs.execution_domain(),
        "binary::binary_operation_vv",
    )?;
    if !ExecutionDomain::same_provenance(rhs.execution_domain(), &output_domain) {
        ExecutionDomain::insert_stream_dependency(
            rhs.execution_domain(),
            &output_domain,
            "binary::binary_operation_vv",
        )?;
    }
    let exec_options = output_domain.explicit_exec_options_raw();
    binary_operation_vv_impl(
        lhs,
        rhs,
        op,
        output_type,
        &output_domain,
        exec_options.as_ref(),
        "binary::binary_operation_vv",
    )
}

pub(crate) fn binary_operation_vv_impl(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
    output_domain: &std::sync::Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle_for("binary::binary_operation_vv")?;
    let mut out = ptr::null_mut();
    let op_ffi = op.to_raw();
    // SAFETY: lhs and rhs handles are valid (NonNull). dtype_handle was
    // returned by a successful to_raw_handle_for call.
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_binary_operation_vv_ex(
                lhs.raw_handle(),
                rhs.raw_handle(),
                op_ffi,
                dtype_handle,
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_binary_operation_vv(lhs.raw_handle(), rhs.raw_handle(), op_ffi, dtype_handle, &mut out)
        },
    };
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in_with(
        status,
        operation,
        [
            ("operator", format!("{:?}", op).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, std::sync::Arc::clone(output_domain)) }
}

/// Binary operation: column op scalar.
///
/// Works with both numeric and string columns. For strings, only comparison
/// operators are supported — arithmetic operators return an error.
pub fn binary_operation_vs(
    lhs: &ColumnView<'_>,
    rhs: &Scalar,
    op: BinaryOperator,
    output_type: &DataType,
) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(lhs.execution_domain());
    ExecutionDomain::insert_stream_dependency(rhs.execution_domain(), &output_domain, "binary::binary_operation_vs")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    binary_operation_vs_impl(
        lhs,
        rhs,
        op,
        output_type,
        &output_domain,
        exec_options.as_ref(),
        "binary::binary_operation_vs",
    )
}

pub(crate) fn binary_operation_vs_impl(
    lhs: &ColumnView<'_>,
    rhs: &Scalar,
    op: BinaryOperator,
    output_type: &DataType,
    output_domain: &std::sync::Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle_for("binary::binary_operation_vs")?;
    let mut out = ptr::null_mut();
    let op_ffi = op.to_raw();
    // SAFETY: lhs (ColumnView) and rhs (Scalar) handles are valid (NonNull).
    // dtype_handle was returned by a successful to_raw_handle_for call.
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_binary_operation_vs_ex(
                lhs.raw_handle(),
                rhs.raw_handle(),
                op_ffi,
                dtype_handle,
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_binary_operation_vs(lhs.raw_handle(), rhs.raw_handle(), op_ffi, dtype_handle, &mut out)
        },
    };
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in_with(
        status,
        operation,
        [
            ("operator", format!("{:?}", op).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, std::sync::Arc::clone(output_domain)) }
}

/// Binary operation: scalar op column.
///
/// Works with both numeric and string columns. For strings, only comparison
/// operators are supported — arithmetic operators return an error.
pub fn binary_operation_sv(
    lhs: &Scalar,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
) -> Result<Column> {
    // Use the column's domain as primary (consistent with binary_operation_vs
    // which also uses the column's domain).
    let output_domain = std::sync::Arc::clone(rhs.execution_domain());
    ExecutionDomain::insert_stream_dependency(lhs.execution_domain(), &output_domain, "binary::binary_operation_sv")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    binary_operation_sv_impl(
        lhs,
        rhs,
        op,
        output_type,
        &output_domain,
        exec_options.as_ref(),
        "binary::binary_operation_sv",
    )
}

fn binary_operation_sv_impl(
    lhs: &Scalar,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
    output_domain: &std::sync::Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle_for("binary::binary_operation_sv")?;
    let mut out = ptr::null_mut();
    let op_ffi = op.to_raw();
    // SAFETY: lhs (Scalar) and rhs (ColumnView) handles are valid (NonNull).
    // dtype_handle was returned by a successful to_raw_handle_for call.
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_binary_operation_sv_ex(
                lhs.raw_handle(),
                rhs.raw_handle(),
                op_ffi,
                dtype_handle,
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_binary_operation_sv(lhs.raw_handle(), rhs.raw_handle(), op_ffi, dtype_handle, &mut out)
        },
    };
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in_with(
        status,
        operation,
        [
            ("operator", format!("{:?}", op).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, std::sync::Arc::clone(output_domain)) }
}
