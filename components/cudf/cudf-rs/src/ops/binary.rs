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

use std::{ptr, sync::Arc};

use super::unary;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        scalar::Scalar,
        types::{DataType, TypeId},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
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
    /// Null-aware logical AND (Spark 3-valued logic): `false && null` is `false`,
    /// `true && null` is `null`. Operates on `Bool8` columns.
    ///
    /// Discriminant 31 (`GENERIC_BINARY`) is intentionally skipped: it requires
    /// user-provided PTX and is not exposed here, but the gap keeps these
    /// discriminants aligned with `cudf::binary_operator`.
    NullLogicalAnd = 32,
    /// Null-aware logical OR (Spark 3-valued logic): `true || null` is `true`,
    /// `false || null` is `null`. Operates on `Bool8` columns.
    NullLogicalOr = 33,
}

impl BinaryOperator {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            BinaryOperator::Add => "add",
            BinaryOperator::Sub => "sub",
            BinaryOperator::Mul => "mul",
            BinaryOperator::Div => "div",
            BinaryOperator::TrueDiv => "true_div",
            BinaryOperator::FloorDiv => "floor_div",
            BinaryOperator::Mod => "mod",
            BinaryOperator::Pmod => "pmod",
            BinaryOperator::Pymod => "pymod",
            BinaryOperator::Pow => "pow",
            BinaryOperator::IntPow => "int_pow",
            BinaryOperator::LogBase => "log_base",
            BinaryOperator::Atan2 => "atan2",
            BinaryOperator::ShiftLeft => "shift_left",
            BinaryOperator::ShiftRight => "shift_right",
            BinaryOperator::ShiftRightUnsigned => "shift_right_unsigned",
            BinaryOperator::BitwiseAnd => "bitwise_and",
            BinaryOperator::BitwiseOr => "bitwise_or",
            BinaryOperator::BitwiseXor => "bitwise_xor",
            BinaryOperator::LogicalAnd => "logical_and",
            BinaryOperator::LogicalOr => "logical_or",
            BinaryOperator::Equal => "equal",
            BinaryOperator::NotEqual => "not_equal",
            BinaryOperator::Less => "less",
            BinaryOperator::Greater => "greater",
            BinaryOperator::LessEqual => "less_equal",
            BinaryOperator::GreaterEqual => "greater_equal",
            BinaryOperator::NullEquals => "null_equals",
            BinaryOperator::NullNotEquals => "null_not_equals",
            BinaryOperator::NullMax => "null_max",
            BinaryOperator::NullMin => "null_min",
            BinaryOperator::NullLogicalAnd => "null_logical_and",
            BinaryOperator::NullLogicalOr => "null_logical_or",
        }
    }

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
            BinaryOperator::NullLogicalAnd => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_NULL_LOGICAL_AND,
            BinaryOperator::NullLogicalOr => cudf_sys::cudf_binary_operator_t::CUDF_BINARY_NULL_LOGICAL_OR,
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

/// Infer the output `DataType` for a binary operation given type ids only.
///
/// Decimal scale cannot be represented by [`TypeId`], so decimal callers should
/// prefer [`infer_binary_output_dtype`].
pub fn infer_binary_output_type(lhs: TypeId, rhs: TypeId, op: BinaryOperator) -> Result<DataType> {
    infer_binary_output_dtype_with_operation(
        DataType::new(lhs),
        DataType::new(rhs),
        op,
        "binary::infer_binary_output_type",
    )
}

/// Infer the output `DataType` for a binary operation given full operand dtypes.
///
/// This preserves decimal scale and models cudf-compatible temporal rules before
/// falling back to primitive numeric promotion:
/// - comparison operators return `Bool8` for comparable families.
/// - decimal arithmetic keeps cudf fixed-point output scale rules.
/// - timestamp/duration arithmetic requires matching resolution.
/// - primitive numeric arithmetic follows cudf-rs rank promotion.
pub fn infer_binary_output_dtype(lhs: DataType, rhs: DataType, op: BinaryOperator) -> Result<DataType> {
    infer_binary_output_dtype_with_operation(lhs, rhs, op, "binary::infer_binary_output_dtype")
}

pub(crate) fn infer_binary_output_dtype_in(
    lhs: DataType,
    rhs: DataType,
    op: BinaryOperator,
    operation: &'static str,
) -> Result<DataType> {
    infer_binary_output_dtype_with_operation(lhs, rhs, op, operation)
}

fn infer_binary_output_dtype_with_operation(
    lhs: DataType,
    rhs: DataType,
    op: BinaryOperator,
    operation: &'static str,
) -> Result<DataType> {
    let lhs_id = lhs.id();
    let rhs_id = rhs.id();
    let make_err = |msg: &str| {
        Err(Error::new_with(ErrorCode::TypeMismatch, operation, msg.to_string())
            .with_fact(FACT_LEFT_TYPE_ID, lhs_id)
            .with_fact(FACT_RIGHT_TYPE_ID, rhs_id)
            .with_detail("operator", op.name()))
    };

    if op.is_comparison() {
        if comparable_types(lhs_id, rhs_id) {
            return Ok(DataType::new(TypeId::Bool8));
        }
        return make_err("types are not comparable for binary operation");
    }

    if lhs_id.is_decimal() || rhs_id.is_decimal() {
        return infer_decimal_binary_output(lhs, rhs, op, operation, make_err);
    }

    if lhs_id.is_timestamp() || rhs_id.is_timestamp() || lhs_id.is_duration() || rhs_id.is_duration() {
        return infer_temporal_binary_output(lhs_id, rhs_id, op, make_err);
    }

    infer_primitive_binary_output(lhs_id, rhs_id, op, make_err)
}

fn comparable_types(lhs: TypeId, rhs: TypeId) -> bool {
    if lhs == rhs {
        return true;
    }
    if lhs.is_decimal() && rhs.is_decimal() {
        return true;
    }
    if lhs.is_timestamp() && rhs.is_timestamp() {
        return true;
    }
    if lhs.is_duration() && rhs.is_duration() {
        return true;
    }
    lhs.is_numeric() && rhs.is_numeric()
}

fn infer_decimal_binary_output(
    lhs: DataType,
    rhs: DataType,
    op: BinaryOperator,
    operation: &'static str,
    make_err: impl Fn(&str) -> Result<DataType>,
) -> Result<DataType> {
    let lhs_id = lhs.id();
    let rhs_id = rhs.id();
    if !(lhs_id.is_decimal() && rhs_id.is_decimal()) {
        return make_err("decimal binary operations require decimal operands; decimal-float coercion is not implicit");
    }
    if lhs_id != rhs_id {
        return make_err("decimal binary operations require matching decimal storage types");
    }

    let scale = match op {
        BinaryOperator::Add
        | BinaryOperator::Sub
        | BinaryOperator::Mod
        | BinaryOperator::Pmod
        | BinaryOperator::Pymod
        | BinaryOperator::NullMax
        | BinaryOperator::NullMin => lhs.scale().min(rhs.scale()),
        BinaryOperator::Mul => lhs.scale().checked_add(rhs.scale()).ok_or_else(|| {
            Error::new_with(ErrorCode::TypeMismatch, operation, "decimal multiplication output scale overflows i32")
                .with_fact(FACT_LEFT_TYPE_ID, lhs_id)
                .with_fact(FACT_RIGHT_TYPE_ID, rhs_id)
                .with_detail("operator", op.name())
        })?,
        BinaryOperator::Div => lhs.scale().checked_sub(rhs.scale()).ok_or_else(|| {
            Error::new_with(ErrorCode::TypeMismatch, operation, "decimal division output scale overflows i32")
                .with_fact(FACT_LEFT_TYPE_ID, lhs_id)
                .with_fact(FACT_RIGHT_TYPE_ID, rhs_id)
                .with_detail("operator", op.name())
        })?,
        _ => return make_err("unsupported decimal binary operator"),
    };

    Ok(DataType::fixed_point(lhs_id, scale))
}

fn infer_temporal_binary_output(
    lhs: TypeId,
    rhs: TypeId,
    op: BinaryOperator,
    make_err: impl Fn(&str) -> Result<DataType>,
) -> Result<DataType> {
    match op {
        BinaryOperator::Add => {
            if lhs.is_timestamp() && rhs.is_duration() && lhs.timestamp_resolution() == rhs.duration_resolution() {
                return Ok(DataType::new(lhs));
            }
            if lhs.is_duration() && rhs.is_timestamp() && lhs.duration_resolution() == rhs.timestamp_resolution() {
                return Ok(DataType::new(rhs));
            }
            if lhs.is_duration() && rhs.is_duration() && lhs.duration_resolution() == rhs.duration_resolution() {
                return Ok(DataType::new(lhs));
            }
        },
        BinaryOperator::Sub => {
            if lhs.is_timestamp() && rhs.is_duration() && lhs.timestamp_resolution() == rhs.duration_resolution() {
                return Ok(DataType::new(lhs));
            }
            if lhs.is_timestamp() && rhs.is_timestamp() && lhs.timestamp_resolution() == rhs.timestamp_resolution() {
                return Ok(DataType::new(lhs.timestamp_resolution().unwrap().duration_type_id()));
            }
            if lhs.is_duration() && rhs.is_duration() && lhs.duration_resolution() == rhs.duration_resolution() {
                return Ok(DataType::new(lhs));
            }
        },
        BinaryOperator::Mul => {
            // libcudf compiled chrono operators accept integral duration
            // multipliers; floating multipliers are rejected before FFI.
            if lhs.is_duration() && rhs.is_integral() {
                return Ok(DataType::new(lhs));
            }
            if lhs.is_integral() && rhs.is_duration() {
                return Ok(DataType::new(rhs));
            }
        },
        BinaryOperator::Div
            // libcudf compiled chrono operators accept integral duration
            // divisors; floating divisors are rejected before FFI.
            if lhs.is_duration() && rhs.is_integral() => {
                return Ok(DataType::new(lhs));
            },
        BinaryOperator::Mod => {
            if lhs.is_duration() && rhs.is_integral() {
                return Ok(DataType::new(lhs));
            }
            if lhs.is_duration() && rhs.is_duration() && lhs.duration_resolution() == rhs.duration_resolution() {
                return Ok(DataType::new(lhs));
            }
        },
        BinaryOperator::Pymod
            if lhs.is_duration() && rhs.is_duration() && lhs.duration_resolution() == rhs.duration_resolution() => {
                return Ok(DataType::new(lhs));
            },
        BinaryOperator::Pmod => {},
        _ => {},
    }

    make_err("unsupported timestamp or duration binary operation")
}

fn infer_primitive_binary_output(
    lhs: TypeId,
    rhs: TypeId,
    op: BinaryOperator,
    make_err: impl Fn(&str) -> Result<DataType>,
) -> Result<DataType> {
    // Arithmetic: string types not allowed
    if lhs == TypeId::String || rhs == TypeId::String {
        return make_err("cannot perform arithmetic on string types");
    }

    if matches!(
        op,
        BinaryOperator::LogicalAnd
            | BinaryOperator::LogicalOr
            | BinaryOperator::NullLogicalAnd
            | BinaryOperator::NullLogicalOr
    ) {
        if lhs == TypeId::Bool8 && rhs == TypeId::Bool8 {
            return Ok(DataType::new(TypeId::Bool8));
        }
        return make_err("logical binary operators require Bool8 operands");
    }

    if matches!(op, BinaryOperator::BitwiseAnd | BinaryOperator::BitwiseOr | BinaryOperator::BitwiseXor)
        && lhs == TypeId::Bool8
        && rhs == TypeId::Bool8
    {
        return Ok(DataType::new(TypeId::Bool8));
    }

    // Bool8 + Bool8 → Bool8 (legacy bitwise/logical context)
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
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(lhs.execution_domain(), rhs.execution_domain(), &ctx, "binary::binary_operation_vv")?;
    binary_operation_vv_impl(
        lhs,
        rhs,
        op,
        output_type,
        execution.output_domain(),
        execution.exec_options(),
        "binary::binary_operation_vv",
    )
}

pub(crate) fn binary_operation_vv_impl(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let lhs_decimal_division_cast;
    let rhs_decimal_division_cast;
    let lhs_decimal_division_view;
    let rhs_decimal_division_view;
    let (lhs, rhs) = match decimal_division_explicit_type_operand_targets(lhs, rhs, op, output_type, operation)? {
        Some((lhs_target, rhs_target)) => {
            lhs_decimal_division_cast =
                cast_decimal_division_operand(lhs, lhs_target, output_domain, exec_options, operation)?;
            rhs_decimal_division_cast =
                cast_decimal_division_operand(rhs, rhs_target, output_domain, exec_options, operation)?;
            let lhs = if let Some(column) = lhs_decimal_division_cast.as_ref() {
                lhs_decimal_division_view = column.view()?;
                &lhs_decimal_division_view
            } else {
                lhs
            };
            let rhs = if let Some(column) = rhs_decimal_division_cast.as_ref() {
                rhs_decimal_division_view = column.view()?;
                &rhs_decimal_division_view
            } else {
                rhs
            };
            (lhs, rhs)
        },
        None => (lhs, rhs),
    };
    let dtype_handle = output_type.to_raw_handle_for(operation)?;
    let mut out = ptr::null_mut();
    let op_ffi = op.to_raw();
    // SAFETY: lhs and rhs handles are valid (NonNull). dtype_handle was
    // returned by a successful to_raw_handle_for call.
    let status = unsafe {
        cudf_sys::cudf_binary_operation_vv_ex(
            lhs.raw_handle(),
            rhs.raw_handle(),
            op_ffi,
            dtype_handle,
            &mut out,
            exec_options,
        )
    };
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("operator", op.name().to_string()),
            ("output_type_id", output_type.id().to_string()),
            ("output_type_scale", output_type.scale().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

fn decimal_division_explicit_type_operand_targets(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
    operation: &'static str,
) -> Result<Option<(DataType, DataType)>> {
    if op != BinaryOperator::Div || !output_type.id().is_decimal() {
        return Ok(None);
    }

    let lhs_type = lhs.data_type()?;
    let rhs_type = rhs.data_type()?;
    if !(lhs_type.id().is_decimal() && rhs_type.id().is_decimal()) {
        return Ok(None);
    }

    // libcudf fixed-point DIV divides the raw representations and labels the
    // output with the requested scale. To make a caller-selected result scale
    // affect the numeric quotient, preserve the RHS value and pre-scale LHS so
    // raw_lhs / raw_rhs is already in the requested output scale.
    let lhs_scale = rhs_type.scale().checked_add(output_type.scale()).ok_or_else(|| {
        Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "decimal division explicit output scale requires operand scale adjustment that overflows i32",
        )
        .with_fact(FACT_LEFT_TYPE_ID, lhs_type.id())
        .with_fact(FACT_RIGHT_TYPE_ID, rhs_type.id())
        .with_detail("operator", op.name())
        .with_detail("output_type_id", output_type.id().to_string())
        .with_detail("output_type_scale", output_type.scale().to_string())
        .with_detail("rhs_type_scale", rhs_type.scale().to_string())
    })?;

    Ok(Some((
        DataType::fixed_point(output_type.id(), lhs_scale),
        DataType::fixed_point(output_type.id(), rhs_type.scale()),
    )))
}

fn decimal_division_explicit_type_column_scalar_targets(
    lhs: &ColumnView<'_>,
    rhs: &Scalar,
    op: BinaryOperator,
    output_type: &DataType,
    operation: &'static str,
) -> Result<Option<(DataType, DataType)>> {
    if op != BinaryOperator::Div || !output_type.id().is_decimal() {
        return Ok(None);
    }

    let lhs_type = lhs.data_type()?;
    let rhs_type = rhs.data_type()?;
    if !(lhs_type.id().is_decimal() && rhs_type.id().is_decimal()) {
        return Ok(None);
    }

    let lhs_scale = adjusted_decimal_division_lhs_scale(
        rhs_type.scale(),
        output_type,
        lhs_type.id(),
        rhs_type.id(),
        op,
        operation,
    )?;
    Ok(Some((
        DataType::fixed_point(output_type.id(), lhs_scale),
        DataType::fixed_point(output_type.id(), rhs_type.scale()),
    )))
}

fn decimal_division_explicit_type_scalar_column_targets(
    lhs: &Scalar,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
    operation: &'static str,
) -> Result<Option<(DataType, DataType)>> {
    if op != BinaryOperator::Div || !output_type.id().is_decimal() {
        return Ok(None);
    }

    let lhs_type = lhs.data_type()?;
    let rhs_type = rhs.data_type()?;
    if !(lhs_type.id().is_decimal() && rhs_type.id().is_decimal()) {
        return Ok(None);
    }

    let lhs_scale = adjusted_decimal_division_lhs_scale(
        rhs_type.scale(),
        output_type,
        lhs_type.id(),
        rhs_type.id(),
        op,
        operation,
    )?;
    Ok(Some((
        DataType::fixed_point(output_type.id(), lhs_scale),
        DataType::fixed_point(output_type.id(), rhs_type.scale()),
    )))
}

fn adjusted_decimal_division_lhs_scale(
    rhs_scale: i32,
    output_type: &DataType,
    lhs_type_id: TypeId,
    rhs_type_id: TypeId,
    op: BinaryOperator,
    operation: &'static str,
) -> Result<i32> {
    rhs_scale.checked_add(output_type.scale()).ok_or_else(|| {
        Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "decimal division explicit output scale requires operand scale adjustment that overflows i32",
        )
        .with_fact(FACT_LEFT_TYPE_ID, lhs_type_id)
        .with_fact(FACT_RIGHT_TYPE_ID, rhs_type_id)
        .with_detail("operator", op.name())
        .with_detail("output_type_id", output_type.id().to_string())
        .with_detail("output_type_scale", output_type.scale().to_string())
        .with_detail("rhs_type_scale", rhs_scale.to_string())
    })
}

fn cast_decimal_division_operand(
    input: &ColumnView<'_>,
    target: DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Option<Column>> {
    if input.data_type()? == target {
        return Ok(None);
    }

    unary::cast_impl(input, &target, output_domain, exec_options, operation).map(Some).map_err(|error| {
        error
            .with_detail_if_absent("operator", BinaryOperator::Div.name())
            .with_detail_if_absent("decimal_division_operand_target_type_id", target.id().to_string())
            .with_detail_if_absent("decimal_division_operand_target_scale", target.scale().to_string())
            .with_detail_if_absent("reason", "explicit_decimal_division_operand_scale_adjustment")
    })
}

fn rescale_decimal_division_scalar(
    input: &Scalar,
    target: DataType,
    output_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<Option<Scalar>> {
    let source = input.data_type()?;
    if source == target && ExecutionDomain::same_provenance(input.execution_domain(), output_domain) {
        return Ok(None);
    }
    if !input.is_valid()? {
        return Scalar::null_decimal_in_domain(target, output_domain).map(Some).map_err(|error| {
            error
                .with_detail_if_absent("operator", BinaryOperator::Div.name())
                .with_detail_if_absent("decimal_division_scalar_target_type_id", target.id().to_string())
                .with_detail_if_absent("decimal_division_scalar_target_scale", target.scale().to_string())
                .with_detail_if_absent("reason", "explicit_decimal_division_scalar_scale_adjustment")
        });
    }

    let raw = decimal_scalar_raw_value(input, source, operation)?;
    let raw = rescale_decimal_raw(raw, source.scale(), target.scale(), operation).map_err(|error| {
        error
            .with_detail_if_absent("operator", BinaryOperator::Div.name())
            .with_detail_if_absent("decimal_division_scalar_target_type_id", target.id().to_string())
            .with_detail_if_absent("decimal_division_scalar_target_scale", target.scale().to_string())
            .with_detail_if_absent("reason", "explicit_decimal_division_scalar_scale_adjustment")
    })?;
    Scalar::decimal_from_raw_in_domain(raw, target, output_domain).map(Some).map_err(|error| {
        error
            .with_detail_if_absent("operator", BinaryOperator::Div.name())
            .with_detail_if_absent("decimal_division_scalar_target_type_id", target.id().to_string())
            .with_detail_if_absent("decimal_division_scalar_target_scale", target.scale().to_string())
            .with_detail_if_absent("reason", "explicit_decimal_division_scalar_scale_adjustment")
    })
}

fn decimal_scalar_raw_value(input: &Scalar, source: DataType, operation: &'static str) -> Result<i128> {
    match source.id() {
        TypeId::Decimal32 => input.decimal32_raw_value().map(i128::from),
        TypeId::Decimal64 => input.decimal64_raw_value().map(i128::from),
        TypeId::Decimal128 => input.decimal128_raw_value(),
        other => Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "decimal division scalar scale adjustment requires decimal scalar input",
        )
        .with_fact(FACT_LEFT_TYPE_ID, other)
        .with_detail("operator", BinaryOperator::Div.name())),
    }
}

fn rescale_decimal_raw(raw: i128, source_scale: i32, target_scale: i32, operation: &'static str) -> Result<i128> {
    let delta = source_scale.checked_sub(target_scale).ok_or_else(|| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "decimal scalar scale adjustment overflows i32")
            .with_detail("source_scale", source_scale.to_string())
            .with_detail("target_scale", target_scale.to_string())
    })?;
    if delta == 0 {
        return Ok(raw);
    }

    let factor = pow10_i128(delta.unsigned_abs(), operation)?;
    if delta > 0 {
        raw.checked_mul(factor).ok_or_else(|| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "decimal scalar raw value overflows target scale")
                .with_detail("source_scale", source_scale.to_string())
                .with_detail("target_scale", target_scale.to_string())
        })
    } else {
        if raw % factor != 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                operation,
                "decimal scalar raw value cannot be represented exactly at target scale",
            )
            .with_detail("source_scale", source_scale.to_string())
            .with_detail("target_scale", target_scale.to_string()));
        }
        Ok(raw / factor)
    }
}

fn pow10_i128(exponent: u32, operation: &'static str) -> Result<i128> {
    let mut value = 1i128;
    for _ in 0..exponent {
        value = value.checked_mul(10).ok_or_else(|| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "decimal scale adjustment factor overflows i128")
                .with_detail("exponent", exponent.to_string())
        })?;
    }
    Ok(value)
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
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(lhs.execution_domain(), rhs.execution_domain(), &ctx, "binary::binary_operation_vs")?;
    binary_operation_vs_impl(
        lhs,
        rhs,
        op,
        output_type,
        execution.output_domain(),
        execution.exec_options(),
        "binary::binary_operation_vs",
    )
}

pub(crate) fn binary_operation_vs_impl(
    lhs: &ColumnView<'_>,
    rhs: &Scalar,
    op: BinaryOperator,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let lhs_decimal_division_cast;
    let lhs_decimal_division_view;
    let rhs_decimal_division_scalar;
    let (lhs, rhs) = match decimal_division_explicit_type_column_scalar_targets(lhs, rhs, op, output_type, operation)? {
        Some((lhs_target, rhs_target)) => {
            lhs_decimal_division_cast =
                cast_decimal_division_operand(lhs, lhs_target, output_domain, exec_options, operation)?;
            rhs_decimal_division_scalar = rescale_decimal_division_scalar(rhs, rhs_target, output_domain, operation)?;
            let lhs = if let Some(column) = lhs_decimal_division_cast.as_ref() {
                lhs_decimal_division_view = column.view()?;
                &lhs_decimal_division_view
            } else {
                lhs
            };
            let rhs = if let Some(scalar) = rhs_decimal_division_scalar.as_ref() { scalar } else { rhs };
            (lhs, rhs)
        },
        None => (lhs, rhs),
    };
    let dtype_handle = output_type.to_raw_handle_for(operation)?;
    let mut out = ptr::null_mut();
    let op_ffi = op.to_raw();
    // SAFETY: lhs (ColumnView) and rhs (Scalar) handles are valid (NonNull).
    // dtype_handle was returned by a successful to_raw_handle_for call.
    let status = unsafe {
        cudf_sys::cudf_binary_operation_vs_ex(
            lhs.raw_handle(),
            rhs.raw_handle(),
            op_ffi,
            dtype_handle,
            &mut out,
            exec_options,
        )
    };
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("operator", op.name().to_string()),
            ("output_type_id", output_type.id().to_string()),
            ("output_type_scale", output_type.scale().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
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
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(rhs.execution_domain(), lhs.execution_domain(), &ctx, "binary::binary_operation_sv")?;
    binary_operation_sv_impl(
        lhs,
        rhs,
        op,
        output_type,
        execution.output_domain(),
        execution.exec_options(),
        "binary::binary_operation_sv",
    )
}

fn binary_operation_sv_impl(
    lhs: &Scalar,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let lhs_decimal_division_scalar;
    let rhs_decimal_division_cast;
    let rhs_decimal_division_view;
    let (lhs, rhs) = match decimal_division_explicit_type_scalar_column_targets(lhs, rhs, op, output_type, operation)? {
        Some((lhs_target, rhs_target)) => {
            lhs_decimal_division_scalar = rescale_decimal_division_scalar(lhs, lhs_target, output_domain, operation)?;
            rhs_decimal_division_cast =
                cast_decimal_division_operand(rhs, rhs_target, output_domain, exec_options, operation)?;
            let lhs = if let Some(scalar) = lhs_decimal_division_scalar.as_ref() { scalar } else { lhs };
            let rhs = if let Some(column) = rhs_decimal_division_cast.as_ref() {
                rhs_decimal_division_view = column.view()?;
                &rhs_decimal_division_view
            } else {
                rhs
            };
            (lhs, rhs)
        },
        None => (lhs, rhs),
    };
    let dtype_handle = output_type.to_raw_handle_for(operation)?;
    let mut out = ptr::null_mut();
    let op_ffi = op.to_raw();
    // SAFETY: lhs (Scalar) and rhs (ColumnView) handles are valid (NonNull).
    // dtype_handle was returned by a successful to_raw_handle_for call.
    let status = unsafe {
        cudf_sys::cudf_binary_operation_sv_ex(
            lhs.raw_handle(),
            rhs.raw_handle(),
            op_ffi,
            dtype_handle,
            &mut out,
            exec_options,
        )
    };
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("operator", op.name().to_string()),
            ("output_type_id", output_type.id().to_string()),
            ("output_type_scale", output_type.scale().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
