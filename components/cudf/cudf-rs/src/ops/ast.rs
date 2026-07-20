//! Advanced libcudf AST expression handles.
//!
//! These expressions are intended for low-level conditional and mixed join
//! predicates. They own the C shim AST handle and retain any Rust child
//! expressions or literal scalars needed to keep libcudf's borrowed AST
//! references valid until the join call completes.

use std::{
    fmt,
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::{
    core::{
        error::{self, Error, ErrorCode, Result},
        scalar::Scalar,
    },
    runtime::execution_domain::ExecutionDomain,
};

/// A libcudf AST operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AstOperator {
    Add,
    Sub,
    Mul,
    Div,
    TrueDiv,
    FloorDiv,
    Mod,
    PyMod,
    Pow,
    Equal,
    NullEqual,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LogicalAnd,
    NullLogicalAnd,
    LogicalOr,
    NullLogicalOr,
    Identity,
    IsNull,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Sinh,
    Cosh,
    Tanh,
    Arcsinh,
    Arccosh,
    Arctanh,
    Exp,
    Log,
    Sqrt,
    Cbrt,
    Ceil,
    Floor,
    Abs,
    Rint,
    BitInvert,
    Not,
    CastToInt64,
    CastToUint64,
    CastToFloat64,
}

impl AstOperator {
    fn to_raw(self) -> cudf_sys::cudf_ast_operator_t {
        use cudf_sys::cudf_ast_operator_t as Raw;
        match self {
            Self::Add => Raw::CUDF_AST_ADD,
            Self::Sub => Raw::CUDF_AST_SUB,
            Self::Mul => Raw::CUDF_AST_MUL,
            Self::Div => Raw::CUDF_AST_DIV,
            Self::TrueDiv => Raw::CUDF_AST_TRUE_DIV,
            Self::FloorDiv => Raw::CUDF_AST_FLOOR_DIV,
            Self::Mod => Raw::CUDF_AST_MOD,
            Self::PyMod => Raw::CUDF_AST_PYMOD,
            Self::Pow => Raw::CUDF_AST_POW,
            Self::Equal => Raw::CUDF_AST_EQUAL,
            Self::NullEqual => Raw::CUDF_AST_NULL_EQUAL,
            Self::NotEqual => Raw::CUDF_AST_NOT_EQUAL,
            Self::Less => Raw::CUDF_AST_LESS,
            Self::Greater => Raw::CUDF_AST_GREATER,
            Self::LessEqual => Raw::CUDF_AST_LESS_EQUAL,
            Self::GreaterEqual => Raw::CUDF_AST_GREATER_EQUAL,
            Self::BitwiseAnd => Raw::CUDF_AST_BITWISE_AND,
            Self::BitwiseOr => Raw::CUDF_AST_BITWISE_OR,
            Self::BitwiseXor => Raw::CUDF_AST_BITWISE_XOR,
            Self::LogicalAnd => Raw::CUDF_AST_LOGICAL_AND,
            Self::NullLogicalAnd => Raw::CUDF_AST_NULL_LOGICAL_AND,
            Self::LogicalOr => Raw::CUDF_AST_LOGICAL_OR,
            Self::NullLogicalOr => Raw::CUDF_AST_NULL_LOGICAL_OR,
            Self::Identity => Raw::CUDF_AST_IDENTITY,
            Self::IsNull => Raw::CUDF_AST_IS_NULL,
            Self::Sin => Raw::CUDF_AST_SIN,
            Self::Cos => Raw::CUDF_AST_COS,
            Self::Tan => Raw::CUDF_AST_TAN,
            Self::Arcsin => Raw::CUDF_AST_ARCSIN,
            Self::Arccos => Raw::CUDF_AST_ARCCOS,
            Self::Arctan => Raw::CUDF_AST_ARCTAN,
            Self::Sinh => Raw::CUDF_AST_SINH,
            Self::Cosh => Raw::CUDF_AST_COSH,
            Self::Tanh => Raw::CUDF_AST_TANH,
            Self::Arcsinh => Raw::CUDF_AST_ARCSINH,
            Self::Arccosh => Raw::CUDF_AST_ARCCOSH,
            Self::Arctanh => Raw::CUDF_AST_ARCTANH,
            Self::Exp => Raw::CUDF_AST_EXP,
            Self::Log => Raw::CUDF_AST_LOG,
            Self::Sqrt => Raw::CUDF_AST_SQRT,
            Self::Cbrt => Raw::CUDF_AST_CBRT,
            Self::Ceil => Raw::CUDF_AST_CEIL,
            Self::Floor => Raw::CUDF_AST_FLOOR,
            Self::Abs => Raw::CUDF_AST_ABS,
            Self::Rint => Raw::CUDF_AST_RINT,
            Self::BitInvert => Raw::CUDF_AST_BIT_INVERT,
            Self::Not => Raw::CUDF_AST_NOT,
            Self::CastToInt64 => Raw::CUDF_AST_CAST_TO_INT64,
            Self::CastToUint64 => Raw::CUDF_AST_CAST_TO_UINT64,
            Self::CastToFloat64 => Raw::CUDF_AST_CAST_TO_FLOAT64,
        }
    }
}

/// An owned AST expression handle for conditional and mixed join predicates.
///
/// Column references are borrowed references to the left or right table passed
/// to a join call. Literal expressions borrow a retained [`Scalar`]. Operation
/// expressions retain their child expressions, so building a larger predicate
/// by value does not leave libcudf with dangling child references.
pub struct AstExpression {
    handle: NonNull<c_void>,
    children: Vec<AstExpression>,
    literals: Vec<Arc<Scalar>>,
}

impl fmt::Debug for AstExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AstExpression").field("handle", &self.handle).finish()
    }
}

// SAFETY: The handle owns an immutable libcudf AST node. Literal scalar owners
// are immutable and retained; AST expressions are only read during join calls.
unsafe impl Send for AstExpression {}
unsafe impl Sync for AstExpression {}

impl AstExpression {
    /// Create a reference to a column in the left table passed to a join call.
    pub fn left_column(column_index: i32) -> Result<Self> {
        Self::column(column_index, cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_LEFT)
    }

    /// Create a reference to a column in the right table passed to a join call.
    pub fn right_column(column_index: i32) -> Result<Self> {
        Self::column(column_index, cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_RIGHT)
    }

    fn column(column_index: i32, table: cudf_sys::cudf_ast_table_reference_t) -> Result<Self> {
        let operation = "ast::column";
        let mut out = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_ast_column_reference_create(column_index, table, &mut out) },
            operation,
            [("column_index", column_index.to_string())],
        )?;
        Self::from_raw(out, Vec::new(), Vec::new(), operation)
    }

    /// Create a literal AST expression from an owned device scalar.
    pub fn literal(scalar: Scalar) -> Result<Self> {
        let operation = "ast::literal";
        let scalar = Arc::new(scalar);
        let mut out = ptr::null_mut();
        error::check_status_in(unsafe { cudf_sys::cudf_ast_literal_create(scalar.raw_handle(), &mut out) }, operation)?;
        Self::from_raw(out, Vec::new(), vec![scalar], operation)
    }

    /// Create a unary operation expression.
    pub fn unary(op: AstOperator, input: AstExpression) -> Result<Self> {
        let operation = "ast::unary";
        let mut out = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_ast_operation_unary_create(op.to_raw(), input.raw_handle(), &mut out) },
            operation,
            [("operator", format!("{op:?}"))],
        )?;
        Self::from_raw(out, vec![input], Vec::new(), operation)
    }

    /// Create a binary operation expression.
    pub fn binary(op: AstOperator, left: AstExpression, right: AstExpression) -> Result<Self> {
        let operation = "ast::binary";
        let mut out = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_ast_operation_binary_create(op.to_raw(), left.raw_handle(), right.raw_handle(), &mut out)
            },
            operation,
            [("operator", format!("{op:?}"))],
        )?;
        Self::from_raw(out, vec![left, right], Vec::new(), operation)
    }

    pub fn less(left: AstExpression, right: AstExpression) -> Result<Self> {
        Self::binary(AstOperator::Less, left, right)
    }

    pub fn less_equal(left: AstExpression, right: AstExpression) -> Result<Self> {
        Self::binary(AstOperator::LessEqual, left, right)
    }

    pub fn greater(left: AstExpression, right: AstExpression) -> Result<Self> {
        Self::binary(AstOperator::Greater, left, right)
    }

    pub fn greater_equal(left: AstExpression, right: AstExpression) -> Result<Self> {
        Self::binary(AstOperator::GreaterEqual, left, right)
    }

    pub fn equal(left: AstExpression, right: AstExpression) -> Result<Self> {
        Self::binary(AstOperator::Equal, left, right)
    }

    pub fn logical_and(left: AstExpression, right: AstExpression) -> Result<Self> {
        Self::binary(AstOperator::LogicalAnd, left, right)
    }

    /// Cast an AST expression to `INT64`.
    pub fn cast_to_int64(input: AstExpression) -> Result<Self> {
        Self::unary(AstOperator::CastToInt64, input)
    }

    /// Cast an AST expression to `UINT64`.
    pub fn cast_to_uint64(input: AstExpression) -> Result<Self> {
        Self::unary(AstOperator::CastToUint64, input)
    }

    /// Cast an AST expression to `FLOAT64`.
    pub fn cast_to_float64(input: AstExpression) -> Result<Self> {
        Self::unary(AstOperator::CastToFloat64, input)
    }

    /// Create a literal-list membership predicate.
    ///
    /// libcudf AST does not expose a dedicated `IN` operator, so this helper
    /// represents membership as an owned AST disjunction of equality
    /// predicates while retaining the shared input and value expressions for
    /// the lifetime of the resulting predicate.
    pub fn is_in_list(input: AstExpression, values: Vec<AstExpression>) -> Result<Self> {
        Self::in_list(input, values, false)
    }

    /// Create a negated literal-list membership predicate.
    pub fn is_not_in_list(input: AstExpression, values: Vec<AstExpression>) -> Result<Self> {
        Self::in_list(input, values, true)
    }

    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub(crate) fn literal_domains<'a>(&'a self, out: &mut Vec<&'a Arc<ExecutionDomain>>) {
        out.extend(self.literals.iter().map(|scalar| scalar.execution_domain()));
        for child in &self.children {
            child.literal_domains(out);
        }
    }

    fn from_raw(
        handle: *mut c_void,
        children: Vec<AstExpression>,
        literals: Vec<Arc<Scalar>>,
        operation: &'static str,
    ) -> Result<Self> {
        Ok(Self { handle: error::require_non_null(handle, operation, "AST expression")?, children, literals })
    }

    fn in_list(input: AstExpression, values: Vec<AstExpression>, negated: bool) -> Result<Self> {
        let operation = "ast::in_list";
        if values.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, operation, "AST IN list requires at least one value")
                .with_detail("resource", "ast_in_list_values"));
        }

        let mut children = Vec::with_capacity(1 + values.len() * 2);
        let mut value_iter = values.iter();
        let first = value_iter.next().expect("empty values were rejected above");
        let mut predicate_handle = Self::binary_raw(AstOperator::Equal, &input, first, operation)?;
        for value in value_iter {
            let equality_handle = Self::binary_raw(AstOperator::Equal, &input, value, operation)?;
            let mut out = ptr::null_mut();
            error::check_status_in_with_details(
                unsafe {
                    cudf_sys::cudf_ast_operation_binary_create(
                        AstOperator::LogicalOr.to_raw(),
                        predicate_handle,
                        equality_handle,
                        &mut out,
                    )
                },
                operation,
                [("operator", "LogicalOr".to_string())],
            )?;
            children.push(Self::from_raw(predicate_handle, Vec::new(), Vec::new(), operation)?);
            children.push(Self::from_raw(equality_handle, Vec::new(), Vec::new(), operation)?);
            predicate_handle = out;
        }

        let handle = if negated {
            let mut out = ptr::null_mut();
            error::check_status_in_with_details(
                unsafe {
                    cudf_sys::cudf_ast_operation_unary_create(AstOperator::Not.to_raw(), predicate_handle, &mut out)
                },
                operation,
                [("operator", "Not".to_string())],
            )?;
            children.push(Self::from_raw(predicate_handle, Vec::new(), Vec::new(), operation)?);
            out
        } else {
            predicate_handle
        };

        children.push(input);
        children.extend(values);
        Self::from_raw(handle, children, Vec::new(), operation)
    }

    fn binary_raw(
        op: AstOperator,
        left: &AstExpression,
        right: &AstExpression,
        operation: &'static str,
    ) -> Result<*mut c_void> {
        let mut out = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_ast_operation_binary_create(op.to_raw(), left.raw_handle(), right.raw_handle(), &mut out)
            },
            operation,
            [("operator", format!("{op:?}"))],
        )?;
        Ok(out)
    }
}

impl Drop for AstExpression {
    fn drop(&mut self) {
        // AST nodes are host-only metadata. The status is intentionally not
        // mapped to CUDA device health, but the ownership-consuming ABI still
        // guarantees that no C++ exception can cross Drop.
        let _ = unsafe { cudf_sys::cudf_ast_expression_destroy(self.handle.as_ptr()) };
    }
}
