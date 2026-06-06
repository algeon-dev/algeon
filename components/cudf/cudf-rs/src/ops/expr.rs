//! Expression DSL for declarative column operations.
//!
//! Provides a cuDF-Python / Polars-style expression API where users compose
//! column operations as an AST and evaluate them against a [`DataFrame`].
//!
//! # Example
//!
//! ```no_run
//! use cudf::{
//!     DataFrame, Expr, Series,
//!     ops::expr::{col, lit},
//! };
//!
//! let df = DataFrame::from_columns([
//!     ("x", Series::new::<i32>("x", &[1, 2, 3]).unwrap()),
//!     ("y", Series::new::<i32>("y", &[10, 20, 30]).unwrap()),
//! ])
//! .unwrap();
//!
//! let df2 = df
//!     .with_columns([
//!         (col("x") + col("y")).alias("sum"),
//!         (col("x") * lit(2i32)).alias("doubled"),
//!     ])
//!     .unwrap();
//! ```

use std::sync::Arc;

use super::{
    binary::{self, BinaryOperator},
    unary::{self, UnaryOperator},
};
use crate::{
    core::{
        column::Column,
        error::Result,
        scalar::Scalar,
        types::{CudfType, DataType},
    },
    frame::{dataframe::DataFrame, series::Series},
};

// ===========================================================================
// LitValue — type-erased scalar factory (crate-private)
// ===========================================================================

/// A type-erased factory for creating a [`Scalar`] from a literal value.
///
/// Not part of the public API — use [`lit()`] to create literals.
enum LitValueInner {
    /// A numeric scalar factory. The closure creates the Scalar on demand.
    Numeric(Box<dyn Fn() -> Result<Scalar> + Send + Sync>),
    /// A string literal.
    String(String),
}

/// Opaque wrapper for literal values inside expressions.
///
/// Construct via [`lit()`]; not user-constructible.
#[derive(Clone)]
pub struct LitValue(Arc<LitValueInner>);

impl LitValue {
    fn to_scalar(&self) -> Result<Scalar> {
        match &*self.0 {
            LitValueInner::Numeric(f) => f(),
            LitValueInner::String(s) => Scalar::from_string(s),
        }
    }

    fn numeric(f: impl Fn() -> Result<Scalar> + Send + Sync + 'static) -> Self {
        Self(Arc::new(LitValueInner::Numeric(Box::new(f))))
    }

    fn string(s: String) -> Self {
        Self(Arc::new(LitValueInner::String(s)))
    }
}

// ===========================================================================
// Expr enum
// ===========================================================================

/// A declarative expression over DataFrame columns.
///
/// Expressions form a tree that is evaluated eagerly against a DataFrame via
/// [`Expr::eval`]. No optimization is performed — each node is evaluated
/// bottom-up.
#[derive(Clone)]
pub enum Expr {
    /// Reference to a named column.
    Column(String),
    /// A broadcast literal scalar value. Construct via [`lit()`].
    Literal(LitValue),
    /// Element-wise binary operation.
    BinaryOp { lhs: Box<Expr>, op: BinaryOperator, rhs: Box<Expr> },
    /// Element-wise unary operation.
    UnaryOp { op: UnaryOperator, input: Box<Expr> },
    /// Cast to a target type.
    Cast { input: Box<Expr>, target: DataType },
    /// Null-check: produces BOOL8 where `true` = null.
    IsNull(Box<Expr>),
    /// Validity-check: produces BOOL8 where `true` = non-null.
    IsNotNull(Box<Expr>),
    /// Rename the output of an inner expression.
    Alias { inner: Box<Expr>, name: String },
}

impl Expr {
    // -- Builder methods ---------------------------------------------------

    /// Rename this expression's output column.
    pub fn alias(self, name: impl Into<String>) -> Expr {
        Expr::Alias { inner: Box::new(self), name: name.into() }
    }

    /// Cast the result to the given type.
    ///
    /// Accepts either a [`DataType`] (canonical, supports decimal scale) or a
    /// [`crate::TypeId`] (convenience shorthand, scale defaults to 0).
    pub fn cast(self, target: impl Into<DataType>) -> Expr {
        Expr::Cast { input: Box::new(self), target: target.into() }
    }

    /// Element-wise greater-than comparison.
    pub fn gt(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Greater, rhs: Box::new(rhs) }
    }

    /// Element-wise less-than comparison.
    pub fn lt(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Less, rhs: Box::new(rhs) }
    }

    /// Element-wise equality comparison.
    pub fn eq(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Equal, rhs: Box::new(rhs) }
    }

    /// Element-wise inequality comparison.
    pub fn ne(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::NotEqual, rhs: Box::new(rhs) }
    }

    /// Element-wise greater-than-or-equal comparison.
    pub fn ge(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::GreaterEqual, rhs: Box::new(rhs) }
    }

    /// Element-wise less-than-or-equal comparison.
    pub fn le(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::LessEqual, rhs: Box::new(rhs) }
    }

    /// Element-wise null check. Produces BOOL8.
    pub fn is_null(self) -> Expr {
        Expr::IsNull(Box::new(self))
    }

    /// Element-wise validity check. Produces BOOL8.
    pub fn is_not_null(self) -> Expr {
        Expr::IsNotNull(Box::new(self))
    }

    /// Logical NOT (explicit method, equivalent to `!expr`).
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Not, input: Box::new(self) }
    }

    // -- Name resolution ---------------------------------------------------

    /// Returns the output column name, if determinable.
    ///
    /// - `Column(name)` → `Some(name)`
    /// - `Alias { name, .. }` → `Some(name)`
    /// - everything else → `None`
    pub fn output_name(&self) -> Option<&str> {
        match self {
            Expr::Column(name) => Some(name),
            Expr::Alias { name, .. } => Some(name),
            _ => None,
        }
    }

    // -- Evaluation --------------------------------------------------------

    /// Evaluate this expression against a DataFrame, producing a named Series.
    pub fn eval(&self, df: &DataFrame) -> Result<Series> {
        match self {
            Expr::Column(name) => df.series(name),

            Expr::Literal(lit_val) => {
                let scalar = lit_val.to_scalar()?;
                let size = df.num_rows() as i32;
                let col = Column::from_scalar(&scalar, size)?;
                Series::from_column(col)
            },

            Expr::BinaryOp { lhs, op, rhs } => {
                let left = lhs.eval(df)?;
                let right = rhs.eval(df)?;
                let lv = left.as_column_view()?;
                let rv = right.as_column_view()?;
                let out_type = binary::infer_binary_output_type(lv.data_type()?.id(), rv.data_type()?.id(), *op)?;
                let col = binary::binary_operation_vv(&lv, &rv, *op, &out_type)?;
                Series::from_column(col)
            },

            Expr::UnaryOp { op, input } => {
                let series = input.eval(df)?;
                let cv = series.as_column_view()?;
                let col = unary::unary_operation(&cv, *op)?;
                Series::from_column(col)
            },

            Expr::Cast { input, target } => {
                let series = input.eval(df)?;
                series.astype(*target)
            },

            Expr::IsNull(input) => {
                let series = input.eval(df)?;
                series.is_null()
            },

            Expr::IsNotNull(input) => {
                let series = input.eval(df)?;
                series.is_not_null()
            },

            Expr::Alias { inner, name } => {
                let series = inner.eval(df)?;
                Ok(series.rename(name))
            },
        }
    }
}

// ===========================================================================
// Free functions: col(), lit()
// ===========================================================================

/// Create an expression referencing a named column.
pub fn col(name: impl Into<String>) -> Expr {
    Expr::Column(name.into())
}

/// Create a literal expression that broadcasts the value to all rows.
///
/// Supports all [`CudfType`] numeric types, `&str`, and `String`.
pub fn lit<L: IntoLit>(value: L) -> Expr {
    value.into_expr()
}

// ===========================================================================
// IntoLit trait
// ===========================================================================

/// Trait for types that can be converted to a literal [`Expr`].
pub trait IntoLit {
    /// Convert this value into a literal `Expr`.
    fn into_expr(self) -> Expr;
}

// Blanket impl for all CudfType (numeric) types.
impl<T: CudfType + Send + Sync> IntoLit for T {
    fn into_expr(self) -> Expr {
        let val = self;
        Expr::Literal(LitValue::numeric(move || Scalar::from_value(val)))
    }
}

impl IntoLit for &str {
    fn into_expr(self) -> Expr {
        Expr::Literal(LitValue::string(self.to_string()))
    }
}

impl IntoLit for String {
    fn into_expr(self) -> Expr {
        Expr::Literal(LitValue::string(self))
    }
}

// ===========================================================================
// Operator overloads: Add, Sub, Mul, Div, Not
// ===========================================================================

impl std::ops::Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Add, rhs: Box::new(rhs) }
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Sub, rhs: Box::new(rhs) }
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Mul, rhs: Box::new(rhs) }
    }
}

impl std::ops::Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Div, rhs: Box::new(rhs) }
    }
}

impl std::ops::Not for Expr {
    type Output = Expr;

    fn not(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Not, input: Box::new(self) }
    }
}
