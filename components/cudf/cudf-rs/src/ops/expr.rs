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
//! let df = DataFrame::from_series([
//!     Series::new::<i32>("x", &[1, 2, 3]).unwrap(),
//!     Series::new::<i32>("y", &[10, 20, 30]).unwrap(),
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
    reduction::{self, ReduceAggregation, ScanAggregation, ScanType},
    replace::{self, FillPolicy},
    strings::{self, StripSide},
    unary::{self, UnaryOperator},
};
use crate::{
    core::{
        column::Column,
        error::Result,
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy},
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
    Numeric(Box<dyn Fn(&Arc<crate::runtime::execution_domain::ExecutionDomain>) -> Result<Scalar> + Send + Sync>),
    /// A string literal.
    String(String),
}

/// Opaque wrapper for literal values inside expressions.
///
/// Construct via [`lit()`]; not user-constructible.
#[derive(Clone)]
pub struct LitValue(Arc<LitValueInner>);

impl LitValue {
    fn to_scalar_in_domain(&self, domain: &Arc<crate::runtime::execution_domain::ExecutionDomain>) -> Result<Scalar> {
        match &*self.0 {
            LitValueInner::Numeric(f) => f(domain),
            LitValueInner::String(s) => Scalar::from_string_in_domain(s, domain),
        }
    }

    fn numeric(
        f: impl Fn(&Arc<crate::runtime::execution_domain::ExecutionDomain>) -> Result<Scalar> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(LitValueInner::Numeric(Box::new(f))))
    }

    fn string(s: String) -> Self {
        Self(Arc::new(LitValueInner::String(s)))
    }
}

// ===========================================================================
// StringOperator enum
// ===========================================================================

/// String operations that can be applied within an [`Expr`].
#[derive(Debug, Clone)]
pub enum StringOperator {
    Lower,
    Upper,
    Strip,
    LStrip,
    RStrip,
    Len,
    ByteCount,
    Contains(String),
    StartsWith(String),
    EndsWith(String),
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
    /// Replace nulls with a scalar value.
    FillNaScalar { input: Box<Expr>, value: LitValue },
    /// Replace nulls with corresponding values from another expression.
    FillNaExpr { input: Box<Expr>, fill: Box<Expr> },
    /// Forward or backward fill of nulls.
    FillNullPolicy { input: Box<Expr>, policy: FillPolicy },
    /// Cumulative (prefix scan) operation.
    Scan { input: Box<Expr>, agg: ScanAggregation },
    /// Broadcast aggregate: reduce to scalar, then broadcast to row count.
    Reduce { input: Box<Expr>, agg: ReduceAggregation },
    /// String operation.
    StringOp { input: Box<Expr>, op: StringOperator },
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

    /// Exponentiation (`self ^ rhs`). Result type follows cuDF `Pow` rules.
    pub fn pow(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Pow, rhs: Box::new(rhs) }
    }

    /// Integer floor division.
    pub fn floor_div(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::FloorDiv, rhs: Box::new(rhs) }
    }

    /// Element-wise absolute value.
    pub fn abs(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Abs, input: Box::new(self) }
    }

    /// Element-wise square root.
    pub fn sqrt(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Sqrt, input: Box::new(self) }
    }

    /// Element-wise ceiling.
    pub fn ceil(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Ceil, input: Box::new(self) }
    }

    /// Element-wise floor.
    pub fn floor(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Floor, input: Box::new(self) }
    }

    // -- Null fill / cumulative builders (Phase 2) -------------------------

    /// Replace null values with a scalar. Name: `fillna` (matches `Series::fillna`).
    pub fn fillna(self, value: impl IntoLit) -> Expr {
        let lit_value = match value.into_expr() {
            Expr::Literal(lv) => lv,
            _ => unreachable!("IntoLit always produces Expr::Literal"),
        };
        Expr::FillNaScalar { input: Box::new(self), value: lit_value }
    }

    /// Replace null values with per-row values from another expression.
    pub fn fillna_with(self, fill: Expr) -> Expr {
        Expr::FillNaExpr { input: Box::new(self), fill: Box::new(fill) }
    }

    /// Forward fill: replace each null with the nearest preceding non-null value.
    pub fn ffill(self) -> Expr {
        Expr::FillNullPolicy { input: Box::new(self), policy: FillPolicy::Preceding }
    }

    /// Backward fill: replace each null with the nearest following non-null value.
    pub fn bfill(self) -> Expr {
        Expr::FillNullPolicy { input: Box::new(self), policy: FillPolicy::Following }
    }

    /// Inclusive cumulative sum.
    pub fn cumsum(self) -> Expr {
        Expr::Scan { input: Box::new(self), agg: ScanAggregation::Sum }
    }

    /// Inclusive cumulative minimum.
    pub fn cummin(self) -> Expr {
        Expr::Scan { input: Box::new(self), agg: ScanAggregation::Min }
    }

    /// Inclusive cumulative maximum.
    pub fn cummax(self) -> Expr {
        Expr::Scan { input: Box::new(self), agg: ScanAggregation::Max }
    }

    /// Inclusive cumulative product.
    pub fn cumprod(self) -> Expr {
        Expr::Scan { input: Box::new(self), agg: ScanAggregation::Product }
    }

    // -- Aggregation builders (Phase 3) ------------------------------------

    /// Broadcast sum: reduce to scalar, then broadcast to all rows.
    pub fn sum(self) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Sum }
    }

    /// Broadcast mean (always Float64).
    pub fn mean(self) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Mean }
    }

    /// Broadcast minimum.
    #[allow(clippy::should_implement_trait)]
    pub fn min(self) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Min }
    }

    /// Broadcast maximum.
    pub fn max(self) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Max }
    }

    /// Broadcast standard deviation with given degrees-of-freedom correction.
    pub fn std(self, ddof: u32) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Std { ddof } }
    }

    /// Broadcast variance with given degrees-of-freedom correction.
    pub fn var(self, ddof: u32) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Variance { ddof } }
    }

    /// Broadcast median (always Float64).
    pub fn median(self) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Median }
    }

    /// Broadcast logical ANY (true if any element is true).
    pub fn any(self) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Any }
    }

    /// Broadcast logical ALL (true if all elements are true).
    pub fn all(self) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::All }
    }

    /// Broadcast product of all elements.
    pub fn product(self) -> Expr {
        Expr::Reduce { input: Box::new(self), agg: ReduceAggregation::Product }
    }

    // -- String builders (Phase 4a) ----------------------------------------

    /// Convert each string element to lowercase.
    pub fn str_lower(self) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::Lower }
    }

    /// Convert each string element to uppercase.
    pub fn str_upper(self) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::Upper }
    }

    /// Strip whitespace from both sides.
    pub fn str_strip(self) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::Strip }
    }

    /// Strip leading whitespace.
    pub fn str_lstrip(self) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::LStrip }
    }

    /// Strip trailing whitespace.
    pub fn str_rstrip(self) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::RStrip }
    }

    /// Character count per string element.
    pub fn str_len(self) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::Len }
    }

    /// Byte count per string element.
    pub fn str_byte_count(self) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::ByteCount }
    }

    /// Literal substring containment test. Returns BOOL8.
    pub fn str_contains(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::Contains(pat.into()) }
    }

    /// Starts-with test. Returns BOOL8.
    pub fn str_startswith(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::StartsWith(pat.into()) }
    }

    /// Ends-with test. Returns BOOL8.
    pub fn str_endswith(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self), op: StringOperator::EndsWith(pat.into()) }
    }

    // -- Name resolution ---------------------------------------------------

    /// Returns the output column name, if determinable.
    ///
    /// - `Alias` always overrides.
    /// - `Column(name)` → `Some(name)`.
    /// - Single-input nodes (transforms) inherit from their inner expression.
    /// - Multi-input nodes (`BinaryOp`, `Literal`) → `None`.
    pub fn output_name(&self) -> Option<&str> {
        match self {
            Expr::Column(name) => Some(name),
            Expr::Alias { name, .. } => Some(name),
            // Single-input nodes inherit from inner expression.
            Expr::Cast { input, .. }
            | Expr::IsNull(input)
            | Expr::IsNotNull(input)
            | Expr::UnaryOp { input, .. }
            | Expr::FillNaScalar { input, .. }
            | Expr::FillNaExpr { input, .. }
            | Expr::FillNullPolicy { input, .. }
            | Expr::Scan { input, .. }
            | Expr::Reduce { input, .. }
            | Expr::StringOp { input, .. } => input.output_name(),
            // Multi-input or no-source nodes.
            Expr::BinaryOp { .. } | Expr::Literal(_) => None,
        }
    }

    // -- Evaluation --------------------------------------------------------

    /// Evaluate this expression against a DataFrame.
    ///
    /// When the expression has a determinable output name, the returned
    /// Series is renamed accordingly.
    pub fn eval(&self, df: &DataFrame) -> Result<Series> {
        let series = match self {
            Expr::Column(name) => df.series(name),

            Expr::Literal(lit_val) => {
                let scalar = lit_val.to_scalar_in_domain(df.execution_domain())?;
                let size = df.num_rows();
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

            Expr::FillNaScalar { input, value } => {
                let series = input.eval(df)?;
                let scalar = value.to_scalar_in_domain(df.execution_domain())?;
                let cv = series.as_column_view()?;
                let col = replace::replace_nulls(&cv, &scalar)?;
                Series::from_column(col)
            },

            Expr::FillNaExpr { input, fill } => {
                let series = input.eval(df)?;
                let fill_series = fill.eval(df)?;
                let cv = series.as_column_view()?;
                let fill_cv = fill_series.as_column_view()?;
                let col = replace::replace_nulls_column(&cv, &fill_cv)?;
                Series::from_column(col)
            },

            Expr::FillNullPolicy { input, policy } => {
                let series = input.eval(df)?;
                let cv = series.as_column_view()?;
                let col = replace::replace_nulls_policy(&cv, *policy)?;
                Series::from_column(col)
            },

            Expr::Scan { input, agg } => {
                let series = input.eval(df)?;
                let cv = series.as_column_view()?;
                let col = reduction::scan(&cv, *agg, ScanType::Inclusive, NullPolicy::Exclude)?;
                Series::from_column(col)
            },

            Expr::Reduce { input, agg } => {
                let series = input.eval(df)?;
                let dt = series.dtype()?;
                let cv = series.as_column_view()?;
                let out_type = crate::frame::series::reduction::infer_reduce_output_type(&dt, agg)?;
                let scalar = reduction::reduce(&cv, agg, &out_type)?;
                let col = Column::from_scalar(&scalar, df.num_rows())?;
                Series::from_column(col)
            },

            Expr::StringOp { input, op } => {
                let series = input.eval(df)?;
                let cv = series.as_column_view()?;
                let col = match op {
                    StringOperator::Lower => strings::to_lower(&cv)?,
                    StringOperator::Upper => strings::to_upper(&cv)?,
                    StringOperator::Strip => {
                        let empty = Scalar::from_string_in_domain("", df.execution_domain())?;
                        strings::strip(&cv, StripSide::Both, &empty)?
                    },
                    StringOperator::LStrip => {
                        let empty = Scalar::from_string_in_domain("", df.execution_domain())?;
                        strings::strip(&cv, StripSide::Left, &empty)?
                    },
                    StringOperator::RStrip => {
                        let empty = Scalar::from_string_in_domain("", df.execution_domain())?;
                        strings::strip(&cv, StripSide::Right, &empty)?
                    },
                    StringOperator::Len => strings::count_characters(&cv)?,
                    StringOperator::ByteCount => strings::count_bytes(&cv)?,
                    StringOperator::Contains(pat) => {
                        let scalar = Scalar::from_string_in_domain(pat, df.execution_domain())?;
                        strings::str_contains(&cv, &scalar)?
                    },
                    StringOperator::StartsWith(pat) => {
                        let scalar = Scalar::from_string_in_domain(pat, df.execution_domain())?;
                        strings::starts_with(&cv, &scalar)?
                    },
                    StringOperator::EndsWith(pat) => {
                        let scalar = Scalar::from_string_in_domain(pat, df.execution_domain())?;
                        strings::ends_with(&cv, &scalar)?
                    },
                };
                Series::from_column(col)
            },
        }?;

        Ok(match self.output_name() {
            Some(name) => series.rename(name),
            None => series,
        })
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
        Expr::Literal(LitValue::numeric(move |domain| Scalar::from_value_in_domain(val, domain)))
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

impl std::ops::Rem for Expr {
    type Output = Expr;

    fn rem(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Mod, rhs: Box::new(rhs) }
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Negate, input: Box::new(self) }
    }
}

impl std::ops::BitAnd for Expr {
    type Output = Expr;

    fn bitand(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::LogicalAnd, rhs: Box::new(rhs) }
    }
}

impl std::ops::BitOr for Expr {
    type Output = Expr;

    fn bitor(self, rhs: Expr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::LogicalOr, rhs: Box::new(rhs) }
    }
}
