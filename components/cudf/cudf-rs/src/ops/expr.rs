//! Expression DSL for declarative column operations.
//!
//! Provides a cuDF-Python / Polars-style expression API where users compose
//! column operations as an AST and evaluate them against a [`DataFrame`].
//!
//! # Reduction Shape
//!
//! Expr reductions accessed through [`Expr::agg`] are **broadcast
//! aggregates**. They reduce their input to a single scalar internally, then
//! broadcast that scalar back to the DataFrame row count so the result remains
//! row-aligned and can participate in larger expressions.
//!
//! That means `col("x").agg().sum()` is appropriate in expressions like
//! `col("x") / col("x").agg().sum()`, but broadcast reductions do not
//! inherit an output name for
//! [`DataFrame::with_columns`](crate::frame::dataframe::DataFrame::with_columns).
//! Use [`.alias()`](Expr::alias) or an explicit `(name, expr)` pair when you
//! want to materialize the broadcast result as a named column.
//!
//! If you want an actual scalar reduction, use the [`Series`] reduction APIs
//! such as [`Series::sum`] or [`Series::sum_value`] instead.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{
//!     DataFrame, Expr, Runtime,
//!     ops::expr::{col, lit},
//! };
//!
//! let rt = Runtime::new().unwrap();
//! let x = rt.series(&[1i32, 2, 3]).unwrap();
//! let y = rt.series(&[10i32, 20, 30]).unwrap();
//! let df = DataFrame::from_columns([("x", x), ("y", y)]).unwrap();
//!
//! let df2 = df
//!     .with_columns([
//!         (col("x") + col("y")).alias("sum"),
//!         (col("x") * 2i32).alias("doubled"),
//!         (col("x") / col("x").agg().sum()).alias("share_of_total"),
//!     ])
//!     .unwrap();
//! ```

use std::sync::Arc;

use super::{
    binary::{self, BinaryOperator},
    copying,
    reduction::{self, ReduceAgg, ScanAgg, ScanType},
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
    pub(crate) fn to_scalar_in_domain(
        &self,
        domain: &Arc<crate::runtime::execution_domain::ExecutionDomain>,
    ) -> Result<Scalar> {
        match &*self.0 {
            LitValueInner::Numeric(f) => f(domain),
            LitValueInner::String(s) => Scalar::from_str_in_domain(s, domain),
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
    /// Assign a schema/output name to an inner expression.
    Alias { inner: Box<Expr>, name: String },
    /// Replace nulls with a scalar value.
    FillNaScalar { input: Box<Expr>, value: LitValue },
    /// Replace nulls with corresponding values from another expression.
    FillNaExpr { input: Box<Expr>, fill: Box<Expr> },
    /// Forward or backward fill of nulls.
    FillNullPolicy { input: Box<Expr>, policy: FillPolicy },
    /// Cumulative (prefix scan) operation.
    Scan { input: Box<Expr>, agg: ScanAgg },
    /// Broadcast aggregate: reduce to scalar, then broadcast to row count.
    Reduce { input: Box<Expr>, agg: ReduceAgg },
    /// String operation.
    StringOp { input: Box<Expr>, op: StringOperator },
    /// Conditional: row-wise `if cond then true_val else false_val`.
    ///
    /// Construct via [`when()`] builder chain. Output name is `None` — an
    /// explicit `.alias()` is required for use in `with_columns`.
    IfElse { condition: Box<Expr>, true_val: Box<Expr>, false_val: Box<Expr> },
}

impl Expr {
    // -- Builder methods ---------------------------------------------------

    /// Assign a schema/output name used by DataFrame materialization APIs.
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

    /// Enter the aggregation namespace for broadcast aggregate expressions.
    ///
    /// Methods on the returned namespace reduce once, then broadcast the
    /// result back to the frame row count so it can participate in larger
    /// row-aligned expressions.
    pub fn agg(self) -> ExprAggNamespace {
        ExprAggNamespace(self)
    }

    /// Element-wise greater-than comparison.
    pub fn gt(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Greater, rhs: Box::new(rhs.into_expr()) }
    }

    /// Element-wise less-than comparison.
    pub fn lt(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Less, rhs: Box::new(rhs.into_expr()) }
    }

    /// Element-wise equality comparison.
    pub fn eq(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Equal, rhs: Box::new(rhs.into_expr()) }
    }

    /// Element-wise inequality comparison.
    pub fn ne(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::NotEqual, rhs: Box::new(rhs.into_expr()) }
    }

    /// Element-wise greater-than-or-equal comparison.
    pub fn ge(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::GreaterEqual, rhs: Box::new(rhs.into_expr()) }
    }

    /// Element-wise less-than-or-equal comparison.
    pub fn le(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::LessEqual, rhs: Box::new(rhs.into_expr()) }
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
    pub fn pow(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Pow, rhs: Box::new(rhs.into_expr()) }
    }

    /// Integer floor division.
    pub fn floor_div(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::FloorDiv, rhs: Box::new(rhs.into_expr()) }
    }

    /// Element-wise absolute value.
    pub fn abs(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Abs, input: Box::new(self) }
    }

    /// Element-wise natural logarithm.
    pub fn log(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Log, input: Box::new(self) }
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
        Expr::Scan { input: Box::new(self), agg: ScanAgg::Sum }
    }

    /// Inclusive cumulative minimum.
    pub fn cummin(self) -> Expr {
        Expr::Scan { input: Box::new(self), agg: ScanAgg::Min }
    }

    /// Inclusive cumulative maximum.
    pub fn cummax(self) -> Expr {
        Expr::Scan { input: Box::new(self), agg: ScanAgg::Max }
    }

    /// Inclusive cumulative product.
    pub fn cumprod(self) -> Expr {
        Expr::Scan { input: Box::new(self), agg: ScanAgg::Product }
    }

    // -- String builders (Phase 4a) ----------------------------------------

    /// Enter the string namespace for string-specific expression methods.
    pub fn str(self) -> ExprStringNamespace {
        ExprStringNamespace(self)
    }

    // -- Name resolution ---------------------------------------------------

    /// Returns the output column name, if determinable.
    ///
    /// - `Alias` always overrides.
    /// - `Column(name)` → `Some(name)`.
    /// - Single-input nodes (transforms) inherit from their inner expression.
    /// - Broadcast reductions do not inherit a name; use `.alias()`.
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
            | Expr::StringOp { input, .. } => input.output_name(),
            Expr::Reduce { .. } => None,
            // Multi-input or no-source nodes.
            Expr::BinaryOp { .. } | Expr::Literal(_) | Expr::IfElse { .. } => None,
        }
    }

    // -- Evaluation --------------------------------------------------------

    /// Evaluate this expression against a DataFrame.
    ///
    /// Expression naming is schema metadata handled by DataFrame-level APIs.
    /// The returned [`Series`] itself is unnamed.
    pub fn eval(&self, df: &DataFrame) -> Result<Series> {
        match self {
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
                let lv = left.view()?;
                let rv = right.view()?;
                let out_type = binary::infer_binary_output_type(lv.data_type()?.id(), rv.data_type()?.id(), *op)?;
                let col = binary::binary_operation_vv(&lv, &rv, *op, &out_type)?;
                Series::from_column(col)
            },

            Expr::UnaryOp { op, input } => {
                let series = input.eval(df)?;
                let cv = series.view()?;
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
                let _ = name;
                inner.eval(df)
            },

            Expr::FillNaScalar { input, value } => {
                let series = input.eval(df)?;
                let scalar = value.to_scalar_in_domain(df.execution_domain())?;
                let cv = series.view()?;
                let col = replace::replace_nulls(&cv, &scalar)?;
                Series::from_column(col)
            },

            Expr::FillNaExpr { input, fill } => {
                let series = input.eval(df)?;
                let fill_series = fill.eval(df)?;
                let cv = series.view()?;
                let fill_cv = fill_series.view()?;
                let col = replace::replace_nulls_column(&cv, &fill_cv)?;
                Series::from_column(col)
            },

            Expr::FillNullPolicy { input, policy } => {
                let series = input.eval(df)?;
                let cv = series.view()?;
                let col = replace::replace_nulls_policy(&cv, *policy)?;
                Series::from_column(col)
            },

            Expr::Scan { input, agg } => {
                let series = input.eval(df)?;
                let cv = series.view()?;
                let col = reduction::scan(&cv, *agg, ScanType::Inclusive, NullPolicy::Exclude)?;
                Series::from_column(col)
            },

            Expr::Reduce { input, agg } => {
                let series = input.eval(df)?;
                let dt = series.dtype()?;
                let cv = series.view()?;
                let out_type = crate::frame::series::reduction::infer_reduce_output_type(&dt, agg, "expr::reduce")?;
                let scalar = reduction::reduce(&cv, agg, &out_type)?;
                let col = Column::from_scalar(&scalar, df.num_rows())?;
                Series::from_column(col)
            },

            Expr::StringOp { input, op } => {
                let series = input.eval(df)?;
                let cv = series.view()?;
                let col = match op {
                    StringOperator::Lower => strings::to_lower(&cv)?,
                    StringOperator::Upper => strings::to_upper(&cv)?,
                    StringOperator::Strip => {
                        let empty = Scalar::from_str_in_domain("", df.execution_domain())?;
                        strings::strip(&cv, StripSide::Both, &empty)?
                    },
                    StringOperator::LStrip => {
                        let empty = Scalar::from_str_in_domain("", df.execution_domain())?;
                        strings::strip(&cv, StripSide::Left, &empty)?
                    },
                    StringOperator::RStrip => {
                        let empty = Scalar::from_str_in_domain("", df.execution_domain())?;
                        strings::strip(&cv, StripSide::Right, &empty)?
                    },
                    StringOperator::Len => strings::count_characters(&cv)?,
                    StringOperator::ByteCount => strings::count_bytes(&cv)?,
                    StringOperator::Contains(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, df.execution_domain())?;
                        strings::str_contains(&cv, &scalar)?
                    },
                    StringOperator::StartsWith(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, df.execution_domain())?;
                        strings::starts_with(&cv, &scalar)?
                    },
                    StringOperator::EndsWith(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, df.execution_domain())?;
                        strings::ends_with(&cv, &scalar)?
                    },
                };
                Series::from_column(col)
            },

            Expr::IfElse { condition, true_val, false_val } => {
                let cond_series = condition.eval(df)?;
                let true_series = true_val.eval(df)?;
                let false_series = false_val.eval(df)?;
                let cond_cv = cond_series.view()?;
                let true_cv = true_series.view()?;
                let false_cv = false_series.view()?;
                let col = copying::copy_if_else(&true_cv, &false_cv, &cond_cv)?;
                Series::from_column(col)
            },
        }
    }
}

// ===========================================================================
// Aggregation namespace
// ===========================================================================

/// Aggregation namespace for row-aligned broadcast aggregate expressions.
///
/// Construct via [`Expr::agg`]. Methods on this type reduce their input to a
/// scalar internally, then broadcast the result back to the DataFrame row
/// count. This keeps the output usable inside larger row-aligned expressions
/// such as `col("x") / col("x").agg().sum()`.
pub struct ExprAggNamespace(Expr);

impl ExprAggNamespace {
    /// Sum all rows, then materialize the result as a full-length column.
    ///
    /// Use [`Series::sum`](crate::frame::series::Series::sum) or
    /// [`Series::sum_value`](crate::frame::series::Series::sum_value) when you
    /// want a scalar result instead of a broadcast expression operand.
    pub fn sum(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Sum }
    }

    /// Mean of all rows, materialized as a full-length `Float64` column.
    ///
    /// Use [`Series::mean`](crate::frame::series::Series::mean) or
    /// [`Series::mean_value`](crate::frame::series::Series::mean_value) when
    /// you want a scalar result instead of a broadcast expression operand.
    pub fn mean(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Mean }
    }

    /// Minimum of all rows, materialized as a full-length column.
    ///
    /// Use [`Series::min`](crate::frame::series::Series::min) or
    /// [`Series::min_value`](crate::frame::series::Series::min_value) when you
    /// want a scalar result instead of a broadcast expression operand.
    #[allow(clippy::should_implement_trait)]
    pub fn min(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Min }
    }

    /// Maximum of all rows, materialized as a full-length column.
    ///
    /// Use [`Series::max`](crate::frame::series::Series::max) or
    /// [`Series::max_value`](crate::frame::series::Series::max_value) when you
    /// want a scalar result instead of a broadcast expression operand.
    pub fn max(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Max }
    }

    /// Standard deviation of all rows, materialized as a full-length column.
    ///
    /// Use [`Series::std`](crate::frame::series::Series::std) or
    /// [`Series::std_value`](crate::frame::series::Series::std_value) when you
    /// want a scalar result instead of a broadcast expression operand.
    pub fn std(self, ddof: u32) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Std { ddof } }
    }

    /// Variance of all rows, materialized as a full-length column.
    ///
    /// Use [`Series::var`](crate::frame::series::Series::var) or
    /// [`Series::var_value`](crate::frame::series::Series::var_value) when you
    /// want a scalar result instead of a broadcast expression operand.
    pub fn var(self, ddof: u32) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Variance { ddof } }
    }

    /// Median of all rows, materialized as a full-length `Float64` column.
    ///
    /// Use [`Series::median`](crate::frame::series::Series::median) or
    /// [`Series::median_value`](crate::frame::series::Series::median_value)
    /// when you want a scalar result instead of a broadcast expression
    /// operand.
    pub fn median(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Median }
    }

    /// Logical ANY over all rows, materialized as a full-length BOOL8 column.
    ///
    /// Use [`Series::any`](crate::frame::series::Series::any) or
    /// [`Series::any_value`](crate::frame::series::Series::any_value) when you
    /// want a scalar result instead of a broadcast expression operand.
    pub fn any(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Any }
    }

    /// Logical ALL over all rows, materialized as a full-length BOOL8 column.
    ///
    /// Use [`Series::all`](crate::frame::series::Series::all) or
    /// [`Series::all_value`](crate::frame::series::Series::all_value) when you
    /// want a scalar result instead of a broadcast expression operand.
    pub fn all(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::All }
    }

    /// Product of all rows, materialized as a full-length column.
    ///
    /// Use [`Series::product`](crate::frame::series::Series::product) or
    /// [`Series::product_value`](crate::frame::series::Series::product_value)
    /// when you want a scalar result instead of a broadcast expression
    /// operand.
    pub fn product(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Product }
    }
}

// ===========================================================================
// String namespace
// ===========================================================================

/// String namespace for expression methods on string columns.
///
/// Construct via [`Expr::str`]. Methods on this type produce row-aligned
/// expressions that preserve the input column name.
pub struct ExprStringNamespace(Expr);

impl ExprStringNamespace {
    /// Convert each string element to lowercase.
    pub fn lower(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Lower }
    }

    /// Convert each string element to uppercase.
    pub fn upper(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Upper }
    }

    /// Strip whitespace from both sides.
    pub fn strip(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Strip }
    }

    /// Strip leading whitespace.
    pub fn lstrip(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::LStrip }
    }

    /// Strip trailing whitespace.
    pub fn rstrip(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::RStrip }
    }

    /// Character count per string element.
    pub fn len(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Len }
    }

    /// Byte count per string element.
    pub fn byte_count(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::ByteCount }
    }

    /// Literal substring containment test. Returns BOOL8.
    pub fn contains(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Contains(pat.into()) }
    }

    /// Starts-with test. Returns BOOL8.
    pub fn starts_with(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::StartsWith(pat.into()) }
    }

    /// Ends-with test. Returns BOOL8.
    pub fn ends_with(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::EndsWith(pat.into()) }
    }
}

// ===========================================================================
// Free functions: col(), lit(), when()
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

/// Begin a conditional expression: `when(cond).then(x).otherwise(y)`.
///
/// The result is a row-aligned column — not a scalar branch selector.
/// `IfElse` does not inherit an output name, so `.alias()` is required
/// when used inside [`DataFrame::with_columns`].
///
/// # Example
///
/// ```no_run
/// use cudf::ops::expr::{col, lit, when};
///
/// let expr = when(col("age").ge(18i32))
///     .then("adult")
///     .otherwise("minor")
///     .alias("category");
/// ```
pub fn when(condition: Expr) -> WhenBuilder {
    WhenBuilder(condition)
}

// ===========================================================================
// when / then / otherwise builder types
// ===========================================================================

/// Intermediate builder: holds the condition, awaiting `.then()`.
pub struct WhenBuilder(Expr);

impl WhenBuilder {
    /// Provide the value to produce when the condition is `true`.
    pub fn then(self, value: impl IntoExpr) -> ThenBuilder {
        ThenBuilder { condition: self.0, true_val: value.into_expr() }
    }
}

/// Intermediate builder: holds condition + true-branch, awaiting `.otherwise()`.
pub struct ThenBuilder {
    condition: Expr,
    true_val: Expr,
}

impl ThenBuilder {
    /// Provide the value to produce when the condition is `false` (or null),
    /// completing the conditional expression.
    pub fn otherwise(self, value: impl IntoExpr) -> Expr {
        Expr::IfElse {
            condition: Box::new(self.condition),
            true_val: Box::new(self.true_val),
            false_val: Box::new(value.into_expr()),
        }
    }
}

// ===========================================================================
// IntoExpr trait
// ===========================================================================

/// Trait for values that can be converted into an [`Expr`] operand.
pub trait IntoExpr {
    /// Convert this value into an [`Expr`].
    fn into_expr(self) -> Expr;
}

impl IntoExpr for Expr {
    fn into_expr(self) -> Expr {
        self
    }
}

// SAFETY (coherence): This blanket impl covers every `IntoLit` type. `Expr`
// itself must *never* implement `IntoLit`, otherwise the compiler will see
// overlapping implementations of `IntoExpr` and refuse to compile.
impl<T: IntoLit> IntoExpr for T {
    fn into_expr(self) -> Expr {
        IntoLit::into_expr(self)
    }
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

impl<R: IntoExpr> std::ops::Add<R> for Expr {
    type Output = Expr;

    fn add(self, rhs: R) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Add, rhs: Box::new(rhs.into_expr()) }
    }
}

impl<R: IntoExpr> std::ops::Sub<R> for Expr {
    type Output = Expr;

    fn sub(self, rhs: R) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Sub, rhs: Box::new(rhs.into_expr()) }
    }
}

impl<R: IntoExpr> std::ops::Mul<R> for Expr {
    type Output = Expr;

    fn mul(self, rhs: R) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Mul, rhs: Box::new(rhs.into_expr()) }
    }
}

impl<R: IntoExpr> std::ops::Div<R> for Expr {
    type Output = Expr;

    fn div(self, rhs: R) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Div, rhs: Box::new(rhs.into_expr()) }
    }
}

impl std::ops::Not for Expr {
    type Output = Expr;

    fn not(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Not, input: Box::new(self) }
    }
}

impl<R: IntoExpr> std::ops::Rem<R> for Expr {
    type Output = Expr;

    fn rem(self, rhs: R) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Mod, rhs: Box::new(rhs.into_expr()) }
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Negate, input: Box::new(self) }
    }
}

impl<R: IntoExpr> std::ops::BitAnd<R> for Expr {
    type Output = Expr;

    fn bitand(self, rhs: R) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::LogicalAnd, rhs: Box::new(rhs.into_expr()) }
    }
}

impl<R: IntoExpr> std::ops::BitOr<R> for Expr {
    type Output = Expr;

    fn bitor(self, rhs: R) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::LogicalOr, rhs: Box::new(rhs.into_expr()) }
    }
}

macro_rules! impl_scalar_lhs_binop {
    ($trait:ident, $method:ident, $op:expr; $($ty:ty),+ $(,)?) => {
        $(
            impl std::ops::$trait<Expr> for $ty {
                type Output = Expr;

                fn $method(self, rhs: Expr) -> Expr {
                    Expr::BinaryOp { lhs: Box::new(IntoExpr::into_expr(self)), op: $op, rhs: Box::new(rhs) }
                }
            }
        )+
    };
}

impl_scalar_lhs_binop!(Add, add, BinaryOperator::Add; i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
impl_scalar_lhs_binop!(Sub, sub, BinaryOperator::Sub; i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
impl_scalar_lhs_binop!(Mul, mul, BinaryOperator::Mul; i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
impl_scalar_lhs_binop!(Div, div, BinaryOperator::Div; i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
impl_scalar_lhs_binop!(Rem, rem, BinaryOperator::Mod; i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
impl_scalar_lhs_binop!(BitAnd, bitand, BinaryOperator::LogicalAnd; bool);
impl_scalar_lhs_binop!(BitOr, bitor, BinaryOperator::LogicalOr; bool);
