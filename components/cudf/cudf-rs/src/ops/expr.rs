//! Expression DSL for declarative column operations.
//!
//! Use the DSL entry points [`col`], [`lit`], and [`when`] to start
//! expressions, then extend them with arithmetic, comparisons, casts,
//! naming, aggregation expression methods, and string expression methods
//! before evaluating them against a [`DataFrame`].
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
//! # Literal Evaluation and Memory Behavior
//!
//! Literal constructors such as [`lit`], [`lit_decimal32_raw`],
//! [`lit_decimal64_raw`], [`lit_decimal128_raw`], [`lit_timestamp_raw`], and
//! [`lit_duration_raw`] are lazy expression builders. Creating the [`Expr`]
//! does not allocate GPU memory and does not borrow or transfer host memory.
//!
//! During [`Expr::eval`] or [`Expr::eval_in`], each literal is materialized as
//! a device-resident [`Scalar`] in the resolved output execution domain, then
//! broadcast into a newly allocated device [`Column`](crate::core::Column) with
//! the input [`DataFrame`] row count. The returned [`Series`] owns that new
//! column and retains the resolved provenance: by default the input frame's
//! stream and memory resource, or the same-device borrowed stream and memory
//! resource supplied through [`ExecutionOverrides`].
//!
//! Literal evaluation enqueues libcudf work and does not intentionally
//! synchronize the host before returning. Later host-visible reads of the
//! result, such as value extraction or host export, may synchronize the
//! result's execution domain.
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
//! let df = DataFrame::from_series([("x", x), ("y", y)]).unwrap();
//!
//! let df2 = df
//!     .with_columns([
//!         (col("x") + col("y")).alias("sum"),
//!         (col("x") * 2i32).alias("doubled"),
//!         (col("x") / col("x").agg().sum()).alias("share_of_total"),
//!     ])
//!     .unwrap();
//! ```

use std::{cell::RefCell, sync::Arc};

use super::{
    binary::{self, BinaryOperator},
    concatenate, copying,
    datetime::{self, DatetimeComponent},
    reduction::{self, ReduceAgg, ScanAgg, ScanType},
    replace::{self, FillPolicy},
    search,
    strings::{self, StringSliceOptions, StripSide},
    unary::{self, UnaryOperator},
};
use crate::{
    core::{
        column::Column,
        error::{Error, ErrorCode, Result},
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy, TimestampResolution, TypeId},
    },
    frame::{dataframe::DataFrame, series::Series},
    internal::hash::{FastHashMap, fast_hash_map_with_capacity},
    runtime::{ExecutionOverrides, execution_domain::ExecutionDomain},
};

struct ExprEvalContext<'df> {
    df: &'df DataFrame,
    output_domain: Arc<ExecutionDomain>,
    exec_options: Option<cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
    normalized_columns: RefCell<FastHashMap<String, Series>>,
}

impl<'df> ExprEvalContext<'df> {
    fn new(df: &'df DataFrame, overrides: &ExecutionOverrides<'_>, operation: &'static str) -> Result<Self> {
        let output_domain = overrides.resolve_output_domain(df.execution_domain())?;
        let exec_options = output_domain.explicit_exec_options_raw();
        Ok(Self {
            df,
            output_domain,
            exec_options,
            operation,
            normalized_columns: RefCell::new(fast_hash_map_with_capacity(df.num_columns())),
        })
    }

    fn with_output_domain(df: &'df DataFrame, output_domain: Arc<ExecutionDomain>, operation: &'static str) -> Self {
        let exec_options = output_domain.explicit_exec_options_raw();
        Self {
            df,
            output_domain,
            exec_options,
            operation,
            normalized_columns: RefCell::new(fast_hash_map_with_capacity(df.num_columns())),
        }
    }

    fn exec_options(&self) -> Option<&cudf_sys::cudf_exec_options_t> {
        self.exec_options.as_ref()
    }

    fn output_domain(&self) -> Arc<ExecutionDomain> {
        Arc::clone(&self.output_domain)
    }

    fn column(&self, name: &str) -> Result<Series> {
        if let Some(series) = self.normalized_columns.borrow().get(name) {
            return Ok(series.clone());
        }

        let series = self.df.series(name)?;
        let column = series.arc();
        let normalized = if ExecutionDomain::same_provenance(column.execution_domain(), &self.output_domain) {
            column
        } else {
            ExecutionDomain::insert_stream_dependency(column.execution_domain(), &self.output_domain, self.operation)?;
            Arc::new(column.copy_into_domain(Arc::clone(&self.output_domain), self.operation)?)
        };
        let normalized = Series::from_column_arc(normalized)?;
        self.normalized_columns.borrow_mut().insert(name.to_string(), normalized.clone());
        Ok(normalized)
    }

    fn literal(&self, lit_val: &LitValue) -> Result<Series> {
        let scalar = lit_val.to_scalar_in_domain(&self.output_domain)?;
        let col = Column::from_scalar(&scalar, self.df.num_rows())?;
        Series::from_column(col)
    }

    fn wrap_column(&self, column: Column) -> Result<Series> {
        Series::from_column(column)
    }
}

pub(crate) fn eval_many_in_context<'a, I>(
    df: &DataFrame,
    overrides: &ExecutionOverrides<'_>,
    operation: &'static str,
    exprs: I,
) -> Result<(Arc<ExecutionDomain>, Vec<Series>)>
where
    I: IntoIterator<Item = &'a Expr>,
{
    let ctx = ExprEvalContext::new(df, overrides, operation)?;
    let output_domain = ctx.output_domain();
    let results = exprs.into_iter().map(|expr| expr.eval_with_context(&ctx)).collect::<Result<_>>()?;
    Ok((output_domain, results))
}

pub(crate) fn eval_in_output_domain(
    df: &DataFrame,
    output_domain: Arc<ExecutionDomain>,
    operation: &'static str,
    expr: &Expr,
) -> Result<Series> {
    let ctx = ExprEvalContext::with_output_domain(df, output_domain, operation);
    expr.eval_with_context(&ctx)
}
// ===========================================================================
// LitValue — type-erased scalar factory (crate-private)
// ===========================================================================

/// A type-erased factory for creating a [`Scalar`] from a literal value.
///
/// Not part of the public API — use [`lit()`] to create literals.
type NumericLiteralFactory = dyn Fn(&Arc<ExecutionDomain>) -> Result<Scalar> + Send + Sync + 'static;

enum LitValueInner {
    /// A numeric scalar factory. The closure creates the Scalar on demand.
    Numeric(Box<NumericLiteralFactory>),
    /// A string literal.
    String(String),
    /// A typed null literal.
    Null(DataType),
}

/// Opaque wrapper for literal values inside expressions.
///
/// Construct via [`lit()`]; not user-constructible.
#[derive(Clone)]
pub struct LitValue(Arc<LitValueInner>);

impl LitValue {
    pub(crate) fn to_scalar_in_domain(&self, domain: &Arc<ExecutionDomain>) -> Result<Scalar> {
        match &*self.0 {
            LitValueInner::Numeric(f) => f(domain),
            LitValueInner::String(s) => Scalar::from_str_in_domain(s, domain),
            LitValueInner::Null(dtype) => Scalar::null_in_domain(*dtype, domain),
        }
    }

    fn numeric(f: impl Fn(&Arc<ExecutionDomain>) -> Result<Scalar> + Send + Sync + 'static) -> Self {
        Self(Arc::new(LitValueInner::Numeric(Box::new(f))))
    }

    fn string(s: String) -> Self {
        Self(Arc::new(LitValueInner::String(s)))
    }

    fn null(dtype: DataType) -> Self {
        Self(Arc::new(LitValueInner::Null(dtype)))
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
    Like { pattern: String, escape: Option<String> },
    Slice(StringSliceOptions),
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
    /// Datetime component extraction.
    DateTimeComponent { input: Box<Expr>, component: DatetimeComponent },
    /// Calendrical timestamp offset with month, day, and nanosecond fields.
    DateTimeOffset { input: Box<Expr>, months: i32, days: i32, nanoseconds: i64 },
    /// Conditional: row-wise `if cond then true_val else false_val`.
    ///
    /// Construct via the staged [`when()`] conditional expression chain.
    /// Output name is `None` — an explicit `.alias()` is required for use in
    /// `with_columns`.
    IfElse { condition: Box<Expr>, true_val: Box<Expr>, false_val: Box<Expr> },
    /// Set membership against a literal list.
    InList { input: Box<Expr>, values: Vec<Expr>, negated: bool },
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

    /// Access aggregation expression methods for broadcast aggregate
    /// expressions.
    ///
    /// Methods on the returned accessor reduce once, then broadcast the
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

    /// Element-wise set membership against a literal list. Produces BOOL8.
    ///
    /// Values must be literal expressions. The list is materialized once as a
    /// device column and evaluated with cuDF's membership kernel.
    pub fn is_in_list(self, values: Vec<Expr>) -> Expr {
        Expr::InList { input: Box::new(self), values, negated: false }
    }

    /// Element-wise negated set membership against a literal list. Produces BOOL8.
    pub fn is_not_in_list(self, values: Vec<Expr>) -> Expr {
        Expr::InList { input: Box::new(self), values, negated: true }
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

    /// Access string expression methods for string-specific transforms.
    pub fn str(self) -> ExprStringNamespace {
        ExprStringNamespace(self)
    }

    /// Access datetime expression methods for timestamp-specific transforms.
    pub fn dt(self) -> ExprDatetimeNamespace {
        ExprDatetimeNamespace(self)
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
            | Expr::StringOp { input, .. }
            | Expr::DateTimeComponent { input, .. }
            | Expr::DateTimeOffset { input, .. } => input.output_name(),
            Expr::Reduce { .. } => None,
            // Multi-input or no-source nodes.
            Expr::BinaryOp { .. } | Expr::Literal(_) | Expr::IfElse { .. } => None,
            Expr::InList { input, .. } => input.output_name(),
        }
    }

    // -- Evaluation --------------------------------------------------------

    /// Evaluate this expression against a DataFrame.
    ///
    /// Expression naming is schema metadata handled by DataFrame-level APIs.
    /// The returned [`Series`] itself is unnamed.
    pub fn eval(&self, df: &DataFrame) -> Result<Series> {
        let ctx = ExecutionOverrides::new();
        self.eval_in(df, &ctx)
    }

    /// Evaluate this expression against a DataFrame using borrowed execution overrides.
    ///
    /// Every expression node resolves into the same output execution domain.
    /// Column references are shared when that domain matches the source frame
    /// and device-copied when the borrowed override changes retained
    /// provenance. Literal, reduction-broadcast, string, null-fill, and
    /// conditional intermediates are all allocated through the resolved stream
    /// and memory resource.
    pub fn eval_in(&self, df: &DataFrame, overrides: &ExecutionOverrides<'_>) -> Result<Series> {
        self.eval_in_with_operation(df, overrides, "expr::eval_in")
    }

    pub(crate) fn eval_in_with_operation(
        &self,
        df: &DataFrame,
        overrides: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let ctx = ExprEvalContext::new(df, overrides, operation)?;
        self.eval_with_context(&ctx)
    }

    fn eval_with_context(&self, ctx: &ExprEvalContext<'_>) -> Result<Series> {
        match self {
            Expr::Column(name) => ctx.column(name),

            Expr::Literal(lit_val) => ctx.literal(lit_val),

            Expr::BinaryOp { lhs, op, rhs } => {
                let left = lhs.eval_with_context(ctx)?;
                let right = rhs.eval_with_context(ctx)?;
                let lv = left.view()?;
                let rv = right.view()?;
                let out_type =
                    binary::infer_binary_output_dtype_in(lv.data_type()?, rv.data_type()?, *op, ctx.operation)?;
                let col = binary::binary_operation_vv_impl(
                    &lv,
                    &rv,
                    *op,
                    &out_type,
                    &ctx.output_domain,
                    ctx.exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::UnaryOp { op, input } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = unary::unary_operation_impl(&cv, *op, &ctx.output_domain, ctx.exec_options(), ctx.operation)?;
                ctx.wrap_column(col)
            },

            Expr::Cast { input, target } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = unary::cast_impl(&cv, target, &ctx.output_domain, ctx.exec_options(), ctx.operation)?;
                ctx.wrap_column(col)
            },

            Expr::IsNull(input) => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = unary::is_null_impl(&cv, &ctx.output_domain, ctx.exec_options(), ctx.operation)?;
                ctx.wrap_column(col)
            },

            Expr::IsNotNull(input) => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = unary::is_valid_impl(&cv, &ctx.output_domain, ctx.exec_options(), ctx.operation)?;
                ctx.wrap_column(col)
            },

            Expr::Alias { inner, name } => {
                let _ = name;
                inner.eval_with_context(ctx)
            },

            Expr::FillNaScalar { input, value } => {
                let series = input.eval_with_context(ctx)?;
                let scalar = value.to_scalar_in_domain(&ctx.output_domain)?;
                let cv = series.view()?;
                let col = replace::replace_nulls_with_exec_options(
                    &cv,
                    &scalar,
                    &ctx.output_domain,
                    ctx.exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::FillNaExpr { input, fill } => {
                let series = input.eval_with_context(ctx)?;
                let fill_series = fill.eval_with_context(ctx)?;
                let cv = series.view()?;
                let fill_cv = fill_series.view()?;
                let col = replace::replace_nulls_column_with_exec_options(
                    &cv,
                    &fill_cv,
                    &ctx.output_domain,
                    ctx.exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::FillNullPolicy { input, policy } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = replace::replace_nulls_policy_with_exec_options(
                    &cv,
                    *policy,
                    &ctx.output_domain,
                    ctx.exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::Scan { input, agg } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = reduction::scan_impl(
                    &cv,
                    *agg,
                    ScanType::Inclusive,
                    NullPolicy::Exclude,
                    &ctx.output_domain,
                    ctx.exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::Reduce { input, agg } => {
                let series = input.eval_with_context(ctx)?;
                let dt = series.dtype()?;
                let cv = series.view()?;
                let out_type = crate::frame::series::reduction::infer_reduce_output_type(&dt, agg, "expr::reduce")?;
                let scalar =
                    reduction::reduce_impl(&cv, agg, &out_type, &ctx.output_domain, ctx.exec_options(), ctx.operation)?;
                let col = Column::from_scalar(&scalar, ctx.df.num_rows())?;
                ctx.wrap_column(col)
            },

            Expr::StringOp { input, op } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = match op {
                    StringOperator::Lower => {
                        strings::to_lower_impl(&cv, &ctx.output_domain, ctx.exec_options(), ctx.operation)?
                    },
                    StringOperator::Upper => {
                        strings::to_upper_impl(&cv, &ctx.output_domain, ctx.exec_options(), ctx.operation)?
                    },
                    StringOperator::Strip => {
                        let empty = Scalar::from_str_in_domain("", &ctx.output_domain)?;
                        strings::strip_impl(
                            &cv,
                            StripSide::Both,
                            &empty,
                            &ctx.output_domain,
                            ctx.exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::LStrip => {
                        let empty = Scalar::from_str_in_domain("", &ctx.output_domain)?;
                        strings::strip_impl(
                            &cv,
                            StripSide::Left,
                            &empty,
                            &ctx.output_domain,
                            ctx.exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::RStrip => {
                        let empty = Scalar::from_str_in_domain("", &ctx.output_domain)?;
                        strings::strip_impl(
                            &cv,
                            StripSide::Right,
                            &empty,
                            &ctx.output_domain,
                            ctx.exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::Len => {
                        strings::count_characters_impl(&cv, &ctx.output_domain, ctx.exec_options(), ctx.operation)?
                    },
                    StringOperator::ByteCount => {
                        strings::count_bytes_impl(&cv, &ctx.output_domain, ctx.exec_options(), ctx.operation)?
                    },
                    StringOperator::Contains(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, &ctx.output_domain)?;
                        strings::str_contains_impl(&cv, &scalar, &ctx.output_domain, ctx.exec_options(), ctx.operation)?
                    },
                    StringOperator::StartsWith(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, &ctx.output_domain)?;
                        strings::starts_with_impl(&cv, &scalar, &ctx.output_domain, ctx.exec_options(), ctx.operation)?
                    },
                    StringOperator::EndsWith(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, &ctx.output_domain)?;
                        strings::ends_with_impl(&cv, &scalar, &ctx.output_domain, ctx.exec_options(), ctx.operation)?
                    },
                    StringOperator::Like { pattern, escape } => strings::like_impl(
                        &cv,
                        pattern,
                        escape.as_deref(),
                        &ctx.output_domain,
                        ctx.exec_options(),
                        ctx.operation,
                    )?,
                    StringOperator::Slice(options) => {
                        strings::slice_impl(&cv, *options, &ctx.output_domain, ctx.exec_options(), ctx.operation)?
                    },
                };
                ctx.wrap_column(col)
            },

            Expr::DateTimeComponent { input, component } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = datetime::extract_component_impl(
                    &cv,
                    *component,
                    &ctx.output_domain,
                    ctx.exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::DateTimeOffset { input, months, days, nanoseconds } => {
                let series = input.eval_with_context(ctx)?;
                apply_datetime_offset(series, *months, *days, *nanoseconds, ctx)
            },

            Expr::IfElse { condition, true_val, false_val } => {
                let cond_series = condition.eval_with_context(ctx)?;
                let true_series = true_val.eval_with_context(ctx)?;
                let false_series = false_val.eval_with_context(ctx)?;
                let cond_cv = cond_series.view()?;
                let true_cv = true_series.view()?;
                let false_cv = false_series.view()?;
                let col = copying::copy_if_else_impl(
                    &true_cv,
                    &false_cv,
                    &cond_cv,
                    &ctx.output_domain,
                    ctx.exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::InList { input, values, negated } => {
                let input = input.eval_with_context(ctx)?;
                let input_view = input.view()?;
                let haystack = literal_haystack(values, ctx)?;
                let haystack_view = haystack.view()?;
                if haystack_view.null_count()? == haystack_view.len()? {
                    let scalar = Scalar::null_in_domain(DataType::new(TypeId::Bool8), &ctx.output_domain)?;
                    let col = Column::from_scalar(&scalar, ctx.df.num_rows())?;
                    return ctx.wrap_column(col);
                }
                let col = search::contains_impl(
                    &haystack_view,
                    &input_view,
                    &ctx.output_domain,
                    ctx.exec_options(),
                    ctx.operation,
                )?;
                let col = if *negated {
                    let view = col.view()?;
                    unary::unary_operation_impl(
                        &view,
                        UnaryOperator::Not,
                        &ctx.output_domain,
                        ctx.exec_options(),
                        ctx.operation,
                    )?
                } else {
                    col
                };
                ctx.wrap_column(col)
            },
        }
    }
}

fn apply_datetime_offset(
    series: Series,
    months: i32,
    days: i32,
    nanoseconds: i64,
    ctx: &ExprEvalContext<'_>,
) -> Result<Series> {
    let series = if months != 0 {
        let cv = series.view()?;
        let col =
            datetime::add_calendrical_months_impl(&cv, months, &ctx.output_domain, ctx.exec_options(), ctx.operation)?;
        drop(cv);
        ctx.wrap_column(col)?
    } else {
        series
    };

    let resolution = {
        let cv = series.view()?;
        let input_type = cv.data_type()?.id();
        let Some(resolution) = input_type.timestamp_resolution() else {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                ctx.operation,
                "datetime offset requires timestamp input",
            )
            .with_detail("actual_type_id", input_type.to_string())
            .with_detail("expected_family", "timestamp"));
        };
        resolution
    };
    if days == 0 && nanoseconds == 0 {
        return Ok(series);
    }

    let duration_raw = duration_raw_for_resolution(days, nanoseconds, resolution, ctx.operation)?;
    if duration_raw == 0 {
        return Ok(series);
    }

    let scalar = Scalar::duration_from_raw_in_domain(duration_raw, resolution.duration_type_id(), &ctx.output_domain)?;
    let duration = Column::from_scalar(&scalar, ctx.df.num_rows())?;
    let duration = ctx.wrap_column(duration)?;
    let col = {
        let cv = series.view()?;
        let duration_view = duration.view()?;
        let out_type = binary::infer_binary_output_dtype_in(
            cv.data_type()?,
            duration_view.data_type()?,
            BinaryOperator::Add,
            ctx.operation,
        )?;
        binary::binary_operation_vv_impl(
            &cv,
            &duration_view,
            BinaryOperator::Add,
            &out_type,
            &ctx.output_domain,
            ctx.exec_options(),
            ctx.operation,
        )?
    };
    ctx.wrap_column(col)
}

fn duration_raw_for_resolution(
    days: i32,
    nanoseconds: i64,
    resolution: TimestampResolution,
    operation: &'static str,
) -> Result<i64> {
    const NANOS_PER_DAY: i128 = 86_400_000_000_000;
    let total_nanos = i128::from(days)
        .checked_mul(NANOS_PER_DAY)
        .and_then(|value| value.checked_add(i128::from(nanoseconds)))
        .ok_or_else(|| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "datetime offset duration overflows i128")
        })?;
    let divisor = match resolution {
        TimestampResolution::Days => NANOS_PER_DAY,
        TimestampResolution::Seconds => 1_000_000_000,
        TimestampResolution::Milliseconds => 1_000_000,
        TimestampResolution::Microseconds => 1_000,
        TimestampResolution::Nanoseconds => 1,
    };
    if total_nanos % divisor != 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "datetime offset cannot be represented exactly in the input timestamp resolution",
        )
        .with_detail("resolution", format!("{resolution:?}"))
        .with_detail("days", days.to_string())
        .with_detail("nanoseconds", nanoseconds.to_string()));
    }
    i64::try_from(total_nanos / divisor).map_err(|_error| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "datetime offset duration overflows i64")
            .with_detail("resolution", format!("{resolution:?}"))
            .with_detail("days", days.to_string())
            .with_detail("nanoseconds", nanoseconds.to_string())
    })
}

fn literal_haystack(values: &[Expr], ctx: &ExprEvalContext<'_>) -> Result<Column> {
    if values.is_empty() {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            ctx.operation,
            "expression IN list requires at least one value",
        )
        .with_detail("reason", "empty_in_list"));
    }

    let columns = values
        .iter()
        .map(|expr| {
            let Expr::Literal(value) = expr else {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    ctx.operation,
                    "expression IN list currently requires literal values",
                )
                .with_detail("reason", "non_literal_in_list_value"));
            };
            let scalar = value.to_scalar_in_domain(&ctx.output_domain)?;
            Column::from_scalar(&scalar, 1)
        })
        .collect::<Result<Vec<_>>>()?;
    let views = columns.iter().map(Column::view).collect::<Result<Vec<_>>>()?;
    let view_refs = views.iter().collect::<Vec<_>>();
    concatenate::concatenate_columns(&view_refs)
}

// ===========================================================================
// Aggregation expression methods
// ===========================================================================

/// Aggregation expression methods for row-aligned broadcast aggregate
/// expressions.
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
// String expression methods
// ===========================================================================

/// String expression methods on string columns.
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

    /// SQL `LIKE` pattern match. Returns BOOL8.
    ///
    /// `%` matches zero or more characters and `_` matches one character.
    /// This is not a regex API.
    pub fn like(self, pattern: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Like { pattern: pattern.into(), escape: None } }
    }

    /// SQL `LIKE` pattern match with an explicit escape character.
    pub fn like_escape(self, pattern: impl Into<String>, escape: impl Into<String>) -> Expr {
        Expr::StringOp {
            input: Box::new(self.0),
            op: StringOperator::Like { pattern: pattern.into(), escape: Some(escape.into()) },
        }
    }

    /// Slice strings using zero-based character positions.
    pub fn slice(self, options: StringSliceOptions) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Slice(options) }
    }
}

// ===========================================================================
// Datetime expression methods
// ===========================================================================

/// Datetime expression methods on timestamp columns.
///
/// Construct via [`Expr::dt`]. Methods on this type produce row-aligned
/// expressions and preserve the input column name. Component extraction
/// returns `INT16`; calendrical offsets return the input timestamp type.
pub struct ExprDatetimeNamespace(Expr);

impl ExprDatetimeNamespace {
    /// Extract the calendar year.
    pub fn year(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Year }
    }

    /// Extract the one-based calendar month.
    pub fn month(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Month }
    }

    /// Extract the one-based day of month.
    pub fn day(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Day }
    }

    /// Add a calendrical interval to a timestamp expression.
    ///
    /// Month offsets use libcudf calendrical month semantics. Day and
    /// nanosecond offsets are applied as fixed durations in the input
    /// timestamp's resolution.
    pub fn offset(self, months: i32, days: i32, nanoseconds: i64) -> Expr {
        Expr::DateTimeOffset { input: Box::new(self.0), months, days, nanoseconds }
    }
}

// ===========================================================================
// DSL entry points
// ===========================================================================

/// Create an expression referencing a named column.
pub fn col(name: impl Into<String>) -> Expr {
    Expr::Column(name.into())
}

/// Create a literal expression that broadcasts the value to all rows.
///
/// Supports all [`CudfType`] numeric types, `&str`, and `String`.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit<L: IntoLit>(value: L) -> Expr {
    value.into_expr()
}

/// Create a typed null literal expression that broadcasts to all rows.
///
/// The dtype is part of the literal contract, so query planners can lower
/// typed SQL/DataFusion nulls without inventing sentinel values. Constructing
/// the expression is cheap and does not allocate GPU memory. Evaluation creates
/// an invalid device scalar of the requested dtype, then broadcasts it into a
/// newly allocated device column of the input DataFrame row count in the
/// resolved execution domain.
pub fn lit_null(dtype: impl Into<DataType>) -> Expr {
    Expr::Literal(LitValue::null(dtype.into()))
}

/// Create a `DECIMAL32` literal from its raw fixed-point representation.
///
/// `scale` is the cudf fixed-point scale, so a raw value of `12345` with
/// scale `-2` represents `123.45`. Evaluation broadcasts the scalar to the
/// input DataFrame row count.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_decimal32_raw(raw: i32, scale: i32) -> Expr {
    let dtype = DataType::fixed_point(TypeId::Decimal32, scale);
    Expr::Literal(LitValue::numeric(move |domain| Scalar::decimal_from_raw_in_domain(i128::from(raw), dtype, domain)))
}

/// Create a `DECIMAL64` literal from its raw fixed-point representation.
///
/// `scale` is the cudf fixed-point scale, so a raw value of `12345` with
/// scale `-2` represents `123.45`. Evaluation broadcasts the scalar to the
/// input DataFrame row count.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_decimal64_raw(raw: i64, scale: i32) -> Expr {
    let dtype = DataType::fixed_point(TypeId::Decimal64, scale);
    Expr::Literal(LitValue::numeric(move |domain| Scalar::decimal_from_raw_in_domain(i128::from(raw), dtype, domain)))
}

/// Create a `DECIMAL128` literal from its raw fixed-point representation.
///
/// `scale` is the cudf fixed-point scale, so a raw value of `12345` with
/// scale `-2` represents `123.45`. Evaluation broadcasts the scalar to the
/// input DataFrame row count.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_decimal128_raw(raw: i128, scale: i32) -> Expr {
    let dtype = DataType::fixed_point(TypeId::Decimal128, scale);
    Expr::Literal(LitValue::numeric(move |domain| Scalar::decimal_from_raw_in_domain(raw, dtype, domain)))
}

/// Create a timestamp literal from its raw epoch-unit representation.
///
/// No parsing, timezone conversion, or calendar interpretation is performed
/// by this expression helper.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_timestamp_raw(raw: i64, resolution: TimestampResolution) -> Expr {
    Expr::Literal(LitValue::numeric(move |domain| {
        Scalar::timestamp_from_raw_in_domain(raw, resolution.type_id(), domain)
    }))
}

/// Create a duration literal from its raw unit-count representation.
///
/// Evaluation broadcasts the scalar to the input DataFrame row count without
/// string parsing or timezone behavior.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_duration_raw(raw: i64, resolution: TimestampResolution) -> Expr {
    Expr::Literal(LitValue::numeric(move |domain| {
        Scalar::duration_from_raw_in_domain(raw, resolution.duration_type_id(), domain)
    }))
}

/// Begin a conditional expression chain:
/// `when(cond).then(x).otherwise(y)`.
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
// Staged conditional expression chain
// ===========================================================================

/// First stage of a `when(...).then(...).otherwise(...)` chain.
///
/// Holds the condition until you provide the `true` branch with [`Self::then`].
pub struct WhenBuilder(Expr);

impl WhenBuilder {
    /// Provide the value to produce when the condition is `true`.
    pub fn then(self, value: impl IntoExpr) -> ThenBuilder {
        ThenBuilder { condition: self.0, true_val: value.into_expr() }
    }
}

/// Second stage of a `when(...).then(...).otherwise(...)` chain.
///
/// Holds the condition and `true` branch until you complete the expression
/// with [`Self::otherwise`].
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
