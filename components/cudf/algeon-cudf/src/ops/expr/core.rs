use super::*;
use crate::frame::series::reduction::infer_reduce_output_type;

/// A declarative expression over DataFrame columns.
///
/// Expressions form a tree that is evaluated eagerly against a DataFrame via
/// [`Expr::eval`]. No optimization is performed; each node is evaluated
/// bottom-up.
#[derive(Clone)]
pub enum Expr {
    /// Reference to a named column.
    Column(String),
    /// A broadcast literal scalar value. Construct via [`lit()`].
    Literal(ScalarValue),
    /// Element-wise binary operation.
    BinaryOp { lhs: Box<Expr>, op: BinaryOperator, rhs: Box<Expr> },
    /// Element-wise binary operation with caller-selected output dtype.
    BinaryOpWithType { lhs: Box<Expr>, op: BinaryOperator, rhs: Box<Expr>, output_type: DataType },
    /// Element-wise unary operation.
    UnaryOp { op: UnaryOperator, input: Box<Expr> },
    /// Element-wise SQL-style rounding. Half-way ties round away from zero.
    Round { input: Box<Expr>, decimal_places: i32 },
    /// Cast to a target type.
    ///
    /// With a `precision`, the target is a decimal type declared with that
    /// precision; see [`Expr::cast_decimal`]. Without one, the cast is
    /// libcudf's.
    Cast { input: Box<Expr>, target: DataType, precision: Option<u8> },
    /// SQL `TRY_CAST` to a target type: rows that cannot be represented in the
    /// target type evaluate to null instead of raising an error.
    ///
    /// Runtime support is currently limited to `STRING → integer` targets;
    /// other type pairs return an error at evaluation time.
    TryCast { input: Box<Expr>, target: DataType },
    /// Null-check: produces BOOL8 where `true` = null.
    IsNull(Box<Expr>),
    /// Validity-check: produces BOOL8 where `true` = non-null.
    IsNotNull(Box<Expr>),
    /// Assign a schema/output name to an inner expression.
    Alias { inner: Box<Expr>, name: String },
    /// Replace nulls with a scalar value.
    FillNaScalar { input: Box<Expr>, value: ScalarValue },
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
    /// Row-wise string concatenation. Null inputs contribute empty strings.
    StringConcat(Vec<Expr>),
    /// Datetime component extraction.
    DateTimeComponent { input: Box<Expr>, component: DatetimeComponent },
    /// Datetime floor/truncation.
    DateTimeFloor { input: Box<Expr>, frequency: DatetimeRoundingFrequency },
    /// Calendrical timestamp offset with month, day, and nanosecond fields.
    DateTimeOffset { input: Box<Expr>, months: i32, days: i32, nanoseconds: i64 },
    /// Conditional: row-wise `if cond then true_val else false_val`.
    ///
    /// Construct via the staged [`when()`] conditional expression chain.
    /// Output name is `None`; an explicit `.alias()` is required for use in
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
        Expr::Cast { input: Box::new(self), target: target.into(), precision: None }
    }

    /// Cast the result to a decimal `target` declared with `precision` digits,
    /// with the semantics of a DataFusion `CAST`.
    ///
    /// libcudf decimal types store a scale and no precision, so a plain
    /// [`Expr::cast`] wraps values that overflow the declared precision and
    /// truncates where SQL rounds. This cast rounds half away from zero when
    /// digits are dropped, and fails with [`crate::ErrorCode::ValueOutOfRange`]
    /// when a valid row does not fit `precision` after rescaling, or when a
    /// floating-point row is NaN or infinite. Evaluation fails with
    /// [`crate::ErrorCode::UnsupportedDtype`] unless `target` is a decimal type
    /// and `precision` is between 1 and the digits its storage holds.
    pub fn cast_decimal(self, target: DataType, precision: u8) -> Expr {
        Expr::Cast { input: Box::new(self), target, precision: Some(precision) }
    }

    /// Try-cast the result to the given type, evaluating to null for rows
    /// the target type cannot represent rather than raising an error.
    ///
    /// Runtime support is currently limited to `STRING → integer`; other
    /// pairs return an error at evaluation time.
    pub fn try_cast(self, target: impl Into<DataType>) -> Expr {
        Expr::TryCast { input: Box::new(self), target: target.into() }
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

    /// Exponentiation (`self ^ rhs`). Result type follows cuDF `Pow` rules.
    pub fn pow(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::Pow, rhs: Box::new(rhs.into_expr()) }
    }

    /// Integer floor division.
    pub fn floor_div(self, rhs: impl IntoExpr) -> Expr {
        Expr::BinaryOp { lhs: Box::new(self), op: BinaryOperator::FloorDiv, rhs: Box::new(rhs.into_expr()) }
    }

    /// Element-wise binary operation with an explicit output type.
    ///
    /// Use this when the caller needs libcudf's
    /// `binary_operation(..., output_type)` contract instead of inferred
    /// output typing. It is useful for
    /// fixed-point decimal operations where the result scale is a semantic
    /// requirement of the query plan.
    pub fn binary_op_with_type(self, rhs: impl IntoExpr, op: BinaryOperator, output_type: impl Into<DataType>) -> Expr {
        Expr::BinaryOpWithType {
            lhs: Box::new(self),
            op,
            rhs: Box::new(rhs.into_expr()),
            output_type: output_type.into(),
        }
    }

    /// Element-wise absolute value.
    pub fn abs(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Abs, input: Box::new(self) }
    }

    /// Element-wise natural logarithm.
    pub fn log(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Log, input: Box::new(self) }
    }

    /// Element-wise natural exponential (`e ^ self`).
    pub fn exp(self) -> Expr {
        Expr::UnaryOp { op: UnaryOperator::Exp, input: Box::new(self) }
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

    /// Element-wise SQL-style rounding.
    ///
    /// Half-way ties round away from zero, matching DataFusion and common SQL
    /// `ROUND` behavior rather than cuDF's `Rint` half-even unary operator.
    pub fn round(self, decimal_places: i32) -> Expr {
        Expr::Round { input: Box::new(self), decimal_places }
    }

    // -- Null fill / cumulative builders ----------------------------------

    /// Replace null values with a scalar. Name: `fillna` (matches `SeriesOps::fillna`).
    pub fn fillna(self, value: impl Into<ScalarValue>) -> Expr {
        Expr::FillNaScalar { input: Box::new(self), value: value.into() }
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

    // -- String builders ---------------------------------------------------

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
            | Expr::TryCast { input, .. }
            | Expr::IsNull(input)
            | Expr::IsNotNull(input)
            | Expr::UnaryOp { input, .. }
            | Expr::Round { input, .. }
            | Expr::FillNaScalar { input, .. }
            | Expr::FillNaExpr { input, .. }
            | Expr::FillNullPolicy { input, .. }
            | Expr::Scan { input, .. }
            | Expr::StringOp { input, .. }
            | Expr::DateTimeComponent { input, .. }
            | Expr::DateTimeFloor { input, .. }
            | Expr::DateTimeOffset { input, .. } => input.output_name(),
            Expr::Reduce { .. } | Expr::StringConcat(_) => None,
            // Multi-input or no-source nodes.
            Expr::BinaryOp { .. } | Expr::BinaryOpWithType { .. } | Expr::Literal(_) | Expr::IfElse { .. } => None,
            Expr::InList { input, .. } => input.output_name(),
        }
    }

    // -- Evaluation --------------------------------------------------------

    /// Evaluate this expression against a DataFrame.
    ///
    /// Expression naming is schema metadata handled by DataFrame-level APIs.
    /// The returned [`Series`] itself is unnamed.
    pub fn eval(&self, df: &DataFrame) -> Result<Series> {
        let ctx = Exec::RETAINED;
        self.eval_in(df, ctx)
    }

    /// Evaluate this expression against a DataFrame using borrowed execution overrides.
    ///
    /// Every expression node resolves into the same output execution domain.
    /// Column references are shared when that domain matches the source frame
    /// and device-copied when the borrowed override changes retained
    /// provenance. Literal, reduction-broadcast, string, null-fill, and
    /// conditional intermediates are all allocated through the resolved stream
    /// and memory resource.
    pub fn eval_in(&self, df: &DataFrame, overrides: Exec<'_>) -> Result<Series> {
        self.eval_in_with_operation(df, overrides, "expr::eval_in")
    }

    pub(crate) fn eval_in_with_operation(
        &self,
        df: &DataFrame,
        overrides: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let ctx = ExprEvalContext::new(df, overrides, operation)?;
        self.eval_with_context(&ctx)
    }

    pub(super) fn eval_with_context(&self, ctx: &ExprEvalContext<'_>) -> Result<Series> {
        match self {
            Expr::Column(name) => ctx.column(name),

            Expr::Literal(lit_val) => ctx.literal(lit_val),

            Expr::BinaryOp { lhs, op, rhs } => eval_binary_op_with_context(lhs, *op, rhs, None, ctx),

            Expr::BinaryOpWithType { lhs, op, rhs, output_type } => {
                eval_binary_op_with_context(lhs, *op, rhs, Some(output_type), ctx)
            },

            Expr::UnaryOp { op, input } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = unary::unary_operation_impl(
                    &cv,
                    *op,
                    &ctx.output_domain,
                    ctx.required_exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::Round { input, decimal_places } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = unary::round_half_up_impl(
                    &cv,
                    *decimal_places,
                    &ctx.output_domain,
                    ctx.required_exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::Cast { input, target, precision } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let (domain, options, operation) = (&ctx.output_domain, ctx.required_exec_options(), ctx.operation);
                let col = match precision {
                    Some(precision) => {
                        unary::cast_to_decimal_impl(&cv, target, *precision, domain, options, operation)?
                    },
                    None => unary::cast_impl(&cv, target, domain, options, operation)?,
                };
                ctx.wrap_column(col)
            },

            Expr::TryCast { input, target } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let source_type = cv.data_type()?;
                if source_type.id() != TypeId::String || !target.id().is_integral() {
                    return Err(Error::new_with(
                        ErrorCode::UnsupportedDtype,
                        "expr::try_cast",
                        "TRY_CAST currently only supports STRING -> integer targets",
                    )
                    .with_detail("source_type_id", format!("{:?}", source_type.id()).to_lowercase())
                    .with_detail("target_type_id", format!("{:?}", target.id()).to_lowercase()));
                }
                let col = string_ops::try_to_integers_impl(
                    &cv,
                    target,
                    &ctx.output_domain,
                    ctx.required_exec_options(),
                    "expr::try_cast",
                )?;
                ctx.wrap_column(col)
            },

            Expr::IsNull(input) => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = unary::is_null_impl(&cv, &ctx.output_domain, ctx.required_exec_options(), ctx.operation)?;
                ctx.wrap_column(col)
            },

            Expr::IsNotNull(input) => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = unary::is_valid_impl(&cv, &ctx.output_domain, ctx.required_exec_options(), ctx.operation)?;
                ctx.wrap_column(col)
            },

            Expr::Alias { inner, .. } => inner.eval_with_context(ctx),

            Expr::FillNaScalar { input, value } => {
                let series = input.eval_with_context(ctx)?;
                let scalar = Scalar::from_scalar_value_in_domain(value, &ctx.output_domain)?;
                let cv = series.view()?;
                let col = replace::replace_nulls_with_exec_options(
                    &cv,
                    &scalar,
                    &ctx.output_domain,
                    ctx.required_exec_options(),
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
                    ctx.required_exec_options(),
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
                    ctx.required_exec_options(),
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
                    ctx.required_exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::Reduce { input, agg } => {
                let series = input.eval_with_context(ctx)?;
                let dt = series.dtype()?;
                let cv = series.view()?;
                let out_type = infer_reduce_output_type(&dt, agg, "expr::reduce")?;
                let scalar = reduction::reduce_impl(
                    &cv,
                    agg,
                    &out_type,
                    &ctx.output_domain,
                    ctx.required_exec_options(),
                    ctx.operation,
                )?;
                let col = Column::from_scalar(&scalar, ctx.df.num_rows())?;
                ctx.wrap_column(col)
            },

            Expr::StringOp { input, op } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = match op {
                    StringOperator::Lower => {
                        string_ops::to_lower_impl(&cv, &ctx.output_domain, ctx.required_exec_options(), ctx.operation)?
                    },
                    StringOperator::Upper => {
                        string_ops::to_upper_impl(&cv, &ctx.output_domain, ctx.required_exec_options(), ctx.operation)?
                    },
                    StringOperator::Strip(to_strip) => {
                        let to_strip = Scalar::from_str_in_domain(to_strip, &ctx.output_domain)?;
                        string_ops::strip_impl(
                            &cv,
                            StripSide::Both,
                            &to_strip,
                            &ctx.output_domain,
                            ctx.required_exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::LStrip(to_strip) => {
                        let to_strip = Scalar::from_str_in_domain(to_strip, &ctx.output_domain)?;
                        string_ops::strip_impl(
                            &cv,
                            StripSide::Left,
                            &to_strip,
                            &ctx.output_domain,
                            ctx.required_exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::RStrip(to_strip) => {
                        let to_strip = Scalar::from_str_in_domain(to_strip, &ctx.output_domain)?;
                        string_ops::strip_impl(
                            &cv,
                            StripSide::Right,
                            &to_strip,
                            &ctx.output_domain,
                            ctx.required_exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::Len => string_ops::count_characters_impl(
                        &cv,
                        &ctx.output_domain,
                        ctx.required_exec_options(),
                        ctx.operation,
                    )?,
                    StringOperator::ByteCount => string_ops::count_bytes_impl(
                        &cv,
                        &ctx.output_domain,
                        ctx.required_exec_options(),
                        ctx.operation,
                    )?,
                    StringOperator::Contains(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, &ctx.output_domain)?;
                        string_ops::str_contains_impl(
                            &cv,
                            &scalar,
                            &ctx.output_domain,
                            ctx.required_exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::StartsWith(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, &ctx.output_domain)?;
                        string_ops::starts_with_impl(
                            &cv,
                            &scalar,
                            &ctx.output_domain,
                            ctx.required_exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::EndsWith(pat) => {
                        let scalar = Scalar::from_str_in_domain(pat, &ctx.output_domain)?;
                        string_ops::ends_with_impl(
                            &cv,
                            &scalar,
                            &ctx.output_domain,
                            ctx.required_exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::Like { pattern, escape } => string_ops::like_impl(
                        &cv,
                        pattern,
                        escape.as_deref(),
                        &ctx.output_domain,
                        ctx.required_exec_options(),
                        ctx.operation,
                    )?,
                    StringOperator::ReplaceWithBackrefs { pattern, replacement } => {
                        string_ops::replace_with_backrefs_impl(
                            &cv,
                            pattern,
                            replacement,
                            &ctx.output_domain,
                            ctx.required_exec_options(),
                            ctx.operation,
                        )?
                    },
                    StringOperator::Slice(options) => string_ops::slice_impl(
                        &cv,
                        *options,
                        &ctx.output_domain,
                        ctx.required_exec_options(),
                        ctx.operation,
                    )?,
                };
                ctx.wrap_column(col)
            },

            Expr::StringConcat(inputs) => eval_string_concat(inputs, ctx),

            Expr::DateTimeComponent { input, component } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = datetime_ops::extract_component_impl(
                    &cv,
                    *component,
                    &ctx.output_domain,
                    ctx.required_exec_options(),
                    ctx.operation,
                )?;
                ctx.wrap_column(col)
            },

            Expr::DateTimeFloor { input, frequency } => {
                let series = input.eval_with_context(ctx)?;
                let cv = series.view()?;
                let col = datetime_ops::floor_datetimes_impl(
                    &cv,
                    *frequency,
                    &ctx.output_domain,
                    ctx.required_exec_options(),
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
                    ctx.required_exec_options(),
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
                    ctx.required_exec_options(),
                    ctx.operation,
                )?;
                let col = if *negated {
                    let view = col.view()?;
                    unary::unary_operation_impl(
                        &view,
                        UnaryOperator::Not,
                        &ctx.output_domain,
                        ctx.required_exec_options(),
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

fn eval_binary_op_with_context(
    lhs: &Expr,
    op: BinaryOperator,
    rhs: &Expr,
    explicit_output_type: Option<&DataType>,
    ctx: &ExprEvalContext<'_>,
) -> Result<Series> {
    let left = lhs.eval_with_context(ctx)?;
    let right = rhs.eval_with_context(ctx)?;
    let (left, right) = align_decimal_binary_storage(left, right, ctx)?;
    let lv = left.view()?;
    let rv = right.view()?;
    let inferred_out_type;
    let out_type = match explicit_output_type {
        Some(output_type) => output_type,
        None => {
            inferred_out_type =
                binary::infer_binary_output_dtype_in(lv.data_type()?, rv.data_type()?, op, ctx.operation)?;
            &inferred_out_type
        },
    };
    let col = binary::binary_operation_vv_impl(
        &lv,
        &rv,
        op,
        out_type,
        &ctx.output_domain,
        ctx.required_exec_options(),
        ctx.operation,
    )?;
    ctx.wrap_column(col)
}

fn eval_string_concat(inputs: &[Expr], ctx: &ExprEvalContext<'_>) -> Result<Series> {
    if inputs.is_empty() {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            ctx.operation,
            "string concat expression requires at least one input",
        ));
    }
    if inputs.len() == 1 {
        return inputs[0].eval_with_context(ctx);
    }
    let series = inputs.iter().map(|input| input.eval_with_context(ctx)).collect::<Result<Vec<_>>>()?;
    let views = series.iter().map(Series::view).collect::<Result<Vec<_>>>()?;
    let view_refs = views.iter().collect::<Vec<_>>();
    let col = string_ops::concatenate_impl(&view_refs, ctx.operation)?;
    ctx.wrap_column(col)
}

fn align_decimal_binary_storage(left: Series, right: Series, ctx: &ExprEvalContext<'_>) -> Result<(Series, Series)> {
    let left_type = left.dtype()?;
    let right_type = right.dtype()?;
    if !(left_type.id().is_decimal() && right_type.id().is_decimal()) || left_type.id() == right_type.id() {
        return Ok((left, right));
    }

    let common_storage = common_decimal_storage_type(left_type.id(), right_type.id(), ctx.operation)?;
    let left_target = DataType::fixed_point(common_storage, left_type.scale())?;
    let right_target = DataType::fixed_point(common_storage, right_type.scale())?;
    Ok((
        cast_series_to_binary_operand_dtype(left, left_target, ctx)?,
        cast_series_to_binary_operand_dtype(right, right_target, ctx)?,
    ))
}

fn common_decimal_storage_type(left: TypeId, right: TypeId, operation: &'static str) -> Result<TypeId> {
    match (decimal_storage_rank(left), decimal_storage_rank(right)) {
        (Some(left_rank), Some(right_rank)) => Ok(match left_rank.max(right_rank) {
            0 => TypeId::Decimal32,
            1 => TypeId::Decimal64,
            _ => TypeId::Decimal128,
        }),
        _ => Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "decimal binary storage alignment requires decimal operands",
        )
        .with_detail("left_type_id", left.to_string())
        .with_detail("right_type_id", right.to_string())),
    }
}

fn decimal_storage_rank(type_id: TypeId) -> Option<u8> {
    match type_id {
        TypeId::Decimal32 => Some(0),
        TypeId::Decimal64 => Some(1),
        TypeId::Decimal128 => Some(2),
        _ => None,
    }
}

fn cast_series_to_binary_operand_dtype(series: Series, target: DataType, ctx: &ExprEvalContext<'_>) -> Result<Series> {
    if series.dtype()? == target {
        return Ok(series);
    }
    let cv = series.view()?;
    let column = unary::cast_impl(&cv, &target, &ctx.output_domain, ctx.required_exec_options(), ctx.operation)?;
    ctx.wrap_column(column)
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
        let col = datetime_ops::add_calendrical_months_impl(
            &cv,
            months,
            &ctx.output_domain,
            ctx.required_exec_options(),
            ctx.operation,
        )?;
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
            ctx.required_exec_options(),
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
            let scalar = Scalar::from_scalar_value_in_domain(value, &ctx.output_domain)?;
            Column::from_scalar(&scalar, 1)
        })
        .collect::<Result<Vec<_>>>()?;
    let views = columns.iter().map(Column::view).collect::<Result<Vec<_>>>()?;
    let view_refs = views.iter().collect::<Vec<_>>();
    concatenate::concatenate_columns(&view_refs)
}

/// Create an expression referencing a named column.
pub fn col(name: impl Into<String>) -> Expr {
    Expr::Column(name.into())
}

/// Row-wise concatenate string expressions with an empty separator.
///
/// Null input values contribute empty strings. At least one input expression is
/// required; a single input is returned unchanged during evaluation.
pub fn concat(inputs: Vec<Expr>) -> Expr {
    Expr::StringConcat(inputs)
}

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

// Coherence: `ScalarValue` must never gain `From<Expr>`, or this blanket impl
// overlaps `IntoExpr for Expr`.
impl<T: Into<ScalarValue>> IntoExpr for T {
    fn into_expr(self) -> Expr {
        lit(self)
    }
}

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
