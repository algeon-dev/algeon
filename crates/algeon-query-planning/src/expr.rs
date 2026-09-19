//! Expression IR for native plan nodes.
//!
//! Expressions use Arrow datatypes and native column references but do not
//! depend on DataFusion physical-expression objects. Unsupported expression
//! forms should be rejected during capability analysis or validation before
//! reaching cudf execution.

use std::{fmt, ops::Not};

use arrow::datatypes::DataType;

mod column;
mod literal;

pub use self::{
    column::{ColumnQualifier, ColumnRef},
    literal::LiteralValue,
};

/// Resolves a projection expression that only renames one existing column.
///
/// Nested aliases do not evaluate or allocate a new device column. Callers
/// that rely on this for a zero-copy path must still verify that the declared
/// output type matches the referenced input field.
#[doc(hidden)]
pub fn alias_wrapped_column(expr: &Expr) -> Option<&ColumnRef> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Alias { expr, .. } => alias_wrapped_column(expr),
        _ => None,
    }
}

/// Returns whether a string representation cast preserves the source value and row identity.
pub fn representation_cast_preserves_identity(source_type: &DataType, target_type: &DataType) -> bool {
    source_type == target_type
        || matches!(
            (source_type, target_type),
            (DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View, DataType::Utf8View)
                | (DataType::Utf8View, DataType::Utf8 | DataType::LargeUtf8)
                | (DataType::Utf8, DataType::LargeUtf8)
                | (DataType::LargeUtf8, DataType::Utf8)
        )
}

#[doc(hidden)]
pub fn representation_cast_column(expr: &Expr) -> Option<(&ColumnRef, &DataType)> {
    match expr {
        Expr::Alias { expr, .. } => representation_cast_column(expr),
        Expr::Cast { expr, target_type } => alias_wrapped_column(expr).map(|column| (column, target_type)),
        _ => None,
    }
}

#[doc(hidden)]
pub fn interval_month_day_nano(expr: &Expr) -> Option<(i32, i32, i64)> {
    let Expr::Literal(LiteralValue::IntervalMonthDayNano { months, days, nanoseconds }) = expr else {
        return None;
    };
    Some((*months, *days, *nanoseconds))
}

/// Scalar expression IR used by native query-plan nodes.
///
/// `Expr` describes logical expression structure independently of DataFusion's
/// physical-expression APIs. It is intentionally compact: it names input
/// columns, literals, SQL-style predicates, scalar operations, and aliases that
/// a native executor or validator can either lower directly or reject with a
/// structured diagnostic. The `Display` implementation is part of the
/// human-readable diagnostics contract; keep it stable unless the displayed IR
/// is misleading.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A column reference resolved against the current node input schema.
    Column(ColumnRef),
    /// A typed literal value.
    Literal(LiteralValue),
    Not(Box<Expr>),
    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),
    /// Explicit cast to an Arrow data type.
    Cast {
        /// Expression whose result is cast.
        expr: Box<Expr>,
        /// Desired output type after the cast.
        target_type: DataType,
    },
    /// SQL `TRY_CAST` to an Arrow data type. Rows whose value cannot be
    /// represented in `target_type` evaluate to null instead of raising.
    ///
    /// Native execution currently admits only `Utf8` → integer targets;
    /// other type pairs are rejected during validation/lowering.
    TryCast {
        /// Expression whose result is try-cast.
        expr: Box<Expr>,
        /// Desired output type after the try-cast.
        target_type: DataType,
    },
    /// Binary infix operation such as arithmetic, comparison, or boolean logic.
    Binary {
        /// Left operand.
        left: Box<Expr>,
        /// Operation to apply.
        op: BinaryOp,
        /// Right operand.
        right: Box<Expr>,
        /// Optional declared output type for operators whose result type is a
        /// frontend contract rather than something the native backend can infer
        /// from operands alone.
        output_type: Option<DataType>,
    },
    /// SQL `LIKE` or `NOT LIKE` string predicate.
    Like {
        /// String expression being matched.
        expr: Box<Expr>,
        /// SQL LIKE pattern. `%` and `_` keep their SQL wildcard meaning.
        pattern: String,
        /// Optional SQL escape marker. `None` means no explicit escape clause.
        escape: Option<String>,
        /// `true` for `NOT LIKE`; `false` for `LIKE`.
        negated: bool,
    },
    /// SQL substring expression.
    ///
    /// Current native support uses a one-based `start` with `start >= 1`.
    /// When `length` is present, it must be nonnegative and
    /// `(start - 1) + length` must fit in `i32`.
    Substring {
        /// String expression being sliced.
        expr: Box<Expr>,
        /// One-based start position. Current native execution requires `start >= 1`.
        start: i32,
        /// Optional character count. When present, it must be nonnegative and
        /// `(start - 1) + length` must fit in `i32`.
        length: Option<i32>,
    },
    Lower(Box<Expr>),
    Upper(Box<Expr>),
    Trim {
        expr: Box<Expr>,
        trim_chars: String,
    },
    LTrim {
        expr: Box<Expr>,
        trim_chars: String,
    },
    RTrim {
        expr: Box<Expr>,
        trim_chars: String,
    },
    /// Literal substring containment test.
    StringContains {
        expr: Box<Expr>,
        pattern: String,
    },
    /// Literal prefix test.
    StartsWith {
        expr: Box<Expr>,
        prefix: String,
    },
    /// Literal suffix test.
    EndsWith {
        expr: Box<Expr>,
        suffix: String,
    },
    /// SQL `CHARACTER_LENGTH(expr)` string expression.
    StringLength {
        /// String expression whose characters are counted.
        expr: Box<Expr>,
        /// Declared frontend output type.
        output_type: DataType,
    },
    /// SQL `REGEXP_REPLACE(expr, pattern, replacement)` string expression.
    RegexpReplace {
        /// String expression being transformed.
        expr: Box<Expr>,
        /// Regex pattern.
        pattern: String,
        /// Replacement template, including optional back-references.
        replacement: String,
        /// Declared frontend output type.
        output_type: DataType,
    },
    /// SQL `CONCAT(expr, ...)` string expression.
    ///
    /// Native execution follows DataFusion semantics: null inputs contribute
    /// empty strings rather than producing a null output row.
    Concat(Vec<Expr>),
    Abs(Box<Expr>),
    /// SQL one-argument floating-point math function such as `SQRT(expr)`,
    /// `LN(expr)`, or `EXP(expr)`.
    ///
    /// The `output_type` is the frontend-declared result type. It is part of
    /// the native contract because cudf's expression builder can infer a
    /// physical type that still needs to be aligned with DataFusion's exposed
    /// schema. Results follow IEEE 754: domain errors produce NaN or an
    /// infinity instead of raising, and null inputs produce null.
    UnaryMath {
        /// Function applied to each row.
        op: UnaryMathOp,
        /// Numeric expression the function is applied to.
        expr: Box<Expr>,
        /// Declared output type after evaluation.
        output_type: DataType,
    },
    /// SQL `ROUND(expr, decimal_places)` numeric expression.
    ///
    /// The `output_type` is the frontend-declared DataFusion result type. It
    /// is part of the native contract because decimal `ROUND` may reduce scale
    /// even when the lower-level cuDF operation preserves the input storage
    /// type.
    Round {
        /// Numeric or decimal expression being rounded.
        expr: Box<Expr>,
        /// Number of decimal places. Half-way ties round away from zero.
        decimal_places: i32,
        /// Declared output type after rounding.
        output_type: DataType,
    },
    /// SQL `IN` or `NOT IN` membership predicate.
    ///
    /// The IR can represent general SQL `IN` lists. Runnable native plans
    /// currently require a non-empty list of literal, non-null candidate
    /// values.
    InList {
        /// Expression whose value is compared with the list.
        expr: Box<Expr>,
        /// Candidate expressions in SQL list order. Runnable native plans
        /// currently require this list to be non-empty and contain only
        /// literal, non-null values.
        list: Vec<Expr>,
        /// `true` for `NOT IN`; `false` for `IN`.
        negated: bool,
    },
    /// SQL `IS DISTINCT FROM` null-aware comparison.
    ///
    /// Unlike ordinary inequality this always produces a non-null boolean:
    /// two nulls are not distinct, and exactly one null is distinct.
    IsDistinctFrom {
        /// Left comparison operand.
        left: Box<Expr>,
        /// Right comparison operand.
        right: Box<Expr>,
    },
    /// SQL `IS NOT DISTINCT FROM` null-aware comparison.
    ///
    /// Unlike ordinary equality this always produces a non-null boolean:
    /// two nulls are not distinct and therefore compare true.
    IsNotDistinctFrom {
        /// Left comparison operand.
        left: Box<Expr>,
        /// Right comparison operand.
        right: Box<Expr>,
    },
    /// Searched SQL `CASE WHEN ... THEN ... ELSE ... END`.
    Case {
        /// Ordered branches. The first branch whose predicate evaluates true is selected.
        branches: Vec<CaseWhen>,
        /// Result expression used when no branch predicate evaluates true.
        else_expr: Box<Expr>,
    },
    /// SQL date/time field extraction.
    Extract {
        /// Field to extract from the input expression.
        field: DatePart,
        /// Date, timestamp, or compatible temporal expression.
        expr: Box<Expr>,
    },
    /// SQL `DATE_TRUNC(field, expr)` expression.
    DateTrunc {
        /// Field to truncate the input expression to.
        field: DatePart,
        /// Timestamp expression being truncated.
        expr: Box<Expr>,
        /// Declared frontend output type.
        output_type: DataType,
    },
    /// Output alias for projection and aggregate argument naming.
    Alias {
        /// Expression being renamed.
        expr: Box<Expr>,
        /// Field name exposed to downstream schemas and diagnostics.
        alias: String,
    },
}

impl Expr {
    pub fn column(name: impl Into<String>) -> Self {
        Self::Column(ColumnRef::unqualified(name))
    }

    /// Build a column expression qualified by a positional input.
    ///
    /// Use this for multi-input nodes such as joins, where equal leaf column
    /// names may appear on both sides. `name` is the source/input field name;
    /// `display_name` is the projected or user-visible name.
    pub fn input_column(input: usize, name: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self::Column(ColumnRef::input(input, name, display_name))
    }

    pub fn node_column(node_id: impl Into<String>, name: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self::Column(ColumnRef::node(node_id, name, display_name))
    }

    pub fn literal(value: LiteralValue) -> Self {
        Self::Literal(value)
    }

    pub fn binary(left: Self, op: BinaryOp, right: Self) -> Self {
        Self::Binary { left: Box::new(left), op, right: Box::new(right), output_type: None }
    }

    /// Build a binary operation expression with a declared result type.
    pub fn binary_with_output_type(left: Self, op: BinaryOp, right: Self, output_type: DataType) -> Self {
        Self::Binary { left: Box::new(left), op, right: Box::new(right), output_type: Some(output_type) }
    }

    pub fn logical_not(expr: Self) -> Self {
        Self::Not(Box::new(expr))
    }

    pub fn is_null(expr: Self) -> Self {
        Self::IsNull(Box::new(expr))
    }

    pub fn is_not_null(expr: Self) -> Self {
        Self::IsNotNull(Box::new(expr))
    }

    pub fn cast(expr: Self, target_type: DataType) -> Self {
        Self::Cast { expr: Box::new(expr), target_type }
    }

    pub fn try_cast(expr: Self, target_type: DataType) -> Self {
        Self::TryCast { expr: Box::new(expr), target_type }
    }

    pub fn like(expr: Self, pattern: impl Into<String>) -> Self {
        Self::Like { expr: Box::new(expr), pattern: pattern.into(), escape: None, negated: false }
    }

    pub fn not_like(expr: Self, pattern: impl Into<String>) -> Self {
        Self::Like { expr: Box::new(expr), pattern: pattern.into(), escape: None, negated: true }
    }

    pub fn like_escape(expr: Self, pattern: impl Into<String>, escape: impl Into<String>) -> Self {
        Self::Like { expr: Box::new(expr), pattern: pattern.into(), escape: Some(escape.into()), negated: false }
    }

    pub fn not_like_escape(expr: Self, pattern: impl Into<String>, escape: impl Into<String>) -> Self {
        Self::Like { expr: Box::new(expr), pattern: pattern.into(), escape: Some(escape.into()), negated: true }
    }

    /// Build a SQL substring expression.
    ///
    /// Current native support requires `start >= 1`, nonnegative `length` when
    /// present, and `(start - 1) + length` to fit in `i32`.
    pub fn substring(expr: Self, start: i32, length: Option<i32>) -> Self {
        Self::Substring { expr: Box::new(expr), start, length }
    }

    pub fn lower(expr: Self) -> Self {
        Self::Lower(Box::new(expr))
    }

    pub fn upper(expr: Self) -> Self {
        Self::Upper(Box::new(expr))
    }

    pub fn trim(expr: Self, trim_chars: impl Into<String>) -> Self {
        Self::Trim { expr: Box::new(expr), trim_chars: trim_chars.into() }
    }

    pub fn ltrim(expr: Self, trim_chars: impl Into<String>) -> Self {
        Self::LTrim { expr: Box::new(expr), trim_chars: trim_chars.into() }
    }

    pub fn rtrim(expr: Self, trim_chars: impl Into<String>) -> Self {
        Self::RTrim { expr: Box::new(expr), trim_chars: trim_chars.into() }
    }

    pub fn string_contains(expr: Self, pattern: impl Into<String>) -> Self {
        Self::StringContains { expr: Box::new(expr), pattern: pattern.into() }
    }

    pub fn starts_with(expr: Self, prefix: impl Into<String>) -> Self {
        Self::StartsWith { expr: Box::new(expr), prefix: prefix.into() }
    }

    pub fn ends_with(expr: Self, suffix: impl Into<String>) -> Self {
        Self::EndsWith { expr: Box::new(expr), suffix: suffix.into() }
    }

    pub fn string_length(expr: Self, output_type: DataType) -> Self {
        Self::StringLength { expr: Box::new(expr), output_type }
    }

    pub fn regexp_replace(
        expr: Self,
        pattern: impl Into<String>,
        replacement: impl Into<String>,
        output_type: DataType,
    ) -> Self {
        Self::RegexpReplace {
            expr: Box::new(expr),
            pattern: pattern.into(),
            replacement: replacement.into(),
            output_type,
        }
    }

    pub fn concat(exprs: Vec<Self>) -> Self {
        Self::Concat(exprs)
    }

    pub fn abs(expr: Self) -> Self {
        Self::Abs(Box::new(expr))
    }

    pub fn unary_math(op: UnaryMathOp, expr: Self, output_type: DataType) -> Self {
        Self::UnaryMath { op, expr: Box::new(expr), output_type }
    }

    pub fn round(expr: Self, decimal_places: i32, output_type: DataType) -> Self {
        Self::Round { expr: Box::new(expr), decimal_places, output_type }
    }

    /// Build a SQL `IN` predicate.
    ///
    /// The IR accepts expression candidates, but runnable native plans require
    /// a non-empty list of literal, non-null candidate values.
    pub fn in_list(expr: Self, list: Vec<Self>) -> Self {
        Self::InList { expr: Box::new(expr), list, negated: false }
    }

    /// Build a SQL `NOT IN` predicate.
    ///
    /// The IR accepts expression candidates, but runnable native plans require
    /// a non-empty list of literal, non-null candidate values.
    pub fn not_in_list(expr: Self, list: Vec<Self>) -> Self {
        Self::InList { expr: Box::new(expr), list, negated: true }
    }

    pub fn is_distinct_from(left: Self, right: Self) -> Self {
        Self::IsDistinctFrom { left: Box::new(left), right: Box::new(right) }
    }

    pub fn is_not_distinct_from(left: Self, right: Self) -> Self {
        Self::IsNotDistinctFrom { left: Box::new(left), right: Box::new(right) }
    }

    pub fn case(branches: Vec<CaseWhen>, else_expr: Self) -> Self {
        Self::Case { branches, else_expr: Box::new(else_expr) }
    }

    pub fn extract(field: DatePart, expr: Self) -> Self {
        Self::Extract { field, expr: Box::new(expr) }
    }

    pub fn date_trunc(field: DatePart, expr: Self, output_type: DataType) -> Self {
        Self::DateTrunc { field, expr: Box::new(expr), output_type }
    }

    pub fn alias(self, alias: impl Into<String>) -> Self {
        Self::Alias { expr: Box::new(self), alias: alias.into() }
    }

    /// Returns true when this expression tree contains a `REGEXP_REPLACE`.
    ///
    /// Used by capability labeling and runtime preflight paths that size
    /// replace-with-backrefs via the owning cudf output-size contract.
    pub fn contains_regexp_replace(&self) -> bool {
        match self {
            Self::RegexpReplace { .. } => true,
            Self::Not(expr)
            | Self::IsNull(expr)
            | Self::IsNotNull(expr)
            | Self::Cast { expr, .. }
            | Self::TryCast { expr, .. }
            | Self::Like { expr, .. }
            | Self::Substring { expr, .. }
            | Self::Lower(expr)
            | Self::Upper(expr)
            | Self::Trim { expr, .. }
            | Self::LTrim { expr, .. }
            | Self::RTrim { expr, .. }
            | Self::StringContains { expr, .. }
            | Self::StartsWith { expr, .. }
            | Self::EndsWith { expr, .. }
            | Self::StringLength { expr, .. }
            | Self::Abs(expr)
            | Self::UnaryMath { expr, .. }
            | Self::Round { expr, .. }
            | Self::InList { expr, .. }
            | Self::Extract { expr, .. }
            | Self::DateTrunc { expr, .. }
            | Self::Alias { expr, .. } => expr.contains_regexp_replace(),
            Self::Binary { left, right, .. }
            | Self::IsDistinctFrom { left, right }
            | Self::IsNotDistinctFrom { left, right } => {
                left.contains_regexp_replace() || right.contains_regexp_replace()
            },
            Self::Concat(exprs) => exprs.iter().any(Self::contains_regexp_replace),
            Self::Case { branches, else_expr } => {
                branches
                    .iter()
                    .any(|branch| branch.when.contains_regexp_replace() || branch.then.contains_regexp_replace())
                    || else_expr.contains_regexp_replace()
            },
            Self::Column(_) | Self::Literal(_) => false,
        }
    }
}

impl Not for Expr {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Column(column) => write!(f, "{column}"),
            Self::Literal(value) => write!(f, "{value}"),
            Self::Not(expr) => write!(f, "NOT {expr}"),
            Self::IsNull(expr) => write!(f, "{expr} IS NULL"),
            Self::IsNotNull(expr) => write!(f, "{expr} IS NOT NULL"),
            Self::Cast { expr, target_type } => write!(f, "CAST({expr} AS {target_type})"),
            Self::TryCast { expr, target_type } => write!(f, "TRY_CAST({expr} AS {target_type})"),
            Self::Binary { left, op, right, output_type } => {
                write!(f, "({left} {op} {right})")?;
                if let Some(output_type) = output_type {
                    write!(f, "::{output_type}")?;
                }
                Ok(())
            },
            Self::Like { expr, pattern, escape, negated } => {
                let operator = if *negated { "NOT LIKE" } else { "LIKE" };
                write!(f, "{expr} {operator} {pattern:?}")?;
                if let Some(escape) = escape {
                    write!(f, " ESCAPE {escape:?}")?;
                }
                Ok(())
            },
            Self::Substring { expr, start, length } => match length {
                Some(length) => write!(f, "SUBSTRING({expr} FROM {start} FOR {length})"),
                None => write!(f, "SUBSTRING({expr} FROM {start})"),
            },
            Self::Lower(expr) => write!(f, "LOWER({expr})"),
            Self::Upper(expr) => write!(f, "UPPER({expr})"),
            Self::Trim { expr, trim_chars } => write!(f, "TRIM({trim_chars:?} FROM {expr})"),
            Self::LTrim { expr, trim_chars } => write!(f, "LTRIM({expr}, {trim_chars:?})"),
            Self::RTrim { expr, trim_chars } => write!(f, "RTRIM({expr}, {trim_chars:?})"),
            Self::StringContains { expr, pattern } => write!(f, "CONTAINS({expr}, {pattern:?})"),
            Self::StartsWith { expr, prefix } => write!(f, "STARTS_WITH({expr}, {prefix:?})"),
            Self::EndsWith { expr, suffix } => write!(f, "ENDS_WITH({expr}, {suffix:?})"),
            Self::StringLength { expr, output_type } => write!(f, "CHARACTER_LENGTH({expr})::{output_type}"),
            Self::RegexpReplace { expr, pattern, replacement, output_type } => {
                write!(f, "REGEXP_REPLACE({expr}, {pattern:?}, {replacement:?})::{output_type}")
            },
            Self::Concat(exprs) => {
                f.write_str("CONCAT(")?;
                for (index, expr) in exprs.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{expr}")?;
                }
                f.write_str(")")
            },
            Self::Abs(expr) => write!(f, "ABS({expr})"),
            Self::UnaryMath { op, expr, output_type } => write!(f, "{op}({expr})::{output_type}"),
            Self::Round { expr, decimal_places, output_type } => {
                write!(f, "ROUND({expr}, {decimal_places})::{output_type}")
            },
            Self::InList { expr, list, negated } => {
                let operator = if *negated { "NOT IN" } else { "IN" };
                write!(f, "{expr} {operator} (")?;
                for (index, item) in list.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str(")")
            },
            Self::IsDistinctFrom { left, right } => write!(f, "({left} IS DISTINCT FROM {right})"),
            Self::IsNotDistinctFrom { left, right } => write!(f, "({left} IS NOT DISTINCT FROM {right})"),
            Self::Case { branches, else_expr } => {
                f.write_str("CASE")?;
                for branch in branches {
                    write!(f, " WHEN {} THEN {}", branch.when, branch.then)?;
                }
                write!(f, " ELSE {else_expr} END")
            },
            Self::Extract { field, expr } => write!(f, "EXTRACT({field} FROM {expr})"),
            Self::DateTrunc { field, expr, output_type } => {
                write!(f, "DATE_TRUNC({field}, {expr})::{output_type}")
            },
            Self::Alias { expr, alias } => write!(f, "{expr} AS {alias}"),
        }
    }
}

/// One `WHEN predicate THEN value` branch of a searched SQL `CASE` expression.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseWhen {
    /// Boolean predicate for this branch.
    pub when: Expr,
    /// Result expression chosen when `when` evaluates true.
    pub then: Expr,
}

impl CaseWhen {
    /// Build one searched-case branch.
    pub fn new(when: Expr, then: Expr) -> Self {
        Self { when, then }
    }
}

/// Date or timestamp field supported by `EXTRACT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DatePart {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

impl fmt::Display for DatePart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Year => "YEAR",
            Self::Month => "MONTH",
            Self::Day => "DAY",
            Self::Hour => "HOUR",
            Self::Minute => "MINUTE",
            Self::Second => "SECOND",
            Self::Millisecond => "MILLISECOND",
            Self::Microsecond => "MICROSECOND",
            Self::Nanosecond => "NANOSECOND",
        })
    }
}

/// One-argument floating-point math function in native expression IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryMathOp {
    /// Square root; negative inputs produce NaN.
    Sqrt,
    /// Natural logarithm; zero produces negative infinity and negative inputs
    /// produce NaN.
    Ln,
    /// Natural exponential `e ^ x`.
    Exp,
}

impl fmt::Display for UnaryMathOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sqrt => "SQRT",
            Self::Ln => "LN",
            Self::Exp => "EXP",
        })
    }
}

/// Binary operator in native expression IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Arithmetic addition. Temporal handling is context/type-dependent.
    Add,
    /// Arithmetic subtraction. Temporal handling is context/type-dependent.
    Subtract,
    Multiply,
    Divide,
    Modulo,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Modulo => "%",
            Self::BitwiseAnd => "&",
            Self::BitwiseOr => "|",
            Self::BitwiseXor => "^",
            Self::ShiftLeft => "<<",
            Self::ShiftRight => ">>",
            Self::Eq => "=",
            Self::NotEq => "!=",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
            Self::And => "AND",
            Self::Or => "OR",
        })
    }
}

/// Aggregate operation descriptor for a native aggregate node.
///
/// `group_exprs` names the grouping keys. An empty key list represents a
/// global aggregate. `aggr_exprs` lists logical aggregate outputs in output
/// schema order, using each expression's `alias` as the externally visible
/// field name.
///
/// Logical `AVG` is intentionally represented here as one aggregate expression.
/// Execution code must expand it into explicit `SUM` plus non-null `COUNT`
/// state whenever partial/final aggregation is involved, and finalize with
/// `SUM / COUNT`. It must never merge AVG states by averaging partial averages.
///
/// Logical `StddevSamp` defaults to DataFusion sample standard-deviation
/// semantics: null inputs are ignored, inputs are accumulated as Float64
/// Welford state (`count`, `mean`, `m2`), counts 0 and 1 produce null, and
/// finalization is `sqrt(m2 / (count - 1))`. Execution may explicitly select
/// the GPU-native tolerant contract, which keeps SQL null/ddof semantics but
/// does not preserve DataFusion's order-sensitive final bits.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateSpec {
    /// Grouping key expressions in output key order.
    pub group_exprs: Vec<Expr>,
    /// Aggregate expressions in output aggregate order.
    pub aggr_exprs: Vec<AggregateExpr>,
    /// Optional grouping-set masks appended to the group expression contract.
    pub grouping_sets: Option<GroupingSetsSpec>,
}

impl AggregateSpec {
    /// Creates a standard aggregate without grouping sets.
    pub fn new(group_exprs: Vec<Expr>, aggr_exprs: Vec<AggregateExpr>) -> Self {
        Self { group_exprs, aggr_exprs, grouping_sets: None }
    }

    /// Attaches GROUPING SETS / ROLLUP / CUBE masks to this aggregate.
    pub fn with_grouping_sets(mut self, grouping_sets: GroupingSetsSpec) -> Self {
        self.grouping_sets = Some(grouping_sets);
        self
    }
}

/// GROUPING SETS / ROLLUP / CUBE grouping masks.
///
/// `groups` follows DataFusion's physical `PhysicalGroupBy` convention: each
/// mask has one entry per `group_exprs` item, and `true` means that grouping
/// key is replaced by a typed NULL for this grouping set. `grouping_id_alias`
/// names the generated grouping id key appended after the group expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupingSetsSpec {
    /// Grouping-set masks; `true` means the corresponding key is nulled.
    pub groups: Vec<Vec<bool>>,
    /// Output alias for the generated grouping-id column.
    pub grouping_id_alias: String,
}

impl GroupingSetsSpec {
    /// Creates grouping-set metadata from masks and a grouping-id alias.
    pub fn new(groups: Vec<Vec<bool>>, grouping_id_alias: impl Into<String>) -> Self {
        Self { groups, grouping_id_alias: grouping_id_alias.into() }
    }

    /// Returns true when at least one grouping-set mask is present.
    pub fn is_grouping_sets(&self) -> bool {
        !self.groups.is_empty()
    }
}

/// One logical aggregate expression and its output alias.
///
/// `expr` is the aggregate argument. For `COUNT`, normal column expressions
/// count non-null values. Planner-produced count-star forms use a non-null
/// literal or the native zero-column count source representation.
///
/// When `filter` is present, it is evaluated for every input row before this
/// aggregate is updated. Only rows where the predicate evaluates true
/// contribute to this aggregate; false and null predicate values are ignored.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateExpr {
    /// Aggregate function to apply.
    pub function: AggregateFunction,
    /// Aggregate argument expression.
    pub expr: Expr,
    /// Output field alias for this aggregate.
    pub alias: String,
    /// Whether SQL DISTINCT semantics are requested for the argument.
    pub distinct: bool,
    /// Optional per-aggregate SQL `FILTER (WHERE ...)` predicate.
    pub filter: Option<Expr>,
}

impl AggregateExpr {
    /// Creates a non-DISTINCT aggregate expression.
    pub fn new(function: AggregateFunction, expr: Expr, alias: impl Into<String>) -> Self {
        Self { function, expr, alias: alias.into(), distinct: false, filter: None }
    }

    /// Sets the SQL DISTINCT flag.
    pub fn with_distinct(mut self, distinct: bool) -> Self {
        self.distinct = distinct;
        self
    }

    /// Sets an optional SQL aggregate filter predicate.
    pub fn with_filter(mut self, filter: Option<Expr>) -> Self {
        self.filter = filter;
        self
    }
}

/// Logical aggregate functions supported by the native IR.
///
/// `Avg` carries SQL AVG semantics: null inputs are ignored, an all-null or
/// empty input produces null, and distributed execution must track SUM and
/// non-null COUNT as state before final division.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateFunction {
    /// SQL SUM over non-null inputs.
    Sum,
    /// SQL MIN over non-null inputs.
    Min,
    /// SQL MAX over non-null inputs.
    Max,
    /// SQL COUNT over non-null inputs or count-star representation.
    Count,
    /// Exact SQL count-distinct over non-null inputs.
    ///
    /// Nulls do not contribute to the count. The logical output is a count
    /// integer and must be explicitly cast to the frontend-declared output
    /// dtype when libcudf returns a different integer width. Partial/final
    /// execution must not merge per-partition `NUnique` counts by summing
    /// them; bounded execution must aggregate over the complete raw input for
    /// the distinct scope.
    NUnique,
    /// SQL AVG represented logically; execution expands it to sum/count state.
    Avg,
    /// SQL median over non-null inputs.
    ///
    /// Median is not mergeable through the existing partial/final aggregate
    /// state model and must remain single-stage unless an explicit state
    /// representation is added.
    Median,
    /// SQL sample standard deviation.
    StddevSamp,
    /// SQL population standard deviation.
    StddevPop,
    /// SQL sample variance.
    VarSamp,
    /// SQL population variance.
    VarPop,
}

impl AggregateFunction {
    /// Returns true for floating statistical aggregates whose SQL result is
    /// governed by the statistical aggregate execution contract.
    pub const fn is_floating_statistical(self) -> bool {
        matches!(self, Self::StddevSamp | Self::StddevPop | Self::VarSamp | Self::VarPop)
    }

    /// Degrees-of-freedom correction for floating statistical aggregates.
    pub const fn statistical_ddof(self) -> Option<u32> {
        match self {
            Self::StddevSamp | Self::VarSamp => Some(1),
            Self::StddevPop | Self::VarPop => Some(0),
            _ => None,
        }
    }

    /// Stable lower-case function label used in native diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::Count => "count",
            Self::NUnique => "nunique",
            Self::Avg => "avg",
            Self::Median => "median",
            Self::StddevSamp => "stddev_samp",
            Self::StddevPop => "stddev_pop",
            Self::VarSamp => "var_samp",
            Self::VarPop => "var_pop",
        }
    }
}

/// Ordered list of sort keys for a native sort node.
///
/// Sort expressions are evaluated in list order: later keys break ties from
/// earlier keys. A runnable sort node is expected to contain at least one key;
/// an empty list is useful only as an invalid/transient state before validation.
#[derive(Debug, Clone, PartialEq)]
pub struct SortSpec {
    /// Sort keys in precedence order.
    pub expressions: Vec<SortExpr>,
}

impl SortSpec {
    /// Build a sort specification from ordered key expressions.
    pub fn new(expressions: Vec<SortExpr>) -> Self {
        Self { expressions }
    }
}

/// One sort key and its ordering options.
#[derive(Debug, Clone, PartialEq)]
pub struct SortExpr {
    /// Expression that produces the key value for each row.
    pub expr: Expr,
    /// `true` for descending order, `false` for ascending order.
    pub descending: bool,
    /// `true` places nulls before non-nulls, `false` places nulls after them.
    ///
    /// Null placement is explicit and independent of sort direction.
    pub nulls_first: bool,
}

impl SortExpr {
    /// Build one sort key.
    pub fn new(expr: Expr, descending: bool, nulls_first: bool) -> Self {
        Self { expr, descending, nulls_first }
    }
}

/// Row-count limit with SQL offset/fetch semantics.
///
/// `skip` rows are discarded first. When `fetch` is `Some`, at most that many
/// rows are returned after `skip`; when `fetch` is `None`, all remaining rows
/// are returned. If `skip` is greater than the input row count, the result is
/// empty. Finite top-k planning is valid only when `fetch` is `Some` and
/// `skip + fetch` fits in `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LimitSpec {
    /// Number of leading rows to discard.
    pub skip: usize,
    /// Maximum number of rows to return after `skip`; `None` means no fetch bound.
    pub fetch: Option<usize>,
}

impl LimitSpec {
    /// Build a limit with explicit `skip` and `fetch` values.
    pub fn new(skip: usize, fetch: impl Into<Option<usize>>) -> Self {
        Self { skip, fetch: fetch.into() }
    }

    /// Build a limit that returns at most `fetch` rows from the start.
    pub fn fetch(fetch: usize) -> Self {
        Self { skip: 0, fetch: Some(fetch) }
    }

    /// Build an offset-only limit that returns all rows after `skip`.
    pub fn offset(skip: usize) -> Self {
        Self { skip, fetch: None }
    }
}
