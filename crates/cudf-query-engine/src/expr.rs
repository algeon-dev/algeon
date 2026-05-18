use std::{fmt, ops::Not};

use arrow::datatypes::DataType;

mod column;
mod literal;

pub use self::{
    column::{ColumnQualifier, ColumnRef},
    literal::LiteralValue,
};

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
    /// SQL logical negation of a boolean expression.
    Not(Box<Expr>),
    /// SQL `IS NULL` predicate.
    IsNull(Box<Expr>),
    /// SQL `IS NOT NULL` predicate.
    IsNotNull(Box<Expr>),
    /// Explicit cast to an Arrow data type.
    Cast {
        /// Expression whose result is cast.
        expr: Box<Expr>,
        /// Desired output type after the cast.
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
    /// SQL `LOWER(expr)` string expression.
    Lower(Box<Expr>),
    /// SQL `UPPER(expr)` string expression.
    Upper(Box<Expr>),
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
    /// Output alias for projection and aggregate argument naming.
    Alias {
        /// Expression being renamed.
        expr: Box<Expr>,
        /// Field name exposed to downstream schemas and diagnostics.
        alias: String,
    },
}

impl Expr {
    /// Build an unqualified column expression.
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

    /// Build a column expression qualified by a plan-node identifier.
    pub fn node_column(node_id: impl Into<String>, name: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self::Column(ColumnRef::node(node_id, name, display_name))
    }

    /// Build a literal expression.
    pub fn literal(value: LiteralValue) -> Self {
        Self::Literal(value)
    }

    /// Build a binary operation expression.
    pub fn binary(left: Self, op: BinaryOp, right: Self) -> Self {
        Self::Binary { left: Box::new(left), op, right: Box::new(right), output_type: None }
    }

    /// Build a binary operation expression with a declared result type.
    pub fn binary_with_output_type(left: Self, op: BinaryOp, right: Self, output_type: DataType) -> Self {
        Self::Binary { left: Box::new(left), op, right: Box::new(right), output_type: Some(output_type) }
    }

    /// Build a SQL logical negation expression.
    pub fn logical_not(expr: Self) -> Self {
        Self::Not(Box::new(expr))
    }

    /// Build a SQL `IS NULL` predicate.
    pub fn is_null(expr: Self) -> Self {
        Self::IsNull(Box::new(expr))
    }

    /// Build a SQL `IS NOT NULL` predicate.
    pub fn is_not_null(expr: Self) -> Self {
        Self::IsNotNull(Box::new(expr))
    }

    /// Build an explicit cast expression.
    pub fn cast(expr: Self, target_type: DataType) -> Self {
        Self::Cast { expr: Box::new(expr), target_type }
    }

    /// Build a SQL `LIKE` predicate without an explicit escape marker.
    pub fn like(expr: Self, pattern: impl Into<String>) -> Self {
        Self::Like { expr: Box::new(expr), pattern: pattern.into(), escape: None, negated: false }
    }

    /// Build a SQL `NOT LIKE` predicate without an explicit escape marker.
    pub fn not_like(expr: Self, pattern: impl Into<String>) -> Self {
        Self::Like { expr: Box::new(expr), pattern: pattern.into(), escape: None, negated: true }
    }

    /// Build a SQL `LIKE` predicate with an explicit escape marker.
    pub fn like_escape(expr: Self, pattern: impl Into<String>, escape: impl Into<String>) -> Self {
        Self::Like { expr: Box::new(expr), pattern: pattern.into(), escape: Some(escape.into()), negated: false }
    }

    /// Build a SQL `NOT LIKE` predicate with an explicit escape marker.
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

    /// Build a SQL `LOWER(expr)` string expression.
    pub fn lower(expr: Self) -> Self {
        Self::Lower(Box::new(expr))
    }

    /// Build a SQL `UPPER(expr)` string expression.
    pub fn upper(expr: Self) -> Self {
        Self::Upper(Box::new(expr))
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

    /// Build a searched SQL `CASE` expression.
    pub fn case(branches: Vec<CaseWhen>, else_expr: Self) -> Self {
        Self::Case { branches, else_expr: Box::new(else_expr) }
    }

    /// Build a SQL `EXTRACT(field FROM expr)` expression.
    pub fn extract(field: DatePart, expr: Self) -> Self {
        Self::Extract { field, expr: Box::new(expr) }
    }

    /// Wrap this expression in an output alias.
    pub fn alias(self, alias: impl Into<String>) -> Self {
        Self::Alias { expr: Box::new(self), alias: alias.into() }
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
            Self::Case { branches, else_expr } => {
                f.write_str("CASE")?;
                for branch in branches {
                    write!(f, " WHEN {} THEN {}", branch.when, branch.then)?;
                }
                write!(f, " ELSE {else_expr} END")
            },
            Self::Extract { field, expr } => write!(f, "EXTRACT({field} FROM {expr})"),
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
    /// Calendar year field.
    Year,
    /// Calendar month field.
    Month,
    /// Calendar day-of-month field.
    Day,
}

impl fmt::Display for DatePart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Year => "YEAR",
            Self::Month => "MONTH",
            Self::Day => "DAY",
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
    /// Numeric multiplication.
    Multiply,
    /// Numeric division.
    Divide,
    /// Numeric remainder.
    Modulo,
    /// Equality comparison.
    Eq,
    /// Inequality comparison.
    NotEq,
    /// Less-than comparison.
    Lt,
    /// Less-than-or-equal comparison.
    LtEq,
    /// Greater-than comparison.
    Gt,
    /// Greater-than-or-equal comparison.
    GtEq,
    /// Boolean conjunction.
    And,
    /// Boolean disjunction.
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
/// Logical `StddevSamp` carries DataFusion sample standard-deviation semantics:
/// null inputs are ignored, inputs are accumulated as Float64 Welford state
/// (`count`, `mean`, `m2`), counts 0 and 1 produce null, and finalization is
/// `sqrt(m2 / (count - 1))`.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateSpec {
    pub group_exprs: Vec<Expr>,
    pub aggr_exprs: Vec<AggregateExpr>,
    pub grouping_sets: Option<GroupingSetsSpec>,
}

impl AggregateSpec {
    pub fn new(group_exprs: Vec<Expr>, aggr_exprs: Vec<AggregateExpr>) -> Self {
        Self { group_exprs, aggr_exprs, grouping_sets: None }
    }

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
    pub groups: Vec<Vec<bool>>,
    pub grouping_id_alias: String,
}

impl GroupingSetsSpec {
    pub fn new(groups: Vec<Vec<bool>>, grouping_id_alias: impl Into<String>) -> Self {
        Self { groups, grouping_id_alias: grouping_id_alias.into() }
    }

    pub fn is_grouping_sets(&self) -> bool {
        !self.groups.is_empty()
    }
}

/// One logical aggregate expression and its output alias.
///
/// `expr` is the aggregate argument. For `COUNT`, normal column expressions
/// count non-null values. Planner-produced count-star forms use a non-null
/// literal or the native zero-column count source representation.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateExpr {
    pub function: AggregateFunction,
    pub expr: Expr,
    pub alias: String,
}

impl AggregateExpr {
    pub fn new(function: AggregateFunction, expr: Expr, alias: impl Into<String>) -> Self {
        Self { function, expr, alias: alias.into() }
    }
}

/// Logical aggregate functions supported by the native IR.
///
/// `Avg` carries SQL AVG semantics: null inputs are ignored, an all-null or
/// empty input produces null, and distributed execution must track SUM and
/// non-null COUNT as state before final division.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateFunction {
    Sum,
    Min,
    Max,
    Count,
    Avg,
    StddevSamp,
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
/// `skip` rows are discarded first, then up to `fetch` rows are returned. If
/// `skip` is greater than the input row count, the result is empty. If fewer
/// than `fetch` rows remain after skipping, all remaining rows are returned.
/// Planning and execution require `skip + fetch` to fit in `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LimitSpec {
    /// Number of leading rows to discard.
    pub skip: usize,
    /// Maximum number of rows to return after `skip`.
    pub fetch: usize,
}

impl LimitSpec {
    /// Build a limit with explicit `skip` and `fetch` values.
    pub fn new(skip: usize, fetch: usize) -> Self {
        Self { skip, fetch }
    }

    /// Build a limit that returns at most `fetch` rows from the start.
    pub fn fetch(fetch: usize) -> Self {
        Self { skip: 0, fetch }
    }
}
