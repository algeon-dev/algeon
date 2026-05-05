use std::{fmt, ops::Not};

use arrow::datatypes::{DataType, IntervalUnit, TimeUnit};

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(ColumnRef),
    Literal(LiteralValue),
    Not(Box<Expr>),
    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),
    Cast { expr: Box<Expr>, target_type: DataType },
    Binary { left: Box<Expr>, op: BinaryOp, right: Box<Expr> },
    Like { expr: Box<Expr>, pattern: String, escape: Option<String>, negated: bool },
    Substring { expr: Box<Expr>, start: i32, length: Option<i32> },
    InList { expr: Box<Expr>, list: Vec<Expr>, negated: bool },
    Case { branches: Vec<CaseWhen>, else_expr: Box<Expr> },
    Extract { field: DatePart, expr: Box<Expr> },
    Alias { expr: Box<Expr>, alias: String },
}

impl Expr {
    pub fn column(name: impl Into<String>) -> Self {
        Self::Column(ColumnRef::unqualified(name))
    }

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
        Self::Binary { left: Box::new(left), op, right: Box::new(right) }
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

    pub fn substring(expr: Self, start: i32, length: Option<i32>) -> Self {
        Self::Substring { expr: Box::new(expr), start, length }
    }

    pub fn in_list(expr: Self, list: Vec<Self>) -> Self {
        Self::InList { expr: Box::new(expr), list, negated: false }
    }

    pub fn not_in_list(expr: Self, list: Vec<Self>) -> Self {
        Self::InList { expr: Box::new(expr), list, negated: true }
    }

    pub fn case(branches: Vec<CaseWhen>, else_expr: Self) -> Self {
        Self::Case { branches, else_expr: Box::new(else_expr) }
    }

    pub fn extract(field: DatePart, expr: Self) -> Self {
        Self::Extract { field, expr: Box::new(expr) }
    }

    pub fn alias(self, alias: impl Into<String>) -> Self {
        Self::Alias { expr: Box::new(self), alias: alias.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColumnRef {
    qualifier: Option<ColumnQualifier>,
    name: String,
    display_name: String,
}

impl ColumnRef {
    pub fn unqualified(name: impl Into<String>) -> Self {
        let name = name.into();
        Self { qualifier: None, display_name: name.clone(), name }
    }

    pub fn input(input: usize, name: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            qualifier: Some(ColumnQualifier::Input { index: input }),
            name: name.into(),
            display_name: display_name.into(),
        }
    }

    pub fn node(node_id: impl Into<String>, name: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            qualifier: Some(ColumnQualifier::Node { id: node_id.into() }),
            name: name.into(),
            display_name: display_name.into(),
        }
    }

    pub fn relation(relation: impl Into<String>, name: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            qualifier: Some(ColumnQualifier::Relation { name: relation.into() }),
            name: name.into(),
            display_name: display_name.into(),
        }
    }

    pub fn qualifier(&self) -> Option<&ColumnQualifier> {
        self.qualifier.as_ref()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn is_qualified(&self) -> bool {
        self.qualifier.is_some()
    }
}

impl fmt::Display for ColumnRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(qualifier) = &self.qualifier {
            write!(f, "{qualifier}.{}", self.name)
        } else {
            f.write_str(&self.name)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColumnQualifier {
    Input { index: usize },
    Node { id: String },
    Relation { name: String },
}

impl fmt::Display for ColumnQualifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { index } => write!(f, "input{index}"),
            Self::Node { id } => write!(f, "{id}"),
            Self::Relation { name } => f.write_str(name),
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
            Self::Binary { left, op, right } => write!(f, "({left} {op} {right})"),
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

#[derive(Debug, Clone, PartialEq)]
pub struct CaseWhen {
    pub when: Expr,
    pub then: Expr,
}

impl CaseWhen {
    pub fn new(when: Expr, then: Expr) -> Self {
        Self { when, then }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DatePart {
    Year,
    Month,
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

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Utf8(String),
    LargeUtf8(String),
    Decimal128 { raw: i128, precision: u8, scale: i8 },
    Date32(i32),
    Timestamp { raw: i64, unit: TimeUnit },
    Duration { raw: i64, unit: TimeUnit },
    IntervalMonthDayNano { months: i32, days: i32, nanoseconds: i64 },
    Null(DataType),
}

impl LiteralValue {
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Boolean(_) => DataType::Boolean,
            Self::Int8(_) => DataType::Int8,
            Self::Int16(_) => DataType::Int16,
            Self::Int32(_) => DataType::Int32,
            Self::Int64(_) => DataType::Int64,
            Self::UInt8(_) => DataType::UInt8,
            Self::UInt16(_) => DataType::UInt16,
            Self::UInt32(_) => DataType::UInt32,
            Self::UInt64(_) => DataType::UInt64,
            Self::Float32(_) => DataType::Float32,
            Self::Float64(_) => DataType::Float64,
            Self::Utf8(_) => DataType::Utf8,
            Self::LargeUtf8(_) => DataType::LargeUtf8,
            Self::Decimal128 { precision, scale, .. } => DataType::Decimal128(*precision, *scale),
            Self::Date32(_) => DataType::Date32,
            Self::Timestamp { unit, .. } => DataType::Timestamp(*unit, None),
            Self::Duration { unit, .. } => DataType::Duration(*unit),
            Self::IntervalMonthDayNano { .. } => DataType::Interval(IntervalUnit::MonthDayNano),
            Self::Null(data_type) => data_type.clone(),
        }
    }
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean(value) => write!(f, "{value}"),
            Self::Int8(value) => write!(f, "{value}"),
            Self::Int16(value) => write!(f, "{value}"),
            Self::Int32(value) => write!(f, "{value}"),
            Self::Int64(value) => write!(f, "{value}"),
            Self::UInt8(value) => write!(f, "{value}"),
            Self::UInt16(value) => write!(f, "{value}"),
            Self::UInt32(value) => write!(f, "{value}"),
            Self::UInt64(value) => write!(f, "{value}"),
            Self::Float32(value) => write!(f, "{value}"),
            Self::Float64(value) => write!(f, "{value}"),
            Self::Utf8(value) | Self::LargeUtf8(value) => write!(f, "{value:?}"),
            Self::Decimal128 { raw, precision, scale } => {
                write!(f, "DECIMAL128({raw}, precision={precision}, scale={scale})")
            },
            Self::Date32(value) => write!(f, "DATE32({value})"),
            Self::Timestamp { raw, unit } => write!(f, "TIMESTAMP({raw}, {unit:?})"),
            Self::Duration { raw, unit } => write!(f, "DURATION({raw}, {unit:?})"),
            Self::IntervalMonthDayNano { months, days, nanoseconds } => {
                write!(f, "INTERVAL(months={months}, days={days}, nanoseconds={nanoseconds})")
            },
            Self::Null(data_type) => write!(f, "NULL::{data_type}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
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

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateSpec {
    pub group_exprs: Vec<Expr>,
    pub aggr_exprs: Vec<AggregateExpr>,
}

impl AggregateSpec {
    pub fn new(group_exprs: Vec<Expr>, aggr_exprs: Vec<AggregateExpr>) -> Self {
        Self { group_exprs, aggr_exprs }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateFunction {
    Sum,
    Min,
    Max,
    Count,
    Avg,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SortSpec {
    pub expressions: Vec<SortExpr>,
}

impl SortSpec {
    pub fn new(expressions: Vec<SortExpr>) -> Self {
        Self { expressions }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SortExpr {
    pub expr: Expr,
    pub descending: bool,
    pub nulls_first: bool,
}

impl SortExpr {
    pub fn new(expr: Expr, descending: bool, nulls_first: bool) -> Self {
        Self { expr, descending, nulls_first }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LimitSpec {
    pub skip: usize,
    pub fetch: usize,
}

impl LimitSpec {
    pub fn new(skip: usize, fetch: usize) -> Self {
        Self { skip, fetch }
    }

    pub fn fetch(fetch: usize) -> Self {
        Self { skip: 0, fetch }
    }
}
