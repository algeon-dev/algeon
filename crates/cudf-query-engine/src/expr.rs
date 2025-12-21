use std::fmt;

use arrow::datatypes::DataType;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),
    Literal(LiteralValue),
    Not(Box<Expr>),
    Cast { expr: Box<Expr>, target_type: DataType },
    Binary { left: Box<Expr>, op: BinaryOp, right: Box<Expr> },
    Alias { expr: Box<Expr>, alias: String },
}

impl Expr {
    pub fn column(name: impl Into<String>) -> Self {
        Self::Column(name.into())
    }

    pub fn literal(value: LiteralValue) -> Self {
        Self::Literal(value)
    }

    pub fn binary(left: Self, op: BinaryOp, right: Self) -> Self {
        Self::Binary { left: Box::new(left), op, right: Box::new(right) }
    }

    pub fn not(expr: Self) -> Self {
        Self::Not(Box::new(expr))
    }

    pub fn cast(expr: Self, target_type: DataType) -> Self {
        Self::Cast { expr: Box::new(expr), target_type }
    }

    pub fn alias(self, alias: impl Into<String>) -> Self {
        Self::Alias { expr: Box::new(self), alias: alias.into() }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Column(name) => f.write_str(name),
            Self::Literal(value) => write!(f, "{value}"),
            Self::Not(expr) => write!(f, "NOT {expr}"),
            Self::Cast { expr, target_type } => write!(f, "CAST({expr} AS {target_type})"),
            Self::Binary { left, op, right } => write!(f, "({left} {op} {right})"),
            Self::Alias { expr, alias } => write!(f, "{expr} AS {alias}"),
        }
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
    String(String),
    Null,
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
            Self::String(value) => write!(f, "{value:?}"),
            Self::Null => f.write_str("NULL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
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
