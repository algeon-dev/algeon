//! Query-oriented public request types.
//!
//! This module is a stable lowering target for query engines. It reuses the
//! eager [`Expr`](crate::Expr) DSL and adds named projection, aggregate, and
//! sort request types so downstream adapters do not need to maintain a second
//! expression vocabulary just to carry output names and query operator shape.

use std::collections::HashSet;

use crate::{
    DataFrame, Expr, SortOptions,
    core::{
        column::Column,
        error::{Error, ErrorCode, Result},
        scalar::Scalar,
        types::{DataType, TimestampResolution, TypeId},
    },
    frame::series::{Series, reduction::infer_reduce_output_type},
    internal::resolve::duplicate_column_name_error_with_contract,
    ops::{
        binary::BinaryOperator,
        expr::{
            IntoExpr, col, lit, lit_decimal32_raw, lit_decimal64_raw, lit_decimal128_raw, lit_duration_raw, lit_null,
            lit_timestamp_raw,
        },
        groupby::{AggSpec, GroupByAgg},
        reduction::ReduceAgg,
    },
};

const QUERY_TEMP_PREFIX: &str = "__cudf_query_tmp_";

/// A named expression used by query projection and grouping requests.
#[derive(Clone)]
pub struct NamedExpr {
    pub expr: Expr,
    pub name: String,
}

impl NamedExpr {
    pub fn new(expr: impl IntoExpr, name: impl Into<String>) -> Self {
        Self { expr: expr.into_expr(), name: name.into() }
    }
}

/// Literal values accepted by query lowering.
#[derive(Clone)]
pub enum QueryLiteral {
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
    Null(DataType),
    Decimal32Raw { raw: i32, scale: i32 },
    Decimal64Raw { raw: i64, scale: i32 },
    Decimal128Raw { raw: i128, scale: i32 },
    TimestampRaw { raw: i64, resolution: TimestampResolution },
    DurationRaw { raw: i64, resolution: TimestampResolution },
}

impl QueryLiteral {
    pub fn into_expr(self) -> Expr {
        match self {
            Self::Boolean(value) => lit(value),
            Self::Int8(value) => lit(value),
            Self::Int16(value) => lit(value),
            Self::Int32(value) => lit(value),
            Self::Int64(value) => lit(value),
            Self::UInt8(value) => lit(value),
            Self::UInt16(value) => lit(value),
            Self::UInt32(value) => lit(value),
            Self::UInt64(value) => lit(value),
            Self::Float32(value) => lit(value),
            Self::Float64(value) => lit(value),
            Self::Utf8(value) => lit(value),
            Self::Null(dtype) => lit_null(dtype),
            Self::Decimal32Raw { raw, scale } => lit_decimal32_raw(raw, scale),
            Self::Decimal64Raw { raw, scale } => lit_decimal64_raw(raw, scale),
            Self::Decimal128Raw { raw, scale } => lit_decimal128_raw(raw, scale),
            Self::TimestampRaw { raw, resolution } => lit_timestamp_raw(raw, resolution),
            Self::DurationRaw { raw, resolution } => lit_duration_raw(raw, resolution),
        }
    }
}

impl IntoExpr for QueryLiteral {
    fn into_expr(self) -> Expr {
        QueryLiteral::into_expr(self)
    }
}

/// Binary operators accepted by query lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    FloorDiv,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
}

impl QueryBinaryOp {
    pub fn expr<L, R>(self, lhs: L, rhs: R) -> Expr
    where
        L: IntoExpr,
        R: IntoExpr,
    {
        Expr::BinaryOp { lhs: Box::new(lhs.into_expr()), op: self.to_binary_operator(), rhs: Box::new(rhs.into_expr()) }
    }

    fn to_binary_operator(self) -> BinaryOperator {
        match self {
            Self::Add => BinaryOperator::Add,
            Self::Sub => BinaryOperator::Sub,
            Self::Mul => BinaryOperator::Mul,
            Self::Div => BinaryOperator::Div,
            Self::Mod => BinaryOperator::Mod,
            Self::Pow => BinaryOperator::Pow,
            Self::FloorDiv => BinaryOperator::FloorDiv,
            Self::Equal => BinaryOperator::Equal,
            Self::NotEqual => BinaryOperator::NotEqual,
            Self::Less => BinaryOperator::Less,
            Self::Greater => BinaryOperator::Greater,
            Self::LessEqual => BinaryOperator::LessEqual,
            Self::GreaterEqual => BinaryOperator::GreaterEqual,
            Self::LogicalAnd => BinaryOperator::LogicalAnd,
            Self::LogicalOr => BinaryOperator::LogicalOr,
        }
    }
}

impl Expr {
    pub fn column(name: impl Into<String>) -> Self {
        col(name)
    }

    pub fn typed_literal(value: QueryLiteral) -> Self {
        value.into_expr()
    }

    pub fn binary<L, R>(lhs: L, op: QueryBinaryOp, rhs: R) -> Self
    where
        L: IntoExpr,
        R: IntoExpr,
    {
        op.expr(lhs, rhs)
    }
}

/// Query aggregate function names and parameters.
#[derive(Debug, Clone)]
pub enum QueryAggregateFunction {
    Sum,
    Min,
    Max,
    Count { include_nulls: bool },
    Mean,
    Median,
    NUnique,
    Std { ddof: u32 },
    Variance { ddof: u32 },
}

impl QueryAggregateFunction {
    /// Return the public output dtype contract for this aggregate over an
    /// input dtype.
    ///
    /// This is planner-facing metadata. Execution still validates against the
    /// concrete input column and libcudf may reject unsupported dtype/function
    /// combinations at runtime.
    pub fn output_dtype(&self, input: &DataType) -> Result<DataType> {
        match self {
            Self::Count { .. } | Self::NUnique => Ok(DataType::new(TypeId::Int32)),
            Self::Sum => infer_reduce_output_type(input, &ReduceAgg::Sum, "query::aggregate_output_dtype"),
            Self::Min => infer_reduce_output_type(input, &ReduceAgg::Min, "query::aggregate_output_dtype"),
            Self::Max => infer_reduce_output_type(input, &ReduceAgg::Max, "query::aggregate_output_dtype"),
            Self::Mean => infer_reduce_output_type(input, &ReduceAgg::Mean, "query::aggregate_output_dtype"),
            Self::Median => infer_reduce_output_type(input, &ReduceAgg::Median, "query::aggregate_output_dtype"),
            Self::Std { ddof } => {
                infer_reduce_output_type(input, &ReduceAgg::Std { ddof: *ddof }, "query::aggregate_output_dtype")
            },
            Self::Variance { ddof } => {
                infer_reduce_output_type(input, &ReduceAgg::Variance { ddof: *ddof }, "query::aggregate_output_dtype")
            },
        }
    }

    fn to_groupby_agg(&self) -> GroupByAgg {
        match self {
            Self::Sum => GroupByAgg::Sum,
            Self::Min => GroupByAgg::Min,
            Self::Max => GroupByAgg::Max,
            Self::Count { include_nulls } => GroupByAgg::Count { include_nulls: *include_nulls },
            Self::Mean => GroupByAgg::Mean,
            Self::Median => GroupByAgg::Median,
            Self::NUnique => GroupByAgg::NUnique,
            Self::Std { ddof } => GroupByAgg::Std { ddof: *ddof },
            Self::Variance { ddof } => GroupByAgg::Variance { ddof: *ddof },
        }
    }

    fn reduce_global(&self, input: &Series) -> Result<Scalar> {
        match self {
            Self::Sum => input.sum(),
            Self::Min => input.min(),
            Self::Max => input.max(),
            Self::Count { include_nulls } => count_scalar(input, *include_nulls),
            Self::Mean => input.mean(),
            Self::Median => input.median(),
            Self::NUnique => usize_count_scalar(input, input.nunique()?),
            Self::Std { ddof } => input.std(*ddof),
            Self::Variance { ddof } => input.var(*ddof),
        }
    }
}

/// A named aggregate expression.
#[derive(Clone)]
pub struct QueryAggregateExpr {
    pub function: QueryAggregateFunction,
    pub input: Expr,
    pub output_name: String,
}

impl QueryAggregateExpr {
    pub fn new(function: QueryAggregateFunction, input: impl IntoExpr, output_name: impl Into<String>) -> Self {
        Self { function, input: input.into_expr(), output_name: output_name.into() }
    }
}

/// Complete aggregate request for one query aggregate operator.
pub struct QueryAggregateSpec {
    pub group_exprs: Vec<NamedExpr>,
    pub aggr_exprs: Vec<QueryAggregateExpr>,
}

impl QueryAggregateSpec {
    pub fn new<G, A>(group_exprs: G, aggr_exprs: A) -> Self
    where
        G: IntoIterator<Item = NamedExpr>,
        A: IntoIterator<Item = QueryAggregateExpr>,
    {
        Self { group_exprs: group_exprs.into_iter().collect(), aggr_exprs: aggr_exprs.into_iter().collect() }
    }
}

/// A single query sort key.
#[derive(Clone)]
pub struct QuerySortExpr {
    pub expr: Expr,
    pub descending: bool,
    pub nulls_first: bool,
}

impl QuerySortExpr {
    pub fn new(expr: impl IntoExpr, descending: bool, nulls_first: bool) -> Self {
        Self { expr: expr.into_expr(), descending, nulls_first }
    }

    pub fn ascending(expr: impl IntoExpr) -> Self {
        Self::new(expr, false, false)
    }

    pub fn descending(expr: impl IntoExpr) -> Self {
        Self::new(expr, true, false)
    }
}

/// Complete sort request for one query sort operator.
pub struct QuerySortSpec {
    pub expressions: Vec<QuerySortExpr>,
    pub stable: bool,
}

impl QuerySortSpec {
    pub fn new<I>(expressions: I) -> Self
    where I: IntoIterator<Item = QuerySortExpr> {
        Self { expressions: expressions.into_iter().collect(), stable: false }
    }

    pub fn with_stable(mut self, stable: bool) -> Self {
        self.stable = stable;
        self
    }
}

impl DataFrame {
    /// Evaluate named query projection expressions and return only those outputs.
    pub fn project_query<I>(&self, exprs: I) -> Result<Self>
    where I: IntoIterator<Item = NamedExpr> {
        let exprs: Vec<NamedExpr> = exprs.into_iter().collect();
        validate_unique_query_names(exprs.iter().map(|expr| expr.name.as_str()), "dataframe::project_query")?;
        let names: Vec<String> = exprs.iter().map(|expr| expr.name.clone()).collect();
        let projected = self.with_columns(exprs.into_iter().map(|expr| (expr.name, expr.expr)))?;
        projected.select(names)
    }

    /// Filter rows using the public query expression contract.
    pub fn filter_query(&self, predicate: Expr) -> Result<Self> {
        self.filter_expr(predicate)
    }

    /// Execute a query aggregate request.
    ///
    /// Group and aggregate inputs are expressions. This method materializes
    /// expression inputs as temporary GPU columns, uses the existing groupby
    /// or reduction kernels, and returns only the stable requested output
    /// names. Temporary columns are not exposed in the result.
    pub fn aggregate_query(&self, spec: QueryAggregateSpec) -> Result<Self> {
        validate_non_empty_aggregates(spec.aggr_exprs.len(), "dataframe::aggregate_query")?;
        validate_unique_query_names(
            spec.group_exprs
                .iter()
                .map(|expr| expr.name.as_str())
                .chain(spec.aggr_exprs.iter().map(|expr| expr.output_name.as_str())),
            "dataframe::aggregate_query",
        )?;

        if spec.group_exprs.is_empty() {
            return self.aggregate_query_global(spec.aggr_exprs);
        }

        self.aggregate_query_grouped(spec)
    }

    /// Sort rows using query sort expressions.
    pub fn sort_query(&self, spec: QuerySortSpec) -> Result<Self> {
        if spec.expressions.is_empty() {
            return Ok(self.clone());
        }

        let original_names: Vec<String> = self.column_names().into_iter().map(str::to_string).collect();
        let mut reserved = reserved_names(self, std::iter::empty::<&str>());
        let mut key_names = Vec::with_capacity(spec.expressions.len());
        let mut projections = Vec::with_capacity(spec.expressions.len());
        let mut descending = Vec::with_capacity(spec.expressions.len());
        let mut nulls_first = Vec::with_capacity(spec.expressions.len());

        for (index, expr) in spec.expressions.into_iter().enumerate() {
            let key_name = unique_temp_name(&mut reserved, &format!("{QUERY_TEMP_PREFIX}sort_{index}"));
            key_names.push(key_name.clone());
            projections.push((key_name, expr.expr));
            descending.push(expr.descending);
            nulls_first.push(expr.nulls_first);
        }

        let with_keys = self.with_columns(projections)?;
        let options =
            SortOptions::new().with_descending(descending).with_nulls_first(nulls_first).with_stable(spec.stable);
        let sorted = with_keys.sort_values(key_names.iter().map(String::as_str), options)?;
        sorted.select(original_names)
    }

    fn aggregate_query_grouped(&self, spec: QueryAggregateSpec) -> Result<Self> {
        let mut reserved = reserved_names(
            self,
            spec.group_exprs
                .iter()
                .map(|expr| expr.name.as_str())
                .chain(spec.aggr_exprs.iter().map(|expr| expr.output_name.as_str())),
        );
        let mut projections = Vec::with_capacity(spec.group_exprs.len() + spec.aggr_exprs.len());
        let mut group_names = Vec::with_capacity(spec.group_exprs.len());
        let mut agg_input_names = Vec::with_capacity(spec.aggr_exprs.len());

        for group in &spec.group_exprs {
            group_names.push(group.name.clone());
            projections.push((group.name.clone(), group.expr.clone()));
        }

        for (index, aggregate) in spec.aggr_exprs.iter().enumerate() {
            let input_name = unique_temp_name(&mut reserved, &format!("{QUERY_TEMP_PREFIX}agg_{index}"));
            agg_input_names.push(input_name.clone());
            projections.push((input_name, aggregate.input.clone()));
        }

        let working = self.with_columns(projections)?;
        let agg_specs: Vec<AggSpec> = spec
            .aggr_exprs
            .iter()
            .zip(agg_input_names)
            .map(|(aggregate, input_name)| {
                AggSpec::new(input_name, aggregate.function.to_groupby_agg()).with_alias(aggregate.output_name.clone())
            })
            .collect();

        working.groupby(group_names.iter().map(String::as_str)).agg(&agg_specs)
    }

    fn aggregate_query_global(&self, aggregates: Vec<QueryAggregateExpr>) -> Result<Self> {
        let mut output = Vec::with_capacity(aggregates.len());
        for aggregate in aggregates {
            let input = aggregate.input.eval(self)?;
            let scalar = aggregate.function.reduce_global(&input)?;
            let column = Column::from_scalar(&scalar, 1)?;
            output.push((aggregate.output_name, Series::try_from(column)?));
        }
        DataFrame::from_columns(output)
    }
}

fn validate_non_empty_aggregates(count: usize, operation: &'static str) -> Result<()> {
    if count == 0 {
        Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "query aggregate requires at least one aggregate expression",
        )
        .with_detail("resource", "query_aggregate")
        .with_detail("reason", "empty_aggregate_expressions")
        .with_detail("remedy", "provide at least one QueryAggregateExpr"))
    } else {
        Ok(())
    }
}

fn validate_unique_query_names<'a>(names: impl IntoIterator<Item = &'a str>, operation: &'static str) -> Result<()> {
    let names: Vec<&str> = names.into_iter().collect();
    let mut seen = HashSet::with_capacity(names.len());
    for name in &names {
        if !seen.insert(*name) {
            return Err(duplicate_column_name_error_with_contract(
                operation,
                *name,
                names.len(),
                "query output names must be unique",
                "duplicate_query_output_name",
                "give each query expression a unique output name",
            ));
        }
    }
    Ok(())
}

fn reserved_names<'a>(df: &DataFrame, additional: impl IntoIterator<Item = &'a str>) -> HashSet<String> {
    let mut reserved: HashSet<String> = df.column_names().into_iter().map(str::to_string).collect();
    reserved.extend(additional.into_iter().map(str::to_string));
    reserved
}

fn unique_temp_name(reserved: &mut HashSet<String>, base: &str) -> String {
    if reserved.insert(base.to_string()) {
        return base.to_string();
    }

    let mut suffix = 1usize;
    loop {
        let candidate = format!("{base}_{suffix}");
        if reserved.insert(candidate.clone()) {
            return candidate;
        }
        suffix += 1;
    }
}

fn count_scalar(input: &Series, include_nulls: bool) -> Result<Scalar> {
    let count = if include_nulls { input.len() } else { input.len() - input.null_count() };
    usize_count_scalar(input, count)
}

fn usize_count_scalar(input: &Series, count: usize) -> Result<Scalar> {
    let count = i32::try_from(count).map_err(|_| {
        Error::new_with(ErrorCode::UnsupportedOperation, "query::count_scalar", "count exceeds libcudf i32 range")
            .with_detail("resource", "query_count")
            .with_detail("reason", "count_exceeds_i32_range")
            .with_detail("count", count.to_string())
    })?;
    Scalar::from_value_in_domain(count, input.as_column().execution_domain())
}
