use super::{AggregateExpr, AggregateFunction, AggregateSpec, ArrowDataType, Expr};

#[doc(hidden)]
pub fn aggregate_spec_has_avg(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::Avg)
}

#[doc(hidden)]
pub fn aggregate_spec_has_floating_statistical_aggregate(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.function.is_floating_statistical())
}

#[doc(hidden)]
pub fn aggregate_spec_has_median(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::Median)
}

#[doc(hidden)]
pub fn aggregate_spec_requires_whole_input(spec: &AggregateSpec) -> bool {
    aggregate_spec_has_floating_statistical_aggregate(spec)
        || aggregate_spec_has_distinct(spec)
        || aggregate_spec_has_median(spec)
}

#[doc(hidden)]
pub fn aggregate_spec_has_distinct(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(aggregate_expr_is_exact_nunique)
}

#[doc(hidden)]
pub fn aggregate_expr_is_exact_nunique(expr: &AggregateExpr) -> bool {
    expr.function == AggregateFunction::NUnique || expr.distinct && expr.function == AggregateFunction::Count
}

#[doc(hidden)]
pub fn aggregate_spec_has_filters(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.filter.is_some())
}

/// Accumulator the DataFusion-exact host statistical aggregate kernel runs for
/// one aggregate output.
///
/// Selection follows DataFusion's accumulator for the input and output Arrow
/// types, so aggregates evaluated beside STDDEV/VAR keep DataFusion's integer
/// width, wrapping arithmetic, and NaN ordering instead of passing through
/// `f64`.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFusionExactHostAccumulator {
    /// Non-null input count.
    Count,
    /// Wrapping `i64` sum.
    SignedSum,
    /// Wrapping `u64` sum.
    UnsignedSum,
    /// `f64` sum in DataFusion input order.
    FloatSum,
    /// Signed integer minimum; the output keeps the input width.
    SignedMin,
    /// Signed integer maximum; the output keeps the input width.
    SignedMax,
    /// `u64` minimum.
    UnsignedMin,
    /// `u64` maximum.
    UnsignedMax,
    /// `f64` minimum with DataFusion's NaN ordering.
    FloatMin,
    /// `f64` maximum with DataFusion's NaN ordering.
    FloatMax,
    /// `f64` sum divided by the non-null count.
    Avg,
    /// Welford running moments for STDDEV and VAR.
    Welford,
}

/// Returns the exact host accumulator for `expr`, or `None` when the host
/// kernel cannot reproduce DataFusion's result for these types.
///
/// `input_type` is `None` when the aggregate has no input column, as for
/// `COUNT(*)`.
#[doc(hidden)]
pub fn datafusion_exact_host_accumulator(
    expr: &AggregateExpr,
    input_type: Option<&ArrowDataType>,
    output_type: &ArrowDataType,
) -> Option<DataFusionExactHostAccumulator> {
    use DataFusionExactHostAccumulator as Accumulator;

    if expr.distinct || aggregate_expr_is_exact_nunique(expr) {
        return None;
    }
    if expr.function == AggregateFunction::Count {
        return (*output_type == ArrowDataType::Int64).then_some(Accumulator::Count);
    }
    let input_type = input_type?;
    let numeric_input = matches!(
        input_type,
        ArrowDataType::Int32 | ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64
    );
    let same_type = input_type == output_type;
    match (expr.function, output_type) {
        (AggregateFunction::Sum, ArrowDataType::Int64) if same_type => Some(Accumulator::SignedSum),
        (AggregateFunction::Sum, ArrowDataType::UInt64) if same_type => Some(Accumulator::UnsignedSum),
        (AggregateFunction::Sum, ArrowDataType::Float64) if same_type => Some(Accumulator::FloatSum),
        (AggregateFunction::Min, ArrowDataType::Int32 | ArrowDataType::Int64) if same_type => {
            Some(Accumulator::SignedMin)
        },
        (AggregateFunction::Max, ArrowDataType::Int32 | ArrowDataType::Int64) if same_type => {
            Some(Accumulator::SignedMax)
        },
        (AggregateFunction::Min, ArrowDataType::UInt64) if same_type => Some(Accumulator::UnsignedMin),
        (AggregateFunction::Max, ArrowDataType::UInt64) if same_type => Some(Accumulator::UnsignedMax),
        (AggregateFunction::Min, ArrowDataType::Float64) if same_type => Some(Accumulator::FloatMin),
        (AggregateFunction::Max, ArrowDataType::Float64) if same_type => Some(Accumulator::FloatMax),
        (AggregateFunction::Avg, ArrowDataType::Float64) if numeric_input => Some(Accumulator::Avg),
        (function, ArrowDataType::Float64) if function.is_floating_statistical() && numeric_input => {
            Some(Accumulator::Welford)
        },
        _ => None,
    }
}

/// Returns whether the DataFusion-exact host statistical aggregate kernel can
/// group by a key read as `input_type` and emitted as `output_type`.
///
/// String keys are interned, so any string input type can produce any string
/// output type; every other key keeps its input type.
#[doc(hidden)]
pub fn datafusion_exact_host_group_key_supported(input_type: &ArrowDataType, output_type: &ArrowDataType) -> bool {
    let is_string = |data_type: &ArrowDataType| {
        matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
    };
    if is_string(input_type) {
        return is_string(output_type);
    }
    input_type == output_type
        && matches!(
            input_type,
            ArrowDataType::Boolean
                | ArrowDataType::Int32
                | ArrowDataType::Date32
                | ArrowDataType::Int64
                | ArrowDataType::UInt64
                | ArrowDataType::Float64
        )
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AvgStateAliases {
    sum_alias: String,
    count_alias: String,
}

impl AvgStateAliases {
    /// Stable internal columns for logical AVG partial/final state.
    ///
    /// The public aggregate output keeps the logical AVG alias. These names
    /// exist only while carrying SUM and non-null COUNT state between partial
    /// and final aggregate stages.
    #[doc(hidden)]
    pub fn for_aggregate_index(index: usize) -> Self {
        Self {
            sum_alias: format!("__algeon_native_avg_{index}_sum"),
            count_alias: format!("__algeon_native_avg_{index}_count"),
        }
    }

    #[doc(hidden)]
    pub fn sum(&self) -> &str {
        self.sum_alias.as_str()
    }

    #[doc(hidden)]
    pub fn count(&self) -> &str {
        self.count_alias.as_str()
    }

    #[doc(hidden)]
    pub fn sum_expr(&self) -> Expr {
        Expr::column(self.sum_alias.clone())
    }

    #[doc(hidden)]
    pub fn count_expr(&self) -> Expr {
        Expr::column(self.count_alias.clone())
    }

    #[doc(hidden)]
    pub fn output_names(&self) -> [String; 2] {
        [self.sum_alias.clone(), self.count_alias.clone()]
    }
}
