//! SQL AVG as mergeable partial state.
//!
//! Native execution never averages partial averages. Partial aggregation
//! computes `SUM(arg)` and the non-null `COUNT(arg)`, merging sums both, and
//! the final value divides the merged sum by the merged count. Each rule
//! follows DataFusion 55 `avg`:
//!
//! - DataFusion casts integer and floating arguments to Float64 before
//!   accumulating, so the sum state of a floating output is Float64 and the
//!   division runs in Float64 before the cast to the declared output.
//! - A `Decimal128(p, s)` argument sums as `Decimal128(38, s)`. The final value
//!   scales the sum to the declared output scale and truncates the quotient
//!   toward zero, as DataFusion's `DecimalAverager` does.
//! - A merged count of zero produces null.

use arrow::datatypes::DECIMAL128_MAX_PRECISION;

use super::{AggregateExpr, AggregateFunction, ArrowDataType, Expr, QueryError, QueryErrorCode, Result};
use crate::expr::{BinaryOp, CaseWhen, LiteralValue};

/// Digits DataFusion reserves above a decimal argument's precision for the
/// AVG sum. A wider sum than `Decimal128(38, _)` is a Decimal256, which native
/// execution does not hold, so its additions could overflow where
/// DataFusion's do not.
const DATAFUSION_AVG_SUM_HEADROOM_DIGITS: u8 = 13;

/// Type of the non-null count state.
pub const AVG_COUNT_STATE_TYPE: ArrowDataType = ArrowDataType::Int64;

/// Returns the SUM state type of an AVG over an argument declared as
/// `input_type` with the declared `output_type`.
///
/// Beyond [`avg_sum_state_type`], this rejects a `Decimal128` argument that
/// DataFusion sums as Decimal256. Planning validation and plan rewrites that
/// replace an AVG use it; execution frames do not carry decimal precision.
pub(crate) fn avg_declared_sum_state_type(
    input_type: &ArrowDataType,
    output_type: &ArrowDataType,
) -> Result<ArrowDataType> {
    if let ArrowDataType::Decimal128(precision, _) = input_type
        && precision.saturating_add(DATAFUSION_AVG_SUM_HEADROOM_DIGITS) > DECIMAL128_MAX_PRECISION
    {
        return Err(unsupported_avg_types(input_type, output_type));
    }
    avg_sum_state_type(input_type, output_type)
}

/// Returns the SUM state type of an AVG over `input_type` with the declared
/// `output_type`.
///
/// Fails with `UnsupportedType` for a floating output over a non-numeric
/// argument, a decimal output over a non-decimal argument, and an output scale
/// below the argument scale, where DataFusion either never plans the pair or
/// fails.
pub fn avg_sum_state_type(input_type: &ArrowDataType, output_type: &ArrowDataType) -> Result<ArrowDataType> {
    match (input_type, output_type) {
        (input, ArrowDataType::Float32 | ArrowDataType::Float64) if sums_as_float64(input) => {
            Ok(ArrowDataType::Float64)
        },
        (ArrowDataType::Decimal128(_, input_scale), ArrowDataType::Decimal128(_, output_scale))
            if output_scale >= input_scale =>
        {
            Ok(ArrowDataType::Decimal128(DECIMAL128_MAX_PRECISION, *input_scale))
        },
        _ => Err(unsupported_avg_types(input_type, output_type)),
    }
}

/// Returns the type an AVG argument of `input_type` is cast to before the
/// partial aggregate sums it, or `None` when it sums as it is.
///
/// A cuDF groupby sum accumulates an integer argument in Int64 and a Float32
/// argument in Float32. Both round differently from the Float64 accumulation
/// that [`avg_sum_state_type`] declares, an integer sum once it passes 2^53. A
/// `Decimal128` argument sums as it is.
pub fn avg_argument_cast_type(input_type: &ArrowDataType) -> Option<ArrowDataType> {
    (sums_as_float64(input_type) && *input_type != ArrowDataType::Float64).then_some(ArrowDataType::Float64)
}

fn sums_as_float64(input_type: &ArrowDataType) -> bool {
    input_type.is_integer() || input_type.is_floating()
}

fn unsupported_avg_types(input_type: &ArrowDataType, output_type: &ArrowDataType) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedType,
        "query_engine.aggregate.avg_state_type",
        "avg has no native sum state that reproduces DataFusion for this input and output type",
    )
    .with_fact("input_type", format!("{input_type:?}"))
    .with_fact("output_type", format!("{output_type:?}"))
}

/// Returns the type the final division runs in for a declared AVG output:
/// Float64 for a floating output, `Decimal128(38, t)` for a `Decimal128(_, t)`
/// output, and `None` for any other output type.
pub(crate) fn avg_division_type(output_type: &ArrowDataType) -> Option<ArrowDataType> {
    match output_type {
        ArrowDataType::Float32 | ArrowDataType::Float64 => Some(ArrowDataType::Float64),
        ArrowDataType::Decimal128(_, scale) => Some(ArrowDataType::Decimal128(DECIMAL128_MAX_PRECISION, *scale)),
        _ => None,
    }
}

/// Partial aggregates of one AVG: `SUM(arg)` as `sum_alias` and the non-null
/// `COUNT(arg)` as `count_alias`. A caller with a `FILTER` clause attaches it
/// to both.
pub(crate) fn avg_partial_aggregates(arg: &Expr, sum_alias: &str, count_alias: &str) -> [AggregateExpr; 2] {
    [
        AggregateExpr::new(AggregateFunction::Sum, arg.clone(), sum_alias),
        AggregateExpr::new(AggregateFunction::Count, arg.clone(), count_alias),
    ]
}

/// Merge aggregates of AVG state: each state column is summed under its own
/// name.
pub(crate) fn avg_merge_aggregates(sum: &str, count: &str) -> [AggregateExpr; 2] {
    [
        AggregateExpr::new(AggregateFunction::Sum, Expr::column(sum), sum),
        AggregateExpr::new(AggregateFunction::Sum, Expr::column(count), count),
    ]
}

/// The AVG value of merged `sum` and `count` state at the declared
/// `output_type`.
///
/// A zero count gives null. A null count, which a global merge over no
/// partial rows produces, also gives null because the division propagates it.
pub fn avg_final_expr(sum: Expr, count: Expr, output_type: &ArrowDataType) -> Result<Expr> {
    let division_type = avg_division_type(output_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.aggregate.avg_division_type",
            "avg output must be Float32, Float64, or Decimal128",
        )
        .with_fact("output_type", format!("{output_type:?}"))
    })?;
    let count_is_zero = Expr::binary(count.clone(), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(0)));
    let sum = Expr::cast(sum, division_type.clone());
    // The IR declares a division's output type only for decimals, whose
    // quotient scale it fixes; Float64 operands divide to Float64.
    let quotient = match division_type {
        ArrowDataType::Decimal128(precision, _) => Expr::binary_with_output_type(
            sum,
            BinaryOp::Divide,
            Expr::cast(count, ArrowDataType::Decimal128(precision, 0)),
            division_type,
        ),
        _ => Expr::binary(sum, BinaryOp::Divide, Expr::cast(count, division_type)),
    };
    Ok(Expr::case(
        vec![CaseWhen::new(count_is_zero, Expr::literal(LiteralValue::Null(output_type.clone())))],
        Expr::cast(quotient, output_type.clone()),
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Partial aggregates of the AVG at this index.
    #[doc(hidden)]
    pub fn partial_aggregates(&self, arg: &Expr) -> [AggregateExpr; 2] {
        avg_partial_aggregates(arg, self.sum(), self.count())
    }

    /// Merge aggregates of the AVG state at this index.
    #[doc(hidden)]
    pub fn merge_aggregates(&self) -> [AggregateExpr; 2] {
        avg_merge_aggregates(self.sum(), self.count())
    }
}
