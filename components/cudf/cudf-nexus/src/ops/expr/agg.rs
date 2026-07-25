use super::*;

/// Aggregation expression methods for row-aligned broadcast aggregate
/// expressions.
///
/// Construct via [`Expr::agg`]. Methods on this type reduce their input to a
/// scalar internally, then broadcast the result back to the DataFrame row
/// count. This keeps the output usable inside larger row-aligned expressions
/// such as `col("x") / col("x").agg().sum()`.
pub struct ExprAggNamespace(pub(super) Expr);

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
