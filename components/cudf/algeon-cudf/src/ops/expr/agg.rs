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
    /// Use [`SeriesOps::sum`](crate::frame::series::SeriesOps::sum) or
    /// [`SeriesOps::sum_value`](crate::frame::series::SeriesOps::sum_value) when you
    /// want a scalar result. This method returns a broadcast expression operand.
    pub fn sum(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Sum }
    }

    /// Mean of all rows, materialized as a full-length `Float64` column.
    ///
    /// Use [`SeriesOps::mean`](crate::frame::series::SeriesOps::mean) or
    /// [`SeriesOps::mean_value`](crate::frame::series::SeriesOps::mean_value) when
    /// you want a scalar result. This method returns a broadcast expression operand.
    pub fn mean(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Mean }
    }

    /// Minimum of all rows, materialized as a full-length column.
    ///
    /// Use [`SeriesOps::min`](crate::frame::series::SeriesOps::min) or
    /// [`SeriesOps::min_value`](crate::frame::series::SeriesOps::min_value) when you
    /// want a scalar result. This method returns a broadcast expression operand.
    pub fn min(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Min }
    }

    /// Maximum of all rows, materialized as a full-length column.
    ///
    /// Use [`SeriesOps::max`](crate::frame::series::SeriesOps::max) or
    /// [`SeriesOps::max_value`](crate::frame::series::SeriesOps::max_value) when you
    /// want a scalar result. This method returns a broadcast expression operand.
    pub fn max(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Max }
    }

    /// Standard deviation of all rows, materialized as a full-length column.
    ///
    /// Use [`SeriesOps::std`](crate::frame::series::SeriesOps::std) or
    /// [`SeriesOps::std_value`](crate::frame::series::SeriesOps::std_value) when you
    /// want a scalar result. This method returns a broadcast expression operand.
    pub fn std(self, ddof: u32) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Std { ddof } }
    }

    /// Variance of all rows, materialized as a full-length column.
    ///
    /// Use [`SeriesOps::var`](crate::frame::series::SeriesOps::var) or
    /// [`SeriesOps::var_value`](crate::frame::series::SeriesOps::var_value) when you
    /// want a scalar result. This method returns a broadcast expression operand.
    pub fn var(self, ddof: u32) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Variance { ddof } }
    }

    /// Median of all rows, materialized as a full-length `Float64` column.
    ///
    /// Use [`SeriesOps::median`](crate::frame::series::SeriesOps::median) or
    /// [`SeriesOps::median_value`](crate::frame::series::SeriesOps::median_value)
    /// when you want a scalar result. This method returns a broadcast expression operand.
    pub fn median(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Median }
    }

    /// Logical ANY over all rows, materialized as a full-length BOOL8 column.
    ///
    /// Use [`SeriesOps::any`](crate::frame::series::SeriesOps::any) or
    /// [`SeriesOps::any_value`](crate::frame::series::SeriesOps::any_value) when you
    /// want a scalar result. This method returns a broadcast expression operand.
    pub fn any(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Any }
    }

    /// Logical ALL over all rows, materialized as a full-length BOOL8 column.
    ///
    /// Use [`SeriesOps::all`](crate::frame::series::SeriesOps::all) or
    /// [`SeriesOps::all_value`](crate::frame::series::SeriesOps::all_value) when you
    /// want a scalar result. This method returns a broadcast expression operand.
    pub fn all(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::All }
    }

    /// Product of all rows, materialized as a full-length column.
    ///
    /// Use [`SeriesOps::product`](crate::frame::series::SeriesOps::product) or
    /// [`SeriesOps::product_value`](crate::frame::series::SeriesOps::product_value)
    /// when you want a scalar result. This method returns a broadcast expression operand.
    pub fn product(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Product }
    }
}
