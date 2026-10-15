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
    /// For a scalar result, use
    /// [`SeriesOps::sum`](crate::frame::series::SeriesOps::sum) or
    /// [`SeriesOps::sum_value`](crate::frame::series::SeriesOps::sum_value).
    pub fn sum(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Sum }
    }

    /// Mean of all rows, materialized as a full-length `Float64` column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::mean`](crate::frame::series::SeriesOps::mean) or
    /// [`SeriesOps::mean_value`](crate::frame::series::SeriesOps::mean_value).
    pub fn mean(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Mean }
    }

    /// Minimum of all rows, materialized as a full-length column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::min`](crate::frame::series::SeriesOps::min) or
    /// [`SeriesOps::min_value`](crate::frame::series::SeriesOps::min_value).
    pub fn min(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Min }
    }

    /// Maximum of all rows, materialized as a full-length column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::max`](crate::frame::series::SeriesOps::max) or
    /// [`SeriesOps::max_value`](crate::frame::series::SeriesOps::max_value).
    pub fn max(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Max }
    }

    /// Standard deviation of all rows, materialized as a full-length column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::std`](crate::frame::series::SeriesOps::std) or
    /// [`SeriesOps::std_value`](crate::frame::series::SeriesOps::std_value).
    pub fn std(self, ddof: u32) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Std { ddof } }
    }

    /// Variance of all rows, materialized as a full-length column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::var`](crate::frame::series::SeriesOps::var) or
    /// [`SeriesOps::var_value`](crate::frame::series::SeriesOps::var_value).
    pub fn var(self, ddof: u32) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Variance { ddof } }
    }

    /// Median of all rows, materialized as a full-length `Float64` column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::median`](crate::frame::series::SeriesOps::median) or
    /// [`SeriesOps::median_value`](crate::frame::series::SeriesOps::median_value).
    pub fn median(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Median }
    }

    /// Logical ANY over all rows, materialized as a full-length BOOL8 column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::any`](crate::frame::series::SeriesOps::any) or
    /// [`SeriesOps::any_value`](crate::frame::series::SeriesOps::any_value).
    pub fn any(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Any }
    }

    /// Logical ALL over all rows, materialized as a full-length BOOL8 column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::all`](crate::frame::series::SeriesOps::all) or
    /// [`SeriesOps::all_value`](crate::frame::series::SeriesOps::all_value).
    pub fn all(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::All }
    }

    /// Product of all rows, materialized as a full-length column.
    ///
    /// For a scalar result, use
    /// [`SeriesOps::product`](crate::frame::series::SeriesOps::product) or
    /// [`SeriesOps::product_value`](crate::frame::series::SeriesOps::product_value).
    pub fn product(self) -> Expr {
        Expr::Reduce { input: Box::new(self.0), agg: ReduceAgg::Product }
    }
}
