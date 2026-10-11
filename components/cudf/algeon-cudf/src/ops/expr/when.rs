use super::*;

/// `when(cond).then(x).otherwise(y)`.
///
/// The result is a row-aligned column. The condition is evaluated for each row.
/// `IfElse` does not inherit an output name, so `.alias()` is required
/// when used inside [`FrameOps::with_columns`](crate::FrameOps::with_columns).
///
/// # Example
///
/// ```no_run
/// use cudf::ops::expr::{col, lit, when};
///
/// let expr = when(col("age").ge(18i32))
///     .then("adult")
///     .otherwise("minor")
///     .alias("category");
/// ```
pub fn when(condition: Expr) -> WhenBuilder {
    WhenBuilder(condition)
}

/// First stage of a `when(...).then(...).otherwise(...)` chain.
///
/// Holds the condition until you provide the `true` branch with [`Self::then`].
pub struct WhenBuilder(Expr);

impl WhenBuilder {
    /// Provide the value to produce when the condition is `true`.
    pub fn then(self, value: impl IntoExpr) -> ThenBuilder {
        ThenBuilder { condition: self.0, true_val: value.into_expr() }
    }
}

/// Second stage of a `when(...).then(...).otherwise(...)` chain.
///
/// Holds the condition and `true` branch until you complete the expression
/// with [`Self::otherwise`].
pub struct ThenBuilder {
    condition: Expr,
    true_val: Expr,
}

impl ThenBuilder {
    /// Provide the value to produce when the condition is `false` (or null),
    /// completing the conditional expression.
    pub fn otherwise(self, value: impl IntoExpr) -> Expr {
        Expr::IfElse {
            condition: Box::new(self.condition),
            true_val: Box::new(self.true_val),
            false_val: Box::new(value.into_expr()),
        }
    }
}
