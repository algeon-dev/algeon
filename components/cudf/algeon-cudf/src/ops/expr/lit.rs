use super::*;

/// Create a literal expression that broadcasts the value to all rows.
///
/// Accepts every [`CudfType`](crate::CudfType) primitive, `&str`, `String`, and a
/// [`ScalarValue`] for decimal, timestamp, duration, and typed null values.
///
/// Evaluation creates a device scalar and broadcasts it into a newly allocated
/// device column in the resolved execution domain, retaining that provenance
/// on the returned [`Series`]. Evaluation does not intentionally synchronize
/// the host before returning; later host-visible reads may synchronize.
pub fn lit(value: impl Into<ScalarValue>) -> Expr {
    Expr::Literal(value.into())
}

/// Create a typed null literal expression that broadcasts to all rows.
///
/// The dtype is part of the literal contract, so query planners can lower
/// typed SQL/DataFusion nulls without inventing sentinel values. Constructing
/// the expression does not allocate GPU memory. Evaluation creates an invalid
/// device scalar of the requested dtype, then broadcasts it into a newly
/// allocated device column of the input DataFrame row count in the resolved
/// execution domain.
pub fn lit_null(dtype: impl Into<DataType>) -> Expr {
    lit(ScalarValue::Null(dtype.into()))
}
