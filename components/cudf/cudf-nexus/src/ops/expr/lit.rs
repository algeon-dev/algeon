use std::sync::Arc;

use super::*;

type NumericLiteralFactory = dyn Fn(&Arc<ExecutionDomain>) -> Result<Scalar> + Send + Sync + 'static;

enum LitValueInner {
    /// A numeric scalar factory. The closure creates the Scalar on demand.
    Numeric { factory: Box<NumericLiteralFactory>, dtype: DataType },
    /// A string literal.
    String(String),
    /// A typed null literal.
    Null(DataType),
}

/// Opaque wrapper for literal values inside expressions.
///
/// Construct via [`lit()`]; not user-constructible.
#[derive(Clone)]
pub struct LitValue(Arc<LitValueInner>);

impl LitValue {
    pub(crate) fn to_scalar_in_domain(&self, domain: &Arc<ExecutionDomain>) -> Result<Scalar> {
        match &*self.0 {
            LitValueInner::Numeric { factory, .. } => factory(domain),
            LitValueInner::String(s) => Scalar::from_str_in_domain(s, domain),
            LitValueInner::Null(dtype) => Scalar::null_in_domain(*dtype, domain),
        }
    }

    fn numeric(
        dtype: DataType,
        factory: impl Fn(&Arc<ExecutionDomain>) -> Result<Scalar> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(LitValueInner::Numeric { factory: Box::new(factory), dtype }))
    }

    fn string(s: String) -> Self {
        Self(Arc::new(LitValueInner::String(s)))
    }

    fn null(dtype: DataType) -> Self {
        Self(Arc::new(LitValueInner::Null(dtype)))
    }

    pub(super) fn fixed_width_metadata(&self) -> Option<(DataType, bool)> {
        match &*self.0 {
            LitValueInner::Numeric { dtype, .. } => Some((*dtype, false)),
            LitValueInner::Null(dtype) => Some((*dtype, true)),
            LitValueInner::String(_) => None,
        }
    }
}

/// Create a literal expression that broadcasts the value to all rows.
///
/// Supports all [`CudfType`] numeric types, `&str`, and `String`.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit<L: IntoLit>(value: L) -> Expr {
    value.into_lit()
}

/// Create a typed null literal expression that broadcasts to all rows.
///
/// The dtype is part of the literal contract, so query planners can lower
/// typed SQL/DataFusion nulls without inventing sentinel values. Constructing
/// the expression is cheap and does not allocate GPU memory. Evaluation creates
/// an invalid device scalar of the requested dtype, then broadcasts it into a
/// newly allocated device column of the input DataFrame row count in the
/// resolved execution domain.
pub fn lit_null(dtype: impl Into<DataType>) -> Expr {
    Expr::Literal(LitValue::null(dtype.into()))
}

fn decimal_literal_dtype(id: TypeId, scale: i32) -> DataType {
    match DataType::fixed_point(id, scale) {
        Ok(dtype) => dtype,
        Err(_) => unreachable!("decimal literal constructors pass a decimal type"),
    }
}

pub fn lit_decimal32_raw(raw: i32, scale: i32) -> Expr {
    let dtype = decimal_literal_dtype(TypeId::Decimal32, scale);
    Expr::Literal(LitValue::numeric(dtype, move |domain| {
        Scalar::decimal_from_raw_in_domain(i128::from(raw), dtype, domain)
    }))
}

/// Create a `DECIMAL64` literal from its raw fixed-point representation.
///
/// `scale` is the cudf fixed-point scale, so a raw value of `12345` with
/// scale `-2` represents `123.45`. Evaluation broadcasts the scalar to the
/// input DataFrame row count.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_decimal64_raw(raw: i64, scale: i32) -> Expr {
    let dtype = decimal_literal_dtype(TypeId::Decimal64, scale);
    Expr::Literal(LitValue::numeric(dtype, move |domain| {
        Scalar::decimal_from_raw_in_domain(i128::from(raw), dtype, domain)
    }))
}

/// Create a `DECIMAL128` literal from its raw fixed-point representation.
///
/// `scale` is the cudf fixed-point scale, so a raw value of `12345` with
/// scale `-2` represents `123.45`. Evaluation broadcasts the scalar to the
/// input DataFrame row count.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_decimal128_raw(raw: i128, scale: i32) -> Expr {
    let dtype = decimal_literal_dtype(TypeId::Decimal128, scale);
    Expr::Literal(LitValue::numeric(dtype, move |domain| Scalar::decimal_from_raw_in_domain(raw, dtype, domain)))
}

/// Create a timestamp literal from its raw epoch-unit representation.
///
/// No parsing, timezone conversion, or calendar interpretation is performed
/// by this expression helper.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_timestamp_raw(raw: i64, resolution: TimestampResolution) -> Expr {
    let dtype = DataType::new(resolution.type_id());
    Expr::Literal(LitValue::numeric(dtype, move |domain| {
        Scalar::timestamp_from_raw_in_domain(raw, resolution.type_id(), domain)
    }))
}

/// Create a duration literal from its raw unit-count representation.
///
/// Evaluation broadcasts the scalar to the input DataFrame row count without
/// string parsing or timezone behavior.
///
/// Constructing the expression is allocation-free. Evaluation creates a
/// device scalar and broadcasts it into a newly allocated device column in the
/// resolved execution domain, retaining that provenance on the returned
/// [`Series`]. Evaluation does not intentionally synchronize the host before
/// returning; later host-visible reads may synchronize.
pub fn lit_duration_raw(raw: i64, resolution: TimestampResolution) -> Expr {
    let dtype = DataType::new(resolution.duration_type_id());
    Expr::Literal(LitValue::numeric(dtype, move |domain| {
        Scalar::duration_from_raw_in_domain(raw, resolution.duration_type_id(), domain)
    }))
}

/// Trait for types that can be converted to a literal [`Expr`].
pub trait IntoLit {
    /// Convert this value into a literal `Expr`.
    fn into_lit(self) -> Expr;
}

// Blanket impl for all CudfType (numeric) types.
impl<T: CudfType + Send + Sync> IntoLit for T {
    fn into_lit(self) -> Expr {
        let val = self;
        Expr::Literal(LitValue::numeric(DataType::new(T::TYPE_ID), move |domain| {
            Scalar::from_value_in_domain(val, domain)
        }))
    }
}

impl IntoLit for &str {
    fn into_lit(self) -> Expr {
        Expr::Literal(LitValue::string(self.to_string()))
    }
}

impl IntoLit for String {
    fn into_lit(self) -> Expr {
        Expr::Literal(LitValue::string(self))
    }
}
