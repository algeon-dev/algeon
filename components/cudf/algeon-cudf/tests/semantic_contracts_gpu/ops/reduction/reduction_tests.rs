use cudf::{DataType, Exec, TypeId};

use crate::common::series;

fn test_reduce_decimal_scalar_dtype_preserves_scale() {
    let dt = DataType::fixed_point(TypeId::Decimal128, -2).unwrap();
    let series = series::<i32>("x", &[100, 200, 300]).unwrap().on(Exec::RETAINED).astype(dt).unwrap();

    let result = series.on(Exec::RETAINED).min().unwrap();

    assert_eq!(result.data_type().unwrap(), dt);
}

pub(super) const CASES: &[(&str, fn())] = &[(
    "reduction_tests::test_reduce_decimal_scalar_dtype_preserves_scale",
    test_reduce_decimal_scalar_dtype_preserves_scale,
)];
