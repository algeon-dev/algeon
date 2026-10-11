//! Public parameter validation at the CPU boundary.

use cudf::testing::run_named_cases;
use cugraph::{
    error::ErrorKind,
    params::{
        BfsDepthLimit, ClosedUnitF64, ClusterCount, FractionF64, HalfOpenUnitF64, IterationCount, NonNegativeF64,
        PositiveF64, UnitF64,
    },
};

#[test]
fn parameter_validation_cases_1() {
    run_named_cases(&[
        ("params::tests::half_open_unit_f64_accepts_zero_but_not_one", half_open_unit_f64_accepts_zero_but_not_one),
        ("params::tests::iteration_count_try_into_i32_rejects_overflow", iteration_count_try_into_i32_rejects_overflow),
        ("params::tests::non_negative_f64_accepts_zero", non_negative_f64_accepts_zero),
        ("params::tests::closed_unit_f64_accepts_boundaries", closed_unit_f64_accepts_boundaries),
        ("params::tests::iteration_count_try_from_usize", iteration_count_try_from_usize),
        (
            "params::tests::bfs_depth_limit_try_from_i32_rejects_negative_values",
            bfs_depth_limit_try_from_i32_rejects_negative_values,
        ),
        ("params::tests::positive_f64_accepts_positive_finite_values", positive_f64_accepts_positive_finite_values),
        ("params::tests::cluster_count_requires_at_least_two_clusters", cluster_count_requires_at_least_two_clusters),
        (
            "fraction_f64_rejects_out_of_range_and_non_finite_values",
            fraction_f64_rejects_out_of_range_and_non_finite_values,
        ),
        ("params::tests::iteration_count_accepts_positive_values", iteration_count_accepts_positive_values),
        ("params::tests::unit_f64_accepts_values_in_open_interval", unit_f64_accepts_values_in_open_interval),
        ("params::tests::unit_f64_rejects_negative", unit_f64_rejects_negative),
        ("params::tests::closed_unit_f64_rejects_nan_and_infinity", closed_unit_f64_rejects_nan_and_infinity),
        ("params::tests::closed_unit_f64_rejects_out_of_range", closed_unit_f64_rejects_out_of_range),
        (
            "half_open_unit_f64_rejects_out_of_range_and_non_finite_values",
            half_open_unit_f64_rejects_out_of_range_and_non_finite_values,
        ),
        ("params::tests::iteration_count_rejects_zero", iteration_count_rejects_zero),
        ("params::tests::non_negative_f64_accepts_positive", non_negative_f64_accepts_positive),
        ("params::tests::positive_f64_rejects_infinity", positive_f64_rejects_infinity),
        ("params::tests::unit_f64_rejects_greater_than_one", unit_f64_rejects_greater_than_one),
        ("params::tests::unit_f64_rejects_one", unit_f64_rejects_one),
        ("params::tests::unit_f64_rejects_zero", unit_f64_rejects_zero),
        ("params::tests::iteration_count_try_from_i32", iteration_count_try_from_i32),
        ("params::tests::non_negative_f64_rejects_negative", non_negative_f64_rejects_negative),
        ("params::tests::positive_f64_rejects_nan", positive_f64_rejects_nan),
        ("params::tests::positive_f64_rejects_negative", positive_f64_rejects_negative),
        ("params::tests::non_negative_f64_rejects_nan_and_infinity", non_negative_f64_rejects_nan_and_infinity),
        (
            "bfs_depth_limit_accepts_zero_as_the_sources_only_limit",
            bfs_depth_limit_accepts_zero_as_the_sources_only_limit,
        ),
        (
            "params::tests::fraction_f64_accepts_the_upper_bound_but_not_zero",
            fraction_f64_accepts_the_upper_bound_but_not_zero,
        ),
        ("params::tests::positive_f64_rejects_zero", positive_f64_rejects_zero),
        ("params::tests::unit_f64_rejects_nan_and_infinity", unit_f64_rejects_nan_and_infinity),
    ]);
}

fn half_open_unit_f64_accepts_zero_but_not_one() {
    assert_eq!(HalfOpenUnitF64::new(0.0).unwrap().get(), 0.0);
    assert_eq!(HalfOpenUnitF64::new(0.999).unwrap().get(), 0.999);
    assert!(HalfOpenUnitF64::new(1.0).is_err());
}

fn iteration_count_try_into_i32_rejects_overflow() {
    let count = IterationCount::new(u32::MAX).unwrap();
    let err = i32::try_from(count).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

fn non_negative_f64_accepts_zero() {
    assert_eq!(NonNegativeF64::new(0.0).unwrap().get(), 0.0);
}

fn closed_unit_f64_accepts_boundaries() {
    assert_eq!(ClosedUnitF64::new(0.0).unwrap().get(), 0.0);
    assert_eq!(ClosedUnitF64::new(1.0).unwrap().get(), 1.0);
    assert_eq!(ClosedUnitF64::new(0.5).unwrap().get(), 0.5);
}

fn iteration_count_try_from_usize() {
    let count: IterationCount = 1000usize.try_into().unwrap();
    assert_eq!(count.get(), 1000);

    assert!(IterationCount::try_from(0usize).is_err());

    // Value exceeding u32::MAX
    let huge = u32::MAX as usize + 1;
    let err = IterationCount::try_from(huge).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("u32::MAX"));
}

fn bfs_depth_limit_try_from_i32_rejects_negative_values() {
    let limit: BfsDepthLimit = 7i32.try_into().unwrap();
    assert_eq!(limit.get(), 7);
    assert_eq!(BfsDepthLimit::try_from(0i32).unwrap().get(), 0);

    let err = BfsDepthLimit::try_from(-1i32).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

fn positive_f64_accepts_positive_finite_values() {
    assert_eq!(PositiveF64::new(0.001).unwrap().get(), 0.001);
    assert_eq!(PositiveF64::new(1.0).unwrap().get(), 1.0);
    assert_eq!(PositiveF64::new(1e10).unwrap().get(), 1e10);
    assert_eq!(PositiveF64::new(f64::MIN_POSITIVE).unwrap().get(), f64::MIN_POSITIVE);
}

fn cluster_count_requires_at_least_two_clusters() {
    assert_eq!(ClusterCount::new(2).unwrap().get(), 2);
    assert!(ClusterCount::new(1).is_err());
    assert_eq!(ClusterCount::new(0).unwrap_err().kind(), ErrorKind::InvalidInput);
}

fn fraction_f64_rejects_out_of_range_and_non_finite_values() {
    assert!(FractionF64::new(-0.1).is_err());
    assert!(FractionF64::new(1.0 + f64::EPSILON).is_err());
    assert!(FractionF64::new(f64::NAN).is_err());
    assert!(FractionF64::new(f64::INFINITY).is_err());
    assert_eq!(FractionF64::new(2.0).unwrap_err().kind(), ErrorKind::InvalidInput);
}

fn iteration_count_accepts_positive_values() {
    assert_eq!(IterationCount::new(1).unwrap().get(), 1);
    assert_eq!(IterationCount::new(100).unwrap().get(), 100);
    assert_eq!(IterationCount::new(u32::MAX).unwrap().get(), u32::MAX);
}

fn unit_f64_accepts_values_in_open_interval() {
    assert_eq!(UnitF64::new(0.5).unwrap().get(), 0.5);
    assert_eq!(UnitF64::new(0.001).unwrap().get(), 0.001);
    assert_eq!(UnitF64::new(0.999).unwrap().get(), 0.999);
    assert_eq!(UnitF64::new(0.85).unwrap().get(), 0.85);
}

fn unit_f64_rejects_negative() {
    assert!(UnitF64::new(-0.1).is_err());
}

fn closed_unit_f64_rejects_nan_and_infinity() {
    assert!(ClosedUnitF64::new(f64::NAN).is_err());
    assert!(ClosedUnitF64::new(f64::INFINITY).is_err());
}

fn closed_unit_f64_rejects_out_of_range() {
    assert!(ClosedUnitF64::new(-0.001).is_err());
    assert!(ClosedUnitF64::new(1.001).is_err());
}

fn half_open_unit_f64_rejects_out_of_range_and_non_finite_values() {
    assert!(HalfOpenUnitF64::new(-f64::MIN_POSITIVE).is_err());
    assert!(HalfOpenUnitF64::new(f64::NAN).is_err());
    assert!(HalfOpenUnitF64::new(f64::INFINITY).is_err());
    assert_eq!(HalfOpenUnitF64::new(2.0).unwrap_err().kind(), ErrorKind::InvalidInput);
}

fn iteration_count_rejects_zero() {
    let err = IterationCount::new(0).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("greater than zero"));
}

fn non_negative_f64_accepts_positive() {
    assert_eq!(NonNegativeF64::new(1.5).unwrap().get(), 1.5);
    assert_eq!(NonNegativeF64::new(100.0).unwrap().get(), 100.0);
}

fn positive_f64_rejects_infinity() {
    assert!(PositiveF64::new(f64::INFINITY).is_err());
    assert!(PositiveF64::new(f64::NEG_INFINITY).is_err());
}

fn unit_f64_rejects_greater_than_one() {
    assert!(UnitF64::new(1.01).is_err());
}

fn unit_f64_rejects_one() {
    let err = UnitF64::new(1.0).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

fn unit_f64_rejects_zero() {
    let err = UnitF64::new(0.0).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("(0.0, 1.0)"));
}

fn iteration_count_try_from_i32() {
    let count: IterationCount = 500i32.try_into().unwrap();
    assert_eq!(count.get(), 500);

    let err = IterationCount::try_from(0i32).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);

    let err = IterationCount::try_from(-1i32).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

fn non_negative_f64_rejects_negative() {
    let err = NonNegativeF64::new(-0.001).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("non-negative"));
}

fn positive_f64_rejects_nan() {
    assert!(PositiveF64::new(f64::NAN).is_err());
}

fn positive_f64_rejects_negative() {
    let err = PositiveF64::new(-1.0).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

fn non_negative_f64_rejects_nan_and_infinity() {
    assert!(NonNegativeF64::new(f64::NAN).is_err());
    assert!(NonNegativeF64::new(f64::INFINITY).is_err());
    assert!(NonNegativeF64::new(f64::NEG_INFINITY).is_err());
}

fn bfs_depth_limit_accepts_zero_as_the_sources_only_limit() {
    assert_eq!(BfsDepthLimit::new(0).get(), 0);
    assert_eq!(BfsDepthLimit::new(100).get(), 100);
}

fn fraction_f64_accepts_the_upper_bound_but_not_zero() {
    assert_eq!(FractionF64::new(1.0).unwrap().get(), 1.0);
    assert_eq!(FractionF64::new(f64::MIN_POSITIVE).unwrap().get(), f64::MIN_POSITIVE);
    assert!(FractionF64::new(0.0).is_err());
}

fn positive_f64_rejects_zero() {
    let err = PositiveF64::new(0.0).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("greater than zero"));
}

fn unit_f64_rejects_nan_and_infinity() {
    assert!(UnitF64::new(f64::NAN).is_err());
    assert!(UnitF64::new(f64::INFINITY).is_err());
    assert!(UnitF64::new(f64::NEG_INFINITY).is_err());
}
