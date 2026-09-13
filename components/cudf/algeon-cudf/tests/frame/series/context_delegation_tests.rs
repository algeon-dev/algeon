//! Semantic lock-down for explicit `Series` execution-context overrides.
//!
//! These tests ensure the default high-level `Series` API stays behaviorally
//! aligned with the explicit `*_in(..., &ExecutionOverrides)` surface across
//! representative operation families while the implementation shares a single
//! context-aware core.

use cudf::{
    ErrorCode, ErrorKind, ExecutionOverrides, Series, SortOptions, StringSliceOptions, TypeId,
    core::error::{FactKeyName, FactValue},
    testing::prelude::*,
};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, reset_exec_observation, run_named_cases, sample_runtime,
};

fn assert_i32_series_eq(lhs: &Series, rhs: &Series) {
    assert_eq!(lhs.dtype().unwrap().id(), rhs.dtype().unwrap().id());
    assert_eq!(lhs.to_optional_vec::<i32>().unwrap(), rhs.to_optional_vec::<i32>().unwrap());
}

fn assert_bool_series_eq(lhs: &Series, rhs: &Series) {
    assert_eq!(lhs.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(rhs.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(lhs.to_optional_vec::<bool>().unwrap(), rhs.to_optional_vec::<bool>().unwrap());
}

fn default_series_transform_ops_match_empty_execution_context() {
    let rt = sample_runtime();
    let values = rt.series(&[3i32, 1, 2, 1]).unwrap();
    let maybe = rt.series_opt(&[Some(3i32), None, Some(2), None]).unwrap();
    let gather_map = rt.series(&[2i32, 0]).unwrap();
    let filter_mask = rt.nullable_series(&[true, false, true, false], &[true, true, true, false]).unwrap();
    let fill_values = rt.series(&[9i32, 8, 7, 6]).unwrap();
    let ctx = ExecutionOverrides::new();

    let expected_stream = values.as_column().testing_execution_domain_stream_value();
    let expected_mr = values.as_column().testing_execution_domain_mr_value();

    reset_exec_observation();
    let sorted = values.sort_values(SortOptions::default()).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_exec_observation();
    let sorted_in_context = values.sort_values_in(SortOptions::default(), &ctx).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_i32_series_eq(&sorted, &sorted_in_context);

    assert_i32_series_eq(&values.head(3).unwrap(), &values.head_in(3, &ctx).unwrap());
    assert_i32_series_eq(&values.gather(&gather_map).unwrap(), &values.gather_in(&gather_map, &ctx).unwrap());
    assert_i32_series_eq(&values.filter(&filter_mask).unwrap(), &values.filter_in(&filter_mask, &ctx).unwrap());
    assert_i32_series_eq(&maybe.dropna().unwrap(), &maybe.dropna_in(&ctx).unwrap());
    assert_i32_series_eq(
        &maybe.fillna_with_series(&fill_values).unwrap(),
        &maybe.fillna_with_series_in(&fill_values, &ctx).unwrap(),
    );

    let mut unique_default = values.unique().unwrap().to_vec_strict::<i32>().unwrap();
    let mut unique_scoped = values.unique_in(&ctx).unwrap().to_vec_strict::<i32>().unwrap();
    unique_default.sort_unstable();
    unique_scoped.sort_unstable();
    assert_eq!(unique_default, unique_scoped);
}

fn default_series_numeric_boolean_and_reduction_ops_match_empty_execution_context() {
    let rt = sample_runtime();
    let ints = rt.series(&[3i32, 1, 2, 1]).unwrap();
    let rhs = rt.series(&[10i32, 20, 30, 40]).unwrap();
    let bools = rt.series(&[true, false, true, true]).unwrap();
    let bools_rhs = rt.series(&[true, true, false, true]).unwrap();
    let floats = rt.nullable_series(&[1.0f64, f64::NAN, 0.0, f64::NAN, 2.0], &[true, true, false, true, true]).unwrap();
    let ctx = ExecutionOverrides::new();

    assert_i32_series_eq(&ints.add(&rhs).unwrap(), &ints.add_in(&rhs, &ctx).unwrap());
    assert_bool_series_eq(&ints.eq_scalar(2i32).unwrap(), &ints.eq_scalar_in(2i32, &ctx).unwrap());
    assert_bool_series_eq(&bools.logical_and(&bools_rhs).unwrap(), &bools.logical_and_in(&bools_rhs, &ctx).unwrap());

    assert_eq!(ints.sum_value::<i64>().unwrap(), ints.sum_value_in::<i64>(&ctx).unwrap());
    assert_eq!(ints.minmax_values::<i32>().unwrap(), ints.minmax_values_in::<i32>(&ctx).unwrap());
    assert_eq!(floats.nunique().unwrap(), floats.nunique_in(&ctx).unwrap());
    assert_i32_series_eq(&ints.cumsum().unwrap(), &ints.cumsum_in(&ctx).unwrap());
    assert_i32_series_eq(&ints.abs().unwrap(), &ints.abs_in(&ctx).unwrap());
    assert_bool_series_eq(&bools.logical_not().unwrap(), &bools.logical_not_in(&ctx).unwrap());
}

fn default_series_string_namespace_and_factorization_ops_match_empty_execution_context() {
    let rt = sample_runtime();
    let strings = rt.series(&["Beta", "alpha", "Beta", "gamma"]).unwrap();
    let ctx = ExecutionOverrides::new();

    let lower = strings.str().lower().unwrap();
    let lower_in_context = strings.str().lower_in(&ctx).unwrap();
    assert_eq!(lower.to_strings_strict().unwrap(), lower_in_context.to_strings_strict().unwrap());

    let contains = strings.str().contains("a").unwrap();
    let contains_in_context = strings.str().contains_in("a", &ctx).unwrap();
    assert_bool_series_eq(&contains, &contains_in_context);

    let like = strings.str().like("%a").unwrap();
    let like_in_context = strings.str().like_in("%a", &ctx).unwrap();
    assert_bool_series_eq(&like, &like_in_context);

    let sliced = strings.str().slice(StringSliceOptions::new(Some(1), Some(3))).unwrap();
    let sliced_in_context = strings.str().slice_in(StringSliceOptions::new(Some(1), Some(3)), &ctx).unwrap();
    assert_eq!(sliced.to_strings_strict().unwrap(), sliced_in_context.to_strings_strict().unwrap());

    let codebook = strings.codebook().unwrap();
    let codebook_in_context = strings.codebook_in(&ctx).unwrap();
    assert_eq!(codebook.keys().to_strings_strict().unwrap(), codebook_in_context.keys().to_strings_strict().unwrap());
    assert_eq!(
        codebook.codes().as_series().to_vec_strict::<i32>().unwrap(),
        codebook_in_context.codes().as_series().to_vec_strict::<i32>().unwrap()
    );

    let factorized = strings.factorize().unwrap();
    let factorized_in_context = strings.factorize_in(&ctx).unwrap();
    assert_eq!(
        factorized.row_codes().as_series().to_optional_vec::<i32>().unwrap(),
        factorized_in_context.row_codes().as_series().to_optional_vec::<i32>().unwrap()
    );
    assert_eq!(
        factorized.codebook().keys().to_strings_strict().unwrap(),
        factorized_in_context.codebook().keys().to_strings_strict().unwrap()
    );
}

fn default_and_explicit_context_numeric_cast_validation_errors_match() {
    let rt = sample_runtime();
    let ints = rt.series(&[3i32, 1, 2, 1]).unwrap();
    let ctx = ExecutionOverrides::new();

    let default_err = ints.to_numeric(TypeId::String).unwrap_err();
    let scoped_err = ints.to_numeric_in(TypeId::String, &ctx).unwrap_err();

    assert_eq!(default_err.kind(), ErrorKind::Unsupported);
    assert_eq!(default_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(default_err.operation(), "series::to_numeric");
    assert_eq!(default_err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));

    assert_eq!(scoped_err.kind(), ErrorKind::Unsupported);
    assert_eq!(scoped_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(scoped_err.operation(), "series::to_numeric_in");
    assert_eq!(scoped_err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
}

#[test]
fn context_delegation_tests_cases() {
    run_named_cases(&[
        (
            "default_series_transform_ops_match_empty_execution_context",
            default_series_transform_ops_match_empty_execution_context,
        ),
        (
            "default_series_numeric_boolean_and_reduction_ops_match_empty_execution_context",
            default_series_numeric_boolean_and_reduction_ops_match_empty_execution_context,
        ),
        (
            "default_series_string_namespace_and_factorization_ops_match_empty_execution_context",
            default_series_string_namespace_and_factorization_ops_match_empty_execution_context,
        ),
        (
            "default_and_explicit_context_numeric_cast_validation_errors_match",
            default_and_explicit_context_numeric_cast_validation_errors_match,
        ),
    ]);
}
