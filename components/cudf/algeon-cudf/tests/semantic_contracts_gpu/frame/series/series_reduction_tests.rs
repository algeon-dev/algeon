use cudf::{
    ErrorCode, ErrorKind, Exec, ExecutionBudgetEnforcement, ExecutionBudgetRequest, ReduceAgg, TypeId,
    core::error::{FactKeyName, FactValue},
    testing::prelude::*,
};

use crate::common::{
    error_context_value, execution_budget_token, last_exec_mr_value, nullable_series, reset_exec_observation, series,
    with_test_runtime,
};

fn assert_no_budget_error_details(err: &cudf::Error) {
    assert!(err.budget_exhaustion().is_none(), "an unbudgeted operation carries no budget scope");
}

fn expect_cudf_err<T>(result: cudf::Result<T>, message: &str) -> cudf::Error {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}

fn test_sum_i32() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let total = s.on(Exec::RETAINED).sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(total.value::<i64>().unwrap(), 60);
}

fn test_sum_f64() {
    let s = series::<f64>("x", &[1.5, 2.5, 3.0]).unwrap();
    let total = s.on(Exec::RETAINED).sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(total.value::<f64>().unwrap(), 7.0);
}

fn test_sum_bool8() {
    let s = series::<bool>("x", &[true, false, true, true]).unwrap();
    let total = s.on(Exec::RETAINED).sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(total.value::<i64>().unwrap(), 3);
}

fn test_sum_with_nulls() {
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let total = s.on(Exec::RETAINED).sum().unwrap();

    assert_eq!(total.value::<i64>().unwrap(), 40);
}

fn test_sum_string_reports_series_operation_on_type_error() {
    let s = series("x", &["a", "b"]).unwrap();
    let err = s.on(Exec::RETAINED).sum().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::sum");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_eq!(error_context_value(&err, "input_type").as_deref(), None);
}

fn test_min_i32() {
    let s = series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.on(Exec::RETAINED).min().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(m.value::<i32>().unwrap(), 10);
}

fn test_max_i32() {
    let s = series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.on(Exec::RETAINED).max().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(m.value::<i32>().unwrap(), 30);
}

fn test_min_f64() {
    let s = series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let m = s.on(Exec::RETAINED).min().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 1.0);
}

fn test_max_f64() {
    let s = series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let m = s.on(Exec::RETAINED).max().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 3.0);
}

fn test_prepared_scalar_max_reports_shape_and_returns_scalar() {
    let s = series::<i32>("x", &[30, 10, 40, 20]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Max).unwrap();

    assert_eq!(prepared.input().len(), s.len());
    assert!(matches!(prepared.aggregation(), ReduceAgg::Max));
    assert_eq!(prepared.output_type().id(), TypeId::Int32);
    assert_eq!(prepared.output_shape().output_type().id(), TypeId::Int32);
    assert_eq!(prepared.output_shape().output_rows(), 1);

    let output = prepared.execute(Exec::RETAINED).unwrap();
    assert_eq!(output.output_rows(), 1);
    assert_eq!(output.scalar().data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(output.scalar().value::<i32>().unwrap(), 40);
    assert_eq!(output.into_scalar().value::<i32>().unwrap(), 40);
}

fn test_prepared_scalar_sum_i32_reports_int64_output_and_returns_scalar() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Sum).unwrap();

    assert_eq!(prepared.output_type().id(), TypeId::Int64);
    assert_eq!(prepared.output_shape().output_type().id(), TypeId::Int64);
    assert_eq!(prepared.output_shape().output_rows(), 1);

    let output = prepared.execute(Exec::RETAINED).unwrap();
    assert_eq!(output.output_rows(), 1);
    assert_eq!(output.scalar().data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(output.scalar().value::<i64>().unwrap(), 60);
}

fn test_prepared_scalar_max_execute_with_budget_reports_contract_and_routes_allocator() {
    let s = series::<i32>("x", &[30, 10, 40, 20]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Max).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("q15_scalar_max"))
            .unwrap()
    });

    reset_exec_observation();
    let outcome = prepared.execute_with_budget(&token).unwrap();
    let budgeted_exec_mr = last_exec_mr_value();
    let output = outcome.value();
    let report = outcome.report();

    assert_eq!(output.scalar().value::<i32>().unwrap(), 40);
    assert_eq!(report.request().boundary(), "q15_scalar_max");
    assert_eq!(report.request().budget_bytes(), 64 * 1024 * 1024);
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(report.output_rows(), Some(1));
    assert_eq!(report.retained_bytes(), None);
    assert_eq!(report.output_bytes(), None);
    assert_eq!(
        budgeted_exec_mr,
        token.memory_resource().testing_raw_handle_value(),
        "budgeted scalar reduction should route execution through the device manager resource",
    );
}

fn test_prepared_scalar_q17_sum_extendedprice_with_budget_reports_current_diagnostics() {
    let s = series::<f64>("l_extendedprice", &[1000.0, 1250.5, 749.5]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Sum).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("q17_global_sum"))
            .unwrap()
    });

    let outcome = prepared.execute_with_budget(&token).unwrap();
    let output = outcome.value();
    let report = outcome.report();

    assert_eq!(output.scalar().value::<f64>().unwrap(), 3000.0);
    assert_eq!(report.request().boundary(), "q17_global_sum");
    assert_eq!(report.output_rows(), Some(1));
    assert_eq!(report.retained_bytes(), None);
    assert_eq!(report.output_bytes(), None);
}

fn test_prepared_scalar_max_one_byte_budget_preserves_budget_details() {
    let s = series::<i32>("x", &[30, 10, 40, 20]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Max).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(1).with_boundary("q15_scalar_max")).unwrap()
    });

    let err = expect_cudf_err(
        prepared.execute_with_budget(&token),
        "one-byte budget should not cover scalar reduction allocations",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "series::prepared_scalar_reduction::execute_with_budget");
    assert_eq!(err.budget_exhaustion().expect("budget scope").boundary(), "q15_scalar_max");
    assert_eq!(
        err.budget_exhaustion().and_then(|budget| budget.usage()).expect("member usage").allocation_limit_bytes,
        1
    );
}

fn test_prepared_scalar_max_all_null_reports_one_row_and_typed_null_scalar() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Max).unwrap();
    let output = prepared.execute(Exec::RETAINED).unwrap();
    let scalar = output.scalar();

    assert_eq!(prepared.output_shape().output_rows(), 1);
    assert_eq!(output.output_rows(), 1);
    assert_eq!(scalar.data_type().unwrap().id(), TypeId::Int32);
    assert!(!scalar.is_valid().unwrap());
}

fn test_prepared_scalar_reduction_unsupported_dtype_has_no_budget_details() {
    let s = series("x", &["a", "b"]).unwrap();
    let err = expect_cudf_err(s.prepare_scalar_reduction(ReduceAgg::Sum), "expected unsupported dtype error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::prepare_scalar_reduction");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_no_budget_error_details(&err);
}

fn test_prepared_scalar_reduction_budget_validation_error_has_no_budget_details() {
    let s = series::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Std { ddof: u32::MAX }).unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("std_validation"))
            .unwrap()
    });

    let err = expect_cudf_err(prepared.execute_with_budget(&token), "expected ddof range validation error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::prepared_scalar_reduction::execute_with_budget");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("ddof_exceeds_i32_range"));
    assert_no_budget_error_details(&err);
}

fn test_mean_i32() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let m = s.on(Exec::RETAINED).mean().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(m.value::<f64>().unwrap(), 20.0);
}

fn test_mean_f64() {
    let s = series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();
    let m = s.on(Exec::RETAINED).mean().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 2.0);
}

fn test_median_i32() {
    let s = series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.on(Exec::RETAINED).median().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(m.value::<f64>().unwrap(), 20.0);
}

fn test_std_f64() {
    let s = series::<f64>("x", &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let ddof: u32 = 0;
    let std_val = s.on(Exec::RETAINED).std(ddof).unwrap();

    assert_eq!(std_val.data_type().unwrap().id(), TypeId::Float64);
    let v = std_val.value::<f64>().unwrap();
    assert!((v - 2.0).abs() < 1e-10, "expected std ≈ 2.0, got {v}");
}

fn test_var_f64() {
    let s = series::<f64>("x", &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let ddof: u32 = 0;
    let var_val = s.on(Exec::RETAINED).var(ddof).unwrap();

    assert_eq!(var_val.data_type().unwrap().id(), TypeId::Float64);
    let v = var_val.value::<f64>().unwrap();
    assert!((v - 4.0).abs() < 1e-10, "expected var ≈ 4.0, got {v}");
}

fn test_std_ddof_above_i32_max_uses_series_operation_contract() {
    let s = series::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();

    let err = s.on(Exec::RETAINED).std(u32::MAX).expect_err("expected ddof range validation error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::std");
    assert_eq!(error_context_value(&err, "ddof").as_deref(), Some(u32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("ddof_exceeds_i32_range"));
    assert!(
        err.call_chain().contains(&"reduction::create_reduce_agg_handle(std)"),
        "call_chain did not contain reduction::create_reduce_agg_handle(std): {:?}",
        err.call_chain()
    );
}

fn test_var_ddof_above_i32_max_uses_series_operation_contract() {
    let s = series::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();

    let err = s.on(Exec::RETAINED).var(u32::MAX).expect_err("expected ddof range validation error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::var");
    assert_eq!(error_context_value(&err, "ddof").as_deref(), Some(u32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("ddof_exceeds_i32_range"));
    assert!(
        err.call_chain().contains(&"reduction::create_reduce_agg_handle(variance)"),
        "call_chain did not contain reduction::create_reduce_agg_handle(variance): {:?}",
        err.call_chain()
    );
}

fn test_any_true() {
    let s = series::<bool>("x", &[false, true, false]).unwrap();
    let result = s.on(Exec::RETAINED).any().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Bool8);
    assert!(result.value::<bool>().unwrap());
}

fn test_any_false() {
    let s = series::<bool>("x", &[false, false, false]).unwrap();
    let result = s.on(Exec::RETAINED).any().unwrap();

    assert!(!result.value::<bool>().unwrap());
}

fn test_all_true() {
    let s = series::<bool>("x", &[true, true, true]).unwrap();
    let result = s.on(Exec::RETAINED).all().unwrap();

    assert!(result.value::<bool>().unwrap());
}

fn test_all_false() {
    let s = series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).all().unwrap();

    assert!(!result.value::<bool>().unwrap());
}

fn test_product_i32() {
    let s = series::<i32>("x", &[2, 3, 4]).unwrap();
    let result = s.on(Exec::RETAINED).product().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(result.value::<i32>().unwrap(), 24);
}

fn test_product_f64() {
    let s = series::<f64>("x", &[2.0, 3.0, 4.0]).unwrap();
    let result = s.on(Exec::RETAINED).product().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(result.value::<f64>().unwrap(), 24.0);
}

fn test_minmax_i32() {
    let s = series::<i32>("x", &[30, 10, 20]).unwrap();
    let (min, max) = s.on(Exec::RETAINED).minmax().unwrap();

    assert_eq!(min.value::<i32>().unwrap(), 10);
    assert_eq!(max.value::<i32>().unwrap(), 30);
}

fn test_minmax_f64() {
    let s = series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let (min, max) = s.on(Exec::RETAINED).minmax().unwrap();

    assert_eq!(min.value::<f64>().unwrap(), 1.0);
    assert_eq!(max.value::<f64>().unwrap(), 3.0);
}

fn test_cumsum_i32() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let cs = s.on(Exec::RETAINED).cumsum().unwrap();
    assert_eq!(cs.to_vec_strict::<i32>().unwrap(), vec![10, 30, 60]);
}

fn test_cumsum_f64() {
    let s = series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();
    let cs = s.on(Exec::RETAINED).cumsum().unwrap();

    assert_eq!(cs.to_vec_strict::<f64>().unwrap(), vec![1.0, 3.0, 6.0]);
}

fn test_cummin_i32() {
    let s = series::<i32>("x", &[3, 1, 2]).unwrap();
    let cm = s.on(Exec::RETAINED).cummin().unwrap();
    assert_eq!(cm.to_vec_strict::<i32>().unwrap(), vec![3, 1, 1]);
}

fn test_cummax_i32() {
    let s = series::<i32>("x", &[1, 3, 2]).unwrap();
    let cm = s.on(Exec::RETAINED).cummax().unwrap();
    assert_eq!(cm.to_vec_strict::<i32>().unwrap(), vec![1, 3, 3]);
}

fn test_cumprod_i32() {
    let s = series::<i32>("x", &[2, 3, 4]).unwrap();
    let cp = s.on(Exec::RETAINED).cumprod().unwrap();
    assert_eq!(cp.to_vec_strict::<i32>().unwrap(), vec![2, 6, 24]);
}

fn test_is_nan_f64() {
    let s = series::<f64>("x", &[1.0, f64::NAN, 3.0]).unwrap();
    let mask = s.on(Exec::RETAINED).is_nan().unwrap();
    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn test_is_not_nan_f64() {
    let s = series::<f64>("x", &[1.0, f64::NAN, 3.0]).unwrap();
    let mask = s.on(Exec::RETAINED).is_not_nan().unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn test_abs_i32() {
    let s = series::<i32>("x", &[-3, -1, 0, 2]).unwrap();
    let result = s.on(Exec::RETAINED).abs().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![3, 1, 0, 2]);
}

fn test_abs_f64() {
    let s = series::<f64>("x", &[-3.0, -1.5, 0.0, 2.5]).unwrap();
    let result = s.on(Exec::RETAINED).abs().unwrap();

    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![3.0, 1.5, 0.0, 2.5]);
}

fn test_neg_i32() {
    let s = series::<i32>("x", &[1, -2, 3]).unwrap();
    let result = s.on(Exec::RETAINED).neg().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![-1, 2, -3]);
}

fn test_neg_f64() {
    let s = series::<f64>("x", &[1.5, -2.5]).unwrap();
    let result = s.on(Exec::RETAINED).neg().unwrap();

    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![-1.5, 2.5]);
}

fn test_logical_not_bool() {
    let s = series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).logical_not().unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("series_reduction_tests::test_sum_i32", test_sum_i32),
    ("series_reduction_tests::test_sum_f64", test_sum_f64),
    ("series_reduction_tests::test_sum_bool8", test_sum_bool8),
    ("series_reduction_tests::test_sum_with_nulls", test_sum_with_nulls),
    (
        "series_reduction_tests::test_sum_string_reports_series_operation_on_type_error",
        test_sum_string_reports_series_operation_on_type_error,
    ),
    ("series_reduction_tests::test_min_i32", test_min_i32),
    ("series_reduction_tests::test_max_i32", test_max_i32),
    ("series_reduction_tests::test_min_f64", test_min_f64),
    ("series_reduction_tests::test_max_f64", test_max_f64),
    (
        "series_reduction_tests::test_prepared_scalar_max_reports_shape_and_returns_scalar",
        test_prepared_scalar_max_reports_shape_and_returns_scalar,
    ),
    (
        "series_reduction_tests::test_prepared_scalar_sum_i32_reports_int64_output_and_returns_scalar",
        test_prepared_scalar_sum_i32_reports_int64_output_and_returns_scalar,
    ),
    (
        "series_reduction_tests::test_prepared_scalar_max_execute_with_budget_reports_contract_and_routes_allocator",
        test_prepared_scalar_max_execute_with_budget_reports_contract_and_routes_allocator,
    ),
    (
        "series_reduction_tests::test_prepared_scalar_q17_sum_extendedprice_with_budget_reports_current_diagnostics",
        test_prepared_scalar_q17_sum_extendedprice_with_budget_reports_current_diagnostics,
    ),
    (
        "series_reduction_tests::test_prepared_scalar_max_one_byte_budget_preserves_budget_details",
        test_prepared_scalar_max_one_byte_budget_preserves_budget_details,
    ),
    (
        "series_reduction_tests::test_prepared_scalar_max_all_null_reports_one_row_and_typed_null_scalar",
        test_prepared_scalar_max_all_null_reports_one_row_and_typed_null_scalar,
    ),
    (
        "series_reduction_tests::test_prepared_scalar_reduction_unsupported_dtype_has_no_budget_details",
        test_prepared_scalar_reduction_unsupported_dtype_has_no_budget_details,
    ),
    (
        "series_reduction_tests::test_prepared_scalar_reduction_budget_validation_error_has_no_budget_details",
        test_prepared_scalar_reduction_budget_validation_error_has_no_budget_details,
    ),
    ("series_reduction_tests::test_mean_i32", test_mean_i32),
    ("series_reduction_tests::test_mean_f64", test_mean_f64),
    ("series_reduction_tests::test_median_i32", test_median_i32),
    ("series_reduction_tests::test_std_f64", test_std_f64),
    ("series_reduction_tests::test_var_f64", test_var_f64),
    (
        "series_reduction_tests::test_std_ddof_above_i32_max_uses_series_operation_contract",
        test_std_ddof_above_i32_max_uses_series_operation_contract,
    ),
    (
        "series_reduction_tests::test_var_ddof_above_i32_max_uses_series_operation_contract",
        test_var_ddof_above_i32_max_uses_series_operation_contract,
    ),
    ("series_reduction_tests::test_any_true", test_any_true),
    ("series_reduction_tests::test_any_false", test_any_false),
    ("series_reduction_tests::test_all_true", test_all_true),
    ("series_reduction_tests::test_all_false", test_all_false),
    ("series_reduction_tests::test_product_i32", test_product_i32),
    ("series_reduction_tests::test_product_f64", test_product_f64),
    ("series_reduction_tests::test_minmax_i32", test_minmax_i32),
    ("series_reduction_tests::test_minmax_f64", test_minmax_f64),
    ("series_reduction_tests::test_cumsum_i32", test_cumsum_i32),
    ("series_reduction_tests::test_cumsum_f64", test_cumsum_f64),
    ("series_reduction_tests::test_cummin_i32", test_cummin_i32),
    ("series_reduction_tests::test_cummax_i32", test_cummax_i32),
    ("series_reduction_tests::test_cumprod_i32", test_cumprod_i32),
    ("series_reduction_tests::test_is_nan_f64", test_is_nan_f64),
    ("series_reduction_tests::test_is_not_nan_f64", test_is_not_nan_f64),
    ("series_reduction_tests::test_abs_i32", test_abs_i32),
    ("series_reduction_tests::test_abs_f64", test_abs_f64),
    ("series_reduction_tests::test_neg_i32", test_neg_i32),
    ("series_reduction_tests::test_neg_f64", test_neg_f64),
    ("series_reduction_tests::test_logical_not_bool", test_logical_not_bool),
];
