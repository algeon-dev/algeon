//! Integration tests for `Series` reduction, scan, and unary methods.
//!
//! All tests require a GPU.

use cudf::{
    ErrorCode, ErrorKind, ExecutionBudgetEnforcement, ExecutionBudgetRequest, ExecutionBudgetStatus, ReduceAgg, TypeId,
    core::error::{FactKeyName, FactValue},
    testing::prelude::*,
};

use crate::common::{error_context_value, last_exec_mr_value, reset_exec_observation, with_test_runtime};

fn assert_no_budget_error_details(err: &cudf::Error) {
    for key in ["budget_boundary", "budget_bytes", "budget_enforcement", "budget_status"] {
        assert_eq!(error_context_value(err, key).as_deref(), None, "{key} should not be attached");
    }
}

fn expect_cudf_err<T>(result: cudf::Result<T>, message: &str) -> cudf::Error {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}

// ===========================================================================
// sum
// ===========================================================================

#[test]
fn test_sum_i32() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let total = s.sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(total.value::<i64>().unwrap(), 60);
}

#[test]
fn test_sum_f64() {
    let s = crate::common::series::<f64>("x", &[1.5, 2.5, 3.0]).unwrap();
    let total = s.sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(total.value::<f64>().unwrap(), 7.0);
}

#[test]
fn test_sum_bool8() {
    let s = crate::common::series::<bool>("x", &[true, false, true, true]).unwrap();
    let total = s.sum().unwrap();

    assert_eq!(total.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(total.value::<i64>().unwrap(), 3);
}

#[test]
fn test_sum_with_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let total = s.sum().unwrap();

    assert_eq!(total.value::<i64>().unwrap(), 40);
}

#[test]
fn test_sum_string_reports_series_operation_on_type_error() {
    let s = crate::common::series("x", &["a", "b"]).unwrap();
    let err = s.sum().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::sum");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_eq!(crate::common::error_context_value(&err, "input_type").as_deref(), None);
}

// ===========================================================================
// min / max
// ===========================================================================

#[test]
fn test_min_i32() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.min().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(m.value::<i32>().unwrap(), 10);
}

#[test]
fn test_max_i32() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.max().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(m.value::<i32>().unwrap(), 30);
}

#[test]
fn test_min_f64() {
    let s = crate::common::series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let m = s.min().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 1.0);
}

#[test]
fn test_max_f64() {
    let s = crate::common::series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let m = s.max().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 3.0);
}

#[test]
fn test_prepared_scalar_max_reports_shape_and_returns_scalar() {
    let s = crate::common::series::<i32>("x", &[30, 10, 40, 20]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Max).unwrap();

    assert_eq!(prepared.input().len(), s.len());
    assert!(matches!(prepared.aggregation(), ReduceAgg::Max));
    assert_eq!(prepared.output_type().id(), TypeId::Int32);
    assert_eq!(prepared.output_shape().output_type().id(), TypeId::Int32);
    assert_eq!(prepared.output_shape().output_rows(), 1);

    let output = prepared.execute().unwrap();
    assert_eq!(output.output_rows(), 1);
    assert_eq!(output.scalar().data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(output.scalar().value::<i32>().unwrap(), 40);
    assert_eq!(output.into_scalar().value::<i32>().unwrap(), 40);
}

#[test]
fn test_prepared_scalar_sum_i32_reports_int64_output_and_returns_scalar() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Sum).unwrap();

    assert_eq!(prepared.output_type().id(), TypeId::Int64);
    assert_eq!(prepared.output_shape().output_type().id(), TypeId::Int64);
    assert_eq!(prepared.output_shape().output_rows(), 1);

    let output = prepared.execute().unwrap();
    assert_eq!(output.output_rows(), 1);
    assert_eq!(output.scalar().data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(output.scalar().value::<i64>().unwrap(), 60);
}

#[test]
fn test_prepared_scalar_max_execute_with_budget_reports_contract_and_routes_allocator() {
    let s = crate::common::series::<i32>("x", &[30, 10, 40, 20]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Max).unwrap();
    let token = with_test_runtime(|runtime| {
        runtime
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("q15_scalar_max"))
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
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::StreamBoundReservation);
    assert_eq!(report.status(), ExecutionBudgetStatus::ReservationEnforced);
    assert_eq!(report.output_rows(), Some(1));
    assert_eq!(report.retained_bytes(), None);
    assert_eq!(report.output_bytes(), None);
    assert_eq!(
        budgeted_exec_mr,
        token.memory_resource().testing_raw_handle_value(),
        "budgeted scalar reduction should route execution through the device manager resource",
    );
}

#[test]
fn test_prepared_scalar_q17_sum_extendedprice_with_budget_reports_current_diagnostics() {
    let s = crate::common::series::<f64>("l_extendedprice", &[1000.0, 1250.5, 749.5]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Sum).unwrap();
    let token = with_test_runtime(|runtime| {
        runtime
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("q17_global_sum"))
            .unwrap()
    });

    let outcome = prepared.execute_with_budget(&token).unwrap();
    let output = outcome.value();
    let report = outcome.report();

    assert_eq!(output.scalar().value::<f64>().unwrap(), 3000.0);
    assert_eq!(report.request().boundary(), "q17_global_sum");
    assert_eq!(report.status(), ExecutionBudgetStatus::ReservationEnforced);
    assert_eq!(report.output_rows(), Some(1));
    assert_eq!(report.retained_bytes(), None);
    assert_eq!(report.output_bytes(), None);
}

#[test]
fn test_prepared_scalar_max_one_byte_budget_preserves_budget_details() {
    let s = crate::common::series::<i32>("x", &[30, 10, 40, 20]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Max).unwrap();
    let token = with_test_runtime(|runtime| {
        runtime.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("q15_scalar_max")).unwrap()
    });

    let err = expect_cudf_err(
        prepared.execute_with_budget(&token),
        "one-byte budget should not cover scalar reduction allocations",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "series::prepared_scalar_reduction::execute_with_budget");
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("q15_scalar_max"));
    assert_eq!(error_context_value(&err, "budget_bytes").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "budget_enforcement").as_deref(), Some("stream_bound_reservation"));
    assert_eq!(error_context_value(&err, "budget_status").as_deref(), Some("exceeded_or_lower_layer_rejected"));
}

#[test]
fn test_prepared_scalar_max_all_null_reports_one_row_and_typed_null_scalar() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Max).unwrap();
    let output = prepared.execute().unwrap();
    let scalar = output.scalar();

    assert_eq!(prepared.output_shape().output_rows(), 1);
    assert_eq!(output.output_rows(), 1);
    assert_eq!(scalar.data_type().unwrap().id(), TypeId::Int32);
    assert!(!scalar.is_valid().unwrap());
}

#[test]
fn test_prepared_scalar_reduction_unsupported_dtype_has_no_budget_details() {
    let s = crate::common::series("x", &["a", "b"]).unwrap();
    let err = expect_cudf_err(s.prepare_scalar_reduction(ReduceAgg::Sum), "expected unsupported dtype error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::prepare_scalar_reduction");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_no_budget_error_details(&err);
}

#[test]
fn test_prepared_scalar_reduction_budget_validation_error_has_no_budget_details() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();
    let prepared = s.prepare_scalar_reduction(ReduceAgg::Std { ddof: u32::MAX }).unwrap();
    let token = with_test_runtime(|runtime| {
        runtime
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("std_validation"))
            .unwrap()
    });

    let err = expect_cudf_err(prepared.execute_with_budget(&token), "expected ddof range validation error");

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::prepared_scalar_reduction::execute_with_budget");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("ddof_exceeds_i32_range"));
    assert_no_budget_error_details(&err);
}

// ===========================================================================
// mean / median
// ===========================================================================

#[test]
fn test_mean_i32() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let m = s.mean().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(m.value::<f64>().unwrap(), 20.0);
}

#[test]
fn test_mean_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();
    let m = s.mean().unwrap();

    assert_eq!(m.value::<f64>().unwrap(), 2.0);
}

#[test]
fn test_median_i32() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let m = s.median().unwrap();

    assert_eq!(m.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(m.value::<f64>().unwrap(), 20.0);
}

// ===========================================================================
// std / var
// ===========================================================================

#[test]
fn test_std_f64() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let ddof: u32 = 0;
    let std_val = s.std(ddof).unwrap();

    assert_eq!(std_val.data_type().unwrap().id(), TypeId::Float64);
    let v = std_val.value::<f64>().unwrap();
    assert!((v - 2.0).abs() < 1e-10, "expected std ≈ 2.0, got {v}");
}

#[test]
fn test_var_f64() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let ddof: u32 = 0;
    let var_val = s.var(ddof).unwrap();

    assert_eq!(var_val.data_type().unwrap().id(), TypeId::Float64);
    let v = var_val.value::<f64>().unwrap();
    assert!((v - 4.0).abs() < 1e-10, "expected var ≈ 4.0, got {v}");
}

#[test]
fn test_std_ddof_above_i32_max_uses_series_operation_contract() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();

    let err = s.std(u32::MAX).expect_err("expected ddof range validation error");

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

#[test]
fn test_var_ddof_above_i32_max_uses_series_operation_contract() {
    let s = crate::common::series::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();

    let err = s.var(u32::MAX).expect_err("expected ddof range validation error");

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

// ===========================================================================
// any / all
// ===========================================================================

#[test]
fn test_any_true() {
    let s = crate::common::series::<bool>("x", &[false, true, false]).unwrap();
    let result = s.any().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Bool8);
    assert!(result.value::<bool>().unwrap());
}

#[test]
fn test_any_false() {
    let s = crate::common::series::<bool>("x", &[false, false, false]).unwrap();
    let result = s.any().unwrap();

    assert!(!result.value::<bool>().unwrap());
}

#[test]
fn test_all_true() {
    let s = crate::common::series::<bool>("x", &[true, true, true]).unwrap();
    let result = s.all().unwrap();

    assert!(result.value::<bool>().unwrap());
}

#[test]
fn test_all_false() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.all().unwrap();

    assert!(!result.value::<bool>().unwrap());
}

// ===========================================================================
// product
// ===========================================================================

#[test]
fn test_product_i32() {
    let s = crate::common::series::<i32>("x", &[2, 3, 4]).unwrap();
    let result = s.product().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(result.value::<i32>().unwrap(), 24);
}

#[test]
fn test_product_f64() {
    let s = crate::common::series::<f64>("x", &[2.0, 3.0, 4.0]).unwrap();
    let result = s.product().unwrap();

    assert_eq!(result.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(result.value::<f64>().unwrap(), 24.0);
}

// ===========================================================================
// minmax
// ===========================================================================

#[test]
fn test_minmax_i32() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let (min, max) = s.minmax().unwrap();

    assert_eq!(min.value::<i32>().unwrap(), 10);
    assert_eq!(max.value::<i32>().unwrap(), 30);
}

#[test]
fn test_minmax_f64() {
    let s = crate::common::series::<f64>("x", &[3.0, 1.0, 2.0]).unwrap();
    let (min, max) = s.minmax().unwrap();

    assert_eq!(min.value::<f64>().unwrap(), 1.0);
    assert_eq!(max.value::<f64>().unwrap(), 3.0);
}

// ===========================================================================
// cumsum / cummin / cummax / cumprod
// ===========================================================================

#[test]
fn test_cumsum_i32() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let cs = s.cumsum().unwrap();
    assert_eq!(cs.to_vec_strict::<i32>().unwrap(), vec![10, 30, 60]);
}

#[test]
fn test_cumsum_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();
    let cs = s.cumsum().unwrap();

    assert_eq!(cs.to_vec_strict::<f64>().unwrap(), vec![1.0, 3.0, 6.0]);
}

#[test]
fn test_cummin_i32() {
    let s = crate::common::series::<i32>("x", &[3, 1, 2]).unwrap();
    let cm = s.cummin().unwrap();
    assert_eq!(cm.to_vec_strict::<i32>().unwrap(), vec![3, 1, 1]);
}

#[test]
fn test_cummax_i32() {
    let s = crate::common::series::<i32>("x", &[1, 3, 2]).unwrap();
    let cm = s.cummax().unwrap();
    assert_eq!(cm.to_vec_strict::<i32>().unwrap(), vec![1, 3, 3]);
}

#[test]
fn test_cumprod_i32() {
    let s = crate::common::series::<i32>("x", &[2, 3, 4]).unwrap();
    let cp = s.cumprod().unwrap();
    assert_eq!(cp.to_vec_strict::<i32>().unwrap(), vec![2, 6, 24]);
}

// ===========================================================================
// is_nan / is_not_nan
// ===========================================================================

#[test]
fn test_is_nan_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, f64::NAN, 3.0]).unwrap();
    let mask = s.is_nan().unwrap();
    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_is_not_nan_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, f64::NAN, 3.0]).unwrap();
    let mask = s.is_not_nan().unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

// ===========================================================================
// abs
// ===========================================================================

#[test]
fn test_abs_i32() {
    let s = crate::common::series::<i32>("x", &[-3, -1, 0, 2]).unwrap();
    let result = s.abs().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![3, 1, 0, 2]);
}

#[test]
fn test_abs_f64() {
    let s = crate::common::series::<f64>("x", &[-3.0, -1.5, 0.0, 2.5]).unwrap();
    let result = s.abs().unwrap();

    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![3.0, 1.5, 0.0, 2.5]);
}

// ===========================================================================
// neg
// ===========================================================================

#[test]
fn test_neg_i32() {
    let s = crate::common::series::<i32>("x", &[1, -2, 3]).unwrap();
    let result = s.neg().unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![-1, 2, -3]);
}

#[test]
fn test_neg_f64() {
    let s = crate::common::series::<f64>("x", &[1.5, -2.5]).unwrap();
    let result = s.neg().unwrap();

    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![-1.5, 2.5]);
}

// ===========================================================================
// logical_not
// ===========================================================================

#[test]
fn test_logical_not_bool() {
    let s = crate::common::series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.logical_not().unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}
