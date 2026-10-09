use arrow::{array::Decimal128Array, datatypes::DataType as ArrowDataType};
use cudf::{
    DataFrame, DataType, ErrorCode, ErrorKind, Exec, ExecutionBudgetEnforcement, ExecutionBudgetRequest, TypeId,
    frame::dataframe::InfoDetail,
    ops::{col, lit_null, when},
    testing::prelude::*,
};

use crate::common::{
    assert_missing_column_error, column_from_slice, column_from_slice_nullable, column_from_strs,
    dataframe_from_columns, error_context_value, execution_budget_token, lock_event_observation,
    observed_exec_mr_values, reset_exec_observation, scalar, series, with_test_runtime,
};

fn decimal128_scale_zero_frame() -> DataFrame {
    let amount = series::<i64>("amount", &[1, -2, 0])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal128, 0).unwrap())
        .unwrap();
    DataFrame::from_series([("amount", amount)]).unwrap()
}

fn assert_amount_decimal128_scale_8(df: &DataFrame) {
    let target = DataType::fixed_point(TypeId::Decimal128, -8).unwrap();
    let amount = df.series("amount").unwrap();
    assert_eq!(amount.dtype().unwrap(), target);

    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Decimal128(38, 8));
    let amount = batch.column(0).as_any().downcast_ref::<Decimal128Array>().unwrap();
    assert_eq!(amount.values().as_ref(), &[100_000_000i128, -200_000_000, 0]);
}

fn test_with_column_adds_new_column() {
    let df = dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let new_col = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let result = df.with_column("b", new_col).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    let b_vals: Vec<f64> = result.series("b").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0]);
}

fn test_with_column_replaces_existing_column() {
    let df = dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2]).unwrap(), column_from_slice(&[10i32, 20]).unwrap()],
    )
    .unwrap();

    let replacement = column_from_slice(&[99i32, 100]).unwrap();
    let result = df.with_column("b", replacement).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);

    let b_vals: Vec<i32> = result.series("b").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(b_vals, vec![99, 100]);
}

fn test_with_column_preserves_schema_order_on_replace() {
    let df = dataframe_from_columns(
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        vec![
            column_from_slice(&[1i32]).unwrap(),
            column_from_slice(&[2i32]).unwrap(),
            column_from_slice(&[3i32]).unwrap(),
        ],
    )
    .unwrap();

    let replacement = column_from_slice(&[42i32]).unwrap();
    let result = df.with_column("y", replacement).unwrap();

    assert_eq!(result.column_names(), vec!["x", "y", "z"]);

    let y_val: Vec<i32> = result.series("y").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(y_val, vec![42]);
}

fn test_with_column_mismatched_row_count_returns_error() {
    let df = dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();

    let new_col = column_from_slice(&[10i32, 20]).unwrap();
    let err = df.with_column("b", new_col).expect_err("expected row count mismatch error");
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "dataframe::with_column");
    assert_eq!(error_context_value(&err, "column_name").as_deref(), Some("b"));
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("3"));
}

fn test_fillna_with_scalar_fills_values() {
    let col_a = column_from_slice(&[1i32, 2, 3]).unwrap();
    let col_b = column_from_slice_nullable(&[10i32, 0, 30], &[true, false, true]).unwrap();

    let df = dataframe_from_columns(vec!["a".to_string(), "b".to_string()], vec![col_a, col_b]).unwrap();

    let fill = scalar(99i32).unwrap();
    let result = df.on(Exec::RETAINED).fillna_with_scalar("b", &fill).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    let b_vals: Vec<i32> = result.series("b").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(b_vals, vec![10, 99, 30]);
}

fn test_fillna_with_scalar_nonexistent_column_returns_error() {
    let df = dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32]).unwrap()]).unwrap();

    let fill = scalar(0i32).unwrap();
    let err = df.on(Exec::RETAINED).fillna_with_scalar("missing", &fill).expect_err("expected error");
    assert_missing_column_error(&err, "missing");
}

fn test_fillna_with_scalar_preserves_other_columns() {
    let col_a = column_from_slice(&[1i32, 2]).unwrap();
    let col_b = column_from_slice_nullable(&[10i32, 0], &[true, false]).unwrap();
    let col_c = column_from_slice(&[100i64, 200]).unwrap();

    let df = dataframe_from_columns(vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![col_a, col_b, col_c])
        .unwrap();

    let fill = scalar(99i32).unwrap();
    let result = df.on(Exec::RETAINED).fillna_with_scalar("b", &fill).unwrap();

    assert_eq!(result.column_names(), vec!["a", "b", "c"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2]);

    let c_vals: Vec<i64> = result.series("c").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(c_vals, vec![100, 200]);
}

fn test_cast_column_changes_type() {
    let df = dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();

    let target = DataType::new(TypeId::Float64);
    let result = df.on(Exec::RETAINED).cast_column("b", target).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["a", "b"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    let b_type = result.series("b").unwrap().dtype().unwrap();
    assert_eq!(b_type.id(), TypeId::Float64);
    let b_vals: Vec<f64> = result.series("b").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(b_vals, vec![10.0, 20.0, 30.0]);
}

fn test_cast_column_nonexistent_returns_error() {
    let df = dataframe_from_columns(vec!["a".to_string()], vec![column_from_slice(&[1i32]).unwrap()]).unwrap();

    let target = DataType::new(TypeId::Float64);
    let err = df.on(Exec::RETAINED).cast_column("missing", target).expect_err("expected error");
    assert_missing_column_error(&err, "missing");
}

fn test_cast_column_preserves_schema_order() {
    let df = dataframe_from_columns(
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        vec![
            column_from_slice(&[1i32]).unwrap(),
            column_from_slice(&[2i32]).unwrap(),
            column_from_slice(&[3i32]).unwrap(),
        ],
    )
    .unwrap();

    let target = DataType::new(TypeId::Int64);
    let result = df.on(Exec::RETAINED).cast_column("y", target).unwrap();

    assert_eq!(result.column_names(), vec!["x", "y", "z"]);

    let y_type = result.series("y").unwrap().dtype().unwrap();
    assert_eq!(y_type.id(), TypeId::Int64);
    let y_vals: Vec<i64> = result.series("y").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(y_vals, vec![2]);
}

fn test_with_column_add_preserves_existing_string_column() {
    let df = dataframe_from_columns(
        vec!["name".to_string(), "a".to_string()],
        vec![column_from_strs(&["alice", "bob", "cara"]).unwrap(), column_from_slice(&[1i32, 2, 3]).unwrap()],
    )
    .unwrap();

    let result = df.with_column("b", column_from_slice(&[10i64, 20, 30]).unwrap()).unwrap();

    assert_eq!(result.column_names(), vec!["name", "a", "b"]);

    let names = result.series("name").unwrap().to_strings_strict().unwrap();
    assert_eq!(names, vec!["alice", "bob", "cara"]);

    let a_vals: Vec<i32> = result.series("a").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);

    let b_vals: Vec<i64> = result.series("b").unwrap().to_vec_strict::<i64>().unwrap();
    assert_eq!(b_vals, vec![10, 20, 30]);
}

fn test_fillna_with_scalar_with_string_sibling_column() {
    let df = dataframe_from_columns(
        vec!["name".to_string(), "b".to_string()],
        vec![
            column_from_strs(&["x", "y", "z"]).unwrap(),
            column_from_slice_nullable(&[10i32, 0, 30], &[true, false, true]).unwrap(),
        ],
    )
    .unwrap();

    let fill = scalar(99i32).unwrap();
    let result = df.on(Exec::RETAINED).fillna_with_scalar("b", &fill).unwrap();

    let names = result.series("name").unwrap().to_strings_strict().unwrap();
    assert_eq!(names, vec!["x", "y", "z"]);

    let b_vals: Vec<i32> = result.series("b").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(b_vals, vec![10, 99, 30]);
}

fn test_cast_column_with_string_sibling_column() {
    let df = dataframe_from_columns(
        vec!["name".to_string(), "qty".to_string()],
        vec![column_from_strs(&["a", "b", "c"]).unwrap(), column_from_slice(&[1i32, 2, 3]).unwrap()],
    )
    .unwrap();

    let result = df.on(Exec::RETAINED).cast_column("qty", DataType::new(TypeId::Float64)).unwrap();

    let names = result.series("name").unwrap().to_strings_strict().unwrap();
    assert_eq!(names, vec!["a", "b", "c"]);

    let qty_type = result.series("qty").unwrap().dtype().unwrap();
    assert_eq!(qty_type.id(), TypeId::Float64);

    let qty_vals: Vec<f64> = result.series("qty").unwrap().to_vec_strict::<f64>().unwrap();
    assert_eq!(qty_vals, vec![1.0, 2.0, 3.0]);
}

fn expect_cudf_err<T>(result: cudf::Result<T>, message: &str) -> cudf::Error {
    match result {
        Ok(_) => panic!("{message}"),
        Err(err) => err,
    }
}

fn dataframe_device_allocated_size_bytes(df: &DataFrame) -> usize {
    df.info_with(InfoDetail::DEVICE_ALLOCATED).unwrap().memory_accounting.unwrap().device_allocated_bytes.unwrap()
}

fn assert_all_observed_exec_mrs_are_manager_mr(token: &cudf::ExecutionBudgetToken) {
    let expected_mr = token.memory_resource().testing_raw_handle_value();
    let observed_mrs = observed_exec_mr_values();

    assert!(!observed_mrs.is_empty(), "expected at least one observed budgeted execution allocation route");
    assert!(
        observed_mrs.iter().all(|observed| *observed == expected_mr),
        "expected all observed execution memory resources to be the device manager MR {expected_mr:#x}, got {observed_mrs:?}"
    );
}

fn test_with_columns_with_budget_reports_output_and_routes_allocator() {
    let df = dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(
            runtime,
            ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("shape_with_columns"),
        )
        .unwrap()
    });

    let _event_observation = lock_event_observation();
    reset_exec_observation();
    let outcome =
        token.run("dataframe::with_columns", |exec| df.on(exec).with_columns([("sum", col("a") + col("b"))])).unwrap();
    let result = outcome.value();
    let report = outcome.report();
    let output_bytes = dataframe_device_allocated_size_bytes(result);

    assert_eq!(result.column_names(), &["a", "b", "sum"]);
    assert_eq!(result.series("sum").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
    assert_eq!(report.request().boundary(), "shape_with_columns");
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.output_rows(), Some(3));
    assert_eq!(result.testing_execution_domain_mr_value(), token.memory_resource().testing_raw_handle_value());
    assert_all_observed_exec_mrs_are_manager_mr(&token);
}

fn test_with_columns_with_budget_case_report_records_output() {
    let df = dataframe_from_columns(
        vec!["score".to_string(), "fallback".to_string()],
        vec![column_from_slice(&[1.0f64, 2.0, 3.0]).unwrap(), column_from_slice(&[1.0f64, 9.0, 3.0]).unwrap()],
    )
    .unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("case_preflight"))
            .unwrap()
    });
    let expr = when(col("score").eq(col("fallback"))).then(col("score")).otherwise(lit_null(TypeId::Float64));

    let outcome =
        token.run("dataframe::with_columns", |exec| df.on(exec).with_columns([("derived_score", expr)])).unwrap();
    let report = outcome.report();
    let output_bytes = dataframe_device_allocated_size_bytes(outcome.value());

    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.output_rows(), Some(3));
    assert_eq!(outcome.value().series("derived_score").unwrap().null_count(), 1);
}

fn test_cast_column_with_budget_reports_output_and_routes_allocator() {
    let df = dataframe_from_columns(
        vec!["id".to_string(), "qty".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("shape_cast"))
            .unwrap()
    });

    let _event_observation = lock_event_observation();
    reset_exec_observation();
    let outcome = token
        .run("dataframe::cast_column", |exec| df.on(exec).cast_column("qty", DataType::new(TypeId::Float64)))
        .unwrap();
    let result = outcome.value();
    let report = outcome.report();
    let output_bytes = dataframe_device_allocated_size_bytes(result);

    assert_eq!(result.series("qty").unwrap().dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.series("qty").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(report.request().boundary(), "shape_cast");
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.output_rows(), Some(3));
    assert_eq!(result.testing_execution_domain_mr_value(), token.memory_resource().testing_raw_handle_value());
    assert_all_observed_exec_mrs_are_manager_mr(&token);
}

fn test_cast_column_decimal128_scale_to_scale_preserves_requested_scale() {
    let df = decimal128_scale_zero_frame();

    let result =
        df.on(Exec::RETAINED).cast_column("amount", DataType::fixed_point(TypeId::Decimal128, -8).unwrap()).unwrap();

    assert_amount_decimal128_scale_8(&result);
}

fn test_cast_column_with_budget_decimal128_scale_to_scale_reports_output_and_routes_allocator() {
    let df = decimal128_scale_zero_frame();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(512).with_boundary("shape_cast_decimal_scale"))
            .unwrap()
    });

    let _event_observation = lock_event_observation();
    reset_exec_observation();
    let outcome = token
        .run("dataframe::cast_column", |exec| {
            df.on(exec).cast_column("amount", DataType::fixed_point(TypeId::Decimal128, -8).unwrap())
        })
        .unwrap();
    let result = outcome.value();
    let report = outcome.report();
    let output_bytes = dataframe_device_allocated_size_bytes(result);

    assert_amount_decimal128_scale_8(result);
    assert_eq!(report.request().boundary(), "shape_cast_decimal_scale");
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(report.retained_bytes(), Some(output_bytes));
    assert_eq!(report.output_bytes(), Some(output_bytes));
    assert_eq!(report.output_rows(), Some(3));
    let usage = token.reservation_usage().unwrap();
    assert!(usage.peak_aligned_bytes > 0);
    assert!(usage.peak_aligned_bytes <= report.request().budget_bytes() as u64);
    assert_eq!(usage.failed_allocation_count, 0);
    assert_eq!(result.testing_execution_domain_mr_value(), token.memory_resource().testing_raw_handle_value());
    assert_all_observed_exec_mrs_are_manager_mr(&token);
}

fn test_with_columns_with_budget_reports_exceeded_budget_details() {
    let df = dataframe_from_columns(
        vec!["a".to_string(), "b".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(1).with_boundary("shape_with_columns")).unwrap()
    });

    let err = expect_cudf_err(
        token.run("dataframe::with_columns", |exec| df.on(exec).with_columns([("sum", col("a") + col("b"))])),
        "one-byte budget should not cover with_columns allocations",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "dataframe::with_columns");
    assert_eq!(err.budget_exhaustion().expect("budget scope").boundary(), "shape_with_columns");
    assert_eq!(
        err.budget_exhaustion().and_then(|budget| budget.usage()).expect("member usage").allocation_limit_bytes,
        1
    );
}

fn test_cast_column_with_budget_reports_exceeded_budget_details() {
    let df = dataframe_from_columns(
        vec!["id".to_string(), "qty".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i32, 20, 30]).unwrap()],
    )
    .unwrap();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(1).with_boundary("shape_cast")).unwrap()
    });

    let err = expect_cudf_err(
        token.run("dataframe::cast_column", |exec| df.on(exec).cast_column("qty", DataType::new(TypeId::Float64))),
        "one-byte budget should not cover cast-column allocations",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "dataframe::cast_column");
    assert_eq!(err.budget_exhaustion().expect("budget scope").boundary(), "shape_cast");
    assert_eq!(
        err.budget_exhaustion().and_then(|budget| budget.usage()).expect("member usage").allocation_limit_bytes,
        1
    );
}

fn test_cast_column_with_budget_decimal128_scale_to_scale_exceeded_budget_reports_budget_facts() {
    let df = decimal128_scale_zero_frame();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(1).with_boundary("shape_cast_decimal_scale"))
            .unwrap()
    });

    let err = expect_cudf_err(
        token.run("dataframe::cast_column", |exec| {
            df.on(exec).cast_column("amount", DataType::fixed_point(TypeId::Decimal128, -8).unwrap())
        }),
        "one-byte budget should not cover Decimal128 scale-to-scale cast allocations",
    );

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.operation(), "dataframe::cast_column");
    assert_eq!(err.budget_exhaustion().expect("budget scope").boundary(), "shape_cast_decimal_scale");
    assert_eq!(
        err.budget_exhaustion().and_then(|budget| budget.usage()).expect("member usage").allocation_limit_bytes,
        1
    );
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("column_ops_tests::test_with_column_adds_new_column", test_with_column_adds_new_column),
    ("column_ops_tests::test_with_column_replaces_existing_column", test_with_column_replaces_existing_column),
    (
        "column_ops_tests::test_with_column_preserves_schema_order_on_replace",
        test_with_column_preserves_schema_order_on_replace,
    ),
    (
        "column_ops_tests::test_with_column_mismatched_row_count_returns_error",
        test_with_column_mismatched_row_count_returns_error,
    ),
    ("column_ops_tests::test_fillna_with_scalar_fills_values", test_fillna_with_scalar_fills_values),
    (
        "column_ops_tests::test_fillna_with_scalar_nonexistent_column_returns_error",
        test_fillna_with_scalar_nonexistent_column_returns_error,
    ),
    (
        "column_ops_tests::test_fillna_with_scalar_preserves_other_columns",
        test_fillna_with_scalar_preserves_other_columns,
    ),
    ("column_ops_tests::test_cast_column_changes_type", test_cast_column_changes_type),
    ("column_ops_tests::test_cast_column_nonexistent_returns_error", test_cast_column_nonexistent_returns_error),
    ("column_ops_tests::test_cast_column_preserves_schema_order", test_cast_column_preserves_schema_order),
    (
        "column_ops_tests::test_with_column_add_preserves_existing_string_column",
        test_with_column_add_preserves_existing_string_column,
    ),
    (
        "column_ops_tests::test_fillna_with_scalar_with_string_sibling_column",
        test_fillna_with_scalar_with_string_sibling_column,
    ),
    ("column_ops_tests::test_cast_column_with_string_sibling_column", test_cast_column_with_string_sibling_column),
    (
        "column_ops_tests::test_with_columns_with_budget_reports_output_and_routes_allocator",
        test_with_columns_with_budget_reports_output_and_routes_allocator,
    ),
    (
        "column_ops_tests::test_with_columns_with_budget_case_report_records_output",
        test_with_columns_with_budget_case_report_records_output,
    ),
    (
        "column_ops_tests::test_cast_column_with_budget_reports_output_and_routes_allocator",
        test_cast_column_with_budget_reports_output_and_routes_allocator,
    ),
    (
        "column_ops_tests::test_cast_column_decimal128_scale_to_scale_preserves_requested_scale",
        test_cast_column_decimal128_scale_to_scale_preserves_requested_scale,
    ),
    (
        "column_ops_tests::test_cast_column_with_budget_decimal128_scale_to_scale_reports_output_and_routes_allocator",
        test_cast_column_with_budget_decimal128_scale_to_scale_reports_output_and_routes_allocator,
    ),
    (
        "column_ops_tests::test_with_columns_with_budget_reports_exceeded_budget_details",
        test_with_columns_with_budget_reports_exceeded_budget_details,
    ),
    (
        "column_ops_tests::test_cast_column_with_budget_reports_exceeded_budget_details",
        test_cast_column_with_budget_reports_exceeded_budget_details,
    ),
    (
        "column_ops_tests::test_cast_column_with_budget_decimal128_scale_to_scale_exceeded_budget_reports_budget_facts",
        test_cast_column_with_budget_decimal128_scale_to_scale_exceeded_budget_reports_budget_facts,
    ),
];
