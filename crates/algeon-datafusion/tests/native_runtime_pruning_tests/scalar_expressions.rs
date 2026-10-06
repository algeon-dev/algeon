use std::fs::{self, File};

use algeon_datafusion::{capability::CapabilityReason, report::CandidateOutcome};
use arrow::{
    array::{Array, AsArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array},
    compute::concat_batches,
    datatypes::{Float32Type, Float64Type},
};
use cudf::testing::arrow::{
    arrow_host_export_call_count_for_test, arrow_host_import_call_count_for_test,
    reset_arrow_host_interop_observation_for_test,
};
use datafusion::parquet::arrow::ArrowWriter;

use super::*;

#[tokio::test]
async fn native_runtime_negation_matches_wrapping_integers_and_ieee_float_bits() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-negation");
    let path = dir.join("input.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("i8", DataType::Int8, true),
        Field::new("i16", DataType::Int16, true),
        Field::new("i32", DataType::Int32, true),
        Field::new("i64", DataType::Int64, true),
        Field::new("f32", DataType::Float32, true),
        Field::new("f64", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5, 6])),
            Arc::new(Int8Array::from(vec![Some(i8::MIN), Some(i8::MAX), Some(-1), Some(0), None, Some(1), Some(-2)])),
            Arc::new(Int16Array::from(vec![
                Some(i16::MIN),
                Some(i16::MAX),
                Some(-1),
                Some(0),
                None,
                Some(1),
                Some(-2),
            ])),
            Arc::new(Int32Array::from(vec![
                Some(i32::MIN),
                Some(i32::MAX),
                Some(-1),
                Some(0),
                None,
                Some(1),
                Some(-2),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(i64::MIN),
                Some(i64::MAX),
                Some(-1),
                Some(0),
                None,
                Some(1),
                Some(-2),
            ])),
            Arc::new(Float32Array::from(vec![
                Some(0.0),
                Some(-0.0),
                Some(f32::INFINITY),
                Some(f32::NEG_INFINITY),
                None,
                Some(1.5),
                Some(-1.5),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(0.0),
                Some(-0.0),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
                None,
                Some(1.5),
                Some(-1.5),
            ])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let sql = "SELECT row_id, -i8 AS n8, -i16 AS n16, -i32 AS n32, -i64 AS n64, \
               -f32 AS nf32, -f64 AS nf64 FROM bench ORDER BY row_id";
    let cpu_batches = cpu_sql_output(&path, sql).await;
    let optimization = optimize_with_pruning_template(sql_plan(&path, SessionConfig::new(), sql).await);
    assert_eq!(optimization.report.selected_count(), 1, "{:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "negation must lower to native execution");
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);
    let cpu = concat_batches(&cpu_batches[0].schema(), &cpu_batches).unwrap();
    let native = concat_batches(&native_batches[0].schema(), &native_batches).unwrap();
    for row in 0..native.num_rows() {
        if !cpu.column(5).is_null(row) {
            assert_eq!(
                cpu.column(5).as_primitive::<Float32Type>().value(row).to_bits(),
                native.column(5).as_primitive::<Float32Type>().value(row).to_bits()
            );
            assert_eq!(
                cpu.column(6).as_primitive::<Float64Type>().value(row).to_bits(),
                native.column(6).as_primitive::<Float64Type>().value(row).to_bits()
            );
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_scalar_expressions_match_cpu_and_collect_required_columns() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-scalar_expression-scalars");
    let path = dir.join("input.parquet");
    write_scalar_expression_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_SCALAR_EXPRESSIONS_QUERY).await;
    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_SCALAR_EXPRESSIONS_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.source_diagnostics[0].read_column_count, 3);
    assert_eq!(optimization.source_diagnostics[0].projected_output_column_count, 3);
    assert_eq!(optimization.source_diagnostics[0].pure_filter_column_count, 0);
    assert_algeon_native_exec(&optimization.plan, "scalar plan must rewrite to AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_nullable_scalar_expressions_match_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-scalar_expression-nullable-scalars");
    let path = dir.join("input.parquet");
    write_nullable_scalar_expression_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_NULLABLE_SCALAR_EXPRESSIONS_QUERY).await;
    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_NULLABLE_SCALAR_EXPRESSIONS_QUERY).await;
    let plan_display = displayable(native_physical_plan.as_ref()).indent(true).to_string();
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(
        optimization.report.selected_count(),
        1,
        "outcomes={:?}\n{plan_display}",
        optimization.report.outcomes()
    );
    assert_eq!(
        optimization.report.not_selected_count(),
        0,
        "outcomes={:?}\n{plan_display}",
        optimization.report.outcomes()
    );
    assert_eq!(optimization.source_diagnostics[0].read_column_count, 3);
    assert_eq!(optimization.source_diagnostics[0].projected_output_column_count, 3);
    assert_eq!(optimization.source_diagnostics[0].pure_filter_column_count, 0);
    assert_algeon_native_exec(&optimization.plan, "nullable scalar plan must rewrite to AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_null_semantics_match_cpu_by_expression_family() {
    let cases = [
        ("like", SQL_NULL_SEMANTICS_LIKE_QUERY),
        ("substring", SQL_NULL_SEMANTICS_SUBSTRING_QUERY),
        ("string-case", SQL_NULL_SEMANTICS_STRING_CASE_QUERY),
        ("string-predicates", SQL_NULL_SEMANTICS_STRING_PREDICATES_QUERY),
        ("trim-family", SQL_TRIM_FAMILY_QUERY),
        ("in-list", SQL_NULL_SEMANTICS_IN_LIST_QUERY),
        ("case", SQL_NULL_SEMANTICS_CASE_QUERY),
        ("case-no-else", SQL_NULL_SEMANTICS_CASE_NO_ELSE_QUERY),
        ("extract", SQL_NULL_SEMANTICS_EXTRACT_QUERY),
        ("date-trunc-fixed-grains", SQL_DATE_TRUNC_FIXED_GRAINS_QUERY),
        ("interval", SQL_NULL_SEMANTICS_INTERVAL_QUERY),
        // Integer modulo now has a dedicated DataFusion-baseline retention test below.
        ("bitwise", SQL_BITWISE_SCALAR_QUERY),
        ("is-null", SQL_NULL_SEMANTICS_IS_NULL_QUERY),
        ("round", SQL_ROUND_SCALAR_QUERY),
        ("abs", SQL_ABS_SCALAR_QUERY),
        ("sqrt", SQL_SQRT_SCALAR_QUERY),
        ("string-concat-operator", SQL_STRING_CONCAT_OPERATOR_QUERY),
        ("null-aware-comparison", SQL_NULL_AWARE_COMPARISON_QUERY),
        ("nullif", SQL_NULLIF_SCALAR_QUERY),
        ("integer-to-timestamp-cast", SQL_INTEGER_TO_TIMESTAMP_CAST_QUERY),
    ];

    for (name, sql) in cases {
        let dir = unique_temp_dir(
            format!("algeon-datafusion-native-runtime-scalar_expression-null-semantics-{name}").as_str(),
        );
        let path = dir.join("input.parquet");
        write_nullable_scalar_expression_input_parquet(&path);

        let cpu_batches = cpu_sql_output(&path, sql).await;
        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let plan_display = displayable(native_physical_plan.as_ref()).indent(true).to_string();
        let optimization = optimize_with_pruning_template(native_physical_plan);
        assert_eq!(
            optimization.report.selected_count(),
            1,
            "{name} outcomes={:?}\n{plan_display}",
            optimization.report.outcomes()
        );
        assert_eq!(
            optimization.report.not_selected_count(),
            0,
            "{name} outcomes={:?}\n{plan_display}",
            optimization.report.outcomes()
        );
        assert_algeon_native_exec(&optimization.plan, "{name} must rewrite to AlgeonNativeExec");

        let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
        reset_arrow_host_interop_observation_for_test();
        let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
        let metrics =
            below_native_admission(&optimization.plan).metrics().expect("{name} must expose DataFusion metrics");
        assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0, "{name}");
        assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1, "{name}");

        assert_batches_equal(&cpu_batches, &native_batches);

        fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test]
async fn native_runtime_ln_exp_match_cpu_ieee_results() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-ln-exp");
    let path = dir.join("input.parquet");
    write_nullable_scalar_expression_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_LN_EXP_SCALAR_QUERY).await;
    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_LN_EXP_SCALAR_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "LN/EXP plan must rewrite to AlgeonNativeExec");
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();

    let cpu = concat_batches(&cpu_batches[0].schema(), &cpu_batches).unwrap();
    let native = concat_batches(&native_batches[0].schema(), &native_batches).unwrap();
    assert_eq!(cpu.schema().fields(), native.schema().fields());
    assert_eq!(cpu.num_rows(), 6);
    assert_eq!(native.num_rows(), cpu.num_rows());
    assert_eq!(cpu.column(0).as_ref(), native.column(0).as_ref(), "row_id order must match");

    let cpu_ln_f64 = cpu.column(1).as_primitive::<Float64Type>();
    assert!(cpu_ln_f64.value(0).is_nan(), "fixture must cover LN of a negative input");
    assert_eq!(cpu_ln_f64.value(1), f64::NEG_INFINITY, "fixture must cover LN(0)");
    assert!(cpu_ln_f64.is_null(2), "fixture must cover a null input");
    assert_eq!(cpu.column(3).as_primitive::<Float64Type>().value(4), f64::INFINITY, "fixture must cover EXP overflow");
    assert_eq!(cpu.column(4).as_primitive::<Float32Type>().value(4), f32::INFINITY, "fixture must cover EXP overflow");

    for (column, ulp_distance) in [
        (1, f64_ulp_distance as fn(f64, f64) -> u64),
        (2, f32_ulp_distance),
        (3, f64_ulp_distance),
        (4, f32_ulp_distance),
    ] {
        let name = cpu.schema().field(column).name().clone();
        let expected = float_values(cpu.column(column).as_ref());
        let actual = float_values(native.column(column).as_ref());
        for (row, (expected, actual)) in expected.into_iter().zip(actual).enumerate() {
            assert!(
                ieee_values_match(expected, actual, ulp_distance),
                "{name} row {row}: cpu={expected:?} native={actual:?}"
            );
        }
    }

    fs::remove_dir_all(dir).unwrap();
}

/// Null, NaN, and infinity must match exactly. Finite values may differ by a
/// few units in the last place: DataFusion evaluates LN/EXP with the host libm
/// while cuDF uses CUDA's device math library, and neither is required to be
/// correctly rounded.
fn ieee_values_match(expected: Option<f64>, actual: Option<f64>, ulp_distance: fn(f64, f64) -> u64) -> bool {
    const MAX_ULP_DISTANCE: u64 = 2;
    match (expected, actual) {
        (None, None) => true,
        (Some(expected), Some(actual)) if !expected.is_finite() || !actual.is_finite() => {
            expected == actual || (expected.is_nan() && actual.is_nan())
        },
        (Some(expected), Some(actual)) => ulp_distance(expected, actual) <= MAX_ULP_DISTANCE,
        _ => false,
    }
}

fn float_values(array: &dyn Array) -> Vec<Option<f64>> {
    match array.data_type() {
        DataType::Float64 => array.as_primitive::<Float64Type>().iter().collect(),
        DataType::Float32 => array.as_primitive::<Float32Type>().iter().map(|value| value.map(f64::from)).collect(),
        other => panic!("expected a floating-point column, got {other:?}"),
    }
}

fn f64_ulp_distance(expected: f64, actual: f64) -> u64 {
    expected.to_bits().abs_diff(actual.to_bits())
}

// Both values were widened from Float32, so narrowing them back is exact.
fn f32_ulp_distance(expected: f64, actual: f64) -> u64 {
    u64::from((expected as f32).to_bits().abs_diff((actual as f32).to_bits()))
}

#[tokio::test]
async fn native_runtime_ilike_rejects_before_native_execution() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-ilike-reject");
    let path = dir.join("input.parquet");
    write_nullable_scalar_expression_input_parquet(&path);

    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_NULL_SEMANTICS_ILIKE_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);

    assert_eq!(optimization.report.selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert!(
        matches!(
            optimization.report.outcomes()[0],
            CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
        ),
        "outcomes={:?}",
        optimization.report.outcomes()
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_integer_modulo_rejects_before_native_execution() {
    // SQL integer modulo retains the DataFusion baseline until GPU by-zero semantics match DataFusion.
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-integer-modulo-reject");
    let path = dir.join("input.parquet");
    write_nullable_scalar_expression_input_parquet(&path);

    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_NULL_SEMANTICS_MODULO_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);

    assert_eq!(optimization.report.selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert!(
        matches!(
            optimization.report.outcomes()[0],
            CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
        ),
        "outcomes={:?}",
        optimization.report.outcomes()
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_invalid_scalar_expressions_reject_before_native_execution() {
    for (name, sql) in [
        ("substring-zero-start", SQL_INVALID_SUBSTRING_ZERO_START_QUERY),
        ("substring-dynamic-start", SQL_INVALID_SUBSTRING_DYNAMIC_START_QUERY),
        ("substring-dynamic-length", SQL_INVALID_SUBSTRING_DYNAMIC_LENGTH_QUERY),
        ("substring-negative-length", SQL_INVALID_SUBSTRING_NEGATIVE_LENGTH_QUERY),
        ("substring-stop-overflow", SQL_INVALID_SUBSTRING_STOP_OVERFLOW_QUERY),
    ] {
        let dir = unique_temp_dir(format!("algeon-datafusion-native-runtime-invalid-{name}").as_str());
        let path = dir.join("input.parquet");
        write_scalar_expression_input_parquet(&path);

        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let optimization = optimize_with_pruning_template(native_physical_plan);

        assert_eq!(
            optimization.report.selected_count(),
            0,
            "{name} should not rewrite: {:?}",
            optimization.report.outcomes()
        );
        assert_eq!(
            optimization.report.not_selected_count(),
            1,
            "{name} should reject: {:?}",
            optimization.report.outcomes()
        );
        assert!(
            matches!(
                optimization.report.outcomes()[0],
                CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
            ),
            "{name} outcomes={:?}",
            optimization.report.outcomes()
        );

        fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test]
async fn native_runtime_date_trunc_calendrical_grains_reject_before_native_execution() {
    for (name, sql) in [
        ("date-trunc-month", SQL_DATE_TRUNC_MONTH_UNSUPPORTED_QUERY),
        ("date-trunc-year", SQL_DATE_TRUNC_YEAR_UNSUPPORTED_QUERY),
    ] {
        let dir = unique_temp_dir(format!("algeon-datafusion-native-runtime-invalid-{name}").as_str());
        let path = dir.join("input.parquet");
        write_nullable_scalar_expression_input_parquet(&path);

        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let optimization = optimize_with_pruning_template(native_physical_plan);

        assert_eq!(
            optimization.report.selected_count(),
            0,
            "{name} should not rewrite: {:?}",
            optimization.report.outcomes()
        );
        assert_eq!(
            optimization.report.not_selected_count(),
            1,
            "{name} should reject: {:?}",
            optimization.report.outcomes()
        );
        assert!(
            matches!(
                optimization.report.outcomes()[0],
                CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
            ),
            "{name} outcomes={:?}",
            optimization.report.outcomes()
        );

        fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test]
async fn native_runtime_substring_max_start_length_one_is_in_contract() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-substring-max-start-boundary");
    let path = dir.join("input.parquet");
    write_scalar_expression_input_parquet(&path);

    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_SUBSTRING_MAX_START_LENGTH_ONE_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);

    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());

    fs::remove_dir_all(dir).unwrap();
}
