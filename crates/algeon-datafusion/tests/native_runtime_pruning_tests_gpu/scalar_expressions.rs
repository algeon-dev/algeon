use std::fs::{self, File};

use algeon_datafusion::{
    capability::CapabilityReason,
    error::{DataFusionErrorExt, ErrorCode},
    report::CandidateOutcome,
};
use arrow::{
    array::{
        Array, AsArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int8Array, Int16Array,
        Int32Array, Int64Array, StringArray,
    },
    compute::{cast, concat_batches},
    datatypes::{Float32Type, Float64Type, TimeUnit},
};
use cudf::{
    testing,
    testing::arrow::{
        arrow_host_export_call_count_for_test, arrow_host_import_call_count_for_test,
        reset_arrow_host_interop_observation_for_test,
    },
};
use datafusion::parquet::arrow::ArrowWriter;
use tokio::runtime::Builder as TestRuntimeBuilder;

use super::*;

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
        ("modulo", SQL_NULL_SEMANTICS_MODULO_QUERY),
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

async fn native_runtime_string_length_units_match_cpu() {
    let cases = [("units", SQL_STRING_LENGTH_UNITS_QUERY), ("filter", SQL_STRING_LENGTH_FILTER_QUERY)];

    for (name, sql) in cases {
        let dir = unique_temp_dir(format!("algeon-datafusion-native-runtime-string-length-{name}").as_str());
        let path = dir.join("input.parquet");
        write_string_length_input_parquet(&path);

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
        assert_algeon_native_exec(&optimization.plan, "string length plan must rewrite to AlgeonNativeExec");

        let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
        reset_arrow_host_interop_observation_for_test();
        let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
        let metrics = below_native_admission(&optimization.plan)
            .metrics()
            .expect("AlgeonNativeExec should expose DataFusion metrics");
        assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0, "{name}");
        assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1, "{name}");

        assert_batches_equal(&cpu_batches, &native_batches);

        fs::remove_dir_all(dir).unwrap();
    }
}

async fn native_runtime_timestamp_extract_fields_match_cpu() {
    let cases = [
        ("seconds", SQL_EXTRACT_TIME_FIELDS_SECONDS_QUERY),
        ("milliseconds", SQL_EXTRACT_TIME_FIELDS_MILLISECONDS_QUERY),
        ("microseconds", SQL_EXTRACT_TIME_FIELDS_MICROSECONDS_QUERY),
        ("nanoseconds", SQL_EXTRACT_TIME_FIELDS_NANOSECONDS_QUERY),
        ("minute-filter", SQL_EXTRACT_MINUTE_FILTER_QUERY),
        ("aggregate-input", SQL_EXTRACT_AGGREGATE_INPUT_QUERY),
        ("minute-from-integer-cast", SQL_EXTRACT_MINUTE_FROM_INTEGER_CAST_QUERY),
    ];

    for (name, sql) in cases {
        let dir = unique_temp_dir(format!("algeon-datafusion-native-runtime-timestamp-extract-{name}").as_str());
        let path = dir.join("input.parquet");
        write_timestamp_extract_input_parquet(&path);

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
        assert_algeon_native_exec(&optimization.plan, &format!("{name} must rewrite to AlgeonNativeExec"));

        let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
        reset_arrow_host_interop_observation_for_test();
        let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
        let metrics = below_native_admission(&optimization.plan)
            .metrics()
            .expect("AlgeonNativeExec should expose DataFusion metrics");
        assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0, "{name}");
        assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1, "{name}");

        assert_batches_equal(&cpu_batches, &native_batches);

        fs::remove_dir_all(dir).unwrap();
    }
}

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
/// few units in the last place. DataFusion evaluates LN/EXP with the host libm,
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

fn write_checked_arithmetic_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("id", DataType::Int64, true),
        Field::new("den", DataType::Int64, true),
        Field::new("small", DataType::Int16, true),
        Field::new("score", DataType::Float64, true),
        Field::new("wide", DataType::Float64, true),
        Field::new("narrow", DataType::Float32, true),
    ]));
    // Row 5 pairs a null dividend with a zero divisor: DataFusion never evaluates it.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(Int64Array::from(vec![Some(1), Some(21), Some(-41), Some(60), None, Some(99), Some(0), Some(-7)])),
            Arc::new(Int64Array::from(vec![Some(3), Some(-2), Some(5), None, Some(0), Some(-1), Some(4), Some(2)])),
            Arc::new(Int16Array::from(vec![
                Some(1),
                Some(-2),
                None,
                Some(100),
                Some(127),
                Some(-128),
                Some(0),
                Some(5),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                Some(-2.5),
                None,
                Some(3.99),
                Some(-0.5),
                Some(1000.0),
                Some(-127.9),
                Some(0.0),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
                Some(-0.0),
                Some(2.5),
                None,
                Some(1e300),
                Some(-1.5),
            ])),
            Arc::new(Float32Array::from(vec![
                Some(f32::NAN),
                Some(f32::INFINITY),
                Some(-0.0),
                Some(2.5),
                Some(-2.5),
                None,
                Some(1e30),
                Some(-1.5),
            ])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn assert_native_matches_cpu(path: &std::path::Path, name: &str, sql: &str) {
    assert_native_matches_cpu_of(path, name, sql, sql).await;
}

/// Runs `native_sql` natively and `cpu_sql` on DataFusion, and compares the results.
async fn assert_native_matches_cpu_of(path: &std::path::Path, name: &str, native_sql: &str, cpu_sql: &str) {
    let cpu_batches = cpu_sql_output(path, cpu_sql).await;
    let plan = sql_plan(path, SessionConfig::new(), native_sql).await;
    let plan_display = displayable(plan.as_ref()).indent(true).to_string();
    let optimization = optimize_with_pruning_template(plan);
    assert_eq!(
        optimization.report.selected_count(),
        1,
        "{name} outcomes={:?}\n{plan_display}",
        optimization.report.outcomes()
    );
    assert_algeon_native_exec(&optimization.plan, "the query must rewrite to AlgeonNativeExec");
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);
}

async fn native_runtime_checked_casts_division_floor_and_ceil_match_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-checked-arithmetic");
    let path = dir.join("input.parquet");
    write_checked_arithmetic_input_parquet(&path);

    for (name, sql) in [
        (
            "integer-casts",
            "SELECT row_id, CAST(id AS INT) AS id_i32, CAST(small AS TINYINT) AS small_i8, \
             CAST(row_id AS BIGINT UNSIGNED) AS row_u64, CAST(score AS INT) AS score_i32 \
             FROM bench ORDER BY row_id",
        ),
        ("integer-to-string", "SELECT row_id, CAST(id AS VARCHAR) AS id_text FROM bench ORDER BY row_id"),
        (
            "window-rank-cast",
            "SELECT row_id, CAST(row_number() OVER (ORDER BY row_id) AS BIGINT) AS rank_i64 \
             FROM bench ORDER BY row_id",
        ),
        (
            "integer-division",
            "SELECT row_id, id / den AS quotient, id % den AS remainder, id / 20 AS quotient_20, \
             id % 20 AS remainder_20 FROM bench ORDER BY row_id",
        ),
        (
            "floor-ceil",
            "SELECT row_id, FLOOR(wide) AS floor_wide, CEIL(wide) AS ceil_wide, FLOOR(narrow) AS floor_narrow, \
             CEIL(narrow) AS ceil_narrow, FLOOR(id) AS floor_id, CEIL(score) AS ceil_score FROM bench \
             ORDER BY row_id",
        ),
        (
            "float-remainder",
            "SELECT row_id, wide % 3.0 AS wide_rem, narrow % CAST(3.0 AS FLOAT) AS narrow_rem, \
             score % CAST(den AS DOUBLE) AS score_rem FROM bench ORDER BY row_id",
        ),
        (
            "literal-divisor-in-case",
            "SELECT row_id, CASE WHEN den <> 0 THEN id / 20 ELSE 0 END AS guarded FROM bench ORDER BY row_id",
        ),
    ] {
        assert_native_matches_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_checked_failures_fail_the_query_like_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-checked-failures");
    let path = dir.join("input.parquet");
    write_checked_arithmetic_input_parquet(&path);

    for (name, sql) in [
        ("division-by-zero", "SELECT row_id, id / (den - den) AS quotient FROM bench ORDER BY row_id"),
        ("remainder-by-zero", "SELECT row_id, id % (den - den) AS remainder FROM bench ORDER BY row_id"),
        ("integer-cast-out-of-range", "SELECT row_id, CAST(id * 1000000000000 AS INT) AS id_i32 FROM bench"),
        ("float-cast-out-of-range", "SELECT row_id, CAST(wide AS BIGINT) AS wide_i64 FROM bench"),
    ] {
        assert_native_fails_like_cpu(&path, name, sql).await;
    }

    let invalid_path = dir.join("invalid-float.parquet");
    for (name, value) in
        [("float-cast-nan", f64::NAN), ("float-cast-infinity", f64::INFINITY), ("float-cast-huge", 1e40)]
    {
        let schema = Arc::new(Schema::new(vec![Field::new("wide", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![value]))]).unwrap();
        let mut writer = ArrowWriter::try_new(File::create(&invalid_path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        assert_native_fails_like_cpu(&invalid_path, name, "SELECT CAST(wide AS BIGINT) AS value FROM bench").await;
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn assert_native_fails_like_cpu(path: &std::path::Path, name: &str, sql: &str) {
    let cpu_plan = sql_plan(path, SessionConfig::new(), sql).await;
    let cpu_error = collect(cpu_plan, Arc::new(TaskContext::default())).await.expect_err(name);

    let optimization = optimize_with_pruning_template(sql_plan(path, SessionConfig::new(), sql).await);
    assert_eq!(optimization.report.selected_count(), 1, "{name} outcomes={:?}", optimization.report.outcomes());
    let native_error = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.expect_err(name);
    let algeon_error = native_error.find_algeon_error().expect("native failure keeps its typed Algeon error");
    assert_eq!(
        algeon_error.code(),
        ErrorCode::CudfInvalidInput,
        "{name}: native error {native_error} (cpu: {cpu_error})"
    );
    if matches!(name, "float-cast-nan" | "float-cast-infinity" | "float-cast-huge") {
        assert_eq!(algeon_error.fact_value("cudf_code"), Some("ValueOutOfRange"));
    }
}

fn write_case_branch_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("num", DataType::Int64, true),
        Field::new("den", DataType::Int64, true),
        Field::new("ok", DataType::Boolean, true),
        Field::new("big", DataType::Int64, true),
        Field::new("gate", DataType::Int64, true),
    ]));
    // Row 2 pairs a valid dividend with a zero divisor, and rows 2, 5 and 6 carry a value outside the Int32
    // range. Row 3 has a zero divisor and a zero `gate` under a null dividend, which DataFusion never
    // evaluates. Rows 3, 5 and 8 have a NULL `ok`, so a WHEN on it is NULL and the row falls through.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(Int64Array::from(vec![Some(10), Some(7), None, Some(60), Some(-9), Some(0), Some(20), Some(-41)])),
            Arc::new(Int64Array::from(vec![Some(2), Some(0), Some(0), None, Some(3), Some(-1), Some(5), Some(-2)])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(true),
                None,
                Some(false),
                None,
                Some(false),
                Some(true),
                None,
            ])),
            Arc::new(Int64Array::from(vec![
                Some(5),
                Some(5_000_000_000),
                None,
                Some(-7),
                Some(2_147_483_648),
                Some(-3_000_000_000),
                Some(0),
                Some(2_147_483_647),
            ])),
            Arc::new(Int64Array::from(vec![Some(4), Some(5), Some(0), Some(6), Some(-2), Some(7), Some(5), Some(1)])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn native_runtime_case_branches_that_can_fail_match_cpu() {
    // DataFusion evaluates a THEN or ELSE value, and every WHEN after the first, only on the rows that reach
    // it. Each query below puts a failing row where DataFusion never evaluates the fallible operation.
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-case-branches");
    let path = dir.join("input.parquet");
    write_case_branch_input_parquet(&path);

    for (name, sql) in [
        (
            "divisor-guarded-by-when",
            "SELECT row_id, CASE WHEN den <> 0 THEN num / den END AS v FROM bench ORDER BY row_id",
        ),
        (
            "remainder-guarded-by-when",
            "SELECT row_id, CASE WHEN den <> 0 THEN num % den ELSE -1 END AS v FROM bench ORDER BY row_id",
        ),
        (
            "narrowing-cast-guarded-by-when",
            "SELECT row_id, CASE WHEN big BETWEEN -2147483648 AND 2147483647 THEN CAST(big AS INT) ELSE 0 END AS v \
             FROM bench ORDER BY row_id",
        ),
        (
            "divisor-in-else",
            "SELECT row_id, CASE WHEN den = 0 THEN 0 ELSE num / den END AS v FROM bench ORDER BY row_id",
        ),
        (
            "fallible-when-after-the-first",
            "SELECT row_id, CASE WHEN den = 0 THEN 0 WHEN num / den > 2 THEN 1 ELSE 2 END AS v \
             FROM bench ORDER BY row_id",
        ),
        (
            "null-when-falls-through-to-else",
            "SELECT row_id, CASE WHEN ok THEN 0 ELSE num / den END AS v FROM bench ORDER BY row_id",
        ),
        (
            "nested-case-divisor",
            "SELECT row_id, CASE WHEN num IS NOT NULL THEN CASE WHEN den <> 0 THEN num / den ELSE 0 END ELSE -1 END AS v \
             FROM bench ORDER BY row_id",
        ),
        (
            "nested-case-inner-condition-true-outside-the-outer-branch",
            "SELECT row_id, CASE WHEN den <> 0 THEN CASE WHEN ok THEN num / den ELSE 0 END ELSE -1 END AS v \
             FROM bench ORDER BY row_id",
        ),
        (
            "nested-case-fallible-first-when",
            "SELECT row_id, CASE WHEN den <> 0 THEN CASE WHEN num / den > 1 THEN 1 ELSE 0 END ELSE -1 END AS v \
             FROM bench ORDER BY row_id",
        ),
        (
            "base-expression-form",
            "SELECT row_id, CASE den WHEN 0 THEN 0 ELSE num / den END AS v FROM bench ORDER BY row_id",
        ),
        (
            "base-expression-fallible-when-value",
            "SELECT row_id, CASE num WHEN 100 / gate THEN 1 ELSE 0 END AS v FROM bench ORDER BY row_id",
        ),
    ] {
        assert_native_matches_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_case_branches_fail_the_query_like_cpu_when_an_active_row_fails() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-case-branch-failures");
    let path = dir.join("input.parquet");
    write_case_branch_input_parquet(&path);

    for (name, sql) in [
        ("active-zero-divisor", "SELECT row_id, CASE WHEN num > 5 THEN num / den END AS v FROM bench ORDER BY row_id"),
        (
            "active-cast-out-of-range",
            "SELECT row_id, CASE WHEN row_id >= 5 THEN CAST(big AS INT) END AS v FROM bench ORDER BY row_id",
        ),
        (
            "null-when-row-reaches-a-failing-else",
            "SELECT row_id, CASE WHEN ok THEN 0 ELSE num / (den - 3) END AS v FROM bench ORDER BY row_id",
        ),
        (
            "failing-when-after-the-first",
            "SELECT row_id, CASE WHEN row_id = 1 THEN 0 WHEN num / den > 1 THEN 1 ELSE 2 END AS v \
             FROM bench ORDER BY row_id",
        ),
        (
            "failing-nested-branch",
            "SELECT row_id, CASE WHEN num IS NOT NULL THEN CASE WHEN row_id > 1 THEN num / den END END AS v \
             FROM bench ORDER BY row_id",
        ),
    ] {
        assert_native_fails_like_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

fn write_decimal_cast_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(12, 4), true),
        Field::new("whole", DataType::Int64, true),
        Field::new("score", DataType::Float64, true),
    ]));
    // Row 7 is null; the large values of rows 6 and 8 overflow narrow target precisions.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(
                Decimal128Array::from(vec![
                    Some(1_249),
                    Some(1_250),
                    Some(-1_250),
                    Some(5),
                    Some(-5),
                    Some(123_456_789),
                    None,
                    Some(99_999_995),
                ])
                .with_precision_and_scale(12, 4)
                .unwrap(),
            ),
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(-12_345),
                Some(0),
                Some(999),
                Some(-999),
                Some(123_456_789),
                None,
                Some(99_999),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(0.125),
                Some(-0.125),
                Some(2.5),
                Some(-2.5),
                Some(1234.5678),
                Some(0.285),
                None,
                Some(99_999.995),
            ])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn native_runtime_decimal_casts_match_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-decimal-casts");
    let path = dir.join("input.parquet");
    write_decimal_cast_input_parquet(&path);

    for (name, sql) in [
        (
            "decimal-rescale",
            "SELECT row_id, CAST(amount AS DECIMAL(12, 2)) AS amount_2, CAST(amount AS DECIMAL(20, 8)) AS amount_8, \
             CAST(amount AS DECIMAL(12, 0)) AS amount_0 FROM bench ORDER BY row_id",
        ),
        (
            "integer-to-decimal",
            "SELECT row_id, CAST(whole AS DECIMAL(20, 2)) AS whole_2, CAST(whole AS DECIMAL(38, 0)) AS whole_0 \
             FROM bench ORDER BY row_id",
        ),
        (
            "float-to-decimal",
            "SELECT row_id, CAST(score AS DECIMAL(12, 2)) AS score_2, CAST(score AS DECIMAL(38, 0)) AS score_0, \
             CAST(score AS DECIMAL(15, 6)) AS score_6 FROM bench ORDER BY row_id",
        ),
    ] {
        assert_native_matches_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_decimal_casts_outside_the_target_precision_fail_the_query_like_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-decimal-cast-failures");
    let path = dir.join("input.parquet");
    write_decimal_cast_input_parquet(&path);

    for (name, sql) in [
        ("decimal-to-narrow-precision", "SELECT row_id, CAST(amount AS DECIMAL(5, 2)) AS narrow FROM bench"),
        ("integer-to-narrow-precision", "SELECT row_id, CAST(whole AS DECIMAL(5, 2)) AS narrow FROM bench"),
        ("float-to-narrow-precision", "SELECT row_id, CAST(score AS DECIMAL(5, 2)) AS narrow FROM bench"),
        // 9999.9995 truncates to 9999 but rounds to 10000, one digit more than precision 4 holds.
        (
            "decimal-rounds-into-extra-digit",
            "SELECT row_id, CAST(amount AS DECIMAL(4, 0)) AS narrow FROM bench WHERE row_id IN (1, 8)",
        ),
    ] {
        assert_native_fails_like_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_case_branch_with_a_fallible_decimal_cast_matches_cpu() {
    // DataFusion casts only the rows that reach the THEN branch, so the first query succeeds on the CPU even
    // though row 6 needs 9 digits. The second query reaches row 6 and fails on both engines.
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-decimal-case-guard");
    let path = dir.join("input.parquet");
    write_decimal_cast_input_parquet(&path);

    assert_native_matches_cpu(
        &path,
        "guarded-decimal-cast",
        "SELECT row_id, CASE WHEN row_id < 6 THEN CAST(amount AS DECIMAL(5, 2)) END AS guarded \
         FROM bench ORDER BY row_id",
    )
    .await;
    assert_native_fails_like_cpu(
        &path,
        "active-decimal-cast-overflow",
        "SELECT row_id, CASE WHEN row_id <= 6 THEN CAST(amount AS DECIMAL(5, 2)) END AS guarded \
         FROM bench ORDER BY row_id",
    )
    .await;

    fs::remove_dir_all(dir).unwrap();
}

fn write_logical_operand_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("num", DataType::Int64, true),
        Field::new("den", DataType::Int64, true),
        Field::new("big", DataType::Int64, true),
        Field::new("amount", DataType::Decimal128(12, 2), true),
        Field::new("rate", DataType::Decimal128(12, 2), true),
    ]));
    // The first ten rows have a zero divisor and a `big` outside the Int32 range. Only rows 11 and 12, and row 1
    // for `big`, have a left operand that is TRUE for `den <> 0`, `rate <> 0` and `row_id = 1`. That is at most a
    // fifth of the rows, the point at which DataFusion evaluates the right operand of an AND on the TRUE rows
    // alone, and at least a fifth of the rows are FALSE for `den = 0`, which does the same for an OR.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((1..=12).collect::<Vec<i32>>())),
            Arc::new(Int64Array::from(vec![10, 7, -9, 60, 5, 0, 20, -41, 33, 8, 21, -30])),
            Arc::new(Int64Array::from(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -5])),
            Arc::new(Int64Array::from(vec![
                5,
                5_000_000_000,
                6_000_000_000,
                -7_000_000_000,
                8_000_000_000,
                9_000_000_000,
                1,
                2,
                3,
                4,
                5,
                6,
            ])),
            Arc::new(
                Decimal128Array::from(vec![1_000, -2_500, 77, 0, 5, 5, 123_456, 9, 100, 250, 4_000, -600])
                    .with_precision_and_scale(12, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 400, -300])
                    .with_precision_and_scale(12, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn native_runtime_logical_operators_skip_the_right_operand_on_rows_the_left_operand_decides() {
    // DataFusion evaluates the right operand of AND on the rows where the left operand is TRUE, and of OR on the
    // rows where it is FALSE, when the other rows are in the majority and the left operand has no NULL. Each query
    // below puts a failing right operand on the other rows.
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-logical-operands");
    let path = dir.join("input.parquet");
    write_logical_operand_input_parquet(&path);

    for (name, sql) in [
        ("and-guard-in-filter", "SELECT row_id FROM bench WHERE den <> 0 AND num / den > 1 ORDER BY row_id"),
        ("or-guard-in-filter", "SELECT row_id FROM bench WHERE den = 0 OR num / den > 5 ORDER BY row_id"),
        (
            "guards-in-projection",
            "SELECT row_id, den <> 0 AND num / den > 1 AS and_guard, den = 0 OR num / den > 5 AS or_guard \
             FROM bench ORDER BY row_id",
        ),
        ("negated-guards", "SELECT row_id FROM bench WHERE NOT (den <> 0 AND num / den > 1) ORDER BY row_id"),
        ("remainder-guard", "SELECT row_id FROM bench WHERE den <> 0 AND num % den = 1 ORDER BY row_id"),
        ("narrowing-cast-guard", "SELECT row_id FROM bench WHERE row_id = 1 AND CAST(big AS INT) > 0 ORDER BY row_id"),
        (
            "narrowing-cast-guard-in-or",
            "SELECT row_id FROM bench WHERE row_id > 1 OR CAST(big AS INT) > 0 ORDER BY row_id",
        ),
        (
            "guard-in-case-condition",
            "SELECT row_id, CASE WHEN den <> 0 AND num / den > 1 THEN 1 ELSE 0 END AS hit FROM bench ORDER BY row_id",
        ),
        (
            "guard-in-case-branch",
            "SELECT row_id, CASE WHEN row_id <= 4 THEN den <> 0 AND num / den > 1 END AS v FROM bench \
             ORDER BY row_id",
        ),
        (
            "case-in-the-right-operand",
            "SELECT row_id FROM bench WHERE den <> 0 AND CASE WHEN num > 0 THEN num / den ELSE 0 END > 1 \
             ORDER BY row_id",
        ),
        (
            "second-conjunct-guarded-by-the-first",
            "SELECT row_id FROM bench WHERE den <> 0 AND num / den > 1 AND big / den > 0 ORDER BY row_id",
        ),
        ("decimal-guard-in-filter", "SELECT row_id FROM bench WHERE rate <> 0 AND amount / rate > 1 ORDER BY row_id"),
    ] {
        assert_native_matches_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

fn write_logical_null_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("ok", DataType::Boolean, true),
        Field::new("num", DataType::Int64, true),
        Field::new("den", DataType::Int64, true),
        Field::new("den_zero_where_ok_is_null", DataType::Int64, true),
        Field::new("den_zero_where_ok_is_true", DataType::Int64, true),
        Field::new("den_zero_where_ok_is_false", DataType::Int64, true),
        Field::new("den_zero_where_and_is_decided", DataType::Int64, true),
        Field::new("den_zero_where_or_is_decided", DataType::Int64, true),
    ]));
    // `ok` is TRUE on rows 1, 4 and 7, FALSE on rows 2 and 5, and NULL on rows 3, 6 and 8, so DataFusion never
    // skips the right operand of a logical operator over it. The `den_zero_where_*` columns each carry one or two
    // zero divisors on the rows named.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                None,
                Some(true),
                Some(false),
                None,
                Some(true),
                None,
            ])),
            Arc::new(Int64Array::from(vec![Some(10), Some(7), Some(9), Some(60), Some(-9), Some(0), None, Some(-41)])),
            Arc::new(Int64Array::from(vec![Some(2), Some(-3), Some(4), None, Some(5), Some(1), Some(-2), Some(7)])),
            Arc::new(Int64Array::from(vec![2, -3, 0, 4, 5, 1, -2, 7])),
            Arc::new(Int64Array::from(vec![0, -3, 4, 4, 5, 1, -2, 7])),
            Arc::new(Int64Array::from(vec![2, 0, 4, 4, 5, 1, -2, 7])),
            Arc::new(Int64Array::from(vec![2, 0, 4, 4, 0, 1, -2, 7])),
            Arc::new(Int64Array::from(vec![0, -3, 4, 0, 5, 1, -2, 7])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn scalar_expressions_cases_1() {
    testing::run_named_cases(&[
        ("native_runtime_logical_operators_with_a_null_left_operand_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_logical_operators_with_a_null_left_operand_match_cpu())
        }),
        ("native_runtime_logical_operators_fail_the_query_like_cpu_when_an_undecided_row_fails", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_logical_operators_fail_the_query_like_cpu_when_an_undecided_row_fails())
        }),
        ("native_runtime_logical_operators_never_fail_on_a_row_the_left_operand_decides", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_logical_operators_never_fail_on_a_row_the_left_operand_decides())
        }),
        ("native_runtime_decimal_division_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_decimal_division_matches_cpu())
        }),
        ("native_runtime_scalar_expressions_match_cpu_and_collect_required_columns", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_scalar_expressions_match_cpu_and_collect_required_columns())
        }),
        ("native_runtime_decimal_division_failures_fail_the_query_like_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_decimal_division_failures_fail_the_query_like_cpu())
        }),
    ]);
}

#[test]
fn scalar_expressions_cases_2() {
    testing::run_named_cases(&[
        ("native_runtime_timestamp_and_duration_counts_cast_to_bigint_like_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_timestamp_and_duration_counts_cast_to_bigint_like_cpu())
        }),
        ("native_runtime_timestamp_arithmetic_fails_like_cpu_only_on_rows_cpu_evaluates", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_timestamp_arithmetic_fails_like_cpu_only_on_rows_cpu_evaluates())
        }),
        ("native_runtime_nullable_scalar_expressions_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_nullable_scalar_expressions_match_cpu())
        }),
        ("native_runtime_temporal_unit_casts_truncate_and_fail_like_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_temporal_unit_casts_truncate_and_fail_like_cpu())
        }),
        ("native_runtime_negation_matches_wrapping_integers_and_ieee_float_bits", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_negation_matches_wrapping_integers_and_ieee_float_bits())
        }),
        ("native_runtime_string_length_units_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_string_length_units_match_cpu())
        }),
        ("native_runtime_timestamp_extract_fields_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_timestamp_extract_fields_match_cpu())
        }),
        ("native_runtime_ln_exp_match_cpu_ieee_results", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_ln_exp_match_cpu_ieee_results())
        }),
    ]);
}

#[test]
fn scalar_expressions_cases_3() {
    testing::run_named_cases(&[
        ("native_runtime_ilike_rejects_before_native_execution", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_ilike_rejects_before_native_execution())
        }),
        ("native_runtime_invalid_scalar_expressions_reject_before_native_execution", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_invalid_scalar_expressions_reject_before_native_execution())
        }),
        ("native_runtime_date_trunc_calendrical_grains_reject_before_native_execution", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_date_trunc_calendrical_grains_reject_before_native_execution())
        }),
        ("native_runtime_substring_max_start_length_one_is_in_contract", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_substring_max_start_length_one_is_in_contract())
        }),
        ("native_runtime_checked_casts_division_floor_and_ceil_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_checked_casts_division_floor_and_ceil_match_cpu())
        }),
        ("native_runtime_checked_failures_fail_the_query_like_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_checked_failures_fail_the_query_like_cpu())
        }),
        ("native_runtime_join_filters_with_single_input_expressions_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_join_filters_with_single_input_expressions_match_cpu())
        }),
    ]);
}

#[test]
fn scalar_expressions_cases_4() {
    testing::run_named_cases(&[
        ("native_runtime_case_branches_that_can_fail_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_case_branches_that_can_fail_match_cpu())
        }),
        ("native_runtime_case_branches_fail_the_query_like_cpu_when_an_active_row_fails", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_case_branches_fail_the_query_like_cpu_when_an_active_row_fails())
        }),
        ("native_runtime_decimal_casts_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_decimal_casts_match_cpu())
        }),
        ("native_runtime_decimal_casts_outside_the_target_precision_fail_the_query_like_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_decimal_casts_outside_the_target_precision_fail_the_query_like_cpu())
        }),
        ("native_runtime_case_branch_with_a_fallible_decimal_cast_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_case_branch_with_a_fallible_decimal_cast_matches_cpu())
        }),
        ("native_runtime_logical_operators_skip_the_right_operand_on_rows_the_left_operand_decides", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_logical_operators_skip_the_right_operand_on_rows_the_left_operand_decides())
        }),
    ]);
}

async fn native_runtime_logical_operators_with_a_null_left_operand_match_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-logical-null-operands");
    let path = dir.join("input.parquet");
    write_logical_null_input_parquet(&path);

    for (name, sql) in [
        (
            "kleene-values",
            "SELECT row_id, ok AND num / den > 1 AS and_v, ok OR num / den > 1 AS or_v, \
             NOT ok AND num / den > 1 AS not_and_v FROM bench ORDER BY row_id",
        ),
        ("null-left-in-and-filter", "SELECT row_id FROM bench WHERE ok AND num / den > 1 ORDER BY row_id"),
        ("null-left-in-or-filter", "SELECT row_id FROM bench WHERE ok OR num / den > 1 ORDER BY row_id"),
        (
            "null-left-in-case-branch",
            "SELECT row_id, CASE WHEN row_id > 2 THEN ok AND num / den > 1 ELSE false END AS v FROM bench \
             ORDER BY row_id",
        ),
    ] {
        assert_native_matches_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_logical_operators_fail_the_query_like_cpu_when_an_undecided_row_fails() {
    // A NULL left operand leaves the result open, so the right operand runs on that row like on a TRUE row of an
    // AND or a FALSE row of an OR.
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-logical-failures");
    let path = dir.join("input.parquet");
    write_logical_null_input_parquet(&path);

    for (name, sql) in [
        (
            "null-left-operand-of-and",
            "SELECT row_id, ok AND num / den_zero_where_ok_is_null > 1 AS v FROM bench ORDER BY row_id",
        ),
        (
            "null-left-operand-of-or",
            "SELECT row_id, ok OR num / den_zero_where_ok_is_null > 1 AS v FROM bench ORDER BY row_id",
        ),
        (
            "true-left-operand-of-and",
            "SELECT row_id, ok AND num / den_zero_where_ok_is_true > 1 AS v FROM bench ORDER BY row_id",
        ),
        (
            "false-left-operand-of-or",
            "SELECT row_id, ok OR num / den_zero_where_ok_is_false > 1 AS v FROM bench ORDER BY row_id",
        ),
    ] {
        assert_native_fails_like_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_logical_operators_never_fail_on_a_row_the_left_operand_decides() {
    // `ok` has NULLs, so DataFusion evaluates the right operand on every row of the batch and fails on the zero
    // divisors of the decided rows. Native evaluation leaves those rows to the left operand. The CPU side runs the
    // same logic through a CASE, which evaluates the right operand only on the undecided rows.
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-logical-decided-rows");
    let path = dir.join("input.parquet");
    write_logical_null_input_parquet(&path);

    assert_native_matches_cpu_of(
        &path,
        "and-decided-by-false",
        "SELECT row_id, CASE WHEN ok AND num / den_zero_where_and_is_decided > 1 THEN 1 ELSE 0 END AS v FROM bench \
         ORDER BY row_id",
        "SELECT row_id, CASE WHEN ok IS NULL OR ok THEN \
         CASE WHEN ok AND num / den_zero_where_and_is_decided > 1 THEN 1 ELSE 0 END ELSE 0 END AS v FROM bench \
         ORDER BY row_id",
    )
    .await;
    assert_native_matches_cpu_of(
        &path,
        "or-decided-by-true",
        "SELECT row_id, CASE WHEN ok OR num / den_zero_where_or_is_decided > 1 THEN 1 ELSE 0 END AS v FROM bench \
         ORDER BY row_id",
        "SELECT row_id, CASE WHEN ok IS NULL OR NOT ok THEN \
         CASE WHEN ok OR num / den_zero_where_or_is_decided > 1 THEN 1 ELSE 0 END ELSE 1 END AS v FROM bench \
         ORDER BY row_id",
    )
    .await;

    fs::remove_dir_all(dir).unwrap();
}

fn write_decimal_division_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("grp", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(12, 2), true),
        Field::new("rate", DataType::Decimal128(12, 2), true),
        Field::new("rate_zero", DataType::Decimal128(12, 2), true),
        Field::new("rate_null", DataType::Decimal128(12, 2), true),
        Field::new("rate_nonzero", DataType::Decimal128(12, 2), true),
        Field::new("big_ok", DataType::Decimal128(38, 2), true),
        Field::new("big_over", DataType::Decimal128(38, 2), true),
    ]));
    // `rate` is zero on rows 3 and 5, where `amount` is NULL, so DataFusion never divides there. `rate_zero` is
    // also zero on rows 2 and 7, which carry a valid `amount`. A `Decimal128(38, 2)` dividend by a
    // `Decimal128(12, 2)` divisor is multiplied by 10^6 before it is divided, which stays inside an i128 up to
    // `LIMIT`. `big_ok` holds `LIMIT` and its negation, and `big_over` holds `LIMIT + 1` on row 4, where
    // `rate_null` is NULL. Group 3 has no valid `amount`.
    const LIMIT: i128 = i128::MAX / 1_000_000;
    let decimal = |values: Vec<Option<i128>>, precision: u8| {
        Arc::new(Decimal128Array::from(values).with_precision_and_scale(precision, 2).unwrap())
    };
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3, 4, 4])),
            decimal(vec![Some(1_000), Some(-2_500), None, Some(77), None, None, Some(5), Some(123_456)], 12),
            decimal(vec![Some(250), Some(100), Some(0), Some(-3), Some(0), Some(7), Some(3), Some(9)], 12),
            decimal(vec![Some(250), Some(0), Some(0), Some(-3), Some(0), Some(7), Some(0), Some(5)], 12),
            decimal(vec![Some(250), Some(100), Some(5), None, Some(7), Some(7), Some(3), Some(9)], 12),
            decimal(vec![Some(250), Some(100), Some(5), Some(-3), Some(7), Some(7), Some(3), Some(9)], 12),
            decimal(vec![Some(LIMIT), Some(-LIMIT), Some(7), Some(0), Some(1), Some(2), Some(3), Some(4)], 38),
            decimal(vec![Some(10), Some(20), Some(30), Some(LIMIT + 1), Some(50), Some(60), Some(70), Some(80)], 38),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn native_runtime_decimal_division_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-decimal-division");
    let path = dir.join("input.parquet");
    write_decimal_division_input_parquet(&path);

    for (name, sql) in [
        (
            "decimal-quotient",
            "SELECT row_id, amount / rate AS q, amount / CAST(2.5 AS DECIMAL(2, 1)) AS half, \
             CAST(100 AS DECIMAL(5, 2)) / rate_nonzero AS inverse FROM bench \
             ORDER BY row_id",
        ),
        ("decimal-division-in-filter", "SELECT row_id FROM bench WHERE amount / rate_nonzero > 1 ORDER BY row_id"),
        (
            "case-guarded-zero-divisor",
            "SELECT row_id, CASE WHEN rate_zero <> 0 THEN amount / rate_zero END AS q FROM bench ORDER BY row_id",
        ),
        (
            "case-guarded-literal-divisor",
            "SELECT row_id, CASE WHEN row_id <> 4 THEN big_over / CAST(3 AS DECIMAL(3, 2)) END AS q FROM bench ORDER BY row_id",
        ),
        ("dividend-scaling-at-the-i128-limit", "SELECT row_id, big_ok / rate_nonzero AS q FROM bench ORDER BY row_id"),
        (
            "overflowing-dividend-under-a-null-divisor",
            "SELECT row_id, big_over / rate_null AS q FROM bench ORDER BY row_id",
        ),
        (
            "average-over-groups-with-a-null-sum",
            "SELECT grp, AVG(amount) AS average FROM bench GROUP BY grp ORDER BY grp",
        ),
        ("average-over-no-rows", "SELECT AVG(amount) AS average FROM bench WHERE row_id < 0"),
    ] {
        assert_native_matches_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_decimal_join_residuals_match_cpu_without_host_boundaries() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-decimal-join-residual");
    let path = dir.join("input.parquet");
    write_decimal_division_input_parquet(&path);

    for (name, sql) in [
        (
            "cross-input-decimal-division",
            "SELECT l.row_id FROM bench l JOIN bench r ON l.row_id = r.row_id \
             AND l.amount / r.rate_nonzero > 1 ORDER BY l.row_id",
        ),
        (
            "case-guarded-input-ratios",
            "SELECT l.row_id FROM bench l JOIN bench r ON l.row_id = r.row_id \
             AND CASE WHEN l.rate_zero <> 0 THEN l.amount / l.rate_zero END \
                 > CASE WHEN r.rate_nonzero <> 0 THEN r.amount / r.rate_nonzero END \
             ORDER BY l.row_id",
        ),
        (
            "cross-input-case-guards-scaling-overflow",
            "SELECT l.row_id FROM bench l JOIN bench r ON l.row_id = r.row_id \
             AND CASE WHEN r.row_id <> 4 THEN l.big_over / r.rate_nonzero END > 0 \
             ORDER BY l.row_id",
        ),
        (
            "unmatched-overflowing-dividend",
            "SELECT l.row_id FROM bench l JOIN bench r ON l.row_id = r.row_id \
             AND r.row_id <> 4 AND l.big_over / r.rate_nonzero > 0 ORDER BY l.row_id",
        ),
    ] {
        let cpu = cpu_sql_output(&path, sql).await;
        assert!(cpu.iter().map(RecordBatch::num_rows).sum::<usize>() > 0, "{name} must exercise output rows");
        let plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let optimization = optimize_with_pruning_template_requiring_native(plan);
        assert_algeon_native_exec(&optimization.plan, name);
        let native = collect(optimization.plan, Arc::new(TaskContext::default())).await.unwrap();
        assert_batches_equal(&cpu, &native);
    }

    for (name, sql) in [
        (
            "active-zero-divisor",
            "SELECT l.row_id FROM bench l JOIN bench r ON l.row_id = r.row_id \
             AND l.amount / r.rate_zero > 0",
        ),
        (
            "active-scaling-overflow",
            "SELECT l.row_id FROM bench l JOIN bench r ON l.row_id = r.row_id \
             AND CASE WHEN r.row_id > 0 THEN l.big_over / r.rate_nonzero END > 0",
        ),
    ] {
        assert_native_fails_like_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

fn write_join_filter_string_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("amount", DataType::Int64, true),
    ]));
    // Each key holds two rows, so every row has one partner besides itself. Row 4 has no name and no amount.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3, 4, 4])),
            Arc::new(StringArray::from(vec![
                Some("Barbie"),
                Some("ken"),
                Some("barbie doll"),
                None,
                Some("Bar"),
                Some("alpha"),
                Some("Barbie"),
                Some("zed"),
            ])),
            Arc::new(Int64Array::from(vec![Some(5), Some(60), Some(7), None, Some(70), Some(-1), Some(7), Some(100)])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Join filters that read both inputs stay on the join. Their single-input
/// string predicates and null-aware comparisons are computed on that input
/// before the join, and a cross-input null-aware comparison filters the matched
/// pairs of an inner join.
async fn native_runtime_join_filters_with_single_input_expressions_match_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-join-filter-strings");
    let path = dir.join("input.parquet");
    write_join_filter_string_input_parquet(&path);

    for (name, sql) in [
        (
            "inner-lower-like",
            "SELECT l.row_id, r.row_id AS r_id FROM bench l JOIN bench r ON l.key = r.key \
             AND (LOWER(r.name) LIKE '%bar%' OR l.amount > 50) ORDER BY l.row_id, r_id",
        ),
        (
            "left-like",
            "SELECT l.row_id, r.row_id AS r_id FROM bench l LEFT JOIN bench r ON l.key = r.key \
             AND r.row_id <> l.row_id AND (r.name LIKE 'B%' OR l.amount > 50) ORDER BY l.row_id, r_id",
        ),
        (
            "full-like",
            "SELECT l.row_id, r.row_id AS r_id FROM bench l FULL JOIN bench r ON l.key = r.key \
             AND (r.name LIKE '%ie%' OR l.amount > 60) ORDER BY l.row_id, r_id",
        ),
        (
            "semi-not-like",
            "SELECT row_id FROM bench l WHERE EXISTS (SELECT 1 FROM bench r WHERE r.key = l.key \
             AND r.row_id <> l.row_id AND (r.name NOT LIKE '%a%' OR l.amount < 0)) ORDER BY row_id",
        ),
        (
            "anti-not-like",
            "SELECT row_id FROM bench l WHERE NOT EXISTS (SELECT 1 FROM bench r WHERE r.key = l.key \
             AND r.row_id <> l.row_id AND (r.name NOT LIKE '%a%' OR l.amount < 0)) ORDER BY row_id",
        ),
        (
            "single-input-concatenation-and-is-distinct-from",
            "SELECT l.row_id, r.row_id AS r_id FROM bench l JOIN bench r ON l.key = r.key \
             AND ((r.name || '!') = 'Barbie!' OR (l.amount IS DISTINCT FROM 7)) ORDER BY l.row_id, r_id",
        ),
        (
            "cross-input-is-not-distinct-from",
            "SELECT l.row_id, r.row_id AS r_id FROM bench l JOIN bench r ON l.key = r.key \
             AND l.row_id <> r.row_id AND ((l.name IS NOT DISTINCT FROM r.name) OR l.amount IS NULL) \
             ORDER BY l.row_id, r_id",
        ),
    ] {
        let cpu = cpu_sql_output(&path, sql).await;
        assert!(cpu.iter().map(RecordBatch::num_rows).sum::<usize>() > 0, "{name} must exercise output rows");
        let plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let optimization = optimize_with_pruning_template_requiring_native(plan);
        assert_algeon_native_exec(&optimization.plan, name);
        let native = collect(optimization.plan, Arc::new(TaskContext::default())).await.unwrap();
        assert_batches_equal(&cpu, &native);
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_decimal_division_failures_fail_the_query_like_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-decimal-division-failures");
    let path = dir.join("input.parquet");
    write_decimal_division_input_parquet(&path);

    for (name, sql) in [
        ("zero-divisor", "SELECT row_id, amount / (rate - rate) AS q FROM bench"),
        ("zero-divisor-column", "SELECT row_id, amount / rate_zero AS q FROM bench"),
        (
            "zero-divisor-in-an-active-case-branch",
            "SELECT row_id, CASE WHEN row_id >= 2 THEN amount / rate_zero END AS q FROM bench",
        ),
        ("dividend-scaling-overflow", "SELECT row_id, big_over / rate_nonzero AS q FROM bench"),
        (
            "dividend-scaling-overflow-in-an-active-case-branch",
            "SELECT row_id, CASE WHEN row_id >= 4 THEN big_over / CAST(3 AS DECIMAL(3, 2)) END AS q FROM bench",
        ),
    ] {
        assert_native_fails_like_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

fn write_event_time_input_parquet(path: &std::path::Path) {
    let timestamp = |unit| DataType::Timestamp(unit, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("user_id", DataType::Int32, false),
        Field::new("event_ms", timestamp(TimeUnit::Millisecond), false),
        Field::new("event_us", timestamp(TimeUnit::Microsecond), true),
        Field::new("event_ns", timestamp(TimeUnit::Nanosecond), true),
        Field::new("view_ms", timestamp(TimeUnit::Millisecond), true),
        Field::new("view_ns", timestamp(TimeUnit::Nanosecond), true),
    ]));
    // Each unit holds the same instant plus a sub-unit offset, so a cast that rescaled to a common unit
    // would change the count. Rows 2, 4, 6 and 8 carry NULLs, and row 6 views before it happens.
    let user_ids = vec![1, 2, 1, 3, 1, 2, 2, 1, 3];
    let events =
        [1_700_000_000_000, -86_400_000, 1_700_000_000_250, 0, 1_700_000_090_000, 3_600_000, 7_200_000, -1, 5_000];
    let view_offsets =
        [Some(1_500), Some(3_000_000), None, Some(0), Some(86_400_000), Some(-250), Some(1), None, Some(60_000)];
    let nullable_rows = |keep: [bool; 9], scale: i64| {
        events.iter().zip(keep).map(|(event, keep)| keep.then_some(event * scale + 7)).collect::<Vec<_>>()
    };
    let views = |scale: i64| {
        events
            .iter()
            .zip(view_offsets)
            .map(|(event, offset)| offset.map(|offset| (event + offset) * scale))
            .collect::<Vec<_>>()
    };
    let as_timestamp = |values: Vec<Option<i64>>, unit: TimeUnit| -> ArrayRef {
        cast(&Int64Array::from(values), &timestamp(unit)).unwrap()
    };
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((1..=9).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(user_ids)),
            as_timestamp(events.iter().map(|event| Some(*event)).collect(), TimeUnit::Millisecond),
            as_timestamp(
                nullable_rows([true, true, true, false, true, true, true, false, true], 1_000),
                TimeUnit::Microsecond,
            ),
            as_timestamp(
                nullable_rows([true, false, true, true, true, false, true, true, true], 1_000_000),
                TimeUnit::Nanosecond,
            ),
            as_timestamp(views(1), TimeUnit::Millisecond),
            as_timestamp(views(1_000_000), TimeUnit::Nanosecond),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn native_runtime_timestamp_and_duration_counts_cast_to_bigint_like_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-temporal-counts");
    let path = dir.join("input.parquet");
    write_event_time_input_parquet(&path);

    for (name, sql) in [
        (
            "stored-counts-per-unit",
            "SELECT row_id, CAST(event_ms AS BIGINT) AS ms, CAST(event_us AS BIGINT) AS us, \
             CAST(event_ns AS BIGINT) AS ns FROM bench ORDER BY row_id",
        ),
        (
            "elapsed-counts-per-unit",
            "SELECT row_id, CAST(view_ms - event_ms AS BIGINT) AS elapsed_ms, \
             CAST(view_ns - event_ns AS BIGINT) AS elapsed_ns FROM bench ORDER BY row_id",
        ),
        (
            "elapsed-count-across-units",
            "SELECT row_id, CAST(view_ns - event_ms AS BIGINT) AS elapsed_ns FROM bench ORDER BY row_id",
        ),
        (
            "gap-to-the-next-event",
            "SELECT row_id, CAST(LEAD(event_ms) OVER (PARTITION BY user_id ORDER BY event_ms) - event_ms AS BIGINT) \
             AS gap_ms FROM bench ORDER BY row_id",
        ),
        ("stored-count-filter", "SELECT row_id FROM bench WHERE CAST(event_ms AS BIGINT) > 1000000 ORDER BY row_id"),
        (
            "elapsed-count-filter",
            "SELECT row_id FROM bench WHERE CAST(view_ms - event_ms AS BIGINT) >= 1000 ORDER BY row_id",
        ),
        (
            "elapsed-count-aggregate",
            "SELECT user_id, SUM(CAST(view_ms - event_ms AS BIGINT)) AS total_ms FROM bench GROUP BY user_id \
             ORDER BY user_id",
        ),
    ] {
        assert_native_matches_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

fn write_timestamp_overflow_input_parquet(path: &std::path::Path) {
    let timestamp = DataType::Timestamp(TimeUnit::Nanosecond, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("ok", DataType::Boolean, false),
        Field::new("base", timestamp.clone(), false),
        Field::new("x", timestamp.clone(), true),
        Field::new("y", timestamp.clone(), true),
    ]));
    let timestamps = |values: Vec<Option<i64>>| -> ArrayRef { cast(&Int64Array::from(values), &timestamp).unwrap() };
    // Row 2 overflows in `base + (x - y)` and row 4 in `x - y`, both where `ok` is false, so a guarded CASE never
    // evaluates them. Row 3 would overflow in the addition under a NULL `y`. Row 5 lands exactly on the limit.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(BooleanArray::from(vec![true, false, true, false, true])),
            timestamps(vec![Some(10), Some(i64::MAX - 1), Some(i64::MAX - 1), Some(0), Some(i64::MAX - 1)]),
            timestamps(vec![Some(5), Some(5), Some(5), Some(i64::MAX), Some(1)]),
            timestamps(vec![Some(2), Some(0), None, Some(i64::MIN), Some(0)]),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn native_runtime_timestamp_arithmetic_fails_like_cpu_only_on_rows_cpu_evaluates() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-timestamp-overflow");
    let path = dir.join("input.parquet");
    write_timestamp_overflow_input_parquet(&path);

    // Rows 2 and 4 overflow where `ok` is false, row 3 carries a NULL `y`, and row 5 sits on the limit.
    assert_native_matches_cpu(
        &path,
        "guarded-subtraction-and-addition",
        "SELECT row_id, CASE WHEN ok THEN CAST(x - y AS BIGINT) END AS elapsed, \
         CASE WHEN ok THEN CAST(base + (x - y) AS BIGINT) END AS shifted FROM bench ORDER BY row_id",
    )
    .await;
    for (name, sql) in [
        (
            "subtraction-overflow",
            "SELECT row_id, CASE WHEN row_id <> 2 THEN CAST(x - y AS BIGINT) END AS elapsed FROM bench",
        ),
        (
            "addition-overflow",
            "SELECT row_id, CASE WHEN row_id <> 4 THEN CAST(base + (x - y) AS BIGINT) END AS shifted FROM bench",
        ),
    ] {
        assert_native_fails_like_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

fn write_temporal_unit_cast_input_parquet(path: &std::path::Path) {
    let timestamp = |unit| DataType::Timestamp(unit, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("ts_ms", timestamp(TimeUnit::Millisecond), true),
        Field::new("ts_ns", timestamp(TimeUnit::Nanosecond), true),
        Field::new("s_ms", timestamp(TimeUnit::Second), true),
        Field::new("ms_us", timestamp(TimeUnit::Millisecond), true),
        Field::new("ms_ns", timestamp(TimeUnit::Millisecond), true),
        Field::new("us_ns", timestamp(TimeUnit::Microsecond), true),
        Field::new("dur_s", DataType::Duration(TimeUnit::Second), true),
        Field::new("base_ms", timestamp(TimeUnit::Millisecond), true),
        Field::new("day", DataType::Date32, true),
    ]));
    // `ts_ms` and `ts_ns` carry negative counts that are not a multiple of the coarser unit, where truncation
    // toward zero and flooring differ. In every other column rows 1 to 6 hold the largest and smallest counts
    // whose product with the unit ratio fits Int64, then 0, -1, 1 and a NULL. Row 7 is the first count above
    // that limit and row 8 the first below it.
    let counts = |limit: i64| {
        vec![Some(limit), Some(-limit), Some(0), Some(-1), Some(1), None, Some(limit + 1), Some(-limit - 1)]
    };
    let typed = |values: Vec<Option<i64>>, data_type: DataType| -> ArrayRef {
        cast(&Int64Array::from(values), &data_type).unwrap()
    };
    let ts_ms = vec![Some(1_500), Some(-1_500), Some(-1), None, Some(-999), Some(0), Some(i64::MIN), Some(i64::MAX)];
    let ts_ns = vec![
        Some(1_500_000_001),
        Some(-1_500_000_001),
        Some(-1),
        None,
        Some(-999_999_999),
        Some(0),
        Some(i64::MIN),
        Some(i64::MAX),
    ];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((1..=8).collect::<Vec<_>>())),
            typed(ts_ms, timestamp(TimeUnit::Millisecond)),
            typed(ts_ns, timestamp(TimeUnit::Nanosecond)),
            typed(counts(i64::MAX / 1_000), timestamp(TimeUnit::Second)),
            typed(counts(i64::MAX / 1_000), timestamp(TimeUnit::Millisecond)),
            typed(counts(i64::MAX / 1_000_000), timestamp(TimeUnit::Millisecond)),
            typed(counts(i64::MAX / 1_000), timestamp(TimeUnit::Microsecond)),
            typed(counts(i64::MAX / 1_000), DataType::Duration(TimeUnit::Second)),
            typed(vec![Some(0); 8], timestamp(TimeUnit::Millisecond)),
            Arc::new(Date32Array::from(vec![
                Some(106_751),
                Some(-106_751),
                Some(0),
                Some(-1),
                Some(1),
                None,
                Some(106_752),
                Some(-106_752),
            ])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn native_runtime_temporal_unit_casts_truncate_and_fail_like_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-temporal-unit-casts");
    let path = dir.join("input.parquet");
    write_temporal_unit_cast_input_parquet(&path);

    for (name, sql) in [
        (
            "coarser-unit-truncates-toward-zero",
            "SELECT row_id, CAST(CAST(ts_ms AS TIMESTAMP(0)) AS BIGINT) AS ms_s, \
             CAST(CAST(ts_ns AS TIMESTAMP(0)) AS BIGINT) AS ns_s, CAST(CAST(ts_ns AS TIMESTAMP(3)) AS BIGINT) AS ns_ms, \
             CAST(CAST(ts_ns AS TIMESTAMP(6)) AS BIGINT) AS ns_us FROM bench ORDER BY row_id",
        ),
        (
            "date-of-a-timestamp-floors",
            "SELECT row_id, CAST(CAST(CAST(ts_ns AS DATE) AS TIMESTAMP(3)) AS BIGINT) AS day_ms FROM bench \
             ORDER BY row_id",
        ),
        (
            "finer-unit-within-int64",
            "SELECT row_id, CASE WHEN row_id <= 6 THEN CAST(s_ms AS BIGINT) END AS s_count, \
             CASE WHEN row_id <= 6 THEN CAST(CAST(s_ms AS TIMESTAMP(3)) AS BIGINT) END AS s_ms, \
             CASE WHEN row_id <= 6 THEN CAST(CAST(ms_us AS TIMESTAMP(6)) AS BIGINT) END AS ms_us, \
             CASE WHEN row_id <= 6 THEN CAST(CAST(ms_ns AS TIMESTAMP(9)) AS BIGINT) END AS ms_ns, \
             CASE WHEN row_id <= 6 THEN CAST(CAST(us_ns AS TIMESTAMP(9)) AS BIGINT) END AS us_ns, \
             CASE WHEN row_id <= 6 THEN CAST(CAST(day AS TIMESTAMP(9)) AS BIGINT) END AS day_ns, \
             CASE WHEN row_id <= 6 THEN CAST(base_ms + dur_s AS BIGINT) END AS shifted, \
             CASE WHEN row_id <= 6 THEN CAST(us_ns - ms_ns AS BIGINT) END AS elapsed_us FROM bench ORDER BY row_id",
        ),
    ] {
        assert_native_matches_cpu(&path, name, sql).await;
    }

    for (name, sql) in [
        (
            "timestamp-above-the-limit",
            "SELECT row_id, CASE WHEN row_id <> 8 THEN CAST(CAST(ms_us AS TIMESTAMP(6)) AS BIGINT) END AS us FROM bench",
        ),
        (
            "timestamp-below-the-limit",
            "SELECT row_id, CASE WHEN row_id <> 7 THEN CAST(CAST(ms_us AS TIMESTAMP(6)) AS BIGINT) END AS us FROM bench",
        ),
        ("seconds-to-milliseconds", "SELECT row_id, CAST(CAST(s_ms AS TIMESTAMP(3)) AS BIGINT) AS ms FROM bench"),
        ("timestamp-to-nanoseconds", "SELECT row_id, CAST(CAST(ms_ns AS TIMESTAMP(9)) AS BIGINT) AS ns FROM bench"),
        ("coerced-timestamp-difference", "SELECT row_id, CAST(ts_ns - ms_ns AS BIGINT) AS elapsed_ns FROM bench"),
        ("coerced-duration-addition", "SELECT row_id, CAST(base_ms + dur_s AS BIGINT) AS shifted FROM bench"),
        ("date-to-nanoseconds", "SELECT row_id, CAST(CAST(day AS TIMESTAMP(9)) AS BIGINT) AS day_ns FROM bench"),
    ] {
        assert_native_fails_like_cpu(&path, name, sql).await;
    }

    fs::remove_dir_all(dir).unwrap();
}
