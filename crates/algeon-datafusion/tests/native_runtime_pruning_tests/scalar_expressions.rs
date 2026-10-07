use std::fs::{self, File};

use algeon_datafusion::{
    capability::CapabilityReason,
    error::{DataFusionErrorExt, ErrorCode},
    report::CandidateOutcome,
};
use arrow::{
    array::{
        Array, AsArray, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
        Int64Array,
    },
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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
