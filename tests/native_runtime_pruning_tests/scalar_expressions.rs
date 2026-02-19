use super::*;

#[tokio::test]
async fn native_runtime_phase2_scalar_expressions_match_cpu_and_collect_required_columns() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-phase2-scalars");
    let path = dir.join("input.parquet");
    write_phase2_scalar_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_PHASE2_SCALAR_EXPRESSIONS_QUERY).await;
    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_PHASE2_SCALAR_EXPRESSIONS_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.source_diagnostics[0].read_column_count, 3);
    assert_eq!(lowering.source_diagnostics[0].projected_output_column_count, 3);
    assert_eq!(lowering.source_diagnostics[0].pure_filter_column_count, 0);

    let context = bounded_execution_context();
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_phase2_nullable_scalar_expressions_match_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-phase2-nullable-scalars");
    let path = dir.join("input.parquet");
    write_nullable_phase2_scalar_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_PHASE2_NULLABLE_SCALAR_EXPRESSIONS_QUERY).await;
    let native_physical_plan =
        sql_plan(&path, SessionConfig::new(), SQL_PHASE2_NULLABLE_SCALAR_EXPRESSIONS_QUERY).await;
    let plan_display = displayable(native_physical_plan.as_ref()).indent(true).to_string();
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}\n{plan_display}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}\n{plan_display}", lowering.report.outcomes());
    assert_eq!(lowering.source_diagnostics[0].read_column_count, 3);
    assert_eq!(lowering.source_diagnostics[0].projected_output_column_count, 3);
    assert_eq!(lowering.source_diagnostics[0].pure_filter_column_count, 0);

    let context = bounded_execution_context();
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_phase2_null_semantics_match_cpu_by_expression_family() {
    let cases = [
        ("like", SQL_PHASE2_NULL_SEMANTICS_LIKE_QUERY),
        ("substring", SQL_PHASE2_NULL_SEMANTICS_SUBSTRING_QUERY),
        ("string-case", SQL_PHASE2_NULL_SEMANTICS_STRING_CASE_QUERY),
        ("in-list", SQL_PHASE2_NULL_SEMANTICS_IN_LIST_QUERY),
        ("case", SQL_PHASE2_NULL_SEMANTICS_CASE_QUERY),
        ("case-no-else", SQL_PHASE2_NULL_SEMANTICS_CASE_NO_ELSE_QUERY),
        ("extract", SQL_PHASE2_NULL_SEMANTICS_EXTRACT_QUERY),
        ("interval", SQL_PHASE2_NULL_SEMANTICS_INTERVAL_QUERY),
        ("modulo", SQL_PHASE2_NULL_SEMANTICS_MODULO_QUERY),
        ("bitwise", SQL_PHASE7_BITWISE_SCALAR_QUERY),
        ("is-null", SQL_PHASE2_NULL_SEMANTICS_IS_NULL_QUERY),
        ("round", SQL_PHASE5_ROUND_SCALAR_QUERY),
        ("abs", SQL_PHASE7_ABS_SCALAR_QUERY),
    ];

    for (name, sql) in cases {
        let dir = unique_temp_dir(format!("datafusion-cudf-native-runtime-phase2-null-semantics-{name}").as_str());
        let path = dir.join("input.parquet");
        write_nullable_phase2_scalar_input_parquet(&path);

        let cpu_batches = cpu_sql_output(&path, sql).await;
        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let plan_display = displayable(native_physical_plan.as_ref()).indent(true).to_string();
        let lowering = lower_physical_plan_to_native(native_physical_plan);
        assert_eq!(lowering.rewritten_count(), 1, "{name} outcomes={:?}\n{plan_display}", lowering.report.outcomes());
        assert_eq!(lowering.rejected_count(), 0, "{name} outcomes={:?}\n{plan_display}", lowering.report.outcomes());

        let context = bounded_execution_context();
        let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
        cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
        let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
        assert_eq!(result.metrics().boundary.host_to_device_uploads, 0, "{name}");
        assert_eq!(result.metrics().boundary.device_to_host_materializations, 1, "{name}");

        let QueryOutput::HostArrow(native_batch) = result.output() else {
            panic!("{name} native host sink should return a RecordBatch");
        };
        assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

        fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test]
async fn native_runtime_invalid_phase2_scalar_expressions_reject_before_native_execution() {
    for (name, sql) in [
        ("substring-zero-start", SQL_INVALID_SUBSTRING_ZERO_START_QUERY),
        ("substring-dynamic-start", SQL_INVALID_SUBSTRING_DYNAMIC_START_QUERY),
        ("substring-dynamic-length", SQL_INVALID_SUBSTRING_DYNAMIC_LENGTH_QUERY),
        ("substring-negative-length", SQL_INVALID_SUBSTRING_NEGATIVE_LENGTH_QUERY),
        ("substring-stop-overflow", SQL_INVALID_SUBSTRING_STOP_OVERFLOW_QUERY),
    ] {
        let dir = unique_temp_dir(format!("datafusion-cudf-native-runtime-invalid-{name}").as_str());
        let path = dir.join("input.parquet");
        write_phase2_scalar_input_parquet(&path);

        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let lowering = lower_physical_plan_to_native(native_physical_plan);

        assert_eq!(lowering.rewritten_count(), 0, "{name} should not rewrite: {:?}", lowering.report.outcomes());
        assert_eq!(lowering.rejected_count(), 1, "{name} should reject: {:?}", lowering.report.outcomes());
        assert!(
            matches!(
                lowering.report.outcomes()[0],
                datafusion_cudf::CandidateOutcome::Rejected {
                    reason: datafusion_cudf::FallbackReason::UnsupportedExpression,
                    ..
                }
            ),
            "{name} outcomes={:?}",
            lowering.report.outcomes()
        );

        fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test]
async fn native_runtime_substring_max_start_length_one_is_in_contract() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-substring-max-start-boundary");
    let path = dir.join("input.parquet");
    write_phase2_scalar_input_parquet(&path);

    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_SUBSTRING_MAX_START_LENGTH_ONE_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);

    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());

    fs::remove_dir_all(dir).unwrap();
}
