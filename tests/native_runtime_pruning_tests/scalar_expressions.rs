use super::*;

#[tokio::test]
async fn native_runtime_scalar_expressions_match_cpu_and_collect_required_columns() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-scalar_expression-scalars");
    let path = dir.join("input.parquet");
    write_scalar_expression_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_SCALAR_EXPRESSIONS_QUERY).await;
    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_SCALAR_EXPRESSIONS_QUERY).await;
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
async fn native_runtime_nullable_scalar_expressions_match_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-scalar_expression-nullable-scalars");
    let path = dir.join("input.parquet");
    write_nullable_scalar_expression_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_NULLABLE_SCALAR_EXPRESSIONS_QUERY).await;
    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_NULLABLE_SCALAR_EXPRESSIONS_QUERY).await;
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
        // Integer modulo now has a dedicated CPU-fallback test below.
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
            format!("datafusion-nexus-native-runtime-scalar_expression-null-semantics-{name}").as_str(),
        );
        let path = dir.join("input.parquet");
        write_nullable_scalar_expression_input_parquet(&path);

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
async fn native_runtime_ilike_rejects_before_native_execution() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-ilike-reject");
    let path = dir.join("input.parquet");
    write_nullable_scalar_expression_input_parquet(&path);

    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_NULL_SEMANTICS_ILIKE_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);

    assert_eq!(lowering.rewritten_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert!(
        matches!(
            lowering.report.outcomes()[0],
            datafusion_nexus::CandidateOutcome::Rejected {
                reason: datafusion_nexus::FallbackReason::UnsupportedExpression,
                ..
            }
        ),
        "outcomes={:?}",
        lowering.report.outcomes()
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_integer_modulo_rejects_before_native_execution() {
    // SQL integer modulo stays on CPU until GPU by-zero semantics match DataFusion.
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-integer-modulo-reject");
    let path = dir.join("input.parquet");
    write_nullable_scalar_expression_input_parquet(&path);

    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_NULL_SEMANTICS_MODULO_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);

    assert_eq!(lowering.rewritten_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert!(
        matches!(
            lowering.report.outcomes()[0],
            datafusion_nexus::CandidateOutcome::Rejected {
                reason: datafusion_nexus::FallbackReason::UnsupportedExpression,
                ..
            }
        ),
        "outcomes={:?}",
        lowering.report.outcomes()
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
        let dir = unique_temp_dir(format!("datafusion-nexus-native-runtime-invalid-{name}").as_str());
        let path = dir.join("input.parquet");
        write_scalar_expression_input_parquet(&path);

        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let lowering = lower_physical_plan_to_native(native_physical_plan);

        assert_eq!(lowering.rewritten_count(), 0, "{name} should not rewrite: {:?}", lowering.report.outcomes());
        assert_eq!(lowering.rejected_count(), 1, "{name} should reject: {:?}", lowering.report.outcomes());
        assert!(
            matches!(
                lowering.report.outcomes()[0],
                datafusion_nexus::CandidateOutcome::Rejected {
                    reason: datafusion_nexus::FallbackReason::UnsupportedExpression,
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
async fn native_runtime_date_trunc_calendrical_grains_reject_before_native_execution() {
    for (name, sql) in [
        ("date-trunc-month", SQL_DATE_TRUNC_MONTH_UNSUPPORTED_QUERY),
        ("date-trunc-year", SQL_DATE_TRUNC_YEAR_UNSUPPORTED_QUERY),
    ] {
        let dir = unique_temp_dir(format!("datafusion-nexus-native-runtime-invalid-{name}").as_str());
        let path = dir.join("input.parquet");
        write_nullable_scalar_expression_input_parquet(&path);

        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let lowering = lower_physical_plan_to_native(native_physical_plan);

        assert_eq!(lowering.rewritten_count(), 0, "{name} should not rewrite: {:?}", lowering.report.outcomes());
        assert_eq!(lowering.rejected_count(), 1, "{name} should reject: {:?}", lowering.report.outcomes());
        assert!(
            matches!(
                lowering.report.outcomes()[0],
                datafusion_nexus::CandidateOutcome::Rejected {
                    reason: datafusion_nexus::FallbackReason::UnsupportedExpression,
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
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-substring-max-start-boundary");
    let path = dir.join("input.parquet");
    write_scalar_expression_input_parquet(&path);

    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_SUBSTRING_MAX_START_LENGTH_ONE_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);

    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());

    fs::remove_dir_all(dir).unwrap();
}
