use super::*;

#[test]
fn explicit_numeric_cast_lowers_to_native_expr_and_reads_inner_column() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let filter_schema = scan_schema;
    let cast_payload = Arc::new(CastExpr::new(col("payload", &filter_schema).unwrap(), DataType::Int64, None));
    let x00 = binary(cast_payload, Operator::Plus, col("m00", &filter_schema).unwrap(), &filter_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_projected_x00(scan, predicate, x00);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let projection = native_projection_exprs(&result.plans[0])
        .into_iter()
        .find(|expr| matches!(expr, Expr::Alias { alias, .. } if alias == "x00"))
        .expect("native plan should include x00 projection");
    assert!(matches!(
        projection,
        Expr::Alias { expr, .. }
            if matches!(
                expr.as_ref(),
                Expr::Binary { left, .. }
                    if matches!(
                        left.as_ref(),
                        Expr::Cast { expr, target_type }
                            if matches!(expr.as_ref(), Expr::Column(column) if column.name() == "payload")
                                && *target_type == DataType::Int64
                    )
            )
    ));
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn try_cast_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-try-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let try_cast = Arc::new(TryCastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int64));
    let predicate = binary(try_cast, Operator::GtEq, lit(ScalarValue::Int64(Some(10))), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn in_list_with_null_literal_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-in-list-null");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let predicate = in_list(
        col("payload", &scan_schema).unwrap(),
        vec![lit(ScalarValue::Int32(Some(10))), lit(ScalarValue::Int32(None))],
        &false,
        scan_schema.as_ref(),
    )
    .unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn substring_unsupported_arity_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-substring-arity");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let bad_substring: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
        "substring",
        unicode::substr(),
        vec![col("label", &scan_schema).unwrap()],
        Arc::new(Field::new("bad_substring", DataType::Utf8, true)),
        Arc::new(ConfigOptions::new()),
    ));
    let plan = grouped_count_projection_plan(scan, bad_substring, "bad_substring");

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn non_numeric_cast_targets_are_rejected_during_native_planning() {
    for (name, target_type, literal) in [
        ("string", DataType::Utf8, ScalarValue::Utf8(Some("10".to_string()))),
        (
            "timestamp",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            ScalarValue::TimestampNanosecond(Some(10), None),
        ),
    ] {
        let dir = unique_temp_dir(format!("datafusion-cudf-native-lowering-{name}-cast").as_str());
        let path = dir.join("input.parquet");
        write_numeric_parquet_file(&path);
        let scan = parquet_scan(&path);
        let scan_schema = scan.schema();
        let cast = Arc::new(CastExpr::new(col("payload", &scan_schema).unwrap(), target_type, None));
        let predicate = binary(cast, Operator::Eq, lit(literal), &scan_schema).unwrap();
        let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

        let result = lower_physical_plan_to_native(plan);

        assert!(result.plans.is_empty(), "{name} cast should not lower");
        assert_eq!(result.rejected_count(), 1, "{name} cast outcomes={:?}", result.report.outcomes());
        assert!(
            matches!(
                result.report.outcomes()[0],
                CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
            ),
            "{name} cast outcomes={:?}",
            result.report.outcomes()
        );

        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn explicit_numeric_to_decimal_cast_lowers_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-numeric-decimal-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let cast = Arc::new(CastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Decimal128(10, 0), None));
    let predicate = binary(cast, Operator::Eq, lit(ScalarValue::Decimal128(Some(10), 10, 0)), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    let native = result.plans.first().expect("decimal cast should lower to a native plan").to_string();
    assert!(native.contains("CAST(payload AS Decimal128(10, 0))"), "{native}");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn lossy_numeric_cast_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-lossy-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let cast = Arc::new(CastExpr::new(col("m00", &scan_schema).unwrap(), DataType::Int32, None));
    let predicate = binary(cast, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_physical_plan_without_scan_predicate_lowers_to_native_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-rewrite");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_NO_FILTER_BENCHMARK_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("direct-friendly SQL no-filter plan should lower to native");
    };
    assert_eq!(candidate.shape, "SortExec(fetch)(ProjectionExec(AggregateExec(LocalParquetScan)))");
    assert_eq!(candidate.operator_count, 4);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "absent");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "m01".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn every_rejected_native_candidate_has_stable_rejection_code() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-rejection-codes");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let rejected_results = [
        lower_physical_plan_to_native(aggregate_sort_limit_plan(
            parquet_scan_with_partial_byte_range(&path),
            Operator::Plus,
        )),
        lower_physical_plan_to_native(aggregate_sort_limit_plan_without_filter(
            parquet_scan_with_pushed_down_predicate(&path),
            Operator::Plus,
        )),
    ];

    for result in rejected_results {
        assert_eq!(result.rejected_count(), 1);
        for outcome in result.report.outcomes() {
            let CandidateOutcome::Rejected { reason, .. } = outcome else {
                panic!("expected only rejected native candidates");
            };
            assert!(!reason.code().is_empty());
        }
    }

    fs::remove_dir_all(dir).unwrap();
}
