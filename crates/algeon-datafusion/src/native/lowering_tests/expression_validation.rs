use super::*;

#[test]
fn explicit_numeric_cast_lowers_to_native_expr_and_reads_inner_column() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-cast");
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

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
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
fn scan_only_partial_reader_predicate_keeps_native_residual_filter() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-partial-reader-residual");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let scan = parquet_scan_with_pushed_down_partial_reader_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    let reader_predicate = source
        .reader_predicate
        .as_ref()
        .expect("supported scan predicate conjunct should be pushed into the reader")
        .to_string();
    assert!(reader_predicate.contains("payload = 10"), "{reader_predicate}");
    assert!(!reader_predicate.contains("LIKE"), "{reader_predicate}");

    let filters = native_filter_exprs(&result.plans[0]);
    assert_eq!(filters.len(), 1, "unsupported scan predicate residual must remain as a native filter");
    let residual = filters[0].to_string();
    assert!(residual.contains("payload = 10"), "{residual}");
    assert!(residual.contains("LIKE"), "{residual}");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn decimal_division_lowers_with_declared_binary_output_type() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-decimal-divide");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let ratio = binary(
        col("amount", &scan_schema).unwrap(),
        Operator::Divide,
        col("amount", &scan_schema).unwrap(),
        &scan_schema,
    )
    .unwrap();
    let plan: Arc<dyn ExecutionPlan> =
        Arc::new(ProjectionExec::try_new([ProjectionExpr { expr: ratio, alias: "ratio".to_string() }], scan).unwrap());

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let projection = native_projection_exprs(&result.plans[0])
        .into_iter()
        .find(|expr| matches!(expr, Expr::Alias { alias, .. } if alias == "ratio"))
        .expect("native plan should include ratio projection");
    assert!(matches!(
        projection,
        Expr::Alias { expr, .. }
            if matches!(
                expr.as_ref(),
                Expr::Binary {
                    op: query_engine::expr::BinaryOp::Divide,
                    output_type: Some(DataType::Decimal128(_, _)),
                    ..
                }
            )
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_round_decimal_integral_cast_lowers_to_native_expr() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-round-decimal-integral-cast");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_ROUND_DECIMAL_INTEGRAL_CAST_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let projection = native_projection_exprs(&result.plans[0])
        .into_iter()
        .find(|expr| matches!(expr, Expr::Alias { alias, .. } if alias == "segment"))
        .expect("native plan should include segment projection");
    assert!(matches!(
        projection,
        Expr::Alias { expr, .. }
            if matches!(
                expr.as_ref(),
                Expr::Cast { expr, target_type }
                    if *target_type == DataType::Int32
                        && matches!(expr.as_ref(), Expr::Round { output_type: DataType::Decimal128(_, 0), .. })
            )
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_nullif_lowers_to_case_with_null_aware_comparison() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-nullif");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan =
        sql_plan(&path, direct_friendly_sql_session_config(), "SELECT NULLIF(payload, 10) AS maybe_payload FROM bench")
            .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let projection = native_projection_exprs(&result.plans[0])
        .into_iter()
        .find(|expr| matches!(expr, Expr::Alias { alias, .. } if alias == "maybe_payload"))
        .expect("native plan should include NULLIF projection");
    assert!(
        matches!(
        projection,
        Expr::Alias { expr, .. }
            if matches!(
                expr.as_ref(),
                Expr::Case { branches, else_expr }
                    if branches.len() == 1
                        && matches!(
                            &branches[0].when,
                            Expr::IsNotDistinctFrom { left, right }
                                if matches!(
                                    left.as_ref(),
                                    Expr::Cast { expr, target_type }
                                        if *target_type == DataType::Int64
                                            && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "payload")
                                )
                                    && matches!(right.as_ref(), Expr::Literal(LiteralValue::Int64(10)))
                        )
                        && matches!(&branches[0].then, Expr::Literal(LiteralValue::Null(DataType::Int64)))
                        && matches!(
                            else_expr.as_ref(),
                            Expr::Cast { expr, target_type }
                                if *target_type == DataType::Int64
                                    && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "payload")
                        )
            )
        ),
        "unexpected NULLIF projection: {projection:?}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_ilike_ascii_pattern_remains_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-ilike");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan =
        sql_plan(&path, direct_friendly_sql_session_config(), "SELECT label ILIKE 'A%' AS label_match FROM bench")
            .await;

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_ilike_non_ascii_pattern_remains_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-ilike-non-ascii");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan =
        sql_plan(&path, direct_friendly_sql_session_config(), "SELECT label ILIKE 'Å%' AS label_match FROM bench")
            .await;

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_trim_family_lowers_to_native_string_trim_expressions() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-trim-family");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT TRIM(label) AS label_trim, LTRIM(label) AS label_ltrim, RTRIM(label) AS label_rtrim, BTRIM(label, 'ab') AS label_btrim, BTRIM(label, '') AS label_btrim_empty FROM bench",
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let projections = native_projection_exprs(&result.plans[0]);
    assert!(projections.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, expr }
            if alias == "label_trim"
                && matches!(expr.as_ref(), Expr::Trim { trim_chars, .. } if trim_chars == " ")
    )));
    assert!(projections.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, expr }
            if alias == "label_ltrim"
                && matches!(expr.as_ref(), Expr::LTrim { trim_chars, .. } if trim_chars == " ")
    )));
    assert!(projections.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, expr }
            if alias == "label_rtrim"
                && matches!(expr.as_ref(), Expr::RTrim { trim_chars, .. } if trim_chars == " ")
    )));
    assert!(projections.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, expr }
            if alias == "label_btrim"
                && matches!(expr.as_ref(), Expr::Trim { trim_chars, .. } if trim_chars == "ab")
    )));
    assert!(projections.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, expr }
            if alias == "label_btrim_empty"
                && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "label")
    )));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn string_predicates_lower_to_native_string_predicate_expressions() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-string-predicates");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let string_predicate = |name: &str, fun: Arc<datafusion::logical_expr::ScalarUDF>, literal: &str| {
        Arc::new(ScalarFunctionExpr::new(
            name,
            fun,
            vec![col("label", &scan_schema).unwrap(), lit(ScalarValue::Utf8(Some(literal.to_string())))],
            Arc::new(Field::new(name, DataType::Boolean, true)),
            Arc::new(ConfigOptions::default()),
        )) as Arc<dyn PhysicalExpr>
    };
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr {
                    expr: string_predicate("contains", string::contains(), "a"),
                    alias: "label_contains".to_string(),
                },
                ProjectionExpr {
                    expr: string_predicate("starts_with", string::starts_with(), "a"),
                    alias: "label_starts".to_string(),
                },
                ProjectionExpr {
                    expr: string_predicate("ends_with", string::ends_with(), "z"),
                    alias: "label_ends".to_string(),
                },
            ],
            scan,
        )
        .unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let projections = native_projection_exprs(&result.plans[0]);
    assert!(projections.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, expr }
            if alias == "label_contains"
                && matches!(expr.as_ref(), Expr::StringContains { expr, pattern } if pattern == "a" && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "label"))
    )));
    assert!(projections.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, expr }
            if alias == "label_starts"
                && matches!(expr.as_ref(), Expr::StartsWith { expr, prefix } if prefix == "a" && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "label"))
    )));
    assert!(projections.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, expr }
            if alias == "label_ends"
                && matches!(expr.as_ref(), Expr::EndsWith { expr, suffix } if suffix == "z" && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "label"))
    )));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn date_trunc_plural_and_nanosecond_grains_reject_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-date-trunc-grains");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);

    for grain in ["milliseconds", "microseconds", "nanosecond"] {
        let scan = parquet_scan(&path);
        let scan_schema = scan.schema();
        let date_trunc: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "date_trunc",
            datetime::date_trunc(),
            vec![lit(ScalarValue::Utf8(Some(grain.to_string()))), col("event_ts", &scan_schema).unwrap()],
            Arc::new(Field::new("trunc", DataType::Timestamp(TimeUnit::Nanosecond, None), false)),
            Arc::new(ConfigOptions::default()),
        ));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new([ProjectionExpr { expr: date_trunc, alias: format!("trunc_{grain}") }], scan)
                .unwrap(),
        );

        let result = lower_physical_plan_to_native(plan);

        assert_eq!(result.selected_count(), 0, "{grain} outcomes={:?}", result.report.outcomes());
        assert_eq!(result.not_selected_count(), 1, "{grain} outcomes={:?}", result.report.outcomes());
        assert!(matches!(
            result.report.outcomes()[0],
            CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
        ));
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn date_trunc_output_unit_mismatch_rejects_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-date-trunc-output-unit");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);

    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let date_trunc: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
        "date_trunc",
        datetime::date_trunc(),
        vec![lit(ScalarValue::Utf8(Some("day".to_string()))), col("event_ts", &scan_schema).unwrap()],
        Arc::new(Field::new("trunc", DataType::Timestamp(TimeUnit::Millisecond, None), false)),
        Arc::new(ConfigOptions::default()),
    ));
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new([ProjectionExpr { expr: date_trunc, alias: "trunc_day".to_string() }], scan).unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn try_cast_from_non_string_source_is_rejected_during_native_planning() {
    // Native admission for TRY_CAST is currently limited to `STRING → integer`.
    // Wrapping a numeric column in TRY_CAST keeps the structured rejection so
    // the adapter falls back to CPU instead of treating it as a plain CAST.
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-try-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let try_cast = Arc::new(TryCastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int64));
    let predicate = binary(try_cast, Operator::GtEq, lit(ScalarValue::Int64(Some(10))), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn in_list_with_null_literal_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-in-list-null");
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
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn list_utf8_is_null_filter_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-list-is-null");
    let path = dir.join("input.parquet");
    write_list_utf8_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(IsNullExpr::new(col("tags", &scan_schema).unwrap()));
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());
    let filter_schema = filter.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: col("m00", &filter_schema).unwrap(), alias: "m00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn list_utf8_count_window_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-list-count-window");
    let path = dir.join("input.parquet");
    write_list_utf8_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT key, COUNT(tags) OVER (PARTITION BY key) AS tag_count FROM bench",
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn list_utf8_lag_window_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-list-lag-window");
    let path = dir.join("input.parquet");
    write_list_utf8_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT key, LAG(tags) OVER (PARTITION BY key ORDER BY key) AS prev_tags FROM bench",
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dynamic_filter_expression_lowers_as_current_inner_expression() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-dynamic-filter");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let payload = col("payload", &scan_schema).unwrap();
    let dynamic_filter: Arc<dyn PhysicalExpr> =
        Arc::new(DynamicFilterPhysicalExpr::new(vec![Arc::clone(&payload)], lit(ScalarValue::Boolean(Some(true)))));
    let payload_filter = binary(payload, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema).unwrap();
    let predicate = binary(dynamic_filter, Operator::And, payload_filter, &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn substring_unsupported_arity_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-substring-arity");
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
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn non_numeric_cast_targets_are_rejected_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-string-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let cast = Arc::new(CastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Utf8, None));
    let predicate = binary(cast, Operator::Eq, lit(ScalarValue::Utf8(Some("10".to_string()))), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty(), "string cast should not lower");
    assert_eq!(result.not_selected_count(), 1, "string cast outcomes={:?}", result.report.outcomes());
    assert!(
        matches!(
            result.report.outcomes()[0],
            CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedType, .. }
        ),
        "string cast outcomes={:?}",
        result.report.outcomes()
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn integer_to_timestamp_cast_lowers_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-timestamp-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let cast = Arc::new(CastExpr::new(
        col("payload", &scan_schema).unwrap(),
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        None,
    ));
    let predicate =
        binary(cast, Operator::Eq, lit(ScalarValue::TimestampNanosecond(Some(10), None)), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    let native = result.plans.first().expect("timestamp cast should lower to a native plan").to_string();
    assert!(native.contains("CAST(payload AS Timestamp(ns))"), "{native}");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn explicit_numeric_to_decimal_cast_lowers_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-numeric-decimal-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let cast = Arc::new(CastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Decimal128(10, 0), None));
    let predicate = binary(cast, Operator::Eq, lit(ScalarValue::Decimal128(Some(10), 10, 0)), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    let native = result.plans.first().expect("decimal cast should lower to a native plan").to_string();
    assert!(native.contains("CAST(payload AS Decimal128(10, 0))"), "{native}");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn lossy_numeric_cast_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-lossy-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let cast = Arc::new(CastExpr::new(col("m00", &scan_schema).unwrap(), DataType::Int32, None));
    let predicate = binary(cast, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_physical_plan_without_scan_predicate_lowers_to_native_plan() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-sql-rewrite");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_NO_FILTER_BENCHMARK_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
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
fn integer_division_is_rejected_during_native_planning() {
    // Integer division stays on CPU until libcudf exposes checked by-zero semantics.
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-integer-divide");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let ratio =
        binary(col("m00", &scan_schema).unwrap(), Operator::Divide, col("m01", &scan_schema).unwrap(), &scan_schema)
            .unwrap();
    let plan: Arc<dyn ExecutionPlan> =
        Arc::new(ProjectionExec::try_new([ProjectionExpr { expr: ratio, alias: "ratio".to_string() }], scan).unwrap());

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty(), "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn integer_modulo_is_rejected_during_native_planning() {
    // Integer modulo stays on CPU until libcudf exposes checked by-zero semantics.
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-integer-modulo");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let remainder =
        binary(col("m00", &scan_schema).unwrap(), Operator::Modulo, col("m01", &scan_schema).unwrap(), &scan_schema)
            .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new([ProjectionExpr { expr: remainder, alias: "remainder".to_string() }], scan).unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty(), "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn every_rejected_native_candidate_has_stable_rejection_code() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-rejection-codes");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let rejected_results =
        [lower_physical_plan_to_native(aggregate_sort_limit_plan(parquet_scan(&path), Operator::Modulo))];

    for result in rejected_results {
        assert_eq!(result.not_selected_count(), 1);
        for outcome in result.report.outcomes() {
            let CandidateOutcome::NotSupported { capability_reason, .. } = outcome else {
                panic!("expected only rejected native candidates");
            };
            assert!(!capability_reason.code().is_empty());
        }
    }

    fs::remove_dir_all(dir).unwrap();
}
