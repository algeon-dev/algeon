use super::*;

#[test]
fn decimal128_type_support_policy_accepts_valid_arrow_decimals() {
    let decimal_with_fractional_scale = DataType::Decimal128(38, 2);
    let decimal_with_negative_scale = DataType::Decimal128(20, -1);

    assert!(is_supported_arrow_type(&decimal_with_fractional_scale));
    assert_eq!(
        cudf_dtype_from_arrow_type(&decimal_with_fractional_scale),
        Some(CudfDataType::fixed_point(TypeId::Decimal128, -2))
    );
    assert!(is_supported_arrow_type(&decimal_with_negative_scale));
    assert_eq!(
        cudf_dtype_from_arrow_type(&decimal_with_negative_scale),
        Some(CudfDataType::fixed_point(TypeId::Decimal128, 1))
    );
}

#[test]
fn decimal128_type_support_policy_rejects_invalid_arrow_decimals() {
    let invalid_precision = DataType::Decimal128(39, 2);
    let invalid_scale = DataType::Decimal128(10, 11);

    assert!(!is_supported_arrow_type(&invalid_precision));
    assert_eq!(cudf_dtype_from_arrow_type(&invalid_precision), None);
    assert!(!is_supported_arrow_type(&invalid_scale));
    assert_eq!(cudf_dtype_from_arrow_type(&invalid_scale), None);
}

#[test]
fn timestamp_with_timezone_type_support_policy_is_carry_only() {
    let timestamp_tz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));

    assert!(is_supported_arrow_type(&timestamp_tz));
    assert_eq!(cudf_dtype_from_arrow_type(&timestamp_tz), Some(CudfDataType::new(TypeId::TimestampMicroseconds)));
    assert!(is_supported_arrow_implicit_cast(&timestamp_tz, &timestamp_tz));
    assert!(!is_supported_arrow_cast(&DataType::Timestamp(TimeUnit::Microsecond, None), &timestamp_tz));
    assert!(!is_supported_arrow_cast(&timestamp_tz, &DataType::Timestamp(TimeUnit::Microsecond, None)));
}

#[test]
fn list_string_type_support_policy_is_carry_only() {
    let list_utf8 = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let large_list_large_utf8 = DataType::LargeList(Arc::new(Field::new("element", DataType::LargeUtf8, true)));
    let list_utf8_view = DataType::List(Arc::new(Field::new("item", DataType::Utf8View, true)));
    let list_int = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));

    for data_type in [&list_utf8, &large_list_large_utf8, &list_utf8_view] {
        assert!(is_supported_arrow_type(data_type), "{data_type:?}");
        assert!(is_supported_list_carry_type(data_type), "{data_type:?}");
        assert_eq!(cudf_dtype_from_arrow_type(data_type), Some(CudfDataType::new(TypeId::List)));
        assert!(is_supported_arrow_implicit_cast(data_type, data_type), "{data_type:?}");
    }

    assert!(!is_supported_arrow_type(&list_int));
    assert!(!is_supported_list_carry_type(&list_int));
    assert_eq!(cudf_dtype_from_arrow_type(&list_int), None);
    assert!(!is_supported_arrow_cast(&list_utf8, &large_list_large_utf8));
}

#[test]
fn list_string_columns_are_rejected_as_native_operator_operands() {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tags", list_type.clone(), true),
    ]));

    let sort_error = native_validation_error_for_plan(|builder| {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        builder.add_sort(
            source,
            SortSpec::new(vec![SortExpr::new(Expr::column("tags"), false, false)]),
            Arc::clone(&schema),
        )
    });
    assert_eq!(sort_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(sort_error.fact_value("context"), Some("sort key"));
    assert_eq!(sort_error.fact_value("expected"), Some("expected scalar operator key"));

    let filter_error = native_validation_error_for_plan(|builder| {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        let predicate = Expr::binary(Expr::column("tags"), BinaryOp::Eq, Expr::column("tags"));
        builder.add_filter(source, predicate, Arc::clone(&schema))
    });
    assert_eq!(filter_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(filter_error.fact_value("reason"), Some("list_operator_unsupported"));

    let is_null_error = native_validation_error_for_plan(|builder| {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        builder.add_filter(source, Expr::is_null(Expr::column("tags")), Arc::clone(&schema))
    });
    assert_eq!(is_null_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(is_null_error.fact_value("context"), Some("IS NULL input"));
    assert_eq!(is_null_error.fact_value("expected"), Some("expected scalar operator value"));

    let is_not_null_error = native_validation_error_for_plan(|builder| {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        builder.add_filter(source, Expr::is_not_null(Expr::column("tags")), Arc::clone(&schema))
    });
    assert_eq!(is_not_null_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(is_not_null_error.fact_value("context"), Some("IS NULL input"));
    assert_eq!(is_not_null_error.fact_value("expected"), Some("expected scalar operator value"));
}

#[test]
fn list_string_columns_are_rejected_as_native_window_operands() {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("tags", list_type.clone(), true),
    ]));

    let aggregate_output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("tags", list_type.clone(), true),
        Field::new("tag_count", DataType::Int64, false),
    ]));
    let aggregate_error = native_validation_error_for_plan(|builder| {
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
        builder.add_window(
            source,
            WindowSpec::new(
                vec![Expr::column("key")],
                Vec::new(),
                vec![WindowFunction::aggregate(AggregateFunction::Count, Expr::column("tags"), "tag_count")],
            ),
            aggregate_output_schema,
        )
    });
    assert_eq!(aggregate_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(aggregate_error.fact_value("context"), Some("window aggregate input"));
    assert_eq!(aggregate_error.fact_value("expected"), Some("expected scalar operator value"));

    let lag_output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("tags", list_type.clone(), true),
        Field::new("prev_tags", list_type, true),
    ]));
    let lag_error = native_validation_error_for_plan(|builder| {
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
        builder.add_window(
            source,
            WindowSpec::new(
                vec![Expr::column("key")],
                vec![SortExpr::new(Expr::column("key"), false, false)],
                vec![WindowFunction::lag(Expr::column("tags"), 1, None, "prev_tags")],
            ),
            lag_output_schema,
        )
    });
    assert_eq!(lag_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(lag_error.fact_value("context"), Some("window LAG / LEAD input"));
    assert_eq!(lag_error.fact_value("expected"), Some("expected scalar operator value"));
}

#[test]
fn list_string_null_predicates_are_rejected_in_native_post_join_residuals() {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false), Field::new("tags", list_type, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));

    let error = native_validation_error_for_plan(|builder| {
        let left =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&input_schema))));
        let right =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&input_schema))));
        let spec = JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::is_null(Expr::input_column(0, "tags", "tags")),
            BinaryOp::Or,
            Expr::is_not_null(Expr::input_column(1, "tags", "tags")),
        ))
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")]);
        builder.add_join(left, right, spec, output_schema)
    });

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.fact_value("context"), Some("post-join residual IS NULL input"));
    assert_eq!(error.fact_value("expected"), Some("expected scalar operator value"));
}

fn native_validation_error_for_plan(build_root: impl FnOnce(&mut QueryPlanBuilder) -> NodeId) -> QueryError {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(runtime).unwrap();
    let mut builder = QueryPlanBuilder::new();
    let root = build_root(&mut builder);
    let plan = builder.finish_with_host_sink(root);

    match nexus_query_engine::execute_query_plan(&context, &plan) {
        Ok(_) => panic!("plan validation should reject before execution"),
        Err(error) => error,
    }
}

#[test]
fn decimal128_cast_support_uses_type_support_policy() {
    let valid_source = DataType::Decimal128(20, -1);
    let valid_target = DataType::Decimal128(38, 2);
    let float64_target = DataType::Float64;
    let float64_source = DataType::Float64;
    let int32_source = DataType::Int32;
    let int32_target = DataType::Int32;
    let scale_zero_decimal = DataType::Decimal128(21, 0);
    let fractional_decimal = DataType::Decimal128(21, 6);
    let invalid_decimal = DataType::Decimal128(39, 2);

    assert!(is_supported_arrow_cast(&valid_source, &valid_target));
    assert!(is_supported_arrow_cast(&valid_source, &float64_target));
    assert!(is_supported_arrow_cast(&float64_source, &valid_target));
    assert!(is_supported_arrow_cast(&int32_source, &valid_target));
    assert!(is_supported_arrow_cast(&scale_zero_decimal, &int32_target));
    assert!(is_supported_arrow_implicit_cast(&valid_source, &valid_target));
    assert!(!is_supported_arrow_implicit_cast(&valid_source, &float64_target));
    assert!(!is_supported_arrow_implicit_cast(&float64_source, &valid_target));
    assert!(!is_supported_arrow_implicit_cast(&scale_zero_decimal, &int32_target));
    assert!(!is_supported_arrow_cast(&fractional_decimal, &int32_target));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &invalid_decimal));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &valid_target));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &float64_target));
    assert!(!is_supported_arrow_implicit_cast(&invalid_decimal, &float64_target));
    assert!(!is_supported_arrow_cast(&valid_source, &invalid_decimal));
    assert!(!is_supported_arrow_cast(&float64_source, &invalid_decimal));
}

#[test]
fn float64_to_int64_cast_support_is_explicit_and_not_implicit() {
    let float64_source = DataType::Float64;
    let int64_target = DataType::Int64;
    let int32_target = DataType::Int32;
    let float32_source = DataType::Float32;

    assert!(is_supported_arrow_cast(&float64_source, &int64_target));
    assert!(!is_supported_arrow_implicit_cast(&float64_source, &int64_target));
    assert!(!is_supported_arrow_cast(&float64_source, &int32_target));
    assert!(!is_supported_arrow_cast(&float32_source, &int64_target));
}
