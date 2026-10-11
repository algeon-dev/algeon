use std::sync::Arc;

use arrow::datatypes::{Decimal32Type, Decimal64Type, UnionFields, UnionMode, validate_decimal_precision_and_scale};
use cudf::testing::run_named_cases;
use query_engine::{exec::preflight_query_plan, rapids_backend::rapids_backend};
use query_planning::{backend::BackendCapabilities, datatypes::is_narrow_decimal_carry_type, expr::DatePart};

use super::*;

fn decimal128_type_support_policy_accepts_valid_arrow_decimals() {
    let decimal_with_fractional_scale = DataType::Decimal128(38, 2);
    let decimal_with_negative_scale = DataType::Decimal128(20, -1);

    assert!(is_supported_arrow_type(&decimal_with_fractional_scale));
    assert_eq!(
        cudf_dtype_from_arrow_type(&decimal_with_fractional_scale).ok(),
        Some(CudfDataType::fixed_point(TypeId::Decimal128, -2).unwrap())
    );
    assert!(is_supported_arrow_type(&decimal_with_negative_scale));
    assert_eq!(
        cudf_dtype_from_arrow_type(&decimal_with_negative_scale).ok(),
        Some(CudfDataType::fixed_point(TypeId::Decimal128, 1).unwrap())
    );
}

#[test]
fn t_type_policy_cases() {
    run_named_cases(&[
        (
            "type_support::decimal128_type_support_policy_accepts_valid_arrow_decimals",
            decimal128_type_support_policy_accepts_valid_arrow_decimals,
        ),
        (
            "type_support::decimal128_type_support_policy_rejects_invalid_arrow_decimals",
            decimal128_type_support_policy_rejects_invalid_arrow_decimals,
        ),
        (
            "type_support::decimal32_and_decimal64_type_support_policy_accepts_valid_arrow_decimals",
            decimal32_and_decimal64_type_support_policy_accepts_valid_arrow_decimals,
        ),
        (
            "type_support::decimal32_and_decimal64_type_support_policy_rejects_invalid_arrow_decimals",
            decimal32_and_decimal64_type_support_policy_rejects_invalid_arrow_decimals,
        ),
        (
            "type_support::timestamp_with_timezone_type_support_policy_is_carry_only",
            timestamp_with_timezone_type_support_policy_is_carry_only,
        ),
        (
            "type_support::temporal_type_support_policy_accepts_date64_and_time_types",
            temporal_type_support_policy_accepts_date64_and_time_types,
        ),
    ]);
}

fn decimal128_type_support_policy_rejects_invalid_arrow_decimals() {
    let invalid_precision = DataType::Decimal128(39, 2);
    let invalid_scale = DataType::Decimal128(10, 11);

    assert!(!is_supported_arrow_type(&invalid_precision));
    assert!(cudf_dtype_from_arrow_type(&invalid_precision).is_err());
    assert!(!is_supported_arrow_type(&invalid_scale));
    assert!(cudf_dtype_from_arrow_type(&invalid_scale).is_err());
}

fn decimal32_and_decimal64_type_support_policy_accepts_valid_arrow_decimals() {
    let cases = [
        (DataType::Decimal32(9, 2), TypeId::Decimal32, -2),
        (DataType::Decimal32(9, -1), TypeId::Decimal32, 1),
        (DataType::Decimal64(18, 2), TypeId::Decimal64, -2),
        (DataType::Decimal64(18, -1), TypeId::Decimal64, 1),
    ];

    for (arrow_type, cudf_type_id, cudf_scale) in cases {
        assert!(is_supported_arrow_type(&arrow_type), "{arrow_type:?}");
        assert!(is_narrow_decimal_carry_type(&arrow_type), "{arrow_type:?}");
        assert_eq!(
            cudf_dtype_from_arrow_type(&arrow_type).ok(),
            Some(CudfDataType::fixed_point(cudf_type_id, cudf_scale).unwrap()),
            "{arrow_type:?}"
        );
        assert!(is_supported_arrow_implicit_cast(&arrow_type, &arrow_type, rapids_backend()), "{arrow_type:?}");
    }
}

fn decimal32_and_decimal64_type_support_policy_rejects_invalid_arrow_decimals() {
    assert!(validate_decimal_precision_and_scale::<Decimal32Type>(10, 2).is_err());
    let invalid_decimal32_precision = DataType::Decimal32(10, 2);
    assert!(!is_supported_arrow_type(&invalid_decimal32_precision));
    assert!(!is_narrow_decimal_carry_type(&invalid_decimal32_precision));
    assert!(cudf_dtype_from_arrow_type(&invalid_decimal32_precision).is_err());

    assert!(validate_decimal_precision_and_scale::<Decimal32Type>(9, 10).is_err());
    let invalid_decimal32_scale = DataType::Decimal32(9, 10);
    assert!(!is_supported_arrow_type(&invalid_decimal32_scale));
    assert!(!is_narrow_decimal_carry_type(&invalid_decimal32_scale));
    assert!(cudf_dtype_from_arrow_type(&invalid_decimal32_scale).is_err());

    assert!(validate_decimal_precision_and_scale::<Decimal64Type>(19, 2).is_err());
    let invalid_decimal64_precision = DataType::Decimal64(19, 2);
    assert!(!is_supported_arrow_type(&invalid_decimal64_precision));
    assert!(!is_narrow_decimal_carry_type(&invalid_decimal64_precision));
    assert!(cudf_dtype_from_arrow_type(&invalid_decimal64_precision).is_err());

    assert!(validate_decimal_precision_and_scale::<Decimal64Type>(18, 19).is_err());
    let invalid_decimal64_scale = DataType::Decimal64(18, 19);
    assert!(!is_supported_arrow_type(&invalid_decimal64_scale));
    assert!(!is_narrow_decimal_carry_type(&invalid_decimal64_scale));
    assert!(cudf_dtype_from_arrow_type(&invalid_decimal64_scale).is_err());
}

fn timestamp_with_timezone_type_support_policy_is_carry_only() {
    let timestamp_tz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));

    assert!(is_supported_arrow_type(&timestamp_tz));
    assert_eq!(cudf_dtype_from_arrow_type(&timestamp_tz).ok(), Some(CudfDataType::new(TypeId::TimestampMicroseconds)));
    assert!(is_supported_arrow_implicit_cast(&timestamp_tz, &timestamp_tz, rapids_backend()));
    assert!(!is_supported_arrow_cast(
        &DataType::Timestamp(TimeUnit::Microsecond, None),
        &timestamp_tz,
        rapids_backend()
    ));
    assert!(!is_supported_arrow_cast(
        &timestamp_tz,
        &DataType::Timestamp(TimeUnit::Microsecond, None),
        rapids_backend()
    ));
}

fn temporal_type_support_policy_accepts_date64_and_time_types() {
    let cases = [
        (DataType::Date64, TypeId::TimestampMilliseconds),
        (DataType::Time32(TimeUnit::Second), TypeId::DurationSeconds),
        (DataType::Time32(TimeUnit::Millisecond), TypeId::DurationMilliseconds),
        (DataType::Time64(TimeUnit::Microsecond), TypeId::DurationMicroseconds),
        (DataType::Time64(TimeUnit::Nanosecond), TypeId::DurationNanoseconds),
    ];

    for (arrow_type, cudf_type_id) in cases {
        assert!(is_supported_arrow_type(&arrow_type), "{arrow_type:?}");
        assert_eq!(cudf_dtype_from_arrow_type(&arrow_type).ok(), Some(CudfDataType::new(cudf_type_id)));
        assert!(is_supported_arrow_implicit_cast(&arrow_type, &arrow_type, rapids_backend()), "{arrow_type:?}");
    }
}

#[test]
fn basic_list_type_support_policy_is_exact_and_carry_only() {
    let supported_children = [
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Utf8View,
    ];
    for child in supported_children {
        for list in [
            DataType::List(Arc::new(Field::new("item", child.clone(), true))),
            DataType::LargeList(Arc::new(Field::new("element", child, false))),
        ] {
            assert!(is_supported_arrow_type(&list), "{list:?}");
            assert!(is_supported_list_carry_type(&list), "{list:?}");
            assert_eq!(cudf_dtype_from_arrow_type(&list).ok(), Some(CudfDataType::new(TypeId::List)));
            assert!(is_supported_arrow_implicit_cast(&list, &list, rapids_backend()), "{list:?}");
        }
    }

    let unsupported_children = [
        DataType::Float16,
        DataType::Decimal128(10, 2),
        DataType::Date32,
        DataType::Binary,
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        DataType::List(Arc::new(Field::new("nested", DataType::Int32, true))),
        DataType::Struct(Fields::empty()),
        DataType::Map(Arc::new(Field::new("entries", DataType::Struct(Fields::empty()), false)), false),
        DataType::Union(UnionFields::empty(), UnionMode::Sparse),
    ];
    for child in unsupported_children {
        for list in [
            DataType::List(Arc::new(Field::new("item", child.clone(), true))),
            DataType::LargeList(Arc::new(Field::new("element", child, false))),
        ] {
            assert!(!is_supported_arrow_type(&list), "{list:?}");
            assert!(!is_supported_list_carry_type(&list), "{list:?}");
            assert!(cudf_dtype_from_arrow_type(&list).is_err(), "{list:?}");
        }
    }

    for container in [
        DataType::ListView(Arc::new(Field::new("item", DataType::Int32, true))),
        DataType::LargeListView(Arc::new(Field::new("item", DataType::Int32, true))),
    ] {
        assert!(!is_supported_list_carry_type(&container), "{container:?}");
    }

    let list_utf8 = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let large_list_large_utf8 = DataType::LargeList(Arc::new(Field::new("element", DataType::LargeUtf8, true)));
    assert!(!is_supported_arrow_cast(&list_utf8, &large_list_large_utf8, rapids_backend()));
}

#[test]
fn fixed_size_list_type_support_policy_is_exact_and_carry_only() {
    for child in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
    ] {
        let list = DataType::FixedSizeList(Arc::new(Field::new("item", child, true)), 3);
        assert!(is_supported_arrow_type(&list), "{list:?}");
        assert!(is_supported_list_carry_type(&list), "{list:?}");
        assert_eq!(cudf_dtype_from_arrow_type(&list).ok(), Some(CudfDataType::new(TypeId::List)));
        assert!(is_supported_arrow_implicit_cast(&list, &list, rapids_backend()), "{list:?}");
    }

    for child in [
        DataType::Boolean,
        DataType::Float16,
        DataType::Utf8,
        DataType::Decimal128(10, 2),
        DataType::Date32,
        DataType::Binary,
        DataType::List(Arc::new(Field::new("nested", DataType::Int32, true))),
        DataType::Struct(Fields::empty()),
    ] {
        let list = DataType::FixedSizeList(Arc::new(Field::new("item", child, true)), 3);
        assert!(!is_supported_arrow_type(&list), "{list:?}");
        assert!(!is_supported_list_carry_type(&list), "{list:?}");
        assert!(cudf_dtype_from_arrow_type(&list).is_err(), "{list:?}");
    }

    for dimension in [0, -1] {
        let list = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), dimension);
        assert!(!is_supported_arrow_type(&list), "{list:?}");
        assert!(!is_supported_list_carry_type(&list), "{list:?}");
        assert!(cudf_dtype_from_arrow_type(&list).is_err(), "{list:?}");
    }
}

#[test]
fn list_string_columns_are_rejected_as_native_operator_operands() {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("tags", list_type, true)]));

    let sort_error = native_validation_error_for_plan(|builder| {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("tags"), false, false)])).unwrap()
    });
    assert_eq!(sort_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(sort_error.fact_value("context"), Some("sort key"));
    assert_eq!(sort_error.fact_value("expected"), Some("expected scalar operator key"));

    let filter_error = native_validation_error_for_plan(|builder| {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        let predicate = Expr::binary(Expr::column("tags"), BinaryOp::Eq, Expr::column("tags"));
        builder.add_filter(source, predicate).unwrap()
    });
    assert_eq!(filter_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(filter_error.fact_value("reason"), Some("list_operator_unsupported"));

    let is_null_error = native_validation_error_for_plan(|builder| {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        builder.add_filter(source, Expr::is_null(Expr::column("tags"))).unwrap()
    });
    assert_eq!(is_null_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(is_null_error.fact_value("context"), Some("IS NULL input"));
    assert_eq!(is_null_error.fact_value("expected"), Some("expected scalar operator value"));

    let is_not_null_error = native_validation_error_for_plan(|builder| {
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        builder.add_filter(source, Expr::is_not_null(Expr::column("tags"))).unwrap()
    });
    assert_eq!(is_not_null_error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(is_not_null_error.fact_value("context"), Some("IS NULL input"));
    assert_eq!(is_not_null_error.fact_value("expected"), Some("expected scalar operator value"));
}

#[test]
fn numeric_list_columns_are_rejected_as_native_operator_values_and_keys() {
    for list_type in [
        DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
        DataType::LargeList(Arc::new(Field::new("element", DataType::Float64, false))),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2),
    ] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("values", list_type, true),
        ]));

        let filter_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
            builder
                .add_filter(source, Expr::binary(Expr::column("values"), BinaryOp::Eq, Expr::column("values")))
                .unwrap()
        });
        assert_eq!(filter_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(filter_error.fact_value("reason"), Some("list_operator_unsupported"));

        let sort_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
            builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("values"), false, false)])).unwrap()
        });
        assert_eq!(sort_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(sort_error.fact_value("context"), Some("sort key"));
        assert_eq!(sort_error.fact_value("expected"), Some("expected scalar operator key"));
    }
}

#[test]
fn carry_only_columns_are_rejected_as_native_join_group_and_window_keys() {
    for carry_type in [
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 2),
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
    ] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("tags", carry_type.clone(), true),
        ]));

        let join_error = native_validation_error_for_plan(|builder| {
            let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
            let right =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
            builder
                .add_join(
                    left,
                    right,
                    JoinSpec::new(
                        JoinType::Inner,
                        vec![JoinKey::new(ColumnRef::input(0, "tags", "tags"), ColumnRef::input(1, "tags", "tags"))],
                    )
                    .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id")]),
                    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
                )
                .unwrap()
        });
        assert_eq!(join_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(join_error.fact_value("context"), Some("join left key"));

        let group_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
            builder
                .add_aggregate(
                    source,
                    AggregateSpec::new(vec![Expr::column("tags")], Vec::new()),
                    Arc::new(Schema::new(vec![Field::new("tags", carry_type.clone(), true)])),
                )
                .unwrap()
        });
        assert_eq!(group_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(group_error.fact_value("context"), Some("aggregate group key"));

        for (partition_by, order_by, expected_context) in [
            (vec![Expr::column("tags")], Vec::new(), "window partition key"),
            (vec![Expr::column("id")], vec![SortExpr::new(Expr::column("tags"), false, false)], "window order key"),
        ] {
            let window_error = native_validation_error_for_plan(|builder| {
                let source =
                    builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
                builder
                    .add_window(
                        source,
                        WindowSpec::new(
                            partition_by.clone(),
                            order_by.clone(),
                            vec![WindowFunction::aggregate(
                                AggregateFunction::Count,
                                Expr::column("id"),
                                Field::new("row_count", DataType::Int64, false),
                            )],
                        ),
                    )
                    .unwrap()
            });
            assert_eq!(window_error.code(), QueryErrorCode::UnsupportedType);
            assert_eq!(window_error.fact_value("context"), Some(expected_context));
        }
    }
}

#[test]
fn carry_only_columns_are_rejected_as_native_aggregate_and_window_operands() {
    for carry_type in [
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
    ] {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("tags", carry_type.clone(), true),
        ]));

        // A list is a valid global COUNT argument because the count reads only
        // top-level validity; narrow decimals keep the value contract.
        if is_narrow_decimal_carry_type(&carry_type) {
            let aggregate_error = native_validation_error_for_plan(|builder| {
                let source = builder
                    .add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
                builder
                    .add_aggregate(
                        source,
                        AggregateSpec::new(
                            Vec::new(),
                            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("tags"), "tag_count")],
                        ),
                        Arc::new(Schema::new(vec![Field::new("tag_count", DataType::Int64, false)])),
                    )
                    .unwrap()
            });
            assert_eq!(aggregate_error.code(), QueryErrorCode::UnsupportedType);
            assert_eq!(aggregate_error.fact_value("context"), Some("aggregate input"));
            assert_eq!(aggregate_error.fact_value("expected"), Some("expected scalar or list operator value"));
        }

        let window_aggregate_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
            builder
                .add_window(
                    source,
                    WindowSpec::new(
                        vec![Expr::column("key")],
                        Vec::new(),
                        vec![WindowFunction::aggregate(
                            AggregateFunction::Count,
                            Expr::column("tags"),
                            Field::new("tag_count", DataType::Int64, false),
                        )],
                    ),
                )
                .unwrap()
        });
        assert_eq!(window_aggregate_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(window_aggregate_error.fact_value("context"), Some("window aggregate input"));
        assert_eq!(window_aggregate_error.fact_value("expected"), Some("expected scalar operator value"));

        let lag_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
            builder
                .add_window(
                    source,
                    WindowSpec::new(
                        vec![Expr::column("key")],
                        vec![SortExpr::new(Expr::column("key"), false, false)],
                        vec![WindowFunction::lag(
                            Expr::column("tags"),
                            1,
                            None,
                            Field::new("prev_tags", carry_type.clone(), true),
                        )],
                    ),
                )
                .unwrap()
        });
        assert_eq!(lag_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(lag_error.fact_value("context"), Some("window LAG / LEAD input"));
        assert_eq!(lag_error.fact_value("expected"), Some("expected scalar operator value"));
    }
}

#[test]
fn supported_list_global_count_reads_only_top_level_validity() {
    let count_tags = || AggregateExpr::new(AggregateFunction::Count, Expr::column("tags"), "tag_count");
    let count_output = || Arc::new(Schema::new(vec![Field::new("tag_count", DataType::Int64, false)]));

    for list_type in [
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        DataType::LargeList(Arc::new(Field::new("item", DataType::Int64, true))),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2),
    ] {
        assert!(is_supported_list_carry_type(&list_type), "{list_type:?}");
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("tags", list_type.clone(), true),
        ]));

        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
        let aggregate =
            builder.add_aggregate(source, AggregateSpec::new(Vec::new(), vec![count_tags()]), count_output()).unwrap();
        preflight_query_plan(&builder.finish_with_host_sink(aggregate).unwrap())
            .unwrap_or_else(|error| panic!("global COUNT over {list_type:?} should preflight: {error}"));

        // Anything that reads values, splits by key or filters rows keeps the
        // scalar operator contract.
        let key_and_count_output = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("tag_count", DataType::Int64, false),
        ]));
        for (case, spec, output_schema) in [
            ("distinct", AggregateSpec::new(Vec::new(), vec![count_tags().with_distinct(true)]), count_output()),
            (
                "filtered",
                AggregateSpec::new(
                    Vec::new(),
                    vec![count_tags().with_filter(Some(Expr::is_not_null(Expr::column("key"))))],
                ),
                count_output(),
            ),
            (
                "sum",
                AggregateSpec::new(
                    Vec::new(),
                    vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("tags"), "tag_count")],
                ),
                count_output(),
            ),
            ("grouped", AggregateSpec::new(vec![Expr::column("key")], vec![count_tags()]), key_and_count_output),
        ] {
            let error = native_validation_error_for_plan(|builder| {
                let source = builder
                    .add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
                builder.add_aggregate(source, spec, output_schema).unwrap()
            });
            assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "{case} {list_type:?}");
            assert_eq!(error.fact_value("context"), Some("aggregate input"), "{case} {list_type:?}");
            assert_eq!(error.fact_value("expected"), Some("expected scalar operator key"), "{case} {list_type:?}");
        }
    }
}

#[test]
fn list_string_null_predicates_are_rejected_in_native_join_residuals() {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false), Field::new("tags", list_type, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let left_tags_null = || Expr::is_null(Expr::input_column(0, "tags", "tags"));

    // The mixed join evaluates null checks while matching; IS DISTINCT FROM
    // moves the whole residual to a filter after the equi-join.
    for (other_disjunct, context) in [
        (Expr::is_not_null(Expr::input_column(1, "tags", "tags")), "join residual IS NULL input"),
        (
            Expr::is_distinct_from(Expr::input_column(0, "key", "key"), Expr::input_column(1, "key", "key")),
            "post-join residual IS NULL input",
        ),
    ] {
        let error = native_validation_error_for_plan(|builder| {
            let left =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&input_schema))));
            let right =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&input_schema))));
            let spec = JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
            )
            .with_residual_predicate(Expr::binary(left_tags_null(), BinaryOp::Or, other_disjunct.clone()))
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")]);
            builder.add_join(left, right, spec, Arc::clone(&output_schema)).unwrap()
        });

        assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "{context}");
        assert_eq!(error.fact_value("context"), Some(context));
        assert_eq!(error.fact_value("expected"), Some("expected scalar operator value"), "{context}");
    }
}

#[test]
fn narrow_decimal_carry_types_are_rejected_in_operator_roles_with_structured_local_errors() {
    for amount_type in [DataType::Decimal32(9, 2), DataType::Decimal64(18, 2)] {
        assert!(is_narrow_decimal_carry_type(&amount_type), "{amount_type:?}");
        let amount_type_debug = format!("{amount_type:?}");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("amount", amount_type.clone(), true),
        ]));

        let comparison_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
            builder
                .add_filter(source, Expr::binary(Expr::column("amount"), BinaryOp::Eq, Expr::column("amount")))
                .unwrap()
        });
        assert_eq!(comparison_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(comparison_error.fact_value("reason"), Some("narrow_decimal_comparison_unsupported"));
        assert_eq!(comparison_error.fact_value("left_type"), Some(amount_type_debug.as_str()));
        assert_eq!(comparison_error.fact_value("right_type"), Some(amount_type_debug.as_str()));

        let cast_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
            let output_schema = Arc::new(Schema::new(vec![Field::new("amount", amount_type.clone(), true)]));
            builder
                .add_projection(
                    source,
                    projection_outputs(
                        vec![Expr::cast(Expr::column("amount"), amount_type.clone())],
                        Arc::clone(&output_schema),
                    ),
                )
                .unwrap()
        });
        assert_eq!(cast_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(cast_error.fact_value("source_type"), Some(amount_type_debug.as_str()));
        assert_eq!(cast_error.fact_value("target_type"), Some(amount_type_debug.as_str()));

        let sort_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
            builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("amount"), false, false)])).unwrap()
        });
        assert_eq!(sort_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(sort_error.fact_value("context"), Some("sort key"));
        assert_eq!(sort_error.fact_value("expected"), Some("expected scalar operator key"));
        assert_eq!(sort_error.fact_value("data_type"), Some(amount_type_debug.as_str()));

        let null_predicate_error = native_validation_error_for_plan(|builder| {
            let source =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
            builder.add_filter(source, Expr::is_null(Expr::column("amount"))).unwrap()
        });
        assert_eq!(null_predicate_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(null_predicate_error.fact_value("context"), Some("IS NULL input"));
        assert_eq!(null_predicate_error.fact_value("expected"), Some("expected scalar operator value"));
        assert_eq!(null_predicate_error.fact_value("data_type"), Some(amount_type_debug.as_str()));

        let join_error = native_validation_error_for_plan(|builder| {
            let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
            let right =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
            builder
                .add_join(
                    left,
                    right,
                    JoinSpec::new(
                        JoinType::Inner,
                        vec![JoinKey::new(
                            ColumnRef::input(0, "amount", "amount"),
                            ColumnRef::input(1, "amount", "amount"),
                        )],
                    )
                    .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id")]),
                    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
                )
                .unwrap()
        });
        assert_eq!(join_error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(join_error.fact_value("context"), Some("join left key"));
        assert_eq!(join_error.fact_value("expected"), Some("expected scalar operator key"));
        assert_eq!(join_error.fact_value("data_type"), Some(amount_type_debug.as_str()));
    }
}

#[test]
fn projection_preflight_rejects_unsupported_implicit_output_cast() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_key", DataType::Utf8, false)]));

    let error = native_validation_error_for_plan(|builder| {
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
        builder
            .add_projection(source, projection_outputs(vec![Expr::column("key").alias("bad_key")], output_schema))
            .unwrap()
    });

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.fact_value("reason"), Some("projection_output_schema_mismatch"));
    assert_eq!(error.fact_value("reason_detail"), Some("implicit_cast_not_supported"));
    assert_eq!(error.fact_value("field_name"), Some("bad_key"));
    assert_eq!(error.fact_value("field_index"), Some("0"));
    assert_eq!(error.fact_value("actual_type"), Some("Int32"));
    assert_eq!(error.fact_value("expected_type"), Some("Utf8"));
}

#[test]
fn extract_time_fields_admit_only_timezone_free_timestamps() {
    let preflight_extract = |input_type: DataType, field: DatePart| {
        let input_schema = Arc::new(Schema::new(vec![Field::new("event", input_type, true)]));
        let output_schema = Arc::new(Schema::new(vec![Field::new("part", DataType::Int32, true)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
        let projection = builder
            .add_projection(
                source,
                projection_outputs(vec![Expr::extract(field, Expr::column("event"))], output_schema),
            )
            .unwrap();
        preflight_query_plan(&builder.finish_with_host_sink(projection).unwrap())
    };

    for (input_type, field) in [
        (DataType::Timestamp(TimeUnit::Second, None), DatePart::Hour),
        (DataType::Timestamp(TimeUnit::Millisecond, None), DatePart::Second),
        (DataType::Timestamp(TimeUnit::Nanosecond, None), DatePart::Microsecond),
        (DataType::Date32, DatePart::Day),
    ] {
        preflight_extract(input_type.clone(), field)
            .unwrap_or_else(|error| panic!("EXTRACT {field} over {input_type:?} should preflight: {error}"));
    }

    for input_type in [DataType::Date32, DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()))] {
        let error = preflight_extract(input_type.clone(), DatePart::Second).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "{input_type:?}");
        assert_eq!(error.fact_value("context"), Some("EXTRACT time field input"), "{input_type:?}");
    }

    let error = preflight_extract(DataType::Timestamp(TimeUnit::Nanosecond, None), DatePart::Nanosecond).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.fact_value("field"), Some("NANOSECOND"));
}

#[test]
fn projection_preflight_accepts_equivalent_list_child_names() {
    let input_list = DataType::List(Arc::new(Field::new("element", DataType::Utf8, true)));
    let output_list = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let input_schema = Arc::new(Schema::new(vec![Field::new("tags", input_list, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("tags", output_list, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
    let projection =
        builder.add_projection(source, projection_outputs(vec![Expr::column("tags")], output_schema)).unwrap();

    preflight_query_plan(&builder.finish_with_host_sink(projection).unwrap())
        .expect("list child field names do not require a native cast");
}

fn native_validation_error_for_plan(build_root: impl FnOnce(&mut QueryPlanBuilder) -> NodeId) -> QueryError {
    let mut builder = QueryPlanBuilder::new();
    let root = build_root(&mut builder);
    let plan = builder.finish_with_host_sink(root).unwrap();
    preflight_query_plan(&plan).unwrap_err()
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

    assert!(is_supported_arrow_cast(&valid_source, &valid_target, rapids_backend()));
    assert!(is_supported_arrow_cast(&valid_source, &float64_target, rapids_backend()));
    assert!(is_supported_arrow_cast(&float64_source, &valid_target, rapids_backend()));
    assert!(is_supported_arrow_cast(&int32_source, &valid_target, rapids_backend()));
    assert!(is_supported_arrow_cast(&scale_zero_decimal, &int32_target, rapids_backend()));
    assert!(is_supported_arrow_implicit_cast(&valid_source, &valid_target, rapids_backend()));
    assert!(!is_supported_arrow_implicit_cast(&valid_source, &float64_target, rapids_backend()));
    assert!(!is_supported_arrow_implicit_cast(&float64_source, &valid_target, rapids_backend()));
    assert!(!is_supported_arrow_implicit_cast(&scale_zero_decimal, &int32_target, rapids_backend()));
    assert!(!is_supported_arrow_cast(&fractional_decimal, &int32_target, rapids_backend()));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &invalid_decimal, rapids_backend()));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &valid_target, rapids_backend()));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &float64_target, rapids_backend()));
    assert!(!is_supported_arrow_implicit_cast(&invalid_decimal, &float64_target, rapids_backend()));
    assert!(!is_supported_arrow_cast(&valid_source, &invalid_decimal, rapids_backend()));
    assert!(!is_supported_arrow_cast(&float64_source, &invalid_decimal, rapids_backend()));
}

#[test]
fn range_checked_integral_cast_support_is_explicit_and_not_implicit() {
    let integers = [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
    ];
    for source in &integers {
        for target in &integers {
            assert!(is_supported_arrow_cast(source, target, rapids_backend()), "{source:?} -> {target:?}");
        }
        for source_float in [DataType::Float32, DataType::Float64] {
            assert!(is_supported_arrow_cast(&source_float, source, rapids_backend()), "{source_float:?} -> {source:?}");
            assert!(!is_supported_arrow_implicit_cast(&source_float, source, rapids_backend()));
        }
        for text in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
            assert!(is_supported_arrow_cast(source, &text, rapids_backend()), "{source:?} -> {text:?}");
        }
    }
    assert!(!is_supported_arrow_implicit_cast(&DataType::Int64, &DataType::Int32, rapids_backend()));
    assert!(!is_supported_arrow_implicit_cast(&DataType::UInt64, &DataType::Int64, rapids_backend()));
    assert!(!is_supported_arrow_cast(&DataType::Float64, &DataType::Utf8, rapids_backend()));
    assert!(!is_supported_arrow_cast(&DataType::Boolean, &DataType::Utf8, rapids_backend()));
}

#[test]
fn temporal_count_casts_to_int64_are_explicit_and_cannot_fail() {
    use query_planning::datatypes::{cast_can_fail_at_runtime, is_supported_ast_cast};

    let units = [TimeUnit::Second, TimeUnit::Millisecond, TimeUnit::Microsecond, TimeUnit::Nanosecond];
    for unit in units {
        for source in [DataType::Timestamp(unit, None), DataType::Duration(unit)] {
            assert!(is_supported_arrow_cast(&source, &DataType::Int64, rapids_backend()), "{source:?}");
            assert!(!is_supported_arrow_implicit_cast(&source, &DataType::Int64, rapids_backend()), "{source:?}");
            assert!(!cast_can_fail_at_runtime(&source, &DataType::Int64), "{source:?}");
            // The cuDF AST casts only numeric and decimal values.
            assert!(!is_supported_ast_cast(&source, &DataType::Int64, rapids_backend()), "{source:?}");
            for target in [DataType::Int32, DataType::UInt64, DataType::Float64, DataType::Utf8, DataType::Boolean] {
                assert!(!is_supported_arrow_cast(&source, &target, rapids_backend()), "{source:?} -> {target:?}");
            }
        }
        let zoned = DataType::Timestamp(unit, Some("UTC".into()));
        assert!(!is_supported_arrow_cast(&zoned, &DataType::Int64, rapids_backend()), "{zoned:?}");
    }
    assert!(!is_supported_arrow_cast(&DataType::Date32, &DataType::Int64, rapids_backend()));
    assert!(is_supported_ast_cast(&DataType::Int32, &DataType::Int64, rapids_backend()));
    assert!(is_supported_ast_cast(&DataType::Decimal128(10, 2), &DataType::Float64, rapids_backend()));
    assert!(!is_supported_ast_cast(&DataType::Float64, &DataType::Int64, rapids_backend()));
}

#[test]
fn casts_that_can_fail_at_runtime_exclude_widening_and_identity() {
    use query_planning::datatypes::cast_can_fail_at_runtime;

    for (source, target) in [
        (DataType::Int64, DataType::Int32),
        (DataType::UInt64, DataType::Int64),
        (DataType::Int8, DataType::UInt64),
        (DataType::Float64, DataType::Int64),
        (DataType::Float32, DataType::UInt8),
        (DataType::Decimal128(21, 0), DataType::Int32),
        (DataType::Int64, DataType::Decimal128(20, 2)),
        (DataType::UInt64, DataType::Decimal128(19, 0)),
        (DataType::Float64, DataType::Decimal128(38, 2)),
        (DataType::Float32, DataType::Decimal128(38, 0)),
        (DataType::Decimal128(38, 4), DataType::Decimal128(10, 2)),
        (DataType::Decimal128(10, 2), DataType::Decimal128(10, 4)),
        (DataType::Decimal128(10, 4), DataType::Decimal128(8, 2)),
        // A finer unit multiplies the stored count, which can leave Int64.
        (DataType::Timestamp(TimeUnit::Second, None), DataType::Timestamp(TimeUnit::Millisecond, None)),
        (DataType::Timestamp(TimeUnit::Millisecond, None), DataType::Timestamp(TimeUnit::Nanosecond, None)),
        (DataType::Duration(TimeUnit::Second), DataType::Duration(TimeUnit::Nanosecond)),
        (DataType::Duration(TimeUnit::Microsecond), DataType::Duration(TimeUnit::Nanosecond)),
        (DataType::Date32, DataType::Timestamp(TimeUnit::Microsecond, None)),
        (DataType::Date32, DataType::Timestamp(TimeUnit::Nanosecond, None)),
    ] {
        assert!(cast_can_fail_at_runtime(&source, &target), "{source:?} -> {target:?}");
    }
    for (source, target) in [
        (DataType::Int32, DataType::Int64),
        (DataType::UInt8, DataType::Int16),
        (DataType::Int64, DataType::Int64),
        (DataType::Int64, DataType::Float64),
        (DataType::Int64, DataType::Utf8),
        (DataType::Float64, DataType::Float32),
        (DataType::Int32, DataType::Decimal128(12, 2)),
        (DataType::UInt64, DataType::Decimal128(20, 0)),
        (DataType::Decimal128(10, 2), DataType::Decimal128(12, 4)),
        // Dropping two digits leaves 8, and rounding up adds at most one more.
        (DataType::Decimal128(10, 4), DataType::Decimal128(10, 2)),
        (DataType::Decimal128(10, 2), DataType::Decimal128(10, 2)),
        // A coarser unit divides, and an Int32 day count fits every unit down to milliseconds.
        (DataType::Timestamp(TimeUnit::Millisecond, None), DataType::Timestamp(TimeUnit::Second, None)),
        (DataType::Timestamp(TimeUnit::Nanosecond, None), DataType::Timestamp(TimeUnit::Microsecond, None)),
        (DataType::Duration(TimeUnit::Nanosecond), DataType::Duration(TimeUnit::Second)),
        (DataType::Date32, DataType::Timestamp(TimeUnit::Second, None)),
        (DataType::Date32, DataType::Timestamp(TimeUnit::Millisecond, None)),
    ] {
        assert!(!cast_can_fail_at_runtime(&source, &target), "{source:?} -> {target:?}");
    }
}

#[test]
fn floor_and_ceil_validate_only_over_matching_floating_point_types() {
    use query_planning::expr::UnaryMathOp;

    let schema = Arc::new(Schema::new(vec![
        Field::new("whole", DataType::Int64, true),
        Field::new("score", DataType::Float64, true),
    ]));
    let project = |op: UnaryMathOp, column: &str, input_output: DataType| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        let output_schema = Arc::new(Schema::new(vec![Field::new("out", input_output.clone(), true)]));
        let projection = builder
            .add_projection(
                source,
                projection_outputs(vec![Expr::unary_math(op, Expr::column(column), input_output)], output_schema),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(projection).unwrap();
        preflight_query_plan(&plan)
    };

    for op in [UnaryMathOp::Floor, UnaryMathOp::Ceil] {
        project(op, "score", DataType::Float64).unwrap_or_else(|error| panic!("{op} over Float64: {error:?}"));
        assert_eq!(project(op, "whole", DataType::Int64).unwrap_err().code(), QueryErrorCode::UnsupportedType);
        assert_eq!(project(op, "score", DataType::Float32).unwrap_err().code(), QueryErrorCode::UnsupportedType);
    }
}

#[test]
fn timestamp_difference_is_a_duration_only_between_timezone_free_timestamps_of_one_unit() {
    let timestamp = |unit| DataType::Timestamp(unit, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("later_ms", timestamp(TimeUnit::Millisecond), true),
        Field::new("earlier_ms", timestamp(TimeUnit::Millisecond), true),
        Field::new("earlier_us", timestamp(TimeUnit::Microsecond), true),
        Field::new("zoned", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), true),
        Field::new("day", DataType::Date32, true),
    ]));
    let subtract = |left: &str, right: &str, output: DataType| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        let output_schema = Arc::new(Schema::new(vec![Field::new("out", output, true)]));
        let difference = Expr::binary(Expr::column(left), BinaryOp::Subtract, Expr::column(right));
        let projection = builder.add_projection(source, projection_outputs(vec![difference], output_schema)).unwrap();
        let plan = builder.finish_with_host_sink(projection).unwrap();
        preflight_query_plan(&plan)
    };

    let milliseconds = DataType::Duration(TimeUnit::Millisecond);
    subtract("later_ms", "earlier_ms", milliseconds.clone()).unwrap();
    for (left, right) in [("later_ms", "earlier_us"), ("zoned", "zoned"), ("day", "day")] {
        let error = subtract(left, right, milliseconds.clone()).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "{left} - {right}");
    }
}

/// The output type native validation derives for `left op right`, as `Debug` text, or its rejection.
///
/// The projection declares `left` as its output, so a different derived type surfaces as the schema mismatch that
/// carries it.
fn planned_temporal_binary_type(left: &DataType, op: BinaryOp, right: &DataType) -> Result<String, QueryError> {
    let schema =
        Arc::new(Schema::new(vec![Field::new("left", left.clone(), true), Field::new("right", right.clone(), true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let output_schema = Arc::new(Schema::new(vec![Field::new("out", left.clone(), true)]));
    let result = Expr::binary(Expr::column("left"), op, Expr::column("right"));
    let projection = builder.add_projection(source, projection_outputs(vec![result], output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();
    match preflight_query_plan(&plan) {
        Ok(_) => Ok(format!("{left:?}")),
        Err(error) if error.fact_value("reason") == Some("projection_output_schema_mismatch") => {
            Ok(error.fact_value("actual_type").expect("schema mismatch names the derived type").to_owned())
        },
        Err(error) => Err(error),
    }
}

#[test]
fn timestamp_plus_or_minus_duration_validates_only_for_one_unit_and_never_for_dates() {
    let units = [TimeUnit::Second, TimeUnit::Millisecond, TimeUnit::Microsecond, TimeUnit::Nanosecond];
    let timestamp = |unit| DataType::Timestamp(unit, None);
    let duration = DataType::Duration;
    for unit in units {
        let derived = |left: DataType, op, right: DataType| planned_temporal_binary_type(&left, op, &right).unwrap();
        let expected = format!("{:?}", timestamp(unit));
        assert_eq!(derived(timestamp(unit), BinaryOp::Add, duration(unit)), expected);
        assert_eq!(derived(timestamp(unit), BinaryOp::Subtract, duration(unit)), expected);
        assert_eq!(derived(duration(unit), BinaryOp::Add, timestamp(unit)), expected);
    }

    let rejected_with = |left: DataType, op, right: DataType, reason: &str| {
        let error = planned_temporal_binary_type(&left, op, &right).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "{left:?} {op:?} {right:?}");
        assert_eq!(error.fact_value("reason"), Some(reason), "{left:?} {op:?} {right:?}");
    };
    for unit in units {
        // Arrow has no date plus or minus a duration, and libcudf's date unit is the day.
        rejected_with(DataType::Date32, BinaryOp::Add, duration(unit), "date_duration_arithmetic_unsupported");
        rejected_with(DataType::Date32, BinaryOp::Subtract, duration(unit), "date_duration_arithmetic_unsupported");
        rejected_with(duration(unit), BinaryOp::Add, DataType::Date32, "date_duration_arithmetic_unsupported");
    }
    let milliseconds = timestamp(TimeUnit::Millisecond);
    for unit in [TimeUnit::Second, TimeUnit::Microsecond, TimeUnit::Nanosecond] {
        rejected_with(milliseconds.clone(), BinaryOp::Add, duration(unit), "timestamp_duration_unit_mismatch");
        rejected_with(milliseconds.clone(), BinaryOp::Subtract, duration(unit), "timestamp_duration_unit_mismatch");
        rejected_with(duration(unit), BinaryOp::Add, milliseconds.clone(), "timestamp_duration_unit_mismatch");
    }
}

#[test]
fn temporal_arithmetic_that_validates_has_the_same_operation_in_the_cudf_backend() {
    let temporal_types = [TimeUnit::Second, TimeUnit::Millisecond, TimeUnit::Microsecond, TimeUnit::Nanosecond]
        .into_iter()
        .flat_map(|unit| [DataType::Timestamp(unit, None), DataType::Duration(unit)])
        .chain([DataType::Date32])
        .collect::<Vec<_>>();
    let mut accepted = 0;
    for op in [BinaryOp::Add, BinaryOp::Subtract] {
        for left in &temporal_types {
            for right in &temporal_types {
                let Ok(planned) = planned_temporal_binary_type(left, op, right) else {
                    continue;
                };
                accepted += 1;
                let executed = rapids_backend().binary_output_type(op, left, right).unwrap_or_else(|error| {
                    panic!("planning accepts {left:?} {op:?} {right:?} but execution cannot run it: {error:?}")
                });
                assert_eq!(planned, format!("{executed:?}"), "{left:?} {op:?} {right:?}");
            }
        }
    }
    // Per unit: timestamp + duration, timestamp - duration, duration + timestamp, and timestamp - timestamp.
    assert_eq!(accepted, 16);
}

#[test]
fn only_timestamp_arithmetic_that_arrow_checks_can_fail_at_runtime() {
    use query_planning::datatypes::temporal_arithmetic_can_fail_at_runtime;

    let timestamp = DataType::Timestamp(TimeUnit::Millisecond, None);
    let duration = DataType::Duration(TimeUnit::Millisecond);
    for (op, left, right) in [
        (BinaryOp::Add, &timestamp, &duration),
        (BinaryOp::Add, &duration, &timestamp),
        (BinaryOp::Subtract, &timestamp, &duration),
        (BinaryOp::Subtract, &timestamp, &timestamp),
    ] {
        assert!(temporal_arithmetic_can_fail_at_runtime(op, left, right), "{left:?} {op:?} {right:?}");
    }
    let zoned = DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()));
    let nanoseconds = DataType::Timestamp(TimeUnit::Nanosecond, None);
    for (op, left, right) in [
        // Unsupported combinations never run, so they have nothing to check.
        (BinaryOp::Add, &timestamp, &timestamp),
        (BinaryOp::Subtract, &duration, &timestamp),
        (BinaryOp::Subtract, &timestamp, &nanoseconds),
        (BinaryOp::Subtract, &zoned, &zoned),
        (BinaryOp::Subtract, &DataType::Date32, &DataType::Date32),
        (BinaryOp::Add, &DataType::Date32, &duration),
        (BinaryOp::Add, &timestamp, &DataType::Duration(TimeUnit::Second)),
        (BinaryOp::Subtract, &timestamp, &DataType::Duration(TimeUnit::Nanosecond)),
        // Other operators and operand types.
        (BinaryOp::Multiply, &duration, &DataType::Int64),
        (BinaryOp::Add, &DataType::Int64, &DataType::Int64),
    ] {
        assert!(!temporal_arithmetic_can_fail_at_runtime(op, left, right), "{left:?} {op:?} {right:?}");
    }
}

#[test]
fn backend_dtype_rejection_reports_unsupported_type_with_cudf_reason() {
    let unsupported = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Utf8, true)), 2);

    let error = rapids_backend()
        .binary_output_type(BinaryOp::Add, &unsupported, &DataType::Decimal128(10, 2))
        .expect_err("FixedSizeList<Utf8> has no cuDF dtype");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.fact_value("left_type"), Some(format!("{unsupported:?}").as_str()));
    assert!(error.fact_value("cudf_code").is_some());
    assert_eq!(error.fact_value("cudf_reason"), Some("unsupported_fixed_size_list_child_type"));
}

#[test]
fn avg_preflight_rejects_decimal_arguments_datafusion_sums_as_decimal256() {
    let avg_preflight = |precision: u8| {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("d", DataType::Decimal128(precision, 2), true),
        ]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("avg_d", DataType::Decimal128(precision + 4, 6), true),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
        let spec = AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("d"), "avg_d")],
        );
        let aggregate = builder.add_aggregate(source, spec, output_schema).unwrap();
        preflight_query_plan(&builder.finish_with_host_sink(aggregate).unwrap())
    };

    avg_preflight(25).expect("DataFusion sums a Decimal128(25, 2) AVG argument as Decimal128(38, 2)");
    let error = avg_preflight(26).expect_err("DataFusion sums a Decimal128(26, 2) AVG argument as Decimal256");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "query_engine.aggregate.avg_state_type");
    assert_eq!(error.fact_value("alias"), Some("avg_d"));
}
