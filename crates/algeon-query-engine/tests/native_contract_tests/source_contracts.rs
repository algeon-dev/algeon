use std::{collections::BTreeSet, sync::Arc};

use query_engine::{
    rapids_backend::rapids_analysis_context,
    source::{
        ByteRangeDisposition, LocalParquetFile, LocalParquetSource, SourceFormat, SourceNode, ZERO_COLUMN_COUNT_COLUMN,
    },
};

use super::*;

#[path = "source_contracts/chunk_planning.rs"]
mod chunk_planning;

#[test]
fn source_module_facade_preserves_public_model_and_diagnostic_paths() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let file: LocalParquetFile = LocalParquetFile::new("/tmp/source-facade.parquet").with_file_size_bytes(128);
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["key".to_string()]);
    let node = SourceNode::LocalParquet(source.clone());

    assert!(matches!(node, SourceNode::LocalParquet(_)));
    assert_eq!(source.diagnostics.format, SourceFormat::Parquet);
    assert_eq!(source.diagnostics.byte_range_disposition, ByteRangeDisposition::FullFiles);
    assert_eq!(ZERO_COLUMN_COUNT_COLUMN, ZERO_COLUMN_COUNT_COLUMN);
}

#[test]
fn local_parquet_source_tracks_direct_source_diagnostics() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/source-contract.parquet")
                .with_stable_id("source_contract:3235196281")
                .with_file_size_bytes(3_235_196_281)
                .with_row_groups(vec![0, 2, 4])
                .with_row_group_bytes(vec![10, 20, 30, 40, 50]),
        ],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["payload".to_string()]);

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
    assert_eq!(source.files[0].stable_id, "source_contract:3235196281");
    assert!(source.is_direct_device_read());
    assert_eq!(source.read_column_count(), 2);
    assert_eq!(source.read_column_file_indices, vec![0, 1]);
    assert_eq!(source.projected_output_column_count(), 1);
    assert_eq!(source.projected_output_column_source_refs, vec![SourceColumnRef::File(0)]);
    assert_eq!(source.pure_filter_column_count(), 1);
    assert_eq!(source.pure_filter_column_source_refs, vec![SourceColumnRef::File(1)]);
    assert_eq!(source.selected_row_group_count(), 3);
    assert_eq!(source.diagnostics.file_count, 1);
    assert_eq!(source.diagnostics.ranged_file_count, 0);
    assert_eq!(source.diagnostics.known_bytes, Some(3_235_196_281));
    assert_eq!(source.diagnostics.selected_row_groups, Some(3));
    let stats = source.diagnostics.selected_row_group_stats.as_ref().expect("row-group stats should be known");
    assert_eq!(stats.selected_row_groups, 3);
    assert_eq!(stats.total_row_groups, 5);
    assert_eq!(stats.selected_compressed_bytes, Some(90));
    assert_eq!(stats.total_compressed_bytes, Some(150));
    assert_eq!(source.diagnostics.read_column_count, 2);
    assert_eq!(source.diagnostics.projected_output_column_count, 1);
    assert_eq!(source.diagnostics.pure_filter_column_count, 1);
    assert_eq!(source.diagnostics.predicate_disposition, PredicateDisposition::Absent);
    assert_eq!(source.diagnostics.byte_range_disposition, ByteRangeDisposition::FullFiles);
    assert_eq!(source.diagnostics.partition_column_disposition, PartitionColumnDisposition::Absent);
    assert_eq!(source.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::NotRequired);
    assert_eq!(source.diagnostics.scan_limit_disposition, ScanLimitDisposition::Absent);
    assert_eq!(source.diagnostics.schema_mapping_disposition, SchemaMappingDisposition::DirectByName);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_reader_predicate_sets_reader_pushdown_diagnostics() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let predicate = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate.parquet")
                .with_stable_id("reader-predicate")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["payload".to_string()])
    .with_reader_predicate(predicate);

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
    assert_eq!(source.diagnostics.predicate_disposition, PredicateDisposition::ReaderPushdown);
    assert!(source.reader_predicate.is_some());
}

#[test]
fn local_parquet_source_rejects_reader_predicate_with_position_deletes() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-0001",
        ResolvedParquetSource::local_kvikio("/tmp/delete-0001.parquet"),
    )
    .unwrap();
    let predicate = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-position-delete.parquet")
                .with_stable_id("reader-predicate-position-delete")
                .with_file_size_bytes(1024)
                .with_position_delete_files(vec![delete_file]),
        ],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["payload".to_string()])
    .with_reader_predicate(predicate);

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("row_deletes_reject_reader_predicate"));
}

#[test]
fn local_parquet_source_accepts_contiguous_row_groups_with_position_deletes() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-0001",
        ResolvedParquetSource::local_kvikio("/tmp/delete-0001.parquet"),
    )
    .unwrap();
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/position-delete-row-groups.parquet")
                .with_stable_id("position-delete-row-groups")
                .with_row_groups(vec![1, 2])
                .with_row_group_rows(vec![10, 20, 30])
                .with_position_delete_files(vec![delete_file]),
        ],
        schema,
        vec!["key".to_string()],
    );

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
}

#[test]
fn local_parquet_source_rejects_noncontiguous_row_groups_with_position_deletes() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-0001",
        ResolvedParquetSource::local_kvikio("/tmp/delete-0001.parquet"),
    )
    .unwrap();
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/position-delete-row-groups.parquet")
                .with_stable_id("position-delete-row-groups")
                .with_row_groups(vec![0, 2])
                .with_row_group_rows(vec![10, 20, 30])
                .with_position_delete_files(vec![delete_file]),
        ],
        schema,
        vec!["key".to_string()],
    );

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("position_deletes_require_contiguous_row_groups"));
}

#[test]
fn local_parquet_source_reader_predicate_accepts_decimal_contract() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-decimal.parquet")
                .with_stable_id("reader-predicate-decimal")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["key".to_string(), "amount".to_string()],
    );
    let predicate = Expr::binary(
        Expr::column("amount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 100, precision: 15, scale: 2 }),
    );

    assert!(source.reader_predicate_is_supported(&predicate, rapids_analysis_context()));
}

#[test]
fn local_parquet_source_reader_predicate_rejects_degenerate_roots() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("flag", DataType::Boolean, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-degenerate-root.parquet")
                .with_stable_id("reader-predicate-degenerate-root")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["id".to_string(), "flag".to_string()],
    );

    assert!(!source.reader_predicate_is_supported(&Expr::column("flag"), rapids_analysis_context()));
    assert!(
        !source.reader_predicate_is_supported(&Expr::literal(LiteralValue::Boolean(true)), rapids_analysis_context())
    );
    assert!(!source.reader_predicate_is_supported(
        &Expr::Alias { expr: Box::new(Expr::column("flag")), alias: "flag_alias".to_string() },
        rapids_analysis_context()
    ));

    let safe_operation_root =
        Expr::binary(Expr::column("flag"), BinaryOp::Eq, Expr::literal(LiteralValue::Boolean(true)));
    assert!(source.reader_predicate_is_supported(&safe_operation_root, rapids_analysis_context()));
    assert!(source.reader_predicate_is_supported(&Expr::logical_not(Expr::column("flag")), rapids_analysis_context()));
}

#[test]
fn local_parquet_source_reader_predicate_rejects_list_carry_columns() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tags", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-list.parquet")
                .with_stable_id("reader-predicate-list")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["key".to_string(), "tags".to_string()],
    );
    let predicate = Expr::is_not_null(Expr::column("tags"));

    assert!(!source.reader_predicate_is_supported(&predicate, rapids_analysis_context()));
}

#[test]
fn local_parquet_source_reader_predicate_rejects_narrow_decimal_carry_columns() {
    for amount_type in [DataType::Decimal32(9, 2), DataType::Decimal64(18, 2)] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("amount", amount_type, true),
        ]));
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/reader-predicate-narrow-decimal.parquet")
                    .with_stable_id("reader-predicate-narrow-decimal")
                    .with_file_size_bytes(1024),
            ],
            schema,
            vec!["key".to_string(), "amount".to_string()],
        );

        assert!(
            !source
                .reader_predicate_is_supported(&Expr::is_not_null(Expr::column("amount")), rapids_analysis_context())
        );
    }
}

#[test]
fn local_parquet_source_reader_predicate_accepts_decimal32_range_equality_and_in_list() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("amount", DataType::Decimal128(9, 2), false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-decimal32.parquet")
                .with_stable_id("reader-predicate-decimal32")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["key".to_string(), "amount".to_string()],
    );
    let range_predicate = Expr::binary(
        Expr::column("amount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 100, precision: 9, scale: 2 }),
    );
    let equality_predicate = Expr::binary(
        Expr::column("amount"),
        BinaryOp::Eq,
        Expr::literal(LiteralValue::Decimal128 { raw: 100, precision: 9, scale: 2 }),
    );
    let in_list_predicate = Expr::InList {
        expr: Box::new(Expr::column("amount")),
        list: vec![
            Expr::literal(LiteralValue::Decimal128 { raw: 100, precision: 9, scale: 2 }),
            Expr::literal(LiteralValue::Int32(10)),
        ],
        negated: false,
    };

    assert!(source.reader_predicate_is_supported(&range_predicate, rapids_analysis_context()));
    assert!(source.reader_predicate_is_supported(&equality_predicate, rapids_analysis_context()));
    assert!(source.reader_predicate_is_supported(&in_list_predicate, rapids_analysis_context()));
}

#[test]
fn local_parquet_source_reader_predicate_accepts_column_typed_integer_literals() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-typed-literal.parquet")
                .with_stable_id("reader-predicate-typed-literal")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    );
    let widened_integer = Expr::binary(Expr::column("key"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(10)));
    let mismatched_string =
        Expr::binary(Expr::column("key"), BinaryOp::Eq, Expr::literal(LiteralValue::Utf8("10".to_string())));
    let string_match =
        Expr::binary(Expr::column("payload"), BinaryOp::Eq, Expr::literal(LiteralValue::Utf8("ok".to_string())));

    assert!(source.reader_predicate_is_supported(&widened_integer, rapids_analysis_context()));
    assert!(!source.reader_predicate_is_supported(&mismatched_string, rapids_analysis_context()));
    assert!(source.reader_predicate_is_supported(&string_match, rapids_analysis_context()));
}

#[test]
fn local_parquet_source_reader_predicate_rewrites_fixed_ascii_substring_prefix_in_list() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("phone", DataType::Utf8, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-prefix.parquet")
                .with_stable_id("reader-predicate-prefix")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["phone".to_string(), "payload".to_string()],
    );
    let predicate = Expr::in_list(
        Expr::substring(Expr::column("phone"), 1, Some(2)),
        vec![Expr::literal(LiteralValue::Utf8("13".to_string())), Expr::literal(LiteralValue::Utf8("31".to_string()))],
    );

    assert!(!source.reader_predicate_is_supported(&predicate, rapids_analysis_context()));
    let rewritten = source
        .reader_predicate_supported_expression(&predicate, rapids_analysis_context())
        .expect("fixed ASCII prefix substring IN list should rewrite to reader ranges");

    assert!(source.reader_predicate_is_supported(&rewritten, rapids_analysis_context()));
    let rewritten_text = rewritten.to_string();
    assert!(!rewritten_text.contains("SUBSTRING"), "{rewritten_text}");
    assert!(rewritten_text.contains("phone >= \"13\""), "{rewritten_text}");
    assert!(rewritten_text.contains("phone < \"14\""), "{rewritten_text}");
    assert!(rewritten_text.contains("phone >= \"31\""), "{rewritten_text}");
    assert!(rewritten_text.contains("phone < \"32\""), "{rewritten_text}");
}

#[test]
fn local_parquet_source_reader_predicate_prefix_rewrite_preserves_filter_truth_boundaries() {
    let schema = Arc::new(Schema::new(vec![Field::new("phone", DataType::Utf8, true)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-prefix-boundaries.parquet")
                .with_stable_id("reader-predicate-prefix-boundaries")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["phone".to_string()],
    );
    let predicate = Expr::in_list(
        Expr::substring(Expr::column("phone"), 1, Some(2)),
        vec![Expr::literal(LiteralValue::Utf8("13".to_string())), Expr::literal(LiteralValue::Utf8("31".to_string()))],
    );
    let rewritten = source
        .reader_predicate_supported_expression(&predicate, rapids_analysis_context())
        .expect("fixed ASCII prefix substring IN list should rewrite to reader ranges");

    for phone in [
        None,
        Some(""),
        Some("1"),
        Some("12"),
        Some("13"),
        Some("130"),
        Some("13é"),
        Some("14"),
        Some("30"),
        Some("31"),
        Some("31😀"),
        Some("32"),
    ] {
        let expected = phone.is_some_and(|value| {
            let prefix = value.chars().take(2).collect::<String>();
            matches!(prefix.as_str(), "13" | "31")
        });
        assert_eq!(
            eval_prefix_rewrite_filter_truth(&rewritten, phone),
            expected,
            "phone={phone:?} rewritten={rewritten}"
        );
    }
}

#[test]
fn local_parquet_source_reader_predicate_rejects_unsafe_substring_prefix_rewrites() {
    let schema = Arc::new(Schema::new(vec![Field::new("phone", DataType::Utf8, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/reader-predicate-unsafe-prefix.parquet")
                .with_stable_id("reader-predicate-unsafe-prefix")
                .with_file_size_bytes(1024),
        ],
        schema,
        vec!["phone".to_string()],
    );
    let wrong_start = Expr::in_list(
        Expr::substring(Expr::column("phone"), 2, Some(2)),
        vec![Expr::literal(LiteralValue::Utf8("13".to_string()))],
    );
    let shorter_literal = Expr::in_list(
        Expr::substring(Expr::column("phone"), 1, Some(2)),
        vec![Expr::literal(LiteralValue::Utf8("1".to_string()))],
    );
    let non_ascii_literal = Expr::in_list(
        Expr::substring(Expr::column("phone"), 1, Some(2)),
        vec![Expr::literal(LiteralValue::Utf8("éa".to_string()))],
    );

    assert!(source.reader_predicate_supported_expression(&wrong_start, rapids_analysis_context()).is_none());
    assert!(source.reader_predicate_supported_expression(&shorter_literal, rapids_analysis_context()).is_none());
    assert!(source.reader_predicate_supported_expression(&non_ascii_literal, rapids_analysis_context()).is_none());
}

fn eval_prefix_rewrite_filter_truth(expr: &Expr, phone: Option<&str>) -> bool {
    eval_prefix_rewrite_bool(expr, phone).unwrap_or(false)
}

fn eval_prefix_rewrite_bool(expr: &Expr, phone: Option<&str>) -> Option<bool> {
    match expr {
        Expr::Binary { left, op: BinaryOp::And, right, .. } => {
            sql_and(eval_prefix_rewrite_bool(left, phone), eval_prefix_rewrite_bool(right, phone))
        },
        Expr::Binary { left, op: BinaryOp::Or, right, .. } => {
            sql_or(eval_prefix_rewrite_bool(left, phone), eval_prefix_rewrite_bool(right, phone))
        },
        Expr::Binary { left, op, right, .. } => {
            let left = eval_prefix_rewrite_string(left, phone)?;
            let right = eval_prefix_rewrite_string(right, phone)?;
            Some(match op {
                BinaryOp::Eq => left == right,
                BinaryOp::NotEq => left != right,
                BinaryOp::Gt => left > right,
                BinaryOp::GtEq => left >= right,
                BinaryOp::Lt => left < right,
                BinaryOp::LtEq => left <= right,
                other => panic!("unsupported prefix rewrite operator in test evaluator: {other:?}"),
            })
        },
        Expr::Not(input) => eval_prefix_rewrite_bool(input, phone).map(|value| !value),
        other => panic!("unsupported prefix rewrite expression in test evaluator: {other:?}"),
    }
}

fn eval_prefix_rewrite_string(expr: &Expr, phone: Option<&str>) -> Option<String> {
    match expr {
        Expr::Column(column) if column.name() == "phone" => phone.map(str::to_string),
        Expr::Literal(LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value)) => Some(value.clone()),
        Expr::Alias { expr, .. } => eval_prefix_rewrite_string(expr, phone),
        other => panic!("unsupported prefix rewrite string expression in test evaluator: {other:?}"),
    }
}

fn sql_and(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn sql_or(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

#[test]
fn local_parquet_source_represents_order_insensitive_scan_limit() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_row_group_rows(vec![5, 7])],
        schema,
        vec!["key".to_string()],
    )
    .with_scan_limit(Some(9));

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
    assert_eq!(source.scan_limit, Some(9));
    assert_eq!(source.diagnostics.scan_limit_disposition, ScanLimitDisposition::Represented);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_represents_partition_columns_as_synthetic_source_columns() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("part", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_partition_values(vec![LiteralValue::Int32(7)])],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)])
    .with_column_roles(vec!["key".to_string(), "part".to_string()], vec!["payload".to_string(), "part".to_string()]);

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
    assert_eq!(source.partition_column_count(), 1);
    assert_eq!(source.read_columns, vec!["key".to_string(), "payload".to_string()]);
    assert_eq!(
        source.projected_output_column_source_refs,
        vec![SourceColumnRef::File(0), SourceColumnRef::Partition(0)]
    );
    assert_eq!(source.pure_filter_column_source_refs, vec![SourceColumnRef::File(1), SourceColumnRef::Partition(0)]);
    assert_eq!(source.diagnostics.partition_column_disposition, PartitionColumnDisposition::Represented);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_allows_partition_only_columns_with_row_count_metadata() {
    let schema = Arc::new(Schema::new(vec![Field::new("part", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/source-contract.parquet")
                .with_partition_values(vec![LiteralValue::Int32(7)])
                .with_row_group_rows(vec![3]),
        ],
        schema,
        Vec::new(),
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)])
    .with_column_roles(vec!["part".to_string()], Vec::new());

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
    assert_eq!(source.read_columns, Vec::<String>::new());
    assert_eq!(source.partition_column_count(), 1);
    assert_eq!(source.projected_output_column_source_refs, vec![SourceColumnRef::Partition(0)]);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_represents_missing_nullable_columns_as_typed_nulls() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("missing_payload", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_row_group_rows(vec![3, 4])],
        schema,
        vec!["key".to_string()],
    )
    .with_missing_columns(vec![MissingColumn::new("missing_payload", DataType::Int64, true)])
    .with_column_roles(vec!["key".to_string(), "missing_payload".to_string()], Vec::new());

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
    assert_eq!(source.missing_columns.len(), 1);
    assert_eq!(
        source.projected_output_column_source_refs,
        vec![SourceColumnRef::File(0), SourceColumnRef::MissingNull(0)]
    );
    assert_eq!(source.diagnostics.schema_mapping_disposition, SchemaMappingDisposition::MissingNullableColumns);
    assert_eq!(source.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::NotRequired);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_tracks_field_id_binding_diagnostics_and_output_override_contract() {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("old_status", DataType::Int64, false),
        Field::new("status", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let file = LocalParquetFile::new("/tmp/source-contract.parquet").with_field_id_read_bindings(vec![
        ParquetFieldIdReadBinding::new(2, "status", "old_status", 0, false, DataType::Int64),
    ]);
    let source = LocalParquetSource::from_files(vec![file], source_schema, vec!["old_status".to_string()])
        .with_column_roles(vec!["status".to_string()], Vec::new())
        .with_column_source_refs(vec![0], vec![SourceColumnRef::File(0)], Vec::new())
        .with_output_schema_override(output_schema);

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();

    assert_eq!(source.diagnostics.schema_mapping_disposition, SchemaMappingDisposition::FieldIdBindings);
    assert_eq!(source.diagnostics.schema_mapping_disposition.as_str(), "field_id_bindings");
    assert_eq!(source.source_column_ref("status"), Some(SourceColumnRef::File(0)));
}

#[test]
fn local_parquet_source_rejects_nullable_physical_field_for_required_output_override() {
    let source_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet")],
        source_schema,
        vec!["status".to_string()],
    )
    .with_column_roles(vec!["status".to_string()], Vec::new())
    .with_output_schema_override(output_schema);

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("output_schema_override_field_mismatch"));
    assert_eq!(error.fact_value("source_nullable"), Some("true"));
    assert_eq!(error.fact_value("output_nullable"), Some("false"));
}

#[test]
fn local_parquet_output_override_allows_list_child_field_name_drift() {
    let source_list_type = DataType::List(Arc::new(
        Field::new("element", DataType::Utf8, true)
            .with_metadata(std::collections::HashMap::from([("PARQUET:field_id".to_string(), "58".to_string())])),
    ));
    let output_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let source_schema = Arc::new(Schema::new(vec![Field::new("tags", source_list_type, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("tags", output_list_type, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet")],
        source_schema,
        vec!["tags".to_string()],
    )
    .with_column_roles(vec!["tags".to_string()], Vec::new())
    .with_output_schema_override(output_schema);

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
}

#[test]
fn local_parquet_output_override_allows_only_fixed_size_list_child_name_drift() {
    let source_type = DataType::FixedSizeList(
        Arc::new(
            Field::new("element", DataType::Int32, true)
                .with_metadata(std::collections::HashMap::from([("PARQUET:field_id".to_string(), "58".to_string())])),
        ),
        2,
    );
    let source_schema = Arc::new(Schema::new(vec![Field::new("features", source_type, true)]));
    let source = |output_type| {
        LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/source-contract.parquet")],
            Arc::clone(&source_schema),
            vec!["features".to_string()],
        )
        .with_column_roles(vec!["features".to_string()], Vec::new())
        .with_output_schema_override(Arc::new(Schema::new(vec![Field::new("features", output_type, true)])))
    };

    validate_local_parquet_source(
        &source(DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2)),
        rapids_analysis_context(),
    )
    .unwrap();

    for output_type in [
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 3),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, false)), 2),
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int32, true).with_metadata(std::collections::HashMap::from([(
                "extension".to_string(),
                "output-only".to_string(),
            )]))),
            2,
        ),
    ] {
        let error = validate_local_parquet_source(&source(output_type), rapids_analysis_context()).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("output_schema_override_field_mismatch"));
        assert_eq!(error.fact_value("field_name"), Some("features"));
    }
}

#[test]
fn local_parquet_output_override_rejects_non_parquet_list_child_metadata_drift() {
    let source_list_type = DataType::List(Arc::new(Field::new("element", DataType::Utf8, true).with_metadata(
        std::collections::HashMap::from([
            ("PARQUET:field_id".to_string(), "58".to_string()),
            ("extension".to_string(), "source-only".to_string()),
        ]),
    )));
    let output_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let source_schema = Arc::new(Schema::new(vec![Field::new("tags", source_list_type, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("tags", output_list_type, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet")],
        source_schema,
        vec!["tags".to_string()],
    )
    .with_column_roles(vec!["tags".to_string()], Vec::new())
    .with_output_schema_override(output_schema);

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("output_schema_override_field_mismatch"));
    assert_eq!(error.fact_value("field_name"), Some("tags"));
}

#[test]
fn local_parquet_output_override_allows_timestamp_timezone_normalization_for_logical_timestamptz() {
    let source_type = DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into()));
    let output_type = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let source_schema = Arc::new(Schema::new(vec![Field::new("event_ts", source_type, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("event_ts", output_type, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet")],
        source_schema,
        vec!["event_ts".to_string()],
    )
    .with_column_roles(vec!["event_ts".to_string()], Vec::new())
    .with_output_schema_override(output_schema);

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
}

#[test]
fn local_parquet_source_rejects_output_override_missing_execution_schema_field() {
    let source_schema = Arc::new(Schema::new(vec![Field::new("old_status", DataType::Int64, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let file = LocalParquetFile::new("/tmp/source-contract.parquet").with_field_id_read_bindings(vec![
        ParquetFieldIdReadBinding::new(2, "status", "old_status", 0, true, DataType::Int64),
    ]);
    let source = LocalParquetSource::from_files(vec![file], source_schema, vec!["old_status".to_string()])
        .with_column_roles(vec!["status".to_string()], Vec::new())
        .with_column_source_refs(vec![0], vec![SourceColumnRef::File(0)], Vec::new())
        .with_output_schema_override(output_schema);

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("output_schema_override_field_absent"));
    assert_eq!(error.fact_value("field_name"), Some("status"));
}

#[test]
fn local_parquet_source_accepts_missing_nullable_list_columns() {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("missing_tags", list_type.clone(), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_row_group_rows(vec![3])],
        schema,
        vec!["key".to_string()],
    )
    .with_missing_columns(vec![MissingColumn::new("missing_tags", list_type, true)])
    .with_column_roles(vec!["key".to_string(), "missing_tags".to_string()], Vec::new());

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
}

#[test]
fn local_parquet_source_rejects_missing_lists_with_unsupported_children() {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Decimal128(10, 2), true)));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("missing_values", list_type.clone(), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_row_group_rows(vec![3])],
        schema,
        vec!["key".to_string()],
    )
    .with_missing_columns(vec![MissingColumn::new("missing_values", list_type, true)])
    .with_column_roles(vec!["key".to_string(), "missing_values".to_string()], Vec::new());

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.fact_value("reason"), Some("unsupported_missing_column_type"));
    assert_eq!(error.fact_value("missing_column"), Some("missing_values"));
}

#[test]
fn local_parquet_source_rejects_non_nullable_missing_columns() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("missing_payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_row_group_rows(vec![3])],
        schema,
        vec!["key".to_string()],
    )
    .with_missing_columns(vec![MissingColumn::new("missing_payload", DataType::Int64, false)])
    .with_column_roles(vec!["key".to_string(), "missing_payload".to_string()], Vec::new());

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("non_nullable_missing_column"));
    assert_eq!(error.fact_value("missing_column"), Some("missing_payload"));
}

#[test]
fn local_parquet_source_rejects_invalid_partition_column_contracts() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("part", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet")],
        Arc::clone(&schema),
        vec!["key".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)]);

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("partition_value_count_mismatch"));
    assert_eq!(error.fact_value("partition_columns"), Some("1"));
    assert_eq!(error.fact_value("partition_values"), Some("0"));

    let wrong_type = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_partition_values(vec![LiteralValue::Int64(7)])],
        schema,
        vec!["key".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)]);

    let error = validate_local_parquet_source(&wrong_type, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("partition_value_type_mismatch"));
    assert_eq!(error.fact_value("partition_column"), Some("part"));
}

#[test]
fn local_parquet_source_rejects_narrow_decimal_partition_values_locally() {
    for partition_type in [DataType::Decimal32(9, 2), DataType::Decimal64(18, 2)] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("part", partition_type.clone(), true),
        ]));
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/source-contract.parquet")
                    .with_partition_values(vec![LiteralValue::Null(partition_type.clone())]),
            ],
            schema,
            vec!["key".to_string()],
        )
        .with_partition_columns(vec![PartitionColumn::new("part", partition_type.clone(), true)]);

        let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(error.fact_value("reason"), Some("narrow_decimal_partition_unsupported"));
        assert_eq!(error.fact_value("partition_type"), Some(format!("{partition_type:?}").as_str()));
        assert_eq!(error.fact_value("partition_column"), Some("part"));
    }
}

#[test]
fn local_parquet_source_rejects_null_partition_value_for_non_nullable_column() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("part", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/source-contract.parquet")
                .with_partition_values(vec![LiteralValue::Null(DataType::Int32)]),
        ],
        schema,
        vec!["key".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)]);

    let error = validate_local_parquet_source(&source, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.local_parquet_source.validate");
    assert_eq!(error.fact_value("reason"), Some("non_nullable_partition_value_null"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("partition_index"), Some("0"));
    assert_eq!(error.fact_value("partition_column"), Some("part"));
    assert_eq!(error.fact_value("partition_type"), Some("Int32"));
    assert_eq!(error.fact_value("partition_nullable"), Some("false"));
    assert_eq!(error.fact_value("partition_value"), Some("NULL::Int32"));
}

#[test]
fn source_disposition_names_are_stable_manifest_values() {
    let dispositions = [
        (PredicateDisposition::Absent, "absent"),
        (PredicateDisposition::PruningOnly, "pruning_only"),
        (PredicateDisposition::ReaderPushdown, "reader_pushdown"),
        (PredicateDisposition::NativePostScanOnly, "native_post_scan_only"),
        (PredicateDisposition::Rejected, "rejected"),
    ];

    for (disposition, expected) in dispositions {
        assert_eq!(disposition.as_str(), expected);
    }

    assert_eq!(ByteRangeDisposition::FullFiles.as_str(), "full_files");
    assert_eq!(ByteRangeDisposition::CoalescedFullFiles.as_str(), "coalesced_full_files");
    assert_eq!(ByteRangeDisposition::RowGroupSelection.as_str(), "row_group_selection");
    assert_eq!(ByteRangeDisposition::Rejected.as_str(), "rejected");
    assert_eq!(PartitionColumnDisposition::Absent.as_str(), "absent");
    assert_eq!(PartitionColumnDisposition::Rejected.as_str(), "rejected");
    assert_eq!(PartitionColumnDisposition::Represented.as_str(), "represented");
    assert_eq!(ZeroColumnCountDisposition::NotRequired.as_str(), "not_required");
    assert_eq!(ZeroColumnCountDisposition::Rejected.as_str(), "rejected");
    assert_eq!(ZeroColumnCountDisposition::Represented.as_str(), "represented");
    assert_eq!(ScanLimitDisposition::Absent.as_str(), "absent");
    assert_eq!(ScanLimitDisposition::Rejected.as_str(), "rejected");
    assert_eq!(ScanLimitDisposition::Represented.as_str(), "represented");
    assert_eq!(SchemaMappingDisposition::DirectByName.as_str(), "direct_by_name");
    assert_eq!(SchemaMappingDisposition::MissingNullableColumns.as_str(), "missing_nullable_columns");
    assert_eq!(SchemaMappingDisposition::Rejected.as_str(), "rejected");
    assert_eq!(RejectedSourceFeature::ParquetFileReaderFactory.as_str(), "parquet_file_reader_factory");
}

#[test]
fn local_parquet_source_validation_rejects_invalid_reader_contracts() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let empty_columns = LocalParquetSource::new("/tmp/source-contract.parquet", Arc::clone(&schema), Vec::new());

    let error = validate_local_parquet_source(&empty_columns, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("empty_read_columns"));
    assert_eq!(empty_columns.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::Rejected);
    assert!(empty_columns.diagnostics.rejected_features.iter().any(|feature| feature.as_str() == "zero_column_count"));

    let overflow_row_group = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_row_groups(vec![usize::MAX])],
        schema,
        vec!["key".to_string()],
    );

    let error = validate_local_parquet_source(&overflow_row_group, rapids_analysis_context()).unwrap_err();
    let expected_row_group = usize::MAX.to_string();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("row_group"), Some(expected_row_group.as_str()));
}

#[test]
fn local_parquet_source_validation_rejects_unusable_unique_key_facts() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("nullable_key", DataType::UInt32, true),
    ]));
    let source = LocalParquetSource::new(
        "/tmp/source-unique-key-contract.parquet",
        schema,
        vec!["key".to_string(), "nullable_key".to_string()],
    );

    let valid = source.clone().with_unique_key_columns(vec![BTreeSet::from(["key".to_string()])]);
    validate_local_parquet_source(&valid, rapids_analysis_context()).unwrap();

    for (key, expected_reason) in [
        (BTreeSet::new(), "empty_unique_key"),
        (BTreeSet::from(["missing".to_string()]), "unique_key_column_absent"),
        (BTreeSet::from(["nullable_key".to_string()]), "unique_key_column_nullable"),
    ] {
        let error = validate_local_parquet_source(
            &source.clone().with_unique_key_columns(vec![key]),
            rapids_analysis_context(),
        )
        .unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reason"), Some(expected_reason));
    }

    let duplicate =
        source.with_unique_key_columns(vec![BTreeSet::from(["key".to_string()]), BTreeSet::from(["key".to_string()])]);
    let error = validate_local_parquet_source(&duplicate, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("duplicate_unique_key"));
}

#[test]
fn zero_column_count_source_contract_is_represented_without_data_columns() {
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/source-contract.parquet")
            .with_stable_id("source_contract:count")
            .with_file_size_bytes(300)
            .with_row_groups(vec![0, 2])
            .with_row_group_rows(vec![10, 20, 30]),
    ]);

    validate_local_parquet_source(&source, rapids_analysis_context()).unwrap();
    assert!(source.is_zero_column_count_source());
    assert!(source.is_direct_device_read());
    assert_eq!(source.read_column_count(), 0);
    assert_eq!(source.projected_output_column_count(), 0);
    assert_eq!(source.pure_filter_column_count(), 0);
    assert!(source.read_column_file_indices.is_empty());
    assert!(source.projected_output_column_source_refs.is_empty());
    assert!(source.pure_filter_column_source_refs.is_empty());
    assert!(source.schema.field_with_name(ZERO_COLUMN_COUNT_COLUMN).is_ok());
    assert_eq!(source.diagnostics.read_column_count, 0);
    assert_eq!(source.diagnostics.projected_output_column_count, 0);
    assert_eq!(source.diagnostics.pure_filter_column_count, 0);
    assert_eq!(source.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::Represented);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_validation_rejects_unhonorable_source_facts_before_execution() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let base = LocalParquetSource::new("/tmp/source-contract.parquet", Arc::clone(&schema), vec!["key".to_string()]);

    let mut rejected_byte_range = base.clone();
    rejected_byte_range.diagnostics.byte_range_disposition = ByteRangeDisposition::Rejected;
    let error = validate_local_parquet_source(&rejected_byte_range, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("unsupported_source_contract_fact"));
    assert_eq!(error.fact_value("field"), Some("byte_range"));
    assert_eq!(error.fact_value("disposition"), Some("rejected"));

    let mut inconsistent_partition_columns = base.clone();
    inconsistent_partition_columns.diagnostics.partition_column_disposition = PartitionColumnDisposition::Represented;
    let error = validate_local_parquet_source(&inconsistent_partition_columns, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("empty_partition_columns"));

    let mut rejected_zero_column_count = base.clone();
    rejected_zero_column_count.diagnostics.zero_column_count_disposition = ZeroColumnCountDisposition::Represented;
    let error = validate_local_parquet_source(&rejected_zero_column_count, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("invalid_zero_column_count_columns"));
    assert_eq!(error.fact_value("read_columns"), Some("1"));

    let mut rejected_schema_mapping = base.clone();
    rejected_schema_mapping.diagnostics.schema_mapping_disposition = SchemaMappingDisposition::Rejected;
    let error = validate_local_parquet_source(&rejected_schema_mapping, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("field"), Some("schema_mapping"));
    assert_eq!(error.fact_value("disposition"), Some("rejected"));

    let mut rejected_feature = base;
    rejected_feature.diagnostics.rejected_features.push(RejectedSourceFeature::RowSelection);
    let error = validate_local_parquet_source(&rejected_feature, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("rejected_source_features"));
    assert_eq!(error.fact_value("rejected_features"), Some("row_selection"));
}

#[test]
fn local_parquet_source_validation_rejects_inconsistent_column_and_row_group_contracts() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let two_column_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::UInt32, false),
    ]));

    let mut missing_projected_read =
        LocalParquetSource::new("/tmp/source-contract.parquet", two_column_schema, vec!["key".to_string()])
            .with_column_roles(vec!["payload".to_string()], Vec::new());
    missing_projected_read.diagnostics.rejected_features.clear();
    missing_projected_read.diagnostics.direct_device_read = true;
    missing_projected_read.diagnostics.schema_mapping_disposition = SchemaMappingDisposition::DirectByName;
    missing_projected_read.projected_output_column_source_refs.push(SourceColumnRef::File(1));
    let error = validate_local_parquet_source(&missing_projected_read, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("projected_column_not_available"));
    assert_eq!(error.fact_value("column"), Some("payload"));

    let mut mismatched_diagnostics =
        LocalParquetSource::new("/tmp/source-contract.parquet", Arc::clone(&schema), vec!["key".to_string()]);
    mismatched_diagnostics.diagnostics.read_column_count = 2;
    let error = validate_local_parquet_source(&mismatched_diagnostics, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("diagnostic_column_count_mismatch"));
    assert_eq!(error.fact_value("diagnostic_read_columns"), Some("2"));
    assert_eq!(error.fact_value("read_columns"), Some("1"));

    let missing_schema_mapping =
        LocalParquetSource::new("/tmp/source-contract.parquet", Arc::clone(&schema), vec!["missing".to_string()]);
    assert_eq!(missing_schema_mapping.diagnostics.schema_mapping_disposition, SchemaMappingDisposition::Rejected);
    let error = validate_local_parquet_source(&missing_schema_mapping, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("rejected_source_features"));
    assert_eq!(error.fact_value("rejected_features"), Some("schema_mapping"));

    let row_group_outside_contract = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/source-contract.parquet")
                .with_row_groups(vec![0, 2])
                .with_row_group_bytes(vec![100, 200]),
        ],
        schema,
        vec!["key".to_string()],
    );
    let error = validate_local_parquet_source(&row_group_outside_contract, rapids_analysis_context()).unwrap_err();
    assert_eq!(error.fact_value("reason"), None);
    assert_eq!(error.fact_value("row_group"), Some("2"));
    assert_eq!(error.fact_value("max_row_group"), Some("1"));
}
