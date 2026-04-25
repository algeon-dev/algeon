#[tokio::test]
async fn iceberg_provider_accepts_inexact_filter_without_losing_filter_columns() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t where id = 1").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(planner.requests().len(), 1);
    assert_eq!(planner.requests()[0].scan_limit, None);
    let request_predicate = format!("{:?}", planner.requests()[0].predicate);
    assert!(request_predicate.contains("Comparison"), "{request_predicate}");
    assert!(request_predicate.contains("column: \"id\""), "{request_predicate}");
    assert!(request_predicate.contains("op: Eq"), "{request_predicate}");
    assert!(request_predicate.contains("Int64(1)"), "{request_predicate}");
    assert!(
        planner.requests()[0]
            .projected_column_names
            .as_ref()
            .is_some_and(|columns| { columns.contains(&"id".to_string()) && columns.contains(&"value".to_string()) })
    );
    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, None, "the source must not apply LIMIT before the inexact filter");
    assert!(source.read_columns.contains(&"id".to_string()));
    assert!(source.read_columns.contains(&"value".to_string()));
    assert!(source.reader_predicate.is_some());
    assert_eq!(source.diagnostics.predicate_disposition.as_str(), "reader_pushdown");
}

#[tokio::test]
async fn iceberg_provider_keeps_filter_native_when_position_deletes_are_present() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(position_delete_projection_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t where id = 1").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].position_delete_files.len(), 1);
    assert!(source.reader_predicate.is_none());
    assert_eq!(source.diagnostics.predicate_disposition.as_str(), "absent");
    assert!(
        native_plan_filter_count(&result.plans[0]) >= 1,
        "the native filter must run after position deletes so row ordinals remain file-global"
    );
}

#[tokio::test]
async fn iceberg_provider_pushes_supported_filter_conjuncts_and_keeps_native_filter_for_residual() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t where id = 1 and value not like 'bad%'").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    let reader_predicate = source
        .reader_predicate
        .as_ref()
        .expect("supported conjunct should still push into the table-format reader")
        .to_string();
    assert!(reader_predicate.contains("id = 1"), "{reader_predicate}");
    assert!(!reader_predicate.contains("LIKE"), "{reader_predicate}");
    assert!(
        native_plan_filter_count(&result.plans[0]) >= 1,
        "the native filter must remain because NOT LIKE was only a residual expression"
    );
}

#[tokio::test]
async fn iceberg_provider_keeps_native_filter_for_temporal_table_format_predicate() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ship_date", DataType::Date32, false),
        Field::new("value", DataType::Utf8, true),
    ]));
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(Arc::clone(&schema))
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(11),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "ship_date",
            Some("ship_date"),
            DataType::Date32,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "value",
            Some("value"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let planner = Arc::new(RecordingPlanner::new(plan));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t where ship_date = date '2024-01-01'").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.read_columns.contains(&"ship_date".to_string()));
    assert!(source.reader_predicate.is_none());
    assert_eq!(source.diagnostics.predicate_disposition.as_str(), "absent");
    assert!(
        native_plan_filter_count(&result.plans[0]) >= 1,
        "temporal table-format predicates must remain represented as native filters"
    );
}

#[test]
fn native_lowering_rejects_iceberg_missing_predicate_column_above_hash_join() {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Int64, false),
        Field::new("profile_id", DataType::Int64, false),
    ]));
    let profile_schema = Arc::new(Schema::new(vec![
        Field::new("profile_id", DataType::Int64, false),
        Field::new("age", DataType::Int64, true),
    ]));
    let profile_footer_schema = Arc::new(Schema::new(vec![
        Field::new("profile_id", DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())])),
    ]));
    let edge_scan: Arc<dyn ExecutionPlan> = Arc::new(CudfTableFormatScanExec::new(
        native_admissible_scan_plan_for_schema(
            Arc::clone(&edge_schema),
            "edges-data-0001",
            "/tmp/iceberg/edges.parquet",
        ),
        Arc::clone(&edge_schema),
    ));
    let profile_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("glue")
        .with_logical_schema(Arc::clone(&profile_schema))
        .with_native_source_schema(Arc::clone(&profile_schema))
        .with_data_file(
            TableFormatDataFile::new(
                "profiles-data-0001",
                "s3://bucket/profiles/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(4096)
            .with_row_count(11)
            .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
                "s3://bucket/profiles/data-0001.parquet",
                CredentialMode::SharedProcessEnv,
            ))
            .with_footer_schema(profile_footer_schema),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "profile_id",
            Some("profile_id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "age",
            Some("age"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let profile_scan: Arc<dyn ExecutionPlan> =
        Arc::new(CudfTableFormatScanExec::new(profile_plan, Arc::clone(&profile_schema)));
    let edge_profile_id = col("profile_id", &edge_schema).unwrap();
    let profile_profile_id = col("profile_id", &profile_schema).unwrap();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            edge_scan,
            profile_scan,
            vec![(edge_profile_id, profile_profile_id)],
            None,
            &JoinType::Inner,
            Some(vec![0, 1, edge_schema.fields().len() + profile_schema.index_of("age").unwrap()]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );
    let join_schema = join.schema();
    let predicate =
        binary(col("age", &join_schema).unwrap(), Operator::Gt, lit(ScalarValue::Int64(Some(0))), &join_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, join).unwrap());

    let result = lower_physical_plan_to_native(filter);

    assert_eq!(result.rewritten_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(
        result.report.outcomes().iter().any(|outcome| {
            matches!(
                outcome,
                CandidateOutcome::Rejected { reason: FallbackReason::IcebergFieldIdMappingUnsupported, .. }
            )
        }),
        "predicate-required missing Iceberg field above a join must reject instead of materializing NULLs: {:?}",
        result.report.outcomes()
    );
}

#[test]
fn native_lowering_accepts_iceberg_same_name_signed_widened_predicate_column_above_hash_join() {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Int64, false),
        Field::new("profile_id", DataType::Int64, false),
    ]));
    let profile_schema = Arc::new(Schema::new(vec![
        Field::new("profile_id", DataType::Int64, false),
        Field::new("age", DataType::Int32, true),
    ]));
    let profile_footer_schema = Arc::new(Schema::new(vec![
        Field::new("profile_id", DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())])),
        Field::new("age", DataType::Int16, true),
    ]));
    let edge_scan: Arc<dyn ExecutionPlan> = Arc::new(CudfTableFormatScanExec::new(
        native_admissible_scan_plan_for_schema(
            Arc::clone(&edge_schema),
            "edges-data-0001",
            "/tmp/iceberg/edges.parquet",
        ),
        Arc::clone(&edge_schema),
    ));
    let profile_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("glue")
        .with_logical_schema(Arc::clone(&profile_schema))
        .with_native_source_schema(Arc::clone(&profile_schema))
        .with_data_file(
            TableFormatDataFile::new(
                "profiles-data-0001",
                "s3://bucket/profiles/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(4096)
            .with_row_count(11)
            .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
                "s3://bucket/profiles/data-0001.parquet",
                CredentialMode::SharedProcessEnv,
            ))
            .with_footer_schema(profile_footer_schema),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "profile_id",
            Some("profile_id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "age",
            Some("age"),
            DataType::Int32,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let profile_scan: Arc<dyn ExecutionPlan> =
        Arc::new(CudfTableFormatScanExec::new(profile_plan, Arc::clone(&profile_schema)));
    let edge_profile_id = col("profile_id", &edge_schema).unwrap();
    let profile_profile_id = col("profile_id", &profile_schema).unwrap();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            edge_scan,
            profile_scan,
            vec![(edge_profile_id, profile_profile_id)],
            None,
            &JoinType::Inner,
            Some(vec![0, 1, edge_schema.fields().len() + profile_schema.index_of("age").unwrap()]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );
    let join_schema = join.schema();
    let predicate =
        binary(col("age", &join_schema).unwrap(), Operator::Gt, lit(ScalarValue::Int32(Some(0))), &join_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, join).unwrap());

    let result = lower_physical_plan_to_native(filter);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    let sources = native_local_parquet_sources(&result.plans[0]);
    let profile_source = sources
        .into_iter()
        .find(|source| source.read_columns.contains(&"age".to_string()))
        .expect("profile source should read same-name widened age column");
    assert_eq!(profile_source.schema.field_with_name("age").unwrap().data_type(), &DataType::Int32);
    assert!(profile_source.reader_predicate.is_none());
    assert!(
        profile_source.files.iter().any(|file| {
            file.field_id_read_bindings.iter().any(|binding| {
                binding.logical_name == "age" && binding.physical_name == "age" && binding.data_type == DataType::Int16
            })
        }),
        "source should keep footer-proven physical age type for diagnostics/coercion"
    );
}
