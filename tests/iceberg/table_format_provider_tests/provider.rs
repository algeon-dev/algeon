#[tokio::test]
async fn iceberg_provider_plans_structured_table_format_scan_leaf() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new();
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();

    let scan = find_table_format_scan(physical_plan.as_ref()).expect("plan should contain table-format scan leaf");
    assert_eq!(scan.schema().fields().len(), 1);
    assert_eq!(scan.schema().field(0).name(), "value");
    assert_eq!(scan.scan_plan().expect("native scan facts should be available").format, TableFormat::Iceberg);
    assert_eq!(planner.requests().len(), 1);
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&["value".to_string()][..]));

    assert_eq!(scan.name(), "NexusIcebergScanExec");

    let lowering = lower_physical_plan_to_native(Arc::clone(&physical_plan));
    assert_eq!(lowering.selected_count(), 1);
    let source = native_local_parquet_source(&lowering.plans[0]);
    assert_eq!(source.read_columns, vec!["value".to_string()]);
    assert_eq!(source.projected_output_columns, vec!["value".to_string()]);
}

#[tokio::test]
async fn real_iceberg_provider_exposes_one_canonical_schema_to_cpu_and_native_optimizer() {
    let fixture = RealIcebergProviderFixture::new();
    fixture.write_one_row_and_current_manifest_list().await;
    let provider = Arc::new(NexusIcebergTableProvider::try_new(fixture.table.clone()).unwrap());
    assert!(provider.schema().fields().iter().all(|field| field.metadata().is_empty()));

    let cpu_ctx = SessionContext::new_with_config(direct_friendly_session_config());
    let projection = vec![1];
    let physical_plan = provider.scan(&cpu_ctx.state(), Some(&projection), &[], None).await.unwrap();
    let scan = physical_plan
        .downcast_ref::<NexusIcebergScanExec>()
        .expect("real provider must return the unified Iceberg wrapper");
    assert_eq!(
        scan.cpu_delegate().schema().field(0).metadata().get(PARQUET_FIELD_ID_META_KEY).map(String::as_str),
        Some("2"),
        "the regression fixture must exercise the real iceberg-datafusion field-id schema",
    );
    assert!(scan.schema().field(0).metadata().is_empty());
    assert_eq!(scan.schema().as_ref(), project_schema(&provider.schema(), Some(&projection)).unwrap().as_ref());
    assert_eq!(
        scan.properties().equivalence_properties().schema().as_ref(),
        scan.schema().as_ref(),
        "wrapper equivalences must use the same canonical schema exposed by ExecutionPlan",
    );
    assert_eq!(
        scan.properties().output_partitioning().partition_count(),
        scan.cpu_delegate().properties().output_partitioning().partition_count(),
    );
    assert_eq!(scan.properties().boundedness, scan.cpu_delegate().properties().boundedness);
    assert_eq!(scan.properties().emission_type, scan.cpu_delegate().properties().emission_type);
    assert_eq!(scan.properties().evaluation_type, scan.cpu_delegate().properties().evaluation_type);
    assert_eq!(scan.properties().scheduling_type, scan.cpu_delegate().properties().scheduling_type);

    let batches = collect(Arc::clone(&physical_plan), Arc::new(TaskContext::default()))
        .await
        .expect("real iceberg-datafusion CPU delegate must execute through the canonical wrapper");
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema().as_ref(), scan.schema().as_ref());
    assert!(batches[0].schema().field(0).metadata().is_empty());
    assert_eq!(batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().values(), &[200]);

    let state = SessionStateBuilder::new()
        .with_config(direct_friendly_session_config())
        .with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
        .build();
    let native_ctx = SessionContext::new_with_state(state);
    native_ctx.register_table("real_iceberg", provider).unwrap();
    let native_plan = native_ctx
        .sql("select y from real_iceberg")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .expect("native optimizer schema validation must accept the real Iceberg provider");
    assert!(plan_contains_name(native_plan.as_ref(), "NexusNativeExec"));
    assert!(native_plan.schema().field(0).metadata().is_empty());
}

#[cfg(feature = "cuvs")]
fn cuvs_iceberg_session() -> (NexusGpuBackend, SessionContext) {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 512 * 1024 * 1024,
            max_source_chunk_bytes: 256 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .build()
        .expect("cuVS Iceberg tests require a backend-owned GPU");
    let state = backend
        .install_on(SessionStateBuilder::new().with_config(direct_friendly_session_config()).with_default_features())
        .expect("install backend admission on the cuVS Iceberg session")
        .try_with_cuvs_sql()
        .expect("install cuVS SQL")
        .build();
    (backend, SessionContext::new_with_state(state))
}

#[cfg(feature = "cuvs")]
fn close_cuvs_iceberg_session(backend: NexusGpuBackend, ctx: SessionContext) {
    drop(ctx);
    let close = backend.close_and_wait(Duration::from_secs(10)).expect("drain cuVS Iceberg backend");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[cfg(feature = "cuvs")]
#[tokio::test]
async fn cuvs_sql_plans_a_real_iceberg_relation_then_rejects_bounded_execution() {
    let fixture = RealIcebergProviderFixture::new();
    fixture.write_one_row_and_current_manifest_list().await;
    let provider = Arc::new(NexusIcebergTableProvider::try_new(fixture.table.clone()).unwrap());
    let (backend, ctx) = cuvs_iceberg_session();
    ctx.register_table("vector_iceberg", provider).unwrap();

    let query_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let queries = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![
            Arc::new(Int64Array::from(vec![7_i64])) as ArrayRef,
            Arc::new(arrow::array::Float32Array::from(vec![200.0_f32])) as ArrayRef,
            Arc::new(arrow::array::Float32Array::from(vec![300.0_f32])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("vector_queries", Arc::new(MemTable::try_new(query_schema, vec![vec![queries]]).unwrap()))
        .unwrap();

    let options = r#"{
      "dataset":{"id":"item_id","vector":{"columns":["d0","d1"]}},
      "queries":{"id":"query_id","vector":{"columns":["d0","d1"]}},
      "k":1,
      "metric":"l2_expanded"
    }"#;
    let plan = ctx
        .sql(&format!(
            "SELECT query_id, neighbor_id \
             FROM cuvs_brute_force_knn(\
                 (SELECT x AS item_id, CAST(y AS REAL) AS d0, CAST(z AS REAL) AS d1 FROM vector_iceberg), \
                 (SELECT * FROM vector_queries), \
                 '{}')",
            options.replace('\'', "''")
        ))
        .await
        .expect("plan cuVS Iceberg query")
        .create_physical_plan()
        .await
        .expect("create cuVS Iceberg physical plan");
    assert!(plan_contains_name(plan.as_ref(), "CuvsAlgoExec"));
    assert_eq!(plan.schema().field(0).name(), "query_id");
    assert_eq!(plan.schema().field(1).name(), "neighbor_id");
    assert_cuvs_iceberg_bounded_rejection(plan, &ctx, "cuvs_brute_force_knn").await;
    close_cuvs_iceberg_session(backend, ctx);
}

#[cfg(feature = "cuvs")]
#[tokio::test]
async fn cuvs_sql_plans_uniform_float32_list_iceberg_vectors_then_rejects_bounded_execution() {
    let fixture = RealIcebergProviderFixture::new_float32_vector_list();
    fixture.write_float32_vector_list_rows_and_current_manifest_list(&[(100, &[0.0, 0.0]), (100, &[3.0, 3.0])]).await;
    let provider = Arc::new(NexusIcebergTableProvider::try_new(fixture.table.clone()).unwrap());
    let (backend, ctx) = cuvs_iceberg_session();
    ctx.register_table("vector_iceberg", provider).unwrap();

    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let query_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Int64, false),
        Field::new("embedding", DataType::List(Arc::clone(&item)), false),
    ]));
    let queries = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![
            Arc::new(Int64Array::from(vec![99_i64])) as ArrayRef,
            Arc::new(ListArray::new(
                item,
                OffsetBuffer::new(vec![0_i32, 2].into()),
                Arc::new(Float32Array::from(vec![0.0_f32, 0.0])),
                None,
            )) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("vector_queries", Arc::new(MemTable::try_new(query_schema, vec![vec![queries]]).unwrap()))
        .unwrap();

    let options = r#"{
      "dataset":{"id":"item_id","vector":{"column":"embedding"}},
      "queries":{"id":"query_id","vector":{"column":"embedding"}},
      "k":1,
      "metric":"l2_expanded"
    }"#;
    let plan = ctx
        .sql(&format!(
            "SELECT query_id, neighbor_id, rank \
             FROM cuvs_brute_force_knn(\
                 (SELECT x AS item_id, embedding FROM vector_iceberg), \
                 (SELECT * FROM vector_queries), \
                 '{}')",
            options.replace('\'', "''")
        ))
        .await
        .expect("plan cuVS Iceberg List<Float32> query")
        .create_physical_plan()
        .await
        .expect("create cuVS Iceberg List<Float32> physical plan");
    assert!(plan_contains_name(plan.as_ref(), "CuvsAlgoExec"));
    assert_eq!(plan.schema().field(0).name(), "query_id");
    assert_eq!(plan.schema().field(1).name(), "neighbor_id");
    assert_eq!(plan.schema().field(2).name(), "rank");
    assert_cuvs_iceberg_bounded_rejection(plan, &ctx, "cuvs_brute_force_knn").await;
    close_cuvs_iceberg_session(backend, ctx);
}

#[cfg(feature = "cuvs")]
#[tokio::test]
async fn cuvs_kmeans_and_pca_plan_over_real_iceberg_then_reject_bounded_execution() {
    let fixture = RealIcebergProviderFixture::new();
    fixture.write_clusterable_vector_rows_and_current_manifest_list().await;
    let provider = Arc::new(NexusIcebergTableProvider::try_new(fixture.table.clone()).unwrap());
    let (backend, ctx) = cuvs_iceberg_session();
    ctx.register_table("vector_iceberg", provider).unwrap();

    let relation = "SELECT x AS item_id, CAST(y AS REAL) AS d0, CAST(z AS REAL) AS d1 FROM vector_iceberg";
    let kmeans_options = r#"{
      "input":{"id":"item_id","vector":{"columns":["d0","d1"]}},
      "n_clusters":2,
      "init":"kmeans++"
    }"#;
    let kmeans = ctx
        .sql(&format!(
            "SELECT id, cluster_id FROM cuvs_kmeans(({relation}), '{}') ORDER BY row_ordinal",
            kmeans_options.replace('\'', "''")
        ))
        .await
        .expect("plan cuVS K-means over Iceberg")
        .create_physical_plan()
        .await
        .expect("create cuVS K-means Iceberg physical plan");
    assert!(plan_contains_name(kmeans.as_ref(), "CuvsAlgoExec"));
    assert_eq!(kmeans.schema().field(0).name(), "id");
    assert_eq!(kmeans.schema().field(1).name(), "cluster_id");
    assert_cuvs_iceberg_bounded_rejection(kmeans, &ctx, "cuvs_kmeans").await;

    let pca_options = r#"{
      "input":{"id":"item_id","vector":{"columns":["d0","d1"]}},
      "n_components":1
    }"#;
    let pca = ctx
        .sql(&format!(
            "SELECT id, pc_0 FROM cuvs_pca(({relation}), '{}') ORDER BY row_ordinal",
            pca_options.replace('\'', "''")
        ))
        .await
        .expect("plan cuVS PCA over Iceberg")
        .create_physical_plan()
        .await
        .expect("create cuVS PCA Iceberg physical plan");
    assert!(plan_contains_name(pca.as_ref(), "CuvsAlgoExec"));
    assert_eq!(pca.schema().field(0).name(), "id");
    assert_eq!(pca.schema().field(1).name(), "pc_0");
    assert_cuvs_iceberg_bounded_rejection(pca, &ctx, "cuvs_pca").await;
    close_cuvs_iceberg_session(backend, ctx);
}

#[cfg(feature = "cuvs")]
async fn assert_cuvs_iceberg_bounded_rejection(
    plan: Arc<dyn ExecutionPlan>,
    ctx: &SessionContext,
    expected_operation: &'static str,
) {
    let error = collect(plan, ctx.task_ctx()).await.expect_err("bounded cuVS Iceberg execution must fail closed");
    let typed = error
        .find_nexus_error()
        .expect("bounded cuVS Iceberg rejection must preserve the structured Nexus error");
    assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(typed.operation(), "datafusion_nexus.cuvs_sql.bounded_admission");
    assert_eq!(typed.facts().get("reason"), Some("blocked_lower_level_contract"));
    assert_eq!(typed.facts().get("execution_mode"), Some("strict_bounded"));
    assert_eq!(typed.facts().get("required_contract"), Some("cuvs_vector_algorithm_peak_memory_preflight"));
    assert_eq!(typed.facts().get("host_fallback"), Some("disabled"));
    assert_eq!(typed.facts().get("cuvs_operation"), Some(expected_operation));
}

#[tokio::test]
async fn real_nested_iceberg_provider_canonicalizes_cpu_batches_and_native_list_planning() {
    let fixture = RealIcebergProviderFixture::new_nested();
    fixture.write_one_nested_row_and_current_manifest_list().await;
    let provider = Arc::new(NexusIcebergTableProvider::try_new(fixture.table.clone()).unwrap());

    let cpu_ctx = SessionContext::new_with_config(direct_friendly_session_config());
    let projection = vec![1, 2, 3];
    let physical_plan = provider.scan(&cpu_ctx.state(), Some(&projection), &[], None).await.unwrap();
    let scan = physical_plan.downcast_ref::<NexusIcebergScanExec>().unwrap();
    let delegate_schema = scan.cpu_delegate().schema();
    let canonical_schema = scan.schema();
    assert!(delegate_schema.fields().iter().all(|field| !field.metadata().is_empty()));
    assert!(canonical_schema.fields().iter().all(|field| field.metadata().is_empty()));
    let DataType::List(delegate_element) = delegate_schema.field(0).data_type() else {
        panic!("CPU names field must remain a list");
    };
    let DataType::List(canonical_element) = canonical_schema.field(0).data_type() else {
        panic!("canonical names field must remain a list");
    };
    assert_eq!(delegate_element.metadata().get(PARQUET_FIELD_ID_META_KEY).map(String::as_str), Some("3"));
    assert!(canonical_element.metadata().is_empty());
    assert!(scan.properties().output_ordering().is_none());

    let batches = collect(Arc::clone(&physical_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[0].schema().as_ref(), scan.schema().as_ref());
    assert_eq!(batches[0].column(0).as_any().downcast_ref::<ListArray>().unwrap().value_length(0), 2);
    assert_eq!(batches[0].column(1).as_any().downcast_ref::<StructArray>().unwrap().num_columns(), 2);
    assert_eq!(batches[0].column(2).as_any().downcast_ref::<MapArray>().unwrap().value_length(0), 1);

    let state = SessionStateBuilder::new()
        .with_config(direct_friendly_session_config())
        .with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
        .build();
    let native_ctx = SessionContext::new_with_state(state);
    native_ctx.register_table("real_nested_iceberg", provider).unwrap();
    let native_plan = native_ctx
        .sql("select names from real_nested_iceberg")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .expect("native optimizer must replace the wrapper for supported string-list projection");
    assert!(plan_contains_name(native_plan.as_ref(), "NexusNativeExec"));
    assert!(!plan_contains_name(native_plan.as_ref(), "NexusIcebergScanExec"));
    assert!(native_plan.schema().field(0).metadata().is_empty());
}

#[tokio::test]
async fn iceberg_provider_unprojected_unsupported_map_field_does_not_block_native_projection() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("allRiskEvents", map_utf8_int32_type(), true),
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
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let planner = Arc::new(RecordingPlanner::new(plan));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select id from iceberg_t limit 1").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(planner.requests().len(), 1);
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&["id".to_string()][..]));
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["id".to_string()]);
    assert_eq!(source.projected_output_columns, vec!["id".to_string()]);
}

#[tokio::test]
async fn iceberg_provider_projected_unsupported_map_field_rejects_native_projection() {
    let map_type = map_utf8_int32_type();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("allRiskEvents", map_type.clone(), true),
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
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "allRiskEvents",
            Some("allRiskEvents"),
            map_type,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let planner = Arc::new(RecordingPlanner::new(plan));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select * from iceberg_t limit 1").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedType, .. }
    ));
}

#[tokio::test]
async fn iceberg_provider_propagates_inaccessible_source_resolution_error() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(FailingResolutionPlanner { code: ErrorCode::SourceAccessDenied });
    let provider = test_iceberg_provider(schema, planner);
    let ctx = SessionContext::new();

    let error = provider.scan(&ctx.state(), None, &[], None).await.expect_err("source resolution must be terminal");
    let adapter_error = adapter_error_from(&error);

    assert_eq!(adapter_error.code(), ErrorCode::SourceAccessDenied);
    assert_eq!(adapter_error.operation(), "datafusion_nexus.test.iceberg.resolve_scan");
}

#[tokio::test]
async fn iceberg_provider_propagates_corrupt_source_resolution_error() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(FailingResolutionPlanner { code: ErrorCode::SourceDependencyFailure });
    let provider = test_iceberg_provider(schema, planner);
    let ctx = SessionContext::new();

    let error = provider.scan(&ctx.state(), None, &[], None).await.expect_err("corrupt metadata must be terminal");
    let adapter_error = adapter_error_from(&error);

    assert_eq!(adapter_error.code(), ErrorCode::SourceDependencyFailure);
    assert_eq!(adapter_error.operation(), "datafusion_nexus.test.iceberg.resolve_scan");
}

#[tokio::test]
async fn iceberg_provider_native_probe_failure_keeps_cpu_delegate_executable() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(FailingNativeInspectionPlanner {
        plan: native_admissible_scan_plan(Arc::clone(&schema)),
        code: ErrorCode::SourceTemporaryFailure,
    });
    let provider = test_iceberg_provider(schema, planner);
    let ctx = SessionContext::new();

    let physical_plan = provider.scan(&ctx.state(), None, &[], None).await.expect("native probe failure is optional");
    let scan = physical_plan.downcast_ref::<NexusIcebergScanExec>().expect("provider returns the unified wrapper");
    let NexusIcebergNativeScanOutcome::InspectionFailed { resolved_scan, error } = scan.native_outcome() else {
        panic!("native probe failure must be retained as inspection evidence");
    };
    assert_eq!(error.code(), ErrorCode::SourceTemporaryFailure);
    assert_eq!(error.operation(), "datafusion_nexus.test.iceberg.inspect_native_scan");
    assert_eq!(resolved_scan.snapshot_id.as_deref(), Some("12345"));
    assert_eq!(resolved_scan.data_files.len(), 1);
    assert!(scan.scan_plan().is_none(), "failed inspection must not become an available native candidate");
    assert_eq!(scan.resolved_scan_plan().snapshot_id.as_deref(), Some("12345"));
    assert_eq!(scan.diagnostics().planned_data_file_count, 1);

    let lowering = lower_physical_plan_to_native(Arc::clone(&physical_plan));
    assert!(matches!(
        lowering.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::TableFormatPlanningFailed, .. }
    ));

    let batches = collect(physical_plan, Arc::new(TaskContext::default()))
        .await
        .expect("CPU delegate must execute after optional native probe failure");
    assert!(batches.is_empty());
}

#[tokio::test]
async fn resolved_native_capability_miss_survives_optional_inspection_failure() {
    let schema = iceberg_test_schema();
    let plan =
        native_admissible_scan_plan(Arc::clone(&schema)).with_unsupported_feature(TableFormatUnsupportedFeature::new(
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            "resolved delete-file semantics are unsupported natively",
        ));
    let inspection_count = Arc::new(AtomicUsize::new(0));
    let planner =
        Arc::new(FailingInspectionAfterNotSupportedPlanner { plan, inspection_count: Arc::clone(&inspection_count) });
    let provider = test_iceberg_provider(schema, planner);
    let ctx = SessionContext::new();

    let physical_plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
    let scan = physical_plan.downcast_ref::<NexusIcebergScanExec>().unwrap();
    let NexusIcebergNativeScanOutcome::NotSupported { scan_plan, reason } = scan.native_outcome() else {
        panic!("resolved capability evidence must remain NotSupported");
    };
    assert_eq!(*reason, CapabilityReason::IcebergDeleteFilesUnsupported);
    assert_eq!(scan_plan.snapshot_id.as_deref(), Some("12345"));
    assert_eq!(scan_plan.data_files.len(), 1);
    assert_eq!(inspection_count.load(AtomicOrdering::Relaxed), 1);

    let lowering = lower_physical_plan_to_native(Arc::clone(&physical_plan));
    assert!(matches!(
        lowering.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::IcebergDeleteFilesUnsupported, .. }
    ));
    let batches = collect(physical_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert!(batches.is_empty());
}

#[tokio::test]
async fn predicate_free_limit_inspection_can_clear_dropped_file_rejection() {
    let schema = iceberg_test_schema();
    let inspected_scan = native_admissible_scan_plan(Arc::clone(&schema)).with_scan_limit(Some(1));
    let mut resolved_scan = inspected_scan.clone();
    resolved_scan.data_files.push(
        TableFormatDataFile::new(
            "dropped-orc-file",
            "/tmp/iceberg/dropped.orc",
            TableFormatDataFileFormat::Orc,
            TableFormatReadCapability::local_file_system(),
        )
        .with_size_bytes(4096)
        .with_row_count(1),
    );
    resolved_scan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
        TableFormatUnsupportedFeatureCode::IcebergNonParquetDataFile,
        "dropped-orc-file",
        "the predicate-free LIMIT inspection must remove this dropped file rejection",
    ));
    assert_eq!(resolved_scan.first_not_supported_reason(), Some(CapabilityReason::IcebergNonParquetDataFile));

    let inspection_count = Arc::new(AtomicUsize::new(0));
    let planner = Arc::new(TrimmingInspectionPlanner {
        resolved_scan,
        inspected_scan,
        inspection_count: Arc::clone(&inspection_count),
    });
    let provider = test_iceberg_provider(schema, planner);
    let ctx = SessionContext::new();

    let physical_plan = provider.scan(&ctx.state(), None, &[], Some(1)).await.unwrap();
    let scan = physical_plan.downcast_ref::<NexusIcebergScanExec>().unwrap();
    let NexusIcebergNativeScanOutcome::Available(scan_plan) = scan.native_outcome() else {
        panic!("inspection-trimmed scan must become a native candidate: {:?}", scan.native_outcome());
    };
    assert_eq!(scan_plan.data_files.len(), 1);
    assert_eq!(scan_plan.data_files[0].id, "data-0001");
    assert!(scan_plan.unsupported_features.is_empty());
    assert_eq!(inspection_count.load(AtomicOrdering::Relaxed), 1);

    let lowering = lower_physical_plan_to_native(physical_plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
}

#[tokio::test]
async fn iceberg_provider_source_resolution_timeout_is_terminal() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(SleepingResolutionPlanner::new(Duration::from_millis(50)));
    let provider = test_iceberg_provider(schema, planner)
        .with_catalog_type("local")
        .with_scan_planning_timeout(Some(Duration::from_millis(1)));
    let ctx = SessionContext::new();

    let error =
        provider.scan(&ctx.state(), None, &[], None).await.expect_err("source resolution timeout must remain terminal");
    let adapter_error = adapter_error_from(&error);

    assert_eq!(adapter_error.code(), ErrorCode::PlanningTimeout);
    assert_eq!(adapter_error.fact_value("planning_timeout_stage"), Some("source_resolution_provider_timeout"));
    assert_eq!(adapter_error.fact_value("test_planner_fact"), Some("slow_resolution"));
}

#[tokio::test]
async fn iceberg_provider_scan_planning_timeout_preserves_cpu_delegate_and_structured_evidence() {
    let schema = iceberg_test_schema();
    let planner =
        Arc::new(SleepingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema)), Duration::from_millis(50)));
    let provider = test_iceberg_provider(Arc::clone(&schema), planner)
        .with_catalog_type("local")
        .with_scan_planning_timeout(Some(Duration::from_millis(1)));
    let ctx = SessionContext::new();

    let physical_plan = provider.scan(&ctx.state(), None, &[], None).await.expect("CPU delegate must survive timeout");
    let scan = physical_plan
        .downcast_ref::<NexusIcebergScanExec>()
        .expect("unified provider must return the Iceberg scan wrapper");
    let NexusIcebergNativeScanOutcome::InspectionFailed { error: adapter_error, .. } = scan.native_outcome() else {
        panic!("native timeout must be retained as InspectionFailed evidence");
    };

    assert_eq!(adapter_error.code(), ErrorCode::PlanningTimeout);
    assert_eq!(adapter_error.status(), ErrorStatus::Permanent);
    assert_eq!(adapter_error.fact_value("table_format"), Some("iceberg"));
    assert_eq!(adapter_error.fact_value("catalog_type"), Some("local"));
    assert_eq!(adapter_error.fact_value("planning_timeout"), Some("true"));
    assert_eq!(adapter_error.fact_value("planning_timeout_ms"), Some("1"));
    assert_eq!(adapter_error.fact_value("planning_timeout_stage"), Some("native_inspection_provider_timeout"));
    assert_eq!(adapter_error.fact_value("timeout_retryable"), Some("false"));
    assert_eq!(adapter_error.fact_value("timeout_budget_increase_required"), Some("true"));
    assert_eq!(adapter_error.fact_value("test_planner_fact"), Some("slow"));

    let lowering = lower_physical_plan_to_native(Arc::clone(&physical_plan));
    assert!(matches!(
        lowering.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::TableFormatPlanningFailed, .. }
    ));
    let batches = collect(physical_plan, Arc::new(TaskContext::default()))
        .await
        .expect("CPU delegate must execute after native inspection timeout");
    assert!(batches.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn iceberg_provider_scan_planning_timeout_keeps_cpu_delegate_without_waiting_for_busy_planner() {
    let schema = iceberg_test_schema();
    let planner =
        Arc::new(BusyPlanner::new(native_admissible_scan_plan(Arc::clone(&schema)), Duration::from_millis(30)));
    let provider = test_iceberg_provider(Arc::clone(&schema), planner)
        .with_catalog_type("local")
        .with_scan_planning_timeout(Some(Duration::from_millis(1)));
    let ctx = SessionContext::new();

    let started = Instant::now();
    let physical_plan = provider.scan(&ctx.state(), None, &[], None).await.expect("CPU delegate must survive timeout");
    let elapsed = started.elapsed();
    let scan = physical_plan
        .downcast_ref::<NexusIcebergScanExec>()
        .expect("unified provider must return the Iceberg scan wrapper");
    let NexusIcebergNativeScanOutcome::InspectionFailed { error: adapter_error, .. } = scan.native_outcome() else {
        panic!("native timeout must be retained as InspectionFailed evidence");
    };

    assert_eq!(adapter_error.code(), ErrorCode::PlanningTimeout);
    assert_eq!(adapter_error.fact_value("planning_timeout_stage"), Some("native_inspection_provider_timeout"));
    assert_eq!(adapter_error.fact_value("test_planner_fact"), Some("busy"));
    assert!(
        elapsed < Duration::from_millis(200),
        "provider timeout should not wait for the busy planner to finish; elapsed={elapsed:?}"
    );
    let batches = collect(physical_plan, Arc::new(TaskContext::default()))
        .await
        .expect("CPU delegate must execute after native inspection timeout");
    assert!(batches.is_empty());
}

#[tokio::test]
async fn iceberg_provider_plumbs_explicit_scan_limit_into_table_format_source() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new();
    let projection = vec![1];
    let physical_plan = provider.scan(&ctx.state(), Some(&projection), &[], Some(7)).await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(planner.requests().len(), 1);
    assert_eq!(planner.requests()[0].scan_limit, Some(7));
    assert_eq!(result.selected_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(7));
    assert_eq!(
        source.files[0].row_group_rows.as_deref(),
        None,
        "Iceberg file-level row counts must not synthesize per-row-group metadata"
    );
}

#[test]
fn native_lowering_pushes_root_global_limit_into_table_format_source_scan_limit() {
    let schema = iceberg_test_schema();
    let scan: Arc<dyn ExecutionPlan> =
        Arc::new(test_iceberg_scan_exec(native_admissible_scan_plan(Arc::clone(&schema)), schema));
    let limited: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(scan, 3, Some(7)));

    let result = lower_physical_plan_to_native(limited);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(10), "source reads must cover skip + fetch before the native Limit trims");
    assert_eq!(source.diagnostics.scan_limit_disposition.as_str(), "represented");
    assert!(native_plan_limit_count(&result.plans[0]) >= 1, "native Limit must still preserve root LIMIT semantics");
}
