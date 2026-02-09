use super::*;

pub(super) fn native_exec_leaf_for_report() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/native-execution-report-source-not-read.parquet",
        schema,
        vec!["key".to_string()],
    )));
    Arc::new(NativeCudfExec::try_new(builder.finish_with_host_sink(source), native_execution_config()).unwrap())
}

pub(super) fn composed_native_exec_for_report() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut producer_builder = QueryPlanBuilder::new();
    let producer_source = producer_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/native-composed-report-source-not-read.parquet",
        Arc::clone(&schema),
        vec!["key".to_string()],
    )));
    let producer = Arc::new(
        NativeCudfGpuProducerExec::try_new("fragment_input", producer_builder.finish_gpu_resident(producer_source))
            .unwrap(),
    );

    let mut consumer_builder = QueryPlanBuilder::new();
    let consumer_source = consumer_builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("fragment_input", Arc::clone(&schema))));
    Arc::new(
        NativeCudfComposedExec::try_new(
            consumer_builder.finish_with_host_sink(consumer_source),
            vec![producer],
            native_execution_config(),
        )
        .unwrap(),
    )
}

pub(super) fn native_execution_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1)
}

pub(super) fn accepted_native_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new("SortExec(fetch)(AggregateExec(LocalParquetScan))", 1));
    NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(LocalParquetScan))".to_string(),
        report,
        source_diagnostics: Vec::new(),
        strict_bounded_contract: Default::default(),
        adaptive_pipeline: Default::default(),
    }
}

pub(super) fn accepted_bounded_join_native_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new(
        "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))",
        1,
    ));
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))".to_string(),
        report,
        source_diagnostics: Vec::new(),
        strict_bounded_contract: accepted_strict_bounded_contract_diagnostics(),
        adaptive_pipeline: Default::default(),
    };
    diagnostics
}

pub(super) fn rejected_native_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_rejected(
        GpuCandidate::new("SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))", 1),
        datafusion_cudf::FallbackReason::NativeBoundedJoinUnsupported,
    );
    NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))".to_string(),
        report,
        source_diagnostics: Vec::new(),
        strict_bounded_contract: rejected_strict_bounded_contract_diagnostics(),
        adaptive_pipeline: Default::default(),
    }
}

pub(super) fn accepted_strict_bounded_contract_diagnostics() -> NativeStrictBoundedContractDiagnostics {
    let candidate =
        GpuCandidate::new("SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))", 1);
    let rows = vec![
        strict_bounded_gate_row(
            StrictBoundedContractGateId::Compiled(ReservationGateId::new(0)),
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::Edge(
                cudf_query_engine::pipeline::PipelineEdgeId::new(0),
            )),
            ReservationRequirementScope::Edge,
            ReservationKind::BuildSide,
            "join_build_input",
            "query_engine_build_side_source_reservation",
            ContractQuality::ExactPreflight,
            None,
        ),
        strict_bounded_gate_row(
            StrictBoundedContractGateId::Compiled(ReservationGateId::new(1)),
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::SinkBoundary),
            ReservationRequirementScope::Edge,
            ReservationKind::Materialization,
            "host_arrow_materialization",
            "query_engine_explicit_sink_materialization_reservation",
            ContractQuality::QueryEngineOwnedBoundary,
            None,
        ),
    ];
    NativeStrictBoundedContractDiagnostics {
        accepted_candidates: vec![NativeStrictBoundedAcceptedCandidateDiagnostics {
            candidate,
            report: StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, rows)
                .unwrap(),
            compiled_graph: NativeStrictBoundedCompiledGraphDiagnostics {
                task_count: 2,
                edge_count: 1,
                barrier_count: 0,
                resource_count: 2,
                gate_count: 2,
                sink_kind: "host_arrow_materialization".to_string(),
                sink_gate_count: 1,
            },
        }],
        rejected_candidates: Vec::new(),
        non_join_bounded_plan_count: 0,
        non_join_bounded_plan_reason: None,
    }
}

pub(super) fn rejected_strict_bounded_contract_diagnostics() -> NativeStrictBoundedContractDiagnostics {
    NativeStrictBoundedContractDiagnostics {
        accepted_candidates: Vec::new(),
        rejected_candidates: vec![NativeStrictBoundedRejectedCandidateDiagnostics {
            candidate: GpuCandidate::new(
                "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))",
                1,
            ),
            blocking_contract: StrictBoundedContractGateRow {
                execution_mode: StrictBoundedExecutionMode::UnsupportedStrictBounded,
                contract_scope: ReservationRequirementScope::Node,
                gate_id: StrictBoundedContractGateId::Compiled(ReservationGateId::new(0)),
                owner: StrictBoundedContractOwner::Compiled(ReservationGateOwner::Task(
                    cudf_query_engine::pipeline::PipelineTaskId::new(0),
                )),
                reservation_kind: ReservationKind::MixedSizeData,
                memory_boundary: "mixed_keyed_residual_or_conditional_full_retained_size_data",
                required_contract: "mixed_keyed_residual_or_conditional_full_retained_size_data",
                contract_status: ContractStatus::BlockedByLowerLevel,
                contract_quality: ContractQuality::ObservedRetainedState,
                unresolved_preflight_gap: None,
            },
        }],
        non_join_bounded_plan_count: 0,
        non_join_bounded_plan_reason: None,
    }
}

fn strict_bounded_gate_row(
    gate_id: StrictBoundedContractGateId,
    owner: StrictBoundedContractOwner,
    contract_scope: ReservationRequirementScope,
    reservation_kind: ReservationKind,
    memory_boundary: &'static str,
    required_contract: &'static str,
    contract_quality: ContractQuality,
    unresolved_preflight_gap: Option<&'static str>,
) -> StrictBoundedContractGateRow {
    StrictBoundedContractGateRow {
        execution_mode: StrictBoundedExecutionMode::StrictBounded,
        contract_scope,
        gate_id,
        owner,
        reservation_kind,
        memory_boundary,
        required_contract,
        contract_status: ContractStatus::Modeled,
        contract_quality,
        unresolved_preflight_gap,
    }
}

pub(super) fn native_execution_metrics_set() -> datafusion::physical_plan::metrics::MetricsSet {
    let metrics = ExecutionPlanMetricsSet::new();
    for (name, value) in [
        ("gpu_resident_input_handoffs", 1),
        ("host_to_device_uploads", 0),
        ("host_to_device_uploaded_rows", 0),
        ("host_to_device_uploaded_bytes", 0),
        ("device_to_host_materializations", 1),
        ("device_to_host_materialized_rows", 4),
        ("device_to_host_materialized_bytes", 256),
        ("device_to_device_copies", 2),
        ("device_to_device_copied_bytes", 512),
        ("parquet_files", 1),
        ("parquet_bytes_known", 1024),
        ("row_groups_selected", 2),
        ("read_columns", 4),
        ("resident_dataframe_peak_bytes", 4096),
        ("resident_dataframe_peak_bytes_observed", 1),
        ("allocator_high_water_bytes", 8192),
        ("allocator_high_water_bytes_observed", 1),
        ("reservation_requested_bytes", 128),
        ("reservation_granted_bytes", 128),
        ("reservation_denied_count", 0),
        ("source_chunks_planned", 6),
        ("source_chunks_attempted", 7),
        ("source_chunks_committed", 5),
        ("source_chunks_retried", 2),
        ("source_read_prefetches_started", 3),
        ("source_read_prefetches_completed", 2),
        ("source_read_prefetches_skipped", 1),
        ("source_read_prefetches_abandoned", 1),
        ("source_cache_hits", 4),
        ("source_cache_misses", 5),
        ("source_cache_inserts", 3),
        ("source_cache_insert_skipped", 2),
        ("source_cache_retained_bytes", 65536),
        ("row_groups_planned", 8),
        ("row_groups_committed", 4),
        ("partial_aggregate_count", 5),
        ("partial_aggregate_rows", 9),
        ("final_merge_attempts", 2),
        ("final_merge_levels", 1),
        ("retry_count", 3),
        ("retry_resource_exhausted_count", 1),
        ("retry_reservation_failure_count", 2),
        ("adaptive_retry_count", 10),
        ("adaptive_spill_count", 11),
        ("adaptive_downgrade_count", 12),
        ("adaptive_memory_history_lookup_count", 13),
        ("adaptive_memory_history_hit_count", 8),
        ("adaptive_memory_history_miss_count", 5),
        ("hash_join_build_count", 1),
        ("filtered_join_build_count", 6),
        ("one_shot_join_count", 7),
        ("hash_join_cache_hits", 2),
        ("hash_join_cache_misses", 1),
        ("hash_join_durable_cache_hits", 3),
        ("hash_join_durable_cache_misses", 4),
        ("hash_join_durable_cache_inserts", 5),
        ("hash_join_hash_table_bytes", 16384),
        ("filtered_join_hash_table_bytes", 8192),
        ("filtered_join_probe_state_bytes", 1024),
        ("mixed_join_size_data_bytes", 512),
        ("one_shot_join_index_bytes", 4096),
        ("one_shot_join_materialization_bytes", 8192),
        ("one_shot_join_output_bytes", 16384),
        ("join_gather_map_bytes", 2048),
    ] {
        MetricBuilder::new(&metrics).with_type(MetricType::SUMMARY).counter(name, 0).add(value);
    }
    MetricBuilder::new(&metrics)
        .with_type(MetricType::SUMMARY)
        .with_new_label("node_id", "n2")
        .with_new_label("join_strategy", "one_shot_equi")
        .with_new_label("execution_strategy", "bounded_one_shot_equi_join")
        .with_new_label("join_type", "inner")
        .with_new_label("build_side", "right")
        .with_new_label("key_count", "1")
        .with_new_label("estimated_build_rows", "100")
        .with_new_label("observed_build_rows", "101")
        .with_new_label("build_side_bytes", "1024")
        .with_new_label("probe_reuse_count", "1")
        .with_new_label("reusable_hash_join_rejected_reason", "build_side_too_large")
        .with_new_label("one_shot_equi_rejected_reason", "none")
        .with_new_label("hash_join_build_count", "0")
        .with_new_label("hash_join_hash_table_bytes", "0")
        .with_new_label("filtered_join_build_count", "0")
        .with_new_label("filtered_join_hash_table_bytes", "0")
        .with_new_label("filtered_join_probe_state_bytes", "0")
        .with_new_label("mixed_join_size_data_bytes", "0")
        .with_new_label("one_shot_join_count", "1")
        .with_new_label("one_shot_join_index_bytes", "2048")
        .with_new_label("one_shot_join_materialization_bytes", "4096")
        .with_new_label("one_shot_join_output_bytes", "8192")
        .with_new_label("join_gather_map_bytes", "2048")
        .global_counter("native_join_strategy_rows")
        .add(1);
    metrics.clone_inner()
}

pub(super) fn report_row<'a>(
    report: &'a NativeFallbackPressureReport,
    query_id: &str,
) -> &'a datafusion_cudf::NativeFallbackPressureRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

pub(super) fn planning_row<'a>(
    report: &'a NativeDataFusionPlanningReport,
    query_id: &str,
) -> &'a datafusion_cudf::NativeDataFusionPlanningRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

pub(super) async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(direct_friendly_sql_session_config());
    let df = ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn join_aggregate_sort_limit_sql_plan(
    left: &std::path::Path,
    right: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_repartition_file_scans(false)
            .with_repartition_joins(false)
            .with_repartition_sorts(false)
            .with_target_partitions(1),
    );
    ctx.register_parquet("left_input", left.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.register_parquet("right_input", right.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx
        .sql(
            "
SELECT l.key, SUM(l.m00 + r.m01) AS sum_x00
FROM left_input l
JOIN right_input r ON l.key = r.key
WHERE l.payload >= 10 AND r.payload >= 10
GROUP BY l.key
ORDER BY sum_x00 DESC, l.key ASC
LIMIT 2
",
        )
        .await
        .unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) fn typed_null_and_try_cast_reject_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let typed_null_comparison =
        binary(lit(ScalarValue::Int64(None)), Operator::Eq, lit(ScalarValue::Int64(None)), &scan_schema).unwrap();
    let try_cast = Arc::new(TryCastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int32));
    let try_cast_comparison =
        binary(try_cast, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema).unwrap();
    let predicate = binary(typed_null_comparison, Operator::And, try_cast_comparison, &scan_schema).unwrap();
    aggregate_sort_limit_plan_with_filter(scan, predicate)
}

pub(super) fn aggregate_sort_limit_plan_with_filter(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn datafusion::physical_expr::PhysicalExpr>,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());
    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: col("m00", &filter_schema).unwrap(), alias: "m00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_m00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_m00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(2)));
    Arc::new(LocalLimitExec::new(sort, 2))
}

pub(super) async fn native_datafusion_planning_report(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
) -> NativeDataFusionPlanningReport {
    native_datafusion_planning_report_with_session(query_id, path, sql, direct_friendly_sql_session_config()).await
}

pub(super) async fn native_datafusion_planning_report_with_session(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
    session_config: SessionConfig,
) -> NativeDataFusionPlanningReport {
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1));
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    NativeDataFusionPlanningReport::from_retained_rule(query_id, plan, native_rule.as_ref())
}

pub(super) fn direct_friendly_sql_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

pub(super) fn count_only_sql_session_config() -> SessionConfig {
    let mut config = direct_friendly_sql_session_config().with_collect_statistics(false);
    config.options_mut().execution.collect_statistics = false;
    config
}

pub(super) fn write_numeric_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
            Arc::new(Int64Array::from(vec![5, 10, 15, 20, 25, 30])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn copy_tiny_tpch_as_directory_layout(
    source: &std::path::Path,
    destination: &std::path::Path,
    file_names: &[&str],
) {
    for table in TPCH_TABLES {
        let table_dir = destination.join(table);
        fs::create_dir_all(&table_dir).unwrap();
        let source_file = source.join(format!("{table}.parquet"));
        for file_name in file_names {
            fs::copy(&source_file, table_dir.join(file_name)).unwrap();
        }
    }
}

pub(super) fn copy_tiny_tpch_as_sibling_split_layout(
    source: &std::path::Path,
    destination: &std::path::Path,
    split_tables: &[&str],
    split_count: usize,
) {
    assert!(split_count >= 1, "split_count must be >= 1");
    fs::create_dir_all(destination).unwrap();
    for table in TPCH_TABLES {
        let source_file = source.join(format!("{table}.parquet"));
        if split_tables.contains(table) {
            for shard in 0..split_count {
                let shard_name = format!("{table}_{shard:04}.parquet");
                fs::copy(&source_file, destination.join(shard_name)).unwrap();
            }
        } else {
            fs::copy(&source_file, destination.join(format!("{table}.parquet"))).unwrap();
        }
    }
}

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
