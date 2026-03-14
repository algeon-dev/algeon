use super::*;

pub async fn run_from_env() -> SmokeResult<IcebergNativeSmokeReport> {
    run_report(IcebergNativeSmokeConfig::from_env()?).await
}

pub async fn run_report(config: IcebergNativeSmokeConfig) -> SmokeResult<IcebergNativeSmokeReport> {
    run_report_with_optional_stream(config, None).await
}

pub async fn run_report_streaming(
    config: IcebergNativeSmokeConfig,
    writer: &mut dyn Write,
) -> SmokeResult<IcebergNativeSmokeReport> {
    write_smoke_tsv_header(writer)?;
    run_report_with_optional_stream(config, Some(writer)).await
}

pub(crate) async fn run_report_with_optional_stream(
    config: IcebergNativeSmokeConfig,
    mut stream: Option<&mut dyn Write>,
) -> SmokeResult<IcebergNativeSmokeReport> {
    if config.custom_query_file.is_some() {
        return run_custom_subset_report(config, stream).await;
    }
    if !config.tpch_query_ids.is_empty() {
        let execution_mode = config.execution_mode;
        return run_tpch_subset_report_with_execution_mode(config, execution_mode, stream).await;
    }
    if !config.tpcds_query_ids.is_empty() {
        return run_tpcds_subset_report(config, stream).await;
    }
    eprintln!(
        "[native_iceberg_smoke_report] stage=iceberg_table_load_start fixture_mode={} table_ident={} credential_source={}",
        config.fixture.mode_code(),
        config.fixture.table_ident_display(),
        config.fixture.credential_source(),
    );
    let table = load_iceberg_table(&config.fixture).await?;
    eprintln!(
        "[native_iceberg_smoke_report] stage=iceberg_table_load_finish fixture_mode={} table_ident={}",
        config.fixture.mode_code(),
        config.fixture.table_ident_display(),
    );
    let file_io = table.file_io().clone();
    let catalog_type = config.fixture.catalog_type().to_string();
    let logical_schema = iceberg_table_arrow_schema(&table)?;
    let planner = IcebergTableFormatPlanner::new(table)
        .with_catalog_type(catalog_type.clone())
        .with_parquet_read_strategy_config(config.fixture.parquet_read_strategy_config());
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir = Arc::new(tempfile::Builder::new().prefix("datafusion-nexus-iceberg-stage-").tempdir()?);
    let planner = Arc::new(StagingTableFormatPlanner::new(
        planner,
        file_io,
        config.object_store_staging,
        Arc::clone(&staging_tempdir),
        Arc::clone(&staging_records),
        config.table_alias.clone(),
    ));

    let provider = Arc::new(
        CudfIcebergTableProvider::from_table_format_planner(logical_schema, planner)
            .with_catalog_type(catalog_type.clone()),
    );
    let (ctx, native_rule) =
        native_session_context(config.fixture.source_storage_profile(), NativeExecutionProfile::Smoke).await?;
    ctx.register_table(config.table_alias.as_str(), provider)?;
    let mut row = run_query_with_registered_native_tables(
        config.query_id.clone(),
        config.fixture.mode_code().to_string(),
        catalog_type,
        config.fixture.table_ident_display(),
        config.sql,
        &ctx,
        native_rule.as_ref(),
        Arc::clone(&staging_records),
        false,
        config.fixture.credential_source().to_string(),
        config.cpu_baseline_timeout_secs,
        CpuBaselineSessionProfile::Smoke,
        SmokeQueryExecutionMode::NativeOnly,
    )
    .await?;
    row.expected_count = staging_records_snapshot(&staging_records)
        .first()
        .and_then(|record| expected_count_from_scan_plan(&record.scan_plan));
    row.finalize_count_parity_if_needed();
    if let Some(writer) = stream.as_mut() {
        write_smoke_tsv_row(writer, &row)?;
    }
    Ok(IcebergNativeSmokeReport { rows: vec![row] })
}

#[allow(dead_code)]
pub async fn run_single_tpch_query_report_with_execution_mode(
    mut config: IcebergNativeSmokeConfig,
    query_id: impl Into<String>,
    execution_mode: SmokeQueryExecutionMode,
) -> SmokeResult<IcebergNativeSmokeReportRow> {
    config.tpch_query_ids = vec![query_id.into()];
    let mut report = run_tpch_subset_report_with_execution_mode(config, execution_mode, None).await?;
    let row = report.rows.pop().ok_or_else(|| smoke_error("single TPC-H Iceberg report produced no rows"))?;
    if !report.rows.is_empty() {
        return Err(smoke_error("single TPC-H Iceberg report produced multiple rows"));
    }
    Ok(row)
}

pub(crate) async fn run_tpch_subset_report_with_execution_mode(
    config: IcebergNativeSmokeConfig,
    execution_mode: SmokeQueryExecutionMode,
    mut stream: Option<&mut dyn Write>,
) -> SmokeResult<IcebergNativeSmokeReport> {
    if !matches!(config.fixture, IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }) {
        return Err(smoke_error(
            "TPC-H Iceberg smoke query ids require NEXUS_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
        ));
    };

    let catalog_type = config.fixture.catalog_type().to_string();
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir = Arc::new(tempfile::Builder::new().prefix("datafusion-nexus-iceberg-tpch-stage-").tempdir()?);
    let (ctx, native_rule) =
        native_session_context(config.fixture.source_storage_profile(), NativeExecutionProfile::TpchBenchmark).await?;
    // Footer-pruning object-store registration is best-effort: a failure must
    // leave queries running unpruned, never abort the run.
    if let Err(error) = register_footer_pruning_object_store(&ctx, &config).await {
        eprintln!(
            "[lakehouse_pruning] event=register_object_store_skipped error={}",
            stable_tsv_field(&error.to_string())
        );
    }

    for table_name in TPCH_TABLES {
        let table_fixture = config.fixture.with_tpch_table(table_name, &config.tpch_scale_factor)?;
        eprintln!(
            "[native_iceberg_smoke_report] stage=iceberg_table_load_start fixture_mode={} table_ident={} credential_source={}",
            table_fixture.mode_code(),
            table_fixture.table_ident_display(),
            table_fixture.credential_source(),
        );
        let table = load_iceberg_table(&table_fixture).await?;
        eprintln!(
            "[native_iceberg_smoke_report] stage=iceberg_table_load_finish fixture_mode={} table_ident={}",
            table_fixture.mode_code(),
            table_fixture.table_ident_display(),
        );
        let logical_schema = iceberg_table_arrow_schema(&table)?;
        let file_io = table.file_io().clone();
        let planner = IcebergTableFormatPlanner::new(table)
            .with_catalog_type(catalog_type.clone())
            .with_parquet_read_strategy_config(table_fixture.parquet_read_strategy_config());
        let staging_planner = Arc::new(StagingTableFormatPlanner::new(
            planner,
            file_io,
            config.object_store_staging,
            Arc::clone(&staging_tempdir),
            Arc::clone(&staging_records),
            (*table_name).to_string(),
        ));
        let provider = Arc::new(
            CudfIcebergTableProvider::from_table_format_planner(logical_schema, staging_planner)
                .with_catalog_type(catalog_type.clone()),
        );
        ctx.register_table(*table_name, provider)?;
    }

    let mut rows = Vec::new();
    for query_id in &config.tpch_query_ids {
        clear_staging_records(&staging_records);
        let query = TPCH_QUERIES
            .iter()
            .find(|query| query.id == query_id)
            .ok_or_else(|| smoke_error(format!("unknown TPC-H query id {query_id:?}")))?;
        let table_ident = config.fixture.tpch_all_tables_ident_display(&config.tpch_scale_factor);
        let query_id = query.id.to_string();
        let sql = query.sql.to_string();
        let row = match run_query_with_registered_native_tables(
            query_id.clone(),
            config.fixture.mode_code().to_string(),
            catalog_type.clone(),
            table_ident.clone(),
            sql.clone(),
            &ctx,
            native_rule.as_ref(),
            Arc::clone(&staging_records),
            execution_mode.runs_cpu_baseline(),
            config.fixture.credential_source().to_string(),
            config.cpu_baseline_timeout_secs,
            CpuBaselineSessionProfile::TpchTuned,
            execution_mode,
        )
        .await
        {
            Ok(row) => row,
            Err(error) if config.continue_on_query_error => {
                eprintln!(
                    "[native_iceberg_smoke_report] stage=query_error query_id={} error={}",
                    query_id,
                    stable_tsv_field(&error.to_string()),
                );
                query_error_report_row(
                    query_id,
                    config.fixture.mode_code().to_string(),
                    catalog_type.clone(),
                    table_ident,
                    sql,
                    &staging_records,
                    config.fixture.credential_source().to_string(),
                    error.as_ref(),
                )
            },
            Err(error) => return Err(error),
        };
        if let Some(writer) = stream.as_mut() {
            write_smoke_tsv_row(writer, &row)?;
        }
        rows.push(row);
    }
    Ok(IcebergNativeSmokeReport { rows })
}

pub(crate) async fn run_tpcds_subset_report(
    config: IcebergNativeSmokeConfig,
    mut stream: Option<&mut dyn Write>,
) -> SmokeResult<IcebergNativeSmokeReport> {
    if !matches!(config.fixture, IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }) {
        return Err(smoke_error(
            "TPC-DS Iceberg smoke query ids require NEXUS_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
        ));
    };

    let catalog_type = config.fixture.catalog_type().to_string();
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir = Arc::new(tempfile::Builder::new().prefix("datafusion-nexus-iceberg-tpcds-stage-").tempdir()?);
    let (ctx, native_rule) =
        native_session_context(config.fixture.source_storage_profile(), NativeExecutionProfile::TpcdsBenchmark).await?;
    // Footer-pruning object-store registration is best-effort: a failure must
    // leave queries running unpruned, never abort the run.
    if let Err(error) = register_footer_pruning_object_store(&ctx, &config).await {
        eprintln!(
            "[lakehouse_pruning] event=register_object_store_skipped error={}",
            stable_tsv_field(&error.to_string())
        );
    }

    for table_name in TPCDS_TABLES {
        let table_fixture = config.fixture.with_tpcds_table(table_name, &config.tpcds_scale_factor)?;
        eprintln!(
            "[native_iceberg_smoke_report] stage=iceberg_table_load_start fixture_mode={} table_ident={} credential_source={}",
            table_fixture.mode_code(),
            table_fixture.table_ident_display(),
            table_fixture.credential_source(),
        );
        let table = load_iceberg_table(&table_fixture).await?;
        eprintln!(
            "[native_iceberg_smoke_report] stage=iceberg_table_load_finish fixture_mode={} table_ident={}",
            table_fixture.mode_code(),
            table_fixture.table_ident_display(),
        );
        let logical_schema = iceberg_table_arrow_schema(&table)?;
        let file_io = table.file_io().clone();
        let planner = IcebergTableFormatPlanner::new(table)
            .with_catalog_type(catalog_type.clone())
            .with_parquet_read_strategy_config(table_fixture.parquet_read_strategy_config());
        let staging_planner = Arc::new(StagingTableFormatPlanner::new(
            planner,
            file_io,
            config.object_store_staging,
            Arc::clone(&staging_tempdir),
            Arc::clone(&staging_records),
            (*table_name).to_string(),
        ));
        let provider = Arc::new(
            CudfIcebergTableProvider::from_table_format_planner(logical_schema, staging_planner)
                .with_catalog_type(catalog_type.clone()),
        );
        ctx.register_table(*table_name, provider)?;
    }

    let mut rows = Vec::new();
    for query_id in &config.tpcds_query_ids {
        clear_staging_records(&staging_records);
        let query = TPCDS_QUERIES
            .iter()
            .find(|query| query.id == query_id)
            .ok_or_else(|| smoke_error(format!("unknown TPC-DS query id {query_id:?}")))?;
        let table_ident = config.fixture.tpcds_all_tables_ident_display(&config.tpcds_scale_factor);
        let query_id = query.id.to_string();
        let sql = query.sql.to_string();
        let row = match run_query_with_registered_native_tables(
            query_id.clone(),
            config.fixture.mode_code().to_string(),
            catalog_type.clone(),
            table_ident.clone(),
            sql.clone(),
            &ctx,
            native_rule.as_ref(),
            Arc::clone(&staging_records),
            config.execution_mode.runs_cpu_baseline(),
            config.fixture.credential_source().to_string(),
            config.cpu_baseline_timeout_secs,
            CpuBaselineSessionProfile::Smoke,
            config.execution_mode,
        )
        .await
        {
            Ok(row) => row,
            Err(error) if config.continue_on_query_error => {
                eprintln!(
                    "[native_iceberg_smoke_report] stage=query_error query_id={} error={}",
                    query_id,
                    stable_tsv_field(&error.to_string()),
                );
                query_error_report_row(
                    query_id,
                    config.fixture.mode_code().to_string(),
                    catalog_type.clone(),
                    table_ident,
                    sql,
                    &staging_records,
                    config.fixture.credential_source().to_string(),
                    error.as_ref(),
                )
            },
            Err(error) => return Err(error),
        };
        if let Some(writer) = stream.as_mut() {
            write_smoke_tsv_row(writer, &row)?;
        }
        rows.push(row);
    }
    Ok(IcebergNativeSmokeReport { rows })
}

pub(crate) async fn run_custom_subset_report(
    config: IcebergNativeSmokeConfig,
    mut stream: Option<&mut dyn Write>,
) -> SmokeResult<IcebergNativeSmokeReport> {
    if !matches!(config.fixture, IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }) {
        return Err(smoke_error(
            "custom Iceberg SQL probes require NEXUS_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
        ));
    };
    let workload = config
        .custom_workload
        .ok_or_else(|| smoke_error(format!("{CUSTOM_WORKLOAD_ENV} is required with {CUSTOM_QUERY_FILE_ENV}")))?;
    let query_file =
        config.custom_query_file.as_ref().ok_or_else(|| smoke_error(format!("{CUSTOM_QUERY_FILE_ENV} is required")))?;
    let probes = load_custom_query_manifest(query_file)?;

    let catalog_type = config.fixture.catalog_type().to_string();
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir =
        Arc::new(tempfile::Builder::new().prefix("datafusion-nexus-iceberg-custom-stage-").tempdir()?);
    let native_execution_profile = match workload {
        CustomWorkload::Tpch => NativeExecutionProfile::TpchBenchmark,
        CustomWorkload::Tpcds => NativeExecutionProfile::TpcdsBenchmark,
    };
    let (ctx, native_rule) =
        native_session_context(config.fixture.source_storage_profile(), native_execution_profile).await?;

    match workload {
        CustomWorkload::Tpch => {
            for table_name in TPCH_TABLES {
                let table_fixture = config.fixture.with_tpch_table(table_name, &config.tpch_scale_factor)?;
                register_custom_probe_table(
                    &ctx,
                    &table_fixture,
                    table_name,
                    &catalog_type,
                    config.object_store_staging,
                    Arc::clone(&staging_tempdir),
                    Arc::clone(&staging_records),
                )
                .await?;
            }
        },
        CustomWorkload::Tpcds => {
            if !matches!(
                config.fixture,
                IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }
            ) {
                return Err(smoke_error(
                    "custom TPC-DS probes require NEXUS_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
                ));
            }
            for table_name in TPCDS_TABLES {
                let table_fixture = config.fixture.with_tpcds_table(table_name, &config.tpcds_scale_factor)?;
                register_custom_probe_table(
                    &ctx,
                    &table_fixture,
                    table_name,
                    &catalog_type,
                    config.object_store_staging,
                    Arc::clone(&staging_tempdir),
                    Arc::clone(&staging_records),
                )
                .await?;
            }
        },
    }

    let table_ident = match workload {
        CustomWorkload::Tpch => config.fixture.tpch_all_tables_ident_display(&config.tpch_scale_factor),
        CustomWorkload::Tpcds => config.fixture.tpcds_all_tables_ident_display(&config.tpcds_scale_factor),
    };
    let mut rows = Vec::new();
    for probe in probes {
        clear_staging_records(&staging_records);
        let query_id = probe.id;
        let sql = probe.sql;
        let row = match run_query_with_registered_native_tables(
            query_id.clone(),
            config.fixture.mode_code().to_string(),
            catalog_type.clone(),
            table_ident.clone(),
            sql.clone(),
            &ctx,
            native_rule.as_ref(),
            Arc::clone(&staging_records),
            config.execution_mode.runs_cpu_baseline(),
            config.fixture.credential_source().to_string(),
            config.cpu_baseline_timeout_secs,
            match workload {
                CustomWorkload::Tpch => CpuBaselineSessionProfile::TpchTuned,
                CustomWorkload::Tpcds => CpuBaselineSessionProfile::Smoke,
            },
            config.execution_mode,
        )
        .await
        {
            Ok(row) => row,
            Err(error) if config.continue_on_query_error => {
                eprintln!(
                    "[native_iceberg_smoke_report] stage=query_error query_id={} error={}",
                    query_id,
                    stable_tsv_field(&error.to_string()),
                );
                query_error_report_row(
                    query_id,
                    config.fixture.mode_code().to_string(),
                    catalog_type.clone(),
                    table_ident.clone(),
                    sql,
                    &staging_records,
                    config.fixture.credential_source().to_string(),
                    error.as_ref(),
                )
            },
            Err(error) => return Err(error),
        };
        if let Some(writer) = stream.as_mut() {
            write_smoke_tsv_row(writer, &row)?;
        }
        rows.push(row);
    }
    Ok(IcebergNativeSmokeReport { rows })
}

pub(crate) async fn register_custom_probe_table(
    ctx: &SessionContext,
    table_fixture: &IcebergNativeSmokeFixture,
    table_name: &str,
    catalog_type: &str,
    object_store_staging: ObjectStoreStagingMode,
    staging_tempdir: Arc<tempfile::TempDir>,
    staging_records: Arc<Mutex<Vec<StagingRecord>>>,
) -> SmokeResult<()> {
    eprintln!(
        "[native_iceberg_smoke_report] stage=iceberg_table_load_start fixture_mode={} table_ident={} credential_source={}",
        table_fixture.mode_code(),
        table_fixture.table_ident_display(),
        table_fixture.credential_source(),
    );
    let table = load_iceberg_table(table_fixture).await?;
    eprintln!(
        "[native_iceberg_smoke_report] stage=iceberg_table_load_finish fixture_mode={} table_ident={}",
        table_fixture.mode_code(),
        table_fixture.table_ident_display(),
    );
    let logical_schema = iceberg_table_arrow_schema(&table)?;
    let file_io = table.file_io().clone();
    let planner = IcebergTableFormatPlanner::new(table)
        .with_catalog_type(catalog_type.to_string())
        .with_parquet_read_strategy_config(table_fixture.parquet_read_strategy_config());
    let staging_planner = Arc::new(StagingTableFormatPlanner::new(
        planner,
        file_io,
        object_store_staging,
        staging_tempdir,
        staging_records,
        table_name.to_string(),
    ));
    let provider = Arc::new(
        CudfIcebergTableProvider::from_table_format_planner(logical_schema, staging_planner)
            .with_catalog_type(catalog_type.to_string()),
    );
    ctx.register_table(table_name, provider)?;
    Ok(())
}
pub(crate) fn query_error_report_row(
    query_id: String,
    fixture_mode: String,
    catalog_type: String,
    table_ident: String,
    sql: String,
    staging_records: &Arc<Mutex<Vec<StagingRecord>>>,
    credential_source: String,
    error: &(dyn StdError + Send + Sync),
) -> IcebergNativeSmokeReportRow {
    let records = staging_records_snapshot(staging_records);
    let mut row = IcebergNativeSmokeReportRow::new(
        query_id,
        fixture_mode,
        catalog_type,
        table_ident,
        sql,
        aggregate_scan_diagnostics(&records),
        NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
        Some("query_error".to_string()),
        NativeDataFusionPostNativeExecutionPath::Cpu,
        None,
        aggregate_staging_summary(&records),
        credential_source,
    );
    row.parity_status = ParityStatus::NativeRejected;
    row.parity_detail = Some(stable_tsv_field(&error.to_string()));
    row
}

pub(crate) async fn run_query_with_registered_native_tables(
    query_id: String,
    fixture_mode: String,
    catalog_type: String,
    table_ident: String,
    sql: String,
    ctx: &SessionContext,
    native_rule: &CudfNativePhysicalOptimizerRule,
    staging_records: Arc<Mutex<Vec<StagingRecord>>>,
    run_cpu_parity: bool,
    credential_source: String,
    cpu_baseline_timeout_secs: Option<u64>,
    cpu_baseline_session_profile: CpuBaselineSessionProfile,
    execution_mode: SmokeQueryExecutionMode,
) -> SmokeResult<IcebergNativeSmokeReportRow> {
    eprintln!(
        "[native_iceberg_smoke_report] stage=query_planning_start query_id={} run_cpu_parity={} credential_source={}",
        query_id, run_cpu_parity, credential_source,
    );
    let df = ctx.sql(&sql).await?;
    let task_ctx = df.task_ctx();
    let mut physical_plan = df.create_physical_plan().await?;
    if iceberg_footer_pruning_enabled() {
        let scan_records = staging_records_snapshot(&staging_records);
        if let Err(error) = register_footer_pruning_object_stores_for_scan_records(ctx, &scan_records).await {
            eprintln!(
                "[lakehouse_pruning] event=register_scan_object_stores_skipped error={}",
                stable_tsv_field(&error.to_string())
            );
        }
        let unpruned_plan = Arc::clone(&physical_plan);
        physical_plan = datafusion_nexus::native::lakehouse_pruning::prune_table_format_native_plan(
            physical_plan,
            &task_ctx.runtime_env(),
        )
        .await
        .unwrap_or(unpruned_plan);
    }
    let planning_row = datafusion_nexus::NativeDataFusionPlanningReport::from_retained_rule(
        query_id.clone(),
        Arc::clone(&physical_plan),
        native_rule,
    )
    .rows
    .into_iter()
    .next()
    .ok_or_else(|| smoke_error("native Iceberg smoke report produced no planning rows"))?;
    if smoke_native_plan_debug_enabled() {
        log_native_plan_debug_row(&query_id, &planning_row, physical_plan.as_ref());
    }
    let records = staging_records_snapshot(&staging_records);
    let diagnostics = aggregate_scan_diagnostics(&records);
    let staging_summary = aggregate_staging_summary(&records);
    let mut row = IcebergNativeSmokeReportRow::new(
        query_id,
        fixture_mode,
        catalog_type,
        table_ident,
        sql.clone(),
        diagnostics,
        planning_row.outcome,
        planning_row.reason_code.as_deref().and_then(FallbackReason::from_code).map(|reason| reason.code().to_string()),
        planning_row.post_native_execution_path,
        None,
        staging_summary,
        credential_source,
    );

    let should_execute_native = planning_row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced
        && planning_row.post_native_execution_path == NativeDataFusionPostNativeExecutionPath::Native;
    eprintln!(
        "[native_iceberg_smoke_report] stage=query_planning_finish query_id={} native_planning_outcome={} post_native_execution_path={} planned_data_file_count={} native_source_read_capability={}",
        row.query_id,
        row.native_planning_outcome.as_str(),
        row.post_native_execution_path.as_str(),
        row.diagnostics.planned_data_file_count,
        empty_as_none(&row.diagnostics.native_source_read_capability),
    );
    eprintln!(
        "[native_iceberg_smoke_report] stage=native_optimizer_diagnostics query_id={} shared_subplan_considered_count={} shared_subplan_candidate_count={} shared_subplan_applied_count={} local_source_reuse_considered_count={} local_source_reuse_applied_count={}",
        row.query_id,
        planning_row.native_optimizer.shared_subplan_materialization.considered_count,
        planning_row.native_optimizer.shared_subplan_materialization.candidate_count,
        planning_row.native_optimizer.shared_subplan_materialization.applied_count,
        planning_row.native_optimizer.local_parquet_source_reuse.considered_count,
        planning_row.native_optimizer.local_parquet_source_reuse.applied_count,
    );
    let mut native_batches = None;
    let native_schema = physical_plan.schema();
    if execution_mode.runs_native() && should_execute_native {
        row.native_runtime_started = true;
        eprintln!(
            "[native_iceberg_smoke_report] stage=native_runtime_start query_id={} native_source_read_capability={} direct_device_read_disposition={}",
            row.query_id,
            empty_as_none(&row.diagnostics.native_source_read_capability),
            row.diagnostics.direct_device_read_disposition.code(),
        );
        let started = Instant::now();
        match collect(Arc::clone(&physical_plan), task_ctx.into()).await {
            Ok(batches) => {
                row.native_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
                row.native_output_batches = Some(batches.len());
                row.native_output_rows = Some(batches.iter().map(RecordBatch::num_rows).sum());
                row.native_count_value = count_value_from_batches(&batches);
                let fingerprint = fingerprint_output_batches(native_schema.as_ref(), &batches, &sql);
                row.native_output_fingerprint = Some(fingerprint.native_fingerprint);
                row.comparison_mode = fingerprint.mode;
                native_batches = Some(batches);
                eprintln!(
                    "[native_iceberg_smoke_report] stage=native_runtime_finish query_id={} outcome={} elapsed_ms={} output_rows={} output_batches={}",
                    row.query_id,
                    row.native_runtime_outcome.as_str(),
                    row.native_runtime_elapsed_ms.unwrap_or(0),
                    row.native_output_rows.unwrap_or(0),
                    row.native_output_batches.unwrap_or(0),
                );
            },
            Err(error) => {
                row.native_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Failed;
                row.parity_detail = Some(stable_tsv_field(&error.to_string()));
                eprintln!(
                    "[native_iceberg_smoke_report] stage=native_runtime_finish query_id={} outcome={} elapsed_ms={} error={}",
                    row.query_id,
                    row.native_runtime_outcome.as_str(),
                    row.native_runtime_elapsed_ms.unwrap_or(0),
                    stable_tsv_field(&error.to_string()),
                );
            },
        }
        if let Some(metrics) = physical_plan.metrics() {
            let native_metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics);
            eprintln!(
                "[native_iceberg_smoke_report] stage=native_runtime_metrics query_id={} parquet_files={} row_groups_selected={} read_columns={} source_chunks_planned={} source_chunks_committed={} source_cache_hits={} source_cache_misses={} source_cache_inserts={} hash_join_build_count={} one_shot_join_count={} task_runtime_row_count={} task_operator_row_count={} aggregate_strategy_row_count={} join_strategy_row_count={} source_metadata_load_time_ms={} source_read_queue_wait_time_ms={} parquet_read_time_ms={} source_read_time_ms={} operator_time_ms={} chunk_operator_time_ms={} final_merge_time_ms={} final_sort_time_ms={} materialization_time_ms={}",
                row.query_id,
                native_metrics.parquet_files,
                native_metrics.row_groups_selected,
                native_metrics.read_columns,
                native_metrics.source_chunks_planned,
                native_metrics.source_chunks_committed,
                native_metrics.source_cache_hits,
                native_metrics.source_cache_misses,
                native_metrics.source_cache_inserts,
                native_metrics.hash_join_build_count,
                native_metrics.one_shot_join_count,
                native_metrics.task_runtime_rows.len(),
                native_metrics.task_operator_rows.len(),
                native_metrics.aggregate_strategy_rows.len(),
                native_metrics.join_strategy_rows.len(),
                metric_duration_ms_text(&metrics, "source_metadata_load_time"),
                metric_duration_ms_text(&metrics, "source_read_queue_wait_time"),
                metric_duration_ms_text(&metrics, "parquet_read_time"),
                metric_duration_ms_text(&metrics, "source_read_time"),
                metric_duration_ms_text(&metrics, "operator_time"),
                metric_duration_ms_text(&metrics, "chunk_operator_time"),
                metric_duration_ms_text(&metrics, "final_merge_time"),
                metric_duration_ms_text(&metrics, "final_sort_time"),
                metric_duration_ms_text(&metrics, "materialization_time"),
            );
            eprintln!(
                "[native_iceberg_smoke_report] stage=native_task_operator_rows query_id={} rows={}",
                row.query_id,
                format_task_operator_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] stage=native_task_runtime_rows query_id={} rows={}",
                row.query_id,
                format_task_runtime_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] stage=native_source_read_runtime_rows query_id={} rows={}",
                row.query_id,
                format_source_read_runtime_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] stage=lakehouse_cache_fill_event_snapshot_rows query_id={} rows={}",
                row.query_id,
                format_lakehouse_cache_fill_event_snapshot_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] stage=native_aggregate_strategy_rows query_id={} rows={}",
                row.query_id,
                format_aggregate_strategy_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] stage=native_join_strategy_rows query_id={} rows={}",
                row.query_id,
                format_join_strategy_rows(&native_metrics),
            );
            row.record_native_metrics(native_metrics);
        }
    }

    if execution_mode.runs_cpu_baseline() && run_cpu_parity && should_execute_native {
        row.cpu_runtime_started = true;
        eprintln!(
            "[native_iceberg_smoke_report] stage=cpu_s3_baseline_start query_id={} table_count={} session_profile={}",
            row.query_id,
            records.iter().map(|record| record.table_name.as_str()).collect::<HashSet<_>>().len(),
            cpu_baseline_session_profile.code(),
        );
        let started = Instant::now();
        match collect_cpu_baseline_with_timeout(&sql, &records, cpu_baseline_timeout_secs, cpu_baseline_session_profile)
            .await
        {
            Ok((cpu_schema, cpu_batches)) => {
                row.cpu_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
                row.cpu_output_batches = Some(cpu_batches.len());
                row.cpu_output_rows = Some(cpu_batches.iter().map(RecordBatch::num_rows).sum());
                let cpu_fingerprint = fingerprint_output_batches(cpu_schema.as_ref(), &cpu_batches, &sql);
                row.cpu_output_fingerprint = Some(cpu_fingerprint.cpu_fingerprint);
                row.comparison_mode = cpu_fingerprint.mode;
                if let Some(native_batches) = native_batches.as_ref() {
                    eprintln!(
                        "[native_iceberg_smoke_report] stage=parity_collection_start query_id={} cpu_output_rows={} native_output_rows={}",
                        row.query_id,
                        row.cpu_output_rows.unwrap_or(0),
                        row.native_output_rows.unwrap_or(0),
                    );
                    let comparison = compare_record_batches(
                        cpu_schema.as_ref(),
                        &cpu_batches,
                        native_schema.as_ref(),
                        native_batches,
                        smoke_parity_uses_ordered_rows(&sql, cpu_schema.as_ref()),
                        ParityFloatTolerance::exact(),
                    );
                    row.parity_status = comparison.status;
                    row.parity_detail = comparison.detail;
                    row.cpu_output_fingerprint = Some(comparison.cpu_fingerprint);
                    row.native_output_fingerprint = Some(comparison.native_fingerprint);
                    row.comparison_mode = comparison.mode;
                    eprintln!(
                        "[native_iceberg_smoke_report] stage=parity_collection_finish query_id={} parity_status={} comparison_mode={}",
                        row.query_id,
                        row.parity_status.as_str(),
                        row.comparison_mode.as_str(),
                    );
                } else if row.native_runtime_outcome == NativeDataFusionRuntimeOutcome::Failed {
                    row.parity_status = ParityStatus::NativeFailed;
                }
                eprintln!(
                    "[native_iceberg_smoke_report] stage=cpu_s3_baseline_finish query_id={} outcome={} elapsed_ms={} output_rows={} output_batches={}",
                    row.query_id,
                    row.cpu_runtime_outcome.as_str(),
                    row.cpu_runtime_elapsed_ms.unwrap_or(0),
                    row.cpu_output_rows.unwrap_or(0),
                    row.cpu_output_batches.unwrap_or(0),
                );
            },
            Err(error) => {
                row.cpu_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Failed;
                let cpu_detail = stable_tsv_field(&error.to_string());
                if row.native_runtime_outcome == NativeDataFusionRuntimeOutcome::Failed {
                    row.parity_status = ParityStatus::NativeFailed;
                    if let Some(native_detail) = row.parity_detail.take() {
                        row.parity_detail =
                            Some(format!("native_failure={native_detail};cpu_baseline_failure={cpu_detail}"));
                    } else {
                        row.parity_detail = Some(cpu_detail);
                    }
                } else {
                    row.parity_status = ParityStatus::CpuFailed;
                    row.parity_detail = Some(cpu_detail);
                }
                eprintln!(
                    "[native_iceberg_smoke_report] stage=cpu_s3_baseline_finish query_id={} outcome={} elapsed_ms={} error={}",
                    row.query_id,
                    row.cpu_runtime_outcome.as_str(),
                    row.cpu_runtime_elapsed_ms.unwrap_or(0),
                    stable_tsv_field(&error.to_string()),
                );
            },
        }
    } else {
        row.finalize_count_parity_if_needed();
    }

    Ok(row)
}

fn iceberg_footer_pruning_enabled() -> bool {
    std::env::var("NEXUS_ICEBERG_FOOTER_PRUNING")
        .ok()
        .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true"))
}

fn smoke_native_plan_debug_enabled() -> bool {
    std::env::var(DEBUG_NATIVE_PLAN_ENV)
        .ok()
        .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
}

fn log_native_plan_debug_row(
    query_id: &str,
    planning_row: &datafusion_nexus::NativeDataFusionPlanningRow,
    post_native_plan: &dyn ExecutionPlan,
) {
    eprintln!(
        "[native_iceberg_smoke_report] stage=native_plan_debug query_id={} physical_root={} physical_shape={} native_candidate_shape={} native_candidate_shape_family={} outcome={} reason_code={} post_native_plan_root={} post_native_execution_path={} native_plan_fragment_count={} host_datafusion_boundary_count={}",
        query_id,
        stable_tsv_field(&planning_row.physical_root),
        stable_tsv_field(&planning_row.physical_shape),
        stable_tsv_field(planning_row.native_candidate_shape.as_deref().unwrap_or("none")),
        stable_tsv_field(planning_row.native_candidate_shape_family.as_deref().unwrap_or("none")),
        planning_row.outcome.as_str(),
        stable_tsv_field(planning_row.reason_code.as_deref().unwrap_or("none")),
        stable_tsv_field(&planning_row.post_native_plan_root),
        planning_row.post_native_execution_path.as_str(),
        planning_row.native_plan_fragment_count,
        planning_row.host_datafusion_boundary_count,
    );
    eprintln!(
        "[native_iceberg_smoke_report] stage=post_native_plan query_id={}\n{}",
        query_id,
        displayable(post_native_plan).indent(false)
    );
    let native_optimizer_rows = planning_row.native_optimizer.stable_rule_rows();
    eprintln!(
        "[native_iceberg_smoke_report] stage=native_optimizer_debug query_id={} rows={}",
        query_id,
        if native_optimizer_rows.is_empty() {
            "none".to_string()
        } else {
            stable_tsv_field(&native_optimizer_rows.join(";"))
        }
    );
}

/// Registers a DataFusion object store for the S3 warehouse so footer-only
/// pruning (`prune_table_format_native_plan`) can read Parquet footers/row-group
/// ranges over the object store. The Iceberg catalog reads data files through
/// its own FileIO, so without this the runtime env has no S3 store registered.
/// No-op unless footer pruning is enabled and the fixture is S3-backed.
async fn register_footer_pruning_object_store(
    ctx: &SessionContext,
    config: &IcebergNativeSmokeConfig,
) -> SmokeResult<()> {
    if !iceberg_footer_pruning_enabled() {
        return Ok(());
    }
    let Some(warehouse) = config.fixture.warehouse() else {
        return Ok(());
    };
    if !warehouse.starts_with("s3://") {
        return Ok(());
    }
    let store = super::cpu_baseline::build_cpu_s3_object_store(warehouse).await?;
    let store_url = datafusion::datasource::listing::ListingTableUrl::parse(warehouse)
        .map_err(|error| smoke_error(format!("invalid warehouse url for footer pruning: {error}")))?
        .object_store();
    ctx.register_object_store(store_url.as_ref(), store);
    eprintln!("[lakehouse_pruning] event=register_object_store source=warehouse_configured count=1");
    Ok(())
}

async fn register_footer_pruning_object_stores_for_scan_records(
    ctx: &SessionContext,
    records: &[StagingRecord],
) -> SmokeResult<()> {
    if !iceberg_footer_pruning_enabled() {
        return Ok(());
    }
    let mut store_urls = BTreeSet::new();
    for record in records {
        for file in &record.scan_plan.data_files {
            if let Some(store_url) = s3_store_url_for_location(&file.location) {
                store_urls.insert(store_url);
            }
        }
    }
    for store_url in &store_urls {
        let store = super::cpu_baseline::build_cpu_s3_object_store(store_url).await?;
        let object_store_url = ListingTableUrl::parse(store_url)
            .map_err(|error| smoke_error(format!("invalid scan object-store url for footer pruning: {error}")))?
            .object_store();
        ctx.register_object_store(object_store_url.as_ref(), store);
    }
    if !store_urls.is_empty() {
        eprintln!("[lakehouse_pruning] event=register_scan_object_stores count={}", store_urls.len());
    }
    Ok(())
}

fn s3_store_url_for_location(location: &str) -> Option<String> {
    let rest = location.strip_prefix("s3://")?;
    let bucket = rest.split('/').next()?.trim();
    (!bucket.is_empty()).then(|| format!("s3://{bucket}"))
}
