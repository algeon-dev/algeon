use super::*;

pub(crate) async fn collect_cpu_baseline(
    sql: &str,
    records: &[StagingRecord],
    session_profile: CpuBaselineSessionProfile,
) -> SmokeResult<(arrow::datatypes::SchemaRef, Vec<RecordBatch>)> {
    let ctx = SessionContext::new_with_config(cpu_baseline_session_config(session_profile));
    let mut seen_tables = HashSet::new();
    for record in records {
        if seen_tables.insert(record.table_name.clone()) {
            register_cpu_table_from_scan_plan(&ctx, &record.table_name, &record.scan_plan).await?;
        }
    }
    let df = ctx.sql(sql).await?;
    let task_ctx = df.task_ctx();
    let plan = df.create_physical_plan().await?;
    let schema = plan.schema();
    let batches = collect(plan, task_ctx.into()).await?;
    Ok((schema, batches))
}

pub(crate) async fn collect_cpu_baseline_with_timeout(
    sql: &str,
    records: &[StagingRecord],
    timeout_secs: Option<u64>,
    session_profile: CpuBaselineSessionProfile,
) -> SmokeResult<(arrow::datatypes::SchemaRef, Vec<RecordBatch>)> {
    let future = collect_cpu_baseline(sql, records, session_profile);
    match timeout_secs {
        Some(timeout_secs) => {
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), future).await.map_err(|_| {
                smoke_error(format!("timeout_category=cpu_s3_baseline_timeout;timeout_secs={timeout_secs}"))
            })?
        },
        None => future.await,
    }
}

pub fn cpu_baseline_session_config(profile: CpuBaselineSessionProfile) -> SessionConfig {
    match profile {
        CpuBaselineSessionProfile::Smoke => smoke_session_config(),
        CpuBaselineSessionProfile::TpchTuned => tpch_tuned_cpu_session_config(),
    }
}

pub fn smoke_parity_uses_ordered_rows(sql: &str, schema: &arrow::datatypes::Schema) -> bool {
    sql_has_top_level_order_by(sql) && sql_top_level_order_by_covers_output_columns(sql, schema)
}

pub fn fingerprint_output_batches(
    schema: &arrow::datatypes::Schema,
    batches: &[RecordBatch],
    sql: &str,
) -> datafusion_nexus::ParityComparisonResult {
    compare_record_batches(
        schema,
        batches,
        schema,
        batches,
        smoke_parity_uses_ordered_rows(sql, schema),
        ParityFloatTolerance::exact(),
    )
}

pub(crate) async fn register_cpu_table_from_scan_plan(
    ctx: &SessionContext,
    table_name: &str,
    scan_plan: &TableFormatScanPlan,
) -> SmokeResult<()> {
    let urls = cpu_baseline_listing_urls_from_scan_plan(scan_plan)?;
    if urls.is_empty() {
        return Err(smoke_error(format!(
            "no CPU-readable Parquet data files available for CPU baseline table {table_name}"
        )));
    }

    let url_count = urls.len();
    eprintln!(
        "[native_iceberg_smoke_report] stage=cpu_s3_baseline_registration_start table_name={} parquet_file_count={}",
        table_name, url_count,
    );
    register_cpu_object_stores(ctx, &urls).await?;
    let listing_options = ListingOptions::new(Arc::new(ParquetFormat::default()))
        .with_file_extension(cpu_baseline_listing_file_extension(&urls));
    let inferred_schema = listing_options.infer_schema(&ctx.state(), &urls[0]).await?;
    let config = ListingTableConfig::new_with_multi_paths(urls)
        .with_listing_options(listing_options)
        .with_schema(inferred_schema);
    ctx.register_table(table_name, Arc::new(ListingTable::try_new(config)?))?;
    eprintln!(
        "[native_iceberg_smoke_report] stage=cpu_s3_baseline_registration_finish table_name={} parquet_file_count={}",
        table_name, url_count,
    );
    Ok(())
}

pub fn cpu_baseline_listing_urls_from_scan_plan(scan_plan: &TableFormatScanPlan) -> SmokeResult<Vec<ListingTableUrl>> {
    scan_plan
        .data_files
        .iter()
        .filter(|file| file.format == TableFormatDataFileFormat::Parquet)
        .filter_map(cpu_baseline_listing_url_from_data_file)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Box::new(error) as Box<dyn StdError + Send + Sync>)
}

pub(crate) fn cpu_baseline_listing_url_from_data_file(
    file: &datafusion_nexus::TableFormatDataFile,
) -> Option<datafusion::error::Result<ListingTableUrl>> {
    if let Some(resolved) = &file.resolved_parquet_source
        && resolved.read_strategy == ParquetReadStrategy::LocalCacheKvikio
        && Path::new(&resolved.reader_location).exists()
    {
        return Some(ListingTableUrl::parse(resolved.reader_location.as_str()));
    }

    match &file.read_capability {
        capability if capability == &TableFormatReadCapability::local_file_system() => {
            Some(ListingTableUrl::parse(file.location.as_str()))
        },
        TableFormatReadCapability::ObjectStore { scheme } if scheme.eq_ignore_ascii_case("s3") => {
            Some(ListingTableUrl::parse(file.location.as_str()))
        },
        _ => None,
    }
}

pub fn cpu_baseline_listing_file_extension(urls: &[ListingTableUrl]) -> &'static str {
    if urls.iter().any(|url| url.file_extension() != Some("parquet")) { "" } else { ".parquet" }
}

pub(crate) async fn register_cpu_object_stores(ctx: &SessionContext, urls: &[ListingTableUrl]) -> SmokeResult<()> {
    let mut registered_s3_stores = HashSet::new();
    for url in urls.iter().filter(|url| url.scheme() == "s3") {
        let store_url = url.object_store();
        if registered_s3_stores.insert(store_url.as_str().to_string()) {
            let store = build_cpu_s3_object_store(url.as_str()).await?;
            ctx.register_object_store(store_url.as_ref(), Arc::new(store));
            ctx.runtime_env().object_store(store_url.clone()).map_err(|error| {
                smoke_error(format!("failed to register direct S3 CPU baseline object store: {error}"))
            })?;
            ctx.runtime_env().object_store(url).map_err(|error| {
                smoke_error(format!("failed to resolve direct S3 CPU baseline table URL after registration: {error}"))
            })?;
        }
    }
    Ok(())
}

pub(crate) async fn build_cpu_s3_object_store(s3_url: &str) -> SmokeResult<object_store::aws::AmazonS3> {
    let mut builder = AmazonS3Builder::from_env().with_url(s3_url.to_string());
    if let Ok(endpoint) = std::env::var(REST_S3_ENDPOINT_ENV)
        && !endpoint.trim().is_empty()
    {
        let endpoint = endpoint.trim().to_string();
        if endpoint.starts_with("http://") {
            builder = builder.with_allow_http(true);
        }
        builder = builder.with_endpoint(endpoint);
    }
    if let Ok(path_style) = std::env::var(REST_S3_PATH_STYLE_ENV)
        && matches!(path_style.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    {
        builder = builder.with_virtual_hosted_style_request(false);
    }
    if let Ok(region) = std::env::var(REST_S3_REGION_ENV)
        && !region.trim().is_empty()
    {
        builder = builder.with_region(region.trim());
    }
    if let (Ok(access_key_id), Ok(secret_access_key)) =
        (std::env::var(REST_S3_ACCESS_KEY_ID_ENV), std::env::var(REST_S3_SECRET_ACCESS_KEY_ENV))
        && !access_key_id.trim().is_empty()
        && !secret_access_key.trim().is_empty()
    {
        builder = builder.with_access_key_id(access_key_id.trim()).with_secret_access_key(secret_access_key.trim());
        if let Ok(session_token) = std::env::var(REST_S3_SESSION_TOKEN_ENV)
            && !session_token.trim().is_empty()
        {
            builder = builder.with_token(session_token.trim());
        }
        return Ok(builder.build()?);
    }

    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Ok(profile) = std::env::var(AWS_PROFILE_ENV)
        && !profile.trim().is_empty()
    {
        loader = loader.profile_name(profile);
    }
    let sdk_config = loader.load().await;
    if let Some(region) = sdk_config.region() {
        builder = builder.with_region(region.as_ref());
    }
    if let Some(credentials_provider) = sdk_config.credentials_provider() {
        let credentials = credentials_provider.provide_credentials().await.map_err(|error| {
            smoke_error(format!("failed to load AWS credentials for direct S3 CPU baseline: {error}"))
        })?;
        builder = builder
            .with_access_key_id(credentials.access_key_id())
            .with_secret_access_key(credentials.secret_access_key());
        if let Some(session_token) = credentials.session_token() {
            builder = builder.with_token(session_token);
        }
    }

    Ok(builder.build()?)
}
pub(crate) fn smoke_session_config() -> SessionConfig {
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

pub(crate) fn expected_count_from_scan_plan(plan: &TableFormatScanPlan) -> Option<u64> {
    plan.data_files
        .iter()
        .map(|file| file.row_count)
        .try_fold(0u64, |total, row_count| row_count.map(|row_count| total.saturating_add(row_count)))
}

pub(crate) fn count_value_from_batches(batches: &[RecordBatch]) -> Option<u64> {
    let first_batch = batches.first()?;
    if first_batch.num_rows() != 1 || first_batch.num_columns() != 1 {
        return None;
    }
    let column = first_batch.column(0);
    if column.is_null(0) {
        return None;
    }
    if let Some(array) = column.as_any().downcast_ref::<UInt64Array>() {
        return Some(array.value(0));
    }
    if let Some(array) = column.as_any().downcast_ref::<UInt32Array>() {
        return Some(array.value(0) as u64);
    }
    if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
        return u64::try_from(array.value(0)).ok();
    }
    if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
        return u64::try_from(array.value(0)).ok();
    }
    None
}
