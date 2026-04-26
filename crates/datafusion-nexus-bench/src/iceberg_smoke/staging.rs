use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectStoreStagingSummary {
    pub mode: ObjectStoreStagingMode,
    pub status: ObjectStoreStagingStatus,
    pub staged_data_file_count: usize,
    pub staged_bytes: u64,
}

impl ObjectStoreStagingSummary {
    pub const fn new(mode: ObjectStoreStagingMode, status: ObjectStoreStagingStatus) -> Self {
        Self { mode, status, staged_data_file_count: 0, staged_bytes: 0 }
    }

    pub(crate) fn with_staged(mut self, staged_data_file_count: usize, staged_bytes: u64) -> Self {
        self.staged_data_file_count = staged_data_file_count;
        self.staged_bytes = staged_bytes;
        self
    }
}

impl Default for ObjectStoreStagingSummary {
    fn default() -> Self {
        Self::new(ObjectStoreStagingMode::Auto, ObjectStoreStagingStatus::NotNeeded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectStoreStagingStatus {
    NotNeeded,
    DirectNative,
    Disabled,
    StagedLocal,
}

impl ObjectStoreStagingStatus {
    pub const fn code(self) -> &'static str {
        match self {
            Self::NotNeeded => "not_needed",
            Self::DirectNative => "direct_s3_native",
            Self::Disabled => "disabled",
            Self::StagedLocal => "staged_local",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StagingRecord {
    pub(crate) table_name: String,
    pub(crate) scan_plan: TableFormatScanPlan,
    pub(crate) summary: ObjectStoreStagingSummary,
}

#[derive(Debug, Clone)]
pub(crate) struct StagingTableFormatPlanner {
    inner: IcebergTableFormatPlanner,
    file_io: FileIO,
    mode: ObjectStoreStagingMode,
    tempdir: Arc<tempfile::TempDir>,
    records: Arc<Mutex<Vec<StagingRecord>>>,
    table_name: String,
}

impl StagingTableFormatPlanner {
    pub(crate) fn new(
        inner: IcebergTableFormatPlanner,
        file_io: FileIO,
        mode: ObjectStoreStagingMode,
        tempdir: Arc<tempfile::TempDir>,
        records: Arc<Mutex<Vec<StagingRecord>>>,
        table_name: String,
    ) -> Self {
        Self { inner, file_io, mode, tempdir, records, table_name }
    }
}

impl TableFormatPlanner for StagingTableFormatPlanner {
    fn plan_scan(
        &self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        Box::pin(async move {
            eprintln!(
                "[native_iceberg_smoke_report] stage=table_format_scan_planning_start table_name={} staging_mode={}",
                self.table_name,
                self.mode.code(),
            );
            let scan_plan = self.inner.plan_table_format_scan(request).await?;
            let table_stage_dir = self.tempdir.path().join(sanitize_staged_path_component(&self.table_name));
            tokio::fs::create_dir_all(&table_stage_dir).await.map_err(|error| {
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.table_format.iceberg.stage_object_store_data_files",
                    "failed to create Iceberg object-store staging directory",
                )
                .with_fact("table_name", &self.table_name)
                .with_fact("staging_mode", self.mode.code())
                .with_source(error)
            })?;
            let (scan_plan, summary) =
                stage_object_store_data_files(scan_plan, &self.file_io, self.mode, &table_stage_dir).await.map_err(
                    |error| {
                        Error::new(
                            ErrorCode::InvalidExecutionPlan,
                            "datafusion_nexus.table_format.iceberg.stage_object_store_data_files",
                            "failed to stage Iceberg object-store data files for native smoke execution",
                        )
                        .with_fact("table_name", &self.table_name)
                        .with_fact("staging_mode", self.mode.code())
                        .with_boxed_source(error)
                    },
                )?;
            let scan_plan = scan_plan_with_tpcds_source_unique_keys(&self.table_name, scan_plan);
            self.records.lock().expect("staging records mutex poisoned").push(StagingRecord {
                table_name: self.table_name.clone(),
                scan_plan: scan_plan.clone(),
                summary: summary.clone(),
            });
            let diagnostics = scan_plan.diagnostics();
            eprintln!(
                "[native_iceberg_smoke_report] stage=table_format_scan_planning_finish table_name={} planned_data_file_count={} planned_bytes={} native_source_read_capability={} staging_status={}",
                self.table_name,
                diagnostics.planned_data_file_count,
                optional_u64(diagnostics.planned_bytes),
                empty_as_none(&diagnostics.native_source_read_capability),
                summary.status.code(),
            );
            Ok(scan_plan)
        })
    }
}

fn scan_plan_with_tpcds_source_unique_keys(
    table_name: &str,
    mut scan_plan: TableFormatScanPlan,
) -> TableFormatScanPlan {
    if table_name != "reason" {
        return scan_plan;
    }
    let Some(key) = crate::report_workload::tpcds::tpcds_source_unique_key(table_name) else {
        return scan_plan;
    };

    scan_plan.logical_schema = Arc::new(schema_with_source_unique_keys(scan_plan.logical_schema.as_ref(), &[key]));
    scan_plan.native_source_schema =
        Arc::new(schema_with_source_unique_keys(scan_plan.native_source_schema.as_ref(), &[key]));
    for file in &scan_plan.data_files {
        for path in table_format_data_file_local_paths(file) {
            register_source_unique_keys_for_path(path, &[key]);
        }
    }
    scan_plan
}

fn table_format_data_file_local_paths(file: &datafusion_nexus::table_format::TableFormatDataFile) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = table_format_local_path_from_location(&file.location) {
        paths.push(path);
    }
    if let Some(resolved_source) = file.resolved_parquet_source.as_ref() {
        if let Some(path) = table_format_local_path_from_location(&resolved_source.original_location) {
            paths.push(path);
        }
        if matches!(
            resolved_source.read_strategy(),
            ParquetReadStrategy::LocalKvikio | ParquetReadStrategy::LocalCacheKvikio
        ) {
            paths.push(PathBuf::from(resolved_source.reader_location()));
        }
    }
    paths
}

fn table_format_local_path_from_location(location: &str) -> Option<PathBuf> {
    if let Some(path) = location.strip_prefix("file://localhost/") {
        return Some(PathBuf::from(format!("/{path}")));
    }
    if let Some(path) = location.strip_prefix("file://") {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = location.strip_prefix("file:") {
        return Some(PathBuf::from(path));
    }
    if location.contains("://") {
        return None;
    }
    Some(PathBuf::from(location))
}

pub(crate) async fn stage_object_store_data_files(
    scan_plan: TableFormatScanPlan,
    file_io: &FileIO,
    mode: ObjectStoreStagingMode,
    root: &Path,
) -> SmokeResult<(TableFormatScanPlan, ObjectStoreStagingSummary)> {
    if !scan_plan_has_stageable_object_store_data_files(&scan_plan) {
        return Ok((scan_plan, ObjectStoreStagingSummary::new(mode, ObjectStoreStagingStatus::NotNeeded)));
    }
    if scan_plan_has_direct_native_object_store_data_files(&scan_plan) && mode != ObjectStoreStagingMode::Required {
        return Ok((scan_plan, ObjectStoreStagingSummary::new(mode, ObjectStoreStagingStatus::DirectNative)));
    }
    if mode == ObjectStoreStagingMode::Disabled {
        return Ok((scan_plan, ObjectStoreStagingSummary::new(mode, ObjectStoreStagingStatus::Disabled)));
    }

    let mut staged_files = Vec::new();
    for (index, file) in scan_plan.data_files.iter().enumerate() {
        if file.format == TableFormatDataFileFormat::Parquet
            && matches!(file.read_capability, TableFormatReadCapability::ObjectStore { .. })
        {
            let staged_path = staged_file_path(root, index, &file.location);
            let bytes = file_io.new_input(&file.location)?.read().await?;
            tokio::fs::write(&staged_path, bytes.as_ref()).await?;
            staged_files.push(StagedObjectStoreDataFile {
                id: file.id.clone(),
                local_path: staged_path,
                staged_bytes: bytes.len() as u64,
            });
        }
    }
    let staged_data_file_count = staged_files.len();
    let staged_bytes = staged_files.iter().map(|file| file.staged_bytes).sum();
    Ok((
        rewrite_scan_plan_for_staged_object_store_files(scan_plan, &staged_files),
        ObjectStoreStagingSummary::new(mode, ObjectStoreStagingStatus::StagedLocal)
            .with_staged(staged_data_file_count, staged_bytes),
    ))
}

pub(crate) fn scan_plan_has_stageable_object_store_data_files(scan_plan: &TableFormatScanPlan) -> bool {
    scan_plan.data_files.iter().any(|file| {
        file.format == TableFormatDataFileFormat::Parquet
            && matches!(file.read_capability, TableFormatReadCapability::ObjectStore { .. })
    })
}

pub(crate) fn scan_plan_has_direct_native_object_store_data_files(scan_plan: &TableFormatScanPlan) -> bool {
    scan_plan.data_files.iter().any(|file| {
        file.format == TableFormatDataFileFormat::Parquet
            && matches!(file.read_capability, TableFormatReadCapability::ObjectStore { .. })
            && file.read_capability.is_native_parquet_capable()
            && !scan_plan.unsupported_features.iter().any(|feature| {
                feature.code == TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported
                    && feature.file_id.as_deref() == Some(file.id.as_str())
            })
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedObjectStoreDataFile {
    pub id: String,
    pub local_path: PathBuf,
    pub staged_bytes: u64,
}

pub fn rewrite_scan_plan_for_staged_object_store_files(
    mut scan_plan: TableFormatScanPlan,
    staged_files: &[StagedObjectStoreDataFile],
) -> TableFormatScanPlan {
    let staged_by_id = staged_files.iter().map(|file| (file.id.as_str(), file)).collect::<HashMap<_, _>>();
    for file in &mut scan_plan.data_files {
        if let Some(staged) = staged_by_id.get(file.id.as_str()) {
            file.location = staged.local_path.to_string_lossy().into_owned();
            file.read_capability = TableFormatReadCapability::local_file_system();
            file.resolved_parquet_source = resolve_table_format_parquet_source_with_metadata(
                &file.location,
                &file.read_capability,
                &TableFormatParquetReadStrategyConfig::default(),
                &TableFormatParquetSourceMetadata::from_size_and_row_count(Some(staged.staged_bytes), file.row_count),
            )
            .ok();
            file.size_bytes = Some(staged.staged_bytes);
        }
    }

    let staged_ids = staged_by_id.keys().copied().collect::<HashSet<_>>();
    scan_plan.unsupported_features.retain(|feature| {
        feature.code != TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported
            || !feature.file_id.as_deref().is_some_and(|file_id| staged_ids.contains(file_id))
    });
    scan_plan
}

pub(crate) fn staged_file_path(root: &Path, index: usize, location: &str) -> PathBuf {
    let basename = location
        .rsplit('/')
        .find(|part| !part.is_empty())
        .map(sanitize_staged_path_component)
        .filter(|part| !part.is_empty())
        .unwrap_or_else(|| "data.parquet".to_string());
    root.join(format!("staged-{index:05}-{basename}"))
}

pub(crate) fn sanitize_staged_path_component(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') { ch } else { '_' }).collect()
}

pub(crate) fn staging_records_snapshot(records: &Arc<Mutex<Vec<StagingRecord>>>) -> Vec<StagingRecord> {
    records.lock().expect("staging records mutex poisoned").clone()
}

pub(crate) fn clear_staging_records(records: &Arc<Mutex<Vec<StagingRecord>>>) {
    records.lock().expect("staging records mutex poisoned").clear();
}

pub(crate) fn aggregate_scan_diagnostics(records: &[StagingRecord]) -> TableFormatScanDiagnostics {
    if records.is_empty() {
        return TableFormatScanPlan::new(datafusion_nexus::table_format::TableFormat::Iceberg).diagnostics();
    }

    let diagnostics = records.iter().map(|record| record.scan_plan.diagnostics()).collect::<Vec<_>>();
    let planned_bytes = diagnostics
        .iter()
        .map(|diagnostics| diagnostics.planned_bytes)
        .try_fold(0u64, |total, bytes| bytes.map(|bytes| total.saturating_add(bytes)));
    let mut rejected_features = diagnostics
        .iter()
        .flat_map(|diagnostics| diagnostics.rejected_feature_list.iter().cloned())
        .collect::<Vec<_>>();
    rejected_features.sort();
    rejected_features.dedup();
    let mut capabilities = diagnostics
        .iter()
        .flat_map(|diagnostics| diagnostics.native_source_read_capability.split(','))
        .filter(|capability| !capability.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    capabilities.sort();
    capabilities.dedup();
    let source_read_strategy = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.source_read_strategy.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let credential_mode = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.credential_mode.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let lakehouse_cache_status = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.lakehouse_cache_status.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let lakehouse_cache_read_mode = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.lakehouse_cache_read_mode.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let lakehouse_cache_fill_mode = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.lakehouse_cache_fill_mode.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let snapshot_id =
        joined_optional_values(diagnostics.iter().filter_map(|diagnostics| diagnostics.snapshot_id.clone()));
    let catalog_type =
        joined_optional_values(diagnostics.iter().filter_map(|diagnostics| diagnostics.catalog_type.clone()));
    let table_format = joined_values(diagnostics.iter().map(|diagnostics| diagnostics.table_format.clone()));
    let direct_device_read_disposition = if diagnostics.iter().all(|diagnostics| {
        diagnostics.direct_device_read_disposition == TableFormatDirectDeviceReadDisposition::Eligible
    }) {
        TableFormatDirectDeviceReadDisposition::Eligible
    } else {
        TableFormatDirectDeviceReadDisposition::Rejected
    };

    TableFormatScanDiagnostics {
        table_format,
        snapshot_id,
        catalog_type,
        planned_data_file_count: diagnostics.iter().map(|diagnostics| diagnostics.planned_data_file_count).sum(),
        planned_delete_file_count: diagnostics.iter().map(|diagnostics| diagnostics.planned_delete_file_count).sum(),
        planned_bytes,
        admitted_native_source_file_count: diagnostics
            .iter()
            .map(|diagnostics| diagnostics.admitted_native_source_file_count)
            .sum(),
        rejected_feature_list: rejected_features,
        native_source_read_capability: capabilities.join(","),
        source_read_strategy,
        credential_mode,
        lakehouse_cache_status,
        lakehouse_cache_read_mode,
        lakehouse_cache_fill_mode,
        direct_device_read_disposition,
    }
}

pub(crate) fn aggregate_staging_summary(records: &[StagingRecord]) -> ObjectStoreStagingSummary {
    let Some(first) = records.first() else {
        return ObjectStoreStagingSummary::default();
    };
    let status = if records.iter().any(|record| record.summary.status == ObjectStoreStagingStatus::StagedLocal) {
        ObjectStoreStagingStatus::StagedLocal
    } else if records.iter().any(|record| record.summary.status == ObjectStoreStagingStatus::DirectNative) {
        ObjectStoreStagingStatus::DirectNative
    } else if records.iter().any(|record| record.summary.status == ObjectStoreStagingStatus::Disabled) {
        ObjectStoreStagingStatus::Disabled
    } else {
        ObjectStoreStagingStatus::NotNeeded
    };
    ObjectStoreStagingSummary::new(first.summary.mode, status).with_staged(
        records.iter().map(|record| record.summary.staged_data_file_count).sum(),
        records.iter().map(|record| record.summary.staged_bytes).sum(),
    )
}

pub(crate) fn joined_optional_values(values: impl Iterator<Item = String>) -> Option<String> {
    let joined = joined_values(values);
    if joined.is_empty() { None } else { Some(joined) }
}

pub(crate) fn joined_values(values: impl Iterator<Item = String>) -> String {
    let mut values = values.collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values.join(",")
}
