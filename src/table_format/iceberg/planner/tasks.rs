use super::*;

/// Converts already planned Iceberg file tasks into adapter-owned scan facts.
///
/// Keeping this conversion public gives tests and later provider code a stable
/// way to validate native admission facts without invoking catalog credentials
/// or executing Arrow readers.
pub fn iceberg_file_scan_tasks_to_table_format_scan_plan(
    snapshot_id: Option<String>,
    catalog_type: Option<String>,
    tasks: impl IntoIterator<Item = FileScanTask>,
) -> Result<TableFormatScanPlan> {
    iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(
        snapshot_id,
        catalog_type,
        tasks,
        &TableFormatParquetReadStrategyConfig::from_env(),
    )
}

pub(super) fn iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(
    snapshot_id: Option<String>,
    catalog_type: Option<String>,
    tasks: impl IntoIterator<Item = FileScanTask>,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
) -> Result<TableFormatScanPlan> {
    iceberg_file_scan_tasks_to_table_format_scan_plan_with_config_and_filter_enforcement(
        snapshot_id,
        catalog_type,
        tasks,
        parquet_read_strategy_config,
        false,
        None,
    )
}

pub(super) fn iceberg_file_scan_tasks_to_table_format_scan_plan_with_config_and_filter_enforcement(
    snapshot_id: Option<String>,
    catalog_type: Option<String>,
    tasks: impl IntoIterator<Item = FileScanTask>,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
    parent_filter_enforces_task_predicate: bool,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<TableFormatScanPlan> {
    if let Some(context) = timeout_context {
        context.check("before_convert_file_scan_tasks")?;
    }
    let mut tasks = tasks.into_iter().collect::<Vec<_>>();
    let logical_schema =
        tasks.first().map(|task| iceberg_schema_to_datafusion_arrow_schema(task.schema())).transpose()?;
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg);

    if let Some(snapshot_id) = snapshot_id {
        plan = plan.with_snapshot_id(snapshot_id);
    }
    if let Some(catalog_type) = catalog_type {
        plan = plan.with_catalog_type(catalog_type);
    }
    if let Some(schema) = logical_schema {
        plan = plan.with_logical_schema(Arc::clone(&schema)).with_native_source_schema(schema);
    }
    if let Some(mapping) = count_star_probe_column_mapping_for_tasks(&tasks)? {
        plan.pure_filter_columns.push(mapping);
    }

    for (task_index, task) in tasks.drain(..).enumerate() {
        if let Some(context) = timeout_context {
            context.check("convert_file_scan_task")?;
        }
        append_task_facts(
            &mut plan,
            task_index,
            task,
            parquet_read_strategy_config,
            parent_filter_enforces_task_predicate,
        )?;
    }

    Ok(plan)
}

pub(super) fn iceberg_project_field_ids(schema: &IcebergSchema, column_names: Option<&[String]>) -> Result<Vec<i32>> {
    let column_names = column_names
        .map(|column_names| column_names.to_vec())
        .unwrap_or_else(|| schema.as_struct().fields().iter().map(|field| field.name.clone()).collect());
    let mut field_ids = Vec::with_capacity(column_names.len());
    for column_name in column_names {
        if is_metadata_column_name(&column_name) {
            field_ids.push(get_metadata_field_id(&column_name).map_err(|error| {
                iceberg_planning_error("failed to bind Iceberg metadata column in table-format scan", error)
            })?);
            continue;
        }
        let field_id = schema.field_id_by_name(&column_name).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                PLAN_SCAN_OPERATION,
                "Iceberg scan projection references a column that is absent from the snapshot schema",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("column", column_name.clone())
        })?;
        if schema.as_struct().field_by_id(field_id).is_none() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                PLAN_SCAN_OPERATION,
                "Iceberg scan projection references a nested field that native table-format scans cannot bind",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("column", column_name)
            .with_fact("field_id", field_id));
        }
        field_ids.push(field_id);
    }
    Ok(field_ids)
}

pub(super) async fn load_iceberg_manifest_list(
    table: &Table,
    snapshot: &SnapshotRef,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<ManifestList> {
    let reader = table.manifest_list_reader(snapshot);
    let load = reader.load();
    if let Some(context) = timeout_context {
        tokio::time::timeout(context.remaining("predicate_free_limit_manifest_list")?, load)
            .await
            .map_err(|_elapsed| context.error("predicate_free_limit_manifest_list"))?
            .map_err(|error| iceberg_planning_error("failed to load Iceberg manifest list", error))
    } else {
        load.await.map_err(|error| iceberg_planning_error("failed to load Iceberg manifest list", error))
    }
}

pub(super) async fn load_iceberg_manifest(
    file_io: &FileIO,
    manifest_file: &ManifestFile,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
    stage: &'static str,
) -> Result<Manifest> {
    let load = manifest_file.load_manifest(file_io);
    if let Some(context) = timeout_context {
        tokio::time::timeout(context.remaining(stage)?, load)
            .await
            .map_err(|_elapsed| context.error(stage))?
            .map_err(|error| iceberg_planning_error("failed to load Iceberg manifest", error))
    } else {
        load.await.map_err(|error| iceberg_planning_error("failed to load Iceberg manifest", error))
    }
}

pub(super) async fn load_limited_delete_index(
    manifest_list: &ManifestList,
    file_io: &FileIO,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<LimitedDeleteIndex> {
    if let Some(context) = timeout_context {
        context.check("before_predicate_free_limit_delete_manifests")?;
    }
    let requests = manifest_list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Deletes)
        .filter(|manifest| manifest.has_added_files() || manifest.has_existing_files())
        .cloned()
        .collect::<Vec<_>>();
    let fetches = requests.into_iter().map(|manifest_file| {
        let file_io = file_io.clone();
        async move {
            let partition_spec_id = manifest_file.partition_spec_id;
            load_iceberg_manifest(&file_io, &manifest_file, None, "predicate_free_limit_delete_manifest")
                .await
                .map(|manifest| (partition_spec_id, manifest))
        }
    });
    let mut manifests = futures::stream::iter(fetches).buffer_unordered(REMOTE_ICEBERG_METADATA_PLANNING_CONCURRENCY);
    let mut index = LimitedDeleteIndex::default();
    loop {
        let result = if let Some(context) = timeout_context {
            tokio::time::timeout(context.remaining("predicate_free_limit_delete_manifest")?, manifests.next())
                .await
                .map_err(|_elapsed| context.error("predicate_free_limit_delete_manifest"))?
        } else {
            manifests.next().await
        };
        let Some(result) = result else {
            break;
        };
        let (partition_spec_id, manifest) = result?;
        index.extend_manifest(partition_spec_id, &manifest);
    }
    Ok(index)
}

#[derive(Default)]
pub(super) struct LimitedDeleteIndex {
    pub(super) files: Vec<LimitedDeleteFile>,
}

impl LimitedDeleteIndex {
    pub(super) fn extend_manifest(&mut self, partition_spec_id: i32, manifest: &Manifest) {
        self.files.extend(
            manifest
                .entries()
                .iter()
                .filter(|entry| entry.is_alive() && entry.content_type() != DataContentType::Data)
                .map(|entry| LimitedDeleteFile::from_manifest_entry(partition_spec_id, entry.as_ref())),
        );
    }

    pub(super) fn deletes_for_data_file(
        &self,
        data_file: &DataFile,
        data_partition_spec_id: i32,
        data_sequence_number: Option<i64>,
    ) -> Vec<LimitedDeleteFile> {
        self.files
            .iter()
            .filter(|delete_file| {
                delete_file.applies_to_data_file(data_file, data_partition_spec_id, data_sequence_number)
            })
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone)]
pub(super) struct LimitedDeleteFile {
    pub(super) file_path: String,
    pub(super) file_size_in_bytes: u64,
    pub(super) file_type: DataContentType,
    pub(super) partition_spec_id: i32,
    pub(super) equality_ids: Option<Vec<i32>>,
    pub(super) sequence_number: Option<i64>,
    pub(super) record_count: u64,
    pub(super) partition: Struct,
    pub(super) referenced_data_file: Option<String>,
    pub(super) position_lower_bound: Option<u64>,
    pub(super) position_upper_bound: Option<u64>,
}

impl LimitedDeleteFile {
    pub(super) fn from_manifest_entry(partition_spec_id: i32, entry: &ManifestEntry) -> Self {
        let data_file = entry.data_file();
        Self {
            file_path: entry.file_path().to_string(),
            file_size_in_bytes: entry.file_size_in_bytes(),
            file_type: entry.content_type(),
            partition_spec_id,
            equality_ids: data_file.equality_ids(),
            sequence_number: entry.sequence_number(),
            record_count: entry.record_count(),
            partition: data_file.partition().clone(),
            referenced_data_file: data_file.referenced_data_file(),
            position_lower_bound: iceberg_position_delete_bound(data_file.lower_bounds()),
            position_upper_bound: iceberg_position_delete_bound(data_file.upper_bounds()),
        }
    }

    pub(super) fn applies_to_data_file(
        &self,
        data_file: &DataFile,
        data_partition_spec_id: i32,
        data_sequence_number: Option<i64>,
    ) -> bool {
        match self.file_type {
            DataContentType::PositionDeletes => {
                self.partition_spec_id == data_partition_spec_id
                    && self.partition == *data_file.partition()
                    && position_delete_sequence_applies(self.sequence_number, data_sequence_number)
                    && self.referenced_data_file.as_deref().is_none_or(|referenced| referenced == data_file.file_path())
            },
            DataContentType::EqualityDeletes => {
                equality_delete_sequence_applies(self.sequence_number, data_sequence_number)
                    && (self.partition.fields().is_empty()
                        || (self.partition_spec_id == data_partition_spec_id
                            && self.partition == *data_file.partition()))
            },
            DataContentType::Data => false,
        }
    }

    pub(super) fn to_file_scan_task_delete_file(&self) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile::builder()
            .with_file_path(self.file_path.clone())
            .with_file_size_in_bytes(self.file_size_in_bytes)
            .with_file_type(self.file_type)
            .with_partition_spec_id(self.partition_spec_id)
            .with_equality_ids(self.equality_ids.clone())
            .build()
    }
}

pub(super) fn iceberg_position_delete_bound(bounds: &HashMap<i32, Datum>) -> Option<u64> {
    [RESERVED_FIELD_ID_DELETE_FILE_POS, RESERVED_FIELD_ID_POS]
        .into_iter()
        .find_map(|field_id| bounds.get(&field_id).and_then(iceberg_position_bound_value))
}

pub(super) fn iceberg_position_bound_value(value: &Datum) -> Option<u64> {
    match value.literal() {
        PrimitiveLiteral::Long(value) => u64::try_from(*value).ok(),
        PrimitiveLiteral::Int(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

pub(super) fn attach_manifest_delete_metadata(
    plan: &mut TableFormatScanPlan,
    data_file_index: usize,
    delete_files: &[LimitedDeleteFile],
) {
    let Some(data_file) = plan.data_files.get_mut(data_file_index) else {
        return;
    };
    let data_file_id = data_file.id.clone();
    for limited_delete in delete_files {
        for delete_file in data_file
            .position_delete_files
            .iter_mut()
            .filter(|delete_file| delete_file.location == limited_delete.file_path)
        {
            delete_file.row_count = Some(limited_delete.record_count);
            delete_file.referenced_data_file = limited_delete.referenced_data_file.clone();
            delete_file.position_lower_bound = limited_delete.position_lower_bound;
            delete_file.position_upper_bound = limited_delete.position_upper_bound;
        }
        for delete_file in plan.delete_files.iter_mut().filter(|delete_file| {
            delete_file.data_file_id == data_file_id && delete_file.location == limited_delete.file_path
        }) {
            delete_file.row_count = Some(limited_delete.record_count);
            delete_file.referenced_data_file = limited_delete.referenced_data_file.clone();
            delete_file.position_lower_bound = limited_delete.position_lower_bound;
            delete_file.position_upper_bound = limited_delete.position_upper_bound;
        }
    }
}

pub(super) fn position_delete_sequence_applies(
    delete_sequence_number: Option<i64>,
    data_sequence_number: Option<i64>,
) -> bool {
    data_sequence_number
        .map(|data_sequence_number| delete_sequence_number >= Some(data_sequence_number))
        .unwrap_or(true)
}

pub(super) fn equality_delete_sequence_applies(
    delete_sequence_number: Option<i64>,
    data_sequence_number: Option<i64>,
) -> bool {
    data_sequence_number.map(|data_sequence_number| delete_sequence_number > Some(data_sequence_number)).unwrap_or(true)
}

pub(super) fn file_scan_task_from_manifest_entry(
    entry: &ManifestEntry,
    schema: IcebergSchemaRef,
    field_ids: Arc<Vec<i32>>,
    partition_spec: Option<Arc<PartitionSpec>>,
    deletes: Vec<FileScanTaskDeleteFile>,
    case_sensitive: bool,
) -> FileScanTask {
    FileScanTask::builder()
        .with_file_size_in_bytes(entry.file_size_in_bytes())
        .with_start(0)
        .with_length(entry.file_size_in_bytes())
        .with_record_count(Some(entry.record_count()))
        .with_data_file_path(entry.file_path().to_string())
        .with_data_file_format(entry.file_format())
        .with_schema(schema)
        .with_project_field_ids(field_ids.to_vec())
        .with_predicate(None)
        .with_deletes(deletes)
        .with_partition(Some(entry.data_file().partition().clone()))
        .with_partition_spec(partition_spec)
        .with_name_mapping(None)
        .with_case_sensitive(case_sensitive)
        .build()
}

pub(super) fn manifest_live_rows(manifest: &ManifestFile) -> u64 {
    manifest.added_rows_count.unwrap_or(0).saturating_add(manifest.existing_rows_count.unwrap_or(0))
}

pub(super) fn scan_plan_live_row_lower_bound(plan: &TableFormatScanPlan) -> Option<u64> {
    plan.data_files.iter().try_fold(0u64, |rows, file| Some(rows.saturating_add(data_file_live_row_lower_bound(file)?)))
}

pub(super) fn trim_predicate_free_scan_limit_data_files(plan: &mut TableFormatScanPlan, request_has_predicate: bool) {
    if request_has_predicate {
        return;
    }
    let Some(limit) = plan.scan_limit else {
        return;
    };
    let limit = u64::try_from(limit).unwrap_or(u64::MAX);
    if limit == 0 {
        plan.data_files.clear();
        return;
    }
    let Some(total_live_lower_bound) = plan
        .data_files
        .iter()
        .try_fold(0u64, |rows, file| Some(rows.saturating_add(data_file_live_row_lower_bound(file)?)))
    else {
        return;
    };
    if total_live_lower_bound < limit {
        return;
    }

    plan.data_files.sort_by(|left, right| {
        let left_rows = data_file_live_row_lower_bound(left).unwrap_or(0);
        let right_rows = data_file_live_row_lower_bound(right).unwrap_or(0);
        right_rows.cmp(&left_rows)
    });

    let mut retained = 0usize;
    let mut rows = 0u64;
    for file in &plan.data_files {
        let live_lower_bound = data_file_live_row_lower_bound(file).unwrap_or(0);
        retained += 1;
        rows = rows.saturating_add(live_lower_bound);
        if rows >= limit {
            break;
        }
    }
    if rows >= limit && retained < plan.data_files.len() {
        let original_data_file_ids = plan.data_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
        let original_delete_file_ids = plan.delete_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
        plan.data_files.truncate(retained);
        let retained_data_file_ids = plan.data_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
        plan.delete_files.retain(|delete_file| retained_data_file_ids.contains(&delete_file.data_file_id));
        let retained_delete_file_ids = plan.delete_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
        prune_dropped_file_local_unsupported_features(
            plan,
            &original_data_file_ids,
            &original_delete_file_ids,
            &retained_data_file_ids,
            &retained_delete_file_ids,
        );
    }
}

pub(super) fn data_file_live_row_lower_bound(file: &TableFormatDataFile) -> Option<u64> {
    let data_rows = file.row_count?;
    let delete_rows = file
        .position_delete_files
        .iter()
        .try_fold(0u64, |rows, delete_file| Some(rows.saturating_add(delete_file.row_count?)))?;
    Some(data_rows.saturating_sub(delete_rows))
}

pub(super) async fn attach_remote_delete_file_row_counts_for_scan_limit(
    plan: &mut TableFormatScanPlan,
    file_io: &FileIO,
    request_has_predicate: bool,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<()> {
    if request_has_predicate || plan.scan_limit.is_none() {
        return Ok(());
    }
    if let Some(context) = timeout_context {
        context.check("before_remote_delete_footer_row_count")?;
    }
    let limit = plan.scan_limit.and_then(|limit| u64::try_from(limit).ok()).unwrap_or(u64::MAX);
    let mut candidate_indices = plan
        .data_files
        .iter()
        .enumerate()
        .filter_map(|(index, file)| file.row_count.map(|rows| (index, rows)))
        .collect::<Vec<_>>();
    candidate_indices.sort_by(|(_left_index, left_rows), (_right_index, right_rows)| right_rows.cmp(left_rows));

    let mut retained_indices = Vec::new();
    let mut retained_live_rows = 0u64;
    for (data_file_index, _rows) in candidate_indices {
        attach_remote_delete_file_row_counts_for_data_file(plan, file_io, data_file_index, timeout_context).await?;
        let Some(live_rows) = plan.data_files.get(data_file_index).and_then(data_file_live_row_lower_bound) else {
            continue;
        };
        if live_rows == 0 {
            continue;
        }
        retained_indices.push(data_file_index);
        retained_live_rows = retained_live_rows.saturating_add(live_rows);
        if retained_live_rows >= limit {
            let original_data_file_ids = plan.data_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
            let original_delete_file_ids = plan.delete_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
            let retained_data_files =
                retained_indices.iter().filter_map(|index| plan.data_files.get(*index).cloned()).collect::<Vec<_>>();
            let retained_data_file_ids = retained_data_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
            plan.data_files = retained_data_files;
            plan.delete_files.retain(|delete_file| retained_data_file_ids.contains(&delete_file.data_file_id));
            let retained_delete_file_ids = plan.delete_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
            prune_dropped_file_local_unsupported_features(
                plan,
                &original_data_file_ids,
                &original_delete_file_ids,
                &retained_data_file_ids,
                &retained_delete_file_ids,
            );
            return Ok(());
        }
    }

    Ok(())
}

pub(super) fn prune_dropped_file_local_unsupported_features(
    plan: &mut TableFormatScanPlan,
    original_data_file_ids: &HashSet<String>,
    original_delete_file_ids: &HashSet<String>,
    retained_data_file_ids: &HashSet<String>,
    retained_delete_file_ids: &HashSet<String>,
) {
    plan.unsupported_features.retain(|feature| {
        let Some(file_id) = feature.file_id.as_deref() else {
            return true;
        };
        let known_file_local = original_data_file_ids.contains(file_id) || original_delete_file_ids.contains(file_id);
        !known_file_local || retained_data_file_ids.contains(file_id) || retained_delete_file_ids.contains(file_id)
    });
}

pub(super) async fn attach_remote_delete_file_row_counts_for_data_file(
    plan: &mut TableFormatScanPlan,
    file_io: &FileIO,
    data_file_index: usize,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<()> {
    let Some(data_file) = plan.data_files.get(data_file_index) else {
        return Ok(());
    };
    let requests = data_file
        .position_delete_files
        .iter()
        .enumerate()
        .filter_map(|(delete_index, delete_file)| {
            if delete_file.row_count.is_some()
                || delete_file.format != TableFormatDataFileFormat::Parquet
                || !delete_file.read_capability.is_native_parquet_capable()
            {
                return None;
            }
            Some((delete_index, delete_file.id.clone(), delete_file.location.clone(), delete_file.size_bytes))
        })
        .collect::<Vec<_>>();
    let fetches = requests.into_iter().map(|(delete_index, delete_id, location, size_bytes)| {
        let file_io = file_io.clone();
        async move {
            (delete_index, delete_id, remote_iceberg_parquet_footer_row_count(&file_io, &location, size_bytes).await)
        }
    });
    let mut row_counts = futures::stream::iter(fetches).buffer_unordered(REMOTE_FOOTER_FIELD_ID_PROOF_CONCURRENCY);

    loop {
        let result = if let Some(context) = timeout_context {
            tokio::time::timeout(context.remaining("remote_delete_footer_row_count")?, row_counts.next())
                .await
                .map_err(|_elapsed| context.error("remote_delete_footer_row_count"))?
        } else {
            row_counts.next().await
        };
        let Some((delete_index, delete_id, row_count)) = result else {
            break;
        };
        if let Some(row_count) = row_count {
            if let Some(delete_file) = plan
                .data_files
                .get_mut(data_file_index)
                .and_then(|data_file| data_file.position_delete_files.get_mut(delete_index))
            {
                delete_file.row_count = Some(row_count);
            }
            if let Some(delete_file) = plan.delete_files.iter_mut().find(|delete_file| delete_file.id == delete_id) {
                delete_file.row_count = Some(row_count);
            }
        }
    }

    Ok(())
}

pub(super) async fn attach_remote_footer_schemas_for_field_id_proof(
    plan: &mut TableFormatScanPlan,
    file_io: &FileIO,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<()> {
    if !scan_plan_needs_field_id_footer_schema(plan) {
        return Ok(());
    }

    if let Some(context) = timeout_context {
        context.check("before_remote_footer_schema")?;
    }

    let requests = plan
        .data_files
        .iter()
        .enumerate()
        .filter(|(_index, data_file)| {
            data_file.footer_schema.is_none()
                && !data_file.read_capability.is_native_local_parquet_capable()
                && data_file.is_native_parquet_representable()
        })
        .map(|(index, data_file)| (index, data_file.location.clone(), data_file.size_bytes))
        .collect::<Vec<_>>();
    let fetches = requests.into_iter().map(|(index, location, size_bytes)| {
        let file_io = file_io.clone();
        async move { (index, remote_iceberg_parquet_footer(&file_io, &location, size_bytes).await) }
    });
    let mut footers = futures::stream::iter(fetches).buffer_unordered(REMOTE_FOOTER_FIELD_ID_PROOF_CONCURRENCY);

    loop {
        let result = if let Some(context) = timeout_context {
            tokio::time::timeout(context.remaining("remote_footer_schema")?, footers.next())
                .await
                .map_err(|_elapsed| context.error("remote_footer_schema"))?
        } else {
            footers.next().await
        };
        let Some((index, footer)) = result else {
            break;
        };
        if let Some(footer) = footer
            && let Some(data_file) = plan.data_files.get_mut(index)
        {
            data_file.footer_schema = Some(footer.schema);
            data_file.row_group_null_counts = footer.row_group_null_counts;
        }
    }

    Ok(())
}

pub(super) fn count_star_probe_column_mapping_for_tasks(
    tasks: &[FileScanTask],
) -> Result<Option<TableFormatColumnMapping>> {
    if tasks.is_empty()
        || !tasks.iter().all(|task| task.project_field_ids.is_empty())
        || !tasks.iter().any(|task| task.record_count.is_none())
    {
        return Ok(None);
    }

    let schema = tasks[0].schema();
    for field in schema.as_struct().fields() {
        if !count_star_probe_field_is_direct_file_column_for_missing_count_tasks(tasks, field.id)? {
            continue;
        }
        let data_type = iceberg_field_to_arrow_data_type(field)?;
        if !is_supported_arrow_type(&data_type) {
            continue;
        }
        return Ok(Some(TableFormatColumnMapping::new(
            Some(field.id),
            field.name.clone(),
            Some(field.name.clone()),
            data_type,
            !field.required,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::PureFilter,
        )));
    }

    Ok(None)
}

pub(super) fn count_star_probe_field_is_direct_file_column_for_missing_count_tasks(
    tasks: &[FileScanTask],
    field_id: i32,
) -> Result<bool> {
    for task in tasks.iter().filter(|task| task.record_count.is_none()) {
        let Some(field) = task.schema().as_struct().field_by_id(field_id) else {
            return Ok(false);
        };
        if task.partition_spec.as_deref().is_some_and(|spec| identity_partition_source_field(spec, field_id)) {
            return Ok(false);
        }
        let data_type = iceberg_field_to_arrow_data_type(field)?;
        if !is_supported_arrow_type(&data_type) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn append_task_facts(
    plan: &mut TableFormatScanPlan,
    task_index: usize,
    task: FileScanTask,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
    parent_filter_enforces_task_predicate: bool,
) -> Result<()> {
    let file_id = if task.data_file_path.is_empty() {
        format!("iceberg-data-file-{task_index}")
    } else {
        format!(
            "iceberg-data-file-{task_index}-h{}",
            fnv1a64_hex(credential_free_endpoint_identity(&task.data_file_path).as_bytes())
        )
    };
    let file_format = table_format_data_file_format(task.data_file_format);
    let read_capability = read_capability_for_location(&task.data_file_path);
    let mut source_metadata =
        TableFormatParquetSourceMetadata::from_size_and_row_count(Some(task.file_size_in_bytes), task.record_count);
    if let Some(content_identity) = iceberg_task_content_identity(plan.snapshot_id.as_deref(), &task) {
        source_metadata = source_metadata.with_content_identity(content_identity);
    }
    let resolved_parquet_source =
        if task.data_file_format == DataFileFormat::Parquet && read_capability.is_native_parquet_capable() {
            Some(resolve_table_format_parquet_source_with_metadata(
                &task.data_file_path,
                &read_capability,
                parquet_read_strategy_config,
                &source_metadata,
            )?)
        } else {
            None
        };
    let mut data_file = TableFormatDataFile::new(
        file_id.clone(),
        task.data_file_path.clone(),
        file_format.clone(),
        read_capability.clone(),
    )
    .with_size_bytes(task.file_size_in_bytes);
    if let Some(resolved_parquet_source) = resolved_parquet_source {
        data_file = data_file.with_resolved_parquet_source(resolved_parquet_source);
    }
    if let Some(record_count) = task.record_count {
        data_file = data_file.with_row_count(record_count);
    }
    append_partition_values(plan, &mut data_file, &file_id, &task)?;
    if task.data_file_format != DataFileFormat::Parquet {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergNonParquetDataFile,
            file_id.clone(),
            format!("Iceberg data file format {} is not native Parquet", task.data_file_format),
        ));
    }

    if !read_capability.is_native_parquet_capable() {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported,
            file_id.clone(),
            format!("Iceberg data file location requires {} read capability", read_capability.detail()),
        ));
    }

    if task.predicate.is_some() && !parent_filter_enforces_task_predicate {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergResidualPredicateUnsupported,
            file_id.clone(),
            "Iceberg reader predicate pushdown is not represented by Milestone B native scan facts",
        ));
    }

    if let Some(name_mapping) = task.name_mapping.as_deref() {
        if !name_mapping_is_projection_identity(name_mapping, task.schema(), &task.project_field_ids) {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
                TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported,
                file_id.clone(),
                "Iceberg name mapping is not an identity of the scan schema; non-identity mappings require explicit native field-id mapping support",
            ));
        }
    }

    let partition_spec = task.partition_spec.as_deref();
    for field_id in task.project_field_ids.iter().copied() {
        append_projected_column_mapping(plan, task.schema(), partition_spec, field_id)?;
    }

    for (delete_index, delete_file) in task.deletes.iter().enumerate() {
        append_delete_file_facts(
            plan,
            &mut data_file,
            &file_id,
            delete_index,
            delete_file,
            parquet_read_strategy_config,
        )?;
    }

    plan.data_files.push(data_file);

    Ok(())
}

pub(super) fn append_partition_values(
    plan: &mut TableFormatScanPlan,
    data_file: &mut TableFormatDataFile,
    file_id: &str,
    task: &FileScanTask,
) -> Result<()> {
    let Some(partition) = task.partition.as_ref().filter(|partition| !partition.fields().is_empty()) else {
        return Ok(());
    };
    let Some(partition_spec) = task.partition_spec.as_ref() else {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergPartitionTransformUnsupported,
            file_id,
            "Iceberg partition values are present but partition spec facts are not exposed by this scan task",
        ));
        return Ok(());
    };

    let mut values = Vec::new();
    for (partition_index, partition_field) in partition_spec.fields().iter().enumerate() {
        let Some(partition_value) = partition.fields().get(partition_index) else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                TASK_CONVERSION_OPERATION,
                "Iceberg partition value count does not match the partition spec",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("data_file_id", file_id)
            .with_fact("partition_field", partition_field.name.as_str())
            .with_fact("partition_index", partition_index));
        };
        let Some(source_field) = task.schema().field_by_id(partition_field.source_id) else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                TASK_CONVERSION_OPERATION,
                "Iceberg partition source field is absent from the scan schema",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("data_file_id", file_id)
            .with_fact("source_field_id", partition_field.source_id)
            .with_fact("partition_field", partition_field.name.as_str()));
        };
        let transform = table_format_partition_transform(&partition_field.transform);
        let transform_code = transform.code().to_string();
        let result_type = partition_field.transform.result_type(source_field.field_type.as_ref()).map_err(|error| {
            Error::new(
                ErrorCode::UnsupportedType,
                TASK_CONVERSION_OPERATION,
                "failed to resolve Iceberg partition transform result type",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("data_file_id", file_id)
            .with_fact("source_field_id", partition_field.source_id)
            .with_fact("partition_transform", transform_code)
            .with_source(error)
        })?;
        let data_type = iceberg_type_to_arrow_data_type(&result_type)?;
        let Some(value) = table_format_partition_scalar_value(partition_value.as_ref(), &data_type) else {
            continue;
        };
        let column_name = if partition_field.transform == Transform::Identity {
            source_field.name.clone()
        } else {
            partition_field.name.clone()
        };
        values.push(TableFormatPartitionValue::new(column_name, data_type, !source_field.required, transform, value));
    }
    data_file.partition_values = values;
    Ok(())
}

/// Whether `mapping` resolves every projected top-level field exactly the way
/// the native scan already resolves it — by the schema field name.
///
/// The native Parquet scan matches file columns by name
/// (`append_projected_column_mapping` emits the schema field name as the file
/// column name), so a name mapping is admissible iff following it can never
/// disagree with name-based resolution:
///
/// - every projected field id has exactly one root mapped-field entry, whose
///   single name is exactly the schema field name and which has no nested
///   children, and
/// - no other root entry claims that field name (an alias entry could bind the
///   same file column to a different field id than name-based resolution).
///
/// Java writers attach exactly such identity mappings automatically (e.g.
/// Spark's `add_files` imports of Parquet files that carry no field ids).
/// Anything else keeps the structured rejection.
pub(super) fn name_mapping_is_projection_identity(
    mapping: &IcebergNameMapping,
    schema: &IcebergSchema,
    projected_field_ids: &[i32],
) -> bool {
    projected_field_ids.iter().all(|field_id| {
        let Some(field) = schema.as_struct().field_by_id(*field_id) else {
            return false;
        };
        let mut identity_entries = mapping.fields().iter().filter(|entry| entry.field_id() == Some(*field_id));
        let Some(entry) = identity_entries.next() else {
            return false;
        };
        if identity_entries.next().is_some() {
            return false;
        }
        let identity = entry.fields().is_empty() && entry.names() == std::slice::from_ref(&field.name);
        let name_claimed_elsewhere = mapping
            .fields()
            .iter()
            .any(|other| other.field_id() != Some(*field_id) && other.names().contains(&field.name));
        identity && !name_claimed_elsewhere
    })
}

pub(super) fn append_projected_column_mapping(
    plan: &mut TableFormatScanPlan,
    schema: &IcebergSchema,
    partition_spec: Option<&PartitionSpec>,
    field_id: i32,
) -> Result<()> {
    let Some(field) = schema.field_by_id(field_id) else {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergFieldIdMappingUnsupported,
            field_id.to_string(),
            format!("Iceberg projected field id {field_id} is absent from the scan schema"),
        ));
        return Ok(());
    };

    let data_type = iceberg_field_to_arrow_data_type(field)?;
    let source = if schema.as_struct().field_by_id(field_id).is_some() {
        if partition_spec.is_some_and(|spec| identity_partition_source_field(spec, field_id)) {
            TableFormatColumnSource::IdentityPartitionValue
        } else {
            TableFormatColumnSource::DirectTopLevelFileColumn
        }
    } else {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergNestedProjectionUnsupported,
            field_id.to_string(),
            format!(
                "Iceberg projected field id {field_id} is nested and cannot lower to a top-level native source column"
            ),
        ));
        TableFormatColumnSource::UnsupportedNestedProjection
    };

    let mut mapping = TableFormatColumnMapping::new(
        Some(field_id),
        field.name.clone(),
        Some(field.name.clone()),
        data_type,
        !field.required,
        source,
        TableFormatColumnRole::Projected,
    );
    if let Some(default_value) = field
        .initial_default
        .as_ref()
        .and_then(|default| table_format_partition_scalar_value(Some(default), &mapping.data_type))
    {
        mapping = mapping.with_missing_default_value(default_value);
    }
    plan.projected_columns.push(mapping);

    Ok(())
}

pub(super) fn identity_partition_source_field(partition_spec: &PartitionSpec, field_id: i32) -> bool {
    partition_spec.fields().iter().any(|partition_field| {
        partition_field.source_id == field_id && partition_field.transform == Transform::Identity
    })
}

pub(super) fn table_format_partition_transform(transform: &Transform) -> TableFormatPartitionTransform {
    match transform {
        Transform::Identity => TableFormatPartitionTransform::Identity,
        Transform::Bucket(buckets) => TableFormatPartitionTransform::Bucket { buckets: *buckets },
        Transform::Truncate(width) => TableFormatPartitionTransform::Truncate { width: *width },
        Transform::Year => TableFormatPartitionTransform::Year,
        Transform::Month => TableFormatPartitionTransform::Month,
        Transform::Day => TableFormatPartitionTransform::Day,
        Transform::Hour => TableFormatPartitionTransform::Hour,
        Transform::Void => TableFormatPartitionTransform::Unknown("void".to_string()),
        Transform::Unknown => TableFormatPartitionTransform::Unknown("unknown".to_string()),
    }
}

pub(super) fn table_format_partition_scalar_value(
    value: Option<&Literal>,
    data_type: &DataType,
) -> Option<TableFormatScalarValue> {
    let Some(value) = value else {
        return Some(TableFormatScalarValue::Null);
    };
    let primitive = value.as_primitive_literal()?;
    match primitive {
        PrimitiveLiteral::Boolean(value) => Some(TableFormatScalarValue::Boolean(value)),
        PrimitiveLiteral::Int(value) if matches!(data_type, DataType::Date32) => {
            Some(TableFormatScalarValue::Date32(value))
        },
        PrimitiveLiteral::Int(value) => Some(TableFormatScalarValue::Int32(value)),
        PrimitiveLiteral::Long(value) => Some(TableFormatScalarValue::Int64(value)),
        PrimitiveLiteral::Int128(value) if matches!(data_type, DataType::Decimal128(_, _)) => {
            Some(TableFormatScalarValue::Int128(value))
        },
        PrimitiveLiteral::UInt128(value) if matches!(data_type, DataType::Decimal128(_, _)) => {
            i128::try_from(value).ok().map(TableFormatScalarValue::Int128)
        },
        PrimitiveLiteral::Float(value) => Some(TableFormatScalarValue::Float64(f64::from(value.into_inner()))),
        PrimitiveLiteral::Double(value) => Some(TableFormatScalarValue::Float64(value.into_inner())),
        PrimitiveLiteral::String(value) => Some(TableFormatScalarValue::Utf8(value)),
        PrimitiveLiteral::Binary(_)
        | PrimitiveLiteral::Int128(_)
        | PrimitiveLiteral::UInt128(_)
        | PrimitiveLiteral::AboveMax
        | PrimitiveLiteral::BelowMin => None,
    }
}

pub(super) fn append_delete_file_facts(
    plan: &mut TableFormatScanPlan,
    data_file: &mut TableFormatDataFile,
    data_file_id: &str,
    delete_index: usize,
    delete_file: &FileScanTaskDeleteFile,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
) -> Result<()> {
    let delete_id = if delete_file.file_path.is_empty() {
        format!("{data_file_id}#delete-{delete_index}")
    } else {
        format!(
            "{data_file_id}#delete-{delete_index}-h{}",
            fnv1a64_hex(credential_free_endpoint_identity(&delete_file.file_path).as_bytes())
        )
    };
    let format = table_format_data_file_format_from_path(&delete_file.file_path);
    let content = table_format_delete_content(delete_file.file_type);
    let read_capability = read_capability_for_location(&delete_file.file_path);
    let resolved_parquet_source = if format == TableFormatDataFileFormat::Parquet
        && content == TableFormatDeleteFileContent::PositionDeletes
        && read_capability.is_native_parquet_capable()
    {
        let metadata =
            TableFormatParquetSourceMetadata::from_size_and_row_count(Some(delete_file.file_size_in_bytes), None);
        Some(resolve_table_format_parquet_source_with_metadata(
            &delete_file.file_path,
            &read_capability,
            parquet_read_strategy_config,
            &metadata,
        )?)
    } else {
        None
    };
    let mut table_delete = TableFormatDeleteFile::for_data_file(
        delete_id.clone(),
        data_file_id.to_string(),
        delete_file.file_path.clone(),
        format.clone(),
        content.clone(),
    )
    .with_size_bytes(delete_file.file_size_in_bytes)
    .with_read_capability(read_capability.clone());
    if let Some(resolved_parquet_source) = resolved_parquet_source {
        table_delete = table_delete.with_resolved_parquet_source(resolved_parquet_source);
    }

    let supported = table_delete.is_supported_position_delete();
    plan.delete_files.push(table_delete.clone());
    if supported {
        data_file.push_position_delete_file(table_delete);
    } else {
        let reason = if content != TableFormatDeleteFileContent::PositionDeletes {
            format!("Iceberg {} delete files are not supported by native position-delete execution", content.code())
        } else if format != TableFormatDataFileFormat::Parquet {
            format!("Iceberg position delete file format {} is not native Parquet", format.code())
        } else {
            format!("Iceberg position delete file location requires {} read capability", read_capability.detail())
        };
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            delete_id,
            reason,
        ));
    }
    Ok(())
}

pub(super) fn iceberg_task_content_identity(snapshot_id: Option<&str>, task: &FileScanTask) -> Option<String> {
    let snapshot_id = snapshot_id?.trim();
    if snapshot_id.is_empty() {
        return None;
    }
    let path_hash = fnv1a64_hex(credential_free_endpoint_identity(&task.data_file_path).as_bytes());
    let rows = task.record_count.map_or_else(|| "unknown".to_string(), |rows| rows.to_string());
    Some(format!("iceberg_snapshot_{snapshot_id}_file_{path_hash}_size_{}_rows_{rows}", task.file_size_in_bytes))
}

pub(super) fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

pub(super) fn table_format_data_file_format(format: DataFileFormat) -> TableFormatDataFileFormat {
    match format {
        DataFileFormat::Parquet => TableFormatDataFileFormat::Parquet,
        DataFileFormat::Orc => TableFormatDataFileFormat::Orc,
        DataFileFormat::Avro => TableFormatDataFileFormat::Avro,
        DataFileFormat::Puffin => TableFormatDataFileFormat::Unknown("puffin".to_string()),
    }
}

pub(super) fn table_format_data_file_format_from_path(path: &str) -> TableFormatDataFileFormat {
    let extension = path.rsplit_once('.').map(|(_, extension)| extension.to_ascii_lowercase());
    match extension.as_deref() {
        Some("parquet") => TableFormatDataFileFormat::Parquet,
        Some("orc") => TableFormatDataFileFormat::Orc,
        Some("avro") => TableFormatDataFileFormat::Avro,
        Some(format) => TableFormatDataFileFormat::Unknown(format.to_string()),
        None => TableFormatDataFileFormat::Unknown("unknown".to_string()),
    }
}

pub(super) fn table_format_delete_content(content: DataContentType) -> TableFormatDeleteFileContent {
    match content {
        DataContentType::PositionDeletes => TableFormatDeleteFileContent::PositionDeletes,
        DataContentType::EqualityDeletes => TableFormatDeleteFileContent::EqualityDeletes,
        DataContentType::Data => TableFormatDeleteFileContent::Unknown("data".to_string()),
    }
}

pub(super) fn read_capability_for_location(location: &str) -> TableFormatReadCapability {
    let Some((scheme, _)) = location.split_once("://") else {
        return TableFormatReadCapability::local_file_system();
    };

    if scheme.eq_ignore_ascii_case("file") {
        TableFormatReadCapability::local_file_system()
    } else if scheme.is_empty() {
        TableFormatReadCapability::unsupported("empty_uri_scheme")
    } else {
        TableFormatReadCapability::object_store(scheme.to_ascii_lowercase())
    }
}
