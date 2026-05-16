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
    let mut plan = iceberg_file_scan_tasks_to_table_format_scan_plan_with_filter_enforcement(
        snapshot_id,
        catalog_type,
        tasks,
        false,
        None,
    )?;
    attach_native_parquet_source_strategies(&mut plan, parquet_read_strategy_config)?;
    Ok(plan)
}

pub(super) fn iceberg_file_scan_tasks_to_table_format_scan_plan_with_filter_enforcement(
    snapshot_id: Option<String>,
    catalog_type: Option<String>,
    tasks: impl IntoIterator<Item = FileScanTask>,
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
    if let Some(mapping) = count_star_probe_column_mapping_for_tasks(&mut plan, &tasks)? {
        plan.pure_filter_columns.push(mapping);
    }

    for (task_index, task) in tasks.drain(..).enumerate() {
        if let Some(context) = timeout_context {
            context.check("convert_file_scan_task")?;
        }
        append_task_facts(&mut plan, task_index, task, parent_filter_enforces_task_predicate)?;
    }

    Ok(plan)
}

pub(super) fn attach_native_parquet_source_strategies(
    plan: &mut TableFormatScanPlan,
    config: &TableFormatParquetReadStrategyConfig,
) -> Result<()> {
    for file in &mut plan.data_files {
        if file.format != TableFormatDataFileFormat::Parquet || !file.read_capability.is_native_parquet_capable() {
            continue;
        }
        let mut metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(file.size_bytes, file.row_count);
        if let Some(content_identity) = file.source_content_identity.as_ref() {
            metadata = metadata.with_content_identity(content_identity.clone());
        }
        file.resolved_parquet_source = Some(resolve_table_format_parquet_source_with_metadata(
            &file.location,
            &file.read_capability,
            config,
            &metadata,
        )?);
    }

    for delete_file in &mut plan.delete_files {
        if delete_file.format != TableFormatDataFileFormat::Parquet
            || !delete_file.read_capability.is_native_parquet_capable()
        {
            continue;
        }
        let metadata =
            TableFormatParquetSourceMetadata::from_size_and_row_count(delete_file.size_bytes, delete_file.row_count);
        delete_file.resolved_parquet_source = Some(resolve_table_format_parquet_source_with_metadata(
            &delete_file.location,
            &delete_file.read_capability,
            config,
            &metadata,
        )?);
    }

    let resolved_delete_sources = plan
        .delete_files
        .iter()
        .filter_map(|file| file.resolved_parquet_source.clone().map(|source| (file.id.clone(), source)))
        .collect::<HashMap<_, _>>();
    for data_file in &mut plan.data_files {
        for delete_file in data_file.position_delete_files.iter_mut().chain(data_file.equality_delete_files.iter_mut())
        {
            delete_file.resolved_parquet_source = resolved_delete_sources.get(&delete_file.id).cloned();
        }
    }
    Ok(())
}

pub(super) fn trim_predicate_free_scan_limit_data_files(plan: &mut TableFormatScanPlan, request_has_predicate: bool) {
    if request_has_predicate {
        return;
    }
    let Some(limit) = plan.scan_limit else {
        return;
    };
    let limit = limit as u64;
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

    fn live_row_lower_bound(file: &TableFormatDataFile) -> u64 {
        data_file_live_row_lower_bound(file).expect("all file lower bounds were established above")
    }
    plan.data_files.sort_by_key(|file| std::cmp::Reverse(live_row_lower_bound(file)));

    let mut retained = 0usize;
    let mut rows = 0u64;
    for file in &plan.data_files {
        let live_lower_bound = live_row_lower_bound(file);
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
    if file.has_equality_delete_files() {
        return None;
    }
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
    let limit = match plan.scan_limit {
        Some(limit) if !request_has_predicate => limit as u64,
        _ => return Ok(()),
    };
    if let Some(context) = timeout_context {
        context.check("before_remote_delete_footer_row_count")?;
    }
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
        let row_count = row_count?;
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
        let footer = footer?;
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
    plan: &mut TableFormatScanPlan,
    tasks: &[FileScanTask],
) -> Result<Option<TableFormatColumnMapping>> {
    if tasks.is_empty()
        || !tasks.iter().all(|task| task.project_field_ids.is_empty())
        || !tasks.iter().any(|task| task.record_count.is_none())
    {
        return Ok(None);
    }

    let schema = tasks[0].schema();
    let mut rejected_name_mapping = None;
    for field in schema.as_struct().fields() {
        if !count_star_probe_field_is_direct_file_column_for_missing_count_tasks(tasks, field.id)? {
            continue;
        }
        let data_type = iceberg_field_to_arrow_data_type(field)?;
        if !is_supported_arrow_type(&data_type) {
            continue;
        }
        let name_mapping_candidates = match count_star_probe_name_mapping_candidates_for_tasks(tasks, field.id) {
            Ok(candidates) => candidates,
            Err(code) => {
                rejected_name_mapping.get_or_insert(code);
                continue;
            },
        };
        let mut mapping = TableFormatColumnMapping::new(
            Some(field.id),
            field.name.clone(),
            Some(field.name.clone()),
            data_type,
            !field.required,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::PureFilter,
        );
        if let Some(names) = name_mapping_candidates {
            mapping = mapping.with_field_id_source_names(names);
        }
        return Ok(Some(mapping));
    }

    if let Some(code) = rejected_name_mapping {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
            code,
            "Iceberg name mapping is ambiguous, incomplete, or nested for the COUNT(*) row-presence carrier",
        ));
    }

    Ok(None)
}

fn count_star_probe_name_mapping_candidates_for_tasks(
    tasks: &[FileScanTask],
    field_id: i32,
) -> std::result::Result<Option<Vec<String>>, TableFormatUnsupportedFeatureCode> {
    let mut candidates = None;
    for task in tasks.iter().filter(|task| task.record_count.is_none()) {
        let Some(mapping) = task.name_mapping.as_deref() else {
            continue;
        };
        let (mut task_candidates, rejected_code) =
            name_mapping_candidates_for_field_ids(mapping, task.schema(), &[field_id]);
        if let Some(code) = rejected_code {
            return Err(code);
        }
        let Some(task_candidate) = task_candidates.pop() else {
            return Err(TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported);
        };
        if candidates.as_ref().is_some_and(|candidates| candidates != &task_candidate.names) {
            return Err(TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported);
        }
        candidates = Some(task_candidate.names);
    }
    Ok(candidates)
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
    let source_content_identity = iceberg_task_content_identity(plan.snapshot_id.as_deref(), &task);
    let mut data_file = TableFormatDataFile::new(
        file_id.clone(),
        task.data_file_path.clone(),
        file_format.clone(),
        read_capability.clone(),
    )
    .with_size_bytes(task.file_size_in_bytes);
    if let Some(content_identity) = source_content_identity {
        data_file = data_file.with_source_content_identity(content_identity);
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

    let equality_delete_field_ids = task_equality_delete_field_ids(&task);
    let mut scan_field_ids = task.project_field_ids.clone();
    for field_id in &equality_delete_field_ids {
        if !scan_field_ids.contains(field_id) {
            scan_field_ids.push(*field_id);
        }
    }
    let name_mapping_candidates =
        append_name_mapping_unsupported_features(plan, &file_id, &task.name_mapping, task.schema(), &scan_field_ids);

    let partition_spec = task.partition_spec.as_deref();
    for field_id in task.project_field_ids.iter().copied() {
        let name_mapping_candidates = name_mapping_candidates.as_ref().and_then(|candidates| {
            candidates
                .iter()
                .find(|candidate| candidate.field_id == field_id)
                .map(|candidate| candidate.names.as_slice())
        });
        append_table_format_column_mapping(
            plan,
            task.schema(),
            partition_spec,
            field_id,
            name_mapping_candidates,
            TableFormatColumnRole::Projected,
        )?;
    }
    for field_id in equality_delete_field_ids {
        if plan
            .projected_columns
            .iter()
            .chain(plan.pure_filter_columns.iter())
            .any(|column| column.field_id == Some(field_id))
        {
            continue;
        }
        let name_mapping_candidates = name_mapping_candidates.as_ref().and_then(|candidates| {
            candidates
                .iter()
                .find(|candidate| candidate.field_id == field_id)
                .map(|candidate| candidate.names.as_slice())
        });
        append_table_format_column_mapping(
            plan,
            task.schema(),
            partition_spec,
            field_id,
            name_mapping_candidates,
            TableFormatColumnRole::PureFilter,
        )?;
    }

    for (delete_index, delete_file) in task.deletes.iter().enumerate() {
        append_delete_file_facts(plan, &mut data_file, &file_id, delete_index, delete_file)?;
    }

    plan.data_files.push(data_file);

    Ok(())
}

fn task_equality_delete_field_ids(task: &FileScanTask) -> Vec<i32> {
    let mut field_ids = Vec::new();
    for delete_file in &task.deletes {
        if delete_file.file_type != DataContentType::EqualityDeletes {
            continue;
        }
        for field_id in delete_file.equality_ids.iter().flatten() {
            if !field_ids.contains(field_id) {
                field_ids.push(*field_id);
            }
        }
    }
    field_ids
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
            plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
                TableFormatUnsupportedFeatureCode::IcebergPartitionTransformUnsupported,
                file_id,
                format!(
                    "Iceberg partition field {} references source field id {} absent from the scan schema",
                    partition_field.name, partition_field.source_id
                ),
            ));
            return Ok(());
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

#[derive(Debug, Clone)]
struct ProjectedNameMappingCandidates {
    field_id: i32,
    names: Vec<String>,
}

fn append_name_mapping_unsupported_features(
    plan: &mut TableFormatScanPlan,
    file_id: &str,
    mapping: &Option<Arc<IcebergNameMapping>>,
    schema: &IcebergSchema,
    projected_field_ids: &[i32],
) -> Option<Vec<ProjectedNameMappingCandidates>> {
    let mapping = mapping.as_deref()?;

    let (candidates, rejected_code) = name_mapping_candidates_for_field_ids(mapping, schema, projected_field_ids);
    if let Some(code) = rejected_code {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            code,
            file_id,
            "Iceberg name mapping is ambiguous, incomplete, or nested for the projected top-level scan schema",
        ));
    }
    Some(candidates)
}

fn name_mapping_candidates_for_field_ids(
    mapping: &IcebergNameMapping,
    schema: &IcebergSchema,
    field_ids: &[i32],
) -> (Vec<ProjectedNameMappingCandidates>, Option<TableFormatUnsupportedFeatureCode>) {
    let mut candidates = Vec::with_capacity(field_ids.len());
    let mut rejected_code = None;
    let mut claimed_names = std::collections::BTreeSet::new();
    for field_id in field_ids {
        let Some(field) = schema.as_struct().field_by_id(*field_id) else {
            rejected_code = Some(TableFormatUnsupportedFeatureCode::IcebergFieldIdMappingUnsupported);
            continue;
        };
        let entries = mapping.fields().iter().filter(|entry| entry.field_id() == Some(*field_id)).collect::<Vec<_>>();
        let [entry] = entries.as_slice() else {
            rejected_code = Some(TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported);
            continue;
        };
        if !entry.fields().is_empty() {
            rejected_code = Some(TableFormatUnsupportedFeatureCode::IcebergNestedProjectionUnsupported);
            continue;
        }
        if entry.names().is_empty() {
            rejected_code = Some(TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported);
            continue;
        }
        for name in entry.names() {
            if !claimed_names.insert(name.clone()) {
                rejected_code = Some(TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported);
            }
        }
        candidates.push(ProjectedNameMappingCandidates { field_id: *field_id, names: entry.names().to_vec() });

        let name_claimed_elsewhere = mapping.fields().iter().any(|other| {
            other.field_id() != Some(*field_id)
                && other.names().iter().any(|name| name == &field.name || entry.names().contains(name))
        });
        if name_claimed_elsewhere {
            rejected_code = Some(TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported);
        }
    }

    (candidates, rejected_code)
}

pub(super) fn append_table_format_column_mapping(
    plan: &mut TableFormatScanPlan,
    schema: &IcebergSchema,
    partition_spec: Option<&PartitionSpec>,
    field_id: i32,
    name_mapping_candidates: Option<&[String]>,
    role: TableFormatColumnRole,
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
        role,
    );
    if let Some(names) = name_mapping_candidates {
        mapping = mapping.with_field_id_source_names(names.iter().cloned());
    }
    if let Some(default_value) = field
        .initial_default
        .as_ref()
        .and_then(|default| table_format_partition_scalar_value(Some(default), &mapping.data_type))
    {
        mapping = mapping.with_missing_default_value(default_value);
    }
    match role {
        TableFormatColumnRole::Projected => plan.projected_columns.push(mapping),
        TableFormatColumnRole::PureFilter => plan.pure_filter_columns.push(mapping),
    }

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
    let mut table_delete = TableFormatDeleteFile::for_data_file(
        delete_id.clone(),
        data_file_id.to_string(),
        delete_file.file_path.clone(),
        format.clone(),
        content.clone(),
    )
    .with_size_bytes(delete_file.file_size_in_bytes)
    .with_read_capability(read_capability.clone());
    if let Some(equality_ids) = delete_file.equality_ids.clone() {
        table_delete = table_delete.with_equality_ids(equality_ids);
    }

    plan.delete_files.push(table_delete.clone());
    let native_parquet_delete_candidate = table_delete.format == TableFormatDataFileFormat::Parquet
        && table_delete.read_capability.is_native_parquet_capable();
    if table_delete.content == TableFormatDeleteFileContent::PositionDeletes && native_parquet_delete_candidate {
        data_file.push_position_delete_file(table_delete);
    } else if table_delete.content == TableFormatDeleteFileContent::EqualityDeletes
        && native_parquet_delete_candidate
        && table_delete.equality_ids.as_ref().is_some_and(|ids| !ids.is_empty())
    {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            delete_id,
            "Iceberg equality delete files require delete-file footer field-id proof before native execution",
        ));
    } else {
        let reason = if content != TableFormatDeleteFileContent::PositionDeletes
            && content != TableFormatDeleteFileContent::EqualityDeletes
        {
            format!("Iceberg {} delete files are not supported by native row-delete execution", content.code())
        } else if format != TableFormatDataFileFormat::Parquet {
            format!("Iceberg {} delete file format {} is not native Parquet", content.code(), format.code())
        } else if !read_capability.is_native_parquet_capable() {
            format!(
                "Iceberg {} delete file location requires {} read capability",
                content.code(),
                read_capability.detail()
            )
        } else {
            format!("Iceberg {} delete files require non-empty equality field ids for native execution", content.code())
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
