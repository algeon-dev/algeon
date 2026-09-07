use std::{borrow::Cow, sync::Arc, time::Instant};

use super::{
    ArrowDataType, BinaryOp, CudfSortOptions, DataFrame, ExecutionContext, Expr, Field, GpuDataFrame, LiteralValue,
    LocalParquetFile, LocalParquetSource, NodeId, QueryError, QueryErrorCode, QueryMetrics, Result, Schema,
    SourceChunk, SourceReadMode, acquire_lakehouse_cache_chunk_read_pin, acquire_lakehouse_cache_file_read_pin,
    append_source_order_columns, apply_source_scan_limit_to_frame, attach_row_group_facts,
    coerce_local_parquet_source_schema, col, cudf_error_for_chunk_read, cudf_local_parquet_source_chunk,
    cudf_local_parquet_source_file_selection, cudf_query_error, empty_local_parquet_execution_frame,
    ensure_local_cache_parquet_chunk, ensure_local_cache_parquet_file, execution_overrides, lit,
    parquet_chunk_path_fact, parquet_file_path_fact, parquet_file_stable_id_fact, read_local_parquet_file,
    record_background_cache_chunk_fill_enqueue, record_source_chunk_read_runtime,
};

const ICEBERG_POSITION_DELETE_FILE_PATH_COLUMN: &str = "file_path";
const ICEBERG_POSITION_DELETE_POS_COLUMN: &str = "pos";
const ICEBERG_POSITION_DELETE_ROW_POSITION_COLUMN: &str = "__nexus_iceberg_row_position";

pub(super) fn source_requires_position_delete_chunk_reads(source: &LocalParquetSource) -> bool {
    source.has_row_delete_files()
        && !source.is_count_star_probe_source()
        && (source.scan_limit.is_some()
            || source.files.iter().any(|file| {
                file.has_position_delete_files()
                    && file.row_groups.as_ref().is_some_and(|row_groups| !row_groups.is_empty())
            }))
}

pub(super) fn read_position_delete_parquet_source_as_chunks(
    context: &ExecutionContext,
    source_node_id: NodeId,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let chunk_source = source.clone().with_scan_limit(None);
    let planning_policy = if source.scan_limit.is_some() {
        crate::chunk::SourceChunkPlanningPolicy::for_position_delete_scan_limit()
    } else {
        crate::chunk::SourceChunkPlanningPolicy::default()
    };
    let chunk_plan = crate::chunk::plan_local_parquet_source_chunks_with_execution_context(
        source_node_id,
        &chunk_source,
        planning_policy,
        context,
    )?;
    metrics.source_metadata_load_time += chunk_plan.metadata_load_time;

    let mut frames = Vec::with_capacity(chunk_plan.chunk_count());
    let mut remaining_rows = source.scan_limit;
    for chunk in &chunk_plan.chunks {
        context.check_cancellation("nexus_query_engine.local_parquet_source.position_delete_chunks")?;
        let frame = read_position_delete_source_chunk(context, &chunk_source, chunk, metrics)?.into_frame();
        record_position_delete_source_chunk_top_level_metrics(chunk, metrics);
        let (frame, remaining) = apply_source_scan_limit_to_frame(context, frame, remaining_rows, metrics)?;
        remaining_rows = remaining;
        if frame.num_rows() > 0 {
            frames.push(frame);
        }
        if remaining_rows == Some(0) {
            break;
        }
    }

    let mut frames = frames.into_iter();
    let Some(frame) = frames.next() else {
        let frame = empty_local_parquet_execution_frame(context, source)?;
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = append_source_order_columns(context, frame, 0)?;
        metrics.record_input_batch(0);
        return Ok(GpuDataFrame::new(frame));
    };
    let other_frames = frames.collect::<Vec<_>>();
    if other_frames.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }

    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.local_parquet_source.position_delete_chunks_concat",
            "failed to concatenate native local parquet position-delete chunks",
            error,
        )
        .with_fact("file_count", source.file_count())
        .with_fact("read_columns", source.read_column_count())
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn read_position_delete_source_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    context.check_cancellation("nexus_query_engine.local_parquet_source.position_delete_chunk.read")?;
    if chunk.scan_limit_rows == Some(0) {
        let frame = empty_local_parquet_execution_frame(context, source)?;
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = append_source_order_columns(context, frame, chunk.chunk_index)?;
        metrics.record_input_batch(0);
        return Ok(GpuDataFrame::new(frame));
    }
    if chunk.zero_column_count {
        return Err(attach_row_group_facts(
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.position_delete_chunk",
                "Iceberg position-delete chunk reads require decoded data columns",
            )
            .with_fact("reason", "position_deletes_require_data_column")
            .with_fact("path", parquet_chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index),
            chunk,
        ));
    }

    let _cache_read_pin =
        if chunk.resolved_source.read_strategy() == crate::source::ParquetReadStrategy::LocalCacheKvikio {
            ensure_local_cache_parquet_chunk(context, chunk)?;
            Some(acquire_lakehouse_cache_chunk_read_pin(context, chunk)?)
        } else {
            None
        };
    let file =
        crate::chunk::source_file_for_chunk(source, chunk, "nexus_query_engine.local_parquet_chunk.position_deletes")?;
    let row_position_range =
        if file.has_position_delete_files() { crate::chunk::source_chunk_row_position_range(chunk)? } else { None };
    let (cudf_source, cudf_chunk) = cudf_local_parquet_source_chunk(context, source, chunk)?;
    let _io_lease = context.acquire_io_lease("nexus_query_engine.local_parquet_source.position_delete_chunk.read")?;
    let read_timer = Instant::now();
    let output = DataFrame::read_parquet_source_chunk_with_evidence(
        &context.runtime()?,
        &cudf_source,
        &cudf_chunk,
        SourceReadMode::ExecutionFrame,
    )
    .map_err(|error| {
        attach_row_group_facts(
            cudf_error_for_chunk_read(
                "nexus_query_engine.local_parquet_source.position_delete_chunk.read",
                "failed to read native local parquet position-delete chunk with cudf",
                error,
                chunk,
            )
            .with_fact("path", parquet_chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len()),
            chunk,
        )
    })?;
    record_background_cache_chunk_fill_enqueue(context, chunk, metrics);
    let observed_row_group_count = None;
    let read_evidence = output.read_evidence().cloned();
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    let frame = output.into_frame();
    let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
    let frame = apply_position_deletes_to_data_file(
        context,
        source,
        chunk.source_node_id,
        file,
        chunk.file_index,
        row_position_range,
        frame,
        metrics,
    )?;
    let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
    let frame = append_source_order_columns(context, frame, chunk.chunk_index)?;
    record_source_chunk_read_runtime(
        source,
        chunk,
        observed_row_group_count,
        frame.num_rows() as u64,
        read_elapsed,
        0,
        read_evidence.as_ref(),
        metrics,
    );
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(GpuDataFrame::new(frame))
}

pub(super) fn record_position_delete_source_chunk_top_level_metrics(chunk: &SourceChunk, metrics: &mut QueryMetrics) {
    metrics.parquet_files = metrics.parquet_files.saturating_add(1);
    metrics.parquet_bytes_known =
        metrics.parquet_bytes_known.saturating_add(chunk.known_row_group_bytes.or(chunk.known_file_bytes).unwrap_or(0));
    metrics.read_columns = metrics.read_columns.saturating_add(chunk.read_columns.len() as u64);
    if let Some(row_group_count) = chunk.known_read_row_group_count() {
        metrics.row_groups_selected = metrics.row_groups_selected.saturating_add(row_group_count as u64);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "source-read inputs (file/chunk identity, reader state, budgets) are distinct domain values"
)]
pub(super) fn apply_position_deletes_to_data_file(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    source_node_id: NodeId,
    file: &LocalParquetFile,
    file_index: usize,
    row_position_range: Option<crate::chunk::SourceChunkRowPositionRange>,
    mut frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    if !file.has_row_delete_files() || frame.num_rows() == 0 {
        return Ok(frame);
    }
    if source.read_columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.position_deletes",
            "Iceberg row deletes require at least one decoded data column",
        )
        .with_fact("reason", "row_deletes_require_data_column")
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
        .with_fact("file_index", file_index));
    }

    let row_position_column = unique_position_delete_column_name(&frame);
    let timer = Instant::now();
    let row_positions = frame.row_positions().map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.local_parquet_source.position_deletes.row_positions",
            "failed to construct Iceberg position-delete row positions",
            error,
        )
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
        .with_fact("file_index", file_index)
    })?;
    frame = frame.with_column(row_position_column.as_str(), row_positions).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.local_parquet_source.position_deletes.row_positions",
            "failed to append Iceberg position-delete row positions",
            error,
        )
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
        .with_fact("file_index", file_index)
    })?;
    let row_position_expr = position_delete_row_position_expr(row_position_column.as_str(), row_position_range)?;
    frame = frame.with_execution(&overrides).with_columns([row_position_expr]).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.local_parquet_source.position_deletes.row_positions",
            "failed to normalize Iceberg position-delete row positions",
            error,
        )
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
        .with_fact("file_index", file_index)
    })?;

    let data_file_path = file.resolved_source.original_location.as_str();
    for (delete_index, delete_file) in file.position_delete_files.iter().enumerate() {
        context.check_cancellation("nexus_query_engine.local_parquet_source.position_deletes")?;
        if delete_file.row_count == Some(0) {
            continue;
        }
        if !position_delete_file_matches_referenced_data_file(delete_file, data_file_path) {
            continue;
        }
        if row_position_range.is_some_and(|range| !position_delete_file_overlaps_row_range(delete_file, range)) {
            continue;
        }
        let delete_frame = read_position_delete_file(
            context,
            source_node_id,
            delete_file,
            data_file_path,
            row_position_range,
            file_index,
            delete_index,
            metrics,
        )?;
        if delete_frame.num_rows() == 0 {
            continue;
        }
        let predicates = position_delete_filter_predicates(delete_file, data_file_path, row_position_range)?;
        let matching_deletes = delete_frame
            .with_execution(&overrides)
            .filter_all(predicates)
            .and_then(|frame| frame.select([ICEBERG_POSITION_DELETE_POS_COLUMN]))
            .map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.local_parquet_source.position_deletes.filter",
                    "failed to filter Iceberg position-delete rows for the current data file",
                    error,
                )
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("stable_id", parquet_file_stable_id_fact(file))
                .with_fact("file_index", file_index)
                .with_fact("delete_index", delete_index)
                .with_fact(
                    "delete_stable_id",
                    delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()),
                )
            })?;
        if matching_deletes.num_rows() == 0 {
            continue;
        }
        frame = frame
            .with_execution(&overrides)
            .left_anti_join(
                &matching_deletes,
                cudf::DataFrameJoinOptions::new()
                    .left_on([row_position_column.as_str()])
                    .right_on([ICEBERG_POSITION_DELETE_POS_COLUMN])
                    .null_equality(cudf::NullEquality::Unequal),
            )
            .map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.local_parquet_source.position_deletes.apply",
                    "failed to apply Iceberg position deletes with nexus native anti join",
                    error,
                )
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("stable_id", parquet_file_stable_id_fact(file))
                .with_fact("file_index", file_index)
                .with_fact("delete_index", delete_index)
                .with_fact("delete_rows", matching_deletes.num_rows())
            })?;
    }
    for (delete_index, delete_file) in file.equality_delete_files.iter().enumerate() {
        context.check_cancellation("nexus_query_engine.local_parquet_source.equality_deletes")?;
        if delete_file.row_count == Some(0) {
            continue;
        }
        let delete_frame =
            read_equality_delete_file(context, source_node_id, source, delete_file, file_index, delete_index, metrics)?;
        if delete_frame.num_rows() == 0 {
            continue;
        }
        frame = frame
            .with_execution(&overrides)
            .left_anti_join(
                &delete_frame,
                cudf::DataFrameJoinOptions::new()
                    .left_on(delete_file.equality_columns.iter().map(String::as_str))
                    .right_on(delete_file.equality_columns.iter().map(String::as_str))
                    .null_equality(cudf::NullEquality::Equal),
            )
            .map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.local_parquet_source.equality_deletes.apply",
                    "failed to apply Iceberg equality deletes with nexus native anti join",
                    error,
                )
                .with_fact("path", parquet_file_path_fact(file))
                .with_fact("stable_id", parquet_file_stable_id_fact(file))
                .with_fact("file_index", file_index)
                .with_fact("delete_index", delete_index)
                .with_fact("delete_rows", delete_frame.num_rows())
            })?;
    }
    frame = frame
        .with_execution(&overrides)
        .sort_values([row_position_column.as_str()], CudfSortOptions::new().with_stable(true))
        .map_err(|error| {
            cudf_query_error(
                "nexus_query_engine.local_parquet_source.position_deletes.order",
                "failed to restore Iceberg data-file row order after applying position deletes",
                error,
            )
            .with_fact("path", parquet_file_path_fact(file))
            .with_fact("stable_id", parquet_file_stable_id_fact(file))
            .with_fact("file_index", file_index)
        })?;
    frame = frame.without_columns([row_position_column.as_str()]).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.local_parquet_source.position_deletes.cleanup",
            "failed to remove Iceberg position-delete work column",
            error,
        )
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
        .with_fact("file_index", file_index)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(frame)
}

pub(super) fn position_delete_row_position_expr(
    row_position_column: &str,
    row_position_range: Option<crate::chunk::SourceChunkRowPositionRange>,
) -> Result<cudf::Expr> {
    let int64 = cudf::DataType::new(cudf::TypeId::Int64);
    let expr = col(row_position_column).cast(int64);
    let expr = if let Some(range) = row_position_range
        && range.start > 0
    {
        expr + lit(position_delete_i64_bound(range.start, "start")?)
    } else {
        expr
    };
    Ok(expr.alias(row_position_column))
}

pub(super) fn position_delete_filter_predicates(
    delete_file: &crate::source::ParquetPositionDeleteFile,
    data_file_path: &str,
    row_position_range: Option<crate::chunk::SourceChunkRowPositionRange>,
) -> Result<Vec<cudf::Expr>> {
    let file_path_values = position_delete_file_path_filter_values(delete_file, data_file_path);
    let file_path_predicate =
        col(ICEBERG_POSITION_DELETE_FILE_PATH_COLUMN).is_in_list(file_path_values.into_iter().map(lit).collect());
    let mut predicates = vec![file_path_predicate];
    if let Some(range) = row_position_range {
        let start = position_delete_i64_bound(range.start, "start")?;
        let end = position_delete_i64_bound(range.end, "end")?;
        predicates.push(col(ICEBERG_POSITION_DELETE_POS_COLUMN).ge(lit(start)));
        predicates.push(col(ICEBERG_POSITION_DELETE_POS_COLUMN).lt(lit(end)));
    }
    Ok(predicates)
}

pub(super) fn position_delete_reader_predicate(
    delete_file: &crate::source::ParquetPositionDeleteFile,
    data_file_path: &str,
    row_position_range: Option<crate::chunk::SourceChunkRowPositionRange>,
) -> Result<Expr> {
    let file_path_values = position_delete_file_path_filter_values(delete_file, data_file_path)
        .into_iter()
        .map(|value| Expr::literal(LiteralValue::Utf8(value)))
        .collect();
    let mut predicate = Expr::in_list(Expr::column(ICEBERG_POSITION_DELETE_FILE_PATH_COLUMN), file_path_values);
    if let Some(range) = row_position_range {
        let start = position_delete_i64_bound(range.start, "start")?;
        let end = position_delete_i64_bound(range.end, "end")?;
        predicate = Expr::binary_with_output_type(
            predicate,
            BinaryOp::And,
            Expr::binary_with_output_type(
                Expr::column(ICEBERG_POSITION_DELETE_POS_COLUMN),
                BinaryOp::GtEq,
                Expr::literal(LiteralValue::Int64(start)),
                ArrowDataType::Boolean,
            ),
            ArrowDataType::Boolean,
        );
        predicate = Expr::binary_with_output_type(
            predicate,
            BinaryOp::And,
            Expr::binary_with_output_type(
                Expr::column(ICEBERG_POSITION_DELETE_POS_COLUMN),
                BinaryOp::Lt,
                Expr::literal(LiteralValue::Int64(end)),
                ArrowDataType::Boolean,
            ),
            ArrowDataType::Boolean,
        );
    }
    Ok(predicate)
}

pub(super) fn position_delete_i64_bound(value: u64, bound: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.position_deletes.row_positions",
            "Iceberg position-delete row-position range exceeds Int64 delete column representation",
        )
        .with_fact("reason", "position_delete_row_position_i64_overflow")
        .with_fact("bound", bound)
        .with_fact("value", value)
        .with_fact("max", i64::MAX)
    })
}

pub(super) fn position_delete_file_matches_referenced_data_file(
    delete_file: &crate::source::ParquetPositionDeleteFile,
    data_file_path: &str,
) -> bool {
    delete_file
        .referenced_data_file
        .as_deref()
        .is_none_or(|referenced| position_delete_data_file_paths_match(referenced, data_file_path))
}

pub(super) fn position_delete_file_path_filter_values(
    delete_file: &crate::source::ParquetPositionDeleteFile,
    data_file_path: &str,
) -> Vec<String> {
    let mut values = Vec::with_capacity(8);
    push_position_delete_file_path_variants(&mut values, data_file_path);
    if let Some(referenced) = delete_file
        .referenced_data_file
        .as_deref()
        .filter(|referenced| position_delete_data_file_paths_match(referenced, data_file_path))
    {
        push_position_delete_file_path_variants(&mut values, referenced);
    }
    values
}

pub(super) fn push_position_delete_file_path_variants(values: &mut Vec<String>, location: &str) {
    push_unique_position_delete_file_path_value(values, location);
    let normalized = normalized_local_file_location_text(location);
    let normalized = normalized.as_ref();
    push_unique_position_delete_file_path_value(values, normalized);
    if normalized.starts_with('/') {
        push_unique_position_delete_file_path_value(values, format!("file:{normalized}").as_str());
        push_unique_position_delete_file_path_value(values, format!("file://{normalized}").as_str());
        push_unique_position_delete_file_path_value(values, format!("file://localhost{normalized}").as_str());
    }
}

pub(super) fn push_unique_position_delete_file_path_value(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_string());
    }
}

pub(super) fn position_delete_data_file_paths_match(left: &str, right: &str) -> bool {
    left == right || normalized_local_file_location_text(left) == normalized_local_file_location_text(right)
}

pub(super) fn normalized_local_file_location_text(location: &str) -> Cow<'_, str> {
    if let Some(path) = location.strip_prefix("file://localhost/") {
        return Cow::Owned(format!("/{path}"));
    }
    if let Some(path) = location.strip_prefix("file://") {
        return Cow::Borrowed(path);
    }
    if let Some(path) = location.strip_prefix("file:") {
        return Cow::Borrowed(path);
    }
    Cow::Borrowed(location)
}

pub(super) fn position_delete_file_overlaps_row_range(
    delete_file: &crate::source::ParquetPositionDeleteFile,
    range: crate::chunk::SourceChunkRowPositionRange,
) -> bool {
    if range.start >= range.end {
        return false;
    }
    if delete_file.position_upper_bound.is_some_and(|upper| upper < range.start) {
        return false;
    }
    if delete_file.position_lower_bound.is_some_and(|lower| lower >= range.end) {
        return false;
    }
    true
}

#[expect(
    clippy::too_many_arguments,
    reason = "source-read inputs (file/chunk identity, reader state, budgets) are distinct domain values"
)]
pub(super) fn read_position_delete_file(
    context: &ExecutionContext,
    source_node_id: NodeId,
    delete_file: &crate::source::ParquetPositionDeleteFile,
    data_file_path: &str,
    row_position_range: Option<crate::chunk::SourceChunkRowPositionRange>,
    data_file_index: usize,
    delete_index: usize,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let file = delete_file.to_local_parquet_file();
    let schema = Arc::new(Schema::new(vec![
        Field::new(ICEBERG_POSITION_DELETE_FILE_PATH_COLUMN, ArrowDataType::Utf8, false),
        Field::new(ICEBERG_POSITION_DELETE_POS_COLUMN, ArrowDataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![file.clone()],
        schema,
        vec![ICEBERG_POSITION_DELETE_FILE_PATH_COLUMN.to_string(), ICEBERG_POSITION_DELETE_POS_COLUMN.to_string()],
    )
    .with_reader_predicate(position_delete_reader_predicate(delete_file, data_file_path, row_position_range)?);
    let columns = [ICEBERG_POSITION_DELETE_FILE_PATH_COLUMN, ICEBERG_POSITION_DELETE_POS_COLUMN];
    let frame = {
        let _cache_read_pin =
            if file.resolved_source.read_strategy() == crate::source::ParquetReadStrategy::LocalCacheKvikio {
                ensure_local_cache_parquet_file(context, &file)?;
                Some(acquire_lakehouse_cache_file_read_pin(context, &file)?)
            } else {
                None
            };
        let (cudf_source, cudf_chunk) =
            cudf_local_parquet_source_file_selection(context, &source, &file, data_file_index, delete_index)?;
        read_local_parquet_file(
            context,
            &source,
            source_node_id,
            &file,
            data_file_index,
            &cudf_source,
            &cudf_chunk,
            &columns,
            metrics,
        )?
    };
    coerce_local_parquet_source_schema(frame, &source, None, metrics).map_err(|error| {
        error.with_fact("position_delete_index", delete_index).with_fact(
            "position_delete_stable_id",
            delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()),
        )
    })
}

pub(super) fn read_equality_delete_file(
    context: &ExecutionContext,
    source_node_id: NodeId,
    source: &LocalParquetSource,
    delete_file: &crate::source::ParquetEqualityDeleteFile,
    data_file_index: usize,
    delete_index: usize,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let file = delete_file.to_local_parquet_file();
    let schema_fields = delete_file
        .equality_columns
        .iter()
        .map(|column| {
            source.schema.field_with_name(column).map(|field| field.as_ref().clone()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.local_parquet_source.equality_deletes",
                    "Iceberg equality-delete key column is not present in the data source schema",
                )
                .with_fact("reason", "missing_equality_delete_source_column")
                .with_fact("column", column.to_owned())
                .with_fact("delete_index", delete_index)
                .with_source(error)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let delete_schema = Arc::new(Schema::new(schema_fields));
    let delete_source =
        LocalParquetSource::from_files(vec![file.clone()], delete_schema, delete_file.equality_columns.clone());
    let columns = delete_file.equality_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let frame = {
        let _cache_read_pin =
            if file.resolved_source.read_strategy() == crate::source::ParquetReadStrategy::LocalCacheKvikio {
                ensure_local_cache_parquet_file(context, &file)?;
                Some(acquire_lakehouse_cache_file_read_pin(context, &file)?)
            } else {
                None
            };
        let (cudf_source, cudf_chunk) =
            cudf_local_parquet_source_file_selection(context, &delete_source, &file, data_file_index, delete_index)?;
        read_local_parquet_file(
            context,
            &delete_source,
            source_node_id,
            &file,
            data_file_index,
            &cudf_source,
            &cudf_chunk,
            &columns,
            metrics,
        )?
    };
    let frame = coerce_local_parquet_source_schema(frame, &delete_source, None, metrics).map_err(|error| {
        error.with_fact("equality_delete_index", delete_index).with_fact(
            "equality_delete_stable_id",
            delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()),
        )
    })?;
    frame.select(columns).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.local_parquet_source.equality_deletes.read",
            "failed to select Iceberg equality-delete key columns",
            error,
        )
        .with_fact("equality_delete_index", delete_index)
        .with_fact(
            "equality_delete_stable_id",
            delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()),
        )
    })
}

pub(super) fn unique_position_delete_column_name(frame: &DataFrame) -> String {
    let names = frame.column_names();
    crate::naming::unique_name_with_suffix_from(ICEBERG_POSITION_DELETE_ROW_POSITION_COLUMN, 0, |candidate| {
        names.contains(&candidate)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{CredentialMode, ParquetEqualityDeleteFile, ResolvedParquetSource};

    #[test]
    fn position_and_equality_delete_files_build_typed_remote_kvikio_sources() {
        let context = ExecutionContext::builder().build().expect("execution context builds");

        let position_delete = crate::source::ParquetPositionDeleteFile::from_resolved_parquet_source(
            "position-delete",
            ResolvedParquetSource::remote_kvikio(
                "s3://bucket/table/position-delete.parquet",
                CredentialMode::SharedProcessEnv,
            ),
        )
        .expect("position delete source is valid");
        let position_file = position_delete.to_local_parquet_file();
        let position_schema = Arc::new(Schema::new(vec![
            Field::new(ICEBERG_POSITION_DELETE_FILE_PATH_COLUMN, ArrowDataType::Utf8, false),
            Field::new(ICEBERG_POSITION_DELETE_POS_COLUMN, ArrowDataType::Int64, false),
        ]));
        let position_source = LocalParquetSource::from_files(
            vec![position_file.clone()],
            position_schema,
            vec![ICEBERG_POSITION_DELETE_FILE_PATH_COLUMN.to_string(), ICEBERG_POSITION_DELETE_POS_COLUMN.to_string()],
        )
        .with_reader_predicate(
            position_delete_reader_predicate(&position_delete, "s3://bucket/table/data.parquet", None)
                .expect("position delete predicate builds"),
        );
        let (position_cudf_source, position_chunk) =
            cudf_local_parquet_source_file_selection(&context, &position_source, &position_file, 0, 0)
                .expect("position delete file uses typed remote source builder");
        assert_eq!(
            position_cudf_source.files()[0].remote_source().unwrap().uri(),
            "s3://bucket/table/position-delete.parquet"
        );
        assert_eq!(
            position_chunk.remote_source().unwrap().read_mode(),
            cudf::io::parquet::RemoteReadMode::RequireKvikio
        );
        assert!(position_cudf_source.reader_predicate().is_some());

        let equality_delete = ParquetEqualityDeleteFile::from_resolved_parquet_source(
            "equality-delete",
            ResolvedParquetSource::remote_kvikio(
                "s3://bucket/table/equality-delete.parquet",
                CredentialMode::SharedProcessEnv,
            ),
            vec![1],
            vec!["id".to_string()],
        )
        .expect("equality delete source is valid");
        let equality_file = equality_delete.to_local_parquet_file();
        let equality_source = LocalParquetSource::from_files(
            vec![equality_file.clone()],
            Arc::new(Schema::new(vec![Field::new("id", ArrowDataType::Int64, true)])),
            vec!["id".to_string()],
        );
        let (equality_cudf_source, equality_chunk) =
            cudf_local_parquet_source_file_selection(&context, &equality_source, &equality_file, 0, 1)
                .expect("equality delete file uses typed remote source builder");
        assert_eq!(
            equality_cudf_source.files()[0].remote_source().unwrap().uri(),
            "s3://bucket/table/equality-delete.parquet"
        );
        assert_eq!(
            equality_chunk.remote_source().unwrap().read_mode(),
            cudf::io::parquet::RemoteReadMode::RequireKvikio
        );
    }
}
