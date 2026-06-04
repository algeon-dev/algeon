//! Object-store footer pruning for native plans produced from table formats.

use std::{future::Future, pin::Pin, sync::Arc};

use datafusion::{
    common::Result,
    execution::{object_store::ObjectStoreUrl, runtime_env::RuntimeEnv},
    object_store::{ObjectStoreExt, path::Path as ObjectStorePath},
    parquet::{
        arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask, async_reader::ParquetObjectReader},
        file::{metadata::ParquetMetaData, statistics::Statistics},
    },
    physical_plan::ExecutionPlan,
};
use futures::TryStreamExt;
use nexus_query_engine::{
    DimensionDerivedKeyRange as KeyRange, Expr, LocalParquetSource, NodeId, ParquetSourceLocation, PlanNode, QueryPlan,
    SourceColumnRef, SourceNode, apply_dimension_derived_pruning_with_stats, dimension_key_range_from_batch,
    discover_dimension_derived_pruning_requests, seed_dimension_derived_reader_predicates,
};

use crate::NativeCudfExec;

const MAX_DIMENSION_ROWS: usize = 1_000_000;
const MAX_TARGET_FILES_ACROSS_REQUESTS: usize = 4_000;

type BoxFutureResult<T> = Pin<Box<dyn Future<Output = Result<T>> + Send>>;

/// Applies footer-only dimension-derived row-group pruning to whole-plan native Iceberg scans.
pub async fn prune_table_format_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    runtime_env: &Arc<RuntimeEnv>,
) -> Result<Arc<dyn ExecutionPlan>> {
    prune_execution_plan(plan, runtime_env).await
}

fn prune_execution_plan(
    plan: Arc<dyn ExecutionPlan>,
    runtime_env: &Arc<RuntimeEnv>,
) -> BoxFutureResult<Arc<dyn ExecutionPlan>> {
    let runtime_env = Arc::clone(runtime_env);
    Box::pin(async move {
        if let Some(exec) = plan.as_any().downcast_ref::<NativeCudfExec>() {
            return Ok(prune_native_exec(exec, &runtime_env).await);
        }

        let children = plan.children();
        if children.is_empty() {
            return Ok(plan);
        }

        let mut rewritten_children = Vec::with_capacity(children.len());
        let mut changed = false;
        for child in children {
            let rewritten = prune_execution_plan(Arc::clone(child), &runtime_env).await?;
            changed |= !Arc::ptr_eq(child, &rewritten);
            rewritten_children.push(rewritten);
        }

        if changed { plan.with_new_children(rewritten_children) } else { Ok(plan) }
    })
}

async fn prune_native_exec(exec: &NativeCudfExec, runtime_env: &Arc<RuntimeEnv>) -> Arc<dyn ExecutionPlan> {
    let mut plan = seed_dimension_derived_reader_predicates(exec.native_plan().clone());
    let requests = discover_dimension_derived_pruning_requests(&plan);
    if requests.is_empty() {
        return Arc::new(exec.clone());
    }

    let target_file_count: usize = requests
        .iter()
        .filter_map(|request| local_parquet_source(&plan, request.target_source_node_id))
        .filter(|source| !source.files.iter().all(|file| file.location.as_local_path().is_some()))
        .map(|source| source.files.len())
        .sum();
    if target_file_count > MAX_TARGET_FILES_ACROSS_REQUESTS {
        eprintln!(
            "[lakehouse_pruning] event=skip reason=too_many_target_files target_file_count={} max_target_file_count={}",
            target_file_count, MAX_TARGET_FILES_ACROSS_REQUESTS,
        );
        return Arc::new(exec.clone());
    }

    let mut any_applied = false;
    for request in requests {
        match prune_request(plan.clone(), runtime_env, request).await {
            Ok(RequestOutcome::Applied { plan: patched, target_source_node_id, before, after }) => {
                eprintln!(
                    "[lakehouse_pruning] event=applied target_source_node_id={} before_row_groups={} after_row_groups={}",
                    target_source_node_id.index(),
                    before,
                    after,
                );
                plan = patched;
                any_applied = true;
            },
            Ok(RequestOutcome::Skipped { target_source_node_id, reason }) => {
                eprintln!(
                    "[lakehouse_pruning] event=skip target_source_node_id={} reason={}",
                    target_source_node_id.index(),
                    reason,
                );
            },
            Err(error) => {
                eprintln!("[lakehouse_pruning] event=skip reason=request_error error={}", stable_log_field(&error));
            },
        }
    }

    if !any_applied {
        return Arc::new(exec.clone());
    }

    match NativeCudfExec::try_new(plan, exec.execution_config().clone()) {
        Ok(patched) => Arc::new(patched),
        Err(error) => {
            eprintln!(
                "[lakehouse_pruning] event=skip reason=rebuild_failed error={}",
                stable_log_field(&error.to_string()),
            );
            Arc::new(exec.clone())
        },
    }
}

enum RequestOutcome {
    Applied { plan: QueryPlan, target_source_node_id: NodeId, before: usize, after: usize },
    Skipped { target_source_node_id: NodeId, reason: &'static str },
}

async fn prune_request(
    plan: QueryPlan,
    runtime_env: &Arc<RuntimeEnv>,
    request: nexus_query_engine::DimensionDerivedPruningRequest,
) -> std::result::Result<RequestOutcome, String> {
    let target_source_node_id = request.target_source_node_id;
    let Some(target_source) = local_parquet_source(&plan, target_source_node_id) else {
        return Ok(RequestOutcome::Skipped { target_source_node_id, reason: "missing_target_source" });
    };
    if target_source.files.iter().all(|file| file.location.as_local_path().is_some()) {
        return Ok(RequestOutcome::Skipped { target_source_node_id, reason: "target_is_local_path" });
    }
    if !target_source.files.iter().all(|file| matches!(file.location, ParquetSourceLocation::ObjectStoreUri { .. })) {
        return Ok(RequestOutcome::Skipped { target_source_node_id, reason: "mixed_or_unsupported_target_locations" });
    }

    let Some(dimension_source) = local_parquet_source(&plan, request.dimension_source_node_id) else {
        return Ok(RequestOutcome::Skipped { target_source_node_id, reason: "missing_dimension_source" });
    };
    let Some(predicate) = dimension_source.reader_predicate.as_ref() else {
        return Ok(RequestOutcome::Skipped { target_source_node_id, reason: "missing_dimension_predicate" });
    };

    let Some(range) =
        derive_dimension_key_range(runtime_env, dimension_source, &request.dimension_key_column, predicate).await?
    else {
        return Ok(RequestOutcome::Skipped { target_source_node_id, reason: "empty_or_unsupported_dimension_range" });
    };

    let SourceColumnRef::File(target_key_column_index) = target_source
        .source_column_ref(&request.target_key_column)
        .ok_or_else(|| "missing_target_key_column".to_string())?
    else {
        return Ok(RequestOutcome::Skipped { target_source_node_id, reason: "target_key_not_file_column" });
    };

    let mut per_file_row_group_key_ranges = Vec::with_capacity(target_source.files.len());
    for file in &target_source.files {
        let object = object_store_file(&file.location)?;
        let store = runtime_env.object_store(&object.store_url).map_err(|error| error.to_string())?;
        let metadata = parquet_metadata(store, object.path, file.file_size_bytes).await?;
        let selected_row_groups = selected_or_all_row_groups(file.row_groups.as_deref(), metadata.num_row_groups());
        per_file_row_group_key_ranges.push(row_group_key_ranges(
            metadata.as_ref(),
            target_key_column_index,
            &selected_row_groups,
        ));
    }

    let result =
        apply_dimension_derived_pruning_with_stats(plan, target_source_node_id, &per_file_row_group_key_ranges, range);
    if result.applied {
        Ok(RequestOutcome::Applied {
            plan: result.plan,
            target_source_node_id,
            before: result.before,
            after: result.after,
        })
    } else {
        Ok(RequestOutcome::Skipped { target_source_node_id, reason: "no_fact_row_group_reduction" })
    }
}

async fn derive_dimension_key_range(
    runtime_env: &Arc<RuntimeEnv>,
    source: &LocalParquetSource,
    key_column: &str,
    predicate: &Expr,
) -> std::result::Result<Option<KeyRange>, String> {
    let mut range = None;
    let mut total_rows = 0usize;
    for file in &source.files {
        let object = object_store_file(&file.location)?;
        let store = runtime_env.object_store(&object.store_url).map_err(|error| error.to_string())?;
        let object_meta = store.head(&object.path).await.map_err(|error| error.to_string())?;
        let reader = ParquetObjectReader::new(store, object.path).with_file_size(object_meta.size);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.map_err(|error| error.to_string())?;
        let row_group_count = builder.metadata().num_row_groups();
        let row_groups = selected_or_all_row_groups(file.row_groups.as_deref(), row_group_count);
        let projection = ProjectionMask::all();
        let mut stream = builder
            .with_projection(projection)
            .with_row_groups(row_groups)
            .build()
            .map_err(|error| error.to_string())?;
        while let Some(batch) = stream.try_next().await.map_err(|error| error.to_string())? {
            total_rows = total_rows.saturating_add(batch.num_rows());
            if total_rows > MAX_DIMENSION_ROWS {
                return Err(format!(
                    "dimension_row_guard_exceeded total_rows={} max_rows={}",
                    total_rows, MAX_DIMENSION_ROWS
                ));
            }
            range = dimension_key_range_from_batch(range, &batch, key_column, predicate)
                .ok_or_else(|| "unsupported_dimension_predicate_or_key".to_string())?;
        }
    }
    Ok(range)
}

async fn parquet_metadata(
    store: Arc<dyn datafusion::object_store::ObjectStore>,
    path: ObjectStorePath,
    file_size_bytes: Option<u64>,
) -> std::result::Result<Arc<ParquetMetaData>, String> {
    let object_meta = store.head(&path).await.map_err(|error| error.to_string())?;
    let reader = ParquetObjectReader::new(store, path).with_file_size(file_size_bytes.unwrap_or(object_meta.size));
    let builder = ParquetRecordBatchStreamBuilder::new(reader).await.map_err(|error| error.to_string())?;
    Ok(Arc::clone(builder.metadata()))
}

fn row_group_key_ranges(
    metadata: &ParquetMetaData,
    column_index: usize,
    selected_row_groups: &[usize],
) -> Vec<Option<KeyRange>> {
    selected_row_groups
        .iter()
        .map(|row_group_index| {
            let row_group = metadata.row_groups().get(*row_group_index)?;
            let statistics = row_group.columns().get(column_index)?.statistics()?;
            match statistics {
                Statistics::Int32(stats) => {
                    Some(KeyRange { min: i64::from(*stats.min_opt()?), max: i64::from(*stats.max_opt()?) })
                },
                Statistics::Int64(stats) => Some(KeyRange { min: *stats.min_opt()?, max: *stats.max_opt()? }),
                _ => None,
            }
        })
        .collect()
}

fn selected_or_all_row_groups(selected: Option<&[usize]>, row_group_count: usize) -> Vec<usize> {
    selected.map_or_else(|| (0..row_group_count).collect(), |row_groups| row_groups.to_vec())
}

fn local_parquet_source(plan: &QueryPlan, node_id: NodeId) -> Option<&LocalParquetSource> {
    match &plan.node(node_id)?.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => Some(source),
            SourceNode::GpuResident(_) | SourceNode::LiteralBatch(_) => None,
        },
        _ => None,
    }
}

struct ObjectStoreFile {
    store_url: ObjectStoreUrl,
    path: ObjectStorePath,
}

fn object_store_file(location: &ParquetSourceLocation) -> std::result::Result<ObjectStoreFile, String> {
    let ParquetSourceLocation::ObjectStoreUri { uri, .. } = location else {
        return Err("not_object_store_uri".to_string());
    };
    let (scheme, rest) = uri.split_once("://").ok_or_else(|| "missing_uri_scheme".to_string())?;
    let (authority, path) = rest.split_once('/').ok_or_else(|| "missing_object_path".to_string())?;
    if authority.is_empty() {
        return Err("missing_uri_authority".to_string());
    }
    if path.is_empty() {
        return Err("missing_object_path".to_string());
    }
    let store_url = ObjectStoreUrl::parse(format!("{scheme}://{authority}")).map_err(|error| error.to_string())?;
    let path = ObjectStorePath::parse(path).map_err(|error| error.to_string())?;
    Ok(ObjectStoreFile { store_url, path })
}

fn stable_log_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}
