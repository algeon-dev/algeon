use std::{collections::BTreeSet, path::PathBuf, sync::Arc};

use arrow::datatypes::SchemaRef;
use cudf::{ExecutionOverrides, Runtime, frame::dataframe::ParquetReadOptions};
use datafusion::{
    common::Result as DataFusionResult,
    datasource::{
        physical_plan::{FileScanConfig, FileSource, ParquetSource},
        source::DataSourceExec,
    },
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, Column, Literal, NotExpr},
    },
};

use super::{batch::CudfBatch, metrics::CudfGpuIslandMetrics, normalize::NormalizedGpuIslandPlan};
use crate::error::{Error, ErrorCode, Result as AdapterResult};

#[derive(Debug, Clone)]
pub(super) struct CudfParquetScan {
    partitions: Vec<CudfParquetPartition>,
    read_columns: Vec<String>,
    file_schema: SchemaRef,
}

impl CudfParquetScan {
    pub(super) fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    pub(super) fn read_columns(&self) -> &[String] {
        &self.read_columns
    }
}

#[derive(Debug, Clone)]
struct CudfParquetPartition {
    files: Vec<CudfParquetFile>,
}

#[derive(Debug, Clone)]
struct CudfParquetFile {
    path: PathBuf,
    size: u64,
    row_groups: Option<Vec<i32>>,
}

pub(super) fn try_local_parquet_scan(plan: &NormalizedGpuIslandPlan) -> AdapterResult<Option<CudfParquetScan>> {
    let Some(source_exec) = plan.input.as_any().downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };
    let Some((file_scan, parquet_source)) = source_exec.downcast_to_file_source::<ParquetSource>() else {
        return Ok(None);
    };

    validate_local_parquet_scan(file_scan, parquet_source)?;
    let read_columns = required_parquet_read_columns(file_scan, plan)?;
    let partitions = file_scan
        .file_groups
        .iter()
        .map(|group| {
            let files = group
                .files()
                .iter()
                .map(|file| {
                    if !file.partition_values.is_empty() {
                        return Err(local_parquet_rejection("partition columns are not supported in GPU parquet scan"));
                    }
                    if file.range.is_some() {
                        return Err(local_parquet_rejection(
                            "byte-range parquet files are not supported in GPU parquet scan",
                        ));
                    }
                    Ok(CudfParquetFile {
                        path: local_object_path_to_filesystem(file.object_meta.location.as_ref()),
                        size: file.object_meta.size,
                        row_groups: None,
                    })
                })
                .collect::<AdapterResult<Vec<_>>>()?;
            Ok(CudfParquetPartition { files })
        })
        .collect::<AdapterResult<Vec<_>>>()?;

    Ok(Some(CudfParquetScan { partitions, read_columns, file_schema: Arc::clone(file_scan.file_schema()) }))
}

fn validate_local_parquet_scan(file_scan: &FileScanConfig, parquet_source: &ParquetSource) -> AdapterResult<()> {
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        return Err(local_parquet_rejection("non-local parquet object stores are not supported in GPU parquet scan"));
    }
    if !file_scan.table_partition_cols().is_empty() {
        return Err(local_parquet_rejection("table partition columns are not supported in GPU parquet scan"));
    }
    if file_scan.expr_adapter_factory.is_some() {
        return Err(local_parquet_rejection("schema expression adapters are not supported in GPU parquet scan"));
    }
    if file_scan.limit.is_some() {
        return Err(local_parquet_rejection("pushed-down scan limits are not supported in GPU parquet scan"));
    }
    if file_scan.preserve_order {
        return Err(local_parquet_rejection(
            "order-sensitive file scan semantics are not supported in GPU parquet scan",
        ));
    }
    if parquet_source.filter().is_some() {
        return Err(local_parquet_rejection("pushed-down parquet predicates are not supported in GPU parquet scan"));
    }
    Ok(())
}

fn required_parquet_read_columns(
    file_scan: &FileScanConfig,
    plan: &NormalizedGpuIslandPlan,
) -> AdapterResult<Vec<String>> {
    let file_schema = file_scan.file_schema();
    let mut required = BTreeSet::new();

    if let Some(projection) = file_scan.file_source().projection() {
        for index in projection.column_indices() {
            if index >= file_schema.fields().len() {
                return Err(local_parquet_rejection(
                    "partition-column projections are not supported in GPU parquet scan",
                ));
            }
            required.insert(file_schema.field(index).name().clone());
        }
    }

    if let Some(filter) = &plan.filter {
        collect_required_columns(filter, &mut required)?;
    }

    if !plan.projection.is_empty() {
        for projection in &plan.projection {
            collect_required_columns(&projection.expr, &mut required)?;
        }
    } else if let Some(aggregate) = &plan.aggregate {
        required.extend(aggregate.keys.iter().map(|key| key.input_column.clone()));
        required.extend(aggregate.aggs.iter().map(|agg| agg.input_column.clone()));
    } else if !plan.sort.is_empty() {
        for sort_expr in &plan.sort {
            collect_required_columns(&sort_expr.expr, &mut required)?;
        }
    }

    if required.is_empty() {
        required.extend(file_schema.fields().iter().map(|field| field.name().clone()));
    }

    for column in &required {
        if file_schema.index_of(column).is_err() {
            return Err(local_parquet_rejection("GPU parquet scan requires a column outside the file schema")
                .with_fact("column", column));
        }
    }

    Ok(required.into_iter().collect())
}

fn collect_required_columns(expr: &Arc<dyn PhysicalExpr>, required: &mut BTreeSet<String>) -> AdapterResult<()> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        required.insert(column.name().to_string());
        return Ok(());
    }
    if expr.as_any().downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        collect_required_columns(binary.left(), required)?;
        collect_required_columns(binary.right(), required)?;
        return Ok(());
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        collect_required_columns(not.arg(), required)?;
        return Ok(());
    }
    Err(Error::new(
        ErrorCode::UnsupportedExpression,
        "datafusion_cudf.parquet_scan.required_columns",
        "cannot collect required columns from unsupported physical expression",
    )
    .with_fact("expression", expr))
}

fn local_object_path_to_filesystem(location: &str) -> PathBuf {
    PathBuf::from("/").join(location)
}

fn local_parquet_rejection(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedOperator, "datafusion_cudf.parquet_scan.recognize", message)
}

pub(super) fn read_local_parquet_partition(
    runtime: &Runtime,
    ctx: &ExecutionOverrides<'_>,
    scan: &CudfParquetScan,
    partition: usize,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<Option<CudfBatch>> {
    let Some(partition) = scan.partitions.get(partition) else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_cudf.parquet_scan.execute",
            "local parquet partition does not exist",
        )
        .with_fact("partition", partition)
        .with_fact("partition_count", scan.partitions.len())
        .into());
    };
    let mut batches = Vec::with_capacity(partition.files.len());
    for file in &partition.files {
        batches.push(read_local_parquet_file(runtime, scan, file, metrics)?);
    }

    let mut batches = batches.into_iter();
    let Some(batch) = batches.next() else {
        return Ok(None);
    };
    let other_batches = batches.collect::<Vec<_>>();
    batch.concat_many(ctx, &other_batches, metrics).map(Some)
}

fn read_local_parquet_file(
    runtime: &Runtime,
    scan: &CudfParquetScan,
    file: &CudfParquetFile,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<CudfBatch> {
    debug_assert!(scan.read_columns.iter().all(|column| scan.file_schema.index_of(column).is_ok()));
    let columns = scan.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let options = if let Some(row_groups) = &file.row_groups {
        metrics.row_groups_selected.add(row_groups.len());
        ParquetReadOptions::builder()
            .filepath(&file.path)
            .columns(&columns)
            .row_groups(row_groups)
            .with_runtime(runtime)
            .build()
    } else {
        ParquetReadOptions::builder().filepath(&file.path).columns(&columns).with_runtime(runtime).build()
    };
    let read_timer = metrics.parquet_read_time.timer();
    let frame = cudf::DataFrame::read_parquet_with_options(options).map_err(|error| {
        Error::from_cudf("datafusion_cudf.parquet_scan.read", "failed to read local Parquet file with cudf", error)
            .with_fact("path", file.path.display())
    })?;
    read_timer.done();
    metrics.input_batches.add(1);
    metrics.input_rows.add(frame.num_rows());
    metrics.parquet_files.add(1);
    metrics.parquet_bytes_known.add(file.size as usize);
    metrics.read_columns.add(scan.read_columns.len());
    Ok(CudfBatch::new(frame))
}
