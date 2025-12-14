use std::{any::Any, sync::Arc};

use arrow::record_batch::RecordBatch;
use cudf::{ExecutionOverrides, Runtime};
use datafusion::{
    common::{Result as DataFusionResult, utils::memory::get_record_batch_memory_size},
    execution::context::TaskContext,
    physical_expr::{PhysicalExpr, PhysicalSortExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricsSet, RecordOutput},
        projection::ProjectionExpr,
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, TryStreamExt};

use super::{
    batch::{upload_batch, upload_batch_with_size},
    config::CudfExecutionConfig,
    input::GpuIslandInput,
    lowering::GpuAggregate,
    metrics::CudfGpuIslandMetrics,
    normalize::normalize_gpu_island_plan,
    operators::CudfOperatorChain,
    parquet,
};
use crate::error::{Error, ErrorCode};

#[derive(Debug, Clone)]
pub struct CudfGpuIslandExec {
    input: GpuIslandInput,
    filter: Option<Arc<dyn PhysicalExpr>>,
    projection: Vec<ProjectionExpr>,
    aggregate: Option<GpuAggregate>,
    sort: Vec<PhysicalSortExpr>,
    local_limit: Option<usize>,
    execution_config: CudfExecutionConfig,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl CudfGpuIslandExec {
    pub fn try_new_from_plan(plan: Arc<dyn ExecutionPlan>) -> crate::error::Result<Self> {
        Self::try_new_from_plan_with_execution_config(plan, CudfExecutionConfig::default())
    }

    pub fn try_new_from_plan_with_execution_config(
        plan: Arc<dyn ExecutionPlan>,
        execution_config: CudfExecutionConfig,
    ) -> crate::error::Result<Self> {
        let normalized = normalize_gpu_island_plan(Arc::clone(&plan))?.ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.gpu_island.try_new",
                "cannot build GPU island from physical plan",
            )
            .with_fact("plan_node", plan.name())
        })?;
        let input = GpuIslandInput::try_from_normalized(&normalized)?;
        Ok(Self {
            input,
            filter: normalized.filter,
            projection: normalized.projection,
            aggregate: normalized.aggregate,
            sort: normalized.sort,
            local_limit: normalized.local_limit,
            execution_config,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::clone(plan.properties()),
        })
    }

    fn operator_list(&self) -> String {
        let mut operators = Vec::with_capacity(4);
        if self.filter.is_some() {
            operators.push("filter");
        }
        if !self.projection.is_empty() {
            operators.push("projection");
        }
        if self.aggregate.is_some() {
            operators.push("aggregate");
        }
        if !self.sort.is_empty() {
            operators.push("sort");
        }
        if self.local_limit.is_some() {
            operators.push("local_limit");
        }
        operators.join(",")
    }

    pub fn input_display_name(&self) -> &'static str {
        self.input.display_name()
    }

    pub fn input_boundary_name(&self) -> &'static str {
        self.input.input_boundary()
    }

    pub fn local_parquet_read_columns(&self) -> Option<Vec<String>> {
        self.input.local_parquet_read_columns().map(<[String]>::to_vec)
    }
}

impl DisplayAs for CudfGpuIslandExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let operators = self.operator_list();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "CudfGpuIslandExec: operators=[{}], internal_batch=CudfBatch, internal_execution=operator_chain, input={}, projection_exprs={}",
                    operators,
                    self.input.display_name(),
                    self.projection.len(),
                )?;
                if let Some(aggregate) = &self.aggregate {
                    write!(f, ", group_keys={}, aggregate_exprs={}", aggregate.keys.len(), aggregate.aggs.len(),)?;
                }
                if !self.sort.is_empty() {
                    write!(f, ", sort_exprs={}", self.sort.len())?;
                }
                if let Some(fetch) = self.local_limit {
                    write!(f, ", local_limit={fetch}")?;
                }
                if let Some(read_columns) = self.input.local_parquet_read_columns() {
                    write!(f, ", read_columns=[{}]", read_columns.join(","))?;
                }
                write!(f, ", input_boundary={}, output_boundary=device_to_host", self.input.input_boundary())
            },
            DisplayFormatType::TreeRender => {
                writeln!(f, "operators={operators}")?;
                writeln!(f, "internal_batch=CudfBatch")?;
                writeln!(f, "internal_execution=operator_chain")?;
                writeln!(f, "input={}", self.input.display_name())?;
                writeln!(f, "projection_exprs={}", self.projection.len())?;
                if let Some(aggregate) = &self.aggregate {
                    writeln!(f, "group_keys={}", aggregate.keys.len())?;
                    writeln!(f, "aggregate_exprs={}", aggregate.aggs.len())?;
                }
                if !self.sort.is_empty() {
                    writeln!(f, "sort_exprs={}", self.sort.len())?;
                }
                if let Some(fetch) = self.local_limit {
                    writeln!(f, "local_limit={fetch}")?;
                }
                if let Some(read_columns) = self.input.local_parquet_read_columns() {
                    writeln!(f, "read_columns=[{}]", read_columns.join(","))?;
                }
                writeln!(f, "input_boundary={}", self.input.input_boundary())?;
                writeln!(f, "output_boundary=device_to_host")
            },
        }
    }
}

impl ExecutionPlan for CudfGpuIslandExec {
    fn name(&self) -> &str {
        "CudfGpuIslandExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.input.child().map(|_| self.sort.is_empty()).into_iter().collect()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.input.child().into_iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let input = match &self.input {
            GpuIslandInput::ChildRecordBatches(_) => {
                if children.len() != 1 {
                    return Err(Error::new(
                        ErrorCode::InvalidChildCount,
                        "datafusion_cudf.gpu_island.with_new_children",
                        "GPU island expected exactly one child",
                    )
                    .with_fact("actual_child_count", children.len())
                    .into());
                }
                GpuIslandInput::ChildRecordBatches(children.swap_remove(0))
            },
            GpuIslandInput::LocalParquetScan(scan) => {
                if !children.is_empty() {
                    return Err(Error::new(
                        ErrorCode::InvalidChildCount,
                        "datafusion_cudf.gpu_island.with_new_children",
                        "local parquet GPU island expected no children",
                    )
                    .with_fact("actual_child_count", children.len())
                    .into());
                }
                GpuIslandInput::LocalParquetScan(scan.clone())
            },
        };
        Ok(Arc::new(Self {
            input,
            filter: self.filter.clone(),
            projection: self.projection.clone(),
            aggregate: self.aggregate.clone(),
            sort: self.sort.clone(),
            local_limit: self.local_limit,
            execution_config: self.execution_config.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::clone(&self.cache),
        }))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        let schema = self.schema();
        let filter = self.filter.clone();
        let projection = self.projection.clone();
        let aggregate = self.aggregate.clone();
        let sort = self.sort.clone();
        let local_limit = self.local_limit;
        let runtime = self.execution_config.runtime_for_partition(partition)?;
        let metrics = CudfGpuIslandMetrics::new(&self.metrics, partition);
        let chain = CudfOperatorChain::new(filter, projection, aggregate, sort, local_limit);

        if chain.requires_single_output_batch() {
            let input = self.input.clone();
            let input_partitions = if self.cache.output_partitioning().partition_count() == 1 {
                (0..self.input.output_partition_count()).collect::<Vec<_>>()
            } else {
                vec![partition]
            };
            let output = futures::stream::once(async move {
                let output =
                    execute_blocking_island(input, input_partitions, context, runtime.as_ref(), &chain, &metrics)
                        .await?;
                match output {
                    Some(batch) => Ok(Some(batch.record_output(&metrics.baseline))),
                    None => {
                        metrics.baseline.done();
                        Ok(None)
                    },
                }
            })
            .try_filter_map(|batch| async move { Ok(batch) });
            return Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)));
        }

        let GpuIslandInput::ChildRecordBatches(input_plan) = &self.input else {
            let input = self.input.clone();
            let output = futures::stream::once(async move {
                let output = execute_local_parquet_partition(input, partition, runtime.as_ref(), &chain, &metrics)?;
                match output {
                    Some(batch) => Ok(Some(batch.record_output(&metrics.baseline))),
                    None => {
                        metrics.baseline.done();
                        Ok(None)
                    },
                }
            })
            .try_filter_map(|batch| async move { Ok(batch) });
            return Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)));
        };

        let input = input_plan.execute(partition, context)?;
        let output = futures::stream::try_unfold(
            (input, local_limit, metrics, chain),
            move |(mut input, mut remaining_rows, metrics, chain)| {
                let runtime = Arc::clone(&runtime);
                let metrics = metrics.clone();

                async move {
                    if remaining_rows == Some(0) {
                        metrics.baseline.done();
                        return Ok(None);
                    }

                    let Some(batch) = input.next().await else {
                        metrics.baseline.done();
                        return Ok(None);
                    };
                    let batch = batch?;
                    let output = execute_record_batch(&runtime, &chain, remaining_rows.as_mut(), batch, &metrics)?
                        .record_output(&metrics.baseline);
                    Ok(Some((output, (input, remaining_rows, metrics, chain))))
                }
            },
        );

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

async fn execute_blocking_island(
    input: GpuIslandInput,
    input_partitions: Vec<usize>,
    context: Arc<TaskContext>,
    runtime: &Runtime,
    chain: &CudfOperatorChain,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<Option<RecordBatch>> {
    let ctx = ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let mut frames = Vec::new();

    match input {
        GpuIslandInput::ChildRecordBatches(input) => {
            for child_partition in input_partitions {
                let mut stream = input.execute(child_partition, Arc::clone(&context))?;
                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    frames.push(upload_batch(runtime, batch, metrics)?);
                }
            }
        },
        GpuIslandInput::LocalParquetScan(scan) => {
            for partition in input_partitions {
                if let Some(batch) = parquet::read_local_parquet_partition(runtime, &ctx, &scan, partition, metrics)? {
                    frames.push(batch);
                }
            }
        },
    }

    let mut frames = frames.into_iter();
    let Some(batch) = frames.next() else {
        return Ok(None);
    };
    let other_batches = frames.collect::<Vec<_>>();
    let batch = batch.concat_many(&ctx, &other_batches, metrics)?;

    let mut remaining_rows = chain.local_limit;
    let batch = chain.execute(&ctx, remaining_rows.as_mut(), batch, metrics)?;
    batch.materialize(metrics).map(Some)
}

fn execute_local_parquet_partition(
    input: GpuIslandInput,
    partition: usize,
    runtime: &Runtime,
    chain: &CudfOperatorChain,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<Option<RecordBatch>> {
    let GpuIslandInput::LocalParquetScan(scan) = input else {
        return Err(Error::new(
            ErrorCode::Internal,
            "datafusion_cudf.parquet_scan.execute_partition",
            "expected local parquet scan input",
        )
        .into());
    };
    let ctx = ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let Some(batch) = parquet::read_local_parquet_partition(runtime, &ctx, &scan, partition, metrics)? else {
        return Ok(None);
    };
    let mut remaining_rows = chain.local_limit;
    let batch = chain.execute(&ctx, remaining_rows.as_mut(), batch, metrics)?;
    batch.materialize(metrics).map(Some)
}

fn execute_record_batch(
    runtime: &Runtime,
    chain: &CudfOperatorChain,
    remaining_rows: Option<&mut usize>,
    batch: RecordBatch,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<RecordBatch> {
    let input_rows = batch.num_rows();
    let input_bytes = get_record_batch_memory_size(&batch);
    metrics.input_batches.add(1);
    metrics.input_rows.add(input_rows);

    let ctx = ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let frame = upload_batch_with_size(runtime, batch, input_rows, input_bytes, metrics)?;
    let batch = chain.execute(&ctx, remaining_rows, frame, metrics)?;
    batch.materialize(metrics)
}
