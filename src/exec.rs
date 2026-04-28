use std::{any::Any, sync::Arc};

use arrow::record_batch::RecordBatch;
use cudf::{
    AggSpec, Expr, GroupByAgg, Runtime, SortOptions as CudfSortOptions,
    ops::{col, lit},
};
use datafusion::{
    common::{Result as DataFusionResult, ScalarValue, utils::memory::get_record_batch_memory_size},
    execution::context::TaskContext,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, PhysicalSortExpr,
        expressions::{BinaryExpr, Column, Literal, NotExpr},
    },
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
        SendableRecordBatchStream,
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        limit::LocalLimitExec,
        metrics::{
            BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, MetricsSet, RecordOutput, Time,
        },
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, TryStreamExt};

use crate::error::{Error, ErrorCode, Result as AdapterResult};

/// Execution resources used by cudf-backed GPU islands.
///
/// The default config builds a cudf [`Runtime`] when a partition executes.
/// Supplying a runtime lets callers make placement, stream, allocator, and
/// boundary-observer policy explicit before the adapter starts GPU work.
#[derive(Debug, Clone, Default)]
pub struct CudfExecutionConfig {
    runtime: Option<Arc<Runtime>>,
}

impl CudfExecutionConfig {
    /// Create an execution config that builds a default cudf runtime at execution time.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use one caller-provided runtime for GPU island execution.
    pub fn with_runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self
    }

    /// Use one shared caller-provided runtime for GPU island execution.
    pub fn with_shared_runtime(mut self, runtime: Arc<Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub(crate) fn runtime_for_partition(&self, _partition: usize) -> AdapterResult<Arc<Runtime>> {
        if let Some(runtime) = &self.runtime {
            return Ok(Arc::clone(runtime));
        }

        Runtime::builder().build().map(Arc::new).map_err(|error| {
            Error::from_cudf("datafusion_cudf.runtime.build", "failed to initialize cudf runtime", error)
        })
    }
}

#[derive(Debug, Clone)]
pub struct CudfGpuIslandExec {
    input: Arc<dyn ExecutionPlan>,
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
    pub fn try_new_from_plan(plan: Arc<dyn ExecutionPlan>) -> AdapterResult<Self> {
        Self::try_new_from_plan_with_execution_config(plan, CudfExecutionConfig::default())
    }

    pub fn try_new_from_plan_with_execution_config(
        plan: Arc<dyn ExecutionPlan>,
        execution_config: CudfExecutionConfig,
    ) -> AdapterResult<Self> {
        if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>()
            && let Some(sort) = limit.input().as_any().downcast_ref::<SortExec>()
            && let Some(aggregate) = sort.input().as_any().downcast_ref::<AggregateExec>()
            && let Some(projection) = aggregate.input().as_any().downcast_ref::<ProjectionExec>()
            && let Some(filter) = projection.input().as_any().downcast_ref::<FilterExec>()
        {
            return Ok(Self {
                input: Arc::clone(filter.input()),
                filter: Some(Arc::clone(filter.predicate())),
                projection: projection.expr().to_vec(),
                aggregate: Some(lower_aggregate_exec(aggregate)?),
                sort: sort.expr().to_vec(),
                local_limit: Some(limit.fetch()),
                execution_config,
                metrics: ExecutionPlanMetricsSet::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>()
            && let Some(sort) = limit.input().as_any().downcast_ref::<SortExec>()
            && let Some(projection) = sort.input().as_any().downcast_ref::<ProjectionExec>()
            && let Some(filter) = projection.input().as_any().downcast_ref::<FilterExec>()
        {
            return Ok(Self {
                input: Arc::clone(filter.input()),
                filter: Some(Arc::clone(filter.predicate())),
                projection: projection.expr().to_vec(),
                aggregate: None,
                sort: sort.expr().to_vec(),
                local_limit: Some(limit.fetch()),
                execution_config,
                metrics: ExecutionPlanMetricsSet::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>()
            && let Some(projection) = limit.input().as_any().downcast_ref::<ProjectionExec>()
            && let Some(filter) = projection.input().as_any().downcast_ref::<FilterExec>()
        {
            return Ok(Self {
                input: Arc::clone(filter.input()),
                filter: Some(Arc::clone(filter.predicate())),
                projection: projection.expr().to_vec(),
                aggregate: None,
                sort: Vec::new(),
                local_limit: Some(limit.fetch()),
                execution_config,
                metrics: ExecutionPlanMetricsSet::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        if let Some(sort) = plan.as_any().downcast_ref::<SortExec>()
            && let Some(aggregate) = sort.input().as_any().downcast_ref::<AggregateExec>()
            && let Some(projection) = aggregate.input().as_any().downcast_ref::<ProjectionExec>()
            && let Some(filter) = projection.input().as_any().downcast_ref::<FilterExec>()
        {
            return Ok(Self {
                input: Arc::clone(filter.input()),
                filter: Some(Arc::clone(filter.predicate())),
                projection: projection.expr().to_vec(),
                aggregate: Some(lower_aggregate_exec(aggregate)?),
                sort: sort.expr().to_vec(),
                local_limit: sort.fetch(),
                execution_config,
                metrics: ExecutionPlanMetricsSet::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        if let Some(sort) = plan.as_any().downcast_ref::<SortExec>()
            && let Some(projection) = sort.input().as_any().downcast_ref::<ProjectionExec>()
            && let Some(filter) = projection.input().as_any().downcast_ref::<FilterExec>()
        {
            return Ok(Self {
                input: Arc::clone(filter.input()),
                filter: Some(Arc::clone(filter.predicate())),
                projection: projection.expr().to_vec(),
                aggregate: None,
                sort: sort.expr().to_vec(),
                local_limit: sort.fetch(),
                execution_config,
                metrics: ExecutionPlanMetricsSet::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>()
            && let Some(projection) = aggregate.input().as_any().downcast_ref::<ProjectionExec>()
            && let Some(filter) = projection.input().as_any().downcast_ref::<FilterExec>()
        {
            return Ok(Self {
                input: Arc::clone(filter.input()),
                filter: Some(Arc::clone(filter.predicate())),
                projection: projection.expr().to_vec(),
                aggregate: Some(lower_aggregate_exec(aggregate)?),
                sort: Vec::new(),
                local_limit: None,
                execution_config,
                metrics: ExecutionPlanMetricsSet::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
            if let Some(filter) = projection.input().as_any().downcast_ref::<FilterExec>() {
                return Ok(Self {
                    input: Arc::clone(filter.input()),
                    filter: Some(Arc::clone(filter.predicate())),
                    projection: projection.expr().to_vec(),
                    aggregate: None,
                    sort: Vec::new(),
                    local_limit: None,
                    execution_config,
                    metrics: ExecutionPlanMetricsSet::new(),
                    cache: Arc::clone(plan.properties()),
                });
            }
            return Ok(Self {
                input: Arc::clone(projection.input()),
                filter: None,
                projection: projection.expr().to_vec(),
                aggregate: None,
                sort: Vec::new(),
                local_limit: None,
                execution_config,
                metrics: ExecutionPlanMetricsSet::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
            return Ok(Self {
                input: Arc::clone(filter.input()),
                filter: Some(Arc::clone(filter.predicate())),
                projection: Vec::new(),
                aggregate: None,
                sort: Vec::new(),
                local_limit: None,
                execution_config,
                metrics: ExecutionPlanMetricsSet::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_cudf.gpu_island.try_new",
            "cannot build GPU island from physical plan",
        )
        .with_fact("plan_node", plan.name()))
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
}

impl DisplayAs for CudfGpuIslandExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let operators = self.operator_list();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "CudfGpuIslandExec: operators=[{}], projection_exprs={}", operators, self.projection.len(),)?;
                if let Some(aggregate) = &self.aggregate {
                    write!(f, ", group_keys={}, aggregate_exprs={}", aggregate.keys.len(), aggregate.aggs.len(),)?;
                }
                if !self.sort.is_empty() {
                    write!(f, ", sort_exprs={}", self.sort.len())?;
                }
                if let Some(fetch) = self.local_limit {
                    write!(f, ", local_limit={fetch}")?;
                }
                write!(f, ", input_boundary=host_to_device, output_boundary=device_to_host")
            },
            DisplayFormatType::TreeRender => {
                writeln!(f, "operators={operators}")?;
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
                writeln!(f, "input_boundary=host_to_device")?;
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
        vec![self.sort.is_empty()]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "datafusion_cudf.gpu_island.with_new_children",
                "GPU island expected exactly one child",
            )
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(Arc::new(Self {
            input: children.swap_remove(0),
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

        if aggregate.is_some() || !sort.is_empty() {
            let input = Arc::clone(&self.input);
            let child_partitions = if self.cache.output_partitioning().partition_count() == 1 {
                (0..self.input.output_partitioning().partition_count()).collect::<Vec<_>>()
            } else {
                vec![partition]
            };
            let ops = GpuIslandOps { filter, projection, aggregate, sort, local_limit };
            let output = futures::stream::once(async move {
                let output =
                    execute_blocking_sort(input, child_partitions, context, runtime.as_ref(), &ops, &metrics).await?;
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

        let input = self.input.execute(partition, context)?;
        let output = futures::stream::try_unfold(
            (input, local_limit, metrics),
            move |(mut input, mut remaining_rows, metrics)| {
                let runtime = Arc::clone(&runtime);
                let filter = filter.clone();
                let projection = projection.clone();
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
                    let output =
                        execute_batch(&runtime, &filter, &projection, remaining_rows.as_mut(), batch, &metrics)?
                            .record_output(&metrics.baseline);
                    Ok(Some((output, (input, remaining_rows, metrics))))
                }
            },
        );

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

struct GpuIslandOps {
    filter: Option<Arc<dyn PhysicalExpr>>,
    projection: Vec<ProjectionExpr>,
    aggregate: Option<GpuAggregate>,
    sort: Vec<PhysicalSortExpr>,
    local_limit: Option<usize>,
}

struct FrameOps<'a> {
    filter: &'a Option<Arc<dyn PhysicalExpr>>,
    projection: &'a [ProjectionExpr],
    aggregate: &'a Option<GpuAggregate>,
    sort: &'a [PhysicalSortExpr],
}

impl<'a> FrameOps<'a> {
    fn from_gpu_island_ops(ops: &'a GpuIslandOps) -> Self {
        Self { filter: &ops.filter, projection: &ops.projection, aggregate: &ops.aggregate, sort: &ops.sort }
    }
}

async fn execute_blocking_sort(
    input: Arc<dyn ExecutionPlan>,
    child_partitions: Vec<usize>,
    context: Arc<TaskContext>,
    runtime: &Runtime,
    ops: &GpuIslandOps,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<Option<RecordBatch>> {
    let ctx =
        cudf::ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let mut frames = Vec::new();

    for child_partition in child_partitions {
        let mut stream = input.execute(child_partition, Arc::clone(&context))?;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            frames.push(upload_batch(runtime, batch, metrics)?);
        }
    }

    let mut frames = frames.into_iter();
    let Some(mut frame) = frames.next() else {
        return Ok(None);
    };
    let other_frames = frames.collect::<Vec<_>>();
    if !other_frames.is_empty() {
        let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
        let operator_timer = metrics.cudf_operator_time.timer();
        frame = frame.with_execution(&ctx).concat_many(&other_frame_refs).map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.input.concat",
                "failed to concatenate uploaded RecordBatches inside GPU island",
                error,
            )
        })?;
        operator_timer.done();
    }

    let mut remaining_rows = ops.local_limit;
    frame = execute_frame_ops(&ctx, FrameOps::from_gpu_island_ops(ops), remaining_rows.as_mut(), frame, metrics)?;
    materialize_frame(frame, metrics).map(Some)
}

fn execute_batch(
    runtime: &Runtime,
    filter: &Option<Arc<dyn PhysicalExpr>>,
    projection: &[ProjectionExpr],
    remaining_rows: Option<&mut usize>,
    batch: RecordBatch,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<RecordBatch> {
    let input_rows = batch.num_rows();
    let input_bytes = get_record_batch_memory_size(&batch);
    metrics.input_batches.add(1);
    metrics.input_rows.add(input_rows);

    let ctx =
        cudf::ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let frame = upload_batch_with_size(runtime, batch, input_rows, input_bytes, metrics)?;
    let no_aggregate = None;
    let frame_ops = FrameOps { filter, projection, aggregate: &no_aggregate, sort: &[] };
    let frame = execute_frame_ops(&ctx, frame_ops, remaining_rows, frame, metrics)?;
    materialize_frame(frame, metrics)
}

fn upload_batch(
    runtime: &Runtime,
    batch: RecordBatch,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<cudf::DataFrame> {
    let input_rows = batch.num_rows();
    let input_bytes = get_record_batch_memory_size(&batch);
    metrics.input_batches.add(1);
    metrics.input_rows.add(input_rows);
    upload_batch_with_size(runtime, batch, input_rows, input_bytes, metrics)
}

fn upload_batch_with_size(
    runtime: &Runtime,
    batch: RecordBatch,
    input_rows: usize,
    input_bytes: usize,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<cudf::DataFrame> {
    let upload_timer = metrics.record_batch_upload_time.timer();
    let frame = cudf::DataFrame::from_record_batch(runtime, &batch).map_err(|error| {
        Error::from_cudf("datafusion_cudf.record_batch.upload", "failed to upload RecordBatch to cudf", error)
    })?;
    upload_timer.done();
    metrics.host_to_device_uploads.add(1);
    metrics.host_to_device_uploaded_rows.add(input_rows);
    metrics.host_to_device_uploaded_bytes.add(input_bytes);
    Ok(frame)
}

fn execute_frame_ops(
    ctx: &cudf::ExecutionOverrides<'_>,
    ops: FrameOps<'_>,
    remaining_rows: Option<&mut usize>,
    mut frame: cudf::DataFrame,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<cudf::DataFrame> {
    if let Some(filter) = ops.filter {
        let predicate = lower_expr(filter)?;
        let operator_timer = metrics.cudf_operator_time.timer();
        frame = frame.with_execution(ctx).filter_expr(predicate).map_err(|error| {
            Error::from_cudf("datafusion_cudf.filter.execute", "failed to execute cudf filter expression", error)
        })?;
        operator_timer.done();
    }

    if !ops.projection.is_empty() {
        let mut exprs = Vec::with_capacity(ops.projection.len());
        let mut output_names = Vec::with_capacity(ops.projection.len());
        for projection_expr in ops.projection {
            let expr = lower_expr(&projection_expr.expr)?;
            exprs.push((projection_expr.alias.as_str(), expr));
            output_names.push(projection_expr.alias.as_str());
        }
        let operator_timer = metrics.cudf_operator_time.timer();
        frame = frame.with_execution(ctx).with_columns(exprs).map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.projection.execute",
                "failed to execute cudf projection expressions",
                error,
            )
        })?;
        frame = frame.select(output_names).map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.projection.select",
                "failed to select cudf projection output columns",
                error,
            )
        })?;
        operator_timer.done();
    }

    if let Some(aggregate) = ops.aggregate {
        let keys = aggregate.keys.iter().map(|key| key.input_column.as_str()).collect::<Vec<_>>();
        let specs = aggregate
            .aggs
            .iter()
            .map(|agg| AggSpec::new(agg.input_column.as_str(), agg.agg.clone()).with_alias(agg.output_name.as_str()))
            .collect::<Vec<_>>();
        let operator_timer = metrics.cudf_operator_time.timer();
        frame = frame.groupby(keys).with_execution(ctx).agg(&specs).map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.aggregate.execute",
                "failed to execute cudf groupby aggregate inside GPU island",
                error,
            )
        })?;
        let key_renames = aggregate
            .keys
            .iter()
            .filter(|key| key.input_column != key.output_name)
            .map(|key| (key.input_column.as_str(), key.output_name.as_str()))
            .collect::<Vec<_>>();
        if !key_renames.is_empty() {
            frame = frame.rename_columns(key_renames).map_err(|error| {
                Error::from_cudf(
                    "datafusion_cudf.aggregate.rename_keys",
                    "failed to rename cudf groupby key columns to DataFusion aggregate aliases",
                    error,
                )
            })?;
        }
        operator_timer.done();
    }

    if !ops.sort.is_empty() {
        let sort_plan = lower_sort_exprs(ops.sort)?;
        let sort_options =
            CudfSortOptions::new().with_descending(sort_plan.descending).with_nulls_first(sort_plan.nulls_first);
        let operator_timer = metrics.cudf_operator_time.timer();
        frame = frame.with_execution(ctx).sort_values(sort_plan.columns, sort_options).map_err(|error| {
            Error::from_cudf("datafusion_cudf.sort.execute", "failed to execute cudf sort inside GPU island", error)
        })?;
        operator_timer.done();
    }

    if let Some(remaining_rows) = remaining_rows {
        let output_rows = frame.num_rows();
        let rows_to_take = output_rows.min(*remaining_rows);
        if rows_to_take < output_rows {
            let operator_timer = metrics.cudf_operator_time.timer();
            frame = frame.with_execution(ctx).slice(0, rows_to_take).map_err(|error| {
                Error::from_cudf(
                    "datafusion_cudf.limit.execute",
                    "failed to execute cudf local limit inside GPU island",
                    error,
                )
            })?;
            operator_timer.done();
        }
        *remaining_rows -= rows_to_take;
    }

    Ok(frame)
}

fn materialize_frame(frame: cudf::DataFrame, metrics: &CudfGpuIslandMetrics) -> DataFusionResult<RecordBatch> {
    let materialization_timer = metrics.record_batch_materialization_time.timer();
    let output = frame.to_record_batch().map_err(|error| {
        Error::from_cudf(
            "datafusion_cudf.record_batch.materialize",
            "failed to materialize cudf DataFrame to RecordBatch",
            error,
        )
    })?;
    materialization_timer.done();
    metrics.device_to_host_materializations.add(1);
    metrics.device_to_host_materialized_rows.add(output.num_rows());
    metrics.device_to_host_materialized_bytes.add(get_record_batch_memory_size(&output));
    Ok(output)
}

#[derive(Debug)]
struct LoweredSortExprs {
    columns: Vec<String>,
    descending: Vec<bool>,
    nulls_first: Vec<bool>,
}

#[derive(Debug, Clone)]
struct GpuAggregate {
    keys: Vec<GpuGroupKey>,
    aggs: Vec<GpuAggregateExpr>,
}

#[derive(Debug, Clone)]
struct GpuGroupKey {
    input_column: String,
    output_name: String,
}

#[derive(Debug, Clone)]
struct GpuAggregateExpr {
    input_column: String,
    agg: GroupByAgg,
    output_name: String,
}

fn lower_aggregate_exec(aggregate: &AggregateExec) -> AdapterResult<GpuAggregate> {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned) {
        return Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.aggregate.lower",
            "cannot lower DataFusion aggregate mode that consumes or produces partial aggregate state",
        )
        .with_fact("aggregate_mode", format!("{:?}", aggregate.mode())));
    }

    let group_by = aggregate.group_expr();
    if !group_by.is_single() || group_by.expr().is_empty() {
        return Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.aggregate.lower",
            "cannot lower global aggregates or grouping sets to cudf groupby",
        ));
    }
    if group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null) {
        return Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.aggregate.lower",
            "cannot lower grouping-set null masks to cudf groupby",
        ));
    }
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower",
            "cannot lower aggregate FILTER clauses to cudf groupby",
        ));
    }

    let keys = group_by
        .expr()
        .iter()
        .map(|(expr, alias)| {
            let Some(column) = expr.as_any().downcast_ref::<Column>() else {
                return Err(Error::new(
                    ErrorCode::UnsupportedExpression,
                    "datafusion_cudf.aggregate.lower_group_key",
                    "cannot lower non-column group key expression",
                )
                .with_fact("expression", format!("{expr:?}")));
            };
            Ok(GpuGroupKey { input_column: column.name().to_string(), output_name: alias.clone() })
        })
        .collect::<AdapterResult<Vec<_>>>()?;

    let aggs =
        aggregate.aggr_expr().iter().map(|expr| lower_aggregate_expr(expr)).collect::<AdapterResult<Vec<_>>>()?;

    Ok(GpuAggregate { keys, aggs })
}

fn lower_aggregate_expr(
    expr: &datafusion::physical_expr::aggregate::AggregateFunctionExpr,
) -> AdapterResult<GpuAggregateExpr> {
    if expr.is_distinct() {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower_expr",
            "cannot lower DISTINCT aggregate expressions to cudf groupby",
        )
        .with_fact("aggregate", expr.name()));
    }
    if !expr.order_bys().is_empty() {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower_expr",
            "cannot lower order-sensitive aggregate expressions to cudf groupby",
        )
        .with_fact("aggregate", expr.name()));
    }

    let expressions = expr.expressions();
    let [arg] = expressions.as_slice() else {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower_expr",
            "cannot lower aggregate expression with multiple arguments",
        )
        .with_fact("aggregate", expr.name())
        .with_fact("argument_count", expressions.len()));
    };
    let Some(column) = arg.as_any().downcast_ref::<Column>() else {
        return Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.aggregate.lower_expr",
            "cannot lower aggregate expression over non-column argument",
        )
        .with_fact("aggregate", expr.name())
        .with_fact("argument", format!("{arg:?}")));
    };

    let agg = match expr.fun().name().to_ascii_lowercase().as_str() {
        "sum" => GroupByAgg::Sum,
        "min" => GroupByAgg::Min,
        "max" => GroupByAgg::Max,
        "count" => GroupByAgg::Count { include_nulls: false },
        "avg" => GroupByAgg::Mean,
        _ => {
            return Err(Error::new(
                ErrorCode::UnsupportedOperator,
                "datafusion_cudf.aggregate.lower_expr",
                "cannot lower unsupported aggregate function to cudf groupby",
            )
            .with_fact("aggregate_function", expr.fun().name())
            .with_fact("aggregate", expr.name()));
        },
    };

    Ok(GpuAggregateExpr { input_column: column.name().to_string(), agg, output_name: expr.name().to_string() })
}

fn lower_sort_exprs(sort: &[PhysicalSortExpr]) -> AdapterResult<LoweredSortExprs> {
    let mut columns = Vec::with_capacity(sort.len());
    let mut descending = Vec::with_capacity(sort.len());
    let mut nulls_first = Vec::with_capacity(sort.len());
    for sort_expr in sort {
        let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
            return Err(Error::new(
                ErrorCode::UnsupportedExpression,
                "datafusion_cudf.sort.lower",
                "cannot lower unsupported sort expression",
            )
            .with_fact("expression", format!("{:?}", sort_expr.expr)));
        };
        columns.push(column.name().to_string());
        descending.push(sort_expr.options.descending);
        nulls_first.push(sort_expr.options.nulls_first);
    }
    Ok(LoweredSortExprs { columns, descending, nulls_first })
}

#[derive(Debug, Clone)]
struct CudfGpuIslandMetrics {
    baseline: BaselineMetrics,
    input_batches: Count,
    input_rows: Count,
    host_to_device_uploads: Count,
    host_to_device_uploaded_rows: Count,
    host_to_device_uploaded_bytes: Count,
    device_to_host_materializations: Count,
    device_to_host_materialized_rows: Count,
    device_to_host_materialized_bytes: Count,
    record_batch_upload_time: Time,
    cudf_operator_time: Time,
    record_batch_materialization_time: Time,
}

impl CudfGpuIslandMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            input_batches: summary_counter(metrics, "input_batches", partition),
            input_rows: summary_counter(metrics, "input_rows", partition),
            host_to_device_uploads: summary_counter(metrics, "host_to_device_uploads", partition),
            host_to_device_uploaded_rows: summary_counter(metrics, "host_to_device_uploaded_rows", partition),
            host_to_device_uploaded_bytes: summary_counter(metrics, "host_to_device_uploaded_bytes", partition),
            device_to_host_materializations: summary_counter(metrics, "device_to_host_materializations", partition),
            device_to_host_materialized_rows: summary_counter(metrics, "device_to_host_materialized_rows", partition),
            device_to_host_materialized_bytes: summary_counter(metrics, "device_to_host_materialized_bytes", partition),
            record_batch_upload_time: summary_timer(metrics, "record_batch_upload_time", partition),
            cudf_operator_time: summary_timer(metrics, "cudf_operator_time", partition),
            record_batch_materialization_time: summary_timer(metrics, "record_batch_materialization_time", partition),
        }
    }
}

fn summary_counter(metrics: &ExecutionPlanMetricsSet, name: &'static str, partition: usize) -> Count {
    MetricBuilder::new(metrics).with_type(MetricType::SUMMARY).counter(name, partition)
}

fn summary_timer(metrics: &ExecutionPlanMetricsSet, name: &'static str, partition: usize) -> Time {
    MetricBuilder::new(metrics).with_type(MetricType::SUMMARY).subset_time(name, partition)
}

fn lower_expr(expr: &Arc<dyn PhysicalExpr>) -> AdapterResult<Expr> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        return Ok(col(column.name()));
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = lower_expr(binary.left())?;
        let right = lower_expr(binary.right())?;
        return lower_binary(left, binary.op(), right);
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(lower_expr(not.arg())?.not());
    }
    Err(Error::new(
        ErrorCode::UnsupportedExpression,
        "datafusion_cudf.expr.lower",
        "cannot lower unsupported physical expression",
    )
    .with_fact("expression", expr))
}

fn lower_literal(value: &ScalarValue) -> AdapterResult<Expr> {
    match value {
        ScalarValue::Boolean(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int8(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int16(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int32(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int64(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt8(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt16(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt32(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt64(Some(value)) => Ok(lit(*value)),
        ScalarValue::Float32(Some(value)) => Ok(lit(*value)),
        ScalarValue::Float64(Some(value)) => Ok(lit(*value)),
        _ => Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.expr.lower_literal",
            "cannot lower unsupported or null literal",
        )
        .with_fact("literal", format!("{value:?}"))),
    }
}

fn lower_binary(left: Expr, op: &Operator, right: Expr) -> AdapterResult<Expr> {
    match op {
        Operator::Plus => Ok(left + right),
        Operator::Minus => Ok(left - right),
        Operator::Multiply => Ok(left * right),
        Operator::Divide => Ok(left / right),
        Operator::Modulo => Ok(left % right),
        Operator::Eq => Ok(left.eq(right)),
        Operator::NotEq => Ok(left.ne(right)),
        Operator::Lt => Ok(left.lt(right)),
        Operator::LtEq => Ok(left.le(right)),
        Operator::Gt => Ok(left.gt(right)),
        Operator::GtEq => Ok(left.ge(right)),
        Operator::And => Ok(left & right),
        Operator::Or => Ok(left | right),
        _ => Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.expr.lower_binary",
            "cannot lower unsupported binary operator",
        )
        .with_fact("operator", format!("{op:?}"))),
    }
}
