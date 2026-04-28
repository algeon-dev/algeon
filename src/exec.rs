use std::{any::Any, collections::BTreeSet, path::PathBuf, sync::Arc};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use cudf::{
    AggSpec, DataFrame, ExecutionOverrides, ExecutionPlacement, Expr, GroupByAgg, Runtime,
    SortOptions as CudfSortOptions,
    frame::dataframe::ParquetReadOptions,
    ops::{col, lit},
};
use datafusion::{
    common::{Result as DataFusionResult, ScalarValue, utils::memory::get_record_batch_memory_size},
    datasource::{
        physical_plan::{FileScanConfig, FileSource, ParquetSource},
        source::DataSourceExec,
    },
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

#[derive(Debug, Clone)]
enum GpuIslandInput {
    ChildRecordBatches(Arc<dyn ExecutionPlan>),
    LocalParquetScan(CudfParquetScan),
}

impl GpuIslandInput {
    fn child(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        match self {
            Self::ChildRecordBatches(input) => Some(input),
            Self::LocalParquetScan(_) => None,
        }
    }

    fn output_partition_count(&self) -> usize {
        match self {
            Self::ChildRecordBatches(input) => input.output_partitioning().partition_count(),
            Self::LocalParquetScan(scan) => scan.partitions.len(),
        }
    }

    fn display_name(&self) -> &'static str {
        match self {
            Self::ChildRecordBatches(_) => "child_record_batches",
            Self::LocalParquetScan(_) => "local_parquet_scan",
        }
    }

    fn input_boundary(&self) -> &'static str {
        match self {
            Self::ChildRecordBatches(_) => "host_to_device",
            Self::LocalParquetScan(_) => "cudf_parquet_reader",
        }
    }
}

#[derive(Debug, Clone)]
struct CudfParquetScan {
    partitions: Vec<CudfParquetPartition>,
    read_columns: Vec<String>,
    file_schema: SchemaRef,
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

#[derive(Debug, Clone)]
struct NormalizedGpuIslandPlan {
    input: Arc<dyn ExecutionPlan>,
    filter: Option<Arc<dyn PhysicalExpr>>,
    projection: Vec<ProjectionExpr>,
    aggregate: Option<GpuAggregate>,
    sort: Vec<PhysicalSortExpr>,
    local_limit: Option<usize>,
}

impl CudfGpuIslandExec {
    pub fn try_new_from_plan(plan: Arc<dyn ExecutionPlan>) -> AdapterResult<Self> {
        Self::try_new_from_plan_with_execution_config(plan, CudfExecutionConfig::default())
    }

    pub fn try_new_from_plan_with_execution_config(
        plan: Arc<dyn ExecutionPlan>,
        execution_config: CudfExecutionConfig,
    ) -> AdapterResult<Self> {
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
}

impl GpuIslandInput {
    fn try_from_normalized(plan: &NormalizedGpuIslandPlan) -> AdapterResult<Self> {
        if let Ok(Some(scan)) = try_local_parquet_scan(plan) {
            return Ok(Self::LocalParquetScan(scan));
        }
        Ok(Self::ChildRecordBatches(Arc::clone(&plan.input)))
    }
}

fn normalize_gpu_island_plan(plan: Arc<dyn ExecutionPlan>) -> AdapterResult<Option<NormalizedGpuIslandPlan>> {
    let mut input = Arc::clone(&plan);
    let mut filter = None;
    let mut projection = Vec::new();
    let mut aggregate = None;
    let mut sort = Vec::new();
    let mut local_limit = None;
    let mut operator_count = 0usize;

    if let Some(limit) = input.as_any().downcast_ref::<LocalLimitExec>() {
        local_limit = Some(limit.fetch());
        input = Arc::clone(limit.input());
        operator_count += 1;
    }

    if let Some(sort_exec) = input.as_any().downcast_ref::<SortExec>() {
        sort = sort_exec.expr().to_vec();
        if local_limit.is_none() {
            local_limit = sort_exec.fetch();
        }
        input = Arc::clone(sort_exec.input());
        operator_count += 1;
    }

    if let Some(aggregate_exec) = input.as_any().downcast_ref::<AggregateExec>() {
        aggregate = Some(lower_aggregate_exec(aggregate_exec)?);
        input = Arc::clone(aggregate_exec.input());
        operator_count += 1;
    }

    if let Some(projection_exec) = input.as_any().downcast_ref::<ProjectionExec>() {
        projection = projection_exec.expr().to_vec();
        input = Arc::clone(projection_exec.input());
        operator_count += 1;
    }

    if let Some(filter_exec) = input.as_any().downcast_ref::<FilterExec>() {
        filter = Some(Arc::clone(filter_exec.predicate()));
        input = Arc::clone(filter_exec.input());
        operator_count += 1;
    }

    if operator_count == 0 && input.as_any().downcast_ref::<DataSourceExec>().is_none() {
        return Ok(None);
    }

    Ok(Some(NormalizedGpuIslandPlan { input, filter, projection, aggregate, sort, local_limit }))
}

fn try_local_parquet_scan(plan: &NormalizedGpuIslandPlan) -> AdapterResult<Option<CudfParquetScan>> {
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
    } else if plan.projection.is_empty() {
        required.extend(file_schema.fields().iter().map(|field| field.name().clone()));
    }

    if let Some(filter) = &plan.filter {
        collect_required_columns(filter, &mut required)?;
    }
    for projection in &plan.projection {
        collect_required_columns(&projection.expr, &mut required)?;
    }
    if let Some(aggregate) = &plan.aggregate {
        required.extend(aggregate.keys.iter().map(|key| key.input_column.clone()));
        required.extend(aggregate.aggs.iter().map(|agg| agg.input_column.clone()));
    }
    for sort_expr in &plan.sort {
        collect_required_columns(&sort_expr.expr, &mut required)?;
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

#[derive(Debug)]
struct CudfBatch {
    frame: DataFrame,
    placement: ExecutionPlacement,
}

impl CudfBatch {
    fn new(frame: DataFrame) -> Self {
        let placement = frame.placement();
        Self { frame, placement }
    }

    fn into_frame(self) -> DataFrame {
        debug_assert_eq!(self.placement, self.frame.placement());
        self.frame
    }

    fn concat_many(
        self,
        ctx: &ExecutionOverrides<'_>,
        other_batches: &[Self],
        metrics: &CudfGpuIslandMetrics,
    ) -> DataFusionResult<Self> {
        if other_batches.is_empty() {
            return Ok(self);
        }

        let other_frames = other_batches.iter().map(|batch| &batch.frame).collect::<Vec<_>>();
        let operator_timer = metrics.cudf_operator_time.timer();
        let frame = self.frame.with_execution(ctx).concat_many(&other_frames).map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.input.concat",
                "failed to concatenate uploaded CudfBatches inside GPU island",
                error,
            )
        })?;
        operator_timer.done();
        Ok(Self::new(frame))
    }

    fn materialize(self, metrics: &CudfGpuIslandMetrics) -> DataFusionResult<RecordBatch> {
        debug_assert_eq!(self.placement, self.frame.placement());
        let materialization_timer = metrics.record_batch_materialization_time.timer();
        let output = self.frame.to_record_batch().map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.record_batch.materialize",
                "failed to materialize CudfBatch to RecordBatch",
                error,
            )
        })?;
        materialization_timer.done();
        metrics.device_to_host_materializations.add(1);
        metrics.device_to_host_materialized_rows.add(output.num_rows());
        metrics.device_to_host_materialized_bytes.add(get_record_batch_memory_size(&output));
        Ok(output)
    }
}

#[derive(Clone)]
struct CudfOperatorChain {
    filter: Option<Arc<dyn PhysicalExpr>>,
    projection: Vec<ProjectionExpr>,
    aggregate: Option<GpuAggregate>,
    sort: Vec<PhysicalSortExpr>,
    local_limit: Option<usize>,
}

impl CudfOperatorChain {
    fn new(
        filter: Option<Arc<dyn PhysicalExpr>>,
        projection: Vec<ProjectionExpr>,
        aggregate: Option<GpuAggregate>,
        sort: Vec<PhysicalSortExpr>,
        local_limit: Option<usize>,
    ) -> Self {
        Self { filter, projection, aggregate, sort, local_limit }
    }

    fn requires_single_output_batch(&self) -> bool {
        self.aggregate.is_some() || !self.sort.is_empty()
    }

    fn execute(
        &self,
        ctx: &ExecutionOverrides<'_>,
        remaining_rows: Option<&mut usize>,
        batch: CudfBatch,
        metrics: &CudfGpuIslandMetrics,
    ) -> DataFusionResult<CudfBatch> {
        let mut frame = batch.into_frame();

        if let Some(filter) = &self.filter {
            let predicate = lower_expr(filter)?;
            let operator_timer = metrics.cudf_operator_time.timer();
            frame = frame.with_execution(ctx).filter_expr(predicate).map_err(|error| {
                Error::from_cudf("datafusion_cudf.filter.execute", "failed to execute cudf filter expression", error)
            })?;
            operator_timer.done();
        }

        if !self.projection.is_empty() {
            let mut exprs = Vec::with_capacity(self.projection.len());
            let mut output_names = Vec::with_capacity(self.projection.len());
            for projection_expr in &self.projection {
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

        if let Some(aggregate) = &self.aggregate {
            let keys = aggregate.keys.iter().map(|key| key.input_column.as_str()).collect::<Vec<_>>();
            let specs = aggregate
                .aggs
                .iter()
                .map(|agg| {
                    AggSpec::new(agg.input_column.as_str(), agg.agg.clone()).with_alias(agg.output_name.as_str())
                })
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

        if !self.sort.is_empty() {
            let sort_plan = lower_sort_exprs(&self.sort)?;
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

        Ok(CudfBatch::new(frame))
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
                if let Some(batch) = read_local_parquet_partition(runtime, &ctx, &scan, partition, metrics)? {
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
    let Some(batch) = read_local_parquet_partition(runtime, &ctx, &scan, partition, metrics)? else {
        return Ok(None);
    };
    let mut remaining_rows = chain.local_limit;
    let batch = chain.execute(&ctx, remaining_rows.as_mut(), batch, metrics)?;
    batch.materialize(metrics).map(Some)
}

fn read_local_parquet_partition(
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
    let frame = DataFrame::read_parquet_with_options(options).map_err(|error| {
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

fn upload_batch(runtime: &Runtime, batch: RecordBatch, metrics: &CudfGpuIslandMetrics) -> DataFusionResult<CudfBatch> {
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
) -> DataFusionResult<CudfBatch> {
    let upload_timer = metrics.record_batch_upload_time.timer();
    let frame = DataFrame::from_record_batch(runtime, &batch).map_err(|error| {
        Error::from_cudf("datafusion_cudf.record_batch.upload", "failed to upload RecordBatch to cudf", error)
    })?;
    upload_timer.done();
    metrics.host_to_device_uploads.add(1);
    metrics.host_to_device_uploaded_rows.add(input_rows);
    metrics.host_to_device_uploaded_bytes.add(input_bytes);
    Ok(CudfBatch::new(frame))
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
    parquet_files: Count,
    parquet_bytes_known: Count,
    row_groups_selected: Count,
    read_columns: Count,
    parquet_read_time: Time,
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
            parquet_files: summary_counter(metrics, "parquet_files", partition),
            parquet_bytes_known: summary_counter(metrics, "parquet_bytes_known", partition),
            row_groups_selected: summary_counter(metrics, "row_groups_selected", partition),
            read_columns: summary_counter(metrics, "read_columns", partition),
            parquet_read_time: summary_timer(metrics, "parquet_read_time", partition),
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
