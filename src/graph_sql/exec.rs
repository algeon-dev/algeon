use std::{
    any::Any,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use arrow::{
    datatypes::{Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf_query_engine::{
    ExecutionContext, GpuDataFrame, GpuResidentInputMap, GpuResidentSource, GpuResultHandle, GraphAlgorithm,
    GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions, GraphConstructionPolicy,
    GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeColumnBindings,
    GraphEigenvectorCentralityParams, GraphHitsParams, GraphInDegreesAllParams, GraphKCoreDegreeType, GraphKCoreParams,
    GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams, GraphMstParams, GraphOutDegreesAllParams,
    GraphPageRankParams, GraphSccParams, GraphSpectralClusteringParams, GraphSsspParams, GraphTriangleCountAllParams,
    GraphWccParams, LimitSpec, ProjectionOutput, QueryMetrics, QueryOutput, QueryPlanBuilder, SourceNode,
    execute_query_plan, execute_query_plan_with_gpu_inputs,
};
use datafusion::{
    common::{DataFusionError, Result},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning, expressions::Column},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        aggregates::AggregateExec,
        collect,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet, RecordOutput},
        projection::ProjectionExec,
        stream::RecordBatchStreamAdapter,
        union::UnionExec,
    },
};
use futures::{TryStreamExt, future::BoxFuture};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphGraphConstructionPolicy,
    CugraphKCoreDegreeType, plan_error,
};
use crate::{
    error::{Error, ErrorCode},
    native_exec::{
        GpuResidentProducerExec, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfMetrics,
        gpu_resident_producer as native_gpu_resident_producer,
    },
};

static NEXT_CUGRAPH_GPU_RESULT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct CugraphAlgoExec {
    spec: CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    native_execution_config: NativeCudfExecutionConfig,
    edge_source_canonical_facts: CugraphEdgeCanonicalFacts,
    gpu_result_input_name: String,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl CugraphAlgoExec {
    pub(super) fn try_new(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: NativeCudfExecutionConfig,
    ) -> Result<Self> {
        let id = NEXT_CUGRAPH_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        Self::try_new_with_gpu_result_input_name(
            spec,
            edge_source,
            projection,
            limit,
            native_execution_config,
            format!("__datafusion_cudf_cugraph_result_{id}"),
        )
    }

    fn try_new_with_gpu_result_input_name(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: NativeCudfExecutionConfig,
        gpu_result_input_name: String,
    ) -> Result<Self> {
        let schema = projected_schema(&spec.output_schema, projection.as_ref())?;
        let edge_source_canonical_facts = cugraph_edge_canonical_facts(edge_source.as_ref(), &spec);
        Ok(Self {
            spec,
            edge_source,
            projection,
            limit,
            native_execution_config,
            edge_source_canonical_facts,
            gpu_result_input_name,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        })
    }

    pub(crate) fn edge_normalize_inserted(&self) -> bool {
        self.spec.graph_options.construction_policy == CugraphGraphConstructionPolicy::PythonCugraph
            && !self.edge_source_canonical_facts.satisfies_python_options(&self.spec)
    }

    async fn execute_gpu_result_with_context_impl(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> Result<(String, GpuResultHandle)> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.cugraph_sql.execute_gpu",
                "cuGraph SQL GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let edge_schema = self.edge_source.schema();
        let edge_frame =
            execute_edge_source_to_gpu(native_context, Arc::clone(&self.edge_source), task_context).await?;
        let metrics = NativeCudfMetrics::new(&self.metrics, partition);
        let (handle, query_metrics) =
            execute_cugraph_query_to_gpu(native_context, &self.spec, edge_schema, edge_frame, self)?;
        metrics.record_query_metrics(&query_metrics);
        Ok((self.gpu_result_input_name.clone(), handle))
    }
}

impl GpuResidentProducerExec for CugraphAlgoExec {
    fn gpu_result_input_name(&self) -> &str {
        &self.gpu_result_input_name
    }

    fn execute_gpu_result_with_context<'a>(
        &'a self,
        native_context: &'a ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, Result<(String, GpuResultHandle)>> {
        Box::pin(
            async move { self.execute_gpu_result_with_context_impl(native_context, partition, task_context).await },
        )
    }
}

pub(crate) fn gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducerExec> {
    plan.as_any().downcast_ref::<CugraphAlgoExec>().map(|producer| producer as &dyn GpuResidentProducerExec)
}

impl DisplayAs for CugraphAlgoExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "CugraphAlgoExec: algorithm={:?}, table={}, src_col={}, dst_col={}, weight_col={}, construction_policy={}, memory_control_policy={}, edge_normalize_inserted={}, output_boundary=device_resident_or_device_to_host, gpu_result_input={}",
                self.spec.algorithm,
                self.spec.table_name,
                self.spec.src_col,
                self.spec.dst_col,
                self.spec.weight_col.as_deref().unwrap_or("<none>"),
                self.spec.graph_options.construction_policy.as_str(),
                self.native_execution_config.memory_control_policy(),
                self.edge_normalize_inserted(),
                self.gpu_result_input_name,
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "algorithm={:?}", self.spec.algorithm)?;
                writeln!(f, "table={}", self.spec.table_name)?;
                writeln!(f, "src_col={}", self.spec.src_col)?;
                writeln!(f, "dst_col={}", self.spec.dst_col)?;
                writeln!(f, "weight_col={}", self.spec.weight_col.as_deref().unwrap_or("<none>"))?;
                writeln!(f, "construction_policy={}", self.spec.graph_options.construction_policy.as_str())?;
                writeln!(f, "memory_control_policy={}", self.native_execution_config.memory_control_policy())?;
                writeln!(f, "edge_normalize_inserted={}", self.edge_normalize_inserted())?;
                writeln!(f, "output_boundary=device_resident_or_device_to_host")?;
                writeln!(f, "gpu_result_input={}", self.gpu_result_input_name)
            },
        }
    }
}

impl ExecutionPlan for CugraphAlgoExec {
    fn name(&self) -> &str {
        "CugraphAlgoExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.edge_source]
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "datafusion_cudf.cugraph_sql.with_new_children",
                "cuGraph SQL execution expects exactly one edge-source child",
            )
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(Arc::new(Self::try_new_with_gpu_result_input_name(
            self.spec.clone(),
            Arc::clone(&children[0]),
            self.projection.clone(),
            self.limit,
            self.native_execution_config.clone(),
            self.gpu_result_input_name.clone(),
        )?))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.cugraph_sql.execute",
                "cuGraph SQL execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let schema = self.schema();
        let spec = self.spec.clone();
        let edge_source = Arc::clone(&self.edge_source);
        let projection = self.projection.clone();
        let limit = self.limit;
        let edge_normalize_inserted = self.edge_normalize_inserted();
        let default_execution_config = self.native_execution_config.clone();
        let metrics = NativeCudfMetrics::new(&self.metrics, partition);
        let output = futures::stream::once(async move {
            let execution_config = edge_execution_config(edge_source.as_ref(), &default_execution_config);
            let context_lease = execution_config.execution_context().map_err(|error| {
                DataFusionError::from(
                    error
                        .with_fact("cugraph_sql_phase", "runtime_context")
                        .with_fact("cpu_fallback_after_native_start", false),
                )
            })?;
            let context = context_lease.context();
            let edge_schema = edge_source.schema();
            let edge_frame = execute_edge_source_to_gpu(context, Arc::clone(&edge_source), task_context).await?;
            let (batch, query_metrics) = execute_cugraph_query_to_host(
                context,
                &spec,
                edge_schema,
                edge_frame,
                projection.as_ref(),
                limit,
                edge_normalize_inserted,
            )?;
            metrics.record_query_metrics(&query_metrics);
            Ok::<Option<RecordBatch>, DataFusionError>(Some(batch.record_output(&metrics.baseline)))
        })
        .try_filter_map(|batch| async move { Ok(batch) });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

fn projected_schema(schema: &SchemaRef, projection: Option<&Vec<usize>>) -> Result<SchemaRef> {
    let Some(projection) = projection else {
        return Ok(Arc::clone(schema));
    };
    let fields = projection
        .iter()
        .map(|index| {
            schema
                .fields()
                .get(*index)
                .map(|field| field.as_ref().clone())
                .ok_or_else(|| plan_error(format!("cuGraph output projection index {index} is out of range")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

fn edge_execution_config(
    plan: &dyn ExecutionPlan,
    default_execution_config: &NativeCudfExecutionConfig,
) -> NativeCudfExecutionConfig {
    if let Some(native) = as_native_cudf_exec(plan) {
        return native.execution_config().clone();
    }
    if let Some(native) = as_native_cudf_composed_exec(plan) {
        return native.execution_config().clone();
    }
    default_execution_config.clone()
}

fn as_native_cudf_exec(plan: &dyn ExecutionPlan) -> Option<&NativeCudfExec> {
    if let Some(native) = plan.as_any().downcast_ref::<NativeCudfExec>() {
        return Some(native);
    }
    let children = cooperative_exec_children(plan)?;
    let [child] = children.as_slice() else {
        return None;
    };
    child.as_any().downcast_ref::<NativeCudfExec>()
}

fn as_native_cudf_composed_exec(plan: &dyn ExecutionPlan) -> Option<&NativeCudfComposedExec> {
    if let Some(native) = plan.as_any().downcast_ref::<NativeCudfComposedExec>() {
        return Some(native);
    }
    let children = cooperative_exec_children(plan)?;
    let [child] = children.as_slice() else {
        return None;
    };
    child.as_any().downcast_ref::<NativeCudfComposedExec>()
}

fn cooperative_exec_children(plan: &dyn ExecutionPlan) -> Option<Vec<&Arc<dyn ExecutionPlan>>> {
    (plan.name() == "CooperativeExec").then(|| plan.children())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CugraphEdgeCanonicalFacts {
    add_reverse_present: bool,
    dedup_present: bool,
}

impl CugraphEdgeCanonicalFacts {
    fn satisfies_python_options(self, spec: &CugraphAlgorithmSpec) -> bool {
        self.dedup_present && spec.weight_col.is_none() && (spec.graph_options.directed || self.add_reverse_present)
    }
}

fn cugraph_edge_canonical_facts(plan: &dyn ExecutionPlan, spec: &CugraphAlgorithmSpec) -> CugraphEdgeCanonicalFacts {
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        let mut facts = cugraph_edge_canonical_facts(aggregate.input().as_ref(), spec);
        facts |= aggregate_edge_dedup_facts(aggregate, spec);
        return facts;
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        return cugraph_edge_canonical_facts(projection.input().as_ref(), spec);
    }

    if let Some(union) = plan.as_any().downcast_ref::<UnionExec>()
        && union_has_reverse_edge_expansion(union, spec)
    {
        return CugraphEdgeCanonicalFacts { add_reverse_present: true, ..Default::default() };
    }

    if preserves_edge_list_canonical_facts(plan) {
        let children = plan.children();
        if let [child] = children.as_slice() {
            return cugraph_edge_canonical_facts(child.as_ref(), spec);
        }
    }

    CugraphEdgeCanonicalFacts::default()
}

impl std::ops::BitOrAssign for CugraphEdgeCanonicalFacts {
    fn bitor_assign(&mut self, rhs: Self) {
        self.add_reverse_present |= rhs.add_reverse_present;
        self.dedup_present |= rhs.dedup_present;
    }
}

fn preserves_edge_list_canonical_facts(plan: &dyn ExecutionPlan) -> bool {
    matches!(plan.name(), "CoalesceBatchesExec" | "CoalescePartitionsExec" | "RepartitionExec" | "SortExec")
}

fn aggregate_edge_dedup_facts(aggregate: &AggregateExec, spec: &CugraphAlgorithmSpec) -> CugraphEdgeCanonicalFacts {
    let group_exprs = aggregate.group_expr().expr();
    let groups_source = group_exprs.iter().any(|(expr, alias)| {
        physical_column_name(expr.as_ref()).is_some_and(|name| same_edge_column(name, &spec.src_col))
            || same_edge_column(alias, &spec.src_col)
    });
    let groups_destination = group_exprs.iter().any(|(expr, alias)| {
        physical_column_name(expr.as_ref()).is_some_and(|name| same_edge_column(name, &spec.dst_col))
            || same_edge_column(alias, &spec.dst_col)
    });
    if !(groups_source && groups_destination) {
        return CugraphEdgeCanonicalFacts::default();
    }

    CugraphEdgeCanonicalFacts { dedup_present: true, add_reverse_present: false }
}

fn union_has_reverse_edge_expansion(union: &UnionExec, spec: &CugraphAlgorithmSpec) -> bool {
    let mut has_forward = false;
    let mut has_reverse = false;
    for input in union.inputs() {
        match edge_projection_orientation(input.as_ref(), spec) {
            Some(EdgeProjectionOrientation::Forward) => has_forward = true,
            Some(EdgeProjectionOrientation::Reverse) => has_reverse = true,
            None => {},
        }
    }
    has_forward && has_reverse
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeProjectionOrientation {
    Forward,
    Reverse,
}

fn edge_projection_orientation(
    plan: &dyn ExecutionPlan,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeProjectionOrientation> {
    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        return projection_edge_orientation(projection, spec);
    }

    let children = plan.children();
    if let [child] = children.as_slice()
        && let Some(orientation) = edge_projection_orientation(child.as_ref(), spec)
    {
        return Some(orientation);
    }

    let schema = plan.schema();
    if schema.index_of(&spec.src_col).is_ok() && schema.index_of(&spec.dst_col).is_ok() {
        Some(EdgeProjectionOrientation::Forward)
    } else {
        None
    }
}

fn projection_edge_orientation(
    projection: &ProjectionExec,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeProjectionOrientation> {
    let schema = projection.schema();
    let source_index = schema.index_of(&spec.src_col).ok()?;
    let destination_index = schema.index_of(&spec.dst_col).ok()?;
    let source_input = projection.expr().get(source_index).and_then(|expr| physical_column_name(expr.expr.as_ref()))?;
    let destination_input =
        projection.expr().get(destination_index).and_then(|expr| physical_column_name(expr.expr.as_ref()))?;
    if !projection_preserves_weight(projection, spec) {
        return None;
    }

    if same_edge_column(source_input, &spec.src_col) && same_edge_column(destination_input, &spec.dst_col) {
        Some(EdgeProjectionOrientation::Forward)
    } else if same_edge_column(source_input, &spec.dst_col) && same_edge_column(destination_input, &spec.src_col) {
        Some(EdgeProjectionOrientation::Reverse)
    } else {
        None
    }
}

fn projection_preserves_weight(projection: &ProjectionExec, spec: &CugraphAlgorithmSpec) -> bool {
    let Some(weight_col) = spec.weight_col.as_deref() else {
        return true;
    };
    let Ok(weight_index) = projection.schema().index_of(weight_col) else {
        return false;
    };
    projection
        .expr()
        .get(weight_index)
        .and_then(|expr| physical_column_name(expr.expr.as_ref()))
        .is_some_and(|name| same_edge_column(name, weight_col))
}

fn physical_column_name(expr: &dyn datafusion::physical_expr::PhysicalExpr) -> Option<&str> {
    expr.as_any().downcast_ref::<Column>().map(Column::name)
}

fn same_edge_column(actual: &str, expected: &str) -> bool {
    actual.eq_ignore_ascii_case(expected)
}

async fn execute_edge_source_to_gpu(
    context: &ExecutionContext,
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
) -> Result<GpuDataFrame> {
    if let Some(native) = as_native_cudf_exec(plan.as_ref()) {
        let native_plan = native.native_plan().clone_with_gpu_resident_sink();
        let result = execute_query_plan(context, &native_plan).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_cudf.cugraph_sql.edge_source_native",
                "native cuGraph edge-source execution failed",
                error,
            ))
        })?;
        let (output, _) = result.into_parts();
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.cugraph_sql.edge_source_native",
                "native cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok(handle.into_gpu_frame());
    }
    if let Some(native) = as_native_cudf_composed_exec(plan.as_ref()) {
        let native_plan = native.native_plan().clone_with_gpu_resident_sink();
        let mut gpu_inputs = GpuResidentInputMap::new();
        for child in native.gpu_inputs() {
            let Some(producer) = native_gpu_resident_producer(child.as_ref()) else {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_cudf.cugraph_sql.edge_source_composed",
                    "native composed cuGraph edge-source child was not a GPU-resident producer",
                )
                .with_fact("child_name", child.name())
                .into());
            };
            let (input_name, handle) =
                producer.execute_gpu_result_with_context(context, 0, Arc::clone(&task_context)).await?;
            gpu_inputs.insert(input_name, handle);
        }
        let result = execute_query_plan_with_gpu_inputs(context, &native_plan, &gpu_inputs).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_cudf.cugraph_sql.edge_source_composed",
                "native composed cuGraph edge-source execution failed",
                error,
            ))
        })?;
        let (output, _) = result.into_parts();
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.cugraph_sql.edge_source_composed",
                "native composed cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok(handle.into_gpu_frame());
    }

    let batches = collect(plan, task_context).await?;
    record_batches_to_gpu(context, &batches)
}

fn record_batches_to_gpu(context: &ExecutionContext, batches: &[RecordBatch]) -> Result<GpuDataFrame> {
    let Some(first_batch) = batches.first() else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_cudf.cugraph_sql.edge_source_host_import",
            "cuGraph edge source produced no RecordBatches",
        )
        .into());
    };
    let mut frames = batches
        .iter()
        .map(|batch| {
            cudf::DataFrame::from_record_batch(context.runtime(), batch).map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "datafusion_cudf.cugraph_sql.edge_source_host_import",
                    "failed to import host Arrow edge-source batch into cudf",
                    cudf_query_engine::QueryError::from_cudf(
                        "datafusion_cudf.cugraph_sql.edge_source_host_import",
                        "failed to import host Arrow edge-source batch into cudf",
                        error,
                    ),
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if frames.len() == 1 {
        return Ok(GpuDataFrame::new(frames.remove(0)));
    }
    let first = frames.remove(0);
    let rest = frames.iter().collect::<Vec<_>>();
    let concatenated = first.concat_many(&rest).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_cudf.cugraph_sql.edge_source_host_import",
            "failed to concatenate imported cuGraph edge-source batches",
            cudf_query_engine::QueryError::from_cudf(
                "datafusion_cudf.cugraph_sql.edge_source_host_import",
                "failed to concatenate imported cuGraph edge-source batches",
                error,
            ),
        ))
    })?;
    if concatenated.num_columns() != first_batch.num_columns() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_cudf.cugraph_sql.edge_source_host_import",
            "cuGraph edge-source import changed column count",
        )
        .with_fact("expected_column_count", first_batch.num_columns())
        .with_fact("actual_column_count", concatenated.num_columns())
        .into());
    }
    Ok(GpuDataFrame::new(concatenated))
}

fn execute_cugraph_query_to_host(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
) -> Result<(RecordBatch, QueryMetrics)> {
    let plan = cugraph_query_plan(
        context,
        spec,
        edge_schema,
        edge_frame,
        projection,
        limit,
        CugraphQuerySink::HostArrow,
        edge_normalize_inserted,
    )?;
    let result = execute_query_plan_with_gpu_inputs(context, &plan.plan, &plan.inputs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_cudf.cugraph_sql.execute",
            "native cuGraph SQL query execution failed",
            error,
        ))
    })?;
    let (output, metrics) = result.into_parts();
    let QueryOutput::HostArrow(batch) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_cudf.cugraph_sql.execute",
            "native cuGraph SQL query did not produce host Arrow output",
        )
        .into());
    };
    Ok((batch, metrics))
}

fn execute_cugraph_query_to_gpu(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    exec: &CugraphAlgoExec,
) -> Result<(GpuResultHandle, QueryMetrics)> {
    let plan = cugraph_query_plan(
        context,
        spec,
        edge_schema,
        edge_frame,
        exec.projection.as_ref(),
        exec.limit,
        CugraphQuerySink::GpuResident,
        exec.edge_normalize_inserted(),
    )?;
    let result = execute_query_plan_with_gpu_inputs(context, &plan.plan, &plan.inputs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_cudf.cugraph_sql.execute_gpu",
            "native cuGraph SQL GPU-resident query execution failed",
            error,
        ))
    })?;
    let (output, metrics) = result.into_parts();
    let QueryOutput::GpuResident(handle) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_cudf.cugraph_sql.execute_gpu",
            "native cuGraph SQL query did not produce GPU-resident output",
        )
        .with_fact("result_sink", "host_arrow")
        .into());
    };
    Ok((handle, metrics))
}

#[derive(Debug, Clone, Copy)]
enum CugraphQuerySink {
    HostArrow,
    GpuResident,
}

struct CugraphQueryPlan {
    plan: cudf_query_engine::QueryPlan,
    inputs: GpuResidentInputMap,
}

fn cugraph_query_plan(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    sink: CugraphQuerySink,
    edge_normalize_inserted: bool,
) -> Result<CugraphQueryPlan> {
    let input_name = "__datafusion_cudf_cugraph_edges";
    let handle = GpuResultHandle::new(context, edge_frame, Arc::clone(&edge_schema)).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_cudf.cugraph_sql.gpu_input",
            "failed to bind cuGraph edge source as a GPU-resident native input",
            error,
        ))
    })?;
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(input_name.to_string(), handle);

    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, edge_schema)));
    let mut root = builder.add_graph_algorithm(input, native_graph_spec(spec, edge_normalize_inserted));
    let mut root_schema = Arc::clone(&spec.output_schema);
    if let Some(projection) = projection {
        root_schema = projected_schema(&root_schema, Some(projection))?;
        let outputs = projection
            .iter()
            .map(|index| {
                let field =
                    spec.output_schema.fields().get(*index).map(|field| field.as_ref().clone()).ok_or_else(|| {
                        plan_error(format!("cuGraph output projection index {index} is out of range"))
                    })?;
                Ok(ProjectionOutput::new(cudf_query_engine::Expr::column(field.name()), field))
            })
            .collect::<Result<Vec<_>>>()?;
        root = builder.add_projection(root, outputs);
    }
    if let Some(limit) = limit {
        root = builder.add_limit(root, LimitSpec::fetch(limit), Arc::clone(&root_schema));
    }
    let plan = match sink {
        CugraphQuerySink::HostArrow => builder.finish_with_host_sink_schema(root, root_schema),
        CugraphQuerySink::GpuResident => builder.finish_gpu_resident(root),
    };
    Ok(CugraphQueryPlan { plan, inputs })
}

fn native_graph_spec(spec: &CugraphAlgorithmSpec, edge_normalize_inserted: bool) -> GraphAlgorithmSpec {
    GraphAlgorithmSpec::new(
        native_algorithm(spec.algorithm),
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone()),
        GraphConstructionOptions {
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            policy: if edge_normalize_inserted {
                GraphConstructionPolicy::PythonCugraph
            } else {
                match spec.graph_options.construction_policy {
                    CugraphGraphConstructionPolicy::PythonCugraph | CugraphGraphConstructionPolicy::RawLibcugraph => {
                        GraphConstructionPolicy::RawLibcugraph
                    },
                }
            },
        },
        native_algorithm_params(&spec.params),
        Arc::clone(&spec.output_schema),
    )
}

fn native_algorithm(algorithm: CugraphAlgorithm) -> GraphAlgorithm {
    algorithm.native_algorithm()
}

fn native_algorithm_params(params: &CugraphAlgorithmParams) -> GraphAlgorithmParams {
    match params {
        CugraphAlgorithmParams::PageRank(params) => GraphAlgorithmParams::PageRank(GraphPageRankParams {
            alpha: params.alpha,
            epsilon: params.epsilon,
            max_iterations: params.max_iterations,
        }),
        CugraphAlgorithmParams::Bfs(params) => GraphAlgorithmParams::Bfs(GraphBfsParams {
            source_vertex: params.source_vertex,
            depth_limit: params.depth_limit,
        }),
        CugraphAlgorithmParams::Louvain(params) => {
            GraphAlgorithmParams::Louvain(GraphLouvainParams { resolution: params.resolution })
        },
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            GraphAlgorithmParams::EigenvectorCentrality(GraphEigenvectorCentralityParams {
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
            })
        },
        CugraphAlgorithmParams::KatzCentrality(params) => {
            GraphAlgorithmParams::KatzCentrality(GraphKatzCentralityParams {
                alpha: params.alpha,
                beta: params.beta,
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
            })
        },
        CugraphAlgorithmParams::Hits(params) => GraphAlgorithmParams::Hits(GraphHitsParams {
            epsilon: params.epsilon,
            max_iterations: params.max_iterations,
            normalize: params.normalize,
        }),
        CugraphAlgorithmParams::Sssp(params) => {
            GraphAlgorithmParams::Sssp(GraphSsspParams { source_vertex: params.source_vertex, cutoff: params.cutoff })
        },
        CugraphAlgorithmParams::DegreesAll(_) => GraphAlgorithmParams::DegreesAll(GraphDegreesAllParams),
        CugraphAlgorithmParams::InDegreesAll(_) => GraphAlgorithmParams::InDegreesAll(GraphInDegreesAllParams),
        CugraphAlgorithmParams::OutDegreesAll(_) => GraphAlgorithmParams::OutDegreesAll(GraphOutDegreesAllParams),
        CugraphAlgorithmParams::WeaklyConnectedComponents(_) => {
            GraphAlgorithmParams::WeaklyConnectedComponents(GraphWccParams)
        },
        CugraphAlgorithmParams::StronglyConnectedComponents(_) => {
            GraphAlgorithmParams::StronglyConnectedComponents(GraphSccParams)
        },
        CugraphAlgorithmParams::TriangleCountAll(_) => {
            GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams)
        },
        CugraphAlgorithmParams::CoreNumber(params) => GraphAlgorithmParams::CoreNumber(GraphCoreNumberParams {
            degree_type: sql_degree_type_to_native(params.degree_type),
        }),
        CugraphAlgorithmParams::KCore(params) => GraphAlgorithmParams::KCore(GraphKCoreParams {
            k: params.k,
            degree_type: sql_degree_type_to_native(params.degree_type),
        }),
        CugraphAlgorithmParams::MinimumSpanningTree(_) => GraphAlgorithmParams::MinimumSpanningTree(GraphMstParams),
        CugraphAlgorithmParams::PageRankStrict(params) => GraphAlgorithmParams::PageRankStrict(GraphPageRankParams {
            alpha: params.alpha,
            epsilon: params.epsilon,
            max_iterations: params.max_iterations,
        }),
        CugraphAlgorithmParams::Leiden(params) => GraphAlgorithmParams::Leiden(GraphLeidenParams {
            max_level: params.max_level,
            resolution: params.resolution,
            theta: params.theta,
            seed: params.seed,
        }),
        CugraphAlgorithmParams::Ecg(params) => GraphAlgorithmParams::Ecg(GraphEcgParams {
            min_weight: params.min_weight,
            ensemble_size: params.ensemble_size,
            max_level: params.max_level,
            threshold: params.threshold,
            resolution: params.resolution,
            seed: params.seed,
        }),
        CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
            GraphAlgorithmParams::SpectralModularityMaximization(GraphSpectralClusteringParams {
                n_clusters: params.n_clusters,
                n_eigenvectors: params.n_eigenvectors,
                evs_tolerance: params.evs_tolerance,
                evs_max_iterations: params.evs_max_iterations,
                k_means_tolerance: params.k_means_tolerance,
                k_means_max_iterations: params.k_means_max_iterations,
                seed: params.seed,
            })
        },
        CugraphAlgorithmParams::BalancedCutClustering(params) => {
            GraphAlgorithmParams::BalancedCutClustering(GraphSpectralClusteringParams {
                n_clusters: params.n_clusters,
                n_eigenvectors: params.n_eigenvectors,
                evs_tolerance: params.evs_tolerance,
                evs_max_iterations: params.evs_max_iterations,
                k_means_tolerance: params.k_means_tolerance,
                k_means_max_iterations: params.k_means_max_iterations,
                seed: params.seed,
            })
        },
    }
}

fn sql_degree_type_to_native(degree_type: CugraphKCoreDegreeType) -> GraphKCoreDegreeType {
    match degree_type {
        CugraphKCoreDegreeType::In => GraphKCoreDegreeType::In,
        CugraphKCoreDegreeType::Out => GraphKCoreDegreeType::Out,
        CugraphKCoreDegreeType::InOut => GraphKCoreDegreeType::InOut,
    }
}
