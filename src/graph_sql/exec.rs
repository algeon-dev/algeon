use std::{
    any::Any,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use arrow::{
    array::{Array, ArrayRef, Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray},
    datatypes::{DataType, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::{DataFrame, Series, frame::factorization::Codebook};
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
use nexus_query_engine::{
    BinaryOp, CaseWhen, ExecutionContext, Expr as NativeExpr, GpuDataFrame, GpuResidentInputMap, GpuResidentSource,
    GpuResultHandle, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams,
    GraphBfsPredicateInputs as NativeGraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy,
    GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeColumnBindings,
    GraphEdgePredicateInput as NativeGraphEdgePredicateInput, GraphEigenvectorCentralityParams, GraphHitsParams,
    GraphInDegreesAllParams, GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams,
    GraphLouvainParams, GraphMstParams, GraphOutDegreesAllParams, GraphPageRankParams,
    GraphPredicateInput as NativeGraphPredicateInput, GraphSccParams, GraphSpectralClusteringParams, GraphSsspParams,
    GraphTriangleCountAllParams, GraphWccParams, LimitSpec, LiteralValue, ProjectionOutput, QueryMetrics, QueryOutput,
    QueryPlanBuilder, SourceNode, execute_query_plan, execute_query_plan_with_gpu_inputs, stable_graph_result_schema,
};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsOutputMode, CugraphBfsOutputOptions,
    CugraphBfsSourceVertex, CugraphGraphConstructionPolicy, CugraphKCoreDegreeType, plan_error,
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
    predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
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
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: NativeCudfExecutionConfig,
    ) -> Result<Self> {
        let id = NEXT_CUGRAPH_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        Self::try_new_with_gpu_result_input_name(
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
            format!("__datafusion_nexus_cugraph_result_{id}"),
        )
    }

    fn try_new_with_gpu_result_input_name(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
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
            predicate_sources,
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
        if is_bfs_path_output(&self.spec) {
            if partition != 0 {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.cugraph_sql.execute_gpu",
                    "cuGraph SQL GPU-resident producer exposes exactly one output partition",
                )
                .with_fact("partition", partition)
                .into());
            }
            let edge_schema = self.edge_source.schema();
            let edge_frame =
                execute_edge_source_to_gpu(native_context, Arc::clone(&self.edge_source), Arc::clone(&task_context))
                    .await?;
            let predicate_frames =
                execute_predicate_sources_to_gpu(native_context, &self.predicate_sources, Arc::clone(&task_context))
                    .await?;
            let path_target =
                bfs_path_target_vertex(native_context, &self.spec, &predicate_frames)?.expect("path mode checked");
            let prepared =
                prepare_bfs_logical_gpu_inputs(native_context, &self.spec, edge_schema, edge_frame, predicate_frames)?;
            let (batch, query_metrics) = if let Some(codebook) = prepared.logical_codebook {
                let (handle, query_metrics) = execute_cugraph_query_to_gpu_with_options(
                    native_context,
                    &prepared.execution_spec,
                    prepared.edge_schema,
                    prepared.edge_frame,
                    prepared.predicate_frames,
                    None,
                    None,
                    self.edge_normalize_inserted(),
                )?;
                let decoded = decode_bfs_logical_gpu_output(
                    native_context,
                    handle,
                    &codebook,
                    bfs_decoded_output_schema_for_host(&self.spec),
                )?;
                let batch = decoded.to_host_record_batch().map_err(|error| {
                    DataFusionError::from(Error::from_native(
                        "datafusion_nexus.cugraph_sql.logical_output",
                        "failed to materialize decoded cuGraph logical BFS output",
                        error,
                    ))
                })?;
                (batch, query_metrics)
            } else {
                execute_cugraph_query_to_host(
                    native_context,
                    &self.spec,
                    prepared.edge_schema,
                    prepared.edge_frame,
                    prepared.predicate_frames,
                    None,
                    None,
                    self.edge_normalize_inserted(),
                )?
            };
            let path = reconstruct_bfs_path_batch(&self.spec, &batch, path_target)?;
            let path = project_and_limit_host_batch(path, self.projection.as_ref(), self.limit)?;
            let schema = projected_schema(&self.spec.output_schema, self.projection.as_ref())?;
            let frame = record_batches_to_gpu(native_context, &[path])?;
            let handle = GpuResultHandle::new(native_context, frame, schema).map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "datafusion_nexus.cugraph_sql.path_output",
                    "failed to bind BFS path output as a GPU-resident result",
                    error,
                ))
            })?;
            let metrics = NativeCudfMetrics::new(&self.metrics, partition);
            metrics.record_query_metrics(&query_metrics);
            return Ok((self.gpu_result_input_name.clone(), handle));
        }
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.execute_gpu",
                "cuGraph SQL GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }
        let edge_schema = self.edge_source.schema();
        let edge_frame =
            execute_edge_source_to_gpu(native_context, Arc::clone(&self.edge_source), Arc::clone(&task_context))
                .await?;
        let predicate_frames =
            execute_predicate_sources_to_gpu(native_context, &self.predicate_sources, Arc::clone(&task_context))
                .await?;
        let metrics = NativeCudfMetrics::new(&self.metrics, partition);
        let prepared =
            prepare_bfs_logical_gpu_inputs(native_context, &self.spec, edge_schema, edge_frame, predicate_frames)?;
        let (handle, query_metrics) = execute_cugraph_query_to_gpu_with_options(
            native_context,
            &prepared.execution_spec,
            prepared.edge_schema,
            prepared.edge_frame,
            prepared.predicate_frames,
            if prepared.logical_codebook.is_some() { None } else { self.projection.as_ref() },
            if prepared.logical_codebook.is_some() { None } else { self.limit },
            self.edge_normalize_inserted(),
        )?;
        let handle = if let Some(codebook) = prepared.logical_codebook {
            let decoded =
                decode_bfs_logical_gpu_output(native_context, handle, &codebook, Arc::clone(&self.spec.output_schema))?;
            apply_gpu_projection_limit(
                native_context,
                decoded,
                Arc::clone(&self.spec.output_schema),
                self.projection.as_ref(),
                self.limit,
            )?
        } else {
            handle
        };
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
                "CugraphAlgoExec: algorithm={:?}, table={}, src_col={}, dst_col={}, weight_col={}, edge_id_col={}, predicate_source_count={}, construction_policy={}, memory_control_policy={}, edge_normalize_inserted={}, output_boundary=device_resident_or_device_to_host, gpu_result_input={}",
                self.spec.algorithm,
                self.spec.table_name,
                self.spec.src_col,
                self.spec.dst_col,
                self.spec.weight_col.as_deref().unwrap_or("<none>"),
                self.spec.edge_id_col.as_deref().unwrap_or("<none>"),
                self.predicate_sources.len(),
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
                writeln!(f, "edge_id_col={}", self.spec.edge_id_col.as_deref().unwrap_or("<none>"))?;
                writeln!(f, "predicate_source_count={}", self.predicate_sources.len())?;
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
        let mut children = Vec::with_capacity(1 + self.predicate_sources.len());
        children.push(&self.edge_source);
        children.extend(self.predicate_sources.iter());
        children
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        let expected = 1 + self.predicate_sources.len();
        if children.len() != expected {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "datafusion_nexus.cugraph_sql.with_new_children",
                "cuGraph SQL execution received an unexpected child count",
            )
            .with_fact("expected_child_count", expected)
            .with_fact("actual_child_count", children.len())
            .into());
        }
        let mut children = children.into_iter();
        let edge_source = children.next().expect("child count checked above");
        let predicate_sources = children.collect();
        Ok(Arc::new(Self::try_new_with_gpu_result_input_name(
            self.spec.clone(),
            edge_source,
            predicate_sources,
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
                "datafusion_nexus.cugraph_sql.execute",
                "cuGraph SQL execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let schema = self.schema();
        let spec = self.spec.clone();
        let edge_source = Arc::clone(&self.edge_source);
        let predicate_sources = self.predicate_sources.clone();
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
            let edge_frame =
                execute_edge_source_to_gpu(context, Arc::clone(&edge_source), Arc::clone(&task_context)).await?;
            let predicate_frames =
                execute_predicate_sources_to_gpu(context, &predicate_sources, Arc::clone(&task_context)).await?;
            let path_target = bfs_path_target_vertex(context, &spec, &predicate_frames)?;
            let prepared = prepare_bfs_logical_gpu_inputs(context, &spec, edge_schema, edge_frame, predicate_frames)?;
            let (batch, query_metrics) = if let Some(codebook) = prepared.logical_codebook {
                let (handle, query_metrics) = execute_cugraph_query_to_gpu_with_options(
                    context,
                    &prepared.execution_spec,
                    prepared.edge_schema,
                    prepared.edge_frame,
                    prepared.predicate_frames,
                    None,
                    None,
                    edge_normalize_inserted,
                )?;
                let decoded_schema = bfs_decoded_output_schema_for_host(&spec);
                let decoded = decode_bfs_logical_gpu_output(context, handle, &codebook, decoded_schema)?;
                let batch = decoded.to_host_record_batch().map_err(|error| {
                    DataFusionError::from(Error::from_native(
                        "datafusion_nexus.cugraph_sql.logical_output",
                        "failed to materialize decoded cuGraph logical BFS output",
                        error,
                    ))
                })?;
                let batch = if let Some(target) = path_target {
                    reconstruct_bfs_path_batch(&spec, &batch, target)?
                } else {
                    batch
                };
                (project_and_limit_host_batch(batch, projection.as_ref(), limit)?, query_metrics)
            } else {
                let (batch, query_metrics) = execute_cugraph_query_to_host(
                    context,
                    &spec,
                    prepared.edge_schema,
                    prepared.edge_frame,
                    prepared.predicate_frames,
                    if path_target.is_some() { None } else { projection.as_ref() },
                    if path_target.is_some() { None } else { limit },
                    edge_normalize_inserted,
                )?;
                let batch = if let Some(target) = path_target {
                    let path = reconstruct_bfs_path_batch(&spec, &batch, target)?;
                    project_and_limit_host_batch(path, projection.as_ref(), limit)?
                } else {
                    batch
                };
                (batch, query_metrics)
            };
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
                "datafusion_nexus.cugraph_sql.edge_source_native",
                "native cuGraph edge-source execution failed",
                error,
            ))
        })?;
        let (output, _) = result.into_parts();
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.edge_source_native",
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
                    "datafusion_nexus.cugraph_sql.edge_source_composed",
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
                "datafusion_nexus.cugraph_sql.edge_source_composed",
                "native composed cuGraph edge-source execution failed",
                error,
            ))
        })?;
        let (output, _) = result.into_parts();
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.edge_source_composed",
                "native composed cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok(handle.into_gpu_frame());
    }

    let batches = collect(plan, task_context).await?;
    record_batches_to_gpu(context, &batches)
}

struct CugraphGpuFrame {
    schema: SchemaRef,
    frame: GpuDataFrame,
}

async fn execute_predicate_sources_to_gpu(
    context: &ExecutionContext,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    task_context: Arc<TaskContext>,
) -> Result<Vec<CugraphGpuFrame>> {
    let mut frames = Vec::with_capacity(predicate_sources.len());
    for source in predicate_sources {
        let schema = source.schema();
        let frame = execute_edge_source_to_gpu(context, Arc::clone(source), Arc::clone(&task_context)).await?;
        frames.push(CugraphGpuFrame { schema, frame });
    }
    Ok(frames)
}

struct PreparedCugraphInputs {
    execution_spec: CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    logical_codebook: Option<Codebook>,
}

fn prepare_bfs_logical_gpu_inputs(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
) -> Result<PreparedCugraphInputs> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(PreparedCugraphInputs {
            execution_spec: spec.clone(),
            edge_schema,
            edge_frame,
            predicate_frames,
            logical_codebook: None,
        });
    };
    let CugraphBfsSourceVertex::Utf8(source_vertex) = &params.source_vertex else {
        return Ok(PreparedCugraphInputs {
            execution_spec: spec.clone(),
            edge_schema,
            edge_frame,
            predicate_frames,
            logical_codebook: None,
        });
    };
    if spec.edge_id_col.is_some() || params.predicates.include_edge_ids.is_some() {
        return Err(plan_error(
            "cuGraph BFS logical vertex-id mapping does not yet support edge_id_col or include_edge_ids predicates",
        ));
    }

    let edge_frame = edge_frame.into_frame();
    validate_bfs_logical_edge_schema(spec, &edge_schema)?;
    let mut domain_parts = vec![cudf_series(&edge_frame, &spec.src_col)?, cudf_series(&edge_frame, &spec.dst_col)?];
    let source_series = context.runtime().series(&[source_vertex.as_str()]).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to create GPU source-vertex series for cuGraph logical BFS mapping",
            nexus_query_engine::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to create GPU source-vertex series for cuGraph logical BFS mapping",
                error,
            ),
        ))
    })?;
    domain_parts.push(source_series.clone());
    for (index, frame) in predicate_frames.iter().enumerate() {
        collect_bfs_logical_predicate_domain_parts(params, index, frame, &mut domain_parts)?;
    }
    let domain_refs = domain_parts.iter().collect::<Vec<_>>();
    let domain = Series::concat(&domain_refs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to concatenate cuGraph logical BFS mapping domain on GPU",
            nexus_query_engine::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to concatenate cuGraph logical BFS mapping domain on GPU",
                error,
            ),
        ))
    })?;
    let codebook = domain.codebook().map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to build cuGraph logical BFS codebook on GPU",
            nexus_query_engine::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to build cuGraph logical BFS codebook on GPU",
                error,
            ),
        ))
    })?;

    let source_code = codebook
        .encode(&source_series)
        .map_err(cudf_logical_mapping_error)?
        .into_series()
        .copy_into_runtime(context.runtime())
        .map_err(cudf_logical_mapping_error)?;
    let source_frame =
        DataFrame::from_series([("__source_vertex", source_code)]).map_err(cudf_logical_mapping_error)?;
    let source_schema =
        Arc::new(Schema::new(vec![arrow::datatypes::Field::new("__source_vertex", DataType::Int32, false)]));

    let mapped_edge = encode_bfs_logical_edge_frame(spec, edge_frame, &codebook)?;
    let mapped_edge_schema = remap_bfs_logical_schema(&edge_schema, &[&spec.src_col, &spec.dst_col])?;
    let mut mapped_predicates = Vec::with_capacity(predicate_frames.len() + 1);
    mapped_predicates.push(CugraphGpuFrame { schema: source_schema, frame: GpuDataFrame::new(source_frame) });
    for (index, frame) in predicate_frames.into_iter().enumerate() {
        mapped_predicates.push(encode_bfs_logical_predicate_frame(params, index, frame, &codebook)?);
    }

    let mut execution_spec = spec.clone();
    execution_spec.output_schema = bfs_numeric_execution_schema(params.output);
    Ok(PreparedCugraphInputs {
        execution_spec,
        edge_schema: mapped_edge_schema,
        edge_frame: GpuDataFrame::new(mapped_edge),
        predicate_frames: mapped_predicates,
        logical_codebook: Some(codebook),
    })
}

fn validate_bfs_logical_edge_schema(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<()> {
    for column in [&spec.src_col, &spec.dst_col] {
        let field = edge_schema.field_with_name(column).map_err(|_| {
            plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{column}'"))
        })?;
        if field.is_nullable() {
            return Err(plan_error(format!("cuGraph BFS logical vertex column '{column}' must be non-null")));
        }
        if !is_bfs_logical_vertex_type(field.data_type()) {
            return Err(plan_error(format!(
                "cuGraph BFS logical vertex mapping requires a string edge column '{column}', found {:?}",
                field.data_type()
            )));
        }
    }
    Ok(())
}

fn is_bfs_logical_vertex_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn collect_bfs_logical_predicate_domain_parts(
    params: &super::CugraphBfsParams,
    index: usize,
    frame: &CugraphGpuFrame,
    domain_parts: &mut Vec<Series>,
) -> Result<()> {
    let mut expected_index = 0;
    if let Some(input) = &params.predicates.include_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.exclude_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.target_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if params.predicates.include_edge_ids.is_some() {
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.include_edges
        && index == expected_index
    {
        domain_parts.push(cudf_series(frame.frame.frame(), &input.source_col)?);
        domain_parts.push(cudf_series(frame.frame.frame(), &input.destination_col)?);
    }
    Ok(())
}

fn encode_bfs_logical_edge_frame(
    spec: &CugraphAlgorithmSpec,
    frame: DataFrame,
    codebook: &Codebook,
) -> Result<DataFrame> {
    let frame = encode_column(frame, &spec.src_col, codebook)?;
    encode_column(frame, &spec.dst_col, codebook)
}

fn encode_bfs_logical_predicate_frame(
    params: &super::CugraphBfsParams,
    index: usize,
    frame: CugraphGpuFrame,
    codebook: &Codebook,
) -> Result<CugraphGpuFrame> {
    let mut expected_index = 0;
    let mut raw = frame.frame.into_frame();
    let mut encoded_columns = Vec::new();
    if let Some(input) = &params.predicates.include_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.exclude_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.target_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if params.predicates.include_edge_ids.is_some() {
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.include_edges
        && index == expected_index
    {
        encoded_columns.push(input.source_col.as_str());
        encoded_columns.push(input.destination_col.as_str());
    }
    for column in &encoded_columns {
        raw = encode_column(raw, column, codebook)?;
    }
    Ok(CugraphGpuFrame {
        schema: remap_bfs_logical_schema(&frame.schema, &encoded_columns)?,
        frame: GpuDataFrame::new(raw),
    })
}

fn encode_column(frame: DataFrame, column: &str, codebook: &Codebook) -> Result<DataFrame> {
    let encoded = codebook.encode(&cudf_series(&frame, column)?).map_err(cudf_logical_mapping_error)?.into_series();
    frame.with_column(column, encoded).map_err(cudf_logical_mapping_error)
}

fn remap_bfs_logical_schema(schema: &SchemaRef, columns: &[&str]) -> Result<SchemaRef> {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if columns.iter().any(|column| *column == field.name()) {
                arrow::datatypes::Field::new(field.name(), DataType::Int32, false)
            } else {
                field.as_ref().clone()
            }
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(Schema::new(fields)))
}

fn bfs_numeric_execution_schema(output: CugraphBfsOutputOptions) -> SchemaRef {
    let mut fields = match output.mode {
        CugraphBfsOutputMode::Raw => stable_graph_result_schema(native_algorithm(CugraphAlgorithm::Bfs)),
        CugraphBfsOutputMode::Normalized => Arc::new(Schema::new(vec![
            arrow::datatypes::Field::new("vertex", DataType::Int64, false),
            arrow::datatypes::Field::new("distance", DataType::Int64, true),
            arrow::datatypes::Field::new("predecessor", DataType::Int64, true),
            arrow::datatypes::Field::new("reachable", DataType::Boolean, false),
        ])),
        CugraphBfsOutputMode::Path => stable_graph_result_schema(native_algorithm(CugraphAlgorithm::Bfs)),
    }
    .fields()
    .iter()
    .map(|field| field.as_ref().clone())
    .collect::<Vec<_>>();
    if output.return_target_info {
        fields.push(arrow::datatypes::Field::new("target_found", DataType::Boolean, false));
        fields.push(arrow::datatypes::Field::new("target_distance", DataType::Int64, true));
    }
    Arc::new(Schema::new(fields))
}

fn decode_bfs_logical_gpu_output(
    context: &ExecutionContext,
    handle: GpuResultHandle,
    codebook: &Codebook,
    output_schema: SchemaRef,
) -> Result<GpuResultHandle> {
    let raw = handle.into_gpu_frame().into_frame();
    let vertex = decode_code_column(&raw, "vertex", codebook)?
        .copy_into_runtime(context.runtime())
        .map_err(cudf_logical_mapping_error)?;
    let distance =
        cudf_series(&raw, "distance")?.copy_into_runtime(context.runtime()).map_err(cudf_logical_mapping_error)?;
    let predecessor = decode_code_column(&raw, "predecessor", codebook)?
        .copy_into_runtime(context.runtime())
        .map_err(cudf_logical_mapping_error)?;
    let mut columns = vec![("vertex", vertex), ("distance", distance), ("predecessor", predecessor)];
    if raw.column_names().iter().any(|name| *name == "reachable") {
        let reachable =
            cudf_series(&raw, "reachable")?.copy_into_runtime(context.runtime()).map_err(cudf_logical_mapping_error)?;
        columns.push(("reachable", reachable));
    }
    if raw.column_names().iter().any(|name| *name == "target_found") {
        let target_found = cudf_series(&raw, "target_found")?
            .copy_into_runtime(context.runtime())
            .map_err(cudf_logical_mapping_error)?;
        let target_distance = cudf_series(&raw, "target_distance")?
            .copy_into_runtime(context.runtime())
            .map_err(cudf_logical_mapping_error)?;
        columns.push(("target_found", target_found));
        columns.push(("target_distance", target_distance));
    }
    let frame = DataFrame::from_series(columns).map_err(cudf_logical_mapping_error)?;
    GpuResultHandle::new(context, GpuDataFrame::new(frame), output_schema).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_output",
            "failed to bind decoded cuGraph logical BFS output",
            error,
        ))
    })
}

fn decode_code_column(frame: &DataFrame, column: &str, codebook: &Codebook) -> Result<Series> {
    let codes = cudf_series(frame, column)?.astype(cudf::TypeId::Int32).map_err(cudf_logical_mapping_error)?;
    codebook.decode_series(&codes).map_err(cudf_logical_mapping_error)
}

fn apply_gpu_projection_limit(
    context: &ExecutionContext,
    handle: GpuResultHandle,
    schema: SchemaRef,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
) -> Result<GpuResultHandle> {
    if projection.is_none() && limit.is_none() {
        return Ok(handle);
    }
    let input_name = "__datafusion_nexus_cugraph_logical_output";
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(input_name.to_string(), handle);
    let mut builder = QueryPlanBuilder::new();
    let mut root = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&schema))));
    let input_schema = Arc::clone(&schema);
    let mut root_schema = schema;
    if let Some(projection) = projection {
        let outputs = projection
            .iter()
            .map(|index| {
                let field =
                    input_schema.fields().get(*index).map(|field| field.as_ref().clone()).ok_or_else(|| {
                        plan_error(format!("cuGraph output projection index {index} is out of range"))
                    })?;
                Ok(ProjectionOutput::new(nexus_query_engine::Expr::column(field.name()), field))
            })
            .collect::<Result<Vec<_>>>()?;
        root_schema = projected_schema(&input_schema, Some(projection))?;
        root = builder.add_projection(root, outputs);
    }
    if let Some(limit) = limit {
        root = builder.add_limit(root, LimitSpec::fetch(limit), Arc::clone(&root_schema));
    }
    let plan = builder.finish_gpu_resident(root);
    let result = execute_query_plan_with_gpu_inputs(context, &plan, &inputs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_output",
            "failed to apply GPU projection/limit to decoded cuGraph logical BFS output",
            error,
        ))
    })?;
    let (output, _) = result.into_parts();
    let QueryOutput::GpuResident(handle) = output else {
        return Err(plan_error("decoded cuGraph logical BFS projection did not produce a GPU output"));
    };
    Ok(handle)
}

fn project_and_limit_host_batch(
    batch: RecordBatch,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
) -> Result<RecordBatch> {
    let mut batch = if let Some(limit) = limit { batch.slice(0, limit.min(batch.num_rows())) } else { batch };
    if let Some(projection) = projection {
        batch = batch.project(projection).map_err(DataFusionError::from)?;
    }
    Ok(batch)
}

fn is_bfs_path_output(spec: &CugraphAlgorithmSpec) -> bool {
    matches!(
        &spec.params,
        CugraphAlgorithmParams::Bfs(params) if params.output.mode == CugraphBfsOutputMode::Path
    )
}

fn bfs_decoded_output_schema_for_host(spec: &CugraphAlgorithmSpec) -> SchemaRef {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Arc::clone(&spec.output_schema);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Arc::clone(&spec.output_schema);
    }
    Arc::new(Schema::new(vec![
        arrow::datatypes::Field::new("vertex", DataType::Utf8, false),
        arrow::datatypes::Field::new("distance", DataType::Int64, false),
        arrow::datatypes::Field::new("predecessor", DataType::Utf8, true),
    ]))
}

#[derive(Debug, Clone, Eq)]
enum BfsPathVertex {
    Int64(i64),
    Utf8(String),
}

impl PartialEq for BfsPathVertex {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int64(left), Self::Int64(right)) => left == right,
            (Self::Utf8(left), Self::Utf8(right)) => left == right,
            _ => false,
        }
    }
}

impl Hash for BfsPathVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Int64(value) => {
                0_u8.hash(state);
                value.hash(state);
            },
            Self::Utf8(value) => {
                1_u8.hash(state);
                value.hash(state);
            },
        }
    }
}

#[derive(Debug, Clone)]
struct BfsPathNode {
    distance: Option<i64>,
    predecessor: Option<BfsPathVertex>,
}

fn bfs_path_target_vertex(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    predicate_frames: &[CugraphGpuFrame],
) -> Result<Option<BfsPathVertex>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Ok(None);
    }
    let target =
        params.predicates.target_vertices.as_ref().ok_or_else(|| {
            plan_error("BFS path output requires target_vertices_table and target_vertex_col options")
        })?;
    let mut target_index = 0;
    if params.predicates.include_vertices.is_some() {
        target_index += 1;
    }
    if params.predicates.exclude_vertices.is_some() {
        target_index += 1;
    }
    let frame = predicate_frames
        .get(target_index)
        .ok_or_else(|| plan_error("BFS path output target predicate source was not planned"))?;
    let handle =
        GpuResultHandle::new(context, GpuDataFrame::new(frame.frame.frame().clone()), Arc::clone(&frame.schema))
            .map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "datafusion_nexus.cugraph_sql.path_output",
                    "failed to bind BFS target predicate source for path reconstruction",
                    error,
                ))
            })?;
    let batch = handle.to_host_record_batch().map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.path_output",
            "failed to materialize BFS target predicate source for path reconstruction",
            error,
        ))
    })?;
    if batch.num_rows() != 1 {
        return Err(plan_error(format!(
            "BFS path output requires exactly one target vertex row, found {}",
            batch.num_rows()
        )));
    }
    let column_index = batch.schema().index_of(&target.column).map_err(DataFusionError::from)?;
    vertex_value(batch.column(column_index).as_ref(), 0).map(Some)
}

fn reconstruct_bfs_path_batch(
    spec: &CugraphAlgorithmSpec,
    batch: &RecordBatch,
    target: BfsPathVertex,
) -> Result<RecordBatch> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Err(plan_error("BFS path reconstruction received non-BFS parameters"));
    };
    let source = match &params.source_vertex {
        CugraphBfsSourceVertex::Int64(value) => BfsPathVertex::Int64(*value),
        CugraphBfsSourceVertex::Utf8(value) => BfsPathVertex::Utf8(value.clone()),
    };
    let vertex_col = batch.schema().index_of("vertex").map_err(DataFusionError::from)?;
    let distance_col = batch.schema().index_of("distance").map_err(DataFusionError::from)?;
    let predecessor_col = batch.schema().index_of("predecessor").map_err(DataFusionError::from)?;
    let mut nodes = HashMap::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let vertex = vertex_value(batch.column(vertex_col).as_ref(), row)?;
        let distance = optional_i64_value(batch.column(distance_col).as_ref(), row)?;
        let predecessor = optional_vertex_value(batch.column(predecessor_col).as_ref(), row)?;
        nodes.insert(vertex, BfsPathNode { distance, predecessor });
    }
    let path = bfs_path_vertices(&nodes, source.clone(), target.clone())?;
    let distances = path
        .iter()
        .enumerate()
        .map(|(index, vertex)| nodes.get(vertex).and_then(|node| node.distance).unwrap_or(index as i64))
        .collect::<Vec<_>>();
    build_bfs_path_batch(Arc::clone(&spec.output_schema), source, target, path, distances)
}

fn bfs_path_vertices(
    nodes: &HashMap<BfsPathVertex, BfsPathNode>,
    source: BfsPathVertex,
    target: BfsPathVertex,
) -> Result<Vec<BfsPathVertex>> {
    let Some(target_node) = nodes.get(&target) else {
        return Ok(Vec::new());
    };
    if target_node.distance.is_none() {
        return Ok(Vec::new());
    }
    let mut reversed = Vec::new();
    let mut current = target;
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(current.clone()) {
            return Err(plan_error("BFS path reconstruction encountered a predecessor cycle"));
        }
        reversed.push(current.clone());
        if current == source {
            break;
        }
        let node = nodes.get(&current).ok_or_else(|| {
            plan_error("BFS path reconstruction found a predecessor vertex missing from the BFS output")
        })?;
        let predecessor = node.predecessor.clone().ok_or_else(|| {
            plan_error("BFS path reconstruction reached a vertex without a predecessor before the source")
        })?;
        current = predecessor;
    }
    reversed.reverse();
    Ok(reversed)
}

fn build_bfs_path_batch(
    schema: SchemaRef,
    source: BfsPathVertex,
    target: BfsPathVertex,
    path: Vec<BfsPathVertex>,
    distances: Vec<i64>,
) -> Result<RecordBatch> {
    let path_index =
        Arc::new(Int64Array::from((0..path.len()).map(|value| value as i64).collect::<Vec<_>>())) as ArrayRef;
    let distance = Arc::new(Int64Array::from(distances)) as ArrayRef;
    let (source_column, target_column, vertex_column): (ArrayRef, ArrayRef, ArrayRef) = match (&source, &target) {
        (BfsPathVertex::Int64(source), BfsPathVertex::Int64(target)) => (
            Arc::new(Int64Array::from(vec![*source; path.len()])) as ArrayRef,
            Arc::new(Int64Array::from(vec![*target; path.len()])) as ArrayRef,
            Arc::new(Int64Array::from(
                path.iter()
                    .map(|vertex| match vertex {
                        BfsPathVertex::Int64(value) => *value,
                        BfsPathVertex::Utf8(_) => unreachable!("path vertex type checked by source/target"),
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (BfsPathVertex::Utf8(source), BfsPathVertex::Utf8(target)) => (
            Arc::new(StringArray::from(vec![source.clone(); path.len()])) as ArrayRef,
            Arc::new(StringArray::from(vec![target.clone(); path.len()])) as ArrayRef,
            Arc::new(StringArray::from(
                path.iter()
                    .map(|vertex| match vertex {
                        BfsPathVertex::Utf8(value) => value.clone(),
                        BfsPathVertex::Int64(_) => unreachable!("path vertex type checked by source/target"),
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        _ => {
            return Err(plan_error("BFS path output source and target vertex dtypes must match"));
        },
    };
    RecordBatch::try_new(schema, vec![path_index, source_column, target_column, vertex_column, distance])
        .map_err(DataFusionError::from)
}

fn vertex_value(array: &dyn Array, row: usize) -> Result<BfsPathVertex> {
    if array.is_null(row) {
        return Err(plan_error("BFS path vertex values must be non-null"));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(BfsPathVertex::Int64(array.value(row)));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(BfsPathVertex::Int64(i64::from(array.value(row))));
    }
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    Err(plan_error(format!("BFS path reconstruction does not support vertex array type {:?}", array.data_type())))
}

fn optional_vertex_value(array: &dyn Array, row: usize) -> Result<Option<BfsPathVertex>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let value = vertex_value(array, row)?;
    if value == BfsPathVertex::Int64(-1) {
        return Ok(None);
    }
    Ok(Some(value))
}

fn optional_i64_value(array: &dyn Array, row: usize) -> Result<Option<i64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        let value = array.value(row);
        if value == i64::MAX {
            return Ok(None);
        }
        return Ok(Some(value));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        let value = array.value(row);
        if value == i32::MAX {
            return Ok(None);
        }
        return Ok(Some(i64::from(value)));
    }
    Err(plan_error(format!("BFS path reconstruction does not support distance array type {:?}", array.data_type())))
}

fn cudf_logical_mapping_error(error: cudf::Error) -> DataFusionError {
    Error::from_native(
        "datafusion_nexus.cugraph_sql.logical_mapping",
        "cuGraph logical BFS GPU mapping failed",
        nexus_query_engine::QueryError::from_cudf(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "cuGraph logical BFS GPU mapping failed",
            error,
        ),
    )
    .into()
}

fn cudf_series(frame: &DataFrame, column: &str) -> Result<Series> {
    frame.series(column).map_err(cudf_logical_mapping_error)
}

fn record_batches_to_gpu(context: &ExecutionContext, batches: &[RecordBatch]) -> Result<GpuDataFrame> {
    let Some(first_batch) = batches.first() else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.edge_source_host_import",
            "cuGraph edge source produced no RecordBatches",
        )
        .into());
    };
    let mut frames = batches
        .iter()
        .map(|batch| {
            cudf::DataFrame::from_record_batch(context.runtime(), batch).map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "datafusion_nexus.cugraph_sql.edge_source_host_import",
                    "failed to import host Arrow edge-source batch into cudf",
                    nexus_query_engine::QueryError::from_cudf(
                        "datafusion_nexus.cugraph_sql.edge_source_host_import",
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
            "datafusion_nexus.cugraph_sql.edge_source_host_import",
            "failed to concatenate imported cuGraph edge-source batches",
            nexus_query_engine::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.edge_source_host_import",
                "failed to concatenate imported cuGraph edge-source batches",
                error,
            ),
        ))
    })?;
    if concatenated.num_columns() != first_batch.num_columns() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.edge_source_host_import",
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
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
) -> Result<(RecordBatch, QueryMetrics)> {
    let graph_context = graph_algorithm_execution_context(context);
    let plan = cugraph_query_plan(
        &graph_context,
        spec,
        edge_schema,
        edge_frame,
        predicate_frames,
        projection,
        limit,
        CugraphQuerySink::HostArrow,
        edge_normalize_inserted,
    )?;
    let result = execute_query_plan_with_gpu_inputs(&graph_context, &plan.plan, &plan.inputs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.execute",
            "native cuGraph SQL query execution failed",
            error,
        ))
    })?;
    let (output, metrics) = result.into_parts();
    let QueryOutput::HostArrow(batch) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.execute",
            "native cuGraph SQL query did not produce host Arrow output",
        )
        .into());
    };
    Ok((batch, metrics))
}

fn execute_cugraph_query_to_gpu_with_options(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
) -> Result<(GpuResultHandle, QueryMetrics)> {
    let graph_context = graph_algorithm_execution_context(context);
    let plan = cugraph_query_plan(
        &graph_context,
        spec,
        edge_schema,
        edge_frame,
        predicate_frames,
        projection,
        limit,
        CugraphQuerySink::GpuResident,
        edge_normalize_inserted,
    )?;
    let result = execute_query_plan_with_gpu_inputs(&graph_context, &plan.plan, &plan.inputs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.execute_gpu",
            "native cuGraph SQL GPU-resident query execution failed",
            error,
        ))
    })?;
    let (output, metrics) = result.into_parts();
    let QueryOutput::GpuResident(handle) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.execute_gpu",
            "native cuGraph SQL query did not produce GPU-resident output",
        )
        .with_fact("result_sink", "host_arrow")
        .into());
    };
    Ok((handle, metrics))
}

fn graph_algorithm_execution_context(context: &ExecutionContext) -> ExecutionContext {
    // Graph algorithms are intentionally outside strict bounded admission until
    // libcudf/cuGraph expose lower-level allocation contracts. Edge and
    // predicate child plans still run under the session's configured native
    // policy; only the graph-algorithm native IR fragment uses unbounded
    // execution so it reaches the cuGraph executor instead of bounded generic
    // admission.
    ExecutionContext::from_shared_runtime(context.shared_runtime())
}

#[derive(Debug, Clone, Copy)]
enum CugraphQuerySink {
    HostArrow,
    GpuResident,
}

struct CugraphQueryPlan {
    plan: nexus_query_engine::QueryPlan,
    inputs: GpuResidentInputMap,
}

fn cugraph_query_plan(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    sink: CugraphQuerySink,
    edge_normalize_inserted: bool,
) -> Result<CugraphQueryPlan> {
    let input_name = "__datafusion_nexus_cugraph_edges";
    let handle = GpuResultHandle::new(context, edge_frame, Arc::clone(&edge_schema)).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.gpu_input",
            "failed to bind cuGraph edge source as a GPU-resident native input",
            error,
        ))
    })?;
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(input_name.to_string(), handle);

    let mut builder = QueryPlanBuilder::new();
    let input =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&edge_schema))));
    let predicate_nodes = bind_predicate_gpu_inputs(context, &mut builder, &mut inputs, predicate_frames)?;
    let graph_spec = native_graph_spec(spec, edge_normalize_inserted, &predicate_nodes);
    let mut root_schema = Arc::clone(&graph_spec.output_schema);
    let mut root = builder.add_graph_algorithm(input, graph_spec);
    if let Some(outputs) = bfs_normalized_output_projection(spec, &edge_schema)? {
        root = builder.add_projection(root, outputs);
        root_schema = Arc::clone(&spec.output_schema);
    }
    if let Some(projection) = projection {
        root_schema = projected_schema(&root_schema, Some(projection))?;
        let outputs = projection
            .iter()
            .map(|index| {
                let field =
                    spec.output_schema.fields().get(*index).map(|field| field.as_ref().clone()).ok_or_else(|| {
                        plan_error(format!("cuGraph output projection index {index} is out of range"))
                    })?;
                Ok(ProjectionOutput::new(nexus_query_engine::Expr::column(field.name()), field))
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

fn bind_predicate_gpu_inputs(
    context: &ExecutionContext,
    builder: &mut QueryPlanBuilder,
    inputs: &mut GpuResidentInputMap,
    predicate_frames: Vec<CugraphGpuFrame>,
) -> Result<Vec<nexus_query_engine::NodeId>> {
    let mut nodes = Vec::with_capacity(predicate_frames.len());
    for (index, CugraphGpuFrame { schema, frame }) in predicate_frames.into_iter().enumerate() {
        let input_name = format!("__datafusion_nexus_cugraph_bfs_predicate_{index}");
        let handle = GpuResultHandle::new(context, frame, Arc::clone(&schema)).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.gpu_input",
                "failed to bind cuGraph BFS predicate source as a GPU-resident native input",
                error,
            ))
        })?;
        inputs.insert(input_name.clone(), handle);
        nodes.push(builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema))));
    }
    Ok(nodes)
}

fn native_graph_spec(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
    predicate_nodes: &[nexus_query_engine::NodeId],
) -> GraphAlgorithmSpec {
    let mut edge_columns =
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone());
    if let Some(edge_id) = &spec.edge_id_col {
        edge_columns = edge_columns.with_edge_id(edge_id.clone());
    }
    GraphAlgorithmSpec::new(
        native_algorithm(spec.algorithm),
        edge_columns,
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
        native_algorithm_params(&spec.params, predicate_nodes),
        native_graph_output_schema(spec),
    )
}

fn native_graph_output_schema(spec: &CugraphAlgorithmSpec) -> SchemaRef {
    match &spec.params {
        CugraphAlgorithmParams::Bfs(params) => bfs_raw_graph_output_schema(params.output.return_target_info),
        _ => stable_graph_result_schema(native_algorithm(spec.algorithm)),
    }
}

fn bfs_raw_graph_output_schema(return_target_info: bool) -> SchemaRef {
    let mut fields = stable_graph_result_schema(GraphAlgorithm::Bfs)
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    if return_target_info {
        fields.push(arrow::datatypes::Field::new("target_found", DataType::Boolean, false));
        fields.push(arrow::datatypes::Field::new("target_distance", DataType::Int64, true));
    }
    Arc::new(Schema::new(fields))
}

fn bfs_normalized_output_projection(
    spec: &CugraphAlgorithmSpec,
    edge_schema: &SchemaRef,
) -> Result<Option<Vec<ProjectionOutput>>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    if params.output.mode != CugraphBfsOutputMode::Normalized {
        return Ok(None);
    }

    let unreachable_sentinel = bfs_unreachable_distance_sentinel(spec, edge_schema)?;
    let distance_is_unreachable = NativeExpr::binary(
        NativeExpr::column("distance"),
        BinaryOp::Eq,
        NativeExpr::literal(LiteralValue::Int64(unreachable_sentinel)),
    );
    let predecessor_is_absent = NativeExpr::binary(
        NativeExpr::column("predecessor"),
        BinaryOp::Eq,
        NativeExpr::literal(LiteralValue::Int64(-1)),
    );

    let fields = spec.output_schema.fields();
    let mut outputs = vec![
        ProjectionOutput::new(NativeExpr::column("vertex"), fields[0].as_ref().clone()),
        ProjectionOutput::new(
            NativeExpr::case(
                vec![CaseWhen::new(
                    distance_is_unreachable.clone(),
                    NativeExpr::literal(LiteralValue::Null(DataType::Int64)),
                )],
                NativeExpr::column("distance"),
            ),
            fields[1].as_ref().clone(),
        ),
        ProjectionOutput::new(
            NativeExpr::case(
                vec![CaseWhen::new(predecessor_is_absent, NativeExpr::literal(LiteralValue::Null(DataType::Int64)))],
                NativeExpr::column("predecessor"),
            ),
            fields[2].as_ref().clone(),
        ),
        ProjectionOutput::new(
            NativeExpr::binary(
                NativeExpr::column("distance"),
                BinaryOp::NotEq,
                NativeExpr::literal(LiteralValue::Int64(unreachable_sentinel)),
            ),
            fields[3].as_ref().clone(),
        ),
    ];
    if params.output.return_target_info {
        outputs.push(ProjectionOutput::new(NativeExpr::column("target_found"), fields[4].as_ref().clone()));
        outputs.push(ProjectionOutput::new(NativeExpr::column("target_distance"), fields[5].as_ref().clone()));
    }
    Ok(Some(outputs))
}

fn bfs_unreachable_distance_sentinel(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<i64> {
    let source_field = edge_schema.field_with_name(&spec.src_col).map_err(|_| {
        plan_error(format!(
            "cuGraph BFS output normalization could not find source column '{}' in edge source schema",
            spec.src_col
        ))
    })?;
    match source_field.data_type() {
        DataType::Int32 => Ok(i64::from(i32::MAX)),
        DataType::Int64 => Ok(i64::MAX),
        other => Err(plan_error(format!(
            "cuGraph BFS output normalization requires Int32 or Int64 source vertices, found {other:?}"
        ))),
    }
}

fn native_algorithm(algorithm: CugraphAlgorithm) -> GraphAlgorithm {
    algorithm.native_algorithm()
}

fn native_algorithm_params(
    params: &CugraphAlgorithmParams,
    predicate_nodes: &[nexus_query_engine::NodeId],
) -> GraphAlgorithmParams {
    match params {
        CugraphAlgorithmParams::PageRank(params) => GraphAlgorithmParams::PageRank(GraphPageRankParams {
            alpha: params.alpha,
            epsilon: params.epsilon,
            max_iterations: params.max_iterations,
        }),
        CugraphAlgorithmParams::Bfs(params) => {
            GraphAlgorithmParams::Bfs(Box::new(native_bfs_params(params, predicate_nodes)))
        },
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

fn native_bfs_params(
    params: &super::CugraphBfsParams,
    predicate_nodes: &[nexus_query_engine::NodeId],
) -> GraphBfsParams {
    let mut native = GraphBfsParams::new(params.source_vertex.as_i64().unwrap_or(0), params.depth_limit);
    let mut next = predicate_nodes.iter().copied();
    if params.source_vertex.is_utf8() {
        native.source_vertices = Some(NativeGraphPredicateInput::new(
            next.next().expect("BFS source-vertex node missing"),
            "__source_vertex",
        ));
    }
    native.predicates = NativeGraphBfsPredicateInputs {
        include_vertices: params.predicates.include_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        exclude_vertices: params.predicates.exclude_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        target_vertices: params.predicates.target_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        include_edge_ids: params.predicates.include_edge_ids.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        include_edges: params.predicates.include_edges.as_ref().map(|input| {
            NativeGraphEdgePredicateInput::new(
                next.next().expect("predicate node missing"),
                input.source_col.clone(),
                input.destination_col.clone(),
                input.edge_id_col.clone(),
            )
        }),
    };
    native.return_target_info = params.output.return_target_info;
    native
}

fn sql_degree_type_to_native(degree_type: CugraphKCoreDegreeType) -> GraphKCoreDegreeType {
    match degree_type {
        CugraphKCoreDegreeType::In => GraphKCoreDegreeType::In,
        CugraphKCoreDegreeType::Out => GraphKCoreDegreeType::Out,
        CugraphKCoreDegreeType::InOut => GraphKCoreDegreeType::InOut,
    }
}
