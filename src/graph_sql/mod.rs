//! SQL-facing cuGraph table-function contracts.

use std::{
    any::Any,
    collections::BTreeSet,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use arrow::{
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf_query_engine::{
    ExecutionContext, GpuDataFrame, GpuResidentInputMap, GpuResidentSource, GpuResultHandle, GraphAlgorithm,
    GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions, GraphConstructionPolicy,
    GraphEdgeColumnBindings, GraphLouvainParams, GraphPageRankParams, LimitSpec, ProjectionOutput, QueryMetrics,
    QueryOutput, QueryPlanBuilder, SourceNode, execute_query_plan, execute_query_plan_with_gpu_inputs,
};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::{DataFusionError, Result, ScalarValue, TableReference},
    datasource::{TableType, provider_as_source},
    execution::{
        SessionState,
        context::{SessionContext, TaskContext},
    },
    logical_expr::{Expr, LogicalPlanBuilder, col},
    physical_expr::{EquivalenceProperties, Partitioning},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream, collect,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet, RecordOutput},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::TryStreamExt;
use serde_json::Value;

use crate::{
    error::{Error, ErrorCode},
    native_exec::{
        NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec, NativeCudfMetrics,
    },
};

pub const CUGRAPH_PAGERANK_FUNCTION: &str = "cugraph_pagerank";
pub const CUGRAPH_BFS_FUNCTION: &str = "cugraph_bfs";
pub const CUGRAPH_LOUVAIN_FUNCTION: &str = "cugraph_louvain";

static NEXT_CUGRAPH_GPU_RESULT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct CugraphSqlConfig {
    pub allowed_algorithms: BTreeSet<CugraphAlgorithm>,
    pub default_graph_options: CugraphGraphOptions,
    pub experimental_functions: bool,
}

impl Default for CugraphSqlConfig {
    fn default() -> Self {
        Self {
            allowed_algorithms: CugraphAlgorithm::all().into_iter().collect(),
            default_graph_options: CugraphGraphOptions::default(),
            experimental_functions: false,
        }
    }
}

impl CugraphSqlConfig {
    pub fn with_allowed_algorithms(mut self, algorithms: impl IntoIterator<Item = CugraphAlgorithm>) -> Self {
        self.allowed_algorithms = algorithms.into_iter().collect();
        self
    }

    fn allows(&self, algorithm: CugraphAlgorithm) -> bool {
        self.allowed_algorithms.contains(&algorithm)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CugraphAlgorithm {
    PageRank,
    Bfs,
    Louvain,
}

impl CugraphAlgorithm {
    pub fn all() -> [Self; 3] {
        [Self::PageRank, Self::Bfs, Self::Louvain]
    }

    pub fn function_name(self) -> &'static str {
        match self {
            Self::PageRank => CUGRAPH_PAGERANK_FUNCTION,
            Self::Bfs => CUGRAPH_BFS_FUNCTION,
            Self::Louvain => CUGRAPH_LOUVAIN_FUNCTION,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphAlgorithmSpec {
    pub algorithm: CugraphAlgorithm,
    pub table_name: CugraphTableReference,
    pub src_col: String,
    pub dst_col: String,
    pub weight_col: Option<String>,
    pub graph_options: CugraphGraphOptions,
    pub params: CugraphAlgorithmParams,
    pub output_schema: SchemaRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CugraphTableReference {
    parts: Vec<String>,
}

impl CugraphTableReference {
    pub fn parse(raw: &str) -> Result<Self> {
        let parts = raw.split('.').collect::<Vec<_>>();
        if !(1..=3).contains(&parts.len()) || parts.iter().any(|part| !is_simple_identifier(part)) {
            return Err(plan_error(format!(
                "cuGraph table source '{raw}' must be a simple table, schema.table, or catalog.schema.table identifier"
            )));
        }
        Ok(Self { parts: parts.into_iter().map(ToOwned::to_owned).collect() })
    }

    pub fn parts(&self) -> &[String] {
        &self.parts
    }
}

impl fmt::Display for CugraphTableReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.parts.join("."))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphGraphOptions {
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: CugraphGraphConstructionPolicy,
}

impl Default for CugraphGraphOptions {
    fn default() -> Self {
        Self { directed: true, renumber: true, construction_policy: CugraphGraphConstructionPolicy::PythonCugraph }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphGraphConstructionPolicy {
    PythonCugraph,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CugraphAlgorithmParams {
    PageRank(CugraphPageRankParams),
    Bfs(CugraphBfsParams),
    Louvain(CugraphLouvainParams),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphPageRankParams {
    pub alpha: f64,
    pub epsilon: f64,
    pub max_iterations: u32,
}

impl Default for CugraphPageRankParams {
    fn default() -> Self {
        Self { alpha: 0.85, epsilon: 1e-6, max_iterations: 100 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CugraphBfsParams {
    pub source_vertex: i64,
    pub depth_limit: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphLouvainParams {
    pub resolution: f64,
}

impl Default for CugraphLouvainParams {
    fn default() -> Self {
        Self { resolution: 1.0 }
    }
}

#[derive(Debug)]
pub struct CugraphAlgoTableProvider {
    spec: CugraphAlgorithmSpec,
}

impl CugraphAlgoTableProvider {
    pub fn new(spec: CugraphAlgorithmSpec) -> Self {
        Self { spec }
    }

    pub fn spec(&self) -> &CugraphAlgorithmSpec {
        &self.spec
    }
}

#[async_trait::async_trait]
impl TableProvider for CugraphAlgoTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.spec.output_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !filters.is_empty() {
            return Err(plan_error("cuGraph SQL table functions do not accept pushed filters in the MVP"));
        }
        let session_state = state.as_any().downcast_ref::<SessionState>().ok_or_else(|| {
            plan_error("cuGraph SQL table functions require a DataFusion SessionState for catalog resolution")
        })?;
        let table_ref = datafusion_table_reference(&self.spec.table_name);
        let schema = session_state.schema_for_ref(table_ref.clone())?;
        let provider = schema
            .table(table_ref.table())
            .await?
            .ok_or_else(|| plan_error(format!("No table named '{}' for cuGraph edge source", self.spec.table_name)))?;
        let scan = LogicalPlanBuilder::scan(table_ref, provider_as_source(provider), None)?.build()?;
        let edge_source_plan = LogicalPlanBuilder::from(scan).project(edge_source_projection(&self.spec))?.build()?;
        let edge_source = state.create_physical_plan(&edge_source_plan).await?;
        Ok(Arc::new(CugraphAlgoExec::try_new(self.spec.clone(), edge_source, projection.cloned(), limit)?))
    }
}

#[derive(Debug, Clone)]
pub struct CugraphAlgoExec {
    spec: CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    gpu_result_input_name: String,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl CugraphAlgoExec {
    fn try_new(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
    ) -> Result<Self> {
        let id = NEXT_CUGRAPH_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        Self::try_new_with_gpu_result_input_name(
            spec,
            edge_source,
            projection,
            limit,
            format!("__datafusion_cudf_cugraph_result_{id}"),
        )
    }

    fn try_new_with_gpu_result_input_name(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        gpu_result_input_name: String,
    ) -> Result<Self> {
        let schema = projected_schema(&spec.output_schema, projection.as_ref())?;
        Ok(Self {
            spec,
            edge_source,
            projection,
            limit,
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

    pub(crate) fn gpu_result_input_name(&self) -> &str {
        &self.gpu_result_input_name
    }

    pub(crate) async fn execute_gpu_result_with_context(
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

impl DisplayAs for CugraphAlgoExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "CugraphAlgoExec: algorithm={:?}, table={}, src_col={}, dst_col={}, weight_col={}, output_boundary=device_resident_or_device_to_host, gpu_result_input={}",
                self.spec.algorithm,
                self.spec.table_name,
                self.spec.src_col,
                self.spec.dst_col,
                self.spec.weight_col.as_deref().unwrap_or("<none>"),
                self.gpu_result_input_name,
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "algorithm={:?}", self.spec.algorithm)?;
                writeln!(f, "table={}", self.spec.table_name)?;
                writeln!(f, "src_col={}", self.spec.src_col)?;
                writeln!(f, "dst_col={}", self.spec.dst_col)?;
                writeln!(f, "weight_col={}", self.spec.weight_col.as_deref().unwrap_or("<none>"))?;
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
        let metrics = NativeCudfMetrics::new(&self.metrics, partition);
        let output = futures::stream::once(async move {
            let execution_config = edge_execution_config(edge_source.as_ref());
            let context = execution_config.execution_context().map_err(|error| {
                DataFusionError::from(
                    error
                        .with_fact("cugraph_sql_phase", "runtime_context")
                        .with_fact("cpu_fallback_after_native_start", false),
                )
            })?;
            let edge_schema = edge_source.schema();
            let edge_frame = execute_edge_source_to_gpu(&context, Arc::clone(&edge_source), task_context).await?;
            let (batch, query_metrics) =
                execute_cugraph_query_to_host(&context, &spec, edge_schema, edge_frame, projection.as_ref(), limit)?;
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

fn datafusion_table_reference(table: &CugraphTableReference) -> TableReference {
    match table.parts() {
        [table] => TableReference::bare(table.as_str()),
        [schema, table] => TableReference::partial(schema.as_str(), table.as_str()),
        [catalog, schema, table] => TableReference::full(catalog.as_str(), schema.as_str(), table.as_str()),
        _ => unreachable!("CugraphTableReference::parse validates one to three identifier parts"),
    }
}

fn edge_source_projection(spec: &CugraphAlgorithmSpec) -> Vec<Expr> {
    let mut projection = vec![col(&spec.src_col), col(&spec.dst_col)];
    if let Some(weight) = &spec.weight_col {
        projection.push(col(weight));
    }
    projection
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

fn edge_execution_config(plan: &dyn ExecutionPlan) -> NativeCudfExecutionConfig {
    if let Some(native) = as_native_cudf_exec(plan) {
        return native.execution_config().clone();
    }
    if let Some(native) = as_native_cudf_composed_exec(plan) {
        return native.execution_config().clone();
    }
    NativeCudfExecutionConfig::unbounded()
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
            let Some(producer) = child.as_any().downcast_ref::<NativeCudfGpuProducerExec>() else {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_cudf.cugraph_sql.edge_source_composed",
                    "native composed cuGraph edge-source child was not a GPU-resident producer",
                )
                .with_fact("child_name", child.name())
                .into());
            };
            let (input_name, handle) = producer.execute_gpu_result_with_context(context, 0)?;
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
) -> Result<(RecordBatch, QueryMetrics)> {
    let plan =
        cugraph_query_plan(context, spec, edge_schema, edge_frame, projection, limit, CugraphQuerySink::HostArrow)?;
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
    let mut root = builder.add_graph_algorithm(input, native_graph_spec(spec));
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

fn native_graph_spec(spec: &CugraphAlgorithmSpec) -> GraphAlgorithmSpec {
    GraphAlgorithmSpec::new(
        native_algorithm(spec.algorithm),
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone()),
        GraphConstructionOptions {
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            policy: match spec.graph_options.construction_policy {
                CugraphGraphConstructionPolicy::PythonCugraph => GraphConstructionPolicy::PythonCugraph,
            },
        },
        native_algorithm_params(&spec.params),
        Arc::clone(&spec.output_schema),
    )
}

fn native_algorithm(algorithm: CugraphAlgorithm) -> GraphAlgorithm {
    match algorithm {
        CugraphAlgorithm::PageRank => GraphAlgorithm::PageRank,
        CugraphAlgorithm::Bfs => GraphAlgorithm::Bfs,
        CugraphAlgorithm::Louvain => GraphAlgorithm::Louvain,
    }
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
    }
}

#[derive(Debug)]
struct CugraphTableFunction {
    algorithm: CugraphAlgorithm,
    config: CugraphSqlConfig,
}

impl TableFunctionImpl for CugraphTableFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        let spec = parse_cugraph_algorithm_spec(self.algorithm, args, &self.config)?;
        Ok(Arc::new(CugraphAlgoTableProvider::new(spec)))
    }
}

pub fn cugraph_table_functions(config: CugraphSqlConfig) -> Vec<Arc<TableFunction>> {
    CugraphAlgorithm::all()
        .into_iter()
        .filter(|algorithm| config.allows(*algorithm))
        .map(|algorithm| {
            Arc::new(TableFunction::new(
                algorithm.function_name().to_string(),
                Arc::new(CugraphTableFunction { algorithm, config: config.clone() }),
            ))
        })
        .collect()
}

pub fn register_cugraph_udtfs(ctx: &SessionContext, config: CugraphSqlConfig) {
    for function in cugraph_table_functions(config) {
        ctx.register_udtf(function.name(), Arc::clone(function.function()));
    }
}

pub fn parse_cugraph_algorithm_spec(
    algorithm: CugraphAlgorithm,
    args: &[Expr],
    config: &CugraphSqlConfig,
) -> Result<CugraphAlgorithmSpec> {
    if !config.allows(algorithm) {
        return Err(plan_error(format!("cuGraph algorithm {:?} is not enabled in this session", algorithm)));
    }

    validate_arity(algorithm, args.len())?;
    let table_name = CugraphTableReference::parse(&literal_string_arg(args, 0, "table_name")?)?;

    let mut next = 1;
    let source_vertex = if algorithm == CugraphAlgorithm::Bfs {
        let value = literal_i64_arg(args, next, "source_vertex")?;
        next += 1;
        Some(value)
    } else {
        None
    };

    let (src_col, dst_col) = if args.len() >= next + 2 {
        let src = literal_string_arg(args, next, "src_col")?;
        let dst = literal_string_arg(args, next + 1, "dst_col")?;
        next += 2;
        (src, dst)
    } else {
        ("src".to_string(), "dst".to_string())
    };

    let weight_col = if args.len() > next {
        let weight = literal_optional_string_arg(args, next, "weight_col")?;
        next += 1;
        weight
    } else {
        None
    };

    let mut graph_options = config.default_graph_options.clone();
    let mut params = default_params(algorithm, source_vertex);
    if args.len() > next {
        let options_json = literal_string_arg(args, next, "options_json")?;
        parse_options_json(algorithm, &options_json, &mut graph_options, &mut params)?;
    }

    Ok(CugraphAlgorithmSpec {
        algorithm,
        table_name,
        src_col,
        dst_col,
        weight_col,
        graph_options,
        params,
        output_schema: stable_result_schema(algorithm),
    })
}

pub fn stable_result_schema(algorithm: CugraphAlgorithm) -> SchemaRef {
    let fields = match algorithm {
        CugraphAlgorithm::PageRank => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false)]
        },
        CugraphAlgorithm::Bfs => vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
        CugraphAlgorithm::Louvain => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false)]
        },
    };
    Arc::new(Schema::new(fields))
}

fn validate_arity(algorithm: CugraphAlgorithm, len: usize) -> Result<()> {
    let valid = match algorithm {
        CugraphAlgorithm::PageRank | CugraphAlgorithm::Louvain => matches!(len, 1 | 3 | 4 | 5),
        CugraphAlgorithm::Bfs => matches!(len, 2 | 4 | 5 | 6),
    };
    if valid {
        Ok(())
    } else {
        Err(plan_error(format!(
            "{} received {len} arguments, but expected {}",
            algorithm.function_name(),
            expected_arity(algorithm)
        )))
    }
}

fn expected_arity(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::PageRank | CugraphAlgorithm::Louvain => {
            "table_name [, src_col, dst_col, weight_col, options_json]"
        },
        CugraphAlgorithm::Bfs => "table_name, source_vertex [, src_col, dst_col, weight_col, options_json]",
    }
}

fn default_params(algorithm: CugraphAlgorithm, source_vertex: Option<i64>) -> CugraphAlgorithmParams {
    match algorithm {
        CugraphAlgorithm::PageRank => CugraphAlgorithmParams::PageRank(CugraphPageRankParams::default()),
        CugraphAlgorithm::Bfs => CugraphAlgorithmParams::Bfs(CugraphBfsParams {
            source_vertex: source_vertex.expect("BFS arity validation requires source_vertex"),
            depth_limit: None,
        }),
        CugraphAlgorithm::Louvain => CugraphAlgorithmParams::Louvain(CugraphLouvainParams::default()),
    }
}

fn parse_options_json(
    algorithm: CugraphAlgorithm,
    options_json: &str,
    graph_options: &mut CugraphGraphOptions,
    params: &mut CugraphAlgorithmParams,
) -> Result<()> {
    let value = serde_json::from_str::<Value>(options_json)
        .map_err(|error| plan_error(format!("invalid cuGraph options JSON: {error}")))?;
    let object = value.as_object().ok_or_else(|| plan_error("cuGraph options JSON must be an object"))?;

    for (key, value) in object {
        match key.as_str() {
            "directed" => graph_options.directed = json_bool(value, key)?,
            "renumber" => graph_options.renumber = json_bool(value, key)?,
            "construction_policy" => {
                let policy = json_string(value, key)?;
                if policy != "python_cugraph" {
                    return Err(plan_error(format!(
                        "unsupported cuGraph construction_policy '{policy}', expected 'python_cugraph'"
                    )));
                }
                graph_options.construction_policy = CugraphGraphConstructionPolicy::PythonCugraph;
            },
            "alpha" if algorithm == CugraphAlgorithm::PageRank => {
                let alpha = json_f64(value, key)?;
                if !(0.0..=1.0).contains(&alpha) {
                    return Err(plan_error("PageRank alpha must be in [0, 1]"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.alpha = alpha;
                }
            },
            "epsilon" if algorithm == CugraphAlgorithm::PageRank => {
                let epsilon = json_f64(value, key)?;
                if epsilon <= 0.0 {
                    return Err(plan_error("PageRank epsilon must be positive"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.epsilon = epsilon;
                }
            },
            "max_iterations" if algorithm == CugraphAlgorithm::PageRank => {
                let max_iterations = json_u32(value, key)?;
                if max_iterations < 1 {
                    return Err(plan_error("PageRank max_iterations must be at least 1"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.max_iterations = max_iterations;
                }
            },
            "depth_limit" if algorithm == CugraphAlgorithm::Bfs => {
                let depth_limit = json_i64(value, key)?;
                if depth_limit < 0 {
                    return Err(plan_error("BFS depth_limit must be non-negative"));
                }
                if let CugraphAlgorithmParams::Bfs(params) = params {
                    params.depth_limit = Some(depth_limit);
                }
            },
            "resolution" if algorithm == CugraphAlgorithm::Louvain => {
                let resolution = json_f64(value, key)?;
                if resolution <= 0.0 {
                    return Err(plan_error("Louvain resolution must be positive"));
                }
                if let CugraphAlgorithmParams::Louvain(params) = params {
                    params.resolution = resolution;
                }
            },
            _ => {
                return Err(plan_error(format!("unknown option '{key}' for {}", algorithm.function_name())));
            },
        }
    }

    Ok(())
}

fn literal_string_arg(args: &[Expr], index: usize, name: &str) -> Result<String> {
    match args.get(index) {
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)) => Ok(value.clone()),
            _ => Err(plan_error(format!("{name} must be a non-null string literal"))),
        },
        Some(_) => Err(plan_error(format!("{name} must be a literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}

fn literal_optional_string_arg(args: &[Expr], index: usize, name: &str) -> Result<Option<String>> {
    match args.get(index) {
        Some(Expr::Literal(ScalarValue::Null, _)) => Ok(None),
        Some(Expr::Literal(ScalarValue::Utf8(None), _))
        | Some(Expr::Literal(ScalarValue::LargeUtf8(None), _))
        | Some(Expr::Literal(ScalarValue::Utf8View(None), _)) => Ok(None),
        Some(_) => literal_string_arg(args, index, name).map(Some),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}

fn literal_i64_arg(args: &[Expr], index: usize, name: &str) -> Result<i64> {
    match args.get(index) {
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::Int8(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int16(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int32(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int64(Some(value)) => Ok(*value),
            ScalarValue::UInt8(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt16(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt32(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt64(Some(value)) => {
                i64::try_from(*value).map_err(|_| plan_error(format!("{name} is too large for an Int64 source vertex")))
            },
            _ => Err(plan_error(format!("{name} must be a non-null integer literal"))),
        },
        Some(_) => Err(plan_error(format!("{name} must be a literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}

fn json_bool(value: &Value, key: &str) -> Result<bool> {
    value.as_bool().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a boolean")))
}

fn json_string(value: &Value, key: &str) -> Result<String> {
    value.as_str().map(ToOwned::to_owned).ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a string")))
}

fn json_f64(value: &Value, key: &str) -> Result<f64> {
    let parsed = value.as_f64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a number")))?;
    if parsed.is_finite() { Ok(parsed) } else { Err(plan_error(format!("cuGraph option '{key}' must be finite"))) }
}

fn json_i64(value: &Value, key: &str) -> Result<i64> {
    value.as_i64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be an integer")))
}

fn json_u32(value: &Value, key: &str) -> Result<u32> {
    let parsed =
        value.as_u64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be an unsigned integer")))?;
    u32::try_from(parsed).map_err(|_| plan_error(format!("cuGraph option '{key}' exceeds u32::MAX")))
}

fn is_simple_identifier(part: &str) -> bool {
    let mut chars = part.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}
