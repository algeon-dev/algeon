use std::sync::Arc;

use arrow::datatypes::{DataType, Schema, SchemaRef};
#[cfg(feature = "cugraph")]
use arrow::{
    array::{ArrayRef, Int32Array, Int64Array},
    datatypes::Field,
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionArgs, TableFunctionImpl, TableProvider},
    common::{DataFusionError, Result},
    datasource::TableType,
    execution::{SessionState, context::SessionContext},
    logical_expr::Expr,
    physical_plan::ExecutionPlan,
};
#[cfg(not(feature = "cugraph"))]
use datafusion::{
    common::tree_node::TreeNodeRecursion,
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, PlanProperties, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
#[cfg(feature = "cugraph")]
use datafusion::{
    datasource::{MemTable, provider_as_source},
    logical_expr::{LogicalPlanBuilder, col},
};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphSqlConfig, CugraphTableReference,
    CugraphVertexIdDomain, CugraphVertexPairTable, datafusion_table_reference, parse_cugraph_algorithm_spec,
    plan_error, result_schema_for_vertex_domain, validate_similarity_pair_table_schema,
    validate_vertex_id_domain_support_for_call, vertex_id_domain_for_call,
};
#[cfg(feature = "cugraph")]
use super::{CugraphBfsSource, CugraphBfsSourceVertex};
#[cfg(feature = "cugraph")]
use crate::{
    cugraph_sql::CugraphAlgoExec, native_exec::NexusNativeExecutionConfig,
    session::session_state_with_native_execution_config, session_config::DataFusionNexusSessionConfig,
};

#[cfg(feature = "cugraph")]
#[derive(Debug)]
pub struct CugraphAlgoTableProvider {
    spec: CugraphAlgorithmSpec,
    native_execution_config: NexusNativeExecutionConfig,
}

#[cfg(feature = "cugraph")]
impl CugraphAlgoTableProvider {
    pub fn new(spec: CugraphAlgorithmSpec, native_execution_config: NexusNativeExecutionConfig) -> Self {
        Self { spec, native_execution_config }
    }

    pub fn spec(&self) -> &CugraphAlgorithmSpec {
        &self.spec
    }
}

#[cfg(feature = "cugraph")]
#[async_trait::async_trait]
impl TableProvider for CugraphAlgoTableProvider {
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
        let mut spec = self.spec.clone();
        let table_ref = datafusion_table_reference(&spec.table_name);
        let schema = session_state.schema_for_ref(table_ref.clone())?;
        let provider = schema
            .table(table_ref.table())
            .await?
            .ok_or_else(|| plan_error(format!("No table named '{}' for cuGraph edge source", spec.table_name)))?;
        let vertex_domain = validate_edge_source_schema(&spec, &provider.schema())?;
        resolve_vertex_output_schema(&mut spec, &vertex_domain);
        let native_execution_config =
            native_execution_config_with_request_options(session_state, self.native_execution_config.clone());
        // This nested state lowers the edge/predicate relations only.  The
        // outer cuGraph root owns admission after footer metadata is hydrated;
        // allowing this nested optimizer to bind a ticket would leave a
        // source-only claim on the parent attempt and shadow that graph ticket
        // at execution time.
        let nested_planning_config = native_execution_config.clone().without_query_admission_for_cugraph_subplan();
        let planning_state = session_state_with_native_execution_config(session_state, nested_planning_config)
            .unwrap_or_else(|| session_state.clone());
        let scan = LogicalPlanBuilder::scan(table_ref, provider_as_source(provider), None)?.build()?;
        let edge_source_plan = LogicalPlanBuilder::from(scan).project(edge_source_projection(&spec))?.build()?;
        let edge_source = planning_state.create_physical_plan(&edge_source_plan).await?;
        let predicate_sources = predicate_source_plans(&planning_state, session_state, &spec, &vertex_domain).await?;
        Ok(Arc::new(CugraphAlgoExec::try_new(
            spec,
            edge_source,
            predicate_sources,
            projection.cloned(),
            limit,
            native_execution_config,
        )?))
    }
}

#[cfg(feature = "cugraph")]
fn native_execution_config_with_request_options(
    session_state: &SessionState,
    native_execution_config: NexusNativeExecutionConfig,
) -> NexusNativeExecutionConfig {
    let Some(session_config) = session_state.config_options().extensions.get::<DataFusionNexusSessionConfig>() else {
        return native_execution_config;
    };
    session_config.apply_graph_memory_calibration_request_options(native_execution_config)
}

/// Feature-off table provider. It retains the parser-derived output schema but
/// produces an execution boundary with no input children, so an unavailable
/// build cannot scan, stage, or materialize an edge source on the host.
#[cfg(not(feature = "cugraph"))]
#[derive(Debug)]
struct CugraphUnavailableTableProvider {
    spec: CugraphAlgorithmSpec,
}

#[cfg(not(feature = "cugraph"))]
impl CugraphUnavailableTableProvider {
    fn new(spec: CugraphAlgorithmSpec) -> Self {
        Self { spec }
    }
}

#[cfg(not(feature = "cugraph"))]
#[async_trait::async_trait]
impl TableProvider for CugraphUnavailableTableProvider {
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
        let spec = validate_cugraph_static_sources(state, &self.spec).await?;
        Ok(Arc::new(CugraphFeatureDisabledExec::new(
            spec.algorithm,
            Arc::clone(&spec.output_schema),
            projection.cloned(),
            limit,
        )?))
    }
}

#[cfg(not(feature = "cugraph"))]
#[derive(Debug)]
struct CugraphFeatureDisabledExec {
    algorithm: CugraphAlgorithm,
    cache: Arc<PlanProperties>,
}

/// Whether this plan can only terminate at the zero-child executor for an
/// unavailable cuGraph feature. DataFusion may add output wrappers above that
/// executor, but every path still ends at the same typed rejection before any
/// CPU or GPU data execution can begin.
#[cfg(not(feature = "cugraph"))]
pub(crate) fn is_feature_disabled_execution_plan(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<CugraphFeatureDisabledExec>().is_some() || {
        let children = plan.children();
        !children.is_empty() && children.into_iter().all(|child| is_feature_disabled_execution_plan(child.as_ref()))
    }
}

#[cfg(feature = "cugraph")]
pub(crate) fn is_feature_disabled_execution_plan(_plan: &dyn ExecutionPlan) -> bool {
    false
}

#[cfg(not(feature = "cugraph"))]
impl CugraphFeatureDisabledExec {
    fn new(
        algorithm: CugraphAlgorithm,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
        _limit: Option<usize>,
    ) -> Result<Self> {
        let schema = match projection {
            Some(projection) => Arc::new(Schema::new(
                projection
                    .iter()
                    .map(|index| {
                        schema.fields().get(*index).cloned().ok_or_else(|| {
                            plan_error(format!("cuGraph output projection index {index} is out of range"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )),
            None => schema,
        };
        Ok(Self {
            algorithm,
            cache: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        })
    }
}

#[cfg(not(feature = "cugraph"))]
impl DisplayAs for CugraphFeatureDisabledExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CugraphFeatureDisabledExec: algorithm={}, reason=required_feature_disabled, required_feature=cugraph",
            self.algorithm.function_name()
        )
    }
}

#[cfg(not(feature = "cugraph"))]
impl ExecutionPlan for CugraphFeatureDisabledExec {
    fn name(&self) -> &str {
        "CugraphFeatureDisabledExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(plan_error("feature-disabled cuGraph SQL execution must not have input children"))
        }
    }

    fn execute(&self, _partition: usize, _task_context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        // Flight SQL has already issued a ticket when physical execution
        // starts. Keep the feature-gate error on that ticket's result stream
        // rather than rejecting stream construction, while still performing
        // no source scan, staging, or GPU work.
        let schema = self.schema();
        let error = super::cugraph_feature_disabled(self.algorithm);
        let stream = futures::stream::once(async move { Err(error) });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
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
        #[cfg(feature = "cugraph")]
        {
            Ok(Arc::new(CugraphAlgoTableProvider::new(spec, self.config.native_execution_config.clone())))
        }
        #[cfg(not(feature = "cugraph"))]
        {
            Ok(Arc::new(CugraphUnavailableTableProvider::new(spec)))
        }
    }

    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        #[cfg(feature = "cugraph")]
        {
            let mut spec = parse_cugraph_algorithm_spec(self.algorithm, args.exprs(), &self.config)?;
            maybe_resolve_vertex_output_schema(&mut spec, args.session());
            Ok(Arc::new(CugraphAlgoTableProvider::new(spec, self.config.native_execution_config.clone())))
        }
        #[cfg(not(feature = "cugraph"))]
        {
            let spec = parse_cugraph_algorithm_spec(self.algorithm, args.exprs(), &self.config)?;
            Ok(Arc::new(CugraphUnavailableTableProvider::new(spec)))
        }
    }
}

#[cfg(feature = "cugraph")]
fn maybe_resolve_vertex_output_schema(spec: &mut CugraphAlgorithmSpec, session: &dyn Session) {
    let Some(session_state) = session.as_any().downcast_ref::<SessionState>() else {
        return;
    };
    let table_ref = datafusion_table_reference(&spec.table_name);
    let Ok(schema) = session_state.schema_for_ref(table_ref.clone()) else {
        return;
    };
    let Ok(Some(provider)) = futures::executor::block_on(schema.table(table_ref.table())) else {
        return;
    };
    let Ok(vertex_domain) = validate_edge_source_schema(spec, &provider.schema()) else {
        return;
    };
    resolve_vertex_output_schema(spec, &vertex_domain);
}

#[cfg(feature = "cugraph")]
fn resolve_vertex_output_schema(spec: &mut CugraphAlgorithmSpec, vertex_domain: &CugraphVertexIdDomain) {
    spec.output_schema = result_schema_for_vertex_domain(spec.algorithm, &spec.params, Some(vertex_domain));
}

pub fn cugraph_table_functions(config: CugraphSqlConfig) -> Vec<Arc<TableFunction>> {
    CugraphAlgorithm::all()
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

/// Resolve only catalog schemas needed by a cuGraph call.
///
/// This deliberately stops before logical or physical child planning. The
/// feature-off provider uses it to preserve deterministic request diagnostics
/// while its zero-child execution node prevents source execution and native
/// graph preparation.
#[cfg(not(feature = "cugraph"))]
async fn validate_cugraph_static_sources(
    state: &dyn Session,
    original_spec: &CugraphAlgorithmSpec,
) -> Result<CugraphAlgorithmSpec> {
    let session_state = state.as_any().downcast_ref::<SessionState>().ok_or_else(|| {
        plan_error("cuGraph SQL table functions require a DataFusion SessionState for catalog resolution")
    })?;
    let mut spec = original_spec.clone();
    let table_ref = datafusion_table_reference(&spec.table_name);
    let schema = session_state.schema_for_ref(table_ref.clone())?;
    let provider = schema
        .table(table_ref.table())
        .await?
        .ok_or_else(|| plan_error(format!("No table named '{}' for cuGraph edge source", spec.table_name)))?;
    let vertex_domain = validate_edge_source_schema(&spec, &provider.schema())?;
    spec.output_schema = result_schema_for_vertex_domain(spec.algorithm, &spec.params, Some(&vertex_domain));
    for source in predicate_source_specs(&spec) {
        let table_ref = datafusion_table_reference(&source.table_name);
        let schema = session_state.schema_for_ref(table_ref.clone())?;
        let provider = schema.table(table_ref.table()).await?.ok_or_else(|| {
            plan_error(format!("No table named '{}' for cuGraph side-input source", source.table_name))
        })?;
        validate_predicate_source_schema(&source, &provider.schema(), &vertex_domain)?;
    }
    Ok(spec)
}

#[cfg(feature = "cugraph")]
fn exact_column_expr(name: &str) -> Expr {
    col(format!("\"{name}\""))
}

#[cfg(feature = "cugraph")]
fn edge_source_projection(spec: &CugraphAlgorithmSpec) -> Vec<Expr> {
    let mut projection = vec![exact_column_expr(&spec.src_col), exact_column_expr(&spec.dst_col)];
    if let Some(weight) = &spec.weight_col {
        projection.push(exact_column_expr(weight));
    }
    if let Some(edge_id) = &spec.edge_id_col {
        projection.push(exact_column_expr(edge_id));
    }
    projection
}

#[cfg(feature = "cugraph")]
async fn predicate_source_plans(
    planning_state: &SessionState,
    session_state: &SessionState,
    spec: &CugraphAlgorithmSpec,
    vertex_domain: &CugraphVertexIdDomain,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let mut plans = Vec::new();
    if let Some(plan) = literal_bfs_source_plan(planning_state, spec, vertex_domain).await? {
        plans.push(plan);
    }
    for source in predicate_source_specs(spec) {
        let table_ref = datafusion_table_reference(&source.table_name);
        let schema = session_state.schema_for_ref(table_ref.clone())?;
        let provider = schema.table(table_ref.table()).await?.ok_or_else(|| {
            plan_error(format!("No table named '{}' for cuGraph side-input source", source.table_name))
        })?;
        validate_predicate_source_schema(&source, &provider.schema(), vertex_domain)?;
        let scan = LogicalPlanBuilder::scan(table_ref, provider_as_source(provider), None)?.build()?;
        let projection = source.columns.iter().map(|column| exact_column_expr(&column.name)).collect::<Vec<_>>();
        let plan = LogicalPlanBuilder::from(scan).project(projection)?.build()?;
        plans.push(planning_state.create_physical_plan(&plan).await?);
    }
    Ok(plans)
}

#[cfg(feature = "cugraph")]
async fn literal_bfs_source_plan(
    state: &dyn Session,
    spec: &CugraphAlgorithmSpec,
    vertex_domain: &CugraphVertexIdDomain,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    let CugraphBfsSource::LiteralList(values) = &params.source else {
        return Ok(None);
    };
    if !values.first().is_some_and(|value| matches!(value, CugraphBfsSourceVertex::Int64(_))) {
        return Ok(None);
    }
    let values = values
        .iter()
        .map(|value| match value {
            CugraphBfsSourceVertex::Int64(value) => Ok(*value),
            CugraphBfsSourceVertex::Utf8(_) => Err(plan_error("numeric BFS source list contained a string value")),
        })
        .collect::<Result<Vec<_>>>()?;
    let (data_type, array): (DataType, ArrayRef) = match vertex_domain {
        CugraphVertexIdDomain::Int32 => {
            let values = values
                .into_iter()
                .map(|value| {
                    i32::try_from(value)
                        .map_err(|_| plan_error(format!("BFS source vertex {value} does not fit Int32 edge domain")))
                })
                .collect::<Result<Vec<_>>>()?;
            (DataType::Int32, Arc::new(Int32Array::from(values)) as ArrayRef)
        },
        CugraphVertexIdDomain::Int64 => (DataType::Int64, Arc::new(Int64Array::from(values)) as ArrayRef),
        CugraphVertexIdDomain::StringLogical => return Ok(None),
    };
    let schema = Arc::new(Schema::new(vec![Field::new("__source_vertex", data_type, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array])?;
    let table = MemTable::try_new(schema, vec![vec![batch]])?;
    table.scan(state, None, &[], None).await.map(Some)
}

#[derive(Debug, Clone)]
struct PredicateSourceSpec {
    table_name: CugraphTableReference,
    columns: Vec<PredicateSourceColumn>,
    validation: PredicateSourceValidation,
}

#[derive(Debug, Clone)]
enum PredicateSourceValidation {
    Standard,
    SimilarityVertexPairs(CugraphVertexPairTable),
}

#[derive(Debug, Clone)]
struct PredicateSourceColumn {
    name: String,
    role: PredicateSourceColumnRole,
}

#[derive(Debug, Clone, Copy)]
enum PredicateSourceColumnRole {
    Vertex,
    FloatValue,
}

impl PredicateSourceColumn {
    fn vertex(name: impl Into<String>) -> Self {
        Self { name: name.into(), role: PredicateSourceColumnRole::Vertex }
    }

    fn float_value(name: impl Into<String>) -> Self {
        Self { name: name.into(), role: PredicateSourceColumnRole::FloatValue }
    }
}

fn predicate_source_specs(spec: &CugraphAlgorithmSpec) -> Vec<PredicateSourceSpec> {
    let mut sources = Vec::new();
    match &spec.params {
        CugraphAlgorithmParams::Bfs(params) => {
            if let Some(input) = params.source.relation() {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
                    validation: PredicateSourceValidation::Standard,
                });
            }
            if let Some(input) = &params.predicates.include_vertices {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
                    validation: PredicateSourceValidation::Standard,
                });
            }
            if let Some(input) = &params.predicates.exclude_vertices {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
                    validation: PredicateSourceValidation::Standard,
                });
            }
            if let Some(input) = &params.predicates.target_vertices {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
                    validation: PredicateSourceValidation::Standard,
                });
            }
            if let Some(input) = &params.predicates.include_edge_ids {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
                    validation: PredicateSourceValidation::Standard,
                });
            }
            if let Some(input) = &params.predicates.include_edges {
                let mut columns = vec![
                    PredicateSourceColumn::vertex(input.source_col.clone()),
                    PredicateSourceColumn::vertex(input.destination_col.clone()),
                ];
                if let Some(edge_id) = &input.edge_id_col
                    && !columns.iter().any(|column| column.name == *edge_id)
                {
                    columns.push(PredicateSourceColumn::vertex(edge_id.clone()));
                }
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns,
                    validation: PredicateSourceValidation::Standard,
                });
            }
        },
        CugraphAlgorithmParams::PersonalizedPageRank(params) => {
            sources.push(PredicateSourceSpec {
                table_name: params.personalization.table_name.clone(),
                columns: vec![
                    PredicateSourceColumn::vertex(params.personalization.vertex_col.clone()),
                    PredicateSourceColumn::float_value(params.personalization.value_col.clone()),
                ],
                validation: PredicateSourceValidation::Standard,
            });
        },
        CugraphAlgorithmParams::Similarity(params) => {
            let vertex_pairs = params.vertex_pairs.clone();
            sources.push(PredicateSourceSpec {
                table_name: vertex_pairs.table_name.clone(),
                columns: vec![
                    PredicateSourceColumn::vertex(vertex_pairs.first_vertex_col.clone()),
                    PredicateSourceColumn::vertex(vertex_pairs.second_vertex_col.clone()),
                ],
                validation: PredicateSourceValidation::SimilarityVertexPairs(vertex_pairs),
            });
        },
        _ => {},
    }
    sources
}

fn validate_edge_source_schema(spec: &CugraphAlgorithmSpec, schema: &SchemaRef) -> Result<CugraphVertexIdDomain> {
    let src = schema.field_with_name(&spec.src_col).map_err(|_| {
        DataFusionError::from(crate::error::Error::new(
            crate::error::ErrorCode::FieldNotFound,
            "datafusion_nexus.cugraph_sql.edge_source_schema",
            format!("cuGraph edge source '{}' is missing source column '{}'", spec.table_name, spec.src_col),
        ))
    })?;
    let dst = schema.field_with_name(&spec.dst_col).map_err(|_| {
        DataFusionError::from(crate::error::Error::new(
            crate::error::ErrorCode::FieldNotFound,
            "datafusion_nexus.cugraph_sql.edge_source_schema",
            format!("cuGraph edge source '{}' is missing destination column '{}'", spec.table_name, spec.dst_col),
        ))
    })?;
    let domain = vertex_id_domain_for_call(spec.algorithm, &spec.params, src.data_type(), dst.data_type())?;
    validate_vertex_id_domain_support_for_call(spec.algorithm, &domain)?;
    if let Some(edge_id_col) = &spec.edge_id_col {
        let edge_id = schema.field_with_name(edge_id_col).map_err(|_| {
            DataFusionError::from(crate::error::Error::new(
                crate::error::ErrorCode::FieldNotFound,
                "datafusion_nexus.cugraph_sql.edge_source_schema",
                format!("cuGraph edge source '{}' is missing edge-id column '{edge_id_col}'", spec.table_name),
            ))
        })?;
        if !domain.accepts(edge_id.data_type()) {
            return Err(plan_error(format!(
                "cuGraph edge-id column '{edge_id_col}' must have dtype {}",
                domain.label()
            )));
        }
    }
    super::validate_bfs_logical_edge_id_capability(spec, domain.is_string_logical())?;
    Ok(domain)
}

fn validate_predicate_source_schema(
    source: &PredicateSourceSpec,
    schema: &SchemaRef,
    vertex_domain: &CugraphVertexIdDomain,
) -> Result<()> {
    if let PredicateSourceValidation::SimilarityVertexPairs(vertex_pairs) = &source.validation {
        return validate_similarity_pair_table_schema(vertex_pairs, schema, vertex_domain);
    }
    for column in &source.columns {
        let field = schema.field_with_name(&column.name).map_err(|_| {
            plan_error(format!("cuGraph side-input source '{}' is missing column '{}'", source.table_name, column.name))
        })?;
        match column.role {
            PredicateSourceColumnRole::Vertex => {
                if !vertex_domain.accepts(field.data_type()) {
                    return Err(plan_error(format!(
                        "cuGraph side-input vertex column '{}' in '{}' must have dtype {}",
                        column.name,
                        source.table_name,
                        vertex_domain.label()
                    )));
                }
            },
            PredicateSourceColumnRole::FloatValue => {
                if !matches!(field.data_type(), DataType::Float32 | DataType::Float64) {
                    return Err(plan_error(format!(
                        "cuGraph side-input value column '{}' in '{}' must have dtype Float32 or Float64",
                        column.name, source.table_name
                    )));
                }
            },
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cugraph")]
    use datafusion::execution::SessionStateBuilder;

    #[cfg(feature = "cugraph")]
    use super::*;

    #[cfg(feature = "cugraph")]
    #[test]
    fn native_execution_config_with_request_options_reads_graph_calibration() {
        let mut builder = SessionStateBuilder::new();
        let session_config = builder.config().get_or_insert_with(Default::default);
        let mut nexus_config = DataFusionNexusSessionConfig::default();
        nexus_config.set_request_graph_raw_input_edges(Some(45_564_149));
        session_config.options_mut().extensions.insert(nexus_config);
        let state = builder.build();

        let execution_config = native_execution_config_with_request_options(
            &state,
            NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1),
        );

        let graph_calibration = execution_config.graph_memory_calibration_override();
        assert_eq!(graph_calibration.raw_input_edges, Some(45_564_149));
    }
}
