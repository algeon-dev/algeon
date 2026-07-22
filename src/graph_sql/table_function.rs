use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionArgs, TableFunctionImpl, TableProvider},
    common::Result,
    datasource::{MemTable, TableType, provider_as_source},
    execution::{SessionState, context::SessionContext},
    logical_expr::{Expr, LogicalPlanBuilder, col},
    physical_plan::ExecutionPlan,
};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsSource, CugraphBfsSourceVertex,
    CugraphSqlConfig, CugraphTableReference, CugraphVertexIdDomain, bfs_result_schema_for_vertex_domain,
    datafusion_table_reference, parse_cugraph_algorithm_spec, plan_error, validate_vertex_id_domain_support_for_call,
    vertex_id_domain_for_call,
};
use crate::{
    graph_sql::{CugraphAlgoExec, metadata::cugraph_metadata_table_functions},
    native_exec::NexusNativeExecutionConfig,
    session::session_state_with_native_execution_config,
    session_config::DataFusionNexusSessionConfig,
};

#[derive(Debug)]
pub struct CugraphAlgoTableProvider {
    spec: CugraphAlgorithmSpec,
    native_execution_config: NexusNativeExecutionConfig,
}

impl CugraphAlgoTableProvider {
    pub fn new(spec: CugraphAlgorithmSpec, native_execution_config: NexusNativeExecutionConfig) -> Self {
        Self { spec, native_execution_config }
    }

    pub fn spec(&self) -> &CugraphAlgorithmSpec {
        &self.spec
    }
}

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
        if let CugraphAlgorithmParams::Bfs(params) = &spec.params {
            spec.output_schema = bfs_result_schema_for_vertex_domain(params, &vertex_domain);
        }
        let native_execution_config =
            native_execution_config_with_request_options(session_state, self.native_execution_config.clone());
        let planning_state = session_state_with_native_execution_config(session_state, native_execution_config.clone())
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

fn native_execution_config_with_request_options(
    session_state: &SessionState,
    native_execution_config: NexusNativeExecutionConfig,
) -> NexusNativeExecutionConfig {
    let Some(session_config) = session_state.config_options().extensions.get::<DataFusionNexusSessionConfig>() else {
        return native_execution_config;
    };
    let native_execution_config = native_execution_config
        .with_requested_grant_min_bytes(session_config.request_requested_grant_min_bytes)
        .with_requested_grant_bytes(session_config.request_requested_grant_bytes);
    session_config.apply_graph_memory_calibration_request_options(native_execution_config)
}

#[derive(Debug)]
struct CugraphTableFunction {
    algorithm: CugraphAlgorithm,
    config: CugraphSqlConfig,
}

impl TableFunctionImpl for CugraphTableFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        let spec = parse_cugraph_algorithm_spec(self.algorithm, args, &self.config)?;
        Ok(Arc::new(CugraphAlgoTableProvider::new(spec, self.config.native_execution_config.clone())))
    }

    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let mut spec = parse_cugraph_algorithm_spec(self.algorithm, args.exprs(), &self.config)?;
        maybe_resolve_bfs_relation_output_schema(&mut spec, args.session());
        Ok(Arc::new(CugraphAlgoTableProvider::new(spec, self.config.native_execution_config.clone())))
    }
}

fn maybe_resolve_bfs_relation_output_schema(spec: &mut CugraphAlgorithmSpec, session: &dyn Session) {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return;
    };
    if !matches!(params.source, CugraphBfsSource::Relation(_)) {
        return;
    }
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
    if let CugraphAlgorithmParams::Bfs(params) = &spec.params {
        spec.output_schema = bfs_result_schema_for_vertex_domain(params, &vertex_domain);
    }
}

pub fn cugraph_table_functions(config: CugraphSqlConfig) -> Vec<Arc<TableFunction>> {
    let mut functions: Vec<_> = CugraphAlgorithm::all()
        .map(|algorithm| {
            Arc::new(TableFunction::new(
                algorithm.function_name().to_string(),
                Arc::new(CugraphTableFunction { algorithm, config: config.clone() }),
            ))
        })
        .collect();
    functions.extend(cugraph_metadata_table_functions(config));
    functions
}

pub fn register_cugraph_udtfs(ctx: &SessionContext, config: CugraphSqlConfig) {
    for function in cugraph_table_functions(config) {
        ctx.register_udtf(function.name(), Arc::clone(function.function()));
    }
}

fn edge_source_projection(spec: &CugraphAlgorithmSpec) -> Vec<Expr> {
    let mut projection = vec![col(&spec.src_col), col(&spec.dst_col)];
    if let Some(weight) = &spec.weight_col {
        projection.push(col(weight));
    }
    if let Some(edge_id) = &spec.edge_id_col {
        projection.push(col(edge_id));
    }
    projection
}

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
        let projection = source.columns.iter().map(|column| col(&column.name)).collect::<Vec<_>>();
        let plan = LogicalPlanBuilder::from(scan).project(projection)?.build()?;
        plans.push(planning_state.create_physical_plan(&plan).await?);
    }
    Ok(plans)
}

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
                });
            }
            if let Some(input) = &params.predicates.include_vertices {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
                });
            }
            if let Some(input) = &params.predicates.exclude_vertices {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
                });
            }
            if let Some(input) = &params.predicates.target_vertices {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
                });
            }
            if let Some(input) = &params.predicates.include_edge_ids {
                sources.push(PredicateSourceSpec {
                    table_name: input.table_name.clone(),
                    columns: vec![PredicateSourceColumn::vertex(input.column.clone())],
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
                sources.push(PredicateSourceSpec { table_name: input.table_name.clone(), columns });
            }
        },
        CugraphAlgorithmParams::PersonalizedPageRank(params) => {
            sources.push(PredicateSourceSpec {
                table_name: params.personalization.table_name.clone(),
                columns: vec![
                    PredicateSourceColumn::vertex(params.personalization.vertex_col.clone()),
                    PredicateSourceColumn::float_value(params.personalization.value_col.clone()),
                ],
            });
        },
        _ => {},
    }
    sources
}

fn validate_edge_source_schema(spec: &CugraphAlgorithmSpec, schema: &SchemaRef) -> Result<CugraphVertexIdDomain> {
    let src = schema.field_with_name(&spec.src_col).map_err(|_| {
        plan_error(format!("cuGraph edge source '{}' is missing source column '{}'", spec.table_name, spec.src_col))
    })?;
    let dst = schema.field_with_name(&spec.dst_col).map_err(|_| {
        plan_error(format!(
            "cuGraph edge source '{}' is missing destination column '{}'",
            spec.table_name, spec.dst_col
        ))
    })?;
    let domain = vertex_id_domain_for_call(spec.algorithm, &spec.params, src.data_type(), dst.data_type())?;
    validate_vertex_id_domain_support_for_call(spec.algorithm, &domain)?;
    if let Some(edge_id_col) = &spec.edge_id_col {
        let edge_id = schema.field_with_name(edge_id_col).map_err(|_| {
            plan_error(format!("cuGraph edge source '{}' is missing edge-id column '{edge_id_col}'", spec.table_name))
        })?;
        if domain.is_string_logical() {
            return Err(plan_error("cuGraph BFS logical string vertex-id mapping does not yet support edge_id_col"));
        }
        if !domain.accepts(edge_id.data_type()) {
            return Err(plan_error(format!(
                "cuGraph edge-id column '{edge_id_col}' must have dtype {}",
                domain.label()
            )));
        }
    }
    Ok(domain)
}

fn validate_predicate_source_schema(
    source: &PredicateSourceSpec,
    schema: &SchemaRef,
    vertex_domain: &CugraphVertexIdDomain,
) -> Result<()> {
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
    use datafusion::execution::SessionStateBuilder;

    use super::*;

    #[test]
    fn native_execution_config_with_request_options_reads_session_request_grant() {
        let mut builder = SessionStateBuilder::new();
        let session_config = builder.config().get_or_insert_with(Default::default);
        let mut nexus_config = DataFusionNexusSessionConfig::default();
        nexus_config.request_requested_grant_min_bytes = Some(384);
        nexus_config.request_requested_grant_bytes = Some(512);
        nexus_config.set_request_graph_raw_input_edges(Some(45_564_149));
        session_config.options_mut().extensions.insert(nexus_config);
        let state = builder.build();

        let execution_config = native_execution_config_with_request_options(
            &state,
            NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1),
        );

        assert_eq!(execution_config.requested_grant_min_bytes(), Some(384));
        assert_eq!(execution_config.requested_grant_bytes(), Some(512));
        let graph_calibration = execution_config.graph_memory_calibration_override();
        assert_eq!(graph_calibration.raw_input_edges, Some(45_564_149));
    }
}
