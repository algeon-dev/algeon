use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::Result,
    datasource::{TableType, provider_as_source},
    execution::{SessionState, context::SessionContext},
    logical_expr::{Expr, LogicalPlanBuilder, col},
    physical_plan::ExecutionPlan,
};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphSqlConfig, CugraphTableReference,
    CugraphVertexIdDomain, datafusion_table_reference, parse_cugraph_algorithm_spec, plan_error,
    vertex_id_domain_for_call,
};
use crate::{
    NativeCudfExecutionConfig,
    graph_sql::{CugraphAlgoExec, metadata::cugraph_metadata_table_functions},
};

#[derive(Debug)]
pub struct CugraphAlgoTableProvider {
    spec: CugraphAlgorithmSpec,
    native_execution_config: NativeCudfExecutionConfig,
}

impl CugraphAlgoTableProvider {
    pub fn new(spec: CugraphAlgorithmSpec, native_execution_config: NativeCudfExecutionConfig) -> Self {
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
        let table_ref = datafusion_table_reference(&self.spec.table_name);
        let schema = session_state.schema_for_ref(table_ref.clone())?;
        let provider = schema
            .table(table_ref.table())
            .await?
            .ok_or_else(|| plan_error(format!("No table named '{}' for cuGraph edge source", self.spec.table_name)))?;
        let vertex_domain = validate_edge_source_schema(&self.spec, &provider.schema())?;
        let scan = LogicalPlanBuilder::scan(table_ref, provider_as_source(provider), None)?.build()?;
        let edge_source_plan = LogicalPlanBuilder::from(scan).project(edge_source_projection(&self.spec))?.build()?;
        let edge_source = state.create_physical_plan(&edge_source_plan).await?;
        let predicate_sources = predicate_source_plans(state, session_state, &self.spec, &vertex_domain).await?;
        Ok(Arc::new(CugraphAlgoExec::try_new(
            self.spec.clone(),
            edge_source,
            predicate_sources,
            projection.cloned(),
            limit,
            self.native_execution_config.clone(),
        )?))
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
        Ok(Arc::new(CugraphAlgoTableProvider::new(spec, self.config.native_execution_config.clone())))
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
    state: &dyn Session,
    session_state: &SessionState,
    spec: &CugraphAlgorithmSpec,
    vertex_domain: &CugraphVertexIdDomain,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let mut plans = Vec::new();
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
        plans.push(state.create_physical_plan(&plan).await?);
    }
    Ok(plans)
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
