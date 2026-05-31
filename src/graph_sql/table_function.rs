use std::{any::Any, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::{Result, TableReference},
    datasource::{TableType, provider_as_source},
    execution::{SessionState, context::SessionContext},
    logical_expr::{Expr, LogicalPlanBuilder, col},
    physical_plan::ExecutionPlan,
};

use super::{
    CUGRAPH_BFS_PATH_FUNCTION, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphSqlConfig,
    CugraphTableReference, CugraphVertexIdDomain, parse_cugraph_algorithm_spec, parse_cugraph_bfs_path_spec,
    plan_error, vertex_id_domain_for_call,
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

#[derive(Debug)]
struct CugraphBfsPathTableFunction {
    config: CugraphSqlConfig,
}

impl TableFunctionImpl for CugraphBfsPathTableFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        let spec = parse_cugraph_bfs_path_spec(args, &self.config)?;
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
    functions.push(Arc::new(TableFunction::new(
        CUGRAPH_BFS_PATH_FUNCTION.to_string(),
        Arc::new(CugraphBfsPathTableFunction { config: config.clone() }),
    )));
    functions.extend(cugraph_metadata_table_functions(config));
    functions
}

pub fn register_cugraph_udtfs(ctx: &SessionContext, config: CugraphSqlConfig) {
    for function in cugraph_table_functions(config) {
        ctx.register_udtf(function.name(), Arc::clone(function.function()));
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
            plan_error(format!("No table named '{}' for cuGraph BFS predicate source", source.table_name))
        })?;
        validate_predicate_source_schema(&source, &provider.schema(), vertex_domain)?;
        let scan = LogicalPlanBuilder::scan(table_ref, provider_as_source(provider), None)?.build()?;
        let projection = source.columns.iter().map(col).collect::<Vec<_>>();
        let plan = LogicalPlanBuilder::from(scan).project(projection)?.build()?;
        plans.push(state.create_physical_plan(&plan).await?);
    }
    Ok(plans)
}

#[derive(Debug, Clone)]
struct PredicateSourceSpec {
    table_name: CugraphTableReference,
    columns: Vec<String>,
}

fn predicate_source_specs(spec: &CugraphAlgorithmSpec) -> Vec<PredicateSourceSpec> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Vec::new();
    };
    let mut sources = Vec::new();
    if let Some(input) = &params.predicates.include_vertices {
        sources.push(PredicateSourceSpec { table_name: input.table_name.clone(), columns: vec![input.column.clone()] });
    }
    if let Some(input) = &params.predicates.exclude_vertices {
        sources.push(PredicateSourceSpec { table_name: input.table_name.clone(), columns: vec![input.column.clone()] });
    }
    if let Some(input) = &params.predicates.target_vertices {
        sources.push(PredicateSourceSpec { table_name: input.table_name.clone(), columns: vec![input.column.clone()] });
    }
    if let Some(input) = &params.predicates.include_edge_ids {
        sources.push(PredicateSourceSpec { table_name: input.table_name.clone(), columns: vec![input.column.clone()] });
    }
    if let Some(input) = &params.predicates.include_edges {
        let mut columns = vec![input.source_col.clone(), input.destination_col.clone()];
        if let Some(edge_id) = &input.edge_id_col
            && !columns.iter().any(|column| column == edge_id)
        {
            columns.push(edge_id.clone());
        }
        sources.push(PredicateSourceSpec { table_name: input.table_name.clone(), columns });
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
        let field = schema.field_with_name(column).map_err(|_| {
            plan_error(format!("cuGraph BFS predicate source '{}' is missing column '{column}'", source.table_name))
        })?;
        if field.is_nullable() {
            return Err(plan_error(format!(
                "cuGraph BFS predicate column '{column}' in '{}' must be non-null",
                source.table_name
            )));
        }
        if !vertex_domain.accepts(field.data_type()) {
            return Err(plan_error(format!(
                "cuGraph BFS predicate column '{column}' in '{}' must have dtype {}",
                source.table_name,
                vertex_domain.label()
            )));
        }
    }
    Ok(())
}
