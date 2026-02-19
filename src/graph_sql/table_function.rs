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
    CugraphAlgorithm, CugraphAlgorithmSpec, CugraphSqlConfig, CugraphTableReference, parse_cugraph_algorithm_spec,
    plan_error,
};
use crate::graph_sql::CugraphAlgoExec;

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
