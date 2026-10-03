use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use query_planning::plan::{
    MlAlgorithm, MlAlgorithmParams, MlAlgorithmSpec, MlForestModelSnapshot, MlForestPredictParams, QueryPlan,
    QueryPlanBuilder,
};

use crate::{
    cuml_sql::{CumlOperation, CumlValidatedOptions, error::invalid_call, options::CumlParams},
    error::{Error, Result},
    native_exec::{child_inputs::NativeChildBinding, relation_function::add_projected_relation_source},
};

pub(super) fn lower_cuml_query_plan(
    operation: CumlOperation,
    options: &CumlValidatedOptions,
    model: Option<&MlForestModelSnapshot>,
    children: &[NativeChildBinding],
    output_schema: SchemaRef,
) -> Result<QueryPlan> {
    if children.len() != options.bindings().len() {
        return Err(invalid_call(
            operation,
            "physical_child_count_mismatch",
            "cuML physical child count does not match its validated relation bindings",
        ));
    }
    let mut builder = QueryPlanBuilder::new();
    let mut inputs = Vec::with_capacity(children.len());
    for (child, binding) in children.iter().zip(options.bindings()) {
        if child.role != binding.relation_role().as_str() {
            return Err(invalid_call(
                operation,
                "physical_child_role_mismatch",
                "cuML physical child order does not match semantic relation order",
            ));
        }
        inputs.push(add_projected_relation_source(&mut builder, child).map_err(|error| {
            Error::from_native("algeon_datafusion.cuml_sql.lower", "failed to lower cuML SQL into native ML IR", error)
        })?);
    }
    let params = match options.params() {
        CumlParams::Algorithm(params) => params.clone(),
        CumlParams::ForestPredict { input, .. } => {
            let Some(model) = model else {
                return Err(invalid_call(
                    operation,
                    "model_unbound",
                    "cuml_forest_predict requires a model bound during planning",
                ));
            };
            MlAlgorithmParams::ForestPredict(MlForestPredictParams { input: input.clone(), model: model.clone() })
        },
    };
    let algorithm = match operation {
        CumlOperation::Tsvd => MlAlgorithm::Tsvd,
        CumlOperation::Hdbscan => MlAlgorithm::Hdbscan,
        CumlOperation::LogisticRegression => MlAlgorithm::LogisticRegression,
        CumlOperation::RandomForestClassifier => MlAlgorithm::RandomForestClassifier,
        CumlOperation::RandomForestRegressor => MlAlgorithm::RandomForestRegressor,
        CumlOperation::ForestPredict => MlAlgorithm::ForestPredict,
        CumlOperation::Umap => MlAlgorithm::Umap,
    };
    let root = builder
        .add_ml_algorithm(inputs, MlAlgorithmSpec::new(algorithm, params, Arc::clone(&output_schema)))
        .map_err(|error| {
            Error::from_native("algeon_datafusion.cuml_sql.lower", "failed to lower cuML SQL into native ML IR", error)
        })?;
    builder.finish_with_host_sink_schema(root, output_schema).map_err(|error| {
        Error::from_native("algeon_datafusion.cuml_sql.lower", "failed to lower cuML SQL into native ML IR", error)
    })
}
