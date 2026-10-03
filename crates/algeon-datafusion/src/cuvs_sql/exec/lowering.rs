use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use query_engine::plan::{QueryPlan, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmSpec};

use crate::{
    cuvs_sql::{CuvsOperation, CuvsValidatedOptions, error::invalid_call},
    error::{Error, Result},
    native_exec::{child_inputs::NativeChildBinding, relation_function::add_projected_relation_source},
};

pub(super) fn lower_cuvs_query_plan(
    operation: CuvsOperation,
    options: &CuvsValidatedOptions,
    children: &[NativeChildBinding],
    output_schema: SchemaRef,
) -> Result<QueryPlan> {
    if children.len() != options.bindings().len() {
        return Err(invalid_call(
            operation,
            "physical_child_count_mismatch",
            "cuVS physical child count does not match its validated relation bindings",
        ));
    }

    let mut builder = QueryPlanBuilder::new();
    let mut inputs = Vec::with_capacity(children.len());
    for (child, binding) in children.iter().zip(options.bindings()) {
        if child.role != binding.relation_role().as_str() {
            return Err(invalid_call(
                operation,
                "physical_child_role_mismatch",
                "cuVS physical child order does not match semantic relation order",
            ));
        }
        inputs.push(add_projected_relation_source(&mut builder, child).map_err(|error| {
            Error::from_native(
                "algeon_datafusion.cuvs_sql.lower",
                "failed to lower cuVS SQL into native vector IR",
                error,
            )
        })?);
    }
    let algorithm = match operation {
        CuvsOperation::BruteForceKnn => VectorAlgorithm::BruteForceKnn,
        CuvsOperation::IvfFlat => VectorAlgorithm::IvfFlat,
        CuvsOperation::IvfPq => VectorAlgorithm::IvfPq,
        CuvsOperation::Cagra => VectorAlgorithm::Cagra,
        CuvsOperation::PairwiseDistance => VectorAlgorithm::PairwiseDistance,
        CuvsOperation::TopK => VectorAlgorithm::TopK,
        CuvsOperation::KMeans => VectorAlgorithm::KMeans,
        CuvsOperation::Pca => VectorAlgorithm::Pca,
    };
    let root = builder
        .add_vector_algorithm(
            inputs,
            VectorAlgorithmSpec::new(algorithm, options.params().clone(), Arc::clone(&output_schema)),
        )
        .map_err(|error| {
            Error::from_native(
                "algeon_datafusion.cuvs_sql.lower",
                "failed to lower cuVS SQL into native vector IR",
                error,
            )
        })?;
    builder.finish_with_host_sink_schema(root, output_schema).map_err(|error| {
        Error::from_native("algeon_datafusion.cuvs_sql.lower", "failed to lower cuVS SQL into native vector IR", error)
    })
}
