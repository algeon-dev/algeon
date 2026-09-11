use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result};
use nexus_query_engine::{
    expr::Expr as NativeExpr,
    plan::{
        ProjectionOutput, QueryPlan, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec,
        VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansInit, VectorKMeansMetric, VectorKMeansParams,
        VectorPcaParams, VectorPcaSolver, VectorRelationBinding, VectorRowIdBinding,
    },
    source::{GpuResidentSource, SourceNode},
};

use super::super::{CuvsDenseVectorBinding, CuvsOperation, CuvsValidatedOptions, error::invalid_call};
use crate::{error::Error, native_exec::child_inputs::NativeChildBinding};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CuvsPlanSink {
    HostArrow,
    DeferredHostArrow,
    GpuResident,
}

pub(super) fn lower_cuvs_query_plan(
    operation: CuvsOperation,
    options: &CuvsValidatedOptions,
    children: &[NativeChildBinding],
    output_schema: SchemaRef,
    sink: CuvsPlanSink,
) -> Result<QueryPlan> {
    if children.len() != options.bindings().len() {
        return Err(invalid_call(
            operation,
            "physical_child_count_mismatch",
            "cuVS physical child count does not match its validated relation bindings",
        ));
    }

    let mut builder = QueryPlanBuilder::new();
    let mut relations = Vec::with_capacity(children.len());
    for (child, binding) in children.iter().zip(options.bindings()) {
        if child.role != binding.relation_role().as_str() {
            return Err(invalid_call(
                operation,
                "physical_child_role_mismatch",
                "cuVS physical child order does not match semantic relation order",
            ));
        }
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            child.input_name.clone(),
            Arc::clone(&child.schema),
        )));
        let input = if is_identity_projection(&child.projection, child.schema.fields().len()) {
            source
        } else {
            let outputs = child
                .projection
                .iter()
                .map(|index| {
                    let field = child.schema.field(*index);
                    ProjectionOutput::new(NativeExpr::column(field.name()), field.as_ref().clone())
                })
                .collect();
            builder.add_projection(source, outputs)
        };
        let relation = match binding.vector() {
            CuvsDenseVectorBinding::Columns(columns) => {
                VectorRelationBinding::new(input, VectorRowIdBinding::column(binding.id()), columns.iter().cloned())
            },
            CuvsDenseVectorBinding::VectorColumn(column) => VectorRelationBinding::new_vector_column(
                input,
                VectorRowIdBinding::column(binding.id()),
                column.clone(),
            ),
        };
        relations.push(relation);
    }

    let canonical: serde_json::Value = serde_json::from_str(options.canonical_json()).map_err(|error| {
        DataFusionError::from(
            crate::error::Error::new(
                crate::error::ErrorCode::Internal,
                "datafusion_nexus.cuvs_sql.lower",
                "validated cuVS options were not valid canonical JSON",
            )
            .with_source(error),
        )
    })?;
    let params = match operation {
        CuvsOperation::BruteForceKnn => VectorAlgorithmParams::BruteForceKnn(VectorBruteForceKnnParams::new(
            relations[0].clone(),
            relations[1].clone(),
            value_usize(operation, &canonical, "k")?,
            match value_str(operation, &canonical, "metric")? {
                "l2_expanded" => VectorDistanceMetric::L2Expanded,
                "l2_sqrt_expanded" => VectorDistanceMetric::L2SqrtExpanded,
                "cosine" => VectorDistanceMetric::CosineExpanded,
                "inner_product" => VectorDistanceMetric::InnerProduct,
                _ => unreachable!("canonical kNN metric is registry validated"),
            },
        )),
        CuvsOperation::KMeans => {
            let mut params =
                VectorKMeansParams::new(relations[0].clone(), value_usize(operation, &canonical, "n_clusters")?);
            params.metric = match value_str(operation, &canonical, "metric")? {
                "l2_expanded" => VectorKMeansMetric::L2Expanded,
                "l2_sqrt_expanded" => VectorKMeansMetric::L2SqrtExpanded,
                _ => unreachable!("canonical KMeans metric is registry validated"),
            };
            params.max_iterations = value_u32(operation, &canonical, "max_iter")?;
            params.tolerance = value_f64(operation, &canonical, "tol")?;
            params.n_init = value_u32(operation, &canonical, "n_init")?;
            params.init = match value_str(operation, &canonical, "init")? {
                "kmeans++" => VectorKMeansInit::KMeansPlusPlus,
                "random" => VectorKMeansInit::Random,
                _ => unreachable!("canonical KMeans init is registry validated"),
            };
            VectorAlgorithmParams::KMeans(params)
        },
        CuvsOperation::Pca => {
            let mut params =
                VectorPcaParams::new(relations[0].clone(), value_usize(operation, &canonical, "n_components")?);
            params.whiten = value_bool(operation, &canonical, "whiten")?;
            params.solver = match value_str(operation, &canonical, "solver")? {
                "cov_eig_dq" => VectorPcaSolver::CovEigDq,
                "cov_eig_jacobi" => VectorPcaSolver::CovEigJacobi,
                _ => unreachable!("canonical PCA solver is registry validated"),
            };
            params.tolerance = value_f64(operation, &canonical, "tol")?;
            params.n_iterations = value_u32(operation, &canonical, "n_iterations")?;
            params.flip_signs_based_on_u = value_bool(operation, &canonical, "flip_signs_based_on_u")?;
            VectorAlgorithmParams::Pca(params)
        },
    };
    let algorithm = match operation {
        CuvsOperation::BruteForceKnn => VectorAlgorithm::BruteForceKnn,
        CuvsOperation::KMeans => VectorAlgorithm::KMeans,
        CuvsOperation::Pca => VectorAlgorithm::Pca,
    };
    let root = builder
        .try_add_vector_algorithm(VectorAlgorithmSpec::new(algorithm, params, Arc::clone(&output_schema)))
        .map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cuvs_sql.lower",
                "failed to lower cuVS SQL into native vector IR",
                error,
            ))
        })?;
    Ok(match sink {
        CuvsPlanSink::HostArrow => builder.finish_with_host_sink_schema(root, output_schema),
        CuvsPlanSink::DeferredHostArrow => {
            builder.finish_with_host_sink_schema(root, output_schema).clone_with_deferred_host_materialization_sink()
        },
        CuvsPlanSink::GpuResident => builder.finish_gpu_resident(root),
    })
}

fn is_identity_projection(projection: &[usize], field_count: usize) -> bool {
    projection.len() == field_count && projection.iter().copied().eq(0..field_count)
}

fn value<'a>(
    operation: CuvsOperation,
    canonical: &'a serde_json::Value,
    name: &'static str,
) -> Result<&'a serde_json::Value> {
    canonical.get(name).ok_or_else(|| {
        invalid_call(operation, "canonical_option_missing", format!("validated cuVS options are missing '{name}'"))
    })
}

fn value_str<'a>(operation: CuvsOperation, canonical: &'a serde_json::Value, name: &'static str) -> Result<&'a str> {
    value(operation, canonical, name)?.as_str().ok_or_else(|| {
        invalid_call(
            operation,
            "canonical_option_type_mismatch",
            format!("validated cuVS option '{name}' is not a string"),
        )
    })
}

fn value_usize(operation: CuvsOperation, canonical: &serde_json::Value, name: &'static str) -> Result<usize> {
    let raw = value(operation, canonical, name)?.as_u64().ok_or_else(|| {
        invalid_call(
            operation,
            "canonical_option_type_mismatch",
            format!("validated cuVS option '{name}' is not an integer"),
        )
    })?;
    usize::try_from(raw).map_err(|_| {
        invalid_call(operation, "canonical_option_overflow", format!("validated cuVS option '{name}' exceeds usize"))
    })
}

fn value_u32(operation: CuvsOperation, canonical: &serde_json::Value, name: &'static str) -> Result<u32> {
    let raw = value(operation, canonical, name)?.as_u64().ok_or_else(|| {
        invalid_call(
            operation,
            "canonical_option_type_mismatch",
            format!("validated cuVS option '{name}' is not an integer"),
        )
    })?;
    u32::try_from(raw).map_err(|_| {
        invalid_call(operation, "canonical_option_overflow", format!("validated cuVS option '{name}' exceeds u32"))
    })
}

fn value_f64(operation: CuvsOperation, canonical: &serde_json::Value, name: &'static str) -> Result<f64> {
    value(operation, canonical, name)?.as_f64().ok_or_else(|| {
        invalid_call(
            operation,
            "canonical_option_type_mismatch",
            format!("validated cuVS option '{name}' is not numeric"),
        )
    })
}

fn value_bool(operation: CuvsOperation, canonical: &serde_json::Value, name: &'static str) -> Result<bool> {
    value(operation, canonical, name)?.as_bool().ok_or_else(|| {
        invalid_call(
            operation,
            "canonical_option_type_mismatch",
            format!("validated cuVS option '{name}' is not boolean"),
        )
    })
}
