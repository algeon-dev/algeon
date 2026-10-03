use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use query_engine::plan::{
    QueryPlan, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec,
    VectorBruteForceKnnParams, VectorCagraBuildAlgo, VectorCagraParams, VectorCagraSearchAlgo, VectorDistanceMetric,
    VectorIvfFlatParams, VectorIvfPqCodebookKind, VectorIvfPqParams, VectorKMeansInit, VectorKMeansMetric,
    VectorKMeansParams, VectorPairwiseDistanceParams, VectorPcaParams, VectorPcaSolver, VectorRelationBinding,
    VectorRowIdBinding, VectorTopKParams,
};

use crate::{
    cuvs_sql::{CuvsOperation, CuvsValidatedOptions, DenseVectorBinding, error::invalid_call},
    error::{Error, ErrorCode, Result},
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
    let mut relations = Vec::with_capacity(children.len());
    for (child, binding) in children.iter().zip(options.bindings()) {
        if child.role != binding.relation_role().as_str() {
            return Err(invalid_call(
                operation,
                "physical_child_role_mismatch",
                "cuVS physical child order does not match semantic relation order",
            ));
        }
        let input = add_projected_relation_source(&mut builder, child);
        let relation = match binding.vector() {
            DenseVectorBinding::Columns(columns) => {
                VectorRelationBinding::new(input, VectorRowIdBinding::column(binding.id()), columns.iter().cloned())
            },
            DenseVectorBinding::VectorColumn(column) => VectorRelationBinding::new_vector_column(
                input,
                VectorRowIdBinding::column(binding.id()),
                column.clone(),
            ),
        };
        relations.push(relation);
    }

    let canonical: serde_json::Value = serde_json::from_str(options.canonical_json()).map_err(|error| {
        Error::new(
            ErrorCode::Internal,
            "algeon_datafusion.cuvs_sql.lower",
            "validated cuVS options were not valid canonical JSON",
        )
        .with_source(error)
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
        CuvsOperation::IvfFlat => {
            let metric = match value_str(operation, &canonical, "metric")? {
                "l2_expanded" => VectorDistanceMetric::L2Expanded,
                "l2_sqrt_expanded" => VectorDistanceMetric::L2SqrtExpanded,
                "cosine" => VectorDistanceMetric::CosineExpanded,
                "inner_product" => VectorDistanceMetric::InnerProduct,
                _ => unreachable!("canonical IVF-Flat metric is registry validated"),
            };
            let mut params = VectorIvfFlatParams::new(
                relations[0].clone(),
                relations[1].clone(),
                value_usize(operation, &canonical, "k")?,
                metric,
                value_u32(operation, &canonical, "n_lists")?,
            );
            params.n_probes = value_u32(operation, &canonical, "n_probes")?;
            params.kmeans_n_iters = value_u32(operation, &canonical, "kmeans_n_iters")?;
            params.kmeans_trainset_fraction = value_f64(operation, &canonical, "kmeans_trainset_fraction")?;
            VectorAlgorithmParams::IvfFlat(params)
        },
        CuvsOperation::IvfPq => {
            let metric = match value_str(operation, &canonical, "metric")? {
                "l2_expanded" => VectorDistanceMetric::L2Expanded,
                "l2_sqrt_expanded" => VectorDistanceMetric::L2SqrtExpanded,
                "cosine" => VectorDistanceMetric::CosineExpanded,
                "inner_product" => VectorDistanceMetric::InnerProduct,
                _ => unreachable!("canonical IVF-PQ metric is registry validated"),
            };
            let mut params = VectorIvfPqParams::new(
                relations[0].clone(),
                relations[1].clone(),
                value_usize(operation, &canonical, "k")?,
                metric,
                value_u32(operation, &canonical, "n_lists")?,
            );
            params.n_probes = value_u32(operation, &canonical, "n_probes")?;
            params.kmeans_n_iters = value_u32(operation, &canonical, "kmeans_n_iters")?;
            params.kmeans_trainset_fraction = value_f64(operation, &canonical, "kmeans_trainset_fraction")?;
            params.pq_dim = value_u32(operation, &canonical, "pq_dim")?;
            params.pq_bits = value_u32(operation, &canonical, "pq_bits")?;
            params.codebook_kind = match value_str(operation, &canonical, "codebook_kind")? {
                "per_subspace" => VectorIvfPqCodebookKind::PerSubspace,
                "per_cluster" => VectorIvfPqCodebookKind::PerCluster,
                _ => unreachable!("canonical IVF-PQ codebook kind is registry validated"),
            };
            VectorAlgorithmParams::IvfPq(params)
        },
        CuvsOperation::Cagra => {
            let metric = match value_str(operation, &canonical, "metric")? {
                "l2_expanded" => VectorDistanceMetric::L2Expanded,
                "inner_product" => VectorDistanceMetric::InnerProduct,
                "cosine" => VectorDistanceMetric::CosineExpanded,
                _ => unreachable!("canonical CAGRA metric is registry validated"),
            };
            let mut params = VectorCagraParams::new(
                relations[0].clone(),
                relations[1].clone(),
                value_usize(operation, &canonical, "k")?,
                metric,
            );
            params.graph_degree = value_u32(operation, &canonical, "graph_degree")?;
            params.intermediate_graph_degree = value_u32(operation, &canonical, "intermediate_graph_degree")?;
            params.build_algo = match value_str(operation, &canonical, "build_algo")? {
                "ivf_pq" => VectorCagraBuildAlgo::IvfPq,
                "nn_descent" => VectorCagraBuildAlgo::NnDescent,
                _ => unreachable!("canonical CAGRA build algorithm is registry validated"),
            };
            params.itopk_size = value_u32(operation, &canonical, "itopk_size")?;
            params.search_width = value_u32(operation, &canonical, "search_width")?;
            params.max_iterations = value_u32(operation, &canonical, "max_iterations")?;
            params.search_algo = match value_str(operation, &canonical, "search_algo")? {
                "auto" => VectorCagraSearchAlgo::Auto,
                "single_cta" => VectorCagraSearchAlgo::SingleCta,
                "multi_cta" => VectorCagraSearchAlgo::MultiCta,
                "multi_kernel" => VectorCagraSearchAlgo::MultiKernel,
                _ => unreachable!("canonical CAGRA search algorithm is registry validated"),
            };
            VectorAlgorithmParams::Cagra(params)
        },
        CuvsOperation::PairwiseDistance => VectorAlgorithmParams::PairwiseDistance(VectorPairwiseDistanceParams {
            x: relations[0].clone(),
            y: relations[1].clone(),
            metric: match value_str(operation, &canonical, "metric")? {
                "l2_expanded" => VectorDistanceMetric::L2Expanded,
                "l2_sqrt_expanded" => VectorDistanceMetric::L2SqrtExpanded,
                "cosine" => VectorDistanceMetric::CosineExpanded,
                "inner_product" => VectorDistanceMetric::InnerProduct,
                _ => unreachable!("canonical pairwise metric is registry validated"),
            },
        }),
        CuvsOperation::TopK => VectorAlgorithmParams::TopK(VectorTopKParams {
            input: relations[0].clone(),
            k: value_usize(operation, &canonical, "k")?,
            select_min: value_str(operation, &canonical, "select")? == "min",
        }),
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
        CuvsOperation::IvfFlat => VectorAlgorithm::IvfFlat,
        CuvsOperation::IvfPq => VectorAlgorithm::IvfPq,
        CuvsOperation::Cagra => VectorAlgorithm::Cagra,
        CuvsOperation::PairwiseDistance => VectorAlgorithm::PairwiseDistance,
        CuvsOperation::TopK => VectorAlgorithm::TopK,
        CuvsOperation::KMeans => VectorAlgorithm::KMeans,
        CuvsOperation::Pca => VectorAlgorithm::Pca,
    };
    let root = builder
        .try_add_vector_algorithm(VectorAlgorithmSpec::new(algorithm, params, Arc::clone(&output_schema)))
        .map_err(|error| {
            Error::from_native(
                "algeon_datafusion.cuvs_sql.lower",
                "failed to lower cuVS SQL into native vector IR",
                error,
            )
        })?;
    Ok(builder.finish_with_host_sink_schema(root, output_schema))
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
