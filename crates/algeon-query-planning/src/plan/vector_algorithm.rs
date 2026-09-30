//! Native vector algorithm IR.
//!
//! The contracts in this module describe relational inputs, algorithm parameters, and result schemas without
//! depending on cuVS. The optional cuVS executor is selected later by the runtime.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};

use super::{NodeId, SchemaRef};

mod brute_force;
mod cagra;
mod common;
mod definition;
mod ivf_flat;
mod ivf_pq;
mod kmeans;
mod pairwise;
mod pca;
mod top_k;

pub use brute_force::VectorBruteForceKnnParams;
pub use cagra::{VectorCagraBuildAlgo, VectorCagraParams, VectorCagraSearchAlgo};
pub use common::{VectorDistanceMetric, VectorRelationBinding, VectorRowIdBinding};
pub use definition::{
    CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_CAGRA_FUNCTION, CUVS_IVF_FLAT_FUNCTION, CUVS_IVF_PQ_FUNCTION,
    CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PAIRWISE_DISTANCE_FUNCTION, CUVS_PCA_FUNCTION,
    CUVS_TOP_K_FUNCTION, CuvsOperation, CuvsOperationDescriptor, CuvsOutputContract, CuvsRelationRole,
    CuvsVectorElementSupport, VectorAlgorithm, canonicalize_cuvs_options,
};
pub use ivf_flat::VectorIvfFlatParams;
pub use ivf_pq::{VectorIvfPqCodebookKind, VectorIvfPqParams};
pub use kmeans::{VectorKMeansInit, VectorKMeansMetric, VectorKMeansParams};
pub use pairwise::VectorPairwiseDistanceParams;
pub use pca::{VectorPcaParams, VectorPcaSolver};
pub use top_k::VectorTopKParams;

/// Algorithm-specific vector parameter payload.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorAlgorithmParams {
    /// Exact brute-force kNN parameters.
    BruteForceKnn(VectorBruteForceKnnParams),
    /// Query-local IVF-Flat search parameters.
    IvfFlat(VectorIvfFlatParams),
    /// Query-local IVF-PQ search parameters.
    IvfPq(VectorIvfPqParams),
    /// Query-local graph search parameters.
    Cagra(VectorCagraParams),
    /// Pairwise metric values between two relations.
    PairwiseDistance(VectorPairwiseDistanceParams),
    /// Ranked values selected within each input vector.
    TopK(VectorTopKParams),
    /// K-means fit/predict parameters.
    KMeans(VectorKMeansParams),
    /// PCA fit-transform parameters.
    Pca(VectorPcaParams),
}

impl VectorAlgorithmParams {
    /// Whether this payload matches the declared algorithm.
    pub const fn supports_algorithm(&self, algorithm: VectorAlgorithm) -> bool {
        matches!(
            (self, algorithm),
            (Self::BruteForceKnn(_), VectorAlgorithm::BruteForceKnn)
                | (Self::IvfFlat(_), VectorAlgorithm::IvfFlat)
                | (Self::IvfPq(_), VectorAlgorithm::IvfPq)
                | (Self::Cagra(_), VectorAlgorithm::Cagra)
                | (Self::PairwiseDistance(_), VectorAlgorithm::PairwiseDistance)
                | (Self::TopK(_), VectorAlgorithm::TopK)
                | (Self::KMeans(_), VectorAlgorithm::KMeans)
                | (Self::Pca(_), VectorAlgorithm::Pca)
        )
    }

    /// Visit relation bindings in stable semantic execution order.
    pub fn relation_bindings(&self) -> Box<dyn Iterator<Item = &VectorRelationBinding> + '_> {
        match self {
            Self::BruteForceKnn(params) => Box::new([&params.dataset, &params.queries].into_iter()),
            Self::IvfFlat(params) => Box::new([&params.dataset, &params.queries].into_iter()),
            Self::IvfPq(params) => Box::new([&params.dataset, &params.queries].into_iter()),
            Self::Cagra(params) => Box::new([&params.dataset, &params.queries].into_iter()),
            Self::PairwiseDistance(params) => Box::new([&params.x, &params.y].into_iter()),
            Self::TopK(params) => Box::new(std::iter::once(&params.input)),
            Self::KMeans(params) => Box::new(std::iter::once(&params.input).chain(params.initial_centroids.iter())),
            Self::Pca(params) => Box::new(std::iter::once(&params.input)),
        }
    }

    pub(crate) fn remap_relation_inputs(&mut self, mut remap: impl FnMut(NodeId) -> NodeId) {
        match self {
            Self::BruteForceKnn(params) => {
                params.dataset.input = remap(params.dataset.input);
                params.queries.input = remap(params.queries.input);
            },
            Self::IvfFlat(params) => {
                params.dataset.input = remap(params.dataset.input);
                params.queries.input = remap(params.queries.input);
            },
            Self::IvfPq(params) => {
                params.dataset.input = remap(params.dataset.input);
                params.queries.input = remap(params.queries.input);
            },
            Self::Cagra(params) => {
                params.dataset.input = remap(params.dataset.input);
                params.queries.input = remap(params.queries.input);
            },
            Self::PairwiseDistance(params) => {
                params.x.input = remap(params.x.input);
                params.y.input = remap(params.y.input);
            },
            Self::TopK(params) => params.input.input = remap(params.input.input),
            Self::KMeans(params) => {
                params.input.input = remap(params.input.input);
                if let Some(centroids) = &mut params.initial_centroids {
                    centroids.input = remap(centroids.input);
                }
            },
            Self::Pca(params) => params.input.input = remap(params.input.input),
        }
    }
}

/// Native vector algorithm specification.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorAlgorithmSpec {
    /// Algorithm to execute.
    pub algorithm: VectorAlgorithm,
    /// Algorithm-specific relation and scalar parameters.
    pub params: VectorAlgorithmParams,
    /// Explicit relational output schema.
    pub output_schema: SchemaRef,
}

impl VectorAlgorithmSpec {
    /// Construct a vector algorithm specification.
    pub fn new(algorithm: VectorAlgorithm, params: VectorAlgorithmParams, output_schema: SchemaRef) -> Self {
        assert!(params.supports_algorithm(algorithm), "vector algorithm and parameter variant must match");
        Self { algorithm, params, output_schema }
    }
}

/// Native vector algorithm operation over one or more relation inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorAlgorithmNode {
    /// All semantic inputs in the order returned by
    /// [`VectorAlgorithmParams::relation_bindings`].
    pub inputs: Vec<NodeId>,
    /// Typed algorithm specification.
    pub spec: VectorAlgorithmSpec,
}

impl VectorAlgorithmNode {
    /// Output schema declared by the vector algorithm specification.
    pub fn output_schema(&self) -> &SchemaRef {
        &self.spec.output_schema
    }
}

/// Stable relational result schema for nearest-neighbor search.
pub fn vector_knn_output_schema(query_id_type: DataType, neighbor_id_type: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("query_ordinal", DataType::UInt64, false),
        Field::new("query_id", query_id_type, false),
        Field::new("neighbor_ordinal", DataType::UInt64, false),
        Field::new("neighbor_id", neighbor_id_type, false),
        Field::new("rank", DataType::UInt32, false),
        Field::new("distance", DataType::Float32, false),
    ]))
}

/// Stable long-form row-major pairwise metric result.
pub fn vector_pairwise_distance_output_schema(
    x_id_type: DataType,
    y_id_type: DataType,
    distance_type: DataType,
) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("x_ordinal", DataType::UInt64, false),
        Field::new("x_id", x_id_type, false),
        Field::new("y_ordinal", DataType::UInt64, false),
        Field::new("y_id", y_id_type, false),
        Field::new("distance", distance_type, false),
    ]))
}

/// Stable ranked selection result schema.
pub fn vector_top_k_output_schema(id_type: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("row_ordinal", DataType::UInt64, false),
        Field::new("id", id_type, false),
        Field::new("rank", DataType::UInt32, false),
        Field::new("position", DataType::Int64, false),
        Field::new("value", DataType::Float32, false),
    ]))
}

/// Stable relational result schema for K-means row assignments.
pub fn vector_kmeans_output_schema(row_id_type: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("row_ordinal", DataType::UInt64, false),
        Field::new("id", row_id_type, false),
        Field::new("cluster_id", DataType::Int32, false),
    ]))
}

/// Stable relational result schema for PCA transformed rows.
pub fn vector_pca_output_schema(row_id_type: DataType, n_components: usize) -> SchemaRef {
    let mut fields = Vec::with_capacity(n_components.saturating_add(2));
    fields.push(Field::new("row_ordinal", DataType::UInt64, false));
    fields.push(Field::new("id", row_id_type, false));
    fields.extend((0..n_components).map(|index| Field::new(format!("pc_{index}"), DataType::Float32, false)));
    Arc::new(Schema::new(fields))
}
