//! SQL-facing re-exports of the canonical DataFusion-free cuVS registry.
//!
//! Operation identity, relation roles, option metadata, and output contracts
//! are defined by `query-planning` and exposed through the compatible
//! `query-engine` facade. This module intentionally contains no second
//! operation map or schema definition.

pub use query_engine::plan::{
    CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_CAGRA_FUNCTION, CUVS_ID_TYPES, CUVS_IVF_FLAT_FUNCTION, CUVS_IVF_PQ_FUNCTION,
    CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PAIRWISE_DISTANCE_FUNCTION, CUVS_PCA_FUNCTION,
    CUVS_TOP_K_FUNCTION, CuvsDenseVectorBinding, CuvsIdType, CuvsOperation, CuvsOperationDescriptor,
    CuvsOptionDefinition, CuvsOptionKind, CuvsOptionValidationError, CuvsOutputContract, CuvsRelationRole,
    CuvsVectorElementType, canonicalize_cuvs_options,
};

/// Generate the frozen options schema from the canonical operation descriptor.
pub fn cuvs_options_json_schema(operation: CuvsOperation) -> serde_json::Value {
    operation.descriptor().options_json_schema()
}
