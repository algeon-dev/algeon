//! SQL-facing re-exports of the canonical DataFusion-free cuVS registry.
//!
//! Operation identity, relation roles, option metadata, and output contracts
//! live in `nexus-query-engine`. This module intentionally contains no second
//! operation map or schema definition.

pub use nexus_query_engine::plan::{
    CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_ID_TYPES, CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PCA_FUNCTION,
    CuvsDenseVectorBinding, CuvsIdType, CuvsOperation, CuvsOperationDescriptor, CuvsOptionDefinition, CuvsOptionKind,
    CuvsOptionValidationError, CuvsOutputContract, CuvsRelationArgumentSyntax, CuvsRelationRole,
    canonicalize_cuvs_options,
};

/// Generate the frozen options schema from the canonical operation descriptor.
pub fn cuvs_options_json_schema(operation: CuvsOperation) -> serde_json::Value {
    operation.descriptor().options_json_schema()
}
