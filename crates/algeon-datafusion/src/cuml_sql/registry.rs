//! SQL-facing projection of the canonical cuML planning registry.

pub use query_planning::plan::{
    CUML_FOREST_PREDICT_FUNCTION, CUML_HDBSCAN_FUNCTION, CUML_LOGISTIC_REGRESSION_FUNCTION, CUML_OPERATION_REGISTRY,
    CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION, CUML_RANDOM_FOREST_REGRESSOR_FUNCTION, CUML_TSVD_FUNCTION,
    CUML_UMAP_FUNCTION, CumlOperation, CumlOperationDescriptor, CumlOutputContract, DenseVectorBinding,
    GpuOptionDefinition, GpuOptionKind, GpuOptionValidationError, MlRelationRole, RelationIdType,
    canonicalize_cuml_options,
};

/// Generate the options JSON schema from the canonical operation descriptor.
pub fn cuml_options_json_schema(operation: CumlOperation) -> serde_json::Value {
    operation.descriptor().options_json_schema()
}
