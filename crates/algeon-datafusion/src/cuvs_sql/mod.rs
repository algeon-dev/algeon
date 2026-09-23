//! Relation-aware cuVS SQL planning and execution contracts.
//!
//! The adapter owns SQL recognition, static validation, logical/physical
//! planning, and stable registry projections. Typed vector IR and device
//! execution remain DataFusion-independent in `query-engine`.

mod error;
mod exec;
mod gpu_family;
mod logical;
mod options;
mod query_planner;
mod registry;
mod relation_planner;

pub use exec::CuvsAlgoExec;
pub(crate) use gpu_family::{CuvsGpuFunctionFamily, cuvs_gpu_function_availability};
pub use logical::CuvsLogicalNode;
pub use options::{CuvsValidatedOptions, CuvsVectorRelationBinding};
pub use query_planner::{AlgeonQueryPlanner, CuvsExtensionPlanner};
pub use registry::{
    CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_ID_TYPES, CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY,
    CUVS_PAIRWISE_DISTANCE_FUNCTION, CUVS_PCA_FUNCTION, CUVS_TOP_K_FUNCTION, CuvsDenseVectorBinding, CuvsIdType,
    CuvsOperation, CuvsOperationDescriptor, CuvsOptionDefinition, CuvsOptionKind, CuvsOptionValidationError,
    CuvsOutputContract, CuvsRelationRole, CuvsVectorElementType, canonicalize_cuvs_options, cuvs_options_json_schema,
};
pub use relation_planner::{CuvsRelationPlanner, cuvs_relation_planner};
