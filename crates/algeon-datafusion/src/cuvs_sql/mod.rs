//! Relation-aware cuVS SQL planning and execution contracts.
//!
//! The adapter owns SQL recognition, static validation, logical/physical
//! planning, and stable registry projections. Typed vector IR and device
//! execution remain DataFusion-independent in `query-engine`.

mod error;
mod exec;
mod extension_planner;
mod gpu_family;
mod logical;
mod options;
mod registry;
mod relation_planner;

pub use exec::CuvsAlgoExec;
pub use extension_planner::CuvsExtensionPlanner;
pub(crate) use gpu_family::CuvsGpuFunctionFamily;
pub use logical::CuvsLogicalNode;
pub use options::{CuvsValidatedOptions, CuvsVectorRelationBinding};
pub use query_planning::plan::VectorAlgorithmParams;
pub use registry::{
    CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_CAGRA_FUNCTION, CUVS_IVF_FLAT_FUNCTION, CUVS_IVF_PQ_FUNCTION,
    CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PAIRWISE_DISTANCE_FUNCTION, CUVS_PCA_FUNCTION,
    CUVS_TOP_K_FUNCTION, CuvsOperation, CuvsOperationDescriptor, CuvsOutputContract, CuvsRelationRole,
    DenseVectorBinding, GpuOptionDefinition, GpuOptionKind, GpuOptionValidationError, RELATION_ID_TYPES,
    RelationIdType, VectorElementType, canonicalize_cuvs_options, cuvs_options_json_schema,
};
pub use relation_planner::{CuvsRelationPlanner, cuvs_relation_planner};
