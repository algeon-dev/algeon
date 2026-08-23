//! Relation-aware cuVS SQL planning and execution contracts.
//!
//! The adapter owns SQL recognition, static validation, logical/physical
//! planning, and stable registry projections. Typed vector IR and device
//! execution remain DataFusion-independent in `nexus-query-engine`.

mod error;
mod exec;
#[cfg(test)]
#[path = "exec_tests.rs"]
mod exec_tests;
mod gpu_family;
mod logical;
mod lowering;
mod options;
mod query_planner;
mod registry;
mod relation_planner;

pub use exec::CuvsAlgoExec;
pub(crate) use gpu_family::{CuvsGpuFunctionFamily, cuvs_gpu_function_availability};
pub use logical::CuvsLogicalNode;
pub use options::{CuvsValidatedOptions, CuvsVectorRelationBinding};
pub use query_planner::{CuvsExtensionPlanner, CuvsQueryPlanner};
pub use registry::{
    CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_ID_TYPES, CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PCA_FUNCTION,
    CuvsDenseVectorBinding, CuvsIdType, CuvsOperation, CuvsOperationDescriptor, CuvsOptionDefinition, CuvsOptionKind,
    CuvsOptionValidationError, CuvsOutputContract, CuvsRelationArgumentSyntax, CuvsRelationRole,
    canonicalize_cuvs_options, cuvs_options_json_schema,
};
pub use relation_planner::{CuvsRelationPlanner, cuvs_relation_planner};
