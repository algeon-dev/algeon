//! Relation-aware cuVS SQL planning and execution contracts.
//!
//! The adapter owns SQL recognition, static validation, logical/physical
//! planning, and stable registry projections. Typed vector IR and device
//! execution remain DataFusion-independent in `nexus-query-engine`.

mod error;
mod exec;
mod logical;
mod lowering;
mod metadata;
mod options;
mod query_planner;
mod registry;
mod relation_planner;

pub use exec::CuvsAlgoExec;
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

pub const CUVS_LIST_OPERATIONS_FUNCTION: &str = "cuvs_list_operations";
pub const CUVS_DESCRIBE_OPERATION_FUNCTION: &str = "cuvs_describe_operation";
pub const CUVS_VALIDATE_CALL_FUNCTION: &str = "cuvs_validate_call";

pub(crate) use metadata::cuvs_metadata_table_functions;
