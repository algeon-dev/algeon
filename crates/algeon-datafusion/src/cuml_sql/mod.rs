//! Relation-aware cuML SQL planning, catalog metadata, and execution.
//!
//! Truncated SVD fits and transforms Float32 relations within one query.
//! It returns the input ID, a row ordinal, and pc columns.

mod config;
mod error;
mod exec;
mod extension_planner;
mod gpu_family;
mod logical;
mod model_store;
mod options;
mod registry;
mod relation_planner;

pub use config::{CumlModelStoreConfig, CumlSqlConfig};
pub use exec::CumlAlgoExec;
pub use extension_planner::CumlExtensionPlanner;
pub(crate) use gpu_family::CumlGpuFunctionFamily;
pub use logical::CumlLogicalNode;
#[cfg(feature = "server")]
pub(crate) use logical::forest_model_digests;
pub(crate) use model_store::CumlModelStore;
pub use options::{CumlRelationBinding, CumlValidatedOptions};
pub use registry::{
    CUML_FOREST_PREDICT_FUNCTION, CUML_HDBSCAN_FUNCTION, CUML_LOGISTIC_REGRESSION_FUNCTION, CUML_OPERATION_REGISTRY,
    CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION, CUML_RANDOM_FOREST_REGRESSOR_FUNCTION, CUML_TSVD_FUNCTION,
    CUML_UMAP_FUNCTION, CumlOperation, CumlOperationDescriptor, CumlOutputContract, DenseVectorBinding,
    GpuOptionDefinition, GpuOptionKind, GpuOptionValidationError, MlRelationRole, RelationIdType,
    canonicalize_cuml_options, cuml_options_json_schema,
};
pub(crate) use relation_planner::cuml_relation_planner_with_store;
pub use relation_planner::{CumlRelationPlanner, cuml_relation_planner};
