//! cuML operation definitions and stable ML algorithm identities.

use std::fmt;

use serde_json::{Value, json};

use crate::plan::{
    GpuOptionDefinition, GpuOptionKind, GpuOptionValidationError, GpuOutputFieldDefinition, VectorElementType,
    gpu_function_options::{
        canonicalize_options, dense_relation_binding_schema, option_schema, validate_dense_relation_binding_value,
    },
};

pub const CUML_TSVD_FUNCTION: &str = "cuml_tsvd";
pub const CUML_HDBSCAN_FUNCTION: &str = "cuml_hdbscan";
pub const CUML_UMAP_FUNCTION: &str = "cuml_umap";
pub const CUML_LOGISTIC_REGRESSION_FUNCTION: &str = "cuml_logistic_regression";
pub const CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION: &str = "cuml_random_forest_classifier";
pub const CUML_RANDOM_FOREST_REGRESSOR_FUNCTION: &str = "cuml_random_forest_regressor";
pub const CUML_FOREST_PREDICT_FUNCTION: &str = "cuml_forest_predict";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CumlOperation {
    Tsvd,
    Hdbscan,
    Umap,
    LogisticRegression,
    RandomForestClassifier,
    RandomForestRegressor,
    ForestPredict,
}

impl CumlOperation {
    pub fn all() -> impl ExactSizeIterator<Item = Self> + Clone {
        CUML_OPERATION_REGISTRY.iter().map(|descriptor| descriptor.operation)
    }

    pub fn descriptor(self) -> &'static CumlOperationDescriptor {
        CUML_OPERATION_REGISTRY.iter().find(|descriptor| descriptor.operation == self).expect("cuML registry entry")
    }

    pub fn from_sql_function_name(name: &str) -> Option<Self> {
        CUML_OPERATION_REGISTRY
            .iter()
            .find(|descriptor| descriptor.sql_function_name == name)
            .map(|entry| entry.operation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlRelationRole {
    Input,
    Training,
    Predict,
}

impl MlRelationRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Training => "training",
            Self::Predict => "predict",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::Input => "Dense Float32 rows fitted and transformed in this statement.",
            Self::Training => "Dense Float32 features and non-null Int32 label used to fit the query-local model.",
            Self::Predict => "Independent dense Float32 rows scored with the query-local model; must omit label.",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CumlOutputContract {
    Components,
    HdbscanClusters,
    Embedding,
    Predictions,
    ForestPrediction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CumlOperationDescriptor {
    pub operation: CumlOperation,
    pub sql_function_name: &'static str,
    pub display_name: &'static str,
    pub summary: &'static str,
    pub use_this_when: &'static str,
    pub relation_roles: &'static [MlRelationRole],
    pub vector_element_support: &'static [VectorElementType],
    pub output_contract: CumlOutputContract,
    pub options: &'static [GpuOptionDefinition],
    pub output_fields: &'static [GpuOutputFieldDefinition],
    pub quickstart_sql: &'static str,
    pub limitations: &'static [&'static str],
}

const ELEMENTS: &[VectorElementType] = &[VectorElementType::Float32];
const ROLES: &[MlRelationRole] = &[MlRelationRole::Input];
const SUPERVISED_ROLES: &[MlRelationRole] = &[MlRelationRole::Training, MlRelationRole::Predict];
const SOLVERS: &[&str] = &["cov_eig_dq", "cov_eig_jacobi"];
const COMMON_OPTIONS: &[GpuOptionDefinition] = &[
    GpuOptionDefinition {
        name: "input",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Non-null ID and ordered Float32 features from the input subquery.",
    },
    GpuOptionDefinition {
        name: "n_components",
        required: true,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: None },
        description: "Number of transformed pc_<index> columns.",
    },
    GpuOptionDefinition {
        name: "solver",
        required: false,
        kind: GpuOptionKind::Enum { values: SOLVERS, default: Some("cov_eig_dq") },
        description: "Covariance eigensolver used by the query-local fit.",
    },
    GpuOptionDefinition {
        name: "tol",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 0.0 },
        description: "Finite nonnegative eigensolver tolerance.",
    },
    GpuOptionDefinition {
        name: "n_iterations",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: Some(15) },
        description: "Maximum number of eigensolver iterations.",
    },
];
const HDBSCAN_METHODS: &[&str] = &["eom", "leaf"];
const UMAP_INITIALIZATIONS: &[&str] = &["spectral", "random"];
const HDBSCAN_OPTIONS: &[GpuOptionDefinition] = &[
    COMMON_OPTIONS[0],
    GpuOptionDefinition {
        name: "min_samples",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(5) },
        description: "Minimum samples in a core-distance neighborhood.",
    },
    GpuOptionDefinition {
        name: "min_cluster_size",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 2, maximum: i32::MAX as u64, default: Some(5) },
        description: "Minimum size of a selected cluster.",
    },
    GpuOptionDefinition {
        name: "max_cluster_size",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 0, maximum: i32::MAX as u64, default: Some(0) },
        description: "Maximum cluster size; zero disables the bound.",
    },
    GpuOptionDefinition {
        name: "cluster_selection_epsilon",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 0.0 },
        description: "Nonnegative threshold for cluster selection.",
    },
    GpuOptionDefinition {
        name: "alpha",
        required: false,
        kind: GpuOptionKind::PositiveNumber { default: 1.0 },
        description: "Positive mutual-reachability distance scale.",
    },
    GpuOptionDefinition {
        name: "allow_single_cluster",
        required: false,
        kind: GpuOptionKind::Boolean { default: Some(false) },
        description: "Allow one cluster to cover all input rows.",
    },
    GpuOptionDefinition {
        name: "cluster_selection_method",
        required: false,
        kind: GpuOptionKind::Enum { values: HDBSCAN_METHODS, default: Some("eom") },
        description: "Select clusters by excess of mass or leaf nodes.",
    },
];
const UMAP_OPTIONS: &[GpuOptionDefinition] = &[
    COMMON_OPTIONS[0],
    GpuOptionDefinition {
        name: "n_components",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(2) },
        description: "Number of embedding dimensions.",
    },
    GpuOptionDefinition {
        name: "n_neighbors",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(15) },
        description: "Number of neighbors used to construct the manifold.",
    },
    GpuOptionDefinition {
        name: "n_epochs",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 0, maximum: i32::MAX as u64, default: Some(0) },
        description: "Optimization epochs; zero selects the native default.",
    },
    GpuOptionDefinition {
        name: "learning_rate",
        required: false,
        kind: GpuOptionKind::PositiveNumber { default: 1.0 },
        description: "Positive embedding optimization rate.",
    },
    GpuOptionDefinition {
        name: "min_dist",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 0.1 },
        description: "Minimum distance between embedded points; at most spread.",
    },
    GpuOptionDefinition {
        name: "spread",
        required: false,
        kind: GpuOptionKind::PositiveNumber { default: 1.0 },
        description: "Positive scale for embedded point distances.",
    },
    GpuOptionDefinition {
        name: "set_op_mix_ratio",
        required: false,
        kind: GpuOptionKind::UnitInterval { default: 1.0 },
        description: "Fuzzy-set intersection/union mix in [0, 1].",
    },
    GpuOptionDefinition {
        name: "local_connectivity",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 1.0 },
        description: "Local connectivity adjustment.",
    },
    GpuOptionDefinition {
        name: "repulsion_strength",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 1.0 },
        description: "Repulsive force applied during optimization.",
    },
    GpuOptionDefinition {
        name: "negative_sample_rate",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(5) },
        description: "Negative samples per positive edge.",
    },
    GpuOptionDefinition {
        name: "init",
        required: false,
        kind: GpuOptionKind::Enum { values: UMAP_INITIALIZATIONS, default: Some("spectral") },
        description: "Embedding initialization method.",
    },
    GpuOptionDefinition {
        name: "random_state",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 0, maximum: u64::MAX, default: Some(0) },
        description: "Seed for reproducible initialization.",
    },
];
const PREDICTION_FIELDS: &[GpuOutputFieldDefinition] = &[
    OUTPUT_FIELDS[0],
    GpuOutputFieldDefinition {
        name: "id",
        data_type: "same_as_predict.id",
        nullable: false,
        repeats: None,
        description: "Logical ID from the predict relation.",
    },
    GpuOutputFieldDefinition {
        name: "prediction",
        data_type: "Int32",
        nullable: false,
        repeats: None,
        description: "Predicted zero-based class index.",
    },
];
const LOGISTIC_OPTIONS: &[GpuOptionDefinition] = &[
    GpuOptionDefinition {
        name: "training",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Non-null ID, Float32 features, and a non-null Int32 label.",
    },
    GpuOptionDefinition {
        name: "predict",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Non-null ID and Float32 features without a label.",
    },
    GpuOptionDefinition {
        name: "penalty_l1",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 0.0 },
        description: "L1 penalty.",
    },
    GpuOptionDefinition {
        name: "penalty_l2",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 0.0 },
        description: "L2 penalty.",
    },
    GpuOptionDefinition {
        name: "max_iter",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(1000) },
        description: "Maximum QN iterations.",
    },
    GpuOptionDefinition {
        name: "linesearch_max_iter",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(50) },
        description: "Maximum line-search iterations.",
    },
    GpuOptionDefinition {
        name: "lbfgs_memory",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(5) },
        description: "L-BFGS history length.",
    },
    GpuOptionDefinition {
        name: "fit_intercept",
        required: false,
        kind: GpuOptionKind::Boolean { default: Some(true) },
        description: "Fit an intercept.",
    },
    GpuOptionDefinition {
        name: "penalty_normalized",
        required: false,
        kind: GpuOptionKind::Boolean { default: Some(true) },
        description: "Normalize the penalty by row count.",
    },
    GpuOptionDefinition {
        name: "grad_tol",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 1e-4 },
        description: "QN gradient tolerance.",
    },
    GpuOptionDefinition {
        name: "change_tol",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 1e-5 },
        description: "QN objective change tolerance.",
    },
];
const RF_CLASSIFIER_FIELDS: &[GpuOutputFieldDefinition] = PREDICTION_FIELDS;
const RF_REGRESSOR_FIELDS: &[GpuOutputFieldDefinition] = &[
    PREDICTION_FIELDS[0],
    PREDICTION_FIELDS[1],
    GpuOutputFieldDefinition {
        name: "prediction",
        data_type: "Float32",
        nullable: false,
        repeats: None,
        description: "Predicted regression target.",
    },
];
const RF_CRITERIA: &[&str] = &["gini", "entropy"];
const RF_OPTIONS: &[GpuOptionDefinition] = &[
    LOGISTIC_OPTIONS[0],
    LOGISTIC_OPTIONS[1],
    GpuOptionDefinition {
        name: "n_trees",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(100) },
        description: "Number of trees.",
    },
    GpuOptionDefinition {
        name: "max_depth",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 0, maximum: i32::MAX as u64, default: Some(16) },
        description: "Maximum tree depth; zero uses the native unlimited depth.",
    },
    GpuOptionDefinition {
        name: "max_leaves",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: None },
        description: "Maximum leaf count; omission uses the native unlimited setting.",
    },
    GpuOptionDefinition {
        name: "max_features",
        required: false,
        kind: GpuOptionKind::Fraction { default: 1.0 },
        description: "Fraction of features sampled per split.",
    },
    GpuOptionDefinition {
        name: "max_samples",
        required: false,
        kind: GpuOptionKind::Fraction { default: 1.0 },
        description: "Fraction of training rows sampled per tree.",
    },
    GpuOptionDefinition {
        name: "bootstrap",
        required: false,
        kind: GpuOptionKind::Boolean { default: Some(true) },
        description: "Sample training rows with replacement.",
    },
    GpuOptionDefinition {
        name: "max_n_bins",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 2, maximum: i32::MAX as u64, default: Some(128) },
        description: "Maximum histogram bins.",
    },
    GpuOptionDefinition {
        name: "min_samples_leaf",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(1) },
        description: "Minimum samples per leaf.",
    },
    GpuOptionDefinition {
        name: "min_samples_split",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 2, maximum: i32::MAX as u64, default: Some(2) },
        description: "Minimum samples to split a node.",
    },
    GpuOptionDefinition {
        name: "min_impurity_decrease",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 0.0 },
        description: "Minimum split impurity reduction.",
    },
    GpuOptionDefinition {
        name: "max_batch_size",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(4096) },
        description: "Maximum prediction batch size.",
    },
    GpuOptionDefinition {
        name: "seed",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 0, maximum: u64::MAX, default: Some(0) },
        description: "Training random seed.",
    },
];
const RF_CLASSIFIER_OPTIONS: &[GpuOptionDefinition] = &[
    RF_OPTIONS[0],
    RF_OPTIONS[1],
    RF_OPTIONS[2],
    RF_OPTIONS[3],
    RF_OPTIONS[4],
    RF_OPTIONS[5],
    RF_OPTIONS[6],
    RF_OPTIONS[7],
    RF_OPTIONS[8],
    RF_OPTIONS[9],
    RF_OPTIONS[10],
    RF_OPTIONS[11],
    RF_OPTIONS[12],
    RF_OPTIONS[13],
    GpuOptionDefinition {
        name: "split_criterion",
        required: false,
        kind: GpuOptionKind::Enum { values: RF_CRITERIA, default: Some("gini") },
        description: "Classification split criterion.",
    },
];
const RF_LIMITATIONS: &[&str] = &[
    "Classifier labels must be non-null Int32 contiguous classes 0..C-1; regressor labels must be non-null finite Float32. Predict must omit label and match the training feature dimension.",
    "Prediction imports the fitted forest into nvForest and scores the predict relation on the GPU; a classifier tie resolves to the lowest class index. Training and prediction inputs are limited to i32::MAX rows and the packed feature matrix to i32::MAX elements; the random forest path does not batch rows.",
    "Training uses one RAFT pool stream whose allocations share the query grant. The forest is query-local; an empty predict relation returns an empty result.",
];
const LOGISTIC_LIMITATIONS: &[&str] = &[
    "Training labels must be non-null Int32 contiguous classes 0..C-1; predict must omit label and match the feature dimension.",
    "Binary labels use logistic loss; three or more classes use softmax. Reaching max_iter is not an error.",
    "The model is query-local. An empty predict relation returns an empty result without a native call.",
];
const HDBSCAN_FIELDS: &[GpuOutputFieldDefinition] = &[
    OUTPUT_FIELDS[0],
    OUTPUT_FIELDS[1],
    GpuOutputFieldDefinition {
        name: "cluster_id",
        data_type: "Int64",
        nullable: false,
        repeats: None,
        description: "Selected cluster ID; -1 indicates noise.",
    },
    GpuOutputFieldDefinition {
        name: "probability",
        data_type: "Float32",
        nullable: false,
        repeats: None,
        description: "Membership probability for the selected cluster.",
    },
];
const EMBEDDING_FIELDS: &[GpuOutputFieldDefinition] = &[
    OUTPUT_FIELDS[0],
    OUTPUT_FIELDS[1],
    GpuOutputFieldDefinition {
        name: "dim_<index>",
        data_type: "Float32",
        nullable: false,
        repeats: Some("n_components"),
        description: "Embedded dimension value, repeated n_components times.",
    },
];
const HDBSCAN_LIMITATIONS: &[&str] = &[
    "Only Float32 features are supported; L2SqrtExpanded distance and brute-force kNN are fixed.",
    "Empty input returns an empty result. Other inputs require min_samples < rows, min_cluster_size <= rows, and max_cluster_size <= rows; row-dependent conditions are checked at execution.",
    "Cluster ID -1 denotes noise. The fit and output are query-local.",
];
const UMAP_LIMITATIONS: &[&str] = &[
    "Only Float32 features are supported; L2SqrtExpanded distance, brute-force kNN, and deterministic mode are fixed.",
    "Empty input returns an empty result. Other inputs require n_neighbors < rows, checked at execution.",
    "min_dist must not exceed spread. The fit and embedding are query-local.",
];
const OUTPUT_FIELDS: &[GpuOutputFieldDefinition] = &[
    GpuOutputFieldDefinition {
        name: "row_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Evaluated input row position, including when IDs repeat.",
    },
    GpuOutputFieldDefinition {
        name: "id",
        data_type: "same_as_input.id",
        nullable: false,
        repeats: None,
        description: "Logical input ID.",
    },
    GpuOutputFieldDefinition {
        name: "pc_<index>",
        data_type: "Float32",
        nullable: false,
        repeats: Some("n_components"),
        description: "Transformed component value, repeated n_components times.",
    },
];
const FOREST_MODEL_FORMATS: &[&str] = &["xgboost_json", "xgboost_ubjson", "xgboost_legacy", "lightgbm", "treelite"];
const FOREST_OUTPUTS: &[&str] = &["scores", "class_index"];
const FOREST_OPTIONS: &[GpuOptionDefinition] = &[
    COMMON_OPTIONS[0],
    GpuOptionDefinition {
        name: "model",
        required: true,
        kind: GpuOptionKind::Text { max_chars: 1024 },
        description: "Model file path relative to the server-configured models root.",
    },
    GpuOptionDefinition {
        name: "model_format",
        required: true,
        kind: GpuOptionKind::Enum { values: FOREST_MODEL_FORMATS, default: None },
        description: "Serialized format of the model file.",
    },
    GpuOptionDefinition {
        name: "output",
        required: false,
        kind: GpuOptionKind::Enum { values: FOREST_OUTPUTS, default: Some("scores") },
        description: "Return the model's postprocessed scores, or the index of the largest class score.",
    },
];
const FOREST_FIELDS: &[GpuOutputFieldDefinition] = &[
    OUTPUT_FIELDS[0],
    OUTPUT_FIELDS[1],
    GpuOutputFieldDefinition {
        name: "prediction",
        data_type: "Float32 or Int32",
        nullable: false,
        repeats: None,
        description: "Float32 score of a single-output model with output => 'scores'; Int32 class index with output => 'class_index'.",
    },
    GpuOutputFieldDefinition {
        name: "output_<index>",
        data_type: "Float32",
        nullable: false,
        repeats: Some("model output count"),
        description: "Score of one model output, returned instead of prediction for a multi-output model with output => 'scores'.",
    },
];
const FOREST_LIMITATIONS: &[&str] = &[
    "Scores pretrained XGBoost (JSON, UBJSON, legacy binary), LightGBM text, and Treelite checkpoint forests with nvForest. Multi-target models are rejected at planning.",
    "Features must be non-null Float32 and match the model feature count. Forests with double-precision thresholds widen the features to Float64 on the device.",
    "output => 'class_index' requires a model with at least two outputs and resolves ties to the lowest class index.",
    "model resolves under the server-configured models root, and files above max_model_bytes are rejected. Each statement reads and imports the model once; there is no cross-statement model cache.",
];
const TSVD_LIMITATIONS: &[&str] = &[
    "Empty input returns an empty result without a native call. Nonempty input requires at least two evaluated rows and two feature dimensions. n_components cannot exceed dimension; known dimensions are checked during planning, otherwise execution reports a structured error.",
    "Only Float32 features are supported. Use CAST in the input subquery for other element types.",
    "Component sign is not semantic; this operation returns transformed rows only, not components, explained variance, or mean.",
    "TSVD allocates a temporary rows * n_components score matrix inside the query device cap.",
];

pub const CUML_OPERATION_REGISTRY: &[CumlOperationDescriptor] = &[
    CumlOperationDescriptor {
        operation: CumlOperation::Tsvd,
        sql_function_name: CUML_TSVD_FUNCTION,
        display_name: "TSVD",
        summary: "cuML truncated SVD fits and transforms Float32 input rows.",
        use_this_when: "Reduce input features to query-local transformed rows using truncated SVD.",
        relation_roles: ROLES,
        vector_element_support: ELEMENTS,
        output_contract: CumlOutputContract::Components,
        options: COMMON_OPTIONS,
        output_fields: OUTPUT_FIELDS,
        quickstart_sql: "SELECT id, pc_0, pc_1 FROM cuml_tsvd(input => (SELECT id, d0, d1 FROM t), n_components => 2) ORDER BY row_ordinal;",
        limitations: TSVD_LIMITATIONS,
    },
    CumlOperationDescriptor {
        operation: CumlOperation::Hdbscan,
        sql_function_name: CUML_HDBSCAN_FUNCTION,
        display_name: "HDBSCAN",
        summary: "cuML HDBSCAN fits density-based clusters and returns query-local labels and membership probabilities.",
        use_this_when: "Find dense groups and identify noise among Float32 feature rows.",
        relation_roles: ROLES,
        vector_element_support: ELEMENTS,
        output_contract: CumlOutputContract::HdbscanClusters,
        options: HDBSCAN_OPTIONS,
        output_fields: HDBSCAN_FIELDS,
        quickstart_sql: "SELECT id, cluster_id, probability FROM cuml_hdbscan(input => (SELECT id, d0, d1 FROM t), min_cluster_size => 5) ORDER BY row_ordinal;",
        limitations: HDBSCAN_LIMITATIONS,
    },
    CumlOperationDescriptor {
        operation: CumlOperation::Umap,
        sql_function_name: CUML_UMAP_FUNCTION,
        display_name: "UMAP",
        summary: "cuML UMAP fits a deterministic, query-local manifold embedding for Float32 rows.",
        use_this_when: "Embed dense Float32 feature rows into a chosen number of dimensions.",
        relation_roles: ROLES,
        vector_element_support: ELEMENTS,
        output_contract: CumlOutputContract::Embedding,
        options: UMAP_OPTIONS,
        output_fields: EMBEDDING_FIELDS,
        quickstart_sql: "SELECT id, dim_0, dim_1 FROM cuml_umap(input => (SELECT id, d0, d1 FROM t), n_neighbors => 5) ORDER BY row_ordinal;",
        limitations: UMAP_LIMITATIONS,
    },
    CumlOperationDescriptor {
        operation: CumlOperation::LogisticRegression,
        sql_function_name: CUML_LOGISTIC_REGRESSION_FUNCTION,
        display_name: "Logistic Regression",
        summary: "Fit a query-local cuML logistic or softmax classifier and score independent rows.",
        use_this_when: "Classify Float32 predict rows after fitting contiguous Int32 training labels.",
        relation_roles: SUPERVISED_ROLES,
        vector_element_support: ELEMENTS,
        output_contract: CumlOutputContract::Predictions,
        options: LOGISTIC_OPTIONS,
        output_fields: PREDICTION_FIELDS,
        quickstart_sql: "SELECT id, prediction FROM cuml_logistic_regression(training => (SELECT id, d0, label FROM train), predict => (SELECT id, d0 FROM test)) ORDER BY row_ordinal;",
        limitations: LOGISTIC_LIMITATIONS,
    },
    CumlOperationDescriptor {
        operation: CumlOperation::RandomForestClassifier,
        sql_function_name: CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION,
        display_name: "Random Forest Classifier",
        summary: "Fit a query-local cuML random forest classifier and score independent rows.",
        use_this_when: "Classify Float32 rows after fitting contiguous Int32 training labels.",
        relation_roles: SUPERVISED_ROLES,
        vector_element_support: ELEMENTS,
        output_contract: CumlOutputContract::Predictions,
        options: RF_CLASSIFIER_OPTIONS,
        output_fields: RF_CLASSIFIER_FIELDS,
        quickstart_sql: "SELECT id, prediction FROM cuml_random_forest_classifier(training => (SELECT id, d0, label FROM train), predict => (SELECT id, d0 FROM test)) ORDER BY row_ordinal;",
        limitations: RF_LIMITATIONS,
    },
    CumlOperationDescriptor {
        operation: CumlOperation::RandomForestRegressor,
        sql_function_name: CUML_RANDOM_FOREST_REGRESSOR_FUNCTION,
        display_name: "Random Forest Regressor",
        summary: "Fit a query-local cuML random forest regressor and score independent rows.",
        use_this_when: "Predict Float32 targets from Float32 feature rows.",
        relation_roles: SUPERVISED_ROLES,
        vector_element_support: ELEMENTS,
        output_contract: CumlOutputContract::Predictions,
        options: RF_OPTIONS,
        output_fields: RF_REGRESSOR_FIELDS,
        quickstart_sql: "SELECT id, prediction FROM cuml_random_forest_regressor(training => (SELECT id, d0, label FROM train), predict => (SELECT id, d0 FROM test)) ORDER BY row_ordinal;",
        limitations: RF_LIMITATIONS,
    },
    CumlOperationDescriptor {
        operation: CumlOperation::ForestPredict,
        sql_function_name: CUML_FOREST_PREDICT_FUNCTION,
        display_name: "Forest Predict",
        summary: "Score Float32 rows with a pretrained XGBoost, LightGBM, or Treelite forest through nvForest.",
        use_this_when: "Apply an externally trained tree-ensemble model file to rows in a query.",
        relation_roles: ROLES,
        vector_element_support: ELEMENTS,
        output_contract: CumlOutputContract::ForestPrediction,
        options: FOREST_OPTIONS,
        output_fields: FOREST_FIELDS,
        quickstart_sql: "SELECT id, prediction FROM cuml_forest_predict(input => (SELECT id, d0, d1, d2, d3 FROM t), model => 'scorer.json', model_format => 'xgboost_json') ORDER BY row_ordinal;",
        limitations: FOREST_LIMITATIONS,
    },
];

impl CumlOperationDescriptor {
    pub fn option(&self, name: &str) -> Option<&'static GpuOptionDefinition> {
        self.options.iter().find(|option| option.name == name)
    }

    pub fn options_json_schema(&self) -> Value {
        let relation_schema = dense_relation_binding_schema(self.vector_element_support);
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for option in self.options {
            let binding_schema = if matches!(
                self.operation,
                CumlOperation::LogisticRegression
                    | CumlOperation::RandomForestClassifier
                    | CumlOperation::RandomForestRegressor
            ) && option.name == "training"
            {
                let mut schema = relation_schema.clone();
                schema["properties"]["label"] = json!({"type": "string", "const": "label", "description": if self.operation == CumlOperation::RandomForestRegressor { "Canonical non-null Float32 target column." } else { "Canonical non-null Int32 class column." }});
                schema["required"].as_array_mut().expect("binding required fields").push(json!("label"));
                schema
            } else {
                relation_schema.clone()
            };
            let mut schema = option_schema(option.kind, &binding_schema);
            schema["description"] = json!(option.description);
            properties.insert(option.name.to_owned(), schema);
            if option.required {
                required.push(option.name);
            }
        }
        json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "title": format!("{} options", self.sql_function_name), "type": "object", "additionalProperties": false, "required": required, "properties": properties})
    }
}

pub fn canonicalize_cuml_options(operation: CumlOperation, raw: &str) -> Result<Value, GpuOptionValidationError> {
    let values = canonicalize_options(operation.descriptor().options, raw, "cuML", |option, value| {
        if matches!(
            operation,
            CumlOperation::LogisticRegression
                | CumlOperation::RandomForestClassifier
                | CumlOperation::RandomForestRegressor
        ) && option.name == "predict"
            && value.get("label").is_some()
        {
            return Err(GpuOptionValidationError::new(
                "predict_relation_has_label",
                "cuML predict relation must omit label",
            ));
        }
        validate_dense_relation_binding_value(
            "cuML",
            option,
            value,
            matches!(
                operation,
                CumlOperation::LogisticRegression
                    | CumlOperation::RandomForestClassifier
                    | CumlOperation::RandomForestRegressor
            ) && option.name == "training",
        )
    })?;
    if operation == CumlOperation::Umap && values["min_dist"].as_f64() > values["spread"].as_f64() {
        return Err(GpuOptionValidationError::new("missing_or_invalid_option", "cuML min_dist must not exceed spread")
            .for_argument("min_dist"));
    }
    Ok(Value::Object(values))
}

/// ML algorithm identity carried by the native IR.
///
/// Ref: https://docs.nvidia.com/cuml/latest/api/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlAlgorithm {
    /// <https://docs.nvidia.com/cuml/latest/api/generated/cuml.decomposition.TruncatedSVD/>
    Tsvd,
    /// <https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.hdbscan.HDBSCAN/>
    Hdbscan,
    /// <https://docs.nvidia.com/cuml/latest/api/generated/cuml.manifold.UMAP/>
    Umap,
    /// <https://docs.nvidia.com/cuml/latest/api/generated/cuml.linear_model.LogisticRegression/>
    LogisticRegression,
    /// <https://docs.nvidia.com/cuml/latest/api/generated/cuml.ensemble.RandomForestClassifier/>
    RandomForestClassifier,
    /// <https://docs.nvidia.com/cuml/latest/api/generated/cuml.ensemble.RandomForestRegressor/>
    RandomForestRegressor,
    /// <https://docs.nvidia.com/nvforest/latest/>
    ForestPredict,
}

impl MlAlgorithm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tsvd => "tsvd",
            Self::Hdbscan => "hdbscan",
            Self::Umap => "umap",
            Self::LogisticRegression => "logistic_regression",
            Self::RandomForestClassifier => "random_forest_classifier",
            Self::RandomForestRegressor => "random_forest_regressor",
            Self::ForestPredict => "forest_predict",
        }
    }
}

impl fmt::Display for MlAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
