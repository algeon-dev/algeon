//! DataFusion-free vector algorithm IR.
//!
//! The contracts in this module deliberately describe relational inputs,
//! algorithm parameters, and result schemas without depending on cuVS. The
//! optional cuVS executor is selected later by the runtime.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use super::*;

/// Public SQL names for the query-local cuVS operations.
pub const CUVS_BRUTE_FORCE_KNN_FUNCTION: &str = "cuvs_brute_force_knn";
pub const CUVS_KMEANS_FUNCTION: &str = "cuvs_kmeans";
pub const CUVS_PCA_FUNCTION: &str = "cuvs_pca";

/// Scalar ID types accepted by every cuVS relational operation.
///
/// This is deliberately narrower than the set of Arrow integer types. Native
/// cuVS output/gather paths are only defined for these five stable SQL types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CuvsIdType {
    Int32,
    Int64,
    Utf8,
    LargeUtf8,
    Utf8View,
}

impl CuvsIdType {
    pub fn from_data_type(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Int32 => Some(Self::Int32),
            DataType::Int64 => Some(Self::Int64),
            DataType::Utf8 => Some(Self::Utf8),
            DataType::LargeUtf8 => Some(Self::LargeUtf8),
            DataType::Utf8View => Some(Self::Utf8View),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::Utf8 => "Utf8",
            Self::LargeUtf8 => "LargeUtf8",
            Self::Utf8View => "Utf8View",
        }
    }
}

/// Stable order used in generated descriptions and validation diagnostics.
pub const CUVS_ID_TYPES: &[CuvsIdType] =
    &[CuvsIdType::Int32, CuvsIdType::Int64, CuvsIdType::Utf8, CuvsIdType::LargeUtf8, CuvsIdType::Utf8View];

/// SQL relation argument roles in the canonical operation registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CuvsRelationRole {
    Dataset,
    Queries,
    Input,
}

impl CuvsRelationRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dataset => "dataset",
            Self::Queries => "queries",
            Self::Input => "input",
        }
    }
}

/// Only relation-valued subqueries are admitted by the SQL surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CuvsRelationArgumentSyntax {
    ParenthesizedSubquery,
}

/// Stable relational output contract selected by an operation descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CuvsOutputContract {
    BruteForceKnn,
    KMeansAssignments,
    PcaComponents,
}

/// Canonical scalar option metadata used by JSON schema generation and value
/// validation in the DataFusion adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CuvsOptionKind {
    RelationBinding,
    PositiveInteger { maximum: u64, default: Option<u64> },
    NonnegativeNumber { default: f64 },
    Boolean { default: Option<bool> },
    Enum { values: &'static [&'static str], default: Option<&'static str> },
}

impl CuvsOptionKind {
    pub const fn default_u64(self) -> Option<u64> {
        match self {
            Self::PositiveInteger { default, .. } => default,
            _ => None,
        }
    }

    pub const fn default_f64(self) -> Option<f64> {
        match self {
            Self::NonnegativeNumber { default } => Some(default),
            _ => None,
        }
    }

    pub const fn default_bool(self) -> Option<bool> {
        match self {
            Self::Boolean { default } => default,
            _ => None,
        }
    }

    pub const fn default_str(self) -> Option<&'static str> {
        match self {
            Self::Enum { default, .. } => default,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CuvsOptionDefinition {
    pub name: &'static str,
    pub required: bool,
    pub kind: CuvsOptionKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CuvsOptionValidationError {
    code: &'static str,
    message: String,
}

impl CuvsOptionValidationError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// One operation entry drives SQL names, relation shape, option metadata, and
/// output-schema selection. Keep operation-specific facts in this registry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CuvsOperationDescriptor {
    pub operation: CuvsOperation,
    pub sql_function_name: &'static str,
    pub relation_roles: &'static [CuvsRelationRole],
    pub relation_argument_syntax: CuvsRelationArgumentSyntax,
    pub output_contract: CuvsOutputContract,
    pub options: &'static [CuvsOptionDefinition],
}

fn cuvs_vector_relation_binding_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "vector"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "vector": {
                "oneOf": [
                    {
                        "type": "object", "additionalProperties": false,
                        "required": ["columns"],
                        "properties": {"columns": {"type": "array", "minItems": 1, "uniqueItems": true, "items": {"type": "string", "minLength": 1}}}
                    },
                    {
                        "type": "object", "additionalProperties": false,
                        "required": ["column"],
                        "properties": {"column": {"type": "string", "minLength": 1}}
                    }
                ]
            }
        }
    })
}

const KNN_RELATIONS: &[CuvsRelationRole] = &[CuvsRelationRole::Dataset, CuvsRelationRole::Queries];
const INPUT_RELATION: &[CuvsRelationRole] = &[CuvsRelationRole::Input];
const KNN_METRICS: &[&str] = &["l2_expanded", "l2_sqrt_expanded", "cosine", "inner_product"];
const KMEANS_METRICS: &[&str] = &["l2_expanded", "l2_sqrt_expanded"];
const KMEANS_INIT: &[&str] = &["kmeans++", "random"];
const PCA_SOLVERS: &[&str] = &["cov_eig_dq", "cov_eig_jacobi"];

const KNN_OPTIONS: &[CuvsOptionDefinition] = &[
    CuvsOptionDefinition { name: "dataset", required: true, kind: CuvsOptionKind::RelationBinding },
    CuvsOptionDefinition { name: "queries", required: true, kind: CuvsOptionKind::RelationBinding },
    CuvsOptionDefinition {
        name: "k",
        required: true,
        kind: CuvsOptionKind::PositiveInteger { maximum: u32::MAX as u64, default: None },
    },
    CuvsOptionDefinition {
        name: "metric",
        required: true,
        kind: CuvsOptionKind::Enum { values: KNN_METRICS, default: None },
    },
];
const KMEANS_OPTIONS: &[CuvsOptionDefinition] = &[
    CuvsOptionDefinition { name: "input", required: true, kind: CuvsOptionKind::RelationBinding },
    CuvsOptionDefinition {
        name: "n_clusters",
        required: true,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: None },
    },
    CuvsOptionDefinition {
        name: "metric",
        required: false,
        kind: CuvsOptionKind::Enum { values: KMEANS_METRICS, default: Some("l2_expanded") },
    },
    CuvsOptionDefinition {
        name: "max_iter",
        required: false,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: Some(100) },
    },
    CuvsOptionDefinition { name: "tol", required: false, kind: CuvsOptionKind::NonnegativeNumber { default: 0.0001 } },
    CuvsOptionDefinition {
        name: "n_init",
        required: false,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: Some(1) },
    },
    CuvsOptionDefinition {
        name: "init",
        required: false,
        kind: CuvsOptionKind::Enum { values: KMEANS_INIT, default: Some("kmeans++") },
    },
];
const PCA_OPTIONS: &[CuvsOptionDefinition] = &[
    CuvsOptionDefinition { name: "input", required: true, kind: CuvsOptionKind::RelationBinding },
    CuvsOptionDefinition {
        name: "n_components",
        required: true,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: None },
    },
    CuvsOptionDefinition { name: "whiten", required: false, kind: CuvsOptionKind::Boolean { default: Some(false) } },
    CuvsOptionDefinition {
        name: "solver",
        required: false,
        kind: CuvsOptionKind::Enum { values: PCA_SOLVERS, default: Some("cov_eig_dq") },
    },
    CuvsOptionDefinition { name: "tol", required: false, kind: CuvsOptionKind::NonnegativeNumber { default: 0.0 } },
    CuvsOptionDefinition {
        name: "n_iterations",
        required: false,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: Some(15) },
    },
    CuvsOptionDefinition {
        name: "flip_signs_based_on_u",
        required: false,
        kind: CuvsOptionKind::Boolean { default: Some(false) },
    },
];

/// Operation identity used by native IR and the SQL adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CuvsOperation {
    BruteForceKnn,
    KMeans,
    Pca,
}

impl CuvsOperation {
    pub fn all() -> impl ExactSizeIterator<Item = Self> + Clone {
        CUVS_OPERATION_REGISTRY.iter().map(|descriptor| descriptor.operation)
    }

    pub fn descriptor(self) -> &'static CuvsOperationDescriptor {
        CUVS_OPERATION_REGISTRY
            .iter()
            .find(|descriptor| descriptor.operation == self)
            .expect("cuVS operation registry is missing an operation variant")
    }

    pub fn from_sql_function_name(name: &str) -> Option<Self> {
        CUVS_OPERATION_REGISTRY.iter().find(|descriptor| descriptor.sql_function_name == name).map(|d| d.operation)
    }
}

pub const CUVS_OPERATION_REGISTRY: &[CuvsOperationDescriptor] = &[
    CuvsOperationDescriptor {
        operation: CuvsOperation::BruteForceKnn,
        sql_function_name: CUVS_BRUTE_FORCE_KNN_FUNCTION,
        relation_roles: KNN_RELATIONS,
        relation_argument_syntax: CuvsRelationArgumentSyntax::ParenthesizedSubquery,
        output_contract: CuvsOutputContract::BruteForceKnn,
        options: KNN_OPTIONS,
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::KMeans,
        sql_function_name: CUVS_KMEANS_FUNCTION,
        relation_roles: INPUT_RELATION,
        relation_argument_syntax: CuvsRelationArgumentSyntax::ParenthesizedSubquery,
        output_contract: CuvsOutputContract::KMeansAssignments,
        options: KMEANS_OPTIONS,
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::Pca,
        sql_function_name: CUVS_PCA_FUNCTION,
        relation_roles: INPUT_RELATION,
        relation_argument_syntax: CuvsRelationArgumentSyntax::ParenthesizedSubquery,
        output_contract: CuvsOutputContract::PcaComponents,
        options: PCA_OPTIONS,
    },
];

impl CuvsOperationDescriptor {
    pub fn option(&self, name: &str) -> Option<&'static CuvsOptionDefinition> {
        self.options.iter().find(|option| option.name == name)
    }

    pub fn options_json_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for option in self.options {
            properties.insert(option.name.to_string(), option_schema(option.kind));
            if option.required {
                required.push(Value::String(option.name.to_string()));
            }
        }
        json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": format!("{} options", self.sql_function_name),
            "type": "object",
            "additionalProperties": false,
            "required": required,
            "properties": properties,
        })
    }
}

fn option_schema(kind: CuvsOptionKind) -> Value {
    match kind {
        CuvsOptionKind::RelationBinding => cuvs_vector_relation_binding_schema(),
        CuvsOptionKind::PositiveInteger { maximum, default } => {
            let mut value = json!({"type": "integer", "minimum": 1, "maximum": maximum});
            if let Some(default) = default {
                value["default"] = json!(default);
            }
            value
        },
        CuvsOptionKind::NonnegativeNumber { default } => json!({"type": "number", "minimum": 0.0, "default": default}),
        CuvsOptionKind::Boolean { default } => {
            let mut value = json!({"type": "boolean"});
            if let Some(default) = default {
                value["default"] = json!(default);
            }
            value
        },
        CuvsOptionKind::Enum { values, default } => {
            let mut value = json!({"type": "string", "enum": values});
            if let Some(default) = default {
                value["default"] = json!(default);
            }
            value
        },
    }
}

/// Validate and canonicalize the scalar portion of an options JSON object.
/// Relation binding field names/types are checked by the SQL adapter against
/// its child schema, but presence and object shape are enforced here.
pub fn canonicalize_cuvs_options(operation: CuvsOperation, raw: &str) -> Result<Value, CuvsOptionValidationError> {
    let parsed: Value = serde_json::from_str(raw)
        .map_err(|_| option_error("invalid_options_json", "cuVS options must be valid JSON"))?;
    let mut object = parsed
        .as_object()
        .cloned()
        .ok_or_else(|| option_error("options_not_object", "cuVS options must be a JSON object"))?;
    let descriptor = operation.descriptor();
    if let Some(unknown) = object.keys().find(|key| descriptor.option(key).is_none()) {
        return Err(option_error("unknown_option", format!("unknown cuVS option '{unknown}'")));
    }
    for option in descriptor.options {
        let Some(value) = object.get_mut(option.name) else {
            if option.required {
                return Err(option_error(
                    "missing_or_invalid_option",
                    format!("cuVS option '{}' is required", option.name),
                ));
            }
            let default = match option.kind {
                CuvsOptionKind::PositiveInteger { default: Some(value), .. } => json!(value),
                CuvsOptionKind::NonnegativeNumber { default } => json!(default),
                CuvsOptionKind::Boolean { default: Some(value) } => json!(value),
                CuvsOptionKind::Enum { default: Some(value), .. } => json!(value),
                _ => continue,
            };
            object.insert(option.name.to_string(), default);
            continue;
        };
        validate_option_value(option, value)?;
    }
    Ok(Value::Object(object))
}

fn validate_option_value(option: &CuvsOptionDefinition, value: &Value) -> Result<(), CuvsOptionValidationError> {
    let invalid = || option_error("missing_or_invalid_option", format!("invalid cuVS option '{}'", option.name));
    match option.kind {
        CuvsOptionKind::RelationBinding => {
            validate_relation_binding_value(option, value)?;
        },
        CuvsOptionKind::PositiveInteger { maximum, .. } => {
            if value.as_u64().filter(|value| *value > 0 && *value <= maximum).is_none() {
                return Err(invalid());
            }
        },
        CuvsOptionKind::NonnegativeNumber { .. } => {
            if value.as_f64().filter(|value| value.is_finite() && *value >= 0.0).is_none() {
                return Err(invalid());
            }
        },
        CuvsOptionKind::Boolean { .. } => {
            if !value.is_boolean() {
                return Err(invalid());
            }
        },
        CuvsOptionKind::Enum { values, .. } => {
            if value.as_str().is_none_or(|value| !values.contains(&value)) {
                return Err(invalid());
            }
        },
    }
    Ok(())
}

fn validate_relation_binding_value(
    option: &CuvsOptionDefinition,
    value: &Value,
) -> Result<(), CuvsOptionValidationError> {
    let invalid = || option_error("missing_or_invalid_option", format!("invalid cuVS option '{}'", option.name));
    let binding = value.as_object().ok_or_else(invalid)?;
    if let Some(unknown) = binding.keys().find(|key| !matches!(key.as_str(), "id" | "vector")) {
        return Err(option_error("unknown_option", format!("unknown cuVS option '{unknown}' in {}", option.name)));
    }
    if binding.get("id").and_then(Value::as_str).filter(|value| !value.is_empty()).is_none() {
        return Err(option_error(
            "missing_or_invalid_option",
            format!("cuVS option '{}.id' must be a non-empty string", option.name),
        ));
    }
    let vector = binding.get("vector").and_then(Value::as_object).ok_or_else(|| {
        option_error("missing_or_invalid_option", format!("cuVS option '{}.vector' must be an object", option.name))
    })?;
    if let Some(unknown) = vector.keys().find(|key| !matches!(key.as_str(), "columns" | "column")) {
        return Err(option_error(
            "unknown_option",
            format!("unknown cuVS option '{unknown}' in {}.vector", option.name),
        ));
    }

    match (vector.get("columns"), vector.get("column")) {
        (Some(columns), None) => {
            let columns = columns.as_array().filter(|columns| !columns.is_empty()).ok_or_else(|| {
                option_error(
                    "invalid_vector_binding",
                    format!("cuVS option '{}.vector.columns' must be a non-empty string array", option.name),
                )
            })?;
            let mut unique = BTreeSet::new();
            for column in columns {
                let column = column.as_str().filter(|column| !column.is_empty()).ok_or_else(|| {
                    option_error(
                        "invalid_vector_binding",
                        format!("cuVS option '{}.vector.columns' entries must be non-empty strings", option.name),
                    )
                })?;
                if !unique.insert(column) {
                    return Err(option_error(
                        "duplicate_vector_column",
                        format!("cuVS option '{}.vector.columns' contains duplicate field '{column}'", option.name),
                    ));
                }
            }
        },
        (None, Some(column)) => {
            if column.as_str().filter(|column| !column.is_empty()).is_none() {
                return Err(option_error(
                    "invalid_vector_binding",
                    format!("cuVS option '{}.vector.column' must be a non-empty string", option.name),
                ));
            }
        },
        _ => {
            return Err(option_error(
                "invalid_vector_binding",
                format!("cuVS option '{}.vector' must contain exactly one of columns or column", option.name),
            ));
        },
    }
    Ok(())
}

fn option_error(code: &'static str, message: impl Into<String>) -> CuvsOptionValidationError {
    CuvsOptionValidationError { code, message: message.into() }
}

/// Strict/adaptive bounded memory boundary blocked on lower-level cuVS sizing.
pub const VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY: &str = "vector_algorithm_peak_memory";

/// Lower-level contract required before bounded vector execution can be admitted.
pub const VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT: &str = "cuvs_vector_algorithm_peak_memory_preflight";

/// Vector algorithm identity carried by the native IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorAlgorithm {
    /// Exact brute-force k-nearest-neighbor search.
    BruteForceKnn,
    /// K-means fit followed by row-label prediction.
    KMeans,
    /// Principal-component fit and transform.
    Pca,
}

impl VectorAlgorithm {
    /// Stable diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BruteForceKnn => "brute_force_knn",
            Self::KMeans => "kmeans",
            Self::Pca => "pca",
        }
    }
}

impl fmt::Display for VectorAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Row identity emitted alongside vector algorithm results.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VectorRowIdBinding {
    /// Use the zero-based row ordinal of the evaluated relation.
    RowIndex,
    /// Use a non-null integral column from the evaluated relation.
    Column(String),
}

impl VectorRowIdBinding {
    /// Construct a column-backed row identity.
    pub fn column(name: impl Into<String>) -> Self {
        Self::Column(name.into())
    }

    /// Stable diagnostic label.
    pub fn diagnostic_label(&self) -> &str {
        match self {
            Self::RowIndex => "row_index",
            Self::Column(name) => name,
        }
    }
}

/// Binding from one native relation to a dense vector matrix.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CuvsDenseVectorBinding {
    /// Separate non-null `Float32` columns packed in the listed order.
    Columns(Vec<String>),
    /// One non-null `FixedSizeList<Float32, D>`, `List<Float32>`, or
    /// `LargeList<Float32>` column.
    VectorColumn(String),
}

impl CuvsDenseVectorBinding {
    pub fn columns(columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::Columns(columns.into_iter().map(Into::into).collect())
    }

    pub fn vector_column(column: impl Into<String>) -> Self {
        Self::VectorColumn(column.into())
    }

    pub fn display_label(&self) -> String {
        match self {
            Self::Columns(columns) => columns.join("|"),
            Self::VectorColumn(column) => format!("[{column}]"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorRelationBinding {
    /// Native relation that supplies row ids and feature columns.
    pub input: NodeId,
    /// Row identity emitted by relational results.
    pub row_id: VectorRowIdBinding,
    /// Ordered dense vector binding.
    pub vector: CuvsDenseVectorBinding,
}

impl VectorRelationBinding {
    /// Construct a vector relation binding.
    pub fn new(
        input: NodeId,
        row_id: VectorRowIdBinding,
        feature_columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self { input, row_id, vector: CuvsDenseVectorBinding::columns(feature_columns) }
    }

    /// Construct a binding backed by one list/vector column.
    pub fn new_vector_column(input: NodeId, row_id: VectorRowIdBinding, column: impl Into<String>) -> Self {
        Self { input, row_id, vector: CuvsDenseVectorBinding::vector_column(column) }
    }
}

/// Distance metric used by exact kNN and K-means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorDistanceMetric {
    /// Expanded squared L2 distance.
    L2Expanded,
    /// Square root of expanded L2 distance.
    L2SqrtExpanded,
    /// Expanded cosine distance.
    CosineExpanded,
    /// Inner-product distance.
    InnerProduct,
}

impl VectorDistanceMetric {
    /// Stable diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L2Expanded => "l2_expanded",
            Self::L2SqrtExpanded => "l2_sqrt_expanded",
            Self::CosineExpanded => "cosine_expanded",
            Self::InnerProduct => "inner_product",
        }
    }
}

impl fmt::Display for VectorDistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Exact brute-force kNN parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorBruteForceKnnParams {
    /// Indexed dataset relation.
    pub dataset: VectorRelationBinding,
    /// Query-vector relation.
    pub queries: VectorRelationBinding,
    /// Number of neighbors emitted per query row.
    pub k: usize,
    /// Distance metric.
    pub metric: VectorDistanceMetric,
}

impl VectorBruteForceKnnParams {
    /// Construct exact brute-force kNN parameters.
    pub fn new(
        dataset: VectorRelationBinding,
        queries: VectorRelationBinding,
        k: usize,
        metric: VectorDistanceMetric,
    ) -> Self {
        Self { dataset, queries, k, metric }
    }
}

/// K-means centroid initialization policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorKMeansInit {
    /// K-means++ initialization.
    KMeansPlusPlus,
    /// Random initialization.
    Random,
    /// Initial centroid vectors supplied by a relation binding.
    Relation,
}

impl VectorKMeansInit {
    /// Stable diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::KMeansPlusPlus => "kmeans_plus_plus",
            Self::Random => "random",
            Self::Relation => "relation",
        }
    }
}

impl fmt::Display for VectorKMeansInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// K-means fit/predict parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorKMeansParams {
    /// Training and prediction input relation.
    pub input: VectorRelationBinding,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Maximum fit iterations.
    pub max_iterations: u32,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Number of independent initializations.
    pub n_init: u32,
    /// Centroid initialization policy.
    pub init: VectorKMeansInit,
    /// Optional relation-backed initial centroids.
    pub initial_centroids: Option<VectorRelationBinding>,
    /// Distance metric used for cluster assignment.
    pub metric: VectorKMeansMetric,
}

impl VectorKMeansParams {
    /// Construct K-means parameters with stable defaults.
    pub fn new(input: VectorRelationBinding, n_clusters: usize) -> Self {
        let descriptor = CuvsOperation::KMeans.descriptor();
        Self {
            input,
            n_clusters,
            max_iterations: descriptor
                .option("max_iter")
                .and_then(|option| option.kind.default_u64())
                .expect("canonical KMeans max_iter default") as u32,
            tolerance: descriptor
                .option("tol")
                .and_then(|option| option.kind.default_f64())
                .expect("canonical KMeans tol default"),
            n_init: descriptor
                .option("n_init")
                .and_then(|option| option.kind.default_u64())
                .expect("canonical KMeans n_init default") as u32,
            init: match descriptor.option("init").and_then(|option| option.kind.default_str()) {
                Some("random") => VectorKMeansInit::Random,
                Some("kmeans++") => VectorKMeansInit::KMeansPlusPlus,
                _ => panic!("canonical KMeans init default"),
            },
            initial_centroids: None,
            metric: match descriptor.option("metric").and_then(|option| option.kind.default_str()) {
                Some("l2_sqrt_expanded") => VectorKMeansMetric::L2SqrtExpanded,
                Some("l2_expanded") => VectorKMeansMetric::L2Expanded,
                _ => panic!("canonical KMeans metric default"),
            },
        }
    }

    /// Use relation-backed initial centroids.
    pub fn with_initial_centroids(mut self, centroids: VectorRelationBinding) -> Self {
        self.init = VectorKMeansInit::Relation;
        self.initial_centroids = Some(centroids);
        self
    }
}

/// K-means admits only the native L2 family; this is intentionally not the
/// broader kNN distance enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorKMeansMetric {
    L2Expanded,
    L2SqrtExpanded,
}

impl VectorKMeansMetric {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L2Expanded => "l2_expanded",
            Self::L2SqrtExpanded => "l2_sqrt_expanded",
        }
    }
}

impl fmt::Display for VectorKMeansMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// PCA solver selected by the SQL options contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorPcaSolver {
    CovEigDq,
    CovEigJacobi,
}

impl VectorPcaSolver {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CovEigDq => "cov_eig_dq",
            Self::CovEigJacobi => "cov_eig_jacobi",
        }
    }
}

impl fmt::Display for VectorPcaSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// PCA fit-transform parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorPcaParams {
    /// Input relation to fit and transform.
    pub input: VectorRelationBinding,
    /// Number of output principal components.
    pub n_components: usize,
    /// Whether transformed components are whitened.
    pub whiten: bool,
    /// Covariance eigensolver.
    pub solver: VectorPcaSolver,
    /// Nonnegative eigensolver tolerance.
    pub tolerance: f64,
    /// Maximum eigensolver iterations.
    pub n_iterations: u32,
    /// Native sign-fixing option.
    pub flip_signs_based_on_u: bool,
}

impl VectorPcaParams {
    /// Construct PCA fit-transform parameters.
    pub fn new(input: VectorRelationBinding, n_components: usize) -> Self {
        let descriptor = CuvsOperation::Pca.descriptor();
        Self {
            input,
            n_components,
            whiten: descriptor.option("whiten").and_then(|option| option.kind.default_bool()).unwrap_or(false),
            solver: match descriptor.option("solver").and_then(|option| option.kind.default_str()) {
                Some("cov_eig_jacobi") => VectorPcaSolver::CovEigJacobi,
                Some("cov_eig_dq") => VectorPcaSolver::CovEigDq,
                _ => panic!("canonical PCA solver default"),
            },
            tolerance: descriptor.option("tol").and_then(|option| option.kind.default_f64()).unwrap_or(0.0),
            n_iterations: descriptor
                .option("n_iterations")
                .and_then(|option| option.kind.default_u64())
                .expect("canonical PCA n_iterations default") as u32,
            flip_signs_based_on_u: descriptor
                .option("flip_signs_based_on_u")
                .and_then(|option| option.kind.default_bool())
                .unwrap_or(false),
        }
    }
}

/// Algorithm-specific vector parameter payload.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorAlgorithmParams {
    /// Exact brute-force kNN parameters.
    BruteForceKnn(VectorBruteForceKnnParams),
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
                | (Self::KMeans(_), VectorAlgorithm::KMeans)
                | (Self::Pca(_), VectorAlgorithm::Pca)
        )
    }

    /// Visit relation bindings in stable semantic execution order.
    pub fn relation_bindings(&self) -> Box<dyn Iterator<Item = &VectorRelationBinding> + '_> {
        match self {
            Self::BruteForceKnn(params) => Box::new([&params.dataset, &params.queries].into_iter()),
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

/// DataFusion-free vector algorithm specification.
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

/// Stable relational result schema for brute-force kNN.
pub fn vector_brute_force_knn_output_schema(query_id_type: DataType, neighbor_id_type: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("query_ordinal", DataType::UInt64, false),
        Field::new("query_id", query_id_type, false),
        Field::new("neighbor_ordinal", DataType::UInt64, false),
        Field::new("neighbor_id", neighbor_id_type, false),
        Field::new("rank", DataType::UInt32, false),
        Field::new("distance", DataType::Float32, false),
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
