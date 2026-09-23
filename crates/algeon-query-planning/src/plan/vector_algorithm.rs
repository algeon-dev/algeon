//! DataFusion-free vector algorithm IR.
//!
//! The contracts in this module deliberately describe relational inputs,
//! algorithm parameters, and result schemas without depending on cuVS. The
//! optional cuVS executor is selected later by the runtime.

use std::{collections::BTreeSet, fmt, sync::Arc};

use serde_json::{Value, json};

use super::{DataType, Field, NodeId, Schema, SchemaRef};

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

/// Vector element types accepted by cuVS operation descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CuvsVectorElementType {
    Float32,
    Float64,
    Int8,
    UInt8,
}

impl CuvsVectorElementType {
    pub fn from_data_type(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Float32 => Some(Self::Float32),
            DataType::Float64 => Some(Self::Float64),
            DataType::Int8 => Some(Self::Int8),
            DataType::UInt8 => Some(Self::UInt8),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "Float32",
            Self::Float64 => "Float64",
            Self::Int8 => "Int8",
            Self::UInt8 => "UInt8",
        }
    }
}

/// SQL relation argument roles in the canonical operation registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CuvsRelationRole {
    Dataset,
    Queries,
    Input,
    X,
    Y,
}

impl CuvsRelationRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dataset => "dataset",
            Self::Queries => "queries",
            Self::Input => "input",
            Self::X => "x",
            Self::Y => "y",
        }
    }

    /// Canonical documentation description for this logical relation role.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Dataset => "Dense-vector rows searched by the exact kNN operation.",
            Self::Queries => "Dense-vector query rows matched against the evaluated dataset.",
            Self::Input => "Dense-vector rows consumed by the fit-and-transform operation.",
            Self::X => "Dense-vector rows on the first axis of the pairwise distance result.",
            Self::Y => "Dense-vector rows on the second axis of the pairwise distance result.",
        }
    }
}

/// Stable relational output contract selected by an operation descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CuvsOutputContract {
    KnnNeighbors,
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
    /// User-facing semantics projected through the unified GPU descriptor.
    pub description: &'static str,
}

/// One documented output field in a cuVS relational result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CuvsOutputFieldDefinition {
    pub name: &'static str,
    pub data_type: &'static str,
    pub nullable: bool,
    /// Literal option that determines how many physical output fields exist.
    pub repeats: Option<&'static str>,
    pub description: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CuvsOptionValidationError {
    code: &'static str,
    message: String,
    argument: Option<&'static str>,
}

impl CuvsOptionValidationError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    /// The declared option this error concerns, when it is attributable to one.
    pub fn argument(&self) -> Option<&'static str> {
        self.argument
    }

    fn for_argument(mut self, argument: &'static str) -> Self {
        self.argument = Some(argument);
        self
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
    /// Human-readable function title; SQL names remain canonical identifiers.
    pub display_name: &'static str,
    /// Short catalog summary.
    pub summary: &'static str,
    /// Concrete selection guidance for the generated reference page.
    pub use_this_when: &'static str,
    /// Relation arguments, each passed as a parenthesized subquery; the SQL
    /// surface admits no other relation syntax.
    pub relation_roles: &'static [CuvsRelationRole],
    /// Supported dense-vector element types for every relation binding.
    pub vector_element_support: &'static [CuvsVectorElementType],
    pub output_contract: CuvsOutputContract,
    pub options: &'static [CuvsOptionDefinition],
    pub output_fields: &'static [CuvsOutputFieldDefinition],
    /// Generic SQL that demonstrates the real execution syntax, not only the
    /// metadata-validation envelope.
    pub quickstart_sql: &'static str,
    /// Static and data-dependent lifecycle constraints that apply to this
    /// operation in addition to the common relation/validation contract.
    pub limitations: &'static [&'static str],
}

fn cuvs_vector_relation_binding_schema(element_support: &[CuvsVectorElementType]) -> Value {
    let elements = element_support.iter().map(|element| element.as_str()).collect::<Vec<_>>().join(", ");
    let list_types = element_support
        .iter()
        .map(|element| {
            format!(
                "FixedSizeList<{}, D>, List<{}>, or LargeList<{}>",
                element.as_str(),
                element.as_str(),
                element.as_str()
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "vector"],
        "properties": {
            "id": {
                "type": "string",
                "minLength": 1,
                "description": "Non-null logical row ID column. IDs may repeat; result ordinals disambiguate physical rows."
            },
            "vector": {
                "description": "Choose exactly one dense-vector binding shape.",
                "oneOf": [
                    {
                        "type": "object", "additionalProperties": false,
                        "required": ["columns"],
                        "properties": {
                            "columns": {
                                "type": "array", "minItems": 1, "uniqueItems": true,
                                "description": format!("Ordered, unique non-null {elements} feature columns; order defines vector dimensions."),
                                "items": {"type": "string", "minLength": 1}
                            }
                        }
                    },
                    {
                        "type": "object", "additionalProperties": false,
                        "required": ["column"],
                        "properties": {
                            "column": {
                                "type": "string", "minLength": 1,
                                "description": format!("One non-null {list_types} column.")
                            }
                        }
                    }
                ]
            }
        }
    })
}

const FLOAT32_VECTOR_SUPPORT: &[CuvsVectorElementType] = &[CuvsVectorElementType::Float32];
const KNN_RELATIONS: &[CuvsRelationRole] = &[CuvsRelationRole::Dataset, CuvsRelationRole::Queries];
const INPUT_RELATION: &[CuvsRelationRole] = &[CuvsRelationRole::Input];
const KNN_METRICS: &[&str] = &["l2_expanded", "l2_sqrt_expanded", "cosine", "inner_product"];
const KMEANS_METRICS: &[&str] = &["l2_expanded", "l2_sqrt_expanded"];
const KMEANS_INIT: &[&str] = &["kmeans++", "random"];
const PCA_SOLVERS: &[&str] = &["cov_eig_dq", "cov_eig_jacobi"];

const KNN_OPTIONS: &[CuvsOptionDefinition] = &[
    CuvsOptionDefinition {
        name: "dataset",
        required: true,
        kind: CuvsOptionKind::RelationBinding,
        description: "Names the dataset ID column and one dense-vector binding shape.",
    },
    CuvsOptionDefinition {
        name: "queries",
        required: true,
        kind: CuvsOptionKind::RelationBinding,
        description: "Names the query ID column and one dense-vector binding shape.",
    },
    CuvsOptionDefinition {
        name: "k",
        required: true,
        kind: CuvsOptionKind::PositiveInteger { maximum: u32::MAX as u64, default: None },
        description: "Number of neighbors returned for each evaluated query row.",
    },
    CuvsOptionDefinition {
        name: "metric",
        required: true,
        kind: CuvsOptionKind::Enum { values: KNN_METRICS, default: None },
        description: "Distance or score metric. Distance metrics rank lower values first; inner product ranks higher values first.",
    },
];
const KMEANS_OPTIONS: &[CuvsOptionDefinition] = &[
    CuvsOptionDefinition {
        name: "input",
        required: true,
        kind: CuvsOptionKind::RelationBinding,
        description: "Names the input ID column and one dense-vector binding shape.",
    },
    CuvsOptionDefinition {
        name: "n_clusters",
        required: true,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: None },
        description: "Number of clusters fitted for this one statement.",
    },
    CuvsOptionDefinition {
        name: "metric",
        required: false,
        kind: CuvsOptionKind::Enum { values: KMEANS_METRICS, default: Some("l2_expanded") },
        description: "L2 distance form used while fitting and assigning the current input relation.",
    },
    CuvsOptionDefinition {
        name: "max_iter",
        required: false,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: Some(100) },
        description: "Maximum fitting iterations for this invocation.",
    },
    CuvsOptionDefinition {
        name: "tol",
        required: false,
        kind: CuvsOptionKind::NonnegativeNumber { default: 0.0001 },
        description: "Non-negative convergence tolerance used by the fitted model.",
    },
    CuvsOptionDefinition {
        name: "n_init",
        required: false,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: Some(1) },
        description: "Number of initialization attempts performed within this call.",
    },
    CuvsOptionDefinition {
        name: "init",
        required: false,
        kind: CuvsOptionKind::Enum { values: KMEANS_INIT, default: Some("kmeans++") },
        description: "Centroid initialization strategy for this fitted model.",
    },
];
const PCA_OPTIONS: &[CuvsOptionDefinition] = &[
    CuvsOptionDefinition {
        name: "input",
        required: true,
        kind: CuvsOptionKind::RelationBinding,
        description: "Names the input ID column and one dense-vector binding shape.",
    },
    CuvsOptionDefinition {
        name: "n_components",
        required: true,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: None },
        description: "Number of principal-component columns emitted as pc_0 through pc_<n_components - 1>.",
    },
    CuvsOptionDefinition {
        name: "whiten",
        required: false,
        kind: CuvsOptionKind::Boolean { default: Some(false) },
        description: "Whether the query-local transformed components are whitened.",
    },
    CuvsOptionDefinition {
        name: "solver",
        required: false,
        kind: CuvsOptionKind::Enum { values: PCA_SOLVERS, default: Some("cov_eig_dq") },
        description: "Covariance eigensolver used for this fit-transform call.",
    },
    CuvsOptionDefinition {
        name: "tol",
        required: false,
        kind: CuvsOptionKind::NonnegativeNumber { default: 0.0 },
        description: "Non-negative eigensolver tolerance.",
    },
    CuvsOptionDefinition {
        name: "n_iterations",
        required: false,
        kind: CuvsOptionKind::PositiveInteger { maximum: i32::MAX as u64, default: Some(15) },
        description: "Maximum eigensolver iterations for this invocation.",
    },
    CuvsOptionDefinition {
        name: "flip_signs_based_on_u",
        required: false,
        kind: CuvsOptionKind::Boolean { default: Some(false) },
        description: "Whether cuVS applies its U-based component-sign convention.",
    },
];

const KNN_OUTPUT_FIELDS: &[CuvsOutputFieldDefinition] = &[
    CuvsOutputFieldDefinition {
        name: "query_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based ordinal of the evaluated query row; it disambiguates duplicate query IDs.",
    },
    CuvsOutputFieldDefinition {
        name: "query_id",
        data_type: "same_as_queries.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the queries relation.",
    },
    CuvsOutputFieldDefinition {
        name: "neighbor_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based ordinal of the matched dataset row; it disambiguates duplicate dataset IDs.",
    },
    CuvsOutputFieldDefinition {
        name: "neighbor_id",
        data_type: "same_as_dataset.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the matched dataset row.",
    },
    CuvsOutputFieldDefinition {
        name: "rank",
        data_type: "UInt32",
        nullable: false,
        repeats: None,
        description: "One-based neighbor rank within a query. Order consumers explicitly by query_ordinal, rank.",
    },
    CuvsOutputFieldDefinition {
        name: "distance",
        data_type: "Float32",
        nullable: false,
        repeats: None,
        description: "Metric value; smaller is better for distance metrics, while inner_product prefers larger values.",
    },
];

const KMEANS_OUTPUT_FIELDS: &[CuvsOutputFieldDefinition] = &[
    CuvsOutputFieldDefinition {
        name: "row_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based ordinal of the evaluated input row; it disambiguates duplicate IDs.",
    },
    CuvsOutputFieldDefinition {
        name: "id",
        data_type: "same_as_input.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the input relation.",
    },
    CuvsOutputFieldDefinition {
        name: "cluster_id",
        data_type: "Int32",
        nullable: false,
        repeats: None,
        description: "Query-local numeric assignment label, not a stable business or topic identifier.",
    },
];

const PCA_OUTPUT_FIELDS: &[CuvsOutputFieldDefinition] = &[
    CuvsOutputFieldDefinition {
        name: "row_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based ordinal of the evaluated input row; it disambiguates duplicate IDs.",
    },
    CuvsOutputFieldDefinition {
        name: "id",
        data_type: "same_as_input.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the input relation.",
    },
    CuvsOutputFieldDefinition {
        name: "pc_<index>",
        data_type: "Float32",
        nullable: false,
        repeats: Some("n_components"),
        description: "One transformed principal-component value; the field count is set by the literal n_components option.",
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
        display_name: "Brute-force kNN",
        summary: "Exact brute-force nearest-neighbor search over dense vectors.",
        use_this_when: "Use this when SQL can first filter a candidate relation and you need exact batch reranking, not a persistent ANN index.",
        relation_roles: KNN_RELATIONS,
        vector_element_support: FLOAT32_VECTOR_SUPPORT,
        output_contract: CuvsOutputContract::KnnNeighbors,
        options: KNN_OPTIONS,
        output_fields: KNN_OUTPUT_FIELDS,
        quickstart_sql: r#"SELECT *
FROM cuvs_brute_force_knn(
  dataset => (SELECT id, d0, d1 FROM dataset_vectors),
  queries => (SELECT id, d0, d1 FROM query_vectors),
  k => 8,
  metric => 'l2_expanded'
);"#,
        limitations: &[
            "Builds an exact query-local index over the evaluated dataset relation; it does not persist an ANN index or replace a vector database.",
            "The evaluated dataset must be non-empty and k must not exceed its row count. An empty query relation may return an empty result with the stable schema.",
            "Dataset and query vector dimensions must match after both relation children are evaluated.",
        ],
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::KMeans,
        sql_function_name: CUVS_KMEANS_FUNCTION,
        display_name: "KMeans",
        summary: "K-means clustering assignments for dense vector rows.",
        use_this_when: "Use this when one statement should fit clusters over its evaluated relation and immediately join or aggregate the assignments.",
        relation_roles: INPUT_RELATION,
        vector_element_support: FLOAT32_VECTOR_SUPPORT,
        output_contract: CuvsOutputContract::KMeansAssignments,
        options: KMEANS_OPTIONS,
        output_fields: KMEANS_OUTPUT_FIELDS,
        quickstart_sql: r#"SELECT id, cluster_id
FROM cuvs_kmeans(
  input => (SELECT id, d0, d1 FROM input_vectors),
  n_clusters => 8
)
ORDER BY row_ordinal;"#,
        limitations: &[
            "Fits a model and returns assignments within one statement; it does not return a reusable model, centroids, or inertia.",
            "The evaluated input must be non-empty and n_clusters must not exceed its row count.",
            "cluster_id values are query-local labels. Do not attach permanent business meaning to their numeric values.",
        ],
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::Pca,
        sql_function_name: CUVS_PCA_FUNCTION,
        display_name: "PCA",
        summary: "Principal-component analysis transform for dense vector rows.",
        use_this_when: "Use this when one statement should fit and transform its evaluated input relation for relational export or downstream visualization.",
        relation_roles: INPUT_RELATION,
        vector_element_support: FLOAT32_VECTOR_SUPPORT,
        output_contract: CuvsOutputContract::PcaComponents,
        options: PCA_OPTIONS,
        output_fields: PCA_OUTPUT_FIELDS,
        quickstart_sql: r#"SELECT id, pc_0, pc_1
FROM cuvs_pca(
  input => (SELECT id, d0, d1 FROM input_vectors),
  n_components => 2
)
ORDER BY row_ordinal;"#,
        limitations: &[
            "Fits and transforms the evaluated input in one statement; it does not return a reusable transform, components, or explained-variance metadata.",
            "The evaluated input needs at least two rows and two dimensions, and n_components must not exceed the evaluated dimension.",
            "Component sign orientation is not a semantic label. The output schema changes with the literal n_components option.",
        ],
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
            let mut schema = option_schema(option.kind, self.vector_element_support);
            schema["description"] = Value::String(option.description.to_string());
            properties.insert(option.name.to_string(), schema);
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

fn option_schema(kind: CuvsOptionKind, element_support: &[CuvsVectorElementType]) -> Value {
    match kind {
        CuvsOptionKind::RelationBinding => cuvs_vector_relation_binding_schema(element_support),
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
                )
                .for_argument(option.name));
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
        validate_option_value(option, value).map_err(|error| error.for_argument(option.name))?;
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
    CuvsOptionValidationError { code, message: message.into(), argument: None }
}

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
    /// Separate non-null columns of one supported element type packed in the listed order.
    Columns(Vec<String>),
    /// One non-null fixed-size or variable-size list column of a supported element type.
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
            whiten: descriptor.option("whiten").and_then(|option| option.kind.default_bool()).expect("PCA whiten"),
            solver: match descriptor.option("solver").and_then(|option| option.kind.default_str()) {
                Some("cov_eig_jacobi") => VectorPcaSolver::CovEigJacobi,
                Some("cov_eig_dq") => VectorPcaSolver::CovEigDq,
                _ => panic!("canonical PCA solver default"),
            },
            tolerance: descriptor.option("tol").and_then(|option| option.kind.default_f64()).expect("PCA tolerance"),
            n_iterations: descriptor
                .option("n_iterations")
                .and_then(|option| option.kind.default_u64())
                .expect("canonical PCA n_iterations default") as u32,
            flip_signs_based_on_u: descriptor
                .option("flip_signs_based_on_u")
                .and_then(|option| option.kind.default_bool())
                .expect("PCA sign flip"),
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

    #[doc(hidden)]
    pub fn remap_relation_inputs(&mut self, mut remap: impl FnMut(NodeId) -> NodeId) {
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

/// Stable relational result schema for nearest-neighbor search.
pub fn vector_knn_output_schema(query_id_type: DataType, neighbor_id_type: DataType) -> SchemaRef {
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
