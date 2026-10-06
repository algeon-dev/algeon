//! cuVS operation definitions and stable vector algorithm identities.

use std::fmt;

use serde_json::{Map, Value, json};

use crate::plan::{
    GpuOptionDefinition, GpuOptionKind, GpuOptionValidationError, GpuOutputFieldDefinition, VectorElementType,
    gpu_function_options::{
        canonicalize_options, dense_relation_binding_schema, option_schema as gpu_option_schema,
        validate_dense_relation_binding_value,
    },
};

/// Public SQL names for the query-local cuVS operations.
pub const CUVS_BRUTE_FORCE_KNN_FUNCTION: &str = "cuvs_brute_force_knn";
pub const CUVS_IVF_FLAT_FUNCTION: &str = "cuvs_ivf_flat";
pub const CUVS_IVF_PQ_FUNCTION: &str = "cuvs_ivf_pq";
pub const CUVS_CAGRA_FUNCTION: &str = "cuvs_cagra";
pub const CUVS_KMEANS_FUNCTION: &str = "cuvs_kmeans";
pub const CUVS_PCA_FUNCTION: &str = "cuvs_pca";
pub const CUVS_PAIRWISE_DISTANCE_FUNCTION: &str = "cuvs_pairwise_distance";
pub const CUVS_TOP_K_FUNCTION: &str = "cuvs_top_k";

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
            Self::Dataset => "Dense-vector rows indexed for nearest-neighbor search.",
            Self::Queries => "Dense-vector query rows matched against the evaluated dataset.",
            Self::Input => "Dense-vector rows consumed by this operation.",
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
    PairwiseDistances,
    TopKValues,
}

/// Verified metrics for one vector element type within an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CuvsVectorElementSupport {
    pub element: VectorElementType,
    pub metrics: &'static [&'static str],
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
    pub vector_element_support: &'static [CuvsVectorElementSupport],
    pub output_contract: CuvsOutputContract,
    pub options: &'static [GpuOptionDefinition],
    pub output_fields: &'static [GpuOutputFieldDefinition],
    /// Generic SQL that demonstrates the real execution syntax, not only the
    /// metadata-validation envelope.
    pub quickstart_sql: &'static str,
    /// Static and data-dependent lifecycle constraints that apply to this
    /// operation in addition to the common relation/validation contract.
    pub limitations: &'static [&'static str],
}

const FLOAT32_VECTOR_SUPPORT: &[CuvsVectorElementSupport] =
    &[CuvsVectorElementSupport { element: VectorElementType::Float32, metrics: &[] }];
const KNN_RELATIONS: &[CuvsRelationRole] = &[CuvsRelationRole::Dataset, CuvsRelationRole::Queries];
const INPUT_RELATION: &[CuvsRelationRole] = &[CuvsRelationRole::Input];
const PAIRWISE_RELATIONS: &[CuvsRelationRole] = &[CuvsRelationRole::X, CuvsRelationRole::Y];
const TOP_K_SELECTIONS: &[&str] = &["min", "max"];
const KNN_METRICS: &[&str] = &["l2_expanded", "l2_sqrt_expanded", "cosine", "inner_product"];
const KMEANS_METRICS: &[&str] = &["l2_expanded", "l2_sqrt_expanded"];
const IVF_PQ_INTEGER_METRICS: &[&str] = &["l2_expanded", "l2_sqrt_expanded", "inner_product"];
const BRUTE_FORCE_SUPPORT: &[CuvsVectorElementSupport] =
    &[CuvsVectorElementSupport { element: VectorElementType::Float32, metrics: KNN_METRICS }];
const IVF_SUPPORT: &[CuvsVectorElementSupport] = &[
    CuvsVectorElementSupport { element: VectorElementType::Float32, metrics: KNN_METRICS },
    CuvsVectorElementSupport { element: VectorElementType::Int8, metrics: KNN_METRICS },
    CuvsVectorElementSupport { element: VectorElementType::UInt8, metrics: KNN_METRICS },
];
const IVF_PQ_SUPPORT: &[CuvsVectorElementSupport] = &[
    CuvsVectorElementSupport { element: VectorElementType::Float32, metrics: KNN_METRICS },
    CuvsVectorElementSupport { element: VectorElementType::Int8, metrics: IVF_PQ_INTEGER_METRICS },
    CuvsVectorElementSupport { element: VectorElementType::UInt8, metrics: IVF_PQ_INTEGER_METRICS },
];
const CAGRA_SUPPORT: &[CuvsVectorElementSupport] = &[
    CuvsVectorElementSupport { element: VectorElementType::Float32, metrics: CAGRA_METRICS },
    CuvsVectorElementSupport { element: VectorElementType::Int8, metrics: CAGRA_METRICS },
    CuvsVectorElementSupport { element: VectorElementType::UInt8, metrics: CAGRA_METRICS },
];
const PAIRWISE_SUPPORT: &[CuvsVectorElementSupport] = &[
    CuvsVectorElementSupport { element: VectorElementType::Float32, metrics: KNN_METRICS },
    CuvsVectorElementSupport { element: VectorElementType::Float64, metrics: KNN_METRICS },
];
const KMEANS_SUPPORT: &[CuvsVectorElementSupport] = &[
    CuvsVectorElementSupport { element: VectorElementType::Float32, metrics: KMEANS_METRICS },
    CuvsVectorElementSupport { element: VectorElementType::Float64, metrics: KMEANS_METRICS },
];
const KMEANS_INIT: &[&str] = &["kmeans++", "random"];
const PCA_SOLVERS: &[&str] = &["cov_eig_dq", "cov_eig_jacobi"];

const KNN_OPTIONS: &[GpuOptionDefinition] = &[
    GpuOptionDefinition {
        name: "dataset",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Names the dataset ID column and one dense-vector binding shape.",
    },
    GpuOptionDefinition {
        name: "queries",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Names the query ID column and one dense-vector binding shape.",
    },
    GpuOptionDefinition {
        name: "k",
        required: true,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: None },
        description: "Number of neighbors returned for each evaluated query row.",
    },
    GpuOptionDefinition {
        name: "metric",
        required: true,
        kind: GpuOptionKind::Enum { values: KNN_METRICS, default: None },
        description: "Distance or score metric. Distance metrics rank lower values first; inner product ranks higher values first.",
    },
];
const CAGRA_METRICS: &[&str] = &["l2_expanded", "inner_product", "cosine"];
const CAGRA_BUILD_ALGOS: &[&str] = &["ivf_pq", "nn_descent"];
const CAGRA_SEARCH_ALGOS: &[&str] = &["auto", "single_cta", "multi_cta", "multi_kernel"];
const CAGRA_OPTIONS: &[GpuOptionDefinition] = &[
    KNN_OPTIONS[0],
    KNN_OPTIONS[1],
    KNN_OPTIONS[2],
    GpuOptionDefinition {
        name: "metric",
        required: false,
        kind: GpuOptionKind::Enum { values: CAGRA_METRICS, default: Some("l2_expanded") },
        description: "Distance or score metric; native default l2_expanded. inner_product ranks higher scores first.",
    },
    GpuOptionDefinition {
        name: "graph_degree",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: Some(64) },
        description: "Output graph degree; native default 64. Must be less than dataset rows.",
    },
    GpuOptionDefinition {
        name: "intermediate_graph_degree",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: Some(128) },
        description: "Build graph degree; native default 128. Must exceed graph_degree and be less than dataset rows.",
    },
    GpuOptionDefinition {
        name: "build_algo",
        required: false,
        kind: GpuOptionKind::Enum { values: CAGRA_BUILD_ALGOS, default: Some("ivf_pq") },
        description: "Graph construction algorithm; native default ivf_pq. nn_descent does not support cosine.",
    },
    GpuOptionDefinition {
        name: "itopk_size",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: Some(64) },
        description: "Intermediate search results; native default 64. Must be at least k.",
    },
    GpuOptionDefinition {
        name: "search_width",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: Some(1) },
        description: "Starting graph nodes per iteration; native default 1.",
    },
    GpuOptionDefinition {
        name: "max_iterations",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 0, maximum: u32::MAX as u64, default: Some(0) },
        description: "Maximum search iterations; native default 0 selects automatically.",
    },
    GpuOptionDefinition {
        name: "search_algo",
        required: false,
        kind: GpuOptionKind::Enum { values: CAGRA_SEARCH_ALGOS, default: Some("auto") },
        description: "Search kernel; auto is set explicitly (the native zero-initialized default is single_cta).",
    },
];
const IVF_FLAT_OPTIONS: &[GpuOptionDefinition] = &[
    KNN_OPTIONS[0],
    KNN_OPTIONS[1],
    KNN_OPTIONS[2],
    GpuOptionDefinition {
        name: "metric",
        required: false,
        kind: GpuOptionKind::Enum { values: KNN_METRICS, default: Some("l2_expanded") },
        description: "Distance or score metric; native default l2_expanded. Distance metrics rank lower values first; inner product ranks higher values first.",
    },
    GpuOptionDefinition {
        name: "n_lists",
        required: true,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: None },
        description: "Number of inverted lists; must not exceed the evaluated dataset row count.",
    },
    GpuOptionDefinition {
        name: "n_probes",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: Some(20) },
        description: "Lists probed per query; native default 20, and must not exceed n_lists.",
    },
    GpuOptionDefinition {
        name: "kmeans_n_iters",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: Some(20) },
        description: "Iterations for training list centers; native default 20.",
    },
    GpuOptionDefinition {
        name: "kmeans_trainset_fraction",
        required: false,
        kind: GpuOptionKind::Fraction { default: 0.5 },
        description: "Fraction of the dataset used for training centers; native default 0.5.",
    },
];
const IVF_PQ_CODEBOOK_KINDS: &[&str] = &["per_subspace", "per_cluster"];
const IVF_PQ_OPTIONS: &[GpuOptionDefinition] = &[
    IVF_FLAT_OPTIONS[0],
    IVF_FLAT_OPTIONS[1],
    IVF_FLAT_OPTIONS[2],
    IVF_FLAT_OPTIONS[3],
    IVF_FLAT_OPTIONS[4],
    IVF_FLAT_OPTIONS[5],
    IVF_FLAT_OPTIONS[6],
    IVF_FLAT_OPTIONS[7],
    GpuOptionDefinition {
        name: "pq_dim",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 0, maximum: u32::MAX as u64, default: Some(0) },
        description: "Compressed dimensions; native default 0 selects the dimension automatically. pq_dim * pq_bits must be divisible by 8.",
    },
    GpuOptionDefinition {
        name: "pq_bits",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 4, maximum: 8, default: Some(8) },
        description: "Bits per PQ code; native default 8.",
    },
    GpuOptionDefinition {
        name: "codebook_kind",
        required: false,
        kind: GpuOptionKind::Enum { values: IVF_PQ_CODEBOOK_KINDS, default: Some("per_subspace") },
        description: "PQ codebook training scope; native default per_subspace.",
    },
];
const PAIRWISE_OPTIONS: &[GpuOptionDefinition] = &[
    GpuOptionDefinition {
        name: "x",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "First dense-vector relation.",
    },
    GpuOptionDefinition {
        name: "y",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Second dense-vector relation.",
    },
    GpuOptionDefinition {
        name: "metric",
        required: true,
        kind: GpuOptionKind::Enum { values: KNN_METRICS, default: None },
        description: "Distance metric; inner_product is a similarity where larger is closer.",
    },
];
const TOP_K_OPTIONS: &[GpuOptionDefinition] = &[
    GpuOptionDefinition {
        name: "input",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Dense-vector rows to select from.",
    },
    GpuOptionDefinition {
        name: "k",
        required: true,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: u32::MAX as u64, default: None },
        description: "Number of values selected from each row.",
    },
    GpuOptionDefinition {
        name: "select",
        required: true,
        kind: GpuOptionKind::Enum { values: TOP_K_SELECTIONS, default: None },
        description: "Select the minimum or maximum values.",
    },
];
const KMEANS_OPTIONS: &[GpuOptionDefinition] = &[
    GpuOptionDefinition {
        name: "input",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Names the input ID column and one dense-vector binding shape.",
    },
    GpuOptionDefinition {
        name: "n_clusters",
        required: true,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: None },
        description: "Number of clusters fitted for this one statement.",
    },
    GpuOptionDefinition {
        name: "metric",
        required: false,
        kind: GpuOptionKind::Enum { values: KMEANS_METRICS, default: Some("l2_expanded") },
        description: "L2 distance form used while fitting and assigning the current input relation.",
    },
    GpuOptionDefinition {
        name: "max_iter",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(100) },
        description: "Maximum fitting iterations for this invocation.",
    },
    GpuOptionDefinition {
        name: "tol",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 0.0001 },
        description: "Non-negative convergence tolerance used by the fitted model.",
    },
    GpuOptionDefinition {
        name: "n_init",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(1) },
        description: "Number of initialization attempts performed within this call.",
    },
    GpuOptionDefinition {
        name: "init",
        required: false,
        kind: GpuOptionKind::Enum { values: KMEANS_INIT, default: Some("kmeans++") },
        description: "Centroid initialization strategy for this fitted model.",
    },
];
const PCA_OPTIONS: &[GpuOptionDefinition] = &[
    GpuOptionDefinition {
        name: "input",
        required: true,
        kind: GpuOptionKind::RelationBinding,
        description: "Names the input ID column and one dense-vector binding shape.",
    },
    GpuOptionDefinition {
        name: "n_components",
        required: true,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: None },
        description: "Number of principal-component columns emitted as pc_0 through pc_<n_components - 1>.",
    },
    GpuOptionDefinition {
        name: "whiten",
        required: false,
        kind: GpuOptionKind::Boolean { default: Some(false) },
        description: "Whether the query-local transformed components are whitened.",
    },
    GpuOptionDefinition {
        name: "solver",
        required: false,
        kind: GpuOptionKind::Enum { values: PCA_SOLVERS, default: Some("cov_eig_dq") },
        description: "Covariance eigensolver used for this fit-transform call.",
    },
    GpuOptionDefinition {
        name: "tol",
        required: false,
        kind: GpuOptionKind::NonnegativeNumber { default: 0.0 },
        description: "Non-negative eigensolver tolerance.",
    },
    GpuOptionDefinition {
        name: "n_iterations",
        required: false,
        kind: GpuOptionKind::Integer { minimum: 1, maximum: i32::MAX as u64, default: Some(15) },
        description: "Maximum eigensolver iterations for this invocation.",
    },
    GpuOptionDefinition {
        name: "flip_signs_based_on_u",
        required: false,
        kind: GpuOptionKind::Boolean { default: Some(false) },
        description: "Whether cuVS applies its U-based component-sign convention.",
    },
];

const KNN_OUTPUT_FIELDS: &[GpuOutputFieldDefinition] = &[
    GpuOutputFieldDefinition {
        name: "query_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based ordinal of the evaluated query row; it disambiguates duplicate query IDs.",
    },
    GpuOutputFieldDefinition {
        name: "query_id",
        data_type: "same_as_queries.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the queries relation.",
    },
    GpuOutputFieldDefinition {
        name: "neighbor_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based ordinal of the matched dataset row; it disambiguates duplicate dataset IDs.",
    },
    GpuOutputFieldDefinition {
        name: "neighbor_id",
        data_type: "same_as_dataset.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the matched dataset row.",
    },
    GpuOutputFieldDefinition {
        name: "rank",
        data_type: "UInt32",
        nullable: false,
        repeats: None,
        description: "One-based neighbor rank within a query. Order consumers explicitly by query_ordinal, rank.",
    },
    GpuOutputFieldDefinition {
        name: "distance",
        data_type: "Float32",
        nullable: false,
        repeats: None,
        description: "Metric value; smaller is better for distance metrics, while inner_product prefers larger values.",
    },
];

const PAIRWISE_OUTPUT_FIELDS: &[GpuOutputFieldDefinition] = &[
    GpuOutputFieldDefinition {
        name: "x_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based physical ordinal of the evaluated x row; it disambiguates duplicate x IDs.",
    },
    GpuOutputFieldDefinition {
        name: "x_id",
        data_type: "same_as_x.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the x relation.",
    },
    GpuOutputFieldDefinition {
        name: "y_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based physical ordinal of the evaluated y row; it disambiguates duplicate y IDs.",
    },
    GpuOutputFieldDefinition {
        name: "y_id",
        data_type: "same_as_y.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the y relation.",
    },
    GpuOutputFieldDefinition {
        name: "distance",
        data_type: "same_as_x.vector_element",
        nullable: false,
        repeats: None,
        description: "Float32 for Float32 input, Float64 for Float64 input. Smaller is closer for distance metrics; inner_product prefers larger values.",
    },
];
const TOP_K_OUTPUT_FIELDS: &[GpuOutputFieldDefinition] = &[
    GpuOutputFieldDefinition {
        name: "row_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based physical ordinal of the evaluated input row; it disambiguates duplicate input IDs. Follows the input's top-level ORDER BY on output columns; unspecified without one.",
    },
    GpuOutputFieldDefinition {
        name: "id",
        data_type: "same_as_input.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the input relation.",
    },
    GpuOutputFieldDefinition {
        name: "rank",
        data_type: "UInt32",
        nullable: false,
        repeats: None,
        description: "One-based value rank within an input row. Order consumers explicitly by row_ordinal, rank.",
    },
    GpuOutputFieldDefinition {
        name: "position",
        data_type: "Int64",
        nullable: false,
        repeats: None,
        description: "Zero-based dimension index of the selected value within the input vector.",
    },
    GpuOutputFieldDefinition {
        name: "value",
        data_type: "Float32",
        nullable: false,
        repeats: None,
        description: "Selected Float32 input vector value; rank follows ascending values for min and descending values for max.",
    },
];
const KMEANS_OUTPUT_FIELDS: &[GpuOutputFieldDefinition] = &[
    GpuOutputFieldDefinition {
        name: "row_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based ordinal of the evaluated input row; it disambiguates duplicate IDs. Follows the input's top-level ORDER BY on output columns; unspecified without one.",
    },
    GpuOutputFieldDefinition {
        name: "id",
        data_type: "same_as_input.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the input relation.",
    },
    GpuOutputFieldDefinition {
        name: "cluster_id",
        data_type: "Int32",
        nullable: false,
        repeats: None,
        description: "Query-local numeric assignment label, not a stable business or topic identifier.",
    },
];

const PCA_OUTPUT_FIELDS: &[GpuOutputFieldDefinition] = &[
    GpuOutputFieldDefinition {
        name: "row_ordinal",
        data_type: "UInt64",
        nullable: false,
        repeats: None,
        description: "Zero-based ordinal of the evaluated input row; it disambiguates duplicate IDs. Follows the input's top-level ORDER BY on output columns; unspecified without one.",
    },
    GpuOutputFieldDefinition {
        name: "id",
        data_type: "same_as_input.id",
        nullable: false,
        repeats: None,
        description: "Logical ID copied from the input relation.",
    },
    GpuOutputFieldDefinition {
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
    IvfFlat,
    IvfPq,
    Cagra,
    PairwiseDistance,
    TopK,
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
        vector_element_support: BRUTE_FORCE_SUPPORT,
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
            "Distances repeat between runs; neighbors that tie at the k-th distance can differ, because candidate merge order is not fixed.",
        ],
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::IvfFlat,
        sql_function_name: CUVS_IVF_FLAT_FUNCTION,
        display_name: "IVF-Flat",
        summary: "Query-local inverted-file approximate nearest-neighbor search.",
        use_this_when: "Use this when a query-local approximate index reduces search work over a large evaluated dataset.",
        relation_roles: KNN_RELATIONS,
        vector_element_support: IVF_SUPPORT,
        output_contract: CuvsOutputContract::KnnNeighbors,
        options: IVF_FLAT_OPTIONS,
        output_fields: KNN_OUTPUT_FIELDS,
        quickstart_sql: r#"SELECT *
FROM cuvs_ivf_flat(
  dataset => (SELECT id, d0, d1 FROM dataset_vectors),
  queries => (SELECT id, d0, d1 FROM query_vectors),
  k => 8,
  metric => 'l2_expanded',
  n_lists => 16,
  n_probes => 16
)
ORDER BY query_ordinal, rank;"#,
        limitations: &[
            "Builds and destroys its query-local index in the statement; no index persists across statements.",
            "Results are approximate unless all lists are probed. Fewer than k neighbors may be returned for a query; missing (query, rank) rows are omitted and valid ranks remain contiguous.",
            "The evaluated dataset must be non-empty, k and n_lists must not exceed its row count, and n_probes must not exceed n_lists.",
            "Dataset and query dimensions and element types must match. Cosine requires at least two dimensions.",
            "Neighbors and distances can differ slightly between runs, because index construction accumulates k-means centroid sums with floating-point atomics; there is no seed option.",
        ],
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::IvfPq,
        sql_function_name: CUVS_IVF_PQ_FUNCTION,
        display_name: "IVF-PQ",
        summary: "Query-local product-quantized inverted-file nearest-neighbor search.",
        use_this_when: "Use this when a query-local compressed index can reduce vector search memory, subject to lossy recall.",
        relation_roles: KNN_RELATIONS,
        vector_element_support: IVF_PQ_SUPPORT,
        output_contract: CuvsOutputContract::KnnNeighbors,
        options: IVF_PQ_OPTIONS,
        output_fields: KNN_OUTPUT_FIELDS,
        quickstart_sql: r#"SELECT *
FROM cuvs_ivf_pq(
  dataset => (SELECT id, d0, d1 FROM dataset_vectors),
  queries => (SELECT id, d0, d1 FROM query_vectors),
  k => 8,
  metric => 'l2_expanded',
  n_lists => 16,
  n_probes => 16,
  pq_dim => 2
)
ORDER BY query_ordinal, rank;"#,
        limitations: &[
            "Builds and destroys its query-local index within the statement; no index persists across statements.",
            "Product quantization is lossy even when all lists are probed. Missing neighbors are omitted and valid ranks remain contiguous.",
            "The dataset must be non-empty; k and n_lists must not exceed its row count, and n_probes must not exceed n_lists.",
            "Dataset and query dimensions and element types must match. Cosine requires at least two dimensions. Nonzero pq_dim * pq_bits must be divisible by 8; dimensions are padded when needed.",
            "Neighbors and distances can differ slightly between runs, because index training accumulates k-means centroid sums with floating-point atomics; there is no seed option.",
        ],
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::Cagra,
        sql_function_name: CUVS_CAGRA_FUNCTION,
        display_name: "CAGRA",
        summary: "Query-local graph approximate nearest-neighbor search.",
        use_this_when: "Use this for graph-based approximate search over an evaluated dataset.",
        relation_roles: KNN_RELATIONS,
        vector_element_support: CAGRA_SUPPORT,
        output_contract: CuvsOutputContract::KnnNeighbors,
        options: CAGRA_OPTIONS,
        output_fields: KNN_OUTPUT_FIELDS,
        quickstart_sql: r#"SELECT *
FROM cuvs_cagra(
  dataset => (SELECT id, d0, d1 FROM dataset_vectors),
  queries => (SELECT id, d0, d1 FROM query_vectors),
  k => 8,
  metric => 'l2_expanded',
  graph_degree => 32,
  intermediate_graph_degree => 64
)
ORDER BY query_ordinal, rank;"#,
        limitations: &[
            "Builds and destroys a query-local graph and padded dataset within the statement; no index persists across statements.",
            "Approximate search may return fewer than k neighbors; missing (query, rank) rows are omitted and valid ranks remain contiguous.",
            "The dataset must be non-empty, k and intermediate_graph_degree must be less than or equal to its row count, and graph_degree must be less than intermediate_graph_degree.",
            "k must not exceed itopk_size. single_cta requires itopk_size no greater than 1024. cosine cannot use nn_descent.",
            "Dataset and query dimensions and element types must match.",
            "Neighbors and distances can differ slightly between runs, because graph construction orders edges by atomic arrival; there is no seed option.",
        ],
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::PairwiseDistance,
        sql_function_name: CUVS_PAIRWISE_DISTANCE_FUNCTION,
        display_name: "Pairwise Distance",
        summary: "Pairwise distance or similarity for every x and y vector row.",
        use_this_when: "Use this when every pair between two evaluated relations needs a metric value.",
        relation_roles: PAIRWISE_RELATIONS,
        vector_element_support: PAIRWISE_SUPPORT,
        output_contract: CuvsOutputContract::PairwiseDistances,
        options: PAIRWISE_OPTIONS,
        output_fields: PAIRWISE_OUTPUT_FIELDS,
        quickstart_sql: r#"SELECT *
FROM cuvs_pairwise_distance(
  x => (SELECT id, d0, d1 FROM x_vectors),
  y => (SELECT id, d0, d1 FROM y_vectors),
  metric => 'l2_expanded'
)
ORDER BY x_ordinal, y_ordinal;"#,
        limitations: &[
            "Output rows equal x rows multiplied by y rows; the query allocation cap limits the result size.",
            "Empty x or y returns an empty result with the stable schema.",
            "x and y must share the same vector dimension and element type. inner_product is a similarity where larger is closer.",
        ],
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::TopK,
        sql_function_name: CUVS_TOP_K_FUNCTION,
        display_name: "Vector Top-K",
        summary: "Select the smallest or largest values in every dense vector row.",
        use_this_when: "Use this when each row needs its own ranked scalar values and vector positions.",
        relation_roles: INPUT_RELATION,
        vector_element_support: FLOAT32_VECTOR_SUPPORT,
        output_contract: CuvsOutputContract::TopKValues,
        options: TOP_K_OPTIONS,
        output_fields: TOP_K_OUTPUT_FIELDS,
        quickstart_sql: r#"SELECT *
FROM cuvs_top_k(
  input => (SELECT id, d0, d1 FROM input_vectors),
  k => 1,
  select => 'max'
)
ORDER BY row_ordinal, rank;"#,
        limitations: &[
            "k must not exceed the vector dimension. Empty input returns an empty result with the stable schema.",
            "Equal selected values are ordered by position; when ties straddle rank k, the selected positions among them are unspecified.",
        ],
    },
    CuvsOperationDescriptor {
        operation: CuvsOperation::KMeans,
        sql_function_name: CUVS_KMEANS_FUNCTION,
        display_name: "KMeans",
        summary: "K-means clustering assignments for dense vector rows.",
        use_this_when: "Use this when one statement should fit clusters over its evaluated relation and immediately join or aggregate the assignments.",
        relation_roles: INPUT_RELATION,
        vector_element_support: KMEANS_SUPPORT,
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
            "Cluster sizes can differ slightly between runs, because centroid sums accumulate with floating-point atomics; the native seed is fixed at 0 and cannot be set.",
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
    pub fn element_support(&self, element: VectorElementType) -> Option<&CuvsVectorElementSupport> {
        self.vector_element_support.iter().find(|support| support.element == element)
    }

    pub fn option(&self, name: &str) -> Option<&'static GpuOptionDefinition> {
        self.options.iter().find(|option| option.name == name)
    }

    pub fn options_json_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for option in self.options {
            let mut schema = gpu_option_schema(
                option.kind,
                &dense_relation_binding_schema(
                    &self.vector_element_support.iter().map(|support| support.element).collect::<Vec<_>>(),
                ),
            );
            schema["description"] = Value::String(option.description.to_string());
            properties.insert(option.name.to_string(), schema);
            if option.required {
                required.push(Value::String(option.name.to_string()));
            }
        }
        if let Some(metric) = properties.get_mut("metric") {
            let combinations = self
                .vector_element_support
                .iter()
                .map(|support| format!("{}: {}", support.element.as_str(), support.metrics.join(", ")))
                .collect::<Vec<_>>()
                .join("; ");
            metric["description"] = json!(format!(
                "{} Supported element/metric combinations: {combinations}.",
                self.option("metric").expect("metric option").description
            ));
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

#[cfg(test)]
fn option_schema(kind: GpuOptionKind, element_support: &[CuvsVectorElementSupport]) -> Value {
    gpu_option_schema(
        kind,
        &dense_relation_binding_schema(&element_support.iter().map(|support| support.element).collect::<Vec<_>>()),
    )
}

/// Validate and canonicalize the scalar portion of an options JSON object.
/// Relation binding field names/types are checked by the SQL adapter against
/// its child schema, but presence and object shape are enforced here.
pub fn canonicalize_cuvs_options(
    operation: CuvsOperation,
    raw: &str,
) -> Result<Map<String, Value>, GpuOptionValidationError> {
    let descriptor = operation.descriptor();
    let object = canonicalize_options(descriptor.options, raw, "cuVS", |option, value| {
        validate_dense_relation_binding_value("cuVS", option, value, false)
    })?;
    if matches!(operation, CuvsOperation::IvfFlat | CuvsOperation::IvfPq) {
        let n_probes = object.get("n_probes").and_then(Value::as_u64).expect("validated n_probes");
        let n_lists = object.get("n_lists").and_then(Value::as_u64).expect("validated n_lists");
        if n_probes > n_lists {
            return Err(
                option_error("missing_or_invalid_option", "n_probes must not exceed n_lists").for_argument("n_probes")
            );
        }
    }
    if operation == CuvsOperation::Cagra {
        let integer = |name| object.get(name).and_then(Value::as_u64).expect("validated CAGRA integer");
        if integer("intermediate_graph_degree") <= integer("graph_degree") {
            return Err(option_error(
                "missing_or_invalid_option",
                "intermediate_graph_degree must exceed graph_degree",
            )
            .for_argument("intermediate_graph_degree"));
        }
        if integer("k") > integer("itopk_size") {
            return Err(
                option_error("missing_or_invalid_option", "k must not exceed itopk_size").for_argument("itopk_size")
            );
        }
        if object["search_algo"] == "single_cta" && integer("itopk_size") > 1024 {
            return Err(option_error("missing_or_invalid_option", "single_cta requires itopk_size <= 1024")
                .for_argument("itopk_size"));
        }
        if object["metric"] == "cosine" && object["build_algo"] == "nn_descent" {
            return Err(option_error("missing_or_invalid_option", "nn_descent cannot build a cosine graph")
                .for_argument("build_algo"));
        }
    }
    if operation == CuvsOperation::IvfPq {
        let pq_dim = object.get("pq_dim").and_then(Value::as_u64).expect("validated pq_dim");
        let pq_bits = object.get("pq_bits").and_then(Value::as_u64).expect("validated pq_bits");
        if pq_dim != 0 && (pq_dim * pq_bits) % 8 != 0 {
            return Err(option_error("missing_or_invalid_option", "pq_dim * pq_bits must be divisible by 8")
                .for_argument("pq_dim"));
        }
    }
    Ok(object)
}

fn option_error(code: &'static str, message: impl Into<String>) -> GpuOptionValidationError {
    GpuOptionValidationError::new(code, message)
}

/// Vector algorithm identity carried by the native IR.
///
/// Ref: <https://docs.nvidia.com/cuvs/api-reference/c-api-documentation>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorAlgorithm {
    /// Exact brute-force k-nearest-neighbor search.
    /// <https://docs.nvidia.com/cuvs/api-reference/c-api-neighbors-brute-force>
    BruteForceKnn,
    /// Query-local IVF-Flat approximate nearest-neighbor search.
    /// <https://docs.nvidia.com/cuvs/api-reference/c-api-neighbors-ivf-flat>
    IvfFlat,
    /// Query-local IVF-PQ compressed nearest-neighbor search.
    /// <https://docs.nvidia.com/cuvs/api-reference/c-api-neighbors-ivf-pq>
    IvfPq,
    /// Query-local graph approximate nearest-neighbor search.
    /// <https://docs.nvidia.com/cuvs/api-reference/c-api-neighbors-cagra>
    Cagra,
    /// Pairwise metric values between two relations.
    /// <https://docs.nvidia.com/cuvs/api-reference/c-api-distance-pairwise-distance>
    PairwiseDistance,
    /// Ranked values selected within each input vector.
    /// <https://docs.nvidia.com/cuvs/api-reference/cpp-api-selection-select-k>
    TopK,
    /// K-means fit followed by row-label prediction.
    /// <https://docs.nvidia.com/cuvs/api-reference/c-api-cluster-kmeans>
    KMeans,
    /// Principal-component fit and transform.
    /// <https://docs.nvidia.com/cuvs/api-reference/c-api-preprocessing-pca>
    Pca,
}

impl VectorAlgorithm {
    /// Stable diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BruteForceKnn => "brute_force_knn",
            Self::IvfFlat => "ivf_flat",
            Self::IvfPq => "ivf_pq",
            Self::Cagra => "cagra",
            Self::PairwiseDistance => "pairwise_distance",
            Self::TopK => "top_k",
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        CuvsOperation, GpuOptionDefinition, GpuOptionKind, canonicalize_cuvs_options, option_schema,
        validate_dense_relation_binding_value,
    };
    use crate::plan::{
        GpuOptionValidationError, gpu_function_options::validate_option_value as gpu_validate_option_value,
    };

    fn validate_option_value(
        option: &GpuOptionDefinition,
        value: &serde_json::Value,
    ) -> Result<(), GpuOptionValidationError> {
        gpu_validate_option_value(option, value, "cuVS", &|option, value| {
            validate_dense_relation_binding_value("cuVS", option, value, false)
        })
    }

    #[test]
    fn ivf_flat_rejects_probes_above_lists_with_argument_attribution() {
        let raw = json!({
            "dataset": {"id": "id", "vector": {"columns": ["d0", "d1"]}},
            "queries": {"id": "id", "vector": {"columns": ["d0", "d1"]}},
            "k": 1, "metric": "l2_expanded", "n_lists": 2, "n_probes": 3
        });
        let error = canonicalize_cuvs_options(CuvsOperation::IvfFlat, &raw.to_string()).unwrap_err();
        assert_eq!(error.argument(), Some("n_probes"));
        assert!(error.message().contains("n_probes must not exceed n_lists"));
        for fraction in [0.0, 1.01] {
            let mut invalid = raw.clone();
            invalid["n_probes"] = json!(2);
            invalid["kmeans_trainset_fraction"] = json!(fraction);
            let error = canonicalize_cuvs_options(CuvsOperation::IvfFlat, &invalid.to_string()).unwrap_err();
            assert_eq!(error.argument(), Some("kmeans_trainset_fraction"));
        }
    }

    #[test]
    fn ivf_pq_rejects_invalid_bit_packing_and_probe_count_with_argument_attribution() {
        let mut raw = json!({
            "dataset": {"id": "id", "vector": {"columns": ["d0", "d1"]}},
            "queries": {"id": "id", "vector": {"columns": ["d0", "d1"]}},
            "k": 1, "metric": "l2_expanded", "n_lists": 2,
            "n_probes": 2, "pq_dim": 1, "pq_bits": 5
        });
        let error = canonicalize_cuvs_options(CuvsOperation::IvfPq, &raw.to_string()).unwrap_err();
        assert_eq!(error.argument(), Some("pq_dim"));
        assert!(error.message().contains("divisible by 8"));
        raw["pq_dim"] = json!(8);
        raw["n_probes"] = json!(3);
        assert_eq!(
            canonicalize_cuvs_options(CuvsOperation::IvfPq, &raw.to_string()).unwrap_err().argument(),
            Some("n_probes")
        );
        raw["n_probes"] = json!(2);
        raw["pq_bits"] = json!(9);
        assert_eq!(
            canonicalize_cuvs_options(CuvsOperation::IvfPq, &raw.to_string()).unwrap_err().argument(),
            Some("pq_bits")
        );
    }

    #[test]
    fn cagra_rejects_incompatible_graph_and_search_parameters() {
        let mut raw = json!({
            "dataset": {"id": "id", "vector": {"columns": ["d0", "d1"]}},
            "queries": {"id": "id", "vector": {"columns": ["d0", "d1"]}},
            "k": 8, "metric": "l2_expanded", "graph_degree": 32,
            "intermediate_graph_degree": 16
        });
        assert_eq!(
            canonicalize_cuvs_options(CuvsOperation::Cagra, &raw.to_string()).unwrap_err().argument(),
            Some("intermediate_graph_degree")
        );
        raw["intermediate_graph_degree"] = json!(64);
        raw["itopk_size"] = json!(7);
        assert_eq!(
            canonicalize_cuvs_options(CuvsOperation::Cagra, &raw.to_string()).unwrap_err().argument(),
            Some("itopk_size")
        );
        raw["itopk_size"] = json!(8);
        raw["metric"] = json!("cosine");
        raw["build_algo"] = json!("nn_descent");
        assert_eq!(
            canonicalize_cuvs_options(CuvsOperation::Cagra, &raw.to_string()).unwrap_err().argument(),
            Some("build_algo")
        );
        raw["metric"] = json!("inner_product");
        let normalized = canonicalize_cuvs_options(CuvsOperation::Cagra, &raw.to_string()).unwrap();
        assert_eq!(normalized["search_algo"], "auto");
    }

    #[test]
    fn integer_options_honor_zero_and_bounded_minima() {
        for (minimum, maximum, accepted, rejected) in [(0, 10, [0, 10], [11, 12]), (4, 8, [4, 8], [0, 9])] {
            let option = GpuOptionDefinition {
                name: "integer",
                required: false,
                kind: GpuOptionKind::Integer { minimum, maximum, default: Some(minimum) },
                description: "Integer range.",
            };
            assert_eq!(option_schema(option.kind, &[])["minimum"], json!(minimum));
            for value in accepted {
                validate_option_value(&option, &json!(value)).expect("value inside range");
            }
            for value in rejected {
                assert!(validate_option_value(&option, &json!(value)).is_err());
            }
        }
    }

    #[test]
    fn fraction_options_exclude_zero_and_values_above_one() {
        let option = GpuOptionDefinition {
            name: "fraction",
            required: false,
            kind: GpuOptionKind::Fraction { default: 0.5 },
            description: "Training fraction.",
        };
        assert_eq!(option_schema(option.kind, &[])["exclusiveMinimum"], json!(0));
        for value in [0.5, 1.0] {
            validate_option_value(&option, &json!(value)).expect("fraction inside range");
        }
        for value in [0.0, -0.1, 1.1] {
            assert!(validate_option_value(&option, &json!(value)).is_err());
        }
    }
}
