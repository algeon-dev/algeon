//! Named SQL argument contracts for the cuGraph function family.
//!
//! The table in this module is the adapter-owned name and shape contract. It
//! deliberately keeps native parameter defaults behind typed resolvers so the
//! metadata and binder surfaces do not copy numbers from the native defaults.

use datafusion::common::Result;
use serde_json::{Value, json};

use super::{
    CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION, CUGRAPH_BFS_FUNCTION, CUGRAPH_CORE_NUMBER_FUNCTION,
    CUGRAPH_COSINE_FUNCTION, CUGRAPH_DEGREES_ALL_FUNCTION, CUGRAPH_ECG_FUNCTION,
    CUGRAPH_EDGE_BETWEENNESS_CENTRALITY_FUNCTION, CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION,
    CUGRAPH_FORCE_ATLAS2_FUNCTION, CUGRAPH_HITS_FUNCTION, CUGRAPH_IN_DEGREES_ALL_FUNCTION, CUGRAPH_JACCARD_FUNCTION,
    CUGRAPH_K_CORE_FUNCTION, CUGRAPH_KATZ_CENTRALITY_FUNCTION, CUGRAPH_LEIDEN_FUNCTION, CUGRAPH_LOUVAIN_FUNCTION,
    CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION, CUGRAPH_OUT_DEGREES_ALL_FUNCTION, CUGRAPH_OVERLAP_FUNCTION,
    CUGRAPH_PAGERANK_FUNCTION, CUGRAPH_PERSONALIZED_PAGERANK_FUNCTION, CUGRAPH_SORENSEN_FUNCTION,
    CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION, CUGRAPH_SSSP_FUNCTION,
    CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION, CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
    CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION, CugraphAlgorithm,
    types::{CugraphAlgorithmParams, CugraphGraphOptions},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphArgumentType {
    Boolean,
    Int64,
    UInt32,
    UInt64,
    Float64,
    Utf8,
    VertexId,
    VertexIdArray,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CugraphValueRange {
    MinInclusiveF64(f64),
    MinExclusiveF64(f64),
    ClosedF64 { min: f64, max: f64 },
    OpenClosedF64 { min: f64, max: f64 },
    MinInclusiveI64(i64),
    MinInclusiveU64(u64),
    ClosedU64 { min: u64, max: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphAllowedValue {
    Boolean(bool),
    Text(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphRelationColumnType {
    VertexId,
    EdgeId,
    Numeric,
    Float32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CugraphRelationColumnDefinition {
    pub name: &'static str,
    pub column_type: CugraphRelationColumnType,
    pub required: bool,
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CugraphArgumentGroup {
    pub name: &'static str,
    pub min_count: u8,
    pub max_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphArgumentApplicability {
    Always,
    WhenIn { argument: &'static str, values: &'static [&'static str] },
    WhenAnyEquals { conditions: &'static [(&'static str, &'static str)] },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphNativeDefault {
    Alpha,
    Epsilon,
    MaxIterations,
    MaxLevel,
    DepthLimit,
    OutputMode,
    ReturnTargetInfo,
    Threshold,
    Resolution,
    Beta,
    Normalize,
    Normalized,
    Seeds,
    K,
    IncludeEndpoints,
    ExactVertexThreshold,
    Cutoff,
    DegreeType,
    Theta,
    Seed,
    MinWeight,
    EnsembleSize,
    NClusters,
    NEigenvectors,
    EvsTolerance,
    EvsMaxIterations,
    KMeansTolerance,
    KMeansMaxIterations,
    MaxIter,
    OutboundAttractionDistribution,
    LinLogMode,
    PreventOverlapping,
    OverlapScalingRatio,
    EdgeWeightInfluence,
    JitterTolerance,
    BarnesHutOptimize,
    BarnesHutTheta,
    ScalingRatio,
    StrongGravityMode,
    Gravity,
    Verbose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphDefaultValue {
    None,
    GraphDirected,
    GraphRenumber,
    GraphConstructionPolicy,
    Native(CugraphNativeDefault),
}

impl CugraphDefaultValue {
    pub fn resolve(self, algorithm: CugraphAlgorithm) -> Option<Value> {
        self.resolve_with_graph_options(algorithm, &CugraphGraphOptions::default())
    }

    pub fn resolve_with_graph_options(
        self,
        algorithm: CugraphAlgorithm,
        defaults: &CugraphGraphOptions,
    ) -> Option<Value> {
        match self {
            Self::None => None,
            Self::GraphDirected => Some(json!(algorithm.effective_default_graph_options(defaults).directed)),
            Self::GraphRenumber => Some(json!(defaults.renumber)),
            Self::GraphConstructionPolicy => Some(json!(defaults.construction_policy.as_str())),
            Self::Native(field) => native_default(field, algorithm).ok(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CugraphValueDefinition {
    pub value_type: CugraphArgumentType,
    pub nullable: bool,
    pub default: CugraphDefaultValue,
    pub range: Option<CugraphValueRange>,
    pub allowed_values: &'static [CugraphAllowedValue],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CugraphArgumentKind {
    Relation { columns: &'static [CugraphRelationColumnDefinition] },
    Value(CugraphValueDefinition),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CugraphArgumentDefinition {
    pub name: &'static str,
    pub kind: CugraphArgumentKind,
    pub required: bool,
    pub group: Option<CugraphArgumentGroup>,
    pub applies_when: CugraphArgumentApplicability,
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CugraphFunctionDefinition {
    pub algorithm: CugraphAlgorithm,
    pub sql_function_name: &'static str,
    pub arguments: &'static [CugraphArgumentDefinition],
}

const EMPTY_ALLOWED: &[CugraphAllowedValue] = &[];
const ALLOWED_FALSE: &[CugraphAllowedValue] = &[CugraphAllowedValue::Boolean(false)];
const ALLOWED_TRUE: &[CugraphAllowedValue] = &[CugraphAllowedValue::Boolean(true)];
const CONSTRUCTION_POLICIES: &[CugraphAllowedValue] =
    &[CugraphAllowedValue::Text("python_cugraph"), CugraphAllowedValue::Text("raw_libcugraph")];
const OUTPUT_MODES: &[CugraphAllowedValue] =
    &[CugraphAllowedValue::Text("raw"), CugraphAllowedValue::Text("normalized"), CugraphAllowedValue::Text("path")];
const DEGREE_TYPES: &[CugraphAllowedValue] =
    &[CugraphAllowedValue::Text("in"), CugraphAllowedValue::Text("out"), CugraphAllowedValue::Text("in_out")];
const WHEN_RAW_OR_NORMALIZED: &[&str] = &["raw", "normalized"];
const WHEN_TARGET_VERTICES: &[(&str, &str)] = &[("output_mode", "path"), ("return_target_info", "true")];

const EDGE_COLUMNS: &[CugraphRelationColumnDefinition] = &[
    CugraphRelationColumnDefinition {
        name: "src",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "source vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "dst",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "destination vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "weight",
        column_type: CugraphRelationColumnType::Numeric,
        required: false,
        description: "optional edge weight",
    },
    CugraphRelationColumnDefinition {
        name: "edge_id",
        column_type: CugraphRelationColumnType::EdgeId,
        required: false,
        description: "optional edge identifier",
    },
];
const MST_EDGE_COLUMNS: &[CugraphRelationColumnDefinition] = &[
    CugraphRelationColumnDefinition {
        name: "src",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "source vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "dst",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "destination vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "weight",
        column_type: CugraphRelationColumnType::Numeric,
        required: true,
        description: "edge weight used by the spanning tree",
    },
    CugraphRelationColumnDefinition {
        name: "edge_id",
        column_type: CugraphRelationColumnType::EdgeId,
        required: false,
        description: "optional edge identifier",
    },
];
const PERSONALIZATION_COLUMNS: &[CugraphRelationColumnDefinition] = &[
    CugraphRelationColumnDefinition {
        name: "vertex",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "personalization vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "value",
        column_type: CugraphRelationColumnType::Numeric,
        required: true,
        description: "personalization value",
    },
];
const VERTEX_COLUMNS: &[CugraphRelationColumnDefinition] = &[CugraphRelationColumnDefinition {
    name: "vertex",
    column_type: CugraphRelationColumnType::VertexId,
    required: true,
    description: "vertex identifier",
}];
const EDGE_ID_COLUMNS: &[CugraphRelationColumnDefinition] = &[CugraphRelationColumnDefinition {
    name: "edge_id",
    column_type: CugraphRelationColumnType::EdgeId,
    required: true,
    description: "edge identifier",
}];
const INCLUDE_EDGES_COLUMNS: &[CugraphRelationColumnDefinition] = &[
    CugraphRelationColumnDefinition {
        name: "src",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "source vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "dst",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "destination vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "edge_id",
        column_type: CugraphRelationColumnType::EdgeId,
        required: false,
        description: "optional edge identifier",
    },
];
const VERTEX_PAIRS_COLUMNS: &[CugraphRelationColumnDefinition] = &[
    CugraphRelationColumnDefinition {
        name: "first",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "first vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "second",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "second vertex identifier",
    },
];
const VERTEX_ATTRIBUTES_COLUMNS: &[CugraphRelationColumnDefinition] = &[
    CugraphRelationColumnDefinition {
        name: "vertex",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "radius",
        column_type: CugraphRelationColumnType::Float32,
        required: false,
        description: "optional node radius",
    },
    CugraphRelationColumnDefinition {
        name: "mobility",
        column_type: CugraphRelationColumnType::Float32,
        required: false,
        description: "optional node mobility",
    },
    CugraphRelationColumnDefinition {
        name: "mass",
        column_type: CugraphRelationColumnType::Float32,
        required: false,
        description: "optional node mass",
    },
];
const INITIAL_POSITIONS_COLUMNS: &[CugraphRelationColumnDefinition] = &[
    CugraphRelationColumnDefinition {
        name: "vertex",
        column_type: CugraphRelationColumnType::VertexId,
        required: true,
        description: "vertex identifier",
    },
    CugraphRelationColumnDefinition {
        name: "x",
        column_type: CugraphRelationColumnType::Float32,
        required: true,
        description: "initial x coordinate",
    },
    CugraphRelationColumnDefinition {
        name: "y",
        column_type: CugraphRelationColumnType::Float32,
        required: true,
        description: "initial y coordinate",
    },
];

const SOURCE_GROUP: CugraphArgumentGroup =
    CugraphArgumentGroup { name: "source_vertex_selector", min_count: 1, max_count: 1 };
const SSSP_SOURCE_GROUP: CugraphArgumentGroup =
    CugraphArgumentGroup { name: "sssp_source_vertex", min_count: 1, max_count: 1 };
const VERTEX_FILTER_GROUP: CugraphArgumentGroup =
    CugraphArgumentGroup { name: "bfs_vertex_filter", min_count: 0, max_count: 1 };
const BETWEENNESS_SELECTOR_GROUP: CugraphArgumentGroup =
    CugraphArgumentGroup { name: "betweenness_selector", min_count: 0, max_count: 1 };

const ARG_EDGES: CugraphArgumentDefinition = CugraphArgumentDefinition {
    name: "edges",
    kind: CugraphArgumentKind::Relation { columns: EDGE_COLUMNS },
    required: true,
    group: None,
    applies_when: CugraphArgumentApplicability::Always,
    description: "edge relation with canonical src and dst columns, plus optional weight and edge_id columns",
};
const ARG_EDGES_WEIGHTED: CugraphArgumentDefinition = CugraphArgumentDefinition {
    kind: CugraphArgumentKind::Relation { columns: MST_EDGE_COLUMNS },
    description: "weighted edge relation with canonical src, dst, and required weight columns",
    ..ARG_EDGES
};
const ARG_DIRECTED: CugraphArgumentDefinition = CugraphArgumentDefinition {
    name: "directed",
    kind: CugraphArgumentKind::Value(CugraphValueDefinition {
        value_type: CugraphArgumentType::Boolean,
        nullable: false,
        default: CugraphDefaultValue::GraphDirected,
        range: None,
        allowed_values: EMPTY_ALLOWED,
    }),
    required: false,
    group: None,
    applies_when: CugraphArgumentApplicability::Always,
    description: "whether the edge relation is directed; some algorithms require false or true",
};
const ARG_DIRECTED_FALSE: CugraphArgumentDefinition = CugraphArgumentDefinition {
    kind: CugraphArgumentKind::Value(CugraphValueDefinition {
        allowed_values: ALLOWED_FALSE,
        ..match_value(ARG_DIRECTED)
    }),
    ..ARG_DIRECTED
};
const ARG_DIRECTED_TRUE: CugraphArgumentDefinition = CugraphArgumentDefinition {
    kind: CugraphArgumentKind::Value(CugraphValueDefinition {
        allowed_values: ALLOWED_TRUE,
        ..match_value(ARG_DIRECTED)
    }),
    ..ARG_DIRECTED
};

const ARG_RENUMBER: CugraphArgumentDefinition = value_arg(
    "renumber",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::GraphRenumber,
    None,
    EMPTY_ALLOWED,
    "whether to renumber vertex identifiers during graph construction",
);
const ARG_CONSTRUCTION_POLICY: CugraphArgumentDefinition = value_arg(
    "construction_policy",
    CugraphArgumentType::Utf8,
    false,
    CugraphDefaultValue::GraphConstructionPolicy,
    None,
    CONSTRUCTION_POLICIES,
    "graph construction policy",
);

const ARG_PERSONALIZATION: CugraphArgumentDefinition = relation_arg(
    "personalization",
    PERSONALIZATION_COLUMNS,
    true,
    None,
    CugraphArgumentApplicability::Always,
    "personalization relation with vertex and numeric value columns",
);
const ARG_SOURCES: CugraphArgumentDefinition = relation_arg(
    "sources",
    VERTEX_COLUMNS,
    false,
    Some(SOURCE_GROUP),
    CugraphArgumentApplicability::Always,
    "optional BFS source relation; exactly one of sources, source_vertex, and source_vertices is required",
);
const ARG_INCLUDE_VERTICES: CugraphArgumentDefinition = CugraphArgumentDefinition {
    group: Some(VERTEX_FILTER_GROUP),
    ..relation_arg(
        "include_vertices",
        VERTEX_COLUMNS,
        false,
        None,
        CugraphArgumentApplicability::Always,
        "vertices retained by BFS predicate filtering",
    )
};
const ARG_EXCLUDE_VERTICES: CugraphArgumentDefinition = CugraphArgumentDefinition {
    group: Some(VERTEX_FILTER_GROUP),
    ..relation_arg(
        "exclude_vertices",
        VERTEX_COLUMNS,
        false,
        None,
        CugraphArgumentApplicability::Always,
        "vertices removed by BFS predicate filtering",
    )
};
const ARG_TARGET_VERTICES: CugraphArgumentDefinition = relation_arg(
    "target_vertices",
    VERTEX_COLUMNS,
    false,
    None,
    CugraphArgumentApplicability::WhenAnyEquals { conditions: WHEN_TARGET_VERTICES },
    "BFS target vertices required for path output or target information",
);
const ARG_INCLUDE_EDGE_IDS: CugraphArgumentDefinition = relation_arg(
    "include_edge_ids",
    EDGE_ID_COLUMNS,
    false,
    None,
    CugraphArgumentApplicability::Always,
    "edge identifiers retained by BFS predicate filtering",
);
const ARG_INCLUDE_EDGES: CugraphArgumentDefinition = relation_arg(
    "include_edges",
    INCLUDE_EDGES_COLUMNS,
    false,
    None,
    CugraphArgumentApplicability::Always,
    "edge predicate relation with src, dst, and optional edge_id columns",
);
const ARG_VERTEX_PAIRS: CugraphArgumentDefinition = relation_arg(
    "vertex_pairs",
    VERTEX_PAIRS_COLUMNS,
    true,
    None,
    CugraphArgumentApplicability::Always,
    "explicit vertex pairs used by similarity scoring",
);
const ARG_VERTEX_ATTRIBUTES: CugraphArgumentDefinition = relation_arg(
    "vertex_attributes",
    VERTEX_ATTRIBUTES_COLUMNS,
    false,
    None,
    CugraphArgumentApplicability::Always,
    "optional per-vertex ForceAtlas2 attributes; at least one attribute column is required",
);
const ARG_INITIAL_POSITIONS: CugraphArgumentDefinition = relation_arg(
    "initial_positions",
    INITIAL_POSITIONS_COLUMNS,
    false,
    None,
    CugraphArgumentApplicability::Always,
    "optional complete ForceAtlas2 initial positions",
);

const ARG_ALPHA: CugraphArgumentDefinition = value_arg(
    "alpha",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Alpha),
    Some(CugraphValueRange::ClosedF64 { min: 0.0, max: 1.0 }),
    EMPTY_ALLOWED,
    "PageRank damping factor in [0, 1]",
);
const ARG_KATZ_ALPHA: CugraphArgumentDefinition = value_arg(
    "alpha",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Alpha),
    Some(CugraphValueRange::OpenClosedF64 { min: 0.0, max: 1.0 }),
    EMPTY_ALLOWED,
    "Katz alpha in (0, 1]",
);
const ARG_EPSILON: CugraphArgumentDefinition = value_arg(
    "epsilon",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Epsilon),
    Some(CugraphValueRange::MinExclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "positive convergence tolerance",
);
const ARG_MAX_ITERATIONS: CugraphArgumentDefinition = value_arg(
    "max_iterations",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::MaxIterations),
    Some(CugraphValueRange::MinInclusiveU64(1)),
    EMPTY_ALLOWED,
    "maximum iteration count, at least 1",
);
const ARG_DEPTH_LIMIT: CugraphArgumentDefinition = value_arg(
    "depth_limit",
    CugraphArgumentType::Int64,
    true,
    CugraphDefaultValue::Native(CugraphNativeDefault::DepthLimit),
    Some(CugraphValueRange::MinInclusiveI64(0)),
    EMPTY_ALLOWED,
    "optional non-negative BFS depth limit",
);
const ARG_OUTPUT_MODE: CugraphArgumentDefinition = value_arg(
    "output_mode",
    CugraphArgumentType::Utf8,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::OutputMode),
    None,
    OUTPUT_MODES,
    "BFS output shape",
);
const ARG_RETURN_TARGET_INFO: CugraphArgumentDefinition = CugraphArgumentDefinition {
    kind: CugraphArgumentKind::Value(CugraphValueDefinition {
        value_type: CugraphArgumentType::Boolean,
        nullable: false,
        default: CugraphDefaultValue::Native(CugraphNativeDefault::ReturnTargetInfo),
        range: None,
        allowed_values: EMPTY_ALLOWED,
    }),
    name: "return_target_info",
    required: false,
    group: None,
    applies_when: CugraphArgumentApplicability::WhenIn { argument: "output_mode", values: WHEN_RAW_OR_NORMALIZED },
    description: "whether raw or normalized BFS output includes target information",
};
const ARG_MAX_LEVEL: CugraphArgumentDefinition = value_arg(
    "max_level",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::MaxLevel),
    Some(CugraphValueRange::MinInclusiveU64(1)),
    EMPTY_ALLOWED,
    "maximum hierarchy level, at least 1",
);
const ARG_THRESHOLD: CugraphArgumentDefinition = value_arg(
    "threshold",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Threshold),
    Some(CugraphValueRange::MinInclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "non-negative convergence threshold",
);
const ARG_RESOLUTION: CugraphArgumentDefinition = value_arg(
    "resolution",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Resolution),
    Some(CugraphValueRange::MinExclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "positive community resolution",
);
const ARG_BETA: CugraphArgumentDefinition = value_arg(
    "beta",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Beta),
    Some(CugraphValueRange::MinExclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "positive Katz beta",
);
const ARG_NORMALIZE: CugraphArgumentDefinition = value_arg(
    "normalize",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Normalize),
    None,
    EMPTY_ALLOWED,
    "whether HITS scores are normalized",
);
const ARG_SEEDS: CugraphArgumentDefinition = CugraphArgumentDefinition {
    group: Some(BETWEENNESS_SELECTOR_GROUP),
    ..value_arg(
        "seeds",
        CugraphArgumentType::VertexIdArray,
        true,
        CugraphDefaultValue::Native(CugraphNativeDefault::Seeds),
        None,
        EMPTY_ALLOWED,
        "optional non-empty homogeneous seed vertex array",
    )
};
const ARG_K: CugraphArgumentDefinition = CugraphArgumentDefinition {
    group: Some(BETWEENNESS_SELECTOR_GROUP),
    ..value_arg(
        "k",
        CugraphArgumentType::UInt64,
        true,
        CugraphDefaultValue::Native(CugraphNativeDefault::K),
        Some(CugraphValueRange::MinInclusiveU64(1)),
        EMPTY_ALLOWED,
        "optional sample count, at least 1",
    )
};
const ARG_K_CORE: CugraphArgumentDefinition = value_arg(
    "k",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::K),
    Some(CugraphValueRange::MinInclusiveU64(1)),
    EMPTY_ALLOWED,
    "k-core degree threshold, at least 1",
);
const ARG_NORMALIZED: CugraphArgumentDefinition = value_arg(
    "normalized",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Normalized),
    None,
    EMPTY_ALLOWED,
    "whether centrality scores are normalized",
);
const ARG_INCLUDE_ENDPOINTS: CugraphArgumentDefinition = value_arg(
    "include_endpoints",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::IncludeEndpoints),
    None,
    EMPTY_ALLOWED,
    "whether betweenness includes endpoints",
);
const ARG_EXACT_VERTEX_THRESHOLD: CugraphArgumentDefinition = value_arg(
    "exact_vertex_threshold",
    CugraphArgumentType::UInt64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::ExactVertexThreshold),
    None,
    EMPTY_ALLOWED,
    "vertex count threshold for exact betweenness",
);
const ARG_SOURCE_VERTEX: CugraphArgumentDefinition = CugraphArgumentDefinition {
    name: "source_vertex",
    kind: CugraphArgumentKind::Value(CugraphValueDefinition {
        value_type: CugraphArgumentType::VertexId,
        nullable: false,
        default: CugraphDefaultValue::None,
        range: None,
        allowed_values: EMPTY_ALLOWED,
    }),
    required: false,
    group: Some(SOURCE_GROUP),
    applies_when: CugraphArgumentApplicability::Always,
    description: "scalar source vertex; BFS source selectors are mutually exclusive",
};
const ARG_SOURCE_VERTICES: CugraphArgumentDefinition = CugraphArgumentDefinition {
    name: "source_vertices",
    kind: CugraphArgumentKind::Value(CugraphValueDefinition {
        value_type: CugraphArgumentType::VertexIdArray,
        nullable: false,
        default: CugraphDefaultValue::None,
        range: None,
        allowed_values: EMPTY_ALLOWED,
    }),
    required: false,
    group: Some(SOURCE_GROUP),
    applies_when: CugraphArgumentApplicability::Always,
    description: "non-empty homogeneous BFS source vertex array; mutually exclusive with source_vertex and sources",
};
const ARG_SSSP_SOURCE_VERTEX: CugraphArgumentDefinition = CugraphArgumentDefinition {
    required: true,
    group: Some(SSSP_SOURCE_GROUP),
    description: "required scalar SSSP source vertex",
    ..ARG_SOURCE_VERTEX
};
const ARG_CUTOFF: CugraphArgumentDefinition = value_arg(
    "cutoff",
    CugraphArgumentType::Float64,
    true,
    CugraphDefaultValue::Native(CugraphNativeDefault::Cutoff),
    Some(CugraphValueRange::MinInclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "optional non-negative SSSP distance cutoff",
);
const ARG_DEGREE_TYPE: CugraphArgumentDefinition = value_arg(
    "degree_type",
    CugraphArgumentType::Utf8,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::DegreeType),
    None,
    DEGREE_TYPES,
    "degree direction",
);
const ARG_SEED: CugraphArgumentDefinition = value_arg(
    "seed",
    CugraphArgumentType::UInt64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Seed),
    None,
    EMPTY_ALLOWED,
    "random initialization seed",
);
const ARG_THETA: CugraphArgumentDefinition = value_arg(
    "theta",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Theta),
    Some(CugraphValueRange::MinExclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "positive Leiden refinement scale",
);
const ARG_MIN_WEIGHT: CugraphArgumentDefinition = value_arg(
    "min_weight",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::MinWeight),
    Some(CugraphValueRange::MinInclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "minimum ECG edge weight",
);
const ARG_ENSEMBLE_SIZE: CugraphArgumentDefinition = value_arg(
    "ensemble_size",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::EnsembleSize),
    Some(CugraphValueRange::MinInclusiveU64(1)),
    EMPTY_ALLOWED,
    "number of ECG ensemble iterations",
);
const ARG_N_CLUSTERS: CugraphArgumentDefinition = value_arg(
    "n_clusters",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::NClusters),
    Some(CugraphValueRange::MinInclusiveU64(1)),
    EMPTY_ALLOWED,
    "number of spectral clusters",
);
const ARG_N_EIGENVECTORS: CugraphArgumentDefinition = value_arg(
    "n_eigenvectors",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::NEigenvectors),
    Some(CugraphValueRange::MinInclusiveU64(1)),
    EMPTY_ALLOWED,
    "number of spectral eigenvectors",
);
const ARG_EVS_TOLERANCE: CugraphArgumentDefinition = value_arg(
    "evs_tolerance",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::EvsTolerance),
    Some(CugraphValueRange::MinInclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "non-negative eigenvalue solver tolerance",
);
const ARG_EVS_MAX_ITERATIONS: CugraphArgumentDefinition = value_arg(
    "evs_max_iterations",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::EvsMaxIterations),
    Some(CugraphValueRange::ClosedU64 { min: 1, max: i32::MAX as u64 }),
    EMPTY_ALLOWED,
    "eigenvalue solver iteration limit",
);
const ARG_K_MEANS_TOLERANCE: CugraphArgumentDefinition = value_arg(
    "k_means_tolerance",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::KMeansTolerance),
    Some(CugraphValueRange::MinInclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "non-negative k-means tolerance",
);
const ARG_K_MEANS_MAX_ITERATIONS: CugraphArgumentDefinition = value_arg(
    "k_means_max_iterations",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::KMeansMaxIterations),
    Some(CugraphValueRange::ClosedU64 { min: 1, max: i32::MAX as u64 }),
    EMPTY_ALLOWED,
    "k-means iteration limit",
);
const ARG_MAX_ITER: CugraphArgumentDefinition = value_arg(
    "max_iter",
    CugraphArgumentType::UInt32,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::MaxIter),
    Some(CugraphValueRange::ClosedU64 { min: 1, max: i32::MAX as u64 }),
    EMPTY_ALLOWED,
    "ForceAtlas2 iteration limit",
);
const ARG_OUTBOUND_ATTRACTION_DISTRIBUTION: CugraphArgumentDefinition = value_arg(
    "outbound_attraction_distribution",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::OutboundAttractionDistribution),
    None,
    EMPTY_ALLOWED,
    "distribute ForceAtlas2 attraction along outbound edges",
);
const ARG_LIN_LOG_MODE: CugraphArgumentDefinition = value_arg(
    "lin_log_mode",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::LinLogMode),
    None,
    EMPTY_ALLOWED,
    "use the ForceAtlas2 lin-log attraction model",
);
const ARG_PREVENT_OVERLAPPING: CugraphArgumentDefinition = value_arg(
    "prevent_overlapping",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::PreventOverlapping),
    None,
    EMPTY_ALLOWED,
    "prevent node overlap; requires a radius attribute",
);
const ARG_OVERLAP_SCALING_RATIO: CugraphArgumentDefinition = value_arg(
    "overlap_scaling_ratio",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::OverlapScalingRatio),
    Some(CugraphValueRange::MinExclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "positive overlap repulsion scaling ratio",
);
const ARG_EDGE_WEIGHT_INFLUENCE: CugraphArgumentDefinition = value_arg(
    "edge_weight_influence",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::EdgeWeightInfluence),
    Some(CugraphValueRange::MinInclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "non-negative edge weight influence",
);
const ARG_JITTER_TOLERANCE: CugraphArgumentDefinition = value_arg(
    "jitter_tolerance",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::JitterTolerance),
    Some(CugraphValueRange::MinInclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "non-negative ForceAtlas2 jitter tolerance",
);
const ARG_BARNES_HUT_OPTIMIZE: CugraphArgumentDefinition = value_arg(
    "barnes_hut_optimize",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::BarnesHutOptimize),
    None,
    EMPTY_ALLOWED,
    "use the Barnes-Hut ForceAtlas2 approximation",
);
const ARG_BARNES_HUT_THETA: CugraphArgumentDefinition = value_arg(
    "barnes_hut_theta",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::BarnesHutTheta),
    Some(CugraphValueRange::ClosedF64 { min: 0.0, max: 1.0 }),
    EMPTY_ALLOWED,
    "Barnes-Hut speed and accuracy trade-off",
);
const ARG_SCALING_RATIO: CugraphArgumentDefinition = value_arg(
    "scaling_ratio",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::ScalingRatio),
    Some(CugraphValueRange::MinExclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "positive ForceAtlas2 repulsion scaling ratio",
);
const ARG_STRONG_GRAVITY_MODE: CugraphArgumentDefinition = value_arg(
    "strong_gravity_mode",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::StrongGravityMode),
    None,
    EMPTY_ALLOWED,
    "use stronger gravity for distant nodes",
);
const ARG_GRAVITY: CugraphArgumentDefinition = value_arg(
    "gravity",
    CugraphArgumentType::Float64,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Gravity),
    Some(CugraphValueRange::MinInclusiveF64(0.0)),
    EMPTY_ALLOWED,
    "non-negative ForceAtlas2 gravity strength",
);
const ARG_VERBOSE: CugraphArgumentDefinition = value_arg(
    "verbose",
    CugraphArgumentType::Boolean,
    false,
    CugraphDefaultValue::Native(CugraphNativeDefault::Verbose),
    None,
    EMPTY_ALLOWED,
    "whether ForceAtlas2 emits convergence information",
);

const fn value_arg(
    name: &'static str,
    value_type: CugraphArgumentType,
    nullable: bool,
    default: CugraphDefaultValue,
    range: Option<CugraphValueRange>,
    allowed_values: &'static [CugraphAllowedValue],
    description: &'static str,
) -> CugraphArgumentDefinition {
    CugraphArgumentDefinition {
        name,
        kind: CugraphArgumentKind::Value(CugraphValueDefinition {
            value_type,
            nullable,
            default,
            range,
            allowed_values,
        }),
        required: false,
        group: None,
        applies_when: CugraphArgumentApplicability::Always,
        description,
    }
}

const fn relation_arg(
    name: &'static str,
    columns: &'static [CugraphRelationColumnDefinition],
    required: bool,
    group: Option<CugraphArgumentGroup>,
    applies_when: CugraphArgumentApplicability,
    description: &'static str,
) -> CugraphArgumentDefinition {
    CugraphArgumentDefinition {
        name,
        kind: CugraphArgumentKind::Relation { columns },
        required,
        group,
        applies_when,
        description,
    }
}

const fn match_value(definition: CugraphArgumentDefinition) -> CugraphValueDefinition {
    match definition.kind {
        CugraphArgumentKind::Value(value) => value,
        CugraphArgumentKind::Relation { .. } => panic!("expected a value argument"),
    }
}

const PAGERANK_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY, ARG_ALPHA, ARG_EPSILON, ARG_MAX_ITERATIONS];
const PERSONALIZED_PAGERANK_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_PERSONALIZATION,
    ARG_ALPHA,
    ARG_EPSILON,
    ARG_MAX_ITERATIONS,
];
const BFS_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_SOURCES,
    ARG_INCLUDE_VERTICES,
    ARG_EXCLUDE_VERTICES,
    ARG_TARGET_VERTICES,
    ARG_INCLUDE_EDGE_IDS,
    ARG_INCLUDE_EDGES,
    ARG_SOURCE_VERTEX,
    ARG_SOURCE_VERTICES,
    ARG_DEPTH_LIMIT,
    ARG_OUTPUT_MODE,
    ARG_RETURN_TARGET_INFO,
];
const LOUVAIN_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED_FALSE,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_MAX_LEVEL,
    ARG_THRESHOLD,
    ARG_RESOLUTION,
];
const EIGENVECTOR_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY, ARG_EPSILON, ARG_MAX_ITERATIONS];
const KATZ_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_KATZ_ALPHA,
    ARG_BETA,
    ARG_EPSILON,
    ARG_MAX_ITERATIONS,
];
const HITS_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY, ARG_EPSILON, ARG_MAX_ITERATIONS, ARG_NORMALIZE];
const BETWEENNESS_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_SEEDS,
    ARG_K,
    ARG_NORMALIZED,
    ARG_INCLUDE_ENDPOINTS,
    ARG_EXACT_VERTEX_THRESHOLD,
];
const EDGE_BETWEENNESS_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_SEEDS,
    ARG_K,
    ARG_NORMALIZED,
    ARG_EXACT_VERTEX_THRESHOLD,
];
const SSSP_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY, ARG_SSSP_SOURCE_VERTEX, ARG_CUTOFF];
const DEGREES_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY];
const IN_DEGREES_ARGUMENTS: &[CugraphArgumentDefinition] = DEGREES_ARGUMENTS;
const OUT_DEGREES_ARGUMENTS: &[CugraphArgumentDefinition] = DEGREES_ARGUMENTS;
const WCC_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED_FALSE, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY];
const SCC_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED_TRUE, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY];
const TRIANGLE_ARGUMENTS: &[CugraphArgumentDefinition] = WCC_ARGUMENTS;
const CORE_NUMBER_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED_FALSE, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY, ARG_DEGREE_TYPE];
const K_CORE_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED_FALSE, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY, ARG_K_CORE, ARG_DEGREE_TYPE];
const MST_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES_WEIGHTED, ARG_DIRECTED_FALSE, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY];
const LEIDEN_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED_FALSE,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_MAX_LEVEL,
    ARG_RESOLUTION,
    ARG_THETA,
    ARG_SEED,
];
const ECG_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED_FALSE,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_MIN_WEIGHT,
    ARG_ENSEMBLE_SIZE,
    ARG_MAX_LEVEL,
    ARG_THRESHOLD,
    ARG_RESOLUTION,
    ARG_SEED,
];
const SPECTRAL_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED_FALSE,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_N_CLUSTERS,
    ARG_N_EIGENVECTORS,
    ARG_EVS_TOLERANCE,
    ARG_EVS_MAX_ITERATIONS,
    ARG_K_MEANS_TOLERANCE,
    ARG_K_MEANS_MAX_ITERATIONS,
    ARG_SEED,
];
const FORCE_ATLAS2_ARGUMENTS: &[CugraphArgumentDefinition] = &[
    ARG_EDGES,
    ARG_DIRECTED_FALSE,
    ARG_RENUMBER,
    ARG_CONSTRUCTION_POLICY,
    ARG_VERTEX_ATTRIBUTES,
    ARG_INITIAL_POSITIONS,
    ARG_MAX_ITER,
    ARG_OUTBOUND_ATTRACTION_DISTRIBUTION,
    ARG_LIN_LOG_MODE,
    ARG_PREVENT_OVERLAPPING,
    ARG_OVERLAP_SCALING_RATIO,
    ARG_EDGE_WEIGHT_INFLUENCE,
    ARG_JITTER_TOLERANCE,
    ARG_BARNES_HUT_OPTIMIZE,
    ARG_BARNES_HUT_THETA,
    ARG_SCALING_RATIO,
    ARG_STRONG_GRAVITY_MODE,
    ARG_GRAVITY,
    ARG_VERBOSE,
    ARG_SEED,
];
const SIMILARITY_ARGUMENTS: &[CugraphArgumentDefinition] =
    &[ARG_EDGES, ARG_DIRECTED_FALSE, ARG_RENUMBER, ARG_CONSTRUCTION_POLICY, ARG_VERTEX_PAIRS];

pub const CUGRAPH_ARGUMENT_DEFINITIONS: &[CugraphFunctionDefinition] = &[
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::PageRank,
        sql_function_name: CUGRAPH_PAGERANK_FUNCTION,
        arguments: PAGERANK_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Bfs,
        sql_function_name: CUGRAPH_BFS_FUNCTION,
        arguments: BFS_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Louvain,
        sql_function_name: CUGRAPH_LOUVAIN_FUNCTION,
        arguments: LOUVAIN_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::EigenvectorCentrality,
        sql_function_name: CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION,
        arguments: EIGENVECTOR_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::KatzCentrality,
        sql_function_name: CUGRAPH_KATZ_CENTRALITY_FUNCTION,
        arguments: KATZ_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Hits,
        sql_function_name: CUGRAPH_HITS_FUNCTION,
        arguments: HITS_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::BetweennessCentrality,
        sql_function_name: CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION,
        arguments: BETWEENNESS_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::EdgeBetweennessCentrality,
        sql_function_name: CUGRAPH_EDGE_BETWEENNESS_CENTRALITY_FUNCTION,
        arguments: EDGE_BETWEENNESS_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Sssp,
        sql_function_name: CUGRAPH_SSSP_FUNCTION,
        arguments: SSSP_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::DegreesAll,
        sql_function_name: CUGRAPH_DEGREES_ALL_FUNCTION,
        arguments: DEGREES_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::InDegreesAll,
        sql_function_name: CUGRAPH_IN_DEGREES_ALL_FUNCTION,
        arguments: IN_DEGREES_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::OutDegreesAll,
        sql_function_name: CUGRAPH_OUT_DEGREES_ALL_FUNCTION,
        arguments: OUT_DEGREES_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::WeaklyConnectedComponents,
        sql_function_name: CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION,
        arguments: WCC_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::StronglyConnectedComponents,
        sql_function_name: CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION,
        arguments: SCC_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::TriangleCountAll,
        sql_function_name: CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
        arguments: TRIANGLE_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::CoreNumber,
        sql_function_name: CUGRAPH_CORE_NUMBER_FUNCTION,
        arguments: CORE_NUMBER_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::KCore,
        sql_function_name: CUGRAPH_K_CORE_FUNCTION,
        arguments: K_CORE_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::MinimumSpanningTree,
        sql_function_name: CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION,
        arguments: MST_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::PersonalizedPageRank,
        sql_function_name: CUGRAPH_PERSONALIZED_PAGERANK_FUNCTION,
        arguments: PERSONALIZED_PAGERANK_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Leiden,
        sql_function_name: CUGRAPH_LEIDEN_FUNCTION,
        arguments: LEIDEN_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Ecg,
        sql_function_name: CUGRAPH_ECG_FUNCTION,
        arguments: ECG_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::SpectralModularityMaximization,
        sql_function_name: CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION,
        arguments: SPECTRAL_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::ForceAtlas2,
        sql_function_name: CUGRAPH_FORCE_ATLAS2_FUNCTION,
        arguments: FORCE_ATLAS2_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Jaccard,
        sql_function_name: CUGRAPH_JACCARD_FUNCTION,
        arguments: SIMILARITY_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Sorensen,
        sql_function_name: CUGRAPH_SORENSEN_FUNCTION,
        arguments: SIMILARITY_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Overlap,
        sql_function_name: CUGRAPH_OVERLAP_FUNCTION,
        arguments: SIMILARITY_ARGUMENTS,
    },
    CugraphFunctionDefinition {
        algorithm: CugraphAlgorithm::Cosine,
        sql_function_name: CUGRAPH_COSINE_FUNCTION,
        arguments: SIMILARITY_ARGUMENTS,
    },
];

pub fn definition_for_algorithm(algorithm: CugraphAlgorithm) -> &'static CugraphFunctionDefinition {
    CUGRAPH_ARGUMENT_DEFINITIONS
        .iter()
        .find(|definition| definition.algorithm == algorithm)
        .expect("cuGraph argument definitions are missing an algorithm")
}

pub fn arguments_for_algorithm(algorithm: CugraphAlgorithm) -> &'static [CugraphArgumentDefinition] {
    definition_for_algorithm(algorithm).arguments
}

fn native_default(field: CugraphNativeDefault, algorithm: CugraphAlgorithm) -> Result<Value> {
    use CugraphAlgorithmParams::*;

    let params = CugraphAlgorithmParams::from_native_defaults(algorithm, None)?;
    Ok(match (field, params) {
        (CugraphNativeDefault::Alpha, PageRank(params)) => json!(params.alpha),
        (CugraphNativeDefault::Alpha, PersonalizedPageRank(params)) => json!(params.pagerank.alpha),
        (CugraphNativeDefault::Alpha, KatzCentrality(params)) => json!(params.alpha),
        (CugraphNativeDefault::Epsilon, PageRank(params)) => json!(params.epsilon),
        (CugraphNativeDefault::Epsilon, PersonalizedPageRank(params)) => json!(params.pagerank.epsilon),
        (CugraphNativeDefault::Epsilon, EigenvectorCentrality(params)) => json!(params.epsilon),
        (CugraphNativeDefault::Epsilon, KatzCentrality(params)) => json!(params.epsilon),
        (CugraphNativeDefault::Epsilon, Hits(params)) => json!(params.epsilon),
        (CugraphNativeDefault::MaxIterations, PageRank(params)) => json!(params.max_iterations),
        (CugraphNativeDefault::MaxIterations, PersonalizedPageRank(params)) => json!(params.pagerank.max_iterations),
        (CugraphNativeDefault::MaxIterations, EigenvectorCentrality(params)) => json!(params.max_iterations),
        (CugraphNativeDefault::MaxIterations, KatzCentrality(params)) => json!(params.max_iterations),
        (CugraphNativeDefault::MaxIterations, Hits(params)) => json!(params.max_iterations),
        (CugraphNativeDefault::MaxLevel, Louvain(params)) => json!(params.max_level),
        (CugraphNativeDefault::MaxLevel, Leiden(params)) => json!(params.max_level),
        (CugraphNativeDefault::MaxLevel, Ecg(params)) => json!(params.max_level),
        (CugraphNativeDefault::DepthLimit, Bfs(params)) => json!(params.depth_limit),
        (CugraphNativeDefault::OutputMode, Bfs(params)) => json!(params.output.mode.as_str()),
        (CugraphNativeDefault::ReturnTargetInfo, Bfs(params)) => json!(params.output.return_target_info),
        (CugraphNativeDefault::Threshold, Louvain(params)) => json!(params.threshold),
        (CugraphNativeDefault::Threshold, Ecg(params)) => json!(params.threshold),
        (CugraphNativeDefault::Resolution, Louvain(params)) => json!(params.resolution),
        (CugraphNativeDefault::Resolution, Leiden(params)) => json!(params.resolution),
        (CugraphNativeDefault::Resolution, Ecg(params)) => json!(params.resolution),
        (CugraphNativeDefault::Beta, KatzCentrality(params)) => json!(params.beta),
        (CugraphNativeDefault::Normalize, Hits(params)) => json!(params.normalize),
        (CugraphNativeDefault::Normalized, BetweennessCentrality(params)) => json!(params.normalized),
        (CugraphNativeDefault::Normalized, EdgeBetweennessCentrality(params)) => json!(params.normalized),
        (CugraphNativeDefault::Seeds, BetweennessCentrality(_)) => Value::Null,
        (CugraphNativeDefault::Seeds, EdgeBetweennessCentrality(_)) => Value::Null,
        (CugraphNativeDefault::K, BetweennessCentrality(params)) => json!(params.k),
        (CugraphNativeDefault::K, EdgeBetweennessCentrality(params)) => json!(params.k),
        (CugraphNativeDefault::K, KCore(params)) => json!(params.k),
        (CugraphNativeDefault::IncludeEndpoints, BetweennessCentrality(params)) => json!(params.include_endpoints),
        (CugraphNativeDefault::ExactVertexThreshold, BetweennessCentrality(params)) => {
            json!(params.exact_vertex_threshold)
        },
        (CugraphNativeDefault::ExactVertexThreshold, EdgeBetweennessCentrality(params)) => {
            json!(params.exact_vertex_threshold)
        },
        (CugraphNativeDefault::Cutoff, Sssp(params)) => json!(params.cutoff),
        (CugraphNativeDefault::DegreeType, CoreNumber(params)) => json!(params.degree_type.as_str()),
        (CugraphNativeDefault::DegreeType, KCore(params)) => json!(params.degree_type.as_str()),
        (CugraphNativeDefault::Theta, Leiden(params)) => json!(params.theta),
        (CugraphNativeDefault::Seed, Leiden(params)) => json!(params.seed),
        (CugraphNativeDefault::Seed, Ecg(params)) => json!(params.seed),
        (CugraphNativeDefault::Seed, SpectralModularityMaximization(params)) => json!(params.seed),
        (CugraphNativeDefault::Seed, ForceAtlas2(params)) => json!(params.seed),
        (CugraphNativeDefault::MinWeight, Ecg(params)) => json!(params.min_weight),
        (CugraphNativeDefault::EnsembleSize, Ecg(params)) => json!(params.ensemble_size),
        (CugraphNativeDefault::NClusters, SpectralModularityMaximization(params)) => json!(params.n_clusters),
        (CugraphNativeDefault::NEigenvectors, SpectralModularityMaximization(params)) => json!(params.n_eigenvectors),
        (CugraphNativeDefault::EvsTolerance, SpectralModularityMaximization(params)) => json!(params.evs_tolerance),
        (CugraphNativeDefault::EvsMaxIterations, SpectralModularityMaximization(params)) => {
            json!(params.evs_max_iterations)
        },
        (CugraphNativeDefault::KMeansTolerance, SpectralModularityMaximization(params)) => {
            json!(params.k_means_tolerance)
        },
        (CugraphNativeDefault::KMeansMaxIterations, SpectralModularityMaximization(params)) => {
            json!(params.k_means_max_iterations)
        },
        (CugraphNativeDefault::MaxIter, ForceAtlas2(params)) => json!(params.max_iter),
        (CugraphNativeDefault::OutboundAttractionDistribution, ForceAtlas2(params)) => {
            json!(params.outbound_attraction_distribution)
        },
        (CugraphNativeDefault::LinLogMode, ForceAtlas2(params)) => json!(params.lin_log_mode),
        (CugraphNativeDefault::PreventOverlapping, ForceAtlas2(params)) => json!(params.prevent_overlapping),
        (CugraphNativeDefault::OverlapScalingRatio, ForceAtlas2(params)) => json!(params.overlap_scaling_ratio),
        (CugraphNativeDefault::EdgeWeightInfluence, ForceAtlas2(params)) => json!(params.edge_weight_influence),
        (CugraphNativeDefault::JitterTolerance, ForceAtlas2(params)) => json!(params.jitter_tolerance),
        (CugraphNativeDefault::BarnesHutOptimize, ForceAtlas2(params)) => json!(params.barnes_hut_optimize),
        (CugraphNativeDefault::BarnesHutTheta, ForceAtlas2(params)) => json!(params.barnes_hut_theta),
        (CugraphNativeDefault::ScalingRatio, ForceAtlas2(params)) => json!(params.scaling_ratio),
        (CugraphNativeDefault::StrongGravityMode, ForceAtlas2(params)) => json!(params.strong_gravity_mode),
        (CugraphNativeDefault::Gravity, ForceAtlas2(params)) => json!(params.gravity),
        (CugraphNativeDefault::Verbose, ForceAtlas2(params)) => json!(params.verbose),
        (field, params) => panic!("cuGraph argument default {field:?} is not defined for {params:?}"),
    })
}
