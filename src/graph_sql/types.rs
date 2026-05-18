use std::{collections::BTreeSet, fmt, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::{DataFusionError, Result};

pub const CUGRAPH_PAGERANK_FUNCTION: &str = "cugraph_pagerank";
pub const CUGRAPH_BFS_FUNCTION: &str = "cugraph_bfs";
pub const CUGRAPH_LOUVAIN_FUNCTION: &str = "cugraph_louvain";

#[derive(Debug, Clone)]
pub struct CugraphSqlConfig {
    pub allowed_algorithms: BTreeSet<CugraphAlgorithm>,
    pub default_graph_options: CugraphGraphOptions,
    pub experimental_functions: bool,
}

impl Default for CugraphSqlConfig {
    fn default() -> Self {
        Self {
            allowed_algorithms: CugraphAlgorithm::all().into_iter().collect(),
            default_graph_options: CugraphGraphOptions::default(),
            experimental_functions: false,
        }
    }
}

impl CugraphSqlConfig {
    pub fn with_allowed_algorithms(mut self, algorithms: impl IntoIterator<Item = CugraphAlgorithm>) -> Self {
        self.allowed_algorithms = algorithms.into_iter().collect();
        self
    }

    pub(crate) fn allows(&self, algorithm: CugraphAlgorithm) -> bool {
        self.allowed_algorithms.contains(&algorithm)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CugraphAlgorithm {
    PageRank,
    Bfs,
    Louvain,
}

impl CugraphAlgorithm {
    pub fn all() -> [Self; 3] {
        [Self::PageRank, Self::Bfs, Self::Louvain]
    }

    pub fn function_name(self) -> &'static str {
        match self {
            Self::PageRank => CUGRAPH_PAGERANK_FUNCTION,
            Self::Bfs => CUGRAPH_BFS_FUNCTION,
            Self::Louvain => CUGRAPH_LOUVAIN_FUNCTION,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphAlgorithmSpec {
    pub algorithm: CugraphAlgorithm,
    pub table_name: CugraphTableReference,
    pub src_col: String,
    pub dst_col: String,
    pub weight_col: Option<String>,
    pub graph_options: CugraphGraphOptions,
    pub params: CugraphAlgorithmParams,
    pub output_schema: SchemaRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CugraphTableReference {
    parts: Vec<String>,
}

impl CugraphTableReference {
    pub fn parse(raw: &str) -> Result<Self> {
        let parts = raw.split('.').collect::<Vec<_>>();
        if !(1..=3).contains(&parts.len()) || parts.iter().any(|part| !is_simple_identifier(part)) {
            return Err(DataFusionError::Plan(format!(
                "cuGraph table source '{raw}' must be a simple table, schema.table, or catalog.schema.table identifier"
            )));
        }
        Ok(Self { parts: parts.into_iter().map(ToOwned::to_owned).collect() })
    }

    pub fn parts(&self) -> &[String] {
        &self.parts
    }
}

impl fmt::Display for CugraphTableReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.parts.join("."))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphGraphOptions {
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: CugraphGraphConstructionPolicy,
}

impl Default for CugraphGraphOptions {
    fn default() -> Self {
        Self { directed: true, renumber: true, construction_policy: CugraphGraphConstructionPolicy::PythonCugraph }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphGraphConstructionPolicy {
    PythonCugraph,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CugraphAlgorithmParams {
    PageRank(CugraphPageRankParams),
    Bfs(CugraphBfsParams),
    Louvain(CugraphLouvainParams),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphPageRankParams {
    pub alpha: f64,
    pub epsilon: f64,
    pub max_iterations: u32,
}

impl Default for CugraphPageRankParams {
    fn default() -> Self {
        Self { alpha: 0.85, epsilon: 1e-6, max_iterations: 100 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CugraphBfsParams {
    pub source_vertex: i64,
    pub depth_limit: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphLouvainParams {
    pub resolution: f64,
}

impl Default for CugraphLouvainParams {
    fn default() -> Self {
        Self { resolution: 1.0 }
    }
}

pub fn stable_result_schema(algorithm: CugraphAlgorithm) -> SchemaRef {
    let fields = match algorithm {
        CugraphAlgorithm::PageRank => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false)]
        },
        CugraphAlgorithm::Bfs => vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
        CugraphAlgorithm::Louvain => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false)]
        },
    };
    Arc::new(Schema::new(fields))
}

fn is_simple_identifier(part: &str) -> bool {
    let mut chars = part.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}
