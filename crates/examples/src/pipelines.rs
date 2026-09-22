//! Predefined citation-network pipelines exposed by the example HTTP API.
//!
//! This module validates request parameters and renders them into SQL or
//! cuGraph SQL calls that run through the shared DataFusion session.

use std::collections::BTreeMap;

use datafusion::common::ScalarValue;
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy)]
pub struct PipelineRegistry {
    pipelines: &'static [Pipeline],
}

impl Default for PipelineRegistry {
    fn default() -> Self {
        Self { pipelines: PIPELINES }
    }
}

impl PipelineRegistry {
    pub fn all(&self) -> &'static [Pipeline] {
        self.pipelines
    }

    pub fn get(&self, name: &str) -> Option<&'static Pipeline> {
        self.pipelines.iter().find(|pipeline| pipeline.name == name)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Pipeline {
    pub name: &'static str,
    pub capability: &'static str,
    pub description: &'static str,
    params: &'static [ParamSpec],
    kind: PipelineKind,
}

impl Pipeline {
    pub fn summary(&self) -> PipelineSummary {
        PipelineSummary {
            name: self.name,
            capability: self.capability,
            description: self.description,
            params: self.params.iter().map(ParamSpec::schema).collect(),
        }
    }

    pub fn render_request(&self, body: &Value) -> Result<RenderedPipeline, PipelineValidationError> {
        let values = request_values_from_body(body)?;
        self.render(values, MissingPolicy::Require)
    }

    pub fn render_defaults(&self) -> Result<RenderedPipeline, PipelineValidationError> {
        self.render(None, MissingPolicy::UseDefault)
    }

    fn render(
        &self,
        values: Option<&serde_json::Map<String, Value>>,
        missing_policy: MissingPolicy,
    ) -> Result<RenderedPipeline, PipelineValidationError> {
        validate_no_unknown_params(values, self.params)?;
        match self.kind {
            PipelineKind::SqlAggregateTopk => {
                let year_from = int_param(self, values, "year_from", missing_policy)?;
                let year_to = int_param(self, values, "year_to", missing_policy)?;
                if year_from > year_to {
                    return Err(PipelineValidationError::new(
                        "invalid_param_range",
                        "year_from must be less than or equal to year_to",
                    )
                    .with_param("year_from"));
                }
                let limit = int_param(self, values, "limit", missing_policy)?;
                Ok(RenderedPipeline {
                    sql: format!(
                        "SELECT paper_id, title, year, n_citation \
                         FROM papers \
                         WHERE year BETWEEN $1 AND $2 \
                         ORDER BY n_citation DESC NULLS LAST, paper_id ASC \
                         LIMIT {limit}"
                    ),
                    param_values: vec![
                        ScalarValue::Int16(Some(year_from as i16)),
                        ScalarValue::Int16(Some(year_to as i16)),
                    ],
                })
            },
            PipelineKind::SqlJoinFacets => {
                let fos_name = string_param(self, values, "fos_name", missing_policy)?;
                Ok(RenderedPipeline {
                    sql: "SELECT p.year, COUNT(*) AS papers, AVG(p.n_citation) AS avg_citation \
                          FROM paper_fos f \
                          JOIN papers p ON p.paper_id = f.paper_id \
                          WHERE f.fos_name = $1 \
                          GROUP BY p.year \
                          ORDER BY p.year"
                        .to_string(),
                    param_values: vec![ScalarValue::Utf8(Some(fos_name))],
                })
            },
            PipelineKind::GraphPagerankTopk => {
                let limit = int_param(self, values, "limit", missing_policy)?;
                Ok(RenderedPipeline {
                    sql: format!(
                        "WITH pr AS ( \
                           SELECT vertex, value \
                           FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM citation_edges)) \
                         ) \
                         SELECT p.paper_id, p.title, p.year, pr.value AS pagerank \
                         FROM pr \
                         JOIN papers p ON p.paper_id = pr.vertex \
                         ORDER BY pr.value DESC, p.paper_id ASC \
                         LIMIT {limit}"
                    ),
                    param_values: Vec::new(),
                })
            },
            PipelineKind::GraphBfsHops => {
                let seed = int_param(self, values, "seed_paper_id", missing_policy)?;
                let max_depth = int_param(self, values, "max_depth", missing_policy)?;
                Ok(RenderedPipeline {
                    sql: format!(
                        "WITH bfs AS ( \
                           SELECT vertex, distance, reachable \
                           FROM cugraph_bfs(\
                             edges => (SELECT src, dst FROM citation_edges), source_vertex => {seed}, \
                             depth_limit => {max_depth}, output_mode => 'normalized') \
                         ) \
                         SELECT b.distance, COUNT(*) AS papers, MIN(p.title) AS sample_title \
                         FROM bfs b \
                         JOIN papers p ON p.paper_id = b.vertex \
                         WHERE b.reachable \
                         GROUP BY b.distance \
                         ORDER BY b.distance"
                    ),
                    param_values: Vec::new(),
                })
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct PipelineSummary {
    pub name: &'static str,
    pub capability: &'static str,
    pub description: &'static str,
    pub params: Vec<ParamSchema>,
}

#[derive(Debug, Serialize)]
pub struct ParamSchema {
    pub name: &'static str,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<i64>,
    pub default: Value,
    pub description: &'static str,
}

#[derive(Debug, Clone)]
pub struct RenderedPipeline {
    pub sql: String,
    pub param_values: Vec<ScalarValue>,
}

#[derive(Debug, Clone)]
pub struct PipelineValidationError {
    code: &'static str,
    message: String,
    param: Option<&'static str>,
}

impl PipelineValidationError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), param: None }
    }

    fn with_param(mut self, param: &'static str) -> Self {
        self.param = Some(param);
        self
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn param(&self) -> Option<&'static str> {
        self.param
    }
}

#[derive(Debug, Clone, Copy)]
enum PipelineKind {
    SqlAggregateTopk,
    SqlJoinFacets,
    GraphPagerankTopk,
    GraphBfsHops,
}

#[derive(Debug, Clone, Copy)]
enum ParamKind {
    Int64,
    Utf8,
}

impl ParamKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Int64 => "int64",
            Self::Utf8 => "utf8",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ParamDefault {
    Int(i64),
    Str(&'static str),
}

impl ParamDefault {
    fn to_json(self) -> Value {
        match self {
            Self::Int(value) => json!(value),
            Self::Str(value) => json!(value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ParamSpec {
    name: &'static str,
    kind: ParamKind,
    required: bool,
    min: Option<i64>,
    max: Option<i64>,
    default: ParamDefault,
    description: &'static str,
}

impl ParamSpec {
    fn schema(&self) -> ParamSchema {
        ParamSchema {
            name: self.name,
            kind: self.kind.as_str(),
            required: self.required,
            min: self.min,
            max: self.max,
            default: self.default.to_json(),
            description: self.description,
        }
    }
}

fn request_values_from_body(body: &Value) -> Result<Option<&serde_json::Map<String, Value>>, PipelineValidationError> {
    let object = body.as_object().ok_or_else(|| {
        PipelineValidationError::new("invalid_request_body", "pipeline parameters must be a JSON object")
    })?;
    Ok(Some(object))
}

#[derive(Debug, Clone, Copy)]
enum MissingPolicy {
    Require,
    UseDefault,
}

const AGGREGATE_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        name: "year_from",
        kind: ParamKind::Int64,
        required: true,
        min: Some(1800),
        max: Some(2200),
        default: ParamDefault::Int(2010),
        description: "Inclusive lower publication year.",
    },
    ParamSpec {
        name: "year_to",
        kind: ParamKind::Int64,
        required: true,
        min: Some(1800),
        max: Some(2200),
        default: ParamDefault::Int(2020),
        description: "Inclusive upper publication year.",
    },
    ParamSpec {
        name: "limit",
        kind: ParamKind::Int64,
        required: true,
        min: Some(1),
        max: Some(1000),
        default: ParamDefault::Int(10),
        description: "Maximum result rows.",
    },
];

const JOIN_PARAMS: &[ParamSpec] = &[ParamSpec {
    name: "fos_name",
    kind: ParamKind::Utf8,
    required: true,
    min: None,
    max: Some(256),
    default: ParamDefault::Str("Artificial intelligence"),
    description: "Field-of-study name matched in paper_fos.",
}];

const PAGERANK_PARAMS: &[ParamSpec] = &[ParamSpec {
    name: "limit",
    kind: ParamKind::Int64,
    required: true,
    min: Some(1),
    max: Some(1000),
    default: ParamDefault::Int(10),
    description: "Maximum result rows.",
}];

const BFS_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        name: "seed_paper_id",
        kind: ParamKind::Int64,
        required: true,
        min: Some(0),
        max: None,
        default: ParamDefault::Int(2_963_403_868),
        description: "Raw AMiner paper id used as the BFS source vertex.",
    },
    ParamSpec {
        name: "max_depth",
        kind: ParamKind::Int64,
        required: true,
        min: Some(0),
        max: Some(8),
        default: ParamDefault::Int(2),
        description: "Maximum BFS depth in hops.",
    },
];

const PIPELINES: &[Pipeline] = &[
    Pipeline {
        name: "sql_aggregate_topk",
        capability: "tabular_filter_sort_limit",
        description: "Top cited papers in a publication-year range.",
        params: AGGREGATE_PARAMS,
        kind: PipelineKind::SqlAggregateTopk,
    },
    Pipeline {
        name: "sql_join_facets",
        capability: "join_group_aggregate",
        description: "Per-year paper counts and citation averages for one field of study.",
        params: JOIN_PARAMS,
        kind: PipelineKind::SqlJoinFacets,
    },
    Pipeline {
        name: "graph_pagerank_topk",
        capability: "graph_centrality_join_back",
        description: "PageRank scores joined back to paper metadata.",
        params: PAGERANK_PARAMS,
        kind: PipelineKind::GraphPagerankTopk,
    },
    Pipeline {
        name: "graph_bfs_hops",
        capability: "graph_traversal_aggregate",
        description: "Breadth-first traversal grouped by hop distance.",
        params: BFS_PARAMS,
        kind: PipelineKind::GraphBfsHops,
    },
];

fn validate_no_unknown_params(
    values: Option<&serde_json::Map<String, Value>>,
    params: &[ParamSpec],
) -> Result<(), PipelineValidationError> {
    let Some(values) = values else {
        return Ok(());
    };
    for key in values.keys() {
        if !params.iter().any(|param| param.name == key) {
            return Err(PipelineValidationError::new("unknown_param", format!("unknown pipeline parameter `{key}`")));
        }
    }
    Ok(())
}

fn int_param(
    pipeline: &Pipeline,
    values: Option<&serde_json::Map<String, Value>>,
    name: &'static str,
    missing_policy: MissingPolicy,
) -> Result<i64, PipelineValidationError> {
    let spec = param_spec(pipeline, name);
    let value = match values.and_then(|values| values.get(name)) {
        Some(value) => value.as_i64().ok_or_else(|| {
            PipelineValidationError::new("invalid_param_type", format!("`{name}` must be an integer")).with_param(name)
        })?,
        None if matches!(missing_policy, MissingPolicy::UseDefault) => match spec.default {
            ParamDefault::Int(value) => value,
            ParamDefault::Str(_) => unreachable!("integer param has string default"),
        },
        None => {
            return Err(PipelineValidationError::new("missing_param", format!("missing required parameter `{name}`"))
                .with_param(name));
        },
    };
    if let Some(min) = spec.min
        && value < min
    {
        return Err(PipelineValidationError::new(
            "invalid_param_range",
            format!("`{name}` must be greater than or equal to {min}"),
        )
        .with_param(name));
    }
    if let Some(max) = spec.max
        && value > max
    {
        return Err(PipelineValidationError::new(
            "invalid_param_range",
            format!("`{name}` must be less than or equal to {max}"),
        )
        .with_param(name));
    }
    Ok(value)
}

fn string_param(
    pipeline: &Pipeline,
    values: Option<&serde_json::Map<String, Value>>,
    name: &'static str,
    missing_policy: MissingPolicy,
) -> Result<String, PipelineValidationError> {
    let spec = param_spec(pipeline, name);
    let value = match values.and_then(|values| values.get(name)) {
        Some(value) => value.as_str().ok_or_else(|| {
            PipelineValidationError::new("invalid_param_type", format!("`{name}` must be a string")).with_param(name)
        })?,
        None if matches!(missing_policy, MissingPolicy::UseDefault) => match spec.default {
            ParamDefault::Str(value) => value,
            ParamDefault::Int(_) => unreachable!("string param has integer default"),
        },
        None => {
            return Err(PipelineValidationError::new("missing_param", format!("missing required parameter `{name}`"))
                .with_param(name));
        },
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(
            PipelineValidationError::new("invalid_param_range", format!("`{name}` must not be empty")).with_param(name)
        );
    }
    if let Some(max) = spec.max
        && trimmed.chars().count() > max as usize
    {
        return Err(PipelineValidationError::new(
            "invalid_param_range",
            format!("`{name}` must contain at most {max} characters"),
        )
        .with_param(name));
    }
    Ok(trimmed.to_string())
}

fn param_spec(pipeline: &Pipeline, name: &'static str) -> &'static ParamSpec {
    pipeline
        .params
        .iter()
        .find(|param| param.name == name)
        .expect("pipeline renderer requested an undeclared parameter")
}

pub fn summaries(registry: PipelineRegistry) -> Vec<PipelineSummary> {
    registry.all().iter().map(Pipeline::summary).collect()
}

pub fn default_param_body(pipeline: &Pipeline) -> Value {
    let params = pipeline
        .params
        .iter()
        .map(|param| (param.name.to_string(), param.default.to_json()))
        .collect::<BTreeMap<_, _>>();
    json!(params)
}
