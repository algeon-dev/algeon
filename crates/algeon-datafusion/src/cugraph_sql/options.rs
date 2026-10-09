use serde_json::{Map, Value};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphBetweennessSeeds, CugraphBfsOutputMode, CugraphBfsSource,
    CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphKCoreDegreeType, CugraphPageRankParams,
    CugraphSpectralClusteringParams, CugraphSqlConfig, CugraphVertexSelector,
    definitions::{CugraphArgumentKind, CugraphValueDefinition, arguments_for_algorithm},
    error::{invalid_option, plan, unknown_option, unsupported_option},
};
use crate::{
    error::{Error, Result},
    gpu_functions::unknown_name_message,
};

pub(super) fn default_options(
    algorithm: CugraphAlgorithm,
    config: &CugraphSqlConfig,
    source: Option<CugraphBfsSource>,
) -> Result<(CugraphGraphOptions, CugraphAlgorithmParams)> {
    let mut graph_options = config.default_graph_options.clone();
    // Several algorithms require (or are only semantically meaningful on)
    // undirected graphs. Set directed=false before parsing user-supplied
    // options so default calls match the Python cuGraph oracle semantics.
    //
    // - WCC: cuGraph WCC requires a symmetric graph by API contract.
    // - triangle_count_all / core_number / k_core: Python cuGraph enforces
    //   undirected; libcugraph accepts directed but produces different
    //   (non-comparable) results, so the adapter default aligns with the
    //   Python oracle to keep parity meaningful.
    // - minimum_spanning_tree / spectral_modularity_maximization: cuGraph
    //   requires an undirected/symmetric graph for these cuGraph C API operations.
    graph_options = algorithm.effective_default_graph_options(&graph_options);
    let params = CugraphAlgorithmParams::from_native_defaults(algorithm, source)?;
    Ok((graph_options, params))
}

pub(super) fn validate_graph_directedness(
    algorithm: CugraphAlgorithm,
    graph_options: &CugraphGraphOptions,
) -> Result<()> {
    let Some(required) = algorithm.native_algorithm().definition().required_graph_directedness() else {
        return Ok(());
    };
    if graph_options.directed == required {
        return Ok(());
    }
    if algorithm.is_similarity() && !required {
        return Err(unsupported_option(format!(
            "{} requires directed=false so cuGraph receives a symmetric graph",
            algorithm.function_name()
        )));
    }
    let message = match (algorithm, required) {
        (CugraphAlgorithm::StronglyConnectedComponents, true) => {
            "cugraph_strongly_connected_components requires directed=true; use \
             cugraph_weakly_connected_components for weak/undirected components"
        },
        (CugraphAlgorithm::WeaklyConnectedComponents, false) => {
            "cugraph_weakly_connected_components requires directed=false so cuGraph receives a symmetric graph"
        },
        (CugraphAlgorithm::MinimumSpanningTree, false) => {
            "cugraph_minimum_spanning_tree requires directed=false so cuGraph receives an undirected graph"
        },
        (CugraphAlgorithm::SpectralModularityMaximization, false) => {
            "cugraph_spectral_modularity_maximization requires directed=false so cuGraph receives an undirected graph"
        },
        _ if required => "cuGraph algorithm requires directed=true",
        _ => "cuGraph algorithm requires directed=false",
    };
    Err(unsupported_option(message))
}

pub(super) fn apply_options(
    algorithm: CugraphAlgorithm,
    object: &Map<String, Value>,
    graph_options: &mut CugraphGraphOptions,
    params: &mut CugraphAlgorithmParams,
) -> Result<()> {
    for (key, value) in object {
        apply_option(algorithm, key, value, graph_options, params)
            .map_err(|error| annotate_option_error(error, algorithm, key))?;
    }

    let cross_option = match params {
        CugraphAlgorithmParams::BetweennessCentrality(_) | CugraphAlgorithmParams::EdgeBetweennessCentrality(_) => {
            validate_betweenness_options(params).map_err(|error| error.with_log_safe_fact("argument", "seeds"))
        },
        CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
            validate_spectral_options(params).map_err(|error| error.with_log_safe_fact("argument", "n_eigenvectors"))
        },
        _ => Ok(()),
    };
    cross_option.map_err(|error| error.with_log_safe_fact("cugraph_operation", algorithm.function_name()))
}

/// Rejects the first option name the function does not declare as a value
/// argument, before any option value is interpreted.
pub(super) fn require_declared_options(algorithm: CugraphAlgorithm, object: &Map<String, Value>) -> Result<()> {
    for key in object.keys() {
        declared_value_definition(algorithm, key).map_err(|error| annotate_option_error(error, algorithm, key))?;
    }
    Ok(())
}

fn annotate_option_error(error: Error, algorithm: CugraphAlgorithm, key: &str) -> Error {
    error.with_fact("argument", key).with_log_safe_fact("cugraph_operation", algorithm.function_name())
}

fn apply_option(
    algorithm: CugraphAlgorithm,
    key: &str,
    value: &Value,
    graph_options: &mut CugraphGraphOptions,
    params: &mut CugraphAlgorithmParams,
) -> Result<()> {
    let definition = declared_value_definition(algorithm, key)?;
    require_declared_range(algorithm, key, &definition, value)?;
    match key {
        "directed" => graph_options.directed = json_bool(value, key)?,
        "renumber" => graph_options.renumber = json_bool(value, key)?,
        "construction_policy" => {
            let policy = json_string(value, key)?;
            graph_options.construction_policy = match policy.as_str() {
                "python_cugraph" => CugraphGraphConstructionPolicy::PythonCugraph,
                "raw_libcugraph" => CugraphGraphConstructionPolicy::RawLibcugraph,
                other => {
                    return Err(invalid_option(format!(
                        "unsupported cuGraph construction_policy '{other}', expected 'python_cugraph' or 'raw_libcugraph'"
                    )));
                },
            };
        },
        "alpha" if is_pagerank_algorithm(algorithm) => {
            let alpha = json_f64(value, key)?;
            if let Some(params) = pagerank_params_mut(params) {
                params.alpha = alpha;
            }
        },
        "epsilon" if is_pagerank_algorithm(algorithm) => {
            let epsilon = json_f64(value, key)?;
            if let Some(params) = pagerank_params_mut(params) {
                params.epsilon = epsilon;
            }
        },
        "max_iterations" if is_pagerank_algorithm(algorithm) => {
            let max_iterations = json_u32(value, key)?;
            if let Some(params) = pagerank_params_mut(params) {
                params.max_iterations = max_iterations;
            }
        },
        "depth_limit" if algorithm == CugraphAlgorithm::Bfs => {
            if let CugraphAlgorithmParams::Bfs(params) = params {
                params.depth_limit = match value {
                    Value::Null => None,
                    _ => Some(json_i64(value, key)?),
                };
            }
        },
        "output_mode" if algorithm == CugraphAlgorithm::Bfs => {
            let mode = match json_string(value, key)?.as_str() {
                "raw" => CugraphBfsOutputMode::Raw,
                "normalized" => CugraphBfsOutputMode::Normalized,
                "path" => CugraphBfsOutputMode::Path,
                other => {
                    return Err(invalid_option(format!(
                        "unsupported BFS output_mode '{other}', expected 'raw', 'normalized', or 'path'"
                    )));
                },
            };
            if let CugraphAlgorithmParams::Bfs(params) = params {
                params.output.mode = mode;
            }
        },
        "return_target_info" if algorithm == CugraphAlgorithm::Bfs => {
            if let CugraphAlgorithmParams::Bfs(params) = params {
                params.output.return_target_info = json_bool(value, key)?;
            }
        },
        "max_level"
            if matches!(algorithm, CugraphAlgorithm::Louvain | CugraphAlgorithm::Leiden | CugraphAlgorithm::Ecg) =>
        {
            let max_level = json_u32(value, key)?;
            match params {
                CugraphAlgorithmParams::Louvain(params) => params.max_level = max_level,
                CugraphAlgorithmParams::Leiden(params) => params.max_level = max_level,
                CugraphAlgorithmParams::Ecg(params) => params.max_level = max_level,
                _ => {},
            }
        },
        "threshold" if matches!(algorithm, CugraphAlgorithm::Louvain | CugraphAlgorithm::Ecg) => {
            let threshold = json_f64(value, key)?;
            match params {
                CugraphAlgorithmParams::Louvain(params) => params.threshold = threshold,
                CugraphAlgorithmParams::Ecg(params) => params.threshold = threshold,
                _ => {},
            }
        },
        "resolution"
            if matches!(algorithm, CugraphAlgorithm::Louvain | CugraphAlgorithm::Leiden | CugraphAlgorithm::Ecg) =>
        {
            let resolution = json_f64(value, key)?;
            match params {
                CugraphAlgorithmParams::Louvain(params) => params.resolution = resolution,
                CugraphAlgorithmParams::Leiden(params) => params.resolution = resolution,
                CugraphAlgorithmParams::Ecg(params) => params.resolution = resolution,
                _ => {},
            }
        },
        "epsilon" if algorithm == CugraphAlgorithm::EigenvectorCentrality => {
            let epsilon = json_f64(value, key)?;
            if let CugraphAlgorithmParams::EigenvectorCentrality(params) = params {
                params.epsilon = epsilon;
            }
        },
        "max_iterations" if algorithm == CugraphAlgorithm::EigenvectorCentrality => {
            let max_iterations = json_u32(value, key)?;
            if let CugraphAlgorithmParams::EigenvectorCentrality(params) = params {
                params.max_iterations = max_iterations;
            }
        },
        "alpha" if algorithm == CugraphAlgorithm::KatzCentrality => {
            let alpha = json_f64(value, key)?;
            if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                params.alpha = alpha;
            }
        },
        "beta" if algorithm == CugraphAlgorithm::KatzCentrality => {
            let beta = json_f64(value, key)?;
            if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                params.beta = beta;
            }
        },
        "epsilon" if algorithm == CugraphAlgorithm::KatzCentrality => {
            let epsilon = json_f64(value, key)?;
            if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                params.epsilon = epsilon;
            }
        },
        "max_iterations" if algorithm == CugraphAlgorithm::KatzCentrality => {
            let max_iterations = json_u32(value, key)?;
            if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                params.max_iterations = max_iterations;
            }
        },
        "epsilon" if algorithm == CugraphAlgorithm::Hits => {
            let epsilon = json_f64(value, key)?;
            if let CugraphAlgorithmParams::Hits(params) = params {
                params.epsilon = epsilon;
            }
        },
        "max_iterations" if algorithm == CugraphAlgorithm::Hits => {
            let max_iterations = json_u32(value, key)?;
            if let CugraphAlgorithmParams::Hits(params) = params {
                params.max_iterations = max_iterations;
            }
        },
        "normalize" if algorithm == CugraphAlgorithm::Hits => {
            let normalize = json_bool(value, key)?;
            if let CugraphAlgorithmParams::Hits(params) = params {
                params.normalize = normalize;
            }
        },
        "seeds" if is_betweenness_algorithm(algorithm) => {
            let seeds = parse_betweenness_seeds(value, key)?;
            match params {
                CugraphAlgorithmParams::BetweennessCentrality(params) => params.seeds = seeds,
                CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => params.seeds = seeds,
                _ => {},
            }
        },
        "k" if is_betweenness_algorithm(algorithm) => {
            let k = match value {
                Value::Null => None,
                _ => Some(json_u64(value, key)?),
            };
            match params {
                CugraphAlgorithmParams::BetweennessCentrality(params) => params.k = k,
                CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => params.k = k,
                _ => {},
            }
        },
        "normalized" if is_betweenness_algorithm(algorithm) => {
            let normalized = json_bool(value, key)?;
            match params {
                CugraphAlgorithmParams::BetweennessCentrality(params) => params.normalized = normalized,
                CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => params.normalized = normalized,
                _ => {},
            }
        },
        "include_endpoints" if algorithm == CugraphAlgorithm::BetweennessCentrality => {
            let include_endpoints = json_bool(value, key)?;
            if let CugraphAlgorithmParams::BetweennessCentrality(params) = params {
                params.include_endpoints = include_endpoints;
            }
        },
        "exact_vertex_threshold" if is_betweenness_algorithm(algorithm) => {
            let exact_vertex_threshold = json_u64(value, key)?;
            match params {
                CugraphAlgorithmParams::BetweennessCentrality(params) => {
                    params.exact_vertex_threshold = exact_vertex_threshold
                },
                CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
                    params.exact_vertex_threshold = exact_vertex_threshold
                },
                _ => {},
            }
        },
        "cutoff" if algorithm == CugraphAlgorithm::Sssp => {
            if let CugraphAlgorithmParams::Sssp(params) = params {
                params.cutoff = match value {
                    Value::Null => None,
                    _ => Some(json_f64(value, key)?),
                };
            }
        },
        "degree_type" if matches!(algorithm, CugraphAlgorithm::CoreNumber | CugraphAlgorithm::KCore) => {
            let degree_type = parse_degree_type(json_string(value, key)?.as_str(), key)?;
            match params {
                CugraphAlgorithmParams::CoreNumber(params) => params.degree_type = degree_type,
                CugraphAlgorithmParams::KCore(params) => params.degree_type = degree_type,
                _ => {},
            }
        },
        "k" if algorithm == CugraphAlgorithm::KCore => {
            let k = json_u32(value, key)?;
            if let CugraphAlgorithmParams::KCore(params) = params {
                params.k = k;
            }
        },
        "seed"
            if algorithm == CugraphAlgorithm::Leiden
                || algorithm == CugraphAlgorithm::Ecg
                || algorithm == CugraphAlgorithm::SpectralModularityMaximization
                || algorithm == CugraphAlgorithm::ForceAtlas2 =>
        {
            let seed = json_u64(value, key)?;
            match params {
                CugraphAlgorithmParams::Leiden(params) => params.seed = seed,
                CugraphAlgorithmParams::Ecg(params) => params.seed = seed,
                CugraphAlgorithmParams::SpectralModularityMaximization(params) => params.seed = seed,
                CugraphAlgorithmParams::ForceAtlas2(params) => params.seed = seed,
                _ => {},
            }
        },
        "min_weight" if algorithm == CugraphAlgorithm::Ecg => {
            let min_weight = json_f64(value, key)?;
            if let CugraphAlgorithmParams::Ecg(params) = params {
                params.min_weight = min_weight;
            }
        },
        "ensemble_size" if algorithm == CugraphAlgorithm::Ecg => {
            let ensemble_size = json_u32(value, key)?;
            if let CugraphAlgorithmParams::Ecg(params) = params {
                params.ensemble_size = ensemble_size;
            }
        },
        "n_clusters" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let n_clusters = json_u32(value, key)?;
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.n_clusters = n_clusters;
            }
        },
        "n_eigenvectors" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let n_eigenvectors = json_u32(value, key)?;
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.n_eigenvectors = n_eigenvectors;
            }
        },
        "evs_tolerance" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let evs_tolerance = json_f64(value, key)?;
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.evs_tolerance = evs_tolerance;
            }
        },
        "max_iter" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let max_iter = json_u32(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.max_iter = max_iter;
            }
        },
        "outbound_attraction_distribution" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let value = json_bool(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.outbound_attraction_distribution = value;
            }
        },
        "lin_log_mode" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let value = json_bool(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.lin_log_mode = value;
            }
        },
        "prevent_overlapping" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let value = json_bool(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.prevent_overlapping = value;
            }
        },
        "overlap_scaling_ratio" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let overlap_scaling_ratio = json_f64(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.overlap_scaling_ratio = overlap_scaling_ratio;
            }
        },
        "edge_weight_influence" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let edge_weight_influence = json_f64(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.edge_weight_influence = edge_weight_influence;
            }
        },
        "jitter_tolerance" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let jitter_tolerance = json_f64(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.jitter_tolerance = jitter_tolerance;
            }
        },
        "barnes_hut_optimize" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let value = json_bool(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.barnes_hut_optimize = value;
            }
        },
        "barnes_hut_theta" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let barnes_hut_theta = json_f64(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.barnes_hut_theta = barnes_hut_theta;
            }
        },
        "scaling_ratio" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let scaling_ratio = json_f64(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.scaling_ratio = scaling_ratio;
            }
        },
        "strong_gravity_mode" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let value = json_bool(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.strong_gravity_mode = value;
            }
        },
        "gravity" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let gravity = json_f64(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.gravity = gravity;
            }
        },
        "verbose" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let value = json_bool(value, key)?;
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.verbose = value;
            }
        },
        _ => {
            return Err(plan(format!(
                "{} declares option '{key}' but no parameter binds it",
                algorithm.function_name()
            )));
        },
    }
    Ok(())
}

fn is_pagerank_algorithm(algorithm: CugraphAlgorithm) -> bool {
    matches!(algorithm, CugraphAlgorithm::PageRank | CugraphAlgorithm::PersonalizedPageRank)
}

fn is_betweenness_algorithm(algorithm: CugraphAlgorithm) -> bool {
    matches!(algorithm, CugraphAlgorithm::BetweennessCentrality | CugraphAlgorithm::EdgeBetweennessCentrality)
}

fn parse_betweenness_seeds(value: &Value, key: &str) -> Result<Option<CugraphBetweennessSeeds>> {
    if value.is_null() {
        return Ok(None);
    }
    let values = value
        .as_array()
        .ok_or_else(|| invalid_option(format!("cuGraph option '{key}' must be an integer or string array")))?;
    if values.is_empty() {
        return Err(invalid_option("betweenness seeds must not be empty"));
    }
    let vertices = values
        .iter()
        .map(|value| {
            if let Some(value) = value.as_i64() {
                Ok(CugraphVertexSelector::Int64(value))
            } else if let Some(value) = value.as_str() {
                Ok(CugraphVertexSelector::Utf8(value.to_string()))
            } else {
                Err(invalid_option(format!("cuGraph option '{key}' entries must be integers or strings")))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let string_domain = vertices.first().is_some_and(CugraphVertexSelector::is_utf8);
    if vertices.iter().any(|vertex| vertex.is_utf8() != string_domain) {
        return Err(invalid_option("betweenness seeds must not mix integer and string vertices"));
    }
    Ok(Some(CugraphBetweennessSeeds { vertices }))
}

fn validate_betweenness_options(params: &CugraphAlgorithmParams) -> Result<()> {
    match params {
        CugraphAlgorithmParams::BetweennessCentrality(params) if params.seeds.is_some() && params.k.is_some() => {
            return Err(invalid_option("betweenness seeds and k are mutually exclusive"));
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) if params.seeds.is_some() && params.k.is_some() => {
            return Err(invalid_option("betweenness seeds and k are mutually exclusive"));
        },
        _ => {},
    }
    Ok(())
}

/// libcugraph rejects more eigenvectors than clusters
/// (`legacy/spectral_clustering.cu`); no per-option range can express that.
fn validate_spectral_options(params: &CugraphSpectralClusteringParams) -> Result<()> {
    if params.n_eigenvectors > params.n_clusters {
        return Err(invalid_option(format!(
            "spectral n_eigenvectors ({}) must not exceed n_clusters ({})",
            params.n_eigenvectors, params.n_clusters
        )));
    }
    Ok(())
}

fn pagerank_params_mut(params: &mut CugraphAlgorithmParams) -> Option<&mut CugraphPageRankParams> {
    match params {
        CugraphAlgorithmParams::PageRank(params) => Some(params),
        CugraphAlgorithmParams::PersonalizedPageRank(params) => Some(&mut params.pagerank),
        _ => None,
    }
}

fn parse_degree_type(value: &str, key: &str) -> Result<CugraphKCoreDegreeType> {
    match value {
        "in" => Ok(CugraphKCoreDegreeType::In),
        "out" => Ok(CugraphKCoreDegreeType::Out),
        "in_out" => Ok(CugraphKCoreDegreeType::InOut),
        other => Err(invalid_option(format!("unsupported '{key}' value '{other}', expected 'in', 'out', or 'in_out'"))),
    }
}

fn json_bool(value: &Value, key: &str) -> Result<bool> {
    value.as_bool().ok_or_else(|| invalid_option(format!("cuGraph option '{key}' must be a boolean")))
}

fn json_string(value: &Value, key: &str) -> Result<String> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| invalid_option(format!("cuGraph option '{key}' must be a string")))
}

fn json_f64(value: &Value, key: &str) -> Result<f64> {
    let parsed = value.as_f64().ok_or_else(|| invalid_option(format!("cuGraph option '{key}' must be a number")))?;
    if parsed.is_finite() { Ok(parsed) } else { Err(invalid_option(format!("cuGraph option '{key}' must be finite"))) }
}

/// The descriptor of the value argument `key`, the single source of the option
/// names, value kinds and ranges this function accepts.
fn declared_value_definition(algorithm: CugraphAlgorithm, key: &str) -> Result<CugraphValueDefinition> {
    let arguments = arguments_for_algorithm(algorithm);
    arguments
        .iter()
        .find_map(|argument| match argument.kind {
            CugraphArgumentKind::Value(definition) if argument.name == key => Some(definition),
            _ => None,
        })
        .ok_or_else(|| {
            let declared_values = arguments
                .iter()
                .filter(|argument| matches!(argument.kind, CugraphArgumentKind::Value(_)))
                .map(|argument| argument.name);
            unknown_option(unknown_name_message("option", key, algorithm.function_name(), declared_values))
        })
}

/// Rejects a number outside the range the descriptor declares, so the bounds
/// advertised through `gpu_describe_function` are the bounds enforced before
/// any graph is staged. Null and non-numeric values pass here; the typed
/// extraction in [`apply_option`] reports them.
fn require_declared_range(
    algorithm: CugraphAlgorithm,
    key: &str,
    definition: &CugraphValueDefinition,
    value: &Value,
) -> Result<()> {
    let (Some(range), Value::Number(number)) = (definition.range, value) else {
        return Ok(());
    };
    if range.contains(number) == Some(false) {
        return Err(invalid_option(format!(
            "{} option '{key}' must be {range}, got {number}",
            algorithm.function_name()
        )));
    }
    Ok(())
}

fn json_i64(value: &Value, key: &str) -> Result<i64> {
    value.as_i64().ok_or_else(|| invalid_option(format!("cuGraph option '{key}' must be an integer")))
}

fn json_u32(value: &Value, key: &str) -> Result<u32> {
    let parsed =
        value.as_u64().ok_or_else(|| invalid_option(format!("cuGraph option '{key}' must be an unsigned integer")))?;
    u32::try_from(parsed).map_err(|_| invalid_option(format!("cuGraph option '{key}' exceeds u32::MAX")))
}

fn json_u64(value: &Value, key: &str) -> Result<u64> {
    value.as_u64().ok_or_else(|| invalid_option(format!("cuGraph option '{key}' must be an unsigned 64-bit integer")))
}
