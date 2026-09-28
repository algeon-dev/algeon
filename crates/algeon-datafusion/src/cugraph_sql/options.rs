use serde_json::{Map, Value};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphBetweennessSeeds, CugraphBfsOutputMode, CugraphBfsSource,
    CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphKCoreDegreeType, CugraphPageRankParams,
    CugraphSqlConfig, CugraphVertexSelector, error::unsupported_option, plan,
};
use crate::error::Result;

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
        return Err(plan(format!(
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
        apply_option(algorithm, key, value, graph_options, params).map_err(|error| error.with_fact("argument", key))?;
    }

    if is_betweenness_algorithm(algorithm) {
        validate_betweenness_options(params)?;
    }
    Ok(())
}

fn apply_option(
    algorithm: CugraphAlgorithm,
    key: &str,
    value: &Value,
    graph_options: &mut CugraphGraphOptions,
    params: &mut CugraphAlgorithmParams,
) -> Result<()> {
    match key {
        "directed" => graph_options.directed = json_bool(value, key)?,
        "renumber" => graph_options.renumber = json_bool(value, key)?,
        "construction_policy" => {
            let policy = json_string(value, key)?;
            graph_options.construction_policy = match policy.as_str() {
                "python_cugraph" => CugraphGraphConstructionPolicy::PythonCugraph,
                "raw_libcugraph" => CugraphGraphConstructionPolicy::RawLibcugraph,
                other => {
                    return Err(plan(format!(
                        "unsupported cuGraph construction_policy '{other}', expected 'python_cugraph' or 'raw_libcugraph'"
                    )));
                },
            };
        },
        "alpha" if is_pagerank_algorithm(algorithm) => {
            let alpha = json_f64(value, key)?;
            if !(0.0..=1.0).contains(&alpha) {
                return Err(plan("PageRank alpha must be in [0, 1]"));
            }
            if let Some(params) = pagerank_params_mut(params) {
                params.alpha = alpha;
            }
        },
        "epsilon" if is_pagerank_algorithm(algorithm) => {
            let epsilon = json_f64(value, key)?;
            if epsilon <= 0.0 {
                return Err(plan("PageRank epsilon must be positive"));
            }
            if let Some(params) = pagerank_params_mut(params) {
                params.epsilon = epsilon;
            }
        },
        "max_iterations" if is_pagerank_algorithm(algorithm) => {
            let max_iterations = json_u32(value, key)?;
            if max_iterations < 1 {
                return Err(plan("PageRank max_iterations must be at least 1"));
            }
            if let Some(params) = pagerank_params_mut(params) {
                params.max_iterations = max_iterations;
            }
        },
        "depth_limit" if algorithm == CugraphAlgorithm::Bfs => {
            if let CugraphAlgorithmParams::Bfs(params) = params {
                params.depth_limit = match value {
                    Value::Null => None,
                    _ => {
                        let depth_limit = json_i64(value, key)?;
                        if depth_limit < 0 {
                            return Err(plan("BFS depth_limit must be non-negative"));
                        }
                        Some(depth_limit)
                    },
                };
            }
        },
        "output_mode" if algorithm == CugraphAlgorithm::Bfs => {
            let mode = match json_string(value, key)?.as_str() {
                "raw" => CugraphBfsOutputMode::Raw,
                "normalized" => CugraphBfsOutputMode::Normalized,
                "path" => CugraphBfsOutputMode::Path,
                other => {
                    return Err(plan(format!(
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
        "max_level" if algorithm == CugraphAlgorithm::Louvain => {
            let max_level = json_u32(value, key)?;
            if max_level < 1 {
                return Err(plan("Louvain max_level must be at least 1"));
            }
            if let CugraphAlgorithmParams::Louvain(params) = params {
                params.max_level = max_level;
            }
        },
        "threshold" if algorithm == CugraphAlgorithm::Louvain => {
            let threshold = json_f64(value, key)?;
            if threshold < 0.0 {
                return Err(plan("Louvain threshold must be non-negative"));
            }
            if let CugraphAlgorithmParams::Louvain(params) = params {
                params.threshold = threshold;
            }
        },
        "resolution" if algorithm == CugraphAlgorithm::Louvain => {
            let resolution = json_f64(value, key)?;
            if resolution <= 0.0 {
                return Err(plan("Louvain resolution must be positive"));
            }
            if let CugraphAlgorithmParams::Louvain(params) = params {
                params.resolution = resolution;
            }
        },
        "epsilon" if algorithm == CugraphAlgorithm::EigenvectorCentrality => {
            let epsilon = json_f64(value, key)?;
            // libcugraph's eigenvector convergence test is
            //   diff_sum < vertex_count * epsilon
            // so epsilon=0 makes the threshold zero on any non-empty
            // graph: the solver never converges and burns
            // max_iterations of GPU work before failing. Reject the
            // value at planning so the failure surfaces in the SQL
            // layer instead of after a wasted GPU dispatch.
            if epsilon <= 0.0 {
                return Err(plan("eigenvector centrality epsilon must be positive"));
            }
            if let CugraphAlgorithmParams::EigenvectorCentrality(params) = params {
                params.epsilon = epsilon;
            }
        },
        "max_iterations" if algorithm == CugraphAlgorithm::EigenvectorCentrality => {
            let max_iterations = json_u32(value, key)?;
            if max_iterations < 1 {
                return Err(plan("eigenvector centrality max_iterations must be at least 1"));
            }
            if let CugraphAlgorithmParams::EigenvectorCentrality(params) = params {
                params.max_iterations = max_iterations;
            }
        },
        "alpha" if algorithm == CugraphAlgorithm::KatzCentrality => {
            let alpha = json_f64(value, key)?;
            // alpha must be in (0, 1]. libcugraph enforces alpha <= 1
            // (Katz attenuation factor must be smaller than the
            // inverse of the graph's largest eigenvalue, which is
            // upper-bounded by 1); alpha=0 makes Katz centrality
            // degenerate to the constant `beta` vector. Reject both
            // at planning so the failure surfaces before GPU work.
            if alpha <= 0.0 {
                return Err(plan("Katz centrality alpha must be positive"));
            }
            if alpha > 1.0 {
                return Err(plan("Katz centrality alpha must be <= 1"));
            }
            if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                params.alpha = alpha;
            }
        },
        "beta" if algorithm == CugraphAlgorithm::KatzCentrality => {
            let beta = json_f64(value, key)?;
            // beta=0 combined with the zero initial vector leaves the
            // solver iterating on all-zero scores; normalization then
            // produces an all-zero output that the kernel rejects.
            // Reject at planning since the `cugraph` crate's `NonNegativeF64`
            // wrapper accepts zero.
            if beta <= 0.0 {
                return Err(plan("Katz centrality beta must be positive"));
            }
            if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                params.beta = beta;
            }
        },
        "epsilon" if algorithm == CugraphAlgorithm::KatzCentrality => {
            let epsilon = json_f64(value, key)?;
            // Mirrors the eigenvector-centrality reasoning: libcugraph's
            // convergence test never accepts at epsilon=0 on a non-empty
            // graph, so reject at planning instead of burning GPU work.
            if epsilon <= 0.0 {
                return Err(plan("Katz centrality epsilon must be positive"));
            }
            if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                params.epsilon = epsilon;
            }
        },
        "max_iterations" if algorithm == CugraphAlgorithm::KatzCentrality => {
            let max_iterations = json_u32(value, key)?;
            if max_iterations < 1 {
                return Err(plan("Katz centrality max_iterations must be at least 1"));
            }
            if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                params.max_iterations = max_iterations;
            }
        },
        "epsilon" if algorithm == CugraphAlgorithm::Hits => {
            let epsilon = json_f64(value, key)?;
            // Same reasoning as the other centrality solvers:
            // libcugraph's HITS convergence test never accepts at
            // epsilon=0 on a non-empty graph, so reject at planning.
            if epsilon <= 0.0 {
                return Err(plan("HITS epsilon must be positive"));
            }
            if let CugraphAlgorithmParams::Hits(params) = params {
                params.epsilon = epsilon;
            }
        },
        "max_iterations" if algorithm == CugraphAlgorithm::Hits => {
            let max_iterations = json_u32(value, key)?;
            if max_iterations < 1 {
                return Err(plan("HITS max_iterations must be at least 1"));
            }
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
                _ => {
                    let k = json_u64(value, key)?;
                    if k == 0 {
                        return Err(plan("betweenness k must be at least 1"));
                    }
                    Some(k)
                },
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
                    _ => {
                        let cutoff = json_f64(value, key)?;
                        if cutoff < 0.0 {
                            return Err(plan("SSSP cutoff must be non-negative"));
                        }
                        Some(cutoff)
                    },
                };
            }
        },
        "degree_type" if algorithm == CugraphAlgorithm::CoreNumber => {
            let degree_type = parse_degree_type(json_string(value, key)?.as_str(), key)?;
            if let CugraphAlgorithmParams::CoreNumber(params) = params {
                params.degree_type = degree_type;
            }
        },
        "k" if algorithm == CugraphAlgorithm::KCore => {
            let k = json_u32(value, key)?;
            if k < 1 {
                return Err(plan("k_core k must be at least 1"));
            }
            if let CugraphAlgorithmParams::KCore(params) = params {
                params.k = k;
            }
        },
        "degree_type" if algorithm == CugraphAlgorithm::KCore => {
            let degree_type = parse_degree_type(json_string(value, key)?.as_str(), key)?;
            if let CugraphAlgorithmParams::KCore(params) = params {
                params.degree_type = degree_type;
            }
        },
        // Leiden options.
        "max_level" if algorithm == CugraphAlgorithm::Leiden => {
            let max_level = json_u32(value, key)?;
            if max_level < 1 {
                return Err(plan("Leiden max_level must be at least 1"));
            }
            if let CugraphAlgorithmParams::Leiden(params) = params {
                params.max_level = max_level;
            }
        },
        "resolution" if algorithm == CugraphAlgorithm::Leiden || algorithm == CugraphAlgorithm::Ecg => {
            let resolution = json_f64(value, key)?;
            if resolution <= 0.0 {
                return Err(plan("resolution must be positive"));
            }
            match params {
                CugraphAlgorithmParams::Leiden(params) => params.resolution = resolution,
                CugraphAlgorithmParams::Ecg(params) => params.resolution = resolution,
                _ => {},
            }
        },
        "theta" if algorithm == CugraphAlgorithm::Leiden => {
            let theta = json_f64(value, key)?;
            if theta <= 0.0 {
                return Err(plan("Leiden theta must be positive"));
            }
            if let CugraphAlgorithmParams::Leiden(params) = params {
                params.theta = theta;
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
        // ECG options.
        "min_weight" if algorithm == CugraphAlgorithm::Ecg => {
            let min_weight = json_f64(value, key)?;
            if min_weight < 0.0 {
                return Err(plan("ECG min_weight must be non-negative"));
            }
            if let CugraphAlgorithmParams::Ecg(params) = params {
                params.min_weight = min_weight;
            }
        },
        "ensemble_size" if algorithm == CugraphAlgorithm::Ecg => {
            let ensemble_size = json_u32(value, key)?;
            if ensemble_size < 1 {
                return Err(plan("ECG ensemble_size must be at least 1"));
            }
            if let CugraphAlgorithmParams::Ecg(params) = params {
                params.ensemble_size = ensemble_size;
            }
        },
        "max_level" if algorithm == CugraphAlgorithm::Ecg => {
            let max_level = json_u32(value, key)?;
            if max_level < 1 {
                return Err(plan("ECG max_level must be at least 1"));
            }
            if let CugraphAlgorithmParams::Ecg(params) = params {
                params.max_level = max_level;
            }
        },
        "threshold" if algorithm == CugraphAlgorithm::Ecg => {
            let threshold = json_f64(value, key)?;
            if threshold < 0.0 {
                return Err(plan("ECG threshold must be non-negative"));
            }
            if let CugraphAlgorithmParams::Ecg(params) = params {
                params.threshold = threshold;
            }
        },
        // SpectralModularityMaximization options.
        "n_clusters" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let n_clusters = json_u32(value, key)?;
            if n_clusters < 1 {
                return Err(plan("n_clusters must be at least 1"));
            }
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.n_clusters = n_clusters;
            }
        },
        "n_eigenvectors" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let n_eigenvectors = json_u32(value, key)?;
            if n_eigenvectors < 1 {
                return Err(plan("n_eigenvectors must be at least 1"));
            }
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.n_eigenvectors = n_eigenvectors;
            }
        },
        "evs_tolerance" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let evs_tolerance = json_f64(value, key)?;
            if evs_tolerance < 0.0 {
                return Err(plan("evs_tolerance must be non-negative"));
            }
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.evs_tolerance = evs_tolerance;
            }
        },
        "evs_max_iterations" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let evs_max_iterations = json_u32(value, key)?;
            if evs_max_iterations < 1 {
                return Err(plan("evs_max_iterations must be at least 1"));
            }
            if evs_max_iterations > i32::MAX as u32 {
                return Err(plan("evs_max_iterations must be at most i32::MAX"));
            }
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.evs_max_iterations = evs_max_iterations;
            }
        },
        "k_means_tolerance" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let k_means_tolerance = json_f64(value, key)?;
            if k_means_tolerance < 0.0 {
                return Err(plan("k_means_tolerance must be non-negative"));
            }
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.k_means_tolerance = k_means_tolerance;
            }
        },
        "k_means_max_iterations" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
            let k_means_max_iterations = json_u32(value, key)?;
            if k_means_max_iterations < 1 {
                return Err(plan("k_means_max_iterations must be at least 1"));
            }
            if k_means_max_iterations > i32::MAX as u32 {
                return Err(plan("k_means_max_iterations must be at most i32::MAX"));
            }
            if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                params.k_means_max_iterations = k_means_max_iterations;
            }
        },
        "max_iter" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let max_iter = json_u32(value, key)?;
            if max_iter < 1 {
                return Err(plan("ForceAtlas2 max_iter must be at least 1"));
            }
            if max_iter > i32::MAX as u32 {
                return Err(plan("ForceAtlas2 max_iter must be at most i32::MAX"));
            }
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
            if overlap_scaling_ratio <= 0.0 {
                return Err(plan("ForceAtlas2 overlap_scaling_ratio must be positive"));
            }
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.overlap_scaling_ratio = overlap_scaling_ratio;
            }
        },
        "edge_weight_influence" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let edge_weight_influence = json_f64(value, key)?;
            if edge_weight_influence < 0.0 {
                return Err(plan("ForceAtlas2 edge_weight_influence must be non-negative"));
            }
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.edge_weight_influence = edge_weight_influence;
            }
        },
        "jitter_tolerance" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let jitter_tolerance = json_f64(value, key)?;
            if jitter_tolerance < 0.0 {
                return Err(plan("ForceAtlas2 jitter_tolerance must be non-negative"));
            }
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
            if !(0.0..=1.0).contains(&barnes_hut_theta) {
                return Err(plan("ForceAtlas2 barnes_hut_theta must be in [0, 1]"));
            }
            if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                params.barnes_hut_theta = barnes_hut_theta;
            }
        },
        "scaling_ratio" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
            let scaling_ratio = json_f64(value, key)?;
            if scaling_ratio <= 0.0 {
                return Err(plan("ForceAtlas2 scaling_ratio must be positive"));
            }
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
            if gravity < 0.0 {
                return Err(plan("ForceAtlas2 gravity must be non-negative"));
            }
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
            return Err(unsupported_option(format!("unknown option '{key}' for {}", algorithm.function_name())));
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
    let values =
        value.as_array().ok_or_else(|| plan(format!("cuGraph option '{key}' must be an integer or string array")))?;
    if values.is_empty() {
        return Err(plan("betweenness seeds must not be empty"));
    }
    let vertices = values
        .iter()
        .map(|value| {
            if let Some(value) = value.as_i64() {
                Ok(CugraphVertexSelector::Int64(value))
            } else if let Some(value) = value.as_str() {
                Ok(CugraphVertexSelector::Utf8(value.to_string()))
            } else {
                Err(plan(format!("cuGraph option '{key}' entries must be integers or strings")))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let string_domain = vertices.first().is_some_and(CugraphVertexSelector::is_utf8);
    if vertices.iter().any(|vertex| vertex.is_utf8() != string_domain) {
        return Err(plan("betweenness seeds must not mix integer and string vertices"));
    }
    Ok(Some(CugraphBetweennessSeeds { vertices }))
}

fn validate_betweenness_options(params: &CugraphAlgorithmParams) -> Result<()> {
    match params {
        CugraphAlgorithmParams::BetweennessCentrality(params) if params.seeds.is_some() && params.k.is_some() => {
            return Err(plan("betweenness seeds and k are mutually exclusive"));
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) if params.seeds.is_some() && params.k.is_some() => {
            return Err(plan("betweenness seeds and k are mutually exclusive"));
        },
        _ => {},
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
        other => Err(plan(format!("unsupported '{key}' value '{other}', expected 'in', 'out', or 'in_out'"))),
    }
}

fn json_bool(value: &Value, key: &str) -> Result<bool> {
    value.as_bool().ok_or_else(|| plan(format!("cuGraph option '{key}' must be a boolean")))
}

fn json_string(value: &Value, key: &str) -> Result<String> {
    value.as_str().map(ToOwned::to_owned).ok_or_else(|| plan(format!("cuGraph option '{key}' must be a string")))
}

fn json_f64(value: &Value, key: &str) -> Result<f64> {
    let parsed = value.as_f64().ok_or_else(|| plan(format!("cuGraph option '{key}' must be a number")))?;
    if parsed.is_finite() { Ok(parsed) } else { Err(plan(format!("cuGraph option '{key}' must be finite"))) }
}

fn json_i64(value: &Value, key: &str) -> Result<i64> {
    value.as_i64().ok_or_else(|| plan(format!("cuGraph option '{key}' must be an integer")))
}

fn json_u32(value: &Value, key: &str) -> Result<u32> {
    let parsed = value.as_u64().ok_or_else(|| plan(format!("cuGraph option '{key}' must be an unsigned integer")))?;
    u32::try_from(parsed).map_err(|_| plan(format!("cuGraph option '{key}' exceeds u32::MAX")))
}

fn json_u64(value: &Value, key: &str) -> Result<u64> {
    value.as_u64().ok_or_else(|| plan(format!("cuGraph option '{key}' must be an unsigned 64-bit integer")))
}
