use arrow::array::{Array, AsArray};
use datafusion::{
    common::{Result, ScalarValue},
    logical_expr::Expr,
};
use serde_json::Value;

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBetweennessSeeds, CugraphBfsOutputMode,
    CugraphBfsOutputOptions, CugraphBfsPredicateInputs, CugraphBfsSource, CugraphBfsSourceVertex,
    CugraphGraphConstructionPolicy, CugraphGraphEdgePredicateTable, CugraphGraphOptions, CugraphGraphPredicateTable,
    CugraphKCoreDegreeType, CugraphPageRankParams, CugraphSqlConfig, CugraphTableReference, literal_string_arg,
    plan_error, result_schema_for_params,
};

pub fn parse_cugraph_algorithm_spec(
    algorithm: CugraphAlgorithm,
    args: &[Expr],
    config: &CugraphSqlConfig,
) -> Result<CugraphAlgorithmSpec> {
    validate_arity(algorithm, args.len())?;
    let table_name = CugraphTableReference::parse(&literal_string_arg(args, 0, "table_name")?)?;

    let mut next = 1;
    let source = if algorithm.requires_source_vertex() {
        let value = literal_source_arg(args, next, algorithm)?;
        next += 1;
        value
    } else {
        None
    };

    let (src_col, dst_col) = if args.len() >= next + 2 {
        let src = literal_string_arg(args, next, "src_col")?;
        let dst = literal_string_arg(args, next + 1, "dst_col")?;
        next += 2;
        (src, dst)
    } else {
        ("src".to_string(), "dst".to_string())
    };

    let weight_col = if args.len() > next {
        let weight = literal_optional_string_arg(args, next, "weight_col")?;
        next += 1;
        weight
    } else {
        None
    };

    let mut graph_options = config.default_graph_options.clone();
    let mut edge_id_col = None;
    // Several algorithms require (or are only semantically meaningful on)
    // undirected graphs.  Set directed=false before parsing user-supplied
    // options so default calls match the Python cuGraph oracle semantics.
    //
    // - WCC: cuGraph WCC requires a symmetric graph by API contract.
    // - triangle_count_all / core_number / k_core: Python cuGraph enforces
    //   undirected; libcugraph accepts directed but produces different
    //   (non-comparable) results, so the adapter default aligns with the
    //   Python oracle to keep parity meaningful.
    // - minimum_spanning_tree / spectral_modularity_maximization: cuGraph
    //   requires an undirected/symmetric graph for these legacy algorithms.
    graph_options = algorithm.effective_default_graph_options(&graph_options);
    let mut params = default_params(algorithm, source.clone());
    let has_options_json = args.len() > next;
    let mut source_relation = None;
    if args.len() > next {
        let options_json = literal_string_arg(args, next, "options_json")?;
        source_relation =
            parse_options_json(algorithm, &options_json, &mut graph_options, &mut edge_id_col, &mut params)?;
    }
    apply_source_arg(algorithm, source, source_relation, &mut params)?;
    if algorithm == CugraphAlgorithm::PersonalizedPageRank && !has_options_json {
        return Err(plan_error(
            "cugraph_personalized_pagerank requires options: personalization_table, personalization_vertex_col, personalization_value_col",
        ));
    }
    validate_graph_directedness(algorithm, &graph_options)?;

    // MST is a weighted algorithm: require an explicit weight column so
    // the graph is built with edge weights rather than silently producing
    // a meaningless unweighted result.
    if algorithm.native_algorithm().weight_policy().requires_weight() && weight_col.is_none() {
        return Err(plan_error(
            "cugraph_minimum_spanning_tree requires a weight column; supply one as the third argument",
        ));
    }

    let output_schema = result_schema_for_params(algorithm, &params);
    Ok(CugraphAlgorithmSpec {
        algorithm,
        table_name,
        src_col,
        dst_col,
        weight_col,
        edge_id_col,
        graph_options,
        params,
        output_schema,
    })
}

fn validate_arity(algorithm: CugraphAlgorithm, len: usize) -> Result<()> {
    let valid =
        if algorithm.requires_source_vertex() { matches!(len, 2 | 4 | 5 | 6) } else { matches!(len, 1 | 3 | 4 | 5) };
    if valid {
        Ok(())
    } else {
        Err(plan_error(format!(
            "{} received {len} arguments, but expected {}",
            algorithm.function_name(),
            expected_arity(algorithm)
        )))
    }
}

fn expected_arity(algorithm: CugraphAlgorithm) -> &'static str {
    if algorithm == CugraphAlgorithm::Bfs {
        "table_name, source_vertex_or_vertices [, src_col, dst_col, weight_col, options_json]"
    } else if algorithm.requires_source_vertex() {
        "table_name, source_vertex [, src_col, dst_col, weight_col, options_json]"
    } else {
        "table_name [, src_col, dst_col, weight_col, options_json]"
    }
}

fn default_params(algorithm: CugraphAlgorithm, source: Option<CugraphBfsSource>) -> CugraphAlgorithmParams {
    CugraphAlgorithmParams::from_native_defaults(algorithm, source)
}

fn apply_source_arg(
    algorithm: CugraphAlgorithm,
    source: Option<CugraphBfsSource>,
    source_relation: Option<CugraphGraphPredicateTable>,
    params: &mut CugraphAlgorithmParams,
) -> Result<()> {
    match algorithm {
        CugraphAlgorithm::Bfs => {
            let CugraphAlgorithmParams::Bfs(params) = params else {
                return Err(plan_error("cugraph_bfs carried non-BFS parameters"));
            };
            match (source, source_relation) {
                (Some(_), Some(_)) => Err(plan_error(
                    "BFS source_vertex_or_vertices cannot be combined with source_vertices_table/source_vertex_col options",
                )),
                (Some(source), None) => {
                    params.source = source;
                    Ok(())
                },
                (None, Some(source_relation)) => {
                    params.source = CugraphBfsSource::Relation(source_relation);
                    Ok(())
                },
                (None, None) => Err(plan_error(
                    "BFS source_vertex_or_vertices may be NULL only when source_vertices_table and source_vertex_col are provided",
                )),
            }
        },
        _ if algorithm.requires_source_vertex() => {
            if source.is_none() {
                return Err(plan_error(format!("{} requires source_vertex", algorithm.function_name())));
            }
            if source_relation.is_some() {
                return Err(plan_error(format!(
                    "{} does not accept source_vertices_table/source_vertex_col options",
                    algorithm.function_name()
                )));
            }
            Ok(())
        },
        _ => {
            if source_relation.is_some() {
                return Err(plan_error(format!(
                    "{} does not accept source_vertices_table/source_vertex_col options",
                    algorithm.function_name()
                )));
            }
            Ok(())
        },
    }
}

fn validate_graph_directedness(algorithm: CugraphAlgorithm, graph_options: &CugraphGraphOptions) -> Result<()> {
    let Some(required) = algorithm.native_algorithm().definition().required_graph_directedness() else {
        return Ok(());
    };
    if graph_options.directed == required {
        return Ok(());
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
    Err(plan_error(message))
}

fn parse_options_json(
    algorithm: CugraphAlgorithm,
    options_json: &str,
    graph_options: &mut CugraphGraphOptions,
    edge_id_col: &mut Option<String>,
    params: &mut CugraphAlgorithmParams,
) -> Result<Option<CugraphGraphPredicateTable>> {
    let value = serde_json::from_str::<Value>(options_json)
        .map_err(|error| plan_error(format!("invalid cuGraph options JSON: {error}")))?;
    let object = value.as_object().ok_or_else(|| plan_error("cuGraph options JSON must be an object"))?;

    for (key, value) in object {
        match key.as_str() {
            "directed" => graph_options.directed = json_bool(value, key)?,
            "renumber" => graph_options.renumber = json_bool(value, key)?,
            "construction_policy" => {
                let policy = json_string(value, key)?;
                graph_options.construction_policy = match policy.as_str() {
                    "python_cugraph" => CugraphGraphConstructionPolicy::PythonCugraph,
                    "raw_libcugraph" => CugraphGraphConstructionPolicy::RawLibcugraph,
                    other => {
                        return Err(plan_error(format!(
                            "unsupported cuGraph construction_policy '{other}', expected 'python_cugraph' or 'raw_libcugraph'"
                        )));
                    },
                };
            },
            "alpha" if is_pagerank_algorithm(algorithm) => {
                let alpha = json_f64(value, key)?;
                if !(0.0..=1.0).contains(&alpha) {
                    return Err(plan_error("PageRank alpha must be in [0, 1]"));
                }
                if let Some(params) = pagerank_params_mut(params) {
                    params.alpha = alpha;
                }
            },
            "epsilon" if is_pagerank_algorithm(algorithm) => {
                let epsilon = json_f64(value, key)?;
                if epsilon <= 0.0 {
                    return Err(plan_error("PageRank epsilon must be positive"));
                }
                if let Some(params) = pagerank_params_mut(params) {
                    params.epsilon = epsilon;
                }
            },
            "max_iterations" if is_pagerank_algorithm(algorithm) => {
                let max_iterations = json_u32(value, key)?;
                if max_iterations < 1 {
                    return Err(plan_error("PageRank max_iterations must be at least 1"));
                }
                if let Some(params) = pagerank_params_mut(params) {
                    params.max_iterations = max_iterations;
                }
            },
            "personalization_table" if algorithm == CugraphAlgorithm::PersonalizedPageRank => {
                let table = CugraphTableReference::parse(&json_string(value, key)?)?;
                if let CugraphAlgorithmParams::PersonalizedPageRank(params) = params {
                    params.personalization.table_name = table;
                }
            },
            "personalization_vertex_col" if algorithm == CugraphAlgorithm::PersonalizedPageRank => {
                let column = json_string(value, key)?;
                if let CugraphAlgorithmParams::PersonalizedPageRank(params) = params {
                    params.personalization.vertex_col = column;
                }
            },
            "personalization_value_col" if algorithm == CugraphAlgorithm::PersonalizedPageRank => {
                let column = json_string(value, key)?;
                if let CugraphAlgorithmParams::PersonalizedPageRank(params) = params {
                    params.personalization.value_col = column;
                }
            },
            "depth_limit" if algorithm == CugraphAlgorithm::Bfs => {
                if let CugraphAlgorithmParams::Bfs(params) = params {
                    params.depth_limit = match value {
                        Value::Null => None,
                        _ => {
                            let depth_limit = json_i64(value, key)?;
                            if depth_limit < 0 {
                                return Err(plan_error("BFS depth_limit must be non-negative"));
                            }
                            Some(depth_limit)
                        },
                    };
                }
            },
            key if algorithm == CugraphAlgorithm::Bfs && is_bfs_predicate_option(key) => {},
            "max_level" if algorithm == CugraphAlgorithm::Louvain => {
                let max_level = json_u32(value, key)?;
                if max_level < 1 {
                    return Err(plan_error("Louvain max_level must be at least 1"));
                }
                if let CugraphAlgorithmParams::Louvain(params) = params {
                    params.max_level = max_level;
                }
            },
            "threshold" if algorithm == CugraphAlgorithm::Louvain => {
                let threshold = json_f64(value, key)?;
                if threshold < 0.0 {
                    return Err(plan_error("Louvain threshold must be non-negative"));
                }
                if let CugraphAlgorithmParams::Louvain(params) = params {
                    params.threshold = threshold;
                }
            },
            "resolution" if algorithm == CugraphAlgorithm::Louvain => {
                let resolution = json_f64(value, key)?;
                if resolution <= 0.0 {
                    return Err(plan_error("Louvain resolution must be positive"));
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
                    return Err(plan_error("eigenvector centrality epsilon must be positive"));
                }
                if let CugraphAlgorithmParams::EigenvectorCentrality(params) = params {
                    params.epsilon = epsilon;
                }
            },
            "max_iterations" if algorithm == CugraphAlgorithm::EigenvectorCentrality => {
                let max_iterations = json_u32(value, key)?;
                if max_iterations < 1 {
                    return Err(plan_error("eigenvector centrality max_iterations must be at least 1"));
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
                    return Err(plan_error("Katz centrality alpha must be positive"));
                }
                if alpha > 1.0 {
                    return Err(plan_error("Katz centrality alpha must be <= 1"));
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
                // Reject at planning since cugraph-rs's `NonNegativeF64`
                // wrapper accepts zero.
                if beta <= 0.0 {
                    return Err(plan_error("Katz centrality beta must be positive"));
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
                    return Err(plan_error("Katz centrality epsilon must be positive"));
                }
                if let CugraphAlgorithmParams::KatzCentrality(params) = params {
                    params.epsilon = epsilon;
                }
            },
            "max_iterations" if algorithm == CugraphAlgorithm::KatzCentrality => {
                let max_iterations = json_u32(value, key)?;
                if max_iterations < 1 {
                    return Err(plan_error("Katz centrality max_iterations must be at least 1"));
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
                    return Err(plan_error("HITS epsilon must be positive"));
                }
                if let CugraphAlgorithmParams::Hits(params) = params {
                    params.epsilon = epsilon;
                }
            },
            "max_iterations" if algorithm == CugraphAlgorithm::Hits => {
                let max_iterations = json_u32(value, key)?;
                if max_iterations < 1 {
                    return Err(plan_error("HITS max_iterations must be at least 1"));
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
                let k = json_u64(value, key)?;
                if k == 0 {
                    return Err(plan_error("betweenness k must be at least 1"));
                }
                match params {
                    CugraphAlgorithmParams::BetweennessCentrality(params) => params.k = Some(k),
                    CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => params.k = Some(k),
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
                                return Err(plan_error("SSSP cutoff must be non-negative"));
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
                    return Err(plan_error("k_core k must be at least 1"));
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
                    return Err(plan_error("Leiden max_level must be at least 1"));
                }
                if let CugraphAlgorithmParams::Leiden(params) = params {
                    params.max_level = max_level;
                }
            },
            "resolution" if algorithm == CugraphAlgorithm::Leiden || algorithm == CugraphAlgorithm::Ecg => {
                let resolution = json_f64(value, key)?;
                if resolution <= 0.0 {
                    return Err(plan_error("resolution must be positive"));
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
                    return Err(plan_error("Leiden theta must be positive"));
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
                    return Err(plan_error("ECG min_weight must be non-negative"));
                }
                if let CugraphAlgorithmParams::Ecg(params) = params {
                    params.min_weight = min_weight;
                }
            },
            "ensemble_size" if algorithm == CugraphAlgorithm::Ecg => {
                let ensemble_size = json_u32(value, key)?;
                if ensemble_size < 1 {
                    return Err(plan_error("ECG ensemble_size must be at least 1"));
                }
                if let CugraphAlgorithmParams::Ecg(params) = params {
                    params.ensemble_size = ensemble_size;
                }
            },
            "max_level" if algorithm == CugraphAlgorithm::Ecg => {
                let max_level = json_u32(value, key)?;
                if max_level < 1 {
                    return Err(plan_error("ECG max_level must be at least 1"));
                }
                if let CugraphAlgorithmParams::Ecg(params) = params {
                    params.max_level = max_level;
                }
            },
            "threshold" if algorithm == CugraphAlgorithm::Ecg => {
                let threshold = json_f64(value, key)?;
                if threshold < 0.0 {
                    return Err(plan_error("ECG threshold must be non-negative"));
                }
                if let CugraphAlgorithmParams::Ecg(params) = params {
                    params.threshold = threshold;
                }
            },
            // SpectralModularityMaximization options.
            "n_clusters" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
                let n_clusters = json_u32(value, key)?;
                if n_clusters < 1 {
                    return Err(plan_error("n_clusters must be at least 1"));
                }
                if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                    params.n_clusters = n_clusters;
                }
            },
            "n_eigenvectors" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
                let n_eigenvectors = json_u32(value, key)?;
                if n_eigenvectors < 1 {
                    return Err(plan_error("n_eigenvectors must be at least 1"));
                }
                if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                    params.n_eigenvectors = n_eigenvectors;
                }
            },
            "evs_tolerance" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
                let evs_tolerance = json_f64(value, key)?;
                if evs_tolerance < 0.0 {
                    return Err(plan_error("evs_tolerance must be non-negative"));
                }
                if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                    params.evs_tolerance = evs_tolerance;
                }
            },
            "evs_max_iterations" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
                let evs_max_iterations = json_u32(value, key)?;
                if evs_max_iterations < 1 {
                    return Err(plan_error("evs_max_iterations must be at least 1"));
                }
                if evs_max_iterations > i32::MAX as u32 {
                    return Err(plan_error("evs_max_iterations must be at most i32::MAX"));
                }
                if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                    params.evs_max_iterations = evs_max_iterations;
                }
            },
            "k_means_tolerance" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
                let k_means_tolerance = json_f64(value, key)?;
                if k_means_tolerance < 0.0 {
                    return Err(plan_error("k_means_tolerance must be non-negative"));
                }
                if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                    params.k_means_tolerance = k_means_tolerance;
                }
            },
            "k_means_max_iterations" if algorithm == CugraphAlgorithm::SpectralModularityMaximization => {
                let k_means_max_iterations = json_u32(value, key)?;
                if k_means_max_iterations < 1 {
                    return Err(plan_error("k_means_max_iterations must be at least 1"));
                }
                if k_means_max_iterations > i32::MAX as u32 {
                    return Err(plan_error("k_means_max_iterations must be at most i32::MAX"));
                }
                if let CugraphAlgorithmParams::SpectralModularityMaximization(params) = params {
                    params.k_means_max_iterations = k_means_max_iterations;
                }
            },
            "max_iter" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
                let max_iter = json_u32(value, key)?;
                if max_iter < 1 {
                    return Err(plan_error("ForceAtlas2 max_iter must be at least 1"));
                }
                if max_iter > i32::MAX as u32 {
                    return Err(plan_error("ForceAtlas2 max_iter must be at most i32::MAX"));
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
                if value {
                    return Err(plan_error(
                        "ForceAtlas2 prevent_overlapping requires vertex_radius side input, which the SQL API does not support yet",
                    ));
                }
                if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                    params.prevent_overlapping = value;
                }
            },
            "overlap_scaling_ratio" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
                let overlap_scaling_ratio = json_f64(value, key)?;
                if overlap_scaling_ratio <= 0.0 {
                    return Err(plan_error("ForceAtlas2 overlap_scaling_ratio must be positive"));
                }
                if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                    params.overlap_scaling_ratio = overlap_scaling_ratio;
                }
            },
            "edge_weight_influence" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
                let edge_weight_influence = json_f64(value, key)?;
                if edge_weight_influence < 0.0 {
                    return Err(plan_error("ForceAtlas2 edge_weight_influence must be non-negative"));
                }
                if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                    params.edge_weight_influence = edge_weight_influence;
                }
            },
            "jitter_tolerance" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
                let jitter_tolerance = json_f64(value, key)?;
                if jitter_tolerance < 0.0 {
                    return Err(plan_error("ForceAtlas2 jitter_tolerance must be non-negative"));
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
                    return Err(plan_error("ForceAtlas2 barnes_hut_theta must be in [0, 1]"));
                }
                if let CugraphAlgorithmParams::ForceAtlas2(params) = params {
                    params.barnes_hut_theta = barnes_hut_theta;
                }
            },
            "scaling_ratio" if algorithm == CugraphAlgorithm::ForceAtlas2 => {
                let scaling_ratio = json_f64(value, key)?;
                if scaling_ratio <= 0.0 {
                    return Err(plan_error("ForceAtlas2 scaling_ratio must be positive"));
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
                    return Err(plan_error("ForceAtlas2 gravity must be non-negative"));
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
                return Err(plan_error(format!("unknown option '{key}' for {}", algorithm.function_name())));
            },
        }
    }

    let mut source_relation = None;
    if algorithm == CugraphAlgorithm::Bfs {
        source_relation = parse_bfs_predicate_options(object, edge_id_col, params)?;
    }
    if algorithm == CugraphAlgorithm::PersonalizedPageRank {
        validate_personalized_pagerank_options(object, params)?;
    }
    if is_betweenness_algorithm(algorithm) {
        validate_betweenness_options(params)?;
    }

    Ok(source_relation)
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
        value.as_array().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be an integer array")))?;
    if values.is_empty() {
        return Err(plan_error("betweenness seeds must not be empty"));
    }
    let vertices = values.iter().map(|value| json_i64(value, key)).collect::<Result<Vec<_>>>()?;
    Ok(Some(CugraphBetweennessSeeds { vertices }))
}

fn validate_betweenness_options(params: &CugraphAlgorithmParams) -> Result<()> {
    match params {
        CugraphAlgorithmParams::BetweennessCentrality(params) if params.seeds.is_some() && params.k.is_some() => {
            return Err(plan_error("betweenness seeds and k are mutually exclusive"));
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) if params.seeds.is_some() && params.k.is_some() => {
            return Err(plan_error("betweenness seeds and k are mutually exclusive"));
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

fn validate_personalized_pagerank_options(
    object: &serde_json::Map<String, Value>,
    params: &CugraphAlgorithmParams,
) -> Result<()> {
    let missing = ["personalization_table", "personalization_vertex_col", "personalization_value_col"]
        .into_iter()
        .filter(|key| !object.contains_key(*key))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(plan_error(format!("cugraph_personalized_pagerank requires options: {}", missing.join(", "))));
    }
    let CugraphAlgorithmParams::PersonalizedPageRank(params) = params else {
        return Err(plan_error("cugraph_personalized_pagerank carried non-personalized PageRank parameters"));
    };
    if params.personalization.vertex_col.is_empty() {
        return Err(plan_error("personalization_vertex_col must not be empty"));
    }
    if params.personalization.value_col.is_empty() {
        return Err(plan_error("personalization_value_col must not be empty"));
    }
    Ok(())
}

fn is_bfs_predicate_option(key: &str) -> bool {
    matches!(
        key,
        "edge_id_col"
            | "source_vertices_table"
            | "source_vertex_col"
            | "include_vertices_table"
            | "include_vertex_col"
            | "exclude_vertices_table"
            | "exclude_vertex_col"
            | "target_vertices_table"
            | "target_vertex_col"
            | "include_edge_ids_table"
            | "include_edge_ids_col"
            | "include_edges_table"
            | "include_edge_src_col"
            | "include_edge_dst_col"
            | "include_edge_id_col"
            | "output_mode"
            | "return_target_info"
    )
}

fn parse_bfs_predicate_options(
    object: &serde_json::Map<String, Value>,
    edge_id_col: &mut Option<String>,
    params: &mut CugraphAlgorithmParams,
) -> Result<Option<CugraphGraphPredicateTable>> {
    if let Some(value) = optional_json_string(object, "edge_id_col")? {
        *edge_id_col = Some(value);
    }
    let output_mode = parse_bfs_output_mode(object)?;
    let return_target_info = object
        .get("return_target_info")
        .map(|value| json_bool(value, "return_target_info"))
        .transpose()?
        .unwrap_or(false);

    let source_vertices =
        parse_vertex_predicate_option(object, "source_vertices_table", "source_vertex_col", "source_vertices")?;
    let include_vertices =
        parse_vertex_predicate_option(object, "include_vertices_table", "include_vertex_col", "include_vertices")?;
    let exclude_vertices =
        parse_vertex_predicate_option(object, "exclude_vertices_table", "exclude_vertex_col", "exclude_vertices")?;
    let target_vertices =
        parse_vertex_predicate_option(object, "target_vertices_table", "target_vertex_col", "target_vertices")?;
    let include_edge_ids = parse_include_edge_ids_option(object, edge_id_col.as_deref())?;
    let include_edges = parse_include_edges_option(object, edge_id_col.as_deref())?;

    if include_vertices.is_some() && exclude_vertices.is_some() {
        return Err(plan_error("BFS include_vertices and exclude_vertices predicates are mutually exclusive"));
    }
    if return_target_info && target_vertices.is_none() {
        return Err(plan_error("BFS return_target_info requires target_vertices_table and target_vertex_col options"));
    }
    if output_mode == CugraphBfsOutputMode::Path && return_target_info {
        return Err(plan_error("BFS path output cannot combine with return_target_info=true"));
    }
    if output_mode == CugraphBfsOutputMode::Path && target_vertices.is_none() {
        return Err(plan_error("BFS path output requires target_vertices_table and target_vertex_col options"));
    }

    if let CugraphAlgorithmParams::Bfs(params) = params {
        params.predicates = CugraphBfsPredicateInputs {
            include_vertices,
            exclude_vertices,
            target_vertices,
            include_edge_ids,
            include_edges,
        };
        params.output = CugraphBfsOutputOptions { mode: output_mode, return_target_info };
    }
    Ok(source_vertices)
}

fn parse_bfs_output_mode(object: &serde_json::Map<String, Value>) -> Result<CugraphBfsOutputMode> {
    let Some(value) = optional_json_string(object, "output_mode")? else {
        return Ok(CugraphBfsOutputMode::Raw);
    };
    match value.as_str() {
        "raw" => Ok(CugraphBfsOutputMode::Raw),
        "normalized" => Ok(CugraphBfsOutputMode::Normalized),
        "path" => Ok(CugraphBfsOutputMode::Path),
        other => {
            Err(plan_error(format!("unsupported BFS output_mode '{other}', expected 'raw', 'normalized', or 'path'")))
        },
    }
}

fn parse_vertex_predicate_option(
    object: &serde_json::Map<String, Value>,
    table_key: &str,
    column_key: &str,
    role: &str,
) -> Result<Option<CugraphGraphPredicateTable>> {
    let table = optional_table_reference(object, table_key)?;
    let column = optional_json_string(object, column_key)?;
    match (table, column) {
        (Some(table), Some(column)) => Ok(Some(CugraphGraphPredicateTable::new(table, column))),
        (Some(_), None) => Err(plan_error(format!("BFS {role} requires option '{column_key}'"))),
        (None, Some(_)) => Err(plan_error(format!("BFS {role} requires option '{table_key}'"))),
        (None, None) => Ok(None),
    }
}

fn parse_include_edge_ids_option(
    object: &serde_json::Map<String, Value>,
    primary_edge_id_col: Option<&str>,
) -> Result<Option<CugraphGraphPredicateTable>> {
    let table = optional_table_reference(object, "include_edge_ids_table")?;
    let column = optional_json_string(object, "include_edge_ids_col")?;
    match (table, column, primary_edge_id_col) {
        (Some(table), Some(column), Some(_)) => Ok(Some(CugraphGraphPredicateTable::new(table, column))),
        (Some(table), None, Some(primary_edge_id_col)) => {
            Ok(Some(CugraphGraphPredicateTable::new(table, primary_edge_id_col)))
        },
        (Some(_), _, None) => Err(plan_error("BFS include_edge_ids requires primary option 'edge_id_col'")),
        (None, Some(_), _) => Err(plan_error("BFS include_edge_ids requires option 'include_edge_ids_table'")),
        (None, None, _) => Ok(None),
    }
}

fn parse_include_edges_option(
    object: &serde_json::Map<String, Value>,
    primary_edge_id_col: Option<&str>,
) -> Result<Option<CugraphGraphEdgePredicateTable>> {
    let table = optional_table_reference(object, "include_edges_table")?;
    let source = optional_json_string(object, "include_edge_src_col")?;
    let destination = optional_json_string(object, "include_edge_dst_col")?;
    let edge_id = optional_json_string(object, "include_edge_id_col")?;
    if edge_id.is_some() && primary_edge_id_col.is_none() {
        return Err(plan_error("BFS include_edges edge-id alignment requires primary option 'edge_id_col'"));
    }
    match (table, source, destination) {
        (Some(table), Some(source), Some(destination)) => {
            Ok(Some(CugraphGraphEdgePredicateTable::new(table, source, destination, edge_id)))
        },
        (Some(_), source, destination) => {
            let missing = [("include_edge_src_col", source.is_none()), ("include_edge_dst_col", destination.is_none())]
                .into_iter()
                .filter_map(|(key, missing)| missing.then_some(key))
                .collect::<Vec<_>>()
                .join(",");
            Err(plan_error(format!("BFS include_edges requires option(s) '{missing}'")))
        },
        (None, Some(_), _) | (None, _, Some(_)) | (None, _, _) if edge_id.is_some() => {
            Err(plan_error("BFS include_edges column options require option 'include_edges_table'"))
        },
        (None, Some(_), _) | (None, _, Some(_)) => {
            Err(plan_error("BFS include_edges column options require option 'include_edges_table'"))
        },
        (None, None, None) => Ok(None),
    }
}

fn optional_table_reference(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<CugraphTableReference>> {
    optional_json_string(object, key)?.map(|value| CugraphTableReference::parse(&value)).transpose()
}

fn optional_json_string(object: &serde_json::Map<String, Value>, key: &str) -> Result<Option<String>> {
    match object.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => json_string(value, key).map(Some),
    }
}

fn parse_degree_type(value: &str, key: &str) -> Result<CugraphKCoreDegreeType> {
    match value {
        "in" => Ok(CugraphKCoreDegreeType::In),
        "out" => Ok(CugraphKCoreDegreeType::Out),
        "in_out" => Ok(CugraphKCoreDegreeType::InOut),
        other => Err(plan_error(format!("unsupported '{key}' value '{other}', expected 'in', 'out', or 'in_out'"))),
    }
}

fn literal_optional_string_arg(args: &[Expr], index: usize, name: &str) -> Result<Option<String>> {
    match args.get(index) {
        Some(Expr::Literal(ScalarValue::Null, _)) => Ok(None),
        Some(Expr::Literal(ScalarValue::Utf8(None), _))
        | Some(Expr::Literal(ScalarValue::LargeUtf8(None), _))
        | Some(Expr::Literal(ScalarValue::Utf8View(None), _)) => Ok(None),
        Some(_) => literal_string_arg(args, index, name).map(Some),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}

fn literal_source_arg(args: &[Expr], index: usize, algorithm: CugraphAlgorithm) -> Result<Option<CugraphBfsSource>> {
    match args.get(index) {
        Some(Expr::Literal(ScalarValue::Null, _)) if algorithm == CugraphAlgorithm::Bfs => Ok(None),
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::List(values) if algorithm == CugraphAlgorithm::Bfs => {
                Ok(Some(CugraphBfsSource::LiteralList(source_list_values(values.as_ref())?)))
            },
            ScalarValue::LargeList(values) if algorithm == CugraphAlgorithm::Bfs => {
                Ok(Some(CugraphBfsSource::LiteralList(source_list_values(values.as_ref())?)))
            },
            ScalarValue::List(_) | ScalarValue::LargeList(_) => {
                Err(plan_error("source_vertex_or_vertices list literal is supported only by cugraph_bfs"))
            },
            _ => literal_source_vertex_value(value, algorithm).map(|value| value.map(CugraphBfsSource::Scalar)),
        },
        Some(Expr::ScalarFunction(function))
            if algorithm == CugraphAlgorithm::Bfs && function.name().eq_ignore_ascii_case("make_array") =>
        {
            Ok(Some(CugraphBfsSource::LiteralList(source_make_array_values(&function.args, algorithm)?)))
        },
        Some(_) if algorithm == CugraphAlgorithm::Bfs => {
            Err(plan_error("source_vertex_or_vertices must be a literal or list literal"))
        },
        Some(_) => Err(plan_error("source_vertex must be a literal")),
        None if algorithm == CugraphAlgorithm::Bfs => Err(plan_error("missing source_vertex_or_vertices argument")),
        None => Err(plan_error("missing source_vertex argument")),
    }
}

fn source_make_array_values(args: &[Expr], algorithm: CugraphAlgorithm) -> Result<Vec<CugraphBfsSourceVertex>> {
    if args.is_empty() {
        return Err(plan_error("source_vertex_or_vertices list literal must not be empty"));
    }
    let mut values = Vec::with_capacity(args.len());
    for arg in args {
        let Expr::Literal(value, _) = arg else {
            return Err(plan_error("source_vertex_or_vertices list literal elements must be literals"));
        };
        let Some(value) = literal_source_vertex_value(value, algorithm)? else {
            return Err(plan_error("source_vertex_or_vertices list literal must not contain null elements"));
        };
        values.push(value);
    }
    validate_source_list_values(values)
}

fn source_list_values(array: &dyn Array) -> Result<Vec<CugraphBfsSourceVertex>> {
    if array.len() != 1 {
        return Err(plan_error("source_vertex_or_vertices list literal must be a single list value"));
    }
    if array.is_null(0) {
        return Err(plan_error("source_vertex_or_vertices list literal must not be null"));
    }
    let values = match array.data_type() {
        arrow::datatypes::DataType::List(_) => array.as_list::<i32>().value(0),
        arrow::datatypes::DataType::LargeList(_) => array.as_list::<i64>().value(0),
        other => {
            return Err(plan_error(format!(
                "source_vertex_or_vertices list literal has unsupported array type {other:?}"
            )));
        },
    };
    if values.is_empty() {
        return Err(plan_error("source_vertex_or_vertices list literal must not be empty"));
    }
    let mut parsed = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            return Err(plan_error("source_vertex_or_vertices list literal must not contain null elements"));
        }
        let value = ScalarValue::try_from_array(values.as_ref(), row).map_err(|error| {
            plan_error(format!("failed to read source_vertex_or_vertices list literal element {row}: {error}"))
        })?;
        let Some(value) = literal_source_vertex_value(&value, CugraphAlgorithm::Bfs)? else {
            return Err(plan_error("source_vertex_or_vertices list literal must not contain null elements"));
        };
        parsed.push(value);
    }
    validate_source_list_values(parsed)
}

fn validate_source_list_values(values: Vec<CugraphBfsSourceVertex>) -> Result<Vec<CugraphBfsSourceVertex>> {
    if values.is_empty() {
        return Err(plan_error("source_vertex_or_vertices list literal must not be empty"));
    }
    let first_is_utf8 = values.first().is_some_and(CugraphBfsSourceVertex::is_utf8);
    if values.iter().any(|value| value.is_utf8() != first_is_utf8) {
        return Err(plan_error("source_vertex_or_vertices list literal must not mix integer and string elements"));
    }
    Ok(values)
}

fn literal_source_vertex_value(
    value: &ScalarValue,
    algorithm: CugraphAlgorithm,
) -> Result<Option<CugraphBfsSourceVertex>> {
    match value {
        ScalarValue::Null => Ok(None),
        ScalarValue::Int8(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::Int16(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::Int32(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::Int64(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(*value))),
        ScalarValue::UInt8(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::UInt16(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::UInt32(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::UInt64(Some(value)) => i64::try_from(*value)
            .map(|value| Some(CugraphBfsSourceVertex::Int64(value)))
            .map_err(|_| plan_error("source_vertex exceeds i64::MAX")),
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value))
            if algorithm == CugraphAlgorithm::Bfs =>
        {
            Ok(Some(CugraphBfsSourceVertex::Utf8(value.clone())))
        },
        ScalarValue::Utf8(Some(_)) | ScalarValue::LargeUtf8(Some(_)) | ScalarValue::Utf8View(Some(_)) => {
            Err(plan_error("source_vertex must be an integer literal for this cuGraph algorithm"))
        },
        ScalarValue::Int8(None)
        | ScalarValue::Int16(None)
        | ScalarValue::Int32(None)
        | ScalarValue::Int64(None)
        | ScalarValue::UInt8(None)
        | ScalarValue::UInt16(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::Utf8(None)
        | ScalarValue::LargeUtf8(None)
        | ScalarValue::Utf8View(None) => Ok(None),
        _ => Err(plan_error("source_vertex must be a non-null integer literal")),
    }
}

fn json_bool(value: &Value, key: &str) -> Result<bool> {
    value.as_bool().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a boolean")))
}

fn json_string(value: &Value, key: &str) -> Result<String> {
    value.as_str().map(ToOwned::to_owned).ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a string")))
}

fn json_f64(value: &Value, key: &str) -> Result<f64> {
    let parsed = value.as_f64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a number")))?;
    if parsed.is_finite() { Ok(parsed) } else { Err(plan_error(format!("cuGraph option '{key}' must be finite"))) }
}

fn json_i64(value: &Value, key: &str) -> Result<i64> {
    value.as_i64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be an integer")))
}

fn json_u32(value: &Value, key: &str) -> Result<u32> {
    let parsed =
        value.as_u64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be an unsigned integer")))?;
    u32::try_from(parsed).map_err(|_| plan_error(format!("cuGraph option '{key}' exceeds u32::MAX")))
}

fn json_u64(value: &Value, key: &str) -> Result<u64> {
    value.as_u64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be an unsigned 64-bit integer")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_arg(value: &str) -> Expr {
        Expr::Literal(ScalarValue::Utf8(Some(value.to_string())), None)
    }

    #[test]
    fn parses_betweenness_explicit_seed_options() {
        let spec = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::BetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"seeds":[0,2,4],"normalized":false,"include_endpoints":true,"exact_vertex_threshold":7}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap();

        let CugraphAlgorithmParams::BetweennessCentrality(params) = spec.params else {
            panic!("expected betweenness params");
        };
        assert_eq!(params.seeds.expect("seeds").vertices, vec![0, 2, 4]);
        assert!(!params.normalized);
        assert!(params.include_endpoints);
        assert_eq!(params.exact_vertex_threshold, 7);
    }

    #[test]
    fn parses_betweenness_k_option() {
        let spec = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::EdgeBetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"k":10,"normalized":false}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap();

        let CugraphAlgorithmParams::EdgeBetweennessCentrality(params) = spec.params else {
            panic!("expected edge betweenness params");
        };
        assert_eq!(params.k, Some(10));
        assert!(params.seeds.is_none());
        assert!(!params.normalized);
    }

    #[test]
    fn rejects_betweenness_seeds_with_k() {
        let error = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::BetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"seeds":[0],"k":1}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("betweenness seeds and k are mutually exclusive"));
    }

    #[test]
    fn parses_betweenness_null_seeds_as_exact_all_vertices() {
        let spec = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::BetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"seeds":null,"normalized":true}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap();

        let CugraphAlgorithmParams::BetweennessCentrality(params) = spec.params else {
            panic!("expected betweenness params");
        };
        assert!(params.seeds.is_none());
        assert!(params.normalized);
    }

    #[test]
    fn rejects_empty_betweenness_seed_list() {
        let error = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::EdgeBetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"seeds":[]}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("betweenness seeds must not be empty"));
    }
}
