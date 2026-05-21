use datafusion::{
    common::{Result, ScalarValue},
    logical_expr::Expr,
};
use serde_json::Value;

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsParams, CugraphCoreNumberParams,
    CugraphDegreesAllParams, CugraphEcgParams, CugraphEigenvectorCentralityParams, CugraphGraphConstructionPolicy,
    CugraphGraphOptions, CugraphHitsParams, CugraphInDegreesAllParams, CugraphKCoreDegreeType, CugraphKCoreParams,
    CugraphKatzCentralityParams, CugraphLeidenParams, CugraphLouvainParams, CugraphMstParams,
    CugraphOutDegreesAllParams, CugraphPageRankParams, CugraphSccParams, CugraphSpectralClusteringParams,
    CugraphSqlConfig, CugraphSsspParams, CugraphTableReference, CugraphTriangleCountAllParams, CugraphWccParams,
    plan_error, stable_result_schema,
};

pub fn parse_cugraph_algorithm_spec(
    algorithm: CugraphAlgorithm,
    args: &[Expr],
    config: &CugraphSqlConfig,
) -> Result<CugraphAlgorithmSpec> {
    if !config.allows(algorithm) {
        return Err(plan_error(format!("cuGraph algorithm {:?} is not enabled in this session", algorithm)));
    }

    validate_arity(algorithm, args.len())?;
    let table_name = CugraphTableReference::parse(&literal_string_arg(args, 0, "table_name")?)?;

    let mut next = 1;
    let source_vertex = if algorithm == CugraphAlgorithm::Bfs || algorithm == CugraphAlgorithm::Sssp {
        let value = literal_i64_arg(args, next, "source_vertex")?;
        next += 1;
        Some(value)
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
    // Several algorithms require (or are only semantically meaningful on)
    // undirected graphs.  Set directed=false before parsing user-supplied
    // options so default calls match the Python cuGraph oracle semantics.
    //
    // - WCC: cuGraph WCC requires a symmetric graph by API contract.
    // - triangle_count_all / core_number / k_core: Python cuGraph enforces
    //   undirected; libcugraph accepts directed but produces different
    //   (non-comparable) results, so the adapter default aligns with the
    //   Python oracle to keep parity meaningful.
    if matches!(
        algorithm,
        CugraphAlgorithm::WeaklyConnectedComponents
            | CugraphAlgorithm::TriangleCountAll
            | CugraphAlgorithm::CoreNumber
            | CugraphAlgorithm::KCore
    ) {
        graph_options.directed = false;
    }
    let mut params = default_params(algorithm, source_vertex);
    if args.len() > next {
        let options_json = literal_string_arg(args, next, "options_json")?;
        parse_options_json(algorithm, &options_json, &mut graph_options, &mut params)?;
    }

    // MST is a weighted algorithm: require an explicit weight column so
    // the graph is built with edge weights rather than silently producing
    // a meaningless unweighted result.
    if algorithm == CugraphAlgorithm::MinimumSpanningTree && weight_col.is_none() {
        return Err(plan_error(
            "cugraph_minimum_spanning_tree requires a weight column; supply one as the third argument",
        ));
    }

    Ok(CugraphAlgorithmSpec {
        algorithm,
        table_name,
        src_col,
        dst_col,
        weight_col,
        graph_options,
        params,
        output_schema: stable_result_schema(algorithm),
    })
}

fn validate_arity(algorithm: CugraphAlgorithm, len: usize) -> Result<()> {
    let valid = match algorithm {
        CugraphAlgorithm::PageRank
        | CugraphAlgorithm::Louvain
        | CugraphAlgorithm::EigenvectorCentrality
        | CugraphAlgorithm::KatzCentrality
        | CugraphAlgorithm::Hits
        | CugraphAlgorithm::DegreesAll
        | CugraphAlgorithm::InDegreesAll
        | CugraphAlgorithm::OutDegreesAll
        | CugraphAlgorithm::WeaklyConnectedComponents
        | CugraphAlgorithm::StronglyConnectedComponents
        | CugraphAlgorithm::TriangleCountAll
        | CugraphAlgorithm::CoreNumber
        | CugraphAlgorithm::KCore
        | CugraphAlgorithm::MinimumSpanningTree
        | CugraphAlgorithm::PageRankStrict
        | CugraphAlgorithm::Leiden
        | CugraphAlgorithm::Ecg
        | CugraphAlgorithm::SpectralModularityMaximization
        | CugraphAlgorithm::BalancedCutClustering => matches!(len, 1 | 3 | 4 | 5),
        CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp => matches!(len, 2 | 4 | 5 | 6),
    };
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
    match algorithm {
        CugraphAlgorithm::PageRank
        | CugraphAlgorithm::Louvain
        | CugraphAlgorithm::EigenvectorCentrality
        | CugraphAlgorithm::KatzCentrality
        | CugraphAlgorithm::Hits
        | CugraphAlgorithm::DegreesAll
        | CugraphAlgorithm::InDegreesAll
        | CugraphAlgorithm::OutDegreesAll
        | CugraphAlgorithm::WeaklyConnectedComponents
        | CugraphAlgorithm::StronglyConnectedComponents
        | CugraphAlgorithm::TriangleCountAll
        | CugraphAlgorithm::CoreNumber
        | CugraphAlgorithm::KCore
        | CugraphAlgorithm::MinimumSpanningTree
        | CugraphAlgorithm::PageRankStrict
        | CugraphAlgorithm::Leiden
        | CugraphAlgorithm::Ecg
        | CugraphAlgorithm::SpectralModularityMaximization
        | CugraphAlgorithm::BalancedCutClustering => "table_name [, src_col, dst_col, weight_col, options_json]",
        CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp => {
            "table_name, source_vertex [, src_col, dst_col, weight_col, options_json]"
        },
    }
}

fn default_params(algorithm: CugraphAlgorithm, source_vertex: Option<i64>) -> CugraphAlgorithmParams {
    match algorithm {
        CugraphAlgorithm::PageRank => CugraphAlgorithmParams::PageRank(CugraphPageRankParams::default()),
        CugraphAlgorithm::Bfs => CugraphAlgorithmParams::Bfs(CugraphBfsParams {
            source_vertex: source_vertex.expect("BFS arity validation requires source_vertex"),
            depth_limit: None,
        }),
        CugraphAlgorithm::Louvain => CugraphAlgorithmParams::Louvain(CugraphLouvainParams::default()),
        CugraphAlgorithm::EigenvectorCentrality => {
            CugraphAlgorithmParams::EigenvectorCentrality(CugraphEigenvectorCentralityParams::default())
        },
        CugraphAlgorithm::KatzCentrality => {
            CugraphAlgorithmParams::KatzCentrality(CugraphKatzCentralityParams::default())
        },
        CugraphAlgorithm::Hits => CugraphAlgorithmParams::Hits(CugraphHitsParams::default()),
        CugraphAlgorithm::Sssp => CugraphAlgorithmParams::Sssp(CugraphSsspParams {
            source_vertex: source_vertex.expect("SSSP arity validation requires source_vertex"),
            cutoff: None,
        }),
        CugraphAlgorithm::DegreesAll => CugraphAlgorithmParams::DegreesAll(CugraphDegreesAllParams),
        CugraphAlgorithm::InDegreesAll => CugraphAlgorithmParams::InDegreesAll(CugraphInDegreesAllParams),
        CugraphAlgorithm::OutDegreesAll => CugraphAlgorithmParams::OutDegreesAll(CugraphOutDegreesAllParams),
        CugraphAlgorithm::WeaklyConnectedComponents => {
            CugraphAlgorithmParams::WeaklyConnectedComponents(CugraphWccParams)
        },
        CugraphAlgorithm::StronglyConnectedComponents => {
            CugraphAlgorithmParams::StronglyConnectedComponents(CugraphSccParams)
        },
        CugraphAlgorithm::TriangleCountAll => CugraphAlgorithmParams::TriangleCountAll(CugraphTriangleCountAllParams),
        CugraphAlgorithm::CoreNumber => CugraphAlgorithmParams::CoreNumber(CugraphCoreNumberParams::default()),
        CugraphAlgorithm::KCore => CugraphAlgorithmParams::KCore(CugraphKCoreParams::default()),
        CugraphAlgorithm::MinimumSpanningTree => CugraphAlgorithmParams::MinimumSpanningTree(CugraphMstParams),
        CugraphAlgorithm::PageRankStrict => CugraphAlgorithmParams::PageRankStrict(CugraphPageRankParams::default()),
        CugraphAlgorithm::Leiden => CugraphAlgorithmParams::Leiden(CugraphLeidenParams::default()),
        CugraphAlgorithm::Ecg => CugraphAlgorithmParams::Ecg(CugraphEcgParams::default()),
        CugraphAlgorithm::SpectralModularityMaximization => {
            CugraphAlgorithmParams::SpectralModularityMaximization(CugraphSpectralClusteringParams::default())
        },
        CugraphAlgorithm::BalancedCutClustering => {
            CugraphAlgorithmParams::BalancedCutClustering(CugraphSpectralClusteringParams::default())
        },
    }
}

fn parse_options_json(
    algorithm: CugraphAlgorithm,
    options_json: &str,
    graph_options: &mut CugraphGraphOptions,
    params: &mut CugraphAlgorithmParams,
) -> Result<()> {
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
            "alpha" if algorithm == CugraphAlgorithm::PageRank => {
                let alpha = json_f64(value, key)?;
                if !(0.0..=1.0).contains(&alpha) {
                    return Err(plan_error("PageRank alpha must be in [0, 1]"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.alpha = alpha;
                }
            },
            "epsilon" if algorithm == CugraphAlgorithm::PageRank => {
                let epsilon = json_f64(value, key)?;
                if epsilon <= 0.0 {
                    return Err(plan_error("PageRank epsilon must be positive"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.epsilon = epsilon;
                }
            },
            "max_iterations" if algorithm == CugraphAlgorithm::PageRank => {
                let max_iterations = json_u32(value, key)?;
                if max_iterations < 1 {
                    return Err(plan_error("PageRank max_iterations must be at least 1"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.max_iterations = max_iterations;
                }
            },
            "depth_limit" if algorithm == CugraphAlgorithm::Bfs => {
                let depth_limit = json_i64(value, key)?;
                if depth_limit < 0 {
                    return Err(plan_error("BFS depth_limit must be non-negative"));
                }
                if let CugraphAlgorithmParams::Bfs(params) = params {
                    params.depth_limit = Some(depth_limit);
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
            "cutoff" if algorithm == CugraphAlgorithm::Sssp => {
                let cutoff = json_f64(value, key)?;
                if cutoff < 0.0 {
                    return Err(plan_error("SSSP cutoff must be non-negative"));
                }
                if let CugraphAlgorithmParams::Sssp(params) = params {
                    params.cutoff = Some(cutoff);
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
            // PageRank strict shares the same options as PageRank (alpha, epsilon, max_iterations).
            "alpha" if algorithm == CugraphAlgorithm::PageRankStrict => {
                let alpha = json_f64(value, key)?;
                if !(0.0..=1.0).contains(&alpha) {
                    return Err(plan_error("PageRank strict alpha must be in [0, 1]"));
                }
                if let CugraphAlgorithmParams::PageRankStrict(params) = params {
                    params.alpha = alpha;
                }
            },
            "epsilon" if algorithm == CugraphAlgorithm::PageRankStrict => {
                let epsilon = json_f64(value, key)?;
                if epsilon <= 0.0 {
                    return Err(plan_error("PageRank strict epsilon must be positive"));
                }
                if let CugraphAlgorithmParams::PageRankStrict(params) = params {
                    params.epsilon = epsilon;
                }
            },
            "max_iterations" if algorithm == CugraphAlgorithm::PageRankStrict => {
                let max_iterations = json_u32(value, key)?;
                if max_iterations < 1 {
                    return Err(plan_error("PageRank strict max_iterations must be at least 1"));
                }
                if let CugraphAlgorithmParams::PageRankStrict(params) = params {
                    params.max_iterations = max_iterations;
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
                    || algorithm == CugraphAlgorithm::BalancedCutClustering =>
            {
                let seed = json_u64(value, key)?;
                match params {
                    CugraphAlgorithmParams::Leiden(params) => params.seed = seed,
                    CugraphAlgorithmParams::Ecg(params) => params.seed = seed,
                    CugraphAlgorithmParams::SpectralModularityMaximization(params) => params.seed = seed,
                    CugraphAlgorithmParams::BalancedCutClustering(params) => params.seed = seed,
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
            // SpectralModularityMaximization and BalancedCutClustering share the same options.
            "n_clusters"
                if algorithm == CugraphAlgorithm::SpectralModularityMaximization
                    || algorithm == CugraphAlgorithm::BalancedCutClustering =>
            {
                let n_clusters = json_u32(value, key)?;
                if n_clusters < 1 {
                    return Err(plan_error("n_clusters must be at least 1"));
                }
                match params {
                    CugraphAlgorithmParams::SpectralModularityMaximization(params) => params.n_clusters = n_clusters,
                    CugraphAlgorithmParams::BalancedCutClustering(params) => params.n_clusters = n_clusters,
                    _ => {},
                }
            },
            "n_eigenvectors"
                if algorithm == CugraphAlgorithm::SpectralModularityMaximization
                    || algorithm == CugraphAlgorithm::BalancedCutClustering =>
            {
                let n_eigenvectors = json_u32(value, key)?;
                if n_eigenvectors < 1 {
                    return Err(plan_error("n_eigenvectors must be at least 1"));
                }
                match params {
                    CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
                        params.n_eigenvectors = n_eigenvectors
                    },
                    CugraphAlgorithmParams::BalancedCutClustering(params) => params.n_eigenvectors = n_eigenvectors,
                    _ => {},
                }
            },
            "evs_tolerance"
                if algorithm == CugraphAlgorithm::SpectralModularityMaximization
                    || algorithm == CugraphAlgorithm::BalancedCutClustering =>
            {
                let evs_tolerance = json_f64(value, key)?;
                if evs_tolerance < 0.0 {
                    return Err(plan_error("evs_tolerance must be non-negative"));
                }
                match params {
                    CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
                        params.evs_tolerance = evs_tolerance
                    },
                    CugraphAlgorithmParams::BalancedCutClustering(params) => params.evs_tolerance = evs_tolerance,
                    _ => {},
                }
            },
            "evs_max_iterations"
                if algorithm == CugraphAlgorithm::SpectralModularityMaximization
                    || algorithm == CugraphAlgorithm::BalancedCutClustering =>
            {
                let evs_max_iterations = json_u32(value, key)?;
                if evs_max_iterations < 1 {
                    return Err(plan_error("evs_max_iterations must be at least 1"));
                }
                match params {
                    CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
                        params.evs_max_iterations = evs_max_iterations
                    },
                    CugraphAlgorithmParams::BalancedCutClustering(params) => {
                        params.evs_max_iterations = evs_max_iterations
                    },
                    _ => {},
                }
            },
            "k_means_tolerance"
                if algorithm == CugraphAlgorithm::SpectralModularityMaximization
                    || algorithm == CugraphAlgorithm::BalancedCutClustering =>
            {
                let k_means_tolerance = json_f64(value, key)?;
                if k_means_tolerance < 0.0 {
                    return Err(plan_error("k_means_tolerance must be non-negative"));
                }
                match params {
                    CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
                        params.k_means_tolerance = k_means_tolerance
                    },
                    CugraphAlgorithmParams::BalancedCutClustering(params) => {
                        params.k_means_tolerance = k_means_tolerance
                    },
                    _ => {},
                }
            },
            "k_means_max_iterations"
                if algorithm == CugraphAlgorithm::SpectralModularityMaximization
                    || algorithm == CugraphAlgorithm::BalancedCutClustering =>
            {
                let k_means_max_iterations = json_u32(value, key)?;
                if k_means_max_iterations < 1 {
                    return Err(plan_error("k_means_max_iterations must be at least 1"));
                }
                match params {
                    CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
                        params.k_means_max_iterations = k_means_max_iterations
                    },
                    CugraphAlgorithmParams::BalancedCutClustering(params) => {
                        params.k_means_max_iterations = k_means_max_iterations
                    },
                    _ => {},
                }
            },
            _ => {
                return Err(plan_error(format!("unknown option '{key}' for {}", algorithm.function_name())));
            },
        }
    }

    Ok(())
}

fn parse_degree_type(value: &str, key: &str) -> Result<CugraphKCoreDegreeType> {
    match value {
        "in" => Ok(CugraphKCoreDegreeType::In),
        "out" => Ok(CugraphKCoreDegreeType::Out),
        "in_out" => Ok(CugraphKCoreDegreeType::InOut),
        other => Err(plan_error(format!("unsupported '{key}' value '{other}', expected 'in', 'out', or 'in_out'"))),
    }
}

fn literal_string_arg(args: &[Expr], index: usize, name: &str) -> Result<String> {
    match args.get(index) {
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)) => Ok(value.clone()),
            _ => Err(plan_error(format!("{name} must be a non-null string literal"))),
        },
        Some(_) => Err(plan_error(format!("{name} must be a literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
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

fn literal_i64_arg(args: &[Expr], index: usize, name: &str) -> Result<i64> {
    match args.get(index) {
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::Int8(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int16(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int32(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int64(Some(value)) => Ok(*value),
            ScalarValue::UInt8(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt16(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt32(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt64(Some(value)) => {
                i64::try_from(*value).map_err(|_| plan_error(format!("{name} is too large for an Int64 source vertex")))
            },
            _ => Err(plan_error(format!("{name} must be a non-null integer literal"))),
        },
        Some(_) => Err(plan_error(format!("{name} must be a literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
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
