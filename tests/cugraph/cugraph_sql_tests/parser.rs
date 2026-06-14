use super::{support::*, *};

#[test]
fn cugraph_bfs_string_source_vertex_uses_logical_output_schema() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit("alice"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"output_mode":"normalized"}"#),
        ],
        &config,
    )
    .unwrap();
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        panic!("expected BFS params");
    };
    assert_eq!(params.source_vertex, CugraphBfsSourceVertex::Utf8("alice".to_string()));
    assert_eq!(
        spec.output_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>(),
        vec![
            Field::new("vertex", DataType::Utf8, false),
            Field::new("distance", DataType::Int64, true),
            Field::new("predecessor", DataType::Utf8, true),
            Field::new("reachable", DataType::Boolean, false),
        ],
    );
}

#[test]
fn cugraph_sql_planning_parses_defaults_explicit_columns_and_null_weight() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[lit("graph_edges")], &config).unwrap();
    assert_eq!(spec.table_name.to_string(), "graph_edges");
    assert_eq!(spec.src_col, "src");
    assert_eq!(spec.dst_col, "dst");
    assert_eq!(spec.weight_col, None);
    assert!(spec.graph_options.directed);
    assert!(spec.graph_options.renumber);
    assert_eq!(spec.graph_options.construction_policy, CugraphGraphConstructionPolicy::PythonCugraph);

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Louvain,
        &[
            lit("main.graph_edges"),
            lit("from_id"),
            lit("to_id"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"directed":false,"renumber":false}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.table_name.to_string(), "main.graph_edges");
    assert_eq!(spec.src_col, "from_id");
    assert_eq!(spec.dst_col, "to_id");
    assert_eq!(spec.weight_col, None);
    assert!(!spec.graph_options.directed);
    assert!(!spec.graph_options.renumber);

    // WCC must default to directed=false since it requires a symmetric graph.
    let spec =
        parse_cugraph_algorithm_spec(CugraphAlgorithm::WeaklyConnectedComponents, &[lit("edges")], &config).unwrap();
    assert!(!spec.graph_options.directed, "WCC must default to directed=false");

    // triangle_count_all / core_number / k_core must also default to
    // directed=false so their parity oracles (which Python cuGraph forces
    // undirected) are computed under the same semantics as the adapter.
    for algorithm in [
        CugraphAlgorithm::TriangleCountAll,
        CugraphAlgorithm::CoreNumber,
        CugraphAlgorithm::KCore,
        CugraphAlgorithm::ForceAtlas2,
    ] {
        let spec = parse_cugraph_algorithm_spec(algorithm, &[lit("edges")], &config).unwrap();
        assert!(!spec.graph_options.directed, "{} must default to directed=false", algorithm.function_name());
    }
}

#[test]
fn cugraph_sql_planning_parses_algorithm_options() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRank,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            lit("weight"),
            lit(r#"{"alpha":0.7,"epsilon":0.001,"max_iterations":25}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.weight_col, Some("weight".to_string()));
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::PageRank(datafusion_nexus::CugraphPageRankParams {
            alpha: 0.7,
            epsilon: 0.001,
            max_iterations: 25,
        })
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(42_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"depth_limit":3}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Bfs(Box::new(datafusion_nexus::CugraphBfsParams {
            source_vertex: CugraphBfsSourceVertex::Int64(42),
            depth_limit: Some(3),
            ..Default::default()
        }))
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(42_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"depth_limit":null}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Bfs(Box::new(datafusion_nexus::CugraphBfsParams {
            source_vertex: CugraphBfsSourceVertex::Int64(42),
            depth_limit: None,
            ..Default::default()
        }))
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(42_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"output_mode":"normalized"}"#),
        ],
        &config,
    )
    .unwrap();
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        panic!("expected BFS params");
    };
    assert_eq!(params.output.mode, CugraphBfsOutputMode::Normalized);
    assert!(!params.output.return_target_info);
    assert!(spec.output_schema.field_with_name("distance").unwrap().is_nullable());
    assert!(spec.output_schema.field_with_name("reachable").is_ok());

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(42_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(
                r#"{"output_mode":"normalized","target_vertices_table":"targets","target_vertex_col":"vertex_id","return_target_info":true}"#,
            ),
        ],
        &config,
    )
    .unwrap();
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        panic!("expected BFS params");
    };
    assert_eq!(params.output.mode, CugraphBfsOutputMode::Normalized);
    assert!(params.output.return_target_info);
    assert!(spec.output_schema.field_with_name("target_found").is_ok());
    assert!(spec.output_schema.field_with_name("target_distance").unwrap().is_nullable());

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Bfs,
        &[
            lit("graph_edges"),
            lit(7_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{
                  "edge_id_col":"edge_id",
                  "include_edges_table":"allowed_edges",
                  "include_edge_src_col":"src",
                  "include_edge_dst_col":"dst",
                  "include_edge_id_col":"edge_id",
                  "exclude_vertices_table":"excluded_vertices",
                  "exclude_vertex_col":"vertex_id",
                  "target_vertices_table":"targets",
                  "target_vertex_col":"vertex_id",
                  "return_target_info":true
                }"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.edge_id_col, Some("edge_id".to_string()));
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        panic!("expected BFS params");
    };
    assert_eq!(params.predicates.exclude_vertices.as_ref().unwrap().table_name.to_string(), "excluded_vertices");
    assert_eq!(params.predicates.exclude_vertices.as_ref().unwrap().column, "vertex_id");
    assert_eq!(params.predicates.target_vertices.as_ref().unwrap().table_name.to_string(), "targets");
    let include_edges = params.predicates.include_edges.as_ref().unwrap();
    assert_eq!(include_edges.table_name.to_string(), "allowed_edges");
    assert_eq!(include_edges.source_col, "src");
    assert_eq!(include_edges.destination_col, "dst");
    assert_eq!(include_edges.edge_id_col.as_deref(), Some("edge_id"));

    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(
                    r#"{"include_vertices_table":"included","include_vertex_col":"vertex_id","exclude_vertices_table":"excluded","exclude_vertex_col":"vertex_id"}"#,
                ),
            ],
            &config,
        ),
        "mutually exclusive",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"include_edge_ids_table":"edge_ids","include_edge_ids_col":"edge_id"}"#),
            ],
            &config,
        ),
        "requires primary option 'edge_id_col'",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"return_target_info":true}"#),
            ],
            &config,
        ),
        "return_target_info requires target_vertices_table",
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Louvain,
        &[lit("graph_edges"), lit("src"), lit("dst"), lit("weight"), lit(r#"{"directed":false,"resolution":2.5}"#)],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Louvain(datafusion_nexus::CugraphLouvainParams { resolution: 2.5 })
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::EigenvectorCentrality,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"epsilon":0.0005,"max_iterations":80}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::EigenvectorCentrality(datafusion_nexus::CugraphEigenvectorCentralityParams {
            epsilon: 0.0005,
            max_iterations: 80,
        })
    );

    // Eigenvector centrality must reject zero max_iterations the same way
    // PageRank does, since the underlying cugraph-rs newtype requires >= 1.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::EigenvectorCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iterations":0}"#),
            ],
            &config,
        ),
        "max_iterations must be at least 1",
    );

    // epsilon=0 must also be rejected at planning. cugraph-rs's
    // `NonNegativeF64` accepts zero, but libcugraph's convergence test
    // `diff_sum < vertex_count * epsilon` never accepts on a non-empty
    // graph when the tolerance is zero, so the solver burns all
    // max_iterations of GPU work before failing.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::EigenvectorCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"epsilon":0}"#),
            ],
            &config,
        ),
        "epsilon must be positive",
    );

    // Katz centrality accepts alpha/beta/epsilon/max_iterations.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::KatzCentrality,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"alpha":0.02,"beta":0.5,"epsilon":1e-7,"max_iterations":250}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::KatzCentrality(datafusion_nexus::CugraphKatzCentralityParams {
            alpha: 0.02,
            beta: 0.5,
            epsilon: 1.0e-7,
            max_iterations: 250,
        }),
    );

    // Same epsilon=0 reasoning as eigenvector centrality: libcugraph
    // never converges when the tolerance is zero on a non-empty graph.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KatzCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"epsilon":0}"#),
            ],
            &config,
        ),
        "epsilon must be positive",
    );

    // alpha must be strictly positive — alpha=0 makes Katz centrality
    // degenerate to the constant `beta` vector.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KatzCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"alpha":0}"#),
            ],
            &config,
        ),
        "alpha must be positive",
    );

    // libcugraph caps alpha at 1 (Katz attenuation factor must be
    // smaller than the inverse of the graph's largest eigenvalue);
    // alpha > 1 fails inside the kernel, so reject at planning.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KatzCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"alpha":2}"#),
            ],
            &config,
        ),
        "alpha must be <= 1",
    );

    // beta=0 with the zero initial vector reaches normalization with
    // all-zero scores. cugraph-rs's `NonNegativeF64` accepts zero, so
    // the only place left to reject it is the SQL planner.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KatzCentrality,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"beta":0}"#),
            ],
            &config,
        ),
        "beta must be positive",
    );

    // HITS accepts epsilon/max_iterations/normalize. Mirrors
    // PageRank's pattern; `normalize` defaults to false matching the
    // cugraph-rs `HitsParams::default()` convention.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Hits,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"epsilon":0.0001,"max_iterations":50,"normalize":true}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Hits(datafusion_nexus::CugraphHitsParams {
            epsilon: 0.0001,
            max_iterations: 50,
            normalize: true,
        }),
    );

    // epsilon=0 is the same trap as the other centrality solvers:
    // libcugraph never converges on a non-empty graph at zero
    // tolerance, so reject at planning.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Hits,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"epsilon":0}"#),
            ],
            &config,
        ),
        "epsilon must be positive",
    );

    // max_iterations=0 is rejected by cugraph-rs's IterationCount
    // newtype at runtime; surface that at planning so the failure
    // path is consistent with PageRank / eigenvector / Katz.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Hits,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iterations":0}"#),
            ],
            &config,
        ),
        "max_iterations must be at least 1",
    );

    // SSSP: source_vertex as second positional arg, optional cutoff in JSON.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Sssp,
        &[
            lit("graph_edges"),
            lit(5_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"cutoff":10.5}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Sssp(datafusion_nexus::CugraphSsspParams { source_vertex: 5, cutoff: Some(10.5) }),
    );
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Sssp,
        &[
            lit("graph_edges"),
            lit(5_i64),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"cutoff":null}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Sssp(datafusion_nexus::CugraphSsspParams { source_vertex: 5, cutoff: None }),
    );
    // cutoff < 0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Sssp,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"cutoff":-1.0}"#),
            ],
            &config,
        ),
        "cutoff must be non-negative",
    );

    // CoreNumber: degree_type option.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::CoreNumber,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"degree_type":"in"}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::CoreNumber(datafusion_nexus::CugraphCoreNumberParams {
            degree_type: datafusion_nexus::CugraphKCoreDegreeType::In,
        }),
    );

    // KCore: k and degree_type.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::KCore,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"k":3,"degree_type":"out"}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::KCore(datafusion_nexus::CugraphKCoreParams {
            k: 3,
            degree_type: datafusion_nexus::CugraphKCoreDegreeType::Out,
        }),
    );
    // k=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::KCore,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"k":0}"#),
            ],
            &config,
        ),
        "k_core k must be at least 1",
    );
    // Invalid degree_type rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::CoreNumber,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"degree_type":"both"}"#),
            ],
            &config,
        ),
        "unsupported 'degree_type' value 'both'",
    );

    // PageRank strict: same options as PageRank.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRankStrict,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"alpha":0.9,"epsilon":0.0001,"max_iterations":50}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::PageRankStrict(datafusion_nexus::CugraphPageRankParams {
            alpha: 0.9,
            epsilon: 0.0001,
            max_iterations: 50,
        }),
    );
    // PageRank strict: epsilon=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRankStrict,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"epsilon":0}"#),
            ],
            &config,
        ),
        "PageRank strict epsilon must be positive",
    );

    // Leiden: max_level, resolution, theta, seed.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Leiden,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"max_level":50,"resolution":0.8,"theta":0.5,"seed":42}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Leiden(datafusion_nexus::CugraphLeidenParams {
            max_level: 50,
            resolution: 0.8,
            theta: 0.5,
            seed: 42,
        }),
    );
    // Leiden: max_level=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Leiden,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_level":0}"#),
            ],
            &config,
        ),
        "Leiden max_level must be at least 1",
    );
    // Leiden: resolution=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Leiden,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"resolution":0}"#),
            ],
            &config,
        ),
        "resolution must be positive",
    );

    // ECG: min_weight, ensemble_size, max_level, threshold, resolution, seed.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Ecg,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"min_weight":0.01,"ensemble_size":5,"max_level":50,"threshold":1e-5,"resolution":0.9,"seed":7}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::Ecg(datafusion_nexus::CugraphEcgParams {
            min_weight: 0.01,
            ensemble_size: 5,
            max_level: 50,
            threshold: 1e-5,
            resolution: 0.9,
            seed: 7,
        }),
    );
    // ECG: ensemble_size=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Ecg,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"ensemble_size":0}"#),
            ],
            &config,
        ),
        "ECG ensemble_size must be at least 1",
    );

    // SpectralModularityMaximization: n_clusters, n_eigenvectors, seed.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::SpectralModularityMaximization,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"n_clusters":4,"n_eigenvectors":3,"seed":99}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::SpectralModularityMaximization(datafusion_nexus::CugraphSpectralClusteringParams {
            n_clusters: 4,
            n_eigenvectors: 3,
            seed: 99,
            ..Default::default()
        }),
    );
    // SpectralModularityMaximization: n_clusters=0 rejected.
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::SpectralModularityMaximization,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"n_clusters":0}"#),
            ],
            &config,
        ),
        "n_clusters must be at least 1",
    );

    // BalancedCutClustering: same params struct as SpectralModularityMaximization.
    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::BalancedCutClustering,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"n_clusters":3,"seed":11}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::BalancedCutClustering(datafusion_nexus::CugraphSpectralClusteringParams {
            n_clusters: 3,
            seed: 11,
            ..Default::default()
        }),
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::ForceAtlas2,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            lit("weight"),
            lit(r#"{"max_iter":25,"outbound_attraction_distribution":true,"lin_log_mode":true,"overlap_scaling_ratio":3.0,"edge_weight_influence":0.5,"jitter_tolerance":0.25,"barnes_hut_optimize":true,"barnes_hut_theta":0.75,"scaling_ratio":4.0,"strong_gravity_mode":true,"gravity":2.0,"verbose":true,"seed":123}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.weight_col, Some("weight".to_string()));
    assert_eq!(
        spec.params,
        CugraphAlgorithmParams::ForceAtlas2(datafusion_nexus::CugraphForceAtlas2Params {
            max_iter: 25,
            outbound_attraction_distribution: true,
            lin_log_mode: true,
            prevent_overlapping: false,
            overlap_scaling_ratio: 3.0,
            edge_weight_influence: 0.5,
            jitter_tolerance: 0.25,
            barnes_hut_optimize: true,
            barnes_hut_theta: 0.75,
            scaling_ratio: 4.0,
            strong_gravity_mode: true,
            gravity: 2.0,
            verbose: true,
            seed: 123,
        }),
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::ForceAtlas2,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"barnes_hut_theta":1.5}"#),
            ],
            &config,
        ),
        "barnes_hut_theta must be in [0, 1]",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::ForceAtlas2,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iter":0}"#),
            ],
            &config,
        ),
        "ForceAtlas2 max_iter must be at least 1",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::ForceAtlas2,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iter":2147483648}"#),
            ],
            &config,
        ),
        "ForceAtlas2 max_iter must be at most i32::MAX",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::ForceAtlas2,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"prevent_overlapping":true}"#),
            ],
            &config,
        ),
        "prevent_overlapping requires vertex_radius side input",
    );
}

#[test]
fn cugraph_sql_planning_parses_construction_policy_options() {
    let config = CugraphSqlConfig::default();

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRank,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"construction_policy":"python_cugraph"}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.graph_options.construction_policy, CugraphGraphConstructionPolicy::PythonCugraph);

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::PageRank,
        &[
            lit("graph_edges"),
            lit("src"),
            lit("dst"),
            datafusion::logical_expr::lit(ScalarValue::Null),
            lit(r#"{"construction_policy":"raw_libcugraph"}"#),
        ],
        &config,
    )
    .unwrap();
    assert_eq!(spec.graph_options.construction_policy, CugraphGraphConstructionPolicy::RawLibcugraph);

    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"construction_policy":"networkx"}"#),
            ],
            &config,
        ),
        "unsupported cuGraph construction_policy 'networkx'",
    );
}

#[test]
fn cugraph_sql_planning_rejects_invalid_arguments() {
    let config = CugraphSqlConfig::default();
    assert_error_contains(
        parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[col("graph_edges")], &config),
        "table_name must be a literal",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[lit("SELECT src FROM edges")], &config),
        "must be a simple table",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"unknown":1}"#),
            ],
            &config,
        ),
        "unknown option 'unknown'",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"alpha":1.2}"#),
            ],
            &config,
        ),
        "alpha must be in [0, 1]",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::PageRank,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"max_iterations":0}"#),
            ],
            &config,
        ),
        "max_iterations must be at least 1",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"depth_limit":-1}"#),
            ],
            &config,
        ),
        "depth_limit must be non-negative",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Bfs,
            &[
                lit("graph_edges"),
                lit(0_i64),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"output_mode":"wide"}"#),
            ],
            &config,
        ),
        "unsupported BFS output_mode 'wide'",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::Louvain,
            &[
                lit("graph_edges"),
                lit("src"),
                lit("dst"),
                datafusion::logical_expr::lit(ScalarValue::Null),
                lit(r#"{"resolution":0}"#),
            ],
            &config,
        ),
        "resolution must be positive",
    );
    // MST without weight_col is rejected at planning time.
    assert_error_contains(
        parse_cugraph_algorithm_spec(CugraphAlgorithm::MinimumSpanningTree, &[lit("graph_edges")], &config),
        "requires a weight column",
    );
    assert_error_contains(
        parse_cugraph_algorithm_spec(
            CugraphAlgorithm::MinimumSpanningTree,
            &[lit("graph_edges"), lit("src"), lit("dst"), datafusion::logical_expr::lit(ScalarValue::Null)],
            &config,
        ),
        "requires a weight column",
    );
}
