use std::{
    fs,
    path::Path,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, BooleanArray, Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::ScalarValue,
    execution::SessionStateBuilder,
    logical_expr::{col, lit},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::displayable,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CUGRAPH_OPERATION_REGISTRY, CudfOptimizerConfig, CudfSessionStateBuilderExt, CugraphAlgorithm,
    CugraphAlgorithmParams, CugraphGraphConstructionPolicy, CugraphSqlConfig, GpuFallbackPolicy,
    NativeCudfComposedExec, parse_cugraph_algorithm_spec, stable_result_schema,
};

#[tokio::test]
async fn cugraph_sql_planning_registers_udtfs_on_session_state_builder() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let functions = state.table_functions();

    for definition in CUGRAPH_OPERATION_REGISTRY {
        assert!(
            functions.contains_key(definition.sql_function_name),
            "SQL surface must register every cuGraph operation registry entry: {}",
            definition.sql_function_name,
        );
    }

    let registered_cugraph_udtfs: std::collections::BTreeSet<&str> =
        functions.keys().filter_map(|name| name.strip_prefix("cugraph_").map(|_| name.as_str())).collect();
    let registry_function_names: std::collections::BTreeSet<&str> =
        CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.sql_function_name).collect();
    assert_eq!(
        registered_cugraph_udtfs, registry_function_names,
        "every `cugraph_*` UDTF on the session must come from the registry; \
         add a registry entry rather than registering a function out of band",
    );
}

#[test]
fn cugraph_sql_operation_registry_matches_native_engine_inventory() {
    use cudf_query_engine::GRAPH_OPERATION_REGISTRY;

    let sql_algorithms: Vec<_> = CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.native_algorithm).collect();
    let native_algorithms: Vec<_> = GRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.algorithm).collect();
    assert_eq!(
        sql_algorithms, native_algorithms,
        "SQL-side cuGraph registry must mirror the cudf-query-engine graph operation registry, in order",
    );

    for entry in CUGRAPH_OPERATION_REGISTRY {
        let native_name = entry.native_algorithm.definition().name;
        assert_eq!(
            entry.sql_function_name,
            format!("cugraph_{native_name}"),
            "SQL UDTF name must follow the cugraph_<native_name> convention",
        );
    }
}

#[tokio::test]
async fn cugraph_sql_planning_exposes_stable_output_schemas() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    assert_sql_schema(&ctx, "SELECT * FROM cugraph_pagerank('graph_edges')", CugraphAlgorithm::PageRank).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_bfs('graph_edges', 0)", CugraphAlgorithm::Bfs).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_louvain('graph_edges')", CugraphAlgorithm::Louvain).await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_eigenvector_centrality('graph_edges')",
        CugraphAlgorithm::EigenvectorCentrality,
    )
    .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_katz_centrality('graph_edges')", CugraphAlgorithm::KatzCentrality)
        .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_hits('graph_edges')", CugraphAlgorithm::Hits).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_sssp('graph_edges', 0)", CugraphAlgorithm::Sssp).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_degrees_all('graph_edges')", CugraphAlgorithm::DegreesAll).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_in_degrees_all('graph_edges')", CugraphAlgorithm::InDegreesAll)
        .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_out_degrees_all('graph_edges')", CugraphAlgorithm::OutDegreesAll)
        .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_weakly_connected_components('graph_edges')",
        CugraphAlgorithm::WeaklyConnectedComponents,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_strongly_connected_components('graph_edges')",
        CugraphAlgorithm::StronglyConnectedComponents,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_triangle_count_all('graph_edges')",
        CugraphAlgorithm::TriangleCountAll,
    )
    .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_core_number('graph_edges')", CugraphAlgorithm::CoreNumber).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_k_core('graph_edges')", CugraphAlgorithm::KCore).await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_minimum_spanning_tree('graph_edges', 'src', 'dst', 'weight')",
        CugraphAlgorithm::MinimumSpanningTree,
    )
    .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_pagerank_strict('graph_edges')", CugraphAlgorithm::PageRankStrict)
        .await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_leiden('graph_edges')", CugraphAlgorithm::Leiden).await;
    assert_sql_schema(&ctx, "SELECT * FROM cugraph_ecg('graph_edges')", CugraphAlgorithm::Ecg).await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_spectral_modularity_maximization('graph_edges')",
        CugraphAlgorithm::SpectralModularityMaximization,
    )
    .await;
    assert_sql_schema(
        &ctx,
        "SELECT * FROM cugraph_balanced_cut_clustering('graph_edges')",
        CugraphAlgorithm::BalancedCutClustering,
    )
    .await;
}

#[test]
fn cugraph_sql_planning_parses_defaults_explicit_columns_and_null_weight() {
    let config = CugraphSqlConfig::default();
    let spec = parse_cugraph_algorithm_spec(CugraphAlgorithm::PageRank, &[lit("graph_edges")], &config).unwrap();
    assert_eq!(spec.table_name.to_string(), "graph_edges");
    assert_eq!(spec.src_col, "src");
    assert_eq!(spec.dst_col, "dst");
    assert_eq!(spec.weight_col, None);
    assert_eq!(spec.graph_options.directed, true);
    assert_eq!(spec.graph_options.renumber, true);
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
    assert_eq!(spec.graph_options.directed, false);
    assert_eq!(spec.graph_options.renumber, false);

    // WCC must default to directed=false since it requires a symmetric graph.
    let spec =
        parse_cugraph_algorithm_spec(CugraphAlgorithm::WeaklyConnectedComponents, &[lit("edges")], &config).unwrap();
    assert_eq!(spec.graph_options.directed, false, "WCC must default to directed=false");

    // triangle_count_all / core_number / k_core must also default to
    // directed=false so their parity oracles (which Python cuGraph forces
    // undirected) are computed under the same semantics as the adapter.
    for algorithm in [CugraphAlgorithm::TriangleCountAll, CugraphAlgorithm::CoreNumber, CugraphAlgorithm::KCore] {
        let spec = parse_cugraph_algorithm_spec(algorithm, &[lit("edges")], &config).unwrap();
        assert_eq!(spec.graph_options.directed, false, "{} must default to directed=false", algorithm.function_name());
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
        CugraphAlgorithmParams::PageRank(datafusion_cudf::CugraphPageRankParams {
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
        CugraphAlgorithmParams::Bfs(datafusion_cudf::CugraphBfsParams { source_vertex: 42, depth_limit: Some(3) })
    );

    let spec = parse_cugraph_algorithm_spec(
        CugraphAlgorithm::Louvain,
        &[lit("graph_edges"), lit("src"), lit("dst"), lit("weight"), lit(r#"{"directed":false,"resolution":2.5}"#)],
        &config,
    )
    .unwrap();
    assert_eq!(spec.params, CugraphAlgorithmParams::Louvain(datafusion_cudf::CugraphLouvainParams { resolution: 2.5 }));

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
        CugraphAlgorithmParams::EigenvectorCentrality(datafusion_cudf::CugraphEigenvectorCentralityParams {
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
        CugraphAlgorithmParams::KatzCentrality(datafusion_cudf::CugraphKatzCentralityParams {
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
        CugraphAlgorithmParams::Hits(datafusion_cudf::CugraphHitsParams {
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
        CugraphAlgorithmParams::Sssp(datafusion_cudf::CugraphSsspParams { source_vertex: 5, cutoff: Some(10.5) }),
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
        CugraphAlgorithmParams::CoreNumber(datafusion_cudf::CugraphCoreNumberParams {
            degree_type: datafusion_cudf::CugraphKCoreDegreeType::In,
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
        CugraphAlgorithmParams::KCore(datafusion_cudf::CugraphKCoreParams {
            k: 3,
            degree_type: datafusion_cudf::CugraphKCoreDegreeType::Out,
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
        CugraphAlgorithmParams::PageRankStrict(datafusion_cudf::CugraphPageRankParams {
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
        CugraphAlgorithmParams::Leiden(datafusion_cudf::CugraphLeidenParams {
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
        CugraphAlgorithmParams::Ecg(datafusion_cudf::CugraphEcgParams {
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
        CugraphAlgorithmParams::SpectralModularityMaximization(datafusion_cudf::CugraphSpectralClusteringParams {
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
        CugraphAlgorithmParams::BalancedCutClustering(datafusion_cudf::CugraphSpectralClusteringParams {
            n_clusters: 3,
            seed: 11,
            ..Default::default()
        }),
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

async fn assert_sql_schema(ctx: &SessionContext, sql: &str, algorithm: CugraphAlgorithm) {
    let df = ctx.sql(sql).await.unwrap();
    let actual = df.schema().inner();
    let expected = stable_result_schema(algorithm);
    assert_eq!(actual.fields().len(), expected.fields().len());
    for (actual, expected) in actual.fields().iter().zip(expected.fields()) {
        assert_eq!(actual.name(), expected.name());
        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(actual.is_nullable(), expected.is_nullable());
    }
}

fn assert_error_contains(result: datafusion::common::Result<datafusion_cudf::CugraphAlgorithmSpec>, expected: &str) {
    let error = result.unwrap_err().to_string();
    assert!(error.contains(expected), "expected error containing {expected:?}, got {error:?}",);
}

#[test]
fn cugraph_sql_planning_schema_contracts_are_exact() {
    assert_eq!(
        schema_fields(CugraphAlgorithm::PageRank),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Bfs),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Louvain),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::EigenvectorCentrality),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::KatzCentrality),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false),],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Hits),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("hub_score", DataType::Float64, false),
            Field::new("authority_score", DataType::Float64, false),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::Sssp),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Float64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::DegreesAll),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("in_degree", DataType::Int64, false),
            Field::new("out_degree", DataType::Int64, false),
        ],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::InDegreesAll),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("in_degree", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::OutDegreesAll),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("out_degree", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::WeaklyConnectedComponents),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("label", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::StronglyConnectedComponents),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("label", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::TriangleCountAll),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("triangle_count", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::CoreNumber),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("core_number", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::KCore),
        vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::MinimumSpanningTree),
        vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)],
    );
    assert_eq!(
        schema_fields(CugraphAlgorithm::PageRankStrict),
        vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false)],
    );
    for algorithm in [
        CugraphAlgorithm::Leiden,
        CugraphAlgorithm::Ecg,
        CugraphAlgorithm::SpectralModularityMaximization,
        CugraphAlgorithm::BalancedCutClustering,
    ] {
        assert_eq!(
            schema_fields(algorithm),
            vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false)],
            "community algorithm {algorithm:?} must expose vertex + partition schema",
        );
    }
}

#[tokio::test]
async fn cugraph_sql_planning_direct_table_louvain_physical_plan_preserves_edge_schema_order() {
    let dir = unique_temp_dir("datafusion-cudf-cugraph-louvain-planning");
    let path = dir.join("tiny_communities.parquet");
    write_tiny_communities(&path);

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_communities", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('tiny_communities', 'src', 'dst', 'weight', '{\"directed\":false}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert_eq!(
        plan.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["vertex", "partition"]
    );
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("construction_policy=python_cugraph"),
        "cuGraph SQL EXPLAIN must expose the selected construction policy:\n{plan_display}"
    );
    assert!(
        plan_display.contains("edge_normalize_inserted=true"),
        "default Python-compatible cuGraph SQL planning should surface the inserted EdgeNormalize step:\n{plan_display}"
    );

    let raw_df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('tiny_communities', 'src', 'dst', 'weight', \
                                  '{\"directed\":false,\"construction_policy\":\"raw_libcugraph\"}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let raw_plan = raw_df.create_physical_plan().await.unwrap();
    let raw_plan_display = displayable(raw_plan.as_ref()).indent(false).to_string();
    assert!(
        raw_plan_display.contains("construction_policy=raw_libcugraph"),
        "raw cuGraph SQL EXPLAIN must expose the selected construction policy:\n{raw_plan_display}"
    );
    assert!(
        raw_plan_display.contains("edge_normalize_inserted=false"),
        "raw cuGraph SQL planning should expose that EdgeNormalize was skipped:\n{raw_plan_display}"
    );

    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn cugraph_sql_planning_skips_edge_normalize_for_precanonicalized_edge_views() {
    let dir = unique_temp_dir("datafusion-cudf-cugraph-normalized-planning");
    let directed_path = dir.join("tiny_directed_edges.parquet");
    let undirected_path = dir.join("tiny_communities.parquet");
    write_tiny_directed_edges(&directed_path);
    write_tiny_communities(&undirected_path);

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", directed_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.register_parquet("tiny_communities", undirected_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.sql(
        "CREATE VIEW canonical_directed_edges AS \
         SELECT src, dst \
         FROM tiny_directed_edges \
         WHERE active \
         GROUP BY src, dst",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx.sql(
        "CREATE VIEW canonical_undirected_edges AS \
         SELECT src, dst \
         FROM ( \
           SELECT src, dst FROM tiny_communities \
           UNION ALL \
           SELECT dst AS src, src AS dst FROM tiny_communities \
         ) e \
         GROUP BY src, dst",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    ctx.sql(
        "CREATE VIEW weighted_min_edges AS \
         SELECT src, dst, MIN(weight) AS weight \
         FROM tiny_directed_edges \
         WHERE active \
         GROUP BY src, dst",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let directed_df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('canonical_directed_edges', 'src', 'dst') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let directed_plan = directed_df.create_physical_plan().await.unwrap();
    let directed_display = displayable(directed_plan.as_ref()).indent(false).to_string();
    assert!(
        directed_display.contains("construction_policy=python_cugraph"),
        "default policy label should stay visible for pre-canonical directed view:\n{directed_display}"
    );
    assert!(
        directed_display.contains("edge_normalize_inserted=false"),
        "directed unweighted GROUP BY src,dst should prove edge dedup provenance:\n{directed_display}"
    );

    let undirected_df = ctx
        .sql(
            "SELECT vertex, partition \
             FROM cugraph_louvain('canonical_undirected_edges', 'src', 'dst', NULL, '{\"directed\":false}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let undirected_plan = undirected_df.create_physical_plan().await.unwrap();
    let undirected_display = displayable(undirected_plan.as_ref()).indent(false).to_string();
    assert!(
        undirected_display.contains("construction_policy=python_cugraph"),
        "default policy label should stay visible for pre-canonical undirected view:\n{undirected_display}"
    );
    assert!(
        undirected_display.contains("edge_normalize_inserted=false"),
        "undirected unweighted UNION ALL reverse expansion + GROUP BY src,dst should prove full provenance:\n{undirected_display}"
    );

    let weighted_df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('weighted_min_edges', 'src', 'dst', 'weight') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let weighted_plan = weighted_df.create_physical_plan().await.unwrap();
    let weighted_display = displayable(weighted_plan.as_ref()).indent(false).to_string();
    assert!(
        weighted_display.contains("edge_normalize_inserted=true"),
        "weighted MIN(weight) aggregate must not prove Python cuGraph keep-last duplicate semantics:\n{weighted_display}"
    );

    fs::remove_dir_all(dir).ok();
}

// Per-algorithm Python-oracle GPU tests for PageRank, BFS, and Louvain
// previously lived here. They have been subsumed by the registry-driven
// `tests/cugraph_parity.rs` harness, which reads checked-in oracle JSON
// produced by `scripts/cugraph_oracle.py`. The composition test below
// stays in place and will move into the harness as a Phase 2
// composition entry.

#[tokio::test]
#[ignore = "requires CUDA plus Python cudf/cugraph for the official oracle"]
async fn cugraph_sql_gpu_pagerank_filter_sort_limit_stays_gpu_resident_until_collect() {
    let dir = unique_temp_dir("datafusion-cudf-cugraph-pagerank-compose");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let Some(oracle) = python_pagerank_oracle(&path) else {
        eprintln!("skipping cuGraph PageRank composition e2e because the Python cuGraph oracle is unavailable");
        fs::remove_dir_all(dir).ok();
        return;
    };
    eprintln!("Python cuGraph oracle versions: {}", oracle.versions);

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_cudf_native(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql(
        "CREATE VIEW graph_edges AS \
         SELECT src, dst, weight FROM tiny_directed_edges WHERE active",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('graph_edges', 'src', 'dst', 'weight') \
             WHERE value > 0 \
             ORDER BY value DESC, vertex ASC \
             LIMIT 3",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("NativeCudfComposedExec"),
        "PageRank post-algorithm filter/sort/limit should be a composed native GPU handoff:\n{plan_display}"
    );
    assert!(
        plan_display.contains("CugraphAlgoExec"),
        "composed native plan should keep cuGraph as the GPU-resident producer child:\n{plan_display}"
    );
    let composed =
        find_composed_exec(plan.as_ref()).expect("PageRank composition plan should contain NativeCudfComposedExec");
    let native_plan = composed.native_plan().to_string();
    for expected in ["GpuResidentSource", "Filter", "Sort", "Limit"] {
        assert!(
            native_plan.contains(expected),
            "composed native consumer should include {expected} over the cuGraph result:\n{native_plan}"
        );
    }

    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let actual = pagerank_rows(&batches);
    let mut expected = oracle.rows.clone();
    expected.retain(|row| row.value > 0.0);
    expected.sort_by(|left, right| {
        right.value.partial_cmp(&left.value).unwrap().then_with(|| left.vertex.cmp(&right.vertex))
    });
    expected.truncate(3);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.vertex, expected.vertex);
        let delta = (actual.value - expected.value).abs();
        assert!(
            delta <= 1.0e-6,
            "vertex {} composed PageRank mismatch: actual={} expected={} delta={}",
            actual.vertex,
            actual.value,
            expected.value,
            delta
        );
    }
    fs::remove_dir_all(dir).ok();
}

fn schema_fields(algorithm: CugraphAlgorithm) -> Vec<Field> {
    stable_result_schema(algorithm).fields().iter().map(|field| field.as_ref().clone()).collect()
}

fn find_composed_exec(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> Option<&NativeCudfComposedExec> {
    if let Some(composed) = plan.as_any().downcast_ref::<NativeCudfComposedExec>() {
        return Some(composed);
    }
    plan.children().into_iter().find_map(|child| find_composed_exec(child.as_ref()))
}

#[derive(Debug, Clone)]
struct PageRankRow {
    vertex: i64,
    value: f64,
}

#[derive(Debug)]
struct PythonPageRankOracle {
    versions: String,
    rows: Vec<PageRankRow>,
}

fn pagerank_rows(batches: &[RecordBatch]) -> Vec<PageRankRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let values = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for index in 0..batch.num_rows() {
            rows.push(PageRankRow { vertex: vertices.value(index), value: values.value(index) });
        }
    }
    rows
}

fn python_pagerank_oracle(path: &Path) -> Option<PythonPageRankOracle> {
    let script = r#"
import json
import sys

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

df = cudf.read_parquet(sys.argv[1])
df = df[df["active"]][["src", "dst", "weight"]]
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight", renumber=True)
result = cugraph.pagerank(graph, alpha=0.85, tol=1.0e-6, max_iter=100)
value_col = "pagerank" if "pagerank" in result.columns else [name for name in result.columns if name != "vertex"][0]
result = result.rename(columns={value_col: "value"})
result["vertex"] = result["vertex"].astype("int64")
result["value"] = result["value"].astype("float64")
result = result.sort_values("vertex")
rows = [{"vertex": int(row.vertex), "value": float(row.value)} for row in result.to_pandas().itertuples()]
print(json.dumps({
    "versions": {
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    },
    "rows": rows,
}))
"#;
    let output = match Command::new("python3").arg("-c").arg(script).arg(path).output() {
        Ok(output) => output,
        Err(error) => {
            handle_python_oracle_unavailable(format!("Python cuGraph oracle unavailable: {error}"));
            return None;
        },
    };
    if output.status.code() == Some(77) {
        handle_python_oracle_unavailable(String::from_utf8_lossy(&output.stderr));
        return None;
    }
    assert!(
        output.status.success(),
        "Python cuGraph oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let versions = value["versions"].to_string();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| PageRankRow { vertex: row["vertex"].as_i64().unwrap(), value: row["value"].as_f64().unwrap() })
        .collect();
    Some(PythonPageRankOracle { versions, rows })
}

fn handle_python_oracle_unavailable(message: impl AsRef<str>) {
    if python_oracle_required() {
        panic!("{}", message.as_ref().trim());
    }
    eprintln!("{}", message.as_ref().trim());
}

fn python_oracle_required() -> bool {
    matches!(
        std::env::var("CUDF_CUGRAPH_ORACLE_REQUIRED").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "required" | "REQUIRED")
    )
}

fn write_tiny_directed_edges(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
        Field::new("active", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 0, 1, 2, 3, 0, 4])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 3, 3, 4, 1, 5])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true, true, true, true, true, true, false])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_tiny_communities(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5, 2, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 0, 4, 5, 3, 3, 3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 4.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
