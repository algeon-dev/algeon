use super::*;

pub(super) fn hits_params(params: &crate::plan::GraphHitsParams) -> Result<cugraph::centrality::HitsParams> {
    Ok(cugraph::centrality::HitsParams {
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.hits_params",
            "invalid cuGraph HITS epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.hits_params",
            "invalid cuGraph HITS max_iterations",
        ))?,
        normalize: params.normalize,
        do_expensive_check: false,
    })
}

pub(super) fn betweenness_centrality_params(
    params: &crate::plan::GraphBetweennessCentralityParams,
) -> cugraph::centrality::BetweennessCentralityParams {
    cugraph::centrality::BetweennessCentralityParams {
        normalized: params.normalized,
        include_endpoints: params.include_endpoints,
        do_expensive_check: false,
    }
}

pub(super) fn edge_betweenness_centrality_params(
    params: &crate::plan::GraphEdgeBetweennessCentralityParams,
) -> cugraph::centrality::EdgeBetweennessCentralityParams {
    cugraph::centrality::EdgeBetweennessCentralityParams { normalized: params.normalized, do_expensive_check: false }
}

pub(super) fn katz_centrality_params(
    params: &crate::plan::GraphKatzCentralityParams,
) -> Result<cugraph::centrality::KatzCentralityParams> {
    Ok(cugraph::centrality::KatzCentralityParams {
        alpha: cugraph::params::NonNegativeF64::new(params.alpha).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality alpha",
        ))?,
        beta: cugraph::params::NonNegativeF64::new(params.beta).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality beta",
        ))?,
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn eigenvector_centrality_params(
    params: &crate::plan::GraphEigenvectorCentralityParams,
) -> Result<cugraph::centrality::EigenvectorCentralityParams> {
    Ok(cugraph::centrality::EigenvectorCentralityParams {
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.eigenvector_centrality_params",
            "invalid cuGraph eigenvector centrality epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.eigenvector_centrality_params",
            "invalid cuGraph eigenvector centrality max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn pagerank_params(
    params: &crate::plan::GraphPageRankParams,
) -> Result<cugraph::centrality::PageRankParams> {
    Ok(cugraph::centrality::PageRankParams {
        alpha: cugraph::params::ClosedUnitF64::new(params.alpha).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank alpha",
        ))?,
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn personalized_pagerank_params(
    params: &crate::plan::GraphPageRankParams,
) -> Result<cugraph::centrality::PersonalizedPageRankParams> {
    Ok(cugraph::centrality::PersonalizedPageRankParams {
        alpha: cugraph::params::ClosedUnitF64::new(params.alpha).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.personalized_pagerank_params",
            "invalid cuGraph personalized PageRank alpha",
        ))?,
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.personalized_pagerank_params",
            "invalid cuGraph personalized PageRank epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.personalized_pagerank_params",
            "invalid cuGraph personalized PageRank max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn bfs_params(params: &crate::plan::GraphBfsParams) -> Result<cugraph::traversal::BfsParams> {
    let depth_limit = match params.depth_limit {
        None | Some(0) => None,
        Some(value) if value > 0 => {
            let value = usize::try_from(value).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.graph_algorithm.bfs_params",
                    "BFS depth_limit does not fit usize",
                )
                .with_fact("depth_limit", value)
            })?;
            Some(cugraph::params::BfsDepthLimit::new(value).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.bfs_params",
                "invalid cuGraph BFS depth_limit",
            ))?)
        },
        Some(value) => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_algorithm.bfs_params",
                "BFS depth_limit must be non-negative",
            )
            .with_fact("depth_limit", value));
        },
    };
    Ok(cugraph::traversal::BfsParams { depth_limit, compute_predecessors: true, do_expensive_check: false })
}

pub(super) fn louvain_params(params: &crate::plan::GraphLouvainParams) -> Result<cugraph::community::LouvainParams> {
    Ok(cugraph::community::LouvainParams {
        max_level: cugraph::params::IterationCount::new(params.max_level).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.louvain_params",
            "invalid cuGraph Louvain max_level",
        ))?,
        threshold: cugraph::params::NonNegativeF64::new(params.threshold).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.louvain_params",
            "invalid cuGraph Louvain threshold",
        ))?,
        resolution: cugraph::params::PositiveF64::new(params.resolution).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.louvain_params",
            "invalid cuGraph Louvain resolution",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn leiden_params(params: &crate::plan::GraphLeidenParams) -> Result<cugraph::community::LeidenParams> {
    Ok(cugraph::community::LeidenParams {
        max_level: cugraph::params::IterationCount::new(params.max_level).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.leiden_params",
            "invalid cuGraph Leiden max_level",
        ))?,
        resolution: cugraph::params::PositiveF64::new(params.resolution).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.leiden_params",
            "invalid cuGraph Leiden resolution",
        ))?,
        theta: cugraph::params::PositiveF64::new(params.theta).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.leiden_params",
            "invalid cuGraph Leiden theta",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn ecg_params(params: &crate::plan::GraphEcgParams) -> Result<cugraph::community::EcgParams> {
    Ok(cugraph::community::EcgParams {
        min_weight: cugraph::params::NonNegativeF64::new(params.min_weight).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG min_weight",
        ))?,
        ensemble_size: cugraph::params::IterationCount::new(params.ensemble_size).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG ensemble_size",
        ))?,
        max_level: cugraph::params::IterationCount::new(params.max_level).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG max_level",
        ))?,
        threshold: cugraph::params::NonNegativeF64::new(params.threshold).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG threshold",
        ))?,
        resolution: cugraph::params::PositiveF64::new(params.resolution).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG resolution",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn spectral_clustering_params(
    params: &crate::plan::GraphSpectralClusteringParams,
) -> Result<cugraph::community::SpectralClusteringParams> {
    let n_clusters = std::num::NonZeroUsize::new(params.n_clusters as usize).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.spectral_clustering_params",
            "spectral clustering n_clusters must be at least 1",
        )
        .with_fact("n_clusters", params.n_clusters)
    })?;
    let n_eigenvectors = std::num::NonZeroUsize::new(params.n_eigenvectors as usize).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.spectral_clustering_params",
            "spectral clustering n_eigenvectors must be at least 1",
        )
        .with_fact("n_eigenvectors", params.n_eigenvectors)
    })?;
    if params.evs_max_iterations > i32::MAX as u32 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.spectral_clustering_params",
            "cuGraph spectral clustering evs_max_iterations must fit the libcugraph int32 iteration count",
        )
        .with_fact("evs_max_iterations", params.evs_max_iterations));
    }
    if params.k_means_max_iterations > i32::MAX as u32 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.spectral_clustering_params",
            "cuGraph spectral clustering k_means_max_iterations must fit the libcugraph int32 iteration count",
        )
        .with_fact("k_means_max_iterations", params.k_means_max_iterations));
    }
    Ok(cugraph::community::SpectralClusteringParams {
        n_clusters,
        n_eigenvectors,
        evs_tolerance: cugraph::params::NonNegativeF64::new(params.evs_tolerance).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.spectral_clustering_params",
            "invalid cuGraph spectral clustering evs_tolerance",
        ))?,
        evs_max_iterations: cugraph::params::IterationCount::new(params.evs_max_iterations).map_err(
            map_cugraph_error(
                "nexus_query_engine.graph_algorithm.spectral_clustering_params",
                "invalid cuGraph spectral clustering evs_max_iterations",
            ),
        )?,
        k_means_tolerance: cugraph::params::NonNegativeF64::new(params.k_means_tolerance).map_err(
            map_cugraph_error(
                "nexus_query_engine.graph_algorithm.spectral_clustering_params",
                "invalid cuGraph spectral clustering k_means_tolerance",
            ),
        )?,
        k_means_max_iterations: cugraph::params::IterationCount::new(params.k_means_max_iterations).map_err(
            map_cugraph_error(
                "nexus_query_engine.graph_algorithm.spectral_clustering_params",
                "invalid cuGraph spectral clustering k_means_max_iterations",
            ),
        )?,
        do_expensive_check: false,
    })
}

pub(super) fn force_atlas2_params(
    params: &crate::plan::GraphForceAtlas2Params,
) -> Result<cugraph::layout::ForceAtlas2Params> {
    if params.prevent_overlapping {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.force_atlas2_params",
            "cuGraph ForceAtlas2 prevent_overlapping requires vertex_radius side input, which the native graph API does not support yet",
        ));
    }
    if params.max_iter > i32::MAX as u32 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.force_atlas2_params",
            "cuGraph ForceAtlas2 max_iter must fit the libcugraph int32 iteration count",
        )
        .with_fact("max_iter", params.max_iter)
        .with_fact("max_supported", i32::MAX));
    }
    Ok(cugraph::layout::ForceAtlas2Params {
        max_iter: cugraph::params::IterationCount::new(params.max_iter).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 max_iter",
        ))?,
        outbound_attraction_distribution: params.outbound_attraction_distribution,
        lin_log_mode: params.lin_log_mode,
        prevent_overlapping: params.prevent_overlapping,
        overlap_scaling_ratio: cugraph::params::PositiveF64::new(params.overlap_scaling_ratio).map_err(
            map_cugraph_error(
                "nexus_query_engine.graph_algorithm.force_atlas2_params",
                "invalid cuGraph ForceAtlas2 overlap_scaling_ratio",
            ),
        )?,
        edge_weight_influence: cugraph::params::NonNegativeF64::new(params.edge_weight_influence).map_err(
            map_cugraph_error(
                "nexus_query_engine.graph_algorithm.force_atlas2_params",
                "invalid cuGraph ForceAtlas2 edge_weight_influence",
            ),
        )?,
        jitter_tolerance: cugraph::params::NonNegativeF64::new(params.jitter_tolerance).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 jitter_tolerance",
        ))?,
        barnes_hut_optimize: params.barnes_hut_optimize,
        barnes_hut_theta: cugraph::params::ClosedUnitF64::new(params.barnes_hut_theta).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 barnes_hut_theta",
        ))?,
        scaling_ratio: cugraph::params::PositiveF64::new(params.scaling_ratio).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 scaling_ratio",
        ))?,
        strong_gravity_mode: params.strong_gravity_mode,
        gravity: cugraph::params::NonNegativeF64::new(params.gravity).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 gravity",
        ))?,
        verbose: params.verbose,
        do_expensive_check: false,
    })
}

pub(super) fn sssp_params(params: &crate::plan::GraphSsspParams) -> Result<cugraph::traversal::SsspParams> {
    let cutoff = params
        .cutoff
        .map(|value| {
            cugraph::params::NonNegativeF64::new(value).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.sssp_params",
                "invalid cuGraph SSSP cutoff",
            ))
        })
        .transpose()?;
    let mut p = cugraph::traversal::SsspParams::default().with_predecessors();
    if let Some(cutoff) = cutoff {
        p = p.with_cutoff(cutoff);
    }
    Ok(p)
}

pub(super) fn k_core_params(
    params: &crate::plan::GraphKCoreParams,
    directed: bool,
) -> Result<cugraph::structure::KCoreParams> {
    let k = std::num::NonZeroUsize::new(params.k as usize).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.k_core_params",
            "KCore k must be at least 1",
        )
        .with_fact("k", params.k)
    })?;
    Ok(cugraph::structure::KCoreParams {
        k,
        degree_type: degree_type_to_cugraph(params.degree_type, directed),
        do_expensive_check: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undirected_core_degree_selectors_use_one_symmetric_adjacency_side() {
        for degree_type in [
            crate::plan::GraphKCoreDegreeType::In,
            crate::plan::GraphKCoreDegreeType::Out,
            crate::plan::GraphKCoreDegreeType::InOut,
        ] {
            assert_eq!(degree_type_to_cugraph(degree_type, false), cugraph::structure::KCoreDegreeType::In);
        }
        assert_eq!(
            degree_type_to_cugraph(crate::plan::GraphKCoreDegreeType::InOut, true),
            cugraph::structure::KCoreDegreeType::InOut
        );
    }

    #[test]
    fn spectral_clustering_params_reject_libcugraph_int_overflow_before_binding() {
        let mut params = crate::plan::GraphSpectralClusteringParams {
            evs_max_iterations: i32::MAX as u32 + 1,
            ..Default::default()
        };
        let error = spectral_clustering_params(&params).expect_err("oversized evs iterations must be rejected");
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        let oversized_value = (i32::MAX as u32 + 1).to_string();
        assert_eq!(error.fact_value("evs_max_iterations"), Some(oversized_value.as_str()));

        params.evs_max_iterations = 500;
        params.k_means_max_iterations = i32::MAX as u32 + 1;
        let error = spectral_clustering_params(&params).expect_err("oversized k-means iterations must be rejected");
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("k_means_max_iterations"), Some(oversized_value.as_str()));
    }
}
