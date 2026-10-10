#[cfg(feature = "cugraph")]
use cugraph::structure::KCoreDegreeType;
use cugraph::{
    centrality::{
        BetweennessCentralityParams, EdgeBetweennessCentralityParams, EigenvectorCentralityParams, HitsParams,
        KatzCentralityParams, PageRankParams, PersonalizedPageRankParams,
    },
    community::{EcgParams, LeidenParams, LouvainParams, SpectralClusteringParams},
    layout::ForceAtlas2Params,
    params::{
        BfsDepthLimit, ClosedUnitF64, ClusterCount, FractionF64, HalfOpenUnitF64, IterationCount, NonNegativeF64,
        PositiveF64,
    },
    similarity::SimilarityParams,
    structure::KCoreParams,
    traversal::{BfsParams, SsspParams},
};
#[cfg(feature = "cugraph")]
use query_planning::plan::GraphKCoreDegreeType;
use query_planning::plan::{
    GraphAlgorithmSpec, GraphBetweennessCentralityParams, GraphBfsParams, GraphEcgParams,
    GraphEdgeBetweennessCentralityParams, GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams,
    GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams, GraphPageRankParams,
    GraphSpectralClusteringParams, GraphSsspParams,
};

use super::{QueryError, QueryErrorCode, Result, map_cugraph_error};

pub(super) fn similarity_params(spec: &GraphAlgorithmSpec) -> SimilarityParams {
    SimilarityParams {
        // The graph definition owns whether a weight binding is meaningful for
        // this algorithm. A present binding is projected into graph
        // construction and must select the corresponding cuGraph metric path.
        use_weight: spec.edge_columns.weight.is_some(),
        // Pair endpoints are relation-backed and must be validated against the
        // constructed graph domain by libcugraph; skipping this turns invalid
        // endpoints into an implementation detail of the selected metric.
        do_expensive_check: true,
    }
}

pub(super) fn hits_params(params: &GraphHitsParams) -> Result<HitsParams> {
    Ok(HitsParams {
        epsilon: NonNegativeF64::new(params.epsilon)
            .map_err(map_cugraph_error("query_engine.graph_algorithm.hits_params", "invalid cuGraph HITS epsilon"))?,
        max_iterations: IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.hits_params",
            "invalid cuGraph HITS max_iterations",
        ))?,
        normalize: params.normalize,
        do_expensive_check: false,
    })
}

pub(super) fn betweenness_centrality_params(params: &GraphBetweennessCentralityParams) -> BetweennessCentralityParams {
    BetweennessCentralityParams {
        normalized: params.normalized,
        include_endpoints: params.include_endpoints,
        do_expensive_check: false,
    }
}

pub(super) fn edge_betweenness_centrality_params(
    params: &GraphEdgeBetweennessCentralityParams,
) -> EdgeBetweennessCentralityParams {
    EdgeBetweennessCentralityParams { normalized: params.normalized, do_expensive_check: false }
}

pub(super) fn katz_centrality_params(params: &GraphKatzCentralityParams) -> Result<KatzCentralityParams> {
    Ok(KatzCentralityParams {
        alpha: NonNegativeF64::new(params.alpha).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality alpha",
        ))?,
        beta: NonNegativeF64::new(params.beta).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality beta",
        ))?,
        epsilon: NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality epsilon",
        ))?,
        max_iterations: IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn eigenvector_centrality_params(
    params: &GraphEigenvectorCentralityParams,
) -> Result<EigenvectorCentralityParams> {
    Ok(EigenvectorCentralityParams {
        epsilon: NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.eigenvector_centrality_params",
            "invalid cuGraph eigenvector centrality epsilon",
        ))?,
        max_iterations: IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.eigenvector_centrality_params",
            "invalid cuGraph eigenvector centrality max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn pagerank_params(params: &GraphPageRankParams) -> Result<PageRankParams> {
    Ok(PageRankParams {
        alpha: ClosedUnitF64::new(params.alpha).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank alpha",
        ))?,
        epsilon: NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank epsilon",
        ))?,
        max_iterations: IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn personalized_pagerank_params(params: &GraphPageRankParams) -> Result<PersonalizedPageRankParams> {
    Ok(PersonalizedPageRankParams {
        alpha: ClosedUnitF64::new(params.alpha).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.personalized_pagerank_params",
            "invalid cuGraph personalized PageRank alpha",
        ))?,
        epsilon: NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.personalized_pagerank_params",
            "invalid cuGraph personalized PageRank epsilon",
        ))?,
        max_iterations: IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.personalized_pagerank_params",
            "invalid cuGraph personalized PageRank max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn bfs_params(params: &GraphBfsParams) -> Result<BfsParams> {
    let depth_limit = params
        .depth_limit
        .map(|value| {
            usize::try_from(value).map(BfsDepthLimit::new).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.graph_algorithm.bfs_params",
                    "BFS depth_limit must be non-negative",
                )
                .with_fact("depth_limit", value)
            })
        })
        .transpose()?;
    Ok(BfsParams { depth_limit, compute_predecessors: true, do_expensive_check: false })
}

pub(super) fn louvain_params(params: &GraphLouvainParams) -> Result<LouvainParams> {
    Ok(LouvainParams {
        max_level: IterationCount::new(params.max_level).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.louvain_params",
            "invalid cuGraph Louvain max_level",
        ))?,
        threshold: NonNegativeF64::new(params.threshold).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.louvain_params",
            "invalid cuGraph Louvain threshold",
        ))?,
        resolution: PositiveF64::new(params.resolution).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.louvain_params",
            "invalid cuGraph Louvain resolution",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn leiden_params(params: &GraphLeidenParams) -> Result<LeidenParams> {
    Ok(LeidenParams {
        max_level: IterationCount::new(params.max_level).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.leiden_params",
            "invalid cuGraph Leiden max_level",
        ))?,
        resolution: PositiveF64::new(params.resolution).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.leiden_params",
            "invalid cuGraph Leiden resolution",
        ))?,
        do_expensive_check: false,
    })
}

pub(super) fn ecg_params(params: &GraphEcgParams) -> Result<EcgParams> {
    Ok(EcgParams {
        min_weight: NonNegativeF64::new(params.min_weight)
            .map_err(map_cugraph_error("query_engine.graph_algorithm.ecg_params", "invalid cuGraph ECG min_weight"))?,
        ensemble_size: IterationCount::new(params.ensemble_size).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG ensemble_size",
        ))?,
        max_level: IterationCount::new(params.max_level)
            .map_err(map_cugraph_error("query_engine.graph_algorithm.ecg_params", "invalid cuGraph ECG max_level"))?,
        threshold: FractionF64::new(params.threshold)
            .map_err(map_cugraph_error("query_engine.graph_algorithm.ecg_params", "invalid cuGraph ECG threshold"))?,
        resolution: FractionF64::new(params.resolution)
            .map_err(map_cugraph_error("query_engine.graph_algorithm.ecg_params", "invalid cuGraph ECG resolution"))?,
        do_expensive_check: false,
    })
}

pub(super) fn spectral_clustering_params(params: &GraphSpectralClusteringParams) -> Result<SpectralClusteringParams> {
    const OPERATION: &str = "query_engine.graph_algorithm.spectral_clustering_params";
    for (name, value) in [("n_clusters", params.n_clusters), ("n_eigenvectors", params.n_eigenvectors)] {
        if value > i32::MAX as u32 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "cuGraph spectral clustering counts must fit the libcugraph int32 vertex type",
            )
            .with_fact(name, value));
        }
    }
    let n_eigenvectors = std::num::NonZeroUsize::new(params.n_eigenvectors as usize).ok_or_else(|| {
        QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, "spectral clustering n_eigenvectors must be at least 1")
            .with_fact("n_eigenvectors", params.n_eigenvectors)
    })?;
    Ok(SpectralClusteringParams {
        n_clusters: ClusterCount::new(params.n_clusters as usize)
            .map_err(map_cugraph_error(OPERATION, "invalid cuGraph spectral clustering n_clusters"))?,
        n_eigenvectors,
        evs_tolerance: HalfOpenUnitF64::new(params.evs_tolerance)
            .map_err(map_cugraph_error(OPERATION, "invalid cuGraph spectral clustering evs_tolerance"))?,
        do_expensive_check: false,
    })
}

pub(super) fn force_atlas2_params(params: &GraphForceAtlas2Params) -> Result<ForceAtlas2Params> {
    if params.max_iter > i32::MAX as u32 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.force_atlas2_params",
            "cuGraph ForceAtlas2 max_iter must fit the libcugraph int32 iteration count",
        )
        .with_fact("max_iter", params.max_iter)
        .with_fact("max_supported", i32::MAX));
    }
    Ok(ForceAtlas2Params {
        max_iter: IterationCount::new(params.max_iter).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 max_iter",
        ))?,
        outbound_attraction_distribution: params.outbound_attraction_distribution,
        lin_log_mode: params.lin_log_mode,
        prevent_overlapping: params.prevent_overlapping,
        overlap_scaling_ratio: PositiveF64::new(params.overlap_scaling_ratio).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 overlap_scaling_ratio",
        ))?,
        edge_weight_influence: NonNegativeF64::new(params.edge_weight_influence).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 edge_weight_influence",
        ))?,
        jitter_tolerance: NonNegativeF64::new(params.jitter_tolerance).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 jitter_tolerance",
        ))?,
        barnes_hut_optimize: params.barnes_hut_optimize,
        barnes_hut_theta: ClosedUnitF64::new(params.barnes_hut_theta).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 barnes_hut_theta",
        ))?,
        scaling_ratio: PositiveF64::new(params.scaling_ratio).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 scaling_ratio",
        ))?,
        strong_gravity_mode: params.strong_gravity_mode,
        gravity: NonNegativeF64::new(params.gravity).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.force_atlas2_params",
            "invalid cuGraph ForceAtlas2 gravity",
        ))?,
        verbose: params.verbose,
        do_expensive_check: params.vertex_attributes.is_some() || params.initial_positions.is_some(),
    })
}

pub(super) fn sssp_params(params: &GraphSsspParams) -> Result<SsspParams> {
    let cutoff = params
        .cutoff
        .map(|value| {
            NonNegativeF64::new(value)
                .map_err(map_cugraph_error("query_engine.graph_algorithm.sssp_params", "invalid cuGraph SSSP cutoff"))
        })
        .transpose()?;
    let mut p = SsspParams::default().with_predecessors();
    if let Some(cutoff) = cutoff {
        p = p.with_cutoff(cutoff);
    }
    Ok(p)
}

pub(super) fn k_core_params(params: &GraphKCoreParams, directed: bool) -> Result<KCoreParams> {
    let k = std::num::NonZeroUsize::new(params.k as usize).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.k_core_params",
            "KCore k must be at least 1",
        )
        .with_fact("k", params.k)
    })?;
    Ok(KCoreParams { k, degree_type: degree_type_to_cugraph(params.degree_type, directed), do_expensive_check: false })
}

#[cfg(feature = "cugraph")]
pub(super) fn degree_type_to_cugraph(degree_type: GraphKCoreDegreeType, directed: bool) -> KCoreDegreeType {
    // libcugraph's INOUT selector sums incoming and outgoing adjacency. The
    // Python cuGraph surface treats all three selectors as the same degree on
    // an undirected graph, whose physical edge list is already symmetric.
    // Passing INOUT for that representation would therefore double every
    // degree and change core membership. Use one side for undirected graphs;
    // incoming and outgoing are equivalent there.
    if !directed {
        return KCoreDegreeType::In;
    }
    match degree_type {
        GraphKCoreDegreeType::In => KCoreDegreeType::In,
        GraphKCoreDegreeType::Out => KCoreDegreeType::Out,
        GraphKCoreDegreeType::InOut => KCoreDegreeType::InOut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undirected_core_degree_selectors_use_one_symmetric_adjacency_side() {
        for degree_type in [GraphKCoreDegreeType::In, GraphKCoreDegreeType::Out, GraphKCoreDegreeType::InOut] {
            assert_eq!(degree_type_to_cugraph(degree_type, false), KCoreDegreeType::In);
        }
        assert_eq!(degree_type_to_cugraph(GraphKCoreDegreeType::InOut, true), KCoreDegreeType::InOut);
    }

    /// `depth_limit` is the number of hops to expand: 0 reaches only the
    /// sources, an absent limit is unlimited, and a negative limit is invalid.
    #[test]
    fn bfs_params_keep_zero_as_a_hop_count_and_reject_negative_limits() {
        let limit = |depth_limit| bfs_params(&GraphBfsParams::new(0, depth_limit)).map(|params| params.depth_limit);
        assert_eq!(limit(None).unwrap(), None);
        assert_eq!(limit(Some(0)).unwrap(), Some(BfsDepthLimit::new(0)));
        assert_eq!(limit(Some(3)).unwrap(), Some(BfsDepthLimit::new(3)));
        assert_eq!(limit(Some(-1)).unwrap_err().code(), QueryErrorCode::InvalidPlan);
    }

    /// libcugraph's ECG accepts threshold and resolution only in (0, 1]
    /// (`ecg_impl.cuh`); anything else is a masked native failure on the GPU.
    #[test]
    fn ecg_params_reject_threshold_and_resolution_outside_the_native_unit_interval() {
        let at_upper_bound = GraphEcgParams { threshold: 1.0, resolution: 1.0, ..Default::default() };
        assert!(ecg_params(&at_upper_bound).is_ok());

        for (threshold, resolution) in [(0.0, 1.0), (1.5, 1.0), (1.0e-7, 0.0), (1.0e-7, 3.0)] {
            let params = GraphEcgParams { threshold, resolution, ..Default::default() };
            assert!(ecg_params(&params).is_err(), "threshold {threshold} resolution {resolution}");
        }
    }

    /// libcugraph's spectral clustering requires at least two clusters and an
    /// eigensolver tolerance in [0, 1) (`spectral_clustering.cu`), and casts the
    /// counts to its int32 vertex type; anything else is a masked native failure or a
    /// wrapped value on the GPU.
    #[test]
    fn spectral_clustering_params_reject_values_outside_the_native_ranges() {
        let at_bounds = GraphSpectralClusteringParams {
            n_clusters: 2,
            n_eigenvectors: i32::MAX as u32,
            evs_tolerance: 0.0,
            ..Default::default()
        };
        assert!(spectral_clustering_params(&at_bounds).is_ok());

        let rejected = [
            GraphSpectralClusteringParams { n_clusters: 1, ..Default::default() },
            GraphSpectralClusteringParams { n_clusters: i32::MAX as u32 + 1, ..Default::default() },
            GraphSpectralClusteringParams { n_eigenvectors: 0, ..Default::default() },
            GraphSpectralClusteringParams { evs_tolerance: 1.0, ..Default::default() },
        ];
        for params in rejected {
            assert!(spectral_clustering_params(&params).is_err(), "{params:?}");
        }
    }
}
