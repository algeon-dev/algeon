//! Integration tests for top-level algorithm parameter structs.
//!
//! Primitive wrapper invariants live in source-local unit tests. Keep this file
//! focused on cross-module defaults and a minimal "ergonomic to construct"
//! smoke for public params types.

use cugraph::{
    centrality::{
        EdgeBetweennessCentralityParams, EigenvectorCentralityParams, HitsParams, KatzCentralityParams, PageRankParams,
        PersonalizedPageRankParams,
    },
    community::{EcgParams, LeidenParams, LouvainParams, SpectralClusteringParams},
    layout::ForceAtlas2Params,
    params::{ClosedUnitF64, IterationCount, NonNegativeF64, PositiveF64},
    sampling::{Node2VecRandomWalkParams, RandomWalkParams},
    similarity::{AllPairsSimilarityParams, SimilarityParams},
};

#[test]
fn pagerank_params_default_has_expected_values() {
    let params = PageRankParams::default();
    assert_eq!(params.alpha.get(), 0.85);
    assert_eq!(params.epsilon.get(), 1e-5);
    assert_eq!(params.max_iterations.get(), 100);
}

#[test]
fn centrality_expansion_params_default_has_expected_values() {
    let ppr = PersonalizedPageRankParams::default();
    assert_eq!(ppr.alpha.get(), 0.85);
    assert_eq!(ppr.epsilon.get(), 1e-5);
    assert_eq!(ppr.max_iterations.get(), 100);
    assert!(!ppr.do_expensive_check);

    let eigenvector = EigenvectorCentralityParams::default();
    assert_eq!(eigenvector.epsilon.get(), 1e-6);
    assert_eq!(eigenvector.max_iterations.get(), 200);
    assert!(!eigenvector.do_expensive_check);

    let katz = KatzCentralityParams::default();
    assert_eq!(katz.alpha.get(), 0.01);
    assert_eq!(katz.beta.get(), 1.0);
    assert_eq!(katz.epsilon.get(), 1e-6);
    assert_eq!(katz.max_iterations.get(), 1000);
    assert!(!katz.do_expensive_check);

    let hits = HitsParams::default();
    assert_eq!(hits.epsilon.get(), 1e-5);
    assert_eq!(hits.max_iterations.get(), 100);
    assert!(!hits.normalize);
    assert!(!hits.do_expensive_check);

    let edge = EdgeBetweennessCentralityParams::default();
    assert!(edge.normalized);
    assert!(!edge.do_expensive_check);
}

#[test]
fn force_atlas2_params_default_has_expected_values() {
    let params = ForceAtlas2Params::default();
    assert_eq!(params.max_iter.get(), 500);
    assert!(!params.outbound_attraction_distribution);
    assert!(!params.lin_log_mode);
    assert!(!params.prevent_overlapping);
    assert_eq!(params.overlap_scaling_ratio.get(), 2.0);
    assert_eq!(params.edge_weight_influence.get(), 1.0);
    assert_eq!(params.jitter_tolerance.get(), 1.0);
    assert!(!params.barnes_hut_optimize);
    assert_eq!(params.barnes_hut_theta.get(), 0.5);
    assert_eq!(params.scaling_ratio.get(), 2.0);
    assert!(!params.strong_gravity_mode);
    assert_eq!(params.gravity.get(), 1.0);
    assert!(!params.verbose);
}

#[test]
fn louvain_params_default_has_expected_values() {
    let params = LouvainParams::default();
    assert_eq!(params.max_level.get(), 100);
    assert_eq!(params.threshold.get(), 1e-7);
    assert_eq!(params.resolution.get(), 1.0);
    assert!(!params.do_expensive_check);
}

#[test]
fn leiden_params_default_has_expected_values() {
    let params = LeidenParams::default();
    assert_eq!(params.max_level.get(), 100);
    assert_eq!(params.resolution.get(), 1.0);
    assert_eq!(params.theta.get(), 1.0);
    assert!(!params.do_expensive_check);
}

#[test]
fn ecg_params_default_has_expected_values() {
    let params = EcgParams::default();
    assert_eq!(params.min_weight.get(), 0.001);
    assert_eq!(params.ensemble_size.get(), 10);
    assert_eq!(params.max_level.get(), 100);
    assert_eq!(params.threshold.get(), 1e-7);
    assert_eq!(params.resolution.get(), 1.0);
    assert!(!params.do_expensive_check);
}

#[test]
fn spectral_clustering_params_default_has_expected_values() {
    let params = SpectralClusteringParams::default();
    assert_eq!(params.n_clusters.get(), 2);
    assert_eq!(params.n_eigenvectors.get(), 2);
    assert_eq!(params.evs_tolerance.get(), 0.01);
    assert_eq!(params.evs_max_iterations.get(), 500);
    assert_eq!(params.k_means_tolerance.get(), 0.001);
    assert_eq!(params.k_means_max_iterations.get(), 100);
    assert!(!params.do_expensive_check);
}

#[test]
fn random_walk_params_default_has_expected_values() {
    let params = RandomWalkParams::default();
    assert_eq!(params.max_length.get(), 1);

    let node2vec = Node2VecRandomWalkParams::default();
    assert_eq!(node2vec.max_length.get(), 1);
    assert_eq!(node2vec.p.get(), 1.0);
    assert_eq!(node2vec.q.get(), 1.0);
}

#[test]
fn similarity_params_default_has_expected_values() {
    let params = SimilarityParams::default();
    assert!(!params.use_weight);
    assert!(!params.do_expensive_check);

    let all_pairs = AllPairsSimilarityParams::default();
    assert!(!all_pairs.use_weight);
    assert_eq!(all_pairs.topk, None);
    assert!(!all_pairs.do_expensive_check);
}

#[test]
fn top_level_algorithm_params_are_ergonomic_to_construct() {
    let pagerank = PageRankParams {
        alpha: ClosedUnitF64::new(0.9).unwrap(),
        epsilon: NonNegativeF64::new(1e-6).unwrap(),
        max_iterations: IterationCount::new(200).unwrap(),
        do_expensive_check: false,
    };
    let force_atlas2 = ForceAtlas2Params {
        max_iter: IterationCount::new(100).unwrap(),
        prevent_overlapping: true,
        overlap_scaling_ratio: PositiveF64::new(3.0).unwrap(),
        barnes_hut_optimize: true,
        barnes_hut_theta: ClosedUnitF64::new(0.8).unwrap(),
        gravity: NonNegativeF64::new(5.0).unwrap(),
        edge_weight_influence: NonNegativeF64::new(0.0).unwrap(),
        ..Default::default()
    };
    let louvain = LouvainParams {
        max_level: IterationCount::new(75).unwrap(),
        threshold: NonNegativeF64::new(1e-6).unwrap(),
        resolution: PositiveF64::new(1.25).unwrap(),
        do_expensive_check: true,
    };

    assert_eq!(pagerank.alpha.get(), 0.9);
    assert_eq!(pagerank.max_iterations.get(), 200);
    assert_eq!(force_atlas2.max_iter.get(), 100);
    assert!(force_atlas2.prevent_overlapping);
    assert!(force_atlas2.barnes_hut_optimize);
    assert_eq!(louvain.max_level.get(), 75);
    assert!(louvain.do_expensive_check);
}
