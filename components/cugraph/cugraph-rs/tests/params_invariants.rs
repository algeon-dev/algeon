//! Integration tests for top-level algorithm parameter structs.
//!
//! Primitive wrapper invariants live in source-local unit tests. Keep this file
//! focused on cross-module defaults and a minimal "ergonomic to construct"
//! smoke for public params types.

use cugraph::{
    centrality::PageRankParams,
    layout::ForceAtlas2Params,
    params::{ClosedUnitF64, IterationCount, NonNegativeF64, PositiveF64},
};

#[test]
fn pagerank_params_default_has_expected_values() {
    let params = PageRankParams::default();
    assert_eq!(params.alpha.get(), 0.85);
    assert_eq!(params.epsilon.get(), 1e-5);
    assert_eq!(params.max_iterations.get(), 100);
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

    assert_eq!(pagerank.alpha.get(), 0.9);
    assert_eq!(pagerank.max_iterations.get(), 200);
    assert_eq!(force_atlas2.max_iter.get(), 100);
    assert!(force_atlas2.prevent_overlapping);
    assert!(force_atlas2.barnes_hut_optimize);
}
