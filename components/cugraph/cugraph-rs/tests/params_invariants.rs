//! Integration tests verifying that the type-level parameter invariants (CX-1)
//! are enforced correctly and that migrated APIs remain compatible.

use cugraph::centrality::PageRankParams;
use cugraph::layout::ForceAtlas2Params;
use cugraph::params::{
    BfsDepthLimit, ClosedUnitF64, IterationCount, NonNegativeF64, PositiveF64, UnitF64,
};

// ---------------------------------------------------------------------------
// PageRankParams compatibility
// ---------------------------------------------------------------------------

#[test]
fn pagerank_params_default_has_expected_values() {
    let params = PageRankParams::default();
    assert_eq!(params.alpha.get(), 0.85);
    assert_eq!(params.epsilon.get(), 1e-5);
    assert_eq!(params.max_iterations.get(), 100);
}

#[test]
fn pagerank_params_custom_construction() {
    let params = PageRankParams {
        alpha: UnitF64::new(0.9).unwrap(),
        epsilon: PositiveF64::new(1e-6).unwrap(),
        max_iterations: IterationCount::new(200).unwrap(),
    };
    assert_eq!(params.alpha.get(), 0.9);
    assert_eq!(params.epsilon.get(), 1e-6);
    assert_eq!(params.max_iterations.get(), 200);
}

#[test]
fn pagerank_params_alpha_rejects_invalid() {
    // 0.0 is not in (0, 1)
    assert!(UnitF64::new(0.0).is_err());
    // 1.0 is not in (0, 1)
    assert!(UnitF64::new(1.0).is_err());
    // negative
    assert!(UnitF64::new(-0.5).is_err());
    // > 1
    assert!(UnitF64::new(1.5).is_err());
}

#[test]
fn pagerank_params_epsilon_rejects_invalid() {
    assert!(PositiveF64::new(0.0).is_err());
    assert!(PositiveF64::new(-1e-5).is_err());
    assert!(PositiveF64::new(f64::NAN).is_err());
    assert!(PositiveF64::new(f64::INFINITY).is_err());
}

#[test]
fn pagerank_params_max_iterations_rejects_zero() {
    assert!(IterationCount::new(0).is_err());
}

// ---------------------------------------------------------------------------
// ForceAtlas2Params compatibility
// ---------------------------------------------------------------------------

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
fn force_atlas2_params_custom_override() {
    let params = ForceAtlas2Params {
        max_iter: IterationCount::new(100).unwrap(),
        barnes_hut_optimize: true,
        barnes_hut_theta: ClosedUnitF64::new(0.8).unwrap(),
        gravity: PositiveF64::new(5.0).unwrap(),
        edge_weight_influence: NonNegativeF64::new(0.0).unwrap(),
        ..Default::default()
    };
    assert_eq!(params.max_iter.get(), 100);
    assert!(params.barnes_hut_optimize);
    assert_eq!(params.barnes_hut_theta.get(), 0.8);
    assert_eq!(params.gravity.get(), 5.0);
    assert_eq!(params.edge_weight_influence.get(), 0.0);
}

#[test]
fn force_atlas2_params_rejects_zero_max_iter() {
    assert!(IterationCount::new(0).is_err());
}

#[test]
fn force_atlas2_barnes_hut_theta_closed_unit_boundaries() {
    // [0.0, 1.0] is valid for ClosedUnitF64
    assert!(ClosedUnitF64::new(0.0).is_ok());
    assert!(ClosedUnitF64::new(1.0).is_ok());
    assert!(ClosedUnitF64::new(-0.01).is_err());
    assert!(ClosedUnitF64::new(1.01).is_err());
}

#[test]
fn force_atlas2_edge_weight_influence_allows_zero() {
    // edge_weight_influence = 0 means "no influence", which is valid
    assert!(NonNegativeF64::new(0.0).is_ok());
    assert!(NonNegativeF64::new(-0.001).is_err());
}

// ---------------------------------------------------------------------------
// Conversion round-trips
// ---------------------------------------------------------------------------

#[test]
fn iteration_count_u32_roundtrip() {
    let count = IterationCount::new(42).unwrap();
    let raw: u32 = count.into();
    let back = IterationCount::new(raw).unwrap();
    assert_eq!(count, back);
}

#[test]
fn iteration_count_i32_roundtrip() {
    let count = IterationCount::new(100).unwrap();
    let raw: i32 = count.into();
    let back = IterationCount::try_from(raw).unwrap();
    assert_eq!(count, back);
}

#[test]
fn positive_f64_f64_roundtrip() {
    let val = PositiveF64::new(3.14).unwrap();
    let raw: f64 = val.into();
    let back = PositiveF64::new(raw).unwrap();
    assert_eq!(val, back);
}

#[test]
fn unit_f64_f64_roundtrip() {
    let val = UnitF64::new(0.85).unwrap();
    let raw: f64 = val.into();
    let back = UnitF64::new(raw).unwrap();
    assert_eq!(val, back);
}

#[test]
fn bfs_depth_limit_rejects_zero() {
    assert!(BfsDepthLimit::new(0).is_err());
}

#[test]
fn bfs_depth_limit_accepts_positive_values() {
    assert_eq!(BfsDepthLimit::new(1).unwrap().get(), 1);
    assert_eq!(BfsDepthLimit::new(10).unwrap().get(), 10);
}

#[test]
fn bfs_depth_limit_try_from_i32_and_usize() {
    let a: BfsDepthLimit = 7i32.try_into().unwrap();
    assert_eq!(a.get(), 7);

    let b: BfsDepthLimit = 9usize.try_into().unwrap();
    assert_eq!(b.get(), 9);

    assert!(BfsDepthLimit::try_from(0i32).is_err());
    assert!(BfsDepthLimit::try_from(-1i32).is_err());
    assert!(BfsDepthLimit::try_from(0usize).is_err());
}

// ---------------------------------------------------------------------------
// Params are Copy + Clone + Debug
// ---------------------------------------------------------------------------

#[test]
fn pagerank_params_is_copy() {
    let params = PageRankParams::default();
    let copy = params;
    // If PageRankParams is Copy, both should be usable:
    assert_eq!(params.alpha.get(), copy.alpha.get());
}

#[test]
fn force_atlas2_params_is_copy() {
    let params = ForceAtlas2Params::default();
    let copy = params;
    assert_eq!(params.max_iter.get(), copy.max_iter.get());
}

#[test]
fn params_are_debug() {
    let pr = PageRankParams::default();
    let _s = format!("{:?}", pr);

    let fa2 = ForceAtlas2Params::default();
    let _s = format!("{:?}", fa2);
}
