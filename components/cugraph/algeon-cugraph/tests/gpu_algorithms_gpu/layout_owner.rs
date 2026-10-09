use cudf::testing::run_named_cases;
use cugraph::{
    array::DeviceArray,
    error::ErrorKind,
    graph::{Forward, GraphBuilder},
    layout::{ForceAtlas2Params, ForceAtlas2VertexAttrs, InitialPositions},
    params::{ClosedUnitF64, IterationCount, NonNegativeF64},
    random::RngState,
    structure::ScoreValues,
    types::GraphProperties,
};

use crate::test_handle;

const TEST_LAYOUT_SECONDARY_SEED: u64 = 7;
const TEST_LAYOUT_SMOKE_ITERATIONS: u32 = 10;
const TEST_LAYOUT_SETTLE_ITERATIONS: u32 = 50;

#[test]
fn gpu_cases_1() {
    run_named_cases(&[
        (
            "layout::force_atlas2::tests::test_force_atlas2_refinement_with_initial_positions",
            test_force_atlas2_refinement_with_initial_positions,
        ),
        ("layout::force_atlas2::tests::test_force_atlas2_with_weights", test_force_atlas2_with_weights),
        ("layout::force_atlas2::tests::test_force_atlas2_barnes_hut_mode", test_force_atlas2_barnes_hut_mode),
        (
            "layout::force_atlas2::tests::test_force_atlas2_rejects_mismatched_initial_positions",
            test_force_atlas2_rejects_mismatched_initial_positions,
        ),
        (
            "layout::force_atlas2::tests::test_force_atlas2_rejects_mismatched_vertex_attrs",
            test_force_atlas2_rejects_mismatched_vertex_attrs,
        ),
        (
            "layout::force_atlas2::tests::test_force_atlas2_rejects_initial_positions_for_i64_graphs",
            test_force_atlas2_rejects_initial_positions_for_i64_graphs,
        ),
        (
            "layout::force_atlas2::tests::test_force_atlas2_accepts_f32_vertex_attrs_for_f64_graphs",
            test_force_atlas2_accepts_f32_vertex_attrs_for_f64_graphs,
        ),
        (
            "layout::force_atlas2::tests::test_force_atlas2_lin_log_and_strong_gravity",
            test_force_atlas2_lin_log_and_strong_gravity,
        ),
    ]);
}

fn test_force_atlas2_refinement_with_initial_positions() {
    // Two-pass layout: first pass with random positions, second with initial positions.
    let handle = test_handle();
    let entered = handle.enter("test_force_atlas2_refinement_with_initial_positions").expect("entered");
    let mut rng = RngState::new(&entered, 123).expect("rng");

    // Square: 0-1-2-3-0
    let src_data: Vec<i32> = vec![0, 1, 2, 3, 1, 2, 3, 0];
    let dst_data: Vec<i32> = vec![1, 2, 3, 0, 0, 1, 2, 3];

    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    // First pass: 50 iterations, random initial positions
    let params1 = ForceAtlas2Params {
        max_iter: IterationCount::new(TEST_LAYOUT_SETTLE_ITERATIONS).unwrap(),
        ..Default::default()
    };
    let result1 = graph.force_atlas2(&entered, &mut rng, &params1).expect("fa2 pass 1");

    let vids1 = result1.vertex_ids().expect("vertex_ids");
    assert_eq!(vids1.len(), 4);

    // Second pass: 10 iterations, using first pass result as initial positions
    let vids1_view = vids1.as_i32().expect("i32 vertex_ids");
    let x1_scores = result1.x().expect("x");
    let x1_view = x1_scores.as_f32().expect("f32 x");
    let y1_scores = result1.y().expect("y");
    let y1_view = y1_scores.as_f32().expect("f32 y");
    let init_pos = InitialPositions { vertices: vids1_view, x: x1_view, y: y1_view };
    let params2 = ForceAtlas2Params {
        max_iter: IterationCount::new(TEST_LAYOUT_SMOKE_ITERATIONS).unwrap(),
        ..Default::default()
    };
    let result2 = graph.force_atlas2_with(&entered, &mut rng, &params2, None, Some(&init_pos)).expect("fa2 pass 2");

    assert_eq!(result2.vertex_ids().expect("vertex_ids").len(), 4);
    assert_eq!(result2.x().expect("x").len(), 4);
    assert_eq!(result2.y().expect("y").len(), 4);

    // Verify coordinates are finite
    let x_vals: Vec<f32> = result2.x().expect("x").to_vec_f32(&entered).expect("x copy");
    let y_vals: Vec<f32> = result2.y().expect("y").to_vec_f32(&entered).expect("y copy");

    for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
        assert!(x.is_finite(), "x[{}] = {} should be finite", i, x);
        assert!(y.is_finite(), "y[{}] = {} should be finite", i, y);
    }
}

fn test_force_atlas2_with_weights() {
    // Test FA2 with edge weights.
    let handle = test_handle();
    let handle = handle.enter("test_force_atlas2_with_weights").expect("entered");
    let mut rng = RngState::new(&handle, TEST_LAYOUT_SECONDARY_SEED).expect("rng");

    let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
    let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];
    let wgt_data: Vec<f32> = vec![1.0, 2.0, 0.5, 1.0, 2.0, 0.5];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &wgt_data).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .weights(&wgt_view)
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = ForceAtlas2Params {
        max_iter: IterationCount::new(TEST_LAYOUT_SETTLE_ITERATIONS).unwrap(),
        edge_weight_influence: NonNegativeF64::new(1.0).unwrap(),
        ..Default::default()
    };

    let result = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2");

    assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 3);
    assert_eq!(result.x().expect("x").len(), 3);
    assert_eq!(result.y().expect("y").len(), 3);
}

fn test_force_atlas2_barnes_hut_mode() {
    // Test Barnes-Hut approximation mode.
    let handle = test_handle();
    let handle = handle.enter("test_force_atlas2_barnes_hut_mode").expect("entered");
    let mut rng = RngState::new(&handle, 55).expect("rng");

    let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
    let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = ForceAtlas2Params {
        max_iter: IterationCount::new(TEST_LAYOUT_SETTLE_ITERATIONS).unwrap(),
        barnes_hut_optimize: true,
        barnes_hut_theta: ClosedUnitF64::new(0.5).unwrap(),
        ..Default::default()
    };

    let result = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2 BH");

    assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 3);

    let x_vals: Vec<f32> = result.x().expect("x").to_vec_f32(&handle).expect("x copy");
    let y_vals: Vec<f32> = result.y().expect("y").to_vec_f32(&handle).expect("y copy");

    for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
        assert!(x.is_finite(), "BH x[{}] = {} should be finite", i, x);
        assert!(y.is_finite(), "BH y[{}] = {} should be finite", i, y);
    }
}

fn test_force_atlas2_rejects_mismatched_initial_positions() {
    let handle = test_handle();
    let handle = handle.enter("test_force_atlas2_rejects_mismatched_initial_positions").expect("entered");
    let mut rng = RngState::new(&handle, 0).expect("rng");

    let src_data: Vec<i32> = vec![0, 1, 1, 0];
    let dst_data: Vec<i32> = vec![1, 0, 0, 1];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    // Create mismatched initial position arrays
    let verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("verts");
    let x_pos = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("x");
    let y_pos = DeviceArray::from_slice(&handle, &[3.0f32]).expect("y"); // wrong length
    let verts_view = verts.view().expect("verts_view");
    let x_pos_view = x_pos.view().expect("x_pos_view");
    let y_pos_view = y_pos.view().expect("y_pos_view");

    let init = InitialPositions { vertices: &verts_view, x: &x_pos_view, y: &y_pos_view };
    let params = ForceAtlas2Params::default();

    let err = graph
        .force_atlas2_with(&handle, &mut rng, &params, None, Some(&init))
        .expect_err("should reject mismatched y length");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("y length"));
}

fn test_force_atlas2_rejects_mismatched_vertex_attrs() {
    let handle = test_handle();
    let handle = handle.enter("test_force_atlas2_rejects_mismatched_vertex_attrs").expect("entered");
    let mut rng = RngState::new(&handle, 0).expect("rng");

    let src_data: Vec<i32> = vec![0, 1, 1, 0];
    let dst_data: Vec<i32> = vec![1, 0, 0, 1];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    // Create mismatched mass arrays
    let mass_verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("mass_verts");
    let mass_vals = DeviceArray::from_slice(&handle, &[1.0f32]).expect("mass_vals"); // wrong length
    let mass_verts_view = mass_verts.view().expect("mass_verts_view");
    let mass_vals_view = mass_vals.view().expect("mass_vals_view");

    let attrs = ForceAtlas2VertexAttrs { mass: Some((&mass_verts_view, &mass_vals_view)), ..Default::default() };
    let params = ForceAtlas2Params::default();

    let err = graph
        .force_atlas2_with(&handle, &mut rng, &params, Some(&attrs), None)
        .expect_err("should reject mismatched mass array lengths");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("mass"));
}

fn test_force_atlas2_rejects_initial_positions_for_i64_graphs() {
    let handle = test_handle();
    let handle = handle.enter("test_force_atlas2_rejects_initial_positions_for_i64_graphs").expect("entered");
    let mut rng = RngState::new(&handle, 0).expect("rng");

    let src = DeviceArray::from_slice(&handle, &[0i64, 1, 1, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i64, 0, 0, 1]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("verts");
    let x_pos = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("x");
    let y_pos = DeviceArray::from_slice(&handle, &[3.0f32, 4.0]).expect("y");
    let verts_view = verts.view().expect("verts_view");
    let x_pos_view = x_pos.view().expect("x_pos_view");
    let y_pos_view = y_pos.view().expect("y_pos_view");

    let init = InitialPositions { vertices: &verts_view, x: &x_pos_view, y: &y_pos_view };
    let params = ForceAtlas2Params::default();

    let err = graph
        .force_atlas2_with(&handle, &mut rng, &params, None, Some(&init))
        .expect_err("should reject i64 graphs with i32 initial position vertices");
    assert_eq!(err.kind(), ErrorKind::UnsupportedType);
    assert!(err.message().contains("initial positions vertex dtype Int32 does not match graph vertex dtype Int64"));
    assert!(err.message().contains("Int64"));
}

fn test_force_atlas2_accepts_f32_vertex_attrs_for_f64_graphs() {
    let handle = test_handle();
    let handle = handle.enter("test_force_atlas2_accepts_f32_vertex_attrs_for_f64_graphs").expect("entered");
    let mut rng = RngState::new(&handle, 0).expect("rng");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 1, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 0, 0, 1]).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f64, 1.0, 1.0, 1.0]).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .weights(&wgt_view)
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let mass_verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("mass_verts");
    let mass_vals = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("mass_vals");
    let mass_verts_view = mass_verts.view().expect("mass_verts_view");
    let mass_vals_view = mass_vals.view().expect("mass_vals_view");

    let attrs = ForceAtlas2VertexAttrs { mass: Some((&mass_verts_view, &mass_vals_view)), ..Default::default() };
    let params = ForceAtlas2Params::default();

    let result = graph.force_atlas2_with(&handle, &mut rng, &params, Some(&attrs), None).expect("force_atlas2");
    assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 2);
    assert!(matches!(result.x().expect("x"), ScoreValues::F32(_)));
    assert!(matches!(result.y().expect("y"), ScoreValues::F32(_)));
}

fn test_force_atlas2_lin_log_and_strong_gravity() {
    // Test with lin_log_mode and strong_gravity_mode enabled.
    let handle = test_handle();
    let handle = handle.enter("test_force_atlas2_lin_log_and_strong_gravity").expect("entered");
    let mut rng = RngState::new(&handle, 77).expect("rng");

    let src_data: Vec<i32> = vec![0, 1, 2, 3, 1, 2, 3, 0];
    let dst_data: Vec<i32> = vec![1, 2, 3, 0, 0, 1, 2, 3];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = ForceAtlas2Params {
        max_iter: IterationCount::new(30).unwrap(),
        lin_log_mode: true,
        strong_gravity_mode: true,
        gravity: NonNegativeF64::new(2.0).unwrap(),
        ..Default::default()
    };

    let result = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2");

    assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 4);
    assert_eq!(result.x().expect("x").len(), 4);
    assert_eq!(result.y().expect("y").len(), 4);
}
