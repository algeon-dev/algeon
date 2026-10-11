use cugraph::{
    array::DeviceArray,
    centrality::{InitialGuess, PageRankParams, PrecomputedWeights},
    error::ErrorKind,
    graph::{GraphBuilder, Transposed},
};

use crate::test_handle;

pub(super) fn test_pagerank_star_graph() {
    let handle = test_handle();
    let handle = handle.enter("test_pagerank_star_graph").expect("entered");

    // Star graph: 1->0, 2->0, 3->0 (node 0 is the hub)
    let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
    let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

    let params = PageRankParams::default();
    let result = graph.pagerank(&handle, &params).expect("pagerank");

    assert!(result.converged().expect("converged"), "PageRank should converge");
    assert!(result.num_iterations() > 0, "should do at least 1 iteration");

    let vertices_view = result.vertex_ids().expect("vertex_ids");
    let values_view = result.scores().expect("scores");

    assert_eq!(vertices_view.len(), 4);
    assert_eq!(values_view.len(), 4);

    // Copy results to host and verify hub (node 0) has the highest rank
    let vertices: Vec<i32> = vertices_view.to_vec_i32(&handle).expect("vert copy");
    let values: Vec<f32> = values_view.to_vec_f32(&handle).expect("val copy");

    // Find the PageRank of vertex 0 (hub); should be highest
    let hub_idx = vertices.iter().position(|&v| v == 0).expect("hub vertex");
    let hub_rank = values[hub_idx];

    for (i, &v) in vertices.iter().enumerate() {
        if v != 0 {
            assert!(hub_rank > values[i], "hub rank {} should be > leaf rank {} (vertex {})", hub_rank, values[i], v);
        }
    }
}

pub(super) fn test_pagerank_rejects_mismatched_precomputed_weights() {
    let handle = test_handle();
    let handle = handle.enter("test_pagerank_rejects_mismatched_precomputed_weights").expect("entered");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

    let verts = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("verts");
    let sums = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("sums"); // wrong length
    let verts_view = verts.view().expect("verts_view");
    let sums_view = sums.view().expect("sums_view");

    let pw = PrecomputedWeights { vertices: &verts_view, sums: &sums_view };
    let params = PageRankParams::default();

    let err = graph.pagerank_with(&handle, &params, Some(&pw), None).expect_err("should reject mismatched lengths");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("precomputed weights"));
}

pub(super) fn test_pagerank_rejects_mismatched_initial_guess() {
    let handle = test_handle();
    let handle = handle.enter("test_pagerank_rejects_mismatched_initial_guess").expect("entered");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

    let verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("verts");
    let vals = DeviceArray::from_slice(&handle, &[0.5f32]).expect("vals"); // wrong length
    let verts_view = verts.view().expect("verts_view");
    let vals_view = vals.view().expect("vals_view");

    let ig = InitialGuess { vertices: &verts_view, values: &vals_view };
    let params = PageRankParams::default();

    let err = graph.pagerank_with(&handle, &params, None, Some(&ig)).expect_err("should reject mismatched lengths");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("initial guess"));
}
