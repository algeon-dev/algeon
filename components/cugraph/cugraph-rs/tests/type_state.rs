//! Tests for type-state encoding of `store_transposed` on [`Graph`].
//!
//! The `Graph<T>` type parameter encodes whether the graph is stored in
//! transposed (`Graph<Transposed>`) or forward (`Graph<Forward>`) form.
//! Algorithms that require a specific layout accept only the correct variant,
//! catching mismatches at compile time.

use cugraph::graph::{Forward, Graph, Transposed};
use static_assertions::assert_not_impl_any;

// ---------------------------------------------------------------------------
// Marker-type assertions
// ---------------------------------------------------------------------------

#[test]
fn graph_forward_is_not_send_or_sync() {
    assert_not_impl_any!(Graph<Forward>: Send, Sync);
}

#[test]
fn graph_transposed_is_not_send_or_sync() {
    assert_not_impl_any!(Graph<Transposed>: Send, Sync);
}

#[test]
fn graph_default_type_param_is_forward() {
    // `Graph` (no explicit parameter) must be the same as `Graph<Forward>`.
    fn accepts_forward(_g: &Graph<Forward>) {}
    fn returns_default() -> Option<Graph> {
        None
    }
    // If this compiles, the default type parameter is `Forward`.
    if let Some(ref g) = returns_default() {
        accepts_forward(g);
    }
}

#[test]
fn graph_forward_is_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<Graph<Forward>>();
}

#[test]
fn graph_transposed_is_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<Graph<Transposed>>();
}

// ---------------------------------------------------------------------------
// Builder produces correctly-typed graphs
// ---------------------------------------------------------------------------

#[test]
fn build_forward_returns_graph_forward() {
    use cugraph::{
        array::device_array_from_host,
        graph::GraphBuilder,
        handle::ResourceHandle,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    let graph: Graph<Forward> =
        GraphBuilder::new(props, &src_view, &dst_view).build_forward(&handle).expect("build_forward");

    // Verify type is Graph<Forward> by assigning to an explicit type binding.
    let _: &Graph<Forward> = &graph;
}

#[test]
fn build_transposed_returns_graph_transposed() {
    use cugraph::{
        array::device_array_from_host,
        graph::GraphBuilder,
        handle::ResourceHandle,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    let graph: Graph<Transposed> =
        GraphBuilder::new(props, &src_view, &dst_view).build_transposed(&handle).expect("build_transposed");

    let _: &Graph<Transposed> = &graph;
}

// ---------------------------------------------------------------------------
// Algorithm type-safety: correct variant compiles
// ---------------------------------------------------------------------------

#[test]
fn pagerank_accepts_graph_transposed() {
    use cugraph::{
        array::device_array_from_host,
        centrality::{PageRankParams, pagerank},
        graph::GraphBuilder,
        handle::ResourceHandle,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let src = device_array_from_host(&handle, &[1i32, 2, 3, 0, 0, 0], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[0i32, 0, 0, 1, 2, 3], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    let mut graph = GraphBuilder::new(props, &src_view, &dst_view).build_transposed(&handle).expect("build_transposed");

    let params = PageRankParams::default();
    let result = pagerank(&handle, &mut graph, &params).expect("pagerank");
    assert!(result.converged().expect("converged"));
}

#[test]
fn bfs_accepts_graph_forward() {
    use cugraph::{
        array::device_array_from_host,
        graph::GraphBuilder,
        handle::ResourceHandle,
        traversal::bfs,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 3], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    let mut graph = GraphBuilder::new(props, &src_view, &dst_view).build_forward(&handle).expect("build_forward");

    let sources_arr = device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
    let mut sources_view = sources_arr.view();

    let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");
    assert_eq!(result.vertices().len(), 4);
}

#[test]
fn betweenness_centrality_accepts_either_variant() {
    use cugraph::{
        array::device_array_from_host,
        centrality::{BetweennessCentralityParams, betweenness_centrality},
        graph::GraphBuilder,
        handle::ResourceHandle,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    let params = BetweennessCentralityParams::default();

    // Forward variant
    let mut graph_fwd =
        GraphBuilder::new(props, &src_view, &dst_view).renumber(true).build_forward(&handle).expect("build_forward");
    let _r1 = betweenness_centrality(&handle, &mut graph_fwd, None, &params).expect("betweenness forward");

    // Transposed variant
    let mut graph_trans = GraphBuilder::new(props, &src_view, &dst_view)
        .renumber(true)
        .build_transposed(&handle)
        .expect("build_transposed");
    let _r2 = betweenness_centrality(&handle, &mut graph_trans, None, &params).expect("betweenness transposed");
}

#[test]
fn force_atlas2_accepts_either_variant() {
    use cugraph::{
        array::device_array_from_host,
        graph::GraphBuilder,
        handle::ResourceHandle,
        layout::{ForceAtlas2Params, force_atlas2},
        params::IterationCount,
        random::RngState,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let mut rng = RngState::new(&handle, 42).expect("rng");

    // Symmetric triangle
    let src = device_array_from_host(&handle, &[0i32, 1, 2, 1, 2, 0], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0, 0, 1, 2], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: true, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let params = ForceAtlas2Params { max_iter: IterationCount::new(10).unwrap(), ..Default::default() };

    // Forward
    let mut graph_fwd =
        GraphBuilder::new(props, &src_view, &dst_view).renumber(true).build_forward(&handle).expect("build_forward");
    let _r1 = force_atlas2(&handle, &mut rng, &mut graph_fwd, &params, None, None).expect("fa2 forward");

    // Transposed
    let mut graph_trans = GraphBuilder::new(props, &src_view, &dst_view)
        .renumber(true)
        .build_transposed(&handle)
        .expect("build_transposed");
    let _r2 = force_atlas2(&handle, &mut rng, &mut graph_trans, &params, None, None).expect("fa2 transposed");
}

#[test]
fn in_degrees_requires_transposed() {
    use cugraph::{
        array::device_array_from_host,
        graph::GraphBuilder,
        handle::ResourceHandle,
        structure::in_degrees,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let src = device_array_from_host(&handle, &[1i32, 2, 3], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[0i32, 0, 0], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    let mut graph = GraphBuilder::new(props, &src_view, &dst_view)
        .renumber(true)
        .build_transposed(&handle)
        .expect("build_transposed");

    let result = in_degrees(&handle, &mut graph, None).expect("in_degrees");
    assert_eq!(result.vertices().len(), 4);
}

#[test]
fn out_degrees_accepts_either_variant() {
    use cugraph::{
        array::device_array_from_host,
        graph::GraphBuilder,
        handle::ResourceHandle,
        structure::out_degrees,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let src = device_array_from_host(&handle, &[1i32, 2, 3], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[0i32, 0, 0], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    // Forward
    let mut graph_fwd =
        GraphBuilder::new(props, &src_view, &dst_view).renumber(true).build_forward(&handle).expect("build_forward");
    let _r = out_degrees(&handle, &mut graph_fwd, None).expect("out_degrees forward");

    // Transposed
    let mut graph_trans = GraphBuilder::new(props, &src_view, &dst_view)
        .renumber(true)
        .build_transposed(&handle)
        .expect("build_transposed");
    let _r = out_degrees(&handle, &mut graph_trans, None).expect("out_degrees transposed");
}

#[test]
fn degrees_requires_transposed() {
    use cugraph::{
        array::device_array_from_host,
        graph::GraphBuilder,
        handle::ResourceHandle,
        structure::degrees,
        types::{DataType, GraphProperties},
    };

    let handle = ResourceHandle::new().expect("handle");
    let src = device_array_from_host(&handle, &[1i32, 2, 3], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[0i32, 0, 0], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    let mut graph = GraphBuilder::new(props, &src_view, &dst_view)
        .renumber(true)
        .build_transposed(&handle)
        .expect("build_transposed");

    let result = degrees(&handle, &mut graph, None).expect("degrees");
    assert_eq!(result.vertices().len(), 4);
}

// ---------------------------------------------------------------------------
// Transposition trait is sealed
// ---------------------------------------------------------------------------

#[test]
fn transposition_trait_is_implemented_for_forward_and_transposed() {
    use cugraph::graph::Transposition;

    fn assert_transposition<T: Transposition>() {}
    assert_transposition::<Forward>();
    assert_transposition::<Transposed>();
}
