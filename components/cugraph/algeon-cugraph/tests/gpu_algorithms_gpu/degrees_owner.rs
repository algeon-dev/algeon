use cudf::testing::run_named_cases;
use cugraph::{
    DeviceArray, Graph, GraphBuilder, Transposed, handle::EnteredContext, structure::DegreesResult,
    types::GraphProperties,
};

use crate::test_handle;

/// Creates a star graph where vertices 1,2,3 all point to vertex 0.
/// With store_transposed=true and renumber=true (needed for degree computation).
fn make_star_graph(handle: &EnteredContext<'_>) -> Graph<Transposed> {
    // Directed star: 1->0, 2->0, 3->0
    let src_data: Vec<i32> = vec![1, 2, 3];
    let dst_data: Vec<i32> = vec![0, 0, 0];

    let src = DeviceArray::from_slice(handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(handle, &dst_data).expect("dst");

    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(handle).expect("star graph")
}

/// Builds a vertex -> (in_degree, out_degree) map from a DegreesResult
/// using the DegreeValues API.
fn degree_map(handle: &EnteredContext<'_>, result: &DegreesResult) -> std::collections::HashMap<i32, (i64, i64)> {
    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vert copy");
    let in_degs = result.in_degrees().expect("in_degrees").to_vec_i64(handle).expect("in copy");
    let out_degs = result.out_degrees().expect("out_degrees").to_vec_i64(handle).expect("out copy");

    let mut map = std::collections::HashMap::new();
    for i in 0..vertices.len() {
        map.insert(vertices[i], (in_degs[i], out_degs[i]));
    }
    map
}

#[test]
fn gpu_cases_1() {
    run_named_cases(&[
        ("structure::degrees::tests::test_degrees_symmetric_graph", test_degrees_symmetric_graph),
        ("structure::degrees::tests::test_in_degrees_star_graph", test_in_degrees_star_graph),
        ("structure::degrees::tests::test_degrees_star_graph", test_degrees_star_graph),
        ("structure::degrees::tests::test_degrees_isolated_vertex", test_degrees_isolated_vertex),
        ("structure::degrees::tests::test_out_degrees_star_graph", test_out_degrees_star_graph),
        ("structure::degrees::tests::test_degrees_cycle_graph", test_degrees_cycle_graph),
        (
            "structure::degrees::tests::test_degrees_with_source_vertices_subset",
            test_degrees_with_source_vertices_subset,
        ),
    ]);
}

fn test_degrees_symmetric_graph() {
    let handle = test_handle();
    let entered = handle.enter("test_degrees_symmetric_graph").expect("entered");

    // Symmetric (undirected) triangle: 0<->1, 1<->2, 2<->0
    let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 0];
    let dst_data: Vec<i32> = vec![1, 0, 2, 1, 0, 2];

    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");

    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(&entered)
        .expect("graph");

    let result = graph.degrees_all(&entered).expect("degrees");
    let map = degree_map(&entered, &result);

    // Symmetric graph: in_degree == out_degree for every vertex
    assert_eq!(map.len(), 3);
    for v in 0..3 {
        let (in_d, out_d) = map[&v];
        assert_eq!(in_d, out_d, "symmetric graph vertex {} should have in_degree == out_degree", v);
        assert_eq!(in_d, 2, "vertex {} should have degree 2 in triangle", v);
    }
}

fn test_in_degrees_star_graph() {
    let handle = test_handle();
    let entered = handle.enter("test_in_degrees_star_graph").expect("entered");
    let mut graph = make_star_graph(&entered);

    let result = graph.in_degrees_all(&entered).expect("in_degrees");

    let vert_ids = result.vertex_ids().expect("vertex_ids");
    let in_vals = result.in_degrees().expect("in_degrees");

    assert_eq!(vert_ids.len(), 4);
    assert_eq!(in_vals.len(), 4);

    let vertices: Vec<i32> = vert_ids.to_vec_i32(&entered).expect("vert copy");
    let in_degs = in_vals.to_vec_i64(&entered).expect("in copy");

    let mut map = std::collections::HashMap::new();
    for i in 0..vertices.len() {
        map.insert(vertices[i], in_degs[i]);
    }

    // Hub vertex 0 receives 3 edges
    assert_eq!(map[&0], 3, "hub should have in_degree=3");
    for v in [1, 2, 3] {
        assert_eq!(map[&v], 0, "leaf {} should have in_degree=0", v);
    }
}

fn test_degrees_star_graph() {
    let handle = test_handle();
    let entered = handle.enter("test_degrees_star_graph").expect("entered");
    let mut graph = make_star_graph(&entered);

    let result = graph.degrees_all(&entered).expect("degrees");

    let vert_view = result.vertex_ids().expect("vertex_ids");
    let in_vals = result.in_degrees().expect("in_degrees");
    let out_vals = result.out_degrees().expect("out_degrees");

    // Star with 4 vertices
    assert_eq!(vert_view.len(), 4);
    assert_eq!(in_vals.len(), 4);
    assert_eq!(out_vals.len(), 4);

    let map = degree_map(&entered, &result);

    // Vertex 0: hub; receives edges from 1,2,3 → in=3, out=0
    assert_eq!(map[&0], (3, 0), "hub should have in=3, out=0");

    // Vertices 1,2,3: leaves; each sends one edge → in=0, out=1
    for v in [1, 2, 3] {
        assert_eq!(map[&v], (0, 1), "leaf vertex {} should have in=0, out=1", v);
    }
}

fn test_degrees_isolated_vertex() {
    let handle = test_handle();
    let entered = handle.enter("test_degrees_isolated_vertex").expect("entered");

    // Graph: 0->1, 3->4; vertices 0,1,3,4 appear, and every degree is checked.
    let src_data: Vec<i32> = vec![0, 3];
    let dst_data: Vec<i32> = vec![1, 4];

    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");

    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&entered).expect("graph");

    let result = graph.degrees_all(&entered).expect("degrees");
    let map = degree_map(&entered, &result);

    // Vertex 0: out=1, in=0
    assert_eq!(map[&0], (0, 1), "vertex 0 should have in=0, out=1");
    // Vertex 1: out=0, in=1
    assert_eq!(map[&1], (1, 0), "vertex 1 should have in=1, out=0");
    // Vertex 3: out=1, in=0
    assert_eq!(map[&3], (0, 1), "vertex 3 should have in=0, out=1");
    // Vertex 4: out=0, in=1
    assert_eq!(map[&4], (1, 0), "vertex 4 should have in=1, out=0");
}

fn test_out_degrees_star_graph() {
    let handle = test_handle();
    let entered = handle.enter("test_out_degrees_star_graph").expect("entered");
    let mut graph = make_star_graph(&entered);

    let result = graph.out_degrees_all(&entered).expect("out_degrees");

    let vert_ids = result.vertex_ids().expect("vertex_ids");
    let out_vals = result.out_degrees().expect("out_degrees");

    assert_eq!(vert_ids.len(), 4);
    assert_eq!(out_vals.len(), 4);

    let vertices: Vec<i32> = vert_ids.to_vec_i32(&entered).expect("vert copy");
    let out_degs = out_vals.to_vec_i64(&entered).expect("out copy");

    let mut map = std::collections::HashMap::new();
    for i in 0..vertices.len() {
        map.insert(vertices[i], out_degs[i]);
    }

    // Hub vertex 0 sends 0 edges
    assert_eq!(map[&0], 0, "hub should have out_degree=0");
    // Leaf vertices each send 1 edge
    for v in [1, 2, 3] {
        assert_eq!(map[&v], 1, "leaf {} should have out_degree=1", v);
    }
}

fn test_degrees_cycle_graph() {
    let handle = test_handle();
    let entered = handle.enter("test_degrees_cycle_graph").expect("entered");

    // Directed cycle: 0->1, 1->2, 2->3, 3->0
    let src_data: Vec<i32> = vec![0, 1, 2, 3];
    let dst_data: Vec<i32> = vec![1, 2, 3, 0];

    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");

    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&entered).expect("graph");

    let result = graph.degrees_all(&entered).expect("degrees");
    let map = degree_map(&entered, &result);

    // Every vertex in a directed cycle has in=1, out=1
    assert_eq!(map.len(), 4);
    for v in 0..4 {
        assert_eq!(map[&v], (1, 1), "vertex {} in cycle should have in=1, out=1", v);
    }
}

fn test_degrees_with_source_vertices_subset() {
    let handle = test_handle();
    let entered = handle.enter("test_degrees_with_source_vertices_subset").expect("entered");
    let mut graph = make_star_graph(&entered);

    // Only query degrees for vertices 0 and 1
    let query = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("query");
    let query_view = query.view().expect("query_view");

    let result = graph.degrees(&entered, &query_view).expect("degrees subset");

    let vert_view = result.vertex_ids().expect("vertex_ids");
    assert_eq!(vert_view.len(), 2, "should return degrees for 2 vertices");

    let map = degree_map(&entered, &result);
    assert!(map.contains_key(&0), "should contain vertex 0");
    assert!(map.contains_key(&1), "should contain vertex 1");
    assert_eq!(map[&0], (3, 0), "hub should have in=3, out=0");
    assert_eq!(map[&1], (0, 1), "leaf 1 should have in=0, out=1");
}
