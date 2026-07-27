use cugraph::{
    BfsParams, DeviceArray, Forward, GraphBuilder, PageRankParams, ResourceHandle, ScoreValues, Transposed, VertexIds,
};

#[test]
fn bfs_predecessors_are_none_when_not_requested() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 3]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    assert!(result.predecessors().expect("predecessors").is_none());
}

#[test]
fn pagerank_results_decode_runtime_dtypes() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[1i64, 2, 3, 0, 0, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[0i64, 0, 0, 1, 2, 3]).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f64; 6]).expect("wgt");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view())
        .weights(&wgt.view())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");

    assert!(matches!(result.vertex_ids().expect("vertex_ids"), VertexIds::I64(_)));
    assert!(matches!(result.scores().expect("scores"), ScoreValues::F64(_)));
}

#[test]
fn degree_results_decode_vertex_ids_without_panic() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i64, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i64, 2, 0]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Transposed>(&handle).expect("graph");
    let result = graph.degrees_all(&handle).expect("degrees");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));
    assert_eq!(vertex_ids.to_vec_i64(&handle).expect("copy").len(), 3);
}
