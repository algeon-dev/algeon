use cugraph::{
    array::{DeviceArray, DeviceArrayView},
    centrality::{BetweennessCentralityResult, PageRankResult},
    community::LouvainResult,
    graph::{Forward, Graph, Transposed},
    layout::LayoutResult,
    random::RngState,
    similarity::{
        CosineSimilarityResult, JaccardSimilarityResult, OverlapSimilarityResult, SorensenSimilarityResult, VertexPairs,
    },
    traversal::{BfsResult, ExtractPathsResult, SsspResult},
};
use static_assertions::assert_not_impl_any;

#[test]
fn graph_and_result_types_are_not_send_or_sync() {
    assert_not_impl_any!(DeviceArray<i32>: Send, Sync);
    assert_not_impl_any!(DeviceArrayView<'static, i32>: Send, Sync);

    assert_not_impl_any!(Graph: Send, Sync);
    assert_not_impl_any!(Graph<Forward>: Send, Sync);
    assert_not_impl_any!(Graph<Transposed>: Send, Sync);
    assert_not_impl_any!(PageRankResult: Send, Sync);
    assert_not_impl_any!(BetweennessCentralityResult: Send, Sync);
    assert_not_impl_any!(LouvainResult: Send, Sync);
    assert_not_impl_any!(BfsResult: Send, Sync);
    assert_not_impl_any!(SsspResult: Send, Sync);
    assert_not_impl_any!(ExtractPathsResult: Send, Sync);
    assert_not_impl_any!(LayoutResult: Send, Sync);
    assert_not_impl_any!(VertexPairs: Send, Sync);
    assert_not_impl_any!(JaccardSimilarityResult: Send, Sync);
    assert_not_impl_any!(SorensenSimilarityResult: Send, Sync);
    assert_not_impl_any!(OverlapSimilarityResult: Send, Sync);
    assert_not_impl_any!(CosineSimilarityResult: Send, Sync);
    assert_not_impl_any!(RngState: Send, Sync);
}
