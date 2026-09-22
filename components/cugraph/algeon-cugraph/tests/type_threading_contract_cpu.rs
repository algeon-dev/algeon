use cugraph::{
    array::{DeviceArray, DeviceArrayView},
    centrality::{BetweennessCentralityResult, CentralityResult, EdgeBetweennessCentralityResult, HitsResult},
    community::{EcgResult, LeidenResult, LouvainResult, SpectralClusteringResult},
    graph::{Forward, Graph, ParkedGraph, Transposed},
    layout::LayoutResult,
    random::RngState,
    sampling::{RandomWalkResult, SampleResult},
    similarity::{
        CosineSimilarityResult, JaccardSimilarityResult, OverlapSimilarityResult, SorensenSimilarityResult, VertexPairs,
    },
    traversal::{BfsResult, ExtractPathsResult, SsspResult},
};
use static_assertions::{assert_impl_all, assert_not_impl_any};

#[test]
fn graph_and_result_types_are_not_send_or_sync() {
    assert_not_impl_any!(DeviceArray<i32>: Send, Sync);
    assert_not_impl_any!(DeviceArrayView<'static, i32>: Send, Sync);

    assert_not_impl_any!(Graph: Send, Sync);
    assert_not_impl_any!(Graph<Forward>: Send, Sync);
    assert_not_impl_any!(Graph<Transposed>: Send, Sync);
    assert_impl_all!(ParkedGraph<Forward>: Send);
    assert_impl_all!(ParkedGraph<Transposed>: Send);
    assert_not_impl_any!(ParkedGraph<Forward>: Sync);
    assert_not_impl_any!(ParkedGraph<Transposed>: Sync);
    assert_not_impl_any!(CentralityResult: Send, Sync);
    assert_not_impl_any!(BetweennessCentralityResult: Send, Sync);
    assert_not_impl_any!(HitsResult: Send, Sync);
    assert_not_impl_any!(EdgeBetweennessCentralityResult: Send, Sync);
    assert_not_impl_any!(LouvainResult: Send, Sync);
    assert_not_impl_any!(LeidenResult: Send, Sync);
    assert_not_impl_any!(EcgResult: Send, Sync);
    assert_not_impl_any!(SpectralClusteringResult: Send, Sync);
    assert_not_impl_any!(BfsResult: Send, Sync);
    assert_not_impl_any!(SsspResult: Send, Sync);
    assert_not_impl_any!(ExtractPathsResult: Send, Sync);
    assert_not_impl_any!(RandomWalkResult: Send, Sync);
    assert_not_impl_any!(SampleResult: Send, Sync);
    assert_not_impl_any!(LayoutResult: Send, Sync);
    assert_not_impl_any!(VertexPairs: Send, Sync);
    assert_not_impl_any!(JaccardSimilarityResult: Send, Sync);
    assert_not_impl_any!(SorensenSimilarityResult: Send, Sync);
    assert_not_impl_any!(OverlapSimilarityResult: Send, Sync);
    assert_not_impl_any!(CosineSimilarityResult: Send, Sync);
    assert_not_impl_any!(RngState: Send, Sync);
}
