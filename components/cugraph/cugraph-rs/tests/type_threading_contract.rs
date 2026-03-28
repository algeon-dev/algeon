use cugraph::{
    array::{DeviceArray, DeviceArrayView, HostArrayView},
    centrality::{CentralityResult, PageRankResult},
    graph::{Forward, Graph, Transposed},
    layout::LayoutResult,
    random::RngState,
    traversal::BfsResult,
};
use static_assertions::assert_not_impl_any;

#[test]
fn graph_and_result_types_are_not_send_or_sync() {
    assert_not_impl_any!(DeviceArray: Send, Sync);
    assert_not_impl_any!(DeviceArrayView<'static>: Send, Sync);
    assert_not_impl_any!(HostArrayView<'static>: Send, Sync);

    assert_not_impl_any!(Graph: Send, Sync);
    assert_not_impl_any!(Graph<Forward>: Send, Sync);
    assert_not_impl_any!(Graph<Transposed>: Send, Sync);
    assert_not_impl_any!(CentralityResult: Send, Sync);
    assert_not_impl_any!(PageRankResult: Send, Sync);
    assert_not_impl_any!(BfsResult: Send, Sync);
    assert_not_impl_any!(LayoutResult: Send, Sync);
    assert_not_impl_any!(RngState: Send, Sync);
}
