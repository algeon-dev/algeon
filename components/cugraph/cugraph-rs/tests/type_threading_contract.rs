use static_assertions::assert_not_impl_any;

use cugraph::array::{DeviceArray, DeviceArrayView, HostArrayView};
use cugraph::centrality::PageRankResult;
use cugraph::graph::Graph;
use cugraph::traversal::BfsResult;

#[test]
fn graph_and_result_types_are_not_send_or_sync() {
    assert_not_impl_any!(DeviceArray: Send, Sync);
    assert_not_impl_any!(DeviceArrayView<'static>: Send, Sync);
    assert_not_impl_any!(HostArrayView<'static>: Send, Sync);

    assert_not_impl_any!(Graph: Send, Sync);
    assert_not_impl_any!(PageRankResult: Send, Sync);
    assert_not_impl_any!(BfsResult: Send, Sync);
}
