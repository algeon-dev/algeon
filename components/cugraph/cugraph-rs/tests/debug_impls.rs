//! Compile-time verification that all public types implement Debug.

use std::fmt::Debug;

fn assert_debug<T: Debug>() {}

#[test]
fn all_public_types_implement_debug() {
    assert_debug::<cugraph::handle::ResourceHandle>();
    assert_debug::<cugraph::array::DeviceArray>();
    assert_debug::<cugraph::array::DeviceArrayView<'_>>();
    assert_debug::<cugraph::array::HostArrayView<'_>>();
    assert_debug::<cugraph::graph::Graph>();
    assert_debug::<cugraph::random::RngState>();
    assert_debug::<cugraph::centrality::CentralityResult>();
    assert_debug::<cugraph::traversal::BfsResult>();
    assert_debug::<cugraph::structure::DegreesResult>();
    assert_debug::<cugraph::layout::LayoutResult>();
    assert_debug::<cugraph::layout::ForceAtlas2VertexAttrs<'_>>();
    assert_debug::<cugraph::layout::InitialPositions<'_>>();
}
