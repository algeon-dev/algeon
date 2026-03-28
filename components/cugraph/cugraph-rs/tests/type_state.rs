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
// Transposition trait is sealed
// ---------------------------------------------------------------------------

#[test]
fn transposition_trait_is_implemented_for_forward_and_transposed() {
    use cugraph::graph::Transposition;

    fn assert_transposition<T: Transposition>() {}
    assert_transposition::<Forward>();
    assert_transposition::<Transposed>();
}
