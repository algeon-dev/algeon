//! Tests for `GraphBuilder` — the builder pattern for `Graph` construction.
//!
//! These tests verify the builder's configuration API, default values, trait
//! implementations, and validation logic. GPU-dependent tests (actual graph
//! construction) live alongside the `GraphBuilder` tests in `graph.rs`.

use cugraph::{
    array::device_array_from_host,
    error::ErrorKind,
    graph::GraphBuilder,
    handle::ResourceHandle,
    types::{DataType, GraphProperties},
};

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

#[test]
fn builder_is_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<GraphBuilder<'_>>();
}

#[test]
fn builder_is_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<GraphBuilder<'_>>();
}

// ---------------------------------------------------------------------------
// Default values
// ---------------------------------------------------------------------------

#[test]
fn builder_defaults_are_correct() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let builder = GraphBuilder::new(props, &src_view, &dst_view);
    let debug = format!("{:?}", builder);

    // renumber defaults to false
    assert!(debug.contains("renumber: false"), "renumber should default to false");
    // All newly-exposed options default to false
    assert!(debug.contains("drop_self_loops: false"), "drop_self_loops should default to false");
    assert!(debug.contains("drop_multi_edges: false"), "drop_multi_edges should default to false");
    assert!(debug.contains("symmetrize: false"), "symmetrize should default to false");
    assert!(debug.contains("do_expensive_check: false"), "do_expensive_check should default to false");
}

// ---------------------------------------------------------------------------
// Chainable setters
// ---------------------------------------------------------------------------

#[test]
fn builder_setters_are_chainable() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 0], DataType::Int32).expect("dst");
    let wgt = device_array_from_host(&handle, &[1.0f32, 2.0], DataType::Float32).expect("wgt");
    let eids = device_array_from_host(&handle, &[10i32, 20], DataType::Int32).expect("eids");
    let etids = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("etids");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let wgt_view = wgt.view();
    let eids_view = eids.view();
    let etids_view = etids.view();

    // All setters should be callable in a chain without compile error.
    let _builder = GraphBuilder::new(props, &src_view, &dst_view)
        .weights(&wgt_view)
        .edge_ids(&eids_view)
        .edge_type_ids(&etids_view)
        .renumber(true)
        .drop_self_loops(true)
        .drop_multi_edges(true)
        .symmetrize(true)
        .do_expensive_check(true);
}

// ---------------------------------------------------------------------------
// Validation — length mismatches
// ---------------------------------------------------------------------------

#[test]
fn builder_rejects_mismatched_src_dst_lengths() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let err = GraphBuilder::new(props, &src_view, &dst_view)
        .build_forward(&handle)
        .expect_err("should reject mismatched src/dst lengths");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[test]
fn builder_rejects_mismatched_weights_length() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let wgt = device_array_from_host(&handle, &[1.0f32, 2.0], DataType::Float32).expect("wgt");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let wgt_view = wgt.view();
    let err = GraphBuilder::new(props, &src_view, &dst_view)
        .weights(&wgt_view)
        .build_forward(&handle)
        .expect_err("should reject mismatched weights length");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("weights"));
}

#[test]
fn builder_rejects_mismatched_edge_ids_length() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let eids = device_array_from_host(&handle, &[10i32, 20], DataType::Int32).expect("eids");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let eids_view = eids.view();
    let err = GraphBuilder::new(props, &src_view, &dst_view)
        .edge_ids(&eids_view)
        .build_forward(&handle)
        .expect_err("should reject mismatched edge_ids length");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("edge_ids"));
}

#[test]
fn builder_rejects_mismatched_edge_type_ids_length() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let etids = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("etids");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let etids_view = etids.view();
    let err = GraphBuilder::new(props, &src_view, &dst_view)
        .edge_type_ids(&etids_view)
        .build_forward(&handle)
        .expect_err("should reject mismatched edge_type_ids length");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("edge_type_ids"));
}

#[test]
fn builder_rejects_mismatched_edge_start_time_ids_length() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let times = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("times");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let times_view = times.view();
    let err = GraphBuilder::new(props, &src_view, &dst_view)
        .edge_start_time_ids(&times_view)
        .build_forward(&handle)
        .expect_err("should reject mismatched edge_start_time_ids length");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("edge_start_time_ids"));
}

#[test]
fn builder_rejects_mismatched_edge_end_time_ids_length() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let times = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("times");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let times_view = times.view();
    let err = GraphBuilder::new(props, &src_view, &dst_view)
        .edge_end_time_ids(&times_view)
        .build_forward(&handle)
        .expect_err("should reject mismatched edge_end_time_ids length");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("edge_end_time_ids"));
}

#[test]
fn builder_with_all_optional_arrays() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
    let wgt = device_array_from_host(&handle, &[1.0f32, 2.0, 3.0], DataType::Float32).expect("wgt");
    let eids = device_array_from_host(&handle, &[100i32, 101, 102], DataType::Int32).expect("eids");
    let etids = device_array_from_host(&handle, &[0i32, 1, 0], DataType::Int32).expect("etids");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();
    let wgt_view = wgt.view();
    let eids_view = eids.view();
    let etids_view = etids.view();

    let _graph = GraphBuilder::new(props, &src_view, &dst_view)
        .weights(&wgt_view)
        .edge_ids(&eids_view)
        .edge_type_ids(&etids_view)
        .renumber(true)
        .build_forward(&handle)
        .expect("builder with all optional arrays");
}

#[test]
fn builder_clone_produces_independent_copy() {
    let handle = ResourceHandle::new().expect("handle");

    let src = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("src");
    let dst = device_array_from_host(&handle, &[1i32, 0], DataType::Int32).expect("dst");
    let props = GraphProperties { is_symmetric: false, is_multigraph: false };

    let src_view = src.view();
    let dst_view = dst.view();

    let mut builder = GraphBuilder::new(props, &src_view, &dst_view);
    builder.renumber(true);
    let mut cloned = builder.clone();
    cloned.renumber(false);

    // Original should still have renumber=true
    let debug = format!("{:?}", builder);
    assert!(debug.contains("renumber: true"), "original builder should be unaffected by clone mutation");
}
