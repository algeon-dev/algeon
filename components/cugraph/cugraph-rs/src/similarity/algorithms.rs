//! FFI calls for similarity algorithms.

use std::ptr;

use cugraph_sys::{
    bool_t, cugraph_all_pairs_cosine_similarity_coefficients, cugraph_all_pairs_jaccard_coefficients,
    cugraph_all_pairs_overlap_coefficients, cugraph_all_pairs_sorensen_coefficients,
    cugraph_cosine_similarity_coefficients, cugraph_create_vertex_pairs, cugraph_error_t, cugraph_jaccard_coefficients,
    cugraph_overlap_coefficients, cugraph_similarity_result_t, cugraph_sorensen_coefficients,
    cugraph_type_erased_device_array_view_t, cugraph_vertex_pairs_t,
};

use crate::{
    array::DeviceArrayView,
    error::{Result, check_error, require_non_null},
    graph::Graph,
    handle::EnteredContext,
    similarity::{
        AllPairsSimilarityParams, CosineSimilarityResult, JaccardSimilarityResult, OverlapSimilarityResult,
        SimilarityParams, SorensenSimilarityResult, VertexPairs, result::SimilarityResultInner,
    },
    types::{CugraphElement, to_bool_t},
};

type PairwiseSimilarityFn = unsafe extern "C" fn(
    *const cugraph_sys::cugraph_resource_handle_t,
    *mut cugraph_sys::cugraph_graph_t,
    *const cugraph_vertex_pairs_t,
    bool_t,
    bool_t,
    *mut *mut cugraph_similarity_result_t,
    *mut *mut cugraph_error_t,
) -> cugraph_sys::cugraph_error_code_t;

type AllPairsSimilarityFn = unsafe extern "C" fn(
    *const cugraph_sys::cugraph_resource_handle_t,
    *mut cugraph_sys::cugraph_graph_t,
    *const cugraph_type_erased_device_array_view_t,
    bool_t,
    usize,
    bool_t,
    *mut *mut cugraph_similarity_result_t,
    *mut *mut cugraph_error_t,
) -> cugraph_sys::cugraph_error_code_t;

pub(crate) fn create_vertex_pairs<T, V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    first: &DeviceArrayView<'_, V>,
    second: &DeviceArrayView<'_, V>,
) -> Result<VertexPairs> {
    if first.len() != second.len() {
        return Err(crate::Error::new(crate::ErrorKind::InvalidInput, "create_vertex_pairs")
            .with_message("first and second vertex-pair inputs must have the same length")
            .with_context("first_len", first.len().to_string())
            .with_context("second_len", second.len().to_string()));
    }

    let mut result: *mut cugraph_vertex_pairs_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: handle, graph, and input views are live for the duration of the
    // call, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_create_vertex_pairs(
            handle.handle_ptr(),
            graph.as_ptr(),
            first.as_ptr().cast_const(),
            second.as_ptr().cast_const(),
            &mut result,
            &mut error,
        );
        check_error(code, error, "create_vertex_pairs")?;
    }

    let result = require_non_null(result, "create_vertex_pairs", "vertex_pairs")?;
    // SAFETY: on success, cuGraph transfers ownership of the vertex pairs.
    unsafe { VertexPairs::from_raw(result, handle.device(), "create_vertex_pairs") }
}

fn pairwise_similarity(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<impl crate::graph::Transposition>,
    pairs: VertexPairs,
    params: &SimilarityParams,
    operation: &'static str,
    call: PairwiseSimilarityFn,
) -> Result<SimilarityResultInner> {
    let mut result: *mut cugraph_similarity_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: handle, graph, and vertex pairs are live for the duration of the
    // call, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = call(
            handle.handle_ptr(),
            graph.as_ptr(),
            pairs.as_ptr_const(),
            to_bool_t(params.use_weight),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, operation)?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { SimilarityResultInner::from_pairwise_raw(result, pairs, handle.device(), operation) }
}

fn all_pairs_similarity<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<impl crate::graph::Transposition>,
    vertices: Option<&DeviceArrayView<'_, V>>,
    params: &AllPairsSimilarityParams,
    operation: &'static str,
    call: AllPairsSimilarityFn,
) -> Result<SimilarityResultInner> {
    let mut result: *mut cugraph_similarity_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let vertices_ptr = vertices.map(|v| v.as_ptr().cast_const()).unwrap_or(ptr::null());
    let topk = params.topk.map(|value| value.get()).unwrap_or(usize::MAX);

    // SAFETY: handle, graph, and optional vertices view are live for the
    // duration of the call, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = call(
            handle.handle_ptr(),
            graph.as_ptr(),
            vertices_ptr,
            to_bool_t(params.use_weight),
            topk,
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, operation)?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { SimilarityResultInner::from_all_pairs_raw(result, handle.device(), operation) }
}

macro_rules! define_pairwise {
    ($fn_name:ident, $result_ty:ident, $c_fn:path, $operation:literal) => {
        pub(crate) fn $fn_name<T>(
            handle: &EnteredContext<'_>,
            graph: &mut Graph<T>,
            pairs: VertexPairs,
            params: &SimilarityParams,
        ) -> Result<$result_ty>
        where
            T: crate::graph::Transposition,
        {
            let inner = pairwise_similarity(handle, graph, pairs, params, $operation, $c_fn)?;
            Ok($result_ty::from_inner(inner))
        }
    };
}

macro_rules! define_all_pairs {
    ($fn_name:ident, $result_ty:ident, $c_fn:path, $operation:literal) => {
        pub(crate) fn $fn_name<T, V: CugraphElement>(
            handle: &EnteredContext<'_>,
            graph: &mut Graph<T>,
            vertices: Option<&DeviceArrayView<'_, V>>,
            params: &AllPairsSimilarityParams,
        ) -> Result<$result_ty>
        where
            T: crate::graph::Transposition,
        {
            let inner = all_pairs_similarity(handle, graph, vertices, params, $operation, $c_fn)?;
            Ok($result_ty::from_inner(inner))
        }
    };
}

define_pairwise!(jaccard_coefficients, JaccardSimilarityResult, cugraph_jaccard_coefficients, "jaccard_coefficients");
define_pairwise!(
    sorensen_coefficients,
    SorensenSimilarityResult,
    cugraph_sorensen_coefficients,
    "sorensen_coefficients"
);
define_pairwise!(overlap_coefficients, OverlapSimilarityResult, cugraph_overlap_coefficients, "overlap_coefficients");
define_pairwise!(
    cosine_similarity_coefficients,
    CosineSimilarityResult,
    cugraph_cosine_similarity_coefficients,
    "cosine_similarity_coefficients"
);

define_all_pairs!(
    all_pairs_jaccard_coefficients,
    JaccardSimilarityResult,
    cugraph_all_pairs_jaccard_coefficients,
    "all_pairs_jaccard_coefficients"
);
define_all_pairs!(
    all_pairs_sorensen_coefficients,
    SorensenSimilarityResult,
    cugraph_all_pairs_sorensen_coefficients,
    "all_pairs_sorensen_coefficients"
);
define_all_pairs!(
    all_pairs_overlap_coefficients,
    OverlapSimilarityResult,
    cugraph_all_pairs_overlap_coefficients,
    "all_pairs_overlap_coefficients"
);
define_all_pairs!(
    all_pairs_cosine_similarity_coefficients,
    CosineSimilarityResult,
    cugraph_all_pairs_cosine_similarity_coefficients,
    "all_pairs_cosine_similarity_coefficients"
);
