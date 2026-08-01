//! Community subgraph extraction algorithms.

use std::{num::NonZeroUsize, ptr};

use cugraph_sys::{cugraph_error_t, cugraph_extract_ego, cugraph_induced_subgraph_result_t, cugraph_k_truss_subgraph};

use crate::{
    array::DeviceArrayView,
    error::{Error, Result, check_error, require_non_null},
    graph::Graph,
    handle::EnteredContext,
    structure::{
        subgraph::InducedSubgraphResult,
        typed_results::{IntegerValues, ScoreValues, SizeValues, VertexIds},
    },
    types::{CugraphElement, to_bool_t},
};

/// Parameters for [`Graph::k_truss_subgraph`](crate::graph::Graph::k_truss_subgraph).
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct KTrussSubgraphParams {
    /// K value for the extracted truss. Defaults to 3.
    pub k: NonZeroUsize,
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

impl Default for KTrussSubgraphParams {
    fn default() -> Self {
        Self {
            // SAFETY: the literal `3` is non-zero.
            k: unsafe { NonZeroUsize::new_unchecked(3) },
            do_expensive_check: false,
        }
    }
}

/// Parameters for [`Graph::extract_ego`](crate::graph::Graph::extract_ego).
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ExtractEgoParams {
    /// Number of hops to expand from each source vertex.
    pub radius: NonZeroUsize,
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

impl Default for ExtractEgoParams {
    fn default() -> Self {
        Self {
            // SAFETY: the literal `1` is non-zero.
            radius: unsafe { NonZeroUsize::new_unchecked(1) },
            do_expensive_check: false,
        }
    }
}

/// Extract a k-truss subgraph.
pub(crate) fn k_truss_subgraph<T>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    params: &KTrussSubgraphParams,
) -> Result<KTrussSubgraphResult> {
    let mut result: *mut cugraph_induced_subgraph_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "k_truss_subgraph")?;

    // SAFETY: `graph` and `handle` are live for the duration of the call, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_k_truss_subgraph(
            handle.handle_ptr(),
            graph_ptr,
            params.k.get(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "k_truss_subgraph")?;
    }

    let result = require_non_null(result, "k_truss_subgraph", "induced_subgraph_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    let inner = unsafe {
        InducedSubgraphResult::from_raw(result, handle.device(), "k_truss_subgraph", handle.runtime_bound_lease())
    }?;
    Ok(KTrussSubgraphResult(inner))
}

/// Extract ego subgraphs for source vertices.
pub(crate) fn extract_ego<T, V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    source_vertices: &DeviceArrayView<'_, V>,
    params: &ExtractEgoParams,
) -> Result<ExtractEgoResult> {
    let mut result: *mut cugraph_induced_subgraph_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "extract_ego")?;

    // SAFETY: `source_vertices` is a live device view validated by `Graph`,
    // and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_extract_ego(
            handle.handle_ptr(),
            graph_ptr,
            source_vertices.as_ptr().cast_const(),
            params.radius.get(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "extract_ego")?;
    }

    let result = require_non_null(result, "extract_ego", "induced_subgraph_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    let inner = unsafe {
        InducedSubgraphResult::from_raw(result, handle.device(), "extract_ego", handle.runtime_bound_lease())
    }?;
    Ok(ExtractEgoResult(inner))
}

macro_rules! induced_result_accessors {
    ($ty:ident, $df_fn:ident) => {
        impl $ty {
            /// Get source vertex ids.
            pub fn sources(&self) -> Result<VertexIds<'_>> {
                self.0.sources(concat!(stringify!($ty), "::sources"))
            }

            /// Get destination vertex ids.
            pub fn destinations(&self) -> Result<VertexIds<'_>> {
                self.0.destinations(concat!(stringify!($ty), "::destinations"))
            }

            /// Get optional edge weights.
            pub fn weights(&self) -> Result<Option<ScoreValues<'_>>> {
                self.0.weights(concat!(stringify!($ty), "::weights"))
            }

            /// Get optional edge ids.
            pub fn edge_ids(&self) -> Result<Option<IntegerValues<'_>>> {
                self.0.edge_ids(concat!(stringify!($ty), "::edge_ids"))
            }

            /// Get optional edge type ids.
            pub fn edge_type_ids(&self) -> Result<Option<IntegerValues<'_>>> {
                self.0.edge_type_ids(concat!(stringify!($ty), "::edge_type_ids"))
            }

            /// Get optional subgraph edge offsets.
            pub fn subgraph_offsets(&self) -> Result<Option<SizeValues<'_>>> {
                self.0.subgraph_offsets(concat!(stringify!($ty), "::subgraph_offsets"))
            }

            /// Convert this result to a cudf DataFrame.
            ///
            /// Returns `"source"` and `"destination"` columns, plus optional
            /// `"weight"`, `"edge_id"`, and `"edge_type"` columns when present.
            pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
                crate::cudf_interop::$df_fn(self, runtime)
                    .map_err(|e: Error| e.with_operation(concat!(stringify!($ty), "::to_dataframe")))
            }
        }
    };
}

/// Result of a k-truss subgraph extraction.
pub struct KTrussSubgraphResult(InducedSubgraphResult);

impl std::fmt::Debug for KTrussSubgraphResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KTrussSubgraphResult").finish_non_exhaustive()
    }
}

induced_result_accessors!(KTrussSubgraphResult, k_truss_subgraph_result_to_dataframe);

/// Result of ego subgraph extraction.
pub struct ExtractEgoResult(InducedSubgraphResult);

impl std::fmt::Debug for ExtractEgoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractEgoResult").finish_non_exhaustive()
    }
}

induced_result_accessors!(ExtractEgoResult, extract_ego_result_to_dataframe);
