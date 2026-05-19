//! Minimum spanning tree algorithm.

use std::ptr;

use cugraph_sys::{cugraph_error_t, cugraph_induced_subgraph_result_t, cugraph_minimum_spanning_tree};

use crate::{
    error::{Error, Result, check_error, require_non_null},
    graph::Graph,
    handle::EnteredContext,
    structure::{
        subgraph::InducedSubgraphResult,
        typed_results::{IntegerValues, ScoreValues, SizeValues, VertexIds},
    },
    types::to_bool_t,
};

/// Parameters for [`Graph::minimum_spanning_tree`](crate::graph::Graph::minimum_spanning_tree).
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct MinimumSpanningTreeParams {
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

/// Compute a minimum spanning tree.
pub(crate) fn minimum_spanning_tree<T>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    params: &MinimumSpanningTreeParams,
) -> Result<MinimumSpanningTreeResult> {
    let mut result: *mut cugraph_induced_subgraph_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: `graph` and `handle` are live for the duration of the call, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_minimum_spanning_tree(
            handle.handle_ptr(),
            graph.as_ptr(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "minimum_spanning_tree")?;
    }

    let result = require_non_null(result, "minimum_spanning_tree", "induced_subgraph_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    let inner = unsafe { InducedSubgraphResult::from_raw(result, handle.device(), "minimum_spanning_tree") }?;
    Ok(MinimumSpanningTreeResult(inner))
}

/// Result of a minimum spanning tree computation.
pub struct MinimumSpanningTreeResult(InducedSubgraphResult);

impl std::fmt::Debug for MinimumSpanningTreeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinimumSpanningTreeResult").finish_non_exhaustive()
    }
}

impl MinimumSpanningTreeResult {
    /// Get source vertex ids.
    pub fn sources(&self) -> Result<VertexIds<'_>> {
        self.0.sources("MinimumSpanningTreeResult::sources")
    }

    /// Get destination vertex ids.
    pub fn destinations(&self) -> Result<VertexIds<'_>> {
        self.0.destinations("MinimumSpanningTreeResult::destinations")
    }

    /// Get optional edge weights.
    pub fn weights(&self) -> Result<Option<ScoreValues<'_>>> {
        self.0.weights("MinimumSpanningTreeResult::weights")
    }

    /// Get optional edge ids.
    pub fn edge_ids(&self) -> Result<Option<IntegerValues<'_>>> {
        self.0.edge_ids("MinimumSpanningTreeResult::edge_ids")
    }

    /// Get optional edge type ids.
    pub fn edge_type_ids(&self) -> Result<Option<IntegerValues<'_>>> {
        self.0.edge_type_ids("MinimumSpanningTreeResult::edge_type_ids")
    }

    /// Get optional subgraph edge offsets.
    pub fn subgraph_offsets(&self) -> Result<Option<SizeValues<'_>>> {
        self.0.subgraph_offsets("MinimumSpanningTreeResult::subgraph_offsets")
    }

    /// Convert this result to a cudf DataFrame.
    ///
    /// Returns `"source"` and `"destination"` columns, plus optional
    /// `"weight"`, `"edge_id"`, and `"edge_type"` columns when present.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::minimum_spanning_tree_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("MinimumSpanningTreeResult::to_dataframe"))
    }
}
