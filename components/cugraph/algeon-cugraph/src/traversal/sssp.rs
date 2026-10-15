//! Single-Source Shortest Path (SSSP) traversal algorithm.

use std::{fmt::Debug, ptr};

use cugraph_sys::{cugraph_error_t, cugraph_paths_result_t, cugraph_sssp};

use crate::{
    cudf_interop::sssp_result_to_dataframe,
    error::{Error, ErrorKind, Result, check_error},
    graph::{Forward, Graph},
    handle::EnteredContext,
    params::NonNegativeF64,
    structure::{
        VertexSelection,
        typed_results::{Distances, VertexIds},
    },
    traversal::paths::{ExtractPathsResult, PathsResult, extract_paths},
    types::{CugraphElement, to_bool_t},
};

/// Parameters for the SSSP traversal algorithm.
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct SsspParams {
    /// Maximum edge-weight sum to consider. `None` means no cutoff.
    pub cutoff: Option<NonNegativeF64>,
    /// Whether predecessor vertices should be included in the result.
    pub compute_predecessors: bool,
    /// Whether to run expensive input validation checks in the C library.
    pub do_expensive_check: bool,
}
impl SsspParams {
    /// Enable computation of predecessor vertices in the SSSP result.
    pub fn with_predecessors(mut self) -> Self {
        self.compute_predecessors = true;
        self
    }

    /// Set the maximum edge-weight sum to consider.
    pub fn with_cutoff(mut self, cutoff: NonNegativeF64) -> Self {
        self.cutoff = Some(cutoff);
        self
    }

    /// Enable expensive input validation checks in the C library.
    pub fn with_expensive_check(mut self) -> Self {
        self.do_expensive_check = true;
        self
    }

    /// The maximum edge-weight sum, or `None` for unlimited traversal.
    pub fn cutoff(&self) -> Option<NonNegativeF64> {
        self.cutoff
    }

    /// Whether predecessor vertices will be computed.
    pub fn compute_predecessors(&self) -> bool {
        self.compute_predecessors
    }

    /// Whether expensive input validation checks are enabled.
    pub fn do_expensive_check(&self) -> bool {
        self.do_expensive_check
    }
}

pub(crate) fn sssp<V>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    source: V,
    params: &SsspParams,
) -> Result<SsspResult>
where
    V: CugraphElement + TryInto<usize> + Copy + Debug,
{
    // `cugraph_sssp` takes the source as `size_t`, so a negative vertex id cannot be addressed.
    let source = source.try_into().map_err(|_| {
        Error::new(ErrorKind::InvalidInput, "sssp")
            .with_message(format!("SSSP source vertex {source:?} is negative; SSSP requires a non-negative vertex id"))
            .with_context("source", format!("{source:?}"))
    })?;
    let cutoff = params.cutoff.map(f64::from).unwrap_or(f64::MAX);

    let mut result: *mut cugraph_paths_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "sssp")?;

    // SAFETY: `graph` stays valid for the duration of the call, `source` was
    // validated by `Graph::sssp`, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_sssp(
            handle.handle_ptr(),
            graph_ptr,
            source,
            cutoff,
            to_bool_t(params.compute_predecessors),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "sssp")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the paths result to
    // the caller.
    let paths = unsafe { PathsResult::from_raw(result, handle.device(), "sssp", handle.runtime_bound_lease()) }?;
    Ok(SsspResult { paths })
}

/// Result of an SSSP computation.
pub struct SsspResult {
    pub(crate) paths: PathsResult,
}

impl std::fmt::Debug for SsspResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SsspResult").finish_non_exhaustive()
    }
}

impl SsspResult {
    /// Get the vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.paths.vertex_ids("SsspResult::vertex_ids")
    }

    /// Get the weighted SSSP distances as a runtime-typed device array view.
    ///
    /// Returns [`Distances::F32`] or [`Distances::F64`] depending on the
    /// graph's edge-weight type.
    pub fn distances(&self) -> Result<Distances<'_>> {
        self.paths.distances("SsspResult::distances")
    }

    /// Get the predecessor vertices as a runtime-typed device array view.
    ///
    /// Returns `None` if `compute_predecessors` was `false` in the SSSP call.
    pub fn predecessors(&self) -> Result<Option<VertexIds<'_>>> {
        self.paths.predecessors("SsspResult::predecessors")
    }

    /// Extract concrete row-major paths to the requested destination vertices.
    ///
    /// `sources` must contain the single source vertex originally used to
    /// compute this SSSP result. cuGraph requires this input again when
    /// extracting paths from the predecessor tree.
    pub fn extract_paths<'a, V>(
        &self,
        handle: &EnteredContext<'_>,
        graph: &mut Graph<Forward>,
        sources: impl Into<VertexSelection<'a, V>>,
        destinations: impl Into<VertexSelection<'a, V>>,
    ) -> Result<ExtractPathsResult>
    where
        V: CugraphElement + TryInto<i64> + Copy + Debug + 'a,
        <V as TryInto<i64>>::Error: Debug,
    {
        extract_paths(handle, graph, &self.paths, sources, destinations, "SsspResult::extract_paths")
    }

    /// Convert this SSSP result to a cudf DataFrame.
    ///
    /// Returns `vertex` and `distance` columns, plus a `predecessor` column
    /// when predecessors were computed.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        sssp_result_to_dataframe(self, runtime).map_err(|e: Error| e.with_operation("SsspResult::to_dataframe"))
    }
}
