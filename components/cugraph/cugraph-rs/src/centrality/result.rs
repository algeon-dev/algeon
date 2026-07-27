//! Result types for centrality algorithms.
//!
//! Both PageRank and betweenness centrality return a
//! `cugraph_centrality_result_t` from the C API. This module provides a shared
//! internal wrapper ([`CentralityResult`]) and algorithm-specific newtypes
//! ([`PageRankResult`], [`BetweennessCentralityResult`]) that expose only the
//! methods relevant to each algorithm.

use cugraph_sys::{
    cugraph_centrality_result_converged, cugraph_centrality_result_free, cugraph_centrality_result_get_num_iterations,
    cugraph_centrality_result_get_values, cugraph_centrality_result_get_vertices, cugraph_centrality_result_t,
};

use crate::{
    array::RawDeviceArrayView,
    error::{Error, ErrorKind, Result},
    structure::typed_results::{ScoreValues, VertexIds},
    types::try_from_bool_t,
};

/// Internal shared wrapper for `cugraph_centrality_result_t`.
///
/// Not part of the public API. Algorithm-specific newtypes ([`PageRankResult`],
/// [`BetweennessCentralityResult`]) wrap this and expose the relevant subset of
/// methods.
pub(crate) struct CentralityResult {
    raw: *mut cugraph_centrality_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl CentralityResult {
    /// Wrap a raw centrality result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph centrality
    /// C API function. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(raw: *mut cugraph_centrality_result_t) -> Self {
        Self { raw, _nosend_nosync: std::marker::PhantomData }
    }

    pub(crate) fn vertices_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_centrality_result_get_vertices(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "CentralityResult::vertices_raw")
                .with_message("C API returned null pointer for vertices"));
        }
        Ok(unsafe { RawDeviceArrayView::from_borrowed(view_ptr) })
    }

    pub(crate) fn values_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_centrality_result_get_values(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "CentralityResult::values_raw")
                .with_message("C API returned null pointer for values"));
        }
        Ok(unsafe { RawDeviceArrayView::from_borrowed(view_ptr) })
    }

    pub(crate) fn num_iterations(&self) -> usize {
        unsafe { cugraph_centrality_result_get_num_iterations(self.raw) }
    }

    pub(crate) fn converged(&self) -> Result<bool> {
        let b = unsafe { cugraph_centrality_result_converged(self.raw) };
        try_from_bool_t(b).ok_or_else(|| {
            Error::new(ErrorKind::Unexpected, "CentralityResult::converged")
                .with_message(format!("invalid bool_t value returned by cuGraph: {}", b.0))
        })
    }
}

impl Drop for CentralityResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_centrality_result_free(self.raw) };
        }
    }
}

// ---------------------------------------------------------------------------
// PageRankResult
// ---------------------------------------------------------------------------

/// Result of a PageRank computation.
///
/// Provides access to vertex ids, centrality values, iteration count, and
/// convergence status. Freed on drop.
pub struct PageRankResult(CentralityResult);

impl std::fmt::Debug for PageRankResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageRankResult").finish_non_exhaustive()
    }
}

impl PageRankResult {
    /// Wrap a raw centrality result pointer as a `PageRankResult`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by `cugraph_pagerank`.
    /// Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(raw: *mut cugraph_centrality_result_t) -> Self {
        Self(unsafe { CentralityResult::from_raw(raw) })
    }

    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.0.vertices_raw()?)
    }

    /// Get the PageRank scores as a runtime-typed device array view.
    ///
    /// Returns [`ScoreValues::F32`] or [`ScoreValues::F64`] depending on the
    /// graph's weight type.
    pub fn scores(&self) -> Result<ScoreValues<'_>> {
        ScoreValues::from_raw(self.0.values_raw()?)
    }

    /// Number of iterations performed by the PageRank solver.
    pub fn num_iterations(&self) -> usize {
        self.0.num_iterations()
    }

    /// Whether the PageRank algorithm converged within the maximum iterations.
    pub fn converged(&self) -> Result<bool> {
        self.0.converged()
    }
}

#[cfg(feature = "cudf-interop")]
impl PageRankResult {
    /// Convert this PageRank result to a cudf DataFrame.
    ///
    /// Returns a 2-column DataFrame:
    /// - `"vertex"` — vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"value"` — PageRank values with the underlying result dtype (`Float32` or `Float64`)
    ///
    /// Performs a device-to-device copy. The returned DataFrame owns its
    /// memory independently.
    pub fn to_dataframe(&self) -> Result<cudf::DataFrame> {
        use crate::cudf_interop::RawDeviceArrayViewExt;

        let vert_col =
            self.0.vertices_raw()?.to_cudf_column().map_err(|e| e.with_operation("PageRankResult::to_dataframe"))?;
        let val_col =
            self.0.values_raw()?.to_cudf_column().map_err(|e| e.with_operation("PageRankResult::to_dataframe"))?;

        cudf::DataFrame::new(vec!["vertex".into(), "value".into()], vec![vert_col, val_col]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "PageRankResult::to_dataframe")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

// ---------------------------------------------------------------------------
// BetweennessCentralityResult
// ---------------------------------------------------------------------------

/// Result of a betweenness centrality computation.
///
/// Provides access to vertex ids and their betweenness centrality values.
/// Unlike [`PageRankResult`], this does not expose iteration count or
/// convergence status since betweenness centrality is not an iterative
/// algorithm. Freed on drop.
pub struct BetweennessCentralityResult(CentralityResult);

impl std::fmt::Debug for BetweennessCentralityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BetweennessCentralityResult").finish_non_exhaustive()
    }
}

impl BetweennessCentralityResult {
    /// Wrap a raw centrality result pointer as a `BetweennessCentralityResult`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_betweenness_centrality`. Ownership is transferred to this
    /// struct.
    pub(crate) unsafe fn from_raw(raw: *mut cugraph_centrality_result_t) -> Self {
        Self(unsafe { CentralityResult::from_raw(raw) })
    }

    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.0.vertices_raw()?)
    }

    /// Get the betweenness centrality scores as a runtime-typed device array view.
    ///
    /// Returns [`ScoreValues::F32`] or [`ScoreValues::F64`] depending on the
    /// graph's weight type.
    pub fn scores(&self) -> Result<ScoreValues<'_>> {
        ScoreValues::from_raw(self.0.values_raw()?)
    }
}

#[cfg(feature = "cudf-interop")]
impl BetweennessCentralityResult {
    /// Convert this betweenness centrality result to a cudf DataFrame.
    ///
    /// Returns a 2-column DataFrame:
    /// - `"vertex"` — vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"value"` — betweenness centrality values with the underlying result
    ///   dtype (`Float32` or `Float64`)
    ///
    /// Performs a device-to-device copy. The returned DataFrame owns its
    /// memory independently.
    pub fn to_dataframe(&self) -> Result<cudf::DataFrame> {
        use crate::cudf_interop::RawDeviceArrayViewExt;

        let vert_col = self
            .0
            .vertices_raw()?
            .to_cudf_column()
            .map_err(|e| e.with_operation("BetweennessCentralityResult::to_dataframe"))?;
        let val_col = self
            .0
            .values_raw()?
            .to_cudf_column()
            .map_err(|e| e.with_operation("BetweennessCentralityResult::to_dataframe"))?;

        cudf::DataFrame::new(vec!["vertex".into(), "value".into()], vec![vert_col, val_col]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "BetweennessCentralityResult::to_dataframe")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}
