//! Result types for centrality algorithms.
//!
//! Both PageRank and betweenness centrality return a
//! `cugraph_centrality_result_t` from the C API. This module provides a shared
//! internal wrapper ([`CentralityResult`]) and algorithm-specific newtypes
//! ([`PageRankResult`], [`BetweennessCentralityResult`]) that expose only the
//! methods relevant to each algorithm.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_centrality_result_converged, cugraph_centrality_result_free, cugraph_centrality_result_get_num_iterations,
    cugraph_centrality_result_get_values, cugraph_centrality_result_get_vertices, cugraph_centrality_result_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result, require_non_null},
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
    device: i32,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl CentralityResult {
    /// Wrap a raw centrality result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph centrality
    /// C API function. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(raw: *mut cugraph_centrality_result_t, device: i32) -> Result<Self> {
        let raw = require_non_null(raw, "CentralityResult::from_raw", "centrality_result")?;
        Ok(Self { raw, device, _nosend_nosync: PhantomData })
    }

    pub(crate) fn vertices_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        // SAFETY: `self.raw` is a live centrality result owned by this
        // wrapper for the duration of the call.
        let view_ptr = require_non_null(
            unsafe { cugraph_centrality_result_get_vertices(self.raw) },
            "CentralityResult::vertices_raw",
            "vertices",
        )?;
        // SAFETY: `view_ptr` belongs to the live centrality result and was
        // validated to be non-null above.
        unsafe {
            RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), "CentralityResult::vertices_raw", "vertices")
        }
    }

    pub(crate) fn values_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        // SAFETY: `self.raw` is a live centrality result owned by this
        // wrapper for the duration of the call.
        let view_ptr = require_non_null(
            unsafe { cugraph_centrality_result_get_values(self.raw) },
            "CentralityResult::values_raw",
            "values",
        )?;
        // SAFETY: `view_ptr` belongs to the live centrality result and was
        // validated to be non-null above.
        unsafe {
            RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), "CentralityResult::values_raw", "values")
        }
    }

    pub(crate) fn num_iterations(&self) -> usize {
        // SAFETY: `self.raw` is a live centrality result owned by this
        // wrapper for the duration of the call.
        unsafe { cugraph_centrality_result_get_num_iterations(self.raw) }
    }

    pub(crate) fn converged(&self) -> Result<bool> {
        // SAFETY: `self.raw` is a live centrality result owned by this
        // wrapper for the duration of the call.
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
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "CentralityResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and has not been
            // freed yet.
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
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_pagerank_allow_nonconvergence`.
    /// Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(raw: *mut cugraph_centrality_result_t, device: i32) -> Result<Self> {
        // SAFETY: the caller transfers ownership of a live cuGraph result to
        // this wrapper on success.
        Ok(Self(unsafe { CentralityResult::from_raw(raw, device) }?))
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

impl PageRankResult {
    /// Convert this PageRank result to a cudf DataFrame.
    ///
    /// Returns a 2-column DataFrame:
    /// - `"vertex"` — vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"value"` — PageRank values with the underlying result dtype (`Float32` or `Float64`)
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::page_rank_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("PageRankResult::to_dataframe"))
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
    pub(crate) unsafe fn from_raw(raw: *mut cugraph_centrality_result_t, device: i32) -> Result<Self> {
        // SAFETY: the caller transfers ownership of a live cuGraph result to
        // this wrapper on success.
        Ok(Self(unsafe { CentralityResult::from_raw(raw, device) }?))
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

impl BetweennessCentralityResult {
    /// Convert this betweenness centrality result to a cudf DataFrame.
    ///
    /// Returns a 2-column DataFrame:
    /// - `"vertex"` — vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"value"` — betweenness centrality values with the underlying result
    ///   dtype (`Float32` or `Float64`)
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::betweenness_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("BetweennessCentralityResult::to_dataframe"))
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn centrality_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { CentralityResult::from_raw(ptr::null_mut(), 0) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "CentralityResult::from_raw");
    }
}
