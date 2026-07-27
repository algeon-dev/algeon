//! Shared result type for centrality algorithms.
//!
//! Both PageRank and betweenness centrality return a
//! `cugraph_centrality_result_t` from the C API. This module provides a single
//! safe wrapper that is reused by all centrality algorithms.

use cugraph_sys::{
    cugraph_centrality_result_converged, cugraph_centrality_result_free,
    cugraph_centrality_result_get_num_iterations, cugraph_centrality_result_get_values,
    cugraph_centrality_result_get_vertices, cugraph_centrality_result_t,
};

use crate::array::DeviceArrayView;
use crate::error::{Error, ErrorKind, Result};
use crate::types::try_from_bool_t;

/// Result of a centrality computation (PageRank, betweenness centrality, etc.).
///
/// Provides access to vertex ids and their centrality values. Iteration count
/// and convergence status are algorithm-dependent metadata (for iterative
/// algorithms like PageRank they are meaningful; for non-iterative algorithms
/// they may be default values).
///
/// Freed on drop.
pub struct CentralityResult {
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
        Self {
            raw,
            _nosend_nosync: std::marker::PhantomData,
        }
    }

    /// Get the vertex ids as a device array view.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn vertices(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_centrality_result_get_vertices(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Get the centrality values as a device array view.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn values(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_centrality_result_get_values(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Number of iterations performed.
    ///
    /// For iterative algorithms (for example PageRank), this is the number of
    /// iterations the solver executed. For non-iterative algorithms, this may
    /// be `0`.
    pub fn num_iterations(&self) -> usize {
        unsafe { cugraph_centrality_result_get_num_iterations(self.raw) }
    }

    /// Whether the algorithm converged within the maximum iterations.
    ///
    /// For iterative algorithms this reports convergence. For non-iterative
    /// algorithms, this may be a default value from the C API result object.
    pub fn converged(&self) -> Result<bool> {
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
