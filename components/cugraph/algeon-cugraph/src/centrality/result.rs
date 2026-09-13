//! Result types for centrality algorithms.
//!
//! Both PageRank and betweenness centrality return a
//! `cugraph_centrality_result_t` from the C API. This module provides a shared
//! internal wrapper ([`CentralityResult`]) and algorithm-specific newtypes
//! ([`PageRankResult`], [`BetweennessCentralityResult`]) that expose only the
//! methods relevant to each algorithm.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_centrality_result_converged, cugraph_centrality_result_get_num_iterations,
    cugraph_centrality_result_get_values, cugraph_centrality_result_get_vertices, cugraph_centrality_result_t,
    cugraph_edge_centrality_result_get_dst_vertices, cugraph_edge_centrality_result_get_edge_ids,
    cugraph_edge_centrality_result_get_src_vertices, cugraph_edge_centrality_result_get_values,
    cugraph_edge_centrality_result_t, cugraph_hits_result_get_authorities,
    cugraph_hits_result_get_hub_score_differences, cugraph_hits_result_get_hubs,
    cugraph_hits_result_get_number_of_iterations, cugraph_hits_result_get_vertices, cugraph_hits_result_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    cudf_interop::{
        betweenness_result_to_dataframe, edge_betweenness_centrality_result_to_dataframe,
        eigenvector_centrality_result_to_dataframe, hits_result_to_dataframe, katz_centrality_result_to_dataframe,
        page_rank_result_to_dataframe,
    },
    error::{Error, ErrorKind, Result, require_non_null},
    handle::ResourceHandleLease,
    structure::typed_results::{IntegerValues, ScoreValues, VertexIds},
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
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl CentralityResult {
    /// Wrap a raw centrality result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph centrality
    /// C API function. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_centrality_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "CentralityResult::from_raw", "centrality_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    pub(crate) fn vertices_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        // SAFETY: `self.raw` is a live centrality result owned by this
        // wrapper for the duration of the call.
        let view_ptr = require_non_null(
            unsafe { cugraph_centrality_result_get_vertices(self.raw) },
            "CentralityResult::vertices_raw",
            "vertices",
        )?;
        // SAFETY: cuGraph returns freshly allocated view metadata backed by
        // the live centrality result arrays.
        unsafe {
            RawDeviceArrayView::from_owned_view(
                view_ptr,
                Some(self.device),
                "CentralityResult::vertices_raw",
                "vertices",
            )
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
        // SAFETY: cuGraph returns freshly allocated view metadata backed by
        // the live centrality result arrays.
        unsafe {
            RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), "CentralityResult::values_raw", "values")
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
        drop_device_resource(
            &mut self.raw,
            self.device,
            "CentralityResult::drop",
            DeviceResourceKind::CentralityResult,
        );
    }
}

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
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_centrality_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: the caller transfers ownership of a live cuGraph result to
        // this wrapper on success.
        Ok(Self(unsafe { CentralityResult::from_raw(raw, device, resource_handle_lease) }?))
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
    /// - `"vertex"`: vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"value"`: PageRank values with the underlying result dtype (`Float32` or `Float64`)
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        page_rank_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("PageRankResult::to_dataframe"))
    }
}

/// Result of an eigenvector centrality computation.
pub struct EigenvectorCentralityResult(CentralityResult);

impl std::fmt::Debug for EigenvectorCentralityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EigenvectorCentralityResult").finish_non_exhaustive()
    }
}

impl EigenvectorCentralityResult {
    /// Wrap a raw centrality result pointer as an `EigenvectorCentralityResult`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_eigenvector_centrality`. Ownership is transferred to this
    /// struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_centrality_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: caller upholds the contract documented on this fn, which matches
        // `CentralityResult::from_raw`'s requirements (valid non-null pointer with
        // transferred ownership).
        Ok(Self(unsafe { CentralityResult::from_raw(raw, device, resource_handle_lease) }?))
    }

    /// Get the vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.0.vertices_raw()?)
    }

    /// Get the eigenvector centrality scores as a runtime-typed device array view.
    pub fn scores(&self) -> Result<ScoreValues<'_>> {
        ScoreValues::from_raw(self.0.values_raw()?)
    }

    /// Convert this eigenvector centrality result to a cudf DataFrame.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        eigenvector_centrality_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("EigenvectorCentralityResult::to_dataframe"))
    }

    /// Number of power-iteration steps performed by the eigenvector
    /// centrality solver.
    pub fn num_iterations(&self) -> usize {
        self.0.num_iterations()
    }

    /// Whether the eigenvector centrality solver converged within the
    /// configured `max_iterations`.
    pub fn converged(&self) -> Result<bool> {
        self.0.converged()
    }
}

/// Result of a Katz centrality computation.
pub struct KatzCentralityResult(CentralityResult);

impl std::fmt::Debug for KatzCentralityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KatzCentralityResult").finish_non_exhaustive()
    }
}

impl KatzCentralityResult {
    /// Wrap a raw centrality result pointer as a `KatzCentralityResult`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_katz_centrality`. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_centrality_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: caller upholds the contract documented on this fn, which matches
        // `CentralityResult::from_raw`'s requirements (valid non-null pointer with
        // transferred ownership).
        Ok(Self(unsafe { CentralityResult::from_raw(raw, device, resource_handle_lease) }?))
    }

    /// Get the vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.0.vertices_raw()?)
    }

    /// Get the Katz centrality scores as a runtime-typed device array view.
    pub fn scores(&self) -> Result<ScoreValues<'_>> {
        ScoreValues::from_raw(self.0.values_raw()?)
    }

    /// Convert this Katz centrality result to a cudf DataFrame.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        katz_centrality_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("KatzCentralityResult::to_dataframe"))
    }

    /// Number of power-iteration steps performed by the Katz centrality
    /// solver.
    pub fn num_iterations(&self) -> usize {
        self.0.num_iterations()
    }

    /// Whether the Katz centrality solver converged within the configured
    /// `max_iterations`.
    pub fn converged(&self) -> Result<bool> {
        self.0.converged()
    }
}

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

pub(crate) struct EdgeCentralityResult {
    raw: *mut cugraph_edge_centrality_result_t,
    device: i32,
    has_edge_ids: bool,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl EdgeCentralityResult {
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph edge
    /// centrality C API function. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_edge_centrality_result_t,
        device: i32,
        has_edge_ids: bool,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "EdgeCentralityResult::from_raw", "edge_centrality_result")?;
        Ok(Self {
            raw,
            device,
            has_edge_ids,
            _resource_handle_lease: resource_handle_lease,
            _nosend_nosync: PhantomData,
        })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_edge_centrality_result_t,
        ) -> *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live edge centrality result owned by
            // this wrapper for the duration of the call.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: cuGraph returns freshly allocated view metadata backed by
        // the live edge centrality result arrays.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    pub(crate) fn sources(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_edge_centrality_result_get_src_vertices, operation, "sources")?)
    }

    pub(crate) fn destinations(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(
            cugraph_edge_centrality_result_get_dst_vertices,
            operation,
            "destinations",
        )?)
    }

    pub(crate) fn values(&self, operation: &'static str) -> Result<ScoreValues<'_>> {
        ScoreValues::from_raw(self.raw_view(cugraph_edge_centrality_result_get_values, operation, "values")?)
    }

    pub(crate) fn edge_ids(&self, operation: &'static str) -> Result<Option<IntegerValues<'_>>> {
        if !self.has_edge_ids {
            return Ok(None);
        }
        IntegerValues::from_raw(self.raw_view(cugraph_edge_centrality_result_get_edge_ids, operation, "edge_ids")?)
            .map(Some)
    }
}

impl Drop for EdgeCentralityResult {
    fn drop(&mut self) {
        drop_device_resource(
            &mut self.raw,
            self.device,
            "EdgeCentralityResult::drop",
            DeviceResourceKind::EdgeCentralityResult,
        );
    }
}

/// Result of an edge betweenness centrality computation.
pub struct EdgeBetweennessCentralityResult(EdgeCentralityResult);

impl std::fmt::Debug for EdgeBetweennessCentralityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeBetweennessCentralityResult").finish_non_exhaustive()
    }
}

impl EdgeBetweennessCentralityResult {
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_edge_betweenness_centrality`. Ownership is transferred to
    /// this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_edge_centrality_result_t,
        device: i32,
        has_edge_ids: bool,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: caller upholds the contract documented on this fn, which matches
        // `EdgeCentralityResult::from_raw`'s requirements (valid non-null pointer
        // with transferred ownership).
        Ok(Self(unsafe { EdgeCentralityResult::from_raw(raw, device, has_edge_ids, resource_handle_lease) }?))
    }

    /// Source vertices for each edge centrality score.
    pub fn sources(&self) -> Result<VertexIds<'_>> {
        self.0.sources("EdgeBetweennessCentralityResult::sources")
    }

    /// Destination vertices for each edge centrality score.
    pub fn destinations(&self) -> Result<VertexIds<'_>> {
        self.0.destinations("EdgeBetweennessCentralityResult::destinations")
    }

    /// Optional edge ids. Returns `None` for graphs built without edge ids.
    pub fn edge_ids(&self) -> Result<Option<IntegerValues<'_>>> {
        self.0.edge_ids("EdgeBetweennessCentralityResult::edge_ids")
    }

    /// Edge betweenness centrality scores.
    pub fn scores(&self) -> Result<ScoreValues<'_>> {
        self.0.values("EdgeBetweennessCentralityResult::scores")
    }

    /// Convert this edge betweenness centrality result to a cudf DataFrame.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        edge_betweenness_centrality_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("EdgeBetweennessCentralityResult::to_dataframe"))
    }
}

/// Result of a HITS computation.
pub struct HitsResult {
    raw: *mut cugraph_hits_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for HitsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HitsResult").finish_non_exhaustive()
    }
}

impl HitsResult {
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by `cugraph_hits`.
    /// Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_hits_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "HitsResult::from_raw", "hits_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_hits_result_t,
        ) -> *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live HITS result owned by this wrapper
            // for the duration of the call.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: cuGraph returns freshly allocated view metadata backed by
        // the live HITS result arrays.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    /// Vertex ids for the hub and authority scores.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_hits_result_get_vertices, "HitsResult::vertex_ids", "vertices")?)
    }

    /// Hub scores.
    pub fn hub_scores(&self) -> Result<ScoreValues<'_>> {
        ScoreValues::from_raw(self.raw_view(cugraph_hits_result_get_hubs, "HitsResult::hub_scores", "hubs")?)
    }

    /// Authority scores.
    pub fn authority_scores(&self) -> Result<ScoreValues<'_>> {
        ScoreValues::from_raw(self.raw_view(
            cugraph_hits_result_get_authorities,
            "HitsResult::authority_scores",
            "authorities",
        )?)
    }

    /// Difference between hub scores in the final two iterations.
    pub fn hub_score_differences(&self) -> f64 {
        // SAFETY: `self.raw` is a live HITS result owned by this wrapper.
        unsafe { cugraph_hits_result_get_hub_score_differences(self.raw) }
    }

    /// Number of HITS iterations performed.
    pub fn number_of_iterations(&self) -> usize {
        // SAFETY: `self.raw` is a live HITS result owned by this wrapper.
        unsafe { cugraph_hits_result_get_number_of_iterations(self.raw) }
    }

    /// Convert this HITS result to a cudf DataFrame.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        hits_result_to_dataframe(self, runtime).map_err(|e: Error| e.with_operation("HitsResult::to_dataframe"))
    }
}

impl Drop for HitsResult {
    fn drop(&mut self) {
        drop_device_resource(&mut self.raw, self.device, "HitsResult::drop", DeviceResourceKind::HitsResult);
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
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_centrality_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: the caller transfers ownership of a live cuGraph result to
        // this wrapper on success.
        Ok(Self(unsafe { CentralityResult::from_raw(raw, device, resource_handle_lease) }?))
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
    /// - `"vertex"`: vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"value"`: betweenness centrality values with the underlying result
    ///   dtype (`Float32` or `Float64`)
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        betweenness_result_to_dataframe(self, runtime)
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
        let err = match unsafe { CentralityResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "CentralityResult::from_raw");
    }

    #[test]
    fn edge_centrality_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { EdgeCentralityResult::from_raw(ptr::null_mut(), 0, false, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "EdgeCentralityResult::from_raw");
    }

    #[test]
    fn hits_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { HitsResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "HitsResult::from_raw");
    }
}
