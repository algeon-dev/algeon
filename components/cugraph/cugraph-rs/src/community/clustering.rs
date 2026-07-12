//! Legacy spectral clustering APIs.

use std::{num::NonZeroUsize, ptr};

use cugraph_sys::{
    cugraph_analyze_clustering_edge_cut, cugraph_analyze_clustering_modularity, cugraph_analyze_clustering_ratio_cut,
    cugraph_balanced_cut_clustering, cugraph_clustering_result_free, cugraph_clustering_result_get_clusters,
    cugraph_clustering_result_get_vertices, cugraph_clustering_result_t, cugraph_error_t,
    cugraph_spectral_modularity_maximization,
};

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::{EnteredContext, ResourceHandleLease},
    params::{IterationCount, NonNegativeF64},
    random::RngState,
    structure::VertexIds,
    types::{CugraphElement, to_bool_t},
};

/// Parameters shared by legacy spectral clustering algorithms.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct SpectralClusteringParams {
    /// Desired number of clusters.
    pub n_clusters: NonZeroUsize,
    /// Number of eigenvectors to use.
    pub n_eigenvectors: NonZeroUsize,
    /// Eigenvalue-solver tolerance.
    pub evs_tolerance: NonNegativeF64,
    /// Eigenvalue-solver maximum iteration count.
    pub evs_max_iterations: IterationCount,
    /// k-means tolerance.
    pub k_means_tolerance: NonNegativeF64,
    /// k-means maximum iteration count.
    pub k_means_max_iterations: IterationCount,
    /// Whether to run expensive input validation checks in the C library.
    pub do_expensive_check: bool,
}

impl Default for SpectralClusteringParams {
    fn default() -> Self {
        Self {
            n_clusters: NonZeroUsize::new(2).expect("literal is non-zero"),
            n_eigenvectors: NonZeroUsize::new(2).expect("literal is non-zero"),
            evs_tolerance: NonNegativeF64::new(0.01).expect("literal is non-negative"),
            evs_max_iterations: IterationCount::new(500).expect("literal is non-zero"),
            k_means_tolerance: NonNegativeF64::new(0.001).expect("literal is non-negative"),
            k_means_max_iterations: IterationCount::new(100).expect("literal is non-zero"),
            do_expensive_check: false,
        }
    }
}

/// Parameters for deprecated balanced-cut clustering.
#[deprecated(note = "cuGraph deprecates balanced cut clustering; use spectral_modularity_maximization instead")]
pub type BalancedCutClusteringParams = SpectralClusteringParams;

pub(crate) struct ClusteringResult {
    raw: *mut cugraph_clustering_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl ClusteringResult {
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph
    /// clustering C API. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_clustering_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "ClusteringResult::from_raw", "clustering_result")?;
        Ok(Self {
            raw,
            device,
            _resource_handle_lease: resource_handle_lease,
            _nosend_nosync: std::marker::PhantomData,
        })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_clustering_result_t,
        ) -> *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live clustering result owned by this
            // wrapper for the duration of the call.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: cuGraph clustering getters allocate view metadata for the
        // caller. The backing device allocation remains owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    fn vertices(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_clustering_result_get_vertices, operation, "vertices")?)
    }

    fn clusters(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_clustering_result_get_clusters, operation, "clusters")?)
    }
}

impl Drop for ClusteringResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "ClusteringResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and has not been
            // freed yet.
            unsafe { cugraph_clustering_result_free(self.raw) };
        }
    }
}

macro_rules! define_clustering_result {
    ($name:ident, $df_fn:ident) => {
        /// Result of a legacy spectral clustering computation.
        pub struct $name(ClusteringResult);

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }

        impl $name {
            /// # Safety
            ///
            /// `raw` must be a valid, non-null pointer returned by the
            /// matching cuGraph C API. Ownership is transferred to this
            /// struct.
            pub(crate) unsafe fn from_raw(
                raw: *mut cugraph_clustering_result_t,
                device: i32,
                resource_handle_lease: Option<ResourceHandleLease>,
            ) -> Result<Self> {
                // SAFETY: caller upholds the contract documented on this fn, which
                // matches `ClusteringResult::from_raw`'s requirements (valid non-null
                // pointer with transferred ownership).
                Ok(Self(unsafe { ClusteringResult::from_raw(raw, device, resource_handle_lease) }?))
            }

            /// Vertex ids in the clustering output.
            pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
                self.0.vertices(concat!(stringify!($name), "::vertex_ids"))
            }

            /// Per-vertex cluster assignments.
            pub fn clusters(&self) -> Result<VertexIds<'_>> {
                self.0.clusters(concat!(stringify!($name), "::clusters"))
            }

            /// Convert this result to a cudf DataFrame.
            pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
                crate::cudf_interop::$df_fn(self, runtime)
                    .map_err(|e: Error| e.with_operation(concat!(stringify!($name), "::to_dataframe")))
            }
        }
    };
}

define_clustering_result!(SpectralClusteringResult, spectral_clustering_result_to_dataframe);

#[deprecated(
    note = "cuGraph deprecates balanced cut clustering; use SpectralClusteringResult from spectral_modularity_maximization instead"
)]
pub struct BalancedCutClusteringResult(ClusteringResult);

#[allow(deprecated)]
impl std::fmt::Debug for BalancedCutClusteringResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BalancedCutClusteringResult").finish_non_exhaustive()
    }
}

#[allow(deprecated)]
impl BalancedCutClusteringResult {
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_balanced_cut_clustering`. Ownership is transferred to this
    /// struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_clustering_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: caller upholds the contract documented on this fn, which matches
        // `ClusteringResult::from_raw`'s requirements (valid non-null pointer with
        // transferred ownership).
        Ok(Self(unsafe { ClusteringResult::from_raw(raw, device, resource_handle_lease) }?))
    }

    /// Vertex ids in the clustering output.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.0.vertices("BalancedCutClusteringResult::vertex_ids")
    }

    /// Per-vertex cluster assignments.
    pub fn clusters(&self) -> Result<VertexIds<'_>> {
        self.0.clusters("BalancedCutClusteringResult::clusters")
    }

    /// Convert this result to a cudf DataFrame.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::balanced_cut_clustering_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("BalancedCutClusteringResult::to_dataframe"))
    }
}

pub(crate) fn spectral_modularity_maximization(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    params: &SpectralClusteringParams,
) -> Result<SpectralClusteringResult> {
    let raw = run_clustering(
        handle,
        rng_state,
        graph,
        params,
        cugraph_spectral_modularity_maximization,
        "spectral_modularity_maximization",
    )?;
    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { SpectralClusteringResult::from_raw(raw, handle.device(), handle.runtime_bound_lease()) }
}

#[allow(deprecated)]
pub(crate) fn balanced_cut_clustering(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    params: &BalancedCutClusteringParams,
) -> Result<BalancedCutClusteringResult> {
    let raw =
        run_clustering(handle, rng_state, graph, params, cugraph_balanced_cut_clustering, "balanced_cut_clustering")?;
    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { BalancedCutClusteringResult::from_raw(raw, handle.device(), handle.runtime_bound_lease()) }
}

type ClusteringFn = unsafe extern "C" fn(
    *const cugraph_sys::cugraph_resource_handle_t,
    *mut cugraph_sys::cugraph_rng_state_t,
    *mut cugraph_sys::cugraph_graph_t,
    usize,
    usize,
    f64,
    i32,
    f64,
    i32,
    cugraph_sys::bool_t,
    *mut *mut cugraph_clustering_result_t,
    *mut *mut cugraph_error_t,
) -> cugraph_sys::cugraph_error_code_t;

fn run_clustering(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    params: &SpectralClusteringParams,
    function: ClusteringFn,
    operation: &'static str,
) -> Result<*mut cugraph_clustering_result_t> {
    let evs_max_iterations = i32::try_from(params.evs_max_iterations)?;
    let k_means_max_iterations = i32::try_from(params.k_means_max_iterations)?;
    let mut result: *mut cugraph_clustering_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, operation)?;

    // SAFETY: `graph` and `rng_state` are live cuGraph objects, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = function(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            params.n_clusters.get(),
            params.n_eigenvectors.get(),
            params.evs_tolerance.get(),
            evs_max_iterations,
            params.k_means_tolerance.get(),
            k_means_max_iterations,
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, operation)?;
    }

    require_non_null(result, operation, "clustering_result")
}

pub(crate) fn analyze_clustering_modularity<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    n_clusters: NonZeroUsize,
    vertices: &DeviceArrayView<'_, V>,
    clusters: &DeviceArrayView<'_, V>,
) -> Result<f64> {
    analyze_clustering(
        handle,
        graph,
        n_clusters,
        vertices,
        clusters,
        cugraph_analyze_clustering_modularity,
        "analyze_clustering_modularity",
    )
}

pub(crate) fn analyze_clustering_edge_cut<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    n_clusters: NonZeroUsize,
    vertices: &DeviceArrayView<'_, V>,
    clusters: &DeviceArrayView<'_, V>,
) -> Result<f64> {
    analyze_clustering(
        handle,
        graph,
        n_clusters,
        vertices,
        clusters,
        cugraph_analyze_clustering_edge_cut,
        "analyze_clustering_edge_cut",
    )
}

pub(crate) fn analyze_clustering_ratio_cut<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    n_clusters: NonZeroUsize,
    vertices: &DeviceArrayView<'_, V>,
    clusters: &DeviceArrayView<'_, V>,
) -> Result<f64> {
    analyze_clustering(
        handle,
        graph,
        n_clusters,
        vertices,
        clusters,
        cugraph_analyze_clustering_ratio_cut,
        "analyze_clustering_ratio_cut",
    )
}

type AnalyzeFn = unsafe extern "C" fn(
    *const cugraph_sys::cugraph_resource_handle_t,
    *mut cugraph_sys::cugraph_graph_t,
    usize,
    *const cugraph_sys::cugraph_type_erased_device_array_view_t,
    *const cugraph_sys::cugraph_type_erased_device_array_view_t,
    *mut f64,
    *mut *mut cugraph_error_t,
) -> cugraph_sys::cugraph_error_code_t;

fn analyze_clustering<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    n_clusters: NonZeroUsize,
    vertices: &DeviceArrayView<'_, V>,
    clusters: &DeviceArrayView<'_, V>,
    function: AnalyzeFn,
    operation: &'static str,
) -> Result<f64> {
    graph.validate_vertex_input_dtype(V::data_type(), operation, "vertices")?;
    if vertices.len() != clusters.len() {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message("vertices and clusters must have the same length")
            .with_context("vertices_len", vertices.len().to_string())
            .with_context("clusters_len", clusters.len().to_string()));
    }

    let mut score = 0.0f64;
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, operation)?;

    // SAFETY: `vertices` and `clusters` are live device views matching the
    // graph vertex dtype, `graph` is live, and `score`/`error` are valid
    // out-pointers.
    unsafe {
        let code = function(
            handle.handle_ptr(),
            graph_ptr,
            n_clusters.get(),
            vertices.as_ptr(),
            clusters.as_ptr(),
            &mut score,
            &mut error,
        );
        check_error(code, error, operation)?;
    }

    Ok(score)
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn clustering_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { ClusteringResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "ClusteringResult::from_raw");
    }
}
