//! Result types for community-detection algorithms.
//!
//! Louvain, Leiden, and ECG return `cugraph_hierarchical_clustering_result_t`.
//! This module keeps ownership and raw accessor logic in an internal wrapper so
//! hierarchical community algorithms can reuse it.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_hierarchical_clustering_result_get_clusters, cugraph_hierarchical_clustering_result_get_modularity,
    cugraph_hierarchical_clustering_result_get_vertices, cugraph_hierarchical_clustering_result_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    error::{Error, Result, require_non_null},
    handle::ResourceHandleLease,
    structure::typed_results::VertexIds,
};

/// Internal shared wrapper for `cugraph_hierarchical_clustering_result_t`.
pub(crate) struct HierarchicalClusteringResult {
    raw: *mut cugraph_hierarchical_clustering_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl HierarchicalClusteringResult {
    /// Wrap a raw hierarchical clustering result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph
    /// hierarchical clustering C API. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_hierarchical_clustering_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "HierarchicalClusteringResult::from_raw", "hierarchical_clustering_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    pub(crate) fn vertices_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live hierarchical clustering result
            // owned by this wrapper for the duration of the call.
            unsafe { cugraph_hierarchical_clustering_result_get_vertices(self.raw) },
            "HierarchicalClusteringResult::vertices_raw",
            "vertices",
        )?;
        // SAFETY: cuGraph hierarchical clustering getters allocate view
        // metadata for the caller. The backing device allocation remains owned
        // by `self.raw`.
        unsafe {
            RawDeviceArrayView::from_owned_view(
                view_ptr,
                Some(self.device),
                "HierarchicalClusteringResult::vertices_raw",
                "vertices",
            )
        }
    }

    pub(crate) fn clusters_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live hierarchical clustering result
            // owned by this wrapper for the duration of the call.
            unsafe { cugraph_hierarchical_clustering_result_get_clusters(self.raw) },
            "HierarchicalClusteringResult::clusters_raw",
            "clusters",
        )?;
        // SAFETY: cuGraph hierarchical clustering getters allocate view
        // metadata for the caller. The backing device allocation remains owned
        // by `self.raw`.
        unsafe {
            RawDeviceArrayView::from_owned_view(
                view_ptr,
                Some(self.device),
                "HierarchicalClusteringResult::clusters_raw",
                "clusters",
            )
        }
    }

    pub(crate) fn modularity(&self) -> f64 {
        // SAFETY: `self.raw` is a live hierarchical clustering result owned by
        // this wrapper for the duration of the call.
        unsafe { cugraph_hierarchical_clustering_result_get_modularity(self.raw) }
    }
}

impl Drop for HierarchicalClusteringResult {
    fn drop(&mut self) {
        drop_device_resource(
            &mut self.raw,
            self.device,
            "HierarchicalClusteringResult::drop",
            DeviceResourceKind::HierarchicalClusteringResult,
        );
    }
}

macro_rules! define_hierarchical_result {
    ($name:ident, $modularity_doc:literal, $df_fn:ident) => {
        /// Result of a hierarchical community-detection computation.
        ///
        /// Provides access to vertex ids, community assignments, and the
        /// scalar modularity score. Freed on drop.
        pub struct $name(HierarchicalClusteringResult);

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }

        impl $name {
            /// Wrap a raw hierarchical clustering result pointer.
            ///
            /// # Safety
            ///
            /// `raw` must be a valid, non-null pointer returned by the
            /// matching cuGraph C API. Ownership is transferred to this struct.
            pub(crate) unsafe fn from_raw(
                raw: *mut cugraph_hierarchical_clustering_result_t,
                device: i32,
                resource_handle_lease: Option<ResourceHandleLease>,
            ) -> Result<Self> {
                // SAFETY: the caller transfers ownership of a live cuGraph
                // result to this wrapper on success.
                Ok(Self(unsafe { HierarchicalClusteringResult::from_raw(raw, device, resource_handle_lease) }?))
            }

            /// Get the vertex ids as a runtime-typed device array view.
            ///
            /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on
            /// the graph's vertex type.
            pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
                VertexIds::from_raw(self.0.vertices_raw()?)
            }

            /// Get the per-vertex cluster assignments as a runtime-typed
            /// device array view.
            ///
            /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on
            /// the graph's vertex type.
            pub fn clusters(&self) -> Result<VertexIds<'_>> {
                VertexIds::from_raw(self.0.clusters_raw()?)
            }

            #[doc = $modularity_doc]
            pub fn modularity(&self) -> f64 {
                self.0.modularity()
            }

            /// Convert this result to a cudf DataFrame.
            ///
            /// Returns a 2-column DataFrame:
            /// - `"vertex"`: vertex ids with the underlying result dtype
            /// - `"partition"`: cluster assignments with the underlying result dtype
            ///
            /// Device-to-device copy: unavoidable because the cugraph C API
            /// owns result memory and frees it on drop. Matches Python
            /// cudf/cugraph behavior.
            pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
                crate::cudf_interop::$df_fn(self, runtime)
                    .map_err(|e: Error| e.with_operation(concat!(stringify!($name), "::to_dataframe")))
            }
        }
    };
}

define_hierarchical_result!(LouvainResult, "Modularity score returned by Louvain.", louvain_result_to_dataframe);
define_hierarchical_result!(LeidenResult, "Modularity score returned by Leiden.", leiden_result_to_dataframe);
define_hierarchical_result!(EcgResult, "Modularity score returned by ECG.", ecg_result_to_dataframe);

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn hierarchical_clustering_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { HierarchicalClusteringResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "HierarchicalClusteringResult::from_raw");
    }
}
