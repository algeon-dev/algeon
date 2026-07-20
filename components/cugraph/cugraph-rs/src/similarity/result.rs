//! Similarity result wrappers.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_similarity_result_get_similarity, cugraph_similarity_result_get_vertex_pairs, cugraph_similarity_result_t,
    cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    error::{Error, Result, require_non_null},
    handle::ResourceHandleLease,
    similarity::VertexPairs,
    structure::typed_results::{ScoreValues, VertexIds},
};

pub(crate) struct SimilarityResultInner {
    raw: *mut cugraph_similarity_result_t,
    pairs: VertexPairs,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl SimilarityResultInner {
    pub(crate) unsafe fn from_pairwise_raw(
        raw: *mut cugraph_similarity_result_t,
        pairs: VertexPairs,
        device: i32,
        operation: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "similarity_result")?;
        Ok(Self { raw, pairs, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    pub(crate) unsafe fn from_all_pairs_raw(
        raw: *mut cugraph_similarity_result_t,
        device: i32,
        operation: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "similarity_result")?;
        let pairs_raw = match require_non_null(
            // SAFETY: `raw` is a live all-pairs similarity result.
            unsafe { cugraph_similarity_result_get_vertex_pairs(raw) },
            operation,
            "vertex_pairs",
        ) {
            Ok(pairs_raw) => pairs_raw,
            Err(error) => {
                let mut raw = raw;
                drop_device_resource(&mut raw, device, operation, DeviceResourceKind::SimilarityResult);
                return Err(error);
            },
        };
        // SAFETY: all-pairs similarity transfers ownership of the nested
        // vertex-pairs object, but `cugraph_similarity_result_free` does not
        // release it. Wrapping it here closes that native ownership gap.
        let pairs = unsafe { VertexPairs::from_raw(pairs_raw, device, operation, resource_handle_lease.clone()) }?;
        Ok(Self { raw, pairs, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_similarity_result_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live similarity result owned by this wrapper.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: cuGraph similarity getters allocate view metadata for the
        // caller. The backing device allocation remains owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    fn scores(&self, operation: &'static str) -> Result<ScoreValues<'_>> {
        ScoreValues::from_raw(self.raw_view(cugraph_similarity_result_get_similarity, operation, "similarity")?)
    }

    fn vertex_pairs(&self) -> &VertexPairs {
        &self.pairs
    }
}

impl Drop for SimilarityResultInner {
    fn drop(&mut self) {
        drop_device_resource(
            &mut self.raw,
            self.device,
            "SimilarityResultInner::drop",
            DeviceResourceKind::SimilarityResult,
        );
    }
}

macro_rules! define_similarity_result {
    ($name:ident, $score_op:literal, $df_fn:ident) => {
        /// Result of a similarity coefficient computation.
        pub struct $name(pub(crate) SimilarityResultInner);

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }

        impl $name {
            pub(crate) fn from_inner(inner: SimilarityResultInner) -> Self {
                Self(inner)
            }

            /// Get the first vertex id column.
            pub fn first(&self) -> Result<VertexIds<'_>> {
                self.0.vertex_pairs().first()
            }

            /// Get the second vertex id column.
            pub fn second(&self) -> Result<VertexIds<'_>> {
                self.0.vertex_pairs().second()
            }

            /// Get the similarity score column.
            pub fn scores(&self) -> Result<ScoreValues<'_>> {
                self.0.scores($score_op)
            }

            /// Get the vertex pairs associated with this result.
            pub fn vertex_pairs(&self) -> &VertexPairs {
                self.0.vertex_pairs()
            }

            /// Convert this result to a cudf DataFrame with `"first"`,
            /// `"second"`, and `"similarity"` columns.
            pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
                crate::cudf_interop::$df_fn(self, runtime)
                    .map_err(|error: Error| error.with_operation(concat!(stringify!($name), "::to_dataframe")))
            }
        }
    };
}

define_similarity_result!(
    JaccardSimilarityResult,
    "JaccardSimilarityResult::scores",
    jaccard_similarity_result_to_dataframe
);
define_similarity_result!(
    SorensenSimilarityResult,
    "SorensenSimilarityResult::scores",
    sorensen_similarity_result_to_dataframe
);
define_similarity_result!(
    OverlapSimilarityResult,
    "OverlapSimilarityResult::scores",
    overlap_similarity_result_to_dataframe
);
define_similarity_result!(
    CosineSimilarityResult,
    "CosineSimilarityResult::scores",
    cosine_similarity_result_to_dataframe
);

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn similarity_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err =
            match unsafe { SimilarityResultInner::from_all_pairs_raw(ptr::null_mut(), 0, "test_similarity", None) } {
                Err(err) => err,
                Ok(_) => panic!("null must be rejected"),
            };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_similarity");
    }
}
