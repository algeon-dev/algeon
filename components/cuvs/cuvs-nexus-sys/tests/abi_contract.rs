use std::{
    ffi::c_int,
    mem::{align_of, offset_of, size_of},
};

use cuvs_sys::{
    DLManagedTensor, EXPECTED_CUVS_VERSION, cuvsError_t, cuvsKMeansFit_v2, cuvsKMeansParams_v2, cuvsKMeansParams_v2_t,
    cuvsKMeansParamsCreate_v2, cuvsKMeansParamsDestroy_v2, cuvsKMeansPredict_v2, cuvsPcaParams, cuvsResources_t,
    cuvsVersionGet, validate_runtime_version,
};

unsafe extern "C" {
    fn cuvs_nexus_sizeof_kmeans_params_v2() -> usize;
    fn cuvs_nexus_alignof_kmeans_params_v2() -> usize;
    fn cuvs_nexus_offsetof_kmeans_streaming_batch_size_v2() -> usize;
    fn cuvs_nexus_sizeof_pca_params() -> usize;
    fn cuvs_nexus_alignof_pca_params() -> usize;
}

#[test]
fn generated_rust_layout_matches_the_pinned_c_headers() {
    assert_eq!(size_of::<cuvsKMeansParams_v2>(), 72);
    assert_eq!(align_of::<cuvsKMeansParams_v2>(), 8);
    assert_eq!(offset_of!(cuvsKMeansParams_v2, hierarchical), 48);
    assert_eq!(offset_of!(cuvsKMeansParams_v2, streaming_batch_size), 56);
    assert_eq!(offset_of!(cuvsKMeansParams_v2, init_size), 64);
    assert_eq!(size_of::<cuvsPcaParams>(), 20);
    assert_eq!(align_of::<cuvsPcaParams>(), 4);

    unsafe {
        assert_eq!(cuvs_nexus_sizeof_kmeans_params_v2(), size_of::<cuvsKMeansParams_v2>());
        assert_eq!(cuvs_nexus_alignof_kmeans_params_v2(), align_of::<cuvsKMeansParams_v2>());
        assert_eq!(
            cuvs_nexus_offsetof_kmeans_streaming_batch_size_v2(),
            offset_of!(cuvsKMeansParams_v2, streaming_batch_size)
        );
        assert_eq!(cuvs_nexus_sizeof_pca_params(), size_of::<cuvsPcaParams>());
        assert_eq!(cuvs_nexus_alignof_pca_params(), align_of::<cuvsPcaParams>());
    }
}

#[test]
fn loaded_runtime_matches_the_exact_binding_version() {
    assert_eq!(validate_runtime_version().expect("cuvsVersionGet must succeed"), EXPECTED_CUVS_VERSION);
}

#[test]
fn generated_bindings_expose_the_pinned_kmeans_v2_function_surface() {
    let _: unsafe extern "C" fn(*mut cuvsKMeansParams_v2_t) -> cuvsError_t = cuvsKMeansParamsCreate_v2;
    let _: unsafe extern "C" fn(cuvsKMeansParams_v2_t) -> cuvsError_t = cuvsKMeansParamsDestroy_v2;
    let _: unsafe extern "C" fn(
        cuvsResources_t,
        cuvsKMeansParams_v2_t,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut f64,
        *mut c_int,
    ) -> cuvsError_t = cuvsKMeansFit_v2;
    let _: unsafe extern "C" fn(
        cuvsResources_t,
        cuvsKMeansParams_v2_t,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        bool,
        *mut f64,
    ) -> cuvsError_t = cuvsKMeansPredict_v2;
    let _: unsafe extern "C" fn(*mut u16, *mut u16, *mut u16) -> cuvsError_t = cuvsVersionGet;
}
