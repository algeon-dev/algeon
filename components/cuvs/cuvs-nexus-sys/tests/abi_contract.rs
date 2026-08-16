use std::{
    ffi::c_int,
    mem::{align_of, offset_of, size_of},
};

use cuvs_sys::{
    DLManagedTensor, EXPECTED_CUVS_VERSION, cuvsError_t, cuvsKMeansFit, cuvsKMeansParams, cuvsKMeansParams_t,
    cuvsKMeansParamsCreate, cuvsKMeansParamsDestroy, cuvsKMeansPredict, cuvsPcaParams, cuvsResources_t, cuvsVersionGet,
    validate_runtime_version,
};

unsafe extern "C" {
    fn cuvs_nexus_sizeof_kmeans_params() -> usize;
    fn cuvs_nexus_alignof_kmeans_params() -> usize;
    fn cuvs_nexus_offsetof_kmeans_device_buffer_samples() -> usize;
    fn cuvs_nexus_sizeof_pca_params() -> usize;
    fn cuvs_nexus_alignof_pca_params() -> usize;
}

#[test]
fn generated_rust_layout_matches_the_pinned_c_headers() {
    assert_eq!(size_of::<cuvsKMeansParams>(), 72);
    assert_eq!(align_of::<cuvsKMeansParams>(), 8);
    assert_eq!(offset_of!(cuvsKMeansParams, hierarchical), 48);
    assert_eq!(offset_of!(cuvsKMeansParams, device_buffer_samples), 56);
    assert_eq!(offset_of!(cuvsKMeansParams, init_size), 64);
    assert_eq!(size_of::<cuvsPcaParams>(), 20);
    assert_eq!(align_of::<cuvsPcaParams>(), 4);

    unsafe {
        assert_eq!(cuvs_nexus_sizeof_kmeans_params(), size_of::<cuvsKMeansParams>());
        assert_eq!(cuvs_nexus_alignof_kmeans_params(), align_of::<cuvsKMeansParams>());
        assert_eq!(
            cuvs_nexus_offsetof_kmeans_device_buffer_samples(),
            offset_of!(cuvsKMeansParams, device_buffer_samples)
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
fn generated_bindings_expose_the_pinned_kmeans_function_surface() {
    let _: unsafe extern "C" fn(*mut cuvsKMeansParams_t) -> cuvsError_t = cuvsKMeansParamsCreate;
    let _: unsafe extern "C" fn(cuvsKMeansParams_t) -> cuvsError_t = cuvsKMeansParamsDestroy;
    let _: unsafe extern "C" fn(
        cuvsResources_t,
        cuvsKMeansParams_t,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut f64,
        *mut c_int,
    ) -> cuvsError_t = cuvsKMeansFit;
    let _: unsafe extern "C" fn(
        cuvsResources_t,
        cuvsKMeansParams_t,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        bool,
        *mut f64,
    ) -> cuvsError_t = cuvsKMeansPredict;
    let _: unsafe extern "C" fn(*mut u16, *mut u16, *mut u16) -> cuvsError_t = cuvsVersionGet;
}
