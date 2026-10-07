use std::ffi::{CStr, CString};

use cuml_sys::{
    CUML_ERROR_CUDA, CUML_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT, CUML_ERROR_FAILURE_SOURCE_UNKNOWN,
    CUML_ERROR_HOST_OUT_OF_MEMORY, CUML_ERROR_INVALID_ARGUMENT, CUML_ERROR_NULL, CUML_ERROR_OUT_OF_MEMORY,
    CUML_ERROR_SIZE_OVERFLOW, CUML_ERROR_UNKNOWN, CUML_SUCCESS, cuml_clear_error, cuml_error_detail_t,
    cuml_handle_create, cuml_handle_t, cuml_last_error_detail, cuml_umap_fit_params, cuml_umap_transform_params,
    test_support::cuml_test_raise_error,
};

unsafe extern "C" {
    fn cuml_test_raise_exception(kind: i32) -> u32;
}

fn detail() -> cuml_error_detail_t {
    let pointer = unsafe { cuml_last_error_detail() };
    assert!(!pointer.is_null());
    unsafe { *pointer }
}

fn message(detail: &cuml_error_detail_t) -> String {
    unsafe { CStr::from_ptr(detail.message.as_ptr()) }.to_str().unwrap().to_owned()
}

#[test]
fn native_error_boundary_copies_status_and_message_into_tls_detail() {
    let text = CString::new("native error detail contract").unwrap();
    let status = unsafe { cuml_test_raise_error(CUML_ERROR_INVALID_ARGUMENT, text.as_ptr()) };
    assert_eq!(status, CUML_ERROR_INVALID_ARGUMENT);
    let detail = detail();
    assert_eq!(detail.status, CUML_ERROR_INVALID_ARGUMENT);
    assert_eq!(message(&detail), "native error detail contract");

    let unknown = CString::new("unknown detail").unwrap();
    assert_eq!(unsafe { cuml_test_raise_error(CUML_ERROR_UNKNOWN, unknown.as_ptr()) }, CUML_ERROR_UNKNOWN);
    unsafe { cuml_clear_error() };
    assert!(unsafe { cuml_last_error_detail() }.is_null());
}

#[test]
fn umap_null_arguments_replace_previous_native_error_detail() {
    let previous = CString::new("previous allocation failure").unwrap();
    unsafe { cuml_test_raise_error(CUML_ERROR_OUT_OF_MEMORY, previous.as_ptr()) };
    let status = unsafe {
        cuml_umap_fit_params(std::ptr::null_mut(), std::ptr::null(), 0, 0, std::ptr::null(), std::ptr::null_mut())
    };
    assert_eq!(status, CUML_ERROR_NULL);
    let fit_detail = detail();
    assert_eq!(fit_detail.status, CUML_ERROR_NULL);
    assert!(message(&fit_detail).contains("UMAP fit"));
    let status = unsafe {
        cuml_umap_transform_params(
            std::ptr::null_mut(),
            std::ptr::null(),
            0,
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(status, CUML_ERROR_NULL);
    let transform_detail = detail();
    assert_eq!(transform_detail.status, CUML_ERROR_NULL);
    assert!(message(&transform_detail).contains("UMAP transform"));
}

#[test]
fn native_exception_translation_populates_and_clears_error_details() {
    let cases = [
        (0, CUML_ERROR_OUT_OF_MEMORY, CUML_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT),
        (1, CUML_ERROR_OUT_OF_MEMORY, CUML_ERROR_FAILURE_SOURCE_UNKNOWN),
        (2, CUML_ERROR_OUT_OF_MEMORY, CUML_ERROR_FAILURE_SOURCE_UNKNOWN),
        (3, CUML_ERROR_CUDA, CUML_ERROR_FAILURE_SOURCE_UNKNOWN),
        (4, CUML_ERROR_INVALID_ARGUMENT, CUML_ERROR_FAILURE_SOURCE_UNKNOWN),
        (5, CUML_ERROR_HOST_OUT_OF_MEMORY, CUML_ERROR_FAILURE_SOURCE_UNKNOWN),
        (6, CUML_ERROR_SIZE_OVERFLOW, CUML_ERROR_FAILURE_SOURCE_UNKNOWN),
        (7, CUML_ERROR_INVALID_ARGUMENT, CUML_ERROR_FAILURE_SOURCE_UNKNOWN),
        (8, CUML_ERROR_UNKNOWN, CUML_ERROR_FAILURE_SOURCE_UNKNOWN),
    ];

    for (kind, expected_status, expected_source) in cases {
        let status = unsafe { cuml_test_raise_exception(kind) };
        assert_eq!(status, expected_status, "exception kind {kind}");
        let detail = detail();
        assert_eq!(detail.status, expected_status, "exception kind {kind}");
        assert_eq!(detail.failure_source, expected_source, "exception kind {kind}");
        assert!(!message(&detail).is_empty(), "exception kind {kind}");
        if kind == 0 {
            assert_eq!((detail.domain_id, detail.member_id, detail.generation), (11, 17, 23));
            assert_eq!((detail.requested_bytes, detail.alignment, detail.cuda_error), (4096, 256, 2));
        } else if kind == 1 || kind == 2 || kind == 5 {
            assert_eq!((detail.domain_id, detail.member_id, detail.generation), (0, 0, 0));
            assert_eq!((detail.requested_bytes, detail.alignment, detail.cuda_error), (0, 0, -1));
        }
    }

    let status = unsafe {
        cuml_handle_create(0, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
    };
    assert_eq!(status, CUML_ERROR_NULL);
    let null_call_detail = detail();
    assert_eq!(null_call_detail.status, CUML_ERROR_NULL);
    assert_eq!(message(&null_call_detail), "handle output is null");

    let mut out_handle: cuml_handle_t = std::ptr::dangling_mut::<u8>().cast();
    let status = unsafe {
        cuml_handle_create(
            0,
            std::ptr::null_mut(),
            std::ptr::dangling_mut::<u8>().cast(),
            std::ptr::dangling_mut::<u8>().cast(),
            &mut out_handle,
        )
    };
    assert_eq!(status, CUML_ERROR_NULL);
    assert!(out_handle.is_null());
    assert_eq!(detail().status, CUML_ERROR_NULL);

    unsafe { cuml_clear_error() };
    assert!(unsafe { cuml_last_error_detail() }.is_null());
    assert_eq!(CUML_SUCCESS, 0);
}
