use cuml_sys::*;

#[test]
fn error_status_and_detail_abi_are_fixed() {
    assert_eq!(CUML_SUCCESS, 0);
    assert_eq!(CUML_ERROR_NULL, 1);
    assert_eq!(CUML_ERROR_INVALID_ARGUMENT, 2);
    assert_eq!(CUML_ERROR_INVALID_STATE, 3);
    assert_eq!(CUML_ERROR_OUT_OF_MEMORY, 4);
    assert_eq!(CUML_ERROR_HOST_OUT_OF_MEMORY, 5);
    assert_eq!(CUML_ERROR_CUDA, 6);
    assert_eq!(CUML_ERROR_FATAL_CUDA, 7);
    assert_eq!(CUML_ERROR_SIZE_OVERFLOW, 8);
    assert_eq!(CUML_ERROR_UNSUPPORTED, 9);
    assert_eq!(CUML_ERROR_UNKNOWN, 10);
    assert_eq!(core::mem::size_of::<cuml_error_detail_t>(), 568);
    assert_eq!(core::mem::align_of::<cuml_error_detail_t>(), 8);
    assert_eq!(core::mem::offset_of!(cuml_error_detail_t, message), 52);
}
