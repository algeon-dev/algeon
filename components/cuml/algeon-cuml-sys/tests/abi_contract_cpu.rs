use cuml_sys::*;

#[test]
fn installed_cuml_runtime_version_matches_bindings() {
    assert_eq!(validate_runtime_version().expect("installed cuML version"), EXPECTED_CUML_VERSION);
    assert_eq!(
        unsafe { cuml_version_get(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) },
        CUML_ERROR_NULL
    );

    let (mut major, mut minor, mut patch) = (0, 0, 0);
    assert_eq!(unsafe { cuml_forest_inference_version_get(&mut major, &mut minor, &mut patch) }, CUML_SUCCESS);
    let (expected_major, expected_minor, expected_patch) = EXPECTED_CUML_VERSION;
    assert_eq!(
        (major, minor, patch),
        (i32::from(expected_major), i32::from(expected_minor), i32::from(expected_patch))
    );
}
