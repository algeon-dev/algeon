use std::ptr;

#[test]
fn metadata_only_owner_preflight_returns_checked_components() {
    let mut hash_table_bytes = 0usize;
    let mut key_view_bytes = 0usize;
    let status = unsafe {
        cudf_sys::cudf_hash_join_metadata_only_owner_preflight_ex(
            1024,
            0.5,
            2,
            &mut hash_table_bytes,
            &mut key_view_bytes,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(hash_table_bytes > 0);
    assert!(key_view_bytes > 0);

    let mut one_view_bytes = 0usize;
    let mut one_hash_table_bytes = 0usize;
    let status = unsafe {
        cudf_sys::cudf_hash_join_metadata_only_owner_preflight_ex(
            1024,
            0.5,
            1,
            &mut one_hash_table_bytes,
            &mut one_view_bytes,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(one_hash_table_bytes, hash_table_bytes);
    assert!(key_view_bytes > one_view_bytes);
}

#[test]
fn metadata_only_owner_preflight_rejects_invalid_load_factor() {
    for load_factor in [0.0, -0.5, 1.000_001, f64::NAN] {
        let mut hash_table_bytes = 0usize;
        let mut key_view_bytes = 0usize;
        let status = unsafe {
            cudf_sys::cudf_hash_join_metadata_only_owner_preflight_ex(
                1024,
                load_factor,
                1,
                &mut hash_table_bytes,
                &mut key_view_bytes,
            )
        };
        assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
        assert_eq!(hash_table_bytes, 0);
        assert_eq!(key_view_bytes, 0);
    }
}

#[test]
fn metadata_only_owner_preflight_rejects_row_count_overflow() {
    let mut hash_table_bytes = 0usize;
    let mut key_view_bytes = 0usize;
    let row_overflow = usize::try_from(i64::from(i32::MAX) + 1).expect("test platform is 64-bit");
    let status = unsafe {
        cudf_sys::cudf_hash_join_metadata_only_owner_preflight_ex(
            row_overflow,
            0.5,
            1,
            &mut hash_table_bytes,
            &mut key_view_bytes,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_SIZE_OVERFLOW);
    assert_eq!(hash_table_bytes, 0);
    assert_eq!(key_view_bytes, 0);
}

#[test]
fn metadata_only_owner_preflight_rejects_key_view_count_overflow() {
    let mut hash_table_bytes = 0usize;
    let mut key_view_bytes = 0usize;
    let status = unsafe {
        cudf_sys::cudf_hash_join_metadata_only_owner_preflight_ex(
            1024,
            0.5,
            usize::MAX,
            &mut hash_table_bytes,
            &mut key_view_bytes,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_SIZE_OVERFLOW);
    assert_eq!(hash_table_bytes, 0);
    assert_eq!(key_view_bytes, 0);
}

#[test]
fn metadata_only_owner_preflight_rejects_null_outputs() {
    let mut key_view_bytes = 0usize;
    let status = unsafe {
        cudf_sys::cudf_hash_join_metadata_only_owner_preflight_ex(1024, 0.5, 1, ptr::null_mut(), &mut key_view_bytes)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    let mut hash_table_bytes = 0usize;
    let status = unsafe {
        cudf_sys::cudf_hash_join_metadata_only_owner_preflight_ex(1024, 0.5, 1, &mut hash_table_bytes, ptr::null_mut())
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
}
