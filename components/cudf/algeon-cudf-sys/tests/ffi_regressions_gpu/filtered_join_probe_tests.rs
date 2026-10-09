use std::ptr;

use crate::{
    helpers::{make_i32_column, make_table_from_one_column, make_table_view, read_i32_column_data_with_exec},
    support::{
        OwnedMemoryResource, OwnedStream, last_exec_mr_raw, last_exec_stream_value, required_exec_options,
        reset_exec_observation, stream_value,
    },
};

type BeginProbe = unsafe extern "C" fn(
    cudf_sys::cudf_filtered_join_handle,
    cudf_sys::cudf_table_view_handle,
    *mut cudf_sys::cudf_filtered_join_probe_handle,
    *const cudf_sys::cudf_exec_options_t,
) -> cudf_sys::cudf_status_t;

fn column_len(col: cudf_sys::cudf_column_handle) -> usize {
    let view = {
        let mut out: cudf_sys::cudf_column_view_handle = ptr::null_mut();
        let status = unsafe { cudf_sys::cudf_column_view(col, &mut out) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        assert!(!out.is_null());
        out
    };
    let mut size = 0;
    let status = unsafe { cudf_sys::cudf_column_view_size(view, &mut size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(size >= 0);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    size as usize
}

fn read_all_i32(col: cudf_sys::cudf_column_handle, exec: &cudf_sys::cudf_exec_options_t) -> Vec<i32> {
    read_i32_column_data_with_exec(col, column_len(col), exec)
}

fn filtered_join_probe_state_abi_materializes_retained_semi_and_anti_indices() {
    let target_mr = OwnedMemoryResource::new_test_stats();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let build_tbl = make_table_from_one_column(make_i32_column(&[2, 4]));
    let build_tv = make_table_view(build_tbl);
    let mut join: cudf_sys::cudf_filtered_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_filtered_join_create_ex(build_tv, cudf_sys::CUDF_NULL_EQUALITY_EQUAL, 0.5, &mut join, &exec)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 4]));
    let probe_tv = make_table_view(probe_tbl);

    for (name, begin_probe, expected_size, expected_indices) in [
        ("semi", cudf_sys::cudf_filtered_join_begin_left_semi_probe_ex as BeginProbe, 3usize, vec![1, 2, 3]),
        ("anti", cudf_sys::cudf_filtered_join_begin_left_anti_probe_ex, 1usize, vec![0]),
    ] {
        let target_before = target_mr.allocation_count();
        reset_exec_observation();

        let mut state: cudf_sys::cudf_filtered_join_probe_handle = ptr::null_mut();
        let status = unsafe { begin_probe(join, probe_tv, &mut state, &exec) };

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} probe-state begin failed");
        assert!(!state.is_null(), "{name} probe-state handle was not published");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} probe-state begin must allocate through the explicit execution memory resource"
        );

        let mut output_size = usize::MAX;
        let status = unsafe { cudf_sys::cudf_filtered_join_probe_output_size(state, &mut output_size) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        assert_eq!(output_size, expected_size);

        let mut state_bytes = usize::MAX;
        let status = unsafe { cudf_sys::cudf_filtered_join_probe_device_allocated_size(state, &mut state_bytes) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        assert!(state_bytes > 0, "{name} non-empty retained probe state should report retained device bytes");

        reset_exec_observation();
        let mut indices: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status = unsafe { cudf_sys::cudf_filtered_join_probe_materialize_indices_ex(state, &mut indices, &exec) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} probe-state materialization failed");
        assert!(!indices.is_null());
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(read_all_i32(indices, &exec), expected_indices, "{name} materialized indices");

        unsafe {
            cudf_sys::cudf_column_destroy(indices);
            cudf_sys::cudf_filtered_join_probe_destroy(state);
        }
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_view_destroy(probe_tv);
        cudf_sys::cudf_table_destroy(probe_tbl);
        cudf_sys::cudf_filtered_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

fn filtered_join_probe_state_empty_build_anti_uses_zero_byte_retained_state() {
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());

    let build_tbl = make_table_from_one_column(make_i32_column(&[]));
    let build_tv = make_table_view(build_tbl);
    let mut join: cudf_sys::cudf_filtered_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_filtered_join_create_ex(build_tv, cudf_sys::CUDF_NULL_EQUALITY_EQUAL, 0.5, &mut join, &exec)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_one_column(make_i32_column(&[10, 20, 30]));
    let probe_tv = make_table_view(probe_tbl);
    let mut state: cudf_sys::cudf_filtered_join_probe_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_filtered_join_begin_left_anti_probe_ex(join, probe_tv, &mut state, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!state.is_null());

    let mut output_size = usize::MAX;
    let status = unsafe { cudf_sys::cudf_filtered_join_probe_output_size(state, &mut output_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(output_size, 3);

    let mut state_bytes = usize::MAX;
    let status = unsafe { cudf_sys::cudf_filtered_join_probe_device_allocated_size(state, &mut state_bytes) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(state_bytes, 0);

    let mut indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_filtered_join_probe_materialize_indices_ex(state, &mut indices, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(read_all_i32(indices, &exec), vec![0, 1, 2]);

    unsafe {
        cudf_sys::cudf_column_destroy(indices);
        cudf_sys::cudf_filtered_join_probe_destroy(state);
        cudf_sys::cudf_table_view_destroy(probe_tv);
        cudf_sys::cudf_table_destroy(probe_tbl);
        cudf_sys::cudf_filtered_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

pub(super) const CASES: &[(&str, fn())] = &[
    (
        "filtered_join_probe_tests::filtered_join_probe_state_abi_materializes_retained_semi_and_anti_indices",
        filtered_join_probe_state_abi_materializes_retained_semi_and_anti_indices,
    ),
    (
        "filtered_join_probe_tests::filtered_join_probe_state_empty_build_anti_uses_zero_byte_retained_state",
        filtered_join_probe_state_empty_build_anti_uses_zero_byte_retained_state,
    ),
    (
        "filtered_join_retains_destroy_provenance_after_owner_drop",
        filtered_join_retains_destroy_provenance_after_owner_drop,
    ),
    (
        "filtered_probe_retains_its_own_destroy_provenance_after_build_and_owner_drop",
        filtered_probe_retains_its_own_destroy_provenance_after_build_and_owner_drop,
    ),
];

fn filtered_join_retains_destroy_provenance_after_owner_drop() {
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_test_stats();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let build = make_table_from_one_column(make_i32_column(&[2, 4]));
    let build_view = make_table_view(build);
    let mut join = ptr::null_mut();
    unsafe {
        assert_eq!(
            cudf_sys::cudf_filtered_join_create_ex(
                build_view,
                cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                0.5,
                &mut join,
                &exec
            ),
            cudf_sys::CUDF_SUCCESS
        );
    }
    assert!(!join.is_null());
    stream.synchronize();
    assert!(mr.active_allocation_count() > 0);
    drop(stream);
    drop(mr);
    unsafe {
        assert_eq!(cudf_sys::cudf_filtered_join_destroy(join), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_table_view_destroy(build_view), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_table_destroy(build), cudf_sys::CUDF_SUCCESS);
    }
}

fn filtered_probe_retains_its_own_destroy_provenance_after_build_and_owner_drop() {
    let build_stream = OwnedStream::new();
    let build_mr = OwnedMemoryResource::new_test_stats();
    let build_exec = required_exec_options(&build_stream.borrowed(), &build_mr.borrowed());
    let build = make_table_from_one_column(make_i32_column(&[2, 4]));
    let build_view = make_table_view(build);
    let mut join = ptr::null_mut();
    unsafe {
        assert_eq!(
            cudf_sys::cudf_filtered_join_create_ex(
                build_view,
                cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                0.5,
                &mut join,
                &build_exec
            ),
            cudf_sys::CUDF_SUCCESS
        );
    }
    build_stream.synchronize();
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_test_stats();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let probe = make_table_from_one_column(make_i32_column(&[1, 2, 4]));
    let probe_view = make_table_view(probe);
    let mut state = ptr::null_mut();
    unsafe {
        assert_eq!(
            cudf_sys::cudf_filtered_join_begin_left_semi_probe_ex(join, probe_view, &mut state, &exec),
            cudf_sys::CUDF_SUCCESS
        );
    }
    assert!(!state.is_null());
    stream.synchronize();
    assert!(mr.active_allocation_count() > 0);
    // The probe owns its membership result and no longer reads the build hash table.
    unsafe {
        assert_eq!(cudf_sys::cudf_filtered_join_destroy(join), cudf_sys::CUDF_SUCCESS);
    }
    drop(stream);
    drop(mr);
    unsafe {
        assert_eq!(cudf_sys::cudf_filtered_join_probe_destroy(state), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_table_view_destroy(probe_view), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_table_destroy(probe), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_table_view_destroy(build_view), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_table_destroy(build), cudf_sys::CUDF_SUCCESS);
    }
}
