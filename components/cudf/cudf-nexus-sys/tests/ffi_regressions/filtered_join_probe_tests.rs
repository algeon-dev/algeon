use std::ptr;

use crate::{
    helpers::{make_i32_column, make_table_from_one_column, make_table_view, read_i32_column_data},
    support::{
        CurrentDeviceResourceRestore, OwnedMemoryResource, OwnedStream, current_device_resource_raw, last_exec_mr_raw,
        last_exec_stream_value, lock_shim_state, required_exec_options, reset_exec_observation, stream_value,
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

fn read_all_i32(col: cudf_sys::cudf_column_handle) -> Vec<i32> {
    read_i32_column_data(col, column_len(col))
}

#[test]
fn filtered_join_probe_state_abi_materializes_retained_semi_and_anti_indices() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
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
        let current_before = current_mr.allocation_count();
        let current_borrowed = current_mr.borrowed();
        reset_exec_observation();
        crate::support::set_current_device_resource(&current_borrowed)
            .expect("failed to install current sentinel memory resource");

        let mut state: cudf_sys::cudf_filtered_join_probe_handle = ptr::null_mut();
        let status = unsafe { begin_probe(join, probe_tv, &mut state, &exec) };
        let restored_current = current_device_resource_raw();
        unsafe { crate::support::set_current_device_resource_raw(previous_raw) }
            .expect("failed to restore previous current device resource");

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} probe-state begin failed");
        assert!(!state.is_null(), "{name} probe-state handle was not published");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert_eq!(
            restored_current,
            current_mr.raw_device_resource(),
            "{name} probe-state begin must restore the caller's current memory resource"
        );
        // Device-domain model: libcudf-internal temporaries deliberately
        // resolve through the device's current resource (here the sentinel),
        // so no equality is asserted on `current_mr.allocation_count()`. The
        // retained probe state itself must still route through the explicit
        // execution memory resource.
        let _ = current_before;
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
        assert_eq!(read_all_i32(indices), expected_indices, "{name} materialized indices");

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

#[test]
fn filtered_join_probe_state_empty_build_anti_uses_zero_byte_retained_state() {
    let _shim_state = lock_shim_state();
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
    assert_eq!(read_all_i32(indices), vec![0, 1, 2]);

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
