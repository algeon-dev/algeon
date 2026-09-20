use super::*;

#[test]
fn io_ex_entries_use_required_exec_options() {
    for (source, function, end_marker) in [
        (IO_PARQUET_CPP, "cudf_io_read_parquet_ex", "cudf_status_t cudf_io_read_parquet_from_host_buffer_ex"),
        (IO_PARQUET_CPP, "cudf_io_read_parquet_from_host_buffer_ex", "cudf_io_write_parquet_ex"),
        (IO_PARQUET_CPP, "cudf_io_write_parquet_ex", "cudf_status_t cudf_io_read_parquet_with_metadata_ex"),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_with_metadata_ex",
            "cudf_status_t cudf_io_read_parquet_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_with_metadata_filter_ex",
            "cudf_status_t cudf_io_read_remote_parquet_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_remote_parquet_with_metadata_filter_ex",
            "cudf_status_t cudf_io_read_parquet_multi_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_multi_with_metadata_filter_ex",
            "cudf_status_t cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex",
            "cudf_status_t cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex",
            "int32_t cudf_io_remote_parquet_kvikio_remote_io_enabled",
        ),
    ] {
        let body = function_region_by_function_name(source, function, end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(exec_options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(exec_options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn remote_parquet_device_read_orders_external_device_write_after_requested_stream() {
    let async_body =
        function_region(IO_PARQUET_CPP, "std::future<size_t> device_read_async", "size_t device_read(size_t offset,");
    let sync_pointer_body =
        function_region(IO_PARQUET_CPP, "size_t device_read(size_t offset,", "std::unique_ptr<buffer> device_read");
    let owned_buffer_body =
        function_region(IO_PARQUET_CPP, "std::unique_ptr<buffer> device_read", "[[nodiscard]] size_t size()");

    assert!(
        source_contract_support::contains_normalized(
            async_body,
            "stream.synchronize(); cudf_record_synchronized_stream(stream); return handle_.pread(dst, read_size, offset, read_size);"
        ),
        "remote Parquet device_read_async must wait for prior requested-stream work before KvikIO writes into the supplied device pointer"
    );
    assert!(
        source_contract_support::contains_normalized(
            sync_pointer_body,
            "return device_read_async(offset, size, dst, stream).get();"
        ) && !sync_pointer_body.contains("handle_.pread"),
        "remote Parquet synchronous pointer device_read must delegate through synchronized device_read_async"
    );
    assert!(
        source_contract_support::contains_normalized(
            owned_buffer_body,
            "device_read(offset, read_size, reinterpret_cast<uint8_t*>(out.data()), stream);"
        ) && !owned_buffer_body.contains("handle_.pread"),
        "remote Parquet owned-buffer device_read must delegate through synchronized pointer device_read"
    );
}

#[test]
fn io_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("IO native header", IO_H, "cudf_status_t cudf_io_read_parquet("),
        ("Parquet native source", IO_PARQUET_CPP, "cudf_status_t cudf_io_read_parquet("),
        ("Parquet Rust FFI", IO_PARQUET_FFI_RS, "pub fn cudf_io_read_parquet("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_read_parquet_with_metadata("),
        ("Parquet native source", IO_PARQUET_CPP, "cudf_status_t cudf_io_read_parquet_with_metadata("),
        ("Parquet Rust FFI", IO_PARQUET_FFI_RS, "pub fn cudf_io_read_parquet_with_metadata("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_write_parquet("),
        ("Parquet native source", IO_PARQUET_CPP, "cudf_status_t cudf_io_write_parquet("),
        ("Parquet Rust FFI", IO_PARQUET_FFI_RS, "pub fn cudf_io_write_parquet("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    let marker = "cudf_io_write_parquet_ex(table, filepath, column_names, num_columns, row_group_size_rows, dictionary_policy, nullptr)";
    assert!(
        !contains_normalized(IO_PARQUET_CPP, marker),
        "Parquet native source must not delegate to strict `_ex` with null options via `{marker}`"
    );
}
