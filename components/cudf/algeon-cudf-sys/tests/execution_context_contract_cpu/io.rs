use super::{IO_PARQUET_CPP, function_region, source_contract_support};

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
