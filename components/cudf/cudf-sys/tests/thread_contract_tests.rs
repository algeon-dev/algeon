use static_assertions::{assert_impl_all, assert_not_impl_any};

assert_impl_all!(cudf_sys::Stream: Send, Sync);
assert_impl_all!(cudf_sys::Event: Send, Sync);
assert_impl_all!(cudf_sys::MemoryResource: Send, Sync);
assert_not_impl_any!(cudf_sys::ExecutionContext<'static>: Send, Sync);

#[test]
fn owning_runtime_handles_move_across_thread_boundaries() {
    let stream = cudf_sys::Stream::new().expect("failed to create CUDA stream");
    let event = cudf_sys::Event::new().expect("failed to create CUDA event");
    let mr = cudf_sys::MemoryResource::new_cuda().expect("failed to create CUDA memory resource");

    event.record(&stream).expect("failed to record event");

    std::thread::spawn(move || {
        stream.synchronize().expect("worker thread failed to synchronize stream");
        event.synchronize().expect("worker thread failed to synchronize event");
        assert!(!mr.raw_device_resource().expect("worker thread failed to read raw memory resource").is_null());
    })
    .join()
    .expect("worker thread panicked");
}

#[test]
fn owning_runtime_handles_can_be_shared_by_reference_across_threads() {
    let stream = cudf_sys::Stream::new().expect("failed to create CUDA stream");
    let event = cudf_sys::Event::new().expect("failed to create CUDA event");
    let mr = cudf_sys::MemoryResource::new_cuda().expect("failed to create CUDA memory resource");

    event.record(&stream).expect("failed to record event");

    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let _ = stream.is_ready().expect("worker thread failed to query stream readiness");
                let _ = event.is_ready().expect("worker thread failed to query event readiness");
                assert!(!mr.raw_device_resource().expect("worker thread failed to read raw memory resource").is_null());
            })
            .join()
            .expect("worker thread panicked");
    });
}
