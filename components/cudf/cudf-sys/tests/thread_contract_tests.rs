mod support;

use static_assertions::{assert_impl_all, assert_not_impl_any};

use support::{OwnedEvent, OwnedMemoryResource, OwnedStream};

assert_impl_all!(cudf_sys::BorrowedStream<'static>: Send);
assert_not_impl_any!(cudf_sys::BorrowedStream<'static>: Sync);
assert_not_impl_any!(cudf_sys::BorrowedStream<'static>: Copy);
assert_impl_all!(cudf_sys::BorrowedEvent<'static>: Send, Sync);
assert_not_impl_any!(cudf_sys::BorrowedEvent<'static>: Copy);
assert_impl_all!(cudf_sys::BorrowedMemoryResource<'static>: Send, Sync);
assert_not_impl_any!(cudf_sys::BorrowedMemoryResource<'static>: Copy);
assert_not_impl_any!(cudf_sys::ExecutionContext<'static>: Send, Sync);

#[test]
fn borrowed_runtime_wrappers_are_non_owning_and_need_no_drop() {
    assert!(!std::mem::needs_drop::<cudf_sys::BorrowedStream<'static>>());
    assert!(!std::mem::needs_drop::<cudf_sys::BorrowedEvent<'static>>());
    assert!(!std::mem::needs_drop::<cudf_sys::BorrowedMemoryResource<'static>>());
}

#[test]
fn borrowed_runtime_handles_move_across_thread_boundaries() {
    let stream = OwnedStream::new();
    let event = OwnedEvent::new();
    let mr = OwnedMemoryResource::new_cuda();

    event.record(&stream);

    let borrowed_stream = stream.borrowed();
    let borrowed_event = event.borrowed();
    let borrowed_mr = mr.borrowed();

    std::thread::scope(|scope| {
        scope
            .spawn(move || {
                let mut stream_ready = 0;
                let stream_status = unsafe {
                    cudf_sys::cudf_stream_is_ready(borrowed_stream.as_raw(), &mut stream_ready)
                };
                assert_eq!(stream_status, cudf_sys::CUDF_SUCCESS);

                let mut event_ready = 0;
                let event_status = unsafe {
                    cudf_sys::cudf_event_is_ready(borrowed_event.as_raw(), &mut event_ready)
                };
                assert_eq!(event_status, cudf_sys::CUDF_SUCCESS);

                let mut raw_mr = std::ptr::null_mut();
                let mr_status = unsafe {
                    cudf_sys::cudf_memory_resource_raw(borrowed_mr.as_raw(), &mut raw_mr)
                };
                assert_eq!(mr_status, cudf_sys::CUDF_SUCCESS);
                assert!(!raw_mr.is_null());
            })
            .join()
            .expect("worker thread panicked");
    });
}

#[test]
fn borrowed_event_and_memory_resource_can_be_shared_by_reference_across_threads() {
    let stream = OwnedStream::new();
    let event = OwnedEvent::new();
    let mr = OwnedMemoryResource::new_cuda();

    event.record(&stream);
    let borrowed_event = event.borrowed();
    let borrowed_mr = mr.borrowed();

    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let mut event_ready = 0;
                let event_status = unsafe {
                    cudf_sys::cudf_event_is_ready(borrowed_event.as_raw(), &mut event_ready)
                };
                assert_eq!(event_status, cudf_sys::CUDF_SUCCESS);

                let mut raw_mr = std::ptr::null_mut();
                let mr_status = unsafe {
                    cudf_sys::cudf_memory_resource_raw(borrowed_mr.as_raw(), &mut raw_mr)
                };
                assert_eq!(mr_status, cudf_sys::CUDF_SUCCESS);
                assert!(!raw_mr.is_null());
            })
            .join()
            .expect("worker thread panicked");
    });
}
