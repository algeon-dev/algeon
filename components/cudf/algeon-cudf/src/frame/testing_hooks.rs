use std::sync::atomic::{AtomicUsize, Ordering};

static DEVICE_TO_DEVICE_COPY_COUNT: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn record_device_to_device_copy() {
    DEVICE_TO_DEVICE_COPY_COUNT.fetch_add(1, Ordering::AcqRel);
}

pub(crate) fn device_to_device_copy_count_for_test() -> usize {
    DEVICE_TO_DEVICE_COPY_COUNT.load(Ordering::Acquire)
}
