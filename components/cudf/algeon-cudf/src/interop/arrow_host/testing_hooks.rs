use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow::{
    array::{Array, Int32Array, PrimitiveArray},
    buffer::{NullBuffer, ScalarBuffer},
    datatypes::Int32Type,
    record_batch::RecordBatch,
};

use crate::{core::error::Result, runtime::Runtime};

thread_local! {
    static HOST_ARROW_EXPORT_CALL_BASELINE: Cell<usize> = const { Cell::new(0) };
    static HOST_ARROW_IMPORT_CALL_BASELINE: Cell<usize> = const { Cell::new(0) };
}

static HOST_ARROW_EXPORT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static HOST_ARROW_IMPORT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

pub(super) fn record_host_export_call() {
    HOST_ARROW_EXPORT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
}

pub(super) fn record_host_import_call() {
    HOST_ARROW_IMPORT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
}

fn reset_call_count_baseline(counter: &AtomicUsize, baseline: &Cell<usize>) {
    baseline.set(counter.load(Ordering::Acquire));
}

fn call_count_since(counter: &AtomicUsize, baseline: &Cell<usize>) -> usize {
    counter.load(Ordering::Acquire).saturating_sub(baseline.get())
}

pub(crate) fn trigger_numeric_to_arrow_error_for_test() -> Result<()> {
    let buffer = ScalarBuffer::from(vec![1i32]);
    let null_buffer = NullBuffer::from(vec![true, false]);
    PrimitiveArray::<Int32Type>::try_new(buffer, Some(null_buffer))
        .map(|_| ())
        .map_err(super::map_numeric_to_arrow_error)
}

pub(crate) fn trigger_record_batch_error_for_test() -> Result<()> {
    let columns = vec![
        ("left".to_string(), Arc::new(Int32Array::from(vec![1, 2])) as Arc<dyn Array>, false),
        ("right".to_string(), Arc::new(Int32Array::from(vec![10])) as Arc<dyn Array>, false),
    ];
    RecordBatch::try_from_iter_with_nullable(columns).map(|_| ()).map_err(super::map_record_batch_error)
}

pub(crate) fn trigger_string_array_downcast_mismatch_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let wrong = Int32Array::from(vec![1, 2]);
    super::string_array_to_column(&runtime, &wrong).map(|_| ())
}

pub(crate) fn trigger_large_string_array_downcast_mismatch_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let wrong = Int32Array::from(vec![1, 2]);
    super::large_string_array_to_column(&runtime, &wrong).map(|_| ())
}

pub(crate) fn reset_host_interop_observation_for_test() {
    HOST_ARROW_EXPORT_CALL_BASELINE.with(|baseline| {
        reset_call_count_baseline(&HOST_ARROW_EXPORT_CALL_COUNT, baseline);
    });
    HOST_ARROW_IMPORT_CALL_BASELINE.with(|baseline| {
        reset_call_count_baseline(&HOST_ARROW_IMPORT_CALL_COUNT, baseline);
    });
}

pub(crate) fn host_export_call_count_for_test() -> usize {
    HOST_ARROW_EXPORT_CALL_BASELINE.with(|baseline| call_count_since(&HOST_ARROW_EXPORT_CALL_COUNT, baseline))
}

pub(crate) fn host_import_call_count_for_test() -> usize {
    HOST_ARROW_IMPORT_CALL_BASELINE.with(|baseline| call_count_since(&HOST_ARROW_IMPORT_CALL_COUNT, baseline))
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::{call_count_since, reset_call_count_baseline};

    #[test]
    fn resetting_one_call_count_observer_does_not_rewind_another() {
        let counter = AtomicUsize::new(0);
        let first_baseline = Cell::new(0);
        let second_baseline = Cell::new(0);

        counter.fetch_add(2, Ordering::Release);
        assert_eq!(call_count_since(&counter, &first_baseline), 2);

        reset_call_count_baseline(&counter, &second_baseline);
        assert_eq!(call_count_since(&counter, &second_baseline), 0);
        assert_eq!(call_count_since(&counter, &first_baseline), 2);

        counter.fetch_add(1, Ordering::Release);
        assert_eq!(call_count_since(&counter, &second_baseline), 1);
        assert_eq!(call_count_since(&counter, &first_baseline), 3);
    }
}
