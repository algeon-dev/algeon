use std::sync::Arc;

use arrow::{
    array::{Array, Int32Array, PrimitiveArray},
    record_batch::RecordBatch,
};

use crate::{
    core::error::Result,
    runtime::{BoundaryObservation, Runtime},
};

pub(crate) fn trigger_numeric_to_arrow_error_for_test() -> Result<()> {
    let buffer = arrow::buffer::ScalarBuffer::from(vec![1i32]);
    let null_buffer = arrow::buffer::NullBuffer::from(vec![true, false]);
    PrimitiveArray::<arrow::datatypes::Int32Type>::try_new(buffer, Some(null_buffer))
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
    super::reset_host_interop_observation();
}

pub(crate) fn host_export_call_count_for_test() -> usize {
    super::host_export_call_count()
}

pub(crate) fn host_import_call_count_for_test() -> usize {
    super::host_import_call_count()
}

pub(crate) fn last_host_boundary_observation_for_test() -> Option<BoundaryObservation> {
    super::last_host_boundary_observation()
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::super::{call_count_since, reset_call_count_baseline};

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
