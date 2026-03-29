use std::{ffi::c_void, ptr, sync::Mutex};

use cudf::{
    Agg, AggSpec, DataFrame,
    column::Column,
    groupby::{Aggregation, AggregationRequest, NullPolicy, groupby_aggregate},
    table::TableView,
};

static CURRENT_MR_LOCK: Mutex<()> = Mutex::new(());

fn sample_df() -> DataFrame {
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = Column::from_slice(&[1i64, 2, 3, 4]).unwrap();
    DataFrame::new(vec!["key".to_string(), "a".to_string(), "b".to_string()], vec![keys, a, b]).unwrap()
}

fn current_device_resource_raw() -> *mut c_void {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read current device resource");
    out
}

struct CurrentDeviceResourceRestore {
    previous_raw: *mut c_void,
}

impl Drop for CurrentDeviceResourceRestore {
    fn drop(&mut self) {
        unsafe { cudf_sys::test_set_current_device_resource_raw(self.previous_raw) }
            .expect("failed to restore previous current device resource");
    }
}

fn allocation_count_delta<T>(op: impl FnOnce() -> T) -> i64 {
    let _guard = CURRENT_MR_LOCK.lock().unwrap();
    let previous_raw = current_device_resource_raw();
    let stats = cudf_sys::MemoryResource::new_test_stats().expect("failed to create stats memory resource");
    let _restore = CurrentDeviceResourceRestore { previous_raw };

    cudf_sys::test_set_current_device_resource(&stats).expect("failed to install stats memory resource");

    let before = stats.test_allocation_count().expect("failed to read allocation count before operation");
    let _result = op();
    let after = stats.test_allocation_count().expect("failed to read allocation count after operation");
    after - before
}

#[test]
fn groupby_agg_allocates_no_more_than_low_level_borrowed_primitive() {
    let df = sample_df();

    let high_level_allocations = allocation_count_delta(|| {
        let result = df
            .groupby(&["key"])
            .agg(&[AggSpec::new("a", Agg::Sum), AggSpec::new("b", Agg::Sum), AggSpec::new("a", Agg::Min)])
            .unwrap();

        assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum", "a_min"]);
        assert_eq!(result.num_rows(), 2);
        result
    });

    let low_level_allocations = allocation_count_delta(|| {
        let key_view = df.series_view("key").unwrap();
        let key_table = TableView::from_column_views(&[key_view.as_column_view()]).unwrap();

        let a_view = df.series_view("a").unwrap();
        let b_view = df.series_view("b").unwrap();
        let a_aggs = [Aggregation::Sum, Aggregation::Min];
        let b_aggs = [Aggregation::Sum];
        let requests = [
            AggregationRequest { values: a_view.as_column_view(), aggregations: &a_aggs },
            AggregationRequest { values: b_view.as_column_view(), aggregations: &b_aggs },
        ];

        let result = groupby_aggregate(&key_table, &requests, NullPolicy::Exclude).unwrap();
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].len(), 2);
        assert_eq!(result.results[1].len(), 1);
        result
    });

    assert_eq!(
        high_level_allocations, low_level_allocations,
        "high-level groupby should not allocate more than the equivalent low-level borrowed primitive",
    );
}
