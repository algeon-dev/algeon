use std::{ffi::c_void, ptr, sync::Mutex};

use cudf::{
    DataFrame, SortOptions,
    column::Column,
    sorting::{self, NullOrder, Order},
};

static CURRENT_MR_LOCK: Mutex<()> = Mutex::new(());

fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[2i32, 1, 2, 1]).unwrap();
    let scores = Column::from_slice(&[20i32, 10, 40, 30]).unwrap();
    let tags = Column::from_slice(&[200i32, 100, 400, 300]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string(), "tag".to_string()], vec![ids, scores, tags]).unwrap()
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
fn subset_key_sort_values_allocates_no_more_than_low_level_borrowed_sort_by_key() {
    let df = sample_df();

    let high_level_allocations = allocation_count_delta(|| {
        let result = df.sort_values(&["id", "score"], SortOptions::new().with_descending(vec![false, true])).unwrap();

        assert_eq!(result.column_names(), vec!["id", "score", "tag"]);
        assert_eq!(result.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 2, 2]);
        assert_eq!(result.series_view("score").unwrap().to_vec::<i32>().unwrap(), vec![30, 10, 40, 20]);
        assert_eq!(result.series_view("tag").unwrap().to_vec::<i32>().unwrap(), vec![300, 100, 400, 200]);
        result
    });

    let low_level_allocations = allocation_count_delta(|| {
        let values = df.temp_table_view().unwrap();
        let keys = values.table().select_columns(&[0, 1]).unwrap();

        let result_table = sorting::sort_by_key(
            values.table(),
            &keys,
            Some(&[Order::Ascending, Order::Descending]),
            Some(&[NullOrder::After, NullOrder::After]),
        )
        .unwrap();

        let result =
            DataFrame::from_table(result_table, vec!["id".to_string(), "score".to_string(), "tag".to_string()])
                .unwrap();

        assert_eq!(result.column_names(), vec!["id", "score", "tag"]);
        assert_eq!(result.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 2, 2]);
        assert_eq!(result.series_view("score").unwrap().to_vec::<i32>().unwrap(), vec![30, 10, 40, 20]);
        assert_eq!(result.series_view("tag").unwrap().to_vec::<i32>().unwrap(), vec![300, 100, 400, 200]);
        result
    });

    assert_eq!(
        high_level_allocations, low_level_allocations,
        "subset-key sort_values() should not allocate more than the equivalent low-level borrowed sort_by_key",
    );
}
