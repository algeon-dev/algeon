use std::{ffi::c_void, ptr};

use cudf::{Column, DataFrame, MergeOptions};

use crate::common::CURRENT_DEVICE_RESOURCE_LOCK;

fn left_df() -> DataFrame {
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10i32, 20, 30]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df() -> DataFrame {
    let ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let values = Column::from_slice(&[200i32, 300, 400]).unwrap();
    DataFrame::new(vec!["id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn left_df_asym() -> DataFrame {
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10i32, 20, 30]).unwrap();
    DataFrame::new(vec!["left_id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df_asym() -> DataFrame {
    let ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let values = Column::from_slice(&[200i32, 300, 400]).unwrap();
    DataFrame::new(vec!["right_id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
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
    let _guard = CURRENT_DEVICE_RESOURCE_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
fn merge_same_key_allocates_no_more_than_merge_split() {
    let left = left_df();
    let right = right_df();
    let options = MergeOptions::new().on(&["id"]);

    let merge_allocations = allocation_count_delta(|| {
        let result = left.merge(&right, options.clone()).unwrap();
        assert_eq!(result.column_names(), vec!["id", "score", "value"]);
        assert_eq!(result.num_rows(), 2);
        let _ = result.series_view("id").unwrap().to_vec::<i32>().unwrap();
        result
    });

    let split_allocations = allocation_count_delta(|| {
        let result = left.merge_split(&right, options.clone()).unwrap();
        assert_eq!(result.left.column_names(), vec!["id", "score"]);
        assert_eq!(result.right.column_names(), vec!["id", "value"]);
        assert_eq!(result.left.num_rows(), 2);
        assert_eq!(result.right.num_rows(), 2);
        let _ = result.left.series_view("id").unwrap().to_vec::<i32>().unwrap();
        let _ = result.right.series_view("id").unwrap().to_vec::<i32>().unwrap();
        result
    });

    assert_eq!(
        merge_allocations, split_allocations,
        "merge() should not allocate more than merge_split() for an inner join on shared keys",
    );
}

#[test]
fn merge_asymmetric_keys_allocates_no_more_than_merge_split() {
    let left = left_df_asym();
    let right = right_df_asym();
    let options = MergeOptions::new().how_left().left_on(&["left_id"]).right_on(&["right_id"]);

    let merge_allocations = allocation_count_delta(|| {
        let result = left.merge(&right, options.clone()).unwrap();
        assert_eq!(result.column_names(), vec!["left_id", "score", "right_id", "value"]);
        assert_eq!(result.num_rows(), 3);
        let _ = result.series_view("left_id").unwrap().to_vec::<i32>().unwrap();
        result
    });

    let split_allocations = allocation_count_delta(|| {
        let result = left.merge_split(&right, options.clone()).unwrap();
        assert_eq!(result.left.column_names(), vec!["left_id", "score"]);
        assert_eq!(result.right.column_names(), vec!["right_id", "value"]);
        assert_eq!(result.left.num_rows(), 3);
        assert_eq!(result.right.num_rows(), 3);
        let _ = result.left.series_view("left_id").unwrap().to_vec::<i32>().unwrap();
        let _ = result.right.series_view("right_id").unwrap().to_vec::<i32>().unwrap();
        result
    });

    assert_eq!(
        merge_allocations, split_allocations,
        "merge() should not allocate more than merge_split() for asymmetric non-full joins",
    );
}
