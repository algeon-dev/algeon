#[path = "../ops/groupby/dataframe_groupby_allocation_tests.rs"]
mod dataframe_groupby_allocation_tests;
#[path = "../ops/join/dataframe_merge_allocation_tests.rs"]
mod dataframe_merge_allocation_tests;
#[path = "../ops/sorting/dataframe_sort_values_allocation_tests.rs"]
mod dataframe_sort_values_allocation_tests;

use crate::{
    allocation::allocation_count_delta,
    common::{column_from_slice, dataframe_from_columns},
};

#[test]
fn dataframe_prepare_host_export_does_not_allocate_extra_device_memory_when_columns_are_shared() {
    let baseline = dataframe_from_columns(
        vec!["id".to_string(), "value".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i64, 20, 30]).unwrap()],
    )
    .unwrap();
    let original = dataframe_from_columns(
        vec!["id".to_string(), "value".to_string()],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i64, 20, 30]).unwrap()],
    )
    .unwrap();
    let shared = original;

    let _ = baseline.try_prepare_host_table_export().unwrap();
    let _ = shared.try_prepare_host_table_export().unwrap();

    let unique_allocations = allocation_count_delta(|| {
        let export = baseline.try_prepare_host_table_export().unwrap();
        assert_eq!(export.table().num_rows(), 3);
        assert_eq!(export.table().names().unwrap(), &["id".to_string(), "value".to_string()]);
        export
    });

    let shared_allocations = allocation_count_delta(|| {
        let export = shared.try_prepare_host_table_export().unwrap();
        assert_eq!(export.table().num_rows(), 3);
        assert_eq!(export.table().names().unwrap(), &["id".to_string(), "value".to_string()]);
        export
    });

    assert_eq!(
        shared_allocations, unique_allocations,
        "DataFrame host export should not allocate extra device memory just because its columns are Arc-shared",
    );
}
