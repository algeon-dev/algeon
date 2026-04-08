#[path = "common/allocation.rs"]
mod allocation;
#[path = "common/mod.rs"]
mod common;

// These tests temporarily replace the process-global current device resource
// via cudf-sys test hooks. Keep them in their own integration-test binary so
// they do not run in the same libtest process as the broader interop suite.

mod interop_allocation_tests {
    use cudf::{core::Column, interop::ProducerReadiness};

    use crate::allocation::allocation_count_delta;

    #[test]
    fn dataframe_prepare_host_export_does_not_allocate_extra_device_memory_when_columns_are_shared() {
        let baseline = crate::common::dataframe_from_columns(
            vec!["id".to_string(), "value".to_string()],
            vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[10i64, 20, 30]).unwrap()],
        )
        .unwrap();
        let original = crate::common::dataframe_from_columns(
            vec!["id".to_string(), "value".to_string()],
            vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[10i64, 20, 30]).unwrap()],
        )
        .unwrap();
        let shared = original.clone();

        let _ = baseline.try_prepare_host_table_export().unwrap();
        let _ = shared.try_prepare_host_table_export().unwrap();

        let unique_allocations = allocation_count_delta(|| {
            let export = baseline.try_prepare_host_table_export().unwrap();
            assert_eq!(export.table().num_rows(), 3);
            assert_eq!(export.table().names().unwrap(), &["id".to_string(), "value".to_string()]);
            assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
            export
        });

        let shared_allocations = allocation_count_delta(|| {
            let export = shared.try_prepare_host_table_export().unwrap();
            assert_eq!(export.table().num_rows(), 3);
            assert_eq!(export.table().names().unwrap(), &["id".to_string(), "value".to_string()]);
            assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
            export
        });

        assert_eq!(
            shared_allocations, unique_allocations,
            "DataFrame host export should not allocate extra device memory just because its columns are Arc-shared",
        );
    }
}
