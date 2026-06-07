use cudf::{
    Column, DataFrame,
    interop::{
        CompletionAuthority, ExportProfile, LeaseCompletionResult, PartitionedDataFrameExport, Placement,
        ProducerReadiness,
    },
};

#[test]
fn partitioned_dataframe_export_groups_columns_by_retained_provenance_in_first_appearance_order() {
    let left = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(7001).unwrap();
    let middle = Column::from_slice(&[10i32, 20, 30]).unwrap().__debug_rebind_execution_domain_explicit(7002).unwrap();
    let right =
        Column::from_slice(&[100i32, 200, 300]).unwrap().__debug_rebind_execution_domain_explicit(7001).unwrap();

    let producer = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap();
    let export = producer.try_prepare_device_partitioned_table_export().unwrap();

    assert_eq!(export.num_rows(), 3);
    assert_eq!(export.schema_names().unwrap(), &["left".to_string(), "middle".to_string(), "right".to_string()]);
    assert_eq!(export.partitions().len(), 2);
    assert_eq!(export.partitions()[0].top_level_column_indices(), &[0, 2]);
    assert_eq!(export.partitions()[1].top_level_column_indices(), &[1]);
    assert_eq!(export.partitions()[0].export().columns().len(), 2);
    assert_eq!(export.partitions()[1].export().columns().len(), 1);
    assert_eq!(export.partitions()[0].export().diagnostics().profile(), ExportProfile::PreparedDeviceExport);
    assert_eq!(export.partitions()[1].export().diagnostics().profile(), ExportProfile::PreparedDeviceExport);

    let first =
        export.partitions()[0].export().try_activate_read_lease(CompletionAuthority::SynchronousByReturn).unwrap();
    let second =
        export.partitions()[1].export().try_activate_read_lease(CompletionAuthority::SynchronousByReturn).unwrap();
    assert_eq!(first.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    assert_eq!(second.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn partitioned_dataframe_export_prepare_host_table_export_returns_one_host_visible_root() {
    let left = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(7101).unwrap();
    let right = Column::from_slice(&[4i32, 5, 6]).unwrap().__debug_rebind_execution_domain_explicit(7102).unwrap();

    let export = PartitionedDataFrameExport::new(vec!["left".into(), "right".into()], vec![left, right]).unwrap();
    let prepared = export.try_prepare_host_table_export().unwrap();

    assert_eq!(prepared.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(prepared.table().num_rows(), 3);
    assert_eq!(prepared.table().names().unwrap(), &["left".to_string(), "right".to_string()]);
    assert_eq!(prepared.columns().len(), 2);
    assert!(prepared.bindings().iter().all(|binding| matches!(binding.placement(), Placement::Host)));
    assert!(matches!(prepared.readiness(), ProducerReadiness::HostVisible));

    let lease = prepared.try_activate_read_lease(CompletionAuthority::SynchronousByReturn).unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn partitioned_dataframe_export_owned_transfer_returns_one_partition_per_retained_provenance() {
    let left = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(7201).unwrap();
    let middle = Column::from_slice(&[10i32, 20, 30]).unwrap().__debug_rebind_execution_domain_explicit(7202).unwrap();
    let right =
        Column::from_slice(&[100i32, 200, 300]).unwrap().__debug_rebind_execution_domain_explicit(7201).unwrap();

    let transfer = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap()
    .try_into_owned_device_partitioned_table_transfer()
    .unwrap();

    assert_eq!(transfer.num_rows(), 3);
    assert_eq!(transfer.schema_names().unwrap(), &["left".to_string(), "middle".to_string(), "right".to_string()]);
    assert_eq!(transfer.partitions().len(), 2);
    assert_eq!(transfer.partitions()[0].top_level_column_indices(), &[0, 2]);
    assert_eq!(transfer.partitions()[1].top_level_column_indices(), &[1]);
    assert_eq!(transfer.partitions()[0].transfer().table().num_rows(), 3);
    assert_eq!(transfer.partitions()[0].transfer().columns().len(), 2);
    assert_eq!(transfer.partitions()[1].transfer().columns().len(), 1);
}

#[test]
fn ordinary_dataframe_still_rejects_mixed_provenance_device_export() {
    let left = Column::from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(7301).unwrap();
    let right = Column::from_slice(&[4i32, 5, 6]).unwrap().__debug_rebind_execution_domain_explicit(7302).unwrap();
    let frame = DataFrame::new(vec!["left".into(), "right".into()], vec![left, right]);

    assert!(frame.is_err());
}
