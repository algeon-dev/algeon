use cudf::{DeviceId, DeviceMemoryManager, DeviceMemoryManagerOptions, ErrorCode, QueryMemoryDomainOptions};

#[test]
fn static_partition_rejects_private_query_memory_domain() {
    let device = DeviceId::current().unwrap();
    let options = DeviceMemoryManagerOptions {
        allocation_limit_bytes: Some(64 * 1024 * 1024),
        ..DeviceMemoryManagerOptions::default()
    };
    let manager = DeviceMemoryManager::install_for_device_with_options(device, options).unwrap();

    let error = manager
        .query_memory_domain(
            92_001,
            QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(16 * 1024 * 1024),
        )
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error.detail_value("reason"), Some("query_domain_incompatible_with_allocation_limit"));
}
