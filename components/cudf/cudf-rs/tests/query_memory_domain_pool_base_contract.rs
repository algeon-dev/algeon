use cudf::{
    DeviceId, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerOptions, ErrorKind,
    QueryMemoryDomainOptions,
};

#[test]
fn query_memory_domain_rejects_pool_backed_manager() {
    let device = DeviceId::current().unwrap();
    let manager = DeviceMemoryManager::install_for_device_with_options(
        device,
        DeviceMemoryManagerOptions {
            base: DeviceMemoryManagerBase::Pool {
                initial_pool_size: 64 * 1024 * 1024,
                maximum_pool_size: Some(64 * 1024 * 1024),
            },
        },
    )
    .unwrap();

    let err = manager
        .query_memory_domain(50_001, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}
