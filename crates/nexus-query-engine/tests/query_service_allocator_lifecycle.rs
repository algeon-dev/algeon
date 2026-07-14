use cudf::{DeviceId, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerOptions};
use nexus_query_engine::{
    error::QueryErrorCode,
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus,
        QueryMemoryDomainPolicy, QueryService, QueryServiceConfig, QueryServiceTenantConfig, S3SourceStorageProfile,
        SourceStorageProfile,
    },
};

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

fn pool_config(device: DeviceId) -> QueryServiceConfig {
    QueryServiceConfig {
        device_ordinal: device.ordinal(),
        device_memory_manager_base: DeviceMemoryManagerBase::Pool {
            initial_pool_size: 64 * 1024 * 1024,
            maximum_pool_size: Some(64 * 1024 * 1024),
        },
        ..QueryServiceConfig::default()
    }
}

#[test]
fn rejected_and_conflicting_pool_configs_do_not_mutate_the_allocator() {
    let device = DeviceId::current().unwrap();
    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), None);

    let config =
        QueryServiceConfig { query_memory_domain: Some(QueryMemoryDomainPolicy::default()), ..pool_config(device) };
    let error = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("query_memory_domain_requires_cuda_async_manager"));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        None,
        "validation must reject before the process-global allocator is installed",
    );

    let empty_tenants =
        QueryService::new_with_deployment(pool_config(device), vec![], non_gds_deployment()).unwrap_err();
    assert_eq!(empty_tenants.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(empty_tenants.fact_value("reason"), Some("empty"));

    let invalid_tenant = QueryService::new_with_deployment(
        pool_config(device),
        vec![QueryServiceTenantConfig::new("invalid", 0)],
        non_gds_deployment(),
    )
    .unwrap_err();
    assert_eq!(invalid_tenant.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(invalid_tenant.fact_value("reason"), Some("zero"));

    let duplicate_tenant = QueryService::new_with_deployment(
        pool_config(device),
        vec![
            QueryServiceTenantConfig::new("duplicate", 8 * 1024 * 1024),
            QueryServiceTenantConfig::new("duplicate", 8 * 1024 * 1024),
        ],
        non_gds_deployment(),
    )
    .unwrap_err();
    assert_eq!(duplicate_tenant.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(duplicate_tenant.fact_value("reason"), Some("duplicate"));

    let empty_profile_config = QueryServiceConfig {
        source_storage_profiles: vec![SourceStorageProfile::s3("  ", S3SourceStorageProfile::default_chain())],
        ..pool_config(device)
    };
    let empty_profile = QueryService::new_with_deployment(
        empty_profile_config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();
    assert_eq!(empty_profile.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(empty_profile.fact_value("reason"), Some("empty_profile_id"));

    let profile = SourceStorageProfile::s3("duplicate", S3SourceStorageProfile::default_chain());
    let duplicate_profile_config =
        QueryServiceConfig { source_storage_profiles: vec![profile.clone(), profile], ..pool_config(device) };
    let duplicate_profile = QueryService::new_with_deployment(
        duplicate_profile_config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();
    assert_eq!(duplicate_profile.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(duplicate_profile.fact_value("reason"), Some("duplicate_profile_id"));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        None,
        "resource-independent tenant and storage-profile validation must precede allocator installation",
    );

    DeviceMemoryManager::install_for_device(device).unwrap();
    let conflict = QueryService::new_with_deployment(
        pool_config(device),
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(conflict.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(conflict.fact_value("reason"), Some("device_memory_manager_options_conflict"));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        Some(DeviceMemoryManagerOptions::default()),
        "a conflicting service must not replace the first-installed allocator",
    );
}
