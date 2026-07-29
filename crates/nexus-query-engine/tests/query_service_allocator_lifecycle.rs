use std::time::Duration;

use cudf::{DeviceId, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerOptions};
use nexus_query_engine::{
    error::QueryErrorCode,
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuBackendMemoryShare, GpuBackendTenantDraft, GpuMemoryOwnership,
        KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryMemoryDomainPolicy, QueryServiceConfig,
        QueryServiceTenantConfig, S3SourceStorageProfile, SourceStorageProfile, TenantId,
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

fn backend_draft(config: QueryServiceConfig, tenants: Vec<QueryServiceTenantConfig>) -> GpuBackendCoreDraft {
    let device_ordinal = config.device_ordinal;
    let max_active_queries = config.max_concurrent_queries.max(1);
    let stream_pool_size = config.stream_pool_size.unwrap_or(max_active_queries).max(1);
    let tenants = tenants
        .into_iter()
        .map(|tenant| {
            GpuBackendTenantDraft::new(
                tenant.tenant_id,
                GpuBackendMemoryShare::Bytes(tenant.min_device_budget_bytes),
                GpuBackendMemoryShare::Bytes(tenant.device_budget_bytes),
            )
        })
        .collect();
    let controller = GpuBackendControllerDraft::from_legacy_policy(config, tenants);
    let device = GpuBackendDeviceConfig::new(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)
        .with_max_active_queries(max_active_queries)
        .with_installation_bounds(stream_pool_size, max_active_queries, max_active_queries);
    GpuBackendCoreDraft::new(vec![device], controller, non_gds_deployment())
}

#[test]
fn rejected_configs_and_unidentified_allocator_do_not_mutate_the_allocator() {
    let device = DeviceId::current().unwrap();
    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), None);

    let config =
        QueryServiceConfig { query_memory_domain: Some(QueryMemoryDomainPolicy::default()), ..pool_config(device) };
    let error =
        backend_draft(config, vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)]).validate().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("query_memory_domain_requires_cuda_async_manager"));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        None,
        "validation must reject before the process-global allocator is installed",
    );

    let empty_tenants = backend_draft(pool_config(device), vec![]).validate().unwrap_err();
    assert_eq!(empty_tenants.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(empty_tenants.fact_value("reason"), Some("empty"));

    let invalid_tenant =
        backend_draft(pool_config(device), vec![QueryServiceTenantConfig::new("invalid", 0)]).validate().unwrap_err();
    assert_eq!(invalid_tenant.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(invalid_tenant.fact_value("reason"), Some("zero"));

    let duplicate_tenant = backend_draft(
        pool_config(device),
        vec![
            QueryServiceTenantConfig::new("duplicate", 8 * 1024 * 1024),
            QueryServiceTenantConfig::new("duplicate", 8 * 1024 * 1024),
        ],
    )
    .validate()
    .unwrap_err();
    assert_eq!(duplicate_tenant.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(duplicate_tenant.fact_value("reason"), Some("duplicate"));

    let empty_profile_config = QueryServiceConfig {
        source_storage_profiles: vec![SourceStorageProfile::s3("  ", S3SourceStorageProfile::default_chain())],
        ..pool_config(device)
    };
    let empty_profile =
        backend_draft(empty_profile_config, vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)])
            .validate()
            .unwrap_err();
    assert_eq!(empty_profile.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(empty_profile.fact_value("reason"), Some("empty_profile_id"));

    let profile = SourceStorageProfile::s3("duplicate", S3SourceStorageProfile::default_chain());
    let duplicate_profile_config =
        QueryServiceConfig { source_storage_profiles: vec![profile.clone(), profile], ..pool_config(device) };
    let duplicate_profile =
        backend_draft(duplicate_profile_config, vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)])
            .validate()
            .unwrap_err();
    assert_eq!(duplicate_profile.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(duplicate_profile.fact_value("reason"), Some("duplicate_profile_id"));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        None,
        "resource-independent tenant and storage-profile validation must precede allocator installation",
    );

    DeviceMemoryManager::install_for_device(device).unwrap();
    let incompatible = GpuBackendCore::builder(backend_draft(
        pool_config(device),
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
    ))
    .build()
    .unwrap_err();

    assert_eq!(incompatible.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(incompatible.fact_value("reason"), Some("installed_callback_unrecoverable"));
    assert_eq!(incompatible.fact_value("installation_status"), Some("Unidentified"));
    assert_eq!(incompatible.fact_value("installation_retry"), Some("PermanentlyUnusable"));
    assert_eq!(incompatible.fact_value("installation_cause"), Some("FailureCallbackMissing"));
    assert_eq!(incompatible.fact_value("device_ordinal"), Some(device.ordinal().to_string().as_str()));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        Some(DeviceMemoryManagerOptions::default()),
        "an unidentified allocator must not be replaced by backend construction",
    );
}

#[test]
fn owned_allocator_options_conflict_is_typed_and_does_not_mutate_the_allocator() {
    let device = DeviceId::current().unwrap();
    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), None);

    let installed_config = pool_config(device);
    let installed_options =
        DeviceMemoryManagerOptions { base: installed_config.device_memory_manager_base, allocation_limit_bytes: None };
    let backend = GpuBackendCore::builder(backend_draft(
        installed_config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
    ))
    .build()
    .unwrap();
    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), Some(installed_options));

    let close = backend.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
    drop(backend);

    let requested_options = DeviceMemoryManagerOptions {
        base: DeviceMemoryManagerBase::Pool {
            initial_pool_size: 128 * 1024 * 1024,
            maximum_pool_size: Some(128 * 1024 * 1024),
        },
        allocation_limit_bytes: None,
    };
    let conflicting_config = QueryServiceConfig {
        device_ordinal: device.ordinal(),
        device_memory_manager_base: requested_options.base,
        ..QueryServiceConfig::default()
    };
    let conflict = GpuBackendCore::builder(backend_draft(
        conflicting_config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
    ))
    .build()
    .unwrap_err();

    assert_eq!(conflict.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(conflict.fact_value("reason"), Some("device_memory_manager_options_conflict"));
    assert_eq!(conflict.fact_value("installation_status"), Some("Owned"));
    assert_eq!(conflict.fact_value("installation_retry"), Some("Reusable"));
    assert_eq!(conflict.fact_value("installation_cause"), Some("OptionsMismatch"));
    assert_eq!(conflict.fact_value("existing_options"), Some(format!("{installed_options:?}").as_str()));
    assert_eq!(conflict.fact_value("requested_options"), Some(format!("{requested_options:?}").as_str()));
    assert_eq!(conflict.fact_value("device_ordinal"), Some(device.ordinal().to_string().as_str()));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        Some(installed_options),
        "a conflicting backend must not replace the identified allocator",
    );
}

#[test]
fn compatible_allocator_reuse_preserves_authoritative_installation_identity_across_backend_generations() {
    let device = DeviceId::current().unwrap();
    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), None);

    let first = GpuBackendCore::builder(backend_draft(
        pool_config(device),
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
    ))
    .build()
    .unwrap();
    let first_identity = first.identity();
    let first_pools = first.session_capability().gpu_execution_pool_reports().unwrap();
    let [first_pool] = first_pools.as_slice() else {
        panic!("expected one worker pool for the single configured device");
    };
    let first_pool_id = first_pool.pool_id;
    let [first_record] = first.device_startup_records() else {
        panic!("expected one startup record for the single configured device");
    };
    let first_record = first_record.clone();
    let first_installation = DeviceMemoryManager::installation_state_for_device(device).unwrap();

    assert_eq!(first_record.memory_manager_installation, first_installation);
    assert_eq!(first_installation.device_ordinal, device.ordinal());
    assert_eq!(first_installation.status, cudf::DeviceMemoryManagerInstallationStatus::Owned);
    assert_eq!(first_installation.retry, cudf::DeviceMemoryManagerInstallationRetry::Reusable);
    assert!(first_installation.identity.is_some(), "an owned reusable allocator must retain its installation identity");
    assert!(
        first_installation.installation_generation.is_some_and(|generation| generation > 0),
        "an owned reusable allocator must retain its authoritative installation generation: {first_installation:?}"
    );
    assert!(first_record.device_health_generation > 0);
    assert!(first_record.backend_health_generation > 0);
    assert!(first_pool_id > 0, "the first backend generation must publish a stable worker-pool identity");

    let close = first.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
    drop(first);

    let second = GpuBackendCore::builder(backend_draft(
        pool_config(device),
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
    ))
    .build()
    .unwrap();
    let second_identity = second.identity();
    let second_pools = second.session_capability().gpu_execution_pool_reports().unwrap();
    let [second_pool] = second_pools.as_slice() else {
        panic!("expected one worker pool for the single configured device");
    };
    let second_pool_id = second_pool.pool_id;
    let [second_record] = second.device_startup_records() else {
        panic!("expected one startup record for the single configured device");
    };
    let second_record = second_record.clone();
    let second_installation = DeviceMemoryManager::installation_state_for_device(device).unwrap();

    assert!(
        second_identity.resource_generation > first_identity.resource_generation,
        "a newly published backend must receive a later resource generation"
    );
    assert_ne!(second_identity.resource_core, first_identity.resource_core);
    assert_ne!(second_identity.observation_core, first_identity.observation_core);
    assert_ne!(
        second_pool_id, first_pool_id,
        "a separately published backend generation must construct a distinct worker pool"
    );
    assert_eq!(second_installation, first_installation, "compatible backend generations must reuse the same allocator");
    assert_eq!(second_record.memory_manager_installation, second_installation);
    assert_eq!(second_record.memory_manager_installation.identity, first_record.memory_manager_installation.identity);
    assert_eq!(
        second_record.memory_manager_installation.installation_generation,
        first_record.memory_manager_installation.installation_generation,
        "the process allocator generation is stable when an identical backend reuses it"
    );
    assert_eq!(
        second_record.device_health_generation, first_record.device_health_generation,
        "device health remains bound to the reused allocator installation"
    );
    assert!(
        second_record.backend_health_generation > first_record.backend_health_generation,
        "each published backend retains a distinct authoritative backend generation"
    );

    let close = second.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_clone_sessions_share_identity_ledger_worker_pools_and_startup_records() {
    let device = DeviceId::current().unwrap();
    let config = QueryServiceConfig {
        max_concurrent_queries: 1,
        max_admission_queue_len: 2,
        admission_wait_timeout: Duration::from_secs(10),
        ..pool_config(device)
    };
    let core =
        GpuBackendCore::builder(backend_draft(config, vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)]))
            .build()
            .unwrap();
    let cloned_core = core.clone();
    let first_session = core.session_capability();
    let second_session = cloned_core.session_capability();
    let identity = core.identity();

    assert_eq!(cloned_core.identity(), identity);
    assert_eq!(first_session.identity(), identity);
    assert_eq!(second_session.identity(), identity);
    assert!(first_session.same_backend(&second_session));
    assert!(identity.resource_core > 0);
    assert!(identity.observation_core > 0);
    assert!(identity.resource_generation > 0);
    assert_eq!(core.device_startup_records(), cloned_core.device_startup_records());

    let first_device_records = first_session.device_admission_snapshots().unwrap();
    let second_device_records = second_session.device_admission_snapshots().unwrap();
    assert_eq!(first_device_records, second_device_records);
    assert_eq!(first_device_records.len(), 1);
    assert_eq!(first_device_records[0].device_ordinal, device.ordinal());
    assert_eq!(first_device_records[0].process_active_query_slots, 0);

    let first_worker_pools = first_session.gpu_execution_pool_reports().unwrap();
    let second_worker_pools = second_session.gpu_execution_pool_reports().unwrap();
    assert_eq!(first_worker_pools, second_worker_pools);
    assert_eq!(first_worker_pools.len(), 1);
    assert!(first_worker_pools[0].pool_id > 0);
    assert_eq!(first_worker_pools[0].pool_id, second_worker_pools[0].pool_id);
    assert_eq!(first_worker_pools[0].device_ordinal, device.ordinal());
    assert_eq!(first_worker_pools[0].worker_count, 1);

    let tenant = TenantId::new("tenant");
    let first = first_session.admit_query(tenant.clone()).unwrap();
    let first_sequence = first.admission_sequence();
    let queued_session = second_session.clone();
    let queued = tokio::spawn(async move { queued_session.admit_async_with_request(tenant, Default::default()).await });

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let first_snapshot = first_session.admission_snapshot().unwrap();
        let second_snapshot = second_session.admission_snapshot().unwrap();
        if first_snapshot.active_queries == 1 && first_snapshot.queued_queries == 1 {
            assert_eq!(second_snapshot, first_snapshot, "cloned sessions must observe one shared admission ledger");
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "second cloned session did not queue behind the first session's admission: first={first_snapshot:?}, second={second_snapshot:?}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    first.finish();
    drop(first);
    let queued = tokio::time::timeout(Duration::from_secs(10), queued)
        .await
        .expect("queued cloned-session admission must wake after the shared slot is released")
        .expect("queued cloned-session task must not panic")
        .expect("queued cloned-session admission must succeed");
    assert!(queued.admission_sequence() > first_sequence);
    queued.finish();
    drop(queued);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let snapshot = second_session.admission_snapshot().unwrap();
        if snapshot.active_queries == 0 && snapshot.queued_queries == 0 {
            break;
        }
        assert!(tokio::time::Instant::now() < deadline, "shared admission ledger did not drain: {snapshot:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    drop(first_session);
    drop(second_session);
    let close = core.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
    drop(cloned_core);
}
