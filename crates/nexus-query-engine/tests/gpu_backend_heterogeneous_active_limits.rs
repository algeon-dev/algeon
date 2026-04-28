use std::time::Duration;

use nexus_query_engine::runtime::{
    DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
    GpuBackendDeviceConfig, GpuBackendMemoryShare, GpuBackendTenantDraft, GpuDeviceMemoryLimit, GpuMemoryOwnership,
    KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryAdmissionRequest, QueryServiceConfig, SlotGuarantee, TenantId,
};

const DEVICE_CAPACITY_BYTES: u64 = 64 * 1024 * 1024;
const QUERY_GRANT_BYTES: u64 = 4 * 1024 * 1024;

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

#[test]
fn heterogeneous_device_active_limits_are_not_collapsed_to_the_largest_cap() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }

    let controller = GpuBackendControllerDraft::from_legacy_policy(
        QueryServiceConfig {
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_millis(250),
            ..QueryServiceConfig::default()
        },
        vec![GpuBackendTenantDraft::new(
            "tenant",
            GpuBackendMemoryShare::Bytes(QUERY_GRANT_BYTES),
            GpuBackendMemoryShare::Bytes(QUERY_GRANT_BYTES),
        )],
    );
    let device = |device_ordinal, max_active_queries| {
        GpuBackendDeviceConfig::new(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: DEVICE_CAPACITY_BYTES })
            .with_max_active_queries(max_active_queries)
            .with_installation_bounds(max_active_queries, max_active_queries, max_active_queries)
    };
    let core = GpuBackendCore::builder(GpuBackendCoreDraft::new(
        vec![device(0, 2), device(1, 1)],
        controller,
        non_gds_deployment(),
    ))
    .build()
    .unwrap();
    let session = core.session_capability();
    let tenant = TenantId::new("tenant");

    let first = session
        .try_admit_query_with_request(tenant.clone(), QueryAdmissionRequest::new())
        .unwrap()
        .expect("first query admits");
    let second = session
        .try_admit_query_with_request(tenant.clone(), QueryAdmissionRequest::new())
        .unwrap()
        .expect("second query admits");
    let third = session
        .try_admit_query_with_request(tenant.clone(), QueryAdmissionRequest::new())
        .unwrap()
        .expect("third query admits");
    assert_eq!((first.device_ordinal(), second.device_ordinal(), third.device_ordinal()), (0, 1, 0));
    assert!(
        session.try_admit_query_with_request(tenant, QueryAdmissionRequest::new()).unwrap().is_none(),
        "the device-1 cap remains one instead of inheriting device 0's cap of two",
    );

    let snapshots = session.device_admission_snapshots().unwrap();
    assert_eq!(snapshots.iter().map(|snapshot| snapshot.process_active_query_slots).collect::<Vec<_>>(), vec![2, 1]);

    first.finish();
    second.finish();
    third.finish();
    core.close_and_wait(Duration::from_secs(5)).unwrap();
}
