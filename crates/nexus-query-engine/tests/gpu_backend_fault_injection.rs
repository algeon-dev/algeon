use std::time::{Duration, Instant};

use nexus_query_engine::runtime::{
    DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
    GpuBackendDeviceConfig, GpuDeviceMemoryLimit, GpuMemoryOwnership, KvikioCompatModeStatus, NvidiaFsProbeStatus,
    QueryExecutionConfig, QueueBounds,
};

const DEVICE_LIMIT_BYTES: u64 = 512 * 1024 * 1024;
const DEVICE_FLOOR_BYTES: u64 = 64 * 1024 * 1024;

fn nongds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

fn device(ordinal: usize) -> GpuBackendDeviceConfig {
    GpuBackendDeviceConfig::new(ordinal, GpuMemoryOwnership::WholeDeviceExclusive)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: DEVICE_LIMIT_BYTES })
        .with_attempt_floor_bytes(DEVICE_FLOOR_BYTES)
        .with_max_active_queries(1)
        .with_installation_bounds(2, 1, 1)
}

#[test]
fn public_fault_injector_quarantines_one_device_and_preserves_healthy_fallback() {
    let available_devices =
        cudf::DeviceId::count().expect("CUDA device count is available for physical fault-injection validation");
    assert!(
        available_devices >= 2,
        "non-qualifying physical fault-injection validation requires at least two CUDA devices; found {available_devices}"
    );

    let controller = GpuBackendControllerDraft::new(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 1 },
    );
    let core =
        GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device(0), device(1)], controller, nongds_deployment()))
            .build()
            .expect("two-device backend builds");

    let session = core.session_capability();
    let pre_ack = session.admit_query().expect("target device admits work before controlled quarantine");
    assert_eq!(pre_ack.device_ordinal(), 0, "the deterministic first placement supplies the live target grant");
    let reports = session.device_ledger_reports().expect("ledger reports remain public");
    let target_before = reports.iter().find(|report| report.device_ordinal == 0).expect("target ledger is present");
    assert_eq!(target_before.active_attempts, 1);
    assert_eq!(target_before.grants.len(), 1, "the pre-ack target grant remains live across the barrier");

    let injection = core.fault_injector().quarantine_device(0).expect("public controlled quarantine succeeds");
    assert_eq!(injection.device_ordinal(), 0);
    assert_eq!(injection.device_health_generation(), 1);
    assert!(injection.newly_quarantined());

    let repeated = core.fault_injector().quarantine_device(0).expect("controlled quarantine is idempotent");
    assert_eq!(repeated.device_health_generation(), 1);
    assert!(!repeated.newly_quarantined());

    let reports = session.device_ledger_reports().expect("ledger reports remain public");
    assert_eq!(
        reports.iter().find(|report| report.device_ordinal == 0).map(|report| report.health.as_str()),
        Some("closed")
    );
    assert_eq!(
        reports.iter().find(|report| report.device_ordinal == 1).map(|report| report.health.as_str()),
        Some("open")
    );
    assert_eq!(
        reports.iter().find(|report| report.device_ordinal == 0).map(|report| report.grants.len()),
        Some(1),
        "quarantine closes future placement but does not revoke the pre-ack immutable grant"
    );

    let post_ack = session.admit_query().expect("healthy device remains admissible after quarantine acknowledgement");
    assert_eq!(post_ack.device_ordinal(), 1, "post-ack placement falls back from the quarantined device");
    post_ack.finish();
    drop(post_ack);
    pre_ack.finish();
    drop(pre_ack);

    let release_deadline = Instant::now() + Duration::from_secs(10);
    let target_after = loop {
        let released = session.device_ledger_reports().expect("released target ledger is observable");
        let target_after = released.iter().find(|report| report.device_ordinal == 0).expect("target ledger is present");
        if target_after.active_attempts == 0 && target_after.grants.is_empty() {
            break target_after.clone();
        }
        assert!(Instant::now() < release_deadline, "pre-ack work must release before the physical validation timeout");
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(target_after.active_attempts, 0, "pre-ack work releases normally after the quarantine barrier");
    assert!(target_after.grants.is_empty(), "no target grant remains before backend close");
    core.close_and_wait(Duration::from_secs(10)).expect("backend closes after controlled quarantine");
}
