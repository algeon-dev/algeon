//! Model tests for the fixed-cap per-device immutable-grant ledger.

use std::sync::{Arc, Mutex};

use super::*;
use crate::{
    observability::{AdmissionRequestId, AttemptId},
    runtime::gpu::{DeviceCapabilities, DeviceResourceEnvelope, LedgerTransitionKind},
};

struct StreamSynchronizeHookReset;

impl StreamSynchronizeHookReset {
    fn new() -> Self {
        unsafe { cudf_sys::test_support::cudf_test_stream_synchronize_reset() };
        Self
    }
}

impl Drop for StreamSynchronizeHookReset {
    fn drop(&mut self) {
        unsafe { cudf_sys::test_support::cudf_test_stream_synchronize_reset() };
    }
}

fn stream_synchronize_call_count() -> i32 {
    let mut count = 0;
    let status = unsafe { cudf_sys::test_support::cudf_test_stream_synchronize_call_count(&mut count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    count
}

fn profile(
    managed_capacity_bytes: u64,
    default_query_limit_bytes: u64,
    max_active_attempts: usize,
) -> DeviceResourceProfile {
    let max_active_attempts = max_active_attempts.max(1);
    let slots = max_active_attempts.saturating_mul(2);
    DeviceResourceProfile::resolve(DeviceResourceEnvelope {
        device_ordinal: 0,
        device_capabilities: DeviceCapabilities::default(),
        configured_limit_bytes: managed_capacity_bytes,
        headroom_bytes: 0,
        backend_reserve_bytes: 0,
        default_query_limit_bytes,
        max_active_attempts,
        source_work_per_attempt: 1,
        stream_pool_size: slots,
        worker_job_capacity: slots,
        cache_cap_bytes: 0,
    })
    .expect("test profile must resolve")
}

fn cache_profile(
    managed_capacity_bytes: u64,
    default_query_limit_bytes: u64,
    max_active_attempts: usize,
    cache_cap_bytes: u64,
) -> DeviceResourceProfile {
    let max_active_attempts = max_active_attempts.max(1);
    let slots = max_active_attempts.saturating_mul(2);
    DeviceResourceProfile::resolve(DeviceResourceEnvelope {
        device_ordinal: 0,
        device_capabilities: DeviceCapabilities::default(),
        configured_limit_bytes: managed_capacity_bytes,
        headroom_bytes: 0,
        backend_reserve_bytes: 0,
        default_query_limit_bytes,
        max_active_attempts,
        source_work_per_attempt: 1,
        stream_pool_size: slots,
        worker_job_capacity: slots,
        cache_cap_bytes,
    })
    .expect("test profile must resolve")
}

fn request(id: u64) -> AdmissionRequestId {
    AdmissionRequestId::new(id)
}

#[test]
fn explicit_query_caps_are_immutable_and_not_clamped_to_transient_capacity() {
    let ledger = DeviceLedger::new(profile(1_000, 800, 2));
    let first = ledger.try_admit(request(1), Some(400)).unwrap().expect("first cap admits");

    assert_eq!(first.budget_bytes(), 400);
    assert_eq!(ledger.snapshot().unwrap().live_grants_bytes, 400);
    assert!(ledger.try_admit(request(2), Some(700)).unwrap().is_none());
    assert_eq!(ledger.candidate_query_cap_bytes(Some(700)).unwrap(), None);

    drop(first);
    let second = ledger.try_admit(request(2), Some(700)).unwrap().expect("cap fits after release");
    assert_eq!(second.budget_bytes(), 700);
}

#[test]
fn default_query_limit_is_not_clamped_to_transient_capacity() {
    let ledger = DeviceLedger::new(profile(1_000, 800, 2));
    let first = ledger.try_admit(request(1), Some(400)).unwrap().expect("first cap admits");

    assert!(ledger.try_admit(request(2), None).unwrap().is_none());
    assert_eq!(ledger.candidate_query_cap_bytes(None).unwrap(), None);

    drop(first);
    let default_grant = ledger.try_admit(request(2), None).unwrap().expect("default cap fits after release");
    assert_eq!(default_grant.budget_bytes(), 800);
}

#[test]
fn frozen_resident_transition_moves_budget_to_resident_and_preserves_cap_boundary() {
    let ledger = DeviceLedger::new(profile(1_000, 800, 2));
    let grant = ledger.try_admit(request(1), Some(700)).unwrap().expect("grant admits");

    let before = ledger.snapshot().unwrap();
    assert_eq!(before.active_attempts, 1);
    assert_eq!(before.live_grants_bytes, 700);
    assert_eq!(before.frozen_resident_bytes, 0);

    grant.freeze_resident(400).unwrap();
    assert!(grant.is_frozen_resident());
    let frozen = ledger.snapshot().unwrap();
    assert_eq!(frozen.active_attempts, 0);
    assert_eq!(frozen.live_grants_bytes, 0);
    assert_eq!(frozen.frozen_resident_bytes, 400);
    assert_eq!(frozen.admissible_query_cap_bytes(None), Some(600));

    let report = ledger.report().unwrap();
    assert_eq!(report.grants.len(), 1);
    assert_eq!(report.grants[0].grant_bytes, 700);
    assert_eq!(report.grants[0].frozen_resident_bytes, Some(400));

    let next = ledger.try_admit(request(2), None).unwrap().expect("resident headroom fits default cap");
    assert_eq!(next.budget_bytes(), 600);
    assert_eq!(ledger.snapshot().unwrap().frozen_resident_bytes, 400);
    assert_eq!(ledger.snapshot().unwrap().live_grants_bytes, 600);

    drop(next);
    drop(grant);
    let released = ledger.snapshot().unwrap();
    assert_eq!(released.frozen_resident_bytes, 0);
    assert_eq!(released.live_grants_bytes, 0);
}

#[test]
fn frozen_resident_accepts_empty_outputs_but_remains_bounded_and_terminal() {
    let empty_ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let empty_grant = empty_ledger.try_admit(request(1), Some(500)).unwrap().expect("empty grant admits");
    empty_grant.freeze_resident(0).unwrap();
    assert!(empty_grant.retain_resident_owner(1).is_err());

    let ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let grant = ledger.try_admit(request(2), Some(500)).unwrap().expect("bounded grant admits");
    assert!(grant.freeze_resident(501).is_err());
    grant.freeze_resident(300).unwrap();
    grant.freeze_resident(300).unwrap();
    assert!(grant.freeze_resident(301).is_err());
    assert!(grant.retain_resident_owner(1).is_err());
}

#[test]
fn resident_owners_are_observation_not_an_allocation_gate() {
    let ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let grant = ledger.try_admit(request(1), Some(300)).unwrap().expect("grant admits");

    let first = grant.retain_resident_owner(250).unwrap().expect("positive owner bytes");
    let second = grant.retain_resident_owner(1).unwrap().expect("positive owner bytes");
    assert_eq!(grant.resident_owner_bytes(), 251);
    assert_eq!(ledger.snapshot().unwrap().resident_owner_bytes, 251);

    drop(first);
    drop(second);
    assert_eq!(grant.resident_owner_bytes(), 0);
    assert_eq!(ledger.snapshot().unwrap().resident_owner_bytes, 0);
}

#[test]
fn zero_query_caps_are_rejected_and_closed_admission_does_not_mutate_grants() {
    let ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let error = ledger.try_admit(request(1), Some(0)).expect_err("zero cap must fail closed");
    assert_eq!(error.fact_value("reason"), Some("zero_query_cap"));
    assert_eq!(ledger.snapshot().unwrap().sequence, 0);

    ledger.close_admission("test closure").unwrap();
    assert!(ledger.try_admit(request(2), Some(100)).unwrap().is_none());
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.active_attempts, 0);
    assert!(matches!(snapshot.health, DeviceAdmissionHealth::Closed { .. }));
}

#[test]
fn grant_identity_is_preserved_for_fixed_cap_admission() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 2));
    let attempt_id = AttemptId::mint();
    let grant = ledger.try_admit_for_attempt(request(9), attempt_id, Some(250)).unwrap().expect("fixed cap admits");

    assert_eq!(grant.identity().attempt_id, attempt_id);
    assert_eq!(grant.identity().generation, grant.grant_id());
    assert_eq!(grant.admission_request_id(), request(9));
    assert_eq!(grant.budget_bytes(), 250);
}

#[test]
fn transition_snapshots_expose_budget_to_resident_boundary() {
    let ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let observed_kinds = Arc::clone(&kinds);
    ledger
        .register_transition_observer(Arc::new(move |transition| {
            observed_kinds.lock().expect("transition log").push(transition.kind);
        }))
        .unwrap();

    let grant = ledger.try_admit(request(1), Some(500)).unwrap().expect("grant admits");
    grant.freeze_resident(200).unwrap();

    assert_eq!(
        kinds.lock().unwrap().as_slice(),
        &[LedgerTransitionKind::GrantAdmitted, LedgerTransitionKind::GrantFrozenResident],
    );
}

#[test]
fn prepared_resident_transition_keeps_full_grant_until_report_publication() {
    let ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let grant = ledger.try_admit(request(1), Some(800)).unwrap().expect("grant admits");

    grant.prepare_frozen_resident_for_test(200).unwrap();
    let before_report = ledger.snapshot().unwrap();
    assert_eq!(before_report.active_attempts, 1);
    assert_eq!(before_report.live_grants_bytes, 800);
    assert_eq!(before_report.frozen_resident_bytes, 0);
    assert!(ledger.try_admit(request(2), Some(1)).unwrap().is_none());

    grant.publish_frozen_resident().unwrap();
    let after_report = ledger.snapshot().unwrap();
    assert_eq!(after_report.active_attempts, 0);
    assert_eq!(after_report.live_grants_bytes, 0);
    assert_eq!(after_report.frozen_resident_bytes, 200);
    let exact_remaining = ledger.try_admit(request(2), Some(800)).unwrap().expect("remaining capacity admits");
    assert_eq!(exact_remaining.budget_bytes(), 800);
}

#[test]
fn final_resident_consumer_fence_retains_capacity_until_completion() {
    let _reset = StreamSynchronizeHookReset::new();
    let ledger = DeviceLedger::new(profile(1_000, 800, 2));
    let grant = ledger.try_admit(request(1), Some(800)).unwrap().expect("producer grant admits");
    let producer_owner = grant.retain_resident_owner(200).expect("resident owner").expect("positive owner bytes");
    producer_owner
        .set_completion_stream(cudf::runtime::CudaStream::new().expect("completion stream"))
        .expect("completion stream binds once");
    let consumer_owner = producer_owner.clone();
    grant.freeze_resident(200).expect("resident charge freezes");
    drop(producer_owner);
    drop(grant);
    assert_eq!(ledger.snapshot().unwrap().frozen_resident_bytes, 200);

    unsafe { cudf_sys::test_support::cudf_test_stream_block_synchronize() };
    let release = std::thread::spawn(move || drop(consumer_owner));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while stream_synchronize_call_count() == 0 {
        assert!(std::time::Instant::now() < deadline, "final resident owner reaches its completion fence");
        std::thread::yield_now();
    }
    assert!(!release.is_finished(), "the final owner waits for consumer completion");
    assert_eq!(ledger.snapshot().unwrap().frozen_resident_bytes, 200);
    assert!(ledger.try_admit(request(2), Some(801)).unwrap().is_none());

    unsafe { cudf_sys::test_support::cudf_test_stream_unblock_synchronize() };
    release.join().expect("final resident release completes");
    assert_eq!(ledger.snapshot().unwrap().frozen_resident_bytes, 0);
    let replacement = ledger.try_admit(request(3), Some(1_000)).unwrap().expect("released capacity is reusable");
    drop(replacement);
}

#[test]
fn resident_completion_failure_closes_device_admission() {
    let _reset = StreamSynchronizeHookReset::new();
    let ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let grant = ledger.try_admit(request(1), Some(800)).unwrap().expect("producer grant admits");
    let resident_owner = grant.retain_resident_owner(200).expect("resident owner").expect("positive owner bytes");
    resident_owner
        .set_completion_stream(cudf::runtime::CudaStream::new().expect("completion stream"))
        .expect("completion stream binds once");
    grant.freeze_resident(200).expect("resident charge freezes");
    drop(grant);

    unsafe { cudf_sys::test_support::cudf_test_stream_fail_synchronize() };
    drop(resident_owner);
    let snapshot = ledger.snapshot().expect("ledger snapshot after failed fence");
    let DeviceAdmissionHealth::Closed { reason, .. } = snapshot.health else {
        panic!("a failed resident completion fence must close device admission");
    };
    assert!(reason.contains("resident completion fence failed"));
    assert!(ledger.try_admit(request(2), Some(1)).unwrap().is_none());
}

#[test]
fn cache_residency_remains_separate_from_fixed_query_grants() {
    let ledger = DeviceLedger::new(cache_profile(1_000, 600, 1, 300));
    assert!(ledger.try_reserve_evictable_bytes(300).unwrap());
    assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, 300);

    let grant = ledger.try_admit(request(1), Some(100)).unwrap().expect("query cap fits with cache charge");
    assert_eq!(grant.budget_bytes(), 100);
    assert_eq!(ledger.snapshot().unwrap().live_grants_bytes, 100);
    assert!(!ledger.try_admit(request(2), Some(601)).unwrap().is_some());
}
