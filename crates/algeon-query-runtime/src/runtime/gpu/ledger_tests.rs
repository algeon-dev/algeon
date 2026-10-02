//! Model tests for the fixed-cap per-device immutable-grant ledger.

use std::sync::{Arc, Mutex};

use cudf::runtime::CudaStream;
use cudf_sys::{CUDF_SUCCESS, test_support};

use super::*;
use crate::{
    observability::{AdmissionRequestId, AttemptId},
    runtime::gpu::{DeviceCapabilities, DeviceResourceEnvelope, LedgerTransitionKind},
};

struct EventSynchronizeHookReset;

impl EventSynchronizeHookReset {
    fn new() -> Self {
        unsafe { test_support::cudf_test_event_reset_observation() };
        Self
    }
}

impl Drop for EventSynchronizeHookReset {
    fn drop(&mut self) {
        unsafe { test_support::cudf_test_event_reset_observation() };
    }
}

struct StreamSynchronizeHookReset;

impl StreamSynchronizeHookReset {
    fn new() -> Self {
        unsafe { test_support::cudf_test_stream_synchronize_reset() };
        Self
    }
}

impl Drop for StreamSynchronizeHookReset {
    fn drop(&mut self) {
        unsafe { test_support::cudf_test_stream_synchronize_reset() };
    }
}

fn event_synchronize_call_count() -> i32 {
    let mut count = 0;
    let status = unsafe { test_support::cudf_test_event_synchronize_call_count(&mut count) };
    assert_eq!(status, CUDF_SUCCESS);
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

/// The capacity invariant holds across the whole device, not just pairwise:
/// frozen resident bytes plus every simultaneously committed query cap stay
/// within the managed capacity, and the ledger refuses the first byte past it.
#[test]
fn frozen_resident_bytes_plus_every_committed_cap_stay_within_managed_capacity() {
    const MANAGED_CAPACITY_BYTES: u64 = 1_000;
    let ledger = DeviceLedger::new(profile(MANAGED_CAPACITY_BYTES, 300, 4));
    let producer = ledger.try_admit(request(1), Some(400)).unwrap().expect("producer admits");
    producer.freeze_resident(250).unwrap();

    let small = ledger.try_admit(request(2), Some(300)).unwrap().expect("small cap admits into resident headroom");
    let large = ledger.try_admit(request(3), Some(450)).unwrap().expect("large cap admits into resident headroom");
    assert_eq!(small.budget_bytes(), 300);
    assert_eq!(large.budget_bytes(), 450);

    let committed = ledger.snapshot().unwrap();
    assert_eq!(committed.frozen_resident_bytes, 250);
    assert_eq!(committed.live_grants_bytes, 750);
    assert_eq!(committed.frozen_resident_bytes + committed.live_grants_bytes, MANAGED_CAPACITY_BYTES);
    assert_eq!(committed.active_attempts, 2, "the frozen producer no longer holds an active attempt");
    assert!(ledger.try_admit(request(4), Some(1)).unwrap().is_none(), "nothing fits once the invariant is tight");

    drop(small);
    let after_release = ledger.snapshot().unwrap();
    assert_eq!(after_release.frozen_resident_bytes, 250);
    assert_eq!(after_release.live_grants_bytes, 450);
    assert_eq!(after_release.admissible_query_cap_bytes(Some(300)), Some(300));

    drop(large);
    drop(producer);
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
fn grant_release_completes_after_the_ledger_lock_is_poisoned() {
    let ledger = DeviceLedger::new(profile(1_000, 800, 2));
    let released_live_bytes = Arc::new(Mutex::new(Vec::new()));
    let observed_live_bytes = Arc::clone(&released_live_bytes);
    ledger
        .register_admission_observer(Arc::new(move |transition| {
            observed_live_bytes.lock().expect("release log").push(transition.snapshot.live_grants_bytes);
        }))
        .unwrap();
    let grant = ledger.try_admit(request(1), Some(400)).unwrap().expect("grant admits");

    let poisoner = ledger.clone();
    std::thread::spawn(move || {
        let _state = poisoner.inner.state.lock().unwrap();
        panic!("poison the ledger state mutex");
    })
    .join()
    .expect_err("the poisoning thread panics while holding the ledger lock");
    assert!(ledger.snapshot().is_err(), "a poisoned ledger refuses snapshots and admission");

    drop(grant);

    let state = ledger.inner.state.lock().unwrap_or_else(PoisonError::into_inner);
    assert!(state.grants.is_empty(), "release removes the grant despite the poisoned lock");
    assert_eq!(ledger.snapshot_locked(&state).live_grants_bytes, 0);
    assert_eq!(released_live_bytes.lock().unwrap().as_slice(), &[0], "release observers still wake waiters");
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
    let _reset = EventSynchronizeHookReset::new();
    let ledger = DeviceLedger::new(profile(1_000, 800, 2));
    let grant = ledger.try_admit(request(1), Some(800)).unwrap().expect("producer grant admits");
    let producer_owner = grant.retain_resident_owner(200).expect("resident owner").expect("positive owner bytes");
    producer_owner.record_completion(&CudaStream::new().expect("completion stream")).expect("completion event records");
    let consumer_owner = producer_owner.clone();
    grant.freeze_resident(200).expect("resident charge freezes");
    drop(producer_owner);
    drop(grant);
    assert_eq!(ledger.snapshot().unwrap().frozen_resident_bytes, 200);

    unsafe { test_support::cudf_test_event_block_synchronize() };
    let release = std::thread::spawn(move || drop(consumer_owner));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while event_synchronize_call_count() == 0 {
        assert!(std::time::Instant::now() < deadline, "final resident owner reaches its completion fence");
        std::thread::yield_now();
    }
    assert!(!release.is_finished(), "the final owner waits for consumer completion");
    assert_eq!(ledger.snapshot().unwrap().frozen_resident_bytes, 200);
    assert!(ledger.try_admit(request(2), Some(801)).unwrap().is_none());

    unsafe { test_support::cudf_test_event_unblock_synchronize() };
    release.join().expect("final resident release completes");
    assert_eq!(ledger.snapshot().unwrap().frozen_resident_bytes, 0);
    let replacement = ledger.try_admit(request(3), Some(1_000)).unwrap().expect("released capacity is reusable");
    drop(replacement);
}

#[test]
fn final_resident_release_waits_for_its_completion_event_not_its_pooled_stream() {
    let _event_reset = EventSynchronizeHookReset::new();
    let _stream_reset = StreamSynchronizeHookReset::new();
    let ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let grant = ledger.try_admit(request(1), Some(800)).unwrap().expect("producer grant admits");
    let resident_owner = grant.retain_resident_owner(200).expect("resident owner").expect("positive owner bytes");
    resident_owner.record_completion(&CudaStream::new().expect("completion stream")).expect("completion event records");
    grant.freeze_resident(200).expect("resident charge freezes");
    drop(grant);

    // A blocked stream synchronization stands in for another query's later
    // work on the pooled producer stream.
    unsafe { test_support::cudf_test_stream_block_synchronize() };
    let (released_sender, released_receiver) = std::sync::mpsc::channel();
    let release = std::thread::spawn(move || {
        drop(resident_owner);
        released_sender.send(()).expect("test waits for the release");
    });
    let released = released_receiver.recv_timeout(std::time::Duration::from_secs(10));
    unsafe { test_support::cudf_test_stream_unblock_synchronize() };
    release.join().expect("final resident release completes");

    assert!(released.is_ok(), "the final release waits only for the work its event fences");
    assert_eq!(event_synchronize_call_count(), 1);
    assert_eq!(ledger.snapshot().unwrap().frozen_resident_bytes, 0);
}

#[test]
fn resident_completion_failure_closes_device_admission() {
    let _reset = EventSynchronizeHookReset::new();
    let ledger = DeviceLedger::new(profile(1_000, 800, 1));
    let grant = ledger.try_admit(request(1), Some(800)).unwrap().expect("producer grant admits");
    let resident_owner = grant.retain_resident_owner(200).expect("resident owner").expect("positive owner bytes");
    resident_owner.record_completion(&CudaStream::new().expect("completion stream")).expect("completion event records");
    grant.freeze_resident(200).expect("resident charge freezes");
    drop(grant);

    unsafe { test_support::cudf_test_event_fail_synchronize() };
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

const CONTRACT_GIB: u64 = 1024 * 1024 * 1024;

/// The contract fixture from the bounded-concurrency plan: `M = 24 GiB`,
/// `K = 8 GiB`, `N = 2`, so `Q = 16 GiB` and the derived default is 8 GiB.
fn contract_profile() -> DeviceResourceProfile {
    cache_profile(24 * CONTRACT_GIB, 8 * CONTRACT_GIB, 2, 8 * CONTRACT_GIB)
}

#[test]
fn derived_default_grants_admit_two_parallel_attempts_and_queue_the_third() {
    let ledger = DeviceLedger::new(contract_profile());
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.managed_capacity_bytes, 24 * CONTRACT_GIB);
    assert_eq!(snapshot.cache_cap_bytes, 8 * CONTRACT_GIB);
    assert_eq!(snapshot.query_share_bytes, 16 * CONTRACT_GIB);
    assert_eq!(snapshot.default_query_limit_bytes, 8 * CONTRACT_GIB);

    let first = ledger.try_admit(request(1), None).unwrap().expect("first default grant admits");
    assert_eq!(first.budget_bytes(), 8 * CONTRACT_GIB);
    let second = ledger.try_admit(request(2), None).unwrap().expect("second default grant admits");
    assert_eq!(second.budget_bytes(), 8 * CONTRACT_GIB);
    assert!(ledger.try_admit(request(3), None).unwrap().is_none(), "third default grant queues");
    assert!(ledger.try_admit(request(4), Some(1)).unwrap().is_none(), "no byte fits once Q is committed");

    assert!(ledger.try_reserve_evictable_bytes(8 * CONTRACT_GIB).unwrap(), "cache fills its own share");
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.cache_evictable_resident_bytes, 8 * CONTRACT_GIB);
    assert!(
        snapshot.frozen_resident_bytes
            + snapshot.live_grants_bytes
            + snapshot.cache_evictable_resident_bytes
            + snapshot.cache_pinned_in_use_bytes
            <= snapshot.managed_capacity_bytes,
        "grants plus cache never overcommit the device"
    );
    let report = ledger.report().unwrap();
    assert_eq!(report.available_bytes, 0);
    assert_eq!(report.free_bytes, 0);
}

#[test]
fn explicit_large_grants_serialize_without_spending_cache_bytes() {
    let ledger = DeviceLedger::new(contract_profile());
    assert!(ledger.try_reserve_evictable_bytes(8 * CONTRACT_GIB).unwrap(), "cache fills first");

    let first = ledger.try_admit(request(1), Some(12 * CONTRACT_GIB)).unwrap().expect("first 12 GiB admits into Q");
    assert_eq!(first.budget_bytes(), 12 * CONTRACT_GIB);
    assert!(
        ledger.try_admit(request(2), Some(12 * CONTRACT_GIB)).unwrap().is_none(),
        "second 12 GiB waits: 24 GiB of grants exceed Q = 16 GiB, and a full cache changes nothing"
    );
    assert_eq!(ledger.snapshot().unwrap().live_grants_bytes, 12 * CONTRACT_GIB);

    drop(first);
    let second = ledger.try_admit(request(2), Some(12 * CONTRACT_GIB)).unwrap().expect("second admits after release");
    assert_eq!(second.budget_bytes(), 12 * CONTRACT_GIB);
}

#[test]
fn cache_and_query_orders_promise_identical_capacity() {
    // Cache-first and query-first admission must agree: the cache owns only
    // `K` and queries own only `Q`, so neither order promises the same bytes
    // twice. `M = 10_000, K = 4_000, Q = 6_000`, default `5_000`, `N = 2`.
    let cache_first = DeviceLedger::new(cache_profile(10_000, 5_000, 2, 4_000));
    assert!(cache_first.try_reserve_evictable_bytes(4_000).unwrap());
    let first = cache_first.try_admit(request(1), None).unwrap().expect("first default admits");
    assert_eq!(first.budget_bytes(), 5_000);
    assert!(
        cache_first.try_admit(request(2), None).unwrap().is_none(),
        "second default queues: 10_000 of grants exceed Q = 6_000"
    );

    let query_first = DeviceLedger::new(cache_profile(10_000, 5_000, 2, 4_000));
    let only = query_first.try_admit(request(1), None).unwrap().expect("first default admits");
    assert_eq!(only.budget_bytes(), 5_000);
    assert!(
        query_first.try_reserve_evictable_bytes(4_000).unwrap(),
        "cache still fills its own share beside a live grant"
    );
    assert!(
        query_first.try_admit(request(2), None).unwrap().is_none(),
        "query-first order queues exactly like cache-first order"
    );
}

#[test]
fn explicit_caps_beyond_the_query_share_queue_instead_of_clamping() {
    let ledger = DeviceLedger::new(cache_profile(1_000, 700, 2, 300));
    assert_eq!(ledger.snapshot().unwrap().query_share_bytes, 700);
    assert!(!ledger.snapshot().unwrap().can_ever_fit_query_cap(Some(800)));
    assert!(
        ledger.try_admit(request(1), Some(800)).unwrap().is_none(),
        "an 800-byte cap never fits Q = 700 even though M = 1_000 has room"
    );
    let fits = ledger.try_admit(request(1), Some(700)).unwrap().expect("Q-exact cap admits");
    assert_eq!(fits.budget_bytes(), 700);
}

#[test]
fn frozen_resident_returns_grant_minus_resident_in_query_share_terms() {
    let ledger = DeviceLedger::new(cache_profile(16 * CONTRACT_GIB, 8 * CONTRACT_GIB, 2, 8 * CONTRACT_GIB));
    let grant = ledger.try_admit(request(1), Some(8 * CONTRACT_GIB)).unwrap().expect("grant admits");
    let owner = grant.retain_resident_owner(3 * CONTRACT_GIB).unwrap().expect("resident owner");
    grant.freeze_resident(3 * CONTRACT_GIB).unwrap();
    drop(owner);

    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.active_attempts, 0);
    assert_eq!(snapshot.live_grants_bytes, 0);
    assert_eq!(snapshot.frozen_resident_bytes, 3 * CONTRACT_GIB);

    // Only `G - R` returns: the frozen 3 GiB stays committed while the grant
    // lives, so the report projects `Q - R = 5 GiB` of headroom.
    let report = ledger.report().unwrap();
    assert_eq!(report.frozen_resident_bytes, 3 * CONTRACT_GIB);
    assert_eq!(report.available_bytes, 5 * CONTRACT_GIB);
    assert_eq!(report.free_bytes, 5 * CONTRACT_GIB);

    drop(grant);
    let released = ledger.snapshot().unwrap();
    assert_eq!(released.frozen_resident_bytes, 0);
    assert_eq!(released.live_grants_bytes, 0);
}

#[test]
fn failed_attempt_capacity_returns_only_after_cleanup_while_survivors_hold_theirs() {
    use std::sync::{Arc, Barrier};

    // M = 1_000, K = 400, Q = 600, N = 3, default 300: two defaults exactly
    // fill the query share beside a full cache.
    let ledger = Arc::new(DeviceLedger::new(cache_profile(1_000, 300, 3, 400)));
    assert!(ledger.try_reserve_evictable_bytes(400).unwrap(), "cache fills first");
    let survivor = ledger.try_admit(request(1), None).unwrap().expect("survivor admits");
    let failing = ledger.try_admit(request(2), None).unwrap().expect("failing attempt admits");
    assert!(
        ledger.try_admit(request(3), None).unwrap().is_none(),
        "900 of grants exceed Q = 600 even though M still shows room"
    );

    // Delayed cleanup holds the failed attempt's whole grant: barriers, not
    // sleeps, prove the capacity is not reissued until the grant drops.
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let cleanup = std::thread::spawn({
        let entered = Arc::clone(&entered);
        let release = Arc::clone(&release);
        move || {
            entered.wait();
            release.wait();
            drop(failing);
        }
    });
    entered.wait();
    assert!(
        ledger.try_admit(request(4), None).unwrap().is_none(),
        "cleanup holds its capacity until the grant actually drops"
    );
    assert_eq!(ledger.snapshot().unwrap().live_grants_bytes, 600);
    release.wait();
    cleanup.join().unwrap();

    let replacement = ledger.try_admit(request(5), None).unwrap().expect("capacity returns after cleanup");
    assert_eq!(replacement.budget_bytes(), 300);
    assert_eq!(survivor.budget_bytes(), 300, "the surviving attempt is never disturbed");
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.live_grants_bytes, 600);
    assert_eq!(snapshot.cache_evictable_resident_bytes, 400, "the full cache survives beside both grants");
}

#[test]
fn resident_owners_share_the_grant_generation_lease() {
    use std::sync::Arc;

    use crate::runtime::context::{QueryResourceEpoch, QueryServiceEpoch, QueryServiceResourceEpoch};

    let ledger = DeviceLedger::new(profile(1_000, 800, 2));
    let grant = ledger.try_admit(request(1), Some(300)).unwrap().expect("grant admits");
    let unleased = grant.retain_resident_owner(100).unwrap().expect("unleased owner");
    assert!(!unleased.has_resource_epoch());

    let service_epoch = QueryServiceEpoch::new(None);
    let query_epoch = QueryResourceEpoch::new(None);
    let anchor = query_epoch.acquire(Arc::new(()), "test.resident_lease").unwrap();
    let epoch = QueryServiceResourceEpoch::acquire(&service_epoch, Some(&anchor), "test.resident_lease").unwrap();
    assert!(epoch.has_query_anchor());
    let leased_grant = grant.with_resource_epoch(epoch);
    let owner = leased_grant.retain_resident_owner(100).unwrap().expect("leased owner");
    assert!(owner.has_resource_epoch(), "resident owners hold the backend generation lease");
    assert!(
        !owner.retains_query_anchor(),
        "resident owners must not retain the query anchor under test or the report handoff stalls"
    );
    let derived = owner.retain_derived_owner(50).unwrap().expect("derived owner");
    assert!(derived.has_resource_epoch(), "aliases share the one generation lease");
    assert!(!derived.retains_query_anchor());
}
