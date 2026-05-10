//! Pure model tests for the versioned per-device immutable-grant ledger.

use std::sync::{
    Arc, Barrier, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use super::*;
use crate::runtime::{backend_metrics::BackendMetricsStore, gpu::DeviceCapabilities};

fn requirement(floor: u64, hint: Option<u64>) -> AttemptRequirement {
    AttemptRequirement {
        configured_floor_bytes: floor,
        capability_floor_raise_bytes: 0,
        required_phase_floor_bytes: 0,
        source_stage_bytes: None,
        soft_working_set_hint_bytes: hint,
        estimate_quality: crate::admission::EstimateQuality::Modeled,
        required_capabilities: Vec::new(),
        owner_terms: Vec::new(),
        bound_derivation: None,
        requires_composed_producer_binding: false,
        attested_cugraph_graph_execution_scopes: Vec::new(),
        reservation_shapes: Vec::new(),
    }
}

fn profile(managed: u64, floor: u64, max_active: usize) -> DeviceResourceProfile {
    DeviceResourceProfile::resolve(0, managed, 0, 0, floor, max_active, 1, max_active * 2, max_active * 2)
        .expect("profile must resolve")
}

fn cache_profile(managed: u64, floor: u64, max_active: usize, cache_cap: u64) -> DeviceResourceProfile {
    DeviceResourceProfile::resolve_with_capabilities_and_cache_cap(
        0,
        managed,
        0,
        0,
        floor,
        max_active,
        1,
        max_active * 2,
        max_active * 2,
        DeviceCapabilities::default(),
        cache_cap,
    )
    .expect("profile must resolve")
}

fn request(id: u64) -> AdmissionRequestId {
    AdmissionRequestId::new(id)
}

#[test]
fn admission_equation_preserves_empty_slot_floors() {
    // managed 1000, floor 100, A_d = 3: first grant can reach
    // 1000 - 0 - 100*(3-1) = 800; second 1000 - 800 - 100 = 100;
    // third fits its exact floor; a fourth exceeds the count limit.
    let ledger = DeviceLedger::new(profile(1_000, 100, 3));
    let first = ledger.try_admit(request(1), &requirement(100, Some(900))).unwrap().expect("first admits");
    assert_eq!(first.budget_bytes(), 800);
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.live_grants_bytes, 800);
    assert_eq!(snapshot.active_attempts, 1);
    let second = ledger.try_admit(request(2), &requirement(100, Some(500))).unwrap().expect("second admits");
    assert_eq!(second.budget_bytes(), 100);
    let third = ledger.try_admit(request(3), &requirement(100, None)).unwrap().expect("third admits at floor");
    assert_eq!(third.budget_bytes(), 100);
    assert!(ledger.try_admit(request(4), &requirement(100, None)).unwrap().is_none());
}

#[test]
fn cache_cap_zero_rejects_and_nonzero_cap_is_aggregate() {
    let disabled = DeviceLedger::new(cache_profile(1_000, 100, 1, 0));
    assert!(!disabled.try_reserve_evictable_bytes(1).unwrap());
    let ledger = DeviceLedger::new(cache_profile(1_000, 100, 1, 300));
    assert!(ledger.try_reserve_evictable_bytes(300).unwrap());
    assert!(!ledger.try_reserve_evictable_bytes(1).unwrap());
    assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, 300);
    assert_eq!(ledger.snapshot().unwrap().cache_pinned_in_use_bytes, 0);
}

#[test]
fn two_modeled_attempts_leave_space_for_the_first_source_cache_promotion() {
    // The immutable grants remain the sole admission and cache-capacity
    // authority. A planner-issued source target merely avoids giving each
    // small source attempt the full device envelope.
    let ledger = DeviceLedger::new(cache_profile(1_000, 100, 2, 400));
    let first = ledger.try_admit(request(1), &requirement(100, Some(250))).unwrap().expect("first admits");
    let second = ledger.try_admit(request(2), &requirement(100, Some(250))).unwrap().expect("second admits");

    assert_eq!(first.budget_bytes(), 250);
    assert_eq!(second.budget_bytes(), 250);
    assert!(ledger.try_reserve_evictable_bytes(400).unwrap(), "cache charge fits beside immutable grants");
    let charged = ledger.snapshot().unwrap();
    assert_eq!(charged.live_grants_bytes, 500);
    assert_eq!(charged.cache_evictable_resident_bytes, 400);
    assert_eq!(charged.live_reserved_bytes, 0);
    assert!(
        !ledger.try_reserve_evictable_bytes(1).unwrap(),
        "the cache must not exceed the remaining capacity behind immutable grants"
    );
    ledger.release_cache_residency_precharge_excess(400).unwrap();
    drop(second);
    drop(first);
}

#[test]
fn pinned_cache_bytes_reduce_admission_headroom() {
    let ledger = DeviceLedger::new(cache_profile(300, 100, 1, 200));
    assert!(ledger.try_reserve_evictable_bytes(200).unwrap());
    ledger.pin_cache_residency_charge(200).unwrap();
    assert_eq!(ledger.candidate_max_new_grant_bytes(100, request(901)).unwrap(), Some(100));
    assert!(ledger.try_admit(request(901), &requirement(100, None)).unwrap().is_some());
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.cache_evictable_resident_bytes, 0);
    assert_eq!(snapshot.cache_pinned_in_use_bytes, 200);
}

#[test]
fn grant_clamps_soft_hint_into_floor_and_max() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 2));
    let below_floor = ledger.try_admit(request(1), &requirement(100, Some(10))).unwrap().expect("admits");
    assert_eq!(below_floor.budget_bytes(), 100, "hint below the floor never shrinks the grant");
    drop(below_floor);
    let above_max = ledger.try_admit(request(2), &requirement(100, Some(u64::MAX))).unwrap().expect("admits");
    assert_eq!(above_max.budget_bytes(), 900, "hint above max clamps to the equation");
}

#[test]
fn count_limit_is_enforced_independently_of_memory() {
    let ledger = DeviceLedger::new(profile(u64::MAX / 4, 1, 2));
    let _one = ledger.try_admit(request(1), &requirement(1, None)).unwrap().expect("admits");
    let _two = ledger.try_admit(request(2), &requirement(1, None)).unwrap().expect("admits");
    assert!(ledger.try_admit(request(3), &requirement(1, None)).unwrap().is_none());
}

#[test]
fn release_refunds_exactly_and_bumps_sequence() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 2));
    let before = ledger.snapshot().unwrap().sequence;
    let grant = ledger.try_admit(request(1), &requirement(100, Some(400))).unwrap().expect("admits");
    let mid = ledger.snapshot().unwrap();
    assert!(mid.sequence > before);
    assert_eq!(mid.live_grants_bytes, 400);
    drop(grant);
    let after = ledger.snapshot().unwrap();
    assert!(after.sequence > mid.sequence);
    assert_eq!(after.live_grants_bytes, 0);
    assert_eq!(after.active_attempts, 0);
    let next = ledger.try_admit(request(2), &requirement(100, Some(900))).unwrap().expect("readmits");
    assert_eq!(next.budget_bytes(), 900);
}

#[test]
fn transition_observer_covers_every_sequence_bump_with_its_atomic_snapshot() {
    let observations = Arc::new(Mutex::new(Vec::<LedgerTransition>::new()));
    let first = DeviceLedger::new(cache_profile(1_000, 100, 2, 700));
    let observations_for_first = Arc::clone(&observations);
    first
        .register_transition_observer(Arc::new(move |transition| {
            observations_for_first.lock().unwrap().push(transition.clone());
        }))
        .unwrap();

    let grant = first.try_admit(request(1), &requirement(100, Some(400))).unwrap().expect("admits");
    assert!(grant.try_reserve_live_bytes(40).unwrap());
    grant.release_live_bytes(40);
    assert!(first.try_reserve_evictable_bytes(10).unwrap());
    first.release_cache_residency_precharge_excess(10).unwrap();
    assert!(first.protect_slot(request(2), 100).unwrap());
    assert!(first.end_protection(request(2)).unwrap());
    drop(grant);
    assert_eq!(first.close_admission("test close").unwrap(), 1);

    let second = DeviceLedger::new(profile(1_000, 100, 1));
    let observations_for_second = Arc::clone(&observations);
    second
        .register_transition_observer(Arc::new(move |transition| {
            observations_for_second.lock().unwrap().push(transition.clone());
        }))
        .unwrap();
    assert!(second.close_admission_if_open("test first close").unwrap());

    let observations = observations.lock().unwrap();
    let expected = [
        (LedgerTransitionKind::GrantAdmitted, 1),
        (LedgerTransitionKind::LiveReservationCharged, 2),
        (LedgerTransitionKind::LiveReservationReleased, 3),
        (LedgerTransitionKind::EvictableCharged, 4),
        (LedgerTransitionKind::EvictableReleased, 5),
        (LedgerTransitionKind::ProtectionInstalled, 6),
        (LedgerTransitionKind::ProtectionRemoved, 7),
        (LedgerTransitionKind::GrantReleased, 8),
        (LedgerTransitionKind::AdmissionClosed, 9),
        (LedgerTransitionKind::AdmissionClosed, 1),
    ];
    assert_eq!(observations.len(), expected.len());
    for (transition, (kind, sequence)) in observations.iter().zip(expected) {
        assert_eq!(transition.kind, kind);
        assert_eq!(transition.snapshot.sequence, sequence);
    }
}

#[test]
fn transition_observer_snapshots_match_sequences_during_concurrent_admit_and_release() {
    let ledger = Arc::new(DeviceLedger::new(cache_profile(1_000, 100, 2, 0)));
    let observations = Arc::new(Mutex::new(Vec::<LedgerTransition>::new()));
    let observations_for_observer = Arc::clone(&observations);
    ledger
        .register_transition_observer(Arc::new(move |transition| {
            observations_for_observer.lock().unwrap().push(transition.clone());
        }))
        .unwrap();

    let start = Arc::new(Barrier::new(3));
    let workers = (0..2)
        .map(|worker| {
            let ledger = Arc::clone(&ledger);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                for round in 0..8 {
                    let request_id = u64::try_from(worker * 100 + round + 1).unwrap();
                    loop {
                        if let Some(grant) =
                            ledger.try_admit(request(request_id), &requirement(100, Some(100))).unwrap()
                        {
                            drop(grant);
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    start.wait();
    for worker in workers {
        worker.join().expect("ledger worker must finish");
    }

    let observations = observations.lock().unwrap();
    assert_eq!(observations.len(), 32);
    for (index, transition) in observations.iter().enumerate() {
        assert_eq!(transition.snapshot.sequence, u64::try_from(index + 1).unwrap());
        assert!(matches!(transition.kind, LedgerTransitionKind::GrantAdmitted | LedgerTransitionKind::GrantReleased));
    }
}

#[test]
fn backend_metrics_observer_remains_exact_after_concurrent_admit_release_and_reservation_updates() {
    let ledger = Arc::new(DeviceLedger::new(profile(1_000, 100, 2)));
    let metrics = BackendMetricsStore::new(91, &[0]).expect("fixed metrics store accepts the fixture device");
    let last_max_live_reserved = Arc::new(AtomicU64::new(0));
    let metrics_for_observer = metrics.clone();
    let last_max_for_observer = Arc::clone(&last_max_live_reserved);
    ledger
        .register_transition_observer(Arc::new(move |transition| {
            metrics_for_observer.record_ledger_transition(transition);
            last_max_for_observer.fetch_max(transition.snapshot.live_reserved_bytes, Ordering::AcqRel);
        }))
        .unwrap();

    let start = Arc::new(Barrier::new(3));
    let workers = (0..2)
        .map(|worker| {
            let ledger = Arc::clone(&ledger);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                for round in 0..8 {
                    let request_id = u64::try_from(1_000 + worker * 100 + round).unwrap();
                    loop {
                        if let Some(grant) =
                            ledger.try_admit(request(request_id), &requirement(100, Some(100))).unwrap()
                        {
                            assert!(grant.try_reserve_live_bytes(20).unwrap());
                            grant.release_live_bytes(20);
                            drop(grant);
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    start.wait();
    for worker in workers {
        worker.join().expect("ledger worker must finish");
    }

    let ledger_snapshot = ledger.snapshot().unwrap();
    let metrics_snapshot = metrics.snapshot();
    let device = metrics_snapshot.device(0).expect("configured metrics device exists");
    assert_eq!(device.ledger_sequence, ledger_snapshot.sequence);
    assert_eq!(device.admission.active_attempts, u64::try_from(ledger_snapshot.active_attempts).unwrap());
    assert_eq!(device.admission.live_grant_bytes, ledger_snapshot.live_grants_bytes);
    assert_eq!(device.admission.live_reserved_bytes, ledger_snapshot.live_reserved_bytes);
    assert_eq!(device.admission.cache_evictable_resident_bytes, 0);
    assert_eq!(device.admission.cache_pinned_in_use_bytes, 0);
    assert_eq!(device.admission.max_live_reserved_bytes, last_max_live_reserved.load(Ordering::Acquire));
}

#[test]
fn transition_observer_panics_and_same_ledger_reentry_are_isolated() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 2));
    let grant = ledger.try_admit(request(1), &requirement(100, Some(400))).unwrap().expect("admits");
    let retained_grant = Arc::new(Mutex::new(Some(grant.clone())));
    let reentry_failures = Arc::new(AtomicUsize::new(0));
    let reentry_done = Arc::new(AtomicBool::new(false));
    let weak = Arc::downgrade(&ledger.inner);
    let retained_for_observer = Arc::clone(&retained_grant);
    let reentry_failures_for_observer = Arc::clone(&reentry_failures);
    let reentry_done_for_observer = Arc::clone(&reentry_done);
    ledger
        .register_transition_observer(Arc::new(move |_| {
            if reentry_done_for_observer.swap(true, Ordering::Relaxed) {
                return;
            }
            let same_ledger = DeviceLedger { inner: weak.upgrade().expect("ledger remains alive") };
            let snapshot_error = same_ledger.snapshot().expect_err("Result-returning reentry must fail structurally");
            assert_eq!(snapshot_error.fact_value("reason"), Some("ledger_observer_reentrancy"));
            reentry_failures_for_observer.fetch_add(1, Ordering::Relaxed);
            let retained_grant =
                retained_for_observer.lock().unwrap().as_ref().expect("retained grant is present").clone();
            for panic in [
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = same_ledger.protection();
                })),
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| same_ledger.release_grant(u64::MAX))),
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    retained_grant.release_live_bytes(1);
                })),
            ] {
                let panic = panic.expect_err("same-ledger transition-observer reentry must panic");
                let error = panic.downcast_ref::<QueryError>().expect("reentry panic carries QueryError");
                assert_eq!(error.fact_value("reason"), Some("ledger_observer_reentrancy"));
                reentry_failures_for_observer.fetch_add(1, Ordering::Relaxed);
            }
        }))
        .unwrap();
    let survived_panic = Arc::new(AtomicBool::new(false));
    let survived_panic_for_observer = Arc::clone(&survived_panic);
    ledger
        .register_transition_observer(Arc::new(move |_| {
            survived_panic_for_observer.store(true, Ordering::Relaxed);
            panic!("test observer panic");
        }))
        .unwrap();
    let after_panic = Arc::new(AtomicBool::new(false));
    let after_panic_for_observer = Arc::clone(&after_panic);
    ledger
        .register_transition_observer(Arc::new(move |_| {
            after_panic_for_observer.store(true, Ordering::Relaxed);
        }))
        .unwrap();

    assert!(ledger.protect_slot(request(2), 100).unwrap());
    assert_eq!(reentry_failures.load(Ordering::Relaxed), 4);
    assert!(survived_panic.load(Ordering::Relaxed));
    assert!(after_panic.load(Ordering::Relaxed));
    assert!(ledger.snapshot().is_ok(), "isolated observer panics must not poison the ledger mutex");

    drop(retained_grant.lock().unwrap().take());
    drop(grant);
}

#[test]
fn release_observer_runs_after_unlock_with_a_post_release_snapshot() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let weak = Arc::downgrade(&ledger.inner);
    let observed = Arc::new(Mutex::new(None));
    let observed_by_callback = Arc::clone(&observed);
    ledger
        .register_release_observer(Arc::new(move |snapshot| {
            let same_ledger = DeviceLedger { inner: weak.upgrade().expect("ledger remains alive") };
            assert_eq!(snapshot.kind, LedgerTransitionKind::GrantReleased);
            assert_eq!(
                same_ledger.snapshot().unwrap(),
                snapshot.snapshot.clone(),
                "release observer runs after the ledger mutex is unlocked"
            );
            *observed_by_callback.lock().unwrap() = Some(snapshot.snapshot.clone());
        }))
        .unwrap();
    let grant = ledger.try_admit(request(1), &requirement(100, Some(400))).unwrap().expect("admits");
    drop(grant);

    let observed = observed.lock().unwrap().clone().expect("release observer runs");
    assert_eq!(observed.active_attempts, 0);
    assert_eq!(observed.live_grants_bytes, 0);
}

#[test]
fn protection_withholds_one_slot_and_floor_reserve() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 2));
    let waiter = request(9);
    assert!(ledger.protect_slot(waiter, 100).unwrap());
    // A competing candidate sees one protected slot: max_new_grant =
    // 1000 - 0 - max(100, 0) - 100*(2-1-1) = 900.
    let competing = ledger.try_admit(request(1), &requirement(100, Some(950))).unwrap().expect("admits");
    assert_eq!(competing.budget_bytes(), 900);
    // A second candidate is now count-ineligible: the protected slot is not
    // also counted as an ordinary empty slot.
    assert!(ledger.try_admit(request(2), &requirement(100, None)).unwrap().is_none());
    // No competing protection can be installed.
    assert!(!ledger.protect_slot(request(10), 100).unwrap());
    // Releasing the competing grant feeds the protected liveness up to the
    // configured floor deficit.
    drop(competing);
    let protection = ledger.protection().expect("protection persists");
    assert_eq!(protection.liveness_bytes, 100);
    // The protected waiter is admitted by consuming the protection; its grant
    // replaces the reserve rather than being subtracted twice.
    let protected = ledger.try_admit(waiter, &requirement(100, Some(800))).unwrap().expect("protected admits");
    assert_eq!(protected.budget_bytes(), 800);
    assert!(ledger.protection().is_none());
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.live_grants_bytes, 800);
    assert!(!snapshot.protected_slot);
    assert_eq!(snapshot.protected_reserve_bytes, 0);
}

#[test]
fn refreshing_the_same_protection_is_a_noop_without_a_transition() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 2));
    let waiter = request(9);
    assert!(ledger.protect_slot(waiter, 100).unwrap());
    let before = ledger.snapshot().unwrap();
    let protection = ledger.protection().unwrap();

    assert!(ledger.protect_slot(waiter, 200).unwrap());
    assert_eq!(ledger.snapshot().unwrap(), before);
    assert_eq!(ledger.protection(), Some(protection));
}

#[test]
fn protected_liveness_grows_only_up_to_the_floor_deficit() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 3));
    let waiter = request(9);
    assert!(ledger.protect_slot(waiter, 100).unwrap());
    let first = ledger.try_admit(request(1), &requirement(100, Some(200))).unwrap().expect("admits");
    let second = ledger.try_admit(request(2), &requirement(100, Some(200))).unwrap().expect("admits");
    assert_eq!(second.budget_bytes(), 200, "protection reserves one slot and the floor reserve, not more");
    drop(first);
    assert_eq!(ledger.protection().unwrap().liveness_bytes, 100);
    drop(second);
    assert_eq!(
        ledger.protection().unwrap().liveness_bytes,
        100,
        "liveness stops at the configured-floor deficit and never exceeds it"
    );
}

#[test]
fn end_protection_releases_the_withheld_slot() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let waiter = request(9);
    assert!(ledger.protect_slot(waiter, 100).unwrap());
    assert!(ledger.end_protection(waiter).unwrap());
    assert!(!ledger.end_protection(waiter).unwrap());
    let grant = ledger.try_admit(request(1), &requirement(100, Some(1_000))).unwrap().expect("admits");
    assert_eq!(grant.budget_bytes(), 1_000);
}

#[test]
fn closed_health_blocks_new_admission_but_releases_still_refund() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 2));
    let grant = ledger.try_admit(request(1), &requirement(100, None)).unwrap().expect("admits");
    assert_eq!(ledger.close_admission("test_quarantine").unwrap(), 1);
    assert!(ledger.try_admit(request(2), &requirement(100, None)).unwrap().is_none());
    drop(grant);
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.live_grants_bytes, 0);
    assert!(matches!(snapshot.health, DeviceAdmissionHealth::Closed { .. }));
    assert_eq!(ledger.close_admission("again").unwrap(), 2);
}

#[test]
fn heterogeneous_ledgers_are_independent_envelopes() {
    let small = DeviceLedger::new(DeviceResourceProfile::resolve(0, 500, 0, 0, 100, 1, 1, 2, 2).unwrap());
    let large = DeviceLedger::new(DeviceResourceProfile::resolve(1, 5_000, 0, 0, 100, 1, 1, 2, 2).unwrap());
    let grant = large.try_admit(request(1), &requirement(100, Some(4_000))).unwrap().expect("large admits");
    assert_eq!(grant.budget_bytes(), 4_000);
    let small_grant = small.try_admit(request(3), &requirement(100, Some(400))).unwrap().expect("small admits");
    assert_eq!(small_grant.budget_bytes(), 400);
    let clamped = small.try_admit(request(2), &requirement(100, Some(4_000))).unwrap();
    assert!(clamped.is_none(), "a second attempt exceeds the one-slot envelope");
    assert_eq!(small.snapshot().unwrap().live_grants_bytes, 400);
    assert_eq!(large.snapshot().unwrap().live_grants_bytes, 4_000);
}

#[test]
fn zero_request_floor_resolves_against_each_candidate_profile() {
    let small = DeviceLedger::new(DeviceResourceProfile::resolve(0, 64, 0, 0, 8, 2, 1, 4, 4).unwrap());
    let large = DeviceLedger::new(DeviceResourceProfile::resolve(1, 128, 0, 0, 16, 1, 1, 2, 2).unwrap());
    let zero_floor = AttemptRequirement {
        configured_floor_bytes: 0,
        capability_floor_raise_bytes: 0,
        required_phase_floor_bytes: 0,
        source_stage_bytes: None,
        soft_working_set_hint_bytes: Some(1),
        estimate_quality: crate::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
        owner_terms: Vec::new(),
        bound_derivation: None,
        requires_composed_producer_binding: false,
        attested_cugraph_graph_execution_scopes: Vec::new(),
        reservation_shapes: Vec::new(),
    };

    assert_eq!(small.required_floor_bytes_for(&zero_floor).unwrap(), 8);
    assert_eq!(large.required_floor_bytes_for(&zero_floor).unwrap(), 16);
    assert_eq!(small.try_admit(request(1), &zero_floor).unwrap().unwrap().budget_bytes(), 8);
    assert_eq!(small.try_admit(request(2), &zero_floor).unwrap().unwrap().budget_bytes(), 8);
    assert_eq!(large.try_admit(request(3), &zero_floor).unwrap().unwrap().budget_bytes(), 16);
}

#[test]
fn request_floor_cannot_lower_the_candidate_profile_floor() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let requirement = AttemptRequirement {
        configured_floor_bytes: 1,
        capability_floor_raise_bytes: 7,
        required_phase_floor_bytes: 0,
        source_stage_bytes: None,
        soft_working_set_hint_bytes: Some(1),
        estimate_quality: crate::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
        owner_terms: Vec::new(),
        bound_derivation: None,
        requires_composed_producer_binding: false,
        attested_cugraph_graph_execution_scopes: Vec::new(),
        reservation_shapes: Vec::new(),
    };

    assert_eq!(ledger.required_floor_bytes_for(&requirement).unwrap(), 107);
    assert_eq!(ledger.try_admit(request(1), &requirement).unwrap().unwrap().budget_bytes(), 107);
}

#[test]
fn requirement_floor_above_the_candidate_profile_floor_is_preserved() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let requirement = AttemptRequirement {
        configured_floor_bytes: 120,
        capability_floor_raise_bytes: 7,
        required_phase_floor_bytes: 0,
        source_stage_bytes: None,
        soft_working_set_hint_bytes: Some(1),
        estimate_quality: crate::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
        owner_terms: Vec::new(),
        bound_derivation: None,
        requires_composed_producer_binding: false,
        attested_cugraph_graph_execution_scopes: Vec::new(),
        reservation_shapes: Vec::new(),
    };

    assert_eq!(ledger.required_floor_bytes_for(&requirement).unwrap(), 127);
    assert_eq!(ledger.try_admit(request(1), &requirement).unwrap().unwrap().budget_bytes(), 127);
}

#[test]
fn phase_floor_is_a_hard_candidate_minimum_without_adding_the_profile_floor() {
    let ledger = DeviceLedger::new(profile(191, 64, 2));
    let mut requirement = requirement(0, None);
    requirement.required_phase_floor_bytes = 128;

    assert_eq!(ledger.required_floor_bytes_for(&requirement).unwrap(), 128);
    assert!(
        ledger.try_admit(request(1), &requirement).unwrap().is_none(),
        "the candidate ceiling is 127 after reserving the other profile floor, below the resolved phase minimum"
    );
}

#[test]
fn overflowing_floor_is_rejected_structurally() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let overflowing = AttemptRequirement {
        configured_floor_bytes: 0,
        capability_floor_raise_bytes: u64::MAX,
        required_phase_floor_bytes: 0,
        source_stage_bytes: None,
        soft_working_set_hint_bytes: None,
        estimate_quality: crate::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
        owner_terms: Vec::new(),
        bound_derivation: None,
        requires_composed_producer_binding: false,
        attested_cugraph_graph_execution_scopes: Vec::new(),
        reservation_shapes: Vec::new(),
    };
    assert!(ledger.try_admit(request(1), &overflowing).is_err());
}

#[test]
fn live_reservation_accounting_stays_exact() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let grant = ledger.try_admit(request(1), &requirement(100, Some(500))).unwrap().expect("admits");
    assert!(grant.try_reserve_live_bytes(200).unwrap());
    assert!(!grant.try_reserve_live_bytes(400).unwrap(), "live bytes cannot exceed the immutable grant");
    assert_eq!(grant.live_reserved_bytes(), 200);
    assert_eq!(grant.remaining_live_reservation_bytes().unwrap(), 300);
    let retained = grant
        .retain_live_reservation(100)
        .expect("retention fits the immutable grant")
        .expect("positive reservation is retained");
    assert!(retained.belongs_to(&grant));
    assert_eq!(retained.bytes(), 100);
    assert_eq!(grant.live_reserved_bytes(), 300);
    let derived = retained.retain_derived_live_reservation(50).unwrap().expect("derived");
    assert_eq!(grant.live_reserved_bytes(), 350);
    assert!(
        retained.retain_derived_live_reservation(1_000).is_err(),
        "derived retention cannot exceed the immutable grant"
    );
    drop(derived);
    assert_eq!(grant.live_reserved_bytes(), 300);
    grant.release_live_bytes(200);
    assert_eq!(grant.live_reserved_bytes(), 100);
    drop(retained);
    assert_eq!(grant.live_reserved_bytes(), 0);
}

#[test]
fn snapshot_aggregates_live_reservations_and_versions_every_observable_accounting_change() {
    let profile = DeviceResourceProfile::resolve_with_capabilities_and_cache_cap(
        0,
        1_000,
        0,
        0,
        100,
        3,
        2,
        9,
        9,
        DeviceCapabilities::default(),
        700,
    )
    .expect("cache test profile must resolve");
    let ledger = DeviceLedger::new(profile);
    let empty = ledger.snapshot().unwrap();
    assert_eq!(empty.live_reserved_bytes, 0);
    assert_eq!(empty.root_streams_available, 3);
    assert_eq!(empty.auxiliary_streams_available, 6);
    assert_eq!(empty.root_worker_slots_available, 3);
    assert_eq!(empty.auxiliary_worker_slots_available, 6);
    assert_eq!(empty.worker_depth, 0);

    let grant = ledger.try_admit(request(1), &requirement(100, Some(100))).unwrap().expect("admits");
    let admitted = ledger.snapshot().unwrap();
    assert!(admitted.sequence > empty.sequence);
    assert_eq!(admitted.root_streams_available, 2);
    assert_eq!(admitted.auxiliary_streams_available, 4);
    assert_eq!(admitted.root_worker_slots_available, 2);
    assert_eq!(admitted.auxiliary_worker_slots_available, 4);
    assert_eq!(admitted.worker_depth, 3);

    assert!(grant.try_reserve_live_bytes(40).unwrap());
    let reserved = ledger.snapshot().unwrap();
    assert!(reserved.sequence > admitted.sequence);
    assert_eq!(reserved.live_reserved_bytes, 40);
    let report = ledger.report().unwrap();
    assert_eq!(report.sequence, reserved.sequence);
    assert_eq!(report.live_reserved_bytes, 40);
    assert_eq!(report.grants.iter().map(|row| row.live_reserved_bytes).sum::<u64>(), 40);

    grant.release_live_bytes(40);
    let released = ledger.snapshot().unwrap();
    assert!(released.sequence > reserved.sequence);
    assert_eq!(released.live_reserved_bytes, 0);

    assert!(ledger.try_reserve_evictable_bytes(10).unwrap());
    let cache_charged = ledger.snapshot().unwrap();
    assert!(cache_charged.sequence > released.sequence);
    ledger.release_cache_residency_precharge_excess(10).unwrap();
    let cache_released = ledger.snapshot().unwrap();
    assert!(cache_released.sequence > cache_charged.sequence);
}

#[test]
fn direct_retention_rejects_new_bytes_above_the_immutable_grant() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let grant = ledger.try_admit(request(1), &requirement(100, Some(100))).unwrap().expect("admits");
    assert!(grant.try_reserve_live_bytes(100).unwrap());

    let error = match grant.retain_live_reservation(1) {
        Err(error) => error,
        Ok(_) => panic!("new retained bytes must exceed the immutable grant"),
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("reason"), Some("retained_reservation_exceeds_immutable_grant"));
    assert_eq!(grant.live_reserved_bytes(), 100);
}

#[test]
fn evictable_cache_bytes_obey_a_cumulative_bound_across_reserve_and_exact_precharge_return() {
    let ledger = DeviceLedger::new(cache_profile(1_000, 100, 1, 900));
    let grant = ledger.try_admit(request(1), &requirement(100, Some(400))).unwrap().expect("admits");
    assert!(ledger.try_reserve_evictable_bytes(300).unwrap());
    ledger.release_cache_residency_precharge_excess(100).unwrap();
    assert!(ledger.try_reserve_evictable_bytes(400).unwrap());
    assert_eq!(ledger.evictable_bytes().unwrap(), 600);
    assert!(
        !ledger.try_reserve_evictable_bytes(1).unwrap(),
        "live grants and all cache reservations share one device-capacity bound"
    );
    ledger.release_cache_residency_precharge_excess(250).unwrap();
    assert!(ledger.try_reserve_evictable_bytes(250).unwrap());
    assert_eq!(ledger.evictable_bytes().unwrap(), 600);
    drop(grant);
}
