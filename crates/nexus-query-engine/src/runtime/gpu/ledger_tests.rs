//! Pure model tests for the versioned per-device immutable-grant ledger.

use super::*;

fn requirement(floor: u64, hint: Option<u64>) -> AttemptRequirement {
    AttemptRequirement {
        configured_floor_bytes: floor,
        capability_floor_raise_bytes: 0,
        soft_working_set_hint_bytes: hint,
        estimate_quality: crate::admission::EstimateQuality::Modeled,
        required_capabilities: Vec::new(),
    }
}

fn profile(managed: u64, floor: u64, max_active: usize) -> DeviceResourceProfile {
    DeviceResourceProfile::resolve(0, managed, 0, 0, floor, max_active, 1, max_active * 2, max_active * 2)
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
        soft_working_set_hint_bytes: Some(1),
        estimate_quality: crate::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
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
        soft_working_set_hint_bytes: Some(1),
        estimate_quality: crate::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
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
        soft_working_set_hint_bytes: Some(1),
        estimate_quality: crate::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
    };

    assert_eq!(ledger.required_floor_bytes_for(&requirement).unwrap(), 127);
    assert_eq!(ledger.try_admit(request(1), &requirement).unwrap().unwrap().budget_bytes(), 127);
}

#[test]
fn overflowing_floor_is_rejected_structurally() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let overflowing = AttemptRequirement {
        configured_floor_bytes: 0,
        capability_floor_raise_bytes: u64::MAX,
        soft_working_set_hint_bytes: None,
        estimate_quality: crate::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
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
    let profile = DeviceResourceProfile::resolve(0, 1_000, 0, 0, 100, 3, 2, 9, 9).unwrap();
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
    assert_eq!(ledger.release_evictable_bytes(10).unwrap(), 10);
    let cache_released = ledger.snapshot().unwrap();
    assert!(cache_released.sequence > cache_charged.sequence);

    assert!(ledger.try_reserve_evictable_bytes_from_grant(&grant, 10).unwrap());
    let grant_cache_charged = ledger.snapshot().unwrap();
    assert!(grant_cache_charged.sequence > cache_released.sequence);
    assert_eq!(ledger.release_evictable_bytes(10).unwrap(), 10);
    assert!(ledger.snapshot().unwrap().sequence > grant_cache_charged.sequence);
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
fn evictable_cache_accounting_is_separate_from_the_admission_equation() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    assert!(ledger.try_reserve_evictable_bytes(300).unwrap());
    assert_eq!(ledger.evictable_bytes().unwrap(), 300);
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.cache_evictable_resident_bytes, 0, "P2 admission snapshots expose zero cache residency");
    assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
    assert_eq!(ledger.release_evictable_bytes(400).unwrap(), 300);
    assert_eq!(ledger.evictable_bytes().unwrap(), 0);
}

#[test]
fn grant_can_cover_and_reserve_evictable_from_grant() {
    let ledger = DeviceLedger::new(profile(1_000, 100, 1));
    let grant = ledger.try_admit(request(1), &requirement(100, Some(400))).unwrap().expect("admits");
    assert!(ledger.grant_can_cover_evictable_bytes(&grant, 400).unwrap());
    assert!(!ledger.grant_can_cover_evictable_bytes(&grant, 401).unwrap());
    assert!(ledger.try_reserve_evictable_bytes_from_grant(&grant, 200).unwrap());
    // Over-accounting while the grant is live is explicitly tolerated: each
    // call is bounded by the grant, not by the cumulative evictable charge.
    assert!(ledger.try_reserve_evictable_bytes_from_grant(&grant, 300).unwrap());
    assert!(!ledger.try_reserve_evictable_bytes_from_grant(&grant, 500).unwrap());
    assert_eq!(ledger.evictable_bytes().unwrap(), 500);
}
