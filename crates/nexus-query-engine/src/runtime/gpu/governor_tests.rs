use std::{collections::HashSet, sync::mpsc, thread, time::Duration};

use super::*;

#[test]
fn grants_never_exceed_capacity_and_release_on_drop() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let first = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 70)).unwrap().unwrap();
    let second = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 20, 70)).unwrap().unwrap();

    assert_eq!(first.budget_bytes(), 70);
    assert_eq!(second.budget_bytes(), 30);
    assert_eq!(governor.available_bytes().unwrap(), 0);
    assert!(governor.try_request(GrantRequest::new(AdmissionRequestId::new(3), 1, 1)).unwrap().is_none());

    drop(first);
    assert_eq!(governor.available_bytes().unwrap(), 70);
    drop(second);
    assert_eq!(governor.available_bytes().unwrap(), 100);
}

#[test]
fn small_query_receives_want_not_capacity_partition() {
    let governor = MemoryGovernor::new(0, 96).unwrap();
    let small = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 8, 12)).unwrap().unwrap();

    assert_eq!(small.budget_bytes(), 12);
    assert_eq!(governor.available_bytes().unwrap(), 84);
}

#[test]
fn try_request_returns_none_under_pressure() {
    let governor = MemoryGovernor::new(0, 64).unwrap();
    let _active = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 64, 64)).unwrap().unwrap();

    assert!(governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 1, 1)).unwrap().is_none());
}

#[test]
fn try_grow_charges_available_capacity_only() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();
    let other = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 50, 50)).unwrap().unwrap();

    assert!(!grant.try_grow(20).unwrap());
    drop(other);
    assert!(grant.try_grow(20).unwrap());
    assert_eq!(grant.budget_bytes(), 60);
    assert_eq!(governor.available_bytes().unwrap(), 40);
}

#[test]
fn hard_cap_rejects_current_budget_undercut() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 60)).unwrap().unwrap();

    let error = grant.install_hard_cap_bytes_for_current_budget(|_budget| Ok(50)).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("hard_cap_below_current_budget"));
    assert_eq!(grant.hard_cap_bytes(), None);
}

#[test]
fn try_grow_clamps_to_hard_cap() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();

    assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|_budget| Ok(55)).unwrap(), (40, 55));
    assert!(grant.try_grow(40).unwrap());
    assert_eq!(grant.budget_bytes(), 55);
    assert_eq!(governor.available_bytes().unwrap(), 45);
    assert!(!grant.try_grow(1).unwrap());
    assert_eq!(grant.budget_bytes(), 55);
}

#[test]
fn try_grow_to_returns_false_above_hard_cap() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();

    assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|_budget| Ok(60)).unwrap(), (40, 60));
    assert!(!grant.try_grow_to(90).unwrap());
    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(governor.available_bytes().unwrap(), 60);
    assert!(grant.try_grow_to(60).unwrap());
    assert_eq!(grant.budget_bytes(), 60);
    assert_eq!(governor.available_bytes().unwrap(), 40);
}

#[test]
fn hard_cap_install_returns_existing_effective_cap() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();

    assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|budget| Ok(budget * 2)).unwrap(), (40, 80));
    assert!(grant.try_grow_to(60).unwrap());
    assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|budget| Ok(budget * 3)).unwrap(), (60, 80));
    assert_eq!(grant.hard_cap_bytes(), Some(80));
}

#[test]
fn try_grow_to_rereads_budget_after_evictable_pressure_observer() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();
    let other = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 40, 40)).unwrap().unwrap();
    assert!(governor.try_reserve_evictable_bytes(20).unwrap());

    let other_slot = Arc::new(Mutex::new(Some(other)));
    let observer_grant = grant.clone();
    let observer_governor = governor.clone();
    let observer_other_slot = Arc::clone(&other_slot);
    let (sender, receiver) = mpsc::channel();
    governor
        .register_evictable_memory_observer(Arc::new(move |pressure| {
            sender.send(pressure).unwrap();
            drop(observer_other_slot.lock().unwrap().take());
            observer_governor.release_evictable_bytes(pressure.requested_evict_bytes)?;
            assert!(observer_grant.try_grow(30)?);
            Ok(pressure.requested_evict_bytes)
        }))
        .unwrap();

    assert!(grant.try_grow_to(60).unwrap());

    let pressure = receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(pressure.requested_evict_bytes, 20);
    assert_eq!(grant.budget_bytes(), 70);
    assert_eq!(governor.report().unwrap().outstanding_bytes, 70);
}

#[test]
fn retain_derived_live_reservation_fails_when_target_exceeds_hard_cap() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 50)).unwrap().unwrap();
    assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|_budget| Ok(60)).unwrap(), (50, 60));
    grant.record_live_reservation_delta(50);
    let retained = grant.retain_live_reservation(50).unwrap();
    grant.record_live_reservation_delta(-50);

    let error = match retained.retain_derived_live_reservation(20) {
        Ok(_) => panic!("derived retention above hard cap must fail"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("target_budget_bytes"), Some("70"));
    assert_eq!(error.fact_value("budget_bytes"), Some("50"));
    assert_eq!(grant.budget_bytes(), 50);
    assert_eq!(grant.live_reserved_bytes(), 50);
}

#[test]
fn try_shrink_to_refunds_capacity_without_dropping_grant() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert!(grant.try_shrink_to(50).unwrap());
    assert_eq!(grant.budget_bytes(), 50);
    assert_eq!(governor.available_bytes().unwrap(), 50);

    let other = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 40, 40)).unwrap().unwrap();
    assert_eq!(other.budget_bytes(), 40);
    assert_eq!(governor.available_bytes().unwrap(), 10);
}

#[test]
fn try_shrink_to_preserves_admission_floor() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert!(grant.try_shrink_to(10).unwrap());
    assert_eq!(grant.min_bytes(), 30);
    assert_eq!(grant.budget_bytes(), 30);
    assert_eq!(governor.available_bytes().unwrap(), 70);

    assert!(grant.try_shrink_to(0).unwrap());
    assert_eq!(grant.budget_bytes(), 30);
    assert_eq!(governor.available_bytes().unwrap(), 70);
}

#[test]
fn try_shrink_to_does_not_grow_grant() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 40)).unwrap().unwrap();

    assert!(grant.try_shrink_to(90).unwrap());
    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(governor.available_bytes().unwrap(), 60);
}

#[test]
fn shrink_request_does_not_refund_until_acknowledged() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(40));
    assert_eq!(grant.budget_bytes(), 80);
    assert_eq!(governor.available_bytes().unwrap(), 20);

    let blocked = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 30, 30)).unwrap();
    assert!(blocked.is_none());
}

#[test]
fn shrink_request_clamps_to_floor_and_keeps_lowest_target() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert_eq!(grant.request_shrink_to(60).unwrap(), Some(60));
    assert_eq!(grant.request_shrink_to(10).unwrap(), Some(30));
    assert_eq!(grant.request_shrink_to(50).unwrap(), Some(30));
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(30));
}

#[test]
fn apply_requested_shrink_refunds_only_to_live_reserved_bytes() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
    assert!(!grant.try_apply_requested_shrink(55).unwrap());
    assert_eq!(grant.budget_bytes(), 55);
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(40));
    assert_eq!(governor.available_bytes().unwrap(), 45);

    assert!(grant.try_apply_requested_shrink(35).unwrap());
    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(governor.available_bytes().unwrap(), 60);
}

#[test]
fn retained_live_reservation_blocks_shrink_until_final_handle_drop() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    grant.record_live_reservation_delta(50);
    let retained = grant.retain_live_reservation(50).unwrap();
    let retained_clone = retained.clone();
    grant.record_live_reservation_delta(-50);

    assert_eq!(grant.live_reserved_bytes(), 50);
    assert_eq!(grant.request_shrink_to(30).unwrap(), Some(30));
    assert!(!grant.try_apply_requested_shrink(grant.live_reserved_bytes()).unwrap());
    assert_eq!(grant.budget_bytes(), 50);

    drop(retained);
    assert_eq!(grant.live_reserved_bytes(), 50);
    drop(retained_clone);
    assert_eq!(grant.live_reserved_bytes(), 0);
    assert!(grant.try_apply_requested_shrink(grant.live_reserved_bytes()).unwrap());
    assert_eq!(grant.budget_bytes(), 30);
}

#[test]
fn direct_shrink_clears_satisfied_request() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
    assert!(grant.try_shrink_to(40).unwrap());

    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
}

#[test]
fn direct_shrink_above_requested_target_keeps_request_pending() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
    assert!(grant.try_shrink_to(60).unwrap());

    assert_eq!(grant.budget_bytes(), 60);
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(40));
}

#[test]
fn clear_requested_shrink_rescinds_pending_request_without_refund() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
    assert!(grant.clear_requested_shrink().unwrap());

    assert_eq!(grant.budget_bytes(), 80);
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(governor.available_bytes().unwrap(), 20);
    assert!(!grant.clear_requested_shrink().unwrap());
}

#[test]
fn budget_decrease_observer_runs_on_shrink_not_grow() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 40)).unwrap().unwrap();
    let observer_governor = governor.clone();
    let (sender, receiver) = mpsc::channel();
    grant
        .notify_on_budget_decrease(Arc::new(move || {
            sender.send(observer_governor.available_bytes().unwrap()).unwrap();
        }))
        .unwrap();

    assert!(grant.try_grow_to(70).unwrap());
    assert!(receiver.try_recv().is_err());

    assert!(grant.try_shrink_to(50).unwrap());
    assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 50);
}

#[test]
fn budget_change_observer_runs_on_grow_and_shrink() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 40)).unwrap().unwrap();
    let (sender, receiver) = mpsc::channel();
    let initial =
        grant.notify_on_budget_change(Arc::new(move |change| sender.send(change.budget_bytes).unwrap())).unwrap();
    assert_eq!(initial.budget_bytes, 40);
    assert_eq!(initial.sequence, 0);

    assert!(grant.try_grow_to(70).unwrap());
    assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 70);

    assert!(grant.try_shrink_to(50).unwrap());
    assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 50);

    assert!(grant.try_grow(0).unwrap());
    assert!(grant.try_shrink_to(90).unwrap());
    assert!(receiver.try_recv().is_err());
}

#[test]
fn current_budget_change_snapshots_match_published_pairs_under_concurrency() -> Result<()> {
    fn locked_pair(grant: &MemoryGrant) -> (u64, u64) {
        let state = grant.inner.governor.state.lock().unwrap();
        let budget = state.grants.get(&grant.inner.grant_id).unwrap().budget_bytes;
        let sequence = grant.inner.budget_change_sequence.load(Ordering::Acquire);
        (budget, sequence)
    }

    let governor = MemoryGovernor::new(0, 200)?;
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 50, 100))?.unwrap();
    let published = Arc::new(Mutex::new(vec![locked_pair(&grant)]));
    let writer_grant = grant.clone();
    let writer_published = Arc::clone(&published);
    let writer = thread::spawn(move || -> Result<()> {
        for iteration in 0..10_000 {
            if iteration % 2 == 0 {
                assert!(writer_grant.try_grow_to(150)?);
            } else {
                assert!(writer_grant.try_shrink_to(100)?);
            }
            writer_published.lock().unwrap().push(locked_pair(&writer_grant));
            thread::yield_now();
        }
        Ok(())
    });

    let mut observed = Vec::new();
    while !writer.is_finished() {
        let change = grant.current_budget_change()?;
        observed.push((change.budget_bytes, change.sequence));
        thread::yield_now();
    }
    writer.join().unwrap()?;
    for _ in 0..100 {
        let change = grant.current_budget_change()?;
        observed.push((change.budget_bytes, change.sequence));
    }

    let published = published.lock().unwrap().iter().copied().collect::<HashSet<_>>();
    assert!(!observed.is_empty());
    for pair in observed {
        assert!(published.contains(&pair), "observed torn budget snapshot {pair:?}");
    }
    Ok(())
}

#[test]
fn budget_decrease_observer_runs_only_when_requested_shrink_is_applied() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();
    let (sender, receiver) = mpsc::channel();
    grant.notify_on_budget_decrease(Arc::new(move || sender.send(()).unwrap())).unwrap();

    assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
    assert!(receiver.try_recv().is_err());

    assert!(!grant.try_apply_requested_shrink(55).unwrap());
    receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(receiver.try_recv().is_err());
}

#[test]
fn shrink_request_after_satisfied_direct_shrink_is_not_recorded() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

    assert!(grant.try_shrink_to(40).unwrap());
    assert_eq!(grant.request_shrink_to(50).unwrap(), None);

    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
    assert!(grant.try_grow_to(70).unwrap());
    assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
}

#[test]
fn blocked_request_wakes_after_shrink() {
    let governor = MemoryGovernor::new(0, 64).unwrap();
    let active = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 32, 64)).unwrap().unwrap();
    let waiting_governor = governor.clone();
    let (sender, receiver) = mpsc::channel();
    let waiter = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        let grant = waiting_governor
            .request(GrantRequest::new(AdmissionRequestId::new(2), 32, 32).with_deadline(deadline))
            .unwrap();
        sender.send(grant.budget_bytes()).unwrap();
    });

    thread::sleep(Duration::from_millis(25));
    assert!(active.try_shrink_to(32).unwrap());

    assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 32);
    waiter.join().unwrap();
    assert_eq!(governor.available_bytes().unwrap(), 32);
}

#[test]
fn blocked_request_wakes_after_release_without_deadlock() {
    let governor = MemoryGovernor::new(0, 64).unwrap();
    let active = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 64, 64)).unwrap().unwrap();
    let waiting_governor = governor.clone();
    let (sender, receiver) = mpsc::channel();
    let waiter = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        let grant = waiting_governor
            .request(GrantRequest::new(AdmissionRequestId::new(2), 32, 32).with_deadline(deadline))
            .unwrap();
        sender.send(grant.budget_bytes()).unwrap();
    });

    thread::sleep(Duration::from_millis(25));
    drop(active);
    assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 32);
    waiter.join().unwrap();
    assert_eq!(governor.available_bytes().unwrap(), 64);
}

#[test]
fn evictable_bytes_are_reported_as_grantable_but_not_free() {
    let governor = MemoryGovernor::new(0, 100).unwrap();

    assert!(governor.try_reserve_evictable_bytes(60).unwrap());
    assert_eq!(governor.evictable_bytes().unwrap(), 60);
    assert_eq!(governor.free_bytes().unwrap(), 40);
    assert_eq!(governor.available_bytes().unwrap(), 100);

    let report = governor.report().unwrap();
    assert_eq!(report.outstanding_bytes, 0);
    assert_eq!(report.evictable_bytes, 60);
    assert_eq!(report.free_bytes, 40);
    assert_eq!(report.available_bytes, 100);

    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 50)).unwrap().unwrap();
    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(governor.free_bytes().unwrap(), 0);
    assert_eq!(governor.available_bytes().unwrap(), 60);
    assert!(!governor.try_reserve_evictable_bytes(1).unwrap());
}

#[test]
fn evictable_bytes_can_be_reserved_from_active_grant_budget() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 100, 100)).unwrap().unwrap();

    assert_eq!(governor.free_bytes().unwrap(), 0);
    assert!(!governor.try_reserve_evictable_bytes(100).unwrap());
    assert!(governor.grant_can_cover_evictable_bytes(&grant, 100).unwrap());
    assert!(governor.try_reserve_evictable_bytes_from_grant(&grant, 100).unwrap());

    let active_report = governor.report().unwrap();
    assert_eq!(active_report.outstanding_bytes, 100);
    assert_eq!(active_report.evictable_bytes, 100);
    assert_eq!(active_report.free_bytes, 0);

    drop(grant);
    let retained_report = governor.report().unwrap();
    assert_eq!(retained_report.outstanding_bytes, 0);
    assert_eq!(retained_report.evictable_bytes, 100);
    assert_eq!(retained_report.free_bytes, 0);
}

#[test]
fn evictable_grant_transfer_respects_active_grant_budget() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 50)).unwrap().unwrap();

    assert!(!governor.grant_can_cover_evictable_bytes(&grant, 51).unwrap());
    assert!(!governor.try_reserve_evictable_bytes_from_grant(&grant, 51).unwrap());
    assert!(governor.grant_can_cover_evictable_bytes(&grant, 50).unwrap());
    assert!(governor.try_reserve_evictable_bytes_from_grant(&grant, 50).unwrap());
}

#[test]
fn pressure_observer_releases_evictable_bytes_before_grant() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    assert!(governor.try_reserve_evictable_bytes(80).unwrap());
    let observer_governor = governor.clone();
    let (sender, receiver) = mpsc::channel();
    governor
        .register_evictable_memory_observer(Arc::new(move |pressure| {
            let report = observer_governor.report().unwrap();
            sender.send((pressure, report.free_bytes)).unwrap();
            observer_governor.release_evictable_bytes(pressure.requested_evict_bytes)
        }))
        .unwrap();

    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 40, 50)).unwrap().unwrap();

    assert_eq!(grant.budget_bytes(), 40);
    let (pressure, observer_free_bytes) = receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(pressure.device_ordinal, 0);
    assert_eq!(pressure.admission_request_id, AdmissionRequestId::new(7));
    assert_eq!(pressure.requested_evict_bytes, 20);
    assert_eq!(pressure.min_grant_bytes, 40);
    assert_eq!(pressure.want_grant_bytes, 50);
    assert_eq!(observer_free_bytes, 20);
    let report = governor.report().unwrap();
    assert_eq!(report.outstanding_bytes, 40);
    assert_eq!(report.evictable_bytes, 60);
    assert_eq!(report.free_bytes, 0);
    assert_eq!(report.available_bytes, 60);
}

#[test]
fn grant_growth_releases_evictable_bytes_before_growing() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 20, 40)).unwrap().unwrap();
    assert!(governor.try_reserve_evictable_bytes(50).unwrap());
    let observer_governor = governor.clone();
    let (sender, receiver) = mpsc::channel();
    governor
        .register_evictable_memory_observer(Arc::new(move |pressure| {
            sender.send(pressure).unwrap();
            observer_governor.release_evictable_bytes(pressure.requested_evict_bytes)
        }))
        .unwrap();

    assert!(grant.try_grow(30).unwrap());

    let pressure = receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(pressure.device_ordinal, 0);
    assert_eq!(pressure.admission_request_id, AdmissionRequestId::new(7));
    assert_eq!(pressure.requested_evict_bytes, 20);
    assert_eq!(pressure.min_grant_bytes, 30);
    assert_eq!(pressure.want_grant_bytes, 30);
    let report = governor.report().unwrap();
    assert_eq!(grant.budget_bytes(), 70);
    assert_eq!(report.outstanding_bytes, 70);
    assert_eq!(report.evictable_bytes, 30);
    assert_eq!(report.free_bytes, 0);
}

#[test]
fn pressure_observer_cannot_fake_released_evictable_bytes() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    assert!(governor.try_reserve_evictable_bytes(80).unwrap());
    governor.register_evictable_memory_observer(Arc::new(move |pressure| Ok(pressure.requested_evict_bytes))).unwrap();

    assert!(governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 40, 50)).unwrap().is_none());
    let report = governor.report().unwrap();
    assert_eq!(report.outstanding_bytes, 0);
    assert_eq!(report.evictable_bytes, 80);
    assert_eq!(report.free_bytes, 20);
}

#[test]
fn pressure_observer_error_does_not_skip_later_observer() {
    let governor = MemoryGovernor::new(0, 100).unwrap();
    assert!(governor.try_reserve_evictable_bytes(80).unwrap());
    let failing_observer_called = Arc::new(AtomicU64::new(0));
    let succeeding_observer_called = Arc::new(AtomicU64::new(0));
    let failing_count = Arc::clone(&failing_observer_called);
    governor
        .register_evictable_memory_observer(Arc::new(move |_pressure| {
            failing_count.fetch_add(1, Ordering::AcqRel);
            Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_governor.test_observer",
                "synthetic observer failure",
            ))
        }))
        .unwrap();
    let observer_governor = governor.clone();
    let succeeding_count = Arc::clone(&succeeding_observer_called);
    governor
        .register_evictable_memory_observer(Arc::new(move |pressure| {
            succeeding_count.fetch_add(1, Ordering::AcqRel);
            observer_governor.release_evictable_bytes(pressure.requested_evict_bytes)
        }))
        .unwrap();

    let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 40, 50)).unwrap().unwrap();

    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(failing_observer_called.load(Ordering::Acquire), 1);
    assert_eq!(succeeding_observer_called.load(Ordering::Acquire), 1);
    let report = governor.report().unwrap();
    assert_eq!(report.outstanding_bytes, 40);
    assert_eq!(report.evictable_bytes, 60);
    assert_eq!(report.free_bytes, 0);
}

#[test]
fn blocked_request_wakes_after_evictable_release() {
    let governor = MemoryGovernor::new(0, 64).unwrap();
    assert!(governor.try_reserve_evictable_bytes(64).unwrap());
    let waiting_governor = governor.clone();
    let (sender, receiver) = mpsc::channel();
    let waiter = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        let grant = waiting_governor
            .request(GrantRequest::new(AdmissionRequestId::new(2), 32, 32).with_deadline(deadline))
            .unwrap();
        sender.send(grant.budget_bytes()).unwrap();
    });

    thread::sleep(Duration::from_millis(25));
    assert_eq!(governor.release_evictable_bytes(32).unwrap(), 32);

    assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 32);
    waiter.join().unwrap();
    let report = governor.report().unwrap();
    assert_eq!(report.outstanding_bytes, 0);
    assert_eq!(report.evictable_bytes, 32);
    assert_eq!(report.free_bytes, 32);
}

#[test]
fn report_is_separate_grant_surface() {
    let governor = MemoryGovernor::new(1, 100).unwrap();
    let _grant = governor
        .try_request(
            GrantRequest::new(AdmissionRequestId::new(7), 10, 25).with_qos(QueryQos::new(super::super::Priority::High)),
        )
        .unwrap()
        .unwrap();

    let report = governor.report().unwrap();
    assert_eq!(report.device_ordinal, 1);
    assert_eq!(report.capacity_bytes, 100);
    assert_eq!(report.outstanding_bytes, 25);
    assert_eq!(report.evictable_bytes, 0);
    assert_eq!(report.free_bytes, 75);
    assert_eq!(report.available_bytes, 75);
    assert_eq!(report.grants.len(), 1);
    assert_eq!(report.grants[0].priority, "high");
}
