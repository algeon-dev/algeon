use std::{
    collections::HashSet,
    panic::AssertUnwindSafe,
    sync::{
        Arc, Barrier, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use super::*;
use crate::observability::{DeterministicRuntimeObservationClock, QueryBoundaryKind};

#[test]
fn sized_preflight_rejects_before_payload_or_topology_builder_runs() {
    let session = RuntimeObservationSession::with_clock(
        "preflight-query",
        None,
        RuntimeObservationConfig::bounded(0, 0),
        Arc::new(DeterministicRuntimeObservationClock::new(Some(1))),
    );
    let payload_builds = Arc::new(AtomicUsize::new(0));
    let payload_counter = Arc::clone(&payload_builds);
    let disposition =
        session.handle().record_with_context_sized(RuntimeEventContext::default(), 1_000_000, move || {
            payload_counter.fetch_add(1, Ordering::SeqCst);
            RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started }
        });
    assert!(matches!(
        disposition,
        RuntimeObservationRecordDisposition::Dropped { reason: RuntimeObservationLossReason::RecordLimit, .. }
    ));
    assert_eq!(payload_builds.load(Ordering::SeqCst), 0);

    let topology_builds = Arc::new(AtomicUsize::new(0));
    let topology_counter = Arc::clone(&topology_builds);
    let disposition = session.handle().record_topology(RuntimeObservationRecordClass::PlanNode, move || {
        topology_counter.fetch_add(1, Ordering::SeqCst);
        RuntimeTopologyRecord::PlanNode(super::super::PlanNodeRow {
            plan_execution_id: PlanExecutionId::new("plan"),
            node_id: super::super::RuntimePlanNodeId::new("node"),
            operator_kind: "never-built".to_string(),
        })
    });
    assert!(matches!(
        disposition,
        RuntimeObservationRecordDisposition::Dropped { reason: RuntimeObservationLossReason::RecordLimit, .. }
    ));
    assert_eq!(topology_builds.load(Ordering::SeqCst), 0);
}

#[test]
fn timestamped_drop_samples_clamped_clock_without_building_payload() {
    let clock = Arc::new(DeterministicRuntimeObservationClock::new(Some(1)));
    clock.set_monotonic_ns(77);
    let session = RuntimeObservationSession::with_clock(
        "timestamped-preflight-query",
        None,
        RuntimeObservationConfig::bounded(0, 0),
        clock,
    );
    let payload_builds = Arc::new(AtomicUsize::new(0));
    let payload_counter = Arc::clone(&payload_builds);
    let timestamp = session.handle().record_with_context_timestamped(RuntimeEventContext::default(), move |_| {
        payload_counter.fetch_add(1, Ordering::SeqCst);
        RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started }
    });

    assert_eq!(timestamp, Some(77));
    assert_eq!(payload_builds.load(Ordering::SeqCst), 0);
    let snapshot = session.live_snapshot().unwrap();
    assert!(snapshot.events().is_empty());
    assert!(snapshot.footer().truncated);
}

#[test]
fn concurrent_owner_finishers_wait_for_and_share_one_result_cell() {
    let owner =
        RuntimeObservationOwner::new("finish-race-query", None, RuntimeObservationConfig::bounded(64, 16 * 1024));
    let racers = 16;
    let barrier = Arc::new(Barrier::new(racers));
    let threads = (0..racers)
        .map(|_| {
            let owner = owner.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).expect("enabled owner seals")
            })
        })
        .collect::<Vec<_>>();
    let snapshots = threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>();
    assert!(snapshots.iter().all(|snapshot| snapshot.is_same_snapshot(&snapshots[0])));
    assert_eq!(snapshots[0].footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
}

#[test]
fn coordinator_owned_finish_seals_without_an_attempt_terminal() {
    let owner =
        RuntimeObservationOwner::new("coordinator-owned-query", None, RuntimeObservationConfig::bounded(64, 16 * 1024));

    let sealed = owner.finish_query(None).expect("enabled coordinator owner seals");

    assert_eq!(sealed.footer().terminal, None);
    assert!(sealed.events().iter().all(|event| {
        !matches!(
            event.payload,
            RuntimeObservationPayload::QueryBoundary {
                boundary: QueryBoundaryKind::Succeeded
                    | QueryBoundaryKind::Failed
                    | QueryBoundaryKind::Cancelled
                    | QueryBoundaryKind::Dropped
                    | QueryBoundaryKind::Panicked,
            }
        )
    }));
}

#[test]
fn non_reentrant_finisher_waits_for_observer_delivery() {
    let owner =
        RuntimeObservationOwner::new("blocking-observer-query", None, RuntimeObservationConfig::bounded(64, 16 * 1024));
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let observer_gate = Arc::clone(&gate);
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    owner.register_seal_observer(move |_| {
        entered_tx.send(()).expect("test receives observer entry");
        let (lock, changed) = &*observer_gate;
        let mut released = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*released {
            released = changed.wait(released).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    });

    let primary_owner = owner.clone();
    let primary =
        std::thread::spawn(move || primary_owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).unwrap());
    entered_rx.recv().expect("blocking observer entered delivery");
    assert!(matches!(
        &*owner.lock_lifecycle(),
        RuntimeObservationOwnerLifecycleState::Delivering { snapshot: Some(_), .. }
    ));

    let (started_tx, started_rx) = mpsc::sync_channel(0);
    let (completed_tx, completed_rx) = mpsc::sync_channel(1);
    let secondary_owner = owner.clone();
    let secondary_thread = std::thread::spawn(move || {
        started_tx.send(()).expect("test observes second finisher start");
        let snapshot = secondary_owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).unwrap();
        completed_tx.send(snapshot).expect("test receives second finisher result");
    });
    started_rx.recv().expect("second finisher started");
    assert!(matches!(completed_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));

    let (lock, changed) = &*gate;
    *lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
    changed.notify_all();
    let primary = primary.join().expect("primary finisher completed");
    let secondary = completed_rx.recv().expect("second finisher resumes after delivery");
    secondary_thread.join().expect("second finisher thread completed");
    assert!(primary.is_same_snapshot(&secondary));
}

#[test]
fn late_observer_registration_joins_the_delivery_fence() {
    enum LateDeliveryEvent {
        RegistrationReturned,
        ObserverEntered,
    }

    let owner = RuntimeObservationOwner::new(
        "late-observer-delivery-query",
        None,
        RuntimeObservationConfig::bounded(64, 16 * 1024),
    );
    let primary_gate = Arc::new((Mutex::new(false), Condvar::new()));
    let primary_observer_gate = Arc::clone(&primary_gate);
    let (primary_entered_tx, primary_entered_rx) = mpsc::sync_channel(1);
    owner.register_seal_observer(move |_| {
        primary_entered_tx.send(()).expect("test receives primary observer entry");
        let (lock, changed) = &*primary_observer_gate;
        let mut released = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*released {
            released = changed.wait(released).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    });

    let primary_owner = owner.clone();
    let primary =
        std::thread::spawn(move || primary_owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).unwrap());
    primary_entered_rx.recv().expect("primary observer entered delivery");

    let late_gate = Arc::new((Mutex::new(false), Condvar::new()));
    let late_observer_gate = Arc::clone(&late_gate);
    let (event_tx, event_rx) = mpsc::channel();
    let late_owner = owner.clone();
    let late_registration = std::thread::spawn(move || {
        let observer_events = event_tx.clone();
        late_owner.register_seal_observer(move |_| {
            observer_events.send(LateDeliveryEvent::ObserverEntered).expect("test receives late observer entry");
            let (lock, changed) = &*late_observer_gate;
            let mut released = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            while !*released {
                released = changed.wait(released).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        });
        event_tx.send(LateDeliveryEvent::RegistrationReturned).expect("test observes late registration return");
    });
    assert!(matches!(event_rx.recv().expect("late registration event"), LateDeliveryEvent::RegistrationReturned));

    let (started_tx, started_rx) = mpsc::sync_channel(0);
    let (completed_tx, completed_rx) = mpsc::sync_channel(1);
    let secondary_owner = owner.clone();
    let secondary = std::thread::spawn(move || {
        started_tx.send(()).expect("test observes second finisher start");
        completed_tx
            .send(secondary_owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).unwrap())
            .expect("test receives second finisher result");
    });
    started_rx.recv().expect("second finisher started");

    let (lock, changed) = &*primary_gate;
    *lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
    changed.notify_all();
    assert!(matches!(
        event_rx.recv().expect("late observer entered after primary release"),
        LateDeliveryEvent::ObserverEntered
    ));
    assert!(matches!(
        &*owner.lock_lifecycle(),
        RuntimeObservationOwnerLifecycleState::Delivering { snapshot: Some(_), .. }
    ));
    assert!(matches!(completed_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));

    let (lock, changed) = &*late_gate;
    *lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
    changed.notify_all();
    let primary = primary.join().expect("primary finisher completed");
    let secondary_snapshot = completed_rx.recv().expect("second finisher resumes after late observer delivery");
    secondary.join().expect("second finisher thread completed");
    late_registration.join().expect("late registration thread completed");
    assert!(primary.is_same_snapshot(&secondary_snapshot));
}

#[test]
fn seal_observer_can_reenter_owner_finish_after_result_publication() {
    let owner = RuntimeObservationOwner::new(
        "reentrant-observer-query",
        None,
        RuntimeObservationConfig::bounded(64, 16 * 1024),
    );
    let callback_owner = owner.clone();
    let (callback_tx, callback_rx) = mpsc::channel();
    owner.register_seal_observer(move |observed| {
        let in_delivery = in_runtime_observation_delivery();
        let reentered = callback_owner
            .finish_query(Some(RuntimeTerminalOutcome::Succeeded))
            .expect("reentrant finisher receives the published result");
        callback_tx.send((in_delivery, reentered.is_same_snapshot(&observed))).unwrap();
    });

    let finisher = std::thread::spawn(move || owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).unwrap());
    assert_eq!(callback_rx.recv_timeout(Duration::from_secs(2)).expect("observer must not deadlock"), (true, true));
    let snapshot = finisher.join().unwrap();
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
}

#[test]
fn seal_observer_delivery_context_is_scoped_to_the_callback() {
    let owner = RuntimeObservationOwner::new(
        "observer-delivery-context-query",
        None,
        RuntimeObservationConfig::bounded(64, 16 * 1024),
    );
    assert!(!in_runtime_observation_delivery());
    let (context_tx, context_rx) = mpsc::sync_channel(1);
    owner.register_seal_observer(move |_| {
        context_tx.send(in_runtime_observation_delivery()).expect("test receives delivery context");
    });

    owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).expect("owner seals");
    assert!(context_rx.recv().expect("observer ran"));
    assert!(!in_runtime_observation_delivery());
}

#[test]
fn late_delivery_observer_panic_is_isolated_before_completion() {
    let owner = RuntimeObservationOwner::new(
        "late-observer-panic-query",
        None,
        RuntimeObservationConfig::bounded(64, 16 * 1024),
    );
    let callback_owner = owner.clone();
    let (registered_tx, registered_rx) = mpsc::sync_channel(1);
    owner.register_seal_observer(move |_| {
        callback_owner.register_seal_observer(|_| panic!("late observer panic must be isolated"));
        registered_tx.send(()).expect("test observes late observer registration");
    });

    let snapshot =
        owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).expect("owner completes after late observer panic");
    registered_rx.recv().expect("initial observer registered a late observer");
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    assert!(matches!(&*owner.lock_lifecycle(), RuntimeObservationOwnerLifecycleState::Sealed(Some(_))));
}

struct BlockingPanickingClock {
    calls: AtomicUsize,
    entered: Barrier,
    release: Barrier,
}

impl RuntimeObservationClock for BlockingPanickingClock {
    fn monotonic_ns(&self) -> u64 {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 3 {
            self.entered.wait();
            self.release.wait();
            panic!("injected recorder clock panic");
        }
        u64::try_from(call).unwrap()
    }

    fn unix_epoch_ns(&self) -> Option<u64> {
        Some(1)
    }
}

#[test]
fn clock_panic_publishes_degraded_panicked_result_and_wakes_waiters() {
    let clock = Arc::new(BlockingPanickingClock {
        calls: AtomicUsize::new(0),
        entered: Barrier::new(2),
        release: Barrier::new(2),
    });
    let session = RuntimeObservationSession::with_clock(
        "clock-panic-query",
        None,
        RuntimeObservationConfig::bounded(64, 16 * 1024),
        clock.clone(),
    );
    let owner = RuntimeObservationOwner::from_session(session);
    let primary_owner = owner.clone();
    let primary = std::thread::spawn(move || {
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            primary_owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded))
        }))
    });
    clock.entered.wait();

    let waiters = (0..8)
        .map(|_| {
            let owner = owner.clone();
            std::thread::spawn(move || {
                owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).expect("waiter receives recovered seal")
            })
        })
        .collect::<Vec<_>>();
    clock.release.wait();

    assert!(primary.join().unwrap().is_err(), "the injected clock panic still unwinds the winning finisher");
    let snapshots = waiters.into_iter().map(|waiter| waiter.join().unwrap()).collect::<Vec<_>>();
    assert!(snapshots.iter().all(|snapshot| snapshot.is_same_snapshot(&snapshots[0])));
    assert_eq!(snapshots[0].footer().terminal, Some(RuntimeTerminalOutcome::Panicked));
    assert!(snapshots[0].footer().degraded);
}

#[test]
fn poisoned_recorder_recovers_and_marks_snapshot_degraded() {
    let session = RuntimeObservationSession::with_clock(
        "poison-query",
        None,
        RuntimeObservationConfig::bounded(8, 4096),
        Arc::new(DeterministicRuntimeObservationClock::new(Some(1))),
    );
    let inner = session.handle.inner.as_ref().expect("enabled session has state");
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _state = inner.state.lock().expect("initial mutex is healthy");
        panic!("poison recorder for recovery test");
    }));

    let result = session.record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started });
    assert!(matches!(result, RuntimeObservationRecordDisposition::Retained { .. }));
    let snapshot = session.live_snapshot().expect("live snapshot remains available");
    assert!(snapshot.footer().degraded);
    assert_eq!(snapshot.footer().poison_recoveries, 1);
}

#[test]
fn sequence_saturation_is_truthfully_exported_in_the_footer() {
    let session = RuntimeObservationSession::with_clock(
        "overflow-query",
        None,
        RuntimeObservationConfig::bounded(8, 4096),
        Arc::new(DeterministicRuntimeObservationClock::new(Some(1))),
    );
    let inner = session.handle.inner.as_ref().expect("enabled session has state");
    inner.state.lock().unwrap().next_sequence = u64::MAX;

    let disposition =
        session.record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started });
    assert!(matches!(disposition, RuntimeObservationRecordDisposition::Retained { .. }));
    let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded).unwrap();
    assert!(snapshot.footer().counter_overflow);
    assert_eq!(snapshot.footer().last_attempted_sequence.map(ObservationSequence::get), Some(u64::MAX));
    assert_eq!(serde_json::to_value(snapshot.footer()).unwrap()["counter_overflow"], true);
}

#[test]
fn source_location_ids_are_keyed_stable_unlinkable_and_unique() {
    let locations = [
        "/srv/private/customer-42/orders.parquet",
        "s3://alice:secret@bucket/private/key.parquet?token=top-secret",
        "https://catalog.example/table?warehouse=private&credential=hunter2",
    ];
    let first = RuntimeObservationSession::with_source_location_test_key(
        "source-id-first",
        None,
        RuntimeObservationConfig::bounded(1, 1024),
        [0x11; blake3::KEY_LEN],
    );
    let second = RuntimeObservationSession::with_source_location_test_key(
        "source-id-second",
        None,
        RuntimeObservationConfig::bounded(1, 1024),
        [0x22; blake3::KEY_LEN],
    );
    let first_handle = first.handle();
    let second_handle = second.handle();

    for location in locations {
        let id = first_handle.source_location_id(location).expect("enabled session has a key");
        assert_eq!(Some(id.clone()), first_handle.source_location_id(location));
        assert_ne!(Some(id.clone()), second_handle.source_location_id(location));
        assert!(id.as_str().starts_with("source_location:blake3_keyed_256:"));
        assert_eq!(id.as_str().len(), "source_location:blake3_keyed_256:".len() + 64);
        assert!(!id.as_str().contains(":len:"));
        assert!(!id.as_str().contains(location));
    }

    let protected = first_handle
        .source_location_id("s3://private-bucket/customer.csv?token=secret")
        .expect("enabled session has a key");
    let attacker = RuntimeObservationSession::with_source_location_test_key(
        "source-id-attacker",
        None,
        RuntimeObservationConfig::bounded(1, 1024),
        [0_u8; blake3::KEY_LEN],
    );
    for guess in [
        "s3://private-bucket/customer.csv?token=secret",
        "s3://private-bucket/customer.csv",
        "/customer.csv",
        "customer.csv",
    ] {
        assert_ne!(Some(protected.clone()), attacker.handle().source_location_id(guess));
    }

    let unique = (0..20_000)
        .map(|index| {
            first_handle
                .source_location_id(&format!("s3://fixture-bucket/partition={index}/data.parquet?revision={index}"))
                .expect("enabled session has a key")
        })
        .collect::<HashSet<_>>();
    assert_eq!(unique.len(), 20_000);
}

#[test]
fn seal_observers_receive_snapshot_identity_and_isolate_panics() {
    let owner = RuntimeObservationOwner::new("observer-query", None, RuntimeObservationConfig::bounded(8, 4096));
    let delivered = Arc::new(Mutex::new(Vec::new()));
    let delivered_for_observer = Arc::clone(&delivered);
    owner.register_seal_observer(move |snapshot| delivered_for_observer.lock().unwrap().push(snapshot));
    owner.register_seal_observer(|_| panic!("observer failure must be isolated"));

    let sealed = owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).unwrap();
    let delivered = delivered.lock().unwrap();
    assert_eq!(delivered.len(), 1);
    assert!(delivered[0].is_same_snapshot(&sealed));
    drop(delivered);

    let late = Arc::new(Mutex::new(None));
    let late_for_observer = Arc::clone(&late);
    owner.register_seal_observer(move |snapshot| *late_for_observer.lock().unwrap() = Some(snapshot));
    assert!(late.lock().unwrap().as_ref().is_some_and(|snapshot| snapshot.is_same_snapshot(&sealed)));
}
