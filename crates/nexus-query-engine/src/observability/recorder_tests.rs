use std::{
    collections::HashSet,
    panic::AssertUnwindSafe,
    sync::{
        Arc, Barrier, Mutex,
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
                owner.finish_query(RuntimeTerminalOutcome::Succeeded).expect("enabled owner seals")
            })
        })
        .collect::<Vec<_>>();
    let snapshots = threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>();
    assert!(snapshots.iter().all(|snapshot| snapshot.is_same_snapshot(&snapshots[0])));
    assert_eq!(snapshots[0].footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
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
        let reentered = callback_owner
            .finish_query(RuntimeTerminalOutcome::Succeeded)
            .expect("reentrant finisher receives the published result");
        callback_tx.send(reentered.is_same_snapshot(&observed)).unwrap();
    });

    let finisher = std::thread::spawn(move || owner.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap());
    assert!(callback_rx.recv_timeout(Duration::from_secs(2)).expect("observer must not deadlock"));
    let snapshot = finisher.join().unwrap();
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
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
        std::panic::catch_unwind(AssertUnwindSafe(|| primary_owner.finish_query(RuntimeTerminalOutcome::Succeeded)))
    });
    clock.entered.wait();

    let waiters = (0..8)
        .map(|_| {
            let owner = owner.clone();
            std::thread::spawn(move || {
                owner.finish_query(RuntimeTerminalOutcome::Succeeded).expect("waiter receives recovered seal")
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

    let sealed = owner.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap();
    let delivered = delivered.lock().unwrap();
    assert_eq!(delivered.len(), 1);
    assert!(delivered[0].is_same_snapshot(&sealed));
    drop(delivered);

    let late = Arc::new(Mutex::new(None));
    let late_for_observer = Arc::clone(&late);
    owner.register_seal_observer(move |snapshot| *late_for_observer.lock().unwrap() = Some(snapshot));
    assert!(late.lock().unwrap().as_ref().is_some_and(|snapshot| snapshot.is_same_snapshot(&sealed)));
}
