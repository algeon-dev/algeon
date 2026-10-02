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

use query_planning::error::{QueryError, QueryErrorCode};

use super::*;
use crate::{
    metrics::{BudgetedExecutionEvent, SourceReadRuntimeMetrics},
    observability::{
        CleanupFailurePhase, DeterministicRuntimeObservationClock, PlanNodeRow, QueryBoundaryKind, RuntimePlanNodeId,
    },
    runtime::{LakehouseCacheFillErrorEvent, LakehouseCacheFillEvent, LakehouseCacheFillOutcome},
};

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
        RuntimeTopologyRecord::PlanNode(PlanNodeRow {
            plan_execution_id: PlanExecutionId::new("plan"),
            node_id: RuntimePlanNodeId::new("node"),
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
    let secondary_owner = owner;
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
        Arc::clone(&clock) as _,
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

fn metrics_session(query_id: &str, config: RuntimeObservationConfig) -> RuntimeObservationSession {
    RuntimeObservationSession::with_clock(
        query_id,
        None,
        config,
        Arc::new(DeterministicRuntimeObservationClock::new(Some(1))),
    )
}

fn plan() -> PlanExecutionId {
    PlanExecutionId::new("plan-a")
}

fn source_row(batch_index: u64) -> SourceReadRuntimeMetrics {
    SourceReadRuntimeMetrics { source_node_id: "source".to_string(), batch_index, ..Default::default() }
}

#[test]
fn disabled_query_session_keeps_exact_totals_without_capturing_trace() {
    let session = metrics_session("disabled-metrics", RuntimeObservationConfig::disabled());
    let handle = session.handle();

    handle.record_source_read(&plan(), SourceReadFact::InputBatch { rows: 7 });
    handle.record_source_read(&plan(), SourceReadFact::InputBatch { rows: 5 });
    handle.record_source_read_row(&plan(), source_row(0));
    let payload_builds = Arc::new(AtomicUsize::new(0));
    let payload_counter = Arc::clone(&payload_builds);
    let disposition = handle.record(move || {
        payload_counter.fetch_add(1, Ordering::SeqCst);
        RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started }
    });

    // Trace behaviour is exactly what a disabled session always had.
    assert!(!handle.is_enabled());
    assert!(matches!(disposition, RuntimeObservationRecordDisposition::Disabled));
    assert_eq!(payload_builds.load(Ordering::SeqCst), 0, "a disabled trace must not build payloads");
    assert!(handle.live_snapshot().is_none());

    // Metrics survive it.
    let checkpoint = handle.metrics_checkpoint(&plan()).expect("a query-bound session always has metrics");
    assert_eq!(checkpoint.summary.source_read.input_batches, 2);
    assert_eq!(checkpoint.summary.source_read.input_rows, 12);
    assert!(checkpoint.source_read_rows.is_empty(), "disabled capture keeps no detail rows");
    assert_eq!(checkpoint.completeness.capture_mode, MetricsCaptureMode::Disabled);
    assert!(!checkpoint.completeness.detail_is_complete(), "no capture is not a complete capture");
}

#[test]
fn bounded_capture_truncates_detail_but_keeps_totals_exact() {
    let session = metrics_session("bounded-metrics", RuntimeObservationConfig::bounded(2, 1024 * 1024));
    let handle = session.handle();

    for batch_index in 0..5 {
        handle.record_source_read(&plan(), SourceReadFact::ChunkCommitted { chunk_count: 1, row_groups: 2 });
        handle.record_source_read_row(&plan(), source_row(batch_index));
    }

    let checkpoint = handle.metrics_checkpoint(&plan()).unwrap();
    assert_eq!(checkpoint.summary.source_read.source_chunks_committed, 5);
    assert_eq!(checkpoint.summary.source_read.row_groups_committed, 10);
    let retained = checkpoint.source_read_rows.iter().map(|row| row.batch_index).collect::<Vec<_>>();
    assert_eq!(retained, vec![0, 1], "retention keeps the earliest rows up to the record limit");
    assert_eq!(checkpoint.completeness.detail_rows_dropped, 3);
    assert!(checkpoint.completeness.truncated);
    assert!(!checkpoint.completeness.detail_is_complete());

    // The loss is visible through the trace footer too, as its own class.
    let snapshot = handle.live_snapshot().unwrap();
    let loss = snapshot
        .footer()
        .dropped_by_class
        .iter()
        .find(|loss| loss.class == RuntimeObservationRecordClass::SourceReadDetail)
        .map(|loss| loss.count);
    assert_eq!(loss, Some(3));
    assert!(snapshot.footer().truncated);
}

#[test]
fn budget_detail_cannot_crowd_out_later_strategy_decisions() {
    let session = metrics_session("budget-share", RuntimeObservationConfig::bounded(8, 1024 * 1024));
    let handle = session.handle();
    let event = BudgetedExecutionEvent {
        query_id: None,
        operation: "groupby".to_string(),
        boundary: "partial".to_string(),
        budget_bytes: 64,
        enforcement: "stream_bound_reservation".to_string(),
        status: "within_budget".to_string(),
        allocator_high_water_bytes: Some(32),
        retained_bytes: None,
        output_bytes: None,
        output_rows: None,
        context_facts: std::collections::BTreeMap::new(),
    };

    for _ in 0..20 {
        let event = event.clone();
        handle.apply_metrics(&plan(), move |metrics| metrics.budgeted_execution_events.push(event));
    }
    let facts = std::collections::BTreeMap::from([("execution_stage", "partial".to_string())]);
    handle.apply_metrics(&plan(), |metrics| metrics.record_aggregate_strategy_decision("aggregate", &facts));

    let checkpoint = handle.metrics_checkpoint(&plan()).unwrap();
    assert_eq!(checkpoint.metrics.budgeted_execution_events.len(), 4, "budget detail keeps half of the records");
    assert_eq!(checkpoint.metrics.aggregate_strategy_decisions.len(), 1);
    assert_eq!(checkpoint.completeness.detail_rows_dropped, 16);
}

#[test]
fn zero_capacity_capture_loses_every_row_and_no_total() {
    let session = metrics_session("zero-cap-metrics", RuntimeObservationConfig::bounded(0, 0));
    let handle = session.handle();

    handle.record_source_read(&plan(), SourceReadFact::QueueWait(Duration::from_millis(4)));
    handle.record_source_read_row(&plan(), source_row(0));

    let checkpoint = handle.metrics_checkpoint(&plan()).unwrap();
    assert_eq!(checkpoint.summary.source_read.source_read_queue_wait_time, Duration::from_millis(4));
    assert!(checkpoint.source_read_rows.is_empty());
    assert_eq!(checkpoint.completeness.capture_mode, MetricsCaptureMode::Bounded);
    assert_eq!(checkpoint.completeness.detail_rows_dropped, 1);
}

#[test]
fn sealing_freezes_metrics_whether_or_not_trace_is_captured() {
    for config in [RuntimeObservationConfig::disabled(), RuntimeObservationConfig::native_execution()] {
        let session = metrics_session("sealed-metrics", config);
        let handle = session.handle();
        handle.record_source_read(&plan(), SourceReadFact::ChunkAttempted);

        let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded);
        assert_eq!(snapshot.is_some(), config.is_enabled(), "only a captured trace produces a sealed snapshot");

        handle.record_source_read(&plan(), SourceReadFact::ChunkAttempted);
        handle.record_source_read_row(&plan(), source_row(9));
        let checkpoint = handle.metrics_checkpoint(&plan()).unwrap();
        assert!(checkpoint.sealed);
        assert_eq!(checkpoint.summary.source_read.source_chunks_attempted, 1, "facts after the seal are ignored");
        assert!(checkpoint.source_read_rows.is_empty());
    }
}

#[test]
fn revision_advances_only_when_metrics_change() {
    let session = metrics_session("revision-metrics", RuntimeObservationConfig::native_execution());
    let handle = session.handle();

    let initial = handle.metrics_checkpoint(&plan()).unwrap().revision;
    assert_eq!(handle.metrics_checkpoint(&plan()).unwrap().revision, initial, "reading must not advance the revision");

    handle.record_source_read(&plan(), SourceReadFact::PrefetchStarted);
    let after_fact = handle.metrics_checkpoint(&plan()).unwrap().revision;
    assert!(after_fact > initial);

    handle.record_source_read_row(&plan(), source_row(0));
    assert!(handle.metrics_checkpoint(&plan()).unwrap().revision > after_fact);
}

#[test]
fn saturated_totals_are_reported_as_inexact() {
    let session = metrics_session("saturated-metrics", RuntimeObservationConfig::disabled());
    let handle = session.handle();

    handle.record_source_read(&plan(), SourceReadFact::InputBatch { rows: u64::MAX });
    assert!(!handle.metrics_checkpoint(&plan()).unwrap().completeness.totals_saturated);
    handle.record_source_read(&plan(), SourceReadFact::InputBatch { rows: 1 });

    let checkpoint = handle.metrics_checkpoint(&plan()).unwrap();
    assert_eq!(checkpoint.summary.source_read.input_rows, u64::MAX);
    assert!(checkpoint.completeness.totals_saturated);
}

#[test]
fn the_disabled_sentinel_has_no_metrics_owner() {
    let handle = RuntimeObservationHandle::disabled();

    handle.record_source_read(&plan(), SourceReadFact::ChunkAttempted);

    assert!(handle.metrics_checkpoint(&plan()).is_none(), "a handle with no query has nothing to reduce into");
}

#[test]
fn each_plan_execution_projects_only_its_own_facts() {
    // One query, two plan executions sharing one owner. A consumer that sums
    // the two projections must see each fact once.
    let session = metrics_session("two-plan-metrics", RuntimeObservationConfig::native_execution());
    let handle = session.handle();
    let first = PlanExecutionId::new("plan-a");
    let second = PlanExecutionId::new("plan-b");

    handle.record_source_read(&first, SourceReadFact::InputBatch { rows: 3 });
    handle.record_source_read(&second, SourceReadFact::InputBatch { rows: 4 });
    handle.record_source_read_row(&second, source_row(1));

    let first_checkpoint = handle.metrics_checkpoint(&first).unwrap();
    let second_checkpoint = handle.metrics_checkpoint(&second).unwrap();
    assert_eq!(first_checkpoint.summary.source_read.input_rows, 3);
    assert!(first_checkpoint.source_read_rows.is_empty());
    assert_eq!(second_checkpoint.summary.source_read.input_rows, 4);
    assert_eq!(second_checkpoint.source_read_rows.len(), 1);

    let unseen = handle.metrics_checkpoint(&PlanExecutionId::new("plan-never-ran")).unwrap();
    assert_eq!(unseen.summary, QueryMetricsSummary::default(), "an execution with no facts reports empty totals");
}

#[test]
fn a_frozen_plan_ignores_later_facts_and_seals_its_frozen_metrics() {
    let session = metrics_session("frozen-plan-metrics", RuntimeObservationConfig::native_execution());
    let handle = session.handle();
    let frozen = PlanExecutionId::new("plan-frozen");
    let open = PlanExecutionId::new("plan-open");

    handle.record_source_read(&frozen, SourceReadFact::InputBatch { rows: 3 });
    let checkpoint = handle.freeze_plan_metrics(&frozen).unwrap();
    handle.record_source_read(&frozen, SourceReadFact::InputBatch { rows: 4 });
    handle.record_source_read_row(&frozen, source_row(1));
    handle.record_source_read(&open, SourceReadFact::InputBatch { rows: 5 });

    assert_eq!(checkpoint.summary.source_read.input_rows, 3);
    assert_eq!(handle.metrics_checkpoint(&frozen).unwrap(), checkpoint, "a late fact leaves the frozen value");
    let sealed = session.seal(RuntimeTerminalOutcome::Succeeded).unwrap();
    let frozen_metrics = sealed.query_metrics(&frozen).expect("the seal carries the frozen plan");
    assert_eq!(frozen_metrics.input_rows, 3);
    assert!(frozen_metrics.source_read_runtime_rows.is_empty());
    assert_eq!(frozen_metrics.query_id.as_str(), "frozen-plan-metrics");
    assert_eq!(sealed.query_metrics(&open).expect("an open plan freezes at seal").input_rows, 5);
}

fn cache_fill_event(cache_path_hash: &str, error: Option<LakehouseCacheFillErrorEvent>) -> LakehouseCacheFillEvent {
    LakehouseCacheFillEvent {
        outcome: if error.is_some() { LakehouseCacheFillOutcome::Failed } else { LakehouseCacheFillOutcome::Completed },
        bytes: error.is_none().then_some(4096),
        elapsed: Duration::from_millis(7),
        cache_path_hash: cache_path_hash.to_string(),
        stable_id_hash: "stable-id-hash".to_string(),
        file_index: Some(2),
        chunk_index: None,
        error,
    }
}

#[test]
fn cache_fill_events_reach_the_next_plan_to_freeze_once() {
    let session = metrics_session("cache-fill-events", RuntimeObservationConfig::native_execution());
    let handle = session.handle();
    let first = PlanExecutionId::new("plan-a");
    let second = PlanExecutionId::new("plan-b");

    handle.record_lakehouse_cache_fill(&cache_fill_event("completed-path", None));
    handle.record_lakehouse_cache_fill(&cache_fill_event(
        "failed-path",
        Some(LakehouseCacheFillErrorEvent {
            code: "cudf_dependency_failure".to_string(),
            kind: "external".to_string(),
            status: "temporary".to_string(),
            operation: "query_engine.lakehouse_cache.background_fill.fetch".to_string(),
        }),
    ));
    let mut first_metrics = QueryMetrics::default();
    first_metrics.apply_owner_checkpoint(&handle.freeze_plan_metrics(&first).unwrap());
    handle.record_lakehouse_cache_fill(&cache_fill_event("later-path", None));
    let mut second_metrics = QueryMetrics::default();
    second_metrics.apply_owner_checkpoint(&handle.freeze_plan_metrics(&second).unwrap());

    let first_events = &first_metrics.lakehouse_cache_fill_events;
    assert_eq!(first_events.len(), 2);
    assert_eq!((first_events[0].outcome, first_events[0].bytes), ("completed", Some(4096)));
    assert_eq!(first_events[1].outcome, "failed");
    assert_eq!(first_events[1].error_operation.as_deref(), Some("query_engine.lakehouse_cache.background_fill.fetch"));
    assert_eq!(
        second_metrics
            .lakehouse_cache_fill_events
            .iter()
            .map(|event| event.cache_path_hash.as_str())
            .collect::<Vec<_>>(),
        ["later-path"],
        "a frozen plan keeps its events and the next plan takes only later ones",
    );
}

#[test]
fn cache_fill_events_share_the_query_record_limit_and_count_their_loss() {
    let session = metrics_session("cache-fill-loss", RuntimeObservationConfig::bounded(1, 64 * 1024));
    let handle = session.handle();
    let plan = PlanExecutionId::new("plan");

    handle.record_lakehouse_cache_fill(&cache_fill_event("kept-path", None));
    handle.record_lakehouse_cache_fill(&cache_fill_event("dropped-path", None));
    let checkpoint = handle.freeze_plan_metrics(&plan).unwrap();
    let mut metrics = QueryMetrics::default();
    metrics.apply_owner_checkpoint(&checkpoint);
    let sealed = session.seal(RuntimeTerminalOutcome::Succeeded).unwrap();

    assert_eq!(metrics.lakehouse_cache_fill_events.len(), 1);
    assert_eq!(metrics.lakehouse_cache_fill_events[0].cache_path_hash, "kept-path");
    assert_eq!(checkpoint.completeness.detail_rows_dropped, 1, "the dropped event counts against its plan");
    assert!(checkpoint.completeness.truncated);
    assert!(sealed.footer().truncated);
    assert!(
        sealed
            .footer()
            .dropped_by_class
            .iter()
            .any(|loss| loss.class == RuntimeObservationRecordClass::LakehouseCacheFillDetail && loss.count == 1),
        "{:?}",
        sealed.footer().dropped_by_class,
    );
}

#[test]
fn cache_fill_events_no_plan_carries_are_lost_at_the_seal() {
    let session = metrics_session("cache-fill-unassigned", RuntimeObservationConfig::native_execution());
    let handle = session.handle();
    let plan = PlanExecutionId::new("plan");

    handle.record_lakehouse_cache_fill(&cache_fill_event("carried-path", None));
    let carried = handle.freeze_plan_metrics(&plan).unwrap();
    handle.record_lakehouse_cache_fill(&cache_fill_event("late-path", None));
    let sealed = session.seal(RuntimeTerminalOutcome::Succeeded).unwrap();
    let after_seal = handle.freeze_plan_metrics(&PlanExecutionId::new("after-seal")).unwrap();

    assert_eq!(carried.metrics.lakehouse_cache_fill_events.len(), 1);
    assert!(
        after_seal.metrics.lakehouse_cache_fill_events.is_empty(),
        "a freeze after the seal returns the sealed value"
    );
    assert!(sealed.footer().truncated);
    assert!(
        sealed
            .footer()
            .dropped_by_reason
            .iter()
            .any(|loss| loss.reason == RuntimeObservationLossReason::Unassigned && loss.count == 1),
        "{:?}",
        sealed.footer().dropped_by_reason,
    );
}

#[test]
fn cleanup_failure_tally_never_publishes_a_saturated_exact_count() {
    let mut tally = CleanupFailureTally::with_count_for_test(u64::MAX);
    tally.record(CleanupFailureSummary::from_native_error(
        CleanupFailurePhase::QueryServiceCleanup,
        &QueryError::new(QueryErrorCode::Internal, "test.cleanup", "overflow probe"),
    ));

    assert_eq!(tally.count(), u64::MAX);
    assert!(!tally.complete());
}
