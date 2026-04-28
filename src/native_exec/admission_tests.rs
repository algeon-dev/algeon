use std::time::Duration;

use arrow::datatypes::Schema;
use datafusion::{
    execution::context::TaskContext,
    physical_plan::{ExecutionPlan, empty::EmptyExec},
};
use futures::StreamExt;
use nexus_query_engine::runtime::{
    ExecutionCancellation, QueryGrantRequest, QueryServiceConfig, QueryServiceTenantConfig,
};

use super::*;

fn admission_only_controller(tenant_id: &TenantId) -> TestAdmissionController {
    TestAdmissionController::new(
        QueryServiceConfig {
            max_concurrent_queries: 1,
            device_capacity_bytes: Some(1024),
            max_admission_queue_len: 1,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new(tenant_id.clone(), 1024)],
    )
    .unwrap()
}

fn admission_only_ticket() -> (Arc<QueryAdmissionTicket>, TestAdmissionController, TenantId) {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let ticket = QueryAdmissionTicket::new(controller.source(tenant_id.clone()), None, None);
    (ticket, controller, tenant_id)
}

fn insert_test_round(
    ticket: &Arc<QueryAdmissionTicket>,
    task_context: &Arc<TaskContext>,
    active_contexts: usize,
    epoch: u64,
) {
    let task_key = QueryAdmissionTicket::task_key(task_context);
    ticket.state.lock().unwrap().rounds.insert(
        task_key,
        TicketRound {
            _task_context: Arc::clone(task_context),
            handle: None,
            active_contexts,
            active_streams: 0,
            epoch,
            admission_facts_recorded: false,
            terminal_outcome: None,
            observation: ticket.new_observation(),
        },
    );
}

#[test]
fn admission_ticket_carries_requested_grant_bytes_for_admission() {
    let tenant_id = TenantId::new("tenant");
    let controller = TestAdmissionController::new(
        QueryServiceConfig {
            max_concurrent_queries: 1,
            device_capacity_bytes: Some(1024),
            max_admission_queue_len: 1,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), 256, 1024)],
    )
    .unwrap();
    let ticket = QueryAdmissionTicket::new(controller.source(tenant_id.clone()), Some(384), Some(512));

    assert_eq!(ticket.requested_grant_min_bytes(), Some(384));
    assert_eq!(ticket.requested_grant_bytes(), Some(512));
    assert_eq!(
        controller
            .requested_grant_band(
                &tenant_id,
                QueryGrantRequest {
                    requested_grant_min_bytes: ticket.requested_grant_min_bytes(),
                    requested_grant_bytes: ticket.requested_grant_bytes(),
                },
            )
            .unwrap()
            .want_bytes,
        512
    );
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_owner_precedes_enqueue_and_seals_rejection() {
    let configured_tenant = TenantId::new("configured");
    let controller = admission_only_controller(&configured_tenant);
    let owner =
        RuntimeObservationOwner::new("ticket-rejected-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let ticket = QueryAdmissionTicket::new_with_observation(
        controller.source(TenantId::new("unknown")),
        None,
        None,
        owner.clone(),
    );

    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
        .await
        .expect_err("unknown tenant rejects before admission");

    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::InvalidPlan);
    let snapshot = owner.sealed_snapshot().expect("admission rejection seals the ticket owner");
    assert_eq!(snapshot.manifest().query_id.as_str(), "ticket-rejected-query");
    assert!(matches!(
        snapshot.footer().terminal,
        Some(RuntimeTerminalOutcome::Failed { ref error_code, ref error_status })
            if error_code == nexus_query_engine::error::QueryErrorCode::InvalidPlan.as_str()
                && error_status == nexus_query_engine::error::QueryErrorCode::InvalidPlan.status().as_str()
    ));
    assert!(snapshot.events().iter().any(|event| matches!(
        &event.payload,
        nexus_query_engine::observability::RuntimeObservationPayload::Admission(observation)
            if matches!(
                observation.lifecycle,
                nexus_query_engine::observability::AdmissionLifecycleState::Rejected { .. }
            )
    )));
}

#[tokio::test(flavor = "current_thread")]
async fn coordinated_ticket_reuses_one_observation_session_across_three_legal_epochs() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let root =
        RuntimeObservationOwner::new("coordinated-ticket", None, RuntimeObservationConfig::bounded(64, 64 * 1024));
    let capture = NativeMetricsCapture::coordinated("coordinated-ticket", root.handle())
        .with_observation_activity_factory(Arc::new(|| Some(nexus_query_engine::runtime::QueryActivityLease::new(()))))
        .new_execution();
    assert_eq!(capture.retention_stats().capacity, 0);
    let ticket = QueryAdmissionTicket::new_with_observation_and_capture(
        controller.source(tenant_id),
        None,
        None,
        NativeRuntimeObservation::Coordinated(root.handle()),
        Some(capture.clone()),
    );
    let task_context = Arc::new(TaskContext::default());

    for _ in 0..3 {
        ticket
            .acquire_context(Arc::clone(&task_context), ExecutionCancellation::new())
            .await
            .expect_err("admission-only service has no runtime execution context");
        assert!(root.sealed_snapshot().is_none(), "admission epochs cannot seal the coordinator root");
    }
    drop(ticket);
    assert!(root.sealed_snapshot().is_none(), "ticket drop cannot seal the coordinator root");

    let snapshot = root.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap();
    assert_eq!(snapshot.manifest().query_id.as_str(), "coordinated-ticket");
    assert!(capture.runtime_observations().is_empty());
    assert_eq!(
        snapshot
            .events()
            .iter()
            .filter(|event| matches!(
                &event.payload,
                nexus_query_engine::observability::RuntimeObservationPayload::Admission(observation)
                    if matches!(
                        observation.lifecycle,
                        nexus_query_engine::observability::AdmissionLifecycleState::Admitted
                    )
            ))
            .count(),
        3
    );
}

#[tokio::test(flavor = "current_thread")]
async fn coordinated_activity_rejection_is_a_structured_admission_failure() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let root = RuntimeObservationOwner::new(
        "coordinated-activity-rejected",
        None,
        RuntimeObservationConfig::bounded(64, 64 * 1024),
    );
    let capture = NativeMetricsCapture::coordinated("coordinated-activity-rejected", root.handle())
        .with_observation_activity_factory(Arc::new(|| None))
        .new_execution();
    let ticket = QueryAdmissionTicket::new_with_observation_and_capture(
        controller.source(tenant_id),
        None,
        None,
        NativeRuntimeObservation::Coordinated(root.handle()),
        Some(capture),
    );

    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
        .await
        .expect_err("coordinator activity rejection must reject native admission");

    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::Cancelled);
    assert_eq!(error.operation(), "datafusion_nexus.native_exec.admission.observation_activity");
    assert_eq!(error.fact_value("reason"), Some("query_coordinator_activity_rejected"));
    assert_eq!(error.fact_value("admission_stage"), Some("observation_activity"));
    let snapshot = controller.admission_snapshot();
    assert_eq!(snapshot.active_queries, 0);
    assert_eq!(snapshot.queued_queries, 0);
    assert!(root.sealed_snapshot().is_none(), "coordinated rejection cannot seal the coordinator root");
    assert!(root.live_snapshot().unwrap().events().iter().any(|event| matches!(
        &event.payload,
        nexus_query_engine::observability::RuntimeObservationPayload::Admission(observation)
            if matches!(
                &observation.lifecycle,
                nexus_query_engine::observability::AdmissionLifecycleState::Rejected { reason }
                    if reason == "query_coordinator_activity_rejected"
            )
    )));
}

#[test]
fn standalone_ticket_drop_transfers_one_sealed_snapshot_to_its_local_collector() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let capture = NativeMetricsCapture::without_sink("standalone-ticket").new_execution();
    let observation = capture.new_runtime_observation("standalone-ticket", None);
    let ticket = QueryAdmissionTicket::new_with_observation_and_capture(
        controller.source(tenant_id),
        None,
        None,
        observation,
        Some(capture.clone()),
    );

    drop(ticket);

    let retained = capture.runtime_observations();
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].manifest().query_id.as_str(), "standalone-ticket");
    assert_eq!(retained[0].footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
    assert_eq!(capture.retention_stats().recorded_total, 1);
}

#[test]
fn dropping_ticket_before_first_round_seals_initial_owner_once_and_isolates_observer_panics() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let owner =
        RuntimeObservationOwner::new("ticket-dropped-before-round", None, RuntimeObservationConfig::bounded(16, 4096));
    let delivered = Arc::new(AtomicU64::new(0));
    let delivered_for_observer = Arc::clone(&delivered);
    owner.register_seal_observer(move |_| {
        delivered_for_observer.fetch_add(1, Ordering::Relaxed);
    });
    owner.register_seal_observer(|_| panic!("ticket drop must isolate observer failure"));
    let ticket = QueryAdmissionTicket::new_with_observation(controller.source(tenant_id), None, None, owner.clone());

    let dropped = catch_unwind(AssertUnwindSafe(|| drop(ticket)));

    assert!(dropped.is_ok());
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
    let snapshot = owner.sealed_snapshot().expect("last ticket owner drop must seal the initial observation");
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
    assert_eq!(delivered.load(Ordering::Relaxed), 1, "ticket terminalization must remain exactly once");
}

#[test]
fn unwinding_unadmitted_round_guard_seals_panicked_exactly_once() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let owner =
        RuntimeObservationOwner::new("unadmitted-round-panic", None, RuntimeObservationConfig::bounded(16, 4096));
    let delivered = Arc::new(AtomicU64::new(0));
    let delivered_for_observer = Arc::clone(&delivered);
    owner.register_seal_observer(move |_| {
        delivered_for_observer.fetch_add(1, Ordering::Relaxed);
    });
    let ticket = QueryAdmissionTicket::new_with_observation(controller.source(tenant_id), None, None, owner.clone());
    let guard = ticket.begin_execution_round(Arc::new(TaskContext::default())).unwrap();

    let unwind = catch_unwind(AssertUnwindSafe(move || {
        let _guard = guard;
        panic!("unwind unadmitted round");
    }));

    assert!(unwind.is_err());
    assert_eq!(owner.sealed_snapshot().unwrap().footer().terminal, Some(RuntimeTerminalOutcome::Panicked));
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
    assert!(ticket.state.lock().is_ok(), "round unwind must not poison ticket state");
    drop(ticket);
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
}

#[test]
fn unwinding_admit_reset_guard_seals_panicked_and_resets_without_poisoning() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let owner = RuntimeObservationOwner::new("admit-reset-panic", None, RuntimeObservationConfig::bounded(16, 4096));
    let delivered = Arc::new(AtomicU64::new(0));
    let delivered_for_observer = Arc::clone(&delivered);
    owner.register_seal_observer(move |_| {
        delivered_for_observer.fetch_add(1, Ordering::Relaxed);
    });
    let ticket = QueryAdmissionTicket::new_with_observation(controller.source(tenant_id), None, None, owner.clone());
    let task_key = 41;
    ticket.state.lock().unwrap().admitting.insert(task_key);
    let guard = TicketAdmitResetGuard {
        ticket: Arc::clone(&ticket),
        task_key,
        disarmed: false,
        observation: NativeRuntimeObservation::Local(owner.clone()),
    };

    let unwind = catch_unwind(AssertUnwindSafe(move || {
        let _guard = guard;
        panic!("unwind admit reset");
    }));

    assert!(unwind.is_err());
    assert_eq!(owner.sealed_snapshot().unwrap().footer().terminal, Some(RuntimeTerminalOutcome::Panicked));
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
    assert!(!ticket.state.lock().unwrap().admitting.contains(&task_key));
    drop(ticket);
    assert_eq!(delivered.load(Ordering::Relaxed), 1);
}

#[test]
fn cancellation_before_first_context_acquire_seals_initial_owner() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let owner = RuntimeObservationOwner::new(
        "ticket-cancelled-before-context",
        None,
        RuntimeObservationConfig::bounded(16, 4096),
    );
    let ticket = QueryAdmissionTicket::new_with_observation(controller.source(tenant_id), None, None, owner.clone());
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    let round = ticket.begin_execution_round(task_context).unwrap();
    let round_owner = round.observation_owner();
    ticket.state.lock().unwrap().rounds.get_mut(&task_key).unwrap().active_contexts = 1;
    let pending = QueryAdmissionPendingContextGuard {
        ticket: Arc::clone(&ticket),
        task_key,
        epoch: round.epoch(),
        released: false,
        observation: round_owner.into(),
    };

    drop(pending);
    drop(round);

    let snapshot = owner.sealed_snapshot().expect("cancelled first acquire must seal the initial owner");
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Cancelled));
}

#[test]
fn final_context_release_removes_round_without_waiting_for_task_context_drop() {
    let (ticket, _service, _tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    insert_test_round(&ticket, &task_context, 1, 7);

    ticket.release_epoch_context(task_key, 7, false);

    assert!(
        !ticket.state.lock().unwrap().rounds.contains_key(&task_key),
        "finished execution rounds must not linger on a live TaskContext"
    );
}

#[test]
fn overlapping_context_releases_keep_round_until_last_context_finishes() {
    let (ticket, _service, _tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    insert_test_round(&ticket, &task_context, 2, 11);

    ticket.release_epoch_context(task_key, 11, false);
    assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_contexts), Some(1));

    ticket.release_epoch_context(task_key, 11, false);
    assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
}

#[test]
fn cancelled_pending_context_acquire_releases_ticket_epoch_and_admission() {
    let (ticket, controller, tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    let handle = controller.admit_for_test(&tenant_id).unwrap();
    ticket.state.lock().unwrap().rounds.insert(
        task_key,
        TicketRound {
            _task_context: Arc::clone(&task_context),
            handle: Some(Arc::new(handle)),
            active_contexts: 1,
            active_streams: 0,
            epoch: 13,
            admission_facts_recorded: false,
            terminal_outcome: None,
            observation: ticket.new_observation(),
        },
    );

    let pending = QueryAdmissionPendingContextGuard {
        ticket: Arc::clone(&ticket),
        task_key,
        epoch: 13,
        released: false,
        observation: ticket.new_observation(),
    };
    drop(pending);

    assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
    assert_eq!(controller.admission_snapshot().active_queries, 0);
}

#[test]
fn execution_round_guard_keeps_round_until_stream_finishes() {
    let (ticket, _service, _tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    let round = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
    ticket.state.lock().unwrap().rounds.get_mut(&task_key).unwrap().active_contexts = 1;

    ticket.release_epoch_context(task_key, round.epoch(), false);
    assert_eq!(
        ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_streams),
        Some(1),
        "context release must not finish the execution round while the root stream is alive"
    );

    drop(round);
    assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
}

#[test]
fn active_execution_round_retains_task_context_until_release() {
    let (ticket, _service, _tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    let weak_task_context = Arc::downgrade(&task_context);
    let round = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();

    drop(task_context);

    assert!(
        weak_task_context.upgrade().is_some(),
        "active ticket rounds must retain the TaskContext whose pointer keys the round"
    );
    assert!(ticket.state.lock().unwrap().rounds.contains_key(&task_key));

    drop(round);

    assert!(weak_task_context.upgrade().is_none(), "finished ticket rounds should release their TaskContext");
}

#[test]
fn admission_wrapper_prefers_existing_ambient_ticket() {
    let (root_ticket, _root_service, _root_tenant_id) = admission_only_ticket();
    let (child_ticket, _child_service, _child_tenant_id) = admission_only_ticket();
    let task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&root_ticket));
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    let child = Arc::new(EmptyExec::new(Arc::new(Schema::empty()))) as Arc<dyn ExecutionPlan>;
    let wrapped = NativeAdmissionExec::new(child, Arc::clone(&child_ticket));

    let stream = wrapped.execute(0, Arc::clone(&task_context)).unwrap();

    assert!(root_ticket.state.lock().unwrap().rounds.contains_key(&task_key));
    assert!(
        child_ticket.state.lock().unwrap().rounds.is_empty(),
        "nested admission wrapper must not mint a second execution round when an ambient root ticket exists"
    );

    drop(stream);
    assert!(!root_ticket.state.lock().unwrap().rounds.contains_key(&task_key));
}

#[test]
fn admission_facts_marker_records_once_per_ticket_epoch() {
    let (ticket, _service, _tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    insert_test_round(&ticket, &task_context, 1, 3);

    assert!(ticket.mark_admission_facts_recorded(task_key, 3));
    assert!(
        !ticket.mark_admission_facts_recorded(task_key, 3),
        "the same execution epoch must not record query-level admission facts twice"
    );
    assert!(
        !ticket.mark_admission_facts_recorded(task_key, 4),
        "stale or future epochs must not consume the marker for the active round"
    );
}

#[test]
fn execution_round_reexecution_with_same_task_context_uses_fresh_epoch() {
    let (ticket, _service, _tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());

    let first = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
    assert_eq!(first.epoch(), 1);
    drop(first);

    let second = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
    assert_eq!(second.epoch(), 2);
}

#[test]
fn sequential_execution_epochs_use_distinct_owners_and_plan_invocations() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let initial_owner =
        RuntimeObservationOwner::new("stable-external-query", None, RuntimeObservationConfig::bounded(64, 64 * 1024));
    let ticket =
        QueryAdmissionTicket::new_with_observation(controller.source(tenant_id), None, None, initial_owner.clone());
    let task_context = Arc::new(TaskContext::default());

    let first_round = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
    let first_owner = first_round.observation_owner();
    let first_plan = nexus_query_engine::observability::RuntimePlanExecutionLifecycle::from_owner(
        first_owner.clone(),
        nexus_query_engine::observability::PlanExecutionId::new("epoch-plan-1"),
    );
    first_plan.finish(RuntimeTerminalOutcome::Succeeded);
    let first_snapshot = first_owner.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap();
    drop(first_round);

    let second_round = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
    let second_owner = second_round.observation_owner();
    assert!(second_owner.live_snapshot().is_some(), "the next epoch must not reuse the sealed first owner");
    let second_plan = nexus_query_engine::observability::RuntimePlanExecutionLifecycle::from_owner(
        second_owner.clone(),
        nexus_query_engine::observability::PlanExecutionId::new("epoch-plan-2"),
    );
    second_plan.finish(RuntimeTerminalOutcome::Succeeded);
    let second_snapshot = second_owner.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap();
    drop(second_round);

    assert_eq!(first_snapshot.manifest().query_id, second_snapshot.manifest().query_id);
    assert!(
        first_snapshot
            .events()
            .iter()
            .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "epoch-plan-1") })
    );
    assert!(
        second_snapshot
            .events()
            .iter()
            .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "epoch-plan-2") })
    );
    assert!(
        !second_snapshot
            .events()
            .iter()
            .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "epoch-plan-1") })
    );
}

#[test]
fn concurrent_execution_epochs_have_distinct_live_owners_with_shared_external_identity() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let ticket = QueryAdmissionTicket::new_with_observation(
        controller.source(tenant_id),
        None,
        None,
        RuntimeObservationOwner::new(
            "concurrent-external-query",
            None,
            RuntimeObservationConfig::bounded(64, 64 * 1024),
        ),
    );
    let first_round = ticket.begin_execution_round(Arc::new(TaskContext::default())).unwrap();
    let second_round = ticket.begin_execution_round(Arc::new(TaskContext::default())).unwrap();
    let first_owner = first_round.observation_owner();
    let second_owner = second_round.observation_owner();

    let first_plan = nexus_query_engine::observability::RuntimePlanExecutionLifecycle::from_owner(
        first_owner.clone(),
        nexus_query_engine::observability::PlanExecutionId::new("concurrent-plan-1"),
    );
    let second_plan = nexus_query_engine::observability::RuntimePlanExecutionLifecycle::from_owner(
        second_owner.clone(),
        nexus_query_engine::observability::PlanExecutionId::new("concurrent-plan-2"),
    );
    let first_live = first_owner.live_snapshot().unwrap();
    let second_live = second_owner.live_snapshot().unwrap();
    assert_eq!(first_live.manifest().query_id, second_live.manifest().query_id);
    assert!(
        first_live
            .events()
            .iter()
            .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "concurrent-plan-1") })
    );
    assert!(
        second_live
            .events()
            .iter()
            .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "concurrent-plan-2") })
    );
    assert!(first_owner.sealed_snapshot().is_none());
    assert!(second_owner.sealed_snapshot().is_none());

    first_plan.finish(RuntimeTerminalOutcome::Succeeded);
    second_plan.finish(RuntimeTerminalOutcome::Succeeded);
    let _ = first_owner.finish_query(RuntimeTerminalOutcome::Succeeded);
    let _ = second_owner.finish_query(RuntimeTerminalOutcome::Succeeded);
    drop(first_round);
    drop(second_round);
}

#[test]
fn admission_wrapper_repeated_execute_with_same_context_shares_one_round() {
    let (ticket, _service, _tenant_id) = admission_only_ticket();
    let input = Arc::new(EmptyExec::new(Arc::new(Schema::empty()))) as Arc<dyn ExecutionPlan>;
    let wrapped = NativeAdmissionExec::new(input, Arc::clone(&ticket));
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);

    let first = wrapped.execute(0, Arc::clone(&task_context)).unwrap();
    let second = wrapped.execute(0, Arc::clone(&task_context)).unwrap();

    {
        let state = ticket.state.lock().unwrap();
        assert_eq!(state.rounds.len(), 1, "one caller TaskContext must own one execution round");
        let round = state.rounds.get(&task_key).expect("round must be keyed by the caller TaskContext");
        assert_eq!(round.active_streams, 2);
    }

    drop(first);
    assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_streams), Some(1));

    drop(second);
    assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
}

#[tokio::test]
async fn admission_wrapper_dropped_sibling_overrides_shared_round_success() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let owner =
        RuntimeObservationOwner::new("wrapper-dropped-sibling", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let ticket = QueryAdmissionTicket::new_with_observation(controller.source(tenant_id), None, None, owner.clone());
    let input = Arc::new(EmptyExec::new(Arc::new(Schema::empty()))) as Arc<dyn ExecutionPlan>;
    let wrapped = NativeAdmissionExec::new(input, Arc::clone(&ticket));
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);

    let mut completed = wrapped.execute(0, Arc::clone(&task_context)).unwrap();
    let dropped = wrapped.execute(0, Arc::clone(&task_context)).unwrap();
    assert!(completed.next().await.is_none());
    assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_streams), Some(1));

    drop(dropped);

    assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
    assert_eq!(
        owner.sealed_snapshot().expect("the last sibling seals the shared round").footer().terminal,
        Some(RuntimeTerminalOutcome::Dropped),
    );
}

#[tokio::test]
async fn admission_wrapper_releases_round_when_stream_finishes() {
    let tenant_id = TenantId::new("tenant");
    let controller = admission_only_controller(&tenant_id);
    let owner = RuntimeObservationOwner::new("wrapper-success", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let ticket = QueryAdmissionTicket::new_with_observation(controller.source(tenant_id), None, None, owner.clone());
    let input = Arc::new(EmptyExec::new(Arc::new(Schema::empty()))) as Arc<dyn ExecutionPlan>;
    let wrapped = Arc::new(NativeAdmissionExec::new(input, Arc::clone(&ticket))) as Arc<dyn ExecutionPlan>;
    assert!(plan_is_query_admission_wrapped(wrapped.as_ref()));
    assert!(wrapped.downcast_ref::<EmptyExec>().is_some());
    let rewrapped = wrap_plan_with_query_admission_ticket(Arc::clone(&wrapped), Arc::clone(&ticket));
    assert!(Arc::ptr_eq(&wrapped, &rewrapped));
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);

    let mut stream = wrapped.execute(0, Arc::clone(&task_context)).unwrap();
    assert_eq!(wrapped.name(), "EmptyExec");
    assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_streams), Some(1));

    assert!(stream.next().await.is_none());
    assert!(
        !ticket.state.lock().unwrap().rounds.contains_key(&task_key),
        "terminal stream poll should release the root execution-round guard"
    );
    assert_eq!(
        owner.sealed_snapshot().expect("wrapper terminal poll seals the invocation").footer().terminal,
        Some(RuntimeTerminalOutcome::Succeeded),
    );
}

#[test]
fn stale_epoch_release_does_not_remove_newer_round() {
    let (ticket, _service, _tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    insert_test_round(&ticket, &task_context, 1, 2);

    ticket.release_epoch_context(task_key, 1, false);

    assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.epoch), Some(2));
}

#[tokio::test]
async fn active_same_task_round_reuses_existing_handle_instead_of_re_admitting() {
    let (ticket, controller, tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());
    let task_key = QueryAdmissionTicket::task_key(&task_context);
    let handle = controller.admit_for_test(&tenant_id).unwrap();
    ticket.state.lock().unwrap().rounds.insert(
        task_key,
        TicketRound {
            _task_context: Arc::clone(&task_context),
            handle: Some(Arc::new(handle)),
            active_contexts: 1,
            active_streams: 0,
            epoch: 1,
            admission_facts_recorded: false,
            terminal_outcome: None,
            observation: ticket.new_observation(),
        },
    );
    assert_eq!(controller.admission_snapshot().active_queries, 1);

    let acquire = tokio::time::timeout(
        Duration::from_millis(100),
        ticket.acquire_context(Arc::clone(&task_context), ExecutionCancellation::new()),
    )
    .await
    .expect("overlapping acquisition should reuse the active round instead of waiting for another admission");
    assert!(acquire.is_err(), "admission-only services cannot acquire runtime contexts");
    assert_eq!(
        controller.admission_snapshot().active_queries,
        1,
        "failed overlapping context acquisition must not drop the original active round"
    );

    ticket.release_epoch_context(task_key, 1, false);
    assert_eq!(controller.admission_snapshot().active_queries, 0);
}

#[tokio::test]
async fn failed_context_acquire_releases_ticket_epoch_for_reexecution() {
    let (ticket, controller, _tenant_id) = admission_only_ticket();
    let task_context = Arc::new(TaskContext::default());

    let first = ticket.acquire_context(Arc::clone(&task_context), ExecutionCancellation::new()).await;
    assert!(first.is_err(), "admission-only services cannot acquire runtime contexts");
    assert_eq!(controller.admission_snapshot().active_queries, 0);

    let second = tokio::time::timeout(
        Duration::from_millis(100),
        ticket.acquire_context(Arc::clone(&task_context), ExecutionCancellation::new()),
    )
    .await
    .expect("second execution round should not wait on the failed first round");
    assert!(second.is_err(), "admission-only services cannot acquire runtime contexts");
    assert_eq!(controller.admission_snapshot().active_queries, 0);
}
