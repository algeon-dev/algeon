use nexus_query_engine::{
    MemoryReservationEventAction, MemoryReservationEventOutcome,
    exec::reservation_contract_testing::{
        probe_alias_contract, probe_alias_outlives_source_release_contract, probe_denial_capacity_after_alias,
        probe_join_materialization_txn_rollback_contract, probe_source_cache_alias_relabel_contract,
        probe_transfer_contract,
    },
};

#[test]
fn reservation_tracker_alias_token_does_not_inflate_live_ledger() {
    let probe = probe_alias_contract().expect("alias contract probe should succeed");

    assert_eq!(probe.reserved_after_reserve, 60);
    assert_eq!(probe.reserved_after_alias_or_transfer, 60);
    assert_eq!(probe.reserved_after_intermediate_release, 60);
    assert_eq!(probe.reserved_after_final_release, 0);

    let actions = probe.metrics.reservation_events.iter().map(|event| event.action).collect::<Vec<_>>();
    assert_eq!(
        actions,
        [
            MemoryReservationEventAction::Reserve,
            MemoryReservationEventAction::AliasAcquire,
            MemoryReservationEventAction::AliasRelease,
            MemoryReservationEventAction::Release,
        ]
    );
    let deltas = probe.metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).collect::<Vec<_>>();
    assert_eq!(deltas, [60, 0, 0, -60]);
    let current = probe.metrics.reservation_events.iter().map(|event| event.current_reserved_bytes).collect::<Vec<_>>();
    assert_eq!(current, [60, 60, 60, 0]);
    assert_eq!(probe.metrics.reservation_requested_bytes, 60);
    assert_eq!(probe.metrics.reservation_granted_bytes, 60);
    assert_eq!(probe.metrics.reservation_live_peak_bytes, 60);
    assert_eq!(probe.metrics.reservation_denied_count, 0);
}

#[test]
fn reservation_tracker_transfer_relabels_without_changing_live_ledger() {
    let probe = probe_transfer_contract().expect("transfer contract probe should succeed");

    assert_eq!(probe.reserved_after_reserve, 60);
    assert_eq!(probe.reserved_after_alias_or_transfer, 60);
    assert_eq!(probe.reserved_after_intermediate_release, 60);
    assert_eq!(probe.reserved_after_final_release, 0);

    let actions = probe.metrics.reservation_events.iter().map(|event| event.action).collect::<Vec<_>>();
    assert_eq!(
        actions,
        [
            MemoryReservationEventAction::Reserve,
            MemoryReservationEventAction::Transfer,
            MemoryReservationEventAction::Release,
        ]
    );
    let boundaries =
        probe.metrics.reservation_events.iter().map(|event| event.memory_boundary.as_str()).collect::<Vec<_>>();
    assert_eq!(boundaries, ["join_input_projection", "join_probe_side", "join_probe_side"]);
    let deltas = probe.metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).collect::<Vec<_>>();
    assert_eq!(deltas, [60, 0, -60]);
    let current = probe.metrics.reservation_events.iter().map(|event| event.current_reserved_bytes).collect::<Vec<_>>();
    assert_eq!(current, [60, 60, 0]);
    assert_eq!(probe.metrics.reservation_requested_bytes, 60);
    assert_eq!(probe.metrics.reservation_granted_bytes, 60);
    assert_eq!(probe.metrics.reservation_live_peak_bytes, 60);
}

#[test]
fn reservation_tracker_alias_keeps_charge_live_after_source_release() {
    let probe = probe_alias_outlives_source_release_contract().expect("deferred alias release probe should succeed");

    assert_eq!(probe.reserved_after_reserve, 60);
    assert_eq!(probe.reserved_after_alias_or_transfer, 60);
    assert_eq!(probe.reserved_after_intermediate_release, 60);
    assert_eq!(probe.reserved_after_final_release, 0);

    let actions = probe.metrics.reservation_events.iter().map(|event| event.action).collect::<Vec<_>>();
    assert_eq!(
        actions,
        [
            MemoryReservationEventAction::Reserve,
            MemoryReservationEventAction::AliasAcquire,
            MemoryReservationEventAction::Release,
            MemoryReservationEventAction::AliasRelease,
        ]
    );
    let deltas = probe.metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).collect::<Vec<_>>();
    assert_eq!(deltas, [60, 0, 0, -60]);
    let current = probe.metrics.reservation_events.iter().map(|event| event.current_reserved_bytes).collect::<Vec<_>>();
    assert_eq!(current, [60, 60, 60, 0]);
    assert_eq!(probe.metrics.reservation_requested_bytes, 60);
    assert_eq!(probe.metrics.reservation_granted_bytes, 60);
    assert_eq!(probe.metrics.reservation_live_peak_bytes, 60);
}

#[test]
fn reservation_tracker_denial_capacity_ignores_alias_tokens() {
    let probe = probe_denial_capacity_after_alias().expect("denial capacity probe should run");

    assert_eq!(probe.reserved_after_denial, 70);
    assert_eq!(probe.error.fact_value("granted_bytes"), Some("30"));
    assert_eq!(probe.error.fact_value("requested_bytes"), Some("40"));
    assert_eq!(probe.error.fact_value("memory_boundary"), Some("join_probe_side"));
    assert_eq!(probe.metrics.reservation_denied_count, 1);
    assert_eq!(probe.metrics.reservation_requested_bytes, 110);
    assert_eq!(probe.metrics.reservation_granted_bytes, 70);
    assert_eq!(probe.metrics.reservation_live_peak_bytes, 70);

    let denied = probe
        .metrics
        .reservation_events
        .iter()
        .find(|event| event.outcome == MemoryReservationEventOutcome::Denied)
        .expect("probe should record a denial event");
    assert_eq!(denied.current_reserved_bytes, 70);
    assert_eq!(denied.granted_bytes, 30);
    assert_eq!(denied.charged_delta_bytes, 0);
}

#[test]
fn join_materialization_txn_rollback_records_release_events() {
    let probe = probe_join_materialization_txn_rollback_contract().expect("rollback probe should run");

    assert_eq!(probe.reserved_after_rollback, 0);
    let events = &probe.metrics.reservation_events;
    let actions = events.iter().map(|event| event.action).collect::<Vec<_>>();
    assert_eq!(
        actions,
        [
            MemoryReservationEventAction::Reserve,
            MemoryReservationEventAction::Reserve,
            MemoryReservationEventAction::Release,
            MemoryReservationEventAction::Release,
        ]
    );
    let boundaries = events.iter().map(|event| event.memory_boundary.as_str()).collect::<Vec<_>>();
    assert_eq!(boundaries, ["join_one_shot_indices", "join_output", "join_output", "join_one_shot_indices"]);
    let deltas = events.iter().map(|event| event.charged_delta_bytes).collect::<Vec<_>>();
    assert_eq!(deltas, [30, 40, -40, -30]);
    let current = events.iter().map(|event| event.current_reserved_bytes).collect::<Vec<_>>();
    assert_eq!(current, [30, 70, 30, 0]);

    let task = probe.metrics.task_runtime_metrics.get("t4").expect("rollback releases should be task-attributed");
    assert_eq!(task.operator_kind, "join_materialize");
    assert_eq!(task.reservation_released_bytes, 70);
    let operator =
        probe.metrics.task_operator_metrics.get("join_materialize").expect("operator aggregate should be recorded");
    assert_eq!(operator.reservation_released_bytes, 70);
}

#[test]
fn source_cache_alias_transfer_relabels_join_boundary_without_double_counting() {
    let probe = probe_source_cache_alias_relabel_contract().expect("source cache alias relabel probe should run");

    assert_eq!(probe.reserved_after_cache_insert, 0);
    assert_eq!(probe.reserved_after_borrow, 0);
    assert_eq!(probe.reserved_after_relabel, 0);
    assert_eq!(probe.reserved_after_alias_release, 0);
    assert_eq!(probe.reserved_after_cache_release, 0);
    assert_eq!(probe.protected_after_borrow, 1);
    assert_eq!(probe.protected_after_relabel, 1);
    assert_eq!(probe.protected_after_alias_release, 0);

    let actions = probe.metrics.reservation_events.iter().map(|event| event.action).collect::<Vec<_>>();
    assert_eq!(
        actions,
        [
            MemoryReservationEventAction::Reserve,
            MemoryReservationEventAction::Reserve,
            MemoryReservationEventAction::Release,
            MemoryReservationEventAction::AliasAcquire,
            MemoryReservationEventAction::AliasRelabel,
            MemoryReservationEventAction::AliasRelease,
            MemoryReservationEventAction::Release,
        ]
    );
    let boundaries =
        probe.metrics.reservation_events.iter().map(|event| event.memory_boundary.as_str()).collect::<Vec<_>>();
    assert_eq!(
        boundaries,
        [
            "join_source_read",
            "source_cache_retained",
            "join_source_read",
            "source_cache_borrowed",
            "join_probe_side",
            "join_probe_side",
            "source_cache_retained",
        ]
    );
    let deltas = probe.metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).collect::<Vec<_>>();
    assert_eq!(deltas, [60, 0, 0, 0, 0, 0, -60]);
    let current = probe.metrics.reservation_events.iter().map(|event| event.current_reserved_bytes).collect::<Vec<_>>();
    assert_eq!(current, [60, 60, 60, 60, 60, 60, 0]);
    assert_eq!(probe.metrics.reservation_requested_bytes, 120);
    assert_eq!(probe.metrics.reservation_granted_bytes, 120);
    assert_eq!(probe.metrics.reservation_live_peak_bytes, 60);
    assert_eq!(probe.metrics.reservation_denied_count, 0);
}
