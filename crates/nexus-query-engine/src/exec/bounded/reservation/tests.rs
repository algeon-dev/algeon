use arrow::datatypes::{DataType, Field, Schema};

use super::{tracker::ReservationBudget, *};
use crate::{
    observability::{
        AdmissionRequestId, PlanExecutionId, RuntimeObservationConfig, RuntimeObservationLossReason,
        RuntimeObservationPayload, RuntimeObservationSession, RuntimeQueryLifecycle, RuntimeReservationAction,
        RuntimeReservationDisposition, RuntimeTerminalOutcome, write_runtime_observation_jsonl,
    },
    runtime::{GrantRequest, MemoryGovernor},
    source::LocalParquetSource,
};

fn test_request(kind: ReservationKind, boundary: &'static str, bytes: u64, budget: u64) -> MemoryReservationRequest {
    MemoryReservationRequest::new(kind, bytes, budget).with_memory_boundary(boundary).with_stage("test")
}

fn observed_projection_with_limits(
    config: RuntimeObservationConfig,
) -> Result<(QueryMetrics, crate::observability::SealedRuntimeObservationSnapshot)> {
    let lifecycle = RuntimeQueryLifecycle::new(
        RuntimeObservationSession::new("reservation-cap-parity", None, config),
        PlanExecutionId::new("plan-reservation-cap-parity"),
    );
    let scope = lifecycle.execution_scope();
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget_and_observation(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::Static(100),
        Some(&scope),
    );
    let request = |kind, boundary, bytes| {
        test_request(kind, boundary, bytes, 100).with_node_id("cap-node").with_task_registration("cap-task", 1, "unary")
    };
    let mut owned =
        reservations.try_reserve_work(request(ReservationKind::OperatorOutput, "reserve", 60), &mut metrics)?;
    reservations.try_grow(&mut owned, request(ReservationKind::OperatorOutput, "grow", 10), &mut metrics)?;
    reservations.shrink_to_observed(&mut owned, 50, &mut metrics);
    let denied = reservations.try_reserve_work(request(ReservationKind::OperatorOutput, "denied", 60), &mut metrics);
    assert!(denied.is_err());
    drop(owned);
    drop(reservations);
    metrics.finalize_reservation_projection();
    let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
    Ok((metrics, snapshot))
}

fn assert_exact_reservation_projection(actual: &QueryMetrics, expected: &QueryMetrics) {
    assert_eq!(actual.reservation_requested_bytes, expected.reservation_requested_bytes);
    assert_eq!(actual.reservation_granted_bytes, expected.reservation_granted_bytes);
    assert_eq!(actual.reservation_live_peak_bytes, expected.reservation_live_peak_bytes);
    assert_eq!(actual.reservation_denied_count, expected.reservation_denied_count);
    assert_eq!(actual.max_task_reservation_bytes, expected.max_task_reservation_bytes);
    assert_eq!(actual.reservation_events, expected.reservation_events);
    assert_eq!(actual.task_runtime_metrics, expected.task_runtime_metrics);
    assert_eq!(actual.task_operator_metrics, expected.task_operator_metrics);
}

#[test]
fn public_reservation_projection_is_exact_across_trace_record_and_byte_caps() -> Result<()> {
    let (uncapped, complete_trace) = observed_projection_with_limits(RuntimeObservationConfig::bounded(64, 64 * 1024))?;
    assert_eq!(uncapped.reservation_requested_bytes, 130);
    assert_eq!(uncapped.reservation_granted_bytes, 70);
    assert_eq!(uncapped.reservation_live_peak_bytes, 70);
    assert_eq!(uncapped.reservation_denied_count, 1);
    assert_eq!(uncapped.reservation_events.len(), 5);
    let task = &uncapped.task_runtime_metrics["cap-task"];
    assert_eq!(task.reservation_requested_bytes, 130);
    assert_eq!(task.reservation_granted_bytes, 70);
    assert_eq!(task.reservation_released_bytes, 70);
    assert_eq!(task.reservation_live_peak_bytes, 70);
    assert!(!complete_trace.footer().truncated);
    assert!(!complete_trace.footer().degraded, "exact reservation preflight must match retained accounting");

    for config in [
        RuntimeObservationConfig::bounded(0, 64 * 1024),
        RuntimeObservationConfig::bounded(1, 64 * 1024),
        RuntimeObservationConfig::bounded(64, 1),
        RuntimeObservationConfig::bounded(64, 512),
    ] {
        let (capped, trace) = observed_projection_with_limits(config)?;
        assert_exact_reservation_projection(&capped, &uncapped);
        assert!(trace.footer().truncated);
        assert!(trace.events().len() < complete_trace.events().len());
        assert!(trace.footer().first_dropped_sequence.is_some());
    }
    let (_, record_limited) = observed_projection_with_limits(RuntimeObservationConfig::bounded(0, 64 * 1024))?;
    assert!(
        record_limited
            .footer()
            .dropped_by_reason
            .iter()
            .any(|loss| loss.reason == RuntimeObservationLossReason::RecordLimit)
    );
    let (_, oversized) = observed_projection_with_limits(RuntimeObservationConfig::bounded(64, 1))?;
    assert!(
        oversized.footer().dropped_by_reason.iter().any(|loss| loss.reason == RuntimeObservationLossReason::Oversized)
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum TestReleaseMode {
    Explicit,
    Drop,
    Unwind,
}

fn projection_for_release_mode(mode: TestReleaseMode) -> Result<QueryMetrics> {
    let mut metrics = QueryMetrics::new("release-projection-parity");
    metrics.record_task_execution("release-task", "unary");
    metrics.record_task_elapsed("release-task", "unary", std::time::Duration::from_millis(7));
    let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
    let active = reservations.try_reserve_work(
        test_request(ReservationKind::OperatorOutput, "release-parity", 60, 100).with_task_registration(
            "release-task",
            1,
            "unary",
        ),
        &mut metrics,
    )?;
    match mode {
        TestReleaseMode::Explicit => reservations.release(active, &mut metrics),
        TestReleaseMode::Drop => drop(active),
        TestReleaseMode::Unwind => {
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _active = active;
                panic!("exercise reservation unwind cleanup");
            }));
            assert!(unwind.is_err());
        },
    }
    assert_eq!(reservations.reserved_bytes(), 0);
    drop(reservations);
    metrics.finalize_reservation_projection();
    Ok(metrics)
}

#[test]
fn disabled_observation_drop_and_unwind_keep_the_exact_public_projection() -> Result<()> {
    let explicit = projection_for_release_mode(TestReleaseMode::Explicit)?;
    let dropped = projection_for_release_mode(TestReleaseMode::Drop)?;
    let unwound = projection_for_release_mode(TestReleaseMode::Unwind)?;

    assert_exact_reservation_projection(&dropped, &explicit);
    assert_exact_reservation_projection(&unwound, &explicit);
    assert_eq!(explicit.reservation_events.len(), 2);
    assert_eq!(explicit.reservation_events.iter().map(|event| event.charged_delta_bytes).sum::<i128>(), 0);
    let task = &explicit.task_runtime_metrics["release-task"];
    assert_eq!(task.elapsed_time, std::time::Duration::from_millis(7));
    assert_eq!(task.reservation_requested_bytes, 60);
    assert_eq!(task.reservation_granted_bytes, 60);
    assert_eq!(task.reservation_released_bytes, 60);
    assert_eq!(task.live_reserved_bytes, 0);
    assert_eq!(explicit.task_operator_metrics["unary"].task_count, 1);
    Ok(())
}

#[test]
fn cloning_query_metrics_detaches_the_reservation_projection_ledger() -> Result<()> {
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
    let active = reservations
        .try_reserve_work(test_request(ReservationKind::OperatorOutput, "clone-ledger", 60, 100), &mut metrics)?;
    let mut cloned = metrics.clone();

    drop(active);
    metrics.finalize_reservation_projection();
    cloned.finalize_reservation_projection();

    assert_eq!(metrics.reservation_events.len(), 2);
    assert_eq!(metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).sum::<i128>(), 0);
    assert_eq!(cloned.reservation_events.len(), 1);
    assert_eq!(cloned.reservation_events[0].charged_delta_bytes, 60);
    Ok(())
}

#[test]
fn adaptive_reservation_tracker_observes_live_grant_growth() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 40, 40))?.expect("initial grant should fit");
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );

    let first = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "first", 30, reservations.budget()),
        &mut metrics,
    )?;
    let denied = match reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "before_growth", 20, reservations.budget()),
        &mut metrics,
    ) {
        Ok(_) => panic!("only 10 bytes should remain before grant growth"),
        Err(error) => error,
    };
    assert_eq!(denied.fact_value("granted_bytes"), Some("10"));

    assert!(grant.try_grow_to(70)?);
    assert_eq!(reservations.budget(), 70);
    assert_eq!(reservations.remaining_bytes(), 40);

    let second = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "after_growth", 20, reservations.budget()),
        &mut metrics,
    )?;

    let granted = metrics
        .reservation_events
        .iter()
        .rev()
        .find(|event| {
            event.outcome == MemoryReservationEventOutcome::Granted
                && event.action == MemoryReservationEventAction::Reserve
                && event.memory_boundary == "after_growth"
        })
        .expect("post-growth reservation should be recorded");
    assert_eq!(granted.budget_bytes, 70);

    reservations.release(second, &mut metrics);
    reservations.release(first, &mut metrics);
    Ok(())
}

#[test]
fn static_reservation_tracker_ignores_later_grant_growth() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(8), 40, 40))?.expect("initial grant should fit");
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, grant.budget_bytes());

    let first = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "first", 30, reservations.budget()),
        &mut metrics,
    )?;
    assert!(grant.try_grow_to(70)?);
    assert_eq!(grant.budget_bytes(), 70);
    assert_eq!(reservations.budget(), 40);
    assert_eq!(reservations.remaining_bytes(), 10);

    let denied = match reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "after_external_growth", 20, reservations.budget()),
        &mut metrics,
    ) {
        Ok(_) => panic!("static reservation tracker must keep the original budget"),
        Err(error) => error,
    };
    assert_eq!(denied.fact_value("granted_bytes"), Some("10"));

    reservations.release(first, &mut metrics);
    Ok(())
}

#[test]
fn static_reservation_tracker_acknowledges_grant_shrink_as_noop() -> Result<()> {
    let reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);

    assert!(reservations.acknowledge_requested_grant_shrink()?);
    assert_eq!(reservations.budget(), 100);
    Ok(())
}

#[test]
fn grant_backed_static_tracker_retains_live_output_bytes() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 10, 40))?.expect("initial grant should fit");
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::GrantBackedStatic { budget_bytes: 40, grant: grant.clone() },
    );

    let active = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "strict_output", 30, reservations.budget()),
        &mut metrics,
    )?;
    assert_eq!(grant.live_reserved_bytes(), 30);

    let retained = active.retain_grant_live_reservation().expect("strict grant should retain live output bytes");
    drop(active);

    assert_eq!(grant.live_reserved_bytes(), 30);
    assert!(grant.try_shrink_to(0)?);
    assert_eq!(grant.budget_bytes(), 30);

    drop(retained);
    assert_eq!(grant.live_reserved_bytes(), 0);
    assert!(grant.try_shrink_to(0)?);
    assert_eq!(grant.budget_bytes(), 10);
    Ok(())
}

#[test]
fn adaptive_reservation_tracker_acknowledges_shrink_without_underflowing_live_reservations() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 80))?.expect("initial grant should fit");
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );
    let live = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "live", 55, reservations.budget()),
        &mut metrics,
    )?;

    assert_eq!(grant.request_shrink_to(40)?, Some(40));
    assert!(!reservations.acknowledge_requested_grant_shrink()?);

    assert_eq!(reservations.reserved_bytes(), 55);
    assert_eq!(grant.budget_bytes(), 55);
    assert_eq!(grant.requested_shrink_target_bytes()?, Some(40));
    assert_eq!(governor.available_bytes()?, 45);

    reservations.release(live, &mut metrics);
    Ok(())
}

#[test]
fn adaptive_shrink_ack_uses_grant_scoped_live_bytes_across_trackers() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 90))?.expect("initial grant should fit");
    let mut metrics = QueryMetrics::default();
    let mut first_tracker = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );
    let mut second_tracker = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );
    let first_live = first_tracker.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "first_live", 60, first_tracker.budget()),
        &mut metrics,
    )?;
    let second_live = second_tracker.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "second_live", 20, second_tracker.budget()),
        &mut metrics,
    )?;

    assert_eq!(grant.live_reserved_bytes(), 80);
    assert_eq!(grant.request_shrink_to(40)?, Some(40));
    assert!(!second_tracker.acknowledge_requested_grant_shrink()?);

    assert_eq!(second_tracker.raw_reserved_bytes(), 20);
    assert_eq!(grant.budget_bytes(), 80);
    assert_eq!(grant.requested_shrink_target_bytes()?, Some(40));
    assert_eq!(governor.available_bytes()?, 20);

    first_tracker.release(first_live, &mut metrics);
    assert!(second_tracker.acknowledge_requested_grant_shrink()?);
    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(grant.requested_shrink_target_bytes()?, None);
    assert_eq!(governor.available_bytes()?, 60);

    second_tracker.release(second_live, &mut metrics);
    assert_eq!(grant.live_reserved_bytes(), 0);
    Ok(())
}

#[test]
fn adaptive_reservation_tracker_denies_when_shared_grant_live_budget_is_exhausted() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 70))?.expect("initial grant should fit");
    let mut metrics = QueryMetrics::default();
    let mut first_tracker = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );
    let mut second_tracker = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );
    let first_live = first_tracker.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "first_live", 50, first_tracker.budget()),
        &mut metrics,
    )?;

    let denied = match second_tracker.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "second_live", 30, second_tracker.budget()),
        &mut metrics,
    ) {
        Ok(_) => panic!("second tracker must be capped by the shared grant live footprint"),
        Err(error) => error,
    };

    assert_eq!(denied.fact_value("granted_bytes"), Some("20"));
    assert_eq!(grant.live_reserved_bytes(), 50);

    first_tracker.release(first_live, &mut metrics);
    assert_eq!(grant.live_reserved_bytes(), 0);
    Ok(())
}

#[test]
fn adaptive_reservation_tracker_borrowed_resident_does_not_double_charge_grant_live_bytes() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 70))?.expect("initial grant should fit");
    let retained_input = grant.retain_live_reservation(50).expect("input handle should retain live bytes");
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );

    let borrowed = reservations.borrow_resident(
        test_request(ReservationKind::SourceRead, "gpu_resident_source", 50, reservations.budget()),
        &mut metrics,
    );

    assert_eq!(reservations.raw_reserved_bytes(), 0);
    assert_eq!(borrowed.bytes(), 50);
    assert_eq!(borrowed.charged_bytes(), 0);
    assert_eq!(grant.live_reserved_bytes(), 50);

    let work = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "consumer_work", 20, reservations.budget()),
        &mut metrics,
    )?;
    assert_eq!(grant.live_reserved_bytes(), 70);

    reservations.release(borrowed, &mut metrics);
    assert_eq!(grant.live_reserved_bytes(), 70);
    reservations.release(work, &mut metrics);
    assert_eq!(grant.live_reserved_bytes(), 50);
    drop(retained_input);
    assert_eq!(grant.live_reserved_bytes(), 0);
    Ok(())
}

#[test]
fn alias_reservation_over_borrowed_resident_releases_without_charging() {
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);

    let borrowed = reservations.borrow_resident(
        test_request(ReservationKind::SourceRead, "gpu_resident_source", 40, reservations.budget()),
        &mut metrics,
    );
    let lineage = Rc::clone(borrowed.lineage.as_ref().expect("borrowed resident should carry lineage"));
    let alias = reservations.alias_reservation(
        &borrowed,
        test_request(ReservationKind::OperatorOutput, "borrowed_alias", 40, reservations.budget()),
        &mut metrics,
    );

    assert!(alias.is_alias());
    assert_eq!(alias.charged_bytes(), 0);
    assert_eq!(reservations.reserved_bytes(), 0);
    assert_eq!(reservations.raw_reserved_bytes(), 0);
    assert_eq!(lineage.alias_count.get(), 1);

    reservations.release(alias, &mut metrics);
    assert_eq!(lineage.alias_count.get(), 0);
    assert_eq!(lineage.deferred_charged_release_bytes.get(), 0);
    assert_eq!(reservations.reserved_bytes(), 0);

    reservations.release(borrowed, &mut metrics);
    assert_eq!(lineage.deferred_charged_release_bytes.get(), 0);
    assert_eq!(reservations.reserved_bytes(), 0);
    assert_eq!(reservations.raw_reserved_bytes(), 0);
}

#[test]
fn adaptive_reservation_tracker_completes_shrink_after_live_bytes_drop() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 80))?.expect("initial grant should fit");
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );
    let live = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "live", 55, reservations.budget()),
        &mut metrics,
    )?;

    assert_eq!(grant.request_shrink_to(40)?, Some(40));
    assert!(!reservations.acknowledge_requested_grant_shrink()?);
    reservations.release(live, &mut metrics);
    assert!(reservations.acknowledge_requested_grant_shrink()?);

    assert_eq!(reservations.reserved_bytes(), 0);
    assert_eq!(grant.budget_bytes(), 40);
    assert_eq!(grant.requested_shrink_target_bytes()?, None);
    assert_eq!(governor.available_bytes()?, 60);
    Ok(())
}

#[test]
fn adaptive_reservation_tracker_shrink_ack_counts_source_cache_baseline() -> Result<()> {
    let governor = MemoryGovernor::new(0, 100)?;
    let grant =
        governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 80))?.expect("initial grant should fit");
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::AdaptiveGrant(grant.clone()),
    );

    reservations.try_reserve_source_cache_baseline(55, &mut metrics)?;
    assert_eq!(reservations.reserved_bytes(), 0);
    assert_eq!(reservations.raw_reserved_bytes(), 55);
    assert_eq!(grant.request_shrink_to(40)?, Some(40));

    assert!(!reservations.acknowledge_requested_grant_shrink()?);

    assert_eq!(grant.budget_bytes(), 55);
    assert_eq!(grant.requested_shrink_target_bytes()?, Some(40));
    assert_eq!(governor.available_bytes()?, 45);
    Ok(())
}

#[test]
fn reconcile_work_reservation_to_output_grows_with_output_request_context() -> Result<()> {
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
    let mut work = reservations.try_reserve_work(
        test_request(ReservationKind::OperatorOutput, "operator_work", 40, reservations.budget()),
        &mut metrics,
    )?;
    let output_request = test_request(ReservationKind::OperatorOutput, "operator_output", 60, reservations.budget());

    reconcile_work_reservation_to_output(&mut reservations, &mut work, 60, &output_request, &mut metrics)?;

    assert_eq!(work.bytes(), 60);
    assert_eq!(reservations.reserved_bytes(), 60);
    let actions = metrics.reservation_events.iter().map(|event| event.action).collect::<Vec<_>>();
    assert_eq!(actions, [MemoryReservationEventAction::Reserve, MemoryReservationEventAction::Grow]);
    let boundaries = metrics.reservation_events.iter().map(|event| event.memory_boundary.as_str()).collect::<Vec<_>>();
    assert_eq!(boundaries, ["operator_work", "operator_output"]);

    let output = reservations.transfer(work, output_request, &mut metrics);
    reservations.release(output, &mut metrics);
    Ok(())
}

#[test]
fn reconcile_work_reservation_to_output_shrinks_with_output_request_context() -> Result<()> {
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
    let mut work = reservations.try_reserve_work(
        test_request(ReservationKind::OperatorOutput, "operator_work", 60, reservations.budget()),
        &mut metrics,
    )?;
    let output_request = test_request(ReservationKind::OperatorOutput, "operator_output", 40, reservations.budget());

    reconcile_work_reservation_to_output(&mut reservations, &mut work, 40, &output_request, &mut metrics)?;

    assert_eq!(work.bytes(), 40);
    assert_eq!(reservations.reserved_bytes(), 40);
    let actions = metrics.reservation_events.iter().map(|event| event.action).collect::<Vec<_>>();
    assert_eq!(actions, [MemoryReservationEventAction::Reserve, MemoryReservationEventAction::Shrink]);
    let deltas = metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).collect::<Vec<_>>();
    assert_eq!(deltas, [60, -20]);
    let boundaries = metrics.reservation_events.iter().map(|event| event.memory_boundary.as_str()).collect::<Vec<_>>();
    assert_eq!(boundaries, ["operator_work", "operator_output"]);

    let output = reservations.transfer(work, output_request, &mut metrics);
    reservations.release(output, &mut metrics);
    Ok(())
}

#[test]
fn reconcile_work_reservation_to_output_denial_leaves_release_to_caller() -> Result<()> {
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 50);
    let mut work = reservations.try_reserve_work(
        test_request(ReservationKind::OperatorOutput, "operator_work", 40, reservations.budget()),
        &mut metrics,
    )?;
    let output_request = test_request(ReservationKind::OperatorOutput, "operator_output", 60, reservations.budget());

    let error = reconcile_work_reservation_to_output(&mut reservations, &mut work, 60, &output_request, &mut metrics)
        .expect_err("grow beyond budget should be denied");

    assert_eq!(error.fact_value("memory_boundary"), Some("operator_output"));
    assert_eq!(error.fact_value("requested_bytes"), Some("20"));
    assert_eq!(error.fact_value("granted_bytes"), Some("10"));
    assert_eq!(work.bytes(), 40);
    assert_eq!(reservations.reserved_bytes(), 40);
    let denied = metrics
        .reservation_events
        .iter()
        .find(|event| event.outcome == MemoryReservationEventOutcome::Denied)
        .expect("grow denial should be recorded");
    assert_eq!(denied.action, MemoryReservationEventAction::Grow);
    assert_eq!(denied.memory_boundary, "operator_output");

    reservations.release(work, &mut metrics);
    assert_eq!(reservations.reserved_bytes(), 0);
    Ok(())
}

#[test]
fn typed_reservation_lineage_balances_transfer_alias_and_deferred_release() -> Result<()> {
    let lifecycle = RuntimeQueryLifecycle::new(
        RuntimeObservationSession::new("reservation-lineage", None, RuntimeObservationConfig::bounded(64, 64 * 1024)),
        PlanExecutionId::new("plan-reservation-lineage"),
    );
    let scope = lifecycle.execution_scope();
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget_and_observation(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::Static(100),
        Some(&scope),
    );
    let request = |kind, boundary, bytes| {
        test_request(kind, boundary, bytes, 100).with_node_id("lineage-node").with_task_registration(
            "lineage-task",
            1,
            "unary",
        )
    };
    let mut owned =
        reservations.try_reserve_work(request(ReservationKind::OperatorOutput, "work", 60), &mut metrics)?;
    reservations.try_grow(&mut owned, request(ReservationKind::OperatorOutput, "work_grow", 10), &mut metrics)?;
    reservations.shrink_to_observed(&mut owned, 50, &mut metrics);
    let transferred = reservations.transfer(owned, request(ReservationKind::ProbeSide, "probe", 50), &mut metrics);
    let alias = reservations.alias_reservation(
        &transferred,
        request(ReservationKind::OperatorOutput, "alias", 50),
        &mut metrics,
    );
    let alias = reservations.relabel_alias_reservation(
        alias,
        request(ReservationKind::ProbeSide, "relabeled_alias", 50),
        &mut metrics,
    );
    reservations.release(transferred, &mut metrics);
    assert_eq!(reservations.reserved_bytes(), 50, "owner charge remains live through its alias");
    reservations.release(alias, &mut metrics);
    assert_eq!(reservations.reserved_bytes(), 0);
    drop(reservations);

    let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
    let events = snapshot
        .events()
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::Reservation(observation) => Some((
                &observation.reservation_id,
                &observation.lineage_id,
                observation.action,
                observation.disposition,
                observation.charged_delta_bytes,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(events.iter().map(|event| event.4).sum::<i128>(), 0);
    assert!(events.iter().any(|event| {
        event.2 == RuntimeReservationAction::Transfer && event.3 == RuntimeReservationDisposition::Applied
    }));
    assert!(events.iter().any(|event| {
        event.2 == RuntimeReservationAction::Relabel && event.3 == RuntimeReservationDisposition::Applied
    }));
    assert!(events.iter().any(|event| {
        event.2 == RuntimeReservationAction::Release
            && event.3 == RuntimeReservationDisposition::AliasDeferred
            && event.4 == 0
    }));
    let alias =
        events.iter().find(|event| event.2 == RuntimeReservationAction::Alias).expect("alias acquisition is recorded");
    let root =
        events.iter().find(|event| event.2 == RuntimeReservationAction::Reserve).expect("root reserve is recorded");
    assert_ne!(alias.0, root.0);
    assert_eq!(alias.1, root.1);

    assert_eq!(metrics.reservation_requested_bytes, 70);
    assert_eq!(metrics.reservation_granted_bytes, 70);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.reservation_events.last().map(|event| event.current_reserved_bytes), Some(0));
    assert!(metrics.reservation_events.iter().all(|event| {
        event.query_id.as_ref().is_some_and(|query_id| query_id.as_str() == "reservation-lineage")
            && event.node_id.as_ref().is_some_and(|node_id| node_id.as_str() == "lineage-node")
            && event.compiled_task_id.as_ref().is_some_and(|task_id| task_id.as_str() == "lineage-task")
    }));
    let task = metrics.task_runtime_metrics.get("lineage-task").expect("typed task projection exists");
    assert_eq!(task.operator_kind, "unary");
    assert_eq!(task.reservation_requested_bytes, 70);
    assert_eq!(task.reservation_granted_bytes, 70);
    assert_eq!(task.reservation_released_bytes, 70);
    assert_eq!(task.reservation_live_peak_bytes, 70);
    Ok(())
}

#[test]
fn source_cache_promotion_and_eviction_keep_typed_meaning_and_legacy_public_totals() -> Result<()> {
    let lifecycle = RuntimeQueryLifecycle::new(
        RuntimeObservationSession::new(
            "source-cache-promotion",
            None,
            RuntimeObservationConfig::bounded(32, 64 * 1024),
        ),
        PlanExecutionId::new("plan-source-cache-promotion"),
    );
    let scope = lifecycle.execution_scope();
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget_and_observation(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::Static(100),
        Some(&scope),
    );
    let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
    let source = LocalParquetSource::new("/tmp/source-cache-promotion.parquet", schema, vec!["c0".to_string()]);
    let key = GpuLocalParquetSourceCacheKey::from_source(&source);
    let request = |kind, boundary, bytes| {
        test_request(kind, boundary, bytes, 100).with_node_id("source-cache-node").with_task_registration(
            "source-cache-task",
            1,
            "source_read",
        )
    };
    let mut source_reservation = reservations
        .try_reserve_owned_resident(request(ReservationKind::SourceRead, "source_read", 40), &mut metrics)?;
    reservations.promote_source_cache_insert(
        &mut source_reservation,
        60,
        request(ReservationKind::SourceCacheRetained, "source_cache_retained", 60),
        &mut metrics,
        &key,
    )?;
    reservations.release(source_reservation, &mut metrics);
    reservations.release_source_cache_baseline_bytes(60, &mut metrics);
    assert_eq!(reservations.raw_reserved_bytes(), 0);
    drop(reservations);

    let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
    let typed = snapshot
        .events()
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::Reservation(observation) => {
                Some((observation.action, observation.charged_delta_bytes))
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(typed.iter().any(|event| event.0 == RuntimeReservationAction::SourceCachePromote && event.1 == 20));
    assert!(typed.iter().any(|event| event.0 == RuntimeReservationAction::SourceCacheEvict && event.1 == -60));
    assert!(!typed.iter().any(|event| event.0 == RuntimeReservationAction::Transfer));
    assert_eq!(typed.iter().map(|event| event.1).sum::<i128>(), 0);

    let promotion = metrics
        .reservation_events
        .iter()
        .find(|event| event.memory_boundary == "source_cache_retained" && event.charged_delta_bytes > 0)
        .expect("legacy promotion row is retained");
    assert_eq!(promotion.action, MemoryReservationEventAction::Reserve);
    assert_eq!(promotion.requested_bytes, 60);
    assert_eq!(promotion.granted_bytes, 60);
    let eviction = metrics
        .reservation_events
        .iter()
        .find(|event| event.stage.as_deref() == Some("source_cache_eviction"))
        .expect("legacy eviction row is retained");
    assert_eq!(eviction.action, MemoryReservationEventAction::Release);
    assert_eq!(eviction.charged_delta_bytes, -60);
    assert_eq!(eviction.current_reserved_bytes, 0);
    assert_eq!(metrics.reservation_requested_bytes, 100);
    assert_eq!(metrics.reservation_granted_bytes, 100);
    let task = &metrics.task_runtime_metrics["source-cache-task"];
    assert_eq!(task.reservation_requested_bytes, 100);
    assert_eq!(task.reservation_granted_bytes, 60);
    assert_eq!(task.reservation_released_bytes, 60);
    assert_eq!(task.live_reserved_bytes, 0);
    Ok(())
}

#[test]
fn typed_reservations_replace_source_paths_with_stable_opaque_ids() -> Result<()> {
    let sensitive_locations = [
        "/srv/private/customer-42/orders.parquet",
        "s3://alice:secret@bucket/private/key.parquet?token=top-secret",
        "catalog/customer/partition=private",
    ];
    let source_location_key = [0x5a; blake3::KEY_LEN];
    let session = RuntimeObservationSession::with_source_location_test_key(
        "source-location-privacy",
        None,
        RuntimeObservationConfig::bounded(32, 64 * 1024),
        source_location_key,
    );
    let expected_ids = sensitive_locations
        .iter()
        .map(|location| session.handle().source_location_id(location).expect("enabled session has a digest key"))
        .collect::<Vec<_>>();
    let lifecycle = RuntimeQueryLifecycle::new(session, PlanExecutionId::new("plan-source-location-privacy"));
    let scope = lifecycle.execution_scope();
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget_and_observation(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::Static(16),
        Some(&scope),
    );
    for (index, location) in sensitive_locations.iter().enumerate() {
        let reservation = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::SourceRead, "private_source", 1, reservations.budget())
                .with_path(location)
                .with_file_index(index)
                .with_row_groups(vec![index, index + 2]),
            &mut metrics,
        )?;
        reservations.release(reservation, &mut metrics);
    }
    drop(reservations);
    let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
    assert!(!snapshot.footer().degraded, "Parquet row-group preflight must use the exact retained size");
    let ids = snapshot
        .events()
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::Reservation(observation) => observation.source_location_id.clone(),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), sensitive_locations.len() * 2, "reserve and release retain opaque correlation");
    for (index, expected) in expected_ids.iter().enumerate() {
        assert_eq!(&ids[index * 2], expected);
        assert_eq!(&ids[index * 2 + 1], expected);
        assert!(expected.as_str().starts_with("source_location:blake3_keyed_256:"));
        assert!(!expected.as_str().contains(":len:"));
    }
    assert_ne!(ids[0], ids[2]);
    assert_ne!(ids[2], ids[4]);

    let mut jsonl = Vec::new();
    write_runtime_observation_jsonl(&snapshot, &mut jsonl).expect("privacy snapshot exports");
    let jsonl = String::from_utf8(jsonl).expect("JSONL is UTF-8");
    for location in sensitive_locations {
        assert!(!jsonl.contains(location), "raw source location leaked: {location}");
    }
    assert!(!jsonl.contains("alice:secret"));
    assert!(!jsonl.contains("top-secret"));
    assert!(!jsonl.contains(&"5a".repeat(blake3::KEY_LEN)));
    assert!(jsonl.contains("source_location_id"));
    assert!(
        metrics
            .reservation_events
            .iter()
            .any(|event| { event.path.as_deref() == Some("/srv/private/customer-42/orders.parquet") }),
        "required public reservation diagnostics remain unchanged"
    );
    Ok(())
}

#[test]
fn typed_reservation_drop_unwind_and_denial_are_distinct_and_balanced() -> Result<()> {
    let lifecycle = RuntimeQueryLifecycle::new(
        RuntimeObservationSession::new("reservation-terminal", None, RuntimeObservationConfig::bounded(64, 64 * 1024)),
        PlanExecutionId::new("plan-reservation-terminal"),
    );
    let scope = lifecycle.execution_scope();
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget_and_observation(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::Static(100),
        Some(&scope),
    );

    let dropped = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "drop", 20, reservations.budget()),
        &mut metrics,
    )?;
    drop(dropped);
    assert_eq!(reservations.reserved_bytes(), 0);

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _unwinding = reservations
            .try_reserve_owned_resident(
                test_request(ReservationKind::OperatorOutput, "unwind", 30, reservations.budget()),
                &mut metrics,
            )
            .expect("unwind reservation fits");
        panic!("reservation unwind probe");
    }));
    assert!(unwind.is_err());
    assert_eq!(reservations.reserved_bytes(), 0);

    let live = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "live", 80, reservations.budget()),
        &mut metrics,
    )?;
    let denial = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::ProbeSide, "denied", 30, reservations.budget()),
        &mut metrics,
    );
    assert!(denial.is_err());
    reservations.release(live, &mut metrics);
    drop(reservations);

    let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
    let events = snapshot
        .events()
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::Reservation(observation) => {
                Some((observation.action, observation.disposition, observation.charged_delta_bytes))
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(events.iter().map(|event| event.2).sum::<i128>(), 0);
    assert!(events.iter().any(|event| event.1 == RuntimeReservationDisposition::DropRelease));
    assert!(events.iter().any(|event| event.1 == RuntimeReservationDisposition::UnwindRelease));
    assert!(events.iter().any(|event| {
        event.0 == RuntimeReservationAction::Deny && event.1 == RuntimeReservationDisposition::Denied
    }));
    Ok(())
}

#[test]
fn typed_reservation_rollback_and_cleanup_releases_are_balanced() -> Result<()> {
    let lifecycle = RuntimeQueryLifecycle::new(
        RuntimeObservationSession::new("reservation-cleanup", None, RuntimeObservationConfig::bounded(32, 64 * 1024)),
        PlanExecutionId::new("plan-reservation-cleanup"),
    );
    let scope = lifecycle.execution_scope();
    let mut metrics = QueryMetrics::default();
    let mut reservations = ReservationTracker::new_with_budget_and_observation(
        MemorySpaceId::Device { device_ordinal: 0 },
        ReservationBudget::Static(100),
        Some(&scope),
    );
    let rollback = reservations.try_reserve_work(
        test_request(ReservationKind::GatherMap, "rollback", 30, reservations.budget()),
        &mut metrics,
    )?;
    {
        let _rollback_scope =
            reservations.enter_release_observation_scope(RuntimeReservationDisposition::RollbackRelease);
        reservations.release(rollback, &mut metrics);
    }
    let cleanup = reservations.try_reserve_owned_resident(
        test_request(ReservationKind::OperatorOutput, "cleanup", 40, reservations.budget()),
        &mut metrics,
    )?;
    {
        let _cleanup_scope =
            reservations.enter_release_observation_scope(RuntimeReservationDisposition::CleanupRelease);
        reservations.release(cleanup, &mut metrics);
    }
    reservations.try_reserve_source_cache_baseline(20, &mut metrics)?;
    reservations.release_source_cache_baseline_bytes(20, &mut metrics);
    drop(reservations);

    let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
    let events = snapshot
        .events()
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::Reservation(observation) => {
                Some((observation.action, observation.disposition, observation.charged_delta_bytes))
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(events.iter().map(|event| event.2).sum::<i128>(), 0);
    assert!(events.iter().any(|event| {
        event.0 == RuntimeReservationAction::Rollback && event.1 == RuntimeReservationDisposition::RollbackRelease
    }));
    assert!(events.iter().any(|event| {
        event.0 == RuntimeReservationAction::Release && event.1 == RuntimeReservationDisposition::CleanupRelease
    }));
    Ok(())
}
