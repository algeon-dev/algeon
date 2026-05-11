//! Maintenance loop, shutdown plan, and clock abstraction.

use std::{future::Future, pin::Pin, sync::Arc, time::Instant};

use tokio::{sync::futures::Notified, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use super::{
    super::observability::{CapturedError, CapturedQuery, attempt_outcome_label},
    QueryCoordinator,
    artifacts::{
        compose_terminal_artifact, incomplete_runtime_control, unavailable_stats_payload, unavailable_tombstone_stats,
    },
    finalization::{ShutdownDiscarded, ShutdownTimeoutOutput, ShutdownTimeoutSnapshot},
    retention::checked_deadline,
    types::{QueryPhase, QueryTerminal, QueryTerminalOutcome, RetainedUnavailableReason},
};
use crate::adapter_error::{Error, ErrorCode};

pub(crate) trait CoordinatorMaintenanceClock: Send + Sync + 'static {
    fn now(&self) -> Instant;

    fn sleep_until(&self, deadline: Instant) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

pub(crate) struct TokioCoordinatorMaintenanceClock;

impl CoordinatorMaintenanceClock for TokioCoordinatorMaintenanceClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep_until(&self, deadline: Instant) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)))
    }
}

/// The one server-owned task that advances ticket/retention/shutdown deadlines.
pub(crate) struct QueryCoordinatorMaintenance {
    pub(crate) coordinator: Arc<QueryCoordinator>,
    pub(crate) clock: Arc<dyn CoordinatorMaintenanceClock>,
    pub(crate) stop: CancellationToken,
    pub(crate) task: JoinHandle<()>,
}

impl QueryCoordinatorMaintenance {
    pub(crate) fn start(coordinator: Arc<QueryCoordinator>) -> Self {
        Self::start_with_clock(coordinator, Arc::new(TokioCoordinatorMaintenanceClock))
    }

    pub(crate) fn start_with_clock(
        coordinator: Arc<QueryCoordinator>,
        clock: Arc<dyn CoordinatorMaintenanceClock>,
    ) -> Self {
        let stop = CancellationToken::new();
        let task =
            tokio::spawn(run_coordinator_maintenance(Arc::clone(&coordinator), Arc::clone(&clock), stop.clone()));
        Self { coordinator, clock, stop, task }
    }

    /// Join only after graceful finalization or the configured forced,
    /// incomplete shutdown conversion has made the live registry empty.
    pub(crate) async fn finish(
        mut self,
        plan: QueryCoordinatorShutdownPlan,
    ) -> std::result::Result<bool, tokio::task::JoinError> {
        let mut deadline_expired = false;
        loop {
            let now = self.clock.now();
            if now >= plan.deadline() {
                self.coordinator.force_shutdown_timeout(now);
                deadline_expired = true;
                break;
            }
            if self.coordinator.is_drained() {
                break;
            }
            let mut notified = Box::pin(self.coordinator.notified());
            notified.as_mut().enable();
            if self.coordinator.is_drained() {
                let now = self.clock.now();
                if now >= plan.deadline() {
                    self.coordinator.force_shutdown_timeout(now);
                    deadline_expired = true;
                }
                break;
            }
            tokio::select! {
                () = &mut notified => {},
                () = self.clock.sleep_until(plan.deadline()) => {
                    self.coordinator.force_shutdown_timeout(self.clock.now());
                    deadline_expired = true;
                    break;
                },
            }
        }
        let forced = self.coordinator.shutdown_forced();
        self.stop.cancel();
        (&mut self.task).await?;
        Ok(forced || deadline_expired || self.clock.now() >= plan.deadline())
    }
}

/// Immutable coordinator shutdown handoff captured before mutable coordinator
/// state can be finalized or discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueryCoordinatorShutdownPlan {
    pub(crate) deadline: Instant,
    pub(crate) live_count: usize,
}

impl QueryCoordinatorShutdownPlan {
    pub(crate) fn deadline(self) -> Instant {
        self.deadline
    }

    pub(crate) fn live_count(self) -> usize {
        self.live_count
    }
}

impl Drop for QueryCoordinatorMaintenance {
    fn drop(&mut self) {
        self.stop.cancel();
        self.task.abort();
    }
}

pub(crate) async fn run_coordinator_maintenance(
    coordinator: Arc<QueryCoordinator>,
    clock: Arc<dyn CoordinatorMaintenanceClock>,
    stop: CancellationToken,
) {
    loop {
        let mut notified = Box::pin(coordinator.notified());
        notified.as_mut().enable();
        coordinator.maintenance_at(clock.now());
        if stop.is_cancelled() {
            return;
        }
        match coordinator.next_deadline() {
            Some(deadline) => {
                tokio::select! {
                    () = stop.cancelled() => return,
                    () = &mut notified => {},
                    () = clock.sleep_until(deadline) => {},
                }
            },
            None => {
                tokio::select! {
                    () = stop.cancelled() => return,
                    () = &mut notified => {},
                }
            },
        }
    }
}

impl QueryCoordinator {
    pub(crate) fn maintenance_at(&self, now: Instant) {
        // Releasing an unconsumed ticket can terminalize its attempt. That
        // synchronously seals the observation, whose report observer reenters
        // the coordinator. Keep the payload alive until after the state mutex
        // is released, just as cancellation and shutdown do for their
        // callback-capable handles.
        let (shutdown_due, expired_ticket_payloads) = {
            let mut state = self.lock_state();
            let expired_tickets = state
                .records
                .iter()
                .filter(|(_, record)| {
                    record.phase == QueryPhase::Ticketed
                        && record.ticket.as_ref().is_some_and(|ticket| now >= ticket.deadline)
                })
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>();
            let mut expired_ticket_payloads = Vec::with_capacity(expired_tickets.len());
            for key in expired_tickets {
                if let Some(record) = state.records.get_mut(&key) {
                    if let Some(ticket) = record.ticket.take() {
                        expired_ticket_payloads.push(ticket);
                    }
                    if record.terminal_outcome.is_none() {
                        let mut terminal = QueryTerminal::Dropped;
                        Self::apply_terminal_to_capture(record, &mut terminal, now);
                        record.terminal_outcome = Some(terminal.compact_outcome());
                        record.terminal = Some(terminal);
                    }
                    record.unavailable = Some(RetainedUnavailableReason::TicketExpired);
                    record.phase = QueryPhase::Finalizing;
                    state.ticketed_count = state.ticketed_count.checked_sub(1).expect("expired ticket count underflow");
                }
            }
            let expired_retained = state
                .records
                .iter()
                .filter(|(_, record)| {
                    record.phase == QueryPhase::Retained
                        && record.retention_deadline.is_some_and(|deadline| now >= deadline)
                })
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>();
            for key in expired_retained {
                Self::evict_retained_locked(&self.config, &mut state, &key, now);
            }
            let expired_tombstones = state
                .records
                .iter()
                .filter(|(_, record)| {
                    record.phase == QueryPhase::Tombstone
                        && record.tombstone_deadline.is_some_and(|deadline| now >= deadline)
                })
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>();
            for key in expired_tombstones {
                Self::remove_tombstone_locked(&mut state, &key);
            }
            Self::enforce_retention_locked(&self.config, &mut state, now);
            (
                state.shutdown_deadline.is_some_and(|deadline| now >= deadline) && Self::live_count_locked(&state) > 0,
                expired_ticket_payloads,
            )
        };
        drop(expired_ticket_payloads);
        self.finalize_ready();
        if shutdown_due {
            self.force_shutdown_timeout(now);
        }
    }

    pub(crate) fn begin_shutdown(&self, now: Instant) -> QueryCoordinatorShutdownPlan {
        let shutdown_error = Arc::new(CapturedError::from_adapter_error(&Error::new(
            ErrorCode::Cancelled,
            "datafusion_nexus.server.flight.shutdown",
            "server shutting down; statement aborted",
        )));
        // Attempt cancellation synchronously seals its observation and calls
        // back into the coordinator to attach the immutable report. Take all
        // callback-capable handles while locked, but invoke them only after
        // the state transition is visible and the mutex has been released.
        let (live_count, deadline, tickets_to_cancel, executions_to_cancel) = {
            let mut state = self.lock_state();
            state.accepting = false;
            let deadline = *state.shutdown_deadline.get_or_insert(checked_deadline(now, self.config.shutdown_grace));
            let mut removed_tickets = 0usize;
            let mut tickets_to_cancel = Vec::new();
            let mut executions_to_cancel = Vec::new();
            for record in state.records.values_mut() {
                if !matches!(
                    record.phase,
                    QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                ) {
                    continue;
                }
                if let Some(ticket) = record.ticket.take() {
                    tickets_to_cancel.push(ticket);
                    removed_tickets = removed_tickets.checked_add(1).expect("removed ticket count overflow");
                }
                if let Some(cancellation) = &record.execution_cancellation {
                    executions_to_cancel.push(cancellation.clone());
                }
                if record.terminal_outcome.is_none() {
                    let mut terminal =
                        QueryTerminal::Cancelled { stage: "server_shutdown", error: shutdown_error.clone() };
                    Self::apply_terminal_to_capture(record, &mut terminal, now);
                    record.terminal_outcome = Some(terminal.compact_outcome());
                    record.terminal = Some(terminal);
                }
                if record.phase == QueryPhase::Ticketed {
                    record.phase = QueryPhase::Finalizing;
                }
            }
            state.ticketed_count =
                state.ticketed_count.checked_sub(removed_tickets).expect("shutdown ticket count underflow");
            (Self::live_count_locked(&state), deadline, tickets_to_cancel, executions_to_cancel)
        };
        for ticket in tickets_to_cancel {
            ticket.native_cancellation.cancel();
            if let Some(attempt) = ticket.attempt.as_ref() {
                attempt.cancel_before_stream();
            }
        }
        for cancellation in executions_to_cancel {
            cancellation.cancel();
        }
        self.notify.notify_waiters();
        QueryCoordinatorShutdownPlan { deadline, live_count }
    }

    pub(crate) fn notified(&self) -> Notified<'_> {
        self.notify.notified()
    }

    pub(crate) fn next_deadline(&self) -> Option<Instant> {
        let state = self.lock_state();
        let shutdown_deadline = (Self::live_count_locked(&state) > 0).then_some(state.shutdown_deadline).flatten();
        state
            .records
            .values()
            .flat_map(|record| {
                [
                    record.ticket.as_ref().map(|ticket| ticket.deadline),
                    record.retention_deadline,
                    record.tombstone_deadline,
                ]
                .into_iter()
                .flatten()
            })
            .chain(shutdown_deadline)
            .min()
    }

    pub(crate) fn is_drained(&self) -> bool {
        Self::live_count_locked(&self.lock_state()) == 0
    }

    #[cfg(test)]
    pub(crate) fn shutdown_deadline(&self) -> Option<Instant> {
        self.lock_state().shutdown_deadline
    }

    pub(crate) fn shutdown_forced(&self) -> bool {
        self.lock_state().shutdown_forced
    }

    pub(crate) fn force_shutdown_timeout(&self, now: Instant) -> usize {
        let snapshots = {
            let state = self.lock_state();
            state
                .records
                .iter()
                .filter(|(_, record)| {
                    matches!(
                        record.phase,
                        QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                    )
                })
                .map(|(key, record)| ShutdownTimeoutSnapshot {
                    key: key.clone(),
                    captured: record.captured.as_ref().map(CapturedQuery::shutdown_snapshot),
                    attempt_outcome: record.attempt_outcome.clone(),
                    hot: record.hot.snapshot(),
                    artifact_max_bytes: record.artifact_max_bytes,
                    live_stats_cache: {
                        let cache = Arc::clone(&record.live_stats_cache);
                        cache.close();
                        cache
                    },
                })
                .collect::<Vec<_>>()
        };
        if snapshots.is_empty() {
            let mut state = self.lock_state();
            state.shutdown_deadline = None;
            return 0;
        }

        let outputs = snapshots
            .into_iter()
            .map(|snapshot| {
                snapshot.live_stats_cache.wait_drained();
                let reason = RetainedUnavailableReason::ShutdownTimeout;
                let outcome = snapshot.attempt_outcome.as_ref().map(attempt_outcome_label).unwrap_or("dropped");
                let shutdown_delivery_outcome = QueryTerminalOutcome::ShutdownTimeout.stats_outcome();
                let delivery_outcome = (outcome != shutdown_delivery_outcome).then_some(shutdown_delivery_outcome);
                let control = incomplete_runtime_control(snapshot.key.query_id(), reason);
                let composed = compose_terminal_artifact(control, None, snapshot.artifact_max_bytes, |remaining| {
                    snapshot.captured.as_ref().map_or_else(
                        || {
                            unavailable_stats_payload(
                                snapshot.key.query_id(),
                                outcome,
                                delivery_outcome,
                                reason,
                                remaining,
                            )
                        },
                        |captured| {
                            captured.to_query_stats_payload(
                                now,
                                snapshot.hot.output_batches,
                                snapshot.hot.output_rows,
                                snapshot.hot.output_bytes,
                                outcome,
                                delivery_outcome,
                                remaining,
                            )
                        },
                    )
                });
                let bundle = composed.map(|artifact| artifact.bundle);
                let tombstone_stats = bundle
                    .is_none()
                    .then(|| {
                        unavailable_tombstone_stats(
                            snapshot.key.query_id(),
                            outcome,
                            delivery_outcome,
                            reason,
                            snapshot.artifact_max_bytes,
                        )
                    })
                    .flatten();
                ShutdownTimeoutOutput {
                    key: snapshot.key,
                    live_stats_cache: snapshot.live_stats_cache,
                    bundle,
                    tombstone_stats,
                }
            })
            .collect::<Vec<_>>();

        let (converted, discarded) = {
            let mut state = self.lock_state();
            let mut converted = 0usize;
            let mut discarded = Vec::new();
            for output in outputs {
                let is_live = state.records.get(&output.key).is_some_and(|record| {
                    matches!(
                        record.phase,
                        QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                    )
                });
                if !is_live {
                    output.live_stats_cache.abandon();
                    continue;
                }
                let charge_bytes = output.bundle.as_ref().map_or(0, |bundle| bundle.charge_bytes);
                let tombstone_bytes =
                    output.tombstone_stats.as_ref().map_or(0, |stats| u64::try_from(stats.len()).unwrap_or(u64::MAX));
                let terminal_stats = output
                    .bundle
                    .as_ref()
                    .map(|bundle| Arc::clone(&bundle.stats))
                    .or_else(|| output.tombstone_stats.as_ref().map(Arc::clone));
                let retention_order = if output.bundle.is_some() {
                    state.next_retention_order =
                        state.next_retention_order.checked_add(1).expect("retention order overflow");
                    Some(state.next_retention_order)
                } else {
                    None
                };
                let (logical_capture_bytes, was_ticketed, heavy) = {
                    let record = state.records.get_mut(&output.key).expect("live shutdown record remains present");
                    let was_ticketed = record.ticket.is_some();
                    let heavy = ShutdownDiscarded {
                        _captured: record.captured.take(),
                        _observation_owner: record.observation_owner.take(),
                        _terminal: record.terminal.take(),
                        _ticket: record.ticket.take(),
                        _execution_cancellation: record.execution_cancellation.take(),
                    };
                    let logical_capture_bytes = std::mem::take(&mut record.logical_capture_bytes);
                    record.terminal_outcome = Some(QueryTerminalOutcome::ShutdownTimeout);
                    record.active_leases = 0;
                    record.active_activities = 0;
                    record.finalization_claim = None;
                    record.phase = if output.bundle.is_some() { QueryPhase::Retained } else { QueryPhase::Tombstone };
                    record.retained = output.bundle;
                    record.unavailable = Some(RetainedUnavailableReason::ShutdownTimeout);
                    record.retention_deadline =
                        retention_order.map(|_| checked_deadline(now, self.config.retained_ttl));
                    record.retention_order = retention_order;
                    record.tombstone_deadline =
                        retention_order.is_none().then(|| checked_deadline(now, self.config.tombstone_ttl));
                    record.tombstone_stats = output.tombstone_stats;
                    (logical_capture_bytes, was_ticketed, heavy)
                };
                output.live_stats_cache.publish_optional_terminal(terminal_stats);
                if was_ticketed {
                    state.ticketed_count = state.ticketed_count.checked_sub(1).expect("ticketed count underflow");
                }
                state.active_logical_capture_bytes = state
                    .active_logical_capture_bytes
                    .checked_sub(logical_capture_bytes)
                    .expect("forced shutdown releases each active capture reservation exactly once");
                if retention_order.is_some() {
                    state.retained_count = state.retained_count.checked_add(1).expect("retained count overflow");
                    state.retained_bytes =
                        state.retained_bytes.checked_add(charge_bytes).expect("retained bytes overflow");
                } else {
                    state.tombstone_count = state.tombstone_count.checked_add(1).expect("tombstone count overflow");
                    state.tombstone_bytes =
                        state.tombstone_bytes.checked_add(tombstone_bytes).expect("tombstone bytes overflow");
                }
                converted = converted.checked_add(1).expect("converted query count overflow");
                discarded.push(heavy);
            }
            state.shutdown_deadline = None;
            state.shutdown_forced |= converted > 0;
            Self::enforce_retention_locked(&self.config, &mut state, now);
            (converted, discarded)
        };
        drop(discarded);
        self.notify.notify_waiters();
        converted
    }
}
