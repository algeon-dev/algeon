//! Finalization claim, unlocked render, and locked install paths.

#[cfg(test)]
use std::sync::atomic::Ordering;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Instant,
};

use nexus_query_engine::{
    observability::{RuntimeObservationOwner, RuntimeTerminalOutcome, write_runtime_observation_jsonl},
    runtime::ExecutionCancellation,
};

#[cfg(test)]
use super::artifacts::{FinalizationTestGate, run_coordinator_render_probe};
use super::{
    super::observability::{CapturedQuery, ShutdownCaptureSnapshot},
    QueryCoordinator,
    artifacts::{
        LimitedVecWriter, LiveStatsCache, compose_terminal_artifact, incomplete_runtime_control, runtime_control,
        unavailable_stats_payload, unavailable_tombstone_stats,
    },
    retention::checked_deadline,
    types::{
        CoordinatorTicketPayload, QueryHotCounterSnapshot, QueryKey, QueryPhase, QueryTerminal, QueryTerminalOutcome,
        RetainedQueryBundle, RetainedUnavailableReason,
    },
};

pub(in crate::server) struct FinalizationInputs {
    pub(in crate::server) key: QueryKey,
    pub(in crate::server) claim: u64,
    pub(in crate::server) terminal_outcome: QueryTerminalOutcome,
    pub(in crate::server) runtime_outcome: RuntimeTerminalOutcome,
    pub(in crate::server) captured: CapturedQuery,
    pub(in crate::server) observation_owner: Option<RuntimeObservationOwner>,
    pub(in crate::server) live_stats_cache: Arc<LiveStatsCache>,
    pub(in crate::server) hot: QueryHotCounterSnapshot,
    pub(in crate::server) artifact_max_bytes: u64,
    pub(in crate::server) preselected_unavailable: Option<RetainedUnavailableReason>,
    #[cfg(test)]
    pub(in crate::server) finalization_gate: Option<Arc<FinalizationTestGate>>,
}

pub(in crate::server) struct FinalizationFailureContext {
    pub(in crate::server) key: QueryKey,
    pub(in crate::server) claim: u64,
    pub(in crate::server) live_stats_cache: Arc<LiveStatsCache>,
    pub(in crate::server) outcome: &'static str,
    pub(in crate::server) artifact_max_bytes: u64,
}

impl FinalizationFailureContext {
    pub(in crate::server) fn from_inputs(inputs: &FinalizationInputs) -> Self {
        let outcome = inputs.terminal_outcome.stats_outcome();
        Self {
            key: inputs.key.clone(),
            claim: inputs.claim,
            live_stats_cache: Arc::clone(&inputs.live_stats_cache),
            outcome,
            artifact_max_bytes: inputs.artifact_max_bytes,
        }
    }
}

pub(in crate::server) struct ShutdownTimeoutSnapshot {
    pub(in crate::server) key: QueryKey,
    pub(in crate::server) captured: Option<ShutdownCaptureSnapshot>,
    pub(in crate::server) hot: QueryHotCounterSnapshot,
    pub(in crate::server) artifact_max_bytes: u64,
    pub(in crate::server) live_stats_cache: Arc<LiveStatsCache>,
}

pub(in crate::server) struct ShutdownTimeoutOutput {
    pub(in crate::server) key: QueryKey,
    pub(in crate::server) live_stats_cache: Arc<LiveStatsCache>,
    pub(in crate::server) bundle: Option<Arc<RetainedQueryBundle>>,
    pub(in crate::server) tombstone_stats: Option<Arc<[u8]>>,
}

pub(in crate::server) struct ShutdownDiscarded {
    pub(in crate::server) _captured: Option<CapturedQuery>,
    pub(in crate::server) _observation_owner: Option<RuntimeObservationOwner>,
    pub(in crate::server) _terminal: Option<QueryTerminal>,
    pub(in crate::server) _ticket: Option<CoordinatorTicketPayload>,
    pub(in crate::server) _execution_cancellation: Option<ExecutionCancellation>,
}

pub(in crate::server) struct FinalizationOutput {
    pub(in crate::server) key: QueryKey,
    pub(in crate::server) claim: u64,
    pub(in crate::server) live_stats_cache: Arc<LiveStatsCache>,
    pub(in crate::server) bundle: Option<Arc<RetainedQueryBundle>>,
    pub(in crate::server) unavailable: Option<RetainedUnavailableReason>,
    pub(in crate::server) tombstone_stats: Option<Arc<[u8]>>,
}

impl FinalizationOutput {
    pub(in crate::server) fn unavailable(
        key: QueryKey,
        claim: u64,
        live_stats_cache: Arc<LiveStatsCache>,
        outcome: &'static str,
        reason: RetainedUnavailableReason,
        artifact_max_bytes: u64,
    ) -> Self {
        let control = incomplete_runtime_control(key.query_id(), reason);
        let composed = compose_terminal_artifact(control, None, artifact_max_bytes, |remaining| {
            unavailable_stats_payload(key.query_id(), outcome, reason, remaining)
        });
        let bundle = composed.map(|artifact| artifact.bundle);
        let tombstone_stats = bundle
            .is_none()
            .then(|| unavailable_tombstone_stats(key.query_id(), outcome, reason, artifact_max_bytes))
            .flatten();
        Self { key, claim, live_stats_cache, bundle, unavailable: Some(reason), tombstone_stats }
    }

    pub(in crate::server) fn panicked(context: FinalizationFailureContext) -> Self {
        Self::unavailable(
            context.key,
            context.claim,
            context.live_stats_cache,
            context.outcome,
            RetainedUnavailableReason::FinalizationPanicked,
            context.artifact_max_bytes,
        )
    }
}

impl QueryCoordinator {
    pub(in crate::server) fn finalize_ready(&self) {
        loop {
            let Some(inputs) = self.take_finalization_inputs() else { break };
            let failure = FinalizationFailureContext::from_inputs(&inputs);
            let output = catch_unwind(AssertUnwindSafe(|| Self::finalize_outside_lock(inputs)))
                .unwrap_or_else(|_| FinalizationOutput::panicked(failure));
            self.install_finalization(output, Instant::now());
        }
    }

    pub(in crate::server) fn take_finalization_inputs(&self) -> Option<FinalizationInputs> {
        #[cfg(test)]
        let finalization_gate =
            self.finalization_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let mut state = self.lock_state();
        let key = state.records.iter().find_map(|(key, record)| {
            (record.terminal_outcome.is_some()
                && record.active_leases == 0
                && record.active_activities == 0
                && record.finalization_claim.is_none()
                && matches!(record.phase, QueryPhase::Planning | QueryPhase::Executing | QueryPhase::Finalizing))
            .then(|| key.clone())
        })?;
        let record = state.records.get_mut(&key).expect("selected finalization record remains present");
        record.phase = QueryPhase::Finalizing;
        let claim = key.generation;
        record.finalization_claim = Some(claim);
        let terminal = record.terminal.take().expect("finalization requires a terminal payload");
        let terminal_outcome = terminal.compact_outcome();
        debug_assert_eq!(record.terminal_outcome, Some(terminal_outcome));
        let runtime_outcome = terminal.runtime_outcome();
        drop(terminal);
        let live_stats_cache = Arc::clone(&record.live_stats_cache);
        live_stats_cache.close();
        let inputs = FinalizationInputs {
            key,
            claim,
            terminal_outcome,
            runtime_outcome,
            captured: record.captured.take().expect("finalizing live query retains captured stats inputs"),
            observation_owner: record.observation_owner.take(),
            live_stats_cache: Arc::clone(&live_stats_cache),
            hot: record.hot.snapshot(),
            artifact_max_bytes: record.artifact_max_bytes,
            preselected_unavailable: record.unavailable,
            #[cfg(test)]
            finalization_gate,
        };
        drop(state);
        live_stats_cache.wait_drained();
        Some(inputs)
    }

    pub(in crate::server) fn finalize_outside_lock(mut inputs: FinalizationInputs) -> FinalizationOutput {
        #[cfg(test)]
        if let Some(gate) = &inputs.finalization_gate {
            gate.wait();
        }
        #[cfg(test)]
        run_coordinator_render_probe("finalization projection");
        inputs.captured.set_output_metrics(
            inputs.hot.output_batches,
            inputs.hot.output_rows,
            inputs.hot.output_bytes,
            inputs.hot.overflow,
        );
        inputs.captured.seal(Instant::now());
        let stats_outcome = inputs.terminal_outcome.stats_outcome();
        let Some(owner) = inputs.observation_owner else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                RetainedUnavailableReason::CaptureDisabled,
                inputs.artifact_max_bytes,
            );
        };
        let Some(snapshot) = owner.finish_query(inputs.runtime_outcome) else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                RetainedUnavailableReason::CaptureDisabled,
                inputs.artifact_max_bytes,
            );
        };
        let mut incomplete_reason = snapshot
            .footer()
            .counter_overflow
            .then_some(RetainedUnavailableReason::CounterOverflow)
            .or_else(|| inputs.hot.overflow.then_some(RetainedUnavailableReason::CounterOverflow))
            .or(inputs.preselected_unavailable)
            .or_else(|| {
                (snapshot.footer().truncated || snapshot.footer().degraded)
                    .then_some(RetainedUnavailableReason::CaptureTruncated)
            });
        let control = incomplete_reason.map_or_else(
            || runtime_control(inputs.key.query_id(), "sealed", false, None),
            |reason| incomplete_runtime_control(inputs.key.query_id(), reason),
        );
        let Some(jsonl_limit) = inputs.artifact_max_bytes.checked_sub(u64::try_from(control.len()).unwrap_or(u64::MAX))
        else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                RetainedUnavailableReason::TooLarge,
                inputs.artifact_max_bytes,
            );
        };
        let mut jsonl = LimitedVecWriter::new(usize::try_from(jsonl_limit).unwrap_or(usize::MAX));
        if write_runtime_observation_jsonl(&snapshot, &mut jsonl).is_err() {
            let reason = if jsonl.overflowed {
                RetainedUnavailableReason::TooLarge
            } else {
                RetainedUnavailableReason::EncodeFailed
            };
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                reason,
                inputs.artifact_max_bytes,
            );
        }
        let jsonl = Arc::<[u8]>::from(jsonl.bytes);
        let Some(mut composed) =
            compose_terminal_artifact(control, Some(Arc::clone(&jsonl)), inputs.artifact_max_bytes, |remaining| {
                inputs.captured.to_query_stats_payload(remaining)
            })
        else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                RetainedUnavailableReason::TooLarge,
                inputs.artifact_max_bytes,
            );
        };
        if incomplete_reason.is_none() && composed.stats_incomplete {
            incomplete_reason = Some(RetainedUnavailableReason::CaptureTruncated);
            let control =
                incomplete_runtime_control(inputs.key.query_id(), RetainedUnavailableReason::CaptureTruncated);
            let Some(recomposed) =
                compose_terminal_artifact(control, Some(jsonl), inputs.artifact_max_bytes, |remaining| {
                    inputs.captured.to_query_stats_payload(remaining)
                })
            else {
                return FinalizationOutput::unavailable(
                    inputs.key,
                    inputs.claim,
                    inputs.live_stats_cache,
                    stats_outcome,
                    RetainedUnavailableReason::TooLarge,
                    inputs.artifact_max_bytes,
                );
            };
            composed = recomposed;
        }
        FinalizationOutput {
            key: inputs.key,
            claim: inputs.claim,
            live_stats_cache: inputs.live_stats_cache,
            bundle: Some(composed.bundle),
            unavailable: incomplete_reason,
            tombstone_stats: None,
        }
    }

    pub(in crate::server) fn install_finalization(&self, mut output: FinalizationOutput, now: Instant) {
        let mut state = self.lock_state();
        let Some(record) = state.records.get(&output.key) else {
            drop(state);
            output.live_stats_cache.abandon();
            return;
        };
        if record.phase != QueryPhase::Finalizing
            || record.key != output.key
            || record.finalization_claim != Some(output.claim)
        {
            drop(state);
            output.live_stats_cache.abandon();
            return;
        }
        #[cfg(test)]
        self.finalization_installs.fetch_add(1, Ordering::SeqCst);
        let logical_capture_bytes = record.logical_capture_bytes;
        let terminal_outcome = record.terminal_outcome.map_or("unknown", QueryTerminalOutcome::stats_outcome);
        state.active_logical_capture_bytes = state
            .active_logical_capture_bytes
            .checked_sub(logical_capture_bytes)
            .expect("live finalization releases its active capture reservation exactly once");
        {
            let record = state.records.get_mut(&output.key).expect("validated finalization record remains present");
            record.logical_capture_bytes = 0;
            record.captured = None;
            record.execution_cancellation = None;
            record.finalization_claim = None;
        }
        let charge_bytes = output.bundle.as_ref().map_or(0, |bundle| bundle.charge_bytes);
        if charge_bytes > self.config.retained_max_bytes {
            output.unavailable = Some(RetainedUnavailableReason::TooLarge);
            output.bundle = None;
            output.tombstone_stats = unavailable_tombstone_stats(
                output.key.query_id(),
                terminal_outcome,
                RetainedUnavailableReason::TooLarge,
                self.config.per_query_artifact_max_bytes,
            );
        }
        let terminal_stats = match output.bundle.take() {
            None => {
                let reason = output.unavailable.unwrap_or(RetainedUnavailableReason::TooLarge);
                let (tombstone_bytes, terminal_stats) = {
                    let record =
                        state.records.get_mut(&output.key).expect("validated finalization record remains present");
                    record.phase = QueryPhase::Tombstone;
                    record.unavailable = Some(reason);
                    record.retention_deadline = None;
                    record.retention_order = None;
                    record.tombstone_deadline = Some(checked_deadline(now, self.config.tombstone_ttl));
                    record.retained = None;
                    record.tombstone_stats = output.tombstone_stats.take().or_else(|| {
                        unavailable_tombstone_stats(
                            output.key.query_id(),
                            terminal_outcome,
                            reason,
                            self.config.per_query_artifact_max_bytes,
                        )
                    });
                    let terminal_stats = record.tombstone_stats.as_ref().map(Arc::clone);
                    let tombstone_bytes =
                        terminal_stats.as_ref().map_or(0, |stats| u64::try_from(stats.len()).unwrap_or(u64::MAX));
                    (tombstone_bytes, terminal_stats)
                };
                state.tombstone_count = state.tombstone_count.checked_add(1).expect("tombstone count overflow");
                state.tombstone_bytes =
                    state.tombstone_bytes.checked_add(tombstone_bytes).expect("tombstone bytes overflow");
                terminal_stats
            },
            Some(bundle) => {
                state.next_retention_order =
                    state.next_retention_order.checked_add(1).expect("retention order overflow");
                let retention_order = state.next_retention_order;
                let record = state.records.get_mut(&output.key).expect("validated finalization record remains present");
                record.phase = QueryPhase::Retained;
                record.retention_deadline = Some(checked_deadline(now, self.config.retained_ttl));
                record.retention_order = Some(retention_order);
                record.tombstone_deadline = None;
                let terminal_stats = Arc::clone(&bundle.stats);
                record.retained = Some(bundle);
                record.unavailable = output.unavailable;
                record.tombstone_stats = None;
                state.retained_count = state.retained_count.checked_add(1).expect("retained count overflow");
                state.retained_bytes = state.retained_bytes.checked_add(charge_bytes).expect("retained bytes overflow");
                Some(terminal_stats)
            },
        };
        Self::enforce_retention_locked(&self.config, &mut state, now);
        if Self::live_count_locked(&state) == 0 {
            state.shutdown_deadline = None;
        }
        drop(state);
        output.live_stats_cache.publish_optional_terminal(terminal_stats);
        self.notify.notify_waiters();
    }
}
