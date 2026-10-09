//! Finalization claim, unlocked render, and locked install paths.

#[cfg(test)]
use std::sync::atomic::Ordering;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Instant,
};

use query_runtime::{
    attempt::AttemptCancellation,
    observability::{
        QueryAttemptReport, RuntimeTerminalOutcome, SealedRuntimeObservationSnapshot, write_runtime_observation_jsonl,
    },
};

#[cfg(test)]
use super::artifacts::{FinalizationTestGate, run_coordinator_render_probe};
use super::{
    super::observability::{CapturedQuery, ShutdownCaptureSnapshot, attempt_outcome_label},
    QueryCoordinator,
    artifacts::{
        LimitedVecWriter, LiveStatsCache, compose_terminal_artifact, incomplete_runtime_control,
        sealed_runtime_control, unavailable_stats_payload, unavailable_tombstone_stats,
    },
    retention::checked_deadline,
    types::{
        CoordinatorTicketPayload, QueryHotCounterSnapshot, QueryKey, QueryPhase, QueryTerminal, QueryTerminalOutcome,
        RetainedQueryBundle, RetainedUnavailableReason,
    },
};

pub(crate) struct FinalizationInputs {
    pub(crate) key: QueryKey,
    pub(crate) claim: u64,
    pub(crate) terminal_outcome: QueryTerminalOutcome,
    pub(crate) attempt_outcome: Option<RuntimeTerminalOutcome>,
    /// `None` when the query registered without a capture.
    pub(crate) captured: Option<CapturedQuery>,
    pub(crate) attempt_owned: bool,
    pub(crate) attempt_report: Option<Arc<QueryAttemptReport>>,
    pub(crate) attempt_snapshot: Option<SealedRuntimeObservationSnapshot>,
    pub(crate) live_stats_cache: Arc<LiveStatsCache>,
    pub(crate) hot: QueryHotCounterSnapshot,
    pub(crate) artifact_max_bytes: u64,
    pub(crate) preselected_unavailable: Option<RetainedUnavailableReason>,
    #[cfg(test)]
    pub(crate) finalization_gate: Option<Arc<FinalizationTestGate>>,
}

pub(crate) struct FinalizationFailureContext {
    pub(crate) key: QueryKey,
    pub(crate) claim: u64,
    pub(crate) live_stats_cache: Arc<LiveStatsCache>,
    pub(crate) outcome: &'static str,
    pub(crate) delivery_outcome: Option<&'static str>,
    pub(crate) artifact_max_bytes: u64,
}

impl FinalizationFailureContext {
    pub(crate) fn from_inputs(inputs: &FinalizationInputs) -> Self {
        let delivery_outcome = inputs.terminal_outcome.stats_outcome();
        let outcome = inputs.attempt_outcome.as_ref().map(attempt_outcome_label).unwrap_or(delivery_outcome);
        Self {
            key: inputs.key.clone(),
            claim: inputs.claim,
            live_stats_cache: Arc::clone(&inputs.live_stats_cache),
            outcome,
            delivery_outcome: (outcome != delivery_outcome).then_some(delivery_outcome),
            artifact_max_bytes: inputs.artifact_max_bytes,
        }
    }
}

pub(crate) struct ShutdownTimeoutSnapshot {
    pub(crate) key: QueryKey,
    pub(crate) captured: Option<ShutdownCaptureSnapshot>,
    pub(crate) attempt_outcome: Option<RuntimeTerminalOutcome>,
    pub(crate) hot: QueryHotCounterSnapshot,
    pub(crate) artifact_max_bytes: u64,
    pub(crate) live_stats_cache: Arc<LiveStatsCache>,
}

pub(crate) struct ShutdownTimeoutOutput {
    pub(crate) key: QueryKey,
    pub(crate) live_stats_cache: Arc<LiveStatsCache>,
    pub(crate) bundle: Option<Arc<RetainedQueryBundle>>,
    pub(crate) tombstone_stats: Option<Arc<[u8]>>,
}

pub(crate) struct ShutdownDiscarded {
    pub(crate) _captured: Option<CapturedQuery>,
    pub(crate) _terminal: Option<QueryTerminal>,
    pub(crate) _ticket: Option<CoordinatorTicketPayload>,
    pub(crate) _execution_cancellation: Option<AttemptCancellation>,
}

pub(crate) struct FinalizationOutput {
    pub(crate) key: QueryKey,
    pub(crate) claim: u64,
    pub(crate) live_stats_cache: Arc<LiveStatsCache>,
    pub(crate) bundle: Option<Arc<RetainedQueryBundle>>,
    pub(crate) unavailable: Option<RetainedUnavailableReason>,
    pub(crate) tombstone_stats: Option<Arc<[u8]>>,
}

impl FinalizationOutput {
    pub(crate) fn unavailable(
        key: QueryKey,
        claim: u64,
        live_stats_cache: Arc<LiveStatsCache>,
        outcome: &'static str,
        delivery_outcome: Option<&'static str>,
        reason: RetainedUnavailableReason,
        artifact_max_bytes: u64,
    ) -> Self {
        let control = incomplete_runtime_control(key.query_id(), reason);
        let composed = compose_terminal_artifact(control, None, artifact_max_bytes, |remaining| {
            unavailable_stats_payload(key.query_id(), outcome, delivery_outcome, reason, remaining)
        });
        let bundle = composed.map(|artifact| artifact.bundle);
        let tombstone_stats = bundle
            .is_none()
            .then(|| unavailable_tombstone_stats(key.query_id(), outcome, delivery_outcome, reason, artifact_max_bytes))
            .flatten();
        Self { key, claim, live_stats_cache, bundle, unavailable: Some(reason), tombstone_stats }
    }

    pub(crate) fn panicked(context: FinalizationFailureContext) -> Self {
        Self::unavailable(
            context.key,
            context.claim,
            context.live_stats_cache,
            context.outcome,
            context.delivery_outcome,
            RetainedUnavailableReason::FinalizationPanicked,
            context.artifact_max_bytes,
        )
    }
}

impl QueryCoordinator {
    pub(crate) fn finalize_ready(&self) {
        while let Some(inputs) = self.take_finalization_inputs() {
            let failure = FinalizationFailureContext::from_inputs(&inputs);
            let output = catch_unwind(AssertUnwindSafe(|| Self::finalize_outside_lock(inputs)))
                .unwrap_or_else(|_| FinalizationOutput::panicked(failure));
            self.install_finalization(output, Instant::now());
        }
    }

    pub(crate) fn take_finalization_inputs(&self) -> Option<FinalizationInputs> {
        #[cfg(test)]
        let finalization_gate =
            self.finalization_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let mut state = self.lock_state();
        let key = state.records.iter().find_map(|(key, record)| {
            (record.terminal_outcome.is_some()
                && record.active_leases == 0
                && (!record.requires_attempt_report || record.attempt_report.is_some())
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
        // Only the compact outcome is kept; release the payload before `wait_drained` blocks.
        drop(terminal);
        let live_stats_cache = Arc::clone(&record.live_stats_cache);
        live_stats_cache.close();
        let inputs = FinalizationInputs {
            key,
            claim,
            terminal_outcome,
            attempt_outcome: record.attempt_outcome.clone(),
            captured: record.captured.take(),
            attempt_owned: record.attempt_owned,
            attempt_report: record.attempt_report.take(),
            attempt_snapshot: record.attempt_snapshot.take(),
            live_stats_cache: Arc::clone(&live_stats_cache),
            hot: record.hot.snapshot(),
            artifact_max_bytes: record.artifact_max_bytes,
            preselected_unavailable: record.unavailable,
            #[cfg(test)]
            finalization_gate,
        };
        // Renderers drain without the state lock; holding it would stall every coordinator call.
        drop(state);
        live_stats_cache.wait_drained();
        Some(inputs)
    }

    pub(crate) fn finalize_outside_lock(mut inputs: FinalizationInputs) -> FinalizationOutput {
        #[cfg(test)]
        if let Some(gate) = &inputs.finalization_gate {
            gate.wait();
        }
        #[cfg(test)]
        run_coordinator_render_probe("finalization projection");
        let delivery_outcome = inputs.terminal_outcome.stats_outcome();
        let stats_outcome = inputs.attempt_outcome.as_ref().map(attempt_outcome_label).unwrap_or(delivery_outcome);
        let stats_delivery_outcome = (stats_outcome != delivery_outcome).then_some(delivery_outcome);
        // Flight receives the attempt's snapshot only after the attempt has
        // sealed it, and the coordinator projects that immutable result. A
        // coordinator-only statement has no runtime observation; its stats
        // come from the coordinator's own capture alone.
        let snapshot = inputs.attempt_report.as_ref().and_then(|_| inputs.attempt_snapshot.take());
        let mut captured = match (inputs.captured.take(), &snapshot) {
            (Some(captured), Some(_)) => captured,
            (Some(captured), None) if !inputs.attempt_owned => captured,
            _ => {
                return FinalizationOutput::unavailable(
                    inputs.key,
                    inputs.claim,
                    inputs.live_stats_cache,
                    stats_outcome,
                    stats_delivery_outcome,
                    inputs.preselected_unavailable.unwrap_or(RetainedUnavailableReason::CaptureDisabled),
                    inputs.artifact_max_bytes,
                );
            },
        };
        captured.set_output_metrics(
            inputs.hot.output_batches,
            inputs.hot.output_rows,
            inputs.hot.output_bytes,
            inputs.hot.overflow,
        );
        captured.seal(Instant::now());
        let footer = snapshot.as_ref().map(SealedRuntimeObservationSnapshot::footer);
        let mut incomplete_reason = footer
            .is_some_and(|footer| footer.counter_overflow)
            .then_some(RetainedUnavailableReason::CounterOverflow)
            .or_else(|| inputs.hot.overflow.then_some(RetainedUnavailableReason::CounterOverflow))
            .or(inputs.preselected_unavailable)
            .or_else(|| {
                footer
                    .is_some_and(|footer| footer.truncated || footer.degraded)
                    .then_some(RetainedUnavailableReason::CaptureTruncated)
            });
        let control = sealed_runtime_control(
            inputs.key.query_id(),
            inputs.attempt_report.as_deref(),
            snapshot.is_some(),
            incomplete_reason,
        );
        let jsonl = match &snapshot {
            Some(snapshot) => {
                let Some(jsonl_limit) = inputs.artifact_max_bytes.checked_sub(control.len() as u64) else {
                    return FinalizationOutput::unavailable(
                        inputs.key,
                        inputs.claim,
                        inputs.live_stats_cache,
                        stats_outcome,
                        stats_delivery_outcome,
                        RetainedUnavailableReason::TooLarge,
                        inputs.artifact_max_bytes,
                    );
                };
                let mut jsonl = LimitedVecWriter::new(usize::try_from(jsonl_limit).unwrap_or(usize::MAX));
                if write_runtime_observation_jsonl(snapshot, &mut jsonl).is_err() {
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
                        stats_delivery_outcome,
                        reason,
                        inputs.artifact_max_bytes,
                    );
                }
                Some(Arc::<[u8]>::from(jsonl.bytes))
            },
            None => None,
        };
        let Some(mut composed) =
            compose_terminal_artifact(control, jsonl.clone(), inputs.artifact_max_bytes, |remaining| {
                captured.to_query_stats_payload(remaining)
            })
        else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                stats_delivery_outcome,
                RetainedUnavailableReason::TooLarge,
                inputs.artifact_max_bytes,
            );
        };
        if incomplete_reason.is_none() && composed.stats_incomplete {
            incomplete_reason = Some(RetainedUnavailableReason::CaptureTruncated);
            let control = sealed_runtime_control(
                inputs.key.query_id(),
                inputs.attempt_report.as_deref(),
                jsonl.is_some(),
                Some(RetainedUnavailableReason::CaptureTruncated),
            );
            let Some(recomposed) = compose_terminal_artifact(control, jsonl, inputs.artifact_max_bytes, |remaining| {
                captured.to_query_stats_payload(remaining)
            }) else {
                return FinalizationOutput::unavailable(
                    inputs.key,
                    inputs.claim,
                    inputs.live_stats_cache,
                    stats_outcome,
                    stats_delivery_outcome,
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

    pub(crate) fn install_finalization(&self, mut output: FinalizationOutput, now: Instant) {
        let mut state = self.lock_state();
        let Some(record) = state.records.get(&output.key) else {
            // `abandon` locks only the live stats cache; keep it out of the state lock window.
            drop(state);
            output.live_stats_cache.abandon();
            return;
        };
        if record.phase != QueryPhase::Finalizing
            || record.key != output.key
            || record.finalization_claim != Some(output.claim)
        {
            // `abandon` locks only the live stats cache; keep it out of the state lock window.
            drop(state);
            output.live_stats_cache.abandon();
            return;
        }
        #[cfg(test)]
        self.finalization_installs.fetch_add(1, Ordering::SeqCst);
        let logical_capture_bytes = record.logical_capture_bytes;
        let delivery_outcome = record.terminal_outcome.map_or("unknown", QueryTerminalOutcome::stats_outcome);
        let terminal_outcome = record.attempt_outcome.as_ref().map(attempt_outcome_label).unwrap_or(delivery_outcome);
        let terminal_delivery_outcome = (terminal_outcome != delivery_outcome).then_some(delivery_outcome);
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
                terminal_delivery_outcome,
                RetainedUnavailableReason::TooLarge,
                self.config.per_query_artifact_max_bytes,
            );
        }
        let terminal_stats = match output.bundle.take() {
            None => {
                let reason = output.unavailable.expect("missing bundle carries an unavailable reason");
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
                            terminal_delivery_outcome,
                            reason,
                            self.config.per_query_artifact_max_bytes,
                        )
                    });
                    let terminal_stats = record.tombstone_stats.as_ref().map(Arc::clone);
                    let tombstone_bytes = terminal_stats.as_ref().map_or(0, |stats| stats.len() as u64);
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
        // Publish and notify outside the state lock; woken coordinator waiters lock it again.
        drop(state);
        output.live_stats_cache.publish_optional_terminal(terminal_stats);
        self.notify.notify_waiters();
    }
}
