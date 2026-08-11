//! Retained eviction, tombstone, and byte-cap accounting.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use super::{
    super::observability::attempt_outcome_label,
    QueryCoordinator,
    artifacts::{TOMBSTONE_STATS_MAX_BYTES, unavailable_tombstone_stats},
    registry::QueryRegistryState,
    types::{QueryCoordinatorConfig, QueryKey, QueryPhase, QueryTerminalOutcome, RetainedUnavailableReason},
};

pub(crate) fn checked_deadline(now: Instant, duration: Duration) -> Instant {
    now.checked_add(duration).unwrap_or(now)
}

impl QueryCoordinator {
    pub(crate) fn enforce_retention_locked(
        config: &QueryCoordinatorConfig,
        state: &mut QueryRegistryState,
        now: Instant,
    ) {
        while state.retained_count > config.retained_max_count || state.retained_bytes > config.retained_max_bytes {
            let victim = state
                .records
                .iter()
                .filter(|&(_, record)| record.phase == QueryPhase::Retained)
                .map(|(key, record)| {
                    (record.retention_order.expect("retained phase owns a retention order"), key.clone())
                })
                .min()
                .map(|(_, key)| key)
                .expect("retained pressure requires a retained victim");
            Self::evict_retained_locked(config, state, &victim, now);
        }
        let tombstone_max_bytes = (config.tombstone_max_count as u64).saturating_mul(TOMBSTONE_STATS_MAX_BYTES);
        while state.tombstone_count > config.tombstone_max_count || state.tombstone_bytes > tombstone_max_bytes {
            let victim = state
                .records
                .iter()
                .filter(|&(_, record)| record.phase == QueryPhase::Tombstone)
                .map(|(key, record)| (record.tombstone_deadline.expect("tombstone phase owns a deadline"), key.clone()))
                .min()
                .map(|(_, key)| key)
                .expect("tombstone pressure requires a tombstone victim");
            Self::remove_tombstone_locked(state, &victim);
        }
    }

    pub(crate) fn remove_tombstone_locked(state: &mut QueryRegistryState, key: &QueryKey) -> bool {
        let Some(record) = state.records.get(key) else { return false };
        if record.phase != QueryPhase::Tombstone {
            return false;
        }
        let bytes = record.tombstone_stats.as_ref().map_or(0, |stats| stats.len() as u64);
        state.records.remove(key);
        state.tombstone_count = state.tombstone_count.checked_sub(1).expect("tombstone count underflow");
        state.tombstone_bytes = state.tombstone_bytes.checked_sub(bytes).expect("tombstone bytes underflow");
        if state.current.get(key.query_id()).is_some_and(|current| current == key) {
            state.current.remove(key.query_id());
        }
        true
    }

    pub(crate) fn evict_retained_locked(
        config: &QueryCoordinatorConfig,
        state: &mut QueryRegistryState,
        key: &QueryKey,
        now: Instant,
    ) {
        let Some(record) = state.records.get_mut(key) else { return };
        if record.phase != QueryPhase::Retained {
            return;
        }
        let delivery_outcome = record.terminal_outcome.map_or("unknown", QueryTerminalOutcome::stats_outcome);
        let terminal_outcome = record.attempt_outcome.as_ref().map(attempt_outcome_label).unwrap_or(delivery_outcome);
        let delivery_outcome = (terminal_outcome != delivery_outcome).then_some(delivery_outcome);
        if let Some(bundle) = record.retained.take() {
            state.retained_bytes = state
                .retained_bytes
                .checked_sub(bundle.charge_bytes)
                .expect("retained eviction releases an installed bundle charge exactly once");
        }
        state.retained_count = state.retained_count.checked_sub(1).expect("retained count underflow");
        state.tombstone_count = state.tombstone_count.checked_add(1).expect("tombstone count overflow");
        record.phase = QueryPhase::Tombstone;
        record.unavailable = Some(RetainedUnavailableReason::Evicted);
        record.tombstone_stats = unavailable_tombstone_stats(
            key.query_id(),
            terminal_outcome,
            delivery_outcome,
            RetainedUnavailableReason::Evicted,
            config.per_query_artifact_max_bytes,
        );
        let tombstone_bytes = record.tombstone_stats.as_ref().map_or(0, |stats| stats.len() as u64);
        record.live_stats_cache.publish_optional_terminal(record.tombstone_stats.as_ref().map(Arc::clone));
        state.tombstone_bytes = state.tombstone_bytes.checked_add(tombstone_bytes).expect("tombstone bytes overflow");
        record.retention_deadline = None;
        record.retention_order = None;
        record.tombstone_deadline = Some(checked_deadline(now, config.tombstone_ttl));
    }
}
