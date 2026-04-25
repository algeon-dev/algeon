//! Retained eviction, tombstone, and byte-cap accounting.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use super::{
    QueryCoordinator,
    artifacts::{TOMBSTONE_STATS_MAX_BYTES, unavailable_tombstone_stats},
    registry::QueryRegistryState,
    types::{QueryCoordinatorConfig, QueryKey, QueryPhase, QueryTerminalOutcome, RetainedUnavailableReason},
};

pub(in crate::server) fn checked_deadline(now: Instant, duration: Duration) -> Instant {
    now.checked_add(duration).unwrap_or(now)
}

impl QueryCoordinator {
    pub(in crate::server) fn enforce_retention_locked(
        config: &QueryCoordinatorConfig,
        state: &mut QueryRegistryState,
        now: Instant,
    ) {
        while state.retained_count > config.retained_max_count || state.retained_bytes > config.retained_max_bytes {
            let victim = state
                .records
                .iter()
                .filter_map(|(key, record)| {
                    (record.phase == QueryPhase::Retained)
                        .then_some((record.retention_order.unwrap_or(u64::MAX), key.clone()))
                })
                .min()
                .map(|(_, key)| key)
                .expect("retained pressure requires a retained victim");
            Self::evict_retained_locked(config, state, &victim, now);
        }
        let tombstone_max_bytes =
            u64::try_from(config.tombstone_max_count).unwrap_or(u64::MAX).saturating_mul(TOMBSTONE_STATS_MAX_BYTES);
        while state.tombstone_count > config.tombstone_max_count || state.tombstone_bytes > tombstone_max_bytes {
            let victim = state
                .records
                .iter()
                .filter_map(|(key, record)| {
                    (record.phase == QueryPhase::Tombstone)
                        .then_some((record.tombstone_deadline.unwrap_or(now), key.clone()))
                })
                .min()
                .map(|(_, key)| key)
                .expect("tombstone pressure requires a tombstone victim");
            Self::remove_tombstone_locked(state, &victim);
        }
    }

    pub(in crate::server) fn remove_tombstone_locked(state: &mut QueryRegistryState, key: &QueryKey) -> bool {
        let Some(record) = state.records.get(key) else { return false };
        if record.phase != QueryPhase::Tombstone {
            return false;
        }
        let bytes = record.tombstone_stats.as_ref().map_or(0, |stats| u64::try_from(stats.len()).unwrap_or(u64::MAX));
        state.records.remove(key);
        state.tombstone_count = state.tombstone_count.checked_sub(1).expect("tombstone count underflow");
        state.tombstone_bytes = state.tombstone_bytes.checked_sub(bytes).expect("tombstone bytes underflow");
        if state.current.get(key.query_id()).is_some_and(|current| current == key) {
            state.current.remove(key.query_id());
        }
        true
    }

    pub(in crate::server) fn evict_retained_locked(
        config: &QueryCoordinatorConfig,
        state: &mut QueryRegistryState,
        key: &QueryKey,
        now: Instant,
    ) {
        let Some(record) = state.records.get_mut(key) else { return };
        if record.phase != QueryPhase::Retained {
            return;
        }
        let terminal_outcome = record.terminal_outcome.map_or("unknown", QueryTerminalOutcome::stats_outcome);
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
            RetainedUnavailableReason::Evicted,
            config.per_query_artifact_max_bytes,
        );
        let tombstone_bytes =
            record.tombstone_stats.as_ref().map_or(0, |stats| u64::try_from(stats.len()).unwrap_or(u64::MAX));
        record.live_stats_cache.publish_optional_terminal(record.tombstone_stats.as_ref().map(Arc::clone));
        state.tombstone_bytes = state.tombstone_bytes.checked_add(tombstone_bytes).expect("tombstone bytes overflow");
        record.retention_deadline = None;
        record.retention_order = None;
        record.tombstone_deadline = Some(checked_deadline(now, config.tombstone_ttl));
    }
}
