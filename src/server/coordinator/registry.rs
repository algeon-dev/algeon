//! Query record and registry lookup/registration helpers.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Instant,
};

use nexus_query_engine::{
    observability::{RuntimeObservationConfig, RuntimeObservationOwner, RuntimeQueryId, RuntimeTraceId},
    runtime::ExecutionCancellation,
};
use uuid::Uuid;

#[cfg(test)]
use super::types::CoordinatorArtifact;
use super::{
    super::observability::{CapturedError, CapturedQuery},
    QueryCoordinator,
    artifacts::LiveStatsCache,
    ticket::QueryPlanningLease,
    types::{
        CoordinatorTicketPayload, ObservationId, QueryCoordinatorError, QueryCoordinatorErrorKind, QueryHotCounters,
        QueryKey, QueryPhase, QueryTerminal, QueryTerminalOutcome, RetainedQueryBundle, RetainedUnavailableReason,
    },
};

#[derive(Debug)]
pub(in crate::server) struct QueryRecord {
    pub(in crate::server) key: QueryKey,
    pub(in crate::server) observation_id: ObservationId,
    pub(in crate::server) captured: Option<CapturedQuery>,
    pub(in crate::server) phase: QueryPhase,
    pub(in crate::server) terminal: Option<QueryTerminal>,
    pub(in crate::server) terminal_outcome: Option<QueryTerminalOutcome>,
    pub(in crate::server) active_leases: usize,
    pub(in crate::server) active_activities: usize,
    pub(in crate::server) observation_owner: Option<RuntimeObservationOwner>,
    pub(in crate::server) hot: Arc<QueryHotCounters>,
    pub(in crate::server) logical_capture_bytes: u64,
    pub(in crate::server) artifact_max_bytes: u64,
    pub(in crate::server) capture_version: u64,
    pub(in crate::server) live_stats_cache: Arc<LiveStatsCache>,
    pub(in crate::server) ticket: Option<CoordinatorTicketPayload>,
    pub(in crate::server) execution_cancellation: Option<ExecutionCancellation>,
    pub(in crate::server) retained: Option<Arc<RetainedQueryBundle>>,
    pub(in crate::server) unavailable: Option<RetainedUnavailableReason>,
    pub(in crate::server) retention_deadline: Option<Instant>,
    pub(in crate::server) retention_order: Option<u64>,
    pub(in crate::server) tombstone_deadline: Option<Instant>,
    pub(in crate::server) tombstone_stats: Option<Arc<[u8]>>,
    pub(in crate::server) finalization_claim: Option<u64>,
}

#[derive(Debug, Default)]
pub(in crate::server) struct QueryRegistryState {
    pub(in crate::server) next_generation: u64,
    pub(in crate::server) current: HashMap<String, QueryKey>,
    pub(in crate::server) records: BTreeMap<QueryKey, QueryRecord>,
    pub(in crate::server) accepting: bool,
    pub(in crate::server) active_logical_capture_bytes: u64,
    pub(in crate::server) ticketed_count: usize,
    pub(in crate::server) retained_count: usize,
    pub(in crate::server) retained_bytes: u64,
    pub(in crate::server) tombstone_count: usize,
    pub(in crate::server) tombstone_bytes: u64,
    pub(in crate::server) next_retention_order: u64,
    pub(in crate::server) shutdown_deadline: Option<Instant>,
    pub(in crate::server) shutdown_forced: bool,
}

impl QueryCoordinator {
    pub(in crate::server) fn register_new(
        self: &Arc<Self>,
        sql: impl Into<String>,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<(String, QueryPlanningLease), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let lease = self
            .register_locked(&mut state, query_id.clone(), sql.into(), correlation_id, trace_id, observation_config)
            .map_err(|error| error.with_query_id(query_id.clone()))?;
        Ok((query_id, lease))
    }

    #[cfg(test)]
    pub(in crate::server) fn register_with_id(
        self: &Arc<Self>,
        query_id: impl Into<String>,
        sql: impl Into<String>,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<QueryPlanningLease, QueryCoordinatorError> {
        let query_id = query_id.into();
        let mut state = self.lock_state();
        self.register_locked(&mut state, query_id.clone(), sql.into(), correlation_id, trace_id, observation_config)
            .map_err(|error| error.with_query_id(query_id))
    }

    pub(in crate::server) fn register_locked(
        self: &Arc<Self>,
        state: &mut QueryRegistryState,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<QueryPlanningLease, QueryCoordinatorError> {
        let runtime_capture_bytes = observation_config.limits().max_retained_bytes;
        let logical_capture_bytes = runtime_capture_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(self.config.per_query_capture_max_bytes))
            .and_then(|bytes| bytes.checked_add(self.config.per_query_artifact_max_bytes))
            .ok_or_else(|| {
                QueryCoordinatorError::new(
                    QueryCoordinatorErrorKind::LogicalCaptureBytes,
                    "per-query observation capture reservation overflows u64",
                )
            })?;
        if !state.accepting {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::RegistrationCapacity,
                "query registration is closed",
            ));
        }
        if state.current.get(&query_id).and_then(|key| state.records.get(key)).is_some_and(|record| {
            matches!(
                record.phase,
                QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
            )
        }) {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::DuplicateLiveQueryId,
                "query id is already registered to a live query",
            ));
        }
        let live_count = state
            .records
            .values()
            .filter(|record| {
                matches!(
                    record.phase,
                    QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                )
            })
            .count();
        if live_count >= self.config.max_live_queries {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::RegistrationCapacity,
                "maximum live query count reached",
            ));
        }
        let Some(next_active_capture_bytes) = state.active_logical_capture_bytes.checked_add(logical_capture_bytes)
        else {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::LogicalCaptureBytes,
                "active logical observation capture accounting overflowed",
            ));
        };
        if next_active_capture_bytes > self.config.max_active_logical_capture_bytes {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::LogicalCaptureBytes,
                "active logical observation capture budget exhausted",
            ));
        }
        let generation = state.next_generation.checked_add(1).ok_or_else(|| {
            QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::GenerationExhausted,
                "query generation space exhausted",
            )
        })?;
        state.next_generation = generation;
        let key = QueryKey { query_id: Arc::from(query_id.as_str()), generation };
        let observation_id = ObservationId(Arc::from(format!("{query_id}:{generation}")));
        let owner = RuntimeObservationOwner::new(
            RuntimeQueryId::new(query_id.clone()),
            trace_id.map(RuntimeTraceId::new),
            observation_config,
        );
        let captured = CapturedQuery::new(
            query_id.clone(),
            sql,
            correlation_id,
            Instant::now(),
            self.config.per_query_capture_max_bytes,
            observation_config.limits().max_retained_records,
        );
        let hot = Arc::new(QueryHotCounters::default());
        state.active_logical_capture_bytes = next_active_capture_bytes;
        state.current.insert(query_id.clone(), key.clone());
        state.records.insert(
            key.clone(),
            QueryRecord {
                key: key.clone(),
                observation_id: observation_id.clone(),
                captured: Some(captured),
                phase: QueryPhase::Planning,
                terminal: None,
                terminal_outcome: None,
                active_leases: 1,
                active_activities: 0,
                observation_owner: Some(owner),
                hot: Arc::clone(&hot),
                logical_capture_bytes,
                artifact_max_bytes: self.config.per_query_artifact_max_bytes,
                capture_version: 0,
                live_stats_cache: Arc::new(LiveStatsCache::new()),
                ticket: None,
                execution_cancellation: None,
                retained: None,
                unavailable: None,
                retention_deadline: None,
                retention_order: None,
                tombstone_deadline: None,
                tombstone_stats: None,
                finalization_claim: None,
            },
        );
        Ok(QueryPlanningLease { coordinator: Arc::downgrade(self), key, observation_id, hot, released: false })
    }

    pub(in crate::server) fn live_count_locked(state: &QueryRegistryState) -> usize {
        state
            .records
            .values()
            .filter(|record| {
                matches!(
                    record.phase,
                    QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                )
            })
            .count()
    }

    pub(in crate::server) fn record_mut<'a>(
        state: &'a mut QueryRegistryState,
        key: &QueryKey,
    ) -> Result<&'a mut QueryRecord, QueryCoordinatorError> {
        let current = state.current.get(key.query_id()).ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "unknown query generation")
        })?;
        if current != key {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::StaleGeneration,
                "stale query generation",
            ));
        }
        state.records.get_mut(key).ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "unknown query generation")
        })
    }

    pub(in crate::server) fn cancel_ticket(&self, query_id: &str, error: CapturedError) -> bool {
        let cancelled = {
            let mut state = self.lock_state();
            let Some(key) = state.current.get(query_id).cloned() else { return false };
            let Some(record) = state.records.get_mut(&key) else { return false };
            if record.phase != QueryPhase::Ticketed {
                return false;
            }
            if let Some(ticket) = record.ticket.take() {
                ticket.native_cancellation.cancel();
            }
            if record.terminal_outcome.is_none() {
                let mut terminal = QueryTerminal::Cancelled { stage: "cancel_query", error: Arc::new(error) };
                Self::apply_terminal_to_capture(record, &mut terminal, Instant::now());
                record.terminal_outcome = Some(terminal.compact_outcome());
                record.terminal = Some(terminal);
            }
            record.phase = QueryPhase::Finalizing;
            state.ticketed_count = state.ticketed_count.checked_sub(1).expect("cancelled ticket count underflow");
            true
        };
        if cancelled {
            self.notify.notify_waiters();
            self.finalize_ready();
        }
        cancelled
    }

    #[cfg(all(test, feature = "iceberg"))]
    pub(in crate::server) fn ticket_query_ids(&self) -> Vec<String> {
        let state = self.lock_state();
        state
            .records
            .values()
            .filter(|record| record.phase == QueryPhase::Ticketed)
            .map(|record| record.key.query_id().to_string())
            .collect()
    }

    #[cfg(test)]
    pub(in crate::server) fn ticket_correlation_id(&self, query_id: &str) -> Option<Option<String>> {
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        let record = state.records.get(key)?;
        if record.phase != QueryPhase::Ticketed {
            return None;
        }
        Some(record.ticket.as_ref()?.correlation_id.clone())
    }

    #[cfg(test)]
    pub(in crate::server) fn ticket_cancellation(&self, query_id: &str) -> Option<ExecutionCancellation> {
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        let record = state.records.get(key)?;
        if record.phase != QueryPhase::Ticketed {
            return None;
        }
        Some(record.ticket.as_ref()?.native_cancellation.clone())
    }

    /// Close registration, record a server-shutdown cancellation cause, and
    /// request cancellation for every ticketed or executing native query.
    pub(in crate::server) fn apply_terminal_to_capture(
        record: &mut QueryRecord,
        terminal: &mut QueryTerminal,
        now: Instant,
    ) {
        match terminal {
            QueryTerminal::Succeeded => {
                if let Some(captured) = &mut record.captured {
                    captured.finish_success(now);
                }
            },
            QueryTerminal::Failed { stage, error, .. } | QueryTerminal::Cancelled { stage, error } => {
                if let Some(captured) = &mut record.captured {
                    captured.bound_error(Arc::make_mut(error));
                    captured.finish_failed(stage, Arc::clone(error), now);
                }
            },
            QueryTerminal::Dropped | QueryTerminal::Panicked => {
                if let Some(captured) = &mut record.captured {
                    captured.finish_dropped(now);
                }
            },
        }
    }

    pub(in crate::server) fn bump_capture_version(record: &mut QueryRecord) {
        record.capture_version = record.capture_version.checked_add(1).unwrap_or_else(|| {
            if let Some(captured) = &mut record.captured {
                captured.mark_counter_overflow();
            }
            u64::MAX
        });
    }

    #[cfg(test)]
    pub(in crate::server) fn artifact(&self, query_id: &str) -> Option<CoordinatorArtifact> {
        self.finalize_ready();
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        let record = state.records.get(key)?;
        Some(match record.phase {
            QueryPhase::Planning | QueryPhase::Ticketed => CoordinatorArtifact::Pending,
            QueryPhase::Executing => CoordinatorArtifact::Active,
            QueryPhase::Finalizing => CoordinatorArtifact::Finalizing,
            QueryPhase::Retained => match record.unavailable {
                Some(reason) => CoordinatorArtifact::Unavailable(reason),
                None => CoordinatorArtifact::Retained(Arc::clone(
                    record.retained.as_ref().expect("retained phase owns a terminal bundle"),
                )),
            },
            QueryPhase::Tombstone => {
                CoordinatorArtifact::Unavailable(record.unavailable.unwrap_or(RetainedUnavailableReason::Evicted))
            },
        })
    }

    #[cfg(test)]
    pub(in crate::server) fn terminal(&self, query_id: &str) -> Option<QueryTerminalOutcome> {
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        state.records.get(key)?.terminal_outcome
    }
}
