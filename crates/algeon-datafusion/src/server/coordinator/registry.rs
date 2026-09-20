//! Query record and registry lookup/registration helpers.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Instant,
};

use query_engine::{
    observability::{
        QueryAttemptReport, RuntimeObservationConfig, RuntimeObservationOwner, RuntimeQueryId, RuntimeTerminalOutcome,
        RuntimeTraceId, SealedRuntimeObservationSnapshot,
    },
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
use crate::attempt::server_internal::AlgeonAttemptServerExt;

#[derive(Debug)]
pub(crate) struct QueryRecord {
    pub(crate) key: QueryKey,
    pub(crate) observation_id: ObservationId,
    pub(crate) captured: Option<CapturedQuery>,
    pub(crate) phase: QueryPhase,
    pub(crate) terminal: Option<QueryTerminal>,
    pub(crate) terminal_outcome: Option<QueryTerminalOutcome>,
    pub(crate) active_leases: usize,
    pub(crate) active_activities: usize,
    pub(crate) observation_owner: Option<RuntimeObservationOwner>,
    /// Flight attempt reports are immutable handoffs from the actual attempt
    /// owner. `None` remains valid when an attempt never begins, but records
    /// with an attempt cannot finalize until this arrives.
    pub(crate) attempt_report: Option<Arc<QueryAttemptReport>>,
    /// The canonical attempt outcome survives handoff of the full immutable
    /// report so unavailable and tombstone projections never reconstruct it
    /// from Flight delivery state.
    pub(crate) attempt_outcome: Option<RuntimeTerminalOutcome>,
    /// The immutable snapshot sealed by the same attempt as
    /// `attempt_report`. This is retained only for artifact projection; the
    /// coordinator never receives observation-sealing authority.
    pub(crate) attempt_snapshot: Option<SealedRuntimeObservationSnapshot>,
    pub(crate) requires_attempt_report: bool,
    pub(crate) hot: Arc<QueryHotCounters>,
    pub(crate) logical_capture_bytes: u64,
    pub(crate) artifact_max_bytes: u64,
    pub(crate) capture_version: u64,
    pub(crate) live_stats_cache: Arc<LiveStatsCache>,
    pub(crate) ticket: Option<CoordinatorTicketPayload>,
    pub(crate) execution_cancellation: Option<ExecutionCancellation>,
    pub(crate) retained: Option<Arc<RetainedQueryBundle>>,
    pub(crate) unavailable: Option<RetainedUnavailableReason>,
    pub(crate) retention_deadline: Option<Instant>,
    pub(crate) retention_order: Option<u64>,
    pub(crate) tombstone_deadline: Option<Instant>,
    pub(crate) tombstone_stats: Option<Arc<[u8]>>,
    pub(crate) finalization_claim: Option<u64>,
}

#[derive(Debug, Default)]
pub(crate) struct QueryRegistryState {
    pub(crate) next_generation: u64,
    pub(crate) current: HashMap<String, QueryKey>,
    pub(crate) records: BTreeMap<QueryKey, QueryRecord>,
    pub(crate) accepting: bool,
    pub(crate) active_logical_capture_bytes: u64,
    pub(crate) ticketed_count: usize,
    pub(crate) retained_count: usize,
    pub(crate) retained_bytes: u64,
    pub(crate) tombstone_count: usize,
    pub(crate) tombstone_bytes: u64,
    pub(crate) next_retention_order: u64,
    pub(crate) shutdown_deadline: Option<Instant>,
    pub(crate) shutdown_forced: bool,
}

impl QueryCoordinator {
    pub(crate) fn register_new(
        self: &Arc<Self>,
        sql: impl Into<String>,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<(String, QueryPlanningLease), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let lease = self
            .register_locked(
                &mut state,
                query_id.clone(),
                sql.into(),
                correlation_id,
                trace_id,
                observation_config,
                true,
            )
            .map_err(|error| error.with_query_id(query_id.clone()))?;
        Ok((query_id, lease))
    }

    /// Registers a Flight query whose runtime observation is owned by the
    /// invocation's `AlgeonAttempt`. The coordinator still retains planning,
    /// cancellation, and artifact state, but never receives a second sealing
    /// authority for the same execution.
    pub(crate) fn register_new_with_attempt_observation(
        self: &Arc<Self>,
        sql: impl Into<String>,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<(String, QueryPlanningLease), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let lease = self
            .register_locked(
                &mut state,
                query_id.clone(),
                sql.into(),
                correlation_id,
                trace_id,
                observation_config,
                false,
            )
            .map_err(|error| error.with_query_id(query_id.clone()))?;
        Ok((query_id, lease))
    }

    #[cfg(test)]
    pub(crate) fn register_with_id(
        self: &Arc<Self>,
        query_id: impl Into<String>,
        sql: impl Into<String>,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<QueryPlanningLease, QueryCoordinatorError> {
        let query_id = query_id.into();
        let mut state = self.lock_state();
        self.register_locked(
            &mut state,
            query_id.clone(),
            sql.into(),
            correlation_id,
            trace_id,
            observation_config,
            true,
        )
        .map_err(|error| error.with_query_id(query_id))
    }

    #[cfg(test)]
    pub(crate) fn register_with_id_with_attempt_observation(
        self: &Arc<Self>,
        query_id: impl Into<String>,
        sql: impl Into<String>,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<QueryPlanningLease, QueryCoordinatorError> {
        let query_id = query_id.into();
        let mut state = self.lock_state();
        self.register_locked(
            &mut state,
            query_id.clone(),
            sql.into(),
            correlation_id,
            trace_id,
            observation_config,
            false,
        )
        .map_err(|error| error.with_query_id(query_id))
    }

    #[expect(clippy::too_many_arguments, reason = "service wiring parameters are distinct ownership values")]
    pub(crate) fn register_locked(
        self: &Arc<Self>,
        state: &mut QueryRegistryState,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
        create_runtime_observation_owner: bool,
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
        let owner = create_runtime_observation_owner.then(|| {
            RuntimeObservationOwner::new(
                RuntimeQueryId::new(query_id.clone()),
                trace_id.map(RuntimeTraceId::new),
                observation_config,
            )
        });
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
        state.current.insert(query_id, key.clone());
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
                observation_owner: owner,
                attempt_report: None,
                attempt_outcome: None,
                attempt_snapshot: None,
                requires_attempt_report: !create_runtime_observation_owner,
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
        Ok(QueryPlanningLease {
            coordinator: Arc::downgrade(self),
            key,
            observation_id,
            hot,
            attempt_report_handoff_registered: std::cell::Cell::new(false),
            released: false,
        })
    }

    /// Whether `query_id` still resolves to a record. False once the record
    /// has been evicted, so callers that key state off a query id can release
    /// it on the same schedule the coordinator releases the query itself.
    pub(crate) fn knows_query(&self, query_id: &str) -> bool {
        self.lock_state().current.contains_key(query_id)
    }

    pub(crate) fn live_count_locked(state: &QueryRegistryState) -> usize {
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

    /// Attach the report published by the single execution-attempt owner.
    /// This is a retained read-only projection: the coordinator has no
    /// terminalization or observation-sealing authority over the report.
    pub(crate) fn attach_attempt_report(
        &self,
        key: &QueryKey,
        report: Arc<QueryAttemptReport>,
        snapshot: Option<SealedRuntimeObservationSnapshot>,
    ) -> Result<(), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let record = Self::record_mut(&mut state, key)?;
        if !record.requires_attempt_report {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "query does not accept an attempt-owned report",
            ));
        }
        if let Some(existing) = &record.attempt_report {
            if existing.attempt_id != report.attempt_id {
                return Err(QueryCoordinatorError::new(
                    QueryCoordinatorErrorKind::InvalidTransition,
                    "query already has a different attempt report",
                ));
            }
            return Ok(());
        }
        if let Some(snapshot) = snapshot.as_ref() {
            let Some(snapshot_report) = snapshot.attempt_report() else {
                return Err(QueryCoordinatorError::new(
                    QueryCoordinatorErrorKind::InvalidTransition,
                    "attempt observation snapshot omitted its immutable attempt report",
                ));
            };
            if snapshot_report != report.as_ref() || snapshot.footer().terminal.as_ref() != Some(&report.outcome) {
                return Err(QueryCoordinatorError::new(
                    QueryCoordinatorErrorKind::InvalidTransition,
                    "attempt observation snapshot report is not identical to its completed attempt",
                ));
            }
        }
        if let Some(captured) = &mut record.captured {
            captured.set_attempt_report(Arc::clone(&report));
        }
        record.attempt_outcome = Some(report.outcome.clone());
        record.attempt_report = Some(report);
        record.attempt_snapshot = snapshot;
        Self::bump_capture_version(record);
        drop(state);
        self.finalize_ready();
        Ok(())
    }

    /// Allows a begin-attempt rejection to finalize its coordinator record.
    /// No attempt exists in that one backend-closing/error window, so there is
    /// no immutable completion to await; the retained artifact is explicitly
    /// marked unavailable instead of waiting forever.
    pub(crate) fn allow_missing_attempt_report(&self, query_id: &str) {
        let mut state = self.lock_state();
        let Some(key) = state.current.get(query_id).cloned() else { return };
        let Some(record) = state.records.get_mut(&key) else { return };
        record.requires_attempt_report = false;
    }

    /// Stops awaiting an attempt report that sealed successfully but could not
    /// be retained by the coordinator. This records a delivery failure; it
    /// does not invent a coordinator-owned attempt outcome.
    pub(crate) fn abandon_attempt_report_delivery(
        &self,
        key: &QueryKey,
        attempt_outcome: Option<RuntimeTerminalOutcome>,
    ) {
        let should_finalize = {
            let mut state = self.lock_state();
            let Ok(record) = Self::record_mut(&mut state, key) else { return };
            // A rejected full handoff must not retain the report or snapshot,
            // but the already-sealed attempt outcome remains the canonical
            // compact projection for unavailable/tombstone stats.
            if record.attempt_outcome.is_none() {
                record.attempt_outcome = attempt_outcome;
            }
            record.requires_attempt_report = false;
            record.unavailable = Some(RetainedUnavailableReason::AttemptReportDeliveryFailed);
            Self::bump_capture_version(record);
            record.terminal_outcome.is_some()
                && record.active_leases == 0
                && record.active_activities == 0
                && !matches!(record.phase, QueryPhase::Retained | QueryPhase::Tombstone)
        };
        self.notify.notify_waiters();
        if should_finalize {
            self.finalize_ready();
        }
    }

    pub(crate) fn record_mut<'a>(
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

    pub(crate) fn cancel_ticket(&self, query_id: &str, error: CapturedError) -> bool {
        // An attempt can synchronously publish its sealed report when it is
        // cancelled. Its observer re-enters this coordinator, so finish the
        // ticket state transition before invoking either cancellation handle.
        let ticket = {
            let mut state = self.lock_state();
            let Some(key) = state.current.get(query_id).cloned() else { return false };
            let Some(record) = state.records.get_mut(&key) else { return false };
            if record.phase != QueryPhase::Ticketed {
                return false;
            }
            let ticket = record.ticket.take();
            if record.terminal_outcome.is_none() {
                let mut terminal = QueryTerminal::Cancelled { stage: "cancel_query", error: Arc::new(error) };
                Self::apply_terminal_to_capture(record, &mut terminal, Instant::now());
                record.terminal_outcome = Some(terminal.compact_outcome());
                record.terminal = Some(terminal);
            }
            record.phase = QueryPhase::Finalizing;
            state.ticketed_count = state.ticketed_count.checked_sub(1).expect("cancelled ticket count underflow");
            ticket
        };
        if let Some(ticket) = ticket {
            ticket.native_cancellation.cancel();
            if let Some(attempt) = ticket.attempt.as_ref() {
                attempt.cancel_before_stream();
            }
        }
        self.notify.notify_waiters();
        self.finalize_ready();
        true
    }

    #[cfg(all(test, feature = "iceberg"))]
    pub(crate) fn ticket_query_ids(&self) -> Vec<String> {
        let state = self.lock_state();
        state
            .records
            .values()
            .filter(|record| record.phase == QueryPhase::Ticketed)
            .map(|record| record.key.query_id().to_string())
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn ticket_correlation_id(&self, query_id: &str) -> Option<Option<String>> {
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        let record = state.records.get(key)?;
        if record.phase != QueryPhase::Ticketed {
            return None;
        }
        Some(record.ticket.as_ref()?.correlation_id.clone())
    }

    #[cfg(test)]
    pub(crate) fn ticket_cancellation(&self, query_id: &str) -> Option<ExecutionCancellation> {
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
    pub(crate) fn apply_terminal_to_capture(record: &mut QueryRecord, terminal: &mut QueryTerminal, now: Instant) {
        match terminal {
            QueryTerminal::Succeeded => {
                if let Some(captured) = &mut record.captured {
                    captured.finish_success(now);
                }
            },
            QueryTerminal::Failed { stage, error, .. } => {
                if let Some(captured) = &mut record.captured {
                    captured.bound_error(Arc::make_mut(error));
                    captured.finish_failed(stage, Arc::clone(error), now);
                }
            },
            QueryTerminal::Cancelled { stage, error } => {
                if let Some(captured) = &mut record.captured {
                    captured.bound_error(Arc::make_mut(error));
                    if record.requires_attempt_report {
                        captured.finish_failed(stage, Arc::clone(error), now);
                    } else {
                        captured.finish_cancelled(stage, Arc::clone(error), now);
                    }
                }
            },
            QueryTerminal::Dropped | QueryTerminal::Panicked => {
                if let Some(captured) = &mut record.captured {
                    captured.finish_dropped(now);
                }
            },
        }
    }

    pub(crate) fn bump_capture_version(record: &mut QueryRecord) {
        record.capture_version = record.capture_version.checked_add(1).unwrap_or_else(|| {
            if let Some(captured) = &mut record.captured {
                captured.mark_counter_overflow();
            }
            u64::MAX
        });
    }

    #[cfg(test)]
    pub(crate) fn artifact(&self, query_id: &str) -> Option<CoordinatorArtifact> {
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
            QueryPhase::Tombstone => CoordinatorArtifact::Unavailable(
                record.unavailable.expect("tombstone phase owns an unavailable reason"),
            ),
        })
    }

    #[cfg(test)]
    pub(crate) fn terminal(&self, query_id: &str) -> Option<QueryTerminalOutcome> {
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        state.records.get(key)?.terminal_outcome
    }
}
