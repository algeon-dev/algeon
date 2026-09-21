//! Record-path handle methods and bounded detail retention.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use super::{
    AttemptReportSealProvider, BudgetWindow, BudgetWindowPeaks, CleanupFailureSummary, DeferredRuntimeObservationSeal,
    LiveRuntimeObservationSnapshot, MetricsCaptureMode, MetricsCompleteness, OsSourceLocationDigestKeyProvider,
    PlanExecutionId, PlanExecutionMetrics, QueryMetrics, QueryMetricsCheckpoint, RUNTIME_OBSERVATION_JSONL_SCHEMA,
    RuntimeClockDomain, RuntimeEventContext, RuntimeObservationClock, RuntimeObservationConfig,
    RuntimeObservationHandle, RuntimeObservationLimits, RuntimeObservationLossReason, RuntimeObservationManifest,
    RuntimeObservationPayload, RuntimeObservationRecordClass, RuntimeObservationRecordDisposition,
    RuntimeObservationRecorderState, RuntimeObservationSealObserver, RuntimeObservationSessionInner, RuntimeQueryId,
    RuntimeSourceLocationId, RuntimeTerminalOutcome, RuntimeTopologyRecord, RuntimeTraceId,
    SealedRuntimeObservationSnapshot, SourceLocationDigestKeyProvider, SourceLocationDigestKeyState,
    SourceLocationDigestKeyUnavailable, SourceReadFact, SourceReadRuntimeMetrics, SystemRuntimeObservationClock,
};

/// Growing detail families and the loss class each is retained under.
///
/// Families bounded by plan topology; graph calibration rows, which a later
/// measurement updates in place, task mappings, and the keyed task summaries;
/// are not listed: they cannot grow with data, and retaining them in full keeps
/// in-place updates correct.
fn detail_lengths(metrics: &QueryMetrics) -> [usize; 5] {
    [
        metrics.operators.len(),
        metrics.join_strategy_decisions.len(),
        metrics.aggregate_strategy_decisions.len(),
        metrics.dynamic_filter_runtime_rows.len(),
        metrics.budgeted_execution_events.len(),
    ]
}

const DETAIL_CLASSES: [RuntimeObservationRecordClass; 5] = [
    RuntimeObservationRecordClass::OperatorDetail,
    RuntimeObservationRecordClass::JoinStrategyDetail,
    RuntimeObservationRecordClass::AggregateStrategyDetail,
    RuntimeObservationRecordClass::DynamicFilterDetail,
    RuntimeObservationRecordClass::BudgetDetail,
];

impl RuntimeObservationHandle {
    /// Creates a query-bound session using a system monotonic clock.
    ///
    /// A disabled config still allocates the session: it turns off trace
    /// capture, not the query's metrics reduction. Only [`Self::disabled`]
    /// produces a session with no recorder at all.
    pub(super) fn new(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
    ) -> Self {
        Self::with_clock(query_id, trace_id, config, Arc::new(SystemRuntimeObservationClock::new()))
    }

    /// Creates a session using an injected clock.
    pub(super) fn with_clock(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
        clock: Arc<dyn RuntimeObservationClock>,
    ) -> Self {
        Self::with_clock_and_source_location_digest_key(
            query_id,
            trace_id,
            config,
            clock,
            &OsSourceLocationDigestKeyProvider,
        )
    }

    pub(super) fn with_clock_and_source_location_digest_key(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
        clock: Arc<dyn RuntimeObservationClock>,
        source_location_digest_key_provider: &dyn SourceLocationDigestKeyProvider,
    ) -> Self {
        let manifest = RuntimeObservationManifest {
            schema: RUNTIME_OBSERVATION_JSONL_SCHEMA.to_string(),
            query_id: query_id.into(),
            trace_id,
            unix_epoch_ns: clock.unix_epoch_ns(),
            clock_domain: RuntimeClockDomain::HostMonotonic,
            limits: config.limits(),
        };
        let (source_location_digest_key, state) = match source_location_digest_key_provider.source_location_digest_key()
        {
            Ok(key) => (
                SourceLocationDigestKeyState::Available(key),
                RuntimeObservationRecorderState { next_sequence: 1, ..Default::default() },
            ),
            Err(SourceLocationDigestKeyUnavailable) => {
                let mut dropped_by_reason = BTreeMap::new();
                dropped_by_reason.insert(RuntimeObservationLossReason::SourceLocationKeyUnavailable, 1);
                (
                    SourceLocationDigestKeyState::Unavailable,
                    RuntimeObservationRecorderState {
                        next_sequence: 1,
                        dropped_by_reason,
                        degraded: true,
                        ..Default::default()
                    },
                )
            },
        };
        Self {
            config,
            inner: Some(Arc::new(RuntimeObservationSessionInner {
                manifest,
                source_location_digest_key,
                clock,
                state: Mutex::new(state),
                sealed: AtomicBool::new(false),
            })),
            attempt_cleanup_failure_recorder: None,
        }
    }

    /// Returns a disabled session.
    pub const fn disabled() -> Self {
        Self { config: RuntimeObservationConfig::disabled(), inner: None, attempt_cleanup_failure_recorder: None }
    }

    /// Returns whether this session constructs and records trace payloads.
    pub const fn is_enabled(&self) -> bool {
        self.config.is_enabled() && self.inner.is_some()
    }

    // Every pre-existing handle method is a trace operation and keeps its
    // exact disabled behaviour by going through this accessor: no lazy payload
    // runs, and no trace snapshot is produced. Only the metrics methods below
    // read `inner` directly, because metrics reduction outlives trace capture.
    fn trace_inner(&self) -> Option<&Arc<RuntimeObservationSessionInner>> {
        self.inner.as_ref().filter(|_| self.config.is_enabled())
    }

    /// Returns the immutable recorder configuration.
    pub const fn config(&self) -> RuntimeObservationConfig {
        self.config
    }

    /// Routes an attempt-owned cleanup failure into the lifecycle before the
    /// caller attempts an optional timeline projection.
    #[doc(hidden)]
    pub fn record_attempt_cleanup_failure(&self, summary: CleanupFailureSummary) {
        if let Some(recorder) = &self.attempt_cleanup_failure_recorder {
            recorder(summary);
        }
    }

    /// Returns the immutable query identity when recording is enabled.
    pub fn query_id(&self) -> Option<&RuntimeQueryId> {
        self.trace_inner().map(|inner| &inner.manifest.query_id)
    }

    /// Returns the immutable optional parent trace identity.
    pub fn trace_id(&self) -> Option<&RuntimeTraceId> {
        self.trace_inner().and_then(|inner| inner.manifest.trace_id.as_ref())
    }

    /// Returns a keyed, query-scoped source-location identity when recording is enabled.
    pub fn source_location_id(&self, value: &str) -> Option<RuntimeSourceLocationId> {
        self.trace_inner().and_then(|inner| inner.source_location_digest_key.source_location_id(value))
    }

    /// Exact retained byte size of a keyed source-location id without hashing
    /// or allocating the id itself.
    pub fn source_location_id_retained_bytes(&self) -> Option<u64> {
        self.trace_inner().and_then(|inner| inner.source_location_digest_key.retained_id_bytes())
    }

    /// Lazily constructs and records a query-level event.
    pub fn record<F>(&self, build_payload: F) -> RuntimeObservationRecordDisposition
    where F: FnOnce() -> RuntimeObservationPayload {
        self.record_with_context(RuntimeEventContext::default(), build_payload)
    }

    /// Lazily constructs and records an event for one plan execution.
    pub fn record_for_plan<F>(
        &self,
        plan_execution_id: PlanExecutionId,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        self.record_with_context(RuntimeEventContext::for_plan(plan_execution_id), build_payload)
    }

    /// Lazily constructs an event before acquiring recorder order and records it infallibly.
    pub fn record_with_context<F>(
        &self,
        context: RuntimeEventContext,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        let Some(inner) = self.trace_inner() else {
            return RuntimeObservationRecordDisposition::Disabled;
        };
        if inner.sealed.load(Ordering::Acquire) {
            return RuntimeObservationRecordDisposition::Sealed;
        }
        inner.record_event_lazy(context, build_payload)
    }

    /// Records an event after an exact borrowed-data size preflight.
    ///
    /// The payload closure is not invoked when the record or byte cap already
    /// rejects `logical_bytes`, so variable-size strings and row-group vectors
    /// are never cloned merely to discover that capture is exhausted.
    pub fn record_with_context_sized<F>(
        &self,
        context: RuntimeEventContext,
        logical_bytes: u64,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        let Some(inner) = self.trace_inner() else {
            return RuntimeObservationRecordDisposition::Disabled;
        };
        if inner.sealed.load(Ordering::Acquire) {
            return RuntimeObservationRecordDisposition::Sealed;
        }
        inner.record_event_sized(context, logical_bytes, build_payload)
    }

    /// Records an event whose payload depends on its exact recorder timestamp.
    ///
    /// The payload builder runs while recorder order is held and receives the
    /// same query-relative monotonic timestamp retained on the event. The
    /// timestamp is returned even when bounded capture drops the event, and
    /// `None` is returned only when recording is disabled or already sealed.
    #[doc(hidden)]
    pub fn record_with_context_timestamped<F>(&self, context: RuntimeEventContext, build_payload: F) -> Option<u64>
    where F: FnOnce(u64) -> RuntimeObservationPayload {
        let inner = self.trace_inner()?;
        if inner.sealed.load(Ordering::Acquire) {
            return None;
        }
        inner.record_event_timestamped(context, build_payload).1
    }

    /// Lazily constructs and records one static topology row.
    pub fn record_topology<F>(
        &self,
        class: RuntimeObservationRecordClass,
        build_record: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeTopologyRecord,
    {
        let Some(inner) = self.trace_inner() else {
            return RuntimeObservationRecordDisposition::Disabled;
        };
        if inner.sealed.load(Ordering::Acquire) {
            return RuntimeObservationRecordDisposition::Sealed;
        }
        inner.record_topology_lazy(class, build_record)
    }

    /// Captures an immutable incomplete view. Returns none after sealing or when disabled.
    pub fn live_snapshot(&self) -> Option<LiveRuntimeObservationSnapshot> {
        let inner = self.trace_inner()?;
        let state = inner.lock_state();
        if state.sealed_snapshot.is_some() {
            return None;
        }
        Some(LiveRuntimeObservationSnapshot { data: Arc::new(inner.snapshot_data(&state, false, None)) })
    }

    pub(super) fn seal_owned(&self, outcome: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        self.seal_owned_deferred(Some(outcome), None).invoke_observers()
    }

    pub(super) fn seal_owned_deferred(
        &self,
        outcome: Option<RuntimeTerminalOutcome>,
        attempt_report_provider: Option<&AttemptReportSealProvider>,
    ) -> DeferredRuntimeObservationSeal {
        let Some(inner) = self.inner.as_ref() else {
            return DeferredRuntimeObservationSeal { snapshot: None, observers: Vec::new() };
        };
        let mut state = inner.lock_state();
        if !self.config.is_enabled() {
            // No trace snapshot exists to seal, but the metrics reduction must
            // still stop changing once the query's terminal outcome is fixed.
            inner.sealed.store(true, Ordering::Release);
            return DeferredRuntimeObservationSeal { snapshot: None, observers: Vec::new() };
        }
        if let Some(snapshot) = &state.sealed_snapshot {
            return DeferredRuntimeObservationSeal { snapshot: Some(snapshot.clone()), observers: Vec::new() };
        }
        inner.sealed.store(true, Ordering::Release);
        let snapshot = SealedRuntimeObservationSnapshot {
            data: Arc::new(inner.take_sealed_snapshot_data(&mut state, outcome, attempt_report_provider)),
        };
        state.sealed_snapshot = Some(snapshot.clone());
        let observers = std::mem::take(&mut state.seal_observers);
        DeferredRuntimeObservationSeal { snapshot: Some(snapshot), observers }
    }

    pub(super) fn mark_degraded(&self) {
        if let Some(inner) = self.trace_inner() {
            inner.lock_state().degraded = true;
        }
    }

    pub(super) fn register_seal_observer(
        &self,
        observer: RuntimeObservationSealObserver,
    ) -> Option<(RuntimeObservationSealObserver, SealedRuntimeObservationSnapshot)> {
        let inner = self.trace_inner()?;
        let mut state = inner.lock_state();
        if let Some(snapshot) = &state.sealed_snapshot {
            Some((observer, snapshot.clone()))
        } else {
            state.seal_observers.push(observer);
            None
        }
    }

    /// Returns the sealed immutable snapshot when a terminal outcome already won.
    pub fn sealed_snapshot(&self) -> Option<SealedRuntimeObservationSnapshot> {
        let inner = self.trace_inner()?;
        inner.lock_state().sealed_snapshot.clone()
    }

    /// Returns whether both handles refer to the same recorder session.
    ///
    /// Disabled handles have no recorder identity and therefore never compare
    /// as the same session. This is intended for ownership handoff boundaries
    /// that must reject two independently-created sessions for one external
    /// query instead of silently replacing the first.
    pub fn is_same_session(&self, other: &Self) -> bool {
        match (self.trace_inner(), other.trace_inner()) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            (None, _) | (_, None) => false,
        }
    }

    /// Returns whether a sealed snapshot was produced by this recorder session.
    pub fn owns_snapshot(&self, snapshot: &SealedRuntimeObservationSnapshot) -> bool {
        self.sealed_snapshot().is_some_and(|sealed| sealed.is_same_snapshot(snapshot))
    }

    /// Folds one source-read fact into a plan execution's exact totals.
    ///
    /// This runs whether or not trace capture is enabled, and before any
    /// retention decision, so the totals never depend on what was retained.
    /// Facts arriving after the owner sealed are ignored.
    pub fn record_source_read(&self, plan_execution_id: &PlanExecutionId, fact: SourceReadFact) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = inner.lock_state();
        // Sealing flips this flag while holding the same lock, so reading it
        // here cannot race a concurrent seal.
        if inner.sealed.load(Ordering::Acquire) {
            return;
        }
        let execution = state.metrics.entry(plan_execution_id.clone()).or_default();
        if !execution.summary.source_read.apply(&fact) {
            execution.totals_saturated = true;
        }
        state.metrics_revision = state.metrics_revision.saturating_add(1);
    }

    /// Offers one source-read detail row for bounded retention.
    ///
    /// Rows never feed totals; record the matching [`SourceReadFact`]s for
    /// those. With trace capture disabled the row is intentionally not kept;
    /// otherwise it is retained under the recorder's record and byte limits,
    /// which are shared by the whole query, and a declined row is counted as a
    /// `SourceReadDetail` loss.
    pub fn record_source_read_row(&self, plan_execution_id: &PlanExecutionId, row: SourceReadRuntimeMetrics) {
        let Some(inner) = self.trace_inner() else {
            return;
        };
        let mut state = inner.lock_state();
        if inner.sealed.load(Ordering::Acquire) {
            return;
        }
        let logical_bytes = debug_logical_bytes(&row);
        let sequence = state.take_sequence();
        let admitted = state
            .admit(inner.manifest.limits, RuntimeObservationRecordClass::SourceReadDetail, sequence, logical_bytes)
            .is_ok();
        let execution = state.metrics.entry(plan_execution_id.clone()).or_default();
        if admitted {
            execution.source_read_rows.push(row);
        } else {
            execution.source_read_rows_dropped = execution.source_read_rows_dropped.saturating_add(1);
        }
        state.metrics_revision = state.metrics_revision.saturating_add(1);
    }

    /// Returns whether this handle reaches a query-bound owner, which is
    /// where metrics are reduced even when trace capture is off.
    pub const fn has_owner(&self) -> bool {
        self.inner.is_some()
    }

    /// Returns the data-boundary counters of a plan execution.
    pub(crate) fn boundary_metrics(&self, plan_execution_id: &PlanExecutionId) -> crate::metrics::BoundaryMetrics {
        let Some(inner) = &self.inner else {
            return crate::metrics::BoundaryMetrics::default();
        };
        let state = inner.lock_state();
        state.metrics.get(plan_execution_id).map(|execution| execution.reduction.boundary.clone()).unwrap_or_default()
    }

    /// Returns the progress counters of a plan execution.
    pub(crate) fn execution_progress(&self, plan_execution_id: &PlanExecutionId) -> super::ExecutionProgress {
        let Some(inner) = &self.inner else {
            return super::ExecutionProgress::default();
        };
        let state = inner.lock_state();
        state.metrics.get(plan_execution_id).map_or_else(Default::default, |execution| super::ExecutionProgress {
            source_chunks_planned: execution.summary.source_read.source_chunks_planned,
            source_chunks_committed: execution.summary.source_read.source_chunks_committed,
            partial_aggregate_count: execution.reduction.partial_aggregate_count,
        })
    }

    /// Returns one internally consistent view of a plan execution's metrics.
    ///
    /// Available for any query-bound session, including one whose trace
    /// capture is disabled. A plan execution that recorded nothing yet reports
    /// empty totals. Returns `None` only for [`Self::disabled`].
    pub fn metrics_checkpoint(&self, plan_execution_id: &PlanExecutionId) -> Option<QueryMetricsCheckpoint> {
        let inner = self.inner.as_ref()?;
        let state = inner.lock_state();
        let empty = PlanExecutionMetrics::default();
        let execution = state.metrics.get(plan_execution_id).unwrap_or(&empty);
        let detail_rows_dropped = execution.source_read_rows_dropped.saturating_add(execution.detail_rows_dropped);
        Some(QueryMetricsCheckpoint {
            revision: state.metrics_revision,
            sealed: inner.sealed.load(Ordering::Acquire),
            summary: execution.summary.clone(),
            source_read_rows: execution.source_read_rows.clone(),
            metrics: execution.reduction.clone(),
            completeness: MetricsCompleteness {
                capture_mode: if self.config.is_enabled() {
                    MetricsCaptureMode::Bounded
                } else {
                    MetricsCaptureMode::Disabled
                },
                detail_rows_dropped,
                truncated: detail_rows_dropped > 0,
                totals_saturated: execution.totals_saturated || state.counter_overflow,
            },
        })
    }

    /// Applies one typed metrics fact to a plan execution's reduction.
    ///
    /// Only `MetricsRecorder` calls this, with a closure that invokes one
    /// existing `QueryMetrics` reduction; execution code never receives the
    /// reduction itself. Rows the fact appended to a growing detail family are
    /// then offered for bounded retention, after open budget windows have seen
    /// them, so no total depends on what was kept.
    pub(crate) fn apply_metrics(&self, plan_execution_id: &PlanExecutionId, reduce: impl FnOnce(&mut QueryMetrics)) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = inner.lock_state();
        if inner.sealed.load(Ordering::Acquire) {
            return;
        }
        let mut execution = state.metrics.remove(plan_execution_id).unwrap_or_default();
        let before = detail_lengths(&execution.reduction);
        reduce(&mut execution.reduction);
        let new_budget_events = execution.reduction.budgeted_execution_events.get(before[4]..).unwrap_or_default();
        for event in new_budget_events {
            for peaks in execution.budget_windows.values_mut() {
                peaks.observed_peak_bytes = peaks.observed_peak_bytes.max(event.allocator_high_water_bytes);
                peaks.retained_bytes = peaks.retained_bytes.max(event.retained_bytes);
            }
        }
        retain_new_detail_rows(&mut state, &mut execution, before, inner.manifest.limits, self.config.is_enabled());
        state.metrics.insert(plan_execution_id.clone(), execution);
        state.metrics_revision = state.metrics_revision.saturating_add(1);
    }

    /// Opens a budget window on a plan execution.
    pub(crate) fn open_budget_window(&self, plan_execution_id: &PlanExecutionId) -> BudgetWindow {
        let Some(inner) = &self.inner else {
            return BudgetWindow(0);
        };
        let mut state = inner.lock_state();
        let execution = state.metrics.entry(plan_execution_id.clone()).or_default();
        execution.next_budget_window = execution.next_budget_window.saturating_add(1);
        let id = execution.next_budget_window;
        execution.budget_windows.insert(id, BudgetWindowPeaks::default());
        BudgetWindow(id)
    }

    /// Closes a budget window and returns the peaks it saw.
    pub(crate) fn take_budget_window(
        &self,
        plan_execution_id: &PlanExecutionId,
        window: BudgetWindow,
    ) -> BudgetWindowPeaks {
        let Some(inner) = &self.inner else {
            return BudgetWindowPeaks::default();
        };
        let mut state = inner.lock_state();
        state
            .metrics
            .get_mut(plan_execution_id)
            .and_then(|execution| execution.budget_windows.remove(&window.0))
            .unwrap_or_default()
    }

    /// Closes a budget window, folding the peaks it saw into one task.
    pub(crate) fn close_budget_window(
        &self,
        plan_execution_id: &PlanExecutionId,
        window: BudgetWindow,
        compiled_task_id: &str,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = inner.lock_state();
        let sealed = inner.sealed.load(Ordering::Acquire);
        let Some(execution) = state.metrics.get_mut(plan_execution_id) else {
            return;
        };
        let Some(peaks) = execution.budget_windows.remove(&window.0) else {
            return;
        };
        if sealed || (peaks.observed_peak_bytes.is_none() && peaks.retained_bytes.is_none()) {
            return;
        }
        execution.reduction.record_task_budget_observation(
            compiled_task_id,
            peaks.observed_peak_bytes,
            peaks.retained_bytes,
        );
        state.metrics_revision = state.metrics_revision.saturating_add(1);
    }
}

fn retain_new_detail_rows(
    state: &mut RuntimeObservationRecorderState,
    execution: &mut PlanExecutionMetrics,
    before: [usize; 5],
    limits: RuntimeObservationLimits,
    trace_enabled: bool,
) {
    let after = detail_lengths(&execution.reduction);
    for family in 0..before.len() {
        let (start, end) = (before[family], after[family]);
        if end <= start {
            continue;
        }
        let mut keep = Vec::with_capacity(end - start);
        for row in start..end {
            // With trace capture off, detail rows are intentionally not kept
            // and are not losses: nothing was attempted.
            let admitted = trace_enabled && {
                let logical_bytes = detail_row_logical_bytes(&execution.reduction, family, row);
                let sequence = state.take_sequence();
                state.admit_detail(limits, DETAIL_CLASSES[family], sequence, logical_bytes).is_ok()
            };
            if trace_enabled && !admitted {
                execution.detail_rows_dropped = execution.detail_rows_dropped.saturating_add(1);
            }
            keep.push(admitted);
        }
        retain_detail_family(&mut execution.reduction, family, start, &keep);
    }
}

fn detail_row_logical_bytes(metrics: &QueryMetrics, family: usize, row: usize) -> u64 {
    match family {
        0 => debug_logical_bytes(&metrics.operators[row]),
        1 => debug_logical_bytes(&metrics.join_strategy_decisions[row]),
        2 => debug_logical_bytes(&metrics.aggregate_strategy_decisions[row]),
        3 => debug_logical_bytes(&metrics.dynamic_filter_runtime_rows[row]),
        _ => debug_logical_bytes(&metrics.budgeted_execution_events[row]),
    }
}

fn retain_detail_family(metrics: &mut QueryMetrics, family: usize, start: usize, keep: &[bool]) {
    fn retain<T>(rows: &mut Vec<T>, start: usize, keep: &[bool]) {
        let appended = rows.split_off(start);
        rows.extend(appended.into_iter().zip(keep).filter_map(|(row, kept)| kept.then_some(row)));
    }
    match family {
        0 => retain(&mut metrics.operators, start, keep),
        1 => retain(&mut metrics.join_strategy_decisions, start, keep),
        2 => retain(&mut metrics.aggregate_strategy_decisions, start, keep),
        3 => retain(&mut metrics.dynamic_filter_runtime_rows, start, keep),
        _ => retain(&mut metrics.budgeted_execution_events, start, keep),
    }
}

/// Deterministic logical size of a detail row: the length of its `Debug`
/// rendering, measured without allocating it. This is the same measure the
/// DataFusion metrics projection used when it selected rows on its own.
fn debug_logical_bytes(row: &impl fmt::Debug) -> u64 {
    struct Counter(u64);
    impl fmt::Write for Counter {
        fn write_str(&mut self, value: &str) -> fmt::Result {
            self.0 = self.0.saturating_add(value.len() as u64);
            Ok(())
        }
    }
    let mut counter = Counter(0);
    let _ = fmt::write(&mut counter, format_args!("{row:?}"));
    counter.0
}
