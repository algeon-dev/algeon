//! Record-path handle methods and bounded detail retention.

use std::{
    collections::BTreeMap,
    fmt, mem,
    panic::{self, AssertUnwindSafe},
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use query_planning::correlation::{QueryId, TraceId};

use super::{
    AttemptReportSealProvider, BudgetWindow, BudgetWindowPeaks, CleanupFailureSummary, DeferredRuntimeObservationSeal,
    LiveRuntimeObservationSnapshot, MetricsCaptureMode, MetricsCompleteness, OsSourceLocationDigestKeyProvider,
    PlanExecutionId, PlanExecutionMetrics, QueryMetrics, QueryMetricsCheckpoint, RUNTIME_OBSERVATION_JSONL_SCHEMA,
    RuntimeClockDomain, RuntimeEventContext, RuntimeObservationClock, RuntimeObservationConfig,
    RuntimeObservationHandle, RuntimeObservationLimits, RuntimeObservationLossReason, RuntimeObservationManifest,
    RuntimeObservationPayload, RuntimeObservationRecordClass, RuntimeObservationRecordDisposition,
    RuntimeObservationRecorderState, RuntimeObservationSealObserver, RuntimeObservationSessionInner,
    RuntimeSourceLocationId, RuntimeTerminalOutcome, RuntimeTopologyRecord, SealedRuntimeObservationSnapshot,
    SourceLocationDigestKeyProvider, SourceLocationDigestKeyState, SourceLocationDigestKeyUnavailable, SourceReadFact,
    SourceReadRuntimeMetrics, SystemRuntimeObservationClock,
};
use crate::{
    metrics::BoundaryMetrics,
    observability::{LakehouseCacheFillEvent, LakehouseCacheFillEventSnapshot, recorder::ExecutionProgress},
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
    /// A disabled config still allocates the session. Trace capture is off,
    /// and the query's metrics reduction continues. Only [`Self::disabled`]
    /// produces a session with no recorder at all.
    pub(super) fn new(
        query_id: impl Into<QueryId>,
        trace_id: Option<TraceId>,
        config: RuntimeObservationConfig,
    ) -> Self {
        Self::with_clock(query_id, trace_id, config, Arc::new(SystemRuntimeObservationClock::new()))
    }

    pub(super) fn with_clock(
        query_id: impl Into<QueryId>,
        trace_id: Option<TraceId>,
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
        query_id: impl Into<QueryId>,
        trace_id: Option<TraceId>,
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
                seal_published: Condvar::new(),
            })),
        }
    }

    /// Returns a disabled session.
    pub const fn disabled() -> Self {
        Self { config: RuntimeObservationConfig::disabled(), inner: None }
    }

    /// Returns whether this session constructs and records trace payloads.
    pub(crate) const fn is_enabled(&self) -> bool {
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

    /// Counts one cleanup failure in the session's exact tally, which the
    /// attempt report reads at seal, before the caller attempts an optional
    /// timeline projection. Failures after the seal are ignored.
    #[doc(hidden)]
    pub fn record_attempt_cleanup_failure(&self, summary: CleanupFailureSummary) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = inner.lock_state();
        // Sealing flips this flag while holding the same lock.
        if inner.sealed.load(Ordering::Acquire) {
            return;
        }
        state.cleanup_failures.record(summary);
    }

    /// Returns the immutable query identity when recording is enabled.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.trace_inner().map(|inner| &inner.manifest.query_id)
    }

    /// Returns the immutable optional parent trace identity.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_inner().and_then(|inner| inner.manifest.trace_id.as_ref())
    }

    /// Returns a keyed, query-scoped source-location identity when recording is enabled.
    pub fn source_location_id(&self, value: &str) -> Option<RuntimeSourceLocationId> {
        self.trace_inner().and_then(|inner| inner.source_location_digest_key.source_location_id(value))
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
    #[cfg(test)]
    pub(crate) fn record_with_context_sized<F>(
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
        loop {
            if let Some(snapshot) = &state.sealed_snapshot {
                return DeferredRuntimeObservationSeal { snapshot: Some(snapshot.clone()), observers: Vec::new() };
            }
            if !inner.sealed.load(Ordering::Acquire) {
                break;
            }
            state = inner.wait_for_seal(state);
        }
        inner.sealed.store(true, Ordering::Release);
        // Every exit after `sealed` publishes a snapshot, so a seal waiting in
        // `wait_for_seal` never waits on one that unwound.
        match panic::catch_unwind(AssertUnwindSafe(|| {
            self.publish_seal(inner, state, outcome, attempt_report_provider)
        })) {
            Ok(deferred) => deferred,
            Err(panic) => {
                let mut state = inner.lock_state();
                if state.sealed_snapshot.is_none() {
                    let mut data = inner.take_sealed_snapshot_data(&mut state, Some(RuntimeTerminalOutcome::Panicked));
                    data.footer.degraded = true;
                    state.sealed_snapshot = Some(SealedRuntimeObservationSnapshot { data: Arc::new(data) });
                }
                let deferred = DeferredRuntimeObservationSeal {
                    snapshot: state.sealed_snapshot.clone(),
                    observers: mem::take(&mut state.seal_observers),
                };
                inner.seal_published.notify_all();
                // Observers may re-enter the session, which locks the recorder state.
                drop(state);
                deferred.invoke_observers();
                panic::resume_unwind(panic);
            },
        }
    }

    /// Builds and publishes the sealed snapshot of a seal that has set
    /// `sealed`.
    fn publish_seal<'a>(
        &self,
        inner: &'a RuntimeObservationSessionInner,
        mut state: MutexGuard<'a, RuntimeObservationRecorderState>,
        outcome: Option<RuntimeTerminalOutcome>,
        attempt_report_provider: Option<&AttemptReportSealProvider>,
    ) -> DeferredRuntimeObservationSeal {
        settle_cache_fill_events(&mut state, inner.manifest.limits);
        let mut data = inner.take_sealed_snapshot_data(&mut state, outcome);
        data.plans = self.sealed_plan_metrics(&state, &inner.manifest);
        let mut provider_panic = None;
        if let Some(provider) = attempt_report_provider {
            let cleanup_failures = state.cleanup_failures.clone();
            // The provider runs the attempt's report sinks, which may read or
            // record through this session; recording is already closed.
            drop(state);
            // The provider builds the exact immutable report that completion
            // receivers publish, from the complete footer.
            match panic::catch_unwind(AssertUnwindSafe(|| provider(&data.footer, &cleanup_failures))) {
                Ok(report) => {
                    debug_assert_eq!(data.footer.terminal.as_ref(), Some(&report.outcome));
                    data.footer.attempt_report = Some(report);
                },
                Err(panic) => {
                    data.footer.terminal = Some(RuntimeTerminalOutcome::Panicked);
                    data.footer.degraded = true;
                    provider_panic = Some(panic);
                },
            }
            state = inner.lock_state();
        }
        let snapshot = SealedRuntimeObservationSnapshot { data: Arc::new(data) };
        state.sealed_snapshot = Some(snapshot.clone());
        let observers = mem::take(&mut state.seal_observers);
        inner.seal_published.notify_all();
        let deferred = DeferredRuntimeObservationSeal { snapshot: Some(snapshot), observers };
        if let Some(panic) = provider_panic {
            // Observers run outside the state lock. The snapshot is already
            // published, so a recovery seal returns it without waiting.
            drop(state);
            deferred.invoke_observers();
            panic::resume_unwind(panic);
        }
        deferred
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

    /// Returns the sealed immutable snapshot when a terminal outcome was already fixed.
    pub fn sealed_snapshot(&self) -> Option<SealedRuntimeObservationSnapshot> {
        let inner = self.trace_inner()?;
        inner.lock_state().sealed_snapshot.clone()
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
        if execution.is_frozen() {
            return;
        }
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
        if inner.sealed.load(Ordering::Acquire)
            || state.metrics.get(plan_execution_id).is_some_and(PlanExecutionMetrics::is_frozen)
        {
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

    /// Offers one background lakehouse cache-fill completion for bounded
    /// retention.
    ///
    /// With trace capture disabled the event is not kept. Otherwise the next
    /// plan execution to freeze its metrics offers it under the query's record
    /// and byte limits, and a declined event is a `LakehouseCacheFillDetail`
    /// loss in that plan's completeness. The seal settles events no plan froze
    /// with: an unfrozen plan execution carries them, or each is an
    /// `Unassigned` loss.
    #[doc(hidden)]
    pub fn record_lakehouse_cache_fill(&self, event: &LakehouseCacheFillEvent) {
        let Some(inner) = self.trace_inner() else {
            return;
        };
        let mut state = inner.lock_state();
        if inner.sealed.load(Ordering::Acquire) {
            return;
        }
        let waiting = u64::try_from(state.cache_fill_events.len()).unwrap_or(u64::MAX);
        if waiting < inner.manifest.limits.max_retained_records {
            state.cache_fill_events.push(LakehouseCacheFillEventSnapshot::from(event));
        } else {
            state.cache_fill_events_overflow = state.cache_fill_events_overflow.saturating_add(1);
        }
    }

    /// Returns whether this handle reaches a query-bound owner, which is
    /// where metrics are reduced even when trace capture is off.
    pub(crate) const fn has_owner(&self) -> bool {
        self.inner.is_some()
    }

    /// Returns the data-boundary counters of a plan execution.
    pub(crate) fn boundary_metrics(&self, plan_execution_id: &PlanExecutionId) -> BoundaryMetrics {
        let Some(inner) = &self.inner else {
            return BoundaryMetrics::default();
        };
        let state = inner.lock_state();
        state.metrics.get(plan_execution_id).map(|execution| execution.reduction.boundary.clone()).unwrap_or_default()
    }

    /// Returns the progress counters of a plan execution.
    pub(crate) fn execution_progress(&self, plan_execution_id: &PlanExecutionId) -> super::ExecutionProgress {
        let Some(inner) = &self.inner else {
            return ExecutionProgress::default();
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
        Some(self.plan_checkpoint(&state, plan_execution_id, inner.sealed.load(Ordering::Acquire)))
    }

    /// Freezes a plan execution's metrics and returns them. Facts recorded
    /// for that plan afterwards are ignored; freezing again returns the same
    /// checkpoint. Returns `None` only for [`Self::disabled`].
    pub(crate) fn freeze_plan_metrics(&self, plan_execution_id: &PlanExecutionId) -> Option<QueryMetricsCheckpoint> {
        let inner = self.inner.as_ref()?;
        let mut state = inner.lock_state();
        let mut execution = state.metrics.remove(plan_execution_id).unwrap_or_default();
        // After the seal a plan's metrics are the sealed snapshot's value; the
        // seal already settled every waiting cache-fill event.
        if !execution.is_frozen() && !inner.sealed.load(Ordering::Acquire) {
            assign_cache_fill_events(&mut state, &mut execution, inner.manifest.limits);
        }
        state.metrics.insert(plan_execution_id.clone(), execution);
        let checkpoint = self.plan_checkpoint(&state, plan_execution_id, true);
        state.metrics.entry(plan_execution_id.clone()).or_default().frozen = Some(checkpoint.clone());
        Some(checkpoint)
    }

    /// Every plan execution's metrics as a sealed snapshot carries them: the
    /// frozen value, or the reduction at seal.
    fn sealed_plan_metrics(
        &self,
        state: &RuntimeObservationRecorderState,
        manifest: &RuntimeObservationManifest,
    ) -> BTreeMap<PlanExecutionId, QueryMetrics> {
        state
            .metrics
            .keys()
            .map(|plan_execution_id| {
                let mut metrics = QueryMetrics {
                    query_id: QueryId::new(manifest.query_id.as_str()),
                    trace_id: manifest.trace_id.as_ref().map(|trace_id| TraceId::new(trace_id.as_str())),
                    ..QueryMetrics::default()
                };
                metrics.apply_owner_checkpoint(&self.plan_checkpoint(state, plan_execution_id, true));
                (plan_execution_id.clone(), metrics)
            })
            .collect()
    }

    fn plan_checkpoint(
        &self,
        state: &RuntimeObservationRecorderState,
        plan_execution_id: &PlanExecutionId,
        sealed: bool,
    ) -> QueryMetricsCheckpoint {
        let empty = PlanExecutionMetrics::default();
        let execution = state.metrics.get(plan_execution_id).unwrap_or(&empty);
        if let Some(frozen) = &execution.frozen {
            return frozen.clone();
        }
        let detail_rows_dropped = execution.source_read_rows_dropped.saturating_add(execution.detail_rows_dropped);
        QueryMetricsCheckpoint {
            revision: state.metrics_revision,
            sealed,
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
        }
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
        if inner.sealed.load(Ordering::Acquire)
            || state.metrics.get(plan_execution_id).is_some_and(PlanExecutionMetrics::is_frozen)
        {
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
        if sealed || execution.is_frozen() || (peaks.observed_peak_bytes.is_none() && peaks.retained_bytes.is_none()) {
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

/// Offers the waiting cache-fill events for retention as `execution`'s rows. A
/// declined or overflowed event counts against `execution`'s completeness.
fn assign_cache_fill_events(
    state: &mut RuntimeObservationRecorderState,
    execution: &mut PlanExecutionMetrics,
    limits: RuntimeObservationLimits,
) {
    for event in mem::take(&mut state.cache_fill_events) {
        let logical_bytes = debug_logical_bytes(&event);
        let sequence = state.take_sequence();
        if state
            .admit_detail(limits, RuntimeObservationRecordClass::LakehouseCacheFillDetail, sequence, logical_bytes)
            .is_ok()
        {
            execution.reduction.lakehouse_cache_fill_events.push(event);
        } else {
            execution.detail_rows_dropped = execution.detail_rows_dropped.saturating_add(1);
        }
    }
    let overflow = mem::take(&mut state.cache_fill_events_overflow);
    drop_cache_fill_events(state, overflow, RuntimeObservationLossReason::RecordLimit);
    execution.detail_rows_dropped = execution.detail_rows_dropped.saturating_add(overflow);
}

/// Settles the cache-fill events no plan froze with when the session seals:
/// the first plan execution still unfrozen carries them, and with none left
/// each is an `Unassigned` loss.
fn settle_cache_fill_events(state: &mut RuntimeObservationRecorderState, limits: RuntimeObservationLimits) {
    if state.cache_fill_events.is_empty() && state.cache_fill_events_overflow == 0 {
        return;
    }
    let carrier = state.metrics.iter().find(|(_, execution)| !execution.is_frozen()).map(|(id, _)| id.clone());
    match carrier {
        Some(carrier) => {
            let mut execution = state.metrics.remove(&carrier).unwrap_or_default();
            assign_cache_fill_events(state, &mut execution, limits);
            state.metrics.insert(carrier, execution);
        },
        None => {
            let unassigned = u64::try_from(mem::take(&mut state.cache_fill_events).len()).unwrap_or(u64::MAX);
            drop_cache_fill_events(state, unassigned, RuntimeObservationLossReason::Unassigned);
            let overflow = mem::take(&mut state.cache_fill_events_overflow);
            drop_cache_fill_events(state, overflow, RuntimeObservationLossReason::RecordLimit);
        },
    }
}

fn drop_cache_fill_events(
    state: &mut RuntimeObservationRecorderState,
    count: u64,
    reason: RuntimeObservationLossReason,
) {
    for _ in 0..count {
        let sequence = state.take_sequence();
        state.record_drop(RuntimeObservationRecordClass::LakehouseCacheFillDetail, sequence, reason);
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
