//! Explicit execution-attempt adapter boundary.
//!
//! One [`AlgeonAttempt`] crosses exactly these terminal states: it begins
//! admitted or rejected, plans at most once into one result stream, then
//! seals with one immutable [`QueryAttemptReport`]. Disposition is
//! runtime-owned: the report records whether the attempt obtained a plan and
//! how it terminated, and the engine owns every attempt number behind it.
//! [`AttemptDisposition`] never shares an enum with plan-time coverage because
//! the runtime must not depend on DataFusion adapter types, and `Unplanned`
//! describes attempt lifecycle, not physical-plan shape. The adapter maps a
//! completed [`crate::planner::FinalPlanDisposition`] to
//! [`AttemptDisposition`] exactly once (`DataFusion` to `CpuFallback`,
//! `Mixed` to `PartialNative`, `Native` to `Native`); a planning failure
//! seals as `Unplanned` by leaving the disposition unset, never by mapping
//! from a plan classification.

use std::{
    any::Any,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    pin::Pin,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
    time::Instant,
};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::{SessionState, SessionStateBuilder, context::TaskContext},
    physical_plan::{
        ExecutionPlan, RecordBatchStream, SendableRecordBatchStream, execute_stream as execute_physical_stream,
    },
    prelude::{DataFrame, SessionContext},
};
use futures::{FutureExt, Stream, future::BoxFuture};
#[cfg(test)]
use query_engine::admission::AttemptCapabilityEvidence;
use query_engine::{
    attempt::{
        AttemptAdmissionBinding, AttemptAdmissionRejection, AttemptCompletionFactory, AttemptHandle, AttemptOwner,
        BeginAttemptRequest,
    },
    error::QueryError,
    observability::AttemptFailureSummary,
    runtime::GpuBackendSession,
};
pub use query_engine::{
    attempt::{AttemptBackendCountersSnapshot, AttemptCancellation, AttemptCompletion},
    graph_observation::GraphMeasuredPeakRow,
    observability::{
        AttemptCleanupDisposition, AttemptDisposition, AttemptId, AttemptQueryId, AttemptTraceContext,
        QueryAttemptReport, RuntimeObservationConfig, RuntimeTerminalOutcome, SealedRuntimeObservationSnapshot,
    },
};

#[cfg(test)]
use crate::native_exec::admission::{QueryAdmissionSource, QueryAdmissionTicket};
pub use crate::native_exec::{
    GraphClaimCalibrationExecution, GraphClaimCalibrationNonPhysicalReason, SealedAttemptGraphClaimCalibrationSnapshot,
};
#[cfg(feature = "server")]
use crate::planner::native::InvocationExecution;
use crate::{
    backend::AlgeonGpuBackend,
    error::{DataFusionErrorExt, Error, ErrorCode, Result},
    final_plan_observation::observe_final_plan,
    native_exec::{
        SealedAttemptMetricsTarget,
        admission::{AttemptTicketBinding, NativeAdmissionExec, task_context_with_query_admission_ticket},
        native_execution_metrics, sealed_attempt_metrics_target, transparent_root_wrapper_input,
    },
    native_report::NativeDataFusionExecutionMetrics,
    planner::{
        FinalPlanCoverage, FinalPlanDisposition, NativePlanningDiagnostics, classify_final_plan, native::Invocation,
    },
};

/// Inputs used to create an execution attempt before DataFusion planning.
///
/// [`Self::default`] enables bounded native observation using
/// [`RuntimeObservationConfig::NATIVE_EXECUTION_MAX_RECORDS`] and
/// [`RuntimeObservationConfig::NATIVE_EXECUTION_MAX_BYTES`].
#[derive(Debug, Clone)]
pub struct AttemptOptions {
    /// Optional caller correlation copied into the terminal attempt report.
    pub query_id: Option<String>,
    /// Optional opaque distributed-trace correlation copied into attempt
    /// observation and the terminal report.
    pub trace_context: Option<String>,
    /// Invocation-owner deadline in the current process's monotonic
    /// [`Instant`] clock domain.
    ///
    /// This is not a wall-clock or wire timestamp. A remote deadline must be
    /// converted by the invocation owner before beginning the attempt.
    pub deadline: Option<Instant>,
    /// Query-local runtime observation retention policy.
    pub observation: RuntimeObservationConfig,
}

impl Default for AttemptOptions {
    fn default() -> Self {
        Self {
            query_id: None,
            trace_context: None,
            deadline: None,
            observation: RuntimeObservationConfig::native_execution(),
        }
    }
}

/// Immutable native-plan and boundary evidence captured by the elected root
/// at the same seal point as its attempt report.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SealedAttemptNativeExecutionSnapshot {
    pub attempt_id: AttemptId,
    pub coverage: FinalPlanCoverage,
    pub terminal_host_materialization_count: usize,
    pub intermediate_device_to_host_materialization_count: usize,
    pub metrics: NativeDataFusionExecutionMetrics,
}

/// Per-invocation state propagated through DataFusion configuration.
///
/// This is deliberately private: the extension carries an invocation binding
/// through DataFusion planning, but it must not hand terminalization or
/// observation-sealing authority to custom optimizer rules.
#[derive(Debug)]
struct AttemptExecutionState {
    handle: AttemptHandle,
    admission_rejection: AttemptAdmissionRejection,
    admission_binding: AttemptAdmissionBinding,
    query_id: Option<AttemptQueryId>,
    query_cap_bytes: Option<u64>,
    cancellation: AttemptCancellation,
    /// The one elected native root that may receive the fixed-cardinality
    /// attempt metric after this attempt's report has sealed. The target is
    /// consumed at seal, so no wrapper can publish a second row.
    sealed_metrics_target: Mutex<Option<(SealedAttemptMetricsTarget, FinalPlanCoverage)>>,
    /// Set once the terminal authority moved into the attempt's one result
    /// stream; planning after that cannot replace the executing target.
    result_stream_started: AtomicBool,
    /// Typed graph claim/peak rows frozen by the elected target at the same
    /// seal point as the immutable attempt report.
    sealed_graph_claim_calibration: OnceLock<SealedAttemptGraphClaimCalibrationSnapshot>,
    sealed_native_execution: OnceLock<SealedAttemptNativeExecutionSnapshot>,
}

impl AttemptExecutionState {
    fn set_sealed_metrics_target(&self, plan: &dyn ExecutionPlan) {
        let Some(target) = sealed_attempt_metrics_target(plan) else { return };
        let target = target.with_execution_metrics(native_execution_metrics(plan));
        let mut slot = self.sealed_metrics_target.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Planning supplies a fallback for paths such as CTAS; the actual
        // execution root replaces it before terminal publication so a replan
        // never projects into an unexecuted wrapper.
        *slot = Some((target, observe_final_plan(plan)));
    }

    fn publish_sealed_metrics(&self, report: &QueryAttemptReport) {
        let target = self.sealed_metrics_target.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some((target, observation)) = target {
            let (snapshot, metrics) = target.record(report);
            let intermediate_device_to_host_materialization_count =
                metrics.device_to_host_materializations.saturating_sub(observation.host_materialization_count());
            let _ = self.sealed_graph_claim_calibration.set(snapshot);
            let _ = self.sealed_native_execution.set(SealedAttemptNativeExecutionSnapshot {
                attempt_id: report.attempt_id,
                coverage: observation,
                terminal_host_materialization_count: observation.host_materialization_count(),
                intermediate_device_to_host_materialization_count,
                metrics,
            });
        }
    }

    fn sealed_graph_claim_calibration_snapshot(&self) -> Option<SealedAttemptGraphClaimCalibrationSnapshot> {
        self.sealed_graph_claim_calibration.get().cloned()
    }

    fn sealed_native_execution_snapshot(&self) -> Option<SealedAttemptNativeExecutionSnapshot> {
        self.sealed_native_execution.get().cloned()
    }
}

/// The attempt an [`Invocation`] belongs to, as planning and execution see it
/// through the invocation's config extension.
#[derive(Debug, Clone)]
pub(crate) struct InvocationAttempt {
    state: Arc<AttemptExecutionState>,
}

impl InvocationAttempt {
    fn new(state: Arc<AttemptExecutionState>) -> Self {
        Self { state }
    }

    /// The correlation id selected by the attempt owner, if the caller
    /// supplied one. This is planning metadata only.
    pub(crate) fn query_id(&self) -> Option<&AttemptQueryId> {
        self.state.query_id.as_ref()
    }

    /// The attempt's cancellation, the one token native work of this
    /// invocation observes.
    pub(crate) fn cancellation(&self) -> &AttemptCancellation {
        &self.state.cancellation
    }

    /// Engine-minted execution identity for this invocation. Graph execution
    /// services use it only to link an intentional non-physical follower to
    /// its physical leader in the sealed diagnostic projection.
    #[cfg(feature = "cugraph")]
    pub(crate) fn attempt_id(&self) -> AttemptId {
        self.state.handle.attempt_id()
    }

    /// Create the opaque native-admission binding for this invocation.
    ///
    /// Only adapter-owned planner code can construct this binding. In
    /// particular, the extension never exposes the raw `AttemptHandle` or
    /// observation-seal authority to DataFusion extensions.
    pub(crate) fn ticket_binding(&self) -> AttemptTicketBinding {
        AttemptTicketBinding {
            handle: self.state.handle.clone(),
            admission_rejection: self.state.admission_rejection.clone(),
            admission_binding: self.state.admission_binding.clone(),
            cancellation: self.state.cancellation.clone(),
            requested_query_cap_bytes: self.state.query_cap_bytes,
        }
    }

    /// Records the native physical plan that first needs admission.
    ///
    /// CTAS may execute that plan from `execute_logical_plan` before the
    /// adapter receives the outer DataFrame's physical plan.  Establish the
    /// attempt's final-plan disposition at ticket construction so the later
    /// query-service grant always has a valid lifecycle predecessor.
    pub(crate) fn native_admission_planned(&self, plan: &dyn ExecutionPlan) {
        let _ = self.state.handle.set_disposition(disposition_for(classify_final_plan(plan)));
        let _ = self.state.handle.record_planning_finished();
        if !self.state.result_stream_started.load(Ordering::Acquire) {
            self.state.set_sealed_metrics_target(plan);
        }
    }
}

/// Owner of one adapter invocation attempt.
///
/// One attempt executes one invocation into one result stream. The
/// terminalization authority moves into that stream when execution begins;
/// until then the attempt itself commits planning outcomes. A second
/// [`sql`](Self::sql), [`execute_dataframe`](Self::execute_dataframe), or
/// [`execute_physical_plan`](Self::execute_physical_plan) call finds no
/// authority left and fails instead of running again.
pub struct AlgeonAttempt {
    owner: Mutex<Option<AttemptOwner>>,
    state: Arc<AttemptExecutionState>,
    invocation: Invocation,
    completion_factory: AttemptCompletionFactory,
    session: GpuBackendSession,
}

impl std::fmt::Debug for AlgeonAttempt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlgeonAttempt").field("attempt_id", &self.attempt_id()).finish()
    }
}

impl AlgeonGpuBackend {
    /// Begins an attempt after validating opaque caller correlation fields.
    pub fn begin_attempt(&self, options: AttemptOptions) -> Result<AlgeonAttempt> {
        self.begin_attempt_with_cap(options, None)
    }

    /// Begins an attempt that requests an exact immutable query memory cap.
    ///
    /// The cap is carried through the attempt-bound admission ticket into the
    /// attempt's one query-service admission. Admission rejects a cap that
    /// cannot fit instead of resizing it.
    pub fn begin_attempt_with_query_cap_bytes(
        &self,
        options: AttemptOptions,
        query_cap_bytes: u64,
    ) -> Result<AlgeonAttempt> {
        self.begin_attempt_with_cap(options, Some(query_cap_bytes))
    }

    fn begin_attempt_with_cap(&self, options: AttemptOptions, query_cap_bytes: Option<u64>) -> Result<AlgeonAttempt> {
        AlgeonAttempt::begin(self.session_capability(), options, query_cap_bytes)
    }

    /// Drains up to `max` reports whose completion receivers were all
    /// dropped before publication. Drained reports leave the queue; the
    /// backend retains no copy.
    pub fn take_abandoned_attempt_reports(&self, max: usize) -> Vec<Arc<QueryAttemptReport>> {
        self.session_capability().take_abandoned_attempt_reports(max)
    }

    pub fn attempt_counters_snapshot(&self) -> AttemptBackendCountersSnapshot {
        self.session_capability().attempt_counters_snapshot()
    }
}

impl AlgeonAttempt {
    fn begin(session: &GpuBackendSession, options: AttemptOptions, query_cap_bytes: Option<u64>) -> Result<Self> {
        let query_id = options.query_id.map(AttemptQueryId::new).transpose().map_err(|error| {
            Error::from_native("algeon_datafusion.attempt.begin", "invalid attempt query correlation", error)
        })?;
        let trace_context = options.trace_context.map(AttemptTraceContext::new).transpose().map_err(|error| {
            Error::from_native("algeon_datafusion.attempt.begin", "invalid attempt trace correlation", error)
        })?;
        let state_query_id = query_id.clone();
        let begin = session
            .begin_attempt(BeginAttemptRequest {
                query_id,
                trace_context,
                deadline: options.deadline,
                observation: options.observation,
            })
            .map_err(|error| {
                Error::from_native("algeon_datafusion.attempt.begin", "failed to begin execution attempt", error)
            })?;
        let state = Arc::new(AttemptExecutionState {
            handle: begin.owner.handle(),
            admission_rejection: begin.owner.admission_rejection(),
            admission_binding: begin.owner.admission_binding(),
            query_id: state_query_id,
            query_cap_bytes,
            cancellation: begin.owner.cancellation(),
            sealed_metrics_target: Mutex::new(None),
            result_stream_started: AtomicBool::new(false),
            sealed_graph_claim_calibration: OnceLock::new(),
            sealed_native_execution: OnceLock::new(),
        });
        // The attempt object holds the minting factory, not a receiver, so
        // only caller-held receivers count toward the abandoned-report
        // decision.
        let completion_factory = begin.completion;
        let metrics_state = Arc::clone(&state);
        begin.owner.register_sealed_report_sink(move |report| {
            metrics_state.publish_sealed_metrics(&report);
        });
        Ok(Self {
            owner: Mutex::new(Some(begin.owner)),
            invocation: Invocation::for_attempt(InvocationAttempt::new(Arc::clone(&state))),
            state,
            completion_factory,
            session: session.clone(),
        })
    }

    pub fn attempt_id(&self) -> AttemptId {
        self.state.handle.attempt_id()
    }

    /// Returns an independent counted receiver for this attempt's immutable terminal report.
    ///
    /// Each call mints a distinct counted receiver. Receivers do not hold GPU
    /// leases or delay resource cleanup. To observe terminal reports reliably across
    /// all outcomes (planning failures, cancellation, stream errors, or normal EOF),
    /// callers should obtain a receiver before planning starts and retain it until publication.
    ///
    /// When terminal publication occurs with zero active counted receivers, the sealed
    /// report is best-effort routed to the backend's configured abandoned-report queue.
    /// Receivers created after publication can still read the immutable report, but
    /// will not undo an already-routed abandoned report.
    pub fn completion(&self) -> AttemptCompletion {
        self.completion_factory.completion()
    }

    /// Returns the immutable graph claim-versus-peak snapshot after this
    /// attempt has sealed a native execution root.
    ///
    /// `None` means either the attempt has not sealed yet or no native root
    /// was elected. The snapshot is diagnostic only and cannot mutate the
    /// engine's admission, ledger, or allocator state.
    pub fn sealed_graph_claim_calibration_snapshot(&self) -> Option<SealedAttemptGraphClaimCalibrationSnapshot> {
        self.state.sealed_graph_claim_calibration_snapshot()
    }
    pub fn sealed_native_execution_snapshot(&self) -> Option<SealedAttemptNativeExecutionSnapshot> {
        self.state.sealed_native_execution_snapshot()
    }
    pub fn cancellation(&self) -> AttemptCancellation {
        self.state.cancellation.clone()
    }

    /// Native planning diagnostics of this attempt's latest planning pass.
    ///
    /// `None` means the attempt has not planned, or planned on a session
    /// without the native optimizer rule.
    pub fn planning_diagnostics(&self) -> Option<NativePlanningDiagnostics> {
        self.invocation.planning_diagnostics()
    }

    /// Registers a read-only handoff for the immutable report produced at the
    /// attempt's sole observation seal point.
    ///
    /// The observer has no terminalization authority. The attempt owner
    /// publishes the report before it delivers its sealed observation, so the
    /// callback can retain the exact completed report together with the
    /// snapshot that embeds it without a detached completion task.
    pub fn register_sealed_report_observer<F>(&self, observer: F)
    where F: Fn(Option<Arc<QueryAttemptReport>>, SealedRuntimeObservationSnapshot) + Send + Sync + 'static {
        self.register_sealed_execution_observer(move |report, snapshot, _native_execution| observer(report, snapshot));
    }

    /// Like [`Self::register_sealed_report_observer`], and also hands over the
    /// native execution snapshot. It is sealed by the report sink, which runs
    /// before the owner delivers the sealed observation.
    pub(crate) fn register_sealed_execution_observer<F>(&self, observer: F)
    where F: Fn(
                Option<Arc<QueryAttemptReport>>,
                SealedRuntimeObservationSnapshot,
                Option<SealedAttemptNativeExecutionSnapshot>,
            ) + Send
            + Sync
            + 'static {
        let factory = self.completion_factory.clone();
        let state = Arc::clone(&self.state);
        self.state.admission_binding.register_seal_observer(move |snapshot| {
            let report = factory.try_report();
            if report.is_none() {
                // `AttemptOwner::finish_with_footer` publishes the report
                // before the owner exposes this sealed snapshot. Keep the
                // coordinator fail-closed if that engine invariant is ever
                // violated rather than retaining a fabricated projection.
                // The invariant is preserved in every build; emit its
                // operational diagnostic before retaining a fail-closed
                // projection.
                tracing::error!("attempt observation sealed before its immutable report was published");
            }
            observer(report, snapshot, state.sealed_native_execution_snapshot());
        });
    }

    /// Commits through the still-owned terminalization authority. Once the
    /// authority moved into a result stream this is a no-op; the stream is the
    /// only remaining    /// committer and the engine state machine enforces
    /// exactly-once regardless.
    fn commit(&self, outcome: RuntimeTerminalOutcome) {
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(owner) = owner.as_ref() {
            let _ = owner.commit_terminal(outcome);
        }
    }

    fn commit_datafusion_error(&self, error: &DataFusionError) {
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(owner) = owner.as_ref() {
            commit_datafusion_error(owner, error, &self.state.cancellation);
        }
    }

    /// Moves the terminalization authority into the result stream. An attempt
    /// executes exactly one invocation.
    fn take_owner(&self) -> DataFusionResult<AttemptOwner> {
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take().ok_or_else(|| {
            DataFusionError::from(Error::new(
                ErrorCode::Internal,
                "algeon_datafusion.attempt.execute",
                "execution attempt already has a result stream",
            ))
        })?;
        self.state.result_stream_started.store(true, Ordering::Release);
        Ok(owner)
    }

    fn invocation_state(&self, state: SessionState) -> SessionState {
        let mut builder = SessionStateBuilder::new_from_existing(state);
        self.invocation.install(builder.config().get_or_insert_with(Default::default));
        builder.build()
    }

    fn invocation_context(&self, ctx: &SessionContext) -> SessionContext {
        SessionContext::new_with_state(self.invocation_state(ctx.state()))
    }

    fn cancelled_error(&self, operation: &'static str) -> DataFusionError {
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let error = match owner.as_ref() {
            Some(owner) => commit_effective_cancellation(owner, &self.state.cancellation, operation),
            None => self.state.cancellation.check(operation).err(),
        }
        .expect("cancellation was checked after it was requested");
        DataFusionError::from(Error::from_native(operation, "execution attempt was cancelled", error))
    }

    /// Rejects planning when the backend started closing after this attempt
    /// won the begin gate: the attempt terminalizes `unplanned` with bounded
    /// `backend_closing` evidence instead of producing a plan.
    fn fail_if_backend_closing(&self, operation: &'static str) -> DataFusionResult<()> {
        if !self.session.is_closing() {
            return Ok(());
        }
        let error =
            Error::new(ErrorCode::Cancelled, operation, "backend is closing; the attempt cannot start planning")
                .with_fact("reason", "backend_closing");
        // Close and caller cancellation are independent typed causes.
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(owner) = owner.as_ref() {
            let _ = owner.commit_failure_summary(AttemptFailureSummary::backend_closing());
        }
        Err(DataFusionError::from(error))
    }

    /// Plans SQL in a fresh invocation-local session state.
    ///
    /// DataFusion executes a CTAS input while planning the statement; that
    /// execution admits under this attempt and shares the grant with the
    /// statement's result, which [`Self::execute_physical_plan`] runs.
    pub fn plan_sql<'a>(
        &'a self,
        ctx: &'a SessionContext,
        sql: &'a str,
    ) -> BoxFuture<'a, DataFusionResult<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            let invocation = self.invocation_context(ctx);
            self.plan_logical("algeon_datafusion.attempt.plan_sql", async {
                invocation.sql(sql).await?.create_physical_plan().await
            })
            .await
        })
    }

    /// Plans and executes SQL in this attempt's single invocation.
    pub fn sql<'a>(
        &'a self,
        ctx: &'a SessionContext,
        sql: &'a str,
    ) -> impl Future<Output = DataFusionResult<AttemptStream>> + Send + 'a {
        self.plan_sql(ctx, sql).map(move |plan| plan.and_then(|plan| self.execute_physical_plan(plan, ctx)))
    }

    /// Plans and executes an existing `DataFrame` in this attempt's single
    /// invocation. The frame's logical plan is replanned inside the
    /// invocation-local session state so native fragments bind this attempt's
    /// admission handle.
    pub async fn execute_dataframe(&self, df: DataFrame) -> DataFusionResult<AttemptStream> {
        let operation = "algeon_datafusion.attempt.execute_dataframe";
        let (state, logical) = df.into_parts();
        let invocation = SessionContext::new_with_state(self.invocation_state(state));
        let plan = self
            .plan_logical(operation, async { DataFrame::new(invocation.state(), logical).create_physical_plan().await })
            .await?;
        self.start_stream(plan, invocation.task_ctx())
    }

    /// Executes an already-optimized physical plan in this attempt's single
    /// invocation.
    pub fn execute_physical_plan(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        ctx: &SessionContext,
    ) -> DataFusionResult<AttemptStream> {
        self.check_startable("algeon_datafusion.attempt.execute")?;
        if self.state.handle.disposition().is_none() {
            let _ = self.state.handle.set_disposition(disposition_for(classify_final_plan(plan.as_ref())));
        }
        let invocation = self.invocation_context(ctx);
        self.start_stream(plan, invocation.task_ctx())
    }

    /// Fails, after committing the matching terminal outcome, when the attempt
    /// is cancelled or past its deadline, or when the backend is closing.
    fn check_startable(&self, operation: &'static str) -> DataFusionResult<()> {
        if self.cancellation().check(operation).is_err() {
            return Err(self.cancelled_error(operation));
        }
        self.fail_if_backend_closing(operation)
    }

    /// Runs `planning` in this attempt's invocation and records the planning
    /// boundary and the final plan's disposition. A planning error or panic
    /// commits the attempt's terminal outcome.
    async fn plan_logical(
        &self,
        operation: &'static str,
        planning: impl Future<Output = DataFusionResult<Arc<dyn ExecutionPlan>>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        self.check_startable(operation)?;
        let _ = self.state.handle.record_planning_started();
        let plan = match AssertUnwindSafe(planning).catch_unwind().await {
            Ok(Ok(plan)) => plan,
            Ok(Err(error)) => {
                self.commit_datafusion_error(&error);
                return Err(error);
            },
            Err(panic) => {
                self.commit(RuntimeTerminalOutcome::Panicked);
                resume_unwind(panic);
            },
        };
        if self.cancellation().check(operation).is_err() {
            return Err(self.cancelled_error(operation));
        }
        let _ = self.state.handle.set_disposition(disposition_for(classify_final_plan(plan.as_ref())));
        let _ = self.state.handle.record_planning_finished();
        Ok(plan)
    }

    /// Moves the terminalization authority into the one stream that executes
    /// `plan`, then elects the plan's native root for the sealed metrics.
    fn start_stream(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<AttemptStream> {
        let owner = self.take_owner()?;
        self.state.set_sealed_metrics_target(plan.as_ref());
        let task_context = self.task_context_with_attempt_ticket(plan.as_ref(), task_context);
        match execute_physical_stream(plan, task_context) {
            Ok(inner) => Ok(AttemptStream::new(inner, owner, Arc::clone(&self.state))),
            Err(error) => {
                commit_datafusion_error(&owner, &error, &self.state.cancellation);
                Err(error)
            },
        }
    }

    /// Binds the invocation's admission ticket to `task_context` when `plan`
    /// admits native work. For a plan planned outside this attempt, the
    /// source and evidence its admission node holds mint the ticket.
    fn task_context_with_attempt_ticket(
        &self,
        plan: &dyn ExecutionPlan,
        task_context: Arc<TaskContext>,
    ) -> Arc<TaskContext> {
        let Some(admission) = native_admission_plan(plan) else { return task_context };
        let (ticket, _) = self
            .invocation
            .attempt_admission_ticket(admission.source(), Some(admission.evidence()))
            .expect("an attempt's invocation carries its attempt");
        task_context_with_query_admission_ticket(task_context, ticket)
    }
}

/// Runs `execute` as one internal attempt begun on `session`, for a native
/// admission root executed without an attempt of its own (a plain DataFusion
/// `collect`). `execute` receives a task context carrying the attempt's
/// invocation; the returned stream commits the attempt's terminal outcome on
/// end of stream, error, drop, or panic.
pub(crate) fn execute_in_internal_attempt(
    session: &GpuBackendSession,
    plan: &dyn ExecutionPlan,
    task_context: Arc<TaskContext>,
    execute: impl FnOnce(Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream>,
) -> DataFusionResult<SendableRecordBatchStream> {
    let attempt = AlgeonAttempt::begin(session, AttemptOptions::default(), None).map_err(DataFusionError::from)?;
    let _ = attempt.state.handle.set_disposition(disposition_for(classify_final_plan(plan)));
    let owner = attempt.take_owner()?;
    attempt.state.set_sealed_metrics_target(plan);
    let mut session_config = task_context.session_config().clone();
    attempt.invocation.install(&mut session_config);
    let task_context = Arc::new(TaskContext::new(
        task_context.task_id(),
        task_context.session_id(),
        session_config,
        task_context.scalar_functions().clone(),
        task_context.higher_order_functions().clone(),
        task_context.aggregate_functions().clone(),
        task_context.window_functions().clone(),
        task_context.runtime_env(),
    ));
    match execute(task_context) {
        Ok(inner) => Ok(Box::pin(AttemptStream::new(inner, owner, Arc::clone(&attempt.state)))),
        Err(error) => {
            commit_datafusion_error(&owner, &error, &attempt.state.cancellation);
            Err(error)
        },
    }
}

/// Begins a native attempt on `source`'s backend and mints its invocation's
/// ticket from `evidence`, for tests that drive a ticket without planning.
/// The ticket admits at most once, under the returned attempt.
#[cfg(test)]
pub(crate) fn attempt_ticket_for_test(
    source: QueryAdmissionSource,
    evidence: Option<AttemptCapabilityEvidence>,
) -> (AlgeonAttempt, Arc<QueryAdmissionTicket>) {
    attempt_ticket_with_options_for_test(source, evidence, AttemptOptions::default())
}

/// [`attempt_ticket_for_test`] for an attempt begun with `options`.
#[cfg(test)]
pub(crate) fn attempt_ticket_with_options_for_test(
    source: QueryAdmissionSource,
    evidence: Option<AttemptCapabilityEvidence>,
    options: AttemptOptions,
) -> (AlgeonAttempt, Arc<QueryAdmissionTicket>) {
    let attempt = AlgeonAttempt::begin(source.session(), options, None).expect("a test attempt begins");
    attempt.state.handle.set_disposition(AttemptDisposition::Native).expect("a new attempt accepts a disposition");
    let (ticket, _) = attempt
        .invocation
        .attempt_admission_ticket(source, evidence)
        .expect("an attempt's invocation carries its attempt");
    (attempt, ticket)
}

fn native_admission_plan(plan: &dyn ExecutionPlan) -> Option<&NativeAdmissionExec> {
    // `NativeAdmissionExec` is usually the physical root, but DDL plans can
    // execute it below a DataFusion operator, and a GPU function occurrence
    // can wrap it. The invocation task context must carry the same attempt
    // ticket before any of these shapes starts execution.
    let mut current = plan;
    loop {
        if let Some(admission) = (current as &dyn Any).downcast_ref::<NativeAdmissionExec>() {
            return Some(admission);
        }
        let Some(input) = transparent_root_wrapper_input(current) else {
            break;
        };
        current = input.as_ref();
    }
    current.children().into_iter().find_map(|child| native_admission_plan(child.as_ref()))
}

fn disposition_for(disposition: FinalPlanDisposition) -> AttemptDisposition {
    match disposition {
        FinalPlanDisposition::DataFusion => AttemptDisposition::CpuFallback,
        FinalPlanDisposition::Native => AttemptDisposition::Native,
        FinalPlanDisposition::Mixed => AttemptDisposition::PartialNative,
    }
}

/// Captures a returned DataFusion error by reference before reducing it to a
/// closed terminal outcome. Only native errors contribute a bounded message;
/// adapter-only errors use the deliberately text-free fallback.
fn commit_datafusion_error(owner: &AttemptOwner, error: &DataFusionError, cancellation: &AttemptCancellation) {
    if commit_effective_cancellation(owner, cancellation, "algeon_datafusion.attempt.terminal_error").is_some() {
        return;
    }
    match error.find_algeon_error() {
        Some(error) => commit_adapter_error(owner, error),
        None => {
            let _ = owner.commit_failure_summary(AttemptFailureSummary::datafusion_fallback());
        },
    }
}

/// Commits an effective cancellation or deadline as the attempt's terminal
/// error, so the report keeps its bounded reason, and returns that error.
/// `None` means the attempt is neither cancelled nor past its deadline.
fn commit_effective_cancellation(
    owner: &AttemptOwner,
    cancellation: &AttemptCancellation,
    operation: &'static str,
) -> Option<QueryError> {
    let error = cancellation.check(operation).err()?;
    let _ = owner.commit_query_error(&error);
    Some(error)
}

fn commit_adapter_error(owner: &AttemptOwner, error: &Error) {
    if error.code() == ErrorCode::Cancelled {
        let _ = owner.commit_terminal(RuntimeTerminalOutcome::Cancelled);
    } else if let Some(native) = error.source_error().and_then(|source| source.downcast_ref::<QueryError>()) {
        let _ = owner.commit_query_error(native);
    } else {
        let _ = owner.commit_failure_summary(AttemptFailureSummary::datafusion_fallback());
    }
}

/// Result stream that commits the attempt at EOF, error, cancellation, drop, or panic.
///
/// The stream owns the attempt's terminalization authority: success is
/// terminal only at EOF, every other path is exactly-once, and dropping the
/// stream initiates cooperative cancellation without blocking the runtime
/// worker it is dropped on.
pub struct AttemptStream {
    /// Kept only until the wrapper commits its terminal outcome. Dropping it
    /// releases no grant: the attempt's terminal commit does that.
    inner: Option<SendableRecordBatchStream>,
    schema: SchemaRef,
    owner: AttemptOwner,
    state: Arc<AttemptExecutionState>,
    physical_start_recorded: bool,
    terminal: bool,
}

impl AttemptStream {
    fn new(inner: SendableRecordBatchStream, owner: AttemptOwner, state: Arc<AttemptExecutionState>) -> Self {
        let schema = inner.schema();
        Self { inner: Some(inner), schema, owner, state, physical_start_recorded: false, terminal: false }
    }

    /// Returns the immutable graph claim-versus-peak snapshot once the owning
    /// attempt has sealed. This is the same projection exposed by
    /// [`AlgeonAttempt::sealed_graph_claim_calibration_snapshot`].
    pub fn sealed_graph_claim_calibration_snapshot(&self) -> Option<SealedAttemptGraphClaimCalibrationSnapshot> {
        self.state.sealed_graph_claim_calibration_snapshot()
    }

    fn commit(&self, outcome: RuntimeTerminalOutcome) {
        let _ = self.owner.commit_terminal(outcome);
    }

    fn commit_datafusion_error(&self, error: &DataFusionError) {
        commit_datafusion_error(&self.owner, error, &self.state.cancellation);
    }

    /// Commits the outer terminal result before destroying the physical
    /// stream, so the attempt's outcome is the root stream's result rather
    /// than a state its inputs' teardown reached first.
    fn finish_terminal(&mut self, outcome: RuntimeTerminalOutcome) {
        self.terminal = true;
        self.commit(outcome);
        drop(self.inner.take());
    }

    /// Finalizes an effective cancellation through its typed native error so
    /// deadline expiry retains its bounded reason in the immutable report.
    fn finish_cancelled(&mut self) {
        self.terminal = true;
        let _ = commit_effective_cancellation(
            &self.owner,
            &self.state.cancellation,
            "algeon_datafusion.attempt.result_drain",
        );
        drop(self.inner.take());
    }

    fn cancelled_error(&self) -> DataFusionError {
        let operation = "algeon_datafusion.attempt.result_drain";
        match self.state.cancellation.check(operation) {
            Err(error) => {
                DataFusionError::from(Error::from_native(operation, "execution attempt was cancelled", error))
            },
            Ok(()) => {
                DataFusionError::from(Error::new(ErrorCode::Cancelled, operation, "execution attempt was cancelled"))
            },
        }
    }
}

impl RecordBatchStream for AttemptStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Stream for AttemptStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.terminal {
            return Poll::Ready(None);
        }
        if !this.state.handle.is_terminal()
            && this.state.cancellation.check("algeon_datafusion.attempt.result_drain").is_err()
        {
            // The cancellation error is the stream's final item. The handle
            // is terminal after `commit`, so it cannot guard a later poll.
            this.finish_cancelled();
            return Poll::Ready(Some(Err(this.cancelled_error())));
        }
        if !this.physical_start_recorded {
            this.physical_start_recorded = true;
            // CPU-only plans begin physical execution at first poll; native
            // plans record it truthfully at admission instead (a queued or
            // rejected admission therefore leaves this boundary absent).
            if this.state.handle.disposition() == Some(AttemptDisposition::CpuFallback) {
                let _ = this.state.handle.record_physical_execution_started();
            }
        }
        let poll = catch_unwind(AssertUnwindSafe(|| {
            Pin::new(this.inner.as_mut().expect("live attempt stream retains its inner stream")).poll_next(cx)
        }));
        match poll {
            Err(panic) => {
                this.finish_terminal(RuntimeTerminalOutcome::Panicked);
                resume_unwind(panic);
            },
            Ok(Poll::Ready(None)) => {
                this.finish_terminal(RuntimeTerminalOutcome::Succeeded);
                Poll::Ready(None)
            },
            Ok(Poll::Ready(Some(Err(error)))) => {
                this.terminal = true;
                this.commit_datafusion_error(&error);
                drop(this.inner.take());
                Poll::Ready(Some(Err(error)))
            },
            Ok(other) => other,
        }
    }
}

impl Drop for AttemptStream {
    fn drop(&mut self) {
        if !self.state.handle.is_terminal() {
            // Flight's unwind guard cancels the shared token while it records
            // the coordinator-side panic.  Preserve the primary unwind
            // outcome for the immutable attempt report instead of letting
            // that cleanup cancellation downgrade it to `Cancelled`.
            if std::thread::panicking() {
                self.finish_terminal(RuntimeTerminalOutcome::Panicked);
            } else if self.state.cancellation.is_cancelled() || self.state.cancellation.is_expired() {
                self.finish_cancelled();
            } else {
                self.state.cancellation.cancel();
                self.finish_terminal(RuntimeTerminalOutcome::Dropped);
            }
        }
        drop(self.inner.take());
    }
}

impl std::fmt::Debug for AttemptStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttemptStream").field("attempt_id", &self.owner.attempt_id()).finish_non_exhaustive()
    }
}

/// Server hooks for attempt phases before, or instead of, a result stream.
///
/// An attempt commits one terminal outcome. While [`AlgeonAttempt`] still
/// holds its terminalization authority, the committing hooks below commit
/// through it; once that authority has moved into a result stream they
/// commit nothing and the stream is the only committer. The attempt owner
/// rejects a later terminal commit with `DoubleTerminalization`, which these
/// hooks ignore, so the first committed outcome stands.
#[cfg(feature = "server")]
impl AlgeonAttempt {
    /// Records that planning started and returns a session context bound to
    /// this attempt, whose planning copies `execution` into every GPU node.
    ///
    /// Fails, after committing the matching terminal outcome, when the
    /// attempt is already cancelled or past its deadline, or when the backend
    /// is closing.
    pub(crate) fn planning_context(
        &self,
        ctx: &SessionContext,
        execution: InvocationExecution,
    ) -> DataFusionResult<SessionContext> {
        self.check_startable("algeon_datafusion.attempt.flight_planning")?;
        let _ = self.state.handle.record_planning_started();
        self.invocation.set_execution(execution);
        Ok(self.invocation_context(ctx))
    }

    /// Commits the terminal outcome for a DataFusion planning error.
    ///
    /// An effective cancellation or deadline takes precedence over `error`.
    /// Otherwise an Algeon error carried by `error` is committed as by
    /// [`Self::planning_failed_adapter`], and any other error commits a
    /// text-free DataFusion fallback summary.
    pub(crate) fn planning_failed(&self, error: &DataFusionError) {
        self.commit_datafusion_error(error);
    }

    /// Commits the terminal outcome for an Algeon DataFusion adapter planning
    /// error.
    ///
    /// An effective cancellation or deadline takes precedence over `error`.
    /// Otherwise a `Cancelled` error commits `Cancelled`, an error whose source
    /// is a native query error commits that error, and any other error commits
    /// a text-free DataFusion fallback summary.
    pub(crate) fn planning_failed_adapter(&self, error: &Error) {
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(owner) = owner.as_ref()
            && commit_effective_cancellation(
                owner,
                &self.state.cancellation,
                "algeon_datafusion.attempt.terminal_error",
            )
            .is_none()
        {
            commit_adapter_error(owner, error);
        }
    }

    /// Records the final physical plan's execution disposition and the end of
    /// planning without committing a terminal outcome.
    pub(crate) fn physical_planning_finished(&self, plan: &dyn ExecutionPlan) {
        let _ = self.state.handle.set_disposition(disposition_for(classify_final_plan(plan)));
        let _ = self.state.handle.record_planning_finished();
    }

    /// Cancels the attempt's cancellation token and commits `Cancelled`.
    pub(crate) fn cancel_before_stream(&self) {
        self.state.cancellation.cancel();
        self.commit(RuntimeTerminalOutcome::Cancelled);
    }

    /// Commits `Succeeded` for a request that finishes without a result
    /// stream, such as DDL. A CTAS input that ran during planning shares the
    /// attempt's grant, which the attempt releases once this outcome commits.
    pub(crate) fn complete_without_result_stream(&self) {
        self.commit(RuntimeTerminalOutcome::Succeeded);
    }
}

#[cfg(all(test, not(feature = "cugraph")))]
mod tests {
    use std::sync::Arc;

    use arrow::{
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::{
        datasource::MemTable,
        execution::SessionStateBuilder,
        prelude::{SessionConfig, SessionContext},
    };
    use futures::StreamExt;

    use super::*;
    use crate::{
        backend::{AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit, InstalledTestGpuBackend},
        cugraph_sql::CugraphSqlConfig,
        error::ErrorCode,
        native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
        planner::AlgeonGpuExecutionMode,
        session::AlgeonSessionStateBuilderExt,
    };

    #[tokio::test]
    async fn feature_disabled_cugraph_rejection_stays_typed_through_an_attempt_stream() {
        let backend = InstalledTestGpuBackend::with_config_and_execution_mode(
            AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
            ),
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 }),
            AlgeonGpuQueuePolicy::default(),
            AlgeonGpuExecutionMode::NativeRequired,
        );
        let builder = backend
            .backend()
            .install_on(
                SessionStateBuilder::new_with_default_features()
                    .with_config(SessionConfig::new().with_target_partitions(1)),
            )
            .expect("feature-off cuGraph SQL backend install must succeed");
        let state = builder
            .try_with_cugraph_sql(CugraphSqlConfig::default())
            .expect("feature-off cuGraph SQL must install on an attempt-capable backend")
            .build();
        let ctx = SessionContext::new_with_state(state);
        let edges = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let empty_batch = RecordBatch::new_empty(Arc::clone(&edges));
        ctx.register_table("edges", Arc::new(MemTable::try_new(edges, vec![vec![empty_batch]]).unwrap()))
            .expect("feature-off source schema must register");

        let attempt = backend.backend().begin_attempt(AttemptOptions::default()).unwrap();
        let mut stream = attempt
            .sql(&ctx, "SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM edges))")
            .await
            .expect("feature-off execution must fail from the returned result stream");
        let error = stream
            .next()
            .await
            .expect("feature-off stream must emit one terminal rejection")
            .expect_err("feature-off cuGraph execution must reject");
        let typed = error.find_algeon_error().expect("attempt stream must retain the typed feature rejection");
        assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
        assert_eq!(typed.operation(), "algeon_datafusion.cugraph_sql.execute");
        assert_eq!(typed.fact_value("reason"), Some("required_feature_disabled"));
        assert_eq!(typed.fact_value("required_feature"), Some("cugraph"));
        assert_eq!(typed.fact_value("host_fallback"), Some("disabled"));
        drop(stream);
        let _report = attempt.completion().await;
    }
}
