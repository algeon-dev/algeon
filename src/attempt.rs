//! Explicit execution-attempt adapter boundary.

use std::{
    any::Any,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Instant,
};

use datafusion::{
    common::{
        DataFusionError, Result as DataFusionResult,
        config::{ConfigExtension, ExtensionOptions},
    },
    execution::SessionStateBuilder,
    physical_plan::{
        ExecutionPlan, RecordBatchStream, SendableRecordBatchStream, execute_stream as execute_physical_stream,
    },
    prelude::{DataFrame, SessionContext},
};
use futures::{FutureExt, Stream, future::BoxFuture};
use nexus_query_engine::{
    attempt::{AttemptAdmissionBinding, AttemptAdmissionRejection, AttemptHandle, AttemptOwner, BeginAttemptRequest},
    observability::AttemptFailureSummary,
};
pub use nexus_query_engine::{
    attempt::{AttemptBackendCountersSnapshot, AttemptCancellation, AttemptCompletion},
    graph_memory::GraphClaimCalibrationRow,
    observability::{
        AttemptDisposition, AttemptId, AttemptQueryId, AttemptTraceContext, QueryAttemptReport,
        RuntimeObservationConfig, RuntimeTerminalOutcome, SealedRuntimeObservationSnapshot,
    },
};

pub use crate::native_exec::{
    GraphClaimCalibrationExecution, GraphClaimCalibrationNonPhysicalReason, SealedAttemptGraphClaimCalibrationSnapshot,
};
use crate::{
    backend::NexusGpuBackend,
    error::{DataFusionErrorExt, Error, Result},
    native_exec::{
        SealedAttemptMetricsTarget,
        admission::{
            AttemptTicketBinding, NativeAdmissionExec, QueryAdmissionTicket, task_context_with_query_admission_ticket,
        },
        sealed_attempt_metrics_target, transparent_root_wrapper_input,
    },
    planner::{FinalPlanDisposition, classify_final_plan},
};

/// Inputs used to create an execution attempt before DataFusion planning.
#[derive(Debug, Clone)]
pub struct AttemptOptions {
    pub query_id: Option<String>,
    pub trace_context: Option<String>,
    pub deadline: Option<Instant>,
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
    cancellation: AttemptCancellation,
    ticket: Mutex<Option<Arc<crate::native_exec::admission::QueryAdmissionTicket>>>,
    /// The one elected native root that may receive the fixed-cardinality
    /// attempt metric after this attempt's report has sealed. The target is
    /// consumed at seal, so no wrapper can publish a second row.
    sealed_metrics_target: Mutex<Option<SealedAttemptMetricsTarget>>,
    /// Typed graph claim/peak rows frozen by the elected target at the same
    /// seal point as the immutable attempt report.
    sealed_graph_claim_calibration: Mutex<Option<SealedAttemptGraphClaimCalibrationSnapshot>>,
}

impl AttemptExecutionState {
    fn set_sealed_metrics_target(&self, plan: &dyn ExecutionPlan) {
        let Some(target) = sealed_attempt_metrics_target(plan) else { return };
        let mut slot = self.sealed_metrics_target.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Planning supplies a fallback for paths such as CTAS; the actual
        // execution root replaces it before terminal publication so a replan
        // never projects into an unexecuted wrapper.
        *slot = Some(target);
    }

    fn publish_sealed_metrics(&self, report: &QueryAttemptReport) {
        let target = self.sealed_metrics_target.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(target) = target {
            let snapshot = target.record(report);
            *self.sealed_graph_claim_calibration.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(snapshot);
        }
    }

    fn sealed_graph_claim_calibration_snapshot(&self) -> Option<SealedAttemptGraphClaimCalibrationSnapshot> {
        self.sealed_graph_claim_calibration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
}

/// Config extension carrying attempt-local state.  It intentionally has no SQL
/// settings: attempts are installed only into invocation-local session states.
#[derive(Debug, Clone)]
pub(crate) struct NexusAttemptExtension {
    state: Arc<AttemptExecutionState>,
}

impl NexusAttemptExtension {
    fn new(state: Arc<AttemptExecutionState>) -> Self {
        Self { state }
    }

    /// The correlation id selected by the attempt owner, if the caller
    /// supplied one. This is planning metadata only.
    pub(crate) fn query_id(&self) -> Option<&AttemptQueryId> {
        self.state.query_id.as_ref()
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
        self.state.set_sealed_metrics_target(plan);
    }

    /// Retain the planner-created ticket so every execution path in this
    /// invocation can bind it to its task context.
    pub(crate) fn bind_ticket(&self, ticket: Arc<QueryAdmissionTicket>) {
        *self.state.ticket.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ticket);
    }

    /// Recover the opaque ticket from an invocation-local session config.
    ///
    /// DataFusion executes CTAS inside `execute_logical_plan`, before the
    /// adapter receives a physical plan to decorate. The inner DataFrame
    /// inherits this extension but not an `ActiveQueryAdmissionTicket`, so
    /// native admission must be able to promote this exact planner-created
    /// ticket into that task context.
    pub(crate) fn bound_ticket(&self) -> Option<Arc<QueryAdmissionTicket>> {
        self.state.ticket.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
}

impl ConfigExtension for NexusAttemptExtension {
    const PREFIX: &'static str = "datafusion_nexus_attempt";
}

impl ExtensionOptions for NexusAttemptExtension {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }
    fn set(&mut self, _key: &str, _value: &str) -> DataFusionResult<()> {
        Err(DataFusionError::Configuration("execution-attempt extension has no options".to_string()))
    }
    fn entries(&self) -> Vec<datafusion::common::config::ConfigEntry> {
        Vec::new()
    }
}

/// Owner of one adapter invocation attempt.
///
/// The terminalization authority moves into the result stream when execution
/// begins; until then the attempt itself commits planning outcomes.
pub struct NexusAttempt {
    owner: Mutex<Option<AttemptOwner>>,
    state: Arc<AttemptExecutionState>,
    completion: AttemptCompletion,
    session: nexus_query_engine::runtime::GpuBackendSession,
}

impl std::fmt::Debug for NexusAttempt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NexusAttempt").field("attempt_id", &self.attempt_id()).finish()
    }
}

impl NexusGpuBackend {
    /// Begins an attempt after validating opaque caller correlation fields.
    pub fn begin_attempt(&self, options: AttemptOptions) -> Result<NexusAttempt> {
        let query_id = options.query_id.map(AttemptQueryId::new).transpose().map_err(|error| {
            Error::from_native("datafusion_nexus.attempt.begin", "invalid attempt query correlation", error)
        })?;
        let trace_context = options.trace_context.map(AttemptTraceContext::new).transpose().map_err(|error| {
            Error::from_native("datafusion_nexus.attempt.begin", "invalid attempt trace correlation", error)
        })?;
        let state_query_id = query_id.clone();
        let begin = self
            .session_capability()
            .begin_attempt(BeginAttemptRequest {
                query_id,
                trace_context,
                deadline: options.deadline,
                observation: options.observation,
            })
            .map_err(|error| {
                Error::from_native("datafusion_nexus.attempt.begin", "failed to begin execution attempt", error)
            })?;
        let state = Arc::new(AttemptExecutionState {
            handle: begin.owner.handle(),
            admission_rejection: begin.owner.admission_rejection(),
            admission_binding: begin.owner.admission_binding(),
            query_id: state_query_id,
            cancellation: begin.owner.cancellation(),
            ticket: Mutex::new(None),
            sealed_metrics_target: Mutex::new(None),
            sealed_graph_claim_calibration: Mutex::new(None),
        });
        // The attempt object is a factory, not a receiver: the begin
        // completion is dropped here so only caller-held receivers count
        // toward the abandoned-report decision.
        let completion = begin.completion.uncounted();
        let metrics_state = Arc::clone(&state);
        begin.owner.register_sealed_report_sink(move |report| {
            metrics_state.publish_sealed_metrics(&report);
        });
        Ok(NexusAttempt {
            owner: Mutex::new(Some(begin.owner)),
            state,
            completion,
            session: self.session_capability().clone(),
        })
    }

    /// Drains reports whose completion receivers were all dropped.
    pub fn take_abandoned_attempt_reports(&self, max: usize) -> Vec<Arc<QueryAttemptReport>> {
        self.session_capability().take_abandoned_attempt_reports(max)
    }

    pub fn attempt_counters_snapshot(&self) -> AttemptBackendCountersSnapshot {
        self.session_capability().attempt_counters_snapshot()
    }
}

impl NexusAttempt {
    pub fn attempt_id(&self) -> AttemptId {
        self.state.handle.attempt_id()
    }
    pub fn completion(&self) -> AttemptCompletion {
        self.completion.clone()
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
    pub fn cancellation(&self) -> AttemptCancellation {
        self.state.cancellation.clone()
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
        let completion = self.completion.uncounted();
        self.state.admission_binding.register_seal_observer(move |snapshot| {
            let report = completion.try_report();
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
            observer(report, snapshot);
        });
    }

    /// Commits through the still-owned terminalization authority. Once the
    /// authority moved into a result stream this is a no-op; the stream is the
    /// only remaining committer and the engine state machine enforces
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
        self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take().ok_or_else(|| {
            DataFusionError::from(Error::new(
                crate::error::ErrorCode::Internal,
                "datafusion_nexus.attempt.execute",
                "execution attempt already has a result stream",
            ))
        })
    }

    fn invocation_state(&self, state: datafusion::execution::SessionState) -> datafusion::execution::SessionState {
        let mut builder = SessionStateBuilder::new_from_existing(state);
        let config = builder.config().get_or_insert_with(Default::default);
        config.options_mut().extensions.insert(NexusAttemptExtension::new(Arc::clone(&self.state)));
        builder.build()
    }

    fn invocation_context(&self, ctx: &SessionContext) -> SessionContext {
        SessionContext::new_with_state(self.invocation_state(ctx.state()))
    }

    /// Begin attempt-owned planning in an invocation-local DataFusion state.
    /// Flight uses this before it issues a ticket so logical-planning failures
    /// terminalize the same attempt that later owns physical execution.
    pub fn planning_context(&self, ctx: &SessionContext) -> DataFusionResult<SessionContext> {
        if self.cancellation().check("datafusion_nexus.attempt.flight_planning").is_err() {
            return Err(self.cancelled_error("datafusion_nexus.attempt.flight_planning"));
        }
        self.fail_if_backend_closing("datafusion_nexus.attempt.flight_planning")?;
        let _ = self.state.handle.record_planning_started();
        Ok(self.invocation_context(ctx))
    }

    fn cancelled_error(&self, operation: &'static str) -> DataFusionError {
        let error = self.cancellation().check(operation).expect_err("cancellation was checked after it was requested");
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(owner) = owner.as_ref() {
            let _ = owner.commit_query_error(&error);
        }
        DataFusionError::from(Error::from_native(operation, "execution attempt was cancelled", error))
    }

    pub fn planning_failed(&self, error: &DataFusionError) {
        self.commit_datafusion_error(error);
    }

    /// Commits an adapter-side planning rejection through the same owner as
    /// DataFusion failures, without exposing that owner to callers.
    pub fn planning_failed_adapter(&self, error: &Error) {
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(owner) = owner.as_ref() {
            if let Some(native) =
                error.source_error().and_then(|source| source.downcast_ref::<nexus_query_engine::error::QueryError>())
            {
                let _ = owner.commit_query_error(native);
            } else if error.code() == crate::error::ErrorCode::Cancelled {
                let _ = owner.commit_terminal(RuntimeTerminalOutcome::Cancelled);
            } else {
                let _ = owner.commit_failure_summary(AttemptFailureSummary::datafusion_fallback());
            }
        }
    }

    /// Records the completed physical-planning boundary for Flight's split
    /// GetFlightInfo/DoGet invocation. This remains separate from execution:
    /// the owner does not move into an [`AttemptStream`] until DoGet starts
    /// the finalized plan.
    pub fn physical_planning_finished(&self, plan: &dyn ExecutionPlan) {
        let _ = self.state.handle.set_disposition(disposition_for(classify_final_plan(plan)));
        let _ = self.state.handle.record_planning_finished();
    }

    /// Propagate a Flight cancellation before a result stream exists.
    pub fn cancel_before_stream(&self) {
        self.state.cancellation.cancel();
        self.commit(RuntimeTerminalOutcome::Cancelled);
    }

    /// Completes an invocation whose protocol result is deliberately empty.
    ///
    /// Flight uses this for a deduplicated prepared DDL execution: there is
    /// no DataFusion result stream to transfer the owner into, but the same
    /// pre-created attempt still needs one immutable terminal report.
    pub fn complete_without_result_stream(&self) {
        self.commit(RuntimeTerminalOutcome::Succeeded);
    }

    /// Rejects planning when the backend started closing after this attempt
    /// won the begin gate: the attempt terminalizes `unplanned` with bounded
    /// `backend_closing` evidence instead of producing a plan.
    fn fail_if_backend_closing(&self, operation: &'static str) -> DataFusionResult<()> {
        if !self.session.is_closing() {
            return Ok(());
        }
        let error = Error::new(
            crate::error::ErrorCode::Cancelled,
            operation,
            "backend is closing; the attempt cannot start planning",
        )
        .with_fact("reason", "backend_closing");
        // Close and caller cancellation are independent typed causes.
        let owner = self.owner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(owner) = owner.as_ref() {
            let _ = owner.commit_failure_summary(AttemptFailureSummary::backend_closing());
        }
        Err(DataFusionError::from(error))
    }

    /// Plans SQL in a fresh invocation-local session state.
    pub fn plan_sql<'a>(
        &'a self,
        ctx: &'a SessionContext,
        sql: &'a str,
    ) -> BoxFuture<'a, DataFusionResult<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            if self.cancellation().check("datafusion_nexus.attempt.plan_sql").is_err() {
                return Err(self.cancelled_error("datafusion_nexus.attempt.plan_sql"));
            }
            self.fail_if_backend_closing("datafusion_nexus.attempt.plan_sql")?;
            let invocation = self.invocation_context(ctx);
            let handle = self.state.handle.clone();
            let _ = handle.record_planning_started();
            let planned = AssertUnwindSafe(async { invocation.sql(sql).await?.create_physical_plan().await })
                .catch_unwind()
                .await;
            let plan = match planned {
                Ok(Ok(plan)) => plan,
                Ok(Err(error)) => {
                    self.planning_failed(&error);
                    return Err(error);
                },
                Err(panic) => {
                    self.commit(RuntimeTerminalOutcome::Panicked);
                    resume_unwind(panic);
                },
            };
            if self.cancellation().check("datafusion_nexus.attempt.plan_sql").is_err() {
                return Err(self.cancelled_error("datafusion_nexus.attempt.plan_sql"));
            }
            let _ = handle.set_disposition(disposition_for(classify_final_plan(plan.as_ref())));
            let _ = handle.record_planning_finished();
            Ok(plan)
        })
    }

    /// Plans and executes SQL in one invocation.
    pub fn sql<'a>(
        &'a self,
        ctx: &'a SessionContext,
        sql: &'a str,
    ) -> impl Future<Output = DataFusionResult<AttemptStream>> + Send + 'a {
        self.plan_sql(ctx, sql).map(move |plan| plan.and_then(|plan| self.execute_physical_plan(plan, ctx)))
    }

    /// Plans and executes an existing `DataFrame` in one invocation. The
    /// frame's logical plan is replanned inside the invocation-local session
    /// state so native fragments bind this attempt's admission handle.
    pub async fn execute_dataframe(&self, df: DataFrame) -> DataFusionResult<AttemptStream> {
        if self.cancellation().check("datafusion_nexus.attempt.execute_dataframe").is_err() {
            return Err(self.cancelled_error("datafusion_nexus.attempt.execute_dataframe"));
        }
        self.fail_if_backend_closing("datafusion_nexus.attempt.execute_dataframe")?;
        let handle = self.state.handle.clone();
        let _ = handle.record_planning_started();
        let (state, logical) = df.into_parts();
        let invocation = SessionContext::new_with_state(self.invocation_state(state));
        let planned =
            AssertUnwindSafe(async { DataFrame::new(invocation.state(), logical).create_physical_plan().await })
                .catch_unwind()
                .await;
        let plan = match planned {
            Ok(Ok(plan)) => plan,
            Ok(Err(error)) => {
                self.planning_failed(&error);
                return Err(error);
            },
            Err(panic) => {
                self.commit(RuntimeTerminalOutcome::Panicked);
                resume_unwind(panic);
            },
        };
        if self.cancellation().check("datafusion_nexus.attempt.execute_dataframe").is_err() {
            return Err(self.cancelled_error("datafusion_nexus.attempt.execute_dataframe"));
        }
        let _ = handle.set_disposition(disposition_for(classify_final_plan(plan.as_ref())));
        let _ = handle.record_planning_finished();
        self.state.set_sealed_metrics_target(plan.as_ref());
        let owner = self.take_owner()?;
        let task_context = self.task_context_with_attempt_ticket(plan.as_ref(), invocation.task_ctx());
        match execute_physical_stream(plan, task_context) {
            Ok(inner) => Ok(AttemptStream::new(inner, owner, Arc::clone(&self.state), self.completion.uncounted())),
            Err(error) => {
                commit_datafusion_error(&owner, &error, &self.state.cancellation);
                Err(error)
            },
        }
    }

    /// Executes an already-optimized physical plan under this attempt.
    pub fn execute_physical_plan(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        ctx: &SessionContext,
    ) -> DataFusionResult<AttemptStream> {
        if self.cancellation().check("datafusion_nexus.attempt.execute").is_err() {
            return Err(self.cancelled_error("datafusion_nexus.attempt.execute"));
        }
        self.fail_if_backend_closing("datafusion_nexus.attempt.execute")?;
        if self.state.handle.disposition().is_none() {
            let _ = self.state.handle.set_disposition(disposition_for(classify_final_plan(plan.as_ref())));
        }
        self.state.set_sealed_metrics_target(plan.as_ref());
        let owner = self.take_owner()?;
        let invocation = self.invocation_context(ctx);
        let task_context = self.task_context_with_attempt_ticket(plan.as_ref(), invocation.task_ctx());
        match execute_physical_stream(plan, task_context) {
            Ok(inner) => Ok(AttemptStream::new(inner, owner, Arc::clone(&self.state), self.completion.uncounted())),
            Err(error) => {
                commit_datafusion_error(&owner, &error, &self.state.cancellation);
                Err(error)
            },
        }
    }

    fn task_context_with_attempt_ticket(
        &self,
        plan: &dyn ExecutionPlan,
        task_context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Arc<datafusion::execution::context::TaskContext> {
        let Some(admission) = native_admission_plan(plan) else { return task_context };
        let ticket = {
            let mut state = self.state.ticket.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Arc::clone(state.get_or_insert_with(|| {
                let donor = admission.ticket();
                QueryAdmissionTicket::new_attempt_bound(
                    donor.source(),
                    donor.attempt_capability_proof_for_rebind(),
                    AttemptTicketBinding {
                        handle: self.state.handle.clone(),
                        admission_rejection: self.state.admission_rejection.clone(),
                        admission_binding: self.state.admission_binding.clone(),
                        cancellation: self.state.cancellation.clone(),
                    },
                    donor.metrics_capture(),
                )
            }))
        };
        task_context_with_query_admission_ticket(task_context, ticket)
    }
}

fn native_admission_plan(plan: &dyn ExecutionPlan) -> Option<&NativeAdmissionExec> {
    // `NativeAdmissionExec` is a transparent wrapper: `name`, `fmt`, and
    // `downcast_delegate` all forward to its input, so detection must cast
    // the trait object directly instead of `ExecutionPlan::downcast_ref`.
    // It is usually the physical root, but DDL plans can execute it below a
    // non-transparent DataFusion operator. The invocation task context must
    // carry the same attempt ticket before either shape starts execution.
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
        FinalPlanDisposition::Rejected => AttemptDisposition::Unplanned,
    }
}

/// Captures a returned DataFusion error by reference before reducing it to a
/// closed terminal outcome. Only native errors contribute a bounded message;
/// adapter-only errors use the deliberately text-free fallback.
fn commit_datafusion_error(owner: &AttemptOwner, error: &DataFusionError, cancellation: &AttemptCancellation) {
    if cancellation.is_cancelled() || cancellation.is_expired() {
        let cancellation_error = cancellation
            .check("datafusion_nexus.attempt.terminal_error")
            .expect_err("an effective cancellation must produce a terminal error");
        let _ = owner.commit_query_error(&cancellation_error);
        return;
    }
    match error.find_nexus_error() {
        Some(error) if error.code() == crate::error::ErrorCode::Cancelled => {
            let _ = owner.commit_terminal(RuntimeTerminalOutcome::Cancelled);
        },
        Some(error) => {
            if let Some(native) =
                error.source_error().and_then(|source| source.downcast_ref::<nexus_query_engine::error::QueryError>())
            {
                let _ = owner.commit_query_error(native);
            } else {
                let _ = owner.commit_failure_summary(AttemptFailureSummary::datafusion_fallback());
            }
        },
        None => {
            let _ = owner.commit_failure_summary(AttemptFailureSummary::datafusion_fallback());
        },
    }
}

/// Result stream that commits the attempt at EOF, error, cancellation, drop, or panic.
///
/// The stream owns the attempt's terminalization authority: success is
/// terminal only at EOF, every other path is exactly-once, and dropping the
/// stream initiates cooperative cancellation without blocking the runtime
/// worker it is dropped on.
pub struct AttemptStream {
    /// Kept only until the wrapper commits its terminal outcome. Releasing it
    /// afterwards drops a native admission round, so its cleanup commit sees
    /// the attempt's canonical terminal state before it returns capacity.
    inner: Option<SendableRecordBatchStream>,
    schema: arrow::datatypes::SchemaRef,
    owner: AttemptOwner,
    state: Arc<AttemptExecutionState>,
    completion: AttemptCompletion,
    physical_start_recorded: bool,
    terminal: bool,
}

impl AttemptStream {
    fn new(
        inner: SendableRecordBatchStream,
        owner: AttemptOwner,
        state: Arc<AttemptExecutionState>,
        completion: AttemptCompletion,
    ) -> Self {
        let schema = inner.schema();
        Self { inner: Some(inner), schema, owner, state, completion, physical_start_recorded: false, terminal: false }
    }

    /// An independent receiver for this attempt's terminal report.
    pub fn completion(&self) -> AttemptCompletion {
        self.completion.clone()
    }

    /// Returns the immutable graph claim-versus-peak snapshot once the owning
    /// attempt has sealed. This is the same projection exposed by
    /// [`NexusAttempt::sealed_graph_claim_calibration_snapshot`].
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
    /// stream. Native admission rounds release their query handle from that
    /// stream's drop path, and their `AttemptTerminalCommit` must therefore
    /// observe this terminal state rather than racing it at EOF or error.
    fn finish_terminal(&mut self, outcome: RuntimeTerminalOutcome) {
        self.terminal = true;
        self.commit(outcome);
        drop(self.inner.take());
    }

    /// Finalizes an effective cancellation through its typed native error so
    /// deadline expiry retains its bounded reason in the immutable report.
    fn finish_cancelled(&mut self) {
        self.terminal = true;
        let error = self
            .state
            .cancellation
            .check("datafusion_nexus.attempt.result_drain")
            .expect_err("an effective cancellation must produce a terminal error");
        let _ = self.owner.commit_query_error(&error);
        drop(self.inner.take());
    }

    fn cancelled_error(&self) -> DataFusionError {
        let operation = "datafusion_nexus.attempt.result_drain";
        match self.state.cancellation.check(operation) {
            Err(error) => {
                DataFusionError::from(Error::from_native(operation, "execution attempt was cancelled", error))
            },
            Ok(()) => DataFusionError::from(Error::new(
                crate::error::ErrorCode::Cancelled,
                operation,
                "execution attempt was cancelled",
            )),
        }
    }
}

impl RecordBatchStream for AttemptStream {
    fn schema(&self) -> arrow::datatypes::SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Stream for AttemptStream {
    type Item = DataFusionResult<arrow::record_batch::RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.terminal {
            return Poll::Ready(None);
        }
        if !this.state.handle.is_terminal()
            && this.state.cancellation.check("datafusion_nexus.attempt.result_drain").is_err()
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
        backend::{GpuDeviceMemoryLimit, InstalledTestGpuBackend, NexusGpuDeviceProfile, NexusGpuQueuePolicy},
        cugraph_sql::CugraphSqlConfig,
        error::ErrorCode,
        native_exec::NexusNativeExecutionConfig,
        planner::FinalPlanRequirement,
    };

    #[tokio::test]
    async fn feature_disabled_cugraph_rejection_stays_typed_through_an_attempt_stream() {
        let backend = InstalledTestGpuBackend::with_config_and_final_plan_requirements(
            NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1),
            NexusGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 }),
            NexusGpuQueuePolicy::default(),
            [FinalPlanRequirement::NoDataFusionCpu],
        );
        let state = backend
            .backend()
            .install_on_with_cugraph_sql(
                SessionStateBuilder::new_with_default_features()
                    .with_config(SessionConfig::new().with_target_partitions(1)),
                CugraphSqlConfig::default(),
            )
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
            .sql(&ctx, "SELECT * FROM cugraph_pagerank('edges')")
            .await
            .expect("feature-off execution must fail from the returned result stream");
        let error = stream
            .next()
            .await
            .expect("feature-off stream must emit one terminal rejection")
            .expect_err("feature-off cuGraph execution must reject");
        let typed = error.find_nexus_error().expect("attempt stream must retain the typed feature rejection");
        assert_eq!(typed.code(), ErrorCode::UnsupportedOperator);
        assert_eq!(typed.operation(), "datafusion_nexus.cugraph_sql.execute");
        assert_eq!(typed.fact_value("reason"), Some("required_feature_disabled"));
        assert_eq!(typed.fact_value("required_feature"), Some("cugraph"));
        assert_eq!(typed.fact_value("host_fallback"), Some("disabled"));
        drop(stream);
        let _report = attempt.completion().await;
    }
}
