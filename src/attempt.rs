//! Explicit execution-attempt adapter boundary.

use std::{
    any::Any,
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
use futures::{FutureExt, Stream};
pub use nexus_query_engine::{
    attempt::{AttemptBackendCountersSnapshot, AttemptCancellation, AttemptCompletion},
    observability::{AttemptId, AttemptQueryId, AttemptTraceContext, QueryAttemptReport, RuntimeObservationConfig},
};
use nexus_query_engine::{
    attempt::{AttemptHandle, AttemptOwner, BeginAttemptRequest},
    observability::{AttemptDisposition, RuntimeObservationOwner, RuntimeTerminalOutcome},
};

use crate::{
    backend::NexusGpuBackend,
    error::{DataFusionErrorExt, Error, Result},
    native_exec::{
        admission::{
            AttemptTicketBinding, NativeAdmissionExec, QueryAdmissionTicket, task_context_with_query_admission_ticket,
        },
        transparent_root_wrapper_input,
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
#[derive(Debug)]
pub struct AttemptExecutionState {
    pub handle: AttemptHandle,
    pub query_id: Option<AttemptQueryId>,
    pub observation_owner: RuntimeObservationOwner,
    pub cancellation: AttemptCancellation,
    pub(crate) ticket: Mutex<Option<Arc<crate::native_exec::admission::QueryAdmissionTicket>>>,
}

/// Config extension carrying attempt-local state.  It intentionally has no SQL
/// settings: attempts are installed only into invocation-local session states.
#[derive(Debug, Clone)]
pub struct NexusAttemptExtension {
    pub state: Arc<AttemptExecutionState>,
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
            query_id: state_query_id,
            observation_owner: begin.owner.observation_owner(),
            cancellation: begin.owner.cancellation(),
            ticket: Mutex::new(None),
        });
        // The attempt object is a factory, not a receiver: the begin
        // completion is dropped here so only caller-held receivers count
        // toward the abandoned-report decision.
        let completion = begin.completion.uncounted();
        Ok(NexusAttempt {
            owner: Mutex::new(Some(begin.owner)),
            state,
            completion,
            session: self.session_capability().clone(),
        })
    }

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
    pub fn cancellation(&self) -> AttemptCancellation {
        self.state.cancellation.clone()
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
        config.options_mut().extensions.insert(NexusAttemptExtension { state: Arc::clone(&self.state) });
        builder.build()
    }

    fn invocation_context(&self, ctx: &SessionContext) -> SessionContext {
        SessionContext::new_with_state(self.invocation_state(ctx.state()))
    }

    fn cancelled_error(&self, operation: &'static str) -> DataFusionError {
        let error = self.cancellation().check(operation).expect_err("cancellation was checked after it was requested");
        DataFusionError::from(Error::from_native(operation, "execution attempt was cancelled", error))
    }

    fn planning_failed(&self, error: &DataFusionError) {
        self.commit(terminal_outcome_for_error(error, &self.state.cancellation));
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
        let outcome = RuntimeTerminalOutcome::Failed {
            error_code: "backend_closing".to_string(),
            error_status: "permanent".to_string(),
        };
        self.commit(outcome);
        Err(DataFusionError::from(error))
    }

    /// Plans SQL in a fresh invocation-local session state.
    pub async fn plan_sql(&self, ctx: &SessionContext, sql: &str) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if self.cancellation().check("datafusion_nexus.attempt.plan_sql").is_err() {
            self.commit(RuntimeTerminalOutcome::Cancelled);
            return Err(self.cancelled_error("datafusion_nexus.attempt.plan_sql"));
        }
        self.fail_if_backend_closing("datafusion_nexus.attempt.plan_sql")?;
        let invocation = self.invocation_context(ctx);
        let handle = self.state.handle.clone();
        let _ = handle.record_planning_started();
        let planned =
            AssertUnwindSafe(async { invocation.sql(sql).await?.create_physical_plan().await }).catch_unwind().await;
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
            self.commit(RuntimeTerminalOutcome::Cancelled);
            return Err(self.cancelled_error("datafusion_nexus.attempt.plan_sql"));
        }
        let _ = handle.set_disposition(disposition_for(classify_final_plan(plan.as_ref())));
        let _ = handle.record_planning_finished();
        Ok(plan)
    }

    /// Plans and executes SQL in one invocation.
    pub async fn sql(&self, ctx: &SessionContext, sql: &str) -> DataFusionResult<AttemptStream> {
        let plan = self.plan_sql(ctx, sql).await?;
        self.execute_physical_plan(plan, ctx)
    }

    /// Plans and executes an existing `DataFrame` in one invocation. The
    /// frame's logical plan is replanned inside the invocation-local session
    /// state so native fragments bind this attempt's admission handle.
    pub async fn execute_dataframe(&self, df: DataFrame) -> DataFusionResult<AttemptStream> {
        if self.cancellation().check("datafusion_nexus.attempt.execute_dataframe").is_err() {
            self.commit(RuntimeTerminalOutcome::Cancelled);
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
            self.commit(RuntimeTerminalOutcome::Cancelled);
            return Err(self.cancelled_error("datafusion_nexus.attempt.execute_dataframe"));
        }
        let _ = handle.set_disposition(disposition_for(classify_final_plan(plan.as_ref())));
        let _ = handle.record_planning_finished();
        let owner = self.take_owner()?;
        let task_context = self.task_context_with_attempt_ticket(plan.as_ref(), invocation.task_ctx());
        match execute_physical_stream(plan, task_context) {
            Ok(inner) => Ok(AttemptStream::new(inner, owner, Arc::clone(&self.state), self.completion.uncounted())),
            Err(error) => {
                let _ = owner.commit_terminal(terminal_outcome_for_error(&error, &self.state.cancellation));
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
            self.commit(RuntimeTerminalOutcome::Cancelled);
            return Err(self.cancelled_error("datafusion_nexus.attempt.execute"));
        }
        self.fail_if_backend_closing("datafusion_nexus.attempt.execute")?;
        if self.state.handle.disposition().is_none() {
            let _ = self.state.handle.set_disposition(disposition_for(classify_final_plan(plan.as_ref())));
        }
        let owner = self.take_owner()?;
        let invocation = self.invocation_context(ctx);
        let task_context = self.task_context_with_attempt_ticket(plan.as_ref(), invocation.task_ctx());
        match execute_physical_stream(plan, task_context) {
            Ok(inner) => Ok(AttemptStream::new(inner, owner, Arc::clone(&self.state), self.completion.uncounted())),
            Err(error) => {
                let _ = owner.commit_terminal(terminal_outcome_for_error(&error, &self.state.cancellation));
                Err(error)
            },
        }
    }

    fn task_context_with_attempt_ticket(
        &self,
        plan: &dyn ExecutionPlan,
        task_context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Arc<datafusion::execution::context::TaskContext> {
        let Some(admission) = native_admission_root(plan) else { return task_context };
        let ticket = {
            let mut state = self.state.ticket.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Arc::clone(state.get_or_insert_with(|| {
                let donor = admission.ticket();
                QueryAdmissionTicket::new_attempt_bound(
                    donor.source(),
                    donor.requested_grant_min_bytes_for_rebind(),
                    donor.requested_grant_bytes_for_rebind(),
                    donor.placement_hint_for_rebind(),
                    #[cfg(feature = "cugraph")]
                    donor.warm_graph_requested_grant_bytes_for_rebind(),
                    AttemptTicketBinding {
                        handle: self.state.handle.clone(),
                        cancellation: self.state.cancellation.clone(),
                        observation_owner: self.state.observation_owner.clone(),
                    },
                    donor.metrics_capture(),
                )
            }))
        };
        task_context_with_query_admission_ticket(task_context, ticket)
    }
}

fn native_admission_root(plan: &dyn ExecutionPlan) -> Option<&NativeAdmissionExec> {
    // `NativeAdmissionExec` is a transparent wrapper: `name`, `fmt`, and
    // `downcast_delegate` all forward to its input, so detection must cast
    // the trait object directly instead of `ExecutionPlan::downcast_ref`.
    let mut current = plan;
    loop {
        if let Some(admission) = (current as &dyn Any).downcast_ref::<NativeAdmissionExec>() {
            return Some(admission);
        }
        current = transparent_root_wrapper_input(current)?.as_ref();
    }
}

fn disposition_for(disposition: FinalPlanDisposition) -> AttemptDisposition {
    match disposition {
        FinalPlanDisposition::DataFusion => AttemptDisposition::CpuFallback,
        FinalPlanDisposition::Native => AttemptDisposition::Native,
        FinalPlanDisposition::Mixed => AttemptDisposition::PartialNative,
        FinalPlanDisposition::Rejected => AttemptDisposition::Unplanned,
    }
}

/// Classifies a stream or planning error into the bounded terminal outcome.
fn terminal_outcome_for_error(error: &DataFusionError, cancellation: &AttemptCancellation) -> RuntimeTerminalOutcome {
    if cancellation.is_cancelled() || cancellation.is_expired() {
        return RuntimeTerminalOutcome::Cancelled;
    }
    match error.find_nexus_error() {
        Some(error) if error.code() == crate::error::ErrorCode::Cancelled => RuntimeTerminalOutcome::Cancelled,
        Some(error) => RuntimeTerminalOutcome::Failed {
            error_code: error.code().as_str().to_string(),
            error_status: error.status().as_str().to_string(),
        },
        None => RuntimeTerminalOutcome::Failed {
            error_code: "datafusion_error".to_string(),
            error_status: "permanent".to_string(),
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
    inner: SendableRecordBatchStream,
    owner: AttemptOwner,
    state: Arc<AttemptExecutionState>,
    completion: AttemptCompletion,
    physical_start_recorded: bool,
}

impl AttemptStream {
    fn new(
        inner: SendableRecordBatchStream,
        owner: AttemptOwner,
        state: Arc<AttemptExecutionState>,
        completion: AttemptCompletion,
    ) -> Self {
        Self { inner, owner, state, completion, physical_start_recorded: false }
    }

    /// An independent receiver for this attempt's terminal report.
    pub fn completion(&self) -> AttemptCompletion {
        self.completion.clone()
    }

    fn commit(&self, outcome: RuntimeTerminalOutcome) {
        let _ = self.owner.commit_terminal(outcome);
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
        self.inner.schema()
    }
}

impl Stream for AttemptStream {
    type Item = DataFusionResult<arrow::record_batch::RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if !this.state.handle.is_terminal()
            && this.state.cancellation.check("datafusion_nexus.attempt.result_drain").is_err()
        {
            this.commit(RuntimeTerminalOutcome::Cancelled);
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
        let poll = catch_unwind(AssertUnwindSafe(|| Pin::new(&mut this.inner).poll_next(cx)));
        match poll {
            Err(panic) => {
                this.commit(RuntimeTerminalOutcome::Panicked);
                resume_unwind(panic);
            },
            Ok(Poll::Ready(None)) => {
                this.commit(RuntimeTerminalOutcome::Succeeded);
                Poll::Ready(None)
            },
            Ok(Poll::Ready(Some(Err(error)))) => {
                let outcome = terminal_outcome_for_error(&error, &this.state.cancellation);
                this.commit(outcome);
                Poll::Ready(Some(Err(error)))
            },
            Ok(other) => other,
        }
    }
}

impl Drop for AttemptStream {
    fn drop(&mut self) {
        if !self.state.handle.is_terminal() {
            self.state.cancellation.cancel();
            self.commit(RuntimeTerminalOutcome::Dropped);
        }
    }
}

impl std::fmt::Debug for AttemptStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttemptStream").field("attempt_id", &self.owner.attempt_id()).finish_non_exhaustive()
    }
}
