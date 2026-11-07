//! Attempt admission tickets and the native admission root node.
//!
//! An invocation mints one `QueryAdmissionTicket`; every native fragment the
//! attempt executes admits through it and shares the one grant it obtains. The
//! ticket reaches fragments through the DataFusion task context: the admission
//! root installs an `ActiveQueryAdmissionTicket` extension when it executes,
//! and a fragment with no active execution falls back to the ticket its
//! invocation minted (CTAS runs its input before any root executes). Admission
//! itself happens when the first fragment polls its stream, on the host
//! runtime, and waits in the service queue until a grant is placed, the
//! attempt is cancelled or the queue deadline passes. The runtime releases the
//! grant once the attempt is terminal and no fragment is running; nothing in
//! this module releases it.

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    any::Any,
    fmt::{self, Formatter},
    mem,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::Duration,
};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, Statistics, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::PhysicalExpr,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream, StatisticsArgs,
        StatisticsContext, coalesce_partitions::CoalescePartitionsExec,
    },
};
use query_planning::{
    admission::AttemptCapabilityEvidence,
    error::{QueryError, QueryErrorCode, Result as QueryEngineResult},
    plan::QueryPlan,
};
use query_runtime::{
    attempt::{
        AttemptAdmissionBinding, AttemptAdmissionRejection, AttemptCancellation, AttemptExecutionRound, AttemptHandle,
        AttemptHostOccurrence,
    },
    observability::{AttemptAdmissionWaitFact, AttemptComposedScheduleProjection},
    runtime::{AdmittedQueryContext, ExecutionCancellation, GpuBackendSession, QueryAdmissionRequest, QueryHandle},
};
use tokio::sync::Notify;

use crate::{
    attempt::execute_in_internal_attempt,
    error::{Error, ErrorCode, Result},
    native_exec::{metrics::NativeExecutionMetricsCapture, occurrence::SharedHostOccurrenceResults},
    planner::native::Invocation,
};

#[cfg(test)]
static NEXT_TICKET_ID: AtomicU64 = AtomicU64::new(1);

/// Backend-owned service admission source copied into optimizer configs.
#[derive(Debug, Clone)]
pub(crate) struct QueryAdmissionSource {
    session: GpuBackendSession,
    /// Largest device total of the backend; part of the identity
    /// `same_backend` compares.
    max_device_total_memory_bytes: u64,
}

impl QueryAdmissionSource {
    pub(crate) fn new(session: GpuBackendSession, max_device_total_memory_bytes: u64) -> Self {
        Self { session, max_device_total_memory_bytes }
    }

    pub fn backend_matches(&self, session: &GpuBackendSession) -> bool {
        self.session.same_backend(session)
    }

    /// The backend session an execution without an attempt begins its
    /// internal attempt on.
    pub(crate) fn session(&self) -> &GpuBackendSession {
        &self.session
    }

    pub(crate) fn same_backend(&self, other: &Self) -> bool {
        self.max_device_total_memory_bytes == other.max_device_total_memory_bytes
            && self.session.same_backend(&other.session)
    }
}

/// Immutable facts assigned when query-service admission succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueryAdmissionFacts {
    pub(crate) admission_sequence: u64,
    pub(crate) admission_wait: Duration,
    pub(crate) admission_wait_fact: Option<AttemptAdmissionWaitFact>,
    pub(crate) device_ordinal: usize,
    pub(crate) query_grant_bytes: u64,
}

impl QueryAdmissionFacts {
    fn from_handle(handle: &TicketQueryHandle) -> Self {
        Self {
            admission_sequence: handle.admission_sequence(),
            admission_wait: handle.admission_wait(),
            admission_wait_fact: handle.admission_wait_fact(),
            device_ordinal: handle.device_ordinal(),
            query_grant_bytes: handle.device_budget_bytes(),
        }
    }
}

/// The runtime handle of the attempt's one admission, narrowed to what
/// fragments use.
#[derive(Debug)]
struct TicketQueryHandle(Arc<QueryHandle>);

impl TicketQueryHandle {
    fn admission_sequence(&self) -> u64 {
        self.0.admission_sequence()
    }

    fn admission_wait(&self) -> Duration {
        self.0.admission_wait()
    }

    fn admission_wait_fact(&self) -> Option<AttemptAdmissionWaitFact> {
        self.0.admission_wait_fact()
    }

    fn device_ordinal(&self) -> usize {
        self.0.device_ordinal()
    }

    fn device_budget_bytes(&self) -> u64 {
        self.0.device_budget_bytes()
    }

    fn join_fragment(&self, fragment: &AttemptCapabilityEvidence) -> QueryEngineResult<()> {
        self.0.join_fragment(fragment)
    }

    async fn acquire_context_with_cancellation_async(
        &self,
        cancellation: ExecutionCancellation,
    ) -> QueryEngineResult<AdmittedQueryContext> {
        self.0.acquire_context_with_cancellation_async(cancellation).await
    }
}

/// The one admission ticket of an attempt's invocation.
///
/// Every native fragment the attempt executes, including sequential
/// executions such as a CTAS input and its result, admits through this
/// ticket and shares the one grant it obtains. The grant belongs to the
/// attempt: the runtime releases it once the attempt's outcome is terminal
/// and no fragment is still running, so a fragment's stream ending releases
/// nothing.
#[derive(Debug)]
pub(crate) struct QueryAdmissionTicket {
    attempt: AttemptTicketBinding,
    #[cfg(test)]
    id: u64,
    source: QueryAdmissionSource,
    /// Engine-issued evidence for the native fragments this ticket may execute.
    /// `None` is intentionally not an empty explicit evidence: all-modeled
    /// services may admit it for resource-only use, while exact profiles
    /// reject it and execution cannot use it to authorize a native plan.
    attempt_capability_evidence: Option<AttemptCapabilityEvidence>,
    /// The evidence as fragments bind to the ticket: refined before the
    /// claim, then joined by fragments planned after it.
    rebound_capability_evidence: Mutex<Option<AttemptCapabilityEvidence>>,
    admission: Mutex<TicketAdmission>,
    notify: Arc<Notify>,
}

/// The attempt's one admission as fragments reach it.
#[derive(Debug, Default)]
struct TicketAdmission {
    /// What the attempt's one admission resolved to: the handle every
    /// fragment shares, or the error every fragment reports.
    outcome: Option<QueryEngineResult<Arc<TicketQueryHandle>>>,
    /// True while one fragment is claiming the admission. Other fragments wait on
    /// `QueryAdmissionTicket::notify`; none claims a second time.
    admitting: bool,
    /// Set by the first context; admission facts are recorded once per
    /// attempt.
    admission_facts_recorded: bool,
    /// Fragments bound while the claim was in flight; they join its grant
    /// when it resolves.
    pending_fragments: Vec<AttemptCapabilityEvidence>,
}

/// The ticket of one DataFusion execution of a native admission root, with
/// the occurrence round and the shared host occurrence results of that
/// execution. A later execution of the same plan starts with neither.
#[derive(Debug)]
struct ActiveQueryAdmissionTicket {
    ticket: Arc<QueryAdmissionTicket>,
    execution_round: AttemptExecutionRound,
    host_occurrence_results: Arc<SharedHostOccurrenceResults>,
}

pub(crate) fn query_admission_ticket_from_task_context(
    task_context: &TaskContext,
) -> Option<Arc<QueryAdmissionTicket>> {
    task_context
        .session_config()
        .get_extension::<ActiveQueryAdmissionTicket>()
        .map(|extension| Arc::clone(&extension.ticket))
}

/// Host occurrence evaluations shared by the readers of one DataFusion
/// execution, when `task_context` executes under an admission root.
pub(crate) fn shared_host_occurrence_results(task_context: &TaskContext) -> Option<Arc<SharedHostOccurrenceResults>> {
    task_context
        .session_config()
        .get_extension::<ActiveQueryAdmissionTicket>()
        .map(|extension| Arc::clone(&extension.host_occurrence_results))
}

/// The ticket that native work in `task_context` admits under: the active
/// execution's ticket, else the ticket its invocation minted.
///
/// DataFusion executes CTAS inside `execute_logical_plan`, before the attempt
/// can decorate a physical plan, so the inner plan's task context carries the
/// invocation but no active execution ticket.
pub(crate) fn invocation_admission_ticket(task_context: &TaskContext) -> Option<Arc<QueryAdmissionTicket>> {
    query_admission_ticket_from_task_context(task_context)
        .or_else(|| Invocation::of_task(task_context).and_then(Invocation::minted_admission_ticket).map(Arc::clone))
}

/// Native work that finds no attempt ticket runs outside the native admission
/// node every native plan is rooted in, so it has no attempt to admit under.
pub(crate) fn attempt_execution_required_error(operation: &'static str) -> Error {
    Error::new(ErrorCode::InvalidExecutionPlan, operation, "native execution requires an attempt invocation")
        .with_fact("reason", "attempt_execution_required")
}

/// Invocation-scoped attempt state attached to an admission ticket.
#[derive(Debug, Clone)]
pub(crate) struct AttemptTicketBinding {
    pub(crate) handle: AttemptHandle,
    pub(crate) admission_rejection: AttemptAdmissionRejection,
    pub(crate) admission_binding: AttemptAdmissionBinding,
    pub(crate) cancellation: AttemptCancellation,
    /// Exact cap supplied by the invocation owner, if any. It is copied into
    /// the attempt's query-service admission request.
    pub(crate) requested_query_cap_bytes: Option<u64>,
}

/// Starts one DataFusion execution under `ticket`: the returned task context
/// carries the ticket and a fresh occurrence round. A task context already
/// executing under a ticket keeps it.
pub(crate) fn task_context_with_query_admission_ticket(
    task_context: Arc<TaskContext>,
    ticket: Arc<QueryAdmissionTicket>,
) -> Arc<TaskContext> {
    if query_admission_ticket_from_task_context(&task_context).is_some() {
        return task_context;
    }

    let execution_round = AttemptExecutionRound::new(
        ticket.current_capability_evidence().and_then(|evidence| evidence.execution_manifest().cloned()),
    );
    let mut session_config = task_context.session_config().clone();
    session_config.set_extension(Arc::new(ActiveQueryAdmissionTicket {
        ticket,
        execution_round,
        host_occurrence_results: Arc::default(),
    }));
    Arc::new(TaskContext::new(
        task_context.task_id(),
        task_context.session_id(),
        session_config,
        task_context.scalar_functions().clone(),
        task_context.higher_order_functions().clone(),
        task_context.aggregate_functions().clone(),
        task_context.window_functions().clone(),
        task_context.runtime_env(),
    ))
}

/// Context acquired through a [`QueryAdmissionTicket`].
#[derive(Debug)]
pub(crate) struct TicketAdmissionContext {
    admission_context: AdmittedQueryContext,
    task_context: Arc<TaskContext>,
    admission_facts: QueryAdmissionFacts,
    record_admission_facts: bool,
}

impl TicketAdmissionContext {
    pub(crate) fn context(&self) -> &AdmittedQueryContext {
        &self.admission_context
    }

    pub(crate) fn task_context(&self) -> &Arc<TaskContext> {
        &self.task_context
    }

    pub(crate) fn admission_facts(&self) -> QueryAdmissionFacts {
        self.admission_facts
    }

    pub(crate) fn should_record_admission_facts(&self) -> bool {
        self.record_admission_facts
    }
}

/// Resolves an in-flight admission whose claiming fragment stopped waiting
/// for it. The attempt has no second claim. The fragments waiting on that
/// admission therefore receive its abandonment and do not claim one of their own.
struct TicketAdmissionGuard {
    ticket: Arc<QueryAdmissionTicket>,
    resolved: bool,
}

impl QueryAdmissionTicket {
    /// Creates the one ticket of an attempt's invocation.
    pub(crate) fn new(
        source: QueryAdmissionSource,
        attempt_capability_evidence: Option<AttemptCapabilityEvidence>,
        attempt: AttemptTicketBinding,
    ) -> Arc<Self> {
        Arc::new(Self {
            attempt,
            #[cfg(test)]
            id: NEXT_TICKET_ID.fetch_add(1, Ordering::Relaxed),
            source,
            attempt_capability_evidence,
            rebound_capability_evidence: Mutex::new(None),
            admission: Mutex::new(TicketAdmission::default()),
            notify: Arc::new(Notify::new()),
        })
    }

    pub(crate) fn source(&self) -> QueryAdmissionSource {
        self.source.clone()
    }

    pub(crate) fn attempt_capability_evidence_for_rebind(&self) -> Option<AttemptCapabilityEvidence> {
        self.current_capability_evidence()
    }

    pub(crate) fn has_execution_manifest(&self) -> bool {
        self.current_capability_evidence().as_ref().and_then(AttemptCapabilityEvidence::execution_manifest).is_some()
    }

    pub(crate) fn begin_host_occurrence(
        self: &Arc<Self>,
        task_context: &Arc<TaskContext>,
        occurrence_id: &str,
        producer_identity: &str,
        plan: &QueryPlan,
    ) -> Result<Option<AttemptHostOccurrence>> {
        if !self.has_execution_manifest() {
            return Ok(None);
        }
        let execution_round = task_context
            .session_config()
            .get_extension::<ActiveQueryAdmissionTicket>()
            .filter(|active| Arc::ptr_eq(&active.ticket, self))
            .map(|active| active.execution_round.clone())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.native_exec.host_occurrence",
                    "a GPU function occurrence began outside its invocation execution round",
                )
                .with_log_safe_fact("reason", "host_occurrence_execution_round_missing")
                .with_log_safe_fact("occurrence_id", occurrence_id)
                .with_log_safe_fact("producer_identity", producer_identity)
            })?;
        execution_round.begin_host_occurrence(occurrence_id, producer_identity, plan).map_err(|error| {
            Error::from_native(
                "algeon_datafusion.native_exec.host_occurrence",
                "the GPU function cannot begin in its admitted execution schedule",
                error,
            )
        })
    }

    pub(crate) fn validate_host_occurrence(&self, producer_identity: &str, plan: &QueryPlan) -> Result<Option<()>> {
        let Some(evidence) = self.current_capability_evidence() else {
            return Ok(None);
        };
        let Some(manifest) = evidence.execution_manifest() else {
            return Ok(None);
        };
        manifest.validate_host_producer(producer_identity, plan).map(Some).map_err(|error| {
            Error::from_native(
                "algeon_datafusion.native_exec.host_occurrence",
                "the executing GPU function does not match an admitted host producer",
                error,
            )
        })
    }

    fn current_capability_evidence(&self) -> Option<AttemptCapabilityEvidence> {
        self.rebound_capability_evidence
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .or_else(|| self.attempt_capability_evidence.clone())
    }

    /// Binds the evidence of fragments this ticket's attempt executes.
    ///
    /// Before the attempt claims its admission, fragments contribute deltas
    /// to the evidence the claim submits, so one preparation cannot replace
    /// another fragment's scope or the final invocation manifest. The claim
    /// fixes placement. A fragment it does not cover then joins the grant:
    /// the runtime checks that the granted device provides every capability
    /// the fragment requires and adds the fragment's plan identity. A
    /// fragment bound while the claim is in flight joins when it resolves.
    pub(crate) fn bind_fragment_evidence(
        &self,
        fragments: impl IntoIterator<Item = AttemptCapabilityEvidence>,
    ) -> QueryEngineResult<()> {
        let mut admission = self.lock_admission()?;
        let mut rebound = self.rebound_capability_evidence.lock().map_err(|_| {
            QueryError::new(
                QueryErrorCode::Internal,
                "algeon_datafusion.native_exec.admission_ticket",
                "query admission capability evidence lock was poisoned",
            )
            .with_fact("reason", "capability_evidence_rebind_lock_poisoned")
        })?;
        let current = rebound.as_ref().or(self.attempt_capability_evidence.as_ref()).cloned();
        if !admission.admitting && admission.outcome.is_none() {
            *rebound = Some(AttemptCapabilityEvidence::aggregate(current.into_iter().chain(fragments))?);
            return Ok(());
        }
        let mut evidence = current.unwrap_or_default();
        for fragment in fragments {
            if evidence.covers(&fragment) {
                continue;
            }
            match &admission.outcome {
                Some(Ok(handle)) => handle.join_fragment(&fragment)?,
                Some(Err(error)) => return Err(error.clone()),
                None => admission.pending_fragments.push(fragment.clone()),
            }
            evidence.join(&fragment)?;
        }
        *rebound = Some(evidence);
        Ok(())
    }

    pub(crate) fn record_attempt_gpu_execution_started(&self) {
        let _ = self.attempt.handle.record_gpu_execution_started();
    }

    /// Publishes the engine-owned composed schedule to the attempt report.
    pub(crate) fn record_attempt_composed_schedule_projection(
        &self,
        projection: AttemptComposedScheduleProjection,
    ) -> QueryEngineResult<()> {
        self.attempt.handle.record_composed_schedule_projection(projection).map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "algeon_datafusion.native_exec.admission_ticket",
                "failed to publish the admitted composed schedule to the attempt report",
            )
            .with_fact("reason", "composed_schedule_projection_record_failed")
            .with_fact("lifecycle_error", error.to_string())
        })
    }

    /// The cancellation of the attempt this ticket admits under.
    pub(crate) fn attempt_cancellation(&self) -> ExecutionCancellation {
        self.attempt.cancellation.execution_cancellation()
    }

    fn terminalize_attempt_admission_error(&self, error: &QueryError) {
        let _ = self.attempt.admission_rejection.reject(error);
    }

    #[cfg(test)]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    #[cfg(test)]
    pub(crate) fn attempt_capability_evidence(&self) -> Option<AttemptCapabilityEvidence> {
        self.current_capability_evidence()
    }

    /// Metrics projection onto the attempt's observation.
    pub(crate) fn metrics_capture(&self) -> NativeExecutionMetricsCapture {
        NativeExecutionMetricsCapture::coordinated(self.attempt.admission_binding.observation_handle())
    }

    /// Acquires an execution context under the attempt's one admission,
    /// admitting it the first time a fragment asks.
    pub(crate) async fn acquire_context(
        self: &Arc<Self>,
        task_context: Arc<TaskContext>,
    ) -> QueryEngineResult<TicketAdmissionContext> {
        loop {
            let notify = Arc::clone(&self.notify);
            // The waiter is registered (`enable`) while the admission lock is
            // held. `Notify::notify_waiters` wakes only already registered
            // waiters, so a resolution between releasing the lock and the
            // await would otherwise be missed.
            let notified = notify.notified_owned();
            tokio::pin!(notified);
            let action = {
                let mut admission = self.lock_admission()?;
                match admission.outcome.as_ref() {
                    Some(Ok(handle)) => TicketAcquireAction::Reuse(Arc::clone(handle)),
                    Some(Err(error)) => return Err(error.clone()),
                    None if admission.admitting => {
                        notified.as_mut().enable();
                        TicketAcquireAction::Wait
                    },
                    None => {
                        admission.admitting = true;
                        TicketAcquireAction::Admit
                    },
                }
            };
            let handle = match action {
                TicketAcquireAction::Reuse(handle) => handle,
                TicketAcquireAction::Admit => self.admit().await?,
                TicketAcquireAction::Wait => {
                    notified.await;
                    continue;
                },
            };
            return self.acquire_from_handle(handle, task_context).await;
        }
    }

    /// Requests the attempt's one admission and records its outcome for
    /// every fragment. Callers set `admitting` first.
    async fn admit(self: &Arc<Self>) -> QueryEngineResult<Arc<TicketQueryHandle>> {
        // Dropping this future (the claiming fragment's stream was dropped)
        // resolves the admission as abandoned for the waiters.
        let mut guard = TicketAdmissionGuard { ticket: Arc::clone(self), resolved: false };
        let request = QueryAdmissionRequest::new(self.attempt.admission_binding.clone())
            .with_execution_cancellation(self.attempt_cancellation());
        let request = match self.attempt.requested_query_cap_bytes {
            Some(query_cap_bytes) => request.with_query_cap_bytes(query_cap_bytes),
            None => request,
        };
        let request = match self.attempt_capability_evidence_for_rebind() {
            Some(evidence) => request.with_query_plan_capability_evidence(evidence),
            None => request,
        };
        let request = match self.attempt.cancellation.deadline() {
            Some(deadline) => request.with_queue_deadline(deadline),
            None => request,
        };
        let admitted = self
            .source
            .session
            .admit_async_with_request(request)
            .await
            .map(|handle| Arc::new(TicketQueryHandle(Arc::new(handle))));
        let mut admission = self.lock_admission()?;
        // A fragment bound while the claim was in flight joins its grant now.
        // One the granted device cannot run fails the attempt's admission:
        // the attempt has no second claim to place it elsewhere.
        let outcome = admitted.and_then(|handle| {
            for fragment in mem::take(&mut admission.pending_fragments) {
                handle.join_fragment(&fragment)?;
            }
            Ok(handle)
        });
        // The rejection is recorded on the attempt here, so the sealed report
        // carries it even when no fragment stream survives to return it.
        if let Err(error) = &outcome {
            self.terminalize_attempt_admission_error(error);
        }
        admission.admitting = false;
        admission.outcome = Some(outcome.clone());
        guard.resolved = true;
        self.notify.notify_waiters();
        outcome
    }

    async fn acquire_from_handle(
        &self,
        handle: Arc<TicketQueryHandle>,
        task_context: Arc<TaskContext>,
    ) -> QueryEngineResult<TicketAdmissionContext> {
        match handle.acquire_context_with_cancellation_async(self.attempt_cancellation()).await {
            Ok(admission_context) => {
                let admission_facts = QueryAdmissionFacts::from_handle(handle.as_ref());
                // Native plans record the physical-execution boundary at
                // admission; `AttemptStream` records it at first poll only for
                // CPU-only plans.
                let _ = self.attempt.handle.record_physical_execution_started();
                Ok(TicketAdmissionContext {
                    admission_context,
                    task_context,
                    admission_facts,
                    record_admission_facts: self.mark_admission_facts_recorded(),
                })
            },
            Err(error) => {
                self.terminalize_attempt_admission_error(&error);
                self.attempt.handle.record_native_failure(&error);
                Err(error)
            },
        }
    }

    /// Whether the caller is the first context of the attempt to record the
    /// admission facts.
    fn mark_admission_facts_recorded(&self) -> bool {
        let Ok(mut admission) = self.admission.lock() else {
            tracing::warn!("failed to mark poisoned query admission ticket facts state");
            return false;
        };
        !mem::replace(&mut admission.admission_facts_recorded, true)
    }

    fn abandon_admission(&self) {
        let Ok(mut admission) = self.admission.lock() else {
            tracing::warn!("failed to resolve poisoned query admission ticket state");
            return;
        };
        admission.admitting = false;
        admission.outcome = Some(Err(QueryError::new(
            QueryErrorCode::Cancelled,
            "algeon_datafusion.native_exec.admission_ticket",
            "the attempt's admission was abandoned before it resolved",
        )
        .with_fact("reason", "attempt_admission_abandoned")));
        self.notify.notify_waiters();
    }

    fn lock_admission(&self) -> QueryEngineResult<MutexGuard<'_, TicketAdmission>> {
        self.admission.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "algeon_datafusion.native_exec.admission_ticket",
                "query admission ticket state mutex was poisoned",
            )
        })
    }
}

enum TicketAcquireAction {
    Reuse(Arc<TicketQueryHandle>),
    Admit,
    Wait,
}

impl Drop for TicketAdmissionGuard {
    fn drop(&mut self) {
        if !self.resolved {
            self.ticket.abandon_admission();
        }
    }
}

/// Root node that runs one DataFusion execution of its plan as one attempt.
///
/// The node holds only planning artifacts: the backend that admits the plan
/// and the capability evidence of its native fragments. An execution inside
/// an attempt uses that attempt's ticket; an execution without one (a plain
/// `collect`) begins an internal attempt. Either way every native fragment
/// below shares one ticket, and the node yields one stream.
#[derive(Debug)]
pub(crate) struct NativeAdmissionExec {
    input: Arc<dyn ExecutionPlan>,
    source: QueryAdmissionSource,
    evidence: AttemptCapabilityEvidence,
}

impl NativeAdmissionExec {
    /// Roots `input` in an admission node; a partitioned input is coalesced
    /// first so one execution is one stream and one attempt.
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        source: QueryAdmissionSource,
        evidence: AttemptCapabilityEvidence,
    ) -> Self {
        let input = if input.properties().output_partitioning().partition_count() > 1 {
            Arc::new(CoalescePartitionsExec::new(input)) as Arc<dyn ExecutionPlan>
        } else {
            input
        };
        Self { input, source, evidence }
    }

    /// The plan this admission node roots.
    pub(crate) fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub(crate) fn source(&self) -> QueryAdmissionSource {
        self.source.clone()
    }

    pub(crate) fn evidence(&self) -> AttemptCapabilityEvidence {
        self.evidence.clone()
    }

    fn execute_with_ticket(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
        ticket: Arc<QueryAdmissionTicket>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        self.input.execute(partition, task_context_with_query_admission_ticket(context, ticket))
    }

    /// The attempt ticket of the invocation `context` carries, minted from
    /// this node's planning artifacts when the invocation has none yet.
    fn invocation_ticket(&self, context: &TaskContext) -> Option<Arc<QueryAdmissionTicket>> {
        invocation_admission_ticket(context).or_else(|| {
            Invocation::of_task(context)?
                .attempt_admission_ticket(self.source(), Some(self.evidence()))
                .map(|(ticket, _)| ticket)
        })
    }
}

pub(crate) fn wrap_plan_with_native_admission(
    plan: Arc<dyn ExecutionPlan>,
    source: QueryAdmissionSource,
    evidence: AttemptCapabilityEvidence,
) -> Arc<dyn ExecutionPlan> {
    if plan_is_query_admission_wrapped(plan.as_ref()) {
        return plan;
    }
    Arc::new(NativeAdmissionExec::new(plan, source, evidence))
}

pub(crate) fn plan_is_query_admission_wrapped(plan: &dyn ExecutionPlan) -> bool {
    (plan as &dyn Any).is::<NativeAdmissionExec>()
}

impl DisplayAs for NativeAdmissionExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "NativeAdmissionExec")
    }
}

impl ExecutionPlan for NativeAdmissionExec {
    fn name(&self) -> &str {
        "NativeAdmissionExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // The node only scopes admission around its input's partitions; a
        // repartition between the two would run outside that scope.
        vec![false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [input] = children.try_into().map_err(|children: Vec<Arc<dyn ExecutionPlan>>| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidChildCount,
                    "algeon_datafusion.native_exec.admission_with_new_children",
                    "native admission expects exactly one child",
                )
                .with_fact("actual_child_count", children.len()),
            )
        })?;
        Ok(Arc::new(Self::new(input, self.source(), self.evidence())))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        // No GPU work and no admission happen here. The ticket is threaded
        // into the task context, and the input's stream admits on its first
        // poll.
        if let Some(ticket) = self.invocation_ticket(&context) {
            return self.execute_with_ticket(partition, context, ticket);
        }
        execute_in_internal_attempt(self.source.session(), self, context, |context| {
            let ticket = self.invocation_ticket(&context).ok_or_else(|| {
                DataFusionError::from(Error::new(
                    ErrorCode::Internal,
                    "algeon_datafusion.native_exec.internal_attempt",
                    "an internal attempt's invocation must mint the admission ticket",
                ))
            })?;
            self.execute_with_ticket(partition, context, ticket)
        })
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        StatisticsContext::new().compute(self.input.as_ref(), &StatisticsArgs::new().with_partition(partition))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{Arc, Mutex},
        task::Poll,
        time::{Duration, Instant},
    };

    use arrow::{
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    };
    use cudf::testing;
    use datafusion::{
        common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
        execution::context::TaskContext,
        physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr},
        physical_plan::{
            DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
            execution_plan::{Boundedness, EmissionType},
            stream::RecordBatchStreamAdapter,
        },
    };
    use futures::{
        FutureExt, StreamExt,
        future::join,
        stream::{empty, once, poll_fn},
    };
    use query_planning::{
        admission::{
            AttemptCapabilityEvidence, AttemptExecutionManifest, AttemptExecutionOccurrence, ExecutionCapability,
        },
        error::QueryErrorCode,
        plan::{QueryPlan, QueryPlanBuilder},
        source::{GpuResidentSource, SourceNode},
    };
    use query_runtime::{
        observability::{QueryAttemptReport, RuntimeTerminalOutcome},
        runtime::{DeviceCapabilities, GpuDeviceMemoryLimit},
    };
    use tokio::{runtime::Builder as TestRuntimeBuilder, time::timeout};

    use super::*;
    use crate::{
        attempt::{AttemptOptions, attempt_ticket_for_test, attempt_ticket_with_options_for_test},
        backend::{AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, InstalledTestGpuBackend},
        error::{DataFusionErrorExt, Error, ErrorCode},
        native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
    };

    #[derive(Clone, Copy, Debug)]
    enum LifecyclePlanBehavior {
        Success,
        ExecuteError,
        StreamError,
        CancelledStream,
        PanickingStream,
        PanickingExecute,
    }

    #[derive(Debug)]
    struct LifecycleExec {
        schema: SchemaRef,
        properties: Arc<PlanProperties>,
        behavior: LifecyclePlanBehavior,
        observed_ticket: Arc<Mutex<Option<u64>>>,
    }

    impl LifecycleExec {
        fn new(behavior: LifecyclePlanBehavior, observed_ticket: Arc<Mutex<Option<u64>>>) -> Self {
            let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
            let properties = Arc::new(PlanProperties::new(
                EquivalenceProperties::new(Arc::clone(&schema)),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            ));
            Self { schema, properties, behavior, observed_ticket }
        }

        fn typed_error(behavior: LifecyclePlanBehavior) -> DataFusionError {
            let (code, operation, message) = match behavior {
                LifecyclePlanBehavior::ExecuteError => (
                    ErrorCode::Internal,
                    "algeon_datafusion.test.composed_producer",
                    "synthetic composed producer failed",
                ),
                LifecyclePlanBehavior::StreamError => (
                    ErrorCode::Internal,
                    "algeon_datafusion.test.composed_consumer",
                    "synthetic composed consumer failed",
                ),
                LifecyclePlanBehavior::CancelledStream => (
                    ErrorCode::Cancelled,
                    "algeon_datafusion.test.composed_consumer_cancelled",
                    "synthetic composed consumer was cancelled",
                ),
                LifecyclePlanBehavior::Success
                | LifecyclePlanBehavior::PanickingStream
                | LifecyclePlanBehavior::PanickingExecute => {
                    unreachable!("this behavior does not produce a typed error")
                },
            };
            DataFusionError::from(Error::new(code, operation, message))
        }
    }

    impl DisplayAs for LifecycleExec {
        fn fmt_as(&self, _format: DisplayFormatType, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("LifecycleExec")
        }
    }

    impl ExecutionPlan for LifecycleExec {
        fn name(&self) -> &str {
            "LifecycleExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.properties
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            if children.is_empty() {
                Ok(self)
            } else {
                Err(DataFusionError::Execution("lifecycle test plan has no children".to_string()))
            }
        }

        fn execute(
            &self,
            _partition: usize,
            task_context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            if let Some(ticket) = query_admission_ticket_from_task_context(&task_context) {
                *self.observed_ticket.lock().expect("test ticket observation mutex") = Some(ticket.id());
            }
            match self.behavior {
                LifecyclePlanBehavior::Success => {
                    let stream = empty::<DataFusionResult<RecordBatch>>();
                    Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
                },
                LifecyclePlanBehavior::ExecuteError => Err(Self::typed_error(self.behavior)),
                LifecyclePlanBehavior::PanickingExecute => panic!("injected input execute panic"),
                LifecyclePlanBehavior::PanickingStream => {
                    let stream = poll_fn(|_cx| -> Poll<Option<DataFusionResult<RecordBatch>>> {
                        panic!("injected inner stream panic")
                    });
                    Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
                },
                LifecyclePlanBehavior::StreamError | LifecyclePlanBehavior::CancelledStream => {
                    let error = Self::typed_error(self.behavior);
                    let stream = once(async move { Err::<RecordBatch, DataFusionError>(error) });
                    Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
                },
            }
        }
    }

    fn lifecycle_exec(
        behavior: LifecyclePlanBehavior,
        observed_ticket: Arc<Mutex<Option<u64>>>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(LifecycleExec::new(behavior, observed_ticket))
    }

    /// Roots `input` in the admission node the final optimizer pass places.
    fn admission_exec(backend: &InstalledTestGpuBackend, input: Arc<dyn ExecutionPlan>) -> NativeAdmissionExec {
        NativeAdmissionExec::new(input, admission_source(backend), AttemptCapabilityEvidence::empty())
    }

    /// The report of the one internal attempt that executions without an
    /// attempt of their own began on `backend`.
    fn sole_internal_attempt_report(backend: &InstalledTestGpuBackend) -> Arc<QueryAttemptReport> {
        let reports = backend.backend().take_abandoned_attempt_reports(8);
        assert_eq!(reports.len(), 1, "one execution must publish exactly one attempt report");
        Arc::clone(&reports[0])
    }

    fn admission_source(backend: &InstalledTestGpuBackend) -> QueryAdmissionSource {
        backend
            .optimizer_config()
            .native_execution_config()
            .query_admission_source()
            .expect("installed backend must provide admission authority")
    }

    fn explicit_sink_capability_evidence() -> AttemptCapabilityEvidence {
        use arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        let plan = builder.finish_with_host_sink(source).unwrap();
        AttemptCapabilityEvidence::for_query_plan(&plan)
    }

    fn resident_host_plan(input_name: &str) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
        builder.finish_with_host_sink(source).unwrap()
    }

    fn host_occurrence_dependencies_complete_once_and_fail_closed_on_drop() {
        let backend = InstalledTestGpuBackend::new();
        let first_plan = resident_host_plan("first");
        let second_plan = resident_host_plan("second");
        let manifest = AttemptExecutionManifest::new(vec![
            AttemptExecutionOccurrence::host_arrow("gpu-first", &first_plan, vec![], false),
            AttemptExecutionOccurrence::host_arrow("gpu-second", &second_plan, vec!["gpu-first".to_string()], false),
        ])
        .unwrap();
        let evidence = AttemptCapabilityEvidence::empty().with_execution_manifest(manifest).unwrap();
        let (_attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), Some(evidence));
        let task_context =
            task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&ticket));

        let error = ticket
            .begin_host_occurrence(&task_context, "gpu-second", "gpu-second", &second_plan)
            .expect_err("a dependent occurrence cannot begin first");
        assert_eq!(error.fact_value("reason"), Some("host_occurrence_predecessor_incomplete"));

        drop(ticket.begin_host_occurrence(&task_context, "gpu-first", "gpu-first", &first_plan).unwrap().unwrap());
        assert!(
            ticket.begin_host_occurrence(&task_context, "gpu-second", "gpu-second", &second_plan).is_err(),
            "dropping an unfinished occurrence must not satisfy its dependents"
        );

        ticket.begin_host_occurrence(&task_context, "gpu-first", "gpu-first", &first_plan).unwrap().unwrap().complete();
        ticket
            .begin_host_occurrence(&task_context, "gpu-second", "gpu-second", &second_plan)
            .unwrap()
            .unwrap()
            .complete();
        let duplicate = ticket
            .begin_host_occurrence(&task_context, "gpu-second", "gpu-second", &second_plan)
            .expect_err("one final-plan occurrence cannot execute twice");
        assert_eq!(duplicate.fact_value("reason"), Some("host_occurrence_duplicate_execution"));

        let next_task_context =
            task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&ticket));
        ticket
            .begin_host_occurrence(&next_task_context, "gpu-first", "gpu-first", &first_plan)
            .expect("a new invocation round has independent occurrence state")
            .unwrap()
            .complete();
    }

    async fn fragments_bound_after_the_claim_join_its_grant() {
        let backend = InstalledTestGpuBackend::new();
        let first_plan = resident_host_plan("first-fragment");
        let second_plan = resident_host_plan("second-fragment");
        let (_attempt, ticket) =
            attempt_ticket_for_test(admission_source(&backend), Some(AttemptCapabilityEvidence::empty()));

        // Before the claim, each fragment adds its delta to the claim's evidence.
        ticket.bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&first_plan)]).unwrap();
        ticket.bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&second_plan)]).unwrap();
        let evidence = ticket.attempt_capability_evidence().unwrap();
        assert!(evidence.execution_scopes().iter().any(|scope| scope.matches(&first_plan)));
        assert!(evidence.execution_scopes().iter().any(|scope| scope.matches(&second_plan)));

        let context =
            ticket.acquire_context(Arc::new(TaskContext::default())).await.expect("the combined evidence admits");
        let admitted = backend.backend().diagnostics().unwrap().admission;
        ticket
            .bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&first_plan)])
            .expect("evidence the claim covers binds nothing new");
        let third_plan = resident_host_plan("third-fragment");
        ticket
            .bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&third_plan)])
            .expect("a fragment planned after the claim joins its grant");
        assert_eq!(
            backend.backend().diagnostics().unwrap().admission,
            admitted,
            "joining the grant changes neither the queue nor the ledger"
        );
        let execution = context.context().context();
        execution.validate_admitted_plan_capabilities(&third_plan).expect("the joined fragment may execute");
        let never_bound = resident_host_plan("never-bound-fragment");
        let error = execution
            .validate_admitted_plan_capabilities(&never_bound)
            .expect_err("a plan no fragment bound stays outside the admitted identities");
        assert_eq!(error.fact_value("reason"), Some("admitted_capability_scope_mismatch"));

        // The grant belongs to the attempt. Once the context that a fragment
        // acquired ends, a later fragment still joins the same grant.
        drop(context);
        let fourth_plan = resident_host_plan("fourth-fragment");
        ticket
            .bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&fourth_plan)])
            .expect("a finished fragment does not end the attempt's grant");
        let context =
            ticket.acquire_context(Arc::new(TaskContext::default())).await.expect("the attempt's grant is reused");
        context
            .context()
            .context()
            .validate_admitted_plan_capabilities(&fourth_plan)
            .expect("the joined fragment may execute");
        assert_eq!(backend.backend().diagnostics().unwrap().admission, admitted);
    }

    async fn a_failed_admission_is_the_outcome_every_later_fragment_gets() {
        let backend = InstalledTestGpuBackend::new();
        let (attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), None);
        attempt.cancellation().cancel();

        let error = ticket
            .acquire_context(Arc::new(TaskContext::default()))
            .await
            .expect_err("cancelled admission must fail before publishing a handle");
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(backend.backend().diagnostics().unwrap().admission.active_queries, 0);

        {
            let admission = ticket.admission.lock().expect("ticket admission mutex");
            assert!(!admission.admitting, "a failed admission must not strand its in-flight marker");
            assert!(matches!(admission.outcome, Some(Err(_))), "a failed admission records its error, not a handle");
        }

        let retry = timeout(Duration::from_secs(5), ticket.acquire_context(Arc::new(TaskContext::default())))
            .await
            .expect("a later fragment must not wait on the resolved admission")
            .expect_err("an attempt admits at most once");
        assert_eq!(
            (retry.code(), retry.operation(), retry.message(), retry.facts()),
            (error.code(), error.operation(), error.message(), error.facts()),
            "a later fragment gets the admission's own error, not a second claim's"
        );
        assert_eq!(backend.backend().diagnostics().unwrap().admission.active_queries, 0);
    }

    /// Fragments waiting while another fragment claims the attempt's admission
    /// get that admission's outcome; none of them claims again.
    async fn concurrent_fragments_share_the_one_admission_outcome() {
        let backend = InstalledTestGpuBackend::with_config(
            AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
            ),
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
                .with_max_active_attempts(1),
            AlgeonGpuQueuePolicy::default(),
        );
        let (holder_attempt, holder) = attempt_ticket_for_test(admission_source(&backend), None);
        let held_context = holder
            .acquire_context(Arc::new(TaskContext::default()))
            .await
            .expect("the holder must consume the device-local admission slot");

        // The claim queues behind the holder until the waiting attempt's
        // deadline rejects it, while a second fragment waits on that claim.
        let (_waiting_attempt, waiting) = attempt_ticket_with_options_for_test(
            admission_source(&backend),
            None,
            AttemptOptions { deadline: Some(Instant::now() + Duration::from_millis(200)), ..AttemptOptions::default() },
        );
        let (claiming, waiting_fragment) = timeout(
            Duration::from_secs(5),
            join(
                waiting.acquire_context(Arc::new(TaskContext::default())),
                waiting.acquire_context(Arc::new(TaskContext::default())),
            ),
        )
        .await
        .expect("both fragments resolve at the claim's deadline");
        let claiming = claiming.expect_err("the claim is rejected at its deadline");
        let waiting_fragment = waiting_fragment.expect_err("the waiting fragment gets the claim's outcome");
        assert_eq!(claiming.fact_value("reason"), Some("deadline_expired"));
        assert_eq!(
            (
                waiting_fragment.code(),
                waiting_fragment.operation(),
                waiting_fragment.message(),
                waiting_fragment.facts()
            ),
            (claiming.code(), claiming.operation(), claiming.message(), claiming.facts()),
            "the waiting fragment gets the same typed error instead of claiming again"
        );
        let snapshot = backend.backend().diagnostics().unwrap().admission;
        assert_eq!(snapshot.active_queries, 1, "only the holder owns a grant");
        assert_eq!(snapshot.queued_queries, 0);
        assert_eq!(snapshot.queue_high_watermark, 1, "the attempt queued one admission request");

        drop(held_context);
        drop(holder_attempt);
    }

    async fn pending_context_cancellation_keeps_the_shared_grant_until_the_attempt_ends() {
        let backend = InstalledTestGpuBackend::with_config(
            AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
            ),
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
                .with_max_active_attempts(1),
            AlgeonGpuQueuePolicy::default(),
        );
        let (attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), None);
        let task_context = Arc::new(TaskContext::default());
        let first_context =
            ticket.acquire_context(Arc::clone(&task_context)).await.expect("the first context must be admitted");

        let facts = first_context.admission_facts();
        assert!(facts.admission_sequence > 0 && facts.query_grant_bytes > 0);

        attempt.cancellation().cancel();
        let error = ticket
            .acquire_context(Arc::clone(&task_context))
            .await
            .expect_err("a cancelled secondary context must not consume another context lease");
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(backend.backend().diagnostics().unwrap().admission.active_queries, 1);

        drop(first_context);
        assert_eq!(
            backend.backend().diagnostics().unwrap().admission.active_queries,
            1,
            "a cancelled attempt keeps its grant until its outcome is committed"
        );
        drop(attempt);
        let (_replacement_attempt, replacement_ticket) = attempt_ticket_for_test(admission_source(&backend), None);
        let replacement =
            timeout(Duration::from_secs(5), replacement_ticket.acquire_context(Arc::new(TaskContext::default())))
                .await
                .expect("the grant must be returned once the attempt ends")
                .expect("released capacity must admit a replacement attempt");
        assert_eq!(backend.backend().diagnostics().unwrap().admission.active_queries, 1);
        drop(replacement);
    }

    async fn admission_deadline_cancels_a_waiter_without_leaking_admission_state_or_capacity() {
        let backend = InstalledTestGpuBackend::with_config(
            AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
            ),
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
                .with_max_active_attempts(1),
            AlgeonGpuQueuePolicy::default().with_max_queue_wait(Duration::from_secs(1)),
        );
        let (holder_attempt, holder) = attempt_ticket_for_test(admission_source(&backend), None);
        let held_context = holder
            .acquire_context(Arc::new(TaskContext::default()))
            .await
            .expect("the holder must consume the device-local admission slot");
        assert_eq!(backend.backend().diagnostics().unwrap().admission.active_queries, 1);

        let deadline = Instant::now() + Duration::from_millis(100);
        let (_waiter_attempt, waiter) = attempt_ticket_with_options_for_test(
            admission_source(&backend),
            None,
            AttemptOptions { deadline: Some(deadline), ..AttemptOptions::default() },
        );
        let error = timeout(Duration::from_secs(5), waiter.acquire_context(Arc::new(TaskContext::default())))
            .await
            .expect("a queued waiter must terminate at its deadline")
            .expect_err("deadline expiry must reject the waiter");
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
        let snapshot = backend.backend().diagnostics().unwrap().admission;
        assert_eq!(snapshot.active_queries, 1, "the holder still owns the only grant");
        assert_eq!(snapshot.queued_queries, 0, "a cancelled waiter must be removed from the queue");
        {
            let admission = waiter.admission.lock().expect("waiter ticket admission mutex");
            assert!(!admission.admitting, "deadline cleanup must clear the in-flight marker");
            assert!(matches!(admission.outcome, Some(Err(_))));
        }

        drop(held_context);
        drop(holder_attempt);
        let (_replacement_attempt, replacement_ticket) = attempt_ticket_for_test(admission_source(&backend), None);
        let replacement =
            timeout(Duration::from_secs(5), replacement_ticket.acquire_context(Arc::new(TaskContext::default())))
                .await
                .expect("released capacity must be reusable after waiter cancellation")
                .expect("a new attempt admits after the waiter's deadline cleanup");
        drop(replacement);
    }

    async fn planner_capability_evidence_rejects_at_the_native_ticket_boundary_before_admission() {
        let mut unavailable = explicit_sink_capability_evidence();
        unavailable.required_capabilities = vec![ExecutionCapability::new("unavailable")];
        let backend = InstalledTestGpuBackend::with_config(
            AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
            ),
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
                .with_device_capabilities(DeviceCapabilities::only(Vec::new())),
            AlgeonGpuQueuePolicy::default(),
        );
        let (_attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), Some(unavailable.clone()));

        assert_eq!(ticket.attempt_capability_evidence(), Some(unavailable));
        let error = ticket
            .acquire_context(Arc::new(TaskContext::default()))
            .await
            .expect_err("an aggregated unsupported capability must reject before admission");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
        assert_eq!(error.fact_value("reason"), Some("required_device_capability_unavailable"));
        assert_eq!(error.fact_value("required_capabilities"), Some("unavailable"));
        assert_eq!(error.fact_value("configured_device_count"), Some("1"));
        assert_eq!(error.fact_value("compatible_device_count"), Some("0"));
        assert_eq!(
            backend.backend().diagnostics().unwrap().admission.active_queries,
            0,
            "rejection must precede grant mutation"
        );
        assert_eq!(
            backend.backend().diagnostics().unwrap().admission.queued_queries,
            0,
            "rejection must not become a capacity waiter"
        );
    }

    async fn panic_in_the_inner_stream_poll_reports_panicked_once() {
        let backend = InstalledTestGpuBackend::new();
        let exec = admission_exec(
            &backend,
            lifecycle_exec(LifecyclePlanBehavior::PanickingStream, Arc::new(Mutex::new(None))),
        );
        let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("synthetic stream starts");

        let unwind = AssertUnwindSafe(stream.next()).catch_unwind().await;

        assert!(unwind.is_err(), "the inner panic must keep unwinding");
        drop(stream);
        assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Panicked);
    }

    fn panic_in_the_input_execute_reports_panicked_once() {
        let backend = InstalledTestGpuBackend::new();
        let exec = admission_exec(
            &backend,
            lifecycle_exec(LifecyclePlanBehavior::PanickingExecute, Arc::new(Mutex::new(None))),
        );

        let unwind = catch_unwind(AssertUnwindSafe(|| exec.execute(0, Arc::new(TaskContext::default()))));

        assert!(unwind.is_err(), "the input panic must keep unwinding");
        drop(exec);
        assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Panicked);
    }

    #[test]
    fn tests_tg_admticket_cases_1() {
        testing::run_named_cases(&[
            ("one_attempt_invocation_mints_one_admission_ticket", || {
                TestRuntimeBuilder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(one_attempt_invocation_mints_one_admission_ticket())
            }),
            ("concurrent_fragments_share_the_one_admission_outcome", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(concurrent_fragments_share_the_one_admission_outcome())
            }),
            ("fragments_bound_after_the_claim_join_its_grant", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(fragments_bound_after_the_claim_join_its_grant())
            }),
            (
                "host_occurrence_dependencies_complete_once_and_fail_closed_on_drop",
                host_occurrence_dependencies_complete_once_and_fail_closed_on_drop,
            ),
            ("pending_context_cancellation_keeps_the_shared_grant_until_the_attempt_ends", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(pending_context_cancellation_keeps_the_shared_grant_until_the_attempt_ends())
            }),
            ("planner_capability_evidence_rejects_at_the_native_ticket_boundary_before_admission", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(planner_capability_evidence_rejects_at_the_native_ticket_boundary_before_admission())
            }),
            ("a_failed_admission_is_the_outcome_every_later_fragment_gets", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(a_failed_admission_is_the_outcome_every_later_fragment_gets())
            }),
            ("admission_deadline_cancels_a_waiter_without_leaking_admission_state_or_capacity", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(admission_deadline_cancels_a_waiter_without_leaking_admission_state_or_capacity())
            }),
        ]);
    }
    /// An attempt mints its one admission ticket while planning; executing the
    /// plan admits under that ticket without minting another.
    async fn one_attempt_invocation_mints_one_admission_ticket() {
        use arrow::array::{Int64Array, StringArray};
        use datafusion::{
            execution::SessionStateBuilder,
            prelude::{ParquetReadOptions, SessionContext},
        };
        use parquet::arrow::ArrowWriter;

        let backend = InstalledTestGpuBackend::new();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(vec!["a", "b", "a"])), Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let ctx = SessionContext::new_with_state(
            backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build(),
        );
        ctx.register_parquet("rows", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
        let next_ticket_id = || attempt_ticket_for_test(admission_source(&backend), None).1.id();

        let before = next_ticket_id();
        let attempt = backend.backend().begin_attempt(AttemptOptions::default()).unwrap();
        let mut stream = attempt.sql(&ctx, "SELECT k, SUM(v) FROM rows GROUP BY k").await.unwrap();
        while let Some(batch) = stream.next().await {
            batch.unwrap();
        }
        let report = attempt.completion().await;
        let after = next_ticket_id();

        assert!(report.admission.is_some(), "the aggregate executes natively under the attempt");
        assert_eq!(after - before - 1, 1, "the attempt minted exactly one admission ticket");
    }

    async fn internal_attempt_reports_success_once_at_consumer_eof() {
        let backend = InstalledTestGpuBackend::new();
        let observed_ticket = Arc::new(Mutex::new(None));
        let exec =
            admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::Success, Arc::clone(&observed_ticket)));

        let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("synthetic consumer executes");
        assert!(stream.next().await.is_none(), "consumer reaches host EOF");
        assert!(stream.next().await.is_none(), "polling an already-complete consumer remains EOF");
        drop(stream);

        assert!(
            observed_ticket.lock().expect("test ticket observation mutex").is_some(),
            "the internal attempt's ticket owns the execution round"
        );
        assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Succeeded);
    }

    async fn ambient_attempt_ticket_owns_the_round_without_an_internal_attempt() {
        let backend = InstalledTestGpuBackend::new();
        let (_attempt, active_ticket) = attempt_ticket_for_test(admission_source(&backend), None);
        let observed_ticket = Arc::new(Mutex::new(None));
        let exec =
            admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::Success, Arc::clone(&observed_ticket)));
        let task_context =
            task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));

        let mut stream = exec.execute(0, task_context).expect("ambient ticket stream starts");
        assert!(stream.next().await.is_none());
        drop(stream);

        assert_eq!(
            *observed_ticket.lock().expect("test ticket observation mutex"),
            Some(active_ticket.id()),
            "the active invocation ticket owns the execution round"
        );
        assert_eq!(
            backend.backend().attempt_counters_snapshot().attempts_begun,
            1,
            "an execution inside an attempt begins no internal attempt"
        );
    }

    fn internal_attempt_input_execute_error_reports_one_failure() {
        let backend = InstalledTestGpuBackend::new();
        let exec =
            admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::ExecuteError, Arc::new(Mutex::new(None))));

        let error = match exec.execute(0, Arc::new(TaskContext::default())) {
            Ok(_) => panic!("input error must reach the admission node"),
            Err(error) => error,
        };
        assert!(error.find_algeon_error().is_some(), "input failure remains typed at the DataFusion boundary");
        assert!(matches!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Failed { .. }));
    }

    async fn internal_attempt_stream_error_reports_one_failure() {
        let backend = InstalledTestGpuBackend::new();
        let exec =
            admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::StreamError, Arc::new(Mutex::new(None))));

        let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("consumer stream starts");
        let error = stream.next().await.expect("consumer emits one error").expect_err("consumer error is preserved");
        assert!(error.find_algeon_error().is_some());
        drop(stream);

        assert!(matches!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Failed { .. }));
    }

    #[test]
    fn tests_tg_admlife_cases_1() {
        testing::run_named_cases(&[
            ("internal_attempt_stream_cancellation_reports_cancelled", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(internal_attempt_stream_cancellation_reports_cancelled())
            }),
            ("panic_in_the_input_execute_reports_panicked_once", panic_in_the_input_execute_reports_panicked_once),
            ("ambient_attempt_ticket_owns_the_round_without_an_internal_attempt", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(ambient_attempt_ticket_owns_the_round_without_an_internal_attempt())
            }),
            ("internal_attempt_stream_error_reports_one_failure", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(internal_attempt_stream_error_reports_one_failure())
            }),
            (
                "dropping_task_context_before_stream_does_not_end_the_internal_attempt",
                dropping_task_context_before_stream_does_not_end_the_internal_attempt,
            ),
            ("panic_in_the_inner_stream_poll_reports_panicked_once", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(panic_in_the_inner_stream_poll_reports_panicked_once())
            }),
            ("internal_attempt_reports_success_once_at_consumer_eof", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(internal_attempt_reports_success_once_at_consumer_eof())
            }),
            (
                "dropping_stream_without_polling_reports_dropped_once",
                dropping_stream_without_polling_reports_dropped_once,
            ),
            (
                "internal_attempt_input_execute_error_reports_one_failure",
                internal_attempt_input_execute_error_reports_one_failure,
            ),
        ]);
    }

    async fn internal_attempt_stream_cancellation_reports_cancelled() {
        let backend = InstalledTestGpuBackend::new();
        let exec = admission_exec(
            &backend,
            lifecycle_exec(LifecyclePlanBehavior::CancelledStream, Arc::new(Mutex::new(None))),
        );

        let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("cancelled consumer stream starts");
        let error =
            stream.next().await.expect("consumer emits cancellation").expect_err("cancellation is an error item");
        assert_eq!(error.find_algeon_error().expect("cancelled error remains typed").code(), ErrorCode::Cancelled);
        drop(stream);

        assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Cancelled);
    }

    fn dropping_task_context_before_stream_does_not_end_the_internal_attempt() {
        let backend = InstalledTestGpuBackend::new();
        let exec = admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::Success, Arc::new(Mutex::new(None))));
        let task_context = Arc::new(TaskContext::default());

        let stream = exec.execute(0, Arc::clone(&task_context)).expect("synthetic stream starts");
        drop(task_context);
        assert!(
            backend.backend().take_abandoned_attempt_reports(8).is_empty(),
            "the stream owns the attempt after its task context is dropped"
        );
        drop(stream);
        assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Dropped);
    }

    fn dropping_stream_without_polling_reports_dropped_once() {
        let backend = InstalledTestGpuBackend::new();
        let exec = admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::Success, Arc::new(Mutex::new(None))));

        let stream = exec.execute(0, Arc::new(TaskContext::default())).expect("synthetic stream starts");
        drop(stream);
        assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Dropped);
    }
}
