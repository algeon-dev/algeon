//! Attempt admission tickets and the native admission root node.

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
static NEXT_TICKET_ID: AtomicU64 = AtomicU64::new(1);

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
use query_engine::{
    admission::AttemptCapabilityEvidence,
    attempt::{
        AttemptAdmissionBinding, AttemptAdmissionRejection, AttemptCancellation, AttemptExecutionRound, AttemptHandle,
        AttemptHostOccurrence,
    },
    error::{QueryError, QueryErrorCode, Result as QueryEngineResult},
    observability::{AttemptAdmissionWaitFact, AttemptComposedScheduleProjection},
    plan::QueryPlan,
    runtime::{AdmittedQueryContext, ExecutionCancellation, GpuBackendSession, QueryAdmissionRequest, QueryHandle},
};
use tokio::sync::Notify;

use crate::{
    attempt::execute_in_internal_attempt,
    error::{Error, ErrorCode},
    native_exec::metrics::NativeExecutionMetricsCapture,
    planner::native::Invocation,
};

/// Backend-owned service admission source copied into optimizer configs.
#[derive(Debug, Clone)]
pub(crate) struct QueryAdmissionSource {
    session: GpuBackendSession,
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
    admitting: bool,
    admission_facts_recorded: bool,
    /// Fragments bound while the claim was in flight; they join its grant
    /// when it resolves.
    pending_fragments: Vec<AttemptCapabilityEvidence>,
}

/// The ticket of one DataFusion execution of a native admission root, with
/// the occurrence round of that execution.
#[derive(Debug)]
struct ActiveQueryAdmissionTicket {
    ticket: Arc<QueryAdmissionTicket>,
    execution_round: AttemptExecutionRound,
}

pub(crate) fn query_admission_ticket_from_task_context(
    task_context: &TaskContext,
) -> Option<Arc<QueryAdmissionTicket>> {
    task_context
        .session_config()
        .get_extension::<ActiveQueryAdmissionTicket>()
        .map(|extension| Arc::clone(&extension.ticket))
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
    session_config.set_extension(Arc::new(ActiveQueryAdmissionTicket { ticket, execution_round }));
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
/// for it. The attempt has no second claim, so the fragments waiting on that
/// admission get its abandonment rather than a claim of their own.
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
    ) -> DataFusionResult<Option<AttemptHostOccurrence>> {
        if !self.has_execution_manifest() {
            return Ok(None);
        }
        let execution_round = task_context
            .session_config()
            .get_extension::<ActiveQueryAdmissionTicket>()
            .filter(|active| Arc::ptr_eq(&active.ticket, self))
            .map(|active| active.execution_round.clone())
            .ok_or_else(|| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "algeon_datafusion.native_exec.host_occurrence",
                        "a GPU function occurrence began outside its invocation execution round",
                    )
                    .with_log_safe_fact("reason", "host_occurrence_execution_round_missing")
                    .with_log_safe_fact("occurrence_id", occurrence_id)
                    .with_log_safe_fact("producer_identity", producer_identity),
                )
            })?;
        execution_round.begin_host_occurrence(occurrence_id, producer_identity, plan).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.native_exec.host_occurrence",
                "the GPU function cannot begin in its admitted execution schedule",
                error,
            ))
        })
    }

    pub(crate) fn validate_host_occurrence(
        &self,
        producer_identity: &str,
        plan: &QueryPlan,
    ) -> DataFusionResult<Option<()>> {
        let Some(evidence) = self.current_capability_evidence() else {
            return Ok(None);
        };
        let Some(manifest) = evidence.execution_manifest() else {
            return Ok(None);
        };
        manifest.validate_host_producer(producer_identity, plan).map(Some).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.native_exec.host_occurrence",
                "the executing GPU function does not match an admitted host producer",
                error,
            ))
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

pub(crate) fn query_admission_wrapped_input(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    (plan as &dyn Any).downcast_ref::<NativeAdmissionExec>().map(|wrapper| &wrapper.input)
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
#[path = "admission_tests.rs"]
mod tests;
