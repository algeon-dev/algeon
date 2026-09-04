//! Plan-scoped query-service admission tickets for backend-managed execution.

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
static NEXT_TICKET_ID: AtomicU64 = AtomicU64::new(1);
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{Arc, Mutex, OnceLock},
    task::{Context, Poll},
    time::Duration,
};

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::{OrderingRequirements, PhysicalExpr},
    physical_plan::{
        ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, InputDistributionRequirements,
        PlanProperties, RecordBatchStream, ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs,
        StatisticsContext, metrics::MetricsSet,
    },
};
use futures::Stream;
#[cfg(test)]
use nexus_query_engine::metrics::QueryId;
use nexus_query_engine::{
    admission::AttemptCapabilityEvidence,
    attempt::{AttemptAdmissionBinding, AttemptCancellation, AttemptHandle},
    observability::{
        AttemptAdmissionFact, AttemptComposedScheduleProjection, RuntimeObservationConfig, RuntimeObservationOwner,
        RuntimeQueryId, RuntimeTerminalOutcome, RuntimeTraceId,
    },
    runtime::{AdmittedQueryContext, ExecutionCancellation, GpuBackendSession, QueryAdmissionRequest, QueryHandle},
};
use tokio::sync::Notify;

use crate::{
    attempt::NexusAttemptExtension,
    error::{DataFusionErrorExt, Error, ErrorCode},
    native_exec::metrics::{NativeExecutionMetricsCapture, NativeRuntimeObservation},
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
    pub(crate) admission_wait_fact: Option<nexus_query_engine::observability::AttemptAdmissionWaitFact>,
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

    fn admission_wait_fact(&self) -> Option<nexus_query_engine::observability::AttemptAdmissionWaitFact> {
        self.0.admission_wait_fact()
    }

    fn device_ordinal(&self) -> usize {
        self.0.device_ordinal()
    }

    fn device_budget_bytes(&self) -> u64 {
        self.0.device_budget_bytes()
    }

    async fn acquire_context_with_cancellation_async(
        &self,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::error::Result<AdmittedQueryContext> {
        self.0.acquire_context_with_cancellation_async(cancellation).await
    }

    fn finish(&self) {
        self.0.finish()
    }
}

/// Lazy admission ticket shared by every native node emitted in one optimize pass.
#[derive(Debug)]
pub(crate) struct QueryAdmissionTicket {
    attempt: Option<AttemptTicketBinding>,
    #[cfg(test)]
    id: u64,
    #[cfg(test)]
    source: Option<QueryAdmissionSource>,
    #[cfg(not(test))]
    source: QueryAdmissionSource,
    /// Engine-issued evidence for the native fragments this ticket may execute.
    /// `None` is intentionally not an empty explicit evidence: all-modeled
    /// services may admit it for resource-only use, while exact profiles
    /// reject it and execution cannot use it to authorize a native plan.
    attempt_capability_evidence: Option<AttemptCapabilityEvidence>,
    /// A composed root can resolve footer-bound child evidence only after its
    /// outer DataFusion stream has already selected this ticket. That one
    /// pre-admission replacement preserves the active stream's lifecycle
    /// ownership instead of admitting a second ticket for the same attempt.
    rebound_capability_evidence: OnceLock<AttemptCapabilityEvidence>,
    observation_query_id: Option<RuntimeQueryId>,
    observation_trace_id: Option<RuntimeTraceId>,
    observation_config: RuntimeObservationConfig,
    observation_capture: Option<NativeExecutionMetricsCapture>,
    state: Mutex<TicketState>,
    notify: Arc<Notify>,
}

#[derive(Debug)]
struct ActiveQueryAdmissionTicket {
    ticket: Arc<QueryAdmissionTicket>,
    round_task_context: Arc<TaskContext>,
}

pub(crate) fn query_admission_ticket_from_task_context(
    task_context: &TaskContext,
) -> Option<Arc<QueryAdmissionTicket>> {
    task_context
        .session_config()
        .get_extension::<ActiveQueryAdmissionTicket>()
        .map(|extension| Arc::clone(&extension.ticket))
}

/// Invocation-scoped attempt state attached to an admission ticket.
#[derive(Debug, Clone)]
pub(crate) struct AttemptTicketBinding {
    pub(crate) handle: AttemptHandle,
    pub(crate) admission_rejection: nexus_query_engine::attempt::AttemptAdmissionRejection,
    pub(crate) admission_binding: AttemptAdmissionBinding,
    pub(crate) cancellation: AttemptCancellation,
    /// Exact cap supplied by the invocation owner, if any. It is copied into
    /// every query-service admission request for this attempt.
    pub(crate) requested_query_cap_bytes: Option<u64>,
}

pub(crate) fn task_context_with_query_admission_ticket(
    task_context: Arc<TaskContext>,
    ticket: Arc<QueryAdmissionTicket>,
) -> Arc<TaskContext> {
    if query_admission_ticket_from_task_context(&task_context).is_some() {
        return task_context;
    }

    let mut session_config = task_context.session_config().clone();
    session_config
        .set_extension(Arc::new(ActiveQueryAdmissionTicket { ticket, round_task_context: Arc::clone(&task_context) }));
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

#[derive(Debug)]
struct TicketState {
    rounds: HashMap<usize, TicketRound>,
    admitting: HashSet<usize>,
    next_epoch: u64,
    initial_observation: Option<NativeRuntimeObservation>,
}

#[derive(Debug)]
struct TicketRound {
    _task_context: Arc<TaskContext>,
    handle: Option<Arc<TicketQueryHandle>>,
    active_contexts: usize,
    active_streams: usize,
    epoch: u64,
    admission_facts_recorded: bool,
    terminal_outcome: Option<RuntimeTerminalOutcome>,
    observation: NativeRuntimeObservation,
}

/// Context acquired through a [`QueryAdmissionTicket`].
#[derive(Debug)]
pub(crate) struct TicketAdmissionContext {
    admission_context: AdmittedQueryContext,
    task_context: Arc<TaskContext>,
    admission_facts: QueryAdmissionFacts,
    record_admission_facts: bool,
    _guard: QueryAdmissionTicketContextGuard,
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

#[derive(Debug)]
pub(crate) struct QueryAdmissionTicketContextGuard {
    ticket: Arc<QueryAdmissionTicket>,
    task_key: usize,
    epoch: u64,
    released: bool,
}

#[derive(Debug)]
pub(crate) struct QueryAdmissionExecutionRoundGuard {
    ticket: Arc<QueryAdmissionTicket>,
    task_key: usize,
    epoch: u64,
    terminal_outcome_recorded: bool,
    released: bool,
}

#[derive(Debug)]
struct QueryAdmissionPendingContextGuard {
    ticket: Arc<QueryAdmissionTicket>,
    task_key: usize,
    epoch: u64,
    released: bool,
    observation: NativeRuntimeObservation,
}

struct TicketAdmitResetGuard {
    ticket: Arc<QueryAdmissionTicket>,
    task_key: usize,
    disarmed: bool,
    observation: NativeRuntimeObservation,
}

impl QueryAdmissionTicket {
    #[cfg(test)]
    pub(crate) fn new(
        source: QueryAdmissionSource,
        attempt_capability_evidence: Option<AttemptCapabilityEvidence>,
    ) -> Arc<Self> {
        Self::new_with_capability_evidence_and_observation(
            source,
            attempt_capability_evidence,
            RuntimeObservationOwner::new(QueryId::mint(), None, RuntimeObservationConfig::native_execution()),
            None,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_with_observation(
        source: QueryAdmissionSource,
        attempt_capability_evidence: Option<AttemptCapabilityEvidence>,
        observation_owner: RuntimeObservationOwner,
    ) -> Arc<Self> {
        Self::new_with_capability_evidence_and_observation(source, attempt_capability_evidence, observation_owner, None)
    }

    pub(crate) fn new_with_capability_evidence_and_observation(
        source: QueryAdmissionSource,
        attempt_capability_evidence: Option<AttemptCapabilityEvidence>,
        observation: impl Into<NativeRuntimeObservation>,
        observation_capture: Option<NativeExecutionMetricsCapture>,
    ) -> Arc<Self> {
        let observation = observation.into();
        let observation_handle = observation.handle();
        let observation_query_id = observation_handle.query_id().cloned();
        let observation_trace_id = observation_handle.trace_id().cloned();
        let observation_config = observation_handle.config();
        Arc::new(Self {
            attempt: None,
            #[cfg(test)]
            id: NEXT_TICKET_ID.fetch_add(1, Ordering::Relaxed),
            #[cfg(test)]
            source: Some(source),
            #[cfg(not(test))]
            source,
            attempt_capability_evidence,
            rebound_capability_evidence: OnceLock::new(),
            observation_query_id,
            observation_trace_id,
            observation_config,
            observation_capture,
            state: Mutex::new(TicketState {
                rounds: HashMap::new(),
                admitting: HashSet::new(),
                next_epoch: 0,
                initial_observation: Some(observation),
            }),
            notify: Arc::new(Notify::new()),
        })
    }

    /// Creates a ticket whose lifecycle can be exercised without constructing
    /// a GPU backend.  The test-only ticket deliberately cannot acquire a
    /// native context; it is used to drive the execution-round and stream
    /// guards through their CPU-visible cleanup paths.
    #[cfg(test)]
    pub(crate) fn new_for_lifecycle_test(observation_owner: RuntimeObservationOwner) -> Arc<Self> {
        let observation = NativeRuntimeObservation::Local(observation_owner);
        let observation_handle = observation.handle();
        let observation_query_id = observation_handle.query_id().cloned();
        let observation_trace_id = observation_handle.trace_id().cloned();
        let observation_config = observation_handle.config();
        Arc::new(Self {
            attempt: None,
            id: NEXT_TICKET_ID.fetch_add(1, Ordering::Relaxed),
            source: None,
            attempt_capability_evidence: None,
            rebound_capability_evidence: OnceLock::new(),
            observation_query_id,
            observation_trace_id,
            observation_config,
            observation_capture: None,
            state: Mutex::new(TicketState {
                rounds: HashMap::new(),
                admitting: HashSet::new(),
                next_epoch: 0,
                initial_observation: Some(observation),
            }),
            notify: Arc::new(Notify::new()),
        })
    }

    /// Creates a ticket bound to one explicit adapter invocation.
    pub(crate) fn new_attempt_bound(
        source: QueryAdmissionSource,
        attempt_capability_evidence: Option<AttemptCapabilityEvidence>,
        binding: AttemptTicketBinding,
        _observation_capture: Option<NativeExecutionMetricsCapture>,
    ) -> Arc<Self> {
        let observation = binding.admission_binding.observation_handle();
        let observation_capture = Some(NativeExecutionMetricsCapture::coordinated(observation.clone()));
        let mut ticket = Self::new_with_capability_evidence_and_observation(
            source,
            attempt_capability_evidence,
            NativeRuntimeObservation::Coordinated(observation),
            observation_capture,
        );
        Arc::get_mut(&mut ticket).expect("new admission ticket is uniquely owned").attempt = Some(binding);
        ticket
    }

    pub(crate) fn source(&self) -> QueryAdmissionSource {
        #[cfg(test)]
        {
            self.source.clone().expect("lifecycle-only test ticket has no admission source")
        }
        #[cfg(not(test))]
        {
            self.source.clone()
        }
    }

    fn admission_source(&self) -> &QueryAdmissionSource {
        #[cfg(test)]
        {
            self.source.as_ref().expect("lifecycle-only test ticket cannot acquire a native context")
        }
        #[cfg(not(test))]
        {
            &self.source
        }
    }

    pub(crate) fn attempt_capability_evidence_for_rebind(&self) -> Option<AttemptCapabilityEvidence> {
        self.current_capability_evidence().cloned()
    }

    fn current_capability_evidence(&self) -> Option<&AttemptCapabilityEvidence> {
        self.rebound_capability_evidence.get().or(self.attempt_capability_evidence.as_ref())
    }

    /// Replaces this ticket's evidence exactly once before any admission request
    /// has started. The outer admission stream may already be alive, but it
    /// has not acquired a query-service handle yet; preserving that ticket is
    /// what keeps the eventual host-output cursor's resources alive to EOF.
    pub(crate) fn rebind_capability_evidence_before_admission(
        &self,
        evidence: AttemptCapabilityEvidence,
    ) -> nexus_query_engine::error::Result<()> {
        let state = self.lock_state()?;
        if !state.admitting.is_empty() || state.rounds.values().any(|round| round.handle.is_some()) {
            return Err(nexus_query_engine::error::QueryError::new(
                nexus_query_engine::error::QueryErrorCode::InvalidPlan,
                "datafusion_nexus.native_exec.admission_ticket",
                "cannot replace a query admission capability evidence after admission has started",
            )
            .with_fact("reason", "capability_evidence_rebind_after_admission"));
        }
        match self.rebound_capability_evidence.get() {
            Some(existing) if existing == &evidence => Ok(()),
            Some(_) => Err(nexus_query_engine::error::QueryError::new(
                nexus_query_engine::error::QueryErrorCode::InvalidPlan,
                "datafusion_nexus.native_exec.admission_ticket",
                "one query admission ticket cannot carry conflicting composed capability evidence_items",
            )
            .with_fact("reason", "conflicting_capability_evidence_rebind")),
            None => self.rebound_capability_evidence.set(evidence).map_err(|_| {
                nexus_query_engine::error::QueryError::new(
                    nexus_query_engine::error::QueryErrorCode::Internal,
                    "datafusion_nexus.native_exec.admission_ticket",
                    "query admission capability evidence changed while rebinding the composed root",
                )
                .with_fact("reason", "capability_evidence_rebind_race")
            }),
        }
    }

    /// Reissues an unused ticket with the complete post-preparation evidence.
    ///
    /// Preparation is the last point at which a composed GPU island can learn
    /// its child metadata.  Preserve the attempt binding and observation
    /// identity when one exists; the caller is responsible for installing the
    /// returned ticket as the invocation's canonical ticket before admission.
    pub(crate) fn rebind_with_source_and_capability_evidence(
        &self,
        source: QueryAdmissionSource,
        attempt_capability_evidence: AttemptCapabilityEvidence,
    ) -> Arc<Self> {
        match &self.attempt {
            Some(binding) => Self::new_attempt_bound(
                source,
                Some(attempt_capability_evidence),
                binding.clone(),
                self.metrics_capture(),
            ),
            None => Self::new_with_capability_evidence_and_observation(
                source,
                Some(attempt_capability_evidence),
                self.new_observation(),
                self.metrics_capture(),
            ),
        }
    }

    pub(crate) fn is_attempt_bound(&self) -> bool {
        self.attempt.is_some()
    }

    pub(crate) fn record_attempt_gpu_execution_started(&self) {
        if let Some(binding) = &self.attempt {
            let _ = binding.handle.record_gpu_execution_started();
        }
    }

    /// Publishes the engine-owned composed schedule when this ticket carries
    /// an attempt lifecycle binding. Standalone tickets have no sealed report
    /// owner and therefore intentionally retain no projection.
    pub(crate) fn record_attempt_composed_schedule_projection(
        &self,
        projection: AttemptComposedScheduleProjection,
    ) -> nexus_query_engine::error::Result<()> {
        let Some(binding) = &self.attempt else {
            return Ok(());
        };
        binding.handle.record_composed_schedule_projection(projection).map_err(|error| {
            nexus_query_engine::error::QueryError::new(
                nexus_query_engine::error::QueryErrorCode::Internal,
                "datafusion_nexus.native_exec.admission_ticket",
                "failed to publish the admitted composed schedule to the attempt report",
            )
            .with_fact("reason", "composed_schedule_projection_record_failed")
            .with_fact("lifecycle_error", error.to_string())
        })
    }

    /// Captures a native failure while it is still borrowed by this adapter
    /// path.  This is deliberately record-only: the post-cleanup finalizer
    /// remains the sole admitted terminal/seal authority.
    fn record_attempt_native_failure(&self, error: &nexus_query_engine::error::QueryError) {
        if let Some(binding) = &self.attempt {
            binding.handle.record_native_failure(error);
        }
    }

    /// Extracts an engine error from a returned DataFusion error before the
    /// admission round releases an admitted handle.  The outer attempt owner
    /// still receives the unchanged DataFusion error and remains responsible
    /// for ordinary non-admitted terminalization.
    fn record_attempt_datafusion_failure(&self, error: &DataFusionError) {
        let Some(error) = error.find_nexus_error() else { return };
        let Some(native) =
            error.source_error().and_then(|source| source.downcast_ref::<nexus_query_engine::error::QueryError>())
        else {
            return;
        };
        self.record_attempt_native_failure(native);
    }

    fn attempt_cancellation(&self, fallback: ExecutionCancellation) -> ExecutionCancellation {
        self.attempt.as_ref().map(|binding| binding.cancellation.execution_cancellation()).unwrap_or(fallback)
    }

    fn terminalize_attempt_admission_error(&self, error: &nexus_query_engine::error::QueryError) {
        let Some(binding) = &self.attempt else { return };
        let _ = binding.admission_rejection.reject(error);
    }

    #[cfg(test)]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    #[cfg(test)]
    pub(crate) fn attempt_capability_evidence(&self) -> Option<&AttemptCapabilityEvidence> {
        self.current_capability_evidence()
    }

    pub(crate) fn metrics_capture(&self) -> Option<NativeExecutionMetricsCapture> {
        self.observation_capture.clone()
    }

    fn new_observation(&self) -> NativeRuntimeObservation {
        match &self.observation_query_id {
            Some(query_id) => match &self.observation_capture {
                Some(capture) => capture.new_runtime_observation(query_id.clone(), self.observation_trace_id.clone()),
                None => NativeRuntimeObservation::Local(RuntimeObservationOwner::new(
                    query_id.clone(),
                    self.observation_trace_id.clone(),
                    self.observation_config,
                )),
            },
            None => NativeRuntimeObservation::Local(RuntimeObservationOwner::disabled()),
        }
    }

    fn take_epoch_observation(&self, state: &mut TicketState) -> NativeRuntimeObservation {
        state.initial_observation.take().unwrap_or_else(|| self.new_observation())
    }

    pub(crate) fn begin_execution_round(
        self: &Arc<Self>,
        task_context: Arc<TaskContext>,
    ) -> nexus_query_engine::error::Result<QueryAdmissionExecutionRoundGuard> {
        let task_key = Self::task_key(&task_context);
        let mut state = self.lock_state()?;
        let stale_rounds = Self::remove_stale_rounds_locked(&mut state);
        let epoch = if let Some(round) = state.rounds.get_mut(&task_key) {
            round.active_streams = round.active_streams.saturating_add(1);
            round.epoch
        } else {
            state.next_epoch = state.next_epoch.saturating_add(1);
            let epoch = state.next_epoch;
            let observation = self.take_epoch_observation(&mut state);
            state.rounds.insert(
                task_key,
                TicketRound {
                    _task_context: Arc::clone(&task_context),
                    handle: None,
                    active_contexts: 0,
                    active_streams: 1,
                    epoch,
                    admission_facts_recorded: false,
                    terminal_outcome: None,
                    observation,
                },
            );
            epoch
        };
        drop(state);
        self.finish_orphaned_rounds(stale_rounds);
        Ok(QueryAdmissionExecutionRoundGuard {
            ticket: Arc::clone(self),
            task_key,
            epoch,
            terminal_outcome_recorded: false,
            released: false,
        })
    }

    pub(crate) async fn acquire_context(
        self: &Arc<Self>,
        task_context: Arc<TaskContext>,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::error::Result<TicketAdmissionContext> {
        let task_key = Self::task_key(&task_context);
        loop {
            let notify = Arc::clone(&self.notify);
            let notified = notify.notified_owned();
            tokio::pin!(notified);
            let (action, stale_rounds) = {
                let mut state = self.lock_state()?;
                let stale_rounds = Self::remove_stale_rounds_locked(&mut state);
                let action = if let Some(round) = state.rounds.get_mut(&task_key)
                    && let Some(handle) = round.handle.as_ref().cloned()
                {
                    let epoch = round.epoch;
                    let observation = round.observation.clone();
                    round.active_contexts = round.active_contexts.saturating_add(1);
                    TicketAcquireAction::Reuse { handle, epoch, observation }
                } else {
                    notified.as_mut().enable();
                    if state.admitting.contains(&task_key) {
                        TicketAcquireAction::Wait
                    } else {
                        if !state.rounds.contains_key(&task_key) {
                            state.next_epoch = state.next_epoch.saturating_add(1);
                            let epoch = state.next_epoch;
                            let observation = self.take_epoch_observation(&mut state);
                            state.rounds.insert(
                                task_key,
                                TicketRound {
                                    _task_context: Arc::clone(&task_context),
                                    handle: None,
                                    active_contexts: 0,
                                    active_streams: 0,
                                    epoch,
                                    admission_facts_recorded: false,
                                    terminal_outcome: None,
                                    observation,
                                },
                            );
                        }
                        let round = state.rounds.get(&task_key).expect("ticket round was just established");
                        let epoch = round.epoch;
                        let observation = round.observation.clone();
                        state.admitting.insert(task_key);
                        TicketAcquireAction::Admit { epoch, observation }
                    }
                };
                (action, stale_rounds)
            };
            self.finish_orphaned_rounds(stale_rounds);

            match action {
                TicketAcquireAction::Reuse { handle, epoch, observation } => {
                    return self
                        .acquire_from_handle(
                            handle,
                            task_key,
                            epoch,
                            Arc::clone(&task_context),
                            cancellation,
                            observation,
                        )
                        .await;
                },
                TicketAcquireAction::Wait => {
                    notified.await;
                    continue;
                },
                TicketAcquireAction::Admit { epoch, observation } => {
                    return self
                        .admit_and_acquire_context(
                            task_key,
                            Arc::clone(&task_context),
                            epoch,
                            cancellation,
                            observation,
                        )
                        .await;
                },
            }
        }
    }

    async fn admit_and_acquire_context(
        self: &Arc<Self>,
        task_key: usize,
        task_context: Arc<TaskContext>,
        epoch: u64,
        cancellation: ExecutionCancellation,
        observation: NativeRuntimeObservation,
    ) -> nexus_query_engine::error::Result<TicketAdmissionContext> {
        let cancellation = self.attempt_cancellation(cancellation);
        if let Some(binding) = &self.attempt {
            let _ = binding.handle.record_admission_requested();
        }
        let mut reset_guard = TicketAdmitResetGuard {
            ticket: Arc::clone(self),
            task_key,
            disarmed: false,
            observation: observation.clone(),
        };
        let request = QueryAdmissionRequest::new();
        let request = match self.attempt.as_ref().and_then(|binding| binding.requested_query_cap_bytes) {
            Some(query_cap_bytes) => request.with_query_cap_bytes(query_cap_bytes),
            None => request,
        };
        let request = request.with_execution_cancellation(cancellation.clone());
        let request = match self.attempt_capability_evidence_for_rebind() {
            Some(evidence) => request.with_query_plan_capability_evidence(evidence),
            None => request,
        };
        let request = if let Some(binding) = &self.attempt
            && let Some(deadline) = binding.cancellation.deadline()
        {
            request.with_queue_deadline(deadline)
        } else {
            request
        };
        let request = match (&self.attempt, &observation) {
            (Some(binding), _) => request.with_attempt_binding(binding.admission_binding.clone()),
            (None, NativeRuntimeObservation::Local(owner)) => request.with_runtime_observation_owner(owner.clone()),
            (None, NativeRuntimeObservation::Coordinated(handle)) => {
                request.with_runtime_observation_handle(handle.clone())
            },
        };
        let observation_activity = if self.attempt.is_none() {
            self.observation_capture.as_ref().and_then(NativeExecutionMetricsCapture::new_observation_activity)
        } else {
            None
        };
        let request = match observation_activity {
            Some(activity) => request.with_runtime_observation_activity_result(activity),
            None => request,
        };
        let handle_result = self
            .admission_source()
            .session
            .admit_async_with_request(request)
            .await
            .map(|handle| TicketQueryHandle(Arc::new(handle)));
        let handle = match handle_result {
            Ok(handle) => Arc::new(handle),
            Err(error) => {
                self.terminalize_attempt_admission_error(&error);
                self.finish_observation_for_error(&observation, &error);
                reset_guard.reset();
                return Err(error);
            },
        };

        // Query-service admission has now published the immutable grant and
        // moved the sole post-cleanup finalizer into the returned handle. The
        // attempt lifecycle must record that fact before any later context
        // acquisition can fail; otherwise the handle's necessary cleanup
        // would have no truthful admitted lifecycle to finalize.
        let admission_facts = QueryAdmissionFacts::from_handle(handle.as_ref());
        if let Some(binding) = &self.attempt {
            let _ = binding.handle.record_admitted(AttemptAdmissionFact {
                admission_sequence: admission_facts.admission_sequence,
                admission_wait_ns: admission_facts.admission_wait.as_nanos().min(u128::from(u64::MAX)) as u64,
                device_ordinal: admission_facts.device_ordinal,
                grant_bytes: admission_facts.query_grant_bytes,
            });
            if let Some(fact) = admission_facts.admission_wait_fact {
                let _ = binding.handle.record_admission_wait(fact);
            }
        }

        let epoch = {
            let mut state = match self.lock_state() {
                Ok(state) => state,
                Err(error) => {
                    reset_guard.disarmed = true;
                    self.record_observation_outcome_for_error(&observation, &error);
                    drop(handle);
                    return Err(error);
                },
            };
            state.admitting.remove(&task_key);
            let Some(round) = state.rounds.get_mut(&task_key) else {
                reset_guard.disarmed = true;
                self.notify.notify_waiters();
                let error = nexus_query_engine::error::QueryError::new(
                    nexus_query_engine::error::QueryErrorCode::Internal,
                    "datafusion_nexus.native_exec.admission_ticket",
                    "query admission ticket execution round disappeared during admission",
                );
                self.record_observation_outcome_for_error(&observation, &error);
                drop(state);
                drop(handle);
                return Err(error);
            };
            if round.epoch != epoch {
                reset_guard.disarmed = true;
                self.notify.notify_waiters();
                let error = nexus_query_engine::error::QueryError::new(
                    nexus_query_engine::error::QueryErrorCode::Internal,
                    "datafusion_nexus.native_exec.admission_ticket",
                    "query admission ticket execution round changed during admission",
                );
                self.record_observation_outcome_for_error(&observation, &error);
                drop(state);
                drop(handle);
                return Err(error);
            }
            round.handle = Some(Arc::clone(&handle));
            round.active_contexts = round.active_contexts.saturating_add(1);
            reset_guard.disarmed = true;
            self.notify.notify_waiters();
            epoch
        };
        self.acquire_from_handle(handle, task_key, epoch, task_context, cancellation, observation).await
    }

    async fn acquire_from_handle(
        self: &Arc<Self>,
        handle: Arc<TicketQueryHandle>,
        task_key: usize,
        epoch: u64,
        task_context: Arc<TaskContext>,
        cancellation: ExecutionCancellation,
        observation: NativeRuntimeObservation,
    ) -> nexus_query_engine::error::Result<TicketAdmissionContext> {
        let mut pending_guard = QueryAdmissionPendingContextGuard {
            ticket: Arc::clone(self),
            task_key,
            epoch,
            released: false,
            observation: observation.clone(),
        };
        match handle.acquire_context_with_cancellation_async(self.attempt_cancellation(cancellation)).await {
            Ok(admission_context) => {
                pending_guard.released = true;
                let admission_facts = QueryAdmissionFacts::from_handle(handle.as_ref());
                if let Some(binding) = &self.attempt {
                    let _ = binding.handle.record_physical_execution_started();
                }
                Ok(TicketAdmissionContext {
                    admission_context,
                    task_context,
                    admission_facts,
                    record_admission_facts: self.mark_admission_facts_recorded(task_key, epoch),
                    _guard: QueryAdmissionTicketContextGuard {
                        ticket: Arc::clone(self),
                        task_key,
                        epoch,
                        released: false,
                    },
                })
            },
            Err(error) => {
                self.terminalize_attempt_admission_error(&error);
                self.record_observation_outcome_for_error(&observation, &error);
                Err(error)
            },
        }
    }

    fn mark_admission_facts_recorded(&self, task_key: usize, epoch: u64) -> bool {
        let Ok(mut state) = self.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to mark poisoned query admission ticket facts state");
            return false;
        };
        let Some(round) = state.rounds.get_mut(&task_key) else {
            return false;
        };
        if round.epoch != epoch || round.admission_facts_recorded {
            return false;
        }
        round.admission_facts_recorded = true;
        true
    }

    fn release_epoch_context(self: &Arc<Self>, task_key: usize, epoch: u64, force: bool) {
        let Ok(mut state) = self.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to release poisoned query admission ticket state");
            return;
        };
        let Some(round) = state.rounds.get_mut(&task_key) else {
            return;
        };
        if round.epoch != epoch || round.active_contexts == 0 {
            return;
        }
        round.active_contexts -= 1;
        let remove = round.active_contexts == 0 && (force || round.active_streams == 0);
        let removed = remove.then(|| state.rounds.remove(&task_key)).flatten();
        if removed.is_some() {
            self.notify.notify_waiters();
        }
        drop(state);
        self.finish_round(removed);
    }

    fn release_execution_round(self: &Arc<Self>, task_key: usize, epoch: u64) {
        let Ok(mut state) = self.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to release poisoned query admission execution round state");
            return;
        };
        let Some(round) = state.rounds.get_mut(&task_key) else {
            return;
        };
        if round.epoch != epoch || round.active_streams == 0 {
            return;
        }
        round.active_streams -= 1;
        let remove = round.active_streams == 0 && round.active_contexts == 0;
        let removed = remove.then(|| state.rounds.remove(&task_key)).flatten();
        if removed.is_some() {
            self.notify.notify_waiters();
        }
        drop(state);
        self.finish_round(removed);
    }

    fn finish_round(&self, round: Option<TicketRound>) {
        let Some(round) = round else { return };
        match round.handle {
            Some(handle) => handle.finish(),
            None => self.finish_observation(
                &round.observation,
                round.terminal_outcome.unwrap_or_else(|| Self::drop_outcome(RuntimeTerminalOutcome::Dropped)),
            ),
        }
    }

    fn finish_orphaned_rounds(&self, rounds: Vec<TicketRound>) {
        for round in rounds {
            self.finish_round(Some(round));
        }
    }

    fn finish_observation(&self, observation: &NativeRuntimeObservation, outcome: RuntimeTerminalOutcome) {
        Self::finish_observation_isolated(observation, outcome, self.observation_capture.as_ref());
    }

    fn finish_observation_isolated(
        observation: &NativeRuntimeObservation,
        outcome: RuntimeTerminalOutcome,
        _capture: Option<&NativeExecutionMetricsCapture>,
    ) {
        if let NativeRuntimeObservation::Local(owner) = observation {
            let _ = catch_unwind(AssertUnwindSafe(|| owner.finish_query(Some(outcome))));
        }
    }

    fn drop_outcome(normal: RuntimeTerminalOutcome) -> RuntimeTerminalOutcome {
        if std::thread::panicking() { RuntimeTerminalOutcome::Panicked } else { normal }
    }

    fn reset_admitting(&self, task_key: usize) {
        let Ok(mut state) = self.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to reset poisoned query admission ticket state");
            return;
        };
        state.admitting.remove(&task_key);
        self.notify.notify_waiters();
    }

    fn observation_outcome_for_error(error: &nexus_query_engine::error::QueryError) -> RuntimeTerminalOutcome {
        RuntimeTerminalOutcome::from_query_error(error)
    }

    fn record_observation_outcome_for_error(
        &self,
        observation: &NativeRuntimeObservation,
        error: &nexus_query_engine::error::QueryError,
    ) {
        // An attempt-bound ticket has only a coordinated observation handle
        // here, so its release can otherwise finalise before the outer
        // adapter gets a chance to inspect this borrowed error.
        self.record_attempt_native_failure(error);
        if let NativeRuntimeObservation::Local(owner) = observation {
            owner.record_terminal_outcome(Self::observation_outcome_for_error(error));
        }
    }

    fn finish_observation_for_error(
        &self,
        observation: &NativeRuntimeObservation,
        error: &nexus_query_engine::error::QueryError,
    ) {
        self.finish_observation(observation, Self::observation_outcome_for_error(error));
    }

    fn lock_state(&self) -> nexus_query_engine::error::Result<std::sync::MutexGuard<'_, TicketState>> {
        self.state.lock().map_err(|_error| {
            nexus_query_engine::error::QueryError::new(
                nexus_query_engine::error::QueryErrorCode::Internal,
                "datafusion_nexus.native_exec.admission_ticket",
                "query admission ticket state mutex was poisoned",
            )
        })
    }

    fn task_key(task_context: &Arc<TaskContext>) -> usize {
        task_context
            .session_config()
            .get_extension::<ActiveQueryAdmissionTicket>()
            .map(|extension| Arc::as_ptr(&extension.round_task_context) as usize)
            .unwrap_or_else(|| Arc::as_ptr(task_context) as usize)
    }

    fn remove_stale_rounds_locked(state: &mut TicketState) -> Vec<TicketRound> {
        let stale_keys = state
            .rounds
            .iter()
            .filter_map(|(task_key, round)| {
                (round.active_contexts == 0 && round.active_streams == 0).then_some(*task_key)
            })
            .collect::<Vec<_>>();
        stale_keys.into_iter().filter_map(|task_key| state.rounds.remove(&task_key)).collect()
    }
}

impl Drop for QueryAdmissionTicket {
    fn drop(&mut self) {
        let observation_capture = self.observation_capture.clone();
        let state = self.state.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner);
        let initial_observation = state.initial_observation.take();
        let orphaned_rounds = state.rounds.drain().map(|(_, round)| round).collect::<Vec<_>>();
        let outcome =
            if std::thread::panicking() { RuntimeTerminalOutcome::Panicked } else { RuntimeTerminalOutcome::Dropped };
        if let Some(observation) = initial_observation {
            Self::finish_observation_isolated(&observation, outcome.clone(), observation_capture.as_ref());
        }
        for round in orphaned_rounds {
            match round.handle {
                Some(handle) => handle.finish(),
                None => {
                    Self::finish_observation_isolated(&round.observation, outcome.clone(), observation_capture.as_ref())
                },
            }
        }
    }
}

enum TicketAcquireAction {
    Reuse { handle: Arc<TicketQueryHandle>, epoch: u64, observation: NativeRuntimeObservation },
    Admit { epoch: u64, observation: NativeRuntimeObservation },
    Wait,
}

impl Drop for QueryAdmissionTicketContextGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        self.ticket.release_epoch_context(self.task_key, self.epoch, false);
    }
}

impl Drop for QueryAdmissionPendingContextGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        if let NativeRuntimeObservation::Local(owner) = &self.observation {
            owner.record_terminal_outcome(QueryAdmissionTicket::drop_outcome(RuntimeTerminalOutcome::Cancelled));
        }
        self.ticket.release_epoch_context(self.task_key, self.epoch, true);
    }
}

impl QueryAdmissionExecutionRoundGuard {
    fn record_terminal_outcome(&mut self, outcome: RuntimeTerminalOutcome) {
        self.terminal_outcome_recorded = true;
        let observation = self.ticket.state.lock().ok().and_then(|mut state| {
            let round = state.rounds.get_mut(&self.task_key).filter(|round| round.epoch == self.epoch)?;
            round.terminal_outcome = Some(match round.terminal_outcome.take() {
                None | Some(RuntimeTerminalOutcome::Succeeded) => outcome.clone(),
                Some(existing) => existing,
            });
            Some(round.observation.clone())
        });
        if let Some(NativeRuntimeObservation::Local(owner)) = observation {
            owner.record_terminal_outcome(outcome);
        }
    }

    fn record_datafusion_error(&mut self, error: &DataFusionError) {
        self.ticket.record_attempt_datafusion_failure(error);
        let outcome = match error.find_nexus_error() {
            Some(error) if error.code() == ErrorCode::Cancelled => RuntimeTerminalOutcome::Cancelled,
            Some(_) | None => RuntimeTerminalOutcome::failed(
                &nexus_query_engine::observability::AttemptFailureSummary::datafusion_fallback(),
            ),
        };
        self.record_terminal_outcome(outcome);
    }
}

impl Drop for QueryAdmissionExecutionRoundGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        if !self.terminal_outcome_recorded {
            self.record_terminal_outcome(QueryAdmissionTicket::drop_outcome(RuntimeTerminalOutcome::Dropped));
        }
        self.released = true;
        self.ticket.release_execution_round(self.task_key, self.epoch);
    }
}

impl TicketAdmitResetGuard {
    fn reset(&mut self) {
        if !self.disarmed {
            self.disarmed = true;
            self.ticket.reset_admitting(self.task_key);
        }
    }
}

impl Drop for TicketAdmitResetGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            self.ticket.finish_observation(
                &self.observation,
                QueryAdmissionTicket::drop_outcome(RuntimeTerminalOutcome::Cancelled),
            );
        }
        self.reset();
    }
}

/// Transparent root wrapper that scopes a plan ticket to one DataFusion stream execution.
#[derive(Debug)]
pub(crate) struct NativeAdmissionExec {
    input: Arc<dyn ExecutionPlan>,
    ticket: Arc<QueryAdmissionTicket>,
}

impl NativeAdmissionExec {
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, ticket: Arc<QueryAdmissionTicket>) -> Self {
        Self { input, ticket }
    }

    pub(crate) fn ticket(&self) -> &Arc<QueryAdmissionTicket> {
        &self.ticket
    }
}

pub(crate) fn wrap_plan_with_query_admission_ticket(
    plan: Arc<dyn ExecutionPlan>,
    ticket: Arc<QueryAdmissionTicket>,
) -> Arc<dyn ExecutionPlan> {
    if plan_is_query_admission_wrapped(plan.as_ref()) {
        return plan;
    }
    Arc::new(NativeAdmissionExec::new(plan, ticket))
}

pub(crate) fn plan_is_query_admission_wrapped(plan: &dyn ExecutionPlan) -> bool {
    (plan as &dyn Any).is::<NativeAdmissionExec>()
}

pub(crate) fn query_admission_wrapped_input(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    (plan as &dyn Any).downcast_ref::<NativeAdmissionExec>().map(|wrapper| &wrapper.input)
}

impl DisplayAs for NativeAdmissionExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.input.fmt_as(t, f)
    }
}

impl ExecutionPlan for NativeAdmissionExec {
    fn name(&self) -> &str {
        self.input.name()
    }

    fn downcast_delegate(&self) -> Option<&dyn ExecutionPlan> {
        Some(self.input.as_ref())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        // The input is a downcast delegate flattened out of `children()`, so
        // its expressions are only reachable through this forward.
        self.input.apply_expressions(f)
    }

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        self.input.input_distribution_requirements()
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        self.input.required_input_ordering()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.input.maintains_input_order()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        self.input.benefits_from_input_partitioning()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.input.children()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let input = Arc::clone(&self.input)
            .replace_children(children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))?;
        Ok(Arc::new(Self::new(input, Arc::clone(&self.ticket))))
    }

    fn reset_state(self: Arc<Self>) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let input = Arc::clone(&self.input).reset_state()?;
        Ok(Arc::new(Self::new(input, Arc::clone(&self.ticket))))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        let task_ticket = query_admission_ticket_from_task_context(&context).or_else(|| {
            context
                .session_config()
                .options()
                .extensions
                .get::<NexusAttemptExtension>()
                .and_then(NexusAttemptExtension::bound_ticket)
        });
        if task_ticket.is_none() && self.ticket.is_attempt_bound() {
            return Err(DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_exec.attempt_required",
                    "attempt-bound native admission requires an invocation task context",
                )
                .with_fact("reason", "attempt_execution_required"),
            ));
        }
        let ticket = task_ticket.unwrap_or_else(|| Arc::clone(&self.ticket));
        let context = task_context_with_query_admission_ticket(context, Arc::clone(&ticket));
        let mut round_guard = ticket.begin_execution_round(Arc::clone(&context)).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.native_exec.admission_round",
                "failed to begin backend query-service admission execution round",
                error,
            ))
        })?;
        let stream = match self.input.execute(partition, context) {
            Ok(stream) => stream,
            Err(error) => {
                round_guard.record_datafusion_error(&error);
                return Err(error);
            },
        };
        Ok(Box::pin(NativeAdmissionRoundStream::new(stream, round_guard)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.input.metrics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<datafusion::common::Statistics>> {
        StatisticsContext::new().compute(self.input.as_ref(), &StatisticsArgs::new().with_partition(partition))
    }
}

struct NativeAdmissionRoundStream {
    inner: SendableRecordBatchStream,
    round_guard: Option<QueryAdmissionExecutionRoundGuard>,
}

impl NativeAdmissionRoundStream {
    fn new(inner: SendableRecordBatchStream, round_guard: QueryAdmissionExecutionRoundGuard) -> Self {
        Self { inner, round_guard: Some(round_guard) }
    }
}

impl Unpin for NativeAdmissionRoundStream {}

impl Stream for NativeAdmissionRoundStream {
    type Item = DataFusionResult<arrow::record_batch::RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                if let Some(round_guard) = &mut this.round_guard {
                    round_guard.record_terminal_outcome(RuntimeTerminalOutcome::Succeeded);
                }
                Poll::Ready(None)
            },
            Poll::Ready(Some(Err(error))) => {
                if let Some(round_guard) = &mut this.round_guard {
                    round_guard.record_datafusion_error(&error);
                }
                Poll::Ready(Some(Err(error)))
            },
            other => other,
        }
    }
}

impl RecordBatchStream for NativeAdmissionRoundStream {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;
