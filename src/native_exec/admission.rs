//! Plan-scoped query-service admission tickets for backend-managed execution.

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_expr::{Distribution, OrderingRequirements},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream, SendableRecordBatchStream,
        metrics::MetricsSet,
    },
};
use futures::Stream;
#[cfg(test)]
use nexus_query_engine::metrics::QueryId;
use nexus_query_engine::{
    observability::{
        RuntimeObservationConfig, RuntimeObservationOwner, RuntimeQueryId, RuntimeTerminalOutcome, RuntimeTraceId,
    },
    runtime::{
        ExecutionCancellation, QueryAdmissionRequest, QueryGrantBand, QueryGrantRequest, QueryHandle,
        QueryPlacementHint, QueryQos, QueryService, TenantId, TenantQueryContext,
    },
};
use tokio::sync::Notify;

#[cfg(test)]
use crate::native_exec::metrics::NativeMetricsCapture;
use crate::{
    error::Error,
    native_exec::metrics::{NativeExecutionMetricsCapture, NativeRuntimeObservation},
};

/// Backend-owned service admission source copied into optimizer configs.
#[derive(Debug, Clone)]
pub(crate) struct QueryAdmissionSource {
    service: QueryService,
    tenant_id: TenantId,
    qos: QueryQos,
}

impl QueryAdmissionSource {
    pub(crate) fn new(service: QueryService, tenant_id: TenantId) -> Self {
        Self { service, tenant_id, qos: QueryQos::default() }
    }

    #[cfg(feature = "server")]
    pub(crate) fn service_matches(&self, service: &QueryService) -> bool {
        self.service.same_service(service)
    }

    #[cfg(feature = "server")]
    pub(crate) fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    pub(crate) fn requested_grant_band(
        &self,
        request: QueryGrantRequest,
    ) -> nexus_query_engine::error::Result<QueryGrantBand> {
        self.service.tenant_requested_grant_band_with_grant_request(&self.tenant_id, request)
    }

    pub(crate) fn max_device_total_memory_bytes(&self) -> u64 {
        self.service.max_device_total_memory_bytes()
    }
}

/// Immutable facts assigned when query-service admission succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueryAdmissionFacts {
    pub(crate) admission_sequence: u64,
    pub(crate) admission_wait: Duration,
    pub(crate) device_ordinal: usize,
    pub(crate) query_grant_bytes: u64,
}

impl QueryAdmissionFacts {
    pub(crate) fn from_handle(handle: &QueryHandle) -> Self {
        Self {
            admission_sequence: handle.admission_sequence(),
            admission_wait: handle.admission_wait(),
            device_ordinal: handle.device_ordinal(),
            query_grant_bytes: handle.device_budget_bytes(),
        }
    }
}

/// Lazy admission ticket shared by every native node emitted in one optimize pass.
#[derive(Debug)]
pub(crate) struct QueryAdmissionTicket {
    #[cfg(test)]
    id: u64,
    source: QueryAdmissionSource,
    requested_grant_min_bytes: Option<u64>,
    requested_grant_bytes: Option<u64>,
    #[cfg(feature = "cugraph")]
    warm_graph_requested_grant_bytes: Option<u64>,
    placement_hint: QueryPlacementHint,
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
    handle: Option<Arc<QueryHandle>>,
    active_contexts: usize,
    active_streams: usize,
    epoch: u64,
    admission_facts_recorded: bool,
    observation: NativeRuntimeObservation,
}

/// Context acquired through a [`QueryAdmissionTicket`].
#[derive(Debug)]
pub(crate) struct TicketTenantQueryContext {
    tenant_context: TenantQueryContext,
    task_context: Arc<TaskContext>,
    admission_facts: QueryAdmissionFacts,
    record_admission_facts: bool,
    _guard: QueryAdmissionTicketContextGuard,
}

impl TicketTenantQueryContext {
    pub(crate) fn context(&self) -> &TenantQueryContext {
        &self.tenant_context
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
        requested_grant_min_bytes: Option<u64>,
        requested_grant_bytes: Option<u64>,
    ) -> Arc<Self> {
        Self::new_with_placement_hint_and_observation(
            source,
            requested_grant_min_bytes,
            requested_grant_bytes,
            QueryPlacementHint::default(),
            RuntimeObservationOwner::new(QueryId::mint(), None, RuntimeObservationConfig::native_execution()),
            None,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_with_observation(
        source: QueryAdmissionSource,
        requested_grant_min_bytes: Option<u64>,
        requested_grant_bytes: Option<u64>,
        observation_owner: RuntimeObservationOwner,
    ) -> Arc<Self> {
        Self::new_with_placement_hint_and_observation(
            source,
            requested_grant_min_bytes,
            requested_grant_bytes,
            QueryPlacementHint::default(),
            observation_owner,
            None,
        )
    }

    pub(crate) fn new_with_observation_and_capture(
        source: QueryAdmissionSource,
        requested_grant_min_bytes: Option<u64>,
        requested_grant_bytes: Option<u64>,
        observation: impl Into<NativeRuntimeObservation>,
        observation_capture: Option<NativeExecutionMetricsCapture>,
    ) -> Arc<Self> {
        Self::new_with_placement_hint_and_observation(
            source,
            requested_grant_min_bytes,
            requested_grant_bytes,
            QueryPlacementHint::default(),
            observation,
            observation_capture,
        )
    }

    fn new_with_placement_hint_and_observation(
        source: QueryAdmissionSource,
        requested_grant_min_bytes: Option<u64>,
        requested_grant_bytes: Option<u64>,
        placement_hint: QueryPlacementHint,
        observation: impl Into<NativeRuntimeObservation>,
        observation_capture: Option<NativeExecutionMetricsCapture>,
    ) -> Arc<Self> {
        Self::new_with_placement_hint_and_warm_graph_grant_and_observation_and_capture(
            source,
            requested_grant_min_bytes,
            requested_grant_bytes,
            placement_hint,
            #[cfg(feature = "cugraph")]
            None,
            observation,
            observation_capture,
        )
    }

    pub(crate) fn new_with_placement_hint_and_warm_graph_grant_and_observation_and_capture(
        source: QueryAdmissionSource,
        requested_grant_min_bytes: Option<u64>,
        requested_grant_bytes: Option<u64>,
        placement_hint: QueryPlacementHint,
        #[cfg(feature = "cugraph")] warm_graph_requested_grant_bytes: Option<u64>,
        observation: impl Into<NativeRuntimeObservation>,
        observation_capture: Option<NativeExecutionMetricsCapture>,
    ) -> Arc<Self> {
        #[cfg(test)]
        static NEXT_TICKET_ID: AtomicU64 = AtomicU64::new(1);
        let observation = observation.into();
        let observation_handle = observation.handle();
        let observation_query_id = observation_handle.query_id().cloned();
        let observation_trace_id = observation_handle.trace_id().cloned();
        let observation_config = observation_handle.config();
        Arc::new(Self {
            #[cfg(test)]
            id: NEXT_TICKET_ID.fetch_add(1, Ordering::Relaxed),
            source,
            requested_grant_min_bytes,
            requested_grant_bytes,
            #[cfg(feature = "cugraph")]
            warm_graph_requested_grant_bytes,
            placement_hint,
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

    #[cfg(test)]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    #[cfg(test)]
    pub(crate) fn requested_grant_bytes(&self) -> Option<u64> {
        self.requested_grant_bytes
    }

    #[cfg(all(test, feature = "cugraph"))]
    pub(crate) fn warm_graph_requested_grant_bytes(&self) -> Option<u64> {
        self.warm_graph_requested_grant_bytes
    }

    #[cfg(all(test, feature = "cugraph"))]
    pub(crate) fn placement_hint(&self) -> &QueryPlacementHint {
        &self.placement_hint
    }

    #[cfg(test)]
    pub(crate) fn requested_grant_min_bytes(&self) -> Option<u64> {
        self.requested_grant_min_bytes
    }

    #[cfg(test)]
    pub(crate) fn observation_owner(&self) -> RuntimeObservationOwner {
        match self.new_observation() {
            NativeRuntimeObservation::Local(owner) => owner,
            NativeRuntimeObservation::Coordinated(_) => panic!("coordinated tickets do not expose seal authority"),
        }
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
                    observation,
                },
            );
            epoch
        };
        drop(state);
        self.finish_orphaned_rounds(stale_rounds);
        Ok(QueryAdmissionExecutionRoundGuard { ticket: Arc::clone(self), task_key, epoch, released: false })
    }

    pub(crate) async fn acquire_context(
        self: &Arc<Self>,
        task_context: Arc<TaskContext>,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::error::Result<TicketTenantQueryContext> {
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
    ) -> nexus_query_engine::error::Result<TicketTenantQueryContext> {
        let mut reset_guard = TicketAdmitResetGuard {
            ticket: Arc::clone(self),
            task_key,
            disarmed: false,
            observation: observation.clone(),
        };
        let request = QueryAdmissionRequest::new()
            .with_qos(self.source.qos)
            .with_grant_request(QueryGrantRequest {
                requested_grant_min_bytes: self.requested_grant_min_bytes,
                requested_grant_bytes: self.requested_grant_bytes,
            })
            .with_placement_hint(self.placement_hint.clone());
        let request = match &observation {
            NativeRuntimeObservation::Local(owner) => request.with_runtime_observation_owner(owner.clone()),
            NativeRuntimeObservation::Coordinated(handle) => request.with_runtime_observation_handle(handle.clone()),
        };
        let request =
            match self.observation_capture.as_ref().and_then(NativeExecutionMetricsCapture::new_observation_activity) {
                Some(activity) => request.with_runtime_observation_activity_result(activity),
                None => request,
            };
        #[cfg(feature = "cugraph")]
        let request = match self.warm_graph_requested_grant_bytes {
            Some(requested_grant_bytes) => request.with_warm_graph_grant_request(QueryGrantRequest {
                requested_grant_min_bytes: self.requested_grant_min_bytes,
                requested_grant_bytes: Some(requested_grant_bytes),
            }),
            None => request,
        };
        let handle = match self.source.service.admit_async_with_request(self.source.tenant_id.clone(), request).await {
            Ok(handle) => Arc::new(handle),
            Err(error) => {
                self.finish_observation_for_error(&observation, &error);
                reset_guard.reset();
                return Err(error);
            },
        };

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
        handle: Arc<QueryHandle>,
        task_key: usize,
        epoch: u64,
        task_context: Arc<TaskContext>,
        cancellation: ExecutionCancellation,
        observation: NativeRuntimeObservation,
    ) -> nexus_query_engine::error::Result<TicketTenantQueryContext> {
        let mut pending_guard = QueryAdmissionPendingContextGuard {
            ticket: Arc::clone(self),
            task_key,
            epoch,
            released: false,
            observation: observation.clone(),
        };
        match handle.acquire_context_with_cancellation_async(cancellation).await {
            Ok(tenant_context) => {
                pending_guard.released = true;
                Ok(TicketTenantQueryContext {
                    tenant_context,
                    task_context,
                    admission_facts: QueryAdmissionFacts::from_handle(handle.as_ref()),
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
            None => self.finish_observation(&round.observation, Self::drop_outcome(RuntimeTerminalOutcome::Dropped)),
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
        capture: Option<&NativeExecutionMetricsCapture>,
    ) {
        if let NativeRuntimeObservation::Local(owner) = observation
            && let Ok(Some(snapshot)) = catch_unwind(AssertUnwindSafe(|| owner.finish_query(outcome)))
            && let Some(capture) = capture
        {
            capture.record_runtime_observation(Arc::new(snapshot));
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
        if error.code() == nexus_query_engine::error::QueryErrorCode::Cancelled {
            RuntimeTerminalOutcome::Cancelled
        } else {
            RuntimeTerminalOutcome::Failed { error_code: error.code().to_string() }
        }
    }

    fn record_observation_outcome_for_error(
        &self,
        observation: &NativeRuntimeObservation,
        error: &nexus_query_engine::error::QueryError,
    ) {
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
    Reuse { handle: Arc<QueryHandle>, epoch: u64, observation: NativeRuntimeObservation },
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
    #[cfg(test)]
    fn epoch(&self) -> u64 {
        self.epoch
    }

    #[cfg(test)]
    fn observation_owner(&self) -> RuntimeObservationOwner {
        self.ticket
            .state
            .lock()
            .unwrap()
            .rounds
            .get(&self.task_key)
            .filter(|round| round.epoch == self.epoch)
            .expect("execution round owner remains live")
            .observation
            .clone()
            .into_local_owner()
    }
}

impl Drop for QueryAdmissionExecutionRoundGuard {
    fn drop(&mut self) {
        if self.released {
            return;
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

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.input.required_input_distribution()
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
        let input = Arc::clone(&self.input).with_new_children(children)?;
        Ok(Arc::new(Self::new(input, Arc::clone(&self.ticket))))
    }

    fn reset_state(self: Arc<Self>) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let input = Arc::clone(&self.input).reset_state()?;
        Ok(Arc::new(Self::new(input, Arc::clone(&self.ticket))))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        let ticket = query_admission_ticket_from_task_context(&context).unwrap_or_else(|| Arc::clone(&self.ticket));
        let context = task_context_with_query_admission_ticket(context, Arc::clone(&ticket));
        let round_guard = ticket.begin_execution_round(Arc::clone(&context)).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.native_exec.admission_round",
                "failed to begin backend query-service admission execution round",
                error,
            ))
        })?;
        let stream = self.input.execute(partition, context)?;
        Ok(Box::pin(NativeAdmissionRoundStream::new(stream, round_guard)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.input.metrics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<datafusion::common::Statistics>> {
        self.input.partition_statistics(partition)
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
                this.round_guard = None;
                Poll::Ready(None)
            },
            Poll::Ready(Some(Err(error))) => {
                this.round_guard = None;
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
mod tests {
    use std::time::Duration;

    use arrow::datatypes::Schema;
    use datafusion::{
        execution::context::TaskContext,
        physical_plan::{ExecutionPlan, empty::EmptyExec},
    };
    use futures::StreamExt;
    use nexus_query_engine::runtime::{
        DeploymentIoCapability, ExecutionCancellation, GdsDriverProbeStatus, KvikioCompatModeStatus,
        NvidiaFsProbeStatus, QueryServiceConfig, QueryServiceTenantConfig,
    };

    use super::*;

    fn nongds_deployment() -> DeploymentIoCapability {
        DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        )
    }

    fn admission_only_service(tenant_id: &TenantId) -> QueryService {
        QueryService::new_admission_only(
            QueryServiceConfig {
                max_concurrent_queries: 1,
                device_capacity_bytes: Some(1024),
                max_admission_queue_len: 1,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::new(tenant_id.clone(), 1024)],
            nongds_deployment(),
        )
        .unwrap()
    }

    fn admission_only_ticket() -> (Arc<QueryAdmissionTicket>, QueryService, TenantId) {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let ticket =
            QueryAdmissionTicket::new(QueryAdmissionSource::new(service.clone(), tenant_id.clone()), None, None);
        (ticket, service, tenant_id)
    }

    fn insert_test_round(
        ticket: &Arc<QueryAdmissionTicket>,
        task_context: &Arc<TaskContext>,
        active_contexts: usize,
        epoch: u64,
    ) {
        let task_key = QueryAdmissionTicket::task_key(task_context);
        ticket.state.lock().unwrap().rounds.insert(
            task_key,
            TicketRound {
                _task_context: Arc::clone(task_context),
                handle: None,
                active_contexts,
                active_streams: 0,
                epoch,
                admission_facts_recorded: false,
                observation: ticket.new_observation(),
            },
        );
    }

    #[test]
    fn admission_ticket_carries_requested_grant_bytes_for_admission() {
        let tenant_id = TenantId::new("tenant");
        let service = QueryService::new_admission_only(
            QueryServiceConfig {
                max_concurrent_queries: 1,
                device_capacity_bytes: Some(1024),
                max_admission_queue_len: 1,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), 256, 1024)],
            nongds_deployment(),
        )
        .unwrap();
        let ticket =
            QueryAdmissionTicket::new(QueryAdmissionSource::new(service.clone(), tenant_id), Some(384), Some(512));

        assert_eq!(ticket.requested_grant_min_bytes(), Some(384));
        assert_eq!(ticket.requested_grant_bytes(), Some(512));
        assert_eq!(
            service
                .tenant_requested_budget_with_grant_request(
                    &ticket.source.tenant_id,
                    QueryGrantRequest {
                        requested_grant_min_bytes: ticket.requested_grant_min_bytes(),
                        requested_grant_bytes: ticket.requested_grant_bytes(),
                    },
                )
                .unwrap(),
            512
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn admission_ticket_owner_precedes_enqueue_and_seals_rejection() {
        let configured_tenant = TenantId::new("configured");
        let service = admission_only_service(&configured_tenant);
        let owner = RuntimeObservationOwner::new(
            "ticket-rejected-query",
            None,
            RuntimeObservationConfig::bounded(32, 32 * 1024),
        );
        let ticket = QueryAdmissionTicket::new_with_observation(
            QueryAdmissionSource::new(service, TenantId::new("unknown")),
            None,
            None,
            owner.clone(),
        );

        let error = ticket
            .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
            .await
            .expect_err("unknown tenant rejects before admission");

        assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::InvalidPlan);
        let snapshot = owner.sealed_snapshot().expect("admission rejection seals the ticket owner");
        assert_eq!(snapshot.manifest().query_id.as_str(), "ticket-rejected-query");
        assert!(matches!(
            snapshot.footer().terminal,
            Some(RuntimeTerminalOutcome::Failed { ref error_code })
                if error_code == nexus_query_engine::error::QueryErrorCode::InvalidPlan.as_str()
        ));
        assert!(snapshot.events().iter().any(|event| matches!(
            &event.payload,
            nexus_query_engine::observability::RuntimeObservationPayload::Admission(observation)
                if matches!(
                    observation.lifecycle,
                    nexus_query_engine::observability::AdmissionLifecycleState::Rejected { .. }
                )
        )));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn coordinated_ticket_reuses_one_observation_session_across_three_legal_epochs() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let root =
            RuntimeObservationOwner::new("coordinated-ticket", None, RuntimeObservationConfig::bounded(64, 64 * 1024));
        let capture = NativeMetricsCapture::coordinated("coordinated-ticket", root.handle())
            .with_observation_activity_factory(Arc::new(|| {
                Some(nexus_query_engine::runtime::QueryActivityLease::new(()))
            }))
            .new_execution();
        assert_eq!(capture.retention_stats().capacity, 0);
        let ticket = QueryAdmissionTicket::new_with_observation_and_capture(
            QueryAdmissionSource::new(service, tenant_id),
            None,
            None,
            NativeRuntimeObservation::Coordinated(root.handle()),
            Some(capture.clone()),
        );
        let task_context = Arc::new(TaskContext::default());

        for _ in 0..3 {
            ticket
                .acquire_context(Arc::clone(&task_context), ExecutionCancellation::new())
                .await
                .expect_err("admission-only service has no runtime execution context");
            assert!(root.sealed_snapshot().is_none(), "admission epochs cannot seal the coordinator root");
        }
        drop(ticket);
        assert!(root.sealed_snapshot().is_none(), "ticket drop cannot seal the coordinator root");

        let snapshot = root.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap();
        assert_eq!(snapshot.manifest().query_id.as_str(), "coordinated-ticket");
        assert!(capture.runtime_observations().is_empty());
        assert_eq!(
            snapshot
                .events()
                .iter()
                .filter(|event| matches!(
                    &event.payload,
                    nexus_query_engine::observability::RuntimeObservationPayload::Admission(observation)
                        if matches!(
                            observation.lifecycle,
                            nexus_query_engine::observability::AdmissionLifecycleState::Admitted
                        )
                ))
                .count(),
            3
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn coordinated_activity_rejection_is_a_structured_admission_failure() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let root = RuntimeObservationOwner::new(
            "coordinated-activity-rejected",
            None,
            RuntimeObservationConfig::bounded(64, 64 * 1024),
        );
        let capture = NativeMetricsCapture::coordinated("coordinated-activity-rejected", root.handle())
            .with_observation_activity_factory(Arc::new(|| None))
            .new_execution();
        let ticket = QueryAdmissionTicket::new_with_observation_and_capture(
            QueryAdmissionSource::new(service.clone(), tenant_id),
            None,
            None,
            NativeRuntimeObservation::Coordinated(root.handle()),
            Some(capture),
        );

        let error = ticket
            .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
            .await
            .expect_err("coordinator activity rejection must reject native admission");

        assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::Cancelled);
        assert_eq!(error.operation(), "datafusion_nexus.native_exec.admission.observation_activity");
        assert_eq!(error.fact_value("reason"), Some("query_coordinator_activity_rejected"));
        assert_eq!(error.fact_value("admission_stage"), Some("observation_activity"));
        let snapshot = service.admission_snapshot().unwrap();
        assert_eq!(snapshot.active_queries, 0);
        assert_eq!(snapshot.queued_queries, 0);
        assert!(root.sealed_snapshot().is_none(), "coordinated rejection cannot seal the coordinator root");
        assert!(root.live_snapshot().unwrap().events().iter().any(|event| matches!(
            &event.payload,
            nexus_query_engine::observability::RuntimeObservationPayload::Admission(observation)
                if matches!(
                    &observation.lifecycle,
                    nexus_query_engine::observability::AdmissionLifecycleState::Rejected { reason }
                        if reason == "query_coordinator_activity_rejected"
                )
        )));
    }

    #[test]
    fn standalone_ticket_drop_transfers_one_sealed_snapshot_to_its_local_collector() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let capture = NativeMetricsCapture::without_sink("standalone-ticket").new_execution();
        let observation = capture.new_runtime_observation("standalone-ticket", None);
        let ticket = QueryAdmissionTicket::new_with_observation_and_capture(
            QueryAdmissionSource::new(service, tenant_id),
            None,
            None,
            observation,
            Some(capture.clone()),
        );

        drop(ticket);

        let retained = capture.runtime_observations();
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].manifest().query_id.as_str(), "standalone-ticket");
        assert_eq!(retained[0].footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
        assert_eq!(capture.retention_stats().recorded_total, 1);
    }

    #[test]
    fn dropping_ticket_before_first_round_seals_initial_owner_once_and_isolates_observer_panics() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let owner = RuntimeObservationOwner::new(
            "ticket-dropped-before-round",
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        );
        let delivered = Arc::new(AtomicU64::new(0));
        let delivered_for_observer = Arc::clone(&delivered);
        owner.register_seal_observer(move |_| {
            delivered_for_observer.fetch_add(1, Ordering::Relaxed);
        });
        owner.register_seal_observer(|_| panic!("ticket drop must isolate observer failure"));
        let ticket = QueryAdmissionTicket::new_with_observation(
            QueryAdmissionSource::new(service, tenant_id),
            None,
            None,
            owner.clone(),
        );

        let dropped = catch_unwind(AssertUnwindSafe(|| drop(ticket)));

        assert!(dropped.is_ok());
        assert_eq!(delivered.load(Ordering::Relaxed), 1);
        let snapshot = owner.sealed_snapshot().expect("last ticket owner drop must seal the initial observation");
        assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
        assert_eq!(delivered.load(Ordering::Relaxed), 1, "ticket terminalization must remain exactly once");
    }

    #[test]
    fn unwinding_unadmitted_round_guard_seals_panicked_exactly_once() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let owner =
            RuntimeObservationOwner::new("unadmitted-round-panic", None, RuntimeObservationConfig::bounded(16, 4096));
        let delivered = Arc::new(AtomicU64::new(0));
        let delivered_for_observer = Arc::clone(&delivered);
        owner.register_seal_observer(move |_| {
            delivered_for_observer.fetch_add(1, Ordering::Relaxed);
        });
        let ticket = QueryAdmissionTicket::new_with_observation(
            QueryAdmissionSource::new(service, tenant_id),
            None,
            None,
            owner.clone(),
        );
        let guard = ticket.begin_execution_round(Arc::new(TaskContext::default())).unwrap();

        let unwind = catch_unwind(AssertUnwindSafe(move || {
            let _guard = guard;
            panic!("unwind unadmitted round");
        }));

        assert!(unwind.is_err());
        assert_eq!(owner.sealed_snapshot().unwrap().footer().terminal, Some(RuntimeTerminalOutcome::Panicked));
        assert_eq!(delivered.load(Ordering::Relaxed), 1);
        assert!(ticket.state.lock().is_ok(), "round unwind must not poison ticket state");
        drop(ticket);
        assert_eq!(delivered.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unwinding_admit_reset_guard_seals_panicked_and_resets_without_poisoning() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let owner =
            RuntimeObservationOwner::new("admit-reset-panic", None, RuntimeObservationConfig::bounded(16, 4096));
        let delivered = Arc::new(AtomicU64::new(0));
        let delivered_for_observer = Arc::clone(&delivered);
        owner.register_seal_observer(move |_| {
            delivered_for_observer.fetch_add(1, Ordering::Relaxed);
        });
        let ticket = QueryAdmissionTicket::new_with_observation(
            QueryAdmissionSource::new(service, tenant_id),
            None,
            None,
            owner.clone(),
        );
        let task_key = 41;
        ticket.state.lock().unwrap().admitting.insert(task_key);
        let guard = TicketAdmitResetGuard {
            ticket: Arc::clone(&ticket),
            task_key,
            disarmed: false,
            observation: NativeRuntimeObservation::Local(owner.clone()),
        };

        let unwind = catch_unwind(AssertUnwindSafe(move || {
            let _guard = guard;
            panic!("unwind admit reset");
        }));

        assert!(unwind.is_err());
        assert_eq!(owner.sealed_snapshot().unwrap().footer().terminal, Some(RuntimeTerminalOutcome::Panicked));
        assert_eq!(delivered.load(Ordering::Relaxed), 1);
        assert!(!ticket.state.lock().unwrap().admitting.contains(&task_key));
        drop(ticket);
        assert_eq!(delivered.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn cancellation_before_first_context_acquire_seals_initial_owner() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let owner = RuntimeObservationOwner::new(
            "ticket-cancelled-before-context",
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        );
        let ticket = QueryAdmissionTicket::new_with_observation(
            QueryAdmissionSource::new(service, tenant_id),
            None,
            None,
            owner.clone(),
        );
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        let round = ticket.begin_execution_round(task_context).unwrap();
        let round_owner = round.observation_owner();
        ticket.state.lock().unwrap().rounds.get_mut(&task_key).unwrap().active_contexts = 1;
        let pending = QueryAdmissionPendingContextGuard {
            ticket: Arc::clone(&ticket),
            task_key,
            epoch: round.epoch(),
            released: false,
            observation: round_owner.into(),
        };

        drop(pending);
        drop(round);

        let snapshot = owner.sealed_snapshot().expect("cancelled first acquire must seal the initial owner");
        assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Cancelled));
    }

    #[test]
    fn final_context_release_removes_round_without_waiting_for_task_context_drop() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        insert_test_round(&ticket, &task_context, 1, 7);

        ticket.release_epoch_context(task_key, 7, false);

        assert!(
            !ticket.state.lock().unwrap().rounds.contains_key(&task_key),
            "finished execution rounds must not linger on a live TaskContext"
        );
    }

    #[test]
    fn overlapping_context_releases_keep_round_until_last_context_finishes() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        insert_test_round(&ticket, &task_context, 2, 11);

        ticket.release_epoch_context(task_key, 11, false);
        assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_contexts), Some(1));

        ticket.release_epoch_context(task_key, 11, false);
        assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
    }

    #[test]
    fn cancelled_pending_context_acquire_releases_ticket_epoch_and_admission() {
        let (ticket, service, tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        let handle = service.admit_query(tenant_id).unwrap();
        ticket.state.lock().unwrap().rounds.insert(
            task_key,
            TicketRound {
                _task_context: Arc::clone(&task_context),
                handle: Some(Arc::new(handle)),
                active_contexts: 1,
                active_streams: 0,
                epoch: 13,
                admission_facts_recorded: false,
                observation: ticket.new_observation(),
            },
        );

        let pending = QueryAdmissionPendingContextGuard {
            ticket: Arc::clone(&ticket),
            task_key,
            epoch: 13,
            released: false,
            observation: ticket.new_observation(),
        };
        drop(pending);

        assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    }

    #[test]
    fn execution_round_guard_keeps_round_until_stream_finishes() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        let round = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
        ticket.state.lock().unwrap().rounds.get_mut(&task_key).unwrap().active_contexts = 1;

        ticket.release_epoch_context(task_key, round.epoch(), false);
        assert_eq!(
            ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_streams),
            Some(1),
            "context release must not finish the execution round while the root stream is alive"
        );

        drop(round);
        assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
    }

    #[test]
    fn active_execution_round_retains_task_context_until_release() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        let weak_task_context = Arc::downgrade(&task_context);
        let round = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();

        drop(task_context);

        assert!(
            weak_task_context.upgrade().is_some(),
            "active ticket rounds must retain the TaskContext whose pointer keys the round"
        );
        assert!(ticket.state.lock().unwrap().rounds.contains_key(&task_key));

        drop(round);

        assert!(weak_task_context.upgrade().is_none(), "finished ticket rounds should release their TaskContext");
    }

    #[test]
    fn admission_wrapper_prefers_existing_ambient_ticket() {
        let (root_ticket, _root_service, _root_tenant_id) = admission_only_ticket();
        let (child_ticket, _child_service, _child_tenant_id) = admission_only_ticket();
        let task_context =
            task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&root_ticket));
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        let child = Arc::new(EmptyExec::new(Arc::new(Schema::empty()))) as Arc<dyn ExecutionPlan>;
        let wrapped = NativeAdmissionExec::new(child, Arc::clone(&child_ticket));

        let stream = wrapped.execute(0, Arc::clone(&task_context)).unwrap();

        assert!(root_ticket.state.lock().unwrap().rounds.contains_key(&task_key));
        assert!(
            child_ticket.state.lock().unwrap().rounds.is_empty(),
            "nested admission wrapper must not mint a second execution round when an ambient root ticket exists"
        );

        drop(stream);
        assert!(!root_ticket.state.lock().unwrap().rounds.contains_key(&task_key));
    }

    #[test]
    fn admission_facts_marker_records_once_per_ticket_epoch() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        insert_test_round(&ticket, &task_context, 1, 3);

        assert!(ticket.mark_admission_facts_recorded(task_key, 3));
        assert!(
            !ticket.mark_admission_facts_recorded(task_key, 3),
            "the same execution epoch must not record query-level admission facts twice"
        );
        assert!(
            !ticket.mark_admission_facts_recorded(task_key, 4),
            "stale or future epochs must not consume the marker for the active round"
        );
    }

    #[test]
    fn execution_round_reexecution_with_same_task_context_uses_fresh_epoch() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());

        let first = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
        assert_eq!(first.epoch(), 1);
        drop(first);

        let second = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
        assert_eq!(second.epoch(), 2);
    }

    #[test]
    fn sequential_execution_epochs_use_distinct_owners_and_plan_invocations() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let initial_owner = RuntimeObservationOwner::new(
            "stable-external-query",
            None,
            RuntimeObservationConfig::bounded(64, 64 * 1024),
        );
        let ticket = QueryAdmissionTicket::new_with_observation(
            QueryAdmissionSource::new(service, tenant_id),
            None,
            None,
            initial_owner.clone(),
        );
        let task_context = Arc::new(TaskContext::default());

        let first_round = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
        let first_owner = first_round.observation_owner();
        let first_plan = nexus_query_engine::observability::RuntimePlanExecutionLifecycle::from_owner(
            first_owner.clone(),
            nexus_query_engine::observability::PlanExecutionId::new("epoch-plan-1"),
        );
        first_plan.finish(RuntimeTerminalOutcome::Succeeded);
        let first_snapshot = first_owner.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap();
        drop(first_round);

        let second_round = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
        let second_owner = second_round.observation_owner();
        assert!(second_owner.live_snapshot().is_some(), "the next epoch must not reuse the sealed first owner");
        let second_plan = nexus_query_engine::observability::RuntimePlanExecutionLifecycle::from_owner(
            second_owner.clone(),
            nexus_query_engine::observability::PlanExecutionId::new("epoch-plan-2"),
        );
        second_plan.finish(RuntimeTerminalOutcome::Succeeded);
        let second_snapshot = second_owner.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap();
        drop(second_round);

        assert_eq!(first_snapshot.manifest().query_id, second_snapshot.manifest().query_id);
        assert!(
            first_snapshot
                .events()
                .iter()
                .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "epoch-plan-1") })
        );
        assert!(
            second_snapshot
                .events()
                .iter()
                .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "epoch-plan-2") })
        );
        assert!(
            !second_snapshot
                .events()
                .iter()
                .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "epoch-plan-1") })
        );
    }

    #[test]
    fn concurrent_execution_epochs_have_distinct_live_owners_with_shared_external_identity() {
        let tenant_id = TenantId::new("tenant");
        let service = admission_only_service(&tenant_id);
        let ticket = QueryAdmissionTicket::new_with_observation(
            QueryAdmissionSource::new(service, tenant_id),
            None,
            None,
            RuntimeObservationOwner::new(
                "concurrent-external-query",
                None,
                RuntimeObservationConfig::bounded(64, 64 * 1024),
            ),
        );
        let first_round = ticket.begin_execution_round(Arc::new(TaskContext::default())).unwrap();
        let second_round = ticket.begin_execution_round(Arc::new(TaskContext::default())).unwrap();
        let first_owner = first_round.observation_owner();
        let second_owner = second_round.observation_owner();

        let first_plan = nexus_query_engine::observability::RuntimePlanExecutionLifecycle::from_owner(
            first_owner.clone(),
            nexus_query_engine::observability::PlanExecutionId::new("concurrent-plan-1"),
        );
        let second_plan = nexus_query_engine::observability::RuntimePlanExecutionLifecycle::from_owner(
            second_owner.clone(),
            nexus_query_engine::observability::PlanExecutionId::new("concurrent-plan-2"),
        );
        let first_live = first_owner.live_snapshot().unwrap();
        let second_live = second_owner.live_snapshot().unwrap();
        assert_eq!(first_live.manifest().query_id, second_live.manifest().query_id);
        assert!(
            first_live
                .events()
                .iter()
                .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "concurrent-plan-1") })
        );
        assert!(
            second_live
                .events()
                .iter()
                .any(|event| { event.plan_execution_id.as_ref().is_some_and(|id| id.as_str() == "concurrent-plan-2") })
        );
        assert!(first_owner.sealed_snapshot().is_none());
        assert!(second_owner.sealed_snapshot().is_none());

        first_plan.finish(RuntimeTerminalOutcome::Succeeded);
        second_plan.finish(RuntimeTerminalOutcome::Succeeded);
        let _ = first_owner.finish_query(RuntimeTerminalOutcome::Succeeded);
        let _ = second_owner.finish_query(RuntimeTerminalOutcome::Succeeded);
        drop(first_round);
        drop(second_round);
    }

    #[test]
    fn admission_wrapper_repeated_execute_with_same_context_shares_one_round() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let input = Arc::new(EmptyExec::new(Arc::new(Schema::empty()))) as Arc<dyn ExecutionPlan>;
        let wrapped = NativeAdmissionExec::new(input, Arc::clone(&ticket));
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);

        let first = wrapped.execute(0, Arc::clone(&task_context)).unwrap();
        let second = wrapped.execute(0, Arc::clone(&task_context)).unwrap();

        {
            let state = ticket.state.lock().unwrap();
            assert_eq!(state.rounds.len(), 1, "one caller TaskContext must own one execution round");
            let round = state.rounds.get(&task_key).expect("round must be keyed by the caller TaskContext");
            assert_eq!(round.active_streams, 2);
        }

        drop(first);
        assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_streams), Some(1));

        drop(second);
        assert!(!ticket.state.lock().unwrap().rounds.contains_key(&task_key));
    }

    #[tokio::test]
    async fn admission_wrapper_releases_round_when_stream_finishes() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let input = Arc::new(EmptyExec::new(Arc::new(Schema::empty()))) as Arc<dyn ExecutionPlan>;
        let wrapped = Arc::new(NativeAdmissionExec::new(input, Arc::clone(&ticket))) as Arc<dyn ExecutionPlan>;
        assert!(plan_is_query_admission_wrapped(wrapped.as_ref()));
        assert!(wrapped.downcast_ref::<EmptyExec>().is_some());
        let rewrapped = wrap_plan_with_query_admission_ticket(Arc::clone(&wrapped), Arc::clone(&ticket));
        assert!(Arc::ptr_eq(&wrapped, &rewrapped));
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);

        let mut stream = wrapped.execute(0, Arc::clone(&task_context)).unwrap();
        assert_eq!(wrapped.name(), "EmptyExec");
        assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.active_streams), Some(1));

        assert!(stream.next().await.is_none());
        assert!(
            !ticket.state.lock().unwrap().rounds.contains_key(&task_key),
            "terminal stream poll should release the root execution-round guard"
        );
    }

    #[test]
    fn stale_epoch_release_does_not_remove_newer_round() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        insert_test_round(&ticket, &task_context, 1, 2);

        ticket.release_epoch_context(task_key, 1, false);

        assert_eq!(ticket.state.lock().unwrap().rounds.get(&task_key).map(|round| round.epoch), Some(2));
    }

    #[tokio::test]
    async fn active_same_task_round_reuses_existing_handle_instead_of_re_admitting() {
        let (ticket, service, tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());
        let task_key = QueryAdmissionTicket::task_key(&task_context);
        let handle = service.admit_async(tenant_id.clone()).await.unwrap();
        ticket.state.lock().unwrap().rounds.insert(
            task_key,
            TicketRound {
                _task_context: Arc::clone(&task_context),
                handle: Some(Arc::new(handle)),
                active_contexts: 1,
                active_streams: 0,
                epoch: 1,
                admission_facts_recorded: false,
                observation: ticket.new_observation(),
            },
        );
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

        let acquire = tokio::time::timeout(
            Duration::from_millis(100),
            ticket.acquire_context(Arc::clone(&task_context), ExecutionCancellation::new()),
        )
        .await
        .expect("overlapping acquisition should reuse the active round instead of waiting for another admission");
        assert!(acquire.is_err(), "admission-only services cannot acquire runtime contexts");
        assert_eq!(
            service.admission_snapshot().unwrap().active_queries,
            1,
            "failed overlapping context acquisition must not drop the original active round"
        );

        ticket.release_epoch_context(task_key, 1, false);
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    }

    #[tokio::test]
    async fn failed_context_acquire_releases_ticket_epoch_for_reexecution() {
        let (ticket, service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());

        let first = ticket.acquire_context(Arc::clone(&task_context), ExecutionCancellation::new()).await;
        assert!(first.is_err(), "admission-only services cannot acquire runtime contexts");
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);

        let second = tokio::time::timeout(
            Duration::from_millis(100),
            ticket.acquire_context(Arc::clone(&task_context), ExecutionCancellation::new()),
        )
        .await
        .expect("second execution round should not wait on the failed first round");
        assert!(second.is_err(), "admission-only services cannot acquire runtime contexts");
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    }
}
