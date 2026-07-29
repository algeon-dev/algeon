//! Plan-scoped query-service admission tickets for backend-managed execution.

#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
#[cfg(test)]
use nexus_query_engine::{
    error::{QueryError, QueryErrorCode},
    observability::{
        AdmissionLifecycleState, AdmissionRequestId, RuntimeAdmissionObservation, RuntimeObservationHandle,
        RuntimeObservationPayload,
    },
    runtime::{QueryActivityLease, QueryServiceAdmissionSnapshot, QueryServiceConfig, QueryServiceTenantConfig},
};
use nexus_query_engine::{
    observability::{
        RuntimeObservationConfig, RuntimeObservationOwner, RuntimeQueryId, RuntimeTerminalOutcome, RuntimeTraceId,
    },
    runtime::{
        ExecutionCancellation, GpuBackendSession, QueryAdmissionRequest, QueryGrantBand, QueryGrantRequest,
        QueryHandle, QueryPlacementHint, QueryQos, TenantId, TenantQueryContext,
    },
};
use tokio::sync::Notify;

#[cfg(test)]
use crate::native_exec::metrics::NativeMetricsCapture;
use crate::{
    error::{DataFusionErrorExt, Error, ErrorCode},
    native_exec::metrics::{NativeExecutionMetricsCapture, NativeRuntimeObservation},
};

/// Backend-owned service admission source copied into optimizer configs.
#[derive(Debug, Clone)]
pub(crate) struct QueryAdmissionSource {
    capability: QueryAdmissionCapability,
    tenant_id: TenantId,
    qos: QueryQos,
    max_device_total_memory_bytes: u64,
}

#[derive(Debug, Clone)]
enum QueryAdmissionCapability {
    Backend(GpuBackendSession),
    #[cfg(test)]
    Test(TestAdmissionController),
}

impl QueryAdmissionSource {
    pub(crate) fn new(session: GpuBackendSession, tenant_id: TenantId, max_device_total_memory_bytes: u64) -> Self {
        Self {
            capability: QueryAdmissionCapability::Backend(session),
            tenant_id,
            qos: QueryQos::default(),
            max_device_total_memory_bytes,
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn backend_matches(&self, session: &GpuBackendSession) -> bool {
        match &self.capability {
            QueryAdmissionCapability::Backend(backend) => backend.same_backend(session),
            #[cfg(test)]
            QueryAdmissionCapability::Test(_) => false,
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    pub(crate) fn requested_grant_band(
        &self,
        request: QueryGrantRequest,
    ) -> nexus_query_engine::error::Result<QueryGrantBand> {
        match &self.capability {
            QueryAdmissionCapability::Backend(session) => session.requested_grant_band(&self.tenant_id, request),
            #[cfg(test)]
            QueryAdmissionCapability::Test(controller) => controller.requested_grant_band(&self.tenant_id, request),
        }
    }

    pub(crate) fn max_device_total_memory_bytes(&self) -> u64 {
        self.max_device_total_memory_bytes
    }
}

/// CUDA-free admission ledger used only by root-crate state-machine tests.
///
/// It deliberately cannot create an engine [`QueryHandle`] or execution
/// context. Keeping it in the adapter test build preserves pure admission
/// coverage without reopening a production resource-construction seam.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct TestAdmissionController {
    inner: Arc<TestAdmissionControllerInner>,
}

#[cfg(test)]
#[derive(Debug)]
struct TestAdmissionControllerInner {
    config: QueryServiceConfig,
    tenants: HashMap<TenantId, TestTenantGrantBand>,
    state: Mutex<TestAdmissionState>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct TestTenantGrantBand {
    min_bytes: u64,
    want_bytes: u64,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct TestAdmissionState {
    active_queries: usize,
    active_grant_bytes: u64,
    next_admission_sequence: u64,
}

#[cfg(test)]
impl TestAdmissionController {
    pub(crate) fn new(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
    ) -> nexus_query_engine::error::Result<Self> {
        let capacity_bytes = config.device_capacity_bytes.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "datafusion_nexus.native_exec.test_admission_controller",
                "test admission controller requires an explicit device capacity",
            )
            .with_fact("field", "device_capacity_bytes")
            .with_fact("reason", "missing")
        })?;
        if capacity_bytes == 0 || config.max_concurrent_queries == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "datafusion_nexus.native_exec.test_admission_controller",
                "test admission controller requires positive capacity and concurrency",
            )
            .with_fact("reason", "zero_bound"));
        }
        let mut by_tenant = HashMap::with_capacity(tenants.len());
        for tenant in tenants {
            if tenant.tenant_id.as_str().trim().is_empty()
                || tenant.min_device_budget_bytes == 0
                || tenant.device_budget_bytes < tenant.min_device_budget_bytes
                || tenant.device_budget_bytes > capacity_bytes
            {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "datafusion_nexus.native_exec.test_admission_controller",
                    "invalid test tenant admission grant band",
                )
                .with_fact("tenant_id", tenant.tenant_id.as_str())
                .with_fact("reason", "invalid_tenant_grant_band"));
            }
            let tenant_id = tenant.tenant_id;
            if by_tenant
                .insert(
                    tenant_id.clone(),
                    TestTenantGrantBand {
                        min_bytes: tenant.min_device_budget_bytes,
                        want_bytes: tenant.device_budget_bytes,
                    },
                )
                .is_some()
            {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "datafusion_nexus.native_exec.test_admission_controller",
                    "duplicate test admission tenant",
                )
                .with_fact("tenant_id", tenant_id.as_str())
                .with_fact("reason", "duplicate_tenant"));
            }
        }
        Ok(Self {
            inner: Arc::new(TestAdmissionControllerInner {
                config,
                tenants: by_tenant,
                state: Mutex::new(TestAdmissionState::default()),
            }),
        })
    }

    pub(crate) fn source(&self, tenant_id: TenantId) -> QueryAdmissionSource {
        QueryAdmissionSource {
            capability: QueryAdmissionCapability::Test(self.clone()),
            tenant_id,
            qos: QueryQos::default(),
            max_device_total_memory_bytes: self
                .inner
                .config
                .device_capacity_bytes
                .expect("validated test controller has an explicit device capacity"),
        }
    }

    pub(crate) fn admission_snapshot(&self) -> QueryServiceAdmissionSnapshot {
        let state = self.inner.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        QueryServiceAdmissionSnapshot {
            active_queries: state.active_queries,
            queued_queries: 0,
            queued_by_tenant: Vec::new(),
            next_tenant_id: None,
        }
    }

    pub(crate) fn requested_grant_band(
        &self,
        tenant_id: &TenantId,
        request: QueryGrantRequest,
    ) -> nexus_query_engine::error::Result<QueryGrantBand> {
        let configured = self.inner.tenants.get(tenant_id).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "datafusion_nexus.native_exec.test_admission_controller",
                "tenant is not configured for test admission",
            )
            .with_fact("tenant_id", tenant_id.as_str())
            .with_fact("reason", "unknown_tenant")
        })?;
        let min_bytes = request.requested_grant_min_bytes.unwrap_or(configured.min_bytes);
        let want_bytes = request.requested_grant_bytes.unwrap_or(configured.want_bytes);
        if min_bytes < configured.min_bytes || want_bytes < min_bytes || want_bytes > configured.want_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "datafusion_nexus.native_exec.test_admission_controller",
                "requested test admission grant is outside the configured tenant band",
            )
            .with_fact("tenant_id", tenant_id.as_str())
            .with_fact("reason", "requested_grant_out_of_range"));
        }
        Ok(QueryGrantBand { min_bytes, want_bytes })
    }

    fn admit(
        &self,
        tenant_id: &TenantId,
        request: QueryAdmissionRequest,
        observation: RuntimeObservationHandle,
        activity: Option<QueryActivityLease>,
    ) -> nexus_query_engine::error::Result<TestAdmissionHandle> {
        let grant = self.requested_grant_band(tenant_id, request.grant_request)?;
        let capacity_bytes =
            self.inner.config.device_capacity_bytes.expect("validated test controller has an explicit device capacity");
        let (admission_sequence, admitted_grant_bytes) = {
            let mut state = self.inner.state.lock().map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "datafusion_nexus.native_exec.test_admission_controller",
                    "test admission state mutex was poisoned",
                )
            })?;
            if state.active_queries >= self.inner.config.max_concurrent_queries {
                return Err(QueryError::new(
                    QueryErrorCode::ServiceOverloaded,
                    "datafusion_nexus.native_exec.test_admission_controller",
                    "test admission controller has no available slot",
                )
                .with_fact("reason", "capacity_exhausted"));
            }
            let remaining_slots =
                self.inner.config.max_concurrent_queries.saturating_sub(state.active_queries.saturating_add(1));
            let reserved_bytes = match self.inner.config.slot_guarantee {
                nexus_query_engine::runtime::SlotGuarantee::GuaranteedMinimum => self
                    .inner
                    .tenants
                    .values()
                    .map(|tenant| tenant.min_bytes)
                    .max()
                    .unwrap_or(0)
                    .saturating_mul(remaining_slots as u64),
                nexus_query_engine::runtime::SlotGuarantee::Optimistic => 0,
            };
            let available_bytes =
                capacity_bytes.saturating_sub(state.active_grant_bytes).saturating_sub(reserved_bytes);
            let admitted_grant_bytes = grant.want_bytes.min(available_bytes);
            if admitted_grant_bytes < grant.min_bytes {
                return Err(QueryError::new(
                    QueryErrorCode::ServiceOverloaded,
                    "datafusion_nexus.native_exec.test_admission_controller",
                    "test admission controller has insufficient grant capacity",
                )
                .with_fact("reason", "grant_capacity_exhausted"));
            }
            state.next_admission_sequence = state.next_admission_sequence.saturating_add(1);
            let admission_sequence = state.next_admission_sequence;
            state.active_queries = state.active_queries.saturating_add(1);
            state.active_grant_bytes = state.active_grant_bytes.saturating_add(admitted_grant_bytes);
            (admission_sequence, admitted_grant_bytes)
        };
        record_test_admission_lifecycle(
            &observation,
            admission_sequence,
            tenant_id,
            AdmissionLifecycleState::Admitted,
            Some(self.inner.config.device_ordinal),
            Some(admitted_grant_bytes),
        );
        Ok(TestAdmissionHandle {
            controller: self.clone(),
            tenant_id: tenant_id.clone(),
            admission_sequence,
            device_ordinal: self.inner.config.device_ordinal,
            grant_bytes: admitted_grant_bytes,
            observation,
            activity: Mutex::new(activity),
            released: AtomicBool::new(false),
        })
    }

    fn admit_for_test(&self, tenant_id: &TenantId) -> nexus_query_engine::error::Result<TicketQueryHandle> {
        let observation =
            RuntimeObservationOwner::new("test-admission-controller", None, RuntimeObservationConfig::disabled())
                .handle();
        self.admit(tenant_id, QueryAdmissionRequest::new(), observation, None).map(TicketQueryHandle::Test)
    }

    pub(crate) fn admitted_grant_bytes(&self, tenant_id: &TenantId) -> nexus_query_engine::error::Result<u64> {
        let observation =
            RuntimeObservationOwner::new("test-admission-projection", None, RuntimeObservationConfig::disabled())
                .handle();
        let handle = self.admit(tenant_id, QueryAdmissionRequest::new(), observation, None)?;
        let grant_bytes = handle.grant_bytes;
        handle.finish();
        Ok(grant_bytes)
    }

    pub(crate) fn admit_for_projection(
        &self,
        tenant_id: &TenantId,
    ) -> nexus_query_engine::error::Result<TestAdmissionPermit> {
        let observation =
            RuntimeObservationOwner::new("test-admission-projection", None, RuntimeObservationConfig::disabled())
                .handle();
        self.admit(tenant_id, QueryAdmissionRequest::new(), observation, None)
            .map(|handle| TestAdmissionPermit { handle })
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct TestAdmissionPermit {
    handle: TestAdmissionHandle,
}

#[cfg(test)]
impl TestAdmissionPermit {
    pub(crate) fn device_ordinal(&self) -> usize {
        self.handle.device_ordinal
    }

    pub(crate) fn device_budget_bytes(&self) -> u64 {
        self.handle.grant_bytes
    }
}

#[cfg(test)]
#[derive(Debug)]
struct TestAdmissionHandle {
    controller: TestAdmissionController,
    tenant_id: TenantId,
    admission_sequence: u64,
    device_ordinal: usize,
    grant_bytes: u64,
    observation: RuntimeObservationHandle,
    activity: Mutex<Option<QueryActivityLease>>,
    released: AtomicBool,
}

#[cfg(test)]
impl TestAdmissionHandle {
    fn acquire_context(&self) -> nexus_query_engine::error::Result<TenantQueryContext> {
        Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "datafusion_nexus.native_exec.test_admission_controller",
            "CUDA-free test admission cannot create a runtime execution context",
        )
        .with_fact("reason", "test_controller_has_no_runtime_context"))
    }

    fn finish(&self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        self.activity.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        let mut state = self.controller.inner.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active_queries = state.active_queries.saturating_sub(1);
        state.active_grant_bytes = state.active_grant_bytes.saturating_sub(self.grant_bytes);
        drop(state);
        record_test_admission_lifecycle(
            &self.observation,
            self.admission_sequence,
            &self.tenant_id,
            AdmissionLifecycleState::Released,
            Some(self.device_ordinal),
            Some(self.grant_bytes),
        );
    }
}

#[cfg(test)]
impl Drop for TestAdmissionHandle {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
fn record_test_admission_lifecycle(
    observation: &RuntimeObservationHandle,
    admission_sequence: u64,
    tenant_id: &TenantId,
    lifecycle: AdmissionLifecycleState,
    device_ordinal: Option<usize>,
    grant_bytes: Option<u64>,
) {
    observation.record(|| {
        RuntimeObservationPayload::Admission(Box::new(RuntimeAdmissionObservation {
            admission_request_id: AdmissionRequestId::new(admission_sequence),
            tenant_id: tenant_id.as_str().to_string(),
            lifecycle,
            queue_depth: 0,
            device_ordinal: device_ordinal.and_then(|ordinal| u32::try_from(ordinal).ok()),
            grant_bytes,
        }))
    });
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
    pub(crate) fn from_query_handle(handle: &QueryHandle) -> Self {
        Self {
            admission_sequence: handle.admission_sequence(),
            admission_wait: handle.admission_wait(),
            device_ordinal: handle.device_ordinal(),
            query_grant_bytes: handle.device_budget_bytes(),
        }
    }

    fn from_handle(handle: &TicketQueryHandle) -> Self {
        Self {
            admission_sequence: handle.admission_sequence(),
            admission_wait: handle.admission_wait(),
            device_ordinal: handle.device_ordinal(),
            query_grant_bytes: handle.device_budget_bytes(),
        }
    }
}

#[derive(Debug)]
enum TicketQueryHandle {
    Backend(Arc<QueryHandle>),
    #[cfg(test)]
    Test(TestAdmissionHandle),
}

impl TicketQueryHandle {
    fn admission_sequence(&self) -> u64 {
        match self {
            Self::Backend(handle) => handle.admission_sequence(),
            #[cfg(test)]
            Self::Test(handle) => handle.admission_sequence,
        }
    }

    fn admission_wait(&self) -> Duration {
        match self {
            Self::Backend(handle) => handle.admission_wait(),
            #[cfg(test)]
            Self::Test(_) => Duration::ZERO,
        }
    }

    fn device_ordinal(&self) -> usize {
        match self {
            Self::Backend(handle) => handle.device_ordinal(),
            #[cfg(test)]
            Self::Test(handle) => handle.device_ordinal,
        }
    }

    fn device_budget_bytes(&self) -> u64 {
        match self {
            Self::Backend(handle) => handle.device_budget_bytes(),
            #[cfg(test)]
            Self::Test(handle) => handle.grant_bytes,
        }
    }

    async fn acquire_context_with_cancellation_async(
        &self,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::error::Result<TenantQueryContext> {
        match self {
            Self::Backend(handle) => handle.acquire_context_with_cancellation_async(cancellation).await,
            #[cfg(test)]
            Self::Test(handle) => {
                drop(cancellation);
                handle.acquire_context()
            },
        }
    }

    fn finish(&self) {
        match self {
            Self::Backend(handle) => handle.finish(),
            #[cfg(test)]
            Self::Test(handle) => handle.finish(),
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
        let observation_activity =
            self.observation_capture.as_ref().and_then(NativeExecutionMetricsCapture::new_observation_activity);
        #[cfg(test)]
        let (request, test_activity) = match (&self.source.capability, observation_activity) {
            (QueryAdmissionCapability::Test(_), Some(Ok(activity))) => (request, Some(activity)),
            (QueryAdmissionCapability::Test(_), Some(Err(error))) => {
                record_test_admission_lifecycle(
                    &observation.handle(),
                    0,
                    &self.source.tenant_id,
                    AdmissionLifecycleState::Rejected {
                        reason: error.fact_value("reason").unwrap_or_else(|| error.code().as_str()).to_string(),
                    },
                    None,
                    None,
                );
                self.finish_observation_for_error(&observation, &error);
                reset_guard.reset();
                return Err(error);
            },
            (QueryAdmissionCapability::Test(_), None) => (request, None),
            (QueryAdmissionCapability::Backend(_), Some(activity)) => {
                (request.with_runtime_observation_activity_result(activity), None)
            },
            (QueryAdmissionCapability::Backend(_), None) => (request, None),
        };
        #[cfg(not(test))]
        let request = match observation_activity {
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
        let handle_result = match &self.source.capability {
            QueryAdmissionCapability::Backend(session) => session
                .admit_async_with_request(self.source.tenant_id.clone(), request)
                .await
                .map(|handle| TicketQueryHandle::Backend(Arc::new(handle))),
            #[cfg(test)]
            QueryAdmissionCapability::Test(controller) => controller
                .admit(&self.source.tenant_id, request, observation.handle(), test_activity)
                .map(TicketQueryHandle::Test),
        };
        let handle = match handle_result {
            Ok(handle) => Arc::new(handle),
            Err(error) => {
                #[cfg(test)]
                if matches!(&self.source.capability, QueryAdmissionCapability::Test(_)) {
                    record_test_admission_lifecycle(
                        &observation.handle(),
                        0,
                        &self.source.tenant_id,
                        AdmissionLifecycleState::Rejected {
                            reason: error.fact_value("reason").unwrap_or_else(|| error.code().as_str()).to_string(),
                        },
                        None,
                        None,
                    );
                }
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
        handle: Arc<TicketQueryHandle>,
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
            RuntimeTerminalOutcome::Failed {
                error_code: error.code().to_string(),
                error_status: error.status().as_str().to_string(),
            }
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
        let outcome = match error.find_nexus_error() {
            Some(error) if error.code() == ErrorCode::Cancelled => RuntimeTerminalOutcome::Cancelled,
            Some(error) => RuntimeTerminalOutcome::Failed {
                error_code: error.code().as_str().to_string(),
                error_status: error.status().as_str().to_string(),
            },
            None => RuntimeTerminalOutcome::Failed {
                error_code: "datafusion_error".to_string(),
                error_status: "permanent".to_string(),
            },
        };
        self.record_terminal_outcome(outcome);
    }

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
                if let Some(round_guard) = &mut this.round_guard {
                    round_guard.record_terminal_outcome(RuntimeTerminalOutcome::Succeeded);
                }
                this.round_guard = None;
                Poll::Ready(None)
            },
            Poll::Ready(Some(Err(error))) => {
                if let Some(round_guard) = &mut this.round_guard {
                    round_guard.record_datafusion_error(&error);
                }
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
#[path = "admission_tests.rs"]
mod tests;
