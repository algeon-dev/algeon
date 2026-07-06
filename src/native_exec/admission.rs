//! Plan-scoped query-service admission tickets for backend-managed execution.

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
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
use nexus_query_engine::{ExecutionCancellation, QueryHandle, QueryQos, QueryService, TenantId, TenantQueryContext};
use tokio::sync::Notify;

use crate::error::Error;

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
}

/// Lazy admission ticket shared by every native node emitted in one optimize pass.
#[derive(Debug)]
pub(crate) struct QueryAdmissionTicket {
    #[cfg(test)]
    id: u64,
    source: QueryAdmissionSource,
    state: Mutex<TicketState>,
    notify: Arc<Notify>,
}

#[derive(Debug, Default)]
struct TicketState {
    rounds: HashMap<usize, TicketRound>,
    admitting: HashSet<usize>,
    next_epoch: u64,
}

#[derive(Debug)]
struct TicketRound {
    handle: Option<Arc<QueryHandle>>,
    active_contexts: usize,
    active_streams: usize,
    epoch: u64,
}

/// Context acquired through a [`QueryAdmissionTicket`].
#[derive(Debug)]
pub(crate) struct TicketTenantQueryContext {
    tenant_context: TenantQueryContext,
    _guard: QueryAdmissionTicketContextGuard,
}

impl TicketTenantQueryContext {
    pub(crate) fn context(&self) -> &TenantQueryContext {
        &self.tenant_context
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

struct TicketAdmitResetGuard {
    ticket: Arc<QueryAdmissionTicket>,
    task_key: usize,
    disarmed: bool,
}

impl QueryAdmissionTicket {
    pub(crate) fn new(source: QueryAdmissionSource) -> Arc<Self> {
        #[cfg(test)]
        static NEXT_TICKET_ID: AtomicU64 = AtomicU64::new(1);
        Arc::new(Self {
            #[cfg(test)]
            id: NEXT_TICKET_ID.fetch_add(1, Ordering::Relaxed),
            source,
            state: Mutex::new(TicketState::default()),
            notify: Arc::new(Notify::new()),
        })
    }

    #[cfg(test)]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn begin_execution_round(
        self: &Arc<Self>,
        task_context: Arc<TaskContext>,
    ) -> nexus_query_engine::Result<QueryAdmissionExecutionRoundGuard> {
        let task_key = Self::task_key(&task_context);
        let mut state = self.lock_state()?;
        Self::remove_stale_rounds_locked(&mut state);
        let epoch = if let Some(round) = state.rounds.get_mut(&task_key) {
            round.active_streams = round.active_streams.saturating_add(1);
            round.epoch
        } else {
            state.next_epoch = state.next_epoch.saturating_add(1);
            let epoch = state.next_epoch;
            state.rounds.insert(task_key, TicketRound { handle: None, active_contexts: 0, active_streams: 1, epoch });
            epoch
        };
        Ok(QueryAdmissionExecutionRoundGuard { ticket: Arc::clone(self), task_key, epoch, released: false })
    }

    pub(crate) async fn acquire_context(
        self: &Arc<Self>,
        task_context: Arc<TaskContext>,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::Result<TicketTenantQueryContext> {
        let task_key = Self::task_key(&task_context);
        loop {
            let notify = Arc::clone(&self.notify);
            let notified = notify.notified_owned();
            tokio::pin!(notified);
            let action = {
                let mut state = self.lock_state()?;
                Self::remove_stale_rounds_locked(&mut state);
                if let Some(round) = state.rounds.get_mut(&task_key)
                    && let Some(handle) = round.handle.as_ref().cloned()
                {
                    let epoch = round.epoch;
                    round.active_contexts = round.active_contexts.saturating_add(1);
                    TicketAcquireAction::Reuse { handle, epoch }
                } else {
                    notified.as_mut().enable();
                    if state.admitting.contains(&task_key) {
                        TicketAcquireAction::Wait
                    } else {
                        let epoch = state.rounds.get(&task_key).map(|round| round.epoch);
                        state.admitting.insert(task_key);
                        TicketAcquireAction::Admit { epoch }
                    }
                }
            };

            match action {
                TicketAcquireAction::Reuse { handle, epoch } => {
                    return self.acquire_from_handle(handle, task_key, epoch, cancellation);
                },
                TicketAcquireAction::Wait => {
                    notified.await;
                    continue;
                },
                TicketAcquireAction::Admit { epoch } => {
                    return self.admit_and_acquire_context(task_key, epoch, cancellation).await;
                },
            }
        }
    }

    async fn admit_and_acquire_context(
        self: &Arc<Self>,
        task_key: usize,
        epoch: Option<u64>,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::Result<TicketTenantQueryContext> {
        let mut reset_guard = TicketAdmitResetGuard { ticket: Arc::clone(self), task_key, disarmed: false };
        let handle =
            match self.source.service.admit_async_with_qos(self.source.tenant_id.clone(), self.source.qos).await {
                Ok(handle) => Arc::new(handle),
                Err(error) => {
                    reset_guard.reset();
                    return Err(error);
                },
            };

        let epoch = {
            let mut state = self.lock_state()?;
            state.admitting.remove(&task_key);
            let epoch = if let Some(epoch) = epoch {
                epoch
            } else {
                state.next_epoch = state.next_epoch.saturating_add(1);
                let epoch = state.next_epoch;
                state
                    .rounds
                    .insert(task_key, TicketRound { handle: None, active_contexts: 0, active_streams: 0, epoch });
                epoch
            };
            let Some(round) = state.rounds.get_mut(&task_key) else {
                reset_guard.disarmed = true;
                self.notify.notify_waiters();
                drop(state);
                drop(handle);
                return Err(nexus_query_engine::QueryError::new(
                    nexus_query_engine::QueryErrorCode::Internal,
                    "datafusion_nexus.native_exec.admission_ticket",
                    "query admission ticket execution round disappeared during admission",
                ));
            };
            if round.epoch != epoch {
                reset_guard.disarmed = true;
                self.notify.notify_waiters();
                drop(state);
                drop(handle);
                return Err(nexus_query_engine::QueryError::new(
                    nexus_query_engine::QueryErrorCode::Internal,
                    "datafusion_nexus.native_exec.admission_ticket",
                    "query admission ticket execution round changed during admission",
                ));
            }
            round.handle = Some(Arc::clone(&handle));
            round.active_contexts = round.active_contexts.saturating_add(1);
            reset_guard.disarmed = true;
            self.notify.notify_waiters();
            epoch
        };
        self.acquire_from_handle(handle, task_key, epoch, cancellation)
    }

    fn acquire_from_handle(
        self: &Arc<Self>,
        handle: Arc<QueryHandle>,
        task_key: usize,
        epoch: u64,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::Result<TicketTenantQueryContext> {
        match handle.acquire_context_with_cancellation(cancellation) {
            Ok(tenant_context) => Ok(TicketTenantQueryContext {
                tenant_context,
                _guard: QueryAdmissionTicketContextGuard { ticket: Arc::clone(self), task_key, epoch, released: false },
            }),
            Err(error) => {
                self.release_epoch_context(task_key, epoch, true);
                Err(error)
            },
        }
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
        if round.active_contexts == 0 && (force || round.active_streams == 0) {
            state.rounds.remove(&task_key);
            self.notify.notify_waiters();
        }
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
        if round.active_streams == 0 && round.active_contexts == 0 {
            state.rounds.remove(&task_key);
            self.notify.notify_waiters();
        }
    }

    fn reset_admitting(&self, task_key: usize) {
        let Ok(mut state) = self.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to reset poisoned query admission ticket state");
            return;
        };
        state.admitting.remove(&task_key);
        self.notify.notify_waiters();
    }

    fn lock_state(&self) -> nexus_query_engine::Result<std::sync::MutexGuard<'_, TicketState>> {
        self.state.lock().map_err(|_error| {
            nexus_query_engine::QueryError::new(
                nexus_query_engine::QueryErrorCode::Internal,
                "datafusion_nexus.native_exec.admission_ticket",
                "query admission ticket state mutex was poisoned",
            )
        })
    }

    fn task_key(task_context: &Arc<TaskContext>) -> usize {
        Arc::as_ptr(task_context) as usize
    }

    fn remove_stale_rounds_locked(state: &mut TicketState) {
        state.rounds.retain(|_task_key, round| round.active_contexts > 0 || round.active_streams > 0);
    }
}

enum TicketAcquireAction {
    Reuse { handle: Arc<QueryHandle>, epoch: u64 },
    Admit { epoch: Option<u64> },
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

impl QueryAdmissionExecutionRoundGuard {
    #[cfg(test)]
    fn epoch(&self) -> u64 {
        self.epoch
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
        let round_guard = self.ticket.begin_execution_round(Arc::clone(&context)).map_err(|error| {
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
    use datafusion::{execution::context::TaskContext, physical_plan::empty::EmptyExec};
    use futures::StreamExt;
    use nexus_query_engine::{
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
        let ticket = QueryAdmissionTicket::new(QueryAdmissionSource::new(service.clone(), tenant_id.clone()));
        (ticket, service, tenant_id)
    }

    fn insert_test_round(
        ticket: &Arc<QueryAdmissionTicket>,
        task_context: &Arc<TaskContext>,
        active_contexts: usize,
        epoch: u64,
    ) {
        let task_key = QueryAdmissionTicket::task_key(task_context);
        ticket
            .state
            .lock()
            .unwrap()
            .rounds
            .insert(task_key, TicketRound { handle: None, active_contexts, active_streams: 0, epoch });
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
    fn execution_round_reexecution_with_same_task_context_uses_fresh_epoch() {
        let (ticket, _service, _tenant_id) = admission_only_ticket();
        let task_context = Arc::new(TaskContext::default());

        let first = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
        assert_eq!(first.epoch(), 1);
        drop(first);

        let second = ticket.begin_execution_round(Arc::clone(&task_context)).unwrap();
        assert_eq!(second.epoch(), 2);
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
            TicketRound { handle: Some(Arc::new(handle)), active_contexts: 1, active_streams: 0, epoch: 1 },
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
