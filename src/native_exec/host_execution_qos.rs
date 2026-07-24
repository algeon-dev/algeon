//! Host-side statement concurrency bound.

use std::{
    any::Any,
    collections::{HashMap, HashSet},
    future::Future,
    num::NonZeroUsize,
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
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

use crate::error::{Error, ErrorCode};

#[derive(Debug, Clone)]
pub(crate) struct HostExecutionQosLimiter {
    inner: Arc<HostExecutionQosLimiterInner>,
}

#[derive(Debug)]
struct HostExecutionQosLimiterInner {
    permits: Arc<Semaphore>,
    max_concurrent_statements: NonZeroUsize,
    wait_timeout: Option<Duration>,
    state: Mutex<HostExecutionQosTicketState>,
    notify: Arc<Notify>,
}

impl HostExecutionQosLimiter {
    #[cfg(test)]
    pub(crate) fn new(max_concurrent_statements: NonZeroUsize) -> Self {
        Self::new_with_wait_timeout(max_concurrent_statements, None)
    }

    pub(crate) fn new_with_wait_timeout(
        max_concurrent_statements: NonZeroUsize,
        wait_timeout: Option<Duration>,
    ) -> Self {
        let wait_timeout = wait_timeout.filter(|timeout| !timeout.is_zero());
        Self {
            inner: Arc::new(HostExecutionQosLimiterInner {
                permits: Arc::new(Semaphore::new(max_concurrent_statements.get())),
                max_concurrent_statements,
                wait_timeout,
                state: Mutex::default(),
                notify: Arc::new(Notify::new()),
            }),
        }
    }

    pub(crate) fn max_concurrent_statements(&self) -> NonZeroUsize {
        self.inner.max_concurrent_statements
    }

    pub(crate) fn wait_timeout(&self) -> Option<Duration> {
        self.inner.wait_timeout
    }

    #[cfg(test)]
    pub(crate) fn available_permits(&self) -> usize {
        self.inner.permits.available_permits()
    }
}

#[derive(Debug)]
struct HostExecutionQosTicket {
    limiter: HostExecutionQosLimiter,
}

#[derive(Debug, Default)]
struct HostExecutionQosTicketState {
    rounds: HashMap<usize, HostExecutionQosTicketRound>,
    acquiring: HashSet<usize>,
    next_epoch: u64,
}

#[derive(Debug)]
struct HostExecutionQosTicketRound {
    _task_context: Arc<TaskContext>,
    permit: Option<Arc<OwnedSemaphorePermit>>,
    active_streams: usize,
    epoch: u64,
}

#[derive(Debug)]
struct HostExecutionQosExecutionRoundGuard {
    ticket: Arc<HostExecutionQosTicket>,
    task_key: usize,
    epoch: u64,
    released: bool,
}

struct HostExecutionQosAcquireResetGuard {
    ticket: Arc<HostExecutionQosTicket>,
    task_key: usize,
    disarmed: bool,
}

impl HostExecutionQosTicket {
    fn new(limiter: HostExecutionQosLimiter) -> Arc<Self> {
        Arc::new(Self { limiter })
    }

    fn begin_execution_round(
        self: &Arc<Self>,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<HostExecutionQosExecutionRoundGuard> {
        let task_key = Self::task_key(&task_context);
        let mut state = self.lock_state()?;
        Self::remove_stale_rounds_locked(&mut state);
        let epoch = if let Some(round) = state.rounds.get_mut(&task_key) {
            round.active_streams = round.active_streams.saturating_add(1);
            round.epoch
        } else {
            state.next_epoch = state.next_epoch.saturating_add(1);
            let epoch = state.next_epoch;
            state.rounds.insert(
                task_key,
                HostExecutionQosTicketRound {
                    _task_context: Arc::clone(&task_context),
                    permit: None,
                    active_streams: 1,
                    epoch,
                },
            );
            epoch
        };
        Ok(HostExecutionQosExecutionRoundGuard { ticket: Arc::clone(self), task_key, epoch, released: false })
    }

    async fn acquire_permit_for_epoch(
        self: Arc<Self>,
        task_context: Arc<TaskContext>,
        epoch: u64,
    ) -> DataFusionResult<()> {
        let task_key = Self::task_key(&task_context);
        loop {
            let notify = Arc::clone(&self.limiter.inner.notify);
            let notified = notify.notified_owned();
            tokio::pin!(notified);
            let action = {
                let mut state = self.lock_state()?;
                Self::remove_stale_rounds_locked(&mut state);
                let Some(round) = state.rounds.get(&task_key) else {
                    return Err(host_execution_qos_admission_error(
                        "host execution QoS execution round disappeared before admission",
                    ));
                };
                if round.epoch != epoch {
                    return Err(host_execution_qos_admission_error(
                        "host execution QoS execution round changed before admission",
                    ));
                }
                if round.permit.is_some() {
                    HostExecutionQosAcquireAction::Ready
                } else {
                    notified.as_mut().enable();
                    if state.acquiring.contains(&task_key) {
                        HostExecutionQosAcquireAction::Wait
                    } else {
                        state.acquiring.insert(task_key);
                        HostExecutionQosAcquireAction::Acquire
                    }
                }
            };

            match action {
                HostExecutionQosAcquireAction::Ready => return Ok(()),
                HostExecutionQosAcquireAction::Wait => {
                    notified.await;
                },
                HostExecutionQosAcquireAction::Acquire => {
                    let mut reset_guard =
                        HostExecutionQosAcquireResetGuard { ticket: Arc::clone(&self), task_key, disarmed: false };
                    let acquire = self.limiter.inner.permits.clone().acquire_owned();
                    let permit = match self.limiter.wait_timeout() {
                        Some(wait_timeout) => match tokio::time::timeout(wait_timeout, acquire).await {
                            Ok(result) => result,
                            Err(_elapsed) => {
                                return Err(host_execution_qos_wait_timeout_error(
                                    self.limiter.max_concurrent_statements(),
                                    wait_timeout,
                                ));
                            },
                        },
                        None => acquire.await,
                    }
                    .map_err(|error| {
                        DataFusionError::External(Box::new(
                            Error::new(
                                ErrorCode::ServiceOverloaded,
                                "datafusion_nexus.host_execution_qos.admission",
                                "host execution QoS admission queue was closed",
                            )
                            .with_source(error),
                        ))
                    })?;
                    let mut state = self.lock_state()?;
                    state.acquiring.remove(&task_key);
                    let Some(round) = state.rounds.get_mut(&task_key) else {
                        reset_guard.disarmed = true;
                        self.limiter.inner.notify.notify_waiters();
                        return Err(host_execution_qos_admission_error(
                            "host execution QoS execution round disappeared during admission",
                        ));
                    };
                    if round.epoch != epoch {
                        reset_guard.disarmed = true;
                        self.limiter.inner.notify.notify_waiters();
                        return Err(host_execution_qos_admission_error(
                            "host execution QoS execution round changed during admission",
                        ));
                    }
                    round.permit = Some(Arc::new(permit));
                    reset_guard.disarmed = true;
                    self.limiter.inner.notify.notify_waiters();
                    return Ok(());
                },
            }
        }
    }

    fn release_execution_round(self: &Arc<Self>, task_key: usize, epoch: u64) {
        let Ok(mut state) = self.limiter.inner.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to release poisoned host execution QoS admission state");
            return;
        };
        let Some(round) = state.rounds.get_mut(&task_key) else {
            return;
        };
        if round.epoch != epoch || round.active_streams == 0 {
            return;
        }
        round.active_streams -= 1;
        if round.active_streams == 0 {
            state.rounds.remove(&task_key);
            self.limiter.inner.notify.notify_waiters();
        }
    }

    fn reset_acquiring(&self, task_key: usize) {
        let Ok(mut state) = self.limiter.inner.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to reset poisoned host execution QoS admission state");
            return;
        };
        state.acquiring.remove(&task_key);
        self.limiter.inner.notify.notify_waiters();
    }

    fn lock_state(&self) -> DataFusionResult<std::sync::MutexGuard<'_, HostExecutionQosTicketState>> {
        self.limiter
            .inner
            .state
            .lock()
            .map_err(|_| host_execution_qos_admission_error("host execution QoS admission state mutex was poisoned"))
    }

    fn task_key(task_context: &Arc<TaskContext>) -> usize {
        Arc::as_ptr(task_context) as usize
    }

    fn remove_stale_rounds_locked(state: &mut HostExecutionQosTicketState) {
        state.rounds.retain(|_task_key, round| round.active_streams > 0);
    }
}

enum HostExecutionQosAcquireAction {
    Ready,
    Acquire,
    Wait,
}

impl Drop for HostExecutionQosExecutionRoundGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        self.ticket.release_execution_round(self.task_key, self.epoch);
    }
}

impl Drop for HostExecutionQosAcquireResetGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            self.disarmed = true;
            self.ticket.reset_acquiring(self.task_key);
        }
    }
}

#[derive(Debug)]
pub(crate) struct HostExecutionQosExec {
    input: Arc<dyn ExecutionPlan>,
    ticket: Arc<HostExecutionQosTicket>,
}

impl HostExecutionQosExec {
    fn new(input: Arc<dyn ExecutionPlan>, limiter: HostExecutionQosLimiter) -> Self {
        Self { input, ticket: HostExecutionQosTicket::new(limiter) }
    }
}

pub(crate) fn wrap_plan_with_host_execution_qos(
    plan: Arc<dyn ExecutionPlan>,
    limiter: HostExecutionQosLimiter,
) -> Arc<dyn ExecutionPlan> {
    if plan_is_host_execution_qos_limited(plan.as_ref()) {
        return plan;
    }
    Arc::new(HostExecutionQosExec::new(plan, limiter))
}

pub(crate) fn plan_is_host_execution_qos_limited(plan: &dyn ExecutionPlan) -> bool {
    (plan as &dyn Any).is::<HostExecutionQosExec>()
}

pub(crate) fn host_execution_qos_wrapped_input(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    (plan as &dyn Any).downcast_ref::<HostExecutionQosExec>().map(|wrapper| &wrapper.input)
}

impl DisplayAs for HostExecutionQosExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.input.fmt_as(t, f)
    }
}

impl ExecutionPlan for HostExecutionQosExec {
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
        Ok(Arc::new(Self { input, ticket: Arc::clone(&self.ticket) }))
    }

    fn reset_state(self: Arc<Self>) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let input = Arc::clone(&self.input).reset_state()?;
        Ok(Arc::new(Self { input, ticket: Arc::clone(&self.ticket) }))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        let round_guard = self.ticket.begin_execution_round(Arc::clone(&context))?;
        Ok(Box::pin(HostExecutionQosStream::new(
            Arc::clone(&self.input),
            partition,
            context,
            Arc::clone(&self.ticket),
            round_guard,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.input.metrics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<datafusion::common::Statistics>> {
        self.input.partition_statistics(partition)
    }
}

type BoxAcquireFuture = Pin<Box<dyn Future<Output = DataFusionResult<()>> + Send>>;

struct HostExecutionQosStream {
    schema: SchemaRef,
    state: HostExecutionQosStreamState,
    round_guard: Option<HostExecutionQosExecutionRoundGuard>,
}

enum HostExecutionQosStreamState {
    Waiting { input: Arc<dyn ExecutionPlan>, partition: usize, context: Arc<TaskContext>, acquire: BoxAcquireFuture },
    Running(SendableRecordBatchStream),
    Done,
}

impl HostExecutionQosStream {
    fn new(
        input: Arc<dyn ExecutionPlan>,
        partition: usize,
        context: Arc<TaskContext>,
        ticket: Arc<HostExecutionQosTicket>,
        round_guard: HostExecutionQosExecutionRoundGuard,
    ) -> Self {
        let schema = input.schema();
        let epoch = round_guard.epoch;
        let acquire = Box::pin(ticket.acquire_permit_for_epoch(Arc::clone(&context), epoch));
        Self {
            schema,
            state: HostExecutionQosStreamState::Waiting { input, partition, context, acquire },
            round_guard: Some(round_guard),
        }
    }
}

impl Unpin for HostExecutionQosStream {}

impl Stream for HostExecutionQosStream {
    type Item = DataFusionResult<arrow::record_batch::RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match &mut this.state {
                HostExecutionQosStreamState::Waiting { input, partition, context, acquire } => {
                    match acquire.as_mut().poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Err(error)) => {
                            this.round_guard = None;
                            this.state = HostExecutionQosStreamState::Done;
                            return Poll::Ready(Some(Err(error)));
                        },
                        Poll::Ready(Ok(())) => {
                            let stream = match input.execute(*partition, Arc::clone(context)) {
                                Ok(stream) => stream,
                                Err(error) => {
                                    this.round_guard = None;
                                    this.state = HostExecutionQosStreamState::Done;
                                    return Poll::Ready(Some(Err(error)));
                                },
                            };
                            this.state = HostExecutionQosStreamState::Running(stream);
                        },
                    }
                },
                HostExecutionQosStreamState::Running(stream) => match stream.as_mut().poll_next(cx) {
                    Poll::Ready(None) => {
                        this.round_guard = None;
                        this.state = HostExecutionQosStreamState::Done;
                        return Poll::Ready(None);
                    },
                    Poll::Ready(Some(Err(error))) => {
                        this.round_guard = None;
                        this.state = HostExecutionQosStreamState::Done;
                        return Poll::Ready(Some(Err(error)));
                    },
                    other => return other,
                },
                HostExecutionQosStreamState::Done => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for HostExecutionQosStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

fn host_execution_qos_admission_error(message: &'static str) -> DataFusionError {
    DataFusionError::External(Box::new(Error::new(
        ErrorCode::Internal,
        "datafusion_nexus.host_execution_qos.admission",
        message,
    )))
}

fn host_execution_qos_wait_timeout_error(limit: NonZeroUsize, waited: Duration) -> DataFusionError {
    DataFusionError::External(Box::new(
        Error::new(
            ErrorCode::ServiceOverloaded,
            "datafusion_nexus.host_execution_qos.admission",
            "host execution QoS admission wait timed out",
        )
        .with_fact("reason", "host_execution_qos_wait_timeout")
        .with_fact("limit", limit.get())
        .with_fact("waited_ms", waited.as_millis()),
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        task::{Context, Poll},
        time::Duration,
    };

    use arrow::{datatypes::Schema, record_batch::RecordBatch};
    use datafusion::physical_plan::{collect, empty::EmptyExec};
    use futures::StreamExt;

    use super::{
        super::unbounded_native::{UnboundedNativeExecutionLimiter, wrap_plan_with_unbounded_native_execution_limit},
        *,
    };

    #[derive(Debug)]
    struct PendingExec {
        inner: EmptyExec,
        execute_count: Arc<AtomicUsize>,
    }

    impl PendingExec {
        fn new(schema: SchemaRef, execute_count: Arc<AtomicUsize>) -> Self {
            Self { inner: EmptyExec::new(schema), execute_count }
        }
    }

    impl DisplayAs for PendingExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt_as(t, f)
        }
    }

    impl ExecutionPlan for PendingExec {
        fn name(&self) -> &str {
            "PendingExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn required_input_distribution(&self) -> Vec<Distribution> {
            self.inner.required_input_distribution()
        }

        fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
            self.inner.required_input_ordering()
        }

        fn maintains_input_order(&self) -> Vec<bool> {
            self.inner.maintains_input_order()
        }

        fn benefits_from_input_partitioning(&self) -> Vec<bool> {
            self.inner.benefits_from_input_partitioning()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            if !children.is_empty() {
                return Err(DataFusionError::Internal("PendingExec does not accept children".to_string()));
            }
            Ok(self)
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            self.execute_count.fetch_add(1, Ordering::SeqCst);
            Ok(Box::pin(PendingStream { schema: self.schema() }))
        }

        fn metrics(&self) -> Option<MetricsSet> {
            None
        }

        fn partition_statistics(
            &self,
            partition: Option<usize>,
        ) -> DataFusionResult<Arc<datafusion::common::Statistics>> {
            self.inner.partition_statistics(partition)
        }
    }

    struct PendingStream {
        schema: SchemaRef,
    }

    impl Stream for PendingStream {
        type Item = DataFusionResult<arrow::record_batch::RecordBatch>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl RecordBatchStream for PendingStream {
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
    }

    #[derive(Debug)]
    struct CountingEmptyExec {
        inner: EmptyExec,
        execute_count: Arc<AtomicUsize>,
    }

    impl CountingEmptyExec {
        fn new(schema: SchemaRef, execute_count: Arc<AtomicUsize>) -> Self {
            Self { inner: EmptyExec::new(schema), execute_count }
        }
    }

    impl DisplayAs for CountingEmptyExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt_as(t, f)
        }
    }

    impl ExecutionPlan for CountingEmptyExec {
        fn name(&self) -> &str {
            "CountingEmptyExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn required_input_distribution(&self) -> Vec<Distribution> {
            self.inner.required_input_distribution()
        }

        fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
            self.inner.required_input_ordering()
        }

        fn maintains_input_order(&self) -> Vec<bool> {
            self.inner.maintains_input_order()
        }

        fn benefits_from_input_partitioning(&self) -> Vec<bool> {
            self.inner.benefits_from_input_partitioning()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            if !children.is_empty() {
                return Err(DataFusionError::Internal("CountingEmptyExec does not accept children".to_string()));
            }
            Ok(self)
        }

        fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
            self.execute_count.fetch_add(1, Ordering::SeqCst);
            self.inner.execute(partition, context)
        }

        fn metrics(&self) -> Option<MetricsSet> {
            None
        }

        fn partition_statistics(
            &self,
            partition: Option<usize>,
        ) -> DataFusionResult<Arc<datafusion::common::Statistics>> {
            self.inner.partition_statistics(partition)
        }
    }

    #[derive(Debug)]
    struct NestedCollectExec {
        inner: EmptyExec,
        nested: Arc<dyn ExecutionPlan>,
    }

    impl NestedCollectExec {
        fn new(schema: SchemaRef, nested: Arc<dyn ExecutionPlan>) -> Self {
            Self { inner: EmptyExec::new(schema), nested }
        }
    }

    impl DisplayAs for NestedCollectExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt_as(t, f)
        }
    }

    impl ExecutionPlan for NestedCollectExec {
        fn name(&self) -> &str {
            "NestedCollectExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn required_input_distribution(&self) -> Vec<Distribution> {
            self.inner.required_input_distribution()
        }

        fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
            self.inner.required_input_ordering()
        }

        fn maintains_input_order(&self) -> Vec<bool> {
            self.inner.maintains_input_order()
        }

        fn benefits_from_input_partitioning(&self) -> Vec<bool> {
            self.inner.benefits_from_input_partitioning()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.nested]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            let [nested]: [Arc<dyn ExecutionPlan>; 1] = children
                .try_into()
                .map_err(|_| DataFusionError::Internal("NestedCollectExec expects one child".to_string()))?;
            Ok(Arc::new(Self { inner: EmptyExec::new(self.schema()), nested }))
        }

        fn execute(&self, _partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
            Ok(Box::pin(NestedCollectStream {
                schema: self.schema(),
                collect: Some(Box::pin(collect(Arc::clone(&self.nested), context))),
            }))
        }

        fn metrics(&self) -> Option<MetricsSet> {
            None
        }

        fn partition_statistics(
            &self,
            partition: Option<usize>,
        ) -> DataFusionResult<Arc<datafusion::common::Statistics>> {
            self.inner.partition_statistics(partition)
        }
    }

    type BoxCollectFuture = Pin<Box<dyn Future<Output = DataFusionResult<Vec<RecordBatch>>> + Send>>;

    struct NestedCollectStream {
        schema: SchemaRef,
        collect: Option<BoxCollectFuture>,
    }

    impl Stream for NestedCollectStream {
        type Item = DataFusionResult<RecordBatch>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            let Some(collect) = this.collect.as_mut() else {
                return Poll::Ready(None);
            };
            match collect.as_mut().poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Ok(_batches)) => {
                    this.collect = None;
                    Poll::Ready(None)
                },
                Poll::Ready(Err(error)) => {
                    this.collect = None;
                    Poll::Ready(Some(Err(error)))
                },
            }
        }
    }

    impl RecordBatchStream for NestedCollectStream {
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
    }

    fn adapter_error(error: &DataFusionError) -> &Error {
        match error {
            DataFusionError::External(source) => source.downcast_ref::<Error>().expect("expected adapter error"),
            other => panic!("expected adapter error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn host_execution_qos_limiter_holds_one_per_task_context_until_stream_drops() {
        let limiter = HostExecutionQosLimiter::new(NonZeroUsize::new(1).unwrap());
        let execute_count = Arc::new(AtomicUsize::new(0));
        let input =
            Arc::new(PendingExec::new(Arc::new(Schema::empty()), Arc::clone(&execute_count))) as Arc<dyn ExecutionPlan>;
        let plan = wrap_plan_with_host_execution_qos(input, limiter.clone());

        let mut first = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), first.next()).await.is_err(),
            "first stream should enter the pending input after admission"
        );
        assert_eq!(execute_count.load(Ordering::SeqCst), 1);
        assert_eq!(limiter.available_permits(), 0);

        let mut second = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), second.next()).await.is_err(),
            "second stream should wait for the host execution QoS permit"
        );
        assert_eq!(execute_count.load(Ordering::SeqCst), 1);

        drop(first);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), second.next()).await.is_err(),
            "second stream should enter the pending input after the first stream drops"
        );
        assert_eq!(execute_count.load(Ordering::SeqCst), 2);
        assert_eq!(limiter.available_permits(), 0);
    }

    #[tokio::test]
    async fn host_execution_qos_limiter_reuses_outer_round_for_nested_collect_with_same_task_context() {
        let limiter = HostExecutionQosLimiter::new(NonZeroUsize::new(1).unwrap());
        let inner_execute_count = Arc::new(AtomicUsize::new(0));
        let schema = Arc::new(Schema::empty());
        let inner = Arc::new(CountingEmptyExec::new(Arc::clone(&schema), Arc::clone(&inner_execute_count)))
            as Arc<dyn ExecutionPlan>;
        let inner = wrap_plan_with_host_execution_qos(inner, limiter.clone());
        let outer = Arc::new(NestedCollectExec::new(Arc::clone(&schema), inner)) as Arc<dyn ExecutionPlan>;
        let outer = wrap_plan_with_host_execution_qos(outer, limiter.clone());

        tokio::time::timeout(Duration::from_secs(1), collect(outer, Arc::new(TaskContext::default())))
            .await
            .expect("nested host QoS collect should not self-deadlock")
            .expect("nested host QoS collect should succeed");

        assert_eq!(inner_execute_count.load(Ordering::SeqCst), 1);
        assert_eq!(limiter.available_permits(), 1);
    }

    #[tokio::test]
    async fn host_execution_qos_limiter_times_out_waiting_sibling_statement() {
        let limiter = HostExecutionQosLimiter::new_with_wait_timeout(
            NonZeroUsize::new(1).unwrap(),
            Some(Duration::from_millis(5)),
        );
        let execute_count = Arc::new(AtomicUsize::new(0));
        let input =
            Arc::new(PendingExec::new(Arc::new(Schema::empty()), Arc::clone(&execute_count))) as Arc<dyn ExecutionPlan>;
        let plan = wrap_plan_with_host_execution_qos(input, limiter.clone());

        let mut first = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(50), first.next()).await.is_err());
        assert_eq!(limiter.available_permits(), 0);

        let mut second = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let error = tokio::time::timeout(Duration::from_secs(1), second.next())
            .await
            .expect("wait timeout should return promptly")
            .expect("timeout should produce a stream item")
            .expect_err("saturated host execution QoS limiter should reject after wait timeout");
        let error = adapter_error(&error);
        assert_eq!(error.code(), ErrorCode::ServiceOverloaded);
        assert_eq!(error.fact_value("reason"), Some("host_execution_qos_wait_timeout"));
        assert_eq!(error.fact_value("limit"), Some("1"));
        assert_eq!(error.fact_value("waited_ms"), Some("5"));

        drop(first);
        assert_eq!(limiter.available_permits(), 1);
    }

    #[tokio::test]
    async fn host_execution_qos_timeout_does_not_start_gpu_admission() {
        let host_limiter = HostExecutionQosLimiter::new_with_wait_timeout(
            NonZeroUsize::new(1).unwrap(),
            Some(Duration::from_millis(5)),
        );
        let first_execute_count = Arc::new(AtomicUsize::new(0));
        let first_input = Arc::new(PendingExec::new(Arc::new(Schema::empty()), Arc::clone(&first_execute_count)))
            as Arc<dyn ExecutionPlan>;
        let first_plan = wrap_plan_with_host_execution_qos(first_input, host_limiter.clone());
        let mut first = first_plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(50), first.next()).await.is_err());
        assert_eq!(first_execute_count.load(Ordering::SeqCst), 1);

        let gpu_limiter = UnboundedNativeExecutionLimiter::new(NonZeroUsize::new(1).unwrap());
        let second_execute_count = Arc::new(AtomicUsize::new(0));
        let second_input = Arc::new(PendingExec::new(Arc::new(Schema::empty()), Arc::clone(&second_execute_count)))
            as Arc<dyn ExecutionPlan>;
        let native_admission = wrap_plan_with_unbounded_native_execution_limit(second_input, gpu_limiter.clone());
        let second_plan = wrap_plan_with_host_execution_qos(native_admission, host_limiter.clone());
        let mut second = second_plan.execute(0, Arc::new(TaskContext::default())).unwrap();

        let error = tokio::time::timeout(Duration::from_secs(1), second.next())
            .await
            .expect("host QoS timeout should finish promptly")
            .expect("host QoS timeout should produce a stream item")
            .expect_err("second statement must time out before GPU admission");
        let error = adapter_error(&error);
        assert_eq!(error.fact_value("reason"), Some("host_execution_qos_wait_timeout"));
        assert_eq!(second_execute_count.load(Ordering::SeqCst), 0);
        assert_eq!(gpu_limiter.available_permits(), 1, "timed-out host wait must not start GPU admission");

        drop(first);
        assert_eq!(host_limiter.available_permits(), 1);
    }

    #[tokio::test]
    async fn host_execution_qos_limiter_cancellation_while_waiting_resets_acquire_state() {
        let limiter = HostExecutionQosLimiter::new(NonZeroUsize::new(1).unwrap());
        let execute_count = Arc::new(AtomicUsize::new(0));
        let input =
            Arc::new(PendingExec::new(Arc::new(Schema::empty()), Arc::clone(&execute_count))) as Arc<dyn ExecutionPlan>;
        let plan = wrap_plan_with_host_execution_qos(input, limiter);

        let mut first = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(50), first.next()).await.is_err());
        assert_eq!(execute_count.load(Ordering::SeqCst), 1);

        let waiting_context = Arc::new(TaskContext::default());
        let mut second = plan.execute(0, Arc::clone(&waiting_context)).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), second.next()).await.is_err(),
            "second stream should be waiting for the held permit"
        );
        drop(second);
        drop(first);

        let mut third = plan.execute(0, waiting_context).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), third.next()).await.is_err(),
            "third stream with the same TaskContext should acquire after the cancelled wait"
        );
        assert_eq!(execute_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn host_execution_qos_precedes_gpu_admission_and_cancellation_releases_it() {
        let host_limiter = HostExecutionQosLimiter::new(NonZeroUsize::new(1).unwrap());
        let gpu_limiter = UnboundedNativeExecutionLimiter::new(NonZeroUsize::new(1).unwrap());
        let gpu_blocker = gpu_limiter.acquire().await.expect("test GPU permit should acquire");
        let execute_count = Arc::new(AtomicUsize::new(0));
        let input =
            Arc::new(PendingExec::new(Arc::new(Schema::empty()), Arc::clone(&execute_count))) as Arc<dyn ExecutionPlan>;
        let native_admission = wrap_plan_with_unbounded_native_execution_limit(input, gpu_limiter.clone());
        let plan = wrap_plan_with_host_execution_qos(native_admission, host_limiter.clone());

        let mut stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), stream.next()).await.is_err(),
            "the outer host permit must be acquired before waiting for GPU admission"
        );
        assert_eq!(host_limiter.available_permits(), 0, "host QoS is held while the GPU boundary waits");
        assert_eq!(execute_count.load(Ordering::SeqCst), 0, "GPU admission must precede child execution");

        drop(stream);
        assert_eq!(host_limiter.available_permits(), 1, "cancelling before GPU admission releases host QoS");
        assert_eq!(gpu_limiter.available_permits(), 0, "only the test blocker holds the GPU admission permit");
        drop(gpu_blocker);
        assert_eq!(gpu_limiter.available_permits(), 1, "cancellation leaves no GPU admission permit behind");
    }
}
