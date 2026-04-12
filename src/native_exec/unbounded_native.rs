//! Execution-time slot bound for unbounded native GPU plans.

use std::{
    any::Any,
    future::Future,
    num::NonZeroUsize,
    pin::Pin,
    sync::Arc,
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
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::{Error, ErrorCode};

#[derive(Debug, Clone)]
pub(crate) struct UnboundedNativeExecutionLimiter {
    permits: Arc<Semaphore>,
    #[cfg(test)]
    max_concurrent_queries: NonZeroUsize,
}

impl UnboundedNativeExecutionLimiter {
    #[cfg(test)]
    pub(crate) fn new(max_concurrent_queries: NonZeroUsize) -> Self {
        Self { permits: Arc::new(Semaphore::new(max_concurrent_queries.get())), max_concurrent_queries }
    }

    pub(crate) fn from_shared_semaphore(
        permits: Arc<Semaphore>,
        #[cfg_attr(not(test), allow(unused_variables))] max_concurrent_queries: NonZeroUsize,
    ) -> Self {
        Self {
            permits,
            #[cfg(test)]
            max_concurrent_queries,
        }
    }

    #[cfg(test)]
    pub(crate) fn max_concurrent_queries(&self) -> NonZeroUsize {
        self.max_concurrent_queries
    }

    pub(crate) async fn acquire(&self) -> DataFusionResult<OwnedSemaphorePermit> {
        self.permits.clone().acquire_owned().await.map_err(|error| {
            DataFusionError::External(Box::new(
                Error::new(
                    ErrorCode::ServiceOverloaded,
                    "datafusion_nexus.unbounded_native_execution.admission",
                    "unbounded native execution admission queue was closed",
                )
                .with_source(error),
            ))
        })
    }

    #[cfg(test)]
    pub(crate) fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }
}

#[derive(Debug)]
pub(crate) struct UnboundedNativeExecutionAdmissionExec {
    input: Arc<dyn ExecutionPlan>,
    limiter: UnboundedNativeExecutionLimiter,
}

impl UnboundedNativeExecutionAdmissionExec {
    fn new(input: Arc<dyn ExecutionPlan>, limiter: UnboundedNativeExecutionLimiter) -> Self {
        Self { input, limiter }
    }
}

pub(crate) fn wrap_plan_with_unbounded_native_execution_limit(
    plan: Arc<dyn ExecutionPlan>,
    limiter: UnboundedNativeExecutionLimiter,
) -> Arc<dyn ExecutionPlan> {
    if plan_is_unbounded_native_execution_limited(plan.as_ref()) {
        return plan;
    }
    Arc::new(UnboundedNativeExecutionAdmissionExec::new(plan, limiter))
}

pub(crate) fn plan_is_unbounded_native_execution_limited(plan: &dyn ExecutionPlan) -> bool {
    (plan as &dyn Any).is::<UnboundedNativeExecutionAdmissionExec>()
}

impl DisplayAs for UnboundedNativeExecutionAdmissionExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.input.fmt_as(t, f)
    }
}

impl ExecutionPlan for UnboundedNativeExecutionAdmissionExec {
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
        Ok(Arc::new(Self { input, limiter: self.limiter.clone() }))
    }

    fn reset_state(self: Arc<Self>) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let input = Arc::clone(&self.input).reset_state()?;
        Ok(Arc::new(Self { input, limiter: self.limiter.clone() }))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(Box::pin(UnboundedNativeExecutionAdmissionStream::new(
            Arc::clone(&self.input),
            partition,
            context,
            self.limiter.clone(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.input.metrics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<datafusion::common::Statistics>> {
        self.input.partition_statistics(partition)
    }
}

type BoxAcquireFuture = Pin<Box<dyn Future<Output = DataFusionResult<OwnedSemaphorePermit>> + Send>>;

struct UnboundedNativeExecutionAdmissionStream {
    schema: SchemaRef,
    state: UnboundedNativeExecutionAdmissionStreamState,
    permit: Option<OwnedSemaphorePermit>,
}

enum UnboundedNativeExecutionAdmissionStreamState {
    Waiting { input: Arc<dyn ExecutionPlan>, partition: usize, context: Arc<TaskContext>, acquire: BoxAcquireFuture },
    Running(SendableRecordBatchStream),
    Done,
}

impl UnboundedNativeExecutionAdmissionStream {
    fn new(
        input: Arc<dyn ExecutionPlan>,
        partition: usize,
        context: Arc<TaskContext>,
        limiter: UnboundedNativeExecutionLimiter,
    ) -> Self {
        let schema = input.schema();
        let acquire = Box::pin(async move { limiter.acquire().await });
        Self {
            schema,
            state: UnboundedNativeExecutionAdmissionStreamState::Waiting { input, partition, context, acquire },
            permit: None,
        }
    }
}

impl Unpin for UnboundedNativeExecutionAdmissionStream {}

impl Stream for UnboundedNativeExecutionAdmissionStream {
    type Item = DataFusionResult<arrow::record_batch::RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match &mut this.state {
                UnboundedNativeExecutionAdmissionStreamState::Waiting { input, partition, context, acquire } => {
                    match acquire.as_mut().poll(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Err(error)) => {
                            this.state = UnboundedNativeExecutionAdmissionStreamState::Done;
                            return Poll::Ready(Some(Err(error)));
                        },
                        Poll::Ready(Ok(permit)) => {
                            let stream = match input.execute(*partition, Arc::clone(context)) {
                                Ok(stream) => stream,
                                Err(error) => {
                                    this.state = UnboundedNativeExecutionAdmissionStreamState::Done;
                                    return Poll::Ready(Some(Err(error)));
                                },
                            };
                            this.permit = Some(permit);
                            this.state = UnboundedNativeExecutionAdmissionStreamState::Running(stream);
                        },
                    }
                },
                UnboundedNativeExecutionAdmissionStreamState::Running(stream) => match stream.as_mut().poll_next(cx) {
                    Poll::Ready(None) => {
                        this.permit = None;
                        this.state = UnboundedNativeExecutionAdmissionStreamState::Done;
                        return Poll::Ready(None);
                    },
                    Poll::Ready(Some(Err(error))) => {
                        this.permit = None;
                        this.state = UnboundedNativeExecutionAdmissionStreamState::Done;
                        return Poll::Ready(Some(Err(error)));
                    },
                    other => return other,
                },
                UnboundedNativeExecutionAdmissionStreamState::Done => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for UnboundedNativeExecutionAdmissionStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
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

    use arrow::datatypes::Schema;
    use datafusion::physical_plan::empty::EmptyExec;
    use futures::StreamExt;

    use super::*;

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

    #[tokio::test]
    async fn unbounded_native_limiter_holds_permit_until_stream_drops() {
        let limiter = UnboundedNativeExecutionLimiter::new(NonZeroUsize::new(1).unwrap());
        let execute_count = Arc::new(AtomicUsize::new(0));
        let input =
            Arc::new(PendingExec::new(Arc::new(Schema::empty()), Arc::clone(&execute_count))) as Arc<dyn ExecutionPlan>;
        let plan = wrap_plan_with_unbounded_native_execution_limit(input, limiter.clone());

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
            "second stream should wait for the native execution permit"
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
}
