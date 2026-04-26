//! Shared helpers for the native execution wrappers.
//!
//! Each wrapper (`NexusNativeExec`, `NexusNativeComposedExec`,
//! `NexusNativeGpuProducerExec`) advertises the same conservative
//! `PlanProperties` and uses the same error tagging at the DataFusion runtime
//! boundary, so those concerns live here rather than being duplicated.

use std::{
    collections::BTreeSet,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, utils::memory::get_record_batch_memory_size},
    execution::{
        context::TaskContext,
        memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation},
    },
    physical_expr::{EquivalenceProperties, Partitioning},
    physical_plan::{
        PlanProperties, RecordBatchStream, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        metrics::RecordOutput,
    },
};
use futures::Stream;
use nexus_query_engine::{
    observability::RuntimeObservationHandle,
    plan::{PlanNode, QueryPlan},
    runtime::{GpuExecutionJobKind, GpuExecutionPool, resident::GpuHostBatchCursor},
    source::SourceNode,
};

use super::metrics::NexusNativeMetrics;
use crate::error::{Error, ErrorCode};

pub(super) const NATIVE_NEXUS_EXEC_HOST_OUTPUT_MEMORY_CONSUMER: &str = "NexusNativeExec[host-output]";
pub(super) const NATIVE_NEXUS_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER: &str = "NexusNativeComposedExec[host-output]";

struct HostOutputReservation {
    reservation: Option<MemoryReservation>,
    estimated_bytes: usize,
}

impl HostOutputReservation {
    fn acquire(
        pool: &Arc<dyn MemoryPool>,
        enabled: bool,
        consumer_name: &'static str,
        estimated_bytes: usize,
    ) -> DataFusionResult<Self> {
        let reservation = enabled
            .then(|| {
                let reservation = MemoryConsumer::new(consumer_name).with_can_spill(false).register(pool);
                reservation.try_grow(estimated_bytes)?;
                Ok::<MemoryReservation, DataFusionError>(reservation)
            })
            .transpose()?;
        Ok(Self { reservation, estimated_bytes })
    }

    fn reconcile(&self, actual_bytes: usize) -> DataFusionResult<()> {
        let released_bytes = self.estimated_bytes.checked_sub(actual_bytes).ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::CudfInvariantViolation,
                    "datafusion_nexus.native_exec.host_output_reservation",
                    "materialized Arrow batch exceeded its pre-copy destination estimate",
                )
                .with_fact("estimated_materialization_bytes", self.estimated_bytes)
                .with_fact("actual_materialization_bytes", actual_bytes)
                .with_fact("reason", "arrow_destination_estimate_underrun"),
            )
        })?;
        if let Some(reservation) = self.reservation.as_ref() {
            // The estimate was acquired before D2H. Reconciliation may only
            // release surplus bytes; it never performs a second fallible pool
            // growth after the host allocation already happened.
            if released_bytes != 0 {
                reservation.shrink(released_bytes);
            }
        }
        Ok(())
    }

    fn into_inner(mut self) -> Option<MemoryReservation> {
        self.reservation.take()
    }
}

/// Conservative `PlanProperties` shared by all native execution wrappers:
/// one unknown bounded partition, `Final` emission, no ordering guarantees.
pub(super) fn conservative_properties(schema: SchemaRef) -> Arc<PlanProperties> {
    properties_with_equivalence(schema.clone(), EquivalenceProperties::new(schema))
        .expect("fresh equivalence properties must match schema")
}

/// DataFusion-facing native wrappers expose one bounded output partition and
/// may adopt the replaced subtree's semantic equivalence properties when the
/// native plan has already proven it reproduces the declared ordering.
pub(super) fn properties_with_equivalence(
    schema: SchemaRef,
    eq_properties: EquivalenceProperties,
) -> crate::error::Result<Arc<PlanProperties>> {
    if eq_properties.schema().as_ref() != schema.as_ref() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.native_exec.properties",
            "native wrapper equivalence properties must match output schema",
        )
        .with_fact("equivalence_schema_fields", eq_properties.schema().fields().len())
        .with_fact("output_schema_fields", schema.fields().len()));
    }
    Ok(Arc::new(PlanProperties::new(
        eq_properties,
        Partitioning::UnknownPartitioning(1),
        EmissionType::Final,
        Boundedness::Bounded,
    )))
}

pub(super) struct NativeHostOutputCursor {
    cursor: GpuHostBatchCursor,
    pool: GpuExecutionPool,
    observation: RuntimeObservationHandle,
    metrics: NexusNativeMetrics,
}

impl NativeHostOutputCursor {
    pub(super) fn new(
        cursor: GpuHostBatchCursor,
        pool: GpuExecutionPool,
        observation: RuntimeObservationHandle,
        metrics: NexusNativeMetrics,
    ) -> Self {
        Self { cursor, pool, observation, metrics }
    }

    fn next(
        self,
        task_context: Arc<TaskContext>,
        account_memory_pool: bool,
        consumer_name: &'static str,
    ) -> BoxedNativeHostOutputStepFuture {
        Box::pin(async move {
            let Self { cursor, pool, observation, metrics } = self;
            let task_pool = pool.clone();
            let task_observation = observation.clone();
            let (cursor, batch, reservation) = run_gpu_execution_task(
                task_pool,
                task_observation,
                "datafusion_nexus.native_exec.host_batch_cursor",
                move |_gpu_stream_scope| async move {
                    let mut cursor = cursor;
                    let Some(prepared) = cursor.prepare_next_batch().map_err(|error| {
                        DataFusionError::from(native_query_runtime_error(
                            "datafusion_nexus.native_exec.host_batch_cursor",
                            "bounded native host-output cursor failed to size its next chunk",
                            error,
                        ))
                    })?
                    else {
                        return Ok::<_, DataFusionError>((cursor, None, None));
                    };

                    // Reserve from the TaskContext's shared DataFusion pool
                    // before any D2H value copy. Each native output stream has
                    // at most one live reservation, while the shared pool
                    // accounts concurrent queries globally.
                    let reservation = HostOutputReservation::acquire(
                        task_context.memory_pool(),
                        account_memory_pool,
                        consumer_name,
                        prepared.estimated_materialization_bytes(),
                    )?;
                    let batch = cursor.materialize_prepared_batch(prepared).map_err(|error| {
                        DataFusionError::from(native_query_runtime_error(
                            "datafusion_nexus.native_exec.host_batch_cursor",
                            "bounded native host-output cursor failed to materialize its reserved chunk",
                            error,
                        ))
                    })?;
                    reservation.reconcile(get_record_batch_memory_size(&batch))?;
                    Ok::<_, DataFusionError>((cursor, Some(batch), reservation.into_inner()))
                },
            )
            .await?;

            let Some(batch) = batch else {
                return Ok(NativeHostOutputStep { cursor: None, batch: None, reservation: None });
            };
            metrics.record_deferred_host_materialization(batch.num_rows(), get_record_batch_memory_size(&batch));
            let batch = batch.record_output(&metrics.baseline);
            Ok(NativeHostOutputStep {
                cursor: Some(Self { cursor, pool, observation, metrics }),
                batch: Some(batch),
                reservation,
            })
        })
    }
}

struct NativeHostOutputStep {
    cursor: Option<NativeHostOutputCursor>,
    batch: Option<RecordBatch>,
    reservation: Option<MemoryReservation>,
}

type BoxedNativeHostOutputFuture =
    Pin<Box<dyn Future<Output = DataFusionResult<NativeHostOutputCursor>> + Send + 'static>>;
type BoxedNativeHostOutputStepFuture =
    Pin<Box<dyn Future<Output = DataFusionResult<NativeHostOutputStep>> + Send + 'static>>;

enum NativeHostOutputState {
    LoadingInitial(BoxedNativeHostOutputFuture),
    LoadingNext(BoxedNativeHostOutputStepFuture),
    Ready(Box<NativeHostOutputStep>),
    Done,
}

struct NativeHostOutputStream {
    schema: SchemaRef,
    state: NativeHostOutputState,
    reservation: Option<MemoryReservation>,
    task_context: Arc<TaskContext>,
    account_memory_pool: bool,
    consumer_name: &'static str,
}

impl Stream for NativeHostOutputStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let state = std::mem::replace(&mut self.state, NativeHostOutputState::Done);
            match state {
                NativeHostOutputState::LoadingInitial(mut future) => match future.as_mut().poll(cx) {
                    Poll::Pending => {
                        self.state = NativeHostOutputState::LoadingInitial(future);
                        return Poll::Pending;
                    },
                    Poll::Ready(Ok(cursor)) => {
                        self.state = NativeHostOutputState::LoadingNext(cursor.next(
                            Arc::clone(&self.task_context),
                            self.account_memory_pool,
                            self.consumer_name,
                        ));
                    },
                    Poll::Ready(Err(error)) => {
                        self.state = NativeHostOutputState::Done;
                        return Poll::Ready(Some(Err(error)));
                    },
                },
                NativeHostOutputState::LoadingNext(mut future) => match future.as_mut().poll(cx) {
                    Poll::Pending => {
                        self.state = NativeHostOutputState::LoadingNext(future);
                        return Poll::Pending;
                    },
                    Poll::Ready(Ok(mut output)) => {
                        let Some(batch) = output.batch.take() else {
                            self.state = NativeHostOutputState::Done;
                            continue;
                        };
                        self.reservation = output.reservation.take();
                        output.batch = Some(batch);
                        self.state = NativeHostOutputState::Ready(Box::new(output));
                    },
                    Poll::Ready(Err(error)) => {
                        self.state = NativeHostOutputState::Done;
                        return Poll::Ready(Some(Err(error)));
                    },
                },
                NativeHostOutputState::Ready(mut output) => {
                    if let Some(batch) = output.batch.take() {
                        self.state = NativeHostOutputState::Ready(output);
                        return Poll::Ready(Some(Ok(batch)));
                    }
                    self.reservation = None;
                    let Some(cursor) = output.cursor.take() else {
                        self.state = NativeHostOutputState::Done;
                        continue;
                    };
                    self.state = NativeHostOutputState::LoadingNext(cursor.next(
                        Arc::clone(&self.task_context),
                        self.account_memory_pool,
                        self.consumer_name,
                    ));
                },
                NativeHostOutputState::Done => {
                    self.reservation = None;
                    return Poll::Ready(None);
                },
            }
        }
    }
}

impl RecordBatchStream for NativeHostOutputStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Builds a bounded native host-output stream, charging at most the currently
/// yielded Arrow chunk to DataFusion's host-side memory pool.
pub(super) fn native_host_output_stream<F>(
    consumer_name: &'static str,
    schema: SchemaRef,
    task_context: Arc<TaskContext>,
    account_memory_pool: bool,
    future: F,
) -> SendableRecordBatchStream
where
    F: Future<Output = DataFusionResult<NativeHostOutputCursor>> + Send + 'static,
{
    Box::pin(NativeHostOutputStream {
        schema,
        state: NativeHostOutputState::LoadingInitial(Box::pin(future)),
        reservation: None,
        task_context,
        account_memory_pool,
        consumer_name,
    })
}

/// Runs a host-facing native query future on its device's dedicated GPU executor.
pub(crate) async fn run_gpu_execution_task<B, F, T>(
    pool: GpuExecutionPool,
    observation: RuntimeObservationHandle,
    operation: &'static str,
    build_future: B,
) -> DataFusionResult<T>
where
    B: FnOnce(nexus_query_engine::observability::RuntimeStreamLeaseScope) -> F + Send + 'static,
    F: Future<Output = DataFusionResult<T>> + Send + 'static,
    T: Send + 'static,
{
    pool.run_observed_scoped(GpuExecutionJobKind::QueryExecution, observation, build_future).await.map_err(|error| {
        DataFusionError::from(native_query_runtime_error(operation, "dedicated GPU execution task failed", error))
    })?
}

/// Collects `input_name`s of every GPU-resident source node in the plan.
///
/// Used to look up upstream DataFusion children that are expected to deliver
/// GPU-resident batches into the wrapper.
pub(super) fn gpu_resident_source_names(plan: &QueryPlan) -> BTreeSet<String> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) => Some(source.input_name.clone()),
                SourceNode::LocalParquet(_) => None,
                SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .collect()
}

/// Joins iterator items with `,`. Used to render stable comma-separated lists
/// for plan-display strings.
pub(super) fn comma_separated<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    values.into_iter().collect::<Vec<_>>().join(",")
}

/// Renders the native IR as a single physical-plan display detail.
pub(super) fn native_plan_one_line(plan: &QueryPlan) -> String {
    plan.to_string().lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>().join(" | ")
}

/// Renders the native IR under a DataFusion tree-render detail.
pub(super) fn write_native_plan_tree(f: &mut std::fmt::Formatter<'_>, plan: &QueryPlan) -> std::fmt::Result {
    writeln!(f, "native_plan:")?;
    for line in plan.to_string().lines() {
        writeln!(f, "  {line}")?;
    }
    Ok(())
}

/// Maps a `QueryError` from native runtime execution into the adapter [`Error`]
/// while tagging the stable diagnostic facts callers assert on.
pub(super) fn native_query_runtime_error(
    operation: &'static str,
    message: impl Into<String>,
    source: nexus_query_engine::error::QueryError,
) -> Error {
    mark_native_runtime_error(Error::from_native(operation, message, source), "runtime_execution")
}

/// Tags an error with `native_datafusion_stage=<stage>` and
/// `cpu_fallback_after_native_start=false`.
///
/// Callers use these facts to distinguish context setup from query runtime,
/// and to assert that a started native attempt is terminal rather than replayed
/// through DataFusion.
pub(super) fn mark_native_runtime_error(error: Error, stage: &'static str) -> Error {
    error
        .with_log_safe_fact("native_datafusion_stage", stage)
        .with_log_safe_fact("cpu_fallback_after_native_start", false)
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{Arc, Barrier},
    };

    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

    use super::{HostOutputReservation, mark_native_runtime_error};
    use crate::error::{DataFusionErrorExt, Error, ErrorCode};

    const TEST_CONSUMER: &str = "NativeHostOutputReservation[test]";

    #[test]
    fn native_runtime_error_marks_started_attempt_terminal_without_datafusion_replay() {
        let error = mark_native_runtime_error(
            Error::new(ErrorCode::MemoryReservationFailed, "datafusion_nexus.test", "memory reservation failed"),
            "runtime_execution",
        );

        assert_eq!(error.fact_value("native_datafusion_stage"), Some("runtime_execution"));
        assert_eq!(
            error.fact_value("cpu_fallback_after_native_start"),
            Some("false"),
            "a started native attempt must be terminal rather than replayed through DataFusion"
        );
    }

    #[test]
    fn host_output_reservations_account_concurrent_queries_globally_and_release_exactly() {
        const PER_QUERY_BYTES: usize = 1024;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4 * PER_QUERY_BYTES));
        let barrier = Arc::new(Barrier::new(5));
        let workers = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let reservation =
                        HostOutputReservation::acquire(&pool, true, TEST_CONSUMER, PER_QUERY_BYTES).unwrap();
                    barrier.wait();
                    barrier.wait();
                    drop(reservation);
                })
            })
            .collect::<Vec<_>>();

        barrier.wait();
        assert_eq!(pool.reserved(), 4 * PER_QUERY_BYTES);
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn host_output_reservation_reconciles_and_releases_on_cancellation_drop() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4096));
        let reservation = HostOutputReservation::acquire(&pool, true, TEST_CONSUMER, 2048).unwrap();
        assert_eq!(pool.reserved(), 2048);
        reservation.reconcile(512).unwrap();
        assert_eq!(pool.reserved(), 512);

        // Dropping the stream/future on cooperative cancellation drops this
        // guard before another D2H step can start.
        drop(reservation);
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn host_output_reservation_releases_during_unwind() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4096));
        let panic_pool = Arc::clone(&pool);
        let result = catch_unwind(AssertUnwindSafe(move || {
            let _reservation = HostOutputReservation::acquire(&panic_pool, true, TEST_CONSUMER, 2048).unwrap();
            panic!("injected host materialization panic");
        }));

        assert!(result.is_err());
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn host_output_reservation_estimate_underrun_is_a_structured_invariant_without_post_copy_growth() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4096));
        let reservation = HostOutputReservation::acquire(&pool, true, TEST_CONSUMER, 128).unwrap();
        assert_eq!(pool.reserved(), 128);

        let error = reservation.reconcile(129).expect_err("actual bytes above estimate must be an invariant");
        let adapter = error.find_nexus_error().expect("structured adapter error must survive DataFusion conversion");
        assert_eq!(adapter.code(), ErrorCode::CudfInvariantViolation);
        assert_eq!(adapter.fact_value("estimated_materialization_bytes"), Some("128"));
        assert_eq!(adapter.fact_value("actual_materialization_bytes"), Some("129"));
        assert_eq!(pool.reserved(), 128, "reconciliation must never attempt post-D2H pool growth");

        drop(reservation);
        assert_eq!(pool.reserved(), 0);
    }
}
