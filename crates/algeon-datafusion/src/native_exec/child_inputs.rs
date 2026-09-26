//! Reusable preparation for accelerator operators with relational children.
//!
//! Preparation happens before native admission. Direct device-resident
//! producers are retained as producers; ordinary DataFusion children are
//! executed once in deterministic partition/batch order and staged in the
//! DataFusion memory pool until one shared native context is available.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow::{array::new_empty_array, datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::{
    common::{DataFusionError, Result, utils::memory::get_record_batch_memory_size},
    execution::{
        context::TaskContext,
        memory_pool::{MemoryConsumer, MemoryReservation},
    },
    physical_plan::ExecutionPlan,
};
use futures::{StreamExt, TryStreamExt, stream::iter};
use query_engine::{
    native_error::cudf_query_error,
    runtime::{
        ExecutionCancellation, ExecutionContext,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
};

use super::{
    execute_gpu_resident_producer_with_context, gpu_resident_producer, prepare_gpu_resident_producer_for_execution,
};
use crate::error::{Error, ErrorCode};

const MULTI_CHILD_HOST_INPUT_MEMORY_CONSUMER: &str = "NativeMultiChild[host-input-staging]";

/// Default cumulative Arrow staging ceiling for one multi-child accelerator
/// operation.
pub(crate) const DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct NativeChildInputSpec {
    pub(crate) input_name: String,
    pub(crate) role: &'static str,
    pub(crate) plan: Arc<dyn ExecutionPlan>,
    /// Required columns in stable semantic order. Duplicate indices are
    /// removed by the frontend before constructing this spec.
    pub(crate) projection: Vec<usize>,
}

impl std::fmt::Debug for NativeChildInputSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeChildInputSpec")
            .field("input_name", &self.input_name)
            .field("role", &self.role)
            .field("plan", &self.plan.name())
            .field("projection", &self.projection)
            .finish()
    }
}

#[derive(Debug)]
enum StagedNativeChild {
    Producer { spec: NativeChildInputSpec, prepared_plan: Arc<dyn ExecutionPlan> },
    Host { spec: NativeChildInputSpec, schema: SchemaRef, batches: Vec<RecordBatch> },
}

#[derive(Debug)]
pub(crate) struct StagedNativeChildren {
    children: Vec<StagedNativeChild>,
    reservation: MemoryReservation,
}

#[derive(Debug, Clone)]
pub(crate) struct NativeChildBinding {
    pub(crate) input_name: String,
    pub(crate) role: &'static str,
    pub(crate) schema: SchemaRef,
    /// Projection from `schema` into the exact ID/vector column set consumed
    /// by the accelerator. Host-staged inputs are already projected and use an
    /// identity mapping; device producers remain device-resident and are
    /// projected by the native consumer plan.
    pub(crate) projection: Vec<usize>,
}

pub(crate) struct UploadedNativeChildren {
    pub(crate) inputs: GpuResidentInputMap,
    pub(crate) bindings: Vec<NativeChildBinding>,
    /// One exact Arrow-to-cuDF upload boundary for every ordinary child that
    /// was staged on the host before entering the shared native context.
    pub(crate) host_to_device_uploads: Vec<HostToDeviceUpload>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostToDeviceUpload {
    pub(crate) rows: u64,
    pub(crate) bytes: u64,
}

pub(crate) async fn stage_native_children(
    specs: Vec<NativeChildInputSpec>,
    task_context: Arc<TaskContext>,
    cancellation: &ExecutionCancellation,
    max_staging_bytes: usize,
) -> Result<StagedNativeChildren> {
    check_cancellation(cancellation)?;
    let reservation = MemoryConsumer::new(MULTI_CHILD_HOST_INPUT_MEMORY_CONSUMER)
        .with_can_spill(false)
        .register(task_context.memory_pool());
    // Claimed before any retained batch through compare-exchange, so interleaved partition
    // polls cannot jointly overshoot `max_staging_bytes`; the pool reservation below stays the
    // authoritative per-byte enforcement and is released with `reservation` on every path.
    let staged_bytes = AtomicUsize::new(0);
    let mut children = Vec::with_capacity(specs.len());

    for spec in specs {
        validate_projection(&spec)?;
        if gpu_resident_producer(spec.plan.as_ref()).is_some() {
            let prepared_plan =
                prepare_gpu_resident_producer_for_execution(Arc::clone(&spec.plan), Arc::clone(&task_context)).await?;
            if gpu_resident_producer(prepared_plan.as_ref()).is_none() {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.native_multi_child.prepare",
                    "GPU-resident child preparation returned a non-producer plan",
                )
                .with_fact("input_role", spec.role)
                .with_fact("child_name", prepared_plan.name())
                .into());
            }
            children.push(StagedNativeChild::Producer { spec, prepared_plan });
            continue;
        }
        if contains_gpu_resident_producer(spec.plan.as_ref()) {
            super::validate_host_wrapped_gpu_execution(spec.plan.as_ref(), task_context.as_ref())?;
        }
        let schema = Arc::new(spec.plan.schema().project(&spec.projection).map_err(DataFusionError::from)?);
        let partition_count = spec.plan.properties().output_partitioning().partition_count();
        let batches = stage_child_partitions_bounded(
            &spec,
            Arc::clone(&task_context),
            cancellation,
            partition_count,
            max_staging_bytes,
            &staged_bytes,
            &reservation,
        )
        .await?;
        children.push(StagedNativeChild::Host { spec, schema, batches });
    }

    Ok(StagedNativeChildren { children, reservation })
}

impl StagedNativeChildren {
    pub(crate) async fn upload(
        self,
        context: &ExecutionContext,
        task_context: Arc<TaskContext>,
    ) -> Result<UploadedNativeChildren> {
        let Self { children, reservation } = self;
        let mut inputs = GpuResidentInputMap::new();
        let mut bindings = Vec::with_capacity(children.len());
        let mut host_to_device_uploads = Vec::new();

        for child in children {
            match child {
                StagedNativeChild::Producer { spec, prepared_plan } => {
                    let (_producer_name, handle) = execute_gpu_resident_producer_with_context(
                        prepared_plan.as_ref(),
                        context,
                        0,
                        Arc::clone(&task_context),
                    )
                    .await?;
                    let schema = Arc::clone(handle.schema());
                    validate_projection_for_schema(&spec, &schema)?;
                    inputs.insert(spec.input_name.clone(), handle);
                    bindings.push(NativeChildBinding {
                        input_name: spec.input_name,
                        role: spec.role,
                        schema,
                        projection: spec.projection,
                    });
                },
                StagedNativeChild::Host { spec, schema, batches } => {
                    let rows = batches.iter().fold(0u64, |rows, batch| {
                        rows.saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX))
                    });
                    let bytes = batches.iter().fold(0u64, |bytes, batch| {
                        bytes.saturating_add(u64::try_from(get_record_batch_memory_size(batch)).unwrap_or(u64::MAX))
                    });
                    let frame = record_batches_to_gpu(context, Arc::clone(&schema), &batches)?;
                    let handle = GpuResultHandle::new(context, frame, Arc::clone(&schema)).map_err(|error| {
                        DataFusionError::from(Error::from_native(
                            "algeon_datafusion.native_multi_child.host_input",
                            "failed to bind a staged Arrow child as a GPU-resident input",
                            error,
                        ))
                    })?;
                    inputs.insert(spec.input_name.clone(), handle);
                    bindings.push(NativeChildBinding {
                        input_name: spec.input_name,
                        role: spec.role,
                        projection: (0..schema.fields().len()).collect(),
                        schema,
                    });
                    host_to_device_uploads.push(HostToDeviceUpload { rows, bytes });
                },
            }
        }
        // Uploads own their device buffers, so the completed host staging quota can be released before native compute.
        drop(reservation);
        Ok(UploadedNativeChildren { inputs, bindings, host_to_device_uploads })
    }
}

fn validate_projection(spec: &NativeChildInputSpec) -> Result<()> {
    if spec.input_name.is_empty() || spec.projection.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native_multi_child.prepare",
            "native child input names and projections must be non-empty",
        )
        .with_fact("input_role", spec.role)
        .with_fact("input_name", &spec.input_name)
        .with_fact("projection_count", spec.projection.len())
        .into());
    }
    validate_projection_for_schema(spec, &spec.plan.schema())
}

fn validate_projection_for_schema(spec: &NativeChildInputSpec, schema: &SchemaRef) -> Result<()> {
    if let Some(index) = spec.projection.iter().copied().find(|index| *index >= schema.fields().len()) {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native_multi_child.prepare",
            "native child projection references a missing field",
        )
        .with_fact("input_role", spec.role)
        .with_fact("projection_index", index)
        .with_fact("schema_field_count", schema.fields().len())
        .into());
    }
    Ok(())
}

fn contains_gpu_resident_producer(plan: &dyn ExecutionPlan) -> bool {
    gpu_resident_producer(plan).is_some()
        || plan.children().into_iter().any(|child| contains_gpu_resident_producer(child.as_ref()))
}

fn check_cancellation(cancellation: &ExecutionCancellation) -> Result<()> {
    cancellation.check("algeon_datafusion.native_multi_child.host_input_staging").map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.native_multi_child.host_input_staging",
            "multi-child accelerator input staging was cancelled",
            error,
        ))
    })
}

/// Stages one ordinary child's partitions with every partition in flight, in stable order.
///
/// A DataFusion operator can hold one output partition until its siblings progress (a
/// CollectLeft hash join's dynamic filter waits for every probe partition to report build
/// bounds), so a window narrower than the partition count deadlocks. `buffered` yields
/// partitions in ordinal order regardless of completion order: ready-order merging would
/// reorder `LIMIT`/`OFFSET` rows and exact statistical summation. Bytes are claimed against
/// `max_staging_bytes` before the pool reservation grows, in first-seen order; the retained
/// total therefore never exceeds the cap. Any error or cancellation drops the in-flight
/// partitions and returns every claimed byte with `reservation`.
async fn stage_child_partitions_bounded(
    spec: &NativeChildInputSpec,
    task_context: Arc<TaskContext>,
    cancellation: &ExecutionCancellation,
    partition_count: usize,
    max_staging_bytes: usize,
    staged_bytes: &AtomicUsize,
    reservation: &MemoryReservation,
) -> Result<Vec<RecordBatch>> {
    let partitions = iter(0..partition_count)
        .map(|partition| {
            stage_child_partition(
                spec,
                Arc::clone(&task_context),
                cancellation,
                partition,
                max_staging_bytes,
                staged_bytes,
                reservation,
            )
        })
        .buffered(partition_count.max(1))
        .try_collect::<Vec<Vec<RecordBatch>>>()
        .await?;
    check_cancellation(cancellation)?;
    Ok(partitions.into_iter().flatten().collect())
}

/// Executes one partition's stream to completion, projecting and reserving each batch.
async fn stage_child_partition(
    spec: &NativeChildInputSpec,
    task_context: Arc<TaskContext>,
    cancellation: &ExecutionCancellation,
    partition: usize,
    max_staging_bytes: usize,
    staged_bytes: &AtomicUsize,
    reservation: &MemoryReservation,
) -> Result<Vec<RecordBatch>> {
    check_cancellation(cancellation)?;
    let mut stream = spec.plan.execute(partition, task_context)?;
    let mut batches = Vec::new();
    while let Some(batch) = tokio::select! {
        batch = stream.try_next() => batch?,
        () = cancellation.cancelled_or_expired() => None,
    } {
        check_cancellation(cancellation)?;
        let batch = batch.project(&spec.projection).map_err(DataFusionError::from)?;
        reserve_batch(spec, &batch, max_staging_bytes, staged_bytes, reservation)?;
        batches.push(batch);
    }
    check_cancellation(cancellation)?;
    Ok(batches)
}

fn reserve_batch(
    spec: &NativeChildInputSpec,
    batch: &RecordBatch,
    max_staging_bytes: usize,
    staged_bytes: &AtomicUsize,
    reservation: &MemoryReservation,
) -> Result<()> {
    let batch_bytes = get_record_batch_memory_size(batch);
    // Claim first through compare-exchange so interleaved partition polls cannot jointly
    // overshoot the cumulative cap; a denied pool reservation below releases the claim.
    let staged = loop {
        let staged = staged_bytes.load(Ordering::Relaxed);
        let next = staged
            .checked_add(batch_bytes)
            .ok_or_else(|| host_staging_limit_error(spec, staged, batch_bytes, max_staging_bytes))?;
        if next > max_staging_bytes {
            return Err(host_staging_limit_error(spec, staged, batch_bytes, max_staging_bytes));
        }
        if staged_bytes.compare_exchange(staged, next, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            break staged;
        }
    };
    reservation.try_grow(batch_bytes).map_err(|source| {
        staged_bytes.fetch_sub(batch_bytes, Ordering::Relaxed);
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "algeon_datafusion.native_multi_child.host_input_reservation",
                "DataFusion host memory pool denied multi-child accelerator staging",
            )
            .with_fact("input_role", spec.role)
            .with_fact("requested_bytes", batch_bytes)
            .with_fact("staged_bytes", staged)
            .with_source(source),
        )
    })?;
    Ok(())
}

fn host_staging_limit_error(
    spec: &NativeChildInputSpec,
    staged_bytes: usize,
    next_batch_bytes: usize,
    max_staging_bytes: usize,
) -> DataFusionError {
    Error::new(
        ErrorCode::HostStagingLimitExceeded,
        "algeon_datafusion.native_multi_child.host_input_staging",
        "multi-child accelerator input staging exceeded its cumulative byte limit",
    )
    .with_fact("input_role", spec.role)
    .with_fact("staged_bytes", staged_bytes)
    .with_fact("next_batch_bytes", next_batch_bytes)
    .with_fact("max_staging_bytes", max_staging_bytes)
    .into()
}

fn record_batches_to_gpu(
    context: &ExecutionContext,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> Result<GpuDataFrame> {
    let runtime = context.runtime().map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.native_multi_child.host_input_runtime",
            "query service closed before staged Arrow input acquired its cuDF runtime",
            error,
        ))
    })?;
    let imported = if batches.is_empty() {
        let arrays = schema.fields().iter().map(|field| new_empty_array(field.data_type())).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).map_err(DataFusionError::from)?;
        cudf::DataFrame::from_record_batch(&runtime, &batch)
    } else {
        cudf::DataFrame::from_record_batches(&runtime, batches)
    }
    .map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.native_multi_child.host_input",
            "failed to import staged Arrow input into cuDF",
            cudf_query_error(
                "algeon_datafusion.native_multi_child.host_input",
                "failed to import staged Arrow input into cuDF",
                error,
            ),
        ))
    })?;
    Ok(GpuDataFrame::new(imported))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::{Duration, Instant},
    };

    use arrow::{
        array::{Float32Array, Int64Array},
        datatypes::{DataType, Field, Schema},
    };
    use datafusion::{
        common::tree_node::TreeNodeRecursion,
        physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr},
        physical_plan::{
            DisplayAs, DisplayFormatType, PlanProperties, SendableRecordBatchStream,
            execution_plan::{Boundedness, EmissionType},
            stream::RecordBatchStreamAdapter,
        },
    };
    use futures::{
        future::BoxFuture,
        stream::{once, unfold},
    };
    use tokio::{
        sync::mpsc,
        task::yield_now,
        time::{Instant as TokioInstant, sleep},
    };

    use super::*;
    use crate::{error::DataFusionErrorExt, native_exec::GpuResidentProducer};

    #[derive(Debug)]
    struct CountingExec {
        schema: SchemaRef,
        partitions: Vec<RecordBatch>,
        execute_count: Arc<AtomicUsize>,
        cache: Arc<PlanProperties>,
    }

    impl CountingExec {
        fn new(partitions: Vec<RecordBatch>, execute_count: Arc<AtomicUsize>) -> Self {
            let schema = partitions[0].schema();
            let cache = Arc::new(PlanProperties::new(
                EquivalenceProperties::new(Arc::clone(&schema)),
                Partitioning::UnknownPartitioning(partitions.len()),
                EmissionType::Incremental,
                Boundedness::Bounded,
            ));
            Self { schema, partitions, execute_count, cache }
        }
    }

    impl DisplayAs for CountingExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("CountingExec")
        }
    }

    impl ExecutionPlan for CountingExec {
        fn name(&self) -> &str {
            "CountingExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.cache
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
            assert!(children.is_empty());
            Ok(self)
        }

        fn execute(&self, partition: usize, _context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
            self.execute_count.fetch_add(1, Ordering::SeqCst);
            let batch = self
                .partitions
                .get(partition)
                .cloned()
                .ok_or_else(|| DataFusionError::Execution(format!("missing test partition {partition}")))?;
            let stream = once(async move { Ok(batch) });
            Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
        }
    }

    /// Partitions with staggered open latency still stage in stable ordinal order, each
    /// executed exactly once, so ordered output and exact statistical inputs never observe
    /// ready order. The async open delay models remote partition latency.
    #[derive(Debug)]
    struct DelayedExec {
        inner: CountingExec,
        open_delays_ms: Vec<u64>,
        live: Arc<AtomicUsize>,
        max_live: Arc<AtomicUsize>,
    }

    impl DelayedExec {
        fn wrap(
            partitions: Vec<RecordBatch>,
            open_delays_ms: Vec<u64>,
        ) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            let execute_count = Arc::new(AtomicUsize::new(0));
            let live = Arc::new(AtomicUsize::new(0));
            let max_live = Arc::new(AtomicUsize::new(0));
            let exec = Self {
                inner: CountingExec::new(partitions, Arc::clone(&execute_count)),
                open_delays_ms,
                live: Arc::clone(&live),
                max_live: Arc::clone(&max_live),
            };
            (exec, execute_count, live, max_live)
        }
    }

    impl DisplayAs for DelayedExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("DelayedExec")
        }
    }

    impl ExecutionPlan for DelayedExec {
        fn name(&self) -> &str {
            "DelayedExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
            assert!(children.is_empty());
            Ok(self)
        }

        fn execute(&self, partition: usize, context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
            let live_back = Arc::clone(&self.live);
            let max_live = Arc::clone(&self.max_live);
            let delay_ms = self.open_delays_ms[partition];
            let current = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            max_live.fetch_max(current, Ordering::SeqCst);
            let stream = self.inner.execute(partition, context)?;
            let opener = async move {
                if delay_ms > 0 {
                    sleep(Duration::from_millis(delay_ms)).await;
                }
                live_back.fetch_sub(1, Ordering::SeqCst);
            };
            let delayed =
                once(opener).filter_map(|()| async { None::<Result<RecordBatch, DataFusionError>> }).chain(stream);
            Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.inner.schema), delayed)))
        }
    }

    #[derive(Debug)]
    struct SlowDripExec {
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        delay_ms: u64,
        cache: Arc<PlanProperties>,
    }

    impl SlowDripExec {
        /// One partition dripping `batches` in order with `delay_ms` between them. A zero
        /// delay keeps serial reservation order deterministic for cumulative-cap tests; a
        /// long drip makes mid-flight cancellation deterministic for cancel tests.
        fn wrap(batches: Vec<RecordBatch>, delay_ms: u64) -> Self {
            let schema = batches[0].schema();
            let cache = Arc::new(PlanProperties::new(
                EquivalenceProperties::new(Arc::clone(&schema)),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            ));
            Self { schema, batches, delay_ms, cache }
        }
    }

    impl DisplayAs for SlowDripExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("SlowDripExec")
        }
    }

    impl ExecutionPlan for SlowDripExec {
        fn name(&self) -> &str {
            "SlowDripExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.cache
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
            assert!(children.is_empty());
            Ok(self)
        }

        fn execute(&self, partition: usize, _context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
            assert_eq!(partition, 0);
            let batches = self.batches.clone();
            let delay_ms = self.delay_ms;
            let drip = iter(batches.into_iter().map(Ok)).then(move |batch| async move {
                if delay_ms > 0 {
                    sleep(Duration::from_millis(delay_ms)).await;
                }
                batch
            });
            Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), drip)))
        }
    }

    /// One partition that yields the batches its sender sends and stays
    /// pending until the next one, so a test decides when input arrives.
    #[derive(Debug)]
    struct ChannelExec {
        schema: SchemaRef,
        receiver: Mutex<Option<mpsc::UnboundedReceiver<RecordBatch>>>,
        cache: Arc<PlanProperties>,
    }

    impl ChannelExec {
        fn new(schema: SchemaRef) -> (mpsc::UnboundedSender<RecordBatch>, Self) {
            let (sender, receiver) = mpsc::unbounded_channel();
            let cache = Arc::new(PlanProperties::new(
                EquivalenceProperties::new(Arc::clone(&schema)),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            ));
            (sender, Self { schema, receiver: Mutex::new(Some(receiver)), cache })
        }
    }

    impl DisplayAs for ChannelExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("ChannelExec")
        }
    }

    impl ExecutionPlan for ChannelExec {
        fn name(&self) -> &str {
            "ChannelExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.cache
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
            assert!(children.is_empty());
            Ok(self)
        }

        fn execute(&self, partition: usize, _context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
            assert_eq!(partition, 0);
            let receiver = self.receiver.lock().unwrap().take().expect("the channel partition executes once");
            let batches =
                unfold(
                    receiver,
                    |mut receiver| async move { receiver.recv().await.map(|batch| (Ok(batch), receiver)) },
                );
            Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), batches)))
        }
    }

    #[derive(Debug)]
    pub(crate) struct FakeProducerExec {
        inner: CountingExec,
        prepare_count: Arc<AtomicUsize>,
    }

    impl DisplayAs for FakeProducerExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt_as(t, f)
        }
    }

    impl ExecutionPlan for FakeProducerExec {
        fn name(&self) -> &str {
            "FakeProducerExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
            assert!(children.is_empty());
            Ok(self)
        }

        fn execute(&self, partition: usize, context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
            self.inner.execute(partition, context)
        }
    }

    impl GpuResidentProducer for FakeProducerExec {
        fn gpu_result_input_name(&self) -> &str {
            "fake_producer"
        }

        fn prepare_execution_plan<'a>(
            &'a self,
            plan: Arc<dyn ExecutionPlan>,
            _task_context: Arc<TaskContext>,
        ) -> BoxFuture<'a, Result<Arc<dyn ExecutionPlan>>> {
            self.prepare_count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(plan) })
        }

        fn execute_gpu_result_with_context<'a>(
            &'a self,
            _native_context: &'a ExecutionContext,
            _partition: usize,
            _task_context: Arc<TaskContext>,
        ) -> BoxFuture<'a, Result<(String, GpuResultHandle)>> {
            Box::pin(async {
                Err(DataFusionError::Execution(
                    "fake producer execution requires no CPU contract implementation".to_string(),
                ))
            })
        }
    }

    #[derive(Debug)]
    struct HostWrapperExec {
        child: Arc<dyn ExecutionPlan>,
    }

    impl DisplayAs for HostWrapperExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("HostWrapperExec")
        }
    }

    impl ExecutionPlan for HostWrapperExec {
        fn name(&self) -> &str {
            "HostWrapperExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.child.properties()
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.child]
        }

        fn with_new_children(
            self: Arc<Self>,
            mut children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            if children.len() != 1 {
                return Err(DataFusionError::Execution("HostWrapperExec expects one child".to_string()));
            }
            Ok(Arc::new(Self { child: children.remove(0) }))
        }

        fn execute(&self, partition: usize, context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
            self.child.execute(partition, context)
        }
    }

    fn batch(id: i64, feature: f32, unused: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("feature", DataType::Float32, false),
            Field::new("unused", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![id])),
                Arc::new(Float32Array::from(vec![feature])),
                Arc::new(Int64Array::from(vec![unused])),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn ordinary_children_execute_each_partition_once_and_stage_only_projection() {
        let execute_count = Arc::new(AtomicUsize::new(0));
        let plan =
            Arc::new(CountingExec::new(vec![batch(10, 1.0, 100), batch(20, 2.0, 200)], Arc::clone(&execute_count)))
                as Arc<dyn ExecutionPlan>;
        let staged = stage_native_children(
            vec![NativeChildInputSpec { input_name: "input".to_string(), role: "input", plan, projection: vec![0, 1] }],
            Arc::new(TaskContext::default()),
            &ExecutionCancellation::new(),
            DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
        )
        .await
        .unwrap();

        assert_eq!(execute_count.load(Ordering::SeqCst), 2);
        let StagedNativeChild::Host { schema, batches, .. } = &staged.children[0] else {
            panic!("ordinary child must be host staged")
        };
        assert_eq!(schema.fields().iter().map(|field| field.name()).collect::<Vec<_>>(), ["id", "feature"]);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_columns(), 2);
        assert_eq!(batches[1].num_columns(), 2);
        assert_eq!(batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0), 10);
        assert_eq!(batches[1].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0), 20);
    }

    #[tokio::test]
    async fn direct_producer_prepares_without_invoking_host_arrow_execute() {
        let execute_count = Arc::new(AtomicUsize::new(0));
        let prepare_count = Arc::new(AtomicUsize::new(0));
        let plan = Arc::new(FakeProducerExec {
            inner: CountingExec::new(vec![batch(10, 1.0, 100)], Arc::clone(&execute_count)),
            prepare_count: Arc::clone(&prepare_count),
        }) as Arc<dyn ExecutionPlan>;
        let staged = stage_native_children(
            vec![NativeChildInputSpec { input_name: "input".to_string(), role: "input", plan, projection: vec![0, 1] }],
            Arc::new(TaskContext::default()),
            &ExecutionCancellation::new(),
            DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
        )
        .await
        .unwrap();

        assert!(matches!(staged.children[0], StagedNativeChild::Producer { .. }));
        assert_eq!(prepare_count.load(Ordering::SeqCst), 1);
        assert_eq!(execute_count.load(Ordering::SeqCst), 0, "producer must not cross the Arrow execute boundary");
    }

    #[tokio::test]
    async fn producer_hidden_below_host_wrapper_without_manifest_is_rejected() {
        let producer = Arc::new(FakeProducerExec {
            inner: CountingExec::new(vec![batch(10, 1.0, 100)], Arc::new(AtomicUsize::new(0))),
            prepare_count: Arc::new(AtomicUsize::new(0)),
        }) as Arc<dyn ExecutionPlan>;
        let wrapped = Arc::new(HostWrapperExec { child: producer }) as Arc<dyn ExecutionPlan>;
        let error = stage_native_children(
            vec![NativeChildInputSpec {
                input_name: "input".to_string(),
                role: "input",
                plan: wrapped,
                projection: vec![0, 1],
            }],
            Arc::new(TaskContext::default()),
            &ExecutionCancellation::new(),
            DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
        )
        .await
        .unwrap_err();
        let typed = error.find_algeon_error().expect("structured wrapper rejection");
        assert_eq!(typed.facts().get("reason"), Some("host_wrapped_gpu_occurrence_manifest_missing"));
    }

    fn delayed_child(
        values: Vec<(i64, f32, i64)>,
        open_delays_ms: Vec<u64>,
    ) -> (Arc<dyn ExecutionPlan>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let partitions = values.into_iter().map(|(id, feature, unused)| batch(id, feature, unused)).collect();
        let (exec, execute_count, _live, max_live) = DelayedExec::wrap(partitions, open_delays_ms);
        (Arc::new(exec) as Arc<dyn ExecutionPlan>, execute_count, max_live)
    }

    #[tokio::test]
    async fn polling_stages_partitions_in_stable_order_not_ready_order() {
        // Partition 0 opens slowly and partition 1 instantly: ready order is inverted, but
        // staged order must stay ordinal so ordered output and exact statistics never shift.
        let (plan, execute_count, _max_live) = delayed_child(vec![(10, 1.0, 100), (20, 2.0, 200)], vec![150, 0]);
        let staged = stage_native_children(
            vec![NativeChildInputSpec { input_name: "input".to_string(), role: "input", plan, projection: vec![0, 1] }],
            Arc::new(TaskContext::default()),
            &ExecutionCancellation::new(),
            DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
        )
        .await
        .unwrap();

        assert_eq!(execute_count.load(Ordering::SeqCst), 2, "every partition executes exactly once");
        let StagedNativeChild::Host { batches, .. } = &staged.children[0] else {
            panic!("ordinary child must be host staged")
        };
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0), 10);
        assert_eq!(batches[1].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0), 20);
    }

    #[tokio::test]
    async fn polling_keeps_every_partition_in_flight() {
        // Eight partitions with 100ms open latency each: a serial poller needs at least 800ms,
        // and a window narrower than the partition count would leave some partitions unopened
        // while others wait on them. The bound below stays far under the serial sum while
        // leaving wide margin for loaded CI runners.
        let values = (1..=8).map(|id| (id, id as f32, id)).collect();
        let (plan, execute_count, max_live) = delayed_child(values, vec![100; 8]);
        let started = Instant::now();
        let staged = stage_native_children(
            vec![NativeChildInputSpec { input_name: "input".to_string(), role: "input", plan, projection: vec![0] }],
            Arc::new(TaskContext::default()),
            &ExecutionCancellation::new(),
            DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
        )
        .await
        .unwrap();
        let elapsed = started.elapsed();

        assert_eq!(execute_count.load(Ordering::SeqCst), 8);
        assert_eq!(max_live.load(Ordering::SeqCst), 8, "staging polls every partition at once");
        let StagedNativeChild::Host { batches, .. } = &staged.children[0] else {
            panic!("ordinary child must be host staged")
        };
        assert_eq!(batches.len(), 8);
        assert!(
            elapsed < Duration::from_millis(350),
            "eight overlapped 100ms opens must stage well under the 800ms serial sum, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn bounded_polling_enforces_the_cumulative_cap_and_releases_the_pool() {
        // One partition drips two batches serially, so reservation order is deterministic: the
        // cap fits the first batch but rejects the second, exercising cumulative contention
        // (not just a first-batch rejection) and the release of the claimed first batch.
        let task_context = Arc::new(TaskContext::default());
        // Sizes are measured after projection, matching the reserve-before-push accounting.
        let projected = batch(10, 1.0, 100).project(&[0, 1]).expect("test projection applies");
        let projected_bytes = get_record_batch_memory_size(&projected);
        let plan =
            Arc::new(SlowDripExec::wrap(vec![batch(10, 1.0, 100), batch(20, 2.0, 200)], 0)) as Arc<dyn ExecutionPlan>;
        let error = stage_native_children(
            vec![NativeChildInputSpec { input_name: "input".to_string(), role: "input", plan, projection: vec![0, 1] }],
            Arc::clone(&task_context),
            &ExecutionCancellation::new(),
            projected_bytes + projected_bytes / 2,
        )
        .await
        .unwrap_err();
        let typed = error.find_algeon_error().expect("structured staging cap rejection");
        assert_eq!(
            typed.facts().get("staged_bytes"),
            Some(projected_bytes.to_string().as_str()),
            "the second batch contends against the retained first batch"
        );
        assert_eq!(task_context.memory_pool().reserved(), 0, "a capped run returns every pool byte");
    }

    #[tokio::test]
    async fn cancelled_bounded_polling_releases_every_claimed_reservation() {
        // One partition drips many small batches with sleeps between them: cancellation
        // requested mid-drip fires at the next cooperative batch boundary, deterministically
        // long before the stream could finish.
        let task_context = Arc::new(TaskContext::default());
        let batches = (0..20).map(|id| batch(id, id as f32, id)).collect::<Vec<_>>();
        let plan = Arc::new(SlowDripExec::wrap(batches, 50)) as Arc<dyn ExecutionPlan>;
        let cancellation = ExecutionCancellation::new();
        let staging = tokio::spawn({
            let task_context = Arc::clone(&task_context);
            let cancellation = cancellation.clone();
            async move {
                stage_native_children(
                    vec![NativeChildInputSpec {
                        input_name: "input".to_string(),
                        role: "input",
                        plan,
                        projection: vec![0, 1],
                    }],
                    task_context,
                    &cancellation,
                    DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
                )
                .await
            }
        });
        sleep(Duration::from_millis(100)).await;
        cancellation.cancel();
        let error = staging.await.expect("staging task joins").unwrap_err();
        let typed = error.find_algeon_error().expect("structured staging cancellation");
        assert!(typed.message().contains("cancelled"), "mid-flight cancellation, not completion, ends the run");
        assert_eq!(task_context.memory_pool().reserved(), 0, "cancellation returns every claimed byte");
    }

    /// Staging that waits on a pending input wakes on cancellation, not a
    /// timer.
    #[tokio::test(start_paused = true)]
    async fn cancelling_staging_while_its_input_is_pending_wakes_it_without_a_timer() {
        let task_context = Arc::new(TaskContext::default());
        let first = batch(0, 0.0, 0);
        let (sender, plan) = ChannelExec::new(first.schema());
        let plan = Arc::new(plan) as Arc<dyn ExecutionPlan>;
        let cancellation = ExecutionCancellation::new();
        let staging = tokio::spawn({
            let task_context = Arc::clone(&task_context);
            let cancellation = cancellation.clone();
            async move {
                stage_native_children(
                    vec![NativeChildInputSpec {
                        input_name: "input".to_string(),
                        role: "input",
                        plan,
                        projection: vec![0, 1],
                    }],
                    task_context,
                    &cancellation,
                    DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
                )
                .await
            }
        });
        sender.send(first).unwrap();
        while task_context.memory_pool().reserved() == 0 {
            yield_now().await;
        }

        let started = TokioInstant::now();
        cancellation.cancel();
        let error = staging.await.expect("staging task joins").unwrap_err();

        let typed = error.find_algeon_error().expect("structured staging cancellation");
        assert!(typed.message().contains("cancelled"), "cancellation, not input, ends the run");
        assert_eq!(started.elapsed(), Duration::ZERO, "cancellation wakes staging without a timer");
        assert_eq!(task_context.memory_pool().reserved(), 0, "cancellation returns every claimed byte");
        drop(sender);
    }
}
