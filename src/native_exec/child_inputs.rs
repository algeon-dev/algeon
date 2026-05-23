//! Reusable preparation for accelerator operators with relational children.
//!
//! Preparation happens before native admission. Direct device-resident
//! producers are retained as producers; ordinary DataFusion children are
//! executed once in deterministic partition/batch order and staged in the
//! DataFusion memory pool until one shared native context is available.

use std::sync::Arc;

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::{
    common::{DataFusionError, Result, utils::memory::get_record_batch_memory_size},
    execution::{
        context::TaskContext,
        memory_pool::{MemoryConsumer, MemoryReservation},
    },
    physical_plan::ExecutionPlan,
};
use futures::TryStreamExt;
use nexus_query_engine::runtime::{
    ExecutionCancellation, ExecutionContext,
    resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
};

use super::producer::{
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
    let mut reservation = MemoryConsumer::new(MULTI_CHILD_HOST_INPUT_MEMORY_CONSUMER)
        .with_can_spill(false)
        .register(task_context.memory_pool());
    let mut staged_bytes = 0usize;
    let mut children = Vec::with_capacity(specs.len());

    for spec in specs {
        validate_projection(&spec)?;
        if gpu_resident_producer(spec.plan.as_ref()).is_some() {
            let prepared_plan =
                prepare_gpu_resident_producer_for_execution(Arc::clone(&spec.plan), Arc::clone(&task_context)).await?;
            if gpu_resident_producer(prepared_plan.as_ref()).is_none() {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_multi_child.prepare",
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
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_multi_child.prepare",
                "an ordinary Arrow child cannot hide a GPU-resident producer below a host execution wrapper",
            )
            .with_fact("input_role", spec.role)
            .with_fact("child_name", spec.plan.name())
            .with_log_safe_fact("reason", "gpu_producer_below_host_wrapper")
            .into());
        }

        let schema = Arc::new(spec.plan.schema().project(&spec.projection).map_err(DataFusionError::from)?);
        let partition_count = spec.plan.properties().output_partitioning().partition_count();
        let mut batches = Vec::new();
        for partition in 0..partition_count {
            check_cancellation(cancellation)?;
            let mut stream = spec.plan.execute(partition, Arc::clone(&task_context))?;
            while let Some(batch) = stream.try_next().await? {
                check_cancellation(cancellation)?;
                let batch = batch.project(&spec.projection).map_err(DataFusionError::from)?;
                reserve_batch(&spec, &batch, max_staging_bytes, &mut staged_bytes, &mut reservation)?;
                batches.push(batch);
            }
        }
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
                            "datafusion_nexus.native_multi_child.host_input",
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
        drop(reservation);
        Ok(UploadedNativeChildren { inputs, bindings, host_to_device_uploads })
    }
}

fn validate_projection(spec: &NativeChildInputSpec) -> Result<()> {
    if spec.input_name.is_empty() || spec.projection.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.native_multi_child.prepare",
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
            "datafusion_nexus.native_multi_child.prepare",
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
    cancellation.check("datafusion_nexus.native_multi_child.host_input_staging").map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.native_multi_child.host_input_staging",
            "multi-child accelerator input staging was cancelled",
            error,
        ))
    })
}

fn reserve_batch(
    spec: &NativeChildInputSpec,
    batch: &RecordBatch,
    max_staging_bytes: usize,
    staged_bytes: &mut usize,
    reservation: &mut MemoryReservation,
) -> Result<()> {
    let batch_bytes = get_record_batch_memory_size(batch);
    let next = staged_bytes
        .checked_add(batch_bytes)
        .ok_or_else(|| host_staging_limit_error(spec, *staged_bytes, batch_bytes, max_staging_bytes))?;
    if next > max_staging_bytes {
        return Err(host_staging_limit_error(spec, *staged_bytes, batch_bytes, max_staging_bytes));
    }
    reservation.try_grow(batch_bytes).map_err(|source| {
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "datafusion_nexus.native_multi_child.host_input_reservation",
                "DataFusion host memory pool denied multi-child accelerator staging",
            )
            .with_fact("input_role", spec.role)
            .with_fact("requested_bytes", batch_bytes)
            .with_fact("staged_bytes", *staged_bytes)
            .with_source(source),
        )
    })?;
    *staged_bytes = next;
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
        "datafusion_nexus.native_multi_child.host_input_staging",
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
            "datafusion_nexus.native_multi_child.host_input_runtime",
            "query service closed before staged Arrow input acquired its cuDF runtime",
            error,
        ))
    })?;
    let imported = if batches.is_empty() {
        let arrays =
            schema.fields().iter().map(|field| arrow::array::new_empty_array(field.data_type())).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).map_err(DataFusionError::from)?;
        cudf::DataFrame::from_record_batch(&runtime, &batch)
    } else {
        cudf::DataFrame::from_record_batches(&runtime, batches)
    }
    .map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.native_multi_child.host_input",
            "failed to import staged Arrow input into cuDF",
            nexus_query_engine::error::QueryError::from_cudf(
                "datafusion_nexus.native_multi_child.host_input",
                "failed to import staged Arrow input into cuDF",
                error,
            ),
        ))
    })?;
    Ok(GpuDataFrame::new(imported))
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
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
    use futures::future::BoxFuture;

    use super::*;
    use crate::{
        error::DataFusionErrorExt,
        native_exec::producer::{GpuResidentProducer, register_gpu_resident_producer_discovery},
    };

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
            let stream = futures::stream::once(async move { Ok(batch) });
            Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
        }
    }

    #[derive(Debug)]
    struct FakeProducerExec {
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

    fn fake_producer_discovery(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
        plan.downcast_ref::<FakeProducerExec>().map(|producer| producer as &dyn GpuResidentProducer)
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
        register_gpu_resident_producer_discovery(fake_producer_discovery);
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
    async fn producer_hidden_below_host_wrapper_is_rejected() {
        register_gpu_resident_producer_discovery(fake_producer_discovery);
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
        let typed = error.find_nexus_error().expect("structured wrapper rejection");
        assert_eq!(typed.facts().get("reason"), Some("gpu_producer_below_host_wrapper"));
    }
}
