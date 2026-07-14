//! Shared helpers for the native execution wrappers.
//!
//! Each wrapper (`NativeCudfExec`, `NativeCudfComposedExec`,
//! `NativeCudfGpuProducerExec`) advertises the same conservative
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
        memory_pool::{MemoryConsumer, MemoryReservation},
    },
    physical_expr::{EquivalenceProperties, Partitioning},
    physical_plan::{
        PlanProperties, RecordBatchStream, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
    },
};
use futures::Stream;
use nexus_query_engine::{
    plan::{PlanNode, QueryPlan},
    runtime::{GpuExecutionJobKind, GpuExecutionPool},
    source::SourceNode,
};

use crate::error::{Error, ErrorCode};

pub(super) const NATIVE_CUDF_EXEC_HOST_OUTPUT_MEMORY_CONSUMER: &str = "NativeCudfExec[host-output]";
pub(super) const NATIVE_CUDF_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER: &str = "NativeCudfComposedExec[host-output]";

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

struct NativeHostOutputBatch {
    batch: RecordBatch,
    reservation: Option<MemoryReservation>,
}

type BoxedNativeHostOutputFuture =
    Pin<Box<dyn Future<Output = DataFusionResult<NativeHostOutputBatch>> + Send + 'static>>;

enum NativeHostOutputState {
    Loading(BoxedNativeHostOutputFuture),
    Ready(Option<RecordBatch>),
    Done,
}

struct NativeHostOutputStream {
    schema: SchemaRef,
    state: NativeHostOutputState,
    reservation: Option<MemoryReservation>,
}

impl Stream for NativeHostOutputStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                NativeHostOutputState::Loading(future) => match future.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(output)) => {
                        self.reservation = output.reservation;
                        self.state = NativeHostOutputState::Ready(Some(output.batch));
                    },
                    Poll::Ready(Err(error)) => {
                        self.state = NativeHostOutputState::Done;
                        return Poll::Ready(Some(Err(error)));
                    },
                },
                NativeHostOutputState::Ready(batch) => {
                    let batch = batch.take();
                    self.state = NativeHostOutputState::Done;
                    return Poll::Ready(batch.map(Ok));
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

/// Builds a one-batch native host-output stream, optionally charging the
/// materialized Arrow batch to DataFusion's host-side memory pool.
pub(super) fn native_host_output_stream<F>(
    consumer_name: &'static str,
    schema: SchemaRef,
    task_context: Arc<TaskContext>,
    account_memory_pool: bool,
    future: F,
) -> SendableRecordBatchStream
where
    F: Future<Output = DataFusionResult<RecordBatch>> + Send + 'static,
{
    let future = async move {
        let batch = future.await?;
        let reservation = account_memory_pool
            .then(|| {
                let reservation =
                    MemoryConsumer::new(consumer_name).with_can_spill(false).register(task_context.memory_pool());
                reservation.try_grow(get_record_batch_memory_size(&batch))?;
                Ok::<MemoryReservation, datafusion::common::DataFusionError>(reservation)
            })
            .transpose()?;
        Ok(NativeHostOutputBatch { batch, reservation })
    };

    Box::pin(NativeHostOutputStream {
        schema,
        state: NativeHostOutputState::Loading(Box::pin(future)),
        reservation: None,
    })
}

/// Runs a host-facing native query future on its device's dedicated GPU executor.
pub(crate) async fn run_gpu_execution_task<F, T>(
    pool: GpuExecutionPool,
    operation: &'static str,
    future: F,
) -> DataFusionResult<T>
where
    F: Future<Output = DataFusionResult<T>> + Send + 'static,
    T: Send + 'static,
{
    pool.run(GpuExecutionJobKind::QueryExecution, future).await.map_err(|error| {
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
/// and to assert that native execution does not silently fall back to CPU
/// once started.
pub(super) fn mark_native_runtime_error(error: Error, stage: &'static str) -> Error {
    error.with_fact("native_datafusion_stage", stage).with_fact("cpu_fallback_after_native_start", false)
}
