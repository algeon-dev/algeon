//! Final-plan binding for host-materializing GPU UDTF occurrences.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError},
};

use arrow::record_batch::RecordBatch;
use datafusion::{
    common::{
        DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion,
        utils::memory::get_record_batch_memory_size,
    },
    execution::{
        context::TaskContext,
        memory_pool::{MemoryConsumer, MemoryReservation},
    },
    physical_expr::PhysicalExpr,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, stream};
use tokio::sync::OnceCell;

use super::admission::shared_host_occurrence_results;
use crate::error::{Error, ErrorCode, Result};

const SHARED_HOST_RESULT_MEMORY_CONSUMER: &str = "GpuFunctionHostOccurrenceExec[shared-host-result]";

#[derive(Debug, Clone)]
struct ActiveGpuFunctionOccurrence {
    occurrence_id: Arc<str>,
    producer_identity: Arc<str>,
}

/// A transparent final-plan node that binds a shared physical producer to one
/// manifest occurrence. Its task-context lineage lets runtime dependency
/// validation use the physical path selected by DataFusion.
///
/// DataFusion copies a CTE or view at each reference, so one GPU function
/// invocation can reach several host boundaries. Binding gives every copy with
/// the same producer identity and host plan one occurrence id and records the
/// number of these readers. When there are several, the first reader to poll
/// drives one execution of the input, which begins and completes the single
/// manifest entry, and every reader replays the retained Arrow batches. A
/// reader that a consumer polls before another reader has drained still
/// drives the shared execution, so no reader waits on the order in which the
/// consumer polls its inputs.
#[derive(Debug)]
pub(crate) struct GpuFunctionHostOccurrenceExec {
    input: Arc<dyn ExecutionPlan>,
    occurrence_id: Arc<str>,
    producer_identity: Arc<str>,
    readers: usize,
    account_memory_pool: bool,
    properties: Arc<PlanProperties>,
}

impl GpuFunctionHostOccurrenceExec {
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        occurrence_id: impl Into<Arc<str>>,
        producer_identity: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            properties: Arc::clone(input.properties()),
            input,
            occurrence_id: occurrence_id.into(),
            producer_identity: producer_identity.into(),
            readers: 1,
            account_memory_pool: false,
        }
    }

    /// Marks this node as one of `readers` final-plan nodes bound to the same
    /// occurrence. With more than one reader, the retained batches are
    /// charged to the DataFusion memory pool when `account_memory_pool` is
    /// set, the policy native host output follows.
    pub(crate) fn with_readers(mut self, readers: usize, account_memory_pool: bool) -> Self {
        self.readers = readers;
        self.account_memory_pool = account_memory_pool;
        self
    }

    pub(crate) fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub(crate) fn occurrence_id(&self) -> &str {
        &self.occurrence_id
    }

    pub(crate) fn producer_identity(&self) -> &str {
        &self.producer_identity
    }

    pub(crate) fn readers(&self) -> usize {
        self.readers
    }

    fn execute_shared(
        &self,
        task_context: Arc<TaskContext>,
        results: &SharedHostOccurrenceResults,
    ) -> SendableRecordBatchStream {
        let evaluation = results.reader_evaluation(&self.occurrence_id, self.readers);
        let input = Arc::clone(&self.input);
        let account_memory_pool = self.account_memory_pool;
        // The reader that polls first runs the input. When it is dropped
        // before the result is ready, the next reader to poll starts again.
        let output = async move {
            evaluation.get_or_init(|| evaluate_host_result(input, task_context, account_memory_pool)).await.clone()
        };
        let batches = stream::once(output).flat_map(|result| match result {
            Ok(retained) => {
                stream::iter((0..retained.batches.len()).map(move |index| Ok(retained.batches[index].clone())))
                    .left_stream()
            },
            Err(error) => stream::iter([Err(DataFusionError::Shared(error))]).right_stream(),
        });
        Box::pin(RecordBatchStreamAdapter::new(self.input.schema(), batches))
    }
}

async fn evaluate_host_result(
    input: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
    account_memory_pool: bool,
) -> SharedHostOutput {
    let reservation = account_memory_pool.then(|| {
        MemoryConsumer::new(SHARED_HOST_RESULT_MEMORY_CONSUMER)
            .with_can_spill(false)
            .register(task_context.memory_pool())
    });
    let mut batches = Vec::new();
    let collected = async {
        let mut input = input.execute(0, task_context)?;
        while let Some(batch) = input.next().await {
            let batch = batch?;
            if let Some(reservation) = &reservation {
                reservation.try_grow(get_record_batch_memory_size(&batch))?;
            }
            batches.push(batch);
        }
        Ok::<_, DataFusionError>(())
    }
    .await;
    match collected {
        Ok(()) => Ok(Arc::new(RetainedHostResult { batches, _reservation: reservation })),
        Err(error) => Err(Arc::new(error)),
    }
}

/// The Arrow result of one shared occurrence, retained until the last reader
/// stream releases it.
struct RetainedHostResult {
    batches: Vec<RecordBatch>,
    _reservation: Option<MemoryReservation>,
}

type SharedHostOutput = std::result::Result<Arc<RetainedHostResult>, Arc<DataFusionError>>;

struct PendingReaders {
    evaluation: Arc<OnceCell<SharedHostOutput>>,
    remaining: usize,
}

/// Shared host occurrence evaluations of one DataFusion execution of a native
/// admission root, by occurrence id.
///
/// An entry leaves the map once its last reader has taken the evaluation, so
/// the retained batches live as long as a reader stream holds them. Entries of
/// readers that never execute drop with the execution. An entry holds no task
/// context, which keeps it from owning the execution that owns this map.
#[derive(Default)]
pub(crate) struct SharedHostOccurrenceResults {
    pending: Mutex<HashMap<Arc<str>, PendingReaders>>,
}

impl std::fmt::Debug for SharedHostOccurrenceResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("SharedHostOccurrenceResults").field("pending_occurrences", &pending.len()).finish()
    }
}

impl SharedHostOccurrenceResults {
    /// The evaluation every one of the `readers` of `occurrence_id` awaits.
    fn reader_evaluation(&self, occurrence_id: &Arc<str>, readers: usize) -> Arc<OnceCell<SharedHostOutput>> {
        let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = pending
            .entry(Arc::clone(occurrence_id))
            .or_insert_with(|| PendingReaders { evaluation: Arc::default(), remaining: readers });
        let evaluation = Arc::clone(&entry.evaluation);
        entry.remaining = entry.remaining.saturating_sub(1);
        if entry.remaining == 0 {
            pending.remove(occurrence_id);
        }
        evaluation
    }
}

pub(crate) fn gpu_function_occurrence_from_task_context(
    task_context: &TaskContext,
    producer_identity: &str,
) -> Result<Option<String>> {
    let Some(active) = task_context.session_config().get_extension::<ActiveGpuFunctionOccurrence>() else {
        return Ok(None);
    };
    if active.producer_identity.as_ref() != producer_identity {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native_exec.gpu_function_occurrence",
            "a GPU function executed under another producer's occurrence lineage",
        )
        .with_log_safe_fact("reason", "gpu_function_occurrence_producer_mismatch")
        .with_log_safe_fact("expected_producer_identity", producer_identity)
        .with_log_safe_fact("actual_producer_identity", active.producer_identity.as_ref()));
    }
    Ok(Some(active.occurrence_id.to_string()))
}

fn task_context_with_gpu_function_occurrence(
    task_context: Arc<TaskContext>,
    occurrence_id: Arc<str>,
    producer_identity: Arc<str>,
) -> Arc<TaskContext> {
    let mut session_config = task_context.session_config().clone();
    session_config.set_extension(Arc::new(ActiveGpuFunctionOccurrence { occurrence_id, producer_identity }));
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

impl DisplayAs for GpuFunctionHostOccurrenceExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuFunctionHostOccurrenceExec: occurrence_id={}", self.occurrence_id)?;
        if self.readers > 1 {
            write!(f, ", readers={}", self.readers)?;
        }
        Ok(())
    }
}

impl ExecutionPlan for GpuFunctionHostOccurrenceExec {
    fn name(&self) -> &str {
        "GpuFunctionHostOccurrenceExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [input] = children.try_into().map_err(|children: Vec<Arc<dyn ExecutionPlan>>| {
            DataFusionError::Internal(format!("GPU function occurrence expected one child, got {}", children.len()))
        })?;
        Ok(Arc::new(
            Self::new(input, Arc::clone(&self.occurrence_id), Arc::clone(&self.producer_identity))
                .with_readers(self.readers, self.account_memory_pool),
        ))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        let shared_results = (self.readers > 1 && partition == 0)
            .then(|| shared_host_occurrence_results(task_context.as_ref()))
            .flatten();
        let task_context = task_context_with_gpu_function_occurrence(
            task_context,
            Arc::clone(&self.occurrence_id),
            Arc::clone(&self.producer_identity),
        );
        match shared_results {
            Some(results) => Ok(self.execute_shared(task_context, &results)),
            None => self.input.execute(partition, task_context),
        }
    }
}
