//! Final-plan binding for one host-materializing GPU function occurrence.

use std::sync::Arc;

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::PhysicalExpr,
    physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream},
};

use crate::error::{Error, ErrorCode};

#[derive(Debug, Clone)]
struct ActiveGpuFunctionOccurrence {
    occurrence_id: Arc<str>,
    producer_identity: Arc<str>,
}

/// A transparent final-plan node that binds a shared physical producer to one
/// concrete invocation occurrence. Its task-context lineage lets runtime
/// dependency validation use the physical path selected by DataFusion.
#[derive(Debug)]
pub(crate) struct GpuFunctionHostOccurrenceExec {
    input: Arc<dyn ExecutionPlan>,
    occurrence_id: Arc<str>,
    producer_identity: Arc<str>,
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
        }
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
}

pub(crate) fn gpu_function_occurrence_from_task_context(
    task_context: &TaskContext,
    producer_identity: &str,
) -> DataFusionResult<Option<String>> {
    let Some(active) = task_context.session_config().get_extension::<ActiveGpuFunctionOccurrence>() else {
        return Ok(None);
    };
    if active.producer_identity.as_ref() != producer_identity {
        return Err(DataFusionError::from(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_exec.gpu_function_occurrence",
                "a GPU function executed under another producer's occurrence lineage",
            )
            .with_log_safe_fact("reason", "gpu_function_occurrence_producer_mismatch")
            .with_log_safe_fact("expected_producer_identity", producer_identity)
            .with_log_safe_fact("actual_producer_identity", active.producer_identity.as_ref()),
        ));
    }
    Ok(Some(active.occurrence_id.to_string()))
}

pub(crate) fn gpu_function_occurrence_wrapped_input(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    plan.downcast_ref::<GpuFunctionHostOccurrenceExec>().map(GpuFunctionHostOccurrenceExec::input)
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
        write!(f, "GpuFunctionHostOccurrenceExec: occurrence_id={}", self.occurrence_id)
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
        Ok(Arc::new(Self::new(input, Arc::clone(&self.occurrence_id), Arc::clone(&self.producer_identity))))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        self.input.execute(
            partition,
            task_context_with_gpu_function_occurrence(
                task_context,
                Arc::clone(&self.occurrence_id),
                Arc::clone(&self.producer_identity),
            ),
        )
    }
}
