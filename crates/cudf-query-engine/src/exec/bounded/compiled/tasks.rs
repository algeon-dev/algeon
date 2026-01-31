use super::{
    reservations::validate_task_runtime_gates,
    scheduler::{CompiledTaskExecutor, CompiledTaskOutput},
    state::CompiledPipelineState,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    exec::{
        bounded::{
            ReservationTracker, ResidentGpuFrame, execute_bounded_generic_gpu_resident_source,
            execute_bounded_generic_local_parquet_source, execute_bounded_generic_unary_from_frame,
        },
        types::{GpuDataFrame, GpuResidentInputMap},
    },
    metrics::QueryMetrics,
    pipeline::{CompiledPipeline, PipelineTask, PipelineTaskKind, ResidentResourceId},
    plan::{NodeId, PlanNode, QueryPlan},
    runtime::{ExecutionContext, MemoryControlPolicy},
    source::SourceNode,
};

const OPERATION: &str = "cudf_query_engine.exec.bounded.compiled.tasks";

/// GPU-resident payload stored in the compiled scheduler resource table.
#[doc(hidden)]
pub struct CompiledGpuResource {
    frame: ResidentGpuFrame,
}

impl CompiledGpuResource {
    pub(in crate::exec::bounded) fn new(frame: ResidentGpuFrame) -> Self {
        Self { frame }
    }

    pub fn num_rows(&self) -> usize {
        self.frame.frame.num_rows()
    }

    pub fn num_columns(&self) -> usize {
        self.frame.frame.num_columns()
    }

    pub(in crate::exec::bounded) fn clone_frame(&self) -> GpuDataFrame {
        self.frame.frame.clone()
    }

    pub(in crate::exec::bounded) fn reservation_bytes(&self) -> u64 {
        self.frame.reservation.bytes()
    }

    pub(in crate::exec::bounded) fn into_resident_frame(self) -> ResidentGpuFrame {
        self.frame
    }
}

/// D.1 compiled task interpreter for source and unary GPU work.
#[doc(hidden)]
pub struct CompiledGpuTaskExecutor<'a> {
    context: &'a ExecutionContext,
    plan: &'a QueryPlan,
    pipeline: &'a CompiledPipeline,
    gpu_inputs: &'a GpuResidentInputMap,
    metrics: &'a mut QueryMetrics,
    reservations: ReservationTracker,
}

impl<'a> CompiledGpuTaskExecutor<'a> {
    pub fn new(
        context: &'a ExecutionContext,
        plan: &'a QueryPlan,
        pipeline: &'a CompiledPipeline,
        gpu_inputs: &'a GpuResidentInputMap,
        metrics: &'a mut QueryMetrics,
    ) -> Result<Self> {
        let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
            return Err(invalid("compiled bounded task execution requires bounded memory control")
                .with_fact("memory_control_policy", context.memory_control_policy().as_str()));
        };
        let reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
        Ok(Self { context, plan, pipeline, gpu_inputs, metrics, reservations })
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.reservations.reserved_bytes()
    }

    fn execute_source_task(&mut self, task: &PipelineTask) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled source task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let frame = match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => execute_bounded_generic_local_parquet_source(
                    self.context,
                    source,
                    node_id,
                    &mut self.reservations,
                    self.metrics,
                ),
                SourceNode::GpuResident(source) => execute_bounded_generic_gpu_resident_source(
                    self.context,
                    source,
                    node_id,
                    self.gpu_inputs,
                    &mut self.reservations,
                    self.metrics,
                ),
            },
            _ => Err(invalid("compiled source task does not reference a source plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    fn execute_unary_task(
        &mut self,
        task: &PipelineTask,
        state: &CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = *task.input_resources().first().ok_or_else(|| {
            invalid("compiled unary task has no input resource")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        let output = single_output_resource(task)?;
        let input_payload = state.resource_payload(input)?;
        let input_frame = input_payload.clone_frame();
        let input_reserved_bytes = input_payload.reservation_bytes();
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled unary task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let frame = match &node.kind {
            PlanNode::Filter(filter) => execute_bounded_generic_unary_from_frame(
                self.context,
                node_id,
                "compiled_filter",
                "compiled_filter_output",
                input_frame,
                input_reserved_bytes,
                &mut self.reservations,
                self.metrics,
                true,
                |context, input, token, metrics| {
                    crate::exec::operators::execute_filter_with_budget(
                        context,
                        filter,
                        input,
                        token.expect("budget token is created for compiled bounded filter stages"),
                        metrics,
                    )
                },
            ),
            PlanNode::Projection(projection) => execute_bounded_generic_unary_from_frame(
                self.context,
                node_id,
                "compiled_projection",
                "compiled_projection_output",
                input_frame,
                input_reserved_bytes,
                &mut self.reservations,
                self.metrics,
                true,
                |context, input, token, metrics| {
                    crate::exec::operators::execute_projection_with_budget(
                        context,
                        projection,
                        input,
                        token.expect("budget token is created for compiled bounded projection stages"),
                        metrics,
                    )
                },
            ),
            _ => Err(invalid("compiled unary task does not reference a filter or projection plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }
}

impl CompiledTaskExecutor for CompiledGpuTaskExecutor<'_> {
    type Resource = CompiledGpuResource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskOutput<Self::Resource>> {
        validate_task_runtime_gates(self.pipeline, task)?;
        match task.kind() {
            PipelineTaskKind::SourceRead | PipelineTaskKind::GpuResidentSource => self.execute_source_task(task),
            PipelineTaskKind::Unary => self.execute_unary_task(task, state),
            PipelineTaskKind::Aggregate
            | PipelineTaskKind::JoinBuild
            | PipelineTaskKind::JoinProbe
            | PipelineTaskKind::JoinMaterialize
            | PipelineTaskKind::FinalSort
            | PipelineTaskKind::Limit
            | PipelineTaskKind::SinkMaterialize => {
                Err(invalid("compiled task kind is outside the D.1.3 source/unary execution envelope")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("pipeline_task_kind", task.kind().as_str()))
            },
        }
    }

    fn release_resource(&mut self, _resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        let frame = payload.into_resident_frame();
        self.reservations.release(frame.reservation);
        Ok(())
    }
}

fn plan_node(task: &PipelineTask) -> Result<NodeId> {
    task.plan_node().ok_or_else(|| {
        invalid("compiled executable task is missing its plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("pipeline_task_kind", task.kind().as_str())
    })
}

fn single_output_resource(task: &PipelineTask) -> Result<ResidentResourceId> {
    if task.output_resources().len() != 1 {
        return Err(invalid("compiled executable task must declare exactly one output resource")
            .with_fact("compiled_task_id", task.id())
            .with_fact("pipeline_task_kind", task.kind().as_str())
            .with_fact("output_resource_count", task.output_resources().len()));
    }
    Ok(task.output_resources()[0])
}

fn attach_task_facts(error: QueryError, task: &PipelineTask, node_id: NodeId) -> QueryError {
    error
        .with_fact_if_absent("compiled_task_id", task.id())
        .with_fact_if_absent("pipeline_task_kind", task.kind().as_str())
        .with_fact_if_absent("plan_node_id", node_id)
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
