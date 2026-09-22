use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

use arrow::record_batch::RecordBatch;

use super::{
    CompiledGpuResource,
    helpers::invalid,
    resource::CompiledGpuResourcePayload,
    scheduler::{CompiledTaskExecutor, CompiledTaskOutput},
    state::CompiledPipelineState,
};
use crate::{
    admission::canonical_execution_scope_fingerprint,
    dynamic_filter::{DynamicFilterId, DynamicFilterSet},
    error::{QueryError, QueryErrorCode, Result},
    exec::{
        ExecutionMetrics,
        bounded::{
            BoundedCompiledJoinBuildState, BoundedCompiledJoinProbeState, BoundedExecutionAttempt, ResidentGpuFrame,
            release_bounded_compiled_join_probe_state,
        },
    },
    native_error::cudf_query_error,
    pipeline::{CompiledPipeline, PipelineTask, PipelineTaskKind, ResidentResourceId, SinkBoundaryKind},
    plan::{NodeId, QueryPlan},
    runtime::{
        ExecutionContext,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
};

mod admission;
mod aggregate;
mod dynamic_filters;
mod edge_normalize;
mod graph;
mod join;
mod routing;
mod source;
mod terminal;
mod unary;

pub(super) fn join_build_key_uniqueness_probe_can_fallback(error: &QueryError) -> bool {
    matches!(
        error.code(),
        QueryErrorCode::QueryMemoryLimitExceeded
            | QueryErrorCode::CudfResourceExhausted
            | QueryErrorCode::CudfTemporaryFailure
            | QueryErrorCode::MemoryReservationFailed
            | QueryErrorCode::ServiceOverloaded
    )
}

/// Compiled task interpreter for bounded GPU work.
#[doc(hidden)]
pub struct CompiledGpuTaskExecutor<'a> {
    pub(super) execution: BoundedExecutionAttempt<'a>,
    pub(super) aggregate_dispatch: Option<super::lanes::AggregateLaneDispatch>,
    pub(super) plan: &'a QueryPlan,
    pub(super) pipeline: &'a CompiledPipeline,
    pub(super) gpu_inputs: GpuResidentInputMap,
    pub(super) dynamic_filters: DynamicFilterSet,
    pub(super) source_post_decode_applied_routes: BTreeMap<ResidentResourceId, BTreeSet<DynamicFilterId>>,
    pub(super) join_builds: BTreeMap<NodeId, BoundedCompiledJoinBuildState>,
    pub(super) join_probes: BTreeMap<NodeId, BoundedCompiledJoinProbeState>,
    pub(super) host_output: Option<RecordBatch>,
}

impl<'a> CompiledGpuTaskExecutor<'a> {
    pub fn new(
        context: &'a ExecutionContext,
        plan: &'a QueryPlan,
        pipeline: &'a CompiledPipeline,
        gpu_inputs: &'a GpuResidentInputMap,
        metrics: &'a ExecutionMetrics,
    ) -> Result<Self> {
        let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
            return Err(invalid("compiled bounded task execution requires bounded memory control")
                .with_fact("memory_control_policy", context.memory_control_policy().as_str()));
        };
        let execution = BoundedExecutionAttempt::new(context, device_budget_bytes, metrics)?;
        let gpu_inputs = gpu_inputs.clone();
        Ok(Self {
            execution,
            aggregate_dispatch: None,
            plan,
            pipeline,
            gpu_inputs,
            dynamic_filters: DynamicFilterSet::new(),
            source_post_decode_applied_routes: BTreeMap::new(),
            join_builds: BTreeMap::new(),
            join_probes: BTreeMap::new(),
            host_output: None,
        })
    }

    /// Merges a successful lane's dynamic-filter set into the coordinator.
    pub(super) fn merge_lane_dynamic_filters(&mut self, filters: DynamicFilterSet) {
        self.dynamic_filters.merge_from(&filters);
    }

    pub fn take_host_output(&mut self) -> Result<RecordBatch> {
        self.host_output.take().ok_or_else(|| {
            invalid("compiled host sink did not produce a host Arrow output")
                .with_fact("compiled_task_id", self.pipeline.sink().input())
                .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str())
        })
    }

    pub fn take_gpu_output(
        &mut self,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<GpuResultHandle> {
        if self.pipeline.sink().kind() != SinkBoundaryKind::GpuResidentOutput {
            return Err(invalid("compiled GPU output requires a GPU-resident sink boundary")
                .with_fact("sink_boundary_task_id", self.pipeline.sink().input())
                .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
        }
        let output = state.take_sink_output_resource(self.pipeline)?;
        let resource = output.id();
        let payload = output.into_payload();
        let frame = match payload.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame,
            other => {
                <Self as CompiledTaskExecutor>::release_resource(
                    self,
                    resource,
                    CompiledGpuResource { payload: other },
                )?;
                return Err(invalid("compiled GPU sink output must be a resident frame")
                    .with_fact("sink_boundary_task_id", self.pipeline.sink().input())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
            },
        };
        let ResidentGpuFrame { frame, .. } = *frame;
        let frame = if self.aggregate_dispatch.is_some() {
            // Lane leases remain live through this handoff; the resident result
            // must not retain columns backed by a leased worker stream. The
            // adoption runs whenever the run owns a dispatcher, even if this
            // task fell back to serial work: lane-backed content can reach
            // the sink through sibling tasks sharing the same run (join probe
            // chunks, aggregate source chunks), and tracking every producer
            // is more coupling than one output-sized copy.
            let root_runtime = self.execution.context().runtime()?;
            let adopted = frame.frame().copy_into_runtime(&root_runtime).map_err(|error| {
                cudf_query_error(
                    "query_engine.compiled_gpu_sink.root_adoption",
                    "failed to adopt aggregate output onto the root query runtime",
                    error,
                )
            })?;
            GpuDataFrame::try_new_with_source_residency(adopted, &frame)?
        } else {
            frame
        };
        let output_bytes = frame.device_allocated_size_bytes()?;
        let resident_owner = self.execution.context().retain_resident_output_owner(output_bytes)?;
        let frame = frame.with_producer_execution_scope(canonical_execution_scope_fingerprint(self.plan));
        let handle = match GpuResultHandle::new_with_resident_owner(
            self.execution.context(),
            frame,
            Arc::clone(self.plan.output_schema()),
            resident_owner,
        ) {
            Ok(handle) => handle.with_producer_execution_plan(self.plan.clone()),
            Err(error) => {
                return Err(error
                    .with_fact_if_absent("sink_boundary_task_id", self.pipeline.sink().input())
                    .with_fact_if_absent("resident_resource_id", resource)
                    .with_fact_if_absent("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
            },
        };
        Ok(handle)
    }
}

impl CompiledTaskExecutor for CompiledGpuTaskExecutor<'_> {
    type Resource = CompiledGpuResource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskOutput<Self::Resource>> {
        self.execution.context().check_cancellation("query_engine.exec.bounded.compiled.task")?;
        self.execution.metrics().record_compiled_pipeline_task_mapping(
            task.id(),
            task.kind().as_str(),
            task.plan_node(),
        );
        self.execution.metrics().record_compiled_pipeline_task_execution(task.id(), task.kind().as_str());
        let budget_window = self.execution.metrics().open_budget_window();
        let timer = Instant::now();
        let result = match task.kind() {
            PipelineTaskKind::SourceRead
            | PipelineTaskKind::GpuResidentSource
            | PipelineTaskKind::LiteralBatchSource => self.execute_source_task(task),
            PipelineTaskKind::Unary => self.execute_unary_task(task, state),
            PipelineTaskKind::JoinBuild => self.execute_join_build_task(task, state),
            PipelineTaskKind::JoinProbe => self.execute_join_probe_task(task, state),
            PipelineTaskKind::JoinMaterialize => self.execute_join_materialize_task(task),
            PipelineTaskKind::Aggregate => self.execute_aggregate_task(task, state),
            PipelineTaskKind::Window => self.execute_window_task(task, state),
            PipelineTaskKind::FinalSort => self.execute_final_sort_task(task, state),
            PipelineTaskKind::Limit => self.execute_limit_task(task, state),
            PipelineTaskKind::Union => self.execute_union_task(task, state),
            PipelineTaskKind::EdgeNormalize => self.execute_edge_normalize_task(task, state),
            PipelineTaskKind::GraphAlgorithm => self.execute_graph_algorithm_task(task, state),
            PipelineTaskKind::SinkMaterialize => self.execute_sink_materialize_task(task, state),
        };
        self.execution.metrics().close_budget_window(task.id(), budget_window);
        self.execution.metrics().record_task_elapsed(task.id(), task.kind().as_str(), timer.elapsed());
        result
    }

    fn release_resource(&mut self, _resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        match payload.payload {
            CompiledGpuResourcePayload::ResidentFrame { .. } => {},
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                stream.release_all(&mut self.execution);
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {},
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => {
                prefetch.abandon(&mut self.execution);
            },
        }
        Ok(())
    }

    fn cleanup_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        self.release_resource(resource, payload)
    }

    fn cleanup_after_error(&mut self) -> Result<()> {
        while let Some((_, probe)) = self.join_probes.pop_first() {
            release_bounded_compiled_join_probe_state(&mut self.execution, probe);
        }
        self.join_builds.clear();
        Ok(())
    }

    fn before_scheduling_cycle(&mut self) -> Result<()> {
        Ok(())
    }
}
