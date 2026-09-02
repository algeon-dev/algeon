use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use arrow::record_batch::RecordBatch;

use super::{
    CompiledGpuResource,
    helpers::{invalid, task_diagnostic_resident_resource},
    resource::CompiledGpuResourcePayload,
    scheduler::{CompiledSchedulingCycle, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput},
    state::CompiledPipelineState,
};
use crate::{
    dynamic_filter::{DynamicFilterId, DynamicFilterSet},
    error::{QueryError, QueryErrorCode, Result},
    exec::bounded::{
        BoundedCompiledJoinBuildState, BoundedCompiledJoinProbeState, BoundedExecutionAttempt, ReservationTracker,
        ResidentGpuFrame, release_bounded_compiled_join_build_state, release_bounded_compiled_join_probe_state,
    },
    metrics::QueryMetrics,
    pipeline::{CompiledPipeline, PipelineTask, PipelineTaskKind, ResidentResourceId, SinkBoundaryKind},
    plan::{NodeId, QueryPlan},
    runtime::{
        ExecutionContext,
        resident::{GpuResidentInputMap, GpuResultHandle},
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
        QueryErrorCode::CudfResourceExhausted
            | QueryErrorCode::CudfTemporaryFailure
            | QueryErrorCode::MemoryReservationFailed
            | QueryErrorCode::ServiceOverloaded
    )
}

/// Compiled task interpreter for bounded GPU work.
#[doc(hidden)]
pub struct CompiledGpuTaskExecutor<'a> {
    pub(super) execution: BoundedExecutionAttempt<'a>,
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
        metrics: &'a mut QueryMetrics,
        observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
    ) -> Result<Self> {
        let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
            return Err(invalid("compiled bounded task execution requires bounded memory control")
                .with_fact("memory_control_policy", context.memory_control_policy().as_str()));
        };
        let execution = BoundedExecutionAttempt::new(context, device_budget_bytes, metrics, observation_scope)?;
        let gpu_inputs = execution.reservations().augment_gpu_inputs(gpu_inputs)?;
        Ok(Self {
            execution,
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

    pub fn reserved_bytes(&self) -> u64 {
        self.execution.reservations().reserved_bytes()
    }

    pub fn remaining_bytes(&self) -> u64 {
        self.execution.reservations().remaining_bytes()
    }

    pub(in crate::exec::bounded) fn into_reservations(self) -> ReservationTracker {
        let Self { gpu_inputs, execution, .. } = self;
        // The post-execution invariant must observe the tracker after all augmented input guards release.
        drop(gpu_inputs);
        execution.into_reservations()
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
        let ResidentGpuFrame { frame, mut reservation } = *frame;
        let live_reservation = reservation.transfer_grant_live_reservation();
        let frame =
            frame.with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(self.plan));
        let handle = match GpuResultHandle::new_with_live_reservation(
            self.execution.context(),
            frame,
            self.plan.output_schema().clone(),
            live_reservation,
        ) {
            Ok(handle) => handle,
            Err(error) => {
                self.execution.release(reservation);
                return Err(error
                    .with_fact_if_absent("sink_boundary_task_id", self.pipeline.sink().input())
                    .with_fact_if_absent("resident_resource_id", resource)
                    .with_fact_if_absent("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
            },
        };
        self.execution.release(reservation);
        Ok(handle)
    }
}

pub(super) fn string_floor_max_bytes(data_type: &arrow::datatypes::DataType) -> Option<u64> {
    match data_type {
        arrow::datatypes::DataType::Utf8
        | arrow::datatypes::DataType::LargeUtf8
        | arrow::datatypes::DataType::Utf8View => Some(0),
        _ => None,
    }
}

impl CompiledTaskExecutor for CompiledGpuTaskExecutor<'_> {
    type Resource = CompiledGpuResource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskOutput<Self::Resource>> {
        let attempt_id = crate::observability::task_attempt_id(task.id().to_string(), 1);
        let span_id = crate::observability::RuntimeSpanId::new((task.id().index() as u64).saturating_add(2));
        self.execute_task_attempt(task, state, &attempt_id, span_id)
    }

    fn execute_task_attempt(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
        attempt_id: &crate::observability::TaskAttemptId,
        span_id: crate::observability::RuntimeSpanId,
    ) -> Result<CompiledTaskOutput<Self::Resource>> {
        self.execution.context().check_cancellation("nexus_query_engine.exec.bounded.compiled.task")?;
        self.execution.metrics_mut().record_compiled_pipeline_task_mapping(
            task.id(),
            task.kind().as_str(),
            task.plan_node(),
        );
        self.execution.metrics_mut().record_compiled_pipeline_task_execution(task.id(), task.kind().as_str());
        let _task_observation_scope = self.execution.reservations().enter_task_observation_scope(
            attempt_id.clone(),
            task.kind().as_str(),
            span_id,
            task_diagnostic_resident_resource(task),
        );
        let budget_event_start = self.execution.metrics_mut().budgeted_execution_event_count();
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
        self.execution.metrics_mut().record_task_budget_observations_since(task.id(), budget_event_start);
        self.execution.metrics_mut().record_task_elapsed(task.id(), task.kind().as_str(), timer.elapsed());
        result
    }

    fn release_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        let _resource_observation_scope = self.execution.reservations().enter_resource_observation_scope(resource);
        match payload.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                self.execution.release(frame.reservation);
            },
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
        let _cleanup_observation_scope = self
            .execution
            .reservations()
            .enter_release_observation_scope(crate::observability::RuntimeReservationDisposition::CleanupRelease);
        self.release_resource(resource, payload)
    }

    fn cleanup_after_error(&mut self) -> Result<()> {
        let _cleanup_observation_scope = self
            .execution
            .reservations()
            .enter_release_observation_scope(crate::observability::RuntimeReservationDisposition::CleanupRelease);
        while let Some((_, probe)) = self.join_probes.pop_first() {
            release_bounded_compiled_join_probe_state(&mut self.execution, probe);
        }
        while let Some((_, build)) = self.join_builds.pop_first() {
            release_bounded_compiled_join_build_state(&mut self.execution, build);
        }
        Ok(())
    }

    fn before_scheduling_cycle(&mut self) -> Result<CompiledSchedulingCycle> {
        Ok(CompiledSchedulingCycle)
    }

    fn live_reserved_bytes(&self) -> Option<u64> {
        Some(self.reserved_bytes())
    }

    fn remaining_reserved_bytes(&self) -> Option<u64> {
        Some(self.remaining_bytes())
    }

    fn can_admit_task(
        &self,
        task: &PipelineTask,
        state: &CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskAdmission> {
        self.can_admit_with_current_reservations(task, state)
    }
}
