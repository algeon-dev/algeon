use std::collections::{BTreeMap, BTreeSet};

use arrow::record_batch::RecordBatch;

use super::{
    join_strategy::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection},
    reservations::validate_task_runtime_gates,
    scheduler::{CompiledTaskExecutor, CompiledTaskOutput},
    state::CompiledPipelineState,
};
use crate::{
    capability::{KeyedHashJoinStrategyInputFacts, keyed_hash_join_strategy_input_facts_for_plan},
    error::{QueryError, QueryErrorCode, Result},
    exec::{
        bounded::{
            ActiveReservation, BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput,
            BoundedCompiledJoinProbeState, BoundedJoinLeafLocalParquetSourceStream, ReservationTracker,
            ResidentGpuFrame, execute_bounded_compiled_join_build_from_frame,
            execute_bounded_compiled_join_build_from_resident_frame, execute_bounded_compiled_join_materialize,
            execute_bounded_compiled_join_materialize_preserving_stream,
            execute_bounded_compiled_join_probe_from_frame, execute_bounded_compiled_join_probe_from_frame_stream,
            execute_bounded_compiled_join_probe_from_resident_frame, execute_bounded_compiled_join_probe_from_stream,
            execute_bounded_generic_gpu_resident_source, execute_bounded_generic_local_parquet_source,
            execute_bounded_generic_unary, execute_bounded_generic_unary_from_frame,
            execute_bounded_generic_unary_from_frame_with_reservation_kind, execute_bounded_join_leaf_source,
            execute_compiled_bounded_aggregate_from_frame, execute_compiled_bounded_aggregate_from_stream,
            execute_final_sort_node_from_committed_with_retry, execute_final_sort_node_from_frame_with_retry,
            materialize_bounded_join_leaf_local_parquet_source_stream,
            plan_bounded_join_leaf_local_parquet_source_stream,
        },
        materialize_bounded_frame_to_host,
        types::{GpuDataFrame, GpuResidentInputMap},
    },
    metrics::QueryMetrics,
    pipeline::{
        CompiledPipeline, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentResourceId,
        SinkBoundaryKind,
    },
    plan::{JoinNode, JoinType, LimitNode, NodeId, PlanNode, QueryPlan, ResultSink, SortNode},
    runtime::{ExecutionContext, MemoryControlPolicy, ReservationKind},
    source::SourceNode,
};

const OPERATION: &str = "cudf_query_engine.exec.bounded.compiled.tasks";

/// GPU-resident payload stored in the compiled scheduler resource table.
#[doc(hidden)]
pub struct CompiledGpuResource {
    payload: CompiledGpuResourcePayload,
}

enum CompiledGpuResourcePayload {
    ResidentFrame { frame: ResidentGpuFrame, applied_limit: Option<NodeId> },
    ResidentFrameStream(Vec<ResidentGpuFrame>),
    JoinLeafLocalParquetSource(BoundedJoinLeafLocalParquetSourceStream),
}

impl CompiledGpuResource {
    pub(in crate::exec::bounded) fn new(frame: ResidentGpuFrame) -> Self {
        Self { payload: CompiledGpuResourcePayload::ResidentFrame { frame, applied_limit: None } }
    }

    pub(in crate::exec::bounded) fn frame_with_applied_limit(frame: ResidentGpuFrame, limit: NodeId) -> Self {
        Self { payload: CompiledGpuResourcePayload::ResidentFrame { frame, applied_limit: Some(limit) } }
    }

    pub(in crate::exec::bounded) fn join_leaf_local_parquet_source(
        stream: BoundedJoinLeafLocalParquetSourceStream,
    ) -> Self {
        Self { payload: CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) }
    }

    pub(in crate::exec::bounded) fn frame_stream(frames: Vec<ResidentGpuFrame>) -> Self {
        Self { payload: CompiledGpuResourcePayload::ResidentFrameStream(frames) }
    }

    pub fn num_rows(&self) -> usize {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.frame.num_rows(),
            CompiledGpuResourcePayload::ResidentFrameStream(frames) => {
                frames.iter().map(|frame| frame.frame.num_rows()).sum()
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => 0,
        }
    }

    pub fn num_columns(&self) -> usize {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.frame.num_columns(),
            CompiledGpuResourcePayload::ResidentFrameStream(frames) => {
                frames.first().map_or(0, |frame| frame.frame.num_columns())
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => 0,
        }
    }

    pub(in crate::exec::bounded) fn clone_frame(&self) -> GpuDataFrame {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.frame.clone(),
            CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                unreachable!("resident frame streams must be consumed by stream-aware compiled tasks")
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                unreachable!("deferred join leaf sources must be consumed by join probe tasks")
            },
        }
    }

    pub(in crate::exec::bounded) fn reservation_bytes(&self) -> u64 {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.reservation.bytes(),
            CompiledGpuResourcePayload::ResidentFrameStream(frames) => {
                frames.iter().fold(0u64, |total, frame| total.saturating_add(frame.reservation.bytes()))
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => 0,
        }
    }

    pub(in crate::exec::bounded) fn reservation(&self) -> &ActiveReservation {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => &frame.reservation,
            CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                unreachable!("resident frame streams do not expose a single reservation")
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                unreachable!("deferred join leaf sources do not own a resident reservation")
            },
        }
    }

    fn as_join_leaf_local_parquet_source(&self) -> Option<&BoundedJoinLeafLocalParquetSourceStream> {
        match &self.payload {
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => Some(stream),
            CompiledGpuResourcePayload::ResidentFrame { .. } | CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                None
            },
        }
    }

    fn applied_limit(&self) -> Option<NodeId> {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { applied_limit, .. } => *applied_limit,
            CompiledGpuResourcePayload::ResidentFrameStream(_)
            | CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => None,
        }
    }
}

/// Compiled task interpreter for bounded GPU work.
#[doc(hidden)]
pub struct CompiledGpuTaskExecutor<'a> {
    context: &'a ExecutionContext,
    plan: &'a QueryPlan,
    pipeline: &'a CompiledPipeline,
    gpu_inputs: &'a GpuResidentInputMap,
    metrics: &'a mut QueryMetrics,
    reservations: ReservationTracker,
    join_builds: BTreeMap<NodeId, BoundedCompiledJoinBuildState>,
    join_probes: BTreeMap<NodeId, BoundedCompiledJoinProbeState>,
    host_output: Option<RecordBatch>,
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
        Ok(Self {
            context,
            plan,
            pipeline,
            gpu_inputs,
            metrics,
            reservations,
            join_builds: BTreeMap::new(),
            join_probes: BTreeMap::new(),
            host_output: None,
        })
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.reservations.reserved_bytes()
    }

    pub fn take_host_output(&mut self) -> Result<RecordBatch> {
        self.host_output.take().ok_or_else(|| {
            invalid("compiled host sink did not produce a host Arrow output")
                .with_fact("compiled_task_id", self.pipeline.sink().input())
                .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str())
        })
    }

    fn execute_source_task(&mut self, task: &PipelineTask) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled source task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let resource = match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source)
                    if !self.resource_has_multiple_consumers(output)
                        && (self.task_feeds_direct_join_input(task.id())
                            || self.task_feeds_any_join_input_through_unary_chain(task.id())) =>
                {
                    let has_source_adjacent_stages = self.task_feeds_any_join_input_through_unary_chain(task.id());
                    let stream = plan_bounded_join_leaf_local_parquet_source_stream(
                        self.context,
                        source,
                        node_id,
                        has_source_adjacent_stages,
                        self.metrics,
                    )?;
                    if has_source_adjacent_stages || stream.chunk_count() > 1 {
                        Ok(CompiledGpuResource::join_leaf_local_parquet_source(stream))
                    } else {
                        materialize_bounded_join_leaf_local_parquet_source_stream(
                            self.context,
                            stream,
                            &mut self.reservations,
                            self.metrics,
                        )
                        .map(CompiledGpuResource::new)
                    }
                },
                source if self.task_feeds_join_input(task.id()) => execute_bounded_join_leaf_source(
                    self.context,
                    source,
                    node_id,
                    self.gpu_inputs,
                    &mut self.reservations,
                    self.metrics,
                )
                .map(CompiledGpuResource::new),
                SourceNode::LocalParquet(source) => execute_bounded_generic_local_parquet_source(
                    self.context,
                    source,
                    node_id,
                    &mut self.reservations,
                    self.metrics,
                )
                .map(CompiledGpuResource::new),
                SourceNode::GpuResident(source) => execute_bounded_generic_gpu_resident_source(
                    self.context,
                    source,
                    node_id,
                    self.gpu_inputs,
                    &mut self.reservations,
                    self.metrics,
                )
                .map(CompiledGpuResource::new),
            },
            _ => Err(invalid("compiled source task does not reference a source plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, resource)]))
    }

    fn execute_unary_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = *task.input_resources().first().ok_or_else(|| {
            invalid("compiled unary task has no input resource")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled unary task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let frame = match &node.kind {
            PlanNode::Filter(filter) => {
                if state.can_consume_task_input_resource(task, input) {
                    let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
                    match input_payload.payload {
                        CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                            Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                                stream.with_filter_stage(node_id, filter.clone()),
                            ))
                        },
                        CompiledGpuResourcePayload::ResidentFrame { frame, .. } => execute_bounded_generic_unary(
                            self.context,
                            node_id,
                            "compiled_filter",
                            "compiled_filter_output",
                            frame,
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
                        )
                        .map(CompiledGpuResource::new),
                        CompiledGpuResourcePayload::ResidentFrameStream(frames) => {
                            let mut outputs = Vec::with_capacity(frames.len());
                            for frame in frames {
                                match execute_bounded_generic_unary(
                                    self.context,
                                    node_id,
                                    "compiled_filter",
                                    "compiled_filter_output",
                                    frame,
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
                                ) {
                                    Ok(output) => outputs.push(output),
                                    Err(error) => {
                                        release_resident_frame_stream_reverse(
                                            &mut self.reservations,
                                            outputs,
                                            self.metrics,
                                        );
                                        return Err(attach_task_facts(error, task, node_id));
                                    },
                                }
                            }
                            Ok(CompiledGpuResource::frame_stream(outputs))
                        },
                    }
                } else {
                    let input_payload = state.resource_payload(input)?;
                    if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                        Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                            stream.with_filter_stage(node_id, filter.clone()),
                        ))
                    } else {
                        execute_bounded_generic_unary_from_frame(
                            self.context,
                            node_id,
                            "compiled_filter",
                            "compiled_filter_output",
                            input_payload.clone_frame(),
                            input_payload.reservation_bytes(),
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
                        )
                        .map(CompiledGpuResource::new)
                    }
                }
            },
            PlanNode::Projection(projection) => {
                if state.can_consume_task_input_resource(task, input) {
                    let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
                    match input_payload.payload {
                        CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                            Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                                stream.with_projection_stage(node_id, projection.clone()),
                            ))
                        },
                        CompiledGpuResourcePayload::ResidentFrame { frame, .. } => execute_bounded_generic_unary(
                            self.context,
                            node_id,
                            "compiled_projection",
                            "compiled_projection_output",
                            frame,
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
                        )
                        .map(CompiledGpuResource::new),
                        CompiledGpuResourcePayload::ResidentFrameStream(frames) => {
                            let mut outputs = Vec::with_capacity(frames.len());
                            for frame in frames {
                                match execute_bounded_generic_unary(
                                    self.context,
                                    node_id,
                                    "compiled_projection",
                                    "compiled_projection_output",
                                    frame,
                                    &mut self.reservations,
                                    self.metrics,
                                    true,
                                    |context, input, token, metrics| {
                                        crate::exec::operators::execute_projection_with_budget(
                                            context,
                                            projection,
                                            input,
                                            token.expect(
                                                "budget token is created for compiled bounded projection stages",
                                            ),
                                            metrics,
                                        )
                                    },
                                ) {
                                    Ok(output) => outputs.push(output),
                                    Err(error) => {
                                        release_resident_frame_stream_reverse(
                                            &mut self.reservations,
                                            outputs,
                                            self.metrics,
                                        );
                                        return Err(attach_task_facts(error, task, node_id));
                                    },
                                }
                            }
                            Ok(CompiledGpuResource::frame_stream(outputs))
                        },
                    }
                } else {
                    let input_payload = state.resource_payload(input)?;
                    if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                        Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                            stream.with_projection_stage(node_id, projection.clone()),
                        ))
                    } else {
                        execute_bounded_generic_unary_from_frame(
                            self.context,
                            node_id,
                            "compiled_projection",
                            "compiled_projection_output",
                            input_payload.clone_frame(),
                            input_payload.reservation_bytes(),
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
                        )
                        .map(CompiledGpuResource::new)
                    }
                }
            },
            _ => Err(invalid("compiled unary task does not reference a filter or projection plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, frame)]))
    }

    fn execute_join_build_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        if self.join_builds.contains_key(&node_id) || self.join_probes.contains_key(&node_id) {
            return Err(invalid("compiled join build task found existing state for its join node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let input = single_input_resource(task)?;
        let node = join_node(self.plan, task, node_id)?;
        let build = if state.can_consume_task_input_resource(task, input) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            self.reservations.set_diagnostic_context(Some(task.id()), Some(input));
            let input_frame = match input_payload.payload {
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                    materialize_bounded_join_leaf_local_parquet_source_stream(
                        self.context,
                        stream,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrame { frame, .. } => Ok(frame),
                CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                    Err(invalid("compiled join build task cannot consume a resident frame stream")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id))
                },
            }?;
            let strategy = join_strategy(
                self.plan,
                node,
                task,
                node_id,
                input_frame.frame.num_rows() as u64,
                input_frame.reservation.bytes(),
            )?;
            if let Some(facts) = strategy.facts.as_ref() {
                self.metrics.record_join_strategy_decision(node_id, facts);
            }
            execute_bounded_compiled_join_build_from_resident_frame(
                self.context,
                node_id,
                node,
                strategy.strategy,
                input_frame,
                &mut self.reservations,
                self.metrics,
            )
        } else {
            let input_payload = state.resource_payload(input)?;
            if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                let input_frame = materialize_bounded_join_leaf_local_parquet_source_stream(
                    self.context,
                    stream,
                    &mut self.reservations,
                    self.metrics,
                )?;
                let strategy = join_strategy(
                    self.plan,
                    node,
                    task,
                    node_id,
                    input_frame.frame.num_rows() as u64,
                    input_frame.reservation.bytes(),
                )?;
                if let Some(facts) = strategy.facts.as_ref() {
                    self.metrics.record_join_strategy_decision(node_id, facts);
                }
                execute_bounded_compiled_join_build_from_resident_frame(
                    self.context,
                    node_id,
                    node,
                    strategy.strategy,
                    input_frame,
                    &mut self.reservations,
                    self.metrics,
                )
            } else {
                let strategy = join_strategy(
                    self.plan,
                    node,
                    task,
                    node_id,
                    input_payload.num_rows() as u64,
                    input_payload.reservation_bytes(),
                )?;
                if let Some(facts) = strategy.facts.as_ref() {
                    self.metrics.record_join_strategy_decision(node_id, facts);
                }
                execute_bounded_compiled_join_build_from_frame(
                    self.context,
                    node_id,
                    node,
                    strategy.strategy,
                    input_payload.clone_frame(),
                    input_payload.reservation(),
                    &mut self.reservations,
                    self.metrics,
                )
            }
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        self.join_builds.insert(node_id, build);
        Ok(CompiledTaskOutput::empty())
    }

    fn execute_join_probe_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let node = join_node(self.plan, task, node_id)?;
        let build = self.join_builds.remove(&node_id).ok_or_else(|| {
            invalid("compiled join probe task ran before join build state was available")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        if self.join_probes.contains_key(&node_id) {
            return Err(invalid("compiled join probe task found existing probe state for its join node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let probe = if state.can_consume_task_input_resource(task, input) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            self.reservations.set_diagnostic_context(Some(task.id()), Some(input));
            match input_payload.payload {
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                    execute_bounded_compiled_join_probe_from_stream(
                        self.context,
                        node_id,
                        node,
                        build,
                        stream,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        self.context,
                        node_id,
                        node,
                        build,
                        frame,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrameStream(frames) => {
                    execute_bounded_compiled_join_probe_from_frame_stream(
                        self.context,
                        node_id,
                        node,
                        build,
                        frames,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
            }
        } else {
            let input_payload = state.resource_payload(input)?;
            if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                execute_bounded_compiled_join_probe_from_stream(
                    self.context,
                    node_id,
                    node,
                    build,
                    stream,
                    &mut self.reservations,
                    self.metrics,
                )
            } else {
                execute_bounded_compiled_join_probe_from_frame(
                    self.context,
                    node_id,
                    node,
                    build,
                    input_payload.clone_frame(),
                    input_payload.reservation(),
                    &mut self.reservations,
                    self.metrics,
                )
            }
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        self.join_probes.insert(node_id, probe);
        Ok(CompiledTaskOutput::empty())
    }

    fn execute_join_materialize_task(
        &mut self,
        task: &PipelineTask,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let node = join_node(self.plan, task, node_id)?;
        let probe = self.join_probes.remove(&node_id).ok_or_else(|| {
            invalid("compiled join materialize task ran before join probe state was available")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        let resource = if self.join_materialize_output_can_preserve_stream(task.id(), output) {
            match execute_bounded_compiled_join_materialize_preserving_stream(
                self.context,
                node_id,
                node,
                probe,
                &mut self.reservations,
                self.metrics,
            )
            .map_err(|error| attach_task_facts(error, task, node_id))?
            {
                BoundedCompiledJoinMaterializeOutput::Frame(frame) => CompiledGpuResource::new(*frame),
                BoundedCompiledJoinMaterializeOutput::Stream(frames) => CompiledGpuResource::frame_stream(frames),
            }
        } else {
            let frame = execute_bounded_compiled_join_materialize(
                self.context,
                node_id,
                node,
                probe,
                &mut self.reservations,
                self.metrics,
            )
            .map_err(|error| attach_task_facts(error, task, node_id))?;
            CompiledGpuResource::new(frame)
        };
        Ok(CompiledTaskOutput::new(vec![(output, resource)]))
    }

    fn execute_aggregate_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled aggregate task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(invalid("compiled aggregate task does not reference an aggregate plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        };
        let can_consume_stream = state.can_consume_task_input_resource(task, input)
            && matches!(state.resource_payload(input)?.payload, CompiledGpuResourcePayload::ResidentFrameStream(_));
        let frame = if can_consume_stream {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            match input_payload.payload {
                CompiledGpuResourcePayload::ResidentFrameStream(frames) => {
                    execute_compiled_bounded_aggregate_from_stream(
                        self.context,
                        node_id,
                        aggregate,
                        self.plan
                            .node(aggregate.input)
                            .ok_or_else(|| {
                                invalid("compiled aggregate task references a missing aggregate input node")
                                    .with_fact("compiled_task_id", task.id())
                                    .with_fact("plan_node_id", aggregate.input)
                            })?
                            .output_schema(),
                        node.output_schema(),
                        frames,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                    Err(invalid("compiled aggregate task cannot consume a deferred join leaf source stream")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id))
                },
                CompiledGpuResourcePayload::ResidentFrame { .. } => {
                    unreachable!("only frame streams are consumed here")
                },
            }
        } else {
            let input_payload = state.resource_payload(input)?;
            execute_compiled_bounded_aggregate_from_frame(
                self.context,
                node_id,
                aggregate,
                node.output_schema(),
                input_payload.clone_frame(),
                input_payload.reservation_bytes(),
                &mut self.reservations,
                self.metrics,
            )
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    fn execute_final_sort_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let sort = sort_node(self.plan, task, node_id)?.clone();
        if let Some((limit_node_id, limit)) = self.downstream_fused_limit(task, output)?
            && state.can_consume_task_input_resource(task, input)
        {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            let CompiledGpuResourcePayload::ResidentFrame { frame: input_frame, .. } = input_payload.payload else {
                return Err(invalid("compiled final sort can only fuse limit over resident frame input")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id));
            };
            let frame = execute_final_sort_node_from_committed_with_retry(
                self.context,
                node_id,
                &sort,
                Some(&limit),
                input_frame,
                &mut self.reservations,
                self.metrics,
            )
            .map_err(|error| attach_task_facts(error, task, node_id))?;
            return Ok(CompiledTaskOutput::new(vec![(
                output,
                CompiledGpuResource::frame_with_applied_limit(frame, limit_node_id),
            )]));
        }

        let input_payload = state.resource_payload(input)?;
        let frame = execute_final_sort_node_from_frame_with_retry(
            self.context,
            node_id,
            &sort,
            input_payload.clone_frame(),
            &mut self.reservations,
            self.metrics,
        )
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    fn downstream_fused_limit(
        &self,
        task: &PipelineTask,
        output: ResidentResourceId,
    ) -> Result<Option<(NodeId, LimitNode)>> {
        let Some(resource) = self.pipeline.resources().iter().find(|resource| resource.id() == output) else {
            return Ok(None);
        };
        let [consumer] = resource.consumers() else {
            return Ok(None);
        };
        let Some(consumer_task) = self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer) else {
            return Err(invalid("compiled final sort output references a missing consumer task")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", output));
        };
        if consumer_task.kind() != PipelineTaskKind::Limit {
            return Ok(None);
        }
        let limit_node_id = plan_node(consumer_task)?;
        let Some(node) = self.plan.node(limit_node_id) else {
            return Err(invalid("compiled final sort downstream limit references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("limit_task_id", consumer_task.id())
                .with_fact("plan_node_id", limit_node_id));
        };
        let PlanNode::Limit(limit) = &node.kind else {
            return Err(invalid("compiled final sort downstream limit task does not reference a limit plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("limit_task_id", consumer_task.id())
                .with_fact("plan_node_id", limit_node_id));
        };
        if Some(limit.input) != task.plan_node() {
            return Ok(None);
        }
        Ok(Some((limit_node_id, limit.clone())))
    }

    fn execute_limit_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let limit = limit_node(self.plan, task, node_id)?;
        if state.resource_payload(input)?.applied_limit() == Some(node_id) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = input_payload.payload else {
                return Err(invalid("compiled fused limit can only pass through resident frame input")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id));
            };
            return Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]));
        }

        let input_payload = state.resource_payload(input)?;
        let frame = execute_bounded_generic_unary_from_frame_with_reservation_kind(
            self.context,
            node_id,
            "compiled_limit",
            "compiled_limit_output",
            input_payload.clone_frame(),
            input_payload.reservation_bytes(),
            ReservationKind::FinalSort,
            &mut self.reservations,
            self.metrics,
            false,
            |context, input, _token, metrics| crate::exec::operators::execute_limit(context, limit, input, metrics),
        )
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    fn execute_sink_materialize_task(
        &mut self,
        task: &PipelineTask,
        state: &CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        if self.pipeline.sink().kind() != SinkBoundaryKind::HostArrowMaterialization
            || self.pipeline.sink().input() != task.id()
        {
            return Err(invalid("compiled sink materialize task does not match the pipeline sink boundary")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str())
                .with_fact("sink_boundary_task_id", self.pipeline.sink().input())
                .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
        }
        if self.host_output.is_some() {
            return Err(invalid("compiled host sink materialized more than once")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let input = single_input_resource(task)?;
        let input_payload = state.resource_payload(input)?;
        let ResultSink::HostArrow { input: final_node_id, schema } = self.plan.sink() else {
            return Err(invalid("compiled sink materialize task requires a host Arrow sink")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        };
        let executable_schema = self
            .plan
            .node(*final_node_id)
            .ok_or_else(|| {
                invalid("compiled sink task references a missing final plan node")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", final_node_id)
            })?
            .output_schema();
        let output = materialize_bounded_frame_to_host(
            self.context,
            input_payload.clone_frame(),
            input_payload.reservation_bytes(),
            *final_node_id,
            executable_schema,
            schema,
            &mut self.reservations,
            self.metrics,
        )
        .map_err(|error| {
            error
                .with_fact_if_absent("compiled_task_id", task.id())
                .with_fact_if_absent("pipeline_task_kind", task.kind().as_str())
                .with_fact_if_absent("resident_resource_id", input)
                .with_fact_if_absent("plan_node_id", final_node_id)
                .with_fact_if_absent("sink_boundary_kind", self.pipeline.sink().kind().as_str())
        })?;
        self.host_output = Some(output);
        Ok(CompiledTaskOutput::empty())
    }

    fn task_feeds_join_input(&self, task: PipelineTaskId) -> bool {
        let mut pending = vec![task];
        let mut visited = BTreeSet::new();

        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }

            for edge in self.pipeline.edges().iter().filter(|edge| edge.from() == current) {
                match edge.kind() {
                    PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput => return true,
                    PipelineEdgeKind::ResidentFrame => pending.push(edge.to()),
                    PipelineEdgeKind::Control
                    | PipelineEdgeKind::SharedResidentProducer
                    | PipelineEdgeKind::BarrierOutput
                    | PipelineEdgeKind::SinkInput => {},
                }
            }
        }

        false
    }

    fn task_feeds_direct_join_input(&self, task: PipelineTaskId) -> bool {
        self.pipeline.edges().iter().any(|edge| {
            edge.from() == task
                && matches!(edge.kind(), PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput)
        })
    }

    fn resource_has_multiple_consumers(&self, resource: ResidentResourceId) -> bool {
        self.pipeline
            .resources()
            .iter()
            .find(|candidate| candidate.id() == resource)
            .is_some_and(|resource| resource.consumers().len() > 1)
    }

    fn join_materialize_output_can_preserve_stream(&self, task: PipelineTaskId, resource: ResidentResourceId) -> bool {
        let Some(first_resource) = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource) else {
            return false;
        };
        if first_resource.producer() != task {
            return false;
        }
        let mut current_resource = resource;
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(current_resource) {
                return false;
            }
            let Some(resource) = self.pipeline.resources().iter().find(|candidate| candidate.id() == current_resource)
            else {
                return false;
            };
            let [consumer] = resource.consumers() else {
                return false;
            };
            let Some(consumer_task) = self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer) else {
                return false;
            };
            match consumer_task.kind() {
                PipelineTaskKind::Aggregate => return true,
                PipelineTaskKind::JoinProbe => return self.join_probe_task_can_consume_frame_stream(consumer_task),
                PipelineTaskKind::Unary => {
                    let [next_resource] = consumer_task.output_resources() else {
                        return false;
                    };
                    current_resource = *next_resource;
                },
                _ => return false,
            }
        }
    }

    fn join_probe_task_can_consume_frame_stream(&self, task: &PipelineTask) -> bool {
        let Some(node_id) = task.plan_node() else {
            return false;
        };
        let Some(node) = self.plan.node(node_id) else {
            return false;
        };
        let PlanNode::Join(join) = &node.kind else {
            return false;
        };
        join.spec.residual_predicate.is_none() && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
    }

    fn task_feeds_any_join_input_through_unary_chain(&self, task: PipelineTaskId) -> bool {
        self.task_feeds_join_input_through_unary_chain(task, PipelineEdgeKind::JoinBuildInput).is_some()
            || self.task_feeds_join_input_through_unary_chain(task, PipelineEdgeKind::JoinProbeInput).is_some()
    }

    fn task_feeds_join_input_through_unary_chain(
        &self,
        task: PipelineTaskId,
        target_kind: PipelineEdgeKind,
    ) -> Option<PipelineTaskId> {
        let mut current = task;
        let mut saw_unary = false;
        loop {
            let current_task = self.pipeline.tasks().iter().find(|candidate| candidate.id() == current)?;
            let [output_resource] = current_task.output_resources() else {
                return None;
            };
            let resource = self.pipeline.resources().iter().find(|resource| resource.id() == *output_resource)?;
            let [consumer] = resource.consumers() else {
                return None;
            };
            let edge = self.pipeline.edges().iter().find(|edge| edge.from() == current && edge.to() == *consumer)?;
            match edge.kind() {
                kind if kind == target_kind && saw_unary => return Some(*consumer),
                kind if kind == target_kind => return None,
                PipelineEdgeKind::ResidentFrame => {
                    let consumer_task = self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer)?;
                    if consumer_task.kind() != PipelineTaskKind::Unary {
                        return None;
                    }
                    saw_unary = true;
                    current = *consumer;
                },
                PipelineEdgeKind::Control
                | PipelineEdgeKind::SharedResidentProducer
                | PipelineEdgeKind::BarrierOutput
                | PipelineEdgeKind::JoinBuildInput
                | PipelineEdgeKind::JoinProbeInput
                | PipelineEdgeKind::SinkInput => return None,
            }
        }
    }
}

impl CompiledTaskExecutor for CompiledGpuTaskExecutor<'_> {
    type Resource = CompiledGpuResource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskOutput<Self::Resource>> {
        validate_task_runtime_gates(self.pipeline, task)?;
        self.metrics.record_compiled_pipeline_task_execution(task.id(), task.kind().as_str());
        self.reservations.set_diagnostic_context(Some(task.id()), task_diagnostic_resident_resource(task));
        let result = match task.kind() {
            PipelineTaskKind::SourceRead | PipelineTaskKind::GpuResidentSource => self.execute_source_task(task),
            PipelineTaskKind::Unary => self.execute_unary_task(task, state),
            PipelineTaskKind::JoinBuild => self.execute_join_build_task(task, state),
            PipelineTaskKind::JoinProbe => self.execute_join_probe_task(task, state),
            PipelineTaskKind::JoinMaterialize => self.execute_join_materialize_task(task),
            PipelineTaskKind::Aggregate => self.execute_aggregate_task(task, state),
            PipelineTaskKind::FinalSort => self.execute_final_sort_task(task, state),
            PipelineTaskKind::Limit => self.execute_limit_task(task, state),
            PipelineTaskKind::SinkMaterialize => self.execute_sink_materialize_task(task, state),
        };
        self.reservations.clear_diagnostic_context();
        result
    }

    fn release_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        match payload.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                self.reservations.set_diagnostic_context(None::<PipelineTaskId>, Some(resource));
                self.reservations.release(frame.reservation, self.metrics);
                self.reservations.clear_diagnostic_context();
            },
            CompiledGpuResourcePayload::ResidentFrameStream(frames) => {
                self.reservations.set_diagnostic_context(None::<PipelineTaskId>, Some(resource));
                for frame in frames {
                    self.reservations.release(frame.reservation, self.metrics);
                }
                self.reservations.clear_diagnostic_context();
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {},
        }
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

fn single_input_resource(task: &PipelineTask) -> Result<ResidentResourceId> {
    if task.input_resources().len() != 1 {
        return Err(invalid("compiled executable task must declare exactly one input resource")
            .with_fact("compiled_task_id", task.id())
            .with_fact("pipeline_task_kind", task.kind().as_str())
            .with_fact("input_resource_count", task.input_resources().len()));
    }
    Ok(task.input_resources()[0])
}

fn task_diagnostic_resident_resource(task: &PipelineTask) -> Option<ResidentResourceId> {
    if task.output_resources().len() == 1 {
        Some(task.output_resources()[0])
    } else if task.input_resources().len() == 1 {
        Some(task.input_resources()[0])
    } else {
        None
    }
}

fn release_resident_frame_stream_reverse(
    reservations: &mut ReservationTracker,
    mut frames: Vec<ResidentGpuFrame>,
    metrics: &mut QueryMetrics,
) {
    while let Some(frame) = frames.pop() {
        drop(frame.frame);
        reservations.release(frame.reservation, metrics);
    }
}

fn join_node<'a>(plan: &'a QueryPlan, task: &PipelineTask, node_id: NodeId) -> Result<&'a JoinNode> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid("compiled join task references a missing plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
    })?;
    let PlanNode::Join(join) = &node.kind else {
        return Err(invalid("compiled join task does not reference a join plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
            .with_fact("pipeline_task_kind", task.kind().as_str()));
    };
    Ok(join)
}

fn join_strategy(
    plan: &QueryPlan,
    node: &JoinNode,
    task: &PipelineTask,
    node_id: NodeId,
    observed_build_rows: u64,
    build_side_bytes: u64,
) -> Result<CompiledJoinStrategySelection> {
    let plan_facts = keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, node);
    let strategy_facts = KeyedHashJoinStrategyInputFacts {
        estimated_build_rows: plan_facts.estimated_build_rows,
        observed_build_rows: Some(observed_build_rows),
        build_side_bytes: Some(build_side_bytes),
        probe_side_bytes: plan_facts.probe_side_bytes,
        probe_reuse_count: plan_facts.probe_reuse_count,
    };
    CompiledJoinExecutionStrategy::select_from_join_node_with_strategy_facts(node, strategy_facts)
        .map_err(|decision| {
            let (reason, facts) = match decision {
                crate::capability::KeyedHashJoinStrategyDecision::Rejected { reason, facts } => (reason, facts),
                crate::capability::KeyedHashJoinStrategyDecision::Selected { .. } => {
                    unreachable!("selected strategy decisions do not map to strategy-selection errors")
                },
            };
            let mut error = invalid("compiled join task rejected its keyed hash join execution strategy")
                .with_fact("reason", reason)
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str());
            for (key, value) in facts {
                error = error.with_fact(key, value);
            }
            error
        })?
        .ok_or_else(|| {
            invalid("compiled join task references an unsupported join execution strategy")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", node.spec.build_side)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
                .with_fact("output_projection_len", node.spec.output_projection.len())
        })
}

fn sort_node<'a>(plan: &'a QueryPlan, task: &PipelineTask, node_id: NodeId) -> Result<&'a SortNode> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid("compiled final sort task references a missing plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
    })?;
    let PlanNode::Sort(sort) = &node.kind else {
        return Err(invalid("compiled final sort task does not reference a sort plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
            .with_fact("pipeline_task_kind", task.kind().as_str()));
    };
    Ok(sort)
}

fn limit_node<'a>(plan: &'a QueryPlan, task: &PipelineTask, node_id: NodeId) -> Result<&'a LimitNode> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid("compiled limit task references a missing plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
    })?;
    let PlanNode::Limit(limit) = &node.kind else {
        return Err(invalid("compiled limit task does not reference a limit plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
            .with_fact("pipeline_task_kind", task.kind().as_str()));
    };
    Ok(limit)
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
