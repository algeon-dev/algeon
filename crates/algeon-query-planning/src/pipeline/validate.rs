//! CompiledPipeline invariant validation.
//!
//! Every check here is a hard invariant of the compiled pipeline model:
//! `compile_pipeline` is expected to produce structures that already
//! satisfy these checks. Violations surface as
//! [`QueryErrorCode::Invalid`](crate::error::QueryErrorCode::Invalid) and
//! are bugs in the producer, not user errors.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    pipeline::model::{
        CompiledPipeline, PipelineBarrierId, PipelineBarrierKind, PipelineEdgeId, PipelineEdgeKind, PipelineTask,
        PipelineTaskId, PipelineTaskKind, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundaryKind,
        SortStrategy,
    },
    plan::NodeId,
};

const OPERATION: &str = "query_engine.pipeline.validate";

/// A violated compiled-pipeline schedule contract.
///
/// `reason` is the stable snake-case name of the contract that failed, so
/// admission can report the specific conflict instead of classifying the
/// error by its operation string.
#[derive(Debug)]
pub(super) struct ScheduleRejection {
    reason: &'static str,
    error: QueryError,
}

impl ScheduleRejection {
    pub(super) const fn reason(&self) -> &'static str {
        self.reason
    }

    pub(super) const fn error(&self) -> &QueryError {
        &self.error
    }

    pub(super) fn into_error(self) -> QueryError {
        self.error
    }
}

pub(super) fn validate_pipeline(pipeline: &CompiledPipeline) -> Result<()> {
    validate_schedule(pipeline).map_err(ScheduleRejection::into_error)
}

pub(super) fn validate_schedule(pipeline: &CompiledPipeline) -> std::result::Result<(), ScheduleRejection> {
    let task_ids = named("compiled_pipeline_task_identity_invalid", validate_tasks(pipeline))?;
    named("compiled_pipeline_edge_identity_invalid", validate_edges(pipeline, &task_ids))?;
    named("compiled_pipeline_barrier_identity_invalid", validate_barriers(pipeline, &task_ids))?;
    let resource_ids = named("compiled_pipeline_resource_identity_invalid", validate_resources(pipeline, &task_ids))?;

    named("compiled_pipeline_task_dependency_missing", validate_task_dependencies(pipeline, &task_ids))?;
    named("compiled_pipeline_dependency_cycle", validate_task_graph_is_acyclic(pipeline))?;
    named("compiled_pipeline_resource_reference_missing", validate_task_resource_references(pipeline, &resource_ids))?;
    named("compiled_pipeline_resource_flow_invalid", validate_resource_flow(pipeline))?;
    named("compiled_pipeline_sink_output_lifetime_invalid", validate_sink_output_lifetimes(pipeline))?;
    named("compiled_pipeline_barrier_dominance_invalid", validate_barrier_dominance(pipeline))?;
    named("compiled_pipeline_task_resource_arity_invalid", validate_task_resource_arity(pipeline))?;
    named("compiled_pipeline_stage_boundary_incomplete", validate_stage_boundary_coverage(pipeline))?;
    named("compiled_pipeline_join_build_before_probe_missing", validate_join_build_before_probe(pipeline))?;
    named("compiled_pipeline_fused_topk_limit_binding_invalid", validate_fused_topk_limit_binding(pipeline))?;

    if !task_ids.contains(&pipeline.sink().input()) {
        return Err(ScheduleRejection {
            reason: "compiled_pipeline_sink_input_task_missing",
            error: invalid("pipeline sink input task does not exist").with_fact("task_id", pipeline.sink().input()),
        });
    }

    Ok(())
}

fn named<T>(reason: &'static str, result: Result<T>) -> std::result::Result<T, ScheduleRejection> {
    result.map_err(|error| ScheduleRejection { reason, error })
}

fn validate_tasks(pipeline: &CompiledPipeline) -> Result<BTreeSet<PipelineTaskId>> {
    let mut ids = BTreeSet::new();
    for (index, task) in pipeline.tasks().iter().enumerate() {
        if task.id().index() != index {
            return Err(invalid("pipeline task id does not match task position")
                .with_fact("task_id", task.id())
                .with_fact("position", index));
        }
        if !ids.insert(task.id()) {
            return Err(invalid("pipeline task id is duplicated").with_fact("task_id", task.id()));
        }
    }
    Ok(ids)
}

fn validate_edges(
    pipeline: &CompiledPipeline,
    task_ids: &BTreeSet<PipelineTaskId>,
) -> Result<BTreeSet<PipelineEdgeId>> {
    let mut ids = BTreeSet::new();
    for (index, edge) in pipeline.edges().iter().enumerate() {
        if edge.id().index() != index {
            return Err(invalid("pipeline edge id does not match edge position")
                .with_fact("edge_id", edge.id())
                .with_fact("position", index));
        }
        if !ids.insert(edge.id()) {
            return Err(invalid("pipeline edge id is duplicated").with_fact("edge_id", edge.id()));
        }
        if !task_ids.contains(&edge.from()) {
            return Err(invalid("pipeline edge producer task does not exist")
                .with_fact("edge_id", edge.id())
                .with_fact("task_id", edge.from()));
        }
        if !task_ids.contains(&edge.to()) {
            return Err(invalid("pipeline edge consumer task does not exist")
                .with_fact("edge_id", edge.id())
                .with_fact("task_id", edge.to()));
        }
    }
    Ok(ids)
}

fn validate_barriers(
    pipeline: &CompiledPipeline,
    task_ids: &BTreeSet<PipelineTaskId>,
) -> Result<BTreeSet<PipelineBarrierId>> {
    let mut ids = BTreeSet::new();
    for (index, barrier) in pipeline.barriers().iter().enumerate() {
        if barrier.id().index() != index {
            return Err(invalid("pipeline barrier id does not match barrier position")
                .with_fact("barrier_id", barrier.id())
                .with_fact("position", index));
        }
        if !ids.insert(barrier.id()) {
            return Err(invalid("pipeline barrier id is duplicated").with_fact("barrier_id", barrier.id()));
        }
        for task in barrier.upstream_tasks().iter().chain(barrier.downstream_tasks()) {
            if !task_ids.contains(task) {
                return Err(invalid("pipeline barrier task does not exist")
                    .with_fact("barrier_id", barrier.id())
                    .with_fact("task_id", task));
            }
        }
    }
    Ok(ids)
}

fn validate_resources(
    pipeline: &CompiledPipeline,
    task_ids: &BTreeSet<PipelineTaskId>,
) -> Result<BTreeSet<ResidentResourceId>> {
    let mut ids = BTreeSet::new();
    for (index, resource) in pipeline.resources().iter().enumerate() {
        if resource.id().index() != index {
            return Err(invalid("pipeline resident resource id does not match resource position")
                .with_fact("resource_id", resource.id())
                .with_fact("position", index));
        }
        if !ids.insert(resource.id()) {
            return Err(invalid("pipeline resident resource id is duplicated").with_fact("resource_id", resource.id()));
        }
        if !task_ids.contains(&resource.producer()) {
            return Err(invalid("pipeline resident resource producer task does not exist")
                .with_fact("resource_id", resource.id())
                .with_fact("task_id", resource.producer()));
        }
        for consumer in resource.consumers() {
            if !task_ids.contains(consumer) {
                return Err(invalid("pipeline resident resource consumer task does not exist")
                    .with_fact("resource_id", resource.id())
                    .with_fact("task_id", consumer));
            }
        }
        validate_lifetime(resource.id(), resource.lifetime(), task_ids)?;
    }
    Ok(ids)
}

fn validate_lifetime(
    resource: ResidentResourceId,
    lifetime: ResidentLifetime,
    task_ids: &BTreeSet<PipelineTaskId>,
) -> Result<()> {
    match lifetime {
        ResidentLifetime::TaskLocal { task } => {
            if !task_ids.contains(&task) {
                return Err(invalid("pipeline task-local lifetime task does not exist")
                    .with_fact("resource_id", resource)
                    .with_fact("task_id", task));
            }
        },
        ResidentLifetime::BetweenTasks { producer } => {
            if !task_ids.contains(&producer) {
                return Err(invalid("pipeline resident lifetime producer task does not exist")
                    .with_fact("resource_id", resource)
                    .with_fact("task_id", producer));
            }
        },
        ResidentLifetime::SinkOutput { producer } => {
            if !task_ids.contains(&producer) {
                return Err(invalid("pipeline sink-output lifetime producer task does not exist")
                    .with_fact("resource_id", resource)
                    .with_fact("task_id", producer));
            }
        },
    }
    Ok(())
}

fn validate_task_dependencies(pipeline: &CompiledPipeline, task_ids: &BTreeSet<PipelineTaskId>) -> Result<()> {
    let mut incoming = BTreeSet::new();
    for edge in pipeline.edges() {
        incoming.insert(edge.to());
    }

    for task in pipeline.tasks() {
        if matches!(
            task.kind(),
            PipelineTaskKind::SourceRead | PipelineTaskKind::GpuResidentSource | PipelineTaskKind::LiteralBatchSource
        ) {
            continue;
        }
        if !incoming.contains(&task.id()) {
            return Err(invalid("pipeline non-source task has no input dependency").with_fact("task_id", task.id()));
        }
    }

    for task in pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::JoinProbe) {
        let has_build_dependency = pipeline
            .edges()
            .iter()
            .any(|edge| edge.to() == task.id() && is_join_build_for_probe(pipeline, edge.from(), task.id()));
        if !has_build_dependency {
            return Err(
                invalid("pipeline join probe task does not depend on its build task").with_fact("task_id", task.id())
            );
        }
    }

    for task in task_ids {
        if pipeline.tasks().iter().all(|candidate| candidate.id() != *task) {
            return Err(invalid("pipeline task id set contains missing task").with_fact("task_id", task));
        }
    }

    Ok(())
}

fn is_join_build_for_probe(pipeline: &CompiledPipeline, build_id: PipelineTaskId, probe_id: PipelineTaskId) -> bool {
    let Some(build) = pipeline.tasks().iter().find(|task| task.id() == build_id) else {
        return false;
    };
    let Some(probe) = pipeline.tasks().iter().find(|task| task.id() == probe_id) else {
        return false;
    };
    build.kind() == PipelineTaskKind::JoinBuild
        && probe.kind() == PipelineTaskKind::JoinProbe
        && (build.plan_node().is_none() || probe.plan_node().is_none() || build.plan_node() == probe.plan_node())
}

fn validate_task_graph_is_acyclic(pipeline: &CompiledPipeline) -> Result<()> {
    let adjacency = adjacency(pipeline);
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();

    for task in pipeline.tasks() {
        visit_task(task.id(), &adjacency, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn visit_task(
    task: PipelineTaskId,
    adjacency: &BTreeMap<PipelineTaskId, Vec<PipelineTaskId>>,
    visiting: &mut BTreeSet<PipelineTaskId>,
    visited: &mut BTreeSet<PipelineTaskId>,
) -> Result<()> {
    if visited.contains(&task) {
        return Ok(());
    }
    if !visiting.insert(task) {
        return Err(invalid("pipeline task dependency graph contains a cycle").with_fact("task_id", task));
    }
    for downstream in adjacency.get(&task).into_iter().flatten() {
        visit_task(*downstream, adjacency, visiting, visited)?;
    }
    visiting.remove(&task);
    visited.insert(task);
    Ok(())
}

fn validate_task_resource_references(
    pipeline: &CompiledPipeline,
    resource_ids: &BTreeSet<ResidentResourceId>,
) -> Result<()> {
    for task in pipeline.tasks() {
        for resource in task.input_resources().iter().chain(task.output_resources()) {
            if !resource_ids.contains(resource) {
                return Err(invalid("pipeline task references missing resident resource")
                    .with_fact("task_id", task.id())
                    .with_fact("resource_id", resource));
            }
        }
    }
    for edge in pipeline.edges() {
        for resource in edge.resources() {
            if !resource_ids.contains(resource) {
                return Err(invalid("pipeline edge references missing resident resource")
                    .with_fact("edge_id", edge.id())
                    .with_fact("resource_id", resource));
            }
        }
    }
    Ok(())
}

fn validate_resource_flow(pipeline: &CompiledPipeline) -> Result<()> {
    for resource in pipeline.resources() {
        let producer = task_by_id(pipeline, resource.producer())?;
        if !producer.output_resources().contains(&resource.id()) {
            return Err(invalid("pipeline resident resource producer task does not output resource")
                .with_fact("resource_id", resource.id())
                .with_fact("task_id", resource.producer()));
        }

        for consumer_id in resource.consumers() {
            let consumer = task_by_id(pipeline, *consumer_id)?;
            if !consumer.input_resources().contains(&resource.id()) {
                return Err(invalid("pipeline resident resource consumer task does not input resource")
                    .with_fact("resource_id", resource.id())
                    .with_fact("task_id", consumer_id));
            }
            if !task_reaches(pipeline, resource.producer(), *consumer_id) {
                return Err(invalid("pipeline resident resource producer does not dominate consumer")
                    .with_fact("resource_id", resource.id())
                    .with_fact("producer_task", resource.producer())
                    .with_fact("consumer_task", consumer_id));
            }
            if !edge_carries_resource(pipeline, resource.producer(), *consumer_id, resource.id()) {
                return Err(invalid("pipeline resident resource consumer has no carrying dependency edge")
                    .with_fact("resource_id", resource.id())
                    .with_fact("producer_task", resource.producer())
                    .with_fact("consumer_task", consumer_id));
            }
        }

        match resource.lifetime() {
            ResidentLifetime::TaskLocal { task } => {
                if task != resource.producer() {
                    return Err(invalid("pipeline task-local lifetime does not match resource producer")
                        .with_fact("resource_id", resource.id())
                        .with_fact("lifetime_task", task)
                        .with_fact("producer_task", resource.producer()));
                }
                if !resource.consumers().is_empty() {
                    return Err(invalid("pipeline task-local lifetime has external consumers")
                        .with_fact("resource_id", resource.id())
                        .with_fact("consumer_count", resource.consumers().len()));
                }
            },
            ResidentLifetime::BetweenTasks { producer } => {
                if producer != resource.producer() {
                    return Err(invalid("pipeline between-tasks lifetime producer does not match resource producer")
                        .with_fact("resource_id", resource.id())
                        .with_fact("lifetime_producer", producer)
                        .with_fact("producer_task", resource.producer()));
                }
                if resource.consumers().is_empty() {
                    return Err(invalid("pipeline between-tasks lifetime has no consumers")
                        .with_fact("resource_id", resource.id()));
                }
            },
            ResidentLifetime::SinkOutput { producer } => {
                if producer != resource.producer() {
                    return Err(invalid("pipeline sink-output lifetime producer does not match resource producer")
                        .with_fact("resource_id", resource.id())
                        .with_fact("lifetime_producer", producer)
                        .with_fact("producer_task", resource.producer()));
                }
                if !resource.consumers().is_empty() {
                    return Err(invalid("pipeline sink-output lifetime has task consumers")
                        .with_fact("resource_id", resource.id())
                        .with_fact("consumer_count", resource.consumers().len()));
                }
            },
        }
        validate_shared_resident_resource_edges(pipeline, resource)?;
    }

    for edge in pipeline.edges() {
        for resource_id in edge.resources() {
            let resource = resource_by_id(pipeline, *resource_id)?;
            if resource.producer() != edge.from() {
                return Err(invalid("pipeline edge resource producer does not match edge producer")
                    .with_fact("edge_id", edge.id())
                    .with_fact("resource_id", resource_id)
                    .with_fact("edge_producer", edge.from())
                    .with_fact("resource_producer", resource.producer()));
            }
            if !resource.consumers().contains(&edge.to()) {
                return Err(invalid("pipeline edge resource consumer is not recorded on resource")
                    .with_fact("edge_id", edge.id())
                    .with_fact("resource_id", resource_id)
                    .with_fact("edge_consumer", edge.to()));
            }
            if edge.kind() == PipelineEdgeKind::SharedResidentProducer && resource.consumers().len() < 2 {
                return Err(invalid("pipeline shared resident producer edge does not fan out to multiple consumers")
                    .with_fact("edge_id", edge.id())
                    .with_fact("resource_id", resource_id)
                    .with_fact("consumer_count", resource.consumers().len()));
            }
        }
    }

    validate_task_input_resource_consumers(pipeline)?;

    Ok(())
}

fn validate_sink_output_lifetimes(pipeline: &CompiledPipeline) -> Result<()> {
    let sink_outputs = pipeline
        .resources()
        .iter()
        .filter(|resource| matches!(resource.lifetime(), ResidentLifetime::SinkOutput { .. }))
        .collect::<Vec<_>>();
    match pipeline.sink().kind() {
        SinkBoundaryKind::HostArrowMaterialization => {
            if let Some(resource) = sink_outputs.first() {
                return Err(invalid("pipeline host sink cannot retain sink-output resident resources")
                    .with_fact("resident_resource_id", resource.id())
                    .with_fact("sink_boundary_kind", pipeline.sink().kind().as_str())
                    .with_fact("sink_output_resource_count", sink_outputs.len()));
            }
        },
        SinkBoundaryKind::GpuResidentOutput => {
            let [resource] = sink_outputs.as_slice() else {
                return Err(invalid("pipeline GPU sink must retain exactly one sink-output resident resource")
                    .with_fact("sink_boundary_task_id", pipeline.sink().input())
                    .with_fact("sink_boundary_kind", pipeline.sink().kind().as_str())
                    .with_fact("sink_output_resource_count", sink_outputs.len()));
            };
            if resource.producer() != pipeline.sink().input() {
                return Err(invalid(
                    "pipeline GPU sink-output resource producer must match the sink boundary input task",
                )
                .with_fact("resident_resource_id", resource.id())
                .with_fact("producer_task", resource.producer())
                .with_fact("sink_boundary_task_id", pipeline.sink().input()));
            }
            let producer = task_by_id(pipeline, resource.producer())?;
            if !producer.output_resources().contains(&resource.id()) {
                return Err(invalid(
                    "pipeline GPU sink-output resource is not produced by the sink boundary input task",
                )
                .with_fact("resident_resource_id", resource.id())
                .with_fact("producer_task", resource.producer())
                .with_fact("sink_boundary_task_id", pipeline.sink().input()));
            }
        },
    }
    Ok(())
}

fn validate_shared_resident_resource_edges(pipeline: &CompiledPipeline, resource: &ResidentResource) -> Result<()> {
    let shared_edges = pipeline
        .edges()
        .iter()
        .filter(|edge| {
            edge.kind() == PipelineEdgeKind::SharedResidentProducer && edge.resources().contains(&resource.id())
        })
        .collect::<Vec<_>>();
    if shared_edges.is_empty() {
        return Ok(());
    }

    if resource.consumers().len() < 2 {
        return Err(invalid("pipeline shared resident resource does not fan out to multiple consumers")
            .with_fact("resource_id", resource.id())
            .with_fact("consumer_count", resource.consumers().len()));
    }

    for edge in &shared_edges {
        if edge.resources() != [resource.id()] {
            return Err(invalid("pipeline shared resident producer edge must carry exactly one resource identity")
                .with_fact("edge_id", edge.id())
                .with_fact("resource_id", resource.id())
                .with_fact("edge_resource_count", edge.resources().len()));
        }
    }

    let shared_resource_edge_consumers = pipeline
        .edges()
        .iter()
        .filter(|edge| {
            edge.resources().contains(&resource.id())
                && matches!(
                    edge.kind(),
                    PipelineEdgeKind::SharedResidentProducer
                        | PipelineEdgeKind::JoinBuildInput
                        | PipelineEdgeKind::JoinProbeInput
                        | PipelineEdgeKind::SinkInput
                )
        })
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    let recorded_consumers = resource.consumers().iter().copied().collect::<BTreeSet<_>>();
    if shared_resource_edge_consumers != recorded_consumers {
        let missing = recorded_consumers.difference(&shared_resource_edge_consumers).next().copied();
        let extra = shared_resource_edge_consumers.difference(&recorded_consumers).next().copied();
        let mut error = invalid("pipeline shared resident producer edges do not cover recorded consumers")
            .with_fact("resource_id", resource.id())
            .with_fact("producer_task", resource.producer());
        if let Some(missing) = missing {
            error = error.with_fact("missing_consumer_task", missing);
        }
        if let Some(extra) = extra {
            error = error.with_fact("extra_consumer_task", extra);
        }
        return Err(error);
    }

    Ok(())
}

fn validate_task_input_resource_consumers(pipeline: &CompiledPipeline) -> Result<()> {
    for task in pipeline.tasks() {
        for resource_id in task.input_resources() {
            let resource = resource_by_id(pipeline, *resource_id)?;
            if !resource.consumers().contains(&task.id()) {
                return Err(invalid("pipeline task input resource consumer is not recorded on resource")
                    .with_fact("resource_id", resource.id())
                    .with_fact("producer_task", resource.producer())
                    .with_fact("consumer_task", task.id()));
            }
            if !edge_carries_resource(pipeline, resource.producer(), task.id(), resource.id()) {
                return Err(invalid("pipeline task input resource has no carrying dependency edge")
                    .with_fact("resource_id", resource.id())
                    .with_fact("producer_task", resource.producer())
                    .with_fact("consumer_task", task.id()));
            }
        }
    }
    Ok(())
}

/// Each task kind declares how many resident resources it consumes and
/// produces; a schedule whose tasks disagree cannot be interpreted.
fn validate_task_resource_arity(pipeline: &CompiledPipeline) -> Result<()> {
    for task in pipeline.tasks() {
        let inputs = task.input_resources().len();
        let outputs = task.output_resources().len();
        let satisfied = match task.kind() {
            PipelineTaskKind::SourceRead
            | PipelineTaskKind::GpuResidentSource
            | PipelineTaskKind::LiteralBatchSource => inputs == 0 && outputs == 1,
            PipelineTaskKind::Unary
            | PipelineTaskKind::Aggregate
            | PipelineTaskKind::Window
            | PipelineTaskKind::FinalSort
            | PipelineTaskKind::Limit
            | PipelineTaskKind::EdgeNormalize => inputs == 1 && outputs == 1,
            PipelineTaskKind::Union
            | PipelineTaskKind::GraphAlgorithm
            | PipelineTaskKind::VectorAlgorithm
            | PipelineTaskKind::MlAlgorithm => inputs >= 1 && outputs == 1,
            PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe => inputs == 1 && outputs == 0,
            PipelineTaskKind::JoinMaterialize => inputs == 0 && outputs == 1,
            PipelineTaskKind::SinkMaterialize => inputs == 1 && outputs == 0,
        };
        if !satisfied {
            return Err(invalid("pipeline task resource arity does not match its task contract")
                .with_fact("task_id", task.id())
                .with_fact("task_kind", task.kind().as_str())
                .with_fact("input_resource_count", inputs)
                .with_fact("output_resource_count", outputs));
        }
    }
    Ok(())
}

/// A fused top-k claim names the one limit task that consumes the sort output
/// and drops the fused marker: `applied_limit` is set by exactly one producer
/// and consumed by exactly one limit pass-through, so two producers claiming
/// the same limit, or a claim whose limit task does not consume the sort
/// output, would apply the fetch/skip twice or not at all.
fn validate_fused_topk_limit_binding(pipeline: &CompiledPipeline) -> Result<()> {
    let mut claimed_limits = BTreeMap::<(PipelineTaskId, NodeId), PipelineTaskId>::new();
    for task in pipeline.tasks() {
        let Some(SortStrategy::FusedTopK { limit_task, limit_node }) = task.sort_strategy() else {
            continue;
        };
        if let Some(first_claimant) = claimed_limits.insert((limit_task, limit_node), task.id()) {
            return Err(invalid("two fused top-k producers claim the same limit")
                .with_fact("first_sort_task", first_claimant)
                .with_fact("second_sort_task", task.id())
                .with_fact("limit_task", limit_task)
                .with_fact("limit_node", limit_node));
        }
        let Some(limit) = pipeline.tasks().iter().find(|candidate| candidate.id() == limit_task) else {
            return Err(invalid("fused top-k limit task does not exist")
                .with_fact("sort_task", task.id())
                .with_fact("limit_task", limit_task)
                .with_fact("limit_node", limit_node));
        };
        if limit.kind() != PipelineTaskKind::Limit {
            return Err(invalid("fused top-k limit task is not a limit")
                .with_fact("sort_task", task.id())
                .with_fact("limit_task", limit_task)
                .with_fact("limit_task_kind", limit.kind().as_str())
                .with_fact("limit_node", limit_node));
        }
        if limit.plan_node() != Some(limit_node) {
            return Err(invalid("fused top-k limit task serves a different plan node")
                .with_fact("sort_task", task.id())
                .with_fact("limit_task", limit_task)
                .with_fact("limit_node", limit_node));
        }
        let consumes_sort_output =
            task.output_resources().iter().any(|resource| limit.input_resources().contains(resource));
        if !consumes_sort_output {
            return Err(invalid("fused top-k limit task does not consume the sort output")
                .with_fact("sort_task", task.id())
                .with_fact("limit_task", limit_task)
                .with_fact("limit_node", limit_node));
        }
    }
    Ok(())
}

/// A blocking operator only produces a usable output once it has consumed its
/// whole input, so its boundary barrier must gate every task that reads that
/// output. The sink materialization edge is excluded: the sink boundary
/// barrier gates the final output instead.
fn validate_stage_boundary_coverage(pipeline: &CompiledPipeline) -> Result<()> {
    for task in pipeline.tasks() {
        let barrier_kind = match task.kind() {
            PipelineTaskKind::Aggregate | PipelineTaskKind::Window => PipelineBarrierKind::AggregateBoundary,
            PipelineTaskKind::FinalSort => PipelineBarrierKind::SortBoundary,
            _ => continue,
        };
        let consumers = direct_data_consumers(pipeline, task.id());
        let gated = pipeline.barriers().iter().any(|barrier| {
            barrier.kind() == barrier_kind
                && barrier.upstream_tasks() == [task.id()]
                && consumers.iter().all(|consumer| barrier.downstream_tasks().contains(consumer))
        });
        if !gated {
            return Err(invalid("pipeline blocking boundary does not gate every downstream consumer")
                .with_fact("task_id", task.id())
                .with_fact("task_kind", task.kind().as_str())
                .with_fact("barrier_kind", barrier_kind.as_str())
                .with_fact("downstream_consumer_count", consumers.len()));
        }
    }
    Ok(())
}

/// A probe reads retained build state, so the build task must complete before
/// it starts. The compiler emits the barrier from the join task group; this
/// check proves the emitted schedule actually carries it.
fn validate_join_build_before_probe(pipeline: &CompiledPipeline) -> Result<()> {
    for probe in pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::JoinProbe) {
        let build = pipeline
            .tasks()
            .iter()
            .find(|task| task.kind() == PipelineTaskKind::JoinBuild && task.plan_node() == probe.plan_node());
        let gated = build.is_some_and(|build| {
            pipeline.barriers().iter().any(|barrier| {
                barrier.kind() == PipelineBarrierKind::JoinBuildBeforeProbe
                    && barrier.upstream_tasks() == [build.id()]
                    && barrier.downstream_tasks().contains(&probe.id())
            })
        });
        if !gated {
            return Err(invalid("pipeline join probe is not gated by its build barrier")
                .with_fact("probe_task", probe.id())
                .with_fact("build_task", build.map_or_else(|| "none".to_string(), |build| build.id().to_string())));
        }
    }
    Ok(())
}

fn direct_data_consumers(pipeline: &CompiledPipeline, task: PipelineTaskId) -> BTreeSet<PipelineTaskId> {
    pipeline
        .edges()
        .iter()
        .filter(|edge| {
            edge.from() == task
                && matches!(
                    edge.kind(),
                    PipelineEdgeKind::ResidentFrame
                        | PipelineEdgeKind::SharedResidentProducer
                        | PipelineEdgeKind::JoinBuildInput
                        | PipelineEdgeKind::JoinProbeInput
                )
        })
        .map(|edge| edge.to())
        .collect()
}

fn validate_barrier_dominance(pipeline: &CompiledPipeline) -> Result<()> {
    for barrier in pipeline.barriers() {
        if barrier.upstream_tasks().is_empty() {
            return Err(invalid("pipeline barrier has no upstream tasks").with_fact("barrier_id", barrier.id()));
        }

        for downstream in barrier.downstream_tasks() {
            if !barrier.upstream_tasks().iter().any(|upstream| task_reaches(pipeline, *upstream, *downstream)) {
                return Err(invalid("pipeline barrier upstream tasks do not dominate downstream task")
                    .with_fact("barrier_id", barrier.id())
                    .with_fact("downstream_task", downstream));
            }
        }

        match barrier.kind() {
            PipelineBarrierKind::JoinBuildBeforeProbe => {
                let [build] = barrier.upstream_tasks() else {
                    return Err(invalid("pipeline join build-before-probe barrier must have one build task")
                        .with_fact("barrier_id", barrier.id())
                        .with_fact("upstream_count", barrier.upstream_tasks().len()));
                };
                let [probe] = barrier.downstream_tasks() else {
                    return Err(invalid("pipeline join build-before-probe barrier must have one probe task")
                        .with_fact("barrier_id", barrier.id())
                        .with_fact("downstream_count", barrier.downstream_tasks().len()));
                };
                validate_barrier_task_pair(
                    pipeline,
                    barrier.id(),
                    *build,
                    PipelineTaskKind::JoinBuild,
                    *probe,
                    PipelineTaskKind::JoinProbe,
                )?;
            },
            PipelineBarrierKind::JoinMaterialization => {
                let [probe] = barrier.upstream_tasks() else {
                    return Err(invalid("pipeline join materialization barrier must have one probe task")
                        .with_fact("barrier_id", barrier.id())
                        .with_fact("upstream_count", barrier.upstream_tasks().len()));
                };
                let [materialize] = barrier.downstream_tasks() else {
                    return Err(invalid("pipeline join materialization barrier must have one materialize task")
                        .with_fact("barrier_id", barrier.id())
                        .with_fact("downstream_count", barrier.downstream_tasks().len()));
                };
                validate_barrier_task_pair(
                    pipeline,
                    barrier.id(),
                    *probe,
                    PipelineTaskKind::JoinProbe,
                    *materialize,
                    PipelineTaskKind::JoinMaterialize,
                )?;
            },
            PipelineBarrierKind::AggregateBoundary => {
                validate_barrier_upstream_kinds(
                    pipeline,
                    barrier.id(),
                    barrier.upstream_tasks(),
                    &[PipelineTaskKind::Aggregate, PipelineTaskKind::Window],
                )?;
            },
            PipelineBarrierKind::SortBoundary => {
                validate_barrier_upstream_kinds(
                    pipeline,
                    barrier.id(),
                    barrier.upstream_tasks(),
                    &[PipelineTaskKind::FinalSort],
                )?;
            },
            PipelineBarrierKind::SourceChunkBoundary => {
                validate_barrier_upstream_kinds(
                    pipeline,
                    barrier.id(),
                    barrier.upstream_tasks(),
                    &[PipelineTaskKind::SourceRead],
                )?;
            },
            PipelineBarrierKind::SinkBoundary => {},
        }
    }
    Ok(())
}

fn validate_barrier_task_pair(
    pipeline: &CompiledPipeline,
    barrier: PipelineBarrierId,
    upstream_id: PipelineTaskId,
    expected_upstream: PipelineTaskKind,
    downstream_id: PipelineTaskId,
    expected_downstream: PipelineTaskKind,
) -> Result<()> {
    let upstream = task_by_id(pipeline, upstream_id)?;
    let downstream = task_by_id(pipeline, downstream_id)?;
    if upstream.kind() != expected_upstream {
        return Err(invalid("pipeline barrier upstream task has wrong kind")
            .with_fact("barrier_id", barrier)
            .with_fact("task_id", upstream_id)
            .with_fact("expected_kind", expected_upstream.as_str())
            .with_fact("actual_kind", upstream.kind().as_str()));
    }
    if downstream.kind() != expected_downstream {
        return Err(invalid("pipeline barrier downstream task has wrong kind")
            .with_fact("barrier_id", barrier)
            .with_fact("task_id", downstream_id)
            .with_fact("expected_kind", expected_downstream.as_str())
            .with_fact("actual_kind", downstream.kind().as_str()));
    }
    if upstream.plan_node().is_some()
        && downstream.plan_node().is_some()
        && upstream.plan_node() != downstream.plan_node()
    {
        return Err(invalid("pipeline join barrier tasks do not belong to the same plan node")
            .with_fact("barrier_id", barrier)
            .with_fact("upstream_task", upstream_id)
            .with_fact("downstream_task", downstream_id));
    }
    Ok(())
}

fn validate_barrier_upstream_kinds(
    pipeline: &CompiledPipeline,
    barrier: PipelineBarrierId,
    upstream_tasks: &[PipelineTaskId],
    expected: &[PipelineTaskKind],
) -> Result<()> {
    for task_id in upstream_tasks {
        let task = task_by_id(pipeline, *task_id)?;
        if !expected.contains(&task.kind()) {
            return Err(invalid("pipeline barrier upstream task has wrong kind")
                .with_fact("barrier_id", barrier)
                .with_fact("task_id", task_id)
                .with_fact("expected_kind", expected.iter().map(|kind| kind.as_str()).collect::<Vec<_>>().join("|"))
                .with_fact("actual_kind", task.kind().as_str()));
        }
    }
    Ok(())
}

fn task_by_id(pipeline: &CompiledPipeline, task_id: PipelineTaskId) -> Result<&PipelineTask> {
    pipeline
        .tasks()
        .iter()
        .find(|task| task.id() == task_id)
        .ok_or_else(|| invalid("pipeline task does not exist").with_fact("task_id", task_id))
}

fn resource_by_id(pipeline: &CompiledPipeline, resource_id: ResidentResourceId) -> Result<&ResidentResource> {
    pipeline
        .resources()
        .iter()
        .find(|resource| resource.id() == resource_id)
        .ok_or_else(|| invalid("pipeline resident resource does not exist").with_fact("resource_id", resource_id))
}

fn adjacency(pipeline: &CompiledPipeline) -> BTreeMap<PipelineTaskId, Vec<PipelineTaskId>> {
    let mut adjacency = BTreeMap::<PipelineTaskId, Vec<PipelineTaskId>>::new();
    for edge in pipeline.edges() {
        adjacency.entry(edge.from()).or_default().push(edge.to());
    }
    adjacency
}

fn task_reaches(pipeline: &CompiledPipeline, from: PipelineTaskId, to: PipelineTaskId) -> bool {
    if from == to {
        return true;
    }
    let adjacency = adjacency(pipeline);
    let mut stack = vec![from];
    let mut visited = BTreeSet::new();
    while let Some(task) = stack.pop() {
        if !visited.insert(task) {
            continue;
        }
        for downstream in adjacency.get(&task).into_iter().flatten() {
            if *downstream == to {
                return true;
            }
            stack.push(*downstream);
        }
    }
    false
}

fn edge_carries_resource(
    pipeline: &CompiledPipeline,
    from: PipelineTaskId,
    to: PipelineTaskId,
    resource: ResidentResourceId,
) -> bool {
    pipeline.edges().iter().any(|edge| edge.from() == from && edge.to() == to && edge.resources().contains(&resource))
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
