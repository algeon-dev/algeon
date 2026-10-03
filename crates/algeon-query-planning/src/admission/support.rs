use std::collections::{BTreeMap, BTreeSet};

use crate::{
    capability::{
        CapabilityAnalysis, CapabilityDecision, EdgeConsumer, Reason, Rejection, RejectionOwner, SubgraphInvariantKind,
        classify_join,
    },
    error::Result,
    pipeline::{
        CompiledPipeline, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentResource,
        SinkBoundaryKind,
    },
    plan::{NodeId, PlanNode, QueryPlan},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompiledBoundedPipelineSupport {
    /// A compiled-execution contract rejects the graph. The projection names
    /// the owner, the contract that failed, and its facts.
    Unsupported(Rejection),
    /// The graph is inside the compiled executor envelope.
    Supported,
}

/// Checks compiled graph shape only; it does not perform capability admission.
pub fn compiled_bounded_pipeline_support(pipeline: &CompiledPipeline) -> Result<CompiledBoundedPipelineSupport> {
    match join_task_groups(pipeline) {
        JoinTaskGroups::Complete(groups) if groups.is_empty() && !pipeline_tasks_within_compiled_contract(pipeline) => {
            return Ok(admission_rejection(
                Reason::CompiledPipelineRequiresJoinTaskGroup,
                [("pipeline_task_count", pipeline.tasks().len().to_string())],
            ));
        },
        JoinTaskGroups::Complete(_) => {},
        JoinTaskGroups::Incomplete { plan_node } => {
            return Ok(admission_rejection(
                Reason::CompiledPipelineJoinTaskGroupIncomplete,
                [("plan_node_id", plan_node.map_or_else(|| "none".to_string(), |node| node.to_string()))],
            ));
        },
    }
    if let Some((producer, kind)) = unsupported_shared_resident_producer(pipeline) {
        return Ok(admission_rejection(
            Reason::CompiledPipelineSharedProducerTaskKindUnsupported,
            [("producer_task_id", producer.to_string()), ("producer_task_kind", kind.as_str().to_string())],
        ));
    }
    let sink_path = sink_data_path(pipeline);
    if let Some(task) = stage_task_outside_sink_path(pipeline, &sink_path) {
        return Ok(admission_rejection(
            Reason::CompiledPipelineStageTaskPositionUnsupported,
            [("task_id", task.id().to_string()), ("task_kind", task.kind().as_str().to_string())],
        ));
    }

    Ok(CompiledBoundedPipelineSupport::Supported)
}

/// Checks graph shape plus plan-node compatibility for compiled execution.
///
/// `analysis` is the capability analysis the pipeline was compiled from; the
/// shared-producer lineage contract reads its invariants instead of running a
/// second analysis.
pub fn compiled_bounded_query_plan_support(
    plan: &QueryPlan,
    analysis: &CapabilityAnalysis,
    pipeline: &CompiledPipeline,
) -> Result<CompiledBoundedPipelineSupport> {
    match compiled_bounded_pipeline_support(pipeline)? {
        CompiledBoundedPipelineSupport::Supported => {},
        unsupported => return Ok(unsupported),
    }
    if let Some(rejection) = shared_resident_producer_lineage_rejection(analysis, pipeline) {
        return Ok(rejection);
    }
    if let Some(projection) =
        task_outside_contract_rejection(pipeline).or_else(|| window_input_ordering_rejection(plan, pipeline))
    {
        return Ok(CompiledBoundedPipelineSupport::Unsupported(projection));
    }

    let join_groups = match join_task_groups(pipeline) {
        JoinTaskGroups::Complete(groups) => groups,
        JoinTaskGroups::Incomplete { plan_node } => {
            return Ok(admission_rejection(
                Reason::CompiledPipelineJoinTaskGroupIncomplete,
                [("plan_node_id", plan_node.map_or_else(|| "none".to_string(), |node| node.to_string()))],
            ));
        },
    };
    for join_node in join_groups {
        let Some(PlanNode::Join(join)) = plan.node(join_node).map(|node| &node.kind) else {
            return Ok(node_rejection(
                join_node,
                Reason::CompiledPipelineJoinTaskGroupPlanNodeMismatch,
                [("plan_node_id", join_node.to_string())],
            ));
        };
        if let Err(rejection) = classify_join(join_node, join) {
            return Ok(CompiledBoundedPipelineSupport::Unsupported(rejection));
        }
    }
    Ok(CompiledBoundedPipelineSupport::Supported)
}

/// Pipelines whose every task contract the compiled executor already owns.
///
/// Admission is capability-driven, not shape-driven: any pipeline composed
/// only of these task kinds is judged by the same task/resource contracts as
/// join plans, whether it contains a union. A union of independent
/// branches is the first acceptance case, not an admission switch.
/// Aggregate and final-sort tasks use the same compiled contracts; join
/// build/probe/materialize tasks run on the coordinator through the same
/// claim/output-ready/commit protocol while their independent input branches
/// overlap on lanes, reusing one build for many probe chunks. Graph and
/// edge-normalize tasks keep their existing stream/MR/handle affinity and run
/// as serial coordinator tasks coexisting with other independent work until
/// their component owners prove safe overlap. A window is admitted only when
/// its input ordering is proved, and stays on the serial coordinator outside
/// this set.
#[doc(hidden)]
pub fn pipeline_tasks_within_migrated_contract(pipeline: &CompiledPipeline) -> bool {
    !pipeline.tasks().is_empty() && pipeline.tasks().iter().all(|task| task_kind_within_migrated_contract(task.kind()))
}

/// Every task of `pipeline` has a compiled task contract: the lane-eligible
/// kinds above, plus windows, whose input ordering admission checks
/// separately.
pub(super) fn pipeline_tasks_within_compiled_contract(pipeline: &CompiledPipeline) -> bool {
    !pipeline.tasks().is_empty()
        && pipeline
            .tasks()
            .iter()
            .all(|task| task.kind() == PipelineTaskKind::Window || task_kind_within_migrated_contract(task.kind()))
}

pub(super) fn task_kind_within_migrated_contract(kind: PipelineTaskKind) -> bool {
    matches!(
        kind,
        PipelineTaskKind::SourceRead
            | PipelineTaskKind::GpuResidentSource
            | PipelineTaskKind::LiteralBatchSource
            | PipelineTaskKind::Unary
            | PipelineTaskKind::Aggregate
            | PipelineTaskKind::JoinBuild
            | PipelineTaskKind::JoinProbe
            | PipelineTaskKind::JoinMaterialize
            | PipelineTaskKind::FinalSort
            | PipelineTaskKind::Limit
            | PipelineTaskKind::Union
            | PipelineTaskKind::EdgeNormalize
            | PipelineTaskKind::GraphAlgorithm
            | PipelineTaskKind::VectorAlgorithm
            | PipelineTaskKind::MlAlgorithm
            | PipelineTaskKind::SinkMaterialize
    )
}

fn admission_rejection(
    reason: Reason,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CompiledBoundedPipelineSupport {
    CompiledBoundedPipelineSupport::Unsupported(Rejection::new(RejectionOwner::Plan, reason, facts))
}

fn node_rejection(
    node_id: NodeId,
    reason: Reason,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CompiledBoundedPipelineSupport {
    CompiledBoundedPipelineSupport::Unsupported(Rejection::new(RejectionOwner::Node(node_id), reason, facts))
}

fn unsupported_shared_resident_producer(pipeline: &CompiledPipeline) -> Option<(PipelineTaskId, PipelineTaskKind)> {
    let tasks = task_index(pipeline);
    pipeline.edges().iter().filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer).find_map(|edge| {
        let task = tasks.get(&edge.from())?;
        (!matches!(
            task.kind(),
            PipelineTaskKind::SourceRead
                | PipelineTaskKind::GpuResidentSource
                | PipelineTaskKind::LiteralBatchSource
                | PipelineTaskKind::Unary
                | PipelineTaskKind::Aggregate
                | PipelineTaskKind::Window
                | PipelineTaskKind::Limit
                | PipelineTaskKind::JoinMaterialize
                | PipelineTaskKind::Union
                | PipelineTaskKind::VectorAlgorithm
                | PipelineTaskKind::MlAlgorithm
        ))
        .then(|| (task.id(), task.kind()))
    })
}

fn computed_shared_resident_producer_tasks(pipeline: &CompiledPipeline) -> Vec<&PipelineTask> {
    let tasks = task_index(pipeline);
    pipeline
        .edges()
        .iter()
        .filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer)
        .filter_map(|edge| {
            let task = *tasks.get(&edge.from())?;
            matches!(
                task.kind(),
                PipelineTaskKind::Unary
                    | PipelineTaskKind::Aggregate
                    | PipelineTaskKind::Window
                    | PipelineTaskKind::Limit
                    | PipelineTaskKind::JoinMaterialize
                    | PipelineTaskKind::Union
                    | PipelineTaskKind::VectorAlgorithm
                    | PipelineTaskKind::MlAlgorithm
            )
            .then_some(task)
        })
        .collect()
}

/// A computed (non-source) shared producer is only reusable when the analysis
/// proved the shared subplan lineage that keeps every consumer reading the same
/// producer occurrence.
fn shared_resident_producer_lineage_rejection(
    analysis: &CapabilityAnalysis,
    pipeline: &CompiledPipeline,
) -> Option<CompiledBoundedPipelineSupport> {
    let computed = computed_shared_resident_producer_tasks(pipeline);
    let first = computed.first()?;
    let satisfied_shared_producers = analysis
        .invariants
        .iter()
        .filter_map(|invariant| {
            (matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
                && invariant.decision == CapabilityDecision::Satisfied)
                .then(|| invariant.nodes.first().copied())
                .flatten()
        })
        .collect::<BTreeSet<_>>();
    if satisfied_shared_producers.is_empty() {
        return Some(admission_rejection(
            Reason::CompiledPipelineSharedProducerLineageUnproven,
            [("producer_task_id", first.id().to_string())],
        ));
    }
    computed.iter().filter_map(|task| task.plan_node()).find(|node| !satisfied_shared_producers.contains(node)).map(
        |node| {
            admission_rejection(
                Reason::CompiledPipelineSharedProducerLineageUnproven,
                [("producer_node_id", node.to_string())],
            )
        },
    )
}

/// The chain of tasks that carries the final output, walked back from the sink
/// to the first join materialization or union fan-in.
///
/// The walk stops where the chain stops being a single resident frame hand-off;
/// it classifies tasks for the stage-position contract and refuses nothing.
fn sink_data_path(pipeline: &CompiledPipeline) -> BTreeSet<PipelineTaskId> {
    let tasks = task_index(pipeline);
    let mut path = BTreeSet::new();
    let Some(mut current) = sink_path_terminal_task(pipeline) else {
        return path;
    };
    loop {
        if !path.insert(current) {
            return path;
        }
        let Some(task) = tasks.get(&current).copied() else {
            return path;
        };
        if matches!(task.kind(), PipelineTaskKind::JoinMaterialize | PipelineTaskKind::Union) {
            return path;
        }
        let incoming = pipeline
            .edges()
            .iter()
            .filter(|edge| edge.to() == current && edge.kind() == PipelineEdgeKind::ResidentFrame)
            .collect::<Vec<_>>();
        let [incoming] = incoming.as_slice() else {
            return path;
        };
        let [resource] = incoming.resources() else {
            return path;
        };
        if task.input_resources() != [*resource] {
            return path;
        }
        current = incoming.from();
    }
}

fn sink_path_terminal_task(pipeline: &CompiledPipeline) -> Option<PipelineTaskId> {
    match pipeline.sink().kind() {
        SinkBoundaryKind::HostArrowMaterialization => {
            let sink_materialize = single_task_of_kind(pipeline, PipelineTaskKind::SinkMaterialize)?;
            let sink_edge = pipeline
                .edges()
                .iter()
                .find(|edge| edge.kind() == PipelineEdgeKind::SinkInput && edge.to() == sink_materialize.id())?;
            Some(sink_edge.from())
        },
        SinkBoundaryKind::GpuResidentOutput => Some(pipeline.sink().input()),
    }
}

fn stage_task_outside_sink_path<'a>(
    pipeline: &'a CompiledPipeline,
    sink_path: &BTreeSet<PipelineTaskId>,
) -> Option<&'a PipelineTask> {
    pipeline.tasks().iter().find(|task| {
        if !matches!(task.kind(), PipelineTaskKind::Window | PipelineTaskKind::Union | PipelineTaskKind::FinalSort) {
            return false;
        }

        !(sink_path.contains(&task.id())
            || task_feeds_modeled_input_or_shared_branch(pipeline, task.id())
            || (stage_task_kind_can_feed_sink_path_union(task.kind())
                && task_feeds_sink_path_union(pipeline, task.id(), sink_path)))
    })
}

fn stage_task_kind_can_feed_sink_path_union(kind: PipelineTaskKind) -> bool {
    matches!(
        kind,
        PipelineTaskKind::Unary
            | PipelineTaskKind::Aggregate
            | PipelineTaskKind::Window
            | PipelineTaskKind::FinalSort
            | PipelineTaskKind::Limit
            | PipelineTaskKind::Union
    )
}

fn task_feeds_modeled_input_or_shared_branch(pipeline: &CompiledPipeline, task: PipelineTaskId) -> bool {
    let mut pending = vec![task];
    let mut visited = BTreeSet::new();

    while let Some(current) = pending.pop() {
        if !visited.insert(current) {
            continue;
        }

        for edge in pipeline.edges().iter().filter(|edge| edge.from() == current) {
            match edge.kind() {
                PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput => return true,
                PipelineEdgeKind::SharedResidentProducer => return true,
                PipelineEdgeKind::ResidentFrame => {
                    if pipeline.tasks().iter().any(|consumer| {
                        consumer.id() == edge.to()
                            && matches!(
                                consumer.kind(),
                                PipelineTaskKind::VectorAlgorithm | PipelineTaskKind::MlAlgorithm
                            )
                    }) {
                        return true;
                    }
                    pending.push(edge.to());
                },
                PipelineEdgeKind::Control | PipelineEdgeKind::BarrierOutput | PipelineEdgeKind::SinkInput => {},
            }
        }
    }

    false
}

fn task_feeds_sink_path_union(
    pipeline: &CompiledPipeline,
    task: PipelineTaskId,
    sink_path: &BTreeSet<PipelineTaskId>,
) -> bool {
    let tasks = task_index(pipeline);
    let mut pending = vec![task];
    let mut visited = BTreeSet::new();

    while let Some(current) = pending.pop() {
        if !visited.insert(current) {
            continue;
        }
        if current != task
            && sink_path.contains(&current)
            && tasks.get(&current).is_some_and(|task| task.kind() == PipelineTaskKind::Union)
        {
            return true;
        }

        for edge in pipeline.edges().iter().filter(|edge| edge.from() == current) {
            if edge.kind() == PipelineEdgeKind::ResidentFrame {
                pending.push(edge.to());
            }
        }
    }

    false
}

fn task_index(pipeline: &CompiledPipeline) -> BTreeMap<PipelineTaskId, &PipelineTask> {
    pipeline.tasks().iter().map(|task| (task.id(), task)).collect()
}

#[derive(Debug, Default)]
struct PartialJoinTaskGroup<'a> {
    build: Option<&'a PipelineTask>,
    probe: Option<&'a PipelineTask>,
    materialize: Option<&'a PipelineTask>,
}

/// Plan nodes whose build/probe/materialize trio the compiled graph declares
/// exactly once, or the plan node whose trio it did not.
enum JoinTaskGroups {
    Complete(Vec<NodeId>),
    Incomplete { plan_node: Option<NodeId> },
}

fn join_task_groups(pipeline: &CompiledPipeline) -> JoinTaskGroups {
    let mut partials = BTreeMap::<Option<NodeId>, PartialJoinTaskGroup<'_>>::new();
    for task in pipeline.tasks().iter().filter(|task| {
        matches!(
            task.kind(),
            PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe | PipelineTaskKind::JoinMaterialize
        )
    }) {
        let plan_node = task.plan_node();
        if plan_node.is_none() {
            return JoinTaskGroups::Incomplete { plan_node };
        }
        let partial = partials.entry(plan_node).or_default();
        let slot = match task.kind() {
            PipelineTaskKind::JoinBuild => &mut partial.build,
            PipelineTaskKind::JoinProbe => &mut partial.probe,
            PipelineTaskKind::JoinMaterialize => &mut partial.materialize,
            _ => unreachable!("filtered to join tasks"),
        };
        if slot.replace(task).is_some() {
            return JoinTaskGroups::Incomplete { plan_node };
        }
    }

    let mut groups = Vec::with_capacity(partials.len());
    for (plan_node, partial) in partials {
        let (Some(plan_node), Some(_), Some(_), Some(_)) =
            (plan_node, partial.build, partial.probe, partial.materialize)
        else {
            return JoinTaskGroups::Incomplete { plan_node };
        };
        groups.push(plan_node);
    }
    JoinTaskGroups::Complete(groups)
}

fn single_task_of_kind(pipeline: &CompiledPipeline, kind: PipelineTaskKind) -> Option<&PipelineTask> {
    let mut matches = pipeline.tasks().iter().filter(|task| task.kind() == kind);
    let first = matches.next();
    if matches.next().is_some() {
        return None;
    }
    first
}

/// The first task no compiled task contract covers. Windows are covered by
/// [`window_input_ordering_rejection`] instead.
fn task_outside_contract_rejection(pipeline: &CompiledPipeline) -> Option<Rejection> {
    let outside = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() != PipelineTaskKind::Window && !task_kind_within_migrated_contract(task.kind()))?;
    Some(Rejection {
        owner: RejectionOwner::Plan,
        reason: Reason::CompiledTaskKindOutsideContract,
        facts: BTreeMap::from([
            ("pipeline_task_kind", outside.kind().as_str().to_string()),
            ("pipeline_task_count", pipeline.tasks().len().to_string()),
        ]),
    })
}

/// A window that numbers rows in input order whose input the compiler did not
/// record as ordered by the window keys.
fn window_input_ordering_rejection(plan: &QueryPlan, pipeline: &CompiledPipeline) -> Option<Rejection> {
    let task = pipeline
        .tasks()
        .iter()
        .filter(|task| task.kind() == PipelineTaskKind::Window)
        .find(|task| !window_input_ordering_established(plan, pipeline, task))?;
    let producer = task
        .input_resources()
        .first()
        .and_then(|input| pipeline.resources().iter().find(|resource| resource.id() == *input))
        .and_then(|resource| pipeline.tasks().iter().find(|candidate| candidate.id() == resource.producer()))
        .and_then(PipelineTask::plan_node);
    // A window or producer task without a plan node has no edge to name, and
    // its ordering is still unproven.
    let owner = match (producer, task.plan_node()) {
        (Some(producer), Some(window)) => RejectionOwner::Edge { producer, consumer: EdgeConsumer::Node(window) },
        _ => RejectionOwner::Plan,
    };
    Some(Rejection {
        owner,
        reason: Reason::WindowInputOrderingNotEstablished,
        facts: BTreeMap::from([("window_task", task.id().to_string())]),
    })
}

fn window_input_ordering_established(plan: &QueryPlan, pipeline: &CompiledPipeline, task: &PipelineTask) -> bool {
    let Some(PlanNode::Window(window)) = task.plan_node().and_then(|node_id| plan.node(node_id)).map(|node| &node.kind)
    else {
        return false;
    };
    if !window.spec.counts_rows_in_input_order() {
        return true;
    }
    let [input] = task.input_resources() else {
        return false;
    };
    pipeline
        .resources()
        .iter()
        .find(|resource| resource.id() == *input)
        .and_then(ResidentResource::ordering)
        .is_some_and(|ordering| ordering.covers_window(&window.spec))
}
