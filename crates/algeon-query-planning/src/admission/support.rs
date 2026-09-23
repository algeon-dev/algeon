use std::collections::{BTreeMap, BTreeSet};

use crate::{
    capability::{
        CapabilityAnalysis, CapabilityDecision, CapabilityRejectionCategory, CapabilityRejectionOwner,
        CapabilityRejectionProjection, CapabilityScope, CompiledJoinExecutionStrategy, KeyedHashJoinStrategyDecision,
        SubgraphInvariantKind,
    },
    error::Result,
    pipeline::{CompiledPipeline, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, SinkBoundaryKind},
    plan::{NodeId, PlanNode, QueryPlan},
};

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompiledBoundedPipelineSupport {
    /// A compiled-execution contract rejects the graph. The projection names
    /// the owner, the contract that failed, and its facts.
    Unsupported(CapabilityRejectionProjection),
    /// The graph is inside the compiled executor envelope.
    Supported,
}

#[doc(hidden)]
/// Checks compiled graph shape only; it does not perform capability admission.
pub fn compiled_bounded_pipeline_support(pipeline: &CompiledPipeline) -> Result<CompiledBoundedPipelineSupport> {
    pipeline.validate()?;

    match join_task_groups(pipeline) {
        JoinTaskGroups::Complete(groups) if groups.is_empty() && !pipeline_tasks_within_migrated_contract(pipeline) => {
            return Ok(admission_rejection(
                "compiled_pipeline_requires_join_task_group",
                [("pipeline_task_count", pipeline.tasks().len().to_string())],
            ));
        },
        JoinTaskGroups::Complete(_) => {},
        JoinTaskGroups::Incomplete { plan_node } => {
            return Ok(admission_rejection(
                "compiled_pipeline_join_task_group_incomplete",
                [("plan_node_id", plan_node.map_or_else(|| "none".to_string(), |node| node.to_string()))],
            ));
        },
    }
    if let Some((producer, kind)) = unsupported_shared_resident_producer(pipeline) {
        return Ok(admission_rejection(
            "compiled_pipeline_shared_producer_task_kind_unsupported",
            [("producer_task_id", producer.to_string()), ("producer_task_kind", kind.as_str().to_string())],
        ));
    }
    let sink_path = sink_data_path(pipeline);
    if let Some(task) = unmigrated_graph_composition_task(pipeline, &sink_path) {
        return Ok(match task.plan_node() {
            Some(node_id) => node_rejection(
                node_id,
                "compiled_graph_task_composition_not_migrated",
                [("task_id", task.id().to_string()), ("pipeline_task_kind", task.kind().as_str().to_string())],
            ),
            None => admission_rejection(
                "compiled_graph_task_composition_not_migrated",
                [("task_id", task.id().to_string()), ("pipeline_task_kind", task.kind().as_str().to_string())],
            ),
        });
    }
    if let Some(task) = stage_task_outside_sink_path(pipeline, &sink_path) {
        return Ok(admission_rejection(
            "compiled_pipeline_stage_task_position_unsupported",
            [("task_id", task.id().to_string()), ("task_kind", task.kind().as_str().to_string())],
        ));
    }

    Ok(CompiledBoundedPipelineSupport::Supported)
}

#[doc(hidden)]
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

    let join_groups = match join_task_groups(pipeline) {
        JoinTaskGroups::Complete(groups) => groups,
        JoinTaskGroups::Incomplete { plan_node } => {
            return Ok(admission_rejection(
                "compiled_pipeline_join_task_group_incomplete",
                [("plan_node_id", plan_node.map_or_else(|| "none".to_string(), |node| node.to_string()))],
            ));
        },
    };
    for join_node in join_groups {
        let Some(PlanNode::Join(join)) = plan.node(join_node).map(|node| &node.kind) else {
            return Ok(node_rejection(
                join_node,
                "compiled_pipeline_join_task_group_plan_node_mismatch",
                [("plan_node_id", join_node.to_string())],
            ));
        };
        match CompiledJoinExecutionStrategy::from_plan_join_node(plan, join_node, join) {
            Ok(Some(_)) => {},
            Ok(None) | Err(KeyedHashJoinStrategyDecision::Selected { .. }) => {
                return Ok(node_rejection(
                    join_node,
                    "compiled_pipeline_join_execution_strategy_unavailable",
                    [("plan_node_id", join_node.to_string())],
                ));
            },
            Err(KeyedHashJoinStrategyDecision::Rejected { reason, facts }) => {
                return Ok(node_rejection(join_node, reason, facts));
            },
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
/// their component owners prove safe overlap. Windows stay on their existing
/// path until their compiled contract migrates.
#[doc(hidden)]
pub fn pipeline_tasks_within_migrated_contract(pipeline: &CompiledPipeline) -> bool {
    !pipeline.tasks().is_empty()
        && pipeline.tasks().iter().all(|task| {
            matches!(
                task.kind(),
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
                    | PipelineTaskKind::SinkMaterialize
            )
        })
}

fn admission_rejection(
    reason: &'static str,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CompiledBoundedPipelineSupport {
    CompiledBoundedPipelineSupport::Unsupported(CapabilityRejectionProjection {
        owner: CapabilityRejectionOwner::Admission(reason),
        category: CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
        contract_scope: CapabilityScope::Subgraph,
        boundary: "compiled_pipeline",
        reason,
        facts: BTreeMap::from_iter(facts),
    })
}

fn node_rejection(
    node_id: NodeId,
    reason: &'static str,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CompiledBoundedPipelineSupport {
    CompiledBoundedPipelineSupport::Unsupported(CapabilityRejectionProjection {
        owner: CapabilityRejectionOwner::Node(node_id),
        category: CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
        contract_scope: CapabilityScope::Node,
        boundary: "compiled_pipeline",
        reason,
        facts: BTreeMap::from_iter(facts),
    })
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
            "compiled_pipeline_shared_producer_lineage_unproven",
            [("producer_task_id", first.id().to_string())],
        ));
    }
    computed.iter().filter_map(|task| task.plan_node()).find(|node| !satisfied_shared_producers.contains(node)).map(
        |node| {
            admission_rejection(
                "compiled_pipeline_shared_producer_lineage_unproven",
                [("producer_node_id", node.to_string())],
            )
        },
    )
}

/// Graph composition the compiled executor has no output contract for yet.
///
/// A graph or edge-normalize task can feed a join input, where the join task
/// group owns the hand-off. Carrying its output on to the sink through the
/// stages above the joins has no compiled contract; it used to be refused
/// implicitly, by not appearing in a sink-path task-kind allowlist.
fn unmigrated_graph_composition_task<'a>(
    pipeline: &'a CompiledPipeline,
    sink_path: &BTreeSet<PipelineTaskId>,
) -> Option<&'a PipelineTask> {
    pipeline.tasks().iter().find(|task| {
        matches!(task.kind(), PipelineTaskKind::GraphAlgorithm | PipelineTaskKind::EdgeNormalize)
            && sink_path.contains(&task.id())
    })
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
                        consumer.id() == edge.to() && consumer.kind() == PipelineTaskKind::VectorAlgorithm
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
