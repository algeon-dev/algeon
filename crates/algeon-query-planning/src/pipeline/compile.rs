//! QueryPlan to CompiledPipeline graph construction.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    contract::{plan_joins, resolve_task_and_resource_contracts},
    physical::{AggregatePlan, JoinPlan, SourceReadPlan, TaskPhysical},
    validate::{ScheduleRejection, validate_schedule},
};
use crate::{
    capability::{
        CapabilityAnalysis, EdgeConsumer, EdgeContract, EdgeContractKind, InputRole, Reason, Rejection, RejectionOwner,
        ResultSinkKind, VectorInputRole, input_role,
    },
    dynamic_filter::{DynamicFilterConsumer, DynamicFilterProducer},
    error::{QueryError, QueryErrorCode, Result},
    pipeline::model::{
        CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
        PipelineEdgeKind, PipelineTask, PipelineTaskId, ResidentLifetime, ResidentResource, ResidentResourceId,
        SinkBoundary, SinkBoundaryKind, SortStrategy,
    },
    plan::{MlRelationRole, NodeId, PlanNode, QueryPlan, ResultSink},
    source::SourceNode,
};

const OPERATION: &str = "query_engine.pipeline.compile";

/// Outcome of compiling one plan into a validated bounded pipeline.
///
/// A plan the compiler or the schedule validator refuses is a structured
/// rejection, not an error: admission reports it with the contract that
/// failed. Malformed inputs the compiler cannot classify stay `Err`.
#[derive(Debug)]
pub(crate) enum PipelineCompilation {
    /// The plan compiled into a schedule that satisfies every pipeline contract.
    Compiled(CompiledPipeline),
    /// A compiler or schedule contract rejects the plan.
    Rejected(Rejection),
}

/// Compile a native plan and matching capability analysis into a symbolic pipeline.
///
/// `analysis` must cover the same reachable plan graph. Any local node, edge,
/// or invariant rejection is returned as an error instead of being converted to
/// a report-family fallback. The resulting graph is consumed by the one
/// capability-first bounded executor.
pub fn compile_pipeline(plan: &QueryPlan, analysis: &CapabilityAnalysis) -> Result<CompiledPipeline> {
    let pipeline = match compile_pipeline_graph(plan, analysis)? {
        GraphCompilation::Graph(pipeline) => pipeline,
        GraphCompilation::Rejected(rejection) => return Err(contract_rejection_error(&rejection)),
    };
    validate_schedule(&pipeline).map_err(ScheduleRejection::into_error)?;
    Ok(pipeline)
}

/// Compiles a plan and reports a refused plan as a structured rejection.
pub(crate) fn compile_pipeline_outcome(plan: &QueryPlan, analysis: &CapabilityAnalysis) -> Result<PipelineCompilation> {
    let pipeline = match compile_pipeline_graph(plan, analysis) {
        Ok(GraphCompilation::Graph(pipeline)) => pipeline,
        Ok(GraphCompilation::Rejected(rejection)) => return Ok(PipelineCompilation::Rejected(rejection)),
        Err(error) if error.code() == QueryErrorCode::InvalidPlan => {
            return Ok(PipelineCompilation::Rejected(compile_rejection(
                Reason::CompiledPipelineGraphNotConstructible,
                &error,
            )));
        },
        Err(error) => return Err(error),
    };
    match validate_schedule(&pipeline) {
        Ok(()) => Ok(PipelineCompilation::Compiled(pipeline)),
        Err(rejection) => Ok(PipelineCompilation::Rejected(compile_rejection(rejection.reason(), rejection.error()))),
    }
}

fn compile_rejection(reason: Reason, error: &QueryError) -> Rejection {
    let mut facts = error.facts().iter().map(|(key, value)| (key, value.to_string())).collect::<BTreeMap<_, _>>();
    facts.insert("pipeline_compile_error", error.message().to_string());
    Rejection { owner: RejectionOwner::Plan, reason, facts }
}

/// A constructed graph, or the execution contract that refuses to connect it.
enum GraphCompilation {
    Graph(CompiledPipeline),
    Rejected(Rejection),
}

fn contract_rejection_error(rejection: &Rejection) -> QueryError {
    rejection.facts.iter().fold(
        QueryError::new(
            QueryErrorCode::UnsupportedPipelineContract,
            OPERATION,
            "compiled pipeline refuses a rejected part of the plan",
        )
        .with_fact("capability_reason", rejection.reason)
        .with_fact("rejection_owner", rejection.owner),
        |error, (key, value)| error.with_fact(key, value.as_str()),
    )
}

fn compile_pipeline_graph(plan: &QueryPlan, analysis: &CapabilityAnalysis) -> Result<GraphCompilation> {
    if let Some(rejection) = analysis.first_rejection() {
        return Ok(GraphCompilation::Rejected(rejection));
    }

    let node_order = reachable_postorder(plan);
    let node_set = node_order.iter().copied().collect::<BTreeSet<_>>();
    let dependencies = collect_dependencies(plan, &node_order, &node_set)?;
    validate_analysis_covers_plan(plan, analysis, &node_order, &dependencies)?;
    let task_ids = assign_task_ids(plan, &node_order)?;
    let resource_ids = assign_resource_ids(&node_order);
    let join_plans = match plan_joins(plan, &node_order, &analysis.policy.keyed_hash_join) {
        Ok(join_plans) => join_plans,
        Err(rejection) => return Ok(GraphCompilation::Rejected(rejection)),
    };
    let mut tasks = build_tasks(plan, &node_order, &dependencies, &task_ids, &resource_ids, &join_plans)?;
    let mut edges = build_plan_edges(plan, analysis, &dependencies, &task_ids, &resource_ids)?;
    append_join_internal_edges(plan, &node_order, &task_ids, &mut edges)?;
    append_dynamic_filter_control_edges(plan, &task_ids, &mut edges)?;

    let sink_task = append_sink_task(plan, &task_ids, &resource_ids, &mut tasks, &mut edges)?;
    let resources = build_resources(plan, &node_order, &dependencies, &task_ids, &resource_ids, sink_task)?;
    let (tasks, resources) = match resolve_task_and_resource_contracts(plan, tasks, resources, &edges) {
        Ok(resolved) => resolved,
        Err(rejection) => return Ok(GraphCompilation::Rejected(rejection)),
    };
    let barriers = build_barriers(plan, &node_order, &dependencies, &task_ids, sink_task)?;
    let sink = build_sink_boundary(plan, &task_ids, sink_task)?;

    Ok(GraphCompilation::Graph(CompiledPipeline::new(tasks, edges, barriers, resources, sink)))
}

#[derive(Debug, Clone, Copy)]
struct PlanDependency {
    producer: NodeId,
    consumer: NodeId,
    input_role: InputRole,
}

#[derive(Debug, Clone, Copy)]
enum NodeTasks {
    Single { task: PipelineTaskId },
    Join { build: PipelineTaskId, probe: PipelineTaskId, materialize: PipelineTaskId },
}

impl NodeTasks {
    fn output_task(self) -> PipelineTaskId {
        match self {
            Self::Single { task } => task,
            Self::Join { materialize, .. } => materialize,
        }
    }

    fn input_task(self, input_role: InputRole) -> PipelineTaskId {
        match (self, input_role) {
            (Self::Join { build, .. }, InputRole::JoinBuildInput { .. }) => build,
            (Self::Join { probe, .. }, InputRole::JoinProbeInput { .. }) => probe,
            (Self::Single { task }, _) => task,
            (Self::Join { probe, .. }, _) => probe,
        }
    }

    fn join_build(self) -> Option<PipelineTaskId> {
        match self {
            Self::Join { build, .. } => Some(build),
            Self::Single { .. } => None,
        }
    }

    fn join_probe(self) -> Option<PipelineTaskId> {
        match self {
            Self::Join { probe, .. } => Some(probe),
            Self::Single { .. } => None,
        }
    }

    fn join_materialize(self) -> Option<PipelineTaskId> {
        match self {
            Self::Join { materialize, .. } => Some(materialize),
            Self::Single { .. } => None,
        }
    }
}

/// Every node in dependency post-order from the sink, so each input precedes
/// its consumers. Plan construction makes every node reachable.
fn reachable_postorder(plan: &QueryPlan) -> Vec<NodeId> {
    let mut order = Vec::with_capacity(plan.node_count());
    let mut visited = vec![false; plan.node_count()];
    visit_reachable(plan, plan.sink().input(), &mut visited, &mut order);
    order
}

fn visit_reachable(plan: &QueryPlan, node_id: NodeId, visited: &mut [bool], order: &mut Vec<NodeId>) {
    if std::mem::replace(&mut visited[node_id.index()], true) {
        return;
    }
    for input in plan.nodes()[node_id.index()].input_ids() {
        visit_reachable(plan, *input, visited, order);
    }
    order.push(node_id);
}

fn collect_dependencies(
    plan: &QueryPlan,
    node_order: &[NodeId],
    node_set: &BTreeSet<NodeId>,
) -> Result<Vec<PlanDependency>> {
    let mut dependencies = Vec::new();
    for consumer in node_order {
        let node = plan
            .node(*consumer)
            .ok_or_else(|| invalid("reachable query plan node is missing").with_fact("node_id", consumer))?;
        for (input_index, producer) in node.input_ids().iter().enumerate() {
            if !node_set.contains(producer) {
                return Err(invalid("query plan dependency producer is not sink-reachable")
                    .with_fact("producer", producer)
                    .with_fact("consumer", consumer));
            }
            dependencies.push(PlanDependency {
                producer: *producer,
                consumer: *consumer,
                input_role: input_role(&node.kind, input_index),
            });
        }
    }
    Ok(dependencies)
}

fn validate_analysis_covers_plan(
    plan: &QueryPlan,
    analysis: &CapabilityAnalysis,
    node_order: &[NodeId],
    dependencies: &[PlanDependency],
) -> Result<()> {
    for node_id in node_order {
        if analysis.nodes.iter().all(|node| node.node_id != *node_id) {
            return Err(
                invalid("capability analysis does not cover a sink-reachable query node").with_fact("node_id", node_id)
            );
        }
    }
    for dependency in dependencies {
        if edge_contract_for_dependency(analysis, dependency).is_none() {
            return Err(invalid("capability analysis does not cover a query plan dependency")
                .with_fact("producer", dependency.producer)
                .with_fact("consumer", dependency.consumer)
                .with_fact("input_role", input_role_label(dependency.input_role)));
        }
    }
    let sink_consumer = EdgeConsumer::Sink(sink_consumer_kind(plan.sink()));
    if analysis.edges.iter().all(|edge| edge.producer != plan.sink().input() || edge.consumer != sink_consumer) {
        return Err(invalid("capability analysis does not cover the query sink edge")
            .with_fact("producer", plan.sink().input())
            .with_fact("consumer", edge_consumer_label(sink_consumer)));
    }
    Ok(())
}

fn assign_task_ids(plan: &QueryPlan, node_order: &[NodeId]) -> Result<BTreeMap<NodeId, NodeTasks>> {
    let mut task_ids = BTreeMap::new();
    let mut next = 0;
    for node_id in node_order {
        let node = plan
            .node(*node_id)
            .ok_or_else(|| invalid("reachable query plan node is missing").with_fact("node_id", node_id))?;
        let tasks = if matches!(node.kind, PlanNode::Join(_)) {
            let build = PipelineTaskId::new(next);
            let probe = PipelineTaskId::new(next + 1);
            let materialize = PipelineTaskId::new(next + 2);
            next += 3;
            NodeTasks::Join { build, probe, materialize }
        } else {
            let task = PipelineTaskId::new(next);
            next += 1;
            NodeTasks::Single { task }
        };
        task_ids.insert(*node_id, tasks);
    }
    Ok(task_ids)
}

fn assign_resource_ids(node_order: &[NodeId]) -> BTreeMap<NodeId, ResidentResourceId> {
    node_order.iter().enumerate().map(|(index, node_id)| (*node_id, ResidentResourceId::new(index))).collect()
}

fn build_tasks(
    plan: &QueryPlan,
    node_order: &[NodeId],
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
    join_plans: &BTreeMap<NodeId, JoinPlan>,
) -> Result<Vec<PipelineTask>> {
    let mut tasks = Vec::new();
    for node_id in node_order {
        let node = plan
            .node(*node_id)
            .ok_or_else(|| invalid("reachable query plan node is missing").with_fact("node_id", node_id))?;
        let output = resource_for(resource_ids, *node_id)?;
        match &node.kind {
            PlanNode::Source(source) => {
                let physical = match source.as_ref() {
                    SourceNode::LocalParquet(_) => TaskPhysical::SourceRead(SourceReadPlan::FRAME),
                    SourceNode::GpuResident(_) => TaskPhysical::GpuResidentSource,
                    SourceNode::LiteralBatch(_) => TaskPhysical::LiteralBatchSource,
                };
                let task = single_task(task_ids, *node_id)?;
                tasks.push(PipelineTask::for_plan_node(task, physical, *node_id).with_output_resources(vec![output]));
            },
            PlanNode::Filter(_) | PlanNode::Projection(_) => {
                let task = single_task(task_ids, *node_id)?;
                tasks.push(
                    PipelineTask::for_plan_node(task, TaskPhysical::Unary, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::Aggregate(_) => {
                let task = single_task(task_ids, *node_id)?;
                tasks.push(
                    PipelineTask::for_plan_node(task, TaskPhysical::Aggregate(AggregatePlan::default()), *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::Window(_) => {
                let task = single_task(task_ids, *node_id)?;
                tasks.push(
                    PipelineTask::for_plan_node(task, TaskPhysical::Window, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::Sort(_) => {
                let task = single_task(task_ids, *node_id)?;
                tasks.push(
                    PipelineTask::for_plan_node(task, TaskPhysical::FinalSort(SortStrategy::FullSort), *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::Limit(_) => {
                let task = single_task(task_ids, *node_id)?;
                tasks.push(
                    PipelineTask::for_plan_node(task, TaskPhysical::Limit, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::Union(_) => {
                let task = single_task(task_ids, *node_id)?;
                tasks.push(
                    PipelineTask::for_plan_node(task, TaskPhysical::Union, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::GraphAlgorithm(_) => {
                let task = single_task(task_ids, *node_id)?;
                tasks.push(
                    PipelineTask::for_plan_node(task, TaskPhysical::GraphAlgorithm, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::VectorAlgorithm(_) | PlanNode::MlAlgorithm(_) => {
                let task = single_task(task_ids, *node_id)?;
                let physical = if matches!(node.kind, PlanNode::MlAlgorithm(_)) {
                    TaskPhysical::MlAlgorithm
                } else {
                    TaskPhysical::VectorAlgorithm
                };
                tasks.push(
                    PipelineTask::for_plan_node(task, physical, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::EdgeNormalize(_) => {
                let task = single_task(task_ids, *node_id)?;
                tasks.push(
                    PipelineTask::for_plan_node(task, TaskPhysical::EdgeNormalize, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output]),
                );
            },
            PlanNode::Join(_) => {
                let NodeTasks::Join { build, probe, materialize } = tasks_for(task_ids, *node_id)? else {
                    return Err(invalid("join node was not assigned join pipeline tasks").with_fact("node_id", node_id));
                };
                let join_plan = join_plans
                    .get(node_id)
                    .ok_or_else(|| invalid("join node has no join plan").with_fact("node_id", node_id))?;
                tasks.push(
                    PipelineTask::for_plan_node(build, TaskPhysical::JoinBuild(join_plan.clone()), *node_id)
                        .with_input_resources(join_input_resources(
                            *node_id,
                            InputRoleSelector::Build,
                            dependencies,
                            resource_ids,
                        )?),
                );
                tasks.push(
                    PipelineTask::for_plan_node(probe, TaskPhysical::JoinProbe(join_plan.clone()), *node_id)
                        .with_input_resources(join_input_resources(
                            *node_id,
                            InputRoleSelector::Probe,
                            dependencies,
                            resource_ids,
                        )?),
                );
                tasks.push(
                    PipelineTask::for_plan_node(materialize, TaskPhysical::JoinMaterialize, *node_id)
                        .with_output_resources(vec![output]),
                );
            },
        }
    }
    Ok(tasks)
}

#[derive(Debug, Clone, Copy)]
enum InputRoleSelector {
    Build,
    Probe,
}

fn input_resources(
    consumer: NodeId,
    dependencies: &[PlanDependency],
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
) -> Result<Vec<ResidentResourceId>> {
    dependencies
        .iter()
        .filter(|dependency| dependency.consumer == consumer)
        .map(|dependency| resource_for(resource_ids, dependency.producer))
        .collect()
}

fn join_input_resources(
    consumer: NodeId,
    selector: InputRoleSelector,
    dependencies: &[PlanDependency],
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
) -> Result<Vec<ResidentResourceId>> {
    dependencies
        .iter()
        .filter(|dependency| {
            dependency.consumer == consumer
                && matches!(
                    (selector, dependency.input_role),
                    (InputRoleSelector::Build, InputRole::JoinBuildInput { .. })
                        | (InputRoleSelector::Probe, InputRole::JoinProbeInput { .. })
                )
        })
        .map(|dependency| resource_for(resource_ids, dependency.producer))
        .collect()
}

fn build_plan_edges(
    plan: &QueryPlan,
    analysis: &CapabilityAnalysis,
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
) -> Result<Vec<PipelineEdge>> {
    let mut edges = Vec::new();
    for dependency in dependencies {
        let contract = edge_contract_for_dependency(analysis, dependency).ok_or_else(|| {
            invalid("capability analysis does not cover a query plan dependency")
                .with_fact("producer", dependency.producer)
                .with_fact("consumer", dependency.consumer)
                .with_fact("input_role", input_role_label(dependency.input_role))
        })?;
        let producer_task = tasks_for(task_ids, dependency.producer)?.output_task();
        let consumer_task = tasks_for(task_ids, dependency.consumer)?.input_task(dependency.input_role);
        let edge_id = PipelineEdgeId::new(edges.len());
        let kind = edge_kind_for_dependency(contract, dependency.input_role);
        edges.push(
            PipelineEdge::new(edge_id, producer_task, consumer_task, kind)
                .with_resources(vec![resource_for(resource_ids, dependency.producer)?]),
        );
    }

    if plan.node_count() == 0 {
        return Err(invalid("query plan has no nodes"));
    }

    Ok(edges)
}

/// Emits the build-before-probe and probe-before-materialize ordering a join
/// task group always requires. The schedule validator proves the resulting
/// graph stays acyclic once dynamic-filter readiness edges are added.
fn append_join_internal_edges(
    plan: &QueryPlan,
    node_order: &[NodeId],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    edges: &mut Vec<PipelineEdge>,
) -> Result<()> {
    for node_id in node_order {
        let Some(node) = plan.node(*node_id) else {
            return Err(invalid("reachable query plan node is missing").with_fact("node_id", node_id));
        };
        if !matches!(node.kind, PlanNode::Join(_)) {
            continue;
        }
        let tasks = tasks_for(task_ids, *node_id)?;
        let build = tasks
            .join_build()
            .ok_or_else(|| invalid("join node is missing build task").with_fact("node_id", node_id))?;
        let probe = tasks
            .join_probe()
            .ok_or_else(|| invalid("join node is missing probe task").with_fact("node_id", node_id))?;
        let materialize = tasks
            .join_materialize()
            .ok_or_else(|| invalid("join node is missing materialize task").with_fact("node_id", node_id))?;
        edges.push(PipelineEdge::new(PipelineEdgeId::new(edges.len()), build, probe, PipelineEdgeKind::Control));
        edges.push(PipelineEdge::new(PipelineEdgeId::new(edges.len()), probe, materialize, PipelineEdgeKind::Control));
    }
    Ok(())
}

fn append_dynamic_filter_control_edges(
    plan: &QueryPlan,
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    edges: &mut Vec<PipelineEdge>,
) -> Result<()> {
    for route in plan.dynamic_filter_routes() {
        let Some(producer) = dynamic_filter_producer_task(task_ids, &route.producer)? else {
            continue;
        };
        let Some(consumer) = dynamic_filter_consumer_task(task_ids, &route.consumer)? else {
            continue;
        };
        if producer == consumer || edges.iter().any(|edge| edge.from() == producer && edge.to() == consumer) {
            continue;
        }
        edges.push(PipelineEdge::new(PipelineEdgeId::new(edges.len()), producer, consumer, PipelineEdgeKind::Control));
    }
    Ok(())
}

fn dynamic_filter_producer_task(
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    producer: &DynamicFilterProducer,
) -> Result<Option<PipelineTaskId>> {
    match producer {
        DynamicFilterProducer::HashJoinBuild { join_node_id, .. } => tasks_for(task_ids, *join_node_id)
            .and_then(|tasks| {
                tasks.join_build().ok_or_else(|| invalid("dynamic-filter hash-join producer is missing a build task"))
            })
            .map(Some),
        DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } => {
            tasks_for(task_ids, *aggregate_node_id).map(|tasks| Some(tasks.output_task()))
        },
    }
}

fn dynamic_filter_consumer_task(
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    consumer: &DynamicFilterConsumer,
) -> Result<Option<PipelineTaskId>> {
    match consumer {
        DynamicFilterConsumer::LocalParquetSource { source_node_id } => {
            tasks_for(task_ids, *source_node_id).map(|tasks| Some(tasks.output_task()))
        },
        DynamicFilterConsumer::HashJoinProbe { join_node_id, .. } => tasks_for(task_ids, *join_node_id)
            .and_then(|tasks| {
                tasks.join_probe().ok_or_else(|| invalid("dynamic-filter hash-join consumer is missing a probe task"))
            })
            .map(Some),
        DynamicFilterConsumer::Filter { filter_node_id } => {
            tasks_for(task_ids, *filter_node_id).map(|tasks| Some(tasks.output_task()))
        },
    }
}

fn append_sink_task(
    plan: &QueryPlan,
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
    tasks: &mut Vec<PipelineTask>,
    edges: &mut Vec<PipelineEdge>,
) -> Result<Option<PipelineTaskId>> {
    let ResultSink::HostArrow { input, .. } = plan.sink() else {
        return Ok(None);
    };
    let sink_task = PipelineTaskId::new(tasks.len());
    let input_resource = resource_for(resource_ids, *input)?;
    tasks.push(PipelineTask::new(sink_task, TaskPhysical::SinkMaterialize).with_input_resources(vec![input_resource]));
    let edge_id = PipelineEdgeId::new(edges.len());
    edges.push(
        PipelineEdge::new(edge_id, tasks_for(task_ids, *input)?.output_task(), sink_task, PipelineEdgeKind::SinkInput)
            .with_resources(vec![input_resource]),
    );
    Ok(Some(sink_task))
}

fn build_resources(
    plan: &QueryPlan,
    node_order: &[NodeId],
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
    sink_task: Option<PipelineTaskId>,
) -> Result<Vec<ResidentResource>> {
    let mut resources = Vec::new();
    for node_id in node_order {
        let resource = resource_for(resource_ids, *node_id)?;
        let producer = tasks_for(task_ids, *node_id)?.output_task();
        let mut consumers = dependencies
            .iter()
            .filter(|dependency| dependency.producer == *node_id)
            .map(|dependency| {
                tasks_for(task_ids, dependency.consumer).map(|tasks| tasks.input_task(dependency.input_role))
            })
            .collect::<Result<Vec<_>>>()?;
        if plan.sink().input() == *node_id
            && let Some(sink_task) = sink_task
        {
            consumers.push(sink_task);
        }
        consumers.sort_unstable();
        consumers.dedup();
        let lifetime = if consumers.is_empty()
            && matches!(plan.sink(), ResultSink::GpuResident { input } | ResultSink::DeferredHostArrow { input, .. } if *input == *node_id)
        {
            ResidentLifetime::SinkOutput { producer }
        } else if consumers.is_empty() {
            ResidentLifetime::TaskLocal { task: producer }
        } else {
            ResidentLifetime::BetweenTasks { producer }
        };
        resources.push(ResidentResource::new(resource, producer, consumers, lifetime));
    }
    Ok(resources)
}

fn build_barriers(
    plan: &QueryPlan,
    node_order: &[NodeId],
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    sink_task: Option<PipelineTaskId>,
) -> Result<Vec<PipelineBarrier>> {
    let mut barriers = Vec::new();
    for node_id in node_order {
        let node = plan
            .node(*node_id)
            .ok_or_else(|| invalid("reachable query plan node is missing").with_fact("node_id", node_id))?;
        match &node.kind {
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => {
                let barrier_id = PipelineBarrierId::new(barriers.len());
                barriers.push(
                    PipelineBarrier::new(barrier_id, PipelineBarrierKind::SourceChunkBoundary)
                        .with_upstream_tasks(vec![single_task(task_ids, *node_id)?])
                        .with_downstream_tasks(downstream_tasks(*node_id, dependencies, task_ids)?),
                );
            },
            PlanNode::Aggregate(_) | PlanNode::Window(_) => {
                let barrier_id = PipelineBarrierId::new(barriers.len());
                barriers.push(
                    PipelineBarrier::new(barrier_id, PipelineBarrierKind::AggregateBoundary)
                        .with_upstream_tasks(vec![single_task(task_ids, *node_id)?])
                        .with_downstream_tasks(downstream_tasks(*node_id, dependencies, task_ids)?),
                );
            },
            PlanNode::Join(_) => {
                let tasks = tasks_for(task_ids, *node_id)?;
                let build = tasks
                    .join_build()
                    .ok_or_else(|| invalid("join node is missing build task").with_fact("node_id", node_id))?;
                let probe = tasks
                    .join_probe()
                    .ok_or_else(|| invalid("join node is missing probe task").with_fact("node_id", node_id))?;
                let materialize = tasks
                    .join_materialize()
                    .ok_or_else(|| invalid("join node is missing materialize task").with_fact("node_id", node_id))?;
                let build_before_probe_barrier = PipelineBarrierId::new(barriers.len());
                barriers.push(
                    PipelineBarrier::new(build_before_probe_barrier, PipelineBarrierKind::JoinBuildBeforeProbe)
                        .with_upstream_tasks(vec![build])
                        .with_downstream_tasks(vec![probe]),
                );
                let materialization_barrier = PipelineBarrierId::new(barriers.len());
                barriers.push(
                    PipelineBarrier::new(materialization_barrier, PipelineBarrierKind::JoinMaterialization)
                        .with_upstream_tasks(vec![probe])
                        .with_downstream_tasks(vec![materialize]),
                );
            },
            PlanNode::Sort(_) => {
                let barrier_id = PipelineBarrierId::new(barriers.len());
                barriers.push(
                    PipelineBarrier::new(barrier_id, PipelineBarrierKind::SortBoundary)
                        .with_upstream_tasks(vec![single_task(task_ids, *node_id)?])
                        .with_downstream_tasks(downstream_tasks(*node_id, dependencies, task_ids)?),
                );
            },
            PlanNode::Source(_)
            | PlanNode::Filter(_)
            | PlanNode::Projection(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_)
            | PlanNode::MlAlgorithm(_) => {},
        }
    }
    let sink_input = match sink_task {
        Some(task) => task,
        None => tasks_for(task_ids, plan.sink().input())?.output_task(),
    };
    let sink_barrier = PipelineBarrierId::new(barriers.len());
    barriers.push(
        PipelineBarrier::new(sink_barrier, PipelineBarrierKind::SinkBoundary).with_upstream_tasks(vec![sink_input]),
    );
    Ok(barriers)
}

fn downstream_tasks(
    producer: NodeId,
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
) -> Result<Vec<PipelineTaskId>> {
    let mut tasks = dependencies
        .iter()
        .filter(|dependency| dependency.producer == producer)
        .map(|dependency| tasks_for(task_ids, dependency.consumer).map(|tasks| tasks.input_task(dependency.input_role)))
        .collect::<Result<Vec<_>>>()?;
    tasks.sort_unstable();
    tasks.dedup();
    Ok(tasks)
}

fn build_sink_boundary(
    plan: &QueryPlan,
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    sink_task: Option<PipelineTaskId>,
) -> Result<SinkBoundary> {
    let kind = match plan.sink() {
        ResultSink::HostArrow { .. } => SinkBoundaryKind::HostArrowMaterialization,
        ResultSink::GpuResident { .. } | ResultSink::DeferredHostArrow { .. } => SinkBoundaryKind::GpuResidentOutput,
    };
    let input = match sink_task {
        Some(task) => task,
        None => tasks_for(task_ids, plan.sink().input())?.output_task(),
    };
    Ok(SinkBoundary::new(kind, input))
}

fn edge_contract_for_dependency<'a>(
    analysis: &'a CapabilityAnalysis,
    dependency: &PlanDependency,
) -> Option<&'a EdgeContract> {
    analysis.edges.iter().find(|edge| {
        edge.producer == dependency.producer
            && edge.consumer == EdgeConsumer::Node(dependency.consumer)
            && edge.input_role == dependency.input_role
    })
}

fn edge_kind_for_dependency(contract: &EdgeContract, input_role: InputRole) -> PipelineEdgeKind {
    match input_role {
        InputRole::JoinBuildInput { .. } => PipelineEdgeKind::JoinBuildInput,
        InputRole::JoinProbeInput { .. } => PipelineEdgeKind::JoinProbeInput,
        _ if contract.contract_kinds.contains(&EdgeContractKind::SharedProducer) => {
            PipelineEdgeKind::SharedResidentProducer
        },
        _ => PipelineEdgeKind::ResidentFrame,
    }
}

fn tasks_for(task_ids: &BTreeMap<NodeId, NodeTasks>, node_id: NodeId) -> Result<NodeTasks> {
    task_ids
        .get(&node_id)
        .copied()
        .ok_or_else(|| invalid("query plan node has no assigned pipeline task").with_fact("node_id", node_id))
}

fn single_task(task_ids: &BTreeMap<NodeId, NodeTasks>, node_id: NodeId) -> Result<PipelineTaskId> {
    match tasks_for(task_ids, node_id)? {
        NodeTasks::Single { task } => Ok(task),
        NodeTasks::Join { .. } => {
            Err(invalid("join node does not have a single pipeline task").with_fact("node_id", node_id))
        },
    }
}

fn resource_for(resource_ids: &BTreeMap<NodeId, ResidentResourceId>, node_id: NodeId) -> Result<ResidentResourceId> {
    resource_ids
        .get(&node_id)
        .copied()
        .ok_or_else(|| invalid("query plan node has no assigned resident resource").with_fact("node_id", node_id))
}

fn edge_consumer_label(consumer: EdgeConsumer) -> String {
    match consumer {
        EdgeConsumer::Node(node) => node.to_string(),
        EdgeConsumer::Sink(kind) => format!("sink:{kind:?}"),
    }
}

fn sink_consumer_kind(sink: &ResultSink) -> ResultSinkKind {
    match sink {
        ResultSink::HostArrow { .. } => ResultSinkKind::HostArrow,
        ResultSink::GpuResident { .. } | ResultSink::DeferredHostArrow { .. } => ResultSinkKind::GpuResident,
    }
}

fn input_role_label(role: InputRole) -> &'static str {
    match role {
        InputRole::UnaryInput => "unary_input",
        InputRole::AggregateInput => "aggregate_input",
        InputRole::WindowInput => "window_input",
        InputRole::SortInput => "sort_input",
        InputRole::LimitInput => "limit_input",
        InputRole::UnionInput { .. } => "union_input",
        InputRole::GraphAlgorithmInput => "graph_algorithm_input",
        InputRole::MlAlgorithmInput { role: MlRelationRole::Input } => "ml_input",
        InputRole::MlAlgorithmInput { role: MlRelationRole::Training } => "ml_training_input",
        InputRole::MlAlgorithmInput { role: MlRelationRole::Predict } => "ml_predict_input",
        InputRole::VectorAlgorithmInput { role } => match role {
            VectorInputRole::Dataset => "vector_dataset_input",
            VectorInputRole::Queries => "vector_queries_input",
            VectorInputRole::Input => "vector_input",
            VectorInputRole::X => "vector_x_input",
            VectorInputRole::Y => "vector_y_input",
            VectorInputRole::InitialCentroids => "vector_initial_centroids_input",
        },
        InputRole::JoinBuildInput { .. } => "join_build_input",
        InputRole::JoinProbeInput { .. } => "join_probe_input",
        InputRole::SinkInput => "sink_input",
    }
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
