//! QueryPlan to CompiledPipeline graph construction.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    capability::{
        CapabilityAnalysis, CapabilityDecision, EdgeConsumer, EdgeContract, EdgeContractKind, InputRole,
        NodeCapability, ReservationRequirement, ResultSinkKind, SubgraphInvariantKind,
    },
    error::{QueryError, QueryErrorCode},
    pipeline::model::{
        CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
        PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
        ReservationGateOwner, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
    },
    plan::{JoinSide, NodeId, PlanNode, QueryPlan, ResultSink},
    runtime::ReservationKind,
    source::SourceNode,
};

const OPERATION: &str = "cudf_query_engine.pipeline.compile";

pub fn compile_pipeline(plan: &QueryPlan, analysis: &CapabilityAnalysis) -> crate::Result<CompiledPipeline> {
    reject_local_capability_rejections(analysis)?;

    let node_order = reachable_postorder(plan)?;
    let node_set = node_order.iter().copied().collect::<BTreeSet<_>>();
    let dependencies = collect_dependencies(plan, &node_order, &node_set)?;
    validate_analysis_covers_plan(plan, analysis, &node_order, &dependencies)?;
    let task_ids = assign_task_ids(plan, &node_order)?;
    let resource_ids = assign_resource_ids(&node_order);
    let mut gate_builder = GateBuilder::default();

    let mut tasks =
        build_tasks(plan, analysis, &node_order, &dependencies, &task_ids, &resource_ids, &mut gate_builder)?;
    let mut edges = build_plan_edges(plan, analysis, &dependencies, &task_ids, &resource_ids, &mut gate_builder)?;
    append_join_internal_edges(plan, &node_order, analysis, &task_ids, &mut edges)?;

    let sink_compilation =
        append_sink_task(plan, analysis, &task_ids, &resource_ids, &mut tasks, &mut edges, &mut gate_builder)?;
    let resources = build_resources(plan, &node_order, &dependencies, &task_ids, &resource_ids, sink_compilation.task)?;
    let barriers = build_barriers(
        plan,
        &node_order,
        analysis,
        &dependencies,
        &task_ids,
        sink_compilation.task,
        &mut gate_builder,
    )?;
    let sink = build_sink_boundary(plan, analysis, &task_ids, sink_compilation, &mut gate_builder)?;

    let pipeline = CompiledPipeline::new(tasks, edges, barriers, resources, gate_builder.finish(), sink);
    pipeline.validate()?;
    Ok(pipeline)
}

fn reject_local_capability_rejections(analysis: &CapabilityAnalysis) -> crate::Result<()> {
    if let Some(node) =
        analysis.nodes.iter().find(|node| matches!(node.executable, CapabilityDecision::Rejected { .. }))
        && let CapabilityDecision::Rejected { reason, .. } = &node.executable
    {
        return Err(capability_rejection_error("node", reason).with_fact("node_id", node.node_id));
    }

    if let Some(edge) = analysis.edges.iter().find(|edge| matches!(edge.decision, CapabilityDecision::Rejected { .. }))
        && let CapabilityDecision::Rejected { reason, .. } = &edge.decision
    {
        return Err(capability_rejection_error("edge", reason)
            .with_fact("producer", edge.producer)
            .with_fact("consumer", edge_consumer_label(edge.consumer)));
    }

    if let Some(invariant) =
        analysis.invariants.iter().find(|invariant| matches!(invariant.decision, CapabilityDecision::Rejected { .. }))
        && let CapabilityDecision::Rejected { reason, .. } = &invariant.decision
    {
        return Err(capability_rejection_error("invariant", reason)
            .with_fact("invariant_id", invariant.invariant_id.index())
            .with_fact("invariant_kind", invariant_kind_label(&invariant.kind)));
    }

    Ok(())
}

fn capability_rejection_error(scope: &'static str, reason: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        OPERATION,
        "pipeline compiler cannot compile a rejected capability analysis",
    )
    .with_fact("capability_scope", scope)
    .with_fact("capability_reason", reason)
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

#[derive(Debug, Default)]
struct GateBuilder {
    gates: Vec<ReservationGate>,
}

impl GateBuilder {
    fn attach_requirements<'a>(
        &mut self,
        owner: ReservationGateOwner,
        requirements: impl IntoIterator<Item = &'a ReservationRequirement>,
    ) -> Vec<ReservationGateId> {
        requirements
            .into_iter()
            .map(|requirement| {
                let id = ReservationGateId::new(self.gates.len());
                self.gates.push(ReservationGate::from_requirement(id, owner, requirement.clone()));
                id
            })
            .collect()
    }

    fn finish(self) -> Vec<ReservationGate> {
        self.gates
    }
}

#[derive(Debug, Clone, Copy)]
struct SinkCompilation {
    task: Option<PipelineTaskId>,
    edge: Option<PipelineEdgeId>,
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

fn reachable_postorder(plan: &QueryPlan) -> crate::Result<Vec<NodeId>> {
    let mut order = Vec::new();
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    visit_reachable(plan, plan.sink().input(), &mut visiting, &mut visited, &mut order)?;
    Ok(order)
}

fn visit_reachable(
    plan: &QueryPlan,
    node_id: NodeId,
    visiting: &mut BTreeSet<NodeId>,
    visited: &mut BTreeSet<NodeId>,
    order: &mut Vec<NodeId>,
) -> crate::Result<()> {
    if visited.contains(&node_id) {
        return Ok(());
    }
    if !visiting.insert(node_id) {
        return Err(invalid("query plan contains a cycle reachable from the sink").with_fact("node_id", node_id));
    }
    let node = plan
        .node(node_id)
        .ok_or_else(|| invalid("query plan sink references a missing node").with_fact("node_id", node_id))?;
    for input in node.input_ids() {
        visit_reachable(plan, *input, visiting, visited, order)?;
    }
    visiting.remove(&node_id);
    visited.insert(node_id);
    order.push(node_id);
    Ok(())
}

fn collect_dependencies(
    plan: &QueryPlan,
    node_order: &[NodeId],
    node_set: &BTreeSet<NodeId>,
) -> crate::Result<Vec<PlanDependency>> {
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
) -> crate::Result<()> {
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

fn input_role(kind: &PlanNode, input_index: usize) -> InputRole {
    match kind {
        PlanNode::Aggregate(_) => InputRole::AggregateInput,
        PlanNode::Window(_) => InputRole::WindowInput,
        PlanNode::Sort(_) => InputRole::SortInput,
        PlanNode::Limit(_) => InputRole::LimitInput,
        PlanNode::Union(_) => InputRole::UnionInput { input_index },
        PlanNode::Join(join) => {
            let side = if input_index == 0 { JoinSide::Left } else { JoinSide::Right };
            if side == join.spec.build_side {
                InputRole::JoinBuildInput { side }
            } else {
                InputRole::JoinProbeInput { side }
            }
        },
        _ => InputRole::UnaryInput,
    }
}

fn assign_task_ids(plan: &QueryPlan, node_order: &[NodeId]) -> crate::Result<BTreeMap<NodeId, NodeTasks>> {
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
    analysis: &CapabilityAnalysis,
    node_order: &[NodeId],
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
    gate_builder: &mut GateBuilder,
) -> crate::Result<Vec<PipelineTask>> {
    let mut tasks = Vec::new();
    for node_id in node_order {
        let node = plan
            .node(*node_id)
            .ok_or_else(|| invalid("reachable query plan node is missing").with_fact("node_id", node_id))?;
        let capability = node_capability_for(analysis, *node_id)?;
        let output = resource_for(resource_ids, *node_id)?;
        match &node.kind {
            PlanNode::Source(source) => {
                let kind = match source.as_ref() {
                    SourceNode::LocalParquet(_) => PipelineTaskKind::SourceRead,
                    SourceNode::GpuResident(_) => PipelineTaskKind::GpuResidentSource,
                };
                let task = single_task(task_ids, *node_id)?;
                let gates = gate_builder
                    .attach_requirements(ReservationGateOwner::Task(task), capability.local_requirements.iter());
                tasks.push(
                    PipelineTask::for_plan_node(task, kind, *node_id)
                        .with_output_resources(vec![output])
                        .with_gates(gates),
                );
            },
            PlanNode::Filter(_) | PlanNode::Projection(_) => {
                let task = single_task(task_ids, *node_id)?;
                let gates = gate_builder
                    .attach_requirements(ReservationGateOwner::Task(task), capability.local_requirements.iter());
                tasks.push(
                    PipelineTask::for_plan_node(task, PipelineTaskKind::Unary, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output])
                        .with_gates(gates),
                );
            },
            PlanNode::Aggregate(_) => {
                let task = single_task(task_ids, *node_id)?;
                let gates = gate_builder
                    .attach_requirements(ReservationGateOwner::Task(task), capability.local_requirements.iter());
                tasks.push(
                    PipelineTask::for_plan_node(task, PipelineTaskKind::Aggregate, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output])
                        .with_gates(gates),
                );
            },
            PlanNode::Window(_) => {
                let task = single_task(task_ids, *node_id)?;
                let gates = gate_builder
                    .attach_requirements(ReservationGateOwner::Task(task), capability.local_requirements.iter());
                tasks.push(
                    PipelineTask::for_plan_node(task, PipelineTaskKind::Window, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output])
                        .with_gates(gates),
                );
            },
            PlanNode::Sort(_) => {
                let task = single_task(task_ids, *node_id)?;
                let gates = gate_builder
                    .attach_requirements(ReservationGateOwner::Task(task), capability.local_requirements.iter());
                tasks.push(
                    PipelineTask::for_plan_node(task, PipelineTaskKind::FinalSort, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output])
                        .with_gates(gates),
                );
            },
            PlanNode::Limit(_) => {
                let task = single_task(task_ids, *node_id)?;
                let gates = gate_builder
                    .attach_requirements(ReservationGateOwner::Task(task), capability.local_requirements.iter());
                tasks.push(
                    PipelineTask::for_plan_node(task, PipelineTaskKind::Limit, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output])
                        .with_gates(gates),
                );
            },
            PlanNode::Union(_) => {
                let task = single_task(task_ids, *node_id)?;
                let gates = gate_builder
                    .attach_requirements(ReservationGateOwner::Task(task), capability.local_requirements.iter());
                tasks.push(
                    PipelineTask::for_plan_node(task, PipelineTaskKind::Union, *node_id)
                        .with_input_resources(input_resources(*node_id, dependencies, resource_ids)?)
                        .with_output_resources(vec![output])
                        .with_gates(gates),
                );
            },
            PlanNode::Join(_) => {
                let NodeTasks::Join { build, probe, materialize } = tasks_for(task_ids, *node_id)? else {
                    return Err(invalid("join node was not assigned join pipeline tasks").with_fact("node_id", node_id));
                };
                let build_gates = gate_builder.attach_requirements(
                    ReservationGateOwner::Task(build),
                    capability.local_requirements.iter().filter(|requirement| {
                        join_node_requirement_task(requirement.kind, build, materialize) == build
                    }),
                );
                let materialize_gates = gate_builder.attach_requirements(
                    ReservationGateOwner::Task(materialize),
                    capability.local_requirements.iter().filter(|requirement| {
                        join_node_requirement_task(requirement.kind, build, materialize) == materialize
                    }),
                );
                tasks.push(
                    PipelineTask::for_plan_node(build, PipelineTaskKind::JoinBuild, *node_id)
                        .with_input_resources(join_input_resources(
                            *node_id,
                            InputRoleSelector::Build,
                            dependencies,
                            resource_ids,
                        )?)
                        .with_gates(build_gates),
                );
                tasks.push(
                    PipelineTask::for_plan_node(probe, PipelineTaskKind::JoinProbe, *node_id).with_input_resources(
                        join_input_resources(*node_id, InputRoleSelector::Probe, dependencies, resource_ids)?,
                    ),
                );
                tasks.push(
                    PipelineTask::for_plan_node(materialize, PipelineTaskKind::JoinMaterialize, *node_id)
                        .with_output_resources(vec![output])
                        .with_gates(materialize_gates),
                );
            },
        }
    }
    Ok(tasks)
}

fn join_node_requirement_task(
    kind: ReservationKind,
    build: PipelineTaskId,
    materialize: PipelineTaskId,
) -> PipelineTaskId {
    match kind {
        ReservationKind::HashTable => build,
        ReservationKind::FilteredProbeState | ReservationKind::MixedSizeData => materialize,
        ReservationKind::OperatorOutput
        | ReservationKind::SourceCacheRetained
        | ReservationKind::ChunkOperator
        | ReservationKind::PartialState
        | ReservationKind::FinalMerge
        | ReservationKind::SourceRead
        | ReservationKind::BuildSide
        | ReservationKind::ProbeSide
        | ReservationKind::GatherMap
        | ReservationKind::JoinOutput
        | ReservationKind::FinalSort
        | ReservationKind::Materialization
        | ReservationKind::CapabilityContract => materialize,
    }
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
) -> crate::Result<Vec<ResidentResourceId>> {
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
) -> crate::Result<Vec<ResidentResourceId>> {
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
    gate_builder: &mut GateBuilder,
) -> crate::Result<Vec<PipelineEdge>> {
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
        let gates = gate_builder
            .attach_requirements(ReservationGateOwner::Edge(edge_id), contract.reservation_requirements.iter());
        edges.push(
            PipelineEdge::new(edge_id, producer_task, consumer_task, kind)
                .with_resources(vec![resource_for(resource_ids, dependency.producer)?])
                .with_gates(gates),
        );
    }

    if plan.node_count() == 0 {
        return Err(invalid("query plan has no nodes"));
    }

    Ok(edges)
}

fn append_join_internal_edges(
    plan: &QueryPlan,
    node_order: &[NodeId],
    analysis: &CapabilityAnalysis,
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    edges: &mut Vec<PipelineEdge>,
) -> crate::Result<()> {
    for node_id in node_order {
        let Some(node) = plan.node(*node_id) else {
            return Err(invalid("reachable query plan node is missing").with_fact("node_id", node_id));
        };
        if !matches!(node.kind, PlanNode::Join(_)) {
            continue;
        }
        require_build_before_probe_invariant(analysis, *node_id)?;
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

fn append_sink_task(
    plan: &QueryPlan,
    analysis: &CapabilityAnalysis,
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
    tasks: &mut Vec<PipelineTask>,
    edges: &mut Vec<PipelineEdge>,
    gate_builder: &mut GateBuilder,
) -> crate::Result<SinkCompilation> {
    let ResultSink::HostArrow { input, .. } = plan.sink() else {
        return Ok(SinkCompilation { task: None, edge: None });
    };
    let sink_task = PipelineTaskId::new(tasks.len());
    let input_resource = resource_for(resource_ids, *input)?;
    tasks.push(
        PipelineTask::new(sink_task, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![input_resource]),
    );
    let edge_id = PipelineEdgeId::new(edges.len());
    let sink_contract = sink_edge_contract(plan, analysis)?;
    let edge_gates = gate_builder.attach_requirements(
        ReservationGateOwner::Edge(edge_id),
        sink_contract
            .reservation_requirements
            .iter()
            .filter(|requirement| requirement.kind != ReservationKind::Materialization),
    );
    edges.push(
        PipelineEdge::new(edge_id, tasks_for(task_ids, *input)?.output_task(), sink_task, PipelineEdgeKind::SinkInput)
            .with_resources(vec![input_resource])
            .with_gates(edge_gates),
    );
    Ok(SinkCompilation { task: Some(sink_task), edge: Some(edge_id) })
}

fn build_resources(
    plan: &QueryPlan,
    node_order: &[NodeId],
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
    sink_task: Option<PipelineTaskId>,
) -> crate::Result<Vec<ResidentResource>> {
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
            .collect::<crate::Result<Vec<_>>>()?;
        if plan.sink().input() == *node_id
            && let Some(sink_task) = sink_task
        {
            consumers.push(sink_task);
        }
        consumers.sort_unstable();
        consumers.dedup();
        let lifetime =
            if consumers.is_empty() && matches!(plan.sink(), ResultSink::GpuResident { input } if *input == *node_id) {
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
    analysis: &CapabilityAnalysis,
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    sink_task: Option<PipelineTaskId>,
    gate_builder: &mut GateBuilder,
) -> crate::Result<Vec<PipelineBarrier>> {
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
                require_build_before_probe_invariant(analysis, *node_id)?;
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
                let build_before_probe_gates = gate_builder.attach_requirements(
                    ReservationGateOwner::Barrier(build_before_probe_barrier),
                    barrier_requirements_for_node(analysis, PipelineBarrierKind::JoinBuildBeforeProbe, *node_id),
                );
                barriers.push(
                    PipelineBarrier::new(build_before_probe_barrier, PipelineBarrierKind::JoinBuildBeforeProbe)
                        .with_upstream_tasks(vec![build])
                        .with_downstream_tasks(vec![probe])
                        .with_gates(build_before_probe_gates),
                );
                let materialization_barrier = PipelineBarrierId::new(barriers.len());
                let materialization_gates = gate_builder.attach_requirements(
                    ReservationGateOwner::Barrier(materialization_barrier),
                    barrier_requirements_for_node(analysis, PipelineBarrierKind::JoinMaterialization, *node_id),
                );
                barriers.push(
                    PipelineBarrier::new(materialization_barrier, PipelineBarrierKind::JoinMaterialization)
                        .with_upstream_tasks(vec![probe])
                        .with_downstream_tasks(vec![materialize])
                        .with_gates(materialization_gates),
                );
            },
            PlanNode::Sort(_) => {
                let barrier_id = PipelineBarrierId::new(barriers.len());
                let gates = gate_builder.attach_requirements(
                    ReservationGateOwner::Barrier(barrier_id),
                    barrier_requirements_for_node(analysis, PipelineBarrierKind::SortBoundary, *node_id),
                );
                barriers.push(
                    PipelineBarrier::new(barrier_id, PipelineBarrierKind::SortBoundary)
                        .with_upstream_tasks(vec![single_task(task_ids, *node_id)?])
                        .with_downstream_tasks(downstream_tasks(*node_id, dependencies, task_ids)?)
                        .with_gates(gates),
                );
            },
            PlanNode::Source(_)
            | PlanNode::Filter(_)
            | PlanNode::Projection(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_) => {},
        }
    }
    let sink_input = sink_task.unwrap_or_else(|| {
        tasks_for(task_ids, plan.sink().input()).expect("sink input task must be assigned").output_task()
    });
    let sink_barrier = PipelineBarrierId::new(barriers.len());
    let sink_gates = gate_builder.attach_requirements(
        ReservationGateOwner::Barrier(sink_barrier),
        barrier_requirements_for_node(analysis, PipelineBarrierKind::SinkBoundary, plan.sink().input()),
    );
    barriers.push(
        PipelineBarrier::new(sink_barrier, PipelineBarrierKind::SinkBoundary)
            .with_upstream_tasks(vec![sink_input])
            .with_gates(sink_gates),
    );
    Ok(barriers)
}

fn barrier_requirements_for_node(
    analysis: &CapabilityAnalysis,
    barrier_kind: PipelineBarrierKind,
    node_id: NodeId,
) -> Vec<&ReservationRequirement> {
    analysis
        .invariants
        .iter()
        .filter(|invariant| invariant_matches_barrier_node(invariant, barrier_kind, node_id))
        .flat_map(|invariant| invariant.reservation_requirements.iter())
        .collect()
}

fn invariant_matches_barrier_node(
    invariant: &crate::capability::SubgraphInvariant,
    barrier_kind: PipelineBarrierKind,
    node_id: NodeId,
) -> bool {
    match (&invariant.kind, barrier_kind) {
        (SubgraphInvariantKind::BuildBeforeProbe { join_id }, PipelineBarrierKind::JoinBuildBeforeProbe) => {
            *join_id == node_id
        },
        (SubgraphInvariantKind::FinalBarrierOrder, PipelineBarrierKind::SortBoundary) => {
            invariant.nodes.contains(&node_id)
        },
        (SubgraphInvariantKind::PostRootStageOrder { root_join_id }, PipelineBarrierKind::JoinMaterialization) => {
            *root_join_id == node_id
        },
        _ => false,
    }
}

fn downstream_tasks(
    producer: NodeId,
    dependencies: &[PlanDependency],
    task_ids: &BTreeMap<NodeId, NodeTasks>,
) -> crate::Result<Vec<PipelineTaskId>> {
    let mut tasks = dependencies
        .iter()
        .filter(|dependency| dependency.producer == producer)
        .map(|dependency| tasks_for(task_ids, dependency.consumer).map(|tasks| tasks.input_task(dependency.input_role)))
        .collect::<crate::Result<Vec<_>>>()?;
    tasks.sort_unstable();
    tasks.dedup();
    Ok(tasks)
}

fn build_sink_boundary(
    plan: &QueryPlan,
    analysis: &CapabilityAnalysis,
    task_ids: &BTreeMap<NodeId, NodeTasks>,
    sink_compilation: SinkCompilation,
    gate_builder: &mut GateBuilder,
) -> crate::Result<SinkBoundary> {
    let kind = match plan.sink() {
        ResultSink::HostArrow { .. } => SinkBoundaryKind::HostArrowMaterialization,
        ResultSink::GpuResident { .. } => SinkBoundaryKind::GpuResidentOutput,
    };
    let input = sink_compilation.task.unwrap_or_else(|| {
        tasks_for(task_ids, plan.sink().input()).expect("sink input task must be assigned").output_task()
    });
    let sink_contract = sink_edge_contract(plan, analysis)?;
    let sink_gates = gate_builder.attach_requirements(
        ReservationGateOwner::SinkBoundary,
        sink_contract.reservation_requirements.iter().filter(|requirement| {
            requirement.kind == ReservationKind::Materialization || sink_compilation.edge.is_none()
        }),
    );
    Ok(SinkBoundary::new(kind, input).with_gates(sink_gates))
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

fn sink_edge_contract<'a>(plan: &QueryPlan, analysis: &'a CapabilityAnalysis) -> crate::Result<&'a EdgeContract> {
    let consumer = EdgeConsumer::Sink(sink_consumer_kind(plan.sink()));
    analysis.edges.iter().find(|edge| edge.producer == plan.sink().input() && edge.consumer == consumer).ok_or_else(
        || {
            invalid("capability analysis does not cover the query sink edge")
                .with_fact("producer", plan.sink().input())
                .with_fact("consumer", edge_consumer_label(consumer))
        },
    )
}

fn node_capability_for(analysis: &CapabilityAnalysis, node_id: NodeId) -> crate::Result<&NodeCapability> {
    analysis.nodes.iter().find(|node| node.node_id == node_id).ok_or_else(|| {
        invalid("capability analysis does not cover a sink-reachable query node").with_fact("node_id", node_id)
    })
}

fn edge_kind_for_dependency(contract: &EdgeContract, input_role: InputRole) -> PipelineEdgeKind {
    match input_role {
        InputRole::JoinBuildInput { .. } => PipelineEdgeKind::JoinBuildInput,
        InputRole::JoinProbeInput { .. } => PipelineEdgeKind::JoinProbeInput,
        _ if { contract.contract_kinds.iter().any(|kind| matches!(kind, EdgeContractKind::SharedProducer { .. })) } => {
            PipelineEdgeKind::SharedResidentProducer
        },
        _ => PipelineEdgeKind::ResidentFrame,
    }
}

fn require_build_before_probe_invariant(analysis: &CapabilityAnalysis, join_id: NodeId) -> crate::Result<()> {
    let Some(invariant) = analysis
        .invariants
        .iter()
        .find(|invariant| matches!(invariant.kind, SubgraphInvariantKind::BuildBeforeProbe { join_id: candidate } if candidate == join_id))
    else {
        return Err(invalid("join node is missing build-before-probe invariant").with_fact("node_id", join_id));
    };
    if invariant.decision != CapabilityDecision::Satisfied {
        return Err(invalid("join build-before-probe invariant is not satisfied")
            .with_fact("node_id", join_id)
            .with_fact("invariant_id", invariant.invariant_id.index()));
    }
    Ok(())
}

fn tasks_for(task_ids: &BTreeMap<NodeId, NodeTasks>, node_id: NodeId) -> crate::Result<NodeTasks> {
    task_ids
        .get(&node_id)
        .copied()
        .ok_or_else(|| invalid("query plan node has no assigned pipeline task").with_fact("node_id", node_id))
}

fn single_task(task_ids: &BTreeMap<NodeId, NodeTasks>, node_id: NodeId) -> crate::Result<PipelineTaskId> {
    match tasks_for(task_ids, node_id)? {
        NodeTasks::Single { task } => Ok(task),
        NodeTasks::Join { .. } => {
            Err(invalid("join node does not have a single pipeline task").with_fact("node_id", node_id))
        },
    }
}

fn resource_for(
    resource_ids: &BTreeMap<NodeId, ResidentResourceId>,
    node_id: NodeId,
) -> crate::Result<ResidentResourceId> {
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
        ResultSink::GpuResident { .. } => ResultSinkKind::GpuResident,
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
        InputRole::JoinBuildInput { .. } => "join_build_input",
        InputRole::JoinProbeInput { .. } => "join_probe_input",
        InputRole::SinkInput => "sink_input",
    }
}

fn invariant_kind_label(kind: &SubgraphInvariantKind) -> &'static str {
    match kind {
        SubgraphInvariantKind::BuildBeforeProbe { .. } => "build_before_probe",
        SubgraphInvariantKind::ClosedJoinSubtree { .. } => "closed_join_subtree",
        SubgraphInvariantKind::TopologicalBuildProbeSchedule => "topological_build_probe_schedule",
        SubgraphInvariantKind::SharedResidentSubplan => "shared_resident_subplan",
        SubgraphInvariantKind::AggregateDerivesFromSharedProducer => "aggregate_derives_from_shared_producer",
        SubgraphInvariantKind::FinalBarrierOrder => "final_barrier_order",
        SubgraphInvariantKind::PostRootStageOrder { .. } => "post_root_stage_order",
        SubgraphInvariantKind::ProjectionTransparentResidency => "projection_transparent_residency",
        SubgraphInvariantKind::ResidentUnaryChainPreservesResidency => "resident_unary_chain_preserves_residency",
        SubgraphInvariantKind::ResidualUsesBothInputs { .. } => "residual_uses_both_inputs",
        SubgraphInvariantKind::ResidentLifetimeDominatesConsumers => "resident_lifetime_dominates_consumers",
    }
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
