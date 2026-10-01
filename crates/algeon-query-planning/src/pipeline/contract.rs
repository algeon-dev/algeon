//! Resolution of compiled task and resource execution contracts.
//!
//! The compiled graph records how each task consumes its inputs and what
//! payload form each resource may carry, so the executor reads one decision
//! instead of re-deriving it from the plan on every task.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    capability::{
        CapabilityRejectionCategory, CapabilityRejectionOwner, CapabilityRejectionProjection, CapabilityScope,
        CompiledJoinExecutionStrategy, EdgeConsumer, KeyedHashJoinStrategyOptions, keyed_hash_join_capable,
        keyed_residual_left_match_join_capable,
    },
    expr::{Expr, SortExpr},
    pipeline::model::{
        AcceptedInputForm, PipelineEdge, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind,
        ResidentDataForm, ResidentResource, ResidentResourceId, ResourceOrdering, ResourceSharing, SortStrategy,
    },
    plan::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, ProjectionNode, QueryPlan},
    source::SourceNode,
};

/// Fills in the execution contract of every task and resident resource.
///
/// Tasks are visited in their assigned order, which is a reachable postorder of
/// the plan, so a consumer's contract is always resolved before the resource
/// its producer writes.
pub(super) fn resolve_task_and_resource_contracts(
    plan: &QueryPlan,
    keyed_hash_join: &KeyedHashJoinStrategyOptions,
    tasks: Vec<PipelineTask>,
    resources: Vec<ResidentResource>,
    edges: &[PipelineEdge],
) -> Result<(Vec<PipelineTask>, Vec<ResidentResource>), CapabilityRejectionProjection> {
    let graph = ContractGraph { plan, keyed_hash_join, tasks, resources, edges };
    let graph = graph.with_task_contracts();
    let graph = graph.with_resource_contracts();
    graph.into_parts()
}

struct ContractGraph<'a> {
    plan: &'a QueryPlan,
    keyed_hash_join: &'a KeyedHashJoinStrategyOptions,
    tasks: Vec<PipelineTask>,
    resources: Vec<ResidentResource>,
    edges: &'a [PipelineEdge],
}

impl ContractGraph<'_> {
    fn with_task_contracts(mut self) -> Self {
        for index in 0..self.tasks.len() {
            let contract = match self.tasks[index].kind() {
                PipelineTaskKind::Unary | PipelineTaskKind::Aggregate => {
                    self.tasks[index].clone().with_accepted_input_form(AcceptedInputForm::FrameStreamOrSourceStream)
                },
                PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe => {
                    let mut task =
                        self.tasks[index].clone().with_accepted_input_form(AcceptedInputForm::AnyProducedForm);
                    if let Some(strategy) = self.plan_join_strategy(index) {
                        task = task.with_plan_join_strategy(strategy);
                    }
                    task
                },
                PipelineTaskKind::FinalSort => {
                    let strategy = self.sort_strategy(index);
                    let accepted = match strategy {
                        SortStrategy::FusedTopK { .. } => AcceptedInputForm::FrameOrStream,
                        SortStrategy::FullSort => AcceptedInputForm::Frame,
                    };
                    self.tasks[index].clone().with_accepted_input_form(accepted).with_sort_strategy(strategy)
                },
                PipelineTaskKind::SourceRead
                | PipelineTaskKind::GpuResidentSource
                | PipelineTaskKind::LiteralBatchSource
                | PipelineTaskKind::Window
                | PipelineTaskKind::JoinMaterialize
                | PipelineTaskKind::Limit
                | PipelineTaskKind::Union
                | PipelineTaskKind::EdgeNormalize
                | PipelineTaskKind::GraphAlgorithm
                | PipelineTaskKind::VectorAlgorithm
                | PipelineTaskKind::MlAlgorithm
                | PipelineTaskKind::SinkMaterialize => {
                    self.tasks[index].clone().with_accepted_input_form(AcceptedInputForm::Frame)
                },
            };
            self.tasks[index] = contract;
        }
        self
    }

    fn with_resource_contracts(mut self) -> Self {
        for index in 0..self.resources.len() {
            let data_form = self.resource_data_form(&self.resources[index]);
            let sharing = self.resource_sharing(&self.resources[index], data_form);
            let ordering = self.resource_ordering(&self.resources[index]);
            let resolved = self.resources[index].clone().with_consumption(data_form, sharing).with_ordering(ordering);
            self.resources[index] = resolved;
        }
        self
    }

    /// Row ordering a resource's rows are known to carry.
    ///
    /// A final sort establishes the global order of its sort keys either by
    /// sorting the whole input or by fusing the downstream limit into a
    /// global top-k. Filters and limits preserve their input ordering; a
    /// projection preserves it only when every key column passes through
    /// under its own name. Every other producer makes no ordering promise.
    fn resource_ordering(&self, resource: &ResidentResource) -> Option<ResourceOrdering> {
        let producer = self.task(resource.producer())?;
        match producer.kind() {
            PipelineTaskKind::FinalSort => {
                let node_id = producer.plan_node()?;
                let PlanNode::Sort(sort) = self.plan.node(node_id).map(|node| &node.kind)? else {
                    return None;
                };
                Some(ResourceOrdering::global(sort.spec.expressions.clone()))
            },
            PipelineTaskKind::Unary => {
                let ordering = self.single_input_ordering(producer)?.clone();
                let node_id = producer.plan_node()?;
                match self.plan.node(node_id).map(|node| &node.kind)? {
                    PlanNode::Filter(_) => Some(ordering),
                    PlanNode::Projection(projection)
                        if projection_preserves_ordering_keys(projection, ordering.keys()) =>
                    {
                        Some(ordering)
                    },
                    _ => None,
                }
            },
            PipelineTaskKind::Limit => self.single_input_ordering(producer).cloned(),
            PipelineTaskKind::VectorAlgorithm | PipelineTaskKind::MlAlgorithm => None,
            _ => None,
        }
    }

    fn single_input_ordering(&self, task: &PipelineTask) -> Option<&ResourceOrdering> {
        match task.input_resources() {
            [input] => self.resource(*input).and_then(ResidentResource::ordering),
            _ => None,
        }
    }

    fn into_parts(self) -> Result<(Vec<PipelineTask>, Vec<ResidentResource>), CapabilityRejectionProjection> {
        if let Some(rejection) = self.exclusive_consumer_rejection() {
            return Err(rejection);
        }
        Ok((self.tasks, self.resources))
    }

    /// A task that moves its whole input out of the schedule cannot share it
    /// with a second consumer; the compiler has no conversion that gives both
    /// consumers a correct payload.
    fn exclusive_consumer_rejection(&self) -> Option<CapabilityRejectionProjection> {
        let resource = self.resources.iter().find(|resource| {
            resource.consumers().len() > 1
                && resource
                    .consumers()
                    .iter()
                    .any(|consumer| self.task(*consumer).is_some_and(task_consumes_input_exclusively))
        })?;
        let producer = self.task(resource.producer())?;
        let consumer = resource
            .consumers()
            .iter()
            .filter_map(|consumer| self.task(*consumer))
            .find(|consumer| task_consumes_input_exclusively(consumer))?;
        Some(CapabilityRejectionProjection {
            owner: CapabilityRejectionOwner::Edge {
                producer: producer.plan_node()?,
                consumer: EdgeConsumer::Node(consumer.plan_node()?),
            },
            category: CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
            contract_scope: CapabilityScope::Edge,
            boundary: "compiled_pipeline",
            reason: "consumer_requires_exclusive_input",
            facts: BTreeMap::from([
                ("producer_task", producer.id().to_string()),
                ("producer_task_kind", producer.kind().as_str().to_string()),
                ("consumer_task", consumer.id().to_string()),
                ("consumer_task_kind", consumer.kind().as_str().to_string()),
                ("consumer_count", resource.consumers().len().to_string()),
                ("resident_resource_id", resource.id().to_string()),
            ]),
        })
    }

    fn resource_data_form(&self, resource: &ResidentResource) -> ResidentDataForm {
        let Some(producer) = self.task(resource.producer()) else {
            return ResidentDataForm::Frame;
        };
        match producer.kind() {
            PipelineTaskKind::SourceRead if self.deferrable_source_read(producer) => {
                if self.consumers_accept(resource, AcceptedInputForm::FrameStreamOrSourceStream) {
                    ResidentDataForm::SourceStream
                } else {
                    ResidentDataForm::Frame
                }
            },
            PipelineTaskKind::Unary
                if self.unary_input_form(producer) == ResidentDataForm::SourceStream
                    && self.consumers_accept(resource, AcceptedInputForm::FrameStreamOrSourceStream) =>
            {
                ResidentDataForm::SourceStream
            },
            PipelineTaskKind::Unary | PipelineTaskKind::Aggregate | PipelineTaskKind::JoinMaterialize => {
                if self.consumers_accept_frame_stream(resource) {
                    ResidentDataForm::FrameStream
                } else {
                    ResidentDataForm::Frame
                }
            },
            // A union of fully materialized branch frames carries no wider contract than its
            // branches: when its single consumer pulls chunk by chunk (unary maps, partial
            // aggregates, stream probes, fused top-k), chaining the branch frames avoids a full
            // concat copy and its peak. Frame-only consumers (sink, limit, full sort, window,
            // join strategies that concatenate first) keep the concat. Exact-order downstream
            // consumers drain the stream in branch order before their own required concat, so
            // summation order and branch order are preserved.
            PipelineTaskKind::Union => {
                if resource.consumers().len() == 1 && self.consumers_accept_frame_stream(resource) {
                    ResidentDataForm::FrameStream
                } else {
                    ResidentDataForm::Frame
                }
            },
            PipelineTaskKind::VectorAlgorithm | PipelineTaskKind::MlAlgorithm => ResidentDataForm::Frame,
            _ => ResidentDataForm::Frame,
        }
    }

    fn resource_sharing(&self, resource: &ResidentResource, data_form: ResidentDataForm) -> ResourceSharing {
        if resource.consumers().len() <= 1 {
            return ResourceSharing::SingleConsume;
        }
        if self.payload_opens_independent_cursors(resource, data_form)
            && self.consumers_can_take_independent_cursors(resource)
        {
            ResourceSharing::IndependentCursorStream
        } else {
            ResourceSharing::SharedImmutableFrame
        }
    }

    /// Whether each consumer can open its own cursor over this payload: an
    /// eager frame and a deferred source read can, as can an aggregate's
    /// final-merge stream and a unary map over a payload that can. A lazy
    /// join stream holds probe state it reads once, so its consumers share a
    /// materialized copy instead.
    fn payload_opens_independent_cursors(&self, resource: &ResidentResource, data_form: ResidentDataForm) -> bool {
        match data_form {
            ResidentDataForm::Frame | ResidentDataForm::SourceStream => true,
            ResidentDataForm::FrameStream => match self.task(resource.producer()) {
                Some(producer) if producer.kind() == PipelineTaskKind::Aggregate => true,
                Some(producer) if producer.kind() == PipelineTaskKind::Unary => match producer.input_resources() {
                    [input] => self
                        .resource(*input)
                        .is_some_and(|input| self.payload_opens_independent_cursors(input, input.data_form())),
                    _ => false,
                },
                _ => false,
            },
        }
    }

    fn consumers_accept(&self, resource: &ResidentResource, form: AcceptedInputForm) -> bool {
        !resource.consumers().is_empty()
            && resource.consumers().iter().all(|consumer| {
                self.task(*consumer).is_some_and(|consumer| match consumer.kind() {
                    PipelineTaskKind::Unary => self
                        .single_output(consumer)
                        .and_then(|output| self.resource(output))
                        .is_some_and(|output| self.consumers_accept(output, form)),
                    _ => consumer.accepted_input_form() >= form,
                })
            })
    }

    /// Whether every consumer can pull this resource chunk by chunk.
    ///
    /// Join tasks accept every producer form, but keeping an upstream producer
    /// as a resident frame stream is narrowed to the strategies whose kernels
    /// consume probe chunks directly and to keyed-hash builds; every other
    /// strategy concatenates the stream first, so the narrowing shapes memory
    /// rather than deciding support.
    fn consumers_accept_frame_stream(&self, resource: &ResidentResource) -> bool {
        !resource.consumers().is_empty()
            && resource.consumers().iter().all(|consumer| {
                self.task(*consumer).is_some_and(|consumer| match consumer.kind() {
                    PipelineTaskKind::Unary => self
                        .single_output(consumer)
                        .and_then(|output| self.resource(output))
                        .is_some_and(|output| self.consumers_accept_frame_stream(output)),
                    PipelineTaskKind::JoinBuild => self.join_node(consumer).is_some_and(keyed_hash_join_capable),
                    PipelineTaskKind::JoinProbe => self.join_probe_accepts_frame_stream(consumer),
                    _ => consumer.accepted_input_form() >= AcceptedInputForm::FrameOrStream,
                })
            })
    }

    fn join_probe_accepts_frame_stream(&self, probe: &PipelineTask) -> bool {
        let Some(join) = self.join_node(probe) else {
            return false;
        };
        if join.spec.residual_predicate.is_some() {
            let strategy = probe.plan_join_strategy();
            if keyed_residual_left_match_join_capable(join)
                && strategy == Some(CompiledJoinExecutionStrategy::MixedKeyedResidual)
            {
                return true;
            }
            return strategy == Some(CompiledJoinExecutionStrategy::OneShotEqui);
        }
        match join.spec.join_type {
            JoinType::Inner | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => true,
            JoinType::Left => join.spec.build_side == JoinSide::Right,
            JoinType::Full | JoinType::Cross => false,
        }
    }

    /// A shared producer only stays deferred when every consumer re-streams it
    /// from the producer with its own cursor, and at least one of them is an
    /// aggregate whose chunked input is the reason to defer at all.
    fn consumers_can_take_independent_cursors(&self, resource: &ResidentResource) -> bool {
        let mut feeds_aggregate = false;
        let all_consumers_can_clone = resource.consumers().iter().all(|consumer| {
            let Some(edge) = self.edges.iter().find(|edge| {
                edge.from() == resource.producer()
                    && edge.to() == *consumer
                    && edge.resources().contains(&resource.id())
            }) else {
                return false;
            };
            match edge.kind() {
                PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput => true,
                PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer => {
                    self.task(*consumer).is_some_and(|consumer| match consumer.kind() {
                        PipelineTaskKind::Aggregate => {
                            feeds_aggregate = true;
                            true
                        },
                        PipelineTaskKind::Unary => {
                            let unary_feeds_aggregate = self.task_feeds_aggregate_input(consumer.id());
                            feeds_aggregate |= unary_feeds_aggregate;
                            self.task_feeds_join_input(consumer.id()) || unary_feeds_aggregate
                        },
                        _ => false,
                    })
                },
                PipelineEdgeKind::Control | PipelineEdgeKind::BarrierOutput | PipelineEdgeKind::SinkInput => false,
            }
        });
        all_consumers_can_clone && feeds_aggregate
    }

    fn task_feeds_join_input(&self, task: PipelineTaskId) -> bool {
        let mut pending = vec![task];
        let mut visited = BTreeSet::new();
        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }
            for edge in self.edges.iter().filter(|edge| edge.from() == current) {
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

    fn task_feeds_aggregate_input(&self, task: PipelineTaskId) -> bool {
        let mut pending = vec![task];
        let mut visited = BTreeSet::new();
        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }
            for edge in self.edges.iter().filter(|edge| {
                edge.from() == current
                    && matches!(edge.kind(), PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer)
            }) {
                match self.task(edge.to()).map(PipelineTask::kind) {
                    Some(PipelineTaskKind::Aggregate) => return true,
                    Some(PipelineTaskKind::Unary) => pending.push(edge.to()),
                    _ => {},
                }
            }
        }
        false
    }

    fn sort_strategy(&self, index: usize) -> SortStrategy {
        let task = &self.tasks[index];
        let Some(consumer) = self
            .single_output(task)
            .and_then(|output| self.resource(output))
            .filter(|output| output.consumers().len() == 1)
            .and_then(|output| self.task(output.consumers()[0]))
        else {
            return SortStrategy::FullSort;
        };
        if consumer.kind() != PipelineTaskKind::Limit {
            return SortStrategy::FullSort;
        }
        let (Some(sort_node_id), Some(limit_node_id)) = (task.plan_node(), consumer.plan_node()) else {
            return SortStrategy::FullSort;
        };
        let Some(PlanNode::Limit(limit)) = self.plan.node(limit_node_id).map(|node| &node.kind) else {
            return SortStrategy::FullSort;
        };
        let Some(PlanNode::Sort(sort)) = self.plan.node(sort_node_id).map(|node| &node.kind) else {
            return SortStrategy::FullSort;
        };
        // Finite top-k planning needs a fetch bound (`topk_candidate_rows`
        // rejects an offset-only limit as unbounded): without one the limit
        // runs as an ordinary frame limit below a full sort.
        if limit.spec.fetch.is_none() {
            return SortStrategy::FullSort;
        }
        if limit.input != sort_node_id || !self.sort_input_allows_limit_fusion(sort.input) {
            return SortStrategy::FullSort;
        }
        SortStrategy::FusedTopK { limit_task: consumer.id(), limit_node: limit_node_id }
    }

    /// A directly GPU-resident sort input is already one whole frame, so fusing
    /// the limit into a top-k over it saves nothing and loses the plain sort
    /// path's retry behaviour.
    fn sort_input_allows_limit_fusion(&self, sort_input: NodeId) -> bool {
        !matches!(
            self.plan.node(sort_input).map(|node| &node.kind),
            Some(PlanNode::Source(source)) if matches!(source.as_ref(), SourceNode::GpuResident(_))
        )
    }

    fn plan_join_strategy(&self, index: usize) -> Option<CompiledJoinExecutionStrategy> {
        let task = &self.tasks[index];
        let node_id = task.plan_node()?;
        let join = self.join_node(task)?;
        CompiledJoinExecutionStrategy::from_plan_join_node(self.plan, node_id, join, self.keyed_hash_join)
            .ok()
            .flatten()
    }

    /// A unary stage re-stages whatever its input carries, so its output can
    /// stay a deferred source stream exactly when its input is one. The input
    /// resource is always resolved first: resource ids follow the plan's
    /// reachable postorder.
    fn unary_input_form(&self, task: &PipelineTask) -> ResidentDataForm {
        match task.input_resources() {
            [input] => self.resource(*input).map_or(ResidentDataForm::Frame, ResidentResource::data_form),
            _ => ResidentDataForm::Frame,
        }
    }

    fn deferrable_source_read(&self, task: &PipelineTask) -> bool {
        let Some(PlanNode::Source(source)) =
            task.plan_node().and_then(|node| self.plan.node(node)).map(|node| &node.kind)
        else {
            return false;
        };
        matches!(source.as_ref(), SourceNode::LocalParquet(_))
    }

    fn join_node(&self, task: &PipelineTask) -> Option<&JoinNode> {
        match &self.plan.node(task.plan_node()?)?.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        }
    }

    fn single_output(&self, task: &PipelineTask) -> Option<ResidentResourceId> {
        match task.output_resources() {
            [output] => Some(*output),
            _ => None,
        }
    }

    fn task(&self, id: PipelineTaskId) -> Option<&PipelineTask> {
        self.tasks.get(id.index()).filter(|task| task.id() == id)
    }

    fn resource(&self, id: ResidentResourceId) -> Option<&ResidentResource> {
        self.resources.get(id.index()).filter(|resource| resource.id() == id)
    }
}

fn task_consumes_input_exclusively(task: &PipelineTask) -> bool {
    task.kind() == PipelineTaskKind::EdgeNormalize
}

/// Whether a projection carries every ordering key through under its own
/// name, so a globally ordered input stays globally ordered above it.
fn projection_preserves_ordering_keys(projection: &ProjectionNode, keys: &[SortExpr]) -> bool {
    keys.iter().all(|key| match &key.expr {
        Expr::Column(column) => projection.outputs.iter().any(|output| {
            output.field.name() == column.name()
                && passthrough_column_name(&output.expr).is_some_and(|name| name == column.name())
        }),
        _ => false,
    })
}

fn passthrough_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.name()),
        Expr::Alias { expr, .. } => passthrough_column_name(expr),
        _ => None,
    }
}
