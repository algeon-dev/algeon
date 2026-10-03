//! Resolution of compiled task and resource execution contracts.
//!
//! The compiled graph records how each task consumes its inputs and what
//! payload form each resource may carry, so the executor reads one decision
//! instead of re-deriving it from the plan on every task.

use std::collections::{BTreeMap, BTreeSet};

use arrow::datatypes::DataType;

use crate::{
    capability::{
        EdgeConsumer, JoinForm, KeyedHashJoinStrategyOptions, Reason, Rejection, RejectionOwner, choose_join_strategy,
        classify_join, join_probe_streams_chunks, keyed_hash_join_strategy_input_facts_for_plan,
    },
    dynamic_filter::{DynamicFilterConsumer, DynamicFilterProducer},
    expr::{Expr, SortExpr},
    pipeline::{
        model::{
            AcceptedInputForm, PipelineEdge, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind,
            ResidentDataForm, ResidentResource, ResidentResourceId, ResourceOrdering, ResourceSharing, SortStrategy,
        },
        physical::{
            AggregateInputHints, AggregatePlan, JoinPlan, SourceOutput, SourceReadPlan, TaskPhysical, TopKFusion,
        },
    },
    plan::{AggregateNode, JoinNode, NodeId, PlanNode, ProjectionNode, QueryPlan},
    source::SourceNode,
};

/// Decides each join's operator form and initial strategy once, from the
/// plan alone. A join no executor runs refuses the graph.
pub(super) fn plan_joins(
    plan: &QueryPlan,
    node_order: &[NodeId],
    keyed_hash_join: &KeyedHashJoinStrategyOptions,
) -> Result<BTreeMap<NodeId, JoinPlan>, Rejection> {
    let mut plans = BTreeMap::new();
    for node_id in node_order {
        let Some(PlanNode::Join(join)) = plan.node(*node_id).map(|node| &node.kind) else {
            continue;
        };
        let form = classify_join(*node_id, join)?;
        let input = keyed_hash_join_strategy_input_facts_for_plan(plan, *node_id, join);
        let planned = choose_join_strategy(form, join, input, keyed_hash_join).strategy();
        plans.insert(*node_id, JoinPlan::for_form(form, planned));
    }
    Ok(plans)
}

/// Fills in the execution contract of every task and resident resource.
///
/// Tasks are visited in their assigned order, which is a reachable postorder of
/// the plan, so a consumer's contract is always resolved before the resource
/// its producer writes.
pub(super) fn resolve_task_and_resource_contracts(
    plan: &QueryPlan,
    tasks: Vec<PipelineTask>,
    resources: Vec<ResidentResource>,
    edges: &[PipelineEdge],
) -> Result<(Vec<PipelineTask>, Vec<ResidentResource>), Rejection> {
    let graph = ContractGraph { plan, tasks, resources, edges };
    let graph = graph.with_task_contracts();
    let graph = graph.with_resource_contracts();
    let graph = graph.with_join_strategies_narrowed_to_probe_input();
    let graph = graph.with_source_read_plans();
    let graph = graph.with_aggregate_plans();
    graph.into_parts()
}

struct ContractGraph<'a> {
    plan: &'a QueryPlan,
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
                    self.tasks[index].clone().with_accepted_input_form(AcceptedInputForm::AnyProducedForm)
                },
                PipelineTaskKind::FinalSort => {
                    let strategy = self.sort_strategy(index);
                    let accepted = match strategy {
                        SortStrategy::FusedTopK { .. } => AcceptedInputForm::FrameOrStream,
                        SortStrategy::FullSort => AcceptedInputForm::Frame,
                    };
                    self.tasks[index]
                        .clone()
                        .with_accepted_input_form(accepted)
                        .with_physical(TaskPhysical::FinalSort(strategy))
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
            // Consumers of a shared immutable resource read one copy, so the
            // producer materializes a deferred source read instead of handing
            // each consumer a source it would read again.
            let data_form =
                if sharing == ResourceSharing::SharedImmutableFrame && data_form == ResidentDataForm::SourceStream {
                    ResidentDataForm::Frame
                } else {
                    data_form
                };
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

    /// A probe input kept as a frame stream was planned from the planned
    /// strategy's probe pulling it chunk by chunk. A refinement may pick only
    /// strategies whose probe does the same, so the memory shape admission
    /// assumed holds.
    fn with_join_strategies_narrowed_to_probe_input(mut self) -> Self {
        let streamed_probe_joins = self
            .tasks
            .iter()
            .filter(|task| task.kind() == PipelineTaskKind::JoinProbe)
            .filter(|probe| match probe.input_resources() {
                [input] => {
                    self.resource(*input).is_some_and(|input| input.data_form() == ResidentDataForm::FrameStream)
                },
                _ => false,
            })
            .filter_map(PipelineTask::plan_node)
            .collect::<BTreeSet<_>>();
        for index in 0..self.tasks.len() {
            let Some(node_id) = self.tasks[index].plan_node().filter(|node| streamed_probe_joins.contains(node)) else {
                continue;
            };
            let Some(PlanNode::Join(join)) = self.plan.node(node_id).map(|node| &node.kind) else {
                continue;
            };
            let physical = match self.tasks[index].physical() {
                TaskPhysical::JoinBuild(plan) => TaskPhysical::JoinBuild(
                    plan.narrowed(|strategy| join_probe_streams_chunks(join, plan.form(), strategy)),
                ),
                TaskPhysical::JoinProbe(plan) => TaskPhysical::JoinProbe(
                    plan.narrowed(|strategy| join_probe_streams_chunks(join, plan.form(), strategy)),
                ),
                _ => continue,
            };
            self.tasks[index] = self.tasks[index].clone().with_physical(physical);
        }
        self
    }

    fn with_source_read_plans(mut self) -> Self {
        for index in 0..self.tasks.len() {
            if self.tasks[index].kind() != PipelineTaskKind::SourceRead {
                continue;
            }
            let plan = self.source_read_plan(&self.tasks[index]);
            self.tasks[index] = self.tasks[index].clone().with_physical(TaskPhysical::SourceRead(plan));
        }
        self
    }

    /// How a local Parquet source task reads its source, from its output
    /// resource contract and the consumers it reaches.
    ///
    /// A source that feeds a join, or that applies dynamic-filter routes for
    /// its only consumer, reads as a join leaf. A deferred output stays a
    /// stream shaped for the join inputs and aggregates it feeds; any other
    /// source reads into one frame.
    fn source_read_plan(&self, task: &PipelineTask) -> SourceReadPlan {
        let (Some(node_id), Some(output)) =
            (task.plan_node(), self.single_output(task).and_then(|output| self.resource(output)))
        else {
            return SourceReadPlan::FRAME;
        };
        let multiple_consumers = output.sharing() != ResourceSharing::SingleConsume;
        // A resource consumers share immutably never stays a source stream,
        // so a source-stream output is one its consumers may take as is.
        let deferred = output.data_form() == ResidentDataForm::SourceStream;
        let feeds_direct_join = self.task_feeds_direct_join_input(task.id());
        let join_adjacent = self.task_feeds_join_input_through_unary_chain(task.id());
        let aggregate_path = self.aggregate_input_path(task.id());
        let feeds_aggregate = aggregate_path.is_some();
        let aggregate_adjacent = aggregate_path == Some(true);
        let feeds_join = feeds_direct_join || join_adjacent;
        let routes = self.source_has_dynamic_filter_routes(node_id);
        let (output, prefetch_eligible) = if !deferred && (feeds_join || (routes && !multiple_consumers)) {
            (SourceOutput::JoinLeafFrame { adjacent_stages: join_adjacent }, false)
        } else if !deferred || (!feeds_join && !feeds_aggregate && !routes) {
            (SourceOutput::Frame, false)
        } else {
            let adjacent_stages = join_adjacent || aggregate_adjacent;
            let chunk_hints = AggregateInputHints {
                adjacent_stages,
                grouped_string_key: adjacent_stages
                    && self
                        .reachable_aggregate(task.id())
                        .is_some_and(|aggregate| self.aggregate_groups_by_string_column(aggregate)),
            };
            let output = if feeds_aggregate && feeds_join {
                SourceOutput::SharedJoinAggregateStream { adjacent_stages, chunk_hints }
            } else if feeds_aggregate {
                SourceOutput::AggregateInputStream { adjacent_stages: aggregate_adjacent, chunk_hints }
            } else {
                SourceOutput::JoinLeafStream { adjacent_stages: join_adjacent }
            };
            (output, feeds_direct_join && !multiple_consumers && !routes)
        };
        SourceReadPlan { output, dynamic_filter_routes: routes, prefetch_eligible }
    }

    fn with_aggregate_plans(mut self) -> Self {
        for index in 0..self.tasks.len() {
            if self.tasks[index].kind() != PipelineTaskKind::Aggregate {
                continue;
            }
            let topk = self.aggregate_topk_fusion(&self.tasks[index]);
            self.tasks[index] =
                self.tasks[index].clone().with_physical(TaskPhysical::Aggregate(AggregatePlan { topk }));
        }
        self
    }

    /// Propagates a final sort's fused top-k through an exclusively consumed
    /// projection chain to the aggregate below it. An aggregate that produces
    /// a dynamic filter keeps its complete output for that filter.
    fn aggregate_topk_fusion(&self, task: &PipelineTask) -> Option<TopKFusion> {
        let node_id = task.plan_node()?;
        if self.plan.dynamic_filter_routes().iter().any(|route| {
            matches!(
                route.producer,
                DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } if aggregate_node_id == node_id
            )
        }) {
            return None;
        }
        let mut current = task;
        let mut post_aggregate_projections = Vec::new();
        loop {
            let resource = self.resource(self.single_output(current)?)?;
            let [consumer] = resource.consumers() else {
                return None;
            };
            current = self.task(*consumer)?;
            let current_node = current.plan_node()?;
            match &self.plan.node(current_node)?.kind {
                PlanNode::Projection(_) => post_aggregate_projections.push(current_node),
                PlanNode::Sort(_) => {
                    let SortStrategy::FusedTopK { limit_node, .. } = current.sort_strategy()? else {
                        return None;
                    };
                    let PlanNode::Limit(_) = &self.plan.node(limit_node)?.kind else {
                        return None;
                    };
                    return Some(TopKFusion { post_aggregate_projections, sort_node: current_node, limit_node });
                },
                _ => return None,
            }
        }
    }

    fn task_feeds_direct_join_input(&self, task: PipelineTaskId) -> bool {
        self.edges.iter().any(|edge| {
            edge.from() == task
                && matches!(edge.kind(), PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput)
        })
    }

    /// Whether the task's output reaches a join input through at least one
    /// unary stage. A chain whose last resource several join inputs share
    /// still feeds those joins.
    fn task_feeds_join_input_through_unary_chain(&self, task: PipelineTaskId) -> bool {
        [PipelineEdgeKind::JoinBuildInput, PipelineEdgeKind::JoinProbeInput]
            .into_iter()
            .any(|target| self.join_input_through_unary_chain(task, target))
    }

    fn join_input_through_unary_chain(&self, task: PipelineTaskId, target: PipelineEdgeKind) -> bool {
        let mut current = task;
        let mut saw_unary = false;
        loop {
            let Some(resource) =
                self.task(current).and_then(|task| self.single_output(task)).and_then(|output| self.resource(output))
            else {
                return false;
            };
            let edge_kind = |consumer: PipelineTaskId| {
                self.edges.iter().find(|edge| edge.from() == current && edge.to() == consumer)
            };
            if resource.consumers().len() > 1 {
                return saw_unary
                    && resource.consumers().iter().all(|consumer| {
                        edge_kind(*consumer).is_some_and(|edge| {
                            matches!(edge.kind(), PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput)
                        })
                    })
                    && resource
                        .consumers()
                        .iter()
                        .any(|consumer| edge_kind(*consumer).is_some_and(|edge| edge.kind() == target));
            }
            let [consumer] = resource.consumers() else {
                return false;
            };
            let Some(edge) = edge_kind(*consumer) else {
                return false;
            };
            match edge.kind() {
                kind if kind == target => return saw_unary,
                PipelineEdgeKind::ResidentFrame
                    if self.task(*consumer).is_some_and(|consumer| consumer.kind() == PipelineTaskKind::Unary) =>
                {
                    saw_unary = true;
                    current = *consumer;
                },
                _ => return false,
            }
        }
    }

    /// `Some(false)` when an aggregate consumes the task's output directly,
    /// `Some(true)` when one does only through unary stages, and `None` when
    /// no aggregate does.
    fn aggregate_input_path(&self, task: PipelineTaskId) -> Option<bool> {
        let mut pending = vec![(task, false)];
        let mut visited = BTreeSet::new();
        while let Some((current, saw_unary)) = pending.pop() {
            if !visited.insert((current, saw_unary)) {
                continue;
            }
            for edge in self.edges.iter().filter(|edge| {
                edge.from() == current
                    && matches!(edge.kind(), PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer)
            }) {
                match self.task(edge.to()).map(PipelineTask::kind) {
                    Some(PipelineTaskKind::Aggregate) => return Some(saw_unary),
                    Some(PipelineTaskKind::Unary) => pending.push((edge.to(), true)),
                    _ => {},
                }
            }
        }
        None
    }

    /// The first aggregate the task's output reaches through unary stages.
    fn reachable_aggregate(&self, task: PipelineTaskId) -> Option<&AggregateNode> {
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
                let Some(consumer) = self.task(edge.to()) else {
                    continue;
                };
                match consumer.kind() {
                    PipelineTaskKind::Aggregate => {
                        return match &self.plan.node(consumer.plan_node()?)?.kind {
                            PlanNode::Aggregate(aggregate) => Some(aggregate),
                            _ => None,
                        };
                    },
                    PipelineTaskKind::Unary => pending.push(edge.to()),
                    _ => {},
                }
            }
        }
        None
    }

    /// Whether a plain grouped aggregate keys on a string column of its input.
    fn aggregate_groups_by_string_column(&self, aggregate: &AggregateNode) -> bool {
        let Some(input) = self.plan.node(aggregate.input) else {
            return false;
        };
        !aggregate.spec.group_exprs.is_empty()
            && aggregate.spec.grouping_sets.is_none()
            && aggregate.spec.group_exprs.iter().filter_map(passthrough_column_name).any(|column| {
                input.output_schema().field_with_name(column).is_ok_and(|field| {
                    matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
                })
            })
    }

    fn source_has_dynamic_filter_routes(&self, source_node_id: NodeId) -> bool {
        self.plan.dynamic_filter_routes().iter().any(|route| {
            matches!(
                &route.consumer,
                DynamicFilterConsumer::LocalParquetSource { source_node_id: consumer } if *consumer == source_node_id
            )
        })
    }

    fn into_parts(self) -> Result<(Vec<PipelineTask>, Vec<ResidentResource>), Rejection> {
        if let Some(rejection) = self.exclusive_consumer_rejection() {
            return Err(rejection);
        }
        Ok((self.tasks, self.resources))
    }

    /// A task that moves its whole input out of the schedule cannot share it
    /// with a second consumer; the compiler has no conversion that gives both
    /// consumers a correct payload.
    fn exclusive_consumer_rejection(&self) -> Option<Rejection> {
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
        Some(Rejection {
            owner: RejectionOwner::Edge {
                producer: producer.plan_node()?,
                consumer: EdgeConsumer::Node(consumer.plan_node()?),
            },
            reason: Reason::ConsumerRequiresExclusiveInput,
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

    /// How several consumers read one resource. Replaying the producer is
    /// legal when its payload opens independent cursors and every consumer
    /// takes its own cursor; it is chosen when a consumer feeds an aggregate,
    /// whose chunked input is the reason to keep the producer deferred.
    /// Otherwise the consumers share one materialized copy.
    fn resource_sharing(&self, resource: &ResidentResource, data_form: ResidentDataForm) -> ResourceSharing {
        if resource.consumers().len() <= 1 {
            return ResourceSharing::SingleConsume;
        }
        if self.payload_opens_independent_cursors(resource, data_form)
            && self.consumers_take_own_cursors(resource)
            && self.consumers_feed_aggregate(resource)
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
                    PipelineTaskKind::JoinBuild => {
                        consumer.join_plan().is_some_and(|plan| plan.form() == JoinForm::KeyedEqui)
                    },
                    PipelineTaskKind::JoinProbe => self.join_probe_accepts_frame_stream(consumer),
                    _ => consumer.accepted_input_form() >= AcceptedInputForm::FrameOrStream,
                })
            })
    }

    fn join_probe_accepts_frame_stream(&self, probe: &PipelineTask) -> bool {
        let (Some(join), Some(plan)) = (self.join_node(probe), probe.join_plan()) else {
            return false;
        };
        join_probe_streams_chunks(join, plan.form(), plan.planned())
    }

    /// Whether every consumer re-streams the resource from the producer with
    /// its own cursor: join inputs do, as do aggregates and unary maps whose
    /// output reaches a join or an aggregate.
    fn consumers_take_own_cursors(&self, resource: &ResidentResource) -> bool {
        resource.consumers().iter().all(|consumer| match self.consumer_edge_kind(resource, *consumer) {
            Some(PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput) => true,
            Some(PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer) => {
                self.task(*consumer).is_some_and(|consumer| match consumer.kind() {
                    PipelineTaskKind::Aggregate => true,
                    PipelineTaskKind::Unary => {
                        self.task_feeds_join_input(consumer.id()) || self.task_feeds_aggregate_input(consumer.id())
                    },
                    _ => false,
                })
            },
            Some(PipelineEdgeKind::Control | PipelineEdgeKind::BarrierOutput | PipelineEdgeKind::SinkInput) | None => {
                false
            },
        })
    }

    /// Whether some consumer, directly or through unary maps, is an aggregate.
    fn consumers_feed_aggregate(&self, resource: &ResidentResource) -> bool {
        resource.consumers().iter().any(|consumer| {
            matches!(
                self.consumer_edge_kind(resource, *consumer),
                Some(PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer)
            ) && self.task(*consumer).is_some_and(|consumer| match consumer.kind() {
                PipelineTaskKind::Aggregate => true,
                PipelineTaskKind::Unary => self.task_feeds_aggregate_input(consumer.id()),
                _ => false,
            })
        })
    }

    fn consumer_edge_kind(&self, resource: &ResidentResource, consumer: PipelineTaskId) -> Option<PipelineEdgeKind> {
        self.edges
            .iter()
            .find(|edge| {
                edge.from() == resource.producer() && edge.to() == consumer && edge.resources().contains(&resource.id())
            })
            .map(|edge| edge.kind())
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
