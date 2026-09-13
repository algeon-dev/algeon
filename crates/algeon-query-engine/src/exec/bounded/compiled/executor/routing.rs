use std::collections::BTreeSet;

use arrow::datatypes::DataType;

use super::CompiledGpuTaskExecutor;
use crate::{
    exec::bounded::SourceAdjacentAggregateChunkCapHints,
    expr::Expr,
    pipeline::{
        PipelineEdgeKind, PipelineTaskId, PipelineTaskKind, ResidentDataForm, ResidentResourceId, ResourceSharing,
    },
    plan::{AggregateNode, NodeId, PlanNode},
    source::SourceNode,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn task_feeds_join_input(&self, task: PipelineTaskId) -> bool {
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

    pub(super) fn task_feeds_direct_join_input(&self, task: PipelineTaskId) -> bool {
        self.pipeline.edges().iter().any(|edge| {
            edge.from() == task
                && matches!(edge.kind(), PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput)
        })
    }

    pub(super) fn task_feeds_aggregate_input(&self, task: PipelineTaskId) -> bool {
        self.task_feeds_aggregate_input_path(task).is_some()
    }

    pub(super) fn task_feeds_aggregate_input_through_unary_chain(&self, task: PipelineTaskId) -> bool {
        self.task_feeds_aggregate_input_path(task).unwrap_or(false)
    }

    pub(super) fn source_adjacent_aggregate_chunk_cap_hints(
        &self,
        task: PipelineTaskId,
    ) -> SourceAdjacentAggregateChunkCapHints {
        let has_source_adjacent_stages = self.task_feeds_aggregate_input_through_unary_chain(task)
            || self.task_feeds_any_join_input_through_unary_chain(task);
        let Some(aggregate_node_id) = self.reachable_aggregate_plan_node(task) else {
            return SourceAdjacentAggregateChunkCapHints { has_source_adjacent_stages, grouped_string_key: false };
        };
        let Some(aggregate_node) = self.plan.node(aggregate_node_id) else {
            return SourceAdjacentAggregateChunkCapHints { has_source_adjacent_stages, grouped_string_key: false };
        };
        let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
            return SourceAdjacentAggregateChunkCapHints { has_source_adjacent_stages, grouped_string_key: false };
        };
        let grouped_string_key = has_source_adjacent_stages
            && !aggregate.spec.group_exprs.is_empty()
            && aggregate.spec.grouping_sets.is_none()
            && self.aggregate_has_string_group_key(aggregate);
        SourceAdjacentAggregateChunkCapHints { has_source_adjacent_stages, grouped_string_key }
    }

    fn reachable_aggregate_plan_node(&self, task: PipelineTaskId) -> Option<NodeId> {
        let mut pending = vec![task];
        let mut visited = BTreeSet::new();

        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }
            for edge in self.pipeline.edges().iter().filter(|edge| edge.from() == current) {
                if !matches!(edge.kind(), PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer) {
                    continue;
                }
                let Some(consumer_task) = self.pipeline.tasks().iter().find(|candidate| candidate.id() == edge.to())
                else {
                    continue;
                };
                match consumer_task.kind() {
                    PipelineTaskKind::Aggregate => return consumer_task.plan_node(),
                    PipelineTaskKind::Unary => pending.push(edge.to()),
                    _ => {},
                }
            }
        }

        None
    }

    fn aggregate_has_string_group_key(&self, aggregate: &AggregateNode) -> bool {
        let Some(input_node) = self.plan.node(aggregate.input) else {
            return false;
        };
        let input_schema = &input_node.facts.output_schema;
        for expr in &aggregate.spec.group_exprs {
            let Some(column_name) = self.plain_group_key_column_name(expr) else {
                continue;
            };
            if input_schema.field_with_name(column_name).is_ok_and(|field| {
                matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
            }) {
                return true;
            }
        }
        false
    }

    fn plain_group_key_column_name<'a>(&self, expr: &'a Expr) -> Option<&'a str> {
        match expr {
            Expr::Column(column) => Some(column.name()),
            Expr::Alias { expr, .. } => self.plain_group_key_column_name(expr),
            _ => None,
        }
    }

    pub(super) fn task_feeds_aggregate_input_path(&self, task: PipelineTaskId) -> Option<bool> {
        let mut pending = vec![(task, false)];
        let mut visited = BTreeSet::new();

        while let Some((current, saw_unary)) = pending.pop() {
            if !visited.insert((current, saw_unary)) {
                continue;
            }

            for edge in self.pipeline.edges().iter().filter(|edge| edge.from() == current) {
                if !matches!(edge.kind(), PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer) {
                    continue;
                }
                let Some(consumer_task) = self.pipeline.tasks().iter().find(|candidate| candidate.id() == edge.to())
                else {
                    continue;
                };
                match consumer_task.kind() {
                    PipelineTaskKind::Aggregate => return Some(saw_unary),
                    PipelineTaskKind::Unary => pending.push((edge.to(), true)),
                    _ => {},
                }
            }
        }

        None
    }

    /// The compiled contract for how this resource's consumers read it.
    pub(super) fn resource_sharing(&self, resource: ResidentResourceId) -> ResourceSharing {
        self.pipeline
            .resources()
            .iter()
            .find(|candidate| candidate.id() == resource)
            .map_or(ResourceSharing::SingleConsume, |resource| resource.sharing())
    }

    /// The strongest payload form the compiled contract lets this resource
    /// carry. The producer may still hand over a weaker one.
    pub(super) fn resource_data_form(&self, resource: ResidentResourceId) -> ResidentDataForm {
        self.pipeline
            .resources()
            .iter()
            .find(|candidate| candidate.id() == resource)
            .map_or(ResidentDataForm::Frame, |resource| resource.data_form())
    }

    pub(super) fn local_parquet_source_node_for_resource(&self, resource: ResidentResourceId) -> Option<NodeId> {
        let producer = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource)?.producer();
        let source_node_id = self.pipeline.tasks().iter().find(|candidate| candidate.id() == producer)?.plan_node()?;
        self.plan.node(source_node_id).and_then(|node| match &node.kind {
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => Some(source_node_id),
            _ => None,
        })
    }

    // The materialize task preserves chunks only when the compiled contract
    // lets its output stay a stream and one consumer owns it.
    pub(super) fn join_probe_output_consumed_as_stream(&self, node_id: NodeId) -> bool {
        let Some(materialize) = self
            .pipeline
            .tasks()
            .iter()
            .find(|task| task.kind() == PipelineTaskKind::JoinMaterialize && task.plan_node() == Some(node_id))
        else {
            return false;
        };
        let [output] = materialize.output_resources() else {
            return false;
        };
        self.resource_data_form(*output) == ResidentDataForm::FrameStream
            && self.resource_sharing(*output) == ResourceSharing::SingleConsume
    }

    pub(super) fn task_feeds_any_join_input_through_unary_chain(&self, task: PipelineTaskId) -> bool {
        self.task_feeds_join_input_through_unary_chain(task, PipelineEdgeKind::JoinBuildInput).is_some()
            || self.task_feeds_join_input_through_unary_chain(task, PipelineEdgeKind::JoinProbeInput).is_some()
    }

    // A chain that ends in a resource shared by several join inputs still
    // feeds joins: every consumer re-streams the leaf with the chain's stages
    // (the scheduler never hands a multi-consumer resource to one owner), so
    // no frame carrying the whole staged leaf has to stay resident for them.
    pub(super) fn task_feeds_join_input_through_unary_chain(
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
            if resource.consumers().len() > 1 {
                if !saw_unary {
                    return None;
                }
                let mut matched = None;
                for consumer in resource.consumers() {
                    let edge =
                        self.pipeline.edges().iter().find(|edge| edge.from() == current && edge.to() == *consumer)?;
                    if !matches!(edge.kind(), PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput) {
                        return None;
                    }
                    if edge.kind() == target_kind {
                        matched.get_or_insert(*consumer);
                    }
                }
                return matched;
            }
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
