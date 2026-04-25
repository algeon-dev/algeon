use super::*;

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

    pub(super) fn resource_has_multiple_consumers(&self, resource: ResidentResourceId) -> bool {
        self.pipeline
            .resources()
            .iter()
            .find(|candidate| candidate.id() == resource)
            .is_some_and(|resource| resource.consumers().len() > 1)
    }

    pub(super) fn shared_source_consumers_can_clone_source_stream(
        &self,
        task: PipelineTaskId,
        resource: ResidentResourceId,
    ) -> bool {
        let Some(resource) = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource) else {
            return false;
        };
        if resource.consumers().len() <= 1 {
            return true;
        }
        let mut feeds_aggregate = false;
        let all_consumers_can_clone = resource.consumers().iter().all(|consumer| {
            let Some(edge) = self.pipeline.edges().iter().find(|edge| {
                edge.from() == task && edge.to() == *consumer && edge.resources().contains(&resource.id())
            }) else {
                return false;
            };
            match edge.kind() {
                PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput => true,
                PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer => {
                    self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer).is_some_and(
                        |consumer_task| match consumer_task.kind() {
                            PipelineTaskKind::Aggregate => {
                                feeds_aggregate = true;
                                true
                            },
                            PipelineTaskKind::Unary => {
                                let unary_feeds_aggregate = self.task_feeds_aggregate_input(consumer_task.id());
                                feeds_aggregate |= unary_feeds_aggregate;
                                self.task_feeds_join_input(consumer_task.id()) || unary_feeds_aggregate
                            },
                            _ => false,
                        },
                    )
                },
                PipelineEdgeKind::Control | PipelineEdgeKind::BarrierOutput | PipelineEdgeKind::SinkInput => false,
            }
        });
        all_consumers_can_clone && feeds_aggregate
    }

    pub(super) fn local_parquet_source_node_for_resource(&self, resource: ResidentResourceId) -> Option<NodeId> {
        let producer = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource)?.producer();
        let source_node_id = self.pipeline.tasks().iter().find(|candidate| candidate.id() == producer)?.plan_node()?;
        self.plan.node(source_node_id).and_then(|node| match &node.kind {
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => Some(source_node_id),
            _ => None,
        })
    }

    pub(super) fn join_materialize_output_can_preserve_stream(
        &self,
        task: PipelineTaskId,
        resource: ResidentResourceId,
    ) -> bool {
        let Some(first_resource) = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource) else {
            return false;
        };
        if first_resource.producer() != task {
            return false;
        }
        if first_resource.consumers().len() > 1 {
            return first_resource.consumers().iter().all(|consumer| {
                self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer).is_some_and(
                    |consumer_task| {
                        consumer_task.kind() == PipelineTaskKind::JoinProbe
                            && self.join_probe_task_can_consume_frame_stream(consumer_task)
                    },
                )
            });
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
            if resource.consumers().len() > 1 {
                return resource.consumers().iter().all(|consumer| {
                    self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer).is_some_and(
                        |consumer_task| {
                            consumer_task.kind() == PipelineTaskKind::JoinProbe
                                && self.join_probe_task_can_consume_frame_stream(consumer_task)
                        },
                    )
                });
            }
            let [consumer] = resource.consumers() else {
                return false;
            };
            let Some(consumer_task) = self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer) else {
                return false;
            };
            match consumer_task.kind() {
                PipelineTaskKind::Aggregate => return true,
                PipelineTaskKind::JoinProbe => return self.join_probe_task_can_consume_frame_stream(consumer_task),
                PipelineTaskKind::FinalSort => return self.final_sort_task_can_consume_frame_stream(consumer_task),
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

    pub(super) fn final_sort_task_can_consume_frame_stream(&self, task: &PipelineTask) -> bool {
        let [output] = task.output_resources() else {
            return false;
        };
        self.downstream_fused_limit(task, *output).ok().flatten().is_some()
    }

    pub(super) fn join_probe_task_can_consume_frame_stream(&self, task: &PipelineTask) -> bool {
        let Some(node_id) = task.plan_node() else {
            return false;
        };
        let Some(node) = self.plan.node(node_id) else {
            return false;
        };
        let PlanNode::Join(join) = &node.kind else {
            return false;
        };
        if join.spec.residual_predicate.is_some() {
            return matches!(
                CompiledJoinExecutionStrategy::from_plan_join_node(self.plan, node_id, join),
                Ok(Some(CompiledJoinExecutionStrategy::OneShotEqui))
            );
        }
        match join.spec.join_type {
            JoinType::Inner | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => true,
            JoinType::Left => join.spec.build_side == JoinSide::Right,
            JoinType::Full | JoinType::Cross => false,
        }
    }

    pub(super) fn task_feeds_any_join_input_through_unary_chain(&self, task: PipelineTaskId) -> bool {
        self.task_feeds_join_input_through_unary_chain(task, PipelineEdgeKind::JoinBuildInput).is_some()
            || self.task_feeds_join_input_through_unary_chain(task, PipelineEdgeKind::JoinProbeInput).is_some()
    }

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
