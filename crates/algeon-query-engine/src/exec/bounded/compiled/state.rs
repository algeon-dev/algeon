use std::collections::{BTreeMap, BTreeSet};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{
        CompiledPipeline, PipelineTask, PipelineTaskId, ResidentLifetime, ResidentResource, ResidentResourceId,
    },
};

const OPERATION: &str = "query_engine.exec.bounded.compiled.state";

/// Mutable execution state for one compiled pipeline run.
#[doc(hidden)]
#[derive(Debug)]
pub struct CompiledPipelineState<T> {
    completed_tasks: BTreeSet<PipelineTaskId>,
    execution_order: Vec<PipelineTaskId>,
    resources: BTreeMap<ResidentResourceId, ResidentResourceSlot<T>>,
}

#[derive(Debug)]
struct ResidentResourceSlot<T> {
    definition: ResidentResource,
    payload: Option<T>,
    released: bool,
    consumed_by: Option<PipelineTaskId>,
}

/// A resident resource payload released after its last compiled consumer.
#[doc(hidden)]
#[derive(Debug)]
pub struct ReleasedCompiledResource<T> {
    id: ResidentResourceId,
    payload: T,
}

impl<T> ReleasedCompiledResource<T> {
    pub fn id(&self) -> ResidentResourceId {
        self.id
    }

    pub fn into_payload(self) -> T {
        self.payload
    }
}

/// A single-consumer resident resource moved into a compiled task executor.
#[doc(hidden)]
#[derive(Debug)]
pub struct ConsumedCompiledResource<T> {
    id: ResidentResourceId,
    payload: T,
}

impl<T> ConsumedCompiledResource<T> {
    pub fn id(&self) -> ResidentResourceId {
        self.id
    }

    pub fn into_payload(self) -> T {
        self.payload
    }
}

impl<T> CompiledPipelineState<T> {
    pub fn new(pipeline: &CompiledPipeline) -> Result<Self> {
        let mut resources = BTreeMap::new();
        for resource in pipeline.resources() {
            let previous = resources.insert(
                resource.id(),
                ResidentResourceSlot {
                    definition: resource.clone(),
                    payload: None,
                    released: false,
                    consumed_by: None,
                },
            );
            if previous.is_some() {
                return Err(invalid("compiled resource table contains a duplicate resource")
                    .with_fact("resident_resource_id", resource.id()));
            }
        }

        Ok(Self { completed_tasks: BTreeSet::new(), execution_order: Vec::new(), resources })
    }

    pub fn completed_task_count(&self) -> usize {
        self.completed_tasks.len()
    }

    pub fn execution_order(&self) -> &[PipelineTaskId] {
        &self.execution_order
    }

    pub fn is_task_completed(&self, task: PipelineTaskId) -> bool {
        self.completed_tasks.contains(&task)
    }

    pub fn is_resource_live(&self, resource: ResidentResourceId) -> bool {
        self.resources.get(&resource).is_some_and(|slot| slot.payload.is_some() && !slot.released)
    }

    pub fn is_resource_released(&self, resource: ResidentResourceId) -> bool {
        self.resources.get(&resource).is_some_and(|slot| slot.released)
    }

    pub fn live_resource_count(&self) -> usize {
        self.resources.values().filter(|slot| slot.payload.is_some() && !slot.released).count()
    }

    pub fn drain_live_resources(&mut self) -> Vec<ReleasedCompiledResource<T>> {
        let mut released = Vec::new();
        for (id, slot) in &mut self.resources {
            if slot.released {
                continue;
            }
            if let Some(payload) = slot.payload.take() {
                slot.released = true;
                released.push(ReleasedCompiledResource { id: *id, payload });
            }
        }
        released
    }

    pub fn resource_payload(&self, resource: ResidentResourceId) -> Result<&T> {
        let slot = self.resource_slot(resource)?;
        if slot.released {
            return Err(invalid("compiled task requested a released resident resource")
                .with_fact("resident_resource_id", resource));
        }
        slot.payload.as_ref().ok_or_else(|| {
            invalid("compiled task requested an unproduced resident resource")
                .with_fact("resident_resource_id", resource)
        })
    }

    pub fn install_lane_claim_payload(&mut self, resource: ResidentResourceId, payload: T) -> Result<()> {
        let slot = self.resource_slot_mut(resource)?;
        if slot.released {
            return Err(invalid("compiled lane scratch state received a released resident resource")
                .with_fact("resident_resource_id", resource));
        }
        if slot.payload.is_some() {
            return Err(invalid("compiled lane scratch state received an already-produced resident resource")
                .with_fact("resident_resource_id", resource));
        }
        slot.payload = Some(payload);
        Ok(())
    }

    pub fn replace_resource_payload(&mut self, resource: ResidentResourceId, payload: T) -> Result<T> {
        let slot = self.resource_slot_mut(resource)?;
        if slot.released {
            return Err(invalid("compiled task tried to replace a released resident resource")
                .with_fact("resident_resource_id", resource));
        }
        slot.payload.replace(payload).ok_or_else(|| {
            invalid("compiled task tried to replace an unproduced resident resource")
                .with_fact("resident_resource_id", resource)
        })
    }

    pub fn prepare_task(&self, task: &PipelineTask) -> Result<()> {
        if self.completed_tasks.contains(&task.id()) {
            return Err(invalid("compiled task executed more than once")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }

        for resource in task.input_resources() {
            let slot = self.resource_slot(*resource)?;
            if !slot.definition.consumers().contains(&task.id()) {
                return Err(invalid("compiled task input is not a recorded resource consumer")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
            if slot.released {
                return Err(invalid("compiled task input resource was already released")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
            if slot.payload.is_none() {
                return Err(invalid("compiled task input resource is not produced")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
        }

        Ok(())
    }

    pub fn can_consume_task_input_resource(&self, task: &PipelineTask, resource: ResidentResourceId) -> bool {
        self.resource_slot(resource).is_ok_and(|slot| {
            task.input_resources().contains(&resource)
                && !slot.released
                && slot.payload.is_some()
                && matches!(slot.definition.lifetime(), ResidentLifetime::BetweenTasks { .. })
                && slot.definition.consumers() == [task.id()]
        })
    }

    pub fn consume_task_input_resource(
        &mut self,
        task: &PipelineTask,
        resource: ResidentResourceId,
    ) -> Result<ConsumedCompiledResource<T>> {
        if self.completed_tasks.contains(&task.id()) {
            return Err(invalid("compiled task consumed an input after completion")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", resource)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        if !task.input_resources().contains(&resource) {
            return Err(invalid("compiled task tried to consume an undeclared input resource")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", resource)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }

        let slot = self.resource_slot_mut(resource)?;
        if !slot.definition.consumers().contains(&task.id()) {
            return Err(invalid("compiled task input is not a recorded resource consumer")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", resource)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        if slot.definition.consumers() != [task.id()] {
            return Err(invalid("compiled task cannot consume a multi-consumer resident resource")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", resource)
                .with_fact("pipeline_task_kind", task.kind().as_str())
                .with_fact("consumer_count", slot.definition.consumers().len()));
        }
        if !matches!(slot.definition.lifetime(), ResidentLifetime::BetweenTasks { .. }) {
            return Err(invalid("compiled task can only consume a between-tasks resident resource")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", resource)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        if slot.released {
            return Err(invalid("compiled task input resource was already released")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", resource)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let payload = slot.payload.take().ok_or_else(|| {
            invalid("compiled task input resource is not produced")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", resource)
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        slot.released = true;
        slot.consumed_by = Some(task.id());

        Ok(ConsumedCompiledResource { id: resource, payload })
    }

    pub fn complete_task(
        &mut self,
        task: &PipelineTask,
        produced_resources: Vec<(ResidentResourceId, T)>,
    ) -> Result<Vec<ReleasedCompiledResource<T>>> {
        self.prepare_task_completion(task)?;
        let produced = self.install_task_outputs(task, produced_resources)?;

        if !self.completed_tasks.insert(task.id()) {
            return Err(invalid("compiled task completed more than once")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        self.execution_order.push(task.id());

        let mut released = Vec::new();
        for resource in produced {
            if self.resource_lifetime(resource)? == (ResidentLifetime::TaskLocal { task: task.id() }) {
                released.push(self.release_resource(resource, task.id())?);
            }
        }
        for resource in task.input_resources() {
            if self.should_release_after_consumer_completion(*resource, task.id())? {
                released.push(self.release_resource(*resource, task.id())?);
            }
        }

        Ok(released)
    }

    pub fn ensure_finished(&self, pipeline: &CompiledPipeline) -> Result<()> {
        for task in pipeline.tasks() {
            if !self.completed_tasks.contains(&task.id()) {
                return Err(invalid("compiled pipeline finished with an incomplete task")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
        }

        for slot in self.resources.values() {
            match slot.definition.lifetime() {
                ResidentLifetime::TaskLocal { .. } | ResidentLifetime::BetweenTasks { .. } => {
                    if !slot.released {
                        return Err(invalid("compiled pipeline finished with an unreleased resident resource")
                            .with_fact("resident_resource_id", slot.definition.id()));
                    }
                },
                ResidentLifetime::SinkOutput { .. } => {},
            }
        }

        Ok(())
    }

    pub fn take_sink_output_resource(&mut self, pipeline: &CompiledPipeline) -> Result<ReleasedCompiledResource<T>> {
        let sink_task = pipeline.sink().input();
        let matching = self
            .resources
            .iter()
            .filter_map(|(id, slot)| match slot.definition.lifetime() {
                ResidentLifetime::SinkOutput { producer } if producer == sink_task => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [resource] = matching.as_slice() else {
            return Err(invalid("compiled GPU sink expected exactly one sink-output resident resource")
                .with_fact("sink_boundary_task_id", sink_task)
                .with_fact("sink_boundary_kind", pipeline.sink().kind().as_str())
                .with_fact("sink_output_resource_count", matching.len()));
        };
        if !self.completed_tasks.contains(&sink_task) {
            return Err(invalid("compiled GPU sink output producer did not complete")
                .with_fact("sink_boundary_task_id", sink_task)
                .with_fact("resident_resource_id", *resource));
        }

        let slot = self.resource_slot_mut(*resource)?;
        if slot.released {
            return Err(invalid("compiled GPU sink output resource was already released")
                .with_fact("sink_boundary_task_id", sink_task)
                .with_fact("resident_resource_id", *resource));
        }
        let payload = slot.payload.take().ok_or_else(|| {
            invalid("compiled GPU sink output resource has no resident payload")
                .with_fact("sink_boundary_task_id", sink_task)
                .with_fact("resident_resource_id", *resource)
        })?;
        slot.released = true;
        Ok(ReleasedCompiledResource { id: *resource, payload })
    }

    fn install_task_outputs(
        &mut self,
        task: &PipelineTask,
        produced_resources: Vec<(ResidentResourceId, T)>,
    ) -> Result<Vec<ResidentResourceId>> {
        let expected = task.output_resources().iter().copied().collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut produced_ids = Vec::with_capacity(produced_resources.len());

        for (resource, payload) in produced_resources {
            if !seen.insert(resource) {
                return Err(invalid("compiled task produced the same resident resource twice")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
            if !expected.contains(&resource) {
                return Err(invalid("compiled task produced an undeclared resident resource")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }

            let slot = self.resource_slot_mut(resource)?;
            if slot.definition.producer() != task.id() {
                return Err(invalid("compiled task is not the recorded resident resource producer")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("recorded_producer_task", slot.definition.producer())
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
            if slot.payload.is_some() || slot.released {
                return Err(invalid("compiled resident resource has already been produced")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }

            slot.payload = Some(payload);
            produced_ids.push(resource);
        }

        if let Some(resource) = expected.difference(&seen).next() {
            return Err(invalid("compiled task did not produce a declared resident resource")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", *resource)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }

        Ok(produced_ids)
    }

    fn should_release_after_consumer_completion(
        &self,
        resource: ResidentResourceId,
        consumer: PipelineTaskId,
    ) -> Result<bool> {
        let slot = self.resource_slot(resource)?;
        if slot.released {
            return Ok(false);
        }
        match slot.definition.lifetime() {
            ResidentLifetime::BetweenTasks { .. } => {
                if !slot.definition.consumers().contains(&consumer) {
                    return Err(invalid("compiled task completed as an unrecorded resident resource consumer")
                        .with_fact("resident_resource_id", resource)
                        .with_fact("compiled_task_id", consumer));
                }
                Ok(slot.definition.consumers().iter().all(|consumer| self.completed_tasks.contains(consumer)))
            },
            ResidentLifetime::TaskLocal { .. } => Err(invalid("compiled task consumed a task-local resident resource")
                .with_fact("resident_resource_id", resource)
                .with_fact("compiled_task_id", consumer)),
            ResidentLifetime::SinkOutput { .. } => {
                Err(invalid("compiled task consumed a sink-output resident resource")
                    .with_fact("resident_resource_id", resource)
                    .with_fact("compiled_task_id", consumer))
            },
        }
    }

    fn prepare_task_completion(&self, task: &PipelineTask) -> Result<()> {
        if self.completed_tasks.contains(&task.id()) {
            return Err(invalid("compiled task completed more than once")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }

        for resource in task.input_resources() {
            let slot = self.resource_slot(*resource)?;
            if slot.consumed_by == Some(task.id()) {
                continue;
            }
            if !slot.definition.consumers().contains(&task.id()) {
                return Err(invalid("compiled task input is not a recorded resource consumer")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
            if slot.released {
                return Err(invalid("compiled task input resource was already released")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
            if slot.payload.is_none() {
                return Err(invalid("compiled task input resource is not produced")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
        }

        Ok(())
    }

    fn release_resource(
        &mut self,
        resource: ResidentResourceId,
        consumer: PipelineTaskId,
    ) -> Result<ReleasedCompiledResource<T>> {
        let incomplete_consumer = {
            let slot = self.resource_slot(resource)?;
            slot.definition.consumers().iter().copied().find(|consumer| !self.completed_tasks.contains(consumer))
        };
        if let Some(incomplete_consumer) = incomplete_consumer {
            return Err(invalid("compiled resident resource release requested before all consumers completed")
                .with_fact("resident_resource_id", resource)
                .with_fact("compiled_task_id", consumer)
                .with_fact("incomplete_consumer_task", incomplete_consumer));
        }

        let slot = self.resource_slot_mut(resource)?;
        if slot.released {
            return Err(invalid("compiled resident resource was released more than once")
                .with_fact("resident_resource_id", resource)
                .with_fact("compiled_task_id", consumer));
        }
        let payload = slot.payload.take().ok_or_else(|| {
            invalid("compiled resident resource release found no resident payload")
                .with_fact("resident_resource_id", resource)
                .with_fact("compiled_task_id", consumer)
        })?;
        slot.released = true;
        Ok(ReleasedCompiledResource { id: resource, payload })
    }

    fn resource_lifetime(&self, resource: ResidentResourceId) -> Result<ResidentLifetime> {
        Ok(self.resource_slot(resource)?.definition.lifetime())
    }

    fn resource_slot(&self, resource: ResidentResourceId) -> Result<&ResidentResourceSlot<T>> {
        self.resources.get(&resource).ok_or_else(|| {
            invalid("compiled resident resource table is missing a declared resource")
                .with_fact("resident_resource_id", resource)
        })
    }

    fn resource_slot_mut(&mut self, resource: ResidentResourceId) -> Result<&mut ResidentResourceSlot<T>> {
        self.resources.get_mut(&resource).ok_or_else(|| {
            invalid("compiled resident resource table is missing a declared resource")
                .with_fact("resident_resource_id", resource)
        })
    }
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
