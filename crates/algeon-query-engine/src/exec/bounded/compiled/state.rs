//! Ledger of resident resource payloads for one compiled pipeline run.
//!
//! The ledger enforces the resident lifetimes the compiled pipeline declares
//! (`TaskLocal`, `BetweenTasks`, `SinkOutput`): a payload is produced once by
//! its recorded producer, read only while live, and leaves the ledger once.
//! It is generic over the payload so the scheduler can be tested with
//! plain values; the GPU run stores `CompiledGpuResource`.
//!
//! Every error here is `InvalidPlan`. The pipeline validator rules out each
//! condition before execution, so reaching one means the executor or the
//! scheduler broke the compiled contract, and the facts name the task and
//! resource involved.

use std::collections::{BTreeMap, BTreeSet};

use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{
        CompiledPipeline, PipelineTask, PipelineTaskId, ResidentLifetime, ResidentResource, ResidentResourceId,
    },
};

const OPERATION: &str = "query_engine.exec.bounded.compiled.state";

/// Mutable execution state for one compiled pipeline run.
///
/// Owned by the scheduler for the run and lent to the executor by mutable
/// reference during each task, so no lock is involved: lane workers never see
/// it and commit their outputs through the coordinator.
#[derive(Debug)]
pub struct CompiledPipelineState<T> {
    completed_tasks: BTreeSet<PipelineTaskId>,
    /// Completion order, appended in `complete_task`. Diagnostics only; the
    /// scheduler reads `completed_tasks`.
    execution_order: Vec<PipelineTaskId>,
    resources: BTreeMap<ResidentResourceId, ResidentResourceSlot<T>>,
}

/// One declared resource and where its payload is. The definition is the
/// pipeline's and never changes; only `state` moves.
#[derive(Debug)]
struct ResidentResourceSlot<T> {
    definition: ResidentResource,
    state: SlotState<T>,
}

/// Where one resident resource payload is in its produce, read, and release
/// protocol.
#[derive(Debug)]
enum SlotState<T> {
    /// The producer has not committed the payload.
    Unproduced,
    /// The payload is stored and its consumers read it from the state.
    Live(T),
    /// The resource's sole consumer took the payload out of the state. The
    /// consumer owns releasing it; `complete_task` for that consumer treats
    /// the input as satisfied and does not release it a second time.
    Consumed { by: PipelineTaskId },
    /// The payload left the state when its lifetime ended.
    Released,
}

impl<T> SlotState<T> {
    fn is_live(&self) -> bool {
        matches!(self, Self::Live(_))
    }

    fn is_released(&self) -> bool {
        matches!(self, Self::Consumed { .. } | Self::Released)
    }

    /// Moves a live payload out and leaves `next` behind; any other state is
    /// left unchanged.
    fn take_live(&mut self, next: Self) -> Option<T> {
        if !self.is_live() {
            return None;
        }
        match std::mem::replace(self, next) {
            Self::Live(payload) => Some(payload),
            Self::Unproduced | Self::Consumed { .. } | Self::Released => None,
        }
    }
}

/// A resident resource payload released after its last compiled consumer.
///
/// Once one of these exists the ledger has no record of the payload any more;
/// the holder must pass it to the executor's release hook or drop it.
#[derive(Debug)]
pub(in crate::exec::bounded) struct ReleasedCompiledResource<T> {
    id: ResidentResourceId,
    payload: T,
}

impl<T> ReleasedCompiledResource<T> {
    pub(in crate::exec::bounded) fn id(&self) -> ResidentResourceId {
        self.id
    }

    pub(in crate::exec::bounded) fn into_payload(self) -> T {
        self.payload
    }
}

impl<T> CompiledPipelineState<T> {
    pub(crate) fn new(pipeline: &CompiledPipeline) -> Result<Self> {
        let mut resources = BTreeMap::new();
        for resource in pipeline.resources() {
            let previous = resources.insert(
                resource.id(),
                ResidentResourceSlot { definition: resource.clone(), state: SlotState::Unproduced },
            );
            if previous.is_some() {
                return Err(invalid("compiled resource table contains a duplicate resource")
                    .with_fact("resident_resource_id", resource.id()));
            }
        }

        Ok(Self { completed_tasks: BTreeSet::new(), execution_order: Vec::new(), resources })
    }

    pub(crate) fn completed_task_count(&self) -> usize {
        self.completed_tasks.len()
    }

    #[cfg(test)]
    pub fn execution_order(&self) -> &[PipelineTaskId] {
        &self.execution_order
    }

    pub(crate) fn is_task_completed(&self, task: PipelineTaskId) -> bool {
        self.completed_tasks.contains(&task)
    }

    pub(crate) fn is_resource_live(&self, resource: ResidentResourceId) -> bool {
        self.resources.get(&resource).is_some_and(|slot| slot.state.is_live())
    }

    #[cfg(test)]
    pub fn is_resource_released(&self, resource: ResidentResourceId) -> bool {
        self.resources.get(&resource).is_some_and(|slot| slot.state.is_released())
    }

    pub fn live_resource_count(&self) -> usize {
        self.resources.values().filter(|slot| slot.state.is_live()).count()
    }

    /// Takes every live payload out for error cleanup, leaving each slot
    /// `Released`. Lifetimes are ignored here: after a failed run nothing will
    /// read the sink output either.
    pub(in crate::exec::bounded) fn drain_live_resources(&mut self) -> Vec<ReleasedCompiledResource<T>> {
        let mut released = Vec::new();
        for (id, slot) in &mut self.resources {
            if let Some(payload) = slot.state.take_live(SlotState::Released) {
                released.push(ReleasedCompiledResource { id: *id, payload });
            }
        }
        released
    }

    pub(crate) fn resource_payload(&self, resource: ResidentResourceId) -> Result<&T> {
        match &self.resource_slot(resource)?.state {
            SlotState::Live(payload) => Ok(payload),
            SlotState::Consumed { .. } | SlotState::Released => {
                Err(invalid("compiled task requested a released resident resource")
                    .with_fact("resident_resource_id", resource))
            },
            SlotState::Unproduced => Err(invalid("compiled task requested an unproduced resident resource")
                .with_fact("resident_resource_id", resource)),
        }
    }

    /// Swaps the payload of a live resource and returns the previous one. The
    /// slot stays live and keeps its lifetime, so the caller owns releasing
    /// the returned payload. Used to install a filtered copy of a shared
    /// source frame for the consumers that have not read it yet.
    pub(crate) fn replace_resource_payload(&mut self, resource: ResidentResourceId, payload: T) -> Result<T> {
        match &mut self.resource_slot_mut(resource)?.state {
            SlotState::Live(stored) => Ok(std::mem::replace(stored, payload)),
            SlotState::Consumed { .. } | SlotState::Released => {
                Err(invalid("compiled task tried to replace a released resident resource")
                    .with_fact("resident_resource_id", resource))
            },
            SlotState::Unproduced => Err(invalid("compiled task tried to replace an unproduced resident resource")
                .with_fact("resident_resource_id", resource)),
        }
    }

    /// Checks that `task` can run: not yet completed, a recorded consumer of
    /// each declared input, and every input live. Read-only, so a rejected
    /// claim leaves the ledger as it was.
    pub(crate) fn prepare_task(&self, task: &PipelineTask) -> Result<()> {
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
            ensure_live_input(task, *resource, &slot.state)?;
        }

        Ok(())
    }

    /// Whether `consume_task_input_resource` would succeed: a live
    /// between-tasks resource whose only recorded consumer is `task`.
    pub(crate) fn can_consume_task_input_resource(&self, task: &PipelineTask, resource: ResidentResourceId) -> bool {
        self.resource_slot(resource).is_ok_and(|slot| {
            task.input_resources().contains(&resource)
                && slot.state.is_live()
                && matches!(slot.definition.lifetime(), ResidentLifetime::BetweenTasks { .. })
                && slot.definition.consumers() == [task.id()]
        })
    }

    /// Moves a sole-consumer input out of the ledger into `task`'s ownership.
    ///
    /// From here on the executor releases the payload, including when the
    /// task then fails; error cleanup will not see it. Only `BetweenTasks`
    /// resources with a single consumer qualify: a shared resource must
    /// stay stored for its other consumers, and a task-local resource is never
    /// an input.
    pub(crate) fn consume_task_input_resource(
        &mut self,
        task: &PipelineTask,
        resource: ResidentResourceId,
    ) -> Result<T> {
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
        ensure_live_input(task, resource, &slot.state)?;
        Ok(slot.state.take_live(SlotState::Consumed { by: task.id() }).expect("live input resource has a payload"))
    }

    /// Commits `task`: installs its outputs, marks it complete, and returns
    /// the payloads whose lifetime ended with it.
    ///
    /// Those are the task's own `TaskLocal` outputs and each `BetweenTasks`
    /// input whose every recorded consumer has now completed. Validation runs
    /// before any mutation, but output installation itself is not atomic. An
    /// undeclared or duplicate output fails the call after earlier outputs
    /// were installed live, and the task is not marked complete; error cleanup
    /// drains those live payloads.
    pub(in crate::exec::bounded) fn complete_task(
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

    /// Final check after the last task: every task completed and every
    /// `TaskLocal` or `BetweenTasks` payload left the ledger. Sink outputs are
    /// exempt because the caller takes them from the returned state.
    pub(crate) fn ensure_finished(&self, pipeline: &CompiledPipeline) -> Result<()> {
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
                    if !slot.state.is_released() {
                        return Err(invalid("compiled pipeline finished with an unreleased resident resource")
                            .with_fact("resident_resource_id", slot.definition.id()));
                    }
                },
                ResidentLifetime::SinkOutput { .. } => {},
            }
        }

        Ok(())
    }

    /// Takes the one `SinkOutput` payload produced by the sink's input task.
    /// Succeeds once per run; a second call finds the slot released.
    pub(in crate::exec::bounded) fn take_sink_output_resource(
        &mut self,
        pipeline: &CompiledPipeline,
    ) -> Result<ReleasedCompiledResource<T>> {
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
        if slot.state.is_released() {
            return Err(invalid("compiled GPU sink output resource was already released")
                .with_fact("sink_boundary_task_id", sink_task)
                .with_fact("resident_resource_id", *resource));
        }
        let payload = slot.state.take_live(SlotState::Released).ok_or_else(|| {
            invalid("compiled GPU sink output resource has no resident payload")
                .with_fact("sink_boundary_task_id", sink_task)
                .with_fact("resident_resource_id", *resource)
        })?;
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
            if !matches!(slot.state, SlotState::Unproduced) {
                return Err(invalid("compiled resident resource has already been produced")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }

            slot.state = SlotState::Live(payload);
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

    /// Whether completing `consumer` ends the lifetime of its input `resource`.
    /// An already released slot (taken by `consume_task_input_resource`)
    /// answers `false` so the payload is not released twice.
    fn should_release_after_consumer_completion(
        &self,
        resource: ResidentResourceId,
        consumer: PipelineTaskId,
    ) -> Result<bool> {
        let slot = self.resource_slot(resource)?;
        if slot.state.is_released() {
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

    /// `prepare_task` for the completion side: the same checks, except that an
    /// input this task consumed during execution is accepted in its
    /// `Consumed` state.
    fn prepare_task_completion(&self, task: &PipelineTask) -> Result<()> {
        if self.completed_tasks.contains(&task.id()) {
            return Err(invalid("compiled task completed more than once")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }

        for resource in task.input_resources() {
            let slot = self.resource_slot(*resource)?;
            if matches!(slot.state, SlotState::Consumed { by } if by == task.id()) {
                continue;
            }
            if !slot.definition.consumers().contains(&task.id()) {
                return Err(invalid("compiled task input is not a recorded resource consumer")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            }
            ensure_live_input(task, *resource, &slot.state)?;
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
        if slot.state.is_released() {
            return Err(invalid("compiled resident resource was released more than once")
                .with_fact("resident_resource_id", resource)
                .with_fact("compiled_task_id", consumer));
        }
        let payload = slot.state.take_live(SlotState::Released).ok_or_else(|| {
            invalid("compiled resident resource release found no resident payload")
                .with_fact("resident_resource_id", resource)
                .with_fact("compiled_task_id", consumer)
        })?;
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

fn ensure_live_input<T>(task: &PipelineTask, resource: ResidentResourceId, state: &SlotState<T>) -> Result<()> {
    let message = match state {
        SlotState::Live(_) => return Ok(()),
        SlotState::Consumed { .. } | SlotState::Released => "compiled task input resource was already released",
        SlotState::Unproduced => "compiled task input resource is not produced",
    };
    Err(invalid(message)
        .with_fact("compiled_task_id", task.id())
        .with_fact("resident_resource_id", resource)
        .with_fact("pipeline_task_kind", task.kind().as_str()))
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
