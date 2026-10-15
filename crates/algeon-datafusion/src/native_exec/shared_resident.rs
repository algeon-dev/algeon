//! GPU-resident results that several units of one composed island read.
//!
//! The planner moves a subplan that several native units repeat into one
//! producer, which every reading unit lists among its GPU-resident inputs.
//! The units run nested inside the island root's execution, and the root
//! installs one result scope in the task context its producers receive. The
//! first unit to run the producer computes the result and retains it here,
//! and each later unit takes a clone. The scope drops its copy when the last
//! unit takes the result; the device buffers then live as long as a reading
//! unit holds its input map.
//!
//! Repeatable GPU functions also register their reading occurrences here.
//! Their logical invocation name and native output scope distinguish separate
//! calls and pushed result options. Counts exclude the children of reused
//! parents because those children execute only for the first reader.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, PoisonError},
};

use datafusion::{
    common::Result as DataFusionResult,
    execution::context::TaskContext,
    physical_plan::{ExecutionPlan, coop::CooperativeExec},
};
use query_planning::plan::PlanFingerprint;
use query_runtime::runtime::resident::GpuResultHandle;

use super::{AlgeonNativeExec, producer::gpu_resident_producer};
use crate::error::Result;

/// Shared resident results of one composed island execution, by producer
/// input name. The planner names each shared producer uniquely across the
/// island's units.
#[derive(Default)]
pub(crate) struct SharedResidentResults {
    results: Mutex<BTreeMap<ResidentProducerIdentity, RetainedResult>>,
    reusable_consumers: BTreeMap<ReusableProducerIdentity, usize>,
    shared_consumers: BTreeMap<String, usize>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReusableProducerIdentity {
    invocation: String,
    scope: Vec<PlanFingerprint>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ResidentProducerIdentity {
    Shared(String),
    Reusable(ReusableProducerIdentity),
}

struct RetainedResult {
    handle: GpuResultHandle,
    remaining_consumers: usize,
}

impl SharedResidentResults {
    fn for_gpu_inputs(inputs: &[Arc<dyn ExecutionPlan>]) -> Result<Self> {
        let mut reusable_consumers = BTreeMap::new();
        let mut shared_consumers = BTreeMap::new();
        for input in inputs {
            count_consumers(input.as_ref(), &mut reusable_consumers, &mut shared_consumers)?;
        }
        reusable_consumers.retain(|_, consumers| *consumers > 1);
        Ok(Self { reusable_consumers, shared_consumers, ..Self::default() })
    }

    /// Takes the result a previous unit computed for `input_name`, releasing
    /// the retained copy when this is its last consumer.
    pub(crate) fn take(&self, input_name: &str) -> Option<GpuResultHandle> {
        self.take_identity(&ResidentProducerIdentity::Shared(input_name.to_string()))
    }

    fn take_identity(&self, identity: &ResidentProducerIdentity) -> Option<GpuResultHandle> {
        let mut results = self.results.lock().unwrap_or_else(PoisonError::into_inner);
        let retained = results.get_mut(identity)?;
        retained.remaining_consumers -= 1;
        let handle = retained.handle.clone();
        if retained.remaining_consumers == 0 {
            results.remove(identity);
        }
        Some(handle)
    }

    /// Retains `handle` for the remaining units that execute a read. A reused
    /// function skips its inputs, reducing their planned consumer counts.
    pub(crate) fn retain(&self, input_name: &str, handle: &GpuResultHandle) {
        let remaining_consumers = self.shared_consumers.get(input_name).copied().unwrap_or(1) - 1;
        self.retain_identity(ResidentProducerIdentity::Shared(input_name.to_string()), handle, remaining_consumers);
    }

    fn retain_identity(
        &self,
        identity: ResidentProducerIdentity,
        handle: &GpuResultHandle,
        remaining_consumers: usize,
    ) {
        if remaining_consumers == 0 {
            return;
        }
        self.results
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(identity, RetainedResult { handle: handle.clone(), remaining_consumers });
    }

    pub(crate) fn reusable_identity(&self, plan: &dyn ExecutionPlan) -> Result<Option<ReusableProducerIdentity>> {
        let Some(identity) = reusable_producer_identity(plan)? else {
            return Ok(None);
        };
        Ok(self.reusable_consumers.contains_key(&identity).then_some(identity))
    }

    pub(crate) fn take_reusable(&self, identity: &ReusableProducerIdentity) -> Option<GpuResultHandle> {
        self.take_identity(&ResidentProducerIdentity::Reusable(identity.clone()))
    }

    pub(crate) fn retain_reusable(&self, identity: ReusableProducerIdentity, handle: &GpuResultHandle) {
        let consumers = self.reusable_consumers[&identity];
        self.retain_identity(ResidentProducerIdentity::Reusable(identity), handle, consumers - 1);
    }
}

fn count_consumers(
    plan: &dyn ExecutionPlan,
    consumers: &mut BTreeMap<ReusableProducerIdentity, usize>,
    shared_consumers: &mut BTreeMap<String, usize>,
) -> Result<()> {
    if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        return count_consumers(cooperative.input().as_ref(), consumers, shared_consumers);
    }
    let Some(producer) = gpu_resident_producer(plan) else {
        return Ok(());
    };
    if let Some(identity) = reusable_producer_identity(plan)? {
        let count = consumers.entry(identity).or_default();
        *count += 1;
        // A reused parent does not execute its input producers again.
        if *count > 1 {
            return Ok(());
        }
    }
    if plan.downcast_ref::<AlgeonNativeExec>().is_some_and(|unit| unit.shared_consumer_count > 1) {
        let count = shared_consumers.entry(producer.gpu_result_input_name().to_string()).or_default();
        *count += 1;
        if *count > 1 {
            return Ok(());
        }
    }
    for child in plan.children() {
        count_consumers(child.as_ref(), consumers, shared_consumers)?;
    }
    Ok(())
}

fn reusable_producer_identity(plan: &dyn ExecutionPlan) -> Result<Option<ReusableProducerIdentity>> {
    if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        return reusable_producer_identity(cooperative.input().as_ref());
    }
    let Some(producer) = gpu_resident_producer(plan) else {
        return Ok(None);
    };
    let Some(mut scope) = producer.reusable_gpu_result_scope()? else {
        return Ok(None);
    };
    for child in plan.children() {
        if !append_input_execution_scopes(child.as_ref(), &mut scope)? {
            return Ok(None);
        }
    }
    Ok(Some(ReusableProducerIdentity { invocation: producer.gpu_result_input_name().to_string(), scope }))
}

fn append_input_execution_scopes(plan: &dyn ExecutionPlan, scopes: &mut Vec<PlanFingerprint>) -> Result<bool> {
    if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        return append_input_execution_scopes(cooperative.input().as_ref(), scopes);
    }
    let Some(producer) = gpu_resident_producer(plan) else {
        return Ok(false);
    };
    scopes.extend(
        producer
            .gpu_result_capability_evidence()?
            .execution_scopes()
            .iter()
            .map(|scope| scope.execution_scope_fingerprint()),
    );
    for child in plan.children() {
        if !append_input_execution_scopes(child.as_ref(), scopes)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `task_context` carrying a fresh result scope for one composed island.
///
/// A nested island replaces the scope of the island that stages its host
/// input, and its units read only results of their own island.
pub(crate) fn task_context_with_shared_resident_results(
    task_context: Arc<TaskContext>,
    gpu_inputs: &[Arc<dyn ExecutionPlan>],
) -> DataFusionResult<Arc<TaskContext>> {
    let mut session_config = task_context.session_config().clone();
    session_config.set_extension(Arc::new(SharedResidentResults::for_gpu_inputs(gpu_inputs)?));
    Ok(Arc::new(TaskContext::new(
        task_context.task_id(),
        task_context.session_id(),
        session_config,
        task_context.scalar_functions().clone(),
        task_context.higher_order_functions().clone(),
        task_context.aggregate_functions().clone(),
        task_context.window_functions().clone(),
        task_context.runtime_env(),
    )))
}

/// The result scope the enclosing composed island installed, if any.
pub(crate) fn shared_resident_results(task_context: &TaskContext) -> Option<Arc<SharedResidentResults>> {
    task_context.session_config().get_extension::<SharedResidentResults>()
}

#[cfg(all(test, feature = "cugraph"))]
mod tests;
