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

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, PoisonError},
};

use datafusion::execution::context::TaskContext;
use query_runtime::runtime::resident::GpuResultHandle;

/// Shared resident results of one composed island execution, by producer
/// input name. The planner names each shared producer uniquely across the
/// island's units.
#[derive(Default)]
pub(crate) struct SharedResidentResults {
    results: Mutex<BTreeMap<String, RetainedResult>>,
}

struct RetainedResult {
    handle: GpuResultHandle,
    remaining_consumers: usize,
}

impl SharedResidentResults {
    /// Takes the result a previous unit computed for `input_name`, releasing
    /// the retained copy when this is its last consumer.
    pub(crate) fn take(&self, input_name: &str) -> Option<GpuResultHandle> {
        let mut results = self.results.lock().unwrap_or_else(PoisonError::into_inner);
        let retained = results.get_mut(input_name)?;
        retained.remaining_consumers -= 1;
        let handle = retained.handle.clone();
        if retained.remaining_consumers == 0 {
            results.remove(input_name);
        }
        Some(handle)
    }

    /// Retains `handle` for the `remaining_consumers` units that read it
    /// after the unit that computed it.
    pub(crate) fn retain(&self, input_name: &str, handle: &GpuResultHandle, remaining_consumers: usize) {
        if remaining_consumers == 0 {
            return;
        }
        self.results
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(input_name.to_string(), RetainedResult { handle: handle.clone(), remaining_consumers });
    }
}

/// `task_context` carrying a fresh result scope for one composed island.
///
/// A nested island replaces the scope of the island that stages its host
/// input, and its units read only results of their own island.
pub(crate) fn task_context_with_shared_resident_results(task_context: Arc<TaskContext>) -> Arc<TaskContext> {
    let mut session_config = task_context.session_config().clone();
    session_config.set_extension(Arc::new(SharedResidentResults::default()));
    Arc::new(TaskContext::new(
        task_context.task_id(),
        task_context.session_id(),
        session_config,
        task_context.scalar_functions().clone(),
        task_context.higher_order_functions().clone(),
        task_context.aggregate_functions().clone(),
        task_context.window_functions().clone(),
        task_context.runtime_env(),
    ))
}

/// The result scope the enclosing composed island installed, if any.
pub(crate) fn shared_resident_results(task_context: &TaskContext) -> Option<Arc<SharedResidentResults>> {
    task_context.session_config().get_extension::<SharedResidentResults>()
}
