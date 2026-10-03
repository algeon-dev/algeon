//! Per-invocation native planning state.
//!
//! The native optimizer rules are shared across every query planned on a
//! session, so state kept on a rule leaks across queries: whichever invocation
//! planned last wins, and a later reader sees another query's field names,
//! Arrow types, and candidate shapes. Each invocation therefore carries its
//! own state in `ConfigOptions.extensions`, which DataFusion passes to every
//! rule in the same planning pipeline and copies into the task context that
//! executes the plan.

use std::{
    any::Any,
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use datafusion::{
    common::{
        DataFusionError, Result as DataFusionResult,
        config::{ConfigEntry, ConfigExtension, ConfigOptions, ExtensionOptions},
    },
    execution::TaskContext,
    prelude::SessionConfig,
};
use query_planning::admission::AttemptCapabilityEvidence;
#[cfg(feature = "server")]
use query_runtime::runtime::IoLifecycle;

use super::NativePlanningDiagnostics;
use crate::{
    attempt::InvocationAttempt,
    native::{FooterCache, FooterScope},
    native_exec::{
        AlgeonNativeExecutionConfig,
        admission::{QueryAdmissionSource, QueryAdmissionTicket},
    },
};

/// State owned by one adapter invocation.
///
/// Inserted once into invocation-local `ConfigOptions.extensions`; optimizer
/// rules read it through `&ConfigOptions` and execution reads it through
/// `TaskContext::session_config()`. Planning records hold one entry per
/// optimizer stage run, in run order; the latest entry describes the current
/// planning result.
#[derive(Debug, Clone)]
pub(crate) struct Invocation(Arc<InvocationState>);

#[derive(Debug)]
struct InvocationState {
    attempt: Option<InvocationAttempt>,
    admission: OnceLock<Arc<QueryAdmissionTicket>>,
    #[cfg(feature = "server")]
    execution: Mutex<Option<InvocationExecution>>,
    planning: Mutex<Vec<NativePlanningDiagnostics>>,
    // Held from the first optimizer stage of a planning pass until the final
    // rewrite ends it, so the count-only probe and the final rewrite parse each
    // footer once without keeping footers alive while the plan executes.
    footers: Mutex<Option<Arc<FooterCache>>>,
}

/// Execution inputs that the Flight server attaches to an attempt's
/// invocation. Planning copies them into the execution config of every GPU
/// node it builds for that invocation, and every such node accounts its host
/// output against DataFusion's memory pool, which bounds Flight result streams.
#[cfg(feature = "server")]
#[derive(Debug, Clone)]
pub(crate) struct InvocationExecution {
    /// The server generation gate for file-backed native I/O.
    pub(crate) io_lifecycle: IoLifecycle,
    pub(crate) trace_id: Option<String>,
}

impl Invocation {
    /// Creates an invocation that belongs to no attempt: planning records its
    /// diagnostics here and binds no attempt admission.
    pub(crate) fn detached() -> Self {
        Self::new(None)
    }

    pub(crate) fn for_attempt(attempt: InvocationAttempt) -> Self {
        Self::new(Some(attempt))
    }

    fn new(attempt: Option<InvocationAttempt>) -> Self {
        Self(Arc::new(InvocationState {
            attempt,
            admission: OnceLock::new(),
            #[cfg(feature = "server")]
            execution: Mutex::new(None),
            planning: Mutex::new(Vec::new()),
            footers: Mutex::new(None),
        }))
    }

    /// Returns the invocation carried by these options, if one was installed.
    pub(crate) fn of_options(options: &ConfigOptions) -> Option<&Self> {
        options.extensions.get::<Self>()
    }

    /// Returns the invocation carried by the task's session config, if one was
    /// installed.
    pub(crate) fn of_task(task: &TaskContext) -> Option<&Self> {
        task.session_config().options().extensions.get::<Self>()
    }

    /// Makes this invocation the one that `config` plans and executes under,
    /// replacing any invocation the config already carried.
    pub(crate) fn install(&self, config: &mut SessionConfig) {
        config.options_mut().extensions.insert(self.clone());
    }

    /// The attempt this invocation belongs to, if any.
    pub(crate) fn attempt(&self) -> Option<&InvocationAttempt> {
        self.0.attempt.as_ref()
    }

    /// The attempt admission ticket of this invocation, if a fragment minted it.
    pub(crate) fn minted_admission_ticket(&self) -> Option<&Arc<QueryAdmissionTicket>> {
        self.0.admission.get()
    }

    /// Returns this attempt invocation's one admission ticket and whether this
    /// call minted it from `source` and `evidence`, or `None` for an invocation
    /// without an attempt. The first fragment's ticket stands: a caller that
    /// did not mint extends its evidence before admission instead of minting a
    /// replacement.
    pub(crate) fn attempt_admission_ticket(
        &self,
        source: QueryAdmissionSource,
        evidence: Option<AttemptCapabilityEvidence>,
    ) -> Option<(Arc<QueryAdmissionTicket>, bool)> {
        let attempt = self.attempt()?;
        let mut minted = false;
        let ticket = self.0.admission.get_or_init(|| {
            minted = true;
            QueryAdmissionTicket::new(source, evidence, attempt.ticket_binding())
        });
        Some((Arc::clone(ticket), minted))
    }

    /// Replaces the per-query execution inputs used by the next planning pass.
    #[cfg(feature = "server")]
    pub(crate) fn set_execution(&self, execution: InvocationExecution) {
        *self.0.execution.lock().unwrap_or_else(PoisonError::into_inner) = Some(execution);
    }

    /// Applies this invocation's per-query execution inputs to a config that
    /// planning is about to copy into a GPU node.
    pub(crate) fn execution_config(&self, config: AlgeonNativeExecutionConfig) -> AlgeonNativeExecutionConfig {
        #[cfg(feature = "server")]
        if let Some(execution) = self.0.execution.lock().unwrap_or_else(PoisonError::into_inner).clone() {
            let config = config.with_io_lifecycle(execution.io_lifecycle).with_host_output_memory_pool_accounting(true);
            return match execution.trace_id {
                Some(trace_id) => config.with_trace_id(trace_id),
                None => config,
            };
        }
        config
    }

    /// Enters a footer scope over this invocation's current planning pass.
    pub(crate) fn footer_scope(&self) -> FooterScope {
        let cache = Arc::clone(
            self.0
                .footers
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert_with(|| Arc::new(FooterCache::for_scope())),
        );
        FooterScope::join(cache)
    }

    /// Ends the current planning pass: its footers are released, and a later
    /// pass of the same invocation parses them again.
    pub(crate) fn end_planning_pass(&self) {
        self.0.footers.lock().unwrap_or_else(PoisonError::into_inner).take();
    }

    /// Appends one optimizer stage's planning record to this invocation.
    pub(crate) fn record_planning(&self, diagnostics: NativePlanningDiagnostics) {
        self.0.planning.lock().unwrap_or_else(PoisonError::into_inner).push(diagnostics);
    }

    /// Returns the latest planning record of this invocation, if it planned.
    pub(crate) fn planning_diagnostics(&self) -> Option<NativePlanningDiagnostics> {
        self.0.planning.lock().unwrap_or_else(PoisonError::into_inner).last().cloned()
    }
}

impl ConfigExtension for Invocation {
    const PREFIX: &'static str = "algeon_invocation";
}

impl ExtensionOptions for Invocation {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, _key: &str, _value: &str) -> DataFusionResult<()> {
        Err(DataFusionError::Configuration("invocation extension has no options".to_string()))
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        Vec::new()
    }
}
