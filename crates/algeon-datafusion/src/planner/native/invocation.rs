//! Per-invocation native planning records.
//!
//! The native optimizer rules are shared across every query planned on a
//! session, so planning records kept on the rule itself leak across queries:
//! whichever invocation planned last wins, and a later reader sees another
//! query's field names, Arrow types, and candidate shapes. Each invocation
//! therefore carries its own record store in `ConfigOptions.extensions`,
//! which DataFusion passes to every rule in the same planning pipeline.

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use datafusion::common::{
    DataFusionError, Result as DataFusionResult,
    config::{ConfigEntry, ConfigExtension, ConfigOptions, ExtensionOptions},
};

use super::NativePlanningDiagnostics;

/// Planning records owned by one adapter invocation.
///
/// Inserted once into invocation-local `ConfigOptions.extensions`; optimizer
/// rules in the same pipeline read it through `&ConfigOptions`. One entry per
/// optimizer stage run, in run order; the latest entry describes the current
/// planning result.
#[derive(Debug, Clone)]
pub(crate) struct Invocation(Arc<InvocationState>);

#[derive(Debug)]
struct InvocationState {
    planning: Mutex<Vec<NativePlanningDiagnostics>>,
}

impl Invocation {
    /// Creates an empty per-invocation record store.
    pub(crate) fn new() -> Self {
        Self(Arc::new(InvocationState { planning: Mutex::new(Vec::new()) }))
    }

    /// Returns the invocation carried by these options, if one was installed.
    pub(crate) fn of_options(options: &ConfigOptions) -> Option<&Self> {
        options.extensions.get::<Self>()
    }

    /// Appends one optimizer stage's planning record to this invocation.
    pub(crate) fn record_planning(&self, diagnostics: NativePlanningDiagnostics) {
        self.0.planning.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(diagnostics);
    }

    /// Returns the latest planning record of this invocation, if it planned.
    pub(crate) fn planning_diagnostics(&self) -> Option<NativePlanningDiagnostics> {
        self.0.planning.lock().unwrap_or_else(std::sync::PoisonError::into_inner).last().cloned()
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
