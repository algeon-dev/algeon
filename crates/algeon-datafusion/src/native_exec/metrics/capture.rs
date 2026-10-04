//! Runtime-observation projection limits carried across native execution wrappers.

use std::fmt;

use query_runtime::observability::{RuntimeObservationConfig, RuntimeObservationHandle, RuntimeObservationLimits};

/// Cloneable config-time observation wiring.
#[derive(Clone)]
pub struct NativeMetricsCapture {
    observation_config: RuntimeObservationConfig,
    metric_capture_limits: RuntimeObservationLimits,
}

impl NativeMetricsCapture {
    pub fn standalone() -> Self {
        let observation_config = RuntimeObservationConfig::native_execution();
        Self { observation_config, metric_capture_limits: observation_config.limits() }
    }

    #[cfg(test)]
    pub(crate) fn with_runtime_observation_config(mut self, config: RuntimeObservationConfig) -> Self {
        self.observation_config = config;
        self.metric_capture_limits = config.limits();
        self
    }

    pub const fn runtime_observation_config(&self) -> RuntimeObservationConfig {
        self.observation_config
    }

    pub const fn metric_capture_limits(&self) -> RuntimeObservationLimits {
        self.metric_capture_limits
    }

    pub fn new_execution(&self) -> NativeExecutionMetricsCapture {
        NativeExecutionMetricsCapture { wiring: self.clone() }
    }
}

/// Per-wrapper projection configuration. It owns neither terminal state nor
/// sealed runtime observations.
#[derive(Clone)]
pub struct NativeExecutionMetricsCapture {
    wiring: NativeMetricsCapture,
}

impl NativeExecutionMetricsCapture {
    /// Projection limits of the attempt observation that `observation` records into.
    pub(crate) fn coordinated(observation: RuntimeObservationHandle) -> Self {
        let observation_config = observation.config();
        Self { wiring: NativeMetricsCapture { observation_config, metric_capture_limits: observation_config.limits() } }
    }

    pub const fn metric_capture_limits(&self) -> RuntimeObservationLimits {
        self.wiring.metric_capture_limits()
    }
}

impl fmt::Debug for NativeMetricsCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeMetricsCapture").finish_non_exhaustive()
    }
}

impl fmt::Debug for NativeExecutionMetricsCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeExecutionMetricsCapture").finish_non_exhaustive()
    }
}
