//! Stable DataFusion metric contracts and native-execution metric translation.
//!
//! Implementation lives in the sibling modules `capture`, `contract`, `record`,
//! and `registry`. This facade re-exports the metric types and metric names the
//! rest of the adapter uses.

mod capture;
mod contract;
mod record;
mod registry;

pub use capture::{NativeExecutionMetricsCapture, NativeMetricsCapture};
pub use contract::NATIVE_ALGEON_DATAFUSION_METRIC_NAMES;
pub(crate) use record::{
    AlgeonNativeMetrics, DeferredHostMetrics, GraphClaimCalibrationCapture, SealedAttemptMetricsTarget,
};
pub use record::{
    GraphClaimCalibrationExecution, GraphClaimCalibrationNonPhysicalReason, SealedAttemptGraphClaimCalibrationSnapshot,
};

/// DataFusion counter name for server runtime-artifact capture loss.
///
/// Each execution wrapper records one when its bounded runtime-artifact capture loses rows.
pub const SERVER_CAPTURE_LOSS_METRIC_NAME: &str = "algeon_server_capture_loss";

#[cfg(test)]
mod tests;
