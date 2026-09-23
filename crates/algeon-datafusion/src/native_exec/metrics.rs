//! Stable DataFusion metric contracts and native-execution metric translation.
//!
//! Responsibility-specific implementation lives in the sibling modules. This
//! facade keeps the adapter's metric boundary explicit without retaining a
//! second implementation path.

mod capture;
mod contract;
mod record;
mod registry;

pub use capture::{NativeExecutionMetricsCapture, NativeMetricsCapture};
pub use contract::NATIVE_ALGEON_DATAFUSION_METRIC_NAMES;
pub(crate) use record::{AlgeonNativeMetrics, GraphClaimCalibrationCapture, SealedAttemptMetricsTarget};
pub use record::{
    GraphClaimCalibrationExecution, GraphClaimCalibrationNonPhysicalReason, SealedAttemptGraphClaimCalibrationSnapshot,
};

pub const SERVER_CAPTURE_LOSS_METRIC_NAME: &str = "algeon_server_capture_loss";

#[cfg(test)]
mod tests;
