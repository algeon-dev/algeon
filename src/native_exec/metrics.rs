//! Stable DataFusion metric contracts and native-execution metric translation.
//!
//! Responsibility-specific implementation lives in the sibling modules. This
//! facade keeps the adapter's metric boundary explicit without retaining a
//! second implementation path.

mod capture;
mod contract;
mod record;
mod registry;

#[cfg(all(feature = "server", test))]
pub(crate) use capture::NativeMetricsCaptureSink;
#[cfg(feature = "server")]
pub(crate) use capture::NativeObservationActivityFactory;
pub(crate) use capture::{NativeExecutionMetricsCapture, NativeMetricsCapture, NativeRuntimeObservation};
pub use contract::NATIVE_NEXUS_DATAFUSION_METRIC_NAMES;
pub(crate) use record::NexusNativeMetrics;

pub(crate) const SERVER_CAPTURE_LOSS_METRIC_NAME: &str = "datafusion_nexus_server_capture_loss";

#[cfg(test)]
mod tests;
