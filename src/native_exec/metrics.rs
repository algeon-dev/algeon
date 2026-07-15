//! Stable DataFusion metric contracts and native-execution metric translation.
//!
//! Responsibility-specific implementation lives in the sibling modules. This
//! facade keeps the adapter's metric boundary explicit without retaining a
//! second implementation path.

mod capture;
mod contract;
mod record;
mod registry;

pub(crate) use capture::NativeMetricsCapture;
#[cfg(feature = "server")]
pub(crate) use capture::NativeMetricsCaptureSink;
pub use contract::NATIVE_CUDF_DATAFUSION_METRIC_NAMES;
pub(crate) use record::NativeCudfMetrics;

#[cfg(test)]
mod tests;
