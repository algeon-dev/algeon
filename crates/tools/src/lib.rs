//! Support for operator tools and explicit acceptance fixtures.

#[cfg(feature = "embedded-backend-fixture")]
pub mod embedded_fixture;
#[cfg(feature = "otlp-metrics")]
pub mod otlp_metrics;
