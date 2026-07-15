//! Query-keyed capture forwarding for server observability.

use std::{
    fmt,
    sync::{Arc, Weak},
};

use nexus_query_engine::{metrics::QueryMetrics, runtime::LakehouseCacheFillEventSnapshot};

use crate::native_exec::admission::QueryAdmissionFacts;

/// Query-keyed sink for native metrics captured while DataFusion wrapper metrics are populated.
pub(crate) trait NativeMetricsCaptureSink: Send + Sync {
    fn record_admission_facts(&self, query_id: &str, facts: QueryAdmissionFacts);

    fn record_query_metrics(&self, query_id: &str, metrics: &QueryMetrics);

    fn record_lakehouse_cache_fill_events(&self, query_id: &str, events: &[LakehouseCacheFillEventSnapshot]);
}

/// Cheap cloneable handle copied into native execution wrappers for one server query.
#[derive(Clone)]
pub(crate) struct NativeMetricsCapture {
    query_id: Arc<str>,
    sink: Weak<dyn NativeMetricsCaptureSink>,
}

impl NativeMetricsCapture {
    #[cfg(feature = "server")]
    pub(crate) fn new(query_id: impl Into<Arc<str>>, sink: Arc<dyn NativeMetricsCaptureSink>) -> Self {
        Self { query_id: query_id.into(), sink: Arc::downgrade(&sink) }
    }

    #[cfg(feature = "server")]
    pub(crate) fn query_id(&self) -> &str {
        &self.query_id
    }

    pub(super) fn record_query_metrics(&self, metrics: &QueryMetrics) {
        if let Some(sink) = self.sink.upgrade() {
            if metrics.query_id().as_str() == self.query_id.as_ref() {
                sink.record_query_metrics(&self.query_id, metrics);
            } else {
                sink.record_query_metrics(&self.query_id, &metrics.clone().with_query_id(self.query_id.as_ref()));
            }
        }
    }

    pub(super) fn record_lakehouse_cache_fill_events(&self, events: &[LakehouseCacheFillEventSnapshot]) {
        if let Some(sink) = self.sink.upgrade() {
            sink.record_lakehouse_cache_fill_events(&self.query_id, events);
        }
    }

    pub(super) fn record_admission_facts(&self, facts: QueryAdmissionFacts) {
        if let Some(sink) = self.sink.upgrade() {
            sink.record_admission_facts(&self.query_id, facts);
        }
    }
}

impl fmt::Debug for NativeMetricsCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeMetricsCapture").field("query_id", &self.query_id).finish_non_exhaustive()
    }
}
