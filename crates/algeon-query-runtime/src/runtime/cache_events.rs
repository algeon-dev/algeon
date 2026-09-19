//! Structured completion events for asynchronous lakehouse cache fills.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

/// Structured outcome for an asynchronous lakehouse cache fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub enum LakehouseCacheFillOutcome {
    /// The complete object was written to the local lakehouse cache.
    Completed,
    /// The background fill failed after it was accepted for execution.
    Failed,
}

/// Error facts captured for a failed asynchronous lakehouse cache fill.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct LakehouseCacheFillErrorEvent {
    #[doc(hidden)]
    pub code: String,
    #[doc(hidden)]
    pub kind: String,
    #[doc(hidden)]
    pub status: String,
    #[doc(hidden)]
    pub operation: String,
}

/// Structured asynchronous lakehouse cache fill event.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct LakehouseCacheFillEvent {
    #[doc(hidden)]
    pub outcome: LakehouseCacheFillOutcome,
    #[doc(hidden)]
    pub bytes: Option<u64>,
    #[doc(hidden)]
    pub elapsed: Duration,
    #[doc(hidden)]
    pub cache_path_hash: String,
    #[doc(hidden)]
    pub stable_id_hash: String,
    #[doc(hidden)]
    pub file_index: Option<usize>,
    #[doc(hidden)]
    pub chunk_index: Option<usize>,
    #[doc(hidden)]
    pub error: Option<LakehouseCacheFillErrorEvent>,
}

/// Credential-free snapshot of a background lakehouse cache-fill completion event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LakehouseCacheFillEventSnapshot {
    /// Stable event outcome.
    pub outcome: &'static str,
    /// Complete-object bytes written for successful fills, when known.
    pub bytes: Option<u64>,
    /// Elapsed time from accepted background job start to completion/failure.
    pub elapsed: Duration,
    /// Stable hash of the local cache path.
    pub cache_path_hash: String,
    /// Stable hash of the source stable id.
    pub stable_id_hash: String,
    /// Source file index associated with the fill, when known.
    pub file_index: Option<usize>,
    /// Source chunk index associated with the fill, when known.
    pub chunk_index: Option<usize>,
    /// Stable native error code for failed fills, when present.
    pub error_code: Option<String>,
    /// Stable native error kind for failed fills, when present.
    pub error_kind: Option<String>,
    /// Stable native error status for failed fills, when present.
    pub error_status: Option<String>,
    /// Stable native error operation for failed fills, when present.
    pub error_operation: Option<String>,
}

impl LakehouseCacheFillOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

impl From<&LakehouseCacheFillEvent> for LakehouseCacheFillEventSnapshot {
    fn from(event: &LakehouseCacheFillEvent) -> Self {
        Self {
            outcome: event.outcome.as_str(),
            bytes: event.bytes,
            elapsed: event.elapsed,
            cache_path_hash: event.cache_path_hash.clone(),
            stable_id_hash: event.stable_id_hash.clone(),
            file_index: event.file_index,
            chunk_index: event.chunk_index,
            error_code: event.error.as_ref().map(|error| error.code.clone()),
            error_kind: event.error.as_ref().map(|error| error.kind.clone()),
            error_status: event.error.as_ref().map(|error| error.status.clone()),
            error_operation: event.error.as_ref().map(|error| error.operation.clone()),
        }
    }
}

/// Query-context-local event sink for asynchronous lakehouse cache fill completion.
#[derive(Debug, Clone, Default)]
#[doc(hidden)]
pub struct LakehouseCacheFillEventSink {
    events: Arc<Mutex<Vec<LakehouseCacheFillEvent>>>,
}

impl LakehouseCacheFillEventSink {
    #[doc(hidden)]
    pub fn record(&self, event: LakehouseCacheFillEvent) {
        match self.events.lock() {
            Ok(mut events) => events.push(event),
            Err(_error) => {
                eprintln!("[query-engine] warning: failed to record poisoned lakehouse cache fill event sink");
            },
        }
    }

    #[doc(hidden)]
    pub fn events(&self) -> Vec<LakehouseCacheFillEvent> {
        match self.events.lock() {
            Ok(events) => events.clone(),
            Err(_error) => Vec::new(),
        }
    }

    #[doc(hidden)]
    pub fn drain(&self) -> Vec<LakehouseCacheFillEvent> {
        match self.events.lock() {
            Ok(mut events) => events.drain(..).collect(),
            Err(_error) => Vec::new(),
        }
    }
}
