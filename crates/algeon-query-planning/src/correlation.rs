//! Low-level diagnostic correlation identities.
//!
//! These value types are intentionally below error, metrics, runtime, and
//! observability.  Keeping them here prevents diagnostic error construction
//! from depending on the metrics aggregate merely to name a query or trace.

use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::plan::NodeId;

static NEXT_QUERY_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Query-wide diagnostic correlation id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct QueryId(String);

impl QueryId {
    /// Sentinel used by tests and manually-constructed metrics that have not
    /// been attached to a query entrypoint yet.
    pub const UNASSIGNED: &'static str = "unassigned";

    /// Mints a process-local query id.
    pub fn mint() -> Self {
        let sequence = NEXT_QUERY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos() as u64);
        Self(format!("q-{nanos:016x}-{sequence:08x}"))
    }

    /// Wraps a caller-provided stable query id.
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        if value.trim().is_empty() { Self::default() } else { Self(value) }
    }

    /// Returns the report-facing query id string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns true when this id is not the default sentinel.
    pub fn is_assigned(&self) -> bool {
        self.0 != Self::UNASSIGNED
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self(Self::UNASSIGNED.to_string())
    }
}

impl From<String> for QueryId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for QueryId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<&String> for QueryId {
    fn from(value: &String) -> Self {
        Self::new(value.clone())
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Decoding applies the same canonicalization as [`QueryId::new`], so a
/// blank serialized id reads back as [`QueryId::UNASSIGNED`].
impl<'de> Deserialize<'de> for QueryId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        String::deserialize(deserializer).map(Self::new)
    }
}

/// Optional parent trace id used to join multiple query ids under one request.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TraceId(String);

impl TraceId {
    /// Wraps a caller-provided trace id.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the report-facing trace id string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for TraceId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for TraceId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed diagnostic id for a plan-local node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanNodeId(String);

impl PlanNodeId {
    /// Wraps a caller-provided plan node id label.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the report-facing plan node id string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<NodeId> for PlanNodeId {
    fn from(value: NodeId) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for PlanNodeId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for PlanNodeId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for PlanNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed diagnostic id for a compiled native task.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompiledTaskId(String);

impl CompiledTaskId {
    /// Wraps a caller-provided compiled task id label.
    pub fn new(value: impl ToString) -> Self {
        Self(value.to_string())
    }

    /// Returns the report-facing compiled task id string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for CompiledTaskId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for CompiledTaskId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for CompiledTaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
