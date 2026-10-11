//! Identity of one runtime task attempt.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
/// Compiled task identity scoped to one native plan invocation.
pub struct RuntimeTaskId(String);

impl RuntimeTaskId {
    /// Creates a task identity from its scope-local label without validating the text.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the original scope-local task label.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for RuntimeTaskId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for RuntimeTaskId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for RuntimeTaskId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
/// Execution attempt number for one compiled task, independent of the query attempt identity.
pub struct RuntimeAttemptId(u32);

impl RuntimeAttemptId {
    /// Creates a task execution attempt number from a caller-provided value.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Compiled task identity paired with its execution attempt number.
pub struct TaskAttemptId {
    /// Compiled task whose execution attempt this value identifies.
    pub task_id: RuntimeTaskId,
    pub(crate) attempt: RuntimeAttemptId,
}

impl TaskAttemptId {
    /// Pairs a compiled task identity with its execution attempt number.
    pub const fn new(task_id: RuntimeTaskId, attempt: RuntimeAttemptId) -> Self {
        Self { task_id, attempt }
    }
}
