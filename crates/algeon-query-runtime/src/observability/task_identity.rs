//! Identity of one runtime task attempt.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeTaskId(String);

impl RuntimeTaskId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

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
pub struct RuntimeAttemptId(u32);

impl RuntimeAttemptId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskAttemptId {
    pub task_id: RuntimeTaskId,
    pub(crate) attempt: RuntimeAttemptId,
}

impl TaskAttemptId {
    pub const fn new(task_id: RuntimeTaskId, attempt: RuntimeAttemptId) -> Self {
        Self { task_id, attempt }
    }
}
