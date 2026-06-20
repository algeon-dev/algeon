//! Minimal query QoS labels for GPU resource admission.

/// Query priority label carried by memory grant requests.
///
/// Phase 4 only defines the type. Priority scheduling behavior is deferred to
/// a later phase, so the governor treats all priorities identically today.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Best-effort work.
    Low,
    /// Default query priority.
    #[default]
    Normal,
    /// Latency-sensitive work.
    High,
}

impl Priority {
    /// Stable report-facing string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }
}

/// Query-level QoS metadata carried by resource requests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryQos {
    priority: Priority,
}

impl QueryQos {
    /// Creates QoS metadata for a priority.
    pub const fn new(priority: Priority) -> Self {
        Self { priority }
    }

    /// Returns the request priority label.
    pub const fn priority(self) -> Priority {
        self.priority
    }
}
