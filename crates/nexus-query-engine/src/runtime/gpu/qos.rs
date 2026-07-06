//! Minimal query QoS labels for GPU resource admission.

/// Query priority label carried by memory grant requests and service admission.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Best-effort work.
    Low,
    /// Default query priority.
    #[default]
    Normal,
    /// Latency-sensitive work.
    High,
    /// Operator-critical work that should pass lower-priority queued work when
    /// a service admission slot and memory grant are available.
    Critical,
}

impl Priority {
    /// Admission order from highest to lowest priority.
    pub const DESCENDING: [Self; 4] = [Self::Critical, Self::High, Self::Normal, Self::Low];

    /// Stable report-facing string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// Query-level QoS metadata carried by resource requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryQos {
    priority: Priority,
    budget_weight: u32,
}

impl QueryQos {
    /// Default budget weight for admission metadata.
    pub const DEFAULT_BUDGET_WEIGHT: u32 = 1;

    /// Creates QoS metadata for a priority.
    pub const fn new(priority: Priority) -> Self {
        Self { priority, budget_weight: Self::DEFAULT_BUDGET_WEIGHT }
    }

    /// Creates QoS metadata with an explicit non-zero budget weight.
    ///
    /// `priority` controls queued admission order. `budget_weight` controls
    /// proportional adaptive grant growth when released memory is shared across
    /// partial active grants.
    pub const fn with_budget_weight(mut self, budget_weight: u32) -> Self {
        self.budget_weight = if budget_weight == 0 { Self::DEFAULT_BUDGET_WEIGHT } else { budget_weight };
        self
    }

    /// Returns the request priority label.
    pub const fn priority(self) -> Priority {
        self.priority
    }

    /// Returns the requested budget weight.
    pub const fn budget_weight(self) -> u32 {
        self.budget_weight
    }
}

impl Default for QueryQos {
    fn default() -> Self {
        Self::new(Priority::Normal)
    }
}
