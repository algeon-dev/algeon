//! Runtime memory-resource and source-scheduling policies.

use query_planning::error::{QueryError, QueryErrorCode, Result};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryResourcePolicy {
    #[default]
    DefaultCuda,
    Pool {
        initial_pool_size: usize,
        maximum_pool_size: Option<usize>,
    },
    ExternalRuntime,
}

/// Native source scheduling controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSchedulerPolicy {
    max_source_reads_in_flight: usize,
}

impl SourceSchedulerPolicy {
    /// Creates a source scheduler with one read at a time and no prefetch.
    pub const fn serialized() -> Self {
        Self { max_source_reads_in_flight: 1 }
    }

    /// Sets the maximum concurrent source reads. Zero is rejected by
    /// [`Self::validate`]; the default permits two reads.
    pub const fn new(max_source_reads_in_flight: usize) -> Self {
        Self { max_source_reads_in_flight }
    }

    /// Returns the configured concurrent source-read bound.
    pub const fn max_source_reads_in_flight(self) -> usize {
        self.max_source_reads_in_flight
    }

    /// Returns whether the policy permits more than one concurrent source read.
    pub const fn enables_prefetch(self) -> bool {
        self.max_source_reads_in_flight > 1
    }

    /// Rejects zero concurrent source reads with `InvalidConfiguration`.
    pub fn validate(self) -> Result<()> {
        if self.max_source_reads_in_flight == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidConfiguration,
                "query_engine.source_scheduler_policy.validate",
                "source scheduler max_source_reads_in_flight must be greater than zero",
            )
            .with_fact("field", "max_source_reads_in_flight")
            .with_fact("reason", "zero"));
        }
        Ok(())
    }
}

impl Default for SourceSchedulerPolicy {
    fn default() -> Self {
        Self { max_source_reads_in_flight: 2 }
    }
}
