use crate::error::QueryError;
pub use crate::execution_contracts::MemoryReservationRequest;

/// Retry/resume facts attached to bounded retry failures.
///
/// These fields document what bounded execution can safely retry after a
/// reservation denial or retryable `CudfResourceExhausted` failure: the retry
/// action, resume boundary, already-committed chunks/partials, and merge fan-in.
/// The fact keys emitted by
/// [`Self::attach_to_error`] are stable diagnostics used by tests and higher
/// level error reports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RetryFacts {
    /// Stable retry action string, for example splitting a chunk or reducing
    /// merge fan-in.
    pub(crate) retry_action: Option<String>,
    /// Stable boundary where execution can resume after a retry.
    pub(crate) resume_boundary: Option<String>,
    /// Number of source chunks already committed before the failure.
    pub(crate) committed_chunks: Option<usize>,
    /// Number of partial results already committed before the failure.
    pub(crate) committed_partials: Option<usize>,
    /// Merge level being processed when the failure occurred.
    pub(crate) merge_level: Option<usize>,
    /// Merge fan-in attempted when the failure occurred.
    pub(crate) merge_fan_in: Option<usize>,
}

impl RetryFacts {
    /// Creates empty retry facts.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Sets the stable retry-action diagnostic value.
    #[must_use]
    pub(crate) fn with_retry_action(mut self, retry_action: impl Into<String>) -> Self {
        self.retry_action = Some(retry_action.into());
        self
    }

    /// Sets the stable resume-boundary diagnostic value.
    #[must_use]
    pub(crate) fn with_resume_boundary(mut self, resume_boundary: impl Into<String>) -> Self {
        self.resume_boundary = Some(resume_boundary.into());
        self
    }

    /// Records committed source chunks.
    #[must_use]
    pub(crate) fn with_committed_chunks(mut self, committed_chunks: usize) -> Self {
        self.committed_chunks = Some(committed_chunks);
        self
    }

    /// Records committed partial results.
    #[must_use]
    pub(crate) fn with_committed_partials(mut self, committed_partials: usize) -> Self {
        self.committed_partials = Some(committed_partials);
        self
    }

    /// Records the merge level involved in a retryable failure.
    #[must_use]
    pub(crate) fn with_merge_level(mut self, merge_level: usize) -> Self {
        self.merge_level = Some(merge_level);
        self
    }

    /// Records the merge fan-in involved in a retryable failure.
    #[must_use]
    pub(crate) fn with_merge_fan_in(mut self, merge_fan_in: usize) -> Self {
        self.merge_fan_in = Some(merge_fan_in);
        self
    }

    /// Attaches all present retry facts to an error.
    ///
    /// This preserves the original error code/status and only adds structured
    /// facts. Fact keys are stable diagnostic contracts.
    pub(crate) fn attach_to_error(&self, mut error: QueryError) -> QueryError {
        if let Some(retry_action) = &self.retry_action {
            error = error.with_fact_if_absent("retry_action", retry_action);
        }
        if let Some(resume_boundary) = &self.resume_boundary {
            error = error.with_fact_if_absent("resume_boundary", resume_boundary);
        }
        if let Some(committed_chunks) = self.committed_chunks {
            error = error.with_fact_if_absent("committed_chunks", committed_chunks);
        }
        if let Some(committed_partials) = self.committed_partials {
            error = error.with_fact_if_absent("committed_partials", committed_partials);
        }
        if let Some(merge_level) = self.merge_level {
            error = error.with_fact_if_absent("merge_level", merge_level);
        }
        if let Some(merge_fan_in) = self.merge_fan_in {
            error = error.with_fact_if_absent("merge_fan_in", merge_fan_in);
        }
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ReservationKind;

    #[test]
    fn retry_facts_attach_stable_resume_and_merge_context() {
        let request = MemoryReservationRequest::new(ReservationKind::FinalMerge, 2048, 1024);
        let error = RetryFacts::new()
            .with_retry_action("reduce_merge_fan_in")
            .with_resume_boundary("final_merge")
            .with_committed_chunks(8)
            .with_committed_partials(8)
            .with_merge_level(1)
            .with_merge_fan_in(2)
            .attach_to_error(request.denied_error(1024));

        assert_eq!(error.fact_value("retry_action"), Some("reduce_merge_fan_in"));
        assert_eq!(error.fact_value("resume_boundary"), Some("final_merge"));
        assert_eq!(error.fact_value("committed_chunks"), Some("8"));
        assert_eq!(error.fact_value("committed_partials"), Some("8"));
        assert_eq!(error.fact_value("merge_level"), Some("1"));
        assert_eq!(error.fact_value("merge_fan_in"), Some("2"));
    }
}
