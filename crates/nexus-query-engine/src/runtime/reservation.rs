use super::*;

/// Reservation request submitted to bounded execution admission.
///
/// A request describes capacity submitted before retaining or committing a
/// bounded memory boundary. Some requests happen before invoking a cudf
/// operation; retained-output requests may happen after an operation, once
/// output bytes are observed. The internal reservation tracker grants a request
/// only if it fits in the remaining bounded budget. The returned internal guard
/// (`ActiveReservation`) represents admitted capacity and releases that capacity
/// either when explicitly released or when dropped.
///
/// Core reservation fields are recorded in metrics, and optional context is
/// included in failure facts when present. Keep field names, fact keys, and
/// memory-boundary strings stable unless report consumers are migrated
/// deliberately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReservationRequest {
    /// Category of memory being reserved.
    pub reservation_kind: ReservationKind,
    /// Memory space charged by this request.
    ///
    /// Direct request construction defaults to `GPU:0` for simple contract
    /// tests. Bounded execution stamps requests with
    /// [`ExecutionContext::device_memory_space`] before recording metrics or
    /// emitting denial errors.
    pub memory_space: MemorySpaceId,
    /// Stable boundary name for diagnostics.
    ///
    /// Defaults to `reservation_kind.as_str()` but callers should override it
    /// when a more precise boundary is part of the contract, such as
    /// `join_hash_table` or `host_materialization`.
    pub memory_boundary: String,
    /// Bytes requested for admission.
    pub requested_bytes: u64,
    /// Total bounded device budget used for this admission check.
    pub budget_bytes: u64,
    /// Optional plan node id associated with this reservation.
    pub node_id: Option<String>,
    /// Optional execution stage, such as a bounded join or retry stage.
    pub stage: Option<String>,
    /// Optional source chunk index.
    pub chunk_index: Option<usize>,
    /// Optional source file index.
    pub file_index: Option<usize>,
    /// Optional source path.
    pub path: Option<String>,
    /// Optional source row groups covered by the request.
    pub row_groups: Option<Vec<usize>>,
    /// Optional retry attempt number.
    pub attempt: Option<usize>,
    /// Optional maximum retry attempts for the current retry path.
    pub max_attempts: Option<usize>,
    /// Optional compiled pipeline task id associated with this reservation.
    pub compiled_task_id: Option<String>,
    /// Optional compiled pipeline resident resource id associated with this reservation.
    pub resident_resource_id: Option<String>,
    /// Optional quality label for source-read sizing diagnostics.
    pub source_read_contract_quality: Option<String>,
    /// Optional chunk-planning byte signal used before source-read admission.
    pub source_read_planning_bytes: Option<u64>,
    /// Optional coarse source-read byte estimate retained for diagnostics.
    pub source_read_estimated_bytes: Option<u64>,
}

impl MemoryReservationRequest {
    /// Creates a reservation request for a kind, requested byte count, and budget.
    pub fn new(reservation_kind: ReservationKind, requested_bytes: u64, budget_bytes: u64) -> Self {
        Self {
            reservation_kind,
            memory_space: MemorySpaceId::Device { device_ordinal: 0 },
            memory_boundary: reservation_kind.as_str().to_string(),
            requested_bytes,
            budget_bytes,
            node_id: None,
            stage: None,
            chunk_index: None,
            file_index: None,
            path: None,
            row_groups: None,
            attempt: None,
            max_attempts: None,
            compiled_task_id: None,
            resident_resource_id: None,
            source_read_contract_quality: None,
            source_read_planning_bytes: None,
            source_read_estimated_bytes: None,
        }
    }

    /// Overrides the memory space charged by this request.
    #[must_use]
    pub fn with_memory_space(mut self, memory_space: MemorySpaceId) -> Self {
        self.memory_space = memory_space;
        self
    }

    /// Overrides the stable memory-boundary diagnostic string.
    #[must_use]
    pub fn with_memory_boundary(mut self, memory_boundary: impl Into<String>) -> Self {
        self.memory_boundary = memory_boundary.into();
        self
    }

    /// Overrides the requested byte count while preserving request diagnostics.
    #[must_use]
    pub fn with_requested_bytes(mut self, requested_bytes: u64) -> Self {
        self.requested_bytes = requested_bytes;
        self
    }

    /// Attaches a plan node id to the request.
    #[must_use]
    pub fn with_node_id(mut self, node_id: impl ToString) -> Self {
        self.node_id = Some(node_id.to_string());
        self
    }

    /// Attaches an execution stage to the request.
    #[must_use]
    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    /// Attaches a source chunk index to the request.
    #[must_use]
    pub fn with_chunk_index(mut self, chunk_index: usize) -> Self {
        self.chunk_index = Some(chunk_index);
        self
    }

    /// Attaches a source file index to the request.
    #[must_use]
    pub fn with_file_index(mut self, file_index: usize) -> Self {
        self.file_index = Some(file_index);
        self
    }

    /// Attaches a source path to the request.
    #[must_use]
    pub fn with_path(mut self, path: impl ToString) -> Self {
        self.path = Some(path.to_string());
        self
    }

    /// Attaches source row groups to the request.
    #[must_use]
    pub fn with_row_groups(mut self, row_groups: impl Into<Vec<usize>>) -> Self {
        self.row_groups = Some(row_groups.into());
        self
    }

    /// Attaches retry attempt metadata to the request.
    #[must_use]
    pub fn with_attempt(mut self, attempt: usize, max_attempts: usize) -> Self {
        self.attempt = Some(attempt);
        self.max_attempts = Some(max_attempts);
        self
    }

    /// Attaches a compiled pipeline task id to the request.
    #[must_use]
    pub fn with_compiled_task_id(mut self, compiled_task_id: impl ToString) -> Self {
        self.compiled_task_id = Some(compiled_task_id.to_string());
        self
    }

    /// Attaches a compiled pipeline resident resource id to the request.
    #[must_use]
    pub fn with_resident_resource_id(mut self, resident_resource_id: impl ToString) -> Self {
        self.resident_resource_id = Some(resident_resource_id.to_string());
        self
    }

    /// Attaches source-read sizing contract diagnostics to the request.
    #[must_use]
    pub fn with_source_read_contract(
        mut self,
        quality: impl Into<String>,
        planning_bytes: Option<u64>,
        estimated_bytes: Option<u64>,
    ) -> Self {
        self.source_read_contract_quality = Some(quality.into());
        self.source_read_planning_bytes = planning_bytes;
        self.source_read_estimated_bytes = estimated_bytes;
        self
    }

    /// Builds the stable reservation-denial error for this request.
    ///
    /// `granted_bytes` is the capacity that remained when admission failed, not
    /// a partial grant. The emitted facts are report-facing and should stay in
    /// sync with reservation metrics.
    pub fn denied_error(&self, granted_bytes: u64) -> QueryError {
        let mut error = QueryError::new(
            crate::error::QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.memory.reserve",
            "memory reservation failed before native cudf execution",
        )
        .with_fact("memory_boundary", &self.memory_boundary)
        .with_fact("reservation_kind", self.reservation_kind)
        .with_fact("requested_bytes", self.requested_bytes)
        .with_fact("granted_bytes", granted_bytes)
        .with_fact("budget_bytes", self.budget_bytes);

        if let Some(node_id) = &self.node_id {
            error = error.with_fact("node_id", node_id);
        }
        if let Some(stage) = &self.stage {
            error = error.with_fact("stage", stage);
        }
        if let Some(chunk_index) = self.chunk_index {
            error = error.with_fact("chunk_index", chunk_index);
        }
        if let Some(file_index) = self.file_index {
            error = error.with_fact("file_index", file_index);
        }
        if let Some(path) = &self.path {
            error = error.with_fact("path", path);
        }
        if let Some(row_groups) = &self.row_groups {
            error = error.with_fact("row_groups", comma_separated_usize(row_groups));
        }
        if let Some(attempt) = self.attempt {
            error = error.with_fact("attempt", attempt);
        }
        if let Some(max_attempts) = self.max_attempts {
            error = error.with_fact("max_attempts", max_attempts);
        }
        if let Some(compiled_task_id) = &self.compiled_task_id {
            error = error.with_fact("compiled_task_id", compiled_task_id);
        }
        if let Some(resident_resource_id) = &self.resident_resource_id {
            error = error.with_fact("resident_resource_id", resident_resource_id);
        }
        if let Some(quality) = &self.source_read_contract_quality {
            error = error.with_fact("source_read_contract_quality", quality);
        }
        if let Some(planning_bytes) = self.source_read_planning_bytes {
            error = error.with_fact("source_read_planning_bytes", planning_bytes);
        }
        if let Some(estimated_bytes) = self.source_read_estimated_bytes {
            error = error.with_fact("source_read_estimated_bytes", estimated_bytes);
        }

        error
    }
}

/// Retry/resume facts attached to bounded retry failures.
///
/// These fields document what bounded execution can safely retry after a
/// reservation denial or retryable `CudfResourceExhausted` failure: the retry
/// action, resume boundary, already-committed chunks/partials, and merge fan-in.
/// The fact keys emitted by
/// [`Self::attach_to_error`] are stable diagnostics used by tests and higher
/// level error reports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetryFacts {
    /// Stable retry action string, for example splitting a chunk or reducing
    /// merge fan-in.
    pub retry_action: Option<String>,
    /// Stable boundary where execution can resume after a retry.
    pub resume_boundary: Option<String>,
    /// Number of source chunks already committed before the failure.
    pub committed_chunks: Option<usize>,
    /// Number of partial results already committed before the failure.
    pub committed_partials: Option<usize>,
    /// Merge level being processed when the failure occurred.
    pub merge_level: Option<usize>,
    /// Merge fan-in attempted when the failure occurred.
    pub merge_fan_in: Option<usize>,
}

impl RetryFacts {
    /// Creates empty retry facts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the stable retry-action diagnostic value.
    #[must_use]
    pub fn with_retry_action(mut self, retry_action: impl Into<String>) -> Self {
        self.retry_action = Some(retry_action.into());
        self
    }

    /// Sets the stable resume-boundary diagnostic value.
    #[must_use]
    pub fn with_resume_boundary(mut self, resume_boundary: impl Into<String>) -> Self {
        self.resume_boundary = Some(resume_boundary.into());
        self
    }

    /// Records committed source chunks.
    #[must_use]
    pub fn with_committed_chunks(mut self, committed_chunks: usize) -> Self {
        self.committed_chunks = Some(committed_chunks);
        self
    }

    /// Records committed partial results.
    #[must_use]
    pub fn with_committed_partials(mut self, committed_partials: usize) -> Self {
        self.committed_partials = Some(committed_partials);
        self
    }

    /// Records the merge level involved in a retryable failure.
    #[must_use]
    pub fn with_merge_level(mut self, merge_level: usize) -> Self {
        self.merge_level = Some(merge_level);
        self
    }

    /// Records the merge fan-in involved in a retryable failure.
    #[must_use]
    pub fn with_merge_fan_in(mut self, merge_fan_in: usize) -> Self {
        self.merge_fan_in = Some(merge_fan_in);
        self
    }

    /// Attaches all present retry facts to an error.
    ///
    /// This preserves the original error code/status and only adds structured
    /// facts. Fact keys are stable diagnostic contracts.
    pub fn attach_to_error(&self, mut error: QueryError) -> QueryError {
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

fn comma_separated_usize(values: &[usize]) -> String {
    let mut output = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}
