/// Slot model used to derive a per-source-task chunk target from a device budget.
///
/// Each slot represents one live region the size of a chunk's decoded read
/// columns ([`crate::chunk::decoded`]) that may coexist during the source
/// task. A consumer that runs several chunks at once keeps every chunk's
/// regions live together, so the slots count once per concurrent chunk. A
/// consumer that reads one chunk ahead of itself adds one slot: the next
/// chunk's decoded output is live next to the current chunk's regions, and its
/// read scratch, bounded by the executor to one chunk target, reuses the
/// preflight slack slot the current chunk no longer needs once its own read
/// has finished. Stages that widen their input, such as a projection adding
/// columns, are not modeled. The model is a planning heuristic. It does not
/// record the executor's allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceTaskMemoryModel {
    source_read_slots: u64,
    source_adjacent_operator_slots: u64,
    downstream_operator_slots: u64,
    retained_output_slots: u64,
    source_read_preflight_slack_slots: u64,
    concurrent_chunks: u64,
    read_ahead_slots: u64,
}

impl SourceTaskMemoryModel {
    /// Source -> optional source-local unary stages -> aggregate.
    pub const fn aggregate_pipeline(has_source_adjacent_stages: bool) -> Self {
        Self {
            source_read_slots: 1,
            // Source-adjacent filter/projection chains can retain a large input,
            // an operator work region, and the transformed output before the
            // downstream aggregate receives it.
            source_adjacent_operator_slots: if has_source_adjacent_stages { 3 } else { 0 },
            downstream_operator_slots: 1,
            retained_output_slots: 1,
            source_read_preflight_slack_slots: 1,
            concurrent_chunks: 1,
            read_ahead_slots: 0,
        }
    }

    /// Shared source reused by a join leaf and an aggregate branch.
    pub const fn shared_join_aggregate_stream(has_source_adjacent_stages: bool) -> Self {
        Self {
            source_read_slots: 1,
            source_adjacent_operator_slots: if has_source_adjacent_stages { 3 } else { 0 },
            downstream_operator_slots: 1,
            retained_output_slots: 2,
            source_read_preflight_slack_slots: 1,
            concurrent_chunks: 1,
            read_ahead_slots: 0,
        }
    }

    /// Source -> optional source-local unary stages -> join build/probe.
    pub const fn join_leaf_stream(has_source_adjacent_stages: bool) -> Self {
        Self {
            source_read_slots: 1,
            source_adjacent_operator_slots: if has_source_adjacent_stages { 1 } else { 0 },
            downstream_operator_slots: 1,
            retained_output_slots: 2,
            source_read_preflight_slack_slots: 1,
            concurrent_chunks: 1,
            read_ahead_slots: 0,
        }
    }

    /// Source -> source-local unary stages -> one frame concatenated from the
    /// stage outputs, for consumers that take only frames.
    ///
    /// The decoded chunk and its stage output are live next to the retained
    /// stage outputs of earlier chunks, and the final concatenation copies
    /// those outputs once more. The last two regions grow with the stage
    /// output and do not scale with the chunk. Because of that, their single
    /// slots assume stages that do not widen their input.
    pub const fn unary_chain_frame() -> Self {
        Self {
            source_read_slots: 1,
            source_adjacent_operator_slots: 1,
            downstream_operator_slots: 1,
            retained_output_slots: 1,
            source_read_preflight_slack_slots: 1,
            concurrent_chunks: 1,
            read_ahead_slots: 0,
        }
    }

    /// Returns live slots reserved for the source read itself.
    pub const fn source_read_slots(self) -> u64 {
        self.source_read_slots
    }

    /// Returns live slots reserved for filters/projections adjacent to the source.
    pub const fn source_adjacent_operator_slots(self) -> u64 {
        self.source_adjacent_operator_slots
    }

    /// Returns live slots reserved for the immediate downstream operator.
    pub const fn downstream_operator_slots(self) -> u64 {
        self.downstream_operator_slots
    }

    /// Returns live slots reserved for output retained after the task completes.
    pub const fn retained_output_slots(self) -> u64 {
        self.retained_output_slots
    }

    /// Returns slack slots for source-read preflight overhead.
    pub const fn source_read_preflight_slack_slots(self) -> u64 {
        self.source_read_preflight_slack_slots
    }

    /// Returns this model for a consumer that runs up to `chunks` source
    /// chunks at once, each keeping its own slots live. Zero counts as one.
    pub const fn with_concurrent_chunks(mut self, chunks: usize) -> Self {
        self.concurrent_chunks = if chunks > 1 { chunks as u64 } else { 1 };
        self
    }

    /// Returns this model for a consumer that keeps one chunk read ahead of
    /// itself when `read_ahead` is true, and the model unchanged otherwise.
    pub const fn with_read_ahead(mut self, read_ahead: bool) -> Self {
        self.read_ahead_slots = if read_ahead { 1 } else { 0 };
        self
    }

    /// Returns the live slots of every concurrent chunk together, plus the
    /// read-ahead slot.
    pub const fn live_slot_count(self) -> u64 {
        (self.source_read_slots
            + self.source_adjacent_operator_slots
            + self.downstream_operator_slots
            + self.retained_output_slots
            + self.source_read_preflight_slack_slots)
            * self.concurrent_chunks
            + self.read_ahead_slots
    }
}
