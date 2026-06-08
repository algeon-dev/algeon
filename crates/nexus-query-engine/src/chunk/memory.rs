/// Slot model used to derive a per-source-task chunk target from a device budget.
///
/// Each slot represents one live chunk-sized region that may coexist during the
/// source task. The model is a planning heuristic, not an executor allocation
/// ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceTaskMemoryModel {
    source_read_slots: u64,
    source_adjacent_operator_slots: u64,
    downstream_operator_slots: u64,
    retained_output_slots: u64,
    source_read_preflight_slack_slots: u64,
}

impl SourceTaskMemoryModel {
    /// Source -> optional source-local unary stages -> aggregate.
    pub const fn aggregate_pipeline(has_source_adjacent_stages: bool) -> Self {
        Self {
            source_read_slots: 1,
            // Source-adjacent filter/projection chains can retain a large input,
            // an operator work region, and the transformed output before the
            // downstream aggregate takes ownership.
            source_adjacent_operator_slots: if has_source_adjacent_stages { 3 } else { 0 },
            downstream_operator_slots: 1,
            retained_output_slots: 1,
            source_read_preflight_slack_slots: 1,
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

    pub const fn live_slot_count(self) -> u64 {
        self.source_read_slots
            + self.source_adjacent_operator_slots
            + self.downstream_operator_slots
            + self.retained_output_slots
            + self.source_read_preflight_slack_slots
    }
}
