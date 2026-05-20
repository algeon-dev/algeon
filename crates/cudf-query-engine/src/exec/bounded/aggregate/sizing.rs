// Closed-form upper bound for the device bytes a bounded
// partial-aggregate materialize can produce. Sibling to
// `exec/bounded/join/sizing.rs:join_output_upper_bound_bytes`, but
// with the row-count rule baked in for global vs grouped aggregates.
//
// The bound is the same per-column formula as the join helper applied
// to the partial-state output schema with the right row count:
//   - Global aggregate output: 1 row.
//   - Grouped aggregate output: at most `input_rows` rows (every input
//     row could be a unique group key). Without a libcudf-level
//     grouped-aggregate output-row-count preflight, this is the only
//     deterministic ceiling; design v8 risk #7 / Phase 4.G tracks the
//     follow-up to tighten it.
//
// Caller responsibility: build the per-column descriptors from the
// partial-state schema. Grouped partial output begins with the group
// key columns (`exec/lowering/aggregate.rs:lower_partial_aggregate_spec`
// places `output_key_names` first), followed by aggregate state
// columns. For string-typed group keys the caller must supply
// `input_max_string_length_bytes` from the input frame. See
// `docs/design/2026-05-20-tpcds-sf100-q72-join-materialize-fix.md`
// Phase 4.A.2.

use crate::{
    error::Result,
    exec::bounded::join::{JoinOutputColumnUpperBoundInput, join_output_upper_bound_bytes},
};

/// Row-count shape of the partial-aggregate output.
#[derive(Debug, Clone, Copy)]
pub enum PartialAggregateShape {
    /// Global (no GROUP BY) — partial output has exactly 1 row.
    Global,
    /// Grouped — partial output has at most `input_rows` rows. This is
    /// the conservative bound used in strict mode until a libcudf
    /// grouped-aggregate row-count preflight lands (Phase 4.G follow-up
    /// per the q72 design doc).
    Grouped { input_rows: u64 },
}

impl PartialAggregateShape {
    pub fn output_row_count_upper_bound(&self) -> u64 {
        match self {
            Self::Global => 1,
            Self::Grouped { input_rows } => *input_rows,
        }
    }
}

/// Closed-form upper bound (in bytes) on the device allocation a
/// partial-aggregate output table can occupy.
///
/// Delegates per-column sizing to the join helper. Returns `Ok(0)`
/// when `columns.is_empty()`. Returns
/// `QueryErrorCode::UnsupportedPlanShape` for any column whose Arrow
/// dtype is not supported by strict-bounded sizing; the caller's
/// admission gate is expected to reject such plans earlier.
pub fn partial_aggregate_upper_bound_bytes(
    shape: PartialAggregateShape,
    columns: &[JoinOutputColumnUpperBoundInput],
) -> Result<u64> {
    join_output_upper_bound_bytes(shape.output_row_count_upper_bound(), columns)
}
