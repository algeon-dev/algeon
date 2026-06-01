//! Comparator stubs for the Milestone 1a parity harness.
//!
//! The full comparator families described in the plan
//! (`docs/plan/2026-05-20-cugraph-rs-full-api-support-plan.md`,
//! "Comparator families") are implemented incrementally alongside the
//! `#[ignore]` GPU execution rows that exercise them. For Milestone 1a the
//! CPU-only test driver only asserts that each parity row declares a
//! comparator family — the actual comparison runs once Python cuGraph
//! is available locally and the GPU execution tests are opt-in via
//! `--include-ignored`.

use datafusion_nexus::cugraph_parity::ParityComparator;

pub fn family_label(comparator: ParityComparator) -> &'static str {
    comparator.as_str()
}
