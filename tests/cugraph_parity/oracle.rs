//! Oracle JSON lookup with structured "needs refresh" diagnostics.

use std::path::Path;

use datafusion_nexus::cugraph_parity::{OracleLookup, ParityManifest, ParitySpec, load_oracle, oracle_path};

/// Inspect the oracle JSON for one parity row. Returns either a parsed
/// document or a structured status the caller can render as a skip.
///
/// The manifest is passed through so the oracle's
/// `header.dataset_sha256` is cross-checked against the currently
/// vendored dataset bytes; a stale oracle generated from older bytes
/// surfaces as a `Mismatch` rather than silently passing.
pub fn lookup(data_root: &Path, manifest: &ParityManifest, spec: &ParitySpec) -> OracleLookup {
    load_oracle(data_root, spec, Some(manifest))
}

/// Render the refresh command for a parity row so a skipped test still
/// tells the operator exactly how to populate the oracle JSON.
///
/// The binary derives parameters from the parity spec registry, so the
/// command only carries the `--algorithm` / `--dataset` selectors that
/// `refresh_cugraph_oracle` actually accepts; the target path is
/// printed on a separate line so a copy-paste of the command line is
/// shell-runnable as-is.
pub fn refresh_hint(data_root: &Path, spec: &ParitySpec) -> String {
    format!(
        "to populate oracle JSON for `{label}`, run:\n      \
         cargo run --release --features cugraph --bin refresh_cugraph_oracle -- \
         --algorithm {algorithm} --dataset {dataset}\n      \
         (writes to {oracle})",
        label = spec.label,
        algorithm = spec.algorithm.native_algorithm().as_str(),
        dataset = spec.dataset,
        oracle = oracle_path(data_root, spec).display(),
    )
}
