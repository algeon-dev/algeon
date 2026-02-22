//! Dataset loading helpers shared across cuGraph parity rows.
//!
//! The default `cargo test --features cugraph` invocation reaches this
//! module only through the CPU-only manifest verification tests in
//! `tests/cugraph_parity.rs`. GPU execution paths (currently
//! `#[ignore]`) will reuse the same dataset loading entry points so the
//! Rust and Python sides observe byte-identical input.

use std::path::PathBuf;

use datafusion_cudf::cugraph_parity::{
    ParityManifest, ParityManifestError, data_root, load_manifest, verify_dataset_checksums,
};

pub struct LoadedDatasets {
    pub data_root: PathBuf,
    pub manifest: ParityManifest,
}

pub fn load_and_verify() -> Result<LoadedDatasets, ParityManifestError> {
    let root = data_root();
    let manifest = load_manifest(&root)?;
    verify_dataset_checksums(&root, &manifest)?;
    Ok(LoadedDatasets { data_root: root, manifest })
}
