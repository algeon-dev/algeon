//! Dataset loading helpers shared across cuGraph parity rows.
//!
//! The default `cargo test --features cugraph` invocation reaches this
//! module only through the CPU-only manifest verification tests in
//! `tests/cugraph_parity.rs`. GPU execution paths (currently
//! `#[ignore]`) will reuse the same dataset loading entry points so the
//! Rust and Python sides observe byte-identical input.

use std::{
    fs,
    path::{Path, PathBuf},
};

use datafusion_cudf::cugraph_parity::{ParityManifest, ParityManifestError, data_root, load_manifest};
use sha2::{Digest, Sha256};

pub struct LoadedDatasets {
    pub data_root: PathBuf,
    pub manifest: ParityManifest,
}

#[derive(Debug)]
pub enum LoadAndVerifyError {
    Manifest(ParityManifestError),
    DatasetMissing { id: String, path: PathBuf, source: std::io::Error },
    DatasetSha256Mismatch { id: String, expected: String, actual: String, path: PathBuf },
}

impl std::fmt::Display for LoadAndVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Manifest(source) => write!(f, "{source}"),
            Self::DatasetMissing { id, path, source } => {
                write!(f, "cuGraph dataset '{id}' missing at {}: {source}", path.display())
            },
            Self::DatasetSha256Mismatch { id, expected, actual, path } => write!(
                f,
                "cuGraph dataset '{id}' SHA-256 mismatch: expected {expected}, got {actual} at {}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for LoadAndVerifyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Manifest(source) => Some(source),
            Self::DatasetMissing { source, .. } => Some(source),
            Self::DatasetSha256Mismatch { .. } => None,
        }
    }
}

impl From<ParityManifestError> for LoadAndVerifyError {
    fn from(source: ParityManifestError) -> Self {
        Self::Manifest(source)
    }
}

pub fn load_and_verify() -> Result<LoadedDatasets, LoadAndVerifyError> {
    let root = data_root();
    let manifest = load_manifest(&root)?;
    verify_dataset_checksums(&root, &manifest)?;
    Ok(LoadedDatasets { data_root: root, manifest })
}

fn verify_dataset_checksums(data_root: &Path, manifest: &ParityManifest) -> Result<(), LoadAndVerifyError> {
    for dataset in &manifest.datasets {
        verify_one_checksum(data_root, &dataset.id, &dataset.path, &dataset.sha256)?;
    }
    for reference in &manifest.references {
        verify_one_checksum(data_root, &reference.id, &reference.path, &reference.sha256)?;
    }
    Ok(())
}

fn verify_one_checksum(data_root: &Path, id: &str, relpath: &str, expected: &str) -> Result<(), LoadAndVerifyError> {
    let path = data_root.join(relpath);
    let bytes = fs::read(&path).map_err(|source| LoadAndVerifyError::DatasetMissing {
        id: id.to_string(),
        path: path.clone(),
        source,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual = hex::encode(hasher.finalize());
    if actual != expected {
        return Err(LoadAndVerifyError::DatasetSha256Mismatch {
            id: id.to_string(),
            expected: expected.to_string(),
            actual,
            path,
        });
    }
    Ok(())
}
