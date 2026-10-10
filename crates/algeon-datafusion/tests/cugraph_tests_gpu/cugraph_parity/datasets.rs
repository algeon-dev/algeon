//! Dataset loading helpers shared across cuGraph parity rows.
//!
//! Both the CPU manifest checks and the registry-driven GPU acceptance test
//! use these entry points so the Rust and Python sides observe byte-identical
//! input.

use std::{
    fs,
    path::{Path, PathBuf},
};

use algeon_datafusion::cugraph_sql::parity::{
    ParityDataset, ParityManifest, ParityManifestError, data_root, load_manifest,
};
use sha2::{Digest, Sha256};

pub struct LoadedDatasets {
    pub data_root: PathBuf,
    pub manifest: ParityManifest,
}

#[derive(Debug, Clone, Copy)]
pub struct ParityEdge {
    pub source: i64,
    pub destination: i64,
    pub weight: f64,
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
            Self::Manifest(source) => write!(
                f,
                "{source}. Run `fixture/graph/prepare.sh` to copy cuGraph datasets from components/cugraph/cugraph"
            ),
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

pub fn load_edges(data_root: &Path, dataset: &ParityDataset) -> Result<Vec<ParityEdge>, String> {
    if dataset.format != "space_separated_edges" || dataset.edge_columns != ["dst", "src", "weight"] {
        return Err(format!(
            "dataset '{}' has unsupported edge format {} / {:?}",
            dataset.id, dataset.format, dataset.edge_columns
        ));
    }
    let path = data_root.join(&dataset.path);
    let raw = fs::read_to_string(&path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    raw.lines()
        .enumerate()
        .map(|(index, line)| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 3 {
                return Err(format!("{}:{} expected dst src weight, got {line:?}", path.display(), index + 1));
            }
            let parse_i64 = |value: &str, name: &str| {
                value
                    .parse::<i64>()
                    .map_err(|error| format!("{}:{} invalid {name} {value:?}: {error}", path.display(), index + 1))
            };
            let destination = parse_i64(fields[0], "destination")?;
            let source = parse_i64(fields[1], "source")?;
            let weight = fields[2]
                .parse::<f64>()
                .map_err(|error| format!("{}:{} invalid weight {:?}: {error}", path.display(), index + 1, fields[2]))?;
            if !weight.is_finite() {
                return Err(format!("{}:{} weight must be finite", path.display(), index + 1));
            }
            Ok(ParityEdge { source, destination, weight })
        })
        .collect()
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
