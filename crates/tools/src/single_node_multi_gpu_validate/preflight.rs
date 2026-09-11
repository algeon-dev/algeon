//! Read-only fixture preflight: layout validation and algorithm-qualified
//! fingerprints for one TPC-H and one canonical DBLP fixture.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use serde::Serialize;
use workloads::{
    clickbench::describe_clickbench_dataset_with_content_fingerprint,
    graph::{DblpDatasetMetadata, WorkloadResult, describe_canonical_dblp_dataset, describe_dblp_dataset},
    metadata::WorkloadDatasetMetadata,
    tpcds::describe_tpcds_dataset_with_content_fingerprint,
    tpch::describe_tpch_dataset_with_content_fingerprint,
};

use super::profile::{FixtureKind, FixtureWorkloadProfile};
use crate::ToolResult;

#[derive(Debug)]
pub(super) struct FixturePreflightError {
    pub(super) workload: &'static str,
    pub(super) reason: &'static str,
    pub(super) path: Option<PathBuf>,
    pub(super) detail: Option<String>,
}

impl FixturePreflightError {
    pub(super) fn selection(detail: impl Into<String>) -> Self {
        Self { workload: "fixture_set", reason: "invalid_selection", path: None, detail: Some(detail.into()) }
    }

    pub(super) fn missing(workload: &'static str, path: &Path) -> Self {
        Self { workload, reason: "missing_fixture", path: Some(path.to_path_buf()), detail: None }
    }

    pub(super) fn invalid_layout(workload: &'static str, path: &Path, detail: impl Into<String>) -> Self {
        Self { workload, reason: "invalid_layout", path: Some(path.to_path_buf()), detail: Some(detail.into()) }
    }
}

impl std::fmt::Display for FixturePreflightError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "fixture_preflight workload={} reason={}", self.workload, self.reason)?;
        if let Some(path) = &self.path {
            write!(formatter, " path={}", path.display())?;
        }
        if let Some(detail) = &self.detail {
            write!(formatter, " detail={detail}")?;
        }
        Ok(())
    }
}

impl std::error::Error for FixturePreflightError {}

pub(super) struct FixturePreflight;

/// How many fixture kinds a mode requires. `--fixture-preflight` pins the
/// contract; a measured run validates whatever mix it will execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FixtureSelectionPolicy {
    BothKindsExactly,
    AtLeastOneKind,
}

impl FixturePreflight {
    pub(super) fn collect(
        workloads: &[FixtureWorkloadProfile],
        policy: FixtureSelectionPolicy,
    ) -> ToolResult<Vec<FixturePreflightArtifact>> {
        Self::collect_with_dblp_contract(workloads, policy, DblpFixtureContract::CanonicalV12)
    }

    #[cfg(test)]
    pub(super) fn collect_sample_fixture_for_artifact_test(
        workloads: &[FixtureWorkloadProfile],
        policy: FixtureSelectionPolicy,
    ) -> ToolResult<Vec<FixturePreflightArtifact>> {
        Self::collect_with_dblp_contract(workloads, policy, DblpFixtureContract::Sample)
    }

    fn collect_with_dblp_contract(
        workloads: &[FixtureWorkloadProfile],
        policy: FixtureSelectionPolicy,
        dblp_contract: DblpFixtureContract,
    ) -> ToolResult<Vec<FixturePreflightArtifact>> {
        match policy {
            FixtureSelectionPolicy::BothKindsExactly if workloads.len() != 2 => {
                return Err(FixturePreflightError::selection(
                    "validation requires exactly one tpch and one dblp fixture",
                )
                .into());
            },
            FixtureSelectionPolicy::AtLeastOneKind if workloads.is_empty() => {
                return Err(FixturePreflightError::selection("a measured run requires at least one fixture").into());
            },
            _ => {},
        }
        let mut preflighted = Vec::with_capacity(workloads.len());
        let mut kinds = BTreeSet::new();
        for workload in workloads {
            let is_new_kind = kinds.insert(workload.kind);
            if policy == FixtureSelectionPolicy::BothKindsExactly && !is_new_kind {
                return Err(FixturePreflightError::selection("fixture kinds must be unique").into());
            }
            if policy == FixtureSelectionPolicy::AtLeastOneKind && !is_new_kind && workload.kind != FixtureKind::Dblp {
                return Err(FixturePreflightError::selection("only DBLP supports multiple fixture selections").into());
            }
            if policy == FixtureSelectionPolicy::BothKindsExactly
                && workload.kind == FixtureKind::Dblp
                && workload.scale != "v12"
            {
                return Err(FixturePreflightError::selection(
                    "validation requires the canonical DBLP V12 fixture, not a multi-graph slice",
                )
                .into());
            }
            if !workload.path.is_dir() {
                return Err(FixturePreflightError::missing(workload.kind.label(), &workload.path).into());
            }
            preflighted.push(Self::collect_one(workload, dblp_contract)?);
        }
        if policy == FixtureSelectionPolicy::BothKindsExactly
            && kinds != BTreeSet::from([FixtureKind::Tpch, FixtureKind::Dblp])
        {
            return Err(FixturePreflightError::selection("validation requires both tpch and dblp fixture kinds").into());
        }
        Ok(preflighted)
    }

    fn collect_one(
        workload: &FixtureWorkloadProfile,
        dblp_contract: DblpFixtureContract,
    ) -> ToolResult<FixturePreflightArtifact> {
        match workload.kind {
            FixtureKind::Tpch | FixtureKind::Tpcds | FixtureKind::Clickbench => {
                let (dataset_metadata, content_fingerprint) = match workload.kind {
                    FixtureKind::Tpch => {
                        let metadata = describe_tpch_dataset_with_content_fingerprint(
                            &workload.path,
                            &workload.scale,
                            "p6_fixture_preflight",
                            &workload.preparation_command,
                            true,
                        )
                        .map_err(|error| {
                            FixturePreflightError::invalid_layout("tpch", &workload.path, error.to_string())
                        })?;
                        (metadata.to_neutral_metadata(), metadata.content_fingerprint)
                    },
                    FixtureKind::Tpcds => {
                        let metadata = describe_tpcds_dataset_with_content_fingerprint(
                            &workload.path,
                            &workload.scale,
                            "p6_fixture_preflight",
                            &workload.preparation_command,
                            true,
                        )
                        .map_err(|error| {
                            FixturePreflightError::invalid_layout("tpcds", &workload.path, error.to_string())
                        })?;
                        (metadata.to_neutral_metadata(), metadata.content_fingerprint)
                    },
                    FixtureKind::Clickbench => {
                        let metadata = describe_clickbench_dataset_with_content_fingerprint(
                            &workload.path,
                            "p6_fixture_preflight",
                            &workload.preparation_command,
                            true,
                        )
                        .map_err(|error| {
                            FixturePreflightError::invalid_layout("clickbench", &workload.path, error.to_string())
                        })?;
                        (metadata.to_neutral_metadata(), metadata.content_fingerprint)
                    },
                    FixtureKind::Dblp => unreachable!("dblp is handled by its own preflight arm"),
                };
                let content_fingerprint = content_fingerprint.ok_or_else(|| {
                    FixturePreflightError::invalid_layout(
                        workload.kind.label(),
                        &workload.path,
                        "content fingerprint was not produced",
                    )
                })?;
                Ok(FixturePreflightArtifact {
                    kind: workload.kind,
                    path: workload.path.display().to_string(),
                    scale: workload.scale.clone(),
                    preparation_command: workload.preparation_command.clone(),
                    layout_validation: "complete",
                    fingerprints: vec![
                        FixtureFingerprint::fnv1a64("metadata", dataset_metadata.metadata_fingerprint.clone()),
                        FixtureFingerprint::fnv1a64("content", content_fingerprint),
                    ],
                    dataset_metadata,
                })
            },
            FixtureKind::Dblp => {
                let metadata = describe_dblp_for_contract(workload, dblp_contract).map_err(|error| {
                    FixturePreflightError::invalid_layout("dblp", &workload.path, error.to_string())
                })?;
                let mut fingerprints = vec![
                    FixtureFingerprint::sha256("manifest", metadata.manifest_sha256.clone()),
                    FixtureFingerprint::sha256("source_metadata", metadata.source_metadata_sha256.clone()),
                    FixtureFingerprint::sha256("source_content", metadata.source_content_sha256.clone()),
                ];
                if let Some(slice) = &metadata.multi_graph_slice {
                    fingerprints.push(FixtureFingerprint::sha256("edge_rowset", slice.edge_rowset_sha256.clone()));
                }
                Ok(FixturePreflightArtifact {
                    kind: FixtureKind::Dblp,
                    path: workload.path.display().to_string(),
                    scale: workload.scale.clone(),
                    preparation_command: workload.preparation_command.clone(),
                    layout_validation: "complete",
                    dataset_metadata: metadata.to_neutral_metadata(),
                    fingerprints,
                })
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DblpFixtureContract {
    CanonicalV12,
    #[cfg(test)]
    Sample,
}

fn describe_dblp_for_contract(
    workload: &FixtureWorkloadProfile,
    contract: DblpFixtureContract,
) -> WorkloadResult<DblpDatasetMetadata> {
    match contract {
        DblpFixtureContract::CanonicalV12 => {
            if workload.scale == "v12" {
                return describe_canonical_dblp_dataset(&workload.path, &workload.scale, &workload.preparation_command);
            }
            if !workload.scale.starts_with("v12_slice_") {
                return Err(workloads::graph::WorkloadError::InvalidDataset(
                    "DBLP fixture scale must be v12 or v12_slice_<index>".to_string(),
                ));
            }
            let metadata = describe_dblp_dataset(&workload.path, &workload.scale, &workload.preparation_command)?;
            let slice = metadata.multi_graph_slice.as_ref().ok_or_else(|| {
                workloads::graph::WorkloadError::InvalidDataset(
                    "DBLP slice profile requires generator-authenticated multi_graph_slice metadata".to_string(),
                )
            })?;
            let expected_scale = format!("v12_slice_{:02}", slice.slice_index);
            if workload.scale != expected_scale {
                return Err(workloads::graph::WorkloadError::InvalidDataset(format!(
                    "DBLP slice profile scale {} does not match fixture slice index {}; expected {expected_scale}",
                    workload.scale, slice.slice_index
                )));
            }
            Ok(metadata)
        },
        #[cfg(test)]
        DblpFixtureContract::Sample => {
            workloads::graph::describe_dblp_dataset(&workload.path, &workload.scale, &workload.preparation_command)
        },
    }
}

#[derive(Debug, Serialize)]
pub(super) struct FixturePreflightArtifact {
    pub(super) kind: FixtureKind,
    pub(super) path: String,
    pub(super) scale: String,
    pub(super) preparation_command: String,
    pub(super) layout_validation: &'static str,
    pub(super) dataset_metadata: WorkloadDatasetMetadata,
    pub(super) fingerprints: Vec<FixtureFingerprint>,
}

#[derive(Debug, Serialize)]
pub(super) struct FixtureFingerprint {
    pub(super) scope: &'static str,
    pub(super) algorithm: &'static str,
    pub(super) cryptographic: bool,
    pub(super) value: String,
}

impl FixtureFingerprint {
    pub(super) fn fnv1a64(scope: &'static str, value: String) -> Self {
        Self { scope, algorithm: "fnv1a64", cryptographic: false, value }
    }

    pub(super) fn sha256(scope: &'static str, value: String) -> Self {
        Self { scope, algorithm: "sha256", cryptographic: true, value }
    }
}
