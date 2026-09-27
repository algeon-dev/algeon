use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    provenance::FileIdentity,
    timeseries::{
        DeviceSeriesRow, FleetUtilizationInterval, FleetUtilizationPolicy, ProcessSeriesRow, fleet_utilization_interval,
    },
    util::{sha256_file, write_new_named_artifact},
};
use crate::{SINGLE_NODE_RUN_ARTIFACT_SCHEMA, ToolResult};

const QUALIFICATION_MANIFEST_SCHEMA: &str = "algeon_tools.single_node_multi_gpu.qualification_manifest";
const QUALIFICATION_REPORT_SCHEMA: &str = "algeon_tools.single_node_multi_gpu.qualification_report";
const QUALIFICATION_REPORT_FILE: &str = "qualification-report.json";
const PRODUCTION_CLAIM: &str = "cache-off production-qualified";
const PRODUCTION_MANIFEST: &[u8] = include_bytes!("../../fixtures/qualification/cache-off-production.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationManifest {
    schema: String,
    target_claim: String,
    cells: Vec<CellManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CellManifest {
    name: String,
    profile_path: PathBuf,
    profile_sha256: String,
    minimum_repeats: usize,
    maximum_sample_gap_ms: u64,
    required_queries: Vec<String>,
    #[serde(default)]
    required_clauses: Vec<String>,
    #[serde(default)]
    required_evidence: Vec<String>,
    #[serde(default)]
    mixed_interval: Option<MixedIntervalManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedIntervalManifest {
    device_ordinals: Vec<usize>,
    minimum_utilization_percent: u32,
    minimum_duration_seconds: u64,
    maximum_sample_gap_ms: u64,
    minimum_active_attempts: usize,
    minimum_completed_attempts: u64,
    minimum_successful_native_attempts: u64,
}

#[derive(Debug, Serialize)]
struct QualificationReport {
    schema: &'static str,
    target_claim: String,
    status: &'static str,
    manifest: FileIdentity,
    full_gate_log: FileIdentity,
    consumed_files: Vec<FileIdentity>,
    source_and_binary_identity: Value,
    cells: Vec<VerifiedCell>,
}

#[derive(Debug, Serialize)]
struct VerifiedCell {
    name: String,
    repeats: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    mixed_interval: Option<FleetUtilizationInterval>,
}

pub(super) fn verify_bundle(
    manifest_path: &Path,
    bundle_dir: &Path,
    gate_log: &Path,
    out_dir: &Path,
) -> ToolResult<PathBuf> {
    let manifest_bytes = fs::read(manifest_path).map_err(|_| "cannot read qualification manifest")?;
    let manifest: QualificationManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.schema != QUALIFICATION_MANIFEST_SCHEMA {
        return Err("unsupported qualification manifest schema".into());
    }
    if manifest.target_claim.is_empty() {
        return Err("qualification manifest has no target claim".into());
    }
    if manifest.cells.is_empty() {
        return Err("qualification manifest has no cells".into());
    }
    if manifest.target_claim == PRODUCTION_CLAIM {
        if manifest_bytes != PRODUCTION_MANIFEST {
            return Err("production qualification requires the built-in committed manifest".into());
        }
        verify_production_mixed_contract(&manifest)?;
    }
    let gate_bytes = fs::read(gate_log).map_err(|_| "cannot read full-gate log")?;
    if !gate_bytes
        .windows(b"OK: canonical full pre-merge surface passed.".len())
        .any(|window| window == b"OK: canonical full pre-merge surface passed.")
    {
        return Err("full-gate log has no canonical PASS footer".into());
    }
    if gate_bytes.split(|byte| *byte == b'\n').any(|line| line.ends_with(b"| HIT")) {
        return Err("full-gate log reused a cached phase".into());
    }
    let gate_source_start = gate_source_identity(&gate_bytes, "CHECK SOURCE START: ")?;
    let gate_source_end = gate_source_identity(&gate_bytes, "CHECK SOURCE END: ")?;
    if gate_source_start != gate_source_end {
        return Err("full-gate source identity changed during the gate".into());
    }
    let manifest_parent = manifest_path.parent().ok_or("qualification manifest has no parent")?;
    let mut failures = Vec::new();
    let mut cell_names = BTreeSet::new();
    for cell in &manifest.cells {
        if cell.name.is_empty() || !cell_names.insert(cell.name.as_str()) {
            failures.push(format!("{} is an empty or duplicate cell name", cell.name));
        }
        if cell.minimum_repeats == 0 {
            failures.push(format!("{} has zero required repeats", cell.name));
        }
        let profile_path = manifest_parent.join(&cell.profile_path);
        match sha256_file(&profile_path) {
            Ok(actual) if actual == cell.profile_sha256 => {},
            Ok(_) => failures.push(format!("{} profile hash mismatch", cell.name)),
            Err(_) => failures.push(format!("{} profile cannot be hashed", cell.name)),
        }
    }
    let artifact_paths = artifact_paths(bundle_dir)?;
    let mut artifacts = Vec::new();
    for path in &artifact_paths {
        let value: Value = serde_json::from_slice(&fs::read(path)?)?;
        if value.get("schema").and_then(Value::as_str) == Some(SINGLE_NODE_RUN_ARTIFACT_SCHEMA) {
            artifacts.push((path, value));
        }
    }
    let mut process_identities = BTreeSet::new();
    let mut common_identity = None;
    let mut cells = Vec::new();
    for cell in &manifest.cells {
        let matching = artifacts
            .iter()
            .filter(|(_, artifact)| {
                artifact.pointer("/profile/sha256").and_then(Value::as_str) == Some(&cell.profile_sha256)
            })
            .collect::<Vec<_>>();
        if matching.len() < cell.minimum_repeats {
            failures.push(format!("{} has {}/{} required repeats", cell.name, matching.len(), cell.minimum_repeats));
        }
        let mut mixed_interval = None;
        let mut cell_fixtures = None;
        for (path, artifact) in &matching {
            verify_artifact(cell, artifact, &mut failures, path);
            let process = artifact.pointer("/provenance/process").cloned().unwrap_or(Value::Null);
            let process_key = serde_json::to_string(&process)?;
            if !process_identities.insert(process_key) {
                failures.push(format!("{} repeats a process identity", path.display()));
            }
            let identity = artifact_identity(artifact);
            if let Some(expected) = &common_identity {
                if expected != &identity {
                    failures.push(format!("{} source or binary identity differs", path.display()));
                }
            } else {
                common_identity = Some(identity.clone());
            }
            let fixtures = artifact.pointer("/execution/fixtures").cloned().unwrap_or(Value::Null);
            if let Some(expected) = &cell_fixtures {
                if expected != &fixtures {
                    failures.push(format!("{} fixture identities differ within the cell", path.display()));
                }
            } else {
                cell_fixtures = Some(fixtures);
            }
            if let Some(policy) = &cell.mixed_interval {
                match reduce_mixed_interval(artifact, policy) {
                    Ok(Some(interval)) => {
                        if mixed_interval
                            .as_ref()
                            .is_none_or(|best: &FleetUtilizationInterval| interval.duration_ns > best.duration_ns)
                        {
                            mixed_interval = Some(interval);
                        }
                    },
                    Ok(None) => failures.push(format!("{} has no qualifying mixed interval", path.display())),
                    Err(error) => failures.push(format!("{} mixed interval: {error}", path.display())),
                }
            }
        }
        cells.push(VerifiedCell { name: cell.name.clone(), repeats: matching.len(), mixed_interval });
    }
    if let Some(identity) = &common_identity {
        for (repository, gate_key) in [
            ("root", "root"),
            ("cudf", "gitlink:components/cudf/cudf"),
            ("cugraph", "gitlink:components/cugraph/cugraph"),
            ("cuvs", "gitlink:components/cuvs/cuvs"),
            ("cuml", "gitlink:components/cuml/cuml"),
        ] {
            let revision = gate_source_end.get(gate_key).expect("validated gate identity is complete");
            let pointer = format!("/repositories/{repository}/revision");
            if identity.pointer(&pointer).and_then(Value::as_str) != Some(revision) {
                failures.push(format!("full-gate {repository} revision differs from qualification artifacts"));
            }
        }
    }
    if !failures.is_empty() {
        return Err(format!("qualification bundle rejected:\n{}", failures.join("\n")).into());
    }
    let consumed_files = artifact_paths
        .iter()
        .map(|path| FileIdentity::collect(path, "qualification artifact"))
        .collect::<ToolResult<Vec<_>>>()?;
    let report = QualificationReport {
        schema: QUALIFICATION_REPORT_SCHEMA,
        target_claim: manifest.target_claim,
        status: "evidenced",
        manifest: FileIdentity::collect(manifest_path, "qualification manifest")?,
        full_gate_log: FileIdentity::collect(gate_log, "full-gate log")?,
        consumed_files,
        source_and_binary_identity: common_identity.unwrap_or(Value::Null),
        cells,
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    write_new_named_artifact(out_dir, QUALIFICATION_REPORT_FILE, &bytes)
}

fn verify_production_mixed_contract(manifest: &QualificationManifest) -> ToolResult<()> {
    let valid = manifest.cells.iter().any(|cell| {
        let Some(mixed) = &cell.mixed_interval else { return false };
        mixed.device_ordinals.iter().copied().collect::<BTreeSet<_>>().len() == 4
            && mixed.device_ordinals.len() == 4
            && mixed.minimum_utilization_percent >= 80
            && mixed.minimum_duration_seconds >= 60
            && mixed.maximum_sample_gap_ms > 0
            && mixed.minimum_active_attempts > 0
            && mixed.minimum_completed_attempts > 0
            && mixed.minimum_successful_native_attempts > 0
    });
    if !valid {
        return Err("production qualification manifest has no mandatory four-device mixed contract".into());
    }
    Ok(())
}

fn gate_source_identity(bytes: &[u8], prefix: &str) -> ToolResult<BTreeMap<String, String>> {
    let text = std::str::from_utf8(bytes).map_err(|_| "full-gate log is not UTF-8")?;
    let rows = text.lines().filter_map(|line| line.strip_prefix(prefix)).collect::<Vec<_>>();
    if rows.len() != 1 {
        return Err(format!("full-gate log requires exactly one {prefix}identity").into());
    }
    let mut identity = BTreeMap::new();
    let mut fields = rows[0].split_whitespace();
    let root = fields.next().ok_or("full-gate source identity has no root")?;
    let gitlinks = fields.next().ok_or("full-gate source identity has no gitlinks")?;
    if fields.next().is_some() {
        return Err("full-gate source identity has unexpected fields".into());
    }
    let (name, value) = root.split_once('=').ok_or("invalid full-gate root identity")?;
    let (revision, state) = value.rsplit_once(':').ok_or("invalid full-gate root identity")?;
    if name != "root"
        || revision.len() != 40
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        || state != "clean"
    {
        return Err("invalid or dirty full-gate root identity".into());
    }
    identity.insert(name.to_string(), revision.to_string());

    let (name, value) = gitlinks.split_once('=').ok_or("invalid full-gate gitlink identity")?;
    if name != "gitlinks" {
        return Err("invalid full-gate gitlink identity".into());
    }
    let expected_paths =
        ["components/cudf/cudf", "components/cugraph/cugraph", "components/cuvs/cuvs", "components/cuml/cuml"];
    for entry in value.split(',') {
        let (revision, path) = entry
            .strip_prefix('=')
            .and_then(|entry| entry.split_once('@'))
            .ok_or("invalid or dirty full-gate gitlink identity")?;
        if revision.len() != 40
            || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
            || !expected_paths.contains(&path)
            || identity.insert(format!("gitlink:{path}"), revision.to_string()).is_some()
        {
            return Err("invalid or dirty full-gate gitlink identity".into());
        }
    }
    if identity.len() != expected_paths.len() + 1 {
        return Err("full-gate source identity is incomplete".into());
    }
    Ok(identity)
}

fn artifact_paths(root: &Path) -> ToolResult<Vec<PathBuf>> {
    let mut pending = vec![root.to_path_buf()];
    let mut paths = Vec::new();
    while let Some(path) = pending.pop() {
        let metadata = fs::metadata(&path).map_err(|_| format!("cannot stat bundle path {}", path.display()))?;
        if metadata.is_dir() {
            for entry in fs::read_dir(&path)? {
                pending.push(entry?.path());
            }
        } else if path.extension().is_some_and(|extension| extension == "json") {
            paths.push(path);
        }
    }
    paths.sort_unstable();
    Ok(paths)
}

fn verify_artifact(cell: &CellManifest, artifact: &Value, failures: &mut Vec<String>, path: &Path) {
    if artifact.get("status").and_then(Value::as_str) != Some("measured_run")
        || artifact.pointer("/execution/measurement/qualifying").and_then(Value::as_bool) != Some(true)
    {
        failures.push(format!("{} is not a qualifying measured run", path.display()));
    }
    if artifact
        .pointer("/execution/qualification_failures")
        .and_then(Value::as_array)
        .is_none_or(|rows| !rows.is_empty())
    {
        failures.push(format!("{} has qualification failures", path.display()));
    }
    let maximum_gap_ns = cell.maximum_sample_gap_ms.checked_mul(1_000_000);
    let observed_gap_ns = artifact
        .pointer("/execution/measurement/time_series/maximum_cuda_nvml_sample_gap/nanoseconds")
        .and_then(Value::as_u64);
    if maximum_gap_ns.is_none()
        || observed_gap_ns.zip(maximum_gap_ns).is_none_or(|(observed, maximum)| observed > maximum)
    {
        failures.push(format!("{} exceeds or omits its physical sample-gap bound", path.display()));
    }
    for repository in ["root", "cudf", "cugraph", "cuvs", "cuml"] {
        let pointer = format!("/provenance/repositories/{repository}/dirty_manifest/entries");
        if artifact.pointer(&pointer).and_then(Value::as_array).is_none_or(|entries| !entries.is_empty()) {
            failures.push(format!("{} has dirty {repository} source", path.display()));
        }
    }
    for pointer in [
        "/provenance/repositories/root/revision",
        "/provenance/repositories/cudf/revision",
        "/provenance/repositories/cugraph/revision",
        "/provenance/repositories/cuvs/revision",
        "/provenance/repositories/cuml/revision",
        "/provenance/cuml_build_inputs/cmake_config/sha256",
        "/provenance/cuml_build_inputs/libcuml_selector/sha256",
        "/provenance/cuml_build_inputs/libcuml_resolved/sha256",
        "/provenance/cuml_build_inputs/libcuml_rust_selector/sha256",
        "/provenance/cuml_build_inputs/libcuml_rust_resolved/sha256",
        "/provenance/cuml_build_inputs/version",
        "/provenance/native_libraries/libcudf_resolved/sha256",
        "/provenance/native_libraries/libcugraph_resolved/sha256",
        "/provenance/runner/executable/sha256",
        "/provenance/process/boot_id",
    ] {
        if artifact.pointer(pointer).and_then(Value::as_str).is_none_or(str::is_empty) {
            failures.push(format!("{} is missing identity {pointer}", path.display()));
        }
    }
    for pointer in ["/provenance/process/pid", "/provenance/process/process_start_ticks"] {
        if artifact.pointer(pointer).and_then(Value::as_u64).is_none() {
            failures.push(format!("{} is missing identity {pointer}", path.display()));
        }
    }
    let selected = artifact
        .pointer("/execution/measurement/selected_queries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|query| query.get("label").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    for query in &cell.required_queries {
        if !selected.contains(query.as_str()) {
            failures.push(format!("{} is missing query {query}", path.display()));
        }
    }
    for clause in &cell.required_clauses {
        if !clause_statuses(artifact, clause).contains(&"evidenced") {
            failures.push(format!("{} is missing evidenced clause {clause}", path.display()));
        }
    }
    for pointer in &cell.required_evidence {
        if !pointer.starts_with("/execution/measurement/") || artifact.pointer(pointer).is_none_or(Value::is_null) {
            failures.push(format!("{} is missing required evidence {pointer}", path.display()));
        }
    }
    if contains_failed_status(artifact) {
        failures.push(format!("{} contains a failed clause", path.display()));
    }
    verify_allocator_bounds(artifact, failures, path);
    let mut allocation_counts = std::collections::BTreeMap::<u64, Vec<u64>>::new();
    for row in artifact
        .pointer("/execution/measurement/time_series/device_rows")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let (Some(ordinal), Some(count)) = (
            row.get("device_ordinal").and_then(Value::as_u64),
            row.get("allocation_failure_count").and_then(Value::as_u64),
        ) {
            allocation_counts.entry(ordinal).or_default().push(count);
        }
        let live_grants = row.get("live_grants_bytes").and_then(Value::as_u64);
        let resident_owner = row.get("resident_owner_bytes").and_then(Value::as_u64);
        let managed_capacity = row.get("managed_capacity_bytes").and_then(Value::as_u64);
        if live_grants
            .zip(resident_owner)
            .zip(managed_capacity)
            .is_none_or(|((grants, owners), capacity)| owners > grants || grants > capacity)
        {
            failures.push(format!("{} contains an invalid ledger byte snapshot", path.display()));
        }
    }
    if allocation_counts.values().any(|counts| counts.iter().any(|count| *count != 0)) {
        failures.push(format!("{} recorded an allocator safety delta", path.display()));
    }
}

fn verify_allocator_bounds(artifact: &Value, failures: &mut Vec<String>, path: &Path) {
    let attempts =
        artifact.pointer("/execution/measurement/aggregate/attempts").and_then(Value::as_array).into_iter().flatten();
    for attempt in attempts {
        let resource_evidence = attempt.pointer("/terminal_evidence/resource_evidence");
        if resource_evidence.is_none_or(Value::is_null) {
            if attempt.get("device_ordinal").is_some_and(|ordinal| !ordinal.is_null())
                || attempt.get("grant_bytes").is_some_and(|grant| !grant.is_null())
            {
                failures.push(format!("{} admitted attempt has no resource evidence", path.display()));
            }
            continue;
        }
        let Some(grant) =
            attempt.pointer("/terminal_evidence/resource_evidence/grant_cap_bytes/value").and_then(Value::as_u64)
        else {
            failures.push(format!("{} attempt has no grant cap", path.display()));
            continue;
        };
        let Some(domain) =
            attempt.pointer("/terminal_evidence/resource_evidence/domain_cap_bytes/value").and_then(Value::as_u64)
        else {
            failures.push(format!("{} attempt has no domain cap", path.display()));
            continue;
        };
        for usage in ["outcome_terminal_usage", "post_cleanup_usage"] {
            let Some(peak) = attempt
                .pointer(&format!("/terminal_evidence/resource_evidence/{usage}/value/usage/peak_aligned_bytes"))
                .and_then(Value::as_u64)
            else {
                failures.push(format!("{} attempt has no {usage} peak", path.display()));
                continue;
            };
            if peak > grant || peak > domain {
                failures.push(format!("{} attempt {usage} peak exceeds its grant", path.display()));
            }
            let pool = format!("/terminal_evidence/resource_evidence/{usage}/value/pool_usage");
            let reserved_current = attempt.pointer(&format!("{pool}/reserved_current_bytes")).and_then(Value::as_u64);
            let reserved_high = attempt.pointer(&format!("{pool}/reserved_high_bytes")).and_then(Value::as_u64);
            let used_current = attempt.pointer(&format!("{pool}/used_current_bytes")).and_then(Value::as_u64);
            let used_high = attempt.pointer(&format!("{pool}/used_high_bytes")).and_then(Value::as_u64);
            let Some(((reserved_current, reserved_high), (used_current, used_high))) =
                reserved_current.zip(reserved_high).zip(used_current.zip(used_high))
            else {
                failures.push(format!("{} attempt has no {usage} query-domain pool usage", path.display()));
                continue;
            };
            if reserved_current > reserved_high || used_current > used_high || used_current > reserved_current {
                failures.push(format!("{} attempt {usage} has invalid query-domain pool counters", path.display()));
            }
            if used_high > reserved_high {
                failures
                    .push(format!("{} attempt {usage} pool high-water usage exceeds reserved bytes", path.display()));
            }
        }
    }
}

fn artifact_identity(artifact: &Value) -> Value {
    serde_json::json!({
        "repositories": artifact.pointer("/provenance/repositories"),
        "native_libraries": artifact.pointer("/provenance/native_libraries"),
        "cuml_build_inputs": artifact.pointer("/provenance/cuml_build_inputs"),
        "runner": artifact.pointer("/provenance/runner"),
    })
}

fn clause_statuses<'a>(value: &'a Value, clause: &str) -> Vec<&'a str> {
    let mut statuses = Vec::new();
    match value {
        Value::Array(values) => values.iter().for_each(|value| statuses.extend(clause_statuses(value, clause))),
        Value::Object(object) => {
            if object.get("clause").and_then(Value::as_str) == Some(clause)
                && let Some(status) = object.get("status").and_then(Value::as_str)
            {
                statuses.push(status);
            }
            for value in object.values() {
                statuses.extend(clause_statuses(value, clause));
            }
        },
        _ => {},
    }
    statuses
}

fn contains_failed_status(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(contains_failed_status),
        Value::Object(object) => {
            object.get("status").and_then(Value::as_str) == Some("failed")
                || object.values().any(contains_failed_status)
        },
        _ => false,
    }
}

fn reduce_mixed_interval(
    artifact: &Value,
    policy: &MixedIntervalManifest,
) -> ToolResult<Option<FleetUtilizationInterval>> {
    let device_rows: Vec<DeviceSeriesRow> = serde_json::from_value(
        artifact.pointer("/execution/measurement/time_series/device_rows").cloned().ok_or("missing device rows")?,
    )?;
    let process_rows: Vec<ProcessSeriesRow> = serde_json::from_value(
        artifact.pointer("/execution/measurement/time_series/process_rows").cloned().ok_or("missing process rows")?,
    )?;
    let policy = FleetUtilizationPolicy {
        device_ordinals: policy.device_ordinals.clone(),
        minimum_utilization_percent: policy.minimum_utilization_percent,
        minimum_duration_ns: policy.minimum_duration_seconds.checked_mul(1_000_000_000).ok_or("duration overflow")?,
        maximum_sample_gap_ns: policy.maximum_sample_gap_ms.checked_mul(1_000_000).ok_or("sample gap overflow")?,
        minimum_active_attempts: policy.minimum_active_attempts,
        minimum_completed_attempts: policy.minimum_completed_attempts,
        minimum_successful_native_attempts: policy.minimum_successful_native_attempts,
    };
    fleet_utilization_interval(&device_rows, &process_rows, &policy).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::single_node_multi_gpu_validate::util::sha256_bytes;

    #[test]
    fn verifier_accepts_distinct_source_bound_repeats_and_writes_content_digests() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("algeon-qualification-verifier-{nonce}"));
        let bundle = root.join("bundle");
        fs::create_dir_all(&bundle).unwrap();
        let profile = root.join("profile.json");
        fs::write(&profile, b"{}").unwrap();
        let profile_sha256 = sha256_bytes(b"{}");
        let manifest = root.join("manifest.json");
        fs::write(
            &manifest,
            serde_json::to_vec(&serde_json::json!({
                "schema": QUALIFICATION_MANIFEST_SCHEMA,
                "target_claim": "test",
                "cells": [{
                    "name": "repeat",
                    "profile_path": "profile.json",
                    "profile_sha256": profile_sha256,
                    "minimum_repeats": 2,
                    "maximum_sample_gap_ms": 1,
                    "required_queries": ["pagerank_topk"]
                }]
            }))
            .unwrap(),
        )
        .unwrap();
        for pid in [10, 11] {
            let artifact = qualifying_artifact(&profile_sha256, pid);
            fs::write(bundle.join(format!("run-{pid}.json")), serde_json::to_vec(&artifact).unwrap()).unwrap();
        }
        let gate = root.join("check-all.log");
        let revision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        fs::write(&gate, gate_log(revision)).unwrap();
        let report_dir = root.join("report");
        let report = verify_bundle(&manifest, &bundle, &gate, &report_dir).unwrap();
        let report: Value = serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
        assert_eq!(report["status"], "evidenced");
        assert_eq!(report["consumed_files"].as_array().unwrap().len(), 2);

        let other_revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        fs::write(&gate, gate_log_with_cuvs_revision(revision, other_revision)).unwrap();
        assert!(
            verify_bundle(&manifest, &bundle, &gate, &root.join("mismatched-cuvs-report"))
                .unwrap_err()
                .to_string()
                .contains("cuvs revision differs")
        );

        fs::write(&gate, gate_log_with_cuml_revision(revision, other_revision)).unwrap();
        assert!(
            verify_bundle(&manifest, &bundle, &gate, &root.join("mismatched-cuml-report"))
                .unwrap_err()
                .to_string()
                .contains("cuml revision differs")
        );

        let artifact_path = bundle.join("run-10.json");
        let mut artifact: Value = serde_json::from_slice(&fs::read(&artifact_path).unwrap()).unwrap();
        artifact["provenance"].as_object_mut().unwrap().remove("cuml_build_inputs");
        fs::write(&artifact_path, serde_json::to_vec(&artifact).unwrap()).unwrap();
        fs::write(&gate, gate_log(revision)).unwrap();
        assert!(
            verify_bundle(&manifest, &bundle, &gate, &root.join("missing-cuml-report"))
                .unwrap_err()
                .to_string()
                .contains("missing identity /provenance/cuml_build_inputs/")
        );
        fs::write(&artifact_path, serde_json::to_vec(&qualifying_artifact(&profile_sha256, 10)).unwrap()).unwrap();

        fs::write(&gate, gate_log(other_revision)).unwrap();
        assert!(
            verify_bundle(&manifest, &bundle, &gate, &root.join("mismatched-report"))
                .unwrap_err()
                .to_string()
                .contains("revision differs")
        );

        fs::write(
            &gate,
            b"phase | command | cache\nfoo | cargo test | HIT\nOK: canonical full pre-merge surface passed.\n",
        )
        .unwrap();
        assert_eq!(
            verify_bundle(&manifest, &bundle, &gate, &root.join("cached-report")).unwrap_err().to_string(),
            "full-gate log reused a cached phase"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn built_in_production_manifest_pins_the_mandatory_mixed_contract() {
        let manifest: QualificationManifest = serde_json::from_slice(PRODUCTION_MANIFEST).unwrap();
        assert_eq!(manifest.target_claim, PRODUCTION_CLAIM);
        verify_production_mixed_contract(&manifest).unwrap();
    }

    #[test]
    fn built_in_manifest_accepts_the_committed_dry_run_artifact_shape() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("algeon-production-qualification-verifier-{nonce}"));
        let bundle = root.join("bundle");
        fs::create_dir_all(&bundle).unwrap();
        let manifest_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/qualification/cache-off-production.json");
        let manifest: QualificationManifest = serde_json::from_slice(PRODUCTION_MANIFEST).unwrap();
        let mut pid = 100_u32;
        for cell in &manifest.cells {
            for repeat in 0..cell.minimum_repeats {
                let mut artifact = qualifying_artifact(&cell.profile_sha256, pid);
                pid += 1;
                artifact["execution"]["measurement"]["selected_queries"] = Value::Array(
                    cell.required_queries.iter().map(|label| serde_json::json!({"label": label})).collect(),
                );
                artifact["execution"]["measurement"]["aggregate"]["scenarios"] = serde_json::json!({
                    "required": cell.required_clauses.iter().map(|clause| {
                        serde_json::json!({"clause": clause, "status": "evidenced"})
                    }).collect::<Vec<_>>()
                });
                match cell.name.as_str() {
                    "overload-fairness" => {
                        artifact["execution"]["measurement"]["aggregate"]["scenarios"]["overload_fairness"] =
                            serde_json::json!({"status": "evidenced"});
                    },
                    "device-quarantine" => {
                        artifact["execution"]["measurement"]["aggregate"]["quarantine_fallback"] =
                            serde_json::json!({"evidenced": true});
                    },
                    "after-admission-deadline" => {
                        artifact["execution"]["measurement"]["deadline_barrier"] =
                            serde_json::json!({"evidenced": true});
                    },
                    "caller-cancellation" => {
                        artifact["execution"]["measurement"]["cancellation_barrier"] =
                            serde_json::json!({"evidenced": true});
                    },
                    "four-gpu-mixed" => add_mixed_series(&mut artifact),
                    _ => {},
                }
                fs::write(bundle.join(format!("{}-{repeat}.json", cell.name)), serde_json::to_vec(&artifact).unwrap())
                    .unwrap();
            }
        }
        let gate = root.join("check-all.log");
        fs::write(&gate, gate_log("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")).unwrap();
        verify_bundle(&manifest_path, &bundle, &gate, &root.join("report")).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn verifier_rejects_failed_statuses_and_scopes_allocator_bounds_to_owned_evidence() {
        assert!(contains_failed_status(&serde_json::json!({"status": "failed"})));

        let path = Path::new("artifact.json");
        let mut artifact = serde_json::json!({
            "execution": {"measurement": {"aggregate": {"attempts": [{
                "device_ordinal": null,
                "grant_bytes": null,
                "terminal_evidence": {"capacity_disposition": "not_admitted"}
            }]}}}
        });
        let mut failures = Vec::new();
        verify_allocator_bounds(&artifact, &mut failures, path);
        assert!(failures.is_empty());

        artifact["execution"]["measurement"]["aggregate"]["attempts"][0]["device_ordinal"] = Value::from(0);
        verify_allocator_bounds(&artifact, &mut failures, path);
        assert_eq!(failures, ["artifact.json admitted attempt has no resource evidence"]);
    }

    fn qualifying_artifact(profile_sha256: &str, pid: u32) -> Value {
        let mut artifact: Value =
            serde_json::from_str(include_str!("../../fixtures/qualification/dry-run/run-artifact-template.json"))
                .unwrap();
        artifact["profile"]["sha256"] = Value::String(profile_sha256.to_string());
        artifact["provenance"]["process"]["pid"] = Value::from(pid);
        artifact["provenance"]["process"]["process_start_ticks"] = Value::from(pid);
        artifact
    }

    fn gate_log(revision: &str) -> String {
        gate_log_with_revisions(revision, revision, revision)
    }

    fn gate_log_with_cuvs_revision(revision: &str, cuvs_revision: &str) -> String {
        gate_log_with_revisions(revision, cuvs_revision, revision)
    }

    fn gate_log_with_cuml_revision(revision: &str, cuml_revision: &str) -> String {
        gate_log_with_revisions(revision, revision, cuml_revision)
    }

    fn gate_log_with_revisions(revision: &str, cuvs_revision: &str, cuml_revision: &str) -> String {
        format!(
            "CHECK SOURCE START: root={revision}:clean gitlinks=={revision}@components/cudf/cudf,={revision}@components/cugraph/cugraph,={cuvs_revision}@components/cuvs/cuvs,={cuml_revision}@components/cuml/cuml\n\
             CHECK SOURCE END: root={revision}:clean gitlinks=={revision}@components/cudf/cudf,={revision}@components/cugraph/cugraph,={cuvs_revision}@components/cuvs/cuvs,={cuml_revision}@components/cuml/cuml\n\
             OK: canonical full pre-merge surface passed.\n"
        )
    }

    fn add_mixed_series(artifact: &mut Value) {
        let mut device_rows = Vec::new();
        let mut process_rows = Vec::new();
        for second in 0..=60_u64 {
            let offset_ns = second * 1_000_000_000;
            for device_ordinal in 0..4 {
                device_rows.push(serde_json::json!({
                    "offset_ns": offset_ns, "device_ordinal": device_ordinal, "health_open": true,
                    "gpu_utilization_pct": 80, "memory_utilization_pct": 80,
                    "cuda_free_bytes": 20, "nvml_used_bytes": 80, "active_attempts": 1,
                    "managed_capacity_bytes": 100, "live_grants_bytes": 80,
                    "live_grant_ids": [device_ordinal + 1], "resident_owner_bytes": 40,
                    "cache_evictable_resident_bytes": 0, "cache_pinned_in_use_bytes": 0,
                    "cache_physical_current_bytes": null, "cache_physical_peak_bytes": null,
                    "allocation_failure_count": 0, "ledger_sequence": second,
                    "claim_snapshot_end_sequence": second, "claim_snapshot_stable": true,
                    "active_query_jobs": 1, "queued_jobs": 0
                }));
            }
            process_rows.push(serde_json::json!({
                "offset_ns": offset_ns, "active_queries": 4, "queued_queries": 0,
                "backend_generation": 1, "metric_generation": second, "observation_pull_ns": 1,
                "completed_attempts": second, "successful_native_attempts": second
            }));
        }
        artifact["execution"]["measurement"]["time_series"]["device_rows"] = Value::Array(device_rows);
        artifact["execution"]["measurement"]["time_series"]["process_rows"] = Value::Array(process_rows);
    }
}
