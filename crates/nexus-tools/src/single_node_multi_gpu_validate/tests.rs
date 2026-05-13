use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Float64Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{
    arrow::ArrowWriter,
    file::{metadata::KeyValue, properties::WriterProperties},
};
use serde_json::Value;

use super::{
    aggregate::{
        AttemptObservation, AttemptTerminalEvidence, ClauseStatus, DeviceConfiguration, DeviceOccupancyPeaks,
        MeasuredAggregate, OverloadFairnessEvidence, scenario_clauses_pass,
    },
    metrics_audit::{MetricSnapshotObservation, MetricsAudit},
    profile::{
        ArrivalMode, CacheMode, DeviceOwnership, DeviceProfile, FixtureKind, FixtureWorkloadProfile,
        GraphClaimCalibrationProfile, InstallationBounds, RateOverrunPolicy, SchedulingProfile, VramCalibrationProfile,
    },
    provenance::{
        CmakePackageVersion, CuvsBuildInputs, DirtyManifest, FileIdentity, NativeLibraryIdentities, ProcessIdentity,
        RapidsVersions, RepositoryProvenance, RepositoryProvenanceSet, RunnerIdentity, SourcePackageVersion,
        ToolchainVersions,
    },
    redact::ensure_redacted_json,
    schedule::{ArrivalPlan, SchedulePhase},
    timeseries::DeviceSeriesRow,
    util::{output_parent, write_new_artifact},
    workload::{SelectedQueryDescriptor, select_queries},
    *,
};

fn valid_profile_json() -> &'static str {
    include_str!("../../fixtures/p6-dry-run-profile.json")
}

fn overload_fairness_profile_json() -> &'static str {
    include_str!("../../fixtures/p6-overload-fairness-profile.json")
}

#[test]
fn parser_accepts_exact_dry_run_contract() {
    let options = CliOptions::parse(vec![
        "--config".to_string(),
        "run-profile.json".to_string(),
        "--out-dir".to_string(),
        "run-artifacts".to_string(),
        "--dry-run".to_string(),
    ])
    .unwrap();
    assert_eq!(options.config, Some(PathBuf::from("run-profile.json")));
    assert_eq!(options.out_dir, PathBuf::from("run-artifacts"));
    assert_eq!(options.mode, ValidationMode::DryRun);

    let preflight = CliOptions::parse(vec![
        "--config".to_string(),
        "run-profile.json".to_string(),
        "--out-dir".to_string(),
        "run-artifacts".to_string(),
        "--fixture-preflight".to_string(),
    ])
    .unwrap();
    assert_eq!(preflight.mode, ValidationMode::FixturePreflight);
}

#[test]
fn parser_accepts_bundle_verification_without_a_run_profile() {
    let options = CliOptions::parse(
        [
            "--manifest",
            "qualification.json",
            "--bundle-dir",
            "bundle",
            "--gate-log",
            "check-all.log",
            "--out-dir",
            "report",
            "--verify-bundle",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap();
    assert_eq!(options.mode, ValidationMode::VerifyBundle);
    assert_eq!(options.config, None);
    assert_eq!(options.manifest, Some(PathBuf::from("qualification.json")));
}

#[test]
fn qualification_profiles_parse_and_manifest_hashes_are_committed() {
    for source in [
        include_str!("../../fixtures/qualification/pagerank-topk.json"),
        include_str!("../../fixtures/qualification/bfs-topk.json"),
        include_str!("../../fixtures/qualification/four-gpu-mixed.json"),
    ] {
        let profile = RunProfile::parse_bytes(source.as_bytes()).unwrap();
        profile.validate().unwrap();
        select_queries(&profile.workloads).unwrap();
        assert_eq!(profile.cache.mode, CacheMode::Disabled);
    }
    let manifest: Value =
        serde_json::from_str(include_str!("../../fixtures/qualification/cache-off-production.json")).unwrap();
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/qualification");
    for cell in manifest["cells"].as_array().unwrap() {
        let path = manifest_dir.join(cell["profile_path"].as_str().unwrap());
        assert_eq!(super::util::sha256_file(&path).unwrap(), cell["profile_sha256"].as_str().unwrap());
    }
}

#[test]
fn parser_rejects_duplicate_unknown_missing_and_non_dry_run_options() {
    for (args, expected) in [
        (vec!["--config", "a", "--config", "b", "--out-dir", "out", "--dry-run"], "duplicate option --config"),
        (vec!["--config", "a", "--out-dir", "out", "--unexpected", "--dry-run"], "unknown option --unexpected"),
        (
            vec!["--config", "a", "--out-dir", "out", "--dry-run", "--debug-capture-fingerprints"],
            "unknown option --debug-capture-fingerprints",
        ),
        (
            vec!["--config", "a", "--out-dir", "out", "--dry-run", "--generate-baselines"],
            "unknown option --generate-baselines",
        ),
        (vec!["--out-dir", "out", "--dry-run"], "missing required option --config"),
        (vec!["--config", "a", "--out-dir", "out"], "missing required validation mode"),
        (vec!["--config", "a", "--out-dir", "out", "--dry-run", "--fixture-preflight"], "mutually exclusive"),
        (vec!["--config", "a", "--out-dir", "out", "--measured-run", "--dry-run"], "mutually exclusive"),
        (
            vec!["--config", "a", "--out-dir", "out", "--dry-run", "--nsight-trace", "t"],
            "only valid with --measured-run",
        ),
        (
            vec!["--config", "a", "--out-dir", "out", "--dry-run", "--overload-fairness"],
            "--overload-fairness is only valid with --measured-run",
        ),
        (
            vec![
                "--config",
                "a",
                "--out-dir",
                "out",
                "--measured-run",
                "--overload-fairness",
                "--domain-exhaustion-probe",
            ],
            "--domain-exhaustion-probe cannot be combined with --overload-fairness",
        ),
        (
            vec![
                "--config",
                "a",
                "--out-dir",
                "out",
                "--measured-run",
                "--quarantine-cycles",
                "2",
                "--close-under-load",
            ],
            "--quarantine-cycles cannot be combined with --close-under-load",
        ),
        (
            vec!["--config", "a", "--out-dir", "out", "--measured-run", "--quarantine-cycles", "0"],
            "--quarantine-cycles must be greater than zero",
        ),
        (
            vec!["--config", "a", "--out-dir", "out", "--measured-run", "--quarantine-cycles"],
            "missing value for --quarantine-cycles",
        ),
        (
            vec!["--config", "a", "--out-dir", "out", "--dry-run", "--close-under-load"],
            "--close-under-load is only valid with --measured-run",
        ),
        (
            vec!["--config", "a", "--out-dir", "out", "--dry-run", "--slow-report-sink-ms", "10"],
            "--slow-report-sink-ms is only valid with --measured-run",
        ),
        (
            vec!["--config", "a", "--out-dir", "out", "--measured-run", "--slow-report-sink-ms", "0"],
            "--slow-report-sink-ms must be greater than zero",
        ),
        (
            vec!["--config", "a", "--out-dir", "out", "--measured-run", "--otlp-export-interval-ms", "100"],
            "--otlp-export-interval-ms requires --otlp-metrics-endpoint",
        ),
        (vec!["--config", "--out-dir", "out", "--dry-run"], "missing value for --config"),
    ] {
        let error = CliOptions::parse(args.into_iter().map(str::to_string).collect()).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn parser_records_explicit_otlp_cell_without_making_it_a_scenario() {
    let options = CliOptions::parse(
        [
            "--config",
            "run-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--otlp-metrics-endpoint",
            "http://collector.example.test:4317",
            "--otlp-export-interval-ms",
            "750",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap();
    assert_eq!(options.otlp_metrics_endpoint.as_deref(), Some("http://collector.example.test:4317"));
    assert_eq!(options.otlp_export_interval_ms, Some(750));
}

#[test]
fn parser_records_the_dedicated_completion_sink_probe() {
    let options = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--measured-run", "--slow-report-sink-ms", "25"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    assert_eq!(options.slow_report_sink, Some(std::time::Duration::from_millis(25)));
}

#[test]
fn overload_fairness_profile_is_a_finite_bounded_cohort() {
    let profile = RunProfile::parse_bytes(overload_fairness_profile_json().as_bytes()).unwrap();
    profile.validate_overload_fairness_scenario().unwrap();
    assert_eq!(profile.scheduling.max_rounds, Some(1));
    assert_eq!(profile.scheduling.warmup_seconds, 0);
    assert_eq!(profile.scheduling.measurement_seconds, 120);
    assert_eq!(profile.scheduling.concurrency, 8);
    assert!(profile.backend.device_profiles.iter().all(|device| device.attempt_floor_bytes == 1_073_741_824));

    let mut no_overload = profile.clone();
    no_overload.scheduling.concurrency = 4;
    let error = no_overload.validate_overload_fairness_scenario().unwrap_err();
    assert!(error.to_string().contains("above configured active-attempt capacity"), "{error}");

    let mut unbounded = profile.clone();
    unbounded.scheduling.max_rounds = None;
    let error = unbounded.validate_overload_fairness_scenario().unwrap_err();
    assert!(error.to_string().contains("max_rounds=1"), "{error}");

    let options = CliOptions::parse(
        [
            "--config",
            "p6-overload-fairness-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--overload-fairness",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap();
    assert!(options.overload_fairness);
}

#[test]
fn profile_v1_defaults_preserve_complete_runner_sections() {
    let profile = RunProfile::parse_bytes(valid_profile_json().as_bytes()).unwrap();
    assert_eq!(profile.schema, RUN_PROFILE_SCHEMA);
    assert_eq!(profile.version, RUN_PROFILE_VERSION);
    assert_eq!(profile.backend.device_profiles.len(), 0);
    assert_eq!(profile.backend.queue.max_queued_attempts, 64);
    assert_eq!(profile.session.shared_session_count, 1);
    assert_eq!(profile.scheduling.arrival_mode, ArrivalMode::Concurrency);
    assert_eq!(profile.cache.mode, CacheMode::Disabled);
    assert_eq!(profile.completion_drain_grace_seconds, 0);
}

#[test]
fn profile_bounds_completion_drain_grace_without_changing_legacy_defaults() {
    assert_eq!(RunProfile::MAX_COMPLETION_DRAIN_GRACE_SECONDS, 60);
    let mut profile = measured_run_profile();
    profile.completion_drain_grace_seconds = RunProfile::MAX_COMPLETION_DRAIN_GRACE_SECONDS;
    profile.validate().unwrap();

    profile.completion_drain_grace_seconds = RunProfile::MAX_COMPLETION_DRAIN_GRACE_SECONDS + 1;
    let error = profile.validate().unwrap_err().to_string();
    assert!(error.contains("completion_drain_grace_seconds must not exceed"), "{error}");
}

#[test]
fn feature_off_cugraph_profile_requires_a_duration_paced_mixed_relational_cell() {
    let mut profile = measured_run_profile();
    profile.workloads.retain(|workload| workload.kind != FixtureKind::Dblp);
    profile.feature_off_cugraph_probe = Some(super::profile::FeatureOffCugraphProbeProfile {
        max_rejection_latency_ms: 250,
        warmup_probe_attempts: 2,
        measurement_probe_attempts: 8,
        min_relational_gpu_reaching_attempts: 1,
    });
    profile.validate().unwrap();
    let plan =
        ArrivalPlan::plan(&profile.scheduling, profile.selected_query_count(), profile.session.shared_session_count)
            .unwrap();
    assert!(plan.is_duration_paced_concurrency());
    assert!(plan.arrivals.is_empty());

    let mut unbounded = profile.clone();
    unbounded.feature_off_cugraph_probe.as_mut().unwrap().max_rejection_latency_ms =
        RunProfile::MAX_FEATURE_OFF_CUGRAPH_REJECTION_LATENCY_MS + 1;
    let error = unbounded.validate().unwrap_err().to_string();
    assert!(error.contains("max_rejection_latency_ms"), "{error}");

    let mut missing_declared_warmup = profile.clone();
    missing_declared_warmup.feature_off_cugraph_probe.as_mut().unwrap().warmup_probe_attempts = 0;
    let error = missing_declared_warmup.validate().unwrap_err().to_string();
    assert!(error.contains("warmup_probe_attempts"), "{error}");

    let mut capped_cell = profile.clone();
    capped_cell.scheduling.max_rounds = Some(2);
    let error = capped_cell.validate().unwrap_err().to_string();
    assert!(error.contains("duration-paced"), "{error}");

    let mut no_warmup = profile.clone();
    no_warmup.scheduling.warmup_seconds = 0;
    let error = no_warmup.validate().unwrap_err().to_string();
    assert!(error.contains("warmup_seconds"), "{error}");

    let mut no_relational = profile.clone();
    no_relational.workloads.clear();
    let error = no_relational.validate().unwrap_err().to_string();
    assert!(error.contains("requires at least one selected relational"), "{error}");

    let mut extra_graph_workload = profile;
    extra_graph_workload.workloads.push(FixtureWorkloadProfile {
        kind: FixtureKind::Dblp,
        path: PathBuf::from("/nonexistent/dblp"),
        scale: "v12".to_string(),
        preparation_command:
            "fixture/fixture.sh graph dblp-ingest --source-json /data/dblp.v12.json --out-root /data/demo".to_string(),
        queries: vec!["pagerank".to_string()],
    });
    let error = extra_graph_workload.validate().unwrap_err().to_string();
    assert!(error.contains("owns the only graph UDTF request"), "{error}");
}

#[cfg(all(
    feature = "single-node-multi-gpu-validate-run",
    not(feature = "single-node-multi-gpu-validate-run-cugraph")
))]
#[test]
fn normal_cugraph_s1_profile_is_rejected_by_the_base_runner() {
    let profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s1-embedded-sf1.json").as_bytes()).unwrap();
    let queries = select_queries(&profile.workloads).unwrap();
    let error = super::measure::validate_cugraph_runner_mode(&profile, &queries).unwrap_err().to_string();
    assert!(error.contains("single-node-multi-gpu-validate-run-cugraph"), "{error}");
    assert!(error.contains("base single-node-multi-gpu-validate-run build has cugraph disabled"), "{error}");
}

#[test]
fn profile_rejects_schema_version_unknown_and_invalid_values() {
    for (replacement, expected) in [("datafusion_nexus.other.v1", "run profile schema"), ("2", "run profile version")] {
        let profile = if replacement == "2" {
            valid_profile_json().replace("\"version\": 1", "\"version\": 2")
        } else {
            valid_profile_json().replace(RUN_PROFILE_SCHEMA, replacement)
        };
        let error = RunProfile::parse_bytes(profile.as_bytes()).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let unknown = valid_profile_json().replace("\"workloads\": []", "\"unknown\": true, \"workloads\": []");
    assert!(RunProfile::parse_bytes(unknown.as_bytes()).is_err());
    let invalid_rate = valid_profile_json()
        .replace("\"workloads\": []", "\"scheduling\": {\"arrival_mode\": \"rate\"}, \"workloads\": []");
    assert!(RunProfile::parse_bytes(invalid_rate.as_bytes()).unwrap_err().to_string().contains("non-zero"));
}

#[test]
fn profile_redaction_guard_rejects_nested_secret_without_echoing_it() {
    let profile =
        valid_profile_json().replace("\"workloads\": []", "\"workloads\": [{\"api_token\": \"do-not-leak\"}]");
    let error = RunProfile::parse_bytes(profile.as_bytes()).unwrap_err();
    assert!(error.to_string().contains("sensitive field"));
    assert!(!error.to_string().contains("do-not-leak"));
}

#[test]
fn profile_rejects_noncanonical_or_credential_bearing_preparation_commands_without_echoing_them() {
    for command in [
        "fixture/fixture.sh tpch generate --api-token do-not-leak",
        "AWS_SECRET_ACCESS_KEY=do-not-leak fixture/fixture.sh tpch generate --scale 1 --out-root /fixtures",
        "fixture/fixture.sh tpch generate --scale 1 --out-root /fixtures; curl https://example.invalid",
    ] {
        let profile = valid_profile_json().replace(
            "\"workloads\": []",
            &format!(
                "\"workloads\": [{{\"kind\": \"tpch\", \"path\": \"/fixtures/tpch\", \"scale\": \"sf1\", \"preparation_command\": \"{command}\"}}]"
            ),
        );
        let error = RunProfile::parse_bytes(profile.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("credential-free direct fixture command"), "{error}");
        assert!(!error.to_string().contains("do-not-leak"), "{error}");
    }
}

#[test]
fn profile_reader_rejects_missing_files_without_publishing_an_artifact() {
    let path = test_root("missing-profile").join("missing.json");
    let error = RunProfile::read(&path).unwrap_err();
    assert!(error.to_string().contains("cannot read --config"));
}

#[test]
fn cmake_version_parser_uses_exact_variable() {
    let config = b"set(CUVS_VERSION_MAJOR 26)\nset(CUVS_VERSION 26.06.00)\n";
    assert_eq!(super::provenance::cmake_set_value(config, "CUVS_VERSION").as_deref(), Some("26.06.00"));
    assert_eq!(super::provenance::cmake_set_value(config, "MISSING"), None);
}

#[test]
fn artifact_v1_has_explicit_dry_run_and_redaction_sections() {
    let profile = RunProfile::parse_bytes(valid_profile_json().as_bytes()).unwrap();
    let artifact = RunArtifact::dry_run(profile, "a".repeat(64), sample_provenance());
    let value = serde_json::to_value(artifact).unwrap();
    assert_eq!(value["schema"], RUN_ARTIFACT_SCHEMA);
    assert_eq!(value["version"], RUN_ARTIFACT_VERSION);
    assert_eq!(value["status"], "dry_run");
    assert_eq!(value["execution"]["selected_workloads"], 0);
    assert_eq!(value["execution"]["attempted_workloads"], 0);
    assert_eq!(value["execution"]["fixture_preflight"], "not_applicable_p6_4");
    assert_eq!(value["redaction"]["raw_config_persisted"], false);
    ensure_redacted_json(&value, "").unwrap();
}

#[test]
fn fixture_preflight_artifact_records_typed_fnv_and_sha256_fingerprints_for_a_validated_sample() {
    let root = test_root("fixture-preflight");
    fs::create_dir(&root).unwrap();
    let tpch_root = root.join("tpch");
    nexus_workloads::tpch::write_tiny_tpch_parquet_dataset(&tpch_root).unwrap();
    let dblp_root = root.join("citation_network").join("v1");
    write_dblp_fixture(&dblp_root, None);
    let profile = fixture_preflight_profile(&tpch_root, &dblp_root.join("parquet"));
    let out_dir = root.join("run-artifacts");

    let fixtures = FixturePreflight::collect_sample_fixture_for_artifact_test(
        &profile.workloads,
        FixtureSelectionPolicy::BothKindsExactly,
    )
    .unwrap();
    let artifact = RunArtifact::fixture_preflight(profile, "a".repeat(64), fixtures, sample_provenance());
    let artifact_path = write_artifact(&out_dir, &artifact).unwrap();
    let artifact: Value = serde_json::from_slice(&fs::read(artifact_path).unwrap()).unwrap();
    assert_eq!(artifact["status"], "fixture_preflight");
    assert_eq!(artifact["execution"]["fixture_preflight"], "complete");
    assert_eq!(artifact["execution"]["selected_workloads"], 2);
    assert_eq!(artifact["execution"]["attempted_workloads"], 0);
    assert_eq!(artifact["execution"]["fixtures"][0]["layout_validation"], "complete");
    assert_eq!(artifact["execution"]["fixtures"][0]["fingerprints"][0]["algorithm"], "fnv1a64");
    assert_eq!(artifact["execution"]["fixtures"][0]["fingerprints"][0]["cryptographic"], false);
    assert_eq!(artifact["execution"]["fixtures"][1]["fingerprints"][0]["algorithm"], "sha256");
    assert_eq!(artifact["execution"]["fixtures"][1]["fingerprints"][0]["cryptographic"], true);
    assert_eq!(artifact["execution"]["fixtures"][1]["dataset_metadata"]["scale_factor"], "v12");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn fixture_preflight_rejects_a_self_consistent_noncanonical_dblp_fixture_before_writing_an_artifact() {
    let root = test_root("noncanonical-dblp-fixture");
    fs::create_dir(&root).unwrap();
    let tpch_root = root.join("tpch");
    nexus_workloads::tpch::write_tiny_tpch_parquet_dataset(&tpch_root).unwrap();
    let dblp_root = root.join("citation_network").join("v1");
    // Give the otherwise self-consistent miniature fixture the complete-edge
    // footer shape. Canonical preflight must then reject its non-V12 counts,
    // rather than merely stopping earlier at missing provenance.
    write_dblp_fixture(&dblp_root, Some("3"));
    let profile = fixture_preflight_profile(&tpch_root, &dblp_root);
    let out_dir = root.join("run-artifacts");

    let error = run_fixture_preflight(profile, "a".repeat(64), &out_dir, || -> ToolResult<Provenance> {
        panic!("provenance must not be collected after a failed preflight")
    })
    .unwrap_err();
    assert!(error.to_string().contains("fixture_preflight workload=dblp reason=invalid_layout"), "{error}");
    assert!(error.to_string().contains("full canonical DBLP V12"), "{error}");
    assert!(!out_dir.exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn missing_fixture_returns_structured_reason_before_creating_an_output_directory() {
    let root = test_root("missing-fixture");
    fs::create_dir(&root).unwrap();
    let dblp_root = root.join("citation_network").join("v1");
    write_dblp_fixture(&dblp_root, None);
    let profile = fixture_preflight_profile(&root.join("missing-tpch"), &dblp_root);
    let out_dir = root.join("run-artifacts");

    let error = run_fixture_preflight(profile, "a".repeat(64), &out_dir, || -> ToolResult<Provenance> {
        panic!("provenance must not be collected after a failed preflight")
    })
    .unwrap_err();
    assert!(error.to_string().contains("fixture_preflight workload=tpch reason=missing_fixture"), "{error}");
    assert!(!out_dir.exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn artifact_writer_is_atomic_and_refuses_existing_output_directories() {
    let root = test_root("artifact-writer");
    fs::create_dir(&root).unwrap();
    let out_dir = root.join("run-artifacts");
    let artifact = write_new_artifact(&out_dir, b"{}\n").unwrap();
    assert_eq!(fs::read(&artifact).unwrap(), b"{}\n");
    let error = write_new_artifact(&out_dir, b"replacement").unwrap_err();
    assert!(error.to_string().contains("already exists"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn relative_out_dir_uses_the_current_directory_as_its_parent() {
    assert_eq!(output_parent(Path::new("run-artifacts")), Path::new("."));
    assert_eq!(output_parent(Path::new("target/run-artifacts")), Path::new("target"));
}

fn test_root(label: &str) -> PathBuf {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    env::temp_dir().join(format!("datafusion-nexus-{label}-{}-{unique}", std::process::id()))
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let path = test_root(label);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn s2_profile_generator(out_dir: &Path) -> Command {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/stress/generate_s2_graph_profiles.py");
    let mut command = Command::new("python3");
    command.arg(script).arg("--out-dir").arg(out_dir);
    command
}

fn generate_s2_profiles(out_dir: &Path, knee_qps: u64) {
    let status = s2_profile_generator(out_dir)
        .arg("--knee-qps")
        .arg(knee_qps.to_string())
        .status()
        .expect("start deterministic S2 profile generator");
    assert!(status.success(), "S2 profile generator must succeed");
    let status = s2_profile_generator(out_dir)
        .arg("--knee-qps")
        .arg(knee_qps.to_string())
        .arg("--check")
        .status()
        .expect("check deterministic S2 profile generator");
    assert!(status.success(), "S2 profile generator check must succeed");
}

fn read_generated_s2_profile(out_dir: &Path, name: &str) -> RunProfile {
    let path = out_dir.join(name);
    let source = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    RunProfile::parse_bytes(&source).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn fixture_preflight_profile(tpch_root: &Path, dblp_root: &Path) -> RunProfile {
    let mut profile = RunProfile::parse_bytes(valid_profile_json().as_bytes()).unwrap();
    profile.workloads = vec![
        FixtureWorkloadProfile {
            kind: FixtureKind::Tpch,
            path: tpch_root.to_path_buf(),
            scale: "tiny".to_string(),
            preparation_command: "fixture/fixture.sh tpch generate --scale 1 --out-root /data/tpch".to_string(),
            queries: Vec::new(),
        },
        FixtureWorkloadProfile {
            kind: FixtureKind::Dblp,
            path: dblp_root.to_path_buf(),
            scale: "v12".to_string(),
            preparation_command:
                "fixture/fixture.sh graph dblp-ingest --source-json /data/dblp.v12.json --out-root /data/demo"
                    .to_string(),
            queries: Vec::new(),
        },
    ];
    profile.validate().unwrap();
    profile
}

fn write_dblp_fixture(root: &Path, graph_vertex_upper_bound: Option<&str>) {
    let parquet = root.join("parquet");
    fs::create_dir_all(&parquet).unwrap();
    let rows = |table: &str| -> usize {
        match table {
            "edges_by_src" | "edges_by_dst" => 2,
            "profiles_by_paper_id" => 3,
            "paper_authors" => 4,
            "paper_fos" => 5,
            _ => unreachable!("fixed DBLP table contract"),
        }
    };
    for &table in nexus_workloads::graph::DBLP_TABLES {
        let footer = matches!(table, "edges_by_src" | "edges_by_dst").then_some(graph_vertex_upper_bound).flatten();
        let topology = matches!(table, "edges_by_src" | "edges_by_dst")
            .then_some(r#"{"source_column_index":1,"destination_column_index":2}"#);
        write_tiny_graph_parquet(&parquet.join(format!("{table}.parquet")), table, rows(table), footer, topology);
    }
    let entry = |table: &str| {
        serde_json::json!({
            "bytes": fs::metadata(parquet.join(format!("{table}.parquet"))).unwrap().len(),
            "num_rows": rows(table),
            "num_row_groups": 1,
            "schema": tiny_graph_schema_names(table)
        })
    };
    let manifest = serde_json::json!({
        "schema_version": 1,
        "generator": "fixture/graph/dblp_ingest.py",
        "dataset": {"dataset_id": "citation-network", "source": "test source"},
        "max_papers": 3,
        "counts": {"papers": 3, "edges": 2, "author_rows": 4, "fos_rows": 5},
        "edge_layouts": {
            "src": {"parquet": entry("edges_by_src")},
            "dst": {"parquet": entry("edges_by_dst")}
        },
        "profiles": entry("profiles_by_paper_id"),
        "paper_authors": entry("paper_authors"),
        "paper_fos": entry("paper_fos")
    });
    let source = serde_json::json!({
        "schema_version": 1,
        "dataset_id": "citation-network",
        "source": "test source",
        "source_json": {"path": "/data/dblp.v12.json", "bytes": 1, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
    });
    fs::write(root.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    fs::write(root.join("source.json"), serde_json::to_vec_pretty(&source).unwrap()).unwrap();
}

fn tiny_graph_schema_names(table: &str) -> Vec<&'static str> {
    if matches!(table, "edges_by_src" | "edges_by_dst") {
        vec!["edge_id", "src", "dst", "weight", "src_bucket", "dst_bucket"]
    } else {
        vec!["value"]
    }
}

fn write_tiny_graph_parquet(
    path: &Path,
    table: &str,
    rows: usize,
    graph_vertex_upper_bound: Option<&str>,
    edge_topology: Option<&str>,
) {
    let is_edge_table = matches!(table, "edges_by_src" | "edges_by_dst");
    let schema = Arc::new(Schema::new(if is_edge_table {
        vec![
            Field::new("edge_id", DataType::Int64, false),
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
            Field::new("src_bucket", DataType::Int32, false),
            Field::new("dst_bucket", DataType::Int32, false),
        ]
    } else {
        vec![Field::new("value", DataType::Int64, false)]
    }));
    let values = (0..i64::try_from(rows).unwrap()).collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = if is_edge_table {
        vec![
            Arc::new(Int64Array::from(values.clone())),
            Arc::new(Int64Array::from(values.clone())),
            Arc::new(Int64Array::from(values.clone())),
            Arc::new(Float64Array::from(vec![1.0; rows])),
            Arc::new(Int32Array::from(vec![0; rows])),
            Arc::new(Int32Array::from(vec![0; rows])),
        ]
    } else {
        vec![Arc::new(Int64Array::from(values))]
    };
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let mut footer_metadata = Vec::new();
    if let Some(value) = graph_vertex_upper_bound {
        footer_metadata.push(KeyValue::new(
            nexus_workloads::graph::DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY.to_string(),
            value.to_string(),
        ));
    }
    if let Some(value) = edge_topology {
        footer_metadata.push(KeyValue::new(
            nexus_workloads::graph::DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY.to_string(),
            value.to_string(),
        ));
    }
    let properties = WriterProperties::builder()
        .set_key_value_metadata((!footer_metadata.is_empty()).then_some(footer_metadata))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn sample_provenance() -> Provenance {
    let identity = || FileIdentity { path: "/fixture/file".to_string(), size_bytes: 1, sha256: "b".repeat(64) };
    let repository = |name| RepositoryProvenance {
        name,
        path: format!("/fixture/{name}"),
        revision: "c".repeat(40),
        dirty_manifest: DirtyManifest {
            format: "git-status-porcelain-v1-z+direct-worktree-content-sha256-v1",
            sha256: "d".repeat(64),
            entries: Vec::new(),
        },
    };
    Provenance {
        process: ProcessIdentity { boot_id: "fixture-boot".to_string(), pid: 7, process_start_ticks: 11 },
        repositories: RepositoryProvenanceSet {
            root: repository("datafusion-nexus"),
            cudf_nexus: repository("cudf-nexus"),
            cugraph_nexus: repository("cugraph-nexus"),
        },
        native_libraries: NativeLibraryIdentities {
            libcudf_resolved: Some(identity()),
            libcugraph_resolved: Some(identity()),
        },
        cuvs_build_inputs: CuvsBuildInputs {
            install_dir: "/fixture/cuvs".to_string(),
            cmake_config: identity(),
            libcuvs_selector: identity(),
            libcuvs_resolved: identity(),
            version: "26.06.00".to_string(),
        },
        toolchains: ToolchainVersions {
            rustc: "rustc 1.94".to_string(),
            cargo: "cargo 1.94".to_string(),
            cuda_toolkit: "Cuda compilation tools".to_string(),
            cuda_driver: "580.1".to_string(),
            nsight_systems: "Nsight Systems".to_string(),
            rapids: RapidsVersions {
                cudf: CmakePackageVersion {
                    install_dir: "/fixture/cudf".to_string(),
                    config: identity(),
                    version: "26.06.01".to_string(),
                },
                cugraph: SourcePackageVersion { source: identity(), version: "26.06.00".to_string() },
                cuvs: CmakePackageVersion {
                    install_dir: "/fixture/cuvs".to_string(),
                    config: identity(),
                    version: "26.06.00".to_string(),
                },
            },
            build_profile: "release",
        },
        runner: RunnerIdentity { executable: identity() },
    }
}

/// The shipped four-ordinal profile must stay parseable, heterogeneous, and
/// plannable, so the documented command cannot drift from the parser.
#[test]
fn the_shipped_measured_run_profile_parses_plans_and_is_heterogeneous() {
    let profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/p6-measured-run-profile.json").as_bytes()).unwrap();
    assert_eq!(profile.backend.device_profiles.len(), 4);
    assert_eq!(profile.session.shared_session_count, 2);
    // The mix must select from both fixtures; the exact count is tuning.
    assert!(profile.selected_query_count() >= 2);
    assert!(profile.workloads.iter().all(|workload| !workload.queries.is_empty()));

    // Heterogeneity must be visible in the profile itself, not just intended.
    let limits =
        profile.backend.device_profiles.iter().map(|device| device.memory_limit_bytes).collect::<BTreeSet<_>>();
    let caps = profile.backend.device_profiles.iter().map(|device| device.max_active_attempts).collect::<BTreeSet<_>>();
    let streams = profile
        .backend
        .device_profiles
        .iter()
        .map(|device| device.installation.stream_pool_size)
        .collect::<BTreeSet<_>>();
    assert!(limits.len() > 1 && caps.len() > 1 && streams.len() > 1);
    // The main P6.5 profile remains whole-device-exclusive; the focused
    // StaticPartition evidence uses its own profile so neither scenario
    // dilutes the other's immutable admission envelope.
    assert!(
        profile.backend.device_profiles.iter().all(|device| device.ownership == DeviceOwnership::WholeDeviceExclusive)
    );

    // Both workload classes are selected, and the plan is buildable.
    let queries = select_queries(&profile.workloads).unwrap();
    assert!(queries.iter().any(|query| query.graph));
    assert!(queries.iter().any(|query| !query.graph));
    let plan = ArrivalPlan::plan(&profile.scheduling, queries.len(), profile.session.shared_session_count).unwrap();
    // Closed-loop rounds of the configured width; the driver paces them and
    // assigns phases against the observed warmup deadline.
    assert_eq!(plan.concurrency, profile.scheduling.concurrency);
    assert_eq!(plan.arrivals.len() % plan.concurrency, 0);
    // Every configured ordinal is reachable by the planned attempt volume.
    assert!(plan.arrivals.len() >= profile.backend.device_profiles.len());
}

#[test]
fn p6_isolated_cache_off_profiles_pin_one_graph_shape_and_declared_model_bands() {
    let profiles = [
        ("bfs", include_str!("../../fixtures/p6-isolated-bfs-cache-off.json")),
        ("pagerank", include_str!("../../fixtures/p6-isolated-pagerank-cache-off.json")),
        ("force_atlas2", include_str!("../../fixtures/p6-isolated-force-atlas2-cache-off.json")),
        ("pagerank_forceatlas2_join", include_str!("../../fixtures/p6-isolated-pagerank-forceatlas2-cache-off.json")),
    ];

    for (query_label, json) in profiles {
        let profile = RunProfile::parse_bytes(json.as_bytes()).expect("isolated P6 profile must remain parseable");
        let [device] = profile.backend.device_profiles.as_slice() else {
            panic!("isolated P6 profile must use exactly one device");
        };
        assert_eq!(profile.scenario_requirements(), Default::default());
        assert!(profile.graph_claim_calibration.is_some());
        assert_eq!(profile.cache.mode, CacheMode::Disabled);
        assert_eq!(device.cache_cap_bytes, 0);
        assert_eq!(device.source_work_per_attempt, 2);
        assert_eq!(device.max_active_attempts, 1);
        assert_eq!(device.installation.stream_pool_size, 3);
        assert_eq!(device.installation.worker_count, 1);
        assert_eq!(device.installation.lane_count, 1);
        assert_eq!(profile.scheduling.arrival_mode, ArrivalMode::Concurrency);
        assert_eq!(profile.scheduling.concurrency, 1);
        assert_eq!(profile.scheduling.max_rounds, Some(1));
        let calibration = profile.vram_calibration.expect("P6 must declare a calibration band");
        assert_eq!(calibration.max_underclaim_bytes, 67_108_864);
        assert_eq!(calibration.max_observed_to_claim_ratio_milli, 1_000);
        assert_eq!(calibration.max_overclaim_bytes, 67_108_864);
        assert_eq!(calibration.max_claim_to_observed_ratio_milli, 1_000);

        let selected = select_queries(&profile.workloads).expect("isolated P6 workload label must resolve");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].label, query_label);
        assert!(selected[0].graph);
        assert_eq!(
            profile.workloads[0].path,
            PathBuf::from("/opt/dlami/nvme/fixture/data/demo-v12-topology-r1/citation_network/v1/parquet")
        );
        let plan = ArrivalPlan::plan(&profile.scheduling, selected.len(), profile.session.shared_session_count)
            .expect("isolated P6 profile must plan its one measured attempt");
        assert_eq!(plan.arrivals.len(), 1);
    }
}

#[test]
fn the_shipped_concurrent_quarantine_profile_has_distinct_pre_and_post_ack_cohorts() {
    let profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/p6-concurrent-quarantine-profile.json").as_bytes())
            .unwrap();
    assert_eq!(profile.name, "p6.6-concurrent-quarantine");
    assert_eq!(profile.scheduling.arrival_mode, ArrivalMode::Concurrency);
    assert_eq!(profile.scheduling.max_rounds, Some(2));
    let plan =
        ArrivalPlan::plan(&profile.scheduling, profile.selected_query_count(), profile.session.shared_session_count)
            .expect("quarantine profile plans");
    assert_eq!(plan.arrivals.len(), plan.concurrency * 2);
    assert!(plan.arrivals.iter().any(|arrival| arrival.round == 0));
    assert!(plan.arrivals.iter().any(|arrival| arrival.round == 1));
}

#[test]
fn the_shipped_domain_exhaustion_profile_pins_a_serial_cap_and_recovery_probe() {
    let profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/p6-domain-exhaustion-probe-profile.json").as_bytes())
            .unwrap();
    assert_eq!(profile.name, "p6.6-domain-exhaustion-probe");
    assert_eq!(profile.backend.device_profiles.len(), 1);
    assert!(matches!(
        &profile.backend.device_profiles[0].ownership,
        DeviceOwnership::StaticPartition { allocation_limit_bytes: 402_653_184 }
    ));
    assert_eq!(profile.scheduling.arrival_mode, ArrivalMode::Concurrency);
    assert_eq!(profile.scheduling.concurrency, 1);
    assert_eq!(profile.scheduling.max_rounds, Some(2));
    let labels =
        profile.workloads.iter().flat_map(|workload| workload.queries.iter().map(String::as_str)).collect::<Vec<_>>();
    assert_eq!(labels, ["q01", "edge_scan"]);
}

#[test]
fn the_shipped_physical_allocator_pressure_profile_pins_real_pressure_and_recovery_shape() {
    let profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/p6-physical-allocator-pressure-profile.json").as_bytes())
            .unwrap();
    assert_eq!(profile.name, "p6.6-physical-allocator-pressure");
    assert_eq!(profile.backend.device_profiles.len(), 1);
    assert!(matches!(
        &profile.backend.device_profiles[0].ownership,
        DeviceOwnership::StaticPartition { allocation_limit_bytes: 17_179_869_184 }
    ));
    assert_eq!(profile.scheduling.arrival_mode, ArrivalMode::Concurrency);
    assert_eq!(profile.scheduling.concurrency, 1);
    assert_eq!(profile.scheduling.max_rounds, Some(2));
    assert_eq!(profile.cache.mode, CacheMode::Disabled);
    assert_eq!(profile.backend.native_execution.max_retry_attempts, 1);
    let labels =
        profile.workloads.iter().flat_map(|workload| workload.queries.iter().map(String::as_str)).collect::<Vec<_>>();
    assert_eq!(labels, ["q01", "edge_scan"]);
    profile.validate_physical_allocator_pressure_scenario().unwrap();
}

#[test]
fn parser_accepts_the_measured_run_contract_with_an_external_trace() {
    let options = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--measured-run", "--nsight-trace", "t.rep"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    assert_eq!(options.mode, ValidationMode::MeasuredRun);
    assert_eq!(options.nsight_trace, Some(PathBuf::from("t.rep")));
}

#[test]
fn parser_accepts_controlled_quarantine_only_for_a_measured_run() {
    let options = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--measured-run", "--quarantine-device", "3"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    assert_eq!(options.quarantine_device, Some(3));

    let error = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--dry-run", "--quarantine-device", "0"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("only valid with --measured-run"), "{error}");
}

#[test]
fn parser_scopes_the_after_admission_deadline_probe_to_measured_runs() {
    let options = CliOptions::parse(
        [
            "--config",
            "run-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--deadline-first-after-ms",
            "2000",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap();
    assert_eq!(options.deadline_first_after, Some(std::time::Duration::from_secs(2)));

    let zero = CliOptions::parse(
        [
            "--config",
            "run-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--deadline-first-after-ms",
            "0",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(zero.contains("greater than zero"), "{zero}");

    let combined = CliOptions::parse(
        [
            "--config",
            "run-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--quarantine-device",
            "0",
            "--deadline-first-after-ms",
            "2000",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(combined.contains("cannot be combined"), "{combined}");
}

#[test]
fn parser_scopes_domain_exhaustion_to_the_measured_run_and_keeps_it_isolated() {
    let options = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--measured-run", "--domain-exhaustion-probe"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    assert!(options.domain_exhaustion_probe);

    let dry_error = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--dry-run", "--domain-exhaustion-probe"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(dry_error.contains("only valid with --measured-run"), "{dry_error}");

    let combined_error = CliOptions::parse(
        [
            "--config",
            "run-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--domain-exhaustion-probe",
            "--deadline-first-after-ms",
            "2000",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(combined_error.contains("cannot be combined"), "{combined_error}");
}

#[test]
fn parser_scopes_physical_allocator_pressure_to_the_measured_run_and_keeps_it_isolated() {
    let options = CliOptions::parse(
        [
            "--config",
            "run-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--physical-allocator-pressure",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap();
    assert!(options.physical_allocator_pressure);

    let dry_error = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--dry-run", "--physical-allocator-pressure"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(dry_error.contains("only valid with --measured-run"), "{dry_error}");

    let combined_error = CliOptions::parse(
        [
            "--config",
            "run-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--physical-allocator-pressure",
            "--domain-exhaustion-probe",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(combined_error.contains("cannot be combined"), "{combined_error}");
}

#[test]
fn parser_scopes_cache_calibration_to_the_measured_run_and_keeps_it_distinct_from_fault_injection() {
    let options = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--measured-run", "--cache-calibration"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    assert!(options.cache_calibration);

    let dry_error = CliOptions::parse(
        ["--config", "run-profile.json", "--out-dir", "run-artifacts", "--dry-run", "--cache-calibration"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(dry_error.contains("only valid with --measured-run"), "{dry_error}");

    let combined_error = CliOptions::parse(
        [
            "--config",
            "run-profile.json",
            "--out-dir",
            "run-artifacts",
            "--measured-run",
            "--cache-calibration",
            "--quarantine-device",
            "0",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
    .unwrap_err()
    .to_string();
    assert!(combined_error.contains("cannot be combined"), "{combined_error}");
}

/// A measured-run profile that selects queries but supplies no device profile,
/// no measurement window, or no queries must be refused before any fixture,
/// backend, or artifact work begins.
#[test]
fn measured_run_preflight_rejects_an_incomplete_profile_before_touching_a_fixture() {
    let base = measured_run_profile();

    let mut no_queries = base.clone();
    for workload in &mut no_queries.workloads {
        workload.queries.clear();
    }
    assert!(
        measured_run_preflight(&no_queries, None, None, false, false, false, false, None, false)
            .unwrap_err()
            .to_string()
            .contains("at least one workload query"),
    );

    let mut no_devices = base.clone();
    no_devices.backend.device_profiles.clear();
    assert!(
        measured_run_preflight(&no_devices, None, None, false, false, false, false, None, false)
            .unwrap_err()
            .to_string()
            .contains("device_profiles")
    );

    let mut no_window = base;
    no_window.scheduling.measurement_seconds = 0;
    assert!(
        measured_run_preflight(&no_window, None, None, false, false, false, false, None, false)
            .unwrap_err()
            .to_string()
            .contains("measurement_seconds")
    );
}

#[test]
fn measured_run_preflight_validates_controlled_quarantine_against_configured_devices() {
    let mut profile = measured_run_profile();
    for workload in &mut profile.workloads {
        workload.queries.clear();
    }
    // This must fail on the explicit control before a fixture is inspected;
    // leave one query selected after checking the no-workload guard below.
    profile.workloads[0].queries.push("q01".to_string());
    let error = measured_run_preflight(&profile, Some(99), None, false, false, false, false, None, false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not configured"), "{error}");

    profile.backend.device_profiles.truncate(1);
    let error = measured_run_preflight(&profile, Some(0), None, false, false, false, false, None, false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("at least two configured devices"), "{error}");
}

#[test]
fn controlled_quarantine_requires_a_live_pre_ack_cohort_and_a_post_ack_cohort() {
    let mut rate = measured_run_profile();
    rate.scheduling.arrival_mode = ArrivalMode::Rate;
    let error =
        measured_run_preflight(&rate, Some(0), None, false, false, false, false, None, false).unwrap_err().to_string();
    assert!(error.contains("arrival_mode=concurrency"), "{error}");

    let mut one_round = measured_run_profile();
    one_round.scheduling.max_rounds = Some(1);
    let error = measured_run_preflight(&one_round, Some(0), None, false, false, false, false, None, false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("max_rounds of at least 2"), "{error}");
}

#[test]
fn after_admission_deadline_requires_a_follow_up_cohort() {
    let mut rate = measured_run_profile();
    rate.scheduling.arrival_mode = ArrivalMode::Rate;
    let error = measured_run_preflight(
        &rate,
        None,
        Some(std::time::Duration::from_millis(1)),
        false,
        false,
        false,
        false,
        None,
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("arrival_mode=concurrency"), "{error}");

    let mut one_round = measured_run_profile();
    one_round.scheduling.max_rounds = Some(1);
    let error = measured_run_preflight(
        &one_round,
        None,
        Some(std::time::Duration::from_millis(1)),
        false,
        false,
        false,
        false,
        None,
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("max_rounds of at least 2"), "{error}");
}

#[test]
fn domain_exhaustion_probe_requires_its_fixed_serial_profile_shape() {
    let mut profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/p6-domain-exhaustion-probe-profile.json").as_bytes())
            .unwrap();
    profile.scheduling.concurrency = 2;
    let error =
        measured_run_preflight(&profile, None, None, true, false, false, false, None, false).unwrap_err().to_string();
    assert!(error.contains("concurrency=1"), "{error}");

    profile.scheduling.concurrency = 1;
    profile.workloads[1].queries = vec!["bfs".to_string()];
    let error =
        measured_run_preflight(&profile, None, None, true, false, false, false, None, false).unwrap_err().to_string();
    assert!(error.contains("q01 followed by edge_scan"), "{error}");
}

#[test]
fn physical_allocator_pressure_requires_its_fixed_static_partition_shape() {
    let mut profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/p6-physical-allocator-pressure-profile.json").as_bytes())
            .unwrap();
    profile.scheduling.concurrency = 2;
    let error =
        measured_run_preflight(&profile, None, None, false, false, true, false, None, false).unwrap_err().to_string();
    assert!(error.contains("concurrency=1"), "{error}");

    profile.scheduling.concurrency = 1;
    profile.workloads[1].queries = vec!["bfs".to_string()];
    let error =
        measured_run_preflight(&profile, None, None, false, false, true, false, None, false).unwrap_err().to_string();
    assert!(error.contains("q01 followed by edge_scan"), "{error}");
}

#[test]
fn quarantine_cycles_preflight_enforces_rotation_capacity_and_load_shape() {
    let profile = measured_run_profile();
    // Two configured devices leave one rotation target.
    let error = measured_run_preflight(&profile, None, None, false, false, false, false, Some(2), false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds the configured device count minus one"), "{error}");

    // Rate mode cannot keep a target-witness cohort live.
    let mut rate = measured_run_profile();
    rate.scheduling.arrival_mode = ArrivalMode::Rate;
    rate.scheduling.arrival_rate_per_second = Some(10);
    let error =
        measured_run_preflight(&rate, None, None, false, false, false, false, Some(1), false).unwrap_err().to_string();
    assert!(error.contains("arrival_mode=concurrency"), "{error}");

    // Concurrency below the device count cannot give every target live work.
    let mut sparse = measured_run_profile();
    sparse.scheduling.concurrency = 1;
    let error = measured_run_preflight(&sparse, None, None, false, false, false, false, Some(1), false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("at least the configured device count"), "{error}");

    // One cycle against a two-device fleet with saturation load passes
    // preflight (and then fails only on the missing fixture paths).
    let error = measured_run_preflight(&profile, None, None, false, false, false, false, Some(1), false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("fixture_preflight"), "{error}");
}

#[test]
fn close_under_load_preflight_requires_a_queued_cohort_at_close() {
    let mut profile = measured_run_profile();
    // Fleet capacity is 2 + 4; concurrency 2 admits every attempt at once, so
    // no waiter exists for close to seal.
    let error =
        measured_run_preflight(&profile, None, None, false, false, false, false, None, true).unwrap_err().to_string();
    assert!(error.contains("above the fleet active-attempt capacity"), "{error}");

    // Storm knobs conflict with the close probe's own outcome accounting.
    profile.scheduling.concurrency = 8;
    profile.scheduling.cancel_every_n = Some(2);
    let error =
        measured_run_preflight(&profile, None, None, false, false, false, false, None, true).unwrap_err().to_string();
    assert!(error.contains("cannot be combined"), "{error}");

    profile.scheduling.cancel_every_n = None;
    let error =
        measured_run_preflight(&profile, None, None, false, false, false, false, None, true).unwrap_err().to_string();
    assert!(error.contains("fixture_preflight"), "{error}");
}

#[test]
fn close_under_load_evidence_requires_the_public_active_and_queued_witness() {
    let evidence = super::artifact::CloseUnderLoadEvidence {
        close_requested_offset_ns: 1,
        close_initiated: true,
        admitted_at_close: 1,
        queued_at_close: 1,
        in_flight_at_close: 2,
        submitted_attempts: 2,
        sealed_reports: 2,
        join_failures: 0,
        backend_closing_sealed: 1,
        in_flight_terminal_after_close: 2,
        in_flight_succeeded_after_close: 1,
        accounting_exact: true,
    };
    assert!(evidence.is_evidenced());

    let no_queued_witness = super::artifact::CloseUnderLoadEvidence { queued_at_close: 0, ..evidence.clone() };
    assert!(!no_queued_witness.is_evidenced());

    let incomplete_in_flight =
        super::artifact::CloseUnderLoadEvidence { in_flight_terminal_after_close: 1, ..evidence.clone() };
    assert!(!incomplete_in_flight.is_evidenced());

    let no_active_witness = super::artifact::CloseUnderLoadEvidence { admitted_at_close: 0, ..evidence };
    assert!(!no_active_witness.is_evidenced());
}

#[test]
fn slow_report_sink_preflight_requires_rate_pressure_above_the_serial_drain_capacity() {
    let mut concurrency = measured_run_profile();
    let error = measured_run_preflight_with_completion_sink(
        &concurrency,
        None,
        None,
        false,
        false,
        false,
        false,
        None,
        false,
        Some(std::time::Duration::from_millis(20)),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("arrival_mode=rate"), "{error}");

    concurrency.scheduling.arrival_mode = ArrivalMode::Rate;
    concurrency.scheduling.arrival_rate_per_second = Some(50);
    let error = measured_run_preflight_with_completion_sink(
        &concurrency,
        None,
        None,
        false,
        false,
        false,
        false,
        None,
        false,
        Some(std::time::Duration::from_millis(20)),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("more completions per second"), "{error}");

    concurrency.scheduling.arrival_rate_per_second = Some(51);
    let error = measured_run_preflight_with_completion_sink(
        &concurrency,
        None,
        None,
        false,
        false,
        false,
        false,
        None,
        false,
        Some(std::time::Duration::from_millis(20)),
    )
    .unwrap_err()
    .to_string();
    // The pressure shape passed; the test fixture is intentionally absent.
    assert!(error.contains("fixture_preflight"), "{error}");
}

#[test]
fn steps_mode_replaces_the_scalar_window_and_rate_fields() {
    use super::profile::ScheduleStepProfile;

    let mut profile = measured_run_profile();
    profile.scheduling.measurement_seconds = 0;
    profile.scheduling.steps = vec![
        ScheduleStepProfile {
            concurrency: Some(1),
            arrival_rate_per_second: None,
            measurement_seconds: 5,
            rate_overrun: None,
        },
        ScheduleStepProfile {
            concurrency: Some(4),
            arrival_rate_per_second: None,
            measurement_seconds: 5,
            rate_overrun: None,
        },
    ];
    profile.validate().unwrap();

    let mut scalar_window = profile.clone();
    scalar_window.scheduling.measurement_seconds = 10;
    let error = scalar_window.validate().unwrap_err().to_string();
    assert!(error.contains("measurement_seconds must be 0"), "{error}");

    let mut scalar_rounds = profile.clone();
    scalar_rounds.scheduling.max_rounds = Some(2);
    let error = scalar_rounds.validate().unwrap_err().to_string();
    assert!(error.contains("max_rounds is not valid"), "{error}");

    let mut zero_window = profile.clone();
    zero_window.scheduling.steps[0].measurement_seconds = 0;
    let error = zero_window.validate().unwrap_err().to_string();
    assert!(error.contains("steps[0].measurement_seconds"), "{error}");

    let mut mixed = profile.clone();
    mixed.scheduling.steps[0].arrival_rate_per_second = Some(10);
    let error = mixed.validate().unwrap_err().to_string();
    assert!(error.contains("requires concurrency for concurrency-mode steps"), "{error}");

    let mut rate = measured_run_profile();
    rate.scheduling.arrival_mode = ArrivalMode::Rate;
    rate.scheduling.measurement_seconds = 0;
    rate.scheduling.steps = vec![ScheduleStepProfile {
        concurrency: None,
        arrival_rate_per_second: Some(50),
        measurement_seconds: 5,
        rate_overrun: Some(RateOverrunPolicy::Drop),
    }];
    rate.validate().unwrap();
    let mut missing_rate = rate.clone();
    missing_rate.scheduling.steps[0].arrival_rate_per_second = None;
    let error = missing_rate.validate().unwrap_err().to_string();
    assert!(error.contains("requires arrival_rate_per_second for rate-mode steps"), "{error}");

    // A measured run accepts a steps profile without the scalar window.
    let error =
        measured_run_preflight(&profile, None, None, false, false, false, false, None, false).unwrap_err().to_string();
    assert!(error.contains("fixture_preflight"), "{error}");
}

#[test]
fn qps_envelope_recovery_must_be_the_final_return_to_the_baseline_rate() {
    use super::profile::{QpsEnvelopeProfile, QpsEnvelopeRecoveryProfile, ScheduleStepProfile};

    let mut profile = measured_run_profile();
    profile.scheduling.arrival_mode = ArrivalMode::Rate;
    profile.scheduling.measurement_seconds = 0;
    profile.scheduling.steps = vec![
        ScheduleStepProfile {
            concurrency: None,
            arrival_rate_per_second: Some(10),
            measurement_seconds: 5,
            rate_overrun: Some(RateOverrunPolicy::Delay),
        },
        ScheduleStepProfile {
            concurrency: None,
            arrival_rate_per_second: Some(100),
            measurement_seconds: 5,
            rate_overrun: Some(RateOverrunPolicy::Drop),
        },
        ScheduleStepProfile {
            concurrency: None,
            arrival_rate_per_second: Some(10),
            measurement_seconds: 5,
            rate_overrun: Some(RateOverrunPolicy::Delay),
        },
    ];
    profile.qps_envelope = Some(QpsEnvelopeProfile {
        min_gpu_reaching_completions_per_step: 1,
        recovery: Some(QpsEnvelopeRecoveryProfile {
            baseline_step_index: 0,
            recovery_step_index: 2,
            max_p95_latency_multiplier: 3,
        }),
    });
    profile.vram_calibration = Some(VramCalibrationProfile {
        max_underclaim_bytes: 64 * 1024 * 1024,
        max_observed_to_claim_ratio_milli: 1_000,
        max_overclaim_bytes: 64 * 1024 * 1024,
        max_claim_to_observed_ratio_milli: 1_000,
    });
    profile.validate().unwrap();

    let mut later_overload = profile.clone();
    later_overload.scheduling.steps.push(ScheduleStepProfile {
        concurrency: None,
        arrival_rate_per_second: Some(100),
        measurement_seconds: 5,
        rate_overrun: Some(RateOverrunPolicy::Drop),
    });
    let error = later_overload.validate().unwrap_err().to_string();
    assert!(error.contains("must follow the final above-baseline"), "{error}");

    let mut no_prior_overload = profile;
    no_prior_overload.scheduling.steps[1].arrival_rate_per_second = Some(10);
    let error = no_prior_overload.validate().unwrap_err().to_string();
    assert!(error.contains("requires a prior offered rate"), "{error}");
}

#[test]
fn storm_knobs_validate_their_shapes() {
    let mut profile = measured_run_profile();
    profile.scheduling.attempt_deadline_ms = Some(250);
    profile.scheduling.cancel_every_n = Some(2);
    profile.scheduling.cancel_after_ms = Some(50);
    profile.validate().unwrap();

    let mut zero_deadline = profile.clone();
    zero_deadline.scheduling.attempt_deadline_ms = Some(0);
    let error = zero_deadline.validate().unwrap_err().to_string();
    assert!(error.contains("attempt_deadline_ms"), "{error}");

    let mut zero_cancel = profile.clone();
    zero_cancel.scheduling.cancel_every_n = Some(0);
    let error = zero_cancel.validate().unwrap_err().to_string();
    assert!(error.contains("cancel_every_n"), "{error}");

    let mut orphan_delay = profile.clone();
    orphan_delay.scheduling.cancel_every_n = None;
    let error = orphan_delay.validate().unwrap_err().to_string();
    assert!(error.contains("cancel_after_ms requires"), "{error}");
}

/// Every directly runnable shipped stress-suite profile must parse and
/// validate, so a broken library entry fails here rather than
/// mid-qualification. Templates and generator bases are instantiated by their
/// deterministic generators before use.
#[test]
fn shipped_stress_profiles_parse_and_validate() {
    let stress_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/stress");
    let mut validated = 0;
    for entry in fs::read_dir(&stress_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        let profile = RunProfile::parse_bytes(&bytes)
            .unwrap_or_else(|error| panic!("{} must parse and validate: {error}", path.display()));
        assert_eq!(profile.schema, RUN_PROFILE_SCHEMA);
        validated += 1;
    }
    assert!(validated >= 20, "expected the stress profile library, found {validated} profiles");
}

#[test]
fn stress_graph_profiles_pin_canonical_or_authenticated_multigraph_dblp_fixtures() {
    let stress_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/stress");
    let canonical_path = PathBuf::from("/opt/dlami/nvme/fixture/data/demo-v12-topology-r1/citation_network/v1/parquet");
    let canonical_preparation_command = "fixture/fixture.sh graph dblp-ingest --source-json /path/to/dblp.v12.json --out-root /opt/dlami/nvme/fixture/data/demo-v12-topology-r1";
    let multigraph_root = "/opt/dlami/nvme/fixture/data/demo-v12-topology-r1/citation_network/v1/slices";
    let multigraph_preparation_command = "fixture/fixture.sh graph dblp-ingest --source-json /path/to/dblp.v12.json --out-root /opt/dlami/nvme/fixture/data/demo-v12-topology-r1 --edge-slice-count 8 --edge-slice-seed 20260804";
    let mut profiles = 0;
    let mut templates = 0;

    for entry in fs::read_dir(&stress_dir).unwrap() {
        let entry = entry.unwrap();
        if !entry.file_type().unwrap().is_file() {
            continue;
        }
        let path = entry.path();
        let source = fs::read_to_string(&path).unwrap();
        if path.extension().and_then(|extension| extension.to_str()) == Some("json") {
            let profile = RunProfile::parse_bytes(source.as_bytes())
                .unwrap_or_else(|error| panic!("{} must parse and validate: {error}", path.display()));
            for workload in profile.workloads.iter().filter(|workload| workload.kind == FixtureKind::Dblp) {
                if workload.scale == "v12" {
                    assert_eq!(workload.path, canonical_path, "{}", path.display());
                    assert_eq!(workload.preparation_command, canonical_preparation_command, "{}", path.display());
                } else {
                    assert!(workload.scale.starts_with("v12_slice_"), "{}", path.display());
                    assert!(workload.path.starts_with(multigraph_root), "{}", path.display());
                    assert_eq!(workload.preparation_command, multigraph_preparation_command, "{}", path.display());
                }
                profiles += 1;
            }
        } else if path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.ends_with(".json.template"))
            && source.contains("\"kind\": \"dblp\"")
        {
            if source.contains(multigraph_root) {
                assert!(source.contains(multigraph_preparation_command), "{}", path.display());
            } else {
                assert!(source.contains(canonical_path.to_str().unwrap()), "{}", path.display());
                assert!(source.contains(canonical_preparation_command), "{}", path.display());
            }
            templates += 1;
        }
    }

    assert!(profiles >= 20, "expected graph stress profiles, found {profiles}");
    assert!(templates >= 4, "expected graph stress templates, found {templates}");
}

#[test]
fn w9_distinct_graph_profiles_use_eight_unique_authenticated_slices_and_a_paired_shared_control() {
    const SLICE_ROOT: &str = "/opt/dlami/nvme/fixture/data/demo-v12-topology-r1/citation_network/v1/slices";
    const PREPARATION_COMMAND: &str = "fixture/fixture.sh graph dblp-ingest --source-json /path/to/dblp.v12.json --out-root /opt/dlami/nvme/fixture/data/demo-v12-topology-r1 --edge-slice-count 8 --edge-slice-seed 20260804";

    let s1 = include_str!("../../fixtures/stress/stress-s1-embedded-distinct-graph-interleave-sf10.json").to_string();
    let s4 = include_str!("../../fixtures/stress/stress-s4-plateau-cache-on-sf10.json.template")
        .replace("\"__PLATEAU_QPS__\"", "1")
        .replace("\"__SPIKE_QPS__\"", "2")
        .replace("\"__DRAIN_QPS__\"", "1");
    let s7 = include_str!("../../fixtures/stress/stress-s7-cache-distinct-graph-churn-sf10.json.template")
        .replace("\"__KNEE_QPS__\"", "1");

    for (name, source) in [("S1", s1), ("S4", s4), ("S7", s7)] {
        let profile = RunProfile::parse_bytes(source.as_bytes()).unwrap_or_else(|error| panic!("{name}: {error}"));
        let slices = profile.workloads.iter().filter(|workload| workload.kind == FixtureKind::Dblp).collect::<Vec<_>>();
        assert_eq!(slices.len(), 8, "{name}");

        let paths = slices.iter().map(|workload| &workload.path).collect::<BTreeSet<_>>();
        assert_eq!(paths.len(), 8, "{name}");
        let scales = slices.iter().map(|workload| workload.scale.as_str()).collect::<BTreeSet<_>>();
        let expected_scales = (0..8).map(|index| format!("v12_slice_{index:02}")).collect::<BTreeSet<_>>();
        assert_eq!(scales, expected_scales.iter().map(String::as_str).collect(), "{name}");

        for (index, workload) in slices.into_iter().enumerate() {
            assert_eq!(workload.path, PathBuf::from(format!("{SLICE_ROOT}/source-hash-{index:02}/parquet")), "{name}");
            assert_eq!(workload.scale, format!("v12_slice_{index:02}"), "{name}");
            assert_eq!(workload.preparation_command, PREPARATION_COMMAND, "{name}");
        }
    }

    let s1_profile = RunProfile::parse_bytes(
        include_str!("../../fixtures/stress/stress-s1-embedded-distinct-graph-interleave-sf10.json").as_bytes(),
    )
    .unwrap();
    let shared_control = RunProfile::parse_bytes(
        include_str!("../../fixtures/stress/stress-s1-embedded-shared-graph-interleave-sf10.json").as_bytes(),
    )
    .unwrap();
    let active_capacity =
        s1_profile.backend.device_profiles.iter().map(|device| device.max_active_attempts).sum::<usize>();
    assert_eq!(s1_profile.scheduling.concurrency, active_capacity);
    assert_eq!(s1_profile.scheduling, shared_control.scheduling);
    assert_eq!(s1_profile.backend, shared_control.backend);
    assert_eq!(s1_profile.session, shared_control.session);
    assert_eq!(s1_profile.cache, shared_control.cache);
    assert_eq!(s1_profile.scenario_requirements, shared_control.scenario_requirements);
    assert_eq!(s1_profile.graph_claim_calibration, shared_control.graph_claim_calibration);
    assert_eq!(s1_profile.completion_drain_grace_seconds, shared_control.completion_drain_grace_seconds);
    assert_eq!(s1_profile.scheduling.measurement_seconds, 600);
    assert_eq!(s1_profile.workloads[0].queries, ["q01", "q06"]);
    assert_eq!(shared_control.workloads[0], s1_profile.workloads[0]);
    let selected_shape = |profile: &RunProfile| {
        select_queries(&profile.workloads)
            .unwrap()
            .into_iter()
            .map(|query| (query.kind, query.label))
            .collect::<Vec<_>>()
    };
    assert_eq!(selected_shape(&s1_profile), selected_shape(&shared_control));
    let s1_graph_queries = select_queries(&s1_profile.workloads)
        .unwrap()
        .into_iter()
        .filter(|query| query.kind == FixtureKind::Dblp)
        .map(|query| query.label)
        .collect::<Vec<_>>();
    assert_eq!(
        s1_graph_queries,
        vec![
            "force_atlas2".to_string(),
            "pagerank".to_string(),
            "betweenness_centrality".to_string(),
            "force_atlas2".to_string(),
            "pagerank".to_string(),
            "betweenness_centrality".to_string(),
            "pagerank".to_string(),
            "bfs_deep".to_string(),
        ]
    );
    let shared_graph_workloads =
        shared_control.workloads.iter().filter(|workload| workload.kind == FixtureKind::Dblp).collect::<Vec<_>>();
    assert_eq!(shared_graph_workloads.len(), 1);
    assert_eq!(shared_graph_workloads[0].scale, "v12");

    let s7_graph_queries = RunProfile::parse_bytes(
        include_str!("../../fixtures/stress/stress-s7-cache-distinct-graph-churn-sf10.json.template")
            .replace("\"__KNEE_QPS__\"", "1")
            .as_bytes(),
    )
    .unwrap()
    .workloads
    .into_iter()
    .filter(|workload| workload.kind == FixtureKind::Dblp)
    .flat_map(|workload| workload.queries)
    .collect::<BTreeSet<_>>();
    assert_eq!(s7_graph_queries, BTreeSet::from(["bfs_deep".to_string(), "pagerank".to_string()]));
}

#[test]
fn graph_claim_calibration_requires_graph_work_and_all_i11_graph_profiles_opt_in() {
    let mut no_graph = RunProfile::parse_bytes(valid_profile_json().as_bytes()).unwrap();
    no_graph.graph_claim_calibration = Some(GraphClaimCalibrationProfile {});
    let error = no_graph.validate().unwrap_err().to_string();
    assert!(error.contains("requires at least one selected cuGraph algorithm workload"), "{error}");

    let mut graph_io_only =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s2-sweep-graph-io-sf10.json").as_bytes())
            .unwrap();
    graph_io_only.graph_claim_calibration = Some(GraphClaimCalibrationProfile {});
    let error = graph_io_only.validate().unwrap_err().to_string();
    assert!(error.contains("requires at least one selected cuGraph algorithm workload"), "{error}");

    for (name, source) in [
        ("S1 embedded sf1", include_str!("../../fixtures/stress/stress-s1-embedded-sf1.json")),
        ("S1 embedded sf10", include_str!("../../fixtures/stress/stress-s1-embedded-sf10.json")),
        ("S1 embedded cache sf10", include_str!("../../fixtures/stress/stress-s1-embedded-cache-sf10.json")),
        (
            "S1 distinct graph interleave",
            include_str!("../../fixtures/stress/stress-s1-embedded-distinct-graph-interleave-sf10.json"),
        ),
        (
            "S1 shared graph interleave control",
            include_str!("../../fixtures/stress/stress-s1-embedded-shared-graph-interleave-sf10.json"),
        ),
        ("S3 heterogeneous", include_str!("../../fixtures/stress/stress-s3-heterogeneous-sf1.json")),
        ("S3 utilization sf1", include_str!("../../fixtures/stress/stress-s3-utilization-sf1.json")),
        (
            "S3 utilization telemetry off",
            include_str!("../../fixtures/stress/stress-s3-utilization-telemetry-off-sf1.json"),
        ),
        ("S3 utilization sf10", include_str!("../../fixtures/stress/stress-s3-utilization-sf10.json")),
        ("S4 plateau sf1", include_str!("../../fixtures/stress/stress-s4-plateau-sf1.json")),
        ("S5 cancellation", include_str!("../../fixtures/stress/stress-s5-cancellation-storm-sf10.json")),
        ("S5 deadline", include_str!("../../fixtures/stress/stress-s5-deadline-storm-sf10.json")),
        ("S5 storms", include_str!("../../fixtures/stress/stress-s5-storms-sf1.json")),
        ("S6 close", include_str!("../../fixtures/stress/stress-s6-close-under-load.json")),
        ("S6 close sf1", include_str!("../../fixtures/stress/stress-s6-close-under-load-sf1.json")),
        ("S6 cycles", include_str!("../../fixtures/stress/stress-s6-quarantine-cycles.json")),
        ("S6 cycles sf1", include_str!("../../fixtures/stress/stress-s6-quarantine-cycles-sf1.json")),
        ("S6 cache quarantine", include_str!("../../fixtures/stress/stress-s6-quarantine-cache-sf10.json")),
        ("S7 cache", include_str!("../../fixtures/stress/stress-s7-cache-sf1.json")),
        ("S7 same-key", include_str!("../../fixtures/stress/stress-s7-cache-same-key-sf1.json")),
        ("S7 thrash", include_str!("../../fixtures/stress/stress-s7-cache-thrash-sf10.json")),
    ] {
        let profile = RunProfile::parse_bytes(source.as_bytes()).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(profile.graph_claim_calibration.is_some(), "{name} must retain I11 graph claim evidence");
        assert!(profile.workloads.iter().any(|workload| workload.kind == FixtureKind::Dblp), "{name}");
    }
    for (name, source) in [
        (
            "S4 plateau cache off",
            include_str!("../../fixtures/stress/stress-s4-plateau-cache-off-sf10.json.template")
                .replace("\"__PLATEAU_QPS__\"", "1")
                .replace("\"__SPIKE_QPS__\"", "2")
                .replace("\"__DRAIN_QPS__\"", "1"),
        ),
        (
            "S4 plateau cache on",
            include_str!("../../fixtures/stress/stress-s4-plateau-cache-on-sf10.json.template")
                .replace("\"__PLATEAU_QPS__\"", "1")
                .replace("\"__SPIKE_QPS__\"", "2")
                .replace("\"__DRAIN_QPS__\"", "1"),
        ),
        (
            "S7 cache stress",
            include_str!("../../fixtures/stress/stress-s7-cache-stress-sf10.json.template")
                .replace("\"__KNEE_QPS__\"", "1"),
        ),
    ] {
        let profile = RunProfile::parse_bytes(source.as_bytes()).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(profile.graph_claim_calibration.is_some(), "{name} must retain I11 graph claim evidence");
    }
    let generated = TestDirectory::new("generated-s2-graph-claim");
    generate_s2_profiles(generated.path(), 42);
    for name in [
        "stress-s2-sweep-g1-force-atlas2-sf10.json",
        "stress-s2-sweep-g2-bfs-deep-sf10.json",
        "stress-s2-sweep-g3-pagerank-sf10.json",
        "stress-s2-sweep-g4-betweenness-sf10.json",
    ] {
        let profile = read_generated_s2_profile(generated.path(), name);
        assert!(profile.graph_claim_calibration.is_some(), "{name} must retain I11 graph claim evidence");
    }
}

#[test]
fn s6_cache_quarantine_profile_keeps_resident_graph_traffic_enabled() {
    let profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s6-quarantine-cache-sf10.json").as_bytes())
            .unwrap();
    assert_eq!(profile.cache.mode, CacheMode::Enabled);
    assert!(profile.backend.device_profiles.iter().all(|device| device.cache_cap_bytes == 6 * 1024 * 1024 * 1024));
    let graph = profile.workloads.iter().find(|workload| workload.kind == FixtureKind::Dblp).unwrap();
    assert!(graph.queries.iter().any(|query| query == "bfs"));
    assert!(graph.queries.iter().any(|query| query == "pagerank"));
    assert_eq!(profile.scheduling.arrival_mode, ArrivalMode::Concurrency);
    assert!(profile.scheduling.concurrency >= 4);
}

#[test]
fn s2_sf10_sweeps_end_at_twice_the_profile_owned_fleet_capacity() {
    let profiles = [
        (
            "scan",
            include_str!("../../fixtures/stress/stress-s2-sweep-scan-sf10.json"),
            FixtureKind::Tpch,
            "q06",
            16_usize,
            &[1, 2, 4, 8, 16, 32][..],
            false,
        ),
        (
            "aggregate",
            include_str!("../../fixtures/stress/stress-s2-sweep-aggregate-sf10.json"),
            FixtureKind::Tpch,
            "q01",
            16,
            &[1, 2, 4, 8, 16, 32][..],
            false,
        ),
        (
            "graph I/O",
            include_str!("../../fixtures/stress/stress-s2-sweep-graph-io-sf10.json"),
            FixtureKind::Dblp,
            "edge_scan",
            8,
            &[1, 2, 4, 8, 16][..],
            false,
        ),
    ];

    for (name, source, kind, query, expected_fleet_capacity, expected_steps, expects_graph_calibration) in profiles {
        let profile = RunProfile::parse_bytes(source.as_bytes()).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(profile.scheduling.arrival_mode, ArrivalMode::Concurrency, "{name}");
        assert_eq!(profile.scheduling.warmup_seconds, 10, "{name}");
        assert_eq!(profile.scheduling.measurement_seconds, 0, "{name}");
        let qps_envelope = profile.qps_envelope.expect("S2 sweep must opt into queue/QPS qualification");
        assert_eq!(qps_envelope.min_gpu_reaching_completions_per_step, 1, "{name}");
        assert_eq!(qps_envelope.recovery, None, "{name}");
        let calibration = profile.vram_calibration.expect("S2 sweep must opt into VRAM calibration");
        assert_eq!(calibration.max_underclaim_bytes, 64 * 1024 * 1024, "{name}");
        assert_eq!(calibration.max_observed_to_claim_ratio_milli, 1_000, "{name}");
        assert_eq!(calibration.max_overclaim_bytes, 64 * 1024 * 1024, "{name}");
        assert_eq!(calibration.max_claim_to_observed_ratio_milli, 1_000, "{name}");
        assert_eq!(profile.workloads.len(), 1, "{name}");
        assert_eq!(profile.workloads[0].kind, kind, "{name}");
        assert_eq!(profile.workloads[0].queries, [query], "{name}");
        assert_eq!(profile.graph_claim_calibration.is_some(), expects_graph_calibration, "{name}");

        let fleet_capacity =
            profile.backend.device_profiles.iter().map(|device| device.max_active_attempts).sum::<usize>();
        assert_eq!(fleet_capacity, expected_fleet_capacity, "{name}");

        let steps = profile
            .scheduling
            .steps
            .iter()
            .map(|step| {
                assert_eq!(step.measurement_seconds, 60, "{name}");
                step.concurrency.expect("S2 concurrency sweep step")
            })
            .collect::<Vec<_>>();
        assert_eq!(steps, expected_steps, "{name}");
        assert_eq!(steps.last().copied(), Some(fleet_capacity * 2), "{name}");
    }
}

#[test]
fn generated_s2_graph_sweeps_preserve_the_algorithm_capacity_overlays() {
    let generated = TestDirectory::new("generated-s2-graph-sweep");
    generate_s2_profiles(generated.path(), 42);
    let expected = [
        ("stress-s2-sweep-g1-force-atlas2-sf10.json", "force_atlas2", 4_usize, &[1, 2, 4, 8][..]),
        ("stress-s2-sweep-g2-bfs-deep-sf10.json", "bfs_deep", 8, &[1, 2, 4, 8, 16][..]),
        ("stress-s2-sweep-g3-pagerank-sf10.json", "pagerank", 8, &[1, 2, 4, 8, 16][..]),
        ("stress-s2-sweep-g4-betweenness-sf10.json", "betweenness_centrality", 8, &[1, 2, 4, 8, 16][..]),
    ];
    for (name, query, fleet_capacity, steps) in expected {
        let profile = read_generated_s2_profile(generated.path(), name);
        assert_eq!(profile.workloads[0].kind, FixtureKind::Dblp, "{name}");
        assert_eq!(profile.workloads[0].queries, [query], "{name}");
        assert!(profile.graph_claim_calibration.is_some(), "{name}");
        assert_eq!(
            profile.backend.device_profiles.iter().map(|device| device.max_active_attempts).sum::<usize>(),
            fleet_capacity,
            "{name}"
        );
        assert_eq!(
            profile.scheduling.steps.iter().map(|step| step.concurrency.unwrap()).collect::<Vec<_>>(),
            steps,
            "{name}"
        );
    }
}

const S2_RATE_ARRIVAL_BUDGET: u64 = 80_000;

fn s2_even_knee_qps(observed_qps_milli: u64) -> u64 {
    let rounded_up_qps = observed_qps_milli.div_ceil(1_000).max(1);
    rounded_up_qps.div_ceil(2) * 2
}

fn s2_rate_phase_seconds(knee_qps: u64) -> Option<u64> {
    let numerator = S2_RATE_ARRIVAL_BUDGET.checked_sub(5_u64.checked_mul(knee_qps)?)?.checked_mul(2)?;
    let denominator = 81_u64.checked_mul(knee_qps)?;
    let seconds = (numerator / denominator).min(60);
    (seconds != 0).then_some(seconds)
}

#[test]
fn generated_s2_rate_profiles_pin_even_knees_and_a_cap_safe_phase_window() {
    assert_eq!(s2_even_knee_qps(278), 2);
    assert_eq!(s2_even_knee_qps(33_417), 34);
    assert_eq!(s2_even_knee_qps(42_000), 42);
    assert_eq!(s2_even_knee_qps(42_001), 44);

    assert_eq!(s2_rate_phase_seconds(32), Some(60));
    assert_eq!(s2_rate_phase_seconds(34), Some(57));
    assert_eq!(s2_rate_phase_seconds(42), Some(46));
    assert_eq!(s2_rate_phase_seconds(1_758), Some(1));
    assert_eq!(s2_rate_phase_seconds(1_760), None);

    let generated = TestDirectory::new("generated-s2-rate");
    for (knee_qps, expected) in
        [(41, "must be a positive even integer"), (1_760, "largest supported positive even knee is 1758")]
    {
        let output = s2_profile_generator(generated.path())
            .arg("--knee-qps")
            .arg(knee_qps.to_string())
            .output()
            .expect("run invalid S2 profile generator invocation");
        assert!(!output.status.success(), "knee {knee_qps} must be rejected");
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
    }
    generate_s2_profiles(generated.path(), 42);
    let profiles = [
        ("stress-s2-rate-scan-sf10.json", false),
        ("stress-s2-rate-aggregate-sf10.json", false),
        ("stress-s2-rate-graph-io-sf10.json", false),
        ("stress-s2-rate-g1-force-atlas2-sf10.json", true),
        ("stress-s2-rate-g2-bfs-deep-sf10.json", true),
        ("stress-s2-rate-g3-pagerank-sf10.json", true),
        ("stress-s2-rate-g4-betweenness-sf10.json", true),
    ];
    let expected_rates = [21, 42, 63, 84, 210, 420];

    for (name, expects_graph_calibration) in profiles {
        let profile = read_generated_s2_profile(generated.path(), name);
        assert_eq!(profile.scheduling.arrival_mode, ArrivalMode::Rate);
        assert_eq!(profile.scheduling.steps.len(), 13);
        for (index, step) in profile.scheduling.steps.iter().enumerate() {
            assert_eq!(step.measurement_seconds, 46);
            assert_eq!(
                step.arrival_rate_per_second,
                Some(if index == 12 { expected_rates[0] } else { expected_rates[index % expected_rates.len()] })
            );
            assert_eq!(
                step.rate_overrun,
                Some(if index < 6 || index == 12 { RateOverrunPolicy::Delay } else { RateOverrunPolicy::Drop })
            );
        }
        let recovery = profile.qps_envelope.and_then(|envelope| envelope.recovery).expect("S2 recovery contract");
        assert_eq!(recovery.baseline_step_index, 0);
        assert_eq!(recovery.recovery_step_index, 12);
        assert_eq!(recovery.max_p95_latency_multiplier, 3);
        let calibration = profile.vram_calibration.expect("S2 rate profile must opt into VRAM calibration");
        assert_eq!(calibration.max_underclaim_bytes, 64 * 1024 * 1024);
        assert_eq!(calibration.max_observed_to_claim_ratio_milli, 1_000);
        assert_eq!(calibration.max_overclaim_bytes, 64 * 1024 * 1024);
        assert_eq!(calibration.max_claim_to_observed_ratio_milli, 1_000);
        assert_eq!(profile.graph_claim_calibration.is_some(), expects_graph_calibration, "{name}");
        let plan = ArrivalPlan::plan(&profile.scheduling, 1, profile.session.shared_session_count).unwrap();
        assert!(!plan.truncated_by_attempt_cap);
        assert!(plan.arrivals.len() < 100_000);
    }
}

#[test]
fn s2_aggregate_profiles_reserve_the_observed_q01_working_set() {
    // The 1 GiB aggregate domain failed with this attempt-exact allocation:
    // 1,068,599,820 bytes live plus a 15,237,120-byte request.  Keep the
    // immutable profile floor above the actual reservation-limit witness,
    // instead of attempting to hide it with runner retries.
    const Q01_OBSERVED_LIVE_BYTES: u64 = 1_068_599_820;
    const Q01_OBSERVED_REQUEST_BYTES: u64 = 15_237_120;
    const Q01_ATTEMPT_FLOOR_BYTES: u64 = 1_342_177_280;
    let sweep =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s2-sweep-aggregate-sf10.json").as_bytes())
            .unwrap();
    let generated = TestDirectory::new("generated-s2-aggregate");
    generate_s2_profiles(generated.path(), 42);
    let rate = read_generated_s2_profile(generated.path(), "stress-s2-rate-aggregate-sf10.json");

    assert_eq!(sweep.backend.device_profiles, rate.backend.device_profiles);
    assert_eq!(sweep.backend.native_execution, rate.backend.native_execution);
    assert_eq!(sweep.backend.native_execution.device_budget_bytes, Q01_ATTEMPT_FLOOR_BYTES);
    for device in &sweep.backend.device_profiles {
        assert_eq!(device.attempt_floor_bytes, Q01_ATTEMPT_FLOOR_BYTES);
        assert!(device.attempt_floor_bytes >= Q01_OBSERVED_LIVE_BYTES + Q01_OBSERVED_REQUEST_BYTES);
        let managed_capacity = device.memory_limit_bytes - device.backend_reserve_bytes - device.cache_cap_bytes;
        assert!(device.attempt_floor_bytes * u64::try_from(device.max_active_attempts).unwrap() <= managed_capacity);
    }
}

#[test]
fn s7_same_key_profile_requires_source_and_bfs_coalescing() {
    let profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s7-cache-same-key-sf1.json").as_bytes())
            .unwrap();
    let stress = profile.cache.stress.as_ref().expect("S7 same-key evidence contract");

    assert_eq!(stress.required_per_device_cap_bytes, 6 * 1024 * 1024 * 1024);
    assert!(stress.require_source_coalescing);
    assert!(stress.require_bfs_singleflight);
    assert!(stress.require_explained_admission_delta);
    let same_key_probe = stress.same_key_probe.as_ref().expect("profile-owned same-key probe");
    assert_eq!(same_key_probe.target_device_ordinal, 0);
    assert_eq!(same_key_probe.cohort_width, 2);
    assert_eq!(same_key_probe.source.kind, FixtureKind::Tpch);
    assert_eq!(same_key_probe.source.label, "q06");
    assert_eq!(same_key_probe.bfs.kind, FixtureKind::Dblp);
    assert_eq!(same_key_probe.bfs.label, "bfs");
    assert_eq!(profile.workloads[0].queries, ["q06"]);
    assert_eq!(profile.workloads[1].queries, ["bfs"]);
}

#[test]
fn s7_same_key_probe_rejects_ambiguous_dblp_slice_queries() {
    let mut profile =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s7-cache-same-key-sf1.json").as_bytes())
            .unwrap();
    profile.workloads.push(FixtureWorkloadProfile {
        kind: FixtureKind::Dblp,
        path: PathBuf::from("/fixtures/dblp/slices/source-hash-07/parquet"),
        scale: "v12_slice_07".to_string(),
        preparation_command: "fixture/fixture.sh graph dblp-ingest --source-json /fixtures/dblp.v12.json --out-root /fixtures/dblp --edge-slice-count 8 --edge-slice-seed 20260804".to_string(),
        queries: vec!["bfs".to_string()],
    });

    let error = profile.validate().unwrap_err().to_string();
    assert!(error.contains("same_key_probe.bfs matches 2 selected dblp query bfs entries"), "{error}");
}

#[test]
fn s1_sf10_cache_ab_profiles_pin_the_sustained_mixed_cohort() {
    let cache_off =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s1-embedded-sf10.json").as_bytes()).unwrap();
    let cache_on =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s1-embedded-cache-sf10.json").as_bytes())
            .unwrap();

    let expected = [
        (FixtureKind::Tpch, "q01"),
        (FixtureKind::Tpch, "q06"),
        (FixtureKind::Tpcds, "q10"),
        (FixtureKind::Clickbench, "q28"),
        (FixtureKind::Dblp, "force_atlas2"),
        (FixtureKind::Dblp, "bfs_deep"),
        (FixtureKind::Dblp, "pagerank"),
        (FixtureKind::Dblp, "betweenness_centrality"),
    ];

    assert_eq!(cache_off.scheduling.arrival_mode, ArrivalMode::Concurrency);
    assert_eq!(cache_off.scheduling.concurrency, expected.len());
    assert_eq!(cache_off.scheduling.warmup_seconds, 0);
    assert_eq!(cache_off.scheduling.measurement_seconds, 600);
    assert_eq!(cache_off.completion_drain_grace_seconds, 30);

    let selected = select_queries(&cache_off.workloads).unwrap();
    let cohort = selected.iter().map(|query| (query.kind, query.label.as_str())).collect::<Vec<_>>();
    assert_eq!(cohort, expected);
    // Duration-paced scheduling selects `sequence % query_count`, so an
    // exact-width list makes every closed-loop round repeat this mix until
    // the profile-owned measurement deadline.
    for sequence in 0..expected.len() * 3 {
        assert_eq!(cohort[sequence % cohort.len()], expected[sequence % expected.len()]);
    }

    let mut derived_cache_off = cache_on.cache_disabled_companion().unwrap();
    derived_cache_off.name.clone_from(&cache_off.name);
    assert_eq!(derived_cache_off, cache_off);
}

#[test]
fn s1_embedded_profiles_pin_the_native_clickbench_fixture() {
    let sf1 =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s1-embedded-sf1.json").as_bytes()).unwrap();
    let sf10 =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s1-embedded-sf10.json").as_bytes()).unwrap();
    let sf10_cache =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s1-embedded-cache-sf10.json").as_bytes())
            .unwrap();

    fn clickbench_workload(profile: &RunProfile) -> &FixtureWorkloadProfile {
        let mut workloads = profile.workloads.iter().filter(|workload| workload.kind == FixtureKind::Clickbench);
        let clickbench = workloads.next().expect("S1 embedded profile must select ClickBench");
        assert!(workloads.next().is_none(), "S1 embedded profile must select ClickBench exactly once");
        clickbench
    }

    let expected_path = PathBuf::from("/opt/dlami/nvme/fixture/data/clickbench");
    let expected_preparation_command =
        "fixture/fixture.sh clickbench fetch --layout single --out /opt/dlami/nvme/fixture/data/clickbench";
    for profile in [&sf1, &sf10, &sf10_cache] {
        let clickbench = clickbench_workload(profile);
        assert_eq!(clickbench.scale, "single");
        assert_eq!(clickbench.path, expected_path);
        assert_eq!(clickbench.preparation_command, expected_preparation_command);
    }
    assert_eq!(clickbench_workload(&sf1).queries, ["q28", "q29", "q34", "q35"]);
    assert_eq!(clickbench_workload(&sf10).queries, ["q28"]);
    assert_eq!(clickbench_workload(&sf10), clickbench_workload(&sf10_cache));
}

#[test]
fn s3_profiles_pin_the_telemetry_ab_and_heterogeneous_envelope_contracts() {
    let telemetry_on =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s3-utilization-sf1.json").as_bytes())
            .unwrap();
    let telemetry_off = RunProfile::parse_bytes(
        include_str!("../../fixtures/stress/stress-s3-utilization-telemetry-off-sf1.json").as_bytes(),
    )
    .unwrap();
    let heterogeneous =
        RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s3-heterogeneous-sf1.json").as_bytes())
            .unwrap();

    assert!(telemetry_on.backend.device_profiles.iter().all(|device| device.install_telemetry));
    assert!(telemetry_off.backend.device_profiles.iter().all(|device| !device.install_telemetry));
    assert_eq!(telemetry_on.scheduling, telemetry_off.scheduling);
    assert_eq!(telemetry_on.session, telemetry_off.session);
    assert_eq!(telemetry_on.backend.queue, telemetry_off.backend.queue);
    assert_eq!(telemetry_on.backend.native_execution, telemetry_off.backend.native_execution);
    assert_eq!(telemetry_on.workloads, telemetry_off.workloads);
    for (on, off) in telemetry_on.backend.device_profiles.iter().zip(&telemetry_off.backend.device_profiles) {
        assert_eq!(on.ordinal, off.ordinal);
        assert_eq!(on.ownership, off.ownership);
        assert_eq!(on.memory_limit_bytes, off.memory_limit_bytes);
        assert_eq!(on.backend_reserve_bytes, off.backend_reserve_bytes);
        assert_eq!(on.cache_cap_bytes, off.cache_cap_bytes);
        assert_eq!(on.attempt_floor_bytes, off.attempt_floor_bytes);
        assert_eq!(on.source_work_per_attempt, off.source_work_per_attempt);
        assert_eq!(on.max_active_attempts, off.max_active_attempts);
        assert_eq!(on.installation, off.installation);
    }

    let requirements = heterogeneous.scenario_requirements();
    assert!(requirements.require_all_device_progress);
    assert!(requirements.require_heterogeneous_device_profiles);
    assert!(
        heterogeneous
            .backend
            .device_profiles
            .windows(2)
            .any(|pair| pair[0].memory_limit_bytes != pair[1].memory_limit_bytes
                || pair[0].max_active_attempts != pair[1].max_active_attempts)
    );
}

/// The fixture paths in `measured_run_profile` do not exist, so preflight must
/// fail with the structured fixture reason rather than proceeding.
#[test]
fn measured_run_preflight_reports_a_missing_fixture_with_a_structured_reason() {
    let error = measured_run_preflight(&measured_run_profile(), None, None, false, false, false, false, None, false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("fixture_preflight"), "{error}");
    assert!(error.contains("reason=missing_fixture"), "{error}");
}

#[test]
fn overload_fairness_preflight_validates_the_finite_cohort_before_fixture_access() {
    let mut profile = measured_run_profile();
    profile.scheduling.concurrency = 8;
    profile.scheduling.max_rounds = Some(1);
    profile.scheduling.warmup_seconds = 0;

    // The profile's fixture paths are deliberately absent. Reaching this
    // structured fixture error proves the overload shape was accepted before
    // the runner touches a backend or writes an artifact.
    let error =
        measured_run_preflight(&profile, None, None, false, true, false, false, None, false).unwrap_err().to_string();
    assert!(error.contains("fixture_preflight"), "{error}");
    assert!(error.contains("reason=missing_fixture"), "{error}");
}

/// A measured run only needs the fixtures it will execute, unlike the P6.4
/// preflight contract which requires exactly one of each kind.
#[test]
fn a_measured_run_may_execute_one_fixture_kind_while_preflight_requires_both() {
    let root = test_root("measured-single-kind");
    fs::create_dir(&root).unwrap();
    let tpch_root = root.join("tpch");
    nexus_workloads::tpch::write_tiny_tpch_parquet_dataset(&tpch_root).unwrap();

    let mut profile = measured_run_profile();
    profile.workloads = vec![FixtureWorkloadProfile {
        kind: FixtureKind::Tpch,
        path: tpch_root,
        scale: "tiny".to_string(),
        preparation_command: "fixture/fixture.sh tpch generate --scale 1 --out-root /data/tpch".to_string(),
        queries: vec!["q01".to_string()],
    }];
    profile.validate().unwrap();

    let fixtures = FixturePreflight::collect(&profile.workloads, FixtureSelectionPolicy::AtLeastOneKind).unwrap();
    assert_eq!(fixtures.len(), 1);
    let error = FixturePreflight::collect(&profile.workloads, FixtureSelectionPolicy::BothKindsExactly).unwrap_err();
    assert!(error.to_string().contains("exactly one tpch and one dblp"), "{error}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn profile_accepts_explicit_repeated_workload_queries_and_rejects_unknown_labels() {
    let mut profile = measured_run_profile();
    profile.workloads[0].queries = vec!["q01".to_string(), "q01".to_string()];
    profile.validate().unwrap();
    assert_eq!(
        select_queries(&profile.workloads)
            .unwrap()
            .into_iter()
            .filter(|query| query.kind == FixtureKind::Tpch)
            .map(|query| query.label)
            .collect::<Vec<_>>(),
        ["q01", "q01"]
    );

    let mut profile = measured_run_profile();
    profile.workloads[0].queries = vec!["q99".to_string()];
    // Label shape is valid, so validation passes and resolution rejects it.
    profile.validate().unwrap();
    assert!(select_queries(&profile.workloads).unwrap_err().to_string().contains("unknown tpch query label"));
}

/// The shipped profile no longer selects static partitioning, so keep the
/// ownership variant's parse and validation path covered here.
#[test]
fn profile_accepts_static_partition_ownership_and_rejects_a_zero_allocation_limit() {
    let mut profile = measured_run_profile();
    profile.backend.device_profiles[0].ownership =
        DeviceOwnership::StaticPartition { allocation_limit_bytes: 4 * 1024 * 1024 * 1024 };
    profile.validate().unwrap();

    profile.backend.device_profiles[0].ownership = DeviceOwnership::StaticPartition { allocation_limit_bytes: 0 };
    let error = profile.validate().unwrap_err().to_string();
    assert!(error.contains("allocation_limit_bytes must be greater than zero"), "{error}");
}

/// The engine needs one root slot plus `source_work_per_attempt` auxiliary
/// slots per active attempt. Catching it locally keeps a misconfigured profile
/// from failing only once a backend build has already started.
#[test]
fn profile_rejects_a_stream_pool_below_the_engine_slot_formula() {
    let mut profile = measured_run_profile();
    // 4 active attempts x (2 source work + 1) needs 12 slots. Worker and lane
    // counts are set to satisfy their own rule so this isolates the pool check.
    profile.backend.device_profiles[0].max_active_attempts = 4;
    profile.backend.device_profiles[0].source_work_per_attempt = 2;
    profile.backend.device_profiles[0].installation.worker_count = 4;
    profile.backend.device_profiles[0].installation.lane_count = 4;
    profile.backend.device_profiles[0].installation.stream_pool_size = 11;
    let error = profile.validate().unwrap_err().to_string();
    assert!(error.contains("stream_pool_size must be at least 12"), "{error}");

    profile.backend.device_profiles[0].installation.stream_pool_size = 12;
    profile.validate().unwrap();
}

#[test]
fn profile_rejects_a_backend_reserve_that_exhausts_the_device_limit() {
    let mut profile = measured_run_profile();
    profile.backend.device_profiles[0].backend_reserve_bytes = profile.backend.device_profiles[0].memory_limit_bytes;
    let error = profile.validate().unwrap_err().to_string();
    assert!(error.contains("backend_reserve_bytes must be below memory_limit_bytes"), "{error}");
}

#[test]
fn cache_mode_matches_the_single_p5_ledger_cache_cap() {
    let mut profile = measured_run_profile();
    profile.backend.device_profiles[0].cache_cap_bytes = 64 * 1024 * 1024;
    let error = profile.validate().unwrap_err().to_string();
    assert!(error.contains("cache.mode disabled"), "{error}");

    profile.cache.mode = CacheMode::Enabled;
    profile.validate().unwrap();

    profile.backend.device_profiles[0].cache_cap_bytes = 0;
    let error = profile.validate().unwrap_err().to_string();
    assert!(error.contains("cache.mode enabled"), "{error}");
}

#[test]
fn cache_calibration_derives_a_disabled_companion_without_changing_any_other_profile_surface() {
    let mut profile = measured_run_profile();
    profile.cache.mode = CacheMode::Enabled;
    profile.backend.device_profiles[0].cache_cap_bytes = 64 * 1024 * 1024;
    profile.validate().unwrap();

    let companion = profile.cache_disabled_companion().unwrap();
    assert_eq!(companion.cache.mode, CacheMode::Disabled);
    assert!(companion.backend.device_profiles.iter().all(|device| device.cache_cap_bytes == 0));
    assert_eq!(companion.workloads, profile.workloads);
    assert_eq!(companion.scheduling, profile.scheduling);
    assert_eq!(companion.session, profile.session);
    assert_eq!(companion.backend.queue, profile.backend.queue);
    assert_eq!(companion.backend.native_execution, profile.backend.native_execution);
    for (before, after) in profile.backend.device_profiles.iter().zip(&companion.backend.device_profiles) {
        assert_eq!(before.ordinal, after.ordinal);
        assert_eq!(before.ownership, after.ownership);
        assert_eq!(before.memory_limit_bytes, after.memory_limit_bytes);
        assert_eq!(before.backend_reserve_bytes, after.backend_reserve_bytes);
        assert_eq!(before.attempt_floor_bytes, after.attempt_floor_bytes);
        assert_eq!(before.source_work_per_attempt, after.source_work_per_attempt);
        assert_eq!(before.max_active_attempts, after.max_active_attempts);
        assert_eq!(before.installation, after.installation);
    }
}

#[test]
fn profile_rejects_a_rate_overrun_policy_without_rate_scheduling() {
    let mut profile = measured_run_profile();
    profile.scheduling.rate_overrun = RateOverrunPolicy::Drop;
    assert!(profile.validate().unwrap_err().to_string().contains("rate_overrun"));
}

#[test]
fn profile_scopes_finite_round_bounds_to_closed_loop_scheduling() {
    let mut profile = measured_run_profile();
    profile.scheduling.max_rounds = Some(0);
    assert!(profile.validate().unwrap_err().to_string().contains("max_rounds"));

    profile.scheduling.max_rounds = Some(1);
    profile.scheduling.arrival_mode = ArrivalMode::Rate;
    profile.scheduling.arrival_rate_per_second = Some(1);
    assert!(profile.validate().unwrap_err().to_string().contains("max_rounds"));
}

/// The end-to-end artifact shape for a measured run, assembled from the pure
/// reduction so it needs no GPU. This is the schema contract P6.5 publishes.
#[test]
fn measured_run_artifact_records_scenarios_throughput_placement_and_declared_gaps() {
    let root = test_root("measured-artifact");
    fs::create_dir(&root).unwrap();
    let out_dir = root.join("run-artifacts");

    let profile = measured_run_profile();
    let queries = select_queries(&profile.workloads).unwrap();
    let plan = ArrivalPlan::plan(&profile.scheduling, queries.len(), profile.session.shared_session_count).unwrap();
    let devices = vec![measured_device(0, 1 << 30, 2), measured_device(1, 2 << 30, 4)];
    let observations = measured_observations();
    let occupancy = BTreeMap::from([(0, measured_occupancy(2)), (1, measured_occupancy(2))]);
    let aggregate = MeasuredAggregate::reduce(&observations, &devices, &occupancy, plan.warmup_ns, plan.measurement_ns);
    assert!(scenario_clauses_pass(&aggregate.scenarios));
    let mut time_series = super::artifact::TimeSeriesEvidence::empty();
    for device in &devices {
        time_series.device_rows.push(DeviceSeriesRow {
            offset_ns: 1,
            device_ordinal: device.ordinal,
            health_open: true,
            gpu_utilization_pct: Some(1),
            memory_utilization_pct: Some(1),
            cuda_free_bytes: Some(device.capacity_bytes),
            nvml_used_bytes: Some(1),
            active_attempts: 1,
            managed_capacity_bytes: device.capacity_bytes,
            live_grants_bytes: device.capacity_bytes,
            live_reserved_bytes: 0,
            cache_evictable_resident_bytes: 0,
            cache_pinned_in_use_bytes: 0,
            cache_physical_current_bytes: None,
            cache_physical_peak_bytes: None,
            allocation_failure_count: 0,
            ledger_sequence: 1,
            claim_snapshot_end_sequence: Some(1),
            claim_snapshot_stable: true,
            active_query_jobs: 1,
            queued_jobs: 0,
        });
    }

    let measurement = super::artifact::MeasuredRunArtifact {
        qualifying: true,
        backend: super::artifact::BackendEvidence {
            resource_core: 1,
            observation_core: 2,
            resource_generation: 3,
            shared_session_count: profile.session.shared_session_count,
            queue_max_queued_attempts: profile.backend.queue.max_queued_attempts,
            queue_max_queue_wait_ms: profile.backend.queue.max_queue_wait_ms,
            queue_max_overtakes_per_waiter: profile.backend.queue.max_overtakes_per_waiter,
            device_startup_records: serde_json::to_value(&devices).unwrap(),
            close: test_close_evidence(),
        },
        selected_queries: queries.iter().map(super::workload::SelectedQuery::descriptor).collect(),
        arrival_plan: super::artifact::ArrivalPlanEvidence::new(&plan, 0, 0, 0, 0),
        cache_state: "disabled",
        composed_schedules: Vec::new(),
        cache_same_key_probe: None,
        qps_envelope: None,
        graph_claim_calibration: None,
        aggregate,
        deadline: test_deadline_evidence(),
        cancellation_barrier: None,
        deadline_barrier: None,
        domain_exhaustion: None,
        physical_allocator_pressure: None,
        quarantine_cycles: None,
        close_under_load: None,
        completion_sink: None,
        feature_off_cugraph: None,
        metrics_audit: measured_metrics_audit(4),
        operability: super::artifact::OperabilityEvidence {
            attribution: super::operability::AttributionCompleteness::from_observations(&observations),
            otlp_export: super::operability::OtlpExportEvidence::not_requested(),
        },
        time_series,
        snapshots: Vec::new(),
        trace: super::artifact::TraceEvidence {
            captured_by_runner: false,
            uri: Some("target/p6.5/attempt-overlap.nsys-rep".to_string()),
            sha256: Some("e".repeat(64)),
        },
    };
    let artifact = RunArtifact::measured_run(profile, "a".repeat(64), Vec::new(), sample_provenance(), measurement);
    let path = write_artifact(&out_dir, &artifact).unwrap();
    let value: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();

    assert_eq!(value["schema"], RUN_ARTIFACT_SCHEMA);
    assert_eq!(value["status"], "measured_run");
    assert_eq!(value["execution"]["workload_execution"], "complete");
    assert_eq!(value["execution"]["attempted_backend_attempts"], 4);
    let measurement = &value["execution"]["measurement"];
    assert_eq!(measurement["qualifying"], true);
    assert_eq!(measurement["cache_state"], "disabled");
    assert_eq!(measurement["arrival_plan"]["seed"], 7);
    assert_eq!(measurement["arrival_plan"]["rate_overrun"], "delay");
    assert!(measurement["aggregate"]["throughput"]["application_qps"].as_f64().unwrap() > 0.0);
    assert!(measurement["aggregate"]["latency"]["p95_ns"].as_u64().unwrap() > 0);
    assert_eq!(measurement["aggregate"]["placement"].as_array().unwrap().len(), 2);
    assert_eq!(measurement["aggregate"]["report_aggregate_equality"]["equal"], true);
    assert_eq!(measurement["aggregate"]["attempts"][0]["admission_sequence"], 0);
    assert_eq!(measurement["aggregate"]["attempts"][0]["admission_requested_ns"], 0);
    assert_eq!(measurement["aggregate"]["attempts"][0]["admitted_ns"], 1);

    // Streams are recorded as profile-derived accounting and Nsight is cited,
    // never claimed by this runner.
    let occupancy = &measurement["aggregate"]["scenarios"]["logical_slot_occupancy"];
    assert_eq!(occupancy["basis"], "profile_derived_logical_slots");
    assert_eq!(occupancy["proves_gpu_overlap"], false);
    assert_eq!(measurement["trace"]["captured_by_runner"], false);

    // The two clauses this phase cannot evidence are declared, not passed.
    let clause = |section: &str, name: &str| -> String {
        measurement["aggregate"]["scenarios"][section]
            .as_array()
            .unwrap()
            .iter()
            .find(|clause| clause["clause"] == name)
            .unwrap()["status"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(clause("same_device", "distinct_grants"), "evidenced");
    assert_eq!(clause("same_device", "distinct_domains"), "evidenced");
    assert_eq!(clause("same_device", "witnessed_concurrent_device_occupancy"), "evidenced");
    assert_eq!(clause("same_device", "nsight_confirmed_overlapping_cuda_ranges"), "not_evidenced");
    assert_eq!(clause("multiple_devices", "progress_on_all_eligible_devices"), "evidenced");
    assert_eq!(clause("multiple_devices", "heterogeneous_ordinal_profiles"), "evidenced");
    assert_eq!(clause("multiple_devices", "device_block_or_quarantine_fallback"), "not_evidenced");

    ensure_redacted_json(&value, "").unwrap();
    // No raw SQL reaches the artifact, only query identity.
    assert!(
        !String::from_utf8(fs::read_to_string(out_dir.join("run-artifact.json")).unwrap().into_bytes())
            .unwrap()
            .contains("SELECT")
    );
    fs::remove_dir_all(root).unwrap();
}

/// A failed clause must flip the artifact status so a non-qualifying run can
/// never be read as a pass.
#[test]
fn a_failed_scenario_clause_marks_the_artifact_non_qualifying() {
    let devices = vec![measured_device(0, 1 << 30, 2), measured_device(1, 2 << 30, 4)];
    // Ordinal 1 is eligible but completes nothing.
    let observations = measured_observations().into_iter().filter(|a| a.device_ordinal == Some(0)).collect::<Vec<_>>();
    let aggregate = MeasuredAggregate::reduce(
        &observations,
        &devices,
        &BTreeMap::from([(0, measured_occupancy(2))]),
        0,
        1_000_000_000,
    );
    assert!(!scenario_clauses_pass(&aggregate.scenarios));

    let profile = measured_run_profile();
    let measurement = super::artifact::MeasuredRunArtifact {
        qualifying: false,
        backend: super::artifact::BackendEvidence {
            resource_core: 1,
            observation_core: 2,
            resource_generation: 3,
            shared_session_count: 1,
            queue_max_queued_attempts: 64,
            queue_max_queue_wait_ms: 30_000,
            queue_max_overtakes_per_waiter: 4,
            device_startup_records: Value::Null,
            close: test_close_evidence(),
        },
        selected_queries: Vec::new(),
        arrival_plan: super::artifact::ArrivalPlanEvidence::new(
            &ArrivalPlan::plan(&profile.scheduling, 1, 1).unwrap(),
            0,
            0,
            0,
            0,
        ),
        cache_state: "disabled",
        composed_schedules: Vec::new(),
        cache_same_key_probe: None,
        qps_envelope: None,
        graph_claim_calibration: None,
        aggregate,
        deadline: test_deadline_evidence(),
        cancellation_barrier: None,
        deadline_barrier: None,
        domain_exhaustion: None,
        physical_allocator_pressure: None,
        quarantine_cycles: None,
        close_under_load: None,
        completion_sink: None,
        feature_off_cugraph: None,
        metrics_audit: measured_metrics_audit(observations.len()),
        operability: super::artifact::OperabilityEvidence {
            attribution: super::operability::AttributionCompleteness::from_observations(&observations),
            otlp_export: super::operability::OtlpExportEvidence::not_requested(),
        },
        time_series: super::artifact::TimeSeriesEvidence::empty(),
        snapshots: Vec::new(),
        trace: super::artifact::TraceEvidence { captured_by_runner: false, uri: None, sha256: None },
    };
    let artifact = RunArtifact::measured_run(profile, "a".repeat(64), Vec::new(), sample_provenance(), measurement);
    let value = serde_json::to_value(artifact).unwrap();
    assert_eq!(value["status"], "measured_run_non_qualifying");
}

/// P6's C=1 composed calibration and P7/P8 contention share the same
/// schedule evidence, but only the latter explicitly declares a bounded
/// overload result. The branch matrix keeps every other composed-artifact
/// failure fail-closed while ensuring a workload label cannot invent that
/// later-phase requirement.
#[test]
fn composed_qualification_evidence_is_fail_closed_and_overload_is_profile_declared() {
    let cases = [
        ("isolated_p6_without_overload_contract", None, true, true, true, None, false, &[][..]),
        ("missing_root_source_hash", Some("root"), true, true, true, None, false, &["missing_source_hash"][..]),
        ("missing_cudf_source_hash", Some("cudf"), true, true, true, None, false, &["missing_source_hash"][..]),
        ("missing_cugraph_source_hash", Some("cugraph"), true, true, true, None, false, &["missing_source_hash"][..]),
        ("missing_binary_hash", None, false, true, true, None, false, &["missing_binary_hash"][..]),
        (
            "missing_physical_memory_sample",
            None,
            true,
            false,
            true,
            None,
            false,
            &["missing_physical_memory_sample"][..],
        ),
        (
            "missing_device_witness",
            None,
            true,
            true,
            false,
            None,
            false,
            &["missing_device_witness", "missing_physical_memory_sample"][..],
        ),
        ("missing_schedule", None, true, true, true, None, false, &["missing_composed_schedule"][..]),
        (
            "undeclared_failed_overload_is_not_a_requirement",
            None,
            true,
            true,
            true,
            Some(ClauseStatus::Failed),
            false,
            &[][..],
        ),
        (
            "declared_missing_overload_fails",
            None,
            true,
            true,
            true,
            None,
            true,
            &["missing_required_overload_outcome"][..],
        ),
        (
            "declared_failed_overload_fails",
            None,
            true,
            true,
            true,
            Some(ClauseStatus::Failed),
            true,
            &["missing_required_overload_outcome"][..],
        ),
        ("declared_evidenced_overload_passes", None, true, true, true, Some(ClauseStatus::Evidenced), true, &[][..]),
    ];

    for (name, missing_source, binary_present, physical_sample, device_witness, overload, require_overload, expected) in
        cases
    {
        let mut provenance = sample_provenance();
        match missing_source {
            Some("root") => provenance.repositories.root.dirty_manifest.sha256.clear(),
            Some("cudf") => provenance.repositories.cudf_nexus.dirty_manifest.sha256.clear(),
            Some("cugraph") => provenance.repositories.cugraph_nexus.dirty_manifest.sha256.clear(),
            None => {},
            Some(other) => panic!("unexpected source selector {other}"),
        }
        if !binary_present {
            provenance.runner.executable.sha256.clear();
        }

        let artifact = RunArtifact::measured_run(
            composed_qualification_profile(require_overload),
            "a".repeat(64),
            Vec::new(),
            provenance,
            composed_qualification_measurement(name != "missing_schedule", physical_sample, device_witness, overload),
        );
        let value = serde_json::to_value(artifact).unwrap();
        let actual = value["execution"]["qualification_failures"]
            .as_array()
            .unwrap()
            .iter()
            .map(|failure| failure["code"].as_str().unwrap())
            .collect::<BTreeSet<_>>();
        let expected = expected.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(actual, expected, "{name}");
        assert_eq!(
            value["status"],
            if expected.is_empty() { "measured_run" } else { "measured_run_non_qualifying" },
            "{name}"
        );
    }
}

fn composed_qualification_profile(require_overload_outcome: bool) -> RunProfile {
    let mut profile = measured_run_profile();
    if require_overload_outcome {
        let mut json = serde_json::to_value(&profile).unwrap();
        json["scenario_requirements"] = serde_json::json!({ "require_overload_outcome": true });
        profile = RunProfile::parse_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    }
    profile
}

fn composed_qualification_measurement(
    schedule_present: bool,
    physical_sample: bool,
    device_witness: bool,
    overload: Option<ClauseStatus>,
) -> super::artifact::MeasuredRunArtifact {
    let mut schedule = super::artifact::ComposedScheduleArtifact::test_complete();
    schedule.attempt_id = "composed-attempt".to_string();
    schedule.grant_id = 17;

    let mut observations = measured_observations();
    let composed_attempt = &mut observations[0];
    composed_attempt.query_kind = FixtureKind::Dblp;
    composed_attempt.query_label = "pagerank_forceatlas2_join".to_string();
    composed_attempt.graph_workload = true;
    composed_attempt.attempt_id = schedule.attempt_id.clone();
    composed_attempt.device_ordinal = Some(schedule.selected_ordinal);
    composed_attempt.grant_bytes = Some(schedule.grant_bytes);
    composed_attempt.grant_id = Some(schedule.grant_id);

    let devices = if device_witness {
        vec![measured_device(schedule.selected_ordinal, 1 << 30, 2)]
    } else {
        vec![measured_device(schedule.selected_ordinal + 1, 1 << 30, 2)]
    };
    let mut aggregate = MeasuredAggregate::reduce(
        &observations,
        &devices,
        &BTreeMap::from([(schedule.selected_ordinal, measured_occupancy(1))]),
        0,
        1_000_000_000,
    );
    if let Some(status) = overload {
        let mut evidence = OverloadFairnessEvidence::from_observations(&[], &[], 0, 0, 0, 0, 0, 0, 0);
        evidence.status = status;
        aggregate.scenarios.overload_fairness = Some(evidence);
    }

    let mut time_series = super::artifact::TimeSeriesEvidence::empty();
    if physical_sample {
        time_series.device_rows.push(DeviceSeriesRow {
            offset_ns: 1,
            device_ordinal: schedule.selected_ordinal,
            health_open: true,
            gpu_utilization_pct: Some(1),
            memory_utilization_pct: Some(1),
            cuda_free_bytes: Some(1),
            nvml_used_bytes: Some(1),
            active_attempts: 1,
            managed_capacity_bytes: schedule.grant_bytes,
            live_grants_bytes: schedule.grant_bytes,
            live_reserved_bytes: schedule.grant_bytes,
            cache_evictable_resident_bytes: 0,
            cache_pinned_in_use_bytes: 0,
            cache_physical_current_bytes: None,
            cache_physical_peak_bytes: None,
            allocation_failure_count: 0,
            ledger_sequence: 1,
            claim_snapshot_end_sequence: Some(1),
            claim_snapshot_stable: true,
            active_query_jobs: 1,
            queued_jobs: 0,
        });
    }

    super::artifact::MeasuredRunArtifact {
        qualifying: true,
        backend: super::artifact::BackendEvidence {
            resource_core: 1,
            observation_core: 2,
            resource_generation: 3,
            shared_session_count: 1,
            queue_max_queued_attempts: 64,
            queue_max_queue_wait_ms: 30_000,
            queue_max_overtakes_per_waiter: 4,
            device_startup_records: serde_json::to_value(&devices).unwrap(),
            close: test_close_evidence(),
        },
        selected_queries: vec![SelectedQueryDescriptor {
            fixture_index: 0,
            kind: FixtureKind::Dblp,
            label: "pagerank_forceatlas2_join".to_string(),
            workload_class: "graph",
        }],
        arrival_plan: super::artifact::ArrivalPlanEvidence::new(
            &ArrivalPlan::plan(&measured_run_profile().scheduling, 1, 1).unwrap(),
            0,
            0,
            0,
            0,
        ),
        cache_state: "disabled",
        composed_schedules: if schedule_present { vec![schedule] } else { Vec::new() },
        cache_same_key_probe: None,
        qps_envelope: None,
        graph_claim_calibration: None,
        aggregate,
        deadline: test_deadline_evidence(),
        cancellation_barrier: None,
        deadline_barrier: None,
        domain_exhaustion: None,
        physical_allocator_pressure: None,
        quarantine_cycles: None,
        close_under_load: None,
        completion_sink: None,
        feature_off_cugraph: None,
        metrics_audit: measured_metrics_audit(observations.len()),
        operability: super::artifact::OperabilityEvidence {
            attribution: super::operability::AttributionCompleteness::from_observations(&observations),
            otlp_export: super::operability::OtlpExportEvidence::not_requested(),
        },
        time_series,
        snapshots: Vec::new(),
        trace: super::artifact::TraceEvidence { captured_by_runner: false, uri: None, sha256: None },
    }
}

fn measured_metrics_audit(reported_attempts: usize) -> MetricsAudit {
    let snapshot = |offset_ns, completed| MetricSnapshotObservation {
        offset_ns,
        backend_generation: 3,
        generation: offset_ns,
        configured_device_ordinals: vec![0, 1],
        // Eight ordinal-labelled descriptor series per configured device plus
        // twenty terminal counters and eighty duration histograms.
        projected_series_count: Some(116),
        expected_series_count: 116,
        terminal_completed: completed,
        total_duration_bucket_counts: vec![0, 0, 0, completed, completed],
        total_duration_count: completed,
    };
    MetricsAudit::from_snapshots(
        &[snapshot(0, 0), snapshot(250_000_000, reported_attempts as u64)],
        &[0, 1],
        reported_attempts,
        250,
        false,
    )
}

fn test_deadline_evidence() -> super::artifact::RunDeadlineEvidence {
    super::artifact::RunDeadlineEvidence {
        measurement_window_ns: 2_000_000_000,
        completion_drain_grace_ns: 0,
        configured_window_ns: 2_000_000_000,
        elapsed_until_reports_ns: 2_000_000_000,
        overrun_ns: 0,
        reports_sealed_within_deadline: true,
    }
}

fn test_close_evidence() -> super::artifact::BackendCloseEvidence {
    use super::artifact::{BackendCloseEvidence, BackendCloseOutcome, BackendCloseSnapshotEvidence};

    let snapshot = BackendCloseSnapshotEvidence {
        closing: true,
        gpu_resource_quiescent: true,
        observation_quiescent: true,
        active_gpu_permits: 0,
        active_observation_permits: 0,
        cleanup_failed: false,
        cleanup_deadline_elapsed_count: 0,
        cleanup_late_success_count: 0,
    };
    BackendCloseEvidence { before: snapshot, outcome: BackendCloseOutcome::Quiescent { snapshot }, after: snapshot }
}

fn measured_run_profile() -> RunProfile {
    let mut profile = RunProfile::parse_bytes(valid_profile_json().as_bytes()).unwrap();
    profile.backend.device_profiles = vec![
        measured_device_profile(0, 1 << 30, 2),
        // A deliberately different envelope, so heterogeneity is evidenced
        // rather than assumed.
        measured_device_profile(1, 2 << 30, 4),
    ];
    profile.scheduling = SchedulingProfile {
        arrival_mode: ArrivalMode::Concurrency,
        concurrency: 2,
        arrival_rate_per_second: None,
        max_rounds: None,
        seed: 7,
        warmup_seconds: 1,
        measurement_seconds: 2,
        rate_overrun: RateOverrunPolicy::Delay,
        steps: Vec::new(),
        attempt_deadline_ms: None,
        cancel_every_n: None,
        cancel_after_ms: None,
    };
    profile.cache.mode = CacheMode::Disabled;
    profile.workloads = vec![
        FixtureWorkloadProfile {
            kind: FixtureKind::Tpch,
            path: PathBuf::from("/nonexistent/tpch"),
            scale: "sf1".to_string(),
            preparation_command: "fixture/fixture.sh tpch generate --scale 1 --out-root /data/tpch".to_string(),
            queries: vec!["q01".to_string(), "q06".to_string()],
        },
        FixtureWorkloadProfile {
            kind: FixtureKind::Dblp,
            path: PathBuf::from("/nonexistent/dblp"),
            scale: "v12".to_string(),
            preparation_command:
                "fixture/fixture.sh graph dblp-ingest --source-json /data/dblp.v12.json --out-root /data/demo"
                    .to_string(),
            queries: vec!["pagerank".to_string(), "edge_scan".to_string()],
        },
    ];
    profile.validate().unwrap();
    profile
}

fn measured_device_profile(ordinal: usize, memory_limit_bytes: u64, max_active_attempts: usize) -> DeviceProfile {
    DeviceProfile {
        ordinal,
        ownership: DeviceOwnership::WholeDeviceExclusive,
        memory_limit_bytes,
        backend_reserve_bytes: 64 * 1024 * 1024,
        cache_cap_bytes: 0,
        attempt_floor_bytes: 128 * 1024 * 1024,
        source_work_per_attempt: 2,
        max_active_attempts,
        install_telemetry: true,
        // The engine needs max_active_attempts x (source_work_per_attempt + 1)
        // stream slots, so size the pool from the same formula.
        installation: InstallationBounds {
            stream_pool_size: max_active_attempts * 3,
            worker_count: max_active_attempts,
            lane_count: max_active_attempts,
        },
    }
}

fn measured_device(ordinal: usize, capacity_bytes: u64, max_active_attempts: usize) -> DeviceConfiguration {
    DeviceConfiguration {
        ordinal,
        model_name: Some("NVIDIA L40S".to_string()),
        uuid: Some(format!("GPU-{ordinal}")),
        physical_total_bytes: Some(46_068 * 1024 * 1024),
        capacity_bytes,
        cache_cap_bytes: 0,
        install_telemetry: true,
        max_active_attempts,
        stream_pool_size: 8,
        worker_count: 2,
        lane_count: 2,
        ownership: "WholeDeviceExclusive".to_string(),
        device_health_generation: 0,
        starting_health_open: true,
    }
}

fn measured_occupancy(peak_active_query_jobs: usize) -> DeviceOccupancyPeaks {
    DeviceOccupancyPeaks {
        peak_active_attempts: 2,
        peak_active_query_jobs,
        peak_live_grants_bytes: 2 * 1024 * 1024,
        peak_live_reserved_bytes: 4 * 1024 * 1024,
        allocation_failure_count: 0,
        min_root_streams_available: 6,
        min_root_worker_slots_available: 1,
    }
}

/// Two overlapping attempts on each of two ordinals, all measurement phase.
fn measured_observations() -> Vec<AttemptObservation> {
    (0..4)
        .map(|index| {
            let ordinal = index % 2;
            AttemptObservation {
                sequence: index,
                phase: SchedulePhase::Measurement,
                step_index: 0,
                query_kind: if index >= 2 {
                    super::profile::FixtureKind::Dblp
                } else {
                    super::profile::FixtureKind::Tpch
                },
                query_label: format!("q{index:02}"),
                graph_workload: index >= 2,
                placement_required: true,
                attempt_id: format!("attempt-{index:032x}"),
                disposition: "Native".to_string(),
                outcome: "succeeded".to_string(),
                failure: None,
                terminal_evidence: terminal_evidence(),
                succeeded: true,
                reached_gpu: true,
                graph_claim_calibration: None,
                composed_schedule: None,
                device_ordinal: Some(ordinal),
                grant_bytes: Some(1024 * (index as u64 + 1)),
                admission_sequence: Some(index as u64),
                admission_wait_ns: Some(1_000),
                admission_requested_ns: Some(0),
                admitted_ns: Some(1),
                admission_wait: None,
                grant_id: Some(index as u64 + 10),
                domain_id: Some(index as u64 + 100),
                total_ns: Some(1_000_000 * (index as u64 + 1)),
                queue_wait_ns: Some(1_000),
                physical_execution_ns: Some(500_000),
                // Both attempts on an ordinal overlap in time.
                submitted_ns: 0,
                outcome_terminal_ns: 1_000_000 * (index as u64 + 1),
            }
        })
        .collect()
}

fn terminal_evidence() -> AttemptTerminalEvidence {
    AttemptTerminalEvidence {
        primary_failure: None,
        cleanup_disposition: serde_json::json!("completed"),
        cleanup_failure_count: 0,
        cleanup_failure_count_complete: true,
        first_cleanup_failure: None,
        capacity_disposition: serde_json::json!("released"),
        health_transition: Some(serde_json::json!("healthy_reuse")),
        native_fault_evidence: None,
        resource_evidence: None,
        retry_advice: serde_json::json!({ "recovery": "do_not_retry", "prerequisite": "none" }),
    }
}
