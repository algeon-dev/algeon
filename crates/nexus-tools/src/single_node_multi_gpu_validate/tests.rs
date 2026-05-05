use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::arrow::ArrowWriter;
use serde_json::Value;

use super::{
    aggregate::{
        AttemptObservation, DeviceConfiguration, DeviceOccupancyPeaks, MeasuredAggregate, scenario_clauses_pass,
    },
    profile::{
        ArrivalMode, CacheMode, DeviceOwnership, DeviceProfile, FixtureKind, FixtureWorkloadProfile,
        InstallationBounds, RateOverrunPolicy, SchedulingProfile,
    },
    provenance::{
        CmakePackageVersion, CuvsBuildInputs, DirtyManifest, FileIdentity, RapidsVersions, RepositoryProvenance,
        RepositoryProvenanceSet, RunnerIdentity, SourcePackageVersion, ToolchainVersions,
    },
    redact::ensure_redacted_json,
    schedule::{ArrivalPlan, SchedulePhase},
    util::{output_parent, write_new_artifact},
    workload::select_queries,
    *,
};

fn valid_profile_json() -> &'static str {
    include_str!("../../fixtures/p6-dry-run-profile.json")
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
    assert_eq!(options.config, PathBuf::from("run-profile.json"));
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
fn parser_rejects_duplicate_unknown_missing_and_non_dry_run_options() {
    for (args, expected) in [
        (vec!["--config", "a", "--config", "b", "--out-dir", "out", "--dry-run"], "duplicate option --config"),
        (vec!["--config", "a", "--out-dir", "out", "--unexpected", "--dry-run"], "unknown option --unexpected"),
        (vec!["--out-dir", "out", "--dry-run"], "missing required option --config"),
        (vec!["--config", "a", "--out-dir", "out"], "missing required validation mode"),
        (vec!["--config", "a", "--out-dir", "out", "--dry-run", "--fixture-preflight"], "mutually exclusive"),
        (vec!["--config", "a", "--out-dir", "out", "--measured-run", "--dry-run"], "mutually exclusive"),
        (
            vec!["--config", "a", "--out-dir", "out", "--dry-run", "--nsight-trace", "t"],
            "only valid with --measured-run",
        ),
        (vec!["--config", "--out-dir", "out", "--dry-run"], "missing value for --config"),
    ] {
        let error = CliOptions::parse(args.into_iter().map(str::to_string).collect()).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
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
    write_dblp_fixture(&dblp_root);
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
    write_dblp_fixture(&dblp_root);
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
    write_dblp_fixture(&dblp_root);
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

fn write_dblp_fixture(root: &Path) {
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
        write_tiny_graph_parquet(&parquet.join(format!("{table}.parquet")), rows(table));
    }
    let entry = |table: &str| {
        serde_json::json!({
            "bytes": fs::metadata(parquet.join(format!("{table}.parquet"))).unwrap().len(),
            "num_rows": rows(table),
            "num_row_groups": 1,
            "schema": ["value"]
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

fn write_tiny_graph_parquet(path: &Path, rows: usize) {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let values = (0..i64::try_from(rows).unwrap()).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))]).unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
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
            format: "git-status-porcelain-v1-z",
            sha256: "d".repeat(64),
            entries: Vec::new(),
        },
    };
    Provenance {
        repositories: RepositoryProvenanceSet {
            root: repository("datafusion-nexus"),
            cudf_nexus: repository("cudf-nexus"),
            cugraph_nexus: repository("cugraph-nexus"),
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
    // Every shipped ordinal is whole-device-exclusive on purpose: placement
    // never selects a `StaticPartition` device, so a static-partition ordinal
    // cannot evidence scenario 2's progress clause today.
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
    assert!(measured_run_preflight(&no_queries).unwrap_err().to_string().contains("at least one workload query"),);

    let mut no_devices = base.clone();
    no_devices.backend.device_profiles.clear();
    assert!(measured_run_preflight(&no_devices).unwrap_err().to_string().contains("device_profiles"));

    let mut no_window = base;
    no_window.scheduling.measurement_seconds = 0;
    assert!(measured_run_preflight(&no_window).unwrap_err().to_string().contains("measurement_seconds"));
}

/// The fixture paths in `measured_run_profile` do not exist, so preflight must
/// fail with the structured fixture reason rather than proceeding.
#[test]
fn measured_run_preflight_reports_a_missing_fixture_with_a_structured_reason() {
    let error = measured_run_preflight(&measured_run_profile()).unwrap_err().to_string();
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
fn profile_rejects_unknown_and_duplicate_workload_query_labels() {
    let mut profile = measured_run_profile();
    profile.workloads[0].queries = vec!["q01".to_string(), "q01".to_string()];
    assert!(profile.validate().unwrap_err().to_string().contains("more than once"));

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
fn profile_rejects_a_rate_overrun_policy_without_rate_scheduling() {
    let mut profile = measured_run_profile();
    profile.scheduling.rate_overrun = RateOverrunPolicy::Drop;
    assert!(profile.validate().unwrap_err().to_string().contains("rate_overrun"));
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
            close: Value::Null,
        },
        selected_queries: queries.iter().map(super::workload::SelectedQuery::descriptor).collect(),
        arrival_plan: super::artifact::ArrivalPlanEvidence::new(&plan, 0, 0, 0),
        cache_state: "disabled",
        aggregate,
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
            close: Value::Null,
        },
        selected_queries: Vec::new(),
        arrival_plan: super::artifact::ArrivalPlanEvidence::new(
            &ArrivalPlan::plan(&profile.scheduling, 1, 1).unwrap(),
            0,
            0,
            0,
        ),
        cache_state: "disabled",
        aggregate,
        snapshots: Vec::new(),
        trace: super::artifact::TraceEvidence { captured_by_runner: false, uri: None, sha256: None },
    };
    let artifact = RunArtifact::measured_run(profile, "a".repeat(64), Vec::new(), sample_provenance(), measurement);
    let value = serde_json::to_value(artifact).unwrap();
    assert_eq!(value["status"], "measured_run_non_qualifying");
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
        seed: 7,
        warmup_seconds: 1,
        measurement_seconds: 2,
        rate_overrun: RateOverrunPolicy::Delay,
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
                query_label: format!("q{index:02}"),
                graph_workload: index >= 2,
                attempt_id: format!("attempt-{index:032x}"),
                disposition: "Native".to_string(),
                outcome: "succeeded".to_string(),
                failure: None,
                succeeded: true,
                reached_gpu: true,
                device_ordinal: Some(ordinal),
                grant_bytes: Some(1024 * (index as u64 + 1)),
                admission_sequence: Some(index as u64),
                admission_wait_ns: Some(1_000),
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
