//! Versioned P6 production-validation run artifacts.
//!
//! P6.3 deliberately owns only profile parsing, provenance capture, and the
//! no-workload dry run. Fixture preflight is P6.4 and public attempt execution
//! is P6.5, so this module must not construct a backend or touch a fixture.

use std::{
    collections::BTreeSet,
    env,
    fs::{self, File},
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::ToolResult;

const RUN_PROFILE_SCHEMA: &str = "datafusion_nexus.single_node_multi_gpu.run_profile.v1";
const RUN_PROFILE_VERSION: u32 = 1;
const RUN_ARTIFACT_SCHEMA: &str = "datafusion_nexus.single_node_multi_gpu.run_artifact.v1";
const RUN_ARTIFACT_VERSION: u32 = 1;
const REDACTION_POLICY: &str = "datafusion_nexus.single_node_multi_gpu.redaction.v1";
const ARTIFACT_FILE_NAME: &str = "run-artifact.json";
const MAX_TOOL_VERSION_BYTES: usize = 16 * 1024;

pub(crate) fn run(args: Vec<String>) -> ToolResult<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return Ok(());
    }

    let options = CliOptions::parse(args)?;
    let (profile, config_sha256) = RunProfile::read(&options.config)?;
    let provenance = Provenance::collect()?;
    let artifact = RunArtifact::new(profile, config_sha256, provenance);
    let encoded = serde_json::to_vec_pretty(&artifact)?;
    ensure_redacted_json(&serde_json::from_slice(&encoded)?, "")?;
    let artifact_path = write_new_artifact(&options.out_dir, &encoded)?;

    println!("single-node-multi-gpu validation dry run: {}", artifact_path.display());
    Ok(())
}

fn print_usage() {
    eprintln!(
        "single-node-multi-gpu-validate\n\
         \n\
         USAGE:\n\
           single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --dry-run\n\
         \n\
         P6.3 writes a provenance-only artifact. Fixture preflight and workload execution begin in P6.4 and P6.5."
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    config: PathBuf,
    out_dir: PathBuf,
}

impl CliOptions {
    fn parse(args: Vec<String>) -> ToolResult<Self> {
        let mut config = None;
        let mut out_dir = None;
        let mut dry_run = false;
        let mut seen = BTreeSet::new();
        let mut index = 0;

        while index < args.len() {
            let flag = &args[index];
            if !seen.insert(flag.clone()) {
                return Err(format!("duplicate option {flag}").into());
            }
            match flag.as_str() {
                "--config" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    config = Some(PathBuf::from(value));
                },
                "--out-dir" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    out_dir = Some(PathBuf::from(value));
                },
                "--dry-run" => dry_run = true,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 1;
        }

        if !dry_run {
            return Err(
                "P6.3 currently requires --dry-run; fixture preflight and workload execution begin in P6.4 and P6.5"
                    .into(),
            );
        }
        let config = config.ok_or("missing required option --config")?;
        let out_dir = out_dir.ok_or("missing required option --out-dir")?;
        if out_dir.as_os_str().is_empty() {
            return Err("--out-dir must not be empty".into());
        }
        Ok(Self { config, out_dir })
    }
}

fn required_option_value(args: &[String], index: &mut usize, flag: &str) -> ToolResult<String> {
    *index += 1;
    let value = args.get(*index).ok_or_else(|| format!("missing value for {flag}"))?;
    if value.starts_with("--") {
        return Err(format!("missing value for {flag}").into());
    }
    Ok(value.clone())
}

/// The v1 profile intentionally fixes the top-level sections P6.5 will consume
/// while allowing P6.3 to prove the zero-workload artifact path. Workload
/// detail stays opaque only until P6.5 owns the public workload grammar; P6.3
/// rejects non-empty selections before persisting any value.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RunProfile {
    schema: String,
    version: u32,
    name: String,
    #[serde(default)]
    backend: BackendProfile,
    #[serde(default)]
    session: SessionProfile,
    #[serde(default)]
    scheduling: SchedulingProfile,
    #[serde(default)]
    cache: CacheProfile,
    #[serde(default)]
    workloads: Vec<Value>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BackendProfile {
    #[serde(default)]
    device_profiles: Vec<DeviceProfile>,
    #[serde(default)]
    queue: QueueProfile,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DeviceProfile {
    ordinal: usize,
    ownership: DeviceOwnership,
    memory_limit_bytes: u64,
    backend_reserve_bytes: u64,
    cache_cap_bytes: u64,
    attempt_floor_bytes: u64,
    source_work_per_attempt: usize,
    max_active_attempts: usize,
    installation: InstallationBounds,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum DeviceOwnership {
    WholeDeviceExclusive,
    StaticPartition { allocation_limit_bytes: u64 },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct InstallationBounds {
    stream_pool_size: usize,
    worker_count: usize,
    lane_count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct QueueProfile {
    #[serde(default = "default_max_queued_attempts")]
    max_queued_attempts: usize,
    #[serde(default = "default_max_queue_wait_ms")]
    max_queue_wait_ms: u64,
    #[serde(default = "default_max_overtakes_per_waiter")]
    max_overtakes_per_waiter: u32,
}

impl Default for QueueProfile {
    fn default() -> Self {
        Self {
            max_queued_attempts: default_max_queued_attempts(),
            max_queue_wait_ms: default_max_queue_wait_ms(),
            max_overtakes_per_waiter: default_max_overtakes_per_waiter(),
        }
    }
}

const fn default_max_queued_attempts() -> usize {
    64
}

const fn default_max_queue_wait_ms() -> u64 {
    30_000
}

const fn default_max_overtakes_per_waiter() -> u32 {
    4
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SessionProfile {
    #[serde(default = "default_shared_session_count")]
    shared_session_count: usize,
}

impl Default for SessionProfile {
    fn default() -> Self {
        Self { shared_session_count: default_shared_session_count() }
    }
}

const fn default_shared_session_count() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SchedulingProfile {
    #[serde(default = "default_arrival_mode")]
    arrival_mode: ArrivalMode,
    #[serde(default = "default_concurrency")]
    concurrency: usize,
    #[serde(default)]
    arrival_rate_per_second: Option<u64>,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    warmup_seconds: u64,
    #[serde(default)]
    measurement_seconds: u64,
}

impl Default for SchedulingProfile {
    fn default() -> Self {
        Self {
            arrival_mode: default_arrival_mode(),
            concurrency: default_concurrency(),
            arrival_rate_per_second: None,
            seed: 0,
            warmup_seconds: 0,
            measurement_seconds: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ArrivalMode {
    Concurrency,
    Rate,
}

const fn default_arrival_mode() -> ArrivalMode {
    ArrivalMode::Concurrency
}

const fn default_concurrency() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CacheProfile {
    #[serde(default)]
    mode: CacheMode,
}

impl Default for CacheProfile {
    fn default() -> Self {
        Self { mode: CacheMode::Disabled }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CacheMode {
    #[default]
    Disabled,
    Enabled,
}

impl RunProfile {
    fn read(path: &Path) -> ToolResult<(Self, String)> {
        let bytes = fs::read(path).map_err(|_| format!("cannot read --config {}", path.display()))?;
        let profile = Self::parse_bytes(&bytes)?;
        Ok((profile, sha256_bytes(&bytes)))
    }

    fn parse_bytes(bytes: &[u8]) -> ToolResult<Self> {
        let raw: Value = serde_json::from_slice(bytes).map_err(|_| "--config must be valid JSON")?;
        ensure_redacted_json(&raw, "")?;
        let profile: Self = serde_json::from_value(raw).map_err(|error| format!("invalid run profile: {error}"))?;
        profile.validate()?;
        Ok(profile)
    }

    fn validate(&self) -> ToolResult<()> {
        if self.schema != RUN_PROFILE_SCHEMA {
            return Err(format!("run profile schema must be {RUN_PROFILE_SCHEMA}").into());
        }
        if self.version != RUN_PROFILE_VERSION {
            return Err(format!("run profile version must be {RUN_PROFILE_VERSION}").into());
        }
        if !is_profile_name(&self.name) {
            return Err(
                "run profile name must be a 1-64 character ASCII identifier using letters, digits, '.', '_' or '-'"
                    .into(),
            );
        }
        if self.session.shared_session_count == 0 {
            return Err("session.shared_session_count must be greater than zero".into());
        }
        if self.scheduling.concurrency == 0 {
            return Err("scheduling.concurrency must be greater than zero".into());
        }
        match (self.scheduling.arrival_mode.clone(), self.scheduling.arrival_rate_per_second) {
            (ArrivalMode::Concurrency, None) => {},
            (ArrivalMode::Concurrency, Some(_)) => {
                return Err(
                    "scheduling.arrival_rate_per_second is only valid when scheduling.arrival_mode is rate".into()
                );
            },
            (ArrivalMode::Rate, Some(rate)) if rate > 0 => {},
            (ArrivalMode::Rate, _) => {
                return Err("rate scheduling requires a non-zero scheduling.arrival_rate_per_second".into());
            },
        }
        validate_device_profiles(&self.backend.device_profiles)?;
        if !self.workloads.is_empty() {
            return Err("P6.3 accepts only a no-workload profile; fixture preflight and workload execution begin in P6.4 and P6.5".into());
        }
        Ok(())
    }
}

fn is_profile_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn validate_device_profiles(device_profiles: &[DeviceProfile]) -> ToolResult<()> {
    let mut ordinals = BTreeSet::new();
    for profile in device_profiles {
        if !ordinals.insert(profile.ordinal) {
            return Err(format!("backend.device_profiles contains duplicate ordinal {}", profile.ordinal).into());
        }
        if profile.memory_limit_bytes == 0
            || profile.attempt_floor_bytes == 0
            || profile.source_work_per_attempt == 0
            || profile.max_active_attempts == 0
            || profile.installation.stream_pool_size == 0
            || profile.installation.worker_count == 0
            || profile.installation.lane_count == 0
        {
            return Err(
                format!("backend.device_profiles[{}] contains a zero device-local bound", profile.ordinal).into()
            );
        }
        if let DeviceOwnership::StaticPartition { allocation_limit_bytes } = profile.ownership
            && allocation_limit_bytes == 0
        {
            return Err(format!(
                "backend.device_profiles[{}].ownership allocation_limit_bytes must be greater than zero",
                profile.ordinal
            )
            .into());
        }
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct RunArtifact {
    schema: &'static str,
    version: u32,
    status: &'static str,
    created_unix_ms: u128,
    profile: ArtifactProfile,
    execution: DryRunExecution,
    provenance: Provenance,
    redaction: RedactionEvidence,
}

impl RunArtifact {
    fn new(profile: RunProfile, config_sha256: String, provenance: Provenance) -> Self {
        Self {
            schema: RUN_ARTIFACT_SCHEMA,
            version: RUN_ARTIFACT_VERSION,
            status: "dry_run",
            created_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_millis()),
            profile: ArtifactProfile { profile, sha256: config_sha256 },
            execution: DryRunExecution {
                fixture_preflight: "not_applicable_p6_4",
                workload_execution: "not_applicable_p6_5",
                attempted_workloads: 0,
                attempted_backend_attempts: 0,
            },
            provenance,
            redaction: RedactionEvidence {
                policy: REDACTION_POLICY,
                raw_config_persisted: false,
                sensitive_keys_rejected: true,
                raw_sql_rejected: true,
                raw_diffs_rejected: true,
                unbounded_errors_persisted: false,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ArtifactProfile {
    #[serde(flatten)]
    profile: RunProfile,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct DryRunExecution {
    fixture_preflight: &'static str,
    workload_execution: &'static str,
    attempted_workloads: u64,
    attempted_backend_attempts: u64,
}

#[derive(Debug, Serialize)]
struct RedactionEvidence {
    policy: &'static str,
    raw_config_persisted: bool,
    sensitive_keys_rejected: bool,
    raw_sql_rejected: bool,
    raw_diffs_rejected: bool,
    unbounded_errors_persisted: bool,
}

#[derive(Debug, Serialize)]
struct Provenance {
    repositories: RepositoryProvenanceSet,
    cuvs_build_inputs: CuvsBuildInputs,
    toolchains: ToolchainVersions,
    runner: RunnerIdentity,
}

impl Provenance {
    fn collect() -> ToolResult<Self> {
        let workspace_root = workspace_root()?;
        let sibling_root = workspace_root.parent().ok_or("workspace root has no parent for sibling provenance")?;
        let cudf_root = sibling_root.join("cudf-nexus");
        let cugraph_root = sibling_root.join("cugraph-nexus");
        Ok(Self {
            repositories: RepositoryProvenanceSet {
                root: RepositoryProvenance::collect("datafusion-nexus", &workspace_root)?,
                cudf_nexus: RepositoryProvenance::collect("cudf-nexus", &cudf_root)?,
                cugraph_nexus: RepositoryProvenance::collect("cugraph-nexus", &cugraph_root)?,
            },
            cuvs_build_inputs: CuvsBuildInputs::collect()?,
            toolchains: ToolchainVersions::collect()?,
            runner: RunnerIdentity::collect()?,
        })
    }
}

fn workspace_root() -> ToolResult<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root =
        manifest_dir.parent().and_then(Path::parent).ok_or("cannot derive workspace root from nexus-tools manifest")?;
    fs::canonicalize(workspace_root).map_err(|_| "cannot canonicalize workspace root".into())
}

#[derive(Debug, Serialize)]
struct RepositoryProvenanceSet {
    root: RepositoryProvenance,
    cudf_nexus: RepositoryProvenance,
    cugraph_nexus: RepositoryProvenance,
}

#[derive(Debug, Serialize)]
struct RepositoryProvenance {
    name: &'static str,
    path: String,
    revision: String,
    dirty_manifest: DirtyManifest,
}

impl RepositoryProvenance {
    fn collect(name: &'static str, path: &Path) -> ToolResult<Self> {
        let canonical_path = fs::canonicalize(path).map_err(|_| format!("cannot canonicalize {name} repository"))?;
        let revision = git_output(&canonical_path, ["rev-parse", "HEAD"], name)?;
        let dirty_manifest = DirtyManifest::collect(&canonical_path, name)?;
        Ok(Self { name, path: canonical_path.display().to_string(), revision, dirty_manifest })
    }
}

#[derive(Debug, Serialize)]
struct DirtyManifest {
    format: &'static str,
    sha256: String,
    entries: Vec<String>,
}

impl DirtyManifest {
    fn collect(repo: &Path, name: &str) -> ToolResult<Self> {
        let output = git_output_bytes(
            repo,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
            name,
        )?;
        let entries = output
            .split(|byte| *byte == b'\0')
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                std::str::from_utf8(entry)
                    .map(str::to_string)
                    .map_err(|_| format!("{name} dirty manifest contains non-UTF-8 paths"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { format: "git-status-porcelain-v1-z", sha256: sha256_bytes(&output), entries })
    }
}

fn git_output<const N: usize>(repo: &Path, args: [&str; N], name: &str) -> ToolResult<String> {
    let output = git_output_bytes(repo, args, name)?;
    String::from_utf8(output)
        .map(|value| value.trim().to_string())
        .map_err(|_| format!("{name} git output is not UTF-8").into())
}

fn git_output_bytes<const N: usize>(repo: &Path, args: [&str; N], name: &str) -> ToolResult<Vec<u8>> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .map_err(|_| format!("cannot execute git for {name}"))?;
    if !output.status.success() {
        return Err(format!("git command failed for {name}").into());
    }
    Ok(output.stdout)
}

#[derive(Debug, Serialize)]
struct CuvsBuildInputs {
    install_dir: String,
    cmake_config: FileIdentity,
    libcuvs_selector: FileIdentity,
    libcuvs_resolved: FileIdentity,
    version: String,
}

impl CuvsBuildInputs {
    fn collect() -> ToolResult<Self> {
        let install_dir = env::var_os("CUVS_INSTALL_DIR")
            .map(PathBuf::from)
            .ok_or("CUVS_INSTALL_DIR is required for production-validation provenance")?;
        let install_dir = fs::canonicalize(&install_dir).map_err(|_| "cannot canonicalize CUVS_INSTALL_DIR")?;
        let cmake_config_path = install_dir.join("lib/cmake/cuvs/cuvs-config.cmake");
        let cmake_config = FileIdentity::collect(&cmake_config_path, "cuvs-config.cmake")?;
        let selector_path = [install_dir.join("lib/libcuvs.so"), install_dir.join("lib64/libcuvs.so")]
            .into_iter()
            .find(|path| path.is_file())
            .ok_or("CUVS_INSTALL_DIR must contain lib/libcuvs.so or lib64/libcuvs.so")?;
        let libcuvs_selector = FileIdentity::collect(&selector_path, "libcuvs.so selector")?;
        let resolved_path = fs::canonicalize(&selector_path).map_err(|_| "cannot resolve selected libcuvs.so")?;
        if !resolved_path.starts_with(&install_dir) {
            return Err("selected libcuvs.so resolves outside CUVS_INSTALL_DIR".into());
        }
        let libcuvs_resolved = FileIdentity::collect(&resolved_path, "resolved libcuvs.so")?;
        let config_bytes = fs::read(&cmake_config_path).map_err(|_| "cannot read cuvs-config.cmake")?;
        let version = cmake_set_value(&config_bytes, "CUVS_VERSION").ok_or("cuvs-config.cmake has no CUVS_VERSION")?;
        Ok(Self {
            install_dir: install_dir.display().to_string(),
            cmake_config,
            libcuvs_selector,
            libcuvs_resolved,
            version,
        })
    }
}

#[derive(Debug, Serialize)]
struct FileIdentity {
    path: String,
    size_bytes: u64,
    sha256: String,
}

impl FileIdentity {
    fn collect(path: &Path, label: &str) -> ToolResult<Self> {
        let metadata = fs::metadata(path).map_err(|_| format!("cannot stat {label}"))?;
        if !metadata.is_file() {
            return Err(format!("{label} is not a regular file").into());
        }
        Ok(Self { path: path.display().to_string(), size_bytes: metadata.len(), sha256: sha256_file(path)? })
    }
}

#[derive(Debug, Serialize)]
struct ToolchainVersions {
    rustc: String,
    cargo: String,
    cuda_toolkit: String,
    cuda_driver: String,
    nsight_systems: String,
    rapids: RapidsVersions,
    build_profile: &'static str,
}

impl ToolchainVersions {
    fn collect() -> ToolResult<Self> {
        Ok(Self {
            rustc: required_command_version("rustc", ["-Vv"])?,
            cargo: required_command_version("cargo", ["-V"])?,
            cuda_toolkit: required_command_version("nvcc", ["--version"])?,
            cuda_driver: required_command_version(
                "nvidia-smi",
                ["--query-gpu=driver_version", "--format=csv,noheader"],
            )?,
            nsight_systems: required_command_version("nsys", ["--version"])?,
            rapids: RapidsVersions::collect()?,
            build_profile: if cfg!(debug_assertions) { "debug" } else { "release" },
        })
    }
}

#[derive(Debug, Serialize)]
struct RapidsVersions {
    cudf: CmakePackageVersion,
    cugraph: SourcePackageVersion,
    cuvs: CmakePackageVersion,
}

impl RapidsVersions {
    fn collect() -> ToolResult<Self> {
        Ok(Self {
            cudf: CmakePackageVersion::collect("CUDF_INSTALL_DIR", "cudf", "CUDF_VERSION")?,
            cugraph: SourcePackageVersion::collect("cugraph-nexus", "cugraph/VERSION")?,
            cuvs: CmakePackageVersion::collect("CUVS_INSTALL_DIR", "cuvs", "CUVS_VERSION")?,
        })
    }
}

#[derive(Debug, Serialize)]
struct SourcePackageVersion {
    source: FileIdentity,
    version: String,
}

impl SourcePackageVersion {
    fn collect(sibling: &str, version_file: &str) -> ToolResult<Self> {
        let workspace_root = workspace_root()?;
        let sibling_root =
            workspace_root.parent().ok_or("workspace root has no parent for RAPIDS provenance")?.join(sibling);
        let version_path = sibling_root.join(version_file);
        let source = FileIdentity::collect(&version_path, "cuGraph VERSION")?;
        let version = fs::read_to_string(&version_path).map_err(|_| "cannot read cuGraph VERSION")?.trim().to_string();
        if version.is_empty() {
            return Err("cuGraph VERSION is empty".into());
        }
        Ok(Self { source, version })
    }
}

#[derive(Debug, Serialize)]
struct CmakePackageVersion {
    install_dir: String,
    config: FileIdentity,
    version: String,
}

impl CmakePackageVersion {
    fn collect(environment: &str, package: &str, version_variable: &str) -> ToolResult<Self> {
        let install_dir = env::var_os(environment)
            .map(PathBuf::from)
            .ok_or_else(|| format!("{environment} is required for RAPIDS provenance"))?;
        let install_dir = fs::canonicalize(&install_dir).map_err(|_| format!("cannot canonicalize {environment}"))?;
        let config_path = install_dir.join("lib").join("cmake").join(package).join(format!("{package}-config.cmake"));
        let config = FileIdentity::collect(&config_path, &format!("{package}-config.cmake"))?;
        let bytes = fs::read(&config_path).map_err(|_| format!("cannot read {package}-config.cmake"))?;
        let version = cmake_set_value(&bytes, version_variable)
            .ok_or_else(|| format!("{package}-config.cmake has no {version_variable}"))?;
        Ok(Self { install_dir: install_dir.display().to_string(), config, version })
    }
}

fn cmake_set_value(contents: &[u8], variable: &str) -> Option<String> {
    let text = std::str::from_utf8(contents).ok()?;
    let prefix = format!("set({variable} ");
    text.lines().find_map(|line| {
        let value = line.trim().strip_prefix(&prefix)?.trim();
        let value = value.strip_suffix(')')?.trim().trim_matches('"');
        (!value.is_empty()).then(|| value.to_string())
    })
}

fn required_command_version<const N: usize>(program: &str, args: [&str; N]) -> ToolResult<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|_| format!("cannot execute required provenance tool {program}"))?;
    if !output.status.success() {
        return Err(format!("required provenance tool {program} failed").into());
    }
    if output.stdout.len() > MAX_TOOL_VERSION_BYTES || output.stderr.len() > MAX_TOOL_VERSION_BYTES {
        return Err(format!("required provenance tool {program} produced an oversized version response").into());
    }
    let value = if output.stdout.is_empty() { output.stderr } else { output.stdout };
    let value = String::from_utf8(value)
        .map_err(|_| format!("required provenance tool {program} returned non-UTF-8 output"))?;
    let value = value.trim();
    if value.is_empty() {
        return Err(format!("required provenance tool {program} returned an empty version").into());
    }
    Ok(value.to_string())
}

#[derive(Debug, Serialize)]
struct RunnerIdentity {
    executable: FileIdentity,
}

impl RunnerIdentity {
    fn collect() -> ToolResult<Self> {
        let executable = env::current_exe().map_err(|_| "cannot resolve validation runner executable")?;
        Ok(Self { executable: FileIdentity::collect(&executable, "validation runner executable")? })
    }
}

fn sha256_file(path: &Path) -> ToolResult<String> {
    let mut file = File::open(path).map_err(|_| format!("cannot read {} for SHA-256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer).map_err(|_| format!("cannot hash {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(hex_digest(hasher.finalize().as_ref()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_digest(digest.as_ref())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn write_new_artifact(out_dir: &Path, contents: &[u8]) -> ToolResult<PathBuf> {
    if out_dir.exists() {
        return Err(format!(
            "--out-dir {} already exists; refuse to mix or overwrite run artifacts",
            out_dir.display()
        )
        .into());
    }
    let parent = output_parent(out_dir);
    if !parent.is_dir() {
        return Err(format!("--out-dir parent {} does not exist", parent.display()).into());
    }
    fs::create_dir(out_dir).map_err(|_| format!("cannot create --out-dir {}", out_dir.display()))?;
    let temporary = out_dir.join(format!(".{ARTIFACT_FILE_NAME}.{}.tmp", std::process::id()));
    let artifact = out_dir.join(ARTIFACT_FILE_NAME);
    let write_result = (|| -> ToolResult<()> {
        let mut file = File::options().write(true).create_new(true).open(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()?;
        fs::rename(&temporary, &artifact)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        let _ = fs::remove_dir(out_dir);
        return Err(error);
    }
    Ok(artifact)
}

fn output_parent(out_dir: &Path) -> &Path {
    out_dir.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."))
}

fn ensure_redacted_json(value: &Value, pointer: &str) -> ToolResult<()> {
    match value {
        Value::Object(values) => {
            for (key, value) in values {
                let field = json_pointer(pointer, key);
                if is_sensitive_key(key) {
                    return Err(
                        format!("sensitive field is forbidden in a production-validation artifact at {field}").into()
                    );
                }
                ensure_redacted_json(value, &field)?;
            }
        },
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                ensure_redacted_json(value, &format!("{pointer}/{index}"))?;
            }
        },
        Value::String(value) if has_inline_secret(value) => {
            return Err(
                format!("credential-like text is forbidden in a production-validation artifact at {pointer}").into()
            );
        },
        _ => {},
    }
    Ok(())
}

fn json_pointer(pointer: &str, field: &str) -> String {
    format!("{pointer}/{}", field.replace('~', "~0").replace('/', "~1"))
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase().replace(['-', '_'], "");
    [
        "password",
        "passwd",
        "secret",
        "token",
        "authorization",
        "credential",
        "apikey",
        "privatekey",
        "accesskey",
        "sessionkey",
        "cookie",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn has_inline_secret(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("://")
        && normalized
            .split("://")
            .nth(1)
            .is_some_and(|rest| rest.split('/').next().is_some_and(|authority| authority.contains('@')))
        || normalized.contains("-----begin") && normalized.contains("private key-----")
        || normalized.contains("bearer ")
        || normalized.contains("x-amz-signature=")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_profile_json() -> &'static str {
        include_str!("../fixtures/p6-dry-run-profile.json")
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
    }

    #[test]
    fn parser_rejects_duplicate_unknown_missing_and_non_dry_run_options() {
        for (args, expected) in [
            (vec!["--config", "a", "--config", "b", "--out-dir", "out", "--dry-run"], "duplicate option --config"),
            (vec!["--config", "a", "--out-dir", "out", "--unexpected", "--dry-run"], "unknown option --unexpected"),
            (vec!["--out-dir", "out", "--dry-run"], "missing required option --config"),
            (vec!["--config", "a", "--out-dir", "out"], "requires --dry-run"),
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
        for (replacement, expected) in
            [("datafusion_nexus.other.v1", "run profile schema"), ("2", "run profile version")]
        {
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
    fn profile_reader_rejects_missing_files_without_publishing_an_artifact() {
        let path = test_root("missing-profile").join("missing.json");
        let error = RunProfile::read(&path).unwrap_err();
        assert!(error.to_string().contains("cannot read --config"));
    }

    #[test]
    fn cmake_version_parser_uses_exact_variable() {
        let config = b"set(CUVS_VERSION_MAJOR 26)\nset(CUVS_VERSION 26.06.00)\n";
        assert_eq!(cmake_set_value(config, "CUVS_VERSION").as_deref(), Some("26.06.00"));
        assert_eq!(cmake_set_value(config, "MISSING"), None);
    }

    #[test]
    fn artifact_v1_has_explicit_dry_run_and_redaction_sections() {
        let profile = RunProfile::parse_bytes(valid_profile_json().as_bytes()).unwrap();
        let artifact = RunArtifact::new(profile, "a".repeat(64), sample_provenance());
        let value = serde_json::to_value(artifact).unwrap();
        assert_eq!(value["schema"], RUN_ARTIFACT_SCHEMA);
        assert_eq!(value["version"], RUN_ARTIFACT_VERSION);
        assert_eq!(value["status"], "dry_run");
        assert_eq!(value["execution"]["attempted_workloads"], 0);
        assert_eq!(value["execution"]["fixture_preflight"], "not_applicable_p6_4");
        assert_eq!(value["redaction"]["raw_config_persisted"], false);
        ensure_redacted_json(&value, "").unwrap();
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
}
