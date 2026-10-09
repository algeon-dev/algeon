use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::UNIX_EPOCH,
};

use super::*;
use crate::clickbench_benchmark::KVIKIO_COMPAT_MODE_ENV;

/// Facts recorded once per benchmark run so an artifact can be attributed to
/// the source revision, native build, GPU, driver, and process limits that
/// produced it. A fact the process cannot observe renders as `unknown`; the
/// line never substitutes a default for a missing fact, because a wrong
/// default (fd limit, cap, GPU) is what makes runs incomparable.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct RunProvenance {
    pub(super) revision: Option<String>,
    pub(super) worktree_dirty: Option<bool>,
    pub(super) cudf_source_revision: Option<String>,
    pub(super) cudf_source_dirty: Option<bool>,
    pub(super) cudf_install_lib_mtime_unix: Option<u64>,
    pub(super) driver_version: Option<String>,
    pub(super) cpu_model: Option<String>,
    pub(super) cpu_logical: Option<usize>,
    pub(super) fd_soft_limit: Option<u64>,
    pub(super) kvikio_compat_mode: Option<String>,
    pub(super) devices: Vec<DeviceProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DeviceProvenance {
    pub(super) ordinal: usize,
    pub(super) model_name: Option<String>,
    pub(super) uuid: Option<String>,
    pub(super) physical_total_bytes: Option<u64>,
    pub(super) cuda_total_bytes: u64,
    pub(super) managed_capacity_bytes: u64,
    /// Immutable per-query cap the ledger grants on this device.
    pub(super) default_query_limit_bytes: u64,
}

impl RunProvenance {
    /// Per-query cap resolved by the installed backend. An explicit budget is
    /// its own cap before installation; an automatic budget is unknown until
    /// the device is discovered.
    pub(super) fn query_cap_bytes(&self, device_budget: NativeTpchDeviceBudget) -> Option<u64> {
        match (self.devices.first(), device_budget) {
            (Some(device), _) => Some(device.default_query_limit_bytes),
            (None, NativeTpchDeviceBudget::Explicit { bytes }) => Some(bytes),
            (None, NativeTpchDeviceBudget::Automatic) => None,
        }
    }
}

const CUDF_INSTALL_LIB: &str = "target/native/cudf-install/lib/libcudf.so";
const CUDF_SOURCE_DIR: &str = "components/cudf/cudf";

impl RunProvenance {
    pub(super) fn collect(resources: &NativeTpchRuntimeResources) -> Self {
        let root = repo_root();
        let (revision, worktree_dirty) = git_identity(&root);
        let (cudf_source_revision, cudf_source_dirty) = git_identity(&root.join(CUDF_SOURCE_DIR));
        Self {
            revision,
            worktree_dirty,
            cudf_source_revision,
            cudf_source_dirty,
            cudf_install_lib_mtime_unix: file_mtime_unix(&root.join(CUDF_INSTALL_LIB)),
            driver_version: fs::read_to_string("/proc/driver/nvidia/version").ok().and_then(parse_driver_version),
            cpu_model: fs::read_to_string("/proc/cpuinfo").ok().and_then(parse_cpu_model),
            cpu_logical: std::thread::available_parallelism().ok().map(usize::from),
            fd_soft_limit: fs::read_to_string("/proc/self/limits").ok().and_then(parse_fd_soft_limit),
            kvikio_compat_mode: std::env::var(KVIKIO_COMPAT_MODE_ENV).ok(),
            devices: resources.device_provenance(),
        }
    }
}

pub(super) fn run_start_line(config: &NativeTpchBenchmarkConfig, provenance: &RunProvenance) -> String {
    let policy = config.native_memory_policy;
    let mut line = format!(
        "[native_tpch_benchmark] stage=run_start revision={} worktree_dirty={} cudf_source_revision={} cudf_source_dirty={} cudf_install_lib_mtime_unix={} rust_profile={} driver_version={} cpu_model={} cpu_logical={} cpu_profile={} fd_soft_limit={} kvikio_compat_mode={} dataset_mode={} query_count={} warmups={} iterations={} native_profile={} profile_source={} device_budget={} query_cap_bytes={} max_source_chunk_bytes={} max_row_groups_per_chunk={} source_chunk_planning_mode={} max_source_reads_in_flight={}",
        unknown_or(provenance.revision.as_deref()),
        unknown_or(provenance.worktree_dirty.map(|dirty| dirty.to_string()).as_deref()),
        unknown_or(provenance.cudf_source_revision.as_deref()),
        unknown_or(provenance.cudf_source_dirty.map(|dirty| dirty.to_string()).as_deref()),
        unknown_or(provenance.cudf_install_lib_mtime_unix.map(|mtime| mtime.to_string()).as_deref()),
        if cfg!(debug_assertions) { "dev" } else { "release" },
        unknown_or(provenance.driver_version.as_deref()),
        unknown_or(provenance.cpu_model.as_deref().map(stable_token).as_deref()),
        unknown_or(provenance.cpu_logical.map(|count| count.to_string()).as_deref()),
        cpu_profile_label(),
        unknown_or(provenance.fd_soft_limit.map(|limit| limit.to_string()).as_deref()),
        provenance.kvikio_compat_mode.as_deref().unwrap_or("unset"),
        config.dataset_mode.as_str(),
        config.query_ids.len(),
        config.warmup_iterations,
        config.iterations,
        policy.profile_name,
        policy.profile_source,
        policy.device_budget,
        unknown_or(provenance.query_cap_bytes(policy.device_budget).map(|bytes| bytes.to_string()).as_deref()),
        policy.max_source_chunk_bytes,
        policy.max_row_groups_per_chunk,
        policy.source_chunk_planning_mode.as_str(),
        config.max_source_reads_in_flight,
    );
    if provenance.devices.is_empty() {
        line.push_str(" gpu_count=unknown");
    } else {
        line.push_str(&format!(" gpu_count={}", provenance.devices.len()));
    }
    for device in &provenance.devices {
        line.push_str(&format!(
            " gpu{ordinal}_model={} gpu{ordinal}_uuid={} gpu{ordinal}_physical_total_bytes={} gpu{ordinal}_cuda_total_bytes={} gpu{ordinal}_managed_capacity_bytes={}",
            unknown_or(device.model_name.as_deref().map(stable_token).as_deref()),
            unknown_or(device.uuid.as_deref()),
            unknown_or(device.physical_total_bytes.map(|bytes| bytes.to_string()).as_deref()),
            device.cuda_total_bytes,
            device.managed_capacity_bytes,
            ordinal = device.ordinal,
        ));
    }
    line
}

fn cpu_profile_label() -> &'static str {
    // Mirrors the CPU baseline selection in `execution::cpu_session_config`.
    match std::env::var("ALGEON_TPCH_CPU_PROFILE").as_deref().map(str::to_ascii_lowercase).as_deref() {
        Ok("untuned") | Ok("report") => "report",
        _ => "tuned",
    }
}

fn unknown_or(value: Option<&str>) -> &str {
    value.filter(|value| !value.is_empty()).unwrap_or("unknown")
}

/// Whitespace inside a value would break the `key=value` line grammar that
/// the stderr log consumers split on.
fn stable_token(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join("_")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn git_identity(dir: &Path) -> (Option<String>, Option<bool>) {
    let revision = git_stdout(dir, &["rev-parse", "--short=12", "HEAD"]).map(|out| out.trim().to_string());
    let dirty = git_stdout(dir, &["status", "--porcelain", "--untracked-files=no"]).map(|out| !out.trim().is_empty());
    (revision, dirty)
}

fn git_stdout(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

fn file_mtime_unix(path: &Path) -> Option<u64> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    modified.duration_since(UNIX_EPOCH).ok().map(|duration| duration.as_secs())
}

pub(super) fn parse_driver_version(proc_version: String) -> Option<String> {
    let nvrm_line = proc_version.lines().find(|line| line.starts_with("NVRM version:"))?;
    nvrm_line
        .split_whitespace()
        .find(|token| {
            token.contains('.') && token.chars().all(|ch| ch.is_ascii_digit() || ch == '.') && !token.starts_with('.')
        })
        .map(str::to_string)
}

pub(super) fn parse_cpu_model(cpuinfo: String) -> Option<String> {
    cpuinfo
        .lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split_once(':'))
        .map(|(_, model)| model.trim().to_string())
        .filter(|model| !model.is_empty())
}

pub(super) fn parse_fd_soft_limit(limits: String) -> Option<u64> {
    let line = limits.lines().find(|line| line.starts_with("Max open files"))?;
    line.split_whitespace().nth(3)?.parse().ok()
}
