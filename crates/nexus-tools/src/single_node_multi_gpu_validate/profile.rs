//! Versioned P6 run-profile parsing and fail-closed validation.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{RUN_PROFILE_SCHEMA, RUN_PROFILE_VERSION, redact::ensure_redacted_json, util::sha256_bytes};
use crate::ToolResult;

/// Upper bound on attempts a single profile may schedule. It keeps the
/// per-attempt report vector bounded without capping any measured rate.
#[cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]
pub(super) const MAX_PLANNED_ATTEMPTS: usize = 100_000;

/// The v1 profile fixes every top-level section the runner consumes.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct RunProfile {
    pub(super) schema: String,
    pub(super) version: u32,
    pub(super) name: String,
    #[serde(default)]
    pub(super) backend: BackendProfile,
    #[serde(default)]
    pub(super) session: SessionProfile,
    #[serde(default)]
    pub(super) scheduling: SchedulingProfile,
    #[serde(default)]
    pub(super) cache: CacheProfile,
    #[serde(default)]
    pub(super) workloads: Vec<FixtureWorkloadProfile>,
}

/// A selected fixture and the command used to prepare it before measurement.
/// The runner records the command but never executes it.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct FixtureWorkloadProfile {
    pub(super) kind: FixtureKind,
    pub(super) path: PathBuf,
    pub(super) scale: String,
    pub(super) preparation_command: String,
    /// Query labels this fixture contributes to the measured mix. An empty
    /// list selects the fixture for preflight only.
    #[serde(default)]
    pub(super) queries: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(super) enum FixtureKind {
    Tpch,
    Dblp,
}

impl FixtureKind {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Tpch => "tpch",
            Self::Dblp => "dblp",
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct BackendProfile {
    #[serde(default)]
    pub(super) device_profiles: Vec<DeviceProfile>,
    #[serde(default)]
    pub(super) queue: QueueProfile,
    /// Per-attempt native execution envelope. Admission requires a bounded
    /// policy, so the runner never offers an unbounded variant.
    #[serde(default)]
    pub(super) native_execution: NativeExecutionProfile,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct DeviceProfile {
    pub(super) ordinal: usize,
    pub(super) ownership: DeviceOwnership,
    pub(super) memory_limit_bytes: u64,
    pub(super) backend_reserve_bytes: u64,
    pub(super) cache_cap_bytes: u64,
    pub(super) attempt_floor_bytes: u64,
    pub(super) source_work_per_attempt: usize,
    pub(super) max_active_attempts: usize,
    pub(super) installation: InstallationBounds,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum DeviceOwnership {
    WholeDeviceExclusive,
    StaticPartition { allocation_limit_bytes: u64 },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct InstallationBounds {
    pub(super) stream_pool_size: usize,
    pub(super) worker_count: usize,
    pub(super) lane_count: usize,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeExecutionProfile {
    #[serde(default = "default_device_budget_bytes")]
    pub(super) device_budget_bytes: u64,
    #[serde(default = "default_max_source_chunk_bytes")]
    pub(super) max_source_chunk_bytes: u64,
    #[serde(default = "default_max_row_groups_per_chunk")]
    pub(super) max_row_groups_per_chunk: usize,
    #[serde(default = "default_max_retry_attempts")]
    pub(super) max_retry_attempts: usize,
    #[serde(default = "default_min_retry_chunk_row_groups")]
    pub(super) min_retry_chunk_row_groups: usize,
}

impl Default for NativeExecutionProfile {
    fn default() -> Self {
        Self {
            device_budget_bytes: default_device_budget_bytes(),
            max_source_chunk_bytes: default_max_source_chunk_bytes(),
            max_row_groups_per_chunk: default_max_row_groups_per_chunk(),
            max_retry_attempts: default_max_retry_attempts(),
            min_retry_chunk_row_groups: default_min_retry_chunk_row_groups(),
        }
    }
}

const fn default_device_budget_bytes() -> u64 {
    1024 * 1024 * 1024
}

const fn default_max_source_chunk_bytes() -> u64 {
    256 * 1024 * 1024
}

const fn default_max_row_groups_per_chunk() -> usize {
    1024
}

const fn default_max_retry_attempts() -> usize {
    3
}

const fn default_min_retry_chunk_row_groups() -> usize {
    1
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct QueueProfile {
    #[serde(default = "default_max_queued_attempts")]
    pub(super) max_queued_attempts: usize,
    #[serde(default = "default_max_queue_wait_ms")]
    pub(super) max_queue_wait_ms: u64,
    #[serde(default = "default_max_overtakes_per_waiter")]
    pub(super) max_overtakes_per_waiter: u32,
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct SessionProfile {
    #[serde(default = "default_shared_session_count")]
    pub(super) shared_session_count: usize,
}

impl Default for SessionProfile {
    fn default() -> Self {
        Self { shared_session_count: default_shared_session_count() }
    }
}

const fn default_shared_session_count() -> usize {
    1
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct SchedulingProfile {
    #[serde(default = "default_arrival_mode")]
    pub(super) arrival_mode: ArrivalMode,
    #[serde(default = "default_concurrency")]
    pub(super) concurrency: usize,
    #[serde(default)]
    pub(super) arrival_rate_per_second: Option<u64>,
    /// Optional exact bound on closed-loop rounds. This is an explicit
    /// scenario shape, not a hidden drain margin: it leaves the remaining
    /// declared window for cooperative finalization.
    #[serde(default)]
    pub(super) max_rounds: Option<usize>,
    #[serde(default)]
    pub(super) seed: u64,
    #[serde(default)]
    pub(super) warmup_seconds: u64,
    #[serde(default)]
    pub(super) measurement_seconds: u64,
    /// How a rate-mode arrival that the queue cannot accept is treated. The
    /// artifact must record this: it is what separates offered from achieved
    /// arrival rate.
    #[serde(default)]
    pub(super) rate_overrun: RateOverrunPolicy,
}

impl Default for SchedulingProfile {
    fn default() -> Self {
        Self {
            arrival_mode: default_arrival_mode(),
            concurrency: default_concurrency(),
            arrival_rate_per_second: None,
            max_rounds: None,
            seed: 0,
            warmup_seconds: 0,
            measurement_seconds: 0,
            rate_overrun: RateOverrunPolicy::Delay,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum ArrivalMode {
    Concurrency,
    Rate,
}

/// Closed-loop (`delay`) keeps every planned arrival and lets it start late;
/// `drop` abandons an arrival whose planned instant has already passed.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum RateOverrunPolicy {
    #[default]
    Delay,
    Drop,
}

const fn default_arrival_mode() -> ArrivalMode {
    ArrivalMode::Concurrency
}

const fn default_concurrency() -> usize {
    1
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheProfile {
    #[serde(default)]
    pub(super) mode: CacheMode,
}

impl Default for CacheProfile {
    fn default() -> Self {
        Self { mode: CacheMode::Disabled }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum CacheMode {
    #[default]
    Disabled,
    Enabled,
}

impl RunProfile {
    pub(super) fn read(path: &Path) -> ToolResult<(Self, String)> {
        let bytes = fs::read(path).map_err(|_| format!("cannot read --config {}", path.display()))?;
        let profile = Self::parse_bytes(&bytes)?;
        Ok((profile, sha256_bytes(&bytes)))
    }

    pub(super) fn parse_bytes(bytes: &[u8]) -> ToolResult<Self> {
        let raw: Value = serde_json::from_slice(bytes).map_err(|_| "--config must be valid JSON")?;
        ensure_redacted_json(&raw, "")?;
        let profile: Self = serde_json::from_value(raw).map_err(|error| format!("invalid run profile: {error}"))?;
        profile.validate()?;
        Ok(profile)
    }

    /// Query labels selected across every workload, in profile order.
    pub(super) fn selected_query_count(&self) -> usize {
        self.workloads.iter().map(|workload| workload.queries.len()).sum()
    }

    /// Returns the cache-disabled half of an opt-in P6.8 paired calibration.
    ///
    /// This preserves every workload, scheduling, admission, and device field
    /// except the one P5 cache control: each ordinal-local `cache_cap_bytes`.
    /// It is intentionally derived rather than read from another profile so a
    /// cache calibration cannot compare two unrelated runs.
    pub(super) fn cache_disabled_companion(&self) -> ToolResult<Self> {
        if self.cache.mode != CacheMode::Enabled {
            return Err("--cache-calibration requires cache.mode enabled in --config".into());
        }
        let mut companion = self.clone();
        companion.cache.mode = CacheMode::Disabled;
        for device in &mut companion.backend.device_profiles {
            device.cache_cap_bytes = 0;
        }
        companion.validate()?;
        Ok(companion)
    }

    /// Validates the finite cohort used by the explicit P6.6 overload probe.
    ///
    /// This is deliberately a profile shape rather than a second admission
    /// policy: the backend still owns queueing, finite overtakes, placement,
    /// and report sealing. The runner only requires enough bounded offered
    /// work for those public facts to be falsifiable.
    pub(super) fn validate_overload_fairness_scenario(&self) -> ToolResult<()> {
        if self.backend.device_profiles.len() < 2 {
            return Err(
                "--overload-fairness requires at least two configured devices for unrelated-device progress".into()
            );
        }
        if self.scheduling.arrival_mode != ArrivalMode::Concurrency {
            return Err(
                "--overload-fairness requires scheduling.arrival_mode=concurrency for one bounded cohort".into()
            );
        }
        if self.scheduling.max_rounds != Some(1) {
            return Err("--overload-fairness requires scheduling.max_rounds=1 for one bounded drain cohort".into());
        }
        if self.scheduling.warmup_seconds != 0 {
            return Err(
                "--overload-fairness requires scheduling.warmup_seconds=0 so every cohort report is measured".into()
            );
        }
        if self.selected_query_count() < 2 {
            return Err(
                "--overload-fairness requires at least two selected workload queries with distinct plan shapes".into(),
            );
        }

        let active_capacity = self.backend.device_profiles.iter().try_fold(0_usize, |total, device| {
            total
                .checked_add(device.max_active_attempts)
                .ok_or("--overload-fairness configured active-attempt capacity overflows")
        })?;
        if self.scheduling.concurrency <= active_capacity {
            return Err(format!(
                "--overload-fairness requires scheduling.concurrency above configured active-attempt capacity {active_capacity}"
            )
            .into());
        }
        let bounded_cohort_capacity = active_capacity
            .checked_add(self.backend.queue.max_queued_attempts)
            .ok_or("--overload-fairness configured queue capacity overflows")?;
        if self.scheduling.concurrency > bounded_cohort_capacity {
            return Err(format!(
                "--overload-fairness scheduling.concurrency must not exceed active-plus-queue capacity {bounded_cohort_capacity}"
            )
            .into());
        }
        let required_queue_slots = usize::try_from(self.backend.queue.max_overtakes_per_waiter)
            .ok()
            .and_then(|overtakes| overtakes.checked_add(1))
            .ok_or("--overload-fairness finite-overtake queue requirement overflows")?;
        if self.backend.queue.max_queued_attempts < required_queue_slots {
            return Err(format!(
                "--overload-fairness requires backend.queue.max_queued_attempts of at least {required_queue_slots} for its finite-overtake witness"
            )
            .into());
        }
        Ok(())
    }

    /// Validates P6.6's fixed two-step physical allocator-pressure probe.
    ///
    /// The runner establishes real external CUDA pressure before `q01`, then
    /// releases it before the `edge_scan` recovery attempt. Profile validation
    /// owns the finite sequencing contract; the sealed report reducer later
    /// proves that the observed physical request remained below every immutable
    /// domain, grant, and manager cap.
    pub(super) fn validate_physical_allocator_pressure_scenario(&self) -> ToolResult<()> {
        if self.backend.device_profiles.len() != 1 {
            return Err(
                "--physical-allocator-pressure requires exactly one configured device for same-device recovery".into(),
            );
        }
        if !matches!(&self.backend.device_profiles[0].ownership, DeviceOwnership::StaticPartition { .. }) {
            return Err("--physical-allocator-pressure requires a static_partition device ownership envelope".into());
        }
        if self.cache.mode != CacheMode::Disabled {
            return Err("--physical-allocator-pressure requires cache.mode=disabled".into());
        }
        if self.scheduling.arrival_mode != ArrivalMode::Concurrency || self.scheduling.concurrency != 1 {
            return Err(
                "--physical-allocator-pressure requires scheduling.arrival_mode=concurrency with concurrency=1".into(),
            );
        }
        if self.scheduling.max_rounds != Some(2) {
            return Err("--physical-allocator-pressure requires scheduling.max_rounds=2".into());
        }
        let labels =
            self.workloads.iter().flat_map(|workload| workload.queries.iter().map(String::as_str)).collect::<Vec<_>>();
        if labels.as_slice() != ["q01", "edge_scan"] {
            return Err("--physical-allocator-pressure requires exactly q01 followed by edge_scan".into());
        }
        Ok(())
    }

    pub(super) fn validate(&self) -> ToolResult<()> {
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
        self.validate_scheduling()?;
        validate_native_execution(&self.backend.native_execution)?;
        validate_queue(&self.backend.queue)?;
        validate_device_profiles(&self.backend.device_profiles)?;
        self.validate_cache_mode()?;
        self.validate_workloads()
    }

    fn validate_scheduling(&self) -> ToolResult<()> {
        let scheduling = &self.scheduling;
        if scheduling.concurrency == 0 {
            return Err("scheduling.concurrency must be greater than zero".into());
        }
        match (scheduling.arrival_mode, scheduling.arrival_rate_per_second) {
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
        if scheduling.arrival_mode == ArrivalMode::Concurrency
            && scheduling.rate_overrun != RateOverrunPolicy::default()
        {
            return Err("scheduling.rate_overrun is only valid when scheduling.arrival_mode is rate".into());
        }
        match (scheduling.arrival_mode, scheduling.max_rounds) {
            (ArrivalMode::Concurrency, Some(0)) => {
                return Err("scheduling.max_rounds must be greater than zero when present".into());
            },
            (ArrivalMode::Rate, Some(_)) => {
                return Err("scheduling.max_rounds is only valid when scheduling.arrival_mode is concurrency".into());
            },
            _ => {},
        }
        Ok(())
    }

    fn validate_workloads(&self) -> ToolResult<()> {
        let mut workload_kinds = BTreeSet::new();
        for workload in &self.workloads {
            if !workload_kinds.insert(workload.kind) {
                return Err(format!("workloads contains duplicate {} fixture selections", workload.kind.label()).into());
            }
            if workload.path.as_os_str().is_empty() {
                return Err(format!("{} fixture path must not be empty", workload.kind.label()).into());
            }
            if !is_fixture_label(&workload.scale) {
                return Err(format!(
                    "{} fixture scale must be a 1-64 character ASCII identifier",
                    workload.kind.label()
                )
                .into());
            }
            if !is_preparation_command(&workload.preparation_command) {
                return Err(format!(
                    "{} fixture preparation_command must be 1-4096 printable characters",
                    workload.kind.label()
                )
                .into());
            }
            if !is_safe_preparation_command(workload.kind, &workload.preparation_command) {
                return Err(format!(
                    "{} fixture preparation_command must be a credential-free direct fixture command",
                    workload.kind.label()
                )
                .into());
            }
            let mut query_labels = BTreeSet::new();
            for query in &workload.queries {
                if !is_fixture_label(query) {
                    return Err(format!(
                        "{} workload query labels must be 1-64 character ASCII identifiers",
                        workload.kind.label()
                    )
                    .into());
                }
                if !query_labels.insert(query.as_str()) {
                    return Err(
                        format!("{} workload selects query {query} more than once", workload.kind.label()).into()
                    );
                }
            }
        }
        Ok(())
    }

    /// The profile's cache mode must describe the single ledger-accounted
    /// cache control the backend actually receives.  A nonzero ordinal-local
    /// cap is the P5 opt-in; there is no second cache switch in the runner.
    fn validate_cache_mode(&self) -> ToolResult<()> {
        let enabled_ordinals = self
            .backend
            .device_profiles
            .iter()
            .filter(|profile| profile.cache_cap_bytes != 0)
            .map(|profile| profile.ordinal)
            .collect::<Vec<_>>();
        match (self.cache.mode, enabled_ordinals.is_empty()) {
            (CacheMode::Disabled, true) | (CacheMode::Enabled, false) => Ok(()),
            (CacheMode::Disabled, false) => Err(format!(
                "cache.mode disabled requires cache_cap_bytes = 0 on every configured device; nonzero on ordinals {enabled_ordinals:?}"
            )
            .into()),
            (CacheMode::Enabled, true) => {
                Err("cache.mode enabled requires a nonzero cache_cap_bytes on at least one configured device".into())
            },
        }
    }
}

fn is_profile_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn is_fixture_label(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= 64
        && label.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn is_preparation_command(command: &str) -> bool {
    !command.is_empty() && command.len() <= 4096 && command.chars().all(|character| !character.is_control())
}

fn is_safe_preparation_command(kind: FixtureKind, command: &str) -> bool {
    if command.contains(['\'', '"', ';', '|', '&', '$', '`', '(', ')', '<', '>', '\\', '=']) {
        return false;
    }
    let arguments = command.split_ascii_whitespace().collect::<Vec<_>>();
    let (prefix, allowed_options, required_options) = match kind {
        FixtureKind::Tpch => (
            ["fixture/fixture.sh", "tpch", "generate"].as_slice(),
            [
                "--scale",
                "--out",
                "--out-root",
                "--variant",
                "--row-group-bytes",
                "--compression",
                "--tpchgen-root",
                "--profile",
                "--num-threads",
                "--parts",
                "--optimize-engine",
                "--staging-root",
            ]
            .as_slice(),
            ["--scale"].as_slice(),
        ),
        FixtureKind::Dblp => (
            ["fixture/fixture.sh", "graph", "dblp-ingest"].as_slice(),
            [
                "--source-json",
                "--out-root",
                "--row-group-size",
                "--profile-row-group-size",
                "--bucket-width",
                "--sort-buffer-rows",
                "--merge-fan-in",
                "--sort-temp-dir",
                "--progress-interval",
            ]
            .as_slice(),
            ["--source-json", "--out-root"].as_slice(),
        ),
    };
    if !arguments.starts_with(prefix) {
        return false;
    }
    let mut seen = BTreeSet::new();
    let mut tail = &arguments[prefix.len()..];
    while let Some(option) = tail.first() {
        if !allowed_options.contains(option) || !seen.insert(*option) {
            return false;
        }
        let Some(value) = tail.get(1) else {
            return false;
        };
        if value.starts_with('-') || value.is_empty() {
            return false;
        }
        tail = &tail[2..];
    }
    required_options.iter().all(|required| seen.contains(required))
        && match kind {
            FixtureKind::Tpch => seen.contains("--out") || seen.contains("--out-root"),
            FixtureKind::Dblp => true,
        }
}

fn validate_native_execution(native_execution: &NativeExecutionProfile) -> ToolResult<()> {
    if native_execution.device_budget_bytes == 0
        || native_execution.max_source_chunk_bytes == 0
        || native_execution.max_row_groups_per_chunk == 0
        || native_execution.min_retry_chunk_row_groups == 0
    {
        return Err("backend.native_execution contains a zero bound".into());
    }
    Ok(())
}

fn validate_queue(queue: &QueueProfile) -> ToolResult<()> {
    if queue.max_queued_attempts == 0 || queue.max_queue_wait_ms == 0 || queue.max_overtakes_per_waiter == 0 {
        return Err("backend.queue bounds must all be greater than zero".into());
    }
    Ok(())
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
        if profile.backend_reserve_bytes >= profile.memory_limit_bytes {
            return Err(format!(
                "backend.device_profiles[{}].backend_reserve_bytes must be below memory_limit_bytes",
                profile.ordinal
            )
            .into());
        }
        // The engine requires one root slot plus `source_work_per_attempt`
        // auxiliary slots for every concurrently active attempt. Rejecting it
        // here keeps a misconfigured profile from reaching a backend build.
        let required_slots = profile
            .max_active_attempts
            .checked_mul(profile.source_work_per_attempt.saturating_add(1))
            .ok_or("backend.device_profiles required stream slots overflow")?;
        if profile.installation.stream_pool_size < required_slots {
            return Err(format!(
                "backend.device_profiles[{}].installation.stream_pool_size must be at least {required_slots} \
                 (max_active_attempts {} x (source_work_per_attempt {} + 1))",
                profile.ordinal, profile.max_active_attempts, profile.source_work_per_attempt
            )
            .into());
        }
        // Worker and lane counts must cover every configured active slot.
        for (field, bound) in
            [("worker_count", profile.installation.worker_count), ("lane_count", profile.installation.lane_count)]
        {
            if bound < profile.max_active_attempts {
                return Err(format!(
                    "backend.device_profiles[{}].installation.{field} must be at least max_active_attempts {}",
                    profile.ordinal, profile.max_active_attempts
                )
                .into());
            }
        }
    }
    Ok(())
}
