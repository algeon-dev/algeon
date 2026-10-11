//! Run-profile parsing and fail-closed validation.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{RUN_PROFILE_SCHEMA, redact::ensure_redacted_json, util::sha256_bytes};
use crate::ToolResult;

/// Upper bound on attempts a single profile may schedule. It keeps the
/// per-attempt report vector bounded without capping any measured rate.
#[cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]
pub(super) const MAX_PLANNED_ATTEMPTS: usize = 100_000;

/// The profile fixes every top-level section the runner consumes.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct RunProfile {
    pub(super) schema: String,
    pub(super) name: String,
    /// Extra bounded time after the warmup-plus-measurement submission window
    /// for already-admitted attempts to reach a native safe boundary, release
    /// their grants, and seal their reports. Profiles without a drain grace leave
    /// this at zero; sustained stress profiles make their allowance explicit.
    #[serde(default)]
    pub(super) completion_drain_grace_seconds: u64,
    /// Selects meaningful overlap/placement clauses for this profile. An absent
    /// value requires the complete default scenario; stress profiles opt into
    /// only the clauses their declared suite can prove.
    #[serde(default)]
    pub(super) scenario_requirements: Option<ScenarioRequirementsProfile>,
    /// Opt-in QPS-envelope contract. The generic runner only asserts the
    /// queue/recovery rules when a profile explicitly declares this workload
    /// class measurement. The runner does not infer the class from a file name.
    #[serde(default)]
    pub(super) qps_envelope: Option<QpsEnvelopeProfile>,
    /// Explicit calibration contract between the admission/cache claims and
    /// the load-induced CUDA memory delta.  The runner does not derive a band
    /// from a workload name. Qualification profiles name the tolerated fixed
    /// overhead and multiplicative pricing slack themselves.
    #[serde(default)]
    pub(super) vram_calibration: Option<VramCalibrationProfile>,
    /// Explicit graph-model calibration contract for graph profiles. The artifact must retain one
    /// typed engine model/request/budget/peak row for every measured graph
    /// attempt; missing, duplicate, unavailable, or truncated rows fail
    /// closed. This profile intentionally owns no numeric tolerance band.
    #[serde(default)]
    pub(super) graph_claim_calibration: Option<GraphClaimCalibrationProfile>,
    /// The base-build feature-off cell. The runner schedules its schema-only cuGraph UDTF
    /// requests beside, but separately from, the relational mix and proves
    /// the feature-disabled contract from each structured source error and
    /// sealed report.
    #[serde(default)]
    pub(super) feature_off_cugraph_probe: Option<FeatureOffCugraphProbeProfile>,
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

/// The reusable validation reducer carries more structural clauses than every stress
/// suite needs. A scan-envelope run, for example, must prove placement
/// progress but cannot be rejected merely because it does not create two
/// concurrent jobs on one device. This profile-owned selector prevents a raw
/// proof obligation from becoming a stress qualification gate.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ScenarioRequirementsProfile {
    /// Require the complete same-device overlap/grant/domain witness.
    #[serde(default)]
    pub(super) require_same_device_overlap: bool,
    /// Require at least one successful placement on every ordinal that was
    /// eligible when the backend started.
    #[serde(default)]
    pub(super) require_all_device_progress: bool,
    /// Require the resolved device envelopes to be heterogeneous.
    #[serde(default)]
    pub(super) require_heterogeneous_device_profiles: bool,
    /// Require the sealed bounded-overload outcome. Contention profiles opt
    /// into this; an isolated calibration never infers it
    /// from a selected workload label.
    #[serde(default)]
    pub(super) require_overload_outcome: bool,
}

/// Profile-owned QPS-envelope requirements.
///
/// A measured envelope must be a multi-phase curve, and every phase must
/// retain some successful GPU work.  Rate profiles may additionally name a
/// post-overload recovery phase and the latency band it must return to.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct QpsEnvelopeProfile {
    /// The minimum number of successful GPU-reaching completions required in
    /// each configured step.  A zero-QPS all-failure cell is diagnostic evidence
    /// only and cannot establish a workload knee.
    #[serde(default = "default_qps_min_gpu_reaching_completions_per_step")]
    pub(super) min_gpu_reaching_completions_per_step: u64,
    /// The explicit low-rate recovery phase after the highest offered rate.
    /// Omitted for a closed-loop sweep, whose role is only knee discovery.
    #[serde(default)]
    pub(super) recovery: Option<QpsEnvelopeRecoveryProfile>,
}

const fn default_qps_min_gpu_reaching_completions_per_step() -> u64 {
    1
}

/// One rate-envelope recovery assertion. The named steps use the same offered
/// rate. The service's cumulative queue high-watermark must remain below its
/// nonzero configured limit through the baseline and reach that limit by an
/// intervening step with a higher offered rate. Recovery p95 total and queue
/// wait must stay inside the profile-owned multiple of the baseline.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct QpsEnvelopeRecoveryProfile {
    pub(super) baseline_step_index: usize,
    pub(super) recovery_step_index: usize,
    pub(super) max_p95_latency_multiplier: u32,
}

/// Profile-owned acceptance band for a claim-versus-device-VRAM calibration.
///
/// `max_*_ratio_milli` avoids floating point in a qualification predicate.
/// The fixed-byte terms accommodate measured, named floors (for example a
/// runtime context) without turning a large pricing error into a harmless
/// percentage.  The reducer always retains the raw deltas and both directions
/// of error; these bounds only decide whether this profile is qualifying.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct VramCalibrationProfile {
    /// Maximum bytes by which CUDA's load-induced used delta may exceed the
    /// ledger/cache claim at the same sampled peak.
    pub(super) max_underclaim_bytes: u64,
    /// Maximum `observed_cuda_delta / accounted_claim` ratio, in thousandths.
    /// A value of 1_000 permits no proportional underclaim.
    pub(super) max_observed_to_claim_ratio_milli: u64,
    /// Maximum bytes by which the ledger/cache claim may exceed the observed
    /// CUDA delta at the sampled peak before it is a capacity-pricing defect.
    pub(super) max_overclaim_bytes: u64,
    /// Maximum `accounted_claim / observed_cuda_delta` ratio, in thousandths.
    /// A value of 1_000 permits no proportional overclaim.
    pub(super) max_claim_to_observed_ratio_milli: u64,
}

/// Opt-in graph-model calibration projection contract.
///
/// The graph-memory model owns any future pricing adjustment. The stress
/// runner only requires complete typed measurement provenance and reports the
/// raw model-to-peak ratio; it must not invent an acceptable ratio band.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct GraphClaimCalibrationProfile {}

/// The bounded feature-off cuGraph UDTF contract exercised by the base runner. The graph
/// request itself is owned by the runner so this profile never needs a DBLP
/// fixture merely to prove that disabled execution does not stage graph data.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct FeatureOffCugraphProbeProfile {
    /// A graph UDTF rejection is a bounded control-plane operation. The measured
    /// report latency must not exceed this cap.
    pub(super) max_rejection_latency_ms: u64,
    /// Exact number of runner-owned UDTF requests issued before the observed
    /// warmup boundary.  Keeping it explicit prevents round-robin workload
    /// selection from changing the qualification denominator.
    pub(super) warmup_probe_attempts: usize,
    /// Exact number of runner-owned UDTF requests issued after the observed
    /// warmup boundary, while relational work is still live in the measured
    /// cell.
    pub(super) measurement_probe_attempts: usize,
    /// GPU-reaching relational successes required during the measurement
    /// phase, proving that the rejection cell ran alongside live native work.
    pub(super) min_relational_gpu_reaching_attempts: u64,
}

impl RunProfile {
    /// The stress runner never grants an unbounded post-window drain. This
    /// matches its bounded backend-close window while allowing a declared
    /// measurement window to stop submission before long GPU work is forced
    /// to a synthetic deadline at its final instant.
    pub(super) const MAX_COMPLETION_DRAIN_GRACE_SECONDS: u64 = 60;
    /// Keeps the feature-off rejection bound meaningful for a service control
    /// path. The configured value still becomes evidence in every artifact.
    pub(super) const MAX_FEATURE_OFF_CUGRAPH_REJECTION_LATENCY_MS: u64 = 1_000;

    /// Profiles without this section use the complete default scenario
    /// contract. Stress profiles spell out their reduced requirements instead
    /// of relying on their filename.
    pub(super) const fn scenario_requirements(&self) -> ScenarioRequirementsProfile {
        match self.scenario_requirements {
            Some(requirements) => requirements,
            None => ScenarioRequirementsProfile {
                require_same_device_overlap: true,
                require_all_device_progress: true,
                require_heterogeneous_device_profiles: true,
                require_overload_outcome: false,
            },
        }
    }
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
    Tpcds,
    Clickbench,
    Dblp,
}

impl FixtureKind {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Tpch => "tpch",
            Self::Tpcds => "tpcds",
            Self::Clickbench => "clickbench",
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
    pub(super) default_query_limit_bytes: u64,
    pub(super) source_work_per_attempt: usize,
    pub(super) max_active_attempts: usize,
    /// Whether this ordinal installs diagnostic CUDA/NVML telemetry. The
    /// telemetry-on/off pair changes only this profile-owned control.
    #[serde(default = "default_install_telemetry")]
    pub(super) install_telemetry: bool,
    pub(super) installation: InstallationBounds,
}

const fn default_install_telemetry() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum DeviceOwnership {
    WholeDeviceExclusive,
    StaticPartition { allocation_limit_bytes: u64 },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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
}

impl Default for QueueProfile {
    fn default() -> Self {
        Self { max_queued_attempts: default_max_queued_attempts(), max_queue_wait_ms: default_max_queue_wait_ms() }
    }
}

const fn default_max_queued_attempts() -> usize {
    64
}

const fn default_max_queue_wait_ms() -> u64 {
    30_000
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct SchedulingProfile {
    #[serde(default = "default_arrival_mode")]
    pub(super) arrival_mode: ArrivalMode,
    #[serde(default = "default_concurrency")]
    pub(super) concurrency: usize,
    #[serde(default)]
    pub(super) arrival_rate_per_second: Option<u64>,
    /// Optional exact bound on closed-loop rounds. This is an explicit
    /// scenario shape that leaves the remaining declared window for cooperative
    /// finalization.
    #[serde(default)]
    pub(super) max_rounds: Option<usize>,
    #[serde(default)]
    pub(super) seed: u64,
    #[serde(default)]
    pub(super) warmup_seconds: u64,
    #[serde(default)]
    pub(super) measurement_seconds: u64,
    /// How a rate-mode arrival that the queue cannot accept is treated. The
    /// setting separates offered from achieved arrival rate, so the artifact must
    /// record it.
    #[serde(default)]
    pub(super) rate_overrun: RateOverrunPolicy,
    /// Multi-phase arrival. A non-empty list switches the run to steps
    /// mode: steps execute sequentially inside one run, each with its own
    /// window and its own concurrency/rate, so one artifact holds a whole
    /// envelope curve. The scalar window/rate fields stay at their defaults.
    #[serde(default)]
    pub(super) steps: Vec<ScheduleStepProfile>,
    /// Every attempt in a deadline storm receives a caller deadline of
    /// `min(submission + attempt_deadline_ms, run deadline)`.
    #[serde(default)]
    pub(super) attempt_deadline_ms: Option<u64>,
    /// Every Nth submitted attempt in a cancellation storm is cancelled
    /// mid-flight through the public attempt cancellation handle.
    #[serde(default)]
    pub(super) cancel_every_n: Option<usize>,
    /// In a cancellation storm, the delay before each cancellation fires.
    /// Requires `cancel_every_n`; a zero delay cancels at the planning
    /// boundary.
    #[serde(default)]
    pub(super) cancel_after_ms: Option<u64>,
}

/// One sequential phase of a multi-phase step schedule. Which of `concurrency` /
/// `arrival_rate_per_second` is set must agree with the profile's
/// `arrival_mode`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ScheduleStepProfile {
    /// Closed-loop in-flight ceiling for this step (concurrency mode only).
    #[serde(default)]
    pub(super) concurrency: Option<usize>,
    /// Poisson offered rate for this step (rate mode only).
    #[serde(default)]
    pub(super) arrival_rate_per_second: Option<u64>,
    pub(super) measurement_seconds: u64,
    /// Per-step overrun policy for rate steps; defaults to `delay`.
    #[serde(default)]
    pub(super) rate_overrun: Option<RateOverrunPolicy>,
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
            steps: Vec::new(),
            attempt_deadline_ms: None,
            cancel_every_n: None,
            cancel_after_ms: None,
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheProfile {
    #[serde(default)]
    pub(super) mode: CacheMode,
    /// The explicit cache evidence contract. Ordinary cache-enabled profiles keep
    /// this absent; only a stress cell turns activity observations into hard
    /// qualification obligations.
    #[serde(default)]
    pub(super) stress: Option<CacheStressProfile>,
}

impl Default for CacheProfile {
    fn default() -> Self {
        Self { mode: CacheMode::Disabled, stress: None }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum CacheMode {
    #[default]
    Disabled,
    Enabled,
}

/// Profile-owned cache evidence requirements. The cache implementations
/// still own eviction, promotion, and coalescing; this merely says which
/// public observations a named load cell must produce before it qualifies.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheStressProfile {
    #[serde(default)]
    pub(super) require_resident_quarantine: bool,
    /// Exact cap expected on every configured cache-enabled ordinal.
    pub(super) required_per_device_cap_bytes: u64,
    /// A thrash cell must cumulatively copy at least this many source-cache
    /// bytes.  Together with an eviction witness this proves the workload
    /// crossed its configured working-set boundary without hard-coding file
    /// sizes in the runner.
    #[serde(default)]
    pub(super) minimum_source_promoted_bytes: u64,
    #[serde(default)]
    pub(super) require_source_eviction: bool,
    #[serde(default)]
    pub(super) require_source_coalescing: bool,
    #[serde(default)]
    pub(super) require_bfs_singleflight: bool,
    #[serde(default)]
    pub(super) require_explained_admission_delta: bool,
    /// Dedicated same-key cohorts run by the validation runner against one
    /// ordinal-local cache domain. Aggregate traffic counters cannot prove
    /// this: equal keys placed on different devices are intentionally
    /// independent caches.
    #[serde(default)]
    pub(super) same_key_probe: Option<CacheSameKeyProbeProfile>,
}

/// Profile-owned exact-key cache probe.
///
/// The runner creates a fresh single-ordinal backend for this bounded probe,
/// then drives source and BFS cohorts separately. That makes cold-cache,
/// cohort width, and allocation-domain ownership explicit. None of them is
/// inferred from an unrelated QPS window.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheSameKeyProbeProfile {
    /// Existing ordinal whose device profile is isolated for this probe.
    pub(super) target_device_ordinal: usize,
    /// Number of concurrent requests in each source/BFS same-key cohort.
    pub(super) cohort_width: usize,
    /// Bound for each cohort's public-attempt completion drain.
    pub(super) max_cohort_seconds: u64,
    pub(super) source: CacheSameKeyQueryProfile,
    pub(super) bfs: CacheSameKeyQueryProfile,
}

/// One selected query owned by a same-key probe cohort. Fixture kind is part
/// of identity because common labels such as `q01` exist in multiple suites.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheSameKeyQueryProfile {
    pub(super) kind: FixtureKind,
    pub(super) label: String,
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

    pub(super) fn selected_query_count(&self) -> usize {
        self.workloads.iter().map(|workload| workload.queries.len()).sum()
    }

    /// Returns the cache-disabled half of an opt-in paired calibration.
    ///
    /// This preserves every workload, scheduling, admission, and device field
    /// except the cache controls: `cache.mode`, `cache.stress`, and each
    /// ordinal-local `cache_cap_bytes`. The companion is derived from this
    /// profile, so a cache calibration cannot compare two unrelated runs.
    pub(super) fn cache_disabled_companion(&self) -> ToolResult<Self> {
        if self.cache.mode != CacheMode::Enabled {
            return Err("--cache-calibration requires cache.mode enabled in --config".into());
        }
        let mut companion = self.clone();
        companion.cache.mode = CacheMode::Disabled;
        companion.cache.stress = None;
        for device in &mut companion.backend.device_profiles {
            device.cache_cap_bytes = 0;
        }
        companion.validate()?;
        Ok(companion)
    }

    /// Validates the finite cohort used by the explicit overload probe.
    ///
    /// This is deliberately a profile shape and adds no second admission
    /// policy: the backend still owns queueing, FIFO placement, and report
    /// sealing. The runner only requires enough bounded offered
    /// work for those public facts to be falsifiable.
    pub(super) fn validate_overload_fairness_scenario(&self) -> ToolResult<()> {
        if self.backend.device_profiles.len() < 2 {
            return Err(
                "--overload-fairness requires at least two configured devices for completion on more than one eligible device".into()
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
        Ok(())
    }

    /// Validates the fixed two-step physical allocator-pressure probe.
    ///
    /// The runner establishes external CUDA pressure before `q01`, then
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
        if !is_profile_name(&self.name) {
            return Err(
                "run profile name must be a 1-64 character ASCII identifier using letters, digits, '.', '_' or '-'"
                    .into(),
            );
        }
        if self.completion_drain_grace_seconds > Self::MAX_COMPLETION_DRAIN_GRACE_SECONDS {
            return Err(format!(
                "completion_drain_grace_seconds must not exceed {}",
                Self::MAX_COMPLETION_DRAIN_GRACE_SECONDS
            )
            .into());
        }
        if self.session.shared_session_count == 0 {
            return Err("session.shared_session_count must be greater than zero".into());
        }
        self.validate_scheduling()?;
        self.validate_qps_envelope()?;
        self.validate_vram_calibration()?;
        validate_native_execution(&self.backend.native_execution)?;
        validate_queue(&self.backend.queue)?;
        validate_device_profiles(&self.backend.device_profiles)?;
        self.validate_cache_mode()?;
        self.validate_workloads()?;
        self.validate_graph_claim_calibration()?;
        self.validate_feature_off_cugraph_probe()
    }

    fn validate_qps_envelope(&self) -> ToolResult<()> {
        let Some(envelope) = self.qps_envelope else {
            return Ok(());
        };
        if self.scheduling.steps.len() < 2 {
            return Err("qps_envelope requires at least two explicit scheduling.steps".into());
        }
        if envelope.min_gpu_reaching_completions_per_step == 0 {
            return Err("qps_envelope.min_gpu_reaching_completions_per_step must be greater than zero".into());
        }
        let Some(recovery) = envelope.recovery else {
            return Ok(());
        };
        if self.scheduling.arrival_mode != ArrivalMode::Rate {
            return Err("qps_envelope.recovery requires scheduling.arrival_mode=rate".into());
        }
        if recovery.max_p95_latency_multiplier == 0 {
            return Err("qps_envelope.recovery.max_p95_latency_multiplier must be greater than zero".into());
        }
        let steps = &self.scheduling.steps;
        let Some(baseline) = steps.get(recovery.baseline_step_index) else {
            return Err("qps_envelope.recovery.baseline_step_index is outside scheduling.steps".into());
        };
        let Some(recovery_step) = steps.get(recovery.recovery_step_index) else {
            return Err("qps_envelope.recovery.recovery_step_index is outside scheduling.steps".into());
        };
        if recovery.recovery_step_index <= recovery.baseline_step_index {
            return Err("qps_envelope.recovery_step_index must follow its baseline_step_index".into());
        }
        let baseline_rate =
            baseline.arrival_rate_per_second.ok_or("qps_envelope recovery baseline must be a rate step")?;
        let recovery_rate =
            recovery_step.arrival_rate_per_second.ok_or("qps_envelope recovery step must be a rate step")?;
        if baseline_rate != recovery_rate {
            return Err("qps_envelope recovery step must use the baseline offered rate".into());
        }
        if !steps[recovery.baseline_step_index + 1..recovery.recovery_step_index]
            .iter()
            .filter_map(|step| step.arrival_rate_per_second)
            .any(|rate| rate > baseline_rate)
        {
            return Err(
                "qps_envelope recovery requires a prior offered rate above the baseline between baseline and recovery"
                    .into(),
            );
        }
        if steps[recovery.recovery_step_index.saturating_add(1)..]
            .iter()
            .filter_map(|step| step.arrival_rate_per_second)
            .any(|rate| rate > baseline_rate)
        {
            return Err("qps_envelope recovery step must follow the final above-baseline offered rate".into());
        }
        Ok(())
    }

    fn validate_vram_calibration(&self) -> ToolResult<()> {
        let Some(calibration) = self.vram_calibration else {
            return Ok(());
        };
        if calibration.max_observed_to_claim_ratio_milli < 1_000 {
            return Err("vram_calibration.max_observed_to_claim_ratio_milli must be at least 1000".into());
        }
        if calibration.max_claim_to_observed_ratio_milli < 1_000 {
            return Err("vram_calibration.max_claim_to_observed_ratio_milli must be at least 1000".into());
        }
        let telemetry_off = self
            .backend
            .device_profiles
            .iter()
            .filter(|device| !device.install_telemetry)
            .map(|device| device.ordinal)
            .collect::<Vec<_>>();
        if !telemetry_off.is_empty() {
            return Err(format!(
                "vram_calibration requires CUDA telemetry on every configured device; disabled ordinals {telemetry_off:?}"
            )
            .into());
        }
        Ok(())
    }

    fn validate_graph_claim_calibration(&self) -> ToolResult<()> {
        if self.graph_claim_calibration.is_none() {
            return Ok(());
        }
        let selected = super::workload::select_queries(&self.workloads)?;
        if !selected.iter().any(|query| query.graph) {
            return Err("graph_claim_calibration requires at least one selected cuGraph algorithm workload".into());
        }
        Ok(())
    }

    fn validate_scheduling(&self) -> ToolResult<()> {
        if self.scheduling.concurrency == 0 {
            return Err("scheduling.concurrency must be greater than zero".into());
        }
        if self.scheduling.steps.is_empty() {
            return self.validate_single_phase_scheduling();
        }
        self.validate_steps_scheduling()
    }

    fn validate_single_phase_scheduling(&self) -> ToolResult<()> {
        let scheduling = &self.scheduling;
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
        self.validate_storm_knobs()
    }

    /// Steps mode replaces the scalar window/rate fields with an explicit
    /// sequential phase list. The shape rules reject leftover single-phase fields,
    /// so a steps profile cannot also describe a single-phase run.
    fn validate_steps_scheduling(&self) -> ToolResult<()> {
        let scheduling = &self.scheduling;
        if scheduling.measurement_seconds != 0 {
            return Err(
                "scheduling.measurement_seconds must be 0 when scheduling.steps is used; each step declares its own window"
                    .into(),
            );
        }
        if scheduling.arrival_rate_per_second.is_some() {
            return Err("scheduling.arrival_rate_per_second is set per step when scheduling.steps is used".into());
        }
        if scheduling.max_rounds.is_some() {
            return Err("scheduling.max_rounds is not valid when scheduling.steps is used".into());
        }
        if scheduling.rate_overrun != RateOverrunPolicy::default() {
            return Err("scheduling.rate_overrun is set per step when scheduling.steps is used".into());
        }
        for (index, step) in scheduling.steps.iter().enumerate() {
            if step.measurement_seconds == 0 {
                return Err(format!("scheduling.steps[{index}].measurement_seconds must be greater than zero").into());
            }
            match (scheduling.arrival_mode, step.concurrency, step.arrival_rate_per_second) {
                (ArrivalMode::Concurrency, Some(width), None) if width > 0 => {
                    if step.rate_overrun.is_some() {
                        return Err(
                            format!("scheduling.steps[{index}].rate_overrun is only valid for rate steps").into()
                        );
                    }
                },
                (ArrivalMode::Concurrency, _, _) => {
                    return Err(format!(
                        "scheduling.steps[{index}] requires concurrency for concurrency-mode steps and no arrival_rate_per_second"
                    )
                    .into());
                },
                (ArrivalMode::Rate, None, Some(rate)) if rate > 0 => {},
                (ArrivalMode::Rate, _, _) => {
                    return Err(format!(
                        "scheduling.steps[{index}] requires arrival_rate_per_second for rate-mode steps and no concurrency"
                    )
                    .into());
                },
            }
        }
        self.validate_storm_knobs()
    }

    fn validate_storm_knobs(&self) -> ToolResult<()> {
        let scheduling = &self.scheduling;
        if scheduling.attempt_deadline_ms == Some(0) {
            return Err("scheduling.attempt_deadline_ms must be greater than zero when present".into());
        }
        if scheduling.cancel_every_n == Some(0) {
            return Err("scheduling.cancel_every_n must be greater than zero when present".into());
        }
        if scheduling.cancel_after_ms.is_some() && scheduling.cancel_every_n.is_none() {
            return Err("scheduling.cancel_after_ms requires scheduling.cancel_every_n".into());
        }
        Ok(())
    }

    fn validate_feature_off_cugraph_probe(&self) -> ToolResult<()> {
        let Some(probe) = self.feature_off_cugraph_probe.as_ref() else {
            return Ok(());
        };
        if probe.max_rejection_latency_ms == 0
            || probe.max_rejection_latency_ms > Self::MAX_FEATURE_OFF_CUGRAPH_REJECTION_LATENCY_MS
        {
            return Err(format!(
                "feature_off_cugraph_probe.max_rejection_latency_ms must be between 1 and {}",
                Self::MAX_FEATURE_OFF_CUGRAPH_REJECTION_LATENCY_MS
            )
            .into());
        }
        if probe.min_relational_gpu_reaching_attempts == 0 {
            return Err(
                "feature_off_cugraph_probe.min_relational_gpu_reaching_attempts must be greater than zero".into()
            );
        }
        if probe.warmup_probe_attempts == 0 || probe.measurement_probe_attempts == 0 {
            return Err(
                "feature_off_cugraph_probe.warmup_probe_attempts and feature_off_cugraph_probe.measurement_probe_attempts must both be greater than zero"
                    .into(),
            );
        }
        if probe
            .warmup_probe_attempts
            .checked_add(probe.measurement_probe_attempts)
            .is_none_or(|attempts| attempts > MAX_PLANNED_ATTEMPTS)
        {
            return Err(format!(
                "feature_off_cugraph_probe total probe attempts must not exceed {MAX_PLANNED_ATTEMPTS}"
            )
            .into());
        }
        if self.scheduling.concurrency < 2 {
            return Err(
                "feature_off_cugraph_probe requires scheduling.concurrency of at least 2 for mixed concurrent traffic"
                    .into(),
            );
        }
        if self.scheduling.arrival_mode != ArrivalMode::Concurrency
            || self.scheduling.max_rounds.is_some()
            || !self.scheduling.steps.is_empty()
        {
            return Err(
                "feature_off_cugraph_probe requires one duration-paced concurrency cell with scheduling.max_rounds omitted"
                    .into(),
            );
        }
        if self.scheduling.warmup_seconds == 0 {
            return Err(
                "feature_off_cugraph_probe requires scheduling.warmup_seconds greater than zero for its warmup probes"
                    .into(),
            );
        }
        if self.workloads.iter().any(|workload| workload.kind == FixtureKind::Dblp && !workload.queries.is_empty()) {
            return Err(
                "feature_off_cugraph_probe owns the only graph UDTF request and does not permit selected dblp workloads"
                    .into(),
            );
        }
        if !self.workloads.iter().any(|workload| workload.kind != FixtureKind::Dblp && !workload.queries.is_empty()) {
            return Err("feature_off_cugraph_probe requires at least one selected relational workload query".into());
        }
        Ok(())
    }

    fn validate_workloads(&self) -> ToolResult<()> {
        let mut workload_kinds = BTreeSet::new();
        let mut dblp_paths = BTreeSet::new();
        for workload in &self.workloads {
            if workload.kind != FixtureKind::Dblp && !workload_kinds.insert(workload.kind) {
                return Err(format!("workloads contains duplicate {} fixture selections", workload.kind.label()).into());
            }
            if workload.kind == FixtureKind::Dblp && !dblp_paths.insert(workload.path.clone()) {
                return Err("workloads contains the same DBLP fixture path more than once".into());
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
            for query in &workload.queries {
                if !is_fixture_label(query) {
                    return Err(format!(
                        "{} workload query labels must be 1-64 character ASCII identifiers",
                        workload.kind.label()
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    /// The profile's cache mode must describe the single ledger-accounted
    /// cache control the backend receives.  A nonzero ordinal-local
    /// cap is the cache opt-in; there is no second cache switch in the runner.
    fn validate_cache_mode(&self) -> ToolResult<()> {
        let enabled_ordinals = self
            .backend
            .device_profiles
            .iter()
            .filter(|profile| profile.cache_cap_bytes != 0)
            .map(|profile| profile.ordinal)
            .collect::<Vec<_>>();
        match (self.cache.mode, enabled_ordinals.is_empty()) {
            (CacheMode::Disabled, true) => {
                if self.cache.stress.is_some() {
                    return Err("cache.stress requires cache.mode enabled".into());
                }
                Ok(())
            },
            (CacheMode::Enabled, false) => {
                if let Some(stress) = self.cache.stress.as_ref() {
                    if stress.required_per_device_cap_bytes == 0 {
                        return Err("cache.stress.required_per_device_cap_bytes must be nonzero".into());
                    }
                    let mismatched = self
                        .backend
                        .device_profiles
                        .iter()
                        .filter(|device| device.cache_cap_bytes != stress.required_per_device_cap_bytes)
                        .map(|device| device.ordinal)
                        .collect::<Vec<_>>();
                    if !mismatched.is_empty() {
                        return Err(format!(
                            "cache.stress requires cache_cap_bytes={} on every configured device; mismatched ordinals {mismatched:?}",
                            stress.required_per_device_cap_bytes
                        )
                        .into());
                    }
                    self.validate_cache_same_key_probe(stress)?;
                }
                Ok(())
            },
            (CacheMode::Disabled, false) => Err(format!(
                "cache.mode disabled requires cache_cap_bytes = 0 on every configured device; nonzero on ordinals {enabled_ordinals:?}"
            )
            .into()),
            (CacheMode::Enabled, true) => {
                Err("cache.mode enabled requires a nonzero cache_cap_bytes on at least one configured device".into())
            },
        }
    }

    fn validate_cache_same_key_probe(&self, stress: &CacheStressProfile) -> ToolResult<()> {
        let Some(probe) = stress.same_key_probe.as_ref() else {
            return Ok(());
        };
        if probe.cohort_width < 2 {
            return Err("cache.stress.same_key_probe.cohort_width must be at least 2".into());
        }
        if probe.max_cohort_seconds == 0 || probe.max_cohort_seconds > Self::MAX_COMPLETION_DRAIN_GRACE_SECONDS {
            return Err(format!(
                "cache.stress.same_key_probe.max_cohort_seconds must be between 1 and {}",
                Self::MAX_COMPLETION_DRAIN_GRACE_SECONDS
            )
            .into());
        }
        let target = self
            .backend
            .device_profiles
            .iter()
            .find(|device| device.ordinal == probe.target_device_ordinal)
            .ok_or_else(|| {
            format!(
                "cache.stress.same_key_probe.target_device_ordinal {} is not configured",
                probe.target_device_ordinal
            )
        })?;
        if target.max_active_attempts < probe.cohort_width {
            return Err(format!(
                "cache.stress.same_key_probe.cohort_width {} exceeds target ordinal {} max_active_attempts {}",
                probe.cohort_width, probe.target_device_ordinal, target.max_active_attempts
            )
            .into());
        }
        if probe.source.kind == FixtureKind::Dblp {
            return Err("cache.stress.same_key_probe.source must select a relational local-Parquet workload".into());
        }
        if probe.bfs.kind != FixtureKind::Dblp || !matches!(probe.bfs.label.as_str(), "bfs" | "bfs_deep") {
            return Err(
                "cache.stress.same_key_probe.bfs must select the DBLP bfs or bfs_deep query for exact-request singleflight"
                    .into(),
            );
        }
        for (name, query) in [("source", &probe.source), ("bfs", &probe.bfs)] {
            let selected = self
                .workloads
                .iter()
                .filter(|workload| workload.kind == query.kind)
                .filter(|workload| workload.queries.iter().any(|label| label == &query.label))
                .count();
            if selected == 0 {
                return Err(format!(
                    "cache.stress.same_key_probe.{name} must name a selected {} query {}",
                    query.kind.label(),
                    query.label
                )
                .into());
            }
            if selected != 1 {
                return Err(format!(
                    "cache.stress.same_key_probe.{name} matches {selected} selected {} query {} entries; a same-key probe requires exactly one fixture selection",
                    query.kind.label(),
                    query.label
                )
                .into());
            }
        }
        Ok(())
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
        FixtureKind::Tpcds => (
            ["fixture/fixture.sh", "tpcds", "generate"].as_slice(),
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
        FixtureKind::Clickbench => (
            ["fixture/fixture.sh", "clickbench", "fetch"].as_slice(),
            ["--layout", "--mode", "--out", "--output-dir", "--jobs"].as_slice(),
            [].as_slice(),
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
                "--edge-slice-count",
                "--edge-slice-seed",
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
            FixtureKind::Tpch | FixtureKind::Tpcds => seen.contains("--out") || seen.contains("--out-root"),
            FixtureKind::Clickbench => seen.contains("--out") || seen.contains("--output-dir"),
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
    if queue.max_queued_attempts == 0 || queue.max_queue_wait_ms == 0 {
        return Err("backend.queue bounds must be greater than zero".into());
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
            || profile.default_query_limit_bytes == 0
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
        // Mirrors `GpuBackendDeviceProfile::derived_installation_bounds`, the
        // runtime's rule. A misconfigured profile fails here, before backend
        // build: one root slot plus `source_work_per_attempt` auxiliary slots
        // and one execution worker per active attempt, plus the worker the
        // runtime reserves for lifecycle jobs.
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
        let required_workers = profile
            .max_active_attempts
            .checked_add(1)
            .ok_or("backend.device_profiles required worker count overflow")?;
        if profile.installation.worker_count < required_workers {
            return Err(format!(
                "backend.device_profiles[{}].installation.worker_count must be at least {required_workers} \
                 (max_active_attempts {} + 1, one worker is reserved for lifecycle jobs)",
                profile.ordinal, profile.max_active_attempts
            )
            .into());
        }
        if profile.installation.lane_count < profile.max_active_attempts {
            return Err(format!(
                "backend.device_profiles[{}].installation.lane_count must be at least max_active_attempts {}",
                profile.ordinal, profile.max_active_attempts
            )
            .into());
        }
    }
    Ok(())
}
