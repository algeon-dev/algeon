//! Versioned P6 production-validation run artifacts.
//!
//! P6.3 owns the dry run, P6.4 the read-only fixture preflight, and P6.5 the
//! measured public-API run. Config parsing, arrival planning, aggregation, and
//! artifact writing are unit-tested library code; only `measure` touches a GPU.

mod aggregate;
mod artifact;
mod cache_calibration;
#[cfg(feature = "single-node-multi-gpu-validate-run")]
mod measure;
mod metrics_audit;
mod preflight;
mod profile;
mod provenance;
mod redact;
mod schedule;
mod util;
mod workload;

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use artifact::{RunArtifact, write_artifact};
use preflight::{FixturePreflight, FixturePreflightArtifact, FixtureSelectionPolicy};
use profile::RunProfile;
use provenance::Provenance;

use crate::ToolResult;

const RUN_PROFILE_SCHEMA: &str = "datafusion_nexus.single_node_multi_gpu.run_profile.v1";
const RUN_PROFILE_VERSION: u32 = 1;
const RUN_ARTIFACT_SCHEMA: &str = "datafusion_nexus.single_node_multi_gpu.run_artifact.v2";
const RUN_ARTIFACT_VERSION: u32 = 2;
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
    let artifact_path = match options.mode {
        ValidationMode::DryRun => run_dry_run(profile, config_sha256, &options.out_dir)?,
        ValidationMode::FixturePreflight => {
            run_fixture_preflight(profile, config_sha256, &options.out_dir, Provenance::collect)?
        },
        ValidationMode::MeasuredRun => run_measured_run(profile, config_sha256, &options)?,
    };

    println!("single-node-multi-gpu validation artifact: {}", artifact_path.display());
    Ok(())
}

fn print_usage() {
    eprintln!(
        r"single-node-multi-gpu-validate

USAGE:
  single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --dry-run
  single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --fixture-preflight
  single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --measured-run [--nsight-trace PATH] [--quarantine-device ORDINAL] [--cancel-first-after-ms MILLISECONDS] [--deadline-first-after-ms MILLISECONDS] [--domain-exhaustion-probe] [--physical-allocator-pressure] [--overload-fairness] [--cache-calibration]

--dry-run accepts only an empty workload list. --fixture-preflight validates one TPC-H and one DBLP fixture without generating, downloading, overwriting, or executing a workload. --measured-run executes the profile's workload mix on the public backend and attempt API and requires the single-node-multi-gpu-validate-run feature. --nsight-trace records an externally captured trace's identity; this runner never captures one itself. --quarantine-device keeps a first closed-loop cohort live until the target has a public grant/job witness, invokes the backend's explicit controlled fault-injection capability, then releases a post-acknowledgement cohort; it does not manufacture a CUDA fault. --cancel-first-after-ms requests cooperative cancellation of the first submitted public attempt; the artifact fails closed unless its sealed report is cancelled. --deadline-first-after-ms gives the first attempt an absolute deadline, requires public post-admission evidence before its typed deadline-expired report, then proves healthy reuse without task abort. --domain-exhaustion-probe runs the profile's fixed serial q01 then edge_scan scenario against one static partition and requires a sealed exact `reservation_limit` allocator event, released capacity, healthy reuse, and same-device recovery. It does not claim a physical allocator failure. --physical-allocator-pressure establishes a real external cuDF sidecar allocation before public q01 SQL on one static partition, releases it before edge_scan recovery, and fails closed unless q01 seals one exact `physical` allocator event below every immutable logical cap with completed cleanup, healthy reuse, and quiescent close. --overload-fairness drives one finite cohort above the resolved active-attempt capacity and validates only sealed admission facts: complete bounded drain, exact finite-overtake arithmetic and protection, and a successful unrelated-device completion before the protected waiter is admitted. --cache-calibration derives a cache-disabled companion from an enabled profile and runs the paired P5 cache calibration."
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    config: PathBuf,
    out_dir: PathBuf,
    mode: ValidationMode,
    /// An externally captured Nsight trace whose identity the artifact records.
    nsight_trace: Option<PathBuf>,
    /// Device ordinal to quarantine through the explicit public validation
    /// capability after a target-device live-work witness and before the
    /// runner releases its post-acknowledgement cohort.
    quarantine_device: Option<usize>,
    /// Request cooperative cancellation of the first submitted public
    /// attempt after this delay. A zero delay still uses the attempt API and
    /// proves the planning boundary without needing a synthetic native fault.
    cancel_first_after: Option<std::time::Duration>,
    /// Absolute deadline interval for a dedicated first-attempt P6.6 probe.
    /// The runner witnesses its admitted work before accepting a typed
    /// `deadline_expired` sealed report and a healthy reuse attempt.
    deadline_first_after: Option<std::time::Duration>,
    /// Drive the dedicated P6.6 ordinary domain-cap exhaustion scenario. It
    /// reuses public SQL rather than injecting an allocator failure.
    domain_exhaustion_probe: bool,
    /// Drive the dedicated P6.6 finite-cohort overload scenario. All outcome
    /// proof comes from sealed reports; it never creates a synthetic queue
    /// request or infers finite overtakes from task ordering.
    overload_fairness: bool,
    /// Drive P6.6's physical allocator pressure probe. The measured-run-only
    /// helper creates external CUDA pressure; no backend mode or synthetic
    /// allocator failure exists in this runner.
    physical_allocator_pressure: bool,
    /// Run the cache-disabled companion and configured P5 cache opt-in as a
    /// paired P6.8 calibration.
    cache_calibration: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationMode {
    DryRun,
    FixturePreflight,
    MeasuredRun,
}

impl CliOptions {
    fn parse(args: Vec<String>) -> ToolResult<Self> {
        let mut config = None;
        let mut out_dir = None;
        let mut mode = None;
        let mut nsight_trace = None;
        let mut quarantine_device = None;
        let mut cancel_first_after = None;
        let mut deadline_first_after = None;
        let mut domain_exhaustion_probe = false;
        let mut overload_fairness = false;
        let mut physical_allocator_pressure = false;
        let mut cache_calibration = false;
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
                "--nsight-trace" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    nsight_trace = Some(PathBuf::from(value));
                },
                "--quarantine-device" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    quarantine_device = Some(
                        value
                            .parse::<usize>()
                            .map_err(|_| "--quarantine-device must be a non-negative device ordinal")?,
                    );
                },
                "--cancel-first-after-ms" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    let milliseconds =
                        value.parse::<u64>().map_err(|_| "--cancel-first-after-ms must be an unsigned integer")?;
                    cancel_first_after = Some(std::time::Duration::from_millis(milliseconds));
                },
                "--deadline-first-after-ms" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    let milliseconds =
                        value.parse::<u64>().map_err(|_| "--deadline-first-after-ms must be an unsigned integer")?;
                    if milliseconds == 0 {
                        return Err("--deadline-first-after-ms must be greater than zero".into());
                    }
                    deadline_first_after = Some(std::time::Duration::from_millis(milliseconds));
                },
                "--domain-exhaustion-probe" => domain_exhaustion_probe = true,
                "--overload-fairness" => overload_fairness = true,
                "--physical-allocator-pressure" => physical_allocator_pressure = true,
                "--cache-calibration" => cache_calibration = true,
                "--dry-run" => set_mode(&mut mode, ValidationMode::DryRun)?,
                "--fixture-preflight" => set_mode(&mut mode, ValidationMode::FixturePreflight)?,
                "--measured-run" => set_mode(&mut mode, ValidationMode::MeasuredRun)?,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 1;
        }

        let mode = mode.ok_or("missing required validation mode --dry-run, --fixture-preflight, or --measured-run")?;
        let config = config.ok_or("missing required option --config")?;
        let out_dir = out_dir.ok_or("missing required option --out-dir")?;
        if out_dir.as_os_str().is_empty() {
            return Err("--out-dir must not be empty".into());
        }
        if nsight_trace.is_some() && mode != ValidationMode::MeasuredRun {
            return Err("--nsight-trace is only valid with --measured-run".into());
        }
        if quarantine_device.is_some() && mode != ValidationMode::MeasuredRun {
            return Err("--quarantine-device is only valid with --measured-run".into());
        }
        if cancel_first_after.is_some() && mode != ValidationMode::MeasuredRun {
            return Err("--cancel-first-after-ms is only valid with --measured-run".into());
        }
        if deadline_first_after.is_some() && mode != ValidationMode::MeasuredRun {
            return Err("--deadline-first-after-ms is only valid with --measured-run".into());
        }
        if domain_exhaustion_probe && mode != ValidationMode::MeasuredRun {
            return Err("--domain-exhaustion-probe is only valid with --measured-run".into());
        }
        if overload_fairness && mode != ValidationMode::MeasuredRun {
            return Err("--overload-fairness is only valid with --measured-run".into());
        }
        if physical_allocator_pressure && mode != ValidationMode::MeasuredRun {
            return Err("--physical-allocator-pressure is only valid with --measured-run".into());
        }
        if cache_calibration && mode != ValidationMode::MeasuredRun {
            return Err("--cache-calibration is only valid with --measured-run".into());
        }
        if cache_calibration && quarantine_device.is_some() {
            return Err("--cache-calibration cannot be combined with --quarantine-device".into());
        }
        if cache_calibration && cancel_first_after.is_some() {
            return Err("--cache-calibration cannot be combined with --cancel-first-after-ms".into());
        }
        if cache_calibration && deadline_first_after.is_some() {
            return Err("--cache-calibration cannot be combined with --deadline-first-after-ms".into());
        }
        if cache_calibration && domain_exhaustion_probe {
            return Err("--cache-calibration cannot be combined with --domain-exhaustion-probe".into());
        }
        if cache_calibration && overload_fairness {
            return Err("--cache-calibration cannot be combined with --overload-fairness".into());
        }
        if cache_calibration && physical_allocator_pressure {
            return Err("--cache-calibration cannot be combined with --physical-allocator-pressure".into());
        }
        if quarantine_device.is_some() && deadline_first_after.is_some() {
            return Err("--quarantine-device cannot be combined with --deadline-first-after-ms".into());
        }
        if cancel_first_after.is_some() && deadline_first_after.is_some() {
            return Err("--cancel-first-after-ms cannot be combined with --deadline-first-after-ms".into());
        }
        if quarantine_device.is_some() && domain_exhaustion_probe {
            return Err("--quarantine-device cannot be combined with --domain-exhaustion-probe".into());
        }
        if cancel_first_after.is_some() && domain_exhaustion_probe {
            return Err("--cancel-first-after-ms cannot be combined with --domain-exhaustion-probe".into());
        }
        if deadline_first_after.is_some() && domain_exhaustion_probe {
            return Err("--deadline-first-after-ms cannot be combined with --domain-exhaustion-probe".into());
        }
        if overload_fairness && quarantine_device.is_some() {
            return Err("--overload-fairness cannot be combined with --quarantine-device".into());
        }
        if overload_fairness && cancel_first_after.is_some() {
            return Err("--overload-fairness cannot be combined with --cancel-first-after-ms".into());
        }
        if overload_fairness && deadline_first_after.is_some() {
            return Err("--overload-fairness cannot be combined with --deadline-first-after-ms".into());
        }
        if overload_fairness && domain_exhaustion_probe {
            return Err("--overload-fairness cannot be combined with --domain-exhaustion-probe".into());
        }
        if physical_allocator_pressure && quarantine_device.is_some() {
            return Err("--physical-allocator-pressure cannot be combined with --quarantine-device".into());
        }
        if physical_allocator_pressure && cancel_first_after.is_some() {
            return Err("--physical-allocator-pressure cannot be combined with --cancel-first-after-ms".into());
        }
        if physical_allocator_pressure && deadline_first_after.is_some() {
            return Err("--physical-allocator-pressure cannot be combined with --deadline-first-after-ms".into());
        }
        if physical_allocator_pressure && domain_exhaustion_probe {
            return Err("--physical-allocator-pressure cannot be combined with --domain-exhaustion-probe".into());
        }
        if physical_allocator_pressure && overload_fairness {
            return Err("--physical-allocator-pressure cannot be combined with --overload-fairness".into());
        }
        Ok(Self {
            config,
            out_dir,
            mode,
            nsight_trace,
            quarantine_device,
            cancel_first_after,
            deadline_first_after,
            domain_exhaustion_probe,
            overload_fairness,
            physical_allocator_pressure,
            cache_calibration,
        })
    }
}

fn set_mode(mode: &mut Option<ValidationMode>, selected: ValidationMode) -> ToolResult<()> {
    if mode.replace(selected).is_some() {
        return Err("--dry-run, --fixture-preflight, and --measured-run are mutually exclusive".into());
    }
    Ok(())
}

fn required_option_value(args: &[String], index: &mut usize, flag: &str) -> ToolResult<String> {
    *index += 1;
    let value = args.get(*index).ok_or_else(|| format!("missing value for {flag}"))?;
    if value.starts_with("--") {
        return Err(format!("missing value for {flag}").into());
    }
    Ok(value.clone())
}

fn run_dry_run(profile: RunProfile, config_sha256: String, out_dir: &Path) -> ToolResult<PathBuf> {
    if !profile.workloads.is_empty() {
        return Err(
            "--dry-run accepts only an empty workloads list; use --fixture-preflight for the P6.4 fixture contract"
                .into(),
        );
    }
    let artifact = RunArtifact::dry_run(profile, config_sha256, Provenance::collect()?);
    write_artifact(out_dir, &artifact)
}

fn run_fixture_preflight(
    profile: RunProfile,
    config_sha256: String,
    out_dir: &Path,
    collect_provenance: impl FnOnce() -> ToolResult<Provenance>,
) -> ToolResult<PathBuf> {
    let fixtures = FixturePreflight::collect(&profile.workloads, FixtureSelectionPolicy::BothKindsExactly)?;
    let artifact = RunArtifact::fixture_preflight(profile, config_sha256, fixtures, collect_provenance()?);
    write_artifact(out_dir, &artifact)
}

/// Validates the measured-run profile and its fixtures before anything else, so
/// a misconfigured run cannot reach a backend build or publish a partial
/// artifact.
fn measured_run_preflight(
    profile: &RunProfile,
    quarantine_device: Option<usize>,
    deadline_first_after: Option<std::time::Duration>,
    domain_exhaustion_probe: bool,
    overload_fairness: bool,
    physical_allocator_pressure: bool,
    cache_calibration: bool,
) -> ToolResult<Vec<FixturePreflightArtifact>> {
    if profile.selected_query_count() == 0 {
        return Err("--measured-run requires at least one workload query selection".into());
    }
    if profile.backend.device_profiles.is_empty() {
        return Err("--measured-run requires at least one backend.device_profiles entry".into());
    }
    if profile.scheduling.measurement_seconds == 0 {
        return Err("--measured-run requires a non-zero scheduling.measurement_seconds".into());
    }
    if let Some(device_ordinal) = quarantine_device {
        if profile.backend.device_profiles.len() < 2 {
            return Err("--quarantine-device requires at least two configured devices for fallback evidence".into());
        }
        if !profile.backend.device_profiles.iter().any(|profile| profile.ordinal == device_ordinal) {
            return Err(
                format!("--quarantine-device {device_ordinal} is not configured by backend.device_profiles").into()
            );
        }
        if profile.scheduling.arrival_mode != profile::ArrivalMode::Concurrency {
            return Err(
                "--quarantine-device requires scheduling.arrival_mode=concurrency for its live pre-ack cohort".into()
            );
        }
        if profile.scheduling.max_rounds.is_some_and(|rounds| rounds < 2) {
            return Err("--quarantine-device requires scheduling.max_rounds of at least 2 when it is set".into());
        }
    }
    if deadline_first_after.is_some() {
        if profile.scheduling.arrival_mode != profile::ArrivalMode::Concurrency {
            return Err("--deadline-first-after-ms requires scheduling.arrival_mode=concurrency".into());
        }
        if profile.scheduling.max_rounds.is_some_and(|rounds| rounds < 2) {
            return Err("--deadline-first-after-ms requires scheduling.max_rounds of at least 2 when it is set".into());
        }
    }
    if domain_exhaustion_probe {
        if profile.backend.device_profiles.len() != 1 {
            return Err(
                "--domain-exhaustion-probe requires exactly one configured device for same-device recovery".into()
            );
        }
        if !matches!(&profile.backend.device_profiles[0].ownership, profile::DeviceOwnership::StaticPartition { .. }) {
            return Err("--domain-exhaustion-probe requires a static_partition device ownership envelope".into());
        }
        if profile.cache.mode != profile::CacheMode::Disabled {
            return Err("--domain-exhaustion-probe requires cache.mode=disabled".into());
        }
        if profile.scheduling.arrival_mode != profile::ArrivalMode::Concurrency || profile.scheduling.concurrency != 1 {
            return Err(
                "--domain-exhaustion-probe requires scheduling.arrival_mode=concurrency with concurrency=1".into()
            );
        }
        if profile.scheduling.max_rounds != Some(2) {
            return Err("--domain-exhaustion-probe requires scheduling.max_rounds=2".into());
        }
        let labels = profile
            .workloads
            .iter()
            .flat_map(|workload| workload.queries.iter().map(String::as_str))
            .collect::<Vec<_>>();
        if labels.as_slice() != &["q01", "edge_scan"] {
            return Err("--domain-exhaustion-probe requires exactly q01 followed by edge_scan".into());
        }
    }
    if overload_fairness {
        profile.validate_overload_fairness_scenario()?;
    }
    if physical_allocator_pressure {
        profile.validate_physical_allocator_pressure_scenario()?;
    }
    if cache_calibration {
        profile.cache_disabled_companion()?;
    }
    let executed =
        profile.workloads.iter().filter(|workload| !workload.queries.is_empty()).cloned().collect::<Vec<_>>();
    FixturePreflight::collect(&executed, FixtureSelectionPolicy::AtLeastOneKind)
}

#[cfg(feature = "single-node-multi-gpu-validate-run")]
fn run_measured_run(profile: RunProfile, config_sha256: String, options: &CliOptions) -> ToolResult<PathBuf> {
    let fixtures = measured_run_preflight(
        &profile,
        options.quarantine_device,
        options.deadline_first_after,
        options.domain_exhaustion_probe,
        options.overload_fairness,
        options.physical_allocator_pressure,
        options.cache_calibration,
    )?;
    let trace = match &options.nsight_trace {
        Some(path) => {
            measure::TraceSelection { uri: Some(path.display().to_string()), sha256: Some(util::sha256_file(path)?) }
        },
        None => measure::TraceSelection { uri: None, sha256: None },
    };
    let provenance = Provenance::collect()?;
    // The tool entrypoint pins a current-thread runtime, which cannot execute
    // concurrent attempts. Own a multi-thread runtime on a dedicated thread so
    // the measured concurrency is real rather than interleaved.
    let artifact = if options.cache_calibration {
        let cache_disabled = profile.cache_disabled_companion()?;
        let disabled_measurement =
            measure::run_on_owned_runtime(&cache_disabled, trace.clone(), None, None, None, false, false, false)?;
        let opt_in_measurement = measure::run_on_owned_runtime(&profile, trace, None, None, None, false, false, false)?;
        RunArtifact::cache_calibration(
            profile,
            config_sha256,
            fixtures,
            provenance,
            disabled_measurement,
            opt_in_measurement,
        )
    } else {
        let measurement = measure::run_on_owned_runtime(
            &profile,
            trace,
            options.quarantine_device,
            options.cancel_first_after,
            options.deadline_first_after,
            options.domain_exhaustion_probe,
            options.overload_fairness,
            options.physical_allocator_pressure,
        )?;
        RunArtifact::measured_run(profile, config_sha256, fixtures, provenance, measurement)
    };
    let path = write_artifact(&options.out_dir, &artifact)?;
    if matches!(artifact.status, "measured_run_non_qualifying" | "cache_calibration_non_qualifying") {
        // The artifact is complete and published; the non-zero exit is what
        // makes a non-qualifying result impossible to mistake for a pass.
        return Err(format!("measured run is non-qualifying; see scenario clauses in {}", path.display()).into());
    }
    Ok(path)
}

#[cfg(not(feature = "single-node-multi-gpu-validate-run"))]
fn run_measured_run(profile: RunProfile, _config_sha256: String, _options: &CliOptions) -> ToolResult<PathBuf> {
    // Validate what does not need the GPU, then refuse rather than emit an
    // artifact that claims a run happened.
    measured_run_preflight(
        &profile,
        _options.quarantine_device,
        _options.deadline_first_after,
        _options.domain_exhaustion_probe,
        _options.overload_fairness,
        _options.physical_allocator_pressure,
        _options.cache_calibration,
    )?;
    Err("--measured-run requires --features single-node-multi-gpu-validate-run".into())
}

#[cfg(test)]
mod tests;
