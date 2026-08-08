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
mod operability;
mod preflight;
mod profile;
mod provenance;
mod redact;
mod schedule;
mod timeseries;
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
const RUN_ARTIFACT_SCHEMA: &str = "datafusion_nexus.single_node_multi_gpu.run_artifact.v3";
const RUN_ARTIFACT_VERSION: u32 = 3;
const COMPOSED_SCHEDULE_SCHEMA: &str = "datafusion_nexus.single_node_multi_gpu.composed_schedule.v1";
const COMPOSED_SCHEDULE_VERSION: u32 = 1;
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
  single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --measured-run [OPTIONS]

`--measured-run` executes the selected public-API workload mix. It never
persists or compares query result rows; it records lifecycle, admission,
telemetry, allocator-attribution, cache, and operability evidence only."
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    config: PathBuf,
    out_dir: PathBuf,
    mode: ValidationMode,
    /// An externally captured Nsight trace whose identity the artifact records.
    nsight_trace: Option<PathBuf>,
    /// W8's explicit live reference-exporter cell. The full URL is runtime
    /// input only; artifacts retain a redacted host:port identity.
    otlp_metrics_endpoint: Option<String>,
    /// Defaults in the measured-run-only exporter configuration when an
    /// endpoint is supplied.
    otlp_export_interval_ms: Option<u64>,
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
    /// S6: quarantine this many ordinals, one cycle at a time, during a
    /// sustained closed-loop run. Each cycle witnesses live work on its
    /// target, injects through the public fault injector, and requires zero
    /// post-acknowledgement placements on the target with healthy fleet
    /// progress on the remaining ordinals.
    quarantine_cycles: Option<usize>,
    /// S6: begin the public close path at the midpoint of the measurement
    /// window while a load cohort is in flight. Queued waiters must seal
    /// `backend_closing`, in-flight attempts must finish, and every submitted
    /// attempt must be accounted for exactly once.
    close_under_load: bool,
    /// I8: hold one public completion receiver at a time behind this delay.
    /// This is a dedicated rate-mode probe, never a backend behavior mode.
    slow_report_sink: Option<std::time::Duration>,
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
        let mut otlp_metrics_endpoint = None;
        let mut otlp_export_interval_ms = None;
        let mut quarantine_device = None;
        let mut cancel_first_after = None;
        let mut deadline_first_after = None;
        let mut domain_exhaustion_probe = false;
        let mut overload_fairness = false;
        let mut physical_allocator_pressure = false;
        let mut cache_calibration = false;
        let mut quarantine_cycles = None;
        let mut close_under_load = false;
        let mut slow_report_sink = None;
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
                "--otlp-metrics-endpoint" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    otlp_metrics_endpoint = Some(value);
                },
                "--otlp-export-interval-ms" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    let milliseconds =
                        value.parse::<u64>().map_err(|_| "--otlp-export-interval-ms must be a positive integer")?;
                    if milliseconds == 0 {
                        return Err("--otlp-export-interval-ms must be greater than zero".into());
                    }
                    otlp_export_interval_ms = Some(milliseconds);
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
                "--quarantine-cycles" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    let cycles =
                        value.parse::<usize>().map_err(|_| "--quarantine-cycles must be a positive integer")?;
                    if cycles == 0 {
                        return Err("--quarantine-cycles must be greater than zero".into());
                    }
                    quarantine_cycles = Some(cycles);
                },
                "--close-under-load" => close_under_load = true,
                "--slow-report-sink-ms" => {
                    let value = required_option_value(&args, &mut index, flag)?;
                    let milliseconds =
                        value.parse::<u64>().map_err(|_| "--slow-report-sink-ms must be a positive integer")?;
                    if milliseconds == 0 {
                        return Err("--slow-report-sink-ms must be greater than zero".into());
                    }
                    slow_report_sink = Some(std::time::Duration::from_millis(milliseconds));
                },
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
        if otlp_metrics_endpoint.is_some() && mode != ValidationMode::MeasuredRun {
            return Err("--otlp-metrics-endpoint is only valid with --measured-run".into());
        }
        if otlp_export_interval_ms.is_some() && otlp_metrics_endpoint.is_none() {
            return Err("--otlp-export-interval-ms requires --otlp-metrics-endpoint".into());
        }
        // Scenario controls are pairwise exclusive and measured-run only. The
        // table keeps that contract in one place; the error names the first
        // conflicting pair.
        let scenario_controls: [(&str, bool); 10] = [
            ("--quarantine-device", quarantine_device.is_some()),
            ("--cancel-first-after-ms", cancel_first_after.is_some()),
            ("--deadline-first-after-ms", deadline_first_after.is_some()),
            ("--domain-exhaustion-probe", domain_exhaustion_probe),
            ("--overload-fairness", overload_fairness),
            ("--physical-allocator-pressure", physical_allocator_pressure),
            ("--cache-calibration", cache_calibration),
            ("--quarantine-cycles", quarantine_cycles.is_some()),
            ("--close-under-load", close_under_load),
            ("--slow-report-sink-ms", slow_report_sink.is_some()),
        ];
        let active = scenario_controls.iter().filter(|(_, active)| *active).map(|(name, _)| *name);
        let mut active = active.peekable();
        if let Some(first) = active.next()
            && let Some(second) = active.next()
        {
            return Err(format!("{first} cannot be combined with {second}").into());
        }
        for (name, active) in scenario_controls {
            if active && mode != ValidationMode::MeasuredRun {
                return Err(format!("{name} is only valid with --measured-run").into());
            }
        }
        if otlp_metrics_endpoint.is_some() && cache_calibration {
            return Err("--otlp-metrics-endpoint cannot be combined with --cache-calibration".into());
        }
        Ok(Self {
            config,
            out_dir,
            mode,
            nsight_trace,
            otlp_metrics_endpoint,
            otlp_export_interval_ms,
            quarantine_device,
            cancel_first_after,
            deadline_first_after,
            domain_exhaustion_probe,
            overload_fairness,
            physical_allocator_pressure,
            cache_calibration,
            quarantine_cycles,
            close_under_load,
            slow_report_sink,
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
#[cfg(any(test, not(feature = "single-node-multi-gpu-validate-run")))]
fn measured_run_preflight(
    profile: &RunProfile,
    quarantine_device: Option<usize>,
    deadline_first_after: Option<std::time::Duration>,
    domain_exhaustion_probe: bool,
    overload_fairness: bool,
    physical_allocator_pressure: bool,
    cache_calibration: bool,
    quarantine_cycles: Option<usize>,
    close_under_load: bool,
) -> ToolResult<Vec<FixturePreflightArtifact>> {
    measured_run_preflight_with_completion_sink(
        profile,
        quarantine_device,
        deadline_first_after,
        domain_exhaustion_probe,
        overload_fairness,
        physical_allocator_pressure,
        cache_calibration,
        quarantine_cycles,
        close_under_load,
        None,
    )
}

/// Extends the established measured-run preflight with I8's dedicated sink
/// shape without widening the helpers used by the existing scenario tests.
fn measured_run_preflight_with_completion_sink(
    profile: &RunProfile,
    quarantine_device: Option<usize>,
    deadline_first_after: Option<std::time::Duration>,
    domain_exhaustion_probe: bool,
    overload_fairness: bool,
    physical_allocator_pressure: bool,
    cache_calibration: bool,
    quarantine_cycles: Option<usize>,
    close_under_load: bool,
    slow_report_sink: Option<std::time::Duration>,
) -> ToolResult<Vec<FixturePreflightArtifact>> {
    if profile.selected_query_count() == 0 {
        return Err("--measured-run requires at least one workload query selection".into());
    }
    if profile.backend.device_profiles.is_empty() {
        return Err("--measured-run requires at least one backend.device_profiles entry".into());
    }
    if profile.scheduling.measurement_seconds == 0 && profile.scheduling.steps.is_empty() {
        return Err(
            "--measured-run requires a non-zero scheduling.measurement_seconds or a scheduling.steps list".into()
        );
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
        if labels.as_slice() != ["q01", "edge_scan"] {
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
    if let Some(cycles) = quarantine_cycles {
        // Each cycle needs a distinct target, and the fleet must keep at
        // least one open ordinal so stabilization between cycles is real.
        if profile.backend.device_profiles.len() < 2 {
            return Err("--quarantine-cycles requires at least two configured devices".into());
        }
        if cycles > profile.backend.device_profiles.len() - 1 {
            return Err(format!(
                "--quarantine-cycles {cycles} exceeds the configured device count minus one {}; each cycle needs a distinct target and the fleet needs one open ordinal",
                profile.backend.device_profiles.len() - 1
            )
            .into());
        }
        if profile.scheduling.arrival_mode != profile::ArrivalMode::Concurrency {
            return Err(
                "--quarantine-cycles requires scheduling.arrival_mode=concurrency for its sustained closed-loop load"
                    .into(),
            );
        }
        if profile.scheduling.concurrency < profile.backend.device_profiles.len() {
            return Err(
                "--quarantine-cycles requires scheduling.concurrency of at least the configured device count so every cycle target has live work"
                    .into(),
            );
        }
    }
    if close_under_load {
        if profile.scheduling.arrival_mode == profile::ArrivalMode::Concurrency && profile.scheduling.concurrency < 2 {
            return Err(
                "--close-under-load requires scheduling.concurrency of at least 2 so close lands with work in flight"
                    .into(),
            );
        }
        // Close must land with a queued cohort, or the probe cannot prove
        // that queued waiters seal `backend_closing`.
        let active_capacity =
            profile.backend.device_profiles.iter().map(|device| device.max_active_attempts).sum::<usize>();
        if profile.scheduling.arrival_mode != profile::ArrivalMode::Concurrency
            || profile.scheduling.concurrency <= active_capacity
        {
            return Err(format!(
                "--close-under-load requires concurrency scheduling with scheduling.concurrency above the fleet active-attempt capacity {active_capacity} so a queued cohort exists at close"
            )
            .into());
        }
    }
    if let Some(sink_delay) = slow_report_sink {
        if profile.scheduling.arrival_mode != profile::ArrivalMode::Rate {
            return Err("--slow-report-sink-ms requires scheduling.arrival_mode=rate".into());
        }
        let sink_delay_ms = sink_delay.as_millis();
        let rates = if profile.scheduling.steps.is_empty() {
            vec![profile.scheduling.arrival_rate_per_second]
        } else {
            profile.scheduling.steps.iter().map(|step| step.arrival_rate_per_second).collect()
        };
        if rates.iter().any(|rate| rate.is_none_or(|rate| u128::from(rate).saturating_mul(sink_delay_ms) <= 1_000)) {
            return Err(
                "--slow-report-sink-ms requires every rate step to offer more completions per second than the one-per-delay sink can drain"
                    .into(),
            );
        }
    }
    if (quarantine_cycles.is_some() || close_under_load)
        && (profile.scheduling.attempt_deadline_ms.is_some() || profile.scheduling.cancel_every_n.is_some())
    {
        return Err(
            "S5 storm knobs (attempt_deadline_ms, cancel_every_n) cannot be combined with --quarantine-cycles or --close-under-load"
                .into(),
        );
    }
    let executed =
        profile.workloads.iter().filter(|workload| !workload.queries.is_empty()).cloned().collect::<Vec<_>>();
    FixturePreflight::collect(&executed, FixtureSelectionPolicy::AtLeastOneKind)
}

#[cfg(feature = "single-node-multi-gpu-validate-run")]
fn run_measured_run(profile: RunProfile, config_sha256: String, options: &CliOptions) -> ToolResult<PathBuf> {
    #[cfg(not(feature = "single-node-multi-gpu-validate-run-otlp"))]
    if options.otlp_metrics_endpoint.is_some() {
        return Err("--otlp-metrics-endpoint requires --features single-node-multi-gpu-validate-run-otlp".into());
    }
    let fixtures = measured_run_preflight_with_completion_sink(
        &profile,
        options.quarantine_device,
        options.deadline_first_after,
        options.domain_exhaustion_probe,
        options.overload_fairness,
        options.physical_allocator_pressure,
        options.cache_calibration,
        options.quarantine_cycles,
        options.close_under_load,
        options.slow_report_sink,
    )?;
    let otlp_export = options
        .otlp_metrics_endpoint
        .as_deref()
        .map(|endpoint| {
            measure::OtlpExportConfig::new(
                endpoint,
                std::time::Duration::from_millis(options.otlp_export_interval_ms.unwrap_or(1_000)),
            )
        })
        .transpose()?;
    let probes = measure::ProbeSelection {
        quarantine_device: options.quarantine_device,
        cancel_first_after: options.cancel_first_after,
        deadline_first_after: options.deadline_first_after,
        domain_exhaustion_probe: options.domain_exhaustion_probe,
        overload_fairness: options.overload_fairness,
        physical_allocator_pressure: options.physical_allocator_pressure,
        quarantine_cycles: options.quarantine_cycles,
        close_under_load: options.close_under_load,
        otlp_export,
        slow_report_sink: options.slow_report_sink,
    };
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
            measure::run_on_owned_runtime(&cache_disabled, trace.clone(), measure::ProbeSelection::default())?;
        let opt_in_measurement = measure::run_on_owned_runtime(&profile, trace, measure::ProbeSelection::default())?;
        RunArtifact::cache_calibration(
            profile,
            config_sha256,
            fixtures,
            provenance,
            disabled_measurement,
            opt_in_measurement,
        )
    } else {
        let measurement = measure::run_on_owned_runtime(&profile, trace, probes)?;
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
        _options.quarantine_cycles,
        _options.close_under_load,
    )?;
    Err("--measured-run requires --features single-node-multi-gpu-validate-run".into())
}

#[cfg(test)]
mod tests;
