//! Versioned P6 production-validation run artifacts.
//!
//! P6.3 owns the dry run, P6.4 the read-only fixture preflight, and P6.5 the
//! measured public-API run. Config parsing, arrival planning, aggregation, and
//! artifact writing are unit-tested library code; only `measure` touches a GPU.

mod aggregate;
mod artifact;
#[cfg(feature = "single-node-multi-gpu-validate-run")]
mod measure;
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
  single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --measured-run [--nsight-trace PATH]

--dry-run accepts only an empty workload list. --fixture-preflight validates one TPC-H and one DBLP fixture without generating, downloading, overwriting, or executing a workload. --measured-run executes the profile's workload mix on the public backend and attempt API and requires the single-node-multi-gpu-validate-run feature. --nsight-trace records an externally captured trace's identity; this runner never captures one itself."
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    config: PathBuf,
    out_dir: PathBuf,
    mode: ValidationMode,
    /// An externally captured Nsight trace whose identity the artifact records.
    nsight_trace: Option<PathBuf>,
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
        Ok(Self { config, out_dir, mode, nsight_trace })
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
fn measured_run_preflight(profile: &RunProfile) -> ToolResult<Vec<FixturePreflightArtifact>> {
    if profile.selected_query_count() == 0 {
        return Err("--measured-run requires at least one workload query selection".into());
    }
    if profile.backend.device_profiles.is_empty() {
        return Err("--measured-run requires at least one backend.device_profiles entry".into());
    }
    if profile.scheduling.measurement_seconds == 0 {
        return Err("--measured-run requires a non-zero scheduling.measurement_seconds".into());
    }
    let executed =
        profile.workloads.iter().filter(|workload| !workload.queries.is_empty()).cloned().collect::<Vec<_>>();
    FixturePreflight::collect(&executed, FixtureSelectionPolicy::AtLeastOneKind)
}

#[cfg(feature = "single-node-multi-gpu-validate-run")]
fn run_measured_run(profile: RunProfile, config_sha256: String, options: &CliOptions) -> ToolResult<PathBuf> {
    let fixtures = measured_run_preflight(&profile)?;
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
    let measurement = measure::run_on_owned_runtime(&profile, trace)?;
    let qualifying = measurement.qualifying;
    let artifact = RunArtifact::measured_run(profile, config_sha256, fixtures, provenance, measurement);
    let path = write_artifact(&options.out_dir, &artifact)?;
    if !qualifying {
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
    measured_run_preflight(&profile)?;
    Err("--measured-run requires --features single-node-multi-gpu-validate-run".into())
}

#[cfg(test)]
mod tests;
