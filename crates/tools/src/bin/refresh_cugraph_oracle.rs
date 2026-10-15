//! Driver that regenerates local cuGraph oracle JSON fixtures.
//!
//! Iterates [`CUGRAPH_PARITY_SPECS`] and invokes
//! `fixture/graph/cugraph_oracle.py` for each generated-oracle row. Python
//! cuGraph is optional per row. Exit code 77 is reported as "oracle unavailable".
//! Generated JSON lives under `fixture/graph/cugraph/oracle/`.
//!
//! Standard invocation:
//!
//! ```bash
//! flock /tmp/cudf-gpu.lock \
//!     cargo run --release -p tools --features cugraph-oracle --bin refresh_cugraph_oracle -- --all
//! ```
//!
//! Execution flags:
//!
//! - `--all`: refresh every parity row's oracle JSON slot.
//! - `--algorithm <name>` / `--dataset <name>`: restrict the run to
//!   matching specs.
//! - `--force`: overwrite drifted JSON. Without it, a drifted row writes a sibling preview.
//! - `--check`: verify local JSON matches a fresh regeneration without writing.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use algeon_datafusion::cugraph_sql::parity::{
    CUGRAPH_DATA_DIR_FROM_CARGO_MANIFEST, CUGRAPH_PARITY_SPECS, ParitySpec, ParityTruthSource, load_manifest,
    oracle_path,
};

const ORACLE_SCRIPT_REL: &str = "fixture/graph/cugraph_oracle.py";

#[derive(Debug, Default)]
struct CliOptions {
    select_all: bool,
    algorithm: Option<String>,
    dataset: Option<String>,
    check_only: bool,
    force: bool,
    python: PathBuf,
    workspace_root: Option<PathBuf>,
}

fn print_usage() {
    eprintln!(
        "refresh_cugraph_oracle: regenerate local cuGraph oracle JSON\n\
         \n\
         USAGE:\n\
             refresh_cugraph_oracle [--all] [--algorithm NAME] [--dataset NAME]\n\
                                   [--check] [--force] [--python PATH]\n\
         \n\
         Refreshes one row per matching parity spec by invoking\n\
         fixture/graph/cugraph_oracle.py. Requires Python cuGraph in PATH.",
    );
}

fn parse_args() -> Result<CliOptions, String> {
    let mut opts = CliOptions {
        python: PathBuf::from(std::env::var("ALGEON_CUGRAPH_PYTHON").unwrap_or_else(|_| "python3".to_string())),
        ..Default::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--all" => opts.select_all = true,
            "--algorithm" => {
                opts.algorithm = Some(args.next().ok_or_else(|| "--algorithm needs a value".to_string())?);
            },
            "--dataset" => {
                opts.dataset = Some(args.next().ok_or_else(|| "--dataset needs a value".to_string())?);
            },
            "--check" => opts.check_only = true,
            "--force" => opts.force = true,
            "--python" => {
                opts.python = PathBuf::from(args.next().ok_or_else(|| "--python needs a value".to_string())?);
            },
            "--workspace-root" => {
                opts.workspace_root =
                    Some(PathBuf::from(args.next().ok_or_else(|| "--workspace-root needs a value".to_string())?));
            },
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            },
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    if !opts.select_all && opts.algorithm.is_none() && opts.dataset.is_none() {
        return Err("specify at least one of --all, --algorithm, --dataset (see --help)".to_string());
    }
    if opts.check_only && opts.force {
        return Err("--check and --force cannot be combined".to_string());
    }
    Ok(opts)
}

#[derive(Debug)]
enum RefreshOutcome {
    Written {
        path: PathBuf,
    },
    UnchangedOnDisk {
        path: PathBuf,
    },
    DriftDetected {
        path: PathBuf,
        preview_written: bool,
    },
    /// `--check` was requested but the local oracle JSON does not
    /// exist yet. Kept distinct from `DriftDetected` to produce an explicit
    /// failure when the on-disk artifact is entirely absent.
    MissingInCheckMode {
        path: PathBuf,
    },
    OracleUnavailable {
        stderr: String,
    },
    PythonError {
        code: i32,
        stderr: String,
    },
}

fn run() -> Result<i32, Box<dyn std::error::Error>> {
    let opts = match parse_args() {
        Ok(opts) => opts,
        Err(error) => {
            eprintln!("{error}");
            print_usage();
            return Ok(2);
        },
    };

    let workspace_root = match &opts.workspace_root {
        Some(root) => root.clone(),
        None => default_workspace_root()?,
    };
    let data_root = fixture_data_root(&workspace_root);
    let manifest = load_manifest(&data_root)?;

    let oracle_script = workspace_root.join(ORACLE_SCRIPT_REL);
    if !oracle_script.exists() {
        return Err(format!("oracle script missing at {}", oracle_script.display()).into());
    }

    let mut summary: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut exit_code = 0i32;
    // Refresh every generated-oracle row regardless of `truth_source`.
    // RefCsv-truth rows (e.g. Katz) still carry a secondary
    // `oracle_relpath` slot for the per-vertex score cross-check. A
    // truth-source filter would make
    // `refresh_cugraph_oracle --algorithm katz_centrality --check` report
    // nothing without an error, and CI would miss a missing secondary oracle. The
    // runtime parity comparator still consults `truth_source` to pick the
    // primary truth at compare time.
    for spec in CUGRAPH_PARITY_SPECS {
        if !spec_matches(&opts, spec) {
            continue;
        }
        if matches!(spec.truth_source, ParityTruthSource::HandVerified { .. }) {
            *summary.entry("hand_verified").or_default() += 1;
            continue;
        }
        if manifest.dataset(spec.dataset).is_none() {
            return Err(format!("parity spec '{}' references unknown dataset '{}'", spec.label, spec.dataset).into());
        }
        let outcome = refresh_one(&opts, &data_root, &oracle_script, spec)?;
        let key = report_outcome(spec, &outcome);
        *summary.entry(key).or_default() += 1;
        if matches!(outcome, RefreshOutcome::DriftDetected { .. }) && !opts.force {
            exit_code = 1;
        }
        if matches!(outcome, RefreshOutcome::MissingInCheckMode { .. } | RefreshOutcome::PythonError { .. }) {
            exit_code = 1;
        }
    }

    eprintln!("refresh_cugraph_oracle summary:");
    for (kind, count) in &summary {
        eprintln!("  {kind}: {count}");
    }
    Ok(exit_code)
}

fn default_workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut current = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if current.join(ORACLE_SCRIPT_REL).is_file() {
            return Ok(current);
        }
        if !current.pop() {
            return Err(format!(
                "could not find workspace root containing {ORACLE_SCRIPT_REL}; pass --workspace-root explicitly"
            )
            .into());
        }
    }
}

fn fixture_data_root(workspace_root: &Path) -> PathBuf {
    if let Ok(env) = std::env::var("ALGEON_CUGRAPH_DATA_ROOT") {
        return PathBuf::from(env);
    }
    workspace_root.join(CUGRAPH_DATA_DIR_FROM_CARGO_MANIFEST)
}

fn spec_matches(opts: &CliOptions, spec: &ParitySpec) -> bool {
    if opts.select_all {
        return true;
    }
    let algorithm_ok = opts.algorithm.as_deref().is_none_or(|name| spec.algorithm.code() == name);
    let dataset_ok = opts.dataset.as_deref().is_none_or(|name| spec.dataset == name);
    algorithm_ok && dataset_ok
}

fn refresh_one(
    opts: &CliOptions,
    data_root: &Path,
    oracle_script: &Path,
    spec: &ParitySpec,
) -> Result<RefreshOutcome, Box<dyn std::error::Error>> {
    let oracle_path =
        oracle_path(data_root, spec).expect("refresh only receives parity specs with a generated oracle path");

    // In `--check` mode, missing local JSON must fail even when Python cuGraph
    // is unavailable. `--force` only controls drift overwrite semantics.
    if opts.check_only && !oracle_path.exists() {
        return Ok(RefreshOutcome::MissingInCheckMode { path: oracle_path });
    }

    let mut command = Command::new(&opts.python);
    command
        .arg(oracle_script)
        .arg("--data-root")
        .arg(data_root)
        .arg("--algorithm")
        .arg(spec.algorithm.code())
        .arg("--dataset")
        .arg(spec.dataset)
        .arg("--params")
        .arg(spec.params_json);
    if let Some(seed) = spec.seed {
        command.arg("--seed").arg(seed.to_string());
    }
    let output = command.output().map_err(|error| format!("failed to spawn {:?}: {error}", opts.python))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Ok(match output.status.code() {
            Some(77) => RefreshOutcome::OracleUnavailable { stderr },
            code => RefreshOutcome::PythonError { code: code.unwrap_or(-1), stderr },
        });
    }
    let payload = String::from_utf8(output.stdout)?;
    let fresh_value: serde_json::Value = serde_json::from_str(&payload)?;

    if let Some(parent) = oracle_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if oracle_path.exists() {
        let existing = fs::read_to_string(&oracle_path)?;
        let existing_value: serde_json::Value = serde_json::from_str(&existing)?;
        if json_canonical(&existing_value) == json_canonical(&fresh_value) {
            return Ok(RefreshOutcome::UnchangedOnDisk { path: oracle_path });
        }
        if opts.check_only && !opts.force {
            return Ok(RefreshOutcome::DriftDetected { path: oracle_path, preview_written: false });
        }
        if !opts.force {
            // Write to a sibling .new file so the operator can inspect
            // the diff manually before overwriting the local fixture.
            let preview = oracle_path.with_extension("new.json");
            fs::write(&preview, serde_json::to_string_pretty(&fresh_value)? + "\n")?;
            return Ok(RefreshOutcome::DriftDetected { path: preview, preview_written: true });
        }
    }
    if !opts.check_only {
        fs::write(&oracle_path, serde_json::to_string_pretty(&fresh_value)? + "\n")?;
    }
    Ok(RefreshOutcome::Written { path: oracle_path })
}

fn json_canonical(value: &serde_json::Value) -> String {
    // serde_json::Value preserves Object insertion order; emit a sorted
    // canonical form so trivially reordered objects compare equal.
    fn sort(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut sorted: std::collections::BTreeMap<&str, serde_json::Value> = std::collections::BTreeMap::new();
                for (key, v) in map {
                    sorted.insert(key, sort(v));
                }
                serde_json::Value::Object(sorted.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
            },
            serde_json::Value::Array(values) => serde_json::Value::Array(values.iter().map(sort).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sort(value)).expect("canonical oracle JSON must be serializable")
}

fn report_outcome(spec: &ParitySpec, outcome: &RefreshOutcome) -> &'static str {
    match outcome {
        RefreshOutcome::Written { path } => {
            eprintln!("  refreshed {label} -> {path}", label = spec.label, path = path.display());
            "refreshed"
        },
        RefreshOutcome::UnchangedOnDisk { path } => {
            eprintln!("  unchanged {label} at {path}", label = spec.label, path = path.display());
            "unchanged"
        },
        RefreshOutcome::DriftDetected { path, preview_written } => {
            if *preview_written {
                eprintln!(
                    "  drift {label}: new oracle written to {path} (rerun with --force to overwrite the local JSON)",
                    label = spec.label,
                    path = path.display()
                );
            } else {
                eprintln!(
                    "  drift {label}: fresh oracle differs from {path} (--check did not write)",
                    label = spec.label,
                    path = path.display()
                );
            }
            "drift"
        },
        RefreshOutcome::MissingInCheckMode { path } => {
            eprintln!(
                "  missing {label}: oracle JSON not on disk at {path} (run without --check to populate it)",
                label = spec.label,
                path = path.display()
            );
            "missing"
        },
        RefreshOutcome::OracleUnavailable { stderr } => {
            eprintln!("  oracle unavailable for {label}: {message}", label = spec.label, message = stderr.trim());
            "oracle_unavailable"
        },
        RefreshOutcome::PythonError { code, stderr } => {
            eprintln!(
                "  python error for {label}: exit={code} stderr={message}",
                label = spec.label,
                message = stderr.trim()
            );
            "python_error"
        },
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(error) => {
            eprintln!("refresh_cugraph_oracle failed: {error}");
            ExitCode::from(1)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_matches_filters_by_algorithm_and_dataset() {
        let opts = CliOptions {
            select_all: false,
            algorithm: Some("pagerank".into()),
            dataset: Some("karate".into()),
            ..Default::default()
        };
        let pagerank_karate = CUGRAPH_PARITY_SPECS
            .iter()
            .find(|spec| spec.label.starts_with("pagerank/karate"))
            .expect("Milestone 1a registry must include karate PageRank");
        assert!(spec_matches(&opts, pagerank_karate));
        let other = CUGRAPH_PARITY_SPECS
            .iter()
            .find(|spec| !spec.label.starts_with("pagerank/karate"))
            .expect("registry has more than one entry");
        assert!(!spec_matches(&opts, other));
    }

    #[test]
    fn select_all_matches_every_spec() {
        let opts = CliOptions { select_all: true, ..Default::default() };
        for spec in CUGRAPH_PARITY_SPECS {
            assert!(spec_matches(&opts, spec), "select-all must match {}", spec.label);
        }
    }

    #[test]
    fn default_workspace_root_finds_repo_root_from_tooling_crate() {
        let root = default_workspace_root().expect("workspace root should be discoverable");
        assert!(root.join(ORACLE_SCRIPT_REL).is_file(), "root={} missing {ORACLE_SCRIPT_REL}", root.display());
    }

    #[test]
    fn fixture_data_root_defaults_under_workspace_root() {
        let root = Path::new("/tmp/example-workspace");
        if std::env::var_os("ALGEON_CUGRAPH_DATA_ROOT").is_some() {
            return;
        }
        assert_eq!(fixture_data_root(root), root.join(CUGRAPH_DATA_DIR_FROM_CARGO_MANIFEST));
    }
}
