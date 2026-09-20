//! Black-box runtime-observability acceptance for the `algeon_server` binary.
//!
//! This is an explicit tool, not a Cargo test. It starts the server as a child
//! process over local Parquet files, drives Flight SQL statements and the
//! query-stats and runtime-artifact actions, restarts the server, and writes one
//! evidence JSON. See `--help` for the options.

mod datafusion_baseline_phase;
mod evidence;
mod failures;
mod flight;
mod native_matrix;
mod params;
mod process;
mod retention_restart;
mod server;
#[cfg(test)]
mod tests;
mod validation;

use std::{error::Error, fs, time::Instant};

use self::{
    evidence::{RunEvidence, millis, scan_fatal_patterns},
    native_matrix::run_matrix,
    params::{RuntimeParams, print_usage},
    server::ensure_parent,
};

type E2eResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub async fn run(args: Vec<String>) -> E2eResult<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return Ok(());
    }
    let params = RuntimeParams::parse(args)?;
    params.validate_filesystem()?;
    ensure_parent(&params.evidence_path)?;
    ensure_parent(&params.server_log_path)?;

    let started = Instant::now();
    let mut evidence = RunEvidence::new(&params);
    let run_result = run_matrix(&params, &mut evidence).await;
    let fatal_result = scan_fatal_patterns(&evidence.server_logs);
    evidence.elapsed_ms = millis(started.elapsed());
    evidence.fatal_patterns = fatal_result;
    if let Err(error) = &run_result {
        evidence.status = "failed".to_string();
        evidence.failure = Some(error.to_string());
    } else if !evidence.fatal_patterns.matches.is_empty() {
        evidence.status = "failed".to_string();
        evidence.failure = Some("fatal server log pattern detected".to_string());
    } else {
        evidence.status = "passed".to_string();
    }
    fs::write(&params.evidence_path, serde_json::to_vec_pretty(&evidence)?)?;

    run_result?;
    if !evidence.fatal_patterns.matches.is_empty() {
        return Err(format!("fatal server log patterns: {:?}", evidence.fatal_patterns.matches).into());
    }
    println!("server_runtime_observability_e2e ok: {}", params.evidence_path.display());
    Ok(())
}
