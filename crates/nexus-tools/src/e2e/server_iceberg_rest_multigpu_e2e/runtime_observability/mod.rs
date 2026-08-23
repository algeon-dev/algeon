use std::{fs, time::Instant};

use arrow::record_batch::RecordBatch;
use arrow_flight::{
    FlightData, FlightInfo, Ticket, decode::FlightRecordBatchStream, error::FlightError,
    flight_service_client::FlightServiceClient, sql::client::FlightSqlServiceClient,
};
use futures::TryStreamExt;
use prost::Message as _;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::{
    sync::{Barrier, mpsc},
    task::JoinSet,
};
use tonic::{Code, Request, transport::Channel};

use super::{
    E2eResult,
    flight::{
        CANCEL_RESULT_CANCELLING, action_frames, cancel_query, get_admission_stats, get_query_stats,
        query_id_from_statement_flight_info,
    },
    server::{ProcessExit, ServerChild, ServerSpawnConfig},
    validation::{parse_u64_field, tsv_section_single_row, wait_for_idle},
};

mod datafusion_baseline_phase;
mod evidence;
mod failures;
mod native_matrix;
mod params;
mod process;
mod retention_restart;
mod util;

use datafusion_baseline_phase::*;
use evidence::*;
use failures::*;
use native_matrix::*;
use params::*;
use process::*;
use retention_restart::*;
use util::*;

pub(super) async fn run(args: Vec<String>) -> E2eResult<()> {
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
    evidence.fatal_patterns = fatal_result.evidence;
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

#[cfg(test)]
mod tests;
