use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use tonic::transport::Channel;

use super::{
    E2eResult,
    flight::{CANCEL_RESULT_CANCELLING, cancel_query, get_admission_stats, lightweight_get_flight_info},
    validation::{parse_u64_field, tsv_section_single_row},
};

pub(super) const SATURATION_FLOOR: u64 = 8;
pub(super) const REQUIRED_SATURATED_PROBES: usize = 20;
pub(super) const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Default)]
pub(super) struct ProbeReport {
    pub(super) rows: Vec<ProbeRow>,
    pub(super) saturated_count: usize,
    pub(super) peak_score: u64,
    pub(super) peak_payload: String,
    pub(super) tickets_issued: usize,
    pub(super) tickets_retired: usize,
}

pub(super) struct ProbeRow {
    pub(super) sequence: usize,
    pub(super) active_queries: u64,
    pub(super) queued_queries: u64,
    pub(super) active_gpu_jobs: u64,
    pub(super) diagnostics_ms: u128,
    pub(super) get_flight_info_ms: u128,
    pub(super) ticket_retire_ms: u128,
    pub(super) saturated: bool,
}

pub(super) async fn run_probes(channel: Channel, stop: Arc<AtomicBool>) -> E2eResult<ProbeReport> {
    let mut report = ProbeReport::default();
    while !stop.load(Ordering::Acquire) {
        let diagnostics_started = Instant::now();
        let payload = tokio::time::timeout(PROBE_TIMEOUT, get_admission_stats(&channel))
            .await
            .map_err(|_| "GetAdmissionStats exceeded two-second liveness bound")??;
        let diagnostics_ms = diagnostics_started.elapsed().as_millis();
        let admission = tsv_section_single_row(&payload, "admission")?;
        let gpu_execution = tsv_section_single_row(&payload, "gpu_execution")?;
        let active_queries = parse_u64_field(&admission, "active_queries", "admission probe")?;
        let queued_queries = parse_u64_field(&admission, "queued_queries", "admission probe")?;
        let active_gpu_jobs = parse_u64_field(&gpu_execution, "active_query_jobs", "admission probe")?;
        let score = active_queries.max(active_gpu_jobs);
        if score > report.peak_score || report.peak_payload.is_empty() {
            report.peak_score = score;
            report.peak_payload.clone_from(&payload);
        }

        let info_started = Instant::now();
        let flight_info = tokio::time::timeout(PROBE_TIMEOUT, lightweight_get_flight_info(&channel))
            .await
            .map_err(|_| "GetFlightInfo(SELECT 1) exceeded two-second liveness bound")??;
        let get_flight_info_ms = info_started.elapsed().as_millis();
        report.tickets_issued += 1;
        let retire_started = Instant::now();
        let cancel_result = tokio::time::timeout(PROBE_TIMEOUT, cancel_query(&channel, &flight_info))
            .await
            .map_err(|_| "CancelQuery for liveness ticket exceeded two-second bound")??;
        if cancel_result.result != CANCEL_RESULT_CANCELLING {
            return Err(format!(
                "liveness ticket CancelQuery returned {}, expected {CANCEL_RESULT_CANCELLING}",
                cancel_result.result
            )
            .into());
        }
        let ticket_retire_ms = retire_started.elapsed().as_millis();
        report.tickets_retired += 1;
        let saturated = score >= SATURATION_FLOOR;
        if saturated {
            report.saturated_count += 1;
        }
        report.rows.push(ProbeRow {
            sequence: report.rows.len() + 1,
            active_queries,
            queued_queries,
            active_gpu_jobs,
            diagnostics_ms,
            get_flight_info_ms,
            ticket_retire_ms,
            saturated,
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    Ok(report)
}
