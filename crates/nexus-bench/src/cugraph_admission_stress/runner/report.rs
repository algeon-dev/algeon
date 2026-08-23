use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::PathBuf,
    time::Instant,
};

use super::*;

pub(super) const QUERY_COLUMNS: &[&str] = &[
    "run_id",
    "label",
    "task_index",
    "iteration",
    "start_epoch_ms",
    "end_epoch_ms",
    "wall_ms",
    "outcome",
    "admission_sequence",
    "admission_wait_ms",
    "device_ordinal",
    "query_grant_bytes",
    "reservation_requested_bytes",
    "reservation_granted_bytes",
    "reservation_live_peak_bytes",
    "reservation_denied_count",
    "allocator_high_water_bytes",
    "max_task_observed_peak_bytes",
    "max_task_reservation_bytes",
    "graph_algorithm",
    "graph_class",
    "graph_outcome",
    "graph_raw_input_edges",
    "graph_input_edges",
    "graph_vertex_estimate",
    "graph_vertex_estimate_source",
    "graph_vertex_width_bytes",
    "graph_weight_width_bytes",
    "graph_directed",
    "graph_renumber",
    "graph_construction_policy",
    "graph_edge_normalize_inserted",
    "graph_calibration_source",
    "graph_seed_count",
    "graph_seed_mode",
    "graph_model_requested_bytes",
    "graph_requested_bytes",
    "graph_budget_bytes",
    "graph_model_error_ratio",
    "graph_statistics_exact_vertices",
    "graph_statistics_degree_max",
    "graph_statistics_degree_p99",
    "graph_statistics_sum_min_degree_oriented",
    "graph_statistics_oriented_edge_count",
    "graph_statistics_peak_chunk_min_degree_sum",
    "graph_statistics_sum_out_degree",
    "graph_peak_bytes",
    "graph_reason",
    "graph_elapsed_ms",
    "graph_output_device_bytes",
    "output_rows",
    "native_lowered",
    "error",
];

pub(super) const SAMPLE_COLUMNS: &[&str] = &[
    "epoch_ms",
    "governor_capacity_bytes",
    "governor_outstanding_bytes",
    "governor_available_bytes",
    "active_grants",
    "active_queries",
    "queued_queries",
    "pool_reserved_current_bytes",
    "pool_reserved_high_bytes",
    "pool_used_current_bytes",
    "pool_used_high_bytes",
    "allocation_failure_count",
    "trim_retry_count",
];

pub(super) const DEVICE_SAMPLE_COLUMNS: &[&str] = &[
    "epoch_ms",
    "device_ordinal",
    "governor_capacity_bytes",
    "governor_outstanding_bytes",
    "governor_available_bytes",
    "active_grants",
    "active_queries",
    "queued_queries",
    "gpu_execution_worker_count",
    "gpu_execution_active_jobs",
    "gpu_execution_queued_jobs",
    "gpu_execution_active_query_jobs",
    "pool_reserved_current_bytes",
    "pool_reserved_high_bytes",
    "pool_used_current_bytes",
    "pool_used_high_bytes",
    "allocation_failure_count",
    "trim_retry_count",
];

impl QueryRow {
    pub(super) fn new(config: &HarnessConfig, request: &Request, iteration: usize, start_epoch_ms: u128) -> Self {
        Self {
            run_id: config.run_id.clone(),
            label: request.spec.label.clone(),
            task_index: request.task_index,
            iteration,
            start_epoch_ms,
            end_epoch_ms: 0,
            wall_ms: 0.0,
            outcome: "unknown",
            admission_sequence: None,
            admission_wait_ms: None,
            device_ordinal: None,
            query_grant_bytes: None,
            reservation_requested_bytes: 0,
            reservation_granted_bytes: 0,
            reservation_live_peak_bytes: 0,
            reservation_denied_count: 0,
            allocator_high_water_bytes: None,
            max_task_observed_peak_bytes: None,
            max_task_reservation_bytes: 0,
            graph_algorithm: None,
            graph_class: None,
            graph_outcome: None,
            graph_raw_input_edges: None,
            graph_input_edges: None,
            graph_vertex_estimate: None,
            graph_vertex_estimate_source: None,
            graph_vertex_width_bytes: None,
            graph_weight_width_bytes: None,
            graph_directed: None,
            graph_renumber: None,
            graph_construction_policy: None,
            graph_edge_normalize_inserted: None,
            graph_calibration_source: None,
            graph_seed_count: None,
            graph_seed_mode: None,
            graph_model_requested_bytes: None,
            graph_requested_bytes: None,
            graph_budget_bytes: None,
            graph_model_error_ratio: None,
            graph_statistics_exact_vertices: None,
            graph_statistics_degree_max: None,
            graph_statistics_degree_p99: None,
            graph_statistics_sum_min_degree_oriented: None,
            graph_statistics_oriented_edge_count: None,
            graph_statistics_peak_chunk_min_degree_sum: None,
            graph_statistics_sum_out_degree: None,
            graph_peak_bytes: None,
            graph_reason: None,
            graph_elapsed_ms: None,
            graph_output_device_bytes: None,
            output_rows: None,
            output_fingerprint: None,
            native_lowered: false,
            error: None,
        }
    }

    pub(super) fn absorb_metrics(&mut self, metrics: &NativeDataFusionExecutionMetrics) {
        self.admission_sequence = metrics.admission_sequence;
        self.admission_wait_ms = metrics.admission_wait_time_ns.map(|ns| ns as f64 / 1_000_000.0);
        self.device_ordinal = metrics.device_ordinal;
        self.query_grant_bytes = metrics.query_grant_bytes;
        self.reservation_requested_bytes = metrics.reservation_requested_bytes;
        self.reservation_granted_bytes = metrics.reservation_granted_bytes;
        self.reservation_live_peak_bytes = metrics.reservation_live_peak_bytes;
        self.reservation_denied_count = metrics.reservation_denied_count;
        self.allocator_high_water_bytes = metrics.allocator_high_water_bytes;
        self.max_task_observed_peak_bytes = metrics.max_task_observed_peak_bytes;
        self.max_task_reservation_bytes = metrics.max_task_reservation_bytes;
        if let Some(reservation) = metrics.graph_reservation_rows.first() {
            self.graph_algorithm = Some(reservation.algorithm.clone());
            self.graph_class = Some(reservation.class.clone());
            self.graph_outcome = Some(reservation.outcome.clone());
            self.graph_raw_input_edges = reservation.raw_input_edges;
            self.graph_input_edges = Some(reservation.input_edges);
            self.graph_vertex_estimate = Some(reservation.vertex_estimate);
            self.graph_vertex_estimate_source = Some(reservation.vertex_estimate_source.clone());
            self.graph_vertex_width_bytes = Some(reservation.vertex_width_bytes);
            self.graph_weight_width_bytes = reservation.weight_width_bytes;
            self.graph_directed = Some(reservation.directed);
            self.graph_renumber = Some(reservation.renumber);
            self.graph_construction_policy = Some(reservation.construction_policy.clone());
            self.graph_edge_normalize_inserted = Some(reservation.edge_normalize_inserted);
            self.graph_calibration_source = Some(reservation.calibration_source.clone());
            self.graph_seed_count = reservation.seed_count;
            self.graph_seed_mode = Some(reservation.seed_mode.clone());
            self.graph_model_requested_bytes = Some(reservation.model_requested_bytes);
            self.graph_requested_bytes = Some(reservation.requested_bytes);
            self.graph_budget_bytes = reservation.budget_bytes;
            self.graph_model_error_ratio = reservation.graph_model_error_ratio.clone();
            self.graph_statistics_exact_vertices = reservation.graph_statistics_exact_vertices;
            self.graph_statistics_degree_max = reservation.graph_statistics_degree_max;
            self.graph_statistics_degree_p99 = reservation.graph_statistics_degree_p99;
            self.graph_statistics_sum_min_degree_oriented = reservation.graph_statistics_sum_min_degree_oriented;
            self.graph_statistics_oriented_edge_count = reservation.graph_statistics_oriented_edge_count;
            self.graph_statistics_peak_chunk_min_degree_sum = reservation.graph_statistics_peak_chunk_min_degree_sum;
            self.graph_statistics_sum_out_degree = reservation.graph_statistics_sum_out_degree;
            self.graph_reason = Some(reservation.reason.clone());
        }
        if let Some(runtime) = metrics.graph_runtime_rows.first() {
            self.graph_elapsed_ms = Some(runtime.elapsed_ns as f64 / 1_000_000.0);
            self.graph_output_device_bytes = runtime.output_device_bytes;
            self.graph_peak_bytes = runtime.graph_peak_bytes;
            if self.graph_algorithm.is_none() {
                self.graph_algorithm = Some(runtime.algorithm.clone());
            }
        }
    }

    pub(super) fn finish_ok(mut self, started: Instant) -> Self {
        self.outcome = "succeeded";
        self.end_epoch_ms = epoch_ms();
        self.wall_ms = duration_ms(started.elapsed());
        self
    }

    pub(super) fn finish_failed(mut self, started: Instant, error: String) -> Self {
        self.outcome = "failed";
        self.end_epoch_ms = epoch_ms();
        self.wall_ms = duration_ms(started.elapsed());
        self.error = Some(error);
        self
    }

    pub(super) fn to_tsv(&self) -> String {
        let fields = [
            self.run_id.clone(),
            self.label.clone(),
            self.task_index.to_string(),
            self.iteration.to_string(),
            self.start_epoch_ms.to_string(),
            self.end_epoch_ms.to_string(),
            format!("{:.3}", self.wall_ms),
            self.outcome.to_string(),
            fmt_opt(self.admission_sequence),
            self.admission_wait_ms.map_or_else(|| "none".to_string(), |value| format!("{value:.3}")),
            fmt_opt(self.device_ordinal),
            fmt_opt(self.query_grant_bytes),
            self.reservation_requested_bytes.to_string(),
            self.reservation_granted_bytes.to_string(),
            self.reservation_live_peak_bytes.to_string(),
            self.reservation_denied_count.to_string(),
            fmt_opt(self.allocator_high_water_bytes),
            fmt_opt(self.max_task_observed_peak_bytes),
            self.max_task_reservation_bytes.to_string(),
            fmt_opt_str(&self.graph_algorithm),
            fmt_opt_str(&self.graph_class),
            fmt_opt_str(&self.graph_outcome),
            fmt_opt(self.graph_raw_input_edges),
            fmt_opt(self.graph_input_edges),
            fmt_opt(self.graph_vertex_estimate),
            fmt_opt_str(&self.graph_vertex_estimate_source),
            fmt_opt(self.graph_vertex_width_bytes),
            fmt_opt(self.graph_weight_width_bytes),
            self.graph_directed.map_or_else(|| "none".to_string(), |value| value.to_string()),
            self.graph_renumber.map_or_else(|| "none".to_string(), |value| value.to_string()),
            fmt_opt_str(&self.graph_construction_policy),
            self.graph_edge_normalize_inserted.map_or_else(|| "none".to_string(), |value| value.to_string()),
            fmt_opt_str(&self.graph_calibration_source),
            fmt_opt(self.graph_seed_count),
            fmt_opt_str(&self.graph_seed_mode),
            fmt_opt(self.graph_model_requested_bytes),
            fmt_opt(self.graph_requested_bytes),
            fmt_opt(self.graph_budget_bytes),
            fmt_opt_str(&self.graph_model_error_ratio),
            fmt_opt(self.graph_statistics_exact_vertices),
            fmt_opt(self.graph_statistics_degree_max),
            fmt_opt(self.graph_statistics_degree_p99),
            fmt_opt(self.graph_statistics_sum_min_degree_oriented),
            fmt_opt(self.graph_statistics_oriented_edge_count),
            fmt_opt(self.graph_statistics_peak_chunk_min_degree_sum),
            fmt_opt(self.graph_statistics_sum_out_degree),
            fmt_opt(self.graph_peak_bytes),
            fmt_opt_str(&self.graph_reason),
            self.graph_elapsed_ms.map_or_else(|| "none".to_string(), |value| format!("{value:.3}")),
            fmt_opt(self.graph_output_device_bytes),
            fmt_opt(self.output_rows),
            self.native_lowered.to_string(),
            self.error.as_deref().map(tsv_field).unwrap_or_else(|| "none".to_string()),
        ];
        debug_assert_eq!(fields.len(), QUERY_COLUMNS.len());
        fields.join("\t")
    }
}

pub(super) fn device_peaks(
    device_ordinals: &[usize],
    device_samples: &[DeviceSampleRow],
) -> BTreeMap<String, serde_json::Value> {
    device_ordinals
        .iter()
        .map(|device_ordinal| {
            let samples =
                device_samples.iter().filter(|sample| sample.device_ordinal == *device_ordinal).collect::<Vec<_>>();
            let peaks = serde_json::json!({
                "pool_used_high_bytes": samples.iter().map(|sample| sample.facts.pool_used_high_bytes).max(),
                "pool_reserved_high_bytes": samples
                    .iter()
                    .map(|sample| sample.facts.pool_reserved_high_bytes)
                    .max(),
                "governor_outstanding_bytes_max": samples
                    .iter()
                    .map(|sample| sample.facts.governor_outstanding_bytes)
                    .max(),
                "gpu_execution_active_query_jobs_max": samples
                    .iter()
                    .map(|sample| sample.facts.gpu_execution_active_query_jobs)
                    .max(),
                "allocation_failure_count_last": samples.last().map(|sample| sample.facts.allocation_failure_count),
                "trim_retry_count_last": samples.last().map(|sample| sample.facts.trim_retry_count),
            });
            (device_ordinal.to_string(), peaks)
        })
        .collect()
}

#[expect(clippy::too_many_arguments, reason = "benchmark harness parameters are independent measurement axes")]
pub(super) fn write_outputs(
    config: &HarnessConfig,
    warmup_rows: &[QueryRow],
    query_rows: &[QueryRow],
    samples: &[SampleRow],
    device_samples: &[DeviceSampleRow],
    makespan_ms: f64,
    effective_active_max: usize,
    bfs_singleflight: datafusion_nexus::native_exec::GraphBfsSingleflightSnapshot,
) -> Result<(), Box<dyn Error>> {
    let queries_path = config.out_dir.join("queries.tsv");
    write_query_rows(&queries_path, query_rows)?;
    if !warmup_rows.is_empty() {
        write_query_rows(&config.out_dir.join("warmup_queries.tsv"), warmup_rows)?;
    }

    let samples_path = config.out_dir.join("samples.tsv");
    let mut samples_file = fs::File::create(&samples_path)?;
    writeln!(samples_file, "{}", SAMPLE_COLUMNS.join("\t"))?;
    for row in samples {
        writeln!(samples_file, "{}", row.to_tsv())?;
    }

    let device_samples_path = config.out_dir.join("device_samples.tsv");
    let mut device_samples_file = fs::File::create(&device_samples_path)?;
    writeln!(device_samples_file, "{}", DEVICE_SAMPLE_COLUMNS.join("\t"))?;
    for row in device_samples {
        writeln!(device_samples_file, "{}", row.to_tsv())?;
    }

    let mut warmup_fingerprints = BTreeMap::<String, BTreeSet<String>>::new();
    for row in warmup_rows {
        if let Some(fingerprint) = &row.output_fingerprint {
            warmup_fingerprints.entry(row.label.clone()).or_default().insert(fingerprint.clone());
        }
    }

    type LabelStats = (usize, usize, Vec<f64>, Vec<f64>, BTreeSet<String>);
    let mut label_stats: BTreeMap<&str, LabelStats> = BTreeMap::new();
    for row in query_rows {
        let entry = label_stats.entry(row.label.as_str()).or_insert((0, 0, Vec::new(), Vec::new(), BTreeSet::new()));
        if row.outcome == "succeeded" {
            entry.0 += 1;
            entry.2.push(row.wall_ms);
            if let Some(wait) = row.admission_wait_ms {
                entry.3.push(wait);
            }
            if let Some(fingerprint) = &row.output_fingerprint {
                entry.4.insert(fingerprint.clone());
            }
        } else {
            entry.1 += 1;
        }
    }
    let peaks_by_device = device_peaks(&config.device_ordinals, device_samples);
    let summary = serde_json::json!({
        "run_id": config.run_id,
        "makespan_ms": makespan_ms,
        "config": {
            "workload": config.workload.iter().map(|(label, count)| format!("{label}:{count}")).collect::<Vec<_>>(),
            "warmup_iterations": config.warmup_iterations,
            "iterations": config.iterations,
            "cuda_profiler_capture": config.cuda_profiler_capture,
            "stagger_ms": config.stagger_ms,
            "max_active_attempts_per_device": config.max_active_attempts_per_device,
            "device_ordinals": config.device_ordinals,
            "attempt_floor_bytes": config.attempt_floor_bytes,
            "max_queue_wait_ms": config.max_queue_wait_ms,
            "budget_fraction": config.budget_fraction,
            "device_budget_bytes": config.device_budget_bytes,
            "raw_input_edges": config.raw_input_edges,
            "query_timeout_ms": config.query_timeout_ms,
            "edges_parquet": config.edges_parquet,
        },
        "labels": label_stats.iter().map(|(label, (ok, failed, walls, waits, fingerprints))| {
            let warmup = warmup_fingerprints.get(*label).cloned().unwrap_or_default();
            let outputs_match_warmup = !warmup.is_empty() && *fingerprints == warmup;
            serde_json::json!({
                "label": label,
                "succeeded": ok,
                "failed": failed,
                "wall_ms_mean": mean(walls),
                "wall_ms_p50": percentile(walls, 0.50),
                "wall_ms_p95": percentile(walls, 0.95),
                "admission_wait_ms_mean": mean(waits),
                "admission_wait_ms_max": waits.iter().cloned().fold(f64::NAN, f64::max),
                "output_fingerprints": fingerprints,
                "output_fingerprint_count": fingerprints.len(),
                "warmup_output_fingerprints": warmup,
                "outputs_match_warmup": outputs_match_warmup,
            })
        }).collect::<Vec<_>>(),
        "peaks": {
            "gpu_execution_active_max": effective_active_max,
            "pool_used_high_bytes": device_samples.iter().map(|s| s.facts.pool_used_high_bytes).max(),
            "pool_reserved_high_bytes": device_samples.iter().map(|s| s.facts.pool_reserved_high_bytes).max(),
            "governor_outstanding_bytes_max": device_samples
                .iter()
                .map(|s| s.facts.governor_outstanding_bytes)
                .max(),
            "active_queries_max": samples.iter().map(|s| s.active_queries).max(),
            "queued_queries_max": samples.iter().map(|s| s.queued_queries).max(),
            "allocation_failure_count_last": samples.last().map(|s| s.allocation_failure_count),
        },
        "peaks_by_device": peaks_by_device,
        "bfs_singleflight": {
            "eligible_requests": bfs_singleflight.eligible_requests,
            "physical_traversals": bfs_singleflight.physical_traversals,
            "follower_copies": bfs_singleflight.follower_copies,
            "groups_formed": bfs_singleflight.groups_formed,
            "overflow_groups": bfs_singleflight.overflow_groups,
            "leader_failures": bfs_singleflight.leader_failures,
            "cancelled_waiters": bfs_singleflight.cancelled_waiters,
            "peak_width": bfs_singleflight.peak_width,
        },
    });
    fs::write(config.out_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

pub(super) fn write_query_rows(path: &PathBuf, rows: &[QueryRow]) -> Result<(), Box<dyn Error>> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "{}", QUERY_COLUMNS.join("\t"))?;
    for row in rows {
        writeln!(file, "{}", row.to_tsv())?;
    }
    Ok(())
}

pub(super) fn query_rows_makespan_ms(rows: &[QueryRow]) -> Option<f64> {
    let start = rows.iter().map(|row| row.start_epoch_ms).min()?;
    let end = rows.iter().map(|row| row.end_epoch_ms).max()?;
    Some(end.saturating_sub(start) as f64)
}

pub(super) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { f64::NAN } else { values.iter().sum::<f64>() / values.len() as f64 }
}

pub(super) fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}
