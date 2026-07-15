use std::{collections::BTreeSet, fs, path::Path};

use serde_json::json;

use super::{
    E2eResult,
    churn::ChurnReport,
    flight::QueryEvidence,
    probe::{PROBE_TIMEOUT, ProbeReport, SATURATION_FLOOR},
    server::{MAX_CONCURRENT_QUERIES, MAX_QUERIES_PER_DEVICE, Params},
    validation::{parse_u64_field, parse_usize_field, tsv_section, tsv_section_single_row},
};

pub(super) fn write_run_metadata(params: &Params) -> E2eResult<()> {
    let git_sha = command_output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let git_dirty = command_output("git", &["status", "--porcelain"]).is_some_and(|value| !value.is_empty());
    let server_sha256 = sha256_file(&params.server_binary).unwrap_or_else(|| "unknown".to_string());
    let driver_sha256 =
        std::env::current_exe().ok().as_deref().and_then(sha256_file).unwrap_or_else(|| "unknown".to_string());
    let metadata = format!(
        "git_sha={git_sha}\ngit_dirty={git_dirty}\nserver_binary={}\nserver_features=server,iceberg,nvml\n\
         server_binary_sha256={server_sha256}\ndriver_binary_sha256={driver_sha256}\n\
         tokio_worker_threads=1\ndevice_ordinals={}\nmax_concurrent_queries={MAX_CONCURRENT_QUERIES}\n\
         max_queries_per_device={MAX_QUERIES_PER_DEVICE}\nrest_uri={}\nwarehouse={}\nnamespace={}\n",
        params.server_binary.display(),
        params.device_csv(),
        params.rest_uri,
        params.warehouse,
        params.namespace,
    );
    fs::write(params.artifact_dir.join("metadata.txt"), metadata)?;
    let mut harness_source = String::new();
    for (path, source) in [
        ("server_iceberg_rest_multigpu_e2e.rs", include_str!("../server_iceberg_rest_multigpu_e2e.rs")),
        ("server.rs", include_str!("server.rs")),
        ("flight.rs", include_str!("flight.rs")),
        ("probe.rs", include_str!("probe.rs")),
        ("churn.rs", include_str!("churn.rs")),
        ("validation.rs", include_str!("validation.rs")),
        ("artifacts.rs", include_str!("artifacts.rs")),
        ("connection_retry.rs", include_str!("../../connection_retry.rs")),
    ] {
        harness_source.push_str(&format!("// ===== {path} =====\n"));
        harness_source.push_str(source);
        if !source.ends_with('\n') {
            harness_source.push('\n');
        }
    }
    fs::write(params.artifact_dir.join("harness-source.rs"), harness_source)?;
    if let Some(diff) = command_output(
        "git",
        &[
            "diff",
            "--binary",
            "--",
            "Cargo.lock",
            "crates/datafusion-nexus-tools/Cargo.toml",
            "crates/datafusion-nexus-tools/src/bin/datafusion_nexus_tools.rs",
            "crates/datafusion-nexus-tools/src/connection_retry.rs",
            "crates/datafusion-nexus-tools/src/e2e/server_iceberg_rest_multigpu_e2e.rs",
            "crates/datafusion-nexus-tools/src/e2e/server_iceberg_rest_multigpu_e2e",
            "fixture/scripts/e2e.sh",
            "docs/issue/2026-07-12-hardware-limit-roadmap.md",
        ],
    ) {
        fs::write(params.artifact_dir.join("source.diff"), diff)?;
    }
    Ok(())
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn sha256_file(path: &Path) -> Option<String> {
    let path = path.to_str()?;
    command_output("sha256sum", &[path]).and_then(|line| line.split_whitespace().next().map(str::to_string))
}

pub(super) fn write_probe_artifacts(params: &Params, probes: &ProbeReport) -> E2eResult<()> {
    let mut output = String::from(
        "sequence\tactive_queries\tqueued_queries\tactive_gpu_jobs\tdiagnostics_ms\tget_flight_info_ms\t\
         ticket_retire_ms\tsaturated\n",
    );
    for row in &probes.rows {
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.sequence,
            row.active_queries,
            row.queued_queries,
            row.active_gpu_jobs,
            row.diagnostics_ms,
            row.get_flight_info_ms,
            row.ticket_retire_ms,
            row.saturated
        ));
    }
    fs::write(params.artifact_dir.join("probes.tsv"), output)?;
    Ok(())
}

pub(super) fn write_query_artifacts(
    params: &Params,
    successful: &[QueryEvidence],
    churn: &ChurnReport,
) -> E2eResult<()> {
    let mut output = String::from(
        "phase\tworkload\tquery_id\tdevice_ordinal\toutput_rows\tsource_cache_hits\tsource_cache_misses\t\
         source_cache_inserts\td2h_count\td2h_bytes\tparity_mode\tcpu_fingerprint\tnative_fingerprint\n",
    );
    for row in successful {
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            tsv_escape(&row.phase),
            row.workload,
            tsv_escape(&row.query_id),
            row.device_ordinal,
            row.output_rows,
            row.source_cache_hits,
            row.source_cache_misses,
            row.source_cache_inserts,
            row.d2h_count,
            row.d2h_bytes,
            row.parity_mode,
            row.cpu_fingerprint,
            row.native_fingerprint,
        ));
    }
    fs::write(params.artifact_dir.join("queries.tsv"), output)?;

    let mut payloads = String::new();
    for row in successful {
        append_query_payload(&mut payloads, &row.phase, &row.query_id, &row.stats_payload);
    }
    for (query_id, payload) in &churn.cancelled {
        append_query_payload(&mut payloads, "churn_cancelled", query_id, payload);
    }
    for (query_id, payload) in &churn.dropped {
        append_query_payload(&mut payloads, "churn_dropped", query_id, payload);
    }
    fs::write(params.artifact_dir.join("query_stats.txt"), payloads)?;
    Ok(())
}

fn append_query_payload(output: &mut String, phase: &str, query_id: &str, payload: &str) {
    output.push_str(&format!("===== phase={} query_id={} =====\n", tsv_escape(phase), tsv_escape(query_id)));
    output.push_str(payload);
    if !payload.ends_with('\n') {
        output.push('\n');
    }
}

fn tsv_escape(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}

pub(super) fn write_summary(
    params: &Params,
    successful: &[QueryEvidence],
    churn: &ChurnReport,
    probes: &ProbeReport,
    after: &str,
) -> E2eResult<()> {
    let gpu = tsv_section_single_row(after, "gpu_execution")?;
    let pools = tsv_section(after, "gpu_execution_pools")?;
    let governors = tsv_section(after, "memory_governor_reports")?;
    let telemetry = tsv_section(after, "device_telemetry")?;
    let device_peaks = pools
        .iter()
        .map(|row| {
            Ok(json!({
                "device_ordinal": parse_usize_field(row, "device_ordinal", "summary pool")?,
                "worker_count": parse_u64_field(row, "worker_count", "summary pool")?,
                "peak_active_query_jobs": parse_u64_field(row, "peak_active_query_jobs", "summary pool")?,
            }))
        })
        .collect::<E2eResult<Vec<_>>>()?;
    let capacities = governors
        .iter()
        .map(|row| {
            Ok(json!({
                "device_ordinal": parse_usize_field(row, "device_ordinal", "summary governor")?,
                "capacity_bytes": parse_u64_field(row, "capacity_bytes", "summary governor")?,
            }))
        })
        .collect::<E2eResult<Vec<_>>>()?;
    let device_totals = telemetry
        .iter()
        .map(|row| {
            Ok(json!({
                "device_ordinal": parse_usize_field(row, "device_ordinal", "summary telemetry")?,
                "cuda_total_bytes": parse_u64_field(row, "cuda_total_bytes", "summary telemetry")?,
                "nvml_total_bytes": parse_u64_field(row, "nvml_total_bytes", "summary telemetry")?,
            }))
        })
        .collect::<E2eResult<Vec<_>>>()?;
    let observed_devices = successful.iter().map(|row| row.device_ordinal).collect::<BTreeSet<_>>();
    let cache_by_device = params
        .devices
        .iter()
        .map(|device| {
            let rows = successful.iter().filter(|row| row.device_ordinal == *device);
            let (hits, misses, inserts) = rows.fold((0_u64, 0_u64, 0_u64), |totals, row| {
                (
                    totals.0.saturating_add(row.source_cache_hits),
                    totals.1.saturating_add(row.source_cache_misses),
                    totals.2.saturating_add(row.source_cache_inserts),
                )
            });
            json!({
                "device_ordinal": device,
                "hits": hits,
                "misses": misses,
                "inserts": inserts,
            })
        })
        .collect::<Vec<_>>();
    let summary = json!({
        "status": "passed",
        "topology": {
            "tokio_worker_threads": 1,
            "device_ordinals": params.devices,
            "max_concurrent_queries": MAX_CONCURRENT_QUERIES,
            "max_queries_per_device": MAX_QUERIES_PER_DEVICE,
        },
        "queries": {
            "successful": successful.len(),
            "cancelled": churn.cancelled.len(),
            "dropped": churn.dropped.len(),
            "observed_devices": observed_devices,
            "all_exact_parity": true,
            "all_native": true,
            "all_positive_d2h": true,
        },
        "liveness": {
            "probe_pairs": probes.rows.len(),
            "saturated_probe_pairs": probes.saturated_count,
            "saturation_floor": SATURATION_FLOOR,
            "timeout_ms": PROBE_TIMEOUT.as_millis(),
            "peak_observed_active": probes.peak_score,
            "statement_tickets_issued": probes.tickets_issued,
            "statement_tickets_retired": probes.tickets_retired,
        },
        "gpu_execution": {
            "peak_active_query_jobs": parse_u64_field(&gpu, "peak_active_query_jobs", "summary GPU")?,
            "pools": device_peaks,
        },
        "memory_governors": capacities,
        "decoded_source_cache": {
            "configured_max_retained_bytes_per_device": 8_u64 * 1024 * 1024 * 1024,
            "rest_host_buffer_identity": "query_scoped_misses_allowed",
            "devices": cache_by_device,
        },
        "telemetry": device_totals,
        "allocation_failures": 0,
        "final_state_idle": true,
    });
    fs::write(params.artifact_dir.join("summary.json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(())
}
