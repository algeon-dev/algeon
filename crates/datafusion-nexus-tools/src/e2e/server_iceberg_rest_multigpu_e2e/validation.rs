use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

use tonic::transport::Channel;

use super::{
    E2eResult,
    flight::{QueryEvidence, get_admission_stats},
    server::{MAX_CONCURRENT_QUERIES, MAX_QUERIES_PER_DEVICE},
};

pub(super) type TsvRow = BTreeMap<String, String>;

pub(super) fn tsv_section(payload: &str, name: &str) -> E2eResult<Vec<TsvRow>> {
    let marker = format!("[{name}]");
    let mut lines = payload.lines().skip_while(|line| *line != marker).skip(1);
    let header = lines.next().ok_or_else(|| format!("missing [{name}] section in payload:\n{payload}"))?;
    if header.is_empty() || header.starts_with('[') {
        return Err(format!("missing [{name}] TSV header in payload:\n{payload}").into());
    }
    let columns = header.split('\t').collect::<Vec<_>>();
    let mut rows = Vec::new();
    for line in lines {
        if line.is_empty() || line.starts_with('[') {
            break;
        }
        if line.starts_with('#') {
            continue;
        }
        let values = line.split('\t').collect::<Vec<_>>();
        if values.len() != columns.len() {
            return Err(format!(
                "[{name}] TSV width mismatch: {} columns, {} values in {line:?}",
                columns.len(),
                values.len()
            )
            .into());
        }
        rows.push(
            columns.iter().zip(values).map(|(column, value)| ((*column).to_string(), value.to_string())).collect(),
        );
    }
    Ok(rows)
}

pub(super) fn tsv_section_single_row(payload: &str, name: &str) -> E2eResult<TsvRow> {
    let rows = tsv_section(payload, name)?;
    if rows.len() != 1 {
        return Err(format!("expected one [{name}] row, got {}", rows.len()).into());
    }
    Ok(rows.into_iter().next().expect("one row"))
}

pub(super) fn field<'a>(row: &'a TsvRow, name: &str, context: &str) -> E2eResult<&'a str> {
    row.get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("missing TSV field {name:?} for {context}; row={row:?}").into())
}

pub(super) fn parse_u64_field(row: &TsvRow, name: &str, context: &str) -> E2eResult<u64> {
    let raw = field(row, name, context)?;
    raw.parse::<u64>().map_err(|error| format!("invalid {name}={raw:?} for {context}: {error}").into())
}

pub(super) fn parse_usize_field(row: &TsvRow, name: &str, context: &str) -> E2eResult<usize> {
    let raw = field(row, name, context)?;
    raw.parse::<usize>().map_err(|error| format!("invalid {name}={raw:?} for {context}: {error}").into())
}

pub(super) fn require_value(row: &TsvRow, name: &str, expected: &str, context: &str) -> E2eResult<()> {
    let actual = field(row, name, context)?;
    if actual != expected {
        return Err(format!("{context} expected {name}={expected:?}, got {actual:?}; row={row:?}").into());
    }
    Ok(())
}

fn row_device_set(rows: &[TsvRow], context: &str) -> E2eResult<BTreeSet<usize>> {
    rows.iter().map(|row| parse_usize_field(row, "device_ordinal", context)).collect()
}

fn expected_device_set(devices: &[usize]) -> BTreeSet<usize> {
    devices.iter().copied().collect()
}

pub(super) fn validate_static_topology(payload: &str, devices: &[usize]) -> E2eResult<()> {
    if !payload.contains("# backend=query_service\n")
        || !payload.contains(&format!(
            "# device_ordinals={}\n",
            devices.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
        ))
        || !payload.contains(&format!("# max_concurrent_queries={MAX_CONCURRENT_QUERIES}\n"))
        || !payload.contains(&format!("# max_queries_per_device={MAX_QUERIES_PER_DEVICE}\n"))
        || !payload.contains("# placement_policy=least_loaded\n")
    {
        return Err(format!("admission payload does not describe the requested production topology:\n{payload}").into());
    }
    let expected = expected_device_set(devices);
    for section in [
        "memory_governor_reports",
        "device_memory_failure_reports",
        "gpu_execution_pools",
        "device_admission_slots",
        "device_telemetry",
    ] {
        let rows = tsv_section(payload, section)?;
        if rows.len() != devices.len() || row_device_set(&rows, section)? != expected {
            return Err(format!("[{section}] must contain exactly devices {expected:?}; rows={rows:?}").into());
        }
    }

    let governors = tsv_section(payload, "memory_governor_reports")?;
    let capacities = governors
        .iter()
        .map(|row| parse_u64_field(row, "capacity_bytes", "memory governor"))
        .collect::<E2eResult<BTreeSet<_>>>()?;
    if capacities.len() != 1 {
        return Err(format!("heterogeneous GPU capacities are unsupported by this acceptance: {capacities:?}").into());
    }

    let telemetry = tsv_section(payload, "device_telemetry")?;
    let mut cuda_totals = BTreeSet::new();
    let mut nvml_totals = BTreeSet::new();
    for row in telemetry {
        require_value(&row, "cuda_status", "available", "device telemetry")?;
        require_value(&row, "nvml_status", "available", "device telemetry")?;
        cuda_totals.insert(parse_u64_field(&row, "cuda_total_bytes", "device telemetry")?);
        nvml_totals.insert(parse_u64_field(&row, "nvml_total_bytes", "device telemetry")?);
    }
    if cuda_totals.len() != 1 || nvml_totals.len() != 1 {
        return Err(
            format!("heterogeneous GPU totals are unsupported: CUDA={cuda_totals:?}, NVML={nvml_totals:?}").into()
        );
    }
    Ok(())
}

pub(super) fn validate_cold_and_warm_cache(
    cold: &[QueryEvidence],
    warm: &[QueryEvidence],
    devices: &[usize],
) -> E2eResult<()> {
    for device in devices {
        let cold_rows = cold.iter().filter(|row| row.device_ordinal == *device).collect::<Vec<_>>();
        let warm_rows = warm.iter().filter(|row| row.device_ordinal == *device).collect::<Vec<_>>();
        let cold_misses = cold_rows.iter().map(|row| row.source_cache_misses).sum::<u64>();
        let cold_inserts = cold_rows.iter().map(|row| row.source_cache_inserts).sum::<u64>();
        let warm_hits = warm_rows.iter().map(|row| row.source_cache_hits).sum::<u64>();
        let warm_misses = warm_rows.iter().map(|row| row.source_cache_misses).sum::<u64>();
        if cold_rows.is_empty() || cold_misses == 0 || cold_inserts == 0 {
            return Err(format!(
                "device {device} lacks cold decoded-source-cache evidence: rows={} misses={cold_misses} inserts={cold_inserts}",
                cold_rows.len()
            )
            .into());
        }
        // REST host-buffer staging assigns query-scoped reader locations, so
        // cross-query key stability is not guaranteed on this substrate.
        // Require active cache lookup evidence on the warm wave, but do not
        // mislabel staging-identity misses as a decoded-cache regression.
        if warm_rows.is_empty() || warm_hits.saturating_add(warm_misses) == 0 {
            return Err(format!(
                "device {device} lacks warm Q06 decoded-cache lookup evidence: rows={} hits={warm_hits} misses={warm_misses}",
                warm_rows.len()
            )
            .into());
        }
    }
    Ok(())
}

pub(super) fn validate_observed_devices(successful: &[QueryEvidence], devices: &[usize]) -> E2eResult<()> {
    let observed = successful.iter().map(|row| row.device_ordinal).collect::<BTreeSet<_>>();
    let expected = expected_device_set(devices);
    if observed != expected {
        return Err(
            format!("successful Flight SQL queries observed devices {observed:?}, expected {expected:?}").into()
        );
    }
    Ok(())
}

pub(super) async fn wait_for_idle(channel: &Channel) -> E2eResult<String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let payload = get_admission_stats(channel).await?;
        if diagnostics_are_idle(&payload)? {
            return Ok(payload);
        }
        if Instant::now() >= deadline {
            return Err(format!("server did not return to an idle resource state within 60 seconds:\n{payload}").into());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn diagnostics_are_idle(payload: &str) -> E2eResult<bool> {
    let admission = tsv_section_single_row(payload, "admission")?;
    if parse_u64_field(&admission, "active_queries", "final admission")? != 0
        || parse_u64_field(&admission, "queued_queries", "final admission")? != 0
    {
        return Ok(false);
    }
    if !tsv_section(payload, "memory_governor_grants")?.is_empty() {
        return Ok(false);
    }
    let gpu = tsv_section_single_row(payload, "gpu_execution")?;
    if parse_u64_field(&gpu, "active_query_jobs", "final GPU execution")? != 0 {
        return Ok(false);
    }
    for row in tsv_section(payload, "gpu_execution_pools")? {
        for field_name in ["active_jobs", "queued_jobs", "active_query_jobs"] {
            if parse_u64_field(&row, field_name, "final GPU pool")? != 0 {
                return Ok(false);
            }
        }
    }
    for row in tsv_section(payload, "device_admission_slots")? {
        if parse_u64_field(&row, "process_active_query_slots", "final device slots")? != 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn validate_final_diagnostics(payload: &str, devices: &[usize]) -> E2eResult<()> {
    validate_static_topology(payload, devices)?;
    if !diagnostics_are_idle(payload)? {
        return Err(format!("final diagnostics are not idle:\n{payload}").into());
    }

    let gpu = tsv_section_single_row(payload, "gpu_execution")?;
    let peak = parse_u64_field(&gpu, "peak_active_query_jobs", "global GPU execution")?;
    if peak != MAX_CONCURRENT_QUERIES as u64 {
        return Err(format!("global GPU executor peak must be exactly {MAX_CONCURRENT_QUERIES}, got {peak}").into());
    }
    for row in tsv_section(payload, "gpu_execution_pools")? {
        require_value(&row, "worker_count", &MAX_QUERIES_PER_DEVICE.to_string(), "GPU execution pool")?;
        require_value(&row, "peak_active_query_jobs", &MAX_QUERIES_PER_DEVICE.to_string(), "GPU execution pool")?;
    }
    for row in tsv_section(payload, "device_memory_failure_reports")? {
        for field_name in
            ["allocation_failure_count", "total_failed_allocation_bytes", "largest_failed_allocation_bytes"]
        {
            require_value(&row, field_name, "0", "device allocation failures")?;
        }
    }
    Ok(())
}
