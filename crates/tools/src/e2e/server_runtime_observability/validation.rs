use std::time::{Duration, Instant};

use tokio::time;
use tonic::transport::Channel;

use super::{E2eResult, flight::get_admission_stats};
use crate::diagnostics_tsv::{TsvRow, tsv_section};

const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

pub(super) fn tsv_section_single_row(payload: &str, name: &str) -> E2eResult<TsvRow> {
    let rows = tsv_section(payload, name)?;
    if rows.len() != 1 {
        return Err(format!("expected one [{name}] row, got {}", rows.len()).into());
    }
    Ok(rows.into_iter().next().expect("one row"))
}

pub(super) fn parse_u64_field(row: &TsvRow, name: &str, context: &str) -> E2eResult<u64> {
    let raw = row
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("missing TSV field {name:?} for {context}; row={row:?}"))?;
    raw.parse::<u64>().map_err(|error| format!("invalid {name}={raw:?} for {context}: {error}").into())
}

pub(super) async fn wait_for_idle(channel: &Channel) -> E2eResult<String> {
    let deadline = Instant::now() + IDLE_TIMEOUT;
    loop {
        let payload = get_admission_stats(channel).await?;
        if diagnostics_are_idle(&payload)? {
            return Ok(payload);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "server did not return to an idle resource state within {} seconds:\n{payload}",
                IDLE_TIMEOUT.as_secs()
            )
            .into());
        }
        time::sleep(Duration::from_millis(50)).await;
    }
}

/// The payload carries per-device ledger counters and no queue depth; queued attempts are
/// not part of this check.
pub(super) fn diagnostics_are_idle(payload: &str) -> E2eResult<bool> {
    for row in tsv_section(payload, "backend_device_admission")? {
        for field_name in ["active_attempts", "live_grant_bytes"] {
            if parse_u64_field(&row, field_name, "backend device admission")? != 0 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
