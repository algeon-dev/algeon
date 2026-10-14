//! `GetAdmissionStats` sampling for `flight-sql-stress`.
//!
//! The action returns `# key=value` header lines with the configured limits
//! and a `[backend_device_admission]` TSV section with one row per device
//! (`algeon-datafusion/src/server/admission.rs`). A sample keeps only the
//! per-device `active_attempts`, `live_grant_bytes` and `health` columns; the
//! limits are kept once, from the first decoded sample.

use std::{collections::BTreeMap, time::Duration};

#[cfg(not(test))]
use arrow_flight::Action;
#[cfg(not(test))]
use futures::TryStreamExt;
use tokio::{
    sync::oneshot,
    time::{Instant, MissedTickBehavior, interval, sleep_until, timeout, timeout_at},
};

#[cfg(not(test))]
use super::FlightClient;
use super::{is_identifier, tsv_field};
use crate::diagnostics_tsv::{TsvRow, tsv_section};

#[cfg(not(test))]
const GET_ADMISSION_STATS_ACTION: &str = "GetAdmissionStats";

pub(super) const ADMISSION_COLUMNS: &[&str] =
    &["sample", "offset_ns", "device_ordinal", "active_attempts", "live_grant_bytes", "health"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DeviceAdmission {
    pub(super) device_ordinal: u64,
    pub(super) active_attempts: u64,
    pub(super) live_grant_bytes: u64,
    pub(super) health: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DeviceLimits {
    pub(super) device_ordinal: u64,
    pub(super) max_active_attempts: u64,
    pub(super) default_query_limit_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AdmissionLimits {
    pub(super) max_queued_attempts: u64,
    pub(super) max_queue_wait_ms: u64,
    pub(super) devices: Vec<DeviceLimits>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AdmissionPayload {
    pub(super) limits: AdmissionLimits,
    pub(super) devices: Vec<DeviceAdmission>,
}

/// Decodes one `GetAdmissionStats` payload. Health is the server's enum name;
/// any value that is not a bounded identifier fails the sample.
pub(super) fn parse_admission_payload(payload: &str) -> Result<AdmissionPayload, String> {
    let headers =
        payload.lines().filter_map(|line| line.strip_prefix("# ")?.split_once('=')).collect::<BTreeMap<&str, &str>>();
    let header_u64 = |key: &str| -> Result<u64, String> {
        headers
            .get(key)
            .ok_or_else(|| format!("admission payload is missing `# {key}`"))?
            .parse::<u64>()
            .map_err(|_| format!("admission payload `# {key}` must be a non-negative integer"))
    };
    let ordinals = headers
        .get("device_ordinals")
        .ok_or_else(|| "admission payload is missing `# device_ordinals`".to_string())?
        .split(',')
        .filter(|ordinal| !ordinal.is_empty())
        .map(|ordinal| {
            ordinal.parse::<u64>().map_err(|_| "admission payload device_ordinals must be integers".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let devices = ordinals
        .iter()
        .map(|ordinal| {
            Ok(DeviceLimits {
                device_ordinal: *ordinal,
                max_active_attempts: header_u64(&format!("device_{ordinal}_max_active_attempts"))?,
                default_query_limit_bytes: header_u64(&format!("device_{ordinal}_default_query_limit_bytes"))?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let limits = AdmissionLimits {
        max_queued_attempts: header_u64("max_queued_attempts")?,
        max_queue_wait_ms: header_u64("max_queue_wait_ms")?,
        devices,
    };
    let devices = tsv_section(payload, "backend_device_admission")?
        .iter()
        .map(|row| {
            let health = row_field(row, "health")?;
            if !is_identifier(health) {
                return Err("admission payload health must be an identifier".to_string());
            }
            Ok(DeviceAdmission {
                device_ordinal: row_u64(row, "device_ordinal")?,
                active_attempts: row_u64(row, "active_attempts")?,
                live_grant_bytes: row_u64(row, "live_grant_bytes")?,
                health: health.to_string(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(AdmissionPayload { limits, devices })
}

fn row_field<'a>(row: &'a TsvRow, column: &str) -> Result<&'a str, String> {
    row.get(column).map(String::as_str).ok_or_else(|| format!("admission payload row is missing `{column}`"))
}

fn row_u64(row: &TsvRow, column: &str) -> Result<u64, String> {
    row_field(row, column)?
        .parse::<u64>()
        .map_err(|_| format!("admission payload `{column}` must be a non-negative integer"))
}

/// Every decoded sample of one run. After the last attempt ended, the sampler
/// keeps polling until every device is idle or its timeout passes.
#[derive(Debug, Clone)]
pub(super) struct AdmissionLog {
    pub(super) interval_ms: u64,
    pub(super) rows: Vec<(u64, u64, DeviceAdmission)>,
    pub(super) samples: u64,
    pub(super) failed_samples: u64,
    pub(super) limits: Option<AdmissionLimits>,
    /// Devices of the last decoded after-run poll.
    pub(super) after_run: Option<Vec<DeviceAdmission>>,
    /// Whether an after-run poll found every device idle; none before the
    /// first after-run poll.
    pub(super) after_run_idle: Option<bool>,
    /// Time from the last attempt's end to the first idle poll.
    pub(super) time_to_idle_ns: Option<u64>,
}

impl AdmissionLog {
    pub(super) const fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms,
            rows: Vec::new(),
            samples: 0,
            failed_samples: 0,
            limits: None,
            after_run: None,
            after_run_idle: None,
            time_to_idle_ns: None,
        }
    }

    /// Records one poll taken `since_run_end` after the last attempt ended and
    /// returns whether every device has no active attempts and no live grant
    /// bytes.
    pub(super) fn observe_after_run(
        &mut self,
        offset_ns: u64,
        since_run_end: Duration,
        payload: Result<String, ()>,
    ) -> bool {
        let devices = self.observe(offset_ns, payload);
        let idle = devices.as_ref().is_some_and(|devices| {
            devices.iter().all(|device| device.active_attempts == 0 && device.live_grant_bytes == 0)
        });
        if devices.is_some() {
            self.after_run = devices;
        }
        self.after_run_idle = Some(idle);
        if idle {
            self.time_to_idle_ns = Some(since_run_end.as_nanos() as u64);
        }
        idle
    }

    /// Records one sample taken at `offset_ns` from run start; returns the
    /// decoded devices when the payload was readable.
    pub(super) fn observe(&mut self, offset_ns: u64, payload: Result<String, ()>) -> Option<Vec<DeviceAdmission>> {
        let Ok(decoded) = payload.and_then(|payload| parse_admission_payload(&payload).map_err(|_| ())) else {
            self.failed_samples += 1;
            return None;
        };
        let sample = self.samples;
        self.samples += 1;
        self.limits.get_or_insert(decoded.limits);
        self.rows.extend(decoded.devices.iter().cloned().map(|device| (sample, offset_ns, device)));
        Some(decoded.devices)
    }

    pub(super) fn to_tsv(&self) -> String {
        let mut output = ADMISSION_COLUMNS.join("\t");
        output.push('\n');
        for (sample, offset_ns, device) in &self.rows {
            output.push_str(&format!(
                "{sample}\t{offset_ns}\t{}\t{}\t{}\t{}\n",
                device.device_ordinal,
                device.active_attempts,
                device.live_grant_bytes,
                tsv_field(&device.health),
            ));
        }
        output
    }

    pub(super) fn summary(&self) -> serde_json::Value {
        let mut summary = serde_json::json!({
            "interval_ms": self.interval_ms,
            "samples": self.samples,
            "failed_samples": self.failed_samples,
        });
        if let Some(limits) = &self.limits {
            summary["limits"] = serde_json::json!({
                "max_queued_attempts": limits.max_queued_attempts,
                "max_queue_wait_ms": limits.max_queue_wait_ms,
                "devices": limits.devices.iter().map(|device| serde_json::json!({
                    "device_ordinal": device.device_ordinal,
                    "max_active_attempts": device.max_active_attempts,
                    "default_query_limit_bytes": device.default_query_limit_bytes,
                })).collect::<Vec<_>>(),
            });
        }
        if let Some(devices) = &self.after_run {
            summary["after_run"] = serde_json::Value::Array(devices.iter().map(device_json).collect());
        }
        if let Some(idle) = self.after_run_idle {
            summary["after_run_idle"] = idle.into();
        }
        if let Some(time_to_idle_ns) = self.time_to_idle_ns {
            summary["time_to_idle_ns"] = time_to_idle_ns.into();
        }
        summary
    }
}

fn device_json(device: &DeviceAdmission) -> serde_json::Value {
    serde_json::json!({
        "device_ordinal": device.device_ordinal,
        "active_attempts": device.active_attempts,
        "live_grant_bytes": device.live_grant_bytes,
        "health": device.health,
    })
}

/// Polls `GetAdmissionStats` on a dedicated connection; see
/// [`sample_admission_from`].
#[cfg(not(test))]
pub(super) async fn sample_admission(
    client: FlightClient,
    started: Instant,
    period: Duration,
    idle_timeout: Duration,
    stop: oneshot::Receiver<()>,
) -> AdmissionLog {
    sample_admission_from(|| fetch_admission_payload(client.clone()), started, period, idle_timeout, stop).await
}

/// Calls `fetch` every `period` from run start until `stop` fires, then every
/// `period` until every device is idle or `idle_timeout` has passed since
/// `stop`. `stop` abandons an in-flight call without recording it. A call
/// unanswered after `idle_timeout`, or after the time left of the after-run
/// budget, is one failed sample, so a server that never answers still lets the
/// sampler return within `idle_timeout` of `stop`.
pub(super) async fn sample_admission_from<F, Fut>(
    mut fetch: F,
    started: Instant,
    period: Duration,
    idle_timeout: Duration,
    mut stop: oneshot::Receiver<()>,
) -> AdmissionLog
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<String, ()>>,
{
    let mut log = AdmissionLog::new(period.as_millis() as u64);
    let mut ticker = interval(period);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        let sample = async {
            ticker.tick().await;
            // A period cap would turn a slow but answering server into failed
            // samples; the after-run budget bounds only a call that hangs.
            timeout(idle_timeout, fetch()).await.unwrap_or(Err(()))
        };
        tokio::select! {
            _ = &mut stop => break,
            payload = sample => {
                log.observe(started.elapsed().as_nanos() as u64, payload);
            },
        }
    }
    let run_end = Instant::now();
    let idle_deadline = run_end + idle_timeout;
    loop {
        let payload = timeout_at(idle_deadline, fetch()).await.unwrap_or(Err(()));
        if log.observe_after_run(started.elapsed().as_nanos() as u64, run_end.elapsed(), payload) {
            return log;
        }
        let next_poll = Instant::now() + period;
        if next_poll >= idle_deadline {
            return log;
        }
        sleep_until(next_poll).await;
    }
}

/// Any action, transport or UTF-8 failure is one failed sample; its detail is
/// not retained.
#[cfg(not(test))]
async fn fetch_admission_payload(mut client: FlightClient) -> Result<String, ()> {
    let action = Action { r#type: GET_ADMISSION_STATS_ACTION.to_string(), body: Vec::new().into() };
    let mut stream = client.do_action(action).await.map_err(|_| ())?;
    let frame = stream.try_next().await.map_err(|_| ())?.ok_or(())?;
    String::from_utf8(frame.body.to_vec()).map_err(|_| ())
}
