//! Capacity-commit disposition for the report-before-reuse release commit
//! (P2.7).
//!
//! At cleanup completion the release samples the attempt domain, decides the
//! disposition, and commits health BEFORE the observation seal publishes the
//! report; capacity is exposed only afterwards and only for healthy reuse.
//!
//! Rules (normative): OOM or grant exhaustion alone never quarantines a
//! device; a persistent CUDA fault (device health evidence) closes new
//! admission on that device before report publication; when fault isolation
//! is unknown the whole backend is marked unhealthy.

use crate::runtime::attempt_domain::AttemptAllocationDomain;

/// How the release commit disposes of the attempt's device capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CapacityCommitDisposition {
    /// Cleanup restored every invariant; slot and grant return for reuse
    /// after the report is published.
    HealthyReuse,
    /// Device-local persistent fault: the device ledger closes before the
    /// report is published. The attempt's bytes return but the closed ledger
    /// admits nothing new.
    DeviceQuarantine { reason: String },
    /// Fault isolation is unknown: new GPU submissions are rejected for the
    /// whole backend while in-flight attempts finish.
    BackendUnhealthy { reason: String },
}

/// Cleanup-path evidence available at the release commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CleanupEvidence {
    /// The stream/domain cleanup completion ran on the device cleanup worker.
    Completed,
    /// The device's stream failed to synchronize during cleanup: device-local
    /// fault evidence. Failed cleanup never returns capacity.
    StreamSynchronizationFailed,
    /// A queued cleanup callback was dropped before it ran (cancellation,
    /// panic, or runtime teardown); isolation of any fault is unknown.
    CallbackAbandoned,
}

impl From<crate::runtime::service_stream::StreamCleanupOutcome> for CleanupEvidence {
    fn from(outcome: crate::runtime::service_stream::StreamCleanupOutcome) -> Self {
        match outcome {
            crate::runtime::service_stream::StreamCleanupOutcome::Completed => Self::Completed,
            crate::runtime::service_stream::StreamCleanupOutcome::StreamSynchronizationFailed => {
                Self::StreamSynchronizationFailed
            },
            crate::runtime::service_stream::StreamCleanupOutcome::CallbackAbandoned => Self::CallbackAbandoned,
        }
    }
}

/// Samples the attempt domain and decides the capacity disposition.
///
/// Returns the disposition and, when the sample succeeded, the domain usage
/// evidence (current/peak) for the health commit reason.
pub(crate) fn decide_capacity_disposition(
    device_ordinal: usize,
    domain: Option<&AttemptAllocationDomain>,
    cleanup_evidence: CleanupEvidence,
) -> (CapacityCommitDisposition, Option<cudf::QueryMemoryDomainUsage>) {
    decide_capacity_disposition_with_terminal_usage(device_ordinal, domain, cleanup_evidence, None)
}

#[cfg(test)]
pub(crate) fn decide_capacity_disposition_with_terminal_usage_for_test(
    device_ordinal: usize,
    domain: Option<&AttemptAllocationDomain>,
    cleanup_evidence: CleanupEvidence,
    terminal_usage: cudf::QueryMemoryDomainUsage,
) -> (CapacityCommitDisposition, Option<cudf::QueryMemoryDomainUsage>) {
    decide_capacity_disposition_with_terminal_usage(device_ordinal, domain, cleanup_evidence, Some(terminal_usage))
}

fn decide_capacity_disposition_with_terminal_usage(
    device_ordinal: usize,
    domain: Option<&AttemptAllocationDomain>,
    cleanup_evidence: CleanupEvidence,
    terminal_usage: Option<cudf::QueryMemoryDomainUsage>,
) -> (CapacityCommitDisposition, Option<cudf::QueryMemoryDomainUsage>) {
    // A ledger-only admission owns no device state: there is nothing to
    // sample, quarantine, or mark unhealthy, and the commit must stay free
    // of CUDA FFI so ledger-only release latency does not depend on driver
    // contention.
    if domain.is_none() {
        return (CapacityCommitDisposition::HealthyReuse, None);
    }
    // Callback abandonment proves neither cleanup nor fault isolation. It
    // must dominate any concurrent device-health observation: quarantining
    // only one already-unhealthy device would otherwise leave the backend
    // accepting work after an unknown-isolation cleanup loss.
    if cleanup_evidence == CleanupEvidence::CallbackAbandoned {
        return (
            CapacityCommitDisposition::BackendUnhealthy {
                reason: "cleanup callback abandoned before completion; fault isolation unknown".to_string(),
            },
            None,
        );
    }
    let Ok(device) = cudf::DeviceId::new(device_ordinal) else {
        return (
            CapacityCommitDisposition::BackendUnhealthy {
                reason: format!("invalid device ordinal {device_ordinal} at release commit"),
            },
            None,
        );
    };
    let health = cudf::runtime::device_health_snapshot(device);
    if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
        return (
            CapacityCommitDisposition::DeviceQuarantine {
                reason: format!(
                    "persistent CUDA fault on device {device_ordinal} (fault_count={}, first_fault={:?})",
                    health.fault_count(),
                    health.first_fault()
                ),
            },
            None,
        );
    }
    if cleanup_evidence == CleanupEvidence::StreamSynchronizationFailed {
        return (
            CapacityCommitDisposition::DeviceQuarantine {
                reason: format!("stream synchronization failed during cleanup on device {device_ordinal}"),
            },
            None,
        );
    }
    let usage = match terminal_usage.map_or_else(|| domain.expect("domain presence checked above").usage(), Ok) {
        Ok(usage) => usage,
        Err(error) => {
            // The domain exists but its final usage cannot be sampled. If the
            // sample faulted the device, native health reflects it above; an
            // otherwise healthy device with an unsamplable domain has unknown
            // isolation.
            let health = cudf::runtime::device_health_snapshot(device);
            let disposition = if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
                CapacityCommitDisposition::DeviceQuarantine {
                    reason: format!(
                        "persistent CUDA fault on device {device_ordinal} (fault_count={}, first_fault={:?})",
                        health.fault_count(),
                        health.first_fault()
                    ),
                }
            } else {
                CapacityCommitDisposition::BackendUnhealthy {
                    reason: format!("attempt domain usage sample failed after cleanup: {error}"),
                }
            };
            return (disposition, None);
        },
    };
    if usage.active_members != 0
        || usage.current_allocations != 0
        || usage.current_bytes != 0
        || usage.current_aligned_bytes != 0
    {
        return (
            CapacityCommitDisposition::DeviceQuarantine {
                reason: format!(
                    "residual attempt domain usage after terminal cleanup on device {device_ordinal} \
                     (domain_id={}, active_members={}, current_allocations={}, current_bytes={}, \
                     current_aligned_bytes={})",
                    usage.domain_id,
                    usage.active_members,
                    usage.current_allocations,
                    usage.current_bytes,
                    usage.current_aligned_bytes
                ),
            },
            Some(usage),
        );
    }
    (CapacityCommitDisposition::HealthyReuse, Some(usage))
}
