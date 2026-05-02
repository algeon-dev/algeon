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

/// One terminal release decision and its bounded native-health projection.
///
/// The decision owns the only device-health reads for this release path. The
/// finalizer receives the resulting evidence directly instead of reconstructing
/// a fault from a later CUDA status or a report-side sample.
#[derive(Debug)]
pub(crate) struct CapacityCommitDecision {
    pub(crate) disposition: CapacityCommitDisposition,
    pub(crate) usage: Option<cudf::QueryMemoryDomainUsage>,
    pub(crate) native_fault_evidence: Option<crate::observability::AttemptNativeFaultEvidence>,
}

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
    let decision = decide_capacity_commit(device_ordinal, domain, cleanup_evidence);
    (decision.disposition, decision.usage)
}

#[cfg(test)]
pub(crate) fn decide_capacity_disposition_with_terminal_usage_for_test(
    device_ordinal: usize,
    domain: Option<&AttemptAllocationDomain>,
    cleanup_evidence: CleanupEvidence,
    terminal_usage: cudf::QueryMemoryDomainUsage,
) -> (CapacityCommitDisposition, Option<cudf::QueryMemoryDomainUsage>) {
    let decision =
        decide_capacity_commit_with_terminal_usage(device_ordinal, domain, cleanup_evidence, Some(terminal_usage));
    (decision.disposition, decision.usage)
}

/// Produces the actual release decision used by finalization.
pub(crate) fn decide_capacity_commit(
    device_ordinal: usize,
    domain: Option<&AttemptAllocationDomain>,
    cleanup_evidence: CleanupEvidence,
) -> CapacityCommitDecision {
    decide_capacity_commit_with_terminal_usage(device_ordinal, domain, cleanup_evidence, None)
}

/// Test-only equivalent of [`decide_capacity_commit`] with an injected final
/// domain sample. Production always samples the owned domain exactly here.
#[cfg(test)]
pub(crate) fn decide_capacity_commit_with_terminal_usage_for_test(
    device_ordinal: usize,
    domain: Option<&AttemptAllocationDomain>,
    cleanup_evidence: CleanupEvidence,
    terminal_usage: cudf::QueryMemoryDomainUsage,
) -> CapacityCommitDecision {
    decide_capacity_commit_with_terminal_usage(device_ordinal, domain, cleanup_evidence, Some(terminal_usage))
}

fn decide_capacity_commit_with_terminal_usage(
    device_ordinal: usize,
    domain: Option<&AttemptAllocationDomain>,
    cleanup_evidence: CleanupEvidence,
    terminal_usage: Option<cudf::QueryMemoryDomainUsage>,
) -> CapacityCommitDecision {
    // A ledger-only admission owns no device state: there is nothing to
    // sample, quarantine, or mark unhealthy, and the commit must stay free
    // of CUDA FFI so ledger-only release latency does not depend on driver
    // contention.
    if domain.is_none() {
        return CapacityCommitDecision {
            disposition: CapacityCommitDisposition::HealthyReuse,
            usage: None,
            native_fault_evidence: None,
        };
    }
    // Callback abandonment proves neither cleanup nor fault isolation. It
    // must dominate any concurrent device-health observation: quarantining
    // only one already-unhealthy device would otherwise leave the backend
    // accepting work after an unknown-isolation cleanup loss.
    if cleanup_evidence == CleanupEvidence::CallbackAbandoned {
        return CapacityCommitDecision {
            disposition: CapacityCommitDisposition::BackendUnhealthy {
                reason: "cleanup callback abandoned before completion; fault isolation unknown".to_string(),
            },
            usage: None,
            native_fault_evidence: Some(unknown_isolation_native_evidence(cleanup_evidence)),
        };
    }
    let Ok(device) = cudf::DeviceId::new(device_ordinal) else {
        return CapacityCommitDecision {
            disposition: CapacityCommitDisposition::BackendUnhealthy {
                reason: format!("invalid device ordinal {device_ordinal} at release commit"),
            },
            usage: None,
            native_fault_evidence: Some(unknown_isolation_native_evidence(cleanup_evidence)),
        };
    };
    let health_before_sample = cudf::runtime::device_health_snapshot(device);
    if health_before_sample.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
        return device_quarantine_decision(device_ordinal, cleanup_evidence, health_before_sample, None);
    }
    if cleanup_evidence == CleanupEvidence::StreamSynchronizationFailed {
        return CapacityCommitDecision {
            disposition: CapacityCommitDisposition::DeviceQuarantine {
                reason: format!("stream synchronization failed during cleanup on device {device_ordinal}"),
            },
            usage: None,
            native_fault_evidence: Some(device_local_native_evidence(cleanup_evidence, health_before_sample)),
        };
    }
    let usage = match terminal_usage.map_or_else(|| domain.expect("domain presence checked above").usage(), Ok) {
        Ok(usage) => usage,
        Err(error) => {
            // The domain exists but its final usage cannot be sampled. If the
            // sample faulted the device, native health reflects it above; an
            // otherwise healthy device with an unsamplable domain has unknown
            // isolation.
            let health_after_sample = cudf::runtime::device_health_snapshot(device);
            if health_after_sample.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
                return device_quarantine_decision(device_ordinal, cleanup_evidence, health_after_sample, None);
            } else {
                return CapacityCommitDecision {
                    disposition: CapacityCommitDisposition::BackendUnhealthy {
                        reason: format!("attempt domain usage sample failed after cleanup: {error}"),
                    },
                    usage: None,
                    // The usage sample did not establish that this attempt's
                    // failure belongs to the device it was executing on.
                    // Retaining that device's later health snapshot here
                    // would present a device-local association while the
                    // report correctly says the isolation is unknown.
                    native_fault_evidence: Some(unknown_isolation_native_evidence(cleanup_evidence)),
                };
            }
        },
    };
    // The sample can itself race a cuDF/cuGraph fault on the same device. A
    // second terminal snapshot makes that race a device-local quarantine
    // before capacity or a report becomes visible, rather than reusing the
    // earlier healthy read.
    let health_after_sample = cudf::runtime::device_health_snapshot(device);
    if health_after_sample.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
        return device_quarantine_decision(device_ordinal, cleanup_evidence, health_after_sample, Some(usage));
    }
    if usage.active_members != 0
        || usage.current_allocations != 0
        || usage.current_bytes != 0
        || usage.current_aligned_bytes != 0
    {
        return CapacityCommitDecision {
            disposition: CapacityCommitDisposition::DeviceQuarantine {
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
            usage: Some(usage),
            native_fault_evidence: Some(device_local_native_evidence(cleanup_evidence, health_after_sample)),
        };
    }
    CapacityCommitDecision {
        disposition: CapacityCommitDisposition::HealthyReuse,
        usage: Some(usage),
        native_fault_evidence: Some(device_local_native_evidence(cleanup_evidence, health_after_sample)),
    }
}

fn device_quarantine_decision(
    device_ordinal: usize,
    cleanup_evidence: CleanupEvidence,
    health: cudf::runtime::DeviceHealthSnapshot,
    usage: Option<cudf::QueryMemoryDomainUsage>,
) -> CapacityCommitDecision {
    CapacityCommitDecision {
        disposition: CapacityCommitDisposition::DeviceQuarantine {
            reason: format!(
                "persistent CUDA fault on device {device_ordinal} (fault_count={}, first_fault={:?})",
                health.fault_count(),
                health.first_fault()
            ),
        },
        usage,
        native_fault_evidence: Some(device_local_native_evidence(cleanup_evidence, health)),
    }
}

fn teardown_health(cleanup_evidence: CleanupEvidence) -> crate::observability::AttemptTeardownHealth {
    match cleanup_evidence {
        CleanupEvidence::Completed => crate::observability::AttemptTeardownHealth::Completed,
        CleanupEvidence::StreamSynchronizationFailed => {
            crate::observability::AttemptTeardownHealth::StreamSynchronizationFailed
        },
        CleanupEvidence::CallbackAbandoned => crate::observability::AttemptTeardownHealth::CallbackAbandoned,
    }
}

fn unknown_isolation_native_evidence(
    cleanup_evidence: CleanupEvidence,
) -> crate::observability::AttemptNativeFaultEvidence {
    crate::observability::AttemptNativeFaultEvidence {
        isolation: crate::observability::AttemptFaultIsolation::Unknown,
        teardown_health: teardown_health(cleanup_evidence),
        device_health: None,
        first_fault: None,
    }
}

fn device_local_native_evidence(
    cleanup_evidence: CleanupEvidence,
    health: cudf::runtime::DeviceHealthSnapshot,
) -> crate::observability::AttemptNativeFaultEvidence {
    use crate::observability::{
        AttemptDeviceHealth, AttemptDeviceHealthStatus, AttemptFaultIsolation, AttemptNativeFault,
        AttemptNativeFaultKind, AttemptNativeStatus,
    };

    let first_fault = health.first_fault().map(|fault| AttemptNativeFault {
        kind: match fault.kind() {
            cudf::runtime::DeviceFaultKind::CurrentDeviceQueryFailed => {
                AttemptNativeFaultKind::CurrentDeviceQueryFailed
            },
            cudf::runtime::DeviceFaultKind::OwnerDeviceEntryFailed => AttemptNativeFaultKind::OwnerDeviceEntryFailed,
            cudf::runtime::DeviceFaultKind::AmbientDeviceRestoreFailed => {
                AttemptNativeFaultKind::AmbientDeviceRestoreFailed
            },
            cudf::runtime::DeviceFaultKind::FatalDeviceOperation => AttemptNativeFaultKind::FatalDeviceOperation,
            cudf::runtime::DeviceFaultKind::TeardownQuarantined => AttemptNativeFaultKind::TeardownQuarantined,
            _ => AttemptNativeFaultKind::Unknown,
        },
        status: fault.native_status().and_then(|status| match status {
            cudf::runtime::DeviceFaultNativeStatus::Cudf(status) => Some(AttemptNativeStatus::Cudf(status)),
            cudf::runtime::DeviceFaultNativeStatus::Cuda(status) => Some(AttemptNativeStatus::Cuda(status)),
            _ => None,
        }),
    });
    crate::observability::AttemptNativeFaultEvidence {
        isolation: AttemptFaultIsolation::DeviceLocal,
        teardown_health: teardown_health(cleanup_evidence),
        device_health: Some(AttemptDeviceHealth {
            status: match health.status() {
                cudf::runtime::DeviceHealthStatus::Healthy => AttemptDeviceHealthStatus::Healthy,
                cudf::runtime::DeviceHealthStatus::Unhealthy => AttemptDeviceHealthStatus::Unhealthy,
            },
            fault_count: health.fault_count(),
            rejected_operation_count: health.rejected_operation_count(),
            quarantined_teardown_count: health.quarantined_teardown_count(),
        }),
        first_fault,
    }
}
