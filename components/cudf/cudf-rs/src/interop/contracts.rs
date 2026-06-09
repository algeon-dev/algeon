use super::*;

pub(super) fn export_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedDeviceExport)
}

pub(super) fn host_export_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedHostExport)
}

pub(super) fn owned_transfer_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::OwnedDeviceTransfer)
}

pub(super) fn owned_transfer_release_contract(device: i32) -> ReleaseContract {
    ReleaseContract::new(
        device,
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
        ReleaseSynchronization::MaySynchronizeInternally,
    )
}

pub(super) fn owned_transfer_runtime_release_contract(device: i32) -> ReleaseContract {
    ReleaseContract::new(
        device,
        TransferCompletionRule::ReleaseWaitsOnConsumerSignal,
        ReleaseSynchronization::Asynchronous,
    )
}

pub(super) fn owned_transfer_runtime_release_contract_with_synchronization(
    device: i32,
    synchronization: ReleaseSynchronization,
) -> Result<ReleaseContract, InteropError> {
    match synchronization {
        ReleaseSynchronization::Asynchronous => Ok(owned_transfer_runtime_release_contract(device)),
        ReleaseSynchronization::MaySynchronizeInternally => Ok(ReleaseContract::new(
            device,
            TransferCompletionRule::ReleaseWaitsOnConsumerSignal,
            ReleaseSynchronization::MaySynchronizeInternally,
        )),
        ReleaseSynchronization::RequiresCallerSynchronization => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "runtime-backed owned-transfer release does not support caller-managed synchronization",
        )
        .with_context("completion", "release_waits_on_consumer_signal")
        .with_context("synchronization", "requires_caller_synchronization")),
    }
}
