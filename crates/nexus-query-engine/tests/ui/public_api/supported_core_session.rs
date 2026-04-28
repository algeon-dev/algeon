use nexus_query_engine::{
    error::Result,
    observability::RuntimeObservationHandle,
    runtime::{
        DeploymentIoCapability, ExecutionContext, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore,
        GpuBackendCoreDraft, GpuBackendDeviceConfig, GpuBackendMemoryShare, GpuBackendSession, GpuBackendTenantDraft,
        GpuExecutionJobKind, GpuExecutorLease, GpuMemoryOwnership, KvikioCompatModeStatus, NvidiaFsProbeStatus,
        QueryHandle, QueryServiceConfig,
    },
};

fn use_session(core: &GpuBackendCore) -> GpuBackendSession {
    core.session_capability()
}

async fn use_executor(executor: &GpuExecutorLease) -> Result<()> {
    executor.run(GpuExecutionJobKind::QueryExecution, async {}).await?;
    executor
        .run_observed_scoped(
            GpuExecutionJobKind::QueryExecution,
            RuntimeObservationHandle::disabled(),
            |scope| async move {
                let _ = scope;
            },
        )
        .await
}

fn context_executor(context: &ExecutionContext) -> Result<GpuExecutorLease> {
    context.gpu_executor()
}

fn query_executor(handle: &QueryHandle) -> Result<GpuExecutorLease> {
    handle.gpu_executor()
}

fn main() {
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let controller = GpuBackendControllerDraft::from_legacy_policy(
        QueryServiceConfig::default(),
        vec![GpuBackendTenantDraft::new("tenant", GpuBackendMemoryShare::Bytes(1), GpuBackendMemoryShare::Bytes(1))],
    );
    let draft = GpuBackendCoreDraft::new(
        vec![GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive)],
        controller,
        deployment,
    );
    let _builder = GpuBackendCore::builder(draft);
    let _supported_session_factory: fn(&GpuBackendCore) -> GpuBackendSession = use_session;
    let _supported_executor_api = use_executor;
    let _supported_context_executor: fn(&ExecutionContext) -> Result<GpuExecutorLease> = context_executor;
    let _supported_query_executor: fn(&QueryHandle) -> Result<GpuExecutorLease> = query_executor;
}
