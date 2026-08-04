use nexus_query_engine::{
    error::Result,
    observability::RuntimeObservationHandle,
    runtime::{
        DeploymentIoCapability, ExecutionContext, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore,
        GpuBackendCoreDraft, GpuBackendDeviceConfig, GpuBackendFaultInjector, GpuBackendSession, GpuExecutionJobKind, GpuExecutorLease,
        GpuMemoryOwnership, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryExecutionConfig, QueryHandle, QueueBounds,
    },
};

fn use_session(core: &GpuBackendCore) -> GpuBackendSession {
    core.session_capability()
}

fn use_fault_injector(core: &GpuBackendCore) -> GpuBackendFaultInjector {
    core.fault_injector()
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
    let controller = GpuBackendControllerDraft::new(QueryExecutionConfig::default(), QueueBounds::default());
    let draft = GpuBackendCoreDraft::new(
        vec![GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive)],
        controller,
        deployment,
    );
    let _builder = GpuBackendCore::builder(draft);
    let _supported_session_factory: fn(&GpuBackendCore) -> GpuBackendSession = use_session;
    let _supported_fault_injector_factory: fn(&GpuBackendCore) -> GpuBackendFaultInjector = use_fault_injector;
    let _supported_executor_api = use_executor;
    let _supported_context_executor: fn(&ExecutionContext) -> Result<GpuExecutorLease> = context_executor;
    let _supported_query_executor: fn(&QueryHandle) -> Result<GpuExecutorLease> = query_executor;
}
