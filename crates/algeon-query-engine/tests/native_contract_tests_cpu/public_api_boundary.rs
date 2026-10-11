use query_planning::error::Result;
#[cfg(feature = "qualification")]
use query_runtime::runtime::GpuBackendFaultInjector;
use query_runtime::{
    observability::RuntimeObservationHandle,
    runtime::{
        AdmittedQueryContext, ExecutionContext, GpuBackendConfig, GpuBackendCore, GpuBackendDeviceConfig,
        GpuBackendDeviceProfile, GpuBackendSession, GpuExecutionJobKind, GpuExecutorLease, GpuMemoryOwnership,
        QueryExecutionConfig, QueryHandle, QueueBounds, ValidatedGpuBackendConfig,
    },
};

fn acquire_admitted_context(session: &GpuBackendSession) -> Result<()> {
    let handle = session.admit_query()?;
    let admitted: AdmittedQueryContext = handle.acquire_context()?;
    let _context: &ExecutionContext = admitted.context();
    Ok(())
}

fn supported_admitted_context_path_compiles() {
    // Never invoked with a real session. This is a compile-time proof (a
    // fn-pointer coercion) that the signature composes.
    let _supported_admitted_context: fn(&GpuBackendSession) -> Result<()> = acquire_admitted_context;
}

const _: fn() = supported_admitted_context_path_compiles;

fn use_session(core: &GpuBackendCore) -> GpuBackendSession {
    core.session_capability()
}

#[cfg(feature = "qualification")]
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

fn supported_core_session_path_compiles() {
    // Naming the construction functions touches no cudf FFI and no
    // device or NVML probe, so this runs in the CPU lane.
    let _validate: fn(GpuBackendConfig) -> Result<ValidatedGpuBackendConfig> = GpuBackendConfig::validate;
    let _construct: fn(ValidatedGpuBackendConfig) -> Result<GpuBackendCore> = GpuBackendCore::new;
    let _config = GpuBackendConfig::new(
        vec![GpuBackendDeviceConfig::new(GpuBackendDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive)],
        QueryExecutionConfig::default(),
        QueueBounds::default(),
    );
    let _supported_session_factory: fn(&GpuBackendCore) -> GpuBackendSession = use_session;
    #[cfg(feature = "qualification")]
    let _supported_fault_injector_factory: fn(&GpuBackendCore) -> GpuBackendFaultInjector = use_fault_injector;
    let _supported_executor_api = use_executor;
    let _supported_context_executor: fn(&ExecutionContext) -> Result<GpuExecutorLease> = context_executor;
    let _supported_query_executor: fn(&QueryHandle) -> Result<GpuExecutorLease> = query_executor;
}

const _: fn() = supported_core_session_path_compiles;
