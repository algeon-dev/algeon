//! This binary used to run 49 `trybuild` fixtures: 2 `pass` cases proving
//! supported construction paths still compile, plus ~47 `compile_fail` cases
//! each asserting one forbidden item is absent from the public API.
//! `trybuild` 1.0.118 hardcodes its own `CARGO_TARGET_DIR`, so every one of
//! those compile_fail cases built a full second copy of this crate's
//! dependency graph (cudf, cugraph, datafusion, ...) in an isolated
//! workspace. That duplicate build was the single most expensive thing in
//! the CPU test phase.
//!
//! The forbidden-item half of that coverage moved to
//! `scripts/ci/check_public_api_surface.sh`, which snapshots this crate's
//! complete public API surface (via `cargo public-api`, built on rustdoc
//! JSON in the ordinary target directory — no duplicate dependency build)
//! and fails on any unreviewed drift, including additions the old
//! hand-written cases never anticipated. Every migrated case reduced to a
//! plain "item is absent/private" fact (rustc E0599/E0432/E0603/E0624/
//! E0560/E0609 in the original fixtures), which a surface snapshot expresses
//! exactly: the forbidden item simply never appears in the checked-in
//! snapshot, and reintroducing it as `pub` shows up as an unreviewed
//! addition.
//!
//! What remains here are the two `pass` cases, unchanged in spirit but no
//! longer routed through trybuild's second workspace: they assert a
//! different thing a surface snapshot cannot express, namely that the
//! supported construction paths still compose and typecheck end to end (not
//! just that each item's signature exists in isolation). Compiling them as
//! ordinary `#[test]` functions in this crate's own test binary proves the
//! same fact for free, in the same build the rest of this binary already
//! pays for.

use nexus_query_engine::{
    error::Result,
    observability::RuntimeObservationHandle,
    runtime::{
        AdmittedQueryContext, DeploymentIoCapability, ExecutionContext, GdsDriverProbeStatus,
        GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft, GpuBackendDeviceConfig,
        GpuBackendFaultInjector, GpuBackendSession, GpuExecutionJobKind, GpuExecutorLease, GpuMemoryOwnership,
        KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryExecutionConfig, QueryHandle, QueueBounds,
    },
};

fn acquire_admitted_context(session: &GpuBackendSession) -> Result<()> {
    let handle = session.admit_query()?;
    let admitted: AdmittedQueryContext = handle.acquire_context()?;
    let _context: &ExecutionContext = admitted.context();
    Ok(())
}

#[test]
fn supported_admitted_context_path_compiles() {
    // Never invoked with a real session: this is a compile-time proof (a
    // fn-pointer coercion) that the signature composes, matching the former
    // trybuild `pass` fixture's own `fn main`, which also never called it.
    let _supported_admitted_context: fn(&GpuBackendSession) -> Result<()> = acquire_admitted_context;
}

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

#[test]
fn supported_core_session_path_compiles() {
    // `GpuBackendCoreDraft`/`GpuBackendControllerDraft`/`DeploymentIoCapability::from_probe_inputs`
    // and `GpuBackendCore::builder` are pure in-memory construction (no cudf
    // FFI, no device/NVML probe — verified by reading their bodies), so this
    // runs safely in the CPU lane exactly as the trybuild `pass` fixture did.
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
