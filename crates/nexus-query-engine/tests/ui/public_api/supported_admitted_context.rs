use nexus_query_engine::{
    error::Result,
    runtime::{ExecutionContext, GpuBackendSession, TenantId, TenantQueryContext},
};

fn acquire_admitted_context(session: &GpuBackendSession) -> Result<()> {
    let handle = session.admit_query(TenantId::from("tenant"))?;
    let admitted: TenantQueryContext = handle.acquire_context()?;
    let _context: &ExecutionContext = admitted.context();
    Ok(())
}

fn main() {
    let _supported_admitted_context: fn(&GpuBackendSession) -> Result<()> = acquire_admitted_context;
}
