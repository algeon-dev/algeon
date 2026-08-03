use nexus_query_engine::{
    error::Result,
    runtime::{AdmittedQueryContext, GpuBackendSession},
};

fn caller_cannot_extract_attempt_observation_owner(session: &GpuBackendSession) -> Result<()> {
    let handle = session.admit_query()?;
    let admitted: AdmittedQueryContext = handle.acquire_context()?;
    let _ = admitted.context().runtime_observation_owner();
    Ok(())
}

fn main() {
    let _cannot_extract_attempt_observation_owner: fn(&GpuBackendSession) -> Result<()> =
        caller_cannot_extract_attempt_observation_owner;
}
