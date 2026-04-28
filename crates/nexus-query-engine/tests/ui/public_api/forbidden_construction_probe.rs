use nexus_query_engine::runtime::{GpuBackendConstructionCheckpoint, GpuBackendConstructionProbe};

fn main() {
    let _ = std::mem::size_of::<GpuBackendConstructionCheckpoint>();
    let _probe: Option<&dyn GpuBackendConstructionProbe> = None;
}
