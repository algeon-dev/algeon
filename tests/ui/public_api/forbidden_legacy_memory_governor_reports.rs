use datafusion_nexus::backend::NexusGpuBackend;

fn main() {
    let _ = NexusGpuBackend::memory_governor_reports;
}
