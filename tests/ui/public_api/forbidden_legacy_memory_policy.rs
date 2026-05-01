use datafusion_nexus::backend::NexusMemoryPolicy;

fn main() {
    let _ = std::mem::size_of::<NexusMemoryPolicy>();
}
