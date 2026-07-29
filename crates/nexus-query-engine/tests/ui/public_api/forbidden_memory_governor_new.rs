use nexus_query_engine::runtime::MemoryGovernor;

fn main() {
    let _ = MemoryGovernor::new(0, 1);
}
