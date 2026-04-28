use nexus_query_engine::runtime::GpuExecutionPool;

fn main() {
    let _ = GpuExecutionPool::new(0, 1);
}
