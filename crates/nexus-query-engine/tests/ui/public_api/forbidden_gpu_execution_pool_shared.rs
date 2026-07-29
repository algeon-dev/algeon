use nexus_query_engine::runtime::GpuExecutionPool;

fn main() {
    let _ = GpuExecutionPool::shared_for_device(0, 1);
}
