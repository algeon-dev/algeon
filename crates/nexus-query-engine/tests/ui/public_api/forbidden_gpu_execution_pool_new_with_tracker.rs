use nexus_query_engine::runtime::{GpuExecutionPool, GpuExecutionQueryTracker};

fn main() {
    let tracker = GpuExecutionQueryTracker::new();
    let _ = GpuExecutionPool::new_with_tracker(0, 1, tracker);
}
