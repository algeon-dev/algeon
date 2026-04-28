use std::sync::Arc;

use nexus_query_engine::runtime::ExecutionContext;

fn construct_context(runtime: Arc<cudf::Runtime>) {
    let _ = ExecutionContext::from_shared_runtime(runtime);
}

fn main() {
    let _ = construct_context;
}
