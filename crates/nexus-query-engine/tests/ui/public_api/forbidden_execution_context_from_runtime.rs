use nexus_query_engine::runtime::ExecutionContext;

fn construct_context(runtime: cudf::Runtime) {
    let _ = ExecutionContext::from_runtime(runtime);
}

fn main() {
    let _ = construct_context;
}
