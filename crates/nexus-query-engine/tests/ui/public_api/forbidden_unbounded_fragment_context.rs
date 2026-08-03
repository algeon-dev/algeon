use nexus_query_engine::runtime::ExecutionContext;

fn downgrade_admitted_context(context: &ExecutionContext) {
    let _ = context.for_unbounded_fragment();
}

fn main() {}
