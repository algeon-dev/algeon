use cudf::ExecutionContext;

fn main() {
    let _ = std::mem::size_of::<ExecutionContext<'static>>();
}
