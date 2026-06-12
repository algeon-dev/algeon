use cudf::__DebugExecutionContext;

fn main() {
    let _ = std::any::type_name::<__DebugExecutionContext<'static>>();
}
