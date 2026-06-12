use cudf::interop::prepared::{CompletionAuthority, PreparedColumnExport, ReadStartAuthority};

fn main() {
    let _: Option<PreparedColumnExport<'static>> = None;
    let _ = CompletionAuthority::SynchronousByReturn;
    let _ = ReadStartAuthority::SynchronousByReturn;
}
