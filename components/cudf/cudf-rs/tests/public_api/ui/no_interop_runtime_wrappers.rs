use cudf::interop::{
    runtime::RuntimeCompletionAuthority,
    transfer::{RuntimeOwnedColumnTransferReleaser, RuntimeOwnedTableTransferReleaser},
};

fn main() {
    let _ = RuntimeCompletionAuthority::SynchronousByReturn;
    let _: Option<RuntimeOwnedColumnTransferReleaser> = None;
    let _: Option<RuntimeOwnedTableTransferReleaser> = None;
}
