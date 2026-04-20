use cudf::interop::transfer::{OwnedColumnTransfer, ReleaseSynchronization, TransferCompletionRule};

fn main() {
    let _: Option<OwnedColumnTransfer> = None;
    let _ = ReleaseSynchronization::Asynchronous;
    let _ = TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes;
}
