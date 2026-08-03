use nexus_query_engine::{attempt::AttemptHandle, observability::RuntimeTerminalOutcome};

fn caller_cannot_terminalize_an_attempt(handle: AttemptHandle) {
    let _ = handle.terminalize(RuntimeTerminalOutcome::Cancelled);
}

fn main() {}
