use nexus_query_engine::attempt::AttemptFinalizer;

fn caller_cannot_receive_a_split_attempt_finalizer(finalizer: AttemptFinalizer) {
    let _ = finalizer;
}

fn main() {}
