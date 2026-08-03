use nexus_query_engine::{attempt::AttemptAdmissionBinding, runtime::QueryAdmissionRequest};

fn caller_cannot_attach_a_finalizer_separately(
    request: QueryAdmissionRequest,
    binding: AttemptAdmissionBinding,
) {
    let _ = request.with_attempt_finalizer(binding);
}

fn main() {}
