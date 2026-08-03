use nexus_query_engine::attempt::AttemptAdmissionBinding;

fn caller_cannot_split_attempt_binding(binding: AttemptAdmissionBinding) {
    let _ = binding.into_parts();
}

fn main() {}
