use nexus_query_engine::attempt::AttemptAdmissionBinding;

fn caller_cannot_detach_attempt_observation_sealing(binding: AttemptAdmissionBinding) {
    let _ = binding.observation_owner();
}

fn main() {}
