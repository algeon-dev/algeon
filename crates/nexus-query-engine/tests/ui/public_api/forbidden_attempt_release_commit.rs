use nexus_query_engine::attempt::AttemptReleaseCommit;

fn caller_cannot_receive_a_split_attempt_release_capability(release_commit: AttemptReleaseCommit) {
    let _ = release_commit;
}

fn main() {}
