fn assert_rejection_facts_are_family_free(facts: &std::collections::BTreeMap<&'static str, String>, label: &str) {
    for retired_key in [
        "bounded_join_execution_family",
        "bounded_join_blocked_join_family",
        "bounded_join_specific_blocker",
        "bounded_join_shape_blocker",
        "bounded_join_lower_level_blockers",
        "bounded_join_required_shape_contract",
        "bounded_join_required_lower_level_contract",
    ] {
        assert_eq!(facts.get(retired_key), None, "{label}: retired rejection fact {retired_key} must not be emitted");
    }
    for (key, value) in facts {
        assert!(
            !value.contains("topology_unsupported")
                && !value.contains("BoundedJoinExecutionFamily")
                && !value.contains("BoundedJoinExecutionPath")
                && !key.ends_with("_pipeline_capability"),
            "{label}: {key}={value:?} must not carry retired family/path vocabulary"
        );
    }
}
