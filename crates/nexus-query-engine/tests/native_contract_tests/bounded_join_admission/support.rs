fn assert_modeled_reservation_facts_are_consistent(
    facts: &std::collections::BTreeMap<&'static str, String>,
    label: &str,
) {
    let kinds = csv_values(facts, "bounded_join_reservation_kinds", label);
    let statuses = csv_pairs(facts, "bounded_join_reservation_statuses", label);
    let requirements = csv_triples(facts, "bounded_join_reservation_requirements", label);
    let modeled = csv_values(facts, "modeled_reservations", label);
    let blocked = csv_values(facts, "blocked_reservations", label);
    let missing = csv_values(facts, "missing_reservations", label);

    let status_kinds = statuses.iter().map(|(kind, _)| *kind).collect::<Vec<_>>();
    let requirement_kinds = requirements.iter().map(|(kind, _, _)| *kind).collect::<Vec<_>>();
    assert_eq!(status_kinds, kinds, "{label}: reservation statuses must cover every reservation kind");
    assert_eq!(requirement_kinds, kinds, "{label}: reservation requirements must cover every reservation kind");

    let expected_modeled =
        statuses.iter().filter_map(|(kind, status)| (*status == "modeled").then_some(*kind)).collect::<Vec<_>>();
    let expected_blocked =
        statuses.iter().filter_map(|(kind, status)| (*status != "modeled").then_some(*kind)).collect::<Vec<_>>();
    assert_eq!(modeled, expected_modeled, "{label}: modeled_reservations must match modeled statuses");
    assert_eq!(blocked, expected_blocked, "{label}: blocked_reservations must match non-modeled statuses");
    assert_eq!(missing, expected_blocked, "{label}: missing_reservations must match non-modeled statuses");

    for (kind, contract, quality) in requirements {
        if quality == "hard_budgeted_report_with_unresolved_preflight" {
            assert!(
                matches!(
                    contract,
                    "source_contract"
                        | "aggregate_contract"
                        | "final_sort_contract"
                        | "join_filtered_build_key_deduplicate_output_contract"
                ),
                "{label}: {kind} hard-budgeted requirement must name a lower-level preflight contract"
            );
        }
    }
}

fn assert_rejection_facts_are_family_free(facts: &std::collections::BTreeMap<&'static str, String>, label: &str) {
    for retired_key in [
        "bounded_join_execution_family",
        "bounded_join_blocked_join_family",
        "bounded_join_specific_blocker",
        "bounded_join_shape_blocker",
    ] {
        assert_eq!(facts.get(retired_key), None, "{label}: retired rejection fact {retired_key} must not be emitted");
    }
    for (key, value) in facts {
        assert!(
            !value.contains("topology_unsupported")
                && !value.contains("BoundedJoinExecutionFamily")
                && !value.contains("BoundedJoinExecutionPath"),
            "{label}: {key}={value:?} must not carry retired family/path vocabulary"
        );
    }
}

fn assert_strict_report_has_modeled_subcontract(
    report: &CapabilityContractReport,
    expected_contract: &'static str,
    label: &str,
) {
    assert!(
        report.gate_rows().iter().any(|row| {
            row.reservation_kind == ReservationKind::CapabilityContract
                && row.memory_boundary == expected_contract
                && row.required_contract == expected_contract
                && row.contract_status == ContractStatus::Modeled
                && row.contract_quality == ContractQuality::CapabilityProof
        }),
        "{label}: strict report should expose modeled subcontract {expected_contract}"
    );
}

fn assert_strict_report_lacks_modeled_subcontract(
    report: &CapabilityContractReport,
    unexpected_contract: &'static str,
    label: &str,
) {
    assert!(
        !report.gate_rows().iter().any(|row| {
            row.reservation_kind == ReservationKind::CapabilityContract
                && row.memory_boundary == unexpected_contract
                && row.required_contract == unexpected_contract
                && row.contract_status == ContractStatus::Modeled
        }),
        "{label}: strict report must not expose modeled subcontract {unexpected_contract}"
    );
}

fn assert_hash_join_prebuild_exact_requirement(
    requirements: &[nexus_query_engine::capability::ReservationRequirement],
    label: &str,
) {
    assert!(
        requirements.iter().any(|requirement| {
            requirement.kind == ReservationKind::HashTable
                && requirement.memory_boundary == "join_keyed_equality_preflight_hash_table"
                && requirement.required_contract == "hash_join_pre_build_reservation_size_bytes"
                && requirement.status == ContractStatus::Modeled
                && requirement.quality == ContractQuality::ExactPreflight
        }),
        "{label}: equality-join output-size preflight must model its temporary hash-table reservation"
    );
}

fn assert_hash_join_prebuild_exact_gate(pipeline: &nexus_query_engine::pipeline::CompiledPipeline, label: &str) {
    assert!(
        pipeline.gates().iter().any(|gate| {
            gate.reservation_kind() == ReservationKind::HashTable
                && gate.memory_boundary() == "join_keyed_equality_preflight_hash_table"
                && gate.required_contract() == "hash_join_pre_build_reservation_size_bytes"
                && gate.status() == ContractStatus::Modeled
                && gate.quality() == ContractQuality::ExactPreflight
        }),
        "{label}: compiled pipeline must carry the temporary hash-table exact preflight gate"
    );
}

fn csv_values<'a>(facts: &'a std::collections::BTreeMap<&'static str, String>, key: &str, label: &str) -> Vec<&'a str> {
    facts
        .get(key)
        .unwrap_or_else(|| panic!("{label}: missing {key}"))
        .split(',')
        .filter(|value| !value.is_empty())
        .collect()
}

fn csv_pairs<'a>(
    facts: &'a std::collections::BTreeMap<&'static str, String>,
    key: &str,
    label: &str,
) -> Vec<(&'a str, &'a str)> {
    csv_values(facts, key, label)
        .into_iter()
        .map(|value| value.split_once(':').unwrap_or_else(|| panic!("{label}: {key} row {value:?} must be name:value")))
        .collect()
}

fn csv_triples<'a>(
    facts: &'a std::collections::BTreeMap<&'static str, String>,
    key: &str,
    label: &str,
) -> Vec<(&'a str, &'a str, &'a str)> {
    csv_values(facts, key, label)
        .into_iter()
        .map(|value| {
            let mut fields = value.splitn(3, ':');
            let first = fields.next().unwrap_or("");
            let second = fields.next().unwrap_or("");
            let third = fields.next().unwrap_or("");
            if first.is_empty() || second.is_empty() || third.is_empty() {
                panic!("{label}: {key} row {value:?} must be name:contract:detail");
            }
            (first, second, third)
        })
        .collect()
}
