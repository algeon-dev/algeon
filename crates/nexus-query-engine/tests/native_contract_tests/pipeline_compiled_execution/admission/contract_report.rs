#[test]
fn d5_contract_gate_rows_are_inventory_backed_and_family_path_free() {
    let mut hard_budgeted_unresolved_rows = 0;
    let mut source_unresolved_rows = 0;
    let mut aggregate_unresolved_rows = 0;
    let mut final_sort_unresolved_rows = 0;
    let mut one_shot_output_unresolved_rows = 0;
    let mut filtered_retained_state_rows = 0;
    let mut mixed_retained_state_rows = 0;
    let mut observed_retained_state_rows = 0;

    for case in d3_1_inventory_cases() {
        let analysis = case.plan.capability_analysis();
        assert!((case.admits)(&analysis), "{}: expected D.5 inventory case to remain admitted", case.label);
        let pipeline = compile_pipeline(&case.plan, &analysis)
            .unwrap_or_else(|error| panic!("{}: compile failed: {error}", case.label));

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{}: strict bounded contract report failed: {error}", case.label));
        assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
        assert!(report.is_strict_bounded_ready(), "{}: compiled gate report should be strict-ready", case.label);
        assert_eq!(
            report.gate_rows().len(),
            pipeline.gates().len(),
            "{}: report should preserve every compiled reservation gate",
            case.label
        );

        let capability_report = StrictBoundedContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("{}: capability contract report failed: {error}", case.label));
        assert_eq!(capability_report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
        assert!(capability_report.is_strict_bounded_ready());

        for row in report.gate_rows() {
            let rendered = row.to_stable_kv_row();
            assert_d5_contract_gate_row_is_family_path_free(case.label, &rendered);
            assert_d5_contract_gate_row_excludes_adaptive_terms(case.label, &rendered);
            assert!(rendered.contains("contract_status="), "{}: row must serialize contract status", case.label);
            assert!(rendered.contains("contract_quality="), "{}: row must serialize contract quality", case.label);
            assert!(
                !rendered.contains("contract_modes"),
                "{}: row must not serialize legacy contract modes",
                case.label
            );

            if row.contract_quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight {
                hard_budgeted_unresolved_rows += 1;
                match row.reservation_kind {
                    ReservationKind::SourceRead => source_unresolved_rows += 1,
                    ReservationKind::OperatorOutput => aggregate_unresolved_rows += 1,
                    ReservationKind::FinalSort => final_sort_unresolved_rows += 1,
                    ReservationKind::JoinOutput => one_shot_output_unresolved_rows += 1,
                    _ => {},
                }
                assert_eq!(
                    row.contract_status,
                    ContractStatus::Modeled,
                    "{}: hard-budgeted unresolved preflight rows stay modeled",
                    case.label
                );
                assert!(
                    row.unresolved_preflight_gap.is_some_and(|gap| gap.starts_with("libcudf_")),
                    "{}: hard-budgeted row should name its lower-level preflight gap",
                    case.label
                );
            }

            if matches!(row.reservation_kind, ReservationKind::FilteredProbeState | ReservationKind::MixedSizeData) {
                observed_retained_state_rows += 1;
                if row.reservation_kind == ReservationKind::FilteredProbeState {
                    filtered_retained_state_rows += 1;
                }
                if row.reservation_kind == ReservationKind::MixedSizeData {
                    mixed_retained_state_rows += 1;
                }
                assert_eq!(
                    row.contract_quality,
                    ContractQuality::ObservedRetainedState,
                    "{}: retained join state must not collapse into exact preflight",
                    case.label
                );
            }

            match row.reservation_kind {
                ReservationKind::HashTable if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert_eq!(
                        row.required_contract, "hash_join_pre_build_reservation_size_bytes",
                        "{}: hash-table exact preflight must name its cudf-rs pre-build sizing owner",
                        case.label
                    );
                },
                ReservationKind::GatherMap if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert_eq!(
                        row.required_contract, "join_gather_map_reservation_size_bytes",
                        "{}: gather-map exact preflight must name its cudf-rs gather-map sizing owner",
                        case.label
                    );
                },
                ReservationKind::JoinOutput if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert!(
                        matches!(
                            row.required_contract,
                            "dataframe_gather_output_reservation_size_bytes"
                                | "projected_cross_join_reservation_size_bytes"
                                | "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes"
                        ),
                        "{}: join-output exact preflight must name an owned lower-level output sizing contract",
                        case.label
                    );
                },
                _ => {},
            }
        }
    }

    assert!(hard_budgeted_unresolved_rows > 0, "D.5 inventory should cover unresolved hard-budgeted gates");
    assert!(source_unresolved_rows > 0, "D.5 inventory should cover source unresolved preflight gates");
    assert!(aggregate_unresolved_rows > 0, "D.5 inventory should cover aggregate unresolved preflight gates");
    assert!(final_sort_unresolved_rows > 0, "D.5 inventory should cover final-sort unresolved preflight gates");
    assert!(
        one_shot_output_unresolved_rows > 0,
        "D.5 inventory should cover one-shot equality join output-size preflight gates"
    );
    assert!(filtered_retained_state_rows > 0, "D.5 inventory should cover filtered semi/anti retained state");
    assert!(mixed_retained_state_rows > 0, "D.5 inventory should cover mixed residual retained state");
    assert!(observed_retained_state_rows > 0, "D.5 inventory should cover observed retained-state gates");
}

#[test]
fn strict_preflight_gap_strict_preflight_gap_registry_tracks_current_retirement_protocol() {
    assert_eq!(
        STRICT_UNRESOLVED_PREFLIGHT_GAPS.len(),
        8,
        "strict unresolved preflight gap retirement must remove entries from this registry"
    );

    let mut markers = BTreeSet::new();
    let mut requirements = BTreeSet::new();
    for gap in STRICT_UNRESOLVED_PREFLIGHT_GAPS {
        assert!(
            markers.insert(gap.unresolved_gap),
            "unresolved preflight markers should be unique: {}",
            gap.unresolved_gap
        );
        assert!(
            requirements.insert(format!("{}:{}", gap.kind.as_str(), gap.required_contract)),
            "reservation-kind/contract pairs should be unique: {}:{}",
            gap.kind.as_str(),
            gap.required_contract
        );
        assert!(
            gap.retirement_requirement.contains("cudf-rs/libcudf"),
            "retirement must wait on an owning lower-level API: {}",
            gap.retirement_requirement
        );
        assert_eq!(lower_level_unresolved_preflight_gap(gap.kind, gap.required_contract), Some(gap.unresolved_gap));
    }

    assert!(markers.contains("libcudf_parquet_decoded_output_preflight_unavailable"));
    assert!(markers.contains("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"));
    assert!(markers.contains("libcudf_window_output_temporary_workspace_preflight_unavailable"));
    assert!(markers.contains("libcudf_regexp_replace_output_preflight_unavailable"));
    assert!(markers.contains("libcudf_edge_normalize_output_preflight_unavailable"));
    assert!(markers.contains("libcudf_final_sort_order_preflight_unavailable"));
    assert!(markers.contains("libcudf_equality_join_output_size_preflight_unavailable"));
}

#[test]
fn d5_contract_report_rejects_unresolved_preflight_gate_labeled_exact() {
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let mut gates = pipeline.gates().to_vec();
    let source_gate_index = gates
        .iter()
        .position(|gate| gate.reservation_kind() == ReservationKind::SourceRead)
        .expect("root join should have source-read gates");
    let source_gate = &gates[source_gate_index];
    gates[source_gate_index] = ReservationGate::new(
        source_gate.id(),
        source_gate.owner(),
        ReservationKind::SourceRead,
        ReservationRequirementScope::Edge,
        "source_read",
        "source_contract",
        ContractStatus::Modeled,
        ContractQuality::ExactPreflight,
    );
    let malformed = CompiledPipeline::new(
        pipeline.tasks().to_vec(),
        pipeline.edges().to_vec(),
        pipeline.barriers().to_vec(),
        pipeline.resources().to_vec(),
        gates,
        pipeline.sink().clone(),
    );

    let error = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &malformed)
        .expect_err("D.5 report construction should reject mislabeled unresolved preflight");

    assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("contract_quality"), Some("exact_preflight"));
}

#[test]
fn strict_preflight_gap_contract_report_rejects_hard_budgeted_quality_without_tracked_gap() {
    let invalid = d5_test_gate_row(
        ReservationKind::SourceRead,
        "source_read",
        "libcudf_decoded_output_preflight_contract_after_api_lands",
        ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        None,
    );

    let error = StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
        .expect_err("F.6 should reject unresolved quality after the placeholder contract is replaced");

    assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(
        error.fact_value("required_contract"),
        Some("libcudf_decoded_output_preflight_contract_after_api_lands")
    );
}

#[test]
fn d5_contract_report_rejects_public_gate_rows_with_invalid_quality_accounting() {
    for invalid in [
        d5_test_gate_row(
            ReservationKind::SourceRead,
            "source_read",
            "source_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::FilteredProbeState,
            "join_filtered_probe_state",
            "filtered_join_retained_probe_state",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::MixedSizeData,
            "join_mixed_size_data",
            "mixed_keyed_residual_or_conditional_full_retained_size_data",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "adapter_guessed_hash_table_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::GatherMap,
            "join_gather_map",
            "adapter_guessed_gather_map_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::JoinOutput,
            "join_output",
            "adapter_guessed_join_output_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
    ] {
        let kind = invalid.reservation_kind;
        let error =
            StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
                .unwrap_err();
        assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reservation_kind"), Some(kind.as_str()));
    }
}

#[test]
fn d5_contract_report_rejects_public_unresolved_preflight_rows_with_non_hard_budgeted_quality() {
    let unresolved_contracts = [
        (ReservationKind::SourceRead, "source_read", "source_contract"),
        (ReservationKind::OperatorOutput, "aggregate_output", "aggregate_contract"),
        (ReservationKind::FinalSort, "final_sort", "final_sort_contract"),
        (ReservationKind::JoinOutput, "join_output", "keyed_equality_join_output_size_preflight"),
    ];
    let non_hard_budgeted_qualities = [
        ContractQuality::ExactPreflight,
        ContractQuality::MetadataBound,
        ContractQuality::ObservedRetainedState,
        ContractQuality::QueryEngineOwnedOrdering,
        ContractQuality::QueryEngineOwnedBoundary,
        ContractQuality::CapabilityProof,
    ];

    for (reservation_kind, memory_boundary, required_contract) in unresolved_contracts {
        for contract_quality in non_hard_budgeted_qualities {
            let invalid =
                d5_test_gate_row(reservation_kind, memory_boundary, required_contract, contract_quality, None);
            let error =
                StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
                    .unwrap_err();
            assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::InvalidPlan);
            assert_eq!(error.fact_value("reservation_kind"), Some(reservation_kind.as_str()));
            assert_eq!(error.fact_value("required_contract"), Some(required_contract));
            assert_eq!(error.fact_value("contract_quality"), Some(contract_quality.as_str()));
        }
    }
}

#[test]
fn d5_contract_report_rejects_public_rows_with_inconsistent_execution_modes_and_scope_owners() {
    let strict_with_blocked = StrictBoundedContractGateRow {
        contract_status: ContractStatus::BlockedByLowerLevel,
        ..d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "hash_join_pre_build_reservation_size_bytes",
            ContractQuality::ExactPreflight,
            None,
        )
    };
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::StrictBounded,
        vec![strict_with_blocked],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("report_execution_mode"), Some("strict_bounded"));

    let unsupported_with_modeled = d5_test_gate_row(
        ReservationKind::HashTable,
        "join_hash_table",
        "hash_join_pre_build_reservation_size_bytes",
        ContractQuality::ExactPreflight,
        None,
    );
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::UnsupportedStrictBounded,
        vec![unsupported_with_modeled],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("report_execution_mode"), Some("unsupported_strict_bounded"));

    let edge_owned_node_scope = StrictBoundedContractGateRow {
        owner: StrictBoundedContractOwner::Compiled(ReservationGateOwner::Edge(PipelineEdgeId::new(0))),
        ..d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "hash_join_pre_build_reservation_size_bytes",
            ContractQuality::ExactPreflight,
            None,
        )
    };
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::StrictBounded,
        vec![edge_owned_node_scope],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("contract_scope"), Some("node"));
    assert_eq!(error.fact_value("owner"), Some("edge:e0"));
}
