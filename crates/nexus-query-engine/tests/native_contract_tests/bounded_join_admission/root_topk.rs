#[test]
fn bounded_root_topk_join_aggregate_pipeline_is_executable_with_modeled_contracts() {
    let plan = root_topk_join_aggregate_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_root_join());
    assert!(!analysis.admits_single_join_post_root_pipeline());
    assert!(!analysis.admits_closed_multi_join_subgraph());
    assert!(!analysis.admits_multi_join_aggregate_over_join_subplan_topk_pipeline());
    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
    assert!(analysis.edges.iter().any(|edge| {
        edge.contract_kinds.contains(&EdgeContractKind::SortInput)
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::FinalSort
                    && requirement.status == ContractStatus::Modeled
                    && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
            })
    }));
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::OperatorOutput
            && requirement.memory_boundary == "final_topk_output"
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::QueryEngineOwnedBoundary
    }));

    let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("root TopK join/aggregate pipeline should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);

    let facts = modeled_facts(&plan, ExecutionBoundaryPolicy::default())
        .expect("root TopK join/aggregate pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(facts.get("bounded_join_execution_family"), None);
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| {
        value.contains("source_read:modeled")
            && value.contains("operator_output:modeled")
            && value.contains("final_sort:modeled")
            && value.contains("materialization:modeled")
    }));
    assert!(facts.get("bounded_join_reservation_requirements").is_some_and(|value| {
        value.contains("final_sort:final_sort_contract:hard_budgeted_report_with_unresolved_preflight")
            && value.contains("operator_output:query_engine_final_topk_output_boundary:query_engine_owned_boundary")
    }));
    assert_modeled_reservation_facts_are_consistent(&facts, "root_topk_join_aggregate");
}

#[test]
fn bounded_root_topk_join_aggregate_rejects_full_scan_without_source_reduction_proof() {
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::MissingSourceReduction);
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_root_topk_join_aggregate_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.root_topk_join_aggregate_admission else {
        panic!("missing source reduction should reject root TopK join/aggregate admission");
    };
    assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof");
    assert!(facts.contains_key("aggregate_id"));
    let projection = analysis
        .first_rejection_projection()
        .expect("missing source reduction should project a local root TopK rejection");
    assert_eq!(projection.reason, "root_topk_join_aggregate_requires_source_reduction_proof");
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_some());
}

#[test]
fn bounded_root_topk_join_aggregate_rejects_reduction_on_unrelated_branch() {
    let plan =
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput);
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_root_topk_join_aggregate_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.root_topk_join_aggregate_admission else {
        panic!("wrong-branch source reduction should reject root TopK join/aggregate admission");
    };
    assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof");
    assert!(facts.contains_key("aggregate_input_node_count"));
    let projection = analysis
        .first_rejection_projection()
        .expect("wrong-branch source reduction should project a local root TopK rejection");
    assert_eq!(projection.reason, "root_topk_join_aggregate_requires_source_reduction_proof");
}

#[test]
fn bounded_root_topk_join_aggregate_allows_shared_source_reuse() {
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SharedSourceFanout);
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "source reuse is a modeled source-read contract, not computed producer fanout"
    );
    let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("shared-source root TopK join/aggregate pipeline should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
}

#[test]
fn bounded_root_topk_join_aggregate_allows_shared_computed_fanout_with_explicit_subplan_proof() {
    for (variant, expected_shared_operator) in [
        (RootTopKJoinAggregateVariant::SharedComputedFanout, "join"),
        (RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout, "aggregate"),
        (RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout, "projection"),
        (RootTopKJoinAggregateVariant::SmallExactSharedProjectionAggregateFanout, "projection"),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let has_expected_shared_producer = analysis.invariants.iter().any(|invariant| {
            if !matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
                || invariant.decision != CapabilityDecision::Satisfied
            {
                return false;
            }
            let Some(shared_producer) = invariant.nodes.first().copied() else {
                return false;
            };
            plan.node(shared_producer).is_some_and(|node| {
                matches!(
                    (&node.kind, expected_shared_operator),
                    (PlanNode::Aggregate(_), "aggregate")
                        | (PlanNode::Join(_), "join")
                        | (PlanNode::Projection(_), "projection")
                )
            })
        });
        assert!(has_expected_shared_producer, "{variant:?}: expected shared {expected_shared_operator} producer proof");

        let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK join/aggregate pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
    }
}

#[test]
fn root_topk_source_reduction_uses_plan_captured_join_strategy_threshold() {
    let strategy_options = KeyedHashJoinStrategyOptions::default().with_small_one_shot_probe_bytes_threshold(2048);
    let optimized = nexus_query_engine::optimizer::optimize_query_plan(
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SmallExactSharedProjectionAggregateFanout),
        &NativeOptimizerOptions { keyed_hash_join_strategy: strategy_options, ..NativeOptimizerOptions::disabled() },
    );
    let plan = optimized.plan;
    let analysis = plan.capability_analysis();

    assert_eq!(plan.keyed_hash_join_strategy_options(), strategy_options);
    assert!(!analysis.admits_root_topk_join_aggregate_pipeline());
    let CapabilityDecision::Rejected { reason, .. } = &analysis.root_topk_join_aggregate_admission else {
        panic!("root TopK admission should reject exact sources above the captured one-shot threshold");
    };
    assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof");
}

#[test]
fn bounded_root_topk_join_aggregate_allows_distinct_and_keyless_subcontracts_with_report_rows() {
    for (variant, expected_contract) in [
        (RootTopKJoinAggregateVariant::DistinctAggregate, "root_topk_distinct_aggregate_subcontract"),
        (
            RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin,
            "root_topk_distinct_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin,
            "root_topk_distinct_aggregate_subcontract",
        ),
        (RootTopKJoinAggregateVariant::KeylessJoin, "root_topk_keyless_or_cross_join_subcontract"),
        (RootTopKJoinAggregateVariant::ProjectedCrossJoin, "root_topk_keyless_or_cross_join_subcontract"),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(
            analysis.admits_root_topk_join_aggregate_pipeline(),
            "{variant:?} should be admitted only through an explicit distinct/keyless subcontract"
        );
        let report = CapabilityContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("{variant:?} capability report should be valid: {error}"));
        assert_admission_report_has_modeled_subcontract(&report, expected_contract, &format!("{variant:?}"));

        let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK join/aggregate pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
        let compiled_report =
            CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
                .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_admission_report_has_modeled_subcontract(&compiled_report, expected_contract, &format!("{variant:?}"));
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_direct_grouped_sources_feeding_keyless_conditionals() {
    let plan = root_topk_keyless_conditional_direct_source_aggregates_plan(false);
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    let report = CapabilityContractReport::from_capability_analysis(&analysis)
        .expect("direct grouped source root TopK capability report should be valid");
    assert_admission_report_has_modeled_subcontract(
        &report,
        "root_topk_keyless_or_cross_join_subcontract",
        "direct grouped source root TopK",
    );
    assert!(nexus_query_engine::admission::bounded_join_modeled_facts(
        &plan,
        ExecutionBoundaryPolicy::default(),
    )
    .is_some());
}

#[test]
fn bounded_root_topk_join_aggregate_checks_each_direct_grouped_keyless_branch() {
    let plan = root_topk_keyless_conditional_direct_source_aggregates_plan(true);
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_root_topk_join_aggregate_pipeline());
    let CapabilityDecision::Rejected { reason, .. } = &analysis.root_topk_join_aggregate_admission else {
        panic!("a direct grouped keyless branch without source reduction should reject");
    };
    assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof");
}

#[test]
fn bounded_root_topk_join_aggregate_allows_sort_fetch_hash_join_input_subcontract_with_report_rows() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchHashJoinInput,
        RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(
            analysis.admits_root_topk_join_aggregate_pipeline(),
            "{variant:?} should be admitted only through an explicit sort-fetch hash-join-input subcontract"
        );
        let report = CapabilityContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("{variant:?} capability report should be valid: {error}"));
        assert_admission_report_has_modeled_subcontract(
            &report,
            "root_topk_sort_fetch_hash_join_input_subcontract",
            &format!("{variant:?}"),
        );

        let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
        let compiled_report =
            CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
                .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_admission_report_has_modeled_subcontract(
            &compiled_report,
            "root_topk_sort_fetch_hash_join_input_subcontract",
            &format!("{variant:?}"),
        );
    }
}

#[test]
fn bounded_root_topk_allows_one_left_join_after_grouped_aggregate_topk() {
    let plan = root_topk_grouped_aggregate_input_left_join_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "decision={:?}",
        analysis.root_topk_join_aggregate_admission
    );
    let report = CapabilityContractReport::from_capability_analysis(&analysis).unwrap();
    assert_admission_report_has_modeled_subcontract(
        &report,
        "root_topk_sort_fetch_hash_join_input_subcontract",
        "grouped aggregate TopK left join",
    );
    let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("grouped aggregate TopK left join should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
}

#[test]
fn bounded_root_topk_join_aggregate_sort_fetch_join_input_subcontract_excludes_keyless_and_cross() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput,
        RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let report = CapabilityContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("{variant:?} capability report should be valid: {error}"));
        assert_admission_report_has_modeled_subcontract(
            &report,
            "root_topk_keyless_or_cross_join_subcontract",
            &format!("{variant:?}"),
        );
        assert_admission_report_lacks_modeled_subcontract(
            &report,
            "root_topk_sort_fetch_hash_join_input_subcontract",
            &format!("{variant:?}"),
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_sort_fetch_join_aggregate_subcontracts_with_report_rows() {
    for (variant, expected_contract) in [
        (
            RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput, "root_topk_keyed_filtered_semi_join_subcontract"),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(
            analysis.admits_root_topk_join_aggregate_pipeline(),
            "{variant:?} should be admitted only through an explicit sort-fetch join-aggregate subcontract: {:?}",
            analysis.root_topk_join_aggregate_admission
        );
        let report = CapabilityContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("{variant:?} capability report should be valid: {error}"));
        assert_admission_report_has_modeled_subcontract(&report, expected_contract, &format!("{variant:?}"));

        let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
        let compiled_report =
            CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
                .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_admission_report_has_modeled_subcontract(&compiled_report, expected_contract, &format!("{variant:?}"));
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_null_filtered_aggregate_fan_in_subcontract() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild,
    ] {
        let plan = nexus_query_engine::optimizer::optimize_query_plan(
            root_topk_join_aggregate_variant_plan(variant),
            &nexus_query_engine::optimizer::NativeOptimizerOptions::default(),
        )
        .plan;
        let analysis = plan.capability_analysis();

        assert!(
            analysis.admits_root_topk_join_aggregate_pipeline(),
            "null-filtered left-join aggregate fan-in should be admitted after rewriting marker filters to LeftAnti ({variant:?}): {:?}",
            analysis.root_topk_join_aggregate_admission
        );
        let report = CapabilityContractReport::from_capability_analysis(&analysis)
            .expect("null-filtered aggregate fan-in capability report should be valid");
        assert_admission_report_has_modeled_subcontract(
            &report,
            "root_topk_keyed_filtered_semi_join_subcontract",
            "null_filtered_aggregate_fan_in",
        );

        let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .expect("null-filtered aggregate fan-in root TopK pipeline should be admitted");
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
        let compiled_report =
            CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
                .expect("null-filtered aggregate fan-in compiled report should be valid");
        assert_admission_report_has_modeled_subcontract(
            &compiled_report,
            "root_topk_keyed_filtered_semi_join_subcontract",
            "null_filtered_aggregate_fan_in",
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_rejects_null_filtered_aggregate_fan_in_contract_gaps() {
    for (variant, expected_reason) in [
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingFilter,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInWrongBuildSide,
            "unsupported_join_node_capability",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInNonUniqueRight,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingBranchReduction,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(!analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let CapabilityDecision::Rejected { reason, .. } = &analysis.root_topk_join_aggregate_admission else {
            panic!("{variant:?} should reject null-filtered aggregate fan-in admission");
        };
        assert_eq!(*reason, expected_reason, "{variant:?}");
    }
}

#[test]
fn bounded_root_topk_join_aggregate_rejects_optimized_null_filtered_unsafe_marker_shapes() {
    for (variant, expected_reason) in [
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingFilter,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes,
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
        ),
    ] {
        let plan = nexus_query_engine::optimizer::optimize_query_plan(
            root_topk_join_aggregate_variant_plan(variant),
            &nexus_query_engine::optimizer::NativeOptimizerOptions::default(),
        )
        .plan;
        let analysis = plan.capability_analysis();

        assert!(!analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let CapabilityDecision::Rejected { reason, .. } = &analysis.root_topk_join_aggregate_admission else {
            panic!("{variant:?} should reject optimized null-filtered aggregate fan-in admission");
        };
        assert_eq!(*reason, expected_reason, "{variant:?}");
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_window_subcontracts_with_report_rows() {
    for variant in [
        RootTopKJoinAggregateVariant::Window,
        RootTopKJoinAggregateVariant::BranchLocalWindow,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindow,
        RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin,
        RootTopKJoinAggregateVariant::SortedAggregateWindow,
        RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(
            analysis.admits_root_topk_join_aggregate_pipeline(),
            "{variant:?} should be admitted only through an explicit window subcontract: {:?}",
            analysis.root_topk_join_aggregate_admission
        );
        let report = CapabilityContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("{variant:?} capability report should be valid: {error}"));
        assert_admission_report_has_modeled_subcontract(&report, "root_topk_window_subcontract", &format!("{variant:?}"));
        if matches!(variant, RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin) {
            assert_admission_report_has_modeled_subcontract(
                &report,
                "root_topk_window_full_outer_join_subcontract",
                &format!("{variant:?}"),
            );
        }
        if matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossWindow) {
            assert_admission_report_has_modeled_subcontract(
                &report,
                "root_topk_keyless_outer_cross_window_subcontract",
                &format!("{variant:?}"),
            );
            assert!(
                report.gate_rows().iter().any(|row| {
                    row.reservation_kind == ReservationKind::GatherMap
                        && row.required_contract == "join_gather_map_reservation_size_bytes"
                        && row.contract_status == ContractStatus::Modeled
                }),
                "{variant:?}: admission report should expose modeled gather-map contract"
            );
            assert!(
                report.gate_rows().iter().any(|row| {
                    row.reservation_kind == ReservationKind::JoinOutput
                        && row.required_contract
                            == "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes"
                        && row.contract_status == ContractStatus::Modeled
                }),
                "{variant:?}: admission report should expose modeled keyless outer-cross output contract"
            );
        }
        assert!(
            report.gate_rows().iter().any(|row| {
                row.reservation_kind == ReservationKind::OperatorOutput
                    && row.required_contract == "window_contract"
                    && row.contract_status == ContractStatus::Modeled
            }),
            "{variant:?}: admission report should expose modeled window output contract"
        );
        if matches!(variant, RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection) {
            assert!(
                report.gate_rows().iter().any(|row| {
                    row.reservation_kind == ReservationKind::OperatorOutput
                        && row.required_contract == "query_engine_rank_filter_output_boundary"
                        && row.contract_status == ContractStatus::Modeled
                }),
                "{variant:?}: admission report should expose modeled rank/filter output boundary"
            );
        }
        assert!(
            report.gate_rows().iter().any(|row| {
                row.reservation_kind == ReservationKind::OperatorOutput
                    && row.required_contract == "query_engine_final_topk_output_boundary"
                    && row.contract_status == ContractStatus::Modeled
            }),
            "{variant:?}: admission report should expose final TopK output boundary"
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_shape_miss_does_not_mask_general_rejection() {
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::MissingGroupedAggregateFragment);
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_root_topk_join_aggregate_pipeline());
    let CapabilityDecision::Rejected { reason, .. } = &analysis.root_topk_join_aggregate_admission else {
        panic!("missing aggregate fragment should reject root TopK join/aggregate admission");
    };
    assert_eq!(*reason, "root_topk_join_aggregate_requires_grouped_aggregate_fragment");
    if let Some(projection) = analysis.first_rejection_projection() {
        assert_ne!(projection.reason, "root_topk_join_aggregate_requires_grouped_aggregate_fragment");
    }
    let admission = nexus_query_engine::admission::bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail");
    if let Some(admission) = admission {
        assert_ne!(
            admission.envelope(),
            nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate,
            "generic root TopK shape miss must not select the canonical root TopK envelope"
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_admission_stays_narrow() {
    for variant in [
        RootTopKJoinAggregateVariant::MissingFetchLimit,
        RootTopKJoinAggregateVariant::FilteredDistinctAggregate,
        RootTopKJoinAggregateVariant::DistinctAggregateUnrelatedKeyedOuterJoin,
        RootTopKJoinAggregateVariant::DistinctAggregateTransitiveKeyedOuterJoin,
        RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate,
        RootTopKJoinAggregateVariant::KeylessOuterCrossJoin,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer,
        RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow,
        RootTopKJoinAggregateVariant::WindowFullOuterLeftOnlyProjection,
        RootTopKJoinAggregateVariant::UnsortedRankWindow,
        RootTopKJoinAggregateVariant::SortedGlobalAggregateWindow,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan,
        RootTopKJoinAggregateVariant::SortFetchLeftSemiWrongBuildSideJoinInput,
        RootTopKJoinAggregateVariant::SortFetchLeftSemiRightOutputJoinInput,
        RootTopKJoinAggregateVariant::SortFetchLeftAntiJoinInput,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(!analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let CapabilityDecision::Rejected { reason, .. } = &analysis.root_topk_join_aggregate_admission else {
            panic!("{variant:?} should reject root TopK join/aggregate admission");
        };
        match variant {
            RootTopKJoinAggregateVariant::MissingFetchLimit => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_fetch_limit")
            },
            RootTopKJoinAggregateVariant::FilteredDistinctAggregate => {
                assert_eq!(*reason, "root_topk_join_aggregate_rejects_unsupported_distinct_state")
            },
            RootTopKJoinAggregateVariant::DistinctAggregateUnrelatedKeyedOuterJoin => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join")
            },
            RootTopKJoinAggregateVariant::DistinctAggregateTransitiveKeyedOuterJoin => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join")
            },
            RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof")
            },
            RootTopKJoinAggregateVariant::KeylessOuterCrossJoin => {
                assert_eq!(*reason, "root_topk_join_aggregate_rejects_keyless_outer_cross_subcontract")
            },
            RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof")
            },
            RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer => {
                assert_eq!(*reason, "root_topk_join_aggregate_rejects_keyless_outer_cross_subcontract")
            },
            RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow => {
                assert_eq!(*reason, "root_topk_join_aggregate_rejects_keyless_outer_cross_subcontract")
            },
            RootTopKJoinAggregateVariant::WindowFullOuterLeftOnlyProjection => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join")
            },
            RootTopKJoinAggregateVariant::UnsortedRankWindow => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_ordered_window_input")
            },
            RootTopKJoinAggregateVariant::SortedGlobalAggregateWindow => {
                assert_eq!(*reason, "unsupported_window_node_capability")
            },
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof")
            },
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof")
            },
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_source_reduction_proof")
            },
            RootTopKJoinAggregateVariant::SortFetchLeftSemiWrongBuildSideJoinInput => {
                assert_eq!(*reason, "unsupported_join_node_capability")
            },
            RootTopKJoinAggregateVariant::SortFetchLeftSemiRightOutputJoinInput => {
                assert_eq!(*reason, "unsupported_join_node_capability")
            },
            RootTopKJoinAggregateVariant::SortFetchLeftAntiJoinInput => {
                assert_eq!(*reason, "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join")
            },
            RootTopKJoinAggregateVariant::Admitted
            | RootTopKJoinAggregateVariant::MissingSourceReduction
            | RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput
            | RootTopKJoinAggregateVariant::SharedSourceFanout
            | RootTopKJoinAggregateVariant::SharedComputedFanout
            | RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout
            | RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout
            | RootTopKJoinAggregateVariant::SmallExactSharedProjectionAggregateFanout
            | RootTopKJoinAggregateVariant::MissingGroupedAggregateFragment
            | RootTopKJoinAggregateVariant::DistinctAggregate
            | RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin
            | RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin
            | RootTopKJoinAggregateVariant::KeylessJoin
            | RootTopKJoinAggregateVariant::ProjectedCrossJoin
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
            | RootTopKJoinAggregateVariant::BranchLocalWindow
            | RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin
            | RootTopKJoinAggregateVariant::SortedAggregateWindow
            | RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection
            | RootTopKJoinAggregateVariant::SortFetchHashJoinInput
            | RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput
            | RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput
            | RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters
            | RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingFilter
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInWrongBuildSide
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInNonUniqueRight
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingBranchReduction
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild
            | RootTopKJoinAggregateVariant::Window => {
                unreachable!("covered by other tests")
            },
        }
    }
}
