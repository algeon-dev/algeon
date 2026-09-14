#[test]
fn bounded_root_topk_join_aggregate_full_scan_below_grouped_aggregate_is_not_a_support_gap() {
    // Capacity is the allocator's: planning proves executability from node, edge,
    // join-capability, and subcontract facts, and the admitted grant's allocator
    // enforces the service-configured cap at runtime, reporting a typed
    // `query_memory_limit_exceeded` failure at its source. A missing, late, or
    // skipped dynamic filter must never change admission or results, so these
    // fixtures, which differ from admitted shapes only in scan-reduction evidence
    // (full scans or large exact sizes below the grouped aggregate), stay admitted.
    for variant in [
        RootTopKJoinAggregateVariant::MissingSourceReduction,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(
            analysis.admits_root_topk_join_aggregate_pipeline(),
            "{variant:?}: a full scan is a capacity question, not a support gap: {:?}",
            analysis.root_topk_join_aggregate_admission
        );
        let admission = bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
        assert!(unsupported_facts(&plan).is_none(), "{variant:?}");
    }
}

#[test]
fn bounded_root_topk_join_aggregate_reduction_on_unrelated_branch_does_not_gate_admission() {
    // Capacity is the allocator's, enforced as a typed cap failure at runtime.
    // Which branch a dynamic filter happens to reduce (or whether one runs at
    // all) must never change admission or results.
    let plan =
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput);
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "decision={:?}",
        analysis.root_topk_join_aggregate_admission
    );
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("unrelated-branch reduction root TopK pipeline should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
}

#[test]
fn bounded_root_topk_join_aggregate_allows_shared_source_reuse() {
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SharedSourceFanout);
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "source reuse is a modeled source-read contract, not computed producer fanout"
    );
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("shared-source root TopK join/aggregate pipeline should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
}

#[test]
fn bounded_root_topk_join_aggregate_allows_shared_computed_fanout_with_explicit_subplan_evidence() {
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
        assert!(has_expected_shared_producer, "{variant:?}: expected shared {expected_shared_operator} producer");

        let admission = bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK join/aggregate pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_distinct_and_keyless_subcontracts() {
    for variant in [
        RootTopKJoinAggregateVariant::DistinctAggregate,
        RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin,
        RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin,
        RootTopKJoinAggregateVariant::KeylessJoin,
        // An unrelated scalar-aggregate branch (over an unreduced fact scan) does
        // not break the keyless subcontract: capacity is the allocator's, enforced
        // as a typed cap failure at runtime, so a missing, late, or skipped
        // dynamic filter must never change admission or results.
        RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate,
        RootTopKJoinAggregateVariant::ProjectedCrossJoin,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(
            analysis.admits_root_topk_join_aggregate_pipeline(),
            "{variant:?} should be admitted only through an explicit distinct/keyless subcontract"
        );
        let admission = bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK join/aggregate pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_direct_grouped_sources_feeding_keyless_conditionals() {
    let plan = root_topk_keyless_conditional_direct_source_aggregates_plan(false);
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    assert!(nexus_query_engine::admission::bounded_join_modeled_facts(&plan).is_some());
}

#[test]
fn bounded_root_topk_join_aggregate_admits_unreduced_direct_grouped_keyless_branches() {
    // Capacity is the allocator's, enforced as a typed cap failure at runtime; a
    // missing, late, or skipped dynamic filter on any direct grouped branch must
    // never change admission or results, so a branch scanning its source in full
    // is not a support gap.
    let plan = root_topk_keyless_conditional_direct_source_aggregates_plan(true);
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "decision={:?}",
        analysis.root_topk_join_aggregate_admission
    );
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("unreduced direct grouped keyless branches should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
}

#[test]
fn bounded_root_topk_join_aggregate_allows_sort_fetch_hash_join_input_subcontract() {
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
        let admission = bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_keyed_left_semi_join_carrying_a_residual_predicate() {
    let plan =
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SortFetchLeftSemiResidualJoinInput);
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "a keyed left-semi join with a residual predicate emits left-only output and is executable on the \
         mixed keyed/residual path, so the top-k envelope must admit it: {:?}",
        analysis.root_topk_join_aggregate_admission
    );
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("keyed left-semi residual root TopK pipeline should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
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
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("grouped aggregate TopK left join should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
}

#[test]
fn bounded_root_topk_join_aggregate_allows_whole_partition_window_filter_before_final_sort() {
    // Pins change B: a whole-partition aggregate window sitting directly between the
    // grouped-aggregate fragment and the final top-k sort/limit, with no projection
    // reinserted before the final sort, is the same composition already admitted
    // whenever an ORDER BY expression happens to force a projection into that chain
    // (q47/q57/q89). q53/q63 hit this path because their ORDER BY uses plain columns.
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::AggregateWindowFilterTopK);
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "decision={:?}",
        analysis.root_topk_join_aggregate_admission
    );
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("whole-partition window filter root TopK pipeline should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
}

#[test]
fn bounded_root_topk_join_aggregate_allows_cumulative_window_full_outer_before_final_sort() {
    // Pins change B: q51's two cumulative-window branches, combined by a FULL OUTER
    // join and followed by a second cumulative window/filter with no projection
    // before the final sort, is admitted through the same window candidacy path.
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::CumulativeWindowFullOuterTopK);
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "decision={:?}",
        analysis.root_topk_join_aggregate_admission
    );
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("cumulative window full-outer root TopK pipeline should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
}

#[test]
fn bounded_root_topk_join_aggregate_window_missing_input_sort_keeps_local_diagnosis() {
    // Pins that a window near-miss inside the new candidacy path keeps its own
    // `requires_ordered_window_input` diagnosis instead of falling back to the
    // generic root-join rejection: without change B this composition is masked by
    // `requires_projection_before_sort`, which is not in the admission-owned
    // projection allowlist, so the projection owner would fall back to a different
    // admission gate entirely. The fixture's join fragment is keyless conditional
    // so no generic envelope admits the whole plan and the projection is observable.
    let plan = root_topk_join_aggregate_variant_plan(
        RootTopKJoinAggregateVariant::AggregateWindowFilterTopKMissingInputSort,
    );
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_root_topk_join_aggregate_pipeline());
    let CapabilityDecision::Rejected { reason, .. } = &analysis.root_topk_join_aggregate_admission else {
        panic!("an ordered window fed directly from the aggregate should reject root TopK join/aggregate admission");
    };
    assert_eq!(*reason, "root_topk_join_aggregate_requires_ordered_window_input");
    let projection = analysis
        .first_rejection_projection()
        .expect("missing window input sort should project a local root TopK rejection");
    assert_eq!(projection.reason, "root_topk_join_aggregate_requires_ordered_window_input");
    assert_eq!(
        projection.owner,
        nexus_query_engine::capability::CapabilityRejectionOwner::Admission("root_topk_join_aggregate_admission"),
        "the window near-miss must keep its own admission-owned diagnosis, not a generic fallback"
    );
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
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_sort_fetch_join_aggregate_subcontracts() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters,
        RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(
            analysis.admits_root_topk_join_aggregate_pipeline(),
            "{variant:?} should be admitted only through an explicit sort-fetch join-aggregate subcontract: {:?}",
            analysis.root_topk_join_aggregate_admission
        );
        let admission = bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .unwrap_or_else(|| panic!("{variant:?} root TopK pipeline should be admitted"));
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_null_filtered_aggregate_fan_in_subcontract() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInIdentityArithmeticProjection,
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
        let admission = bounded_pipeline_admission(&plan)
            .expect("bounded admission should not fail")
            .expect("null-filtered aggregate fan-in root TopK pipeline should be admitted");
        assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::RootTopKJoinAggregate);
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
            RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInUniqueNonAggregateRight,
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
    let admission =
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail");
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
        RootTopKJoinAggregateVariant::KeylessOuterCrossJoin,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer,
        RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow,
        RootTopKJoinAggregateVariant::WindowFullOuterLeftOnlyProjection,
        RootTopKJoinAggregateVariant::UnsortedRankWindow,
        RootTopKJoinAggregateVariant::SortedGlobalAggregateWindow,
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
            RootTopKJoinAggregateVariant::KeylessOuterCrossJoin => {
                assert_eq!(*reason, "root_topk_join_aggregate_rejects_keyless_outer_cross_subcontract")
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
            | RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan
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
            | RootTopKJoinAggregateVariant::AggregateWindowFilterTopK
            | RootTopKJoinAggregateVariant::AggregateWindowFilterTopKMissingInputSort
            | RootTopKJoinAggregateVariant::CumulativeWindowFullOuterTopK
            | RootTopKJoinAggregateVariant::SortFetchHashJoinInput
            | RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiResidualJoinInput
            | RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput
            | RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters
            | RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInIdentityArithmeticProjection
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingFilter
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInWrongBuildSide
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse
            | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInUniqueNonAggregateRight
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
