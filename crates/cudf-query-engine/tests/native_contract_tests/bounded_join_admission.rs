use super::*;

#[path = "bounded_join_admission/aggregate_over_join_subplans.rs"]
mod aggregate_over_join_subplans;
#[path = "bounded_join_admission/mixed_residual.rs"]
mod mixed_residual;
#[path = "bounded_join_admission/plans.rs"]
pub(crate) mod plans;
#[path = "bounded_join_admission/reservation_contract.rs"]
mod reservation_contract;
#[path = "bounded_join_admission/residual_resident_grouped_aggregate.rs"]
mod residual_resident_grouped_aggregate;
pub(crate) use aggregate_over_join_subplans::{
    AggregateOverJoinSubplanInputsVariant, keyless_conditional_aggregate_over_join_subplan_inputs_plan,
};
pub(crate) use mixed_residual::admitted_mixed_residual_multi_join_plan;
use plans::*;
pub(crate) use residual_resident_grouped_aggregate::{
    ResidualResidentGroupedAggregateInputVariant, residual_resident_grouped_aggregate_input_plan,
};

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
                matches!(contract, "source_contract" | "aggregate_contract" | "final_sort_contract"),
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

#[test]
fn large_keyed_inner_build_selects_one_shot_strategy_without_hash_table_gate() {
    let plan = large_keyed_join_strategy_plan(JoinType::Inner);
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections());
    let join = analysis
        .nodes
        .iter()
        .find(|node| matches!(node.operator, NodeOperator::Join { .. }))
        .expect("plan should contain a join capability row");
    assert!(
        join.local_requirements.iter().all(|requirement| requirement.kind != ReservationKind::HashTable),
        "large one-shot keyed joins must not require a reusable hash-table gate"
    );

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("large one-shot join should still be admitted");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:one_shot_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason").map(String::as_str), Some("n2:build_side_too_large"));
    assert_eq!(facts.get("estimated_build_rows").map(String::as_str), Some("n2:100000000"));
    assert_eq!(facts.get("probe_reuse_count").map(String::as_str), Some("n2:1"));

    let pipeline = cudf_query_engine::compile_pipeline(&plan, &analysis).expect("one-shot join should compile");
    assert!(
        pipeline.gates().iter().all(|gate| gate.reservation_kind() != ReservationKind::HashTable),
        "compiled one-shot join pipeline must not carry a reusable hash-table gate"
    );
    assert!(
        pipeline.gates().iter().any(|gate| {
            gate.reservation_kind() == ReservationKind::GatherMap
                && gate.memory_boundary() == "join_one_shot_indices"
                && gate.required_contract() == "one_shot_join_indices_observed_retained_bytes"
                && gate.quality() == ContractQuality::ObservedRetainedState
        }),
        "compiled one-shot join pipeline should report observed retained index bytes honestly"
    );
}

#[test]
fn large_keyed_left_and_full_builds_select_one_shot_strategy_without_hash_table_gate() {
    for join_type in [JoinType::Left, JoinType::Full] {
        let plan = large_keyed_join_strategy_plan(join_type);
        let analysis = plan.capability_analysis();

        assert!(!analysis.has_rejections(), "{join_type} large one-shot join should be admitted");
        let join = analysis
            .nodes
            .iter()
            .find(|node| matches!(node.operator, NodeOperator::Join { .. }))
            .expect("plan should contain a join capability row");
        assert!(
            join.local_requirements.iter().all(|requirement| requirement.kind != ReservationKind::HashTable),
            "{join_type} large one-shot keyed joins must not require a reusable hash-table gate"
        );

        let facts = plan
            .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
            .expect("large one-shot join should report modeled bounded facts");
        assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:one_shot_equi"));
        assert_eq!(
            facts.get("reusable_hash_join_rejected_reason").map(String::as_str),
            Some("n2:build_side_too_large")
        );
        assert_eq!(facts.get("one_shot_equi_rejected_reason"), None);

        let pipeline = cudf_query_engine::compile_pipeline(&plan, &analysis).expect("one-shot join should compile");
        assert!(
            pipeline.gates().iter().all(|gate| gate.reservation_kind() != ReservationKind::HashTable),
            "{join_type} compiled one-shot join pipeline must not carry a reusable hash-table gate"
        );
    }
}

#[test]
fn large_keyed_inner_residual_build_selects_one_shot_strategy_without_hash_or_mixed_size_gate() {
    let plan = large_keyed_residual_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "large keyed inner residual one-shot join should be admitted");
    let join = analysis
        .nodes
        .iter()
        .find(|node| matches!(node.operator, NodeOperator::Join { .. }))
        .expect("plan should contain a join capability row");
    assert!(
        join.local_requirements.iter().all(|requirement| !matches!(
            requirement.kind,
            ReservationKind::HashTable | ReservationKind::MixedSizeData
        )),
        "large one-shot keyed residual joins must not require reusable hash-table or mixed size-data gates"
    );

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("large one-shot residual join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:one_shot_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason").map(String::as_str), Some("n2:build_side_too_large"));
    assert_eq!(facts.get("one_shot_equi_rejected_reason"), None);

    let pipeline =
        cudf_query_engine::compile_pipeline(&plan, &analysis).expect("one-shot residual join should compile");
    assert!(
        pipeline.gates().iter().all(|gate| {
            !matches!(gate.reservation_kind(), ReservationKind::HashTable | ReservationKind::MixedSizeData)
        }),
        "compiled one-shot residual join pipeline must not carry reusable hash-table or mixed size-data gates"
    );
    assert!(
        pipeline.gates().iter().any(|gate| {
            gate.reservation_kind() == ReservationKind::GatherMap
                && gate.memory_boundary() == "join_one_shot_indices"
                && gate.required_contract() == "one_shot_join_indices_observed_retained_bytes"
                && gate.quality() == ContractQuality::ObservedRetainedState
        }),
        "compiled one-shot residual join pipeline should report observed retained index bytes honestly"
    );
}

fn large_keyed_join_strategy_plan(join_type: JoinType) -> cudf_query_engine::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/large-join-left.parquet").with_row_group_rows(vec![10])],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/large-join-right.parquet").with_row_group_rows(vec![100_000_000])],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            join_type,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn large_keyed_residual_join_strategy_plan() -> cudf_query_engine::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/large-residual-join-left.parquet").with_row_group_rows(vec![10])],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/large-residual-join-right.parquet").with_row_group_rows(vec![100_000_000])],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "qty", "qty"),
            BinaryOp::Gt,
            Expr::input_column(1, "order_id", "right_order_id"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
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

#[test]
fn bounded_capability_analysis_runs_as_side_channel_for_multi_join_pipeline() {
    let plan = serial_aggregate_multi_join_spine_plan();

    assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());
    let analysis = plan.capability_analysis();

    assert_eq!(analysis.nodes.len(), plan.node_count());
    assert!(analysis.nodes.iter().any(|node| matches!(node.operator, NodeOperator::Join { .. })));
    assert!(analysis.edges.iter().any(|edge| matches!(edge.input_role, InputRole::JoinBuildInput { .. })));
    assert!(
        analysis
            .invariants
            .iter()
            .any(|invariant| matches!(invariant.kind, SubgraphInvariantKind::BuildBeforeProbe { .. }))
    );
    assert!(
        analysis
            .invariants
            .iter()
            .filter(|invariant| matches!(invariant.kind, SubgraphInvariantKind::BuildBeforeProbe { .. }))
            .all(|invariant| invariant.decision == CapabilityDecision::Satisfied)
    );
    assert!(analysis.edges.iter().any(|edge| {
        matches!(edge.consumer, EdgeConsumer::Sink(ResultSinkKind::HostArrow))
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::Materialization
                    && requirement.status == ContractStatus::Modeled
                    && requirement.quality == ContractQuality::QueryEngineOwnedBoundary
            })
    }));
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::SourceRead
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    }));
}

#[test]
fn bounded_capability_analysis_extracts_node_capabilities_for_root_join() {
    let plan = simple_root_join_plan();

    assert!(plan.capability_analysis().admits_root_join());
    let analysis = plan.capability_analysis();
    assert!(!analysis.has_rejections());
    assert_eq!(analysis.nodes.len(), 3);
    assert!(
        analysis
            .nodes
            .iter()
            .filter(|node| {
                matches!(node.operator, NodeOperator::Source { format: SourceFormatCapability::LocalParquet })
                    && node.executable == CapabilityDecision::Satisfied
            })
            .count()
            == 2
    );
    assert!(analysis.nodes.iter().any(|node| {
        matches!(
            node.operator,
            NodeOperator::Join {
                join_type: JoinType::Inner,
                key_count: 1,
                residual: ResidualRequirement::None,
                build_side: JoinSide::Right,
            }
        ) && node.executable == CapabilityDecision::Satisfied
    }));
}

#[test]
fn bounded_capability_analysis_extracts_node_capabilities_for_single_join_post_root_pipeline() {
    let plan = serial_aggregate_single_join_spine_plan();

    assert!(plan.capability_analysis().admits_single_join_post_root_pipeline());
    let analysis = plan.capability_analysis();
    assert!(!analysis.has_rejections());
    assert!(analysis.nodes.iter().any(|node| {
        matches!(
            node.operator,
            NodeOperator::Aggregate { grouping: cudf_query_engine::capability::AggregateGrouping::Grouped, .. }
        ) && node.executable == CapabilityDecision::Satisfied
    }));
    assert!(analysis.nodes.iter().any(|node| {
        matches!(node.operator, NodeOperator::Sort { ordering: OrderingRequirement::Required })
            && node.executable == CapabilityDecision::Satisfied
    }));
    assert!(
        analysis
            .nodes
            .iter()
            .filter(|node| matches!(node.operator, NodeOperator::Projection))
            .all(|node| node.executable == CapabilityDecision::Satisfied)
    );
}

#[test]
fn bounded_root_and_single_join_admission_uses_capability_proofs_without_path_projection() {
    let root = simple_root_join_plan();
    let root_analysis = root.capability_analysis();
    assert!(root.capability_analysis().admits_root_join());
    assert!(root_analysis.admits_root_join());
    assert!(root_analysis.admits_strict_bounded_join());
    assert!(!root_analysis.admits_closed_multi_join_subgraph());
    assert!(!root_analysis.admits_keyless_conditional_join_input_subplan_pipeline());

    let single = serial_aggregate_single_join_spine_plan();
    let single_analysis = single.capability_analysis();
    assert!(single.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(single_analysis.admits_single_join_post_root_pipeline());
    assert!(single_analysis.admits_strict_bounded_join());
    assert!(!single_analysis.admits_closed_multi_join_subgraph());
    assert!(!single_analysis.admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline());
}

#[test]
fn bounded_capability_analysis_preserves_status_quality_separation() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);

    assert!(plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline());
    let analysis = plan.capability_analysis();
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::FinalSort
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    }));
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::OperatorOutput
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    }));
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::HashTable
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::ExactPreflight
    }));
}

#[test]
fn bounded_capability_analysis_classifies_phase2_reservation_scopes() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let analysis = plan.capability_analysis();
    assert!(!analysis.requirements.is_empty());
    for requirement in &analysis.requirements {
        assert_eq!(
            requirement.scope,
            cudf_query_engine::capability::reservation_requirement_scope(requirement.kind),
            "capability requirement {:?} must carry its canonical scope",
            requirement.kind,
        );
    }
}

#[test]
fn bounded_capability_analysis_decomposes_probe_side_into_edge_and_build_before_probe_invariant() {
    let plan = serial_aggregate_multi_join_spine_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.edges.iter().any(|edge| {
        matches!(edge.input_role, InputRole::JoinProbeInput { .. })
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::ProbeSide
                    && requirement.scope == ReservationRequirementScope::Edge
                    && requirement.status == ContractStatus::Modeled
                    && requirement.quality == ContractQuality::QueryEngineOwnedOrdering
            })
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::BuildBeforeProbe { .. })
            && invariant.required_schedule == ScheduleConstraint::BuildBeforeProbe
    }));
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::ProbeSide
            && requirement.required_contract == "query_engine_probe_side_source_reservation"
            && requirement.quality == ContractQuality::QueryEngineOwnedOrdering
    }));
}

#[test]
fn bounded_capability_analysis_extracts_phase4_multi_join_edge_contracts() {
    let plan = serial_aggregate_multi_join_spine_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.edges.iter().all(|edge| edge.decision == CapabilityDecision::Satisfied));
    assert!(
        analysis.edges.iter().filter(|edge| matches!(edge.input_role, InputRole::JoinBuildInput { .. })).count() >= 2,
        "multi-join pipeline should expose build input edges for each join"
    );
    assert!(
        analysis.edges.iter().filter(|edge| matches!(edge.input_role, InputRole::JoinProbeInput { .. })).count() >= 2,
        "multi-join pipeline should expose probe input edges for each join"
    );
    assert!(analysis.edges.iter().any(|edge| {
        edge.contract_kinds.iter().any(|kind| matches!(kind, EdgeContractKind::ResidentSourceInput { .. }))
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::SourceRead
                    && requirement.scope == ReservationRequirementScope::Edge
                    && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
            })
    }));
    assert!(analysis.edges.iter().any(|edge| {
        edge.contract_kinds.contains(&EdgeContractKind::JoinMaterialization)
            && edge.reservation_requirements.iter().any(|requirement| requirement.kind == ReservationKind::GatherMap)
            && edge.reservation_requirements.iter().any(|requirement| requirement.kind == ReservationKind::JoinOutput)
    }));
    assert!(analysis.edges.iter().any(|edge| {
        edge.contract_kinds.contains(&EdgeContractKind::SortInput)
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::FinalSort
                    && requirement.scope == ReservationRequirementScope::Edge
                    && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
            })
    }));
    assert!(analysis.edges.iter().any(|edge| {
        matches!(edge.consumer, EdgeConsumer::Sink(ResultSinkKind::HostArrow))
            && edge.contract_kinds.contains(&EdgeContractKind::ExplicitSinkMaterialization)
            && edge.required_properties.contains(&RequiredProperty::ResidentFrame)
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::Materialization
                    && requirement.scope == ReservationRequirementScope::Edge
                    && requirement.quality == ContractQuality::QueryEngineOwnedBoundary
            })
    }));
}

#[test]
fn bounded_capability_analysis_solves_phase5_multi_join_invariants() {
    let plan = serial_aggregate_multi_join_spine_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_closed_multi_join_subgraph());
    assert_eq!(analysis.closed_multi_join_subgraph_admission, CapabilityDecision::Satisfied);
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::ClosedJoinSubtree { .. })
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::TopologicalBuildProbeSchedule)
            && invariant.required_schedule == ScheduleConstraint::Topological
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::ResidentLifetimeDominatesConsumers)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(
        analysis
            .invariants
            .iter()
            .filter(|invariant| matches!(invariant.kind, SubgraphInvariantKind::BuildBeforeProbe { .. }))
            .all(|invariant| invariant.decision == CapabilityDecision::Satisfied)
    );
}

#[test]
fn bounded_multi_join_phase6_admission_uses_closed_subgraph_capability_proof() {
    let plan = serial_aggregate_multi_join_spine_plan();
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_closed_multi_join_subgraph(),
        analysis.admits_closed_multi_join_subgraph()
    );
}

#[test]
fn bounded_capability_analysis_rejects_phase5_multi_join_admission_without_family_recognizer() {
    let plan = multi_join_candidate_with_aggregate_inside_join_subtree_plan();

    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_closed_multi_join_subgraph());
    assert_eq!(
        plan.capability_analysis().admits_closed_multi_join_subgraph(),
        analysis.admits_closed_multi_join_subgraph()
    );
    assert!(!analysis.has_rejections(), "the local node and edge capability layers still accept this plan");
    let CapabilityDecision::Rejected { reason, facts } = &analysis.closed_multi_join_subgraph_admission else {
        panic!("aggregate inside the join subtree should reject the closed multi-join subgraph proof");
    };
    assert_eq!(*reason, "closed_multi_join_subgraph_unsupported_node");
    assert_eq!(facts.get("operator").map(String::as_str), Some("aggregate"));
}

#[test]
fn bounded_capability_analysis_projects_reservation_facts_without_family_path_vocabulary() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("keyed hash join resident subplan pipeline should produce modeled facts");

    assert!(facts.get("bounded_join_reservation_kinds").is_some_and(|value| value.contains("source_read")));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| value.contains("source_read:modeled")));
    assert!(facts.get("bounded_join_reservation_requirements").is_some_and(|value| {
        value.contains("source_read:source_contract:hard_budgeted_report_with_unresolved_preflight")
            && !value.contains("BoundedJoinExecutionPath")
            && !value.contains("bounded_join_execution_path")
    }));
    assert!(facts.get("modeled_reservations").is_some_and(|value| value.contains("source_read")));
}

#[test]
fn bounded_capability_analysis_projects_modeled_report_facts_for_simple_root_join() {
    let plan = simple_root_join_plan();

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("root join should produce modeled bounded facts through capability projection");

    assert_eq!(facts.get("bounded_join_execution_family"), None);
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
    assert_modeled_reservation_facts_are_consistent(&facts, "root_join");
}

#[test]
fn bounded_capability_analysis_prevents_modeled_facts_for_capability_rejections() {
    let plan = single_join_pipeline_with_empty_in_list_filter_plan();

    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(plan.bounded_join_modeled_facts(ExecutionBoundaryPolicy::default()).is_none());
    let facts = plan
        .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
        .expect("capability-rejected single-join pipeline should produce rejection facts");

    assert_rejection_facts_are_family_free(&facts, "empty_in_list_filter");
    assert_eq!(facts.get("bounded_join_rejection_scope").map(String::as_str), Some("node"));
    assert_eq!(facts.get("bounded_join_rejection_category").map(String::as_str), Some("unsupported_expression_family"));
    assert_eq!(
        facts.get("bounded_join_rejection_reason").map(String::as_str),
        Some("unsupported_filter_expression_capability")
    );
    assert!(
        facts
            .get("bounded_join_rejection_facts")
            .is_some_and(|value| value.contains("reason=in_list_must_not_be_empty"))
    );
}

#[test]
fn bounded_capability_analysis_can_reject_node_capabilities_without_family_projection() {
    let plan = unsupported_cross_key_join_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.has_rejections());
    let rejected_join = analysis
        .nodes
        .iter()
        .find(|node| matches!(node.operator, NodeOperator::Join { join_type: JoinType::Cross, .. }))
        .expect("unsupported plan should contain a cross join capability");
    let CapabilityDecision::Rejected { reason, facts } = &rejected_join.executable else {
        panic!("unsupported cross-key join should be rejected as a node capability");
    };
    assert_eq!(*reason, "unsupported_join_node_capability");
    assert_eq!(facts.get("join_type").map(String::as_str), Some("cross"));
    assert_eq!(facts.get("key_count").map(String::as_str), Some("1"));
    assert!(plan.bounded_join_modeled_facts(ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn strict_contract_report_projects_rejected_node_capability_when_reservations_are_modeled() {
    let plan = unsupported_cross_key_join_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.has_rejections());
    assert!(analysis.requirements.iter().all(|requirement| requirement.status == ContractStatus::Modeled));

    let report = StrictBoundedContractReport::from_capability_analysis(&analysis)
        .expect("unsupported capability analysis should produce a strict contract report");
    let row = report
        .gate_rows()
        .iter()
        .find(|row| row.contract_status == ContractStatus::RejectedByCapability)
        .expect("unsupported report should expose a local capability rejection row");

    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::UnsupportedStrictBounded);
    assert_eq!(row.contract_scope, ReservationRequirementScope::Node);
    assert!(matches!(row.owner, StrictBoundedContractOwner::CapabilityNode(_)));
    assert_eq!(row.reservation_kind, ReservationKind::CapabilityContract);
    assert_eq!(row.memory_boundary, "node_capability");
    assert_eq!(row.required_contract, "unsupported_join_node_capability");
    assert_eq!(row.contract_quality, ContractQuality::CapabilityProof);
}

#[test]
fn bounded_capability_analysis_rejects_unsupported_filter_expression_as_node_capability() {
    let plan = single_join_pipeline_with_empty_in_list_filter_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_strict_bounded_join());
    let rejected_filter = analysis
        .nodes
        .iter()
        .find(|node| matches!(node.operator, NodeOperator::Filter))
        .expect("plan should contain a filter node");
    let CapabilityDecision::Rejected { reason, facts } = &rejected_filter.executable else {
        panic!("empty IN-list filter should be rejected as a node capability");
    };
    assert_eq!(*reason, "unsupported_filter_expression_capability");
    assert_eq!(facts.get("reason").map(String::as_str), Some("in_list_must_not_be_empty"));
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("in_list"));
}

#[test]
fn bounded_capability_analysis_rejects_unsupported_join_residual_expression_as_node_capability() {
    let plan = root_join_with_like_residual_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_strict_bounded_join());
    let rejected_join = analysis
        .nodes
        .iter()
        .find(|node| matches!(node.operator, NodeOperator::Join { .. }))
        .expect("plan should contain a join node");
    let CapabilityDecision::Rejected { reason, facts } = &rejected_join.executable else {
        panic!("LIKE residual should be rejected as a node capability");
    };
    assert_eq!(*reason, "unsupported_join_residual_expression_capability");
    assert_eq!(facts.get("reason").map(String::as_str), Some("join_residual_expression_family_not_supported"));
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("like"));
}

#[test]
fn bounded_capability_analysis_rejects_one_sided_residual_as_subgraph_invariant() {
    let plan = root_join_with_one_sided_residual_plan();
    let analysis = plan.capability_analysis();

    let residual_join = analysis
        .nodes
        .iter()
        .find(|node| matches!(node.operator, NodeOperator::Join { residual: ResidualRequirement::Required, .. }))
        .expect("plan should contain a residual join");
    assert_eq!(residual_join.executable, CapabilityDecision::Satisfied);
    let invariant = analysis
        .invariants
        .iter()
        .find(|invariant| matches!(invariant.kind, SubgraphInvariantKind::ResidualUsesBothInputs { .. }))
        .expect("residual join should emit an input-coverage invariant");
    let CapabilityDecision::Rejected { reason, facts } = &invariant.decision else {
        panic!("one-sided residual should fail the residual input coverage invariant");
    };
    assert_eq!(*reason, "residual_does_not_reference_both_inputs");
    assert_eq!(facts.get("references_left").map(String::as_str), Some("true"));
    assert_eq!(facts.get("references_right").map(String::as_str), Some("false"));
}

pub(crate) fn unsupported_cross_key_join_plan() -> cudf_query_engine::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("r_key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("r_key", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/unsupported-cross-left.parquet",
        Arc::clone(&left_schema),
        vec!["l_key".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/unsupported-cross-right.parquet",
        Arc::clone(&right_schema),
        vec!["r_key".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Cross,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
        ]),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn simple_root_join_plan() -> cudf_query_engine::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("r_key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-left.parquet",
        left_schema,
        vec!["l_key".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-right.parquet",
        right_schema,
        vec!["r_key".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    builder.finish_with_host_sink(join)
}

fn single_join_pipeline_with_empty_in_list_filter_plan() -> cudf_query_engine::QueryPlan {
    let base = simple_root_join_plan();
    let root = base.root_id();
    let output_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    for node in base.nodes() {
        match &node.kind {
            PlanNode::Source(source) => {
                builder.add_source((**source).clone());
            },
            PlanNode::Join(join) => {
                builder.add_join(join.inputs[0], join.inputs[1], join.spec.clone(), Arc::clone(&output_schema));
            },
            _ => unreachable!("simple root plan only contains sources and one join"),
        }
    }
    let filter = builder.add_filter(root, Expr::in_list(Expr::column("l_key"), Vec::new()), Arc::clone(&output_schema));
    builder.finish_with_host_sink(filter)
}

fn root_join_with_like_residual_plan() -> cudf_query_engine::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("l_text", DataType::Utf8, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("r_key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-residual-left.parquet",
        left_schema,
        vec!["l_key".into(), "l_text".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-residual-right.parquet",
        right_schema,
        vec!["r_key".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_residual_predicate(Expr::like(Expr::input_column(0, "l_text", "l_text"), "%promo%"))
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key")])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn root_join_with_one_sided_residual_plan() -> cudf_query_engine::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("l_qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("r_key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-one-sided-residual-left.parquet",
        left_schema,
        vec!["l_key".into(), "l_qty".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-one-sided-residual-right.parquet",
        right_schema,
        vec!["r_key".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "l_qty", "l_qty"),
            BinaryOp::Gt,
            Expr::literal(LiteralValue::Int32(0)),
        ))
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key")])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn multi_join_candidate_with_aggregate_inside_join_subtree_plan() -> cudf_query_engine::QueryPlan {
    let partsupp_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
    ]));
    let part_schema = Arc::new(Schema::new(vec![Field::new("p_partkey", DataType::Int32, false)]));
    let supplier_schema = Arc::new(Schema::new(vec![Field::new("s_suppkey", DataType::Int32, false)]));
    let supplier_counts_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int32, false),
        Field::new("supplier_count", DataType::Int64, false),
    ]));
    let part_join_schema = Arc::new(Schema::new(vec![
        Field::new("ps_suppkey", DataType::Int32, false),
        Field::new("ps_partkey", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("supplier_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let partsupp = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/phase5-partsupp.parquet",
        partsupp_schema,
        vec!["ps_partkey".into(), "ps_suppkey".into()],
    )));
    let part = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/phase5-part.parquet",
        part_schema,
        vec!["p_partkey".into()],
    )));
    let supplier = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/phase5-supplier.parquet",
        supplier_schema,
        vec!["s_suppkey".into()],
    )));
    let part_join = builder.add_join(
        partsupp,
        part,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                ColumnRef::input(1, "p_partkey", "p_partkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ps_suppkey", "ps_suppkey"), "ps_suppkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&part_join_schema),
    );
    let supplier_counts = builder.add_aggregate(
        supplier,
        AggregateSpec::new(
            vec![Expr::column("s_suppkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("s_suppkey"), "supplier_count")],
        ),
        Arc::clone(&supplier_counts_schema),
    );
    let root = builder.add_join(
        part_join,
        supplier_counts,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ps_suppkey", "ps_suppkey"),
                ColumnRef::input(1, "s_suppkey", "s_suppkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "supplier_count", "supplier_count"), "supplier_count"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(root)
}

#[test]
fn bounded_single_join_pipeline_admits_serial_aggregate_spine() {
    let plan = serial_aggregate_single_join_spine_plan();

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_multi_join_pipeline_admits_serial_aggregate_spine_with_projection_and_sort() {
    let plan = serial_aggregate_multi_join_spine_plan();

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_single_join_pipeline_rejects_aggregate_join_input_leaves() {
    for aggregate_side in [JoinSide::Left, JoinSide::Right] {
        let plan = single_join_with_aggregate_input_plan(aggregate_side);

        assert!(!plan.capability_analysis().admits_root_join());
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
        assert!(
            plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_some(),
            "aggregate {aggregate_side} join input must not be accepted as a bounded root join leaf"
        );
    }
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_pipeline_is_executable_with_modeled_contracts() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(!plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("keyed hash join resident subplan pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(facts.get("bounded_join_execution_family"), None);
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| {
        value.contains("source_read:modeled")
            && value.contains("operator_output:modeled")
            && value.contains("final_sort:modeled")
    }));
    let reservation_requirements = facts
        .get("bounded_join_reservation_requirements")
        .expect("resident subplan pipeline should report shape-local reservation requirements");
    assert!(
        reservation_requirements.contains("source_read:source_contract:hard_budgeted_report_with_unresolved_preflight")
    );
    assert!(
        reservation_requirements
            .contains("operator_output:aggregate_contract:hard_budgeted_report_with_unresolved_preflight")
    );
    assert!(
        reservation_requirements
            .contains("final_sort:final_sort_contract:hard_budgeted_report_with_unresolved_preflight")
    );
    assert_eq!(facts.get("bounded_join_contract_modes"), None);
    assert_eq!(facts.get("bounded_join_unresolved_preflight_contracts"), None);
    assert_eq!(facts.get("bounded_join_contract_report_fields"), None);
    assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);
    assert_modeled_reservation_facts_are_consistent(&facts, "keyed_hash_join_resident_subplan_pipeline");
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_phase6_admission_uses_capability_proof() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline(),
        analysis.admits_keyed_hash_join_resident_subplan_pipeline()
    );
    assert_eq!(analysis.keyed_hash_join_resident_subplan_admission, CapabilityDecision::Satisfied);
    assert!(analysis.edges.iter().any(|edge| {
        edge.contract_kinds
            .contains(&EdgeContractKind::SharedProducer { identity: SharedProducerIdentity::SameNodeIdAfterCse })
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::AggregateDerivesFromSharedProducer)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::FinalBarrierOrder)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_rejects_structural_equivalence_without_same_node_id() {
    let plan = keyed_hash_join_resident_subplan_pipeline_plan(
        KeyedHashJoinResidentSubplanPipelineVariant::StructurallyEquivalentGroupedAggregate,
    );
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.keyed_hash_join_resident_subplan_admission else {
        panic!("structurally equivalent grouped aggregates must not satisfy resident-subplan admission");
    };
    assert_eq!(*reason, "resident_subplan_requires_same_node_id_after_cse");
    assert_eq!(facts.get("identity").map(String::as_str), Some("same_node_id_after_cse"));
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_preserves_node_failure_locality() {
    let plan = keyed_hash_join_resident_subplan_pipeline_plan(
        KeyedHashJoinResidentSubplanPipelineVariant::UnsupportedRootResidualExpression,
    );
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.keyed_hash_join_resident_subplan_admission else {
        panic!("unsupported residual expression should reject keyed hash resident-subplan admission");
    };
    assert_eq!(*reason, "unsupported_join_residual_expression_capability");
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("case"));
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_pipeline_admission_stays_narrow() {
    for variant in [
        KeyedHashJoinResidentSubplanPipelineVariant::ResidualRootJoin,
        KeyedHashJoinResidentSubplanPipelineVariant::UnsupportedRootResidualExpression,
        KeyedHashJoinResidentSubplanPipelineVariant::SortBelowRoot,
        KeyedHashJoinResidentSubplanPipelineVariant::ScalarAggregateNotMax,
        KeyedHashJoinResidentSubplanPipelineVariant::GroupedAggregateNotSum,
        KeyedHashJoinResidentSubplanPipelineVariant::StructurallyEquivalentGroupedAggregate,
    ] {
        let plan = keyed_hash_join_resident_subplan_pipeline_plan(variant);

        assert!(!plan.capability_analysis().admits_root_join(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline(), "{variant:?}");
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("unsupported keyed hash join resident subplan variant should produce facts");
        assert_rejection_facts_are_family_free(&facts, "unsupported_keyed_hash_resident_subplan");
        assert!(facts.get("bounded_join_rejection_reason").is_some(), "{variant:?}");
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_pipeline_is_executable() {
    let plan = aggregate_over_join_subplan_topk_plan();
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(plan.capability_analysis().admits_multi_join_aggregate_over_join_subplan_topk_pipeline());
    assert!(analysis.admits_multi_join_aggregate_over_join_subplan_topk_pipeline());
    assert!(!plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
    assert!(analysis.edges.iter().any(|edge| {
        matches!(edge.input_role, InputRole::JoinBuildInput { .. })
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::BuildSide
                    && requirement.status == ContractStatus::Modeled
                    && requirement.quality == ContractQuality::ExactPreflight
            })
    }));
    assert!(
        analysis
            .invariants
            .iter()
            .any(|invariant| matches!(invariant.kind, SubgraphInvariantKind::PostRootStageOrder { .. }))
    );
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::FinalSort
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    }));

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("aggregate-over-join subplan TopK pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(facts.get("bounded_join_execution_family"), None);
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| {
        value.contains("source_read:modeled")
            && value.contains("operator_output:modeled")
            && value.contains("final_sort:modeled")
    }));
    assert!(facts.get("bounded_join_reservation_requirements").is_some_and(|value| {
        value.contains("final_sort:final_sort_contract:hard_budgeted_report_with_unresolved_preflight")
    }));
    assert_modeled_reservation_facts_are_consistent(&facts, "multi_join_with_aggregate_over_join_subplan_topk");
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_preserves_node_failure_locality() {
    let plan = aggregate_over_join_subplan_topk_variant_plan(
        AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression,
    );
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_multi_join_aggregate_over_join_subplan_topk_pipeline());
    let CapabilityDecision::Rejected { reason, facts } =
        &analysis.multi_join_aggregate_over_join_subplan_topk_admission
    else {
        panic!("unsupported projection expression should reject aggregate-over-join TopK admission");
    };
    assert_eq!(*reason, "unsupported_projection_expression_capability");
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("case"));
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_capability_proof_stays_narrow() {
    for variant in [
        AggregateOverJoinSubplanTopKVariant::MissingFetchLimit,
        AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan,
        AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide,
        AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression,
    ] {
        let plan = aggregate_over_join_subplan_topk_variant_plan(variant);

        assert!(
            !plan.capability_analysis().admits_multi_join_aggregate_over_join_subplan_topk_pipeline(),
            "{variant:?} must not satisfy the aggregate-over-join TopK capability proof"
        );
        assert!(
            plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_some(),
            "{variant:?} should remain rejected with bounded unsupported-shape facts"
        );
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("rejected aggregate-over-join TopK variant should produce facts");
        assert_rejection_facts_are_family_free(&facts, "unsupported_aggregate_over_join_topk");
        assert!(facts.get("bounded_join_rejection_reason").is_some(), "{variant:?}");
    }
}

#[test]
fn bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline_is_executable() {
    let plan = resident_grouped_aggregate_subquery_filter_plan();

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(!plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(
        !plan
            .capability_analysis()
            .admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline()
    );
    assert!(plan.capability_analysis().admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("resident grouped aggregate subquery filter pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(facts.get("bounded_join_execution_family"), None);
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(facts.get("bounded_join_reservation_requirements").is_some_and(|value| {
        value.contains("final_sort:final_sort_contract:hard_budgeted_report_with_unresolved_preflight")
    }));
    assert_modeled_reservation_facts_are_consistent(
        &facts,
        "multi_join_with_resident_grouped_aggregate_subquery_filter",
    );
}

#[test]
fn bounded_multi_join_resident_grouped_aggregate_subquery_filter_phase8_admission_uses_capability_proof() {
    let plan = resident_grouped_aggregate_subquery_filter_plan();
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(),
        analysis.admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline()
    );
    assert_eq!(analysis.multi_join_resident_grouped_aggregate_subquery_filter_admission, CapabilityDecision::Satisfied);
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::FinalBarrierOrder)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::ResidualUsesBothInputs { .. })
            && invariant.decision == CapabilityDecision::Satisfied
    }));
}

#[test]
fn bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline_is_executable() {
    let plan = resident_grouped_aggregate_input_post_join_aggregate_topk_plan();

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(!plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(
        plan.capability_analysis()
            .admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline()
    );
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("resident grouped aggregate TopK pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(facts.get("bounded_join_execution_family"), None);
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| {
        value.contains("source_read:modeled")
            && value.contains("operator_output:modeled")
            && value.contains("final_sort:modeled")
    }));
    assert!(facts.get("bounded_join_reservation_requirements").is_some_and(|value| {
        value.contains("final_sort:final_sort_contract:hard_budgeted_report_with_unresolved_preflight")
    }));
    assert_modeled_reservation_facts_are_consistent(
        &facts,
        "hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk",
    );
}

#[test]
fn bounded_hash_join_resident_grouped_aggregate_topk_phase6_admission_uses_capability_proof() {
    let plan = resident_grouped_aggregate_input_post_join_aggregate_topk_plan();
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis()
            .admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(),
        analysis.admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline()
    );
    assert_eq!(
        analysis.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
        CapabilityDecision::Satisfied
    );
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::PostRootStageOrder { .. })
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::FinalBarrierOrder)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.edges.iter().any(|edge| {
        edge.contract_kinds.contains(&EdgeContractKind::SortInput)
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::FinalSort
                    && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
            })
    }));
}

#[test]
fn bounded_hash_join_resident_grouped_aggregate_topk_preserves_node_failure_locality() {
    let plan = resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter,
    );
    let analysis = plan.capability_analysis();

    assert!(
        !plan
            .capability_analysis()
            .admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline()
    );
    let CapabilityDecision::Rejected { reason, facts } =
        &analysis.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission
    else {
        panic!("unsupported filter should reject resident grouped-aggregate TopK admission");
    };
    assert_eq!(*reason, "unsupported_filter_expression_capability");
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("in_list"));
    assert_eq!(facts.get("reason").map(String::as_str), Some("in_list_must_not_be_empty"));
}

#[test]
fn bounded_hash_join_resident_grouped_aggregate_topk_admission_stays_narrow() {
    for variant in [
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::MissingFetchLimit,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::ScalarPostJoinAggregate,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter,
    ] {
        let plan = resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(variant);

        assert!(
            !plan
                .capability_analysis()
                .admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(),
            "{variant:?} must not be admitted by the resident grouped-aggregate TopK family"
        );
    }
}

#[test]
fn bounded_keyless_conditional_join_input_subplan_pipeline_admits_only_keyless_input_subplan_bounded_shapes() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    for variant in [
        KeylessInputSubplanVariant::GroupedAggregate,
        KeylessInputSubplanVariant::SortInput,
        KeylessInputSubplanVariant::LimitInput,
        KeylessInputSubplanVariant::BareLeftAntiInput,
        KeylessInputSubplanVariant::InnerJoinInput,
        KeylessInputSubplanVariant::NestedJoinInput,
    ] {
        let plan = keyless_conditional_join_input_subplan_plan(variant);

        assert!(!plan.capability_analysis().admits_root_join(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline(), "{variant:?}");
        assert!(
            !plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph(), "{variant:?}");
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("unsupported keyless input-subplan subplan shape should produce facts");
        assert_eq!(
            facts.get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability").map(String::as_str),
            Some("keyless_conditional_join_input_subplan_pipeline")
        );
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

#[test]
fn bounded_keyless_conditional_join_input_subplan_phase8_admission_uses_capability_proof() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline(),
        analysis.admits_keyless_conditional_join_input_subplan_pipeline()
    );
    assert_eq!(analysis.keyless_conditional_join_input_subplan_admission, CapabilityDecision::Satisfied);
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::ResidualUsesBothInputs { .. })
            && invariant.decision == CapabilityDecision::Satisfied
    }));
}

#[test]
fn bounded_keyless_conditional_join_input_subplan_phase8_rejects_non_input_subplan_shape() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::InnerJoinInput);
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.keyless_conditional_join_input_subplan_admission
    else {
        panic!("inner join input should reject keyless input-subplan admission");
    };
    assert_eq!(*reason, "keyless_conditional_input_subplan_requires_projection_wrapped_left_anti_join");
    assert!(facts.contains_key("node_id"));
}
