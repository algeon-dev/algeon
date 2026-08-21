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
            NodeOperator::Aggregate { grouping: nexus_query_engine::capability::AggregateGrouping::Grouped, .. }
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
    assert!(root_analysis.admits_bounded_join());
    assert!(!root_analysis.admits_closed_multi_join_subgraph());
    assert!(!root_analysis.admits_keyless_conditional_join_input_subplan_pipeline());

    let single = serial_aggregate_single_join_spine_plan();
    let single_analysis = single.capability_analysis();
    assert!(single.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(single_analysis.admits_single_join_post_root_pipeline());
    assert!(single_analysis.admits_bounded_join());
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
fn bounded_capability_analysis_classifies_reservation_scopes() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let analysis = plan.capability_analysis();
    assert!(!analysis.requirements.is_empty());
    for requirement in &analysis.requirements {
        assert_eq!(
            requirement.scope,
            nexus_query_engine::capability::reservation_requirement_scope(requirement.kind),
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
fn bounded_capability_analysis_extracts_multi_join_edge_contracts() {
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
fn bounded_capability_analysis_solves_multi_join_invariants() {
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
fn bounded_multi_join_admission_uses_closed_subgraph_capability_proof() {
    let plan = serial_aggregate_multi_join_spine_plan();
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_closed_multi_join_subgraph(),
        analysis.admits_closed_multi_join_subgraph()
    );
}

#[test]
fn bounded_capability_analysis_rejects_multi_join_admission_without_family_recognizer() {
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

    let projection = analysis.first_rejection_projection().expect("bounded envelope rejection should be projected");
    assert!(matches!(projection.owner, nexus_query_engine::capability::CapabilityRejectionOwner::Admission(_)));

    assert!(
        nexus_query_engine::admission::bounded_pipeline_admission(&plan)
            .expect("admission should not fail")
            .is_none(),
        "admission must reject through the same envelope projection"
    );

    let unsupported_facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
        .expect("unsupported envelope should produce bounded rejection facts");
    assert_eq!(unsupported_facts.get("bounded_join_rejection_reason").map(String::as_str), Some(projection.reason));
    for (key, value) in &projection.facts {
        let needle = format!("{key}={value}");
        assert!(
            unsupported_facts.get("bounded_join_rejection_facts").is_some_and(|facts| facts.contains(&needle)),
            "unsupported facts should include projected fact {needle}"
        );
    }

    let report = CapabilityContractReport::from_capability_analysis(&analysis)
        .expect("unsupported envelope should still produce a admission contract report");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Rejected);
    let blocking_row = report
        .gate_rows()
        .iter()
        .find(|row| row.contract_status == ContractStatus::RejectedByCapability)
        .expect("admission report should expose the same envelope projection");
    assert_eq!(blocking_row.required_contract, projection.reason);
    assert_eq!(blocking_row.memory_boundary, projection.memory_boundary);
}

#[test]
fn bounded_capability_analysis_projects_reservation_facts_without_family_path_vocabulary() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let facts = modeled_facts(&plan, ExecutionBoundaryPolicy::default())
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

    let facts = modeled_facts(&plan, ExecutionBoundaryPolicy::default())
        .expect("root join should produce modeled bounded facts through capability projection");

    assert_eq!(facts.get("bounded_join_execution_family"), None);
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
    assert_modeled_reservation_facts_are_consistent(&facts, "root_join");
}

#[test]
fn bounded_capability_analysis_prevents_modeled_facts_for_capability_rejections() {
    let plan = single_join_pipeline_with_empty_in_list_filter_plan();

    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(modeled_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
    let facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
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
    assert!(modeled_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_capability_analysis_rejects_unsupported_join_output_dtype_before_strategy_selection() {
    let plan = root_join_with_unsupported_output_dtype_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.has_rejections());
    let rejected_join = analysis
        .nodes
        .iter()
        .find(|node| matches!(node.operator, NodeOperator::Join { join_type: JoinType::Inner, .. }))
        .expect("unsupported output dtype plan should contain an inner join capability");
    let CapabilityDecision::Rejected { reason, facts } = &rejected_join.executable else {
        panic!("join with unsupported output dtype should be rejected as a node capability");
    };
    assert_eq!(*reason, "unsupported_join_output_dtype");
    assert_eq!(facts.get("field_index").map(String::as_str), Some("0"));
    assert_eq!(facts.get("field_name").map(String::as_str), Some("l_key"));
    assert_eq!(facts.get("dtype").map(String::as_str), Some("Binary"));
    assert!(modeled_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn admission_contract_report_projects_rejected_node_capability_when_reservations_are_modeled() {
    let plan = unsupported_cross_key_join_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.has_rejections());
    assert!(analysis.requirements.iter().all(|requirement| requirement.status == ContractStatus::Modeled));

    let report = CapabilityContractReport::from_capability_analysis(&analysis)
        .expect("unsupported capability analysis should produce a admission contract report");
    let row = report
        .gate_rows()
        .iter()
        .find(|row| row.contract_status == ContractStatus::RejectedByCapability)
        .expect("unsupported report should expose a local capability rejection row");

    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Rejected);
    assert_eq!(row.contract_scope, ReservationRequirementScope::Node);
    assert!(matches!(row.owner, CapabilityContractOwner::CapabilityNode(_)));
    assert_eq!(row.reservation_kind, ReservationKind::CapabilityContract);
    assert_eq!(row.memory_boundary, "node_capability");
    assert_eq!(row.required_contract, "unsupported_join_node_capability");
    assert_eq!(row.contract_quality, ContractQuality::CapabilityProof);
}

#[test]
fn bounded_capability_analysis_rejects_unsupported_filter_expression_as_node_capability() {
    let plan = single_join_pipeline_with_empty_in_list_filter_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_bounded_join());
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

    assert!(!analysis.admits_bounded_join());
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

pub(crate) fn unsupported_cross_key_join_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn simple_root_join_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn root_join_with_unsupported_output_dtype_plan() -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("r_key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Binary, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-unsupported-output-left.parquet",
        left_schema,
        vec!["l_key".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-unsupported-output-right.parquet",
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
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn single_join_pipeline_with_empty_in_list_filter_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn root_join_with_like_residual_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn root_join_with_one_sided_residual_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn multi_join_candidate_with_aggregate_inside_join_subtree_plan() -> nexus_query_engine::plan::QueryPlan {
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
        "/tmp/multi-join-partsupp.parquet",
        partsupp_schema,
        vec!["ps_partkey".into(), "ps_suppkey".into()],
    )));
    let part = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/multi-join-part.parquet",
        part_schema,
        vec!["p_partkey".into()],
    )));
    let supplier = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/multi-join-supplier.parquet",
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
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_multi_join_pipeline_admits_serial_aggregate_spine_with_projection_and_sort() {
    let plan = serial_aggregate_multi_join_spine_plan();

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_multi_join_pipeline_admits_sort_between_joins() {
    let plan = sort_between_two_joins_plan();

    assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_multi_join_pipeline_admits_supported_global_row_number_between_joins() {
    let plan = global_row_number_between_two_joins_plan();

    assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(nexus_query_engine::admission::bounded_pipeline_admission(&plan).unwrap().is_some());
}

#[test]
fn bounded_root_full_sort_admits_shared_projection_fanout_with_filtered_semi_join() {
    let plan = distinct_aggregate_after_two_joins_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_closed_multi_join_subgraph());
    assert!(analysis.admits_root_full_sort_join_aggregate_pipeline());
    assert!(nexus_query_engine::admission::bounded_pipeline_admission(&plan).unwrap().is_some());
}

#[test]
fn bounded_multi_join_pipeline_admits_shared_topk_leaf_with_final_sort() {
    let plan = shared_topk_multi_join_final_sort_plan();

    assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(nexus_query_engine::admission::bounded_pipeline_admission(&plan).unwrap().is_some());
}

#[test]
fn bounded_single_join_pipeline_rejects_aggregate_join_input_leaves() {
    for aggregate_side in [JoinSide::Left, JoinSide::Right] {
        let plan = single_join_with_aggregate_input_plan(aggregate_side);

        assert!(!plan.capability_analysis().admits_root_join());
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
        assert!(
            unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_some(),
            "aggregate {aggregate_side} join input must not be accepted as a bounded root join leaf"
        );
    }
}
