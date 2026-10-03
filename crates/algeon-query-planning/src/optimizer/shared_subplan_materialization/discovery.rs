struct SharedSubplanMaterializationDiscovery {
    diagnostics: SharedSubplanMaterializationDiagnostics,
    families: Vec<SharedAggregateFamilyRewrite>,
}

fn discover_shared_subplan_materialization_candidates(plan: &QueryPlan) -> SharedSubplanMaterializationDiscovery {
    let mut diagnostics = SharedSubplanMaterializationDiagnostics::default();
    let mut families = BTreeMap::<String, Vec<AggregateMaterializationSignature>>::new();

    for node in plan.nodes() {
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            continue;
        };
        if aggregate.spec.grouping_sets.is_some() {
            continue;
        }
        if aggregate.spec.group_exprs.is_empty() && !scalar_filtered_aggregate_supported(&aggregate.spec) {
            continue;
        }
        let Some(signature) = aggregate_materialization_signature(plan, node.id, aggregate.input, &aggregate.spec)
        else {
            continue;
        };
        diagnostics.considered_count += 1;
        families.entry(signature.family_key.clone()).or_default().push(signature);
    }

    // The family key hashes source locations, so families are reported in
    // plan order (by their first aggregate) rather than by key.
    let mut families = families.into_values().filter(|family| family.len() >= 2).collect::<Vec<_>>();
    for family in &mut families {
        family.sort_by_key(|signature| signature.aggregate_node_id);
    }
    families.sort_by_key(|family| family[0].aggregate_node_id);
    let mut rewrite_families = Vec::new();
    for family in families {
        let representative = &family[0];
        let repeated_selected_source_bytes = sum_optional_u64(family.iter().map(|signature| signature.source_bytes));
        let duplicate_selected_source_bytes = repeated_selected_source_bytes.map(|total| {
            let largest = family.iter().filter_map(|signature| signature.source_bytes).max().unwrap_or(0);
            total.saturating_sub(largest)
        });
        let exact_source_variant_count =
            family.iter().map(|signature| signature.exact_source_signature.as_str()).collect::<BTreeSet<_>>().len();

        let rewrite_plan = shared_aggregate_family_rewrite_plan(plan, &family);
        let applied = rewrite_plan.is_ok();
        let rejected_reason = rewrite_plan.as_ref().err().copied();
        let filtered_aggregate_count = shared_scalar_filtered_aggregate_count(&family);
        let filtered_memory_estimate = estimate_shared_scalar_filtered_materialization(&family);
        let filtered_aggregate_memory_shape = if filtered_aggregate_count > 0 {
            if applied {
                if filtered_memory_estimate.is_some() {
                    "runtime_budgeted_segmented_state_available"
                } else {
                    "runtime_budgeted_streaming_state_available"
                }
            } else {
                "streaming_filtered_state_not_applied"
            }
        } else {
            "not_applicable"
        }
        .to_string();
        let filtered_aggregate_memory_threshold_policy =
            if filtered_aggregate_count > 0 { "runtime_budget/2" } else { "not_applicable" }.to_string();
        diagnostics.candidate_count += 1;
        diagnostics.candidates.push(SharedSubplanMaterializationCandidateDiagnostics {
            rule_name: SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
            family_key: representative.family_key.clone(),
            aggregate_node_ids: family.iter().map(|signature| signature.aggregate_node_id).collect(),
            reuse_count: family.len(),
            group_exprs: representative.group_exprs.clone(),
            aggregate_exprs: materialization_family_aggregate_exprs(&family),
            filtered_aggregate_count,
            filtered_aggregate_memory_shape,
            filtered_aggregate_estimation_input_rows: filtered_memory_estimate
                .as_ref()
                .map(|estimate| estimate.input_rows),
            filtered_aggregate_estimated_materialization_bytes: filtered_memory_estimate
                .as_ref()
                .map(|estimate| estimate.estimated_materialization_bytes),
            filtered_aggregate_memory_budget_bytes: None,
            filtered_aggregate_memory_threshold_bytes: None,
            filtered_aggregate_memory_threshold_policy,
            source_fingerprints: representative.source_fingerprints.clone(),
            exact_source_variant_count,
            repeated_selected_source_bytes,
            duplicate_selected_source_bytes,
            applied,
            rejected_reason,
        });
        if let Ok(rewrite_plan) = rewrite_plan {
            rewrite_families.push(rewrite_plan);
        }
    }

    diagnostics.applied_count = diagnostics.candidates.iter().filter(|candidate| candidate.applied).count();
    SharedSubplanMaterializationDiscovery { diagnostics, families: rewrite_families }
}

fn shared_scalar_filtered_aggregate_count(family: &[AggregateMaterializationSignature]) -> usize {
    let Some(representative) = family.first() else {
        return 0;
    };
    if !representative.group_exprs.is_empty() {
        return 0;
    }
    family.iter().map(|signature| signature.aggregate_exprs.len()).sum()
}
