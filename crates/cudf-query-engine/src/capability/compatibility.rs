use super::{model::*, shape::*, *};
use crate::runtime::{BOUNDED_JOIN_EXECUTION_FAMILY_CONTRACTS, BoundedJoinExecutionFamily};

pub(crate) struct CompatibilityReportProjection {
    execution_family: BoundedJoinExecutionFamily,
    required_shape_contract: &'static str,
    reservation_requirements: Vec<ReservationRequirement>,
}

impl CompatibilityReportProjection {
    pub(crate) fn from_analysis(analysis: &CapabilityAnalysis) -> Option<Self> {
        if analysis.has_rejections() || analysis.bounded_join_execution_path().is_none() {
            return None;
        }
        let execution_family = compatibility_execution_family(analysis)?;
        Some(Self {
            execution_family,
            required_shape_contract: required_shape_contract(execution_family),
            reservation_requirements: analysis.compatibility_reservation_requirements(),
        })
    }

    pub(crate) fn insert_modeled_facts(&self, facts: &mut BTreeMap<&'static str, String>) {
        facts.insert("bounded_join_execution_family", self.execution_family.as_str().to_string());
        facts.insert("bounded_join_required_shape_contract", self.required_shape_contract.to_string());
        facts.insert("bounded_join_contract_modes", self.contract_modes_csv());
        facts.insert("bounded_join_unresolved_preflight_contracts", self.unresolved_preflight_contracts_csv());
        facts.insert("bounded_join_contract_report_fields", self.contract_report_fields_csv());
        facts.insert("bounded_join_reservation_kinds", self.reservation_kinds_csv());
        facts.insert("bounded_join_reservation_statuses", self.reservation_statuses_csv());
        facts.insert("bounded_join_reservation_requirements", self.reservation_requirements_csv());
        facts.insert("blocked_reservations", self.blocked_reservations_csv());
        facts.insert("missing_reservations", self.blocked_reservations_csv());
        facts.insert("modeled_reservations", self.modeled_reservations_csv());
        facts.insert("bounded_join_lower_level_blockers", self.lower_level_blockers_csv());
    }

    fn reservation_kinds_csv(&self) -> String {
        csv_join(self.reservation_requirements.iter().map(|requirement| requirement.kind.as_str()))
    }

    fn reservation_statuses_csv(&self) -> String {
        csv_join(
            self.reservation_requirements
                .iter()
                .map(|requirement| format!("{}:{}", requirement.kind.as_str(), requirement.status.as_str())),
        )
    }

    fn reservation_requirements_csv(&self) -> String {
        csv_join(self.reservation_requirements.iter().map(ReservationRequirement::compatibility_fact_row))
    }

    fn modeled_reservations_csv(&self) -> String {
        csv_join(self.reservation_requirements.iter().filter_map(|requirement| {
            (requirement.status == ContractStatus::Modeled).then_some(requirement.kind.as_str())
        }))
    }

    fn blocked_reservations_csv(&self) -> String {
        csv_join(self.reservation_requirements.iter().filter_map(|requirement| {
            (requirement.status != ContractStatus::Modeled).then_some(requirement.kind.as_str())
        }))
    }

    fn lower_level_blockers_csv(&self) -> String {
        csv_join(self.reservation_requirements.iter().filter_map(|requirement| {
            (requirement.status == ContractStatus::BlockedByLowerLevel).then_some(requirement.required_contract)
        }))
    }

    fn contract_modes_csv(&self) -> String {
        csv_join(self.hard_budgeted_contracts().into_iter().map(|contract| format!("{contract}:hard_budgeted_report")))
    }

    fn unresolved_preflight_contracts_csv(&self) -> String {
        csv_join(self.hard_budgeted_contracts().into_iter().filter_map(|contract| {
            unresolved_preflight_contract(contract).map(|detail| format!("{contract}:{detail}"))
        }))
    }

    fn contract_report_fields_csv(&self) -> String {
        csv_join(
            self.hard_budgeted_contracts()
                .into_iter()
                .filter_map(|contract| contract_report_fields(contract).map(|fields| format!("{contract}:{fields}"))),
        )
    }

    fn hard_budgeted_contracts(&self) -> Vec<&'static str> {
        let mut contracts = Vec::new();
        for requirement in &self.reservation_requirements {
            if requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
                && !contracts.contains(&requirement.required_contract)
            {
                contracts.push(requirement.required_contract);
            }
        }
        contracts
    }
}

pub(super) fn compatibility_family_projection(plan: &QueryPlan) -> Option<CompatibilityFamilyProjection> {
    if root_join_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::RootJoin)
    } else if single_join_post_root_pipeline_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::SingleJoinPostRootPipeline)
    } else if keyless_conditional_join_input_subplan_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::KeylessConditionalJoinInputSubplanPipeline)
    } else if keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline)
    } else if matches!(closed_multi_join_subgraph_structure_decision(plan), CapabilityDecision::Satisfied) {
        Some(CompatibilityFamilyProjection::MultiJoinTreePostRootPipeline)
    } else if multi_join_aggregate_over_join_subplan_topk_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::MultiJoinAggregateOverJoinSubplanTopKPipeline)
    } else if mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline)
    } else if hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline)
    } else if multi_join_resident_grouped_aggregate_subquery_filter_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline)
    } else if keyed_hash_join_resident_subplan_shape_decision(plan).is_ok() {
        Some(CompatibilityFamilyProjection::KeyedHashJoinResidentSubplanPipeline)
    } else {
        None
    }
}

fn compatibility_execution_family(analysis: &CapabilityAnalysis) -> Option<BoundedJoinExecutionFamily> {
    match analysis.compatibility_family_projection? {
        CompatibilityFamilyProjection::RootJoin | CompatibilityFamilyProjection::SingleJoinPostRootPipeline => {
            single_join_execution_family(analysis)
        },
        CompatibilityFamilyProjection::KeylessConditionalJoinInputSubplanPipeline => {
            Some(BoundedJoinExecutionFamily::KeylessConditionalJoinInputSubplanPipeline)
        },
        CompatibilityFamilyProjection::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline => {
            Some(BoundedJoinExecutionFamily::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline)
        },
        CompatibilityFamilyProjection::MultiJoinTreePostRootPipeline => Some(multi_join_execution_family(analysis)),
        CompatibilityFamilyProjection::MultiJoinAggregateOverJoinSubplanTopKPipeline => {
            Some(BoundedJoinExecutionFamily::MultiJoinAggregateOverJoinSubplanTopKPipeline)
        },
        CompatibilityFamilyProjection::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline => {
            Some(BoundedJoinExecutionFamily::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline)
        },
        CompatibilityFamilyProjection::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline => {
            Some(BoundedJoinExecutionFamily::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline)
        },
        CompatibilityFamilyProjection::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline => {
            Some(BoundedJoinExecutionFamily::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline)
        },
        CompatibilityFamilyProjection::KeyedHashJoinResidentSubplanPipeline => {
            Some(BoundedJoinExecutionFamily::KeyedHashJoinResidentSubplanPipeline)
        },
    }
}

fn single_join_execution_family(analysis: &CapabilityAnalysis) -> Option<BoundedJoinExecutionFamily> {
    let join = analysis.nodes.iter().find_map(|node| match node.operator {
        NodeOperator::Join { join_type, key_count, residual, .. } => Some((join_type, key_count, residual)),
        _ => None,
    })?;

    match join {
        (JoinType::Cross, 0, ResidualRequirement::None) => Some(BoundedJoinExecutionFamily::RootProjectedCrossJoin),
        (_, 0, ResidualRequirement::Required) => Some(BoundedJoinExecutionFamily::KeylessConditionalJoin),
        (_, _, ResidualRequirement::Required) => Some(BoundedJoinExecutionFamily::RootMixedKeyedResidualJoin),
        (JoinType::LeftSemi | JoinType::LeftAnti, _, ResidualRequirement::None) => {
            Some(BoundedJoinExecutionFamily::RootFilteredSemiAntiJoin)
        },
        _ => Some(BoundedJoinExecutionFamily::RootReusableHashJoin),
    }
}

fn multi_join_execution_family(analysis: &CapabilityAnalysis) -> BoundedJoinExecutionFamily {
    if analysis
        .nodes
        .iter()
        .any(|node| matches!(node.operator, NodeOperator::Join { residual: ResidualRequirement::Required, .. }))
    {
        BoundedJoinExecutionFamily::MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline
    } else {
        BoundedJoinExecutionFamily::MultiJoinReusableHashPipeline
    }
}

fn required_shape_contract(family: BoundedJoinExecutionFamily) -> &'static str {
    BOUNDED_JOIN_EXECUTION_FAMILY_CONTRACTS
        .iter()
        .find(|contract| contract.family() == family)
        .map(|contract| contract.topology_contract())
        .unwrap_or("")
}

fn unresolved_preflight_contract(contract: &str) -> Option<&'static str> {
    match contract {
        "source_contract" => Some("libcudf_parquet_decoded_output_preflight_unavailable"),
        "aggregate_contract" => Some("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"),
        "final_sort_contract" => Some("libcudf_final_sort_order_preflight_unavailable"),
        _ => None,
    }
}

fn contract_report_fields(contract: &str) -> Option<&'static str> {
    match contract {
        "source_contract" => Some("allocator_high_water_bytes+retained_bytes+output_bytes+output_rows+context_facts"),
        "aggregate_contract" => Some("allocator_high_water_bytes+retained_bytes+output_bytes+output_rows"),
        "final_sort_contract" => {
            Some("allocator_high_water_bytes+retained_sort_order_bytes+materialized_output_bytes+output_rows")
        },
        _ => None,
    }
}

fn csv_join<I, T>(values: I) -> String
where
    I: IntoIterator<Item = T>,
    T: ToString,
{
    let mut output = String::new();
    for value in values {
        if !output.is_empty() {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}
