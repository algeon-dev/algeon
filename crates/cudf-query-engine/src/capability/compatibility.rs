use super::{model::*, *};

pub(super) fn compatibility_family_projection(plan: &QueryPlan) -> Option<CompatibilityFamilyProjection> {
    match analyze_bounded_join_plan_support(plan) {
        BoundedJoinPlanSupport::RootJoin => Some(CompatibilityFamilyProjection::RootJoin),
        BoundedJoinPlanSupport::SingleJoinPostRootPipeline => {
            Some(CompatibilityFamilyProjection::SingleJoinPostRootPipeline)
        },
        BoundedJoinPlanSupport::KeylessConditionalJoinInputSubplanPipeline => {
            Some(CompatibilityFamilyProjection::KeylessConditionalJoinInputSubplanPipeline)
        },
        BoundedJoinPlanSupport::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline => {
            Some(CompatibilityFamilyProjection::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinTreePostRootPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinTreePostRootPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinAggregateOverJoinSubplanTopKPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinAggregateOverJoinSubplanTopKPipeline)
        },
        BoundedJoinPlanSupport::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline => {
            Some(CompatibilityFamilyProjection::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline)
        },
        BoundedJoinPlanSupport::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline => {
            Some(CompatibilityFamilyProjection::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline)
        },
        BoundedJoinPlanSupport::KeyedHashJoinResidentSubplanPipeline => {
            Some(CompatibilityFamilyProjection::KeyedHashJoinResidentSubplanPipeline)
        },
        BoundedJoinPlanSupport::Unsupported => None,
    }
}
