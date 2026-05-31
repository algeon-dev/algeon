use super::*;

/// Root keyed/residual join whose inputs are a resident multi-join subtree and
/// a projection-wrapped grouped aggregate, followed by projection, scalar
/// aggregate, and final projection stages.
///
/// The stored ids name the root join, the resident subtree root and wrapper
/// projection, the grouped aggregate and wrapper projection, and each post-root
/// stage that must remain ordered above the join.
#[derive(Debug, Clone, Copy)]
pub(in crate::capability) struct MixedResidualResidentGroupedAggregateInputShape {
    pub(in crate::capability) root_join_id: NodeId,
    pub(in crate::capability) join_subtree_root_id: NodeId,
    pub(in crate::capability) join_subtree_projection_id: NodeId,
    pub(in crate::capability) grouped_aggregate_id: NodeId,
    pub(in crate::capability) grouped_aggregate_projection_id: NodeId,
    pub(in crate::capability) post_root_projection_id: NodeId,
    pub(in crate::capability) post_root_aggregate_id: NodeId,
    pub(in crate::capability) final_projection_id: NodeId,
}

/// Left-semi hash join from a resident multi-join subtree into a resident
/// grouped aggregate, followed by post-join projection, grouped aggregate,
/// projection, sort, and limit.
///
/// The stored ids identify the join, the aggregate-side resident producer and
/// projection, plus the post-root aggregate/top-k chain admitted as one
/// compiled pipeline envelope.
#[derive(Debug, Clone, Copy)]
pub(in crate::capability) struct HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKShape {
    pub(in crate::capability) root_join_id: NodeId,
    pub(in crate::capability) grouped_aggregate_id: NodeId,
    pub(in crate::capability) grouped_aggregate_projection_id: NodeId,
    pub(in crate::capability) post_root_projection_id: NodeId,
    pub(in crate::capability) post_root_aggregate_id: NodeId,
    pub(in crate::capability) final_projection_id: NodeId,
    pub(in crate::capability) final_sort_id: NodeId,
    pub(in crate::capability) final_limit_id: NodeId,
}

/// Top-k query whose root join combines a deep multi-join side with a grouped
/// aggregate computed over a separate join subplan.
///
/// The stored ids pin the root join, aggregate-over-join branch and projection,
/// and the projection/sort/limit stages that consume the joined resident result.
#[derive(Debug, Clone, Copy)]
pub(in crate::capability) struct MultiJoinAggregateOverJoinSubplanTopKShape {
    pub(in crate::capability) root_join_id: NodeId,
    pub(in crate::capability) aggregate_id: NodeId,
    pub(in crate::capability) aggregate_projection_id: NodeId,
    pub(in crate::capability) post_root_projection_id: NodeId,
    pub(in crate::capability) final_sort_id: NodeId,
    pub(in crate::capability) final_limit_id: NodeId,
}

/// Final-sort query where a keyed hash join matches a joined grouped branch
/// against a scalar aggregate derived from the same grouped aggregate producer.
///
/// The stored ids record the final sort, the root join, the branch join, the
/// shared grouped aggregate node that must be common after CSE, and the scalar
/// aggregate consuming that shared producer.
#[derive(Debug, Clone, Copy)]
pub(in crate::capability) struct KeyedHashJoinResidentSubplanShape {
    pub(in crate::capability) final_sort_id: NodeId,
    pub(in crate::capability) root_join_id: NodeId,
    pub(in crate::capability) joined_grouped_branch_join_id: NodeId,
    pub(in crate::capability) shared_grouped_aggregate_id: NodeId,
    pub(in crate::capability) scalar_aggregate_id: NodeId,
}

/// Keyless conditional root join whose input tree contains at least one
/// resident subplan and whose post-root stages are general unary/barrier stages.
///
/// The root join id locates the conditional join; the reachable join count lets
/// admission decide whether multi-join lifetime and schedule invariants are
/// required.
#[derive(Debug, Clone, Copy)]
pub(in crate::capability) struct KeylessConditionalJoinInputSubplanShape {
    pub(in crate::capability) root_join_id: NodeId,
    pub(in crate::capability) reachable_join_count: usize,
}

/// Keyless conditional root join over grouped and scalar aggregate inputs, where
/// both aggregates are projection-wrapped computations over join subplans.
///
/// The stored ids name the root join, both aggregate inputs and their wrapper
/// projections, plus optional projection/sort stages above the root join.
#[derive(Debug, Clone, Copy)]
pub(in crate::capability) struct KeylessConditionalJoinAggregateOverJoinSubplanInputsShape {
    pub(in crate::capability) root_join_id: NodeId,
    pub(in crate::capability) grouped_aggregate_id: NodeId,
    pub(in crate::capability) grouped_aggregate_projection_id: NodeId,
    pub(in crate::capability) scalar_aggregate_id: NodeId,
    pub(in crate::capability) scalar_aggregate_projection_id: NodeId,
    pub(in crate::capability) post_root_projection_id: Option<NodeId>,
    pub(in crate::capability) final_sort_id: Option<NodeId>,
}

/// Final-sort query whose left-semi root join filters a resident multi-join
/// output using a subquery side containing one grouped aggregate comparison.
///
/// The stored ids identify the sort and root join, the grouped aggregate and
/// projection used by the residual comparison, and the residual join that
/// compares the aggregate branch against the rest of the subquery side.
#[derive(Debug, Clone, Copy)]
pub(in crate::capability) struct MultiJoinResidentGroupedAggregateSubqueryFilterShape {
    pub(in crate::capability) final_sort_id: NodeId,
    pub(in crate::capability) root_join_id: NodeId,
    pub(in crate::capability) grouped_aggregate_id: NodeId,
    pub(in crate::capability) grouped_aggregate_projection_id: NodeId,
    pub(in crate::capability) residual_join_id: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::capability) enum OutputLineage {
    SourceField {
        source_id: NodeId,
        field_index: usize,
    },
    AggregateGroup {
        aggregate_id: NodeId,
        group_index: usize,
    },
    AggregateMeasure {
        aggregate_id: NodeId,
        aggr_index: usize,
        function: AggregateFunction,
    },
    ScalarAggregateMeasure {
        aggregate_id: NodeId,
        aggr_index: usize,
        function: AggregateFunction,
        input: Box<OutputLineage>,
    },
    Derived {
        node_id: NodeId,
        field_index: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub(in crate::capability) enum PostRootStage {
    Filter(NodeId),
    Projection(NodeId),
    Aggregate(NodeId),
    Window(NodeId),
    Sort(NodeId),
    Limit(NodeId),
}

#[derive(Debug, Clone, Copy, Default)]
pub(in crate::capability) struct ResidentGroupedAggregateSubqueryFilterStats {
    pub(in crate::capability) join_count: usize,
    pub(in crate::capability) grouped_aggregate_count: usize,
    pub(in crate::capability) residual_comparison_count: usize,
    pub(in crate::capability) aggregate_leaf: Option<(NodeId, NodeId)>,
    pub(in crate::capability) residual_comparison: Option<(NodeId, NodeId, NodeId)>,
}

impl ResidentGroupedAggregateSubqueryFilterStats {
    pub(in crate::capability) fn combine_with_join(
        join_id: NodeId,
        join: &JoinNode,
        left: Self,
        right: Self,
    ) -> Result<Self, CapabilityDecision> {
        let left_has_aggregate = left.grouped_aggregate_count > 0;
        let right_has_aggregate = right.grouped_aggregate_count > 0;
        let residual_refs_both_sides = join
            .spec
            .residual_predicate
            .as_ref()
            .is_some_and(|predicate| expr_references_input(predicate, 0) && expr_references_input(predicate, 1));
        let this_residual_comparison = join.spec.residual_predicate.is_some()
            && residual_refs_both_sides
            && left_has_aggregate != right_has_aggregate;
        let comparison = if this_residual_comparison {
            let aggregate =
                if left_has_aggregate { left.aggregate_leaf } else { right.aggregate_leaf }.ok_or_else(|| {
                    CapabilityDecision::rejected(
                        "resident_grouped_subquery_filter_missing_grouped_aggregate_leaf",
                        [("join_id", join_id.to_string())],
                    )
                })?;
            Some((join_id, aggregate.0, aggregate.1))
        } else {
            left.residual_comparison.or(right.residual_comparison)
        };
        Ok(Self {
            join_count: left.join_count + right.join_count + 1,
            grouped_aggregate_count: left.grouped_aggregate_count + right.grouped_aggregate_count,
            residual_comparison_count: left.residual_comparison_count
                + right.residual_comparison_count
                + usize::from(this_residual_comparison),
            aggregate_leaf: left.aggregate_leaf.or(right.aggregate_leaf),
            residual_comparison: comparison,
        })
    }
}
