use std::{collections::BTreeMap, fmt, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};

use crate::{
    bounded::{
        BoundedJoinPlanSupport,
        analyze_bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline,
        analyze_bounded_join_pipeline, analyze_bounded_join_plan_support,
        analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,
        analyze_bounded_keyless_conditional_join_input_subplan_pipeline,
        analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
        analyze_keyed_hash_join_resident_subplan_pipeline, bounded_join_modeled_facts,
        bounded_join_unsupported_shape_facts, bounded_root_filtered_join_compatible, bounded_root_hash_join_compatible,
        bounded_root_join_compatible, bounded_root_keyless_conditional_join_compatible,
        bounded_root_mixed_join_compatible, bounded_root_projected_cross_join_compatible,
    },
    expr::{AggregateSpec, ColumnRef, Expr, LimitSpec, SortExpr, SortSpec},
    runtime::ExecutionBoundaryPolicy,
    source::SourceNode,
};

mod builder;
mod core;
mod display;
mod facts;
mod join;
mod nodes;
mod sink;

pub use core::{NodeId, PlanNode, QueryNode, QueryPlan};

pub use builder::QueryPlanBuilder;
pub use facts::{DataPlacement, PlanFacts, SchemaBehavior};
pub use join::{
    HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
};
pub use nodes::{AggregateNode, FilterNode, LimitNode, ProjectionNode, ProjectionOutput, SortNode};
pub use sink::ResultSink;
