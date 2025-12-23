mod aggregate;
mod expr;
mod join;
mod join_residual;
mod root;
mod sort;
mod source;
mod unary;

use arrow::datatypes::SchemaRef;
use cudf_query_engine::{
    NodeId as NativeNodeId, QueryPlan as NativeQueryPlan, QueryPlanBuilder as NativeQueryPlanBuilder,
};

use super::normalize::{NormalizedNativeInput, NormalizedNativePlan, NormalizedNativeRoot};
use crate::policy::FallbackReason;

pub(super) fn lower_normalized_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    match &normalized.root {
        NormalizedNativeRoot::Aggregate(root) if normalized.single_source_input().is_some() => {
            root::lower_slice1_plan(normalized, root)
        },
        NormalizedNativeRoot::Aggregate(root) => root::lower_joined_aggregate_plan(normalized, root),
        NormalizedNativeRoot::Input(root) => root::lower_input_root_plan(normalized, root),
    }
}

pub(super) struct LoweredNativeInput {
    pub(super) node: NativeNodeId,
    pub(super) schema: SchemaRef,
}

fn lower_native_input(
    input: &NormalizedNativeInput<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    match input {
        NormalizedNativeInput::Source(source) => source::lower_source_input(source, builder, required_output_columns),
        NormalizedNativeInput::Filter(filter) => unary::lower_filter_input(filter, builder, required_output_columns),
        NormalizedNativeInput::Projection(projection) => {
            unary::lower_projection_input(projection, builder, required_output_columns)
        },
        NormalizedNativeInput::Aggregate(aggregate) => {
            aggregate::lower_aggregate_input(aggregate, builder, required_output_columns)
        },
        NormalizedNativeInput::HashJoin(join) => join::lower_hash_join_input(join, builder, required_output_columns),
        NormalizedNativeInput::NestedLoopJoin(join) => {
            join::lower_nested_loop_join_input(join, builder, required_output_columns)
        },
    }
}
