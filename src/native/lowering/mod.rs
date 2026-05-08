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

/// Lowers a normalized native tree into the DataFusion-free query IR.
///
/// Aggregate roots over `single_source_input` use the slice1 path so source
/// pruning, filter pushdown, zero-column count, aggregate, sort, and limit can
/// be planned together. All other aggregate roots lower their input tree first,
/// which covers joins and recursive unary/aggregate inputs.
pub(super) fn lower_normalized_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    match &normalized.root {
        NormalizedNativeRoot::Aggregate(root) if normalized.single_source_input().is_some() => {
            root::lower_slice1_plan(normalized, root)
        },
        NormalizedNativeRoot::Aggregate(root) => root::lower_joined_aggregate_plan(normalized, root),
        NormalizedNativeRoot::Input(root) => root::lower_input_root_plan(normalized, root),
    }
}

/// Native node plus the schema visible to its parent lowering step.
pub(super) struct LoweredNativeInput {
    pub(super) node: NativeNodeId,
    pub(super) schema: SchemaRef,
}

/// Recursively lowers a normalized input subtree.
///
/// `required_output_columns` is the set of output columns requested by the
/// parent lowering step. An empty slice means the parent requested no output
/// columns. Some root/non-source callers use that as no explicit pruning, but
/// source and filter lowering may still emit only columns needed by predicates,
/// or zero output columns when nothing needs to be projected upward.
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
