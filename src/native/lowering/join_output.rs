use arrow::datatypes::SchemaRef;
use cudf_query_engine::{
    ColumnRef as NativeColumnRef, JoinOutputColumn as NativeJoinOutputColumn, JoinSide as NativeJoinSide,
};
use datafusion::{common::JoinSide as DataFusionJoinSide, logical_expr::JoinType, physical_plan::ExecutionPlan};

use super::{FieldBinding, LoweredNativeInput, RequiredFields, executable_schema_and_bindings, join::CanonicalJoin};
use crate::{
    native::normalize::{NormalizedNativeHashJoin, NormalizedNativeNestedLoopJoin},
    policy::FallbackReason,
};

/// Join output projection plus the exact DataFusion output schema requested by
/// the parent. Native input ordering may change, but this schema must not.
pub(super) struct JoinOutputColumns {
    pub(super) projection: Vec<NativeJoinOutputColumn>,
    pub(super) schema: SchemaRef,
    pub(super) bindings: Vec<FieldBinding>,
}

pub(super) struct JoinOutputRequests {
    pub(super) outputs: Vec<JoinOutputRequest>,
    schema: SchemaRef,
    bindings: Vec<FieldBinding>,
}

impl JoinOutputRequests {
    pub(super) fn lower(
        self,
        left: &LoweredNativeInput,
        right: &LoweredNativeInput,
        canonical: CanonicalJoin,
    ) -> Result<JoinOutputColumns, FallbackReason> {
        let projection = self
            .outputs
            .iter()
            .zip(self.bindings.iter())
            .map(|(request, output_binding)| {
                let side = canonical.native_side_for_datafusion_side(request.datafusion_side)?;
                let input = if request.datafusion_side == DataFusionJoinSide::Left { left } else { right };
                let input_binding = input.binding_for_frontend_index(request.input_index)?;
                let input_ordinal = match side {
                    NativeJoinSide::Left => 0,
                    NativeJoinSide::Right => 1,
                };
                let column = NativeColumnRef::input(
                    input_ordinal,
                    input_binding.native_name.as_str(),
                    input_binding.frontend_name.as_str(),
                );
                Ok(NativeJoinOutputColumn::new(side, column, output_binding.native_name.clone()))
            })
            .collect::<Result<Vec<_>, FallbackReason>>()?;
        Ok(JoinOutputColumns { projection, schema: self.schema, bindings: self.bindings })
    }
}

pub(super) struct JoinOutputRequest {
    pub(super) datafusion_side: DataFusionJoinSide,
    pub(super) input_index: usize,
}

pub(super) fn hash_join_output_requests(
    join: &NormalizedNativeHashJoin<'_>,
    required_output_fields: &RequiredFields,
) -> Result<JoinOutputRequests, FallbackReason> {
    join_output_requests(
        required_output_fields,
        &join.join.left().schema(),
        &join.join.right().schema(),
        &join.join.schema(),
        join.join.projection.as_deref(),
        join.join.join_type(),
    )
}

pub(super) fn nested_loop_join_output_requests(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    required_output_fields: &RequiredFields,
) -> Result<JoinOutputRequests, FallbackReason> {
    join_output_requests(
        required_output_fields,
        &join.join.left().schema(),
        &join.join.right().schema(),
        &join.join.schema(),
        join.join.projection().as_deref(),
        join.join.join_type(),
    )
}

fn join_output_requests(
    required_output_fields: &RequiredFields,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    output_schema: &SchemaRef,
    projection_indices: Option<&[usize]>,
    datafusion_join_type: &JoinType,
) -> Result<JoinOutputRequests, FallbackReason> {
    // DataFusion owns output ordering, projection, aliases, and semi/anti
    // output-side rules. Convert each requested DataFusion output field into a
    // native side/column reference after canonicalization.
    let (schema, bindings) = executable_schema_and_bindings(output_schema, required_output_fields)?;
    let mut outputs = Vec::new();
    for output_index in required_output_fields.requested_indices(output_schema.fields().len())? {
        let join_schema_index = match projection_indices {
            Some(projection) => *projection.get(output_index).ok_or(FallbackReason::UnsupportedExpression)?,
            None => output_index,
        };
        let (datafusion_side, input_index) =
            datafusion_join_output_side(datafusion_join_type, join_schema_index, left_schema)?;
        let input_schema = if datafusion_side == DataFusionJoinSide::Left { left_schema } else { right_schema };
        input_schema.fields().get(input_index).ok_or(FallbackReason::UnsupportedExpression)?;
        outputs.push(JoinOutputRequest { datafusion_side, input_index });
    }
    Ok(JoinOutputRequests { outputs, schema, bindings })
}

pub(super) fn datafusion_join_output_side(
    join_type: &JoinType,
    join_schema_index: usize,
    left_schema: &SchemaRef,
) -> Result<(DataFusionJoinSide, usize), FallbackReason> {
    match join_type {
        JoinType::RightSemi | JoinType::RightAnti => Ok((DataFusionJoinSide::Right, join_schema_index)),
        JoinType::LeftSemi | JoinType::LeftAnti => Ok((DataFusionJoinSide::Left, join_schema_index)),
        _ if join_schema_index < left_schema.fields().len() => Ok((DataFusionJoinSide::Left, join_schema_index)),
        _ => Ok((DataFusionJoinSide::Right, join_schema_index - left_schema.fields().len())),
    }
}
