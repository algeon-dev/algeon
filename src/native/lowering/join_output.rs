use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::{common::JoinSide as DataFusionJoinSide, logical_expr::JoinType, physical_plan::ExecutionPlan};
use nexus_query_engine::{
    expr::ColumnRef as NativeColumnRef,
    plan::{JoinOutputColumn as NativeJoinOutputColumn, JoinSide as NativeJoinSide},
};

use super::{FieldBinding, LoweredNativeInput, RequiredFields, executable_schema_and_bindings, join::CanonicalJoin};
use crate::{
    native::normalize::{NormalizedNativeCrossJoin, NormalizedNativeHashJoin, NormalizedNativeNestedLoopJoin},
    policy::FallbackReason,
};

/// Join output projection plus the exact DataFusion output schema requested by
/// the parent. Native input ordering may change, but this schema must not.
pub(super) struct JoinOutputColumns {
    pub(super) projection: Vec<NativeJoinOutputColumn>,
    pub(super) schema: SchemaRef,
    pub(super) bindings: Vec<FieldBinding>,
    /// Mark column name for LeftMark joins. When set, the output schema
    /// contains this Boolean column as the last field, but it is not in the
    /// `projection` (the executor constructs it).
    pub(super) mark_column_name: Option<String>,
}

/// DataFusion output slots requested from a join before native side remapping.
pub(super) struct JoinOutputRequests {
    pub(super) outputs: Vec<JoinOutputRequest>,
    schema: SchemaRef,
    bindings: Vec<FieldBinding>,
    /// Mark column name for LeftMark joins. When set, the mark column is not
    /// in `outputs` (it is executor-constructed); the name is forwarded to
    /// `JoinOutputColumns::mark_column_name`.
    mark_column_name: Option<String>,
    /// Index into `bindings` corresponding to the mark column. Used to skip
    /// the mark binding when zipping outputs with bindings.
    mark_binding_index: Option<usize>,
}

impl JoinOutputRequests {
    pub(super) fn lower(
        self,
        left: &LoweredNativeInput,
        right: &LoweredNativeInput,
        canonical: CanonicalJoin,
    ) -> Result<JoinOutputColumns, FallbackReason> {
        if self.outputs.is_empty() {
            return hidden_row_count_preserving_output(left, right, canonical);
        }

        // Build an iterator over only the non-mark bindings. When a mark
        // column is present, its binding sits at `mark_binding_index` and
        // must be skipped so that the zip with `outputs` stays aligned.
        let non_mark_bindings: Vec<_> = self
            .bindings
            .iter()
            .enumerate()
            .filter(|(index, _)| self.mark_binding_index.is_none_or(|mark_index| *index != mark_index))
            .map(|(_, binding)| binding)
            .collect();

        let projection = self
            .outputs
            .iter()
            .zip(non_mark_bindings.iter())
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
        Ok(JoinOutputColumns {
            projection,
            schema: self.schema,
            bindings: self.bindings,
            mark_column_name: self.mark_column_name,
        })
    }
}

fn hidden_row_count_preserving_output(
    left: &LoweredNativeInput,
    right: &LoweredNativeInput,
    canonical: CanonicalJoin,
) -> Result<JoinOutputColumns, FallbackReason> {
    const HIDDEN_JOIN_ROW_PRESENCE_COLUMN: &str = "__cudf_join_row_presence";
    let (datafusion_side, input, input_binding) =
        hidden_row_count_input(left, right, canonical).ok_or(FallbackReason::UnsupportedExpression)?;
    let native_side = canonical.native_side_for_datafusion_side(datafusion_side)?;
    let input_ordinal = match native_side {
        NativeJoinSide::Left => 0,
        NativeJoinSide::Right => 1,
    };
    let input_field = input
        .schema
        .field_with_name(input_binding.native_name.as_str())
        .map_err(|_error| FallbackReason::UnsupportedExpression)?;
    let field = input_field.clone().with_name(HIDDEN_JOIN_ROW_PRESENCE_COLUMN);
    let projection = vec![NativeJoinOutputColumn::new(
        native_side,
        NativeColumnRef::input(input_ordinal, input_binding.native_name.as_str(), input_binding.frontend_name.as_str()),
        HIDDEN_JOIN_ROW_PRESENCE_COLUMN.to_string(),
    )];
    let schema = Arc::new(Schema::new(vec![field]));
    let bindings = vec![FieldBinding {
        frontend_index: usize::MAX,
        frontend_name: HIDDEN_JOIN_ROW_PRESENCE_COLUMN.to_string(),
        native_name: HIDDEN_JOIN_ROW_PRESENCE_COLUMN.to_string(),
    }];
    Ok(JoinOutputColumns { projection, schema, bindings, mark_column_name: None })
}

fn hidden_row_count_input<'a>(
    left: &'a LoweredNativeInput,
    right: &'a LoweredNativeInput,
    canonical: CanonicalJoin,
) -> Option<(DataFusionJoinSide, &'a LoweredNativeInput, &'a FieldBinding)> {
    if matches!(
        canonical.native_join_type(),
        nexus_query_engine::plan::JoinType::LeftSemi | nexus_query_engine::plan::JoinType::LeftAnti
    ) {
        return first_binding_for_side(left, right, canonical.datafusion_side_for_native_left());
    }
    first_binding_for_side(left, right, DataFusionJoinSide::Left)
        .or_else(|| first_binding_for_side(left, right, DataFusionJoinSide::Right))
}

fn first_binding_for_side<'a>(
    left: &'a LoweredNativeInput,
    right: &'a LoweredNativeInput,
    side: DataFusionJoinSide,
) -> Option<(DataFusionJoinSide, &'a LoweredNativeInput, &'a FieldBinding)> {
    match side {
        DataFusionJoinSide::Left => left.bindings.first().map(|binding| (DataFusionJoinSide::Left, left, binding)),
        DataFusionJoinSide::Right => right.bindings.first().map(|binding| (DataFusionJoinSide::Right, right, binding)),
        DataFusionJoinSide::None => None,
    }
}

/// One non-mark join output slot expressed in DataFusion input coordinates.
pub(super) struct JoinOutputRequest {
    pub(super) datafusion_side: DataFusionJoinSide,
    pub(super) input_index: usize,
    pub(super) predicate_required: bool,
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

pub(super) fn cross_join_output_requests(
    join: &NormalizedNativeCrossJoin<'_>,
    required_output_fields: &RequiredFields,
) -> Result<JoinOutputRequests, FallbackReason> {
    join_output_requests(
        required_output_fields,
        &join.join.left().schema(),
        &join.join.right().schema(),
        &join.join.schema(),
        None,
        &JoinType::Inner,
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
    let predicate_output_indices = required_output_fields
        .predicate_indices(output_schema.fields().len())?
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let mut outputs = Vec::new();
    let mut mark_column_name: Option<String> = None;
    let mut mark_binding_index: Option<usize> = None;
    for (binding_index, output_index) in
        required_output_fields.requested_indices(output_schema.fields().len())?.into_iter().enumerate()
    {
        let join_schema_index = match projection_indices {
            Some(projection) => *projection.get(output_index).ok_or(FallbackReason::UnsupportedExpression)?,
            None => output_index,
        };
        let (datafusion_side, input_index) =
            datafusion_join_output_side(datafusion_join_type, join_schema_index, left_schema, right_schema)?;
        // The mark column (JoinSide::None) is not a regular output column;
        // the native executor constructs it. Record its native schema name
        // (which may differ from the frontend name after deduplication) and skip.
        if datafusion_side == DataFusionJoinSide::None {
            mark_column_name = Some(bindings[binding_index].native_name.clone());
            mark_binding_index = Some(binding_index);
            continue;
        }
        let input_schema = if datafusion_side == DataFusionJoinSide::Left { left_schema } else { right_schema };
        input_schema.fields().get(input_index).ok_or(FallbackReason::UnsupportedExpression)?;
        outputs.push(JoinOutputRequest {
            datafusion_side,
            input_index,
            predicate_required: predicate_output_indices.contains(&output_index),
        });
    }
    Ok(JoinOutputRequests { outputs, schema, bindings, mark_column_name, mark_binding_index })
}

pub(super) fn datafusion_join_output_side(
    join_type: &JoinType,
    join_schema_index: usize,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
) -> Result<(DataFusionJoinSide, usize), FallbackReason> {
    match join_type {
        JoinType::RightSemi | JoinType::RightAnti => Ok((DataFusionJoinSide::Right, join_schema_index)),
        JoinType::LeftSemi | JoinType::LeftAnti => Ok((DataFusionJoinSide::Left, join_schema_index)),
        // LeftMark output schema (DataFusion): [left_fields..., mark]. The
        // trailing index is the mark column (JoinSide::None); 0..left_len are
        // left-side columns.
        JoinType::LeftMark if join_schema_index >= left_schema.fields().len() => {
            Ok((DataFusionJoinSide::None, join_schema_index))
        },
        JoinType::LeftMark => Ok((DataFusionJoinSide::Left, join_schema_index)),
        // RightMark output schema (DataFusion): [right_fields..., mark]. The
        // trailing index is the mark column (JoinSide::None); 0..right_len are
        // right-side columns. RightMark is canonicalized through input swap in
        // canonical_join, but the DataFusion-original side names are kept here
        // so callers select the correct DataFusion input schema for binding
        // lookups.
        JoinType::RightMark if join_schema_index >= right_schema.fields().len() => {
            Ok((DataFusionJoinSide::None, join_schema_index))
        },
        JoinType::RightMark => Ok((DataFusionJoinSide::Right, join_schema_index)),
        _ if join_schema_index < left_schema.fields().len() => Ok((DataFusionJoinSide::Left, join_schema_index)),
        _ => Ok((DataFusionJoinSide::Right, join_schema_index - left_schema.fields().len())),
    }
}
