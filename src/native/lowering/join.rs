use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};
use cudf_query_engine::{
    ColumnRef as NativeColumnRef, Expr as NativeExpr, JoinKey as NativeJoinKey,
    JoinNullEquality as NativeJoinNullEquality, JoinOutputColumn as NativeJoinOutputColumn, JoinSide as NativeJoinSide,
    JoinSpec as NativeJoinSpec, JoinType as NativeJoinType, QueryPlanBuilder as NativeQueryPlanBuilder,
};
use datafusion::{
    common::{JoinSide as DataFusionJoinSide, NullEquality},
    logical_expr::JoinType,
    physical_expr::{PhysicalExpr, expressions::Column},
    physical_plan::{ExecutionPlan, joins::NestedLoopJoinExec},
};

use super::{
    LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    join_residual::{JoinResidualMaterialization, LoweredJoinResidual, lower_join_filter},
    lower_native_input, projection_outputs,
};
use crate::{
    native::normalize::{NormalizedNativeHashJoin, NormalizedNativeNestedLoopJoin},
    policy::FallbackReason,
};

/// Native join shape after mapping DataFusion-only join variants to the native
/// side convention. `swap_inputs` records whether DataFusion left/right inputs
/// are reversed before building the native join.
#[derive(Debug, Clone, Copy)]
pub(super) struct CanonicalJoin {
    join_type: NativeJoinType,
    swap_inputs: bool,
}

impl CanonicalJoin {
    const fn new(join_type: NativeJoinType, swap_inputs: bool) -> Self {
        Self { join_type, swap_inputs }
    }

    fn order_inputs(
        self,
        left: LoweredNativeInput,
        right: LoweredNativeInput,
    ) -> (LoweredNativeInput, LoweredNativeInput) {
        if self.swap_inputs { (right, left) } else { (left, right) }
    }

    pub(super) fn native_input_index_for_datafusion_side(
        self,
        side: DataFusionJoinSide,
    ) -> Result<usize, FallbackReason> {
        // Residual predicates are expressed in DataFusion sides, while the
        // native IR addresses ordered join inputs. This is the shared mapping.
        match (self.swap_inputs, side) {
            (false, DataFusionJoinSide::Left) | (true, DataFusionJoinSide::Right) => Ok(0),
            (false, DataFusionJoinSide::Right) | (true, DataFusionJoinSide::Left) => Ok(1),
            (_, DataFusionJoinSide::None) => Err(FallbackReason::UnsupportedExpression),
        }
    }

    fn native_side_for_datafusion_side(self, side: DataFusionJoinSide) -> Result<NativeJoinSide, FallbackReason> {
        match self.native_input_index_for_datafusion_side(side)? {
            0 => Ok(NativeJoinSide::Left),
            1 => Ok(NativeJoinSide::Right),
            _ => unreachable!("canonical native join input index is always 0 or 1"),
        }
    }

    const fn build_side(self) -> NativeJoinSide {
        // Left, semi, and anti joins preserve the native left side, so their
        // default build side is right. Inner/full/cross keep the native default.
        match self.join_type {
            NativeJoinType::Left | NativeJoinType::LeftSemi | NativeJoinType::LeftAnti => NativeJoinSide::Right,
            NativeJoinType::Inner | NativeJoinType::Full => NativeJoinSide::Left,
            NativeJoinType::Cross => NativeJoinSide::Left,
        }
    }
}

fn add_join_residual_materializations(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    materializations: &[JoinResidualMaterialization],
) -> Result<LoweredNativeInput, FallbackReason> {
    // Residual casts that cannot live in the predicate AST are appended as
    // synthetic pre-join projection columns on the original DataFusion side.
    // Canonical input ordering happens after this step.
    if materializations.is_empty() {
        return Ok(input);
    }

    let mut expressions =
        input.schema.fields().iter().map(|field| NativeExpr::column(field.name())).collect::<Vec<_>>();
    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut bindings = input.bindings;
    for materialization in materializations {
        expressions.push(materialization.expr.clone());
        fields.push(materialization.field.clone());
        bindings.push(super::FieldBinding {
            frontend_index: usize::MAX - bindings.len(),
            frontend_name: materialization.alias.clone(),
            native_name: materialization.alias.clone(),
        });
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

pub(super) fn lower_hash_join_input(
    join: &NormalizedNativeHashJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    // Flow: canonicalize the join type, derive DataFusion-output-preserving
    // projection, collect keys/residual requirements, lower both children, add
    // residual materializations, then order inputs for the native spec.
    let canonical = canonical_join(join.join.join_type())?;
    let output_requests = hash_join_output_requests(join, required_output_fields)?;
    let residual = join
        .join
        .filter()
        .map(|filter| {
            lower_join_filter(filter, &join.join.left().schema(), &join.join.right().schema(), canonical, None)
        })
        .transpose()?;
    let mut left_required = BTreeSet::<usize>::new();
    let mut right_required = BTreeSet::<usize>::new();
    let key_indices = join
        .join
        .on()
        .iter()
        .map(|(left, right)| {
            let left_schema = join.join.left().schema();
            let right_schema = join.join.right().schema();
            let left_index = join_key_column_index(left, &left_schema)?;
            let right_index = join_key_column_index(right, &right_schema)?;
            left_required.insert(left_index);
            right_required.insert(right_index);
            Ok((left_index, right_index))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    for output in &output_requests.outputs {
        match output.datafusion_side {
            DataFusionJoinSide::Left => {
                left_required.insert(output.input_index);
            },
            DataFusionJoinSide::Right => {
                right_required.insert(output.input_index);
            },
            DataFusionJoinSide::None => {},
        }
    }
    if let Some(residual) = &residual {
        left_required.extend(residual.left_required.iter().copied());
        right_required.extend(residual.right_required.iter().copied());
    }
    let left = lower_native_input(&join.left, builder, &RequiredFields::only(left_required))?;
    let right = lower_native_input(&join.right, builder, &RequiredFields::only(right_required))?;
    let residual = if join.join.filter().is_some() {
        join.join
            .filter()
            .map(|filter| {
                lower_join_filter(
                    filter,
                    &join.join.left().schema(),
                    &join.join.right().schema(),
                    canonical,
                    Some(super::join_residual::JoinResidualInputBindings { left: &left, right: &right }),
                )
            })
            .transpose()?
    } else {
        residual
    };
    let keys = key_indices
        .into_iter()
        .map(|(left_index, right_index)| {
            let left_binding = left.binding_for_frontend_index(left_index)?;
            let right_binding = right.binding_for_frontend_index(right_index)?;
            if canonical.swap_inputs {
                Ok(NativeJoinKey::new(
                    NativeColumnRef::input(0, right_binding.native_name.as_str(), right_binding.frontend_name.as_str()),
                    NativeColumnRef::input(1, left_binding.native_name.as_str(), left_binding.frontend_name.as_str()),
                ))
            } else {
                Ok(NativeJoinKey::new(
                    NativeColumnRef::input(0, left_binding.native_name.as_str(), left_binding.frontend_name.as_str()),
                    NativeColumnRef::input(1, right_binding.native_name.as_str(), right_binding.frontend_name.as_str()),
                ))
            }
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let output_columns = output_requests.lower(&left, &right, canonical)?;
    let (left, right) = if let Some(residual) = &residual {
        (
            add_join_residual_materializations(
                builder,
                left,
                residual.materializations.side(DataFusionJoinSide::Left)?,
            )?,
            add_join_residual_materializations(
                builder,
                right,
                residual.materializations.side(DataFusionJoinSide::Right)?,
            )?,
        )
    } else {
        (left, right)
    };
    let (native_left, native_right) = canonical.order_inputs(left, right);
    let build_side = native_join_build_side(canonical, keys.len(), residual.is_some());
    let mut spec = NativeJoinSpec::new(canonical.join_type, keys)
        .with_output_projection(output_columns.projection)
        .with_null_equality(lower_join_null_equality(join.join.null_equality()))
        .with_build_side(build_side);
    if let Some(residual) = residual {
        spec = spec.with_residual_predicate(residual.predicate);
    }
    let node = builder.add_join(native_left.node, native_right.node, spec, Arc::clone(&output_columns.schema));
    Ok(LoweredNativeInput::with_bindings(node, output_columns.schema, output_columns.bindings))
}

pub(super) fn lower_nested_loop_join_input(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    // Nested-loop lowering shares the residual/materialization path with hash
    // joins, but has no equality keys. Cross joins are represented as keyless
    // native joins without residual predicates.
    let canonical = nested_loop_join_canonical(join, required_output_fields)?;
    let residual = nested_loop_join_residual(join.join, canonical)?;
    let output_requests = nested_loop_join_output_requests(join, required_output_fields)?;
    let mut left_required = BTreeSet::<usize>::new();
    let mut right_required = BTreeSet::<usize>::new();
    for output in &output_requests.outputs {
        match output.datafusion_side {
            DataFusionJoinSide::Left => {
                left_required.insert(output.input_index);
            },
            DataFusionJoinSide::Right => {
                right_required.insert(output.input_index);
            },
            DataFusionJoinSide::None => {},
        }
    }
    if let Some(residual) = &residual {
        left_required.extend(residual.left_required.iter().copied());
        right_required.extend(residual.right_required.iter().copied());
    }
    let left = lower_native_input(&join.left, builder, &RequiredFields::only(left_required))?;
    let right = lower_native_input(&join.right, builder, &RequiredFields::only(right_required))?;
    let residual = match join.join.filter() {
        Some(filter) => Some(lower_join_filter(
            filter,
            &join.join.left().schema(),
            &join.join.right().schema(),
            canonical,
            Some(super::join_residual::JoinResidualInputBindings { left: &left, right: &right }),
        )?),
        None => residual,
    };
    let output_columns = output_requests.lower(&left, &right, canonical)?;
    let (left, right) = if let Some(residual) = &residual {
        (
            add_join_residual_materializations(
                builder,
                left,
                residual.materializations.side(DataFusionJoinSide::Left)?,
            )?,
            add_join_residual_materializations(
                builder,
                right,
                residual.materializations.side(DataFusionJoinSide::Right)?,
            )?,
        )
    } else {
        (left, right)
    };
    let (native_left, native_right) = canonical.order_inputs(left, right);
    let build_side = native_join_build_side(canonical, 0, residual.is_some());
    let mut spec = NativeJoinSpec::new(canonical.join_type, Vec::new())
        .with_output_projection(output_columns.projection)
        .with_build_side(build_side);
    if let Some(residual) = residual {
        spec = spec.with_residual_predicate(residual.predicate);
    }
    let node = builder.add_join(native_left.node, native_right.node, spec, Arc::clone(&output_columns.schema));
    Ok(LoweredNativeInput::with_bindings(node, output_columns.schema, output_columns.bindings))
}

fn native_join_build_side(canonical: CanonicalJoin, key_count: usize, has_residual: bool) -> NativeJoinSide {
    // Keyless semi/anti joins with residuals need the preserved side as the
    // build side so the native executor can evaluate the condition without
    // emitting build-side rows.
    if key_count == 0
        && has_residual
        && matches!(canonical.join_type, NativeJoinType::LeftSemi | NativeJoinType::LeftAnti)
    {
        NativeJoinSide::Left
    } else {
        canonical.build_side()
    }
}

fn nested_loop_join_canonical(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    required_output_fields: &RequiredFields,
) -> Result<CanonicalJoin, FallbackReason> {
    // Predicate-free inner nested-loop joins are native cross joins. Predicate
    // inner joins may swap when only DataFusion-right columns are required,
    // which keeps the output side compact while preserving DataFusion schema.
    if join.join.filter().is_none() && matches!(join.join.join_type(), JoinType::Inner) {
        return Ok(CanonicalJoin::new(NativeJoinType::Cross, false));
    }
    let canonical = canonical_join(join.join.join_type())?;
    if matches!(join.join.join_type(), JoinType::Inner)
        && nested_loop_join_required_outputs_are_right_only(join, required_output_fields)?
    {
        Ok(CanonicalJoin::new(NativeJoinType::Inner, true))
    } else {
        Ok(canonical)
    }
}

fn nested_loop_join_residual(
    join: &NestedLoopJoinExec,
    canonical: CanonicalJoin,
) -> Result<Option<LoweredJoinResidual>, FallbackReason> {
    // Non-inner nested-loop joins require a residual predicate today; a
    // predicate-free inner join has already canonicalized to cross join.
    if let Some(filter) = join.filter() {
        let residual = lower_join_filter(filter, &join.left().schema(), &join.right().schema(), canonical, None)?;
        return Ok(Some(residual));
    }
    if matches!(join.join_type(), JoinType::Inner) {
        return Ok(None);
    }
    Err(FallbackReason::UnsupportedOperator)
}

fn nested_loop_join_required_outputs_are_right_only(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    required_output_fields: &RequiredFields,
) -> Result<bool, FallbackReason> {
    let mut saw_output = false;
    for output_index in required_output_fields.requested_indices(join.join.schema().fields().len())? {
        saw_output = true;
        let join_schema_index = match join.join.projection().as_deref() {
            Some(projection) => *projection.get(output_index).ok_or(FallbackReason::UnsupportedExpression)?,
            None => output_index,
        };
        let (datafusion_side, _) =
            datafusion_join_output_side(join.join.join_type(), join_schema_index, &join.join.left().schema())?;
        if datafusion_side != DataFusionJoinSide::Right {
            return Ok(false);
        }
    }
    Ok(saw_output)
}

/// Join output projection plus the exact DataFusion output schema requested by
/// the parent. Native input ordering may change, but this schema must not.
struct JoinOutputColumns {
    projection: Vec<NativeJoinOutputColumn>,
    schema: SchemaRef,
    bindings: Vec<super::FieldBinding>,
}

struct JoinOutputRequests {
    outputs: Vec<JoinOutputRequest>,
    schema: SchemaRef,
    bindings: Vec<super::FieldBinding>,
}

impl JoinOutputRequests {
    fn lower(
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

struct JoinOutputRequest {
    datafusion_side: DataFusionJoinSide,
    input_index: usize,
}

fn hash_join_output_requests(
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

fn nested_loop_join_output_requests(
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

fn join_key_column_index(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<usize, FallbackReason> {
    let Some(column) = expr.as_any().downcast_ref::<Column>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    input_schema.fields().get(column.index()).map(|_| column.index()).ok_or(FallbackReason::UnsupportedExpression)
}

fn canonical_join(join_type: &JoinType) -> Result<CanonicalJoin, FallbackReason> {
    // Native join variants are left-oriented for right/semi/anti cases. Right
    // joins become left joins with swapped inputs, and right semi/anti become
    // left semi/anti with swapped inputs.
    match join_type {
        JoinType::Inner => Ok(CanonicalJoin::new(NativeJoinType::Inner, false)),
        JoinType::Left => Ok(CanonicalJoin::new(NativeJoinType::Left, false)),
        JoinType::Right => Ok(CanonicalJoin::new(NativeJoinType::Left, true)),
        JoinType::Full => Ok(CanonicalJoin::new(NativeJoinType::Full, false)),
        JoinType::LeftSemi => Ok(CanonicalJoin::new(NativeJoinType::LeftSemi, false)),
        JoinType::RightSemi => Ok(CanonicalJoin::new(NativeJoinType::LeftSemi, true)),
        JoinType::LeftAnti => Ok(CanonicalJoin::new(NativeJoinType::LeftAnti, false)),
        JoinType::RightAnti => Ok(CanonicalJoin::new(NativeJoinType::LeftAnti, true)),
        _ => Err(FallbackReason::UnsupportedOperator),
    }
}

fn datafusion_join_output_side(
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

fn lower_join_null_equality(null_equality: NullEquality) -> NativeJoinNullEquality {
    match null_equality {
        NullEquality::NullEqualsNothing => NativeJoinNullEquality::NullsDoNotMatch,
        NullEquality::NullEqualsNull => NativeJoinNullEquality::NullsMatch,
    }
}
