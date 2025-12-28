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
    LoweredNativeInput,
    expr::resolved_column_name,
    join_residual::{JoinResidualMaterialization, LoweredJoinResidual, lower_join_filter},
    lower_native_input,
};
use crate::{
    native::normalize::{NormalizedNativeHashJoin, NormalizedNativeNestedLoopJoin},
    policy::FallbackReason,
};

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

    const fn datafusion_side_for_native_side(self, side: NativeJoinSide) -> DataFusionJoinSide {
        match (self.swap_inputs, side) {
            (false, NativeJoinSide::Left) | (true, NativeJoinSide::Right) => DataFusionJoinSide::Left,
            (false, NativeJoinSide::Right) | (true, NativeJoinSide::Left) => DataFusionJoinSide::Right,
        }
    }

    const fn build_side(self) -> NativeJoinSide {
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
    if materializations.is_empty() {
        return Ok(input);
    }

    let mut expressions =
        input.schema.fields().iter().map(|field| NativeExpr::column(field.name())).collect::<Vec<_>>();
    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    for materialization in materializations {
        expressions.push(materialization.expr.clone());
        fields.push(materialization.field.clone());
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input.node, expressions, Arc::clone(&schema));
    Ok(LoweredNativeInput { node, schema })
}

pub(super) fn lower_hash_join_input(
    join: &NormalizedNativeHashJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    let canonical = canonical_join(join.join.join_type())?;
    let output_columns = hash_join_output_columns(join, required_output_columns, canonical)?;
    let residual = join
        .join
        .filter()
        .map(|filter| lower_join_filter(filter, &join.join.left().schema(), &join.join.right().schema(), canonical))
        .transpose()?;
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    let keys = join
        .join
        .on()
        .iter()
        .map(|(left, right)| {
            let left_schema = join.join.left().schema();
            let right_schema = join.join.right().schema();
            let left_name = join_key_column_name(left, &left_schema)?;
            let right_name = join_key_column_name(right, &right_schema)?;
            left_required.insert(left_name.to_string());
            right_required.insert(right_name.to_string());
            if canonical.swap_inputs {
                Ok(NativeJoinKey::new(
                    NativeColumnRef::input(0, right_name, right_name),
                    NativeColumnRef::input(1, left_name, left_name),
                ))
            } else {
                Ok(NativeJoinKey::new(
                    NativeColumnRef::input(0, left_name, left_name),
                    NativeColumnRef::input(1, right_name, right_name),
                ))
            }
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    for output in &output_columns.projection {
        match canonical.datafusion_side_for_native_side(output.side) {
            DataFusionJoinSide::Left => {
                left_required.insert(output.column.name().to_string());
            },
            DataFusionJoinSide::Right => {
                right_required.insert(output.column.name().to_string());
            },
            DataFusionJoinSide::None => {},
        }
    }
    if let Some(residual) = &residual {
        left_required.extend(residual.left_required.iter().cloned());
        right_required.extend(residual.right_required.iter().cloned());
    }
    let left_required = left_required.into_iter().collect::<Vec<_>>();
    let right_required = right_required.into_iter().collect::<Vec<_>>();
    let left = lower_native_input(&join.left, builder, &left_required)?;
    let right = lower_native_input(&join.right, builder, &right_required)?;
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
    Ok(LoweredNativeInput { node, schema: output_columns.schema })
}

pub(super) fn lower_nested_loop_join_input(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    let canonical = nested_loop_join_canonical(join, required_output_columns)?;
    let residual = nested_loop_join_residual(join.join, canonical)?;
    let output_columns = nested_loop_join_output_columns(join, required_output_columns, canonical)?;
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    for output in &output_columns.projection {
        match canonical.datafusion_side_for_native_side(output.side) {
            DataFusionJoinSide::Left => {
                left_required.insert(output.column.name().to_string());
            },
            DataFusionJoinSide::Right => {
                right_required.insert(output.column.name().to_string());
            },
            DataFusionJoinSide::None => {},
        }
    }
    if let Some(residual) = &residual {
        left_required.extend(residual.left_required.iter().cloned());
        right_required.extend(residual.right_required.iter().cloned());
    }
    let left_required = left_required.into_iter().collect::<Vec<_>>();
    let right_required = right_required.into_iter().collect::<Vec<_>>();
    let left = lower_native_input(&join.left, builder, &left_required)?;
    let right = lower_native_input(&join.right, builder, &right_required)?;
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
    Ok(LoweredNativeInput { node, schema: output_columns.schema })
}

fn native_join_build_side(canonical: CanonicalJoin, key_count: usize, has_residual: bool) -> NativeJoinSide {
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
    required_output_columns: &[String],
) -> Result<CanonicalJoin, FallbackReason> {
    if join.join.filter().is_none() && matches!(join.join.join_type(), JoinType::Inner) {
        return Ok(CanonicalJoin::new(NativeJoinType::Cross, false));
    }
    let canonical = canonical_join(join.join.join_type())?;
    if matches!(join.join.join_type(), JoinType::Inner)
        && nested_loop_join_required_outputs_are_right_only(join, required_output_columns)?
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
    if let Some(filter) = join.filter() {
        let residual = lower_join_filter(filter, &join.left().schema(), &join.right().schema(), canonical)?;
        return Ok(Some(residual));
    }
    if matches!(join.join_type(), JoinType::Inner) {
        return Ok(None);
    }
    Err(FallbackReason::UnsupportedOperator)
}

fn nested_loop_join_required_outputs_are_right_only(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    required_output_columns: &[String],
) -> Result<bool, FallbackReason> {
    let required = required_output_columns.iter().collect::<BTreeSet<_>>();
    let mut saw_output = false;
    for (output_index, field) in join.join.schema().fields().iter().enumerate() {
        if !required.is_empty() && !required.contains(field.name()) {
            continue;
        }
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

struct JoinOutputColumns {
    projection: Vec<NativeJoinOutputColumn>,
    schema: SchemaRef,
}

fn hash_join_output_columns(
    join: &NormalizedNativeHashJoin<'_>,
    required_output_columns: &[String],
    canonical: CanonicalJoin,
) -> Result<JoinOutputColumns, FallbackReason> {
    join_output_columns(
        required_output_columns,
        &join.join.left().schema(),
        &join.join.right().schema(),
        &join.join.schema(),
        join.join.projection.as_deref(),
        join.join.join_type(),
        canonical,
    )
}

fn nested_loop_join_output_columns(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    required_output_columns: &[String],
    canonical: CanonicalJoin,
) -> Result<JoinOutputColumns, FallbackReason> {
    join_output_columns(
        required_output_columns,
        &join.join.left().schema(),
        &join.join.right().schema(),
        &join.join.schema(),
        join.join.projection().as_deref(),
        join.join.join_type(),
        canonical,
    )
}

fn join_output_columns(
    required_output_columns: &[String],
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    output_schema: &SchemaRef,
    projection_indices: Option<&[usize]>,
    datafusion_join_type: &JoinType,
    canonical: CanonicalJoin,
) -> Result<JoinOutputColumns, FallbackReason> {
    let required = required_output_columns.iter().collect::<BTreeSet<_>>();
    let mut output_fields = Vec::new();
    let mut projection = Vec::new();
    for (output_index, field) in output_schema.fields().iter().enumerate() {
        if !required.is_empty() && !required.contains(field.name()) {
            continue;
        }
        let join_schema_index = match projection_indices {
            Some(projection) => *projection.get(output_index).ok_or(FallbackReason::UnsupportedExpression)?,
            None => output_index,
        };
        let (datafusion_side, input_index) =
            datafusion_join_output_side(datafusion_join_type, join_schema_index, left_schema)?;
        let side = canonical.native_side_for_datafusion_side(datafusion_side)?;
        let input_schema = if datafusion_side == DataFusionJoinSide::Left { left_schema } else { right_schema };
        let input_field = input_schema.fields().get(input_index).ok_or(FallbackReason::UnsupportedExpression)?;
        let output_name = field.name().clone();
        let column = NativeColumnRef::input(
            match side {
                NativeJoinSide::Left => 0,
                NativeJoinSide::Right => 1,
            },
            input_field.name(),
            output_name.clone(),
        );
        projection.push(NativeJoinOutputColumn::new(side, column, output_name));
        output_fields.push(field.as_ref().clone());
    }
    Ok(JoinOutputColumns { projection, schema: Arc::new(Schema::new(output_fields)) })
}

fn join_key_column_name<'a>(
    expr: &'a Arc<dyn PhysicalExpr>,
    input_schema: &'a SchemaRef,
) -> Result<&'a str, FallbackReason> {
    let Some(column) = expr.as_any().downcast_ref::<Column>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    resolved_column_name(column, input_schema)
}

fn canonical_join(join_type: &JoinType) -> Result<CanonicalJoin, FallbackReason> {
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
