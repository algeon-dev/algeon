use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::{
    common::{JoinSide as DataFusionJoinSide, NullEquality},
    logical_expr::JoinType,
    physical_expr::{PhysicalExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan,
        joins::{NestedLoopJoinExec, PartitionMode},
    },
};
use nexus_query_engine::{
    datatypes::{is_supported_list_carry_type, is_timestamp_with_timezone},
    expr::{ColumnRef as NativeColumnRef, Expr as NativeExpr},
    plan::{
        JoinKey as NativeJoinKey, JoinNullEquality as NativeJoinNullEquality, JoinSide as NativeJoinSide,
        JoinSpec as NativeJoinSpec, JoinType as NativeJoinType, QueryPlanBuilder as NativeQueryPlanBuilder,
    },
};

use super::{
    LoweredNativeInput, RequiredFields,
    expr::{collect_required_field_indices, lower_physical_expr_with_bindings, physical_expr_field},
    join_output::{
        JoinOutputRequests, cross_join_output_requests, datafusion_join_output_side, hash_join_output_requests,
        nested_loop_join_output_requests,
    },
    join_residual::{JoinResidualMaterialization, LoweredJoinResidual, lower_join_filter},
    lower_native_input, projection_outputs,
};
use crate::{
    capability::CapabilityReason,
    native::{
        expression_contract::validate_join_residual,
        normalize::{NormalizedNativeCrossJoin, NormalizedNativeHashJoin, NormalizedNativeNestedLoopJoin},
    },
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
    pub(super) const fn new(join_type: NativeJoinType, swap_inputs: bool) -> Self {
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
    ) -> Result<usize, CapabilityReason> {
        // Residual predicates are expressed in DataFusion sides, while the
        // native IR addresses ordered join inputs. This is the shared mapping.
        match (self.swap_inputs, side) {
            (false, DataFusionJoinSide::Left) | (true, DataFusionJoinSide::Right) => Ok(0),
            (false, DataFusionJoinSide::Right) | (true, DataFusionJoinSide::Left) => Ok(1),
            (_, DataFusionJoinSide::None) => Err(CapabilityReason::UnsupportedExpression),
        }
    }

    pub(super) fn native_side_for_datafusion_side(
        self,
        side: DataFusionJoinSide,
    ) -> Result<NativeJoinSide, CapabilityReason> {
        match self.native_input_index_for_datafusion_side(side)? {
            0 => Ok(NativeJoinSide::Left),
            1 => Ok(NativeJoinSide::Right),
            _ => unreachable!("canonical native join input index is always 0 or 1"),
        }
    }

    pub(super) fn datafusion_side_for_native_left(self) -> DataFusionJoinSide {
        if self.swap_inputs { DataFusionJoinSide::Right } else { DataFusionJoinSide::Left }
    }

    pub(super) const fn native_join_type(self) -> NativeJoinType {
        self.join_type
    }

    const fn build_side(self) -> NativeJoinSide {
        // Use right-build orientation whenever join semantics allow it so
        // left-deep joined streams can remain on the probe side.
        match self.join_type {
            NativeJoinType::Left | NativeJoinType::LeftSemi | NativeJoinType::LeftAnti | NativeJoinType::LeftMark => {
                NativeJoinSide::Right
            },
            NativeJoinType::Inner => NativeJoinSide::Right,
            NativeJoinType::Full => NativeJoinSide::Left,
            NativeJoinType::Cross => NativeJoinSide::Left,
        }
    }
}

fn add_join_residual_materializations(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    materializations: &[JoinResidualMaterialization],
) -> Result<LoweredNativeInput, CapabilityReason> {
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

fn add_join_residual_materializations_by_side(
    builder: &mut NativeQueryPlanBuilder,
    left: LoweredNativeInput,
    right: LoweredNativeInput,
    residual: &LoweredJoinResidual,
) -> Result<(LoweredNativeInput, LoweredNativeInput), CapabilityReason> {
    Ok((
        add_join_residual_materializations(builder, left, residual.materializations.side(DataFusionJoinSide::Left)?)?,
        add_join_residual_materializations(builder, right, residual.materializations.side(DataFusionJoinSide::Right)?)?,
    ))
}

#[derive(Debug, Clone, Copy)]
enum JoinKeySide {
    Left,
    Right,
}

struct PreparedJoinKeyExpr {
    column_index: Option<usize>,
    synthetic_alias: Option<String>,
}

struct JoinKeyMaterialization {
    expr: Arc<dyn PhysicalExpr>,
    alias: String,
    field: arrow::datatypes::Field,
}

struct PreparedJoinKeys {
    keys: Vec<(PreparedJoinKeyExpr, PreparedJoinKeyExpr)>,
    left_materializations: Vec<JoinKeyMaterialization>,
    right_materializations: Vec<JoinKeyMaterialization>,
}

fn prepare_join_key_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    side: JoinKeySide,
    key_index: usize,
    required: &mut BTreeSet<usize>,
    materializations: &mut Vec<JoinKeyMaterialization>,
) -> Result<PreparedJoinKeyExpr, CapabilityReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let field = input_schema.fields().get(column.index()).ok_or(CapabilityReason::UnsupportedExpression)?;
        if is_timestamp_with_timezone(field.data_type()) || is_supported_list_carry_type(field.data_type()) {
            return Err(CapabilityReason::UnsupportedType);
        }
        required.insert(column.index());
        return Ok(PreparedJoinKeyExpr { column_index: Some(column.index()), synthetic_alias: None });
    }

    collect_required_field_indices(expr, input_schema, required)?;
    let alias = unique_synthetic_join_key_alias(side, key_index, input_schema, materializations);
    let field = physical_expr_field(expr, input_schema, alias.as_str())?;
    if is_timestamp_with_timezone(field.data_type()) || is_supported_list_carry_type(field.data_type()) {
        return Err(CapabilityReason::UnsupportedType);
    }
    materializations.push(JoinKeyMaterialization { expr: Arc::clone(expr), alias: alias.clone(), field });
    Ok(PreparedJoinKeyExpr { column_index: None, synthetic_alias: Some(alias) })
}

fn prepare_join_keys(
    join: &NormalizedNativeHashJoin<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    left_predicate_required: &mut BTreeSet<usize>,
    right_predicate_required: &mut BTreeSet<usize>,
) -> Result<PreparedJoinKeys, CapabilityReason> {
    let left_schema = join.join.left().schema();
    let right_schema = join.join.right().schema();
    let mut left_materializations = Vec::new();
    let mut right_materializations = Vec::new();
    let mut keys = Vec::with_capacity(join.join.on().len());
    for (key_index, (left, right)) in join.join.on().iter().enumerate() {
        let left = prepare_join_key_expr(
            left,
            &left_schema,
            JoinKeySide::Left,
            key_index,
            left_required,
            &mut left_materializations,
        )?;
        left_predicate_required.extend(left_required.iter().copied());
        let right = prepare_join_key_expr(
            right,
            &right_schema,
            JoinKeySide::Right,
            key_index,
            right_required,
            &mut right_materializations,
        )?;
        right_predicate_required.extend(right_required.iter().copied());
        keys.push((left, right));
    }
    Ok(PreparedJoinKeys { keys, left_materializations, right_materializations })
}

fn add_join_key_materializations(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    input_schema: &SchemaRef,
    materializations: &[JoinKeyMaterialization],
) -> Result<LoweredNativeInput, CapabilityReason> {
    if materializations.is_empty() {
        return Ok(input);
    }

    let mut expressions =
        input.schema.fields().iter().map(|field| NativeExpr::column(field.name())).collect::<Vec<_>>();
    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut bindings = input.bindings;
    for materialization in materializations {
        let expr = lower_physical_expr_with_bindings(&materialization.expr, input_schema, Some(&bindings))?;
        expressions.push(expr.alias(materialization.alias.as_str()));
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

fn unique_synthetic_join_key_alias(
    side: JoinKeySide,
    key_index: usize,
    input_schema: &SchemaRef,
    materializations: &[JoinKeyMaterialization],
) -> String {
    let side = match side {
        JoinKeySide::Left => "left",
        JoinKeySide::Right => "right",
    };
    for attempt in 0.. {
        let alias = format!("__nexus_native_join_key_{side}_{key_index}_{attempt}");
        if input_schema.field_with_name(alias.as_str()).is_err()
            && materializations.iter().all(|materialization| materialization.alias != alias)
        {
            return alias;
        }
    }
    unreachable!("unbounded synthetic join key alias search must find a free name")
}

fn lowered_join_key_ref<'a>(
    key: &'a PreparedJoinKeyExpr,
    input: &'a LoweredNativeInput,
) -> Result<(&'a str, &'a str), CapabilityReason> {
    if let Some(alias) = key.synthetic_alias.as_deref() {
        return Ok((alias, alias));
    }
    let column_index = key.column_index.ok_or(CapabilityReason::UnsupportedExpression)?;
    let binding = input.binding_for_frontend_index(column_index)?;
    Ok((binding.native_name.as_str(), binding.frontend_name.as_str()))
}

fn lower_prepared_join_keys(
    prepared: &[(PreparedJoinKeyExpr, PreparedJoinKeyExpr)],
    left: &LoweredNativeInput,
    right: &LoweredNativeInput,
    canonical: CanonicalJoin,
) -> Result<Vec<NativeJoinKey>, CapabilityReason> {
    prepared
        .iter()
        .map(|(left_key, right_key)| {
            let (left_native, left_frontend) = lowered_join_key_ref(left_key, left)?;
            let (right_native, right_frontend) = lowered_join_key_ref(right_key, right)?;
            if canonical.swap_inputs {
                Ok(NativeJoinKey::new(
                    NativeColumnRef::input(0, right_native, right_frontend),
                    NativeColumnRef::input(1, left_native, left_frontend),
                ))
            } else {
                Ok(NativeJoinKey::new(
                    NativeColumnRef::input(0, left_native, left_frontend),
                    NativeColumnRef::input(1, right_native, right_frontend),
                ))
            }
        })
        .collect()
}

fn extend_required_fields_from_join_outputs(
    output_requests: &JoinOutputRequests,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    left_predicate_required: &mut BTreeSet<usize>,
    right_predicate_required: &mut BTreeSet<usize>,
) {
    for output in &output_requests.outputs {
        match output.datafusion_side {
            DataFusionJoinSide::Left => {
                left_required.insert(output.input_index);
                if output.predicate_required {
                    left_predicate_required.insert(output.input_index);
                }
            },
            DataFusionJoinSide::Right => {
                right_required.insert(output.input_index);
                if output.predicate_required {
                    right_predicate_required.insert(output.input_index);
                }
            },
            DataFusionJoinSide::None => {},
        }
    }
}

fn extend_required_fields_from_residual(
    residual: Option<&LoweredJoinResidual>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
) {
    if let Some(residual) = residual {
        left_required.extend(residual.left_required.iter().copied());
        right_required.extend(residual.right_required.iter().copied());
    }
}

pub(super) fn lower_hash_join_input(
    join: &NormalizedNativeHashJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, CapabilityReason> {
    // Flow: canonicalize the join type, derive DataFusion-output-preserving
    // projection, collect keys/residual requirements, lower both children, add
    // residual materializations, then order inputs for the native spec.
    let canonical = canonical_join(join.join.join_type())?;
    let output_requests = hash_join_output_requests(join, required_output_fields)?;
    let residual = join
        .join
        .filter()
        .map(|filter| {
            let left_schema = join.join.left().schema();
            let right_schema = join.join.right().schema();
            let contract = validate_join_residual(filter, &left_schema, &right_schema)?;
            lower_join_filter(&contract, canonical, None)
        })
        .transpose()?;
    let mut left_required = BTreeSet::<usize>::new();
    let mut right_required = BTreeSet::<usize>::new();
    let mut left_predicate_required = BTreeSet::<usize>::new();
    let mut right_predicate_required = BTreeSet::<usize>::new();
    let prepared_keys = prepare_join_keys(
        join,
        &mut left_required,
        &mut right_required,
        &mut left_predicate_required,
        &mut right_predicate_required,
    )?;
    extend_required_fields_from_join_outputs(
        &output_requests,
        &mut left_required,
        &mut right_required,
        &mut left_predicate_required,
        &mut right_predicate_required,
    );
    extend_required_fields_from_residual(residual.as_ref(), &mut left_required, &mut right_required);
    extend_required_fields_from_residual(
        residual.as_ref(),
        &mut left_predicate_required,
        &mut right_predicate_required,
    );
    let left = lower_native_input(
        &join.left,
        builder,
        &RequiredFields::only_with_predicates(left_required, left_predicate_required),
    )?;
    let right = lower_native_input(
        &join.right,
        builder,
        &RequiredFields::only_with_predicates(right_required, right_predicate_required),
    )?;
    let residual = if join.join.filter().is_some() {
        join.join
            .filter()
            .map(|filter| {
                let left_schema = join.join.left().schema();
                let right_schema = join.join.right().schema();
                let contract = validate_join_residual(filter, &left_schema, &right_schema)?;
                lower_join_filter(
                    &contract,
                    canonical,
                    Some(super::join_residual::JoinResidualInputBindings { left: &left, right: &right }),
                )
            })
            .transpose()?
    } else {
        residual
    };
    let output_columns = output_requests.lower(&left, &right, canonical)?;
    let left =
        add_join_key_materializations(builder, left, &join.join.left().schema(), &prepared_keys.left_materializations)?;
    let right = add_join_key_materializations(
        builder,
        right,
        &join.join.right().schema(),
        &prepared_keys.right_materializations,
    )?;
    let (left, right) = if let Some(residual) = &residual {
        add_join_residual_materializations_by_side(builder, left, right, residual)?
    } else {
        (left, right)
    };
    let keys = lower_prepared_join_keys(&prepared_keys.keys, &left, &right, canonical)?;
    let inferred_lookup_side = unique_lookup_bounded_side(builder, join, canonical, &left, &right, &keys);
    let (native_left, native_right) = canonical.order_inputs(left, right);
    let build_side = native_join_build_side(canonical, keys.len(), residual.is_some());
    let datafusion_order_preserved_side = datafusion_hash_join_order_preserved_side(join, build_side);
    let mut spec = NativeJoinSpec::new(canonical.join_type, keys)
        .with_output_projection(output_columns.projection)
        .with_null_equality(lower_join_null_equality(join.join.null_equality()))
        .with_build_side(build_side)
        .with_datafusion_order_preserved_side(datafusion_order_preserved_side);
    if let Some(residual) = residual {
        spec = spec.with_residual_predicate(residual.predicate);
    }
    if canonical.join_type == NativeJoinType::LeftMark
        && let Some(mark_name) = output_columns.mark_column_name.as_deref()
    {
        spec = spec.with_mark_column_name(mark_name);
    }
    if matches!(canonical.join_type, NativeJoinType::Inner | NativeJoinType::Left)
        && let Some(bounded_side) = inferred_lookup_side
    {
        spec = spec.with_at_most_one_match_per_bounded_row(bounded_side);
    }
    let node = builder.add_join(native_left.node, native_right.node, spec, Arc::clone(&output_columns.schema));
    let output = LoweredNativeInput::with_bindings(node, output_columns.schema, output_columns.bindings);
    if let Some(fetch) = join.fetch {
        let node = builder.add_limit(output.node, fetch, Arc::clone(&output.schema));
        Ok(LoweredNativeInput::with_bindings(node, output.schema, output.bindings))
    } else {
        Ok(output)
    }
}

fn unique_lookup_bounded_side(
    builder: &NativeQueryPlanBuilder,
    join: &NormalizedNativeHashJoin<'_>,
    canonical: CanonicalJoin,
    left: &LoweredNativeInput,
    right: &LoweredNativeInput,
    keys: &[nexus_query_engine::plan::JoinKey],
) -> Option<NativeJoinSide> {
    if join.join.filter().is_some() || join.join.on().is_empty() {
        return None;
    }
    let left_keys = keys.iter().map(|key| key.left.name().to_owned()).collect::<BTreeSet<_>>();
    let right_keys = keys.iter().map(|key| key.right.name().to_owned()).collect::<BTreeSet<_>>();
    if left_keys.len() != keys.len() || right_keys.len() != keys.len() {
        return None;
    }

    let left_unique = builder.output_is_unique_on(left.node, &left_keys).unwrap_or(false);
    let right_unique = builder.output_is_unique_on(right.node, &right_keys).unwrap_or(false);
    let bounded_datafusion_side = match join.join.join_type() {
        JoinType::Inner if right_unique => DataFusionJoinSide::Left,
        JoinType::Inner if left_unique => DataFusionJoinSide::Right,
        JoinType::Left if right_unique => DataFusionJoinSide::Left,
        JoinType::Right if left_unique => DataFusionJoinSide::Right,
        _ => return None,
    };
    canonical.native_side_for_datafusion_side(bounded_datafusion_side).ok()
}

pub(super) fn lower_nested_loop_join_input(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, CapabilityReason> {
    // Nested-loop lowering shares the residual/materialization path with hash
    // joins, but has no equality keys. Cross joins are represented as keyless
    // native joins without residual predicates.
    let canonical = nested_loop_join_canonical(join, required_output_fields)?;
    let residual = nested_loop_join_residual(join.join, canonical)?;
    let output_requests = nested_loop_join_output_requests(join, required_output_fields)?;
    let mut left_required = BTreeSet::<usize>::new();
    let mut right_required = BTreeSet::<usize>::new();
    extend_required_fields_from_residual(residual.as_ref(), &mut left_required, &mut right_required);
    let mut left_predicate_required = BTreeSet::<usize>::new();
    let mut right_predicate_required = BTreeSet::<usize>::new();
    extend_required_fields_from_join_outputs(
        &output_requests,
        &mut left_required,
        &mut right_required,
        &mut left_predicate_required,
        &mut right_predicate_required,
    );
    extend_required_fields_from_residual(
        residual.as_ref(),
        &mut left_predicate_required,
        &mut right_predicate_required,
    );
    let left = lower_native_input(
        &join.left,
        builder,
        &RequiredFields::only_with_predicates(left_required, left_predicate_required),
    )?;
    let right = lower_native_input(
        &join.right,
        builder,
        &RequiredFields::only_with_predicates(right_required, right_predicate_required),
    )?;
    let residual = match join.join.filter() {
        Some(filter) => {
            let left_schema = join.join.left().schema();
            let right_schema = join.join.right().schema();
            let contract = validate_join_residual(filter, &left_schema, &right_schema)?;
            Some(lower_join_filter(
                &contract,
                canonical,
                Some(super::join_residual::JoinResidualInputBindings { left: &left, right: &right }),
            )?)
        },
        None => residual,
    };
    let output_columns = output_requests.lower(&left, &right, canonical)?;
    let (left, right) = if let Some(residual) = &residual {
        add_join_residual_materializations_by_side(builder, left, right, residual)?
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

pub(super) fn lower_cross_join_input(
    join: &NormalizedNativeCrossJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let canonical = CanonicalJoin::new(NativeJoinType::Cross, false);
    let output_requests = cross_join_output_requests(join, required_output_fields)?;
    let mut left_required = BTreeSet::<usize>::new();
    let mut right_required = BTreeSet::<usize>::new();
    let mut left_predicate_required = BTreeSet::<usize>::new();
    let mut right_predicate_required = BTreeSet::<usize>::new();
    extend_required_fields_from_join_outputs(
        &output_requests,
        &mut left_required,
        &mut right_required,
        &mut left_predicate_required,
        &mut right_predicate_required,
    );
    let left = lower_native_input(
        &join.left,
        builder,
        &RequiredFields::only_with_predicates(left_required, left_predicate_required),
    )?;
    let right = lower_native_input(
        &join.right,
        builder,
        &RequiredFields::only_with_predicates(right_required, right_predicate_required),
    )?;
    let output_columns = output_requests.lower(&left, &right, canonical)?;
    let spec = NativeJoinSpec::new(NativeJoinType::Cross, Vec::new())
        .with_output_projection(output_columns.projection)
        .with_build_side(NativeJoinSide::Left);
    let node = builder.add_join(left.node, right.node, spec, Arc::clone(&output_columns.schema));
    Ok(LoweredNativeInput::with_bindings(node, output_columns.schema, output_columns.bindings))
}

fn native_join_build_side(canonical: CanonicalJoin, key_count: usize, has_residual: bool) -> NativeJoinSide {
    // Keyless semi/anti joins with residuals need the preserved side as the
    // build side so the native executor can evaluate the condition without
    // emitting build-side rows.
    if key_count == 0
        && has_residual
        && matches!(canonical.join_type, NativeJoinType::LeftSemi | NativeJoinType::LeftAnti | NativeJoinType::LeftMark)
    {
        NativeJoinSide::Left
    } else {
        canonical.build_side()
    }
}

fn datafusion_hash_join_order_preserved_side(
    join: &NormalizedNativeHashJoin<'_>,
    build_side: NativeJoinSide,
) -> NativeJoinSide {
    match join.join.partition_mode() {
        PartitionMode::CollectLeft => opposite_native_side(build_side),
        PartitionMode::Partitioned | PartitionMode::Auto => NativeJoinSide::Right,
    }
}

const fn opposite_native_side(side: NativeJoinSide) -> NativeJoinSide {
    match side {
        NativeJoinSide::Left => NativeJoinSide::Right,
        NativeJoinSide::Right => NativeJoinSide::Left,
    }
}

fn nested_loop_join_canonical(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    required_output_fields: &RequiredFields,
) -> Result<CanonicalJoin, CapabilityReason> {
    // Predicate-free inner nested-loop joins are native cross joins.
    // Predicate-free left/right joins are also native cross joins when the null-supplying
    // input is provably a scalar aggregate subtree: a scalar aggregate emits
    // exactly one row, so there is no unmatched side to null-extend. Predicate
    // inner joins may swap when only DataFusion-right columns are required,
    // which keeps the output side compact while preserving DataFusion schema.
    if join.join.filter().is_none() && matches!(join.join.join_type(), JoinType::Inner) {
        return Ok(CanonicalJoin::new(NativeJoinType::Cross, false));
    }
    if join.join.filter().is_none()
        && matches!(join.join.join_type(), JoinType::Left)
        && normalized_input_is_scalar_single_row(&join.right)
    {
        return Ok(CanonicalJoin::new(NativeJoinType::Cross, false));
    }
    if join.join.filter().is_none()
        && matches!(join.join.join_type(), JoinType::Right)
        && normalized_input_is_scalar_single_row(&join.left)
    {
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
) -> Result<Option<LoweredJoinResidual>, CapabilityReason> {
    // Non-inner nested-loop joins require a residual predicate unless the
    // canonicalizer has proven that the predicate-free join is a cross join.
    if let Some(filter) = join.filter() {
        let left_schema = join.left().schema();
        let right_schema = join.right().schema();
        let contract = validate_join_residual(filter, &left_schema, &right_schema)?;
        let residual = lower_join_filter(&contract, canonical, None)?;
        return Ok(Some(residual));
    }
    if matches!(join.join_type(), JoinType::Inner)
        || matches!(canonical.join_type, NativeJoinType::Cross | NativeJoinType::Left | NativeJoinType::Full)
    {
        return Ok(None);
    }
    Err(CapabilityReason::UnsupportedOperator)
}

fn normalized_input_is_scalar_single_row(input: &crate::native::normalize::NormalizedNativeInput<'_>) -> bool {
    use crate::native::normalize::NormalizedNativeInput;

    match input {
        NormalizedNativeInput::Projection(projection) => normalized_input_is_scalar_single_row(&projection.input),
        NormalizedNativeInput::Limit(limit) => {
            limit.spec.skip == 0 && limit.spec.fetch != Some(0) && normalized_input_is_scalar_single_row(&limit.input)
        },
        NormalizedNativeInput::Aggregate(aggregate) => aggregate.aggregate.group_expr().expr().is_empty(),
        NormalizedNativeInput::GpuResidentSource(_) => false,
        NormalizedNativeInput::LiteralSource(source) => {
            matches!(source, crate::native::normalize::NormalizedNativeLiteralSource::PlaceholderRow(_))
        },
        NormalizedNativeInput::Source(_)
        | NormalizedNativeInput::Filter(_)
        | NormalizedNativeInput::Sort(_)
        | NormalizedNativeInput::Window(_)
        | NormalizedNativeInput::HashJoin(_)
        | NormalizedNativeInput::CrossJoin(_)
        | NormalizedNativeInput::NestedLoopJoin(_)
        | NormalizedNativeInput::Union(_) => false,
    }
}

fn nested_loop_join_required_outputs_are_right_only(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    required_output_fields: &RequiredFields,
) -> Result<bool, CapabilityReason> {
    let mut saw_output = false;
    for output_index in required_output_fields.requested_indices(join.join.schema().fields().len())? {
        saw_output = true;
        let join_schema_index = match join.join.projection().as_deref() {
            Some(projection) => *projection.get(output_index).ok_or(CapabilityReason::UnsupportedExpression)?,
            None => output_index,
        };
        let (datafusion_side, _) = datafusion_join_output_side(
            join.join.join_type(),
            join_schema_index,
            &join.join.left().schema(),
            &join.join.right().schema(),
        )?;
        if datafusion_side != DataFusionJoinSide::Right {
            return Ok(false);
        }
    }
    Ok(saw_output)
}

fn canonical_join(join_type: &JoinType) -> Result<CanonicalJoin, CapabilityReason> {
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
        JoinType::LeftMark => Ok(CanonicalJoin::new(NativeJoinType::LeftMark, false)),
        JoinType::RightMark => Ok(CanonicalJoin::new(NativeJoinType::LeftMark, true)),
    }
}

fn lower_join_null_equality(null_equality: NullEquality) -> NativeJoinNullEquality {
    match null_equality {
        NullEquality::NullEqualsNothing => NativeJoinNullEquality::NullsDoNotMatch,
        NullEquality::NullEqualsNull => NativeJoinNullEquality::NullsMatch,
    }
}
