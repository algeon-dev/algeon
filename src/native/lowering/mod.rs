mod aggregate;
mod expr;
mod join;
mod join_output;
mod join_residual;
mod root;
mod sort;
mod source;
mod unary;
mod union;
mod window;

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{Field, Schema, SchemaRef};
use cudf_query_engine::{
    Expr as NativeExpr, GpuResidentSource as NativeGpuResidentSource, NodeId as NativeNodeId,
    ProjectionOutput as NativeProjectionOutput, QueryPlan as NativeQueryPlan,
    QueryPlanBuilder as NativeQueryPlanBuilder, SourceNode as NativeSourceNode,
};
use datafusion::physical_expr::expressions::Column;

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
        NormalizedNativeRoot::Aggregate(root)
            if normalized.single_source_input().is_some_and(|source| source.local_parquet_source_exec().is_some()) =>
        {
            root::lower_slice1_plan(normalized, root)
        },
        NormalizedNativeRoot::Aggregate(root)
            if normalized.single_source_input().is_some_and(|source| {
                source.table_format_scan_exec().is_some()
                    && crate::native::validate::native_zero_column_count_source_is_safe(
                        source.filter,
                        root.pre_aggregate_projection,
                        root.aggregate,
                    )
            }) =>
        {
            root::lower_table_format_zero_column_count_plan(normalized, root)
        },
        NormalizedNativeRoot::Aggregate(root) => root::lower_joined_aggregate_plan(normalized, root),
        NormalizedNativeRoot::Input(root) => root::lower_input_root_plan(normalized, root),
    }
}

pub(super) fn projection_outputs(
    expressions: Vec<NativeExpr>,
    schema: &SchemaRef,
) -> Result<Vec<NativeProjectionOutput>, FallbackReason> {
    if expressions.len() != schema.fields().len() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(expressions
        .into_iter()
        .zip(schema.fields().iter())
        .map(|(expr, field)| NativeProjectionOutput::new(expr, field.as_ref().clone()))
        .collect())
}

/// Native node plus the schema visible to its parent lowering step.
#[derive(Debug)]
pub(super) struct LoweredNativeInput {
    pub(super) node: NativeNodeId,
    /// Executable schema carried by the native query engine. Field names in
    /// this schema are unique within the node output.
    pub(super) schema: SchemaRef,
    /// Adapter-side mapping from the DataFusion physical output schema to the
    /// executable native schema above.
    pub(super) bindings: Vec<FieldBinding>,
}

impl LoweredNativeInput {
    pub(super) fn with_bindings(node: NativeNodeId, schema: SchemaRef, bindings: Vec<FieldBinding>) -> Self {
        Self { node, schema, bindings }
    }

    pub(super) fn binding_for_column(&self, column: &Column) -> Result<&FieldBinding, FallbackReason> {
        self.binding_for_frontend_index(column.index())
    }

    pub(super) fn binding_for_frontend_index(&self, frontend_index: usize) -> Result<&FieldBinding, FallbackReason> {
        self.bindings
            .iter()
            .find(|binding| binding.frontend_index == frontend_index)
            .ok_or(FallbackReason::UnsupportedExpression)
    }

    pub(super) fn frontend_schema(&self) -> Result<SchemaRef, FallbackReason> {
        if self.schema.fields().len() != self.bindings.len() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        let fields = self
            .schema
            .fields()
            .iter()
            .zip(self.bindings.iter())
            .map(|(field, binding)| rename_field(field.as_ref(), binding.frontend_name.as_str()))
            .collect::<Vec<_>>();
        Ok(Arc::new(Schema::new(fields)))
    }
}

#[derive(Debug, Clone)]
pub(super) struct FieldBinding {
    pub(super) frontend_index: usize,
    pub(super) frontend_name: String,
    pub(super) native_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RequiredFields {
    All,
    Only(BTreeSet<usize>),
}

impl RequiredFields {
    pub(super) fn all() -> Self {
        Self::All
    }

    pub(super) fn only(required: BTreeSet<usize>) -> Self {
        Self::Only(required)
    }

    pub(super) fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    pub(super) fn contains(&self, index: usize) -> bool {
        match self {
            Self::All => true,
            Self::Only(required) => required.contains(&index),
        }
    }

    pub(super) fn requested_indices(&self, field_count: usize) -> Result<Vec<usize>, FallbackReason> {
        match self {
            Self::All => Ok((0..field_count).collect()),
            Self::Only(required) => {
                if required.iter().any(|index| *index >= field_count) {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                Ok(required.iter().copied().collect())
            },
        }
    }
}

pub(super) fn executable_schema_and_bindings(
    frontend_schema: &SchemaRef,
    required: &RequiredFields,
) -> Result<(SchemaRef, Vec<FieldBinding>), FallbackReason> {
    let indices = required.requested_indices(frontend_schema.fields().len())?;
    let mut name_counts = BTreeMap::<&str, usize>::new();
    for index in &indices {
        let field = frontend_schema.fields().get(*index).ok_or(FallbackReason::UnsupportedExpression)?;
        *name_counts.entry(field.name().as_str()).or_default() += 1;
    }

    let frontend_names = indices
        .iter()
        .map(|index| {
            frontend_schema
                .fields()
                .get(*index)
                .map(|field| field.name().clone())
                .ok_or(FallbackReason::UnsupportedExpression)
        })
        .collect::<Result<BTreeSet<_>, _>>()?;

    let mut used_native_names = BTreeSet::new();
    let mut fields = Vec::with_capacity(indices.len());
    let mut bindings = Vec::with_capacity(indices.len());
    for frontend_index in indices {
        let field = frontend_schema.fields().get(frontend_index).ok_or(FallbackReason::UnsupportedExpression)?;
        let frontend_name = field.name().clone();
        let native_name = if name_counts.get(frontend_name.as_str()).copied().unwrap_or_default() == 1
            && !used_native_names.contains(frontend_name.as_str())
        {
            frontend_name.clone()
        } else {
            unique_synthetic_field_name(frontend_index, frontend_name.as_str(), &frontend_names, &used_native_names)
        };
        used_native_names.insert(native_name.clone());
        fields.push(rename_field(field.as_ref(), native_name.as_str()));
        bindings.push(FieldBinding { frontend_index, frontend_name, native_name });
    }
    Ok((Arc::new(Schema::new(fields)), bindings))
}

/// Recursively lowers a normalized input subtree.
///
/// `required_output_fields` is the set of output field identities requested by
/// the parent lowering step. `All` means no explicit parent pruning. `Only({})`
/// means the parent explicitly needs zero output fields.
fn lower_native_input(
    input: &NormalizedNativeInput<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    match input {
        NormalizedNativeInput::Source(source) => source::lower_source_input(source, builder, required_output_fields),
        NormalizedNativeInput::GpuResidentSource(source) => {
            lower_gpu_resident_source_input(source, builder, required_output_fields)
        },
        NormalizedNativeInput::Filter(filter) => unary::lower_filter_input(filter, builder, required_output_fields),
        NormalizedNativeInput::Projection(projection) => {
            unary::lower_projection_input(projection, builder, required_output_fields)
        },
        NormalizedNativeInput::Limit(limit) => {
            let input = lower_native_input(&limit.input, builder, required_output_fields)?;
            let node = builder.add_limit(input.node, limit.spec, Arc::clone(&input.schema));
            Ok(LoweredNativeInput::with_bindings(node, input.schema, input.bindings))
        },
        NormalizedNativeInput::Aggregate(aggregate) => {
            aggregate::lower_aggregate_input(aggregate, builder, required_output_fields)
        },
        NormalizedNativeInput::Sort(sort) => window::lower_sort_input(sort, builder, required_output_fields),
        NormalizedNativeInput::Window(window) => window::lower_window_input(window, builder, required_output_fields),
        NormalizedNativeInput::HashJoin(join) => join::lower_hash_join_input(join, builder, required_output_fields),
        NormalizedNativeInput::CrossJoin(join) => join::lower_cross_join_input(join, builder, required_output_fields),
        NormalizedNativeInput::NestedLoopJoin(join) => {
            join::lower_nested_loop_join_input(join, builder, required_output_fields)
        },
        NormalizedNativeInput::Union(union) => union::lower_union_input(union, builder, required_output_fields),
    }
}

fn lower_gpu_resident_source_input(
    source: &super::normalize::NormalizedNativeGpuResidentSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let full_schema = source.source_exec.schema();
    let full_required = RequiredFields::all();
    let (source_schema, source_bindings) = executable_schema_and_bindings(&full_schema, &full_required)?;
    let source_node = builder.add_source(NativeSourceNode::GpuResident(NativeGpuResidentSource::new(
        source.input_name.as_str(),
        Arc::clone(&source_schema),
    )));
    if required_output_fields.is_all() {
        return Ok(LoweredNativeInput::with_bindings(source_node, source_schema, source_bindings));
    }

    let (projected_schema, projected_bindings) = executable_schema_and_bindings(&full_schema, required_output_fields)?;
    let expressions = projected_bindings
        .iter()
        .map(|projected| {
            let input = source_bindings
                .iter()
                .find(|binding| binding.frontend_index == projected.frontend_index)
                .ok_or(FallbackReason::UnsupportedExpression)?;
            Ok(NativeProjectionOutput::new(
                NativeExpr::column(input.native_name.as_str()),
                rename_field(
                    source_schema
                        .fields()
                        .get(input.frontend_index)
                        .ok_or(FallbackReason::UnsupportedExpression)?
                        .as_ref(),
                    projected.native_name.as_str(),
                ),
            ))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let projected_node = builder.add_projection(source_node, expressions);
    Ok(LoweredNativeInput::with_bindings(projected_node, projected_schema, projected_bindings))
}

fn rename_field(field: &Field, name: &str) -> Field {
    field.clone().with_name(name)
}

fn unique_synthetic_field_name(
    ordinal: usize,
    frontend_name: &str,
    frontend_names: &BTreeSet<String>,
    used_native_names: &BTreeSet<String>,
) -> String {
    let sanitized = sanitize_field_name(frontend_name);
    let base = format!("__cudf_field_{ordinal}_{sanitized}");
    if !frontend_names.contains(base.as_str()) && !used_native_names.contains(base.as_str()) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if !frontend_names.contains(candidate.as_str()) && !used_native_names.contains(candidate.as_str()) {
            return candidate;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn sanitize_field_name(name: &str) -> String {
    let sanitized =
        name.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' }).collect::<String>();
    if sanitized.is_empty() { "field".to_string() } else { sanitized }
}
