use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::physical_expr::expressions::Column;
use nexus_query_engine::NodeId as NativeNodeId;

use crate::policy::FallbackReason;

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

pub(super) fn rename_field(field: &Field, name: &str) -> Field {
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
