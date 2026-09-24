use datafusion::{common::Result, logical_expr::LogicalPlan};
use serde_json::{Map, Value};

use super::{
    CumlOperation, DenseVectorBinding, GpuOptionKind, MlRelationRole, canonicalize_cuml_options,
    error::{invalid_call, invalid_option},
};
use crate::gpu_functions::dense_relation::infer_dense_relation_binding;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CumlRelationBinding {
    relation_role: MlRelationRole,
    id: String,
    features: DenseVectorBinding,
    label: Option<String>,
}

impl CumlRelationBinding {
    pub fn relation_role(&self) -> MlRelationRole {
        self.relation_role
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn features(&self) -> &DenseVectorBinding {
        &self.features
    }
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CumlValidatedOptions {
    canonical_json: String,
    bindings: Vec<CumlRelationBinding>,
}

impl CumlValidatedOptions {
    pub fn canonical_json(&self) -> &str {
        &self.canonical_json
    }
    pub fn bindings(&self) -> &[CumlRelationBinding] {
        &self.bindings
    }
    pub fn n_components(&self) -> Option<usize> {
        let canonical: Value = serde_json::from_str(&self.canonical_json).expect("validated cuML options contain JSON");
        canonical.get("n_components").and_then(Value::as_u64).map(|count| count as usize)
    }

    pub(super) fn named_values(&self, operation: CumlOperation) -> Map<String, Value> {
        let canonical: Value = serde_json::from_str(&self.canonical_json).expect("validated cuML options contain JSON");
        let canonical = canonical.as_object().expect("validated cuML options are an object");
        operation
            .descriptor()
            .options
            .iter()
            .filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
            .filter_map(|option| canonical.get(option.name).cloned().map(|value| (option.name.to_string(), value)))
            .collect()
    }
}

pub(super) fn validate_options(operation: CumlOperation, raw: &str) -> Result<CumlValidatedOptions> {
    let canonical = canonicalize_cuml_options(operation, raw)
        .map_err(|error| invalid_option(operation, error.code(), error.argument(), error.message()))?;
    let object = canonical.as_object().expect("canonical cuML options are an object");
    let bindings = operation.descriptor().relation_roles.iter().copied().map(|role| binding(object, role)).collect();
    Ok(CumlValidatedOptions { canonical_json: canonical.to_string(), bindings })
}

pub(super) fn options_from_named_values(
    operation: CumlOperation,
    inputs: &[LogicalPlan],
    mut values: Map<String, Value>,
) -> Result<String> {
    let roles = operation.descriptor().relation_roles;
    if inputs.len() != roles.len() {
        return Err(invalid_call(
            operation,
            "invalid_relation_count",
            format!("{} requires {} relation argument(s)", operation.descriptor().sql_function_name, roles.len()),
        ));
    }
    for (role, input) in roles.iter().copied().zip(inputs) {
        if values.contains_key(role.as_str()) {
            return Err(invalid_call(
                operation,
                "unknown_argument",
                format!("cuML relation role '{}' cannot appear in the value options object", role.as_str()),
            ));
        }
        values.insert(
            role.as_str().to_string(),
            infer_dense_relation_binding(
                input,
                role.as_str(),
                "cuML",
                || super::error::field_not_found(operation, role.as_str(), "id"),
                |reason, message| invalid_call(operation, reason, message),
            )?,
        );
    }
    Ok(Value::Object(values).to_string())
}

fn binding(object: &Map<String, Value>, role: MlRelationRole) -> CumlRelationBinding {
    let value = object.get(role.as_str()).and_then(Value::as_object).expect("canonical cuML binding");
    let id = value.get("id").and_then(Value::as_str).expect("canonical id").to_string();
    let vector = value.get("vector").and_then(Value::as_object).expect("canonical features");
    let features = if let Some(columns) = vector.get("columns") {
        DenseVectorBinding::Columns(
            columns
                .as_array()
                .expect("columns array")
                .iter()
                .map(|value| value.as_str().expect("column name").to_string())
                .collect(),
        )
    } else {
        DenseVectorBinding::VectorColumn(
            vector.get("column").and_then(Value::as_str).expect("vector column").to_string(),
        )
    };
    CumlRelationBinding {
        relation_role: role,
        id,
        features,
        label: value.get("label").and_then(Value::as_str).map(str::to_owned),
    }
}
