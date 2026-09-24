//! DataFusion contracts shared by dense-relation SQL families.

use arrow::datatypes::SchemaRef;
use datafusion::{
    catalog::Session,
    common::{DataFusionError, Result, TableReference},
    datasource::provider_as_source,
    execution::SessionState,
    logical_expr::{FetchType, LogicalPlan, LogicalPlanBuilder, SkipType},
};
use query_engine::plan::{GpuOptionDefinition, GpuOptionKind};
use serde_json::{Map, Value, json};

use super::{NamedArgumentDefinition, NamedArgumentKind};
use crate::error::{DataFusionErrorExt, ErrorCode};

pub(crate) fn infer_dense_relation_binding(
    input: &LogicalPlan,
    role: &'static str,
    family: &'static str,
    field_not_found: impl FnOnce() -> DataFusionError,
    invalid_call: impl Fn(&'static str, String) -> DataFusionError,
) -> Result<Value> {
    let schema = input.schema();
    schema.field_with_unqualified_name("id").map_err(|_| field_not_found())?;
    let supervised = family == "cuML" && matches!(role, "training" | "predict");
    let has_label = schema.field_with_unqualified_name("label").is_ok();
    if supervised && role == "training" && !has_label {
        return Err(invalid_call("missing_label", "cuML training relation requires a 'label' column".to_string()));
    }
    if supervised && role == "predict" && has_label {
        return Err(invalid_call("predict_relation_has_label", "cuML predict relation must omit 'label'".to_string()));
    }
    let feature_names = schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .filter(|name| *name != "id" && !(supervised && *name == "label"))
        .collect::<Vec<_>>();
    if feature_names.is_empty() {
        return Err(invalid_call(
            "missing_vector_features",
            format!("{family} {role} relation requires at least one feature field after 'id'"),
        ));
    }
    let vector = if feature_names.contains(&"vector") {
        if feature_names.len() != 1 {
            return Err(invalid_call(
                "vector_field_with_additional_features",
                format!("{family} {role} relation cannot combine 'vector' with other feature fields"),
            ));
        }
        json!({"column": "vector"})
    } else {
        json!({"columns": feature_names})
    };
    if supervised && role == "training" {
        Ok(json!({"id": "id", "vector": vector, "label": "label"}))
    } else {
        Ok(json!({"id": "id", "vector": vector}))
    }
}

pub(crate) struct ResolvedSchemaOnlyInput {
    pub(crate) plan: LogicalPlan,
    pub(crate) schema: SchemaRef,
}

pub(crate) async fn resolve_schema_only_input(
    session: &dyn Session,
    name: &str,
    table_reference: TableReference,
) -> std::result::Result<ResolvedSchemaOnlyInput, String> {
    let state = session.as_any().downcast_ref::<SessionState>().ok_or_else(|| {
        "GPU validation requires a DataFusion SessionState so named relations can be resolved".to_string()
    })?;
    let catalog_options = &state.config_options().catalog;
    let resolved = table_reference.clone().resolve(&catalog_options.default_catalog, &catalog_options.default_schema);
    let catalog =
        state.catalog_list().catalog(&resolved.catalog).ok_or_else(|| format!("table '{name}' was not found"))?;
    let schema = catalog.schema(&resolved.schema).ok_or_else(|| format!("table '{name}' was not found"))?;
    let provider = schema
        .table(&resolved.table)
        .await
        .map_err(|error| format!("could not resolve table '{name}': {error}"))?
        .ok_or_else(|| format!("table '{name}' was not found"))?;
    let provider_schema = provider.schema();
    let plan = LogicalPlanBuilder::scan(table_reference, provider_as_source(provider), None)
        .map_err(|error| format!("could not construct schema-only input for table '{name}': {error}"))?
        .build()
        .map_err(|error| format!("could not construct schema-only input for table '{name}': {error}"))?;
    Ok(ResolvedSchemaOnlyInput { plan, schema: provider_schema })
}

pub(crate) fn dense_relation_argument_definitions(
    roles: impl Iterator<Item = &'static str>,
    options: &[GpuOptionDefinition],
) -> Vec<NamedArgumentDefinition> {
    roles
        .map(|name| NamedArgumentDefinition { name, kind: NamedArgumentKind::Relation, required: true })
        .chain(options.iter().filter_map(|option| {
            (!matches!(option.kind, GpuOptionKind::RelationBinding)).then_some(NamedArgumentDefinition {
                name: option.name,
                kind: NamedArgumentKind::Value,
                required: option.required,
            })
        }))
        .collect()
}

pub(crate) fn known_row_count(plan: &LogicalPlan) -> Option<usize> {
    match plan {
        LogicalPlan::EmptyRelation(empty) => Some(usize::from(empty.produce_one_row)),
        LogicalPlan::Values(values) => Some(values.values.len()),
        LogicalPlan::Projection(projection) => known_row_count(&projection.input),
        LogicalPlan::SubqueryAlias(alias) => known_row_count(&alias.input),
        LogicalPlan::Limit(limit) => {
            let input_rows = match limit.get_skip_type().ok()? {
                SkipType::Literal(skip) => known_row_count(&limit.input)?.saturating_sub(skip),
                SkipType::UnsupportedExpr => return None,
            };
            match limit.get_fetch_type().ok()? {
                FetchType::Literal(fetch) => Some(input_rows.min(fetch.unwrap_or(input_rows))),
                FetchType::UnsupportedExpr => None,
            }
        },
        _ => None,
    }
}

pub(crate) fn validation_request_schema<'a>(
    roles: impl Iterator<Item = &'a str> + Clone,
    options_schema: &Value,
) -> Value {
    let relation_properties = roles
        .clone()
        .map(|role| {
            (
                role.to_string(),
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["table"],
                    "properties": {"table": {"type": "string"}}
                }),
            )
        })
        .collect::<Map<_, _>>();
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": false,
        "required": ["relations", "options"],
        "properties": {
            "relations": {
                "type": "object",
                "additionalProperties": false,
                "required": roles.collect::<Vec<_>>(),
                "properties": relation_properties,
            },
            "options": options_schema,
        }
    })
}

/// Projects the typed rejection and its argument or relation locator into validation details.
pub(crate) fn validation_error(
    error: &DataFusionError,
    details: &mut Value,
    fallback_code: &'static str,
) -> (String, String) {
    let Some(error) = error.find_algeon_error() else {
        details["path"] = Value::String("options".to_string());
        return (fallback_code.to_string(), error.to_string());
    };
    let code = match error.code() {
        ErrorCode::FieldNotFound => "column_not_found",
        ErrorCode::UnsupportedType => "unsupported_type",
        _ => error.facts().get("reason").unwrap_or(fallback_code),
    };
    let facts = error.facts();
    let path = match (facts.get("argument"), facts.get("relation_role"), facts.get("field")) {
        (Some(argument), _, _) => format!("options.{argument}"),
        (None, Some(role), Some(field)) => format!("relations.{role}.fields.{field}"),
        (None, Some(role), None) => format!("relations.{role}"),
        (None, None, _) => "options".to_string(),
    };
    details["path"] = Value::String(path);
    for key in ["argument", "relation_role", "field", "actual_type", "expected_type"] {
        if let Some(value) = facts.get(key) {
            details[key] = Value::String(value.to_string());
        }
    }
    (code.to_string(), error.message().to_string())
}

pub(crate) fn schema_json(schema: &SchemaRef) -> Value {
    json!({
        "fields": schema.fields().iter().map(|field| json!({
            "name": field.name(),
            "data_type": format!("{:?}", field.data_type()),
            "nullable": field.is_nullable(),
            "metadata": field.metadata(),
        })).collect::<Vec<_>>(),
        "metadata": schema.metadata(),
    })
}
