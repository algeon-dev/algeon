//! cuVS adapter for the common GPU-function discovery and validation catalog.
//!
//! This module deliberately reuses [`CuvsLogicalNode`] for static validation.
//! Named validation relations become schema-only `TableScan` logical inputs, so
//! no relation scan, GPU allocation, or cuVS operation is performed here.

use arrow::datatypes::SchemaRef;
use datafusion::{
    catalog::Session,
    common::{DataFusionError, TableReference},
    datasource::provider_as_source,
    execution::{SessionState, context::SessionConfig},
    logical_expr::{LogicalPlan, LogicalPlanBuilder},
};
use serde_json::{Map, Value, json};

use super::{
    CUVS_OPERATION_REGISTRY, CuvsLogicalNode, CuvsOperation, CuvsOptionKind, CuvsOutputContract, CuvsRelationRole,
    options::options_from_named_values,
};
use crate::{
    error::{DataFusionErrorExt, ErrorCode},
    gpu_functions::{
        GpuCallRequest, GpuCallValidation, GpuFunctionAvailability, GpuFunctionDescriptor, GpuFunctionFamily,
    },
};

const CUVS_FAMILY_ID: &str = "cuvs";
const CUVS_PROVIDER: &str = "cuvs";

/// cuVS discovery and static-validation adapter installed into the common GPU
/// function catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CuvsGpuFunctionFamily {
    availability: GpuFunctionAvailability,
}

impl CuvsGpuFunctionFamily {
    pub(crate) fn new(availability: GpuFunctionAvailability) -> Self {
        Self { availability }
    }
}

/// Determine cuVS planning availability for the session being built.
pub(crate) fn cuvs_gpu_function_availability(_config: Option<&SessionConfig>) -> GpuFunctionAvailability {
    if !cfg!(feature = "cuvs") {
        return GpuFunctionAvailability::Unavailable { reason: "required_feature_disabled".to_string() };
    }
    GpuFunctionAvailability::Available
}

#[async_trait::async_trait]
impl GpuFunctionFamily for CuvsGpuFunctionFamily {
    fn family_id(&self) -> &'static str {
        CUVS_FAMILY_ID
    }

    fn descriptors(&self) -> Vec<GpuFunctionDescriptor> {
        CUVS_OPERATION_REGISTRY
            .iter()
            .map(|descriptor| descriptor_for_operation(descriptor.operation, self.availability.clone()))
            .collect()
    }

    async fn validate(
        &self,
        session: &dyn Session,
        function_name: &str,
        request: &GpuCallRequest,
    ) -> GpuCallValidation {
        let Some(operation) = CuvsOperation::from_sql_function_name(function_name) else {
            return GpuCallValidation::invalid(
                "unknown_function",
                format!("unknown cuVS operation function '{function_name}'"),
                function_name,
                request.normalized_json(),
            );
        };
        if !self.availability.is_available() {
            return unavailable_validation(operation, request, &self.availability);
        }

        let mut inputs = Vec::with_capacity(operation.descriptor().relation_roles.len());
        let mut relation_details = Map::new();
        for role in operation.descriptor().relation_roles {
            let role_name = role.as_str();
            let relation = request.relations.get(role_name).expect("common catalog validates relation roles");
            match resolve_schema_only_input(session, &relation.table, relation.reference.clone()).await {
                Ok(resolved) => {
                    relation_details.insert(
                        role_name.to_string(),
                        json!({
                            "table": relation.table,
                            "resolved": true,
                            "schema": schema_json(&resolved.schema),
                        }),
                    );
                    inputs.push(resolved.plan);
                },
                Err(message) => {
                    relation_details.insert(role_name.to_string(), json!({"table": relation.table, "resolved": false}));
                    let mut details = base_details(operation, relation_details);
                    details["path"] = Value::String(format!("relations.{role_name}"));
                    details["relation_role"] = Value::String(role_name.to_string());
                    return GpuCallValidation {
                        valid: false,
                        error_code: Some("table_not_found".to_string()),
                        message,
                        function_name: operation.descriptor().sql_function_name.to_string(),
                        relations_resolved: false,
                        normalized_call: request.normalized_json(),
                        output_schema: json!({}),
                        would_execute_gpu: false,
                        details,
                    };
                },
            }
        }

        let option_values = request.options.as_object().cloned().unwrap_or_default();
        let options_json = match options_from_named_values(operation, &inputs, option_values) {
            Ok(options_json) => options_json,
            Err(error) => {
                let mut details = base_details(operation, relation_details);
                let (error_code, message) = validation_error(&error, &mut details);
                return GpuCallValidation {
                    valid: false,
                    error_code: Some(error_code),
                    message,
                    function_name: operation.descriptor().sql_function_name.to_string(),
                    relations_resolved: true,
                    normalized_call: request.normalized_json(),
                    output_schema: json!({}),
                    would_execute_gpu: false,
                    details,
                };
            },
        };
        match CuvsLogicalNode::try_new(operation, inputs, &options_json) {
            Ok(node) => {
                let canonical_options: Value = serde_json::from_str(node.validated_options().canonical_json())
                    .expect("canonical cuVS options are valid JSON");
                let mut normalized = request.clone();
                normalized.options = canonical_value_options(operation, &canonical_options);
                let mut details = base_details(operation, relation_details);
                details["output_contract"] =
                    Value::String(output_contract_name(operation.descriptor().output_contract).to_string());
                details["validated_options"] = normalized.options.clone();
                GpuCallValidation {
                    valid: true,
                    error_code: None,
                    message: "valid".to_string(),
                    function_name: operation.descriptor().sql_function_name.to_string(),
                    relations_resolved: true,
                    normalized_call: normalized.normalized_json(),
                    output_schema: schema_json(&node.output_schema()),
                    would_execute_gpu: true,
                    details,
                }
            },
            Err(error) => {
                let mut details = base_details(operation, relation_details);
                let (error_code, message) = validation_error(&error, &mut details);
                GpuCallValidation {
                    valid: false,
                    error_code: Some(error_code),
                    message,
                    function_name: operation.descriptor().sql_function_name.to_string(),
                    relations_resolved: true,
                    normalized_call: request.normalized_json(),
                    output_schema: json!({}),
                    would_execute_gpu: false,
                    details,
                }
            },
        }
    }
}

fn descriptor_for_operation(operation: CuvsOperation, availability: GpuFunctionAvailability) -> GpuFunctionDescriptor {
    let descriptor = operation.descriptor();
    let relation_roles = descriptor
        .relation_roles
        .iter()
        .map(|role| {
            json!({
                "name": role.as_str(),
                "required": true,
                "reference": {"type": "table"},
                "fields": {
                    "id": {
                        "required": true,
                        "nullable": false,
                        "data_types": ["Int32", "Int64", "Utf8", "LargeUtf8", "Utf8View"]
                    },
                    "features": {
                        "element_types": descriptor.vector_element_support.iter().map(|element| element.as_str()).collect::<Vec<_>>(),
                        "one_of": [
                            {
                                "shape": "wide",
                                "rule": "Every non-id field is a non-null feature of a supported element type; projection order defines vector dimension order."
                            },
                            {
                                "shape": "list",
                                "field": "vector",
                                "rule": "The non-null vector field is FixedSizeList, List, or LargeList of a supported element type, with no other non-id fields."
                            }
                        ]
                    }
                },
                "description": role.description(),
            })
        })
        .collect::<Vec<_>>();
    let options_schema = value_options_schema(operation);
    let validation_request_schema = validation_request_schema(descriptor.relation_roles, &options_schema);
    GpuFunctionDescriptor {
        function_name: descriptor.sql_function_name.to_string(),
        provider: CUVS_PROVIDER.to_string(),
        signature: signature(operation),
        summary: descriptor.summary.to_string(),
        relation_roles: Value::Array(relation_roles),
        argument_descriptions: argument_descriptions(operation),
        options_schema,
        result_schemas: result_schemas(operation),
        validation_request_schema,
        examples: examples(operation),
        limitations: limitations(operation),
        availability,
    }
}

fn signature(operation: CuvsOperation) -> String {
    let descriptor = operation.descriptor();
    let arguments = descriptor
        .relation_roles
        .iter()
        .map(|role| format!("{} => (relation subquery)", role.as_str()))
        .chain(
            descriptor
                .options
                .iter()
                .filter(|option| !matches!(option.kind, CuvsOptionKind::RelationBinding))
                .map(|option| format!("{} => value", option.name)),
        )
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({arguments})", descriptor.sql_function_name)
}

fn value_options_schema(operation: CuvsOperation) -> Value {
    let mut schema = operation.descriptor().options_json_schema();
    let object = schema.as_object_mut().expect("cuVS options schema is an object");
    let properties = object
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .expect("cuVS options schema properties are an object");
    for role in operation.descriptor().relation_roles {
        properties.remove(role.as_str());
    }
    let required =
        object.get_mut("required").and_then(Value::as_array_mut).expect("cuVS options schema required is an array");
    required.retain(|name| {
        name.as_str().is_none_or(|name| !operation.descriptor().relation_roles.iter().any(|role| role.as_str() == name))
    });
    schema
}

fn argument_descriptions(operation: CuvsOperation) -> Value {
    Value::Array(
        operation
            .descriptor()
            .options
            .iter()
            .filter(|option| !matches!(option.kind, CuvsOptionKind::RelationBinding))
            .map(|option| {
                json!({
                    "name": option.name,
                    "required": option.required,
                    "description": option.description,
                    "schema": value_options_schema(operation)["properties"][option.name].clone(),
                })
            })
            .collect(),
    )
}

fn canonical_value_options(operation: CuvsOperation, canonical: &Value) -> Value {
    let canonical = canonical.as_object().expect("canonical cuVS options are an object");
    Value::Object(
        operation
            .descriptor()
            .options
            .iter()
            .filter(|option| !matches!(option.kind, CuvsOptionKind::RelationBinding))
            .filter_map(|option| canonical.get(option.name).cloned().map(|value| (option.name.to_string(), value)))
            .collect(),
    )
}

fn result_schemas(operation: CuvsOperation) -> Value {
    let descriptor = operation.descriptor();
    let fields = descriptor
        .output_fields
        .iter()
        .map(|field| {
            let mut field_json = json!({
                "name": field.name,
                "data_type": field.data_type,
                "nullable": field.nullable,
                "description": field.description,
            });
            if let Some(repeats) = field.repeats {
                field_json["repeats"] = Value::String(repeats.to_string());
            }
            field_json
        })
        .collect::<Vec<_>>();
    json!({
        "output_contract": output_contract_name(descriptor.output_contract),
        "fields": fields,
    })
}

fn validation_request_schema(roles: &[CuvsRelationRole], options_schema: &Value) -> Value {
    let relation_properties = roles
        .iter()
        .map(|role| {
            (
                role.as_str().to_string(),
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
                "required": roles.iter().map(|role| role.as_str()).collect::<Vec<_>>(),
                "properties": relation_properties,
            },
            "options": options_schema,
        }
    })
}

fn examples(operation: CuvsOperation) -> Value {
    let descriptor = operation.descriptor();
    let options = match operation {
        CuvsOperation::BruteForceKnn => json!({
            "k": 8,
            "metric":"l2_expanded"
        }),
        CuvsOperation::PairwiseDistance => json!({ "metric": "l2_expanded" }),
        CuvsOperation::TopK => json!({ "k": 1, "select": "max" }),
        CuvsOperation::KMeans => json!({
            "n_clusters": 8
        }),
        CuvsOperation::Pca => json!({
            "n_components": 2
        }),
    };
    // The quickstart SQL and the validation envelope read the same tables, so
    // a validated call is the call the quickstart executes.
    let relations = operation
        .descriptor()
        .relation_roles
        .iter()
        .map(|role| (role.as_str().to_string(), json!({"table": example_table(*role)})))
        .collect::<Map<_, _>>();
    debug_assert!(
        relations.values().all(|relation| descriptor.quickstart_sql.contains(relation["table"].as_str().unwrap()))
    );
    json!([{
        "quickstart": {
            "display_name": descriptor.display_name,
            "use_this_when": descriptor.use_this_when,
            "sql": descriptor.quickstart_sql,
        },
        "validation": {
            "function_name": descriptor.sql_function_name,
            "call": {"relations": relations, "options": options}
        }
    }])
}

fn example_table(role: CuvsRelationRole) -> &'static str {
    match role {
        CuvsRelationRole::Dataset => "dataset_vectors",
        CuvsRelationRole::Queries => "query_vectors",
        CuvsRelationRole::Input => "input_vectors",
        CuvsRelationRole::X => "x_vectors",
        CuvsRelationRole::Y => "y_vectors",
    }
}

fn limitations(operation: CuvsOperation) -> Value {
    let common = [
        "Validation resolves named tables or views and reads schemas only; it does not execute relation scans or GPU work.",
        "Execution relation arguments require parenthesized subqueries; dry-run validation accepts registered named relations only.",
    ];
    Value::Array(
        common
            .into_iter()
            .chain(operation.descriptor().limitations.iter().copied())
            .map(|limitation| Value::String(limitation.to_string()))
            .collect(),
    )
}

struct ResolvedSchemaOnlyInput {
    plan: LogicalPlan,
    schema: SchemaRef,
}

async fn resolve_schema_only_input(
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

fn unavailable_validation(
    operation: CuvsOperation,
    request: &GpuCallRequest,
    availability: &GpuFunctionAvailability,
) -> GpuCallValidation {
    let reason = availability.unavailable_reason().expect("unavailable cuVS functions must have a reason");
    GpuCallValidation {
        valid: false,
        error_code: Some(reason.to_string()),
        message: match reason {
            "required_feature_disabled" => {
                "cuVS vector algorithm execution requires the cuvs Cargo feature".to_string()
            },
            _ => "cuVS function is unavailable in this session/build".to_string(),
        },
        function_name: operation.descriptor().sql_function_name.to_string(),
        relations_resolved: false,
        normalized_call: request.normalized_json(),
        output_schema: json!({}),
        would_execute_gpu: false,
        details: json!({
            "provider": CUVS_PROVIDER,
            "operation": operation_name(operation),
            "availability": {"available": false, "reason": reason}
        }),
    }
}

/// Projects a typed validation rejection into the stable error code plus the
/// `details_json` locator: `path` names `options.<argument>` or
/// `relations.<role>[.fields.<field>]`, and the typed facts that located the
/// failure (`argument`, `relation_role`, `field`, `actual_type`,
/// `expected_type`) are copied alongside it.
fn validation_error(error: &DataFusionError, details: &mut Value) -> (String, String) {
    let Some(error) = error.find_algeon_error() else {
        details["path"] = Value::String("options".to_string());
        return ("cuvs_validation_failed".to_string(), error.to_string());
    };
    let code = match error.code() {
        ErrorCode::FieldNotFound => "column_not_found",
        ErrorCode::UnsupportedType => "unsupported_type",
        _ => error.facts().get("reason").unwrap_or("cuvs_validation_failed"),
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

fn base_details(operation: CuvsOperation, relations: Map<String, Value>) -> Value {
    json!({
        "provider": CUVS_PROVIDER,
        "operation": operation_name(operation),
        "relations": relations,
    })
}

fn schema_json(schema: &SchemaRef) -> Value {
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

fn operation_name(operation: CuvsOperation) -> &'static str {
    match operation {
        CuvsOperation::BruteForceKnn => "brute_force_knn",
        CuvsOperation::PairwiseDistance => "pairwise_distance",
        CuvsOperation::TopK => "top_k",
        CuvsOperation::KMeans => "kmeans",
        CuvsOperation::Pca => "pca",
    }
}

fn output_contract_name(contract: CuvsOutputContract) -> &'static str {
    match contract {
        CuvsOutputContract::KnnNeighbors => "knn_neighbors",
        CuvsOutputContract::PairwiseDistances => "pairwise_distances",
        CuvsOutputContract::TopKValues => "top_k_values",
        CuvsOutputContract::KMeansAssignments => "kmeans_assignments",
        CuvsOutputContract::PcaComponents => "pca_components",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_project_every_canonical_cuvs_operation() {
        let family = CuvsGpuFunctionFamily::new(GpuFunctionAvailability::Available);
        let descriptors = family.descriptors();
        assert_eq!(descriptors.len(), CUVS_OPERATION_REGISTRY.len());
        assert_eq!(
            descriptors.iter().map(|descriptor| descriptor.function_name.as_str()).collect::<Vec<_>>(),
            CUVS_OPERATION_REGISTRY.iter().map(|descriptor| descriptor.sql_function_name).collect::<Vec<_>>()
        );
        assert!(descriptors.iter().all(|descriptor| {
            descriptor.provider == CUVS_PROVIDER
                && descriptor.options_schema.is_object()
                && descriptor.validation_request_schema.is_object()
                && descriptor.examples.is_array()
                && descriptor.limitations.is_array()
        }));
    }

    #[test]
    fn availability_follows_the_cuvs_build_feature() {
        let availability = cuvs_gpu_function_availability(None);
        if cfg!(feature = "cuvs") {
            assert!(availability.is_available());
        } else {
            assert_eq!(availability.unavailable_reason(), Some("required_feature_disabled"));
        }
    }
}
