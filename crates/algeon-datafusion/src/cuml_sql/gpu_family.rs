use datafusion::{catalog::Session, execution::context::SessionConfig};
use serde_json::{Map, Value, json};

use super::{
    CUML_OPERATION_REGISTRY, CumlLogicalNode, CumlOperation, GpuOptionKind, MlRelationRole,
    options::options_from_named_values,
};
use crate::gpu_functions::{
    GpuCallRequest, GpuCallValidation, GpuFunctionAvailability, GpuFunctionDescriptor, GpuFunctionFamily,
    dense_relation::{resolve_schema_only_input, schema_json, validation_error, validation_request_schema},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CumlGpuFunctionFamily {
    availability: GpuFunctionAvailability,
}
impl CumlGpuFunctionFamily {
    pub(crate) fn new(availability: GpuFunctionAvailability) -> Self {
        Self { availability }
    }
}

pub(crate) fn cuml_gpu_function_availability(_config: Option<&SessionConfig>) -> GpuFunctionAvailability {
    if !cfg!(feature = "cuml") {
        return GpuFunctionAvailability::Unavailable { reason: "required_feature_disabled".to_string() };
    }
    GpuFunctionAvailability::Available
}

#[async_trait::async_trait]
impl GpuFunctionFamily for CumlGpuFunctionFamily {
    fn family_id(&self) -> &'static str {
        "cuml"
    }
    fn descriptors(&self) -> Vec<GpuFunctionDescriptor> {
        CUML_OPERATION_REGISTRY
            .iter()
            .map(|entry| descriptor_for_operation(entry.operation, self.availability.clone()))
            .collect()
    }
    async fn validate(
        &self,
        session: &dyn Session,
        function_name: &str,
        request: &GpuCallRequest,
    ) -> GpuCallValidation {
        let Some(operation) = CumlOperation::from_sql_function_name(function_name) else {
            return GpuCallValidation::invalid(
                "unknown_function",
                format!("unknown cuML operation function '{function_name}'"),
                function_name,
                request.normalized_json(),
            );
        };
        if !self.availability.is_available() {
            let reason = self.availability.unavailable_reason().expect("unavailable cuML functions have a reason");
            return GpuCallValidation {
                valid: false,
                error_code: Some(reason.to_string()),
                message: "cuML execution requires the cuml Cargo feature".to_string(),
                function_name: function_name.to_string(),
                relations_resolved: false,
                normalized_call: request.normalized_json(),
                output_schema: json!({}),
                would_execute_gpu: false,
                details: json!({"provider": "cuml", "operation": operation.as_name(), "availability": {"available": false, "reason": reason}}),
            };
        }
        let mut inputs = Vec::new();
        let mut relation_details = Map::new();
        for role in operation.descriptor().relation_roles {
            let role_name = role.as_str();
            let relation = request.relations.get(role_name).expect("common catalog validates relation roles");
            match resolve_schema_only_input(session, &relation.table, relation.reference.clone()).await {
                Ok(resolved) => {
                    relation_details.insert(
                        role_name.to_string(),
                        json!({"table": relation.table, "resolved": true, "schema": schema_json(&resolved.schema)}),
                    );
                    inputs.push(resolved.plan);
                },
                Err(message) => {
                    relation_details.insert(role_name.to_string(), json!({"table": relation.table, "resolved": false}));
                    return GpuCallValidation {
                        valid: false,
                        error_code: Some("table_not_found".to_string()),
                        message,
                        function_name: function_name.to_string(),
                        relations_resolved: false,
                        normalized_call: request.normalized_json(),
                        output_schema: json!({}),
                        would_execute_gpu: false,
                        details: json!({"provider": "cuml", "operation": operation.as_name(), "relations": relation_details, "path": format!("relations.{role_name}"), "relation_role": role_name}),
                    };
                },
            }
        }
        let values = request.options.as_object().cloned().unwrap_or_default();
        let result = options_from_named_values(operation, &inputs, values)
            .and_then(|json| CumlLogicalNode::try_new(operation, inputs, &json));
        match result {
            Ok(node) => {
                let canonical: Value =
                    serde_json::from_str(node.validated_options().canonical_json()).expect("canonical cuML JSON");
                let mut normalized = request.clone();
                normalized.options = Value::Object(
                    operation
                        .descriptor()
                        .options
                        .iter()
                        .filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
                        .filter_map(|option| {
                            canonical.get(option.name).cloned().map(|value| (option.name.to_string(), value))
                        })
                        .collect(),
                );
                GpuCallValidation {
                    valid: true,
                    error_code: None,
                    message: "valid".to_string(),
                    function_name: function_name.to_string(),
                    relations_resolved: true,
                    normalized_call: normalized.normalized_json(),
                    output_schema: schema_json(&node.output_schema()),
                    would_execute_gpu: true,
                    details: json!({"provider": "cuml", "operation": operation.as_name(), "relations": relation_details, "output_contract": output_contract(operation), "validated_options": normalized.options}),
                }
            },
            Err(error) => {
                let mut details =
                    json!({"provider": "cuml", "operation": operation.as_name(), "relations": relation_details});
                let (code, message) = validation_error(&error, &mut details, "cuml_validation_failed");
                GpuCallValidation {
                    valid: false,
                    error_code: Some(code),
                    message,
                    function_name: function_name.to_string(),
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

fn descriptor_for_operation(operation: CumlOperation, availability: GpuFunctionAvailability) -> GpuFunctionDescriptor {
    let descriptor = operation.descriptor();
    let relation_roles = descriptor.relation_roles.iter().map(|role| {
        let supervised = matches!(descriptor.operation, CumlOperation::LogisticRegression | CumlOperation::RandomForestClassifier | CumlOperation::RandomForestRegressor);
        let mut fields = json!({
            "id": {"required": true, "nullable": false, "data_types": ["Int32", "Int64", "Utf8", "LargeUtf8", "Utf8View"]},
            "features": {"element_types": descriptor.vector_element_support.iter().map(|element| element.as_str()).collect::<Vec<_>>(),
                "one_of": [
                    {"shape": "wide", "rule": if supervised { "Every non-id, non-label field is a non-null Float32 feature; projection order defines vector dimension order." } else { "Every non-id field is a non-null Float32 feature; projection order defines vector dimension order." }},
                    {"shape": "list", "field": "vector", "rule": if supervised { "The non-null vector field is FixedSizeList, List, or LargeList of non-null Float32 elements; training also contains label." } else { "The non-null vector field is FixedSizeList, List, or LargeList of non-null Float32 elements, with no other non-id fields." }}
                ]},
        });
        if supervised {
            fields["label"] = if *role == MlRelationRole::Training {
                json!({"required": true, "nullable": false, "data_types": [if operation == CumlOperation::RandomForestRegressor { "Float32" } else { "Int32" }]})
            } else {
                json!({"allowed": false})
            };
        }
        json!({"name": role.as_str(), "required": true, "reference": {"type": "table"},
            "fields": fields, "description": role.description()})
    }).collect();
    let mut options_schema = descriptor.options_json_schema();
    let object = options_schema.as_object_mut().expect("cuML schema object");
    let properties = object.get_mut("properties").and_then(Value::as_object_mut).expect("cuML properties");
    for role in descriptor.relation_roles {
        properties.remove(role.as_str());
    }
    let required = object.get_mut("required").and_then(Value::as_array_mut).expect("cuML required options");
    required.retain(|name| {
        name.as_str().is_none_or(|name| !descriptor.relation_roles.iter().any(|role| role.as_str() == name))
    });
    let argument_descriptions = descriptor.options.iter().filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
        .map(|option| json!({"name": option.name, "required": option.required, "description": option.description, "schema": options_schema["properties"][option.name]})).collect();
    let fields = descriptor.output_fields.iter().map(|field| {
        let mut value = json!({"name": field.name, "data_type": field.data_type, "nullable": field.nullable, "description": field.description});
        if let Some(repeats) = field.repeats { value["repeats"] = json!(repeats); }
        value
    }).collect::<Vec<_>>();
    let validation_request_schema =
        validation_request_schema(descriptor.relation_roles.iter().map(|role| role.as_str()), &options_schema);
    let options = match operation {
        CumlOperation::Pca | CumlOperation::Tsvd => json!({"n_components": 2}),
        CumlOperation::Hdbscan
        | CumlOperation::Umap
        | CumlOperation::LogisticRegression
        | CumlOperation::RandomForestClassifier
        | CumlOperation::RandomForestRegressor => json!({}),
    };
    let relations = if matches!(
        operation,
        CumlOperation::LogisticRegression
            | CumlOperation::RandomForestClassifier
            | CumlOperation::RandomForestRegressor
    ) {
        json!({"training": {"table": "training_vectors"}, "predict": {"table": "predict_vectors"}})
    } else {
        json!({"input": {"table": "input_vectors"}})
    };
    GpuFunctionDescriptor {
        function_name: descriptor.sql_function_name.to_string(), provider: "cuml".to_string(),
        signature: format!("{}({})", descriptor.sql_function_name,
            descriptor.options.iter().map(|option| if matches!(option.kind, GpuOptionKind::RelationBinding) {
                format!("{} => (relation subquery)", option.name)
            } else { format!("{} => value", option.name) }).collect::<Vec<_>>().join(", ")),
        summary: descriptor.summary.to_string(), relation_roles: Value::Array(relation_roles),
        argument_descriptions: Value::Array(argument_descriptions), options_schema,
        result_schemas: json!({"output_contract": output_contract(operation), "fields": fields}), validation_request_schema,
        examples: json!([{"quickstart": {"display_name": descriptor.display_name, "use_this_when": descriptor.use_this_when, "sql": descriptor.quickstart_sql},
            "validation": {"function_name": descriptor.sql_function_name, "call": {"relations": relations, "options": options}}}]),
        limitations: Value::Array([
            "Validation resolves named tables or views and reads schemas only; it does not execute relation scans or GPU work.",
            "Execution relation arguments require parenthesized subqueries; dry-run validation accepts registered named relations only.",
        ].into_iter().chain(descriptor.limitations.iter().copied()).map(|text| json!(text)).collect()),
        availability,
    }
}

fn output_contract(operation: CumlOperation) -> &'static str {
    match operation {
        CumlOperation::Pca | CumlOperation::Tsvd => "components",
        CumlOperation::Hdbscan => "hdbscan_clusters",
        CumlOperation::Umap => "embedding",
        CumlOperation::LogisticRegression
        | CumlOperation::RandomForestClassifier
        | CumlOperation::RandomForestRegressor => "predictions",
    }
}

trait CumlOperationName {
    fn as_name(&self) -> &'static str;
}
impl CumlOperationName for CumlOperation {
    fn as_name(&self) -> &'static str {
        match self {
            Self::Pca => "pca",
            Self::Tsvd => "tsvd",
            Self::Hdbscan => "hdbscan",
            Self::Umap => "umap",
            Self::LogisticRegression => "logistic_regression",
            Self::RandomForestClassifier => "random_forest_classifier",
            Self::RandomForestRegressor => "random_forest_regressor",
        }
    }
}
