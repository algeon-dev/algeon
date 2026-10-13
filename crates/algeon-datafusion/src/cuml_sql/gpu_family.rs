use std::{any::Any, sync::Arc};

use datafusion::{catalog::Session, common::DataFusionError};
use serde_json::{Map, Value, json};

use super::{
    CUML_OPERATION_REGISTRY, CumlLogicalNode, CumlOperation, GpuOptionKind, MlRelationRole,
    model_store::{
        CumlModelStore, MODEL_ROOT_UNCONFIGURED, MODEL_ROOT_UNCONFIGURED_MESSAGE, MODELS_ROOT_OPTION,
        bind_operation_model, require_model_store,
    },
    options::{dense_relation_function, validate_options},
};
use crate::gpu_functions::{
    GpuCallProblem, GpuCallProblems, GpuCallRequest, GpuCallValidation, GpuFunctionAvailability, GpuFunctionDescriptor,
    GpuFunctionFamily,
    dense_relation::{
        NULL_VALUES_CONTRACT, resolve_schema_only_input, schema_json, validation_problem, validation_request_schema,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CumlGpuFunctionFamily {
    availability: GpuFunctionAvailability,
    model_store: Option<Arc<CumlModelStore>>,
}
impl CumlGpuFunctionFamily {
    pub(crate) fn new(availability: GpuFunctionAvailability, model_store: Option<Arc<CumlModelStore>>) -> Self {
        Self { availability, model_store }
    }

    fn operation_availability(&self, operation: CumlOperation) -> GpuFunctionAvailability {
        if self.availability.is_available() && require_model_store(operation, self.model_store.as_deref()).is_err() {
            return GpuFunctionAvailability::Unavailable { reason: MODEL_ROOT_UNCONFIGURED.to_string() };
        }
        self.availability.clone()
    }
}

#[async_trait::async_trait]
impl GpuFunctionFamily for CumlGpuFunctionFamily {
    fn family_id(&self) -> &'static str {
        "cuml"
    }
    fn descriptors(&self) -> Vec<GpuFunctionDescriptor> {
        CUML_OPERATION_REGISTRY
            .iter()
            .map(|entry| descriptor_for_operation(entry.operation, self.operation_availability(entry.operation)))
            .collect()
    }

    /// The models root and size limit are not part of the descriptors, so a
    /// reinstall with a different model store conflicts.
    fn same_installation(&self, installed: &dyn GpuFunctionFamily) -> bool {
        (installed as &dyn Any).downcast_ref::<Self>() == Some(self)
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
        let availability = self.operation_availability(operation);
        if !availability.is_available() {
            let reason = availability.unavailable_reason().expect("unavailable cuML functions have a reason");
            let model_root_unconfigured = reason == MODEL_ROOT_UNCONFIGURED;
            let mut details = json!({"provider": "cuml", "operation": operation.as_name(), "availability": {"available": false, "reason": reason}});
            if model_root_unconfigured {
                details["required_option"] = json!(MODELS_ROOT_OPTION);
            }
            return GpuCallValidation {
                valid: false,
                error_code: Some(reason.to_string()),
                message: if model_root_unconfigured {
                    MODEL_ROOT_UNCONFIGURED_MESSAGE
                } else {
                    "cuML execution requires the cuml Cargo feature"
                }
                .to_string(),
                function_name: function_name.to_string(),
                relations_resolved: false,
                normalized_call: request.normalized_json(),
                output_schema: json!({}),
                would_execute_gpu: false,
                details,
            };
        }
        let function = dense_relation_function(operation);
        let mut problems = GpuCallProblems::default();
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
                    problems.push(GpuCallProblem {
                        code: "table_not_found".to_string(),
                        message,
                        details: json!({"provider": "cuml", "operation": operation.as_name(), "relations": relation_details.clone(), "path": format!("relations.{role_name}"), "relation_role": role_name}),
                    });
                },
            }
        }
        let relations_resolved = problems.is_empty();
        let base = json!({"provider": "cuml", "operation": operation.as_name(), "relations": relation_details});
        let bindings = if relations_resolved {
            function.relation_bindings(&inputs).map_err(|errors| {
                for error in errors {
                    problems.push(validation_problem(&error.into(), &base, "cuml_validation_failed"));
                }
            })
        } else {
            Err(())
        };
        let mut values = request.options.as_object().cloned().unwrap_or_default();
        let failing_option =
            |error: &DataFusionError, values: &Map<String, Value>| function.failing_option(error, values);
        let project = |error: &DataFusionError| validation_problem(error, &base, "cuml_validation_failed");
        let node = match &bindings {
            Ok(bindings) => problems.check_options(
                &mut values,
                |values| {
                    let options_json = function.options_json_with_bindings(bindings, values.clone())?;
                    let model = bind_operation_model(
                        operation,
                        self.model_store.as_deref(),
                        &validate_options(operation, &options_json)?,
                    )?;
                    CumlLogicalNode::try_new_with_model(operation, inputs.clone(), &options_json, model)
                },
                failing_option,
                project,
            ),
            // Without every relation's binding, the value options are still
            // checked, against bindings of the declared shape.
            Err(()) => {
                let bindings = function.stand_in_bindings();
                problems.check_options(
                    &mut values,
                    |values| {
                        let options_json = function.options_json_with_bindings(&bindings, values.clone())?;
                        Ok(validate_options(operation, &options_json)?)
                    },
                    failing_option,
                    project,
                );
                None
            },
        };
        if let Some(invalid) = problems.into_invalid(function_name, relations_resolved, request.normalized_json()) {
            return invalid;
        }
        let node = node.expect("a call without problems has a validated node");
        let mut normalized = request.clone();
        normalized.options = Value::Object(node.validated_options().value_options().clone());
        let mut details = base;
        details["output_contract"] = json!(output_contract(operation));
        details["validated_options"] = normalized.options.clone();
        GpuCallValidation {
            valid: true,
            error_code: None,
            message: "valid".to_string(),
            function_name: function_name.to_string(),
            relations_resolved: true,
            normalized_call: normalized.normalized_json(),
            output_schema: schema_json(&node.output_schema()),
            would_execute_gpu: true,
            details,
        }
    }
}

fn descriptor_for_operation(operation: CumlOperation, availability: GpuFunctionAvailability) -> GpuFunctionDescriptor {
    let descriptor = operation.descriptor();
    let relation_roles = descriptor.relation_roles.iter().map(|role| {
        let supervised = matches!(descriptor.operation, CumlOperation::LogisticRegression | CumlOperation::RandomForestClassifier | CumlOperation::RandomForestRegressor);
        let elements = descriptor.vector_element_support.iter().map(|element| element.as_str()).collect::<Vec<_>>();
        let element_types = elements.join(" or ");
        let features = if supervised { "Every non-id, non-label field" } else { "Every non-id field" };
        let list_field = if supervised { "The only non-id, non-label field" } else { "The only non-id field" };
        let mut fields = json!({
            "id": {"required": true, "null_values": NULL_VALUES_CONTRACT, "data_types": ["Int32", "Int64", "Utf8", "LargeUtf8", "Utf8View"]},
            "features": {"null_values": NULL_VALUES_CONTRACT, "element_types": elements,
                "one_of": [
                    {"shape": "wide", "rule": format!("{features} is a {element_types} feature; projection order defines vector dimension order. A nullable field type is accepted and a null value fails the query.")},
                    {"shape": "list", "rule": format!("{list_field} is a FixedSizeList, List, or LargeList of {element_types} elements, under any name. Nullable field and element types are accepted and a null row or element fails the query.")}
                ]},
        });
        if supervised {
            fields["label"] = if *role == MlRelationRole::Training {
                json!({"required": true, "null_values": NULL_VALUES_CONTRACT, "data_types": [if operation == CumlOperation::RandomForestRegressor { "Float32" } else { "Int32" }]})
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
        CumlOperation::Tsvd => json!({"n_components": 2}),
        CumlOperation::ForestPredict => json!({"model": "models/scorer.json", "model_format": "xgboost_json"}),
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
        function_name: descriptor.sql_function_name.to_string(),
        provider: "cuml".to_string(),
        signature: format!(
            "{}({})",
            descriptor.sql_function_name,
            descriptor
                .options
                .iter()
                .map(|option| if matches!(option.kind, GpuOptionKind::RelationBinding) {
                    format!("{} => (relation subquery)", option.name)
                } else {
                    format!("{} => value", option.name)
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        summary: descriptor.summary.to_string(),
        relation_roles: Value::Array(relation_roles),
        argument_descriptions: Value::Array(argument_descriptions),
        options_schema,
        result_schemas: json!({"output_contract": output_contract(operation), "fields": fields}),
        validation_request_schema,
        examples: json!([{"quickstart": {"display_name": descriptor.display_name, "use_this_when": descriptor.use_this_when, "sql": descriptor.quickstart_sql},
            "validation": {"function_name": descriptor.sql_function_name, "call": {"relations": relations, "options": options}}}]),
        limitations: Value::Array(descriptor.limitations.iter().map(|text| json!(text)).collect()),
        availability,
    }
}

fn output_contract(operation: CumlOperation) -> &'static str {
    match operation {
        CumlOperation::Tsvd => "components",
        CumlOperation::Hdbscan => "hdbscan_clusters",
        CumlOperation::Umap => "embedding",
        CumlOperation::LogisticRegression
        | CumlOperation::RandomForestClassifier
        | CumlOperation::RandomForestRegressor => "predictions",
        CumlOperation::ForestPredict => "forest_prediction",
    }
}

trait CumlOperationName {
    fn as_name(&self) -> &'static str;
}
impl CumlOperationName for CumlOperation {
    fn as_name(&self) -> &'static str {
        match self {
            Self::Tsvd => "tsvd",
            Self::Hdbscan => "hdbscan",
            Self::Umap => "umap",
            Self::LogisticRegression => "logistic_regression",
            Self::RandomForestClassifier => "random_forest_classifier",
            Self::RandomForestRegressor => "random_forest_regressor",
            Self::ForestPredict => "forest_predict",
        }
    }
}
