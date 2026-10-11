//! cuVS adapter for the common GPU-function discovery and validation catalog.
//!
//! This module deliberately reuses [`CuvsLogicalNode`] for static validation.
//! Named validation relations become schema-only `TableScan` logical inputs, so
//! no relation scan, GPU allocation, or cuVS operation is performed here.

use datafusion::{catalog::Session, common::DataFusionError};
use serde_json::{Map, Value, json};

use super::{
    CUVS_OPERATION_REGISTRY, CuvsLogicalNode, CuvsOperation, CuvsOutputContract, CuvsRelationRole, GpuOptionKind,
    error::ERRORS,
    options::{dense_relation_function, validate_options},
};
use crate::gpu_functions::{
    GpuCallProblem, GpuCallProblems, GpuCallRequest, GpuCallValidation, GpuFunctionAvailability, GpuFunctionDescriptor,
    GpuFunctionFamily, REQUIRED_FEATURE_DISABLED,
    dense_relation::{
        NULL_VALUES_CONTRACT, resolve_schema_only_input, schema_json, validation_problem, validation_request_schema,
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

        let function = dense_relation_function(operation);
        let mut problems = GpuCallProblems::default();
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
                    let mut details = base_details(operation, relation_details.clone());
                    details["path"] = Value::String(format!("relations.{role_name}"));
                    details["relation_role"] = Value::String(role_name.to_string());
                    problems.push(GpuCallProblem { code: "table_not_found".to_string(), message, details });
                },
            }
        }
        let relations_resolved = problems.is_empty();
        let base = base_details(operation, relation_details);
        let bindings = if relations_resolved {
            function.relation_bindings(&inputs).map_err(|errors| {
                for error in errors {
                    problems.push(validation_problem(&error.into(), &base, "cuvs_validation_failed"));
                }
            })
        } else {
            Err(())
        };
        let mut values = request.options.as_object().cloned().unwrap_or_default();
        let failing_option =
            |error: &DataFusionError, values: &Map<String, Value>| function.failing_option(error, values);
        let project = |error: &DataFusionError| validation_problem(error, &base, "cuvs_validation_failed");
        let node = match &bindings {
            Ok(bindings) => problems.check_options(
                &mut values,
                |values| {
                    let options_json = function.options_json_with_bindings(bindings, values.clone())?;
                    CuvsLogicalNode::try_new(operation, inputs.clone(), &options_json)
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
        let function_name = operation.descriptor().sql_function_name;
        if let Some(invalid) = problems.into_invalid(function_name, relations_resolved, request.normalized_json()) {
            return invalid;
        }
        let node = node.expect("a call without problems has a validated node");
        let mut normalized = request.clone();
        normalized.options = Value::Object(node.validated_options().value_options().clone());
        let mut details = base;
        details["output_contract"] =
            Value::String(output_contract_name(operation.descriptor().output_contract).to_string());
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
                        "null_values": NULL_VALUES_CONTRACT,
                        "data_types": ["Int32", "Int64", "Utf8", "LargeUtf8", "Utf8View"]
                    },
                    "features": {
                        "null_values": NULL_VALUES_CONTRACT,
                        "element_types": descriptor.vector_element_support.iter().map(|support| support.element.as_str()).collect::<Vec<_>>(),
                        "element_metric_support": descriptor.vector_element_support.iter().map(|support| json!({
                            "element": support.element.as_str(), "metrics": support.metrics,
                        })).collect::<Vec<_>>(),
                        "one_of": [
                            {
                                "shape": "wide",
                                "rule": "Every non-id field is a feature of the same supported element type; projection order defines vector dimension order. A nullable field type is accepted and a null value fails the query."
                            },
                            {
                                "shape": "list",
                                "rule": "The only non-id field is a FixedSizeList, List, or LargeList of a supported element type, under any name. Nullable field and element types are accepted and a null row or element fails the query."
                            }
                        ]
                    }
                },
                "description": role.description(),
            })
        })
        .collect::<Vec<_>>();
    let options_schema = value_options_schema(operation);
    let validation_request_schema =
        validation_request_schema(descriptor.relation_roles.iter().map(|role| role.as_str()), &options_schema);
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
                .filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
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
            .filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
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

fn examples(operation: CuvsOperation) -> Value {
    let descriptor = operation.descriptor();
    let options = match operation {
        CuvsOperation::BruteForceKnn => json!({
            "k": 8,
            "metric":"l2_expanded"
        }),
        CuvsOperation::IvfFlat => json!({ "k": 8, "metric": "l2_expanded", "n_lists": 16, "n_probes": 16 }),
        CuvsOperation::IvfPq => json!({ "k": 8, "metric": "l2_expanded", "n_lists": 16, "n_probes": 16, "pq_dim": 2 }),
        CuvsOperation::Cagra => {
            json!({ "k": 8, "metric": "l2_expanded", "graph_degree": 32, "intermediate_graph_degree": 64 })
        },
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
            REQUIRED_FEATURE_DISABLED => ERRORS.feature_disabled_message.to_string(),
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

fn base_details(operation: CuvsOperation, relations: Map<String, Value>) -> Value {
    json!({
        "provider": CUVS_PROVIDER,
        "operation": operation_name(operation),
        "relations": relations,
    })
}

fn operation_name(operation: CuvsOperation) -> &'static str {
    match operation {
        CuvsOperation::BruteForceKnn => "brute_force_knn",
        CuvsOperation::IvfFlat => "ivf_flat",
        CuvsOperation::IvfPq => "ivf_pq",
        CuvsOperation::Cagra => "cagra",
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
}
