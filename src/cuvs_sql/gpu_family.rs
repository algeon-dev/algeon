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

use super::{CUVS_OPERATION_REGISTRY, CuvsLogicalNode, CuvsOperation, CuvsOutputContract, CuvsRelationRole};
use crate::{
    error::{DataFusionErrorExt, ErrorCode},
    gpu_functions::{
        GpuCallRequest, GpuCallValidation, GpuFunctionAvailability, GpuFunctionDescriptor, GpuFunctionFamily,
    },
    session_config::{DataFusionNexusSessionConfig, NativeMemoryPolicyConfig},
};

const CUVS_FAMILY_ID: &str = "cuvs";
const CUVS_PROVIDER: &str = "cuvs";
const CUVS_DOMAIN: &str = "vector";

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
///
/// The SQL relation planner is useful without the cuVS feature so it remains
/// installable for planning contracts. The catalog must nevertheless advertise
/// that execution cannot be planned. cuVS also intentionally has no bounded
/// execution contract until lower-level peak-memory preflight exists.
pub(crate) fn cuvs_gpu_function_availability(config: Option<&SessionConfig>) -> GpuFunctionAvailability {
    if !cfg!(feature = "cuvs") {
        return GpuFunctionAvailability::Unavailable { reason: "required_feature_disabled".to_string() };
    }
    if config
        .and_then(|config| config.options().extensions.get::<DataFusionNexusSessionConfig>())
        .is_some_and(|config| config.native_memory_policy == NativeMemoryPolicyConfig::Bounded)
    {
        return GpuFunctionAvailability::Unavailable { reason: "bounded_execution_unavailable".to_string() };
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
            match resolve_schema_only_input(session, &relation.table).await {
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
                    let mut details = base_details(operation, relation_details);
                    details["failed_relation_role"] = Value::String(role_name.to_string());
                    details["failed_relation_table"] = Value::String(relation.table.clone());
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

        let options_json = request.options.to_string();
        match CuvsLogicalNode::try_new(operation, inputs, &options_json) {
            Ok(node) => {
                let canonical_options: Value = serde_json::from_str(node.validated_options().canonical_json())
                    .expect("canonical cuVS options are valid JSON");
                let mut normalized = request.clone();
                normalized.options = canonical_options;
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
                let (error_code, message) = validation_error(&error);
                details["validation_error_code"] = Value::String(error_code.clone());
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
                "description": relation_role_description(*role),
            })
        })
        .collect::<Vec<_>>();
    let options_schema = descriptor.options_json_schema();
    GpuFunctionDescriptor {
        function_name: descriptor.sql_function_name.to_string(),
        provider: CUVS_PROVIDER.to_string(),
        domain: CUVS_DOMAIN.to_string(),
        signature: signature(operation),
        summary: summary(operation).to_string(),
        relation_roles: Value::Array(relation_roles),
        argument_descriptions: json!([
            {
                "name": "options_json",
                "position": descriptor.relation_roles.len() + 1,
                "type": "JSON string literal",
                "description": "cuVS operation options and relation-column bindings"
            }
        ]),
        options_schema: options_schema.clone(),
        result_schemas: result_schemas(operation),
        validation_request_schema: validation_request_schema(descriptor.relation_roles, options_schema),
        examples: examples(operation),
        limitations: limitations(),
        availability,
    }
}

fn signature(operation: CuvsOperation) -> String {
    let roles = operation
        .descriptor()
        .relation_roles
        .iter()
        .map(|role| format!("({} relation subquery)", role.as_str()))
        .chain(std::iter::once("options_json".to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({roles})", operation.descriptor().sql_function_name)
}

fn summary(operation: CuvsOperation) -> &'static str {
    match operation {
        CuvsOperation::BruteForceKnn => "Exact brute-force nearest-neighbor search over dense vectors.",
        CuvsOperation::KMeans => "K-means clustering assignments for dense vector rows.",
        CuvsOperation::Pca => "Principal-component analysis transform for dense vector rows.",
    }
}

fn relation_role_description(role: CuvsRelationRole) -> &'static str {
    match role {
        CuvsRelationRole::Dataset => "Indexed dense-vector dataset.",
        CuvsRelationRole::Queries => "Dense-vector query relation.",
        CuvsRelationRole::Input => "Dense-vector input relation.",
    }
}

fn result_schemas(operation: CuvsOperation) -> Value {
    let fields = match operation.descriptor().output_contract {
        CuvsOutputContract::BruteForceKnn => json!([
            {"name":"query_ordinal","data_type":"UInt64","nullable":false},
            {"name":"query_id","data_type":"same_as_queries.id","nullable":false},
            {"name":"neighbor_ordinal","data_type":"UInt64","nullable":false},
            {"name":"neighbor_id","data_type":"same_as_dataset.id","nullable":false},
            {"name":"rank","data_type":"UInt32","nullable":false},
            {"name":"distance","data_type":"Float32","nullable":false}
        ]),
        CuvsOutputContract::KMeansAssignments => json!([
            {"name":"row_ordinal","data_type":"UInt64","nullable":false},
            {"name":"id","data_type":"same_as_input.id","nullable":false},
            {"name":"cluster_id","data_type":"Int32","nullable":false}
        ]),
        CuvsOutputContract::PcaComponents => json!([
            {"name":"row_ordinal","data_type":"UInt64","nullable":false},
            {"name":"id","data_type":"same_as_input.id","nullable":false},
            {"name":"pc_<index>","data_type":"Float32","nullable":false,"repeats":"n_components"}
        ]),
    };
    json!({
        "output_contract": output_contract_name(operation.descriptor().output_contract),
        "fields": fields,
    })
}

fn validation_request_schema(roles: &[CuvsRelationRole], options_schema: Value) -> Value {
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
        "required": ["schema_version", "relations", "options"],
        "properties": {
            "schema_version": {"const": 1},
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
    let options = match operation {
        CuvsOperation::BruteForceKnn => json!({
            "dataset":{"id":"item_id","vector":{"columns":["d0","d1"]}},
            "queries":{"id":"query_id","vector":{"columns":["d0","d1"]}},
            "k": 8,
            "metric":"l2_expanded"
        }),
        CuvsOperation::KMeans => json!({
            "input":{"id":"item_id","vector":{"columns":["d0","d1"]}},
            "n_clusters": 8
        }),
        CuvsOperation::Pca => json!({
            "input":{"id":"item_id","vector":{"columns":["d0","d1"]}},
            "n_components": 2
        }),
    };
    let relations = operation
        .descriptor()
        .relation_roles
        .iter()
        .map(|role| (role.as_str().to_string(), json!({"table": format!("{}_vectors", role.as_str())})))
        .collect::<Map<_, _>>();
    json!([{
        "validation": {
            "function_name": operation.descriptor().sql_function_name,
            "call": {"schema_version": 1, "relations": relations, "options": options}
        }
    }])
}

fn limitations() -> Value {
    json!([
        "Validation resolves named tables or views and reads schemas only; it does not execute relation scans or GPU work.",
        "Bounded cuVS execution is unavailable until the lower-level peak-memory preflight contract exists.",
        "Execution relation arguments support parenthesized subqueries; dry-run validation accepts registered named relations only."
    ])
}

struct ResolvedSchemaOnlyInput {
    plan: LogicalPlan,
    schema: SchemaRef,
}

async fn resolve_schema_only_input(
    session: &dyn Session,
    name: &str,
) -> std::result::Result<ResolvedSchemaOnlyInput, String> {
    let state = session.as_any().downcast_ref::<SessionState>().ok_or_else(|| {
        "GPU validation requires a DataFusion SessionState so named relations can be resolved".to_string()
    })?;
    let table_reference = table_reference(name);
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

fn table_reference(name: &str) -> TableReference {
    let parts = name.split('.').collect::<Vec<_>>();
    match parts.as_slice() {
        [table] => TableReference::bare(*table),
        [schema, table] => TableReference::partial(*schema, *table),
        [catalog, schema, table] => TableReference::full(*catalog, *schema, *table),
        _ => unreachable!("common GPU validation parses one- to three-part table references"),
    }
}

fn unavailable_validation(
    operation: CuvsOperation,
    request: &GpuCallRequest,
    availability: &GpuFunctionAvailability,
) -> GpuCallValidation {
    let reason = availability.unavailable_reason().unwrap_or("required_feature_disabled");
    GpuCallValidation {
        valid: false,
        error_code: Some(reason.to_string()),
        message: match reason {
            "required_feature_disabled" => {
                "cuVS vector algorithm execution requires the cuvs Cargo feature".to_string()
            },
            "bounded_execution_unavailable" => {
                "bounded cuVS execution requires a lower-level peak-memory preflight contract".to_string()
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

fn validation_error(error: &DataFusionError) -> (String, String) {
    if let Some(error) = error.find_nexus_error() {
        let code = match error.code() {
            ErrorCode::FieldNotFound => "column_not_found",
            ErrorCode::UnsupportedType => "unsupported_type",
            _ => error.facts().get("reason").unwrap_or("cuvs_validation_failed"),
        };
        return (code.to_string(), error.message().to_string());
    }
    ("cuvs_validation_failed".to_string(), error.to_string())
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
        CuvsOperation::KMeans => "kmeans",
        CuvsOperation::Pca => "pca",
    }
}

fn output_contract_name(contract: CuvsOutputContract) -> &'static str {
    match contract {
        CuvsOutputContract::BruteForceKnn => "brute_force_knn",
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
                && descriptor.domain == CUVS_DOMAIN
                && descriptor.options_schema.is_object()
                && descriptor.validation_request_schema.is_object()
                && descriptor.examples.is_array()
                && descriptor.limitations.is_array()
        }));
    }

    #[test]
    fn no_cuvs_feature_is_never_reported_as_available() {
        let availability = cuvs_gpu_function_availability(None);
        if cfg!(feature = "cuvs") {
            assert!(availability.is_available());
        } else {
            assert_eq!(availability.unavailable_reason(), Some("required_feature_disabled"));
        }
    }
}
