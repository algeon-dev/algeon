//! cuGraph adapter for the shared GPU function discovery and validation catalog.
//!
//! Execution remains owned by the cuGraph SQL table-function path.  This
//! module only projects the existing SQL registry and delegates dry-run
//! validation back to the parser and relation-aware schema checks used by that
//! path.

mod descriptions;
mod json_schema;
mod validate;

const ERROR_UNKNOWN_ALGORITHM: &str = "unknown_algorithm";
const ERROR_INVALID_ARGUMENT: &str = "invalid_argument";
const ERROR_TABLE_NOT_FOUND: &str = "table_not_found";
const ERROR_COLUMN_NOT_FOUND: &str = "column_not_found";
const ERROR_UNSUPPORTED_VERTEX_DTYPE: &str = "unsupported_vertex_dtype";
const ERROR_VERTEX_DTYPE_MISMATCH: &str = "vertex_dtype_mismatch";
const ERROR_UNSUPPORTED_WEIGHT_DTYPE: &str = "unsupported_weight_dtype";
const ERROR_UNSUPPORTED_OPTION: &str = "unsupported_option";

use datafusion::catalog::Session;
use serde_json::{Value, json};

use super::{CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm, CugraphSqlConfig};
use crate::gpu_functions::{
    GpuCallRequest, GpuCallValidation, GpuFunctionAvailability, GpuFunctionDescriptor, GpuFunctionFamily,
};

/// cuGraph's family adapter for the common GPU metadata catalog.
#[derive(Debug, Clone)]
pub(crate) struct CugraphGpuFunctionFamily {
    config: CugraphSqlConfig,
    availability: GpuFunctionAvailability,
}

impl CugraphGpuFunctionFamily {
    pub(crate) fn new(config: CugraphSqlConfig) -> Self {
        Self { config, availability: cugraph_gpu_function_availability() }
    }
}

/// Build-level cuGraph execution availability projected through the common
/// discovery catalog. The SQL parser and schema validator remain available in
/// either build; only native graph execution is feature-owned.
pub(crate) fn cugraph_gpu_function_availability() -> GpuFunctionAvailability {
    if cfg!(feature = "cugraph") {
        GpuFunctionAvailability::Available
    } else {
        GpuFunctionAvailability::Unavailable { reason: "required_feature_disabled".to_string() }
    }
}

#[async_trait::async_trait]
impl GpuFunctionFamily for CugraphGpuFunctionFamily {
    fn family_id(&self) -> &'static str {
        "cugraph"
    }

    fn descriptors(&self) -> Vec<GpuFunctionDescriptor> {
        CUGRAPH_OPERATION_REGISTRY
            .iter()
            .map(|definition| descriptor(definition.algorithm, &self.config, self.availability.clone()))
            .collect()
    }

    async fn validate(
        &self,
        session: &dyn Session,
        function_name: &str,
        request: &GpuCallRequest,
    ) -> GpuCallValidation {
        let Some(edges) = request.relations.get("edges") else {
            // The catalog checks descriptor-declared required roles before it
            // calls a family. Keep this defensive result structured if a
            // family is ever called directly.
            return GpuCallValidation::invalid(
                "missing_relation",
                "missing required relation role 'edges'",
                function_name,
                request.normalized_json(),
            );
        };
        let Some(algorithm) = algorithm_for_canonical_name(function_name) else {
            return GpuCallValidation::invalid(
                "unknown_function",
                format!("unknown cuGraph algorithm function '{function_name}'"),
                function_name,
                request.normalized_json(),
            );
        };

        let row = validate::validate_graph_call(
            session,
            algorithm.function_name(),
            &edges.table,
            &request.options,
            &self.config,
        )
        .await;
        let normalized_options: Value =
            serde_json::from_str(&row.normalized_options_json).expect("cuGraph validation options JSON is valid");
        let normalized_call = json!({
            "schema_version": request.schema_version,
            "relations": { "edges": { "table": edges.table } },
            "options": normalized_options,
        });
        let output_schema: Value =
            serde_json::from_str(&row.output_schema_json).expect("cuGraph validation schema JSON is valid");
        let details = json!({
            "edges": {
                "table": edges.table,
                "resolved": row.table_resolved,
                "src_col_resolved": row.src_col_resolved,
                "dst_col_resolved": row.dst_col_resolved,
                "weight_col_resolved": row.weight_col_resolved,
                "src_dtype": row.src_dtype,
                "dst_dtype": row.dst_dtype,
                "weight_dtype": row.weight_dtype,
            },
        });

        let relations_resolved = row.table_resolved;
        GpuCallValidation {
            valid: row.valid,
            error_code: row.error_code,
            message: row.message,
            function_name: row.function_name,
            relations_resolved,
            normalized_call,
            output_schema,
            would_execute_gpu: row.would_execute_gpu,
            details,
        }
    }
}

fn descriptor(
    algorithm: CugraphAlgorithm,
    config: &CugraphSqlConfig,
    availability: GpuFunctionAvailability,
) -> GpuFunctionDescriptor {
    let result_schemas = value(&json_schema::result_schemas_json(algorithm));
    let result_schemas = match result_schemas {
        Value::Object(mut object) => {
            object.insert(
                "field_descriptions".to_string(),
                value(&descriptions::result_field_descriptions_json(algorithm)),
            );
            Value::Object(object)
        },
        value => value,
    };
    GpuFunctionDescriptor {
        function_name: algorithm.function_name().to_string(),
        provider: "cugraph".to_string(),
        signature: signature(algorithm),
        summary: descriptions::summary(algorithm).to_string(),
        relation_roles: json!([{
            "name": "edges",
            "description": "Required edge-list table or view for graph construction.",
            "vertex_id_contract": json_schema::vertex_id_contract_json(algorithm),
        }]),
        argument_descriptions: argument_descriptions(algorithm),
        options_schema: value(&json_schema::options_schema_json(algorithm, config)),
        result_schemas,
        validation_request_schema: validation_request_schema(),
        examples: examples(algorithm),
        limitations: value(&descriptions::limitations_json(algorithm)),
        availability,
    }
}

fn algorithm_for_canonical_name(name: &str) -> Option<CugraphAlgorithm> {
    CUGRAPH_OPERATION_REGISTRY
        .iter()
        .find(|definition| definition.sql_function_name == name)
        .map(|definition| definition.algorithm)
}

fn signature(algorithm: CugraphAlgorithm) -> String {
    let function_name = algorithm.function_name();
    if algorithm == CugraphAlgorithm::MinimumSpanningTree {
        format!("{function_name}(table_name, src_col, dst_col, weight_col [, options_json])")
    } else if algorithm.is_similarity() || algorithm == CugraphAlgorithm::PersonalizedPageRank {
        format!("{function_name}(table_name, src_col, dst_col, weight_col, options_json)")
    } else if algorithm == CugraphAlgorithm::Bfs {
        format!(
            "{function_name}(table_name, source_vertex_or_vertices [, src_col, dst_col [, weight_col [, options_json]]])"
        )
    } else if algorithm.requires_source_vertex() {
        format!("{function_name}(table_name, source_vertex [, src_col, dst_col [, weight_col [, options_json]]])")
    } else {
        format!("{function_name}(table_name [, src_col, dst_col [, weight_col [, options_json]]])")
    }
}

fn argument_descriptions(algorithm: CugraphAlgorithm) -> Value {
    let mut arguments = value(&descriptions::argument_descriptions_json(algorithm));
    if let Value::Array(arguments) = &mut arguments {
        arguments.retain(|argument| {
            !matches!(argument.get("name").and_then(Value::as_str), Some("table_name" | "options_json"))
        });
    }
    arguments
}

fn validation_request_schema() -> Value {
    json!({
        "type": "object",
        "description": "The common validation envelope supports only version 1. There is no version 0 reader or version negotiation; any other schema_version is rejected before cuGraph family dispatch.",
        "additionalProperties": false,
        "required": ["schema_version", "relations", "options"],
        "properties": {
            "schema_version": {
                "const": 1,
                "description": "Common-envelope forward-compatibility marker. Only 1 dispatches to cuGraph validation; unknown versions are rejected before family dispatch."
            },
            "relations": {
                "type": "object",
                "additionalProperties": false,
                "required": ["edges"],
                "properties": {
                    "edges": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["table"],
                        "properties": { "table": { "type": "string" } }
                    }
                }
            },
            "options": {
                "type": "object",
                "description": "Flat cuGraph option object (no nested 'options'): src_col, dst_col, weight_col, source_vertex, graph options, and algorithm options. See options_schema_json."
            }
        }
    })
}

fn examples(algorithm: CugraphAlgorithm) -> Value {
    value(&descriptions::examples_json(algorithm))
}

fn value(json: &str) -> Value {
    serde_json::from_str(json).expect("cuGraph metadata JSON is valid")
}
