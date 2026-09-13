//! cuGraph adapter for the shared GPU function discovery and validation catalog.
//!
//! SQL planning and dry-run validation project the same named argument
//! definitions into logical relation children and metadata contracts.

mod descriptions;
mod result_schemas;

use arrow::datatypes::Schema;
use datafusion::{
    catalog::Session,
    common::{DataFusionError, TableReference},
    datasource::provider_as_source,
    execution::SessionState,
    logical_expr::{LogicalPlan, LogicalPlanBuilder},
};
use query_engine::plan::GraphVertexIdTypeSupport;
use serde_json::{Value, json};

use super::{
    CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm, CugraphBfsSource, CugraphBfsSourceVertex, CugraphLogicalNode,
    CugraphRelationRole, CugraphSqlConfig,
    definitions::{
        CugraphAllowedValue, CugraphArgumentDefinition, CugraphArgumentKind, CugraphArgumentType,
        CugraphRelationColumnType, CugraphValueDefinition, CugraphValueRange, arguments_for_algorithm,
    },
    relation_planner::relation_role,
};
use crate::{
    error::DataFusionErrorExt,
    gpu_functions::{
        GpuCallRequest, GpuCallValidation, GpuFunctionAvailability, GpuFunctionDescriptor, GpuFunctionFamily,
    },
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
        let Some(algorithm) = algorithm_for_canonical_name(function_name) else {
            return GpuCallValidation::invalid(
                "unknown_function",
                format!("unknown cuGraph algorithm function '{function_name}'"),
                function_name,
                request.normalized_json(),
            );
        };
        if !self.availability.is_available() {
            return GpuCallValidation::invalid(
                self.availability.unavailable_reason().unwrap_or("required_feature_disabled"),
                "cuGraph execution is unavailable in this build",
                function_name,
                request.normalized_json(),
            );
        }

        let mut inputs = Vec::new();
        let mut roles = Vec::new();
        let mut relation_details = serde_json::Map::new();
        for argument in arguments_for_algorithm(algorithm) {
            let CugraphArgumentKind::Relation { .. } = argument.kind else {
                continue;
            };
            let Some(reference) = request.relations.get(argument.name) else {
                if argument.required {
                    let mut invalid = GpuCallValidation::invalid(
                        "missing_relation",
                        format!("missing required relation role '{}'", argument.name),
                        function_name,
                        request.normalized_json(),
                    );
                    invalid.details = json!({
                        "path": format!("relations.{}", argument.name),
                        "relation_role": argument.name,
                    });
                    return invalid;
                }
                continue;
            };
            match resolve_schema_input(session, reference.reference.clone()).await {
                Ok(input) => {
                    let mut detail = json!({
                        "table": reference.table,
                        "resolved": true,
                        "schema": schema_json(input.schema().as_arrow()),
                    });
                    if argument.name == "edges" {
                        for (fact, column) in [
                            ("src_dtype", "src"),
                            ("dst_dtype", "dst"),
                            ("weight_dtype", "weight"),
                            ("edge_id_dtype", "edge_id"),
                        ] {
                            if let Ok(field) = input.schema().field_with_unqualified_name(column) {
                                detail[fact] = Value::String(format!("{:?}", field.data_type()));
                            }
                        }
                    }
                    relation_details.insert(argument.name.to_string(), detail);
                    roles.push(if argument.name == "edges" {
                        CugraphRelationRole::Edges
                    } else {
                        match relation_role(argument.name) {
                            Ok(role) => role,
                            Err(error) => {
                                return invalid_graph_call(algorithm, request, error.to_string(), relation_details);
                            },
                        }
                    });
                    inputs.push(input);
                },
                Err(message) => {
                    relation_details
                        .insert(argument.name.to_string(), json!({"table": reference.table, "resolved": false}));
                    relation_details.insert("path".to_string(), Value::String(format!("relations.{}", argument.name)));
                    relation_details.insert("relation_role".to_string(), Value::String(argument.name.to_string()));
                    return GpuCallValidation {
                        valid: false,
                        error_code: Some("table_not_found".to_string()),
                        message,
                        function_name: function_name.to_string(),
                        relations_resolved: false,
                        normalized_call: request.normalized_json(),
                        output_schema: json!({}),
                        would_execute_gpu: false,
                        details: Value::Object(relation_details),
                    };
                },
            }
        }
        let mut values = request.options.as_object().cloned().unwrap_or_default();
        let source = match take_source(algorithm, &mut values) {
            Ok(source) => source,
            Err(message) => return invalid_graph_call(algorithm, request, message, relation_details),
        };
        match CugraphLogicalNode::try_new(algorithm, inputs, roles, values, source, &self.config) {
            Ok(node) => {
                let mut normalized = request.normalized_json();
                normalized["options"] =
                    normalized_value_options(algorithm, request.options.as_object().unwrap(), &self.config);
                GpuCallValidation {
                    valid: true,
                    error_code: None,
                    message: "valid".to_string(),
                    function_name: function_name.to_string(),
                    relations_resolved: true,
                    normalized_call: normalized,
                    output_schema: schema_json(node.output_schema().as_ref()),
                    would_execute_gpu: true,
                    details: Value::Object(relation_details),
                }
            },
            Err(error) => invalid_graph_error(algorithm, request, &error, relation_details),
        }
    }
}

fn descriptor(
    algorithm: CugraphAlgorithm,
    config: &CugraphSqlConfig,
    availability: GpuFunctionAvailability,
) -> GpuFunctionDescriptor {
    let result_schemas = value(&result_schemas::result_schemas_json(algorithm));
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
    let relation_roles = arguments_for_algorithm(algorithm)
        .iter()
        .filter_map(|argument| {
            let CugraphArgumentKind::Relation { columns } = argument.kind else {
                return None;
            };
            let mut role = json!({
                "name": argument.name,
                "required": argument.required,
                "description": argument.description,
                "fields": columns.iter().map(|column| json!({
                    "name": column.name,
                    "required": column.required,
                    "type": relation_column_type(column.column_type),
                    "description": column.description,
                })).collect::<Vec<_>>(),
            });
            if argument.name == "edges" {
                role["vertex_id_contract"] = vertex_id_contract(algorithm);
            }
            Some(role)
        })
        .collect::<Vec<_>>();
    GpuFunctionDescriptor {
        function_name: algorithm.function_name().to_string(),
        provider: "cugraph".to_string(),
        signature: signature(algorithm),
        summary: descriptions::summary(algorithm).to_string(),
        relation_roles: Value::Array(relation_roles),
        argument_descriptions: argument_descriptions(algorithm, config),
        options_schema: options_schema(algorithm, config),
        result_schemas,
        validation_request_schema: validation_request_schema(algorithm, config),
        examples: examples(algorithm),
        limitations: value(&descriptions::limitations_json(algorithm)),
        availability,
    }
}

async fn resolve_schema_input(session: &dyn Session, reference: TableReference) -> Result<LogicalPlan, String> {
    let state = session
        .as_any()
        .downcast_ref::<SessionState>()
        .ok_or_else(|| "GPU validation requires a DataFusion SessionState".to_string())?;
    let catalog_options = &state.config_options().catalog;
    let resolved = reference.clone().resolve(&catalog_options.default_catalog, &catalog_options.default_schema);
    let catalog = state
        .catalog_list()
        .catalog(&resolved.catalog)
        .ok_or_else(|| format!("catalog '{}' was not found", resolved.catalog))?;
    let schema = catalog
        .schema(&resolved.schema)
        .ok_or_else(|| format!("schema '{}.{}' was not found", resolved.catalog, resolved.schema))?;
    let provider = schema
        .table(&resolved.table)
        .await
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("table '{}' was not found", reference.to_quoted_string()))?;
    LogicalPlanBuilder::scan(reference, provider_as_source(provider), None)
        .and_then(LogicalPlanBuilder::build)
        .map_err(|error| error.to_string())
}

fn take_source(
    algorithm: CugraphAlgorithm,
    values: &mut serde_json::Map<String, Value>,
) -> Result<Option<CugraphBfsSource>, String> {
    let scalar = values.remove("source_vertex");
    let array = values.remove("source_vertices");
    if scalar.is_some() && array.is_some() {
        return Err("source_vertex and source_vertices are mutually exclusive".to_string());
    }
    let source = match scalar {
        Some(value) => Some(CugraphBfsSource::Scalar(vertex_value(&value)?)),
        None => match array {
            Some(Value::Array(values)) if !values.is_empty() => {
                let vertices = values.iter().map(vertex_value).collect::<Result<Vec<_>, _>>()?;
                let string_domain = vertices[0].is_utf8();
                if vertices.iter().any(|vertex| vertex.is_utf8() != string_domain) {
                    return Err("source_vertices must not mix integer and string values".to_string());
                }
                Some(CugraphBfsSource::LiteralList(vertices))
            },
            Some(_) => return Err("source_vertices must be a non-empty array".to_string()),
            None => None,
        },
    };
    if algorithm == CugraphAlgorithm::Sssp && source.is_none() {
        return Err("cugraph_sssp requires source_vertex".to_string());
    }
    Ok(source)
}

fn vertex_value(value: &Value) -> Result<CugraphBfsSourceVertex, String> {
    match value {
        Value::Number(value) => {
            value.as_i64().map(CugraphBfsSourceVertex::Int64).ok_or_else(|| "vertex value must fit Int64".to_string())
        },
        Value::String(value) => Ok(CugraphBfsSourceVertex::Utf8(value.clone())),
        _ => Err("vertex value must be an integer or string".to_string()),
    }
}

fn normalized_value_options(
    algorithm: CugraphAlgorithm,
    supplied: &serde_json::Map<String, Value>,
    config: &CugraphSqlConfig,
) -> Value {
    Value::Object(
        arguments_for_algorithm(algorithm)
            .iter()
            .filter_map(|argument| {
                let CugraphArgumentKind::Value(definition) = argument.kind else {
                    return None;
                };
                supplied
                    .get(argument.name)
                    .cloned()
                    .or_else(|| configured_default(algorithm, definition, config))
                    .map(|value| (argument.name.to_string(), value))
            })
            .collect(),
    )
}

fn invalid_graph_call(
    algorithm: CugraphAlgorithm,
    request: &GpuCallRequest,
    message: String,
    mut details: serde_json::Map<String, Value>,
) -> GpuCallValidation {
    details.entry("path".to_string()).or_insert_with(|| Value::String("options".to_string()));
    GpuCallValidation {
        valid: false,
        error_code: Some("invalid_argument".to_string()),
        message,
        function_name: algorithm.function_name().to_string(),
        relations_resolved: true,
        normalized_call: request.normalized_json(),
        output_schema: json!({}),
        would_execute_gpu: false,
        details: Value::Object(details),
    }
}

fn invalid_graph_error(
    algorithm: CugraphAlgorithm,
    request: &GpuCallRequest,
    error: &DataFusionError,
    mut details: serde_json::Map<String, Value>,
) -> GpuCallValidation {
    let typed = error.find_algeon_error();
    let error_code = typed.and_then(|error| error.fact_value("reason")).unwrap_or("invalid_argument").to_string();
    if let Some(argument) = typed.and_then(|error| error.fact_value("argument")) {
        details.insert("path".to_string(), Value::String(format!("options.{argument}")));
        details.insert("argument".to_string(), Value::String(argument.to_string()));
    } else if let Some(role) = typed.and_then(|error| error.fact_value("relation_role")) {
        details.insert("path".to_string(), Value::String(format!("relations.{role}")));
        details.insert("relation_role".to_string(), Value::String(role.to_string()));
    } else {
        details.entry("path".to_string()).or_insert_with(|| Value::String("options".to_string()));
    }
    GpuCallValidation {
        valid: false,
        error_code: Some(error_code),
        message: error.to_string(),
        function_name: algorithm.function_name().to_string(),
        relations_resolved: true,
        normalized_call: request.normalized_json(),
        output_schema: json!({}),
        would_execute_gpu: false,
        details: Value::Object(details),
    }
}

fn schema_json(schema: &Schema) -> Value {
    json!({
        "fields": schema.fields().iter().map(|field| json!({
            "name": field.name(),
            "data_type": format!("{:?}", field.data_type()),
            "nullable": field.is_nullable(),
        })).collect::<Vec<_>>()
    })
}

fn algorithm_for_canonical_name(name: &str) -> Option<CugraphAlgorithm> {
    CUGRAPH_OPERATION_REGISTRY
        .iter()
        .find(|definition| definition.sql_function_name == name)
        .map(|definition| definition.algorithm)
}

fn signature(algorithm: CugraphAlgorithm) -> String {
    let arguments = arguments_for_algorithm(algorithm)
        .iter()
        .map(|argument| match argument.kind {
            CugraphArgumentKind::Relation { .. } => format!("{} => (relation subquery)", argument.name),
            CugraphArgumentKind::Value(_) => format!("{} => value", argument.name),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({arguments})", algorithm.function_name())
}

fn argument_descriptions(algorithm: CugraphAlgorithm, config: &CugraphSqlConfig) -> Value {
    Value::Array(
        arguments_for_algorithm(algorithm)
            .iter()
            .filter_map(|argument| {
                let CugraphArgumentKind::Value(definition) = argument.kind else {
                    return None;
                };
                Some(json!({
                    "name": argument.name,
                    "required": argument.required,
                    "description": argument.description,
                    "schema": value_schema(algorithm, definition, config),
                }))
            })
            .collect(),
    )
}

fn validation_request_schema(algorithm: CugraphAlgorithm, config: &CugraphSqlConfig) -> Value {
    let relations = arguments_for_algorithm(algorithm)
        .iter()
        .filter(|argument| matches!(argument.kind, CugraphArgumentKind::Relation { .. }))
        .collect::<Vec<_>>();
    let relation_properties = relations
        .iter()
        .map(|argument| {
            (
                argument.name.to_string(),
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["table"],
                    "properties": {"table": {"type": "string"}}
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let required_relations =
        relations.iter().filter(|argument| argument.required).map(|argument| argument.name).collect::<Vec<_>>();
    let mut schema = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["relations", "options"],
        "properties": {
            "relations": {
                "type": "object",
                "additionalProperties": false,
                "required": required_relations,
                "properties": relation_properties
            },
            "options": options_schema(algorithm, config)
        }
    });
    let constraints = group_constraints(algorithm, true);
    if !constraints.is_empty() {
        schema["allOf"] = Value::Array(constraints);
    }
    schema
}

/// Projects argument groups (`min_count`/`max_count` over their members) as
/// JSON Schema so a standard validator rejects the same selector conflicts the
/// binder rejects. A nullable value member counts as present only when it is
/// not `null`, matching the binder. With `with_relations` false the constraint
/// is scoped to the options object: pairs with a relation member are skipped
/// and a minimum count that a relation could satisfy is not asserted.
fn group_constraints(algorithm: CugraphAlgorithm, with_relations: bool) -> Vec<Value> {
    let arguments = arguments_for_algorithm(algorithm);
    let mut group_names = Vec::new();
    for argument in arguments {
        if let Some(group) = argument.group
            && !group_names.contains(&group.name)
        {
            group_names.push(group.name);
        }
    }
    let mut constraints = Vec::new();
    for name in group_names {
        let all_members = arguments
            .iter()
            .filter(|argument| argument.group.is_some_and(|group| group.name == name))
            .collect::<Vec<_>>();
        let group = all_members[0].group.expect("group members carry their group");
        let has_relation_member =
            all_members.iter().any(|argument| matches!(argument.kind, CugraphArgumentKind::Relation { .. }));
        let members = all_members
            .iter()
            .copied()
            .filter(|argument| with_relations || matches!(argument.kind, CugraphArgumentKind::Value(_)))
            .collect::<Vec<_>>();
        if group.max_count == 1 {
            for (index, first) in members.iter().enumerate() {
                for second in &members[index + 1..] {
                    constraints.push(json!({
                        "not": {"allOf": [argument_present(first, with_relations), argument_present(second, with_relations)]}
                    }));
                }
            }
        }
        if group.min_count >= 1 && (with_relations || !has_relation_member) {
            constraints.push(json!({
                "anyOf": members.iter().map(|member| argument_present(member, with_relations)).collect::<Vec<_>>()
            }));
        }
    }
    constraints
}

fn argument_present(argument: &CugraphArgumentDefinition, nested: bool) -> Value {
    match argument.kind {
        CugraphArgumentKind::Relation { .. } => json!({"properties": {"relations": {"required": [argument.name]}}}),
        CugraphArgumentKind::Value(_) => {
            let present =
                json!({"required": [argument.name], "properties": {argument.name: {"not": {"type": "null"}}}});
            if nested { json!({"properties": {"options": present}}) } else { present }
        },
    }
}

fn options_schema(algorithm: CugraphAlgorithm, config: &CugraphSqlConfig) -> Value {
    let values = arguments_for_algorithm(algorithm)
        .iter()
        .filter_map(|argument| match argument.kind {
            CugraphArgumentKind::Value(definition) => Some((argument, definition)),
            CugraphArgumentKind::Relation { .. } => None,
        })
        .collect::<Vec<_>>();
    let properties = values
        .iter()
        .map(|(argument, definition)| {
            let mut schema = value_schema(algorithm, *definition, config);
            schema["description"] = Value::String(argument.description.to_string());
            (argument.name.to_string(), schema)
        })
        .collect::<serde_json::Map<_, _>>();
    let required =
        values.iter().filter(|(argument, _)| argument.required).map(|(argument, _)| argument.name).collect::<Vec<_>>();
    let mut schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": format!("{} value arguments", algorithm.function_name()),
        "description": "Named cuGraph value arguments. Relation fields are derived from resolved relation schemas.",
        "type": "object",
        "additionalProperties": false,
        "required": required,
        "properties": properties,
    });
    let constraints = group_constraints(algorithm, false);
    if !constraints.is_empty() {
        schema["allOf"] = Value::Array(constraints);
    }
    schema
}

fn value_schema(algorithm: CugraphAlgorithm, definition: CugraphValueDefinition, config: &CugraphSqlConfig) -> Value {
    let mut schema = match definition.value_type {
        CugraphArgumentType::Boolean => json!({"type": "boolean"}),
        CugraphArgumentType::Int64 => json!({"type": "integer", "minimum": i64::MIN, "maximum": i64::MAX}),
        CugraphArgumentType::UInt32 => json!({"type": "integer", "minimum": 0, "maximum": u32::MAX}),
        CugraphArgumentType::UInt64 => json!({"type": "integer", "minimum": 0, "maximum": u64::MAX}),
        CugraphArgumentType::Float64 => json!({"type": "number"}),
        CugraphArgumentType::Utf8 => json!({"type": "string"}),
        CugraphArgumentType::VertexId => json!({"type": ["integer", "string"]}),
        CugraphArgumentType::VertexIdArray => json!({
            "oneOf": [
                {"type": "array", "minItems": 1, "items": {"type": "integer"}},
                {"type": "array", "minItems": 1, "items": {"type": "string"}}
            ]
        }),
    };
    if definition.nullable {
        let original = schema;
        schema = json!({"oneOf": [original, {"type": "null"}]});
    }
    if !definition.allowed_values.is_empty() {
        schema["enum"] = Value::Array(
            definition
                .allowed_values
                .iter()
                .map(|value| match value {
                    CugraphAllowedValue::Boolean(value) => Value::Bool(*value),
                    CugraphAllowedValue::Text(value) => Value::String((*value).to_string()),
                })
                .collect(),
        );
    }
    if let Some(range) = definition.range {
        match range {
            CugraphValueRange::MinInclusiveF64(value) => schema["minimum"] = json!(value),
            CugraphValueRange::MinExclusiveF64(value) => schema["exclusiveMinimum"] = json!(value),
            CugraphValueRange::ClosedF64 { min, max } => {
                schema["minimum"] = json!(min);
                schema["maximum"] = json!(max);
            },
            CugraphValueRange::OpenClosedF64 { min, max } => {
                schema["exclusiveMinimum"] = json!(min);
                schema["maximum"] = json!(max);
            },
            CugraphValueRange::MinInclusiveI64(value) => schema["minimum"] = json!(value),
            CugraphValueRange::MinInclusiveU64(value) => schema["minimum"] = json!(value),
            CugraphValueRange::ClosedU64 { min, max } => {
                schema["minimum"] = json!(min);
                schema["maximum"] = json!(max);
            },
        }
    }
    if let Some(default) = configured_default(algorithm, definition, config) {
        schema["default"] = default;
    }
    schema
}

fn configured_default(
    algorithm: CugraphAlgorithm,
    definition: CugraphValueDefinition,
    config: &CugraphSqlConfig,
) -> Option<Value> {
    use super::definitions::CugraphDefaultValue;

    let graph = algorithm.effective_default_graph_options(&config.default_graph_options);
    match definition.default {
        CugraphDefaultValue::GraphDirected => Some(json!(graph.directed)),
        CugraphDefaultValue::GraphRenumber => Some(json!(graph.renumber)),
        CugraphDefaultValue::GraphConstructionPolicy => Some(json!(graph.construction_policy.as_str())),
        default => default.resolve(algorithm),
    }
}

fn relation_column_type(column_type: CugraphRelationColumnType) -> &'static str {
    match column_type {
        CugraphRelationColumnType::VertexId => "vertex_id",
        CugraphRelationColumnType::EdgeId => "integer_edge_id",
        CugraphRelationColumnType::Numeric => "numeric",
        CugraphRelationColumnType::Float32 => "Float32",
    }
}

fn vertex_id_contract(algorithm: CugraphAlgorithm) -> Value {
    let numeric_endpoint_types = match algorithm.native_algorithm().vertex_id_type_support() {
        GraphVertexIdTypeSupport::Int32OrInt64 => json!(["Int32", "Int64"]),
        GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType => json!(["Int32"]),
    };
    let logical_string = algorithm.definition().logical_string_contract();
    json!({
        "numeric_endpoint_types": numeric_endpoint_types,
        "logical_string_endpoint_types": if logical_string.is_some() {
            json!(["Utf8", "LargeUtf8", "Utf8View"])
        } else {
            json!([])
        },
        "logical_string_output_type": logical_string.map(|_| "Utf8"),
        "native_logical_mapping_type": logical_string.map(|contract| format!("{:?}", contract.native_id_type.data_type())),
        "edge_id_supported_for_logical_string": false,
        "call_specific_authority": "gpu_validate_call",
    })
}

fn examples(algorithm: CugraphAlgorithm) -> Value {
    value(&descriptions::examples_json(algorithm))
}

fn value(json: &str) -> Value {
    serde_json::from_str(json).expect("cuGraph metadata JSON is valid")
}
