use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::{
    catalog::{Session, TableProvider},
    common::{Result, ScalarValue, TableReference},
    logical_expr::{Expr, lit},
};
use serde_json::{Map, Value};

use super::{
    super::{
        CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsSource, CugraphBfsSourceVertex,
        CugraphSqlConfig, CugraphTableReference, CugraphVertexIdDomain, bfs_result_schema_for_vertex_domain,
        datafusion_table_reference, parse_cugraph_algorithm_spec, stable_result_schema, vertex_id_domain_for_call,
    },
    ERROR_ALGORITHM_DISABLED, ERROR_COLUMN_NOT_FOUND, ERROR_INVALID_ARGUMENT, ERROR_INVALID_CALL_JSON, ERROR_OK,
    ERROR_TABLE_NOT_FOUND, ERROR_UNKNOWN_ALGORITHM, ERROR_UNSUPPORTED_OPTION, ERROR_UNSUPPORTED_VERTEX_DTYPE,
    ERROR_UNSUPPORTED_WEIGHT_DTYPE, ERROR_VERTEX_DTYPE_MISMATCH, SEVERITY_ERROR, SEVERITY_OK,
    algorithm_for_metadata_name,
    json_schema::{data_type_label, normalized_options_json, schema_json},
};

/// Single-row diagnostic payload returned by graph SQL validation calls.
#[derive(Debug, Clone)]
pub(super) struct ValidationRow {
    pub(super) valid: bool,
    pub(super) severity: String,
    pub(super) error_code: String,
    pub(super) message: String,
    pub(super) function_name: String,
    pub(super) algorithm_enabled: bool,
    pub(super) table_resolved: bool,
    pub(super) src_col_resolved: bool,
    pub(super) dst_col_resolved: bool,
    pub(super) weight_col_resolved: bool,
    pub(super) src_dtype: String,
    pub(super) dst_dtype: String,
    pub(super) weight_dtype: String,
    pub(super) output_schema_json: String,
    pub(super) normalized_options_json: String,
    pub(super) would_execute_gpu: bool,
}

impl ValidationRow {
    fn invalid(function_name: &str, code: &str, message: impl Into<String>) -> Self {
        Self {
            valid: false,
            severity: SEVERITY_ERROR.to_string(),
            error_code: code.to_string(),
            message: message.into(),
            function_name: function_name.to_string(),
            algorithm_enabled: false,
            table_resolved: false,
            src_col_resolved: false,
            dst_col_resolved: false,
            weight_col_resolved: false,
            src_dtype: String::new(),
            dst_dtype: String::new(),
            weight_dtype: String::new(),
            output_schema_json: "{}".to_string(),
            normalized_options_json: "{}".to_string(),
            would_execute_gpu: false,
        }
    }

    fn with_algorithm(mut self, enabled: bool, output_schema_json: String) -> Self {
        self.algorithm_enabled = enabled;
        self.output_schema_json = output_schema_json;
        self
    }
}

pub(super) async fn cugraph_validate_call(
    state: &dyn Session,
    function_name: &str,
    table_name: &str,
    call_json: &str,
    config: &CugraphSqlConfig,
) -> ValidationRow {
    let Some(algorithm) = algorithm_for_metadata_name(function_name) else {
        return ValidationRow::invalid(
            function_name,
            ERROR_UNKNOWN_ALGORITHM,
            format!("unknown cuGraph algorithm function '{function_name}'"),
        );
    };
    let canonical_function_name = algorithm.function_name();
    let output_schema_json = schema_json(&stable_result_schema(algorithm));
    if !config.allows(algorithm) {
        return ValidationRow::invalid(
            canonical_function_name,
            ERROR_ALGORITHM_DISABLED,
            format!("cuGraph algorithm {algorithm:?} is not enabled in this session"),
        )
        .with_algorithm(false, output_schema_json);
    }

    let call = match parse_validate_call_json(algorithm, call_json) {
        Ok(call) => call,
        Err((code, message)) => {
            return ValidationRow::invalid(canonical_function_name, code, message)
                .with_algorithm(true, output_schema_json);
        },
    };

    let mut spec = match build_validate_spec(algorithm, table_name, &call, config) {
        Ok(spec) => spec,
        Err(error) => {
            return ValidationRow::invalid(
                canonical_function_name,
                parse_error_code(&error.to_string()),
                error.to_string(),
            )
            .with_algorithm(true, output_schema_json);
        },
    };
    let mut output_schema_json = schema_json(&spec.output_schema);

    let table_ref = datafusion_table_reference(&spec.table_name);
    let provider = match resolve_table_provider(state, table_ref).await {
        Ok(provider) => provider,
        Err(message) => {
            return ValidationRow::invalid(canonical_function_name, ERROR_TABLE_NOT_FOUND, message)
                .with_algorithm(true, output_schema_json);
        },
    };
    let schema = provider.schema();
    let src = schema.field_with_name(&spec.src_col).ok();
    let dst = schema.field_with_name(&spec.dst_col).ok();
    let weight = spec.weight_col.as_ref().and_then(|name| schema.field_with_name(name).ok());
    let edge_id = spec.edge_id_col.as_ref().and_then(|name| schema.field_with_name(name).ok());
    let src_dtype = src.map(|field| data_type_label(field.data_type())).unwrap_or_default();
    let dst_dtype = dst.map(|field| data_type_label(field.data_type())).unwrap_or_default();
    let weight_dtype = weight.map(|field| data_type_label(field.data_type())).unwrap_or_default();

    if src.is_none()
        || dst.is_none()
        || spec.weight_col.as_ref().is_some_and(|_| weight.is_none())
        || spec.edge_id_col.as_ref().is_some_and(|_| edge_id.is_none())
    {
        let mut missing = Vec::new();
        if src.is_none() {
            missing.push(spec.src_col.as_str());
        }
        if dst.is_none() {
            missing.push(spec.dst_col.as_str());
        }
        if spec.weight_col.as_ref().is_some_and(|_| weight.is_none()) {
            missing.push(spec.weight_col.as_deref().unwrap_or_default());
        }
        if spec.edge_id_col.as_ref().is_some_and(|_| edge_id.is_none()) {
            missing.push(spec.edge_id_col.as_deref().unwrap_or_default());
        }
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: src.is_some(),
            dst_col_resolved: dst.is_some(),
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                canonical_function_name,
                ERROR_COLUMN_NOT_FOUND,
                format!("cuGraph edge source is missing column(s): {}", missing.join(",")),
            )
            .with_algorithm(true, output_schema_json)
        };
    }

    let src_field = src.expect("checked above");
    let dst_field = dst.expect("checked above");
    let edge_id_field = edge_id;
    let vertex_domain =
        match vertex_id_domain_for_call(algorithm, &spec.params, src_field.data_type(), dst_field.data_type()) {
            Ok(domain) => domain,
            Err(error) => {
                let message = error.to_string();
                return ValidationRow {
                    table_resolved: true,
                    src_col_resolved: true,
                    dst_col_resolved: true,
                    weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
                    src_dtype,
                    dst_dtype,
                    weight_dtype,
                    ..ValidationRow::invalid(canonical_function_name, vertex_domain_error_code(&message), message)
                        .with_algorithm(true, output_schema_json)
                };
            },
        };
    if let CugraphAlgorithmParams::Bfs(params) = &spec.params {
        spec.output_schema = bfs_result_schema_for_vertex_domain(params, &vertex_domain);
        output_schema_json = schema_json(&spec.output_schema);
    }
    if let Some(_edge_id_field) = edge_id_field
        && vertex_domain.is_string_logical()
    {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                canonical_function_name,
                ERROR_UNSUPPORTED_OPTION,
                "cuGraph BFS logical string vertex-id mapping does not yet support edge_id_col",
            )
            .with_algorithm(true, output_schema_json)
        };
    }
    if let Some(edge_id_field) = edge_id_field
        && !vertex_domain.accepts(edge_id_field.data_type())
    {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                canonical_function_name,
                ERROR_VERTEX_DTYPE_MISMATCH,
                format!(
                    "cuGraph edge-id column '{}' must have dtype {}",
                    spec.edge_id_col.as_deref().unwrap_or(""),
                    vertex_domain.label()
                ),
            )
            .with_algorithm(true, output_schema_json)
        };
    }
    if let Some(weight_field) = weight
        && !is_supported_weight_dtype(weight_field.data_type())
    {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: true,
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                canonical_function_name,
                ERROR_UNSUPPORTED_WEIGHT_DTYPE,
                format!("cuGraph weight column '{}' has unsupported dtype", spec.weight_col.as_deref().unwrap_or("")),
            )
            .with_algorithm(true, output_schema_json)
        };
    }

    if let Some(row) = validate_bfs_predicate_sources(
        state,
        canonical_function_name,
        &spec,
        &vertex_domain,
        output_schema_json.clone(),
    )
    .await
    {
        return row;
    }
    if let Some(row) = validate_personalized_pagerank_source(
        state,
        canonical_function_name,
        &spec,
        &vertex_domain,
        output_schema_json.clone(),
    )
    .await
    {
        return row;
    }

    ValidationRow {
        valid: true,
        severity: SEVERITY_OK.to_string(),
        error_code: ERROR_OK.to_string(),
        message: "cuGraph call is statically valid".to_string(),
        function_name: canonical_function_name.to_string(),
        algorithm_enabled: true,
        table_resolved: true,
        src_col_resolved: true,
        dst_col_resolved: true,
        weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
        src_dtype,
        dst_dtype,
        weight_dtype,
        output_schema_json,
        normalized_options_json: normalized_options_json(&spec),
        would_execute_gpu: true,
    }
}

async fn validate_bfs_predicate_sources(
    state: &dyn Session,
    function_name: &str,
    spec: &CugraphAlgorithmSpec,
    vertex_domain: &CugraphVertexIdDomain,
    output_schema_json: String,
) -> Option<ValidationRow> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return None;
    };
    let predicates = &params.predicates;
    if let Some(input) = params.source.relation()
        && let Some(row) = validate_predicate_columns(
            state,
            function_name,
            &input.table_name,
            &[("source_vertices", input.column.as_str())],
            vertex_domain,
            &output_schema_json,
        )
        .await
    {
        return Some(row);
    }
    if let Some(input) = &predicates.include_vertices
        && let Some(row) = validate_predicate_columns(
            state,
            function_name,
            &input.table_name,
            &[("include_vertices", input.column.as_str())],
            vertex_domain,
            &output_schema_json,
        )
        .await
    {
        return Some(row);
    }
    if let Some(input) = &predicates.exclude_vertices
        && let Some(row) = validate_predicate_columns(
            state,
            function_name,
            &input.table_name,
            &[("exclude_vertices", input.column.as_str())],
            vertex_domain,
            &output_schema_json,
        )
        .await
    {
        return Some(row);
    }
    if let Some(input) = &predicates.target_vertices
        && let Some(row) = validate_predicate_columns(
            state,
            function_name,
            &input.table_name,
            &[("target_vertices", input.column.as_str())],
            vertex_domain,
            &output_schema_json,
        )
        .await
    {
        return Some(row);
    }
    if let Some(input) = &predicates.include_edge_ids
        && let Some(row) = validate_predicate_columns(
            state,
            function_name,
            &input.table_name,
            &[("include_edge_ids", input.column.as_str())],
            vertex_domain,
            &output_schema_json,
        )
        .await
    {
        return Some(row);
    }
    if let Some(input) = &predicates.include_edges {
        let mut columns = vec![
            ("include_edges_source", input.source_col.as_str()),
            ("include_edges_destination", input.destination_col.as_str()),
        ];
        if let Some(edge_id_col) = &input.edge_id_col {
            columns.push(("include_edges_edge_id", edge_id_col.as_str()));
        }
        if let Some(row) = validate_predicate_columns(
            state,
            function_name,
            &input.table_name,
            &columns,
            vertex_domain,
            &output_schema_json,
        )
        .await
        {
            return Some(row);
        }
    }
    None
}

async fn validate_personalized_pagerank_source(
    state: &dyn Session,
    function_name: &str,
    spec: &CugraphAlgorithmSpec,
    vertex_domain: &CugraphVertexIdDomain,
    output_schema_json: String,
) -> Option<ValidationRow> {
    let CugraphAlgorithmParams::PersonalizedPageRank(params) = &spec.params else {
        return None;
    };
    let input = &params.personalization;
    let provider = match resolve_table_provider(state, datafusion_table_reference(&input.table_name)).await {
        Ok(provider) => provider,
        Err(message) => {
            return Some(predicate_validation_error(
                function_name,
                ERROR_TABLE_NOT_FOUND,
                message,
                &output_schema_json,
            ));
        },
    };
    let schema = provider.schema();
    let vertex = match schema.field_with_name(&input.vertex_col) {
        Ok(field) => field,
        Err(_) => {
            return Some(predicate_validation_error(
                function_name,
                ERROR_COLUMN_NOT_FOUND,
                format!(
                    "cuGraph personalized PageRank source '{}' is missing vertex column '{}'",
                    input.table_name, input.vertex_col
                ),
                &output_schema_json,
            ));
        },
    };
    if !vertex_domain.accepts(vertex.data_type()) {
        return Some(predicate_validation_error(
            function_name,
            ERROR_VERTEX_DTYPE_MISMATCH,
            format!(
                "cuGraph personalized PageRank vertex column '{}' must have dtype {}",
                input.vertex_col,
                vertex_domain.label()
            ),
            &output_schema_json,
        ));
    }
    let value = match schema.field_with_name(&input.value_col) {
        Ok(field) => field,
        Err(_) => {
            return Some(predicate_validation_error(
                function_name,
                ERROR_COLUMN_NOT_FOUND,
                format!(
                    "cuGraph personalized PageRank source '{}' is missing value column '{}'",
                    input.table_name, input.value_col
                ),
                &output_schema_json,
            ));
        },
    };
    if !is_supported_weight_dtype(value.data_type()) {
        return Some(predicate_validation_error(
            function_name,
            ERROR_UNSUPPORTED_WEIGHT_DTYPE,
            format!(
                "cuGraph personalized PageRank value column '{}' must have dtype Float32 or Float64",
                input.value_col
            ),
            &output_schema_json,
        ));
    }
    None
}

async fn validate_predicate_columns(
    state: &dyn Session,
    function_name: &str,
    table_name: &CugraphTableReference,
    columns: &[(&str, &str)],
    vertex_domain: &CugraphVertexIdDomain,
    output_schema_json: &str,
) -> Option<ValidationRow> {
    let provider = match resolve_table_provider(state, datafusion_table_reference(table_name)).await {
        Ok(provider) => provider,
        Err(message) => {
            return Some(predicate_validation_error(function_name, ERROR_TABLE_NOT_FOUND, message, output_schema_json));
        },
    };
    let schema = provider.schema();
    for (role, column) in columns {
        let field = match schema.field_with_name(column) {
            Ok(field) => field,
            Err(_) => {
                return Some(predicate_validation_error(
                    function_name,
                    ERROR_COLUMN_NOT_FOUND,
                    format!("cuGraph BFS predicate source '{}' is missing column '{}'", table_name, column),
                    output_schema_json,
                ));
            },
        };
        if !vertex_domain.accepts(field.data_type()) {
            return Some(predicate_validation_error(
                function_name,
                ERROR_VERTEX_DTYPE_MISMATCH,
                format!(
                    "cuGraph BFS predicate column '{}' for {role} must have dtype {}",
                    column,
                    vertex_domain.label()
                ),
                output_schema_json,
            ));
        }
    }
    None
}

fn predicate_validation_error(
    function_name: &str,
    code: &str,
    message: impl Into<String>,
    output_schema_json: &str,
) -> ValidationRow {
    ValidationRow {
        table_resolved: true,
        src_col_resolved: true,
        dst_col_resolved: true,
        weight_col_resolved: true,
        ..ValidationRow::invalid(function_name, code, message).with_algorithm(true, output_schema_json.to_string())
    }
}

#[derive(Debug, Clone)]
struct ValidateCallJson {
    source_vertex: Option<CugraphBfsSource>,
    src_col: String,
    dst_col: String,
    weight_col: Option<String>,
    options: Map<String, Value>,
}

fn parse_validate_call_json(
    algorithm: CugraphAlgorithm,
    call_json: &str,
) -> std::result::Result<ValidateCallJson, (&'static str, String)> {
    let value = serde_json::from_str::<Value>(call_json)
        .map_err(|error| (ERROR_INVALID_CALL_JSON, format!("invalid cuGraph call JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| (ERROR_INVALID_CALL_JSON, "cuGraph call JSON must be an object".to_string()))?;
    for key in object.keys() {
        if !matches!(key.as_str(), "source_vertex" | "src_col" | "dst_col" | "weight_col" | "options") {
            return Err((ERROR_INVALID_ARGUMENT, format!("unknown cuGraph validate_call key '{key}'")));
        }
    }

    let options = match object.get("options") {
        Some(value) => value
            .as_object()
            .cloned()
            .ok_or_else(|| (ERROR_INVALID_CALL_JSON, "cuGraph call JSON options must be an object".to_string()))?,
        None => Map::new(),
    };
    let source_vertex = match object.get("source_vertex") {
        Some(Value::Null) if algorithm == CugraphAlgorithm::Bfs => None,
        Some(value) => Some(json_source_vertex(value, "source_vertex", algorithm)?),
        None => None,
    };
    if algorithm == CugraphAlgorithm::Bfs && source_vertex.is_none() && !bfs_options_include_source_relation(&options) {
        return Err((
            ERROR_INVALID_ARGUMENT,
            "cugraph_bfs requires source_vertex or source_vertices_table/source_vertex_col".to_string(),
        ));
    }
    if algorithm.requires_source_vertex() && algorithm != CugraphAlgorithm::Bfs && source_vertex.is_none() {
        return Err((ERROR_INVALID_ARGUMENT, format!("{} requires source_vertex", algorithm.function_name())));
    }
    if !algorithm.requires_source_vertex() && source_vertex.is_some() {
        return Err((ERROR_INVALID_ARGUMENT, format!("{} does not accept source_vertex", algorithm.function_name())));
    }
    let src_col = match object.get("src_col") {
        Some(value) => json_string(value, "src_col")?,
        None => "src".to_string(),
    };
    let dst_col = match object.get("dst_col") {
        Some(value) => json_string(value, "dst_col")?,
        None => "dst".to_string(),
    };
    let weight_col = match object.get("weight_col") {
        Some(Value::Null) | None => None,
        Some(value) => Some(json_string(value, "weight_col")?),
    };
    Ok(ValidateCallJson { source_vertex, src_col, dst_col, weight_col, options })
}

fn bfs_options_include_source_relation(options: &Map<String, Value>) -> bool {
    options.contains_key("source_vertices_table") || options.contains_key("source_vertex_col")
}

fn build_validate_spec(
    algorithm: CugraphAlgorithm,
    table_name: &str,
    call: &ValidateCallJson,
    config: &CugraphSqlConfig,
) -> Result<CugraphAlgorithmSpec> {
    let mut args = vec![lit(table_name)];
    if algorithm.requires_source_vertex() {
        if let Some(source_vertex) = &call.source_vertex {
            args.push(source_vertex_expr(source_vertex));
        } else {
            args.push(lit(ScalarValue::Null));
        }
    }
    args.push(lit(call.src_col.as_str()));
    args.push(lit(call.dst_col.as_str()));
    match &call.weight_col {
        Some(weight_col) => args.push(lit(weight_col.as_str())),
        None => args.push(lit(ScalarValue::Null)),
    }
    if !call.options.is_empty() {
        args.push(lit(Value::Object(call.options.clone()).to_string()));
    }
    parse_cugraph_algorithm_spec(algorithm, &args, config)
}

async fn resolve_table_provider(
    state: &dyn Session,
    table_ref: TableReference,
) -> std::result::Result<Arc<dyn TableProvider>, String> {
    let session_state = state
        .as_any()
        .downcast_ref::<datafusion::execution::SessionState>()
        .ok_or_else(|| "cuGraph validation requires a DataFusion SessionState for catalog resolution".to_string())?;
    let schema = session_state
        .schema_for_ref(table_ref.clone())
        .map_err(|error| format!("failed to resolve cuGraph edge source '{}': {error}", table_ref))?;
    schema
        .table(table_ref.table())
        .await
        .map_err(|error| format!("failed to resolve cuGraph edge source '{}': {error}", table_ref))?
        .ok_or_else(|| format!("No table named '{}' for cuGraph edge source", table_ref))
}

fn parse_error_code(message: &str) -> &'static str {
    if message.contains("unknown option") || message.contains("requires vertex_radius side input") {
        ERROR_UNSUPPORTED_OPTION
    } else if message.contains("is not enabled in this session") {
        ERROR_ALGORITHM_DISABLED
    } else {
        ERROR_INVALID_ARGUMENT
    }
}

fn vertex_domain_error_code(message: &str) -> &'static str {
    if message.contains("unsupported dtype") { ERROR_UNSUPPORTED_VERTEX_DTYPE } else { ERROR_VERTEX_DTYPE_MISMATCH }
}

fn json_string(value: &Value, key: &str) -> std::result::Result<String, (&'static str, String)> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| (ERROR_INVALID_ARGUMENT, format!("cuGraph call JSON key '{key}' must be a string")))
}

fn json_i64(value: &Value, key: &str) -> std::result::Result<i64, (&'static str, String)> {
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    if let Some(value) = value.as_u64() {
        return i64::try_from(value)
            .map_err(|_| (ERROR_INVALID_ARGUMENT, format!("cuGraph call JSON key '{key}' exceeds i64::MAX")));
    }
    Err((ERROR_INVALID_ARGUMENT, format!("cuGraph call JSON key '{key}' must be an integer")))
}

fn json_source_vertex(
    value: &Value,
    key: &str,
    algorithm: CugraphAlgorithm,
) -> std::result::Result<CugraphBfsSource, (&'static str, String)> {
    if let Some(values) = value.as_array() {
        if algorithm != CugraphAlgorithm::Bfs {
            return Err((ERROR_INVALID_ARGUMENT, format!("cuGraph call JSON key '{key}' must be an integer")));
        }
        if values.is_empty() {
            return Err((ERROR_INVALID_ARGUMENT, format!("cuGraph call JSON key '{key}' list must not be empty")));
        }
        let values = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if value.is_null() {
                    return Err((
                        ERROR_INVALID_ARGUMENT,
                        format!("cuGraph call JSON key '{key}' list must not contain null element at index {index}"),
                    ));
                }
                json_source_vertex_literal(value, key, algorithm)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let first_is_utf8 = values.first().is_some_and(CugraphBfsSourceVertex::is_utf8);
        if values.iter().any(|value| value.is_utf8() != first_is_utf8) {
            return Err((
                ERROR_INVALID_ARGUMENT,
                format!("cuGraph call JSON key '{key}' list must not mix integer and string elements"),
            ));
        }
        return Ok(CugraphBfsSource::LiteralList(values));
    }
    json_source_vertex_literal(value, key, algorithm).map(CugraphBfsSource::Scalar)
}

fn json_source_vertex_literal(
    value: &Value,
    key: &str,
    algorithm: CugraphAlgorithm,
) -> std::result::Result<CugraphBfsSourceVertex, (&'static str, String)> {
    if let Some(value) = value.as_str() {
        return if algorithm == CugraphAlgorithm::Bfs {
            Ok(CugraphBfsSourceVertex::Utf8(value.to_string()))
        } else {
            Err((ERROR_INVALID_ARGUMENT, format!("cuGraph call JSON key '{key}' must be an integer")))
        };
    }
    json_i64(value, key).map(CugraphBfsSourceVertex::Int64)
}

fn source_vertex_expr(source_vertex: &CugraphBfsSource) -> Expr {
    match source_vertex {
        CugraphBfsSource::Scalar(CugraphBfsSourceVertex::Int64(value)) => lit(*value),
        CugraphBfsSource::Scalar(CugraphBfsSourceVertex::Utf8(value)) => lit(value.as_str()),
        CugraphBfsSource::LiteralList(values) => {
            let first_is_utf8 = values.first().is_some_and(CugraphBfsSourceVertex::is_utf8);
            let scalars = values
                .iter()
                .map(|value| match value {
                    CugraphBfsSourceVertex::Int64(value) => ScalarValue::Int64(Some(*value)),
                    CugraphBfsSourceVertex::Utf8(value) => ScalarValue::Utf8(Some(value.clone())),
                })
                .collect::<Vec<_>>();
            let data_type = if first_is_utf8 { DataType::Utf8 } else { DataType::Int64 };
            lit(ScalarValue::List(ScalarValue::new_list(&scalars, &data_type, false)))
        },
        CugraphBfsSource::Relation(_) => lit(ScalarValue::Null),
    }
}

fn is_supported_weight_dtype(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Float32 | DataType::Float64)
}
