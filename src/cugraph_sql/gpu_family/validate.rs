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
        CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsSource,
        CugraphBfsSourceVertex, CugraphSqlConfig, CugraphVertexIdDomain, PredicateSourceColumnRole,
        datafusion_table_reference, parse_cugraph_algorithm_spec, result_schema_for_vertex_domain,
        validate_similarity_pair_table_schema, validate_vertex_id_domain_support_for_call, vertex_id_domain_for_call,
    },
    ERROR_COLUMN_NOT_FOUND, ERROR_INVALID_ARGUMENT, ERROR_TABLE_NOT_FOUND, ERROR_UNKNOWN_ALGORITHM,
    ERROR_UNSUPPORTED_OPTION, ERROR_UNSUPPORTED_VERTEX_DTYPE, ERROR_UNSUPPORTED_WEIGHT_DTYPE,
    ERROR_VERTEX_DTYPE_MISMATCH,
    json_schema::{data_type_label, metadata_output_schema_json, normalized_options_json, schema_json},
};
use crate::error::{DataFusionErrorExt, ErrorCode};

/// Single-row diagnostic payload returned by graph SQL validation calls.
#[derive(Debug, Clone)]
pub(super) struct ValidationRow {
    pub(super) valid: bool,
    pub(super) error_code: Option<String>,
    pub(super) message: String,
    pub(super) function_name: String,
    pub(super) table_resolved: bool,
    pub(super) src_col_resolved: bool,
    pub(super) dst_col_resolved: bool,
    pub(super) weight_col_resolved: bool,
    pub(super) src_dtype: Option<String>,
    pub(super) dst_dtype: Option<String>,
    pub(super) weight_dtype: Option<String>,
    pub(super) output_schema_json: String,
    pub(super) normalized_options_json: String,
    pub(super) would_execute_gpu: bool,
}

impl ValidationRow {
    fn invalid(function_name: &str, code: &str, message: impl Into<String>) -> Self {
        Self {
            valid: false,
            error_code: Some(code.to_string()),
            message: message.into(),
            function_name: function_name.to_string(),
            table_resolved: false,
            src_col_resolved: false,
            dst_col_resolved: false,
            weight_col_resolved: false,
            src_dtype: None,
            dst_dtype: None,
            weight_dtype: None,
            output_schema_json: "{}".to_string(),
            normalized_options_json: "{}".to_string(),
            would_execute_gpu: false,
        }
    }

    fn with_output_schema(mut self, output_schema_json: String) -> Self {
        self.output_schema_json = output_schema_json;
        self
    }
}

pub(super) async fn validate_graph_call(
    state: &dyn Session,
    function_name: &str,
    table_name: &str,
    options: &Value,
    config: &CugraphSqlConfig,
) -> ValidationRow {
    let Some(algorithm) = CUGRAPH_OPERATION_REGISTRY
        .iter()
        .find(|definition| definition.sql_function_name == function_name)
        .map(|definition| definition.algorithm)
    else {
        return ValidationRow::invalid(
            function_name,
            ERROR_UNKNOWN_ALGORITHM,
            format!("unknown cuGraph algorithm function '{function_name}'"),
        );
    };
    let canonical_function_name = algorithm.function_name();
    let output_schema_json = metadata_output_schema_json(algorithm);

    let call = match parse_validate_options(algorithm, options) {
        Ok(call) => call,
        Err((code, message)) => {
            return ValidationRow::invalid(canonical_function_name, code, message)
                .with_output_schema(output_schema_json);
        },
    };

    let mut spec = match build_validate_spec(algorithm, table_name, &call, config) {
        Ok(spec) => spec,
        Err(error) => {
            return ValidationRow::invalid(canonical_function_name, validation_error_code(&error), error.to_string())
                .with_output_schema(output_schema_json);
        },
    };
    let mut output_schema_json = schema_json(&spec.output_schema);

    let table_ref = datafusion_table_reference(&spec.table_name);
    let provider = match resolve_table_provider(state, table_ref).await {
        Ok(provider) => provider,
        Err(message) => {
            return ValidationRow::invalid(canonical_function_name, ERROR_TABLE_NOT_FOUND, message)
                .with_output_schema(output_schema_json);
        },
    };
    let schema = provider.schema();
    let src = schema.field_with_name(&spec.src_col).ok();
    let dst = schema.field_with_name(&spec.dst_col).ok();
    let weight = spec.weight_col.as_ref().and_then(|name| schema.field_with_name(name).ok());
    let edge_id = spec.edge_id_col.as_ref().and_then(|name| schema.field_with_name(name).ok());
    let src_dtype = src.map(|field| data_type_label(field.data_type()));
    let dst_dtype = dst.map(|field| data_type_label(field.data_type()));
    let weight_dtype = weight.map(|field| data_type_label(field.data_type()));

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
            .with_output_schema(output_schema_json)
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
                    ..ValidationRow::invalid(canonical_function_name, vertex_domain_error_code(&error), message)
                        .with_output_schema(output_schema_json)
                };
            },
        };
    spec.output_schema = result_schema_for_vertex_domain(algorithm, &spec.params, Some(&vertex_domain));
    output_schema_json = schema_json(&spec.output_schema);
    if let Err(error) = validate_vertex_id_domain_support_for_call(algorithm, &vertex_domain) {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(canonical_function_name, ERROR_UNSUPPORTED_VERTEX_DTYPE, error.to_string())
                .with_output_schema(output_schema_json)
        };
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
                format!(
                    "cuGraph {canonical_function_name} logical string vertex-id mapping does not support edge_id_col"
                ),
            )
            .with_output_schema(output_schema_json)
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
            .with_output_schema(output_schema_json)
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
            .with_output_schema(output_schema_json)
        };
    }

    // Side-input validation runs after the edge columns are resolved, so a
    // failure here must still report the resolved edge dtypes -- leaving them
    // NULL would contradict `src_col_resolved`/`dst_col_resolved` being true.
    if let Some(mut row) =
        validate_predicate_sources(state, canonical_function_name, &spec, &vertex_domain, output_schema_json.clone())
            .await
    {
        row.src_dtype = src_dtype.clone();
        row.dst_dtype = dst_dtype.clone();
        row.weight_dtype = weight_dtype.clone();
        return row;
    }

    ValidationRow {
        valid: true,
        error_code: None,
        message: "cuGraph call is statically valid".to_string(),
        function_name: canonical_function_name.to_string(),
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

async fn validate_predicate_sources(
    state: &dyn Session,
    function_name: &str,
    spec: &CugraphAlgorithmSpec,
    vertex_domain: &CugraphVertexIdDomain,
    output_schema_json: String,
) -> Option<ValidationRow> {
    if let CugraphAlgorithmParams::Similarity(params) = &spec.params {
        let provider =
            match resolve_table_provider(state, datafusion_table_reference(&params.vertex_pairs.table_name)).await {
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
        if let Err(error) =
            validate_similarity_pair_table_schema(&params.vertex_pairs, &provider.schema(), vertex_domain)
        {
            let code = similarity_pair_error_code(&error);
            return Some(predicate_validation_error(function_name, code, error.to_string(), &output_schema_json));
        }
        return None;
    }
    for source in spec.predicate_source_specs() {
        let provider = match resolve_table_provider(state, datafusion_table_reference(&source.table_name)).await {
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
        for column in source.columns {
            let field = match schema.field_with_name(&column.name) {
                Ok(field) => field,
                Err(_) => {
                    return Some(predicate_validation_error(
                        function_name,
                        ERROR_COLUMN_NOT_FOUND,
                        format!(
                            "cuGraph side-input source '{}' is missing column '{}'",
                            source.table_name, column.name
                        ),
                        &output_schema_json,
                    ));
                },
            };
            let mismatch = match column.role {
                PredicateSourceColumnRole::Vertex if !vertex_domain.accepts(field.data_type()) => Some((
                    ERROR_VERTEX_DTYPE_MISMATCH,
                    format!(
                        "cuGraph side-input vertex column '{}' in '{}' must have dtype {}",
                        column.name,
                        source.table_name,
                        vertex_domain.label()
                    ),
                )),
                PredicateSourceColumnRole::EdgeId
                    if vertex_domain.is_string_logical() || !vertex_domain.accepts(field.data_type()) =>
                {
                    Some((
                        ERROR_UNSUPPORTED_OPTION,
                        format!(
                            "cuGraph side-input edge-id column '{}' in '{}' requires the numeric edge-id domain",
                            column.name, source.table_name
                        ),
                    ))
                },
                PredicateSourceColumnRole::FloatValue
                    if !matches!(field.data_type(), DataType::Float32 | DataType::Float64) =>
                {
                    Some((
                        ERROR_UNSUPPORTED_WEIGHT_DTYPE,
                        format!(
                            "cuGraph side-input value column '{}' in '{}' must have dtype Float32 or Float64",
                            column.name, source.table_name
                        ),
                    ))
                },
                _ => None,
            };
            if let Some((code, message)) = mismatch {
                return Some(predicate_validation_error(function_name, code, message, &output_schema_json));
            }
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
        ..ValidationRow::invalid(function_name, code, message).with_output_schema(output_schema_json.to_string())
    }
}

#[derive(Debug, Clone)]
struct ValidateCallOptions {
    source_vertex: Option<CugraphBfsSource>,
    src_col: String,
    dst_col: String,
    weight_col: Option<String>,
    options: Map<String, Value>,
}

/// Parse the family-owned, flat `options` object from the common envelope.
///
/// The SQL table-function parser still owns semantic option validation. This
/// parser only removes the common bindings it needs to form a graph call and
/// passes every remaining option through unchanged.
fn parse_validate_options(
    algorithm: CugraphAlgorithm,
    options: &Value,
) -> std::result::Result<ValidateCallOptions, (&'static str, String)> {
    let object =
        options.as_object().ok_or_else(|| (ERROR_INVALID_ARGUMENT, "cuGraph options must be an object".to_string()))?;
    if object.contains_key("options") {
        return Err((
            ERROR_INVALID_ARGUMENT,
            "cuGraph options must be flat; move nested 'options' fields into the common options object".to_string(),
        ));
    }

    let mut options = object.clone();
    let source_vertex = match options.remove("source_vertex") {
        Some(Value::Null) if algorithm == CugraphAlgorithm::Bfs => None,
        Some(value) => Some(json_source_vertex(&value, "source_vertex", algorithm)?),
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
    let src_col = match options.remove("src_col") {
        Some(value) => json_string(&value, "src_col")?,
        None => "src".to_string(),
    };
    let dst_col = match options.remove("dst_col") {
        Some(value) => json_string(&value, "dst_col")?,
        None => "dst".to_string(),
    };
    let weight_col = match options.remove("weight_col") {
        Some(Value::Null) | None => None,
        Some(value) => Some(json_string(&value, "weight_col")?),
    };
    Ok(ValidateCallOptions { source_vertex, src_col, dst_col, weight_col, options })
}

fn bfs_options_include_source_relation(options: &Map<String, Value>) -> bool {
    options.contains_key("source_vertices_table") || options.contains_key("source_vertex_col")
}

fn build_validate_spec(
    algorithm: CugraphAlgorithm,
    table_name: &str,
    call: &ValidateCallOptions,
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

fn validation_error_code(error: &datafusion::common::DataFusionError) -> &'static str {
    match structured_error_code(error) {
        Some(ErrorCode::UnsupportedOperator) => ERROR_UNSUPPORTED_OPTION,
        Some(ErrorCode::InvalidExecutionPlan) | None => ERROR_INVALID_ARGUMENT,
        _ => ERROR_INVALID_ARGUMENT,
    }
}

fn vertex_domain_error_code(error: &datafusion::common::DataFusionError) -> &'static str {
    match structured_error_code(error) {
        Some(ErrorCode::UnsupportedType) => ERROR_UNSUPPORTED_VERTEX_DTYPE,
        Some(ErrorCode::InvalidExecutionPlan) | None => ERROR_VERTEX_DTYPE_MISMATCH,
        _ => ERROR_VERTEX_DTYPE_MISMATCH,
    }
}

fn similarity_pair_error_code(error: &datafusion::common::DataFusionError) -> &'static str {
    match structured_error_code(error) {
        Some(ErrorCode::FieldNotFound) => ERROR_COLUMN_NOT_FOUND,
        Some(ErrorCode::InvalidExecutionPlan) | None => ERROR_VERTEX_DTYPE_MISMATCH,
        _ => ERROR_VERTEX_DTYPE_MISMATCH,
    }
}

fn structured_error_code(error: &datafusion::common::DataFusionError) -> Option<ErrorCode> {
    error.find_nexus_error().map(|error| error.code())
}

fn json_string(value: &Value, key: &str) -> std::result::Result<String, (&'static str, String)> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| (ERROR_INVALID_ARGUMENT, format!("cuGraph option '{key}' must be a string")))
}

fn json_i64(value: &Value, key: &str) -> std::result::Result<i64, (&'static str, String)> {
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    if let Some(value) = value.as_u64() {
        return i64::try_from(value)
            .map_err(|_| (ERROR_INVALID_ARGUMENT, format!("cuGraph option '{key}' exceeds i64::MAX")));
    }
    Err((ERROR_INVALID_ARGUMENT, format!("cuGraph option '{key}' must be an integer")))
}

fn json_source_vertex(
    value: &Value,
    key: &str,
    algorithm: CugraphAlgorithm,
) -> std::result::Result<CugraphBfsSource, (&'static str, String)> {
    if let Some(values) = value.as_array() {
        if algorithm != CugraphAlgorithm::Bfs {
            return Err((ERROR_INVALID_ARGUMENT, format!("cuGraph option '{key}' must be an integer")));
        }
        if values.is_empty() {
            return Err((ERROR_INVALID_ARGUMENT, format!("cuGraph option '{key}' list must not be empty")));
        }
        let values = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if value.is_null() {
                    return Err((
                        ERROR_INVALID_ARGUMENT,
                        format!("cuGraph option '{key}' list must not contain null element at index {index}"),
                    ));
                }
                json_source_vertex_literal(value, key, algorithm)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let first_is_utf8 = values.first().is_some_and(CugraphBfsSourceVertex::is_utf8);
        if values.iter().any(|value| value.is_utf8() != first_is_utf8) {
            return Err((
                ERROR_INVALID_ARGUMENT,
                format!("cuGraph option '{key}' list must not mix integer and string elements"),
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
        return if matches!(algorithm, CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp) {
            Ok(CugraphBfsSourceVertex::Utf8(value.to_string()))
        } else {
            Err((ERROR_INVALID_ARGUMENT, format!("cuGraph option '{key}' must be an integer")))
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

#[cfg(test)]
mod tests {
    use datafusion::common::DataFusionError;

    use super::{
        ERROR_UNSUPPORTED_OPTION, ERROR_UNSUPPORTED_VERTEX_DTYPE, ERROR_VERTEX_DTYPE_MISMATCH, validation_error_code,
        vertex_domain_error_code,
    };
    use crate::error::{Error, ErrorCode};

    #[test]
    fn validation_error_codes_are_derived_from_structured_errors_not_rendered_messages() {
        let unsupported_option: DataFusionError = Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_nexus.test.cugraph_validation",
            "vertex dtype mismatch",
        )
        .into();
        assert_eq!(validation_error_code(&unsupported_option), ERROR_UNSUPPORTED_OPTION);

        let unsupported_vertex_dtype: DataFusionError = Error::new(
            ErrorCode::UnsupportedType,
            "datafusion_nexus.test.cugraph_validation",
            "source and destination vertex columns differ",
        )
        .into();
        assert_eq!(vertex_domain_error_code(&unsupported_vertex_dtype), ERROR_UNSUPPORTED_VERTEX_DTYPE);

        let vertex_dtype_mismatch: DataFusionError = Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.test.cugraph_validation",
            "unsupported vertex dtype",
        )
        .into();
        assert_eq!(vertex_domain_error_code(&vertex_dtype_mismatch), ERROR_VERTEX_DTYPE_MISMATCH);
    }
}
