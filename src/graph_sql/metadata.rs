use std::{any::Any, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::{Result, ScalarValue, TableReference},
    datasource::{MemTable, TableType},
    logical_expr::{Expr, lit},
    physical_plan::ExecutionPlan,
};
use serde_json::{Map, Value, json};

use super::{
    CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_LIST_ALGORITHMS_FUNCTION, CUGRAPH_OPERATION_REGISTRY,
    CUGRAPH_VALIDATE_CALL_FUNCTION, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec,
    CugraphGraphOptions, CugraphSqlConfig, CugraphTableReference, parse_cugraph_algorithm_spec, plan_error,
    stable_result_schema,
};

const SEVERITY_OK: &str = "ok";
const SEVERITY_ERROR: &str = "error";
const ERROR_OK: &str = "ok";
const ERROR_UNKNOWN_ALGORITHM: &str = "unknown_algorithm";
const ERROR_ALGORITHM_DISABLED: &str = "algorithm_disabled";
const ERROR_INVALID_CALL_JSON: &str = "invalid_call_json";
const ERROR_INVALID_ARGUMENT: &str = "invalid_argument";
const ERROR_TABLE_NOT_FOUND: &str = "table_not_found";
const ERROR_COLUMN_NOT_FOUND: &str = "column_not_found";
const ERROR_NULLABLE_EDGE_COLUMN: &str = "nullable_edge_column";
const ERROR_UNSUPPORTED_VERTEX_DTYPE: &str = "unsupported_vertex_dtype";
const ERROR_VERTEX_DTYPE_MISMATCH: &str = "vertex_dtype_mismatch";
const ERROR_UNSUPPORTED_WEIGHT_DTYPE: &str = "unsupported_weight_dtype";
const ERROR_UNSUPPORTED_OPTION: &str = "unsupported_option";

pub(crate) fn cugraph_metadata_table_functions(config: CugraphSqlConfig) -> Vec<Arc<TableFunction>> {
    vec![
        Arc::new(TableFunction::new(
            CUGRAPH_LIST_ALGORITHMS_FUNCTION.to_string(),
            Arc::new(CugraphListAlgorithmsFunction { config: config.clone() }),
        )),
        Arc::new(TableFunction::new(
            CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION.to_string(),
            Arc::new(CugraphDescribeAlgorithmFunction { config: config.clone() }),
        )),
        Arc::new(TableFunction::new(
            CUGRAPH_VALIDATE_CALL_FUNCTION.to_string(),
            Arc::new(CugraphValidateCallFunction { config }),
        )),
    ]
}

#[derive(Debug)]
struct CugraphListAlgorithmsFunction {
    config: CugraphSqlConfig,
}

impl TableFunctionImpl for CugraphListAlgorithmsFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if !args.is_empty() {
            return Err(plan_error("cugraph_list_algorithms does not accept arguments"));
        }
        Ok(Arc::new(CugraphMetadataProvider::list(self.config.clone())))
    }
}

#[derive(Debug)]
struct CugraphDescribeAlgorithmFunction {
    config: CugraphSqlConfig,
}

impl TableFunctionImpl for CugraphDescribeAlgorithmFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        let function_name = literal_string_arg(args, 0, "function_name")?;
        if args.len() != 1 {
            return Err(plan_error("cugraph_describe_algorithm expects exactly one function_name argument"));
        }
        Ok(Arc::new(CugraphMetadataProvider::describe(self.config.clone(), function_name)))
    }
}

#[derive(Debug)]
struct CugraphValidateCallFunction {
    config: CugraphSqlConfig,
}

impl TableFunctionImpl for CugraphValidateCallFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        let function_name = literal_string_arg(args, 0, "function_name")?;
        let table_name = literal_string_arg(args, 1, "table_name")?;
        let call_json = literal_string_arg(args, 2, "call_json")?;
        if args.len() != 3 {
            return Err(plan_error("cugraph_validate_call expects function_name, table_name, call_json"));
        }
        Ok(Arc::new(CugraphMetadataProvider::validate(self.config.clone(), function_name, table_name, call_json)))
    }
}

#[derive(Debug, Clone)]
enum CugraphMetadataKind {
    List,
    Describe { function_name: String },
    Validate { function_name: String, table_name: String, call_json: String },
}

#[derive(Debug, Clone)]
struct CugraphMetadataProvider {
    config: CugraphSqlConfig,
    kind: CugraphMetadataKind,
}

impl CugraphMetadataProvider {
    fn list(config: CugraphSqlConfig) -> Self {
        Self { config, kind: CugraphMetadataKind::List }
    }

    fn describe(config: CugraphSqlConfig, function_name: String) -> Self {
        Self { config, kind: CugraphMetadataKind::Describe { function_name } }
    }

    fn validate(config: CugraphSqlConfig, function_name: String, table_name: String, call_json: String) -> Self {
        Self { config, kind: CugraphMetadataKind::Validate { function_name, table_name, call_json } }
    }
}

#[async_trait::async_trait]
impl TableProvider for CugraphMetadataProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        match self.kind {
            CugraphMetadataKind::List => list_schema(),
            CugraphMetadataKind::Describe { .. } => describe_schema(),
            CugraphMetadataKind::Validate { .. } => validate_schema(),
        }
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let batch = match &self.kind {
            CugraphMetadataKind::List => list_batch(&self.config)?,
            CugraphMetadataKind::Describe { function_name } => describe_batch(function_name, &self.config)?,
            CugraphMetadataKind::Validate { function_name, table_name, call_json } => {
                validate_batch(state, function_name, table_name, call_json, &self.config).await?
            },
        };
        let table = MemTable::try_new(self.schema(), vec![vec![batch]])?;
        table.scan(state, projection, filters, limit).await
    }
}

fn list_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("function_name", DataType::Utf8, false),
        Field::new("algorithm", DataType::Utf8, false),
        Field::new("enabled", DataType::Boolean, false),
        Field::new("experimental", DataType::Boolean, false),
        Field::new("requires_source_vertex", DataType::Boolean, false),
        Field::new("supports_weight", DataType::Boolean, false),
        Field::new("min_args", DataType::Int64, false),
        Field::new("max_args", DataType::Int64, false),
        Field::new("output_schema_json", DataType::Utf8, false),
        Field::new("options_schema_json", DataType::Utf8, false),
        Field::new("default_graph_options_json", DataType::Utf8, false),
    ]))
}

fn describe_schema() -> SchemaRef {
    let mut fields = vec![
        Field::new("valid", DataType::Boolean, false),
        Field::new("error_code", DataType::Utf8, false),
        Field::new("message", DataType::Utf8, false),
    ];
    fields.extend(list_schema().fields().iter().map(|field| field.as_ref().clone()));
    fields.extend([
        Field::new("argument_descriptions_json", DataType::Utf8, false),
        Field::new("option_defaults_json", DataType::Utf8, false),
        Field::new("result_field_descriptions_json", DataType::Utf8, false),
        Field::new("limitations_json", DataType::Utf8, false),
    ]);
    Arc::new(Schema::new(fields))
}

fn validate_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("valid", DataType::Boolean, false),
        Field::new("severity", DataType::Utf8, false),
        Field::new("error_code", DataType::Utf8, false),
        Field::new("message", DataType::Utf8, false),
        Field::new("function_name", DataType::Utf8, false),
        Field::new("algorithm_enabled", DataType::Boolean, false),
        Field::new("table_resolved", DataType::Boolean, false),
        Field::new("src_col_resolved", DataType::Boolean, false),
        Field::new("dst_col_resolved", DataType::Boolean, false),
        Field::new("weight_col_resolved", DataType::Boolean, false),
        Field::new("src_dtype", DataType::Utf8, false),
        Field::new("dst_dtype", DataType::Utf8, false),
        Field::new("weight_dtype", DataType::Utf8, false),
        Field::new("output_schema_json", DataType::Utf8, false),
        Field::new("normalized_options_json", DataType::Utf8, false),
        Field::new("would_execute_gpu", DataType::Boolean, false),
    ]))
}

fn list_batch(config: &CugraphSqlConfig) -> Result<RecordBatch> {
    let rows: Vec<_> =
        CUGRAPH_OPERATION_REGISTRY.iter().map(|definition| algorithm_row(definition.algorithm, config)).collect();
    RecordBatch::try_new(
        list_schema(),
        vec![
            strings(rows.iter().map(|row| row.function_name.clone()).collect()),
            strings(rows.iter().map(|row| row.algorithm.clone()).collect()),
            bools(rows.iter().map(|row| row.enabled).collect()),
            bools(rows.iter().map(|row| row.experimental).collect()),
            bools(rows.iter().map(|row| row.requires_source_vertex).collect()),
            bools(rows.iter().map(|row| row.supports_weight).collect()),
            int64s(rows.iter().map(|row| row.min_args).collect()),
            int64s(rows.iter().map(|row| row.max_args).collect()),
            strings(rows.iter().map(|row| row.output_schema_json.clone()).collect()),
            strings(rows.iter().map(|row| row.options_schema_json.clone()).collect()),
            strings(rows.iter().map(|row| row.default_graph_options_json.clone()).collect()),
        ],
    )
    .map_err(Into::into)
}

fn describe_batch(function_name: &str, config: &CugraphSqlConfig) -> Result<RecordBatch> {
    let Some(algorithm) = algorithm_for_function_name(function_name) else {
        return RecordBatch::try_new(
            describe_schema(),
            vec![
                bools(vec![false]),
                strings(vec![ERROR_UNKNOWN_ALGORITHM.to_string()]),
                strings(vec![format!("unknown cuGraph algorithm function '{function_name}'")]),
                strings(vec![function_name.to_string()]),
                strings(vec!["".to_string()]),
                bools(vec![false]),
                bools(vec![false]),
                bools(vec![false]),
                bools(vec![false]),
                int64s(vec![0]),
                int64s(vec![0]),
                strings(vec!["{}".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["[]".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["[]".to_string()]),
            ],
        )
        .map_err(Into::into);
    };

    let row = algorithm_row(algorithm, config);
    RecordBatch::try_new(
        describe_schema(),
        vec![
            bools(vec![true]),
            strings(vec![ERROR_OK.to_string()]),
            strings(vec![format!("{} is a registered cuGraph SQL table function", row.function_name)]),
            strings(vec![row.function_name]),
            strings(vec![row.algorithm]),
            bools(vec![row.enabled]),
            bools(vec![row.experimental]),
            bools(vec![row.requires_source_vertex]),
            bools(vec![row.supports_weight]),
            int64s(vec![row.min_args]),
            int64s(vec![row.max_args]),
            strings(vec![row.output_schema_json]),
            strings(vec![row.options_schema_json]),
            strings(vec![row.default_graph_options_json]),
            strings(vec![argument_descriptions_json(algorithm)]),
            strings(vec![option_defaults_json(algorithm, config)]),
            strings(vec![result_field_descriptions_json(algorithm)]),
            strings(vec![limitations_json(algorithm)]),
        ],
    )
    .map_err(Into::into)
}

async fn validate_batch(
    state: &dyn Session,
    function_name: &str,
    table_name: &str,
    call_json: &str,
    config: &CugraphSqlConfig,
) -> Result<RecordBatch> {
    let row = validate_call(state, function_name, table_name, call_json, config).await;
    RecordBatch::try_new(
        validate_schema(),
        vec![
            bools(vec![row.valid]),
            strings(vec![row.severity]),
            strings(vec![row.error_code]),
            strings(vec![row.message]),
            strings(vec![row.function_name]),
            bools(vec![row.algorithm_enabled]),
            bools(vec![row.table_resolved]),
            bools(vec![row.src_col_resolved]),
            bools(vec![row.dst_col_resolved]),
            bools(vec![row.weight_col_resolved]),
            strings(vec![row.src_dtype]),
            strings(vec![row.dst_dtype]),
            strings(vec![row.weight_dtype]),
            strings(vec![row.output_schema_json]),
            strings(vec![row.normalized_options_json]),
            bools(vec![row.would_execute_gpu]),
        ],
    )
    .map_err(Into::into)
}

#[derive(Debug, Clone)]
struct AlgorithmMetadataRow {
    function_name: String,
    algorithm: String,
    enabled: bool,
    experimental: bool,
    requires_source_vertex: bool,
    supports_weight: bool,
    min_args: i64,
    max_args: i64,
    output_schema_json: String,
    options_schema_json: String,
    default_graph_options_json: String,
}

fn algorithm_row(algorithm: CugraphAlgorithm, config: &CugraphSqlConfig) -> AlgorithmMetadataRow {
    AlgorithmMetadataRow {
        function_name: algorithm.function_name().to_string(),
        algorithm: algorithm.native_algorithm().definition().name.to_string(),
        enabled: config.allows(algorithm),
        experimental: false,
        requires_source_vertex: requires_source_vertex(algorithm),
        supports_weight: true,
        min_args: if requires_source_vertex(algorithm) { 2 } else { 1 },
        max_args: if requires_source_vertex(algorithm) { 6 } else { 5 },
        output_schema_json: schema_json(&stable_result_schema(algorithm)),
        options_schema_json: options_schema_json(algorithm),
        default_graph_options_json: graph_options_json(&effective_default_graph_options(
            algorithm,
            &config.default_graph_options,
        )),
    }
}

#[derive(Debug, Clone)]
struct ValidationRow {
    valid: bool,
    severity: String,
    error_code: String,
    message: String,
    function_name: String,
    algorithm_enabled: bool,
    table_resolved: bool,
    src_col_resolved: bool,
    dst_col_resolved: bool,
    weight_col_resolved: bool,
    src_dtype: String,
    dst_dtype: String,
    weight_dtype: String,
    output_schema_json: String,
    normalized_options_json: String,
    would_execute_gpu: bool,
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

async fn validate_call(
    state: &dyn Session,
    function_name: &str,
    table_name: &str,
    call_json: &str,
    config: &CugraphSqlConfig,
) -> ValidationRow {
    let Some(algorithm) = algorithm_for_function_name(function_name) else {
        return ValidationRow::invalid(
            function_name,
            ERROR_UNKNOWN_ALGORITHM,
            format!("unknown cuGraph algorithm function '{function_name}'"),
        );
    };
    let output_schema_json = schema_json(&stable_result_schema(algorithm));
    if !config.allows(algorithm) {
        return ValidationRow::invalid(
            function_name,
            ERROR_ALGORITHM_DISABLED,
            format!("cuGraph algorithm {algorithm:?} is not enabled in this session"),
        )
        .with_algorithm(false, output_schema_json);
    }

    let call = match parse_validate_call_json(algorithm, call_json) {
        Ok(call) => call,
        Err((code, message)) => {
            return ValidationRow::invalid(function_name, code, message).with_algorithm(true, output_schema_json);
        },
    };

    let spec = match build_validate_spec(algorithm, table_name, &call, config) {
        Ok(spec) => spec,
        Err(error) => {
            return ValidationRow::invalid(function_name, parse_error_code(&error.to_string()), error.to_string())
                .with_algorithm(true, output_schema_json);
        },
    };

    let table_ref = datafusion_table_reference(&spec.table_name);
    let provider = match resolve_table_provider(state, table_ref).await {
        Ok(provider) => provider,
        Err(message) => {
            return ValidationRow::invalid(function_name, ERROR_TABLE_NOT_FOUND, message)
                .with_algorithm(true, output_schema_json);
        },
    };
    let schema = provider.schema();
    let src = schema.field_with_name(&spec.src_col).ok();
    let dst = schema.field_with_name(&spec.dst_col).ok();
    let weight = spec.weight_col.as_ref().and_then(|name| schema.field_with_name(name).ok());
    let src_dtype = src.map(|field| data_type_label(field.data_type())).unwrap_or_default();
    let dst_dtype = dst.map(|field| data_type_label(field.data_type())).unwrap_or_default();
    let weight_dtype = weight.map(|field| data_type_label(field.data_type())).unwrap_or_default();

    if src.is_none() || dst.is_none() || spec.weight_col.as_ref().is_some_and(|_| weight.is_none()) {
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
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: src.is_some(),
            dst_col_resolved: dst.is_some(),
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                function_name,
                ERROR_COLUMN_NOT_FOUND,
                format!("cuGraph edge source is missing column(s): {}", missing.join(",")),
            )
            .with_algorithm(true, output_schema_json)
        };
    }

    let src_field = src.expect("checked above");
    let dst_field = dst.expect("checked above");
    if src_field.is_nullable() {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                function_name,
                ERROR_NULLABLE_EDGE_COLUMN,
                format!("cuGraph source column '{}' must be non-null", spec.src_col),
            )
            .with_algorithm(true, output_schema_json)
        };
    }
    if dst_field.is_nullable() {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                function_name,
                ERROR_NULLABLE_EDGE_COLUMN,
                format!("cuGraph destination column '{}' must be non-null", spec.dst_col),
            )
            .with_algorithm(true, output_schema_json)
        };
    }
    if let Some(weight_field) = weight
        && weight_field.is_nullable()
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
                function_name,
                ERROR_NULLABLE_EDGE_COLUMN,
                format!("cuGraph weight column '{}' must be non-null", spec.weight_col.as_deref().unwrap_or("")),
            )
            .with_algorithm(true, output_schema_json)
        };
    }
    if !is_supported_vertex_dtype(src_field.data_type()) {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                function_name,
                ERROR_UNSUPPORTED_VERTEX_DTYPE,
                format!("cuGraph source column '{}' has unsupported dtype", spec.src_col),
            )
            .with_algorithm(true, output_schema_json)
        };
    }
    if !is_supported_vertex_dtype(dst_field.data_type()) {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                function_name,
                ERROR_UNSUPPORTED_VERTEX_DTYPE,
                format!("cuGraph destination column '{}' has unsupported dtype", spec.dst_col),
            )
            .with_algorithm(true, output_schema_json)
        };
    }
    if src_field.data_type() != dst_field.data_type() {
        return ValidationRow {
            table_resolved: true,
            src_col_resolved: true,
            dst_col_resolved: true,
            weight_col_resolved: spec.weight_col.is_none() || weight.is_some(),
            src_dtype,
            dst_dtype,
            weight_dtype,
            ..ValidationRow::invalid(
                function_name,
                ERROR_VERTEX_DTYPE_MISMATCH,
                "cuGraph source and destination columns must have the same vertex dtype",
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
                function_name,
                ERROR_UNSUPPORTED_WEIGHT_DTYPE,
                format!("cuGraph weight column '{}' has unsupported dtype", spec.weight_col.as_deref().unwrap_or("")),
            )
            .with_algorithm(true, output_schema_json)
        };
    }

    ValidationRow {
        valid: true,
        severity: SEVERITY_OK.to_string(),
        error_code: ERROR_OK.to_string(),
        message: "cuGraph call is statically valid".to_string(),
        function_name: function_name.to_string(),
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

#[derive(Debug, Clone)]
struct ValidateCallJson {
    source_vertex: Option<i64>,
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

    let source_vertex = match object.get("source_vertex") {
        Some(value) => Some(json_i64(value, "source_vertex")?),
        None => None,
    };
    if requires_source_vertex(algorithm) && source_vertex.is_none() {
        return Err((ERROR_INVALID_ARGUMENT, format!("{} requires source_vertex", algorithm.function_name())));
    }
    if !requires_source_vertex(algorithm) && source_vertex.is_some() {
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
    let options = match object.get("options") {
        Some(value) => value
            .as_object()
            .cloned()
            .ok_or_else(|| (ERROR_INVALID_CALL_JSON, "cuGraph call JSON options must be an object".to_string()))?,
        None => Map::new(),
    };
    Ok(ValidateCallJson { source_vertex, src_col, dst_col, weight_col, options })
}

fn build_validate_spec(
    algorithm: CugraphAlgorithm,
    table_name: &str,
    call: &ValidateCallJson,
    config: &CugraphSqlConfig,
) -> Result<CugraphAlgorithmSpec> {
    let mut args = vec![lit(table_name)];
    if requires_source_vertex(algorithm) {
        args.push(lit(call.source_vertex.expect("source_vertex checked by parse_validate_call_json")));
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

fn datafusion_table_reference(table: &CugraphTableReference) -> TableReference {
    match table.parts() {
        [table] => TableReference::bare(table.as_str()),
        [schema, table] => TableReference::partial(schema.as_str(), table.as_str()),
        [catalog, schema, table] => TableReference::full(catalog.as_str(), schema.as_str(), table.as_str()),
        _ => unreachable!("CugraphTableReference::parse validates one to three identifier parts"),
    }
}

fn algorithm_for_function_name(function_name: &str) -> Option<CugraphAlgorithm> {
    CUGRAPH_OPERATION_REGISTRY
        .iter()
        .find(|definition| definition.sql_function_name == function_name)
        .map(|definition| definition.algorithm)
}

fn requires_source_vertex(algorithm: CugraphAlgorithm) -> bool {
    matches!(algorithm, CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp)
}

fn effective_default_graph_options(algorithm: CugraphAlgorithm, defaults: &CugraphGraphOptions) -> CugraphGraphOptions {
    let mut options = defaults.clone();
    if matches!(
        algorithm,
        CugraphAlgorithm::WeaklyConnectedComponents
            | CugraphAlgorithm::TriangleCountAll
            | CugraphAlgorithm::CoreNumber
            | CugraphAlgorithm::KCore
    ) {
        options.directed = false;
    }
    options
}

fn schema_json(schema: &SchemaRef) -> String {
    json!({
        "fields": schema
            .fields()
            .iter()
            .map(|field| {
                json!({
                    "name": field.name(),
                    "data_type": data_type_label(field.data_type()),
                    "nullable": field.is_nullable(),
                })
            })
            .collect::<Vec<_>>()
    })
    .to_string()
}

fn options_schema_json(algorithm: CugraphAlgorithm) -> String {
    json!({
        "graph_options": {
            "directed": {"type": "boolean"},
            "renumber": {"type": "boolean"},
            "construction_policy": {"type": "string", "enum": ["python_cugraph", "raw_libcugraph"]}
        },
        "algorithm_options": algorithm_option_names(algorithm)
    })
    .to_string()
}

fn option_defaults_json(algorithm: CugraphAlgorithm, config: &CugraphSqlConfig) -> String {
    let graph_options = effective_default_graph_options(algorithm, &config.default_graph_options);
    json!({
        "graph_options": graph_options_value(&graph_options),
        "algorithm_options": algorithm_option_defaults(algorithm)
    })
    .to_string()
}

fn graph_options_json(options: &CugraphGraphOptions) -> String {
    graph_options_value(options).to_string()
}

fn graph_options_value(options: &CugraphGraphOptions) -> Value {
    json!({
        "directed": options.directed,
        "renumber": options.renumber,
        "construction_policy": options.construction_policy.as_str(),
    })
}

fn algorithm_option_names(algorithm: CugraphAlgorithm) -> Vec<&'static str> {
    match algorithm {
        CugraphAlgorithm::PageRank | CugraphAlgorithm::PageRankStrict => vec!["alpha", "epsilon", "max_iterations"],
        CugraphAlgorithm::Bfs => vec!["depth_limit"],
        CugraphAlgorithm::Louvain => vec!["resolution"],
        CugraphAlgorithm::EigenvectorCentrality => vec!["epsilon", "max_iterations"],
        CugraphAlgorithm::KatzCentrality => vec!["alpha", "beta", "epsilon", "max_iterations"],
        CugraphAlgorithm::Hits => vec!["epsilon", "max_iterations", "normalize"],
        CugraphAlgorithm::Sssp => vec!["cutoff"],
        CugraphAlgorithm::CoreNumber => vec!["degree_type"],
        CugraphAlgorithm::KCore => vec!["k", "degree_type"],
        CugraphAlgorithm::Leiden => vec!["max_level", "resolution", "theta", "seed"],
        CugraphAlgorithm::Ecg => vec!["min_weight", "ensemble_size", "max_level", "threshold", "resolution", "seed"],
        CugraphAlgorithm::SpectralModularityMaximization | CugraphAlgorithm::BalancedCutClustering => {
            vec![
                "n_clusters",
                "n_eigenvectors",
                "evs_tolerance",
                "evs_max_iterations",
                "k_means_tolerance",
                "k_means_max_iterations",
                "seed",
            ]
        },
        CugraphAlgorithm::DegreesAll
        | CugraphAlgorithm::InDegreesAll
        | CugraphAlgorithm::OutDegreesAll
        | CugraphAlgorithm::WeaklyConnectedComponents
        | CugraphAlgorithm::StronglyConnectedComponents
        | CugraphAlgorithm::TriangleCountAll
        | CugraphAlgorithm::MinimumSpanningTree => Vec::new(),
    }
}

fn algorithm_option_defaults(algorithm: CugraphAlgorithm) -> Value {
    match algorithm {
        CugraphAlgorithm::PageRank | CugraphAlgorithm::PageRankStrict => {
            let params = super::CugraphPageRankParams::default();
            json!({"alpha": params.alpha, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithm::Bfs => json!({"depth_limit": null}),
        CugraphAlgorithm::Louvain => json!({"resolution": super::CugraphLouvainParams::default().resolution}),
        CugraphAlgorithm::EigenvectorCentrality => {
            let params = super::CugraphEigenvectorCentralityParams::default();
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithm::KatzCentrality => {
            let params = super::CugraphKatzCentralityParams::default();
            json!({"alpha": params.alpha, "beta": params.beta, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithm::Hits => {
            let params = super::CugraphHitsParams::default();
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations, "normalize": params.normalize})
        },
        CugraphAlgorithm::Sssp => json!({"cutoff": null}),
        CugraphAlgorithm::CoreNumber => {
            json!({"degree_type": super::CugraphCoreNumberParams::default().degree_type.as_str()})
        },
        CugraphAlgorithm::KCore => {
            let params = super::CugraphKCoreParams::default();
            json!({"k": params.k, "degree_type": params.degree_type.as_str()})
        },
        CugraphAlgorithm::Leiden => {
            let params = super::CugraphLeidenParams::default();
            json!({"max_level": params.max_level, "resolution": params.resolution, "theta": params.theta, "seed": params.seed})
        },
        CugraphAlgorithm::Ecg => {
            let params = super::CugraphEcgParams::default();
            json!({"min_weight": params.min_weight, "ensemble_size": params.ensemble_size, "max_level": params.max_level, "threshold": params.threshold, "resolution": params.resolution, "seed": params.seed})
        },
        CugraphAlgorithm::SpectralModularityMaximization | CugraphAlgorithm::BalancedCutClustering => {
            let params = super::CugraphSpectralClusteringParams::default();
            json!({"n_clusters": params.n_clusters, "n_eigenvectors": params.n_eigenvectors, "evs_tolerance": params.evs_tolerance, "evs_max_iterations": params.evs_max_iterations, "k_means_tolerance": params.k_means_tolerance, "k_means_max_iterations": params.k_means_max_iterations, "seed": params.seed})
        },
        CugraphAlgorithm::DegreesAll
        | CugraphAlgorithm::InDegreesAll
        | CugraphAlgorithm::OutDegreesAll
        | CugraphAlgorithm::WeaklyConnectedComponents
        | CugraphAlgorithm::StronglyConnectedComponents
        | CugraphAlgorithm::TriangleCountAll
        | CugraphAlgorithm::MinimumSpanningTree => json!({}),
    }
}

fn argument_descriptions_json(algorithm: CugraphAlgorithm) -> String {
    let mut args = vec![
        json!({"name": "table_name", "type": "Utf8", "required": true}),
        json!({"name": "src_col", "type": "Utf8", "required": false, "default": "src"}),
        json!({"name": "dst_col", "type": "Utf8", "required": false, "default": "dst"}),
        json!({"name": "weight_col", "type": "Utf8", "required": algorithm == CugraphAlgorithm::MinimumSpanningTree}),
        json!({"name": "options_json", "type": "Utf8", "required": false}),
    ];
    if requires_source_vertex(algorithm) {
        args.insert(1, json!({"name": "source_vertex", "type": "Int64", "required": true}));
    }
    Value::Array(args).to_string()
}

fn result_field_descriptions_json(algorithm: CugraphAlgorithm) -> String {
    let schema = stable_result_schema(algorithm);
    Value::Array(
        schema
            .fields()
            .iter()
            .map(|field| json!({"name": field.name(), "data_type": data_type_label(field.data_type())}))
            .collect(),
    )
    .to_string()
}

fn limitations_json(_algorithm: CugraphAlgorithm) -> String {
    json!([
        "dry-run validates table resolution, column presence, static dtypes, and options only",
        "dry-run does not scan edge data, construct a graph, or prove source-vertex existence"
    ])
    .to_string()
}

fn normalized_options_json(spec: &CugraphAlgorithmSpec) -> String {
    json!({
        "src_col": spec.src_col,
        "dst_col": spec.dst_col,
        "weight_col": spec.weight_col,
        "graph_options": graph_options_value(&spec.graph_options),
        "algorithm_options": params_json(&spec.params),
    })
    .to_string()
}

fn params_json(params: &CugraphAlgorithmParams) -> Value {
    match params {
        CugraphAlgorithmParams::PageRank(params) | CugraphAlgorithmParams::PageRankStrict(params) => {
            json!({"alpha": params.alpha, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::Bfs(params) => {
            json!({"source_vertex": params.source_vertex, "depth_limit": params.depth_limit})
        },
        CugraphAlgorithmParams::Louvain(params) => json!({"resolution": params.resolution}),
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::KatzCentrality(params) => {
            json!({"alpha": params.alpha, "beta": params.beta, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::Hits(params) => {
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations, "normalize": params.normalize})
        },
        CugraphAlgorithmParams::Sssp(params) => json!({"source_vertex": params.source_vertex, "cutoff": params.cutoff}),
        CugraphAlgorithmParams::CoreNumber(params) => json!({"degree_type": params.degree_type.as_str()}),
        CugraphAlgorithmParams::KCore(params) => json!({"k": params.k, "degree_type": params.degree_type.as_str()}),
        CugraphAlgorithmParams::Leiden(params) => {
            json!({"max_level": params.max_level, "resolution": params.resolution, "theta": params.theta, "seed": params.seed})
        },
        CugraphAlgorithmParams::Ecg(params) => {
            json!({"min_weight": params.min_weight, "ensemble_size": params.ensemble_size, "max_level": params.max_level, "threshold": params.threshold, "resolution": params.resolution, "seed": params.seed})
        },
        CugraphAlgorithmParams::SpectralModularityMaximization(params)
        | CugraphAlgorithmParams::BalancedCutClustering(params) => {
            json!({"n_clusters": params.n_clusters, "n_eigenvectors": params.n_eigenvectors, "evs_tolerance": params.evs_tolerance, "evs_max_iterations": params.evs_max_iterations, "k_means_tolerance": params.k_means_tolerance, "k_means_max_iterations": params.k_means_max_iterations, "seed": params.seed})
        },
        CugraphAlgorithmParams::DegreesAll(_)
        | CugraphAlgorithmParams::InDegreesAll(_)
        | CugraphAlgorithmParams::OutDegreesAll(_)
        | CugraphAlgorithmParams::WeaklyConnectedComponents(_)
        | CugraphAlgorithmParams::StronglyConnectedComponents(_)
        | CugraphAlgorithmParams::TriangleCountAll(_)
        | CugraphAlgorithmParams::MinimumSpanningTree(_) => json!({}),
    }
}

fn parse_error_code(message: &str) -> &'static str {
    if message.contains("unknown option") {
        ERROR_UNSUPPORTED_OPTION
    } else if message.contains("is not enabled in this session") {
        ERROR_ALGORITHM_DISABLED
    } else {
        ERROR_INVALID_ARGUMENT
    }
}

fn literal_string_arg(args: &[Expr], index: usize, name: &str) -> Result<String> {
    match args.get(index) {
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)) => Ok(value.clone()),
            _ => Err(plan_error(format!("{name} must be a non-null string literal"))),
        },
        Some(_) => Err(plan_error(format!("{name} must be a literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
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

fn data_type_label(data_type: &DataType) -> String {
    match data_type {
        DataType::Int32 => "Int32".to_string(),
        DataType::Int64 => "Int64".to_string(),
        DataType::Float32 => "Float32".to_string(),
        DataType::Float64 => "Float64".to_string(),
        DataType::Utf8 => "Utf8".to_string(),
        DataType::LargeUtf8 => "LargeUtf8".to_string(),
        DataType::Boolean => "Boolean".to_string(),
        other => format!("{other:?}"),
    }
}

fn is_supported_vertex_dtype(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Int32 | DataType::Int64)
}

fn is_supported_weight_dtype(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Float32 | DataType::Float64)
}

fn strings(values: Vec<String>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

fn bools(values: Vec<bool>) -> ArrayRef {
    Arc::new(BooleanArray::from(values))
}

fn int64s(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}
