use std::sync::Arc;

use arrow::{
    array::{ArrayRef, BooleanArray, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::Result,
    datasource::{MemTable, TableType},
    logical_expr::Expr,
    physical_plan::ExecutionPlan,
};

use super::{
    super::{
        CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION,
        CUGRAPH_LIST_ALGORITHMS_FUNCTION, CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION, CUGRAPH_OPERATION_REGISTRY,
        CUGRAPH_VALIDATE_CALL_FUNCTION, CugraphAlgorithm, CugraphSqlConfig, plan_error, stable_result_schema,
    },
    ERROR_OK, ERROR_UNKNOWN_ALGORITHM, algorithm_for_metadata_name,
    descriptions::{
        argument_descriptions_json, examples_json, limitations_json, purpose, result_field_descriptions_json, summary,
    },
    json_schema::{graph_options_json, mode_schemas_json, option_defaults_json, options_schema_json, schema_json},
    validate::{cugraph_validate_call, literal_string_arg},
};

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

pub(super) fn list_schema() -> SchemaRef {
    let mut fields = compact_algorithm_fields();
    fields.push(Field::new("metadata_schema_version", DataType::Int64, false));
    Arc::new(Schema::new(fields))
}

fn compact_algorithm_fields() -> Vec<Field> {
    vec![
        Field::new("function_name", DataType::Utf8, false),
        Field::new("algorithm", DataType::Utf8, false),
        Field::new("enabled", DataType::Boolean, false),
        Field::new("experimental", DataType::Boolean, false),
        Field::new("requires_source_vertex", DataType::Boolean, false),
        Field::new("accepts_weight_arg", DataType::Boolean, false),
        Field::new("consumes_weight", DataType::Boolean, false),
        Field::new("weight_col_policy", DataType::Utf8, false),
        Field::new("output_schema_kind", DataType::Utf8, false),
        Field::new("has_mode_schemas", DataType::Boolean, false),
        Field::new("signature", DataType::Utf8, false),
        Field::new("allowed_arg_counts_json", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
    ]
}

pub(super) fn describe_schema() -> SchemaRef {
    let mut fields = vec![
        Field::new("valid", DataType::Boolean, false),
        Field::new("error_code", DataType::Utf8, false),
        Field::new("message", DataType::Utf8, false),
    ];
    fields.extend(compact_algorithm_fields());
    fields.extend([
        Field::new("output_schema_json", DataType::Utf8, false),
        Field::new("options_schema_json", DataType::Utf8, false),
        Field::new("default_graph_options_json", DataType::Utf8, false),
        Field::new("argument_descriptions_json", DataType::Utf8, false),
        Field::new("option_defaults_json", DataType::Utf8, false),
        Field::new("result_field_descriptions_json", DataType::Utf8, false),
        Field::new("limitations_json", DataType::Utf8, false),
        Field::new("purpose", DataType::Utf8, false),
        Field::new("mode_schemas_json", DataType::Utf8, false),
        Field::new("examples_json", DataType::Utf8, false),
        Field::new("metadata_schema_version", DataType::Int64, false),
    ]);
    Arc::new(Schema::new(fields))
}

pub(super) fn validate_schema() -> SchemaRef {
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
            bools(rows.iter().map(|row| row.accepts_weight_arg).collect()),
            bools(rows.iter().map(|row| row.consumes_weight).collect()),
            strings(rows.iter().map(|row| row.weight_col_policy.to_string()).collect()),
            strings(rows.iter().map(|row| row.output_schema_kind.to_string()).collect()),
            bools(rows.iter().map(|row| row.has_mode_schemas).collect()),
            strings(rows.iter().map(|row| row.signature.clone()).collect()),
            strings(rows.iter().map(|row| row.allowed_arg_counts_json.clone()).collect()),
            strings(rows.iter().map(|row| row.summary.clone()).collect()),
            int64s(vec![i64::from(CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION); rows.len()]),
        ],
    )
    .map_err(Into::into)
}

fn describe_batch(function_name: &str, config: &CugraphSqlConfig) -> Result<RecordBatch> {
    let Some(algorithm) = algorithm_for_metadata_name(function_name) else {
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
                bools(vec![false]),
                strings(vec!["".to_string()]),
                strings(vec!["".to_string()]),
                bools(vec![false]),
                strings(vec!["".to_string()]),
                strings(vec!["[]".to_string()]),
                strings(vec!["".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["[]".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["[]".to_string()]),
                strings(vec!["[]".to_string()]),
                strings(vec!["".to_string()]),
                strings(vec!["{}".to_string()]),
                strings(vec!["[]".to_string()]),
                int64s(vec![i64::from(CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION)]),
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
            bools(vec![row.accepts_weight_arg]),
            bools(vec![row.consumes_weight]),
            strings(vec![row.weight_col_policy.to_string()]),
            strings(vec![row.output_schema_kind.to_string()]),
            bools(vec![row.has_mode_schemas]),
            strings(vec![row.signature]),
            strings(vec![row.allowed_arg_counts_json]),
            strings(vec![row.summary]),
            strings(vec![row.output_schema_json]),
            strings(vec![row.options_schema_json]),
            strings(vec![row.default_graph_options_json]),
            strings(vec![argument_descriptions_json(algorithm)]),
            strings(vec![option_defaults_json(algorithm, config)]),
            strings(vec![result_field_descriptions_json(algorithm)]),
            strings(vec![limitations_json(algorithm)]),
            strings(vec![purpose(algorithm).to_string()]),
            strings(vec![mode_schemas_json(algorithm)]),
            strings(vec![examples_json(algorithm)]),
            int64s(vec![i64::from(CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION)]),
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
    let row = cugraph_validate_call(state, function_name, table_name, call_json, config).await;
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
    accepts_weight_arg: bool,
    consumes_weight: bool,
    signature: String,
    allowed_arg_counts_json: String,
    output_schema_json: String,
    options_schema_json: String,
    default_graph_options_json: String,
    summary: String,
    weight_col_policy: &'static str,
    output_schema_kind: &'static str,
    has_mode_schemas: bool,
}

fn title_case_algorithm_name(name: &str) -> String {
    match name {
        "pagerank" => return "PageRank".to_string(),
        "pagerank_strict" => return "PageRank Strict".to_string(),
        "bfs" => return "BFS".to_string(),
        "hits" => return "HITS".to_string(),
        "sssp" => return "SSSP".to_string(),
        "ecg" => return "ECG".to_string(),
        "k_core" => return "K-Core".to_string(),
        _ => {},
    }
    name.split('_')
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn algorithm_row(algorithm: CugraphAlgorithm, config: &CugraphSqlConfig) -> AlgorithmMetadataRow {
    AlgorithmMetadataRow {
        function_name: algorithm.function_name().to_string(),
        algorithm: title_case_algorithm_name(algorithm.native_algorithm().definition().name),
        enabled: config.allows(algorithm),
        experimental: false,
        requires_source_vertex: algorithm.requires_source_vertex(),
        accepts_weight_arg: true,
        consumes_weight: algorithm.native_algorithm().weight_policy().consumes_weight(),
        signature: signature(algorithm),
        allowed_arg_counts_json: allowed_arg_counts_json(algorithm),
        output_schema_json: schema_json(&stable_result_schema(algorithm)),
        options_schema_json: options_schema_json(algorithm),
        default_graph_options_json: graph_options_json(
            &algorithm.effective_default_graph_options(&config.default_graph_options),
        ),
        summary: summary(algorithm).to_string(),
        weight_col_policy: algorithm.weight_col_policy(),
        output_schema_kind: algorithm.output_schema_kind(),
        has_mode_schemas: algorithm.has_mode_schemas(),
    }
}

fn signature(algorithm: CugraphAlgorithm) -> String {
    let function_name = algorithm.function_name();
    if algorithm == CugraphAlgorithm::MinimumSpanningTree {
        format!("{function_name}(table_name, src_col, dst_col, weight_col [, options_json])")
    } else if algorithm.requires_source_vertex() {
        format!("{function_name}(table_name, source_vertex [, src_col, dst_col [, weight_col [, options_json]]])")
    } else {
        format!("{function_name}(table_name [, src_col, dst_col [, weight_col [, options_json]]])")
    }
}

fn allowed_arg_counts_json(algorithm: CugraphAlgorithm) -> String {
    if algorithm == CugraphAlgorithm::MinimumSpanningTree {
        "[4,5]".to_string()
    } else if algorithm.requires_source_vertex() {
        "[2,4,5,6]".to_string()
    } else {
        "[1,3,4,5]".to_string()
    }
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
