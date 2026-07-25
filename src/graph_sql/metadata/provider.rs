use std::sync::Arc;

use arrow::{
    array::{ArrayRef, BooleanArray, StringArray},
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
        CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_LIST_ALGORITHMS_FUNCTION, CUGRAPH_OPERATION_REGISTRY,
        CUGRAPH_VALIDATE_CALL_FUNCTION, CugraphAlgorithm, CugraphSqlConfig, literal_string_arg, plan_error,
    },
    algorithm_for_metadata_name,
    descriptions::{
        argument_descriptions_json, examples_json, limitations_json, primary_example_sql,
        result_field_descriptions_json, summary,
    },
    json_schema::{options_schema_json, result_schemas_json},
    validate::cugraph_validate_call,
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
        if !(1..=2).contains(&args.len()) {
            return Err(plan_error("cugraph_describe_algorithm expects function_name [, verbosity]"));
        }
        let verbosity = if args.len() == 2 {
            describe_verbosity(&literal_string_arg(args, 1, "verbosity")?)?
        } else {
            DescribeVerbosity::Compact
        };
        if algorithm_for_metadata_name(&function_name).is_none() {
            return Err(plan_error(format!("unknown cuGraph algorithm function '{function_name}'")));
        }
        Ok(Arc::new(CugraphMetadataProvider::describe(self.config.clone(), function_name, verbosity)))
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
    Describe { function_name: String, verbosity: DescribeVerbosity },
    Validate { function_name: String, table_name: String, call_json: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DescribeVerbosity {
    Compact,
    Verbose,
}

fn describe_verbosity(value: &str) -> Result<DescribeVerbosity> {
    match value {
        "compact" => Ok(DescribeVerbosity::Compact),
        "verbose" => Ok(DescribeVerbosity::Verbose),
        _ => Err(plan_error("cugraph_describe_algorithm verbosity must be 'compact' or 'verbose'")),
    }
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

    fn describe(config: CugraphSqlConfig, function_name: String, verbosity: DescribeVerbosity) -> Self {
        Self { config, kind: CugraphMetadataKind::Describe { function_name, verbosity } }
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
            CugraphMetadataKind::Describe { verbosity, .. } => describe_schema(verbosity),
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
            CugraphMetadataKind::Describe { function_name, verbosity } => {
                describe_batch(function_name, &self.config, *verbosity)?
            },
            CugraphMetadataKind::Validate { function_name, table_name, call_json } => {
                validate_batch(state, function_name, table_name, call_json, &self.config).await?
            },
        };
        let table = MemTable::try_new(self.schema(), vec![vec![batch]])?;
        table.scan(state, projection, filters, limit).await
    }
}

pub(super) fn list_schema() -> SchemaRef {
    Arc::new(Schema::new(compact_algorithm_fields()))
}

fn compact_algorithm_fields() -> Vec<Field> {
    vec![
        Field::new("function_name", DataType::Utf8, false),
        Field::new("algorithm", DataType::Utf8, false),
        Field::new("source_vertex_policy", DataType::Utf8, false),
        Field::new("weight_col_policy", DataType::Utf8, false),
        Field::new("output_schema_kind", DataType::Utf8, false),
        Field::new("signature", DataType::Utf8, false),
        Field::new("allowed_arg_counts_json", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
    ]
}

fn describe_schema(verbosity: DescribeVerbosity) -> SchemaRef {
    match verbosity {
        DescribeVerbosity::Compact => compact_describe_schema(),
        DescribeVerbosity::Verbose => verbose_describe_schema(),
    }
}

fn compact_describe_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("function_name", DataType::Utf8, false),
        Field::new("algorithm", DataType::Utf8, false),
        Field::new("source_vertex_policy", DataType::Utf8, false),
        Field::new("weight_col_policy", DataType::Utf8, false),
        Field::new("signature", DataType::Utf8, false),
        Field::new("allowed_arg_counts_json", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
        Field::new("primary_example_sql", DataType::Utf8, false),
    ]))
}

fn verbose_describe_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("function_name", DataType::Utf8, false),
        Field::new("algorithm", DataType::Utf8, false),
        Field::new("source_vertex_policy", DataType::Utf8, false),
        Field::new("weight_col_policy", DataType::Utf8, false),
        Field::new("output_schema_kind", DataType::Utf8, false),
        Field::new("signature", DataType::Utf8, false),
        Field::new("allowed_arg_counts_json", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
        Field::new("primary_example_sql", DataType::Utf8, false),
        Field::new("result_schemas_json", DataType::Utf8, false),
        Field::new("options_schema_json", DataType::Utf8, false),
        Field::new("argument_descriptions_json", DataType::Utf8, false),
        Field::new("result_field_descriptions_json", DataType::Utf8, false),
        Field::new("limitations_json", DataType::Utf8, false),
        Field::new("examples_json", DataType::Utf8, false),
    ]))
}

pub(super) fn validate_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("valid", DataType::Boolean, false),
        Field::new("error_code", DataType::Utf8, true),
        Field::new("message", DataType::Utf8, false),
        Field::new("function_name", DataType::Utf8, false),
        Field::new("table_resolved", DataType::Boolean, false),
        Field::new("src_col_resolved", DataType::Boolean, false),
        Field::new("dst_col_resolved", DataType::Boolean, false),
        Field::new("weight_col_resolved", DataType::Boolean, false),
        Field::new("src_dtype", DataType::Utf8, true),
        Field::new("dst_dtype", DataType::Utf8, true),
        Field::new("weight_dtype", DataType::Utf8, true),
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
            strings(rows.iter().map(|row| row.source_vertex_policy.to_string()).collect()),
            strings(rows.iter().map(|row| row.weight_col_policy.to_string()).collect()),
            strings(rows.iter().map(|row| row.output_schema_kind.to_string()).collect()),
            strings(rows.iter().map(|row| row.signature.clone()).collect()),
            strings(rows.iter().map(|row| row.allowed_arg_counts_json.clone()).collect()),
            strings(rows.iter().map(|row| row.summary.clone()).collect()),
        ],
    )
    .map_err(Into::into)
}

fn describe_batch(function_name: &str, config: &CugraphSqlConfig, verbosity: DescribeVerbosity) -> Result<RecordBatch> {
    let Some(algorithm) = algorithm_for_metadata_name(function_name) else {
        return Err(plan_error(format!("unknown cuGraph algorithm function '{function_name}'")));
    };
    let row = algorithm_row(algorithm, config);
    match verbosity {
        DescribeVerbosity::Compact => RecordBatch::try_new(
            compact_describe_schema(),
            vec![
                strings(vec![row.function_name]),
                strings(vec![row.algorithm]),
                strings(vec![row.source_vertex_policy.to_string()]),
                strings(vec![row.weight_col_policy.to_string()]),
                strings(vec![row.signature]),
                strings(vec![row.allowed_arg_counts_json]),
                strings(vec![row.summary]),
                strings(vec![row.primary_example_sql]),
            ],
        )
        .map_err(Into::into),
        DescribeVerbosity::Verbose => RecordBatch::try_new(
            verbose_describe_schema(),
            vec![
                strings(vec![row.function_name]),
                strings(vec![row.algorithm]),
                strings(vec![row.source_vertex_policy.to_string()]),
                strings(vec![row.weight_col_policy.to_string()]),
                strings(vec![row.output_schema_kind.to_string()]),
                strings(vec![row.signature]),
                strings(vec![row.allowed_arg_counts_json]),
                strings(vec![row.summary]),
                strings(vec![row.primary_example_sql]),
                strings(vec![result_schemas_json(algorithm)]),
                strings(vec![row.options_schema_json]),
                strings(vec![argument_descriptions_json(algorithm)]),
                strings(vec![result_field_descriptions_json(algorithm)]),
                strings(vec![limitations_json(algorithm)]),
                strings(vec![examples_json(algorithm)]),
            ],
        )
        .map_err(Into::into),
    }
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
            opt_strings(vec![row.error_code]),
            strings(vec![row.message]),
            strings(vec![row.function_name]),
            bools(vec![row.table_resolved]),
            bools(vec![row.src_col_resolved]),
            bools(vec![row.dst_col_resolved]),
            bools(vec![row.weight_col_resolved]),
            opt_strings(vec![row.src_dtype]),
            opt_strings(vec![row.dst_dtype]),
            opt_strings(vec![row.weight_dtype]),
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
    source_vertex_policy: &'static str,
    signature: String,
    allowed_arg_counts_json: String,
    options_schema_json: String,
    summary: String,
    primary_example_sql: String,
    weight_col_policy: &'static str,
    output_schema_kind: &'static str,
}

fn title_case_algorithm_name(name: &str) -> String {
    match name {
        "pagerank" => return "PageRank".to_string(),
        "personalized_pagerank" => return "Personalized PageRank".to_string(),
        "bfs" => return "BFS".to_string(),
        "hits" => return "HITS".to_string(),
        "sssp" => return "SSSP".to_string(),
        "ecg" => return "ECG".to_string(),
        "k_core" => return "K-Core".to_string(),
        "force_atlas2" => return "ForceAtlas2".to_string(),
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
        source_vertex_policy: source_vertex_policy(algorithm),
        signature: signature(algorithm),
        allowed_arg_counts_json: allowed_arg_counts_json(algorithm),
        options_schema_json: options_schema_json(algorithm, config),
        summary: summary(algorithm).to_string(),
        primary_example_sql: primary_example_sql(algorithm),
        weight_col_policy: algorithm.weight_col_policy(),
        output_schema_kind: algorithm.output_schema_kind(),
    }
}

fn source_vertex_policy(algorithm: CugraphAlgorithm) -> &'static str {
    if algorithm == CugraphAlgorithm::Bfs {
        "required_scalar_or_list_or_relation"
    } else if algorithm.requires_source_vertex() {
        "required_scalar"
    } else {
        "none"
    }
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

fn allowed_arg_counts_json(algorithm: CugraphAlgorithm) -> String {
    serde_json::to_string(algorithm.allowed_arg_counts()).expect("cuGraph SQL arity counts serialize")
}

fn strings(values: Vec<String>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

/// Build a nullable `StringArray` where `None` becomes a SQL NULL.
fn opt_strings(values: Vec<Option<String>>) -> ArrayRef {
    Arc::new(values.into_iter().collect::<StringArray>())
}

fn bools(values: Vec<bool>) -> ArrayRef {
    Arc::new(BooleanArray::from(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_metadata_requires_the_explicit_pair_relation_signature() {
        for algorithm in
            [CugraphAlgorithm::Jaccard, CugraphAlgorithm::Sorensen, CugraphAlgorithm::Overlap, CugraphAlgorithm::Cosine]
        {
            let function_name = algorithm.function_name();
            assert_eq!(
                signature(algorithm),
                format!("{function_name}(table_name, src_col, dst_col, weight_col, options_json)")
            );
            assert_eq!(allowed_arg_counts_json(algorithm), "[5]");
        }
    }
}
