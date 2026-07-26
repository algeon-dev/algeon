use std::sync::Arc;

use arrow::{
    array::{ArrayRef, BooleanArray, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::{Result, ScalarValue},
    datasource::{MemTable, TableType},
    logical_expr::Expr,
    physical_plan::ExecutionPlan,
};

use super::{
    CUVS_DESCRIBE_OPERATION_FUNCTION, CUVS_LIST_OPERATIONS_FUNCTION, CUVS_OPERATION_REGISTRY,
    CUVS_VALIDATE_CALL_FUNCTION, CuvsOperation, CuvsOutputContract, CuvsRelationArgumentSyntax,
    canonicalize_cuvs_options,
};
use crate::error::{Error, ErrorCode};

pub(crate) fn cuvs_metadata_table_functions() -> Vec<Arc<TableFunction>> {
    vec![
        Arc::new(TableFunction::new(CUVS_LIST_OPERATIONS_FUNCTION.to_string(), Arc::new(CuvsListOperationsFunction))),
        Arc::new(TableFunction::new(
            CUVS_DESCRIBE_OPERATION_FUNCTION.to_string(),
            Arc::new(CuvsDescribeOperationFunction),
        )),
        Arc::new(TableFunction::new(CUVS_VALIDATE_CALL_FUNCTION.to_string(), Arc::new(CuvsValidateCallFunction))),
    ]
}

#[derive(Debug)]
struct CuvsListOperationsFunction;

impl TableFunctionImpl for CuvsListOperationsFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if !args.is_empty() {
            return Err(metadata_error("cuvs_list_operations does not accept arguments"));
        }
        Ok(Arc::new(CuvsMetadataProvider { kind: MetadataKind::List }))
    }
}

#[derive(Debug)]
struct CuvsDescribeOperationFunction;

impl TableFunctionImpl for CuvsDescribeOperationFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if args.len() != 1 {
            return Err(metadata_error("cuvs_describe_operation expects function_name"));
        }
        let function_name = literal_string(&args[0], "function_name")?;
        let Some(operation) = CuvsOperation::from_sql_function_name(&function_name) else {
            return Err(metadata_error(format!("unknown cuVS operation function '{function_name}'")));
        };
        Ok(Arc::new(CuvsMetadataProvider { kind: MetadataKind::Describe(operation) }))
    }
}

#[derive(Debug)]
struct CuvsValidateCallFunction;

impl TableFunctionImpl for CuvsValidateCallFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if args.len() != 2 {
            return Err(metadata_error("cuvs_validate_call expects function_name, options_json"));
        }
        let function_name = literal_string(&args[0], "function_name")?;
        let options_json = literal_string(&args[1], "options_json")?;
        Ok(Arc::new(CuvsMetadataProvider { kind: MetadataKind::Validate { function_name, options_json } }))
    }
}

#[derive(Debug, Clone)]
enum MetadataKind {
    List,
    Describe(CuvsOperation),
    Validate { function_name: String, options_json: String },
}

#[derive(Debug, Clone)]
struct CuvsMetadataProvider {
    kind: MetadataKind,
}

#[async_trait::async_trait]
impl TableProvider for CuvsMetadataProvider {
    fn schema(&self) -> SchemaRef {
        match self.kind {
            MetadataKind::List | MetadataKind::Describe(_) => operation_schema(),
            MetadataKind::Validate { .. } => validation_schema(),
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
            MetadataKind::List => {
                operation_batch(CUVS_OPERATION_REGISTRY.iter().map(|descriptor| descriptor.operation))?
            },
            MetadataKind::Describe(operation) => operation_batch(std::iter::once(*operation))?,
            MetadataKind::Validate { function_name, options_json } => validation_batch(function_name, options_json)?,
        };
        MemTable::try_new(self.schema(), vec![vec![batch]])?.scan(state, projection, filters, limit).await
    }
}

fn operation_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("function_name", DataType::Utf8, false),
        Field::new("operation", DataType::Utf8, false),
        Field::new("relation_roles_json", DataType::Utf8, false),
        Field::new("relation_argument_syntax", DataType::Utf8, false),
        Field::new("output_contract", DataType::Utf8, false),
        Field::new("options_schema_json", DataType::Utf8, false),
    ]))
}

fn validation_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("valid", DataType::Boolean, false),
        Field::new("error_code", DataType::Utf8, true),
        Field::new("message", DataType::Utf8, false),
        Field::new("function_name", DataType::Utf8, false),
        Field::new("normalized_options_json", DataType::Utf8, false),
    ]))
}

fn operation_batch(operations: impl Iterator<Item = CuvsOperation>) -> Result<RecordBatch> {
    let operations = operations.collect::<Vec<_>>();
    let descriptors = operations.iter().map(|operation| operation.descriptor()).collect::<Vec<_>>();
    RecordBatch::try_new(
        operation_schema(),
        vec![
            strings(descriptors.iter().map(|descriptor| descriptor.sql_function_name)),
            strings(operations.iter().map(|operation| operation_name(*operation))),
            strings(descriptors.iter().map(|descriptor| {
                serde_json::Value::Array(
                    descriptor
                        .relation_roles
                        .iter()
                        .map(|role| serde_json::Value::String(role.as_str().to_string()))
                        .collect(),
                )
                .to_string()
            })),
            strings(descriptors.iter().map(|descriptor| match descriptor.relation_argument_syntax {
                CuvsRelationArgumentSyntax::ParenthesizedSubquery => "parenthesized_subquery",
            })),
            strings(descriptors.iter().map(|descriptor| output_contract_name(descriptor.output_contract))),
            strings(descriptors.iter().map(|descriptor| descriptor.options_json_schema().to_string())),
        ],
    )
    .map_err(Into::into)
}

fn validation_batch(function_name: &str, options_json: &str) -> Result<RecordBatch> {
    let (valid, error_code, message, canonical_name, normalized) =
        match CuvsOperation::from_sql_function_name(function_name) {
            None => (
                false,
                Some("unknown_operation".to_string()),
                format!("unknown cuVS operation function '{function_name}'"),
                function_name.to_string(),
                "{}".to_string(),
            ),
            Some(operation) => match canonicalize_cuvs_options(operation, options_json) {
                Ok(options) => (
                    true,
                    None,
                    "valid".to_string(),
                    operation.descriptor().sql_function_name.to_string(),
                    options.to_string(),
                ),
                Err(error) => (
                    false,
                    Some(error.code().to_string()),
                    error.message().to_string(),
                    operation.descriptor().sql_function_name.to_string(),
                    "{}".to_string(),
                ),
            },
        };
    RecordBatch::try_new(
        validation_schema(),
        vec![
            Arc::new(BooleanArray::from(vec![valid])),
            Arc::new(StringArray::from(vec![error_code.as_deref()])),
            strings(std::iter::once(message)),
            strings(std::iter::once(canonical_name)),
            strings(std::iter::once(normalized)),
        ],
    )
    .map_err(Into::into)
}

fn literal_string(expr: &Expr, name: &'static str) -> Result<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(value)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(value)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(value)), _) => Ok(value.clone()),
        _ => Err(metadata_error(format!("cuVS metadata argument '{name}' must be a non-null string literal"))),
    }
}

fn strings(values: impl IntoIterator<Item = impl ToString>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values.into_iter().map(|value| value.to_string())))
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

fn metadata_error(message: impl Into<String>) -> datafusion::common::DataFusionError {
    Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.cuvs_sql.metadata", message).into()
}
