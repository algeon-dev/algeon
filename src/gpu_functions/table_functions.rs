//! DataFusion table functions for the unified GPU function control plane.

use std::{any::TypeId, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::{DataFusionError, Result, ScalarValue},
    datasource::{MemTable, TableType, source_as_provider},
    logical_expr::{Expr, TableSource},
    physical_plan::ExecutionPlan,
};
use serde_json::{Map, Value};

use super::{GpuCallRequest, GpuCallValidation, GpuFunctionCatalog, GpuFunctionDescriptor};
use crate::{
    error::{Error, ErrorCode},
    gpu_coverage_sql::cpu_physical_boundary,
};

/// Unified GPU function discovery table-function name.
pub const GPU_LIST_FUNCTIONS_FUNCTION: &str = "gpu_list_functions";
/// Unified GPU function description table-function name.
pub const GPU_DESCRIBE_FUNCTION_FUNCTION: &str = "gpu_describe_function";
/// Unified GPU function dry-run validation table-function name.
pub const GPU_VALIDATE_CALL_FUNCTION: &str = "gpu_validate_call";

/// Build the three common metadata UDTFs over one immutable catalog snapshot.
pub(crate) fn gpu_metadata_table_functions(catalog: Arc<GpuFunctionCatalog>) -> Vec<Arc<TableFunction>> {
    vec![
        Arc::new(TableFunction::new(
            GPU_LIST_FUNCTIONS_FUNCTION.to_string(),
            Arc::new(GpuListFunctions { catalog: Arc::clone(&catalog) }),
        )),
        Arc::new(TableFunction::new(
            GPU_DESCRIBE_FUNCTION_FUNCTION.to_string(),
            Arc::new(GpuDescribeFunction { catalog: Arc::clone(&catalog) }),
        )),
        Arc::new(TableFunction::new(GPU_VALIDATE_CALL_FUNCTION.to_string(), Arc::new(GpuValidateCall { catalog }))),
    ]
}

#[derive(Debug)]
struct GpuListFunctions {
    catalog: Arc<GpuFunctionCatalog>,
}

impl TableFunctionImpl for GpuListFunctions {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if !args.is_empty() {
            return Err(plan_error(format!("{GPU_LIST_FUNCTIONS_FUNCTION} does not accept arguments")));
        }
        Ok(Arc::new(GpuMetadataProvider { catalog: Arc::clone(&self.catalog), kind: MetadataKind::List }))
    }
}

#[derive(Debug)]
struct GpuDescribeFunction {
    catalog: Arc<GpuFunctionCatalog>,
}

impl TableFunctionImpl for GpuDescribeFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if args.len() != 1 {
            return Err(plan_error(format!("{GPU_DESCRIBE_FUNCTION_FUNCTION} expects function_name")));
        }
        let function_name = literal_string(args, 0, "function_name")?;
        if self.catalog.get(&function_name).is_none() {
            return Err(plan_error(format!("unknown GPU function '{function_name}'")));
        }
        Ok(Arc::new(GpuMetadataProvider {
            catalog: Arc::clone(&self.catalog),
            kind: MetadataKind::Describe { function_name },
        }))
    }
}

#[derive(Debug)]
struct GpuValidateCall {
    catalog: Arc<GpuFunctionCatalog>,
}

impl TableFunctionImpl for GpuValidateCall {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if args.len() != 2 {
            return Err(plan_error(format!("{GPU_VALIDATE_CALL_FUNCTION} expects function_name, call_json")));
        }
        Ok(Arc::new(GpuMetadataProvider {
            catalog: Arc::clone(&self.catalog),
            kind: MetadataKind::Validate {
                function_name: literal_string(args, 0, "function_name")?,
                call_json: literal_string(args, 1, "call_json")?,
            },
        }))
    }
}

#[derive(Debug, Clone)]
enum MetadataKind {
    List,
    Describe { function_name: String },
    Validate { function_name: String, call_json: String },
}

#[derive(Debug, Clone)]
struct GpuMetadataProvider {
    catalog: Arc<GpuFunctionCatalog>,
    kind: MetadataKind,
}

/// Whether `source` is backed by a built-in GPU metadata table function.
pub fn is_builtin_gpu_metadata_table_source(source: &Arc<dyn TableSource>) -> bool {
    source_as_provider(source).is_ok_and(|provider| provider.as_ref().type_id() == TypeId::of::<GpuMetadataProvider>())
}

#[async_trait::async_trait]
impl TableProvider for GpuMetadataProvider {
    fn schema(&self) -> SchemaRef {
        match self.kind {
            MetadataKind::List => list_schema(),
            MetadataKind::Describe { .. } => describe_schema(),
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
            MetadataKind::List => list_batch(&self.catalog)?,
            MetadataKind::Describe { function_name } => describe_batch(&self.catalog, function_name)?,
            MetadataKind::Validate { function_name, call_json } => {
                validate_batch(&self.catalog, state, function_name, call_json).await?
            },
        };
        let input =
            MemTable::try_new(self.schema(), vec![vec![batch]])?.scan(state, projection, filters, limit).await?;
        Ok(cpu_physical_boundary(input))
    }
}

/// Stable schema for `gpu_list_functions`.
pub(crate) fn list_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("function_name", DataType::Utf8, false),
        Field::new("provider", DataType::Utf8, false),
        Field::new("available", DataType::Boolean, false),
        Field::new("unavailable_reason", DataType::Utf8, true),
        Field::new("signature", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
        Field::new("relation_roles_json", DataType::Utf8, false),
    ]))
}

/// Stable schema for `gpu_describe_function`.
pub(crate) fn describe_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("function_name", DataType::Utf8, false),
        Field::new("provider", DataType::Utf8, false),
        Field::new("available", DataType::Boolean, false),
        Field::new("unavailable_reason", DataType::Utf8, true),
        Field::new("signature", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
        Field::new("relation_roles_json", DataType::Utf8, false),
        Field::new("argument_descriptions_json", DataType::Utf8, false),
        Field::new("options_schema_json", DataType::Utf8, false),
        Field::new("result_schemas_json", DataType::Utf8, false),
        Field::new("validation_request_schema_json", DataType::Utf8, false),
        Field::new("examples_json", DataType::Utf8, false),
        Field::new("limitations_json", DataType::Utf8, false),
    ]))
}

/// Stable schema for `gpu_validate_call`.
pub(crate) fn validation_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("valid", DataType::Boolean, false),
        Field::new("error_code", DataType::Utf8, true),
        Field::new("message", DataType::Utf8, false),
        Field::new("function_name", DataType::Utf8, false),
        Field::new("relations_resolved", DataType::Boolean, false),
        Field::new("normalized_call_json", DataType::Utf8, false),
        Field::new("output_schema_json", DataType::Utf8, false),
        Field::new("would_execute_gpu", DataType::Boolean, false),
        Field::new("details_json", DataType::Utf8, false),
    ]))
}

fn list_batch(catalog: &GpuFunctionCatalog) -> Result<RecordBatch> {
    RecordBatch::try_new(
        list_schema(),
        vec![
            strings(catalog.entries().iter().map(|entry| entry.descriptor.function_name.as_str())),
            strings(catalog.entries().iter().map(|entry| entry.descriptor.provider.as_str())),
            bools(catalog.entries().iter().map(|entry| entry.descriptor.availability.is_available())),
            optional_strings(catalog.entries().iter().map(|entry| entry.descriptor.availability.unavailable_reason())),
            strings(catalog.entries().iter().map(|entry| entry.descriptor.signature.as_str())),
            strings(catalog.entries().iter().map(|entry| entry.descriptor.summary.as_str())),
            strings(catalog.entries().iter().map(|entry| json(&entry.descriptor.relation_roles))),
        ],
    )
    .map_err(Into::into)
}

fn describe_batch(catalog: &GpuFunctionCatalog, function_name: &str) -> Result<RecordBatch> {
    let descriptor =
        &catalog.get(function_name).expect("describe function existence checked when the UDTF is planned").descriptor;
    RecordBatch::try_new(
        describe_schema(),
        vec![
            strings([descriptor.function_name.as_str()]),
            strings([descriptor.provider.as_str()]),
            bools([descriptor.availability.is_available()]),
            optional_strings([descriptor.availability.unavailable_reason()]),
            strings([descriptor.signature.as_str()]),
            strings([descriptor.summary.as_str()]),
            strings([json(&descriptor.relation_roles)]),
            strings([json(&descriptor.argument_descriptions)]),
            strings([json(&descriptor.options_schema)]),
            strings([json(&descriptor.result_schemas)]),
            strings([json(&descriptor.validation_request_schema)]),
            strings([json(&descriptor.examples)]),
            strings([json(&descriptor.limitations)]),
        ],
    )
    .map_err(Into::into)
}

async fn validate_batch(
    catalog: &GpuFunctionCatalog,
    state: &dyn Session,
    function_name: &str,
    call_json: &str,
) -> Result<RecordBatch> {
    let row = match catalog.get(function_name) {
        None => GpuCallValidation::invalid(
            "unknown_function",
            format!("unknown GPU function '{function_name}'"),
            function_name,
            Value::Object(Map::new()),
        ),
        Some(entry) => match GpuCallRequest::parse(call_json) {
            Err(error) => GpuCallValidation::invalid(
                error.code,
                error.message,
                entry.descriptor.function_name.clone(),
                Value::Object(Map::new()),
            ),
            Ok(request) => match common_request_error(&entry.descriptor, &request) {
                Some((code, message)) => GpuCallValidation::invalid(
                    code,
                    message,
                    entry.descriptor.function_name.clone(),
                    request.normalized_json(),
                ),
                None if !entry.descriptor.availability.is_available() => GpuCallValidation::invalid(
                    entry.descriptor.availability.unavailable_reason().expect("unavailable variant has a reason"),
                    "GPU function is unavailable in this session/build",
                    entry.descriptor.function_name.clone(),
                    request.normalized_json(),
                ),
                None => entry.family.validate(state, &entry.descriptor.function_name, &request).await,
            },
        },
    };
    validation_record_batch(row)
}

fn common_request_error(
    descriptor: &GpuFunctionDescriptor,
    request: &GpuCallRequest,
) -> Option<(&'static str, String)> {
    let roles = descriptor.relation_role_names().expect("catalog validates descriptor roles");
    if let Some(role) = request.relations.keys().find(|role| !roles.contains(&role.as_str())) {
        return Some(("unknown_relation_role", format!("unknown relation role '{role}'")));
    }
    roles
        .iter()
        .find(|role| !request.relations.contains_key(**role))
        .map(|role| ("missing_relation", format!("missing required relation role '{role}'")))
}

fn validation_record_batch(row: GpuCallValidation) -> Result<RecordBatch> {
    RecordBatch::try_new(
        validation_schema(),
        vec![
            bools([row.valid]),
            optional_strings([row.error_code.as_deref()]),
            strings([row.message]),
            strings([row.function_name]),
            bools([row.relations_resolved]),
            strings([json(&row.normalized_call)]),
            strings([json(&row.output_schema)]),
            bools([row.would_execute_gpu]),
            strings([json(&row.details)]),
        ],
    )
    .map_err(Into::into)
}

fn literal_string(args: &[Expr], index: usize, name: &str) -> Result<String> {
    match args.get(index) {
        Some(Expr::Literal(
            ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)),
            _,
        )) => Ok(value.clone()),
        Some(_) => Err(plan_error(format!("{name} must be a non-null string literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}

fn strings(values: impl IntoIterator<Item = impl ToString>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values.into_iter().map(|value| value.to_string())))
}

fn optional_strings<'a>(values: impl IntoIterator<Item = Option<&'a str>>) -> ArrayRef {
    Arc::new(values.into_iter().collect::<StringArray>())
}

fn bools(values: impl IntoIterator<Item = bool>) -> ArrayRef {
    Arc::new(BooleanArray::from_iter(values))
}

fn json(value: &Value) -> String {
    serde_json::to_string(value).expect("serde_json::Value always serializes")
}

fn plan_error(message: impl Into<String>) -> DataFusionError {
    Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.gpu_functions.table_functions", message).into()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Array;
    use datafusion::execution::{SessionStateBuilder, context::SessionContext};
    use serde_json::json;

    use super::*;
    use crate::gpu_functions::{GpuFunctionAvailability, GpuFunctionFamily};

    #[derive(Debug)]
    struct FakeFamily;

    #[async_trait::async_trait]
    impl GpuFunctionFamily for FakeFamily {
        fn family_id(&self) -> &'static str {
            "fake"
        }

        fn descriptors(&self) -> Vec<GpuFunctionDescriptor> {
            vec![GpuFunctionDescriptor {
                function_name: "fake_operation".to_string(),
                provider: "fake".to_string(),
                signature: "fake_operation(input)".to_string(),
                summary: "fake operation".to_string(),
                relation_roles: json!(["input"]),
                argument_descriptions: json!([]),
                options_schema: json!({}),
                result_schemas: json!({}),
                validation_request_schema: json!({}),
                examples: json!([]),
                limitations: json!([]),
                availability: GpuFunctionAvailability::Available,
            }]
        }

        async fn validate(&self, _: &dyn Session, function_name: &str, request: &GpuCallRequest) -> GpuCallValidation {
            GpuCallValidation {
                valid: true,
                error_code: None,
                message: "valid".to_string(),
                function_name: function_name.to_string(),
                relations_resolved: true,
                normalized_call: request.normalized_json(),
                output_schema: json!({"fields": []}),
                would_execute_gpu: true,
                details: json!({"family": "fake"}),
            }
        }
    }

    fn catalog() -> Arc<GpuFunctionCatalog> {
        Arc::new(GpuFunctionCatalog::try_new(vec![Arc::new(FakeFamily)]).unwrap())
    }

    #[test]
    fn public_schemas_have_exact_contract_columns() {
        let names = |schema: SchemaRef| {
            schema
                .fields()
                .iter()
                .map(|field| (field.name().clone(), field.data_type().clone(), field.is_nullable()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names(list_schema()),
            vec![
                ("function_name".into(), DataType::Utf8, false),
                ("provider".into(), DataType::Utf8, false),
                ("available".into(), DataType::Boolean, false),
                ("unavailable_reason".into(), DataType::Utf8, true),
                ("signature".into(), DataType::Utf8, false),
                ("summary".into(), DataType::Utf8, false),
                ("relation_roles_json".into(), DataType::Utf8, false),
            ]
        );
        assert_eq!(
            names(describe_schema()),
            vec![
                ("function_name".into(), DataType::Utf8, false),
                ("provider".into(), DataType::Utf8, false),
                ("available".into(), DataType::Boolean, false),
                ("unavailable_reason".into(), DataType::Utf8, true),
                ("signature".into(), DataType::Utf8, false),
                ("summary".into(), DataType::Utf8, false),
                ("relation_roles_json".into(), DataType::Utf8, false),
                ("argument_descriptions_json".into(), DataType::Utf8, false),
                ("options_schema_json".into(), DataType::Utf8, false),
                ("result_schemas_json".into(), DataType::Utf8, false),
                ("validation_request_schema_json".into(), DataType::Utf8, false),
                ("examples_json".into(), DataType::Utf8, false),
                ("limitations_json".into(), DataType::Utf8, false),
            ]
        );
        assert_eq!(
            names(validation_schema()),
            vec![
                ("valid".into(), DataType::Boolean, false),
                ("error_code".into(), DataType::Utf8, true),
                ("message".into(), DataType::Utf8, false),
                ("function_name".into(), DataType::Utf8, false),
                ("relations_resolved".into(), DataType::Boolean, false),
                ("normalized_call_json".into(), DataType::Utf8, false),
                ("output_schema_json".into(), DataType::Utf8, false),
                ("would_execute_gpu".into(), DataType::Boolean, false),
                ("details_json".into(), DataType::Utf8, false),
            ]
        );
    }

    #[tokio::test]
    async fn fake_family_is_discoverable_describable_and_validated_end_to_end() {
        let mut state_builder = SessionStateBuilder::new();
        let functions = state_builder.table_functions().get_or_insert_with(Default::default);
        for function in gpu_metadata_table_functions(catalog()) {
            functions.insert(function.name().to_string(), function);
        }
        let ctx = SessionContext::new_with_state(state_builder.build());
        let list = ctx.sql("SELECT function_name FROM gpu_list_functions()").await.unwrap().collect().await.unwrap();
        let name = list[0].column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name.value(0), "fake_operation");
        let describe = ctx
            .sql("SELECT provider FROM gpu_describe_function('fake_operation')")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(describe[0].column(0).as_any().downcast_ref::<StringArray>().unwrap().value(0), "fake");
        let valid = ctx
            .sql(
                "SELECT valid, details_json FROM gpu_validate_call('fake_operation', \
                 '{\"schema_version\":1,\"relations\":{\"input\":{\"table\":\"input\"}},\"options\":{}}')",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert!(valid[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap().value(0));
        assert_eq!(
            serde_json::from_str::<Value>(valid[0].column(1).as_any().downcast_ref::<StringArray>().unwrap().value(0))
                .unwrap()["family"],
            "fake"
        );
    }

    #[tokio::test]
    async fn invalid_calls_return_structured_rows() {
        let mut state_builder = SessionStateBuilder::new();
        let functions = state_builder.table_functions().get_or_insert_with(Default::default);
        for function in gpu_metadata_table_functions(catalog()) {
            functions.insert(function.name().to_string(), function);
        }
        let ctx = SessionContext::new_with_state(state_builder.build());
        let batches = ctx
            .sql("SELECT error_code FROM gpu_validate_call('missing', '{\"schema_version\":1,\"relations\":{},\"options\":{}}')")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let code = batches[0].column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(code.value(0), "unknown_function");
        assert!(!code.is_null(0));
    }
}
