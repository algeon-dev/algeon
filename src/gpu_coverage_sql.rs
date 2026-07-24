//! SQL surface for query-level GPU coverage validation.

use std::sync::Arc;

use arrow::{
    array::{ArrayRef, BooleanArray, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::{DataFusionError, Result, ScalarValue},
    datasource::{MemTable, TableType},
    execution::{SessionState, context::SessionContext},
    logical_expr::Expr,
    physical_plan::ExecutionPlan,
};

use crate::{
    error::{Error, ErrorCode},
    gpu_coverage::{
        NEXUS_EXPLAIN_COVERAGE_FUNCTION, QueryGpuCoverage, coverage_validator_recursion_error,
        gpu_coverage_validation_in_progress, validate_query,
    },
};

/// Return the SQL table function that exposes [`validate_query`].
pub(crate) fn gpu_coverage_table_function() -> Arc<TableFunction> {
    Arc::new(TableFunction::new(NEXUS_EXPLAIN_COVERAGE_FUNCTION.to_string(), Arc::new(GpuCoverageTableFunction)))
}

/// Register [`NEXUS_EXPLAIN_COVERAGE_FUNCTION`] on an existing DataFusion context.
pub fn register_gpu_coverage_udtf(ctx: &SessionContext) {
    let function = gpu_coverage_table_function();
    ctx.register_udtf(function.name(), Arc::clone(function.function()));
}

#[derive(Debug)]
struct GpuCoverageTableFunction;

impl TableFunctionImpl for GpuCoverageTableFunction {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if args.len() != 1 {
            return Err(plan_error(format!("{NEXUS_EXPLAIN_COVERAGE_FUNCTION} expects one SQL string argument")));
        }
        Ok(Arc::new(GpuCoverageProvider { sql: literal_string_arg(args, 0, "sql")? }))
    }
}

#[derive(Debug, Clone)]
struct GpuCoverageProvider {
    sql: String,
}

#[async_trait::async_trait]
impl TableProvider for GpuCoverageProvider {
    fn schema(&self) -> SchemaRef {
        coverage_schema()
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
        let session_state = state.as_any().downcast_ref::<SessionState>().ok_or_else(|| {
            plan_error(format!("{NEXUS_EXPLAIN_COVERAGE_FUNCTION} requires a DataFusion SessionState for validation"))
        })?;
        if gpu_coverage_validation_in_progress(session_state) {
            return Err(DataFusionError::from(coverage_validator_recursion_error()));
        }
        let ctx = SessionContext::new_with_state(session_state.clone());
        let coverage = validate_query(&ctx, &self.sql).await.map_err(DataFusionError::from)?;
        let table = MemTable::try_new(coverage_schema(), vec![vec![coverage_batch(&coverage)?]])?;
        table.scan(state, projection, filters, limit).await
    }
}

fn coverage_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("overall", DataType::Utf8, false),
        Field::new("final_disposition", DataType::Utf8, false),
        Field::new("native_fragment_count", DataType::Int64, false),
        Field::new("explicit_gpu_island_count", DataType::Int64, false),
        Field::new("datafusion_boundary_count", DataType::Int64, false),
        Field::new("host_materialization_count", DataType::Int64, false),
        Field::new("confidence", DataType::Utf8, false),
        Field::new("planning_did_io", DataType::Boolean, false),
        Field::new("candidate_outcome_count", DataType::Int64, false),
        Field::new("runtime_caveat_count", DataType::Int64, false),
        Field::new("coverage_json", DataType::Utf8, false),
    ]))
}

fn coverage_batch(coverage: &QueryGpuCoverage) -> Result<RecordBatch> {
    RecordBatch::try_new(
        coverage_schema(),
        vec![
            strings(vec![coverage.overall.as_str().to_string()]),
            strings(vec![coverage.final_disposition.as_str().to_string()]),
            int64s(vec![usize_to_i64(coverage.domains.native_fragment_count, "native_fragment_count")?]),
            int64s(vec![usize_to_i64(coverage.domains.explicit_gpu_island_count, "explicit_gpu_island_count")?]),
            int64s(vec![usize_to_i64(coverage.domains.datafusion_boundary_count, "datafusion_boundary_count")?]),
            int64s(vec![usize_to_i64(coverage.domains.host_materialization_count, "host_materialization_count")?]),
            strings(vec![coverage.confidence.as_str().to_string()]),
            bools(vec![coverage.planning_did_io]),
            int64s(vec![usize_to_i64(coverage.candidate_outcomes.len(), "candidate_outcome_count")?]),
            int64s(vec![usize_to_i64(coverage.runtime_caveats.len(), "runtime_caveat_count")?]),
            strings(vec![coverage.to_json().to_string()]),
        ],
    )
    .map_err(Into::into)
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

fn usize_to_i64(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        Error::new(
            ErrorCode::Internal,
            "datafusion_nexus.gpu_coverage_sql.row",
            "coverage row count exceeded Int64 range",
        )
        .with_fact("field", field)
        .into()
    })
}

fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
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

#[cfg(test)]
mod tests {
    use datafusion::execution::SessionStateBuilder;

    use super::*;
    use crate::{
        native_exec::NexusNativeExecutionConfig, planner::NexusNativeOptimizerConfig,
        session::NexusSessionStateBuilderExt,
    };

    fn test_optimizer_config() -> NexusNativeOptimizerConfig {
        NexusNativeOptimizerConfig::default().with_native_execution_config(NexusNativeExecutionConfig::unbounded())
    }

    fn string_value(batch: &RecordBatch, column: usize, row: usize) -> &str {
        batch.column(column).as_any().downcast_ref::<StringArray>().expect("expected StringArray").value(row)
    }

    #[tokio::test]
    async fn native_session_builder_registers_coverage_table_function() {
        let state = SessionStateBuilder::new_with_default_features().with_nexus_native(test_optimizer_config()).build();
        assert!(state.table_functions().contains_key(NEXUS_EXPLAIN_COVERAGE_FUNCTION));
    }

    #[tokio::test]
    async fn coverage_table_function_returns_summary_and_json() {
        let state = SessionStateBuilder::new_with_default_features().with_nexus_native(test_optimizer_config()).build();
        let ctx = SessionContext::new_with_state(state);

        let batches = ctx
            .sql(
                "SELECT overall, final_disposition, coverage_json \
                 FROM nexus_explain_coverage('SELECT 1')",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 1);
        assert_ne!(string_value(batch, 0, 0), "rejected");
        let json: serde_json::Value = serde_json::from_str(string_value(batch, 2, 0)).unwrap();
        assert_eq!(json["overall"]["verdict"], string_value(batch, 0, 0));
        assert_eq!(json["final_disposition"], string_value(batch, 1, 0));
        assert_eq!(json["validated_under"]["native_optimizer_rule_installed"], true);
        assert_eq!(json["validated_under"]["final_plan_requirements"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn coverage_table_function_rejects_recursive_validation() {
        let state = SessionStateBuilder::new_with_default_features().with_nexus_native(test_optimizer_config()).build();
        let ctx = SessionContext::new_with_state(state);

        let batches = ctx
            .sql(
                "SELECT overall, coverage_json \
                 FROM nexus_explain_coverage('SELECT * FROM nexus_explain_coverage(''SELECT 1'')')",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = &batches[0];
        assert_eq!(string_value(batch, 0, 0), "rejected");
        let json: serde_json::Value = serde_json::from_str(string_value(batch, 1, 0)).unwrap();
        assert_eq!(json["overall"]["reason_code"], "coverage_validator_recursion_unsupported");
    }

    #[tokio::test]
    async fn coverage_table_function_rejects_view_indirect_recursive_validation() {
        let state = SessionStateBuilder::new_with_default_features().with_nexus_native(test_optimizer_config()).build();
        let ctx = SessionContext::new_with_state(state);

        ctx.sql(
            "CREATE VIEW coverage_recursive_view AS \
             SELECT * FROM nexus_explain_coverage('SELECT * FROM coverage_recursive_view')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

        let batches = ctx
            .sql(
                "SELECT overall, coverage_json \
                 FROM nexus_explain_coverage('SELECT * FROM coverage_recursive_view')",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = &batches[0];
        assert_eq!(string_value(batch, 0, 0), "rejected");
        let json: serde_json::Value = serde_json::from_str(string_value(batch, 1, 0)).unwrap();
        assert_eq!(json["overall"]["reason_code"], "coverage_validator_recursion_unsupported");
    }
}
