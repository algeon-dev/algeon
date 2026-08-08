//! SQL surface for query-level GPU coverage validation.

use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::{ArrayRef, Int64Array, StringArray},
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
        EXPLAIN_GPU_SCHEMA_VERSION, ExplainGpuOutputMode, NEXUS_EXPLAIN_COVERAGE_FUNCTION, QueryGpuCoverage,
        QueryGpuCoverageOverall, coverage_validator_recursion_error, gpu_coverage_validation_in_progress,
        validate_query,
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
        let table = MemTable::try_new(
            coverage_schema(),
            vec![vec![coverage_batch(&coverage, ExplainGpuOutputMode::Rows, None)?]],
        )?;
        table.scan(state, projection, filters, limit).await
    }
}

/// Shared Arrow schema for `nexus_explain_coverage` and Flight SQL `EXPLAIN GPU`.
///
/// The summary row carries complete machine-readable evidence in
/// `coverage_json`; every candidate row promotes its reason and remedy into
/// first-class columns for tabular clients.
pub(crate) fn coverage_schema() -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("query_id", DataType::Utf8, true),
            Field::new("row_kind", DataType::Utf8, false),
            Field::new("gpu_path", DataType::Utf8, false),
            Field::new("gpu_fragments", DataType::Int64, true),
            Field::new("cpu_boundaries", DataType::Int64, true),
            Field::new("physical_root", DataType::Utf8, true),
            Field::new("post_native_plan_root", DataType::Utf8, true),
            Field::new("candidate_shape", DataType::Utf8, true),
            Field::new("candidate_family", DataType::Utf8, true),
            Field::new("outcome", DataType::Utf8, false),
            Field::new("reason_code", DataType::Utf8, true),
            Field::new("unsupported_category", DataType::Utf8, true),
            Field::new("remedy_code", DataType::Utf8, true),
            Field::new("remedy_kind", DataType::Utf8, true),
            Field::new("remedy", DataType::Utf8, true),
            Field::new("detail_json", DataType::Utf8, true),
            Field::new("coverage_json", DataType::Utf8, true),
        ],
        HashMap::from([(
            "datafusion_nexus.explain_gpu.schema_version".to_string(),
            EXPLAIN_GPU_SCHEMA_VERSION.to_string(),
        )]),
    ))
}

/// Project shared coverage evidence into the public multi-row Arrow shape.
pub(crate) fn coverage_batch(
    coverage: &QueryGpuCoverage,
    output_mode: ExplainGpuOutputMode,
    query_id: Option<&str>,
) -> Result<RecordBatch> {
    let gpu_fragments = coverage.domains.native_fragment_count + coverage.domains.explicit_gpu_island_count;
    let summary_detail = serde_json::json!({
        "overall": coverage.overall.as_str(),
        "final_disposition": coverage.final_disposition.as_str(),
        "confidence": coverage.confidence.as_str(),
        "planning_did_io": coverage.planning_did_io,
        "runtime_caveat_count": coverage.runtime_caveats.len(),
    })
    .to_string();
    let mut query_ids = vec![query_id.map(str::to_string)];
    let mut row_kind = vec!["summary".to_string()];
    let mut gpu_path = vec![coverage.overall.as_gpu_path().to_string()];
    let mut gpu_fragments_column = vec![Some(usize_to_i64(gpu_fragments, "gpu_fragments")?)];
    let mut cpu_boundaries = vec![Some(usize_to_i64(coverage.domains.datafusion_boundary_count, "cpu_boundaries")?)];
    let mut physical_root = vec![coverage.input_physical_plan_root.clone()];
    let mut post_native_plan_root = vec![coverage.physical_plan_root.clone()];
    let mut candidate_shape = vec![None];
    let mut candidate_family = vec![None];
    let mut outcome = vec![summary_outcome(coverage).to_string()];
    let mut reason_code = vec![summary_reason_code(coverage)];
    let mut unsupported_category = vec![None];
    let mut remedy_code = vec![None];
    let mut remedy_kind = vec![None];
    let mut remedy = vec![None];
    let mut detail_json = vec![Some(summary_detail)];
    let mut coverage_json = vec![Some(coverage.to_json().to_string())];

    if output_mode == ExplainGpuOutputMode::Rows {
        for candidate in &coverage.candidate_outcomes {
            query_ids.push(query_id.map(str::to_string));
            row_kind.push("candidate".to_string());
            gpu_path.push(coverage.overall.as_gpu_path().to_string());
            gpu_fragments_column.push(None);
            cpu_boundaries.push(None);
            physical_root.push(None);
            post_native_plan_root.push(None);
            candidate_shape.push(Some(candidate.detail.clone()));
            candidate_family.push(Some(candidate.candidate_family.clone()));
            outcome.push(candidate.selection.as_str().to_string());
            reason_code.push(candidate.reason_code.clone());
            unsupported_category.push(candidate.category.clone());
            remedy_code.push(candidate.remedy.map(|remedy| remedy.code.to_string()));
            remedy_kind.push(candidate.remedy.map(|remedy| remedy.kind.as_str().to_string()));
            remedy.push(candidate.remedy.map(|remedy| remedy.guidance.to_string()));
            detail_json.push(Some(
                serde_json::json!({
                    "detail": candidate.detail,
                    "cost_evidence": candidate.cost_evidence.as_ref().map(|evidence| evidence.to_json()),
                })
                .to_string(),
            ));
            coverage_json.push(None);
        }
    }

    RecordBatch::try_new(
        coverage_schema(),
        vec![
            optional_strings(query_ids),
            strings(row_kind),
            strings(gpu_path),
            optional_int64s(gpu_fragments_column),
            optional_int64s(cpu_boundaries),
            optional_strings(physical_root),
            optional_strings(post_native_plan_root),
            optional_strings(candidate_shape),
            optional_strings(candidate_family),
            strings(outcome),
            optional_strings(reason_code),
            optional_strings(unsupported_category),
            optional_strings(remedy_code),
            optional_strings(remedy_kind),
            optional_strings(remedy),
            optional_strings(detail_json),
            optional_strings(coverage_json),
        ],
    )
    .map_err(Into::into)
}

fn summary_outcome(coverage: &QueryGpuCoverage) -> &'static str {
    match coverage.final_disposition {
        crate::planner::FinalPlanDisposition::Native | crate::planner::FinalPlanDisposition::Mixed => {
            "accepted_replaced"
        },
        crate::planner::FinalPlanDisposition::DataFusion => "retained_datafusion",
        crate::planner::FinalPlanDisposition::Rejected => "rejected_before_execution",
    }
}

fn summary_reason_code(coverage: &QueryGpuCoverage) -> Option<String> {
    match &coverage.overall {
        QueryGpuCoverageOverall::Rejected { reason_code, .. } => Some(reason_code.clone()),
        _ => None,
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
    Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.gpu_coverage_sql.plan", message).into()
}

fn strings(values: Vec<String>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

fn optional_strings(values: Vec<Option<String>>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

fn optional_int64s(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

#[cfg(test)]
mod tests {
    use arrow::array::Array;
    use datafusion::execution::SessionStateBuilder;

    use super::*;
    use crate::{
        backend::InstalledTestGpuBackend, native_exec::NexusNativeExecutionConfig, planner::NexusNativeOptimizerConfig,
        session::NexusSessionStateBuilderExt,
    };

    fn test_optimizer_config() -> NexusNativeOptimizerConfig {
        NexusNativeOptimizerConfig::default()
            .with_native_execution_config(NexusNativeExecutionConfig::bounded_default())
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
    async fn coverage_table_function_returns_summary_candidate_rows_and_json() {
        let backend = InstalledTestGpuBackend::new();
        let state = backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build();
        let ctx = SessionContext::new_with_state(state);

        let batches = ctx
            .sql(
                "SELECT query_id, row_kind, gpu_path, coverage_json \
                 FROM nexus_explain_coverage('SELECT 1') \
                 WHERE row_kind = 'summary'",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 1);
        let query_id = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert!(query_id.is_null(0), "embedded UDTF has no server query id");
        assert_eq!(string_value(batch, 1, 0), "summary");
        assert_eq!(string_value(batch, 2, 0), "native");
        assert_eq!(
            batch.schema().metadata().get("datafusion_nexus.explain_gpu.schema_version"),
            Some(&EXPLAIN_GPU_SCHEMA_VERSION.to_string())
        );
        let json: serde_json::Value = serde_json::from_str(string_value(batch, 3, 0)).unwrap();
        assert_eq!(json["overall"]["verdict"], "no_cpu_in_planned_path");
        assert_eq!(json["final_disposition"], "native");
        assert_eq!(json["validated_under"]["native_optimizer_rule_installed"], true);
        assert_eq!(json["validated_under"]["final_plan_requirements"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn coverage_table_function_promotes_candidate_remedies_to_columns() {
        let backend = InstalledTestGpuBackend::new();
        let state = backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build();
        let ctx = SessionContext::new_with_state(state);

        let batches = ctx
            .sql(
                "SELECT row_kind, reason_code, remedy_code, remedy_kind, remedy \
                 FROM nexus_explain_coverage('SELECT random() AS r') \
                 WHERE row_kind = 'candidate' AND reason_code = 'unsupported_expression'",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(string_value(batch, 0, 0), "candidate");
        assert_eq!(string_value(batch, 1, 0), "unsupported_expression");
        assert_eq!(string_value(batch, 2, 0), "rewrite_expression");
        assert_eq!(string_value(batch, 3, 0), "user_action");
        assert!(!string_value(batch, 4, 0).is_empty());
    }

    #[tokio::test]
    async fn coverage_table_function_rejects_recursive_validation() {
        let backend = InstalledTestGpuBackend::new();
        let state = backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build();
        let ctx = SessionContext::new_with_state(state);

        let batches = ctx
            .sql(
                "SELECT row_kind, reason_code, coverage_json \
                 FROM nexus_explain_coverage('SELECT * FROM nexus_explain_coverage(''SELECT 1'')') \
                 WHERE row_kind = 'summary'",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = &batches[0];
        assert_eq!(string_value(batch, 0, 0), "summary");
        assert_eq!(string_value(batch, 1, 0), "coverage_validator_recursion_unsupported");
        let json: serde_json::Value = serde_json::from_str(string_value(batch, 2, 0)).unwrap();
        assert_eq!(json["overall"]["reason_code"], "coverage_validator_recursion_unsupported");
    }

    #[tokio::test]
    async fn coverage_table_function_rejects_view_indirect_recursive_validation() {
        let backend = InstalledTestGpuBackend::new();
        let state = backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build();
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
                "SELECT row_kind, reason_code, coverage_json \
                 FROM nexus_explain_coverage('SELECT * FROM coverage_recursive_view') \
                 WHERE row_kind = 'summary'",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = &batches[0];
        assert_eq!(string_value(batch, 0, 0), "summary");
        assert_eq!(string_value(batch, 1, 0), "coverage_validator_recursion_unsupported");
        let json: serde_json::Value = serde_json::from_str(string_value(batch, 2, 0)).unwrap();
        assert_eq!(json["overall"]["reason_code"], "coverage_validator_recursion_unsupported");
    }
}
