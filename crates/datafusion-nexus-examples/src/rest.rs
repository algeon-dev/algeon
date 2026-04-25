//! Axum HTTP API frontend for the embedded backend example.
//!
//! This module exposes health, pipeline execution, planning-report, and GPU
//! admission/memory status endpoints for the shared backend.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow_json::{WriterBuilder, writer::JsonArray};
use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::SessionStateBuilder,
};
use datafusion_nexus::{
    error::{DataFusionErrorExt, Error, ErrorCode, ErrorKind},
    planner::CudfNativePhysicalOptimizerRule,
    report::PlanningReport,
    session::CudfSessionStateBuilderExt,
};
use nexus_query_engine::error::{QueryError, QueryErrorKind};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    AppState,
    pipelines::{Pipeline, PipelineValidationError, RenderedPipeline, default_param_body, summaries},
};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/api/pipelines", get(list_pipelines))
        .route("/api/pipelines/{name}", post(run_pipeline))
        .route("/api/pipelines/{name}/plan", get(plan_pipeline))
        .route("/api/gpu", get(gpu))
        .with_state(state)
}

async fn healthz(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let required = ["papers", "paper_fos", "citation_edges", "citation_edges_by_dst"];
    let mut tables = Vec::with_capacity(required.len());
    for table in required {
        let visible = state.ctx.table_exist(table).map_err(AppError::from_datafusion)?;
        tables.push(json!({ "name": table, "visible": visible }));
    }
    let ready = tables.iter().all(|table| table["visible"].as_bool().unwrap_or(false));
    let status = if ready { "ok" } else { "degraded" };
    Ok(Json(json!({
        "status": status,
        "tables": tables
    })))
}

async fn list_pipelines(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "pipelines": summaries(state.pipelines)
    }))
}

async fn run_pipeline(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, AppError> {
    let Json(body) = body.map_err(AppError::from_json_rejection)?;
    let pipeline = pipeline_by_name(&state, &name)?;
    let rendered = pipeline.render_request(&body).map_err(AppError::from_validation)?;
    let batches = execute_pipeline(&state, rendered).await?;
    let rows = record_batches_to_json(&batches)?;
    Ok(Json(json!({
        "pipeline": pipeline.name,
        "rows": rows
    })))
}

async fn plan_pipeline(State(state): State<AppState>, Path(name): Path<String>) -> Result<Json<Value>, AppError> {
    let pipeline = pipeline_by_name(&state, &name)?;
    let rendered = pipeline.render_defaults().map_err(AppError::from_validation)?;
    let report = planning_report(&state, rendered).await?;
    Ok(Json(json!({
        "pipeline": pipeline.name,
        "params": default_param_body(pipeline),
        "planning_report": report.to_json()
    })))
}

async fn gpu(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let service = state.backend.query_service();
    let snapshot = service.admission_snapshot().map_err(AppError::from_native)?;
    let memory_governors = service.memory_governor_reports().map_err(AppError::from_native)?;
    let memory_failures = service.device_memory_failure_reports();
    Ok(Json(json!({
        "admission": {
            "active_queries": snapshot.active_queries,
            "queued_queries": snapshot.queued_queries,
            "queued_by_tenant": snapshot.queued_by_tenant.into_iter().map(|tenant| json!({
                "tenant_id": tenant.tenant_id.as_str(),
                "queued_queries": tenant.queued_queries
            })).collect::<Vec<_>>(),
            "next_tenant_id": snapshot.next_tenant_id.map(|tenant| tenant.as_str().to_string())
        },
        "memory_governors": memory_governors.into_iter().map(|report| json!({
            "device_ordinal": report.device_ordinal,
            "capacity_bytes": report.capacity_bytes,
            "outstanding_bytes": report.outstanding_bytes,
            "available_bytes": report.available_bytes,
            "evictable_bytes": report.evictable_bytes,
            "free_bytes": report.free_bytes,
            "grants": report.grants.into_iter().map(|grant| json!({
                "grant_id": grant.grant_id,
                "query_id": grant.query_id,
                "priority": grant.priority,
                "budget_bytes": grant.budget_bytes
            })).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "device_memory_failures": memory_failures.into_iter().map(|report| json!({
            "device_ordinal": report.device_ordinal,
            "callback_installed": report.callback_installed,
            "callback_install_unavailable": report.callback_install_unavailable,
            "allocation_failure_count": report.allocation_failure_count,
            "total_failed_allocation_bytes": report.total_failed_allocation_bytes,
            "largest_failed_allocation_bytes": report.largest_failed_allocation_bytes,
            "trim_retry_count": report.trim_retry_count,
            "pool_usage_observation_count": report.pool_usage_observation_count,
            "last_failed_allocation_bytes": report.last_failed_allocation_bytes,
            "last_pool_reserved_current_bytes": report.last_pool_reserved_current_bytes,
            "last_pool_used_current_bytes": report.last_pool_used_current_bytes
        })).collect::<Vec<_>>()
    })))
}

fn pipeline_by_name(state: &AppState, name: &str) -> Result<&'static Pipeline, AppError> {
    state
        .pipelines
        .get(name)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "pipeline_not_found", format!("unknown pipeline `{name}`")))
}

async fn execute_pipeline(state: &AppState, rendered: RenderedPipeline) -> Result<Vec<RecordBatch>, AppError> {
    let mut df = state.ctx.sql(&rendered.sql).await.map_err(AppError::from_datafusion)?;
    if !rendered.param_values.is_empty() {
        df = df.with_param_values(rendered.param_values).map_err(AppError::from_datafusion)?;
    }
    df.collect().await.map_err(AppError::from_datafusion)
}

async fn planning_report(state: &AppState, rendered: RenderedPipeline) -> Result<PlanningReport, AppError> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(state.backend.optimizer_config().clone()));
    let planning_state = SessionStateBuilder::new_from_existing(state.ctx.state())
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let planning_ctx = datafusion::prelude::SessionContext::new_with_state(planning_state);
    let mut df = planning_ctx.sql(&rendered.sql).await.map_err(AppError::from_datafusion)?;
    if !rendered.param_values.is_empty() {
        df = df.with_param_values(rendered.param_values).map_err(AppError::from_datafusion)?;
    }
    let _plan = df.create_physical_plan().await.map_err(AppError::from_datafusion)?;
    native_rule.last_report().ok_or_else(|| {
        AppError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "planning_report_missing",
            "native optimizer did not retain a planning report",
        )
    })
}

fn record_batches_to_json(batches: &[RecordBatch]) -> Result<Value, AppError> {
    let mut buffer = Vec::new();
    {
        let mut writer = WriterBuilder::new().with_explicit_nulls(true).build::<_, JsonArray>(&mut buffer);
        for batch in batches {
            writer.write(batch).map_err(AppError::from_arrow)?;
        }
        writer.finish().map_err(AppError::from_arrow)?;
    }
    serde_json::from_slice(&buffer).map_err(AppError::from_serde)
}

#[derive(Debug, Clone)]
pub struct AppError {
    status: StatusCode,
    code: String,
    message: String,
    retry_after_secs: Option<u64>,
    facts: Vec<(String, String)>,
}

impl AppError {
    fn new(status: StatusCode, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self { status, code: code.into(), message: message.into(), retry_after_secs: None, facts: Vec::new() }
    }

    fn from_validation(error: PipelineValidationError) -> Self {
        let mut facts = Vec::new();
        if let Some(param) = error.param() {
            facts.push(("param".to_string(), param.to_string()));
        }
        Self {
            status: StatusCode::BAD_REQUEST,
            code: error.code().to_string(),
            message: error.message().to_string(),
            retry_after_secs: None,
            facts,
        }
    }

    fn from_json_rejection(error: JsonRejection) -> Self {
        Self::new(error.status(), "invalid_json_body", error.body_text())
    }

    fn from_datafusion(error: DataFusionError) -> Self {
        // Enrich errors produced by Nexus first. If no typed Nexus error is
        // discoverable, keep this application's existing DataFusion policy;
        // ordinary DataFusion errors do not acquire a Nexus identity.
        if let Some(adapter) = error.find_nexus_error() {
            return Self::from_adapter_error(adapter);
        }
        if find_datafusion_resources_exhausted(&error) {
            return Self {
                status: StatusCode::TOO_MANY_REQUESTS,
                code: "resource_exhausted".to_string(),
                message: error.to_string(),
                retry_after_secs: Some(1),
                facts: Vec::new(),
            };
        }
        let status = match &error {
            DataFusionError::Plan(_) | DataFusionError::SchemaError(_, _) | DataFusionError::SQL(_, _) => {
                StatusCode::BAD_REQUEST
            },
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, "datafusion_error", error.to_string())
    }

    fn from_native(error: QueryError) -> Self {
        let status = if error.kind() == QueryErrorKind::ResourceExhausted && error.is_retryable() {
            StatusCode::TOO_MANY_REQUESTS
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        let retry_after_secs = (status == StatusCode::TOO_MANY_REQUESTS).then_some(1);
        Self {
            status,
            code: error.code().as_str().to_string(),
            message: error.message().to_string(),
            retry_after_secs,
            // Native facts are raw in-process diagnostics, not an HTTP
            // disclosure contract. Select client fields explicitly instead.
            facts: Vec::new(),
        }
    }

    fn from_adapter_error(error: &Error) -> Self {
        let status = match error.kind() {
            ErrorKind::ResourceExhausted if error.is_retryable() => StatusCode::TOO_MANY_REQUESTS,
            ErrorKind::Planning | ErrorKind::InvalidInput | ErrorKind::Unsupported | ErrorKind::NotFound => {
                StatusCode::BAD_REQUEST
            },
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            code: error.code().as_str().to_string(),
            message: error.message().to_string(),
            retry_after_secs: (status == StatusCode::TOO_MANY_REQUESTS).then_some(1),
            // Error facts remain available to trusted typed policy through
            // `error.facts()`. This example does not forward them wholesale.
            facts: Vec::new(),
        }
    }

    fn from_arrow(error: arrow::error::ArrowError) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "arrow_json_error", error.to_string())
    }

    fn from_serde(error: serde_json::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "json_error", error.to_string())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        if let Some(seconds) = self.retry_after_secs {
            let value = HeaderValue::from_str(&seconds.to_string()).unwrap_or_else(|_| HeaderValue::from_static("1"));
            headers.insert(header::RETRY_AFTER, value);
        }
        let body = Json(ErrorBody {
            error: ErrorPayload { code: self.code, message: self.message, facts: self.facts.into_iter().collect() },
        });
        (self.status, headers, body).into_response()
    }
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorPayload,
}

#[derive(Debug, Serialize)]
struct ErrorPayload {
    code: String,
    message: String,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    facts: std::collections::BTreeMap<String, String>,
}

fn find_datafusion_resources_exhausted(error: &DataFusionError) -> bool {
    match error {
        DataFusionError::ResourcesExhausted(_) => true,
        DataFusionError::External(source) => {
            source.downcast_ref::<DataFusionError>().is_some_and(find_datafusion_resources_exhausted)
        },
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            find_datafusion_resources_exhausted(source)
        },
        DataFusionError::Collection(errors) => errors.iter().any(find_datafusion_resources_exhausted),
        DataFusionError::Shared(source) => find_datafusion_resources_exhausted(source),
        _ => false,
    }
}

#[allow(dead_code)]
fn _assert_datafusion_result_send_sync<T>(result: DataFusionResult<T>) -> DataFusionResult<T> {
    result
}

fn _pin_error_code_use(code: ErrorCode) -> &'static str {
    code.as_str()
}
