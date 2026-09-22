use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use algeon_datafusion::{
    attempt::{AlgeonAttempt, AttemptOptions, QueryAttemptReport},
    backend::AlgeonGpuBackend,
    error::DataFusionErrorExt,
};
use arrow::{
    array::{Array, Float32Array, Float64Array},
    datatypes::{DataType, SchemaRef},
    error::ArrowError,
    record_batch::RecordBatch,
    util::display::array_value_to_string,
};
use datafusion::{
    common::DataFusionError,
    physical_plan::{RecordBatchStream, displayable},
    prelude::SessionContext,
};
use futures::StreamExt as _;
use query_engine::harness::StableFnv1a64;
use tokio::time::timeout;

use super::{config::GoodputConfig, workload::WorkloadItem};
use crate::plan_artifacts::append_compiled_pipeline_diagnostic;

/// Raw facts observed for one executed arrival, before classification.
#[derive(Debug)]
pub(super) struct AttemptOutcome {
    pub(super) started: Instant,
    pub(super) finished: Instant,
    pub(super) report: Option<Arc<QueryAttemptReport>>,
    pub(super) adapter_error_code: Option<String>,
    pub(super) error_message: Option<String>,
    pub(super) timed_out: bool,
    pub(super) output_rows: Option<usize>,
    pub(super) output_fingerprint: Option<String>,
}

/// Runs one query as one attempt and drains its result stream.
///
/// The completion receiver is taken before planning so a rejection during
/// admission still yields the immutable report rather than a client-side
/// error alone.
pub(super) async fn run_attempt(
    backend: &AlgeonGpuBackend,
    ctx: &SessionContext,
    config: &GoodputConfig,
    item: &WorkloadItem,
    requested_query_cap_bytes: u64,
    query_id: String,
    plan_artifact_dir: Option<&Path>,
) -> AttemptOutcome {
    let started = Instant::now();
    let attempt = match backend.begin_attempt_with_query_cap_bytes(
        AttemptOptions { query_id: Some(query_id), ..AttemptOptions::default() },
        requested_query_cap_bytes,
    ) {
        Ok(attempt) => attempt,
        Err(error) => {
            return AttemptOutcome {
                started,
                finished: Instant::now(),
                report: None,
                adapter_error_code: Some(error.code().as_str().to_string()),
                error_message: Some(error.to_string()),
                timed_out: false,
                output_rows: None,
                output_fingerprint: None,
            };
        },
    };
    let completion = attempt.completion();
    let drained = timeout(config.query_timeout, drain(&attempt, ctx, item, config, plan_artifact_dir)).await;
    let timed_out = drained.is_err();
    if timed_out {
        // A harness-side timeout is not a service outcome. Cancel so the
        // attempt reaches its own terminal state and returns its capacity
        // instead of being abandoned mid-execution.
        attempt.cancellation().cancel();
    }
    let (output, failure) = match drained {
        Ok(Ok(output)) => (Some(output), None),
        Ok(Err(error)) => (None, Some(error)),
        Err(_) => (None, None),
    };
    drop(attempt);
    let report = completion.wait_timeout(config.query_timeout);
    let finished = Instant::now();
    let (output_rows, output_fingerprint) = match output {
        Some(output) => (Some(output.rows), Some(output.fingerprint)),
        None => (None, None),
    };
    AttemptOutcome {
        started,
        finished,
        report,
        adapter_error_code: failure.as_ref().and_then(adapter_error_code),
        error_message: failure
            .as_ref()
            .map(ToString::to_string)
            .or_else(|| timed_out.then(|| format!("harness timeout after {} ms", config.query_timeout.as_millis()))),
        timed_out,
        output_rows,
        output_fingerprint,
    }
}

struct DrainedOutput {
    rows: usize,
    fingerprint: String,
}

async fn drain(
    attempt: &AlgeonAttempt,
    ctx: &SessionContext,
    item: &WorkloadItem,
    config: &GoodputConfig,
    plan_artifact_dir: Option<&Path>,
) -> Result<DrainedOutput, DataFusionError> {
    let mut stream = match plan_artifact_dir {
        Some(dir) => {
            let plan = attempt.plan_sql(ctx, &item.sql).await?;
            let mut text = format!("physical_plan:\n{}\n", displayable(plan.as_ref()).indent(true));
            text.push_str("\ncompiled_pipelines_recompiled_for_diagnostics:\n");
            append_compiled_pipeline_diagnostic(&mut text, &plan);
            std::fs::write(dir.join(format!("{}.plan.txt", item.id)), text)
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
            attempt.execute_physical_plan(plan, ctx)?
        },
        None => attempt.sql(ctx, &item.sql).await?,
    };
    let schema = stream.schema();
    let mut rows = 0_usize;
    let mut retained = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        rows += batch.num_rows();
        if rows <= config.fingerprint_max_rows {
            retained.push(batch);
        } else {
            retained.clear();
        }
    }
    let fingerprint = if rows <= config.fingerprint_max_rows {
        fingerprint_batches(&schema, &retained).map_err(|error| DataFusionError::ArrowError(Box::new(error), None))?
    } else {
        // Beyond the retention bound the harness compares row counts only.
        // Recording that explicitly keeps a truncated comparison from being
        // read as a verified content match.
        format!("rows-only:{rows}")
    };
    Ok(DrainedOutput { rows, fingerprint })
}

fn adapter_error_code(error: &DataFusionError) -> Option<String> {
    error.find_algeon_error().map(|error| error.code().as_str().to_string())
}

/// Order-independent multiset fingerprint over the collected output.
fn fingerprint_batches(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<String, ArrowError> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("algeon-datafusion-goodput-output-multiset-v1");
    hasher.update_field("schema_metadata");
    hasher.update_metadata(schema.metadata());
    for field in schema.fields() {
        hasher.update_field("field");
        hasher.update_field(field.name());
        hasher.update_field(&field.data_type().to_string());
        hasher.update_field(if field.is_nullable() { "nullable" } else { "non_nullable" });
    }
    let mut row_fingerprints = Vec::with_capacity(batches.iter().map(RecordBatch::num_rows).sum());
    for batch in batches {
        if batch.schema().fields() != schema.fields() {
            return Err(ArrowError::SchemaError(
                "goodput output fingerprint requires collected batches to match the planned schema".to_string(),
            ));
        }
        for row in 0..batch.num_rows() {
            let mut row_hasher = StableFnv1a64::new();
            row_hasher.update_field("algeon-datafusion-goodput-row-v1");
            for column in batch.columns() {
                if column.is_null(row) {
                    row_hasher.update_field("null");
                } else {
                    row_hasher.update_field("value");
                    row_hasher.update_field(&render_value(column.as_ref(), row)?);
                }
            }
            row_fingerprints.push(row_hasher.finish());
        }
    }
    row_fingerprints.sort_unstable();
    hasher.update_field(&row_fingerprints.len().to_string());
    for row_fingerprint in row_fingerprints {
        hasher.update_bytes(&row_fingerprint.to_le_bytes());
    }
    Ok(format!("output-fnv1a64:{:016x}", hasher.finish()))
}

/// Renders one value for the output fingerprint.
///
/// Floating-point columns are rendered at a fixed significant-digit precision
/// rather than exactly. A reduction over millions of rows is not required to
/// keep its summation order between runs, so two correct executions can differ
/// in the last bits; comparing at full precision would report those as wrong
/// results. The retained precision is far above the reordering noise, which
/// for a sum of N terms grows like sqrt(N) machine epsilons.
fn render_value(column: &dyn Array, row: usize) -> Result<String, ArrowError> {
    match column.data_type() {
        DataType::Float64 => {
            let values = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| ArrowError::CastError("Float64 column did not downcast".to_string()))?;
            Ok(format!("{:.9e}", values.value(row)))
        },
        DataType::Float32 => {
            let values = column
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| ArrowError::CastError("Float32 column did not downcast".to_string()))?;
            Ok(format!("{:.4e}", values.value(row)))
        },
        _ => array_value_to_string(column, row),
    }
}

/// Wall-clock duration in milliseconds.
pub(super) fn duration_ms(duration: Duration) -> f64 {
    duration.as_nanos() as f64 / 1_000_000.0
}

pub(super) fn optional_ns_ms(value: Option<u64>) -> Option<f64> {
    value.map(|value| value as f64 / 1_000_000.0)
}
