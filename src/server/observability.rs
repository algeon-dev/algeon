use std::{
    fmt::Write,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use datafusion::physical_plan::{ExecutionPlan, displayable, metrics::MetricsSet};
use nexus_query_engine::{LakehouseCacheFillEventSnapshot, QueryMetrics};

use crate::{
    Error,
    native_exec::{NativeCudfMetrics, NativeMetricsCaptureSink},
    native_report::{
        NativeDataFusionExecutionReport, NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningReport,
        NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime,
    },
    planner::NativePlanningDiagnostics,
};

const DEFAULT_QUERY_OBSERVABILITY_CAP: usize = 1024;
const DEFAULT_QUERY_OBSERVABILITY_TTL: Duration = Duration::from_secs(60 * 60);

pub(super) type QueryId = String;

#[derive(Debug)]
pub(super) struct QueryObservability {
    captures: DashMap<QueryId, CapturedQuery>,
    cap: usize,
    ttl: Duration,
}

impl Default for QueryObservability {
    fn default() -> Self {
        Self::new(DEFAULT_QUERY_OBSERVABILITY_CAP, DEFAULT_QUERY_OBSERVABILITY_TTL)
    }
}

impl QueryObservability {
    pub(super) fn new(cap: usize, ttl: Duration) -> Self {
        Self { captures: DashMap::new(), cap: cap.max(1), ttl }
    }

    pub(super) fn start_query(&self, query_id: impl Into<String>, sql: impl Into<String>) {
        let now = Instant::now();
        let query_id = query_id.into();
        self.captures.insert(query_id.clone(), CapturedQuery::new(query_id, sql.into(), now));
        self.enforce_bounds(now);
    }

    pub(super) fn capture_planning(
        &self,
        query_id: &str,
        diagnostics: Option<NativePlanningDiagnostics>,
        planned_plan: Arc<dyn ExecutionPlan>,
    ) {
        let now = Instant::now();
        {
            let mut query = self.ensure_query(query_id, now);
            query.planning = diagnostics
                .map(|diagnostics| CapturedPlanning { diagnostics, planned_plan: Arc::clone(&planned_plan) });
            query.planning_failure = None;
        }
        let physical_explain = displayable(planned_plan.as_ref()).indent(false).to_string();
        let native_explain = native_explain_for_plan(planned_plan.as_ref());
        let mut query = self.ensure_query(query_id, now);
        query.physical_explain = Some(physical_explain);
        query.native_explain = native_explain;
        drop(query);
        self.enforce_bounds(now);
    }

    pub(super) fn capture_planning_failure(
        &self,
        query_id: &str,
        stage: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
    ) {
        let now = Instant::now();
        {
            let mut query = self.ensure_query(query_id, now);
            query.planning_failure = Some(CapturedPlanningFailure {
                stage: stage.into(),
                reason_code: reason_code.into(),
                message: message.into(),
            });
        }
        self.enforce_bounds(now);
    }

    pub(super) fn record_output_batch(&self, query_id: &str, batch: &RecordBatch) {
        let now = Instant::now();
        {
            let mut query = self.ensure_query(query_id, now);
            query.metrics.record_output_batch(batch);
        }
        self.enforce_bounds(now);
    }

    pub(super) fn finish_success(&self, query_id: &str) {
        let now = Instant::now();
        let mut query = self.ensure_query(query_id, now);
        if matches!(query.outcome, QueryOutcome::Running) {
            query.outcome = QueryOutcome::Succeeded;
            query.completed_at = Some(now);
        }
        drop(query);
        self.enforce_bounds(now);
    }

    pub(super) fn finish_failed(&self, query_id: &str, stage: &'static str, error: CapturedError) {
        let now = Instant::now();
        let mut query = self.ensure_query(query_id, now);
        if matches!(query.outcome, QueryOutcome::Running | QueryOutcome::Dropped) {
            query.outcome = QueryOutcome::Failed { stage, error };
            query.completed_at = Some(now);
        }
        drop(query);
        self.enforce_bounds(now);
    }

    pub(super) fn finish_status(&self, query_id: &str, stage: &'static str, status: &tonic::Status) {
        self.finish_failed(query_id, stage, CapturedError::from_status(status));
    }

    pub(super) fn seal(&self, query_id: &str) {
        let now = Instant::now();
        if let Some(mut query) = self.captures.get_mut(query_id) {
            query.sealed = true;
            query.completed_at.get_or_insert(now);
            if matches!(query.outcome, QueryOutcome::Running) {
                query.outcome = QueryOutcome::Dropped;
            }
            query.last_touched_at = now;
        }
        self.enforce_bounds(now);
    }

    pub(super) fn get(&self, query_id: &str) -> Option<CapturedQuery> {
        let now = Instant::now();
        self.enforce_bounds(now);
        let mut query = self.captures.get_mut(query_id)?;
        query.last_touched_at = now;
        Some(query.clone())
    }

    #[allow(dead_code)]
    pub(super) fn delete(&self, query_id: &str) -> bool {
        self.captures.remove(query_id).is_some()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.captures.len()
    }

    fn ensure_query(&self, query_id: &str, now: Instant) -> dashmap::mapref::one::RefMut<'_, String, CapturedQuery> {
        let mut query = self
            .captures
            .entry(query_id.to_string())
            .or_insert_with(|| CapturedQuery::new(query_id.to_string(), String::new(), now));
        query.last_touched_at = now;
        query
    }

    fn enforce_bounds(&self, now: Instant) {
        let expired = self
            .captures
            .iter()
            .filter_map(|entry| {
                let query = entry.value();
                query
                    .sealed
                    .then_some(query.completed_at)
                    .flatten()
                    .filter(|completed| now.saturating_duration_since(*completed) >= self.ttl)
                    .map(|_| entry.key().clone())
            })
            .collect::<Vec<_>>();
        for query_id in expired {
            self.captures.remove(&query_id);
        }

        while self.captures.len() > self.cap {
            let Some(victim) = self
                .captures
                .iter()
                .min_by_key(|entry| {
                    let query = entry.value();
                    (if query.sealed { 0_u8 } else { 1_u8 }, query.last_touched_at)
                })
                .map(|entry| entry.key().clone())
            else {
                break;
            };
            self.captures.remove(&victim);
        }
    }
}

impl NativeMetricsCaptureSink for QueryObservability {
    fn record_query_metrics(&self, query_id: &str, metrics: &QueryMetrics) {
        let now = Instant::now();
        {
            let mut query = self.ensure_query(query_id, now);
            query.metrics.query_metrics.push(metrics.clone());
        }
        self.enforce_bounds(now);
    }

    fn record_lakehouse_cache_fill_events(&self, query_id: &str, events: &[LakehouseCacheFillEventSnapshot]) {
        if events.is_empty() {
            return;
        }
        let now = Instant::now();
        {
            let mut query = self.ensure_query(query_id, now);
            query.metrics.lakehouse_cache_fill_events.extend_from_slice(events);
        }
        self.enforce_bounds(now);
    }
}

#[derive(Debug, Clone)]
pub(super) struct CapturedQuery {
    pub(super) query_id: String,
    sql: String,
    planning: Option<CapturedPlanning>,
    planning_failure: Option<CapturedPlanningFailure>,
    metrics: CapturedMetrics,
    outcome: QueryOutcome,
    physical_explain: Option<String>,
    native_explain: Option<String>,
    started_at: Instant,
    completed_at: Option<Instant>,
    last_touched_at: Instant,
    sealed: bool,
}

impl CapturedQuery {
    fn new(query_id: String, sql: String, now: Instant) -> Self {
        Self {
            query_id,
            sql,
            planning: None,
            planning_failure: None,
            metrics: CapturedMetrics::default(),
            outcome: QueryOutcome::Running,
            physical_explain: None,
            native_explain: None,
            started_at: now,
            completed_at: None,
            last_touched_at: now,
            sealed: false,
        }
    }

    pub(super) fn to_query_stats_payload(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "# datafusion_nexus_query_stats=1");
        let _ = writeln!(output, "# query_id={}", stable_value(&self.query_id));
        let _ = writeln!(output, "# outcome={}", self.outcome.as_str());
        let _ = writeln!(output, "# sealed={}", self.sealed);
        let _ = writeln!(output, "# elapsed_ms={}", self.elapsed_ms());
        if !self.sql.is_empty() {
            let _ = writeln!(output, "# sql={}", stable_value(&self.sql));
        }

        if let Some(planning) = &self.planning {
            output.push_str("\n[planning]\n");
            output.push_str(
                &NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
                    self.query_id.clone(),
                    Arc::clone(&planning.planned_plan),
                    planning.diagnostics.clone(),
                )])
                .to_tsv(),
            );

            output.push_str("\n[execution]\n");
            output.push_str(
                &NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
                    self.query_id.clone(),
                    Arc::clone(&planning.planned_plan),
                    planning.diagnostics.clone(),
                    self.runtime_for_planning(planning),
                )])
                .to_tsv(),
            );
        } else if let Some(failure) = &self.planning_failure {
            output.push_str("\n[planning]\n");
            output.push_str(
                &NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::planning_failure(
                    self.query_id.clone(),
                    failure.stage.clone(),
                    failure.reason_code.clone(),
                    failure.message.clone(),
                )])
                .to_tsv(),
            );

            output.push_str("\n[execution]\n");
            output.push_str(
                &NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::planning_failure(
                    self.query_id.clone(),
                    failure.stage.clone(),
                    failure.reason_code.clone(),
                    failure.message.clone(),
                )])
                .to_tsv(),
            );
        }

        if let Some(error) = self.outcome.error() {
            output.push_str("\n[structured_error]\n");
            error.write_to(&mut output);
        }

        if let Some(physical_explain) = &self.physical_explain {
            output.push_str("\n[physical_plan]\n");
            output.push_str(physical_explain);
            if !physical_explain.ends_with('\n') {
                output.push('\n');
            }
        }

        if let Some(native_explain) = &self.native_explain {
            output.push_str("\n[native_explain]\n");
            output.push_str(native_explain);
            if !native_explain.ends_with('\n') {
                output.push('\n');
            }
        }

        output
    }

    fn runtime_for_planning(&self, planning: &CapturedPlanning) -> NativeDataFusionRuntime {
        match &self.outcome {
            QueryOutcome::Succeeded => {
                let captured_metrics = self.metrics.to_metrics_set();
                let plan_metrics = planning
                    .planned_plan
                    .metrics()
                    .filter(|metrics| metrics.iter().next().is_some())
                    .unwrap_or_else(|| captured_metrics.clone());
                let metrics = if self.metrics.has_native_runtime_metrics() { captured_metrics } else { plan_metrics };
                NativeDataFusionRuntime::Succeeded {
                    output_batches: self.metrics.output_batches.unwrap_or(0),
                    output_rows: self.metrics.output_rows.unwrap_or(0),
                    output_bytes: self.metrics.output_bytes.unwrap_or(0),
                    metrics,
                }
            },
            QueryOutcome::Failed { error, .. } => {
                if self.native_runtime_started_for_failure(error) {
                    NativeDataFusionRuntime::Failed { message: error.diagnostic.clone() }
                } else {
                    NativeDataFusionRuntime::NotStarted
                }
            },
            QueryOutcome::Running | QueryOutcome::Dropped => NativeDataFusionRuntime::NotStarted,
        }
    }

    fn native_runtime_started_for_failure(&self, error: &CapturedError) -> bool {
        self.metrics.has_native_runtime_metrics()
            || error.fact_value("native_datafusion_stage").is_some_and(|stage| stage == "runtime_execution")
    }

    fn elapsed_ms(&self) -> u128 {
        self.completed_at.unwrap_or_else(Instant::now).saturating_duration_since(self.started_at).as_millis()
    }
}

#[derive(Debug, Clone)]
struct CapturedPlanning {
    diagnostics: NativePlanningDiagnostics,
    planned_plan: Arc<dyn ExecutionPlan>,
}

#[derive(Debug, Clone)]
struct CapturedPlanningFailure {
    stage: String,
    reason_code: String,
    message: String,
}

#[derive(Debug, Clone, Default)]
struct CapturedMetrics {
    query_metrics: Vec<QueryMetrics>,
    lakehouse_cache_fill_events: Vec<LakehouseCacheFillEventSnapshot>,
    output_batches: Option<usize>,
    output_rows: Option<usize>,
    output_bytes: Option<usize>,
}

impl CapturedMetrics {
    fn record_output_batch(&mut self, batch: &RecordBatch) {
        *self.output_batches.get_or_insert(0) += 1;
        *self.output_rows.get_or_insert(0) += batch.num_rows();
        *self.output_bytes.get_or_insert(0) += batch.get_array_memory_size();
    }

    fn has_native_runtime_metrics(&self) -> bool {
        !self.query_metrics.is_empty() || !self.lakehouse_cache_fill_events.is_empty()
    }

    fn to_metrics_set(&self) -> MetricsSet {
        let metrics_set = datafusion::physical_plan::metrics::ExecutionPlanMetricsSet::new();
        let metrics = NativeCudfMetrics::new(&metrics_set, 0);
        for query_metrics in &self.query_metrics {
            metrics.record_query_metrics(query_metrics);
        }
        metrics.record_lakehouse_cache_fill_events(&self.lakehouse_cache_fill_events);
        metrics_set.clone_inner()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum QueryOutcome {
    Running,
    Succeeded,
    Failed { stage: &'static str, error: CapturedError },
    Dropped,
}

impl QueryOutcome {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed { .. } => "failed",
            Self::Dropped => "dropped",
        }
    }

    fn error(&self) -> Option<&CapturedError> {
        match self {
            Self::Failed { error, .. } => Some(error),
            Self::Running | Self::Succeeded | Self::Dropped => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CapturedError {
    pub(super) code: String,
    pub(super) kind: String,
    pub(super) status: String,
    pub(super) operation: String,
    pub(super) message: String,
    pub(super) facts: Vec<(String, String)>,
    pub(super) diagnostic: String,
}

impl CapturedError {
    pub(super) fn from_adapter_error(error: &Error) -> Self {
        Self {
            code: error.code().as_str().to_string(),
            kind: error.kind().as_str().to_string(),
            status: error.status().as_str().to_string(),
            operation: error.operation().to_string(),
            message: error.message().to_string(),
            facts: error.facts().iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
            diagnostic: error.to_string(),
        }
    }

    fn from_status(status: &tonic::Status) -> Self {
        let metadata = status.metadata();
        let code = metadata_value(metadata, super::error::ERROR_CODE_METADATA_KEY).unwrap_or("unknown");
        let kind = metadata_value(metadata, super::error::ERROR_KIND_METADATA_KEY).unwrap_or("unknown");
        let error_status = metadata_value(metadata, super::error::ERROR_STATUS_METADATA_KEY).unwrap_or("unknown");
        Self {
            code: code.to_string(),
            kind: kind.to_string(),
            status: error_status.to_string(),
            operation: "unknown".to_string(),
            message: status.message().to_string(),
            facts: Vec::new(),
            diagnostic: format!("grpc_code={:?} message={}", status.code(), status.message()),
        }
    }

    fn write_to(&self, output: &mut String) {
        let _ = writeln!(output, "code={}", stable_value(&self.code));
        let _ = writeln!(output, "kind={}", stable_value(&self.kind));
        let _ = writeln!(output, "status={}", stable_value(&self.status));
        let _ = writeln!(output, "operation={}", stable_value(&self.operation));
        let _ = writeln!(output, "message={}", stable_value(&self.message));
        let _ = writeln!(output, "diagnostic={}", stable_value(&self.diagnostic));
        for (key, value) in &self.facts {
            let _ = writeln!(output, "fact.{}={}", stable_value(key), stable_value(value));
        }
    }

    fn fact_value(&self, key: &str) -> Option<&str> {
        self.facts.iter().find_map(|(fact_key, value)| (fact_key == key).then_some(value.as_str()))
    }
}

pub(super) fn native_explain_for_plan(plan: &dyn ExecutionPlan) -> Option<String> {
    contains_native_wrapper(plan).then(|| displayable(plan).indent(true).to_string())
}

fn contains_native_wrapper(plan: &dyn ExecutionPlan) -> bool {
    if plan.downcast_ref::<crate::NativeCudfExec>().is_some()
        || plan.downcast_ref::<crate::NativeCudfComposedExec>().is_some()
        || plan.downcast_ref::<crate::NativeCudfGpuProducerExec>().is_some()
    {
        return true;
    }
    plan.children().iter().any(|child| contains_native_wrapper(child.as_ref()))
}

fn metadata_value<'a>(metadata: &'a tonic::metadata::MetadataMap, key: &'static str) -> Option<&'a str> {
    metadata.get(key).and_then(|value| value.to_str().ok())
}

fn stable_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::empty::EmptyExec;

    use super::*;
    use crate::report::{GpuCandidate, PlanningReport};

    #[test]
    fn sealed_query_is_retained_until_ttl_or_cap_eviction() {
        let store = QueryObservability::new(8, Duration::from_secs(60));
        store.start_query("q1", "select 1");
        store.finish_success("q1");
        store.seal("q1");

        let captured = store.get("q1").expect("sealed query must be retained");
        assert_eq!(captured.query_id, "q1");
        assert_eq!(captured.outcome, QueryOutcome::Succeeded);
    }

    #[test]
    fn ttl_sweep_removes_sealed_queries() {
        let store = QueryObservability::new(8, Duration::ZERO);
        store.start_query("q1", "select 1");
        store.finish_success("q1");
        store.seal("q1");
        store.start_query("q2", "select 2");

        assert!(store.get("q1").is_none());
        assert!(store.get("q2").is_some());
    }

    #[test]
    fn cap_eviction_removes_oldest_query() {
        let store = QueryObservability::new(1, Duration::from_secs(60));
        store.start_query("q1", "select 1");
        store.finish_success("q1");
        store.seal("q1");
        store.start_query("q2", "select 2");

        assert!(store.get("q1").is_none());
        assert!(store.get("q2").is_some());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn concurrent_queries_do_not_clobber_each_other() {
        let store = Arc::new(QueryObservability::new(64, Duration::from_secs(60)));
        let handles = (0..16)
            .map(|index| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    let query_id = format!("q{index}");
                    let sql = format!("select {index}");
                    store.start_query(query_id.clone(), sql);
                    store.finish_success(&query_id);
                    store.seal(&query_id);
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.join().expect("query thread must not panic");
        }

        for index in 0..16 {
            let query_id = format!("q{index}");
            let captured = store.get(&query_id).expect("query capture must be retained");
            assert_eq!(captured.query_id, query_id);
            assert_eq!(captured.sql, format!("select {index}"));
            assert_eq!(captured.outcome, QueryOutcome::Succeeded);
        }
    }

    #[test]
    fn late_failure_does_not_override_success() {
        let store = QueryObservability::new(8, Duration::from_secs(60));
        store.start_query("q1", "select 1");
        store.finish_success("q1");
        store.finish_failed(
            "q1",
            "cancel_query",
            CapturedError {
                code: "cancelled".to_string(),
                kind: "cancelled".to_string(),
                status: "transient".to_string(),
                operation: "datafusion_nexus.server.flight.cancel_query".to_string(),
                message: "statement was cancelled".to_string(),
                facts: Vec::new(),
                diagnostic: "cancelled".to_string(),
            },
        );

        let captured = store.get("q1").expect("query capture must be retained");
        assert_eq!(captured.outcome, QueryOutcome::Succeeded);
    }

    #[test]
    fn late_success_does_not_override_failure_or_terminal_timestamp() {
        let store = QueryObservability::new(8, Duration::from_secs(60));
        store.start_query("q1", "select 1");
        store.finish_failed("q1", "stream", captured_error_with_facts([]));
        let failed = store.get("q1").expect("query capture must be retained");
        let failed_completed_at = failed.completed_at;

        store.finish_success("q1");

        let captured = store.get("q1").expect("query capture must be retained");
        assert!(matches!(captured.outcome, QueryOutcome::Failed { .. }));
        assert_eq!(captured.completed_at, failed_completed_at);
    }

    #[test]
    fn runtime_context_failure_after_rewrite_does_not_report_native_runtime_started() {
        let store = QueryObservability::new(8, Duration::from_secs(60));
        store.start_query("q1", "select 1");
        let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
        store.capture_planning(
            "q1",
            Some(accepted_native_diagnostics()),
            Arc::new(EmptyExec::new(schema)) as Arc<dyn ExecutionPlan>,
        );
        store.finish_failed(
            "q1",
            "execute_stream",
            captured_error_with_facts([("native_datafusion_stage", "runtime_context")]),
        );
        store.seal("q1");

        let payload = store.get("q1").expect("query capture must be retained").to_query_stats_payload();
        let (header, row) = query_stats_tsv_single_row(&payload, "execution");

        assert_eq!(tsv_value(&header, &row, "native_runtime_started"), "false", "{payload}");
        assert_eq!(tsv_value(&header, &row, "runtime_outcome"), "not_started", "{payload}");
    }

    #[test]
    fn runtime_execution_failure_after_rewrite_reports_native_runtime_started() {
        let store = QueryObservability::new(8, Duration::from_secs(60));
        store.start_query("q1", "select 1");
        let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
        store.capture_planning(
            "q1",
            Some(accepted_native_diagnostics()),
            Arc::new(EmptyExec::new(schema)) as Arc<dyn ExecutionPlan>,
        );
        store.finish_failed(
            "q1",
            "stream",
            captured_error_with_facts([("native_datafusion_stage", "runtime_execution")]),
        );
        store.seal("q1");

        let payload = store.get("q1").expect("query capture must be retained").to_query_stats_payload();
        let (header, row) = query_stats_tsv_single_row(&payload, "execution");

        assert_eq!(tsv_value(&header, &row, "native_runtime_started"), "true", "{payload}");
        assert_eq!(tsv_value(&header, &row, "runtime_outcome"), "failed", "{payload}");
    }

    #[test]
    fn query_stats_payload_includes_metrics_captured_through_native_metrics_bridge() {
        let store = Arc::new(QueryObservability::new(8, Duration::from_secs(60)));
        store.start_query("q1", "select 1");
        let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
        store.capture_planning(
            "q1",
            Some(accepted_native_diagnostics()),
            Arc::new(EmptyExec::new(schema)) as Arc<dyn ExecutionPlan>,
        );

        let mut metrics = QueryMetrics::default();
        metrics.boundary.record_host_to_device_upload(7, 1024);
        metrics.boundary.record_device_to_host_materialization(3, 512);
        metrics.resident_dataframe_peak_bytes = Some(4096);
        metrics.allocator_high_water_bytes = Some(8192);
        metrics.reservation_requested_bytes = 16_384;
        metrics.reservation_granted_bytes = 8_192;
        metrics.reservation_live_peak_bytes = 4_096;

        let sink: Arc<dyn NativeMetricsCaptureSink> = store.clone();
        let capture = crate::native_exec::NativeMetricsCapture::new("q1", sink);
        let metrics_set = datafusion::physical_plan::metrics::ExecutionPlanMetricsSet::new();
        let native_metrics = NativeCudfMetrics::new_with_capture(&metrics_set, 0, Some(capture));
        native_metrics.record_query_metrics(&metrics);

        store.finish_success("q1");
        store.seal("q1");

        let payload = store.get("q1").expect("query capture must be retained").to_query_stats_payload();
        let (header, row) = query_stats_tsv_single_row(&payload, "execution");

        assert_eq!(tsv_value(&header, &row, "native_runtime_started"), "true", "{payload}");
        assert_eq!(tsv_value(&header, &row, "runtime_outcome"), "succeeded", "{payload}");
        assert_eq!(tsv_value(&header, &row, "host_to_device_uploads"), "1", "{payload}");
        assert_eq!(tsv_value(&header, &row, "host_to_device_uploaded_rows"), "7", "{payload}");
        assert_eq!(tsv_value(&header, &row, "host_to_device_uploaded_bytes"), "1024", "{payload}");
        assert_eq!(tsv_value(&header, &row, "device_to_host_materializations"), "1", "{payload}");
        assert_eq!(tsv_value(&header, &row, "device_to_host_materialized_rows"), "3", "{payload}");
        assert_eq!(tsv_value(&header, &row, "device_to_host_materialized_bytes"), "512", "{payload}");
        assert_eq!(tsv_value(&header, &row, "resident_dataframe_peak_bytes"), "4096", "{payload}");
        assert_eq!(tsv_value(&header, &row, "allocator_high_water_bytes"), "8192", "{payload}");
        assert_eq!(tsv_value(&header, &row, "reservation_requested_bytes"), "16384", "{payload}");
        assert_eq!(tsv_value(&header, &row, "reservation_granted_bytes"), "8192", "{payload}");
        assert_eq!(tsv_value(&header, &row, "reservation_live_peak_bytes"), "4096", "{payload}");
    }

    fn accepted_native_diagnostics() -> NativePlanningDiagnostics {
        let mut report = PlanningReport::new();
        report.push_rewritten(GpuCandidate::new("test_native_plan", 1));
        NativePlanningDiagnostics {
            input_plan_root: "EmptyExec".to_string(),
            input_plan_shape: "EmptyExec".to_string(),
            input_join_diagnostics: Vec::new(),
            report,
            native_optimizer: Default::default(),
            source_diagnostics: Vec::new(),
            strict_bounded_contract: Default::default(),
            adaptive_pipeline: Default::default(),
        }
    }

    fn captured_error_with_facts(facts: impl IntoIterator<Item = (&'static str, &'static str)>) -> CapturedError {
        CapturedError {
            code: "native_execution".to_string(),
            kind: "execution".to_string(),
            status: "transient".to_string(),
            operation: "datafusion_nexus.test".to_string(),
            message: "native execution failed".to_string(),
            facts: facts.into_iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
            diagnostic: "native execution failed".to_string(),
        }
    }

    fn query_stats_tsv_single_row<'a>(payload: &'a str, section: &str) -> (Vec<&'a str>, Vec<&'a str>) {
        let marker = format!("[{section}]");
        let mut lines = payload
            .lines()
            .skip_while(|line| *line != marker)
            .skip(1)
            .filter(|line| !line.is_empty() && !line.starts_with('#'));
        let header =
            lines.next().unwrap_or_else(|| panic!("missing [{section}] header in query_stats payload:\n{payload}"));
        let row = lines.next().unwrap_or_else(|| panic!("missing [{section}] row in query_stats payload:\n{payload}"));
        (header.split('\t').collect(), row.split('\t').collect())
    }

    fn tsv_value<'a>(header: &[&str], row: &'a [&str], column: &str) -> &'a str {
        let index = header
            .iter()
            .position(|name| *name == column)
            .unwrap_or_else(|| panic!("missing TSV column {column}; header={header:?}; row={row:?}"));
        row.get(index).copied().unwrap_or_else(|| panic!("missing TSV value for column {column}; row={row:?}"))
    }
}
