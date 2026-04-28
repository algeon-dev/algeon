use std::{
    collections::{BTreeMap, HashSet},
    fmt::{self, Write},
    sync::Arc,
    time::Instant,
};

use datafusion::{
    common::DataFusionError,
    physical_plan::{
        ExecutionPlan, displayable,
        metrics::{Metric, MetricsSet},
    },
};

use crate::{
    error::{Error, ErrorCode, ErrorSourceDiagnostics, Facts},
    final_plan_observation::observe_final_plan,
    native_exec::{
        NexusNativeComposedExec, NexusNativeComposedGpuProducerExec, NexusNativeExec, NexusNativeGpuProducerExec,
    },
    native_report::{
        NativeDataFusionExecutionMetrics, NativeDataFusionExecutionReport, NativeDataFusionExecutionTsvTemplate,
        NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningReport, NativeDataFusionPlanningWorkloadItem,
        NativeDataFusionRuntime,
    },
    planner::NativePlanningDiagnostics,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum CaptureLossKind {
    RetainedByteLimit,
    ArtifactByteLimit,
    CounterOverflow,
}

impl CaptureLossKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RetainedByteLimit => "retained_byte_limit",
            Self::ArtifactByteLimit => "artifact_byte_limit",
            Self::CounterOverflow => "counter_overflow",
        }
    }
}

/// One query's deterministic server-capture envelope.
///
/// All retained free-form strings enter through this type. Replacements first
/// release the old logical charge, so the counters describe live retained
/// capture rather than cumulative traffic.
#[derive(Debug, Clone)]
struct CaptureBudget {
    max_bytes: u64,
    retained_bytes: u64,
    dropped_bytes: u64,
    dropped_fields: u64,
    losses: BTreeMap<CaptureLossKind, u64>,
    counter_overflow: bool,
}

impl CaptureBudget {
    fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            retained_bytes: 0,
            dropped_bytes: 0,
            dropped_fields: 0,
            losses: BTreeMap::new(),
            counter_overflow: false,
        }
    }

    fn remaining(&self) -> usize {
        usize::try_from(self.max_bytes.saturating_sub(self.retained_bytes)).unwrap_or(usize::MAX)
    }

    fn replace_text(&mut self, slot: &mut Arc<str>, value: impl AsRef<str>) {
        self.release(slot.len());
        *slot = self.retain_text(value.as_ref());
    }

    fn replace_optional_text(&mut self, slot: &mut Option<Arc<str>>, value: Option<&str>) {
        if let Some(previous) = slot.take() {
            self.release(previous.len());
        }
        *slot = value.map(|value| self.retain_text(value));
    }

    fn retain_text(&mut self, value: &str) -> Arc<str> {
        let keep = utf8_prefix_len(value, self.remaining());
        let retained = &value[..keep];
        self.retained_bytes = self.retained_bytes.checked_add(keep as u64).unwrap_or_else(|| {
            self.mark_overflow();
            u64::MAX
        });
        if keep < value.len() {
            self.record_loss(CaptureLossKind::RetainedByteLimit, value.len() - keep);
        }
        Arc::from(retained)
    }

    fn retain_string(&mut self, value: &str) -> String {
        let keep = utf8_prefix_len(value, self.remaining());
        let retained = value[..keep].to_string();
        self.retained_bytes = self.retained_bytes.checked_add(keep as u64).unwrap_or_else(|| {
            self.mark_overflow();
            u64::MAX
        });
        if keep < value.len() {
            self.record_loss(CaptureLossKind::RetainedByteLimit, value.len() - keep);
        }
        retained
    }

    fn retain_rendered(&mut self, rendered: BoundedText) -> Arc<str> {
        let text = rendered.text;
        let retained = self.retain_text(&text);
        if rendered.truncated {
            self.record_loss(CaptureLossKind::RetainedByteLimit, rendered.discarded_bytes.max(1));
        }
        retained
    }

    fn release(&mut self, bytes: usize) {
        self.retained_bytes = self.retained_bytes.checked_sub(bytes as u64).unwrap_or_else(|| {
            self.mark_overflow();
            0
        });
    }

    fn try_retain_capacity(&mut self, bytes: u64) -> bool {
        let Some(next) = self.retained_bytes.checked_add(bytes) else {
            self.mark_overflow();
            return false;
        };
        if next > self.max_bytes {
            self.record_loss(CaptureLossKind::RetainedByteLimit, usize::try_from(bytes).unwrap_or(usize::MAX));
            return false;
        }
        self.retained_bytes = next;
        true
    }

    fn release_capacity(&mut self, bytes: u64) {
        self.retained_bytes = self.retained_bytes.checked_sub(bytes).unwrap_or_else(|| {
            self.mark_overflow();
            0
        });
    }

    fn record_loss(&mut self, kind: CaptureLossKind, bytes: usize) {
        self.dropped_bytes = self.dropped_bytes.checked_add(bytes as u64).unwrap_or_else(|| {
            self.mark_overflow();
            u64::MAX
        });
        self.dropped_fields = self.dropped_fields.checked_add(1).unwrap_or_else(|| {
            self.mark_overflow();
            u64::MAX
        });
        let count = self.losses.get(&kind).copied().unwrap_or(0);
        let next = count.checked_add(1).unwrap_or_else(|| {
            self.mark_overflow();
            u64::MAX
        });
        self.losses.insert(kind, next);
    }

    fn mark_loss(&mut self, kind: CaptureLossKind) {
        self.record_loss(kind, 0);
    }

    fn mark_overflow(&mut self) {
        self.counter_overflow = true;
        self.losses.insert(CaptureLossKind::CounterOverflow, u64::MAX);
    }

    fn incomplete(&self) -> bool {
        !self.losses.is_empty() || self.counter_overflow
    }
}

#[derive(Debug)]
struct BoundedText {
    text: String,
    limit: usize,
    truncated: bool,
    discarded_bytes: usize,
}

struct FormatLimitProbe {
    remaining: usize,
    truncated: bool,
}

impl fmt::Write for FormatLimitProbe {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        if value.len() > self.remaining {
            self.truncated = true;
            self.remaining = 0;
            return Err(fmt::Error);
        }
        self.remaining -= value.len();
        Ok(())
    }
}

impl BoundedText {
    fn new(limit: usize) -> Self {
        Self { text: String::with_capacity(limit.min(64 * 1024)), limit, truncated: false, discarded_bytes: 0 }
    }

    fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.text.len())
    }

    fn into_bytes(self) -> Vec<u8> {
        self.text.into_bytes()
    }

    fn write_atomic(&mut self, value: &str) -> bool {
        if value.len() > self.remaining() {
            self.truncated = true;
            self.discarded_bytes = self.discarded_bytes.saturating_add(value.len());
            return false;
        }
        self.text.push_str(value);
        true
    }
}

impl fmt::Write for BoundedText {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let keep = utf8_prefix_len(value, self.remaining());
        self.text.push_str(&value[..keep]);
        if keep < value.len() {
            self.truncated = true;
            self.discarded_bytes = self.discarded_bytes.saturating_add(value.len() - keep);
            return Err(fmt::Error);
        }
        Ok(())
    }
}

fn write_tsv_atomic(output: &mut BoundedText, tsv: &str) -> bool {
    for line in tsv.split_inclusive('\n') {
        if !output.write_atomic(line) {
            return false;
        }
    }
    true
}

fn utf8_prefix_len(value: &str, max_bytes: usize) -> usize {
    let mut keep = value.len().min(max_bytes);
    while keep > 0 && !value.is_char_boundary(keep) {
        keep -= 1;
    }
    keep
}

pub(super) struct StableValue<'a>(pub(super) &'a str);

impl fmt::Display for StableValue<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for segment in self.0.split_inclusive(['\t', '\n', '\r']) {
            let (text, replaced) = segment
                .char_indices()
                .next_back()
                .filter(|(_, ch)| matches!(ch, '\t' | '\n' | '\r'))
                .map_or((segment, false), |(index, _)| (&segment[..index], true));
            formatter.write_str(text)?;
            if replaced {
                formatter.write_str(" ")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(super) struct CapturedQuery {
    pub(super) query_id: String,
    sql: Arc<str>,
    correlation_id: Option<Arc<str>>,
    planning: Option<Arc<CapturedPlanning>>,
    planning_failure: Option<CapturedPlanningFailure>,
    metrics: CapturedMetrics,
    outcome: QueryOutcome,
    physical_explain: Option<Arc<str>>,
    native_explain: Option<Arc<str>>,
    started_at: Instant,
    completed_at: Option<Instant>,
    last_touched_at: Instant,
    sealed: bool,
    capture: CaptureBudget,
    max_metric_records: usize,
}

impl CapturedQuery {
    pub(super) fn new(
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        now: Instant,
        max_capture_bytes: u64,
        max_metric_records: u64,
    ) -> Self {
        let mut capture = CaptureBudget::new(max_capture_bytes);
        let sql = capture.retain_text(&sql);
        let correlation_id = correlation_id.as_deref().map(|value| capture.retain_text(value));
        Self {
            query_id,
            sql,
            correlation_id,
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
            capture,
            max_metric_records: usize::try_from(max_metric_records).unwrap_or(usize::MAX),
        }
    }

    pub(super) fn set_sql(&mut self, sql: String) {
        self.capture.replace_text(&mut self.sql, sql);
        self.last_touched_at = Instant::now();
    }

    pub(super) const fn max_metric_records(&self) -> usize {
        self.max_metric_records
    }

    pub(super) fn record_correlation_id(&mut self, correlation_id: Option<&str>) {
        if let Some(correlation_id) = correlation_id.map(str::trim).filter(|value| !value.is_empty())
            && self.correlation_id.is_none()
        {
            self.capture.replace_optional_text(&mut self.correlation_id, Some(correlation_id));
        }
        self.last_touched_at = Instant::now();
    }

    pub(super) fn render_planning(
        query_id: &str,
        diagnostics: Option<NativePlanningDiagnostics>,
        planned_plan: Arc<dyn ExecutionPlan>,
        max_bytes: usize,
        max_metric_records: usize,
    ) -> CapturedPlanningRender {
        let physical_limit = max_bytes / 2;
        let mut physical_explain = BoundedText::new(physical_limit);
        let _ = write!(physical_explain, "{}", displayable(planned_plan.as_ref()).indent(false));
        let native_limit = max_bytes.saturating_sub(physical_limit);
        let mut native_explain = contains_native_wrapper(planned_plan.as_ref()).then(|| BoundedText::new(native_limit));
        if let Some(native_explain) = &mut native_explain {
            let _ = write!(native_explain, "{}", displayable(planned_plan.as_ref()).indent(true));
        }
        let mut diagnostics_probe = FormatLimitProbe { remaining: max_bytes, truncated: false };
        if let Some(diagnostics) = &diagnostics {
            let _ = write!(diagnostics_probe, "{diagnostics:?}");
        }
        let candidate_count = diagnostics.as_ref().map_or(0, |diagnostics| diagnostics.report.outcomes().len());
        let final_plan_has_gpu_runtime = observe_final_plan(planned_plan.as_ref()).has_gpu_runtime();
        let planning_truncated = diagnostics_probe.truncated
            || physical_explain.truncated
            || native_explain.as_ref().is_some_and(|text| text.truncated)
            || candidate_count > max_metric_records.max(1);
        let planning = diagnostics.filter(|_| !planning_truncated).map(|diagnostics| {
            let diagnostics = Arc::new(diagnostics);
            let planning_report =
                NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new_shared(
                    query_id.to_string(),
                    Arc::clone(&planned_plan),
                    Arc::clone(&diagnostics),
                )]);
            let execution_report =
                NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new_shared(
                    query_id.to_string(),
                    Arc::clone(&planned_plan),
                    Arc::clone(&diagnostics),
                    NativeDataFusionRuntime::NotStarted,
                )]);
            let candidate_count = candidate_count.max(1);
            let metrics = project_native_execution_metrics(
                &planned_plan,
                max_metric_records / candidate_count,
                max_bytes / candidate_count,
            );
            CapturedPlanning::new(
                CapturedTsvReport::from_tsv(planning_report.to_tsv()),
                NativeDataFusionExecutionTsvTemplate::from_report(execution_report),
                metrics,
                final_plan_has_gpu_runtime,
            )
        });
        CapturedPlanningRender { planning, physical_explain, native_explain, diagnostics_truncated: planning_truncated }
    }

    pub(super) fn install_planning(&mut self, rendered: CapturedPlanningRender) {
        if let Some(previous) = self.planning.take() {
            self.capture.release_capacity(previous.retained_capacity_bytes);
        }
        self.planning = rendered.planning.and_then(|planning| {
            self.capture.try_retain_capacity(planning.retained_capacity_bytes).then(|| Arc::new(planning))
        });
        self.planning_failure = None;
        if let Some(previous) = self.physical_explain.take() {
            self.capture.release(previous.len());
        }
        if let Some(previous) = self.native_explain.take() {
            self.capture.release(previous.len());
        }
        self.physical_explain = Some(self.capture.retain_rendered(rendered.physical_explain));
        self.native_explain = rendered.native_explain.map(|text| self.capture.retain_rendered(text));
        if rendered.diagnostics_truncated {
            self.capture.mark_loss(CaptureLossKind::RetainedByteLimit);
        }
        self.last_touched_at = Instant::now();
    }

    pub(super) fn capture_planning_failure(
        &mut self,
        stage: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
    ) {
        let stage = stage.into();
        let reason_code = reason_code.into();
        let message = message.into();
        if let Some(previous) = self.planning_failure.take() {
            self.capture.release(previous.stage.len());
            self.capture.release(previous.reason_code.len());
            self.capture.release(previous.message.len());
        }
        self.planning_failure = Some(CapturedPlanningFailure {
            stage: self.capture.retain_text(&stage),
            reason_code: self.capture.retain_text(&reason_code),
            message: self.capture.retain_text(&message),
        });
        self.last_touched_at = Instant::now();
    }

    pub(super) fn set_output_metrics(&mut self, batches: usize, rows: usize, bytes: usize, overflow: bool) {
        self.metrics.output_batches = Some(batches);
        self.metrics.output_rows = Some(rows);
        self.metrics.output_bytes = Some(bytes);
        if overflow {
            self.capture.mark_loss(CaptureLossKind::CounterOverflow);
        }
    }

    pub(super) fn finish_success(&mut self, now: Instant) {
        if matches!(self.outcome, QueryOutcome::Running) {
            self.outcome = QueryOutcome::Succeeded;
            self.completed_at = Some(now);
        }
    }

    pub(super) fn finish_failed(&mut self, stage: &'static str, error: Arc<CapturedError>, now: Instant) {
        if matches!(self.outcome, QueryOutcome::Running | QueryOutcome::Dropped) {
            self.outcome = QueryOutcome::Failed { stage, error };
            self.completed_at = Some(now);
        }
    }

    pub(super) fn bound_error(&mut self, error: &mut CapturedError) {
        error.apply_capture_budget(&mut self.capture);
    }

    pub(super) fn mark_counter_overflow(&mut self) {
        self.capture.mark_loss(CaptureLossKind::CounterOverflow);
    }

    pub(super) fn shutdown_snapshot(&self) -> ShutdownCaptureSnapshot {
        ShutdownCaptureSnapshot {
            query_id: self.query_id.clone(),
            correlation_id: self.correlation_id.clone(),
            started_at: self.started_at,
        }
    }

    pub(super) fn finish_dropped(&mut self, now: Instant) {
        if matches!(self.outcome, QueryOutcome::Running) {
            self.outcome = QueryOutcome::Dropped;
            self.completed_at = Some(now);
        }
    }

    pub(super) fn seal(&mut self, now: Instant) {
        self.sealed = true;
        self.completed_at.get_or_insert(now);
        if matches!(self.outcome, QueryOutcome::Running) {
            self.outcome = QueryOutcome::Dropped;
        }
        self.last_touched_at = now;
    }

    pub(super) fn to_query_stats_payload(&self, max_bytes: usize) -> QueryStatsPayload {
        let loss_metadata_bytes =
            self.capture.losses.keys().map(|kind| kind.as_str().len().saturating_add(48)).sum::<usize>();
        let trailer_reserve = loss_metadata_bytes.saturating_add(512).min(max_bytes);
        let mut output = BoundedText::new(max_bytes.saturating_sub(trailer_reserve));
        let _ = writeln!(output, "# datafusion_nexus_query_stats=1");
        let _ = writeln!(output, "# query_id={}", StableValue(&self.query_id));
        if let Some(correlation_id) = &self.correlation_id {
            let _ = writeln!(output, "# correlation_id={}", StableValue(correlation_id));
        }
        let _ = writeln!(output, "# outcome={}", self.outcome.as_str());
        let _ = writeln!(output, "# sealed={}", self.sealed);
        let _ = writeln!(output, "# elapsed_ms={}", self.elapsed_ms());
        if !self.sql.is_empty() {
            let _ = writeln!(output, "# sql={}", StableValue(&self.sql));
        }

        let mut metric_projection_truncated = false;
        if let Some(planning) = &self.planning {
            let planning_written =
                output.write_atomic("\n[planning]\n") && planning.planning_report.write_atomic(&mut output);
            metric_projection_truncated = planning.metrics.truncated;
            let runtime = self.runtime_for_planning(planning.metrics.metrics(), planning.final_plan_has_gpu_runtime);
            if planning_written && output.write_atomic("\n[execution]\n") {
                let _ = write_tsv_atomic(&mut output, &planning.execution_report.to_tsv(runtime));
            }
        } else if let Some(failure) = &self.planning_failure
            && output.write_atomic("\n[planning]\n")
        {
            let planning_report =
                NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::planning_failure(
                    self.query_id.clone(),
                    failure.stage.to_string(),
                    failure.reason_code.to_string(),
                    failure.message.to_string(),
                )]);
            if write_tsv_atomic(&mut output, &planning_report.to_tsv()) && output.write_atomic("\n[execution]\n") {
                let execution_report = NativeDataFusionExecutionReport::collect([
                    NativeDataFusionExecutionWorkloadItem::planning_failure(
                        self.query_id.clone(),
                        failure.stage.to_string(),
                        failure.reason_code.to_string(),
                        failure.message.to_string(),
                    ),
                ]);
                let _ = write_tsv_atomic(&mut output, &execution_report.to_tsv());
            }
        }

        if let Some(error) = self.outcome.error() {
            let _ = output.write_str("\n[structured_error]\n");
            error.write_to(&mut output);
        }

        if let Some(physical_explain) = &self.physical_explain {
            let _ = output.write_str("\n[physical_plan]\n");
            let _ = output.write_str(physical_explain);
            if !physical_explain.ends_with('\n') {
                let _ = output.write_char('\n');
            }
        }

        if let Some(native_explain) = &self.native_explain {
            let _ = output.write_str("\n[native_explain]\n");
            let _ = output.write_str(native_explain);
            if !native_explain.ends_with('\n') {
                let _ = output.write_char('\n');
            }
        }

        let artifact_truncated = output.truncated;
        let incomplete = self.capture.incomplete() || metric_projection_truncated || artifact_truncated;
        let mut trailer = String::new();
        let _ = writeln!(trailer, "# incomplete={incomplete}");
        let _ = writeln!(trailer, "# artifact_truncated={artifact_truncated}");
        let _ = writeln!(trailer, "# metric_projection_truncated={metric_projection_truncated}");
        for (kind, count) in &self.capture.losses {
            let _ = writeln!(trailer, "# capture_loss.{}={count}", kind.as_str());
        }
        if artifact_truncated {
            let _ = writeln!(trailer, "# capture_loss.{}=1", CaptureLossKind::ArtifactByteLimit.as_str());
        }
        let _ = writeln!(trailer, "# capture_retained_bytes={}", self.capture.retained_bytes);
        let _ = writeln!(trailer, "# capture_max_bytes={}", self.capture.max_bytes);
        let _ = writeln!(trailer, "# capture_dropped_bytes={}", self.capture.dropped_bytes);
        let _ = writeln!(trailer, "# capture_dropped_fields={}", self.capture.dropped_fields);
        output.text.push_str(&trailer[..utf8_prefix_len(&trailer, max_bytes.saturating_sub(output.text.len()))]);
        QueryStatsPayload { bytes: Arc::from(output.into_bytes()), incomplete }
    }

    fn runtime_for_planning(
        &self,
        metrics: Option<MetricsSet>,
        final_plan_has_gpu_runtime: bool,
    ) -> NativeDataFusionRuntime {
        match &self.outcome {
            QueryOutcome::Succeeded if final_plan_has_gpu_runtime => NativeDataFusionRuntime::Succeeded {
                output_batches: self.metrics.output_batches.unwrap_or(0),
                output_rows: self.metrics.output_rows.unwrap_or(0),
                output_bytes: self.metrics.output_bytes.unwrap_or(0),
                metrics: metrics.unwrap_or_default(),
            },
            QueryOutcome::Succeeded => NativeDataFusionRuntime::NotStarted,
            QueryOutcome::Failed { error, .. } => {
                let stage = error.fact_value("native_datafusion_stage");
                let has_runtime_metric_evidence = metrics.as_ref().is_some_and(|metrics| {
                    NativeDataFusionExecutionMetrics::from_metrics(metrics)
                        != NativeDataFusionExecutionMetrics::default()
                });
                if final_plan_has_gpu_runtime
                    && (stage == Some("runtime_execution")
                        || (stage != Some("runtime_context") && has_runtime_metric_evidence))
                {
                    NativeDataFusionRuntime::Failed { message: error.diagnostic.clone(), metrics }
                } else {
                    NativeDataFusionRuntime::NotStarted
                }
            },
            QueryOutcome::Running | QueryOutcome::Dropped => NativeDataFusionRuntime::NotStarted,
        }
    }

    fn elapsed_ms(&self) -> u128 {
        self.completed_at.unwrap_or_else(Instant::now).saturating_duration_since(self.started_at).as_millis()
    }
}

pub(super) struct QueryStatsPayload {
    pub(super) bytes: Arc<[u8]>,
    pub(super) incomplete: bool,
}

#[derive(Debug, Clone)]
pub(super) struct ShutdownCaptureSnapshot {
    query_id: String,
    correlation_id: Option<Arc<str>>,
    started_at: Instant,
}

impl ShutdownCaptureSnapshot {
    pub(super) fn to_query_stats_payload(
        &self,
        now: Instant,
        output_batches: usize,
        output_rows: usize,
        output_bytes: usize,
        max_bytes: usize,
    ) -> QueryStatsPayload {
        let mut output = BoundedText::new(max_bytes);
        let mut lines = vec![
            "# datafusion_nexus_query_stats=1\n".to_string(),
            format!("# query_id={}\n", StableValue(&self.query_id)),
        ];
        if let Some(correlation_id) = &self.correlation_id {
            lines.push(format!("# correlation_id={}\n", StableValue(correlation_id)));
        }
        lines.extend([
            "# outcome=dropped\n".to_string(),
            "# sealed=true\n".to_string(),
            "# incomplete=true\n".to_string(),
            "# unavailable_reason=shutdown_timeout\n".to_string(),
            format!("# elapsed_ms={}\n", now.saturating_duration_since(self.started_at).as_millis()),
            format!("# output_batches={output_batches}\n"),
            format!("# output_rows={output_rows}\n"),
            format!("# output_bytes={output_bytes}\n"),
        ]);
        for line in lines {
            if !output.write_atomic(&line) {
                break;
            }
        }
        QueryStatsPayload { bytes: Arc::from(output.into_bytes()), incomplete: true }
    }
}

pub(super) struct CapturedPlanningRender {
    planning: Option<CapturedPlanning>,
    physical_explain: BoundedText,
    native_explain: Option<BoundedText>,
    diagnostics_truncated: bool,
}

#[derive(Debug)]
struct CapturedPlanning {
    planning_report: CapturedTsvReport,
    execution_report: NativeDataFusionExecutionTsvTemplate,
    metrics: CapturedMetricHandles,
    final_plan_has_gpu_runtime: bool,
    retained_capacity_bytes: u64,
}

impl CapturedPlanning {
    fn new(
        planning_report: CapturedTsvReport,
        execution_report: NativeDataFusionExecutionTsvTemplate,
        metrics: MetricsProjection,
        final_plan_has_gpu_runtime: bool,
    ) -> Self {
        let retained_capacity_bytes = planning_report
            .retained_capacity_bytes
            .saturating_add(execution_report.retained_capacity_bytes())
            .saturating_add(metrics.handles.retained_capacity_bytes);
        Self {
            planning_report,
            execution_report,
            metrics: CapturedMetricHandles { handles: metrics.handles, truncated: metrics.truncated },
            final_plan_has_gpu_runtime,
            retained_capacity_bytes,
        }
    }
}

#[derive(Debug)]
struct CapturedTsvReport {
    lines: Box<[Box<str>]>,
    retained_capacity_bytes: u64,
}

impl CapturedTsvReport {
    fn from_tsv(tsv: String) -> Self {
        let lines = tsv
            .split_inclusive('\n')
            .map(str::to_owned)
            .map(String::into_boxed_str)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let retained_capacity_bytes = lines
            .iter()
            .fold(0u64, |total, line| total.saturating_add(u64::try_from(line.len()).unwrap_or(u64::MAX)))
            .saturating_add(
                u64::try_from(lines.len().saturating_mul(std::mem::size_of::<Box<str>>())).unwrap_or(u64::MAX),
            );
        Self { lines, retained_capacity_bytes }
    }

    fn write_atomic(&self, output: &mut BoundedText) -> bool {
        for line in &self.lines {
            if !output.write_atomic(line) {
                return false;
            }
        }
        true
    }
}

#[derive(Debug)]
struct CapturedMetricHandles {
    handles: MetricHandles,
    truncated: bool,
}

impl CapturedMetricHandles {
    fn metrics(&self) -> Option<MetricsSet> {
        if !self.handles.saw_native_wrapper {
            return None;
        }
        let mut metrics = MetricsSet::new();
        for metric in &self.handles.metrics {
            metrics.push(Arc::clone(metric));
        }
        Some(metrics)
    }
}

#[derive(Debug)]
struct MetricHandles {
    metrics: Box<[Arc<Metric>]>,
    saw_native_wrapper: bool,
    retained_capacity_bytes: u64,
}

#[derive(Debug, Clone)]
struct CapturedPlanningFailure {
    stage: Arc<str>,
    reason_code: Arc<str>,
    message: Arc<str>,
}

#[derive(Debug, Clone, Default)]
struct CapturedMetrics {
    output_batches: Option<usize>,
    output_rows: Option<usize>,
    output_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum QueryOutcome {
    Running,
    Succeeded,
    Failed { stage: &'static str, error: Arc<CapturedError> },
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
            Self::Failed { error, .. } => Some(error.as_ref()),
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
    pub(super) source_chain: Option<String>,
    pub(super) root_cause: Option<String>,
}

impl CapturedError {
    fn apply_capture_budget(&mut self, budget: &mut CaptureBudget) {
        fn bound(value: &mut String, budget: &mut CaptureBudget) {
            let original = std::mem::take(value);
            *value = budget.retain_string(&original);
        }

        bound(&mut self.code, budget);
        bound(&mut self.kind, budget);
        bound(&mut self.status, budget);
        bound(&mut self.operation, budget);
        bound(&mut self.message, budget);
        bound(&mut self.diagnostic, budget);
        if let Some(value) = &mut self.source_chain {
            bound(value, budget);
        }
        if let Some(value) = &mut self.root_cause {
            bound(value, budget);
        }
        let max_facts = usize::try_from(budget.max_bytes / 16).unwrap_or(usize::MAX).max(1);
        if self.facts.len() > max_facts {
            let dropped = self.facts.len() - max_facts;
            self.facts.truncate(max_facts);
            budget.record_loss(CaptureLossKind::RetainedByteLimit, dropped);
        }
        for (key, value) in &mut self.facts {
            bound(key, budget);
            bound(value, budget);
        }
    }

    pub(super) fn from_adapter_error(error: &Error) -> Self {
        let source_diagnostics = ErrorSourceDiagnostics::from_source(error.source_error());
        let safe_facts = error.facts().log_safe_projection();
        Self {
            code: error.code().as_str().to_string(),
            kind: error.kind().as_str().to_string(),
            status: error.status().as_str().to_string(),
            operation: error.operation().to_string(),
            message: error.message().to_string(),
            facts: safe_facts.retained_entries(),
            diagnostic: source_diagnostics.chain().unwrap_or_else(|| error.message()).to_string(),
            source_chain: source_diagnostics.chain().map(str::to_string),
            root_cause: source_diagnostics.root_cause().map(str::to_string),
        }
    }

    pub(super) fn from_datafusion_error(
        error: &DataFusionError,
        code: ErrorCode,
        operation: &'static str,
        message: String,
        facts: Facts,
    ) -> Self {
        let source_diagnostics = ErrorSourceDiagnostics::from_error(error);
        let diagnostic = source_diagnostics.chain().map(str::to_string).unwrap_or_else(|| message.clone());
        let safe_facts = facts.log_safe_projection();
        Self {
            code: code.as_str().to_string(),
            kind: code.kind().as_str().to_string(),
            status: code.status().as_str().to_string(),
            operation: operation.to_string(),
            diagnostic,
            message,
            facts: safe_facts.retained_entries(),
            source_chain: source_diagnostics.chain().map(str::to_string),
            root_cause: source_diagnostics.root_cause().map(str::to_string),
        }
    }

    pub(super) fn from_status(status: &tonic::Status) -> Self {
        let metadata = status.metadata();
        let code = metadata_value(metadata, super::error::ERROR_CODE_METADATA_KEY).unwrap_or("unknown");
        let kind = metadata_value(metadata, super::error::ERROR_KIND_METADATA_KEY).unwrap_or("unknown");
        let error_status = metadata_value(metadata, super::error::ERROR_STATUS_METADATA_KEY).unwrap_or("unknown");
        let operation = metadata_value(metadata, super::error::ERROR_OPERATION_METADATA_KEY).unwrap_or("unknown");
        let (source_chain, root_cause) = super::error::status_source_diagnostics(status);
        let diagnostic = source_chain
            .clone()
            .unwrap_or_else(|| format!("grpc_code={:?} message={}", status.code(), status.message()));
        Self {
            code: code.to_string(),
            kind: kind.to_string(),
            status: error_status.to_string(),
            operation: operation.to_string(),
            message: status.message().to_string(),
            facts: Vec::new(),
            diagnostic,
            source_chain,
            root_cause,
        }
    }

    fn write_to(&self, output: &mut impl fmt::Write) {
        let _ = writeln!(output, "code={}", StableValue(&self.code));
        let _ = writeln!(output, "kind={}", StableValue(&self.kind));
        let _ = writeln!(output, "status={}", StableValue(&self.status));
        let _ = writeln!(output, "operation={}", StableValue(&self.operation));
        let _ = writeln!(output, "message={}", StableValue(&self.message));
        let _ = writeln!(output, "diagnostic={}", StableValue(&self.diagnostic));
        let _ = writeln!(output, "source_chain={}", StableValue(self.source_chain.as_deref().unwrap_or("")));
        let _ = writeln!(output, "root_cause={}", StableValue(self.root_cause.as_deref().unwrap_or("")));
        for (key, value) in &self.facts {
            let _ = writeln!(output, "fact.{}={}", StableValue(key), StableValue(value));
        }
    }

    fn fact_value(&self, key: &str) -> Option<&str> {
        self.facts.iter().find_map(|(fact_key, value)| (fact_key == key).then_some(value.as_str()))
    }
}

/// Projects the local DataFusion metrics of every native execution wrapper in
/// a physical plan into one query-level view. Physical plans can be DAGs, and
/// wrapper clones can expose the same metric objects, so both plan-node and
/// metric identities are deduplicated before projection.
struct MetricsProjection {
    handles: MetricHandles,
    truncated: bool,
}

fn project_native_execution_metrics(
    plan: &Arc<dyn ExecutionPlan>,
    max_records: usize,
    max_logical_bytes: usize,
) -> MetricsProjection {
    fn visit(
        plan: &Arc<dyn ExecutionPlan>,
        seen_plans: &mut HashSet<*const ()>,
        seen_metrics: &mut HashSet<*const datafusion::physical_plan::metrics::Metric>,
        projected: &mut Vec<Arc<Metric>>,
        saw_native_wrapper: &mut bool,
        max_records: usize,
        retained_metrics: &mut usize,
        remaining_bytes: &mut usize,
        truncated: &mut bool,
    ) {
        if seen_plans.len() >= max_records.max(1) {
            *truncated = true;
            return;
        }
        let plan_identity = Arc::as_ptr(plan) as *const ();
        if !seen_plans.insert(plan_identity) {
            return;
        }

        if is_native_execution_wrapper(plan.as_ref()) {
            *saw_native_wrapper = true;
            if let Some(metrics) = plan.metrics() {
                for metric in metrics.iter() {
                    if !seen_metrics.insert(Arc::as_ptr(metric)) {
                        continue;
                    }
                    if metric.value().name() == crate::native_exec::SERVER_CAPTURE_LOSS_METRIC_NAME {
                        *truncated |= metric.value().as_usize() > 0;
                        continue;
                    }
                    if *retained_metrics >= max_records {
                        *truncated = true;
                        continue;
                    }
                    let logical_bytes = metric.labels().iter().fold(128usize, |total, label| {
                        total.saturating_add(label.name().len()).saturating_add(label.value().len())
                    });
                    if logical_bytes > *remaining_bytes {
                        *truncated = true;
                        continue;
                    }
                    *remaining_bytes -= logical_bytes;
                    projected.push(Arc::clone(metric));
                    *retained_metrics += 1;
                }
            }
        }

        for child in plan.children() {
            visit(
                child,
                seen_plans,
                seen_metrics,
                projected,
                saw_native_wrapper,
                max_records,
                retained_metrics,
                remaining_bytes,
                truncated,
            );
        }
    }

    let mut projected = Vec::new();
    let mut saw_native_wrapper = false;
    let mut truncated = false;
    let mut retained_metrics = 0;
    let mut remaining_bytes = max_logical_bytes;
    visit(
        plan,
        &mut HashSet::new(),
        &mut HashSet::new(),
        &mut projected,
        &mut saw_native_wrapper,
        max_records,
        &mut retained_metrics,
        &mut remaining_bytes,
        &mut truncated,
    );
    let retained_capacity_bytes = projected.iter().fold(0u64, |total, metric| {
        let labels = metric.labels().iter().fold(128u64, |bytes, label| {
            bytes
                .saturating_add(u64::try_from(label.name().len()).unwrap_or(u64::MAX))
                .saturating_add(u64::try_from(label.value().len()).unwrap_or(u64::MAX))
        });
        total.saturating_add(labels).saturating_add(u64::try_from(std::mem::size_of::<Arc<Metric>>()).unwrap_or(0))
    });
    MetricsProjection {
        handles: MetricHandles { metrics: projected.into_boxed_slice(), saw_native_wrapper, retained_capacity_bytes },
        truncated,
    }
}

fn is_native_execution_wrapper(plan: &dyn ExecutionPlan) -> bool {
    if plan.downcast_ref::<NexusNativeExec>().is_some()
        || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
        || plan.downcast_ref::<NexusNativeComposedGpuProducerExec>().is_some()
        || plan.downcast_ref::<NexusNativeGpuProducerExec>().is_some()
        || plan.downcast_ref::<crate::cuvs_sql::CuvsAlgoExec>().is_some()
    {
        return true;
    }
    #[cfg(feature = "cugraph")]
    if plan.downcast_ref::<crate::cugraph_sql::CugraphAlgoExec>().is_some() {
        return true;
    }
    false
}

fn contains_native_wrapper(plan: &dyn ExecutionPlan) -> bool {
    if is_native_execution_wrapper(plan) {
        return true;
    }
    plan.children().iter().any(|child| contains_native_wrapper(child.as_ref()))
}

fn metadata_value<'a>(metadata: &'a tonic::metadata::MetadataMap, key: &'static str) -> Option<&'a str> {
    metadata.get(key).and_then(|value| value.to_str().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_tsv_never_writes_partial_header_or_row() {
        let header = "a\tb\tc\n";
        let row = "1\t2\t3\n";
        let mut header_limited = BoundedText::new(header.len() - 1);
        assert!(!write_tsv_atomic(&mut header_limited, &format!("{header}{row}")));
        assert!(header_limited.text.is_empty());
        assert!(header_limited.truncated);

        let mut row_limited = BoundedText::new(header.len() + row.len() - 1);
        assert!(!write_tsv_atomic(&mut row_limited, &format!("{header}{row}")));
        assert_eq!(row_limited.text, header);
        assert!(row_limited.truncated);
        assert_eq!(row_limited.text.lines().count(), 1);
    }

    #[test]
    fn bounded_tsv_stops_after_first_failed_line_even_when_later_lines_fit() {
        let long_header = "long\theader\tthat-does-not-fit\n";
        let short_row = "x\n";
        let mut header_limited = BoundedText::new(short_row.len());
        assert!(!write_tsv_atomic(&mut header_limited, &format!("{long_header}{short_row}")));
        assert!(header_limited.text.is_empty(), "a row must never appear without its failed header");
        assert!(header_limited.truncated);
        assert_eq!(header_limited.discarded_bytes, long_header.len());

        let header = "a\n";
        let long_row = "row-that-does-not-fit\n";
        let trailing_short_row = "z\n";
        let mut row_limited = BoundedText::new(header.len() + trailing_short_row.len());
        assert!(!write_tsv_atomic(&mut row_limited, &format!("{header}{long_row}{trailing_short_row}")));
        assert_eq!(row_limited.text, header, "rows after the first failed row must not be appended");
        assert!(row_limited.truncated);
        assert_eq!(row_limited.discarded_bytes, long_row.len());
    }
}
