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
use query_engine::observability::{AttemptDisposition, QueryAttemptReport, RuntimeTerminalOutcome};

use crate::server::{
    adapter_error::{Error, ErrorCode, ErrorSourceDiagnostics, Facts},
    native_exec::{
        AlgeonNativeComposedExec, AlgeonNativeComposedGpuProducerExec, AlgeonNativeExec, AlgeonNativeGpuProducerExec,
    },
    native_report::{
        NativeDataFusionExecutionReport, NativeDataFusionExecutionTsvTemplate, NativeDataFusionExecutionWorkloadItem,
        NativeDataFusionPlanningReport, NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime,
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

/// Mask placeholder substituted for every masked `OPTIONS` value. Fixed
/// length so a masked artifact never leaks the secret's length.
const SQL_OPTION_VALUE_MASK: &str = "***";

/// Masks the values inside every DataFusion `OPTIONS (...)` clause of `sql`
/// (as used by `CREATE EXTERNAL TABLE ... OPTIONS (...)` and `COPY ... TO ...
/// OPTIONS (...)`), keeping the option keys intact so an operator can still
/// see which option was set. `sql` with no `OPTIONS` clause is returned
/// byte-identical.
///
/// This is a plain left-to-right scan over the token stream, not a SQL
/// parser: it tracks only string-literal boundaries (so `OPTIONS` inside a
/// string, or a quote inside a value, is never misread) and the `OPTIONS (
/// ... )` key/value shape from `datafusion-sql`'s grammar (`'key' 'value',
/// key2 value2, ...`; no `=`). An `OPTIONS (` clause that never reaches its
/// closing `)` (or a value literal that is never closed) masks to the end of
/// input rather than emit the unscanned tail, so a truncated or malformed
/// statement fails closed.
fn redact_sql_options(sql: &str) -> std::borrow::Cow<'_, str> {
    if !contains_options_keyword(sql) {
        return std::borrow::Cow::Borrowed(sql);
    }
    let bytes = sql.as_bytes();
    let mut output = String::with_capacity(sql.len());
    let mut index = 0;
    // Bytes not part of a quote or the `OPTIONS` keyword are copied through
    // verbatim as UTF-8 substrings of `sql`, never decoded byte-by-byte, so
    // multi-byte characters outside those two ASCII constructs are untouched.
    let mut plain_start = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\'' | b'"' => {
                output.push_str(&sql[plain_start..index]);
                let end = skip_quoted_literal(bytes, index);
                output.push_str(&sql[index..end]);
                index = end;
                plain_start = index;
            },
            _ if matches_keyword_at(bytes, index, b"OPTIONS") => {
                output.push_str(&sql[plain_start..index]);
                let keyword_end = index + "OPTIONS".len();
                output.push_str(&sql[index..keyword_end]);
                let mut cursor = keyword_end;
                skip_whitespace(bytes, &mut cursor);
                if cursor < bytes.len() && bytes[cursor] == b'(' {
                    output.push_str(&sql[keyword_end..cursor]);
                    index = mask_options_clause(sql, bytes, cursor, &mut output);
                } else {
                    index = keyword_end;
                }
                plain_start = index;
            },
            _ => index += 1,
        }
    }
    output.push_str(&sql[plain_start..]);
    std::borrow::Cow::Owned(output)
}

/// Cheap pre-check so SQL with no `OPTIONS` clause never allocates: an ASCII
/// case-insensitive substring search for the bare word `options`, ignoring
/// whether it occurs inside a string literal (an over-approximation is fine
/// here; the real scan below still does the precise, string-aware check).
fn contains_options_keyword(sql: &str) -> bool {
    let bytes = sql.as_bytes();
    (0..bytes.len()).any(|index| matches_keyword_at(bytes, index, b"OPTIONS"))
}

/// Returns true when `bytes[index..]` starts with `keyword` (ASCII
/// case-insensitive) as a standalone word: not preceded or followed by an
/// identifier character, so `OPTIONS` never matches inside a longer
/// identifier such as `my_options_table`.
fn matches_keyword_at(bytes: &[u8], index: usize, keyword: &[u8]) -> bool {
    if index + keyword.len() > bytes.len() {
        return false;
    }
    if index > 0 && is_identifier_byte(bytes[index - 1]) {
        return false;
    }
    if !bytes[index..index + keyword.len()].eq_ignore_ascii_case(keyword) {
        return false;
    }
    let after = index + keyword.len();
    !(after < bytes.len() && is_identifier_byte(bytes[after]))
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn skip_whitespace(bytes: &[u8], cursor: &mut usize) {
    while *cursor < bytes.len() && bytes[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
}

/// Finds the end of a single- or double-quoted string literal starting at
/// `start` (`bytes[start]` is the opening quote), honoring the standard SQL
/// `''`/`""` doubled-quote escape. Returns the index just past the literal's
/// closing quote, or `bytes.len()` when the literal is never closed.
fn skip_quoted_literal(bytes: &[u8], start: usize) -> usize {
    let quote = bytes[start];
    let mut cursor = start + 1;
    loop {
        match bytes[cursor..].iter().position(|&byte| byte == quote) {
            None => return bytes.len(),
            Some(relative) => {
                let quote_index = cursor + relative;
                if bytes.get(quote_index + 1) == Some(&quote) {
                    cursor = quote_index + 2;
                } else {
                    return quote_index + 1;
                }
            },
        }
    }
}

/// Advances `cursor` past whitespace, copying it through to `output`
/// unmasked. Returns `false` (having appended the fail-closed mask) when
/// this reaches end of input.
fn copy_whitespace(sql: &str, bytes: &[u8], cursor: &mut usize, output: &mut String) -> bool {
    let start = *cursor;
    skip_whitespace(bytes, cursor);
    output.push_str(&sql[start..*cursor]);
    if *cursor >= bytes.len() {
        output.push_str(SQL_OPTION_VALUE_MASK);
        return false;
    }
    true
}

/// Masks the value of every `'key' value` / `key 'value'` pair inside an
/// `OPTIONS (...)` clause whose opening `(` is at `sql[paren_index]`,
/// appending the masked clause to `output`. Returns the index in `sql` just
/// past the clause's closing `)`, or `sql.len()` when the clause is never
/// closed (in which case the remainder of `sql` is masked wholesale).
fn mask_options_clause(sql: &str, bytes: &[u8], paren_index: usize, output: &mut String) -> usize {
    output.push('(');
    let mut cursor = paren_index + 1;
    loop {
        if !copy_whitespace(sql, bytes, &mut cursor, output) {
            return bytes.len();
        }
        if bytes[cursor] == b')' {
            output.push(')');
            return cursor + 1;
        }

        // Key: quoted literal or bare identifier (dotted, per
        // `parse_option_key`). Keys are never masked.
        let key_start = cursor;
        cursor = if matches!(bytes[cursor], b'\'' | b'"') {
            skip_quoted_literal(bytes, cursor)
        } else {
            skip_option_word(bytes, cursor)
        };
        output.push_str(&sql[key_start..cursor.min(bytes.len())]);
        if cursor >= bytes.len() {
            output.push_str(SQL_OPTION_VALUE_MASK);
            return bytes.len();
        }

        if !copy_whitespace(sql, bytes, &mut cursor, output) {
            return bytes.len();
        }
        // `)` or `,` here means a malformed pair (key with no value):
        // nothing to mask.
        if bytes[cursor] == b')' {
            output.push(')');
            return cursor + 1;
        }
        if bytes[cursor] != b',' {
            // Value: mask its span, keeping delimiting quotes (if any) intact.
            if matches!(bytes[cursor], b'\'' | b'"') {
                let quote = bytes[cursor];
                let end = skip_quoted_literal(bytes, cursor);
                let closed = end > cursor + 1 && end <= bytes.len() && bytes[end - 1] == quote;
                if !closed {
                    output.push_str(SQL_OPTION_VALUE_MASK);
                    return bytes.len();
                }
                output.push(quote as char);
                output.push_str(SQL_OPTION_VALUE_MASK);
                output.push(quote as char);
                cursor = end;
            } else {
                cursor = skip_option_word(bytes, cursor);
                output.push_str(SQL_OPTION_VALUE_MASK);
            }
            if !copy_whitespace(sql, bytes, &mut cursor, output) {
                return bytes.len();
            }
            if bytes[cursor] == b')' {
                output.push(')');
                return cursor + 1;
            }
        }
        // Consume the pair-separating comma before looping back for the
        // next key. Anything else here is malformed input; the outer loop
        // re-synchronizes on the next `OPTIONS`/quote/end-of-input.
        if bytes[cursor] == b',' {
            output.push(',');
            cursor += 1;
        }
    }
}

/// Skips one bare `OPTIONS` key/value word: a run of identifier bytes
/// (covering dotted keys like `format.compression` and bare values like
/// `snappy` or `55`). Advances by at least one byte so a stray delimiter
/// (malformed input, e.g. a bare `,` where a key was expected) cannot stall
/// the scan.
fn skip_option_word(bytes: &[u8], start: usize) -> usize {
    let mut cursor = start;
    while cursor < bytes.len() && (is_identifier_byte(bytes[cursor]) || bytes[cursor] == b'.') {
        cursor += 1;
    }
    if cursor > start {
        return cursor;
    }
    // No identifier byte here (a stray delimiter or non-ASCII byte where a
    // key/value word was expected): advance to the next UTF-8 char boundary
    // so the caller's `sql[..cursor]` slice never lands mid-character.
    cursor += 1;
    while cursor < bytes.len() && !is_utf8_char_boundary(bytes[cursor]) {
        cursor += 1;
    }
    cursor
}

fn is_utf8_char_boundary(byte: u8) -> bool {
    // Continuation bytes are `10xxxxxx`; every other byte starts a char.
    byte & 0b1100_0000 != 0b1000_0000
}

/// `Display` wrapper that masks `OPTIONS` clause values before newline/CR/tab
/// stabilization, so SQL text is redacted before it is ever formatted for a
/// log line or retained artifact. Distinct from [`StableValue`], which is
/// also used for non-SQL fields (query ids, error facts) that must never be
/// masked.
pub(super) struct RedactedSql<'a>(pub(super) &'a str);

impl fmt::Display for RedactedSql<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redacted = redact_sql_options(self.0);
        StableValue(&redacted).fmt(formatter)
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
    delivery_outcome: Option<CapturedDeliveryOutcome>,
    physical_explain: Option<Arc<str>>,
    native_explain: Option<Arc<str>>,
    last_touched_at: Instant,
    sealed: bool,
    capture: CaptureBudget,
    max_metric_records: usize,
    /// Immutable attempt-completion projection. The coordinator never owns
    /// this lifecycle; it only retains the sealed report for Flight stats.
    attempt_report: Option<Arc<QueryAttemptReport>>,
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
        // Redact before retention: the captured copy backs `GetQueryStats` /
        // `GetQueryRuntimeArtifact` client responses, so it must never carry
        // an unredacted secret, independent of how it is later rendered.
        let sql = capture.retain_text(&redact_sql_options(&sql));
        let correlation_id = correlation_id.as_deref().map(|value| capture.retain_text(value));
        Self {
            query_id,
            sql,
            correlation_id,
            planning: None,
            planning_failure: None,
            metrics: CapturedMetrics::default(),
            delivery_outcome: None,
            physical_explain: None,
            native_explain: None,
            last_touched_at: now,
            sealed: false,
            capture,
            max_metric_records: usize::try_from(max_metric_records).unwrap_or(usize::MAX),
            attempt_report: None,
        }
    }

    pub(super) fn set_sql(&mut self, sql: String) {
        self.capture.replace_text(&mut self.sql, redact_sql_options(&sql));
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

    pub(super) fn set_attempt_report(&mut self, report: Arc<QueryAttemptReport>) {
        self.attempt_report = Some(report);
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

    pub(super) fn finish_success(&mut self, _now: Instant) {
        if self.delivery_outcome.is_none() {
            self.delivery_outcome = Some(CapturedDeliveryOutcome::Succeeded);
        }
    }

    pub(super) fn finish_failed(&mut self, stage: &'static str, error: Arc<CapturedError>, _now: Instant) {
        if matches!(self.delivery_outcome, None | Some(CapturedDeliveryOutcome::Dropped)) {
            self.delivery_outcome = Some(CapturedDeliveryOutcome::Failed { stage, error });
        }
    }

    pub(super) fn finish_cancelled(&mut self, stage: &'static str, error: Arc<CapturedError>, _now: Instant) {
        if matches!(self.delivery_outcome, None | Some(CapturedDeliveryOutcome::Dropped)) {
            self.delivery_outcome = Some(CapturedDeliveryOutcome::Cancelled { stage, error });
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
            elapsed_ms: self.elapsed_ms(),
        }
    }

    pub(super) fn finish_dropped(&mut self, _now: Instant) {
        if self.delivery_outcome.is_none() {
            self.delivery_outcome = Some(CapturedDeliveryOutcome::Dropped);
        }
    }

    pub(super) fn seal(&mut self, now: Instant) {
        self.sealed = true;
        if self.delivery_outcome.is_none() {
            self.delivery_outcome = Some(CapturedDeliveryOutcome::Dropped);
        }
        self.last_touched_at = now;
    }

    pub(super) fn to_query_stats_payload(&self, max_bytes: usize) -> QueryStatsPayload {
        let loss_metadata_bytes =
            self.capture.losses.keys().map(|kind| kind.as_str().len().saturating_add(48)).sum::<usize>();
        let trailer_reserve = loss_metadata_bytes.saturating_add(512).min(max_bytes);
        let mut output = BoundedText::new(max_bytes.saturating_sub(trailer_reserve));
        let _ = writeln!(output, "# algeon_datafusion_query_stats=1");
        let _ = writeln!(output, "# query_id={}", StableValue(&self.query_id));
        if let Some(correlation_id) = &self.correlation_id {
            let _ = writeln!(output, "# correlation_id={}", StableValue(correlation_id));
        }
        let delivery_outcome = self.delivery_outcome.as_ref().map_or("running", CapturedDeliveryOutcome::as_str);
        let outcome = self
            .attempt_report
            .as_ref()
            .map(|report| report.sealed_projection().labels().outcome)
            .unwrap_or(delivery_outcome);
        let _ = writeln!(output, "# outcome={outcome}");
        // A Flight coordinator may report how response delivery ended, but it
        // never owns the attempt lifecycle. Keep that distinct from the
        // canonical, sealed attempt outcome.
        if outcome != delivery_outcome {
            let _ = writeln!(output, "# delivery_outcome={delivery_outcome}");
        }
        let _ = writeln!(output, "# sealed={}", self.sealed);
        let _ = writeln!(
            output,
            "# elapsed_ms={}",
            self.elapsed_ms().map_or_else(|| "none".to_string(), |value| value.to_string())
        );
        if !self.sql.is_empty() {
            let _ = writeln!(output, "# sql={}", StableValue(&self.sql));
        }
        if let Some(report) = &self.attempt_report {
            let labels = report.sealed_projection().labels();
            let _ = writeln!(output, "# attempt_id={}", report.attempt_id);
            let _ = writeln!(output, "# attempt_disposition={}", labels.disposition);
            let _ = writeln!(output, "# attempt_outcome={}", labels.outcome);
            let _ = writeln!(
                output,
                "# attempt_device_ordinal={}",
                report.device_ordinal.map_or_else(|| "none".to_string(), |value| value.to_string())
            );
            let _ = writeln!(output, "# attempt_cleanup_disposition={}", labels.cleanup_disposition);
            let _ = writeln!(output, "# attempt_cleanup_failure_count={}", report.cleanup_failure_count);
            let _ = writeln!(output, "# attempt_capacity_disposition={}", labels.capacity_disposition);
            let _ = writeln!(output, "# attempt_retry_recovery={}", labels.retry_recovery);
            let _ = writeln!(output, "# attempt_retry_prerequisite={}", labels.retry_prerequisite);
            let _ =
                writeln!(output, "# attempt_cleanup_failure_count_complete={}", labels.cleanup_failure_count_complete);
            let _ = writeln!(output, "# attempt_health_transition={}", labels.health_transition);
            let _ = writeln!(output, "# attempt_observation_truncated={}", labels.observation_truncated);
            let _ = writeln!(output, "# attempt_observation_degraded={}", labels.observation_degraded);
            let _ = writeln!(output, "# attempt_native_fault_evidence={}", labels.native_fault_evidence);
            let _ = writeln!(output, "# attempt_resource_evidence={}", labels.resource_evidence);
        }

        let mut metric_projection_truncated = false;
        if let Some(planning) = &self.planning {
            let planning_written =
                output.write_atomic("\n[planning]\n") && planning.planning_report.write_atomic(&mut output);
            metric_projection_truncated = planning.metrics.truncated;
            let runtime = self.runtime_for_planning(planning.metrics.metrics());
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

        if let Some(error) = self.delivery_outcome.as_ref().and_then(CapturedDeliveryOutcome::error) {
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

    fn runtime_for_planning(&self, metrics: Option<MetricsSet>) -> NativeDataFusionRuntime {
        let Some(report) = self.attempt_report.as_deref() else {
            return NativeDataFusionRuntime::NotStarted;
        };
        let native_runtime_started =
            matches!(report.disposition, AttemptDisposition::Native | AttemptDisposition::PartialNative);
        if !native_runtime_started {
            return NativeDataFusionRuntime::NotStarted;
        }
        match &report.outcome {
            RuntimeTerminalOutcome::Succeeded => NativeDataFusionRuntime::Succeeded {
                output_batches: self.metrics.output_batches.unwrap_or(0),
                output_rows: self.metrics.output_rows.unwrap_or(0),
                output_bytes: self.metrics.output_bytes.unwrap_or(0),
                metrics: metrics.unwrap_or_default(),
            },
            RuntimeTerminalOutcome::Failed { .. } => {
                NativeDataFusionRuntime::Failed { message: attempt_failure_message(report), metrics }
            },
            RuntimeTerminalOutcome::Cancelled => NativeDataFusionRuntime::Interrupted {
                outcome: crate::server::native_report::NativeDataFusionRuntimeOutcome::Cancelled,
                metrics,
            },
            RuntimeTerminalOutcome::Dropped => NativeDataFusionRuntime::Interrupted {
                outcome: crate::server::native_report::NativeDataFusionRuntimeOutcome::Dropped,
                metrics,
            },
            RuntimeTerminalOutcome::Panicked => NativeDataFusionRuntime::Interrupted {
                outcome: crate::server::native_report::NativeDataFusionRuntimeOutcome::Panicked,
                metrics,
            },
        }
    }

    fn elapsed_ms(&self) -> Option<u128> {
        self.attempt_report
            .as_deref()
            .and_then(QueryAttemptReport::total_ns)
            .map(|total_ns| u128::from(total_ns) / 1_000_000)
    }
}

fn attempt_failure_message(report: &QueryAttemptReport) -> String {
    report.primary_failure.as_ref().map_or_else(
        || "attempt_failure_summary_unavailable".to_string(),
        |failure| serde_json::to_string(failure).unwrap_or_else(|_| "attempt_failure_summary_unavailable".to_string()),
    )
}

/// The server may retain a terminal outcome before it has the full sealed
/// projection. Its label still comes from the engine's one closed mapping.
pub(super) fn attempt_outcome_label(outcome: &RuntimeTerminalOutcome) -> &'static str {
    outcome.stable_label()
}

pub(super) struct QueryStatsPayload {
    pub(super) bytes: Arc<[u8]>,
    pub(super) incomplete: bool,
}

#[derive(Debug, Clone)]
pub(super) struct ShutdownCaptureSnapshot {
    query_id: String,
    correlation_id: Option<Arc<str>>,
    elapsed_ms: Option<u128>,
}

impl ShutdownCaptureSnapshot {
    #[expect(clippy::too_many_arguments, reason = "service wiring parameters are distinct ownership values")]
    pub(super) fn to_query_stats_payload(
        &self,
        _now: Instant,
        output_batches: usize,
        output_rows: usize,
        output_bytes: usize,
        outcome: &str,
        delivery_outcome: Option<&str>,
        max_bytes: usize,
    ) -> QueryStatsPayload {
        let mut output = BoundedText::new(max_bytes);
        let mut lines = vec![
            "# algeon_datafusion_query_stats=1\n".to_string(),
            format!("# query_id={}\n", StableValue(&self.query_id)),
        ];
        if let Some(correlation_id) = &self.correlation_id {
            lines.push(format!("# correlation_id={}\n", StableValue(correlation_id)));
        }
        lines.push(format!("# outcome={outcome}\n"));
        if let Some(delivery_outcome) = delivery_outcome {
            lines.push(format!("# delivery_outcome={delivery_outcome}\n"));
        }
        lines.extend([
            "# sealed=true\n".to_string(),
            "# incomplete=true\n".to_string(),
            "# unavailable_reason=shutdown_timeout\n".to_string(),
            format!("# elapsed_ms={}\n", self.elapsed_ms.map_or_else(|| "none".to_string(), |value| value.to_string())),
            format!("# delivery_output_batches={output_batches}\n"),
            format!("# delivery_output_rows={output_rows}\n"),
            format!("# delivery_output_bytes={output_bytes}\n"),
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
    retained_capacity_bytes: u64,
}

impl CapturedPlanning {
    fn new(
        planning_report: CapturedTsvReport,
        execution_report: NativeDataFusionExecutionTsvTemplate,
        metrics: MetricsProjection,
    ) -> Self {
        let retained_capacity_bytes = planning_report
            .retained_capacity_bytes
            .saturating_add(execution_report.retained_capacity_bytes())
            .saturating_add(metrics.handles.retained_capacity_bytes);
        Self {
            planning_report,
            execution_report,
            metrics: CapturedMetricHandles { handles: metrics.handles, truncated: metrics.truncated },
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
pub(super) enum CapturedDeliveryOutcome {
    Succeeded,
    Failed { stage: &'static str, error: Arc<CapturedError> },
    Cancelled { stage: &'static str, error: Arc<CapturedError> },
    Dropped,
}

impl CapturedDeliveryOutcome {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed { .. } => "failed",
            Self::Cancelled { .. } => "cancelled",
            Self::Dropped => "dropped",
        }
    }

    fn error(&self) -> Option<&CapturedError> {
        match self {
            Self::Failed { error, .. } | Self::Cancelled { error, .. } => Some(error.as_ref()),
            Self::Succeeded | Self::Dropped => None,
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
        Self {
            code: error.code().as_str().to_string(),
            kind: error.kind().as_str().to_string(),
            status: error.status().as_str().to_string(),
            operation: error.operation().to_string(),
            message: error.message().to_string(),
            // Query-caller capture keeps the log-safe projection plus the
            // explicitly caller-visible schema details. Operational logs keep
            // using the log-safe projection alone and never see identifiers.
            facts: error.facts().retained_capture_entries(),
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
        Self {
            code: code.as_str().to_string(),
            kind: code.kind().as_str().to_string(),
            status: code.status().as_str().to_string(),
            operation: operation.to_string(),
            diagnostic,
            message,
            facts: facts.retained_capture_entries(),
            source_chain: source_diagnostics.chain().map(str::to_string),
            root_cause: source_diagnostics.root_cause().map(str::to_string),
        }
    }

    pub(super) fn from_status(status: &tonic::Status) -> Self {
        let metadata = status.metadata();
        let code = metadata_value(metadata, crate::server::error::ERROR_CODE_METADATA_KEY).unwrap_or("unknown");
        let kind = metadata_value(metadata, crate::server::error::ERROR_KIND_METADATA_KEY).unwrap_or("unknown");
        let error_status =
            metadata_value(metadata, crate::server::error::ERROR_STATUS_METADATA_KEY).unwrap_or("unknown");
        let operation =
            metadata_value(metadata, crate::server::error::ERROR_OPERATION_METADATA_KEY).unwrap_or("unknown");
        let (source_chain, root_cause) = crate::server::error::status_source_diagnostics(status);
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
    #[expect(clippy::too_many_arguments, reason = "service wiring parameters are distinct ownership values")]
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
                    if metric.value().name() == crate::server::native_exec::SERVER_CAPTURE_LOSS_METRIC_NAME {
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
    if plan.downcast_ref::<AlgeonNativeExec>().is_some()
        || plan.downcast_ref::<AlgeonNativeComposedExec>().is_some()
        || plan.downcast_ref::<AlgeonNativeComposedGpuProducerExec>().is_some()
        || plan.downcast_ref::<AlgeonNativeGpuProducerExec>().is_some()
        || plan.downcast_ref::<crate::server::cuvs_sql::CuvsAlgoExec>().is_some()
    {
        return true;
    }
    #[cfg(feature = "cugraph")]
    if plan.downcast_ref::<crate::server::cugraph_sql::CugraphAlgoExec>().is_some() {
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

    fn cancelled_attempt_report() -> QueryAttemptReport {
        use query_engine::observability::{
            AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptId, AttemptTimestamps,
            QUERY_ATTEMPT_REPORT_SCHEMA_V1,
        };

        QueryAttemptReport {
            schema: QUERY_ATTEMPT_REPORT_SCHEMA_V1.to_string(),
            attempt_id: AttemptId::mint(),
            query_id: None,
            trace_context: None,
            disposition: AttemptDisposition::Native,
            outcome: RuntimeTerminalOutcome::Cancelled,
            primary_failure: None,
            retry_advice: query_engine::observability::AttemptRetryAdvice::do_not_retry(),
            device_ordinal: Some(0),
            admission: None,
            admission_wait: None,
            composed_schedule: None,
            timestamps: AttemptTimestamps {
                submitted_ns: 1,
                planning_started_ns: Some(2),
                planning_finished_ns: Some(3),
                admission_requested_ns: Some(4),
                admitted_ns: Some(5),
                physical_execution_started_ns: Some(6),
                gpu_execution_started_ns: Some(7),
                outcome_terminal_ns: 8,
                cleanup_complete_ns: Some(9),
                report_ready_ns: Some(10),
            },
            cleanup_disposition: AttemptCleanupDisposition::Completed,
            cleanup_failure_count: 0,
            cleanup_failure_count_complete: true,
            first_cleanup_failure: None,
            capacity_disposition: AttemptCapacityDisposition::Released,
            final_domain_usage: None,
            health_transition: None,
            native_fault_evidence: None,
            resource_evidence: None,
            observation_lost_records: 0,
            observation_retained_records: 0,
            observation_retained_bytes: 0,
            observation_truncated: false,
            observation_degraded: false,
            unix_epoch_ns: None,
        }
    }

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

    #[test]
    fn redact_sql_options_masks_single_quoted_values_and_keeps_keys() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS PARQUET OPTIONS ('aws.access_key_id' 'AKIAEXAMPLE', \
                   'aws.secret_access_key' 'super-secret-value') LOCATION 's3://bucket/path'";
        let redacted = redact_sql_options(sql);
        assert!(redacted.contains("'aws.access_key_id' '***'"), "{redacted}");
        assert!(redacted.contains("'aws.secret_access_key' '***'"), "{redacted}");
        assert!(!redacted.contains("AKIAEXAMPLE"), "{redacted}");
        assert!(!redacted.contains("super-secret-value"), "{redacted}");
        assert!(redacted.contains("LOCATION 's3://bucket/path'"), "value outside OPTIONS must survive: {redacted}");
    }

    #[test]
    fn redact_sql_options_handles_doubled_single_quote_escapes_inside_a_value() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV OPTIONS ('format.delimiter' 'a''b') LOCATION 'x.csv'";
        let redacted = redact_sql_options(sql);
        assert!(redacted.contains("'format.delimiter' '***'"), "{redacted}");
        assert!(!redacted.contains("a''b"), "{redacted}");
        // The escaped value must not be mistaken for the literal's end: the
        // clause must still close correctly and LOCATION must survive.
        assert!(redacted.contains("LOCATION 'x.csv'"), "{redacted}");
    }

    #[test]
    fn redact_sql_options_fails_closed_on_unterminated_clause() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV OPTIONS ('aws.secret_access_key' 'super-secret";
        let redacted = redact_sql_options(sql);
        assert!(!redacted.contains("super-secret"), "{redacted}");
        // An unterminated clause masks to end of input rather than emit any
        // unscanned tail verbatim.
        assert!(redacted.ends_with(SQL_OPTION_VALUE_MASK), "{redacted}");
    }

    #[test]
    fn redact_sql_options_ignores_options_word_inside_a_string_literal_or_identifier() {
        let in_string = "SELECT 'this mentions OPTIONS but is not a clause' AS label";
        assert_eq!(redact_sql_options(in_string), in_string);

        let as_identifier = "SELECT * FROM my_options_table WHERE options_id = 1";
        assert_eq!(redact_sql_options(as_identifier), as_identifier);
    }

    #[test]
    fn redact_sql_options_is_case_insensitive() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV options ('k' 'secret-value') LOCATION 'x.csv'";
        let redacted = redact_sql_options(sql);
        assert!(!redacted.contains("secret-value"), "{redacted}");

        let sql_mixed = "CREATE EXTERNAL TABLE t STORED AS CSV Options ('k' 'secret-value') LOCATION 'x.csv'";
        let redacted_mixed = redact_sql_options(sql_mixed);
        assert!(!redacted_mixed.contains("secret-value"), "{redacted_mixed}");
    }

    #[test]
    fn redact_sql_options_passes_through_sql_with_no_options_clause_byte_identical() {
        let sql = "SELECT * FROM lineitem WHERE l_orderkey = 1";
        assert_eq!(redact_sql_options(sql), sql);
    }

    #[test]
    fn captured_query_retains_sql_already_redacted() {
        let now = Instant::now();
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV OPTIONS ('aws.secret_access_key' 'super-secret') \
                   LOCATION 'x.csv'"
            .to_string();
        let captured = CapturedQuery::new("query-redacted".to_string(), sql, None, now, 4096, 64);

        let payload = String::from_utf8(captured.to_query_stats_payload(4096).bytes.to_vec()).unwrap();
        assert!(!payload.contains("super-secret"), "{payload}");
        assert!(payload.contains("'aws.secret_access_key' '***'"), "{payload}");
    }

    #[test]
    fn captured_query_set_sql_redacts_the_replacement_too() {
        let now = Instant::now();
        let mut captured =
            CapturedQuery::new("query-redacted-2".to_string(), "select 1".to_string(), None, now, 4096, 64);
        captured.set_sql(
            "CREATE EXTERNAL TABLE t STORED AS CSV OPTIONS ('aws.secret_access_key' 'super-secret') \
             LOCATION 'x.csv'"
                .to_string(),
        );

        let payload = String::from_utf8(captured.to_query_stats_payload(4096).bytes.to_vec()).unwrap();
        assert!(!payload.contains("super-secret"), "{payload}");
        assert!(payload.contains("'aws.secret_access_key' '***'"), "{payload}");
    }

    #[test]
    fn attempt_report_outcome_is_canonical_over_delivery_outcome() {
        let now = Instant::now();
        let mut captured =
            CapturedQuery::new("attempt-canonical".to_string(), "select 1".to_string(), None, now, 4096, 64);
        captured.finish_failed(
            "delivery",
            Arc::new(CapturedError {
                code: "cancelled".to_string(),
                kind: "cancelled".to_string(),
                status: "permanent".to_string(),
                operation: "algeon_datafusion.server.flight.cancel_query".to_string(),
                message: "statement was cancelled".to_string(),
                facts: Vec::new(),
                diagnostic: "statement was cancelled".to_string(),
                source_chain: None,
                root_cause: None,
            }),
            now,
        );
        captured.set_attempt_report(Arc::new(cancelled_attempt_report()));
        captured.seal(now);

        let payload = String::from_utf8(captured.to_query_stats_payload(4096).bytes.to_vec()).unwrap();
        assert!(payload.contains("# outcome=cancelled\n"), "{payload}");
        assert!(payload.contains("# delivery_outcome=failed\n"), "{payload}");
        assert!(payload.contains("# attempt_disposition=native\n"), "{payload}");
        assert!(payload.contains("# attempt_outcome=cancelled\n"), "{payload}");
        assert!(payload.contains("# attempt_cleanup_disposition=completed\n"), "{payload}");
        assert!(payload.contains("# attempt_cleanup_failure_count_complete=true\n"), "{payload}");
        assert!(payload.contains("# attempt_capacity_disposition=released\n"), "{payload}");
        assert!(payload.contains("# attempt_retry_recovery=do_not_retry\n"), "{payload}");
        assert!(payload.contains("# attempt_retry_prerequisite=none\n"), "{payload}");
        assert!(payload.contains("# attempt_health_transition=none\n"), "{payload}");
        assert!(payload.contains("# attempt_observation_truncated=false\n"), "{payload}");
        assert!(payload.contains("# attempt_observation_degraded=false\n"), "{payload}");
        assert!(payload.contains("# attempt_native_fault_evidence=absent\n"), "{payload}");
        assert!(payload.contains("# attempt_resource_evidence=absent\n"), "{payload}");
    }
}
