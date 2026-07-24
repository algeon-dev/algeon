use super::*;

/// Table format that produced a resolved scan plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormat {
    /// Apache Iceberg table metadata.
    Iceberg,
    /// Delta Lake table metadata.
    Delta,
    /// Another table format identified by a stable caller-provided code.
    Other(String),
}

/// Format-neutral request passed to table-format metadata planners.
///
/// `None` projection means all logical columns. `None` snapshot/catalog values
/// leave selection to the table-format implementation. `None` scan limit means
/// no source-level row bound was supplied by DataFusion; `Some(n)` is an
/// order-insensitive upper bound the native source may apply before any
/// inexact parent filter. The filtering flags describe planner intent only;
/// unsupported row-group pruning or row selection must be reported as stable
/// unsupported features instead of executing data reads.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatScanRequest {
    /// Logical projected column names. `None` means all columns; `Some([])` means zero columns.
    pub projected_column_names: Option<Vec<String>>,
    /// Optional table snapshot id in the format's native textual form.
    pub snapshot_id: Option<String>,
    /// Optional stable catalog label, such as `glue`.
    pub catalog_type: Option<String>,
    /// Optional source-level row upper bound supplied by DataFusion.
    pub scan_limit: Option<usize>,
    /// Whether column resolution should be case-sensitive.
    pub case_sensitive: bool,
    /// Planner intent for row-group filtering. Unsupported semantics become scan facts.
    pub row_group_filtering_enabled: bool,
    /// Planner intent for row selection/page-index pruning. Unsupported semantics become scan facts.
    pub row_selection_enabled: bool,
    /// Format-neutral predicate converted from frontend filters for metadata planning.
    pub predicate: Option<TableFormatPredicate>,
}

impl TableFormatScanRequest {
    /// Request all logical columns using default scan-planning options.
    pub fn all_columns() -> Self {
        Self::default()
    }

    /// Sets the exact logical column names to project.
    pub fn with_projected_column_names(mut self, column_names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.projected_column_names = Some(column_names.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the table snapshot id.
    pub fn with_snapshot_id(mut self, snapshot_id: impl Into<String>) -> Self {
        self.snapshot_id = Some(snapshot_id.into());
        self
    }

    /// Sets a stable catalog label.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Sets an order-insensitive source-level scan limit.
    pub fn with_scan_limit(mut self, scan_limit: Option<usize>) -> Self {
        self.scan_limit = scan_limit;
        self
    }

    /// Sets case sensitivity for table-format column resolution.
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Sets whether the format planner should request row-group filtering.
    pub fn with_row_group_filtering_enabled(mut self, row_group_filtering_enabled: bool) -> Self {
        self.row_group_filtering_enabled = row_group_filtering_enabled;
        self
    }

    /// Sets whether the format planner should request row selection/page-index pruning.
    pub fn with_row_selection_enabled(mut self, row_selection_enabled: bool) -> Self {
        self.row_selection_enabled = row_selection_enabled;
        self
    }

    /// Sets a format-neutral metadata predicate converted from frontend filters.
    pub fn with_predicate(mut self, predicate: Option<TableFormatPredicate>) -> Self {
        self.predicate = predicate;
        self
    }
}

impl Default for TableFormatScanRequest {
    fn default() -> Self {
        Self {
            projected_column_names: None,
            snapshot_id: None,
            catalog_type: None,
            scan_limit: None,
            case_sensitive: true,
            row_group_filtering_enabled: true,
            row_selection_enabled: false,
            predicate: None,
        }
    }
}

/// Adapter-owned predicate tree for table-format metadata planning.
///
/// This intentionally avoids DataFusion, Iceberg, and native-engine expression
/// types. Format-specific planners translate this tree into their own metadata
/// predicate APIs when supported.
#[derive(Debug, Clone, PartialEq)]
pub enum TableFormatPredicate {
    /// Boolean conjunction. A planner may use both sides together.
    And(Box<TableFormatPredicate>, Box<TableFormatPredicate>),
    /// Boolean disjunction. A planner should use this only when both sides are supported.
    Or(Box<TableFormatPredicate>, Box<TableFormatPredicate>),
    /// Boolean negation.
    Not(Box<TableFormatPredicate>),
    /// Column equals true.
    BooleanColumn(String),
    /// Null predicate on a column.
    IsNull(String),
    /// Non-null predicate on a column.
    IsNotNull(String),
    /// Binary comparison between a column and a literal.
    Comparison {
        /// Logical column name.
        column: String,
        /// Comparison operator.
        op: TableFormatPredicateOp,
        /// Literal value.
        literal: TableFormatLiteral,
    },
}

/// Format-neutral comparison operators for table-format metadata predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatPredicateOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

/// Literal values supported by format-neutral table-format metadata predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum TableFormatLiteral {
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Utf8(String),
    Date32(i32),
}

/// Adapter-owned table-format metadata planning boundary.
///
/// Source resolution and optional native inspection are deliberately separate.
/// Resolution selects immutable table-format metadata and produces the scan
/// facts shared by both execution capabilities; its failures are terminal.
/// Native inspection may enrich those facts with acceleration-only source and
/// footer evidence; its failures may leave an already-built CPU delegate
/// executable. Neither phase executes table data. Diagnostics must not contain
/// credentials or signed URLs, and unsupported features should map to stable
/// fallback reasons.
pub trait TableFormatPlanner {
    /// Resolves immutable source metadata into scan facts without reading table data.
    fn resolve_scan(&self, request: TableFormatScanRequest) -> BoxFuture<'_, Result<TableFormatScanPlan>>;

    /// Resolves immutable source metadata with a caller-owned wall-clock deadline.
    ///
    /// The default implementation preserves existing planner behavior. Format
    /// planners with long synchronous or ready-loop sections should override
    /// this method and check the deadline at natural cancellation points.
    fn resolve_scan_with_deadline(
        &self,
        request: TableFormatScanRequest,
        _deadline: TableFormatPlanningDeadline,
    ) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        self.resolve_scan(request)
    }

    /// Enriches resolved scan facts with optional native-only inspection evidence.
    ///
    /// The default is the identity transformation for planners whose resolved
    /// facts need no additional native probes.
    fn inspect_native_scan(
        &self,
        _request: TableFormatScanRequest,
        resolved_scan: TableFormatScanPlan,
    ) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        Box::pin(async move { Ok(resolved_scan) })
    }

    /// Performs optional native inspection with a caller-owned wall-clock deadline.
    fn inspect_native_scan_with_deadline(
        &self,
        request: TableFormatScanRequest,
        resolved_scan: TableFormatScanPlan,
        _deadline: TableFormatPlanningDeadline,
    ) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        self.inspect_native_scan(request, resolved_scan)
    }

    /// Credential-free metadata facts to attach when either planning phase fails.
    fn planning_failure_facts(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Caller-owned wall-clock deadline for table-format metadata planning.
#[derive(Debug, Clone, Copy)]
pub struct TableFormatPlanningDeadline {
    deadline: Instant,
    timeout: Duration,
}

impl TableFormatPlanningDeadline {
    /// Builds a deadline from the current instant plus `timeout`.
    pub fn from_timeout(timeout: Duration) -> Self {
        Self { deadline: Instant::now() + timeout, timeout }
    }

    /// Configured timeout duration used to create the deadline.
    pub fn timeout(self) -> Duration {
        self.timeout
    }

    /// Millisecond budget for diagnostics, clamped to one millisecond.
    pub fn timeout_ms(self) -> u128 {
        self.timeout.as_millis().max(1)
    }

    /// Remaining wall-clock duration, or `None` when the deadline has elapsed.
    pub fn remaining(self) -> Option<Duration> {
        self.deadline.checked_duration_since(Instant::now()).filter(|remaining| !remaining.is_zero())
    }

    /// Returns true when the wall-clock deadline has elapsed.
    pub fn is_elapsed(self) -> bool {
        self.remaining().is_none()
    }
}

impl TableFormat {
    /// Stable report/fallback code for this table format.
    pub fn code(&self) -> &str {
        match self {
            Self::Iceberg => "iceberg",
            Self::Delta => "delta",
            Self::Other(format) => format.as_str(),
        }
    }
}
