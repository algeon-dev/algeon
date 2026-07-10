//! Query-level GPU coverage validation.
//!
//! This module answers the developer-facing planning question: whether the
//! optimized physical path contains known DataFusion CPU execution. It is a
//! plan-time validation surface, not a runtime admission or success guarantee.

use std::{
    collections::hash_map::DefaultHasher,
    error::Error as StdError,
    hash::{Hash, Hasher},
    ops::ControlFlow,
    sync::Arc,
};

use datafusion::{
    common::DataFusionError,
    execution::{SessionState, SessionStateBuilder},
    physical_plan::ExecutionPlan,
    prelude::SessionContext,
    sql::{
        parser::{DFParserBuilder, Statement as DfSqlStatement},
        sqlparser::{
            ast::{
                AccessExpr, Array as SqlArray, Assignment, Expr as SqlExpr, ExprWithAlias, ExprWithAliasAndOrderBy,
                Fetch, Function as SqlFunction, FunctionArg, FunctionArgExpr, FunctionArgumentClause,
                FunctionArguments, GroupByExpr, Join as SqlJoin, JoinConstraint, JoinOperator, LimitClause,
                Map as SqlMap, ObjectName, ObjectNamePart, OrderByExpr, OrderByKind, PipeOperator, PivotValueSource,
                Query as SqlQuery, Select as SqlSelect, SelectItem, SetExpr, Statement as SqlStatement, Subscript,
                TableFactor, TableSample, TableWithJoins, Visit, Visitor, WindowSpec, WindowType,
            },
            dialect::dialect_from_str,
        },
    },
};
use serde_json::{Value, json};

#[cfg(feature = "cugraph")]
use crate::graph_sql::CugraphAlgoExec;
use crate::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, Error, ErrorCode, FallbackReason,
    GpuFallbackPolicy, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec,
    NativeDataFusionPostNativeExecutionPath, PlanningReport,
    error::ErrorSourceDiagnostics,
    native_report::post_native_plan_coverage,
    physical_plan::physical_plan_shape,
    session::{CudfSessionStateBuilderExt, installed_cudf_native_optimizer_config},
};

/// JSON schema identifier emitted by [`QueryGpuCoverage::to_json`].
pub const QUERY_GPU_COVERAGE_JSON_SCHEMA: &str = "datafusion_nexus.query_gpu_coverage.v1";
/// Schema version for the query GPU coverage JSON contract.
pub const QUERY_GPU_COVERAGE_SCHEMA_VERSION: u32 = 1;
/// SQL table function that exposes query GPU coverage validation.
pub const NEXUS_EXPLAIN_COVERAGE_FUNCTION: &str = "nexus_explain_coverage";
/// Schema version for the [`NEXUS_EXPLAIN_COVERAGE_FUNCTION`] SQL row contract.
pub const NEXUS_EXPLAIN_COVERAGE_SCHEMA_VERSION: u32 = 1;
pub(crate) const COVERAGE_VALIDATOR_RECURSION_REASON_CODE: &str = "coverage_validator_recursion_unsupported";
pub(crate) const GPU_COVERAGE_REJECTION_REASON_FACT: &str = "gpu_coverage_rejection_reason";

/// Validate a single SQL query against the native GPU planning path.
///
/// The validation uses an isolated retained native optimizer rule installed
/// into a derived [`SessionContext`]. It plans as-if [`GpuFallbackPolicy::GpuPreferred`]
/// to preserve fallback diagnostics, while recording the caller's actual policy
/// in [`QueryGpuCoverage::validated_under`].
///
/// `fallback_reasons` are retained per-candidate optimizer diagnostics, not an
/// alternative verdict. They can be non-empty even when the final optimized plan
/// has [`QueryGpuCoverageOverall::NoCpuInPlannedPath`].
///
/// `validated_under.catalog_snapshot` is a constant label for the cloned
/// planning state, not a catalog or table-state fingerprint. Do not compare
/// coverage verdicts across catalog, view, or table changes solely by this
/// field.
pub async fn validate_query(ctx: &SessionContext, sql: impl AsRef<str>) -> crate::Result<QueryGpuCoverage> {
    let sql = sql.as_ref();
    let config = optimizer_config_for_validation(ctx);
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(config.diagnostic_optimizer_config.clone()));
    let mut planning_state_builder =
        SessionStateBuilder::new_from_existing(ctx.state()).with_cudf_native_rule(Arc::clone(&native_rule));
    mark_coverage_validation_in_progress(&mut planning_state_builder);
    let planning_state = planning_state_builder.build();
    let validated_under = validated_under(&planning_state, &config);
    if let Some(rejection) = preflight_single_query(&planning_state, sql) {
        return Ok(QueryGpuCoverage::rejected(validated_under, rejection));
    }
    let planning_ctx = SessionContext::new_with_state(planning_state);

    let df = match planning_ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return Ok(QueryGpuCoverage::rejected(
                validated_under,
                QueryGpuRejection::from_datafusion_error(QueryGpuRejectionPhase::LogicalPlanning, &error),
            ));
        },
    };
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(error) => {
            let diagnostics = native_rule.last_planning_diagnostics();
            let phase = classify_physical_planning_error(&error);
            let planning_did_io = planning_did_io_for_physical_error(phase, diagnostics.as_ref());
            return Ok(QueryGpuCoverage::rejected_with_diagnostics(
                validated_under,
                QueryGpuRejection::from_datafusion_error(phase, &error),
                diagnostics.as_ref().map(|diagnostics| &diagnostics.report),
                planning_did_io,
            ));
        },
    };

    let diagnostics = native_rule.last_planning_diagnostics();
    let cudf_coverage = post_native_plan_coverage(plan.as_ref());
    let walk = walk_gpu_coverage(plan.as_ref());
    let fallback_reasons =
        diagnostics.as_ref().map(|diagnostics| fallback_reasons_from_report(&diagnostics.report)).unwrap_or_default();
    let planning_did_io = diagnostics.as_ref().is_some_and(planning_diagnostics_did_io);
    Ok(QueryGpuCoverage::planned(
        validated_under,
        plan.as_ref(),
        cudf_coverage.execution_path,
        walk,
        fallback_reasons,
        planning_did_io,
    ))
}

/// Structured answer for "will this query's planned path avoid DataFusion CPU execution?"
///
/// `fallback_reasons` are per-candidate planning diagnostics retained from the
/// native optimizer. They can be non-empty alongside
/// [`QueryGpuCoverageOverall::NoCpuInPlannedPath`] when rejected candidates were
/// not on the final planned path.
///
/// `validated_under.catalog_snapshot` is a constant label describing when the
/// session state was cloned for validation. It is not a catalog fingerprint, so
/// verdicts are not comparable across catalog, view, or table-state changes by
/// that value alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCoverage {
    pub overall: QueryGpuCoverageOverall,
    pub domains: QueryGpuCoverageDomains,
    pub fallback_reasons: Vec<QueryGpuFallbackReason>,
    pub cugraph_calls: Vec<QueryGpuCugraphCall>,
    pub runtime_caveats: Vec<QueryGpuRuntimeCaveat>,
    pub planning_did_io: bool,
    pub validated_under: QueryGpuValidatedUnder,
    pub confidence: QueryGpuCoverageConfidence,
    pub physical_plan_root: Option<String>,
    pub physical_plan_shape: Option<String>,
}

impl QueryGpuCoverage {
    fn planned(
        validated_under: QueryGpuValidatedUnder,
        plan: &dyn ExecutionPlan,
        cudf_native_path: NativeDataFusionPostNativeExecutionPath,
        walk: QueryGpuCoverageWalk,
        fallback_reasons: Vec<QueryGpuFallbackReason>,
        planning_did_io: bool,
    ) -> Self {
        let overall = if walk.has_gpu {
            if walk.all_gpu_covered && root_is_gpu_runtime_node(plan) {
                QueryGpuCoverageOverall::NoCpuInPlannedPath
            } else {
                QueryGpuCoverageOverall::PartialGpuPlanned {
                    gpu_fragments: walk.gpu_fragment_count,
                    cpu_boundaries: walk.cpu_boundary_count,
                }
            }
        } else {
            QueryGpuCoverageOverall::CpuOnly
        };
        let runtime_caveats = runtime_caveats(&walk, &overall, &validated_under);
        let domains = QueryGpuCoverageDomains {
            cudf_native_coverage: QueryGpuCudfNativeCoverage::from_post_native_path(cudf_native_path),
            cugraph_operator: if walk.cugraph_present {
                QueryGpuCugraphOperatorDomain::Present
            } else {
                QueryGpuCugraphOperatorDomain::Absent
            },
            datafusion_cpu_ingest_or_boundary: QueryGpuPresence::from_bool(walk.cpu_boundary_count > 0),
            host_output_materialization: QueryGpuPresence::from_bool(walk.host_output_materialization),
        };
        Self {
            overall,
            domains,
            fallback_reasons,
            cugraph_calls: walk.cugraph_calls,
            runtime_caveats,
            planning_did_io,
            validated_under,
            confidence: QueryGpuCoverageConfidence::PlanTimeOnly,
            physical_plan_root: Some(plan.name().to_string()),
            physical_plan_shape: Some(physical_plan_shape(plan)),
        }
    }

    fn rejected(validated_under: QueryGpuValidatedUnder, rejection: QueryGpuRejection) -> Self {
        Self::rejected_with_diagnostics(validated_under, rejection, None, false)
    }

    fn rejected_with_diagnostics(
        validated_under: QueryGpuValidatedUnder,
        rejection: QueryGpuRejection,
        report: Option<&PlanningReport>,
        planning_did_io: bool,
    ) -> Self {
        let overall = QueryGpuCoverageOverall::Rejected {
            phase: rejection.phase,
            reason_code: rejection.reason_code,
            retryable: rejection.retryable,
            detail: rejection.detail,
        };
        let runtime_caveats = runtime_caveats_for_overall(&overall, &validated_under, Vec::new());
        Self {
            overall,
            domains: QueryGpuCoverageDomains {
                cudf_native_coverage: QueryGpuCudfNativeCoverage::None,
                cugraph_operator: QueryGpuCugraphOperatorDomain::Absent,
                datafusion_cpu_ingest_or_boundary: QueryGpuPresence::Absent,
                host_output_materialization: QueryGpuPresence::Absent,
            },
            fallback_reasons: report.map(fallback_reasons_from_report).unwrap_or_default(),
            cugraph_calls: Vec::new(),
            runtime_caveats,
            planning_did_io,
            validated_under,
            confidence: QueryGpuCoverageConfidence::PlanTimeOnly,
            physical_plan_root: None,
            physical_plan_shape: None,
        }
    }

    /// Serialize the verdict to the stable v1 JSON shape.
    pub fn to_json(&self) -> Value {
        json!({
            "schema": QUERY_GPU_COVERAGE_JSON_SCHEMA,
            "schema_version": QUERY_GPU_COVERAGE_SCHEMA_VERSION,
            "overall": self.overall.to_json(),
            "domains": self.domains.to_json(),
            "fallback_reasons": self.fallback_reasons.iter().map(QueryGpuFallbackReason::to_json).collect::<Vec<_>>(),
            "cugraph_calls": self.cugraph_calls.iter().map(QueryGpuCugraphCall::to_json).collect::<Vec<_>>(),
            "runtime_caveats": self.runtime_caveats.iter().map(QueryGpuRuntimeCaveat::to_json).collect::<Vec<_>>(),
            "planning_did_io": self.planning_did_io,
            "validated_under": self.validated_under.to_json(),
            "confidence": self.confidence.as_str(),
            "physical_plan_root": self.physical_plan_root,
            "physical_plan_shape": self.physical_plan_shape,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryGpuCoverageOverall {
    NoCpuInPlannedPath,
    PartialGpuPlanned { gpu_fragments: usize, cpu_boundaries: usize },
    CpuOnly,
    Rejected { phase: QueryGpuRejectionPhase, reason_code: String, retryable: bool, detail: String },
}

impl QueryGpuCoverageOverall {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NoCpuInPlannedPath => "no_cpu_in_planned_path",
            Self::PartialGpuPlanned { .. } => "partial_gpu_planned",
            Self::CpuOnly => "cpu_only",
            Self::Rejected { .. } => "rejected",
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Self::NoCpuInPlannedPath => json!({
                "verdict": self.as_str(),
                "message": "No DataFusion CPU execution is present in the planned physical path inspected.",
                "runtime_gpu_admission_unknown": true,
            }),
            Self::PartialGpuPlanned { gpu_fragments, cpu_boundaries } => json!({
                "verdict": self.as_str(),
                "gpu_fragments": gpu_fragments,
                "cpu_boundaries": cpu_boundaries,
            }),
            Self::CpuOnly => json!({ "verdict": self.as_str() }),
            Self::Rejected { phase, reason_code, retryable, detail } => json!({
                "verdict": self.as_str(),
                "phase": phase.as_str(),
                "reason_code": reason_code,
                "retryable": retryable,
                "detail": detail,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuRejectionPhase {
    Parsing,
    LogicalPlanning,
    PhysicalPlanning,
    PlanningIo,
    NativeLowering,
}

impl QueryGpuRejectionPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parsing => "parsing",
            Self::LogicalPlanning => "logical_planning",
            Self::PhysicalPlanning => "physical_planning",
            Self::PlanningIo => "planning_io",
            Self::NativeLowering => "native_lowering",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCoverageDomains {
    pub cudf_native_coverage: QueryGpuCudfNativeCoverage,
    pub cugraph_operator: QueryGpuCugraphOperatorDomain,
    pub datafusion_cpu_ingest_or_boundary: QueryGpuPresence,
    pub host_output_materialization: QueryGpuPresence,
}

impl QueryGpuCoverageDomains {
    fn to_json(&self) -> Value {
        json!({
            "cudf_native_coverage": self.cudf_native_coverage.as_str(),
            "cugraph_operator": self.cugraph_operator.as_str(),
            "datafusion_cpu_ingest_or_boundary": self.datafusion_cpu_ingest_or_boundary.as_str(),
            "host_output_materialization": self.host_output_materialization.as_str(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuCudfNativeCoverage {
    Full,
    Partial,
    None,
}

impl QueryGpuCudfNativeCoverage {
    fn from_post_native_path(path: NativeDataFusionPostNativeExecutionPath) -> Self {
        match path {
            NativeDataFusionPostNativeExecutionPath::Native => Self::Full,
            NativeDataFusionPostNativeExecutionPath::PartialNative => Self::Partial,
            NativeDataFusionPostNativeExecutionPath::Cpu => Self::None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Partial => "partial",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuCugraphOperatorDomain {
    Present,
    Absent,
}

impl QueryGpuCugraphOperatorDomain {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Absent => "absent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuPresence {
    Present,
    Absent,
}

impl QueryGpuPresence {
    fn from_bool(present: bool) -> Self {
        if present { Self::Present } else { Self::Absent }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Absent => "absent",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuFallbackReason {
    pub reason_code: String,
    pub kind: QueryGpuFallbackReasonKind,
    pub detail: String,
    pub category: String,
}

impl QueryGpuFallbackReason {
    fn to_json(&self) -> Value {
        json!({
            "reason_code": self.reason_code,
            "kind": self.kind.as_str(),
            "detail": self.detail,
            "category": self.category,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuFallbackReasonKind {
    Operator,
    Expression,
    ArrowType,
    Source,
    PlanShape,
    Policy,
    Preference,
}

impl QueryGpuFallbackReasonKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Operator => "operator",
            Self::Expression => "expression",
            Self::ArrowType => "arrow_type",
            Self::Source => "source",
            Self::PlanShape => "plan_shape",
            Self::Policy => "policy",
            Self::Preference => "preference",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCugraphCall {
    pub algorithm: String,
    pub statically_valid: bool,
    pub notes: Vec<String>,
}

impl QueryGpuCugraphCall {
    fn to_json(&self) -> Value {
        json!({
            "algorithm": self.algorithm,
            "statically_valid": self.statically_valid,
            "notes": self.notes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuRuntimeCaveat {
    pub code: String,
    pub detail: String,
}

impl QueryGpuRuntimeCaveat {
    fn new(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { code: code.into(), detail: detail.into() }
    }

    fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "detail": self.detail,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuValidatedUnder {
    pub actual_policy: String,
    pub diagnostic_policy: String,
    pub optimizer_config_source: String,
    pub execution_mode: String,
    pub session_config_fingerprint: String,
    pub optimizer_fingerprint: String,
    pub catalog_snapshot: String,
}

impl QueryGpuValidatedUnder {
    fn to_json(&self) -> Value {
        json!({
            "actual_policy": self.actual_policy,
            "diagnostic_policy": self.diagnostic_policy,
            "optimizer_config_source": self.optimizer_config_source,
            "execution_mode": self.execution_mode,
            "session_config_fingerprint": self.session_config_fingerprint,
            "optimizer_fingerprint": self.optimizer_fingerprint,
            "catalog_snapshot": self.catalog_snapshot,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuCoverageConfidence {
    PlanTimeOnly,
}

impl QueryGpuCoverageConfidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PlanTimeOnly => "plan_time_only",
        }
    }
}

#[derive(Debug, Clone)]
struct ValidationOptimizerConfig {
    actual_policy: GpuFallbackPolicy,
    optimizer_config_source: &'static str,
    diagnostic_optimizer_config: CudfOptimizerConfig,
}

#[derive(Debug)]
struct QueryGpuCoverageValidationInProgress;

#[derive(Debug, Clone, PartialEq, Eq)]
struct QueryGpuRejection {
    phase: QueryGpuRejectionPhase,
    reason_code: String,
    retryable: bool,
    detail: String,
}

impl QueryGpuRejection {
    fn new(
        phase: QueryGpuRejectionPhase,
        reason_code: impl Into<String>,
        retryable: bool,
        detail: impl Into<String>,
    ) -> Self {
        Self { phase, reason_code: reason_code.into(), retryable, detail: detail.into() }
    }

    fn from_datafusion_error(phase: QueryGpuRejectionPhase, error: &DataFusionError) -> Self {
        if let Some(adapter_error) = find_adapter_error(error) {
            let reason_code = adapter_error
                .fact_value(GPU_COVERAGE_REJECTION_REASON_FACT)
                .or_else(|| adapter_error.fact_value("fallback_reason"))
                .or_else(|| adapter_error.fact_value("leaf_guard_first_rejection"))
                .unwrap_or_else(|| adapter_error.code().as_str())
                .to_string();
            let diagnostics = ErrorSourceDiagnostics::from_error(error);
            return Self::new(
                phase,
                reason_code,
                matches!(adapter_error.status(), crate::ErrorStatus::Temporary),
                diagnostics.chain().unwrap_or_else(|| adapter_error.message()).to_string(),
            );
        }
        let diagnostics = ErrorSourceDiagnostics::from_error(error);
        Self::new(
            phase,
            ErrorCode::DataFusionPlanning.as_str(),
            false,
            diagnostics.chain().map(ToString::to_string).unwrap_or_else(|| error.message().to_string()),
        )
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct QueryGpuCoverageWalk {
    has_gpu: bool,
    all_gpu_covered: bool,
    gpu_fragment_count: usize,
    cpu_boundary_count: usize,
    host_output_materialization: bool,
    cugraph_present: bool,
    cugraph_calls: Vec<QueryGpuCugraphCall>,
}

impl QueryGpuCoverageWalk {
    fn cpu_only() -> Self {
        Self { all_gpu_covered: true, ..Self::default() }
    }

    fn merge_child(&mut self, child: QueryGpuCoverageWalk) {
        self.has_gpu |= child.has_gpu;
        self.all_gpu_covered &= child.all_gpu_covered;
        self.gpu_fragment_count += child.gpu_fragment_count;
        self.cpu_boundary_count += child.cpu_boundary_count;
        self.host_output_materialization |= child.host_output_materialization;
        self.cugraph_present |= child.cugraph_present;
        self.cugraph_calls.extend(child.cugraph_calls);
    }
}

fn optimizer_config_for_validation(ctx: &SessionContext) -> ValidationOptimizerConfig {
    let (optimizer_config_source, optimizer_config) = installed_cudf_native_optimizer_config(&ctx.state())
        .map(|config| ("installed_session_rule", config))
        .unwrap_or_else(|| ("default_gpu_preferred", default_validation_optimizer_config()));
    let actual_policy = optimizer_config.fallback_policy();
    let diagnostic_optimizer_config = optimizer_config.with_fallback_policy(GpuFallbackPolicy::GpuPreferred);
    ValidationOptimizerConfig { actual_policy, optimizer_config_source, diagnostic_optimizer_config }
}

fn default_validation_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1))
}

fn validated_under(state: &SessionState, config: &ValidationOptimizerConfig) -> QueryGpuValidatedUnder {
    let rules = state.physical_optimizers();
    let rule_names = rules.iter().map(|rule| rule.name()).collect::<Vec<_>>().join(",");
    QueryGpuValidatedUnder {
        actual_policy: config.actual_policy.to_string(),
        diagnostic_policy: GpuFallbackPolicy::GpuPreferred.to_string(),
        optimizer_config_source: config.optimizer_config_source.to_string(),
        execution_mode: config.diagnostic_optimizer_config.native_pipeline_execution_mode().to_string(),
        session_config_fingerprint: fingerprint_debug(state.config_options()),
        optimizer_fingerprint: fingerprint_debug(&format!("{rule_names};{:?}", config.diagnostic_optimizer_config)),
        catalog_snapshot: "session_state_cloned_at_validation_call".to_string(),
    }
}

fn mark_coverage_validation_in_progress(builder: &mut SessionStateBuilder) {
    builder.config().get_or_insert_with(Default::default).set_extension(Arc::new(QueryGpuCoverageValidationInProgress));
}

pub(crate) fn gpu_coverage_validation_in_progress(state: &SessionState) -> bool {
    state.config().get_extension::<QueryGpuCoverageValidationInProgress>().is_some()
}

fn coverage_validator_recursion_rejection() -> QueryGpuRejection {
    QueryGpuRejection::new(
        QueryGpuRejectionPhase::Parsing,
        COVERAGE_VALIDATOR_RECURSION_REASON_CODE,
        false,
        format!("{NEXUS_EXPLAIN_COVERAGE_FUNCTION} cannot validate a query that calls itself"),
    )
}

pub(crate) fn coverage_validator_recursion_error() -> Error {
    Error::new(
        ErrorCode::DataFusionPlanning,
        "datafusion_nexus.gpu_coverage_sql.recursion",
        format!("{NEXUS_EXPLAIN_COVERAGE_FUNCTION} cannot validate a query that calls itself"),
    )
    .with_fact(GPU_COVERAGE_REJECTION_REASON_FACT, COVERAGE_VALIDATOR_RECURSION_REASON_CODE)
}

fn preflight_single_query(state: &SessionState, sql: &str) -> Option<QueryGpuRejection> {
    let options = state.config().options();
    let dialect_config = options.sql_parser.dialect;
    let Some(dialect) = dialect_from_str(dialect_config) else {
        return Some(QueryGpuRejection::new(
            QueryGpuRejectionPhase::Parsing,
            ErrorCode::DataFusionPlanning.as_str(),
            false,
            format!("unsupported SQL dialect configured for validation: {dialect_config}"),
        ));
    };
    let statements = match DFParserBuilder::new(sql)
        .with_dialect(dialect.as_ref())
        .with_recursion_limit(options.sql_parser.recursion_limit)
        .build()
        .and_then(|mut parser| parser.parse_statements())
    {
        Ok(statements) => statements,
        Err(error) => {
            return Some(QueryGpuRejection::from_datafusion_error(QueryGpuRejectionPhase::Parsing, &error));
        },
    };
    if statements.is_empty() {
        return Some(QueryGpuRejection::new(
            QueryGpuRejectionPhase::Parsing,
            "empty_sql",
            false,
            "validate_query expects a SQL query statement",
        ));
    }
    if statements.len() != 1 {
        return Some(QueryGpuRejection::new(
            QueryGpuRejectionPhase::Parsing,
            "multi_statement_unsupported",
            false,
            format!("validate_query expects exactly one query statement, got {}", statements.len()),
        ));
    }
    match statements.front() {
        Some(DfSqlStatement::Statement(statement)) => match statement.as_ref() {
            SqlStatement::Query(query) if query_mentions_coverage_table_function(query) => {
                Some(coverage_validator_recursion_rejection())
            },
            SqlStatement::Query(_) => None,
            _ => Some(QueryGpuRejection::new(
                QueryGpuRejectionPhase::Parsing,
                "non_query_statement_unsupported",
                false,
                "validate_query v1 supports SELECT/query statements only",
            )),
        },
        Some(_) => Some(QueryGpuRejection::new(
            QueryGpuRejectionPhase::Parsing,
            "non_query_statement_unsupported",
            false,
            "validate_query v1 supports SELECT/query statements only",
        )),
        None => unreachable!("empty statements are handled before statement matching"),
    }
}

fn query_mentions_coverage_table_function(query: &SqlQuery) -> bool {
    query_ast_mentions_coverage_table_function(query)
        || query
            .with
            .as_ref()
            .is_some_and(|with| with.cte_tables.iter().any(|cte| query_mentions_coverage_table_function(&cte.query)))
        || set_expr_mentions_coverage_table_function(&query.body)
        || query.order_by.as_ref().is_some_and(order_by_mentions_coverage_table_function)
        || query.limit_clause.as_ref().is_some_and(limit_clause_mentions_coverage_table_function)
        || query.fetch.as_ref().is_some_and(fetch_mentions_coverage_table_function)
        || query.pipe_operators.iter().any(pipe_operator_mentions_coverage_table_function)
}

fn query_ast_mentions_coverage_table_function(query: &SqlQuery) -> bool {
    matches!(query.visit(&mut CoverageTableFunctionVisitor), ControlFlow::Break(()))
}

struct CoverageTableFunctionVisitor;

impl Visitor for CoverageTableFunctionVisitor {
    type Break = ();

    fn pre_visit_table_factor(&mut self, table_factor: &TableFactor) -> ControlFlow<Self::Break> {
        let mentions = match table_factor {
            TableFactor::Table { name, args: Some(_), .. } | TableFactor::Function { name, .. } => {
                object_name_is_coverage_table_function(name)
            },
            _ => false,
        };
        if mentions { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    }

    fn pre_visit_expr(&mut self, expr: &SqlExpr) -> ControlFlow<Self::Break> {
        match expr {
            SqlExpr::Function(function) if object_name_is_coverage_table_function(&function.name) => {
                ControlFlow::Break(())
            },
            _ => ControlFlow::Continue(()),
        }
    }
}

fn set_expr_mentions_coverage_table_function(expr: &SetExpr) -> bool {
    match expr {
        SetExpr::Select(select) => select_mentions_coverage_table_function(select),
        SetExpr::Query(query) => query_mentions_coverage_table_function(query),
        SetExpr::SetOperation { left, right, .. } => {
            set_expr_mentions_coverage_table_function(left) || set_expr_mentions_coverage_table_function(right)
        },
        SetExpr::Values(_)
        | SetExpr::Insert(_)
        | SetExpr::Update(_)
        | SetExpr::Delete(_)
        | SetExpr::Merge(_)
        | SetExpr::Table(_) => false,
    }
}

fn select_mentions_coverage_table_function(select: &SqlSelect) -> bool {
    select.projection.iter().any(select_item_mentions_coverage_table_function)
        || select.from.iter().any(table_with_joins_mentions_coverage_table_function)
        || select.lateral_views.iter().any(|view| expr_mentions_coverage_table_function(&view.lateral_view))
        || select.prewhere.as_ref().is_some_and(expr_mentions_coverage_table_function)
        || select.selection.as_ref().is_some_and(expr_mentions_coverage_table_function)
        || group_by_mentions_coverage_table_function(&select.group_by)
        || select.cluster_by.iter().any(expr_mentions_coverage_table_function)
        || select.distribute_by.iter().any(expr_mentions_coverage_table_function)
        || select.sort_by.iter().any(order_by_expr_mentions_coverage_table_function)
        || select.having.as_ref().is_some_and(expr_mentions_coverage_table_function)
        || select.named_window.iter().any(named_window_definition_mentions_coverage_table_function)
        || select.qualify.as_ref().is_some_and(expr_mentions_coverage_table_function)
}

fn select_item_mentions_coverage_table_function(item: &SelectItem) -> bool {
    match item {
        SelectItem::UnnamedExpr(expr)
        | SelectItem::ExprWithAlias { expr, .. }
        | SelectItem::ExprWithAliases { expr, .. } => expr_mentions_coverage_table_function(expr),
        SelectItem::QualifiedWildcard(kind, _) => match kind {
            datafusion::sql::sqlparser::ast::SelectItemQualifiedWildcardKind::Expr(expr) => {
                expr_mentions_coverage_table_function(expr)
            },
            datafusion::sql::sqlparser::ast::SelectItemQualifiedWildcardKind::ObjectName(_) => false,
        },
        SelectItem::Wildcard(_) => false,
    }
}

fn table_with_joins_mentions_coverage_table_function(table: &TableWithJoins) -> bool {
    table_factor_mentions_coverage_table_function(&table.relation)
        || table.joins.iter().any(|join| {
            table_factor_mentions_coverage_table_function(&join.relation)
                || join_operator_mentions_coverage_table_function(&join.join_operator)
        })
}

fn table_factor_mentions_coverage_table_function(factor: &TableFactor) -> bool {
    match factor {
        TableFactor::Table { name, args, .. } => args.is_some() && object_name_is_coverage_table_function(name),
        TableFactor::Function { name, .. } => object_name_is_coverage_table_function(name),
        TableFactor::TableFunction { expr, .. } => expr_mentions_coverage_table_function(expr),
        TableFactor::Derived { subquery, .. } => query_mentions_coverage_table_function(subquery),
        TableFactor::UNNEST { array_exprs, .. } => array_exprs.iter().any(expr_mentions_coverage_table_function),
        TableFactor::NestedJoin { table_with_joins, .. } => {
            table_with_joins_mentions_coverage_table_function(table_with_joins)
        },
        TableFactor::Pivot { table, .. } | TableFactor::Unpivot { table, .. } => {
            table_factor_mentions_coverage_table_function(table)
        },
        _ => false,
    }
}

fn join_operator_mentions_coverage_table_function(operator: &JoinOperator) -> bool {
    match operator {
        JoinOperator::Join(constraint)
        | JoinOperator::Inner(constraint)
        | JoinOperator::Left(constraint)
        | JoinOperator::LeftOuter(constraint)
        | JoinOperator::Right(constraint)
        | JoinOperator::RightOuter(constraint)
        | JoinOperator::FullOuter(constraint)
        | JoinOperator::CrossJoin(constraint)
        | JoinOperator::Semi(constraint)
        | JoinOperator::LeftSemi(constraint)
        | JoinOperator::RightSemi(constraint)
        | JoinOperator::Anti(constraint)
        | JoinOperator::LeftAnti(constraint)
        | JoinOperator::RightAnti(constraint)
        | JoinOperator::StraightJoin(constraint) => join_constraint_mentions_coverage_table_function(constraint),
        JoinOperator::AsOf { match_condition, constraint } => {
            expr_mentions_coverage_table_function(match_condition)
                || join_constraint_mentions_coverage_table_function(constraint)
        },
        JoinOperator::CrossApply
        | JoinOperator::OuterApply
        | JoinOperator::ArrayJoin
        | JoinOperator::LeftArrayJoin
        | JoinOperator::InnerArrayJoin => false,
    }
}

fn pipe_operator_mentions_coverage_table_function(operator: &PipeOperator) -> bool {
    match operator {
        PipeOperator::Limit { expr, offset } => {
            expr_mentions_coverage_table_function(expr)
                || offset.as_ref().is_some_and(expr_mentions_coverage_table_function)
        },
        PipeOperator::Where { expr } => expr_mentions_coverage_table_function(expr),
        PipeOperator::OrderBy { exprs } => exprs.iter().any(order_by_expr_mentions_coverage_table_function),
        PipeOperator::Select { exprs } | PipeOperator::Extend { exprs } => {
            exprs.iter().any(select_item_mentions_coverage_table_function)
        },
        PipeOperator::Set { assignments } => assignments.iter().any(assignment_mentions_coverage_table_function),
        PipeOperator::Aggregate { full_table_exprs, group_by_expr } => {
            full_table_exprs.iter().any(expr_with_alias_and_order_by_mentions_coverage_table_function)
                || group_by_expr.iter().any(expr_with_alias_and_order_by_mentions_coverage_table_function)
        },
        PipeOperator::TableSample { sample } => table_sample_mentions_coverage_table_function(sample),
        PipeOperator::Union { queries, .. }
        | PipeOperator::Intersect { queries, .. }
        | PipeOperator::Except { queries, .. } => queries.iter().any(query_mentions_coverage_table_function),
        PipeOperator::Call { function, .. } => function_mentions_coverage_table_function(function),
        PipeOperator::Pivot { aggregate_functions, value_source, .. } => {
            aggregate_functions.iter().any(expr_with_alias_mentions_coverage_table_function)
                || pivot_value_source_mentions_coverage_table_function(value_source)
        },
        PipeOperator::Join(join) => join_mentions_coverage_table_function(join),
        PipeOperator::Drop { .. }
        | PipeOperator::As { .. }
        | PipeOperator::Rename { .. }
        | PipeOperator::Unpivot { .. } => false,
    }
}

fn assignment_mentions_coverage_table_function(assignment: &Assignment) -> bool {
    expr_mentions_coverage_table_function(&assignment.value)
}

fn expr_with_alias_mentions_coverage_table_function(expr: &ExprWithAlias) -> bool {
    expr_mentions_coverage_table_function(&expr.expr)
}

fn expr_with_alias_and_order_by_mentions_coverage_table_function(expr: &ExprWithAliasAndOrderBy) -> bool {
    expr_with_alias_mentions_coverage_table_function(&expr.expr)
}

fn table_sample_mentions_coverage_table_function(sample: &TableSample) -> bool {
    sample.quantity.as_ref().is_some_and(|quantity| expr_mentions_coverage_table_function(&quantity.value))
        || sample
            .bucket
            .as_ref()
            .and_then(|bucket| bucket.on.as_ref())
            .is_some_and(expr_mentions_coverage_table_function)
        || sample.offset.as_ref().is_some_and(expr_mentions_coverage_table_function)
}

fn pivot_value_source_mentions_coverage_table_function(value_source: &PivotValueSource) -> bool {
    match value_source {
        PivotValueSource::List(values) => values.iter().any(expr_with_alias_mentions_coverage_table_function),
        PivotValueSource::Any(order_by) => order_by.iter().any(order_by_expr_mentions_coverage_table_function),
        PivotValueSource::Subquery(query) => query_mentions_coverage_table_function(query),
    }
}

fn join_mentions_coverage_table_function(join: &SqlJoin) -> bool {
    table_factor_mentions_coverage_table_function(&join.relation)
        || join_operator_mentions_coverage_table_function(&join.join_operator)
}

fn join_constraint_mentions_coverage_table_function(constraint: &JoinConstraint) -> bool {
    match constraint {
        JoinConstraint::On(expr) => expr_mentions_coverage_table_function(expr),
        JoinConstraint::Using(_) | JoinConstraint::Natural | JoinConstraint::None => false,
    }
}

fn group_by_mentions_coverage_table_function(group_by: &GroupByExpr) -> bool {
    match group_by {
        GroupByExpr::All(_) => false,
        GroupByExpr::Expressions(exprs, _) => exprs.iter().any(expr_mentions_coverage_table_function),
    }
}

fn order_by_mentions_coverage_table_function(order_by: &datafusion::sql::sqlparser::ast::OrderBy) -> bool {
    match &order_by.kind {
        OrderByKind::All(_) => false,
        OrderByKind::Expressions(exprs) => exprs.iter().any(order_by_expr_mentions_coverage_table_function),
    }
}

fn order_by_expr_mentions_coverage_table_function(expr: &OrderByExpr) -> bool {
    expr_mentions_coverage_table_function(&expr.expr)
}

fn limit_clause_mentions_coverage_table_function(limit: &LimitClause) -> bool {
    match limit {
        LimitClause::LimitOffset { limit, offset, limit_by } => {
            limit.as_ref().is_some_and(expr_mentions_coverage_table_function)
                || offset.as_ref().is_some_and(|offset| expr_mentions_coverage_table_function(&offset.value))
                || limit_by.iter().any(expr_mentions_coverage_table_function)
        },
        LimitClause::OffsetCommaLimit { offset, limit } => {
            expr_mentions_coverage_table_function(offset) || expr_mentions_coverage_table_function(limit)
        },
    }
}

fn fetch_mentions_coverage_table_function(fetch: &Fetch) -> bool {
    fetch.quantity.as_ref().is_some_and(expr_mentions_coverage_table_function)
}

fn expr_mentions_coverage_table_function(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::InSubquery { expr, subquery, .. } => {
            expr_mentions_coverage_table_function(expr) || query_mentions_coverage_table_function(subquery)
        },
        SqlExpr::Exists { subquery, .. } | SqlExpr::Subquery(subquery) => {
            query_mentions_coverage_table_function(subquery)
        },
        SqlExpr::AnyOp { left, right, .. } | SqlExpr::AllOp { left, right, .. } => {
            expr_mentions_coverage_table_function(left) || expr_mentions_coverage_table_function(right)
        },
        SqlExpr::CompoundFieldAccess { root, access_chain } => {
            expr_mentions_coverage_table_function(root)
                || access_chain.iter().any(access_expr_mentions_coverage_table_function)
        },
        SqlExpr::BinaryOp { left, right, .. }
        | SqlExpr::IsDistinctFrom(left, right)
        | SqlExpr::IsNotDistinctFrom(left, right)
        | SqlExpr::AtTimeZone { timestamp: left, time_zone: right }
        | SqlExpr::Position { expr: left, r#in: right } => {
            expr_mentions_coverage_table_function(left) || expr_mentions_coverage_table_function(right)
        },
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsFalse(expr)
        | SqlExpr::IsNotFalse(expr)
        | SqlExpr::IsTrue(expr)
        | SqlExpr::IsNotTrue(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::IsUnknown(expr)
        | SqlExpr::IsNotUnknown(expr)
        | SqlExpr::IsNormalized { expr, .. }
        | SqlExpr::OuterJoin(expr)
        | SqlExpr::Prior(expr)
        | SqlExpr::Prefixed { value: expr, .. }
        | SqlExpr::Named { expr, .. }
        | SqlExpr::Collate { expr, .. }
        | SqlExpr::Cast { expr, .. }
        | SqlExpr::Extract { expr, .. }
        | SqlExpr::Ceil { expr, .. }
        | SqlExpr::Floor { expr, .. } => expr_mentions_coverage_table_function(expr),
        SqlExpr::Convert { expr, styles, .. } => {
            expr_mentions_coverage_table_function(expr) || styles.iter().any(expr_mentions_coverage_table_function)
        },
        SqlExpr::JsonAccess { value, .. } => expr_mentions_coverage_table_function(value),
        SqlExpr::InList { expr, list, .. } => {
            expr_mentions_coverage_table_function(expr) || list.iter().any(expr_mentions_coverage_table_function)
        },
        SqlExpr::InUnnest { expr, array_expr, .. } => {
            expr_mentions_coverage_table_function(expr) || expr_mentions_coverage_table_function(array_expr)
        },
        SqlExpr::Between { expr, low, high, .. } => {
            expr_mentions_coverage_table_function(expr)
                || expr_mentions_coverage_table_function(low)
                || expr_mentions_coverage_table_function(high)
        },
        SqlExpr::Like { expr, pattern, .. }
        | SqlExpr::ILike { expr, pattern, .. }
        | SqlExpr::SimilarTo { expr, pattern, .. }
        | SqlExpr::RLike { expr, pattern, .. } => {
            expr_mentions_coverage_table_function(expr) || expr_mentions_coverage_table_function(pattern)
        },
        SqlExpr::Substring { expr, substring_from, substring_for, .. } => {
            expr_mentions_coverage_table_function(expr)
                || substring_from.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
                || substring_for.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
        },
        SqlExpr::Trim { expr, trim_what, trim_characters, .. } => {
            expr_mentions_coverage_table_function(expr)
                || trim_what.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
                || trim_characters.as_ref().is_some_and(|exprs| exprs.iter().any(expr_mentions_coverage_table_function))
        },
        SqlExpr::Overlay { expr, overlay_what, overlay_from, overlay_for } => {
            expr_mentions_coverage_table_function(expr)
                || expr_mentions_coverage_table_function(overlay_what)
                || expr_mentions_coverage_table_function(overlay_from)
                || overlay_for.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
        },
        SqlExpr::Function(function) => function_mentions_coverage_table_function(function),
        SqlExpr::Case { operand, conditions, else_result, .. } => {
            operand.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
                || conditions.iter().any(|case| {
                    expr_mentions_coverage_table_function(&case.condition)
                        || expr_mentions_coverage_table_function(&case.result)
                })
                || else_result.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
        },
        SqlExpr::GroupingSets(sets) | SqlExpr::Cube(sets) | SqlExpr::Rollup(sets) => {
            sets.iter().flat_map(|set| set.iter()).any(expr_mentions_coverage_table_function)
        },
        SqlExpr::Tuple(exprs) => exprs.iter().any(expr_mentions_coverage_table_function),
        SqlExpr::Struct { values, .. } => values.iter().any(expr_mentions_coverage_table_function),
        SqlExpr::Dictionary(fields) => fields.iter().any(|field| expr_mentions_coverage_table_function(&field.value)),
        SqlExpr::Map(map) => map_mentions_coverage_table_function(map),
        SqlExpr::Array(array) => array_mentions_coverage_table_function(array),
        SqlExpr::Interval(interval) => expr_mentions_coverage_table_function(&interval.value),
        SqlExpr::Lambda(lambda) => expr_mentions_coverage_table_function(&lambda.body),
        SqlExpr::MemberOf(member_of) => {
            expr_mentions_coverage_table_function(&member_of.value)
                || expr_mentions_coverage_table_function(&member_of.array)
        },
        _ => false,
    }
}

fn access_expr_mentions_coverage_table_function(access: &AccessExpr) -> bool {
    match access {
        AccessExpr::Dot(expr) => expr_mentions_coverage_table_function(expr),
        AccessExpr::Subscript(subscript) => subscript_mentions_coverage_table_function(subscript),
    }
}

fn subscript_mentions_coverage_table_function(subscript: &Subscript) -> bool {
    match subscript {
        Subscript::Index { index } => expr_mentions_coverage_table_function(index),
        Subscript::Slice { lower_bound, upper_bound, stride } => {
            lower_bound.as_ref().is_some_and(expr_mentions_coverage_table_function)
                || upper_bound.as_ref().is_some_and(expr_mentions_coverage_table_function)
                || stride.as_ref().is_some_and(expr_mentions_coverage_table_function)
        },
    }
}

fn array_mentions_coverage_table_function(array: &SqlArray) -> bool {
    array.elem.iter().any(expr_mentions_coverage_table_function)
}

fn map_mentions_coverage_table_function(map: &SqlMap) -> bool {
    map.entries.iter().any(|entry| {
        expr_mentions_coverage_table_function(&entry.key) || expr_mentions_coverage_table_function(&entry.value)
    })
}

fn function_mentions_coverage_table_function(function: &SqlFunction) -> bool {
    object_name_is_coverage_table_function(&function.name)
        || function_args_mentions_coverage_table_function(&function.parameters)
        || function_args_mentions_coverage_table_function(&function.args)
        || function.filter.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
        || function.within_group.iter().any(order_by_expr_mentions_coverage_table_function)
        || function.over.as_ref().is_some_and(window_type_mentions_coverage_table_function)
}

fn function_args_mentions_coverage_table_function(args: &FunctionArguments) -> bool {
    match args {
        FunctionArguments::None => false,
        FunctionArguments::Subquery(query) => query_mentions_coverage_table_function(query),
        FunctionArguments::List(list) => {
            list.args.iter().any(function_arg_mentions_coverage_table_function)
                || list.clauses.iter().any(function_arg_clause_mentions_coverage_table_function)
        },
    }
}

fn function_arg_mentions_coverage_table_function(arg: &FunctionArg) -> bool {
    match arg {
        FunctionArg::Named { arg, .. } => function_arg_expr_mentions_coverage_table_function(arg),
        FunctionArg::ExprNamed { name, arg, .. } => {
            expr_mentions_coverage_table_function(name) || function_arg_expr_mentions_coverage_table_function(arg)
        },
        FunctionArg::Unnamed(arg) => function_arg_expr_mentions_coverage_table_function(arg),
    }
}

fn function_arg_expr_mentions_coverage_table_function(arg: &FunctionArgExpr) -> bool {
    match arg {
        FunctionArgExpr::Expr(expr) => expr_mentions_coverage_table_function(expr),
        FunctionArgExpr::QualifiedWildcard(_) | FunctionArgExpr::Wildcard | FunctionArgExpr::WildcardWithOptions(_) => {
            false
        },
    }
}

fn function_arg_clause_mentions_coverage_table_function(clause: &FunctionArgumentClause) -> bool {
    match clause {
        FunctionArgumentClause::OrderBy(exprs) => exprs.iter().any(order_by_expr_mentions_coverage_table_function),
        FunctionArgumentClause::Limit(expr) => expr_mentions_coverage_table_function(expr),
        FunctionArgumentClause::Having(bound) => expr_mentions_coverage_table_function(&bound.1),
        FunctionArgumentClause::IgnoreOrRespectNulls(_)
        | FunctionArgumentClause::OnOverflow(_)
        | FunctionArgumentClause::Separator(_)
        | FunctionArgumentClause::JsonNullClause(_)
        | FunctionArgumentClause::JsonReturningClause(_) => false,
    }
}

fn window_type_mentions_coverage_table_function(window: &WindowType) -> bool {
    match window {
        WindowType::NamedWindow(_) => false,
        WindowType::WindowSpec(spec) => window_spec_mentions_coverage_table_function(spec),
    }
}

fn named_window_definition_mentions_coverage_table_function(
    definition: &datafusion::sql::sqlparser::ast::NamedWindowDefinition,
) -> bool {
    match &definition.1 {
        datafusion::sql::sqlparser::ast::NamedWindowExpr::NamedWindow(_) => false,
        datafusion::sql::sqlparser::ast::NamedWindowExpr::WindowSpec(spec) => {
            window_spec_mentions_coverage_table_function(spec)
        },
    }
}

fn window_spec_mentions_coverage_table_function(spec: &WindowSpec) -> bool {
    spec.partition_by.iter().any(expr_mentions_coverage_table_function)
        || spec.order_by.iter().any(order_by_expr_mentions_coverage_table_function)
}

fn object_name_is_coverage_table_function(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(ObjectNamePart::as_ident)
        .is_some_and(|ident| ident.value.eq_ignore_ascii_case(NEXUS_EXPLAIN_COVERAGE_FUNCTION))
}

fn classify_physical_planning_error(error: &DataFusionError) -> QueryGpuRejectionPhase {
    let Some(adapter_error) = find_adapter_error(error) else {
        return QueryGpuRejectionPhase::PhysicalPlanning;
    };
    match adapter_error.code() {
        ErrorCode::PlanningTimeout
        | ErrorCode::SourceAccessDenied
        | ErrorCode::SourceUnauthenticated
        | ErrorCode::SourceObjectNotFound
        | ErrorCode::SourceTemporaryFailure => QueryGpuRejectionPhase::PlanningIo,
        ErrorCode::NativeCandidateRejected => QueryGpuRejectionPhase::NativeLowering,
        _ if adapter_error
            .fact_value("fallback_reason")
            .is_some_and(|reason| reason.starts_with("iceberg_") || reason == "table_format_planning_failed") =>
        {
            QueryGpuRejectionPhase::PlanningIo
        },
        _ => QueryGpuRejectionPhase::PhysicalPlanning,
    }
}

fn planning_did_io_for_physical_error(
    phase: QueryGpuRejectionPhase,
    diagnostics: Option<&crate::NativePlanningDiagnostics>,
) -> bool {
    diagnostics.is_some_and(planning_diagnostics_did_io) || phase == QueryGpuRejectionPhase::PlanningIo
}

fn walk_gpu_coverage(plan: &dyn ExecutionPlan) -> QueryGpuCoverageWalk {
    if plan.downcast_ref::<NativeCudfExec>().is_some() {
        return QueryGpuCoverageWalk {
            has_gpu: true,
            all_gpu_covered: true,
            gpu_fragment_count: 1,
            cpu_boundary_count: 0,
            host_output_materialization: true,
            cugraph_present: false,
            cugraph_calls: Vec::new(),
        };
    }
    if plan.downcast_ref::<NativeCudfGpuProducerExec>().is_some() {
        return QueryGpuCoverageWalk {
            has_gpu: true,
            all_gpu_covered: true,
            gpu_fragment_count: 1,
            cpu_boundary_count: 0,
            host_output_materialization: false,
            cugraph_present: false,
            cugraph_calls: Vec::new(),
        };
    }
    if plan.downcast_ref::<NativeCudfComposedExec>().is_some() {
        let mut walk = QueryGpuCoverageWalk {
            has_gpu: true,
            all_gpu_covered: true,
            gpu_fragment_count: 1,
            cpu_boundary_count: 0,
            host_output_materialization: true,
            cugraph_present: false,
            cugraph_calls: Vec::new(),
        };
        for child in plan.children() {
            let child_walk = walk_gpu_coverage(child.as_ref());
            walk.merge_child(child_walk);
        }
        return walk;
    }
    walk_cugraph_or_generic(plan)
}

#[cfg(feature = "cugraph")]
fn walk_cugraph_or_generic(plan: &dyn ExecutionPlan) -> QueryGpuCoverageWalk {
    if let Some(cugraph) = plan.downcast_ref::<CugraphAlgoExec>() {
        let mut walk = QueryGpuCoverageWalk {
            has_gpu: true,
            all_gpu_covered: true,
            gpu_fragment_count: 1,
            cpu_boundary_count: 0,
            host_output_materialization: true,
            cugraph_present: true,
            cugraph_calls: vec![QueryGpuCugraphCall {
                algorithm: cugraph.coverage_algorithm_name().to_string(),
                statically_valid: true,
                notes: vec![
                    "physical cuGraph plan constructed".to_string(),
                    "runtime seed, personalization, and device-side validation are not decided at plan time"
                        .to_string(),
                ],
            }],
        };
        for child in plan.children() {
            let child_walk = walk_gpu_coverage(child.as_ref());
            if !cugraph_child_stays_gpu_resident(child.as_ref()) {
                walk.cpu_boundary_count += 1;
                walk.all_gpu_covered = false;
            }
            walk.merge_child(child_walk);
        }
        return walk;
    }
    walk_generic_datafusion(plan)
}

#[cfg(not(feature = "cugraph"))]
fn walk_cugraph_or_generic(plan: &dyn ExecutionPlan) -> QueryGpuCoverageWalk {
    walk_generic_datafusion(plan)
}

fn walk_generic_datafusion(plan: &dyn ExecutionPlan) -> QueryGpuCoverageWalk {
    let mut walk = QueryGpuCoverageWalk::cpu_only();
    let mut direct_gpu_child = false;
    for child in plan.children() {
        let child_walk = walk_gpu_coverage(child.as_ref());
        direct_gpu_child |= child_walk.has_gpu;
        walk.merge_child(child_walk);
    }
    if walk.has_gpu {
        walk.all_gpu_covered = false;
        if direct_gpu_child {
            walk.cpu_boundary_count += 1;
        }
    }
    walk
}

#[cfg(feature = "cugraph")]
fn cugraph_child_stays_gpu_resident(plan: &dyn ExecutionPlan) -> bool {
    if plan.downcast_ref::<NativeCudfExec>().is_some() || plan.downcast_ref::<NativeCudfComposedExec>().is_some() {
        return true;
    }
    if plan.name() == "CooperativeExec" {
        let children = plan.children();
        return matches!(
            children.as_slice(),
            [child]
                if child.downcast_ref::<NativeCudfExec>().is_some()
                    || child.downcast_ref::<NativeCudfComposedExec>().is_some()
        );
    }
    false
}

fn root_is_gpu_runtime_node(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<NativeCudfExec>().is_some()
        || plan.downcast_ref::<NativeCudfComposedExec>().is_some()
        || plan.downcast_ref::<NativeCudfGpuProducerExec>().is_some()
        || root_is_cugraph_node(plan)
}

#[cfg(feature = "cugraph")]
fn root_is_cugraph_node(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<CugraphAlgoExec>().is_some()
}

#[cfg(not(feature = "cugraph"))]
fn root_is_cugraph_node(_plan: &dyn ExecutionPlan) -> bool {
    false
}

fn fallback_reasons_from_report(report: &PlanningReport) -> Vec<QueryGpuFallbackReason> {
    report
        .outcomes()
        .iter()
        .filter_map(|outcome| match outcome {
            CandidateOutcome::Rejected { candidate, reason } => Some(QueryGpuFallbackReason {
                reason_code: reason.code().to_string(),
                kind: fallback_reason_kind(*reason),
                detail: candidate.shape.clone(),
                category: reason.triage_code().as_str().to_string(),
            }),
            CandidateOutcome::Rewritten { .. } => None,
        })
        .collect()
}

fn fallback_reason_kind(reason: FallbackReason) -> QueryGpuFallbackReasonKind {
    match reason {
        FallbackReason::UnsupportedExpression => QueryGpuFallbackReasonKind::Expression,
        FallbackReason::UnsupportedType => QueryGpuFallbackReasonKind::ArrowType,
        FallbackReason::ExplicitCpuPolicy => QueryGpuFallbackReasonKind::Policy,
        FallbackReason::NativeSmallSelectiveTopkCpuPreferred
        | FallbackReason::NativeSmallSelectiveResidualJoinCpuPreferred
        | FallbackReason::NativeSmallSelectiveFullSortCpuPreferred => QueryGpuFallbackReasonKind::Preference,
        FallbackReason::NativeSourceNonLocalObjectStore
        | FallbackReason::NativeSourcePartitionColumns
        | FallbackReason::NativeSourceExprAdapter
        | FallbackReason::NativeSourceParquetFileReaderFactory
        | FallbackReason::NativeSourceScanLimit
        | FallbackReason::NativeSourcePreserveOrder
        | FallbackReason::NativeSourceOutputOrdering
        | FallbackReason::NativeSourcePushedDownParquetPredicate
        | FallbackReason::NativeSourceEmpty
        | FallbackReason::NativeSourcePartitionValues
        | FallbackReason::NativeSourceByteRange
        | FallbackReason::NativeSourceFileOrdering
        | FallbackReason::NativeSourceRowSelection
        | FallbackReason::NativeSourceSchemaMapping
        | FallbackReason::TableFormatUnsupported
        | FallbackReason::TableFormatPlanningFailed
        | FallbackReason::IcebergNonParquetDataFile
        | FallbackReason::IcebergDeleteFilesUnsupported
        | FallbackReason::IcebergFieldIdMappingUnsupported
        | FallbackReason::IcebergNameMappingUnsupported
        | FallbackReason::IcebergNestedProjectionUnsupported
        | FallbackReason::IcebergPartitionTransformUnsupported
        | FallbackReason::IcebergResidualPredicateUnsupported
        | FallbackReason::IcebergRowSelectionUnsupported
        | FallbackReason::IcebergObjectStoreReadUnsupported
        | FallbackReason::IcebergSnapshotUnsupported => QueryGpuFallbackReasonKind::Source,
        FallbackReason::NativeSortPreservingMergeUnsupported
        | FallbackReason::NativeDistributedAggregateUnsupported
        | FallbackReason::NativePartitionPreservingSortUnsupported
        | FallbackReason::NativeBoundedJoinUnsupported
        | FallbackReason::NativeAdaptivePipelineUnsupported
        | FallbackReason::NativeOrderedOutputUnsupported
        | FallbackReason::TableFormatGpuOnlyLeafNotRewritten => QueryGpuFallbackReasonKind::PlanShape,
        FallbackReason::UnsupportedOperator => QueryGpuFallbackReasonKind::Operator,
    }
}

fn runtime_caveats(
    walk: &QueryGpuCoverageWalk,
    overall: &QueryGpuCoverageOverall,
    validated_under: &QueryGpuValidatedUnder,
) -> Vec<QueryGpuRuntimeCaveat> {
    if !walk.has_gpu {
        return runtime_caveats_for_overall(overall, validated_under, Vec::new());
    }
    let mut caveats = vec![
        QueryGpuRuntimeCaveat::new(
            "runtime_gpu_admission_unknown",
            "query-service grants are acquired at execution time and can wait, shrink, or fail under concurrency",
        ),
        QueryGpuRuntimeCaveat::new(
            "device_memory_runtime_unknown",
            "device allocation, reservation, and retry outcomes are data- and concurrency-dependent",
        ),
        QueryGpuRuntimeCaveat::new(
            "native_runtime_failure_has_no_cpu_rescue",
            "once native GPU execution starts, native failures are surfaced as failures rather than CPU fallback",
        ),
    ];
    if walk.host_output_materialization {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "host_output_materialization",
            "DataFusion-facing query output is materialized back to host Arrow outside the GPU-execution promise",
        ));
    }
    if walk.cugraph_present {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "cugraph_runtime_validation_unknown",
            "cuGraph seed, personalization, weight, and device-side checks can still fail at runtime",
        ));
    }
    caveats.push(QueryGpuRuntimeCaveat::new(
        "execution_mode_plan_time_only",
        format!(
            "validated under native pipeline execution mode {}; strict/adaptive runtime gates still execute later",
            validated_under.execution_mode
        ),
    ));
    runtime_caveats_for_overall(overall, validated_under, caveats)
}

fn runtime_caveats_for_overall(
    overall: &QueryGpuCoverageOverall,
    validated_under: &QueryGpuValidatedUnder,
    mut caveats: Vec<QueryGpuRuntimeCaveat>,
) -> Vec<QueryGpuRuntimeCaveat> {
    if validated_under.actual_policy == GpuFallbackPolicy::GpuOnly.to_string()
        && !matches!(overall, QueryGpuCoverageOverall::NoCpuInPlannedPath)
    {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "gpu_only_policy_would_error",
            "the caller's GpuOnly policy treats non-full-GPU validation outcomes as hard planning failures; validation plans diagnostically under GpuPreferred to retain fallback evidence",
        ));
    }
    caveats
}

fn planning_diagnostics_did_io(diagnostics: &crate::NativePlanningDiagnostics) -> bool {
    diagnostics.input_plan_root.contains("CudfTableFormatScanExec")
        || diagnostics.input_plan_shape.contains("CudfTableFormatScanExec")
        || diagnostics.report.outcomes().iter().any(|outcome| match outcome {
            CandidateOutcome::Rejected { reason, .. } => fallback_reason_is_table_format(*reason),
            CandidateOutcome::Rewritten { .. } => false,
        })
}

fn fallback_reason_is_table_format(reason: FallbackReason) -> bool {
    matches!(
        reason,
        FallbackReason::TableFormatUnsupported
            | FallbackReason::TableFormatPlanningFailed
            | FallbackReason::IcebergNonParquetDataFile
            | FallbackReason::IcebergDeleteFilesUnsupported
            | FallbackReason::IcebergFieldIdMappingUnsupported
            | FallbackReason::IcebergNameMappingUnsupported
            | FallbackReason::IcebergNestedProjectionUnsupported
            | FallbackReason::IcebergPartitionTransformUnsupported
            | FallbackReason::IcebergResidualPredicateUnsupported
            | FallbackReason::IcebergRowSelectionUnsupported
            | FallbackReason::IcebergObjectStoreReadUnsupported
            | FallbackReason::IcebergSnapshotUnsupported
            | FallbackReason::TableFormatGpuOnlyLeafNotRewritten
    )
}

fn find_adapter_error<'a>(error: &'a (dyn StdError + 'static)) -> Option<&'a Error> {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(adapter) = error.downcast_ref::<Error>() {
            return Some(adapter);
        }
        current = error.source();
    }
    None
}

fn fingerprint_debug(value: &impl std::fmt::Debug) -> String {
    let mut hasher = DefaultHasher::new();
    format!("{value:?}").hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::{
        physical_expr::expressions::col,
        physical_plan::projection::{ProjectionExec, ProjectionExpr},
        prelude::SessionConfig,
    };
    use nexus_query_engine::{LocalParquetSource, QueryPlanBuilder, SourceNode};

    use super::*;

    fn test_execution_config() -> NativeCudfExecutionConfig {
        NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1)
    }

    fn unbounded_optimizer_config(policy: GpuFallbackPolicy) -> CudfOptimizerConfig {
        CudfOptimizerConfig::new(policy).with_native_execution_config(NativeCudfExecutionConfig::unbounded())
    }

    fn validation_context(policy: GpuFallbackPolicy) -> SessionContext {
        let state = SessionStateBuilder::new_with_default_features()
            .with_cudf_native(unbounded_optimizer_config(policy))
            .build();
        SessionContext::new_with_state(state)
    }

    fn one_column_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]))
    }

    fn native_exec_leaf() -> Arc<dyn ExecutionPlan> {
        let schema = one_column_schema();
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/query-gpu-coverage-not-read.parquet",
            schema,
            vec!["key".to_string()],
        )));
        Arc::new(NativeCudfExec::try_new(builder.finish_with_host_sink(source), test_execution_config()).unwrap())
    }

    #[cfg(feature = "cugraph")]
    fn composed_exec_with_child(
        child: Arc<dyn ExecutionPlan>,
        input_name: impl Into<String>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = one_column_schema();
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(nexus_query_engine::GpuResidentSource::new(input_name, schema)));
        Arc::new(
            NativeCudfComposedExec::try_new(
                builder.finish_with_host_sink(source),
                vec![child],
                test_execution_config(),
            )
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn validate_query_rejects_multiple_statements_but_not_semicolon_literal() {
        let state = SessionStateBuilder::new()
            .with_cudf_native(
                CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
                    .with_native_execution_config(test_execution_config()),
            )
            .build();
        let ctx = SessionContext::new_with_state(state);

        let literal = validate_query(&ctx, "SELECT ';' AS marker").await.unwrap();
        assert!(!matches!(literal.overall, QueryGpuCoverageOverall::Rejected { .. }));

        let literal_function_name =
            validate_query(&ctx, "SELECT 'nexus_explain_coverage(''SELECT 1'')' AS marker").await.unwrap();
        assert!(!matches!(literal_function_name.overall, QueryGpuCoverageOverall::Rejected { .. }));

        let empty = validate_query(&ctx, "").await.unwrap();
        assert!(matches!(
            empty.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
                if reason_code == "empty_sql"
        ));

        let rejected = validate_query(&ctx, "SELECT ';' AS marker; SELECT 1").await.unwrap();
        assert!(matches!(
            rejected.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
                if reason_code == "multi_statement_unsupported"
        ));

        let recursive_exists =
            validate_query(&ctx, "SELECT EXISTS (SELECT 1 FROM nexus_explain_coverage('SELECT 1')) AS recursive")
                .await
                .unwrap();
        assert!(matches!(
            recursive_exists.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
                if reason_code == "coverage_validator_recursion_unsupported"
        ));

        let recursive_array = validate_query(
            &ctx,
            "SELECT ARRAY[(SELECT count(*) FROM nexus_explain_coverage('SELECT 1'))] AS recursive",
        )
        .await
        .unwrap();
        assert!(matches!(
            recursive_array.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
                if reason_code == "coverage_validator_recursion_unsupported"
        ));

        let recursive_pipe_where =
            validate_query(&ctx, "SELECT 1 |> WHERE EXISTS (SELECT 1 FROM nexus_explain_coverage('SELECT 1'))")
                .await
                .unwrap();
        assert!(matches!(
            recursive_pipe_where.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
                if reason_code == "coverage_validator_recursion_unsupported"
        ));

        let recursive_pipe_select = validate_query(
            &ctx,
            "SELECT 1 |> SELECT (SELECT count(*) FROM nexus_explain_coverage('SELECT 1')) AS recursive",
        )
        .await
        .unwrap();
        assert!(matches!(
            recursive_pipe_select.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
                if reason_code == "coverage_validator_recursion_unsupported"
        ));

        let recursive_unnest = validate_query(
            &ctx,
            "SELECT * FROM UNNEST(ARRAY[(SELECT count(*) FROM nexus_explain_coverage('SELECT 1'))])",
        )
        .await
        .unwrap();
        assert!(matches!(
            recursive_unnest.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
                if reason_code == "coverage_validator_recursion_unsupported"
        ));

        let recursive_named_window = validate_query(
            &ctx,
            "SELECT row_number() OVER w AS rn \
             FROM (SELECT 1 AS x) t \
             WINDOW w AS (PARTITION BY (SELECT count(*) FROM nexus_explain_coverage('SELECT 1')))",
        )
        .await
        .unwrap();
        assert!(matches!(
            recursive_named_window.overall,
            QueryGpuCoverageOverall::Rejected { phase: QueryGpuRejectionPhase::Parsing, ref reason_code, .. }
                if reason_code == "coverage_validator_recursion_unsupported"
        ));
    }

    #[tokio::test]
    async fn validate_query_preflight_uses_session_sql_dialect() {
        let state = SessionStateBuilder::new_with_default_features()
            .with_config(SessionConfig::new().set_str("datafusion.sql_parser.dialect", "postgresql"))
            .with_cudf_native(
                CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
                    .with_native_execution_config(test_execution_config()),
            )
            .build();

        let rejection = preflight_single_query(&state, "SELECT $$nexus_explain_coverage('SELECT 1')$$ AS marker");
        assert!(rejection.is_none(), "postgresql dollar-quoted literal should not be rejected: {rejection:?}");
    }

    #[tokio::test]
    async fn validate_query_reports_cpu_only_and_partial_fallback_axes() {
        let ctx = validation_context(GpuFallbackPolicy::GpuPreferred);

        let cpu = validate_query(&ctx, "SELECT random() AS r").await.unwrap();
        assert_eq!(cpu.overall, QueryGpuCoverageOverall::CpuOnly);
        assert_eq!(cpu.domains.cudf_native_coverage, QueryGpuCudfNativeCoverage::None);
        assert!(
            cpu.fallback_reasons
                .iter()
                .any(|reason| reason.reason_code == FallbackReason::UnsupportedExpression.code()
                    && reason.kind == QueryGpuFallbackReasonKind::Expression)
        );

        let partial = validate_query(&ctx, "SELECT array[1, 2] AS values").await.unwrap();
        assert!(matches!(partial.overall, QueryGpuCoverageOverall::PartialGpuPlanned { .. }));
        assert_eq!(partial.domains.cudf_native_coverage, QueryGpuCudfNativeCoverage::Partial);
        assert_eq!(partial.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Present);
        assert!(
            partial.fallback_reasons.iter().any(|reason| reason.reason_code == FallbackReason::UnsupportedType.code()
                && reason.kind == QueryGpuFallbackReasonKind::ArrowType)
        );
    }

    #[tokio::test]
    async fn validate_query_stamps_actual_policy_but_uses_diagnostic_gpu_preferred() {
        let cpu_only_ctx = validation_context(GpuFallbackPolicy::CpuOnly);
        let cpu_only = validate_query(&cpu_only_ctx, "SELECT 1").await.unwrap();
        assert_eq!(cpu_only.validated_under.actual_policy, GpuFallbackPolicy::CpuOnly.to_string());
        assert_eq!(cpu_only.validated_under.diagnostic_policy, GpuFallbackPolicy::GpuPreferred.to_string());
        assert_eq!(cpu_only.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);

        let gpu_only_ctx = validation_context(GpuFallbackPolicy::GpuOnly);
        let gpu_only = validate_query(&gpu_only_ctx, "SELECT random() AS r").await.unwrap();
        assert_eq!(gpu_only.validated_under.actual_policy, GpuFallbackPolicy::GpuOnly.to_string());
        assert_eq!(gpu_only.validated_under.diagnostic_policy, GpuFallbackPolicy::GpuPreferred.to_string());
        assert_eq!(gpu_only.overall, QueryGpuCoverageOverall::CpuOnly);
        assert!(gpu_only.runtime_caveats.iter().any(|caveat| caveat.code == "gpu_only_policy_would_error"));
    }

    #[tokio::test]
    async fn validate_query_concurrent_calls_do_not_share_retained_rule_state() {
        let ctx = validation_context(GpuFallbackPolicy::GpuPreferred);

        let (native, cpu) =
            tokio::join!(validate_query(&ctx, "SELECT 1 AS value"), validate_query(&ctx, "SELECT random() AS r"));
        let native = native.unwrap();
        let cpu = cpu.unwrap();

        assert_eq!(native.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
        assert!(native.fallback_reasons.is_empty());
        assert_eq!(cpu.overall, QueryGpuCoverageOverall::CpuOnly);
        assert!(cpu.fallback_reasons.iter().any(|reason| reason.reason_code == "unsupported_expression"));
    }

    #[test]
    fn planning_io_phase_marks_planning_did_io_without_retained_diagnostics() {
        let error = DataFusionError::External(Box::new(Error::new(
            ErrorCode::SourceTemporaryFailure,
            "datafusion_nexus.test",
            "temporary source failure",
        )));
        let phase = classify_physical_planning_error(&error);

        assert_eq!(phase, QueryGpuRejectionPhase::PlanningIo);
        assert!(planning_did_io_for_physical_error(phase, None));
        assert!(!planning_did_io_for_physical_error(QueryGpuRejectionPhase::PhysicalPlanning, None));
    }

    #[test]
    fn generic_cpu_parent_over_native_fragment_is_partial() {
        let child = native_exec_leaf();
        let schema = child.schema();
        let plan = Arc::new(
            ProjectionExec::try_new(
                [ProjectionExpr { expr: col("key", &schema).unwrap(), alias: "key".to_string() }],
                child,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let walk = walk_gpu_coverage(plan.as_ref());
        let coverage = QueryGpuCoverage::planned(
            validated_under_for_tests(),
            plan.as_ref(),
            NativeDataFusionPostNativeExecutionPath::PartialNative,
            walk,
            Vec::new(),
            false,
        );

        assert!(matches!(coverage.overall, QueryGpuCoverageOverall::PartialGpuPlanned { .. }));
        assert_eq!(coverage.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Present);
    }

    #[cfg(feature = "cugraph")]
    mod cugraph {
        use arrow::{
            array::{ArrayRef, Int64Array},
            record_batch::RecordBatch,
        };
        use datafusion::{datasource::memory::MemorySourceConfig, physical_plan::coop::CooperativeExec};

        use super::*;
        use crate::graph_sql::{
            CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphGraphConstructionPolicy,
            CugraphGraphOptions, CugraphTableReference, CugraphTriangleCountAllParams, result_schema_for_params,
        };

        fn edge_schema() -> Arc<Schema> {
            Arc::new(Schema::new(vec![
                Field::new("src", DataType::Int64, false),
                Field::new("dst", DataType::Int64, false),
            ]))
        }

        fn memory_edge_source() -> Arc<dyn ExecutionPlan> {
            let schema = edge_schema();
            let src = Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef;
            let dst = Arc::new(Int64Array::from(vec![2_i64, 3])) as ArrayRef;
            let batch = RecordBatch::try_new(Arc::clone(&schema), vec![src, dst]).unwrap();
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
        }

        fn cugraph_spec() -> CugraphAlgorithmSpec {
            let params = CugraphAlgorithmParams::TriangleCountAll(CugraphTriangleCountAllParams);
            CugraphAlgorithmSpec {
                algorithm: CugraphAlgorithm::TriangleCountAll,
                table_name: CugraphTableReference::parse("edges").unwrap(),
                src_col: "src".to_string(),
                dst_col: "dst".to_string(),
                weight_col: None,
                edge_id_col: None,
                graph_options: CugraphGraphOptions {
                    directed: false,
                    renumber: true,
                    construction_policy: CugraphGraphConstructionPolicy::PythonCugraph,
                },
                output_schema: result_schema_for_params(CugraphAlgorithm::TriangleCountAll, &params),
                params,
            }
        }

        fn cugraph_exec(edge_source: Arc<dyn ExecutionPlan>) -> Arc<CugraphAlgoExec> {
            Arc::new(
                CugraphAlgoExec::try_new(cugraph_spec(), edge_source, Vec::new(), None, None, test_execution_config())
                    .unwrap(),
            )
        }

        #[test]
        fn cugraph_child_residency_matches_runtime_one_level_cooperative_unwrap() {
            let single_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_exec_leaf()));
            assert!(cugraph_child_stays_gpu_resident(single_wrapped.as_ref()));

            let inner_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(native_exec_leaf()));
            let nested_wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(inner_wrapped));
            assert!(!cugraph_child_stays_gpu_resident(nested_wrapped.as_ref()));
        }

        #[test]
        fn bare_cugraph_with_native_child_is_no_cpu_even_when_cudf_axis_is_none() {
            let plan: Arc<dyn ExecutionPlan> = cugraph_exec(native_exec_leaf());
            let walk = walk_gpu_coverage(plan.as_ref());
            let coverage = QueryGpuCoverage::planned(
                validated_under_for_tests(),
                plan.as_ref(),
                NativeDataFusionPostNativeExecutionPath::Cpu,
                walk,
                Vec::new(),
                false,
            );

            assert_eq!(coverage.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
            assert_eq!(coverage.domains.cudf_native_coverage, QueryGpuCudfNativeCoverage::None);
            assert_eq!(coverage.domains.cugraph_operator, QueryGpuCugraphOperatorDomain::Present);
            assert_eq!(coverage.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Absent);
        }

        #[test]
        fn wrapped_cugraph_with_cpu_collected_child_is_partial_not_false_native() {
            let cugraph = cugraph_exec(memory_edge_source());
            let input_name = cugraph.coverage_gpu_result_input_name().to_string();
            let plan = composed_exec_with_child(cugraph, input_name);
            let walk = walk_gpu_coverage(plan.as_ref());
            let coverage = QueryGpuCoverage::planned(
                validated_under_for_tests(),
                plan.as_ref(),
                NativeDataFusionPostNativeExecutionPath::Native,
                walk,
                Vec::new(),
                false,
            );

            assert!(matches!(coverage.overall, QueryGpuCoverageOverall::PartialGpuPlanned { .. }));
            assert_eq!(coverage.domains.cudf_native_coverage, QueryGpuCudfNativeCoverage::Full);
            assert_eq!(coverage.domains.cugraph_operator, QueryGpuCugraphOperatorDomain::Present);
            assert_eq!(coverage.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Present);
        }

        #[test]
        fn wrapped_cugraph_with_native_child_stays_full() {
            let cugraph = cugraph_exec(native_exec_leaf());
            let input_name = cugraph.coverage_gpu_result_input_name().to_string();
            let plan = composed_exec_with_child(cugraph, input_name);
            let walk = walk_gpu_coverage(plan.as_ref());
            let coverage = QueryGpuCoverage::planned(
                validated_under_for_tests(),
                plan.as_ref(),
                NativeDataFusionPostNativeExecutionPath::Native,
                walk,
                Vec::new(),
                false,
            );

            assert_eq!(coverage.overall, QueryGpuCoverageOverall::NoCpuInPlannedPath);
            assert_eq!(coverage.domains.datafusion_cpu_ingest_or_boundary, QueryGpuPresence::Absent);
            assert!(!coverage.runtime_caveats.is_empty());
        }
    }

    fn validated_under_for_tests() -> QueryGpuValidatedUnder {
        QueryGpuValidatedUnder {
            actual_policy: "gpu_preferred".to_string(),
            diagnostic_policy: "gpu_preferred".to_string(),
            optimizer_config_source: "test".to_string(),
            execution_mode: "strict_bounded".to_string(),
            session_config_fingerprint: "test".to_string(),
            optimizer_fingerprint: "test".to_string(),
            catalog_snapshot: "test".to_string(),
        }
    }
}
