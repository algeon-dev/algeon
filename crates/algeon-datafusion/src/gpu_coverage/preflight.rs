//! Coverage validation context, preflight, and error classification.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
};

use datafusion::{
    common::DataFusionError,
    execution::{SessionState, SessionStateBuilder},
    sql::{
        parser::{DFParserBuilder, Statement as DfSqlStatement},
        sqlparser::{ast::Statement as SqlStatement, dialect::dialect_from_str},
    },
};

use super::{
    ALGEON_EXPLAIN_COVERAGE_FUNCTION, COVERAGE_VALIDATOR_RECURSION_REASON_CODE, GPU_COVERAGE_REJECTION_REASON_FACT,
    model::{QueryGpuRejectionPhase, QueryGpuValidatedUnder},
    physical::planning_diagnostics_did_io,
    recursion_guard::query_mentions_coverage_table_function,
};
use crate::{
    error::{DataFusionErrorExt, Error, ErrorCode, ErrorSourceDiagnostics},
    planner::{AlgeonNativeOptimizerConfig, DataFusionCpuExecutionPolicy},
    session::installed_algeon_native_optimizer_config,
};

#[derive(Debug, Clone)]
pub(super) struct ValidationOptimizerConfig {
    pub(super) optimizer_config_source: &'static str,
    /// The exact caller-installed rule configuration copied into validation.
    pub(super) optimizer_config: AlgeonNativeOptimizerConfig,
    /// The copied rule configuration after the cloned session's DataFusion
    /// options have been applied.
    pub(super) effective_optimizer_config: AlgeonNativeOptimizerConfig,
}

#[derive(Debug)]
struct QueryGpuCoverageValidationInProgress;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QueryGpuRejection {
    pub(super) phase: QueryGpuRejectionPhase,
    pub(super) reason_code: String,
    /// Coarse source status is diagnostic context only. It is never sealed
    /// attempt retry advice.
    pub(super) source_error_status: Option<&'static str>,
    pub(super) detail: String,
}

impl QueryGpuRejection {
    pub(super) fn new(
        phase: QueryGpuRejectionPhase,
        reason_code: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self { phase, reason_code: reason_code.into(), source_error_status: None, detail: detail.into() }
    }

    pub(super) fn from_datafusion_error(phase: QueryGpuRejectionPhase, error: &DataFusionError) -> Self {
        if let Some(adapter_error) = error.find_algeon_error() {
            let reason_code = adapter_error
                .fact_value(GPU_COVERAGE_REJECTION_REASON_FACT)
                .unwrap_or_else(|| adapter_error.code().as_str())
                .to_string();
            let diagnostics = ErrorSourceDiagnostics::from_error(error);
            return Self {
                phase,
                reason_code,
                source_error_status: Some(adapter_error.status().as_str()),
                detail: diagnostics.chain().unwrap_or_else(|| adapter_error.message()).to_string(),
            };
        }
        let diagnostics = ErrorSourceDiagnostics::from_error(error);
        Self::new(
            phase,
            ErrorCode::DataFusionPlanning.as_str(),
            diagnostics.chain().map(ToString::to_string).unwrap_or_else(|| error.message().to_string()),
        )
    }
}

pub(super) fn optimizer_config_for_validation(state: &SessionState) -> crate::error::Result<ValidationOptimizerConfig> {
    let config = match installed_algeon_native_optimizer_config(state) {
        Some(config) => config,
        None => {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                "algeon_datafusion.gpu_coverage.validation_config",
                "native optimizer rule is not installed on the validation session; install it via AlgeonGpuBackend::install_on or AlgeonSessionStateBuilderExt::with_algeon_native",
            )
            .with_fact("reason", "native_optimizer_rule_not_installed"));
        },
    };
    Ok(ValidationOptimizerConfig {
        optimizer_config_source: "installed_session_rule",
        optimizer_config: config.clone(),
        effective_optimizer_config: config,
    })
}

#[derive(Debug)]
#[expect(dead_code, reason = "fields read through derived Debug format for fingerprint")]
struct StableNativeValidationPolicy {
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    max_source_reads_in_flight: usize,
    source_chunk_planning_mode: query_engine::runtime::SourceChunkPlanningMode,
    statistical_aggregate_execution_mode: query_engine::mode::StatisticalAggregateExecutionMode,
}

pub(super) fn validated_under(state: &SessionState, config: &ValidationOptimizerConfig) -> QueryGpuValidatedUnder {
    let rules = state.physical_optimizers();
    let rule_names = rules.iter().map(|rule| rule.name()).collect::<Vec<_>>().join(",");
    let optimizer_config = &config.effective_optimizer_config;
    // The native execution config holds request-scoped IDs, service
    // handles, cancellation state, and caches. Those values must not
    // turn the plan-time configuration fingerprint into a per-call
    // nonce. Project only the stable session/planner options that
    // affect validation and lowering.
    let execution = optimizer_config.native_execution_config();
    let stable_policy = StableNativeValidationPolicy {
        max_source_chunk_bytes: execution.max_source_chunk_bytes(),
        max_row_groups_per_chunk: execution.max_row_groups_per_chunk(),
        max_retry_attempts: execution.max_retry_attempts(),
        max_source_reads_in_flight: execution.max_source_reads_in_flight(),
        source_chunk_planning_mode: execution.source_chunk_planning_mode(),
        statistical_aggregate_execution_mode: execution.statistical_aggregate_execution_mode(),
    };
    let final_plan_requirements = match optimizer_config.datafusion_cpu_execution_policy() {
        DataFusionCpuExecutionPolicy::Allow => Vec::new(),
        DataFusionCpuExecutionPolicy::Reject => vec!["no_datafusion_cpu".to_string()],
    };
    let optimizer_fingerprint_input = format!(
        "native_optimizer_rule=installed;{rule_names};session_options={stable_policy:?};native_optimizer_options={:?};native_plan_cost_options={:?};final_plan_requirements={final_plan_requirements:?}",
        optimizer_config.native_optimizer_options(),
        optimizer_config.native_plan_cost_options(),
    );
    QueryGpuValidatedUnder {
        optimizer_config_source: config.optimizer_config_source.to_string(),
        native_optimizer_rule_installed: true,
        execution_mode: Some("bounded".to_string()),
        final_plan_requirements,
        session_config_fingerprint: fingerprint_session_config(state.config_options()),
        optimizer_fingerprint: fingerprint_debug(&optimizer_fingerprint_input),
        catalog_snapshot: "session_state_cloned_at_validation_call".to_string(),
    }
}

pub(super) fn mark_coverage_validation_in_progress(builder: &mut SessionStateBuilder) {
    builder.config().get_or_insert_with(Default::default).set_extension(Arc::new(QueryGpuCoverageValidationInProgress));
}

pub(crate) fn gpu_coverage_validation_in_progress(state: &SessionState) -> bool {
    state.config().get_extension::<QueryGpuCoverageValidationInProgress>().is_some()
}

pub(super) fn coverage_validator_recursion_rejection() -> QueryGpuRejection {
    QueryGpuRejection::new(
        QueryGpuRejectionPhase::Parsing,
        COVERAGE_VALIDATOR_RECURSION_REASON_CODE,
        format!("{ALGEON_EXPLAIN_COVERAGE_FUNCTION} cannot validate a query that calls itself"),
    )
}

pub(crate) fn coverage_validator_recursion_error() -> Error {
    Error::new(
        ErrorCode::DataFusionPlanning,
        "algeon_datafusion.gpu_coverage.recursion",
        format!("{ALGEON_EXPLAIN_COVERAGE_FUNCTION} cannot validate a query that calls itself"),
    )
    .with_fact(GPU_COVERAGE_REJECTION_REASON_FACT, COVERAGE_VALIDATOR_RECURSION_REASON_CODE)
}

pub(super) fn preflight_single_query(state: &SessionState, sql: &str) -> Option<QueryGpuRejection> {
    let options = state.config().options();
    let dialect_config = options.sql_parser.dialect;
    let Some(dialect) = dialect_from_str(dialect_config) else {
        return Some(QueryGpuRejection::new(
            QueryGpuRejectionPhase::Parsing,
            ErrorCode::DataFusionPlanning.as_str(),
            format!("unsupported SQL dialect configured for validation: {dialect_config}"),
        ));
    };
    let statements = match DFParserBuilder::new(sql)
        .with_dialect(dialect.as_ref())
        .with_recursion_limit(options.sql_parser.recursion_limit.get())
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
            "validate_query expects a SQL query statement",
        ));
    }
    if statements.len() != 1 {
        return Some(QueryGpuRejection::new(
            QueryGpuRejectionPhase::Parsing,
            "multi_statement_unsupported",
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
                "validate_query v1 supports SELECT/query statements only",
            )),
        },
        Some(_) => Some(QueryGpuRejection::new(
            QueryGpuRejectionPhase::Parsing,
            "non_query_statement_unsupported",
            "validate_query v1 supports SELECT/query statements only",
        )),
        None => unreachable!("empty statements are handled before statement matching"),
    }
}

pub(super) fn classify_physical_planning_error(error: &DataFusionError) -> QueryGpuRejectionPhase {
    let Some(adapter_error) = error.find_algeon_error() else {
        return QueryGpuRejectionPhase::PhysicalPlanning;
    };
    match adapter_error.code() {
        ErrorCode::PlanningTimeout
        | ErrorCode::SourceAccessDenied
        | ErrorCode::SourceUnauthenticated
        | ErrorCode::SourceObjectNotFound
        | ErrorCode::SourceTemporaryFailure => QueryGpuRejectionPhase::PlanningIo,
        _ => QueryGpuRejectionPhase::PhysicalPlanning,
    }
}

pub(super) fn planning_did_io_for_physical_error(
    phase: QueryGpuRejectionPhase,
    diagnostics: Option<&crate::planner::NativePlanningDiagnostics>,
) -> bool {
    diagnostics.is_some_and(planning_diagnostics_did_io) || phase == QueryGpuRejectionPhase::PlanningIo
}

fn fingerprint_debug(value: &impl std::fmt::Debug) -> String {
    let mut hasher = DefaultHasher::new();
    format!("{value:?}").hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Fingerprint only declarative session settings, never opaque invocation
/// extensions. Attempt-local bindings carry admission handles and cancellation
/// state but cannot change coverage planning; including their debug output
/// would make otherwise identical SQL surfaces report different evidence.
fn fingerprint_session_config(config: &datafusion::common::config::ConfigOptions) -> String {
    let mut entries = config.entries().into_iter().map(|entry| (entry.key, entry.value)).collect::<Vec<_>>();
    entries.sort_unstable();
    fingerprint_debug(&entries)
}
