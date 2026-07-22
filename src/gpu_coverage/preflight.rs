//! Coverage validation context, preflight, and error classification.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
};

use datafusion::{
    common::DataFusionError,
    execution::{SessionState, SessionStateBuilder},
    prelude::SessionContext,
    sql::{
        parser::{DFParserBuilder, Statement as DfSqlStatement},
        sqlparser::{ast::Statement as SqlStatement, dialect::dialect_from_str},
    },
};

use super::{
    COVERAGE_VALIDATOR_RECURSION_REASON_CODE, GPU_COVERAGE_REJECTION_REASON_FACT, NEXUS_EXPLAIN_COVERAGE_FUNCTION,
    model::{QueryGpuRejectionPhase, QueryGpuValidatedUnder},
    physical::planning_diagnostics_did_io,
    recursion_guard::query_mentions_coverage_table_function,
};
use crate::{
    error::{DataFusionErrorExt, Error, ErrorCode, ErrorSourceDiagnostics},
    native_exec::NexusNativeExecutionConfig,
    planner::NexusNativeOptimizerConfig,
    policy::GpuFallbackPolicy,
    session::installed_nexus_native_optimizer_config,
};

#[derive(Debug, Clone)]
pub(super) struct ValidationOptimizerConfig {
    pub(super) actual_policy: GpuFallbackPolicy,
    pub(super) optimizer_config_source: &'static str,
    pub(super) diagnostic_optimizer_config: NexusNativeOptimizerConfig,
}

#[derive(Debug)]
struct QueryGpuCoverageValidationInProgress;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QueryGpuRejection {
    pub(super) phase: QueryGpuRejectionPhase,
    pub(super) reason_code: String,
    pub(super) retryable: bool,
    pub(super) detail: String,
}

impl QueryGpuRejection {
    pub(super) fn new(
        phase: QueryGpuRejectionPhase,
        reason_code: impl Into<String>,
        retryable: bool,
        detail: impl Into<String>,
    ) -> Self {
        Self { phase, reason_code: reason_code.into(), retryable, detail: detail.into() }
    }

    pub(super) fn from_datafusion_error(phase: QueryGpuRejectionPhase, error: &DataFusionError) -> Self {
        if let Some(adapter_error) = error.find_nexus_error() {
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
                adapter_error.is_retryable(),
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

pub(super) fn optimizer_config_for_validation(ctx: &SessionContext) -> ValidationOptimizerConfig {
    let (optimizer_config_source, optimizer_config) = installed_nexus_native_optimizer_config(&ctx.state())
        .map(|config| ("installed_session_rule", config))
        .unwrap_or_else(|| ("default_gpu_preferred", default_validation_optimizer_config()));
    let actual_policy = optimizer_config.fallback_policy();
    let diagnostic_optimizer_config = optimizer_config.with_fallback_policy(GpuFallbackPolicy::GpuPreferred);
    ValidationOptimizerConfig { actual_policy, optimizer_config_source, diagnostic_optimizer_config }
}

fn default_validation_optimizer_config() -> NexusNativeOptimizerConfig {
    NexusNativeOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1))
}

pub(super) fn validated_under(state: &SessionState, config: &ValidationOptimizerConfig) -> QueryGpuValidatedUnder {
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

pub(super) fn preflight_single_query(state: &SessionState, sql: &str) -> Option<QueryGpuRejection> {
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

pub(super) fn classify_physical_planning_error(error: &DataFusionError) -> QueryGpuRejectionPhase {
    let Some(adapter_error) = error.find_nexus_error() else {
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
