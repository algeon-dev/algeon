use std::{error::Error, path::PathBuf};

use algeon_datafusion::{
    gpu_coverage::{
        QueryGpuCandidateOutcome, QueryGpuCoverage, QueryGpuPlanningOutcome, QueryGpuPlanningRejection, validate_query,
    },
    planner::AlgeonNativeOptimizerConfig,
    session::AlgeonSessionStateBuilderExt,
};
use datafusion::{
    execution::SessionStateBuilder,
    prelude::{ParquetReadOptions, SessionContext},
};
use serde_json::Value;

type ValidateResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Human,
}

#[derive(Debug)]
struct Config {
    sql: String,
    format: OutputFormat,
    parquet_tables: Vec<ParquetTable>,
}

#[derive(Debug)]
struct ParquetTable {
    table: String,
    path: PathBuf,
}

pub async fn run(args: Vec<String>) -> ValidateResult<()> {
    if args.iter().any(|arg| matches!(arg.as_str(), "-h" | "--help")) {
        println!("{}", validate_query_usage());
        return Ok(());
    }
    let config = parse_args(args)?;
    let ctx = build_context(&config).await?;
    let coverage = validate_query(&ctx, &config.sql).await?;
    match config.format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&coverage_json(&coverage))?),
        OutputFormat::Human => print_human(&coverage),
    }
    Ok(())
}

async fn build_context(config: &Config) -> ValidateResult<SessionContext> {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    for table in &config.parquet_tables {
        ctx.register_parquet(&table.table, path_str(&table.path)?, ParquetReadOptions::default()).await?;
    }
    Ok(ctx)
}

fn parse_args(args: Vec<String>) -> ValidateResult<Config> {
    let mut sql = None;
    let mut format = OutputFormat::Human;
    let mut parquet_tables = Vec::new();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--sql" => {
                sql = Some(next_value(&mut iter, "--sql")?);
            },
            "--format" => {
                let value = next_value(&mut iter, "--format")?;
                format = parse_format(&value)?;
            },
            "--parquet" => {
                parquet_tables.push(parse_parquet_table(&next_value(&mut iter, "--parquet")?)?);
            },
            _ if sql.is_none() && !arg.starts_with('-') => {
                sql = Some(arg);
            },
            _ => return Err(format!("unknown validate-query argument: {arg}\n\n{}", validate_query_usage()).into()),
        }
    }
    let sql = sql.ok_or_else(validate_query_usage)?;
    Ok(Config { sql, format, parquet_tables })
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> ValidateResult<String> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_format(value: &str) -> ValidateResult<OutputFormat> {
    match value {
        "json" => Ok(OutputFormat::Json),
        "human" => Ok(OutputFormat::Human),
        _ => Err("--format must be json or human".into()),
    }
}

fn parse_parquet_table(value: &str) -> ValidateResult<ParquetTable> {
    let Some((table, path)) = value.split_once('=') else {
        return Err("--parquet expects TABLE=PATH".into());
    };
    if table.trim().is_empty() || path.trim().is_empty() {
        return Err("--parquet expects non-empty TABLE=PATH".into());
    }
    Ok(ParquetTable { table: table.to_string(), path: PathBuf::from(path) })
}

fn path_str(path: &PathBuf) -> ValidateResult<&str> {
    path.to_str().ok_or_else(|| format!("path is not valid UTF-8: {path:?}").into())
}

/// The coverage JSON with the trusted diagnostic of a rejection added. This
/// tool plans in the process that prints, so it shows the text a server
/// withholds from its clients.
fn coverage_json(coverage: &QueryGpuCoverage) -> Value {
    let mut json = coverage.to_json();
    if let Some(rejection) = coverage.rejection() {
        json["planning_outcome"]["rejection"]["trusted_diagnostic"] =
            Value::from(rejection.trusted_diagnostic.as_str());
    }
    json
}

fn print_human(coverage: &QueryGpuCoverage) {
    println!("gpu_path: {}", coverage.gpu_path());
    match &coverage.planning_outcome {
        QueryGpuPlanningOutcome::Planned(plan) => {
            println!("final_disposition: {}", plan.disposition().as_str());
            println!("native_fragment_count: {}", plan.native_fragment_count());
            println!("explicit_gpu_island_count: {}", plan.explicit_gpu_island_count());
            println!("datafusion_boundary_count: {}", plan.datafusion_boundary_count());
            println!("host_materialization_count: {}", plan.host_materialization_count());
        },
        QueryGpuPlanningOutcome::Rejected(rejection) => {
            for line in rejection_human_lines(rejection) {
                println!("{line}");
            }
        },
    }
    println!("confidence: plan_time_only");
    println!("planning_did_io: {}", coverage.planning_did_io);
    println!("optimizer_config_source: {}", coverage.validated_under.optimizer_config_source);
    println!("execution_mode: {}", coverage.validated_under.execution_mode.as_deref().unwrap_or("absent"));
    println!("candidate_outcome_count: {}", coverage.candidate_outcomes.len());
    for outcome in &coverage.candidate_outcomes {
        println!("{}", candidate_outcome_human_line(outcome));
    }
    println!("runtime_caveat_count: {}", coverage.runtime_caveats.len());
    for caveat in &coverage.runtime_caveats {
        println!("runtime_caveat: {} - {}", caveat.code, caveat.detail);
    }
    if let Some(root) = &coverage.physical_plan_root {
        println!("physical_plan_root: {root}");
    }
}

fn rejection_human_lines(rejection: &QueryGpuPlanningRejection) -> Vec<String> {
    let mut lines = vec![
        format!("rejection_phase: {}", rejection.phase.as_str()),
        format!("reason_code: {}", rejection.reason_code),
    ];
    if let Some(status) = &rejection.source_error_status {
        lines.push(format!("source_error_status: {status}"));
    }
    lines.push(format!("detail: {}", rejection.trusted_diagnostic));
    lines
}

fn candidate_outcome_human_line(outcome: &QueryGpuCandidateOutcome) -> String {
    format!(
        "candidate_outcome: outcome={} reason_code={} reason_kind={} category={} blocking={} detail={}",
        outcome.selection.as_str(),
        outcome.reason_code.as_deref().unwrap_or("absent"),
        outcome.reason_kind.map_or("absent", |kind| kind.as_str()),
        outcome.category.as_deref().unwrap_or("absent"),
        outcome.blocking.as_ref().map_or("absent", |blocking| blocking.text()),
        outcome.detail
    )
}

fn validate_query_usage() -> String {
    "USAGE:\n  cargo run -p tools --features validate-query -- validate-query --sql <SQL> [--format json|human] [--parquet TABLE=PATH]...\n\n\
     EXAMPLE:\n  cargo run -p tools --features validate-query -- validate-query --sql \"SELECT * FROM lineitem\" --parquet lineitem=/data/tpch/lineitem.parquet --format json"
        .to_string()
}

#[cfg(test)]
mod tests {
    use algeon_datafusion::{gpu_coverage::QueryGpuCandidateReasonKind, report::CandidateOutcomeKind};

    use super::*;

    #[test]
    fn selected_candidate_human_line_marks_optional_reason_fields_absent() {
        let outcome = QueryGpuCandidateOutcome {
            selection: CandidateOutcomeKind::Selected,
            candidate_family: "projection".to_string(),
            reason_code: None,
            reason_kind: None,
            detail: "ProjectionExec".to_string(),
            category: None,
            remedy: None,
            blocking: None,
            cost_evidence: None,
        };

        assert_eq!(
            candidate_outcome_human_line(&outcome),
            "candidate_outcome: outcome=selected reason_code=absent reason_kind=absent category=absent blocking=absent detail=ProjectionExec"
        );

        let unsupported = QueryGpuCandidateOutcome {
            selection: CandidateOutcomeKind::NotSupported,
            candidate_family: "projection".to_string(),
            reason_code: Some("unsupported_expression".to_string()),
            reason_kind: Some(QueryGpuCandidateReasonKind::Expression),
            detail: "ProjectionExec".to_string(),
            category: Some("expression".to_string()),
            remedy: None,
            blocking: None,
            cost_evidence: None,
        };
        assert_eq!(
            candidate_outcome_human_line(&unsupported),
            "candidate_outcome: outcome=not_supported reason_code=unsupported_expression reason_kind=expression category=expression blocking=absent detail=ProjectionExec"
        );
    }

    #[tokio::test]
    async fn build_context_installs_native_optimizer_rule() {
        let config =
            Config { sql: "SELECT 1 AS value".to_string(), format: OutputFormat::Json, parquet_tables: Vec::new() };
        let ctx = build_context(&config).await.expect("context build succeeds");
        let coverage = validate_query(&ctx, &config.sql).await.expect("validate_query succeeds");
        assert!(coverage.validated_under.native_optimizer_rule_installed);
        assert_eq!(coverage.validated_under.optimizer_config_source, "installed_session_rule");
    }

    /// This tool has no server query id and no `GetQueryStats`, so a failure
    /// the server masks for its clients prints its source-chain text.
    #[tokio::test]
    async fn masked_rejection_prints_the_trusted_diagnostic() {
        let sql = "SELECT arrow_cast(1, 'no_such_type')";
        let config = Config { sql: sql.to_string(), format: OutputFormat::Json, parquet_tables: Vec::new() };
        let ctx = build_context(&config).await.expect("context build succeeds");
        let coverage = validate_query(&ctx, sql).await.expect("validate_query succeeds");
        let rejection = coverage.rejection().expect("an unknown Arrow type name fails planning");
        assert!(!rejection.detail.contains("no_such_type"), "the client text stays masked: {}", rejection.detail);

        let lines = rejection_human_lines(rejection);
        let detail = lines.iter().find_map(|line| line.strip_prefix("detail: ")).expect("a detail line");
        assert!(detail.contains("no_such_type"), "{detail}");
        assert!(!detail.contains("GetQueryStats"), "{detail}");

        let json = coverage_json(&coverage);
        let trusted = json["planning_outcome"]["rejection"]["trusted_diagnostic"].as_str();
        assert!(trusted.is_some_and(|text| text.contains("no_such_type")), "{json}");
    }
}
