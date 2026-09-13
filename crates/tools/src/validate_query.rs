use std::{error::Error, path::PathBuf};

use algeon_datafusion::{
    gpu_coverage::{QueryGpuCandidateOutcome, QueryGpuCoverage, QueryGpuPlanningOutcome, validate_query},
    planner::AlgeonNativeOptimizerConfig,
    session::AlgeonSessionStateBuilderExt,
};
use datafusion::{
    execution::SessionStateBuilder,
    prelude::{ParquetReadOptions, SessionContext},
};

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
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&coverage.to_json())?),
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
            println!("rejection_phase: {}", rejection.phase.as_str());
            println!("reason_code: {}", rejection.reason_code);
            if let Some(status) = &rejection.source_error_status {
                println!("source_error_status: {status}");
            }
            println!("detail: {}", rejection.detail);
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

fn candidate_outcome_human_line(outcome: &QueryGpuCandidateOutcome) -> String {
    format!(
        "candidate_outcome: outcome={} reason_code={} reason_kind={} category={} detail={}",
        outcome.selection.as_str(),
        outcome.reason_code.as_deref().unwrap_or("absent"),
        outcome.reason_kind.map_or("absent", |kind| kind.as_str()),
        outcome.category.as_deref().unwrap_or("absent"),
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
            cost_evidence: None,
        };

        assert_eq!(
            candidate_outcome_human_line(&outcome),
            "candidate_outcome: outcome=selected reason_code=absent reason_kind=absent category=absent detail=ProjectionExec"
        );

        let unsupported = QueryGpuCandidateOutcome {
            selection: CandidateOutcomeKind::NotSupported,
            candidate_family: "projection".to_string(),
            reason_code: Some("unsupported_expression".to_string()),
            reason_kind: Some(QueryGpuCandidateReasonKind::Expression),
            detail: "ProjectionExec".to_string(),
            category: Some("expression".to_string()),
            remedy: None,
            cost_evidence: None,
        };
        assert_eq!(
            candidate_outcome_human_line(&unsupported),
            "candidate_outcome: outcome=not_supported reason_code=unsupported_expression reason_kind=expression category=expression detail=ProjectionExec"
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
}
