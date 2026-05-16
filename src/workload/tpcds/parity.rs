use std::{path::Path, sync::Arc};

use datafusion::{
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};

use super::{TPCDS_QUERIES, TpcdsWorkloadResult, register_tpcds_tables};
use crate::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, NativePlanningFailure,
    NativeTpcdsParityCpuOutcome, NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome,
    NativeTpcdsParityWorkloadItem,
};

/// Build parity workload items for the TPC-DS workload. For each selected query
/// the runner executes both CPU DataFusion and native-rule DataFusion and
/// captures schema/batches/diagnostics/metrics needed by the parity report.
pub async fn tpcds_parity_workload_items(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    let cpu_ctx = SessionContext::new_with_config(session_config.clone());
    register_tpcds_tables(&cpu_ctx, root).await?;

    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let native_state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let native_ctx = SessionContext::new_with_state(native_state);
    register_tpcds_tables(&native_ctx, root).await?;

    let mut items = Vec::with_capacity(TPCDS_QUERIES.len());
    for query in TPCDS_QUERIES {
        if !is_query_selected(query.id, query_filter) {
            continue;
        }

        let cpu_outcome = run_cpu_query(&cpu_ctx, query.sql).await;
        let (native_planning, native_runtime) = run_native_query(&native_ctx, &native_rule, query.sql).await;
        items.push(NativeTpcdsParityWorkloadItem {
            query_id: query.id.to_string(),
            sql: query.sql.to_string(),
            cpu_outcome,
            native_planning,
            native_runtime,
        });
    }
    Ok(items)
}

fn is_query_selected(query_id: &str, filter: Option<&[String]>) -> bool {
    match filter {
        Some(allowed) => allowed.iter().any(|allowed_id| allowed_id == query_id),
        None => true,
    }
}

async fn run_cpu_query(ctx: &SessionContext, sql: &str) -> NativeTpcdsParityCpuOutcome {
    let df = match ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "sql_to_logical_plan".to_string(),
                reason_code: "sql_planning_failed".to_string(),
                message: error.to_string(),
            });
        },
    };
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(error) => {
            return NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "physical_planning".to_string(),
                reason_code: "physical_planning_failed".to_string(),
                message: error.to_string(),
            });
        },
    };
    let schema = plan.schema();
    match collect(plan, Arc::new(TaskContext::default())).await {
        Ok(batches) => NativeTpcdsParityCpuOutcome::Succeeded { schema, batches },
        Err(error) => NativeTpcdsParityCpuOutcome::ExecutionFailed { message: error.to_string() },
    }
}

async fn run_native_query(
    ctx: &SessionContext,
    native_rule: &Arc<CudfNativePhysicalOptimizerRule>,
    sql: &str,
) -> (NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome) {
    let df = match ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return (
                NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                    stage: "sql_to_logical_plan".to_string(),
                    reason_code: "sql_planning_failed".to_string(),
                    message: error.to_string(),
                }),
                NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
            );
        },
    };
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(error) => {
            return (
                NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                    stage: "physical_planning".to_string(),
                    reason_code: "physical_planning_failed".to_string(),
                    message: error.to_string(),
                }),
                NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
            );
        },
    };
    let Some(diagnostics) = native_rule.last_planning_diagnostics() else {
        return (
            NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                stage: "native_diagnostics".to_string(),
                reason_code: "missing_native_diagnostics".to_string(),
                message: "native optimizer rule did not retain planning diagnostics".to_string(),
            }),
            NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
        );
    };
    let accepted = diagnostics.report.rewritten_count() > 0;
    let planning_outcome =
        NativeTpcdsParityNativePlanningOutcome::Planned { accepted, planned_plan: Arc::clone(&plan), diagnostics };
    let runtime_outcome =
        if accepted { execute_native(plan).await } else { NativeTpcdsParityNativeRuntimeOutcome::NotStarted };
    (planning_outcome, runtime_outcome)
}

async fn execute_native(plan: Arc<dyn ExecutionPlan>) -> NativeTpcdsParityNativeRuntimeOutcome {
    let schema = plan.schema();
    match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(batches) => {
            let metrics = plan.metrics().unwrap_or_default();
            NativeTpcdsParityNativeRuntimeOutcome::Succeeded { schema, batches, metrics }
        },
        Err(error) => NativeTpcdsParityNativeRuntimeOutcome::Failed { message: error.to_string() },
    }
}
