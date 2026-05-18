use std::sync::Arc;

use datafusion::physical_plan::{ExecutionPlan, metrics::MetricsSet};

use crate::planner::NativePlanningDiagnostics;

#[derive(Debug, Clone)]
pub struct NativeFallbackPressureWorkloadItem {
    pub query_id: String,
    pub input: NativeFallbackPressureWorkloadInput,
}

impl NativeFallbackPressureWorkloadItem {
    pub fn new(query_id: impl Into<String>, plan: Arc<dyn ExecutionPlan>) -> Self {
        Self { query_id: query_id.into(), input: NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) }
    }

    pub fn planning_failure(
        query_id: impl Into<String>,
        stage: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeFallbackPressureWorkloadInput::PlanningFailure(NativePlanningFailure {
                stage: stage.into(),
                reason_code: reason_code.into(),
                message: message.into(),
            }),
        }
    }
}

/// Input captured by the fallback-pressure harness before native execution.
///
/// `PhysicalPlan` carries a DataFusion plan to probe for native candidates.
/// `PlanningFailure` is used when SQL/DataFusion planning failed before a
/// physical plan existed, but the report still needs a row for the workload
/// item.
#[derive(Debug, Clone)]
pub enum NativeFallbackPressureWorkloadInput {
    PhysicalPlan(Arc<dyn ExecutionPlan>),
    PlanningFailure(NativePlanningFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanningFailure {
    pub stage: String,
    pub reason_code: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct NativeDataFusionPlanningWorkloadItem {
    pub query_id: String,
    pub input: NativeDataFusionPlanningWorkloadInput,
}

impl NativeDataFusionPlanningWorkloadItem {
    pub fn new(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: NativePlanningDiagnostics,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionPlanningWorkloadInput::Planned {
                planned_plan,
                native_diagnostics: Box::new(native_diagnostics),
            },
        }
    }

    pub fn planning_failure(
        query_id: impl Into<String>,
        stage: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionPlanningWorkloadInput::PlanningFailure(NativePlanningFailure {
                stage: stage.into(),
                reason_code: reason_code.into(),
                message: message.into(),
            }),
        }
    }
}

/// Input captured after DataFusion planning and the native optimizer rule have
/// run for a planning-report workload item.
///
/// `Planned` stores the post-optimizer plan plus native diagnostics.
/// `PlanningFailure` records SQL/DataFusion planning failures that prevented
/// those diagnostics from being produced.
#[derive(Debug, Clone)]
pub enum NativeDataFusionPlanningWorkloadInput {
    Planned { planned_plan: Arc<dyn ExecutionPlan>, native_diagnostics: Box<NativePlanningDiagnostics> },
    PlanningFailure(NativePlanningFailure),
}

#[derive(Debug, Clone)]
pub struct NativeDataFusionExecutionWorkloadItem {
    pub query_id: String,
    pub input: NativeDataFusionExecutionWorkloadInput,
}

impl NativeDataFusionExecutionWorkloadItem {
    pub fn new(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: NativePlanningDiagnostics,
        runtime: NativeDataFusionRuntime,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionExecutionWorkloadInput::Planned {
                planned_plan,
                native_diagnostics: Box::new(native_diagnostics),
                runtime,
            },
        }
    }

    pub fn planning_failure(
        query_id: impl Into<String>,
        stage: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionExecutionWorkloadInput::PlanningFailure(NativePlanningFailure {
                stage: stage.into(),
                reason_code: reason_code.into(),
                message: message.into(),
            }),
        }
    }
}

/// Input captured by the execution-report harness after planning and optional
/// native execution.
///
/// `Planned` includes the plan, native planning diagnostics, and runtime outcome
/// metrics. `PlanningFailure` is used when the workload could not reach physical
/// planning, so execution was never attempted.
#[derive(Debug, Clone)]
pub enum NativeDataFusionExecutionWorkloadInput {
    Planned {
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: Box<NativePlanningDiagnostics>,
        runtime: NativeDataFusionRuntime,
    },
    PlanningFailure(NativePlanningFailure),
}

#[derive(Debug, Clone)]
pub enum NativeDataFusionRuntime {
    NotStarted,
    Succeeded { output_batches: usize, output_rows: usize, output_bytes: usize, metrics: MetricsSet },
    Failed { message: String },
}
