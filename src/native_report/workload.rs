use std::sync::Arc;

use datafusion::physical_plan::{ExecutionPlan, metrics::MetricsSet};

use crate::planner::NativePlanningDiagnostics;

/// Workload input for fallback-pressure report collection.
#[derive(Debug, Clone)]
pub struct NativeFallbackPressureWorkloadItem {
    /// Stable workload query id.
    pub query_id: String,
    /// Physical plan or planning failure captured before native execution.
    pub input: NativeFallbackPressureWorkloadInput,
}

impl NativeFallbackPressureWorkloadItem {
    /// Captures a physical plan for native candidate probing.
    pub fn new(query_id: impl Into<String>, plan: Arc<dyn ExecutionPlan>) -> Self {
        Self { query_id: query_id.into(), input: NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) }
    }

    /// Captures a pre-physical-planning failure as a reportable workload row.
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
    /// DataFusion physical plan to probe for native candidates.
    PhysicalPlan(Arc<dyn ExecutionPlan>),
    /// SQL or DataFusion planning failure before a physical plan existed.
    PlanningFailure(NativePlanningFailure),
}

/// SQL/DataFusion/native planning failure captured as a report row.
///
/// Current helpers use stable stages `sql_to_logical_plan`,
/// `physical_planning`, and `native_diagnostics`, with reason codes
/// `sql_planning_failed`, `physical_planning_failed`, and
/// `missing_native_diagnostics`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanningFailure {
    /// Planning stage such as `sql_to_logical_plan` or `physical_planning`.
    pub stage: String,
    /// Stable reason code for report consumers.
    pub reason_code: String,
    /// Human-readable error detail.
    pub message: String,
}

/// Workload input for DataFusion native-planning report collection.
#[derive(Debug, Clone)]
pub struct NativeDataFusionPlanningWorkloadItem {
    /// Stable workload query id.
    pub query_id: String,
    /// Post-optimizer plan/diagnostics or a planning failure.
    pub input: NativeDataFusionPlanningWorkloadInput,
}

impl NativeDataFusionPlanningWorkloadItem {
    /// Captures a post-native-optimizer plan and its retained diagnostics.
    pub fn new(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: NativePlanningDiagnostics,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionPlanningWorkloadInput::Planned {
                planned_plan,
                native_diagnostics: Arc::new(native_diagnostics),
            },
        }
    }

    pub fn new_shared(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: Arc<NativePlanningDiagnostics>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionPlanningWorkloadInput::Planned { planned_plan, native_diagnostics },
        }
    }

    /// Captures a planning failure as a reportable workload row.
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
    /// Post-native-optimizer DataFusion plan plus matching native planning diagnostics.
    Planned {
        /// Post-native-optimizer DataFusion plan.
        planned_plan: Arc<dyn ExecutionPlan>,
        /// Native diagnostics retained for the same optimized plan.
        native_diagnostics: Arc<NativePlanningDiagnostics>,
    },
    /// SQL/DataFusion/native diagnostic failure before a reportable plan existed.
    PlanningFailure(NativePlanningFailure),
}

/// Workload input for native execution report collection.
#[derive(Debug, Clone)]
pub struct NativeDataFusionExecutionWorkloadItem {
    /// Stable workload query id.
    pub query_id: String,
    /// Planned query plus runtime outcome, or a planning failure.
    pub input: NativeDataFusionExecutionWorkloadInput,
}

impl NativeDataFusionExecutionWorkloadItem {
    /// Captures a planned query and the native runtime outcome selected by the harness.
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
                native_diagnostics: Arc::new(native_diagnostics),
                runtime,
            },
        }
    }

    pub fn new_shared(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: Arc<NativePlanningDiagnostics>,
        runtime: NativeDataFusionRuntime,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionExecutionWorkloadInput::Planned { planned_plan, native_diagnostics, runtime },
        }
    }

    /// Captures a planning failure as a reportable workload row.
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
    /// Post-native-optimizer plan with matching diagnostics and runtime outcome.
    Planned {
        /// Post-native-optimizer DataFusion plan.
        planned_plan: Arc<dyn ExecutionPlan>,
        /// Native diagnostics retained for the same optimized plan.
        native_diagnostics: Arc<NativePlanningDiagnostics>,
        /// Runtime outcome selected by the workload harness.
        runtime: NativeDataFusionRuntime,
    },
    /// SQL/DataFusion/native diagnostic failure before execution could start.
    PlanningFailure(NativePlanningFailure),
}

/// Native runtime outcome captured by execution-report workload helpers.
#[derive(Debug, Clone)]
pub enum NativeDataFusionRuntime {
    /// Native planning did not produce an executable native fragment.
    NotStarted,
    /// Native runtime succeeded.
    ///
    /// `output_bytes` is host Arrow batch memory, not device memory. `metrics`
    /// are the root-plan DataFusion metrics captured after execution.
    Succeeded {
        /// Number of host Arrow output batches.
        output_batches: usize,
        /// Number of host Arrow output rows.
        output_rows: usize,
        /// Host Arrow batch memory size in bytes.
        output_bytes: usize,
        /// Root-plan DataFusion metrics captured after execution.
        metrics: MetricsSet,
    },
    /// Native runtime started but failed. TSV output counters serialize as `not_run`.
    Failed {
        /// Human-readable failure detail.
        message: String,
        /// Root-plan DataFusion metrics captured after the failure, when available.
        metrics: Option<MetricsSet>,
    },
    /// Native runtime started but did not complete normally.
    ///
    /// `outcome` is one of `Cancelled`, `Dropped`, or `Panicked`; the sole
    /// producer is the sealed attempt-report projection in server capture.
    Interrupted {
        /// The sealed attempt terminal outcome.
        outcome: super::NativeDataFusionRuntimeOutcome,
        /// Root-plan DataFusion metrics captured before interruption, when available.
        metrics: Option<MetricsSet>,
    },
}
