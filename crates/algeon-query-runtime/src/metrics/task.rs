use std::{collections::BTreeMap, time::Duration};

use cudf::ExecutionBudgetReport;

use super::{CompiledTaskId, PlanNodeId, PlanTaskMapping, QueryId, QueryMetrics};

/// Metrics aggregated across tasks of the same operator kind.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskOperatorMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
    /// Operator-kind label shared by the aggregated tasks.
    pub operator_kind: String,
    /// Number of tasks observed for this operator kind.
    pub task_count: u64,
    /// Cumulative elapsed time across tasks.
    pub elapsed_time: Duration,
    /// Largest lower-level observed allocator peak for this operator kind.
    pub max_observed_peak_bytes: Option<u64>,
    /// Largest lower-level retained byte count for this operator kind.
    pub max_retained_bytes: Option<u64>,
}

impl TaskOperatorMetrics {
    pub(super) fn merge_from(&mut self, other: Self) {
        self.task_count = self.task_count.saturating_add(other.task_count);
        self.elapsed_time += other.elapsed_time;
        self.max_observed_peak_bytes = self.max_observed_peak_bytes.max(other.max_observed_peak_bytes);
        self.max_retained_bytes = self.max_retained_bytes.max(other.max_retained_bytes);
    }
}

/// Metrics for one compiled pipeline task.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskRuntimeMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
    /// Stable compiled task id.
    pub compiled_task_id: String,
    /// Operator-kind label for the task.
    pub operator_kind: String,
    /// Cumulative elapsed task time.
    pub elapsed_time: Duration,
    /// Largest lower-level observed allocator peak for the task.
    pub max_observed_peak_bytes: Option<u64>,
    /// Largest lower-level retained byte count for the task.
    pub max_retained_bytes: Option<u64>,
}

impl TaskRuntimeMetrics {
    pub(super) fn merge_from(&mut self, other: Self) {
        if self.operator_kind.is_empty() || self.operator_kind == "unknown" {
            self.operator_kind = other.operator_kind;
        }
        self.elapsed_time += other.elapsed_time;
        self.max_observed_peak_bytes = self.max_observed_peak_bytes.max(other.max_observed_peak_bytes);
        self.max_retained_bytes = self.max_retained_bytes.max(other.max_retained_bytes);
    }
}

/// Successful lower-level execution-budget report captured as a metrics row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedExecutionEvent {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
    /// Native operation label that produced the report.
    pub operation: String,
    /// Lower-level budget boundary label.
    pub boundary: String,
    /// Budget limit in bytes.
    pub budget_bytes: u64,
    /// Lower-level budget enforcement label.
    pub enforcement: String,
    /// Lower-level budget status label; always `reservation_enforced`,
    /// because a budget report exists only for a completed operation.
    pub status: String,
    /// Allocator high-water bytes reported by lower layers.
    pub allocator_high_water_bytes: Option<u64>,
    /// Retained bytes reported by lower layers.
    pub retained_bytes: Option<u64>,
    /// Output bytes reported by lower layers.
    pub output_bytes: Option<u64>,
    /// Output rows reported by lower layers.
    pub output_rows: Option<u64>,
    /// Additional lower-level context facts.
    pub context_facts: BTreeMap<String, String>,
}

impl BudgetedExecutionEvent {
    fn new(operation: impl Into<String>, report: &ExecutionBudgetReport) -> Self {
        Self {
            query_id: None,
            operation: operation.into(),
            boundary: report.request().boundary().to_string(),
            budget_bytes: report.request().budget_bytes() as u64,
            enforcement: report.enforcement().as_str().to_string(),
            status: "reservation_enforced".to_string(),
            allocator_high_water_bytes: report.allocator_high_water_bytes().map(|bytes| bytes as u64),
            retained_bytes: report.retained_bytes().map(|bytes| bytes as u64),
            output_bytes: report.output_bytes().map(|bytes| bytes as u64),
            output_rows: report.output_rows().map(|rows| rows as u64),
            context_facts: report.context_facts().clone(),
        }
    }
}

impl QueryMetrics {
    /// Records a lower-level cuDF execution-budget report.
    pub fn record_budgeted_execution(&mut self, operation: impl Into<String>, report: &ExecutionBudgetReport) {
        let observed_peak_bytes = report.allocator_high_water_bytes().map(|bytes| bytes as u64);
        let retained_bytes = report.retained_bytes().map(|bytes| bytes as u64);
        self.max_task_observed_peak_bytes = self.max_task_observed_peak_bytes.max(observed_peak_bytes);
        self.max_task_retained_bytes = self.max_task_retained_bytes.max(retained_bytes);
        let mut event = BudgetedExecutionEvent::new(operation, report);
        event.query_id = self.inherited_query_id();
        self.budgeted_execution_events.push(event);
    }

    /// Returns the explicit cursor used by one compiled task to delimit its
    /// lower-level budget reports.
    pub fn budgeted_execution_event_count(&self) -> usize {
        self.budgeted_execution_events.len()
    }

    /// Projects reports emitted after `start` onto one explicit compiled task.
    pub fn record_task_budget_observations_since(&mut self, compiled_task_id: impl ToString, start: usize) {
        let observed_peak_bytes = self
            .budgeted_execution_events
            .get(start..)
            .unwrap_or_default()
            .iter()
            .filter_map(|event| event.allocator_high_water_bytes)
            .max();
        let retained_bytes = self
            .budgeted_execution_events
            .get(start..)
            .unwrap_or_default()
            .iter()
            .filter_map(|event| event.retained_bytes)
            .max();
        if observed_peak_bytes.is_some() || retained_bytes.is_some() {
            self.record_task_budget_observation(&compiled_task_id.to_string(), observed_peak_bytes, retained_bytes);
        }
    }

    /// Records execution of a compiled pipeline task.
    pub fn record_compiled_pipeline_task_execution(
        &mut self,
        compiled_task_id: impl ToString,
        operator_kind: impl Into<String>,
    ) {
        self.record_task_execution(compiled_task_id, operator_kind);
    }

    /// Records execution of a task id and operator kind.
    pub fn record_task_execution(&mut self, task_id: impl ToString, operator_kind: impl Into<String>) {
        self.register_task_definition(task_id, operator_kind);
    }

    /// Registers one explicit task definition, incrementing operator task count
    /// only when the task id is first observed.
    pub fn register_task_definition(&mut self, task_id: impl ToString, operator_kind: impl Into<String>) {
        let compiled_task_id = task_id.to_string();
        let operator_kind = operator_kind.into();
        let query_id = self.inherited_query_id();
        let first_definition =
            self.task_kind_by_compiled_task_id.insert(compiled_task_id.clone(), operator_kind.clone()).is_none();
        let task = self.task_runtime_metrics.entry(compiled_task_id.clone()).or_insert_with(|| TaskRuntimeMetrics {
            query_id: query_id.clone(),
            compiled_task_id,
            operator_kind: operator_kind.clone(),
            ..Default::default()
        });
        task.operator_kind = operator_kind.clone();
        let row = self.task_operator_metrics.entry(operator_kind.clone()).or_insert_with(|| TaskOperatorMetrics {
            query_id: query_id.clone(),
            operator_kind,
            ..Default::default()
        });
        if first_definition {
            row.task_count = row.task_count.saturating_add(1);
        }
    }

    /// Records a diagnostic mapping between a compiled task and its source plan node.
    pub fn record_compiled_pipeline_task_mapping(
        &mut self,
        compiled_task_id: impl ToString,
        operator_kind: impl Into<String>,
        plan_node_id: Option<impl Into<PlanNodeId>>,
    ) {
        self.plan_task_mapping_rows.push(PlanTaskMapping {
            query_id: self.query_id.clone(),
            plan_node_id: plan_node_id.map(Into::into),
            compiled_task_id: Some(CompiledTaskId::new(compiled_task_id)),
            operator_kind: Some(operator_kind.into()),
            datafusion_shape: self.plan_task_mapping_datafusion_shape.clone(),
        });
    }

    /// Adds elapsed time to a task and its operator-kind aggregate.
    pub fn record_task_elapsed(&mut self, task_id: impl ToString, operator_kind: impl Into<String>, elapsed: Duration) {
        let compiled_task_id = task_id.to_string();
        let operator_kind = operator_kind.into();
        let query_id = self.inherited_query_id();
        self.task_kind_by_compiled_task_id.insert(compiled_task_id.clone(), operator_kind.clone());
        let row = self.task_runtime_metrics.entry(compiled_task_id.clone()).or_insert_with(|| TaskRuntimeMetrics {
            query_id: query_id.clone(),
            compiled_task_id,
            operator_kind: operator_kind.clone(),
            ..Default::default()
        });
        row.operator_kind = operator_kind.clone();
        row.elapsed_time += elapsed;

        let operator = self.task_operator_metrics.entry(operator_kind.clone()).or_insert_with(|| TaskOperatorMetrics {
            query_id: query_id.clone(),
            operator_kind,
            ..Default::default()
        });
        operator.elapsed_time += elapsed;
    }

    pub(crate) fn record_task_budget_observation(
        &mut self,
        compiled_task_id: &str,
        observed_peak_bytes: Option<u64>,
        retained_bytes: Option<u64>,
    ) {
        let operator_kind =
            self.task_kind_by_compiled_task_id.get(compiled_task_id).cloned().unwrap_or_else(|| "unknown".to_string());
        let query_id = self.inherited_query_id();
        let row = self.task_runtime_metrics.entry(compiled_task_id.to_string()).or_insert_with(|| TaskRuntimeMetrics {
            query_id: query_id.clone(),
            compiled_task_id: compiled_task_id.to_string(),
            operator_kind: operator_kind.clone(),
            ..Default::default()
        });
        row.operator_kind = operator_kind.clone();
        row.max_observed_peak_bytes = row.max_observed_peak_bytes.max(observed_peak_bytes);
        row.max_retained_bytes = row.max_retained_bytes.max(retained_bytes);

        let operator = self.task_operator_metrics.entry(operator_kind.clone()).or_insert_with(|| TaskOperatorMetrics {
            query_id: query_id.clone(),
            operator_kind,
            ..Default::default()
        });
        operator.max_observed_peak_bytes = operator.max_observed_peak_bytes.max(observed_peak_bytes);
        operator.max_retained_bytes = operator.max_retained_bytes.max(retained_bytes);
    }

    /// Builds the query-local mapping surface between plan node ids and compiled task ids.
    pub fn plan_task_mappings(&self) -> Vec<PlanTaskMapping> {
        let mut mappings = BTreeMap::<(Option<String>, Option<String>, Option<String>), PlanTaskMapping>::new();
        for mapping in &self.plan_task_mapping_rows {
            let mut mapping = mapping.clone();
            if mapping.datafusion_shape.is_none() {
                mapping.datafusion_shape.clone_from(&self.plan_task_mapping_datafusion_shape);
            }
            let key = (
                mapping.plan_node_id.as_ref().map(|value| value.as_str().to_string()),
                mapping.compiled_task_id.as_ref().map(|value| value.as_str().to_string()),
                mapping.operator_kind.clone(),
            );
            mappings.entry(key).or_insert(mapping);
        }
        for (compiled_task_id, runtime) in &self.task_runtime_metrics {
            let key = (None, Some(compiled_task_id.clone()), Some(runtime.operator_kind.clone()));
            mappings.entry(key).or_insert_with(|| PlanTaskMapping {
                query_id: runtime.query_id.clone().unwrap_or_else(|| self.query_id.clone()),
                plan_node_id: None,
                compiled_task_id: Some(compiled_task_id.as_str().into()),
                operator_kind: Some(runtime.operator_kind.clone()),
                datafusion_shape: self.plan_task_mapping_datafusion_shape.clone(),
            });
        }
        for operator in &self.operators {
            let key = (Some(operator.node_id.clone()), None, Some(operator.operator_kind.clone()));
            mappings.entry(key).or_insert_with(|| PlanTaskMapping {
                query_id: operator.query_id.clone().unwrap_or_else(|| self.query_id.clone()),
                plan_node_id: Some(PlanNodeId::new(operator.node_id.clone())),
                compiled_task_id: None,
                operator_kind: Some(operator.operator_kind.clone()),
                datafusion_shape: self.plan_task_mapping_datafusion_shape.clone(),
            });
        }
        mappings.into_values().collect()
    }

    /// Records one retry caused by resource exhaustion.
    pub fn record_retry_resource_exhausted(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
        self.retry_resource_exhausted_count = self.retry_resource_exhausted_count.saturating_add(1);
    }

    /// Records one compiled-pipeline execution attempt.
    pub fn record_compiled_pipeline_attempt(&mut self) {
        self.compiled_pipeline_attempts = self.compiled_pipeline_attempts.saturating_add(1);
    }

    /// Adds completed compiled-pipeline task count.
    pub fn record_compiled_pipeline_tasks_executed(&mut self, tasks: u64) {
        self.compiled_pipeline_tasks_executed = self.compiled_pipeline_tasks_executed.saturating_add(tasks);
    }

    /// Records one compiled-pipeline Host Arrow sink.
    pub fn record_compiled_pipeline_host_arrow_sink(&mut self) {
        self.compiled_pipeline_host_arrow_sinks = self.compiled_pipeline_host_arrow_sinks.saturating_add(1);
    }
}
