use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::MetricsSet;

use super::labels::{labels_by_name, optional_string_label, string_label};

const NATIVE_PLAN_TASK_MAPPING_ROWS_METRIC_NAME: &str = "native_plan_task_mapping_rows";

/// Diagnostic mapping between native plan nodes and compiled task ids.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativePlanTaskMappingMetricRow {
    /// Query-wide diagnostic correlation id.
    pub query_id: String,
    /// Native plan node id, when known.
    pub plan_node_id: Option<String>,
    /// Compiled task id, when known.
    pub compiled_task_id: Option<String>,
    /// Native operator/task kind, when known.
    pub operator_kind: Option<String>,
    /// DataFusion shape label, when supplied by the adapter.
    pub datafusion_shape: Option<String>,
}

impl NativePlanTaskMappingMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "query_id={},plan_node_id={},compiled_task_id={},operator_kind={},datafusion_shape={}",
            self.query_id,
            optional_text(self.plan_node_id.as_deref()),
            optional_text(self.compiled_task_id.as_deref()),
            optional_text(self.operator_kind.as_deref()),
            optional_text(self.datafusion_shape.as_deref())
        )
    }

    fn sort_key(&self) -> (Option<String>, Option<String>, Option<String>, Option<String>) {
        (
            self.plan_node_id.clone(),
            self.compiled_task_id.clone(),
            self.operator_kind.clone(),
            self.datafusion_shape.clone(),
        )
    }
}

pub(super) fn plan_task_mapping_rows(metrics: &MetricsSet) -> Vec<NativePlanTaskMappingMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_PLAN_TASK_MAPPING_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "query_id"))
        .map(|metric| labels_to_plan_task_mapping_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.sort_key());
    rows.dedup();
    rows
}

fn labels_to_plan_task_mapping_row(labels: BTreeMap<&str, &str>) -> NativePlanTaskMappingMetricRow {
    NativePlanTaskMappingMetricRow {
        query_id: string_label(&labels, "query_id"),
        plan_node_id: optional_string_label(&labels, "plan_node_id"),
        compiled_task_id: optional_string_label(&labels, "compiled_task_id"),
        operator_kind: optional_string_label(&labels, "operator_kind"),
        datafusion_shape: optional_string_label(&labels, "datafusion_shape"),
    }
}

fn optional_text(value: Option<&str>) -> &str {
    value.unwrap_or("none")
}
