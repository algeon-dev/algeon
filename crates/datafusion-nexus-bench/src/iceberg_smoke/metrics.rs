use super::*;

pub(crate) fn metric_duration_ms_text(metrics: &MetricsSet, name: &str) -> String {
    optional_duration_ms_text(metrics.sum_by_name(name).map(|value| value.as_usize()))
}

pub(crate) fn format_task_operator_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.task_operator_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.task_operator_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(crate) fn format_task_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.task_runtime_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.task_runtime_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(crate) fn format_source_read_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.source_read_runtime_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.source_read_runtime_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(crate) fn format_aggregate_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.aggregate_strategy_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.aggregate_strategy_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(crate) fn format_join_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.join_strategy_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.join_strategy_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}
