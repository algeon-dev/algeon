use super::*;

pub(crate) fn metric_duration_ms_text(metrics: &MetricsSet, name: &str) -> String {
    optional_duration_ms_text(metrics.sum_by_name(name).map(|value| value.as_usize()))
}

pub(crate) fn format_task_operator_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    algeon_datafusion::native_report::format_native_task_operator_metric_rows(&metrics.task_operator_rows)
}

pub(crate) fn format_task_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    algeon_datafusion::native_report::format_native_task_runtime_metric_rows(&metrics.task_runtime_rows)
}

pub(crate) fn format_source_read_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    algeon_datafusion::native_report::format_native_source_read_runtime_metric_rows(&metrics.source_read_runtime_rows)
}

pub(crate) fn format_lakehouse_cache_fill_event_snapshot_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    algeon_datafusion::native_report::format_native_lakehouse_cache_fill_event_metric_rows(
        &metrics.lakehouse_cache_fill_event_snapshot_rows,
    )
}

pub(crate) fn format_aggregate_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    algeon_datafusion::native_report::format_native_aggregate_strategy_metric_rows(&metrics.aggregate_strategy_rows)
}

pub(crate) fn format_join_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    algeon_datafusion::native_report::format_native_join_strategy_metric_rows(&metrics.join_strategy_rows)
}
