//! DataFusion metric registration generated from the ordered contract.

use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricCategory, MetricType, Time,
};

use super::contract::for_each_native_algeon_metric;

macro_rules! metric_storage_type {
    (rows) => {
        Count
    };
    (bytes) => {
        Count
    };
    (timer) => {
        Time
    };
}

macro_rules! register_metric {
    ($metrics:expr, $partition:expr, $name:expr, rows) => {
        MetricBuilder::new($metrics)
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Rows)
            .counter($name, $partition)
    };
    ($metrics:expr, $partition:expr, $name:expr, bytes) => {
        MetricBuilder::new($metrics)
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Bytes)
            .counter($name, $partition)
    };
    ($metrics:expr, $partition:expr, $name:expr, timer) => {
        MetricBuilder::new($metrics)
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Timing)
            .subset_time($name, $partition)
    };
}

macro_rules! define_registry {
    ($( $field:ident => $name:literal: $kind:ident, )*) => {
        #[derive(Debug, Clone)]
        // Fields are written through `AlgeonNativeMetrics: Deref<Target = NativeMetricRegistry>`
        // from the descriptor recording macros. Keep the full ordered registry even when the
        // active plan paths leave some fields unread.
        #[expect(
            dead_code,
            reason = "fields are recorded via macro and subset of metric fields may be unread depending on active plan paths"
        )]
        pub(crate) struct NativeMetricRegistry {
            $(pub(super) $field: metric_storage_type!($kind),)*
        }

        impl NativeMetricRegistry {
            pub(super) fn register(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
                Self {
                    $($field: register_metric!(metrics, partition, $name, $kind),)*
                }
            }
        }
    };
}

for_each_native_algeon_metric!(define_registry);
