use std::time::{Duration, Instant};

use query_engine::{dimension_pruning::local_dimension_pruning_resolver, rapids_backend::rapids_analysis_context};
use query_planning::{
    admission::{admit, bounded_join_unsupported_shape_facts},
    capability::{JoinStrategy, KeyedHashJoinStrategyOptions},
    dynamic_filter::{
        DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterProducer,
        LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON,
    },
    expr::{CaseWhen, ColumnRef},
    optimizer::{
        DynamicFilterRouteDiscoveryOptions, JoinBuildSideSelectionOptions, NativeOptimizerOptions, optimize_query_plan,
    },
    pipeline::{PipelineTaskKind, compile_pipeline},
};
use query_runtime::{metrics::JoinStrategyDecisionMetrics, runtime::ExecutionCancellation};
use support::{
    join_source_with_projected_row_group_estimates, source_adjacent_join_filter_predicate,
    write_category_bucket_parquet, write_dynamic_filter_probe_parquet, write_dynamic_filter_probe_rows_parquet,
    write_partsupp_dynamic_filter_parquet, write_q72_d1_parquet, write_q72_d2_parquet, write_q72_inventory_parquet,
    write_q72_sales_parquet, write_row_grouped_join_left_with_payload_parquet, write_single_i32_parquet,
    write_source_adjacent_join_filter_left_parquet, write_source_adjacent_join_filter_right_parquet,
};

use super::*;

mod budget;
mod dynamic_filter;
mod partitioned;
mod partitioned_build;
mod partitioned_probe;
mod source_adjacent;
mod support;
