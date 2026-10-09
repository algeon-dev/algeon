use std::{collections::BTreeMap, sync::Arc, time::Duration};

use arrow::datatypes::{DataType, Field, Schema};
use query_planning::{
    capability::{AggregateStrategyExecutionOptions, aggregate_strategy_capability_decision},
    error::{MemoryFailure, MemoryUsage, QueryError, QueryErrorCode},
    execution_contracts::MemoryControlPolicy,
};
use query_runtime::{
    observability::{PlanExecutionId, RuntimeObservationConfig, RuntimeObservationSession},
    runtime::StandaloneConfig,
};
use support::{
    filtered_global_q09_like_spec, filtered_global_tpcds_q09_sf100_spec, filtered_grouped_q09_like_spec,
    fixed_width_sum_spec, metrics_with_prior_partial_decision, metrics_with_prior_partial_structural_decision,
    multi_key_sum_spec, partial_strategy_for_multi_key_sum, simple_string_avg_spec, simple_string_sum_spec,
    source_adjacent_group_key_evidence, source_adjacent_unpartitioned_evidence,
};

use super::{
    filtered_global::{
        FilteredGlobalSegmentRuntimeStats, record_filtered_global_segmented_failure,
        select_filtered_global_aggregate_strategy,
    },
    filtered_grouped::select_filtered_grouped_aggregate_strategy,
    partial_strategy::{
        aggregate_structural_key, prepared_sort_groupby_should_select_for_runtime_facts, reduction_ratio_ppm,
        reject_prepared_sort_groupby_for_runtime_facts, select_partial_aggregate_strategy_decision,
        select_partial_aggregate_strategy_decision_with_options, select_prepared_sort_groupby_for_runtime_facts,
        source_adjacent_prepared_sort_groupby_runtime_candidate,
    },
    *,
};

mod filtered_strategy;
mod prior_evidence;
mod runtime_probe;
mod source_adjacent;
mod support;
