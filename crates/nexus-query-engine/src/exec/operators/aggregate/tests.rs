use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use arrow::datatypes::{DataType, Field, Schema};

use super::{
    filtered_global::{
        FilteredGlobalSegmentRuntimeStats, record_filtered_global_segmented_failure,
        select_filtered_global_aggregate_strategy,
    },
    filtered_grouped::select_filtered_grouped_aggregate_strategy,
    partial_strategy::{
        ExactGroupedKeyCardinalityProbe, ExactSingleKeyCardinalityProbe, aggregate_structural_key,
        exact_cardinality_probe_should_select_prepared_sort_groupby, exact_grouped_key_cardinality_probe_candidate,
        exact_grouped_key_cardinality_probe_should_select_prepared_sort_groupby,
        exact_single_key_cardinality_probe_candidate, exact_single_key_cardinality_probe_candidate_with_options,
        prepared_sort_groupby_should_select_for_runtime_facts, reduction_ratio_ppm,
        reject_prepared_sort_groupby_for_runtime_facts, select_partial_aggregate_strategy_decision,
        select_partial_aggregate_strategy_decision_with_options, select_prepared_sort_groupby_for_runtime_facts,
        source_adjacent_prepared_sort_groupby_runtime_candidate,
    },
    *,
};
use crate::{
    capability::{AggregateStrategyExecutionOptions, aggregate_strategy_capability_decision},
    error::{QueryError, QueryErrorCode},
    metrics::QueryMetrics,
    runtime::MemoryControlPolicy,
};

include!("tests/support.rs");
include!("tests/filtered_strategy.rs");
include!("tests/source_adjacent.rs");
include!("tests/runtime_probe.rs");
include!("tests/prior_evidence.rs");
