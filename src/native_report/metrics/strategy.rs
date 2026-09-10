use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::MetricsSet;

use super::labels::{
    labels_by_name, max_optional_usize, optional_str, optional_string_label, optional_usize, optional_usize_label,
    string_label, usize_label,
};

const NATIVE_AGGREGATE_STRATEGY_ROWS_METRIC_NAME: &str = "native_aggregate_strategy_rows";
const NATIVE_JOIN_STRATEGY_ROWS_METRIC_NAME: &str = "native_join_strategy_rows";

/// Aggregate strategy diagnostics with runtime counters merged by aggregate site.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeAggregateStrategyMetricRow {
    /// Native aggregate node id, or a stage-local fallback label.
    pub node_id: String,
    /// Strategy selected for this aggregate stage.
    pub selected_strategy: String,
    /// Aggregate execution stage label.
    pub execution_stage: String,
    /// Native memory mode label.
    pub memory_mode: String,
    /// Number of group-key expressions.
    pub group_key_count: usize,
    /// Stable group-key expression summary.
    pub group_key_exprs: String,
    /// Stable group-key dtype summary.
    pub group_key_dtypes: String,
    /// Stable aggregate function summary.
    pub aggregate_functions: String,
    /// Whether this aggregate row carries a floating statistical aggregate contract.
    pub floating_statistical_aggregate: String,
    /// Selected floating statistical aggregate execution mode.
    pub statistical_aggregate_execution_mode: String,
    /// Required tolerance contract for floating statistical parity.
    pub statistical_aggregate_tolerance_contract: String,
    /// Floating statistical aggregate functions covered by the contract.
    pub statistical_aggregate_functions: String,
    /// Degrees-of-freedom value used by statistical aggregate kernels.
    pub statistical_aggregate_ddof: String,
    /// Count of aggregate filter predicates.
    pub aggregate_filter_count: usize,
    /// Filtered aggregate input materialization strategy.
    pub filtered_input_materialization_strategy: String,
    /// Estimated bytes for single-pass filtered input materialization.
    pub filtered_input_materialization_estimated_bytes: usize,
    /// Budget bytes visible at the filtered-input strategy decision.
    pub filtered_input_materialization_budget_bytes: usize,
    /// Single-pass filtered-input safety threshold in bytes.
    pub filtered_input_materialization_threshold_bytes: usize,
    /// Row count targeted by the segmented filtered global strategy.
    pub filtered_global_rows_per_chunk: usize,
    /// Filtered input expressions materialized together by the segmented strategy.
    pub filtered_global_inputs_per_segment: usize,
    /// Target temporary bytes for each segmented filtered-global materialization.
    pub filtered_global_segment_target_temporary_bytes: usize,
    /// Actual row chunks processed by the segmented filtered global strategy.
    pub filtered_global_actual_chunk_count: usize,
    /// Actual expression segments processed by the segmented filtered global strategy.
    pub filtered_global_actual_segment_count: usize,
    /// Maximum input rows processed by one segmented filtered global row chunk.
    pub filtered_global_max_segment_rows: usize,
    /// Maximum lower-level temporary peak observed during segmented filtered-global execution.
    pub filtered_global_observed_temporary_peak_bytes: usize,
    /// Maximum lower-level retained bytes observed during segmented filtered-global execution.
    pub filtered_global_observed_retained_peak_bytes: usize,
    /// Partial state rows merged by the segmented filtered global strategy.
    pub filtered_global_state_rows_merged: usize,
    /// Row count targeted by the segmented filtered grouped strategy.
    pub filtered_grouped_rows_per_chunk: usize,
    /// Filtered input expressions materialized together by the segmented grouped strategy.
    pub filtered_grouped_inputs_per_segment: usize,
    /// Target temporary bytes for each segmented filtered-grouped materialization.
    pub filtered_grouped_segment_target_temporary_bytes: usize,
    /// Actual row chunks processed by the segmented filtered grouped strategy.
    pub filtered_grouped_actual_chunk_count: usize,
    /// Actual expression segments processed by the segmented filtered grouped strategy.
    pub filtered_grouped_actual_segment_count: usize,
    /// Maximum input rows processed by one segmented filtered grouped row chunk.
    pub filtered_grouped_max_segment_rows: usize,
    /// Maximum lower-level temporary peak observed during segmented filtered-grouped execution.
    pub filtered_grouped_observed_temporary_peak_bytes: usize,
    /// Maximum lower-level retained bytes observed during segmented filtered-grouped execution.
    pub filtered_grouped_observed_retained_peak_bytes: usize,
    /// Partial state rows merged by the segmented filtered grouped strategy.
    pub filtered_grouped_state_rows_merged: usize,
    /// Grouped filtered aggregate high-cardinality admission contract.
    pub filtered_grouped_high_cardinality_admission: String,
    /// Count of distinct aggregate expressions.
    pub distinct_aggregate_count: usize,
    /// Input rows consumed by this aggregate stage.
    pub input_rows: usize,
    /// Output groups produced by this aggregate stage.
    pub output_group_count: usize,
    /// `output_group_count / input_rows` scaled by 1,000,000.
    pub reduction_ratio_ppm: usize,
    /// Chunk rows for partial stages, or zero for non-partial stages.
    pub chunk_row_count: usize,
    /// Source chunk count visible at this decision point.
    pub chunk_count: usize,
    /// Partial aggregate count visible at this decision point.
    pub partial_aggregate_count: usize,
    /// Whether this partial aggregate consumed source-adjacent stage output.
    pub source_adjacent_partial: String,
    /// Source row groups feeding this partial aggregate, or `unknown`.
    pub source_chunk_row_group_count: String,
    /// Whether the source chunk reads a whole file with unknown row groups.
    pub source_chunk_reads_all_row_groups: String,
    /// Known source row count for this chunk before source-adjacent stages.
    pub source_chunk_known_row_count: String,
    /// Best available source-read byte estimate for this chunk.
    pub source_chunk_estimated_source_read_bytes: String,
    /// Whether chunk sizing relied on file-size estimates.
    pub source_chunk_size_estimate_based: String,
    /// Source-adjacent partition strategy, or `unknown`/`none`.
    pub source_adjacent_partition_strategy: String,
    /// Planned source-adjacent partition count, or `unknown`.
    pub source_adjacent_partition_count: String,
    /// Non-empty source-adjacent partition count, or `unknown`.
    pub source_adjacent_non_empty_partition_count: String,
    /// Group-key column count used for source-adjacent partitioning, or `unknown`.
    pub source_adjacent_partition_key_count: String,
    /// Largest number of chunk fragments feeding one source-adjacent partition.
    pub source_adjacent_max_partition_fan_in: String,
    /// Planned partition count for partitioned final merge, or `unknown`/`none`.
    pub partitioned_merge_partition_count: String,
    /// Non-empty partition count for partitioned final merge, or `unknown`/`none`.
    pub partitioned_merge_non_empty_partition_count: String,
    /// Largest fan-in for one partitioned final-merge bucket, or `unknown`/`none`.
    pub partitioned_merge_max_partition_fan_in: String,
    /// Strategy-selection elapsed nanoseconds.
    pub selection_elapsed_ns: usize,
    /// Key-encoding elapsed nanoseconds.
    pub encode_elapsed_ns: usize,
    /// Groupby elapsed nanoseconds.
    pub groupby_elapsed_ns: usize,
    /// Key-decoding elapsed nanoseconds.
    pub decode_elapsed_ns: usize,
    /// Total operator elapsed nanoseconds charged to this strategy.
    pub strategy_operator_elapsed_ns: usize,
    /// Stable structural fingerprint used for cross-node aggregate evidence reuse.
    pub aggregate_structural_key: String,
    /// Evidence source used by partial aggregate cardinality cost models.
    pub prior_cardinality_evidence_scope: String,
    /// Aggregate strategy execution option row version.
    pub aggregate_strategy_options_version: String,
    /// Encoded-key partial aggregation low-cardinality threshold in PPM.
    pub aggregate_encoded_key_partial_low_cardinality_threshold_ppm: usize,
    /// Minimum input rows before encoded-key partial aggregation may run.
    pub aggregate_encoded_key_partial_min_input_rows: usize,
    /// Maximum grouping-key columns eligible for exact grouped-key probes.
    pub aggregate_exact_grouped_key_cardinality_probe_max_key_columns: usize,
    /// Single-stage string-key AVG probe row threshold.
    pub aggregate_exact_single_string_avg_cardinality_probe_min_input_rows: usize,
    /// General single-key exact-cardinality probe row threshold.
    pub aggregate_exact_single_key_cardinality_probe_min_input_rows: usize,
    /// Grouped-key exact-cardinality probe row threshold.
    pub aggregate_exact_grouped_key_cardinality_probe_min_input_rows: usize,
    /// Maximum string-key groups accepted for prepared sort-backed groupby.
    pub aggregate_exact_single_string_key_prepared_max_groups: usize,
    /// Three-key exact-cardinality probe input-row cap.
    pub aggregate_exact_three_key_cardinality_probe_max_input_rows: usize,
    /// Five-key exact-cardinality probe input-row cap.
    pub aggregate_exact_five_key_cardinality_probe_max_input_rows: usize,
    /// Prepared sort-backed groupby low-cardinality threshold in PPM.
    pub aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm: usize,
    /// Source-adjacent prepared sort-backed groupby row-group cap.
    pub aggregate_source_adjacent_prepared_sort_groupby_max_row_groups: usize,
    /// Source-adjacent prepared sort-backed groupby source-read byte cap.
    pub aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes: usize,
    /// Source-adjacent prepared sort-backed groupby fan-in cap.
    pub aggregate_source_adjacent_prepared_sort_groupby_max_fan_in: usize,
    /// Runtime cost-model name used for encoded-key partial selection.
    pub encoded_key_partial_cost_model: String,
    /// Runtime cost-model status for encoded-key partial selection.
    pub encoded_key_partial_cost_model_status: String,
    /// Stable reason for the encoded-key partial runtime selection decision.
    pub encoded_key_partial_selection_reason: String,
    /// Encoded-key partial reduction-ratio threshold scaled by 1,000,000.
    pub encoded_key_partial_reduction_ratio_threshold_ppm: usize,
    /// Prior partial reduction ratio used by the encoded-key cost model.
    pub encoded_key_partial_prior_reduction_ratio_ppm: usize,
    /// Strategy used by the prior partial evidence row.
    pub encoded_key_partial_prior_strategy: String,
    /// Exact-cardinality probe status for a single-key partial aggregate.
    pub exact_cardinality_probe_status: String,
    /// Group-key column measured by the exact-cardinality probe.
    pub exact_cardinality_probe_key: String,
    /// Rows consumed by the exact-cardinality probe.
    pub exact_cardinality_probe_input_rows: usize,
    /// Non-null distinct values measured by the exact-cardinality probe.
    pub exact_cardinality_probe_non_null_distinct_count: String,
    /// Null rows observed by the exact-cardinality probe.
    pub exact_cardinality_probe_null_count: String,
    /// Group count measured by the exact-cardinality probe, including the null group.
    pub exact_cardinality_probe_num_groups: String,
    /// Exact-cardinality probe reduction ratio scaled by 1,000,000.
    pub exact_cardinality_probe_reduction_ratio_ppm: String,
    /// Exact-cardinality probe elapsed nanoseconds.
    pub exact_cardinality_probe_elapsed_ns: usize,
    /// Stable local rejection reason for prepared sort groupby.
    pub prepared_sort_groupby_rejected_reason: String,
    /// Prepared sort groupby contract status.
    pub prepared_sort_groupby_contract_status: String,
    /// Required contract label for prepared sort groupby.
    pub prepared_sort_groupby_required_contract: String,
    /// Runtime cost-model name used for prepared sort groupby selection.
    pub prepared_sort_groupby_cost_model: String,
    /// Runtime cost-model status for prepared sort groupby selection.
    pub prepared_sort_groupby_cost_model_status: String,
    /// Stable reason for the prepared sort groupby runtime selection decision.
    pub prepared_sort_groupby_selection_reason: String,
    /// Exact prepared group count, or `unknown` when the prepare step did not run.
    pub prepared_sort_groupby_num_groups: String,
    /// Exact grouped-key output bytes, or `unknown` when the prepare step did not run.
    pub prepared_sort_groupby_grouped_key_output_bytes: String,
    /// Exact maximum group size, or `unknown` when the prepare step did not run.
    pub prepared_sort_groupby_max_group_size: String,
    /// Stable candidate rejection summary.
    pub rejected_candidate_strategies: String,
}

impl NativeAggregateStrategyMetricRow {
    /// Returns true for the expensive first-observation path where libcudf
    /// groupby is selected to collect cardinality evidence for future chunks.
    pub fn is_libcudf_runtime_probe(&self) -> bool {
        self.execution_stage == "partial"
            && self.selected_strategy == "libcudf_groupby"
            && self.prepared_sort_groupby_cost_model_status == "selected_libcudf_runtime_probe"
            && self.prepared_sort_groupby_selection_reason == "runtime_cardinality_probe_required"
    }

    /// Returns true when a single-key exact-cardinality probe was executed.
    pub fn has_exact_cardinality_probe(&self) -> bool {
        self.execution_stage == "partial" && self.exact_cardinality_probe_status == "measured"
    }

    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "node_id={},selected_strategy={},execution_stage={},memory_mode={},group_key_count={},group_key_exprs={},group_key_dtypes={},aggregate_functions={},floating_statistical_aggregate={},statistical_aggregate_execution_mode={},statistical_aggregate_tolerance_contract={},statistical_aggregate_functions={},statistical_aggregate_ddof={},aggregate_filter_count={},filtered_input_materialization_strategy={},filtered_input_materialization_estimated_bytes={},filtered_input_materialization_budget_bytes={},filtered_input_materialization_threshold_bytes={},filtered_global_rows_per_chunk={},filtered_global_inputs_per_segment={},filtered_global_segment_target_temporary_bytes={},filtered_global_actual_chunk_count={},filtered_global_actual_segment_count={},filtered_global_max_segment_rows={},filtered_global_observed_temporary_peak_bytes={},filtered_global_observed_retained_peak_bytes={},filtered_global_state_rows_merged={},filtered_grouped_rows_per_chunk={},filtered_grouped_inputs_per_segment={},filtered_grouped_segment_target_temporary_bytes={},filtered_grouped_actual_chunk_count={},filtered_grouped_actual_segment_count={},filtered_grouped_max_segment_rows={},filtered_grouped_observed_temporary_peak_bytes={},filtered_grouped_observed_retained_peak_bytes={},filtered_grouped_state_rows_merged={},filtered_grouped_high_cardinality_admission={},distinct_aggregate_count={},input_rows={},output_group_count={},reduction_ratio_ppm={},chunk_row_count={},chunk_count={},partial_aggregate_count={},source_adjacent_partial={},source_chunk_row_group_count={},source_chunk_reads_all_row_groups={},source_chunk_known_row_count={},source_chunk_estimated_source_read_bytes={},source_chunk_size_estimate_based={},source_adjacent_partition_strategy={},source_adjacent_partition_count={},source_adjacent_non_empty_partition_count={},source_adjacent_partition_key_count={},source_adjacent_max_partition_fan_in={},partitioned_merge_partition_count={},partitioned_merge_non_empty_partition_count={},partitioned_merge_max_partition_fan_in={},selection_elapsed_ns={},encode_elapsed_ns={},groupby_elapsed_ns={},decode_elapsed_ns={},strategy_operator_elapsed_ns={},aggregate_structural_key={},prior_cardinality_evidence_scope={},aggregate_strategy_options_version={},aggregate_encoded_key_partial_low_cardinality_threshold_ppm={},aggregate_encoded_key_partial_min_input_rows={},aggregate_exact_grouped_key_cardinality_probe_max_key_columns={},aggregate_exact_single_string_avg_cardinality_probe_min_input_rows={},aggregate_exact_single_key_cardinality_probe_min_input_rows={},aggregate_exact_grouped_key_cardinality_probe_min_input_rows={},aggregate_exact_single_string_key_prepared_max_groups={},aggregate_exact_three_key_cardinality_probe_max_input_rows={},aggregate_exact_five_key_cardinality_probe_max_input_rows={},aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm={},aggregate_source_adjacent_prepared_sort_groupby_max_row_groups={},aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes={},aggregate_source_adjacent_prepared_sort_groupby_max_fan_in={},encoded_key_partial_cost_model={},encoded_key_partial_cost_model_status={},encoded_key_partial_selection_reason={},encoded_key_partial_reduction_ratio_threshold_ppm={},encoded_key_partial_prior_reduction_ratio_ppm={},encoded_key_partial_prior_strategy={},exact_cardinality_probe_status={},exact_cardinality_probe_key={},exact_cardinality_probe_input_rows={},exact_cardinality_probe_non_null_distinct_count={},exact_cardinality_probe_null_count={},exact_cardinality_probe_num_groups={},exact_cardinality_probe_reduction_ratio_ppm={},exact_cardinality_probe_elapsed_ns={},prepared_sort_groupby_rejected_reason={},prepared_sort_groupby_contract_status={},prepared_sort_groupby_required_contract={},prepared_sort_groupby_cost_model={},prepared_sort_groupby_cost_model_status={},prepared_sort_groupby_selection_reason={},prepared_sort_groupby_num_groups={},prepared_sort_groupby_grouped_key_output_bytes={},prepared_sort_groupby_max_group_size={},rejected_candidate_strategies={}",
            self.node_id,
            self.selected_strategy,
            self.execution_stage,
            self.memory_mode,
            self.group_key_count,
            self.group_key_exprs,
            self.group_key_dtypes,
            self.aggregate_functions,
            self.floating_statistical_aggregate,
            self.statistical_aggregate_execution_mode,
            self.statistical_aggregate_tolerance_contract,
            self.statistical_aggregate_functions,
            self.statistical_aggregate_ddof,
            self.aggregate_filter_count,
            self.filtered_input_materialization_strategy,
            self.filtered_input_materialization_estimated_bytes,
            self.filtered_input_materialization_budget_bytes,
            self.filtered_input_materialization_threshold_bytes,
            self.filtered_global_rows_per_chunk,
            self.filtered_global_inputs_per_segment,
            self.filtered_global_segment_target_temporary_bytes,
            self.filtered_global_actual_chunk_count,
            self.filtered_global_actual_segment_count,
            self.filtered_global_max_segment_rows,
            self.filtered_global_observed_temporary_peak_bytes,
            self.filtered_global_observed_retained_peak_bytes,
            self.filtered_global_state_rows_merged,
            self.filtered_grouped_rows_per_chunk,
            self.filtered_grouped_inputs_per_segment,
            self.filtered_grouped_segment_target_temporary_bytes,
            self.filtered_grouped_actual_chunk_count,
            self.filtered_grouped_actual_segment_count,
            self.filtered_grouped_max_segment_rows,
            self.filtered_grouped_observed_temporary_peak_bytes,
            self.filtered_grouped_observed_retained_peak_bytes,
            self.filtered_grouped_state_rows_merged,
            self.filtered_grouped_high_cardinality_admission,
            self.distinct_aggregate_count,
            self.input_rows,
            self.output_group_count,
            self.reduction_ratio_ppm,
            self.chunk_row_count,
            self.chunk_count,
            self.partial_aggregate_count,
            self.source_adjacent_partial,
            self.source_chunk_row_group_count,
            self.source_chunk_reads_all_row_groups,
            self.source_chunk_known_row_count,
            self.source_chunk_estimated_source_read_bytes,
            self.source_chunk_size_estimate_based,
            self.source_adjacent_partition_strategy,
            self.source_adjacent_partition_count,
            self.source_adjacent_non_empty_partition_count,
            self.source_adjacent_partition_key_count,
            self.source_adjacent_max_partition_fan_in,
            self.partitioned_merge_partition_count,
            self.partitioned_merge_non_empty_partition_count,
            self.partitioned_merge_max_partition_fan_in,
            self.selection_elapsed_ns,
            self.encode_elapsed_ns,
            self.groupby_elapsed_ns,
            self.decode_elapsed_ns,
            self.strategy_operator_elapsed_ns,
            self.aggregate_structural_key,
            self.prior_cardinality_evidence_scope,
            self.aggregate_strategy_options_version,
            self.aggregate_encoded_key_partial_low_cardinality_threshold_ppm,
            self.aggregate_encoded_key_partial_min_input_rows,
            self.aggregate_exact_grouped_key_cardinality_probe_max_key_columns,
            self.aggregate_exact_single_string_avg_cardinality_probe_min_input_rows,
            self.aggregate_exact_single_key_cardinality_probe_min_input_rows,
            self.aggregate_exact_grouped_key_cardinality_probe_min_input_rows,
            self.aggregate_exact_single_string_key_prepared_max_groups,
            self.aggregate_exact_three_key_cardinality_probe_max_input_rows,
            self.aggregate_exact_five_key_cardinality_probe_max_input_rows,
            self.aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm,
            self.aggregate_source_adjacent_prepared_sort_groupby_max_row_groups,
            self.aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes,
            self.aggregate_source_adjacent_prepared_sort_groupby_max_fan_in,
            self.encoded_key_partial_cost_model,
            self.encoded_key_partial_cost_model_status,
            self.encoded_key_partial_selection_reason,
            self.encoded_key_partial_reduction_ratio_threshold_ppm,
            self.encoded_key_partial_prior_reduction_ratio_ppm,
            self.encoded_key_partial_prior_strategy,
            self.exact_cardinality_probe_status,
            self.exact_cardinality_probe_key,
            self.exact_cardinality_probe_input_rows,
            self.exact_cardinality_probe_non_null_distinct_count,
            self.exact_cardinality_probe_null_count,
            self.exact_cardinality_probe_num_groups,
            self.exact_cardinality_probe_reduction_ratio_ppm,
            self.exact_cardinality_probe_elapsed_ns,
            self.prepared_sort_groupby_rejected_reason,
            self.prepared_sort_groupby_contract_status,
            self.prepared_sort_groupby_required_contract,
            self.prepared_sort_groupby_cost_model,
            self.prepared_sort_groupby_cost_model_status,
            self.prepared_sort_groupby_selection_reason,
            self.prepared_sort_groupby_num_groups,
            self.prepared_sort_groupby_grouped_key_output_bytes,
            self.prepared_sort_groupby_max_group_size,
            self.rejected_candidate_strategies,
        )
    }

    pub(super) fn same_site(&self, other: &Self) -> bool {
        self.node_id == other.node_id
            && self.selected_strategy == other.selected_strategy
            && self.execution_stage == other.execution_stage
            && self.memory_mode == other.memory_mode
            && self.group_key_count == other.group_key_count
            && self.group_key_exprs == other.group_key_exprs
            && self.group_key_dtypes == other.group_key_dtypes
            && self.aggregate_functions == other.aggregate_functions
            && self.floating_statistical_aggregate == other.floating_statistical_aggregate
            && self.statistical_aggregate_execution_mode == other.statistical_aggregate_execution_mode
            && self.statistical_aggregate_tolerance_contract == other.statistical_aggregate_tolerance_contract
            && self.statistical_aggregate_functions == other.statistical_aggregate_functions
            && self.statistical_aggregate_ddof == other.statistical_aggregate_ddof
            && self.aggregate_filter_count == other.aggregate_filter_count
            && self.filtered_input_materialization_strategy == other.filtered_input_materialization_strategy
            && self.filtered_input_materialization_estimated_bytes
                == other.filtered_input_materialization_estimated_bytes
            && self.filtered_input_materialization_budget_bytes == other.filtered_input_materialization_budget_bytes
            && self.filtered_input_materialization_threshold_bytes
                == other.filtered_input_materialization_threshold_bytes
            && self.filtered_global_rows_per_chunk == other.filtered_global_rows_per_chunk
            && self.filtered_global_inputs_per_segment == other.filtered_global_inputs_per_segment
            && self.filtered_global_segment_target_temporary_bytes
                == other.filtered_global_segment_target_temporary_bytes
            && self.filtered_grouped_rows_per_chunk == other.filtered_grouped_rows_per_chunk
            && self.filtered_grouped_inputs_per_segment == other.filtered_grouped_inputs_per_segment
            && self.filtered_grouped_segment_target_temporary_bytes
                == other.filtered_grouped_segment_target_temporary_bytes
            && self.filtered_grouped_high_cardinality_admission == other.filtered_grouped_high_cardinality_admission
            && self.distinct_aggregate_count == other.distinct_aggregate_count
            && self.source_adjacent_partial == other.source_adjacent_partial
            && self.source_chunk_row_group_count == other.source_chunk_row_group_count
            && self.source_chunk_reads_all_row_groups == other.source_chunk_reads_all_row_groups
            && self.source_chunk_known_row_count == other.source_chunk_known_row_count
            && self.source_chunk_estimated_source_read_bytes == other.source_chunk_estimated_source_read_bytes
            && self.source_chunk_size_estimate_based == other.source_chunk_size_estimate_based
            && self.source_adjacent_partition_strategy == other.source_adjacent_partition_strategy
            && self.source_adjacent_partition_count == other.source_adjacent_partition_count
            && self.source_adjacent_non_empty_partition_count == other.source_adjacent_non_empty_partition_count
            && self.source_adjacent_partition_key_count == other.source_adjacent_partition_key_count
            && self.source_adjacent_max_partition_fan_in == other.source_adjacent_max_partition_fan_in
            && self.partitioned_merge_partition_count == other.partitioned_merge_partition_count
            && self.partitioned_merge_non_empty_partition_count == other.partitioned_merge_non_empty_partition_count
            && self.partitioned_merge_max_partition_fan_in == other.partitioned_merge_max_partition_fan_in
            && self.aggregate_structural_key == other.aggregate_structural_key
            && self.prior_cardinality_evidence_scope == other.prior_cardinality_evidence_scope
            && self.aggregate_strategy_options_version == other.aggregate_strategy_options_version
            && self.aggregate_encoded_key_partial_low_cardinality_threshold_ppm
                == other.aggregate_encoded_key_partial_low_cardinality_threshold_ppm
            && self.aggregate_encoded_key_partial_min_input_rows == other.aggregate_encoded_key_partial_min_input_rows
            && self.aggregate_exact_grouped_key_cardinality_probe_max_key_columns
                == other.aggregate_exact_grouped_key_cardinality_probe_max_key_columns
            && self.aggregate_exact_single_string_avg_cardinality_probe_min_input_rows
                == other.aggregate_exact_single_string_avg_cardinality_probe_min_input_rows
            && self.aggregate_exact_single_key_cardinality_probe_min_input_rows
                == other.aggregate_exact_single_key_cardinality_probe_min_input_rows
            && self.aggregate_exact_grouped_key_cardinality_probe_min_input_rows
                == other.aggregate_exact_grouped_key_cardinality_probe_min_input_rows
            && self.aggregate_exact_single_string_key_prepared_max_groups
                == other.aggregate_exact_single_string_key_prepared_max_groups
            && self.aggregate_exact_three_key_cardinality_probe_max_input_rows
                == other.aggregate_exact_three_key_cardinality_probe_max_input_rows
            && self.aggregate_exact_five_key_cardinality_probe_max_input_rows
                == other.aggregate_exact_five_key_cardinality_probe_max_input_rows
            && self.aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm
                == other.aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm
            && self.aggregate_source_adjacent_prepared_sort_groupby_max_row_groups
                == other.aggregate_source_adjacent_prepared_sort_groupby_max_row_groups
            && self.aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes
                == other.aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes
            && self.aggregate_source_adjacent_prepared_sort_groupby_max_fan_in
                == other.aggregate_source_adjacent_prepared_sort_groupby_max_fan_in
            && self.encoded_key_partial_cost_model == other.encoded_key_partial_cost_model
            && self.encoded_key_partial_cost_model_status == other.encoded_key_partial_cost_model_status
            && self.encoded_key_partial_selection_reason == other.encoded_key_partial_selection_reason
            && self.encoded_key_partial_reduction_ratio_threshold_ppm
                == other.encoded_key_partial_reduction_ratio_threshold_ppm
            && self.encoded_key_partial_prior_reduction_ratio_ppm == other.encoded_key_partial_prior_reduction_ratio_ppm
            && self.encoded_key_partial_prior_strategy == other.encoded_key_partial_prior_strategy
            && self.exact_cardinality_probe_status == other.exact_cardinality_probe_status
            && self.exact_cardinality_probe_key == other.exact_cardinality_probe_key
            && self.exact_cardinality_probe_non_null_distinct_count
                == other.exact_cardinality_probe_non_null_distinct_count
            && self.exact_cardinality_probe_null_count == other.exact_cardinality_probe_null_count
            && self.exact_cardinality_probe_num_groups == other.exact_cardinality_probe_num_groups
            && self.exact_cardinality_probe_reduction_ratio_ppm == other.exact_cardinality_probe_reduction_ratio_ppm
            && self.prepared_sort_groupby_rejected_reason == other.prepared_sort_groupby_rejected_reason
            && self.prepared_sort_groupby_contract_status == other.prepared_sort_groupby_contract_status
            && self.prepared_sort_groupby_required_contract == other.prepared_sort_groupby_required_contract
            && self.prepared_sort_groupby_cost_model == other.prepared_sort_groupby_cost_model
            && self.prepared_sort_groupby_cost_model_status == other.prepared_sort_groupby_cost_model_status
            && self.prepared_sort_groupby_selection_reason == other.prepared_sort_groupby_selection_reason
            && self.rejected_candidate_strategies == other.rejected_candidate_strategies
    }

    pub(super) fn merge_runtime_from(&mut self, other: Self) {
        self.input_rows = self.input_rows.saturating_add(other.input_rows);
        self.output_group_count = self.output_group_count.saturating_add(other.output_group_count);
        self.chunk_row_count = self.chunk_row_count.saturating_add(other.chunk_row_count);
        self.chunk_count = self.chunk_count.max(other.chunk_count);
        self.partial_aggregate_count = self.partial_aggregate_count.max(other.partial_aggregate_count);
        self.filtered_global_state_rows_merged =
            self.filtered_global_state_rows_merged.saturating_add(other.filtered_global_state_rows_merged);
        self.filtered_global_actual_chunk_count =
            self.filtered_global_actual_chunk_count.saturating_add(other.filtered_global_actual_chunk_count);
        self.filtered_global_actual_segment_count =
            self.filtered_global_actual_segment_count.saturating_add(other.filtered_global_actual_segment_count);
        self.filtered_global_max_segment_rows =
            self.filtered_global_max_segment_rows.max(other.filtered_global_max_segment_rows);
        self.filtered_global_observed_temporary_peak_bytes =
            self.filtered_global_observed_temporary_peak_bytes.max(other.filtered_global_observed_temporary_peak_bytes);
        self.filtered_global_observed_retained_peak_bytes =
            self.filtered_global_observed_retained_peak_bytes.max(other.filtered_global_observed_retained_peak_bytes);
        self.filtered_grouped_state_rows_merged =
            self.filtered_grouped_state_rows_merged.saturating_add(other.filtered_grouped_state_rows_merged);
        self.filtered_grouped_actual_chunk_count =
            self.filtered_grouped_actual_chunk_count.saturating_add(other.filtered_grouped_actual_chunk_count);
        self.filtered_grouped_actual_segment_count =
            self.filtered_grouped_actual_segment_count.saturating_add(other.filtered_grouped_actual_segment_count);
        self.filtered_grouped_max_segment_rows =
            self.filtered_grouped_max_segment_rows.max(other.filtered_grouped_max_segment_rows);
        self.filtered_grouped_observed_temporary_peak_bytes = self
            .filtered_grouped_observed_temporary_peak_bytes
            .max(other.filtered_grouped_observed_temporary_peak_bytes);
        self.filtered_grouped_observed_retained_peak_bytes =
            self.filtered_grouped_observed_retained_peak_bytes.max(other.filtered_grouped_observed_retained_peak_bytes);
        self.selection_elapsed_ns = self.selection_elapsed_ns.saturating_add(other.selection_elapsed_ns);
        self.encode_elapsed_ns = self.encode_elapsed_ns.saturating_add(other.encode_elapsed_ns);
        self.groupby_elapsed_ns = self.groupby_elapsed_ns.saturating_add(other.groupby_elapsed_ns);
        self.decode_elapsed_ns = self.decode_elapsed_ns.saturating_add(other.decode_elapsed_ns);
        self.strategy_operator_elapsed_ns =
            self.strategy_operator_elapsed_ns.saturating_add(other.strategy_operator_elapsed_ns);
        self.exact_cardinality_probe_input_rows =
            self.exact_cardinality_probe_input_rows.saturating_add(other.exact_cardinality_probe_input_rows);
        self.exact_cardinality_probe_elapsed_ns =
            self.exact_cardinality_probe_elapsed_ns.saturating_add(other.exact_cardinality_probe_elapsed_ns);
        self.prepared_sort_groupby_num_groups = merge_numeric_string_fact_sum(
            &self.prepared_sort_groupby_num_groups,
            &other.prepared_sort_groupby_num_groups,
        );
        self.prepared_sort_groupby_grouped_key_output_bytes = merge_numeric_string_fact_sum(
            &self.prepared_sort_groupby_grouped_key_output_bytes,
            &other.prepared_sort_groupby_grouped_key_output_bytes,
        );
        self.prepared_sort_groupby_max_group_size = merge_numeric_string_fact_max(
            &self.prepared_sort_groupby_max_group_size,
            &other.prepared_sort_groupby_max_group_size,
        );
        self.reduction_ratio_ppm = if self.input_rows == 0 {
            0
        } else {
            self.output_group_count.saturating_mul(1_000_000).saturating_div(self.input_rows)
        };
    }

    pub(super) fn sort_key(&self) -> (&str, &str, &str, &str, &str) {
        (
            self.node_id.as_str(),
            self.execution_stage.as_str(),
            self.selected_strategy.as_str(),
            self.group_key_dtypes.as_str(),
            self.aggregate_functions.as_str(),
        )
    }
}

fn merge_numeric_string_fact_sum(left: &str, right: &str) -> String {
    match (left.parse::<usize>(), right.parse::<usize>()) {
        (Ok(left), Ok(right)) => left.saturating_add(right).to_string(),
        _ => "unknown".to_string(),
    }
}

fn merge_numeric_string_fact_max(left: &str, right: &str) -> String {
    match (left.parse::<usize>(), right.parse::<usize>()) {
        (Ok(left), Ok(right)) => left.max(right).to_string(),
        _ => "unknown".to_string(),
    }
}

/// Join strategy diagnostics with runtime counters merged by join site.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeJoinStrategyMetricRow {
    /// Native join node id.
    pub node_id: String,
    /// Planner-selected join strategy label.
    pub join_strategy: String,
    /// Runtime execution strategy label.
    pub execution_strategy: String,
    /// Join type label.
    pub join_type: String,
    /// Build-side orientation label.
    pub build_side: String,
    /// Number of equality key pairs.
    pub key_count: usize,
    /// Estimated build-side rows, if known.
    pub estimated_build_rows: Option<usize>,
    /// Observed build-side rows, if known.
    pub observed_build_rows: Option<usize>,
    /// Build-side input bytes, if known.
    pub build_side_bytes: Option<usize>,
    /// Probe-side input bytes, if known.
    pub probe_side_bytes: Option<usize>,
    /// Estimated output rows, if known.
    pub estimated_output_rows: Option<usize>,
    /// Estimated output bytes, if known.
    pub estimated_output_bytes: Option<usize>,
    /// Exact build-key uniqueness proof, if measured.
    pub build_key_unique: Option<String>,
    /// Runtime build-key uniqueness probe status, if attempted.
    pub build_key_uniqueness_probe_status: Option<String>,
    /// Error code observed when a uniqueness probe fell back, if any.
    pub build_key_uniqueness_probe_error_code: Option<String>,
    /// Probe reuse count, if known.
    pub probe_reuse_count: Option<usize>,
    /// Partition count for partitioned join strategies, if known.
    pub partition_count: Option<usize>,
    /// Partition target bytes for partitioned join strategies, if known.
    pub partition_target_bytes: Option<usize>,
    /// Build partition batch count, if known.
    pub build_partition_batch_count: Option<usize>,
    /// Probe partition batch count, if known.
    pub probe_partition_batch_count: Option<usize>,
    /// Build-side partitioned bytes, if known.
    pub build_partitioned_bytes: Option<usize>,
    /// Probe-side partitioned bytes, if known.
    pub probe_partitioned_bytes: Option<usize>,
    /// Rejection reason for reusable hash join, or `None` when not rejected/reported.
    pub reusable_hash_join_rejected_reason: Option<String>,
    /// Rejection reason for partitioned hash join, or `None` when not rejected/reported.
    pub partitioned_hash_join_rejected_reason: Option<String>,
    /// Rejection reason for one-shot equi join, or `None` when not rejected/reported.
    pub one_shot_equi_rejected_reason: Option<String>,
    /// Count of partitioned hash joins executed at this site.
    pub partitioned_hash_join_count: usize,
    /// Count of reusable hash-join builds executed at this site.
    pub hash_join_build_count: usize,
    /// Device bytes retained by hash-join build artifacts.
    pub hash_join_hash_table_bytes: usize,
    /// Count of filtered-join builds executed at this site.
    pub filtered_join_build_count: usize,
    /// Device bytes retained by filtered-join build artifacts.
    pub filtered_join_hash_table_bytes: usize,
    /// Device bytes retained by filtered-join probe states.
    pub filtered_join_probe_state_bytes: usize,
    /// Device bytes retained by mixed join size-data states.
    pub mixed_join_size_data_bytes: usize,
    /// Count of one-shot joins executed at this site.
    pub one_shot_join_count: usize,
    /// Device bytes retained by one-shot join index maps.
    pub one_shot_join_index_bytes: usize,
    /// Reservation bytes requested for one-shot output materialization.
    pub one_shot_join_materialization_bytes: usize,
    /// Observed device bytes of one-shot join output frames.
    pub one_shot_join_output_bytes: usize,
    /// Device bytes retained by join gather maps.
    pub join_gather_map_bytes: usize,
    /// Bytes reserved for join output materialization.
    pub join_output_reservation_bytes: usize,
    /// Join output row count.
    pub join_output_rows: usize,
    /// Join output bytes.
    pub join_output_bytes: usize,
}

impl NativeJoinStrategyMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "node_id={},join_strategy={},execution_strategy={},join_type={},build_side={},key_count={},estimated_build_rows={},observed_build_rows={},build_side_bytes={},probe_side_bytes={},estimated_output_rows={},estimated_output_bytes={},build_key_unique={},build_key_uniqueness_probe_status={},build_key_uniqueness_probe_error_code={},probe_reuse_count={},partition_count={},partition_target_bytes={},build_partition_batch_count={},probe_partition_batch_count={},build_partitioned_bytes={},probe_partitioned_bytes={},reusable_hash_join_rejected_reason={},partitioned_hash_join_rejected_reason={},one_shot_equi_rejected_reason={},partitioned_hash_join_count={},hash_join_build_count={},hash_join_hash_table_bytes={},filtered_join_build_count={},filtered_join_hash_table_bytes={},filtered_join_probe_state_bytes={},mixed_join_size_data_bytes={},one_shot_join_count={},one_shot_join_index_bytes={},one_shot_join_materialization_bytes={},one_shot_join_output_bytes={},join_gather_map_bytes={},join_output_reservation_bytes={},join_output_rows={},join_output_bytes={}",
            self.node_id,
            self.join_strategy,
            self.execution_strategy,
            self.join_type,
            self.build_side,
            self.key_count,
            optional_usize(self.estimated_build_rows),
            optional_usize(self.observed_build_rows),
            optional_usize(self.build_side_bytes),
            optional_usize(self.probe_side_bytes),
            optional_usize(self.estimated_output_rows),
            optional_usize(self.estimated_output_bytes),
            optional_str(self.build_key_unique.as_deref()),
            optional_str(self.build_key_uniqueness_probe_status.as_deref()),
            optional_str(self.build_key_uniqueness_probe_error_code.as_deref()),
            optional_usize(self.probe_reuse_count),
            optional_usize(self.partition_count),
            optional_usize(self.partition_target_bytes),
            optional_usize(self.build_partition_batch_count),
            optional_usize(self.probe_partition_batch_count),
            optional_usize(self.build_partitioned_bytes),
            optional_usize(self.probe_partitioned_bytes),
            optional_str(self.reusable_hash_join_rejected_reason.as_deref()),
            optional_str(self.partitioned_hash_join_rejected_reason.as_deref()),
            optional_str(self.one_shot_equi_rejected_reason.as_deref()),
            self.partitioned_hash_join_count,
            self.hash_join_build_count,
            self.hash_join_hash_table_bytes,
            self.filtered_join_build_count,
            self.filtered_join_hash_table_bytes,
            self.filtered_join_probe_state_bytes,
            self.mixed_join_size_data_bytes,
            self.one_shot_join_count,
            self.one_shot_join_index_bytes,
            self.one_shot_join_materialization_bytes,
            self.one_shot_join_output_bytes,
            self.join_gather_map_bytes,
            self.join_output_reservation_bytes,
            self.join_output_rows,
            self.join_output_bytes,
        )
    }

    pub(super) fn same_site(&self, other: &Self) -> bool {
        self.node_id == other.node_id
            && self.join_strategy == other.join_strategy
            && self.execution_strategy == other.execution_strategy
            && self.join_type == other.join_type
            && self.build_side == other.build_side
            && self.key_count == other.key_count
            && self.estimated_build_rows == other.estimated_build_rows
            && self.estimated_output_rows == other.estimated_output_rows
            && self.estimated_output_bytes == other.estimated_output_bytes
            && self.build_key_unique == other.build_key_unique
            && self.build_key_uniqueness_probe_status == other.build_key_uniqueness_probe_status
            && self.build_key_uniqueness_probe_error_code == other.build_key_uniqueness_probe_error_code
            && self.probe_reuse_count == other.probe_reuse_count
            && self.partition_count == other.partition_count
            && self.partition_target_bytes == other.partition_target_bytes
            && self.reusable_hash_join_rejected_reason == other.reusable_hash_join_rejected_reason
            && self.partitioned_hash_join_rejected_reason == other.partitioned_hash_join_rejected_reason
            && self.one_shot_equi_rejected_reason == other.one_shot_equi_rejected_reason
    }

    pub(super) fn merge_runtime_from(&mut self, other: Self) {
        self.observed_build_rows = max_optional_usize(self.observed_build_rows, other.observed_build_rows);
        self.build_side_bytes = max_optional_usize(self.build_side_bytes, other.build_side_bytes);
        self.probe_side_bytes = max_optional_usize(self.probe_side_bytes, other.probe_side_bytes);
        self.estimated_output_rows = max_optional_usize(self.estimated_output_rows, other.estimated_output_rows);
        self.estimated_output_bytes = max_optional_usize(self.estimated_output_bytes, other.estimated_output_bytes);
        self.build_partition_batch_count =
            max_optional_usize(self.build_partition_batch_count, other.build_partition_batch_count);
        self.probe_partition_batch_count =
            max_optional_usize(self.probe_partition_batch_count, other.probe_partition_batch_count);
        self.build_partitioned_bytes = max_optional_usize(self.build_partitioned_bytes, other.build_partitioned_bytes);
        self.probe_partitioned_bytes = max_optional_usize(self.probe_partitioned_bytes, other.probe_partitioned_bytes);
        self.partitioned_hash_join_count =
            self.partitioned_hash_join_count.saturating_add(other.partitioned_hash_join_count);
        self.hash_join_build_count = self.hash_join_build_count.saturating_add(other.hash_join_build_count);
        self.hash_join_hash_table_bytes =
            self.hash_join_hash_table_bytes.saturating_add(other.hash_join_hash_table_bytes);
        self.filtered_join_build_count = self.filtered_join_build_count.saturating_add(other.filtered_join_build_count);
        self.filtered_join_hash_table_bytes =
            self.filtered_join_hash_table_bytes.saturating_add(other.filtered_join_hash_table_bytes);
        self.filtered_join_probe_state_bytes =
            self.filtered_join_probe_state_bytes.saturating_add(other.filtered_join_probe_state_bytes);
        self.mixed_join_size_data_bytes =
            self.mixed_join_size_data_bytes.saturating_add(other.mixed_join_size_data_bytes);
        self.one_shot_join_count = self.one_shot_join_count.saturating_add(other.one_shot_join_count);
        self.one_shot_join_index_bytes = self.one_shot_join_index_bytes.saturating_add(other.one_shot_join_index_bytes);
        self.one_shot_join_materialization_bytes =
            self.one_shot_join_materialization_bytes.saturating_add(other.one_shot_join_materialization_bytes);
        self.one_shot_join_output_bytes =
            self.one_shot_join_output_bytes.saturating_add(other.one_shot_join_output_bytes);
        self.join_gather_map_bytes = self.join_gather_map_bytes.saturating_add(other.join_gather_map_bytes);
        self.join_output_reservation_bytes =
            self.join_output_reservation_bytes.saturating_add(other.join_output_reservation_bytes);
        self.join_output_rows = self.join_output_rows.saturating_add(other.join_output_rows);
        self.join_output_bytes = self.join_output_bytes.saturating_add(other.join_output_bytes);
    }

    pub(super) fn sort_key(&self) -> (&str, &str, &str, &str, &str, usize) {
        (
            self.node_id.as_str(),
            self.join_strategy.as_str(),
            self.execution_strategy.as_str(),
            self.join_type.as_str(),
            self.build_side.as_str(),
            self.key_count,
        )
    }
}

pub(super) fn join_strategy_rows(metrics: &MetricsSet) -> Vec<NativeJoinStrategyMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_JOIN_STRATEGY_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "node_id"))
        .map(|metric| labels_to_join_strategy_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    rows
}

pub(super) fn aggregate_strategy_rows(metrics: &MetricsSet) -> Vec<NativeAggregateStrategyMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_AGGREGATE_STRATEGY_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "node_id"))
        .map(|metric| labels_to_aggregate_strategy_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    rows
}

fn labels_to_join_strategy_row(labels: BTreeMap<&str, &str>) -> NativeJoinStrategyMetricRow {
    NativeJoinStrategyMetricRow {
        node_id: string_label(&labels, "node_id"),
        join_strategy: string_label(&labels, "join_strategy"),
        execution_strategy: string_label(&labels, "execution_strategy"),
        join_type: string_label(&labels, "join_type"),
        build_side: string_label(&labels, "build_side"),
        key_count: usize_label(&labels, "key_count"),
        estimated_build_rows: optional_usize_label(&labels, "estimated_build_rows"),
        observed_build_rows: optional_usize_label(&labels, "observed_build_rows"),
        build_side_bytes: optional_usize_label(&labels, "build_side_bytes"),
        probe_side_bytes: optional_usize_label(&labels, "probe_side_bytes"),
        estimated_output_rows: optional_usize_label(&labels, "estimated_output_rows"),
        estimated_output_bytes: optional_usize_label(&labels, "estimated_output_bytes"),
        build_key_unique: optional_string_label(&labels, "build_key_unique"),
        build_key_uniqueness_probe_status: optional_string_label(&labels, "build_key_uniqueness_probe_status"),
        build_key_uniqueness_probe_error_code: optional_string_label(&labels, "build_key_uniqueness_probe_error_code"),
        probe_reuse_count: optional_usize_label(&labels, "probe_reuse_count"),
        partition_count: optional_usize_label(&labels, "partition_count"),
        partition_target_bytes: optional_usize_label(&labels, "partition_target_bytes"),
        build_partition_batch_count: optional_usize_label(&labels, "build_partition_batch_count"),
        probe_partition_batch_count: optional_usize_label(&labels, "probe_partition_batch_count"),
        build_partitioned_bytes: optional_usize_label(&labels, "build_partitioned_bytes"),
        probe_partitioned_bytes: optional_usize_label(&labels, "probe_partitioned_bytes"),
        reusable_hash_join_rejected_reason: optional_string_label(&labels, "reusable_hash_join_rejected_reason"),
        partitioned_hash_join_rejected_reason: optional_string_label(&labels, "partitioned_hash_join_rejected_reason"),
        one_shot_equi_rejected_reason: optional_string_label(&labels, "one_shot_equi_rejected_reason"),
        partitioned_hash_join_count: usize_label(&labels, "partitioned_hash_join_count"),
        hash_join_build_count: usize_label(&labels, "hash_join_build_count"),
        hash_join_hash_table_bytes: usize_label(&labels, "hash_join_hash_table_bytes"),
        filtered_join_build_count: usize_label(&labels, "filtered_join_build_count"),
        filtered_join_hash_table_bytes: usize_label(&labels, "filtered_join_hash_table_bytes"),
        filtered_join_probe_state_bytes: usize_label(&labels, "filtered_join_probe_state_bytes"),
        mixed_join_size_data_bytes: usize_label(&labels, "mixed_join_size_data_bytes"),
        one_shot_join_count: usize_label(&labels, "one_shot_join_count"),
        one_shot_join_index_bytes: usize_label(&labels, "one_shot_join_index_bytes"),
        one_shot_join_materialization_bytes: usize_label(&labels, "one_shot_join_materialization_bytes"),
        one_shot_join_output_bytes: usize_label(&labels, "one_shot_join_output_bytes"),
        join_gather_map_bytes: usize_label(&labels, "join_gather_map_bytes"),
        join_output_reservation_bytes: usize_label(&labels, "join_output_reservation_bytes"),
        join_output_rows: usize_label(&labels, "join_output_rows"),
        join_output_bytes: usize_label(&labels, "join_output_bytes"),
    }
}

fn labels_to_aggregate_strategy_row(labels: BTreeMap<&str, &str>) -> NativeAggregateStrategyMetricRow {
    NativeAggregateStrategyMetricRow {
        node_id: string_label(&labels, "node_id"),
        selected_strategy: string_label(&labels, "selected_strategy"),
        execution_stage: string_label(&labels, "execution_stage"),
        memory_mode: string_label(&labels, "memory_mode"),
        group_key_count: usize_label(&labels, "group_key_count"),
        group_key_exprs: string_label(&labels, "group_key_exprs"),
        group_key_dtypes: string_label(&labels, "group_key_dtypes"),
        aggregate_functions: string_label(&labels, "aggregate_functions"),
        floating_statistical_aggregate: string_label(&labels, "floating_statistical_aggregate"),
        statistical_aggregate_execution_mode: string_label(&labels, "statistical_aggregate_execution_mode"),
        statistical_aggregate_tolerance_contract: string_label(&labels, "statistical_aggregate_tolerance_contract"),
        statistical_aggregate_functions: string_label(&labels, "statistical_aggregate_functions"),
        statistical_aggregate_ddof: string_label(&labels, "statistical_aggregate_ddof"),
        aggregate_filter_count: usize_label(&labels, "aggregate_filter_count"),
        filtered_input_materialization_strategy: string_label(&labels, "filtered_input_materialization_strategy"),
        filtered_input_materialization_estimated_bytes: usize_label(
            &labels,
            "filtered_input_materialization_estimated_bytes",
        ),
        filtered_input_materialization_budget_bytes: usize_label(
            &labels,
            "filtered_input_materialization_budget_bytes",
        ),
        filtered_input_materialization_threshold_bytes: usize_label(
            &labels,
            "filtered_input_materialization_threshold_bytes",
        ),
        filtered_global_rows_per_chunk: usize_label(&labels, "filtered_global_rows_per_chunk"),
        filtered_global_inputs_per_segment: usize_label(&labels, "filtered_global_inputs_per_segment"),
        filtered_global_segment_target_temporary_bytes: usize_label(
            &labels,
            "filtered_global_segment_target_temporary_bytes",
        ),
        filtered_global_actual_chunk_count: usize_label(&labels, "filtered_global_actual_chunk_count"),
        filtered_global_actual_segment_count: usize_label(&labels, "filtered_global_actual_segment_count"),
        filtered_global_max_segment_rows: usize_label(&labels, "filtered_global_max_segment_rows"),
        filtered_global_observed_temporary_peak_bytes: usize_label(
            &labels,
            "filtered_global_observed_temporary_peak_bytes",
        ),
        filtered_global_observed_retained_peak_bytes: usize_label(
            &labels,
            "filtered_global_observed_retained_peak_bytes",
        ),
        filtered_global_state_rows_merged: usize_label(&labels, "filtered_global_state_rows_merged"),
        filtered_grouped_rows_per_chunk: usize_label(&labels, "filtered_grouped_rows_per_chunk"),
        filtered_grouped_inputs_per_segment: usize_label(&labels, "filtered_grouped_inputs_per_segment"),
        filtered_grouped_segment_target_temporary_bytes: usize_label(
            &labels,
            "filtered_grouped_segment_target_temporary_bytes",
        ),
        filtered_grouped_actual_chunk_count: usize_label(&labels, "filtered_grouped_actual_chunk_count"),
        filtered_grouped_actual_segment_count: usize_label(&labels, "filtered_grouped_actual_segment_count"),
        filtered_grouped_max_segment_rows: usize_label(&labels, "filtered_grouped_max_segment_rows"),
        filtered_grouped_observed_temporary_peak_bytes: usize_label(
            &labels,
            "filtered_grouped_observed_temporary_peak_bytes",
        ),
        filtered_grouped_observed_retained_peak_bytes: usize_label(
            &labels,
            "filtered_grouped_observed_retained_peak_bytes",
        ),
        filtered_grouped_state_rows_merged: usize_label(&labels, "filtered_grouped_state_rows_merged"),
        filtered_grouped_high_cardinality_admission: string_label(
            &labels,
            "filtered_grouped_high_cardinality_admission",
        ),
        distinct_aggregate_count: usize_label(&labels, "distinct_aggregate_count"),
        input_rows: usize_label(&labels, "input_rows"),
        output_group_count: usize_label(&labels, "output_group_count"),
        reduction_ratio_ppm: usize_label(&labels, "reduction_ratio_ppm"),
        chunk_row_count: usize_label(&labels, "chunk_row_count"),
        chunk_count: usize_label(&labels, "chunk_count"),
        partial_aggregate_count: usize_label(&labels, "partial_aggregate_count"),
        source_adjacent_partial: string_label(&labels, "source_adjacent_partial"),
        source_chunk_row_group_count: string_label(&labels, "source_chunk_row_group_count"),
        source_chunk_reads_all_row_groups: string_label(&labels, "source_chunk_reads_all_row_groups"),
        source_chunk_known_row_count: string_label(&labels, "source_chunk_known_row_count"),
        source_chunk_estimated_source_read_bytes: string_label(&labels, "source_chunk_estimated_source_read_bytes"),
        source_chunk_size_estimate_based: string_label(&labels, "source_chunk_size_estimate_based"),
        source_adjacent_partition_strategy: string_label(&labels, "source_adjacent_partition_strategy"),
        source_adjacent_partition_count: string_label(&labels, "source_adjacent_partition_count"),
        source_adjacent_non_empty_partition_count: string_label(&labels, "source_adjacent_non_empty_partition_count"),
        source_adjacent_partition_key_count: string_label(&labels, "source_adjacent_partition_key_count"),
        source_adjacent_max_partition_fan_in: string_label(&labels, "source_adjacent_max_partition_fan_in"),
        partitioned_merge_partition_count: string_label(&labels, "partitioned_merge_partition_count"),
        partitioned_merge_non_empty_partition_count: string_label(
            &labels,
            "partitioned_merge_non_empty_partition_count",
        ),
        partitioned_merge_max_partition_fan_in: string_label(&labels, "partitioned_merge_max_partition_fan_in"),
        selection_elapsed_ns: usize_label(&labels, "selection_elapsed_ns"),
        encode_elapsed_ns: usize_label(&labels, "encode_elapsed_ns"),
        groupby_elapsed_ns: usize_label(&labels, "groupby_elapsed_ns"),
        decode_elapsed_ns: usize_label(&labels, "decode_elapsed_ns"),
        strategy_operator_elapsed_ns: usize_label(&labels, "strategy_operator_elapsed_ns"),
        aggregate_structural_key: string_label(&labels, "aggregate_structural_key"),
        prior_cardinality_evidence_scope: string_label(&labels, "prior_cardinality_evidence_scope"),
        aggregate_strategy_options_version: string_label(&labels, "aggregate_strategy_options_version"),
        aggregate_encoded_key_partial_low_cardinality_threshold_ppm: usize_label(
            &labels,
            "aggregate_encoded_key_partial_low_cardinality_threshold_ppm",
        ),
        aggregate_encoded_key_partial_min_input_rows: usize_label(
            &labels,
            "aggregate_encoded_key_partial_min_input_rows",
        ),
        aggregate_exact_grouped_key_cardinality_probe_max_key_columns: usize_label(
            &labels,
            "aggregate_exact_grouped_key_cardinality_probe_max_key_columns",
        ),
        aggregate_exact_single_string_avg_cardinality_probe_min_input_rows: usize_label(
            &labels,
            "aggregate_exact_single_string_avg_cardinality_probe_min_input_rows",
        ),
        aggregate_exact_single_key_cardinality_probe_min_input_rows: usize_label(
            &labels,
            "aggregate_exact_single_key_cardinality_probe_min_input_rows",
        ),
        aggregate_exact_grouped_key_cardinality_probe_min_input_rows: usize_label(
            &labels,
            "aggregate_exact_grouped_key_cardinality_probe_min_input_rows",
        ),
        aggregate_exact_single_string_key_prepared_max_groups: usize_label(
            &labels,
            "aggregate_exact_single_string_key_prepared_max_groups",
        ),
        aggregate_exact_three_key_cardinality_probe_max_input_rows: usize_label(
            &labels,
            "aggregate_exact_three_key_cardinality_probe_max_input_rows",
        ),
        aggregate_exact_five_key_cardinality_probe_max_input_rows: usize_label(
            &labels,
            "aggregate_exact_five_key_cardinality_probe_max_input_rows",
        ),
        aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm: usize_label(
            &labels,
            "aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm",
        ),
        aggregate_source_adjacent_prepared_sort_groupby_max_row_groups: usize_label(
            &labels,
            "aggregate_source_adjacent_prepared_sort_groupby_max_row_groups",
        ),
        aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes: usize_label(
            &labels,
            "aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes",
        ),
        aggregate_source_adjacent_prepared_sort_groupby_max_fan_in: usize_label(
            &labels,
            "aggregate_source_adjacent_prepared_sort_groupby_max_fan_in",
        ),
        encoded_key_partial_cost_model: string_label(&labels, "encoded_key_partial_cost_model"),
        encoded_key_partial_cost_model_status: string_label(&labels, "encoded_key_partial_cost_model_status"),
        encoded_key_partial_selection_reason: string_label(&labels, "encoded_key_partial_selection_reason"),
        encoded_key_partial_reduction_ratio_threshold_ppm: usize_label(
            &labels,
            "encoded_key_partial_reduction_ratio_threshold_ppm",
        ),
        encoded_key_partial_prior_reduction_ratio_ppm: usize_label(
            &labels,
            "encoded_key_partial_prior_reduction_ratio_ppm",
        ),
        encoded_key_partial_prior_strategy: string_label(&labels, "encoded_key_partial_prior_strategy"),
        exact_cardinality_probe_status: string_label(&labels, "exact_cardinality_probe_status"),
        exact_cardinality_probe_key: string_label(&labels, "exact_cardinality_probe_key"),
        exact_cardinality_probe_input_rows: usize_label(&labels, "exact_cardinality_probe_input_rows"),
        exact_cardinality_probe_non_null_distinct_count: string_label(
            &labels,
            "exact_cardinality_probe_non_null_distinct_count",
        ),
        exact_cardinality_probe_null_count: string_label(&labels, "exact_cardinality_probe_null_count"),
        exact_cardinality_probe_num_groups: string_label(&labels, "exact_cardinality_probe_num_groups"),
        exact_cardinality_probe_reduction_ratio_ppm: string_label(
            &labels,
            "exact_cardinality_probe_reduction_ratio_ppm",
        ),
        exact_cardinality_probe_elapsed_ns: usize_label(&labels, "exact_cardinality_probe_elapsed_ns"),
        prepared_sort_groupby_rejected_reason: string_label(&labels, "prepared_sort_groupby_rejected_reason"),
        prepared_sort_groupby_contract_status: string_label(&labels, "prepared_sort_groupby_contract_status"),
        prepared_sort_groupby_required_contract: string_label(&labels, "prepared_sort_groupby_required_contract"),
        prepared_sort_groupby_cost_model: string_label(&labels, "prepared_sort_groupby_cost_model"),
        prepared_sort_groupby_cost_model_status: string_label(&labels, "prepared_sort_groupby_cost_model_status"),
        prepared_sort_groupby_selection_reason: string_label(&labels, "prepared_sort_groupby_selection_reason"),
        prepared_sort_groupby_num_groups: string_label(&labels, "prepared_sort_groupby_num_groups"),
        prepared_sort_groupby_grouped_key_output_bytes: string_label(
            &labels,
            "prepared_sort_groupby_grouped_key_output_bytes",
        ),
        prepared_sort_groupby_max_group_size: string_label(&labels, "prepared_sort_groupby_max_group_size"),
        rejected_candidate_strategies: string_label(&labels, "rejected_candidate_strategies"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_strategy_metric_row_projects_strategy_execution_options() {
        let labels = BTreeMap::from([
            ("aggregate_strategy_options_version", "1"),
            ("aggregate_encoded_key_partial_low_cardinality_threshold_ppm", "175000"),
            ("aggregate_encoded_key_partial_min_input_rows", "123"),
            ("aggregate_exact_grouped_key_cardinality_probe_max_key_columns", "4"),
            ("aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm", "200000"),
            ("aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes", "4096"),
        ]);

        let row = labels_to_aggregate_strategy_row(labels);

        assert_eq!(row.aggregate_strategy_options_version, "1");
        assert_eq!(row.aggregate_encoded_key_partial_low_cardinality_threshold_ppm, 175_000);
        assert_eq!(row.aggregate_encoded_key_partial_min_input_rows, 123);
        assert_eq!(row.aggregate_exact_grouped_key_cardinality_probe_max_key_columns, 4);
        assert_eq!(row.aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm, 200_000);
        assert_eq!(row.aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes, 4096);
    }
}
