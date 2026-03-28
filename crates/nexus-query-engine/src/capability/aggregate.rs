//! Aggregate-strategy capability analysis.
//!
//! This module classifies grouped aggregate stages into a selected execution
//! strategy plus locally rejected candidate strategies. It does not enable an
//! optimized path by itself; executor admission must still have a concrete
//! reservation and execution contract before selecting anything except the
//! default libcudf groupby strategy.

use std::collections::BTreeMap;

use arrow::datatypes::{DataType, Schema};

use crate::{
    datatypes::is_arrow_fixed_width,
    error::{QueryError, QueryErrorCode, Result as QueryResult},
    expr::{AggregateFunction, AggregateSpec, Expr},
    plan::{NodeId, PlanNode, QueryPlan},
};

/// Strategy capability facts attached to one aggregate node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateStrategyCapability {
    /// Native aggregate node id.
    pub node_id: NodeId,
    /// Aggregate execution stage represented by this capability proof.
    pub stage: AggregateExecutionStage,
    /// Strategy decision and local candidate rejection facts.
    pub decision: AggregateStrategyCapabilityDecision,
}

/// Aggregate execution stage being evaluated for strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateExecutionStage {
    /// One-shot grouped aggregate over a resident input frame.
    Single,
    /// Chunk-local partial grouped aggregate before final merge.
    Partial,
    /// Final grouped aggregate merge over partial aggregate rows.
    FinalMerge,
}

/// Native aggregate state shape relevant to strict bounded admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::capability) enum AggregateStateContract {
    /// No distinct or group-only dedup state is required.
    Plain,
    /// SQL setop-style dedup: grouped keys with no aggregate expressions.
    DedupGroupOnly,
    /// Exact count-distinct state represented as `COUNT(DISTINCT col)` or
    /// native `NUnique(col)`.
    ExactDistinctCount,
}

/// Aggregate state forms that need a more specific contract before admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::capability) enum UnsupportedAggregateStateContract {
    GroupingSets,
    UnsupportedDistinctFunction,
}

impl UnsupportedAggregateStateContract {
    pub(in crate::capability) const fn as_str(self) -> &'static str {
        match self {
            Self::GroupingSets => "grouping_sets",
            Self::UnsupportedDistinctFunction => "unsupported_distinct_function",
        }
    }
}

pub(in crate::capability) fn aggregate_state_contract(
    spec: &AggregateSpec,
) -> Result<AggregateStateContract, UnsupportedAggregateStateContract> {
    if spec.grouping_sets.is_some() {
        return Err(UnsupportedAggregateStateContract::GroupingSets);
    }
    if spec.aggr_exprs.is_empty() && !spec.group_exprs.is_empty() {
        return Ok(AggregateStateContract::DedupGroupOnly);
    }

    let mut has_exact_distinct_count = false;
    for expr in &spec.aggr_exprs {
        if aggregate_expr_is_exact_distinct_count(expr) {
            has_exact_distinct_count = true;
        } else if expr.distinct {
            return Err(UnsupportedAggregateStateContract::UnsupportedDistinctFunction);
        }
    }

    if has_exact_distinct_count {
        Ok(AggregateStateContract::ExactDistinctCount)
    } else {
        Ok(AggregateStateContract::Plain)
    }
}

fn aggregate_expr_is_exact_distinct_count(expr: &crate::expr::AggregateExpr) -> bool {
    expr.function == AggregateFunction::NUnique || expr.distinct && expr.function == AggregateFunction::Count
}

impl AggregateExecutionStage {
    /// Stable report label for this aggregate stage.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Partial => "partial",
            Self::FinalMerge => "final_merge",
        }
    }
}

/// Native grouped aggregate execution strategy label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateExecutionStrategy {
    /// Existing default path using libcudf groupby directly on group keys.
    LibcudfGroupby,
    /// Candidate path that dictionary/factor encodes eligible group keys.
    EncodedKeyPartial,
    /// Candidate path using cuDF's prepared sort-backed groupby state.
    PreparedSortGroupby,
    /// Candidate path that partitions partial outputs before grouped merge.
    PartitionedMerge,
}

impl AggregateExecutionStrategy {
    /// Stable report label for this aggregate strategy.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LibcudfGroupby => "libcudf_groupby",
            Self::EncodedKeyPartial => "encoded_key_partial",
            Self::PreparedSortGroupby => "prepared_sort_groupby",
            Self::PartitionedMerge => "partitioned_merge",
        }
    }
}

/// Runtime threshold policy for grouped aggregate strategy selection.
///
/// These thresholds choose between the default libcudf groupby path, encoded
/// key partial aggregation, prepared sort-backed groupby, and exact
/// cardinality probes. This is an expert Rust surface only; keep it out of
/// server/session/user-facing configuration until the tuning model has a
/// reviewed production contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct AggregateStrategyExecutionOptions {
    pub(crate) encoded_key_partial_low_cardinality_threshold_ppm: u64,
    pub(crate) encoded_key_partial_min_input_rows: u64,
    pub(crate) exact_grouped_key_cardinality_probe_max_key_columns: usize,
    pub(crate) exact_single_string_avg_cardinality_probe_min_input_rows: u64,
    pub(crate) exact_single_key_cardinality_probe_min_input_rows: u64,
    pub(crate) exact_grouped_key_cardinality_probe_min_input_rows: u64,
    pub(crate) exact_single_string_key_prepared_max_groups: u64,
    pub(crate) exact_three_key_cardinality_probe_max_input_rows: u64,
    pub(crate) exact_five_key_cardinality_probe_max_input_rows: u64,
    pub(crate) prepared_sort_groupby_low_cardinality_threshold_ppm: u64,
    pub(crate) source_adjacent_prepared_sort_groupby_max_row_groups: u64,
    pub(crate) source_adjacent_prepared_sort_groupby_max_source_read_bytes: u64,
    pub(crate) source_adjacent_prepared_sort_groupby_max_fan_in: u64,
}

impl AggregateStrategyExecutionOptions {
    const MAX_RATIO_PPM: u64 = 1_000_000;

    pub const DEFAULT_ENCODED_KEY_PARTIAL_LOW_CARDINALITY_THRESHOLD_PPM: u64 = 100_000;
    pub const DEFAULT_ENCODED_KEY_PARTIAL_MIN_INPUT_ROWS: u64 = 100_000;
    pub const DEFAULT_EXACT_GROUPED_KEY_CARDINALITY_PROBE_MAX_KEY_COLUMNS: usize = 5;
    pub const DEFAULT_EXACT_SINGLE_STRING_AVG_CARDINALITY_PROBE_MIN_INPUT_ROWS: u64 = 100_000;
    pub const DEFAULT_EXACT_SINGLE_KEY_CARDINALITY_PROBE_MIN_INPUT_ROWS: u64 = 500_000;
    pub const DEFAULT_EXACT_GROUPED_KEY_CARDINALITY_PROBE_MIN_INPUT_ROWS: u64 = 500_000;
    pub const DEFAULT_EXACT_SINGLE_STRING_KEY_PREPARED_MAX_GROUPS: u64 = 4_096;
    pub const DEFAULT_EXACT_THREE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS: u64 = 10_000_000;
    pub const DEFAULT_EXACT_FIVE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS: u64 = 7_000_000;
    pub const DEFAULT_PREPARED_SORT_GROUPBY_LOW_CARDINALITY_THRESHOLD_PPM: u64 = 100_000;
    pub const DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS: u64 = 64;
    pub const DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES: u64 = 384 * 1024 * 1024;
    pub const DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_FAN_IN: u64 = 64;

    pub fn encoded_key_partial_low_cardinality_threshold_ppm(&self) -> u64 {
        self.encoded_key_partial_low_cardinality_threshold_ppm
    }

    pub fn encoded_key_partial_min_input_rows(&self) -> u64 {
        self.encoded_key_partial_min_input_rows
    }

    pub fn exact_grouped_key_cardinality_probe_max_key_columns(&self) -> usize {
        self.exact_grouped_key_cardinality_probe_max_key_columns
    }

    pub fn exact_single_string_avg_cardinality_probe_min_input_rows(&self) -> u64 {
        self.exact_single_string_avg_cardinality_probe_min_input_rows
    }

    pub fn exact_single_key_cardinality_probe_min_input_rows(&self) -> u64 {
        self.exact_single_key_cardinality_probe_min_input_rows
    }

    pub fn exact_grouped_key_cardinality_probe_min_input_rows(&self) -> u64 {
        self.exact_grouped_key_cardinality_probe_min_input_rows
    }

    pub fn exact_single_string_key_prepared_max_groups(&self) -> u64 {
        self.exact_single_string_key_prepared_max_groups
    }

    pub fn exact_three_key_cardinality_probe_max_input_rows(&self) -> u64 {
        self.exact_three_key_cardinality_probe_max_input_rows
    }

    pub fn exact_five_key_cardinality_probe_max_input_rows(&self) -> u64 {
        self.exact_five_key_cardinality_probe_max_input_rows
    }

    pub fn prepared_sort_groupby_low_cardinality_threshold_ppm(&self) -> u64 {
        self.prepared_sort_groupby_low_cardinality_threshold_ppm
    }

    pub fn source_adjacent_prepared_sort_groupby_max_row_groups(&self) -> u64 {
        self.source_adjacent_prepared_sort_groupby_max_row_groups
    }

    pub fn source_adjacent_prepared_sort_groupby_max_source_read_bytes(&self) -> u64 {
        self.source_adjacent_prepared_sort_groupby_max_source_read_bytes
    }

    pub fn source_adjacent_prepared_sort_groupby_max_fan_in(&self) -> u64 {
        self.source_adjacent_prepared_sort_groupby_max_fan_in
    }

    /// Validates aggregate strategy policy values.
    ///
    /// Zero-valued thresholds remain accepted because this is an expert Rust
    /// surface and callers may use zero to disable narrow strategy branches.
    /// The checks here reject ratios above one hundred percent and contradictory
    /// nonzero probe bounds so strategy facts remain interpretable.
    pub fn validate(&self) -> QueryResult<()> {
        validate_aggregate_strategy_execution_ppm(
            "encoded_key_partial_low_cardinality_threshold_ppm",
            self.encoded_key_partial_low_cardinality_threshold_ppm,
        )?;
        validate_aggregate_strategy_execution_ppm(
            "prepared_sort_groupby_low_cardinality_threshold_ppm",
            self.prepared_sort_groupby_low_cardinality_threshold_ppm,
        )?;
        validate_aggregate_strategy_execution_min_lte_max(
            "exact_grouped_key_cardinality_probe_min_input_rows",
            self.exact_grouped_key_cardinality_probe_min_input_rows,
            "exact_three_key_cardinality_probe_max_input_rows",
            self.exact_three_key_cardinality_probe_max_input_rows,
            "three_key_probe_max_input_rows_below_grouped_probe_min_input_rows",
        )?;
        validate_aggregate_strategy_execution_min_lte_max(
            "exact_grouped_key_cardinality_probe_min_input_rows",
            self.exact_grouped_key_cardinality_probe_min_input_rows,
            "exact_five_key_cardinality_probe_max_input_rows",
            self.exact_five_key_cardinality_probe_max_input_rows,
            "five_key_probe_max_input_rows_below_grouped_probe_min_input_rows",
        )?;
        validate_aggregate_strategy_execution_min_lte_max(
            "exact_single_string_avg_cardinality_probe_min_input_rows",
            self.exact_single_string_avg_cardinality_probe_min_input_rows,
            "exact_single_key_cardinality_probe_min_input_rows",
            self.exact_single_key_cardinality_probe_min_input_rows,
            "single_string_avg_probe_min_input_rows_above_general_single_key_min_input_rows",
        )?;
        Ok(())
    }

    pub fn with_encoded_key_partial_low_cardinality_threshold_ppm(mut self, threshold_ppm: u64) -> Self {
        self.encoded_key_partial_low_cardinality_threshold_ppm = threshold_ppm;
        self
    }

    pub fn with_encoded_key_partial_min_input_rows(mut self, rows: u64) -> Self {
        self.encoded_key_partial_min_input_rows = rows;
        self
    }

    pub fn with_exact_grouped_key_cardinality_probe_max_key_columns(mut self, columns: usize) -> Self {
        self.exact_grouped_key_cardinality_probe_max_key_columns = columns;
        self
    }

    pub fn with_exact_single_string_avg_cardinality_probe_min_input_rows(mut self, rows: u64) -> Self {
        self.exact_single_string_avg_cardinality_probe_min_input_rows = rows;
        self
    }

    pub fn with_exact_single_key_cardinality_probe_min_input_rows(mut self, rows: u64) -> Self {
        self.exact_single_key_cardinality_probe_min_input_rows = rows;
        self
    }

    pub fn with_exact_grouped_key_cardinality_probe_min_input_rows(mut self, rows: u64) -> Self {
        self.exact_grouped_key_cardinality_probe_min_input_rows = rows;
        self
    }

    pub fn with_exact_single_string_key_prepared_max_groups(mut self, groups: u64) -> Self {
        self.exact_single_string_key_prepared_max_groups = groups;
        self
    }

    pub fn with_exact_three_key_cardinality_probe_max_input_rows(mut self, rows: u64) -> Self {
        self.exact_three_key_cardinality_probe_max_input_rows = rows;
        self
    }

    pub fn with_exact_five_key_cardinality_probe_max_input_rows(mut self, rows: u64) -> Self {
        self.exact_five_key_cardinality_probe_max_input_rows = rows;
        self
    }

    pub fn with_prepared_sort_groupby_low_cardinality_threshold_ppm(mut self, threshold_ppm: u64) -> Self {
        self.prepared_sort_groupby_low_cardinality_threshold_ppm = threshold_ppm;
        self
    }

    pub fn with_source_adjacent_prepared_sort_groupby_max_row_groups(mut self, row_groups: u64) -> Self {
        self.source_adjacent_prepared_sort_groupby_max_row_groups = row_groups;
        self
    }

    pub fn with_source_adjacent_prepared_sort_groupby_max_source_read_bytes(mut self, bytes: u64) -> Self {
        self.source_adjacent_prepared_sort_groupby_max_source_read_bytes = bytes;
        self
    }

    pub fn with_source_adjacent_prepared_sort_groupby_max_fan_in(mut self, fan_in: u64) -> Self {
        self.source_adjacent_prepared_sort_groupby_max_fan_in = fan_in;
        self
    }
}

fn validate_aggregate_strategy_execution_ppm(field: &'static str, value: u64) -> QueryResult<()> {
    if value > AggregateStrategyExecutionOptions::MAX_RATIO_PPM {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.aggregate_strategy_execution_options.validate",
            "aggregate strategy execution ratio threshold exceeds one hundred percent",
        )
        .with_fact("field", field)
        .with_fact("value", value)
        .with_fact("max_value", AggregateStrategyExecutionOptions::MAX_RATIO_PPM)
        .with_fact("reason", "ratio_ppm_above_one_hundred_percent"));
    }
    Ok(())
}

fn validate_aggregate_strategy_execution_min_lte_max(
    min_field: &'static str,
    min_value: u64,
    max_field: &'static str,
    max_value: u64,
    reason: &'static str,
) -> QueryResult<()> {
    if min_value != 0 && max_value != 0 && min_value > max_value {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.aggregate_strategy_execution_options.validate",
            "aggregate strategy execution threshold ordering is inconsistent",
        )
        .with_fact("field", min_field)
        .with_fact("value", min_value)
        .with_fact("bound_field", max_field)
        .with_fact("bound_value", max_value)
        .with_fact("reason", reason));
    }
    Ok(())
}

impl Default for AggregateStrategyExecutionOptions {
    fn default() -> Self {
        Self {
            encoded_key_partial_low_cardinality_threshold_ppm:
                Self::DEFAULT_ENCODED_KEY_PARTIAL_LOW_CARDINALITY_THRESHOLD_PPM,
            encoded_key_partial_min_input_rows: Self::DEFAULT_ENCODED_KEY_PARTIAL_MIN_INPUT_ROWS,
            exact_grouped_key_cardinality_probe_max_key_columns:
                Self::DEFAULT_EXACT_GROUPED_KEY_CARDINALITY_PROBE_MAX_KEY_COLUMNS,
            exact_single_string_avg_cardinality_probe_min_input_rows:
                Self::DEFAULT_EXACT_SINGLE_STRING_AVG_CARDINALITY_PROBE_MIN_INPUT_ROWS,
            exact_single_key_cardinality_probe_min_input_rows:
                Self::DEFAULT_EXACT_SINGLE_KEY_CARDINALITY_PROBE_MIN_INPUT_ROWS,
            exact_grouped_key_cardinality_probe_min_input_rows:
                Self::DEFAULT_EXACT_GROUPED_KEY_CARDINALITY_PROBE_MIN_INPUT_ROWS,
            exact_single_string_key_prepared_max_groups: Self::DEFAULT_EXACT_SINGLE_STRING_KEY_PREPARED_MAX_GROUPS,
            exact_three_key_cardinality_probe_max_input_rows:
                Self::DEFAULT_EXACT_THREE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS,
            exact_five_key_cardinality_probe_max_input_rows:
                Self::DEFAULT_EXACT_FIVE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS,
            prepared_sort_groupby_low_cardinality_threshold_ppm:
                Self::DEFAULT_PREPARED_SORT_GROUPBY_LOW_CARDINALITY_THRESHOLD_PPM,
            source_adjacent_prepared_sort_groupby_max_row_groups:
                Self::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS,
            source_adjacent_prepared_sort_groupby_max_source_read_bytes:
                Self::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES,
            source_adjacent_prepared_sort_groupby_max_fan_in:
                Self::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_FAN_IN,
        }
    }
}

/// Local rejection for an aggregate strategy candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateStrategyCandidateRejection {
    /// Candidate strategy that was rejected.
    pub strategy: AggregateExecutionStrategy,
    /// Stable local rejection code.
    pub reason: &'static str,
}

/// Capability decision for one aggregate strategy selection point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateStrategyCapabilityDecision {
    /// Strategy that is available for execution now.
    pub selected_strategy: AggregateExecutionStrategy,
    /// Candidate optimized strategies that were not admitted.
    pub rejected_candidates: Vec<AggregateStrategyCandidateRejection>,
    /// Stable shape and admission facts for reports.
    pub facts: BTreeMap<&'static str, String>,
}

impl AggregateStrategyCapabilityDecision {
    /// Formats rejected candidates as `strategy:reason` segments for TSV facts.
    pub fn rejected_candidate_summary(&self) -> String {
        if self.rejected_candidates.is_empty() {
            return "none".to_string();
        }
        self.rejected_candidates
            .iter()
            .map(|rejection| format!("{}:{}", rejection.strategy.as_str(), rejection.reason))
            .collect::<Vec<_>>()
            .join(";")
    }
}

/// Returns the aggregate-strategy capability decision for an aggregate stage.
pub fn aggregate_strategy_capability_decision(
    spec: &AggregateSpec,
    input_schema: &Schema,
    stage: AggregateExecutionStage,
) -> AggregateStrategyCapabilityDecision {
    let encoded_key_partial_reason = encoded_key_partial_rejected_reason(spec, input_schema, stage);
    let prepared_sort_groupby_reason = prepared_sort_groupby_rejected_reason(spec, input_schema, stage);
    let partitioned_merge_rejected_reason = partitioned_merge_rejected_reason(spec, stage);
    let selected_strategy = if partitioned_merge_rejected_reason.is_none() {
        AggregateExecutionStrategy::PartitionedMerge
    } else if encoded_key_partial_reason.is_none() {
        AggregateExecutionStrategy::EncodedKeyPartial
    } else {
        AggregateExecutionStrategy::LibcudfGroupby
    };
    let mut rejected_candidates = Vec::new();
    if let Some(reason) = encoded_key_partial_reason {
        rejected_candidates.push(AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::EncodedKeyPartial,
            reason,
        });
    }
    rejected_candidates.push(AggregateStrategyCandidateRejection {
        strategy: AggregateExecutionStrategy::PreparedSortGroupby,
        reason: prepared_sort_groupby_reason.unwrap_or("prepared_sort_groupby_requires_runtime_cardinality_evidence"),
    });
    if let Some(reason) = partitioned_merge_rejected_reason {
        rejected_candidates.push(AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::PartitionedMerge,
            reason,
        });
    }
    let mut facts = BTreeMap::from([
        ("selected_strategy", selected_strategy.as_str().to_string()),
        ("execution_stage", stage.as_str().to_string()),
        ("group_key_count", spec.group_exprs.len().to_string()),
        ("group_key_exprs", aggregate_group_key_exprs(spec)),
        ("group_key_dtypes", aggregate_group_key_dtypes(spec, input_schema)),
        ("aggregate_functions", aggregate_function_list(spec)),
        ("aggregate_filter_count", aggregate_filter_count(spec).to_string()),
        ("distinct_aggregate_count", distinct_aggregate_count(spec).to_string()),
        ("grouping_mode", aggregate_grouping_mode(spec).to_string()),
        ("grouping_sets_present", spec.grouping_sets.is_some().to_string()),
        ("plain_column_group_keys", spec.group_exprs.iter().all(is_plain_column_expr).to_string()),
        ("string_group_key_count", string_group_key_count(spec, input_schema).to_string()),
        ("fixed_width_group_key_count", fixed_width_group_key_count(spec, input_schema).to_string()),
        ("unsupported_group_key_dtypes", unsupported_group_key_dtypes(spec, input_schema)),
        ("hidden_host_preparation", "false".to_string()),
        ("encoded_key_partial_rejected_reason", encoded_key_partial_reason.unwrap_or("none").to_string()),
        (
            "encoded_key_partial_contract_status",
            if encoded_key_partial_reason.is_none() { "available" } else { "blocked_by_shape_or_stage" }.to_string(),
        ),
        ("encoded_key_partial_required_contract", "encoded_key_partial_execution_contract".to_string()),
        (
            "prepared_sort_groupby_rejected_reason",
            prepared_sort_groupby_reason
                .unwrap_or("prepared_sort_groupby_requires_runtime_cardinality_evidence")
                .to_string(),
        ),
        (
            "prepared_sort_groupby_contract_status",
            if prepared_sort_groupby_reason.is_none() { "available" } else { "blocked_by_shape_or_stage" }.to_string(),
        ),
        ("prepared_sort_groupby_required_contract", "prepared_sort_groupby_execution_contract".to_string()),
        ("prepared_sort_groupby_cost_model", "exact_grouped_key_facts_required".to_string()),
        (
            "prepared_sort_groupby_cost_model_status",
            if prepared_sort_groupby_reason.is_none() { "pending_runtime_evidence" } else { "not_applicable" }
                .to_string(),
        ),
        (
            "prepared_sort_groupby_selection_reason",
            if prepared_sort_groupby_reason.is_none() {
                "runtime_cardinality_evidence_required"
            } else {
                "capability_rejected"
            }
            .to_string(),
        ),
        ("prepared_sort_groupby_num_groups", "unknown".to_string()),
        ("prepared_sort_groupby_grouped_key_output_bytes", "unknown".to_string()),
        ("prepared_sort_groupby_max_group_size", "unknown".to_string()),
        ("partitioned_merge_rejected_reason", partitioned_merge_rejected_reason.unwrap_or("none").to_string()),
        (
            "partitioned_merge_contract_status",
            if partitioned_merge_rejected_reason.is_none() { "available" } else { "blocked_by_shape_or_stage" }
                .to_string(),
        ),
        ("partitioned_merge_required_contract", "partitioned_merge_execution_contract".to_string()),
        ("default_strategy_available", "true".to_string()),
    ]);
    facts.insert(
        "rejected_candidate_strategies",
        rejected_candidates
            .iter()
            .map(|rejection| format!("{}:{}", rejection.strategy.as_str(), rejection.reason))
            .collect::<Vec<_>>()
            .join(";"),
    );
    AggregateStrategyCapabilityDecision { selected_strategy, rejected_candidates, facts }
}

/// Collects aggregate strategy capability facts for grouped aggregate nodes.
pub(crate) fn analyze_aggregate_strategies(plan: &QueryPlan) -> Vec<AggregateStrategyCapability> {
    plan.nodes()
        .iter()
        .filter_map(|node| {
            let PlanNode::Aggregate(aggregate) = &node.kind else {
                return None;
            };
            if aggregate.spec.group_exprs.is_empty() {
                return None;
            }
            let input_schema = plan.node(aggregate.input)?.output_schema();
            let stage = AggregateExecutionStage::Partial;
            Some(AggregateStrategyCapability {
                node_id: node.id,
                stage,
                decision: aggregate_strategy_capability_decision(&aggregate.spec, input_schema, stage),
            })
        })
        .collect()
}

fn encoded_key_partial_rejected_reason(
    spec: &AggregateSpec,
    input_schema: &Schema,
    stage: AggregateExecutionStage,
) -> Option<&'static str> {
    if stage != AggregateExecutionStage::Partial {
        return Some("encoded_key_partial_requires_partial_stage");
    }
    if spec.grouping_sets.is_some() {
        return Some("encoded_key_partial_grouping_sets_unsupported");
    }
    if spec.group_exprs.is_empty() {
        return Some("encoded_key_partial_requires_grouped_aggregate");
    }
    if !spec.group_exprs.iter().all(is_plain_column_expr) {
        return Some("encoded_key_partial_requires_plain_column_group_keys");
    }
    if aggregate_filter_count(spec) > 0 {
        return Some("encoded_key_partial_aggregate_filter_unsupported");
    }
    if distinct_aggregate_count(spec) > 0 {
        return Some("encoded_key_partial_distinct_aggregate_unsupported");
    }
    let mut saw_string_key = false;
    for expr in &spec.group_exprs {
        let Some(data_type) = plain_column_group_key_dtype(expr, input_schema) else {
            return Some("encoded_key_partial_group_key_column_missing");
        };
        if aggregate_encoded_key_string_dtype(data_type) {
            saw_string_key = true;
        } else if !aggregate_encoded_key_companion_dtype(data_type) {
            return Some("encoded_key_partial_group_key_dtype_unsupported");
        }
    }
    if !saw_string_key {
        return Some("encoded_key_partial_requires_string_group_key");
    }
    None
}

fn prepared_sort_groupby_rejected_reason(
    spec: &AggregateSpec,
    input_schema: &Schema,
    stage: AggregateExecutionStage,
) -> Option<&'static str> {
    if !matches!(stage, AggregateExecutionStage::Partial | AggregateExecutionStage::Single) {
        return Some("prepared_sort_groupby_requires_partial_or_single_stage");
    }
    if spec.grouping_sets.is_some() {
        return Some("prepared_sort_groupby_grouping_sets_unsupported");
    }
    if spec.group_exprs.is_empty() {
        return Some("prepared_sort_groupby_requires_grouped_aggregate");
    }
    if spec.aggr_exprs.is_empty() {
        return Some("prepared_sort_groupby_requires_aggregate_expressions");
    }
    if !spec.group_exprs.iter().all(is_plain_column_expr) {
        return Some("prepared_sort_groupby_requires_plain_column_group_keys");
    }
    if aggregate_filter_count(spec) > 0 {
        return Some("prepared_sort_groupby_aggregate_filter_unsupported");
    }
    if distinct_aggregate_count(spec) > 0 {
        return Some("prepared_sort_groupby_distinct_aggregate_unsupported");
    }
    if aggregate_spec_has_floating_statistical_aggregate(spec) {
        return Some("prepared_sort_groupby_floating_statistical_aggregate_unsupported");
    }
    if spec.group_exprs.iter().any(|expr| plain_column_group_key_dtype(expr, input_schema).is_none()) {
        return Some("prepared_sort_groupby_group_key_column_missing");
    }
    None
}

fn partitioned_merge_rejected_reason(spec: &AggregateSpec, stage: AggregateExecutionStage) -> Option<&'static str> {
    if spec.grouping_sets.is_some() {
        return Some("partitioned_merge_grouping_sets_unsupported");
    }
    if spec.group_exprs.is_empty() {
        return Some("partitioned_merge_requires_grouped_aggregate");
    }
    if stage != AggregateExecutionStage::FinalMerge {
        return Some("partitioned_merge_requires_final_merge_stage");
    }
    if aggregate_spec_has_floating_statistical_aggregate(spec) {
        return Some("partitioned_merge_floating_statistical_aggregate_unsupported");
    }
    if distinct_aggregate_count(spec) > 0 {
        return Some("partitioned_merge_distinct_aggregate_unsupported");
    }
    None
}

fn aggregate_spec_has_floating_statistical_aggregate(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.function.is_floating_statistical())
}

fn aggregate_group_key_exprs(spec: &AggregateSpec) -> String {
    if spec.group_exprs.is_empty() {
        return "none".to_string();
    }
    spec.group_exprs.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

fn aggregate_group_key_dtypes(spec: &AggregateSpec, input_schema: &Schema) -> String {
    if spec.group_exprs.is_empty() {
        return "none".to_string();
    }
    spec.group_exprs.iter().map(|expr| aggregate_group_key_dtype(expr, input_schema)).collect::<Vec<_>>().join(",")
}

fn aggregate_group_key_dtype(expr: &Expr, input_schema: &Schema) -> String {
    match expr {
        Expr::Column(column) => input_schema
            .field_with_name(column.name())
            .map(|field| format!("{:?}", field.data_type()))
            .unwrap_or_else(|_| "missing_column".to_string()),
        Expr::Alias { expr, .. } => aggregate_group_key_dtype(expr, input_schema),
        _ => "non_column_expr".to_string(),
    }
}

fn aggregate_grouping_mode(spec: &AggregateSpec) -> &'static str {
    if spec.grouping_sets.is_some() {
        "grouping_sets"
    } else if spec.group_exprs.is_empty() {
        "scalar"
    } else {
        "grouped"
    }
}

fn string_group_key_count(spec: &AggregateSpec, input_schema: &Schema) -> usize {
    spec.group_exprs
        .iter()
        .filter(|expr| plain_column_group_key_dtype(expr, input_schema).is_some_and(aggregate_encoded_key_string_dtype))
        .count()
}

fn fixed_width_group_key_count(spec: &AggregateSpec, input_schema: &Schema) -> usize {
    spec.group_exprs
        .iter()
        .filter(|expr| {
            plain_column_group_key_dtype(expr, input_schema).is_some_and(aggregate_encoded_key_companion_dtype)
        })
        .count()
}

fn unsupported_group_key_dtypes(spec: &AggregateSpec, input_schema: &Schema) -> String {
    let dtypes = spec
        .group_exprs
        .iter()
        .filter_map(|expr| {
            let dtype = plain_column_group_key_dtype(expr, input_schema)?;
            (!aggregate_encoded_key_string_dtype(dtype) && !aggregate_encoded_key_companion_dtype(dtype))
                .then(|| format!("{dtype:?}"))
        })
        .collect::<Vec<_>>();
    if dtypes.is_empty() { "none".to_string() } else { dtypes.join(",") }
}

fn plain_column_group_key_dtype<'a>(expr: &Expr, input_schema: &'a Schema) -> Option<&'a DataType> {
    match expr {
        Expr::Column(column) => input_schema.field_with_name(column.name()).ok().map(|field| field.data_type()),
        Expr::Alias { expr, .. } => plain_column_group_key_dtype(expr, input_schema),
        _ => None,
    }
}

fn is_plain_column_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) => true,
        Expr::Alias { expr, .. } => is_plain_column_expr(expr),
        _ => false,
    }
}

fn aggregate_function_list(spec: &AggregateSpec) -> String {
    if spec.aggr_exprs.is_empty() {
        return "none".to_string();
    }
    spec.aggr_exprs
        .iter()
        .map(|expr| {
            let mut label = aggregate_function_label(expr.function).to_string();
            if expr.distinct {
                label.push_str("_distinct");
            }
            label
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn aggregate_function_label(function: AggregateFunction) -> &'static str {
    function.as_str()
}

fn aggregate_filter_count(spec: &AggregateSpec) -> usize {
    spec.aggr_exprs.iter().filter(|expr| expr.filter.is_some()).count()
}

fn distinct_aggregate_count(spec: &AggregateSpec) -> usize {
    spec.aggr_exprs.iter().filter(|expr| aggregate_expr_is_exact_distinct_count(expr)).count()
}

fn aggregate_encoded_key_string_dtype(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn aggregate_encoded_key_companion_dtype(data_type: &DataType) -> bool {
    is_arrow_fixed_width(data_type) || matches!(data_type, DataType::Float16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_strategy_execution_options_default_validate() {
        AggregateStrategyExecutionOptions::default().validate().unwrap();
    }

    #[test]
    fn aggregate_strategy_execution_options_reject_ratio_above_one_hundred_percent() {
        let error = AggregateStrategyExecutionOptions::default()
            .with_prepared_sort_groupby_low_cardinality_threshold_ppm(1_000_001)
            .validate()
            .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("prepared_sort_groupby_low_cardinality_threshold_ppm"));
        assert_eq!(error.fact_value("max_value"), Some("1000000"));
        assert_eq!(error.fact_value("reason"), Some("ratio_ppm_above_one_hundred_percent"));
    }

    #[test]
    fn aggregate_strategy_execution_options_reject_contradictory_probe_bounds() {
        let error = AggregateStrategyExecutionOptions::default()
            .with_exact_grouped_key_cardinality_probe_min_input_rows(10)
            .with_exact_three_key_cardinality_probe_max_input_rows(9)
            .validate()
            .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("exact_grouped_key_cardinality_probe_min_input_rows"));
        assert_eq!(error.fact_value("bound_field"), Some("exact_three_key_cardinality_probe_max_input_rows"));
        assert_eq!(
            error.fact_value("reason"),
            Some("three_key_probe_max_input_rows_below_grouped_probe_min_input_rows")
        );
    }

    #[test]
    fn aggregate_strategy_execution_options_accept_zero_threshold_overrides() {
        AggregateStrategyExecutionOptions::default()
            .with_encoded_key_partial_low_cardinality_threshold_ppm(0)
            .with_prepared_sort_groupby_low_cardinality_threshold_ppm(0)
            .with_exact_grouped_key_cardinality_probe_min_input_rows(0)
            .with_exact_three_key_cardinality_probe_max_input_rows(0)
            .with_source_adjacent_prepared_sort_groupby_max_row_groups(0)
            .with_source_adjacent_prepared_sort_groupby_max_source_read_bytes(0)
            .with_source_adjacent_prepared_sort_groupby_max_fan_in(0)
            .validate()
            .unwrap();
    }
}
