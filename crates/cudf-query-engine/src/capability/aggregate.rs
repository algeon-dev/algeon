//! Aggregate-strategy capability analysis.
//!
//! This module classifies grouped aggregate stages into a selected execution
//! strategy plus locally rejected candidate strategies. It does not enable an
//! optimized path by itself; executor admission must still have a concrete
//! reservation and execution contract before selecting anything except the
//! default libcudf groupby strategy.

#![deny(missing_docs)]

use std::collections::BTreeMap;

use arrow::datatypes::{DataType, Schema};

use crate::{
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
    if stage != AggregateExecutionStage::Partial {
        return Some("prepared_sort_groupby_requires_partial_stage");
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
    if aggregate_spec_has_stddev_samp(spec) {
        return Some("prepared_sort_groupby_stddev_samp_unsupported");
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
    if aggregate_spec_has_stddev_samp(spec) {
        return Some("partitioned_merge_stddev_samp_unsupported");
    }
    if distinct_aggregate_count(spec) > 0 {
        return Some("partitioned_merge_distinct_aggregate_unsupported");
    }
    None
}

fn aggregate_spec_has_stddev_samp(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::StddevSamp)
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
    match function {
        AggregateFunction::Sum => "sum",
        AggregateFunction::Min => "min",
        AggregateFunction::Max => "max",
        AggregateFunction::Count => "count",
        AggregateFunction::NUnique => "nunique",
        AggregateFunction::Avg => "avg",
        AggregateFunction::StddevSamp => "stddev_samp",
    }
}

fn aggregate_filter_count(spec: &AggregateSpec) -> usize {
    spec.aggr_exprs.iter().filter(|expr| expr.filter.is_some()).count()
}

fn distinct_aggregate_count(spec: &AggregateSpec) -> usize {
    spec.aggr_exprs.iter().filter(|expr| expr.distinct).count()
}

fn aggregate_encoded_key_string_dtype(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn aggregate_encoded_key_companion_dtype(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}
