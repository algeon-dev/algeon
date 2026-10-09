//! Cost-model selection among grouped partial-aggregate strategies.
//!
//! Capability validation in query-planning (`aggregate_strategy_capability_decision`)
//! decides which strategies an aggregate shape admits. This module applies the
//! runtime cost model on top of that decision. `LibcudfGroupby` stays selected
//! unless completed partial evidence for the same structural key shows a
//! reduction ratio low enough for `EncodedKeyPartial` or `PreparedSortGroupby`,
//! or a source-adjacent prepared-groupby preflight on the actual input returns
//! exact group facts. No disposable cardinality probe runs on this path.
//!
//! Every selection and rejection writes its reason into `decision.facts`. Those
//! keys feed the aggregate strategy TSV family and are a public contract: a new
//! branch must write the same keys as the existing ones, with a new value.

use std::{collections::BTreeMap, time::Duration};

use arrow::datatypes::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use query_planning::{
    capability::{AggregateStrategyExecutionOptions, aggregate_strategy_capability_decision},
    source::parquet_source_path_fact,
    util::fnv1a64,
};

use super::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateFunction, AggregateSpec,
    AggregateStrategyCandidateRejection, AggregateStrategyCapabilityDecision, ExecutionMetrics, Expr, NodeId,
    PartialAggregateInputEvidence, SourceChunk,
};
use crate::{datatypes::arrow_fixed_width_byte_size, exec::execution_metrics::PriorEvidenceScope};

#[cfg(test)]
pub(super) fn select_partial_aggregate_strategy_decision(
    decision: AggregateStrategyCapabilityDecision,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchema,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    metrics: &ExecutionMetrics,
) -> AggregateStrategyCapabilityDecision {
    select_partial_aggregate_strategy_decision_with_options(
        decision,
        node_id,
        spec,
        input_schema,
        input_rows,
        input_evidence,
        AggregateStrategyExecutionOptions::default(),
        metrics,
    )
}

/// Applies the cost model to a capability decision for one partial aggregate.
///
/// The capability decision names the admissible strategies through the
/// `*_contract_status` facts. Selection then proceeds in order: an input below
/// the encoded-key minimum keeps libcudf groupby; without completed prior
/// evidence for the structural key, both candidates are rejected and the
/// `partial_strategy_unseen_*` facts record which path an unseen shape takes;
/// with evidence, the reduction ratio picks prepared sort-groupby, then
/// encoded-key, else libcudf. Prior evidence is the latest completed partial
/// with the same structural key, preferring the same node and falling back to
/// any node (`ExecutionMetrics::prior_partial_aggregate_cardinality`).
///
/// `input_evidence` is also part of the structural key, so source-adjacent
/// partials reuse evidence only across the same chunk set and partition shape;
/// partials without chunk evidence share one key per aggregate shape.
#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(super) fn select_partial_aggregate_strategy_decision_with_options(
    mut decision: AggregateStrategyCapabilityDecision,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchema,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    options: AggregateStrategyExecutionOptions,
    metrics: &ExecutionMetrics,
) -> AggregateStrategyCapabilityDecision {
    add_aggregate_strategy_execution_option_facts(&mut decision.facts, options);
    add_prepared_sort_groupby_gate_facts(&mut decision, input_rows);
    decision.facts.insert("encoded_key_partial_cost_model", "prior_partial_reduction_ratio".to_string());
    decision.facts.insert(
        "encoded_key_partial_reduction_ratio_threshold_ppm",
        options.encoded_key_partial_low_cardinality_threshold_ppm().to_string(),
    );
    decision.facts.insert("encoded_key_partial_current_input_rows", input_rows.to_string());
    add_partial_aggregate_input_evidence_facts(&mut decision.facts, input_evidence);
    let structural_key = aggregate_structural_key(spec, input_schema, input_evidence);
    decision.facts.insert("aggregate_structural_key", structural_key.clone());
    decision.facts.insert("prior_cardinality_evidence_scope", "absent".to_string());
    let encoded_key_partial_available = encoded_key_partial_contract_available(&decision);
    let prepared_sort_groupby_available = prepared_sort_groupby_contract_available(&decision);
    if !encoded_key_partial_available {
        decision.facts.insert("encoded_key_partial_cost_model_status", "not_applicable".to_string());
        decision.facts.insert("encoded_key_partial_selection_reason", "capability_rejected".to_string());
    }
    if !encoded_key_partial_available && !prepared_sort_groupby_available {
        return decision;
    }
    // Source-adjacent partition facts remain an admissibility envelope. They must
    // not, by themselves, select prepared sort-groupby. Prior completed partial
    // evidence remains the only reused cardinality path allowed to pick prepared.
    let prepared_sort_groupby_runtime_candidate = prepared_sort_groupby_available;
    // Hot-path strategy selection never launches a disposable full-column exact
    // probe. Keep the public fact keys; mark the probe explicitly disabled.
    decision.facts.insert("exact_cardinality_probe_status", "disabled_non_reusable_probe".to_string());

    if input_rows < options.encoded_key_partial_min_input_rows() {
        if encoded_key_partial_available {
            decision = reject_encoded_key_partial_for_cost_model(
                decision,
                "encoded_key_partial_cost_model_small_input",
                "input_rows_below_encoded_key_partial_minimum",
                None,
            );
        }
        return if prepared_sort_groupby_runtime_candidate {
            reject_prepared_sort_groupby_for_cost_model(
                decision,
                "prepared_sort_groupby_cost_model_small_input",
                "input_rows_below_partial_aggregate_strategy_minimum",
            )
        } else {
            decision
        };
    }

    let Some(evidence) = prior_partial_cardinality_evidence(metrics, node_id, &structural_key) else {
        return reject_partial_strategies_without_completed_cardinality_evidence(
            decision,
            encoded_key_partial_available,
            prepared_sort_groupby_runtime_candidate,
            input_rows,
            input_evidence,
            options,
        );
    };
    add_prior_cardinality_facts(&mut decision, &evidence, spec, input_schema);
    if evidence.reduction_ratio_ppm <= options.prepared_sort_groupby_low_cardinality_threshold_ppm()
        && prepared_sort_groupby_runtime_candidate
        && prepared_sort_groupby_should_run_for_prior_evidence(&decision, &evidence)
    {
        return select_prepared_sort_groupby_for_prior_cardinality(decision);
    }
    if evidence.reduction_ratio_ppm <= options.encoded_key_partial_low_cardinality_threshold_ppm()
        && encoded_key_partial_available
    {
        if prepared_sort_groupby_runtime_candidate {
            decision = reject_prepared_sort_groupby_for_cost_model(
                decision,
                "prepared_sort_groupby_encoded_key_preferred_for_simple_string_key",
                "encoded_key_preferred_for_simple_string_key",
            );
        }
        decision.facts.insert("encoded_key_partial_cost_model_status", "selected_low_cardinality".to_string());
        decision
            .facts
            .insert("encoded_key_partial_selection_reason", "prior_reduction_ratio_at_or_below_threshold".to_string());
        return decision;
    }

    if encoded_key_partial_available {
        decision = reject_encoded_key_partial_for_cost_model(
            decision,
            "encoded_key_partial_cost_model_high_cardinality",
            "prior_reduction_ratio_above_threshold",
            Some(evidence.clone()),
        );
    }
    if prepared_sort_groupby_runtime_candidate {
        reject_prepared_sort_groupby_for_cost_model(
            decision,
            "prepared_sort_groupby_cost_model_high_cardinality",
            "prior_reduction_ratio_above_threshold",
        )
    } else {
        decision
    }
}

/// Whether the caller may run the prepared sort-groupby preflight on the
/// actual input after the cost model kept libcudf groupby.
///
/// The preflight sorts the group keys of the input frame itself, which is the cost of
/// the prepared path itself. It is affordable only for a source-adjacent
/// partial whose input is one group-key partition batch with a bounded fan-in,
/// because that batch is already sized by the partition policy. The exact
/// group facts it returns then decide the strategy through
/// [`prepared_sort_groupby_should_select_for_runtime_facts`].
pub(super) fn source_adjacent_prepared_sort_groupby_runtime_candidate(
    decision: &AggregateStrategyCapabilityDecision,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    options: AggregateStrategyExecutionOptions,
) -> bool {
    if decision.selected_strategy != AggregateExecutionStrategy::LibcudfGroupby
        || !prepared_sort_groupby_contract_available(decision)
        || input_rows < options.encoded_key_partial_min_input_rows()
    {
        return false;
    }
    let Some(evidence) = input_evidence else {
        return false;
    };
    evidence.source_adjacent_partial
        && evidence.source_adjacent_partition_strategy == "group_key"
        && evidence.source_adjacent_partition_key_count.is_some_and(|key_count| key_count > 0)
        && evidence
            .source_adjacent_max_partition_fan_in
            .is_some_and(|fan_in| fan_in <= options.source_adjacent_prepared_sort_groupby_max_fan_in())
}

pub(super) fn add_prepared_sort_groupby_facts(
    decision: &mut AggregateStrategyCapabilityDecision,
    num_groups: usize,
    grouped_key_output_bytes: usize,
    max_group_size: usize,
) {
    decision.facts.insert("prepared_sort_groupby_num_groups", num_groups.to_string());
    decision.facts.insert("prepared_sort_groupby_grouped_key_output_bytes", grouped_key_output_bytes.to_string());
    decision.facts.insert("prepared_sort_groupby_max_group_size", max_group_size.to_string());
}

/// Selection from exact preflight facts. A single string key is judged by an
/// absolute group count, because its grouped key output scales with the distinct
/// strings and does not track the reduction ratio. Every other key shape uses the
/// ratio threshold.
pub(super) fn prepared_sort_groupby_should_select_for_runtime_facts(
    decision: &AggregateStrategyCapabilityDecision,
    input_rows: u64,
    num_groups: usize,
    options: AggregateStrategyExecutionOptions,
) -> bool {
    exact_cardinality_probe_values_should_select_prepared_sort_groupby(
        decision,
        num_groups as u64,
        reduction_ratio_ppm(input_rows, num_groups as u64),
        options,
    )
}

pub(super) fn select_prepared_sort_groupby_for_runtime_facts(
    decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    select_prepared_sort_groupby_with_cost_model(
        decision,
        "selected_runtime_prepared_groupby",
        "runtime_preflight_exact_grouped_key_facts",
    )
}

pub(super) fn reject_prepared_sort_groupby_for_runtime_facts(
    decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    reject_prepared_sort_groupby_with_cost_model(
        decision,
        "prepared_sort_groupby_runtime_preflight_high_cardinality",
        "runtime_preflight_exact_grouped_key_facts_above_threshold",
    )
}

pub(in crate::exec::operators::aggregate) fn add_aggregate_strategy_execution_option_facts(
    facts: &mut BTreeMap<&'static str, String>,
    options: AggregateStrategyExecutionOptions,
) {
    facts.insert(
        "aggregate_encoded_key_partial_low_cardinality_threshold_ppm",
        options.encoded_key_partial_low_cardinality_threshold_ppm().to_string(),
    );
    facts.insert(
        "aggregate_encoded_key_partial_min_input_rows",
        options.encoded_key_partial_min_input_rows().to_string(),
    );
    facts.insert(
        "aggregate_exact_grouped_key_cardinality_probe_max_key_columns",
        options.exact_grouped_key_cardinality_probe_max_key_columns().to_string(),
    );
    facts.insert(
        "aggregate_exact_single_string_avg_cardinality_probe_min_input_rows",
        options.exact_single_string_avg_cardinality_probe_min_input_rows().to_string(),
    );
    facts.insert(
        "aggregate_exact_single_key_cardinality_probe_min_input_rows",
        options.exact_single_key_cardinality_probe_min_input_rows().to_string(),
    );
    facts.insert(
        "aggregate_exact_grouped_key_cardinality_probe_min_input_rows",
        options.exact_grouped_key_cardinality_probe_min_input_rows().to_string(),
    );
    facts.insert(
        "aggregate_exact_single_string_key_prepared_max_groups",
        options.exact_single_string_key_prepared_max_groups().to_string(),
    );
    facts.insert(
        "aggregate_exact_three_key_cardinality_probe_max_input_rows",
        options.exact_three_key_cardinality_probe_max_input_rows().to_string(),
    );
    facts.insert(
        "aggregate_exact_five_key_cardinality_probe_max_input_rows",
        options.exact_five_key_cardinality_probe_max_input_rows().to_string(),
    );
    facts.insert(
        "aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm",
        options.prepared_sort_groupby_low_cardinality_threshold_ppm().to_string(),
    );
    facts.insert(
        "aggregate_source_adjacent_prepared_sort_groupby_max_row_groups",
        options.source_adjacent_prepared_sort_groupby_max_row_groups().to_string(),
    );
    facts.insert(
        "aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes",
        options.source_adjacent_prepared_sort_groupby_max_source_read_bytes().to_string(),
    );
    facts.insert(
        "aggregate_source_adjacent_prepared_sort_groupby_max_fan_in",
        options.source_adjacent_prepared_sort_groupby_max_fan_in().to_string(),
    );
}

pub(super) fn add_partial_aggregate_input_evidence_facts(
    facts: &mut BTreeMap<&'static str, String>,
    input_evidence: Option<PartialAggregateInputEvidence>,
) {
    let Some(input_evidence) = input_evidence else {
        facts.insert("source_adjacent_partial", "unknown".to_string());
        facts.insert("source_chunk_fingerprint", "unknown".to_string());
        facts.insert("source_chunk_row_group_count", "unknown".to_string());
        facts.insert("source_chunk_reads_all_row_groups", "unknown".to_string());
        facts.insert("source_chunk_known_row_count", "unknown".to_string());
        facts.insert("source_chunk_estimated_source_read_bytes", "unknown".to_string());
        facts.insert("source_chunk_size_estimate_based", "unknown".to_string());
        facts.insert("source_adjacent_partition_strategy", "unknown".to_string());
        facts.insert("source_adjacent_partition_count", "unknown".to_string());
        facts.insert("source_adjacent_non_empty_partition_count", "unknown".to_string());
        facts.insert("source_adjacent_partition_key_count", "unknown".to_string());
        facts.insert("source_adjacent_max_partition_fan_in", "unknown".to_string());
        return;
    };
    facts.insert("source_adjacent_partial", input_evidence.source_adjacent_partial.to_string());
    facts.insert(
        "source_chunk_fingerprint",
        input_evidence
            .source_chunk_fingerprint
            .map_or_else(|| "unknown".to_string(), |fingerprint| format!("{fingerprint:016x}")),
    );
    facts.insert("source_chunk_row_group_count", input_evidence.source_chunk_row_group_count.to_string());
    facts.insert("source_chunk_reads_all_row_groups", input_evidence.source_chunk_reads_all_row_groups.to_string());
    facts.insert(
        "source_chunk_known_row_count",
        optional_strategy_fact_u64(input_evidence.source_chunk_known_row_count),
    );
    facts.insert(
        "source_chunk_estimated_source_read_bytes",
        optional_strategy_fact_u64(input_evidence.source_chunk_estimated_source_read_bytes),
    );
    facts.insert("source_chunk_size_estimate_based", input_evidence.source_chunk_size_estimate_based.to_string());
    facts.insert("source_adjacent_partition_strategy", input_evidence.source_adjacent_partition_strategy.to_string());
    facts.insert(
        "source_adjacent_partition_count",
        optional_strategy_fact_u64(input_evidence.source_adjacent_partition_count),
    );
    facts.insert(
        "source_adjacent_non_empty_partition_count",
        optional_strategy_fact_u64(input_evidence.source_adjacent_non_empty_partition_count),
    );
    facts.insert(
        "source_adjacent_partition_key_count",
        optional_strategy_fact_u64(input_evidence.source_adjacent_partition_key_count),
    );
    facts.insert(
        "source_adjacent_max_partition_fan_in",
        optional_strategy_fact_u64(input_evidence.source_adjacent_max_partition_fan_in),
    );
}

fn optional_strategy_fact_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

/// A completed partial's cardinality, read back from the facts an earlier
/// `record_grouped_aggregate_strategy_decision` call stored.
#[derive(Debug, Clone)]
struct PriorAggregateCardinalityEvidence {
    input_rows: u64,
    output_group_count: u64,
    reduction_ratio_ppm: u64,
    selected_strategy: String,
    evidence_scope: &'static str,
}

fn prior_partial_cardinality_evidence(
    metrics: &ExecutionMetrics,
    node_id: Option<NodeId>,
    structural_key: &str,
) -> Option<PriorAggregateCardinalityEvidence> {
    let node_id = node_id.map(|node_id| node_id.to_string());
    let (evidence, scope) = metrics.prior_partial_aggregate_cardinality(node_id.as_deref(), structural_key)?;
    Some(PriorAggregateCardinalityEvidence {
        input_rows: evidence.input_rows,
        output_group_count: evidence.output_group_count,
        reduction_ratio_ppm: evidence.reduction_ratio_ppm,
        selected_strategy: evidence.selected_strategy,
        evidence_scope: match scope {
            PriorEvidenceScope::ExactNode => "exact_node",
            PriorEvidenceScope::StructuralReused => "structural_reused",
        },
    })
}

fn encoded_key_partial_contract_available(decision: &AggregateStrategyCapabilityDecision) -> bool {
    decision.facts.get("encoded_key_partial_contract_status").map(String::as_str) == Some("available")
}

fn add_prepared_sort_groupby_gate_facts(decision: &mut AggregateStrategyCapabilityDecision, input_rows: u64) {
    decision.facts.insert("prepared_sort_groupby_cost_model", "exact_grouped_key_facts_required".to_string());
    decision.facts.insert("prepared_sort_groupby_current_input_rows", input_rows.to_string());
    decision.facts.insert("prepared_sort_groupby_num_groups", "unknown".to_string());
    decision.facts.insert("prepared_sort_groupby_grouped_key_output_bytes", "unknown".to_string());
    decision.facts.insert("prepared_sort_groupby_max_group_size", "unknown".to_string());
    decision.facts.insert("exact_cardinality_probe_status", "disabled_non_reusable_probe".to_string());
    decision.facts.insert("exact_cardinality_probe_key", "none".to_string());
    decision.facts.insert("exact_cardinality_probe_input_rows", input_rows.to_string());
    decision.facts.insert("exact_cardinality_probe_non_null_distinct_count", "unknown".to_string());
    decision.facts.insert("exact_cardinality_probe_null_count", "unknown".to_string());
    decision.facts.insert("exact_cardinality_probe_num_groups", "unknown".to_string());
    decision.facts.insert("exact_cardinality_probe_reduction_ratio_ppm", "unknown".to_string());
    decision.facts.insert("exact_cardinality_probe_elapsed_ns", "0".to_string());
    if prepared_sort_groupby_contract_available(decision) {
        decision.facts.insert("prepared_sort_groupby_cost_model_status", "pending_runtime_evidence".to_string());
        decision
            .facts
            .insert("prepared_sort_groupby_selection_reason", "runtime_cardinality_evidence_required".to_string());
    } else {
        decision.facts.insert("prepared_sort_groupby_cost_model_status", "not_applicable".to_string());
        decision.facts.insert("prepared_sort_groupby_selection_reason", "capability_rejected".to_string());
    }
}

fn prepared_sort_groupby_contract_available(decision: &AggregateStrategyCapabilityDecision) -> bool {
    decision.facts.get("prepared_sort_groupby_contract_status").map(String::as_str) == Some("available")
}

/// Whether low prior cardinality should select prepared sort-groupby rather
/// than encoded-key partial. A single simple string key prefers encoded-key,
/// whose dictionary encode is cheaper than sorting the strings; prepared
/// sort-groupby wins once a fixed-width key or a second key is present, and
/// always when the prior run already used it.
fn prepared_sort_groupby_should_run_for_prior_evidence(
    decision: &AggregateStrategyCapabilityDecision,
    evidence: &PriorAggregateCardinalityEvidence,
) -> bool {
    if evidence.selected_strategy == AggregateExecutionStrategy::PreparedSortGroupby.as_str() {
        return true;
    }
    let string_keys =
        decision.facts.get("string_group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    let fixed_width_keys =
        decision.facts.get("fixed_width_group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    let group_keys = decision.facts.get("group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);

    fixed_width_keys > 0 || group_keys > 1 || string_keys == 0
}

fn exact_cardinality_probe_values_should_select_prepared_sort_groupby(
    decision: &AggregateStrategyCapabilityDecision,
    num_groups: u64,
    reduction_ratio_ppm: u64,
    options: AggregateStrategyExecutionOptions,
) -> bool {
    let string_keys =
        decision.facts.get("string_group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    let fixed_width_keys =
        decision.facts.get("fixed_width_group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    let group_keys = decision.facts.get("group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);

    if group_keys == 1 && string_keys == 1 && fixed_width_keys == 0 {
        return num_groups <= options.exact_single_string_key_prepared_max_groups();
    }
    reduction_ratio_ppm <= options.prepared_sort_groupby_low_cardinality_threshold_ppm()
}

/// First-sight path: without completed evidence both optimized candidates are
/// rejected, and the `partial_strategy_unseen_*` facts say whether the shape
/// counts as a large variable-width input. Reports use that to tell a string
/// grouped aggregate that will stream through group-key partitioning from one
/// that libcudf groupby handles whole.
fn reject_partial_strategies_without_completed_cardinality_evidence(
    mut decision: AggregateStrategyCapabilityDecision,
    encoded_key_partial_available: bool,
    prepared_sort_groupby_runtime_candidate: bool,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    options: AggregateStrategyExecutionOptions,
) -> AggregateStrategyCapabilityDecision {
    let large_variable_width = large_variable_width_partial_envelope(&decision, input_rows, input_evidence, options);
    decision.facts.insert(
        "partial_strategy_unseen_envelope",
        if large_variable_width { "large_variable_width" } else { "small_or_fixed_width" }.to_string(),
    );
    if large_variable_width {
        decision.facts.insert("partial_strategy_unseen_path", "streaming_group_key_partition_or_libcudf".to_string());
    } else {
        decision.facts.insert("partial_strategy_unseen_path", "libcudf_groupby".to_string());
    }

    if encoded_key_partial_available {
        decision = reject_encoded_key_partial_for_cost_model(
            decision,
            "encoded_key_partial_cost_model_no_prior_cardinality_evidence",
            if large_variable_width {
                "completed_cardinality_evidence_required_for_large_variable_width"
            } else {
                "completed_cardinality_evidence_required"
            },
            None,
        );
    }
    if prepared_sort_groupby_runtime_candidate {
        decision = reject_prepared_sort_groupby_for_cost_model(
            decision,
            if large_variable_width {
                "prepared_sort_groupby_requires_completed_cardinality_evidence_for_large_variable_width"
            } else {
                "prepared_sort_groupby_requires_completed_cardinality_evidence"
            },
            if large_variable_width {
                "completed_cardinality_evidence_required_for_large_variable_width"
            } else {
                "completed_cardinality_evidence_required"
            },
        );
    }
    decision
}

fn large_variable_width_partial_envelope(
    decision: &AggregateStrategyCapabilityDecision,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    options: AggregateStrategyExecutionOptions,
) -> bool {
    let string_keys =
        decision.facts.get("string_group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    if string_keys == 0 {
        return false;
    }
    if input_rows >= options.encoded_key_partial_min_input_rows() {
        return true;
    }
    let Some(evidence) = input_evidence else {
        return false;
    };
    evidence
        .source_chunk_estimated_source_read_bytes
        .is_some_and(|bytes| bytes > options.source_adjacent_prepared_sort_groupby_max_source_read_bytes())
        || evidence.source_chunk_row_group_count > options.source_adjacent_prepared_sort_groupby_max_row_groups()
        || evidence
            .source_chunk_known_row_count
            .is_some_and(|rows| rows >= options.encoded_key_partial_min_input_rows())
}

fn select_prepared_sort_groupby_for_prior_cardinality(
    decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    select_prepared_sort_groupby_with_cost_model(
        decision,
        "selected_prior_low_cardinality",
        "prior_reduction_ratio_at_or_below_threshold",
    )
}

/// Selecting prepared sort-groupby supersedes an available encoded-key candidate and
/// keeps it in the TSV row. The encoded-key facts are rewritten and its rejection is
/// inserted first, so the row explains both outcomes.
fn select_prepared_sort_groupby_with_cost_model(
    mut decision: AggregateStrategyCapabilityDecision,
    cost_model_status: &'static str,
    selection_reason: &'static str,
) -> AggregateStrategyCapabilityDecision {
    decision.selected_strategy = AggregateExecutionStrategy::PreparedSortGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::PreparedSortGroupby.as_str().to_string());
    decision.facts.insert("prepared_sort_groupby_rejected_reason", "none".to_string());
    decision.facts.insert("prepared_sort_groupby_cost_model_status", cost_model_status.to_string());
    decision.facts.insert("prepared_sort_groupby_selection_reason", selection_reason.to_string());
    decision.rejected_candidates.retain(|rejection| {
        rejection.strategy != AggregateExecutionStrategy::PreparedSortGroupby
            && rejection.strategy != AggregateExecutionStrategy::EncodedKeyPartial
    });
    if decision.facts.get("encoded_key_partial_contract_status").map(String::as_str) == Some("available") {
        decision.facts.insert(
            "encoded_key_partial_rejected_reason",
            "encoded_key_partial_superseded_by_prepared_sort_groupby".to_string(),
        );
        decision.facts.insert("encoded_key_partial_cost_model_status", "not_selected".to_string());
        decision.facts.insert("encoded_key_partial_selection_reason", "prepared_sort_groupby_selected".to_string());
        decision.rejected_candidates.insert(
            0,
            AggregateStrategyCandidateRejection {
                strategy: AggregateExecutionStrategy::EncodedKeyPartial,
                reason: "encoded_key_partial_superseded_by_prepared_sort_groupby",
            },
        );
    }
    update_rejected_candidate_summary(&mut decision);
    decision
}

/// Rejection after a runtime preflight. Unlike [`reject_prepared_sort_groupby_for_cost_model`]
/// this also resets the selected strategy, because the preflight may run after
/// prior evidence had already selected prepared sort-groupby.
fn reject_prepared_sort_groupby_with_cost_model(
    mut decision: AggregateStrategyCapabilityDecision,
    rejection_reason: &'static str,
    selection_reason: &'static str,
) -> AggregateStrategyCapabilityDecision {
    if !prepared_sort_groupby_contract_available(&decision) {
        return decision;
    }
    decision.selected_strategy = AggregateExecutionStrategy::LibcudfGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::LibcudfGroupby.as_str().to_string());
    decision.facts.insert("prepared_sort_groupby_rejected_reason", rejection_reason.to_string());
    decision.facts.insert("prepared_sort_groupby_cost_model_status", "rejected_by_cost_model".to_string());
    decision.facts.insert("prepared_sort_groupby_selection_reason", selection_reason.to_string());
    if let Some(rejection) = decision
        .rejected_candidates
        .iter_mut()
        .find(|rejection| rejection.strategy == AggregateExecutionStrategy::PreparedSortGroupby)
    {
        rejection.reason = rejection_reason;
    } else {
        decision.rejected_candidates.push(AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::PreparedSortGroupby,
            reason: rejection_reason,
        });
    }
    update_rejected_candidate_summary(&mut decision);
    decision
}

fn reject_prepared_sort_groupby_for_cost_model(
    mut decision: AggregateStrategyCapabilityDecision,
    reason: &'static str,
    selection_reason: &'static str,
) -> AggregateStrategyCapabilityDecision {
    if !prepared_sort_groupby_contract_available(&decision) {
        return decision;
    }
    decision.facts.insert("prepared_sort_groupby_rejected_reason", reason.to_string());
    decision.facts.insert("prepared_sort_groupby_cost_model_status", "rejected_by_cost_model".to_string());
    decision.facts.insert("prepared_sort_groupby_selection_reason", selection_reason.to_string());
    if let Some(rejection) = decision
        .rejected_candidates
        .iter_mut()
        .find(|rejection| rejection.strategy == AggregateExecutionStrategy::PreparedSortGroupby)
    {
        rejection.reason = reason;
    } else {
        decision.rejected_candidates.push(AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::PreparedSortGroupby,
            reason,
        });
    }
    update_rejected_candidate_summary(&mut decision);
    decision
}

fn reject_encoded_key_partial_for_cost_model(
    mut decision: AggregateStrategyCapabilityDecision,
    reason: &'static str,
    selection_reason: &'static str,
    evidence: Option<PriorAggregateCardinalityEvidence>,
) -> AggregateStrategyCapabilityDecision {
    decision.selected_strategy = AggregateExecutionStrategy::LibcudfGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::LibcudfGroupby.as_str().to_string());
    decision.facts.insert("encoded_key_partial_rejected_reason", reason.to_string());
    decision.facts.insert("encoded_key_partial_cost_model_status", "rejected_by_cost_model".to_string());
    decision.facts.insert("encoded_key_partial_selection_reason", selection_reason.to_string());
    if let Some(evidence) = evidence {
        add_encoded_key_prior_cardinality_facts(&mut decision, &evidence);
    }
    decision.rejected_candidates.insert(
        0,
        AggregateStrategyCandidateRejection { strategy: AggregateExecutionStrategy::EncodedKeyPartial, reason },
    );
    update_rejected_candidate_summary(&mut decision);
    decision
}

fn add_encoded_key_prior_cardinality_facts(
    decision: &mut AggregateStrategyCapabilityDecision,
    evidence: &PriorAggregateCardinalityEvidence,
) {
    decision.facts.insert("encoded_key_partial_prior_input_rows", evidence.input_rows.to_string());
    decision.facts.insert("encoded_key_partial_prior_output_group_count", evidence.output_group_count.to_string());
    decision.facts.insert("encoded_key_partial_prior_reduction_ratio_ppm", evidence.reduction_ratio_ppm.to_string());
    decision.facts.insert("encoded_key_partial_prior_strategy", evidence.selected_strategy.clone());
    decision.facts.insert("prior_cardinality_evidence_scope", evidence.evidence_scope.to_string());
}

fn add_prior_cardinality_facts(
    decision: &mut AggregateStrategyCapabilityDecision,
    evidence: &PriorAggregateCardinalityEvidence,
    spec: &AggregateSpec,
    input_schema: &ArrowSchema,
) {
    add_encoded_key_prior_cardinality_facts(decision, evidence);
    decision.facts.insert("prepared_sort_groupby_prior_input_rows", evidence.input_rows.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_output_group_count", evidence.output_group_count.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_reduction_ratio_ppm", evidence.reduction_ratio_ppm.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_strategy", evidence.selected_strategy.clone());
    decision.facts.insert(
        "prepared_sort_groupby_prior_grouped_key_output_bytes",
        optional_strategy_fact_u64(estimate_grouped_key_output_bytes(spec, input_schema, evidence.output_group_count)),
    );
    decision.facts.insert("prepared_sort_groupby_prior_max_group_size", optional_strategy_fact_u64(None));
}

fn estimate_grouped_key_output_bytes(
    spec: &AggregateSpec,
    input_schema: &ArrowSchema,
    output_group_count: u64,
) -> Option<u64> {
    let mut bytes_per_group = 0u64;
    for expr in &spec.group_exprs {
        let field = input_schema.field_with_name(plain_group_key_column_name(expr)?).ok()?;
        bytes_per_group = bytes_per_group.checked_add(arrow_fixed_width_byte_size(field.data_type())?)?;
    }
    bytes_per_group.checked_mul(output_group_count)
}

fn plain_group_key_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.name()),
        Expr::Alias { expr, .. } => plain_group_key_column_name(expr),
        _ => None,
    }
}

fn update_rejected_candidate_summary(decision: &mut AggregateStrategyCapabilityDecision) {
    decision.facts.insert(
        "rejected_candidate_strategies",
        decision
            .rejected_candidates
            .iter()
            .map(|rejection| format!("{}:{}", rejection.strategy.as_str(), rejection.reason))
            .collect::<Vec<_>>()
            .join(";"),
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
pub(in crate::exec) fn record_default_grouped_aggregate_strategy_decision(
    metrics: &ExecutionMetrics,
    node_id: Option<String>,
    stage: AggregateExecutionStage,
    aggregate_strategy_options: AggregateStrategyExecutionOptions,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    elapsed: Duration,
) {
    let decision = aggregate_strategy_capability_decision(spec, input_schema.as_ref(), stage);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id,
        stage,
        aggregate_strategy_options,
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision,
        Duration::ZERO,
        Duration::ZERO,
        elapsed,
        Duration::ZERO,
        elapsed,
        None,
    );
}

/// Publishes one grouped aggregate's strategy decision with its runtime facts.
///
/// Recording a `Partial` stage row is also what creates prior evidence: the
/// metrics store parses `aggregate_structural_key`, `input_rows`,
/// `output_group_count`, `reduction_ratio_ppm` and `selected_strategy` back out
/// of these facts for later selections, so those keys must stay present and
/// numeric. `chunk_row_count` is the partial's input rows and zero for the
/// other stages.
#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a projection DTO"
)]
pub(in crate::exec) fn record_grouped_aggregate_strategy_decision(
    metrics: &ExecutionMetrics,
    node_id: Option<String>,
    stage: AggregateExecutionStage,
    aggregate_strategy_options: AggregateStrategyExecutionOptions,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    decision: AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    encode_elapsed: Duration,
    groupby_elapsed: Duration,
    decode_elapsed: Duration,
    strategy_elapsed: Duration,
    input_evidence: Option<PartialAggregateInputEvidence>,
) {
    let mut facts = decision.facts;
    add_aggregate_strategy_execution_option_facts(&mut facts, aggregate_strategy_options);
    add_partial_aggregate_input_evidence_facts(&mut facts, input_evidence);
    facts
        .entry("aggregate_structural_key")
        .or_insert_with(|| aggregate_structural_key(spec, input_schema.as_ref(), input_evidence));
    facts.insert("input_rows", input_rows.to_string());
    facts.insert("chunk_row_count", if stage == AggregateExecutionStage::Partial { input_rows } else { 0 }.to_string());
    let progress = metrics.progress();
    facts.insert("chunk_count", progress.source_chunks_planned.to_string());
    facts.insert("partial_aggregate_count", progress.partial_aggregate_count.to_string());
    facts.insert("output_group_count", output_rows.to_string());
    facts.insert("reduction_ratio_ppm", reduction_ratio_ppm(input_rows, output_rows).to_string());
    facts.insert("selection_elapsed_ns", duration_ns(selection_elapsed).to_string());
    facts.insert("encode_elapsed_ns", duration_ns(encode_elapsed).to_string());
    facts.insert("groupby_elapsed_ns", duration_ns(groupby_elapsed).to_string());
    facts.insert("decode_elapsed_ns", duration_ns(decode_elapsed).to_string());
    facts.insert("strategy_operator_elapsed_ns", duration_ns(strategy_elapsed).to_string());
    metrics
        .record_aggregate_strategy_decision(node_id.unwrap_or_else(|| format!("aggregate.{}", stage.as_str())), &facts);
}

/// Output groups per million input rows; zero for an empty input, which the
/// evidence store also refuses to retain.
pub(super) fn reduction_ratio_ppm(input_rows: u64, output_rows: u64) -> u64 {
    if input_rows == 0 {
        return 0;
    }
    output_rows.saturating_mul(1_000_000).saturating_div(input_rows)
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Identity under which partial cardinality evidence is reused.
///
/// The key hashes the group and aggregate expressions with their input dtypes,
/// the full input schema, and the source-chunk evidence. Including the chunk
/// evidence keeps a reduction ratio measured on one chunk set from steering a
/// partial over a differently shaped input; the trade-off is that evidence
/// rarely carries across chunks of a source-adjacent aggregate unless the
/// chunk fingerprint and partition shape repeat.
pub(super) fn aggregate_structural_key(
    spec: &AggregateSpec,
    input_schema: &ArrowSchema,
    input_evidence: Option<PartialAggregateInputEvidence>,
) -> String {
    let group_exprs = spec
        .group_exprs
        .iter()
        .map(|expr| aggregate_expr_structural_part(expr, input_schema))
        .collect::<Vec<_>>()
        .join(",");
    let aggregate_exprs = spec
        .aggr_exprs
        .iter()
        .map(|expr| {
            format!(
                "fn={};distinct={};arg={};filter={}",
                aggregate_function_structural_name(expr.function),
                expr.distinct,
                aggregate_expr_structural_part(&expr.expr, input_schema),
                expr.filter
                    .as_ref()
                    .map(|filter| aggregate_expr_structural_part(filter, input_schema))
                    .unwrap_or_else(|| "none".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let input_fields = input_schema
        .fields()
        .iter()
        .map(|field| format!("{}:{:?}:nullable={}", field.name(), field.data_type(), field.is_nullable()))
        .collect::<Vec<_>>()
        .join(",");
    let input_evidence = aggregate_input_evidence_structural_part(input_evidence);
    let signature = format!(
        "groups=[{}];aggr=[{}];input=[{}];evidence=[{}]",
        group_exprs, aggregate_exprs, input_fields, input_evidence
    );
    format!("agg-v1-fnv64-{:016x}", fnv1a64(signature.as_bytes()))
}

fn aggregate_function_structural_name(function: AggregateFunction) -> &'static str {
    function.as_str()
}

fn aggregate_expr_structural_part(expr: &Expr, input_schema: &ArrowSchema) -> String {
    match expr {
        Expr::Column(column) => {
            let dtype = input_schema
                .field_with_name(column.name())
                .map(|field| format!("{:?}:nullable={}", field.data_type(), field.is_nullable()))
                .unwrap_or_else(|_| "missing".to_string());
            format!("column:{}:{}", column.name(), dtype)
        },
        Expr::Alias { expr, .. } => aggregate_expr_structural_part(expr, input_schema),
        Expr::Literal(value) => format!("literal:{value:?}"),
        _ => format!("expr:{expr}"),
    }
}

fn aggregate_input_evidence_structural_part(input_evidence: Option<PartialAggregateInputEvidence>) -> String {
    let Some(input_evidence) = input_evidence else {
        return "unknown".to_string();
    };
    format!(
        "source_fingerprint={};source_adjacent={};row_groups={};reads_all={};rows={};bytes={};estimate_based={};partition_strategy={};partition_count={};non_empty_partitions={};partition_key_count={};max_partition_fan_in={}",
        input_evidence
            .source_chunk_fingerprint
            .map_or_else(|| "unknown".to_string(), |fingerprint| format!("{fingerprint:016x}")),
        input_evidence.source_adjacent_partial,
        input_evidence.source_chunk_row_group_count,
        input_evidence.source_chunk_reads_all_row_groups,
        optional_strategy_fact_u64(input_evidence.source_chunk_known_row_count),
        optional_strategy_fact_u64(input_evidence.source_chunk_estimated_source_read_bytes),
        input_evidence.source_chunk_size_estimate_based,
        input_evidence.source_adjacent_partition_strategy,
        optional_strategy_fact_u64(input_evidence.source_adjacent_partition_count),
        optional_strategy_fact_u64(input_evidence.source_adjacent_non_empty_partition_count),
        optional_strategy_fact_u64(input_evidence.source_adjacent_partition_key_count),
        optional_strategy_fact_u64(input_evidence.source_adjacent_max_partition_fan_in)
    )
}

pub(super) fn source_chunk_structural_part(chunk: &SourceChunk) -> String {
    format!(
        "source={};file_index={};stable_id={};location={};path={};read_columns={};partition_values={};row_groups={};reads_all={};source_file_row_groups={};known_rows={};estimated_source_read_bytes={};estimated_chunk_bytes={};known_row_group_bytes={};projected_uncompressed_bytes={};pure_filter_uncompressed_bytes={};scan_limit_rows={};zero_column_count={}",
        chunk.source_node_id,
        chunk.file_index,
        chunk.location.diagnostic_stable_id(chunk.stable_id.as_str()),
        chunk.location.diagnostic_location(),
        parquet_source_path_fact(&chunk.location, &chunk.path),
        chunk.read_columns.join(","),
        chunk.partition_values.iter().map(|value| format!("{value:?}")).collect::<Vec<_>>().join(","),
        format_usize_list(&chunk.row_group_indexes),
        chunk.reads_all_row_groups,
        chunk
            .source_file_row_groups
            .as_ref()
            .map(|row_groups| format_usize_list(row_groups))
            .unwrap_or_else(|| "unknown".to_string()),
        optional_strategy_fact_u64(chunk.known_row_count),
        optional_strategy_fact_u64(chunk.estimated_source_read_bytes),
        optional_strategy_fact_u64(chunk.estimated_chunk_bytes),
        optional_strategy_fact_u64(chunk.known_row_group_bytes),
        optional_strategy_fact_u64(chunk.known_projected_uncompressed_bytes),
        optional_strategy_fact_u64(chunk.known_pure_filter_uncompressed_bytes),
        chunk.scan_limit_rows.map_or_else(|| "none".to_string(), |rows| rows.to_string()),
        chunk.zero_column_count,
    )
}

fn format_usize_list(values: &[usize]) -> String {
    values.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(",")
}
