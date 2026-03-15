use std::collections::BTreeMap;

use super::*;
use crate::capability::aggregate_strategy_capability_decision;

pub(super) const ENCODED_KEY_PARTIAL_LOW_CARDINALITY_THRESHOLD_PPM: u64 = 100_000;
pub(super) const ENCODED_KEY_PARTIAL_MIN_INPUT_ROWS: u64 = 100_000;
pub(super) const EXACT_GROUPED_KEY_CARDINALITY_PROBE_MAX_KEY_COLUMNS: usize = 5;
pub(super) const EXACT_SINGLE_STRING_AVG_CARDINALITY_PROBE_MIN_INPUT_ROWS: u64 = 100_000;
pub(super) const EXACT_SINGLE_KEY_CARDINALITY_PROBE_MIN_INPUT_ROWS: u64 = 500_000;
pub(super) const EXACT_GROUPED_KEY_CARDINALITY_PROBE_MIN_INPUT_ROWS: u64 = 500_000;
pub(super) const EXACT_SINGLE_STRING_KEY_PREPARED_MAX_GROUPS: u64 = 4_096;
pub(super) const EXACT_THREE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS: u64 = 10_000_000;
pub(super) const EXACT_FIVE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS: u64 = 7_000_000;
pub(super) const PREPARED_SORT_GROUPBY_LOW_CARDINALITY_THRESHOLD_PPM: u64 = 100_000;
pub(super) const SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS: u64 = 64;
pub(super) const SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES: u64 = 384 * 1024 * 1024;
pub(super) const SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_FAN_IN: u64 = 64;

pub(super) fn select_partial_aggregate_strategy_decision(
    mut decision: AggregateStrategyCapabilityDecision,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::Schema,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    metrics: &QueryMetrics,
) -> AggregateStrategyCapabilityDecision {
    add_prepared_sort_groupby_gate_facts(&mut decision, input_rows);
    decision.facts.insert("encoded_key_partial_cost_model", "prior_partial_reduction_ratio".to_string());
    decision.facts.insert(
        "encoded_key_partial_reduction_ratio_threshold_ppm",
        ENCODED_KEY_PARTIAL_LOW_CARDINALITY_THRESHOLD_PPM.to_string(),
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
    let mut prepared_sort_groupby_runtime_candidate = prepared_sort_groupby_available;
    if let Some(evidence) = input_evidence
        && evidence.source_adjacent_partial
        && prepared_sort_groupby_available
    {
        if source_adjacent_prepared_sort_groupby_allowed(evidence) {
            return select_prepared_sort_groupby_for_source_adjacent(decision);
        }
        decision = reject_prepared_sort_groupby_for_cost_model(
            decision,
            "prepared_sort_groupby_requires_group_key_source_partition",
            "source_adjacent_row_balanced_partition",
        );
        prepared_sort_groupby_runtime_candidate = false;
    }

    if input_rows < ENCODED_KEY_PARTIAL_MIN_INPUT_ROWS {
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
        if encoded_key_partial_available {
            decision = reject_encoded_key_partial_for_cost_model(
                decision,
                "encoded_key_partial_cost_model_no_prior_cardinality_evidence",
                "runtime_cardinality_probe_required",
                None,
            );
        }
        return if prepared_sort_groupby_runtime_candidate {
            reject_prepared_sort_groupby_for_runtime_probe(decision)
        } else {
            decision
        };
    };
    add_prior_cardinality_facts(&mut decision, &evidence, spec, input_schema);
    if evidence.reduction_ratio_ppm <= PREPARED_SORT_GROUPBY_LOW_CARDINALITY_THRESHOLD_PPM
        && prepared_sort_groupby_runtime_candidate
        && prepared_sort_groupby_should_run_for_prior_evidence(&decision, &evidence)
    {
        return select_prepared_sort_groupby_for_prior_cardinality(decision);
    }
    if evidence.reduction_ratio_ppm <= ENCODED_KEY_PARTIAL_LOW_CARDINALITY_THRESHOLD_PPM
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

#[derive(Debug, Clone)]
struct PriorAggregateCardinalityEvidence {
    input_rows: u64,
    output_group_count: u64,
    reduction_ratio_ppm: u64,
    selected_strategy: String,
    evidence_scope: &'static str,
}

#[derive(Debug, Clone)]
pub(in crate::exec::operators::aggregate) struct ExactSingleKeyCardinalityProbe {
    pub(in crate::exec::operators::aggregate) key_column: String,
    pub(in crate::exec::operators::aggregate) input_rows: u64,
    pub(in crate::exec::operators::aggregate) non_null_distinct_count: u64,
    pub(in crate::exec::operators::aggregate) null_count: u64,
    pub(in crate::exec::operators::aggregate) num_groups: u64,
    pub(in crate::exec::operators::aggregate) reduction_ratio_ppm: u64,
    pub(in crate::exec::operators::aggregate) elapsed: Duration,
}

#[derive(Debug, Clone)]
pub(in crate::exec::operators::aggregate) struct ExactGroupedKeyCardinalityProbe {
    pub(in crate::exec::operators::aggregate) key_columns: Vec<String>,
    pub(in crate::exec::operators::aggregate) input_rows: u64,
    pub(in crate::exec::operators::aggregate) num_groups: u64,
    pub(in crate::exec::operators::aggregate) reduction_ratio_ppm: u64,
    pub(in crate::exec::operators::aggregate) elapsed: Duration,
}

fn prior_partial_cardinality_evidence(
    metrics: &QueryMetrics,
    node_id: Option<NodeId>,
    structural_key: &str,
) -> Option<PriorAggregateCardinalityEvidence> {
    if let Some(node_id) = node_id {
        let node_id = node_id.to_string();
        if let Some(evidence) = metrics.aggregate_strategy_decisions.iter().rev().find_map(|decision| {
            if decision.node_id != node_id
                || decision.facts.get("execution_stage").map(String::as_str) != Some("partial")
                || decision.facts.get("aggregate_structural_key").map(String::as_str) != Some(structural_key)
            {
                return None;
            }
            prior_partial_cardinality_evidence_from_decision(decision, "exact_node")
        }) {
            return Some(evidence);
        }
    }

    metrics.aggregate_strategy_decisions.iter().rev().find_map(|decision| {
        if decision.facts.get("execution_stage").map(String::as_str) != Some("partial")
            || decision.facts.get("aggregate_structural_key").map(String::as_str) != Some(structural_key)
        {
            return None;
        }
        prior_partial_cardinality_evidence_from_decision(decision, "structural_reused")
    })
}

fn prior_partial_cardinality_evidence_from_decision(
    decision: &crate::metrics::AggregateStrategyDecisionMetrics,
    evidence_scope: &'static str,
) -> Option<PriorAggregateCardinalityEvidence> {
    let input_rows = parse_strategy_fact_u64(decision.facts.get("input_rows")?)?;
    if input_rows == 0 {
        return None;
    }
    Some(PriorAggregateCardinalityEvidence {
        input_rows,
        output_group_count: parse_strategy_fact_u64(decision.facts.get("output_group_count")?)?,
        reduction_ratio_ppm: parse_strategy_fact_u64(decision.facts.get("reduction_ratio_ppm")?)?,
        selected_strategy: decision.facts.get("selected_strategy")?.clone(),
        evidence_scope,
    })
}

fn parse_strategy_fact_u64(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
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
    decision.facts.insert("exact_cardinality_probe_status", "not_attempted".to_string());
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

pub(super) fn exact_single_key_cardinality_probe_candidate<'a>(
    decision: &AggregateStrategyCapabilityDecision,
    spec: &'a AggregateSpec,
    input_schema: &arrow::datatypes::Schema,
    input_rows: u64,
) -> Option<&'a str> {
    if decision.selected_strategy != AggregateExecutionStrategy::LibcudfGroupby
        || decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str)
            != Some("selected_libcudf_runtime_probe")
        || !prepared_sort_groupby_contract_available(decision)
        || spec.group_exprs.len() != 1
    {
        return None;
    }
    let key_column = plain_group_key_column_name(&spec.group_exprs[0])?;
    input_schema.field_with_name(key_column).ok()?;
    if input_rows < exact_single_key_cardinality_probe_min_input_rows(decision, spec) {
        return None;
    }
    Some(key_column)
}

pub(super) fn exact_grouped_key_cardinality_probe_candidate(
    decision: &AggregateStrategyCapabilityDecision,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::Schema,
    input_rows: u64,
) -> Option<Vec<String>> {
    if decision.selected_strategy != AggregateExecutionStrategy::LibcudfGroupby
        || decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str)
            != Some("selected_libcudf_runtime_probe")
        || !prepared_sort_groupby_contract_available(decision)
        || !(2..=EXACT_GROUPED_KEY_CARDINALITY_PROBE_MAX_KEY_COLUMNS).contains(&spec.group_exprs.len())
        || input_rows < EXACT_GROUPED_KEY_CARDINALITY_PROBE_MIN_INPUT_ROWS
    {
        return None;
    }

    let mut key_columns = Vec::with_capacity(spec.group_exprs.len());
    let mut key_types = Vec::with_capacity(spec.group_exprs.len());
    for expr in &spec.group_exprs {
        let key_column = plain_group_key_column_name(expr)?;
        let field = input_schema.field_with_name(key_column).ok()?;
        if !exact_grouped_key_cardinality_probe_dtype_supported(field.data_type()) {
            return None;
        }
        key_columns.push(key_column.to_string());
        key_types.push(field.data_type());
    }
    if spec.group_exprs.len() > 2 && !extended_grouped_cardinality_probe_shape_supported(input_rows, &key_types) {
        return None;
    }
    Some(key_columns)
}

fn exact_single_key_cardinality_probe_min_input_rows(
    decision: &AggregateStrategyCapabilityDecision,
    spec: &AggregateSpec,
) -> u64 {
    let single_stage = decision.facts.get("execution_stage").map(String::as_str) == Some("single");
    let single_string_key = decision.facts.get("group_key_count").map(String::as_str) == Some("1")
        && decision.facts.get("string_group_key_count").map(String::as_str) == Some("1")
        && decision.facts.get("fixed_width_group_key_count").map(String::as_str) == Some("0");
    let avg_only =
        !spec.aggr_exprs.is_empty() && spec.aggr_exprs.iter().all(|expr| expr.function == AggregateFunction::Avg);
    if single_stage && single_string_key && avg_only {
        EXACT_SINGLE_STRING_AVG_CARDINALITY_PROBE_MIN_INPUT_ROWS
    } else {
        EXACT_SINGLE_KEY_CARDINALITY_PROBE_MIN_INPUT_ROWS
    }
}

pub(super) fn exact_single_key_cardinality_probe(
    frame: &DataFrame,
    key_column: &str,
    input_rows: u64,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<ExactSingleKeyCardinalityProbe> {
    let timer = Instant::now();
    let series = frame.series(key_column).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.input",
            "failed to access exact-cardinality probe key column",
            error,
        )
    })?;
    let null_count = series.null_count() as u64;
    let prepared = series
        .prepare_scalar_reduction(ReduceAgg::NUnique { null_policy: cudf::NullPolicy::Exclude })
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.prepare_nunique",
                "failed to prepare exact cardinality reduction for grouped aggregate key column",
                error,
            )
        })?;
    let scalar = budgeted_value(
        prepared.execute_with_budget(token).map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.nunique",
                "failed to compute exact cardinality for grouped aggregate key column",
                error,
            )
        })?,
        metrics,
        "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.nunique",
    )
    .into_scalar();
    let non_null_distinct_count = if scalar.is_valid().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.read_nunique",
            "failed to read exact cardinality scalar validity",
            error,
        )
    })? {
        let count = scalar.value::<i32>().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.read_nunique",
                "failed to read exact cardinality scalar value",
                error,
            )
        })?;
        u64::try_from(count).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.read_nunique",
                "exact cardinality scalar was negative",
            )
            .with_fact("distinct_count", count)
        })?
    } else {
        0
    };
    let num_groups = non_null_distinct_count.saturating_add(u64::from(null_count > 0));
    Ok(ExactSingleKeyCardinalityProbe {
        key_column: key_column.to_string(),
        input_rows,
        non_null_distinct_count,
        null_count,
        num_groups,
        reduction_ratio_ppm: reduction_ratio_ppm(input_rows, num_groups),
        elapsed: timer.elapsed(),
    })
}

pub(super) fn exact_grouped_key_cardinality_probe(
    frame: &DataFrame,
    key_columns: &[String],
    input_rows: u64,
    token: &ExecutionBudgetToken,
) -> Result<ExactGroupedKeyCardinalityProbe> {
    let timer = Instant::now();
    let key_refs = key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let keys = frame.select(key_refs.clone()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.exact_grouped_key_cardinality_probe.input",
            "failed to project exact-cardinality probe key columns",
            error,
        )
    })?;
    let overrides = token.execution_overrides();
    let groups = keys
        .with_execution(&overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_refs.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "nexus_query_engine.aggregate.exact_grouped_key_cardinality_probe.distinct_keys",
                    "failed to compute exact grouped-key cardinality for partial aggregate",
                    error,
                ),
                token,
            )
        })?;
    let num_groups = groups.num_rows() as u64;
    Ok(ExactGroupedKeyCardinalityProbe {
        key_columns: key_columns.to_vec(),
        input_rows,
        num_groups,
        reduction_ratio_ppm: reduction_ratio_ppm(input_rows, num_groups),
        elapsed: timer.elapsed(),
    })
}

pub(super) fn add_exact_single_key_cardinality_probe_facts(
    decision: &mut AggregateStrategyCapabilityDecision,
    probe: &ExactSingleKeyCardinalityProbe,
) {
    decision.facts.insert("exact_cardinality_probe_status", "measured".to_string());
    decision.facts.insert("exact_cardinality_probe_key", probe.key_column.clone());
    decision.facts.insert("exact_cardinality_probe_input_rows", probe.input_rows.to_string());
    decision.facts.insert("exact_cardinality_probe_non_null_distinct_count", probe.non_null_distinct_count.to_string());
    decision.facts.insert("exact_cardinality_probe_null_count", probe.null_count.to_string());
    decision.facts.insert("exact_cardinality_probe_num_groups", probe.num_groups.to_string());
    decision.facts.insert("exact_cardinality_probe_reduction_ratio_ppm", probe.reduction_ratio_ppm.to_string());
    decision.facts.insert("exact_cardinality_probe_elapsed_ns", duration_ns(probe.elapsed).to_string());
    decision.facts.insert("prepared_sort_groupby_num_groups", probe.num_groups.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_input_rows", probe.input_rows.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_output_group_count", probe.num_groups.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_reduction_ratio_ppm", probe.reduction_ratio_ppm.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_strategy", "exact_cardinality_probe".to_string());
}

pub(super) fn add_exact_grouped_key_cardinality_probe_facts(
    decision: &mut AggregateStrategyCapabilityDecision,
    probe: &ExactGroupedKeyCardinalityProbe,
) {
    decision.facts.insert("exact_cardinality_probe_status", "measured".to_string());
    decision.facts.insert("exact_cardinality_probe_key", probe.key_columns.join(","));
    decision.facts.insert("exact_cardinality_probe_input_rows", probe.input_rows.to_string());
    decision.facts.insert("exact_cardinality_probe_non_null_distinct_count", "unknown".to_string());
    decision.facts.insert("exact_cardinality_probe_null_count", "unknown".to_string());
    decision.facts.insert("exact_cardinality_probe_num_groups", probe.num_groups.to_string());
    decision.facts.insert("exact_cardinality_probe_reduction_ratio_ppm", probe.reduction_ratio_ppm.to_string());
    decision.facts.insert("exact_cardinality_probe_elapsed_ns", duration_ns(probe.elapsed).to_string());
    decision.facts.insert("prepared_sort_groupby_num_groups", probe.num_groups.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_input_rows", probe.input_rows.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_output_group_count", probe.num_groups.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_reduction_ratio_ppm", probe.reduction_ratio_ppm.to_string());
    decision.facts.insert("prepared_sort_groupby_prior_strategy", "exact_cardinality_probe".to_string());
}

pub(super) fn exact_cardinality_probe_should_select_prepared_sort_groupby(
    decision: &AggregateStrategyCapabilityDecision,
    probe: &ExactSingleKeyCardinalityProbe,
) -> bool {
    exact_cardinality_probe_values_should_select_prepared_sort_groupby(
        decision,
        probe.num_groups,
        probe.reduction_ratio_ppm,
    )
}

pub(super) fn exact_grouped_key_cardinality_probe_should_select_prepared_sort_groupby(
    decision: &AggregateStrategyCapabilityDecision,
    probe: &ExactGroupedKeyCardinalityProbe,
) -> bool {
    exact_cardinality_probe_values_should_select_prepared_sort_groupby(
        decision,
        probe.num_groups,
        probe.reduction_ratio_ppm,
    )
}

fn exact_cardinality_probe_values_should_select_prepared_sort_groupby(
    decision: &AggregateStrategyCapabilityDecision,
    num_groups: u64,
    reduction_ratio_ppm: u64,
) -> bool {
    let string_keys =
        decision.facts.get("string_group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    let fixed_width_keys =
        decision.facts.get("fixed_width_group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    let group_keys = decision.facts.get("group_key_count").and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);

    if group_keys == 1 && string_keys == 1 && fixed_width_keys == 0 {
        return num_groups <= EXACT_SINGLE_STRING_KEY_PREPARED_MAX_GROUPS;
    }
    reduction_ratio_ppm <= PREPARED_SORT_GROUPBY_LOW_CARDINALITY_THRESHOLD_PPM
}

fn source_adjacent_prepared_sort_groupby_allowed(evidence: PartialAggregateInputEvidence) -> bool {
    if evidence.source_adjacent_partition_strategy == "none" {
        if evidence.source_chunk_row_group_count <= SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS {
            return true;
        }
        return evidence
            .source_chunk_estimated_source_read_bytes
            .is_some_and(|bytes| bytes <= SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES);
    }
    if evidence.source_chunk_row_group_count > SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS {
        return false;
    }
    evidence.source_adjacent_partition_strategy == "group_key"
        && evidence
            .source_adjacent_max_partition_fan_in
            .is_some_and(|fan_in| fan_in <= SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_FAN_IN)
}

fn select_prepared_sort_groupby_for_source_adjacent(
    mut decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    decision.selected_strategy = AggregateExecutionStrategy::PreparedSortGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::PreparedSortGroupby.as_str().to_string());
    decision.facts.insert("prepared_sort_groupby_rejected_reason", "none".to_string());
    decision.facts.insert("prepared_sort_groupby_cost_model_status", "selected_source_adjacent_execution".to_string());
    decision
        .facts
        .insert("prepared_sort_groupby_selection_reason", "source_adjacent_partial_prepared_execution".to_string());
    decision.facts.insert(
        "encoded_key_partial_rejected_reason",
        "encoded_key_partial_superseded_by_prepared_sort_groupby".to_string(),
    );
    decision.facts.insert("encoded_key_partial_cost_model_status", "not_selected".to_string());
    decision.facts.insert("encoded_key_partial_selection_reason", "prepared_sort_groupby_selected".to_string());
    decision
        .rejected_candidates
        .retain(|rejection| rejection.strategy != AggregateExecutionStrategy::PreparedSortGroupby);
    decision.rejected_candidates.insert(
        0,
        AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::EncodedKeyPartial,
            reason: "encoded_key_partial_superseded_by_prepared_sort_groupby",
        },
    );
    update_rejected_candidate_summary(&mut decision);
    decision
}

fn select_prepared_sort_groupby_for_prior_cardinality(
    mut decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    decision.selected_strategy = AggregateExecutionStrategy::PreparedSortGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::PreparedSortGroupby.as_str().to_string());
    decision.facts.insert("prepared_sort_groupby_rejected_reason", "none".to_string());
    decision.facts.insert("prepared_sort_groupby_cost_model_status", "selected_prior_low_cardinality".to_string());
    decision
        .facts
        .insert("prepared_sort_groupby_selection_reason", "prior_reduction_ratio_at_or_below_threshold".to_string());
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

pub(super) fn select_prepared_sort_groupby_for_exact_cardinality(
    mut decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    decision.selected_strategy = AggregateExecutionStrategy::PreparedSortGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::PreparedSortGroupby.as_str().to_string());
    decision.facts.insert("prepared_sort_groupby_rejected_reason", "none".to_string());
    decision
        .facts
        .insert("prepared_sort_groupby_cost_model_status", "selected_exact_cardinality_low_cardinality".to_string());
    decision.facts.insert(
        "prepared_sort_groupby_selection_reason",
        "exact_cardinality_reduction_ratio_at_or_below_threshold".to_string(),
    );
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

pub(super) fn reject_prepared_sort_groupby_for_exact_cardinality(
    mut decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    if !prepared_sort_groupby_contract_available(&decision) {
        return decision;
    }
    decision.selected_strategy = AggregateExecutionStrategy::LibcudfGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::LibcudfGroupby.as_str().to_string());
    decision.facts.insert(
        "prepared_sort_groupby_rejected_reason",
        "prepared_sort_groupby_exact_cardinality_high_cardinality".to_string(),
    );
    decision.facts.insert("prepared_sort_groupby_cost_model_status", "rejected_by_cost_model".to_string());
    decision.facts.insert(
        "prepared_sort_groupby_selection_reason",
        "exact_cardinality_reduction_ratio_above_threshold".to_string(),
    );
    if let Some(rejection) = decision
        .rejected_candidates
        .iter_mut()
        .find(|rejection| rejection.strategy == AggregateExecutionStrategy::PreparedSortGroupby)
    {
        rejection.reason = "prepared_sort_groupby_exact_cardinality_high_cardinality";
    } else {
        decision.rejected_candidates.push(AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::PreparedSortGroupby,
            reason: "prepared_sort_groupby_exact_cardinality_high_cardinality",
        });
    }
    update_rejected_candidate_summary(&mut decision);
    decision
}

fn reject_prepared_sort_groupby_for_runtime_probe(
    mut decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    if !prepared_sort_groupby_contract_available(&decision) {
        return decision;
    }
    decision.selected_strategy = AggregateExecutionStrategy::LibcudfGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::LibcudfGroupby.as_str().to_string());
    decision
        .facts
        .insert("prepared_sort_groupby_rejected_reason", "prepared_sort_groupby_runtime_probe_required".to_string());
    decision.facts.insert("prepared_sort_groupby_cost_model_status", "selected_libcudf_runtime_probe".to_string());
    decision.facts.insert("prepared_sort_groupby_selection_reason", "runtime_cardinality_probe_required".to_string());
    if let Some(rejection) = decision
        .rejected_candidates
        .iter_mut()
        .find(|rejection| rejection.strategy == AggregateExecutionStrategy::PreparedSortGroupby)
    {
        rejection.reason = "prepared_sort_groupby_runtime_probe_required";
    } else {
        decision.rejected_candidates.push(AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::PreparedSortGroupby,
            reason: "prepared_sort_groupby_runtime_probe_required",
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
    input_schema: &arrow::datatypes::Schema,
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
    input_schema: &arrow::datatypes::Schema,
    output_group_count: u64,
) -> Option<u64> {
    let mut bytes_per_group = 0u64;
    for expr in &spec.group_exprs {
        let field = input_schema.field_with_name(plain_group_key_column_name(expr)?).ok()?;
        bytes_per_group = bytes_per_group.checked_add(fixed_width_dtype_size(field.data_type())?)?;
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

fn fixed_width_dtype_size(data_type: &ArrowDataType) -> Option<u64> {
    match data_type {
        ArrowDataType::Boolean => Some(1),
        ArrowDataType::Int8 | ArrowDataType::UInt8 => Some(1),
        ArrowDataType::Int16 | ArrowDataType::UInt16 => Some(2),
        ArrowDataType::Int32
        | ArrowDataType::UInt32
        | ArrowDataType::Float32
        | ArrowDataType::Date32
        | ArrowDataType::Time32(_) => Some(4),
        ArrowDataType::Int64
        | ArrowDataType::UInt64
        | ArrowDataType::Float64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Duration(_) => Some(8),
        ArrowDataType::Decimal128(_, _) => Some(16),
        ArrowDataType::Decimal256(_, _) => Some(32),
        _ => None,
    }
}

fn exact_grouped_key_cardinality_probe_dtype_supported(data_type: &ArrowDataType) -> bool {
    fixed_width_dtype_size(data_type).is_some()
        || matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

fn extended_grouped_cardinality_probe_shape_supported(input_rows: u64, key_types: &[&ArrowDataType]) -> bool {
    let string_key_count = key_types
        .iter()
        .filter(|data_type| {
            matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
        })
        .count();
    let fixed_width_key_count =
        key_types.iter().filter(|data_type| fixed_width_dtype_size(data_type).is_some()).count();
    match key_types.len() {
        3 => {
            input_rows <= EXACT_THREE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS
                && string_key_count == 1
                && fixed_width_key_count == 2
        },
        5 => {
            input_rows <= EXACT_FIVE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS
                && string_key_count == 3
                && fixed_width_key_count == 2
        },
        _ => false,
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

pub(in crate::exec) fn record_default_grouped_aggregate_strategy_decision(
    metrics: &mut QueryMetrics,
    node_id: Option<String>,
    stage: AggregateExecutionStage,
    memory_mode: &'static str,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::SchemaRef,
    input_rows: u64,
    output_rows: u64,
    elapsed: Duration,
) {
    let decision = aggregate_strategy_capability_decision(spec, input_schema.as_ref(), stage);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id,
        stage,
        memory_mode,
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

#[allow(clippy::too_many_arguments)]
pub(in crate::exec) fn record_grouped_aggregate_strategy_decision(
    metrics: &mut QueryMetrics,
    node_id: Option<String>,
    stage: AggregateExecutionStage,
    memory_mode: &'static str,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::SchemaRef,
    input_rows: u64,
    output_rows: u64,
    decision: crate::capability::AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    encode_elapsed: Duration,
    groupby_elapsed: Duration,
    decode_elapsed: Duration,
    strategy_elapsed: Duration,
    input_evidence: Option<PartialAggregateInputEvidence>,
) {
    let mut facts = decision.facts;
    add_partial_aggregate_input_evidence_facts(&mut facts, input_evidence);
    facts
        .entry("aggregate_structural_key")
        .or_insert_with(|| aggregate_structural_key(spec, input_schema.as_ref(), input_evidence));
    facts.insert("memory_mode", memory_mode.to_string());
    facts.insert("input_rows", input_rows.to_string());
    facts.insert("chunk_row_count", if stage == AggregateExecutionStage::Partial { input_rows } else { 0 }.to_string());
    facts.insert("chunk_count", metrics.source_chunks_planned.to_string());
    facts.insert("partial_aggregate_count", metrics.partial_aggregate_count.to_string());
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

pub(super) fn reduction_ratio_ppm(input_rows: u64, output_rows: u64) -> u64 {
    if input_rows == 0 {
        return 0;
    }
    output_rows.saturating_mul(1_000_000).saturating_div(input_rows)
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

pub(super) fn aggregate_structural_key(
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::Schema,
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
    format!("agg-v1-fnv64-{:016x}", stable_fnv1a64(signature.as_bytes()))
}

pub(super) fn stable_fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn aggregate_function_structural_name(function: AggregateFunction) -> &'static str {
    function.as_str()
}

fn aggregate_expr_structural_part(expr: &Expr, input_schema: &arrow::datatypes::Schema) -> String {
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
        source_chunk_path_fact(chunk),
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

fn source_chunk_path_fact(chunk: &SourceChunk) -> String {
    if chunk.location.is_local_path() {
        crate::source::redacted_path_diagnostic(&chunk.path)
    } else {
        chunk.location.diagnostic_location()
    }
}
