use std::{
    collections::{BTreeMap, hash_map::Entry},
    sync::Arc,
};

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
        StringArray, StringViewArray, UInt64Array,
    },
    datatypes::SchemaRef as ArrowSchemaRef,
};
use cudf::hash::{FastHashMap, fast_hash_map_with_capacity};

use super::{
    add_floating_statistical_aggregate_contract_facts,
    partial_strategy::{
        add_aggregate_strategy_execution_option_facts, add_partial_aggregate_input_evidence_facts, reduction_ratio_ppm,
    },
    *,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct HostEncodedKeyPart {
    bits: u64,
    is_null: bool,
}

impl HostEncodedKeyPart {
    fn null() -> Self {
        Self { bits: 0, is_null: true }
    }

    fn value(bits: u64) -> Self {
        Self { bits, is_null: false }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum HostGroupLookupKey {
    Empty,
    One(HostEncodedKeyPart),
    Two([HostEncodedKeyPart; 2]),
    Three([HostEncodedKeyPart; 3]),
    Four([HostEncodedKeyPart; 4]),
    Many(Vec<HostEncodedKeyPart>),
}

impl HostGroupLookupKey {
    fn for_row(columns: &[HostKeyColumn<'_>], row: usize) -> Self {
        match columns {
            [] => Self::Empty,
            [c0] => Self::One(c0.encoded_part(row)),
            [c0, c1] => Self::Two([c0.encoded_part(row), c1.encoded_part(row)]),
            [c0, c1, c2] => Self::Three([c0.encoded_part(row), c1.encoded_part(row), c2.encoded_part(row)]),
            [c0, c1, c2, c3] => {
                Self::Four([c0.encoded_part(row), c1.encoded_part(row), c2.encoded_part(row), c3.encoded_part(row)])
            },
            _ => Self::Many(columns.iter().map(|column| column.encoded_part(row)).collect()),
        }
    }

    fn part(&self, index: usize) -> Option<HostEncodedKeyPart> {
        match self {
            Self::Empty => None,
            Self::One(part) => (index == 0).then_some(*part),
            Self::Two(parts) => parts.get(index).copied(),
            Self::Three(parts) => parts.get(index).copied(),
            Self::Four(parts) => parts.get(index).copied(),
            Self::Many(parts) => parts.get(index).copied(),
        }
    }
}

#[derive(Debug, Clone)]
enum HostAggregateState {
    Count { count: u64 },
    Sum { sum: f64, seen: bool },
    Min { value: f64, seen: bool },
    Max { value: f64, seen: bool },
    Avg { sum: f64, count: u64 },
    FloatingStatistical { function: AggregateFunction, count: u64, mean: f64, m2: f64 },
}

impl HostAggregateState {
    fn new(function: AggregateFunction) -> Self {
        match function {
            AggregateFunction::Count | AggregateFunction::NUnique => Self::Count { count: 0 },
            AggregateFunction::Sum => Self::Sum { sum: 0.0, seen: false },
            AggregateFunction::Min => Self::Min { value: 0.0, seen: false },
            AggregateFunction::Max => Self::Max { value: 0.0, seen: false },
            AggregateFunction::Avg => Self::Avg { sum: 0.0, count: 0 },
            AggregateFunction::Median => unreachable!("median is not executed by the host-exact aggregate path"),
            function @ (AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarSamp
            | AggregateFunction::VarPop) => Self::FloatingStatistical { function, count: 0, mean: 0.0, m2: 0.0 },
        }
    }

    fn update(&mut self, value: Option<f64>) {
        match self {
            Self::Count { count } => {
                if value.is_some() {
                    *count += 1;
                }
            },
            Self::Sum { sum, seen } => {
                if let Some(value) = value {
                    *sum += value;
                    *seen = true;
                }
            },
            Self::Min { value: current, seen } => {
                if let Some(value) = value {
                    if !*seen || value < *current {
                        *current = value;
                    }
                    *seen = true;
                }
            },
            Self::Max { value: current, seen } => {
                if let Some(value) = value {
                    if !*seen || value > *current {
                        *current = value;
                    }
                    *seen = true;
                }
            },
            Self::Avg { sum, count } => {
                if let Some(value) = value {
                    *sum += value;
                    *count += 1;
                }
            },
            Self::FloatingStatistical { count, mean, m2, .. } => {
                if let Some(value) = value {
                    let new_count = *count + 1;
                    let delta1 = value - *mean;
                    let new_mean = delta1 / new_count as f64 + *mean;
                    let delta2 = value - new_mean;
                    *m2 += delta1 * delta2;
                    *mean = new_mean;
                    *count = new_count;
                }
            },
        }
    }

    fn evaluate_f64(&self) -> Option<f64> {
        match self {
            Self::Count { count } => Some(*count as f64),
            Self::Sum { sum, seen } => seen.then_some(*sum),
            Self::Min { value, seen } | Self::Max { value, seen } => seen.then_some(*value),
            Self::Avg { sum, count } => (*count > 0).then_some(*sum / *count as f64),
            Self::FloatingStatistical { function, count, m2, .. } => {
                let ddof = function.statistical_ddof().expect("statistical state must carry ddof") as u64;
                if *count <= ddof {
                    None
                } else {
                    let variance = *m2 / (*count - ddof) as f64;
                    match function {
                        AggregateFunction::StddevSamp | AggregateFunction::StddevPop => Some(variance.sqrt()),
                        AggregateFunction::VarSamp | AggregateFunction::VarPop => Some(variance),
                        _ => unreachable!("statistical state must carry a statistical aggregate function"),
                    }
                }
            },
        }
    }

    fn evaluate_i64(&self) -> Option<i64> {
        match self {
            Self::Count { count } => Some(*count as i64),
            _ => self.evaluate_f64().map(|value| value as i64),
        }
    }
}

struct HostAggregateGroup {
    key: HostGroupLookupKey,
    aggregates: Vec<HostAggregateState>,
}

pub(super) fn execute_datafusion_exact_host_aggregate(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    metrics: &mut QueryMetrics,
    extra_strategy_facts: Option<&BTreeMap<&'static str, String>>,
) -> Result<GpuDataFrame> {
    let timer = Instant::now();
    let (frame, schema) = prepare_datafusion_exact_host_aggregate_input_frame(context, input, input_schema, metrics)?;
    let batch = materialize_datafusion_exact_host_aggregate_input(context, frame, schema)?;
    let output_batch = execute_datafusion_exact_host_aggregate_batch(
        context,
        node_id,
        spec,
        &batch,
        output_schema,
        metrics,
        extra_strategy_facts,
    )?;
    let output = upload_datafusion_exact_host_aggregate_output(context, &output_batch)?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn prepare_datafusion_exact_host_aggregate_input_frame(
    context: &ExecutionContext,
    input: GpuDataFrame,
    input_schema: &ArrowSchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<(GpuDataFrame, ArrowSchemaRef)> {
    let frame = sort_by_datafusion_order_if_present(context, input.into_frame(), metrics)?;
    let schema = schema_with_datafusion_order_columns(&frame, input_schema);
    Ok((GpuDataFrame::new(frame), schema))
}

pub(in crate::exec) fn materialize_datafusion_exact_host_aggregate_input(
    context: &ExecutionContext,
    input: GpuDataFrame,
    schema: ArrowSchemaRef,
) -> Result<RecordBatch> {
    let handle = GpuResultHandle::new(context, input, schema)?;
    handle.to_host_record_batch()
}

pub(in crate::exec) fn execute_datafusion_exact_host_aggregate_batch(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    batch: &RecordBatch,
    output_schema: &ArrowSchemaRef,
    metrics: &mut QueryMetrics,
    extra_strategy_facts: Option<&BTreeMap<&'static str, String>>,
) -> Result<RecordBatch> {
    if spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::Median) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.aggregate.floating_statistical_exact.median",
            "DataFusion-exact statistical aggregate path does not support median aggregate expressions",
        ));
    }
    let key_columns = spec
        .group_exprs
        .iter()
        .map(|expr| lower_named_column_expr(expr, "nexus_query_engine.aggregate.floating_statistical_exact.lower_key"))
        .collect::<Result<Vec<_>>>()?;
    let aggregate_columns = spec.aggr_exprs.iter().map(aggregate_input_column).collect::<Result<Vec<_>>>()?;
    let key_accessors =
        key_columns.iter().map(|(input_name, _)| HostKeyColumn::new(batch, input_name)).collect::<Result<Vec<_>>>()?;
    let mut aggregate_input_slots = Vec::<usize>::with_capacity(aggregate_columns.len());
    let mut unique_aggregate_inputs = Vec::<HostAggregateInput>::new();
    for input in aggregate_columns {
        let slot = unique_aggregate_inputs.iter().position(|existing| existing == &input).unwrap_or_else(|| {
            let slot = unique_aggregate_inputs.len();
            unique_aggregate_inputs.push(input);
            slot
        });
        aggregate_input_slots.push(slot);
    }
    let aggregate_accessors = unique_aggregate_inputs
        .iter()
        .map(|input| HostAggregateInputAccessor::new(batch, input))
        .collect::<Result<Vec<_>>>()?;
    let row_count = batch.num_rows();
    let mut group_indices = fast_hash_map_with_capacity::<HostGroupLookupKey, usize>(row_count.min(1_048_576));
    let mut groups = Vec::<HostAggregateGroup>::new();
    let mut aggregate_input_values = vec![None; aggregate_accessors.len()];

    for row in 0..row_count {
        let key = HostGroupLookupKey::for_row(&key_accessors, row);
        let group_index = match group_indices.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = groups.len();
                let key = entry.key().clone();
                entry.insert(index);
                groups.push(HostAggregateGroup {
                    key,
                    aggregates: spec.aggr_exprs.iter().map(|expr| HostAggregateState::new(expr.function)).collect(),
                });
                index
            },
        };
        for (value, accessor) in aggregate_input_values.iter_mut().zip(&aggregate_accessors) {
            *value = accessor.value(row);
        }
        for (aggregate_index, expr) in spec.aggr_exprs.iter().enumerate() {
            let value = aggregate_input_values[aggregate_input_slots[aggregate_index]];
            let state = groups[group_index]
                .aggregates
                .get_mut(aggregate_index)
                .expect("aggregate state index should match aggregate expression index");
            if matches!(expr.function, AggregateFunction::Count | AggregateFunction::NUnique) {
                state.update(value.map(|_| 1.0));
            } else {
                state.update(value);
            }
        }
    }

    if key_columns.is_empty() && groups.is_empty() {
        groups.push(HostAggregateGroup {
            key: HostGroupLookupKey::Empty,
            aggregates: spec.aggr_exprs.iter().map(|expr| HostAggregateState::new(expr.function)).collect(),
        });
    }

    let mut arrays = Vec::<ArrayRef>::with_capacity(output_schema.fields().len());
    for (output_index, field) in output_schema.fields().iter().enumerate() {
        if output_index < key_columns.len() {
            arrays.push(host_key_output_array(&groups, output_index, field.data_type(), &key_accessors[output_index])?);
        } else {
            let aggregate_index = output_index - key_columns.len();
            arrays.push(host_aggregate_output_array(&groups, aggregate_index, field.data_type())?);
        }
    }
    record_datafusion_exact_floating_statistical_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        context.memory_control_policy().as_str(),
        context.aggregate_strategy_execution_options(),
        spec,
        batch.schema().as_ref(),
        row_count as u64,
        groups.len() as u64,
        extra_strategy_facts,
    );
    let output = RecordBatch::try_new(Arc::clone(output_schema), arrays).map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.aggregate.floating_statistical_exact.output",
            "failed to construct DataFusion-exact stddev aggregate Arrow output",
        )
        .with_source(error)
    })?;
    Ok(output)
}

pub(in crate::exec) fn upload_datafusion_exact_host_aggregate_output(
    context: &ExecutionContext,
    output: &RecordBatch,
) -> Result<DataFrame> {
    DataFrame::from_record_batch(&context.runtime()?, output).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.floating_statistical_exact.output",
            "failed to upload DataFusion-exact stddev aggregate output",
            error,
        )
    })
}

fn record_datafusion_exact_floating_statistical_aggregate_strategy_decision(
    metrics: &mut QueryMetrics,
    node_id: Option<String>,
    memory_mode: &'static str,
    aggregate_strategy_options: AggregateStrategyExecutionOptions,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::Schema,
    input_rows: u64,
    output_rows: u64,
    extra_strategy_facts: Option<&BTreeMap<&'static str, String>>,
) {
    let mut facts = BTreeMap::from([
        ("selected_strategy", "datafusion_exact_host_welford".to_string()),
        ("execution_stage", AggregateExecutionStage::Single.as_str().to_string()),
        ("memory_mode", memory_mode.to_string()),
        ("group_key_count", spec.group_exprs.len().to_string()),
        ("group_key_exprs", aggregate_group_key_exprs_for_metrics(spec)),
        ("group_key_dtypes", aggregate_group_key_dtypes_for_metrics(spec, input_schema)),
        ("aggregate_functions", aggregate_function_list_for_metrics(spec)),
        ("aggregate_filter_count", spec.aggr_exprs.iter().filter(|expr| expr.filter.is_some()).count().to_string()),
        ("distinct_aggregate_count", spec.aggr_exprs.iter().filter(|expr| expr.distinct).count().to_string()),
        ("input_rows", input_rows.to_string()),
        ("chunk_row_count", "0".to_string()),
        ("chunk_count", metrics.source_chunks_planned.to_string()),
        ("partial_aggregate_count", metrics.partial_aggregate_count.to_string()),
        ("output_group_count", output_rows.to_string()),
        ("reduction_ratio_ppm", reduction_ratio_ppm(input_rows, output_rows).to_string()),
        ("source_adjacent_partial", "unknown".to_string()),
        ("source_chunk_row_group_count", "unknown".to_string()),
        ("source_chunk_reads_all_row_groups", "unknown".to_string()),
        ("source_chunk_known_row_count", "unknown".to_string()),
        ("source_chunk_estimated_source_read_bytes", "unknown".to_string()),
        ("source_chunk_size_estimate_based", "unknown".to_string()),
        ("source_adjacent_partition_strategy", "unknown".to_string()),
        ("source_adjacent_partition_count", "unknown".to_string()),
        ("source_adjacent_non_empty_partition_count", "unknown".to_string()),
        ("source_adjacent_partition_key_count", "unknown".to_string()),
        ("source_adjacent_max_partition_fan_in", "unknown".to_string()),
        ("partitioned_merge_partition_count", "none".to_string()),
        ("partitioned_merge_non_empty_partition_count", "none".to_string()),
        ("partitioned_merge_max_partition_fan_in", "none".to_string()),
        ("selection_elapsed_ns", "0".to_string()),
        ("encode_elapsed_ns", "0".to_string()),
        ("groupby_elapsed_ns", "0".to_string()),
        ("decode_elapsed_ns", "0".to_string()),
        ("strategy_operator_elapsed_ns", "0".to_string()),
        ("encoded_key_partial_cost_model", "none".to_string()),
        ("encoded_key_partial_cost_model_status", "not_applicable".to_string()),
        ("encoded_key_partial_selection_reason", "floating_statistical_exact_host_path".to_string()),
        ("encoded_key_partial_reduction_ratio_threshold_ppm", "0".to_string()),
        ("encoded_key_partial_prior_reduction_ratio_ppm", "0".to_string()),
        ("encoded_key_partial_prior_strategy", "none".to_string()),
        (
            "prepared_sort_groupby_rejected_reason",
            "prepared_sort_groupby_floating_statistical_aggregate_unsupported".to_string(),
        ),
        ("prepared_sort_groupby_contract_status", "blocked_by_shape_or_stage".to_string()),
        ("prepared_sort_groupby_required_contract", "prepared_sort_groupby_execution_contract".to_string()),
        ("prepared_sort_groupby_cost_model", "exact_grouped_key_facts_required".to_string()),
        ("prepared_sort_groupby_cost_model_status", "not_applicable".to_string()),
        ("prepared_sort_groupby_selection_reason", "capability_rejected".to_string()),
        ("prepared_sort_groupby_num_groups", "unknown".to_string()),
        ("prepared_sort_groupby_grouped_key_output_bytes", "unknown".to_string()),
        ("prepared_sort_groupby_max_group_size", "unknown".to_string()),
        (
            "rejected_candidate_strategies",
            "prepared_sort_groupby:prepared_sort_groupby_floating_statistical_aggregate_unsupported".to_string(),
        ),
    ]);
    add_floating_statistical_aggregate_contract_facts(
        &mut facts,
        spec,
        StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
    );
    add_aggregate_strategy_execution_option_facts(&mut facts, aggregate_strategy_options);
    if let Some(extra_strategy_facts) = extra_strategy_facts {
        add_filtered_materialization_facts(&mut facts, extra_strategy_facts);
    }
    add_partial_aggregate_input_evidence_facts(&mut facts, None);
    metrics.record_aggregate_strategy_decision(
        node_id.unwrap_or_else(|| "aggregate.datafusion_exact_floating_statistical_aggregate".to_string()),
        &facts,
    );
}

fn add_filtered_materialization_facts(
    facts: &mut BTreeMap<&'static str, String>,
    extra_strategy_facts: &BTreeMap<&'static str, String>,
) {
    for key in [
        "aggregate_filter_count",
        "filtered_aggregate_scope",
        "filtered_input_materialization_strategy",
        "filtered_input_materialization_estimated_bytes",
        "filtered_input_materialization_budget_bytes",
        "filtered_input_materialization_threshold_bytes",
        "filtered_input_materialization_budget_fraction",
        "filtered_input_value_bytes",
        "filtered_input_validity_bytes",
        "filtered_input_predicate_bytes",
        "filtered_input_original_frame_retained_bytes",
        "filtered_input_unsupported_stateful_merge",
        "filtered_input_unique_predicate_count",
        "filtered_global_strategy_input_rows",
        "filtered_global_rows_per_chunk",
        "filtered_global_inputs_per_segment",
        "filtered_global_segment_target_temporary_bytes",
    ] {
        if let Some(value) = extra_strategy_facts.get(key) {
            facts.insert(key, value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_materialization_facts_preserve_original_filter_count_for_exact_host_rows() {
        let mut facts = BTreeMap::from([
            ("aggregate_filter_count", "0".to_string()),
            ("selected_strategy", "datafusion_exact_host_welford".to_string()),
        ]);
        let extra_strategy_facts = BTreeMap::from([
            ("aggregate_filter_count", "1".to_string()),
            ("filtered_input_materialization_strategy", "single_pass_with_columns".to_string()),
            ("filtered_input_unsupported_stateful_merge", "true".to_string()),
        ]);

        add_filtered_materialization_facts(&mut facts, &extra_strategy_facts);

        assert_eq!(facts.get("aggregate_filter_count").map(String::as_str), Some("1"));
        assert_eq!(
            facts.get("filtered_input_materialization_strategy").map(String::as_str),
            Some("single_pass_with_columns")
        );
        assert_eq!(facts.get("filtered_input_unsupported_stateful_merge").map(String::as_str), Some("true"));
    }
}

fn aggregate_group_key_exprs_for_metrics(spec: &AggregateSpec) -> String {
    if spec.group_exprs.is_empty() {
        "none".to_string()
    } else {
        spec.group_exprs.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    }
}

fn aggregate_group_key_dtypes_for_metrics(spec: &AggregateSpec, input_schema: &arrow::datatypes::Schema) -> String {
    if spec.group_exprs.is_empty() {
        return "none".to_string();
    }
    spec.group_exprs
        .iter()
        .map(|expr| {
            match lower_named_column_expr(expr, "nexus_query_engine.aggregate.floating_statistical_exact.metrics") {
                Ok((name, _)) => input_schema
                    .field_with_name(name.as_str())
                    .map(|field| format!("{:?}", field.data_type()))
                    .unwrap_or_else(|_| "missing_column".to_string()),
                Err(_) => "non_column_expr".to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn aggregate_function_list_for_metrics(spec: &AggregateSpec) -> String {
    if spec.aggr_exprs.is_empty() {
        return "none".to_string();
    }
    spec.aggr_exprs
        .iter()
        .map(|expr| {
            let label = match expr.function {
                AggregateFunction::Sum => "sum",
                AggregateFunction::Min => "min",
                AggregateFunction::Max => "max",
                AggregateFunction::Count => "count",
                AggregateFunction::NUnique => "nunique",
                AggregateFunction::Avg => "avg",
                function => function.as_str(),
            };
            if expr.distinct { format!("{label}_distinct") } else { label.to_string() }
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn host_group_expr_output_name(expr: &Expr) -> String {
    match expr {
        Expr::Alias { alias, .. } => alias.clone(),
        Expr::Column(column) => column.display_name().to_string(),
        _ => expr.to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum HostAggregateInput {
    Column(String),
    NonNullLiteral,
    NullLiteral,
}

enum HostAggregateInputAccessor<'a> {
    Column(HostNumericColumn<'a>),
    NonNullLiteral,
    NullLiteral,
}

impl<'a> HostAggregateInputAccessor<'a> {
    fn new(batch: &'a RecordBatch, input: &HostAggregateInput) -> Result<Self> {
        match input {
            HostAggregateInput::Column(name) => HostNumericColumn::new(batch, name).map(Self::Column),
            HostAggregateInput::NonNullLiteral => Ok(Self::NonNullLiteral),
            HostAggregateInput::NullLiteral => Ok(Self::NullLiteral),
        }
    }

    fn value(&self, row: usize) -> Option<f64> {
        match self {
            Self::Column(column) => column.value(row),
            Self::NonNullLiteral => Some(1.0),
            Self::NullLiteral => None,
        }
    }
}

enum HostKeyColumn<'a> {
    Boolean(&'a BooleanArray),
    Int32(&'a Int32Array),
    Date32(&'a Date32Array),
    Int64(&'a Int64Array),
    UInt64(&'a UInt64Array),
    Float64(&'a Float64Array),
    Utf8(HostStringKeyColumn),
    LargeUtf8(HostStringKeyColumn),
    Utf8View(HostStringKeyColumn),
}

struct HostStringKeyColumn {
    row_codes: Vec<Option<u64>>,
    values: Vec<String>,
}

impl HostStringKeyColumn {
    fn from_utf8(array: &StringArray) -> Self {
        encode_host_string_keys(array.len(), |row| (!array.is_null(row)).then(|| array.value(row)))
    }

    fn from_large_utf8(array: &LargeStringArray) -> Self {
        encode_host_string_keys(array.len(), |row| (!array.is_null(row)).then(|| array.value(row)))
    }

    fn from_utf8_view(array: &StringViewArray) -> Self {
        encode_host_string_keys(array.len(), |row| (!array.is_null(row)).then(|| array.value(row)))
    }

    fn encoded_part(&self, row: usize) -> HostEncodedKeyPart {
        match self.row_codes[row] {
            Some(code) => HostEncodedKeyPart::value(code),
            None => HostEncodedKeyPart::null(),
        }
    }

    fn value(&self, part: HostEncodedKeyPart) -> Option<&str> {
        if part.is_null { None } else { self.values.get(part.bits as usize).map(String::as_str) }
    }
}

fn encode_host_string_keys<'a>(row_count: usize, value_at: impl Fn(usize) -> Option<&'a str>) -> HostStringKeyColumn {
    let mut interned = FastHashMap::<&'a str, u64>::default();
    let mut values = Vec::<String>::new();
    let mut row_codes = Vec::<Option<u64>>::with_capacity(row_count);
    for row in 0..row_count {
        let Some(value) = value_at(row) else {
            row_codes.push(None);
            continue;
        };
        let code = match interned.entry(value) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let code = values.len() as u64;
                entry.insert(code);
                values.push(value.to_string());
                code
            },
        };
        row_codes.push(Some(code));
    }
    HostStringKeyColumn { row_codes, values }
}

impl<'a> HostKeyColumn<'a> {
    fn new(batch: &'a RecordBatch, column: &str) -> Result<Self> {
        let array = batch.column(batch.schema().index_of(column).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.aggregate.floating_statistical_exact.key",
                "group key column is missing from aggregate input batch",
            )
            .with_fact("column", column)
            .with_source(error)
        })?);
        match array.data_type() {
            ArrowDataType::Boolean => {
                Ok(Self::Boolean(array.as_any().downcast_ref::<BooleanArray>().expect("validated BooleanArray")))
            },
            ArrowDataType::Int32 => {
                Ok(Self::Int32(array.as_any().downcast_ref::<Int32Array>().expect("validated Int32Array")))
            },
            ArrowDataType::Date32 => {
                Ok(Self::Date32(array.as_any().downcast_ref::<Date32Array>().expect("validated Date32Array")))
            },
            ArrowDataType::Int64 => {
                Ok(Self::Int64(array.as_any().downcast_ref::<Int64Array>().expect("validated Int64Array")))
            },
            ArrowDataType::UInt64 => {
                Ok(Self::UInt64(array.as_any().downcast_ref::<UInt64Array>().expect("validated UInt64Array")))
            },
            ArrowDataType::Float64 => {
                Ok(Self::Float64(array.as_any().downcast_ref::<Float64Array>().expect("validated Float64Array")))
            },
            ArrowDataType::Utf8 => Ok(Self::Utf8(HostStringKeyColumn::from_utf8(
                array.as_any().downcast_ref::<StringArray>().expect("validated StringArray"),
            ))),
            ArrowDataType::LargeUtf8 => Ok(Self::LargeUtf8(HostStringKeyColumn::from_large_utf8(
                array.as_any().downcast_ref::<LargeStringArray>().expect("validated LargeStringArray"),
            ))),
            ArrowDataType::Utf8View => Ok(Self::Utf8View(HostStringKeyColumn::from_utf8_view(
                array.as_any().downcast_ref::<StringViewArray>().expect("validated StringViewArray"),
            ))),
            data_type => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.aggregate.floating_statistical_exact.key",
                "DataFusion-exact stddev aggregate does not support this group key type",
            )
            .with_fact("column", column)
            .with_fact("data_type", format!("{data_type:?}"))),
        }
    }

    fn encoded_part(&self, row: usize) -> HostEncodedKeyPart {
        match self {
            Self::Boolean(array) => {
                if array.is_null(row) {
                    HostEncodedKeyPart::null()
                } else {
                    HostEncodedKeyPart::value(u64::from(array.value(row)))
                }
            },
            Self::Int32(array) => {
                if array.is_null(row) {
                    HostEncodedKeyPart::null()
                } else {
                    HostEncodedKeyPart::value(array.value(row) as u32 as u64)
                }
            },
            Self::Date32(array) => {
                if array.is_null(row) {
                    HostEncodedKeyPart::null()
                } else {
                    HostEncodedKeyPart::value(array.value(row) as u32 as u64)
                }
            },
            Self::Int64(array) => {
                if array.is_null(row) {
                    HostEncodedKeyPart::null()
                } else {
                    HostEncodedKeyPart::value(array.value(row) as u64)
                }
            },
            Self::UInt64(array) => {
                if array.is_null(row) {
                    HostEncodedKeyPart::null()
                } else {
                    HostEncodedKeyPart::value(array.value(row))
                }
            },
            Self::Float64(array) => {
                if array.is_null(row) {
                    HostEncodedKeyPart::null()
                } else {
                    HostEncodedKeyPart::value(array.value(row).to_bits())
                }
            },
            Self::Utf8(column) | Self::LargeUtf8(column) | Self::Utf8View(column) => column.encoded_part(row),
        }
    }

    fn bool_value(&self, part: HostEncodedKeyPart) -> Option<bool> {
        if part.is_null {
            None
        } else if matches!(self, Self::Boolean(_)) {
            Some(part.bits != 0)
        } else {
            None
        }
    }

    fn int32_value(&self, part: HostEncodedKeyPart) -> Option<i32> {
        if part.is_null {
            None
        } else if matches!(self, Self::Int32(_) | Self::Date32(_)) {
            Some(part.bits as u32 as i32)
        } else {
            None
        }
    }

    fn int64_value(&self, part: HostEncodedKeyPart) -> Option<i64> {
        if part.is_null {
            None
        } else if matches!(self, Self::Int64(_)) {
            Some(part.bits as i64)
        } else {
            None
        }
    }

    fn uint64_value(&self, part: HostEncodedKeyPart) -> Option<u64> {
        if part.is_null {
            None
        } else if matches!(self, Self::UInt64(_)) {
            Some(part.bits)
        } else {
            None
        }
    }

    fn float64_value(&self, part: HostEncodedKeyPart) -> Option<f64> {
        if part.is_null {
            None
        } else if matches!(self, Self::Float64(_)) {
            Some(f64::from_bits(part.bits))
        } else {
            None
        }
    }

    fn string_value(&self, part: HostEncodedKeyPart) -> Option<&str> {
        match self {
            Self::Utf8(column) | Self::LargeUtf8(column) | Self::Utf8View(column) => column.value(part),
            _ => None,
        }
    }
}

enum HostNumericColumn<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    UInt64(&'a UInt64Array),
    Float64(&'a Float64Array),
}

impl<'a> HostNumericColumn<'a> {
    fn new(batch: &'a RecordBatch, column: &str) -> Result<Self> {
        let array = batch.column(batch.schema().index_of(column).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.aggregate.floating_statistical_exact.value",
                "aggregate input column is missing from aggregate input batch",
            )
            .with_fact("column", column)
            .with_source(error)
        })?);
        match array.data_type() {
            ArrowDataType::Int32 => {
                Ok(Self::Int32(array.as_any().downcast_ref::<Int32Array>().expect("validated Int32Array")))
            },
            ArrowDataType::Int64 => {
                Ok(Self::Int64(array.as_any().downcast_ref::<Int64Array>().expect("validated Int64Array")))
            },
            ArrowDataType::UInt64 => {
                Ok(Self::UInt64(array.as_any().downcast_ref::<UInt64Array>().expect("validated UInt64Array")))
            },
            ArrowDataType::Float64 => {
                Ok(Self::Float64(array.as_any().downcast_ref::<Float64Array>().expect("validated Float64Array")))
            },
            data_type => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.aggregate.floating_statistical_exact.value",
                "DataFusion-exact stddev aggregate requires numeric aggregate inputs",
            )
            .with_fact("column", column)
            .with_fact("data_type", format!("{data_type:?}"))),
        }
    }

    fn value(&self, row: usize) -> Option<f64> {
        match self {
            Self::Int32(array) => (!array.is_null(row)).then(|| array.value(row) as f64),
            Self::Int64(array) => (!array.is_null(row)).then(|| array.value(row) as f64),
            Self::UInt64(array) => (!array.is_null(row)).then(|| array.value(row) as f64),
            Self::Float64(array) => (!array.is_null(row)).then(|| array.value(row)),
        }
    }
}

fn aggregate_input_column(expr: &AggregateExpr) -> Result<HostAggregateInput> {
    if matches!(expr.function, AggregateFunction::Count) && global_count_star_expr(&expr.expr) {
        return Ok(HostAggregateInput::NonNullLiteral);
    }
    aggregate_input_expr(&expr.expr)
}

fn aggregate_input_expr(expr: &Expr) -> Result<HostAggregateInput> {
    match expr {
        Expr::Column(column) => Ok(HostAggregateInput::Column(column.name().to_string())),
        Expr::Alias { expr, .. } => aggregate_input_expr(expr),
        Expr::Literal(LiteralValue::Null(_)) => Ok(HostAggregateInput::NullLiteral),
        Expr::Literal(_) => Ok(HostAggregateInput::NonNullLiteral),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.aggregate.floating_statistical_exact.lower_expr",
            "DataFusion-exact stddev aggregate requires prepared column aggregate inputs",
        )
        .with_fact("expression", expr.to_string())),
    }
}

fn host_key_output_array(
    groups: &[HostAggregateGroup],
    key_index: usize,
    data_type: &ArrowDataType,
    key_column: &HostKeyColumn<'_>,
) -> Result<ArrayRef> {
    host_key_output_array_from_keys(groups.iter().map(|group| &group.key), key_index, data_type, key_column)
}

fn host_key_output_array_from_keys<'a>(
    keys: impl IntoIterator<Item = &'a HostGroupLookupKey>,
    key_index: usize,
    data_type: &ArrowDataType,
    key_column: &HostKeyColumn<'_>,
) -> Result<ArrayRef> {
    match data_type {
        ArrowDataType::Boolean => Ok(Arc::new(BooleanArray::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.bool_value(part)))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Int32 => Ok(Arc::new(Int32Array::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.int32_value(part)))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Date32 => Ok(Arc::new(Date32Array::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.int32_value(part)))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Int64 => Ok(Arc::new(Int64Array::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.int64_value(part)))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::UInt64 => Ok(Arc::new(UInt64Array::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.uint64_value(part)))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Float64 => Ok(Arc::new(Float64Array::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.float64_value(part)))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Utf8 => Ok(Arc::new(StringArray::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.string_value(part)))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::LargeUtf8 => Ok(Arc::new(LargeStringArray::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.string_value(part)))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Utf8View => Ok(Arc::new(StringViewArray::from(
            keys.into_iter()
                .map(|key| key.part(key_index).and_then(|part| key_column.string_value(part)))
                .collect::<Vec<_>>(),
        ))),
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.aggregate.floating_statistical_exact.key_output",
            "DataFusion-exact stddev aggregate does not support this output group key type",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn host_aggregate_output_array(
    groups: &[HostAggregateGroup],
    aggregate_index: usize,
    data_type: &ArrowDataType,
) -> Result<ArrayRef> {
    match data_type {
        ArrowDataType::Int64 => Ok(Arc::new(Int64Array::from(
            groups
                .iter()
                .map(|group| group.aggregates.get(aggregate_index).and_then(HostAggregateState::evaluate_i64))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Float64 => Ok(Arc::new(Float64Array::from(
            groups
                .iter()
                .map(|group| group.aggregates.get(aggregate_index).and_then(HostAggregateState::evaluate_f64))
                .collect::<Vec<_>>(),
        ))),
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.aggregate.floating_statistical_exact.aggregate_output",
            "DataFusion-exact stddev aggregate does not support this aggregate output type",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}
