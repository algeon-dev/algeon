//! DataFusion-exact host execution of aggregates that contain floating variance or standard
//! deviation (`statistical_aggregate_execution_mode = datafusion_exact_host_welford`).
//!
//! DataFusion accumulates variance with Welford's online update in input row order, so the low
//! bits of its f64 result depend on both the update formula and the order rows arrive in. The
//! GPU path in `statistical.rs` reduces in parallel from sum and sum-of-squares moments. It
//! cannot reproduce those bits, and it loses precision when values are large relative to their
//! spread. This mode instead copies the aggregate input to host memory and replays DataFusion's
//! accumulators row by row inside the native wrapper. The native-lowering design page
//! ("Require GPU execution") and the native settings of the configuration page state the
//! user-visible contract.
//!
//! The flow is `prepare_datafusion_exact_host_aggregate_input_frame` (stable sort by the hidden
//! `__algeon_native_df_order_*` columns, when the query tracks DataFusion order),
//! `materialize_datafusion_exact_host_aggregate_input` (device frame to Arrow `RecordBatch`),
//! `execute_datafusion_exact_host_aggregate_batch` (hash grouping and accumulation) and
//! `upload_datafusion_exact_host_aggregate_output` (result back to the device). Every aggregate
//! in the spec runs here, including those that are not statistical. Accumulators come from
//! `datafusion_exact_host_accumulator`: integer sums wrap, MIN and MAX keep their input width,
//! and NaN ordering follows DataFusion.
//!
//! The batch and the accumulator tables occupy system RAM, outside the GPU grant. Only the
//! callers' input preparation runs under an `ExecutionBudgetToken`; the order sort and the
//! upload allocate through the context runtime.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, hash_map::Entry},
    sync::Arc,
    time::Instant,
};

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
        StringArray, StringViewArray, UInt64Array,
    },
    datatypes::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef},
};
use cudf::hash::{FastHashMap, fast_hash_map_with_capacity};

use super::{
    add_floating_statistical_aggregate_contract_facts, cudf_query_error,
    partial_strategy::{
        add_aggregate_strategy_execution_option_facts, add_partial_aggregate_input_evidence_facts, reduction_ratio_ppm,
    },
    *,
};
use crate::prepared::{
    DataFusionExactHostAccumulator, DataFusionExactHostAccumulator as Accumulator, datafusion_exact_host_accumulator,
    datafusion_exact_host_group_key_supported,
};

/// One group key value reduced to 64 bits.
///
/// `is_null` is part of the identity, so NULL keys form one group distinct from every value,
/// as GROUP BY requires. `bits` holds the integer value, the bit pattern of a Float64 (so
/// floats compare by bits), or the interned code of a string.
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

/// Hash-map key for one row's group. Variants up to four key columns keep the lookup key of the
/// common key counts free of a `Vec` allocation.
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

/// One non-null aggregate input value; integer inputs never pass through `f64`.
#[derive(Debug, Clone, Copy)]
enum HostValue {
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    /// A value that only `COUNT` reads.
    Counted,
}

impl HostValue {
    fn signed(self) -> i64 {
        match self {
            Self::Signed(value) => value,
            other => unreachable!("signed accumulator received {other:?}"),
        }
    }

    fn unsigned(self) -> u64 {
        match self {
            Self::Unsigned(value) => value,
            other => unreachable!("unsigned accumulator received {other:?}"),
        }
    }

    fn float(self) -> f64 {
        match self {
            Self::Float(value) => value,
            other => unreachable!("float accumulator received {other:?}"),
        }
    }

    /// DataFusion casts AVG, STDDEV, and VAR inputs to `Float64` before
    /// accumulating; `as f64` is the same conversion.
    fn as_f64(self) -> f64 {
        match self {
            Self::Signed(value) => value as f64,
            Self::Unsigned(value) => value as f64,
            Self::Float(value) => value,
            Self::Counted => unreachable!("numeric accumulator received a count-only value"),
        }
    }
}

/// Running state of one aggregate for one group, selected by its `DataFusionExactHostAccumulator`.
///
/// `seen` marks that at least one non-NULL input arrived, so a group with only NULL inputs
/// evaluates to NULL. COUNT, AVG and Welford carry a count instead.
///
/// `Welford` is Welford's online update as DataFusion's variance accumulator applies it, and
/// `m2` is the running sum of squared deviations from the mean. The order of the
/// floating-point operations in `update` is what makes the result match DataFusion's bits.
#[derive(Debug, Clone)]
enum HostAggregateState {
    Count { count: u64 },
    Signed { value: i64, seen: bool },
    Unsigned { value: u64, seen: bool },
    Float { value: f64, seen: bool },
    Avg { sum: f64, count: u64 },
    Welford { function: AggregateFunction, count: u64, mean: f64, m2: f64 },
}

impl HostAggregateState {
    fn new(accumulator: DataFusionExactHostAccumulator, function: AggregateFunction, grouped: bool) -> Self {
        match accumulator {
            Accumulator::Count => Self::Count { count: 0 },
            Accumulator::SignedSum | Accumulator::SignedMin | Accumulator::SignedMax => {
                Self::Signed { value: 0, seen: false }
            },
            Accumulator::UnsignedSum | Accumulator::UnsignedMin | Accumulator::UnsignedMax => {
                Self::Unsigned { value: 0, seen: false }
            },
            // DataFusion's grouped primitive MIN/MAX starts every group at the
            // finite type bound, so a group holding only an infinity on the
            // far side keeps that bound. Its global accumulator starts from
            // the first value.
            Accumulator::FloatMin if grouped => Self::Float { value: f64::MAX, seen: false },
            Accumulator::FloatMax if grouped => Self::Float { value: f64::MIN, seen: false },
            Accumulator::FloatSum | Accumulator::FloatMin | Accumulator::FloatMax => {
                Self::Float { value: 0.0, seen: false }
            },
            Accumulator::Avg => Self::Avg { sum: 0.0, count: 0 },
            Accumulator::Welford => Self::Welford { function, count: 0, mean: 0.0, m2: 0.0 },
        }
    }

    fn update(&mut self, accumulator: DataFusionExactHostAccumulator, grouped: bool, input: HostValue) {
        match (self, accumulator) {
            (Self::Count { count }, Accumulator::Count) => *count += 1,
            (Self::Signed { value, seen }, Accumulator::SignedSum) => {
                *value = value.wrapping_add(input.signed());
                *seen = true;
            },
            (Self::Signed { value, seen }, Accumulator::SignedMin) => {
                let input = input.signed();
                if !*seen || input < *value {
                    *value = input;
                }
                *seen = true;
            },
            (Self::Signed { value, seen }, Accumulator::SignedMax) => {
                let input = input.signed();
                if !*seen || input > *value {
                    *value = input;
                }
                *seen = true;
            },
            (Self::Unsigned { value, seen }, Accumulator::UnsignedSum) => {
                *value = value.wrapping_add(input.unsigned());
                *seen = true;
            },
            (Self::Unsigned { value, seen }, Accumulator::UnsignedMin) => {
                let input = input.unsigned();
                if !*seen || input < *value {
                    *value = input;
                }
                *seen = true;
            },
            (Self::Unsigned { value, seen }, Accumulator::UnsignedMax) => {
                let input = input.unsigned();
                if !*seen || input > *value {
                    *value = input;
                }
                *seen = true;
            },
            (Self::Float { value, seen }, Accumulator::FloatSum) => {
                *value += input.float();
                *seen = true;
            },
            // Grouped DataFusion MIN/MAX replaces the current value whenever
            // the comparison is unordered, so NaN placement matters; global
            // MIN/MAX uses IEEE 754 total order, where positive NaN is greatest.
            (Self::Float { value, seen }, Accumulator::FloatMin) => {
                let input = input.float();
                let replace = if grouped {
                    matches!(input.partial_cmp(value), Some(Ordering::Less) | None)
                } else {
                    !*seen || input.total_cmp(value).is_lt()
                };
                if replace {
                    *value = input;
                }
                *seen = true;
            },
            (Self::Float { value, seen }, Accumulator::FloatMax) => {
                let input = input.float();
                let replace = if grouped {
                    matches!(input.partial_cmp(value), Some(Ordering::Greater) | None)
                } else {
                    !*seen || input.total_cmp(value).is_gt()
                };
                if replace {
                    *value = input;
                }
                *seen = true;
            },
            (Self::Avg { sum, count }, Accumulator::Avg) => {
                *sum += input.as_f64();
                *count += 1;
            },
            // Same operation order as `update` in DataFusion's `variance.rs`.
            (Self::Welford { count, mean, m2, .. }, Accumulator::Welford) => {
                let value = input.as_f64();
                let new_count = *count + 1;
                let delta1 = value - *mean;
                let new_mean = delta1 / new_count as f64 + *mean;
                let delta2 = value - new_mean;
                *m2 += delta1 * delta2;
                *mean = new_mean;
                *count = new_count;
            },
            (state, accumulator) => unreachable!("{accumulator:?} does not update {state:?}"),
        }
    }

    fn evaluate_i32(&self) -> Option<i32> {
        match self {
            Self::Signed { value, seen } => {
                seen.then(|| i32::try_from(*value).expect("Int32 MIN/MAX state only holds Int32 input values"))
            },
            state => unreachable!("Int32 output does not read {state:?}"),
        }
    }

    fn evaluate_i64(&self) -> Option<i64> {
        match self {
            Self::Count { count } => Some(*count as i64),
            Self::Signed { value, seen } => seen.then_some(*value),
            state => unreachable!("Int64 output does not read {state:?}"),
        }
    }

    fn evaluate_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned { value, seen } => seen.then_some(*value),
            state => unreachable!("UInt64 output does not read {state:?}"),
        }
    }

    fn evaluate_f64(&self) -> Option<f64> {
        match self {
            Self::Float { value, seen } => seen.then_some(*value),
            Self::Avg { sum, count } => (*count > 0).then(|| *sum / *count as f64),
            // Variance is undefined for `count <= ddof`; DataFusion yields NULL there.
            Self::Welford { function, count, m2, .. } => {
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
            state => unreachable!("Float64 output does not read {state:?}"),
        }
    }
}

/// One group in first-seen order of the input rows. Output rows follow this order.
struct HostAggregateGroup {
    key: HostGroupLookupKey,
    aggregates: Vec<HostAggregateState>,
}

/// Runs the whole exact-host flow for `input`: order sort, copy to host, accumulate, upload.
///
/// Takes no budget token. The order sort and the upload allocate through the context runtime,
/// and the host copy is system RAM. `extra_strategy_facts` carries the filtered-aggregate
/// materialization facts of the caller's single-pass decision into the recorded strategy row.
#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(super) fn execute_datafusion_exact_host_aggregate(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    metrics: &ExecutionMetrics,
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
    metrics.add_operator_time(timer.elapsed());
    Ok(GpuDataFrame::new(output))
}

/// Stable-sorts `input` by its hidden DataFusion order columns, when it has them, and extends
/// `input_schema` with those columns. Rows are then visited in the order DataFusion's
/// accumulators would see them, which the floating-point results depend on.
pub(in crate::exec) fn prepare_datafusion_exact_host_aggregate_input_frame(
    context: &ExecutionContext,
    input: GpuDataFrame,
    input_schema: &ArrowSchemaRef,
    metrics: &ExecutionMetrics,
) -> Result<(GpuDataFrame, ArrowSchemaRef)> {
    let frame = sort_by_datafusion_order_if_present(context, input.into_frame(), metrics)?;
    let schema = schema_with_datafusion_order_columns(&frame, input_schema);
    Ok((GpuDataFrame::new(frame), schema))
}

/// Copies `input` to host memory as one Arrow `RecordBatch` with `schema`.
pub(in crate::exec) fn materialize_datafusion_exact_host_aggregate_input(
    context: &ExecutionContext,
    input: GpuDataFrame,
    schema: ArrowSchemaRef,
) -> Result<RecordBatch> {
    let handle = GpuResultHandle::new(context, input, schema)?;
    handle.to_host_record_batch()
}

/// Aggregates `batch` row by row into `output_schema`: group keys first, then one column per
/// aggregate in spec order. Groups appear in order of first occurrence.
///
/// A global aggregate over zero rows returns one row of identity values (COUNT is 0, everything
/// else NULL); a grouped aggregate over zero rows returns none. Fails with `UnsupportedType`
/// when a group key type or an aggregate has no exact accumulator, and `UnsupportedExpression`
/// when an aggregate argument is neither a prepared column nor a literal.
pub(in crate::exec) fn execute_datafusion_exact_host_aggregate_batch(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    batch: &RecordBatch,
    output_schema: &ArrowSchemaRef,
    metrics: &ExecutionMetrics,
    extra_strategy_facts: Option<&BTreeMap<&'static str, String>>,
) -> Result<RecordBatch> {
    let key_columns = spec
        .group_exprs
        .iter()
        .map(|expr| lower_named_column_expr(expr, "query_engine.aggregate.floating_statistical_exact.lower_key"))
        .collect::<Result<Vec<_>>>()?;
    let aggregate_columns = spec.aggr_exprs.iter().map(aggregate_input_column).collect::<Result<Vec<_>>>()?;
    let key_accessors =
        key_columns.iter().map(|(input_name, _)| HostKeyColumn::new(batch, input_name)).collect::<Result<Vec<_>>>()?;
    let batch_schema = batch.schema();
    for ((input_name, _), output_field) in key_columns.iter().zip(output_schema.fields()) {
        let input_type =
            batch_schema.field_with_name(input_name).expect("group key accessor resolved this column").data_type();
        if !datafusion_exact_host_group_key_supported(input_type, output_field.data_type()) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "query_engine.aggregate.floating_statistical_exact.key_output",
                "DataFusion-exact statistical aggregate path does not support this group key output type",
            )
            .with_fact("column", input_name.as_str())
            .with_fact("input_data_type", format!("{input_type:?}"))
            .with_fact("output_data_type", format!("{:?}", output_field.data_type())));
        }
    }
    let mut aggregate_input_slots = Vec::<usize>::with_capacity(aggregate_columns.len());
    let mut unique_aggregate_inputs = Vec::<HostAggregateInput>::new();
    for input in &aggregate_columns {
        let slot = unique_aggregate_inputs.iter().position(|existing| existing == input).unwrap_or_else(|| {
            let slot = unique_aggregate_inputs.len();
            unique_aggregate_inputs.push(input.clone());
            slot
        });
        aggregate_input_slots.push(slot);
    }
    let aggregate_accessors = unique_aggregate_inputs
        .iter()
        .map(|input| HostAggregateInputAccessor::new(batch, input))
        .collect::<Result<Vec<_>>>()?;
    let accumulators = spec
        .aggr_exprs
        .iter()
        .zip(&aggregate_columns)
        .enumerate()
        .map(|(index, (expr, input))| {
            let input_type = match input {
                HostAggregateInput::Column(name) => Some(
                    batch_schema
                        .field_with_name(name)
                        .expect("aggregate input accessor resolved this column")
                        .data_type(),
                ),
                HostAggregateInput::NonNullLiteral | HostAggregateInput::NullLiteral => None,
            };
            let output_type = output_schema.fields().get(key_columns.len() + index).map(|field| field.data_type());
            output_type
                .and_then(|output_type| datafusion_exact_host_accumulator(expr, input_type, output_type))
                .ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "query_engine.aggregate.floating_statistical_exact.accumulator",
                        "DataFusion-exact statistical aggregate path has no exact accumulator for this aggregate",
                    )
                    .with_fact("function", expr.function.as_str())
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("input_data_type", format!("{input_type:?}"))
                    .with_fact("output_data_type", format!("{output_type:?}"))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let grouped = !key_columns.is_empty();
    let new_group_states = || {
        spec.aggr_exprs
            .iter()
            .zip(&accumulators)
            .map(|(expr, accumulator)| HostAggregateState::new(*accumulator, expr.function, grouped))
            .collect::<Vec<_>>()
    };
    let row_count = batch.num_rows();
    // The group count is unknown until the scan ends, so reservation is capped at 2^20 entries.
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
                groups.push(HostAggregateGroup { key, aggregates: new_group_states() });
                index
            },
        };
        for (value, accessor) in aggregate_input_values.iter_mut().zip(&aggregate_accessors) {
            *value = accessor.value(row);
        }
        for (aggregate_index, accumulator) in accumulators.iter().enumerate() {
            let Some(value) = aggregate_input_values[aggregate_input_slots[aggregate_index]] else {
                continue;
            };
            groups[group_index]
                .aggregates
                .get_mut(aggregate_index)
                .expect("aggregate state index should match aggregate expression index")
                .update(*accumulator, grouped, value);
        }
    }

    if !grouped && groups.is_empty() {
        groups.push(HostAggregateGroup { key: HostGroupLookupKey::Empty, aggregates: new_group_states() });
    }

    let mut arrays = Vec::<ArrayRef>::with_capacity(output_schema.fields().len());
    for (output_index, field) in output_schema.fields().iter().enumerate() {
        if output_index < key_columns.len() {
            arrays.push(host_key_output_array(&groups, output_index, field.data_type(), &key_accessors[output_index])?);
        } else {
            let aggregate_index = output_index - key_columns.len();
            arrays.push(host_aggregate_output_array(&groups, aggregate_index, field.data_type()));
        }
    }
    record_datafusion_exact_floating_statistical_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
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
            "query_engine.aggregate.floating_statistical_exact.output",
            "failed to construct DataFusion-exact stddev aggregate Arrow output",
        )
        .with_source(error)
    })?;
    Ok(output)
}

/// Copies the host result back to the device through the context runtime.
pub(in crate::exec) fn upload_datafusion_exact_host_aggregate_output(
    context: &ExecutionContext,
    output: &RecordBatch,
) -> Result<DataFrame> {
    DataFrame::from_record_batch(&context.runtime()?, output).map_err(|error| {
        cudf_query_error(
            "query_engine.aggregate.floating_statistical_exact.output",
            "failed to upload DataFusion-exact stddev aggregate output",
            error,
        )
    })
}

/// Records the strategy row for the host path. Fields that describe GPU strategy choices carry
/// fixed placeholders (`unknown`, `none`, `not_applicable`) because this path selects no GPU
/// strategy.
#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
fn record_datafusion_exact_floating_statistical_aggregate_strategy_decision(
    metrics: &ExecutionMetrics,
    node_id: Option<String>,
    aggregate_strategy_options: AggregateStrategyExecutionOptions,
    spec: &AggregateSpec,
    input_schema: &ArrowSchema,
    input_rows: u64,
    output_rows: u64,
    extra_strategy_facts: Option<&BTreeMap<&'static str, String>>,
) {
    let progress = metrics.progress();
    let mut facts = BTreeMap::from([
        ("selected_strategy", "datafusion_exact_host_welford".to_string()),
        ("execution_stage", AggregateExecutionStage::Single.as_str().to_string()),
        ("group_key_count", spec.group_exprs.len().to_string()),
        ("group_key_exprs", aggregate_group_key_exprs_for_metrics(spec)),
        ("group_key_dtypes", aggregate_group_key_dtypes_for_metrics(spec, input_schema)),
        ("aggregate_functions", aggregate_function_list_for_metrics(spec)),
        ("aggregate_filter_count", spec.aggr_exprs.iter().filter(|expr| expr.filter.is_some()).count().to_string()),
        ("distinct_aggregate_count", spec.aggr_exprs.iter().filter(|expr| expr.distinct).count().to_string()),
        ("input_rows", input_rows.to_string()),
        ("chunk_row_count", "0".to_string()),
        ("chunk_count", progress.source_chunks_planned.to_string()),
        ("partial_aggregate_count", progress.partial_aggregate_count.to_string()),
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

/// Copies the filtered-aggregate materialization facts of the caller's single-pass decision into
/// the host strategy row. `aggregate_filter_count` is copied over the row's own value, because
/// the spec reaching this path has had its filters masked into columns and would report 0.
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
    use query_runtime::runtime::StandaloneConfig;

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

    fn run_exact_host_aggregate(spec: &AggregateSpec, batch: &RecordBatch, output_fields: Vec<Field>) -> RecordBatch {
        let context = ExecutionContext::standalone(StandaloneConfig::default()).expect("execution context builds");
        let output_schema = Arc::new(Schema::new(output_fields));
        let metrics = ExecutionMetrics::detached("exact-host-aggregate");
        execute_datafusion_exact_host_aggregate_batch(&context, None, spec, batch, &output_schema, &metrics, None)
            .expect("exact host aggregate executes")
    }

    fn output_column<T: 'static>(batch: &RecordBatch, index: usize) -> &T {
        batch.column(index).as_any().downcast_ref::<T>().expect("output column has the declared type")
    }

    #[test]
    fn exact_host_integer_aggregates_keep_datafusion_integer_semantics() {
        let beyond_f64_integers = (1_i64 << 53) + 1;
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("key", ArrowDataType::Int32, false),
            Field::new("amount", ArrowDataType::Int64, false),
            Field::new("quantity", ArrowDataType::Int32, false),
            Field::new("units", ArrowDataType::UInt64, false),
            Field::new("label", ArrowDataType::Utf8, true),
            Field::new("price", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            input_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2])),
                Arc::new(Int64Array::from(vec![beyond_f64_integers, 2, i64::MAX, 1])),
                Arc::new(Int32Array::from(vec![7, -3, i32::MIN, 5])),
                Arc::new(UInt64Array::from(vec![u64::MAX, 2, 1, 1])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("b"), Some("c")])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 5.0])),
            ],
        )
        .unwrap();
        let spec = AggregateSpec::new(
            vec![Expr::column("key")],
            vec![
                AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum"),
                AggregateExpr::new(AggregateFunction::Max, Expr::column("amount"), "amount_max"),
                AggregateExpr::new(AggregateFunction::Min, Expr::column("quantity"), "quantity_min"),
                AggregateExpr::new(AggregateFunction::Sum, Expr::column("units"), "units_sum"),
                AggregateExpr::new(AggregateFunction::Count, Expr::column("label"), "label_count"),
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("price"), "price_stddev"),
            ],
        );

        let output = run_exact_host_aggregate(
            &spec,
            &batch,
            vec![
                Field::new("key", ArrowDataType::Int32, false),
                Field::new("amount_sum", ArrowDataType::Int64, true),
                Field::new("amount_max", ArrowDataType::Int64, true),
                Field::new("quantity_min", ArrowDataType::Int32, true),
                Field::new("units_sum", ArrowDataType::UInt64, true),
                Field::new("label_count", ArrowDataType::Int64, false),
                Field::new("price_stddev", ArrowDataType::Float64, true),
            ],
        );

        assert_eq!(output_column::<Int32Array>(&output, 0).values().as_ref(), &[1, 2]);
        // DataFusion sums signed and unsigned integers with wrapping arithmetic.
        assert_eq!(output_column::<Int64Array>(&output, 1).values().as_ref(), &[beyond_f64_integers + 2, i64::MIN]);
        assert_eq!(output_column::<Int64Array>(&output, 2).values().as_ref(), &[beyond_f64_integers, i64::MAX]);
        assert_eq!(output_column::<Int32Array>(&output, 3).values().as_ref(), &[-3, i32::MIN]);
        assert_eq!(output_column::<UInt64Array>(&output, 4).values().as_ref(), &[1, 2]);
        assert_eq!(output_column::<Int64Array>(&output, 5).values().as_ref(), &[1, 2]);
        assert_eq!(output_column::<Float64Array>(&output, 6).values().as_ref(), &[0.5_f64.sqrt(), 2.0_f64.sqrt()]);
    }

    #[test]
    fn exact_host_float_min_max_follow_datafusion_nan_ordering() {
        let min_max = |group_exprs: Vec<Expr>| {
            AggregateSpec::new(
                group_exprs,
                vec![
                    AggregateExpr::new(AggregateFunction::Min, Expr::column("value"), "value_min"),
                    AggregateExpr::new(AggregateFunction::Max, Expr::column("value"), "value_max"),
                ],
            )
        };
        let value_fields = || {
            vec![
                Field::new("value_min", ArrowDataType::Float64, true),
                Field::new("value_max", ArrowDataType::Float64, true),
            ]
        };
        let bits = |array: &Float64Array| array.iter().map(|value| value.map(f64::to_bits)).collect::<Vec<_>>();
        let expected = |values: &[f64]| values.iter().map(|value| Some(value.to_bits())).collect::<Vec<_>>();

        // Grouped DataFusion MIN/MAX starts each group at the finite type bound
        // and replaces the current value on any unordered comparison.
        let grouped_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("key", ArrowDataType::Int32, false),
                Field::new("value", ArrowDataType::Float64, true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 2, 3, 3, 4, 4])),
                Arc::new(Float64Array::from(vec![
                    Some(1.0),
                    Some(f64::NAN),
                    Some(2.0),
                    Some(f64::NEG_INFINITY),
                    Some(-0.0),
                    Some(0.0),
                    Some(f64::INFINITY),
                    None,
                ])),
            ],
        )
        .unwrap();
        let mut grouped_fields = vec![Field::new("key", ArrowDataType::Int32, false)];
        grouped_fields.extend(value_fields());
        let grouped = run_exact_host_aggregate(&min_max(vec![Expr::column("key")]), &grouped_batch, grouped_fields);
        assert_eq!(bits(output_column(&grouped, 1)), expected(&[2.0, f64::NEG_INFINITY, -0.0, f64::MAX]));
        assert_eq!(bits(output_column(&grouped, 2)), expected(&[2.0, f64::MIN, -0.0, f64::INFINITY]));

        // Global DataFusion MIN/MAX uses IEEE 754 total order.
        let global_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("value", ArrowDataType::Float64, true)])),
            vec![Arc::new(Float64Array::from(vec![1.0, f64::NAN, 2.0]))],
        )
        .unwrap();
        let global = run_exact_host_aggregate(&min_max(Vec::new()), &global_batch, value_fields());
        assert_eq!(bits(output_column(&global, 0)), expected(&[1.0]));
        assert!(output_column::<Float64Array>(&global, 1).value(0).is_nan());
    }
}

fn aggregate_group_key_exprs_for_metrics(spec: &AggregateSpec) -> String {
    if spec.group_exprs.is_empty() {
        "none".to_string()
    } else {
        spec.group_exprs.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    }
}

fn aggregate_group_key_dtypes_for_metrics(spec: &AggregateSpec, input_schema: &ArrowSchema) -> String {
    if spec.group_exprs.is_empty() {
        return "none".to_string();
    }
    spec.group_exprs
        .iter()
        .map(|expr| match lower_named_column_expr(expr, "query_engine.aggregate.floating_statistical_exact.metrics") {
            Ok((name, _)) => input_schema
                .field_with_name(name.as_str())
                .map(|field| format!("{:?}", field.data_type()))
                .unwrap_or_else(|_| "missing_column".to_string()),
            Err(_) => "non_column_expr".to_string(),
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

/// Where an aggregate reads its values. A literal reads no column: a non-NULL literal (and
/// COUNT(*)) yields a value on every row, a NULL literal on none.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum HostAggregateInput {
    Column(String),
    NonNullLiteral,
    NullLiteral,
}

enum HostAggregateInputAccessor<'a> {
    Column(HostAggregateColumn<'a>),
    NonNullLiteral,
    NullLiteral,
}

impl<'a> HostAggregateInputAccessor<'a> {
    fn new(batch: &'a RecordBatch, input: &HostAggregateInput) -> Result<Self> {
        match input {
            HostAggregateInput::Column(name) => HostAggregateColumn::new(batch, name).map(Self::Column),
            HostAggregateInput::NonNullLiteral => Ok(Self::NonNullLiteral),
            HostAggregateInput::NullLiteral => Ok(Self::NullLiteral),
        }
    }

    fn value(&self, row: usize) -> Option<HostValue> {
        match self {
            Self::Column(column) => column.value(row),
            Self::NonNullLiteral => Some(HostValue::Counted),
            Self::NullLiteral => None,
        }
    }
}

/// Typed view of one group key column of the host batch. String keys are interned to integer
/// codes when the view is built, so every key part is fixed-width.
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

/// Interns each distinct string to a dense code in first-seen order. NULL rows get no code.
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
                "query_engine.aggregate.floating_statistical_exact.key",
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
                "query_engine.aggregate.floating_statistical_exact.key",
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

enum HostAggregateColumn<'a> {
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    UInt64(&'a UInt64Array),
    Float64(&'a Float64Array),
    /// Any other input type; accumulator selection admits it only for `COUNT`,
    /// which reads validity alone.
    Counted(&'a ArrayRef),
}

impl<'a> HostAggregateColumn<'a> {
    fn new(batch: &'a RecordBatch, column: &str) -> Result<Self> {
        let array = batch.column(batch.schema().index_of(column).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.aggregate.floating_statistical_exact.value",
                "aggregate input column is missing from aggregate input batch",
            )
            .with_fact("column", column)
            .with_source(error)
        })?);
        Ok(match array.data_type() {
            ArrowDataType::Int32 => {
                Self::Int32(array.as_any().downcast_ref::<Int32Array>().expect("validated Int32Array"))
            },
            ArrowDataType::Int64 => {
                Self::Int64(array.as_any().downcast_ref::<Int64Array>().expect("validated Int64Array"))
            },
            ArrowDataType::UInt64 => {
                Self::UInt64(array.as_any().downcast_ref::<UInt64Array>().expect("validated UInt64Array"))
            },
            ArrowDataType::Float64 => {
                Self::Float64(array.as_any().downcast_ref::<Float64Array>().expect("validated Float64Array"))
            },
            _ => Self::Counted(array),
        })
    }

    fn value(&self, row: usize) -> Option<HostValue> {
        match self {
            Self::Int32(array) => (!array.is_null(row)).then(|| HostValue::Signed(i64::from(array.value(row)))),
            Self::Int64(array) => (!array.is_null(row)).then(|| HostValue::Signed(array.value(row))),
            Self::UInt64(array) => (!array.is_null(row)).then(|| HostValue::Unsigned(array.value(row))),
            Self::Float64(array) => (!array.is_null(row)).then(|| HostValue::Float(array.value(row))),
            Self::Counted(array) => (!array.is_null(row)).then_some(HostValue::Counted),
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
            "query_engine.aggregate.floating_statistical_exact.lower_expr",
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
            "query_engine.aggregate.floating_statistical_exact.key_output",
            "DataFusion-exact stddev aggregate does not support this output group key type",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn host_aggregate_output_array(
    groups: &[HostAggregateGroup],
    aggregate_index: usize,
    data_type: &ArrowDataType,
) -> ArrayRef {
    let states = groups.iter().map(|group| &group.aggregates[aggregate_index]);
    match data_type {
        ArrowDataType::Int32 => Arc::new(states.map(HostAggregateState::evaluate_i32).collect::<Int32Array>()),
        ArrowDataType::Int64 => Arc::new(states.map(HostAggregateState::evaluate_i64).collect::<Int64Array>()),
        ArrowDataType::UInt64 => Arc::new(states.map(HostAggregateState::evaluate_u64).collect::<UInt64Array>()),
        ArrowDataType::Float64 => Arc::new(states.map(HostAggregateState::evaluate_f64).collect::<Float64Array>()),
        data_type => unreachable!("exact host accumulator selection admitted aggregate output type {data_type:?}"),
    }
}
