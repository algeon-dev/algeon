//! GPU cases for the final merge's groupby skips: a key-distinct partial converted without the
//! final groupby must equal the groupby's output, and the incremental merge must still group every
//! input set that can repeat a key.

use std::sync::Arc;

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, DurationMicrosecondArray, Float32Array,
        Float64Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt32Array, UInt64Array,
    },
    compute::{SortColumn, lexsort_to_indices, take_record_batch},
    datatypes::{DataType as ArrowDataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::{DataType as CudfDataType, TypeId, testing::run_named_cases};
use query_planning::{
    capability::AggregateExecutionStage,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr},
};
use query_runtime::runtime::StandaloneConfig;

use super::{
    AggregateNode, BoundedExecutionAttempt, CommittedPartial, CompiledIncrementalPartitionedMerge, DataFrame,
    ExecutionBudgetToken, ExecutionContext, ExecutionMetrics, GpuDataFrame, MemoryControlPolicy, NodeId,
    compiled::{
        convert_key_distinct_partial_to_merged_state, key_distinct_partial_skips_final_groupby,
        merge_compiled_partitioned_final_aggregate_partial_refs,
    },
    execution_budget_token_for_boundary, observe_device_memory_bytes,
};

const FINAL_GROUPBY_OPERATION: &str = "query_engine.final_merge.aggregate";

struct Fixture {
    context: ExecutionContext,
    metrics: ExecutionMetrics,
}

impl Fixture {
    fn new() -> Self {
        Self {
            context: ExecutionContext::standalone(StandaloneConfig::default()).expect("standalone execution context"),
            metrics: ExecutionMetrics::detached("final_merge_execution_tests"),
        }
    }

    fn attempt(&self) -> BoundedExecutionAttempt<'_> {
        let MemoryControlPolicy { device_budget_bytes, .. } = self.context.memory_control_policy();
        BoundedExecutionAttempt::new(&self.context, device_budget_bytes, &self.metrics)
    }

    fn partial(&self, batch: &RecordBatch) -> CommittedPartial {
        let runtime = self.context.runtime().expect("execution runtime");
        let frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, batch).expect("partial upload"));
        let bytes = observe_device_memory_bytes(&frame, &self.metrics).expect("partial bytes");
        CommittedPartial::with_device_bytes(frame, bytes)
    }

    /// Casts one uploaded column, for cuDF types the Arrow import does not produce.
    fn cast(&self, partial: CommittedPartial, column: &str, target: CudfDataType) -> CommittedPartial {
        let runtime = self.context.runtime().expect("execution runtime");
        let frame = partial.frame.frame().on(runtime.exec()).cast_column(column, target).expect("state cast");
        let frame = GpuDataFrame::new(frame);
        let bytes = observe_device_memory_bytes(&frame, &self.metrics).expect("cast bytes");
        CommittedPartial::with_device_bytes(frame, bytes)
    }

    fn final_groupby_count(&self) -> usize {
        self.metrics
            .snapshot()
            .budgeted_execution_events
            .iter()
            .filter(|event| event.operation == FINAL_GROUPBY_OPERATION)
            .count()
    }
}

/// One aggregate over key-distinct partial state, with the schemas the merge needs.
struct StateCase {
    aggregate: AggregateNode,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl StateCase {
    /// `aggregates` lists (function, alias, output type); every argument reads column `v`.
    fn new(keys: &[(&str, ArrowDataType)], aggregates: &[(AggregateFunction, &str, ArrowDataType)]) -> Self {
        let key_fields = keys.iter().map(|(name, data_type)| Field::new(*name, data_type.clone(), true));
        let input_schema = Arc::new(Schema::new(
            key_fields.clone().chain(std::iter::once(Field::new("v", ArrowDataType::Int64, true))).collect::<Vec<_>>(),
        ));
        let output_schema = Arc::new(Schema::new(
            key_fields
                .chain(aggregates.iter().map(|(_, alias, data_type)| Field::new(*alias, data_type.clone(), true)))
                .collect::<Vec<_>>(),
        ));
        let spec = AggregateSpec::new(
            keys.iter().map(|(name, _)| Expr::column(*name)).collect(),
            aggregates
                .iter()
                .map(|(function, alias, _)| AggregateExpr::new(*function, Expr::column("v"), *alias))
                .collect(),
        );
        Self { aggregate: AggregateNode { input: NodeId::from_index_for_test(0), spec }, input_schema, output_schema }
    }

    fn eligible(&self, partial: &CommittedPartial) -> bool {
        key_distinct_partial_skips_final_groupby(&self.aggregate, &partial.frame).expect("state types readable")
    }

    fn groupby(&self, fixture: &Fixture, partial: &CommittedPartial) -> RecordBatch {
        let mut execution = fixture.attempt();
        let token = budget_token(&execution);
        merge_compiled_partitioned_final_aggregate_partial_refs(
            &mut execution,
            NodeId::from_index_for_test(1),
            &self.aggregate,
            &self.input_schema,
            &self.output_schema,
            0,
            &[partial],
            &token,
        )
        .expect("final groupby")
        .frame()
        .to_record_batch()
        .expect("groupby download")
    }

    fn convert(&self, fixture: &Fixture, partial: &CommittedPartial) -> RecordBatch {
        let execution = fixture.attempt();
        let token = budget_token(&execution);
        convert_key_distinct_partial_to_merged_state(
            &self.aggregate,
            &self.output_schema,
            partial.frame.clone(),
            &token,
            &fixture.metrics,
        )
        .expect("key-distinct conversion")
        .frame()
        .to_record_batch()
        .expect("conversion download")
    }

    /// Asserts that the conversion applies and equals the groupby in column names, types, row
    /// set and value bits, and that it ran no groupby.
    fn assert_conversion_matches_groupby(&self, fixture: &Fixture, partial: &CommittedPartial) {
        assert!(self.eligible(partial), "{:?}", self.aggregate.spec);
        let groupby_count = fixture.final_groupby_count();
        let converted = self.convert(fixture, partial);
        assert_eq!(fixture.final_groupby_count(), groupby_count, "the conversion runs no groupby");
        let grouped = self.groupby(fixture, partial);
        let keys = self.aggregate.spec.group_exprs.len();
        assert_batches_bitwise_equal(
            &sort_by_leading_columns(&converted, keys),
            &sort_by_leading_columns(&grouped, keys),
        );
    }
}

fn budget_token(execution: &BoundedExecutionAttempt<'_>) -> ExecutionBudgetToken {
    execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "final_aggregate_merge",
        "query_engine.final_merge.execution_tests.budget_token",
        "failed to create final merge test budget token",
    )
    .expect("budget token")
}

fn sort_by_leading_columns(batch: &RecordBatch, count: usize) -> RecordBatch {
    let columns = (0..count)
        .map(|index| SortColumn { values: Arc::clone(batch.column(index)), options: None })
        .collect::<Vec<_>>();
    take_record_batch(batch, &lexsort_to_indices(&columns, None).expect("sort keys")).expect("sorted batch")
}

/// Float columns compare by bit pattern so NaN payloads and signed zeros count as differences.
fn assert_batches_bitwise_equal(left: &RecordBatch, right: &RecordBatch) {
    let fields = |batch: &RecordBatch| {
        batch
            .schema()
            .fields()
            .iter()
            .map(|field| (field.name().clone(), field.data_type().clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(fields(left), fields(right));
    for (index, (left, right)) in left.columns().iter().zip(right.columns()).enumerate() {
        match left.data_type() {
            ArrowDataType::Float64 => assert_eq!(float_bits::<Float64Array>(left), float_bits::<Float64Array>(right)),
            ArrowDataType::Float32 => assert_eq!(float_bits::<Float32Array>(left), float_bits::<Float32Array>(right)),
            _ => assert_eq!(left.to_data(), right.to_data(), "column {index}"),
        }
    }
}

trait FloatBits {
    fn bits(&self) -> Vec<Option<u64>>;
}

impl FloatBits for Float64Array {
    fn bits(&self) -> Vec<Option<u64>> {
        self.iter().map(|value| value.map(f64::to_bits)).collect()
    }
}

impl FloatBits for Float32Array {
    fn bits(&self) -> Vec<Option<u64>> {
        self.iter().map(|value| value.map(|value| u64::from(value.to_bits()))).collect()
    }
}

fn float_bits<A: FloatBits + 'static>(array: &ArrayRef) -> Vec<Option<u64>> {
    array.as_any().downcast_ref::<A>().expect("float column").bits()
}

/// Five unique keys, one of them NULL: a NULL key forms its own group.
fn keys() -> ArrayRef {
    Arc::new(Int64Array::from(vec![Some(3), None, Some(1), Some(7), Some(5)]))
}

fn state_batch(state: ArrayRef) -> RecordBatch {
    RecordBatch::try_from_iter([("k", keys()), ("s", state)]).expect("state batch")
}

fn decimal_state() -> ArrayRef {
    Arc::new(
        Decimal128Array::from(vec![Some(12_345), None, Some(-1), Some(i64::MAX as i128), Some(0)])
            .with_precision_and_scale(25, 2)
            .expect("decimal state"),
    )
}

/// Float states with a NaN, a negative zero, infinities and a NULL.
fn float64_state() -> ArrayRef {
    Arc::new(Float64Array::from(vec![Some(f64::NAN), Some(-0.0), Some(f64::INFINITY), None, Some(-1.5e300)]))
}

fn float32_state() -> ArrayRef {
    Arc::new(Float32Array::from(vec![Some(f32::NAN), Some(-0.0), Some(f32::NEG_INFINITY), None, Some(2.5)]))
}

fn sum_conversion_matches_groupby_for_type_preserving_states() {
    let fixture = Fixture::new();
    let decimal = || ArrowDataType::Decimal128(25, 2);
    let cases: Vec<(ArrayRef, ArrowDataType, Option<CudfDataType>)> = vec![
        (
            Arc::new(Int64Array::from(vec![Some(i64::MAX), None, Some(i64::MIN), Some(0), Some(-7)])),
            ArrowDataType::Int64,
            None,
        ),
        (decimal_state(), decimal(), None),
        (decimal_state(), decimal(), Some(CudfDataType::fixed_point(TypeId::Decimal64, -2).unwrap())),
        (
            Arc::new(
                Decimal128Array::from(vec![Some(1), None, Some(-99_999), Some(0), Some(7)])
                    .with_precision_and_scale(9, 3)
                    .unwrap(),
            ),
            ArrowDataType::Decimal128(9, 3),
            Some(CudfDataType::fixed_point(TypeId::Decimal32, -3).unwrap()),
        ),
        (
            Arc::new(DurationMicrosecondArray::from(vec![Some(1), None, Some(-5), Some(i64::MAX), Some(0)])),
            ArrowDataType::Duration(TimeUnit::Microsecond),
            None,
        ),
    ];
    for (state, output_type, cast) in cases {
        let case = StateCase::new(&[("k", ArrowDataType::Int64)], &[(AggregateFunction::Sum, "s", output_type)]);
        let mut partial = fixture.partial(&state_batch(state));
        if let Some(target) = cast {
            partial = fixture.cast(partial, "s", target);
        }
        case.assert_conversion_matches_groupby(&fixture, &partial);
    }
}

fn min_max_conversion_matches_groupby_for_every_state_type() {
    let fixture = Fixture::new();
    let states: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![Some(i32::MIN), None, Some(0), Some(i32::MAX), Some(4)])),
        Arc::new(UInt64Array::from(vec![Some(u64::MAX), Some(0), None, Some(1), Some(9)])),
        float64_state(),
        float32_state(),
        Arc::new(StringArray::from(vec![Some(""), Some("zz"), None, Some("a"), Some("\u{00e9}")])),
        Arc::new(BooleanArray::from(vec![Some(true), None, Some(false), Some(true), Some(false)])),
        Arc::new(Date32Array::from(vec![Some(0), Some(-1), Some(20_000), None, Some(1)])),
        Arc::new(TimestampMicrosecondArray::from(vec![Some(1), None, Some(-2), Some(i64::MAX), Some(0)])),
        decimal_state(),
    ];
    for state in states {
        let output_type = state.data_type().clone();
        for function in [AggregateFunction::Min, AggregateFunction::Max] {
            let case = StateCase::new(&[("k", ArrowDataType::Int64)], &[(function, "s", output_type.clone())]);
            case.assert_conversion_matches_groupby(&fixture, &fixture.partial(&state_batch(Arc::clone(&state))));
        }
    }
}

fn count_conversion_casts_integer_states_like_the_groupby() {
    let fixture = Fixture::new();
    let states: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![1, 0, 7, i32::MAX, 3])),
        Arc::new(Int64Array::from(vec![1, 0, 7, i64::MAX, 3])),
        Arc::new(UInt32Array::from(vec![1, 0, 7, u32::MAX, 3])),
    ];
    for state in states {
        let case =
            StateCase::new(&[("k", ArrowDataType::Int64)], &[(AggregateFunction::Count, "s", ArrowDataType::Int64)]);
        case.assert_conversion_matches_groupby(&fixture, &fixture.partial(&state_batch(state)));
    }
}

/// Two keys, one of them a string with a NULL, and aggregates whose state columns are in a
/// different order than the output schema.
fn mixed_conversion_matches_groupby_with_composite_keys_and_reordered_state() {
    let fixture = Fixture::new();
    let case = StateCase::new(
        &[("k", ArrowDataType::Int64), ("tag", ArrowDataType::Utf8)],
        &[
            (AggregateFunction::Sum, "total", ArrowDataType::Decimal128(25, 2)),
            (AggregateFunction::Min, "low", ArrowDataType::Utf8),
            (AggregateFunction::Max, "high", ArrowDataType::Float64),
            (AggregateFunction::Count, "n", ArrowDataType::Int64),
        ],
    );
    let batch = RecordBatch::try_from_iter([
        ("n", Arc::new(Int32Array::from(vec![2, 1, 1, 5, 9])) as ArrayRef),
        ("high", float64_state()),
        ("tag", Arc::new(StringArray::from(vec![Some("a"), Some("a"), None, Some("b"), Some("a")]))),
        ("total", decimal_state()),
        ("k", Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(1), Some(1), None]))),
        ("low", Arc::new(StringArray::from(vec![Some("x"), None, Some(""), Some("y"), Some("z")]))),
    ])
    .expect("mixed state batch");
    case.assert_conversion_matches_groupby(&fixture, &fixture.partial(&batch));
}

/// Each shape here would change under the final groupby or needs it to merge, so the conversion
/// must not apply. The integer SUM case shows the type change that rules it out.
fn conversion_rejects_shapes_whose_groupby_changes_the_state() {
    let fixture = Fixture::new();
    let int32_sum =
        StateCase::new(&[("k", ArrowDataType::Int64)], &[(AggregateFunction::Sum, "s", ArrowDataType::Int64)]);
    let int32_state = fixture.partial(&state_batch(Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))));
    assert!(!int32_sum.eligible(&int32_state));
    let grouped = int32_sum.groupby(&fixture, &int32_state);
    assert_eq!(grouped.schema().field_with_name("s").expect("sum column").data_type(), &ArrowDataType::Int64);

    // The groupby adds a lone -0.0 to a zero accumulator and returns 0.0.
    let float_sum =
        StateCase::new(&[("k", ArrowDataType::Int64)], &[(AggregateFunction::Sum, "s", ArrowDataType::Float64)]);
    let float_state = fixture.partial(&state_batch(float64_state()));
    assert!(!float_sum.eligible(&float_state));
    let grouped = float_sum.groupby(&fixture, &float_state);
    let negative_zero_key = grouped
        .column_by_name("k")
        .and_then(|keys| keys.as_any().downcast_ref::<Int64Array>().cloned())
        .and_then(|keys| keys.iter().position(|key| key.is_none()))
        .expect("the NULL key holds -0.0");
    let sums = grouped.column_by_name("s").expect("sum column").as_any().downcast_ref::<Float64Array>().expect("f64");
    assert_eq!(sums.value(negative_zero_key).to_bits(), 0.0_f64.to_bits());

    for (function, state) in [
        (AggregateFunction::Sum, Arc::new(UInt64Array::from(vec![1_u64, 2, 3, 4, 5])) as ArrayRef),
        (AggregateFunction::Sum, float32_state()),
        (AggregateFunction::Sum, Arc::new(BooleanArray::from(vec![true, false, true, true, false]))),
        (AggregateFunction::Count, float64_state()),
        (AggregateFunction::NUnique, Arc::new(Int64Array::from(vec![1_i64, 1, 2, 1, 3]))),
    ] {
        let case = StateCase::new(&[("k", ArrowDataType::Int64)], &[(function, "s", ArrowDataType::Int64)]);
        assert!(!case.eligible(&fixture.partial(&state_batch(state))), "{function:?}");
    }

    let mut distinct_count =
        StateCase::new(&[("k", ArrowDataType::Int64)], &[(AggregateFunction::Count, "s", ArrowDataType::Int64)]);
    distinct_count.aggregate.spec.aggr_exprs[0].distinct = true;
    let count_state = fixture.partial(&state_batch(Arc::new(Int64Array::from(vec![1_i64, 1, 2, 1, 3]))));
    assert!(!distinct_count.eligible(&count_state));

    let with_avg = StateCase::new(
        &[("k", ArrowDataType::Int64)],
        &[(AggregateFunction::Sum, "s", ArrowDataType::Int64), (AggregateFunction::Avg, "a", ArrowDataType::Float64)],
    );
    assert!(!with_avg.eligible(&count_state));

    let mut global =
        StateCase::new(&[("k", ArrowDataType::Int64)], &[(AggregateFunction::Sum, "s", ArrowDataType::Int64)]);
    global.aggregate.spec.group_exprs.clear();
    assert!(!global.eligible(&count_state));
}

/// SUM, MIN, MAX and COUNT over key `k` and value `v`, with integer raw partial state.
fn incremental_case() -> StateCase {
    StateCase::new(
        &[("k", ArrowDataType::Int64)],
        &[
            (AggregateFunction::Sum, "total", ArrowDataType::Int64),
            (AggregateFunction::Min, "low", ArrowDataType::Int64),
            (AggregateFunction::Max, "high", ArrowDataType::Int64),
            (AggregateFunction::Count, "n", ArrowDataType::Int64),
        ],
    )
}

/// Raw partial state with one row per key: total = key, low = high = key, n = 1.
fn raw_partial(keys: impl IntoIterator<Item = i64>) -> RecordBatch {
    let keys = keys.into_iter().collect::<Vec<_>>();
    let column = |values: Vec<i64>| Arc::new(Int64Array::from(values)) as ArrayRef;
    RecordBatch::try_from_iter([
        ("k", column(keys.clone())),
        ("total", column(keys.clone())),
        ("low", column(keys.clone())),
        ("high", column(keys.clone())),
        ("n", Arc::new(Int32Array::from(vec![1; keys.len()])) as ArrayRef),
    ])
    .expect("raw partial")
}

/// Sorted (key, total, low, high, n) rows of a final output.
fn output_rows(frame: &GpuDataFrame) -> Vec<(i64, i64, i64, i64, i64)> {
    let batch = frame.frame().to_record_batch().expect("output download");
    let column = |name: &str| {
        batch.column_by_name(name).expect("output column").as_any().downcast_ref::<Int64Array>().expect("int64").clone()
    };
    let (k, total, low, high, n) = (column("k"), column("total"), column("low"), column("high"), column("n"));
    let mut rows = (0..batch.num_rows())
        .map(|row| (k.value(row), total.value(row), low.value(row), high.value(row), n.value(row)))
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows
}

/// Two partials share keys 50..100 and arrive in one batch, so each partition's single slice
/// repeats those keys and must still go through the groupby. That is one groupby per non-empty
/// partition and none for the projection of the merged states, which records no strategy row of
/// its own.
fn incremental_merge_groups_slices_of_a_multi_partial_batch() {
    let fixture = Fixture::new();
    let case = incremental_case();
    let mut execution = fixture.attempt();
    let mut merge = CompiledIncrementalPartitionedMerge::new(
        &fixture.context,
        NodeId::from_index_for_test(1),
        &case.aggregate,
        &case.input_schema,
        &case.output_schema,
        2,
        None,
    )
    .expect("incremental merge");
    merge.push(&mut execution, fixture.partial(&raw_partial(0..100))).expect("first partial");
    merge.push(&mut execution, fixture.partial(&raw_partial(50..150))).expect("second partial");
    let output = merge.finish(&mut execution).expect("merged output");

    let final_merge_rows = fixture
        .metrics
        .snapshot()
        .aggregate_strategy_decisions
        .into_iter()
        .filter(|decision| {
            decision.facts.get("execution_stage").map(String::as_str)
                == Some(AggregateExecutionStage::FinalMerge.as_str())
        })
        .collect::<Vec<_>>();
    assert_eq!(final_merge_rows.len(), 1, "{final_merge_rows:?}");
    let partitions = final_merge_rows[0]
        .facts
        .get("partitioned_merge_non_empty_partition_count")
        .and_then(|count| count.parse::<usize>().ok())
        .expect("non-empty partition count");
    assert!(partitions > 1, "two partials select at least two partitions");
    assert_eq!(fixture.final_groupby_count(), partitions);

    let rows = output_rows(&output.frame);
    let expected = (0..150)
        .map(|key| {
            let copies = if (50..100).contains(&key) { 2 } else { 1 };
            (key, key * copies, key, key, copies)
        })
        .collect::<Vec<_>>();
    assert_eq!(rows, expected);
}

/// A stream of one partial projects it without any groupby; AVG in the spec keeps the groupby.
fn incremental_merge_projects_a_lone_partial_without_a_groupby() {
    let fixture = Fixture::new();
    let case = incremental_case();
    let mut execution = fixture.attempt();
    let mut merge = CompiledIncrementalPartitionedMerge::new(
        &fixture.context,
        NodeId::from_index_for_test(1),
        &case.aggregate,
        &case.input_schema,
        &case.output_schema,
        1,
        None,
    )
    .expect("incremental merge");
    merge.push(&mut execution, fixture.partial(&raw_partial([4, 9, 2]))).expect("lone partial");
    let output = merge.finish(&mut execution).expect("lone partial output");
    assert_eq!(fixture.final_groupby_count(), 0);
    assert_eq!(output_rows(&output.frame), vec![(2, 2, 2, 2, 1), (4, 4, 4, 4, 1), (9, 9, 9, 9, 1)]);
    assert_eq!(
        output.frame.frame().to_record_batch().expect("output download").schema().fields().len(),
        case.output_schema.fields().len()
    );
}

/// Aligned partials of one partition each come from a single partial aggregate. A lone pending
/// slice converts; two slices of the same partition, which may share keys, still merge.
fn aligned_merge_converts_lone_slices_and_merges_shared_keys() {
    let fixture = Fixture::new();
    let case = incremental_case();
    let keys = ["k".to_string()];
    let mut execution = fixture.attempt();
    let mut merge = CompiledIncrementalPartitionedMerge::new_aligned(
        &fixture.context,
        NodeId::from_index_for_test(1),
        &case.aggregate,
        &case.input_schema,
        &case.output_schema,
        &keys,
    )
    .expect("aligned merge");
    merge.push_aligned(&mut execution, 0, 2, &keys, fixture.partial(&raw_partial([1, 3]))).expect("partition 0");
    merge.push_aligned(&mut execution, 1, 2, &keys, fixture.partial(&raw_partial([2, 4]))).expect("partition 1");
    merge.push_aligned(&mut execution, 1, 2, &keys, fixture.partial(&raw_partial([4, 6]))).expect("partition 1 again");
    let states = merge.finish_partition_states(&mut execution).expect("partition states");
    assert_eq!(fixture.final_groupby_count(), 1, "only partition 1 has two slices to merge");
    let rows = states.partitions.iter().map(|(index, state)| (*index, output_rows(&state.frame))).collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![(0, vec![(1, 1, 1, 1, 1), (3, 3, 3, 3, 1)]), (1, vec![(2, 2, 2, 2, 1), (4, 8, 4, 4, 2), (6, 6, 6, 6, 1)]),]
    );
}

#[test]
fn final_merge_groupby_skip_cases() {
    run_named_cases(&[
        (
            "exec::bounded::aggregate::final_merge::execution_tests::sum_conversion_matches_groupby_for_type_preserving_states",
            sum_conversion_matches_groupby_for_type_preserving_states,
        ),
        (
            "exec::bounded::aggregate::final_merge::execution_tests::min_max_conversion_matches_groupby_for_every_state_type",
            min_max_conversion_matches_groupby_for_every_state_type,
        ),
        (
            "exec::bounded::aggregate::final_merge::execution_tests::count_conversion_casts_integer_states_like_the_groupby",
            count_conversion_casts_integer_states_like_the_groupby,
        ),
        (
            "exec::bounded::aggregate::final_merge::execution_tests::mixed_conversion_matches_groupby_with_composite_keys_and_reordered_state",
            mixed_conversion_matches_groupby_with_composite_keys_and_reordered_state,
        ),
        (
            "exec::bounded::aggregate::final_merge::execution_tests::conversion_rejects_shapes_whose_groupby_changes_the_state",
            conversion_rejects_shapes_whose_groupby_changes_the_state,
        ),
        (
            "exec::bounded::aggregate::final_merge::execution_tests::incremental_merge_groups_slices_of_a_multi_partial_batch",
            incremental_merge_groups_slices_of_a_multi_partial_batch,
        ),
        (
            "exec::bounded::aggregate::final_merge::execution_tests::incremental_merge_projects_a_lone_partial_without_a_groupby",
            incremental_merge_projects_a_lone_partial_without_a_groupby,
        ),
        (
            "exec::bounded::aggregate::final_merge::execution_tests::aligned_merge_converts_lone_slices_and_merges_shared_keys",
            aligned_merge_converts_lone_slices_and_merges_shared_keys,
        ),
    ]);
}
