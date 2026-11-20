//! GPU cases for grouped aggregates over resident frame streams. High-cardinality frames go
//! through the partitioned accumulation, from an eager or a lazy stream, and each group-key
//! partition is aggregated once. A lazy stream is held unstaged up to what the retained-state
//! ledger accepts, then staged in order. Low-cardinality frames keep one partial per frame.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use arrow::{
    array::{ArrayRef, Int64Array},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::testing::run_named_cases;
use query_planning::expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr};
use query_runtime::runtime::StandaloneConfig;

use super::{
    AggregateNode, ArrowSchemaRef, BoundedExecutionAttempt, ExecutionContext, GpuDataFrame, NodeId,
    ResidentFrameStreamPayload, ResidentGpuFrame, Result, execute_compiled_bounded_aggregate_from_stream,
};
use crate::exec::bounded::{
    DataFrame, ExecutionMetrics, MemoryControlPolicy, compiled::ResidentFrameStreamSource, observe_device_memory_bytes,
};

const FINAL_GROUPBY_OPERATION: &str = "query_engine.final_merge.aggregate";
const PARTITION_SLICE_OPERATION: &str = "query_engine.bounded.aggregate.source_adjacent_hash_partition_slice";
const DEVICE_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const FRAME_ROWS: i64 = 1_000_000;
const FRAME_COUNT: i64 = 3;

struct Fixture {
    context: ExecutionContext,
    metrics: ExecutionMetrics,
    device_budget_bytes: u64,
}

impl Fixture {
    fn new() -> Self {
        Self::with_device_budget(DEVICE_BUDGET_BYTES)
    }

    fn with_device_budget(device_budget_bytes: u64) -> Self {
        let memory_control_policy =
            MemoryControlPolicy { device_budget_bytes, ..StandaloneConfig::default().memory_control_policy };
        Self {
            context: ExecutionContext::standalone(StandaloneConfig {
                memory_control_policy,
                ..StandaloneConfig::default()
            })
            .expect("standalone execution context"),
            metrics: ExecutionMetrics::detached("frame_stream_aggregate_tests"),
            device_budget_bytes,
        }
    }

    fn attempt(&self) -> BoundedExecutionAttempt<'_> {
        BoundedExecutionAttempt::new(&self.context, self.device_budget_bytes, &self.metrics)
    }

    /// Group-key partition count the partitioned accumulation recorded on its partials.
    fn partition_count(&self) -> Option<u64> {
        self.metrics.snapshot().aggregate_strategy_decisions.iter().find_map(|decision| {
            (decision.facts.get("source_adjacent_partition_strategy").map(String::as_str) == Some("group_key"))
                .then(|| decision.facts.get("source_adjacent_partition_count")?.parse().ok())
                .flatten()
        })
    }

    fn operation_count(&self, operation: &str) -> usize {
        operation_count(&self.metrics, operation)
    }
}

fn operation_count(metrics: &ExecutionMetrics, operation: &str) -> usize {
    metrics.snapshot().budgeted_execution_events.iter().filter(|event| event.operation == operation).count()
}

fn upload(context: &ExecutionContext, metrics: &ExecutionMetrics, batch: &RecordBatch) -> Result<ResidentGpuFrame> {
    let runtime = context.runtime()?;
    let frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, batch).expect("frame upload"));
    let bytes = observe_device_memory_bytes(&frame, metrics)?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, bytes))
}

fn int64(values: impl Iterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(values))
}

/// Group-only aggregate over the pair (`src`, `dst`), as `SELECT DISTINCT src, dst` lowers.
fn distinct_pairs() -> (AggregateNode, ArrowSchemaRef) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", ArrowDataType::Int64, false),
        Field::new("dst", ArrowDataType::Int64, false),
    ]));
    let spec = AggregateSpec::new(vec![Expr::column("src"), Expr::column("dst")], Vec::new());
    (AggregateNode { input: NodeId::from_index_for_test(0), spec }, schema)
}

/// Frame `index` of the pair stream: every frame holds each of the `FRAME_ROWS` pairs once, so
/// each pair occurs in every frame and only an aggregate across frames removes the copies.
fn pair_frame(index: i64) -> RecordBatch {
    let rows = index * FRAME_ROWS..(index + 1) * FRAME_ROWS;
    let schema = distinct_pairs().1;
    RecordBatch::try_new(
        schema,
        vec![int64(rows.clone().map(|row| row % FRAME_ROWS)), int64(rows.map(|row| row % FRAME_ROWS + 1_000_000_000))],
    )
    .expect("pair frame")
}

/// Sorted (`src`, `dst`) rows of an aggregate output.
fn output_pairs(frames: Vec<ResidentGpuFrame>) -> Vec<(i64, i64)> {
    let mut pairs = Vec::new();
    for frame in frames {
        let batch = frame.frame.frame().to_record_batch().expect("output download");
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .expect("output column")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64")
                .clone()
        };
        let (src, dst) = (column("src"), column("dst"));
        pairs.extend(src.values().iter().copied().zip(dst.values().iter().copied()));
    }
    pairs.sort_unstable();
    pairs
}

fn assert_each_pair_aggregated_once(fixture: &Fixture, pairs: &[(i64, i64)]) {
    let expected = (0..FRAME_ROWS).map(|key| (key, key + 1_000_000_000)).collect::<Vec<_>>();
    assert_eq!(pairs, expected.as_slice());
    let metrics = fixture.metrics.snapshot();
    assert_eq!(metrics.partial_aggregate_input_rows, (FRAME_COUNT * FRAME_ROWS) as u64);
    // Every pair reaches the partials once, so the partials hold the result and no final groupby
    // runs; one partial per frame would hold each pair once per frame.
    assert_eq!(metrics.partial_aggregate_rows, FRAME_ROWS as u64);
    assert_eq!(fixture.operation_count(FINAL_GROUPBY_OPERATION), 0);
}

/// An eager stream of three high-cardinality frames aggregates each group-key partition once,
/// from the slices of every frame.
fn eager_high_cardinality_stream_aggregates_each_partition_once() {
    let fixture = Fixture::new();
    let (aggregate, schema) = distinct_pairs();
    let mut execution = fixture.attempt();
    let frames = (0..FRAME_COUNT)
        .map(|index| upload(&fixture.context, &fixture.metrics, &pair_frame(index)).expect("frame"))
        .collect::<Vec<_>>();
    let output = execute_compiled_bounded_aggregate_from_stream(
        &mut execution,
        NodeId::from_index_for_test(1),
        &aggregate,
        &schema,
        &schema,
        ResidentFrameStreamPayload::new(frames),
        false,
    )
    .expect("aggregate");
    let frames = output.into_resident_frames(&mut execution).expect("output frames");
    let pairs = output_pairs(frames);
    assert_each_pair_aggregated_once(&fixture, &pairs);
    assert!(fixture.metrics.snapshot().aggregate_strategy_decisions.iter().any(|decision| {
        decision.facts.get("source_adjacent_partition_strategy").map(String::as_str) == Some("group_key")
            && decision.facts.get("source_adjacent_max_partition_fan_in").map(String::as_str)
                == Some(FRAME_COUNT.to_string().as_str())
    }));
}

/// Lazy source that uploads one frame per pull and records, at each pull, how many partition
/// slices the consumer had cut so far.
struct PulledFrames {
    next: VecDeque<RecordBatch>,
    slices_at_pull: Arc<Mutex<Vec<usize>>>,
}

impl ResidentFrameStreamSource for PulledFrames {
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        let batch = self.next.pop_front()?;
        self.slices_at_pull
            .lock()
            .expect("pull record")
            .push(operation_count(execution.metrics(), PARTITION_SLICE_OPERATION));
        Some(upload(execution.context(), execution.metrics(), &batch))
    }
}

/// A 1 GiB budget holds 128 MiB of input uncharged and 256 MiB on the retained-state ledger, and
/// sizes 128 MiB partitions; each pair frame is 16 MB.
const HOLD_CASE_DEVICE_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;

/// Runs a lazy stream of `batches` declaring `declared_rows` rows through the aggregate, with the
/// retained-state ledger taken in full first when `ledger_full`.
/// Returns the output frames and the slice count recorded at each pull.
fn run_lazy_stream(
    fixture: &Fixture,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    batches: Vec<RecordBatch>,
    declared_rows: u64,
    ledger_full: bool,
) -> (Vec<ResidentGpuFrame>, Vec<usize>) {
    let mut execution = fixture.attempt();
    let _occupied = ledger_full.then(|| {
        execution
            .charge_retained_join_table(MemoryControlPolicy::retained_state_share_bytes(fixture.device_budget_bytes))
            .expect("an empty ledger accepts its whole share")
    });
    let slices_at_pull = Arc::new(Mutex::new(Vec::new()));
    let inputs = ResidentFrameStreamPayload::from_producer(
        None,
        None,
        declared_rows,
        input_schema.fields().len(),
        Box::new(PulledFrames { next: batches.into(), slices_at_pull: Arc::clone(&slices_at_pull) }),
    );
    let output = execute_compiled_bounded_aggregate_from_stream(
        &mut execution,
        NodeId::from_index_for_test(1),
        aggregate,
        input_schema,
        output_schema,
        inputs,
        false,
    )
    .expect("aggregate");
    let frames = output.into_resident_frames(&mut execution).expect("output frames");
    let slices_at_pull = slices_at_pull.lock().expect("pull record").clone();
    (frames, slices_at_pull)
}

/// A lazy stream is held unstaged while the retained-state ledger accepts its frames, then staged
/// in order, and never held beyond the ledger.
///
/// Twelve 16 MB frames pass the 128 MiB uncharged hold and fit the 256 MiB ledger share, so all of
/// them are pulled before the first is partitioned, and the measured 192 MB sizes two partitions.
/// With the ledger already full, the ninth frame passes the uncharged hold and is refused: the
/// nine pulled frames, scaled to the four-fold declared rows, size six partitions, and each later
/// frame is pulled only after the frames before it were partitioned.
fn lazy_stream_is_held_up_to_the_ledger_then_staged_in_order() {
    let (aggregate, schema) = distinct_pairs();
    let frame_count = 12;
    let expected = (0..FRAME_ROWS).map(|key| (key, key + 1_000_000_000)).collect::<Vec<_>>();

    let fixture = Fixture::with_device_budget(HOLD_CASE_DEVICE_BUDGET_BYTES);
    let (frames, slices_at_pull) = run_lazy_stream(
        &fixture,
        &aggregate,
        &schema,
        &schema,
        (0..frame_count).map(pair_frame).collect(),
        (frame_count * FRAME_ROWS) as u64,
        false,
    );
    assert_eq!(output_pairs(frames), expected);
    assert_eq!(slices_at_pull, vec![0; frame_count as usize]);
    assert_eq!(fixture.partition_count(), Some(2));
    // Each partition is aggregated once, from the slices of every frame.
    let metrics = fixture.metrics.snapshot();
    assert_eq!(metrics.partial_aggregate_input_rows, (frame_count * FRAME_ROWS) as u64);
    assert_eq!(metrics.partial_aggregate_rows, FRAME_ROWS as u64);
    assert_eq!(fixture.operation_count(FINAL_GROUPBY_OPERATION), 0);

    let fixture = Fixture::with_device_budget(HOLD_CASE_DEVICE_BUDGET_BYTES);
    let (frames, slices_at_pull) = run_lazy_stream(
        &fixture,
        &aggregate,
        &schema,
        &schema,
        (0..frame_count).map(pair_frame).collect(),
        (4 * frame_count * FRAME_ROWS) as u64,
        true,
    );
    assert_eq!(output_pairs(frames), expected);
    assert_eq!(fixture.partition_count(), Some(6));
    let held = 9;
    assert!(slices_at_pull[..held].iter().all(|slices| *slices == 0), "{slices_at_pull:?}");
    assert!(slices_at_pull[held - 1..].windows(2).all(|pulls| pulls[0] < pulls[1]), "{slices_at_pull:?}");
}

/// A lazy stream that passes its hold is sized for the partial state its rows produce. Each row
/// is a unique key and a value, 16 bytes, and its partial state holds four SUMs beside the key,
/// about 2.5 times the row. Under a 512 MiB budget, five of six 16 MB frames are pulled before the
/// full ledger refuses at the 64 MiB uncharged hold, and the 96 MB input they estimate needs one
/// 128 MiB partition while its 240 MB of state needs two.
fn lazy_stream_past_its_hold_is_sized_for_wide_partial_state() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowDataType::Int64, false),
        Field::new("v", ArrowDataType::Int64, false),
    ]));
    let sums = (0..4).map(|index| format!("s{index}")).collect::<Vec<_>>();
    let mut output_fields = vec![Field::new("k", ArrowDataType::Int64, false)];
    output_fields.extend(sums.iter().map(|alias| Field::new(alias, ArrowDataType::Int64, true)));
    let output_schema = Arc::new(Schema::new(output_fields));
    let aggregate = AggregateNode {
        input: NodeId::from_index_for_test(0),
        spec: AggregateSpec::new(
            vec![Expr::column("k")],
            sums.iter().map(|alias| AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), alias)).collect(),
        ),
    };
    let frame_count = 6;
    let batches = (0..frame_count)
        .map(|index| {
            let rows = index * FRAME_ROWS..(index + 1) * FRAME_ROWS;
            RecordBatch::try_new(Arc::clone(&input_schema), vec![int64(rows.clone()), int64(rows)])
                .expect("unique-key frame")
        })
        .collect();
    let fixture = Fixture::with_device_budget(512 * 1024 * 1024);
    let (frames, _) = run_lazy_stream(
        &fixture,
        &aggregate,
        &input_schema,
        &output_schema,
        batches,
        (frame_count * FRAME_ROWS) as u64,
        true,
    );
    assert_eq!(fixture.partition_count(), Some(2));
    let mut output_rows = 0;
    for frame in frames {
        let batch = frame.frame.frame().to_record_batch().expect("output download");
        let keys =
            batch.column_by_name("k").expect("key").as_any().downcast_ref::<Int64Array>().expect("int64").clone();
        for alias in &sums {
            let sum = batch.column_by_name(alias).expect("sum").as_any().downcast_ref::<Int64Array>().expect("int64");
            assert_eq!(sum.values(), keys.values());
        }
        output_rows += batch.num_rows();
    }
    assert_eq!(output_rows as i64, frame_count * FRAME_ROWS);
}

/// Low-cardinality frames are each reduced whole, as one partial per frame, and the partials are
/// partitioned after the reduction. Sums and counts are exact.
fn low_cardinality_stream_keeps_one_partial_per_frame() {
    let fixture = Fixture::new();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowDataType::Int64, false),
        Field::new("v", ArrowDataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowDataType::Int64, false),
        Field::new("total", ArrowDataType::Int64, true),
        Field::new("n", ArrowDataType::Int64, false),
    ]));
    let aggregate = AggregateNode {
        input: NodeId::from_index_for_test(0),
        spec: AggregateSpec::new(
            vec![Expr::column("k")],
            vec![
                AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total"),
                AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "n"),
            ],
        ),
    };
    let mut execution = fixture.attempt();
    let frames = (0..FRAME_COUNT)
        .map(|index| {
            let rows = index * FRAME_ROWS..(index + 1) * FRAME_ROWS;
            let batch = RecordBatch::try_new(
                Arc::clone(&input_schema),
                vec![int64(rows.clone().map(|row| row % 4)), int64(rows)],
            )
            .expect("low-cardinality frame");
            upload(&fixture.context, &fixture.metrics, &batch).expect("frame")
        })
        .collect::<Vec<_>>();
    let output = execute_compiled_bounded_aggregate_from_stream(
        &mut execution,
        NodeId::from_index_for_test(1),
        &aggregate,
        &input_schema,
        &output_schema,
        ResidentFrameStreamPayload::new(frames),
        false,
    )
    .expect("aggregate");
    let frames = output.into_resident_frames(&mut execution).expect("output frames");
    let mut rows = Vec::new();
    for frame in frames {
        let batch = frame.frame.frame().to_record_batch().expect("output download");
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .expect("output column")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64")
                .clone()
        };
        let (k, total, n) = (column("k"), column("total"), column("n"));
        rows.extend((0..batch.num_rows()).map(|row| (k.value(row), total.value(row), n.value(row))));
    }
    rows.sort_unstable();
    let all_rows = FRAME_COUNT * FRAME_ROWS;
    let expected = (0..4)
        .map(|key| {
            let values = (0..all_rows).filter(|row| row % 4 == key);
            (key, values.clone().sum::<i64>(), values.count() as i64)
        })
        .collect::<Vec<_>>();
    assert_eq!(rows, expected);
    let metrics = fixture.metrics.snapshot();
    assert_eq!(metrics.partial_aggregate_count, FRAME_COUNT as u64);
    assert_eq!(metrics.partial_aggregate_input_rows, all_rows as u64);
    let local_reductions = metrics
        .aggregate_strategy_decisions
        .iter()
        .filter(|decision| {
            decision.facts.get("source_adjacent_partition_strategy").map(String::as_str) == Some("local_reduction")
        })
        .count();
    assert_eq!(local_reductions, FRAME_COUNT as usize);
}

#[test]
fn frame_stream_aggregate_cases() {
    run_named_cases(&[
        (
            "exec::bounded::aggregate::stream::tests::eager_high_cardinality_stream_aggregates_each_partition_once",
            eager_high_cardinality_stream_aggregates_each_partition_once,
        ),
        (
            "exec::bounded::aggregate::stream::tests::lazy_stream_is_held_up_to_the_ledger_then_staged_in_order",
            lazy_stream_is_held_up_to_the_ledger_then_staged_in_order,
        ),
        (
            "exec::bounded::aggregate::stream::tests::lazy_stream_past_its_hold_is_sized_for_wide_partial_state",
            lazy_stream_past_its_hold_is_sized_for_wide_partial_state,
        ),
        (
            "exec::bounded::aggregate::stream::tests::low_cardinality_stream_keeps_one_partial_per_frame",
            low_cardinality_stream_keeps_one_partial_per_frame,
        ),
    ]);
}
