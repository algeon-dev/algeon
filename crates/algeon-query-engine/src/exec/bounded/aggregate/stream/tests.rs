//! GPU cases for grouped aggregates over resident frame streams. High-cardinality frames go
//! through the partitioned accumulation, from an eager or a lazy stream, and each group-key
//! partition is aggregated once; a lazy stream is pulled one frame at a time. Low-cardinality
//! frames keep one partial per frame.

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
}

impl Fixture {
    fn new() -> Self {
        let memory_control_policy = MemoryControlPolicy {
            device_budget_bytes: DEVICE_BUDGET_BYTES,
            ..StandaloneConfig::default().memory_control_policy
        };
        Self {
            context: ExecutionContext::standalone(StandaloneConfig {
                memory_control_policy,
                ..StandaloneConfig::default()
            })
            .expect("standalone execution context"),
            metrics: ExecutionMetrics::detached("frame_stream_aggregate_tests"),
        }
    }

    fn attempt(&self) -> BoundedExecutionAttempt<'_> {
        BoundedExecutionAttempt::new(&self.context, DEVICE_BUDGET_BYTES, &self.metrics)
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
struct PulledPairFrames {
    next: VecDeque<i64>,
    slices_at_pull: Arc<Mutex<Vec<usize>>>,
}

impl ResidentFrameStreamSource for PulledPairFrames {
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        let index = self.next.pop_front()?;
        self.slices_at_pull
            .lock()
            .expect("pull record")
            .push(operation_count(execution.metrics(), PARTITION_SLICE_OPERATION));
        Some(upload(execution.context(), execution.metrics(), &pair_frame(index)))
    }
}

/// A lazy stream without byte hints is pulled one frame at a time: each frame is partitioned
/// before the next is pulled, and each group-key partition is aggregated once at the end.
fn lazy_high_cardinality_stream_partitions_each_frame_before_the_next_pull() {
    let fixture = Fixture::new();
    let (aggregate, schema) = distinct_pairs();
    let mut execution = fixture.attempt();
    let slices_at_pull = Arc::new(Mutex::new(Vec::new()));
    let inputs = ResidentFrameStreamPayload::from_producer(
        None,
        None,
        (FRAME_COUNT * FRAME_ROWS) as u64,
        2,
        Box::new(PulledPairFrames { next: (0..FRAME_COUNT).collect(), slices_at_pull: Arc::clone(&slices_at_pull) }),
    );
    let output = execute_compiled_bounded_aggregate_from_stream(
        &mut execution,
        NodeId::from_index_for_test(1),
        &aggregate,
        &schema,
        &schema,
        inputs,
        false,
    )
    .expect("aggregate");
    let frames = output.into_resident_frames(&mut execution).expect("output frames");
    let pairs = output_pairs(frames);
    assert_each_pair_aggregated_once(&fixture, &pairs);
    let slices_at_pull = slices_at_pull.lock().expect("pull record").clone();
    assert_eq!(slices_at_pull.len(), FRAME_COUNT as usize);
    assert!(slices_at_pull.windows(2).all(|pulls| pulls[0] < pulls[1]), "{slices_at_pull:?}");
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
            "exec::bounded::aggregate::stream::tests::lazy_high_cardinality_stream_partitions_each_frame_before_the_next_pull",
            lazy_high_cardinality_stream_partitions_each_frame_before_the_next_pull,
        ),
        (
            "exec::bounded::aggregate::stream::tests::low_cardinality_stream_keeps_one_partial_per_frame",
            low_cardinality_stream_keeps_one_partial_per_frame,
        ),
    ]);
}
