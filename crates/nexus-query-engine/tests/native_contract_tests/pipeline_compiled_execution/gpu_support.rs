//! Minimal shared support for the GPU-only compiled-pipeline target.
//!
//! Keep this independent of bounded-join admission fixtures: those are CPU
//! contract tests and must not be compiled into the GPU execution binary.

use super::*;

type Int32Triple = (Option<i32>, Option<i32>, Option<i32>);
type Int32Quad = (Option<i32>, Option<i32>, Option<i32>, Option<i32>);

pub(super) fn runtime_visible_gate_for_task_kind(
    pipeline: &CompiledPipeline,
    task_kind: PipelineTaskKind,
    reservation_kind: ReservationKind,
) -> Option<&ReservationGate> {
    let task = pipeline.tasks().iter().find(|task| task.kind() == task_kind)?;
    task.gates()
        .iter()
        .chain(
            pipeline
                .edges()
                .iter()
                .filter(|edge| edge.from() == task.id() || edge.to() == task.id())
                .flat_map(|edge| edge.gates()),
        )
        .chain(
            pipeline
                .barriers()
                .iter()
                .filter(|barrier| {
                    barrier.upstream_tasks().contains(&task.id()) || barrier.downstream_tasks().contains(&task.id())
                })
                .flat_map(|barrier| barrier.gates()),
        )
        .chain((pipeline.sink().input() == task.id()).then_some(pipeline.sink().gates()).into_iter().flatten())
        .find_map(|gate_id| {
            pipeline.gates().iter().find(|gate| gate.id() == *gate_id && gate.reservation_kind() == reservation_kind)
        })
}

pub(super) fn literal_expr_for_type(data_type: &DataType) -> Expr {
    let literal = match data_type {
        DataType::Decimal128(precision, scale) => {
            LiteralValue::Decimal128 { raw: 12_345, precision: *precision, scale: *scale }
        },
        _ => LiteralValue::Int32(12_345),
    };
    Expr::literal(literal).alias("secret_literal")
}

pub(super) fn int32_values(batch: &RecordBatch, column: usize) -> Vec<Option<i32>> {
    let array = batch.column(column).as_any().downcast_ref::<Int32Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

pub(super) fn int64_values(batch: &RecordBatch, column: usize) -> Vec<Option<i64>> {
    let array = batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

pub(super) fn int32_int64_pairs_sorted(
    batch: &RecordBatch,
    left_column: usize,
    right_column: usize,
) -> Vec<(Option<i32>, Option<i64>)> {
    let mut pairs =
        int32_values(batch, left_column).into_iter().zip(int64_values(batch, right_column)).collect::<Vec<_>>();
    pairs.sort();
    pairs
}

pub(super) fn string_int64_pairs_sorted(
    batch: &RecordBatch,
    left_column: usize,
    right_column: usize,
) -> Vec<(Option<String>, Option<i64>)> {
    let array = batch.column(left_column).as_any().downcast_ref::<StringArray>().unwrap();
    let left = (0..array.len())
        .map(|index| if array.is_null(index) { None } else { Some(array.value(index).to_string()) })
        .collect::<Vec<_>>();
    let mut pairs = left.into_iter().zip(int64_values(batch, right_column)).collect::<Vec<_>>();
    pairs.sort();
    pairs
}

pub(super) fn int32_triples_sorted(
    batch: &RecordBatch,
    first_column: usize,
    second_column: usize,
    third_column: usize,
) -> Vec<Int32Triple> {
    let mut rows = int32_values(batch, first_column)
        .into_iter()
        .zip(int32_values(batch, second_column))
        .zip(int32_values(batch, third_column))
        .map(|((first, second), third)| (first, second, third))
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

pub(super) fn int32_quads_sorted(
    batch: &RecordBatch,
    first_column: usize,
    second_column: usize,
    third_column: usize,
    fourth_column: usize,
) -> Vec<Int32Quad> {
    let mut rows = int32_values(batch, first_column)
        .into_iter()
        .zip(int32_values(batch, second_column))
        .zip(int32_values(batch, third_column))
        .zip(int32_values(batch, fourth_column))
        .map(|(((first, second), third), fourth)| (first, second, third, fourth))
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

pub(super) fn assert_task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind, expected: usize) {
    assert_eq!(
        pipeline.tasks().iter().filter(|task| task.kind() == kind).count(),
        expected,
        "unexpected count for {kind:?} tasks"
    );
}

pub(super) fn bounded_context(runtime: Arc<cudf::Runtime>) -> ExecutionContext {
    bounded_context_with_mode(runtime, NativePipelineExecutionMode::StrictBounded)
}

pub(super) fn bounded_context_with_mode(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
) -> ExecutionContext {
    bounded_context_builder(runtime, mode).build().unwrap()
}

pub(super) fn bounded_context_with_mode_and_memory_history(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
    memory_history: Arc<AdaptiveMemoryHistory>,
) -> ExecutionContext {
    bounded_context_builder(runtime, mode).with_adaptive_memory_history(memory_history).build().unwrap()
}

fn bounded_context_builder(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
) -> nexus_query_engine::runtime::ExecutionContextBuilder {
    ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_native_pipeline_execution_mode(mode)
}

pub(super) fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
