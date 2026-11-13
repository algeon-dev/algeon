use std::{path::PathBuf, sync::Arc};

use query_planning::expr::ColumnRef;

use super::*;

fn left_match_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, true), Field::new("qty", DataType::Int32, true)]))
}

fn write_left_match_file(path: &Path, keys: &[Option<i32>], quantities: &[Option<i32>]) {
    let schema = left_match_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys.to_vec())), Arc::new(Int32Array::from(quantities.to_vec()))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, schema, batch, keys.len().max(1));
}

/// A residual-free semi or anti join that builds its left input; every right
/// file is a separate probe chunk.
fn left_match_plan(left: &Path, rights: &[PathBuf], join_type: JoinType, null_equality: JoinNullEquality) -> QueryPlan {
    let schema = left_match_schema();
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(join_left_source(left, &schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source::from_files(
        rights
            .iter()
            .map(|path| LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len()))
            .collect(),
        Arc::clone(&schema),
        vec!["order_id".to_string()],
    )));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(join_type, vec![join_order_id_key()])
                .with_null_equality(null_equality)
                .with_build_side(JoinSide::Left)
                .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty")]),
            Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, true)])),
        )
        .unwrap();
    builder.finish_with_host_sink(join).unwrap()
}

/// Row-at-a-time reference: a left row is selected by a semi join when some
/// right key equals its key, and by an anti join otherwise. A null key equals
/// another null only under `NullsMatch`.
fn expected_left_match(
    left_keys: &[Option<i32>],
    left_quantities: &[Option<i32>],
    right_keys: &[Option<i32>],
    anti: bool,
    null_equality: JoinNullEquality,
) -> Vec<Option<i32>> {
    let mut expected = left_keys
        .iter()
        .zip(left_quantities)
        .filter(|(key, _)| {
            let matched = right_keys.iter().any(|right| match (key, right) {
                (Some(left), Some(right)) => left == right,
                (None, None) => null_equality == JoinNullEquality::NullsMatch,
                _ => false,
            });
            matched != anti
        })
        .map(|(_, quantity)| *quantity)
        .collect::<Vec<_>>();
    expected.sort();
    expected
}

#[test]
fn bounded_native_hash_left_match_folds_every_right_chunk_before_selecting_left_rows() {
    let dir = unique_temp_dir("query-engine-hash-left-match");
    let left_path = dir.join("left.parquet");
    let empty_path = dir.join("empty.parquet");
    let left_keys = [Some(1), Some(1), Some(2), Some(3), None, Some(4), Some(5), Some(6)];
    let left_quantities = [Some(10), Some(11), Some(20), Some(30), Some(40), None, Some(50), Some(60)];
    write_left_match_file(&left_path, &left_keys, &left_quantities);
    write_left_match_file(&empty_path, &[], &[]);
    // Key 1 repeats within and across chunks, key 5 matches only in the last
    // chunk, and null keys appear on both sides.
    let right_chunks = [
        vec![Some(99), Some(1), Some(1), None, Some(98)],
        vec![Some(2), Some(1), Some(97), Some(2)],
        vec![Some(96), None, Some(5)],
    ];
    let right_paths = right_chunks
        .iter()
        .enumerate()
        .map(|(index, keys)| {
            let path = dir.join(format!("right-{index}.parquet"));
            write_left_match_file(&path, keys, &vec![Some(0); keys.len()]);
            path
        })
        .collect::<Vec<_>>();
    let right_keys = right_chunks.concat();

    const CAP: u64 = 16 * 1024 * 1024;
    for null_equality in [JoinNullEquality::NullsDoNotMatch, JoinNullEquality::NullsMatch] {
        for join_type in [JoinType::LeftSemi, JoinType::LEFT_ANTI] {
            let anti = join_type != JoinType::LeftSemi;
            let cases = [
                (
                    left_path.as_path(),
                    right_paths.as_slice(),
                    expected_left_match(&left_keys, &left_quantities, &right_keys, anti, null_equality),
                ),
                (
                    left_path.as_path(),
                    std::slice::from_ref(&empty_path),
                    expected_left_match(&left_keys, &left_quantities, &[], anti, null_equality),
                ),
                (empty_path.as_path(), right_paths.as_slice(), vec![]),
            ];
            for (left, rights, expected) in cases {
                let label = format!("{join_type:?}/{null_equality:?}/{left:?}/{rights:?}");
                let plan = left_match_plan(left, rights, join_type.clone(), null_equality);
                let context = backend_context_with_grant(CAP);
                let admitted = context.admit_plan(&plan).unwrap();
                let admitted_plan = admit_query_plan(
                    &plan,
                    admitted.statistical_aggregate_execution_mode(),
                    admitted.planning_policy(),
                )
                .unwrap();
                let outcome = execute_query_plan_direct(
                    admitted.context.context(),
                    QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&GpuResidentInputMap::new()),
                );
                let (result, metrics, _) = outcome.into_parts();
                let metrics = metrics.snapshot();
                let output = result.unwrap_or_else(|error| panic!("{label}: {error:?}"));
                let QueryOutput::HostArrow(batch) = &output else {
                    panic!("expected host output");
                };
                let mut observed = int32_array(batch, 0).iter().collect::<Vec<_>>();
                observed.sort();
                assert_eq!(observed, expected, "{label}");
                if left == left_path.as_path() && rights.len() > 1 {
                    let join = metrics.operators.iter().find_map(|operator| operator.join.as_ref()).unwrap();
                    assert_eq!(join.execution_strategy, "bounded_hash_left_match_join", "{label}");
                    assert_eq!(join.build_side, "left", "{label}");
                    assert_eq!(join.probe_rows, right_keys.len() as u64, "{label}");
                    assert!(join.hash_table_bytes >= left_keys.len() as u64, "{label}: {join:?}");
                    assert!(metrics.source_chunks_committed > rights.len() as u64, "{label}: {metrics:?}");
                }
                let usage = admitted.attempt_allocation_domain_usage().unwrap().unwrap();
                assert_eq!(usage.current_bytes, 0, "{label}: completed join retained GPU state: {usage:?}");
                assert_eq!(usage.current_allocations, 0, "{label}");
                admitted.finish();
            }
        }
    }
    fs::remove_dir_all(dir).unwrap();
}
