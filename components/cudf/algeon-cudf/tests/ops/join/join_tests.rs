//! Tests for reusable hash-join handles.
use cudf::{DataFrame, DataFrameHashJoinOptions, DataFrameHashJoinPairIndices, Exec};

use crate::common::sample_runtime;

const JOIN_NO_MATCH: i32 = i32::MIN;

fn sorted_join_index_pairs(indices: &DataFrameHashJoinPairIndices) -> Vec<(i32, i32)> {
    let probe = indices.probe().indices().to_vec_strict::<i32>().unwrap();
    let build = indices.build().indices().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<_> = probe.into_iter().zip(build).collect();
    pairs.sort();
    pairs
}

#[test]
fn test_hash_join_supports_parallel_probes_from_cloned_handle() {
    let rt = sample_runtime();
    let build = DataFrame::from_series([("id", rt.series(&[2i32, 2, 4]).unwrap())]).unwrap();
    let first_probe = DataFrame::from_series([("id", rt.series(&[2i32, 4]).unwrap())]).unwrap();
    let second_probe = DataFrame::from_series([("id", rt.series(&[1i32, 2]).unwrap())]).unwrap();
    let hash_join = build.on(Exec::RETAINED).build_hash_join(["id"], DataFrameHashJoinOptions::new()).unwrap();

    let first_join = hash_join.clone();
    let first = std::thread::spawn(move || {
        let indices = first_join.inner_probe(&first_probe, ["id"], Exec::RETAINED).unwrap();
        sorted_join_index_pairs(&indices)
    });

    let second_join = hash_join;
    let second = std::thread::spawn(move || {
        let indices = second_join.left_probe(&second_probe, ["id"], Exec::RETAINED).unwrap();
        sorted_join_index_pairs(&indices)
    });

    assert_eq!(first.join().unwrap(), vec![(0, 0), (0, 1), (1, 2)]);
    assert_eq!(second.join().unwrap(), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1)]);
}
