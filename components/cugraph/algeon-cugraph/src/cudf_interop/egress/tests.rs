use cudf::{DataFrame, Exec, Runtime, TypeId};
use cudf_sys::test_support;

use super::canonicalize_component_labels;

#[test]
fn canonical_labels_preserve_order_width_and_nulls_without_host_materialization() {
    let runtime = Runtime::new().unwrap();
    for type_id in [TypeId::Int32, TypeId::Int64] {
        for labels in [vec![Some(8i32), Some(3), Some(8), None, Some(3)], vec![None; 5]] {
            let vertex = match type_id {
                TypeId::Int32 => runtime.series(&[90i32, -10, 30, 500, 70]).unwrap(),
                TypeId::Int64 => runtime.series(&[9_000_000_000i64, -10, 30, 500, 70]).unwrap(),
                _ => unreachable!(),
            };
            let label = runtime.series_opt(&labels).unwrap();
            let frame = DataFrame::from_series([("vertex", vertex.clone()), ("label", label)]).unwrap();
            // SAFETY: the test hook resets thread-local observation state without borrowing device storage.
            unsafe { test_support::cudf_test_exec_reset_observation() };
            let result = canonicalize_component_labels(frame, "canonical_labels_test").unwrap();
            let mut sync_sequence = 0;
            // SAFETY: sync_sequence is a live, writable output for the thread-local observation hook.
            let status = unsafe { test_support::cudf_test_exec_last_sync_sequence(&mut sync_sequence) };
            assert_eq!(status, cudf_sys::CUDF_SUCCESS);
            assert_eq!(sync_sequence, 0, "canonicalization must not materialize or upload host columns");
            assert_eq!(result.num_rows(), labels.len());
            let actual_vertex = result.series("vertex").unwrap();
            let actual_label = result.series("label").unwrap();
            assert_eq!(actual_label.dtype().unwrap().id(), type_id);
            let expected = if labels[0].is_some() {
                vec![Some(30i64), Some(-10), Some(30), None, Some(-10)]
            } else {
                vec![None; 5]
            };
            let actual_label = actual_label.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
            assert_eq!(actual_label.to_optional_vec::<i64>().unwrap(), expected);
            assert_eq!(
                actual_vertex.on(Exec::RETAINED).astype(TypeId::Int64).unwrap().to_vec_strict::<i64>().unwrap(),
                vertex.on(Exec::RETAINED).astype(TypeId::Int64).unwrap().to_vec_strict::<i64>().unwrap()
            );
        }
    }
}
