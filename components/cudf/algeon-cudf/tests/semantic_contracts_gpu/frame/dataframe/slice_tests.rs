use std::sync::Arc;

use arrow::{
    array::{ArrayRef, ListArray, StringArray},
    datatypes::Int32Type,
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, ErrorCode, ErrorKind, Exec, ExecutionBudgetRequest,
    frame::dataframe::GpuRecordBatch,
    testing::{
        dataframe::{
            trigger_dataframe_row_range_end_limit_error_for_test, trigger_dataframe_row_range_overflow_error_for_test,
            trigger_dataframe_row_range_start_limit_error_for_test,
        },
        prelude::*,
    },
};

use crate::common::{
    column_from_slice, dataframe_from_columns, error_context_value, execution_budget_token, int64_array,
    nested_dataframe, struct_array, utf8_array, with_test_runtime,
};

fn sample_df() -> DataFrame {
    let ids = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let vals = column_from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    dataframe_from_columns(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap()
}

fn test_head_basic() {
    let df = sample_df();
    let h = df.on(Exec::RETAINED).head(2).unwrap();

    assert_eq!(h.num_rows(), 2);
    assert_eq!(h.num_columns(), 2);
    assert_eq!(h.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20]);
    assert_eq!(h.series("val").unwrap().to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0]);
}

fn test_head_zero_rows() {
    let df = sample_df();
    let h = df.on(Exec::RETAINED).head(0).unwrap();

    assert_eq!(h.num_rows(), 0);
    assert_eq!(h.num_columns(), 2);
    assert_eq!(h.column_names(), vec!["id", "val"]);

    let text: ArrayRef = Arc::new(StringArray::from(vec![Some("prefix"), None, Some("value"), Some("")]));
    let lists: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>([
        Some(vec![Some(10)]),
        None,
        Some(vec![Some(20), None]),
        Some(vec![]),
    ]));
    let batch = RecordBatch::try_from_iter([("text", text), ("nested", lists)]).unwrap();
    let frame = with_test_runtime(|runtime| DataFrame::from_record_batch(runtime, &batch)).unwrap();
    for (offset, length) in [(1, 2), (0, 0), (4, 0)] {
        let sliced = frame.on(Exec::RETAINED).slice(offset, length).unwrap();
        let actual = GpuRecordBatch::new(sliced, batch.schema()).unwrap().to_host_record_batch().unwrap();
        assert_eq!(actual, batch.slice(offset, length));
        let columns = frame
            .column_names()
            .into_iter()
            .map(|name| {
                let values = frame.series(name).unwrap().on(Exec::RETAINED).slice(offset, length).unwrap();
                (name, values)
            })
            .collect::<Vec<_>>();
        let sliced = DataFrame::from_series(columns).unwrap();
        let actual = GpuRecordBatch::new(sliced, batch.schema()).unwrap().to_host_record_batch().unwrap();
        assert_eq!(actual, batch.slice(offset, length));
    }
    let frame = with_test_runtime(|runtime| {
        nested_dataframe(
            runtime,
            vec![(
                "record",
                struct_array(vec![
                    ("value", int64_array(&[10, 20, 30, 40])),
                    ("text", utf8_array(&["prefix", "one", "two", ""])),
                ]),
            )],
        )
    });
    for length in [0, 2] {
        let sliced = frame.on(Exec::RETAINED).slice(1, length).unwrap();
        let value = frame.series("record").unwrap();
        for record in [sliced.series("record").unwrap(), value.on(Exec::RETAINED).slice(1, length).unwrap()] {
            let view = record.testing_view().unwrap();
            assert_eq!(view.len().unwrap(), length);
            assert_eq!(view.num_children().unwrap(), 2);
            assert_eq!(view.child(0).unwrap().to_vec::<i64>().unwrap(), [20, 30][..length]);
            assert_eq!(view.child(1).unwrap().to_strings_lossy().unwrap(), ["one", "two"][..length]);
        }
    }
}

fn test_head_clamps_to_num_rows() {
    let df = sample_df();
    let h = df.on(Exec::RETAINED).head(100).unwrap();

    assert_eq!(h.num_rows(), 5);
    assert_eq!(h.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30, 40, 50]);
}

fn test_head_preserves_schema() {
    let df = sample_df();
    let h = df.on(Exec::RETAINED).head(3).unwrap();
    assert_eq!(h.column_names(), vec!["id", "val"]);
    assert_eq!(h.column_index("id"), Some(0));
    assert_eq!(h.column_index("val"), Some(1));
}

fn test_tail_basic() {
    let df = sample_df();
    let t = df.on(Exec::RETAINED).tail(2).unwrap();

    assert_eq!(t.num_rows(), 2);
    assert_eq!(t.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![40, 50]);
    assert_eq!(t.series("val").unwrap().to_vec_strict::<f64>().unwrap(), vec![4.0, 5.0]);
}

fn test_tail_clamps_to_num_rows() {
    let df = sample_df();
    let t = df.on(Exec::RETAINED).tail(100).unwrap();

    assert_eq!(t.num_rows(), 5);
    assert_eq!(t.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30, 40, 50]);
}

fn test_tail_zero_rows() {
    let df = sample_df();
    let t = df.on(Exec::RETAINED).tail(0).unwrap();

    assert_eq!(t.num_rows(), 0);
    assert_eq!(t.column_names(), vec!["id", "val"]);
}

fn test_tail_preserves_schema() {
    let df = sample_df();
    let t = df.on(Exec::RETAINED).tail(3).unwrap();
    assert_eq!(t.column_names(), vec!["id", "val"]);
}

fn test_slice_basic() {
    let df = sample_df();
    let s = df.on(Exec::RETAINED).slice(1, 2).unwrap();

    assert_eq!(s.num_rows(), 2);
    assert_eq!(s.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(s.series("val").unwrap().to_vec_strict::<f64>().unwrap(), vec![2.0, 3.0]);
}

fn test_slice_clamps_length() {
    let df = sample_df();
    // offset=3, length=100 → should clamp to rows 3..5
    let s = df.on(Exec::RETAINED).slice(3, 100).unwrap();

    assert_eq!(s.num_rows(), 2);
    assert_eq!(s.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![40, 50]);
}

fn test_slice_zero_length() {
    let df = sample_df();
    let s = df.on(Exec::RETAINED).slice(2, 0).unwrap();

    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}

fn test_slice_preserves_schema() {
    let df = sample_df();
    let s = df.on(Exec::RETAINED).slice(0, 3).unwrap();
    assert_eq!(s.column_names(), vec!["id", "val"]);
    assert_eq!(s.column_index("id"), Some(0));
    assert_eq!(s.column_index("val"), Some(1));
}

fn test_slice_offset_at_end() {
    let df = sample_df();
    // offset == num_rows → empty result
    let s = df.on(Exec::RETAINED).slice(5, 10).unwrap();
    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}

fn test_slice_offset_beyond_end() {
    let df = sample_df();
    // offset > num_rows → empty result
    let s = df.on(Exec::RETAINED).slice(999, 10).unwrap();
    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}

fn test_slice_with_budget_reports_member_accounting_on_rejection() {
    let df = sample_df();
    let token = with_test_runtime(|runtime| {
        execution_budget_token(runtime, ExecutionBudgetRequest::new(1).with_boundary("partition_slice")).unwrap()
    });

    let err = token
        .run("dataframe::slice", |exec| df.on(exec).slice(0, 5))
        .expect_err("one-byte budget should not cover the slice copy");

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(err.budget_exhaustion().expect("budget scope").boundary(), "partition_slice");
    // Regression guard: the slice path must charge the caller's budget token. A raw
    // execution lease would drop this member accounting.
    assert!(err.budget_exhaustion().and_then(|budget| budget.usage()).is_some(), "the member was sampled");
    assert_eq!(
        err.budget_exhaustion().and_then(|budget| budget.usage()).expect("member usage").failed_allocation_count,
        1
    );
    drop(err);
    drop(token);
    with_test_runtime(|runtime| runtime.synchronize()).unwrap();

    let values = with_test_runtime(|runtime| runtime.sequence(0i64, 1i64, 8192)).unwrap();
    let frame = DataFrame::from_series([("value", values.clone())]).unwrap();
    let output_rows = 4096;
    for series_slice in [false, true] {
        let token = with_test_runtime(|runtime| {
            execution_budget_token(
                runtime,
                ExecutionBudgetRequest::new(output_rows * 8 + 1024).with_boundary("contiguous_slice"),
            )
            .unwrap()
        });
        let outcome = token
            .run("slice", |exec| {
                if series_slice {
                    values.on(exec).slice(1, output_rows)
                } else {
                    frame.on(exec).slice(1, output_rows)?.series("value")
                }
            })
            .expect("the output budget covers a contiguous copy without an additional row map");
        assert_eq!(outcome.value().len(), output_rows);
        assert_eq!(outcome.value().to_vec_strict::<i64>().unwrap(), (1..=output_rows as i64).collect::<Vec<_>>());
        drop(outcome);
        drop(token);
        with_test_runtime(|runtime| runtime.synchronize()).unwrap();
    }
}

fn test_testing_hook_locks_dataframe_row_range_start_limit_contract() {
    let err = trigger_dataframe_row_range_start_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::gather_range");
    assert_eq!(err.message(), "row range start exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("start_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), None);
}

fn test_testing_hook_locks_dataframe_row_range_end_limit_contract() {
    let err = trigger_dataframe_row_range_end_limit_error_for_test().unwrap_err();
    let expected_end = (i32::MAX as usize).checked_add(1).unwrap().to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::gather_range");
    assert_eq!(err.message(), "row range end exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("end_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(i32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), Some(expected_end.as_str()));
}

fn test_testing_hook_locks_dataframe_row_range_overflow_contract() {
    let err = trigger_dataframe_row_range_overflow_error_for_test().unwrap_err();
    let expected_end = ((i32::MAX as u128) + (usize::MAX as u128)).to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::gather_range");
    assert_eq!(err.message(), "row range end overflowed usize");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("end_overflowed_usize"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within usize"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(i32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), Some(expected_end.as_str()));
    assert_eq!(error_context_value(&err, "length").as_deref(), None);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("slice_tests::test_head_basic", test_head_basic),
    ("slice_tests::test_head_zero_rows", test_head_zero_rows),
    ("slice_tests::test_head_clamps_to_num_rows", test_head_clamps_to_num_rows),
    ("slice_tests::test_head_preserves_schema", test_head_preserves_schema),
    ("slice_tests::test_tail_basic", test_tail_basic),
    ("slice_tests::test_tail_clamps_to_num_rows", test_tail_clamps_to_num_rows),
    ("slice_tests::test_tail_zero_rows", test_tail_zero_rows),
    ("slice_tests::test_tail_preserves_schema", test_tail_preserves_schema),
    ("slice_tests::test_slice_basic", test_slice_basic),
    ("slice_tests::test_slice_clamps_length", test_slice_clamps_length),
    ("slice_tests::test_slice_zero_length", test_slice_zero_length),
    ("slice_tests::test_slice_preserves_schema", test_slice_preserves_schema),
    ("slice_tests::test_slice_offset_at_end", test_slice_offset_at_end),
    ("slice_tests::test_slice_offset_beyond_end", test_slice_offset_beyond_end),
    (
        "slice_tests::test_slice_with_budget_reports_member_accounting_on_rejection",
        test_slice_with_budget_reports_member_accounting_on_rejection,
    ),
    (
        "slice_tests::test_testing_hook_locks_dataframe_row_range_start_limit_contract",
        test_testing_hook_locks_dataframe_row_range_start_limit_contract,
    ),
    (
        "slice_tests::test_testing_hook_locks_dataframe_row_range_end_limit_contract",
        test_testing_hook_locks_dataframe_row_range_end_limit_contract,
    ),
    (
        "slice_tests::test_testing_hook_locks_dataframe_row_range_overflow_contract",
        test_testing_hook_locks_dataframe_row_range_overflow_contract,
    ),
];
