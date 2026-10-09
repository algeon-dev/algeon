use cudf::{StringSliceOptions, testing::RawHandleTestingExt};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, sample_runtime,
};

fn test_str_like_and_slice_in_use_explicit_execution_context() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let s = data_rt.series(&["abc", "axc", "abbc"]).unwrap();
    let ctx = ctx_rt.exec();
    let expected_stream = ctx_rt.stream().testing_raw_handle_value();
    let expected_mr = ctx_rt.memory_resource().testing_raw_handle_value();

    reset_exec_observation();
    let like = s.on(ctx).str().like("a_c").unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), expected_stream);
    assert_eq!(like.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);

    reset_exec_observation();
    let sliced = s.on(ctx).str().slice(StringSliceOptions::new(Some(1), Some(3))).unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(sliced.to_strings_strict().unwrap(), vec!["bc".to_string(), "xc".to_string(), "bb".to_string()]);
}

pub(crate) const CASES: &[(&str, fn())] = &[(
    "ops_contracts::strings_execution_tests::test_str_like_and_slice_in_use_explicit_execution_context",
    test_str_like_and_slice_in_use_explicit_execution_context,
)];
