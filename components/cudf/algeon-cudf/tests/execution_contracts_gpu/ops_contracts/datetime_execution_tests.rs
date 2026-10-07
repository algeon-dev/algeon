use cudf::{Exec, Runtime, TypeId};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
    run_named_cases,
};

fn timestamp_us_series_in_runtime(runtime: &Runtime, values: &[i64]) -> cudf::Result<cudf::Series> {
    runtime
        .series(values)?
        .on(Exec::RETAINED)
        .astype(TypeId::DurationMicroseconds)?
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampMicroseconds)
}

fn test_datetime_extract_in_uses_explicit_execution_context_and_output_provenance() {
    let input_runtime = Runtime::builder().build().unwrap();
    let override_runtime = Runtime::builder().build().unwrap();
    let ts = timestamp_us_series_in_runtime(&input_runtime, &[0, 912_518_055_123_456]).unwrap();
    let ctx = override_runtime.exec();
    let expected_placement = override_runtime.placement();
    let expected_stream = expected_placement.stream_debug;
    let expected_mr = expected_placement.memory_resource_debug;

    reset_exec_observation();
    let result = ts.on(ctx).dt().year().unwrap();

    let placement = result.placement();
    assert_eq!(placement.stream_debug, expected_stream);
    assert_eq!(placement.memory_resource_debug, expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.to_vec_strict::<i16>().unwrap(), vec![1970, 1998]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn datetime_execution_tests_cases() {
    run_named_cases(&[(
        "test_datetime_extract_in_uses_explicit_execution_context_and_output_provenance",
        test_datetime_extract_in_uses_explicit_execution_context_and_output_provenance,
    )]);
}
