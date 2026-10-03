//! Contract tests for execution-domain propagation.
//!
//! This suite intentionally keeps one representative test per operation
//! family. The goal is to protect provenance, execution, and synchronization
//! contracts without locking the library into an API-by-API matrix.
use std::sync::Arc;

use cudf::{
    AggSpec, DataFrame, ErrorCode, ErrorKind, Exec, GroupByAgg, MergeOptions, Runtime, Series, SortOptions,
    core::{
        Column, Table,
        error::{FactKeyName, FactValue},
        memory::DeviceMemoryResource,
    },
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, dataframe_from_columns,
    dependency_wait_call_count_for_stream, explicit_bool_series, explicit_i32_column, explicit_i32_series,
    last_destroyed_memory_resource_value, last_exec_mr_value, last_exec_stream_value,
    last_memory_resource_destroy_sequence, last_sync_sequence, last_sync_stream_value, lock_event_observation,
    reset_dependency_observation, reset_exec_observation, reset_sync_observation, run_named_cases, sample_runtime,
    scalar, series,
};

fn explicit_nullable_i32_series(_name: &str, data: &[i32], validity: &[bool], identity: u64) -> Series {
    Series::from_column(
        column_from_slice_nullable(data, validity).unwrap().testing_rebind_execution_domain_explicit(identity).unwrap(),
    )
    .unwrap()
}

fn explicit_str_series(_name: &str, data: &[&str], identity: u64) -> Series {
    Series::from_column(column_from_strs(data).unwrap().testing_rebind_execution_domain_explicit(identity).unwrap())
        .unwrap()
}

fn assert_column_exec_contract(column: &Column, expected_id: u64, expected_stream: usize, expected_mr: usize) {
    assert_eq!(column.testing_execution_domain_kind(), "explicit");
    assert_eq!(column.testing_execution_domain_id(), expected_id);
    assert_eq!(column.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(column.testing_execution_domain_mr_value(), expected_mr);
}

fn helper_constructors_seed_explicit_runtime_domain() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let expected_id = col.testing_execution_domain_id();
    assert_eq!(col.testing_execution_domain_kind(), "explicit");
    assert_ne!(expected_id, 0);

    let scalar = scalar(7i32).unwrap();
    assert_eq!(scalar.testing_execution_domain_kind(), "explicit");
    assert_eq!(scalar.testing_execution_domain_id(), expected_id);

    let table = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    assert_eq!(table.testing_execution_domain_kind(), "explicit");
    assert_eq!(table.testing_execution_domain_id(), expected_id);

    let series = series::<i32>("id", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([("id", series)]).unwrap();
    assert_eq!(df.testing_execution_domain_kind(), "explicit");
    assert_eq!(df.testing_execution_domain_id(), expected_id);
    assert_eq!(df.series("id").unwrap().as_column().testing_execution_domain_id(), expected_id);
}

fn private_pool_teardown_fences_its_registered_stream_before_resource_release() {
    let _event_observation = lock_event_observation();
    let runtime = Runtime::builder().with_pool_memory_resource(16 * 1024 * 1024, None).build().unwrap();
    let stream = runtime.stream().testing_raw_handle_value();
    let memory_resource = runtime.memory_resource().testing_raw_handle_value();
    let output = runtime.series(&[1_i32, 2, 3]).unwrap();

    // The output's native allocation must be released before the last runtime
    // owner tears down its private pool. Reset after that release so this
    // assertion observes the retirement fence paired with pool destruction.
    drop(output);
    reset_sync_observation();
    drop(runtime);

    assert_eq!(last_sync_stream_value(), stream);
    assert_eq!(last_destroyed_memory_resource_value(), memory_resource);
    assert!(
        last_sync_sequence() < last_memory_resource_destroy_sequence(),
        "private-pool stream retirement must precede native memory-resource destruction"
    );
}

fn private_pool_adaptor_teardown_inherits_the_registered_stream_fence() {
    let _event_observation = lock_event_observation();
    let pool = DeviceMemoryResource::pool(16 * 1024 * 1024, None).unwrap();
    let runtime =
        Runtime::builder().with_memory_resource(pool.with_failure_callback(Arc::new(|_| {})).unwrap()).build().unwrap();
    let stream = runtime.stream().testing_raw_handle_value();
    let memory_resource = runtime.memory_resource().testing_raw_handle_value();
    let output = runtime.series(&[1_i32, 2, 3]).unwrap();

    drop(output);
    reset_sync_observation();
    drop(runtime);

    assert_eq!(last_sync_stream_value(), stream);
    assert_eq!(last_destroyed_memory_resource_value(), memory_resource);
    assert!(
        last_sync_sequence() < last_memory_resource_destroy_sequence(),
        "an adaptor over a private pool must retain the pool retirement fence"
    );

    drop(pool);
}

fn borrowed_override_keeps_device_and_binds_the_override_handles() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let input = data_rt.series(&[3i32, 1, 2]).unwrap();
    let input_placement = input.placement();
    let ctx = ctx_rt.exec();

    let sorted = input.on(ctx).sort_values(SortOptions::default()).unwrap();
    let sorted_placement = sorted.placement();

    assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(sorted_placement.device, input_placement.device);
    assert_eq!(sorted_placement.stream_debug, ctx_rt.stream().testing_raw_handle_value());
    assert_eq!(sorted_placement.memory_resource_debug, ctx_rt.memory_resource().testing_raw_handle_value());
    assert!(!input.same_provenance(&sorted), "a shared device must not hide different concrete handles");

    let err = DataFrame::from_series([("input", input), ("sorted", sorted)]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::CrossProvenanceInputs);
    let left_domain = err.detail_value("left_domain").unwrap();
    let right_domain = err.detail_value("right_domain").unwrap();
    assert!(left_domain.contains(&format!("stream={}", input_placement.stream_debug)));
    assert!(right_domain.contains(&format!("stream={}", sorted_placement.stream_debug)));
    assert!(right_domain.contains(&format!("memory_resource={}", sorted_placement.memory_resource_debug)));
}

fn deep_copy_and_from_table_preserve_explicit_domain_identity() {
    let column = explicit_i32_column(&[1, 2, 3], 41);
    let expected_stream = column.testing_execution_domain_stream_value();
    let expected_mr = column.testing_execution_domain_mr_value();

    reset_exec_observation();
    let column_copy = column.deep_copy().unwrap();
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_column_exec_contract(&column_copy, 41, expected_stream, expected_mr);

    let table =
        Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 42), explicit_i32_column(&[40, 50, 60], 42)])
            .unwrap();
    let table_stream = table.testing_execution_domain_stream_value();
    let table_mr = table.testing_execution_domain_mr_value();

    reset_exec_observation();
    let table_copy = table.deep_copy().unwrap();
    assert_eq!(last_exec_stream_value(), table_stream);
    assert_eq!(last_exec_mr_value(), table_mr);
    assert_eq!(table_copy.testing_execution_domain_id(), 42);
    assert!(table_copy.into_columns().unwrap().iter().all(|col| col.testing_execution_domain_id() == 42));

    let table =
        Table::from_columns(vec![explicit_i32_column(&[7, 8, 9], 43), explicit_i32_column(&[70, 80, 90], 43)]).unwrap();
    let df = DataFrame::from_table(table, vec!["a".into(), "b".into()]).unwrap();
    assert_eq!(df.testing_execution_domain_kind(), "explicit");
    assert_eq!(df.testing_execution_domain_id(), 43);
    assert_eq!(df.series("a").unwrap().as_column().testing_execution_domain_id(), 43);
    assert_eq!(df.series("b").unwrap().as_column().testing_execution_domain_id(), 43);
}

fn dataframe_and_table_construction_reject_mixed_domain_columns() {
    let left = explicit_i32_column(&[1, 2, 3], 100);
    let right = explicit_i32_column(&[4, 5, 6], 200);

    let err = dataframe_from_columns(
        vec!["left".into(), "right".into()],
        vec![left.deep_copy().unwrap(), right.deep_copy().unwrap()],
    )
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::CrossProvenanceInputs);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), "dataframe::new");
    assert!(err.details().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("domain:100(")));
    assert!(err.details().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("domain:200(")));
    assert_eq!(err.fact_value(FactKeyName::LeftDevice), Some(&FactValue::DeviceId(0)));
    assert_eq!(err.fact_value(FactKeyName::RightDevice), Some(&FactValue::DeviceId(0)));
    assert_eq!(err.detail_value("frame_column"), None);

    let err = Table::from_columns(vec![left, right]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::CrossProvenanceInputs);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), "table::from_columns");
    assert!(err.details().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("domain:100(")));
    assert!(err.details().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("domain:200(")));
    assert_eq!(err.fact_value(FactKeyName::LeftDevice), Some(&FactValue::DeviceId(0)));
    assert_eq!(err.fact_value(FactKeyName::RightDevice), Some(&FactValue::DeviceId(0)));
}

fn dataframe_with_column_rejects_mixed_provenance_with_canonical_context() {
    let df = DataFrame::from_series([("id", explicit_i32_series("id", &[1, 2, 3], 300))]).unwrap();
    let foreign = explicit_i32_series("value", &[10, 20, 30], 301);

    let err = df.with_column("value", foreign).unwrap_err();

    assert_eq!(err.code(), ErrorCode::CrossProvenanceInputs);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), "dataframe::with_column");
    assert_eq!(err.fact_value(FactKeyName::ColumnName), Some(&FactValue::Str(std::borrow::Cow::Borrowed("value"))));
    assert!(err.details().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("domain:300(")));
    assert!(err.details().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("domain:301(")));
    assert_eq!(err.fact_value(FactKeyName::LeftDevice), Some(&FactValue::DeviceId(0)));
    assert_eq!(err.fact_value(FactKeyName::RightDevice), Some(&FactValue::DeviceId(0)));
    assert!(err.details().iter().any(|(key, value)| *key == "remedy" && value.contains("copy_into_runtime")));
    assert_eq!(err.detail_value("frame_domain"), None);
    assert_eq!(err.detail_value("series_domain"), None);
    assert_eq!(err.detail_value("frame_device"), None);
    assert_eq!(err.detail_value("series_device"), None);
}

fn representative_high_level_ops_preserve_explicit_domain_identity() {
    let df = DataFrame::from_series([
        ("id", explicit_i32_series("id", &[3, 1, 2], 55)),
        ("value", explicit_i32_series("value", &[30, 10, 20], 55)),
    ])
    .unwrap();

    let sorted = df.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()).unwrap();
    assert_eq!(sorted.testing_execution_domain_id(), df.testing_execution_domain_id());
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);

    let series = explicit_nullable_i32_series("score", &[30, 0, 10, 0], &[true, false, true, false], 56);
    let filled = series.on(Exec::RETAINED).fillna(99i32).unwrap();
    assert_eq!(filled.as_column().testing_execution_domain_id(), series.as_column().testing_execution_domain_id());
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![30, 99, 10, 99]);
}

fn host_materialization_and_io_paths_sync_retained_stream() {
    let _event_observation = lock_event_observation();
    let numeric = explicit_i32_series("id", &[1, 2, 3], 60);
    reset_sync_observation();
    assert_eq!(numeric.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(last_sync_stream_value(), numeric.as_column().testing_execution_domain_stream_value());

    let strings = explicit_str_series("name", &["alice", "bob"], 61);
    reset_sync_observation();
    assert_eq!(strings.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);
    assert_eq!(last_sync_stream_value(), strings.as_column().testing_execution_domain_stream_value());

    let scalar = scalar(7i32).unwrap().testing_rebind_execution_domain_explicit(62).unwrap();
    reset_sync_observation();
    assert_eq!(scalar.value::<i32>().unwrap(), 7);
    assert_eq!(last_sync_stream_value(), scalar.testing_execution_domain_stream_value());

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("execution_domain.parquet");
    let df = DataFrame::from_series([("id", numeric)]).unwrap();
    reset_sync_observation();
    df.write_parquet(&path).unwrap();
    assert_eq!(last_sync_stream_value(), df.testing_execution_domain_stream_value());
}

fn unary_binary_and_reduction_families_use_primary_input_exec_contract() {
    let _event_observation = lock_event_observation();
    let input = explicit_i32_series("input", &[-3, 0, 5], 140);
    let input_stream = input.as_column().testing_execution_domain_stream_value();
    let input_mr = input.as_column().testing_execution_domain_mr_value();

    reset_exec_observation();
    let abs = input.on(Exec::RETAINED).abs().unwrap();
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_column_exec_contract(abs.as_column(), 140, input_stream, input_mr);
    reset_sync_observation();
    assert_eq!(abs.to_vec_strict::<i32>().unwrap(), vec![3, 0, 5]);
    assert_eq!(last_sync_stream_value(), input_stream);

    let rhs = explicit_i32_series("rhs", &[10, 20, 30], 141);
    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let added = input.on(Exec::RETAINED).add(&rhs).unwrap();
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_eq!(dependency_wait_call_count_for_stream(input_stream), 1);
    assert_column_exec_contract(added.as_column(), 140, input_stream, input_mr);
    reset_sync_observation();
    assert_eq!(added.to_vec_strict::<i32>().unwrap(), vec![7, 20, 35]);
    assert_eq!(last_sync_stream_value(), input_stream);

    reset_exec_observation();
    let reduced = input.on(Exec::RETAINED).sum().unwrap();
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_eq!(reduced.testing_execution_domain_id(), 140);
    reset_sync_observation();
    assert_eq!(reduced.value::<i64>().unwrap(), 2);
    assert_eq!(last_sync_stream_value(), input_stream);
}

fn binary_op_reverse_fences_cross_stream_input_after_enqueue() {
    // After a consuming operation returns, every cross-stream input's retained
    // stream must wait on the consumer's
    // completion event. Otherwise dropping the input enqueues its
    // stream-ordered deallocation on its own stream and the pool can reuse
    // the block while the consuming kernel is still reading it.
    let _event_observation = lock_event_observation();
    let input = explicit_i32_series("input", &[-3, 0, 5], 145);
    let rhs = explicit_i32_series("rhs", &[10, 20, 30], 146);
    let input_stream = input.as_column().testing_execution_domain_stream_value();
    let rhs_stream = rhs.as_column().testing_execution_domain_stream_value();

    reset_dependency_observation();
    let added = input.on(Exec::RETAINED).add(&rhs).unwrap();
    // Forward fence: the output (= primary input) stream waited on rhs once.
    assert_eq!(dependency_wait_call_count_for_stream(input_stream), 1);
    // Reverse fence: rhs's retained stream waits on the consumer completion
    // event, so dropping `rhs` now cannot race the add kernel.
    assert_eq!(dependency_wait_call_count_for_stream(rhs_stream), 1);
    drop(rhs);
    assert_eq!(added.to_vec_strict::<i32>().unwrap(), vec![7, 20, 35]);
}

fn filter_and_gather_families_use_data_provenance_for_mixed_domains() {
    let _event_observation = lock_event_observation();
    let values = explicit_i32_series("values", &[30, 10, 20], 150);
    let values_stream = values.as_column().testing_execution_domain_stream_value();
    let values_mr = values.as_column().testing_execution_domain_mr_value();

    let mask = explicit_bool_series("mask", &[true, false, true], 152);
    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let filtered = values.on(Exec::RETAINED).filter(&mask).unwrap();
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);
    assert_eq!(filtered.as_column().testing_execution_domain_id(), 150);
    assert_eq!(dependency_wait_call_count_for_stream(values_stream), 1);
    reset_sync_observation();
    assert_eq!(filtered.to_vec_strict::<i32>().unwrap(), vec![30, 20]);
    assert_eq!(last_sync_stream_value(), values_stream);

    let gather_map = explicit_i32_series("gather_map", &[2, 0], 153);
    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let gathered = values.on(Exec::RETAINED).gather(&gather_map).unwrap();
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);
    assert_eq!(gathered.as_column().testing_execution_domain_id(), 150);
    assert_eq!(dependency_wait_call_count_for_stream(values_stream), 1);
    reset_sync_observation();
    assert_eq!(gathered.to_vec_strict::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(last_sync_stream_value(), values_stream);
}

fn concatenate_join_and_groupby_families_use_first_or_keys_provenance() {
    let _event_observation = lock_event_observation();
    let first = explicit_i32_series("first", &[1, 2], 170);
    let second = explicit_i32_series("second", &[3, 4], 171);
    let first_stream = first.as_column().testing_execution_domain_stream_value();
    let first_mr = first.as_column().testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let concatenated = Series::concat(&[&first, &second], Exec::RETAINED).unwrap();
    assert_eq!(last_exec_stream_value(), first_stream);
    assert_eq!(last_exec_mr_value(), first_mr);
    assert_eq!(concatenated.as_column().testing_execution_domain_id(), 170);
    assert_eq!(dependency_wait_call_count_for_stream(first_stream), 1);
    reset_sync_observation();
    assert_eq!(concatenated.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), first_stream);

    let left = DataFrame::from_series([
        ("id", explicit_i32_series("id", &[1, 2, 3], 180)),
        ("left_payload", explicit_i32_series("left_payload", &[10, 20, 30], 180)),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", explicit_i32_series("id", &[2, 3, 4], 181)),
        ("right_payload", explicit_i32_series("right_payload", &[200, 300, 400], 181)),
    ])
    .unwrap();
    let left_stream = left.testing_execution_domain_stream_value();
    let left_mr = left.testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let joined = left.on(Exec::RETAINED).merge(&right, MergeOptions::new().on(["id"])).unwrap();
    assert_eq!(last_exec_stream_value(), left_stream);
    assert_eq!(last_exec_mr_value(), left_mr);
    assert_eq!(dependency_wait_call_count_for_stream(left_stream), 1);
    assert_eq!(joined.testing_execution_domain_id(), 180);
    reset_sync_observation();
    let mut rows: Vec<_> = joined
        .series("id")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(joined.series("left_payload").unwrap().to_vec_strict::<i32>().unwrap())
        .zip(joined.series("right_payload").unwrap().to_vec_strict::<i32>().unwrap())
        .map(|((id, left_payload), right_payload)| (id, left_payload, right_payload))
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![(2, 20, 200), (3, 30, 300)]);
    assert_eq!(last_sync_stream_value(), left_stream);

    let grouped_input = DataFrame::from_series([
        ("key", explicit_i32_series("key", &[1, 1, 2, 2], 190)),
        ("value", explicit_i32_series("value", &[10, 20, 30, 40], 190)),
    ])
    .unwrap();
    let keys_stream = grouped_input.testing_execution_domain_stream_value();
    let keys_mr = grouped_input.testing_execution_domain_mr_value();

    reset_sync_observation();
    reset_exec_observation();
    let grouped =
        grouped_input.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();
    assert_eq!(last_exec_stream_value(), keys_stream);
    assert_eq!(last_exec_mr_value(), keys_mr);
    assert_eq!(grouped.testing_execution_domain_id(), 190);
    reset_sync_observation();
    let mut sums: Vec<_> = grouped
        .series("key")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(grouped.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap())
        .collect();
    sums.sort_unstable();
    assert_eq!(sums, vec![(1, 30), (2, 70)]);
    assert_eq!(last_sync_stream_value(), keys_stream);
}

fn search_family_keeps_primary_domain_contract() {
    let _event_observation = lock_event_observation();
    let runtime = sample_runtime();
    let haystack = runtime.series(&[10i32, 20, 30, 40]).unwrap();
    let needles = explicit_i32_series("needles", &[20, 50, 10], 201);
    let needles_stream = needles.as_column().testing_execution_domain_stream_value();
    let needles_mr = needles.as_column().testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let contains = needles.on(Exec::RETAINED).isin(&haystack).unwrap();
    assert_eq!(last_exec_stream_value(), needles_stream);
    assert_eq!(last_exec_mr_value(), needles_mr);
    assert_eq!(contains.as_column().testing_execution_domain_id(), needles.as_column().testing_execution_domain_id());
    assert_eq!(dependency_wait_call_count_for_stream(needles_stream), 1);
    reset_sync_observation();
    assert_eq!(contains.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(last_sync_stream_value(), needles_stream);
}

fn replace_family_uses_input_domain_and_orders_replacement_domain() {
    let _event_observation = lock_event_observation();
    let input = explicit_nullable_i32_series("input", &[1, 0, 3], &[true, false, true], 220);
    let replacement = explicit_i32_series("replacement", &[10, 20, 30], 221);
    let input_stream = input.as_column().testing_execution_domain_stream_value();
    let input_mr = input.as_column().testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let filled = input.on(Exec::RETAINED).fillna_with_series(&replacement).unwrap();

    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_column_exec_contract(filled.as_column(), 220, input_stream, input_mr);
    assert_eq!(dependency_wait_call_count_for_stream(input_stream), 1);
    reset_sync_observation();
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(last_sync_stream_value(), input_stream);
}

#[test]
fn execution_domain_contract_tests_cases() {
    run_named_cases(&[
        ("helper_constructors_seed_explicit_runtime_domain", helper_constructors_seed_explicit_runtime_domain),
        (
            "private_pool_teardown_fences_its_registered_stream_before_resource_release",
            private_pool_teardown_fences_its_registered_stream_before_resource_release,
        ),
        (
            "private_pool_adaptor_teardown_inherits_the_registered_stream_fence",
            private_pool_adaptor_teardown_inherits_the_registered_stream_fence,
        ),
        (
            "borrowed_override_keeps_device_and_binds_the_override_handles",
            borrowed_override_keeps_device_and_binds_the_override_handles,
        ),
        (
            "deep_copy_and_from_table_preserve_explicit_domain_identity",
            deep_copy_and_from_table_preserve_explicit_domain_identity,
        ),
        (
            "dataframe_and_table_construction_reject_mixed_domain_columns",
            dataframe_and_table_construction_reject_mixed_domain_columns,
        ),
        (
            "dataframe_with_column_rejects_mixed_provenance_with_canonical_context",
            dataframe_with_column_rejects_mixed_provenance_with_canonical_context,
        ),
        (
            "representative_high_level_ops_preserve_explicit_domain_identity",
            representative_high_level_ops_preserve_explicit_domain_identity,
        ),
        (
            "host_materialization_and_io_paths_sync_retained_stream",
            host_materialization_and_io_paths_sync_retained_stream,
        ),
        (
            "unary_binary_and_reduction_families_use_primary_input_exec_contract",
            unary_binary_and_reduction_families_use_primary_input_exec_contract,
        ),
        (
            "binary_op_reverse_fences_cross_stream_input_after_enqueue",
            binary_op_reverse_fences_cross_stream_input_after_enqueue,
        ),
        (
            "filter_and_gather_families_use_data_provenance_for_mixed_domains",
            filter_and_gather_families_use_data_provenance_for_mixed_domains,
        ),
        (
            "concatenate_join_and_groupby_families_use_first_or_keys_provenance",
            concatenate_join_and_groupby_families_use_first_or_keys_provenance,
        ),
        ("search_family_keeps_primary_domain_contract", search_family_keeps_primary_domain_contract),
        (
            "replace_family_uses_input_domain_and_orders_replacement_domain",
            replace_family_uses_input_domain_and_orders_replacement_domain,
        ),
    ]);
}
