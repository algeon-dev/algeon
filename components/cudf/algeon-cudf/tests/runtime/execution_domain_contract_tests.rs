//! Contract tests for execution-domain propagation.
//!
//! This suite intentionally keeps one representative test per operation
//! family. The goal is to protect provenance, execution, and synchronization
//! contracts without locking the library into an API-by-API matrix.
use cudf::{
    DataFrame, DataType, ErrorCode, ErrorKind, ExecutionOverrides, GroupByAgg, Runtime, Series, SortOptions, TypeId,
    core::{
        Column, Table,
        error::{FactKeyName, FactValue},
        memory::DeviceMemoryResource,
    },
    ops::{
        binary::{self, BinaryOperator},
        concatenate, copying, dictionary, filter,
        groupby::{self, AggregationRequest},
        join,
        reduction::{self, ReduceAgg},
        replace, search, sorting,
        unary::{self, UnaryOperator},
    },
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, dataframe_from_columns,
    dependency_wait_call_count_for_stream, explicit_i32_column, explicit_i32_series,
    last_destroyed_memory_resource_value, last_exec_mr_value, last_exec_stream_value,
    last_memory_resource_destroy_sequence, last_sync_sequence, last_sync_stream_value, lock_event_observation,
    reset_dependency_observation, reset_exec_observation, reset_sync_observation, sample_runtime, scalar,
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

    let series = crate::common::series::<i32>("id", &[1, 2, 3]).unwrap();
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
    let runtime = Runtime::builder().with_memory_resource(pool.limiting(16 * 1024 * 1024).unwrap()).build().unwrap();
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

fn borrowed_override_preserves_workflow_and_allocates_distinct_bundle_identity() {
    let data_rt = sample_runtime();
    let ctx_rt = sample_runtime();
    let input = data_rt.series(&[3i32, 1, 2]).unwrap();
    let input_placement = input.placement();
    let ctx = ExecutionOverrides::new().with_stream(ctx_rt.stream()).with_memory_resource(ctx_rt.memory_resource());

    let sorted = input.sort_values_in(SortOptions::default(), &ctx).unwrap();
    let sorted_placement = sorted.placement();

    assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(sorted_placement.device, input_placement.device);
    assert_eq!(sorted_placement.workflow_id, input_placement.workflow_id);
    assert_ne!(sorted_placement.bundle_id, input_placement.bundle_id);
    assert_eq!(sorted_placement.stream_debug, ctx_rt.stream().testing_raw_handle_value());
    assert_eq!(sorted_placement.memory_resource_debug, ctx_rt.memory_resource().testing_raw_handle_value());
    assert!(!input.same_provenance(&sorted), "same workflow must not hide different concrete handles");

    let err = DataFrame::from_series([("input", input), ("sorted", sorted)]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::CrossProvenanceInputs);
    let left_domain = err.detail_value("left_domain").unwrap();
    let right_domain = err.detail_value("right_domain").unwrap();
    assert!(left_domain.contains(&format!("workflow={}", input_placement.workflow_id)));
    assert!(right_domain.contains(&format!("workflow={}", sorted_placement.workflow_id)));
    assert!(left_domain.contains(&format!("bundle={}", input_placement.bundle_id)));
    assert!(right_domain.contains(&format!("bundle={}", sorted_placement.bundle_id)));
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
    assert!(err.details().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:100(")));
    assert!(err.details().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:200(")));
    assert_eq!(err.fact_value(FactKeyName::LeftDevice), Some(&FactValue::DeviceId(0)));
    assert_eq!(err.fact_value(FactKeyName::RightDevice), Some(&FactValue::DeviceId(0)));
    assert_eq!(err.detail_value("frame_column"), None);

    let err = Table::from_columns(vec![left, right]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::CrossProvenanceInputs);
    assert_eq!(err.kind(), ErrorKind::ExecutionContext);
    assert_eq!(err.operation(), "table::from_columns");
    assert!(err.details().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:100(")));
    assert!(err.details().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:200(")));
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
    assert!(err.details().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:300(")));
    assert!(err.details().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:301(")));
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

    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
    assert_eq!(sorted.testing_execution_domain_id(), df.testing_execution_domain_id());
    assert_eq!(sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);

    let series = explicit_nullable_i32_series("score", &[30, 0, 10, 0], &[true, false, true, false], 56);
    let filled = series.fillna(99i32).unwrap();
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
    let input = explicit_i32_column(&[-3, 0, 5], 140);
    let input_stream = input.testing_execution_domain_stream_value();
    let input_mr = input.testing_execution_domain_mr_value();

    reset_exec_observation();
    let abs = unary::unary_operation(&input.testing_view().unwrap(), UnaryOperator::Abs).unwrap();
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_column_exec_contract(&abs, 140, input_stream, input_mr);
    reset_sync_observation();
    assert_eq!(abs.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![3, 0, 5]);
    assert_eq!(last_sync_stream_value(), input_stream);

    let rhs = explicit_i32_column(&[10, 20, 30], 141);
    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let added = binary::binary_operation_vv(
        &input.testing_view().unwrap(),
        &rhs.testing_view().unwrap(),
        BinaryOperator::Add,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_eq!(dependency_wait_call_count_for_stream(input_stream), 1);
    assert_column_exec_contract(&added, 140, input_stream, input_mr);
    reset_sync_observation();
    assert_eq!(added.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![7, 20, 35]);
    assert_eq!(last_sync_stream_value(), input_stream);

    reset_exec_observation();
    let reduced =
        reduction::reduce(&input.testing_view().unwrap(), &ReduceAgg::Sum, &DataType::new(TypeId::Int64)).unwrap();
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
    let input = explicit_i32_column(&[-3, 0, 5], 145);
    let rhs = explicit_i32_column(&[10, 20, 30], 146);
    let input_stream = input.testing_execution_domain_stream_value();
    let rhs_stream = rhs.testing_execution_domain_stream_value();

    reset_dependency_observation();
    let added = binary::binary_operation_vv(
        &input.testing_view().unwrap(),
        &rhs.testing_view().unwrap(),
        BinaryOperator::Add,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();
    // Forward fence: the output (= primary input) stream waited on rhs once.
    assert_eq!(dependency_wait_call_count_for_stream(input_stream), 1);
    // Reverse fence: rhs's retained stream waits on the consumer completion
    // event, so dropping `rhs` now cannot race the add kernel.
    assert_eq!(dependency_wait_call_count_for_stream(rhs_stream), 1);
    drop(rhs);
    assert_eq!(added.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![7, 20, 35]);
}

fn sort_filter_and_gather_families_use_data_provenance_for_mixed_domains() {
    let _event_observation = lock_event_observation();
    let values = Table::from_columns(vec![explicit_i32_column(&[30, 10, 20], 150)]).unwrap();
    let keys = Table::from_columns(vec![explicit_i32_column(&[3, 1, 2], 151)]).unwrap();
    let values_stream = values.testing_execution_domain_stream_value();
    let values_mr = values.testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let sorted =
        sorting::sort_by_key(&values.testing_view().unwrap(), &keys.testing_view().unwrap(), None, None).unwrap();
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);
    assert_eq!(sorted.testing_execution_domain_id(), 150);
    assert_eq!(dependency_wait_call_count_for_stream(values_stream), 1);
    reset_sync_observation();
    assert_eq!(sorted.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(last_sync_stream_value(), values_stream);

    let mask = column_from_slice(&[true, false, true]).unwrap().testing_rebind_execution_domain_explicit(152).unwrap();
    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let filtered = filter::apply_boolean_mask(&values.testing_view().unwrap(), &mask.testing_view().unwrap()).unwrap();
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);
    assert_eq!(filtered.testing_execution_domain_id(), 150);
    assert_eq!(dependency_wait_call_count_for_stream(values_stream), 1);
    reset_sync_observation();
    assert_eq!(filtered.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![30, 20]);
    assert_eq!(last_sync_stream_value(), values_stream);

    let gather_map = explicit_i32_column(&[2, 0], 153);
    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let gathered = copying::gather(&values.testing_view().unwrap(), &gather_map.testing_view().unwrap()).unwrap();
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);
    assert_eq!(gathered.testing_execution_domain_id(), 150);
    assert_eq!(dependency_wait_call_count_for_stream(values_stream), 1);
    reset_sync_observation();
    assert_eq!(gathered.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(last_sync_stream_value(), values_stream);
}

fn concatenate_join_and_groupby_families_use_first_or_keys_provenance() {
    let _event_observation = lock_event_observation();
    let first = Table::from_columns(vec![explicit_i32_column(&[1, 2], 170)]).unwrap();
    let second = Table::from_columns(vec![explicit_i32_column(&[3, 4], 171)]).unwrap();
    let first_stream = first.testing_execution_domain_stream_value();
    let first_mr = first.testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let concatenated =
        concatenate::concatenate_tables(&[&first.testing_view().unwrap(), &second.testing_view().unwrap()]).unwrap();
    assert_eq!(last_exec_stream_value(), first_stream);
    assert_eq!(last_exec_mr_value(), first_mr);
    assert_eq!(concatenated.testing_execution_domain_id(), 170);
    assert_eq!(dependency_wait_call_count_for_stream(first_stream), 1);
    reset_sync_observation();
    assert_eq!(concatenated.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), first_stream);

    let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 180)]).unwrap();
    let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 3, 4], 181)]).unwrap();
    let left_payload = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 180)]).unwrap();
    let right_payload = Table::from_columns(vec![explicit_i32_column(&[200, 300, 400], 181)]).unwrap();
    let left_stream = left_keys.testing_execution_domain_stream_value();
    let left_mr = left_keys.testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_sync_observation();
    reset_exec_observation();
    let joined = join::inner_join(
        &left_keys.testing_view().unwrap(),
        &right_keys.testing_view().unwrap(),
        &left_payload.testing_view().unwrap(),
        &right_payload.testing_view().unwrap(),
        join::NullEquality::Equal,
    )
    .unwrap();
    assert_eq!(last_exec_stream_value(), left_stream);
    assert_eq!(last_exec_mr_value(), left_mr);
    assert_eq!(dependency_wait_call_count_for_stream(left_stream), 2);
    assert_eq!(joined.left.testing_execution_domain_id(), 180);
    assert_eq!(joined.right.testing_execution_domain_id(), 180);
    reset_sync_observation();
    assert_eq!(joined.left.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
    assert_eq!(joined.right.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
    assert_eq!(last_sync_stream_value(), left_stream);

    let keys = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 2], 190)]).unwrap();
    let values = explicit_i32_column(&[10, 20, 30, 40], 191);
    let aggregations = [GroupByAgg::Sum];
    let values_view = values.testing_view().unwrap();
    let requests = [AggregationRequest { values: &values_view, aggregations: &aggregations }];
    let keys_stream = keys.testing_execution_domain_stream_value();
    let keys_mr = keys.testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let grouped =
        groupby::groupby_aggregate(&keys.testing_view().unwrap(), &requests, groupby::NullPolicy::Exclude).unwrap();
    assert_eq!(last_exec_stream_value(), keys_stream);
    assert_eq!(last_exec_mr_value(), keys_mr);
    assert_eq!(dependency_wait_call_count_for_stream(keys_stream), 1);
    assert_eq!(grouped.keys.testing_execution_domain_id(), 190);
    assert_eq!(grouped.results[0][0].testing_execution_domain_id(), 190);
    reset_sync_observation();
    assert_eq!(grouped.keys.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(grouped.results[0][0].testing_view().unwrap().to_vec::<i64>().unwrap(), vec![30, 70]);
    assert_eq!(last_sync_stream_value(), keys_stream);
}

fn dictionary_and_search_families_keep_primary_domain_contract() {
    let _event_observation = lock_event_observation();
    let runtime = sample_runtime();
    let haystack = runtime.column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let needles = explicit_i32_column(&[20, 50, 10], 201);
    let haystack_stream = haystack.testing_execution_domain_stream_value();
    let haystack_mr = haystack.testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let contains = search::contains(&haystack.testing_view().unwrap(), &needles.testing_view().unwrap()).unwrap();
    assert_eq!(last_exec_stream_value(), haystack_stream);
    assert_eq!(last_exec_mr_value(), haystack_mr);
    assert_eq!(contains.testing_execution_domain_id(), haystack.testing_execution_domain_id());
    assert_eq!(dependency_wait_call_count_for_stream(haystack_stream), 1);
    reset_sync_observation();
    assert_eq!(contains.testing_view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(last_sync_stream_value(), haystack_stream);

    let dict_source = explicit_i32_column(&[429, 111, 213, 111], 210);
    let dict_stream = dict_source.testing_execution_domain_stream_value();
    let dict_mr = dict_source.testing_execution_domain_mr_value();

    reset_exec_observation();
    let dict = dictionary::encode(&dict_source.testing_view().unwrap(), TypeId::Int32).unwrap();
    assert_eq!(last_exec_stream_value(), dict_stream);
    assert_eq!(last_exec_mr_value(), dict_mr);
    assert_eq!(dict.testing_execution_domain_id(), 210);

    reset_sync_observation();
    let mut keys = dictionary::keys(&dict.testing_view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    keys.sort_unstable();
    assert_eq!(keys, vec![111, 213, 429]);
    assert_eq!(last_sync_stream_value(), dict_stream);
}

fn replace_family_uses_input_domain_and_orders_replacement_domain() {
    let _event_observation = lock_event_observation();
    let input = explicit_nullable_i32_series("input", &[1, 0, 3], &[true, false, true], 220);
    let replacement = explicit_i32_column(&[10, 20, 30], 221);
    let input_stream = input.as_column().testing_execution_domain_stream_value();
    let input_mr = input.as_column().testing_execution_domain_mr_value();

    reset_dependency_observation();
    reset_sync_observation();
    reset_exec_observation();
    let filled =
        replace::replace_nulls_column(&input.as_column().testing_view().unwrap(), &replacement.testing_view().unwrap())
            .unwrap();

    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_column_exec_contract(&filled, 220, input_stream, input_mr);
    assert_eq!(dependency_wait_call_count_for_stream(input_stream), 1);
    reset_sync_observation();
    assert_eq!(filled.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(last_sync_stream_value(), input_stream);
}

#[test]
fn execution_domain_contract_tests_cases() {
    crate::common::run_named_cases(&[
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
            "borrowed_override_preserves_workflow_and_allocates_distinct_bundle_identity",
            borrowed_override_preserves_workflow_and_allocates_distinct_bundle_identity,
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
            "sort_filter_and_gather_families_use_data_provenance_for_mixed_domains",
            sort_filter_and_gather_families_use_data_provenance_for_mixed_domains,
        ),
        (
            "concatenate_join_and_groupby_families_use_first_or_keys_provenance",
            concatenate_join_and_groupby_families_use_first_or_keys_provenance,
        ),
        (
            "dictionary_and_search_families_keep_primary_domain_contract",
            dictionary_and_search_families_keep_primary_domain_contract,
        ),
        (
            "replace_family_uses_input_domain_and_orders_replacement_domain",
            replace_family_uses_input_domain_and_orders_replacement_domain,
        ),
    ]);
}
