//! Contract tests for execution-domain propagation.
//!
//! This suite intentionally keeps one representative test per operation
//! family. The goal is to protect provenance, execution, and synchronization
//! contracts without locking the library into an API-by-API matrix.
use cudf::{
    DataFrame, DataType, ErrorKind, GroupByAgg, Series, SortOptions, TypeId,
    core::{Column, Table},
    ops::{
        binary::{self, BinaryOperator},
        concatenate, copying, dictionary, filter,
        groupby::{self, AggregationRequest},
        join,
        reduction::{self, ReduceAgg},
        search, sorting,
        unary::{self, UnaryOperator},
    },
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, dataframe_from_columns, explicit_i32_column,
    explicit_i32_series, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation,
    reset_sync_observation, sample_runtime, scalar,
};

fn explicit_nullable_i32_series(name: &str, data: &[i32], validity: &[bool], identity: u64) -> Series {
    Series::from_column(
        column_from_slice_nullable(data, validity).unwrap().testing_rebind_execution_domain_explicit(identity).unwrap(),
    )
    .unwrap()
    .rename(name)
}

fn explicit_str_series(name: &str, data: &[&str], identity: u64) -> Series {
    Series::from_column(column_from_strs(data).unwrap().testing_rebind_execution_domain_explicit(identity).unwrap())
        .unwrap()
        .rename(name)
}

fn assert_column_exec_contract(column: &Column, expected_id: u64, expected_stream: usize, expected_mr: usize) {
    assert_eq!(column.testing_execution_domain_kind(), "explicit");
    assert_eq!(column.testing_execution_domain_id(), expected_id);
    assert_eq!(column.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(column.testing_execution_domain_mr_value(), expected_mr);
}

#[test]
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
    let df = DataFrame::from_series([series]).unwrap();
    assert_eq!(df.testing_execution_domain_kind(), "explicit");
    assert_eq!(df.testing_execution_domain_id(), expected_id);
    assert_eq!(df.series("id").unwrap().as_column().testing_execution_domain_id(), expected_id);
}

#[test]
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

#[test]
fn dataframe_and_table_construction_reject_mixed_domain_columns() {
    let left = explicit_i32_column(&[1, 2, 3], 100);
    let right = explicit_i32_column(&[4, 5, 6], 200);

    let err = dataframe_from_columns(
        vec!["left".into(), "right".into()],
        vec![left.deep_copy().unwrap(), right.deep_copy().unwrap()],
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::new");
    assert!(err.context().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:100(")));
    assert!(err.context().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:200(")));

    let err = Table::from_columns(vec![left, right]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "table::from_columns");
    assert!(err.context().iter().any(|(key, value)| *key == "left_domain" && value.starts_with("explicit:100(")));
    assert!(err.context().iter().any(|(key, value)| *key == "right_domain" && value.starts_with("explicit:200(")));
}

#[test]
fn representative_high_level_ops_preserve_explicit_domain_identity() {
    let df = DataFrame::from_series([
        explicit_i32_series("id", &[3, 1, 2], 55),
        explicit_i32_series("value", &[30, 10, 20], 55),
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

#[test]
fn host_materialization_and_io_paths_sync_retained_stream() {
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
    let df = DataFrame::from_series([numeric.clone()]).unwrap();
    reset_sync_observation();
    df.write_parquet(&path).unwrap();
    assert_eq!(last_sync_stream_value(), df.testing_execution_domain_stream_value());
}

#[test]
fn unary_binary_and_reduction_families_use_primary_input_exec_contract() {
    let input = explicit_i32_column(&[-3, 0, 5], 140);
    let input_stream = input.testing_execution_domain_stream_value();
    let input_mr = input.testing_execution_domain_mr_value();

    reset_exec_observation();
    let abs = unary::unary_operation(&input.view().unwrap(), UnaryOperator::Abs).unwrap();
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_column_exec_contract(&abs, 140, input_stream, input_mr);
    reset_sync_observation();
    assert_eq!(abs.view().unwrap().to_vec::<i32>().unwrap(), vec![3, 0, 5]);
    assert_eq!(last_sync_stream_value(), input_stream);

    let rhs = explicit_i32_column(&[10, 20, 30], 141);
    reset_sync_observation();
    reset_exec_observation();
    let added = binary::binary_operation_vv(
        &input.view().unwrap(),
        &rhs.view().unwrap(),
        BinaryOperator::Add,
        &DataType::new(TypeId::Int32),
    )
    .unwrap();
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_eq!(last_sync_stream_value(), rhs.testing_execution_domain_stream_value());
    assert_column_exec_contract(&added, 140, input_stream, input_mr);
    reset_sync_observation();
    assert_eq!(added.view().unwrap().to_vec::<i32>().unwrap(), vec![7, 20, 35]);
    assert_eq!(last_sync_stream_value(), input_stream);

    reset_exec_observation();
    let reduced = reduction::reduce(&input.view().unwrap(), &ReduceAgg::Sum, &DataType::new(TypeId::Int64)).unwrap();
    assert_eq!(last_exec_stream_value(), input_stream);
    assert_eq!(last_exec_mr_value(), input_mr);
    assert_eq!(reduced.testing_execution_domain_id(), 140);
    reset_sync_observation();
    assert_eq!(reduced.value::<i64>().unwrap(), 2);
    assert_eq!(last_sync_stream_value(), input_stream);
}

#[test]
fn sort_filter_and_gather_families_use_data_provenance_for_mixed_domains() {
    let values = Table::from_columns(vec![explicit_i32_column(&[30, 10, 20], 150)]).unwrap();
    let keys = Table::from_columns(vec![explicit_i32_column(&[3, 1, 2], 151)]).unwrap();
    let values_stream = values.testing_execution_domain_stream_value();
    let values_mr = values.testing_execution_domain_mr_value();

    reset_exec_observation();
    let sorted = sorting::sort_by_key(&values.view().unwrap(), &keys.view().unwrap(), None, None).unwrap();
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);
    assert_eq!(sorted.testing_execution_domain_id(), 150);
    reset_sync_observation();
    assert_eq!(sorted.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(last_sync_stream_value(), values_stream);

    let mask = column_from_slice(&[true, false, true]).unwrap().testing_rebind_execution_domain_explicit(152).unwrap();
    reset_exec_observation();
    let filtered = filter::apply_boolean_mask(&values.view().unwrap(), &mask.view().unwrap()).unwrap();
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);
    assert_eq!(filtered.testing_execution_domain_id(), 150);
    reset_sync_observation();
    assert_eq!(filtered.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![30, 20]);
    assert_eq!(last_sync_stream_value(), values_stream);

    let gather_map = explicit_i32_column(&[2, 0], 153);
    reset_exec_observation();
    let gathered = copying::gather(&values.view().unwrap(), &gather_map.view().unwrap()).unwrap();
    assert_eq!(last_exec_stream_value(), values_stream);
    assert_eq!(last_exec_mr_value(), values_mr);
    assert_eq!(gathered.testing_execution_domain_id(), 150);
    reset_sync_observation();
    assert_eq!(gathered.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(last_sync_stream_value(), values_stream);
}

#[test]
fn concatenate_join_and_groupby_families_use_first_or_keys_provenance() {
    let first = Table::from_columns(vec![explicit_i32_column(&[1, 2], 170)]).unwrap();
    let second = Table::from_columns(vec![explicit_i32_column(&[3, 4], 171)]).unwrap();
    let first_stream = first.testing_execution_domain_stream_value();
    let first_mr = first.testing_execution_domain_mr_value();

    reset_exec_observation();
    let concatenated = concatenate::concatenate_tables(&[&first.view().unwrap(), &second.view().unwrap()]).unwrap();
    assert_eq!(last_exec_stream_value(), first_stream);
    assert_eq!(last_exec_mr_value(), first_mr);
    assert_eq!(concatenated.testing_execution_domain_id(), 170);
    reset_sync_observation();
    assert_eq!(concatenated.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(last_sync_stream_value(), first_stream);

    let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 180)]).unwrap();
    let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 3, 4], 181)]).unwrap();
    let left_payload = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 180)]).unwrap();
    let right_payload = Table::from_columns(vec![explicit_i32_column(&[200, 300, 400], 181)]).unwrap();
    let left_stream = left_keys.testing_execution_domain_stream_value();
    let left_mr = left_keys.testing_execution_domain_mr_value();

    reset_sync_observation();
    reset_exec_observation();
    let joined = join::inner_join(
        &left_keys.view().unwrap(),
        &right_keys.view().unwrap(),
        &left_payload.view().unwrap(),
        &right_payload.view().unwrap(),
        join::NullEquality::Equal,
    )
    .unwrap();
    assert_eq!(last_exec_stream_value(), left_stream);
    assert_eq!(last_exec_mr_value(), left_mr);
    assert_eq!(last_sync_stream_value(), right_keys.testing_execution_domain_stream_value());
    assert_eq!(joined.left.testing_execution_domain_id(), 180);
    assert_eq!(joined.right.testing_execution_domain_id(), 180);
    reset_sync_observation();
    assert_eq!(joined.left.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
    assert_eq!(joined.right.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![2, 3]);
    assert_eq!(last_sync_stream_value(), left_stream);

    let keys = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 2], 190)]).unwrap();
    let values = explicit_i32_column(&[10, 20, 30, 40], 191);
    let aggregations = [GroupByAgg::Sum];
    let values_view = values.view().unwrap();
    let requests = [AggregationRequest { values: &values_view, aggregations: &aggregations }];
    let keys_stream = keys.testing_execution_domain_stream_value();
    let keys_mr = keys.testing_execution_domain_mr_value();

    reset_sync_observation();
    reset_exec_observation();
    let grouped = groupby::groupby_aggregate(&keys.view().unwrap(), &requests, groupby::NullPolicy::Exclude).unwrap();
    assert_eq!(last_exec_stream_value(), keys_stream);
    assert_eq!(last_exec_mr_value(), keys_mr);
    assert_eq!(last_sync_stream_value(), values.testing_execution_domain_stream_value());
    assert_eq!(grouped.keys.testing_execution_domain_id(), 190);
    assert_eq!(grouped.results[0][0].testing_execution_domain_id(), 190);
    reset_sync_observation();
    assert_eq!(grouped.keys.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(grouped.results[0][0].view().unwrap().to_vec::<i64>().unwrap(), vec![30, 70]);
    assert_eq!(last_sync_stream_value(), keys_stream);
}

#[test]
fn dictionary_and_search_families_keep_primary_domain_contract() {
    let runtime = sample_runtime();
    let haystack = runtime.column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let needles = explicit_i32_column(&[20, 50, 10], 201);
    let haystack_stream = haystack.testing_execution_domain_stream_value();
    let haystack_mr = haystack.testing_execution_domain_mr_value();

    reset_exec_observation();
    let contains = search::contains(&haystack.view().unwrap(), &needles.view().unwrap()).unwrap();
    assert_eq!(last_exec_stream_value(), haystack_stream);
    assert_eq!(last_exec_mr_value(), haystack_mr);
    assert_eq!(contains.testing_execution_domain_id(), haystack.testing_execution_domain_id());
    reset_sync_observation();
    assert_eq!(contains.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(last_sync_stream_value(), haystack_stream);

    let dict_source = explicit_i32_column(&[429, 111, 213, 111], 210);
    let dict_stream = dict_source.testing_execution_domain_stream_value();
    let dict_mr = dict_source.testing_execution_domain_mr_value();

    reset_exec_observation();
    let dict = dictionary::encode(&dict_source.view().unwrap(), TypeId::Int32).unwrap();
    assert_eq!(last_exec_stream_value(), dict_stream);
    assert_eq!(last_exec_mr_value(), dict_mr);
    assert_eq!(dict.testing_execution_domain_id(), 210);

    reset_sync_observation();
    let mut keys = dictionary::keys(&dict.view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    keys.sort_unstable();
    assert_eq!(keys, vec![111, 213, 429]);
    assert_eq!(last_sync_stream_value(), dict_stream);
}
