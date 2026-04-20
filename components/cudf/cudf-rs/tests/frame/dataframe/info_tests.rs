//! Integration tests for DataFrame metadata snapshots via `info()`.
use cudf::{
    DataFrame, DataType, Series, TypeId,
    frame::dataframe::{DataFrameInfo, DataFrameMemoryAccounting, DataTypeCount, InfoDetail},
    ops::dictionary,
    testing::prelude::*,
};
use cudf_sys::{cudf_test_exec_last_stream_value, cudf_test_exec_reset_observation};

use crate::common::sample_runtime;

fn obs_reset() {
    unsafe { cudf_test_exec_reset_observation() };
}

fn obs_last_stream() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_stream_value failed");
    out
}

fn nullable_bitmask_payload_bytes(len: usize) -> usize {
    len.div_ceil(32) * std::mem::size_of::<u32>()
}

fn sample_info_frame() -> DataFrame {
    let ids = crate::common::series::<i32>("ids", &[1, 2, 3]).unwrap();
    let maybe = crate::common::nullable_series::<i32>("maybe", &[10, 0, 30], &[true, false, true]).unwrap();
    let names = crate::common::series("names", &["ab", "cde", ""]).unwrap();

    DataFrame::from_columns(::std::iter::IntoIterator::into_iter([ids, maybe, names]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap()
}

#[test]
fn test_info_default_returns_only_summary_metadata() {
    let df = sample_info_frame();

    let info = df.info().unwrap();

    assert_eq!(info.row_count, 3);
    assert_eq!(info.column_count, 3);
    assert_eq!(
        info.dtypes,
        vec![
            DataTypeCount { data_type: DataType::new(TypeId::Int32), column_count: 2 },
            DataTypeCount { data_type: DataType::new(TypeId::String), column_count: 1 },
        ]
    );
    assert_eq!(info.columns, None);
    assert_eq!(info.memory_accounting, None);
}

#[test]
fn test_info_with_columns_and_null_counts_reports_schema_order_and_counts() {
    let df = sample_info_frame();

    let info = df.info_with(InfoDetail::NULL_COUNTS).unwrap();

    let columns = info.columns.expect("columns should be populated");
    assert_eq!(columns.len(), 3);

    assert_eq!(columns[0].position, 0);
    assert_eq!(columns[0].name, "ids");
    assert_eq!(columns[0].data_type, DataType::new(TypeId::Int32));
    assert!(!columns[0].nullable);
    assert_eq!(columns[0].null_count, Some(0));
    assert_eq!(columns[0].non_null_count, Some(3));
    assert_eq!(columns[0].memory_accounting, None);

    assert_eq!(columns[1].position, 1);
    assert_eq!(columns[1].name, "maybe");
    assert_eq!(columns[1].data_type, DataType::new(TypeId::Int32));
    assert!(columns[1].nullable);
    assert_eq!(columns[1].null_count, Some(1));
    assert_eq!(columns[1].non_null_count, Some(2));
    assert_eq!(columns[1].memory_accounting, None);

    assert_eq!(columns[2].position, 2);
    assert_eq!(columns[2].name, "names");
    assert_eq!(columns[2].data_type, DataType::new(TypeId::String));
    assert!(!columns[2].nullable);
    assert_eq!(columns[2].null_count, Some(0));
    assert_eq!(columns[2].non_null_count, Some(3));
    assert_eq!(columns[2].memory_accounting, None);

    assert_eq!(info.memory_accounting, None);
}

#[test]
fn test_info_with_payload_memory_accounting_reports_expected_bytes() {
    let df = sample_info_frame();

    let info = df.info_with(InfoDetail::COLUMNS | InfoDetail::PAYLOAD_BYTES).unwrap();

    let columns = info.columns.expect("columns should be populated");
    assert_eq!(columns.len(), 3);

    let ids_payload = 3 * std::mem::size_of::<i32>();
    let maybe_payload = (3 * std::mem::size_of::<i32>()) + nullable_bitmask_payload_bytes(3);
    let names_payload = (4 * std::mem::size_of::<i32>()) + 5;

    assert_eq!(columns[0].memory_accounting.as_ref().unwrap().payload_bytes, Some(ids_payload));
    assert_eq!(columns[0].memory_accounting.as_ref().unwrap().device_allocated_bytes, None);
    assert_eq!(columns[1].memory_accounting.as_ref().unwrap().payload_bytes, Some(maybe_payload));
    assert_eq!(columns[1].memory_accounting.as_ref().unwrap().device_allocated_bytes, None);
    assert_eq!(columns[2].memory_accounting.as_ref().unwrap().payload_bytes, Some(names_payload));
    assert_eq!(columns[2].memory_accounting.as_ref().unwrap().device_allocated_bytes, None);

    assert_eq!(
        info.memory_accounting,
        Some(DataFrameMemoryAccounting {
            payload_bytes: Some(ids_payload + maybe_payload + names_payload),
            device_allocated_bytes: None,
        })
    );
}

#[test]
fn test_info_with_device_allocated_memory_accounting_reports_only_device_allocated_bytes() {
    let df = sample_info_frame();

    let device_only = df.info_with(InfoDetail::COLUMNS | InfoDetail::DEVICE_ALLOCATED).unwrap();
    let payload_only = df.info_with(InfoDetail::COLUMNS | InfoDetail::PAYLOAD_BYTES).unwrap();

    let device_columns = device_only.columns.as_ref().expect("columns should be populated");
    let payload_columns = payload_only.columns.as_ref().expect("columns should be populated");
    assert_eq!(device_columns.len(), payload_columns.len());

    for (device_col, payload_col) in device_columns.iter().zip(payload_columns.iter()) {
        let device_memory = device_col.memory_accounting.as_ref().expect("device accounting should be populated");
        let payload_memory = payload_col.memory_accounting.as_ref().expect("payload accounting should be populated");

        assert_eq!(device_memory.payload_bytes, None);
        assert!(device_memory.device_allocated_bytes.is_some());
        assert_eq!(payload_memory.device_allocated_bytes, None);
        assert!(
            device_memory.device_allocated_bytes.unwrap() >= payload_memory.payload_bytes.unwrap(),
            "device-allocated bytes should be at least payload bytes for column {}",
            device_col.name
        );
    }

    let device_memory = device_only.memory_accounting.expect("top-level memory should be populated");
    let payload_memory = payload_only.memory_accounting.expect("top-level memory should be populated");
    assert_eq!(device_memory.payload_bytes, None);
    assert!(device_memory.device_allocated_bytes.is_some());
    assert_eq!(payload_memory.device_allocated_bytes, None);
    assert!(
        device_memory.device_allocated_bytes.unwrap() >= payload_memory.payload_bytes.unwrap(),
        "device-allocated bytes should be at least payload bytes"
    );
}

#[test]
fn test_info_with_both_memory_accounting_modes_reports_top_level_totals_without_columns() {
    let df = sample_info_frame();

    let info = df.info_with(InfoDetail::MEMORY).unwrap();

    let memory = info.memory_accounting.expect("top-level memory should be populated");
    let payload = memory.payload_bytes.expect("payload bytes should be present");
    let device_allocated = memory.device_allocated_bytes.expect("device-allocated bytes should be present");

    assert_eq!(info.columns, None);
    assert_eq!(
        payload,
        (3 * std::mem::size_of::<i32>())
            + ((3 * std::mem::size_of::<i32>()) + nullable_bitmask_payload_bytes(3))
            + ((4 * std::mem::size_of::<i32>()) + 5)
    );
    assert!(device_allocated >= payload, "device-allocated bytes should be at least payload bytes");
}

#[test]
fn test_info_payload_and_device_allocated_memory_for_dictionary_column() {
    let plain = crate::common::series("plain", &["beta", "alpha", "beta", "gamma"]).unwrap();
    let dict = dictionary::encode(&plain.testing_view().unwrap(), TypeId::Int32).unwrap();
    let category = Series::from_column(dict).unwrap().rename("category");
    let df = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([category]).map(|series| {
        let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
        (name, series)
    }))
    .unwrap();

    let info = df.info_with(InfoDetail::COLUMNS | InfoDetail::MEMORY).unwrap();

    assert_eq!(info.dtypes, vec![DataTypeCount { data_type: DataType::new(TypeId::Dictionary32), column_count: 1 }]);

    let columns = info.columns.expect("columns should be populated");
    let column = &columns[0];
    assert_eq!(column.position, 0);
    assert_eq!(column.name, "category");
    assert_eq!(column.data_type, DataType::new(TypeId::Dictionary32));

    let memory = column.memory_accounting.as_ref().expect("memory should be populated");
    let payload = memory.payload_bytes.expect("payload bytes should be present");
    let device_allocated = memory.device_allocated_bytes.expect("device-allocated bytes should be present");

    assert_eq!(payload, (4 * std::mem::size_of::<i32>()) + (4 * std::mem::size_of::<i32>()) + 14);
    assert!(device_allocated >= payload, "device-allocated bytes should be at least payload bytes");

    let top_level = info.memory_accounting.expect("top-level memory should be present");
    assert_eq!(top_level.payload_bytes, Some(payload));
    assert_eq!(top_level.device_allocated_bytes, Some(device_allocated));
}

#[test]
fn test_info_empty_dataframe_returns_valid_empty_snapshot() {
    let df = crate::common::dataframe_from_columns(vec![], vec![]).unwrap();

    let info = df.info_with(InfoDetail::ALL).unwrap();

    assert_eq!(
        info,
        DataFrameInfo {
            row_count: 0,
            column_count: 0,
            dtypes: vec![],
            columns: Some(vec![]),
            memory_accounting: Some(DataFrameMemoryAccounting {
                payload_bytes: Some(0),
                device_allocated_bytes: Some(0),
            }),
        }
    );
}

#[test]
fn test_info_payload_accounting_for_explicit_runtime_string_column_uses_runtime_stream() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().testing_raw_handle_value();
    let df = DataFrame::from_columns(
        ::std::iter::IntoIterator::into_iter([rt.named_series("names", &["ab", "cde", ""]).unwrap()]).map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }),
    )
    .unwrap();

    obs_reset();
    let info = df.info_with(InfoDetail::COLUMNS | InfoDetail::PAYLOAD_BYTES).unwrap();

    assert_eq!(obs_last_stream(), expected_stream);
    assert_eq!(info.columns.as_ref().unwrap()[0].memory_accounting.as_ref().unwrap().payload_bytes, Some((4 * 4) + 5));
    assert_eq!(info.memory_accounting.unwrap().payload_bytes, Some((4 * 4) + 5));
}
