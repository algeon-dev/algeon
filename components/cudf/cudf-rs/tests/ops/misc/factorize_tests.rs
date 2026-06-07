//! Tests for Column::factorize

use cudf::{TypeId, core::Column};

use crate::common::{last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation};

#[test]
fn test_factorize_i32_with_duplicates() {
    let data = vec![429i32, 111, 213, 111, 213, 429, 213];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let (codes, uniques) = view.factorize().unwrap();

    let codes_view = codes.view().unwrap();
    let uniques_view = uniques.view().unwrap();

    // Codes should be Int32.
    assert_eq!(codes_view.data_type().unwrap().id(), TypeId::Int32);
    // Uniques should preserve the original element type.
    assert_eq!(uniques_view.data_type().unwrap().id(), TypeId::Int32);

    // 3 unique values.
    assert_eq!(uniques_view.len().unwrap(), 3);
    // Same number of codes as input rows.
    assert_eq!(codes_view.len().unwrap(), 7);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<i32>().unwrap();

    // Verify mapping: data[i] == uniques[codes[i]].
    for (i, &code) in codes_host.iter().enumerate() {
        assert_eq!(
            data[i], uniques_host[code as usize],
            "data[{}]={} should equal uniques[{}]={}",
            i, data[i], code, uniques_host[code as usize]
        );
    }

    // Verify uniques contain exactly the input distinct values (order is not guaranteed).
    let mut expected_uniques = data.clone();
    expected_uniques.sort();
    expected_uniques.dedup();

    let mut actual_uniques = uniques_host.clone();
    actual_uniques.sort();
    actual_uniques.dedup();

    assert_eq!(actual_uniques, expected_uniques);
}

#[test]
fn test_factorize_string_column() {
    let strings: &[&str] = &["banana", "apple", "cherry", "apple", "banana"];
    let col = Column::from_strings(strings).unwrap();
    let view = col.view().unwrap();

    let (codes, uniques) = view.factorize().unwrap();

    let codes_view = codes.view().unwrap();
    let uniques_view = uniques.view().unwrap();

    assert_eq!(codes_view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(uniques_view.data_type().unwrap().id(), TypeId::String);

    // 3 unique strings.
    assert_eq!(uniques_view.len().unwrap(), 3);
    assert_eq!(codes_view.len().unwrap(), 5);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_strings().unwrap();

    // Verify mapping.
    for (i, &code) in codes_host.iter().enumerate() {
        assert_eq!(
            strings[i], uniques_host[code as usize],
            "strings[{}]={} should equal uniques[{}]={}",
            i, strings[i], code, uniques_host[code as usize]
        );
    }
}

#[test]
fn test_factorize_with_nulls() {
    let data = vec![10i32, 20, 30, 20, 10];
    let validity = vec![true, false, true, true, false];
    let col = Column::from_slice_nullable(&data, &validity).unwrap();
    let view = col.view().unwrap();

    let (codes, uniques) = view.factorize().unwrap();

    let codes_view = codes.view().unwrap();
    let uniques_view = uniques.view().unwrap();

    assert_eq!(codes_view.len().unwrap(), 5);

    // Null positions in input should produce null codes.
    let codes_validity = codes_view.validity().unwrap();
    assert!(codes_validity.is_some(), "codes should have a null mask");
    let codes_mask = codes_validity.unwrap();
    // Indices 1 and 4 were null in the input.
    assert!(!codes_mask[1], "code at null position 1 should be null");
    assert!(!codes_mask[4], "code at null position 4 should be null");

    // Valid codes should map correctly.
    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<i32>().unwrap();
    for (i, (&code, &valid)) in codes_host.iter().zip(codes_mask.iter()).enumerate() {
        if valid {
            assert_eq!(data[i], uniques_host[code as usize], "data[{}]={} should equal uniques[{}]", i, data[i], code,);
        }
    }
}

#[test]
fn test_factorize_empty_column() {
    let col = Column::from_slice::<i32>(&[]).unwrap();
    let view = col.view().unwrap();

    let (codes, uniques) = view.factorize().unwrap();

    let codes_view = codes.view().unwrap();
    let uniques_view = uniques.view().unwrap();

    assert_eq!(codes_view.len().unwrap(), 0);
    assert_eq!(uniques_view.len().unwrap(), 0);
}

#[test]
fn test_factorize_single_element() {
    let col = Column::from_slice(&[42i32]).unwrap();
    let view = col.view().unwrap();

    let (codes, uniques) = view.factorize().unwrap();

    let codes_view = codes.view().unwrap();
    let uniques_view = uniques.view().unwrap();

    assert_eq!(codes_view.len().unwrap(), 1);
    assert_eq!(uniques_view.len().unwrap(), 1);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<i32>().unwrap();

    assert_eq!(codes_host, vec![0]);
    assert_eq!(uniques_host, vec![42]);
}

#[test]
fn test_factorize_already_unique() {
    let data = vec![10i32, 20, 30, 40, 50];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let (codes, uniques) = view.factorize().unwrap();

    let codes_view = codes.view().unwrap();
    let uniques_view = uniques.view().unwrap();

    // All values are unique so uniques == sorted input.
    assert_eq!(uniques_view.len().unwrap(), 5);
    assert_eq!(codes_view.len().unwrap(), 5);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<i32>().unwrap();

    // Since input is already sorted and unique, codes should be identity [0,1,2,3,4].
    assert_eq!(codes_host, vec![0, 1, 2, 3, 4]);
    assert_eq!(uniques_host, data);
}

#[test]
fn test_factorize_f64_column() {
    let data = vec![3.25f64, 2.75, 3.25, 1.5, 2.75];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let (codes, uniques) = view.factorize().unwrap();

    let codes_view = codes.view().unwrap();
    let uniques_view = uniques.view().unwrap();

    assert_eq!(codes_view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(uniques_view.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(uniques_view.len().unwrap(), 3);
    assert_eq!(codes_view.len().unwrap(), 5);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<f64>().unwrap();

    // Verify mapping.
    for (i, &code) in codes_host.iter().enumerate() {
        assert!(
            (data[i] - uniques_host[code as usize]).abs() < 1e-9,
            "data[{}]={} should equal uniques[{}]={}",
            i,
            data[i],
            code,
            uniques_host[code as usize]
        );
    }
}

#[test]
fn test_factorize_codes_are_owned_independent_of_input() {
    // Verify codes and uniques are owned (deep-copied), not borrowing from input.
    let (codes, uniques) = {
        let data = vec![1i32, 2, 1, 3, 2];
        let col = Column::from_slice(&data).unwrap();
        let view = col.view().unwrap();
        view.factorize().unwrap()
        // col is dropped here
    };

    // Should still be readable after input is dropped.
    let codes_host = codes.view().unwrap().to_vec::<i32>().unwrap();
    let uniques_host = uniques.view().unwrap().to_vec::<i32>().unwrap();

    assert_eq!(codes_host.len(), 5);
    assert_eq!(uniques_host.len(), 3);
}

// --- DataFrame::factorize tests ---

#[test]
fn test_dataframe_factorize_by_name() {
    let col = Column::from_slice(&[10i32, 20, 10, 30, 20]).unwrap();
    let df = cudf::DataFrame::new(vec!["values".to_string()], vec![col]).unwrap();

    let (codes, uniques) = df.factorize("values").unwrap();

    let codes_view = codes.as_column_view().unwrap();
    let uniques_view = uniques.as_column_view().unwrap();

    assert_eq!(codes_view.len().unwrap(), 5);
    assert_eq!(uniques_view.len().unwrap(), 3);
}

#[test]
fn test_dataframe_factorize_missing_column() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let df = cudf::DataFrame::new(vec!["x".to_string()], vec![col]).unwrap();

    let result = df.factorize("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_factorize_retains_input_provenance_and_syncs_on_input_stream_end_to_end() {
    let input = Column::from_slice(&[9i32, 7, 9, 8]).unwrap().__debug_rebind_execution_domain_explicit(605).unwrap();
    let expected_stream = input.__debug_execution_domain_stream_value();

    reset_exec_observation();
    let (codes, uniques) = input.view().unwrap().factorize().unwrap();

    assert_eq!(codes.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(uniques.__debug_execution_domain_id(), input.__debug_execution_domain_id());
    assert_eq!(codes.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(uniques.__debug_execution_domain_stream_value(), expected_stream);
    assert_eq!(last_exec_stream_value(), expected_stream);

    reset_sync_observation();
    let codes_host = codes.view().unwrap().to_vec::<i32>().unwrap();
    let mut uniques_host = uniques.view().unwrap().to_vec::<i32>().unwrap();
    uniques_host.sort_unstable();
    assert_eq!(codes_host.len(), 4);
    assert_eq!(uniques_host, vec![7, 8, 9]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}
