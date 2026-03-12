//! Tests for the DataFrame wrapper.

use cudf::column::Column;
use cudf::dataframe::DataFrame;
use cudf::error::ErrorKind;
use cudf::table::Table;

// ---------------------------------------------------------------------------
// Construction and column-by-name access
// ---------------------------------------------------------------------------

#[test]
fn test_new_from_named_columns() {
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();

    let df = DataFrame::new(
        vec!["id".to_string(), "score".to_string()],
        vec![ids, scores],
    )
    .unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows().unwrap(), 3);
}

#[test]
fn test_column_by_name() {
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();

    let df = DataFrame::new(
        vec!["id".to_string(), "score".to_string()],
        vec![ids, scores],
    )
    .unwrap();

    let col = df.column("id").unwrap();
    assert_eq!(col.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col = df.column("score").unwrap();
    assert_eq!(col.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_reject_duplicate_column_names() {
    let c1 = Column::from_slice(&[1i32, 2]).unwrap();
    let c2 = Column::from_slice(&[3i32, 4]).unwrap();

    let result = DataFrame::new(
        vec!["x".to_string(), "x".to_string()],
        vec![c1, c2],
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_reject_mismatched_names_columns_count() {
    let c1 = Column::from_slice(&[1i32, 2]).unwrap();

    let result = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![c1],
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_column_nonexistent_name_returns_error() {
    let c1 = Column::from_slice(&[1i32, 2]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![c1]).unwrap();

    let result = df.column("nonexistent");
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_column_names_preserves_insertion_order() {
    let c1 = Column::from_slice(&[1i32]).unwrap();
    let c2 = Column::from_slice(&[2i32]).unwrap();
    let c3 = Column::from_slice(&[3i32]).unwrap();

    let df = DataFrame::new(
        vec!["zebra".to_string(), "apple".to_string(), "mango".to_string()],
        vec![c1, c2, c3],
    )
    .unwrap();

    let names = df.column_names();
    assert_eq!(names, &["zebra", "apple", "mango"]);
}

#[test]
fn test_column_index() {
    let c1 = Column::from_slice(&[1i32]).unwrap();
    let c2 = Column::from_slice(&[2i32]).unwrap();

    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![c1, c2],
    )
    .unwrap();

    assert_eq!(df.column_index("a"), Some(0));
    assert_eq!(df.column_index("b"), Some(1));
    assert_eq!(df.column_index("c"), None);
}

#[test]
fn test_schema() {
    let c1 = Column::from_slice(&[1i32]).unwrap();
    let c2 = Column::from_slice(&[2.0f64]).unwrap();

    let df = DataFrame::new(
        vec!["id".to_string(), "val".to_string()],
        vec![c1, c2],
    )
    .unwrap();

    let schema = df.schema();
    assert_eq!(schema.len(), 2);
    assert_eq!(schema["id"], 0);
    assert_eq!(schema["val"], 1);
}

#[test]
fn test_table_accessor() {
    let c1 = Column::from_slice(&[1i32, 2]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![c1]).unwrap();

    let table = df.table();
    let view = table.view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 1);
    assert_eq!(view.num_rows().unwrap(), 2);
}

#[test]
fn test_from_table_round_trip() {
    // Build a Table directly, then wrap it in a DataFrame.
    let c1 = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let c2 = Column::from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let df = DataFrame::from_table(table, vec!["a".to_string(), "b".to_string()]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows().unwrap(), 3);

    let col_a = df.column("a").unwrap();
    assert_eq!(col_a.to_vec::<i32>().unwrap(), vec![10, 20, 30]);

    let col_b = df.column("b").unwrap();
    assert_eq!(col_b.to_vec::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_from_table_mismatched_names_count() {
    let c1 = Column::from_slice(&[1i32]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let result = DataFrame::from_table(
        table,
        vec!["a".to_string(), "b".to_string()],
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_into_table() {
    let c1 = Column::from_slice(&[1i32, 2]).unwrap();
    let c2 = Column::from_slice(&[3.0f64, 4.0]).unwrap();
    let df = DataFrame::new(
        vec!["x".to_string(), "y".to_string()],
        vec![c1, c2],
    )
    .unwrap();

    let (table, names) = df.into_table();
    assert_eq!(names, vec!["x", "y"]);

    let view = table.view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 2);
    assert_eq!(view.num_rows().unwrap(), 2);
}

#[test]
fn test_empty_dataframe() {
    let df = DataFrame::new(vec![], vec![]).unwrap();
    assert_eq!(df.num_columns(), 0);
    assert!(df.column_names().is_empty());
}
