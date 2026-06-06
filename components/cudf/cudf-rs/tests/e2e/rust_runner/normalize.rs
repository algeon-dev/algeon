//! Normalized result schema types for the E2E conformance test framework.
//!
//! Both the Rust and Python executors produce results that are converted into
//! these normalized types before comparison. This ensures the comparison logic
//! operates on a single, well-defined schema regardless of language-specific
//! API differences.

use serde::{Deserialize, Serialize};

/// A normalized result from either executor, tagged by kind.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "result_kind", rename_all = "snake_case")]
pub enum NormalizedResult {
    Dataframe(NormalizedDataFrame),
    InfoSnapshot(NormalizedInfoSnapshot),
    ColumnList(NormalizedColumnList),
    Error(NormalizedError),
}

/// Normalized DataFrame result: column metadata + materialized rows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedDataFrame {
    pub columns: Vec<NormalizedColumnMeta>,
    pub row_count: usize,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// Column metadata within a normalized DataFrame result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedColumnMeta {
    pub name: String,
    pub dtype: String,
}

/// Normalized info snapshot result from `info` operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedInfoSnapshot {
    pub row_count: usize,
    pub column_count: usize,
    pub dtype_counts: Vec<DtypeCount>,
    pub columns: Vec<NormalizedColumnInfo>,
    pub payload_bytes: Option<u64>,
    pub device_allocated_bytes: Option<u64>,
}

/// A dtype count entry within an info snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DtypeCount {
    pub dtype: String,
    pub count: usize,
}

/// Per-column info within a normalized info snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedColumnInfo {
    pub name: String,
    pub dtype: String,
    pub nullable: bool,
    pub null_count: Option<usize>,
    pub non_null_count: Option<usize>,
    pub payload_bytes: Option<u64>,
    pub device_allocated_bytes: Option<u64>,
}

/// Normalized column list result from `schema` and `dtypes` operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedColumnList {
    pub entries: Vec<ColumnListEntry>,
}

/// A single entry in a column list result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnListEntry {
    pub name: String,
    pub dtype: String,
}

/// Normalized error result for negative test cases.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedError {
    pub kind: String,
    pub operation: String,
    pub context: Vec<(String, String)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_dataframe_result() {
        let result = NormalizedResult::Dataframe(NormalizedDataFrame {
            columns: vec![
                NormalizedColumnMeta { name: "a".into(), dtype: "int32".into() },
                NormalizedColumnMeta { name: "b".into(), dtype: "float64".into() },
            ],
            row_count: 3,
            rows: vec![
                vec![serde_json::json!(1), serde_json::json!(1.5)],
                vec![serde_json::json!(2), serde_json::Value::Null],
                vec![serde_json::json!(3), serde_json::json!(3.5)],
            ],
        });

        let json = serde_json::to_string(&result).expect("serialize");
        let deserialized: NormalizedResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(result, deserialized);
    }

    #[test]
    fn round_trip_info_snapshot_result() {
        let result = NormalizedResult::InfoSnapshot(NormalizedInfoSnapshot {
            row_count: 4,
            column_count: 3,
            dtype_counts: vec![
                DtypeCount { dtype: "int32".into(), count: 1 },
                DtypeCount { dtype: "float64".into(), count: 1 },
                DtypeCount { dtype: "string".into(), count: 1 },
            ],
            columns: vec![NormalizedColumnInfo {
                name: "a".into(),
                dtype: "int32".into(),
                nullable: true,
                null_count: Some(1),
                non_null_count: Some(3),
                payload_bytes: Some(16),
                device_allocated_bytes: Some(256),
            }],
            payload_bytes: Some(100),
            device_allocated_bytes: Some(512),
        });

        let json = serde_json::to_string(&result).expect("serialize");
        let deserialized: NormalizedResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(result, deserialized);
    }

    #[test]
    fn round_trip_column_list_result() {
        let result = NormalizedResult::ColumnList(NormalizedColumnList {
            entries: vec![
                ColumnListEntry { name: "a".into(), dtype: "int32".into() },
                ColumnListEntry { name: "b".into(), dtype: "float64".into() },
            ],
        });

        let json = serde_json::to_string(&result).expect("serialize");
        let deserialized: NormalizedResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(result, deserialized);
    }

    #[test]
    fn round_trip_error_result() {
        let result = NormalizedResult::Error(NormalizedError {
            kind: "column_not_found".into(),
            operation: "select".into(),
            context: vec![("column".into(), "nonexistent".into())],
        });

        let json = serde_json::to_string(&result).expect("serialize");
        let deserialized: NormalizedResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(result, deserialized);
    }

    #[test]
    fn deserialize_dataframe_result_from_json() {
        let json = r#"{
            "result_kind": "dataframe",
            "columns": [
                { "name": "id", "dtype": "int64" }
            ],
            "row_count": 2,
            "rows": [[1], [2]]
        }"#;

        let result: NormalizedResult = serde_json::from_str(json).expect("parse");
        match result {
            NormalizedResult::Dataframe(df) => {
                assert_eq!(df.row_count, 2);
                assert_eq!(df.columns.len(), 1);
                assert_eq!(df.columns[0].name, "id");
            },
            other => panic!("expected Dataframe, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_error_result_from_json() {
        let json = r#"{
            "result_kind": "error",
            "kind": "column_not_found",
            "operation": "select",
            "context": [["column", "nonexistent"]]
        }"#;

        let result: NormalizedResult = serde_json::from_str(json).expect("parse");
        match result {
            NormalizedResult::Error(err) => {
                assert_eq!(err.kind, "column_not_found");
                assert_eq!(err.operation, "select");
                assert_eq!(err.context.len(), 1);
            },
            other => panic!("expected Error, got {:?}", other),
        }
    }
}
