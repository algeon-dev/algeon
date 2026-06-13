//! Manifest parsing types for the E2E conformance test framework.
//!
//! These types map directly to the JSON fixture format used by the shared
//! Rust and Python test runners. Each fixture file contains datasets (input
//! data definitions) and cases (operation + expected result kind +
//! comparison policy).

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Top-level fixture loaded from a JSON manifest file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fixture {
    pub schema_version: u32,
    pub datasets: Vec<Dataset>,
    pub cases: Vec<Case>,
}

/// A named dataset defining table inputs in a neutral format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dataset {
    pub id: String,
    #[serde(default)]
    pub columns: Vec<DatasetColumn>,
    /// Optional JSONL rows used for nested or mixed-shape datasets that are
    /// easier to express row-wise than as column-wise scalar arrays.
    #[serde(default)]
    pub json_lines: Vec<String>,
    /// Optional columns that should be dictionary-encoded after the base
    /// DataFrame is loaded.
    #[serde(default)]
    pub dictionary_columns: Vec<String>,
}

/// A single column in a dataset with explicit dtype and values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetColumn {
    pub name: String,
    pub dtype: String,
    pub values: Vec<serde_json::Value>,
}

/// A single test case binding a dataset to an operation with comparison policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Case {
    pub id: String,
    pub dataset: String,
    pub operation: Operation,
    pub expect: Expect,
    pub compare: Compare,
}

/// Operation to execute, dispatched by `kind`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Operation {
    Info { detail: Vec<String> },
    Schema {},
    Dtypes {},
    Select { columns: Vec<String> },
    Rename { mapping: IndexMap<String, String> },
    SortValues { by: Vec<String>, ascending: Vec<bool>, null_position: String },
    FilterEq { column: String, value: serde_json::Value, value_dtype: String },
    DropNulls { subset: Option<Vec<String>> },
    Head { n: usize },
    Tail { n: usize },
}

/// Expected result kind for a case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Expect {
    pub result_kind: ResultKind,
}

/// The kind of normalized result a case produces.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResultKind {
    Dataframe,
    InfoSnapshot,
    ColumnList,
    Error,
}

/// Comparison policy for a case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Compare {
    /// Comparison mode for inspection cases (e.g. "strict").
    pub mode: Option<String>,
    /// Row ordering policy: "exact" or "ignore".
    pub row_order: Option<String>,
    /// Float tolerance specification.
    pub float_tol: Option<FloatTol>,
    /// Error comparison mode (e.g. "exact").
    pub error_kind: Option<String>,
}

/// Float tolerance specification for transform comparisons.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FloatTol {
    /// Absolute tolerance.
    pub abs: f64,
    /// Optional relative tolerance.
    pub rel: Option<f64>,
    /// If present, restrict tolerance to these named columns only.
    pub columns: Option<Vec<String>>,
}

// --- Canonical dtype mapping ---

/// Maps a language-native dtype string to its manifest-canonical name.
///
/// Manifest-canonical names: int32, int64, uint32, uint64, float32, float64,
/// bool8, string, dictionary32, list, struct.
/// Both Python cuDF and cudf-rs may use different names for the same types.
pub fn canonical_dtype(native: &str) -> &str {
    match native {
        // Manifest-canonical names pass through
        "int32" | "int64" | "uint32" | "uint64" | "float32" | "float64" | "bool8" | "string" | "dictionary32"
        | "list" | "struct" => native,
        // Rust cudf-rs DataType display names
        "INT32" => "int32",
        "INT64" => "int64",
        "UINT32" => "uint32",
        "UINT64" => "uint64",
        "FLOAT32" => "float32",
        "FLOAT64" => "float64",
        "BOOL8" => "bool8",
        "STRING" => "string",
        "DICTIONARY32" => "dictionary32",
        "LIST" => "list",
        "STRUCT" => "struct",
        // Python cuDF dtype names
        "int" => "int32",
        "long" => "int64",
        "bool" | "boolean" => "bool8",
        "float" => "float32",
        "double" => "float64",
        "str" | "object" => "string",
        "category" | "categorical" => "dictionary32",
        // Fallback: return as-is (executor should handle unknown)
        other => other,
    }
}

impl Fixture {
    /// Build an index of datasets by ID for fast lookup.
    pub fn dataset_index(&self) -> HashMap<&str, &Dataset> {
        self.datasets.iter().map(|d| (d.id.as_str(), d)).collect()
    }

    /// Validate internal consistency: no duplicate IDs, all case dataset refs exist.
    pub fn validate(&self) -> std::result::Result<(), String> {
        // Check duplicate dataset IDs
        let mut seen = std::collections::HashSet::new();
        for ds in &self.datasets {
            if !seen.insert(&ds.id) {
                return Err(format!("duplicate dataset ID: {}", ds.id));
            }
            let has_columns = !ds.columns.is_empty();
            let has_json_lines = !ds.json_lines.is_empty();
            if has_columns == has_json_lines {
                return Err(format!(
                    "dataset '{}' must define exactly one input source: either non-empty 'columns' or non-empty 'json_lines'",
                    ds.id
                ));
            }
        }

        // Check duplicate case IDs
        let mut case_seen = std::collections::HashSet::new();
        for case in &self.cases {
            if !case_seen.insert(&case.id) {
                return Err(format!("duplicate case ID: {}", case.id));
            }
        }

        // Check all case dataset refs exist
        let ds_index = self.dataset_index();
        for case in &self.cases {
            if !ds_index.contains_key(case.dataset.as_str()) {
                return Err(format!("case '{}' references unknown dataset '{}'", case.id, case.dataset));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_fixture_json() {
        let json = r#"{
            "schema_version": 1,
            "datasets": [
                {
                    "id": "nullable_mixed_small",
                    "columns": [
                        { "name": "a", "dtype": "int32", "values": [1, null, 3, 4] },
                        { "name": "b", "dtype": "float64", "values": [1.5, 2.5, null, 4.5] },
                        { "name": "c", "dtype": "string", "values": ["x", null, "z", "w"] }
                    ]
                }
            ],
            "cases": [
                {
                    "id": "info_nullable_mixed",
                    "dataset": "nullable_mixed_small",
                    "operation": {
                        "kind": "info",
                        "detail": ["columns", "null_counts", "payload_bytes", "device_allocated"]
                    },
                    "expect": { "result_kind": "info_snapshot" },
                    "compare": { "mode": "strict" }
                },
                {
                    "id": "schema_nullable_mixed",
                    "dataset": "nullable_mixed_small",
                    "operation": { "kind": "schema" },
                    "expect": { "result_kind": "column_list" },
                    "compare": { "mode": "strict" }
                },
                {
                    "id": "sort_single_key",
                    "dataset": "nullable_mixed_small",
                    "operation": {
                        "kind": "sort_values",
                        "by": ["a"],
                        "ascending": [true],
                        "null_position": "last"
                    },
                    "expect": { "result_kind": "dataframe" },
                    "compare": { "row_order": "exact" }
                },
                {
                    "id": "filter_eq_int",
                    "dataset": "nullable_mixed_small",
                    "operation": {
                        "kind": "filter_eq",
                        "column": "a",
                        "value": 3,
                        "value_dtype": "int32"
                    },
                    "expect": { "result_kind": "dataframe" },
                    "compare": { "row_order": "exact" }
                },
                {
                    "id": "select_missing_error",
                    "dataset": "nullable_mixed_small",
                    "operation": {
                        "kind": "select",
                        "columns": ["nonexistent"]
                    },
                    "expect": { "result_kind": "error" },
                    "compare": { "error_kind": "exact" }
                },
                {
                    "id": "rename_columns",
                    "dataset": "nullable_mixed_small",
                    "operation": {
                        "kind": "rename",
                        "mapping": { "a": "alpha", "b": "beta" }
                    },
                    "expect": { "result_kind": "dataframe" },
                    "compare": { "row_order": "exact" }
                },
                {
                    "id": "head_two",
                    "dataset": "nullable_mixed_small",
                    "operation": { "kind": "head", "n": 2 },
                    "expect": { "result_kind": "dataframe" },
                    "compare": { "row_order": "exact" }
                },
                {
                    "id": "drop_nulls_all",
                    "dataset": "nullable_mixed_small",
                    "operation": { "kind": "drop_nulls" },
                    "expect": { "result_kind": "dataframe" },
                    "compare": { "row_order": "exact" }
                }
            ]
        }"#;

        let fixture: Fixture = serde_json::from_str(json).expect("parse fixture JSON");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.datasets.len(), 1);
        assert_eq!(fixture.cases.len(), 8);

        // Validate internal consistency
        fixture.validate().expect("fixture should be internally consistent");

        // Check info operation parsed correctly
        let info_case = &fixture.cases[0];
        assert_eq!(info_case.id, "info_nullable_mixed");
        match &info_case.operation {
            Operation::Info { detail } => {
                assert_eq!(detail.len(), 4);
                assert_eq!(detail[0], "columns");
            },
            other => panic!("expected Info, got {:?}", other),
        }
        assert_eq!(info_case.expect.result_kind, ResultKind::InfoSnapshot);

        // Check sort_values operation
        let sort_case = &fixture.cases[2];
        match &sort_case.operation {
            Operation::SortValues { by, ascending, null_position } => {
                assert_eq!(by, &["a"]);
                assert_eq!(ascending, &[true]);
                assert_eq!(null_position, "last");
            },
            other => panic!("expected SortValues, got {:?}", other),
        }

        // Check filter_eq operation
        let filter_case = &fixture.cases[3];
        match &filter_case.operation {
            Operation::FilterEq { column, value, value_dtype } => {
                assert_eq!(column, "a");
                assert_eq!(value, &serde_json::json!(3));
                assert_eq!(value_dtype, "int32");
            },
            other => panic!("expected FilterEq, got {:?}", other),
        }

        // Check rename operation with ordered mapping
        let rename_case = &fixture.cases[5];
        match &rename_case.operation {
            Operation::Rename { mapping } => {
                let keys: Vec<&String> = mapping.keys().collect();
                assert_eq!(keys, vec!["a", "b"]);
                assert_eq!(mapping["a"], "alpha");
                assert_eq!(mapping["b"], "beta");
            },
            other => panic!("expected Rename, got {:?}", other),
        }

        // Check drop_nulls without subset
        let drop_case = &fixture.cases[7];
        match &drop_case.operation {
            Operation::DropNulls { subset } => {
                assert!(subset.is_none());
            },
            other => panic!("expected DropNulls, got {:?}", other),
        }

        // Round-trip through serialize then deserialize
        let serialized = serde_json::to_string(&fixture).expect("serialize fixture");
        let deserialized: Fixture = serde_json::from_str(&serialized).expect("deserialize fixture");
        assert_eq!(fixture, deserialized);
    }

    #[test]
    fn canonical_dtype_mapping() {
        // Pass-through canonical names
        assert_eq!(canonical_dtype("int32"), "int32");
        assert_eq!(canonical_dtype("float64"), "float64");
        assert_eq!(canonical_dtype("bool8"), "bool8");
        assert_eq!(canonical_dtype("string"), "string");
        assert_eq!(canonical_dtype("dictionary32"), "dictionary32");
        assert_eq!(canonical_dtype("list"), "list");
        assert_eq!(canonical_dtype("struct"), "struct");

        // Rust cudf-rs uppercase names
        assert_eq!(canonical_dtype("INT32"), "int32");
        assert_eq!(canonical_dtype("FLOAT64"), "float64");
        assert_eq!(canonical_dtype("BOOL8"), "bool8");
        assert_eq!(canonical_dtype("STRING"), "string");
        assert_eq!(canonical_dtype("DICTIONARY32"), "dictionary32");
        assert_eq!(canonical_dtype("LIST"), "list");
        assert_eq!(canonical_dtype("STRUCT"), "struct");

        // Python cuDF names
        assert_eq!(canonical_dtype("bool"), "bool8");
        assert_eq!(canonical_dtype("boolean"), "bool8");
        assert_eq!(canonical_dtype("object"), "string");
        assert_eq!(canonical_dtype("category"), "dictionary32");

        // Unmapped names pass through
        assert_eq!(canonical_dtype("datetime64[ns]"), "datetime64[ns]");
    }

    #[test]
    fn validate_catches_duplicate_dataset_id() {
        let fixture = Fixture {
            schema_version: 1,
            datasets: vec![
                Dataset {
                    id: "ds1".into(),
                    columns: vec![DatasetColumn {
                        name: "a".into(),
                        dtype: "int32".into(),
                        values: vec![serde_json::json!(1)],
                    }],
                    json_lines: vec![],
                    dictionary_columns: vec![],
                },
                Dataset {
                    id: "ds1".into(),
                    columns: vec![DatasetColumn {
                        name: "a".into(),
                        dtype: "int32".into(),
                        values: vec![serde_json::json!(2)],
                    }],
                    json_lines: vec![],
                    dictionary_columns: vec![],
                },
            ],
            cases: vec![],
        };
        assert!(fixture.validate().is_err());
        assert!(fixture.validate().unwrap_err().contains("duplicate dataset ID"));
    }

    #[test]
    fn validate_catches_missing_dataset_ref() {
        let fixture = Fixture {
            schema_version: 1,
            datasets: vec![Dataset {
                id: "ds1".into(),
                columns: vec![DatasetColumn {
                    name: "a".into(),
                    dtype: "int32".into(),
                    values: vec![serde_json::json!(1)],
                }],
                json_lines: vec![],
                dictionary_columns: vec![],
            }],
            cases: vec![Case {
                id: "case1".into(),
                dataset: "nonexistent".into(),
                operation: Operation::Schema {},
                expect: Expect { result_kind: ResultKind::ColumnList },
                compare: Compare::default(),
            }],
        };
        assert!(fixture.validate().is_err());
        assert!(fixture.validate().unwrap_err().contains("unknown dataset"));
    }

    #[test]
    fn validate_catches_duplicate_case_id() {
        let fixture = Fixture {
            schema_version: 1,
            datasets: vec![Dataset {
                id: "ds1".into(),
                columns: vec![DatasetColumn {
                    name: "a".into(),
                    dtype: "int32".into(),
                    values: vec![serde_json::json!(1)],
                }],
                json_lines: vec![],
                dictionary_columns: vec![],
            }],
            cases: vec![
                Case {
                    id: "case1".into(),
                    dataset: "ds1".into(),
                    operation: Operation::Schema {},
                    expect: Expect { result_kind: ResultKind::ColumnList },
                    compare: Compare::default(),
                },
                Case {
                    id: "case1".into(),
                    dataset: "ds1".into(),
                    operation: Operation::Dtypes {},
                    expect: Expect { result_kind: ResultKind::ColumnList },
                    compare: Compare::default(),
                },
            ],
        };
        assert!(fixture.validate().is_err());
        assert!(fixture.validate().unwrap_err().contains("duplicate case ID"));
    }

    #[test]
    fn parse_json_lines_dataset_with_dictionary_columns() {
        let json = r#"{
            "schema_version": 1,
            "datasets": [
                {
                    "id": "dictionary_nested",
                    "json_lines": [
                        "{\"category\":\"beta\",\"nested\":[[1,2]],\"point\":{\"coords\":[1,2],\"label\":\"one\"}}",
                        "{\"category\":\"alpha\",\"nested\":[[],[3]],\"point\":{\"coords\":[],\"label\":\"two\"}}"
                    ],
                    "dictionary_columns": ["category"]
                }
            ],
            "cases": [
                {
                    "id": "schema_dictionary_nested",
                    "dataset": "dictionary_nested",
                    "operation": { "kind": "schema" },
                    "expect": { "result_kind": "column_list" },
                    "compare": { "mode": "strict" }
                }
            ]
        }"#;

        let fixture: Fixture = serde_json::from_str(json).expect("parse fixture JSON");
        fixture.validate().expect("fixture should be internally consistent");

        let dataset = &fixture.datasets[0];
        assert!(dataset.columns.is_empty());
        assert_eq!(dataset.json_lines.len(), 2);
        assert_eq!(dataset.dictionary_columns, vec!["category"]);
    }

    #[test]
    fn validate_rejects_dataset_with_both_columns_and_json_lines() {
        let fixture = Fixture {
            schema_version: 1,
            datasets: vec![Dataset {
                id: "invalid".into(),
                columns: vec![DatasetColumn {
                    name: "a".into(),
                    dtype: "int32".into(),
                    values: vec![serde_json::json!(1)],
                }],
                json_lines: vec!["{\"a\":1}".into()],
                dictionary_columns: vec![],
            }],
            cases: vec![],
        };

        assert!(fixture.validate().is_err());
        assert!(fixture.validate().unwrap_err().contains("must define exactly one input source"));
    }
}
