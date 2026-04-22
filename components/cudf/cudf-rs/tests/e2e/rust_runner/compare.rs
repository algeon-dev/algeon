//! Structured comparison engine for the E2E conformance test framework.
//!
//! Compares normalized results from the Rust and Python executors,
//! producing precise per-field `Mismatch` records for failure reporting.

use super::{
    manifest::{Case, Compare, Operation},
    normalize::{
        NormalizedColumnList, NormalizedColumnMeta, NormalizedDataFrame, NormalizedError, NormalizedInfoSnapshot,
        NormalizedResult,
    },
};

/// A single field-level difference between Rust and Python results.
#[derive(Debug, Clone)]
pub struct Mismatch {
    pub case_id: String,
    pub dataset_id: String,
    pub operation_kind: String,
    pub path: String,
    pub expected: String,
    pub actual: String,
}

impl Mismatch {
    pub fn display(&self) -> String {
        format!(
            "[e2e][{}] dataset={} operation={} path={}
expected: {}
actual:   {}",
            self.case_id, self.dataset_id, self.operation_kind, self.path, self.expected, self.actual
        )
    }
}

/// Compare normalized results from Rust and Python executors.
///
/// Returns a list of mismatches. An empty list means the results are equivalent
/// under the comparison policy defined by the case.
pub fn compare_results(case: &Case, rust: &NormalizedResult, python: &NormalizedResult) -> Vec<Mismatch> {
    let ctx = MismatchCtx {
        case_id: &case.id,
        dataset_id: &case.dataset,
        operation_kind: operation_kind_str(&case.operation),
    };

    // First check that result kinds match
    let rust_kind = result_kind_str(rust);
    let python_kind = result_kind_str(python);
    if rust_kind != python_kind {
        return vec![ctx.mismatch("result_kind", python_kind, rust_kind)];
    }

    match (rust, python) {
        (NormalizedResult::InfoSnapshot(r), NormalizedResult::InfoSnapshot(p)) => {
            compare_info_snapshot(&ctx, &case.compare, r, p)
        },
        (NormalizedResult::ColumnList(r), NormalizedResult::ColumnList(p)) => compare_column_list(&ctx, r, p),
        (NormalizedResult::Dataframe(r), NormalizedResult::Dataframe(p)) => {
            compare_dataframe(&ctx, &case.compare, r, p)
        },
        (NormalizedResult::Error(r), NormalizedResult::Error(p)) => compare_error(&ctx, &case.compare, r, p),
        _ => unreachable!("result kind mismatch already handled above"),
    }
}

// --- Comparison context helper ---

struct MismatchCtx<'a> {
    case_id: &'a str,
    dataset_id: &'a str,
    operation_kind: &'static str,
}

impl<'a> MismatchCtx<'a> {
    fn mismatch(&self, path: &str, expected: &str, actual: &str) -> Mismatch {
        Mismatch {
            case_id: self.case_id.to_string(),
            dataset_id: self.dataset_id.to_string(),
            operation_kind: self.operation_kind.to_string(),
            path: path.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }
}

// --- InfoSnapshot comparison ---

fn compare_info_snapshot(
    ctx: &MismatchCtx<'_>,
    _compare: &Compare,
    rust: &NormalizedInfoSnapshot,
    python: &NormalizedInfoSnapshot,
) -> Vec<Mismatch> {
    let mut mismatches = Vec::new();

    if rust.row_count != python.row_count {
        mismatches.push(ctx.mismatch("row_count", &python.row_count.to_string(), &rust.row_count.to_string()));
    }
    if rust.column_count != python.column_count {
        mismatches.push(ctx.mismatch("column_count", &python.column_count.to_string(), &rust.column_count.to_string()));
    }

    // dtype_counts: order-insensitive set comparison
    compare_dtype_counts(ctx, &rust.dtype_counts, &python.dtype_counts, &mut mismatches);

    // Per-column info comparison (in order)
    let max_cols = rust.columns.len().max(python.columns.len());
    for i in 0..max_cols {
        match (rust.columns.get(i), python.columns.get(i)) {
            (Some(r), Some(p)) => {
                compare_column_info(ctx, i, r, p, &mut mismatches);
            },
            (Some(r), None) => {
                mismatches.push(ctx.mismatch(
                    &format!("columns[{}]", i),
                    "<missing>",
                    &format!("{}:{}", r.name, r.dtype),
                ));
            },
            (None, Some(p)) => {
                mismatches.push(ctx.mismatch(
                    &format!("columns[{}]", i),
                    &format!("{}:{}", p.name, p.dtype),
                    "<missing>",
                ));
            },
            (None, None) => unreachable!(),
        }
    }

    // Top-level memory accounting
    compare_optional_u64(ctx, "payload_bytes", rust.payload_bytes, python.payload_bytes, &mut mismatches);
    compare_optional_u64(
        ctx,
        "device_allocated_bytes",
        rust.device_allocated_bytes,
        python.device_allocated_bytes,
        &mut mismatches,
    );

    mismatches
}

fn compare_dtype_counts(
    ctx: &MismatchCtx<'_>,
    rust: &[super::normalize::DtypeCount],
    python: &[super::normalize::DtypeCount],
    mismatches: &mut Vec<Mismatch>,
) {
    use std::collections::HashMap;

    let rust_map: HashMap<&str, usize> = rust.iter().map(|d| (d.dtype.as_str(), d.count)).collect();
    let python_map: HashMap<&str, usize> = python.iter().map(|d| (d.dtype.as_str(), d.count)).collect();

    // Collect all dtypes from both sides
    let mut all_dtypes: Vec<&str> = rust_map.keys().chain(python_map.keys()).copied().collect();
    all_dtypes.sort();
    all_dtypes.dedup();

    for dtype in all_dtypes {
        let r = rust_map.get(dtype).copied().unwrap_or(0);
        let p = python_map.get(dtype).copied().unwrap_or(0);
        if r != p {
            mismatches.push(ctx.mismatch(&format!("dtype_counts.{}", dtype), &p.to_string(), &r.to_string()));
        }
    }
}

fn compare_column_info(
    ctx: &MismatchCtx<'_>,
    index: usize,
    rust: &super::normalize::NormalizedColumnInfo,
    python: &super::normalize::NormalizedColumnInfo,
    mismatches: &mut Vec<Mismatch>,
) {
    let prefix = format!("columns[{}]", index);

    if rust.name != python.name {
        mismatches.push(ctx.mismatch(&format!("{}.name", prefix), &python.name, &rust.name));
    }
    if rust.dtype != python.dtype {
        mismatches.push(ctx.mismatch(&format!("{}.dtype", prefix), &python.dtype, &rust.dtype));
    }
    if rust.nullable != python.nullable {
        mismatches.push(ctx.mismatch(
            &format!("{}.nullable", prefix),
            &python.nullable.to_string(),
            &rust.nullable.to_string(),
        ));
    }
    compare_optional_usize(ctx, &format!("{}.null_count", prefix), rust.null_count, python.null_count, mismatches);
    compare_optional_usize(
        ctx,
        &format!("{}.non_null_count", prefix),
        rust.non_null_count,
        python.non_null_count,
        mismatches,
    );
    compare_optional_u64(
        ctx,
        &format!("{}.payload_bytes", prefix),
        rust.payload_bytes,
        python.payload_bytes,
        mismatches,
    );
    compare_optional_u64(
        ctx,
        &format!("{}.device_allocated_bytes", prefix),
        rust.device_allocated_bytes,
        python.device_allocated_bytes,
        mismatches,
    );
}

// --- ColumnList comparison ---

fn compare_column_list(
    ctx: &MismatchCtx<'_>,
    rust: &NormalizedColumnList,
    python: &NormalizedColumnList,
) -> Vec<Mismatch> {
    let mut mismatches = Vec::new();

    let max_len = rust.entries.len().max(python.entries.len());
    if rust.entries.len() != python.entries.len() {
        mismatches.push(ctx.mismatch(
            "entries.len",
            &python.entries.len().to_string(),
            &rust.entries.len().to_string(),
        ));
    }

    for i in 0..max_len {
        match (rust.entries.get(i), python.entries.get(i)) {
            (Some(r), Some(p)) => {
                if r.name != p.name {
                    mismatches.push(ctx.mismatch(&format!("entries[{}].name", i), &p.name, &r.name));
                }
                if r.dtype != p.dtype {
                    mismatches.push(ctx.mismatch(&format!("entries[{}].dtype", i), &p.dtype, &r.dtype));
                }
            },
            (Some(r), None) => {
                mismatches.push(ctx.mismatch(
                    &format!("entries[{}]", i),
                    "<missing>",
                    &format!("{}:{}", r.name, r.dtype),
                ));
            },
            (None, Some(p)) => {
                mismatches.push(ctx.mismatch(
                    &format!("entries[{}]", i),
                    &format!("{}:{}", p.name, p.dtype),
                    "<missing>",
                ));
            },
            (None, None) => unreachable!(),
        }
    }

    mismatches
}

// --- DataFrame comparison ---

fn compare_dataframe(
    ctx: &MismatchCtx<'_>,
    compare: &Compare,
    rust: &NormalizedDataFrame,
    python: &NormalizedDataFrame,
) -> Vec<Mismatch> {
    let mut mismatches = Vec::new();

    // Compare column schema
    compare_column_schema(ctx, &rust.columns, &python.columns, &mut mismatches);

    // Compare row count
    if rust.row_count != python.row_count {
        mismatches.push(ctx.mismatch("row_count", &python.row_count.to_string(), &rust.row_count.to_string()));
    }

    // Early return if schema mismatch — row comparison would be meaningless
    if !mismatches.is_empty() {
        return mismatches;
    }

    let ignore_order = compare.row_order.as_deref() == Some("ignore");
    let float_tol = compare.float_tol.as_ref();

    // Determine float column indices for tolerance (only when float_tol is specified)
    let float_col_indices: Vec<usize> = match float_tol {
        Some(ft) => rust
            .columns
            .iter()
            .enumerate()
            .filter(|(_i, col)| {
                let is_float = col.dtype == "float32" || col.dtype == "float64";
                if !is_float {
                    return false;
                }
                // If float_tol specifies columns, only those are tolerated
                match ft.columns.as_ref() {
                    Some(names) => names.contains(&col.name),
                    None => true,
                }
            })
            .map(|(i, _)| i)
            .collect(),
        None => Vec::new(),
    };

    let (rust_rows, python_rows) = if ignore_order {
        let mut r = rust.rows.clone();
        let mut p = python.rows.clone();
        sort_rows(&mut r);
        sort_rows(&mut p);
        (r, p)
    } else {
        (rust.rows.clone(), python.rows.clone())
    };

    // Compare rows cell-by-cell
    for (row_idx, (r_row, p_row)) in rust_rows.iter().zip(python_rows.iter()).enumerate() {
        for (col_idx, (r_val, p_val)) in r_row.iter().zip(p_row.iter()).enumerate() {
            let is_tolerated_float = float_col_indices.contains(&col_idx);
            let matches = if is_tolerated_float {
                compare_values_with_tolerance(r_val, p_val, float_tol.unwrap())
            } else {
                compare_values_exact(r_val, p_val)
            };
            if !matches {
                mismatches.push(ctx.mismatch(
                    &format!("rows[{}][{}]", row_idx, col_idx),
                    &format_json_value(p_val),
                    &format_json_value(r_val),
                ));
            }
        }
    }

    mismatches
}

fn compare_column_schema(
    ctx: &MismatchCtx<'_>,
    rust: &[NormalizedColumnMeta],
    python: &[NormalizedColumnMeta],
    mismatches: &mut Vec<Mismatch>,
) {
    if rust.len() != python.len() {
        mismatches.push(ctx.mismatch("schema.num_columns", &python.len().to_string(), &rust.len().to_string()));
        return;
    }

    for (i, (r, p)) in rust.iter().zip(python.iter()).enumerate() {
        if r.name != p.name {
            mismatches.push(ctx.mismatch(&format!("schema[{}].name", i), &p.name, &r.name));
        }
        if r.dtype != p.dtype {
            mismatches.push(ctx.mismatch(&format!("schema[{}].dtype", i), &p.dtype, &r.dtype));
        }
    }
}

/// Sort rows lexicographically for order-insensitive comparison.
fn sort_rows(rows: &mut [Vec<serde_json::Value>]) {
    rows.sort_by(|a, b| {
        for (av, bv) in a.iter().zip(b.iter()) {
            let cmp = compare_json_values_for_sort(av, bv);
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    });
}

/// Compare two JSON values for sorting purposes.
fn compare_json_values_for_sort(a: &serde_json::Value, b: &serde_json::Value) -> std::cmp::Ordering {
    use serde_json::Value;
    match (a, b) {
        // Nulls sort last
        (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
        (Value::Null, _) => std::cmp::Ordering::Greater,
        (_, Value::Null) => std::cmp::Ordering::Less,
        // Numbers
        (Value::Number(an), Value::Number(bn)) => {
            let af = an.as_f64().unwrap_or(0.0);
            let bf = bn.as_f64().unwrap_or(0.0);
            af.partial_cmp(&bf).unwrap_or(std::cmp::Ordering::Equal)
        },
        // Strings
        (Value::String(a), Value::String(b)) => a.cmp(b),
        // Bools
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        // Mixed types: use kind ordering
        _ => value_kind_rank(a).cmp(&value_kind_rank(b)),
    }
}

fn value_kind_rank(v: &serde_json::Value) -> u8 {
    match v {
        serde_json::Value::Null => 0,
        serde_json::Value::Bool(_) => 1,
        serde_json::Value::Number(_) => 2,
        serde_json::Value::String(_) => 3,
        serde_json::Value::Array(_) => 4,
        serde_json::Value::Object(_) => 5,
    }
}

/// Exact comparison of two JSON values (null-aware).
fn compare_values_exact(rust: &serde_json::Value, python: &serde_json::Value) -> bool {
    rust == python
}

/// Tolerance-aware comparison for float values.
fn compare_values_with_tolerance(
    rust: &serde_json::Value,
    python: &serde_json::Value,
    tol: &super::manifest::FloatTol,
) -> bool {
    // Both null or both present
    match (rust.is_null(), python.is_null()) {
        (true, true) => return true,
        (true, false) | (false, true) => return false,
        _ => {},
    }

    // Extract as f64
    let r = rust.as_f64();
    let p = python.as_f64();
    match (r, p) {
        (Some(rv), Some(pv)) => {
            let abs_diff = (rv - pv).abs();
            if abs_diff <= tol.abs {
                return true;
            }
            if let Some(rel) = tol.rel {
                if abs_diff <= rel * pv.abs() {
                    return true;
                }
            }
            false
        },
        // If not both numeric, fall back to exact comparison
        _ => rust == python,
    }
}

// --- Error comparison ---

fn compare_error(
    ctx: &MismatchCtx<'_>,
    compare: &Compare,
    rust: &NormalizedError,
    python: &NormalizedError,
) -> Vec<Mismatch> {
    let mut mismatches = Vec::new();
    let exact = compare.error_kind.as_deref() == Some("exact");

    if rust.kind != python.kind {
        mismatches.push(ctx.mismatch("error.kind", &python.kind, &rust.kind));
    }

    if exact && rust.operation != python.operation {
        mismatches.push(ctx.mismatch("error.operation", &python.operation, &rust.operation));
    }

    mismatches
}

// --- Helpers ---

fn compare_optional_u64(
    ctx: &MismatchCtx<'_>,
    path: &str,
    rust: Option<u64>,
    python: Option<u64>,
    mismatches: &mut Vec<Mismatch>,
) {
    match (rust, python) {
        (Some(r), Some(p)) if r != p => {
            mismatches.push(ctx.mismatch(path, &p.to_string(), &r.to_string()));
        },
        (Some(r), None) => {
            mismatches.push(ctx.mismatch(path, "<absent>", &r.to_string()));
        },
        (None, Some(p)) => {
            mismatches.push(ctx.mismatch(path, &p.to_string(), "<absent>"));
        },
        _ => {},
    }
}

fn compare_optional_usize(
    ctx: &MismatchCtx<'_>,
    path: &str,
    rust: Option<usize>,
    python: Option<usize>,
    mismatches: &mut Vec<Mismatch>,
) {
    match (rust, python) {
        (Some(r), Some(p)) if r != p => {
            mismatches.push(ctx.mismatch(path, &p.to_string(), &r.to_string()));
        },
        (Some(r), None) => {
            mismatches.push(ctx.mismatch(path, "<absent>", &r.to_string()));
        },
        (None, Some(p)) => {
            mismatches.push(ctx.mismatch(path, &p.to_string(), "<absent>"));
        },
        _ => {},
    }
}

fn format_json_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::String(s) => format!("\"{}\"", s),
        other => other.to_string(),
    }
}

fn result_kind_str(r: &NormalizedResult) -> &'static str {
    match r {
        NormalizedResult::Dataframe(_) => "dataframe",
        NormalizedResult::InfoSnapshot(_) => "info_snapshot",
        NormalizedResult::ColumnList(_) => "column_list",
        NormalizedResult::Error(_) => "error",
    }
}

fn operation_kind_str(op: &Operation) -> &'static str {
    match op {
        Operation::Info { .. } => "info",
        Operation::Schema {} => "schema",
        Operation::Dtypes {} => "dtypes",
        Operation::Select { .. } => "select",
        Operation::Rename { .. } => "rename",
        Operation::SortValues { .. } => "sort_values",
        Operation::FilterEq { .. } => "filter_eq",
        Operation::DropNulls { .. } => "drop_nulls",
        Operation::Head { .. } => "head",
        Operation::Tail { .. } => "tail",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            manifest::{Expect, ResultKind},
            normalize::*,
        },
        *,
    };

    fn make_case(id: &str, dataset: &str, op: Operation, compare: Compare) -> Case {
        Case {
            id: id.to_string(),
            dataset: dataset.to_string(),
            operation: op,
            expect: Expect { result_kind: ResultKind::Dataframe },
            compare,
        }
    }

    #[test]
    fn identical_dataframes_produce_no_mismatches() {
        let df = NormalizedDataFrame {
            columns: vec![
                NormalizedColumnMeta { name: "a".into(), dtype: "int32".into() },
                NormalizedColumnMeta { name: "b".into(), dtype: "float64".into() },
            ],
            row_count: 2,
            rows: vec![
                vec![serde_json::json!(1), serde_json::json!(1.5)],
                vec![serde_json::json!(2), serde_json::json!(2.5)],
            ],
        };

        let case = make_case("test", "ds", Operation::Head { n: 2 }, Compare::default());
        let mismatches =
            compare_results(&case, &NormalizedResult::Dataframe(df.clone()), &NormalizedResult::Dataframe(df));
        assert!(mismatches.is_empty(), "expected no mismatches, got: {:?}", mismatches);
    }

    #[test]
    fn different_values_produce_mismatches() {
        let rust_df = NormalizedDataFrame {
            columns: vec![NormalizedColumnMeta { name: "a".into(), dtype: "int32".into() }],
            row_count: 1,
            rows: vec![vec![serde_json::json!(42)]],
        };
        let python_df = NormalizedDataFrame {
            columns: vec![NormalizedColumnMeta { name: "a".into(), dtype: "int32".into() }],
            row_count: 1,
            rows: vec![vec![serde_json::json!(99)]],
        };

        let case = make_case("test", "ds", Operation::Head { n: 1 }, Compare::default());
        let mismatches =
            compare_results(&case, &NormalizedResult::Dataframe(rust_df), &NormalizedResult::Dataframe(python_df));
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].path, "rows[0][0]");
        assert_eq!(mismatches[0].expected, "99");
        assert_eq!(mismatches[0].actual, "42");
    }

    #[test]
    fn float_tolerance_passes_within_abs() {
        let rust_df = NormalizedDataFrame {
            columns: vec![NormalizedColumnMeta { name: "x".into(), dtype: "float64".into() }],
            row_count: 1,
            rows: vec![vec![serde_json::json!(1.0000001)]],
        };
        let python_df = NormalizedDataFrame {
            columns: vec![NormalizedColumnMeta { name: "x".into(), dtype: "float64".into() }],
            row_count: 1,
            rows: vec![vec![serde_json::json!(1.0)]],
        };

        let compare_policy = Compare {
            float_tol: Some(super::super::manifest::FloatTol { abs: 1e-6, rel: None, columns: None }),
            ..Compare::default()
        };
        let case = make_case("test", "ds", Operation::Head { n: 1 }, compare_policy);
        let mismatches =
            compare_results(&case, &NormalizedResult::Dataframe(rust_df), &NormalizedResult::Dataframe(python_df));
        assert!(mismatches.is_empty(), "expected no mismatches within tolerance");
    }

    #[test]
    fn float_tolerance_fails_beyond_abs() {
        let rust_df = NormalizedDataFrame {
            columns: vec![NormalizedColumnMeta { name: "x".into(), dtype: "float64".into() }],
            row_count: 1,
            rows: vec![vec![serde_json::json!(2.0)]],
        };
        let python_df = NormalizedDataFrame {
            columns: vec![NormalizedColumnMeta { name: "x".into(), dtype: "float64".into() }],
            row_count: 1,
            rows: vec![vec![serde_json::json!(1.0)]],
        };

        let compare_policy = Compare {
            float_tol: Some(super::super::manifest::FloatTol { abs: 1e-6, rel: None, columns: None }),
            ..Compare::default()
        };
        let case = make_case("test", "ds", Operation::Head { n: 1 }, compare_policy);
        let mismatches =
            compare_results(&case, &NormalizedResult::Dataframe(rust_df), &NormalizedResult::Dataframe(python_df));
        assert_eq!(mismatches.len(), 1);
    }

    #[test]
    fn null_mismatch_detected() {
        let rust_df = NormalizedDataFrame {
            columns: vec![NormalizedColumnMeta { name: "a".into(), dtype: "int32".into() }],
            row_count: 1,
            rows: vec![vec![serde_json::Value::Null]],
        };
        let python_df = NormalizedDataFrame {
            columns: vec![NormalizedColumnMeta { name: "a".into(), dtype: "int32".into() }],
            row_count: 1,
            rows: vec![vec![serde_json::json!(0)]],
        };

        let case = make_case("test", "ds", Operation::Head { n: 1 }, Compare::default());
        let mismatches =
            compare_results(&case, &NormalizedResult::Dataframe(rust_df), &NormalizedResult::Dataframe(python_df));
        assert_eq!(mismatches.len(), 1);
        assert!(mismatches[0].actual.contains("null"));
    }

    #[test]
    fn result_kind_mismatch_detected() {
        let rust = NormalizedResult::Dataframe(NormalizedDataFrame { columns: vec![], row_count: 0, rows: vec![] });
        let python = NormalizedResult::Error(NormalizedError {
            kind: "invalid_argument".into(),
            operation: "select".into(),
            context: vec![],
        });

        let case = make_case("test", "ds", Operation::Head { n: 1 }, Compare::default());
        let mismatches = compare_results(&case, &rust, &python);
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].path, "result_kind");
    }
}
