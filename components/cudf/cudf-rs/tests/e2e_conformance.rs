//! E2E conformance test harness entry point.
//!
//! Compares `cudf-rs` behavior against Python cuDF using shared JSON
//! fixture manifests as the single source of truth.
//!
//! Run with: `cargo test -p cudf --test e2e_conformance`

#[path = "e2e/rust_runner/mod.rs"]
mod rust_runner;

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use rust_runner::{
    ExecutionError, Fixture, Mismatch, NormalizedResult, build_dataframe, compare_results, execute_operation,
    manifest::{Dataset, DatasetColumn, Operation},
    normalize_manifest_error,
};

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

/// Load all `.json` fixture files from the given directory.
fn load_all_fixtures(dir: &Path) -> Vec<(PathBuf, Fixture)> {
    let mut fixtures = Vec::new();
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read fixture dir {}: {}", dir.display(), e));

    for entry in entries {
        let entry = entry.expect("read dir entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let contents =
                std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
            let fixture: Fixture =
                serde_json::from_str(&contents).unwrap_or_else(|e| panic!("cannot parse {}: {}", path.display(), e));
            fixture.validate().unwrap_or_else(|e| panic!("fixture validation failed for {}: {}", path.display(), e));
            fixtures.push((path, fixture));
        }
    }

    assert!(!fixtures.is_empty(), "no fixture files found in {}", dir.display());
    fixtures
}

// ---------------------------------------------------------------------------
// Python runner invocation
// ---------------------------------------------------------------------------

/// Resolve the Python interpreter path.
fn python_interpreter() -> String {
    std::env::var("CUDF_E2E_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// Whether the Python cuDF E2E suite is explicitly enabled for this run.
fn e2e_enabled_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(raw) => matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        None => false,
    }
}

/// Whether the Python cuDF E2E suite should run.
fn e2e_enabled() -> bool {
    e2e_enabled_from_env_value(std::env::var("CUDF_E2E").ok().as_deref())
}

/// Check that Python cuDF is importable. Panics if not.
fn check_python_environment() {
    let python = python_interpreter();
    let output = Command::new(&python).arg("-c").arg("import cudf; print(cudf.__version__)").output();

    match output {
        Ok(o) if o.status.success() => {
            let version = String::from_utf8_lossy(&o.stdout);
            eprintln!("[e2e] Python cuDF version: {}", version.trim());
        },
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!(
                "Python cuDF import failed (exit {}).\n\
                 E2E conformance requires Python cuDF.\n\
                 stderr: {}",
                o.status, stderr
            );
        },
        Err(e) => {
            panic!(
                "Failed to invoke Python interpreter '{}': {}\n\
                 E2E conformance requires Python cuDF.",
                python, e
            );
        },
    }
}

/// Run a single case through the Python runner and return the normalized result.
fn run_python_case(fixture_path: &Path, case_id: &str) -> NormalizedResult {
    let python = python_interpreter();
    let runner_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/python_runner.py");

    let output = Command::new(&python)
        .arg(&runner_path)
        .arg("--fixture")
        .arg(fixture_path)
        .arg("--case")
        .arg(case_id)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke Python runner for case '{}': {}", case_id, e));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Python runner failed for case '{}' (exit {}):\n{}", case_id, output.status, stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("cannot parse Python runner output for case '{}': {}\nstdout: {}", case_id, e, stdout)
    })
}

/// Run a Python-runner self-test that does not require Python cuDF.
fn run_python_selftest(args: &[&str]) -> serde_json::Value {
    let python = python_interpreter();
    let runner_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/python_runner.py");

    let output = Command::new(&python)
        .arg(&runner_path)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke Python runner self-test {:?}: {}", args, e));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Python runner self-test {:?} failed (exit {}):\n{}", args, output.status, stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("cannot parse Python runner self-test output {:?}: {}\nstdout: {}", args, e, stdout))
}

#[test]
fn e2e_opt_in_defaults_to_disabled() {
    assert!(!e2e_enabled_from_env_value(None));
    assert!(!e2e_enabled_from_env_value(Some("0")));
    assert!(!e2e_enabled_from_env_value(Some("false")));
    assert!(e2e_enabled_from_env_value(Some("1")));
    assert!(e2e_enabled_from_env_value(Some("true")));
}

#[test]
fn rust_error_normalization_uses_manifest_taxonomy_for_missing_columns_without_message_matching() {
    let err = cudf::Error::new(cudf::ErrorKind::NotFound, "schema lookup failed")
        .with_operation("dataframe::select")
        .with_context("column_name", "missing");

    let normalized = normalize_manifest_error(&err, "select");

    assert_eq!(normalized.kind, "column_not_found");
    assert_eq!(normalized.operation, "select");
}

#[test]
fn rust_error_normalization_no_longer_infers_missing_columns_from_invalid_operation() {
    let err = cudf::Error::new(cudf::ErrorKind::InvalidOperation, "legacy schema lookup failure")
        .with_operation("dataframe::select")
        .with_context("column_name", "missing");

    let normalized = normalize_manifest_error(&err, "select");

    assert_eq!(normalized.kind, "invalid_argument");
    assert_eq!(normalized.operation, "select");
}

fn simple_fixture_dataframe() -> cudf::DataFrame {
    let dataset = Dataset {
        id: "basic".into(),
        columns: vec![
            DatasetColumn { name: "id".into(), dtype: "int32".into(), values: vec![1.into(), 2.into(), 3.into()] },
            DatasetColumn {
                name: "name".into(),
                dtype: "string".into(),
                values: vec!["alice".into(), "bob".into(), "cara".into()],
            },
            DatasetColumn {
                name: "score".into(),
                dtype: "float64".into(),
                values: vec![1.5.into(), 2.5.into(), 3.5.into()],
            },
        ],
        json_lines: vec![],
        dictionary_columns: vec![],
    };

    build_dataframe(&dataset).expect("build test dataframe")
}

fn expect_normalized_missing_column_error(operation: Operation) {
    let df = simple_fixture_dataframe();
    let result = execute_operation(&df, &operation).expect("execute negative fixture operation");

    match result {
        NormalizedResult::Error(err) => {
            assert_eq!(err.kind, "column_not_found");
            assert!(err.context.iter().any(|(key, value)| key == "column_name" && value == "missing"));
        },
        other => panic!("expected normalized error result, got {:?}", other),
    }
}

#[test]
fn execute_select_missing_column_normalizes_from_direct_not_found() {
    expect_normalized_missing_column_error(Operation::Select { columns: vec!["missing".into()] });
}

#[test]
fn execute_rename_missing_column_normalizes_from_direct_not_found() {
    let mut mapping = indexmap::IndexMap::new();
    mapping.insert("missing".into(), "renamed".into());
    expect_normalized_missing_column_error(Operation::Rename { mapping });
}

#[test]
fn execute_sort_values_missing_column_normalizes_from_direct_not_found() {
    expect_normalized_missing_column_error(Operation::SortValues {
        by: vec!["missing".into()],
        ascending: vec![true],
        null_position: "last".into(),
    });
}

#[test]
fn execute_filter_eq_missing_column_normalizes_from_direct_not_found() {
    expect_normalized_missing_column_error(Operation::FilterEq {
        column: "missing".into(),
        value: 1.into(),
        value_dtype: "int32".into(),
    });
}

#[test]
fn execute_drop_nulls_missing_column_normalizes_from_direct_not_found() {
    expect_normalized_missing_column_error(Operation::DropNulls { subset: Some(vec!["missing".into()]) });
}

#[test]
fn python_runner_selftest_projects_info_detail_contract() {
    let projected = run_python_selftest(&["--selftest", "info-detail", "--detail", "columns"]);

    assert_eq!(projected["result_kind"], "info_snapshot");
    assert!(projected["columns"].is_array());
    assert_eq!(projected["columns"][0]["null_count"], serde_json::Value::Null);
    assert_eq!(projected["columns"][0]["payload_bytes"], serde_json::Value::Null);
    assert_eq!(projected["payload_bytes"], serde_json::Value::Null);
    assert_eq!(projected["device_allocated_bytes"], serde_json::Value::Null);
}

#[test]
fn python_runner_selftest_classifies_missing_column_as_column_not_found() {
    let classified = run_python_selftest(&["--selftest", "classify-error", "--error", "key_error"]);
    assert_eq!(classified["kind"], "column_not_found");
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Python cuDF; set CUDF_E2E=1 and run with --ignored"]
fn e2e_conformance() {
    if !e2e_enabled() {
        eprintln!("[e2e] skipping: set CUDF_E2E=1 to run Python cuDF conformance");
        return;
    }

    // 1. Check Python environment
    check_python_environment();

    // 2. Load fixtures
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/fixtures");
    let fixtures = load_all_fixtures(&fixture_dir);

    eprintln!(
        "[e2e] Loaded {} fixture file(s), {} total cases",
        fixtures.len(),
        fixtures.iter().map(|(_, f)| f.cases.len()).sum::<usize>()
    );

    // 3. Run all cases
    let mut all_mismatches: Vec<Mismatch> = Vec::new();
    let mut cases_run = 0;

    for (fixture_path, fixture) in &fixtures {
        let dataset_index = fixture.dataset_index();

        for case in &fixture.cases {
            let dataset = dataset_index
                .get(case.dataset.as_str())
                .unwrap_or_else(|| panic!("dataset '{}' not found (validated earlier)", case.dataset));

            // Build DataFrame from manifest dataset
            let df = build_dataframe(dataset).unwrap_or_else(|e| {
                panic!("[e2e][{}] failed to build DataFrame from dataset '{}': {}", case.id, case.dataset, e)
            });

            // Execute through Rust
            let rust_result = match execute_operation(&df, &case.operation) {
                Ok(result) => result,
                Err(ExecutionError::CudfError(e)) => {
                    // Expected for negative test cases
                    NormalizedResult::Error(rust_runner::NormalizedError {
                        kind: format!("{:?}", e.kind()).to_lowercase(),
                        operation: e.operation().to_string(),
                        context: e.context().iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                    })
                },
                Err(ExecutionError::Framework(msg)) => {
                    panic!("[e2e][{}] framework error: {}", case.id, msg);
                },
            };

            // Execute through Python
            let python_result = run_python_case(fixture_path, &case.id);

            // Compare
            let mismatches = compare_results(case, &rust_result, &python_result);
            all_mismatches.extend(mismatches);

            cases_run += 1;
        }
    }

    eprintln!("[e2e] Ran {} cases", cases_run);

    // 4. Report mismatches
    if !all_mismatches.is_empty() {
        let report: String = all_mismatches.iter().map(|m| m.display()).collect::<Vec<_>>().join("\n\n");
        panic!(
            "E2E conformance failures ({} mismatches across {} cases):\n\n{}",
            all_mismatches.len(),
            cases_run,
            report
        );
    }
}
