use std::{path::PathBuf, process::Command};

use cudf::{DataFrame, Runtime};

const LOUVAIN_GRAPH_SRC_I32: &[i32] = &[0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5];
const LOUVAIN_GRAPH_DST_I32: &[i32] = &[1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4];
const LOUVAIN_GRAPH_WEIGHT_F32: &[f32] =
    &[0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];

fn unique_temp_path(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!("xtask_{prefix}_{}_{}.parquet", std::process::id(), chrono_like_now_nanos()))
}

fn chrono_like_now_nanos() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("system time").as_nanos()
}

fn write_small_louvain_fixture(path: &PathBuf) {
    let runtime = Runtime::new().expect("runtime");
    let src = runtime.series(LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = runtime.series(LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weight = runtime.series(LOUVAIN_GRAPH_WEIGHT_F32).expect("weight");
    let df = DataFrame::from_columns([("src", src), ("dst", dst), ("weight", weight)]).expect("dataframe");
    df.write_parquet(path).expect("write parquet");
}

fn write_small_louvain_alias_string_fixture(path: &PathBuf) {
    let runtime = Runtime::new().expect("runtime");
    let subject = runtime
        .series(&["v0", "v1", "v1", "v2", "v2", "v2", "v3", "v4", "v1", "v3", "v4", "v0", "v1", "v3", "v5", "v5"])
        .expect("sourceId");
    let object = runtime
        .series(&["v1", "v3", "v4", "v0", "v1", "v3", "v5", "v5", "v0", "v1", "v1", "v2", "v2", "v2", "v3", "v4"])
        .expect("targetId");
    let weight = runtime.series(LOUVAIN_GRAPH_WEIGHT_F32).expect("weight");
    let df =
        DataFrame::from_columns([("sourceId", subject), ("targetId", object), ("weight", weight)]).expect("dataframe");
    df.write_parquet(path).expect("write parquet");
}

fn write_small_louvain_alias_string_fixture_with_null_weights(path: &PathBuf) {
    let runtime = Runtime::new().expect("runtime");
    let subject = runtime
        .series(&["v0", "v1", "v1", "v2", "v2", "v2", "v3", "v4", "v1", "v3", "v4", "v0", "v1", "v3", "v5", "v5"])
        .expect("sourceId");
    let object = runtime
        .series(&["v1", "v3", "v4", "v0", "v1", "v3", "v5", "v5", "v0", "v1", "v1", "v2", "v2", "v2", "v3", "v4"])
        .expect("targetId");
    let weight = runtime
        .series_opt(&[
            Some(0.1f32),
            Some(2.1),
            None,
            Some(5.1),
            Some(3.1),
            None,
            Some(7.2),
            Some(3.2),
            Some(0.1),
            Some(2.1),
            None,
            Some(5.1),
            Some(3.1),
            Some(4.1),
            Some(7.2),
            None,
        ])
        .expect("weight");
    let df =
        DataFrame::from_columns([("sourceId", subject), ("targetId", object), ("weight", weight)]).expect("dataframe");
    df.write_parquet(path).expect("write parquet");
}

fn python_parity_environment_available() -> bool {
    Command::new("python3")
        .args(["-c", "import cudf, cugraph"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[test]
fn missing_fixture_returns_exit_2_and_structured_json() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["parity", "louvain", "--fixture", "/definitely/missing/louvain_fixture.parquet"])
        .output()
        .expect("run xtask parity command");

    assert_eq!(output.status.code(), Some(2));
    assert!(
        output.stderr.is_empty(),
        "xtask should keep stdout/stderr machine-oriented for parity output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("stdout json");
    assert_eq!(json["case"], "louvain");
    assert_eq!(json["verdict"], "error");
    assert_eq!(json["error"]["kind"], "environment_error");
    assert!(json["error"]["message"].as_str().expect("error message").contains("fixture not found"));
}

#[test]
fn small_generated_fixture_runs_end_to_end_without_environment_error() {
    if !python_parity_environment_available() {
        eprintln!("skipping end-to-end parity test because python cudf/cugraph is unavailable");
        return;
    }

    let fixture = unique_temp_path("louvain_small_fixture");
    write_small_louvain_fixture(&fixture);

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["parity", "louvain", "--fixture"])
        .arg(&fixture)
        .output()
        .expect("run xtask parity command");

    let _ = std::fs::remove_file(&fixture);

    assert!(
        output.stderr.is_empty(),
        "xtask should keep stdout/stderr machine-oriented for parity output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("stdout json");
    assert_eq!(json["case"], "louvain");
    assert_eq!(output.status.code(), Some(0), "expected parity pass, got {:?}", output.status.code());
    assert_eq!(json["verdict"], "pass");
    assert!(json["python_summary"].is_object(), "python summary missing: {json}");
    assert_eq!(json["rust_summaries"].as_array().expect("rust summaries").len(), 3);
    assert!(json["diffs"].as_array().expect("diffs").is_empty(), "expected no structured diffs: {json}");
}

#[test]
fn aliased_string_fixture_runs_end_to_end() {
    if !python_parity_environment_available() {
        eprintln!("skipping aliased-string parity test because python cudf/cugraph is unavailable");
        return;
    }

    let fixture = unique_temp_path("louvain_alias_string_fixture");
    write_small_louvain_alias_string_fixture(&fixture);

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["parity", "louvain", "--fixture"])
        .arg(&fixture)
        .output()
        .expect("run xtask parity command");

    let _ = std::fs::remove_file(&fixture);

    assert!(
        output.stderr.is_empty(),
        "xtask should keep stdout/stderr machine-oriented for parity output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("stdout json");
    assert_eq!(json["case"], "louvain");
    assert_eq!(output.status.code(), Some(0), "expected parity pass, got {:?}", output.status.code());
    assert_eq!(json["verdict"], "pass");
    assert!(json["python_summary"].is_object(), "python summary missing: {json}");
    assert_eq!(json["rust_summaries"].as_array().expect("rust summaries").len(), 3);
    assert!(json["diffs"].as_array().expect("diffs").is_empty(), "expected no structured diffs: {json}");
}

#[test]
fn aliased_string_fixture_with_null_weights_runs_end_to_end() {
    if !python_parity_environment_available() {
        eprintln!("skipping aliased-string-null-weight parity test because python cudf/cugraph is unavailable");
        return;
    }

    let fixture = unique_temp_path("louvain_alias_string_null_weight_fixture");
    write_small_louvain_alias_string_fixture_with_null_weights(&fixture);

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["parity", "louvain", "--fixture"])
        .arg(&fixture)
        .output()
        .expect("run xtask parity command");

    let _ = std::fs::remove_file(&fixture);

    assert!(
        output.stderr.is_empty(),
        "xtask should keep stdout/stderr machine-oriented for parity output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("stdout json");
    assert_eq!(json["case"], "louvain");
    assert_eq!(output.status.code(), Some(1), "expected parity mismatch, got {:?}", output.status.code());
    assert_eq!(json["verdict"], "mismatch");
    assert!(json["python_summary"].is_object(), "python summary missing: {json}");
    assert_eq!(json["rust_summaries"].as_array().expect("rust summaries").len(), 3);
    assert!(!json["diffs"].as_array().expect("diffs").is_empty(), "expected at least one structured diff: {json}");
}
