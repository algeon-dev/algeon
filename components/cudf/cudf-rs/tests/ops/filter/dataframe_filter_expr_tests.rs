//! Tests for high-level fused row-predicate filtering entry points.
use cudf::{DataFrame, DataType, ErrorCode, ErrorKind, TypeId, ops::expr::col, testing::prelude::*};

use crate::common::{
    column_from_slice, column_from_strs, column_from_strs_nullable, dataframe_device_ptr, dataframe_from_columns,
    error_context_value, last_row_predicate_compute_mask_calls, last_row_predicate_filter_calls,
    last_row_predicate_final_filter_calls, last_row_predicate_mask_api_calls, reset_row_predicate_observation, series,
};

fn edge_df() -> cudf::DataFrame {
    let subject = column_from_strs(&["alice", "", "carol", "dave", "eve", "frank"]).unwrap();
    let object = column_from_strs(&["bob", "zara", "", "dave", "mallory", "grace"]).unwrap();
    let relation = column_from_strs(&["knows", "knows", "likes", "likes", "", "knows"]).unwrap();
    let weight = column_from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap();
    dataframe_from_columns(
        vec!["sourceId".into(), "targetId".into(), "relationId".into(), "weight".into()],
        vec![subject, object, relation, weight],
    )
    .unwrap()
}

fn wide_string_edge_df() -> cudf::DataFrame {
    let subject = column_from_strs(&["alice", "", "carol", "dave", "eve", "frank"]).unwrap();
    let object = column_from_strs(&["bob", "zara", "", "dave", "mallory", "grace"]).unwrap();
    let relation = column_from_strs(&["knows", "knows", "likes", "likes", "", "knows"]).unwrap();
    let payload_a = column_from_strs(&[
        "alpha-alpha-alpha",
        "bravo-bravo-bravo",
        "charlie-charlie-charlie",
        "delta-delta-delta",
        "echo-echo-echo",
        "foxtrot-foxtrot-foxtrot",
    ])
    .unwrap();
    let payload_b = column_from_strs(&[
        "north/north/north",
        "south/south/south",
        "east/east/east",
        "west/west/west",
        "center/center/center",
        "up/up/up",
    ])
    .unwrap();
    let payload_c = column_from_strs(&[
        "graph-edge-payload-a",
        "graph-edge-payload-b",
        "graph-edge-payload-c",
        "graph-edge-payload-d",
        "graph-edge-payload-e",
        "graph-edge-payload-f",
    ])
    .unwrap();
    let payload_d = column_from_strs(&[
        "long-string-0001",
        "long-string-0002",
        "long-string-0003",
        "long-string-0004",
        "long-string-0005",
        "long-string-0006",
    ])
    .unwrap();
    let weight = column_from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap();

    dataframe_from_columns(
        vec![
            "sourceId".into(),
            "targetId".into(),
            "relationId".into(),
            "payloadA".into(),
            "payloadB".into(),
            "payloadC".into(),
            "payloadD".into(),
            "weight".into(),
        ],
        vec![subject, object, relation, payload_a, payload_b, payload_c, payload_d, weight],
    )
    .unwrap()
}

#[test]
fn test_filter_expr_matches_rfc_style_multi_condition_filtering() {
    let df = edge_df();

    let filtered = df
        .filter_expr(
            col("sourceId").is_not_null()
                & col("targetId").is_not_null()
                & col("relationId").is_not_null()
                & col("sourceId").ne("")
                & col("targetId").ne("")
                & col("relationId").ne("")
                & col("sourceId").ne(col("targetId")),
        )
        .unwrap();

    assert_eq!(filtered.num_rows(), 2);
    assert_eq!(
        filtered.series("sourceId").unwrap().to_strings_strict().unwrap(),
        vec!["alice".to_string(), "frank".to_string()]
    );
    assert_eq!(
        filtered.series("targetId").unwrap().to_strings_strict().unwrap(),
        vec!["bob".to_string(), "grace".to_string()]
    );
    assert_eq!(filtered.series("weight").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 6]);
}

#[test]
fn test_filter_expr_and_filter_all_match_chained_filter_on_wide_string_heavy_dataframe() {
    let df = wide_string_edge_df();

    let chained = {
        let filtered = df.filter(&df.series("sourceId").unwrap().ne_scalar("").unwrap()).unwrap();
        let filtered = filtered.filter(&filtered.series("targetId").unwrap().ne_scalar("").unwrap()).unwrap();
        let filtered = filtered.filter(&filtered.series("relationId").unwrap().ne_scalar("").unwrap()).unwrap();
        filtered
            .filter(&filtered.series("sourceId").unwrap().ne(&filtered.series("targetId").unwrap()).unwrap())
            .unwrap()
    };

    let fused_all = df
        .filter_all([
            col("sourceId").ne(""),
            col("targetId").ne(""),
            col("relationId").ne(""),
            col("sourceId").ne(col("targetId")),
        ])
        .unwrap();

    let fused_expr = df
        .filter_expr(
            col("sourceId").ne("")
                & col("targetId").ne("")
                & col("relationId").ne("")
                & col("sourceId").ne(col("targetId")),
        )
        .unwrap();

    assert_eq!(chained.num_rows(), 2);
    assert_eq!(fused_all.num_rows(), chained.num_rows());
    assert_eq!(fused_expr.num_rows(), chained.num_rows());
    assert_eq!(fused_all.column_names(), chained.column_names());
    assert_eq!(fused_expr.column_names(), chained.column_names());

    for name in ["sourceId", "targetId", "relationId", "payloadA", "payloadB", "payloadC", "payloadD"] {
        let expected = chained.series(name).unwrap().to_strings_strict().unwrap();
        assert_eq!(fused_all.series(name).unwrap().to_strings_strict().unwrap(), expected);
        assert_eq!(fused_expr.series(name).unwrap().to_strings_strict().unwrap(), expected);
    }

    let expected_weight = chained.series("weight").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(fused_all.series("weight").unwrap().to_vec_strict::<i32>().unwrap(), expected_weight);
    assert_eq!(fused_expr.series("weight").unwrap().to_vec_strict::<i32>().unwrap(), expected_weight);
}

#[test]
fn test_filter_all_empty_returns_clone_without_materializing_new_table() {
    let df = edge_df();
    let original_ptr = dataframe_device_ptr(&df, "weight");

    let filtered = df.filter_all(std::iter::empty::<cudf::Expr>()).unwrap();

    assert_eq!(filtered.num_rows(), df.num_rows());
    assert_eq!(filtered.column_names(), df.column_names());
    assert_eq!(filtered.testing_execution_domain_id(), df.testing_execution_domain_id());
    assert_eq!(dataframe_device_ptr(&filtered, "weight"), original_ptr);
}

#[test]
fn test_filter_any_empty_returns_empty_dataframe_with_same_schema() {
    let df = edge_df();

    let filtered = df.filter_any(std::iter::empty::<cudf::Expr>()).unwrap();

    assert_eq!(filtered.num_rows(), 0);
    assert_eq!(filtered.column_names(), df.column_names());
    assert_eq!(filtered.column_index("sourceId"), Some(0));
    assert_eq!(filtered.column_index("weight"), Some(3));
}

#[test]
fn test_filter_any_empty_preserves_full_dtype_metadata() {
    let amount =
        series("amount", &[100i64, 250, 999]).unwrap().astype(DataType::fixed_point(TypeId::Decimal64, -2)).unwrap();
    let df = DataFrame::from_columns([("amount", amount.clone())]).unwrap();

    let filtered = df.filter_any(std::iter::empty::<cudf::Expr>()).unwrap();

    assert_eq!(filtered.num_rows(), 0);
    assert_eq!(filtered.column_names(), df.column_names());
    assert_eq!(filtered.series("amount").unwrap().dtype().unwrap(), amount.dtype().unwrap());
}

#[test]
fn test_filter_any_keeps_rows_matching_any_predicate() {
    let df = edge_df();

    let filtered =
        df.filter_any([col("sourceId").eq("alice"), col("weight").ge(5i32), col("targetId").eq("grace")]).unwrap();

    assert_eq!(filtered.num_rows(), 3);
    assert_eq!(
        filtered.series("sourceId").unwrap().to_strings_strict().unwrap(),
        vec!["alice".to_string(), "eve".to_string(), "frank".to_string()]
    );
    assert_eq!(filtered.series("weight").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 5, 6]);
}

#[test]
fn test_filter_expr_drops_null_predicate_rows() {
    let name = column_from_strs_nullable(&["alpha", "ignored", ""], &[true, false, true]).unwrap();
    let value = column_from_slice(&[10i32, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["name".into(), "value".into()], vec![name, value]).unwrap();

    let filtered = df.filter_expr(col("name").ne("")).unwrap();

    assert_eq!(filtered.num_rows(), 1);
    assert_eq!(filtered.series("name").unwrap().to_strings_strict().unwrap(), vec!["alpha".to_string()]);
    assert_eq!(filtered.series("value").unwrap().to_vec_strict::<i32>().unwrap(), vec![10]);
}

#[test]
fn test_filter_expr_timestamp_days_column_predicate_matches_tpch_date_shape() {
    let shipdate = series("shipdate", &[18_262i32, 18_263, 18_264])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    let cutoff = series("cutoff", &[18_263i32, 18_263, 18_263])
        .unwrap()
        .astype(TypeId::DurationDays)
        .unwrap()
        .astype(TypeId::TimestampDays)
        .unwrap();
    let id = series("id", &[1i32, 2, 3]).unwrap();
    let df = DataFrame::from_columns([("id", id), ("shipdate", shipdate), ("cutoff", cutoff)]).unwrap();

    let filtered = df.filter_expr(col("shipdate").lt(col("cutoff"))).unwrap();

    assert_eq!(filtered.num_rows(), 1);
    assert_eq!(filtered.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1]);
}

#[test]
fn test_filter_expr_rejects_unsupported_nodes_before_low_level_filter_runs() {
    let df = edge_df();

    reset_row_predicate_observation();
    let err = df.filter_expr(col("sourceId").str().lower().eq("alice")).unwrap_err();

    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_predicate"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("unsupported_expr_kind"));
    assert_eq!(error_context_value(&err, "expr_kind").as_deref(), Some("string_op"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("rewrite the predicate into currently supported fused-filter operators"),
    );
    assert_eq!(last_row_predicate_filter_calls(), 0);
    assert_eq!(last_row_predicate_compute_mask_calls(), 0);
    assert_eq!(last_row_predicate_final_filter_calls(), 0);
    assert_eq!(last_row_predicate_mask_api_calls(), 0);
}

#[test]
fn test_filter_expr_rejects_mixed_type_comparisons_before_low_level_filter_runs() {
    let df = edge_df();

    reset_row_predicate_observation();
    let err = df.filter_expr(col("sourceId").eq(5i32)).unwrap_err();

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("string"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "operator").as_deref(), Some("equal"));
    assert_eq!(last_row_predicate_filter_calls(), 0);
    assert_eq!(last_row_predicate_compute_mask_calls(), 0);
    assert_eq!(last_row_predicate_final_filter_calls(), 0);
    assert_eq!(last_row_predicate_mask_api_calls(), 0);
}

#[test]
fn test_filter_expr_rejects_non_boolean_predicate_output() {
    let df = edge_df();

    let err = df.filter_expr(col("weight")).unwrap_err();

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("bool8"));
}

#[test]
fn test_filter_expr_rejects_logical_not_on_non_boolean_operand() {
    let df = edge_df();

    let err = df.filter_expr(!col("weight")).unwrap_err();

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("bool8"));
}

#[test]
fn test_filter_expr_uses_fused_row_predicate_path_once() {
    let df = edge_df();

    reset_row_predicate_observation();
    let filtered = df
        .filter_expr(
            col("sourceId").ne("")
                & col("targetId").ne("")
                & col("relationId").ne("")
                & col("sourceId").ne(col("targetId")),
        )
        .unwrap();

    assert_eq!(filtered.num_rows(), 2);
    assert_eq!(last_row_predicate_filter_calls(), 1);
    assert_eq!(last_row_predicate_compute_mask_calls(), 1);
    assert_eq!(last_row_predicate_final_filter_calls(), 1);
    assert_eq!(last_row_predicate_mask_api_calls(), 0);
}
