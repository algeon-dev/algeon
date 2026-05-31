use super::*;

#[test]
fn bounded_native_query_executes_root_filtered_semi_anti_joins_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-filtered-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);

    let semi = execute_bounded_equality_left_only_filtered_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftSemi,
    );
    let anti = execute_bounded_equality_left_only_filtered_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftAnti,
    );

    let QueryOutput::HostArrow(semi_batch) = semi.output() else {
        panic!("left semi bounded filtered join should return host output");
    };
    let mut semi_rows = int32_array(semi_batch, 0).values().iter().copied().collect::<Vec<_>>();
    semi_rows.sort();
    assert_eq!(semi_rows, vec![20, 25, 30]);
    let semi_metrics = semi
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(semi_metrics.join_type, "left_semi");
    assert_eq!(semi_metrics.execution_strategy, "bounded_reusable_filtered_join");
    assert_eq!(semi_metrics.build_side, "right");
    assert_eq!(semi_metrics.probe_side, "left");
    assert_eq!(semi_metrics.key_count, 1);
    assert!(!semi_metrics.residual_predicate);
    assert!(semi_metrics.hash_table_bytes > 0);
    assert!(semi_metrics.filtered_probe_state_bytes > 0);
    assert!(semi_metrics.gather_map_bytes > 0);
    assert_eq!(semi_metrics.output_rows, 3);
    assert_eq!(semi.metrics().hash_join_build_count, 0);
    assert_eq!(semi.metrics().hash_join_hash_table_bytes, 0);
    assert_eq!(semi.metrics().filtered_join_build_count, 1);
    assert_eq!(semi.metrics().filtered_join_hash_table_bytes, semi_metrics.hash_table_bytes);
    assert_eq!(semi.metrics().filtered_join_probe_state_bytes, semi_metrics.filtered_probe_state_bytes);
    assert_granted_reservation_trace(
        semi.metrics(),
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::FilteredProbeState, "join_filtered_probe_state"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );

    let QueryOutput::HostArrow(anti_batch) = anti.output() else {
        panic!("left anti bounded filtered join should return host output");
    };
    let anti_rows = int32_array(anti_batch, 0).values().iter().copied().collect::<Vec<_>>();
    assert_eq!(anti_rows, vec![10]);
    let anti_metrics = anti
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(anti_metrics.join_type, "left_anti");
    assert_eq!(anti_metrics.execution_strategy, "bounded_reusable_filtered_join");
    assert_eq!(anti_metrics.build_side, "right");
    assert_eq!(anti_metrics.probe_side, "left");
    assert_eq!(anti_metrics.key_count, 1);
    assert!(anti_metrics.hash_table_bytes > 0);
    assert!(anti_metrics.filtered_probe_state_bytes > 0);
    assert!(anti_metrics.gather_map_bytes > 0);
    assert_eq!(anti_metrics.output_rows, 1);
    assert_eq!(anti.metrics().hash_join_build_count, 0);
    assert_eq!(anti.metrics().hash_join_hash_table_bytes, 0);
    assert_eq!(anti.metrics().filtered_join_build_count, 1);
    assert_eq!(anti.metrics().filtered_join_hash_table_bytes, anti_metrics.hash_table_bytes);
    assert_eq!(anti.metrics().filtered_join_probe_state_bytes, anti_metrics.filtered_probe_state_bytes);
    assert_granted_reservation_trace(
        anti.metrics(),
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::FilteredProbeState, "join_filtered_probe_state"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_empty_root_filtered_left_semi_join_with_zero_gather_map() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-filtered-semi-empty");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_keys_parquet(&runtime, &right_path, &[8, 9]);

    let result = execute_bounded_equality_left_only_filtered_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftSemi,
    );

    assert_empty_bounded_filtered_join_result(&result, "left_semi");
    assert_zero_byte_gather_map_reservation(result.metrics());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_empty_root_filtered_left_anti_join_with_zero_gather_map() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-filtered-anti-empty");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_keys_parquet(&runtime, &right_path, &[1, 2, 3]);

    let result = execute_bounded_equality_left_only_filtered_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftAnti,
    );

    assert_empty_bounded_filtered_join_result(&result, "left_anti");
    assert_zero_byte_gather_map_reservation(result.metrics());
    fs::remove_dir_all(dir).unwrap();
}
