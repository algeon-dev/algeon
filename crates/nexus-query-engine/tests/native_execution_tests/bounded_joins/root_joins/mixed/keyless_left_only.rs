use super::*;
#[test]
fn bounded_native_query_executes_root_keyless_conditional_left_only_joins_with_left_build_orientation() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-conditional-left-only-joins");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);

    let semi = execute_bounded_keyless_left_only_conditional_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftSemi,
        JoinSide::Left,
    );
    let anti = execute_bounded_keyless_left_only_conditional_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftAnti,
        JoinSide::Left,
    );

    let QueryOutput::HostArrow(semi_batch) = semi.output() else {
        panic!("left semi bounded conditional join should return host output");
    };
    let mut semi_rows = int32_array(&semi_batch, 0).values().iter().copied().collect::<Vec<_>>();
    semi_rows.sort();
    assert_eq!(semi_rows, vec![25, 30]);
    let semi_metrics = semi
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(semi_metrics.join_type, "left_semi");
    assert_eq!(semi_metrics.execution_strategy, "bounded_keyless_conditional_join");
    assert_eq!(semi_metrics.build_side, "left");
    assert_eq!(semi_metrics.key_count, 0);
    assert_eq!(semi_metrics.output_rows, 2);
    assert!(semi_metrics.gather_map_bytes > 0);
    assert_reservation_kind_subsequence(
        semi.metrics(),
        &[
            ReservationKind::BuildSide,
            ReservationKind::ProbeSide,
            ReservationKind::GatherMap,
            ReservationKind::JoinOutput,
            ReservationKind::Materialization,
        ],
    );

    let QueryOutput::HostArrow(anti_batch) = anti.output() else {
        panic!("left anti bounded conditional join should return host output");
    };
    let mut anti_rows = int32_array(&anti_batch, 0).values().iter().copied().collect::<Vec<_>>();
    anti_rows.sort();
    assert_eq!(anti_rows, vec![10, 20]);
    let anti_metrics = anti
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(anti_metrics.join_type, "left_anti");
    assert_eq!(anti_metrics.execution_strategy, "bounded_keyless_conditional_join");
    assert_eq!(anti_metrics.build_side, "left");
    assert_eq!(anti_metrics.key_count, 0);
    assert_eq!(anti_metrics.output_rows, 2);
    assert!(anti_metrics.gather_map_bytes > 0);
    assert_reservation_kind_subsequence(
        anti.metrics(),
        &[
            ReservationKind::BuildSide,
            ReservationKind::ProbeSide,
            ReservationKind::GatherMap,
            ReservationKind::JoinOutput,
            ReservationKind::Materialization,
        ],
    );

    fs::remove_dir_all(dir).unwrap();
}
