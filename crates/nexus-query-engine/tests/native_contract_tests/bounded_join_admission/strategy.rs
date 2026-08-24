#[test]
fn large_keyed_inner_build_selects_partitioned_strategy() {
    let plan = large_keyed_join_strategy_plan(JoinType::Inner);
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections());
    let facts = modeled_facts(&plan).expect("large partitioned join should still be admitted");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:partitioned_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason").map(String::as_str), Some("n2:build_side_too_large"));
    assert_eq!(facts.get("estimated_build_rows").map(String::as_str), Some("n2:100000000"));
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:9663676416"));
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:1024"));
    assert_eq!(facts.get("probe_reuse_count").map(String::as_str), Some("n2:1"));

    nexus_query_engine::pipeline::compile_pipeline(&plan, &analysis).expect("partitioned join should compile");
}

#[test]
fn plan_captured_keyed_hash_join_strategy_options_drive_admission_facts() {
    let strategy_options = KeyedHashJoinStrategyOptions::default()
        .with_large_keyed_hash_build_row_threshold(200_000_000)
        .with_large_keyed_hash_build_bytes_threshold(16 * 1024 * 1024 * 1024)
        .with_small_reusable_hash_build_row_threshold(200_000_000)
        .with_small_reusable_hash_build_bytes_threshold(16 * 1024 * 1024 * 1024);
    let optimized = nexus_query_engine::optimizer::optimize_query_plan(
        large_keyed_join_strategy_plan(JoinType::Inner),
        &NativeOptimizerOptions { keyed_hash_join_strategy: strategy_options, ..NativeOptimizerOptions::disabled() },
    );
    let plan = optimized.plan;

    let facts = modeled_facts(&plan).expect("custom strategy thresholds should keep the join admitted");

    assert_eq!(plan.keyed_hash_join_strategy_options(), strategy_options);
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
}

#[test]
fn large_keyed_left_and_full_builds_select_partitioned_strategy() {
    for join_type in [JoinType::Left, JoinType::Full] {
        let plan = large_keyed_join_strategy_plan(join_type);
        let analysis = plan.capability_analysis();

        assert!(!analysis.has_rejections(), "{join_type} large partitioned join should be admitted");
        let facts = modeled_facts(&plan).expect("large partitioned join should report modeled bounded facts");
        assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:partitioned_hash_equi"));
        assert_eq!(
            facts.get("reusable_hash_join_rejected_reason").map(String::as_str),
            Some("n2:build_side_too_large")
        );
        assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:9663676416"));
        assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:1024"));
        assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);

        nexus_query_engine::pipeline::compile_pipeline(&plan, &analysis).expect("partitioned join should compile");
    }
}

#[test]
fn large_keyed_probe_with_small_build_selects_reusable_strategy() {
    let plan = large_keyed_probe_join_strategy_plan(10, 1024);
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "large probe reusable join with small build should be admitted");
    let facts = modeled_facts(&plan).expect("large probe reusable join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:1024"));
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:3221225472"));
}

#[test]
fn small_single_use_keyed_inner_join_stays_reusable_strategy() {
    let plan = small_keyed_join_strategy_plan(JoinType::Inner, None);
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "small single-use inner join should be admitted");
    let facts = modeled_facts(&plan).expect("small single-use inner join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("estimated_build_rows").map(String::as_str), Some("n2:10"));
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:1024"));
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:1048576"));
    assert_eq!(facts.get("probe_reuse_count").map(String::as_str), Some("n2:1"));
    assert_eq!(facts.get("reusable_prebuild_supported").map(String::as_str), Some("n2:true"));

    nexus_query_engine::pipeline::compile_pipeline(&plan, &analysis).expect("small reusable join should compile");
}

#[test]
fn small_keyed_inner_join_with_build_artifact_keeps_reusable_strategy() {
    let plan = small_keyed_join_strategy_plan(JoinType::Inner, Some("small-reusable-build"));
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "small join with build artifact should be admitted");
    let facts = modeled_facts(&plan).expect("small join with build artifact should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
}

#[test]
fn small_keyed_left_join_keeps_reusable_strategy() {
    let plan = small_keyed_join_strategy_plan(JoinType::Left, None);
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "small left join should be admitted");
    let facts = modeled_facts(&plan).expect("small left join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
}

#[test]
fn byte_small_but_row_non_small_keyed_inner_join_keeps_reusable_strategy() {
    let plan = byte_small_row_non_small_keyed_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "byte-small row-non-small inner join should be admitted");
    let facts = modeled_facts(&plan).expect("byte-small row-non-small inner join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("estimated_build_rows").map(String::as_str), Some("n2:2000000"));
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:16777216"));
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:1048576"));
}

#[test]
fn row_small_but_byte_non_small_keyed_inner_join_keeps_reusable_strategy() {
    let plan = row_small_byte_non_small_keyed_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "row-small byte-non-small inner join should be admitted");
    let facts = modeled_facts(&plan).expect("row-small byte-non-small inner join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("estimated_build_rows").map(String::as_str), Some("n2:10"));
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:134217728"));
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:1048576"));
}

#[test]
fn unknown_probe_with_small_left_build_selects_reusable_strategy() {
    let plan = unknown_probe_small_left_build_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "unknown probe reusable join with small build should be admitted");
    let facts = modeled_facts(&plan).expect("small-build reusable join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:1024"));
    assert_eq!(facts.get("probe_side_bytes"), None);
}

#[test]
fn small_flat_string_keyed_join_defers_null_payload_validation_to_exact_prebuild() {
    let plan = string_keyed_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "flat string keys should use cuDF's exact runtime pre-build contract");
    let facts = modeled_facts(&plan).expect("flat string-key hash join should report a modeled bounded contract");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
}

#[test]
fn large_probe_with_small_flat_string_build_keeps_reusable_strategy() {
    let plan = string_keyed_join_strategy_plan_with_probe(10_000_000, 3 * 1024 * 1024 * 1024, 10, 1024);
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "flat string keys should use cuDF's exact runtime pre-build contract");
    let facts = modeled_facts(&plan).expect("small flat string build should report a modeled bounded contract");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:reusable_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:3221225472"));
}

#[test]
fn large_keyed_probe_with_non_small_build_selects_partitioned_strategy() {
    let plan = large_keyed_probe_join_strategy_plan(2_000_000, 128 * 1024 * 1024);
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "large probe partitioned join should be admitted");
    let facts = modeled_facts(&plan).expect("large probe partitioned join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:partitioned_hash_equi"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason").map(String::as_str), Some("n2:probe_side_too_large"));
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:134217728"));
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:3221225472"));
}

#[test]
fn single_use_non_small_keyed_build_selects_partitioned_strategy() {
    let plan = medium_single_use_keyed_build_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "single-use non-small build should remain admitted");
    let facts = modeled_facts(&plan).expect("single-use non-small build should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:partitioned_hash_equi"));
    assert_eq!(
        facts.get("reusable_hash_join_rejected_reason").map(String::as_str),
        Some("n2:single_use_build_not_small")
    );
    assert_eq!(facts.get("estimated_build_rows").map(String::as_str), Some("n2:2000000"));
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:134217728"));
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:1048576"));

    nexus_query_engine::pipeline::compile_pipeline(&plan, &analysis)
        .expect("single-use non-small partitioned join should compile");
}

#[test]
fn large_keyed_inner_residual_build_selects_partitioned_mixed_strategy() {
    let plan = large_keyed_residual_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "large keyed inner residual partitioned join should be admitted");
    let facts = modeled_facts(&plan).expect("large partitioned residual join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:partitioned_mixed_keyed_residual"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason").map(String::as_str), Some("n2:build_side_too_large"));
    assert_eq!(facts.get("build_side_bytes").map(String::as_str), Some("n2:9663676416"));
    assert_eq!(facts.get("probe_side_bytes").map(String::as_str), Some("n2:1024"));
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("one_shot_equi_rejected_reason"), None);

    nexus_query_engine::pipeline::compile_pipeline(&plan, &analysis).expect("partitioned residual join should compile");
}

#[test]
fn keyed_inner_residual_or_selects_mixed_join_strategy() {
    let plan = large_keyed_residual_or_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.has_rejections(), "keyed residual OR join should be admitted through mixed join");
    let facts = modeled_facts(&plan).expect("keyed residual OR join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:partitioned_mixed_keyed_residual"));
    assert_eq!(facts.get("reusable_hash_join_rejected_reason").map(String::as_str), Some("n2:build_side_too_large"));
    assert_eq!(facts.get("partitioned_hash_join_rejected_reason"), None);
    assert_eq!(facts.get("one_shot_equi_rejected_reason"), None);
}

#[test]
fn keyed_decimal_division_residual_selects_one_shot_post_filter_strategy() {
    let plan = decimal_division_residual_join_strategy_plan();
    let analysis = plan.capability_analysis();

    assert!(
        !analysis.has_rejections(),
        "keyed decimal-division residual join should be admitted through post-join filtering"
    );
    let facts = modeled_facts(&plan).expect("keyed decimal-division residual join should report modeled bounded facts");
    assert_eq!(facts.get("join_strategy").map(String::as_str), Some("n2:one_shot_equi"));
    assert_eq!(
        facts.get("reusable_hash_join_rejected_reason").map(String::as_str),
        Some("n2:residual_requires_post_join_filter")
    );
    assert_eq!(
        facts.get("partitioned_hash_join_rejected_reason").map(String::as_str),
        Some("n2:residual_requires_post_join_filter")
    );
    assert_eq!(facts.get("one_shot_equi_rejected_reason"), None);
}

pub(crate) fn large_keyed_join_strategy_plan(join_type: JoinType) -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-join-left.parquet")
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![1024]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-join-right.parquet")
                .with_row_group_rows(vec![100_000_000])
                .with_row_group_bytes(vec![9 * 1024 * 1024 * 1024]),
        ],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            join_type,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn large_keyed_probe_join_strategy_plan(build_rows: u64, build_bytes: u64) -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-probe-join-left.parquet")
                .with_row_group_rows(vec![10_000_000])
                .with_row_group_bytes(vec![3 * 1024 * 1024 * 1024]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-probe-join-right.parquet")
                .with_row_group_rows(vec![build_rows])
                .with_row_group_bytes(vec![build_bytes]),
        ],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn medium_single_use_keyed_build_join_strategy_plan() -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/medium-single-use-join-left.parquet")
                .with_row_group_rows(vec![4096])
                .with_row_group_bytes(vec![1024 * 1024]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/medium-single-use-join-right.parquet")
                .with_row_group_rows(vec![2_000_000])
                .with_row_group_bytes(vec![128 * 1024 * 1024]),
        ],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn small_keyed_join_strategy_plan(
    join_type: JoinType,
    build_artifact_id: Option<&str>,
) -> nexus_query_engine::plan::QueryPlan {
    sized_keyed_join_strategy_plan(join_type, build_artifact_id, 10, 1024)
}

fn byte_small_row_non_small_keyed_join_strategy_plan() -> nexus_query_engine::plan::QueryPlan {
    sized_keyed_join_strategy_plan(JoinType::Inner, None, 2_000_000, 16 * 1024 * 1024)
}

fn row_small_byte_non_small_keyed_join_strategy_plan() -> nexus_query_engine::plan::QueryPlan {
    sized_keyed_join_strategy_plan(JoinType::Inner, None, 10, 128 * 1024 * 1024)
}

fn sized_keyed_join_strategy_plan(
    join_type: JoinType,
    build_artifact_id: Option<&str>,
    build_rows: u64,
    build_bytes: u64,
) -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/small-join-left.parquet")
                .with_row_group_rows(vec![4096])
                .with_row_group_bytes(vec![1024 * 1024]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/small-join-right.parquet")
                .with_row_group_rows(vec![build_rows])
                .with_row_group_bytes(vec![build_bytes]),
        ],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    )));
    let mut spec = JoinSpec::new(
        join_type,
        vec![JoinKey::new(
            ColumnRef::input(0, "order_id", "left_order_id"),
            ColumnRef::input(1, "order_id", "right_order_id"),
        )],
    )
    .with_output_projection(vec![
        JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
        JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
    ])
    .with_build_side(JoinSide::Right);
    if let Some(id) = build_artifact_id {
        spec = spec.with_build_artifact_id(id);
    }
    let join = builder.add_join(left, right, spec, output_schema);
    builder.finish_with_host_sink(join)
}

fn unknown_probe_small_left_build_join_strategy_plan() -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_source_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category_count", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category_count", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/unknown-probe-small-build-left.parquet")
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![1024]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("unknown_probe", right_source_schema)));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "category_count", "category_count"), "category_count"),
        ])
        .with_build_side(JoinSide::Left),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn string_keyed_join_strategy_plan() -> nexus_query_engine::plan::QueryPlan {
    string_keyed_join_strategy_plan_with_probe(10_000, 1024 * 1024, 10, 1024)
}

fn string_keyed_join_strategy_plan_with_probe(
    probe_rows: u64,
    probe_bytes: u64,
    build_rows: u64,
    build_bytes: u64,
) -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("customer_id", DataType::Utf8, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("customer_id", DataType::Utf8, true),
        Field::new("segment", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("customer_id", DataType::Utf8, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("segment", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/string-key-join-left.parquet")
                .with_row_group_rows(vec![probe_rows])
                .with_row_group_bytes(vec![probe_bytes]),
        ],
        Arc::clone(&left_schema),
        vec!["customer_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/string-key-join-right.parquet")
                .with_row_group_rows(vec![build_rows])
                .with_row_group_bytes(vec![build_bytes]),
        ],
        Arc::clone(&right_schema),
        vec!["customer_id".to_string(), "segment".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "customer_id", "left_customer_id"),
                ColumnRef::input(1, "customer_id", "right_customer_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer_id", "customer_id"), "customer_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "segment", "segment"), "segment"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn large_keyed_residual_join_strategy_plan() -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-residual-join-left.parquet")
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![1024]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-residual-join-right.parquet")
                .with_row_group_rows(vec![100_000_000])
                .with_row_group_bytes(vec![9 * 1024 * 1024 * 1024]),
        ],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "qty", "qty"),
            BinaryOp::Gt,
            Expr::input_column(1, "order_id", "right_order_id"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn large_keyed_residual_or_join_strategy_plan() -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, true),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, true),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-residual-or-join-left.parquet")
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![1024]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-residual-or-join-right.parquet")
                .with_row_group_rows(vec![100_000_000])
                .with_row_group_bytes(vec![9 * 1024 * 1024 * 1024]),
        ],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_residual_predicate(Expr::binary(
            Expr::binary(Expr::input_column(0, "qty", "qty"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(5))),
            BinaryOp::Or,
            Expr::binary(
                Expr::input_column(1, "category", "category"),
                BinaryOp::Eq,
                Expr::literal(LiteralValue::Utf8("promo".to_string())),
            ),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn decimal_division_residual_join_strategy_plan() -> nexus_query_engine::plan::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("sales_cnt_decimal", DataType::Decimal128(17, 2), false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("sales_cnt_decimal", DataType::Decimal128(17, 2), false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("sales_cnt_decimal", DataType::Decimal128(17, 2), false),
        Field::new("right_sales_cnt_decimal", DataType::Decimal128(17, 2), false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/decimal-residual-left.parquet")
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![1024]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "sales_cnt_decimal".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/decimal-residual-right.parquet")
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![1024]),
        ],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "sales_cnt_decimal".to_string()],
    )));
    let residual_ratio = Expr::binary_with_output_type(
        Expr::input_column(0, "sales_cnt_decimal", "sales_cnt_decimal"),
        BinaryOp::Divide,
        Expr::input_column(1, "sales_cnt_decimal", "right_sales_cnt_decimal"),
        DataType::Decimal128(23, 6),
    );
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_residual_predicate(Expr::binary(
            residual_ratio,
            BinaryOp::Lt,
            Expr::literal(LiteralValue::Decimal128 { raw: 900000000000, precision: 23, scale: 6 }),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales_cnt_decimal", "sales_cnt_decimal"), "sales_cnt_decimal"),
            JoinOutputColumn::right(
                ColumnRef::input(1, "sales_cnt_decimal", "right_sales_cnt_decimal"),
                "right_sales_cnt_decimal",
            ),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}
