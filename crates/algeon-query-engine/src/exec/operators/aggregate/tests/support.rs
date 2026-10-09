use super::*;

pub(super) fn multi_key_sum_spec() -> (Schema, AggregateSpec) {
    let schema = Schema::new(vec![
        Field::new("county", DataType::Utf8View, true),
        Field::new("quarter", DataType::Int32, false),
        Field::new("year", DataType::Int32, false),
        Field::new("sales", DataType::Decimal128(17, 2), true),
    ]);
    let spec = AggregateSpec::new(
        vec![Expr::column("county"), Expr::column("quarter"), Expr::column("year")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    (schema, spec)
}

pub(super) fn fixed_width_sum_spec() -> (Schema, AggregateSpec) {
    let schema = Schema::new(vec![
        Field::new("quarter", DataType::Int32, false),
        Field::new("year", DataType::Int32, false),
        Field::new("sales", DataType::Int64, true),
    ]);
    let spec = AggregateSpec::new(
        vec![Expr::column("quarter"), Expr::column("year")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    (schema, spec)
}

pub(super) fn simple_string_sum_spec() -> (Schema, AggregateSpec) {
    let schema =
        Schema::new(vec![Field::new("county", DataType::Utf8View, true), Field::new("sales", DataType::Int64, true)]);
    let spec = AggregateSpec::new(
        vec![Expr::column("county")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    (schema, spec)
}

pub(super) fn simple_string_avg_spec() -> (Schema, AggregateSpec) {
    let schema =
        Schema::new(vec![Field::new("category", DataType::Utf8, true), Field::new("price", DataType::Float64, true)]);
    let spec = AggregateSpec::new(
        vec![Expr::column("category")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("price"), "avg_price")],
    );
    (schema, spec)
}

pub(super) fn filtered_global_q09_like_spec(filter_count: usize) -> (Arc<Schema>, AggregateSpec) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("quantity", DataType::Int64, true),
        Field::new("discount", DataType::Decimal128(7, 2), true),
        Field::new("net_paid", DataType::Decimal128(7, 2), true),
        Field::new("bucket", DataType::Boolean, true),
    ]));
    let mut aggr_exprs = Vec::new();
    for index in 0..filter_count {
        let value = if index % 3 == 0 {
            Expr::column("quantity")
        } else if index % 3 == 1 {
            Expr::column("discount")
        } else {
            Expr::column("net_paid")
        };
        let function = if index % 3 == 0 {
            AggregateFunction::Count
        } else if index % 3 == 1 {
            AggregateFunction::Sum
        } else {
            AggregateFunction::Avg
        };
        aggr_exprs
            .push(AggregateExpr::new(function, value, format!("a{index}")).with_filter(Some(Expr::column("bucket"))));
    }
    (schema, AggregateSpec::new(Vec::new(), aggr_exprs))
}

pub(super) fn filtered_global_tpcds_q09_sf100_spec() -> (Arc<Schema>, AggregateSpec) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("ss_ext_discount_amt", DataType::Decimal128(7, 2), true),
        Field::new("ss_net_paid", DataType::Decimal128(7, 2), true),
    ]));
    let mut aggr_exprs = Vec::new();
    for (bucket, lower) in [1, 21, 41, 61, 81].into_iter().enumerate() {
        let upper = lower + 19;
        let filter = quantity_between(lower, upper);
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Count,
                Expr::literal(LiteralValue::Int64(1)),
                format!("b{bucket}_cnt"),
            )
            .with_filter(Some(filter.clone())),
        );
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Avg,
                Expr::column("ss_ext_discount_amt"),
                format!("b{bucket}_avg_discount"),
            )
            .with_filter(Some(filter.clone())),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("ss_net_paid"), format!("b{bucket}_avg_net_paid"))
                .with_filter(Some(filter)),
        );
    }
    (schema, AggregateSpec::new(Vec::new(), aggr_exprs))
}

fn quantity_between(lower: i32, upper: i32) -> Expr {
    Expr::binary(
        Expr::binary(Expr::column("ss_quantity"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(lower))),
        BinaryOp::And,
        Expr::binary(Expr::column("ss_quantity"), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(upper))),
    )
}

pub(super) fn filtered_grouped_q09_like_spec(filter_count: usize) -> (Arc<Schema>, AggregateSpec) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("quantity", DataType::Int64, true),
        Field::new("discount", DataType::Decimal128(7, 2), true),
        Field::new("net_paid", DataType::Decimal128(7, 2), true),
        Field::new("bucket", DataType::Boolean, true),
    ]));
    let mut aggr_exprs = Vec::new();
    for index in 0..filter_count {
        let value = if index % 3 == 0 {
            Expr::literal(LiteralValue::Int64(1))
        } else if index % 3 == 1 {
            Expr::column("discount")
        } else {
            Expr::column("net_paid")
        };
        let function = if index % 3 == 0 {
            AggregateFunction::Count
        } else if index % 3 == 1 {
            AggregateFunction::Sum
        } else {
            AggregateFunction::Avg
        };
        aggr_exprs
            .push(AggregateExpr::new(function, value, format!("a{index}")).with_filter(Some(Expr::column("bucket"))));
    }
    (schema, AggregateSpec::new(vec![Expr::column("key")], aggr_exprs))
}

pub(super) fn partial_strategy_for_multi_key_sum(input_rows: u64) -> AggregateStrategyCapabilityDecision {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        input_rows,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    )
}

pub(super) fn metrics_with_prior_partial_decision(
    node_id: NodeId,
    spec: &AggregateSpec,
    schema: &Schema,
    input_evidence: Option<PartialAggregateInputEvidence>,
    input_rows: u64,
    output_group_count: u64,
    selected_strategy: &'static str,
) -> ExecutionMetrics {
    let metrics = ExecutionMetrics::detached("aggregate-test");
    let facts = BTreeMap::from([
        ("execution_stage", "partial".to_string()),
        ("aggregate_structural_key", aggregate_structural_key(spec, schema, input_evidence)),
        ("input_rows", input_rows.to_string()),
        ("output_group_count", output_group_count.to_string()),
        ("reduction_ratio_ppm", reduction_ratio_ppm(input_rows, output_group_count).to_string()),
        ("selected_strategy", selected_strategy.to_string()),
    ]);
    metrics.record_aggregate_strategy_decision(node_id, &facts);
    metrics
}

pub(super) fn metrics_with_prior_partial_structural_decision(
    node_id: NodeId,
    spec: &AggregateSpec,
    schema: &Schema,
    input_rows: u64,
    output_group_count: u64,
    selected_strategy: &'static str,
) -> ExecutionMetrics {
    let metrics = ExecutionMetrics::detached("aggregate-test");
    let facts = BTreeMap::from([
        ("execution_stage", "partial".to_string()),
        ("aggregate_structural_key", aggregate_structural_key(spec, schema, None)),
        ("input_rows", input_rows.to_string()),
        ("output_group_count", output_group_count.to_string()),
        ("reduction_ratio_ppm", reduction_ratio_ppm(input_rows, output_group_count).to_string()),
        ("selected_strategy", selected_strategy.to_string()),
    ]);
    metrics.record_aggregate_strategy_decision(node_id, &facts);
    metrics
}

pub(super) fn source_adjacent_group_key_evidence(
    row_group_count: u64,
    max_partition_fan_in: u64,
) -> PartialAggregateInputEvidence {
    source_adjacent_evidence("group_key", row_group_count, Some(max_partition_fan_in))
}

pub(super) fn source_adjacent_unpartitioned_evidence(row_group_count: u64) -> PartialAggregateInputEvidence {
    source_adjacent_evidence("none", row_group_count, None)
}

fn source_adjacent_evidence(
    partition_strategy: &'static str,
    row_group_count: u64,
    max_partition_fan_in: Option<u64>,
) -> PartialAggregateInputEvidence {
    PartialAggregateInputEvidence {
        source_chunk_fingerprint: Some(0xabc),
        source_adjacent_partial: true,
        source_chunk_row_group_count: row_group_count,
        source_chunk_reads_all_row_groups: false,
        source_chunk_known_row_count: Some(row_group_count.saturating_mul(1_000_000)),
        source_chunk_estimated_source_read_bytes: Some(row_group_count.saturating_mul(16 * 1024 * 1024)),
        source_chunk_size_estimate_based: false,
        source_adjacent_partition_strategy: partition_strategy,
        source_adjacent_partition_count: (partition_strategy != "none").then_some(32),
        source_adjacent_non_empty_partition_count: (partition_strategy != "none").then_some(4),
        source_adjacent_partition_key_count: (partition_strategy == "group_key").then_some(2),
        source_adjacent_max_partition_fan_in: max_partition_fan_in,
    }
}
