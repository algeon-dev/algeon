use super::*;

#[test]
fn unbounded_native_query_executes_grouped_decimal_avg_at_output_scale() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    assert_grouped_decimal_avg_keeps_output_scale(&context);
}

#[test]
fn unbounded_native_query_executes_grouped_mixed_integer_and_decimal_avg() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    assert_grouped_mixed_integer_and_decimal_avg(&context);
}

/// Many-group regression for the partial-state grouped AVG path. Each of
/// the 60 groups has exactly two rows; the integer AVG should match each
/// group's `(base_quantity + base_quantity + 1) / 2` and the decimal AVG
/// should match `((base_price + (base_price + 100)) / 2)` at the declared
/// `Decimal128(11, 6)` output scale. This guards against any per-group row
/// alignment bug that does not show up with a tiny three-group fixture.
fn assert_grouped_many_group_decimal_avg(context: &ExecutionContext) {
    let dir = unique_temp_dir("nexus-query-engine-native-grouped-many-groups-avg");
    let path = dir.join("input.parquet");
    let input_schema = write_grouped_decimal_avg_many_groups_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Int32, false),
        Field::new("avg_quantity", DataType::Float64, true),
        Field::new("avg_price_a", DataType::Decimal128(11, 6), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["group_key".to_string(), "quantity".to_string(), "price_a".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("group_key")],
        vec![
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity"), "avg_quantity"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("price_a"), "avg_price_a"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("group_key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 60);
    let keys = int32_array(batch, 0).values().as_ref().to_vec();
    let quantities = float64_array(batch, 1).values().as_ref().to_vec();
    let prices = decimal128_array(batch, 2);
    for (row, key) in keys.iter().enumerate() {
        let g = *key as i64;
        let expected_quantity_avg = ((g * 2 + 1) as f64 + (g * 2 + 2) as f64) / 2.0;
        assert_eq!(quantities[row], expected_quantity_avg, "row {row} (group_key={key}): integer AVG mismatch");
        let base_price = g as i128 * 100 + 50;
        let expected_price_avg_at_scale_6 = ((base_price + (base_price + 100)) / 2) * 10_000;
        assert_eq!(
            prices.value(row),
            expected_price_avg_at_scale_6,
            "row {row} (group_key={key}): decimal AVG mismatch"
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

/// Stress regression for a high-cardinality grouped AVG: four AVGs over one
/// BIGINT and three `Decimal128(7, 2)` columns, sorted by group key, with every
/// row spot-checked. This exercises the per-group partial-state path from the
/// compiled bounded pipeline.
fn assert_grouped_high_cardinality_mixed_avg(context: &ExecutionContext, groups: usize) {
    let dir = unique_temp_dir("nexus-query-engine-native-grouped-high-cardinality-avg");
    let path = dir.join("input.parquet");
    let input_schema = write_grouped_decimal_avg_high_cardinality_parquet(&path, groups);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, false),
        Field::new("avg_quantity", DataType::Float64, true),
        Field::new("avg_price_a", DataType::Decimal128(11, 6), true),
        Field::new("avg_price_b", DataType::Decimal128(11, 6), true),
        Field::new("avg_price_c", DataType::Decimal128(11, 6), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec![
            "group_key".to_string(),
            "quantity".to_string(),
            "price_a".to_string(),
            "price_b".to_string(),
            "price_c".to_string(),
        ],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("group_key")],
        vec![
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity"), "avg_quantity"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("price_a"), "avg_price_a"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("price_b"), "avg_price_b"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("price_c"), "avg_price_c"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("group_key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(context, &plan).unwrap();
    let metrics = result.metrics();
    if metrics.memory_control_policy.as_str() == "bounded" {
        let partial_strategy_rows = metrics
            .aggregate_strategy_decisions
            .iter()
            .filter(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("partial"))
            .collect::<Vec<_>>();
        assert!(
            partial_strategy_rows.len() > 1,
            "bounded high-cardinality fixture should produce multiple partial aggregate strategy rows"
        );
        assert!(partial_strategy_rows.iter().all(|decision| {
            decision.node_id == aggregate.to_string()
                && decision.facts.get("selected_strategy").map(String::as_str) == Some("libcudf_groupby")
                && decision.facts.get("source_adjacent_partial").map(String::as_str) == Some("false")
                && decision.facts.get("source_chunk_row_group_count").map(String::as_str) == Some("1")
                && decision.facts.get("encode_elapsed_ns").map(String::as_str) == Some("0")
                && decision.facts.get("decode_elapsed_ns").map(String::as_str) == Some("0")
        }));
        assert_eq!(
            partial_strategy_rows[0].facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
            Some("encoded_key_partial_cost_model_small_input")
        );
        assert!(partial_strategy_rows[1..].iter().all(|decision| {
            matches!(
                decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
                Some("encoded_key_partial_cost_model_small_input")
                    | Some("encoded_key_partial_cost_model_high_cardinality")
            )
        }));
    }
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), groups);
    let keys: Vec<String> = string_array(batch, 0).iter().map(|v| v.unwrap().to_string()).collect();
    let quantities = float64_array(batch, 1).values().as_ref().to_vec();
    let price_a = decimal128_array(batch, 2);
    let price_b = decimal128_array(batch, 3);
    let price_c = decimal128_array(batch, 4);
    for (row, key) in keys.iter().enumerate() {
        // Keys were written as "KEY00000000".."KEY00001499". Sort by string is
        // numerically equivalent here, so row N has KEY = format("KEY{:08}", N).
        let g: usize = key.strip_prefix("KEY").unwrap().parse().unwrap();
        let g_i = g as i64;
        let g_p = g as i128;
        let expected_quantity = ((g_i * 2) as f64 + (g_i * 2 + 4) as f64) / 2.0;
        let expected_pa = ((g_p * 100 + g_p * 100 + 200) / 2) * 10_000;
        let expected_pb = ((g_p * 100 + 1000 + g_p * 100 + 3000) / 2) * 10_000;
        let expected_pc = ((g_p * 100 + 50 + g_p * 100 + 250) / 2) * 10_000;
        assert_eq!(quantities[row], expected_quantity, "row {row} key={key}: avg_quantity mismatch");
        assert_eq!(price_a.value(row), expected_pa, "row {row} key={key}: avg_price_a mismatch");
        assert_eq!(price_b.value(row), expected_pb, "row {row} key={key}: avg_price_b mismatch");
        assert_eq!(price_c.value(row), expected_pc, "row {row} key={key}: avg_price_c mismatch");
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unbounded_native_query_executes_grouped_high_cardinality_mixed_avg() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    assert_grouped_high_cardinality_mixed_avg(&context, 1500);
}

#[test]
fn unbounded_native_query_executes_grouped_many_group_decimal_avg() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    assert_grouped_many_group_decimal_avg(&context);
}

fn assert_grouped_decimal_avg_keeps_output_scale(context: &ExecutionContext) {
    let dir = unique_temp_dir("nexus-query-engine-native-grouped-decimal-avg-pattern");
    let path = dir.join("input.parquet");
    let input_schema = write_grouped_fractional_decimal_avg_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, false),
        Field::new("avg_amount", DataType::Decimal128(11, 6), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["group_key".to_string(), "amount".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("group_key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "avg_amount")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("group_key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema(), aggregated_schema);
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(string_array(batch, 0).iter().map(|value| value.unwrap()).collect::<Vec<_>>(), vec!["G0", "G1", "G2"]);
    let avg_amount = decimal128_array(batch, 1);
    assert_eq!(avg_amount.precision(), 11);
    assert_eq!(avg_amount.scale(), 6);
    assert_eq!(avg_amount.value(0), 75_030_000, "G0 has one row amount=75.03 -> AVG=75.030000 at scale 6");
    assert_eq!(
        avg_amount.value(1),
        17_325_000,
        "G1 (sum=34.65, count=2) must keep scale-6 precision (17.325000), not truncate to 17.320000"
    );
    assert_eq!(avg_amount.value(2), 69_300_000, "G2 has two rows of 69.30 -> AVG=69.300000 at scale 6");

    fs::remove_dir_all(dir).unwrap();
}

fn assert_grouped_mixed_integer_and_decimal_avg(context: &ExecutionContext) {
    let dir = unique_temp_dir("nexus-query-engine-native-grouped-mixed-avg");
    let path = dir.join("input.parquet");
    let input_schema = write_grouped_mixed_avg_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, false),
        Field::new("avg_quantity", DataType::Float64, true),
        Field::new("avg_price_a", DataType::Decimal128(11, 6), true),
        Field::new("avg_price_b", DataType::Decimal128(11, 6), true),
        Field::new("avg_price_c", DataType::Decimal128(11, 6), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec![
            "group_key".to_string(),
            "quantity".to_string(),
            "price_a".to_string(),
            "price_b".to_string(),
            "price_c".to_string(),
        ],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("group_key")],
        vec![
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity"), "avg_quantity"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("price_a"), "avg_price_a"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("price_b"), "avg_price_b"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("price_c"), "avg_price_c"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("group_key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema(), aggregated_schema);
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(string_array(batch, 0).iter().map(|value| value.unwrap()).collect::<Vec<_>>(), vec!["G0", "G1", "G2"]);
    let quantity = float64_array(batch, 1);
    assert_eq!(quantity.values().as_ref(), &[7.0, 10.0, 5.0]);
    let price_a = decimal128_array(batch, 2);
    assert_eq!(price_a.scale(), 6);
    assert_eq!(price_a.value(0), 75_030_000, "G0 price_a = 75.030000");
    assert_eq!(price_a.value(1), 17_325_000, "G1 price_a = 17.325000");
    assert_eq!(price_a.value(2), 69_300_000, "G2 price_a = 69.300000");
    let price_b = decimal128_array(batch, 3);
    assert_eq!(price_b.value(0), 10_000_000, "G0 price_b = 10.000000");
    assert_eq!(price_b.value(1), 20_000_000, "G1 price_b = 20.000000");
    assert_eq!(price_b.value(2), 30_000_000, "G2 price_b = 30.000000");
    let price_c = decimal128_array(batch, 4);
    assert_eq!(price_c.value(0), 2_000_000, "G0 price_c = 2.000000");
    assert_eq!(price_c.value(1), 5_000_000, "G1 price_c = 5.000000");
    assert_eq!(price_c.value(2), 1_000_000, "G2 price_c = 1.000000");

    fs::remove_dir_all(dir).unwrap();
}
