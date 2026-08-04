use super::*;

pub(crate) fn write_row_grouped_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 3, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 10, 20, 20, 30, 5])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400, 500, 600])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

/// Stress fixture for a four-AVG projection at high group cardinality. Each
/// group's expected average is exact so any GPU groupby row-alignment bug
/// produces a clear divergence.
pub(crate) fn write_grouped_decimal_avg_high_cardinality_parquet(path: &Path, groups: usize) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("price_a", DataType::Decimal128(7, 2), false),
        Field::new("price_b", DataType::Decimal128(7, 2), false),
        Field::new("price_c", DataType::Decimal128(7, 2), false),
    ]));
    let mut keys = Vec::with_capacity(groups * 2);
    let mut quantities = Vec::with_capacity(groups * 2);
    let mut price_a = Vec::with_capacity(groups * 2);
    let mut price_b = Vec::with_capacity(groups * 2);
    let mut price_c = Vec::with_capacity(groups * 2);
    for g in 0..groups {
        let key = format!("KEY{g:08}");
        keys.push(key.clone());
        keys.push(key);
        let base = g as i64;
        quantities.push(base * 2);
        quantities.push(base * 2 + 4);
        let base_p = base as i128 * 100;
        price_a.push(base_p);
        price_a.push(base_p + 200);
        price_b.push(base_p + 1000);
        price_b.push(base_p + 3000);
        price_c.push(base_p + 50);
        price_c.push(base_p + 250);
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(Int64Array::from(quantities)),
            Arc::new(Decimal128Array::from_iter_values(price_a).with_precision_and_scale(7, 2).unwrap()),
            Arc::new(Decimal128Array::from_iter_values(price_b).with_precision_and_scale(7, 2).unwrap()),
            Arc::new(Decimal128Array::from_iter_values(price_c).with_precision_and_scale(7, 2).unwrap()),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(256)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

/// Many-group fixture for AVG over a `BIGINT` column and a `Decimal128(7, 2)`
/// column grouped by an integer key. Each group's expected averages are exact
/// divisors so any per-group row alignment or GPU groupby ordering bug produces
/// a clear divergence.
pub(crate) fn write_grouped_decimal_avg_many_groups_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Int32, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("price_a", DataType::Decimal128(7, 2), false),
    ]));
    let groups = 60usize;
    let mut keys = Vec::with_capacity(groups * 2);
    let mut quantities = Vec::with_capacity(groups * 2);
    let mut prices = Vec::with_capacity(groups * 2);
    for g in 0..groups {
        let key = g as i32;
        let base_quantity = (g as i64) * 2 + 1;
        keys.push(key);
        keys.push(key);
        quantities.push(base_quantity);
        quantities.push(base_quantity + 1);
        let base_price = (g as i128) * 100 + 50;
        prices.push(base_price);
        prices.push(base_price + 100);
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int64Array::from(quantities)),
            Arc::new(Decimal128Array::from_iter_values(prices).with_precision_and_scale(7, 2).unwrap()),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(16)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

/// Fixture for AVG over a `BIGINT` column and three `Decimal128(7, 2)` columns
/// grouped by a string key. The unique group `G1` exercises both the integer
/// and decimal AVG output paths in a single aggregate node.
pub(crate) fn write_grouped_mixed_avg_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("price_a", DataType::Decimal128(7, 2), false),
        Field::new("price_b", DataType::Decimal128(7, 2), false),
        Field::new("price_c", DataType::Decimal128(7, 2), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["G0", "G1", "G1", "G2", "G2"])),
            Arc::new(Int64Array::from(vec![7, 8, 12, 5, 5])),
            Arc::new(
                Decimal128Array::from_iter_values([7_503, 1_730, 1_735, 6_930, 6_930])
                    .with_precision_and_scale(7, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from_iter_values([1_000, 1_500, 2_500, 3_000, 3_000])
                    .with_precision_and_scale(7, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from_iter_values([200, 300, 700, 100, 100]).with_precision_and_scale(7, 2).unwrap(),
            ),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

/// Fractional decimal AVG fixture: a `Decimal128(7, 2)` input grouped by a
/// string key, with one group (`G1`) whose `SUM = 34.65` and `COUNT = 2` so
/// that the true average is `17.325` exactly. Aggregated output schema must
/// declare `Decimal128(11, 6)` so the AVG output keeps scale-6 precision; an
/// output truncated to input scale 2 would round `17.325` down to `17.32` and
/// lose `0.005000` at scale 6.
pub(crate) fn write_grouped_fractional_decimal_avg_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, false),
        Field::new("amount", DataType::Decimal128(7, 2), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["G0", "G1", "G1", "G2", "G2"])),
            Arc::new(
                Decimal128Array::from_iter_values([7_503, 1_730, 1_735, 6_930, 6_930])
                    .with_precision_and_scale(7, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_final_merge_retry_input_parquet(path: &Path, keys_per_row_group: i32) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let keys = (0..3).flat_map(|_| 0..keys_per_row_group).collect::<Vec<_>>();
    let values = vec![1_i64; keys.len()];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys)), Arc::new(Int64Array::from(values))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(keys_per_row_group as usize)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}
