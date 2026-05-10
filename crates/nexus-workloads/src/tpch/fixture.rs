use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, Date32Array, Decimal128Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use super::WorkloadResult;

pub fn write_tiny_tpch_parquet_dataset(root: &Path) -> WorkloadResult<()> {
    fs::create_dir_all(root)?;
    write_customer(root)?;
    write_lineitem(root)?;
    write_nation(root)?;
    write_orders(root)?;
    write_part(root)?;
    write_partsupp(root)?;
    write_region(root)?;
    write_supplier(root)?;
    Ok(())
}

fn write_customer(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "customer",
        schema(vec![
            field("c_custkey", DataType::Int64),
            field("c_name", DataType::Utf8),
            field("c_address", DataType::Utf8),
            field("c_nationkey", DataType::Int64),
            field("c_phone", DataType::Utf8),
            field("c_acctbal", money_type()),
            field("c_mktsegment", DataType::Utf8),
            field("c_comment", DataType::Utf8),
        ]),
        vec![
            i64s([1, 2, 3]),
            strings(["Customer#000000001", "Customer#000000002", "Customer#000000003"]),
            strings(["Address 1", "Address 2", "Address 3"]),
            i64s([1, 2, 3]),
            strings(["13-111-111-1111", "31-222-222-2222", "23-333-333-3333"]),
            money([71156, 12165, -300]),
            strings(["BUILDING", "AUTOMOBILE", "BUILDING"]),
            strings(["regular customer", "priority customer", "careful customer"]),
        ],
    )
}

fn write_lineitem(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "lineitem",
        schema(vec![
            field("l_orderkey", DataType::Int64),
            field("l_partkey", DataType::Int64),
            field("l_suppkey", DataType::Int64),
            field("l_linenumber", DataType::Int32),
            field("l_quantity", money_type()),
            field("l_extendedprice", money_type()),
            field("l_discount", money_type()),
            field("l_tax", money_type()),
            field("l_returnflag", DataType::Utf8),
            field("l_linestatus", DataType::Utf8),
            field("l_shipdate", DataType::Date32),
            field("l_commitdate", DataType::Date32),
            field("l_receiptdate", DataType::Date32),
            field("l_shipinstruct", DataType::Utf8),
            field("l_shipmode", DataType::Utf8),
            field("l_comment", DataType::Utf8),
        ]),
        vec![
            i64s([1, 2, 3]),
            i64s([1, 2, 3]),
            i64s([1, 2, 3]),
            i32s([1, 1, 1]),
            money([1700, 3600, 800]),
            money([211685, 459831, 133096]),
            money([6, 9, 2]),
            money([2, 6, 1]),
            strings(["N", "R", "A"]),
            strings(["O", "F", "F"]),
            dates([(1994, 3, 13), (1995, 9, 18), (1998, 8, 2)]),
            dates([(1994, 2, 12), (1995, 9, 10), (1998, 7, 20)]),
            dates([(1994, 3, 22), (1995, 10, 1), (1998, 8, 15)]),
            strings(["DELIVER IN PERSON", "TAKE BACK RETURN", "NONE"]),
            strings(["TRUCK", "MAIL", "AIR"]),
            strings(["lineitem one", "lineitem two", "lineitem three"]),
        ],
    )
}

fn write_nation(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "nation",
        schema(vec![
            field("n_nationkey", DataType::Int64),
            field("n_name", DataType::Utf8),
            field("n_regionkey", DataType::Int64),
            field("n_comment", DataType::Utf8),
        ]),
        vec![
            i64s([0, 1, 2, 3, 4]),
            strings(["ALGERIA", "BRAZIL", "CANADA", "FRANCE", "GERMANY"]),
            i64s([0, 1, 1, 2, 2]),
            strings(["n0", "n1", "n2", "n3", "n4"]),
        ],
    )
}

fn write_orders(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "orders",
        schema(vec![
            field("o_orderkey", DataType::Int64),
            field("o_custkey", DataType::Int64),
            field("o_orderstatus", DataType::Utf8),
            field("o_totalprice", money_type()),
            field("o_orderdate", DataType::Date32),
            field("o_orderpriority", DataType::Utf8),
            field("o_clerk", DataType::Utf8),
            field("o_shippriority", DataType::Int32),
            field("o_comment", DataType::Utf8),
        ]),
        vec![
            i64s([1, 2, 3]),
            i64s([1, 2, 3]),
            strings(["O", "F", "O"]),
            money([17279949, 3842632, 20565430]),
            dates([(1995, 3, 15), (1993, 10, 14), (1994, 7, 30)]),
            strings(["1-URGENT", "2-HIGH", "5-LOW"]),
            strings(["Clerk#000000001", "Clerk#000000002", "Clerk#000000003"]),
            i32s([0, 0, 0]),
            strings(["order one", "special requests", "order three"]),
        ],
    )
}

fn write_part(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "part",
        schema(vec![
            field("p_partkey", DataType::Int64),
            field("p_name", DataType::Utf8),
            field("p_mfgr", DataType::Utf8),
            field("p_brand", DataType::Utf8),
            field("p_type", DataType::Utf8),
            field("p_size", DataType::Int32),
            field("p_container", DataType::Utf8),
            field("p_retailprice", money_type()),
            field("p_comment", DataType::Utf8),
        ]),
        vec![
            i64s([1, 2, 3]),
            strings(["forest green part", "promo part", "standard part"]),
            strings(["Manufacturer#1", "Manufacturer#2", "Manufacturer#3"]),
            strings(["Brand#12", "Brand#23", "Brand#34"]),
            strings(["LARGE BRASS", "PROMO BURNISHED COPPER", "ECONOMY ANODIZED STEEL"]),
            i32s([15, 7, 23]),
            strings(["SM CASE", "MED BOX", "LG PACK"]),
            money([90100, 120000, 45000]),
            strings(["part one", "part two", "part three"]),
        ],
    )
}

fn write_partsupp(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "partsupp",
        schema(vec![
            field("ps_partkey", DataType::Int64),
            field("ps_suppkey", DataType::Int64),
            field("ps_availqty", DataType::Int32),
            field("ps_supplycost", money_type()),
            field("ps_comment", DataType::Utf8),
        ]),
        vec![
            i64s([1, 2, 3]),
            i64s([1, 2, 3]),
            i32s([3325, 8076, 3956]),
            money([77164, 99349, 33709]),
            strings(["partsupp one", "partsupp two", "partsupp three"]),
        ],
    )
}

fn write_region(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "region",
        schema(vec![
            field("r_regionkey", DataType::Int64),
            field("r_name", DataType::Utf8),
            field("r_comment", DataType::Utf8),
        ]),
        vec![i64s([0, 1, 2]), strings(["AFRICA", "AMERICA", "EUROPE"]), strings(["r0", "r1", "r2"])],
    )
}

fn write_supplier(root: &Path) -> WorkloadResult<()> {
    write_table(
        root,
        "supplier",
        schema(vec![
            field("s_suppkey", DataType::Int64),
            field("s_name", DataType::Utf8),
            field("s_address", DataType::Utf8),
            field("s_nationkey", DataType::Int64),
            field("s_phone", DataType::Utf8),
            field("s_acctbal", money_type()),
            field("s_comment", DataType::Utf8),
        ]),
        vec![
            i64s([1, 2, 3]),
            strings(["Supplier#000000001", "Supplier#000000002", "Supplier#000000003"]),
            strings(["Supplier address 1", "Supplier address 2", "Supplier address 3"]),
            i64s([3, 4, 1]),
            strings(["27-111-111-1111", "29-222-222-2222", "13-333-333-3333"]),
            money([575594, 403200, -28384]),
            strings(["reliable supplier", "supplier with Customer Complaints", "supplier three"]),
        ],
    )
}

fn write_table(root: &Path, table: &str, schema: Arc<Schema>, columns: Vec<ArrayRef>) -> WorkloadResult<()> {
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let file = File::create(root.join(format!("{table}.parquet")))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn schema(fields: Vec<Field>) -> Arc<Schema> {
    Arc::new(Schema::new(fields))
}

fn field(name: &'static str, data_type: DataType) -> Field {
    Field::new(name, data_type, false)
}

fn money_type() -> DataType {
    DataType::Decimal128(15, 2)
}

fn i32s<const N: usize>(values: [i32; N]) -> ArrayRef {
    Arc::new(Int32Array::from(Vec::from(values)))
}

fn i64s<const N: usize>(values: [i64; N]) -> ArrayRef {
    Arc::new(Int64Array::from(Vec::from(values)))
}

fn strings<const N: usize>(values: [&'static str; N]) -> ArrayRef {
    Arc::new(StringArray::from(Vec::from(values)))
}

fn money<const N: usize>(values: [i128; N]) -> ArrayRef {
    Arc::new(Decimal128Array::from_iter_values(values).with_precision_and_scale(15, 2).unwrap())
}

fn dates<const N: usize>(values: [(i32, u32, u32); N]) -> ArrayRef {
    Arc::new(Date32Array::from(values.map(|(year, month, day)| date32(year, month, day)).to_vec()))
}

fn date32(year: i32, month: u32, day: u32) -> i32 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 { adjusted_year } else { adjusted_year - 399 } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month = month as i32;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}
