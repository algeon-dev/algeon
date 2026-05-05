use std::{
    error::Error as StdError,
    fmt::{Display, Formatter},
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, Date32Array, Decimal128Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    error::ArrowError,
    record_batch::RecordBatch,
};
use datafusion::{
    error::DataFusionError,
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, errors::ParquetError, file::properties::WriterProperties},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use parquet::file::reader::{FileReader, SerializedFileReader};

pub use crate::tpch_queries::{TPCH_QUERIES, TPCH_QUERY_COUNT, TpchQuery, tpch_query_ids};
use crate::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfPhysicalOptimizerRule, CudfSessionStateBuilderExt,
    GpuFallbackPolicy, NativeCudfExecutionConfig, NativeDataFusionPlanningWorkloadItem,
    NativeFallbackPressureWorkloadItem,
};

pub const TPCH_TABLES: &[&str] =
    &["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"];
pub const CUDF_TPCH_PARQUET_DIR_ENV: &str = "CUDF_TPCH_PARQUET_DIR";
pub const CUDF_TPCH_SCALE_FACTOR_ENV: &str = "CUDF_TPCH_SCALE_FACTOR";
pub const CUDF_TPCH_GENERATION_COMMAND_ENV: &str = "CUDF_TPCH_GENERATION_COMMAND";

pub type TpchWorkloadResult<T> = std::result::Result<T, TpchWorkloadError>;

#[derive(Debug)]
pub enum TpchWorkloadError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    NonUtf8Path(PathBuf),
    InvalidDataset(String),
}

impl Display for TpchWorkloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Arrow(error) => write!(f, "Arrow error: {error}"),
            Self::Parquet(error) => write!(f, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(f, "DataFusion error: {error}"),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::InvalidDataset(message) => write!(f, "invalid TPC-H dataset: {message}"),
        }
    }
}

impl StdError for TpchWorkloadError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Arrow(error) => Some(error),
            Self::Parquet(error) => Some(error),
            Self::DataFusion(error) => Some(error),
            Self::NonUtf8Path(_) => None,
            Self::InvalidDataset(_) => None,
        }
    }
}

impl From<std::io::Error> for TpchWorkloadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ArrowError> for TpchWorkloadError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<ParquetError> for TpchWorkloadError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}

impl From<DataFusionError> for TpchWorkloadError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}

pub fn tpch_report_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_joins(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

pub fn tpch_native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1))
}

pub fn write_tiny_tpch_parquet_dataset(root: &Path) -> TpchWorkloadResult<()> {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpchDatasetMetadata {
    pub root: PathBuf,
    pub scale_factor: String,
    pub provenance: String,
    pub generation_command: String,
    pub table_count: usize,
    pub file_count: usize,
    pub total_rows: u64,
    pub total_bytes: u64,
    pub fingerprint: String,
    pub tables: Vec<TpchTableMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpchTableMetadata {
    pub table: String,
    pub path: PathBuf,
    pub file_count: usize,
    pub row_count: u64,
    pub byte_size: u64,
    pub row_group_count: usize,
    pub row_group_rows: Vec<u64>,
    pub row_group_bytes: Vec<u64>,
}

impl TpchDatasetMetadata {
    pub fn table_row_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.row_count)
    }

    pub fn table_byte_sizes(&self) -> String {
        format_table_metadata(&self.tables, |table| table.byte_size)
    }

    pub fn table_file_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.file_count as u64)
    }

    pub fn table_row_group_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.row_group_count as u64)
    }
}

pub fn tiny_tpch_dataset_metadata(root: &Path) -> TpchWorkloadResult<TpchDatasetMetadata> {
    describe_tpch_dataset(root, "tiny", "generated_tiny", "datafusion-cudf::write_tiny_tpch_parquet_dataset")
}

pub fn external_tpch_dataset_metadata(
    root: &Path,
    scale_factor: impl Into<String>,
    generation_command: impl Into<String>,
) -> TpchWorkloadResult<TpchDatasetMetadata> {
    describe_tpch_dataset(root, scale_factor, "external_parquet_root", generation_command)
}

pub fn prepare_tpch_report_dataset(tiny_root: &Path) -> TpchWorkloadResult<(PathBuf, TpchDatasetMetadata)> {
    if let Ok(root) = std::env::var(CUDF_TPCH_PARQUET_DIR_ENV) {
        let root = PathBuf::from(root);
        let scale_factor = std::env::var(CUDF_TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|_| "unknown".to_string());
        let generation_command =
            std::env::var(CUDF_TPCH_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unknown_external".to_string());
        let metadata = external_tpch_dataset_metadata(&root, scale_factor, generation_command)?;
        return Ok((root, metadata));
    }

    write_tiny_tpch_parquet_dataset(tiny_root)?;
    let metadata = tiny_tpch_dataset_metadata(tiny_root)?;
    Ok((tiny_root.to_path_buf(), metadata))
}

pub fn describe_tpch_dataset(
    root: &Path,
    scale_factor: impl Into<String>,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
) -> TpchWorkloadResult<TpchDatasetMetadata> {
    let mut tables = Vec::with_capacity(TPCH_TABLES.len());
    for table in TPCH_TABLES {
        tables.push(read_tpch_table_metadata(root, table)?);
    }

    let file_count = tables.iter().map(|table| table.file_count).sum();
    let total_rows = tables.iter().map(|table| table.row_count).sum();
    let total_bytes = tables.iter().map(|table| table.byte_size).sum();
    let fingerprint = tpch_dataset_fingerprint(&tables);
    Ok(TpchDatasetMetadata {
        root: root.to_path_buf(),
        scale_factor: scale_factor.into(),
        provenance: provenance.into(),
        generation_command: generation_command.into(),
        table_count: tables.len(),
        file_count,
        total_rows,
        total_bytes,
        fingerprint,
        tables,
    })
}

pub async fn tpch_fallback_pressure_workload_items(
    root: &Path,
    config: SessionConfig,
) -> TpchWorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
    let ctx = SessionContext::new_with_config(config);
    register_tpch_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCH_QUERIES.len());
    for query in TPCH_QUERIES {
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => NativeFallbackPressureWorkloadItem::new(query.id, plan),
                Err(error) => NativeFallbackPressureWorkloadItem::planning_failure(
                    query.id,
                    "physical_planning",
                    "physical_planning_failed",
                    error.to_string(),
                ),
            },
            Err(error) => NativeFallbackPressureWorkloadItem::planning_failure(
                query.id,
                "sql_to_logical_plan",
                "sql_planning_failed",
                error.to_string(),
            ),
        };
        items.push(item);
    }
    Ok(items)
}

pub async fn tpch_datafusion_planning_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> TpchWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let optimizer_config = tpch_native_optimizer_config();
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let gpu_island_rule = Arc::new(CudfPhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_and_gpu_island_rules(Arc::clone(&native_rule), gpu_island_rule)
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpch_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCH_QUERIES.len());
    for query in TPCH_QUERIES {
        let df = ctx.sql(query.sql).await?;
        let plan = df.create_physical_plan().await?;
        let diagnostics = native_rule.last_planning_diagnostics().ok_or_else(|| {
            TpchWorkloadError::InvalidDataset(format!("missing retained native planning diagnostics for {}", query.id))
        })?;
        items.push(NativeDataFusionPlanningWorkloadItem::new(query.id, plan, diagnostics));
    }
    Ok(items)
}

async fn register_tpch_tables(ctx: &SessionContext, root: &Path) -> TpchWorkloadResult<()> {
    for table in TPCH_TABLES {
        let path = root.join(format!("{table}.parquet"));
        let path = path.to_str().ok_or_else(|| TpchWorkloadError::NonUtf8Path(path.clone()))?;
        ctx.register_parquet(*table, path, ParquetReadOptions::default()).await?;
    }
    Ok(())
}

fn read_tpch_table_metadata(root: &Path, table: &str) -> TpchWorkloadResult<TpchTableMetadata> {
    let path = root.join(format!("{table}.parquet"));
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() {
        return Err(TpchWorkloadError::InvalidDataset(format!("expected {} to be a Parquet file", path.display())));
    }

    let byte_size = metadata.len();
    let file = File::open(&path)?;
    let reader = SerializedFileReader::new(file)?;
    let parquet_metadata = reader.metadata();
    let row_group_count = parquet_metadata.num_row_groups();
    let mut row_group_rows = Vec::with_capacity(row_group_count);
    let mut row_group_bytes = Vec::with_capacity(row_group_count);
    for row_group in parquet_metadata.row_groups() {
        let rows = u64::try_from(row_group.num_rows()).map_err(|_| {
            TpchWorkloadError::InvalidDataset(format!("negative row count in Parquet metadata for {}", path.display()))
        })?;
        let bytes = u64::try_from(row_group.compressed_size()).map_err(|_| {
            TpchWorkloadError::InvalidDataset(format!(
                "negative compressed byte size in Parquet metadata for {}",
                path.display()
            ))
        })?;
        row_group_rows.push(rows);
        row_group_bytes.push(bytes);
    }
    let row_count = row_group_rows.iter().sum();
    Ok(TpchTableMetadata {
        table: table.to_string(),
        path,
        file_count: 1,
        row_count,
        byte_size,
        row_group_count,
        row_group_rows,
        row_group_bytes,
    })
}

fn format_table_metadata(tables: &[TpchTableMetadata], value: impl Fn(&TpchTableMetadata) -> u64) -> String {
    tables.iter().map(|table| format!("{}={}", table.table, value(table))).collect::<Vec<_>>().join(",")
}

fn tpch_dataset_fingerprint(tables: &[TpchTableMetadata]) -> String {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-cudf-tpch-dataset-metadata-v1");
    for table in tables {
        hasher.update_field(&table.table);
        hasher.update_field(&table.file_count.to_string());
        hasher.update_field(&table.byte_size.to_string());
        hasher.update_field(&table.row_count.to_string());
        hasher.update_field(&table.row_group_count.to_string());
        for rows in &table.row_group_rows {
            hasher.update_field(&rows.to_string());
        }
        for bytes in &table.row_group_bytes {
            hasher.update_field(&bytes.to_string());
        }
    }
    format!("metadata-fnv1a64:{:016x}", hasher.finish())
}

struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self { state: Self::OFFSET }
    }

    fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[0]);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    fn finish(self) -> u64 {
        self.state
    }
}

fn write_customer(root: &Path) -> TpchWorkloadResult<()> {
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
            strings(["BUILDING", "AUTOMOBILE", "HOUSEHOLD"]),
            strings(["regular customer", "priority customer", "careful customer"]),
        ],
    )
}

fn write_lineitem(root: &Path) -> TpchWorkloadResult<()> {
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
            money([4, 9, 2]),
            money([2, 6, 1]),
            strings(["N", "R", "A"]),
            strings(["O", "F", "F"]),
            dates([(1996, 3, 13), (1995, 9, 18), (1998, 8, 2)]),
            dates([(1996, 2, 12), (1995, 9, 10), (1998, 7, 20)]),
            dates([(1996, 3, 22), (1995, 10, 1), (1998, 8, 15)]),
            strings(["DELIVER IN PERSON", "TAKE BACK RETURN", "NONE"]),
            strings(["TRUCK", "MAIL", "AIR"]),
            strings(["lineitem one", "lineitem two", "lineitem three"]),
        ],
    )
}

fn write_nation(root: &Path) -> TpchWorkloadResult<()> {
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

fn write_orders(root: &Path) -> TpchWorkloadResult<()> {
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

fn write_part(root: &Path) -> TpchWorkloadResult<()> {
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

fn write_partsupp(root: &Path) -> TpchWorkloadResult<()> {
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

fn write_region(root: &Path) -> TpchWorkloadResult<()> {
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

fn write_supplier(root: &Path) -> TpchWorkloadResult<()> {
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

fn write_table(root: &Path, table: &str, schema: Arc<Schema>, columns: Vec<ArrayRef>) -> TpchWorkloadResult<()> {
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
