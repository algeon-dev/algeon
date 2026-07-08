use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, UInt32Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::{FlightEndpoint, Ticket, sql::client::FlightSqlServiceClient};
use futures::TryStreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use tonic::transport::Endpoint;

type SmokeResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const SOURCE_ENV: &str = "NEXUS_CUGRAPH_SMOKE_SOURCE";
const ICEBERG_EDGE_TABLE_ENV: &str = "NEXUS_CUGRAPH_SMOKE_ICEBERG_EDGE_TABLE";
const ICEBERG_SRC_COL_ENV: &str = "NEXUS_CUGRAPH_SMOKE_ICEBERG_SRC_COL";
const ICEBERG_DST_COL_ENV: &str = "NEXUS_CUGRAPH_SMOKE_ICEBERG_DST_COL";
const ICEBERG_WEIGHT_COL_ENV: &str = "NEXUS_CUGRAPH_SMOKE_ICEBERG_WEIGHT_COL";
const RESULT_MODE_ENV: &str = "NEXUS_CUGRAPH_SMOKE_RESULT_MODE";

const DEFAULT_ICEBERG_SRC_COL: &str = "src";
const DEFAULT_ICEBERG_DST_COL: &str = "dst";
const DEFAULT_ICEBERG_WEIGHT_COL: &str = "weight";

pub async fn run(url_arg: Option<String>) -> SmokeResult<()> {
    let url = server_url(url_arg);
    let source = SmokeSource::from_env()?;
    let result_mode = GraphResultMode::from_env()?;
    let mut client = connect_with_retry(&url).await?;
    let suffix = unique_suffix();
    let raw_table = format!("raw_edges_{suffix}");
    let target_view = format!("target_edges_{suffix}");
    let graph_view = format!("graph_hits_{suffix}");
    let mut cleanup_dir = None;
    let weight_col = source.weight_col();

    match &source {
        SmokeSource::LocalParquet => {
            let dir = unique_temp_dir("datafusion-nexus-flight-sql-cugraph-smoke")?;
            let parquet_path = dir.join("edges.parquet");
            write_edge_parquet(&parquet_path)?;
            execute_batches(
                &mut client,
                &format!(
                    "CREATE EXTERNAL TABLE {raw_table} STORED AS PARQUET LOCATION '{}'",
                    parquet_path.to_string_lossy()
                ),
            )
            .await?;
            execute_batches(
                &mut client,
                &format!("CREATE VIEW {target_view} AS SELECT src, dst, weight FROM {raw_table} WHERE src >= 0"),
            )
            .await?;
            cleanup_dir = Some(dir);
        },
        SmokeSource::Iceberg { edge_table, src_col, dst_col, weight_col } => {
            let weight_select = match weight_col {
                Some(weight_col) => format!(", {weight_col} AS weight"),
                None => String::new(),
            };
            let weight_filter = match weight_col {
                Some(weight_col) => format!(" AND {weight_col} IS NOT NULL"),
                None => String::new(),
            };
            execute_batches(
                &mut client,
                &format!(
                    "CREATE VIEW {target_view} AS \
                     SELECT {src_col} AS src, {dst_col} AS dst{weight_select} \
                     FROM {edge_table} \
                     WHERE {src_col} IS NOT NULL AND {dst_col} IS NOT NULL{weight_filter}",
                ),
            )
            .await?;
        },
    }

    let batches = execute_batches(
        &mut client,
        "SELECT function_name \
         FROM cugraph_list_algorithms() \
         WHERE function_name IN ('cugraph_bfs', 'cugraph_pagerank') \
         ORDER BY function_name",
    )
    .await?;
    let batch = single_batch(&batches)?;
    require_string(batch, 0, 0, "cugraph_bfs")?;
    require_string(batch, 0, 1, "cugraph_pagerank")?;

    let validate_sql = format!(
        r#"SELECT valid, error_code, table_resolved, src_dtype, dst_dtype, weight_dtype, would_execute_gpu
           FROM cugraph_validate_call(
             'cugraph_bfs',
             '{target_view}',
             '{{"source_vertex":0,"src_col":"src","dst_col":"dst","weight_col":{weight_col_json},"options":{{"depth_limit":2}}}}'
           )"#,
        weight_col_json = weight_col.map(|column| format!("\"{column}\"")).unwrap_or_else(|| "null".to_string())
    );
    let batches = execute_batches(&mut client, &validate_sql).await?;
    let batch = single_batch(&batches)?;
    require_bool(batch, 0, 0, true)?;
    require_null(batch, 1, 0)?;
    require_bool(batch, 2, 0, true)?;
    require_string(batch, 3, 0, "Int64")?;
    require_string(batch, 4, 0, "Int64")?;
    match weight_col {
        Some(_) => require_string(batch, 5, 0, "Float64")?,
        None => require_null(batch, 5, 0)?,
    }
    require_bool(batch, 6, 0, true)?;

    let graph_weight_arg = weight_col.map(|column| format!("'{column}'")).unwrap_or_else(|| "NULL".to_string());
    execute_batches(
        &mut client,
        &format!(
            r#"CREATE VIEW {graph_view} AS
               SELECT *
               FROM cugraph_bfs('{target_view}', 0, 'src', 'dst', {graph_weight_arg}, '{{"depth_limit":2}}')"#
        ),
    )
    .await?;
    let schema = execute_schema(&mut client, &format!("SELECT vertex, distance FROM {graph_view} LIMIT 0")).await?;
    if schema.fields().len() != 2 || schema.field(0).name() != "vertex" || schema.field(1).name() != "distance" {
        return Err(format!("unexpected graph schema: {schema:?}").into());
    }

    if result_mode == GraphResultMode::FullUnbounded {
        execute_full_unbounded_graph_smoke(&mut client, &graph_view).await?;
    }

    if let Some(dir) = cleanup_dir {
        let _ = fs::remove_dir_all(&dir);
    }
    println!(
        "flight-sql-cugraph-smoke ok url={url} source={} result_mode={} target_view={target_view} graph_view={graph_view}",
        source.label(),
        result_mode.label()
    );
    Ok(())
}

enum SmokeSource {
    LocalParquet,
    Iceberg { edge_table: String, src_col: String, dst_col: String, weight_col: Option<String> },
}

impl SmokeSource {
    fn from_env() -> SmokeResult<Self> {
        match std::env::var(SOURCE_ENV).unwrap_or_else(|_| "local_parquet".to_string()).as_str() {
            "local_parquet" | "local" => Ok(Self::LocalParquet),
            "iceberg" | "remote_iceberg" => Ok(Self::Iceberg {
                edge_table: required_nonempty_env(ICEBERG_EDGE_TABLE_ENV)?,
                src_col: env_or_default(ICEBERG_SRC_COL_ENV, DEFAULT_ICEBERG_SRC_COL),
                dst_col: env_or_default(ICEBERG_DST_COL_ENV, DEFAULT_ICEBERG_DST_COL),
                weight_col: optional_nonempty_env(ICEBERG_WEIGHT_COL_ENV).or_else(|| {
                    if std::env::var(ICEBERG_WEIGHT_COL_ENV).is_ok() {
                        None
                    } else {
                        Some(DEFAULT_ICEBERG_WEIGHT_COL.to_string())
                    }
                }),
            }),
            other => Err(format!("unsupported {SOURCE_ENV}={other:?}; expected local_parquet or iceberg").into()),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::LocalParquet => "local_parquet",
            Self::Iceberg { .. } => "iceberg",
        }
    }

    fn weight_col(&self) -> Option<&str> {
        match self {
            Self::LocalParquet => Some("weight"),
            Self::Iceberg { weight_col, .. } => weight_col.as_deref(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphResultMode {
    SchemaOnly,
    FullUnbounded,
}

impl GraphResultMode {
    fn from_env() -> SmokeResult<Self> {
        match std::env::var(RESULT_MODE_ENV).unwrap_or_else(|_| "schema_only".to_string()).as_str() {
            "schema_only" | "schema" | "planning" => Ok(Self::SchemaOnly),
            "full_unbounded" => Ok(Self::FullUnbounded),
            other => {
                Err(format!("unsupported {RESULT_MODE_ENV}={other:?}; expected schema_only or full_unbounded").into())
            },
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::SchemaOnly => "schema_only",
            Self::FullUnbounded => "full_unbounded",
        }
    }
}

fn env_or_default(name: &str, default: &str) -> String {
    optional_nonempty_env(name).unwrap_or_else(|| default.to_string())
}

fn required_nonempty_env(name: &str) -> SmokeResult<String> {
    optional_nonempty_env(name).ok_or_else(|| format!("{name} must be set when {SOURCE_ENV}=iceberg").into())
}

fn optional_nonempty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|value| value.trim().to_string()).filter(|value| !value.is_empty())
}

fn server_url(url_arg: Option<String>) -> String {
    url_arg
        .or_else(|| std::env::var("NEXUS_FLIGHT_SQL_URL").ok())
        .unwrap_or_else(|| "http://127.0.0.1:50051".to_string())
}

fn unique_suffix() -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("{}_{}", std::process::id(), nanos)
}

fn unique_temp_dir(prefix: &str) -> SmokeResult<PathBuf> {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", unique_suffix()));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn write_edge_parquet(path: &Path) -> SmokeResult<()> {
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(Int64Array::from(vec![0_i64, 1, 0])) as ArrayRef,
            std::sync::Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            std::sync::Arc::new(Float64Array::from(vec![1.0_f64, 1.0, 1.0])) as ArrayRef,
        ],
    )?;
    let file = fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

async fn connect_with_retry(url: &str) -> SmokeResult<FlightSqlServiceClient<tonic::transport::Channel>> {
    let mut last_error = None;
    for _ in 0..50 {
        match Endpoint::from_shared(url.to_string())?.connect().await {
            Ok(channel) => return Ok(FlightSqlServiceClient::new(channel)),
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(Duration::from_millis(50)).await;
            },
        }
    }
    Err(format!("Flight SQL server did not accept connections at {url}: {last_error:?}").into())
}

async fn execute_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> SmokeResult<Vec<RecordBatch>> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    let ticket = first_ticket(&flight_info.endpoint)?;
    Ok(client.do_get(ticket).await?.try_collect::<Vec<_>>().await?)
}

async fn execute_schema(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> SmokeResult<Schema> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    let ticket = first_ticket(&flight_info.endpoint)?;
    let schema = flight_info.try_decode_schema()?;
    let _ = client.do_get(ticket).await?.try_collect::<Vec<_>>().await?;
    Ok(schema)
}

async fn execute_full_unbounded_graph_smoke(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    graph_view: &str,
) -> SmokeResult<()> {
    let batches = execute_batches(
        client,
        &format!(
            "SELECT distance, COUNT(*) AS vertex_count \
             FROM {graph_view} \
             GROUP BY distance \
             ORDER BY distance"
        ),
    )
    .await?;
    let row_count = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if row_count == 0 {
        return Err("full_unbounded graph smoke returned no aggregate rows".into());
    }
    let schema = batches.first().ok_or("full_unbounded graph smoke returned no batches")?.schema();
    if schema.fields().len() != 2 || schema.field(0).name() != "distance" || schema.field(1).name() != "vertex_count" {
        return Err(format!("unexpected graph aggregate schema: {schema:?}").into());
    }
    let total_vertices = sum_integer_column(&batches, 1)?;
    if total_vertices == 0 {
        return Err("full_unbounded graph smoke returned zero vertices".into());
    }
    Ok(())
}

fn first_ticket(endpoints: &[FlightEndpoint]) -> SmokeResult<Ticket> {
    endpoints.first().and_then(|endpoint| endpoint.ticket.clone()).ok_or_else(|| "missing Flight SQL ticket".into())
}

fn single_batch(batches: &[RecordBatch]) -> SmokeResult<&RecordBatch> {
    if batches.len() != 1 {
        return Err(format!("expected one batch, got {}", batches.len()).into());
    }
    Ok(&batches[0])
}

fn require_string(batch: &RecordBatch, column: usize, row: usize, expected: &str) -> SmokeResult<()> {
    let array = batch.column(column).as_any().downcast_ref::<StringArray>().ok_or("expected StringArray")?;
    if array.is_null(row) {
        return Err(format!("expected {expected:?}, got NULL").into());
    }
    let actual = array.value(row);
    if actual != expected {
        return Err(format!("expected {expected:?}, got {actual:?}").into());
    }
    Ok(())
}

fn require_null(batch: &RecordBatch, column: usize, row: usize) -> SmokeResult<()> {
    if !batch.column(column).is_null(row) {
        return Err(format!("expected NULL at column {column} row {row}").into());
    }
    Ok(())
}

fn require_bool(batch: &RecordBatch, column: usize, row: usize, expected: bool) -> SmokeResult<()> {
    let array = batch.column(column).as_any().downcast_ref::<BooleanArray>().ok_or("expected BooleanArray")?;
    let actual = array.value(row);
    if actual != expected {
        return Err(format!("expected {expected}, got {actual}").into());
    }
    Ok(())
}

fn sum_integer_column(batches: &[RecordBatch], column: usize) -> SmokeResult<u64> {
    let mut total = 0_u64;
    for batch in batches {
        let array = batch.column(column);
        if let Some(array) = array.as_any().downcast_ref::<UInt64Array>() {
            for row in 0..array.len() {
                if !array.is_null(row) {
                    total = total.saturating_add(array.value(row));
                }
            }
        } else if let Some(array) = array.as_any().downcast_ref::<UInt32Array>() {
            for row in 0..array.len() {
                if !array.is_null(row) {
                    total = total.saturating_add(u64::from(array.value(row)));
                }
            }
        } else if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
            for row in 0..array.len() {
                if !array.is_null(row) {
                    let value = array.value(row);
                    if value < 0 {
                        return Err(format!("expected non-negative integer count, got {value}").into());
                    }
                    total = total.saturating_add(value as u64);
                }
            }
        } else {
            return Err(format!("expected integer array for aggregate count, got {:?}", array.data_type()).into());
        }
    }
    Ok(total)
}
