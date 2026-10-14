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
use tonic::transport::Channel;

use crate::connection_retry::{ConnectionRetryConfig, connect_with_retry};

type SmokeResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const RESULT_MODE_ENV: &str = "ALGEON_CUGRAPH_SMOKE_RESULT_MODE";

pub async fn run(url_arg: Option<String>) -> SmokeResult<()> {
    let url = server_url(url_arg);
    let result_mode = GraphResultMode::from_env()?;
    let mut client = connect_flight_sql(&url).await?;
    let suffix = unique_suffix();
    let raw_table = format!("raw_edges_{suffix}");
    let target_view = format!("target_edges_{suffix}");
    let graph_view = format!("graph_hits_{suffix}");
    let dir = unique_temp_dir("algeon-datafusion-flight-sql-cugraph-smoke")?;
    let parquet_path = dir.join("edges.parquet");
    write_edge_parquet(&parquet_path)?;
    execute_batches(
        &mut client,
        &format!("CREATE EXTERNAL TABLE {raw_table} STORED AS PARQUET LOCATION '{}'", parquet_path.to_string_lossy()),
    )
    .await?;
    execute_batches(
        &mut client,
        &format!("CREATE VIEW {target_view} AS SELECT src, dst, weight FROM {raw_table} WHERE src >= 0"),
    )
    .await?;

    let batches = execute_batches(
        &mut client,
        "SELECT function_name \
         FROM gpu_list_functions() \
         WHERE function_name IN ('cugraph_bfs', 'cugraph_pagerank') \
         ORDER BY function_name",
    )
    .await?;
    let batch = single_batch(&batches)?;
    require_string(batch, 0, 0, "cugraph_bfs")?;
    require_string(batch, 0, 1, "cugraph_pagerank")?;

    let validate_sql = format!(
        r#"SELECT valid, error_code, relations_resolved, details_json, would_execute_gpu
           FROM gpu_validate_call(
             'cugraph_bfs',
             '{{"relations":{{"edges":{{"table":"{target_view}"}}}},"options":{{"source_vertex":0,"depth_limit":2}}}}'
           )"#,
    );
    let batches = execute_batches(&mut client, &validate_sql).await?;
    let batch = single_batch(&batches)?;
    require_bool(batch, 0, 0, true)?;
    require_null(batch, 1, 0)?;
    require_bool(batch, 2, 0, true)?;
    let details: serde_json::Value = serde_json::from_str(string_at(batch, 3, 0)?)?;
    if details["edges"]["src_dtype"] != "Int64" || details["edges"]["dst_dtype"] != "Int64" {
        return Err(format!("unexpected cuGraph edge dtypes: {details}").into());
    }
    if details["edges"]["weight_dtype"] != "Float64" {
        return Err(format!("unexpected cuGraph edge weight dtype: {details}").into());
    }
    require_bool(batch, 4, 0, true)?;

    execute_batches(
        &mut client,
        &format!(
            r#"CREATE VIEW {graph_view} AS
               SELECT *
               FROM cugraph_bfs(
                 edges => (SELECT src, dst, weight FROM {target_view}), source_vertex => 0, depth_limit => 2)"#
        ),
    )
    .await?;
    let schema = execute_schema(&mut client, &format!("SELECT * FROM {graph_view} LIMIT 0")).await?;
    let expected_fields = [
        ("vertex", DataType::Int64, false),
        ("distance", DataType::Int64, true),
        ("predecessor", DataType::Int64, true),
        ("reachable", DataType::Boolean, false),
    ];
    if schema.fields().len() != expected_fields.len()
        || schema.fields().iter().zip(expected_fields).any(|(field, (name, data_type, nullable))| {
            field.name() != name || field.data_type() != &data_type || field.is_nullable() != nullable
        })
    {
        return Err(format!("unexpected graph schema: {schema:?}").into());
    }

    if result_mode == GraphResultMode::FullUnbounded {
        execute_full_unbounded_graph_smoke(&mut client, &graph_view).await?;
    }

    let _ = fs::remove_dir_all(&dir);
    println!(
        "flight-sql-cugraph-smoke ok url={url} source=local_parquet result_mode={} target_view={target_view} graph_view={graph_view}",
        result_mode.label()
    );
    Ok(())
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

fn server_url(url_arg: Option<String>) -> String {
    url_arg
        .or_else(|| std::env::var("ALGEON_FLIGHT_SQL_URL").ok())
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

async fn connect_flight_sql(url: &str) -> SmokeResult<FlightSqlServiceClient<Channel>> {
    connect_with_retry(
        url,
        ConnectionRetryConfig::fixed_attempts(50, Duration::from_millis(50)),
        FlightSqlServiceClient::new,
        |error| format!("Flight SQL server did not accept connections at {url}: {:?}", Some(error)),
    )
    .await
}

async fn execute_batches(client: &mut FlightSqlServiceClient<Channel>, sql: &str) -> SmokeResult<Vec<RecordBatch>> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    let ticket = first_ticket(&flight_info.endpoint)?;
    Ok(client.do_get(ticket).await?.try_collect::<Vec<_>>().await?)
}

async fn execute_schema(client: &mut FlightSqlServiceClient<Channel>, sql: &str) -> SmokeResult<Schema> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    let ticket = first_ticket(&flight_info.endpoint)?;
    let schema = flight_info.try_decode_schema()?;
    let _ = client.do_get(ticket).await?.try_collect::<Vec<_>>().await?;
    Ok(schema)
}

async fn execute_full_unbounded_graph_smoke(
    client: &mut FlightSqlServiceClient<Channel>,
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
    let actual = string_at(batch, column, row)?;
    if actual != expected {
        return Err(format!("expected {expected:?}, got {actual:?}").into());
    }
    Ok(())
}

fn string_at(batch: &RecordBatch, column: usize, row: usize) -> SmokeResult<&str> {
    let array = batch.column(column).as_any().downcast_ref::<StringArray>().ok_or("expected StringArray")?;
    if array.is_null(row) {
        return Err(format!("expected string at column {column} row {row}, got NULL").into());
    }
    Ok(array.value(row))
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
