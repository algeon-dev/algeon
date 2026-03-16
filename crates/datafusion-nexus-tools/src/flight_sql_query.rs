use std::{error::Error, time::Duration};

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        LargeStringArray, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::DataType,
    record_batch::RecordBatch,
};
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use tonic::transport::Endpoint;

type QueryResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const DEFAULT_URL: &str = "http://127.0.0.1:50051";
const DEFAULT_SQL: &str = "SELECT 1";
const URL_ENV: &str = "NEXUS_FLIGHT_SQL_URL";

macro_rules! downcast_value {
    ($array:expr, $row:expr, $array_ty:ty, str) => {
        if let Some(values) = $array.as_any().downcast_ref::<$array_ty>() {
            return values.value($row).to_string();
        }
    };
    ($array:expr, $row:expr, $array_ty:ty, $value_ty:ty) => {
        if let Some(values) = $array.as_any().downcast_ref::<$array_ty>() {
            let value: $value_ty = values.value($row);
            return value.to_string();
        }
    };
}

pub async fn run(args: Vec<String>) -> QueryResult<()> {
    let (url, sql) = parse_args(args);
    let mut client = connect_with_retry(&url).await?;
    let batches = execute_batches(&mut client, &sql).await?;
    print_batches(&batches);
    Ok(())
}

fn parse_args(args: Vec<String>) -> (String, String) {
    match args.split_first() {
        Some((first, rest)) if first.starts_with("http://") || first.starts_with("https://") => {
            let sql = join_sql(rest).unwrap_or_else(|| DEFAULT_SQL.to_string());
            (first.clone(), sql)
        },
        Some(_) => {
            let url = std::env::var(URL_ENV).unwrap_or_else(|_| DEFAULT_URL.to_string());
            let sql = join_sql(&args).unwrap_or_else(|| DEFAULT_SQL.to_string());
            (url, sql)
        },
        None => {
            let url = std::env::var(URL_ENV).unwrap_or_else(|_| DEFAULT_URL.to_string());
            (url, DEFAULT_SQL.to_string())
        },
    }
}

fn join_sql(args: &[String]) -> Option<String> {
    let sql = args.join(" ");
    (!sql.trim().is_empty()).then_some(sql)
}

async fn connect_with_retry(url: &str) -> QueryResult<FlightSqlServiceClient<tonic::transport::Channel>> {
    let mut last_error = None;
    for _ in 0..600 {
        match Endpoint::from_shared(url.to_string())?.connect().await {
            Ok(channel) => return Ok(FlightSqlServiceClient::new(channel)),
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(Duration::from_millis(100)).await;
            },
        }
    }
    Err(format!("Flight SQL server did not accept connections at {url}: {last_error:?}").into())
}

async fn execute_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> QueryResult<Vec<RecordBatch>> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    let mut batches = Vec::new();
    for endpoint in flight_info.endpoint {
        let Some(ticket) = endpoint.ticket else {
            continue;
        };
        let mut stream = client.do_get(ticket).await?;
        while let Some(batch) = stream.try_next().await? {
            batches.push(batch);
        }
    }
    Ok(batches)
}

fn print_batches(batches: &[RecordBatch]) {
    let Some(first) = batches.first() else {
        println!("ok");
        return;
    };
    let schema = first.schema();
    let fields = schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    println!("{}", fields.join("\t"));
    for batch in batches {
        for row in 0..batch.num_rows() {
            let values = batch.columns().iter().map(|array| value_to_string(array, row)).collect::<Vec<_>>();
            println!("{}", values.join("\t"));
        }
    }
}

fn value_to_string(array: &ArrayRef, row: usize) -> String {
    if array.is_null(row) {
        return "NULL".to_string();
    }
    downcast_value!(array, row, BooleanArray, bool);
    downcast_value!(array, row, Int8Array, i8);
    downcast_value!(array, row, Int16Array, i16);
    downcast_value!(array, row, Int32Array, i32);
    downcast_value!(array, row, Int64Array, i64);
    downcast_value!(array, row, UInt8Array, u8);
    downcast_value!(array, row, UInt16Array, u16);
    downcast_value!(array, row, UInt32Array, u32);
    downcast_value!(array, row, UInt64Array, u64);
    downcast_value!(array, row, Float32Array, f32);
    downcast_value!(array, row, Float64Array, f64);
    downcast_value!(array, row, StringArray, str);
    downcast_value!(array, row, LargeStringArray, str);
    format!("<unsupported:{}>", data_type_name(array.data_type()))
}

fn data_type_name(data_type: &DataType) -> &'static str {
    match data_type {
        DataType::Null => "Null",
        DataType::Boolean => "Boolean",
        DataType::Int8 => "Int8",
        DataType::Int16 => "Int16",
        DataType::Int32 => "Int32",
        DataType::Int64 => "Int64",
        DataType::UInt8 => "UInt8",
        DataType::UInt16 => "UInt16",
        DataType::UInt32 => "UInt32",
        DataType::UInt64 => "UInt64",
        DataType::Float32 => "Float32",
        DataType::Float64 => "Float64",
        DataType::Utf8 => "Utf8",
        DataType::LargeUtf8 => "LargeUtf8",
        _ => "other",
    }
}
