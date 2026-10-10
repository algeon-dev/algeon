//! Plaintext Flight SQL smoke-test client behind `tools flight-sql-query`.
//!
//! It runs one statement and prints tab-separated rows. It has no
//! authentication or TLS; applications connect through an ADBC driver.

use std::{error::Error, process, time::Duration};

use arrow::{
    error::ArrowError,
    record_batch::RecordBatch,
    util::display::{ArrayFormatter, FormatOptions},
};
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use tonic::transport::Channel;

use crate::connection_retry::{ConnectionRetryConfig, connect_with_retry_reporting};

type QueryResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const DEFAULT_URL: &str = "http://127.0.0.1:50051";
const DEFAULT_SQL: &str = "SELECT 1";
const URL_ENV: &str = "ALGEON_FLIGHT_SQL_URL";
const CONNECT_TIMEOUT_FLAG: &str = "--connect-timeout-secs";
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 5;
const CONNECT_BACKOFF: Duration = Duration::from_millis(100);

struct QueryArgs {
    /// URL as the user wrote it, for messages.
    url: String,
    /// `url` with `grpc://` mapped to the `http://` scheme tonic accepts.
    endpoint: String,
    sql: String,
    connect_timeout_secs: u64,
}

pub async fn run(args: Vec<String>) -> QueryResult<()> {
    // `main` reports a returned error with `Debug`; print `Display` text and
    // exit instead so connection and server errors stay readable.
    let (code, message) = match parse_args(args) {
        Err(message) => (2, message),
        Ok(args) => match query(&args).await {
            Ok(()) => return Ok(()),
            Err(message) => (1, message),
        },
    };
    eprintln!("flight-sql-query: {message}");
    process::exit(code);
}

async fn query(args: &QueryArgs) -> Result<(), String> {
    let mut client = connect_flight_sql(args).await?;
    let batches = execute_batches(&mut client, &args.sql).await.map_err(|error| display_chain(error.as_ref()))?;
    print_batches(&batches).map_err(|error| display_chain(&error))
}

fn parse_args(mut args: Vec<String>) -> Result<QueryArgs, String> {
    let mut connect_timeout_secs = DEFAULT_CONNECT_TIMEOUT_SECS;
    if args.first().map(String::as_str) == Some(CONNECT_TIMEOUT_FLAG) {
        let value = args.get(1).ok_or_else(|| format!("{CONNECT_TIMEOUT_FLAG} needs a value in seconds"))?;
        connect_timeout_secs = value
            .parse::<u64>()
            .ok()
            .filter(|secs| *secs > 0)
            .ok_or_else(|| format!("{CONNECT_TIMEOUT_FLAG} needs a positive integer, got '{value}'"))?;
        args.drain(..2);
    }
    let (url, sql_args) = match args.split_first() {
        Some((first, rest)) if first.contains("://") => (first.clone(), rest),
        _ => (std::env::var(URL_ENV).unwrap_or_else(|_| DEFAULT_URL.to_string()), args.as_slice()),
    };
    let endpoint = plaintext_endpoint(&url)?;
    let sql = join_sql(sql_args).unwrap_or_else(|| DEFAULT_SQL.to_string());
    Ok(QueryArgs { url, endpoint, sql, connect_timeout_secs })
}

fn plaintext_endpoint(url: &str) -> Result<String, String> {
    if let Some(rest) = url.strip_prefix("grpc://") {
        return Ok(format!("http://{rest}"));
    }
    if url.starts_with("http://") {
        return Ok(url.to_string());
    }
    if url.starts_with("grpc+tls://") || url.starts_with("https://") {
        return Err(format!(
            "{url} requires TLS, but this smoke-test tool connects over plaintext only; use an ADBC client with a \
             grpc+tls:// URI for a TLS server"
        ));
    }
    Err(format!("unsupported URL scheme in {url}; use grpc:// or http://"))
}

fn join_sql(args: &[String]) -> Option<String> {
    let sql = args.join(" ");
    (!sql.trim().is_empty()).then_some(sql)
}

async fn connect_flight_sql(args: &QueryArgs) -> Result<FlightSqlServiceClient<Channel>, String> {
    let url = &args.url;
    let budget_secs = args.connect_timeout_secs;
    let mut reported_secs = 0;
    connect_with_retry_reporting(
        &args.endpoint,
        ConnectionRetryConfig::time_budget(Duration::from_secs(budget_secs), CONNECT_BACKOFF),
        FlightSqlServiceClient::new,
        |error| {
            format!(
                "no Flight SQL server accepted connections at {url} within {budget_secs}s: {}",
                display_chain(error)
            )
        },
        |elapsed| {
            let secs = elapsed.as_secs();
            if secs > reported_secs {
                reported_secs = secs;
                eprintln!("waiting for Flight SQL server at {url} ({secs}s/{budget_secs}s)");
            }
        },
    )
    .await
    .map_err(|error| error.to_string())
}

async fn execute_batches(client: &mut FlightSqlServiceClient<Channel>, sql: &str) -> QueryResult<Vec<RecordBatch>> {
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

fn print_batches(batches: &[RecordBatch]) -> Result<(), ArrowError> {
    let Some(first) = batches.first() else {
        println!("ok");
        return Ok(());
    };
    let schema = first.schema();
    let fields = schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    println!("{}", fields.join("\t"));
    let options = FormatOptions::default().with_null("NULL");
    for batch in batches {
        let formatters = batch
            .columns()
            .iter()
            .map(|array| ArrayFormatter::try_new(array.as_ref(), &options))
            .collect::<Result<Vec<_>, _>>()?;
        for row in 0..batch.num_rows() {
            let values = formatters.iter().map(|formatter| formatter.value(row).to_string()).collect::<Vec<_>>();
            println!("{}", values.join("\t"));
        }
    }
    Ok(())
}

/// Renders an error and its sources with `Display`, joined by `: `. Hyper and
/// Arrow Flight wrap sources whose text repeats or extends their wrapper's
/// text; a source that extends the previous segment replaces it, and one
/// already shown is skipped.
fn display_chain(error: &(dyn Error + 'static)) -> String {
    let mut segments = vec![error.to_string()];
    let mut source = error.source();
    while let Some(cause) = source {
        let text = cause.to_string();
        let last = segments.len() - 1;
        if text.starts_with(&segments[last]) {
            segments[last] = text;
        } else if !segments.iter().any(|segment| segment.contains(&text)) {
            segments.push(text);
        }
        source = cause.source();
    }
    segments.join(": ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn grpc_scheme_maps_to_plaintext_http_and_tls_schemes_are_refused() {
        let parsed = parse_args(args(&["grpc://127.0.0.1:50051", "SELECT", "1"])).unwrap();
        assert_eq!(parsed.endpoint, "http://127.0.0.1:50051");
        assert_eq!(parsed.url, "grpc://127.0.0.1:50051");
        assert_eq!(parsed.sql, "SELECT 1");
        assert_eq!(parsed.connect_timeout_secs, DEFAULT_CONNECT_TIMEOUT_SECS);
        assert!(parse_args(args(&["grpc+tls://host:1"])).unwrap_err().contains("plaintext only"));
        assert!(parse_args(args(&["ftp://host:1"])).unwrap_err().contains("unsupported URL scheme"));
    }

    #[test]
    fn connect_timeout_precedes_the_url_and_must_be_a_positive_integer() {
        let parsed = parse_args(args(&["--connect-timeout-secs", "9", "http://h:1", "SELECT 2"])).unwrap();
        assert_eq!(parsed.connect_timeout_secs, 9);
        assert_eq!(parsed.sql, "SELECT 2");
        assert!(parse_args(args(&["--connect-timeout-secs", "0"])).unwrap_err().contains("positive integer"));
        assert!(parse_args(args(&["--connect-timeout-secs"])).unwrap_err().contains("needs a value"));
    }
}
