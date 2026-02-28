use std::io;

#[allow(unused_imports)]
pub(crate) use datafusion_nexus_bench::iceberg_smoke::*;

#[tokio::main]
async fn main() {
    let config = match IcebergNativeSmokeConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("[native_iceberg_smoke_report] error={}", stable_tsv_field(&error.to_string()));
            std::process::exit(1);
        },
    };
    if config.stream_tsv_rows {
        let stdout = io::stdout();
        let mut stdout = stdout.lock();
        match run_report_streaming(config, &mut stdout).await {
            Ok(_report) => {},
            Err(error) => {
                eprintln!("[native_iceberg_smoke_report] error={}", stable_tsv_field(&error.to_string()));
                std::process::exit(1);
            },
        }
    } else {
        match run_report(config).await {
            Ok(report) => print!("{}", report.to_tsv()),
            Err(error) => {
                eprintln!("[native_iceberg_smoke_report] error={}", stable_tsv_field(&error.to_string()));
                std::process::exit(1);
            },
        }
    };
}
