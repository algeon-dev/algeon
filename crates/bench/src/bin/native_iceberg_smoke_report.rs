use std::io;

pub(crate) use bench::iceberg_smoke::{IcebergNativeSmokeConfig, run_report, run_report_streaming, stable_tsv_field};

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
