//! Standalone Arrow Flight SQL server binary for `algeon-datafusion`.

#[cfg(feature = "iceberg")]
use std::sync::Arc;
use std::{
    env,
    error::Error as StdError,
    io::{BufRead, IsTerminal, Write},
    process::ExitCode,
};

#[cfg(feature = "iceberg")]
use algeon_datafusion::server::{backend_from_server_config, build_session_context_with_backend};
use algeon_datafusion::{
    backend::GpuMemoryOwnership,
    build_info::build_info,
    error::ErrorLogDiagnostics,
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_LOG_ENV, FlightSqlServer, LoadedServerConfig, ServerError,
        hash_password,
    },
};

const USAGE: &str = "\
Usage:
  algeon_server [--config <path>] [--log <filter>] [--check-config]
  algeon_server --version
  algeon_server --help
  algeon_server hash-password

Options:
  --config <path>   TOML server configuration (overrides ALGEON_SERVER_CONFIG_FILE)
  --log <filter>    tracing filter such as info or debug (overrides ALGEON_SERVER_LOG)
  --check-config    load and validate the configuration, including the TLS files, without
                    touching the GPU, then print the effective configuration as TOML with
                    the source of each value
  --version         print the version, commit, features and DataFusion version
  --help            print this help

Commands:
  hash-password     read a password from stdin and print its Argon2 PHC string for
                    [[auth.mode.users]] password_hash

Environment:
  ALGEON_SERVER_CONFIG_FILE   same as --config
  ALGEON_SERVER_LOG           same as --log (default: info)
  Every other setting has a TOML path; its override is ALGEON_SERVER_ followed by the
  path in upper case with `.` written as `_`, for example native.execution_mode is
  ALGEON_SERVER_NATIVE_EXECUTION_MODE.

Reference: https://algeon.dev/integration/configuration
";

#[derive(Debug, Default, PartialEq, Eq)]
struct ServeArgs {
    config: Option<String>,
    log: Option<String>,
    check_config: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Serve(ServeArgs),
    Version,
    Help,
    HashPassword,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let mut serve = ServeArgs::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(Command::Help),
            "--version" | "-V" => return Ok(Command::Version),
            "hash-password" if serve == ServeArgs::default() => {
                return match args.next() {
                    None => Ok(Command::HashPassword),
                    Some(extra) => Err(format!("hash-password takes no arguments, got `{extra}`")),
                };
            },
            "--check-config" => serve.check_config = true,
            "--config" | "--log" => {
                let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
                let slot = if arg == "--config" { &mut serve.config } else { &mut serve.log };
                *slot = Some(value);
            },
            other => match other.split_once('=') {
                Some(("--config", value)) => serve.config = Some(value.to_string()),
                Some(("--log", value)) => serve.log = Some(value.to_string()),
                _ => return Err(format!("unknown argument `{other}`")),
            },
        }
    }
    Ok(Command::Serve(serve))
}

fn main() -> ExitCode {
    let command = match parse_args(env::args().skip(1)) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("algeon_server: {error}\n\n{USAGE}");
            return ExitCode::from(2);
        },
    };
    match command {
        Command::Help => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        },
        Command::Version => {
            print_version();
            ExitCode::SUCCESS
        },
        Command::HashPassword => hash_password_main(),
        Command::Serve(args) => serve_main(args),
    }
}

/// Optional features of this binary beyond the library's engine features.
const SERVER_FEATURES: &[&str] = &[
    #[cfg(feature = "iceberg")]
    "iceberg",
];

fn print_version() {
    let build = build_info();
    let mut features = build.features.iter().chain(SERVER_FEATURES).copied().collect::<Vec<_>>();
    features.sort_unstable();
    let features = if features.is_empty() { "none".to_string() } else { features.join(",") };
    println!("algeon_server {}", build.version);
    println!("commit: {}", build.git_commit.unwrap_or("unknown"));
    println!("features: {features}");
    println!("datafusion: {}", build.datafusion_version);
}

fn hash_password_main() -> ExitCode {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        // The standard library cannot turn off terminal echo, so the
        // password is visible while it is typed; pipe it to avoid that.
        eprint!("Password (shown as typed; pipe it on stdin to hide it): ");
        let _ = std::io::stderr().flush();
    }
    let mut line = String::new();
    if let Err(error) = stdin.lock().read_line(&mut line) {
        eprintln!("algeon_server hash-password: failed to read stdin: {error}");
        return ExitCode::FAILURE;
    }
    let password = line.strip_suffix('\n').map(|line| line.strip_suffix('\r').unwrap_or(line)).unwrap_or(&line);
    if password.is_empty() {
        eprintln!("algeon_server hash-password: the password is empty");
        return ExitCode::FAILURE;
    }
    match hash_password(password) {
        Ok(hash) => {
            println!("{hash}");
            ExitCode::SUCCESS
        },
        Err(error) => {
            eprintln!("algeon_server hash-password: {error}");
            ExitCode::FAILURE
        },
    }
}

#[tokio::main]
async fn serve_main(args: ServeArgs) -> ExitCode {
    // `--check-config` prints TOML on stdout, so its log lines go to stderr.
    if let Err(error) = init_tracing(args.log.as_deref(), args.check_config) {
        eprintln!("ERROR Algeon Flight SQL server failed to initialize logging: {error}");
        return ExitCode::FAILURE;
    }
    let failure = if args.check_config { CHECK_CONFIG_FAILED } else { SERVER_FAILED };
    let loaded = match load_config(args.config) {
        Ok(loaded) => loaded,
        Err(error) => {
            log_server_error(&error, failure);
            return ExitCode::FAILURE;
        },
    };
    if args.check_config {
        if let Err(error) = loaded.config.check_tls_material() {
            log_server_error(&error, failure);
            return ExitCode::FAILURE;
        }
        print!("{}", loaded.effective_toml());
        return ExitCode::SUCCESS;
    }
    match serve(loaded).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            log_server_error(&error, failure);
            ExitCode::FAILURE
        },
    }
}

fn load_config(config_flag: Option<String>) -> Result<LoadedServerConfig, ServerError> {
    let mut vars = Vec::new();
    for (name, value) in env::vars_os() {
        let Some(name) = name.to_str().filter(|name| name.starts_with("ALGEON_SERVER_")) else {
            continue;
        };
        if config_flag.is_some() && name == ALGEON_SERVER_CONFIG_FILE_ENV {
            continue;
        }
        let value = value
            .into_string()
            .map_err(|_| ServerError::configuration(format!("environment variable {name} must be valid UTF-8")))?;
        vars.push((name.to_string(), value));
    }
    if let Some(path) = config_flag {
        vars.push((ALGEON_SERVER_CONFIG_FILE_ENV.to_string(), path));
    }
    LoadedServerConfig::from_vars(vars)
}

async fn serve(mut loaded: LoadedServerConfig) -> Result<(), ServerError> {
    // The standalone server owns every GPU it selects.
    let ownership = loaded
        .config
        .gpu_devices()
        .into_iter()
        .map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive))
        .collect::<Vec<_>>();
    loaded.config = loaded.config.with_gpu_memory_ownership(ownership);
    let builder = FlightSqlServer::builder();
    #[cfg(feature = "iceberg")]
    let builder = {
        // Iceberg catalogs register on the shared session, which each
        // statement session clones, so the launcher builds that session.
        let backend = backend_from_server_config(&loaded.config)?;
        let ctx = build_session_context_with_backend(&loaded.config, &backend).await?;
        algeon_iceberg::register_iceberg_catalogs(&ctx, &loaded.config)
            .await
            .map_err(|error| ServerError::session("failed to register the Iceberg catalogs", error))?;
        builder.backend(backend).session_context(Arc::new(ctx))
    };
    #[cfg(not(feature = "iceberg"))]
    if !loaded.config.iceberg_catalogs.is_empty() {
        return Err(ServerError::invalid_config(
            "iceberg.catalogs",
            "needs a server built with the `iceberg` feature; this binary was built without it",
        ));
    }
    builder.loaded_config(loaded).build().await?.serve().await
}

/// Configuration error fields emitted as dedicated tracing attributes rather
/// than concatenated into `error_facts`.
const DEDICATED_FACTS: [&str; 3] = ["config_field", "config_env", "reason"];

const SERVER_FAILED: &str = "Algeon Flight SQL server failed";
const CHECK_CONFIG_FAILED: &str = "Algeon server configuration check failed";

fn log_server_error(error: &ServerError, message: &'static str) {
    let diagnostics = ErrorLogDiagnostics::from_error(error.as_adapter_error());
    let other_facts = error
        .facts()
        .iter()
        .filter(|(key, _)| !DEDICATED_FACTS.contains(key))
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    tracing::error!(
        target: "algeon_server_bin",
        error_code = error.code().as_str(),
        error_kind = error.kind().as_str(),
        error_status = error.status().as_str(),
        error_operation = error.operation(),
        error_message = error.message(),
        config_field = error.fact_value("config_field"),
        config_env = error.fact_value("config_env"),
        reason = error.fact_value("reason"),
        error_facts = (!other_facts.is_empty()).then_some(other_facts.as_str()),
        error_cause = diagnostics.cause().filter(|cause| !repeats_message(error.message(), cause)),
        error_source_path = diagnostics.source_path(),
        error_diagnostics_truncated = diagnostics.truncated().then_some(true),
        message
    );
}

/// Returns true when `cause` repeats text in `message`. Because log
/// diagnostics rewrite line breaks in the cause, the comparison ignores
/// non-alphanumeric characters.
fn repeats_message(message: &str, cause: &str) -> bool {
    let squash = |text: &str| text.chars().filter(|ch| ch.is_alphanumeric()).collect::<String>();
    squash(message).contains(&squash(cause))
}

fn init_tracing(log_flag: Option<&str>, to_stderr: bool) -> Result<(), Box<dyn StdError + Send + Sync>> {
    let configured_log_filter = log_flag
        .map(str::to_string)
        .or_else(|| env::var(ALGEON_SERVER_LOG_ENV).ok())
        .unwrap_or_else(|| "info".to_string());
    let log_filter = format!("{configured_log_filter},algeon_server_bin=error");
    let builder = tracing_subscriber::fmt().with_env_filter(log_filter);
    if to_stderr {
        builder.with_writer(std::io::stderr).with_ansi(std::io::stderr().is_terminal()).try_init()?;
    } else {
        builder.with_ansi(std::io::stdout().is_terminal()).try_init()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Command, String> {
        parse_args(args.iter().map(|arg| arg.to_string()))
    }

    #[test]
    fn flags_select_the_command_and_unknown_arguments_are_rejected() {
        assert_eq!(parse(&[]), Ok(Command::Serve(ServeArgs::default())));
        assert_eq!(
            parse(&["--config", "/etc/algeon/server.toml", "--log=debug", "--check-config"]),
            Ok(Command::Serve(ServeArgs {
                config: Some("/etc/algeon/server.toml".to_string()),
                log: Some("debug".to_string()),
                check_config: true,
            }))
        );
        assert_eq!(parse(&["--version"]), Ok(Command::Version));
        assert_eq!(parse(&["hash-password"]), Ok(Command::HashPassword));
        assert!(parse(&["--config"]).is_err());
        assert!(parse(&["--confg", "x"]).is_err());
        assert!(parse(&["serve"]).is_err());
        assert!(parse(&["hash-password", "extra"]).is_err());
    }

    #[test]
    fn a_cause_that_repeats_the_message_with_rewritten_line_breaks_is_not_logged_again() {
        let message = "failed to parse `x.toml`: TOML parse error at line 2, column 1\n  |\n2 | a = 1\n";
        assert!(repeats_message(message, "TOML parse error at line 2, column 1 | 2 | a = 1"));
        assert!(!repeats_message(
            "failed to bind Flight SQL listener at 127.0.0.1:1",
            "Address already in use (os error 98)"
        ));
    }
}
