use std::{env, error::Error as StdError, io::IsTerminal, process::ExitCode};

use datafusion_nexus::server::{NEXUS_SERVER_LOG_ENV, ServerConfig, ServerError, serve};

fn main() -> ExitCode {
    serve_main()
}

#[tokio::main]
async fn serve_main() -> ExitCode {
    if let Err(error) = init_tracing() {
        eprintln!("ERROR datafusion-nexus Flight SQL server failed to initialize logging: {error}");
        return ExitCode::FAILURE;
    }
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            log_server_error(&error);
            ExitCode::FAILURE
        },
    }
}

fn log_server_error(error: &ServerError) {
    if let Some(source) = StdError::source(error) {
        let source_chain = source_chain_log_message(source);
        let root_cause = root_cause_log_message(source);
        if source_chain == root_cause {
            tracing::error!(
                error_code = error.code().as_str(),
                error_kind = error.kind().as_str(),
                error_status = error.status().as_str(),
                error_operation = error.operation(),
                error_message = error.message(),
                error_source_chain = %source_chain,
                "datafusion-nexus Flight SQL server failed"
            );
            return;
        }
        tracing::error!(
            error_code = error.code().as_str(),
            error_kind = error.kind().as_str(),
            error_status = error.status().as_str(),
            error_operation = error.operation(),
            error_message = error.message(),
            error_source_chain = %source_chain,
            error_root_cause = %root_cause,
            "datafusion-nexus Flight SQL server failed"
        );
    } else {
        tracing::error!(
            error_code = error.code().as_str(),
            error_kind = error.kind().as_str(),
            error_status = error.status().as_str(),
            error_operation = error.operation(),
            error_message = error.message(),
            "datafusion-nexus Flight SQL server failed"
        );
    }
}

fn source_chain_log_message(source: &(dyn StdError + 'static)) -> String {
    let mut messages = Vec::new();
    let mut current = Some(source);
    while let Some(error) = current {
        let message = error_log_message(error);
        if messages.last() != Some(&message) {
            messages.push(message);
        }
        current = error.source();
    }
    messages.join(": ")
}

fn root_cause_log_message(source: &(dyn StdError + 'static)) -> String {
    let mut root = source;
    let mut current = source.source();
    while let Some(error) = current {
        root = error;
        current = error.source();
    }
    error_log_message(root)
}

fn error_log_message(error: &(dyn StdError + 'static)) -> String {
    if let Some(adapter_error) = error.downcast_ref::<datafusion_nexus::error::Error>() {
        return adapter_error.message().to_string();
    }
    if let Some(native_error) = error.downcast_ref::<nexus_query_engine::error::QueryError>() {
        return native_error.message().to_string();
    }
    if let Some(datafusion_error) = error.downcast_ref::<datafusion::common::DataFusionError>() {
        if let Some(project_error) = find_project_error(datafusion_error) {
            return project_error;
        }
        return datafusion_error.message().to_string();
    }
    error.to_string()
}

fn find_project_error(error: &datafusion::common::DataFusionError) -> Option<String> {
    let mut current: Option<&(dyn StdError + 'static)> = Some(error);
    while let Some(error) = current {
        if let Some(adapter_error) = error.downcast_ref::<datafusion_nexus::error::Error>() {
            return Some(adapter_error.message().to_string());
        }
        if let Some(native_error) = error.downcast_ref::<nexus_query_engine::error::QueryError>() {
            return Some(native_error.message().to_string());
        }
        current = error.source();
    }
    None
}

fn init_tracing() -> Result<(), Box<dyn StdError + Send + Sync>> {
    let configured_log_filter = env::var(NEXUS_SERVER_LOG_ENV).unwrap_or_else(|_| "info".to_string());
    let log_filter = format!("{configured_log_filter},datafusion_nexus_server=error");
    tracing_subscriber::fmt().with_env_filter(log_filter).with_ansi(std::io::stderr().is_terminal()).try_init()?;
    Ok(())
}

async fn run() -> Result<(), ServerError> {
    let config = ServerConfig::from_env()?;
    serve(config).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use datafusion_nexus::error::{Error, ErrorCode};

    use super::*;

    #[test]
    fn source_chain_log_message_uses_adapter_human_message_without_structured_fields() {
        let source = Error::new(
            ErrorCode::ServerTransportFailure,
            "datafusion_nexus.server.transport",
            "Flight SQL transport failed",
        );

        let message = source_chain_log_message(&source);

        assert_eq!(message, "Flight SQL transport failed");
        assert_no_diagnostic_fields(&message);
    }

    #[test]
    fn source_chain_log_message_uses_native_human_message_without_structured_fields() {
        let source = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.admit",
            "native query service queue is full",
        );

        let message = source_chain_log_message(&source);

        assert_eq!(message, "native query service queue is full");
        assert_no_diagnostic_fields(&message);
    }

    #[test]
    fn source_chain_log_message_preserves_nested_root_cause() {
        let source = TestError::with_source("transport error", TestError::leaf("Address already in use (os error 98)"));

        assert_eq!(source_chain_log_message(&source), "transport error: Address already in use (os error 98)");
        assert_eq!(root_cause_log_message(&source), "Address already in use (os error 98)");
    }

    #[test]
    fn source_chain_log_message_strips_datafusion_external_project_error_display() {
        let source = datafusion::common::DataFusionError::External(Box::new(Error::new(
            ErrorCode::ServerTransportFailure,
            "datafusion_nexus.server.transport",
            "Flight SQL transport failed",
        )));

        let message = source_chain_log_message(&source);

        assert_eq!(message, "Flight SQL transport failed");
        assert_no_diagnostic_fields(&message);
    }

    #[test]
    fn source_chain_log_message_keeps_datafusion_context_without_child_structured_display() {
        let source = datafusion::common::DataFusionError::Context(
            "server wrapper context".to_string(),
            Box::new(datafusion::common::DataFusionError::External(Box::new(Error::new(
                ErrorCode::ServerTransportFailure,
                "datafusion_nexus.server.transport",
                "Flight SQL transport failed",
            )))),
        );

        let message = source_chain_log_message(&source);

        assert_eq!(message, "Flight SQL transport failed");
        assert_no_diagnostic_fields(&message);
    }

    fn assert_no_diagnostic_fields(message: &str) {
        for needle in ["code=", "kind=", "status=", "operation=", "facts=", "source="] {
            assert!(!message.contains(needle), "source message leaked diagnostic field {needle}: {message}");
        }
    }

    #[derive(Debug)]
    struct TestError {
        message: &'static str,
        source: Option<Box<TestError>>,
    }

    impl TestError {
        fn leaf(message: &'static str) -> Self {
            Self { message, source: None }
        }

        fn with_source(message: &'static str, source: TestError) -> Self {
            Self { message, source: Some(Box::new(source)) }
        }
    }

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.message)
        }
    }

    impl StdError for TestError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
        }
    }
}
