use std::{error::Error as StdError, fmt};

/// Result returned by this leaf crate.
pub type IcebergResult<T> = Result<T, IcebergError>;

/// Machine-readable Iceberg failure code.
///
/// The code describes the response category, never the vendor crate that
/// surfaced the condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergErrorCode {
    Authentication,
    Access,
    NotFound,
    TemporaryDependency,
    InvalidMetadata,
    UnsupportedSchema,
    Timeout,
    Cancelled,
    Internal,
}

impl IcebergErrorCode {
    /// Stable adapter-facing code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::Access => "access",
            Self::NotFound => "not_found",
            Self::TemporaryDependency => "temporary_dependency",
            Self::InvalidMetadata => "invalid_metadata",
            Self::UnsupportedSchema => "unsupported_schema",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::Internal => "internal",
        }
    }
}

/// Retry status normalized from Iceberg and vendor failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergErrorStatus {
    Permanent,
    Temporary,
    Cancelled,
}

/// The leaf operation that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergOperation {
    CatalogConnect,
    CatalogListNamespaces,
    CatalogListTables,
    TableLoad,
    TableResolve,
    ManifestLoad,
    ManifestInterpretation,
    SchemaInterpretation,
    ScanPlanning,
    ScanCandidateBuild,
    SourceAccess,
}

impl IcebergOperation {
    /// Stable operation code for logs and adapter conversions.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CatalogConnect => "catalog_connect",
            Self::CatalogListNamespaces => "catalog_list_namespaces",
            Self::CatalogListTables => "catalog_list_tables",
            Self::TableLoad => "table_load",
            Self::TableResolve => "table_resolve",
            Self::ManifestLoad => "manifest_load",
            Self::ManifestInterpretation => "manifest_interpretation",
            Self::SchemaInterpretation => "schema_interpretation",
            Self::ScanPlanning => "scan_planning",
            Self::ScanCandidateBuild => "scan_candidate_build",
            Self::SourceAccess => "source_access",
        }
    }
}

/// Typed, credential-safe context carried with an error.
///
/// The leaf stores only stable identifiers and numeric Iceberg facts here.
/// URLs, authorization data, raw vendor context, and source chains are
/// intentionally absent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IcebergErrorContext {
    pub catalog_kind: Option<IcebergCatalogErrorKind>,
    pub catalog_handle: Option<String>,
    pub table_identifier: Option<String>,
    pub snapshot_id: Option<i64>,
    pub schema_id: Option<i32>,
    pub field_id: Option<i32>,
    pub file_count: Option<usize>,
}

/// Catalog backend kind as error context, kept independent of backend handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergCatalogErrorKind {
    Glue,
    Rest,
}

/// Normalized Iceberg-domain error.
///
/// Vendor sources are normalized at this boundary and deliberately discarded.
/// This avoids exposing an upstream HTTP error chain that might contain a
/// credential-bearing URL or authorization header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergError {
    pub code: IcebergErrorCode,
    pub status: IcebergErrorStatus,
    pub operation: IcebergOperation,
    pub sanitized_message: String,
    pub typed_context: Box<IcebergErrorContext>,
}

impl IcebergError {
    /// Creates a normalized leaf error.
    pub fn new(
        code: IcebergErrorCode,
        status: IcebergErrorStatus,
        operation: IcebergOperation,
        sanitized_message: impl AsRef<str>,
    ) -> Self {
        Self {
            code,
            status,
            operation,
            sanitized_message: sanitize_message(sanitized_message.as_ref()),
            typed_context: Box::default(),
        }
    }

    /// Adds typed, safe context.
    pub fn with_context(mut self, typed_context: IcebergErrorContext) -> Self {
        self.typed_context = Box::new(typed_context);
        self
    }

    #[cfg(feature = "catalog")]
    /// Normalizes an `iceberg-rust` error without retaining its source chain.
    pub fn from_iceberg_error(operation: IcebergOperation, source: iceberg::Error) -> Self {
        let code = classify_iceberg_error(&source);
        let status = status_for(code, source.retryable());
        Self::new(code, status, operation, source.message())
    }
}

impl fmt::Display for IcebergError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Iceberg {} ({}) during {}: {}",
            self.code.as_str(),
            match self.status {
                IcebergErrorStatus::Permanent => "permanent",
                IcebergErrorStatus::Temporary => "temporary",
                IcebergErrorStatus::Cancelled => "cancelled",
            },
            self.operation.as_str(),
            self.sanitized_message
        )
    }
}

impl StdError for IcebergError {}

/// Classifies a dependency message when its transport condition is more
/// specific than the upstream Iceberg error kind.
pub fn classify_error_message(message: &str, retryable: bool) -> IcebergErrorCode {
    let text = message.to_ascii_lowercase();
    if contains_any(&text, &["cancelled", "canceled", "request aborted", "operation aborted"]) {
        IcebergErrorCode::Cancelled
    } else if contains_any(&text, &["timeout", "timed out", "deadline exceeded"]) {
        IcebergErrorCode::Timeout
    } else if contains_any(
        &text,
        &[
            "unauthenticated",
            "authentication",
            "invalidaccesskey",
            "expiredtoken",
            "expiredtokenexception",
            "unrecognizedclient",
            "invalidclienttokenid",
            "missing credentials",
            "http_401",
            "http status 401",
            "status code: 401",
            " 401",
        ],
    ) {
        IcebergErrorCode::Authentication
    } else if contains_any(
        &text,
        &[
            "access denied",
            "accessdenied",
            "accessdeniedexception",
            "permission denied",
            "permissiondenied",
            "signaturedoesnotmatch",
            "http_403",
            "http status 403",
            "status code: 403",
            " 403",
        ],
    ) {
        IcebergErrorCode::Access
    } else if contains_any(&text, &["not found", "does not exist", "http_404", " 404"]) {
        IcebergErrorCode::NotFound
    } else if retryable {
        IcebergErrorCode::TemporaryDependency
    } else {
        IcebergErrorCode::Internal
    }
}

#[cfg(feature = "catalog")]
fn status_for(code: IcebergErrorCode, retryable: bool) -> IcebergErrorStatus {
    match code {
        IcebergErrorCode::Cancelled => IcebergErrorStatus::Cancelled,
        IcebergErrorCode::TemporaryDependency | IcebergErrorCode::Timeout => IcebergErrorStatus::Temporary,
        _ if retryable => IcebergErrorStatus::Temporary,
        _ => IcebergErrorStatus::Permanent,
    }
}

#[cfg(feature = "catalog")]
fn classify_iceberg_error(source: &iceberg::Error) -> IcebergErrorCode {
    use iceberg::ErrorKind;

    // Inspect vendor source messages only while the error is live. They can
    // carry the HTTP/auth condition while Iceberg's outer message is generic;
    // no source text is retained by the normalized leaf error.
    let mut current: Option<&(dyn StdError + 'static)> = Some(source);
    while let Some(error) = current {
        let message_code = classify_error_message(&error.to_string(), false);
        if message_code != IcebergErrorCode::Internal {
            return message_code;
        }
        current = error.source();
    }
    match source.kind() {
        ErrorKind::NamespaceNotFound | ErrorKind::TableNotFound => IcebergErrorCode::NotFound,
        ErrorKind::DataInvalid | ErrorKind::PreconditionFailed | ErrorKind::CatalogCommitConflicts => {
            IcebergErrorCode::InvalidMetadata
        },
        ErrorKind::FeatureUnsupported => IcebergErrorCode::UnsupportedSchema,
        ErrorKind::Unexpected if source.retryable() => IcebergErrorCode::TemporaryDependency,
        ErrorKind::Unexpected | ErrorKind::NamespaceAlreadyExists | ErrorKind::TableAlreadyExists => {
            IcebergErrorCode::Internal
        },
        _ => IcebergErrorCode::Internal,
    }
}

fn sanitize_message(message: &str) -> String {
    let message = redact_urls(message);
    let mut sanitized = Vec::new();
    let mut redact_remainder = false;
    for word in message.split_whitespace() {
        if redact_remainder {
            sanitized.push("<redacted>".to_string());
            continue;
        }
        let lower = word.to_ascii_lowercase();
        if contains_any(
            &lower,
            &["token=", "secret=", "password=", "access_key=", "credential=", "signature=", "x-amz-"],
        ) || credential_header_with_value(word)
        {
            sanitized.push("<redacted>".to_string());
        } else if credential_header_name(word) {
            // Error text sometimes renders HTTP headers as separate tokens,
            // e.g. `Authorization: Bearer <token>`. There is no reliable
            // message-level delimiter for the header value, so retain no
            // following text rather than risk emitting a credential.
            sanitized.push("<redacted>".to_string());
            redact_remainder = true;
        } else {
            sanitized.push(word.to_string());
        }
    }
    sanitized.join(" ")
}

fn credential_header_with_value(word: &str) -> bool {
    let normalized = normalized_credential_label(word);
    normalized.split_once(':').is_some_and(|(label, value)| !value.is_empty() && is_credential_header(label))
}

fn credential_header_name(word: &str) -> bool {
    let normalized = normalized_credential_label(word);
    normalized.strip_suffix(':').is_some_and(is_credential_header)
}

fn normalized_credential_label(word: &str) -> String {
    word.trim_matches(|character: char| {
        matches!(character, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '\"' | '\'')
    })
    .to_ascii_lowercase()
}

fn is_credential_header(label: &str) -> bool {
    matches!(label, "authorization" | "proxy-authorization" | "cookie" | "set-cookie")
}

/// Removes every URL rather than attempting to preserve individual components.
/// Vendor URL query strings can contain signed credentials under many names;
/// retaining no URL text is the only stable logging contract at this boundary.
fn redact_urls(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());
    let mut remainder = value;
    while let Some(scheme_marker) = remainder.find("://") {
        let scheme_start = remainder[..scheme_marker]
            .rfind(|character: char| !character.is_ascii_alphanumeric() && !matches!(character, '+' | '-' | '.'))
            .map_or(0, |index| index + 1);
        let url_end = remainder[scheme_marker + 3..]
            .find(|character: char| character.is_whitespace() || matches!(character, ')' | ']' | '}' | '"' | '\''))
            .map_or(remainder.len(), |index| scheme_marker + 3 + index);
        sanitized.push_str(&remainder[..scheme_start]);
        sanitized.push_str("<redacted-url>");
        remainder = &remainder[url_end..];
    }
    sanitized.push_str(remainder);
    sanitized
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_are_sanitized_and_do_not_keep_vendor_sources() {
        let error = IcebergError::new(
            IcebergErrorCode::Authentication,
            IcebergErrorStatus::Permanent,
            IcebergOperation::CatalogConnect,
            "request https://user:password@catalog.example/v1 token=not-for-logs",
        );

        let rendered = format!("{error:?}");
        assert!(!rendered.contains("password"));
        assert!(!rendered.contains("not-for-logs"));
        assert!(StdError::source(&error).is_none());
    }

    #[test]
    fn errors_redact_every_signed_url_and_sensitive_query_parameter() {
        let error = IcebergError::new(
            IcebergErrorCode::Authentication,
            IcebergErrorStatus::Permanent,
            IcebergOperation::CatalogConnect,
            "request https://catalog.example/v1?X-Amz-Credential=credential-secret&X-Amz-Signature=signature-secret then https://user:password@store.example/file?token=token-secret X-Amz-Security-Token=session-secret",
        );

        assert!(!error.sanitized_message.contains("catalog.example"));
        assert!(!error.sanitized_message.contains("store.example"));
        assert!(!error.sanitized_message.contains("credential-secret"));
        assert!(!error.sanitized_message.contains("signature-secret"));
        assert!(!error.sanitized_message.contains("token-secret"));
        assert!(!error.sanitized_message.contains("session-secret"));
        assert_eq!(error.sanitized_message.matches("<redacted-url>").count(), 2);
    }

    #[test]
    fn errors_redact_split_http_credential_headers() {
        let error = IcebergError::new(
            IcebergErrorCode::Authentication,
            IcebergErrorStatus::Permanent,
            IcebergOperation::CatalogConnect,
            "request failed Authorization: Bearer not-for-logs Cookie: session=also-not-for-logs",
        );

        assert!(!error.sanitized_message.contains("not-for-logs"));
        assert!(!error.sanitized_message.contains("also-not-for-logs"));
    }

    #[test]
    fn error_message_classification_preserves_auth_access_and_transient_cases() {
        assert_eq!(classify_error_message("service returned AccessDeniedException", false), IcebergErrorCode::Access);
        assert_eq!(
            classify_error_message("ExpiredTokenException from catalog", false),
            IcebergErrorCode::Authentication
        );
        assert_eq!(classify_error_message("catalog request timed out", false), IcebergErrorCode::Timeout);
        assert_eq!(classify_error_message("dependency unavailable", true), IcebergErrorCode::TemporaryDependency);
    }

    #[cfg(feature = "catalog")]
    #[test]
    fn iceberg_source_chain_classifies_auth_and_access_without_retaining_it() {
        use iceberg::{Error, ErrorKind};

        let auth = Error::new(ErrorKind::Unexpected, "catalog request failed")
            .with_source(std::io::Error::other("ExpiredTokenException: token=not-for-logs"));
        let access = Error::new(ErrorKind::Unexpected, "catalog request failed")
            .with_source(std::io::Error::other("PermissionDenied: forbidden"));

        let auth = IcebergError::from_iceberg_error(IcebergOperation::CatalogConnect, auth);
        let access = IcebergError::from_iceberg_error(IcebergOperation::CatalogConnect, access);

        assert_eq!(auth.code, IcebergErrorCode::Authentication);
        assert_eq!(access.code, IcebergErrorCode::Access);
        assert!(!auth.sanitized_message.contains("not-for-logs"));
        assert!(StdError::source(&auth).is_none());
        assert!(StdError::source(&access).is_none());
    }
}
