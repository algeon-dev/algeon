use aws_smithy_types::error::ErrorMetadata;

use super::*;

pub(super) const CATALOG_BACKEND_OP: &str = "datafusion_nexus.iceberg.catalog_backend";

fn catalog_backend_error_code(source: &iceberg::Error, aws_facts: Option<&AwsCatalogErrorFacts>) -> ErrorCode {
    if let Some(aws_facts) = aws_facts {
        if matches!(
            aws_facts.error_code,
            Some("ExpiredTokenException" | "InvalidAccessKeyId" | "UnrecognizedClientException")
        ) || matches!(aws_facts.detail, Some("http_401" | "missing_credentials"))
        {
            return ErrorCode::SourceUnauthenticated;
        }
        if matches!(aws_facts.error_code, Some("AccessDeniedException" | "SignatureDoesNotMatch"))
            || matches!(aws_facts.detail, Some("http_403" | "access_denied"))
        {
            return ErrorCode::SourceAccessDenied;
        }
        if matches!(aws_facts.error_code, Some("EntityNotFoundException")) {
            return ErrorCode::SourceObjectNotFound;
        }
    }
    match source.kind() {
        iceberg::ErrorKind::TableNotFound | iceberg::ErrorKind::NamespaceNotFound => ErrorCode::ObjectNotFound,
        _ if source.retryable() => ErrorCode::SourceTemporaryFailure,
        _ => ErrorCode::CudfDependencyFailure,
    }
}

pub(crate) fn classify_iceberg_scan_error(source: &iceberg::Error) -> ErrorCode {
    let aws_facts = AwsCatalogErrorFacts::from_error(source);
    let code = catalog_backend_error_code(source, aws_facts.as_ref());
    if code != ErrorCode::CudfDependencyFailure {
        return code;
    }
    find_source_error::<opendal::Error>(source)
        .map(classify_opendal_scan_error)
        .unwrap_or(ErrorCode::SourceDependencyFailure)
}

pub(crate) fn classify_opendal_scan_error(source: &opendal::Error) -> ErrorCode {
    if let Some(aws_facts) = AwsCatalogErrorFacts::from_error(source) {
        if matches!(
            aws_facts.error_code,
            Some("ExpiredTokenException" | "InvalidAccessKeyId" | "UnrecognizedClientException")
        ) || matches!(aws_facts.detail, Some("http_401" | "missing_credentials"))
        {
            return ErrorCode::SourceUnauthenticated;
        }
        if matches!(aws_facts.error_code, Some("AccessDeniedException" | "SignatureDoesNotMatch"))
            || matches!(aws_facts.detail, Some("http_403" | "access_denied"))
        {
            return ErrorCode::SourceAccessDenied;
        }
    }
    match source.kind() {
        opendal::ErrorKind::NotFound => ErrorCode::SourceObjectNotFound,
        opendal::ErrorKind::PermissionDenied => ErrorCode::SourceAccessDenied,
        opendal::ErrorKind::RateLimited => ErrorCode::SourceTemporaryFailure,
        _ if source.is_temporary() => ErrorCode::SourceTemporaryFailure,
        _ => ErrorCode::SourceDependencyFailure,
    }
}

pub(crate) fn apply_opendal_scan_error_facts(mut error: Error, source: &opendal::Error) -> Error {
    error = error
        .with_fact("opendal_error_kind", source.kind().into_static())
        .with_fact("opendal_temporary", source.is_temporary())
        .with_fact("opendal_persistent", source.is_persistent());
    if let Some(aws_facts) = AwsCatalogErrorFacts::from_error(source) {
        error = aws_facts.apply(error);
    }
    error
}

pub(crate) fn apply_iceberg_catalog_error_facts(mut error: Error, source: &iceberg::Error) -> Error {
    error = error
        .with_fact("iceberg_error_kind", source.kind().into_static())
        .with_fact("iceberg_retryable", source.retryable());
    let opendal = find_source_error::<opendal::Error>(source);
    if let Some(opendal) = opendal {
        error = apply_opendal_scan_error_facts(error, opendal);
    } else if let Some(aws_facts) = AwsCatalogErrorFacts::from_error(source) {
        error = aws_facts.apply(error);
    }
    error
}

fn find_source_error<'a, E>(mut source: &'a (dyn StdError + 'static)) -> Option<&'a E>
where E: StdError + 'static {
    for _ in 0..32 {
        if let Some(source) = source.downcast_ref::<E>() {
            return Some(source);
        }
        source = source.source()?;
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AwsCatalogErrorFacts {
    error_code: Option<&'static str>,
    request_id: Option<String>,
    http_status: Option<&'static str>,
    retryable: Option<&'static str>,
    detail: Option<&'static str>,
}

impl AwsCatalogErrorFacts {
    fn from_error(source: &(dyn StdError + 'static)) -> Option<Self> {
        let text = error_chain_text(source);
        let typed = typed_aws_facts(source);
        let error_code = typed
            .as_ref()
            .and_then(|metadata| metadata.code().and_then(canonical_aws_error_code))
            .or_else(|| aws_error_code_fact(&text));
        let detail = typed
            .as_ref()
            .and_then(|metadata| metadata.code().and_then(canonical_aws_error_detail))
            .or_else(|| aws_error_detail_fact(&text, error_code));
        let facts = Self {
            error_code,
            request_id: aws_request_id_fact(&text),
            http_status: aws_http_status_fact(&text),
            retryable: aws_retryable_fact(&text),
            detail,
        };
        (facts.error_code.is_some()
            || facts.request_id.is_some()
            || facts.http_status.is_some()
            || facts.retryable.is_some()
            || facts.detail.is_some())
        .then_some(facts)
    }

    fn apply(self, mut error: Error) -> Error {
        if let Some(error_code) = self.error_code {
            error = error.with_fact("aws_error_code", error_code);
        }
        if let Some(request_id) = self.request_id {
            error = error.with_fact("aws_request_id", request_id);
        }
        if let Some(http_status) = self.http_status {
            error = error.with_fact("aws_http_status", http_status);
        }
        if let Some(retryable) = self.retryable {
            error = error.with_fact("aws_retryable", retryable);
        }
        if let Some(detail) = self.detail {
            error = error.with_fact("aws_error_detail", detail);
        }
        error
    }
}

fn error_chain_text(source: &(dyn StdError + 'static)) -> String {
    let mut text = source.to_string();
    let mut current = source.source();
    while let Some(source) = current {
        text.push_str(": ");
        text.push_str(&source.to_string());
        current = source.source();
    }
    text
}

fn typed_aws_facts<'a>(source: &'a (dyn StdError + 'static)) -> Option<&'a ErrorMetadata> {
    let mut current: &(dyn StdError + 'static) = source;
    for _ in 0..32 {
        if let Some(metadata) = current.downcast_ref::<ErrorMetadata>() {
            return Some(metadata);
        }
        current = current.source()?;
    }
    None
}

fn canonical_aws_error_code(code: &str) -> Option<&'static str> {
    match code {
        "ExpiredTokenException" | "ExpiredToken" => Some("ExpiredTokenException"),
        "InvalidAccessKeyId" => Some("InvalidAccessKeyId"),
        "UnrecognizedClientException" => Some("UnrecognizedClientException"),
        "AccessDeniedException" | "AccessDenied" => Some("AccessDeniedException"),
        "SignatureDoesNotMatch" => Some("SignatureDoesNotMatch"),
        "EntityNotFoundException" => Some("EntityNotFoundException"),
        _ => None,
    }
}

fn canonical_aws_error_detail(code: &str) -> Option<&'static str> {
    match canonical_aws_error_code(code) {
        Some("ExpiredTokenException") => Some("expired_token"),
        Some("InvalidAccessKeyId") => Some("invalid_access_key"),
        Some("UnrecognizedClientException") => Some("invalid_client"),
        Some("AccessDeniedException") => Some("access_denied"),
        Some("SignatureDoesNotMatch") => Some("signature_mismatch"),
        Some("EntityNotFoundException") => Some("not_found"),
        _ => None,
    }
}

fn aws_error_code_fact(text: &str) -> Option<&'static str> {
    if contains_any(text, &["ExpiredTokenException", "ExpiredToken"]) {
        Some("ExpiredTokenException")
    } else if text.contains("InvalidAccessKeyId") {
        Some("InvalidAccessKeyId")
    } else if text.contains("UnrecognizedClientException") {
        Some("UnrecognizedClientException")
    } else if text.contains("AccessDeniedException") || text.contains("AccessDenied") {
        Some("AccessDeniedException")
    } else if text.contains("SignatureDoesNotMatch") {
        Some("SignatureDoesNotMatch")
    } else if text.contains("EntityNotFoundException") {
        Some("EntityNotFoundException")
    } else {
        None
    }
}

fn aws_error_detail_fact(text: &str, error_code: Option<&'static str>) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    match error_code {
        Some("ExpiredTokenException") => return Some("expired_token"),
        Some("InvalidAccessKeyId") => return Some("invalid_access_key"),
        Some("UnrecognizedClientException") => return Some("invalid_client"),
        Some("AccessDeniedException") => return Some("access_denied"),
        Some("SignatureDoesNotMatch") => return Some("signature_mismatch"),
        Some("EntityNotFoundException") => return Some("not_found"),
        _ => {},
    }
    if contains_any(&lower, &["401 unauthorized", "status code: 401", "status: 401", "http 401"]) {
        Some("http_401")
    } else if contains_any(&lower, &["403 forbidden", "status code: 403", "status: 403", "http 403"]) {
        Some("http_403")
    } else if contains_any(&lower, &["no credentials", "missing credentials", "could not load credentials"]) {
        Some("missing_credentials")
    } else {
        None
    }
}

fn aws_request_id_fact(text: &str) -> Option<String> {
    extract_quoted_marker(text, "aws_request_id\": \"")
        .or_else(|| extract_quoted_marker(text, "aws_request_id: \""))
        .or_else(|| extract_quoted_marker(text, "x-amz-request-id\": \""))
        .or_else(|| extract_value_after(text, "request id: "))
}

fn aws_http_status_fact(text: &str) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    if contains_any(&lower, &["status code: 401", "status: 401", "http 401", "401 unauthorized"]) {
        Some("401")
    } else if contains_any(&lower, &["status code: 403", "status: 403", "http 403", "403 forbidden"]) {
        Some("403")
    } else if contains_any(&lower, &["status code: 404", "status: 404", "http 404", "404 not found"]) {
        Some("404")
    } else {
        None
    }
}

fn aws_retryable_fact(text: &str) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("retryable: true") {
        Some("true")
    } else if lower.contains("retryable: false") {
        Some("false")
    } else {
        None
    }
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn extract_quoted_marker(text: &str, marker: &str) -> Option<String> {
    let start = text.find(marker)? + marker.len();
    let value = &text[start..];
    let end = value.find('"')?;
    let value = &value[..end];
    (!value.is_empty()).then(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_smithy_metadata_precedes_rendered_error_text() {
        let metadata = ErrorMetadata::builder().code("AccessDeniedException").build();
        let source = std::io::Error::new(std::io::ErrorKind::Other, metadata);
        let facts = AwsCatalogErrorFacts::from_error(&source).expect("typed metadata should be discovered");

        assert_eq!(facts.error_code, Some("AccessDeniedException"));
        assert_eq!(facts.detail, Some("access_denied"));
        assert_eq!(facts.http_status, None);
    }
}

fn extract_value_after(text: &str, marker: &str) -> Option<String> {
    let start = text.to_ascii_lowercase().find(marker)? + marker.len();
    let value = text[start..]
        .split(|ch: char| ch == ',' || ch == ')' || ch == '}' || ch.is_whitespace())
        .next()
        .unwrap_or("")
        .trim_matches('"');
    (!value.is_empty()).then(|| value.to_string())
}
