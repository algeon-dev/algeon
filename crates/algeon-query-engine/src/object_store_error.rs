//! Object-store error classification shared by native metadata reads, adapter
//! footer pruning, and table-format scans.
//!
//! The kind and detail strings are diagnostic fact values; classification
//! reads the typed variant first and falls back to the rendered source chain
//! only for `Generic` errors.

use std::error::Error as StdError;

use object_store::Error as ObjectStoreError;

/// Maximum source-chain layers rendered by [`error_chain_text`].
///
/// Credential and transport markers sit in shallow SDK layers; the cap keeps
/// an unbounded foreign chain from dictating classification work.
const MAX_ERROR_CHAIN_DEPTH: usize = 32;
/// Maximum bytes retained by [`error_chain_text`]. Markers are short SDK codes,
/// so the cap only truncates extreme foreign payloads.
const MAX_ERROR_CHAIN_BYTES: usize = 16_384;

/// Stable `object_store_error_kind` fact value for a typed object-store error.
pub fn object_store_error_kind(error: &ObjectStoreError) -> &'static str {
    match error {
        ObjectStoreError::PermissionDenied { .. } => "permission_denied",
        ObjectStoreError::Unauthenticated { .. } => "unauthenticated",
        ObjectStoreError::NotFound { .. } => "not_found",
        ObjectStoreError::Generic { .. } => "generic",
        ObjectStoreError::InvalidPath { .. } => "invalid_path",
        ObjectStoreError::JoinError { .. } => "join_error",
        ObjectStoreError::NotSupported { .. } => "not_supported",
        ObjectStoreError::AlreadyExists { .. } => "already_exists",
        ObjectStoreError::Precondition { .. } => "precondition",
        ObjectStoreError::NotModified { .. } => "not_modified",
        ObjectStoreError::NotImplemented { .. } => "not_implemented",
        ObjectStoreError::UnknownConfigurationKey { .. } => "unknown_configuration_key",
        _ => "other",
    }
}

/// Stable `object_store_error_detail` fact value for access, credential,
/// HTTP-status, timeout, and transport failures, or `None` when the error
/// carries no recognized marker.
pub fn object_store_error_detail(error: &ObjectStoreError) -> Option<&'static str> {
    match error {
        ObjectStoreError::PermissionDenied { .. } => Some("access_denied"),
        ObjectStoreError::Unauthenticated { .. } => Some("unauthenticated"),
        ObjectStoreError::NotFound { .. } => Some("not_found"),
        ObjectStoreError::Generic { source, .. } => classify_generic_object_store_source(source.as_ref()),
        _ => None,
    }
}

fn classify_generic_object_store_source(source: &(dyn StdError + 'static)) -> Option<&'static str> {
    let lower = error_chain_text(source).to_ascii_lowercase();
    if contains_any(&lower, &["invalidaccesskeyid", "invalid access key", "invalid_access_key"]) {
        return Some("invalid_access_key");
    }
    if contains_any(&lower, &["expiredtoken", "expired token", "token expired"]) {
        return Some("expired_token");
    }
    if contains_any(&lower, &["signaturedoesnotmatch", "signature mismatch", "signature did not match"]) {
        return Some("signature_mismatch");
    }
    if contains_any(&lower, &["accessdenied", "access denied", "permission denied"]) {
        return Some("access_denied");
    }
    if contains_any(
        &lower,
        &[
            "no credentials",
            "missing credentials",
            "credentials not loaded",
            "credential was not loaded",
            "could not load credentials",
            "unable to load credentials",
        ],
    ) {
        return Some("missing_credentials");
    }
    if contains_any(&lower, &["401 unauthorized", "status code: 401", "http 401"]) {
        return Some("http_401");
    }
    if contains_any(&lower, &["403 forbidden", "status code: 403", "http 403"]) {
        return Some("http_403");
    }
    if lower.contains("timeout") || lower.contains("timed out") {
        return Some("timeout");
    }
    if lower.contains("error sending request") || lower.contains("transport error") {
        return Some("transport");
    }
    None
}

/// Renders `source` and its causes joined by `": "`, keeping at most
/// 32 cause layers and 16 KiB of text (cut on a UTF-8 boundary).
pub fn error_chain_text(source: &(dyn StdError + 'static)) -> String {
    let mut text = bounded_layer_text(&source.to_string(), MAX_ERROR_CHAIN_BYTES);
    let mut current = source.source();
    for _ in 0..MAX_ERROR_CHAIN_DEPTH {
        let Some(layer) = current else { break };
        if text.len() >= MAX_ERROR_CHAIN_BYTES {
            break;
        }
        text.push_str(": ");
        let remaining = MAX_ERROR_CHAIN_BYTES.saturating_sub(text.len());
        text.push_str(&bounded_layer_text(&layer.to_string(), remaining));
        current = layer.source();
    }
    text
}

fn bounded_layer_text(layer: &str, max_bytes: usize) -> String {
    if layer.len() <= max_bytes {
        return layer.to_string();
    }
    let mut end = max_bytes;
    while !layer.is_char_boundary(end) {
        end -= 1;
    }
    layer[..end].to_string()
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}
