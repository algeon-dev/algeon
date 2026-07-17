//! Object-store lookup, native storage-profile fallback, and redacted diagnostics.

use std::{collections::HashMap, error::Error as StdError, sync::Arc};

use datafusion::{
    execution::{object_store::ObjectStoreUrl, runtime_env::RuntimeEnv},
    object_store::path::Path as ObjectStorePath,
};
use nexus_query_engine::{
    plan::NodeId,
    runtime::SourceStorageProfile,
    source::{LocalParquetFile, ParquetSourceLocation},
};

use crate::error::ErrorSourceDiagnostics;

pub(super) struct FooterObjectStoreRegistry<'a> {
    runtime_env: &'a Arc<RuntimeEnv>,
    source_storage_profiles: &'a [SourceStorageProfile],
    profile_stores: HashMap<(String, String), Arc<dyn datafusion::object_store::ObjectStore>>,
}

impl<'a> FooterObjectStoreRegistry<'a> {
    pub(super) fn new(runtime_env: &'a Arc<RuntimeEnv>, source_storage_profiles: &'a [SourceStorageProfile]) -> Self {
        Self { runtime_env, source_storage_profiles, profile_stores: HashMap::new() }
    }

    pub(super) fn store_for_file(
        &mut self,
        file: &LocalParquetFile,
        object: &ObjectStoreFile,
        source_node_id: NodeId,
    ) -> std::result::Result<Arc<dyn datafusion::object_store::ObjectStore>, String> {
        match self.runtime_env.object_store(&object.store_url) {
            Ok(store) => Ok(store),
            Err(runtime_error) if object.scheme == "s3" => {
                let Some(profile_id) = file.resolved_source.storage_profile_id() else {
                    return Err(format!(
                        "object_store_unregistered {}",
                        foreign_error_diagnostic(
                            "runtime_object_store_lookup",
                            &runtime_error,
                            &object_error_context(source_node_id, &object.path),
                            Some(&object.path),
                        )
                    ));
                };
                let key = (object.bucket.clone(), profile_id.to_string());
                if let Some(store) = self.profile_stores.get(&key) {
                    return Ok(Arc::clone(store));
                }
                let Some(profile) = self.source_storage_profiles.iter().find(|profile| profile.id() == profile_id)
                else {
                    return Err(format!(
                        "source_storage_profile_missing storage_profile_id={}",
                        stable_log_field(profile_id)
                    ));
                };
                let fallback_region = profile
                    .s3_profile()
                    .region
                    .as_deref()
                    .filter(|region| !region.is_empty())
                    .map(ToOwned::to_owned)
                    .or_else(|| std::env::var("AWS_REGION").ok().filter(|region| !region.is_empty()))
                    .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|region| !region.is_empty()))
                    .unwrap_or_else(|| "us-east-1".to_string());
                let store = profile.build_s3_object_store(&object.bucket, fallback_region).map_err(|error| {
                    foreign_error_diagnostic(
                        "source_storage_profile_store_build",
                        &error,
                        &object_error_context(source_node_id, &object.path),
                        Some(&object.path),
                    )
                })?;
                self.profile_stores.insert(key, Arc::clone(&store));
                Ok(store)
            },
            Err(error) => Err(format!(
                "object_store_unregistered {}",
                foreign_error_diagnostic(
                    "runtime_object_store_lookup",
                    &error,
                    &object_error_context(source_node_id, &object.path),
                    Some(&object.path),
                )
            )),
        }
    }
}

pub(super) struct ObjectStoreFile {
    scheme: String,
    bucket: String,
    pub(super) store_url: ObjectStoreUrl,
    pub(super) path: ObjectStorePath,
}

pub(super) fn object_store_file(location: &ParquetSourceLocation) -> std::result::Result<ObjectStoreFile, String> {
    let ParquetSourceLocation::ObjectStoreUri { uri, .. } = location else {
        return Err("not_object_store_uri".to_string());
    };
    let (scheme, rest) = uri.split_once("://").ok_or_else(|| "missing_uri_scheme".to_string())?;
    let (authority, path) = rest.split_once('/').ok_or_else(|| "missing_object_path".to_string())?;
    if authority.is_empty() {
        return Err("missing_uri_authority".to_string());
    }
    if path.is_empty() {
        return Err("missing_object_path".to_string());
    }
    let store_url =
        ObjectStoreUrl::parse(format!("{scheme}://{authority}")).map_err(|_| "invalid_object_store_url".to_string())?;
    let path = ObjectStorePath::parse(path).map_err(|_| "invalid_object_path".to_string())?;
    Ok(ObjectStoreFile { scheme: scheme.to_string(), bucket: authority.to_string(), store_url, path })
}

pub(super) fn object_error_context(source_node_id: NodeId, path: &ObjectStorePath) -> Vec<(&'static str, String)> {
    let path = path.to_string();
    vec![
        ("source_node_id", source_node_id.to_string()),
        ("object_path_hash", stable_hash_hex(&path)),
        ("object_path_leaf", redacted_object_path_leaf(&path)),
    ]
}

pub(super) fn object_store_pruning_error(
    error: datafusion::object_store::Error,
    context: &[(&'static str, String)],
    path: &ObjectStorePath,
) -> String {
    let kind = object_store_error_kind(&error);
    let mut fields = vec![format!("object_store_error_kind={kind}")];
    if let Some(detail) = object_store_error_detail(&error) {
        fields.push(format!("object_store_error_detail={detail}"));
    }
    if let Some(chain) = ErrorSourceDiagnostics::from_error(&error).chain() {
        fields.push(format!("error_source_chain={}", stable_log_field(&redact_object_path(chain, path))));
    }
    for (key, value) in context {
        fields.push(format!("{key}={}", stable_log_field(value)));
    }
    fields.join(" ")
}

fn object_store_error_kind(error: &datafusion::object_store::Error) -> &'static str {
    match error {
        datafusion::object_store::Error::PermissionDenied { .. } => "permission_denied",
        datafusion::object_store::Error::Unauthenticated { .. } => "unauthenticated",
        datafusion::object_store::Error::NotFound { .. } => "not_found",
        datafusion::object_store::Error::Generic { .. } => "generic",
        datafusion::object_store::Error::InvalidPath { .. } => "invalid_path",
        datafusion::object_store::Error::NotSupported { .. } => "not_supported",
        #[allow(unreachable_patterns)]
        _ => "other",
    }
}

fn object_store_error_detail(error: &datafusion::object_store::Error) -> Option<&'static str> {
    match error {
        datafusion::object_store::Error::PermissionDenied { .. } => Some("access_denied"),
        datafusion::object_store::Error::Unauthenticated { .. } => Some("unauthenticated"),
        datafusion::object_store::Error::NotFound { .. } => Some("not_found"),
        datafusion::object_store::Error::Generic { source, .. } => {
            classify_generic_object_store_source(source.as_ref())
        },
        _ => None,
    }
}

pub(super) fn classify_generic_object_store_source(source: &(dyn StdError + 'static)) -> Option<&'static str> {
    let text = error_chain_text(source);
    let lower = text.to_ascii_lowercase();
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
    if contains_any(&lower, &["no credentials", "missing credentials", "could not load credentials"]) {
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
    None
}

pub(super) fn foreign_error_diagnostic(
    kind: &'static str,
    error: &(dyn StdError + 'static),
    context: &[(&'static str, String)],
    redact_path: Option<&ObjectStorePath>,
) -> String {
    let diagnostics = ErrorSourceDiagnostics::from_error(error);
    let chain = diagnostics.chain().map(str::to_string).unwrap_or_else(|| error.to_string());
    let chain = redact_path.map_or(chain.clone(), |path| redact_object_path(&chain, path));
    let mut fields = vec![format!("{kind}={}", stable_log_field(&chain))];
    if let Some(root_cause) = diagnostics.root_cause_when_distinct() {
        let root_cause =
            redact_path.map_or_else(|| root_cause.to_string(), |path| redact_object_path(root_cause, path));
        fields.push(format!("error_root_cause={}", stable_log_field(&root_cause)));
    }
    for (key, value) in context {
        fields.push(format!("{key}={}", stable_log_field(value)));
    }
    fields.join(" ")
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

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

pub(super) fn stable_log_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn redact_object_path(message: &str, path: &ObjectStorePath) -> String {
    let path = path.to_string();
    let message = if path.is_empty() { message.to_string() } else { message.replace(&path, "<redacted-object-path>") };
    message
        .split_whitespace()
        .map(|token| {
            if token.contains("://") {
                "<redacted-uri>".to_string()
            } else if let Some((prefix, _)) = token.split_once('?') {
                format!("{prefix}?<redacted-query>")
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn redacted_object_path_leaf(path: &str) -> String {
    let path = path.split_once('?').map_or(path, |(path, _)| path);
    path.rsplit('/').next().filter(|leaf| !leaf.is_empty()).unwrap_or("<empty>").to_string()
}

pub(super) fn stable_hash_hex(value: &str) -> String {
    let hash = value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325_u64, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3));
    format!("{hash:016x}")
}
