//! Object-store lookup and redacted diagnostics.

use std::{error::Error as StdError, sync::Arc};

use datafusion::{
    execution::{object_store::ObjectStoreUrl, runtime_env::RuntimeEnv},
    object_store::{Error as ObjectStoreError, ObjectStore, path::Path as ObjectStorePath},
};
use query_engine::{
    object_store_error::{object_store_error_detail, object_store_error_kind},
    plan::NodeId,
    source::ParquetSourceLocation,
    util::fnv1a64,
};

use crate::error::ErrorSourceDiagnostics;

/// Footer reads resolve each object store from the session's runtime
/// environment, which serves `s3://` URLs from the configured S3 storage.
pub(super) struct FooterObjectStoreRegistry<'a> {
    runtime_env: &'a Arc<RuntimeEnv>,
}

impl<'a> FooterObjectStoreRegistry<'a> {
    pub(super) fn new(runtime_env: &'a Arc<RuntimeEnv>) -> Self {
        Self { runtime_env }
    }

    pub(super) fn store_for_file(
        &self,
        object: &ObjectStoreFile,
        source_node_id: NodeId,
    ) -> std::result::Result<Arc<dyn ObjectStore>, String> {
        self.runtime_env.object_store(&object.store_url).map_err(|error| {
            format!(
                "object_store_unregistered {}",
                foreign_error_diagnostic(
                    "runtime_object_store_lookup",
                    &error,
                    &object_error_context(source_node_id, &object.path),
                    Some(&object.path),
                )
            )
        })
    }
}

pub(super) struct ObjectStoreFile {
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
    Ok(ObjectStoreFile { store_url, path })
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
    error: ObjectStoreError,
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
    format!("{:016x}", fnv1a64(value.as_bytes()))
}
