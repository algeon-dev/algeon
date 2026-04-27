//! Common `gpu_validate_call` request and result models.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// Current common validation-envelope version.
pub(crate) const GPU_CALL_SCHEMA_VERSION: u32 = 1;

/// A named table or view reference supplied for a descriptor-defined role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuRelationReference {
    /// One-, two-, or three-part DataFusion table reference.
    pub(crate) table: String,
}

/// Parsed common `gpu_validate_call` request envelope.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GpuCallRequest {
    /// Version of the common request envelope.
    pub(crate) schema_version: u32,
    /// Descriptor-role keyed relation references.
    pub(crate) relations: BTreeMap<String, GpuRelationReference>,
    /// Family-owned options object.
    pub(crate) options: Value,
}

/// Structured invalid request-envelope result used before family validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuCallRequestError {
    /// Stable invalid-row reason code.
    pub(crate) code: &'static str,
    /// Human-readable diagnostic.
    pub(crate) message: String,
}

impl GpuCallRequest {
    /// Parse and validate the version-one common envelope.
    pub(crate) fn parse(call_json: &str) -> Result<Self, GpuCallRequestError> {
        let value: Value = serde_json::from_str(call_json).map_err(|error| GpuCallRequestError {
            code: "invalid_call_json",
            message: format!("call_json must be valid JSON: {error}"),
        })?;
        let Value::Object(object) = value else {
            return Err(invalid("call_json must be a JSON object"));
        };
        reject_unknown_keys(&object, &["schema_version", "relations", "options"])?;

        let schema_version = object
            .get("schema_version")
            .and_then(Value::as_u64)
            .and_then(|version| u32::try_from(version).ok())
            .ok_or_else(|| invalid("call_json.schema_version must be an integer"))?;
        if schema_version != GPU_CALL_SCHEMA_VERSION {
            return Err(GpuCallRequestError {
                code: "unsupported_schema_version",
                message: format!("call_json.schema_version must be {GPU_CALL_SCHEMA_VERSION}, got {schema_version}"),
            });
        }

        let relations = object
            .get("relations")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("call_json.relations must be an object"))?
            .iter()
            .map(|(role, relation)| parse_relation(role, relation).map(|relation| (role.clone(), relation)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let options = object
            .get("options")
            .filter(|options| options.is_object())
            .cloned()
            .ok_or_else(|| invalid("call_json.options must be an object"))?;

        Ok(Self { schema_version, relations, options })
    }

    /// Canonical JSON representation of this common envelope.
    pub(crate) fn normalized_json(&self) -> Value {
        let relations = self
            .relations
            .iter()
            .map(|(role, relation)| (role.clone(), serde_json::json!({ "table": relation.table })))
            .collect::<Map<_, _>>();
        serde_json::json!({
            "schema_version": self.schema_version,
            "relations": relations,
            "options": self.options,
        })
    }
}

/// Common result model projected by `gpu_validate_call`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GpuCallValidation {
    /// Whether all static validation succeeded.
    pub(crate) valid: bool,
    /// Stable reason code for an invalid call.
    pub(crate) error_code: Option<String>,
    /// Human-readable validation result.
    pub(crate) message: String,
    /// Canonical name or the unresolved requested name.
    pub(crate) function_name: String,
    /// Whether all required relations resolved.
    pub(crate) relations_resolved: bool,
    /// Versioned request after family defaults/canonicalization.
    pub(crate) normalized_call: Value,
    /// Concrete output schema when derivable, or an empty JSON object.
    pub(crate) output_schema: Value,
    /// Whether the selected call would target GPU after static planning.
    pub(crate) would_execute_gpu: bool,
    /// Per-relation and family-specific facts.
    pub(crate) details: Value,
}

impl GpuCallValidation {
    /// Construct a common invalid row without invoking a family adapter.
    pub(crate) fn invalid(
        code: impl Into<String>,
        message: impl Into<String>,
        function_name: impl Into<String>,
        normalized_call: Value,
    ) -> Self {
        Self {
            valid: false,
            error_code: Some(code.into()),
            message: message.into(),
            function_name: function_name.into(),
            relations_resolved: false,
            normalized_call,
            output_schema: Value::Object(Map::new()),
            would_execute_gpu: false,
            details: Value::Object(Map::new()),
        }
    }
}

fn parse_relation(role: &str, relation: &Value) -> Result<GpuRelationReference, GpuCallRequestError> {
    let Value::Object(relation) = relation else {
        return Err(invalid(format!("call_json.relations.{role} must be an object")));
    };
    reject_unknown_keys(relation, &["table"])?;
    let table = relation.get("table").and_then(Value::as_str).filter(|table| valid_table_reference(table)).ok_or_else(
        || invalid(format!("call_json.relations.{role}.table must be a one-, two-, or three-part table reference")),
    )?;
    Ok(GpuRelationReference { table: table.to_string() })
}

fn valid_table_reference(table: &str) -> bool {
    let parts = table.split('.').collect::<Vec<_>>();
    (1..=3).contains(&parts.len()) && parts.iter().copied().all(is_simple_identifier)
}

fn is_simple_identifier(part: &str) -> bool {
    let mut chars = part.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn reject_unknown_keys(object: &Map<String, Value>, allowed: &[&str]) -> Result<(), GpuCallRequestError> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(GpuCallRequestError {
            code: "invalid_call_json",
            message: format!("call_json contains unknown key '{key}'"),
        });
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> GpuCallRequestError {
    GpuCallRequestError { code: "invalid_call_json", message: message.into() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_canonicalizes_version_one_envelope() {
        let request = GpuCallRequest::parse(
            r#"{"schema_version":1,"relations":{"input":{"table":"catalog.schema.input"}},"options":{"k":3}}"#,
        )
        .unwrap();
        assert_eq!(request.relations["input"].table, "catalog.schema.input");
        assert_eq!(request.normalized_json()["schema_version"], 1);
    }

    #[test]
    fn rejects_unknown_keys_and_versions() {
        let unknown =
            GpuCallRequest::parse(r#"{"schema_version":1,"relations":{},"options":{},"extra":true}"#).unwrap_err();
        assert_eq!(unknown.code, "invalid_call_json");
        let version = GpuCallRequest::parse(r#"{"schema_version":2,"relations":{},"options":{}}"#).unwrap_err();
        assert_eq!(version.code, "unsupported_schema_version");
    }
}
