//! Recursive redaction guard applied to both the parsed profile and the
//! serialized artifact. It never echoes the offending value.

use serde_json::Value;

use crate::ToolResult;

pub(super) fn ensure_redacted_json(value: &Value, pointer: &str) -> ToolResult<()> {
    match value {
        Value::Object(values) => {
            for (key, value) in values {
                let field = json_pointer(pointer, key);
                if is_sensitive_key(key) {
                    return Err(
                        format!("sensitive field is forbidden in a production-validation artifact at {field}").into()
                    );
                }
                ensure_redacted_json(value, &field)?;
            }
        },
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                ensure_redacted_json(value, &format!("{pointer}/{index}"))?;
            }
        },
        Value::String(value) if has_inline_secret(value) => {
            return Err(
                format!("credential-like text is forbidden in a production-validation artifact at {pointer}").into()
            );
        },
        _ => {},
    }
    Ok(())
}

fn json_pointer(pointer: &str, field: &str) -> String {
    format!("{pointer}/{}", field.replace('~', "~0").replace('/', "~1"))
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase().replace(['-', '_'], "");
    [
        "password",
        "passwd",
        "secret",
        "token",
        "authorization",
        "credential",
        "apikey",
        "privatekey",
        "accesskey",
        "sessionkey",
        "cookie",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn has_inline_secret(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("://")
        && normalized
            .split("://")
            .nth(1)
            .is_some_and(|rest| rest.split('/').next().is_some_and(|authority| authority.contains('@')))
        || normalized.contains("-----begin") && normalized.contains("private key-----")
        || normalized.contains("bearer ")
        || normalized.contains("x-amz-signature=")
}
