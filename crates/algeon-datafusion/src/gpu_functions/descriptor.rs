//! Provider-neutral metadata for one SQL-visible GPU UDTF.

use serde_json::Value;

/// Static planning availability for a GPU UDTF in one session/build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GpuFunctionAvailability {
    /// The function can be statically planned for GPU execution in this session.
    Available,
    /// The function is visible but cannot be planned in this session/build.
    Unavailable { reason: String },
}

/// The reason code of a family whose execution the build leaves out.
pub(crate) const REQUIRED_FEATURE_DISABLED: &str = "required_feature_disabled";

impl GpuFunctionAvailability {
    /// The availability of a family whose execution its Cargo feature
    /// enables.
    pub(crate) fn of_build_feature(enabled: bool) -> Self {
        if enabled { Self::Available } else { Self::Unavailable { reason: REQUIRED_FEATURE_DISABLED.to_string() } }
    }

    /// Returns whether the function is statically plannable.
    pub(crate) const fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    /// Returns the stable reason code when the function is unavailable.
    pub(crate) fn unavailable_reason(&self) -> Option<&str> {
        match self {
            Self::Available => None,
            Self::Unavailable { reason } => Some(reason),
        }
    }
}

/// Shared discovery and description metadata for one execution UDTF.
///
/// Family adapters own the values. The common catalog validates cross-family
/// invariants and owns the stable Arrow projection used by the three metadata
/// metadata UDTFs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuFunctionDescriptor {
    /// Exact canonical execution function name, for example `cuvs_kmeans`.
    pub(crate) function_name: String,
    /// Stable implementation provider identifier, for example `cuvs`.
    pub(crate) provider: String,
    /// Canonical SQL execution signature.
    pub(crate) signature: String,
    /// Short human-readable description.
    pub(crate) summary: String,
    /// Ordered relation-role definitions.
    pub(crate) relation_roles: Value,
    /// Structured descriptions of non-relation arguments.
    pub(crate) argument_descriptions: Value,
    /// JSON Schema for the options object.
    pub(crate) options_schema: Value,
    /// Generic/default and mode-dependent result schemas.
    pub(crate) result_schemas: Value,
    /// JSON Schema accepted by `gpu_validate_call` for this function.
    pub(crate) validation_request_schema: Value,
    /// Structured validation and execution examples.
    pub(crate) examples: Value,
    /// Known static and runtime limitations.
    pub(crate) limitations: Value,
    /// Current session/build availability.
    pub(crate) availability: GpuFunctionAvailability,
}

impl GpuFunctionDescriptor {
    /// Return ordered relation-role names, accepting concise string roles and
    /// verbose role objects with a `name` property.
    pub(crate) fn relation_role_names(&self) -> Result<Vec<&str>, &'static str> {
        let Value::Array(roles) = &self.relation_roles else {
            return Err("relation_roles must be a JSON array");
        };
        roles
            .iter()
            .map(|role| match role {
                Value::String(name) => Ok(name.as_str()),
                Value::Object(role) => {
                    role.get("name").and_then(Value::as_str).ok_or("relation role objects must contain a string name")
                },
                _ => Err("each relation role must be a string or an object with a string name"),
            })
            .collect()
    }

    /// Return relation roles whose descriptor does not mark them optional.
    pub(crate) fn required_relation_role_names(&self) -> Result<Vec<&str>, &'static str> {
        let Value::Array(roles) = &self.relation_roles else {
            return Err("relation_roles must be a JSON array");
        };
        roles
            .iter()
            .filter_map(|role| match role {
                Value::String(name) => Some(Ok(name.as_str())),
                Value::Object(role) => {
                    let required = role.get("required").and_then(Value::as_bool).unwrap_or(true);
                    required.then(|| {
                        role.get("name")
                            .and_then(Value::as_str)
                            .ok_or("relation role objects must contain a string name")
                    })
                },
                _ => Some(Err("each relation role must be a string or an object with a string name")),
            })
            .collect()
    }
}
