//! Shared dense-relation bindings and GPU function option metadata.

use std::collections::BTreeSet;

use arrow::datatypes::DataType;
use serde_json::{Value, json};

/// Scalar ID types accepted by dense relational operations.
///
/// This is deliberately narrower than the set of Arrow integer types. Native
/// cuVS output/gather paths are only defined for these five stable SQL types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationIdType {
    Int32,
    Int64,
    Utf8,
    LargeUtf8,
    Utf8View,
}

impl RelationIdType {
    pub fn from_data_type(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Int32 => Some(Self::Int32),
            DataType::Int64 => Some(Self::Int64),
            DataType::Utf8 => Some(Self::Utf8),
            DataType::LargeUtf8 => Some(Self::LargeUtf8),
            DataType::Utf8View => Some(Self::Utf8View),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::Utf8 => "Utf8",
            Self::LargeUtf8 => "LargeUtf8",
            Self::Utf8View => "Utf8View",
        }
    }
}

/// Stable order used in generated descriptions and validation diagnostics.
pub const RELATION_ID_TYPES: &[RelationIdType] = &[
    RelationIdType::Int32,
    RelationIdType::Int64,
    RelationIdType::Utf8,
    RelationIdType::LargeUtf8,
    RelationIdType::Utf8View,
];

/// Vector element types for dense relation descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorElementType {
    Float32,
    Float64,
    Int8,
    UInt8,
}

impl VectorElementType {
    pub fn from_data_type(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Float32 => Some(Self::Float32),
            DataType::Float64 => Some(Self::Float64),
            DataType::Int8 => Some(Self::Int8),
            DataType::UInt8 => Some(Self::UInt8),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "Float32",
            Self::Float64 => "Float64",
            Self::Int8 => "Int8",
            Self::UInt8 => "UInt8",
        }
    }
}

/// Canonical scalar option metadata used by JSON schema generation and value
/// validation in the DataFusion adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuOptionKind {
    RelationBinding,
    Integer {
        minimum: u64,
        maximum: u64,
        default: Option<u64>,
    },
    NonnegativeNumber {
        default: f64,
    },
    PositiveNumber {
        default: f64,
    },
    UnitInterval {
        default: f64,
    },
    Fraction {
        default: f64,
    },
    Boolean {
        default: Option<bool>,
    },
    Enum {
        values: &'static [&'static str],
        default: Option<&'static str>,
    },
    /// Non-empty string of at most `max_chars` Unicode scalar values, the unit
    /// JSON Schema `maxLength` counts.
    Text {
        max_chars: usize,
    },
}

impl GpuOptionKind {
    pub const fn default_u64(self) -> Option<u64> {
        match self {
            Self::Integer { default, .. } => default,
            _ => None,
        }
    }

    pub const fn default_f64(self) -> Option<f64> {
        match self {
            Self::NonnegativeNumber { default }
            | Self::PositiveNumber { default }
            | Self::UnitInterval { default }
            | Self::Fraction { default } => Some(default),
            _ => None,
        }
    }

    pub const fn default_bool(self) -> Option<bool> {
        match self {
            Self::Boolean { default } => default,
            _ => None,
        }
    }

    pub const fn default_str(self) -> Option<&'static str> {
        match self {
            Self::Enum { default, .. } => default,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuOptionDefinition {
    pub name: &'static str,
    pub required: bool,
    pub kind: GpuOptionKind,
    /// User-facing semantics projected through the unified GPU descriptor.
    pub description: &'static str,
}

/// One documented output field in a GPU relational result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuOutputFieldDefinition {
    pub name: &'static str,
    pub data_type: &'static str,
    pub nullable: bool,
    /// Literal option that determines how many physical output fields exist.
    pub repeats: Option<&'static str>,
    pub description: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuOptionValidationError {
    code: &'static str,
    message: String,
    argument: Option<&'static str>,
}

impl GpuOptionValidationError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    /// The declared option this error concerns, when it is attributable to one.
    pub fn argument(&self) -> Option<&'static str> {
        self.argument
    }

    pub(crate) fn for_argument(mut self, argument: &'static str) -> Self {
        self.argument = Some(argument);
        self
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), argument: None }
    }
}

pub(crate) fn dense_relation_binding_schema(elements: &[VectorElementType]) -> Value {
    let elements_label = elements.iter().map(|element| element.as_str()).collect::<Vec<_>>().join(", ");
    let list_types = elements
        .iter()
        .map(|element| {
            format!(
                "FixedSizeList<{}, D>, List<{}>, or LargeList<{}>",
                element.as_str(),
                element.as_str(),
                element.as_str()
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    json!({
        "type": "object", "additionalProperties": false, "required": ["id", "vector"],
        "properties": {
            "id": {"type": "string", "minLength": 1,
                "description": "Non-null logical row ID column. IDs may repeat; result ordinals disambiguate physical rows."},
            "vector": {"description": "Choose exactly one dense-vector binding shape.", "oneOf": [
                {"type": "object", "additionalProperties": false, "required": ["columns"], "properties": {
                    "columns": {"type": "array", "minItems": 1, "uniqueItems": true,
                        "description": format!("Ordered, unique non-null {elements_label} feature columns; order defines vector dimensions."),
                        "items": {"type": "string", "minLength": 1}}}},
                {"type": "object", "additionalProperties": false, "required": ["column"], "properties": {
                    "column": {"type": "string", "minLength": 1, "description": format!("One non-null {list_types} column.")}}}
            ]}
        }
    })
}

pub(crate) fn validate_dense_relation_binding_value(
    family: &str,
    option: &GpuOptionDefinition,
    value: &Value,
    allow_label: bool,
) -> Result<(), GpuOptionValidationError> {
    let invalid = || {
        GpuOptionValidationError::new("missing_or_invalid_option", format!("invalid {family} option '{}'", option.name))
    };
    let binding = value.as_object().ok_or_else(invalid)?;
    if let Some(unknown) =
        binding.keys().find(|key| !matches!(key.as_str(), "id" | "vector") && !(allow_label && key.as_str() == "label"))
    {
        return Err(GpuOptionValidationError::new(
            "unknown_option",
            format!("unknown {family} option '{unknown}' in {}", option.name),
        ));
    }
    if binding.get("id").and_then(Value::as_str).filter(|value| !value.is_empty()).is_none() {
        return Err(GpuOptionValidationError::new(
            "missing_or_invalid_option",
            format!("{family} option '{}.id' must be a non-empty string", option.name),
        ));
    }
    if allow_label && binding.get("label").and_then(Value::as_str) != Some("label") {
        return Err(GpuOptionValidationError::new(
            "missing_or_invalid_option",
            format!("{family} option '{}.label' must be 'label'", option.name),
        ));
    }
    let vector = binding.get("vector").and_then(Value::as_object).ok_or_else(|| {
        GpuOptionValidationError::new(
            "missing_or_invalid_option",
            format!("{family} option '{}.vector' must be an object", option.name),
        )
    })?;
    if let Some(unknown) = vector.keys().find(|key| !matches!(key.as_str(), "columns" | "column")) {
        return Err(GpuOptionValidationError::new(
            "unknown_option",
            format!("unknown {family} option '{unknown}' in {}.vector", option.name),
        ));
    }
    match (vector.get("columns"), vector.get("column")) {
        (Some(columns), None) => {
            let columns = columns.as_array().filter(|columns| !columns.is_empty()).ok_or_else(|| {
                GpuOptionValidationError::new(
                    "invalid_vector_binding",
                    format!("{family} option '{}.vector.columns' must be a non-empty string array", option.name),
                )
            })?;
            let mut unique = BTreeSet::new();
            for column in columns {
                let column = column.as_str().filter(|column| !column.is_empty()).ok_or_else(|| {
                    GpuOptionValidationError::new(
                        "invalid_vector_binding",
                        format!("{family} option '{}.vector.columns' entries must be non-empty strings", option.name),
                    )
                })?;
                if !unique.insert(column) {
                    return Err(GpuOptionValidationError::new(
                        "duplicate_vector_column",
                        format!("{family} option '{}.vector.columns' contains duplicate field '{column}'", option.name),
                    ));
                }
            }
        },
        (None, Some(column)) => {
            if column.as_str().filter(|column| !column.is_empty()).is_none() {
                return Err(GpuOptionValidationError::new(
                    "invalid_vector_binding",
                    format!("{family} option '{}.vector.column' must be a non-empty string", option.name),
                ));
            }
        },
        _ => {
            return Err(GpuOptionValidationError::new(
                "invalid_vector_binding",
                format!("{family} option '{}.vector' must contain exactly one of columns or column", option.name),
            ));
        },
    }
    Ok(())
}

pub(crate) fn option_schema(kind: GpuOptionKind, relation_schema: &Value) -> Value {
    match kind {
        GpuOptionKind::RelationBinding => relation_schema.clone(),
        GpuOptionKind::Integer { minimum, maximum, default } => {
            let mut value = json!({"type": "integer", "minimum": minimum, "maximum": maximum});
            if let Some(default) = default {
                value["default"] = json!(default);
            }
            value
        },
        GpuOptionKind::NonnegativeNumber { default } => json!({"type": "number", "minimum": 0.0, "default": default}),
        GpuOptionKind::PositiveNumber { default } => {
            json!({"type": "number", "exclusiveMinimum": 0, "default": default})
        },
        GpuOptionKind::UnitInterval { default } => {
            json!({"type": "number", "minimum": 0, "maximum": 1, "default": default})
        },
        GpuOptionKind::Fraction { default } => {
            json!({"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": default})
        },
        GpuOptionKind::Boolean { default } => {
            let mut value = json!({"type": "boolean"});
            if let Some(default) = default {
                value["default"] = json!(default);
            }
            value
        },
        GpuOptionKind::Enum { values, default } => {
            let mut value = json!({"type": "string", "enum": values});
            if let Some(default) = default {
                value["default"] = json!(default);
            }
            value
        },
        GpuOptionKind::Text { max_chars } => json!({"type": "string", "minLength": 1, "maxLength": max_chars}),
    }
}

pub(crate) fn canonicalize_options(
    options: &[GpuOptionDefinition],
    raw: &str,
    family: &str,
    validate_relation: impl Fn(&GpuOptionDefinition, &Value) -> Result<(), GpuOptionValidationError>,
) -> Result<serde_json::Map<String, Value>, GpuOptionValidationError> {
    let parsed: Value = serde_json::from_str(raw).map_err(|_| {
        GpuOptionValidationError::new("invalid_options_json", format!("{family} options must be valid JSON"))
    })?;
    let mut object = parsed.as_object().cloned().ok_or_else(|| {
        GpuOptionValidationError::new("options_not_object", format!("{family} options must be a JSON object"))
    })?;
    if let Some(unknown) = object.keys().find(|key| !options.iter().any(|option| option.name == key.as_str())) {
        return Err(GpuOptionValidationError::new("unknown_option", format!("unknown {family} option '{unknown}'")));
    }
    for option in options {
        let Some(value) = object.get_mut(option.name) else {
            if option.required {
                return Err(GpuOptionValidationError::new(
                    "missing_or_invalid_option",
                    format!("{family} option '{}' is required", option.name),
                )
                .for_argument(option.name));
            }
            let default = match option.kind {
                GpuOptionKind::Integer { default: Some(value), .. } => json!(value),
                GpuOptionKind::NonnegativeNumber { default }
                | GpuOptionKind::PositiveNumber { default }
                | GpuOptionKind::UnitInterval { default }
                | GpuOptionKind::Fraction { default } => json!(default),
                GpuOptionKind::Boolean { default: Some(value) } => json!(value),
                GpuOptionKind::Enum { default: Some(value), .. } => json!(value),
                _ => continue,
            };
            object.insert(option.name.to_string(), default);
            continue;
        };
        validate_option_value(option, value, family, &validate_relation)
            .map_err(|error| error.for_argument(option.name))?;
    }
    Ok(object)
}

pub(crate) fn validate_option_value(
    option: &GpuOptionDefinition,
    value: &Value,
    family: &str,
    validate_relation: &impl Fn(&GpuOptionDefinition, &Value) -> Result<(), GpuOptionValidationError>,
) -> Result<(), GpuOptionValidationError> {
    let invalid = || {
        GpuOptionValidationError::new("missing_or_invalid_option", format!("invalid {family} option '{}'", option.name))
    };
    match option.kind {
        GpuOptionKind::RelationBinding => validate_relation(option, value)?,
        GpuOptionKind::Integer { minimum, maximum, .. } => {
            if value.as_u64().filter(|value| *value >= minimum && *value <= maximum).is_none() {
                return Err(invalid());
            }
        },
        GpuOptionKind::NonnegativeNumber { .. } => {
            if value.as_f64().filter(|value| value.is_finite() && *value >= 0.0).is_none() {
                return Err(invalid());
            }
        },
        GpuOptionKind::PositiveNumber { .. } => {
            if value.as_f64().filter(|value| value.is_finite() && *value > 0.0).is_none() {
                return Err(invalid());
            }
        },
        GpuOptionKind::UnitInterval { .. } => {
            if value.as_f64().filter(|value| value.is_finite() && (0.0..=1.0).contains(value)).is_none() {
                return Err(invalid());
            }
        },
        GpuOptionKind::Fraction { .. } => {
            if value.as_f64().filter(|value| value.is_finite() && *value > 0.0 && *value <= 1.0).is_none() {
                return Err(invalid());
            }
        },
        GpuOptionKind::Boolean { .. } => {
            if !value.is_boolean() {
                return Err(invalid());
            }
        },
        GpuOptionKind::Enum { values, .. } => {
            if value.as_str().is_none_or(|value| !values.contains(&value)) {
                return Err(invalid());
            }
        },
        GpuOptionKind::Text { max_chars } => {
            if value.as_str().is_none_or(|value| value.is_empty() || value.chars().count() > max_chars) {
                return Err(invalid());
            }
        },
    }
    Ok(())
}

/// Binding from one native relation to a dense vector matrix.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DenseVectorBinding {
    /// Separate non-null columns of one supported element type packed in the listed order.
    Columns(Vec<String>),
    /// One non-null fixed-size or variable-size list column of a supported element type.
    VectorColumn(String),
}

impl DenseVectorBinding {
    pub fn columns(columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::Columns(columns.into_iter().map(Into::into).collect())
    }

    pub fn vector_column(column: impl Into<String>) -> Self {
        Self::VectorColumn(column.into())
    }

    pub fn display_label(&self) -> String {
        match self {
            Self::Columns(columns) => columns.join("|"),
            Self::VectorColumn(column) => format!("[{column}]"),
        }
    }
}
