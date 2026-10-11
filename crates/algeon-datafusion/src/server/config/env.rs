use std::{collections::BTreeMap, sync::OnceLock};

use super::{PathBuf, Result, ServerError, file::ServerFile};
use crate::name_suggestion::closest_name;

/// Path to the TOML server config file; `--config` on the `algeon_server`
/// command line takes precedence. This and [`ALGEON_SERVER_LOG_ENV`] are the
/// two `ALGEON_SERVER_*` variables without a TOML location.
pub const ALGEON_SERVER_CONFIG_FILE_ENV: &str = "ALGEON_SERVER_CONFIG_FILE";
/// Tracing log filter for the `algeon_server` process, defaulting to `info`.
/// `--log` takes precedence. Logging starts before the configuration loads,
/// so this variable has no TOML location.
pub const ALGEON_SERVER_LOG_ENV: &str = "ALGEON_SERVER_LOG";

const ENV_PREFIX: &str = "ALGEON_SERVER_";
const PROCESS_LEVEL_NAMES: [&str; 2] = [ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_LOG_ENV];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LeafKind {
    String,
    Integer,
    Boolean,
}

#[derive(Debug, Clone)]
pub(super) struct ScalarLeaf {
    pub(super) path: String,
    pub(super) env_name: String,
    pub(super) kind: LeafKind,
}

/// Arrays, including `admission.device_profiles` and `auth.mode.users`,
/// have no environment override form.
pub(super) fn scalar_leaves() -> &'static [ScalarLeaf] {
    static LEAVES: OnceLock<Vec<ScalarLeaf>> = OnceLock::new();
    LEAVES.get_or_init(|| {
        let template = toml::Table::try_from(ServerFile::leaf_type_template())
            .expect("the server file schema serializes to a TOML table");
        let mut leaves = Vec::new();
        collect_scalar_leaves(&template, "", &mut leaves);
        leaves
    })
}

fn collect_scalar_leaves(table: &toml::Table, prefix: &str, leaves: &mut Vec<ScalarLeaf>) {
    for (key, value) in table {
        let path = if prefix.is_empty() { key.clone() } else { format!("{prefix}.{key}") };
        let kind = match value {
            toml::Value::Table(nested) => {
                collect_scalar_leaves(nested, &path, leaves);
                continue;
            },
            toml::Value::Array(_) => continue,
            toml::Value::String(_) => LeafKind::String,
            toml::Value::Integer(_) => LeafKind::Integer,
            toml::Value::Boolean(_) => LeafKind::Boolean,
            toml::Value::Float(_) | toml::Value::Datetime(_) => {
                unreachable!("the server file schema has no float or datetime leaf at {path}")
            },
        };
        leaves.push(ScalarLeaf { env_name: env_name_for_path(&path), path, kind });
    }
}

pub(super) fn env_name_for_path(path: &str) -> String {
    format!("{ENV_PREFIX}{}", path.replace('.', "_").to_ascii_uppercase())
}

#[derive(Debug, Default)]
pub(super) struct ServerEnvironment {
    pub(super) config_file: Option<PathBuf>,
    pub(super) overrides: BTreeMap<String, EnvOverride>,
}

#[derive(Debug, Clone)]
pub(super) struct EnvOverride {
    pub(super) env_name: String,
    pub(super) value: toml::Value,
}

impl ServerEnvironment {
    /// Reads every `ALGEON_SERVER_*` variable. A name that is neither a
    /// derived leaf name nor a process-level name fails, like an unknown TOML
    /// field. Blank values count as unset.
    pub(super) fn from_vars(vars: impl IntoIterator<Item = (String, String)>) -> Result<Self> {
        let leaves = scalar_leaves();
        let mut environment = Self::default();
        for (name, value) in vars {
            if !name.starts_with(ENV_PREFIX) {
                continue;
            }
            let value = value.trim();
            if name == ALGEON_SERVER_CONFIG_FILE_ENV {
                environment.config_file = (!value.is_empty()).then(|| PathBuf::from(value));
                continue;
            }
            if name == ALGEON_SERVER_LOG_ENV {
                continue;
            }
            let Some(leaf) = leaves.iter().find(|leaf| leaf.env_name == name) else {
                return Err(unknown_variable_error(&name));
            };
            if value.is_empty() {
                continue;
            }
            let value = parse_env_value(leaf, value)?;
            environment.overrides.insert(leaf.path.clone(), EnvOverride { env_name: name, value });
        }
        Ok(environment)
    }

    pub(super) fn apply_to(&self, table: &mut toml::Table) {
        for (path, value) in &self.overrides {
            let mut segments = path.split('.').peekable();
            let mut current = &mut *table;
            while let Some(segment) = segments.next() {
                if segments.peek().is_none() {
                    current.insert(segment.to_string(), value.value.clone());
                    break;
                }
                let entry =
                    current.entry(segment.to_string()).or_insert_with(|| toml::Value::Table(toml::Table::new()));
                if !entry.is_table() {
                    *entry = toml::Value::Table(toml::Table::new());
                }
                current = entry.as_table_mut().expect("entry was just made a table");
            }
        }
    }
}

fn parse_env_value(leaf: &ScalarLeaf, raw: &str) -> Result<toml::Value> {
    let invalid = |expected: &str| {
        ServerError::configuration(format!(
            "{} {expected}, got `{raw}` (set by environment variable {})",
            leaf.path, leaf.env_name
        ))
        .with_fact("config_field", &leaf.path)
        .with_fact("config_env", &leaf.env_name)
    };
    match leaf.kind {
        LeafKind::String => Ok(toml::Value::String(raw.to_string())),
        LeafKind::Integer => raw
            .parse::<u64>()
            .ok()
            .and_then(|value| i64::try_from(value).ok())
            .map(toml::Value::Integer)
            .ok_or_else(|| invalid("must be a non-negative integer no larger than 9223372036854775807")),
        LeafKind::Boolean => match raw.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(toml::Value::Boolean(true)),
            "false" | "0" | "no" | "off" => Ok(toml::Value::Boolean(false)),
            _ => Err(invalid("must be a boolean (`true` or `false`)")),
        },
    }
}

fn unknown_variable_error(name: &str) -> ServerError {
    let known = scalar_leaves().iter().map(|leaf| leaf.env_name.as_str()).chain(PROCESS_LEVEL_NAMES);
    let rule = "Overrides are ALGEON_SERVER_ followed by a TOML path in upper case with `.` written as `_`; \
                https://algeon.dev/integration/configuration#complete-example lists every one";
    let error = match closest_name(name, ENV_PREFIX, known) {
        Some(suggestion) => ServerError::configuration(format!(
            "unknown environment variable {name}; the closest known name is {suggestion}. {rule}"
        ))
        .with_fact("suggestion", suggestion),
        None => ServerError::configuration(format!(
            "unknown environment variable {name}; no known setting has a similar name. {rule}"
        )),
    };
    error.with_fact("reason", "unknown_environment_variable").with_fact("config_env", name)
}
