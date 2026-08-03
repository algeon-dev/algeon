//! Auth configuration types loaded from the server TOML config file.

use std::{collections::BTreeSet, time::Duration};

use serde::{Deserialize, Serialize};

use super::password::validate_password_hash;
use crate::server::error::{Result, ServerError};

/// Top-level auth section of the server TOML config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AuthConfig {
    /// Selected auth mode. When omitted, defaults to [`AuthMode::Disabled`].
    pub mode: AuthMode,
}

/// Authentication mechanism applied to incoming Flight SQL RPCs.
///
/// The TOML representation is a tagged enum keyed on `kind`:
///
/// ```toml
/// [auth]
/// mode = { kind = "disabled" }
///
/// [auth.mode]
/// kind = "basic_bearer"
/// token_ttl_secs = 3600
///
/// [[auth.mode.users]]
/// username = "alice"
/// # generate with: cargo run --bin datafusion_nexus_server -- hash-password
/// password_hash = "$argon2id$v=19$m=19456,t=2,p=1$..."
/// roles = ["reader"]
/// ```
///
/// Future modes such as JWT, OIDC, and mTLS should be added as explicit
/// variants so configuration stays closed and validated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AuthMode {
    /// No authentication. Every RPC is accepted.
    #[default]
    Disabled,
    /// `Basic` handshake exchanged for a server-issued `Bearer` token,
    /// with credentials validated against a static user list.
    BasicBearer {
        /// Static user list, password hashes only.
        #[serde(default)]
        users: Vec<StaticUserConfig>,
        /// TTL applied to issued bearer tokens, in seconds. Omitted or
        /// zero means tokens never expire (not recommended for
        /// production).
        #[serde(default)]
        token_ttl_secs: u64,
    },
}

impl AuthMode {
    /// Bearer-token TTL, or `None` for non-expiring tokens.
    pub fn token_ttl(&self) -> Option<Duration> {
        match self {
            AuthMode::Disabled => None,
            AuthMode::BasicBearer { token_ttl_secs, .. } => {
                if *token_ttl_secs == 0 {
                    None
                } else {
                    Some(Duration::from_secs(*token_ttl_secs))
                }
            },
        }
    }

    /// Validate the config — unique usernames, non-empty fields, hash
    /// shape sanity. Catches obvious config-file mistakes at load time
    /// rather than at handshake time.
    pub fn validate(&self) -> Result<()> {
        match self {
            AuthMode::Disabled => Ok(()),
            AuthMode::BasicBearer { users, .. } => {
                if users.is_empty() {
                    return Err(ServerError::configuration(
                        "auth.mode.basic_bearer requires at least one entry in `users`",
                    ));
                }
                let mut seen = BTreeSet::new();
                for user in users {
                    if user.username.is_empty() {
                        return Err(ServerError::configuration("auth user `username` must be non-empty"));
                    }
                    if !seen.insert(user.username.as_str()) {
                        return Err(ServerError::configuration(format!(
                            "auth users contain duplicate username `{}`",
                            user.username
                        )));
                    }
                    if let Err(error) = validate_password_hash(&user.password_hash) {
                        return Err(ServerError::configuration(format!(
                            "auth user `{}` password_hash must be a valid Argon2 PHC string: {error}",
                            user.username,
                        )));
                    }
                }
                Ok(())
            },
        }
    }
}

/// One row of the static user list under `[auth.mode.basic_bearer]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticUserConfig {
    pub username: String,
    /// Argon2 PHC string (`$argon2id$v=19$...`). Plain passwords are
    /// rejected at load time.
    pub password_hash: String,
    /// Free-form role tags surfaced as `Principal::roles`. Not enforced
    /// by the auth layer itself — downstream code (or a future
    /// authorization layer) decides what they mean.
    #[serde(default)]
    pub roles: Vec<String>,
}
