//! Argon2id password hashing and verification.

use std::fmt;

use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, Salt, SaltString},
};
use rand::{TryRng, rngs::SysRng};

#[derive(Debug)]
pub enum PasswordHashError {
    /// The PHC string in the config file is malformed.
    InvalidHash(String),
    /// Hashing itself failed (out of memory, etc.).
    HashFailure(String),
}

impl fmt::Display for PasswordHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHash(reason) => write!(f, "invalid argon2 password hash: {reason}"),
            Self::HashFailure(reason) => write!(f, "argon2 hashing failed: {reason}"),
        }
    }
}

impl std::error::Error for PasswordHashError {}

/// Parse and validate an Argon2 PHC string without performing a password
/// verification. Configuration loading uses this to reject malformed hashes
/// before the server accepts handshakes.
pub(crate) fn validate_password_hash(phc_string: &str) -> Result<(), PasswordHashError> {
    let parsed = PasswordHash::new(phc_string).map_err(|error| PasswordHashError::InvalidHash(error.to_string()))?;
    if !matches!(parsed.algorithm.as_str(), "argon2d" | "argon2i" | "argon2id") {
        Err(PasswordHashError::InvalidHash(format!(
            "password hash algorithm `{}` is not an Argon2 variant",
            parsed.algorithm
        )))
    } else if parsed.salt.is_none() {
        Err(PasswordHashError::InvalidHash("password hash is missing its salt".to_string()))
    } else if parsed.hash.is_none() {
        Err(PasswordHashError::InvalidHash("password hash is missing its encoded output".to_string()))
    } else {
        Ok(())
    }
}

/// Verify a plaintext password against an Argon2 PHC string. Returns
/// `Ok(true)` on match, `Ok(false)` on mismatch (the only two outcomes
/// callers should branch on). A malformed `phc_string` returns
/// [`PasswordHashError::InvalidHash`] so the config-file mistake
/// surfaces during handshake rather than as a silent auth failure.
pub fn verify_password(password: &str, phc_string: &str) -> Result<bool, PasswordHashError> {
    let parsed = PasswordHash::new(phc_string).map_err(|error| PasswordHashError::InvalidHash(error.to_string()))?;
    match Argon2::default().verify_password(password.as_bytes(), &parsed) {
        Ok(()) => Ok(true),
        Err(argon2::password_hash::Error::Password) => Ok(false),
        Err(error) => Err(PasswordHashError::InvalidHash(error.to_string())),
    }
}

/// Hash a plaintext password with Argon2id and the library defaults
/// (m=19456, t=2, p=1 — current OWASP recommendation as of mid-2024).
///
/// Operators run this once per user when authoring the TOML config and
/// paste the returned PHC string into `password_hash`.
pub fn hash_password(password: &str) -> Result<String, PasswordHashError> {
    let mut salt_bytes = [0u8; Salt::RECOMMENDED_LENGTH];
    SysRng
        .try_fill_bytes(&mut salt_bytes)
        .map_err(|error| PasswordHashError::HashFailure(format!("failed to generate password salt: {error}")))?;
    let salt =
        SaltString::encode_b64(&salt_bytes).map_err(|error| PasswordHashError::HashFailure(error.to_string()))?;
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|error| PasswordHashError::HashFailure(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_and_verify_round_trip() {
        let hash = hash_password("correct horse battery staple").expect("hash");
        assert!(hash.starts_with("$argon2"));
        assert!(verify_password("correct horse battery staple", &hash).expect("verify"));
        assert!(!verify_password("wrong password", &hash).expect("verify"));
    }

    #[test]
    fn malformed_hash_reports_invalid() {
        let err = verify_password("anything", "not a phc string").unwrap_err();
        assert!(matches!(err, PasswordHashError::InvalidHash(_)), "got: {err:?}");
    }

    #[test]
    fn config_validation_requires_a_complete_argon2_phc() {
        let valid = hash_password("correct horse battery staple").expect("hash");
        validate_password_hash(&valid).expect("generated hash is a complete Argon2 PHC");
        let error = validate_password_hash("$argon2id$v=19$m=19456,t=2,p=1$only-a-salt")
            .expect_err("PHC without encoded output must reject");
        assert!(matches!(error, PasswordHashError::InvalidHash(_)), "got: {error:?}");
    }

    #[test]
    fn two_hashes_of_same_password_differ() {
        // Salt is random; consecutive calls must produce distinct hashes
        // that still both verify against the original password.
        let a = hash_password("hunter2").expect("hash a");
        let b = hash_password("hunter2").expect("hash b");
        assert_ne!(a, b);
        assert!(verify_password("hunter2", &a).expect("verify a"));
        assert!(verify_password("hunter2", &b).expect("verify b"));
    }
}
