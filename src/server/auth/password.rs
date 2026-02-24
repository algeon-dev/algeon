//! Argon2id password hashing and verification.

use std::fmt;

use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng},
};

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
    let salt = SaltString::generate(&mut OsRng);
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
