//! Authentication for the Flight SQL server.
//!
//! # Modes
//!
//! - [`AuthMode::Disabled`]: accepts every request; local/trusted use only.
//! - [`AuthMode::BasicBearer`]: verifies Basic credentials from
//!   `do_handshake`, returns a bearer token, and requires that token on later
//!   RPCs.
//!
//! # Additional authentication modes
//!
//! Add these through `AuthMode`, `do_handshake`, and [`AuthInterceptor`] when a
//! deployment needs them.
//!
//! - JWT/JWKS bearer validation.
//! - OAuth2/OIDC token introspection.
//! - mTLS client-certificate principals.
//! - Kerberos/SPNEGO `Negotiate`.

mod config;
mod handshake;
mod interceptor;
mod password;
mod token;

pub use config::{AuthConfig, AuthMode, StaticUserConfig};
pub use handshake::handle_handshake;
pub use interceptor::AuthInterceptor;
pub use password::{PasswordHashError, hash_password, verify_password};
pub use token::{Principal, TokenStore};
