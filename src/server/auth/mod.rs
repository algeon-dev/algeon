//! Authentication for the Flight SQL server.
//!
//! # Modes
//!
//! - [`AuthMode::Disabled`] — no authentication. The server accepts every
//!   request. Suitable only for trusted single-tenant environments (local
//!   dev, internal benchmarks). Not safe for production exposure.
//! - [`AuthMode::BasicBearer`] — clients call `do_handshake` with an HTTP
//!   `authorization: Basic <base64(user:pass)>` header. The server verifies
//!   the credential against a static user list (loaded from the TOML
//!   server config, passwords stored as Argon2 hashes), mints a random
//!   bearer token, and returns it on the handshake response. Every
//!   subsequent RPC must carry `authorization: Bearer <token>`. This is
//!   the [official Flight SQL auth pattern documented in arrow-rs's
//!   `flight_sql_server.rs` example](https://github.com/apache/arrow-rs/blob/main/arrow-flight/examples/flight_sql_server.rs).
//!
//! # TODO: future modes
//!
//! These are intentionally not implemented yet; add them when a concrete
//! deployment needs them. The existing `AuthMode` enum is the extension
//! point — each new mode plugs into the same `do_handshake` /
//! [`AuthInterceptor`] surface.
//!
//! - **JWT (RS256/HS256, JWKS)** — verify caller-supplied JWTs against a
//!   configured signing key or remote JWKS. Crates: `jsonwebtoken`,
//!   `reqwest` (for JWKS fetch + cache). Skips the handshake entirely;
//!   clients present `authorization: Bearer <jwt>` on every RPC and the
//!   interceptor validates signature, `exp`, `aud`, `iss`.
//! - **OAuth2 / OIDC introspection** — call the IdP's RFC 7662
//!   introspection endpoint per token (with a short-TTL cache).
//!   Crate: `oauth2`. Useful when the IdP must remain authoritative for
//!   revocation.
//! - **mTLS (client certificate)** — terminate TLS in tonic with
//!   `Server::tls_config(..)` + `with_client_auth_required(..)`, then
//!   extract the verified peer cert's CN/SAN from the request extensions
//!   as the principal. No handshake; client identity is established at
//!   the transport layer. Best when both client and server live inside a
//!   PKI you control.
//! - **Kerberos / SPNEGO** — for AD-integrated enterprise deployments.
//!   Crate: `cross-krb5` or `libgssapi`. Adds a `Negotiate` handshake
//!   variant.

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
