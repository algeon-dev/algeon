use rustls::crypto::ring;
use tonic::transport::{Certificate, Identity};

use super::{ConfigSources, Path, Result, ServerTlsConfig, file::ServerTlsFileConfig, fs};

/// Installs ring as rustls's process-wide crypto provider unless the process
/// already installed one. tonic builds its TLS acceptor from that provider,
/// and rustls selects none by itself when the build enables both its `ring`
/// and `aws-lc-rs` backends.
pub(in crate::server) fn install_rustls_crypto_provider() {
    // An error means another provider is installed, and tonic uses that one.
    let _ = ring::default_provider().install_default();
}

/// Reads the `[tls]` PEM files into the listener's tonic config. Only file
/// access is checked here. [`FlightSqlServerBuilder::build`] parses the PEM and
/// matches key to certificate, so a config assembled in Rust takes the same
/// validation path.
///
/// [`FlightSqlServerBuilder::build`]: crate::server::FlightSqlServerBuilder::build
pub(super) fn tls_config(
    file: Option<&ServerTlsFileConfig>,
    sources: &ConfigSources,
) -> Result<Option<ServerTlsConfig>> {
    let Some(file) = file else {
        return Ok(None);
    };
    let identity = Identity::from_pem(
        read_pem(sources, "tls.cert_path", &file.cert_path)?,
        read_pem(sources, "tls.key_path", &file.key_path)?,
    );
    install_rustls_crypto_provider();
    let mut config = ServerTlsConfig::new().identity(identity);
    if let Some(path) = &file.client_ca_path {
        config = config.client_ca_root(Certificate::from_pem(read_pem(sources, "tls.client_ca_path", path)?));
    }
    // tonic's config cannot report its client CA, so the listener's startup
    // log can only say whether TLS is on.
    tracing::info!(
        client_certificate_verification = file.client_ca_path.is_some(),
        "loaded the [tls] listener configuration"
    );
    Ok(Some(config))
}

/// The source error is the OS error alone, so no file content reaches a message.
fn read_pem(sources: &ConfigSources, field: &'static str, path: &Path) -> Result<Vec<u8>> {
    fs::read(path)
        .map_err(|error| sources.invalid_with_source(field, format!("`{}` cannot be read", path.display()), error))
}
