use std::{error::Error as StdError, path::PathBuf};

use algeon_datafusion::server::FlightSqlServer;
use rcgen::generate_simple_self_signed;
use tonic::transport::Server;

use super::*;

/// PEM files for one self-signed identity, kept alive with their directory.
struct PemFiles {
    _dir: tempfile::TempDir,
    cert: PathBuf,
    key: PathBuf,
}

fn pem_files(cert_pem: &str, key_pem: &str) -> PemFiles {
    let dir = tempfile::tempdir().expect("tempdir");
    let cert = dir.path().join("cert.pem");
    let key = dir.path().join("key.pem");
    std::fs::write(&cert, cert_pem).expect("write certificate");
    std::fs::write(&key, key_pem).expect("write key");
    PemFiles { _dir: dir, cert, key }
}

fn self_signed_files() -> PemFiles {
    let identity = generate_simple_self_signed(vec!["localhost".to_string()]).expect("self-signed identity");
    pem_files(&identity.cert.pem(), &identity.signing_key.serialize_pem())
}

fn load_config(toml_body: &str) -> ServerResult<ServerConfig> {
    let (_dir, env) = env_with_server_toml(toml_body);
    ServerConfig::from_vars(env.clone())
}

#[test]
fn tls_toml_table_loads_a_listener_config_tonic_accepts() {
    let files = self_signed_files();
    let identity_only = format!("[tls]\ncert_path = {:?}\nkey_path = {:?}\n", files.cert, files.key);
    let with_client_ca = format!("{identity_only}client_ca_path = {:?}\n", files.cert);

    for body in [identity_only, with_client_ca] {
        let tls = load_config(&body).expect("load TLS TOML").tls.expect("configured listener TLS");
        Server::builder().tls_config(tls).expect("tonic accepts the PEM the loader read");
    }

    let env = BTreeMap::<&str, String>::new();
    assert!(ServerConfig::from_vars(env.clone()).unwrap().tls.is_none());
}

#[test]
fn tls_toml_table_names_the_unreadable_file() {
    let files = self_signed_files();
    let missing = files.cert.with_file_name("absent.pem");
    for (field, body) in [
        ("tls.cert_path", format!("[tls]\ncert_path = {missing:?}\nkey_path = {:?}\n", files.key)),
        ("tls.key_path", format!("[tls]\ncert_path = {:?}\nkey_path = {missing:?}\n", files.cert)),
        (
            "tls.client_ca_path",
            format!("[tls]\ncert_path = {:?}\nkey_path = {:?}\nclient_ca_path = {missing:?}\n", files.cert, files.key),
        ),
    ] {
        let error = load_config(&body).expect_err("an unreadable PEM file must reject");
        assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
        assert_eq!(error.fact_value("config_field"), Some(field), "{error}");
        assert!(error.to_string().contains("absent.pem"), "{error}");
    }
}

#[test]
fn tls_toml_table_requires_both_files_and_rejects_unknown_keys() {
    let files = self_signed_files();
    for (needle, body) in [
        ("key_path", format!("[tls]\ncert_path = {:?}\n", files.cert)),
        ("cert_path", format!("[tls]\nkey_path = {:?}\n", files.key)),
        (
            "use_key_log",
            format!("[tls]\ncert_path = {:?}\nkey_path = {:?}\nuse_key_log = true\n", files.cert, files.key),
        ),
    ] {
        let error = load_config(&body).expect_err("an incomplete or unknown [tls] table must reject");
        assert!(error.to_string().contains(needle), "{error}");
    }
}

#[tokio::test]
async fn server_build_rejects_unusable_tls_material_without_echoing_it() {
    // The ownership declaration is absent on purpose. A build that reached the
    // backend first would fail on it instead, so a `tls` error also proves the
    // PEM is checked before any GPU initialization.
    const SECRET: &str = "SECRET-KEY-MATERIAL";
    let valid = generate_simple_self_signed(vec!["localhost".to_string()]).expect("self-signed identity");
    let other = generate_simple_self_signed(vec!["localhost".to_string()]).expect("second identity");
    let cert_pem = valid.cert.pem();
    for (case, files) in [
        ("unparsable key", pem_files(&cert_pem, SECRET)),
        ("unparsable certificate", pem_files(SECRET, &valid.signing_key.serialize_pem())),
        ("key of another certificate", pem_files(&cert_pem, &other.signing_key.serialize_pem())),
    ] {
        let body = format!("[tls]\ncert_path = {:?}\nkey_path = {:?}\n", files.cert, files.key);
        let config = load_config(&body).expect("the loader reads files without parsing PEM");
        let Err(error) = FlightSqlServer::builder().config(config).build().await else {
            panic!("{case}: unusable TLS material must fail the build");
        };

        assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{case}: {error}");
        assert_eq!(error.fact_value("config_field"), Some("tls"), "{case}: {error}");
        let mut rendered = format!("{error} {error:?}");
        let mut source = error.source();
        while let Some(cause) = source {
            rendered.push_str(&format!(" {cause} {cause:?}"));
            source = cause.source();
        }
        assert!(!rendered.contains(SECRET), "{case}: key material leaked into {rendered}");
    }
}
